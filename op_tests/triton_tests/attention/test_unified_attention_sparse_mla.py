# test code is adapted from flashMLA:
# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_decoding.py
import dataclasses
import random
from math import ceil

import pytest
import torch

from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)


def cdiv(a, b):
    return ceil(a / b)


@dataclasses.dataclass
class Param:
    b: int  # Batch size
    s_q: int  # Number of queries for one request
    s_k: int  # Seq len, or mean seq len if varlen == True
    is_varlen: bool
    is_causal: bool
    is_fp8: bool
    topk: int | None = None
    test_performance: bool = True
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    block_size: int = 64
    h_q: int = 128  # Number of q heads
    h_kv: int = 1  # Number of kv heads
    d: int = 576  # Q/K head dim (= dv + RoPE dim)
    dv: int = 512  # V head dim
    seed: int = 0


def generate_test_data(
    t: Param,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    Generate test data from a given configuration
    Return: [cache_seqlens, q, block_table, blocked_k]
    Pay attention: This function changes the random seed
    """
    random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    torch.backends.cudnn.deterministic = True

    assert t.h_q % t.h_kv == 0

    cache_seqlens_cpu = torch.full((t.b,), t.s_k, dtype=torch.int32, device="cpu")
    if t.is_varlen:
        for i in range(t.b):
            cache_seqlens_cpu[i] = max(random.normalvariate(t.s_k, t.s_k / 2), t.s_q)

    if t.have_zero_seqlen_k:
        zeros_mask = torch.randn(t.b, dtype=torch.float32, device="cpu") > 0
        cache_seqlens_cpu[zeros_mask] = 0

    max_seqlen = cache_seqlens_cpu.max().item()
    max_seqlen_pad = cdiv(max_seqlen, 256) * 256
    cache_seqlens = cache_seqlens_cpu.cuda()

    q = torch.randn(t.b, t.s_q, t.h_q, t.d)
    q.clamp_(min=-1.0, max=1.0)

    block_table = torch.arange(
        t.b * max_seqlen_pad // t.block_size, dtype=torch.int32
    ).view(t.b, max_seqlen_pad // t.block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(
        t.b, -1
    )
    blocked_k = torch.randn(block_table.numel(), t.block_size, t.h_kv, t.d) / 10
    blocked_k.clamp_(min=-1.0, max=1.0)

    if t.topk is None:
        for i in range(t.b):
            cur_len = cache_seqlens_cpu[i].item()
            cur_num_blocks = cdiv(cur_len, t.block_size)
            blocked_k[block_table[i][cur_num_blocks:]] = float("nan")
            if cur_len % t.block_size != 0:
                blocked_k[block_table[i][cur_num_blocks - 1]][
                    cur_len % t.block_size :
                ] = float("nan")
            block_table[i][cur_num_blocks:] = 2147480000
        return cache_seqlens, q, block_table, blocked_k, None, None
    else:
        block_table_cpu = block_table.cpu()
        abs_indices = torch.empty(t.b, t.s_q, t.topk, dtype=torch.int32, device="cpu")
        indices_in_kvcache = torch.empty(
            t.b, t.s_q, t.topk, dtype=torch.int32, device="cpu"
        )
        for i in range(t.b):
            # Generate indices
            for j in range(t.s_q):
                cur_abs_indices = torch.randperm(
                    int(cache_seqlens_cpu[i].item()), device="cpu"
                )[: t.topk]
                cur_blocked_indices = block_table_cpu[
                    i, cur_abs_indices // t.block_size
                ] * t.block_size + (cur_abs_indices % t.block_size)
                if len(cur_abs_indices) < t.topk:
                    pad_len = t.topk - len(cur_abs_indices)
                    cur_abs_indices = torch.cat(
                        [cur_abs_indices, torch.full((pad_len,), -1, device="cpu")]
                    )
                    cur_blocked_indices = torch.cat(
                        [cur_blocked_indices, torch.full((pad_len,), -1, device="cpu")]
                    )

                # Mask KV
                perm = torch.randperm(t.topk, device="cpu")
                cur_abs_indices = cur_abs_indices[perm]
                cur_blocked_indices = cur_blocked_indices[perm]

                # Fill it with invalid indices if needed
                if t.is_all_indices_invalid:
                    cur_abs_indices.fill_(-1)
                    cur_blocked_indices.fill_(-1)

                abs_indices[i, j, :] = cur_abs_indices
                indices_in_kvcache[i, j, :] = cur_blocked_indices

        # Mask nonused KV as NaN
        all_indices = indices_in_kvcache.flatten().tolist()
        all_indices = list(set(all_indices))
        if -1 in all_indices:
            all_indices.remove(-1)
        all_indices = torch.tensor(all_indices, dtype=torch.int32, device="cpu")

        blocked_k = blocked_k.view(-1, t.h_kv, t.d)
        nonused_indices_mask = torch.ones(
            blocked_k.size(0) * blocked_k.size(1), dtype=torch.bool, device="cpu"
        )
        nonused_indices_mask[all_indices] = False
        blocked_k[nonused_indices_mask, :, :] = float("nan")
        blocked_k = blocked_k.view(-1, t.block_size, t.h_kv, t.d)

        abs_indices = abs_indices.to(q.device)
        indices_in_kvcache = indices_in_kvcache.to(q.device)

        return cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache


def reference_torch(
    cache_seqlens: torch.Tensor,  # [batch_size]
    block_table: torch.Tensor,  # [batch_size, ?]
    q: torch.Tensor,  # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,  # [?, block_size, h_kv, d]
    dv: int,
    scale: float,
    is_causal: bool,
    indices: torch.Tensor | None = None,  # [batch_size, s_q, topk]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch
    """

    def get_topk_attn_mask(s_q: int, s_k: int, indices: torch.Tensor):
        mask = torch.zeros(s_q, s_k, dtype=torch.bool)
        for i in range(s_q):
            cur_indices = indices[i]
            valid_indices = cur_indices[cur_indices != -1]
            mask[i, valid_indices] = True
        return mask

    def scaled_dot_product_attention(
        batch_idx: int,
        query: torch.Tensor,  # [h_q, s_q, d]
        kv: torch.Tensor,  # [h_kv, s_k, d]
        dv: int,
        scale: float,
        is_causal,
        indices: torch.Tensor | None,  # [s_q, topk]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_q = query.size(0)
        h_kv = kv.size(0)
        s_q = query.shape[-2]
        s_k = kv.shape[-2]
        query = query.float() * scale
        kv = kv.float()
        if h_kv != 1:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)
        kv[kv != kv] = 0.0  # noqa: PLR0124
        attn_weight = query @ kv.transpose(-2, -1)  # [h_q, s_q, s_k]
        if (is_causal and query.size(1) > 1) or indices is not None:
            mask = torch.ones(s_q, s_k, dtype=torch.bool)
            if is_causal:
                assert indices is None
                mask = mask.tril(diagonal=s_k - s_q)
            if indices is not None:
                mask &= get_topk_attn_mask(s_q, s_k, indices)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(q.dtype)
        # attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)  # [h_q, s_q]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]  # [h_q, s_q, dv]
        # Correct for q tokens which has no attendable k
        lonely_q_mask = lse == float("-inf")
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")

        return output

    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    for i in range(b):
        cur_len = cache_seqlens_cpu[i].item()
        cur_num_blocks = cdiv(cur_len, block_size)
        cur_block_indices = block_table[i][0:cur_num_blocks]
        cur_kv = blocked_k[cur_block_indices].view(-1, h_kv, d)[:cur_len, ...]
        cur_out = scaled_dot_product_attention(
            i,
            q[i].transpose(0, 1),
            cur_kv.transpose(0, 1),
            dv,
            scale,
            is_causal,
            indices[i] if indices is not None else None,
        )
        out_ref[i] = cur_out.transpose(0, 1)
    out_ref = out_ref.to(torch.bfloat16)
    return out_ref


def chunk_input(
    cache_seqlens,
    q,
    block_table,
    blocked_k,
    abs_indices,
    indices_in_kvcache,
    dtype=torch.bfloat16,
):
    q_new = q.reshape(-1, q.shape[2], q.shape[3])
    abs_indices = abs_indices.reshape(-1, abs_indices.shape[2])
    indices_in_kvcache = indices_in_kvcache.reshape(-1, indices_in_kvcache.shape[2])
    max_q_len = q.shape[1]
    max_kv_len = max(cache_seqlens)
    query_lens = [q.shape[1]] * q.shape[0]  # B * [q_len,]
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    cache_seqlens = cache_seqlens.to("cuda")
    q_new = q_new.to("cuda")
    block_table = block_table.to("cuda")
    blocked_k = blocked_k.to("cuda")
    abs_indices = abs_indices.to("cuda")
    indices_in_kvcache = indices_in_kvcache.to("cuda")
    return (
        cu_query_lens,
        max_q_len,
        cache_seqlens,
        max_kv_len,
        q_new.to(dtype),
        block_table,
        blocked_k.to(dtype),
        abs_indices,
        indices_in_kvcache,
    )


@pytest.mark.parametrize("s_q", [1, 64, 177])
@pytest.mark.parametrize("s_k", [1, 64, 177])
@pytest.mark.parametrize("top_k", [64, 78])
@pytest.mark.parametrize("num_q_heads", [16, 32])
@pytest.mark.parametrize("lora_dim", [256, 512])
@pytest.mark.parametrize("block_size", [16, 64])
@torch.inference_mode()
def test_triton_unified_attn(
    s_q: int,
    s_k: int,
    top_k: int,
    num_q_heads: int,
    lora_dim: int,
    block_size: int,
) -> None:
    batch = 8
    rope_dim = 64
    total_dim = lora_dim + rope_dim
    softmax_scale = lora_dim**-0.5

    test_p = Param(
        batch,
        s_q,
        s_k,
        d=total_dim,
        dv=lora_dim,
        h_q=num_q_heads,
        block_size=block_size,
        is_varlen=True,
        is_causal=False,
        is_fp8=False,
        topk=top_k,
        test_performance=False,
    )
    cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache = (
        generate_test_data(test_p)
    )
    ref_output = reference_torch(
        cache_seqlens,
        block_table,
        q,
        blocked_k,
        lora_dim,
        softmax_scale,
        False,
        abs_indices,
    )

    (
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        q,
        block_table,
        blocked_k,
        abs_indices,
        indices_in_kvcache,
    ) = chunk_input(
        cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache
    )

    output = torch.empty((*q.shape[:-1], lora_dim), device=q.device, dtype=q.dtype)

    unified_attention_sparse_mla(
        q,
        blocked_k,
        output,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        indices_in_kvcache,
        block_table,
        lora_dim,
    )

    ref_output = ref_output.to(output.device).to(q.dtype)
    output = output.reshape(ref_output.shape)

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(output - ref_output))}"


@torch.inference_mode()
def test_triton_unified_attn_fp8_csr_qh16() -> None:
    """Exercise the GLM sparse-decode ABI: FP8 Q/KV and ragged CSR indices."""
    torch.manual_seed(20260808)
    fp8_dtype = torch.float8_e4m3fn
    batch = 3
    num_q_heads = 16
    lora_dim = 512
    rope_dim = 64
    total_dim = lora_dim + rope_dim
    pool_size = 96
    row_counts = (64, 0, 64)
    softmax_scale = 1.0 / 16.0

    q_source = torch.randn(
        batch, num_q_heads, total_dim, device="cuda", dtype=torch.float32
    ).div_(10.0)
    kv_source = torch.randn(
        pool_size, total_dim, device="cuda", dtype=torch.float32
    ).div_(10.0)
    # GLM-5.2's deployed FP8 sparse-MLA cache uses unit descales.
    q_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    kv_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    q = q_source.to(fp8_dtype)
    kv_flat = kv_source.to(fp8_dtype)
    kv = kv_flat.view(pool_size, 1, 1, total_dim)

    rows = [
        torch.randperm(pool_size, device="cuda", dtype=torch.int64)[:count].to(
            torch.int32
        )
        for count in row_counts
    ]
    topk_indices = torch.cat(rows)
    topk_indptr = torch.tensor(
        [0, row_counts[0], row_counts[0], sum(row_counts)],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor(row_counts, device="cuda", dtype=torch.int32)
    block_table = torch.zeros((batch, 1), device="cuda", dtype=torch.int32)
    output = torch.empty(
        batch, num_q_heads, lora_dim, device="cuda", dtype=torch.bfloat16
    )

    unified_attention_sparse_mla(
        q,
        kv,
        output,
        cu_seqlens_q,
        1,
        seqused_k,
        max(row_counts),
        softmax_scale,
        topk_indices,
        block_table,
        lora_dim,
        q_descale=q_descale,
        kv_descale=kv_descale,
        tile_size=64,
        topk_indptr=topk_indptr,
        topk_count=max(row_counts),
    )

    reference = torch.zeros_like(output)
    q_raw = q.float()
    kv_raw = kv_flat.float()
    qk_factor = float(q_descale.item() * kv_descale.item() * softmax_scale)
    value_scale = float(kv_descale.item())
    for row, indices in enumerate(rows):
        if indices.numel() == 0:
            continue
        selected = kv_raw.index_select(0, indices.long())
        scores = torch.einsum("hd,kd->hk", q_raw[row], selected) * qk_factor
        probabilities = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
        denominator = probabilities.sum(dim=-1, keepdim=True)
        numerator = probabilities.to(fp8_dtype).float() @ selected[:, :lora_dim]
        reference[row] = (numerator * value_scale / denominator).to(torch.bfloat16)

    torch.testing.assert_close(output, reference, atol=3e-2, rtol=2e-2)
    assert torch.count_nonzero(output[1]).item() == 0

    first = output.clone()
    for repeat in range(10):
        unified_attention_sparse_mla(
            q,
            kv,
            output,
            cu_seqlens_q,
            1,
            seqused_k,
            max(row_counts),
            softmax_scale,
            topk_indices,
            block_table,
            lora_dim,
            q_descale=q_descale,
            kv_descale=kv_descale,
            tile_size=64,
            topk_indptr=topk_indptr,
            topk_count=max(row_counts),
        )
        mismatch = output != first
        assert not mismatch.any(), (
            f"repeat {repeat} produced {mismatch.count_nonzero().item()} "
            "non-bitwise-identical elements by row "
            f"{mismatch.reshape(batch, -1).count_nonzero(dim=1).tolist()}"
        )
