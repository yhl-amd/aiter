# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL mxfp8_32 GEMMs on DeepSeek-V4.1's FP8 operands.

V4.1's FP8 linears and its grouped ``wo_a`` carry FP8 e4m3 weights with 32x32
e8m0 block scales and FP8 activations with 1x32 e8m0 scales. On gfx1250 they
run on FlyDSL:

* ``batched_gemm_a8w8_mxscale_bpreshuffle`` (wo_a, ``[M, G, K] x [G, N, K]``)
  with the raw 32x32 w_scale, or the 1x32 one in the n32k4 layout;
* ``gemm_a8w8_blockscale(..., isBpreshuffled=True)`` (the linears) with the
  raw 32x32 w_scale, and ``gemm_a8w8_mxfp8_bpreshuffle`` with the 32x32 scale
  expanded to 1x32 and n32k4-shuffled -- both must give the same answer.

Every case is checked against an FP32 product of the dequantized operands.
Scales vary block to block, so a scale applied to the wrong block (or the right
block of the wrong group) fails the check.

Run: ``pytest op_tests/test_flydsl_mxfp8_32_gemm_gfx1250.py`` (or as a script).
"""

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_scale_n32k4, shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="gfx1250 FlyDSL mxfp8_32 GEMMs",
)

SEED = 0
# DeepSeek-V4.1-Flash wo_a: N = o_lora_rank, K = heads * head_dim / o_groups.
WO_A_N, WO_A_K = 1024, 4096
WO_A_M = (1, 7, 64, 512, 4096)
WO_A_G = (1, 2, 8)  # o_groups / TP at TP8, TP4, TP1
# Tails: M off every tile, a short K, an N that is a multiple of 128 only.
TAIL_BMM = [
    (17, 2, 1024, 4096),
    (129, 2, 1792, 1280),
    (1000, 3, 256, 640),
    (4097, 1, 512, 1024),
]
# V4.1 FP8 linears, per rank at TP4: (N, K).
LINEARS = {
    "wqkv_a": (1792, 5120),  # q_lora_rank + head_dim, replicated
    "wq_b": (8192, 1280),  # heads * head_dim / 4
    "indexer.wq_b": (4096, 1280),  # index_n_heads * index_head_dim
    "wo_b": (5120, 2048),  # o_groups * o_lora_rank / 4
    "shared.w13": (1152, 5120),  # 2 * moe_intermediate_size / 4
    "shared.w2": (5120, 576),  # moe_intermediate_size / 4: K not a 128 multiple
}
LINEAR_M = (1, 7, 64, 512, 4096)
# K a multiple of 32 but not of 128: the last k-tile is ragged (32x32 scale only).
RAGGED_K_BMM = [
    (1, 1, 5120, 576),
    (33, 2, 1024, 544),
    (300, 1, 512, 4160),
    (4096, 1, 5120, 576),
]


def _fp8_and_e8m0(shape, scale_shape, gen):
    """FP8 values and e8m0 scales whose exponent varies over [-4, 4]."""
    q = (torch.randn(shape, generator=gen, device="cuda") * 2).to(dtypes.fp8)
    e = torch.randint(123, 132, scale_shape, generator=gen, device="cuda")
    return q, e.to(torch.uint8)


def _dequant(q, e8m0, rows, cols):
    """q [..., R, C] * 2^(e - 127) over rows x cols blocks, in FP32."""
    s = torch.exp2(e8m0.float() - 127.0)
    s = s.repeat_interleave(rows, -2).repeat_interleave(cols, -1)
    return q.float() * s[..., : q.shape[-2], : q.shape[-1]]


def _check(out, ref):
    out = out.float()
    err = (out - ref).abs()
    # BF16 output rounding (2^-8 relative) plus FP32 accumulation-order noise.
    bound = ref.abs() * 2**-7 + ref.abs().amax() * 1e-4
    bad = (err > bound).sum().item()
    assert bad == 0, (
        f"{bad}/{err.numel()} elements off; max err {err.max().item():.4g} "
        f"(max |ref| {ref.abs().max().item():.4g})"
    )


def _wo_a_operands(m, g, n, k, seed=SEED):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x8, xs = _fp8_and_e8m0((m, g, k), (m, g, k // 32), gen)
    w8, ws = _fp8_and_e8m0((g, n, k), (g, n // 32, k // 32), gen)
    ref = torch.einsum(
        "mgk,gnk->mgn", _dequant(x8, xs, 1, 32), _dequant(w8, ws, 32, 32)
    )
    return x8, xs, w8, ws, ref


def expand_32x32_to_n32k4(ws: torch.Tensor) -> torch.Tensor:
    """[G, N/32, K/32] 32x32 e8m0 -> [G, N, K/32] 1x32, n32k4-shuffled."""
    g, nb, kb = ws.shape
    rows = ws.view(torch.uint8).repeat_interleave(32, dim=1)
    return shuffle_scale_n32k4(rows, experts_cnt=g).view(g, nb * 32, kb)


def _bmm(x8, xs, w8_shuf, ws):
    from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle

    e8m0 = dtypes.fp8_e8m0
    return batched_gemm_a8w8_mxscale_bpreshuffle(
        x8, w8_shuf, xs.view(e8m0), ws.view(e8m0), dtype=dtypes.bf16
    )


@pytest.mark.parametrize("w_scale_block", ["32x32", "1x32"])
@pytest.mark.parametrize("g", WO_A_G)
@pytest.mark.parametrize("m", WO_A_M)
def test_bmm_wo_a(m, g, w_scale_block):
    x8, xs, w8, ws, ref = _wo_a_operands(m, g, WO_A_N, WO_A_K)
    if w_scale_block == "1x32":
        ws = expand_32x32_to_n32k4(ws)
    _check(_bmm(x8, xs, shuffle_weight(w8), ws), ref)


@pytest.mark.parametrize("w_scale_block", ["32x32", "1x32"])
@pytest.mark.parametrize("m,g,n,k", TAIL_BMM)
def test_bmm_tails(m, g, n, k, w_scale_block):
    x8, xs, w8, ws, ref = _wo_a_operands(m, g, n, k, seed=m + n)
    if w_scale_block == "1x32":
        ws = expand_32x32_to_n32k4(ws)
    _check(_bmm(x8, xs, shuffle_weight(w8), ws), ref)


@pytest.mark.parametrize("m,g,n,k", RAGGED_K_BMM)
def test_bmm_ragged_k(m, g, n, k):
    x8, xs, w8, ws, ref = _wo_a_operands(m, g, n, k, seed=m + k)
    _check(_bmm(x8, xs, shuffle_weight(w8), ws), ref)


def test_bmm_ragged_k_refuses_n32k4_scale():
    x8, xs, w8, ws, _ = _wo_a_operands(7, 1, 512, 576)
    rows = ws.view(torch.uint8).repeat_interleave(32, dim=1)
    with pytest.raises(RuntimeError, match="ragged"):
        _bmm(x8, xs, shuffle_weight(w8), rows)


def test_bmm_every_config_wo_a():
    """Every heuristic-eligible tile config, both scale layouts, one shape."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
        _LDS_BYTES,
        bmm_candidates,
        bmm_mx32_kernel_name,
        mx32_lds_bytes,
        run_bmm_a8w8_mxfp8_32_gfx1250,
    )

    m, g, n, k = 200, 2, WO_A_N, WO_A_K
    x8, xs, w8, ws, ref = _wo_a_operands(m, g, n, k)
    w8s, ws_n32k4 = shuffle_weight(w8), expand_32x32_to_n32k4(ws)
    names = set()
    for c in bmm_candidates(g, m, n, k):
        if mx32_lds_bytes(c) > _LDS_BYTES:
            continue
        names.add(
            bmm_mx32_kernel_name(
                c["tile_m"],
                c["tile_n"],
                c["tile_k"],
                c["m_warp"],
                c["n_warp"],
                c["num_buffers"],
            )
        )
    assert names
    for name in sorted(names):
        for raw in (True, False):
            out = torch.empty(m, g, n, dtype=dtypes.bf16, device="cuda")
            run_bmm_a8w8_mxfp8_32_gfx1250(
                x8,
                w8s,
                xs,
                ws if raw else ws_n32k4,
                out,
                kernel_name=name,
                w_scale_32x32=raw,
            )
            _check(out, ref)


def test_bmm_rejects_mismatched_scale():
    """A 1x32 kernel handed a 32x32 scale (and vice versa) must refuse."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
        run_bmm_a8w8_mxfp8_32_gfx1250,
    )

    x8, xs, w8, ws, _ = _wo_a_operands(16, 2, 256, 512)
    out = torch.empty(16, 2, 256, dtype=dtypes.bf16, device="cuda")
    with pytest.raises(RuntimeError, match="w_scale must have shape"):
        run_bmm_a8w8_mxfp8_32_gfx1250(x8, shuffle_weight(w8), xs, ws, out)
    with pytest.raises(RuntimeError, match="w_scale must have shape"):
        run_bmm_a8w8_mxfp8_32_gfx1250(
            x8,
            shuffle_weight(w8),
            xs,
            expand_32x32_to_n32k4(ws),
            out,
            w_scale_32x32=True,
        )


def test_bmm_cuda_graph():
    m, g = 7, 2
    x8, xs, w8, ws, ref = _wo_a_operands(m, g, WO_A_N, WO_A_K)
    w8s = shuffle_weight(w8)
    _bmm(x8, xs, w8s, ws)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _bmm(x8, xs, w8s, ws)
    graph.replay()
    torch.cuda.synchronize()
    _check(out, ref)


def _linear_operands(m, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x8, xs = _fp8_and_e8m0((m, k), (m, k // 32), gen)
    w8, ws = _fp8_and_e8m0((n, k), (n // 32, k // 32), gen)
    ref = _dequant(x8, xs, 1, 32) @ _dequant(w8, ws, 32, 32).T
    return x8, xs, w8, ws, ref


@pytest.mark.parametrize("w_scale_block", ["32x32", "1x32"])
@pytest.mark.parametrize("m", LINEAR_M)
@pytest.mark.parametrize("linear", list(LINEARS))
def test_linear(linear, m, w_scale_block):
    """gemm_a8w8_blockscale's preshuffled group32 route on gfx1250."""
    from aiter import gemm_a8w8_blockscale

    n, k = LINEARS[linear]
    if w_scale_block == "1x32" and k % 128:
        pytest.skip("a ragged K runs with the 32x32 scale only")
    x8, xs, w8, ws, ref = _linear_operands(m, n, k, seed=m + n + k)
    if w_scale_block == "1x32":
        ws = expand_32x32_to_n32k4(ws.view(1, n // 32, k // 32)).view(n, k // 32)
    e8m0 = dtypes.fp8_e8m0
    out = gemm_a8w8_blockscale(
        x8,
        shuffle_weight(w8),
        xs.view(e8m0),
        ws.view(e8m0),
        dtype=dtypes.bf16,
        isBpreshuffled=True,
    )
    _check(out, ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
