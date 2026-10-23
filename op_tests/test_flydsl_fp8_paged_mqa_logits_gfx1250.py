# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL paged FP8 MQA logits vs an fp32 torch reference.

The cache is the packed preshuffled layout ``deepgemm_fp8_paged_mqa_logits``
reads with ``Preshuffle=True``: per page ``KVBlockSize * 128`` FP8 bytes
shuffled with ``shuffle_weight(kv, layout=(min(KVBlockSize, 16), 16))``, then
``KVBlockSize`` fp32 scales. DeepSeek-V4.1 uses 8-token pages.

``python op_tests/test_flydsl_fp8_paged_mqa_logits_gfx1250.py`` times the
kernel (and the Gluon kernel when it accepts the page size) at decode shapes.
"""

import argparse

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="gfx1250 kernel",
)

HEAD_DIM = 128
SENTINEL = 12345.0
GUARD = 16


def _kernel():
    from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits

    return flydsl_fp8_paged_mqa_logits


def make_inputs(
    context_lengths,
    next_n,
    heads,
    block_size,
    padded_table,
    seed=5614,
    num_blocks=None,
):
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(context_lengths)
    max_context = max(max(context_lengths), 1)
    max_pages = (max_context + block_size - 1) // block_size
    if num_blocks is None:
        num_blocks = 2 * max_pages + 3
    q = torch.randn(batch, next_n, heads, HEAD_DIM, generator=gen, device=device).to(
        dtypes.fp8
    )
    kv = torch.randn(num_blocks, block_size, HEAD_DIM, generator=gen, device=device).to(
        dtypes.fp8
    )
    scales = 0.25 + torch.rand(num_blocks, block_size, generator=gen, device=device)
    weights = torch.randn(batch * next_n, heads, generator=gen, device=device)

    packed = torch.empty(
        num_blocks, block_size * (HEAD_DIM + 4), dtype=torch.uint8, device=device
    )
    value_bytes = block_size * HEAD_DIM
    packed[:, :value_bytes] = (
        shuffle_weight(kv, layout=(min(block_size, 16), 16))
        .reshape(num_blocks, -1)
        .view(torch.uint8)
    )
    packed[:, value_bytes:] = scales.view(torch.uint8)
    cache = packed.view(num_blocks, block_size, 1, HEAD_DIM + 4)

    table_width = max(4096, max_pages) if padded_table else max_pages
    tables = torch.zeros(batch, table_width, dtype=torch.int32, device=device)
    for b, ctx in enumerate(context_lengths):
        pages = (ctx + block_size - 1) // block_size
        if pages:
            tables[b, :pages] = torch.randint(
                0, num_blocks, (pages,), generator=gen, device=device
            ).to(torch.int32)
    context_lens = torch.tensor(context_lengths, dtype=torch.int32, device=device)
    return q, kv, scales, weights, cache, tables, context_lens


def reference(q, kv, scales, weights, tables, context_lengths, width):
    batch, next_n = q.shape[:2]
    block_size = kv.shape[1]
    out = torch.full(
        (batch * next_n, width), float("-inf"), device=q.device, dtype=torch.float32
    )
    for b, ctx in enumerate(context_lengths):
        if ctx == 0:
            continue
        pages = (ctx + block_size - 1) // block_size
        page_ids = tables[b, :pages].long()
        keys = kv[page_ids].float().reshape(-1, HEAD_DIM)[:ctx]
        key_scales = scales[page_ids].reshape(-1)[:ctx]
        scores = ((q[b].float() @ keys.T) * key_scales).relu()
        row_w = weights[b * next_n : (b + 1) * next_n]
        logits = (scores * row_w[..., None]).sum(dim=1)
        pos = torch.arange(ctx, device=q.device)
        qpos = ctx - next_n + torch.arange(next_n, device=q.device)
        logits.masked_fill_(pos[None, :] > qpos[:, None], float("-inf"))
        out[b * next_n : (b + 1) * next_n, :ctx] = logits
    return out


def run_case(
    context_lengths,
    next_n,
    heads,
    block_size,
    padded_table=False,
    split_kv=None,
    ctx_2d=False,
    next_n_lens=None,
    wide_cache=False,
    sliced_table=False,
):
    q, kv, scales, weights, cache, tables, context_lens = make_inputs(
        context_lengths, next_n, heads, block_size, padded_table
    )
    if sliced_table:
        # Row-strided page table: the first columns of a wider one.
        wider = torch.full(
            (tables.shape[0], tables.shape[1] + 37),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        wider[:, : tables.shape[1]] = tables
        tables = wider[:, : tables.shape[1]]
    width = max(max(context_lengths), 1)
    storage = torch.full(
        (q.shape[0] * next_n, width + 2 * GUARD), SENTINEL, device="cuda"
    )
    out = storage[:, GUARD:-GUARD]
    out.fill_(float("-inf"))
    lens = context_lens
    if ctx_2d:
        lens = (
            context_lens[:, None]
            - next_n
            + 1
            + torch.arange(next_n, device="cuda", dtype=torch.int32)
        ).contiguous()
    _kernel()(
        q,
        cache,
        weights,
        out,
        lens,
        tables,
        width,
        Preshuffle=True,
        KVBlockSize=block_size,
        SplitKV=split_kv,
        next_n_lens=next_n_lens,
        _wide_cache=wide_cache,
    )
    torch.cuda.synchronize()
    ref = reference(q, kv, scales, weights, tables, context_lengths, width)
    if next_n_lens is not None:
        # Ragged rows: row j of sequence b is live below next_n_lens[b] and is
        # masked as if the sequence had that many query rows.
        lens_host = next_n_lens.tolist()
        for b, ctx in enumerate(context_lengths):
            live = min(lens_host[b], next_n)
            if live == next_n:
                continue
            q_b = q[b : b + 1, :live]
            w_b = weights[b * next_n : b * next_n + live]
            ref[b * next_n : (b + 1) * next_n] = float("-inf")
            if live:
                ref[b * next_n : b * next_n + live] = reference(
                    q_b, kv, scales, w_b, tables[b : b + 1], [ctx], width
                )
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    assert torch.all(storage[:, :GUARD] == SENTINEL)
    assert torch.all(storage[:, -GUARD:] == SENTINEL)


@pytest.mark.parametrize("block_size", [8, 16, 32, 64])
@pytest.mark.parametrize("padded_table", [False, True], ids=["compact", "padded"])
@pytest.mark.parametrize(
    "context_lengths,next_n,heads",
    [
        pytest.param((997, 63, 1, 8, 16, 2048), 1, 32, id="decode-h32"),
        pytest.param((4100, 17, 9, 0), 1, 64, id="decode-h64"),
        pytest.param((3000, 257, 3, 0), 3, 64, id="mtp3-h64"),
        pytest.param((1500, 33, 2, 64), 2, 32, id="mtp2-h32"),
        pytest.param((777, 128, 5), 4, 32, id="mtp4-h32"),
    ],
)
@torch.inference_mode()
def test_paged_mqa_logits(block_size, padded_table, context_lengths, next_n, heads):
    run_case(context_lengths, next_n, heads, block_size, padded_table)


# Split boundaries are tile (16-column) multiples whatever the page size, so
# uneven and over-sized splits must still write every column exactly once.
@pytest.mark.parametrize("block_size", [8, 64])
@pytest.mark.parametrize("split_kv", [1, 3, 7, 64, 1000])
@torch.inference_mode()
def test_paged_mqa_logits_split(block_size, split_kv):
    run_case((5000, 24, 129, 1), 2, 32, block_size, split_kv=split_kv)


@pytest.mark.parametrize("block_size", [8, 16])
@torch.inference_mode()
def test_paged_mqa_logits_ctx_2d(block_size):
    """vLLM-style ``context_lens`` ``[B, next_n]``: only the last column is read."""
    run_case((900, 40, 3), 3, 32, block_size, ctx_2d=True)


@pytest.mark.parametrize("block_size", [8, 64])
@torch.inference_mode()
def test_paged_mqa_logits_sliced_table(block_size):
    run_case((700, 41, 2), 2, 32, block_size, sliced_table=True)


@pytest.mark.parametrize("block_size", [8, 64])
@torch.inference_mode()
def test_paged_mqa_logits_next_n_lens(block_size):
    lens = torch.tensor([4, 1, 0, 3], dtype=torch.int32, device="cuda")
    run_case((600, 130, 31, 9), 4, 32, block_size, next_n_lens=lens)


# The 64-bit addressing path a cache past 2 GiB takes, forced on a small one.
@pytest.mark.parametrize("block_size", [8, 64])
@pytest.mark.parametrize("next_n", [1, 3])
@torch.inference_mode()
def test_paged_mqa_logits_wide_cache_path(block_size, next_n):
    run_case((1000, 77, 5), next_n, 32, block_size, wide_cache=True)


@torch.inference_mode()
def test_paged_mqa_logits_cache_past_2gib():
    """Pages beyond byte 2**31 (and 2**32) of the cache are read correctly."""
    block_size, heads, next_n, ctx = 8, 32, 1, 300
    page_bytes = block_size * (HEAD_DIM + 4)
    num_blocks = (4 << 30) // page_bytes + 64
    pages = (ctx + block_size - 1) // block_size
    q, kv, scales, weights, small, _, context_lens = make_inputs(
        (ctx, ctx), next_n, heads, block_size, False, num_blocks=2 * pages
    )
    cache = torch.zeros(
        num_blocks, block_size, 1, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    # Sequence 0 reads the last pages (past 4 GiB), sequence 1 straddles 2 GiB.
    ids = torch.cat(
        [
            torch.arange(num_blocks - pages, num_blocks),
            torch.arange(pages) + (2 << 30) // page_bytes - pages // 2,
        ]
    ).cuda()
    cache[ids] = small
    tables = ids.view(2, pages).to(torch.int32)
    out = torch.full((2 * next_n, ctx), float("-inf"), device="cuda")
    _kernel()(q, cache, weights, out, context_lens, tables, ctx, KVBlockSize=8)
    torch.cuda.synchronize()
    local = torch.arange(2 * pages, dtype=torch.int32, device="cuda").view(2, pages)
    ref = reference(q, kv, scales, weights, local, (ctx, ctx), ctx)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    del cache


@torch.inference_mode()
def test_paged_mqa_logits_long_context():
    """V4.1-like decode: 8-token pages, a long context, scattered pages."""
    run_case((65536, 8191, 31000), 1, 32, 8, padded_table=True)


@torch.inference_mode()
def test_paged_mqa_logits_wide_output():
    """Row base past 2**31 bytes: the row offset must be computed in 64 bits."""
    batch, next_n, heads, block_size, ctx = 520, 1, 32, 8, 40
    width = 1 << 20
    q, kv, scales, weights, cache, tables, context_lens = make_inputs(
        (ctx,) * batch, next_n, heads, block_size, False
    )
    out = torch.full((batch, width), SENTINEL, device="cuda")
    _kernel()(q, cache, weights, out, context_lens, tables, width, KVBlockSize=8)
    torch.cuda.synchronize()
    ref = reference(q, kv, scales, weights, tables, (ctx,) * batch, ctx)
    torch.testing.assert_close(out[:, :ctx], ref, rtol=1e-2, atol=1e-2)
    assert torch.all(out[:, ctx : ctx + 64] == SENTINEL)
    del out


def _bench(fn, iters=20, reps=5):
    """Device time per call: ``iters`` calls captured in one graph, replayed."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (iters * reps)


def bench_shape(batch, ctx, heads, block_size, next_n, gluon):
    """Time one decode shape; returns (flydsl_us, gluon_us, max |diff|)."""
    from functools import partial

    from aiter.ops.triton.attention.pa_mqa_logits import (
        deepgemm_fp8_paged_mqa_logits,
    )

    pages = (ctx + block_size - 1) // block_size
    q, _, _, weights, cache, _, context_lens = make_inputs(
        (ctx,) * batch,
        next_n,
        heads,
        block_size,
        False,
        num_blocks=pages * batch + 8,
    )
    # Scatter every sequence over distinct pages, as a real pool does.
    tables = (
        torch.randperm(pages * batch, device="cuda", dtype=torch.int64)
        .to(torch.int32)
        .view(batch, pages)
    )
    out = torch.full((batch * next_n, ctx), float("-inf"), device="cuda")
    out_g = out.clone()
    args = (q, cache, weights)
    run_fly = partial(
        _kernel(), *args, out, context_lens, tables, ctx, KVBlockSize=block_size
    )
    t_fly = _bench(run_fly)
    t_gl, diff = float("nan"), float("nan")
    if gluon:
        run_gluon = partial(
            deepgemm_fp8_paged_mqa_logits,
            *args,
            out_g,
            context_lens,
            tables,
            ctx,
            Preshuffle=True,
            KVBlockSize=block_size,
            ChunkK=256,
        )
        try:
            t_gl = _bench(run_gluon)
            finite = torch.isfinite(out_g) & torch.isfinite(out)
            diff = (out - out_g)[finite].abs().max().item()
        except Exception as exc:  # noqa: BLE001
            print(f"gluon failed: {type(exc).__name__}: {exc}"[:200])
    return t_fly, t_gl, diff


def bench(batches, contexts, heads, block_size, next_n, gluon):
    print(
        f"{'B':>5} {'ctx':>7} {'nn':>3} {'H':>3} {'kvb':>4} "
        f"{'flydsl_us':>10} {'GB/s':>8} {'gluon_us':>10} {'speedup':>8} {'maxdiff':>9}"
    )
    for batch in batches:
        for ctx in contexts:
            t_fly, t_gl, diff = bench_shape(
                batch, ctx, heads, block_size, next_n, gluon
            )
            nbytes = batch * ctx * (HEAD_DIM + 4) + batch * next_n * ctx * 4
            print(
                f"{batch:>5} {ctx:>7} {next_n:>3} {heads:>3} {block_size:>4} "
                f"{t_fly:>10.1f} {nbytes / t_fly / 1e3:>8.0f} {t_gl:>10.1f} "
                f"{t_gl / t_fly:>8.2f} {diff:>9.2e}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--batch", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("-c", "--ctx", type=int, nargs="+", default=[8192, 65536])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kvb", type=int, default=8)
    parser.add_argument("--next-n", type=int, default=1)
    parser.add_argument("--no-gluon", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    bench(args.batch, args.ctx, args.heads, args.kvb, args.next_n, not args.no_gluon)
