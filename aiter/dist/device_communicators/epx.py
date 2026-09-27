# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""epx: low-latency intranode expert-parallel dispatch/combine over XGMI.

Dispatch is two one-way hops (per-destination counts, then rows written at offsets every
source derives from those counts), combine is one (expert rows pushed back to their source
and reduced there). There are no remote atomics or remote reads, flags carry an epoch kept on
the device, so a captured CUDA graph replays without any host-side reset.

One ``EpxOp`` serves every MoE layer of a process; calls must alternate dispatch/combine.
"""

import torch
import torch.distributed as dist

from aiter.ops import epx as ops


class _PtrView:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (nbytes,),
            "strides": None,
            "typestr": "|u1",
            "version": 3,
        }


def _align(n: int, a: int = 256) -> int:
    return (n + a - 1) // a * a


class EpxOp:
    MAX_WS = 8

    def __init__(
        self,
        rank: int,
        world_size: int,
        hidden: int,
        max_tokens_per_rank: int,
        experts_per_rank: int,
        topk: int,
        x_bytes_per_elem: int = 1,
        scale_bytes: int = 0,
        out_dtype: torch.dtype = torch.bfloat16,
        group: dist.ProcessGroup | None = None,
        threads: int = 256,
    ):
        assert world_size <= self.MAX_WS, "epx is intranode: world_size <= 8"
        assert out_dtype == torch.bfloat16, "epx combine is bf16 only"
        self.rank, self.ws, self.hidden = rank, world_size, hidden
        self.maxt, self.epr, self.topk = max_tokens_per_rank, experts_per_rank, topk
        self.scale_bytes, self.threads, self.out_dtype = scale_bytes, threads, out_dtype
        row_bytes = hidden * x_bytes_per_elem
        out_row_bytes = hidden * out_dtype.itemsize
        R = world_size * max_tokens_per_rank
        off, lay = 0, []
        for nbytes in (
            64 * self.MAX_WS,  # 0: count mailbox, one slot per source rank
            8 * self.MAX_WS,  # 1: dispatch data flag, one per source rank
            8 * self.MAX_WS,  # 2: combine flag, one per destination rank
            R * row_bytes,  # 3: recv_x
            R * max(scale_bytes, 16),  # 4: recv_scales
            R * topk * 4,  # 5: recv_ids
            R * topk * 4,  # 6: recv_weights
            R * 4,  # 7: recv source (rank << 24 | token)
            R * out_row_bytes,  # 8: combine landing buffer [src][token]
            max_tokens_per_rank * self.MAX_WS * 4,  # 9: slot of each token per dest (local)
            64 + 256,  # 10: epoch | block counters | recv_count | trace (local)
        ):
            lay.append(off)
            off = _align(off + nbytes)
        self.lay, self.nbytes = lay, off

        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.ptr = ops.epx_alloc_uncached(off)
        handle = torch.empty(ops.epx_ipc_handle_size(), dtype=torch.uint8)
        ops.epx_get_ipc_handle(self.ptr, handle.data_ptr())
        handles = [None] * world_size
        dist.all_gather_object(handles, handle.tolist(), group=group)
        self.bases, self._opened = [], []
        for r, h in enumerate(handles):
            if r == rank:
                self.bases.append(self.ptr)
                continue
            ht = torch.tensor(h, dtype=torch.uint8)
            p = ops.epx_open_ipc_handle(ht.data_ptr())
            self.bases.append(p)
            self._opened.append(p)
        self._fa = ops.epx_init(
            self.bases, lay, rank, topk, experts_per_rank, max_tokens_per_rank, scale_bytes
        )
        self._buf = torch.as_tensor(_PtrView(self.ptr, off), device=self.dev)
        self.recv_count = self._region(10, 24, (1,), torch.int32)
        self.trace = self._region(10, 64, (32,), torch.int64)
        self.max_blocks = torch.cuda.get_device_properties(self.dev).multi_processor_count
        self._views = {}
        dist.barrier(group=group)

    def _region(self, idx, extra, shape, dtype):
        start = self.lay[idx] + extra
        n = dtype.itemsize
        for s in shape:
            n *= s
        return self._buf[start : start + n].view(dtype).view(shape)

    def _view(self, idx, shape, dtype):
        key = (idx, shape, dtype)
        v = self._views.get(key)
        if v is None:
            v = self._views[key] = self._region(idx, 0, shape, dtype)
        return v

    def _blocks(self, work_items: int) -> int:
        waves_per_block = self.threads // 64
        return max(1, min(self.max_blocks, -(-work_items // waves_per_block)))

    def dispatch(self, x, scales, topk_ids, topk_weights):
        """x: [T, H] fp8/bf16, scales: [T, scale_bytes] or None, topk_ids int32 [T, K]
        (ids < 0 are dropped), topk_weights float32 [T, K].

        Returns (recv_x, recv_weights, recv_scales, recv_ids, recv_count): views over the
        [world_size * max_tokens, ...] receive buffers whose rows [0, recv_count) are valid,
        grouped by source rank in token order."""
        ops.epx_dispatch(
            self._fa,
            x,
            scales if scales is not None else x,
            topk_ids,
            topk_weights,
            self._blocks(x.shape[0]),
            self.threads,
        )
        R = self.ws * self.maxt
        rx = self._view(3, (R, x.shape[1]), x.dtype)
        rs = (
            self._view(4, (R, scales.shape[1]), scales.dtype)
            if scales is not None
            else None
        )
        rids = self._view(5, (R, self.topk), torch.int32)
        rw = self._view(6, (R, self.topk), torch.float32)
        return rx, rw, rs, rids, self.recv_count

    def combine(self, expert_out, num_tokens, max_recv_rows=None):
        """expert_out: [>= recv_count, H] bf16 rows in receive order. Returns [T, H] bf16,
        the sum over every rank that received the token."""
        out = torch.empty(num_tokens, self.hidden, dtype=self.out_dtype, device=self.dev)
        rows = max_recv_rows if max_recv_rows is not None else num_tokens * self.ws
        ops.epx_combine(
            self._fa,
            expert_out,
            out,
            self._blocks(max(num_tokens, rows) * 2),
            self.threads,
        )
        return out

    def close(self):
        if self._fa:
            ops.epx_destroy(self._fa)
            self._fa = 0
        self._views.clear()
        self._buf = self.recv_count = self.trace = None
        for p in self._opened:
            ops.epx_close_ipc_handle(p)
        self._opened = []
        if self.ptr:
            ops.epx_free(self.ptr)
            self.ptr = 0
