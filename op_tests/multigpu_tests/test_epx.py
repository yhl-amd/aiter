# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""epx dispatch/combine: bit-exact dispatch against an all_gather reference, combine error,
CUDA-graph replay with fresh inputs, and (--bench) per-layer dispatch+combine latency.

    python op_tests/multigpu_tests/test_epx.py [--world-size 8] [--bench]
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.dist.device_communicators.epx import EpxOp

# DeepSeek-V3/V4 geometry: MXFP8 rows (1 byte/elem + one e8m0 scale per 32 elements).
H, EPR, K, MAXT = 7168, 48, 6, 16384
SB = H // 32


class Case:
    def __init__(self, rank, ws, dev):
        self.rank, self.ws, self.dev = rank, ws, dev
        self.E = EPR * ws

    def inputs(self, T, seed):
        g = torch.Generator(device=self.dev).manual_seed(seed * 1000 + self.rank)
        x = (torch.randn(T, H, device=self.dev, generator=g) * 2).to(torch.float8_e4m3fn)
        s = torch.randint(0, 255, (T, SB), device=self.dev, dtype=torch.uint8, generator=g)
        if T:
            ids = torch.stack(
                [torch.randperm(self.E, device=self.dev, generator=g)[:K] for _ in range(T)]
            ).to(torch.int32)
        else:
            ids = torch.empty(0, K, dtype=torch.int32, device=self.dev)
        if T > 3:
            ids[T - 1] = -1  # a fully padded row: sent nowhere
            ids[T - 2, : K // 2] = -1  # partially masked
        w = torch.rand(T, K, device=self.dev, generator=g)
        return x, s, ids, w

    def _gather(self, t):
        out = [torch.empty_like(t) for _ in range(self.ws)]
        dist.all_gather(out, t)
        return out

    def reference(self, x, s, ids, w):
        allx, alls, alli, allw = (self._gather(t) for t in (x, s, ids, w))
        sel = [
            ((alli[r] >= 0) & (alli[r] // EPR == self.rank)).any(1).nonzero().flatten()
            for r in range(self.ws)
        ]
        rx = torch.cat([allx[r][sel[r]].view(torch.uint8) for r in range(self.ws)])
        rs = torch.cat([alls[r][sel[r]] for r in range(self.ws)])
        ri = torch.cat([alli[r][sel[r]] for r in range(self.ws)])
        rw = torch.cat([allw[r][sel[r]] for r in range(self.ws)])
        # The fake expert on rank d scales its rows by d + 1, so the combined row is
        # x * sum over the destination ranks of (d + 1).
        dests = torch.zeros(x.shape[0], self.ws, dtype=torch.bool, device=self.dev)
        for k in range(K):
            v = ids[:, k] >= 0
            dests[v, (ids[v, k] // EPR).long()] = True
        mult = (dests.float() * torch.arange(1, self.ws + 1, device=self.dev).float()).sum(
            1, keepdim=True
        )
        return rx, rs, ri, rw, x.float() * mult


def run_layer(op, rank, ws, x, s, ids, w):
    rx, rw, rs, ri, cnt = op.dispatch(x, s, ids, w)
    R = x.shape[0] * ws
    expert = (rx[:R].float() * (rank + 1)).to(torch.bfloat16)
    return op.combine(expert, x.shape[0], max_recv_rows=R), (rx, rw, rs, ri, cnt)


def check(case, T, seed, res):
    x, s, ids, w = case.inputs(T, seed)
    out, (rx, rw, rs, ri, cnt) = res
    ex, es, ei, ew, eo = case.reference(x, s, ids, w)
    n = int(cnt.item())
    ok = n == ex.shape[0]
    ok = ok and torch.equal(rx[:n].view(torch.uint8), ex) and torch.equal(rs[:n], es)
    ok = ok and torch.equal(ri[:n], ei) and torch.equal(rw[:n], ew)
    err = (
        ((out.float() - eo).abs().max() / eo.abs().max().clamp(min=1e-6)).item() if T else 0.0
    )
    return ok and err < 2e-2, n, err


def all_ok(ok, dev):
    t = torch.tensor([int(ok)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def capture(fn):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res = fn()
    return g, res


def worker(rank, ws, args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=ws)
    dev = torch.device("cuda", rank)
    case = Case(rank, ws, dev)
    op = EpxOp(rank, ws, H, MAXT, EPR, K, x_bytes_per_elem=1, scale_bytes=SB)
    failed = False
    tokens = [int(t) for t in args.tokens.split(",")]

    for T in tokens:
        res = run_layer(op, rank, ws, *case.inputs(T, 1))
        torch.cuda.synchronize()
        ok, n, err = check(case, T, 1, res)
        ok = all_ok(ok, dev)
        failed |= not ok
        if rank == 0:
            print(f"eager T={T:5d} recv(rank0)={n:6d} rel_err={err:.2e} "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)

    # Capture three layers once, replay with fresh inputs copied into the static buffers.
    T = 56
    static = [t.clone() for t in case.inputs(T, 7)]
    g, outs = capture(lambda: [run_layer(op, rank, ws, *static) for _ in range(3)])
    graph_ok = True
    for rnd in range(args.graph_rounds):
        for dst, src in zip(static, case.inputs(T, 100 + rnd)):
            dst.copy_(src)
        torch.cuda.synchronize()
        dist.barrier()
        g.replay()
        torch.cuda.synchronize()
        graph_ok &= check(case, T, 100 + rnd, outs[-1])[0]
    graph_ok = all_ok(graph_ok, dev)
    failed |= not graph_ok
    if rank == 0:
        print(f"graph replay x{args.graph_rounds} (3 layers each) "
              f"{'PASS' if graph_ok else 'FAIL'}", flush=True)

    if args.bench:
        for T in tokens:
            if T == 0:
                continue
            x, s, ids, w = case.inputs(T, 3)
            expert = torch.empty(T * ws, H, dtype=torch.bfloat16, device=dev)

            def layers():
                for _ in range(args.layers):
                    op.dispatch(x, s, ids, w)
                    op.combine(expert, T, max_recv_rows=T * ws)

            gg, _ = capture(layers)
            for _ in range(5):
                gg.replay()
            torch.cuda.synchronize()
            dist.barrier()
            evs = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(args.iters)
            ]
            for a, b in evs:
                a.record()
                gg.replay()
                b.record()
            torch.cuda.synchronize()
            ts = sorted(a.elapsed_time(b) for a, b in evs)
            med = torch.tensor([ts[len(ts) // 2] * 1000 / args.layers], device=dev)
            dist.all_reduce(med, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"tokens={T:5d} dispatch+combine {med.item():.1f} us/layer", flush=True)

    op.close()
    dist.barrier()
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--tokens", default="1,7,42,56,112,224,1024,4096")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--layers", type=int, default=20)
    p.add_argument("--graph-rounds", type=int, default=5)
    p.add_argument("--port", type=int, default=29517)
    args = p.parse_args()
    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
