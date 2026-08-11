# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression test for fused AllReduce+RMSNorm IPC publication ordering.

The production GLM-5.2 prefill shape is ``(3828, 6144)`` with BF16 and TP=8.
The two-stage kernel writes each reduce-scatter partition directly into every
rank's IPC tmp buffer, publishes an end flag, and then reads the local tmp
buffer in stage 2.

Inputs alternate between two recognizable patterns. Every producer partition
and input rank has a distinct value, so a stale or missing IPC write can be
attributed to its producer and reuse phase. This is a regression stress guard,
not a reproducer for the proposed all-writer fence: both current main and the
fence branch pass it in the controls run so far. A failure would still provide
useful attribution through the producer- and phase-specific sentinels.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

set_start_method("spawn", force=True)


def _bf16_sum(values: list[float]) -> float:
    inputs = torch.tensor(values, dtype=torch.bfloat16)
    return float(inputs.float().sum().to(torch.bfloat16).float().item())


def _pattern_value(phase: int, producer: int, input_rank: int) -> float:
    # Keep values and phase sums far apart so stale data cannot look like a
    # normal BF16 rounding difference.
    return float(phase * 256 + producer * 16 + input_rank)


def _make_pattern(
    shape: tuple[int, int], tp: int, rank: int, phase: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    numel = shape[0] * shape[1]
    if numel % tp:
        raise ValueError(f"{shape=} must contain a multiple of {tp=} elements")

    partition_size = numel // tp
    input_tensor = torch.empty(numel, dtype=torch.bfloat16, device=device)
    expected = torch.empty_like(input_tensor)
    for producer in range(tp):
        begin = producer * partition_size
        end = (producer + 1) * partition_size
        input_tensor[begin:end].fill_(_pattern_value(phase, producer, rank))
        expected_value = _bf16_sum(
            [_pattern_value(phase, producer, peer) for peer in range(tp)]
        )
        expected[begin:end].fill_(expected_value)
    return input_tensor.view(shape), expected.view(shape)


def _mismatch_details(
    actual: torch.Tensor,
    expected: torch.Tensor,
    shape: tuple[int, int],
    tp: int,
    sample_limit: int = 8,
) -> dict:
    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    bad_indices = torch.where(actual_flat != expected_flat)[0]
    sample_indices = bad_indices[:sample_limit].cpu().tolist()
    sample_actual = actual_flat[bad_indices[:sample_limit]].float().cpu().tolist()
    sample_expected = expected_flat[bad_indices[:sample_limit]].float().cpu().tolist()
    partition_size = actual_flat.numel() // tp
    sample = [
        {
            "flat": index,
            "row": index // shape[1],
            "col": index % shape[1],
            "producer_partition": index // partition_size,
            "actual": got,
            "expected": want,
        }
        for index, got, want in zip(
            sample_indices, sample_actual, sample_expected, strict=True
        )
    ]
    raw = actual.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "bad_count": int(bad_indices.numel()),
        "first_bad": sample[0] if sample else None,
        "sample": sample,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _run_rank(
    tp: int,
    rank: int,
    shape: tuple[int, int],
    iterations: int,
    distributed_init_method: str,
) -> dict:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp, 1)

    try:
        group = get_tp_group().device_group
        dist.all_reduce(torch.zeros(1, device=device), group=group)
        torch.cuda.synchronize()

        from aiter.dist.communication_op import (
            tensor_model_parallel_fused_allreduce_rmsnorm,
        )

        inputs = []
        references = []
        for phase in range(2):
            input_tensor, expected = _make_pattern(shape, tp, rank, phase, device)
            inputs.append(input_tensor)
            references.append(expected)

        residual = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        weight = torch.ones(shape[-1], dtype=torch.bfloat16, device=device)
        failures = []

        for iteration in range(iterations):
            phase = iteration & 1
            _output, residual_out = tensor_model_parallel_fused_allreduce_rmsnorm(
                inputs[phase], residual, weight, 1e-6
            )
            torch.cuda.synchronize()
            if not torch.equal(residual_out, references[phase]):
                details = _mismatch_details(residual_out, references[phase], shape, tp)
                details.update({"iteration": iteration, "phase": phase})
                failures.append(details)
                print(
                    f"rank={rank} iter={iteration} phase={phase} "
                    f"bad={details['bad_count']} first={details['first_bad']} "
                    f"sha256={details['sha256']}",
                    flush=True,
                )

        return {"rank": rank, "failures": failures}
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test fused AllReduce+RMSNorm IPC publication ordering"
    )
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--m", type=int, default=3828)
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    if args.tp <= 1:
        parser.error("--tp must be greater than one")
    if args.m <= 0 or args.n <= 0:
        parser.error("--m and --n must be positive")
    if args.iters <= 0:
        parser.error("--iters must be positive")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49394"
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    shape = (args.m, args.n)
    with Pool(processes=args.tp) as pool:
        futures = [
            pool.apply_async(
                _run_rank,
                args=(args.tp, rank, shape, args.iters, init_method),
            )
            for rank in range(args.tp)
        ]
        results = [future.get() for future in futures]

    print("SUMMARY", flush=True)
    for result in results:
        first_failure = result["failures"][0] if result["failures"] else None
        print(
            {
                "rank": result["rank"],
                "failure_count": len(result["failures"]),
                "first_failure": first_failure,
            },
            flush=True,
        )
    if any(result["failures"] for result in results):
        raise AssertionError("fused AllReduce+RMSNorm IPC corruption reproduced")


if __name__ == "__main__":
    main()
