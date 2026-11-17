# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""When topk_gating sends a call to the Triton 384-expert sqrtsoftplus top-k.

The predicate reads only tensor metadata, so CPU tensors stand in for the GPU
ones; the arch and the capture state are patched.
"""

import pytest
import torch

pytest.importorskip("triton")
topk_ops = pytest.importorskip("aiter.ops.topk")
from aiter.ops.triton.moe.moe_routing.topk_gating import (
    SQRTSOFTPLUS_384_ARCHS,
)

EXPERTS = 384


def _args(rows, topk=6, experts=EXPERTS, dtype=torch.bfloat16, bias="fp32"):
    logits = torch.empty((rows, experts), dtype=dtype)
    weights = torch.empty((rows, topk), dtype=torch.float32)
    ids = torch.empty((rows, topk), dtype=torch.int32)
    biases = {
        "none": torch.empty(0, dtype=torch.float32),
        "fp32": torch.zeros(experts, dtype=torch.float32),
        "bf16": torch.zeros(experts, dtype=torch.bfloat16),
    }
    return weights, ids, logits, biases[bias]


def _patch(monkeypatch, gfx, capturing):
    monkeypatch.setattr(topk_ops, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(topk_ops, "_sqrtsoftplus_384_arch", topk_ops._UNSET)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)


def _can_use(args):
    return topk_ops._can_use_sqrtsoftplus_384_topk(*args)


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
@pytest.mark.parametrize("capturing", [False, True])
def test_rows_and_capture(monkeypatch, gfx, capturing):
    _patch(monkeypatch, gfx, capturing)
    arch = SQRTSOFTPLUS_384_ARCHS[gfx]
    taken = capturing or not arch.graph_only
    for rows, expected in (
        (0, False),
        (1, taken),
        (arch.max_rows, taken),
        (arch.max_rows + 1, False),
    ):
        assert _can_use(_args(rows)) is expected, rows


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_each_contract_term(monkeypatch, gfx):
    _patch(monkeypatch, gfx, True)
    assert _can_use(_args(8))
    assert _can_use(_args(8, bias="none"))
    assert _can_use(_args(8, topk=1))
    assert _can_use(_args(8, topk=8))
    assert not _can_use(_args(8, dtype=torch.float16))
    assert not _can_use(_args(8, bias="bf16"))
    assert not _can_use(_args(8, topk=9))
    weights, ids, logits, bias = _args(8)
    strided = torch.empty((8, 2 * EXPERTS), dtype=torch.bfloat16)[:, ::2]
    assert not _can_use((weights, ids, strided, bias))
    assert not _can_use((weights, ids, logits.view(1, 8, EXPERTS), bias))
    assert not _can_use((weights[:4], ids, logits, bias))
    assert not _can_use((weights, ids.long(), logits, bias))


def test_an_arch_outside_the_table(monkeypatch):
    _patch(monkeypatch, "gfx942", True)
    assert not _can_use(_args(1))


def test_the_arch_is_looked_up_once(monkeypatch):
    """The entry is read from a module-level value after the first call."""
    calls = []
    gfx = min(SQRTSOFTPLUS_384_ARCHS)
    _patch(monkeypatch, gfx, True)
    monkeypatch.setattr(topk_ops, "get_gfx_runtime", lambda: calls.append(1) or gfx)
    for _ in range(3):
        assert _can_use(_args(1))
    assert len(calls) == 1
    assert topk_ops._sqrtsoftplus_384_arch == SQRTSOFTPLUS_384_ARCHS[gfx]
