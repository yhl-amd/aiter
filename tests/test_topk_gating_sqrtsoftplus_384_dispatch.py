# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""When topk_gating sends a call to the Triton 384-expert sqrtsoftplus top-k.

The dispatch reads only tensor metadata, so CPU tensors stand in for the GPU
ones; the arch is patched, and the two kernels are replaced by recorders.
"""

import pytest
import torch

pytest.importorskip("triton")
topk_ops = pytest.importorskip("aiter.ops.topk")
from aiter.ops.triton.moe.moe_routing import topk_gating as topk_gating_launchers
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


def _patch(monkeypatch, gfx):
    monkeypatch.setattr(topk_ops, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(topk_ops, "_sqrtsoftplus_384_arch", topk_ops._UNSET)


def _can_use(args):
    return topk_ops._can_use_sqrtsoftplus_384_topk(*args)


def _record_kernels(monkeypatch):
    """Replace both kernels; return the list of the kernels that ran."""
    ran = []
    monkeypatch.setattr(topk_ops, "topk_gating_fwd", lambda *a: ran.append("hip"))
    monkeypatch.setattr(
        topk_gating_launchers,
        "_sqrtsoftplus_384_topk_launch",
        lambda *a: ran.append("triton"),
    )
    return ran


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_rows(monkeypatch, gfx):
    _patch(monkeypatch, gfx)
    max_rows = SQRTSOFTPLUS_384_ARCHS[gfx].max_rows
    for rows, expected in (
        (0, False),
        (1, True),
        (max_rows, True),
        (max_rows + 1, False),
    ):
        assert _can_use(_args(rows)) is expected, rows


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_each_contract_term(monkeypatch, gfx):
    _patch(monkeypatch, gfx)
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
    _patch(monkeypatch, "gfx942")
    assert not _can_use(_args(1))


def test_the_arch_is_looked_up_once(monkeypatch):
    """The entry is read from a module-level value after the first call."""
    calls = []
    gfx = min(SQRTSOFTPLUS_384_ARCHS)
    _patch(monkeypatch, gfx)
    monkeypatch.setattr(topk_ops, "get_gfx_runtime", lambda: calls.append(1) or gfx)
    for _ in range(3):
        assert _can_use(_args(1))
    assert len(calls) == 1
    assert topk_ops._sqrtsoftplus_384_arch == SQRTSOFTPLUS_384_ARCHS[gfx]


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_only_a_caller_that_asks_gets_triton(monkeypatch, gfx):
    _patch(monkeypatch, gfx)
    ran = _record_kernels(monkeypatch)
    weights, ids, logits, bias = _args(8)
    topk_ops.topk_gating(weights, ids, logits, bias, True, 1.5, "sqrtsoftplus")
    topk_ops.topk_gating(
        weights, ids, logits, bias, True, 1.5, "sqrtsoftplus", prefer_triton_384=True
    )
    assert ran == ["hip", "triton"]


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
@pytest.mark.parametrize(
    "need_renorm, score_func, rows",
    [(False, "sqrtsoftplus", 8), (True, "sigmoid", 8), (True, "sqrtsoftplus", 0)],
)
def test_a_call_outside_the_limits_runs_hip_when_asked(
    monkeypatch, gfx, need_renorm, score_func, rows
):
    _patch(monkeypatch, gfx)
    ran = _record_kernels(monkeypatch)
    weights, ids, logits, bias = _args(rows)
    topk_ops.topk_gating(
        weights,
        ids,
        logits,
        bias,
        need_renorm,
        1.5,
        score_func,
        prefer_triton_384=True,
    )
    assert ran == ["hip"]


def test_a_default_call_does_no_triton_check(monkeypatch):
    """Without prefer_triton_384 the call goes to HIP as on the merge base."""

    def no_check(*args, **kwargs):
        pytest.fail("a default call ran the Triton dispatch check")

    _patch(monkeypatch, min(SQRTSOFTPLUS_384_ARCHS))
    monkeypatch.setattr(topk_ops, "_can_use_sqrtsoftplus_384_topk", no_check)
    ran = _record_kernels(monkeypatch)
    weights, ids, logits, bias = _args(8)
    topk_ops.topk_gating(weights, ids, logits, bias, True, 1.5, "sqrtsoftplus")
    assert ran == ["hip"]
    assert topk_ops._sqrtsoftplus_384_arch is topk_ops._UNSET
