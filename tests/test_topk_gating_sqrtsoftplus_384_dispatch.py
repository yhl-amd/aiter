# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""When topk_gating_sqrtsoftplus_384 sends a call to the Triton top-k.

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


def _entry(args, need_renorm=True, score_func="sqrtsoftplus", shared=0):
    weights, ids, logits, bias = args
    topk_ops.topk_gating_sqrtsoftplus_384(
        weights, ids, logits, bias, need_renorm, 1.5, score_func, shared
    )


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
def test_only_the_graph_entry_runs_triton(monkeypatch, gfx):
    _patch(monkeypatch, gfx)
    ran = _record_kernels(monkeypatch)
    weights, ids, logits, bias = _args(8)
    topk_ops.topk_gating(weights, ids, logits, bias, True, 1.5, "sqrtsoftplus")
    _entry((weights, ids, logits, bias))
    _entry((weights, ids, logits, None))
    assert ran == ["hip", "triton", "triton"]


def _strided(rows, cols, dtype):
    return torch.empty((rows, 2 * cols), dtype=dtype)[:, ::2]


# Each case breaks one term of the entry's guard; the call must reach HIP.
_OUTSIDE = {
    "no_renorm": lambda: {"args": _args(8), "need_renorm": False},
    "sigmoid": lambda: {"args": _args(8), "score_func": "sigmoid"},
    "softmax": lambda: {"args": _args(8), "score_func": "softmax"},
    "shared": lambda: {"args": _args(8), "shared": 1},
    "no_rows": lambda: {"args": _args(0)},
    "experts_256": lambda: {"args": _args(8, experts=256, bias="none")},
    "experts_512": lambda: {"args": _args(8, experts=512, bias="none")},
    "logits_fp16": lambda: {"args": _args(8, dtype=torch.float16)},
    "bias_bf16": lambda: {"args": _args(8, bias="bf16")},
    "bias_383": lambda: {
        "args": _args(8)[:3] + (torch.zeros(383, dtype=torch.float32),)
    },
    "bias_192": lambda: {
        "args": _args(8)[:3] + (torch.zeros(192, dtype=torch.float32),)
    },
    "bias_stride_2": lambda: {
        "args": _args(8)[:3] + (torch.zeros(768, dtype=torch.float32)[::2],)
    },
    "topk_0": lambda: {"args": _args(8, topk=0)},
    "topk_9": lambda: {"args": _args(8, topk=9)},
    "logits_col_stride": lambda: {
        "args": _args(8)[:2] + (_strided(8, EXPERTS, torch.bfloat16), _args(8)[3])
    },
    # [8, 1, 384] with unit stride on dim 1: every other term holds.
    "logits_3d": lambda: {
        "args": _args(8)[:2]
        + (
            torch.empty((8, EXPERTS), dtype=torch.bfloat16).as_strided(
                (8, 1, EXPERTS), (EXPERTS, 1, 1)
            ),
            _args(8)[3],
        )
    },
    "weights_col_stride": lambda: {
        "args": (_strided(8, 6, torch.float32),) + _args(8)[1:]
    },
    "ids_col_stride": lambda: {
        "args": _args(8)[:1] + (_strided(8, 6, torch.int32),) + _args(8)[2:]
    },
    "weights_fewer_rows": lambda: {
        "args": (torch.empty((4, 6), dtype=torch.float32),) + _args(8)[1:]
    },
    "ids_fewer_rows": lambda: {
        "args": _args(8)[:1] + (torch.empty((4, 6), dtype=torch.int32),) + _args(8)[2:]
    },
    "weights_rows_overlap": lambda: {
        "args": (torch.empty((1, 6), dtype=torch.float32).expand(8, 6),) + _args(8)[1:]
    },
    "ids_rows_overlap": lambda: {
        "args": _args(8)[:1]
        + (torch.empty((1, 6), dtype=torch.int32).expand(8, 6),)
        + _args(8)[2:]
    },
    "outputs_fewer_rows": lambda: {
        "args": (
            torch.empty((4, 6), dtype=torch.float32),
            torch.empty((4, 6), dtype=torch.int32),
        )
        + _args(8)[2:]
    },
    "weights_fp16": lambda: {
        "args": (torch.empty((8, 6), dtype=torch.float16),) + _args(8)[1:]
    },
    "ids_1d": lambda: {
        "args": _args(8)[:1] + (torch.empty(8, dtype=torch.int32),) + _args(8)[2:]
    },
    "ids_int64": lambda: {
        "args": _args(8)[:1] + (torch.empty((8, 6), dtype=torch.int64),) + _args(8)[2:]
    },
    "ids_wider": lambda: {
        "args": _args(8)[:1] + (torch.empty((8, 8), dtype=torch.int32),) + _args(8)[2:]
    },
}


@pytest.mark.parametrize("gfx", sorted(SQRTSOFTPLUS_384_ARCHS))
@pytest.mark.parametrize("case", sorted(_OUTSIDE))
def test_a_call_outside_the_limits_runs_hip(monkeypatch, gfx, case):
    _patch(monkeypatch, gfx)
    ran = _record_kernels(monkeypatch)
    _entry(**_OUTSIDE[case]())
    assert ran == ["hip"], case


def test_the_herd_opt_in_goes_to_topk_gating(monkeypatch):
    _patch(monkeypatch, min(SQRTSOFTPLUS_384_ARCHS))
    monkeypatch.setattr(topk_ops, "_FLYDSL_USE_HERD", True)
    monkeypatch.setattr(topk_ops, "_use_flydsl_herd", lambda *a: False)
    ran = _record_kernels(monkeypatch)
    _entry(_args(8))
    assert ran == ["hip"]


def test_topk_gating_does_no_triton_check(monkeypatch):
    """topk_gating goes to HIP as on the merge base, without the entry's check."""

    def no_check(*args, **kwargs):
        pytest.fail("topk_gating ran the Triton dispatch check")

    _patch(monkeypatch, min(SQRTSOFTPLUS_384_ARCHS))
    monkeypatch.setattr(topk_ops, "_can_use_sqrtsoftplus_384_topk", no_check)
    ran = _record_kernels(monkeypatch)
    weights, ids, logits, bias = _args(8)
    topk_ops.topk_gating(weights, ids, logits, bias, True, 1.5, "sqrtsoftplus")
    assert ran == ["hip"]
    assert topk_ops._sqrtsoftplus_384_arch is topk_ops._UNSET
