# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from ..jit.core import compile_ops

MD_NAME = "module_epx"


@compile_ops("module_epx", develop=True)
def epx_alloc_uncached(bytes: int) -> int: ...


@compile_ops("module_epx", develop=True)
def epx_free(ptr: int) -> None: ...


@compile_ops("module_epx", develop=True)
def epx_ipc_handle_size() -> int: ...


@compile_ops("module_epx", develop=True)
def epx_get_ipc_handle(ptr: int, out_handle_ptr: int) -> None: ...


@compile_ops("module_epx", develop=True)
def epx_open_ipc_handle(handle_ptr: int) -> int: ...


@compile_ops("module_epx", develop=True)
def epx_close_ipc_handle(ptr: int) -> None: ...


@compile_ops("module_epx", develop=True)
def epx_init(
    bases: list[int],
    layout: list[int],
    rank: int,
    topk: int,
    experts_per_rank: int,
    max_tokens_per_rank: int,
    scale_bytes: int,
) -> int: ...


@compile_ops("module_epx", develop=True)
def epx_destroy(fa: int) -> None: ...


@compile_ops("module_epx", develop=True)
def epx_dispatch(
    fa: int,
    x: torch.Tensor,
    scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    blocks: int,
    threads: int,
) -> None: ...


@compile_ops("module_epx", develop=True)
def epx_combine(
    fa: int,
    expert_out: torch.Tensor,
    out: torch.Tensor,
    blocks: int,
    threads: int,
) -> None: ...
