#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// epx: low-latency intranode expert-parallel dispatch/combine over XGMI (see
// csrc/kernels/epx.cu for the protocol, aiter/dist/device_communicators/epx.py for the
// Python op that owns the symmetric buffer).

#include "aiter_tensor.h"
#include <cstdint>
#include <vector>

namespace aiter {

using fptr_t = int64_t;

// Symmetric buffer management (raw uncached device memory + HIP IPC).
int64_t epx_alloc_uncached(int64_t bytes);
void epx_free(int64_t ptr);
void epx_get_ipc_handle(int64_t ptr, int64_t out_handle_ptr);
int64_t epx_open_ipc_handle(int64_t handle_ptr);
void epx_close_ipc_handle(int64_t ptr);
int64_t epx_ipc_handle_size();

// bases: every rank's mapping of the symmetric buffer (index = rank, [rank] is local).
// layout: the 11 byte offsets computed by the Python op (see EpxLayout).
fptr_t epx_init(const std::vector<int64_t>& bases,
                const std::vector<int64_t>& layout,
                int64_t rank,
                int64_t topk,
                int64_t experts_per_rank,
                int64_t max_tokens_per_rank,
                int64_t scale_bytes);
void epx_destroy(fptr_t fa);

// x: [T, H] fp8/bf16 rows, scales: [T, scale_bytes] (pass x when scale_bytes == 0),
// topk_ids int32 [T, K] (ids < 0 are dropped), topk_weights fp32 [T, K].
void epx_dispatch(fptr_t fa,
                  const aiter_tensor_t& x,
                  const aiter_tensor_t& scales,
                  const aiter_tensor_t& topk_ids,
                  const aiter_tensor_t& topk_weights,
                  int64_t blocks,
                  int64_t threads);
// expert_out: [>= recv_count, H] bf16 rows in receive order, out: [T, H] bf16.
void epx_combine(fptr_t fa,
                 const aiter_tensor_t& expert_out,
                 const aiter_tensor_t& out,
                 int64_t blocks,
                 int64_t threads);

} // namespace aiter
