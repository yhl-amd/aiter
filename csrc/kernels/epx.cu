// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// epx: low-latency intranode expert-parallel dispatch/combine for MI3xx XGMI meshes.
//
// Protocol (per dispatch/combine pair, all flags carry a monotonically increasing epoch so
// nothing is ever reset and no reader has to wait for a peer to clear a slot):
//   dispatch: 1) block 0 routes this rank's tokens (dedup per destination GPU), publishes its
//                per-destination counts to every rank (one-way);
//             2) every block waits for all ranks' counts, derives this rank's write offset in
//                each destination (sum of counts of lower ranks), and pushes each token once to
//                every destination that owns one of its experts (read once, write many);
//             3) the last block to finish signals every destination (one-way) and waits for
//                every source's signal; the receive buffer is then a dense prefix.
//   combine:  1) every received row is pushed back to its source rank's slot [src][tok];
//             2) the last block signals every rank; every block waits for all signals and
//                reduces the <= world_size partial rows of each local token (fp32 accumulate).
// Two one-way hops for dispatch, one for combine; no remote atomics, no remote reads,
// no grid barrier (only "last block" counters), CUDA-graph safe (epoch lives on device).
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "epx.h"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstring>

namespace aiter {
namespace {

#define EPX_MAX_WS 8

struct EpxPeers {
  // Per-rank base pointers of the symmetric allocation (index = rank). [my] is local.
  char* base[EPX_MAX_WS];
};

struct EpxLayout {
  // Byte offsets inside each rank's symmetric allocation.
  long long cnt_mail;     // [WS] x CntMail (one slot per source rank)
  long long data_flag;    // [WS] uint64 (one per source rank)
  long long comb_flag;    // [WS] uint64 (one per destination rank)
  long long recv_x;       // [WS*MAXT] x row_bytes
  long long recv_s;       // [WS*MAXT] x scale_bytes
  long long recv_ids;     // [WS*MAXT] x K int32
  long long recv_w;       // [WS*MAXT] x K float
  long long recv_src;     // [WS*MAXT] int32 (src_rank << 24 | tok)
  long long comb_buf;     // [WS][MAXT] x out_row_bytes
};

struct CntMail {
  int cnt[EPX_MAX_WS];
  unsigned long long epoch;
};

struct EpxArgs {
  EpxPeers peers;
  EpxLayout lay;
  int my, ws, T, K, experts_per_rank, maxt;
  int x_vec;        // x row in 16-byte vectors
  int s_bytes;      // scale bytes per row (0 = none)
  int o_vec;        // combine row in 16-byte vectors (bf16 hidden)
  // local, non-symmetric
  const uint4* x;  const unsigned char* xs;  const int* ids;  const float* w;
  int* pos;                   // [MAXT][WS] (slot of token t in destination d, -1 if not sent)
  unsigned long long* epoch;  // [1]
  int* ctr;                   // [4] block counters
  int* recv_count;            // [1]
  unsigned long long* trace;  // [32] phase timestamps (s_memrealtime), EPX_TRACE builds only
  const uint4* exp_out;       // combine input [R, hidden] bf16
  uint4* out;                 // combine output [T, hidden] bf16
};

__device__ __forceinline__ unsigned long long ld_sys(const unsigned long long* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_sys(unsigned long long* p, unsigned long long v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// This thread's stores have left the CU (all targets are uncached, so no L2 writeback is
// needed for them); the one thread that publishes a flag adds the system-scope release.
__device__ __forceinline__ void wait_stores() { __builtin_amdgcn_s_waitcnt(0); }

// Release before publishing a flag. Everything peers read is uncached, so completing this
// thread's stores is enough; a system-scope fence would also write back the whole L2.
#ifndef EPX_TRACE
#define EPX_TRACE 0
#endif
#define TRACE(slot) do { if (EPX_TRACE && threadIdx.x == 0) a.trace[slot] = __builtin_amdgcn_s_memrealtime(); } while (0)

#ifndef EPX_SYS_FENCE
#define EPX_SYS_FENCE 0
#endif
__device__ __forceinline__ void release_for_peers() {
#if EPX_SYS_FENCE
  __threadfence_system();
#else
  wait_stores();
#endif
}

__device__ __forceinline__ void spin_until(const unsigned long long* p, unsigned long long v) {
  while (ld_sys(p) < v) __builtin_amdgcn_s_sleep(1);
}
// Every rank's count row into shared memory, one row per thread in two vector loads (the
// mailbox is uncached: a serial walk costs a DRAM round trip per element).
__device__ __forceinline__ void load_counts(const CntMail* mail, int ws, int (*s_cnt)[EPX_MAX_WS]) {
  const int tid = threadIdx.x;
  if (tid < ws) {
    int r[EPX_MAX_WS];
    const int4 a0 = reinterpret_cast<const int4*>(mail[tid].cnt)[0];
    const int4 a1 = reinterpret_cast<const int4*>(mail[tid].cnt)[1];
    r[0] = a0.x; r[1] = a0.y; r[2] = a0.z; r[3] = a0.w;
    r[4] = a1.x; r[5] = a1.y; r[6] = a1.z; r[7] = a1.w;
    for (int d = 0; d < EPX_MAX_WS; ++d) s_cnt[tid][d] = r[d];
  }
  __syncthreads();
}

// One 32-byte pos row in two vector loads (it lives in uncached memory: never walk it
// element by element, every load is a full DRAM round trip).
__device__ __forceinline__ void load_pos(const int* p, int (&out)[EPX_MAX_WS]) {
  const int4 a0 = reinterpret_cast<const int4*>(p)[0];
  const int4 a1 = reinterpret_cast<const int4*>(p)[1];
  out[0] = a0.x; out[1] = a0.y; out[2] = a0.z; out[3] = a0.w;
  out[4] = a1.x; out[5] = a1.y; out[6] = a1.z; out[7] = a1.w;
}

__device__ __forceinline__ int pick8(const int4& a0, const int4& a1, int d) {
  int r = a0.x;
  r = d == 1 ? a0.y : r; r = d == 2 ? a0.z : r; r = d == 3 ? a0.w : r;
  r = d == 4 ? a1.x : r; r = d == 5 ? a1.y : r; r = d == 6 ? a1.z : r; r = d == 7 ? a1.w : r;
  return r;
}

template <typename T>
__device__ __forceinline__ T* peer_ptr(const EpxArgs& a, int rank, long long off) {
  // Select with constant indices: a runtime index into the kernel-argument array makes the
  // compiler copy the arguments to scratch.
  char* b = a.peers.base[0];
#pragma unroll
  for (int j = 1; j < EPX_MAX_WS; ++j) b = (rank == j) ? a.peers.base[j] : b;
  return reinterpret_cast<T*>(b + off);
}

// Block 0: route every local token, publish counts. One thread per token per chunk; per
// destination a wave ballot gives the rank of the token within the wave.
__device__ void route_and_publish(const EpxArgs& a, unsigned long long e) {
  __shared__ int s_wave[16][EPX_MAX_WS];  // up to 1024 threads
  __shared__ int s_base[EPX_MAX_WS];
  const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6, nwave = blockDim.x >> 6;
  if (tid < EPX_MAX_WS) s_base[tid] = 0;
  __syncthreads();
  for (int c0 = 0; c0 < a.T; c0 += blockDim.x) {
    const int t = c0 + tid;
    unsigned mask = 0;
    if (t < a.T) {
      for (int k = 0; k < a.K; ++k) {
        int e_id = a.ids[t * a.K + k];
        if (e_id < 0) continue;
        int d = e_id / a.experts_per_rank;
        if (d < a.ws) mask |= 1u << d;
      }
    }
    int local[EPX_MAX_WS];
#pragma unroll
    for (int d = 0; d < EPX_MAX_WS; ++d) {
      unsigned long long b = __ballot((mask >> d) & 1u);
      local[d] = __popcll(b & ((1ull << lane) - 1ull));
      if (lane == 0) s_wave[wave][d] = __popcll(b);
    }
    __syncthreads();
    if (tid < a.ws) {  // exclusive scan over waves, per destination
      int run = s_base[tid];
      for (int wv = 0; wv < nwave; ++wv) { int c = s_wave[wv][tid]; s_wave[wv][tid] = run; run += c; }
      s_base[tid] = run;
    }
    __syncthreads();
    if (t < a.T) {
      int r[EPX_MAX_WS];
#pragma unroll
      for (int d = 0; d < EPX_MAX_WS; ++d) r[d] = ((mask >> d) & 1u) ? s_wave[wave][d] + local[d] : -1;
      reinterpret_cast<int4*>(a.pos + t * EPX_MAX_WS)[0] = make_int4(r[0], r[1], r[2], r[3]);
      reinterpret_cast<int4*>(a.pos + t * EPX_MAX_WS)[1] = make_int4(r[4], r[5], r[6], r[7]);
    }
    __syncthreads();
  }
  wait_stores();
  __syncthreads();
  if (tid < a.ws) {  // publish my count row to every rank (incl. myself)
    CntMail* m = peer_ptr<CntMail>(a, tid, a.lay.cnt_mail) + a.my;
    for (int d = 0; d < a.ws; ++d) m->cnt[d] = s_base[d];
    release_for_peers();
    st_sys(&m->epoch, e);
  }
}

__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 2))) epx_dispatch_kernel(EpxArgs a) {
  const unsigned long long e = *a.epoch + 1;
  __shared__ int s_off[EPX_MAX_WS];
  __shared__ int s_total;
  __shared__ bool s_last;
  const int tid = threadIdx.x, lane = tid & 63;
  if (blockIdx.x == 0) {
    TRACE(0);
    route_and_publish(a, e);
    TRACE(1);
  }
  // Wait for every rank's count row (mine included: block 0 of this kernel).
  CntMail* mail = peer_ptr<CntMail>(a, a.my, a.lay.cnt_mail);
  if (tid < a.ws) spin_until(&mail[tid].epoch, e);
  __syncthreads();
  if (blockIdx.x == 0) TRACE(2);
  __shared__ int s_cnt[EPX_MAX_WS][EPX_MAX_WS];
  load_counts(mail, a.ws, s_cnt);
  if (tid < a.ws) {
    int off = 0;  // my offset in destination tid = counts sent to tid by lower ranks
    for (int s = 0; s < a.my; ++s) off += s_cnt[s][tid];
    s_off[tid] = off;
  }
  if (tid == 0) {
    int tot = 0;
    for (int s = 0; s < a.ws; ++s) tot += s_cnt[s][a.my];
    s_total = tot;
  }
  __syncthreads();

  // Push: one wave per token, read once, write to every destination that owns an expert.
  const int nwave_total = gridDim.x * (blockDim.x >> 6);
  const int gw = blockIdx.x * (blockDim.x >> 6) + (tid >> 6);
  const int s_vec = a.s_bytes / 16;
  // One wave per token: read the row once into named registers, write it to every
  // destination that owns one of its experts. The destination loop is fully unrolled over
  // compile-time indices (a runtime-indexed or spilled array would be reloaded from scratch,
  // and every scratch reload waits on all outstanding remote stores).
  for (int t = gw; t < a.T; t += nwave_total) {
    const int4 pa = reinterpret_cast<const int4*>(a.pos + t * EPX_MAX_WS)[0];
    const int4 pb = reinterpret_cast<const int4*>(a.pos + t * EPX_MAX_WS)[1];
    const uint4* src = a.x + (long long)t * a.x_vec;
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    uint4 v0, v1, v2, v3, v4, v5, v6, v7;
#define EPX_LD(j, vj) vj = (lane + 64 * (j) < a.x_vec) ? src[lane + 64 * (j)] : z4
    EPX_LD(0, v0); EPX_LD(1, v1); EPX_LD(2, v2); EPX_LD(3, v3);
    EPX_LD(4, v4); EPX_LD(5, v5); EPX_LD(6, v6); EPX_LD(7, v7);
#undef EPX_LD
    const uint4 sv = lane < s_vec
        ? reinterpret_cast<const uint4*>(a.xs + (long long)t * a.s_bytes)[lane] : z4;
    const int idv = lane < a.K ? a.ids[t * a.K + lane] : 0;
    const float wv = lane < a.K ? a.w[t * a.K + lane] : 0.f;
    const int rot = t % a.ws;  // tokens start on different links
#pragma unroll
    for (int dd = 0; dd < EPX_MAX_WS; ++dd) {
      int d = dd + rot;
      d = d >= a.ws ? d - a.ws : d;
      const int slot = pick8(pa, pb, d);
      if (dd < a.ws && slot >= 0) {
        int off = 0;
#pragma unroll
        for (int jj = 0; jj < EPX_MAX_WS; ++jj) off = (jj == d) ? s_off[jj] : off;
        const long long row = off + slot;
        uint4* dx = peer_ptr<uint4>(a, d, a.lay.recv_x) + row * a.x_vec;
#define EPX_ST(j, vj) if (lane + 64 * (j) < a.x_vec) dx[lane + 64 * (j)] = vj
        EPX_ST(0, v0); EPX_ST(1, v1); EPX_ST(2, v2); EPX_ST(3, v3);
        EPX_ST(4, v4); EPX_ST(5, v5); EPX_ST(6, v6); EPX_ST(7, v7);
#undef EPX_ST
        if (lane < s_vec) (peer_ptr<uint4>(a, d, a.lay.recv_s) + row * s_vec)[lane] = sv;
        if (lane < a.K) {
          peer_ptr<int>(a, d, a.lay.recv_ids)[row * a.K + lane] = idv;
          peer_ptr<float>(a, d, a.lay.recv_w)[row * a.K + lane] = wv;
        }
        if (lane == 0) peer_ptr<int>(a, d, a.lay.recv_src)[row] = (a.my << 24) | t;
      }
    }
  }
  // Last block to finish tells every destination, then waits for every source.
  if (blockIdx.x == 0) TRACE(3);
  wait_stores();
  __syncthreads();
  if (blockIdx.x == 0) TRACE(4);
  if (tid == 0) s_last = atomicAdd(&a.ctr[0], 1) == (int)gridDim.x - 1;
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) a.ctr[0] = 0;
  if (tid < a.ws) {
    release_for_peers();
    st_sys(peer_ptr<unsigned long long>(a, tid, a.lay.data_flag) + a.my, e);
    spin_until(peer_ptr<unsigned long long>(a, a.my, a.lay.data_flag) + tid, e);
  }
  __syncthreads();
  TRACE(5);
  if (tid == 0) *a.recv_count = s_total;
}

__device__ __forceinline__ void acc_bf16x8(float* f, uint4 v) {
  const __hip_bfloat16* h = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
  for (int i = 0; i < 8; ++i) f[i] += __bfloat162float(h[i]);
}
__device__ __forceinline__ uint4 pack_bf16x8(const float* f) {
  uint4 r;
  __hip_bfloat16* h = reinterpret_cast<__hip_bfloat16*>(&r);
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = __float2bfloat16(f[i]);
  return r;
}

__global__ void __launch_bounds__(256) epx_combine_kernel(EpxArgs a) {
  const unsigned long long e = *a.epoch + 1;
  __shared__ bool s_last;
  const int tid = threadIdx.x, lane = tid & 63;
  const int nwave_total = gridDim.x * (blockDim.x >> 6);
  const int gw = blockIdx.x * (blockDim.x >> 6) + (tid >> 6);
  // 1) push every received row back to its source slot [my][tok]. Rows are stored grouped by
  // source, so walk them round-robin across sources: consecutive waves then target different
  // XGMI links instead of saturating one peer at a time.
  __shared__ int s_seg[EPX_MAX_WS + 1];
  __shared__ int s_maxc;
  __shared__ int s_cnt[EPX_MAX_WS][EPX_MAX_WS];
  if (blockIdx.x == 0) TRACE(8);
  load_counts(peer_ptr<CntMail>(a, a.my, a.lay.cnt_mail), a.ws, s_cnt);
  if (tid == 0) {
    int run = 0, mx = 0;
    for (int s = 0; s < a.ws; ++s) {
      const int c = s_cnt[s][a.my];
      s_seg[s] = run; run += c; mx = c > mx ? c : mx;
    }
    s_seg[a.ws] = run;
    s_maxc = mx;
  }
  __syncthreads();
  const int* rsrc = peer_ptr<int>(a, a.my, a.lay.recv_src);
  const int nslots = s_maxc * a.ws;
  // Few rows: split each row across `psplit` waves to keep more stores in flight.
  int psplit = nslots > 0 ? nwave_total / nslots : 1;
  psplit = psplit < 1 ? 1 : (psplit > 4 ? 4 : psplit);
  const int pchunk = (a.o_vec + psplit - 1) / psplit;
  for (int i = gw; i < nslots * psplit; i += nwave_total) {
    const int slot_i = i / psplit, part = i % psplit;
    const int s = slot_i % a.ws, j = slot_i / a.ws;
    const int r = s_seg[s] + j;
    if (r >= s_seg[s + 1]) continue;
    const int tok = rsrc[r] & 0xffffff;
    const uint4* src = a.exp_out + (long long)r * a.o_vec;
    uint4* dst = peer_ptr<uint4>(a, s, a.lay.comb_buf) + ((long long)a.my * a.maxt + tok) * a.o_vec;
    const int k1 = min(a.o_vec, (part + 1) * pchunk);
    for (int k = part * pchunk + lane; k < k1; k += 64) dst[k] = src[k];
  }
  wait_stores();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(&a.ctr[1], 1) == (int)gridDim.x - 1;
  __syncthreads();
  if (s_last) {
    if (tid == 0) a.ctr[1] = 0;
    if (tid < a.ws) {
      release_for_peers();
      st_sys(peer_ptr<unsigned long long>(a, tid, a.lay.comb_flag) + a.my, e);
    }
  }
  // 2) wait for every destination's rows, reduce locally.
  if (blockIdx.x == 0) TRACE(9);
  if (tid < a.ws) spin_until(peer_ptr<unsigned long long>(a, a.my, a.lay.comb_flag) + tid, e);
  __syncthreads();
  if (blockIdx.x == 0) TRACE(10);
  const uint4* cb = peer_ptr<uint4>(a, a.my, a.lay.comb_buf);
  // Split each token's hidden across `split` waves when tokens are few, so small steps still
  // keep enough loads in flight.
  int split = a.T > 0 ? nwave_total / a.T : 1;
  split = split < 1 ? 1 : (split > 8 ? 8 : split);
  const int chunk = (a.o_vec + split - 1) / split;
  for (int u = gw; u < a.T * split; u += nwave_total) {
    const int t = u / split, part = u % split;
    int p[EPX_MAX_WS];
    load_pos(a.pos + t * EPX_MAX_WS, p);
    const int i0 = part * chunk, i1 = min(a.o_vec, i0 + chunk);
    for (int i = i0 + lane; i < i1; i += 64) {
      uint4 v[EPX_MAX_WS];
#pragma unroll
      for (int d = 0; d < EPX_MAX_WS; ++d)
        if (d < a.ws && p[d] >= 0) v[d] = cb[((long long)d * a.maxt + t) * a.o_vec + i];
      float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int d = 0; d < EPX_MAX_WS; ++d)
        if (d < a.ws && p[d] >= 0) acc_bf16x8(f, v[d]);
      a.out[(long long)t * a.o_vec + i] = pack_bf16x8(f);
    }
  }
  // Last block out advances the epoch for the next dispatch.
  __syncthreads();
  if (blockIdx.x == 0) TRACE(11);
  if (tid == 0 && atomicAdd(&a.ctr[2], 1) == (int)gridDim.x - 1) {
    a.ctr[2] = 0;
    *a.epoch = e;
  }
}

// Host-side handle: the symmetric mappings, the layout and the local state pointers.
struct EpxHandle {
  EpxArgs base;  // everything but the per-call tensors
  int maxt;
};

} // namespace

int64_t epx_alloc_uncached(int64_t bytes) {
  void* p = nullptr;
  HIP_CALL(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
  HIP_CALL(hipMemset(p, 0, bytes));
  return reinterpret_cast<int64_t>(p);
}

void epx_free(int64_t ptr) { HIP_CALL(hipFree(reinterpret_cast<void*>(ptr))); }

int64_t epx_ipc_handle_size() { return sizeof(hipIpcMemHandle_t); }

void epx_get_ipc_handle(int64_t ptr, int64_t out_handle_ptr) {
  HIP_CALL(hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t*>(out_handle_ptr),
                              reinterpret_cast<void*>(ptr)));
}

int64_t epx_open_ipc_handle(int64_t handle_ptr) {
  hipIpcMemHandle_t h;
  memcpy(&h, reinterpret_cast<const void*>(handle_ptr), sizeof(h));
  void* p = nullptr;
  HIP_CALL(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

void epx_close_ipc_handle(int64_t ptr) { HIP_CALL(hipIpcCloseMemHandle(reinterpret_cast<void*>(ptr))); }

fptr_t epx_init(const std::vector<int64_t>& bases,
                const std::vector<int64_t>& layout,
                int64_t rank,
                int64_t topk,
                int64_t experts_per_rank,
                int64_t max_tokens_per_rank,
                int64_t scale_bytes) {
  const int ws = bases.size();
  AITER_CHECK(ws >= 1 && ws <= EPX_MAX_WS, "epx: world size must be in [1, 8], got ", ws);
  AITER_CHECK(rank >= 0 && rank < ws, "epx: invalid rank ", rank);
  AITER_CHECK(layout.size() == 11, "epx: expected 11 layout offsets, got ", layout.size());
  AITER_CHECK(topk <= 64, "epx: topk must be <= 64");
  AITER_CHECK(max_tokens_per_rank < (1 << 24), "epx: max_tokens_per_rank must be < 2^24");
  AITER_CHECK(scale_bytes % 16 == 0 && scale_bytes / 16 <= 64,
              "epx: scale_bytes must be a multiple of 16 and <= 1024");
  auto* h = new EpxHandle{};
  EpxArgs& a = h->base;
  for (int i = 0; i < ws; ++i) a.peers.base[i] = reinterpret_cast<char*>(bases[i]);
  a.lay = EpxLayout{layout[0], layout[1], layout[2], layout[3], layout[4],
                    layout[5], layout[6], layout[7], layout[8]};
  a.my = rank;
  a.ws = ws;
  a.K = topk;
  a.experts_per_rank = experts_per_rank;
  a.maxt = max_tokens_per_rank;
  a.s_bytes = scale_bytes;
  // Local state (not written by peers) also lives in the uncached allocation: it is shared
  // between blocks on different XCDs and between consecutive kernels.
  char* local = a.peers.base[rank];
  a.pos = reinterpret_cast<int*>(local + layout[9]);
  a.epoch = reinterpret_cast<unsigned long long*>(local + layout[10]);
  a.ctr = reinterpret_cast<int*>(local + layout[10] + 8);
  a.recv_count = reinterpret_cast<int*>(local + layout[10] + 24);
  a.trace = reinterpret_cast<unsigned long long*>(local + layout[10] + 64);
  h->maxt = max_tokens_per_rank;
  return reinterpret_cast<fptr_t>(h);
}

void epx_destroy(fptr_t fa) { delete reinterpret_cast<EpxHandle*>(fa); }

void epx_dispatch(fptr_t fa,
                  const aiter_tensor_t& x,
                  const aiter_tensor_t& scales,
                  const aiter_tensor_t& topk_ids,
                  const aiter_tensor_t& topk_weights,
                  int64_t blocks,
                  int64_t threads) {
  auto* h = reinterpret_cast<EpxHandle*>(fa);
  EpxArgs a = h->base;
  a.T = x.size(0);
  AITER_CHECK(a.T <= h->maxt, "epx: ", a.T, " tokens exceed max_tokens_per_rank ", h->maxt);
  AITER_CHECK(x.is_contiguous() && topk_ids.is_contiguous() && topk_weights.is_contiguous(),
              "epx: dispatch inputs must be contiguous");
  AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32, "epx: topk_ids must be int32");
  AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32, "epx: topk_weights must be float32");
  AITER_CHECK(a.T == 0 || topk_ids.size(1) == a.K, "epx: topk_ids has the wrong topk");
  const int64_t row_bytes = x.size(1) * x.element_size();
  AITER_CHECK(row_bytes % 16 == 0 && row_bytes / 16 <= 8 * 64,
              "epx: row bytes must be a multiple of 16 and <= 8192");
  a.x_vec = row_bytes / 16;
  a.x = reinterpret_cast<const uint4*>(x.data_ptr());
  if (a.s_bytes) {
    AITER_CHECK(scales.is_contiguous() && scales.size(1) * scales.element_size() == a.s_bytes,
                "epx: scales must be contiguous [T, scale_bytes]");
    a.xs = reinterpret_cast<const unsigned char*>(scales.data_ptr());
  }
  a.ids = reinterpret_cast<const int*>(topk_ids.data_ptr());
  a.w = reinterpret_cast<const float*>(topk_weights.data_ptr());
  hipLaunchKernelGGL(epx_dispatch_kernel, dim3(blocks), dim3(threads), 0,
                     aiter::getCurrentHIPStream(), a);
  HIP_CALL(hipGetLastError());
}

void epx_combine(fptr_t fa,
                 const aiter_tensor_t& expert_out,
                 const aiter_tensor_t& out,
                 int64_t blocks,
                 int64_t threads) {
  auto* h = reinterpret_cast<EpxHandle*>(fa);
  EpxArgs a = h->base;
  AITER_CHECK(out.dtype() == AITER_DTYPE_bf16 && expert_out.dtype() == AITER_DTYPE_bf16,
              "epx: combine is bf16 only");
  AITER_CHECK(expert_out.is_contiguous() && out.is_contiguous(),
              "epx: combine tensors must be contiguous");
  a.T = out.size(0);
  a.o_vec = out.size(1) * out.element_size() / 16;
  a.exp_out = reinterpret_cast<const uint4*>(expert_out.data_ptr());
  a.out = reinterpret_cast<uint4*>(out.data_ptr());
  hipLaunchKernelGGL(epx_combine_kernel, dim3(blocks), dim3(threads), 0,
                     aiter::getCurrentHIPStream(), a);
  HIP_CALL(hipGetLastError());
}

} // namespace aiter
