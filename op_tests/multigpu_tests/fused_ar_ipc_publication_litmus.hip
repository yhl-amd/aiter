// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Standalone 8-GPU litmus for peer-store publication through an end flag.
// It bypasses Python, RMSNorm, and AITER dispatch so that the only protocol is:
//
//   peer data stores -> block barrier -> end-flag store/load -> local data load
//
// Modes isolate three implementations plus a positive control:
//   early   publishes before rank 7 writes; it must fail
//   relaxed models the historical final_sync=true path fixed by AITER #2890
//   main    models current main before the all-writer-fence proposal
//   fixed   adds the proposed all-writer system fence and system acquire
//
// A PASS only says that no violation was observed on that run and hardware. In
// particular, if both main and fixed pass, this test does not justify the extra
// fence. Conversely, early must fail or the test has no useful observation
// window. On 8 x gfx950, relaxed and early fail while main and fixed pass.
//
// Build:
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     fused_ar_ipc_publication_litmus.hip -o /tmp/fused_ar_ipc_litmus
//
// Run:
//   /tmp/fused_ar_ipc_litmus --mode early --iters 2
//   /tmp/fused_ar_ipc_litmus --mode relaxed --iters 1000
//   /tmp/fused_ar_ipc_litmus --mode main  --iters 1000
//   /tmp/fused_ar_ipc_litmus --mode fixed --iters 1000
//   /tmp/fused_ar_ipc_litmus --mode main --iters 10000 \
//     --writer-delay-cycles 5000000

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr int kRanks = 8;
constexpr int kBlocks = 80;
constexpr int kThreads = 512;
constexpr int kThreadsPerTarget = kThreads / kRanks;
constexpr int kTailBurstStores = 32;

enum Mode : int {
    kMain = 0,
    kFixed = 1,
    kRelaxed = 2,
    kEarly = 3,
};

struct PeerPointers {
    uint64_t* data[kRanks];
    uint32_t* flags[kRanks];
};

struct ErrorRecord {
    unsigned long long count;
    unsigned int first_set;
    unsigned int first_iteration;
    unsigned int first_producer;
    unsigned long long first_offset;
    unsigned long long first_actual;
    unsigned long long first_expected;
};

struct Options {
    Mode mode = kMain;
    int iterations = 1000;
    size_t words_per_producer = 1U << 20;  // 8 MiB per producer, 64 MiB per GPU.
    unsigned long long early_delay_cycles = 50'000'000ULL;
    unsigned long long writer_delay_cycles = 0;
};

#define HIP_CHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        hipError_t error = (call);                                                       \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            std::fprintf(stderr, "%s:%d HIP error: %s\n", __FILE__, __LINE__,           \
                         hipGetErrorString(error));                                      \
            std::exit(2);                                                               \
        }                                                                                \
    } while(0)

__host__ __device__ inline uint64_t expected_word(
    unsigned int iteration, unsigned int producer, size_t offset)
{
    return 0xd1b54a32d192ed03ULL ^
           (static_cast<uint64_t>(iteration) * 0x9e3779b97f4a7c15ULL) ^
           (static_cast<uint64_t>(producer) * 0x94d049bb133111ebULL) ^
           static_cast<uint64_t>(offset);
}

template <Mode mode>
__device__ inline void signal_and_wait(
    const PeerPointers& peers, uint32_t* self_flags, int rank, unsigned int iteration)
{
    if(threadIdx.x < kRanks)
    {
        constexpr int store_order =
            mode == kRelaxed ? __ATOMIC_RELAXED : __ATOMIC_RELEASE;
        __scoped_atomic_store_n(
            &peers.flags[threadIdx.x][blockIdx.x * kRanks + rank],
            iteration,
            store_order,
            __MEMORY_SCOPE_SYSTEM);

        constexpr int load_order =
            mode == kRelaxed ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE;
        constexpr int acquire_scope =
            mode == kFixed ? __MEMORY_SCOPE_SYSTEM : __MEMORY_SCOPE_DEVICE;
        while(__scoped_atomic_load_n(
                  &self_flags[blockIdx.x * kRanks + threadIdx.x],
                  load_order,
                  acquire_scope) < iteration)
            ;
    }
    __syncthreads();
}

template <Mode mode>
__global__ void publish_kernel(PeerPointers peers,
                               uint32_t* self_flags,
                               int rank,
                               size_t words_per_producer,
                               unsigned int iteration,
                               unsigned long long early_delay_cycles,
                               unsigned long long writer_delay_cycles)
{
    const int target = threadIdx.x / kThreadsPerTarget;
    const int lane = threadIdx.x % kThreadsPerTarget;

    if constexpr(mode == kEarly)
    {
        // Deliberately publish completion before rank 7 writes. This is a
        // positive control: the validator must observe missing rank-7 data.
        signal_and_wait<mode>(peers, self_flags, rank, iteration);
        if(rank == kRanks - 1)
        {
            const unsigned long long start = clock64();
            while(clock64() - start < early_delay_cycles)
                asm volatile("" ::: "memory");
        }
    }

    // Optional stressor for the exact missing-writer-fence hypothesis. Wave 0
    // reaches the barrier first; waves 1-7 issue their peer stores immediately
    // before arriving. The signal threads are lanes 0-7 of wave 0.
    if(target != 0 && writer_delay_cycles != 0)
    {
        const unsigned long long start = clock64();
        while(clock64() - start < writer_delay_cycles)
            asm volatile("" ::: "memory");
    }

    uint64_t* target_data = peers.data[target] + rank * words_per_producer;
    for(size_t offset = blockIdx.x * kThreadsPerTarget + lane;
        offset < words_per_producer;
        offset += gridDim.x * kThreadsPerTarget)
    {
        target_data[offset] = expected_word(iteration, rank, offset);
    }

    // Finish with back-to-back stores so non-signal waves arrive at the block
    // barrier with the largest practical vmcnt backlog instead of a throttled
    // loop tail. These stores overwrite a valid prefix with the same sentinel.
#pragma unroll
    for(int burst = 0; burst < kTailBurstStores; ++burst)
    {
        const size_t offset =
            (blockIdx.x * kTailBurstStores + burst) * kThreadsPerTarget + lane;
        if(offset < words_per_producer)
            target_data[offset] = expected_word(iteration, rank, offset);
    }
    __syncthreads();

    if constexpr(mode != kEarly)
    {
        if constexpr(mode == kFixed)
        {
            // A fence is per calling thread. Every writer executes it, then a
            // block barrier prevents the signal wave from publishing early.
            __threadfence_system();
            __syncthreads();
        }
        signal_and_wait<mode>(peers, self_flags, rank, iteration);
    }
}

__global__ void validate_kernel(const uint64_t* local_data,
                                size_t words_per_producer,
                                unsigned int iteration,
                                ErrorRecord* errors)
{
    const size_t total_words = kRanks * words_per_producer;
    const size_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    unsigned long long local_errors = 0;

    for(size_t index = thread; index < total_words; index += stride)
    {
        const unsigned int producer = index / words_per_producer;
        const size_t offset = index % words_per_producer;
        const uint64_t expected = expected_word(iteration, producer, offset);
        const uint64_t actual = local_data[index];
        if(actual != expected)
        {
            ++local_errors;
            if(atomicCAS(&errors->first_set, 0U, 1U) == 0U)
            {
                errors->first_iteration = iteration;
                errors->first_producer = producer;
                errors->first_offset = offset;
                errors->first_actual = actual;
                errors->first_expected = expected;
            }
        }
    }

    if(local_errors != 0)
        atomicAdd(&errors->count, local_errors);
}

Mode parse_mode(const char* value)
{
    if(std::strcmp(value, "main") == 0)
        return kMain;
    if(std::strcmp(value, "fixed") == 0)
        return kFixed;
    if(std::strcmp(value, "relaxed") == 0)
        return kRelaxed;
    if(std::strcmp(value, "early") == 0)
        return kEarly;
    std::fprintf(stderr, "unknown mode: %s\n", value);
    std::exit(2);
}

const char* mode_name(Mode mode)
{
    switch(mode)
    {
    case kMain: return "main";
    case kFixed: return "fixed";
    case kRelaxed: return "relaxed";
    case kEarly: return "early";
    }
    return "unknown";
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--mode" && i + 1 < argc)
            options.mode = parse_mode(argv[++i]);
        else if(arg == "--iters" && i + 1 < argc)
            options.iterations = std::stoi(argv[++i]);
        else if(arg == "--words" && i + 1 < argc)
            options.words_per_producer = std::stoull(argv[++i]);
        else if(arg == "--early-delay-cycles" && i + 1 < argc)
            options.early_delay_cycles = std::stoull(argv[++i]);
        else if(arg == "--writer-delay-cycles" && i + 1 < argc)
            options.writer_delay_cycles = std::stoull(argv[++i]);
        else
        {
            std::fprintf(stderr,
                         "usage: %s --mode main|fixed|relaxed|early [--iters N] "
                         "[--words N] [--early-delay-cycles N] "
                         "[--writer-delay-cycles N]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    if(options.iterations <= 0 || options.words_per_producer == 0)
    {
        std::fprintf(stderr, "iterations and words must be positive\n");
        std::exit(2);
    }
    return options;
}

template <Mode mode>
void launch_publish(const PeerPointers& peers,
                    uint32_t* self_flags,
                    int rank,
                    size_t words_per_producer,
                    unsigned int iteration,
                    unsigned long long early_delay_cycles,
                    unsigned long long writer_delay_cycles,
                    hipStream_t stream)
{
    publish_kernel<mode><<<kBlocks, kThreads, 0, stream>>>(
        peers, self_flags, rank, words_per_producer, iteration,
        early_delay_cycles, writer_delay_cycles);
    HIP_CHECK(hipGetLastError());
}

}  // namespace

int main(int argc, char** argv)
{
    const Options options = parse_options(argc, argv);
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if(device_count < kRanks)
    {
        std::fprintf(stderr, "need %d GPUs, found %d\n", kRanks, device_count);
        return 2;
    }

    for(int source = 0; source < kRanks; ++source)
    {
        HIP_CHECK(hipSetDevice(source));
        for(int target = 0; target < kRanks; ++target)
        {
            if(source == target)
                continue;
            int can_access = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access, source, target));
            if(!can_access)
            {
                std::fprintf(stderr, "GPU %d cannot access GPU %d\n", source, target);
                return 2;
            }
            const hipError_t error = hipDeviceEnablePeerAccess(target, 0);
            if(error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
                HIP_CHECK(error);
        }
    }

    const size_t data_bytes =
        kRanks * options.words_per_producer * sizeof(uint64_t);
    const size_t flag_bytes = kBlocks * kRanks * sizeof(uint32_t);
    uint64_t* data[kRanks]{};
    uint32_t* flags[kRanks]{};
    ErrorRecord* errors[kRanks]{};
    hipStream_t streams[kRanks]{};

    for(int rank = 0; rank < kRanks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipStreamCreateWithFlags(&streams[rank], hipStreamNonBlocking));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data[rank]), data_bytes));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&flags[rank]), flag_bytes));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&errors[rank]), sizeof(ErrorRecord)));
        HIP_CHECK(hipMemsetAsync(data[rank], 0, data_bytes, streams[rank]));
        HIP_CHECK(hipMemsetAsync(flags[rank], 0, flag_bytes, streams[rank]));
        HIP_CHECK(hipMemsetAsync(errors[rank], 0, sizeof(ErrorRecord), streams[rank]));
        HIP_CHECK(hipStreamSynchronize(streams[rank]));
    }

    PeerPointers peers{};
    for(int rank = 0; rank < kRanks; ++rank)
    {
        peers.data[rank] = data[rank];
        peers.flags[rank] = flags[rank];
    }

    std::printf("mode=%s iterations=%d producer_payload=%.1f MiB local_buffer=%.1f MiB\n",
                mode_name(options.mode),
                options.iterations,
                options.words_per_producer * sizeof(uint64_t) / 1048576.0,
                data_bytes / 1048576.0);

    for(unsigned int iteration = 1;
        iteration <= static_cast<unsigned int>(options.iterations);
        ++iteration)
    {
        for(int rank = 0; rank < kRanks; ++rank)
        {
            HIP_CHECK(hipSetDevice(rank));
            switch(options.mode)
            {
            case kMain:
                launch_publish<kMain>(peers, flags[rank], rank,
                                      options.words_per_producer, iteration,
                                      options.early_delay_cycles,
                                      options.writer_delay_cycles, streams[rank]);
                break;
            case kFixed:
                launch_publish<kFixed>(peers, flags[rank], rank,
                                       options.words_per_producer, iteration,
                                       options.early_delay_cycles,
                                       options.writer_delay_cycles, streams[rank]);
                break;
            case kRelaxed:
                launch_publish<kRelaxed>(peers, flags[rank], rank,
                                         options.words_per_producer, iteration,
                                         options.early_delay_cycles,
                                         options.writer_delay_cycles, streams[rank]);
                break;
            case kEarly:
                launch_publish<kEarly>(peers, flags[rank], rank,
                                       options.words_per_producer, iteration,
                                       options.early_delay_cycles,
                                       options.writer_delay_cycles, streams[rank]);
                break;
            }
        }

        for(int rank = 0; rank < kRanks; ++rank)
        {
            HIP_CHECK(hipSetDevice(rank));
            validate_kernel<<<256, 256, 0, streams[rank]>>>(
                data[rank], options.words_per_producer, iteration, errors[rank]);
            HIP_CHECK(hipGetLastError());
        }
    }

    for(int rank = 0; rank < kRanks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipStreamSynchronize(streams[rank]));
    }

    unsigned long long total_errors = 0;
    for(int rank = 0; rank < kRanks; ++rank)
    {
        ErrorRecord host{};
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipMemcpy(&host, errors[rank], sizeof(host), hipMemcpyDeviceToHost));
        total_errors += host.count;
        std::printf("rank=%d errors=%llu", rank, host.count);
        if(host.count != 0)
        {
            std::printf(" first={iter=%u producer=%u offset=%llu actual=0x%016llx "
                        "expected=0x%016llx}",
                        host.first_iteration,
                        host.first_producer,
                        host.first_offset,
                        host.first_actual,
                        host.first_expected);
        }
        std::printf("\n");
    }

    for(int rank = 0; rank < kRanks; ++rank)
    {
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipFree(errors[rank]));
        HIP_CHECK(hipFree(flags[rank]));
        HIP_CHECK(hipFree(data[rank]));
        HIP_CHECK(hipStreamDestroy(streams[rank]));
    }

    std::printf("RESULT: %s (%llu mismatched words)\n",
                total_errors == 0 ? "PASS" : "FAIL", total_errors);
    return total_errors == 0 ? 0 : 1;
}
