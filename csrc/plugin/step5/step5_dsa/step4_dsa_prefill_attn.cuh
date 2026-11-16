// SPDX-License-Identifier: Apache-2.0
// BF16 sparse prefill: FP32 online softmax, BF16 probabilities, GQA8/GQA16.

#include "common.h"
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <hggc_bf16.h>
#include <mma.h>
#include <torch/all.h>

namespace dsa_prefill {
namespace config {

constexpr int kMinBlocks = 8;

// PADZ zero-fills rows outside the source extent.
#ifndef AIU_MODS
  #define AIU_MODS "cs.padz"
#endif

// -1 selects dead-tile skipping from the runtime context length.
#ifndef ATTN_SKIP_DEAD_TILES
  #define ATTN_SKIP_DEAD_TILES (-1)
#endif
static constexpr int kSkipDeadTiles = ATTN_SKIP_DEAD_TILES;

#ifndef ATTN_SORT_SLOTS
  #define ATTN_SORT_SLOTS 1
#endif
static constexpr bool kSortSlots = ATTN_SORT_SLOTS != 0;

static constexpr int kMaxSortSlots = 1024;

#ifndef ATTN_SORT_RUNS
  #define ATTN_SORT_RUNS 0
#endif
static constexpr int kSortRuns = ATTN_SORT_RUNS;

static constexpr int kQ2MaxRegions = 8192;  // kv_len <= 65536
static constexpr int kQ2BitmapWords = (kQ2MaxRegions + 31) / 32;  // 256
static constexpr int kQ2MaxUnionSlots = 2 * 544;  // 2 * ceil(513/32)*32

}  // namespace config
}  // namespace dsa_prefill

#define DSA_REGION_BLOCK_SWITCH(REGION_BLOCK, CONST_NAME, ...)   \
  [&] {                                                          \
    switch (REGION_BLOCK) {                                      \
      case 8: {                                                  \
        constexpr static int CONST_NAME = 8;                     \
        __VA_ARGS__();                                           \
        break;                                                   \
      }                                                          \
      case 4: {                                                  \
        constexpr static int CONST_NAME = 4;                     \
        __VA_ARGS__();                                           \
        break;                                                   \
      }                                                          \
      case 16: {                                                 \
        constexpr static int CONST_NAME = 16;                    \
        __VA_ARGS__();                                           \
        break;                                                   \
      }                                                          \
      case 32: {                                                 \
        constexpr static int CONST_NAME = 32;                    \
        __VA_ARGS__();                                           \
        break;                                                   \
      }                                                          \
      default:                                                   \
        TORCH_CHECK(false,                                       \
                    "region_block_size must be 4, 8, 16 or 32, " \
                    "got ",                                      \
                    REGION_BLOCK);                               \
    }                                                            \
  }()

#define DSA_BOOL_SWITCH(COND, CONST_NAME, ...)  \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      __VA_ARGS__();                            \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      __VA_ARGS__();                            \
    }                                           \
  }()

namespace dsa_prefill {

inline bool skip_dead_tiles(int64_t kv_seq_len, int64_t region_block_size,
                            int64_t region_k) {
  if (config::kSkipDeadTiles >= 0) {
    return config::kSkipDeadTiles != 0;
  }
  const int64_t regions_in_context =
      (kv_seq_len + region_block_size - 1) / region_block_size;
  return regions_in_context < region_k;
}

}  // namespace dsa_prefill

namespace dsa_prefill {

// Fixed D192 base tile. Q2/Q4 staging is selected separately below.
struct PrefillTraits {
  static constexpr int kWarpSize = 32;
  static constexpr int kBlockS = 32;
  static constexpr int kHeadDim = 192;
  static constexpr int kQHeads = 8;
  static constexpr int kMma = 16;
  static constexpr int kWarps = 2;
  static constexpr int kCubeC = 64;
  static constexpr int kTokGrp = 8;
  static constexpr int kThreads = kWarps * kWarpSize;
  static constexpr int kKSteps = kHeadDim / kMma;
  static constexpr int kDimTilesPerWarp = kKSteps / kWarps;
  static constexpr int kDimBlks = kHeadDim / kCubeC;
  static constexpr int kTilesPerBlk = kCubeC / kMma;
};

}  // namespace dsa_prefill

// The HGGC plugin build uses stable symbols; standalone keeps internal linkage.
#ifdef DSA_PREFILL_PLUGIN_BUILD
namespace step5_prefill_v023_kernel {
#else
namespace {
#endif

using Traits = dsa_prefill::PrefillTraits;
constexpr int kWarpSize = Traits::kWarpSize;
constexpr int kHeadDim = Traits::kHeadDim;
constexpr int kQHeads = Traits::kQHeads;  // GQA16 uses two head tiles.
constexpr int kMma = Traits::kMma;
constexpr int kThreads = Traits::kThreads;
constexpr int kKSteps = Traits::kKSteps;
constexpr int kCubeC = Traits::kCubeC;
constexpr int kDimBlks = Traits::kDimBlks;
constexpr int kTilesPerBlk = Traits::kTilesPerBlk;
constexpr int kTokGrp = Traits::kTokGrp;

template <typename Frag>
__device__ __forceinline__ void ldm_x4(Frag& f, const __ppu_bfloat16* p) {
  unsigned* d = reinterpret_cast<unsigned*>(&f.x[0]);
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
      "[%4];\n\t"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a));
}

template <typename Frag>
__device__ __forceinline__ void ldm_x4_trans(Frag& f, const __ppu_bfloat16* p) {
  unsigned* d = reinterpret_cast<unsigned*>(&f.x[0]);
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, "
      "%3}, [%4];\n\t"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a));
}

__device__ __forceinline__ void aiu_copy_group(__ppu_bfloat16* dst_shared,
                                               const __ppu_bfloat16* src_tensor,
                                               int dim_w, int start_w,
                                               int cube_w,
                                               int64_t stride_w_bytes) {
  const unsigned dst =
      static_cast<unsigned>(__cvta_generic_to_shared(dst_shared));
  asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile." AIU_MODS
               ".swzl.b16 "
               "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, "
               "%12}, %13;\n\t"
               :
               : "r"(dst), "l"(src_tensor), "r"(kCubeC), "r"(dim_w),
                 "r"(kDimBlks), "r"(kCubeC), "r"(cube_w), "r"(kDimBlks),
                 "l"(stride_w_bytes), "l"(static_cast<int64_t>(kCubeC) * 2),
                 "r"(0), "r"(start_w), "r"(0),
                 "r"(0)  // swzlMode 0 = 128 B
               : "memory");
}

__device__ __forceinline__ void aiu_copy_block(__ppu_bfloat16* dst_shared,
                                               const __ppu_bfloat16* src_tensor,
                                               int dim_w, int start_w,
                                               int cube_w, int start_c,
                                               int64_t stride_w_bytes) {
  const unsigned dst =
      static_cast<unsigned>(__cvta_generic_to_shared(dst_shared));
  asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile." AIU_MODS
               ".linear.b16 "
               "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, "
               "%12}, %13;\n\t"
               :
               : "r"(dst), "l"(src_tensor), "r"(kHeadDim), "r"(dim_w), "r"(1),
                 "r"(kCubeC), "r"(cube_w), "r"(1), "l"(stride_w_bytes),
                 "l"(stride_w_bytes), "r"(start_c), "r"(start_w), "r"(0), "r"(0)
               : "memory");
}

__device__ __forceinline__ int bucket_reorder(int32_t* __restrict__ dst,
                                              int* __restrict__ hist,
                                              const int32_t* __restrict__ src,
                                              int64_t base, int n,
                                              int region_limit) {
  // Widest bucket that keeps every id this row can keep inside kThreads
  // buckets.
  const int bits = region_limit > 0 ? 32 - __clz(region_limit) : 0;
  const int shift = bits > 6 ? bits - 6 : 0;
  const int mine = static_cast<int>(threadIdx.x);

  hist[mine] = 0;
  __syncthreads();
  for (int i = mine; i < n; i += kThreads) {
    const int r = src[base + i];
    if (r >= 0 && r <= region_limit) atomicAdd(&hist[r >> shift], 1);
  }
  __syncthreads();

  int before = 0;
  int live = 0;
#pragma unroll 8
  for (int b = 0; b < kThreads; ++b) {
    const int c = hist[b];
    if (b < mine) before += c;
    live += c;
  }
  __syncthreads();  // every thread has read hist; now it becomes the cursor
  hist[mine] = before;
  __syncthreads();

  for (int i = mine; i < n; i += kThreads) {
    const int r = src[base + i];
    if (r >= 0 && r <= region_limit) dst[atomicAdd(&hist[r >> shift], 1)] = r;
  }

  __syncthreads();
  // One thread per bucket sorts it ascending, keeping the duplicate and valid-ID
  // rules without depending on experimental macros.
  const int end = hist[mine];
  for (int i = before + 1; i < end; ++i) {
    const int32_t v = dst[i];
    int j = i - 1;
    while (j >= before && dst[j] > v) {
      dst[j + 1] = dst[j];
      --j;
    }
    dst[j + 1] = v;
  }
  __syncthreads();
  return live;
}

#ifndef DSA_Q2_WARPS
  #define DSA_Q2_WARPS 2
#endif
#ifndef DSA_QUERY_BLOCK_S
  #define DSA_QUERY_BLOCK_S 32
#endif
#ifndef DSA_Q4_BLOCK_S
  #define DSA_Q4_BLOCK_S 16
#endif
static_assert(DSA_Q2_WARPS == 1 || DSA_Q2_WARPS == 2);
static_assert(DSA_QUERY_BLOCK_S == 16 || DSA_QUERY_BLOCK_S == 32 ||
              DSA_QUERY_BLOCK_S == 64);

__device__ __forceinline__ int bitset_reorder_union(
    int32_t* __restrict__ dst, int* __restrict__ hist, int32_t* __restrict__ bm,
    const int32_t* __restrict__ src, int64_t base0, int64_t base1, int n,
    int region_limit, bool has_t1, int bm_words) {
  constexpr int kThreads = DSA_Q2_WARPS * 32;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int words = region_limit >= 0 ? (region_limit >> 5) + 1 : 0;
  int32_t* bm1 = bm + bm_words;
  for (int w = tid; w < words; w += kThreads) {
    bm[w] = 0;
    bm1[w] = 0;
  }
  __syncthreads();
  for (int i = tid; i < n; i += kThreads) {
    const int r0 = src[base0 + i];
    if (r0 >= 0 && r0 <= region_limit)
      atomicOr(&bm[r0 >> 5], static_cast<int>(1u << (r0 & 31)));

    if (has_t1) {
      const int r1 = src[base1 + i];
      if (r1 >= 0 && r1 <= region_limit)
        atomicOr(&bm1[r1 >> 5], static_cast<int>(1u << (r1 & 31)));
    }
  }
  __syncthreads();
  constexpr int kWordsPerThread =
      dsa_prefill::config::kQ2BitmapWords / kThreads;
  unsigned a[kWordsPerThread], b[kWordsPerThread];
  int count = 0;
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    const int w = tid * kWordsPerThread + j;
    a[j] = w < words ? static_cast<unsigned>(bm[w]) : 0u;
    b[j] = w < words ? static_cast<unsigned>(bm1[w]) : 0u;
    count += __popc(a[j] | b[j]);
  }
  int inclusive = count;
#pragma unroll
  for (int d = 1; d < 32; d *= 2) {
    const int prev = __shfl_up_sync(0xffffffffu, inclusive, d);
    if (lane >= d) inclusive += prev;
  }
  if (lane == 31) hist[warp] = inclusive;
  __syncthreads();
  int rank = inclusive - count + (warp == 0 ? 0 : hist[0]);
  const int live = hist[0] + (DSA_Q2_WARPS == 2 ? hist[1] : 0);
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    unsigned bits = a[j] | b[j];
    while (bits) {
      const int bit = __ffs(bits) - 1;
      const unsigned mask = 1u << bit;
      const int r = (tid * kWordsPerThread + j) * 32 + bit;
      const unsigned m0 = (a[j] & mask) ? (1u << 30) : 0u;
      const unsigned m1 = (b[j] & mask) ? (1u << 29) : 0u;
      dst[rank++] = static_cast<int32_t>(r | m0 | m1);
      bits &= bits - 1;
    }
  }
  return live;
}

#ifndef DSA_Q4_WARPS
  #define DSA_Q4_WARPS 2
#endif
static_assert(DSA_Q4_WARPS == 2 || DSA_Q4_WARPS == 4);
// Q4 slots pack a 13-bit region ID and four 4-bit visible-token counts.
static_assert(dsa_prefill::config::kQ2MaxRegions <= (1 << 13));

template <int QUAD_WARPS = DSA_Q4_WARPS>
__device__ __forceinline__ int bitset_reorder_quad(
    int32_t* dst, int* hist, int32_t* bm, const int32_t* src,
    const int64_t* positions, int q_base, int q_len, int kv_len, int n,
    int64_t stride_rq, int64_t head_offset) {
  constexpr int kWords = dsa_prefill::config::kQ2BitmapWords;
  constexpr int kQuadThreads = QUAD_WARPS * 32;
  constexpr int kPerThread = kWords / kQuadThreads;
  int horizons[4] = {-1, -1, -1, -1};
  const int tid = threadIdx.x;

  for (int i = tid; i < 4 * kWords; i += kQuadThreads) bm[i] = 0;
  __syncthreads();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    if (q_base + q < q_len) {
      const int64_t pos = positions[q_base + q];
      const int horizon = (pos < 0 ? -1 : static_cast<int>(pos < kv_len - 1 ? pos : kv_len - 1));
      horizons[q] = horizon;
      const int limit = horizon >= 0 ? horizon / 8 : -1;
      for (int i = tid; i < n; i += kQuadThreads) {
        const int r = src[(q_base + q) * stride_rq + head_offset + i];
        if (r >= 0 && r <= limit)
          atomicOr(&bm[q * kWords + (r >> 5)],
                   static_cast<int>(1u << (r & 31)));
      }
    }
  }
  __syncthreads();
  unsigned bits[kPerThread], members[4][kPerThread];
  int count = 0;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    bits[j] = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      members[q][j] =
          static_cast<unsigned>(bm[q * kWords + tid * kPerThread + j]);
      bits[j] |= members[q][j];
    }
    count += __popc(bits[j]);
  }
  int inclusive = count;
#pragma unroll
  for (int d = 1; d < 32; d *= 2) {
    const int prev = __shfl_up_sync(0xffffffffu, inclusive, d);
    if ((tid & 31) >= d) inclusive += prev;
  }
  if ((tid & 31) == 31) hist[tid >> 5] = inclusive;
  __syncthreads();
  int rank = inclusive - count;
  int live = 0;
#pragma unroll
  for (int w = 0; w < QUAD_WARPS; ++w) {
    if (w < (tid >> 5)) rank += hist[w];
    live += hist[w];
  }
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    while (bits[j]) {
      const int bit = __ffs(bits[j]) - 1;
      const int r = (tid * kPerThread + j) * 32 + bit;
      unsigned packed = r;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const unsigned count = members[q][j] & (1u << bit)
                                   ? min(8, max(0, horizons[q] - r * 8 + 1))
                                   : 0;
        packed |= count << (13 + q * 4);
      }
      dst[rank++] = static_cast<int32_t>(packed);
      bits[j] &= bits[j] - 1;
    }
  }
  return live;
}

// Q8 has its own encoding: 13-bit region ID + 8-bit membership; it does not
// extend Q4's count encoding.
template <int Threads>
__device__ __forceinline__ int bitset_reorder_oct(
    int32_t* dst, int* hist, int32_t* bm, const int32_t* src,
    const int64_t* positions, int q_base, int q_len, int kv_len, int n,
    int64_t stride_rq, int64_t head_offset) {
  constexpr int Words = dsa_prefill::config::kQ2BitmapWords;
  static_assert(Words % Threads == 0 && 8 * 544 >= 8 * 513);
  constexpr int PerThread = Words / Threads;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 8 * Words; i += Threads) bm[i] = 0;
  __syncthreads();
  for (int q = 0; q < 8; ++q) {
    if (q_base + q >= q_len) continue;
    const int64_t pos = positions[q_base + q];
    const int horizon = pos < 0 ? -1 : int(min(pos, int64_t(kv_len) - 1));
    for (int i = tid; i < n; i += Threads) {
      const int r = src[int64_t(q_base + q) * stride_rq + head_offset + i];
      if (horizon >= 0 && r >= 0 && r <= horizon / 8)
        atomicOr(&bm[q * Words + (r >> 5)], int(1u << (r & 31)));
    }
  }
  __syncthreads();
  unsigned members[8][PerThread], bits[PerThread];
  int count = 0;
  for (int j = 0; j < PerThread; ++j) {
    bits[j] = 0;
    for (int q = 0; q < 8; ++q) {
      members[q][j] = unsigned(bm[q * Words + tid * PerThread + j]);
      bits[j] |= members[q][j];
    }
    count += __popc(bits[j]);
  }
  int inclusive = count;
  for (int d = 1; d < 32; d *= 2) {
    const int prev = __shfl_up_sync(0xffffffffu, inclusive, d);
    if (lane >= d) inclusive += prev;
  }
  if (lane == 31) hist[warp] = inclusive;
  __syncthreads();
  int rank = inclusive - count, live = 0;
  for (int w = 0; w < Threads / 32; ++w) {
    if (w < warp) rank += hist[w];
    live += hist[w];
  }
  for (int j = 0; j < PerThread; ++j) {
    while (bits[j]) {
      const int bit = __ffs(bits[j]) - 1;
      unsigned packed = (tid * PerThread + j) * 32 + bit;
      for (int q = 0; q < 8; ++q)
        if (members[q][j] & (1u << bit)) packed |= 1u << (13 + q);
      dst[rank++] = int32_t(packed);
      bits[j] &= bits[j] - 1;
    }
  }
  __syncthreads();
  return live;
}

// Reduce across the four lanes that share a head (t1 == head).
__device__ __forceinline__ float head_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float head_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// Column of accumulator/matrix_a element `i` for this lane, i.e. `n` for the QK
// output and `k` for the PV A operand.  Row is `t1 + 8*v1`.
__device__ __forceinline__ int frag_col(int t0, int i) {
  return 2 * t0 + (i & 1) + 8 * ((i >> 2) & 1);
}

__device__ __forceinline__ bool frag_is_real_row(int i) {
  return ((i >> 1) & 1) == 0;
}

__device__ __forceinline__ int frag_token(int i) { return (i >> 1) & 1; }

__device__ __forceinline__ int vmask_bit(int ti, int i) {
  return ti * 4 + (i & 1) + 2 * ((i >> 2) & 1);
}

__device__ __forceinline__ int vmask_bit2(int ti, int i) {
  return ti * 8 + ((i >> 1) & 1) * 4 + (i & 1) + 2 * ((i >> 2) & 1);
}

#ifndef DSA_QUERY_TILE
  #define DSA_QUERY_TILE 2
#endif
#ifndef DSA_ADAPTIVE_Q4
  #define DSA_ADAPTIVE_Q4 1
#endif
#ifndef DSA_GLOBAL_Q4
  #define DSA_GLOBAL_Q4 1
#endif
static_assert(DSA_QUERY_TILE == 2 || DSA_QUERY_TILE == 4);

__device__ __forceinline__ bool classify_query_group(
    const int32_t* ids, const int64_t* positions, int group, int head,
    int q_len, int kv_len, int region_k, int64_t stride_rq, int64_t stride_rh,
    unsigned char* scratch, int* pair_count = nullptr,
    int* quad_count = nullptr) {
  auto* bm = reinterpret_cast<unsigned (*)[256]>(scratch);
  auto* sums = reinterpret_cast<int (*)[2]>(scratch + 2048);
  const int tid = threadIdx.x;
  const int last_full = q_len >= 4 ? q_len - 4 : 0;
  const int first_row = group * 4 < last_full ? group * 4 : last_full;
  for (int w = tid; w < 256; w += 64) bm[0][w] = bm[1][w] = 0;
  __syncthreads();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int row = first_row + q;
    if (row < q_len) {
      const int64_t pos = positions[row];
      const int horizon = (pos < 0 ? -1 : static_cast<int>(pos < kv_len - 1 ? pos : kv_len - 1));
      const int limit = horizon >= 0 ? horizon / 8 : -1;
      for (int i = tid; i < region_k; i += 64) {
        const int r = ids[row * stride_rq + head * stride_rh + i];
        if (r >= 0 && r <= limit) atomicOr(&bm[q / 2][r >> 5], 1u << (r & 31));
      }
    }
  }
  __syncthreads();
  int pairs = 0, quad = 0;
  for (int w = tid; w < 256; w += 64) {
    const unsigned a = bm[0][w], b = bm[1][w];
    pairs += __popc(a) + __popc(b);
    quad += __popc(a | b);
  }
#pragma unroll
  for (int d = 16; d > 0; d /= 2) {
    pairs += __shfl_down_sync(0xffffffffu, pairs, d);
    quad += __shfl_down_sync(0xffffffffu, quad, d);
  }
  if ((tid & 31) == 0) {
    sums[tid >> 5][0] = pairs;
    sums[tid >> 5][1] = quad;
  }
  __syncthreads();
  const bool use_quad =
      (sums[0][1] + sums[1][1]) * 10 < (sums[0][0] + sums[1][0]) * 8;
  if (pair_count) *pair_count = sums[0][0] + sums[1][0];
  if (quad_count) *quad_count = sums[0][1] + sums[1][1];
  __syncthreads();
  return use_quad;
}

__global__ void step4_dsa_classify_query_groups(const int32_t* ids,
                                                const int64_t* positions,
                                                int32_t* flags, int q_len,
                                                int kv_heads, int kv_len,
                                                int region_k, int64_t stride_rq,
                                                int64_t stride_rh) {
  __shared__ __align__(128) unsigned char scratch[2176];
  int pair_count, quad_count;
  const bool quad = classify_query_group(ids, positions, blockIdx.x / kv_heads,
                                         blockIdx.x % kv_heads, q_len, kv_len,
                                         region_k, stride_rq, stride_rh,
                                         scratch, &pair_count, &quad_count);
  if (threadIdx.x == 0) {
    if (DSA_GLOBAL_Q4) {
      flags[2 * blockIdx.x] = pair_count;
      flags[2 * blockIdx.x + 1] = quad_count;
    } else {
      flags[blockIdx.x] = quad;
    }
  }
}

__global__ void step4_dsa_reduce_query_groups(int32_t* flags, int groups) {
  __shared__ int64_t sums[2][2];
  const int tid = threadIdx.x;
  int64_t pairs = 0, quad = 0;
  for (int i = tid; i < groups; i += 64) {
    pairs += flags[2 * i];
    quad += flags[2 * i + 1];
  }
#pragma unroll
  for (int d = 16; d > 0; d /= 2) {
    pairs += __shfl_down_sync(0xffffffffu, pairs, d);
    quad += __shfl_down_sync(0xffffffffu, quad, d);
  }
  if ((tid & 31) == 0) {
    sums[tid >> 5][0] = pairs;
    sums[tid >> 5][1] = quad;
  }
  __syncthreads();
  if (tid == 0)
    flags[2 * groups] =
        (sums[0][1] + sums[1][1]) * 10 < (sums[0][0] + sums[1][0]) * 8;
}

constexpr int query_stage_size(int block_q, int region_block,
                               int head_tiles = 1) {
  return block_q == 8 ? 64 : block_q >= 2 && region_block == 8
             ? (block_q == 4 ? (head_tiles == 2 ? 64 : DSA_Q4_BLOCK_S)
                             : DSA_QUERY_BLOCK_S)
             : Traits::kBlockS;
}

constexpr int query_scratch_size(int block_q, int region_block, bool sorted,
                                 int head_tiles = 1) {
  const int slots = block_q >= 2
                        ? (block_q / 2) * dsa_prefill::config::kQ2MaxUnionSlots
                        : (sorted ? dsa_prefill::config::kMaxSortSlots : 1);
  const int hist = sorted || block_q >= 2 ? kThreads : 1;
  return 2 * query_stage_size(block_q, region_block, head_tiles) * kHeadDim *
             2 +
         (slots + hist) * 4;
}

template <int REGION_BLOCK, bool SKIP_DEAD, bool SORTED, int BLOCK_Q,
          bool FILTERED = false, int HEAD_TILES = 1>
__device__ __forceinline__ void attention_body(
    const __ppu_bfloat16* __restrict__ query,
    const __ppu_bfloat16* __restrict__ key,
    const __ppu_bfloat16* __restrict__ value,
    const int32_t* __restrict__ region_ids,
    const int64_t* __restrict__ query_positions,
    __ppu_bfloat16* __restrict__ output, int64_t stride_qt, int64_t stride_qh,
    int64_t stride_kt, int64_t stride_kh, int64_t stride_vt, int64_t stride_vh,
    int64_t stride_rq, int64_t stride_rh, int64_t stride_ot, int64_t stride_oh,
    int kv_heads, int region_k, int kv_len, int q_len, float scale,
    const int32_t* group_flags, unsigned char* scratch, int program) {
  if constexpr (FILTERED) {
    const int group = ((blockIdx.x / kv_heads) * BLOCK_Q) / 4;
    const int flag_index =
        DSA_GLOBAL_Q4 ? 0 : group * kv_heads + blockIdx.x % kv_heads;
    if ((group_flags[flag_index] != 0) != (BLOCK_Q == 4)) return;
  }
  using namespace awmma;
  // Pipeline Q4 and filtered Q2; the unfiltered Q2 path avoids extra registers.
  constexpr bool kPipelinedKV =
      REGION_BLOCK == 8 &&
      (BLOCK_Q == 4 || (FILTERED && BLOCK_Q == 2 && DSA_Q2_WARPS == 2));
  constexpr bool kMergedHeads = HEAD_TILES == 2 && BLOCK_Q == 4;
  constexpr bool kPackedLimits = BLOCK_Q == 4;
  constexpr bool kOneWarp = BLOCK_Q == 2 && DSA_Q2_WARPS == 1;
  constexpr bool kFullDim =
      BLOCK_Q == 8 || kMergedHeads || kOneWarp || (BLOCK_Q == 4 && DSA_Q4_WARPS == 2);
  constexpr int kDimTilesPerWarp = kFullDim ? 12 : Traits::kDimTilesPerWarp;
  constexpr int kBlockS = query_stage_size(BLOCK_Q, REGION_BLOCK, HEAD_TILES);
  constexpr int kTokTiles = kBlockS / kMma;
  constexpr int kTokGrps = kBlockS / kTokGrp;
  constexpr int kRegionsPerTile = kBlockS / REGION_BLOCK;

  using Stage = __ppu_bfloat16[kTokGrps][kDimBlks][kTokGrp][kCubeC];
  auto& sK = *reinterpret_cast<Stage*>(scratch);
  auto& sV = *reinterpret_cast<Stage*>(scratch + sizeof(Stage));
  auto* sBitmap = reinterpret_cast<int32_t*>(scratch);

  constexpr int kSlotCap =
      (BLOCK_Q >= 2) ? (BLOCK_Q / 2) * dsa_prefill::config::kQ2MaxUnionSlots
                     : (SORTED ? dsa_prefill::config::kMaxSortSlots : 1);
  auto* sSlots = reinterpret_cast<int32_t*>(scratch + 2 * sizeof(Stage));
  auto* sHist = reinterpret_cast<int*>(sSlots + kSlotCap);

  const int q_pair = program / kv_heads;
  const int kv_head = program - q_pair * kv_heads;
  // Each CTA owns BLOCK_Q consecutive query tokens.
  const int warp = threadIdx.x / kWarpSize;
  const int pair_in_cta =
      BLOCK_Q == 8 ? warp % 4 : kMergedHeads ? warp % 2
          : (BLOCK_Q == 4 ? warp / (DSA_Q4_WARPS / 2) : 0);
  const int head_tile = BLOCK_Q == 8 ? warp / 4
      : kMergedHeads ? warp / 2 : static_cast<int>(blockIdx.z);
  const int q_token = q_pair * BLOCK_Q + 2 * pair_in_cta;
  const int q_token1 = q_token + 1;
  const bool has_t1 = (BLOCK_Q >= 2) && (q_token1 < q_len);
  const int lane = threadIdx.x % kWarpSize;
  const int t0 = lane & 3;
  const int head =
      lane >> 2;  // this lane's head for every fragment row v1 == 0

  const int lr = lane & 7;
  const int lu = lane >> 3;
  const int a_col = 8 * (lu >> 1);
  const int b_col = 8 * (lu & 1);

  const int swzl_row = (REGION_BLOCK >= kTokGrp) ? lr : 0;

  const int64_t qpos = q_token < q_len ? query_positions[q_token] : -1;
  const int64_t qpos1 = has_t1 ? query_positions[q_token1] : qpos;

  const int horizon = qpos < 0 ? -1 : static_cast<int>(
      qpos < static_cast<int64_t>(kv_len) - 1 ? qpos : kv_len - 1);
  const int region_limit = horizon >= 0 ? horizon / REGION_BLOCK : -1;
  // The union must stage every region either token can reach, so its horizon is
  // the later token's; the per-token causal test in the mask handles the gap.
  const int64_t qpos_u = qpos1 > qpos ? qpos1 : qpos;
  const int horizon_u = qpos_u < 0 ? -1 : static_cast<int>(
      qpos_u < static_cast<int64_t>(kv_len) - 1 ? qpos_u : kv_len - 1);
  const int region_limit_u = horizon_u >= 0 ? horizon_u / REGION_BLOCK : -1;

  const int64_t rid_base = static_cast<int64_t>(q_token) * stride_rq +
                           static_cast<int64_t>(kv_head) * stride_rh;
  const int64_t rid_base1 = static_cast<int64_t>(q_token1) * stride_rq +
                            static_cast<int64_t>(kv_head) * stride_rh;

  int walk_len = region_k;
  if constexpr (BLOCK_Q == 8) {
    walk_len = bitset_reorder_oct<HEAD_TILES * 128>(
        sSlots, sHist, sBitmap, region_ids, query_positions, q_pair * BLOCK_Q,
        q_len, kv_len, region_k, stride_rq, int64_t(kv_head) * stride_rh);
  } else if constexpr (BLOCK_Q == 4) {
    walk_len = bitset_reorder_quad<kMergedHeads ? 4 : DSA_Q4_WARPS>(
        sSlots, sHist, sBitmap, region_ids, query_positions, q_pair * BLOCK_Q,
        q_len, kv_len, region_k, stride_rq,
        static_cast<int64_t>(kv_head) * stride_rh);
    if constexpr (kMergedHeads && REGION_BLOCK == 8) {
      if (threadIdx.x < kRegionsPerTile) sSlots[walk_len + threadIdx.x] = 0;
    }
    __syncthreads();
  } else if constexpr (BLOCK_Q == 2) {
    walk_len = bitset_reorder_union(
        sSlots, sHist, sBitmap, region_ids, rid_base, rid_base1, region_k,
        region_limit_u, has_t1, dsa_prefill::config::kQ2BitmapWords);
    __syncthreads();
  } else if constexpr (SORTED) {
    walk_len = bucket_reorder(sSlots, sHist, region_ids, rid_base, region_k,
                              region_limit);
    __syncthreads();
  }

  fragment<accumulator, kMma, kMma, kMma, float> oacc[kDimTilesPerWarp];
#pragma unroll
  for (int i = 0; i < kDimTilesPerWarp; ++i) fill_fragment(oacc[i], 0.0f);
  const int dim0 = kFullDim ? 0 : (warp % 2) * kDimTilesPerWarp * kMma;

  float running_max[2] = {-INFINITY, -INFINITY};
  float denom[2] = {0.0f, 0.0f};

  fragment<matrix_a, kMma, kMma, kMma, __ppu_bfloat16, row_major> qfrag[kKSteps];
  {
    const __ppu_bfloat16* qrow =
        query + static_cast<int64_t>(q_token) * stride_qt +
        (static_cast<int64_t>(kv_head) * (kQHeads * HEAD_TILES) +
         (HEAD_TILES == 1
              ? 0
              : head_tile *
                    kQHeads) +
         head) *
            stride_qh;
    const __ppu_bfloat16* qrow1 =
        query + static_cast<int64_t>(q_token1) * stride_qt +
        (static_cast<int64_t>(kv_head) * (kQHeads * HEAD_TILES) +
         (HEAD_TILES == 1
              ? 0
              : head_tile *
                    kQHeads) +
         head) *
            stride_qh;
#pragma unroll
    for (int k = 0; k < kKSteps; ++k) {
      fill_fragment(qfrag[k], __float2bfloat16(0.0f));
      if (q_token < q_len) {
        *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[0]) =
            *reinterpret_cast<const __ppu_bfloat162*>(qrow + k * kMma + 2 * t0);
        *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[4]) =
            *reinterpret_cast<const __ppu_bfloat162*>(qrow + k * kMma + 2 * t0 +
                                                     8);
      }
      if constexpr (BLOCK_Q >= 2) {
        if (has_t1) {
          *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[2]) =
              *reinterpret_cast<const __ppu_bfloat162*>(qrow1 + k * kMma +
                                                       2 * t0);
          *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[6]) =
              *reinterpret_cast<const __ppu_bfloat162*>(qrow1 + k * kMma +
                                                       2 * t0 + 8);
        }
      }
    }
  }

  if constexpr (SKIP_DEAD) {
    constexpr int kVVecs = sizeof(sV) / sizeof(uint4);
    uint4* const vvec = reinterpret_cast<uint4*>(&sV[0][0][0][0]);
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    for (int i = threadIdx.x; i < kVVecs; i += kThreads) vvec[i] = zero;
    __syncthreads();
  }

  // Slots that cannot contribute are dropped by the reorder rather than tested
  // here, so under SORTED every tile but the last is full of live regions.
  for (int slot0 = 0; slot0 < walk_len; slot0 += kRegionsPerTile) {
    int rids[kRegionsPerTile];
    // Q2 stores two membership bits; Q4 stores two 4-bit visible-token counts
    // for the query pair handled by this warp.
    int rmemb[BLOCK_Q >= 2 ? kRegionsPerTile : 1];
    if constexpr (kMergedHeads && REGION_BLOCK == 8) {
#pragma unroll
      for (int rr = 0; rr < kRegionsPerTile; rr += 4) {
        const uint4 pack = *reinterpret_cast<const uint4*>(sSlots + slot0 + rr);
        const unsigned vals[4] = {pack.x, pack.y, pack.z, pack.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          rids[rr + j] = vals[j] & 0x1fffu;
          rmemb[rr + j] = (vals[j] >> (13 + pair_in_cta * 8)) & 255u;
        }
      }
    } else {
#pragma unroll
      for (int rr = 0; rr < kRegionsPerTile; ++rr) {
        const int slot = slot0 + rr;
        if constexpr (BLOCK_Q >= 2) {
          if (slot >= walk_len) {
            rids[rr] = -1;
            rmemb[rr] = 0;
          } else {
            const int packed = sSlots[slot];
            rids[rr] = packed & (BLOCK_Q == 8 || kPackedLimits ? 0x1FFF
                                               : (BLOCK_Q == 4 ? 0x07FFFFFF
                                                               : 0x1FFFFFFF));
            if constexpr (BLOCK_Q == 8)
              rmemb[rr] = (unsigned(packed) >> (13 + pair_in_cta * 2)) & 3u;
            else if constexpr (kPackedLimits)
              rmemb[rr] = (packed >> (13 + pair_in_cta * 8)) & 255;
            else
              rmemb[rr] = ((packed >> (30 - 2 * pair_in_cta)) & 1) |
                          (((packed >> (29 - 2 * pair_in_cta)) & 1) << 1);
          }
        } else if (slot >= walk_len) {
          rids[rr] = -1;
        } else if constexpr (SORTED) {
          rids[rr] = sSlots[slot];
        } else {
          rids[rr] = region_ids[rid_base + static_cast<int64_t>(slot)];
        }
      }
    }

    bool tile_live = false;
#pragma unroll
    for (int rr = 0; rr < kRegionsPerTile; ++rr)
      if (rids[rr] >= 0 && rids[rr] <= region_limit) tile_live = true;
    if (SKIP_DEAD && !tile_live) continue;

#pragma unroll
    for (int copy_round = 0; copy_round < (kOneWarp ? 2 : 1); ++copy_round) {
      if constexpr (kMergedHeads && REGION_BLOCK == 8) {
        if ((warp & 1) || slot0 != 0) continue;
      } else if (warp >= 2)
        continue;
      if constexpr (kPipelinedKV && !kMergedHeads) {
        // Later K tiles were prefetched before the preceding PV step.
        if (warp == 0 && slot0 != 0) continue;
        if (warp == 1) continue;
      }
      // K/V copies use warps 0/1; merged GQA16 Q4 also uses warps 2/3.
      // The instruction must be reached uniformly by the whole warp.
      const int copy_warp = kMergedHeads && REGION_BLOCK == 8 ? 0
                            : kOneWarp                        ? copy_round
                                                              : warp;
      const __ppu_bfloat16* tensor =
          (copy_warp == 0 ? key : value) +
          static_cast<int64_t>(kv_head) *
              (copy_warp == 0 ? stride_kh : stride_vh);
      const int64_t stride_bytes = (copy_warp == 0 ? stride_kt : stride_vt) * 2;
      __ppu_bfloat16* dstbuf =
          (copy_warp == 0 ? &sK[0][0][0][0] : &sV[0][0][0][0]);
      constexpr int kGrpElems = kDimBlks * kTokGrp * kCubeC;

      if constexpr (REGION_BLOCK >= kTokGrp) {
#pragma unroll
        for (int tg = 0; tg < kTokGrps; ++tg) {
          if constexpr (kMergedHeads && REGION_BLOCK == 8) {
            if (tg / (kTokGrps / 2) != warp / 2) continue;
          }
          const int r = rids[(tg * kTokGrp) / REGION_BLOCK];
          int start = r * REGION_BLOCK + (tg * kTokGrp) % REGION_BLOCK;
          if constexpr (SKIP_DEAD) {
            if (r < 0 || r > region_limit) continue;
            if constexpr (REGION_BLOCK > kTokGrp) {
              if (start > horizon) continue;
            }
          } else {
            // A negative region means "no slot".  Clamping to region 0 keeps
            // the copy in bounds; the validity mask discards the result.
            if (r < 0) start = 0;
          }
          aiu_copy_group(dstbuf + tg * kGrpElems, tensor, kv_len, start,
                         kTokGrp, stride_bytes);
        }
      } else {
#pragma unroll
        for (int rr = 0; rr < kRegionsPerTile; ++rr) {
          const int r = rids[rr];
          // A region below the limit has its whole first group inside the
          // horizon, and here a group is never a later part of one region.
          if (SKIP_DEAD && (r < 0 || r > region_limit)) continue;
          const int start = r >= 0 ? r * REGION_BLOCK : 0;
          const int tg = (rr * REGION_BLOCK) / kTokGrp;
          const int within = (rr * REGION_BLOCK) % kTokGrp;
#pragma unroll
          for (int db = 0; db < kDimBlks; ++db) {
            aiu_copy_block(
                dstbuf + tg * kGrpElems + (db * kTokGrp + within) * kCubeC,
                tensor, kv_len, start, REGION_BLOCK, db * kCubeC, stride_bytes);
          }
        }
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    }

    unsigned vmask = 0;
    if constexpr (BLOCK_Q >= 2) {
      const int off = 2 * t0;
      // Union membership excludes invalid regions and out-of-range queries.
      const int qp32[2] = {
          horizon, qpos1 < 0 ? -1 : static_cast<int>(qpos1 < static_cast<int64_t>(kv_len) - 1
                                        ? qpos1
                                        : kv_len - 1)};
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int rr = ti * 2 + half;
          const int r = rids[rr];
          const int memb = rmemb[rr];
          const int base = r * REGION_BLOCK + off;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            // i == half * 4 + e, so tau is e >> 1 and the column parity e & 1.
            const int tau = e >> 1;
            const int tok = base + (e & 1);
            const bool valid = kPackedLimits
                                   ? off + (e & 1) < ((memb >> (tau * 4)) & 15)
                                   : (((memb >> tau) & 1) && r >= 0 && tok >= 0
                                      && tok < kv_len && tok <= qp32[tau]);
            if (valid) vmask |= 1u << (ti * 8 + tau * 4 + (e & 1) + 2 * half);
          }
        }
      }
    } else {
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (!frag_is_real_row(i)) continue;
          const int t = ti * kMma + frag_col(t0, i);
          const int r = rids[t / REGION_BLOCK];
          const int64_t tok =
              static_cast<int64_t>(r) * REGION_BLOCK + (t % REGION_BLOCK);
          if (r >= 0 && tok >= 0 && tok < static_cast<int64_t>(kv_len) &&
              tok <= qpos) {
            vmask |= 1u << vmask_bit(ti, i);
          }
        }
      }
    }

    if constexpr (BLOCK_Q == 8) {
      // Q8 single stage: both K/V copy warps wait for completion before the whole CTA
      // reads the shared buffer.
      if (warp < 2) asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
    } else if constexpr (kOneWarp) {
      asm volatile("ppu.cp.async.wait_group 1;\n\t" ::: "memory");
    } else if constexpr (BLOCK_Q >= 2) {
      if (kMergedHeads && REGION_BLOCK == 8 ? (warp & 1) == 0 : warp == 0)
        asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
    } else {
      asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
    }
    __syncthreads();

    // The preceding barrier also protects V buffer reuse from earlier readers.
#pragma hggc dislicm
    {
      if constexpr (kPipelinedKV) {
        if (kMergedHeads && REGION_BLOCK == 8 ? (warp & 1) != 0 : warp == 1) {
#pragma unroll
          for (int tg = 0; tg < kTokGrps; ++tg) {
            if constexpr (kMergedHeads && REGION_BLOCK == 8) {
              if (tg / (kTokGrps / 2) != warp / 2) continue;
            }
            const int r = rids[tg];
            aiu_copy_group(&sV[0][0][0][0] + tg * kDimBlks * kTokGrp * kCubeC,
                           value + static_cast<int64_t>(kv_head) * stride_vh,
                           kv_len, r >= 0 ? r * REGION_BLOCK : 0, kTokGrp,
                           stride_vt * 2);
          }
          asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
        }
      }
    }

    // QK^T over the current staging tile; col_major reads K transposed.
    fragment<accumulator, kMma, kMma, kMma, float> sacc[kTokTiles];
    {
      fragment<matrix_b, kMma, kMma, kMma, __ppu_bfloat16, col_major> kfrag;
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) fill_fragment(sacc[ti], 0.0f);
#pragma unroll
      for (int k = 0; k < kKSteps; ++k) {
        const int kdb = k / kTilesPerBlk;
        const int kchunk =
            (((k % kTilesPerBlk) * kMma + b_col) >> 3) ^ swzl_row;
#pragma unroll
        for (int ti = 0; ti < kTokTiles; ++ti) {
          // The unit's eight token rows are ti*16 + 8*(lu>>1) .. +7, which is
          // exactly token group ti*2 + (lu>>1), row lr inside it.
          ldm_x4(kfrag, &sK[ti * 2 + (lu >> 1)][kdb][lr][kchunk << 3]);
          mma_sync(sacc[ti], qfrag[k], kfrag, sacc[ti]);
        }
      }
    }

    if constexpr (BLOCK_Q >= 2) {
      float local_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = frag_token(i);
          const float score =
              (vmask & (1u << vmask_bit2(ti, i))) ? sacc[ti].x[i] : -INFINITY;
          local_max[tau] = fmaxf(local_max[tau], score);
          sacc[ti].x[i] = score;
        }
      }
      float new_max[2], rescale[2];
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        const float tile_max = head_max(local_max[tau]);
        const bool has_valid = tile_max > -INFINITY;
        new_max[tau] =
            has_valid ? fmaxf(running_max[tau], tile_max) : running_max[tau];
        rescale[tau] = has_valid
                           ? __expf((running_max[tau] - new_max[tau]) * scale)
                           : 1.0f;
      }

      fragment<matrix_a, kMma, kMma, kMma, __ppu_bfloat16, row_major>
          pfrag[kTokTiles];
      float local_sum[2] = {0.0f, 0.0f};
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const int tau = frag_token(i);
          // Avoid -INF - -INF for rows with no visible tokens.
          const float offset = new_max[tau] == -INFINITY ? 0.0f : new_max[tau];
          float x0 = (sacc[ti].x[i] - offset) * scale;
          float x1 = (sacc[ti].x[i + 1] - offset) * scale;
          if (!(scale > 0.0f)) {
            if (!(vmask & (1u << vmask_bit2(ti, i)))) x0 = -INFINITY;
            if (!(vmask & (1u << vmask_bit2(ti, i + 1)))) x1 = -INFINITY;
          }
          const __ppu_bfloat162 pb =
              __floats2bfloat162_rn(__expf(x0), __expf(x1));
          *reinterpret_cast<__ppu_bfloat162*>(&pfrag[ti].x[i]) = pb;
          const float2 rounded = __bfloat1622float2(pb);
          // Accumulate rounded probabilities in fragment element order.
          local_sum[tau] += rounded.x;
          local_sum[tau] += rounded.y;
        }
      }
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        denom[tau] = denom[tau] * rescale[tau] + local_sum[tau];
        running_max[tau] = new_max[tau];
      }

      // Rescale each output-accumulator element by its own token's factor.
      bool needs_rescale = true;
      // Q4 skips rescaling only when both factors equal one in every lane.
      if constexpr (BLOCK_Q == 4)
        needs_rescale =
            !__all_sync(0xffffffffu, rescale[0] == 1.0f && rescale[1] == 1.0f);
      if (needs_rescale) {
#pragma unroll
        for (int j = 0; j < kDimTilesPerWarp; ++j) {
#pragma unroll
          for (int i = 0; i < 8; ++i) oacc[j].x[i] *= rescale[frag_token(i)];
        }
      }

      // Complete V copies before any warp reads the staging buffer.
      if (kOneWarp ||
          (kMergedHeads && REGION_BLOCK == 8 ? (warp & 1) != 0 : warp == 1))
        asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
      __syncthreads();
      // K readers have finished: reuse the buffer to overlap the next K load
      // with PV. The next iteration waits before QK reads it.
#pragma hggc dislicm
      {
        if constexpr (kPipelinedKV) {
          if ((kMergedHeads && REGION_BLOCK == 8 ? (warp & 1) == 0
                                                 : warp == 0) &&
              slot0 + kRegionsPerTile < walk_len) {
            if constexpr (kMergedHeads && REGION_BLOCK == 8) {
#pragma unroll
              for (int base = 0; base < kTokGrps / 2; base += 4) {
                const uint4 pack = *reinterpret_cast<const uint4*>(
                    sSlots + slot0 + kRegionsPerTile +
                    (warp / 2) * (kTokGrps / 2) + base);
                const unsigned vals[4] = {pack.x, pack.y, pack.z, pack.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  const int r = vals[j] & 0x1fffu;
                  aiu_copy_group(
                      &sK[0][0][0][0] +
                          ((warp / 2) * (kTokGrps / 2) + base + j) * kDimBlks *
                              kTokGrp * kCubeC,
                      key + static_cast<int64_t>(kv_head) * stride_kh, kv_len,
                      r * REGION_BLOCK, kTokGrp, stride_kt * 2);
                }
              }
            } else {
#pragma unroll
              for (int tg = 0; tg < kTokGrps; ++tg) {
                const int slot = slot0 + kRegionsPerTile + tg;
                const int packed = slot < walk_len ? sSlots[slot] : 0;
                const int r = packed & (kPackedLimits ? 0x1FFF : 0x07FFFFFF);
                aiu_copy_group(
                    &sK[0][0][0][0] + tg * kDimBlks * kTokGrp * kCubeC,
                    key + static_cast<int64_t>(kv_head) * stride_kh, kv_len,
                    r * REGION_BLOCK, kTokGrp, stride_kt * 2);
              }
            }
            asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
          }
        }
      }
      // PV over this warp's half of the output dims.
      {
        fragment<matrix_b, kMma, kMma, kMma, __ppu_bfloat16, row_major> vfrag;
#pragma unroll
        for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
          for (int j = 0; j < kDimTilesPerWarp; ++j) {
            const int vdim = dim0 + j * kMma;
            const int vdb = vdim / kCubeC;
            const int vchunk = (((vdim % kCubeC) + a_col) >> 3) ^ swzl_row;
            ldm_x4_trans(vfrag, &sV[ti * 2 + (lu & 1)][vdb][lr][vchunk << 3]);
            mma_sync(oacc[j], pfrag[ti], vfrag, oacc[j]);
          }
        }
      }
      // On the pipelined path V reuse is guarded by the barrier before the next QK;
      // the last round does not reuse it.
      if constexpr (!kPipelinedKV) __syncthreads();
      continue;
    }

    float local_max = -INFINITY;
#pragma unroll
    for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (!frag_is_real_row(i)) continue;
        if (vmask & (1u << vmask_bit(ti, i))) {
          local_max = fmaxf(local_max, sacc[ti].x[i]);
        }
      }
    }
    const float tile_max = head_max(local_max);
    const bool has_valid = tile_max > -INFINITY;
    const float new_max =
        has_valid ? fmaxf(running_max[0], tile_max) : running_max[0];
    const float rescale =
        has_valid ? __expf((running_max[0] - new_max) * scale) : 1.0f;

    // Probabilities go straight into the PV A operand: the accumulator and
    // matrix_a element maps are identical, so element i here is element i
    // there.
    fragment<matrix_a, kMma, kMma, kMma, __ppu_bfloat16, row_major>
        pfrag[kTokTiles];
    float local_sum = 0.0f;
#pragma unroll
    for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float p = 0.0f;
        if (frag_is_real_row(i) && (vmask & (1u << vmask_bit(ti, i)))) {
          p = __expf((sacc[ti].x[i] - new_max) * scale);
        }
        // Storing as bf16 *is* the `prob.to(tl.bfloat16)` rounding the Triton
        // kernel applies before its PV dot.
        const __ppu_bfloat16 pb = __float2bfloat16(p);
        pfrag[ti].x[i] = pb;
        local_sum += __bfloat162float(pb);
      }
    }
    denom[0] = denom[0] * rescale + local_sum;
    running_max[0] = new_max;

#pragma unroll
    for (int j = 0; j < kDimTilesPerWarp; ++j) {
#pragma unroll
      for (int i = 0; i < 8; ++i) oacc[j].x[i] *= rescale;
    }

    {
      fragment<matrix_b, kMma, kMma, kMma, __ppu_bfloat16, row_major> vfrag;
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
        for (int j = 0; j < kDimTilesPerWarp; ++j) {
          // dim0 is warp * 96, which is not a multiple of kCubeC, so the block
          // and inner offset are computed from the absolute dim.
          const int vdim = dim0 + j * kMma;
          const int vdb = vdim / kCubeC;
          const int vchunk = (((vdim % kCubeC) + a_col) >> 3) ^ swzl_row;
          // A-style address form, so the token group is ti*2 + (lu&1).
          ldm_x4_trans(vfrag, &sV[ti * 2 + (lu & 1)][vdb][lr][vchunk << 3]);
          mma_sync(oacc[j], pfrag[ti], vfrag, oacc[j]);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (BLOCK_Q >= 2) {
    // Two query tokens: element i's row-half bit is its token, so token 0 owns
    // elements {0,1,4,5} and token 1 {2,3,6,7}, written to their own output
    // row.
    const float d0 = head_sum(denom[0]);
    const float d1 = head_sum(denom[1]);
    const float inv0 = d0 > 0.0f ? 1.0f / d0 : 0.0f;
    const float inv1 = d1 > 0.0f ? 1.0f / d1 : 0.0f;
    const int64_t gh =
        static_cast<int64_t>(kv_head) * (kQHeads * HEAD_TILES) +
        (HEAD_TILES == 1
             ? 0
             : head_tile *
                   kQHeads) +
        head;
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      if (q_token + tau >= q_len) continue;
      const float inv = tau == 0 ? inv0 : inv1;
      const int qt = tau == 0 ? q_token : q_token1;
      const int64_t o_base =
          static_cast<int64_t>(qt) * stride_ot + gh * stride_oh;
#pragma unroll
      for (int j = 0; j < kDimTilesPerWarp; ++j) {
        const int64_t base = o_base + dim0 + j * kMma;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          // i0 = 2*tau + 4*half: token 0 -> {0,4}, token 1 -> {2,6}; +1 gives
          // the adjacent column, so the two pair into one 32-bit store.
          const int i0 = 2 * tau + 4 * half;
          const __ppu_bfloat162 pair = __floats2bfloat162_rn(
              oacc[j].x[i0] * inv, oacc[j].x[i0 + 1] * inv);
          *reinterpret_cast<__ppu_bfloat162*>(output + base + 2 * t0 +
                                             8 * half) = pair;
        }
      }
    }
  } else {
    const float d = head_sum(denom[0]);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    const int64_t gh =
        static_cast<int64_t>(kv_head) * (kQHeads * HEAD_TILES) +
        (HEAD_TILES == 1
             ? 0
             : head_tile *
                   kQHeads) +
        head;
    const int64_t o_base =
        static_cast<int64_t>(q_token) * stride_ot + gh * stride_oh;
#pragma unroll
    for (int j = 0; j < kDimTilesPerWarp; ++j) {
      const int64_t base = o_base + dim0 + j * kMma;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        // i = 0,1 for half 0 (columns 2*t0, 2*t0+1); i = 4,5 for half 1.
        const int i0 = half * 4;
        const __ppu_bfloat162 pair =
            __floats2bfloat162_rn(oacc[j].x[i0] * inv, oacc[j].x[i0 + 1] * inv);
        *reinterpret_cast<__ppu_bfloat162*>(output + base + 2 * t0 + 8 * half) =
            pair;
      }
    }
  }
}

// clang-format off
template <int REGION_BLOCK, bool SKIP_DEAD, bool SORTED, int BLOCK_Q, bool FILTERED = false, int HEAD_TILES = 1>
__global__ __launch_bounds__((BLOCK_Q == 8 ? HEAD_TILES * 128 : HEAD_TILES == 2 && BLOCK_Q == 4 ? 128 : BLOCK_Q == 4 ? DSA_Q4_WARPS * 32 : (BLOCK_Q == 2 ? DSA_Q2_WARPS * 32 : kThreads)),
                            (BLOCK_Q == 8 ? 1 : HEAD_TILES == 2 && BLOCK_Q == 4 ? 4 : BLOCK_Q == 4 ? 16 / DSA_Q4_WARPS : dsa_prefill::config::kMinBlocks))
void step4_dsa_prefill_attn_kernel(
    const __ppu_bfloat16* query, const __ppu_bfloat16* key, const __ppu_bfloat16* value,
    const int32_t* region_ids, const int64_t* query_positions, __ppu_bfloat16* output,
    int64_t stride_qt, int64_t stride_qh, int64_t stride_kt, int64_t stride_kh,
    int64_t stride_vt, int64_t stride_vh, int64_t stride_rq, int64_t stride_rh,
    int64_t stride_ot, int64_t stride_oh,
    int kv_heads, int region_k, int kv_len, int q_len, float scale,
    const int32_t* group_flags = nullptr) {
  // clang-format on
#if !defined(DSA_PREFILL_PLUGIN_BUILD) || __HGGC_ARCH__ >= 150
  constexpr int bytes =
      query_scratch_size(BLOCK_Q, REGION_BLOCK, SORTED, HEAD_TILES);
  __shared__ __align__(128) unsigned char scratch[bytes];
  attention_body<REGION_BLOCK, SKIP_DEAD, SORTED, BLOCK_Q, FILTERED,
                 HEAD_TILES>(query, key, value, region_ids, query_positions,
                             output, stride_qt, stride_qh, stride_kt, stride_kh,
                             stride_vt, stride_vh, stride_rq, stride_rh,
                             stride_ot, stride_oh, kv_heads, region_k, kv_len,
                             q_len, scale, group_flags, scratch, blockIdx.x);
#endif
}

// Explicit candidates launch full blocks only; the tail rows of a partial block
// launch a separate deterministic Q1.
template <int BQ, int HeadTiles>
void launch_explicit(const at::Tensor& query, const at::Tensor& key,
                     const at::Tensor& value, const at::Tensor& ids,
                     const at::Tensor& pos, at::Tensor& out, int kv_len,
                     float scale, int first, int rows) {
  if (!rows) return;
  constexpr int Threads = BQ == 8 ? HeadTiles * 128
      : BQ == 4 ? (HeadTiles == 2 ? 128 : DSA_Q4_WARPS * 32)
            : BQ == 2 ? DSA_Q2_WARPS * 32 : kThreads;
  const dim3 grid(rows / BQ, 1, BQ >= 4 ? 1 : HeadTiles);
  auto stream = get_current_cuda_stream(query.get_device());
  step4_dsa_prefill_attn_kernel<8, false, true, BQ, false, HeadTiles>
      <<<grid, Threads, 0, stream>>>(
          reinterpret_cast<const __ppu_bfloat16*>(query.data_ptr()) + first * query.stride(0),
          reinterpret_cast<const __ppu_bfloat16*>(key.data_ptr()),
          reinterpret_cast<const __ppu_bfloat16*>(value.data_ptr()),
          ids.data_ptr<int32_t>() + first * ids.stride(0),
          pos.data_ptr<int64_t>() + first,
          reinterpret_cast<__ppu_bfloat16*>(out.data_ptr()) + first * out.stride(0),
          query.stride(0), query.stride(1), key.stride(0), key.stride(1),
          value.stride(0), value.stride(1), ids.stride(0), ids.stride(1),
          out.stride(0), out.stride(1), 1, ids.size(2), kv_len, rows, scale);
}

void step4_dsa_prefill_attn(at::Tensor const& query, at::Tensor const& key,
                            at::Tensor const& value,
                            at::Tensor const& region_ids,
                            at::Tensor const& query_positions,
                            at::Tensor& output, int64_t region_block_size,
                            int64_t kv_seq_len, double scaling, int64_t query_tile = 0) {
  TORCH_CHECK(query_tile == 0 || query_tile == 2 || query_tile == 4 || query_tile == 8,
              "query_tile must be 0, 2, 4 or 8");
  step5_dsa::check_devices(query,
                           {key, value, region_ids, query_positions, output});
  const c10::cuda::CUDAGuard device_guard(query.device());
  step5_dsa::check_scale(scaling);
  TORCH_CHECK(output.dim() == 3, "output must be rank-3");
  TORCH_CHECK(query_positions.is_contiguous(),
              "query_positions must be contiguous");
  TORCH_CHECK(region_block_size == 4 || region_block_size == 8 ||
                  region_block_size == 16 || region_block_size == 32,
              "region_block_size must be 4, 8, 16 or 32");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "query/key/value must be rank-3");
  TORCH_CHECK(region_ids.dim() == 3, "region_ids must be rank-3");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 &&
                  key.scalar_type() == at::kBFloat16 &&
                  value.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "this kernel is specialised for bfloat16");
  TORCH_CHECK(region_ids.scalar_type() == at::kInt, "region_ids must be int32");
  TORCH_CHECK(query_positions.scalar_type() == at::kLong,
              "query_positions must be int64");
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1,
              "key/value must have unit stride along head_dim");
  TORCH_CHECK(key.stride(0) % 8 == 0 && key.stride(1) % 8 == 0 &&
                  value.stride(0) % 8 == 0 && value.stride(1) % 8 == 0,
              "key/value row strides must be multiples of 8 bf16 for 128-bit "
              "loads");
  // The Q fragments load two adjacent bf16 straight from global, so the row
  // start of every (token, head) pair must land on a 4 B boundary.
  TORCH_CHECK(query.stride(2) == 1 && query.stride(0) % 2 == 0 &&
                  query.stride(1) % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(query.data_ptr()) % 4 == 0,
              "query must have unit stride along head_dim, even row strides, "
              "and 4-byte-aligned storage");
  // The epilogue pairs two adjacent dims into one 32-bit store.
  TORCH_CHECK(
      output.stride(2) == 1 && output.stride(0) % 2 == 0 &&
          output.stride(1) % 2 == 0,
      "output must have unit stride along head_dim and even row strides");

  const int64_t q_len = query.size(0);
  const int64_t q_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t kv_len_alloc = key.size(0);
  const int64_t kv_heads = key.size(1);

  TORCH_CHECK(head_dim == kHeadDim, "specialised for head_dim == 192, got ",
              head_dim);
  TORCH_CHECK(value.size(2) == kHeadDim, "value_dim must equal head_dim");
  TORCH_CHECK(kv_heads > 0 && (q_heads == kv_heads * kQHeads ||
                               q_heads == kv_heads * kQHeads * 2),
              "specialised for 8 or 16",
              " query heads per KV head, got q_heads=", q_heads,
              " kv_heads=", kv_heads);
  TORCH_CHECK(region_ids.size(0) == q_len && region_ids.size(1) == kv_heads,
              "region_ids geometry does not match query/key");
  TORCH_CHECK(query_positions.numel() == q_len,
              "query_positions must have one entry per query token");
  TORCH_CHECK(kv_seq_len >= 0 && kv_seq_len <= kv_len_alloc,
              "kv_seq_len must be within the contiguous K/V length");
  TORCH_CHECK(output.size(0) == q_len && output.size(1) == q_heads &&
                  output.size(2) == kHeadDim,
              "output geometry does not match query");

  TORCH_CHECK(key.sizes() == value.sizes() && key.size(2) == kHeadDim,
              "key and value must have matching [tokens,kv_heads,192] shapes");
  TORCH_CHECK(region_ids.stride(2) == 1,
              "region IDs must have contiguous slots");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(key.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(value.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(output.data_ptr()) % 4 == 0,
              "key/value must be 16-byte aligned and output 4-byte aligned");
  TORCH_CHECK(
      output.numel() == 0 ||
          (output.stride(1) >= 192 &&
           output.stride(0) >= (q_heads - 1) * output.stride(1) + 192) ||
          (output.stride(0) >= 192 &&
           output.stride(1) >= (q_len - 1) * output.stride(0) + 192),
      "output rows must not overlap");
  step5_dsa::check_output(output,
                          {query, key, value, region_ids, query_positions});
  step5_dsa::check_int32({q_len, q_heads, kv_heads, kv_seq_len,
                          region_ids.size(2), q_len * kv_heads * 2});
  TORCH_CHECK(region_ids.size(2) <= 1024,
              "at most 1024 region slots are supported");
  const int64_t region_k = region_ids.size(2);
  if (query_tile) {
    TORCH_CHECK(kv_heads == 1 && region_block_size == 8 && region_k <= 513
                    && kv_seq_len <= 65536,
                "explicit v023 Q2/Q4/Q8 requires one KV head, region8, slots<=513, context<=65536");
    if (q_len == 0 || region_k == 0 || kv_seq_len == 0) {
      output.zero_();
      return;
    }
    DSA_BOOL_SWITCH(q_heads == 16, kGqa16, [&] {
      constexpr int HeadTiles = kGqa16 ? 2 : 1;
      DSA_BOOL_SWITCH(query_tile == 2, kPair, [&] {
        DSA_BOOL_SWITCH(query_tile == 8, kOct, [&] {
          constexpr int BQ = kPair ? 2 : kOct ? 8 : 4;
          const int full = q_len / BQ * BQ;
          launch_explicit<BQ, HeadTiles>(query, key, value, region_ids,
              query_positions, output, kv_seq_len, scaling, 0, full);
          launch_explicit<1, HeadTiles>(query, key, value, region_ids,
              query_positions, output, kv_seq_len, scaling, full, q_len - full);
        });
      });
    });
    step5_runtime::check(hggcGetLastError());
    return;
  }
  if (q_len == 0 || region_k == 0 || kv_seq_len == 0) {
    output.zero_();
    return;
  }

  const bool sorted = dsa_prefill::config::kSortSlots &&
                      region_k <= dsa_prefill::config::kMaxSortSlots;

  auto stream = get_current_cuda_stream(query.get_device());

  const bool skip_dead =
      !sorted &&
      dsa_prefill::skip_dead_tiles(kv_seq_len, region_block_size, region_k);

  const int64_t regions_in_context =
      (kv_seq_len + region_block_size - 1) / region_block_size;
  static const bool kQ2Disabled = std::getenv("STEP4_DSA_NO_Q2") != nullptr;
  const int block_q = (!kQ2Disabled && sorted && !skip_dead &&
                       region_block_size == 8 && region_k <= 513 &&
                       regions_in_context <= dsa_prefill::config::kQ2MaxRegions)
                          ? DSA_QUERY_TILE
                          : 1;
  static const bool kQ4Disabled = std::getenv("STEP4_DSA_NO_Q4") != nullptr;
  DSA_BOOL_SWITCH(q_heads == kv_heads * kQHeads * 2, kGqa16, [&] {
    constexpr int kHeadTiles = kGqa16 ? 2 : 1;
    if (DSA_ADAPTIVE_Q4 && !kQ4Disabled && block_q >= 2 && q_len >= 128) {
      const int64_t groups = (q_len + 3) / 4 * kv_heads;
      auto flags = at::empty({DSA_GLOBAL_Q4 ? 2 * groups + 1 : groups},
                             region_ids.options());
      step4_dsa_classify_query_groups<<<groups, 64, 0, stream>>>(
          region_ids.data_ptr<int32_t>(), query_positions.data_ptr<int64_t>(),
          flags.data_ptr<int32_t>(), q_len, kv_heads, kv_seq_len, region_k,
          region_ids.stride(0), region_ids.stride(1));
      if (DSA_GLOBAL_Q4)
        step4_dsa_reduce_query_groups<<<1, 64, 0, stream>>>(
            flags.data_ptr<int32_t>(), groups);
      {
        for (int tile : {2, 4}) {
          DSA_BOOL_SWITCH(tile == 4, kQuad, [&] {
            constexpr int kBQ = kQuad ? 4 : 2;
            const dim3 adaptive_grid((q_len + kBQ - 1) / kBQ * kv_heads, 1,
                                     kGqa16 && kQuad ? 1 : kHeadTiles);
            const dim3 adaptive_block(kGqa16 && kQuad ? 128
                                      : kQuad         ? DSA_Q4_WARPS * 32
                                                      : DSA_Q2_WARPS * 32);
            step4_dsa_prefill_attn_kernel<8, false, true, kBQ, true, kHeadTiles>
                <<<adaptive_grid, adaptive_block, 0, stream>>>(
                    reinterpret_cast<const __ppu_bfloat16*>(query.data_ptr()),
                    reinterpret_cast<const __ppu_bfloat16*>(key.data_ptr()),
                    reinterpret_cast<const __ppu_bfloat16*>(value.data_ptr()),
                    region_ids.data_ptr<int32_t>(),
                    query_positions.data_ptr<int64_t>(),
                    reinterpret_cast<__ppu_bfloat16*>(output.data_ptr()),
                    query.stride(0), query.stride(1), key.stride(0),
                    key.stride(1), value.stride(0), value.stride(1),
                    region_ids.stride(0), region_ids.stride(1),
                    output.stride(0), output.stride(1), kv_heads, region_k,
                    kv_seq_len, q_len, static_cast<float>(scaling),
                    flags.data_ptr<int32_t>() +
                        (DSA_GLOBAL_Q4 ? 2 * groups : 0));
          });
        }
      }
      step5_runtime::check(hggcGetLastError());
      return;
    }
    const dim3 block(kGqa16 && block_q == 4 ? 128
                     : block_q == 4
                         ? DSA_Q4_WARPS * 32
                         : (block_q == 2 ? DSA_Q2_WARPS * 32 : kThreads));
    const int64_t q_pairs = (q_len + block_q - 1) / block_q;
    const dim3 grid(static_cast<unsigned>(q_pairs * kv_heads), 1,
                    kGqa16 && block_q == 4 ? 1 : kHeadTiles);

    DSA_REGION_BLOCK_SWITCH(region_block_size, kRegionBlock, [&] {
      DSA_BOOL_SWITCH(skip_dead, kSkipDead, [&] {
        DSA_BOOL_SWITCH(sorted, kSorted, [&] {
          DSA_BOOL_SWITCH(block_q >= 2, kIsQ2, [&] {
            constexpr int kBQ = kIsQ2 ? DSA_QUERY_TILE : 1;
            step4_dsa_prefill_attn_kernel<kRegionBlock, kSkipDead, kSorted, kBQ,
                                          false, kHeadTiles>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __ppu_bfloat16*>(query.data_ptr()),
                    reinterpret_cast<const __ppu_bfloat16*>(key.data_ptr()),
                    reinterpret_cast<const __ppu_bfloat16*>(value.data_ptr()),
                    region_ids.data_ptr<int32_t>(),
                    query_positions.data_ptr<int64_t>(),
                    reinterpret_cast<__ppu_bfloat16*>(output.data_ptr()),
                    query.stride(0), query.stride(1), key.stride(0),
                    key.stride(1), value.stride(0), value.stride(1),
                    region_ids.stride(0), region_ids.stride(1),
                    output.stride(0), output.stride(1),
                    static_cast<int>(kv_heads), static_cast<int>(region_k),
                    static_cast<int>(kv_seq_len), static_cast<int>(q_len),
                    static_cast<float>(scaling));
          });
        });
      });
    });
  });
  step5_runtime::check(hggcGetLastError());
}

}  // namespace

#ifndef DSA_PREFILL_PLUGIN_BUILD
TORCH_LIBRARY(step4_dsa_attn, m) {
  m.def(
      "prefill_attn(Tensor query, Tensor key, Tensor value, Tensor region_ids, "
      "Tensor query_positions, Tensor(a!) output, int region_block_size, "
      "int kv_seq_len, float scaling, int query_tile=0) -> ()");
}

TORCH_LIBRARY_IMPL(step4_dsa_attn, CUDA, m) {
  m.impl("prefill_attn", &step4_dsa_prefill_attn);
}
#endif
