// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Ported from vllm-step5-dsa-890p-20261110/step5_sparse_decode.cu.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hggc_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <optional>
#include "step5_native_device.cuh"
#include "step5_runtime.h"
namespace step5_native_decode {
using namespace step5_dsa_native;
constexpr int kMaxSplits = 32, kMergeWarps = 4, kSingleWarpRows = 16, kDirectRows = 4096;
constexpr int kSingleRowTile = 8, kSingleRowMaxSplits = 96, kSingleRowMergeWidth = 32, kSingleRowMergeThreads = 256;
struct Decode {
  PagedKV kv;
  const __ppu_bfloat16 *q, *gate;
  const int64_t *regions, *positions;
  const bool* active;
  __ppu_bfloat16* out;
  float *partial, *partial_lse;
  int rows, heads, slots, splits, groups, rows_per_request;
  int64_t qs, qh, os, oh, rs, gs, gh;
  bool clear_inactive;
  __device__ float gate_value(int row, int head) const {
    if (!gate) return 1.f;
    const int h = (row % groups) * heads + head;
    const float x = __bfloat162float(gate[int64_t(row / groups) * gs + h * gh]);
    return __bfloat162float(__float2bfloat16_rn(__fdiv_rn(1.f, 1.f + expf(-x))));
  }
  __device__ __ppu_bfloat162 output_pair(float a, float b, float sigmoid) const {
    if (gate) { a = __bfloat162float(__float2bfloat16_rn(a)) * sigmoid; b = __bfloat162float(__float2bfloat16_rn(b)) * sigmoid; }
    return __floats2bfloat162_rn(a, b);
  }
  float scale;
};
// Kernel definitions stay visible to both architectures; the arch guards cover
// device computation only.
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
  return x;
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}
template <int Heads, int TG, int Warps, bool Direct, bool SingleRow = false>
__global__ __launch_bounds__(32 * Warps, Warps == 1 ? 4 : 1) void packed_attention(Decode p) {
#if __HGGC_ARCH__ >= 150
  constexpr bool Fast = Warps > 1 || SingleRow;
  static_assert(!SingleRow || (TG == kSingleRowTile && Warps == 1 && !Direct));
  constexpr int NT = TG / 2, Halves = Heads / 8;
  constexpr float kLog2e = 1.4426950408889634f, kLn2 = 0.6931471805599453f;
  const int warp = Warps == 1 ? 0 : __builtin_ppu_to_uniform_b32(threadIdx.x / 32);
  const int block = blockIdx.x * Warps + warp;
  if constexpr (Warps > 1) { if (block >= (Direct ? p.rows : p.rows * p.splits)) return; }
  const int row = SingleRow ? 0 : Direct ? block : Warps == 1 ? block / p.splits : block % p.rows;
  const int split = SingleRow ? block : Direct ? 0 : Warps == 1 ? block % p.splits : block / p.rows;
  const int lane = threadIdx.x & 31, t0 = lane & 3, lr = lane & 7, lu = lane >> 3, head = lane >> 2;
  const int a_col = 8 * (lu >> 1), b_col = 8 * (lu & 1);
  const int pos = checked_position(p.positions[row]);
  const bool active = p.active[row];
  if (!active && !p.clear_inactive && (Direct || p.splits == 1)) return;
  const int head_base = SingleRow ? blockIdx.y * Heads : 0, group = SingleRow ? 0 : row % p.groups;
  const int request = SingleRow ? 0 : row / p.rows_per_request;
  const int live = active && pos >= 0 ? min(p.slots, pos / 8 + 1) : 0, tiles = (live + TG - 1) / TG;
  const int stride = (live + p.splits - 1) / p.splits;
  const int begin = SingleRow ? split * TG : Direct ? 0 : Warps == 1 ? split * stride : (split * tiles / p.splits) * TG;
  const int end = SingleRow ? min(live, begin + TG) : Direct ? live : Warps == 1 ? min(live, begin + stride) : min(live, ((split + 1) * tiles / p.splits) * TG);
  const int width = end - begin;
  const bool write_output = Direct || p.splits == 1 || (SingleRow && live <= TG);
  if constexpr (SingleRow) { if (p.splits > 1 && width <= 0) return; }
  const auto* regions = p.regions + int64_t(row) * p.rs + p.slots - live;
  const auto* kv = p.kv.data + group * p.kv.group_stride;
  __shared__ __align__(128) KvStage<TG> key_stage[Warps], value_stage[Warps];
  auto& sk = key_stage[warp]; auto& sv = value_stage[warp];
  MmaOperand qfrag[kHeadMmaTiles]; MmaAccumulator output[kHeadMmaTiles];
#pragma unroll
  for (int d = 0; d < kHeadMmaTiles; ++d) {
    clear_fragment(qfrag[d]); clear_fragment(output[d]);
#pragma unroll
    for (int tau = 0; tau < Halves; ++tau) {
      const auto* src = p.q + int64_t(row) * p.qs + (head_base + head + tau * 8) * p.qh;
      qfrag[d].x[tau] = *reinterpret_cast<const uint32_t*>(src + 16 * d + 2 * t0);
      qfrag[d].x[tau + 2] = *reinterpret_cast<const uint32_t*>(src + 16 * d + 2 * t0 + 8);
    }
  }
  float maximum[Halves], denom[Halves];
#pragma unroll
  for (int tau = 0; tau < Halves; ++tau) { maximum[tau] = Fast ? -FLT_MAX : -INFINITY; denom[tau] = 0.f; }
  const float scale2 = p.scale * kLog2e;
  uint32_t region = 0; int visible = 0;
  for (int base = 0; base < width; base += TG) {
    if ((base & 31) == 0) {
      const int64_t id = base + lane < width ? regions[begin + base + lane] : -1;
      const int phys = id >= 0 && id <= pos / 8 && pos >= 0 ? p.kv.physical_region(request, id) : -1;
      visible = phys < 0 ? 0 : int(min(int64_t(kRegionTokens), int64_t(pos) - id * 8 + 1));
      // Every branch broadcasts the physical region; the 64-bit stride is no longer
      // packed into a 32-bit lane value.
      region = uint32_t(max(phys, 0));
      if constexpr (Direct) { if (!__any_sync(0xffffffffu, visible != 0)) { base += 32 - TG; continue; } }
    }
    int ids[TG], counts[TG]; bool nonempty = false, full = true;
#pragma unroll
    for (int r = 0; r < TG; ++r) {
      ids[r] = __builtin_ppu_to_uniform_b32(__shfl_sync(0xffffffffu, region, (base & 31) + r));
      counts[r] = __builtin_ppu_to_uniform_b32(__shfl_sync(0xffffffffu, visible, (base & 31) + r));
      nonempty |= counts[r] != 0; full &= counts[r] == kRegionTokens;
    }
    if (!nonempty) continue;
#pragma unroll
    for (int r = 0; r < TG; ++r) {
      const int start = counts[r] == 0 ? kRegionTokens : 0;
      const int64_t offset = p.kv.offset(uint32_t(ids[r]));
      aiu_copy_group(&sk[r][0][0][0], kv + offset, counts[r] + start, start, kRegionTokens, p.kv.row_bytes());
      aiu_copy_group(&sv[r][0][0][0], kv + offset + kHeadDim, counts[r] + start, start, kRegionTokens, p.kv.row_bytes());
    }
    asm volatile("ppu.cp.async.commit_group;" ::: "memory"); asm volatile("ppu.cp.async.wait_group 0;" ::: "memory"); __syncwarp();
    MmaAccumulator scores[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) clear_fragment(scores[t]);
#pragma unroll
    for (int d = 0; d < kHeadMmaTiles; ++d) {
      const int chunk = (((d % 4) * 16 + b_col) >> 3) ^ lr; MmaOperand key[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) ldm_x4(key[t], &sk[2 * t + (lu >> 1)][d / 4][lr][chunk * 8]);
#pragma unroll
      for (int t = 0; t < NT; ++t) mma_bf16(scores[t], qfrag[d], key[t], scores[t]);
    }
    float peak[Halves];
#pragma unroll
    for (int tau = 0; tau < Halves; ++tau) peak[tau] = -INFINITY;
    if (Fast && full) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { const int tau = (i >> 1) & 1; if (tau < Halves) peak[tau] = fmaxf(peak[tau], scores[t].x[i]); }
      }
    } else {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = (i >> 1) & 1;
          const bool valid = tau < Halves && 2 * t0 + (i & 1) < counts[2 * t + (i >> 2)];
          scores[t].x[i] = valid ? scores[t].x[i] : -INFINITY;
          if (tau < Halves) peak[tau] = fmaxf(peak[tau], scores[t].x[i]);
        }
      }
    }
    if constexpr (SingleRow) {
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau) maximum[tau] = head_max(peak[tau]) * scale2;
    } else if constexpr (!Fast) {
      float rescale[Halves];
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau) {
        const float tile_peak = head_max(peak[tau]) * scale2, next = fmaxf(maximum[tau], tile_peak);
        rescale[tau] = next > -INFINITY ? __exp2f(maximum[tau] - next) : 1.f; denom[tau] *= rescale[tau]; maximum[tau] = next;
      }
#pragma unroll
      for (int d = 0; d < kHeadMmaTiles; ++d) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { const int tau = (i >> 1) & 1; if (tau < Halves) output[d].x[i] *= rescale[tau]; }
      }
    } else {
      bool increase = false;
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau) increase |= peak[tau] * scale2 > maximum[tau];
      if (__any_sync(0xffffffffu, increase)) {
        float rescale[Halves];
#pragma unroll
        for (int tau = 0; tau < Halves; ++tau) {
          const float local = head_max(peak[tau]) * scale2;
          const float bound = maximum[tau] == -FLT_MAX ? local : ceilf(local * 0.125f) * 8.f;
          const float next = local > maximum[tau] ? bound : maximum[tau];
          rescale[tau] = __exp2f(maximum[tau] - next); denom[tau] *= rescale[tau]; maximum[tau] = next;
        }
#pragma unroll
        for (int d = 0; d < kHeadMmaTiles; ++d) {
#pragma unroll
          for (int i = 0; i < 8; ++i) { const int tau = (i >> 1) & 1; if (tau < Halves) output[d].x[i] *= rescale[tau]; }
        }
      }
    }
    MmaOperand prob[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int tau = (i >> 1) & 1; float a = 0.f, b = 0.f;
        if (tau < Halves && ((Fast && !SingleRow) || maximum[tau] > -INFINITY)) {
          a = __exp2f(fmaf(scores[t].x[i], scale2, -maximum[tau])); b = __exp2f(fmaf(scores[t].x[i + 1], scale2, -maximum[tau])); denom[tau] += a + b;
        }
        *reinterpret_cast<__ppu_bfloat162*>(&prob[t].x[i / 2]) = __floats2bfloat162_rn(a, b);
      }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int d = 0; d < kHeadMmaTiles; ++d) {
        MmaOperand value; const int chunk = (((d % 4) * 16 + a_col) >> 3) ^ lr;
        ldm_x4_trans(value, &sv[2 * t + (lu & 1)][d / 4][lr][chunk * 8]); mma_bf16(output[d], prob[t], value, output[d]);
      }
    }
    __syncwarp(); if constexpr (SingleRow) break;
  }
#pragma unroll
  for (int tau = 0; tau < Halves; ++tau) {
    const int head_index = head_base + head + tau * 8; const int64_t h = int64_t(row) * Heads + head_index, scalar = h * p.splits + split;
    const float norm = head_sum(denom[tau]), inv = SingleRow && !write_output ? 1.f : (norm > 0 ? 1.f / norm : 0.f);
    const float lse = norm > 0 ? maximum[tau] * kLn2 + __logf(norm) : -INFINITY;
    if (t0 == 0 && !write_output) {
      if constexpr (SingleRow) { p.partial_lse[scalar] = norm > 0.f ? maximum[tau] : -INFINITY; p.partial_lse[scalar + p.heads * p.splits] = norm; }
      else p.partial_lse[scalar] = lse;
    }
    const float sigmoid = write_output ? p.gate_value(row, head_index) : 1.f;
    if constexpr (!Direct && Fast) { if (!write_output && norm == 0.f) continue; }
#pragma unroll
    for (int d = 0; d < kHeadMmaTiles; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int dim = d * 16 + 2 * t0 + half * 8, i = 2 * tau + 4 * half;
        const float a = output[d].x[i] * inv, b = output[d].x[i + 1] * inv;
        if (write_output) *reinterpret_cast<__ppu_bfloat162*>(p.out + int64_t(row) * p.os + head_index * p.oh + dim) = p.output_pair(a, b, sigmoid);
        else *reinterpret_cast<float2*>(p.partial + scalar * kHeadDim + dim) = make_float2(a, b);
      }
    }
  }
#else
  __builtin_trap();
#endif
}
template <int Threads> __global__ void merge_attention(Decode p) {
  const int64_t h = blockIdx.x; const int tid = threadIdx.x, row = h / p.heads, head = h % p.heads;
  if (!p.active[row] && !p.clear_inactive) return;
  const int64_t first = h * p.splits; __shared__ float weights[kMaxSplits];
  if (tid < 32) {
    const float local = tid < p.splits ? p.partial_lse[first + tid] : -INFINITY, peak = warp_max(local);
    const float weight = peak > -INFINITY ? __expf(local - peak) : 0.f, sum = warp_sum(weight);
    weights[tid] = sum > 0 ? weight / sum : 0.f;
  }
  __syncthreads(); if (tid >= kHeadDim / 4) return;
  float4 result = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s = 0; s < p.splits; ++s) {
    const float4 data = reinterpret_cast<const float4*>(p.partial + (first + s) * kHeadDim)[tid]; const float w = weights[s];
    result.x = fmaf(w, data.x, result.x); result.y = fmaf(w, data.y, result.y); result.z = fmaf(w, data.z, result.z); result.w = fmaf(w, data.w, result.w);
  }
  auto* dst = reinterpret_cast<__ppu_bfloat162*>(p.out + int64_t(row) * p.os + head * p.oh + tid * 4);
  const float sigmoid = p.gate_value(row, head); dst[0] = p.output_pair(result.x, result.y, sigmoid); dst[1] = p.output_pair(result.z, result.w, sigmoid);
}
template <int Threads> __global__ void merge_attention_warp(Decode p) {
  const int64_t h = int64_t(blockIdx.x) * kMergeWarps + threadIdx.x / 32;
  const int lane = threadIdx.x & 31, row = h / p.heads, head = h % p.heads;
  if (!p.active[row] && !p.clear_inactive) return;
  const int64_t first = h * p.splits;
  const float local = lane < p.splits ? p.partial_lse[first + lane] : -INFINITY, peak = warp_max(local);
  const float weight = peak > -INFINITY ? __expf(local - peak) : 0.f, sum = warp_sum(weight), w = sum > 0 ? weight / sum : 0.f;
  uint32_t active = __ballot_sync(0xffffffffu, w != 0.f); float2 result[3] = {};
  while (active) {
    const int split = __ffs(active) - 1; const float scale = __shfl_sync(0xffffffffu, w, split);
    const auto* data = reinterpret_cast<const float2*>(p.partial + (first + split) * kHeadDim) + lane;
#pragma unroll
    for (int d = 0; d < 3; ++d) { const float2 value = data[d * 32]; result[d].x = fmaf(scale, value.x, result[d].x); result[d].y = fmaf(scale, value.y, result[d].y); }
    active &= active - 1;
  }
  auto* dst = reinterpret_cast<__ppu_bfloat162*>(p.out + int64_t(row) * p.os + head * p.oh) + lane;
  const float sigmoid = p.gate_value(row, head);
#pragma unroll
  for (int d = 0; d < 3; ++d) dst[d * 32] = p.output_pair(result[d].x, result[d].y, sigmoid);
}
template <int Threads> __global__ void merge_single_row(Decode p) {
  constexpr int Width = kSingleRowMergeWidth, Lanes = Width / 4, Tiles = kHeadDim / Width;
  constexpr int SplitGroups = kSingleRowMergeThreads / Lanes, Warps = kSingleRowMergeThreads / 32;
  constexpr int Loads = (kSingleRowMaxSplits + SplitGroups - 1) / SplitGroups;
  const int head = blockIdx.x / Tiles, feature_lane = threadIdx.x % Lanes, dim = (blockIdx.x % Tiles) * Width + 4 * feature_lane;
  const int split_group = threadIdx.x / Lanes, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if (!p.active[0] && !p.clear_inactive) return;
  __shared__ float weights[kSingleRowMaxSplits]; __shared__ float4 sums[Warps - 1][Lanes];
  const int pos = checked_position(p.positions[0]); const int live = p.active[0] && pos >= 0 ? min(p.slots, pos / 8 + 1) : 0;
  const int live_splits = (live + kSingleRowTile - 1) / kSingleRowTile;
  if (live_splits == 1) return;
  const int first = head * p.splits;
  if (threadIdx.x < 32) {
    float peaks[kSingleRowMaxSplits / 32], peak = -INFINITY;
#pragma unroll
    for (int i = 0; i < kSingleRowMaxSplits / 32; ++i) { const int split = lane + i * 32; peaks[i] = split < live_splits ? p.partial_lse[first + split] : -INFINITY; peak = fmaxf(peak, peaks[i]); }
    peak = warp_max(peak); float norm = 0.f;
#pragma unroll
    for (int i = 0; i < kSingleRowMaxSplits / 32; ++i) {
      const int split = lane + i * 32; peaks[i] = peak > -INFINITY ? __exp2f(peaks[i] - peak) : 0.f;
      if (split < live_splits) norm = fmaf(peaks[i], p.partial_lse[p.heads * p.splits + first + split], norm);
    }
    norm = warp_sum(norm); const float inverse = norm > 0.f ? 1.f / norm : 0.f;
#pragma unroll
    for (int i = 0; i < kSingleRowMaxSplits / 32; ++i) weights[lane + i * 32] = peaks[i] * inverse;
  }
  __syncthreads();
  const auto* partial = reinterpret_cast<const float4*>(p.partial + int64_t(first) * kHeadDim + dim);
  float4 values[Loads]; float split_weights[Loads];
#pragma unroll
  for (int i = 0; i < Loads; ++i) {
    const int split = split_group + i * SplitGroups; split_weights[i] = split < live_splits ? weights[split] : 0.f;
    values[i] = split_weights[i] != 0.f ? partial[split * (kHeadDim / 4)] : float4{};
  }
  float4 sum{};
#pragma unroll
  for (int i = 0; i < Loads; ++i) {
    const float4 value = values[i]; const float weight = split_weights[i];
    sum.x = fmaf(weight, value.x, sum.x); sum.y = fmaf(weight, value.y, sum.y); sum.z = fmaf(weight, value.z, sum.z); sum.w = fmaf(weight, value.w, sum.w);
  }
#pragma unroll
  for (int delta = 16; delta >= Lanes; delta /= 2) {
    sum.x += __shfl_xor_sync(0xffffffffu, sum.x, delta); sum.y += __shfl_xor_sync(0xffffffffu, sum.y, delta); sum.z += __shfl_xor_sync(0xffffffffu, sum.z, delta); sum.w += __shfl_xor_sync(0xffffffffu, sum.w, delta);
  }
  if (warp != 0 && lane < Lanes) sums[warp - 1][feature_lane] = sum;
  __syncthreads();
  if (warp == 0 && lane < Lanes) {
#pragma unroll
    for (int i = 0; i < Warps - 1; ++i) { const float4 value = sums[i][feature_lane]; sum.x += value.x; sum.y += value.y; sum.z += value.z; sum.w += value.w; }
    const float sigmoid = p.gate_value(0, head); auto* out = reinterpret_cast<__ppu_bfloat162*>(p.out + head * p.oh + dim);
    out[0] = p.output_pair(sum.x, sum.y, sigmoid); out[1] = p.output_pair(sum.z, sum.w, sigmoid);
  }
}
void launch_single_row(Decode p, hggcStream_t stream) {
  const dim3 blocks(p.splits, p.heads / 8);
  packed_attention<8, kSingleRowTile, 1, false, true><<<blocks, 32, 0, stream>>>(p);
  if (p.splits > 1) { constexpr int tiles = kHeadDim / kSingleRowMergeWidth; merge_single_row<kSingleRowMergeThreads><<<p.heads * tiles, kSingleRowMergeThreads, 0, stream>>>(p); }
}
template <int Warps, bool Direct = false> void launch(Decode p, hggcStream_t stream) {
  constexpr int TG = Warps == 1 ? 4 : 2;
  const int work = Direct ? p.rows : p.rows * p.splits, blocks = (work + Warps - 1) / Warps;
  if (p.heads == 16) packed_attention<16, TG, Warps, Direct><<<blocks, 32 * Warps, 0, stream>>>(p);
  else packed_attention<8, TG, Warps, Direct><<<blocks, 32 * Warps, 0, stream>>>(p);
  if (!Direct && p.splits > 1) {
    const int output_heads = p.rows * p.heads;
    if constexpr (Warps > 1) merge_attention_warp<32 * kMergeWarps><<<output_heads / kMergeWarps, 32 * kMergeWarps, 0, stream>>>(p);
    else merge_attention<64><<<output_heads, 64, 0, stream>>>(p);
  }
}
void run(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& regions, const at::Tensor& table,
    const at::Tensor& positions, const at::Tensor& active, const at::Tensor& out,
    const std::optional<at::Tensor>& gate, int64_t query_len, bool clear_inactive, double scale) {
  const int rows = q.size(0), heads = q.size(1);
  if (!rows) return;
  const c10::cuda::CUDAGuard guard(q.device());
  int desired = rows >= kDirectRows ? 1 : rows <= 4 ? kMaxSplits : rows <= 32 ? 16 : rows <= 160 ? 12 : rows <= 320 ? 8 : 4;
  const int slots = regions.size(1), partitions = rows <= kSingleWarpRows ? slots : (slots + 1) / 2;
  const bool single = rows == 1;
  const int splits = single ? (slots + kSingleRowTile - 1) / kSingleRowTile : std::min(desired, partitions);
  TORCH_CHECK(!single || splits <= kSingleRowMaxSplits, "single-row KV tiles exceed capacity");
  const auto options = q.options().dtype(at::kFloat);
  const int64_t partial_rows = splits > 1 ? int64_t(rows) * heads * splits : 0;
  auto partial = at::empty({partial_rows, kHeadDim}, options), partial_lse = at::empty({(single ? 2 : 1) * partial_rows}, options);
  Decode p{}; p.kv = paged_kv(cache, table); p.q = reinterpret_cast<const __ppu_bfloat16*>(q.data_ptr());
  p.out = reinterpret_cast<__ppu_bfloat16*>(out.data_ptr()); p.gate = gate ? reinterpret_cast<const __ppu_bfloat16*>(gate->data_ptr()) : nullptr;
  p.regions = regions.data_ptr<int64_t>(); p.positions = positions.data_ptr<int64_t>(); p.active = active.data_ptr<bool>();
  p.partial = partial.data_ptr<float>(); p.partial_lse = partial_lse.data_ptr<float>();
  p.rows = rows; p.heads = heads; p.slots = slots; p.splits = splits; p.groups = cache.size(1); p.rows_per_request = p.groups * query_len;
  p.qs = q.stride(0); p.qh = q.stride(1); p.os = out.stride(0); p.oh = out.stride(1); p.rs = regions.stride(0);
  p.gs = gate ? gate->stride(0) : 0; p.gh = gate ? gate->stride(1) : 0; p.clear_inactive = clear_inactive || gate.has_value(); p.scale = scale;
  const auto stream = get_current_cuda_stream(q.get_device());
  if (single) launch_single_row(p, stream); else if (rows >= kDirectRows) launch<2, true>(p, stream);
  else if (rows <= kSingleWarpRows) launch<1>(p, stream); else launch<2>(p, stream);
  step5_runtime::check(hggcGetLastError());
}
}  // namespace step5_native_decode
