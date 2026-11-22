// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>
#include <climits>
#include <initializer_list>
#include <optional>
#include <cstring>
#include <cstdint>
namespace step5_native_checks {
inline void tensor(const at::Tensor& t, const at::Tensor& q, at::ScalarType dtype, int rank) {
  TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.layout() == at::kStrided &&
              t.scalar_type() == dtype && t.dim() == rank, "native DSA: unsupported tensor device/dtype/rank");
  int64_t span = 1;
  for (int axis = 0; axis < rank; ++axis) {
    const int64_t n = t.size(axis), s = t.stride(axis);
    TORCH_CHECK(n <= INT_MAX && s >= 0 && s <= INT64_MAX / 2 && (n <= 1 || s > 0), "native DSA: geometry/stride out of range");
    if (n > 1) {
      TORCH_CHECK(s <= (INT64_MAX / 8 - span) / (n - 1), "native DSA: address computation overflow");
      span += (n - 1) * s;
    }
  }
}
inline void no_alias(const at::Tensor& out, std::initializer_list<at::Tensor> inputs) {
  for (const auto& t : inputs) TORCH_CHECK(!out.is_alias_of(t), "native DSA: output aliases input");
}
inline void common(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& regions,
    const at::Tensor& table, const at::Tensor& pos, const at::Tensor& out, double scale) {
  tensor(q, q, at::kBFloat16, 3); tensor(out, q, at::kBFloat16, 3);
  tensor(cache, q, at::kBFloat16, 4); tensor(regions, q, at::kLong, 2);
  tensor(table, q, at::kInt, 2); tensor(pos, q, at::kLong, 1);
  const auto* prop = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(prop->major == 8 && prop->minor == 9 && std::strstr(prop->name, "890P"), "native DSA: only PPU 1.5/890P is supported");
  TORCH_CHECK(q.size(2) == 192 && (q.size(1) == 8 || q.size(1) == 16) && out.sizes() == q.sizes(), "native DSA: unsupported head/output geometry");
  TORCH_CHECK(cache.size(0) > 0 && cache.size(1) > 0 && cache.size(3) == 384 &&
    (cache.size(2) == 16 || cache.size(2) == 32 || cache.size(2) == 64 || cache.size(2) == 128), "native DSA: unsupported cache geometry");
  TORCH_CHECK(regions.size(0) == q.size(0) && regions.size(1) > 0 && regions.size(1) <= 513 &&
    regions.stride(1) == 1 && pos.numel() == q.size(0) && pos.is_contiguous(), "native DSA: unsupported selection/position");
  TORCH_CHECK(table.stride(1) == 1 && table.size(0) > 0 && table.size(1) > 0 &&
    table.size(1) * cache.size(2) <= INT_MAX - 8 && table.stride(0) >= table.size(1), "native DSA: unsupported block table");
  TORCH_CHECK(cache.stride(3) == 1 && reinterpret_cast<uintptr_t>(cache.data_ptr()) % 128 == 0,
    "native DSA: unsupported cache alignment");
  for (int i = 0; i < 3; ++i) TORCH_CHECK(cache.stride(i) % 64 == 0, "native DSA: unsupported cache stride");
  // A single group's stride is not used for addressing; matches the Python
  // LBNHC/BLNHC checks.
  TORCH_CHECK(cache.stride(2) >= 384 &&
    (cache.size(1) == 1 || cache.stride(1) >= cache.size(2) * cache.stride(2)) &&
    cache.stride(0) >= (cache.size(1) - 1) * cache.stride(1) + cache.size(2) * cache.stride(2), "native DSA: unsupported cache overlap");
  // Tags encode only a 28-bit physical region; the address span is guarded by
  // the int64 overflow checks in tensor().
  TORCH_CHECK(cache.size(0) * cache.size(2) / 8 < (int64_t(1) << 28), "native DSA: offset encoding capacity exceeded (physical region)");
  for (const auto& t : {q, out}) TORCH_CHECK(t.stride(2) == 1 && t.stride(0) % 2 == 0 &&
    t.stride(1) % 2 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0, "native DSA: unsupported query/output alignment");
  TORCH_CHECK(!out.numel() || (out.stride(1) >= 192 && out.stride(0) >= (q.size(1) - 1) * out.stride(1) + 192), "native DSA: unsupported output overlap");
  TORCH_CHECK(q.size(0) * q.size(1) * 32 <= INT_MAX && std::isfinite(scale) && float(scale) > 0 && scale <= 3.4028234663852886e38, "native DSA: unsupported launch geometry/scale");
  no_alias(out, {q, cache, regions, table, pos});
}
inline void prefill(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& regions,
    const at::Tensor& table, const at::Tensor& owners, const at::Tensor& pos,
    const at::Tensor& tiles, const at::Tensor& out, int64_t logical_regions, double scale, int64_t query_tile) {
  common(q, cache, regions, table, pos, out, scale);
  tensor(owners, q, at::kLong, 1); tensor(tiles, q, at::kInt, 2);
  TORCH_CHECK(cache.size(1) == 1 && q.stride(1) == 192 && q.stride(0) % 64 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 128 == 0,
    "native prefill: unsupported group/query");
  TORCH_CHECK(owners.numel() == q.size(0) && owners.is_contiguous() && tiles.size(0) == 2 &&
    tiles.is_contiguous() && (q.size(0) == 0 || tiles.size(1) > 0) && tiles.size(1) <= q.size(0), "native prefill: unsupported owner/tile");
  TORCH_CHECK(logical_regions > 0 && logical_regions <= INT_MAX / 8 &&
    (query_tile == 0 || query_tile == 1 || query_tile == 2 || query_tile == 4 || query_tile == 8), "native prefill: unsupported capacity/tile");
  no_alias(out, {owners, tiles});
}
inline void decode(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& regions,
    const at::Tensor& table, const at::Tensor& pos, const at::Tensor& active, const at::Tensor& out,
    const std::optional<at::Tensor>& gate, int64_t query_len, double scale) {
  common(q, cache, regions, table, pos, out, scale); tensor(active, q, at::kBool, 1);
  TORCH_CHECK(query_len > 0 && query_len <= INT_MAX / cache.size(1), "native decode: unsupported query_len");
  const int64_t per = query_len * cache.size(1);
  TORCH_CHECK(q.size(0) % per == 0 && table.size(0) >= q.size(0) / per &&
    active.numel() == q.size(0) && active.is_contiguous(), "native decode: unsupported request layout");
  no_alias(out, {active});
  if (gate) {
    tensor(*gate, q, at::kBFloat16, 2);
    TORCH_CHECK(gate->size(0) == q.size(0) / cache.size(1) && gate->size(1) == cache.size(1) * q.size(1), "native decode: unsupported gate shape");
    no_alias(out, {*gate});
  }
}
}  // namespace step5_native_checks
