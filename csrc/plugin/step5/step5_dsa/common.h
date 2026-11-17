// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <initializer_list>
#include <limits>

#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "../step5_runtime.h"

namespace step5_dsa {

inline void check_devices(const at::Tensor& input,
                          std::initializer_list<at::Tensor> tensors) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == input.device(),
                "all tensors must be on the same CUDA device");
  }
}

inline void check_output(const at::Tensor& output,
                         std::initializer_list<at::Tensor> inputs) {
  at::assert_no_internal_overlap(output);
  for (const auto& input : inputs) {
    TORCH_CHECK(!output.is_alias_of(input),
                "output must not alias an input or workspace");
  }
}

inline void check_int32(std::initializer_list<int64_t> values) {
  for (const int64_t value : values) {
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
                "DSA geometry exceeds int32 indexing capacity");
  }
}

inline void check_scale(double scale) {
  TORCH_CHECK(std::isfinite(scale) && static_cast<float>(scale) > 0 &&
                  scale <= std::numeric_limits<float>::max(),
              "scaling must be finite and positive in float32");
}

}  // namespace step5_dsa
