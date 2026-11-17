// SPDX-License-Identifier: Apache-2.0
// Private op registration and pre-launch safety checks; the standalone
// experimental SO is not loaded.
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>
#include <limits>

#define DECLARE_SELECTOR(NS) \
namespace NS { \
void step4_dsa_radix_select( \
    torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, \
    torch::Tensor&, torch::Tensor&, torch::Tensor&, \
    int64_t, int64_t, int64_t, bool); \
}
DECLARE_SELECTOR(step4_dsa_w256)
DECLARE_SELECTOR(step4_dsa_w1024)
#undef DECLARE_SELECTOR

namespace {
void step5_selector_v2_dispatch(torch::Tensor const& scores, torch::Tensor const& positions,
            torch::Tensor const& table, torch::Tensor& workspace,
            torch::Tensor& regions, torch::Tensor& starts,
            int64_t region_block, int64_t page_size, int64_t topk,
            bool packed, int64_t threads) {
  TORCH_CHECK(scores.is_cuda(), "scores must be on PPU/CUDA");
  for (auto const& tensor : {positions, table, workspace, regions, starts}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == scores.device(),
                "selector tensors must share a device");
  }
  TORCH_CHECK(scores.dim() == 3, "scores must have rank 3");
  constexpr int64_t max_int = std::numeric_limits<int>::max();
  TORCH_CHECK(scores.numel() <= max_int && workspace.numel() <= max_int &&
                  regions.numel() <= max_int && starts.numel() <= max_int &&
                  table.numel() <= max_int && region_block > 0 &&
                  region_block <= max_int && page_size > 0 && page_size <= max_int &&
                  scores.size(2) <= max_int / region_block &&
                  scores.size(2) <= max_int - 16384,
              "selector geometry exceeds int32 indexing");
  for (auto const& output : {workspace, regions, starts}) {
    at::assert_no_internal_overlap(output);
    for (auto const& input : {scores, positions, table}) {
      at::assert_no_overlap(output, input);
    }
  }
  at::assert_no_overlap(workspace, regions);
  at::assert_no_overlap(workspace, starts);
  at::assert_no_overlap(regions, starts);
  TORCH_CHECK(threads == 0 || threads == 256 || threads == 1024,
              "threads must be 0, 256 or 1024");
  c10::cuda::CUDAGuard guard(scores.device());
  // An empty batch still validates every tensor shape but never launches a
  // zero-sized grid.
  TORCH_CHECK(scores.size(0) > 0 && scores.size(1) > 0,
              "use the Python wrapper for empty rows");
  bool use256 = threads == 256 ||
      (threads == 0 && scores.size(0) * scores.size(1) >= 1000 && scores.size(2) <= 16384);
  auto launch = use256 ? step4_dsa_w256::step4_dsa_radix_select
                       : step4_dsa_w1024::step4_dsa_radix_select;
  launch(scores, positions, table, workspace, regions, starts,
         region_block, page_size, topk, packed);
}
}

TORCH_LIBRARY_FRAGMENT(_ppu_C, m) {
  m.def("step5_dsa_selector_v2(Tensor scores, Tensor positions, Tensor table, "
        "Tensor(a!) workspace, Tensor(b!) regions, Tensor(c!) starts, "
        "int region_block, int page_size, int topk, bool packed, int threads=0) -> ()");
}
TORCH_LIBRARY_IMPL(_ppu_C, CUDA, m) {
  m.impl("step5_dsa_selector_v2", &step5_selector_v2_dispatch);
}
