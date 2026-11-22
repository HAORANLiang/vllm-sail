// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/util/Exception.h>
#include "libtorch_stable/torch_utils.h"

namespace step5_runtime {

// CUDA-named Torch interfaces use different error and stream types in SDK 2.2.
// Reuse get_current_cuda_stream() from torch_utils.h for raw HGGC launches.
inline void check(hggcError_t error) {
  TORCH_CHECK(error == hggcSuccess, "Step-5 HGGC runtime error: ",
              hggcGetErrorString(error));
}

}  // namespace step5_runtime
