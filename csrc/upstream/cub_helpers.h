#pragma once

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #if CUB_VERSION >= 200800
    #include <hggc/std/functional>
using CubAddOp = hggc::std::plus<>;
using CubMaxOp = hggc::maximum<>;
  #else   // if CUB_VERSION < 200800
using CubAddOp = cub::Sum;
using CubMaxOp = cub::Max;
  #endif  // CUB_VERSION
#else
  #include <hipcub/hipcub.hpp>
namespace cub = hipcub;
using CubAddOp = hipcub::Sum;
using CubMaxOp = hipcub::Max;
#endif  // USE_ROCM
