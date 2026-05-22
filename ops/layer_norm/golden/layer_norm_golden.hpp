#pragma once
#include <cstdint>
#include <vector>

namespace harness {
namespace layer_norm {

// Layer Normalization over the last dimension.
//
// Input shapes:
//   2D: x=[M, K], gamma=[K], beta=[K]
//   3D: x=[B, S, H], gamma=[H], beta=[H]
//
// mean = reduce_mean(x over last dim)
// var  = reduce_mean((x - mean)^2 over last dim)
// y    = (x - mean) / sqrt(var + eps) * gamma + beta
//
// For both 2D and 3D input, the computation is identical:
// normalize along the last axis.
//
// Template parameter T: float (always accumulate in float32)

void layer_norm_golden(const float* __restrict__ x,
                       const float* __restrict__ gamma,
                       const float* __restrict__ beta,
                       float* __restrict__ y,
                       int64_t rows,
                       int64_t last_dim,
                       float eps);

}  // namespace layer_norm
}  // namespace harness
