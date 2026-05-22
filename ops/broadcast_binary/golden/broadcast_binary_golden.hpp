#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace harness {
namespace golden {
namespace broadcast_binary {

// CPU golden: z[b,c,h,w] = x[b,c,h,w] * y[broadcast,c,broadcast,broadcast] + bias[c]
//
// Inputs:
//   x: [B, C, H, W] - primary input (NCHW contiguous)
//   y: [Yb, C, 1, 1] - broadcast weight, Yb is 1 or B
//   bias: [C] - additive bias vector, broadcast along C
//   z: [B, C, H, W] - output
//
// Parameters:
//   B, C, H, W: tensor dimensions
//   y_batch: 1 if y broadcast over batch, B if per-batch

void compute(const float* x,
             const float* y,
             const float* bias,
             float* z,
             int64_t B, int64_t C, int64_t H, int64_t W,
             int64_t y_batch);

}  // namespace broadcast_binary
}  // namespace golden
}  // namespace harness
