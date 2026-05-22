#ifndef RMS_NORM_ROPE_FUSED_TILING_H
#define RMS_NORM_ROPE_FUSED_TILING_H
#include <cstdint>

struct rms_norm_rope_fused_tiling_data {
    uint32_t B;
    uint32_t S;
    uint32_t H;
    uint32_t rotaryDim;
    uint32_t tileNum;
    float    eps;
};

#endif
