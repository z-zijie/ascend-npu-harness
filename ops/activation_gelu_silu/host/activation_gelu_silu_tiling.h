#ifndef ACTIVATION_GELU_SILU_TILING_H
#define ACTIVATION_GELU_SILU_TILING_H
#include <cstdint>

struct activation_gelu_silu_tiling_data {
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t actType;  // 0 = GELU, 1 = SiLU
};

#endif
