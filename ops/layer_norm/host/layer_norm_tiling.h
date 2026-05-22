#ifndef LAYER_NORM_TILING_H
#define LAYER_NORM_TILING_H
#include <cstdint>

struct layer_norm_tiling_data {
    uint32_t rows;
    uint32_t lastDim;
    uint32_t tileNum;
    float    eps;
};

#endif
