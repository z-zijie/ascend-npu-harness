#ifndef AXPY_SCALAR_AFFINE_TILING_H
#define AXPY_SCALAR_AFFINE_TILING_H
#include <cstdint>
struct axpy_scalar_affine_tiling_data {
    uint32_t totalLength;
    uint32_t tileNum;
};
#endif
