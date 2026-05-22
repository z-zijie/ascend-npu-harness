#ifndef TRANSPOSE_2D_4D_TILING_H
#define TRANSPOSE_2D_4D_TILING_H
#include <cstdint>

struct transpose_2d_4d_tiling_data {
    uint32_t mode;    // 0=2D transpose, 1=NCHW->NHWC, 2=NHWC->NCHW
    uint32_t M;       // for 2D: M, for 4D: N
    uint32_t N_dim;   // for 2D: N, for 4D: C
    uint32_t H;       // for 4D only
    uint32_t W;       // for 4D only
    uint32_t tileNum;
};

#endif
