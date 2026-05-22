#ifndef MATMUL_BIAS_TILING_H
#define MATMUL_BIAS_TILING_H
#include <cstdint>

struct matmul_bias_tiling_data {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t tileM;
    uint32_t tileK;
    uint32_t tileN;
};

#endif
