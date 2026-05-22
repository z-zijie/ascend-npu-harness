#ifndef BROADCAST_BINARY_TILING_H
#define BROADCAST_BINARY_TILING_H
#include <cstdint>

struct broadcast_binary_tiling_data {
    uint32_t B;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t y_batch;
    uint32_t tileNum;
};

#endif
