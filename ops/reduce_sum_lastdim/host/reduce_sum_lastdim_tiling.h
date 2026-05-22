#ifndef REDUCE_SUM_LASTDIM_TILING_H
#define REDUCE_SUM_LASTDIM_TILING_H
#include <cstdint>

struct reduce_sum_lastdim_tiling_data {
    uint32_t totalRows;
    uint32_t K;
    uint32_t tileNum;
};

#endif
