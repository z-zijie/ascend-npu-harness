#ifndef FLASH_ATTENTION_FWD_FUSED_TILING_H
#define FLASH_ATTENTION_FWD_FUSED_TILING_H
#include <cstdint>

struct flash_attention_fwd_fused_tiling_data {
    uint32_t B;
    uint32_t H;
    uint32_t S_q;
    uint32_t S_k;
    uint32_t D;
    uint32_t D_v;
    float    scale;
    uint32_t maskType;  // 0=none, 1=causal
};

#endif
