// AscendC kernel for broadcast_binary: z = x * y + bias
// Target arch: dav-2201
// x: [B, C, H, W] flat layout. y: [y_batch, C] broadcast. bias: [C] broadcast.
// Strategy: each (b,c) row of H*W elements is a work item; within a (b,c),
// y_val and bias_val are constant scalars — use AscendC::Muls then AscendC::Adds.

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "broadcast_binary_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelBroadcastBinary {
public:
    __aicore__ inline KernelBroadcastBinary() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR z,
                                 uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                                 uint32_t y_batch, uint32_t tileNum)
    {
        this->B       = B;
        this->C       = C;
        this->H       = H;
        this->W       = W;
        this->y_batch = y_batch;
        this->tileNum = tileNum;

        // Each (b,c) has H*W elements. Total work = B*C*(H*W).
        uint32_t totalElements = B * C * H * W;
        this->blockLength = totalElements / AscendC::GetBlockNum();
        this->tileLength  = this->blockLength / tileNum / BUFFER_NUM;
        if (this->tileLength == 0) { this->tileLength = 1; }

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, this->blockLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + offset, this->blockLength);

        // y and bias are small: stored separately as scalars per (b,c)
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), y_batch * C);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), C);

        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // Given a flat index into this block's portion, compute (b,c) and local hw offset
    __aicore__ inline void FlatToBC(uint32_t flatGlobal, uint32_t& b, uint32_t& c) const
    {
        uint32_t bc_idx = flatGlobal / (H * W);
        b = bc_idx / C;
        c = bc_idx % C;
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        // Determine (b,c) for the first element in this tile
        uint32_t flatGlobal = this->blockLength * AscendC::GetBlockIdx()
                              + static_cast<uint32_t>(progress) * this->tileLength;
        uint32_t b, c;
        FlatToBC(flatGlobal, b, c);

        // Load broadcast scalars for this (b,c) pair
        uint32_t y_idx = (y_batch == 1) ? 0 : b;
        float y_scalar = yGm.GetValue(y_idx * C + c);
        float bias_scalar = biasGm.GetValue(c);

        // z = x * y_scalar + bias_scalar  (fully vectorized)
        AscendC::Muls(zLocal, xLocal, y_scalar, this->tileLength);
        AscendC::Adds(zLocal, zLocal, bias_scalar, this->tileLength);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm, yGm, biasGm, zGm;
    uint32_t B = 0, C = 0, H = 0, W = 0, y_batch = 0;
    uint32_t blockLength = 0, tileNum = 0, tileLength = 0;
};

struct __tiling__ BroadcastBinaryTilingData {
    uint32_t B;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t y_batch;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void broadcast_binary_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BroadcastBinaryTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelBroadcastBinary op;
    op.Init(x, y, bias, z,
            tilingData.B, tilingData.C, tilingData.H, tilingData.W,
            tilingData.y_batch, tilingData.tileNum);
    op.Process();
}

#else
namespace {
const int broadcast_binary_kernel_stub = 0;
}
#endif  // HARNESS_HAS_ASCEND
