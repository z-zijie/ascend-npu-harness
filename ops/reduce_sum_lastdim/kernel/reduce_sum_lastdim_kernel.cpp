// AscendC kernel for reduce_sum_lastdim: sum over last dimension
// [B, M, K] -> [B, M]  (or 2D: [M, K] -> [M])
// Target arch: dav-2201.
// Uses AscendC::ReduceSum on tiles, accumulates in float32.

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "reduce_sum_lastdim_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelReduceSumLastdim {
public:
    __aicore__ inline KernelReduceSumLastdim() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out,
                                 uint32_t totalRows, uint32_t K,
                                 uint32_t tileNum)
    {
        ASSERT(tileNum != 0);
        this->totalRows  = totalRows;
        this->K          = K;
        this->rowsPerBlock = totalRows / AscendC::GetBlockNum();
        if (this->rowsPerBlock == 0) { this->rowsPerBlock = 1; }

        // Tiles slice along K dimension
        this->tileNum    = tileNum;
        this->tileLenK   = K / tileNum / BUFFER_NUM;
        if (this->tileLenK == 0) { this->tileLenK = 1; }

        // Global buffers for input (rows × K) and output (rows)
        uint32_t rowOffset = this->rowsPerBlock * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x)   + rowOffset * K, this->rowsPerBlock * K);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out) + rowOffset, this->rowsPerBlock);

        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLenK * sizeof(float));
        // Scratch buffer for reduction intermediate
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Outer loop: each row assigned to this block
        for (uint32_t r = 0; r < this->rowsPerBlock; r++) {
            float rowSum = 0.0f;

            // Inner loop: tile along K dimension
            int32_t loopCount = this->tileNum * BUFFER_NUM;
            for (int32_t i = 0; i < loopCount; i++) {
                CopyIn(i, r);
                rowSum += ReduceTile(i);
            }
            // Write scalar result for this row
            outGm.SetValue(r, rowSum);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        uint32_t srcOffset = row * K + progress * this->tileLenK;
        AscendC::DataCopy(xLocal, xGm[srcOffset], this->tileLenK);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline float ReduceTile(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        // Use AscendC::ReduceSum on the tile — returns scalar result
        float tileSum = AscendC::ReduceSum<float>(xLocal, nullptr, this->tileLenK);
        inQueueX.FreeTensor(xLocal);
        return tileSum;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpBuf;
    AscendC::GlobalTensor<float> xGm, outGm;
    uint32_t totalRows = 0, K = 0, rowsPerBlock = 0;
    uint32_t tileNum = 0, tileLenK = 0;
};

struct __tiling__ ReduceSumTilingData {
    uint32_t totalRows;
    uint32_t K;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void reduce_sum_lastdim_kernel(
    GM_ADDR x, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReduceSumTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelReduceSumLastdim op;
    op.Init(x, out, tilingData.totalRows, tilingData.K, tilingData.tileNum);
    op.Process();
}

#else
namespace {
const int reduce_sum_lastdim_kernel_stub = 0;
}
#endif  // HARNESS_HAS_ASCEND
