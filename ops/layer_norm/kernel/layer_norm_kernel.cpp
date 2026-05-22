// AscendC kernel for Layer Normalization
// y = (x - mean) / sqrt(var + eps) * gamma + beta
// Target arch: dav-2201.
//
// Multi-pass algorithm:
//   Pass 1 (Stats):  For each row, tile over last_dim, reduce to get mean and var.
//                    mean = ReduceSum(x) / K
//                    var  = ReduceSum((x - mean)^2) / K
//   Pass 2 (Normalize): For each row, tile over last_dim, compute:
//                    y = (x - mean) / sqrt(var + eps) * gamma + beta
// All element-wise ops are vectorized (AscendC::Add, AscendC::Mul, AscendC::Div, etc.)

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "layer_norm_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLayerNorm {
public:
    __aicore__ inline KernelLayerNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                 uint32_t rows, uint32_t lastDim, uint32_t tileNum, float eps)
    {
        ASSERT(tileNum != 0);
        this->rows       = rows;
        this->lastDim    = lastDim;
        this->eps        = eps;
        this->tileNum    = tileNum;
        this->rowsPerBlock = rows / AscendC::GetBlockNum();
        if (this->rowsPerBlock == 0) { this->rowsPerBlock = 1; }

        this->tileLength = lastDim / tileNum / BUFFER_NUM;
        if (this->tileLength == 0) { this->tileLength = 1; }

        uint32_t rowOffset = this->rowsPerBlock * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x)     + rowOffset * lastDim,
                            this->rowsPerBlock * lastDim);
        gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma), lastDim);
        betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta),  lastDim);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y)     + rowOffset * lastDim,
                            this->rowsPerBlock * lastDim);

        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1,   BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2,   BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(gammaBuf,  BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(betaBuf,   BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Process each row assigned to this block
        for (uint32_t r = 0; r < this->rowsPerBlock; r++) {
            // ---- Pass 1: compute mean and var for this row ----
            float sum_x   = 0.0f;
            float sum_x2  = 0.0f;
            int32_t loopCount = this->tileNum * BUFFER_NUM;
            for (int32_t i = 0; i < loopCount; i++) {
                CopyInX(i, r);
                float s  = ReduceTileSum(i);
                float s2 = ReduceTileSumSq(i);
                sum_x  += s;
                sum_x2 += s2;
            }
            float mean = sum_x / static_cast<float>(this->lastDim);
            float var  = (sum_x2 / static_cast<float>(this->lastDim)) - (mean * mean);
            float invStd = 1.0f / AscendC::Sqrt(var + this->eps);

            // ---- Pass 2: normalize ----
            for (int32_t i = 0; i < loopCount; i++) {
                CopyInXPass2(i, r);
                NormalizeTile(i, mean, invStd, r);
                CopyOut(i, r);
            }
        }
    }

private:
    // ---- Pass 1 helpers ----
    __aicore__ inline void CopyInX(int32_t progress, uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        uint32_t offset = row * this->lastDim + progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline float ReduceTileSum(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        // Copy to tmp for squared computation later (reload in ReduceTileSumSq)
        AscendC::LocalTensor<float> xCopy  = tmpBuf2.AllocTensor<float>();
        AscendC::DataCopy(xCopy, xLocal, this->tileLength);

        float s = AscendC::ReduceSum<float>(xLocal, nullptr, this->tileLength);
        inQueueX.FreeTensor(xLocal);
        // Enque xCopy so ReduceTileSumSq can dequeue it
        tmpBuf2.EnQue<float>(xCopy);
        return s;
    }

    __aicore__ inline float ReduceTileSumSq(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> xCopy = tmpBuf2.DeQue<float>();
        AscendC::LocalTensor<float> xSq   = tmpBuf1.AllocTensor<float>();
        // xSq = xCopy^2
        AscendC::Mul(xSq, xCopy, xCopy, this->tileLength);
        float s2 = AscendC::ReduceSum<float>(xSq, nullptr, this->tileLength);
        tmpBuf2.FreeTensor(xCopy);
        tmpBuf1.FreeTensor(xSq);
        return s2;
    }

    // ---- Pass 2 helpers ----
    __aicore__ inline void CopyInXPass2(int32_t progress, uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal  = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> gLocal  = gammaBuf.AllocTensor<float>();
        AscendC::LocalTensor<float> bLocal  = betaBuf.AllocTensor<float>();
        uint32_t offset = row * this->lastDim + progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset],  this->tileLength);
        AscendC::DataCopy(gLocal, gammaGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(bLocal, betaGm[progress * this->tileLength],  this->tileLength);
        inQueueX.EnQue(xLocal);
        gammaBuf.EnQue(gLocal);
        betaBuf.EnQue(bLocal);
    }

    __aicore__ inline void NormalizeTile(int32_t progress, float mean, float invStd, uint32_t row)
    {
        (void)progress; (void)row;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> gLocal = gammaBuf.DeQue<float>();
        AscendC::LocalTensor<float> bLocal = betaBuf.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        uint32_t n = this->tileLength;

        // norm = (x - mean) * invStd
        AscendC::Sub(yLocal, xLocal, mean, n);
        AscendC::Muls(yLocal, yLocal, invStd, n);
        // y = norm * gamma + beta
        AscendC::Mul(yLocal, yLocal, gLocal, n);
        AscendC::Add(yLocal, yLocal, bLocal, n);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        gammaBuf.FreeTensor(gLocal);
        betaBuf.FreeTensor(bLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t row)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        uint32_t offset = row * this->lastDim + progress * this->tileLength;
        AscendC::DataCopy(yGm[offset], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpBuf1, tmpBuf2;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> gammaBuf, betaBuf;
    AscendC::GlobalTensor<float> xGm, gammaGm, betaGm, yGm;
    uint32_t rows = 0, lastDim = 0, rowsPerBlock = 0;
    uint32_t tileNum = 0, tileLength = 0;
    float eps = 1e-5f;
};

struct __tiling__ LayerNormTilingData {
    uint32_t rows;
    uint32_t lastDim;
    uint32_t tileNum;
    float    eps;
};

extern "C" __global__ __aicore__ void layer_norm_kernel(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LayerNormTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLayerNorm op;
    op.Init(x, gamma, beta, y,
            tilingData.rows, tilingData.lastDim,
            tilingData.tileNum, tilingData.eps);
    op.Process();
}

#else
namespace harness { namespace layer_norm {
const int layer_norm_kernel_stub = 0;
}}  // namespace harness::layer_norm
#endif  // HARNESS_HAS_ASCEND
