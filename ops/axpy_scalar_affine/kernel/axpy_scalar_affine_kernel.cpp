#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "axpy_scalar_affine_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelAxpyScalarAffine {
public:
    __aicore__ inline KernelAxpyScalarAffine() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float alpha, float beta,
                                 uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(tileNum != 0);
        this->alpha = alpha; this->beta = beta;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + offset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) { CopyIn(i); Compute(i); CopyOut(i); }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        (void)progress;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Muls(yLocal, xLocal, this->alpha, this->tileLength);   // y = alpha * x
        AscendC::Adds(yLocal, yLocal, this->beta, this->tileLength);    // y = y + beta
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm, yGm;
    float alpha = 1.0f, beta = 0.0f;
    uint32_t blockLength = 0, tileNum = 0, tileLength = 0;
};

struct axpy_scalar_affine_tiling_data {
    uint32_t totalLength;
    uint32_t tileNum;
    float alpha;
    float beta;
};

extern "C" __global__ __aicore__
void axpy_scalar_affine_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(axpy_scalar_affine_tiling_data);
    GET_TILING_DATA(tilingData, tiling);
    KernelAxpyScalarAffine op;
    op.Init(x, y, tilingData.alpha, tilingData.beta, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
#else
namespace { const int axpy_scalar_affine_kernel_stub = 0; }
#endif
