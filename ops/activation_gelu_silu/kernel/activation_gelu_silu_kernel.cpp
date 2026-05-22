// AscendC kernel for activation_gelu_silu
// type=0: GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// type=1: SiLU: y = x / (1 + exp(-x))
// Target arch: dav-2201. Fully vectorized with AscendC vector ops.

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "activation_gelu_silu_tiling.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr float   kSqrt2OverPi = 0.7978845608f;
constexpr float   kGeluCoeff   = 0.044715f;

class KernelActivationGeluSilu {
public:
    __aicore__ inline KernelActivationGeluSilu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t totalLength, uint32_t tileNum, uint32_t actType)
    {
        ASSERT(tileNum != 0);
        this->tileNum  = tileNum;
        this->actType  = actType;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileLength  = this->blockLength / tileNum / BUFFER_NUM;
        if (this->tileLength == 0) { this->tileLength = 1; }

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + offset, this->blockLength);

        // UB buffers: one for input, two temporaries, one for output
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        // Extra buffers for intermediate results
        pipe.InitBuffer(tmpBuf1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            if (this->actType == 0) {
                ComputeGelu(i);
            } else {
                ComputeSilu(i);
            }
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    // GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    __aicore__ inline void ComputeGelu(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> xLocal  = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> t1      = tmpBuf1.AllocTensor<float>();
        AscendC::LocalTensor<float> t2      = tmpBuf2.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal  = outQueueY.AllocTensor<float>();

        uint32_t n = this->tileLength;

        // t1 = x^2
        AscendC::Mul(t1, xLocal, xLocal, n);
        // t1 = x^3 = t1 * x
        AscendC::Mul(t1, t1, xLocal, n);
        // t1 = 0.044715 * x^3
        AscendC::Muls(t1, t1, kGeluCoeff, n);
        // t1 = x + 0.044715 * x^3
        AscendC::Add(t1, xLocal, t1, n);
        // t1 = sqrt(2/pi) * (x + 0.044715 * x^3)
        AscendC::Muls(t1, t1, kSqrt2OverPi, n);
        // t2 = tanh(t1)
        AscendC::Tanh(t2, t1, n);
        // t2 = 1 + tanh(t1)
        AscendC::Adds(t2, t2, 1.0f, n);
        // y = x * (1 + tanh(...))
        AscendC::Mul(yLocal, xLocal, t2, n);
        // y = 0.5 * y
        AscendC::Muls(yLocal, yLocal, 0.5f, n);

        outQueueY.EnQue<float>(yLocal);
        tmpBuf1.FreeTensor(t1);
        tmpBuf2.FreeTensor(t2);
        inQueueX.FreeTensor(xLocal);
    }

    // SiLU: y = x / (1 + exp(-x))
    __aicore__ inline void ComputeSilu(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> t1     = tmpBuf1.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        uint32_t n = this->tileLength;

        // t1 = -x
        AscendC::Neg(t1, xLocal, n);
        // t1 = exp(-x)
        AscendC::Exp(t1, t1, n);
        // t1 = 1 + exp(-x)
        AscendC::Adds(t1, t1, 1.0f, n);
        // y = x / (1 + exp(-x))
        AscendC::Div(yLocal, xLocal, t1, n);

        outQueueY.EnQue<float>(yLocal);
        tmpBuf1.FreeTensor(t1);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t actType = 0;
    uint32_t blockLength = 0, tileNum = 0, tileLength = 0;
};

struct __tiling__ ActivationTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t actType;   // 0=GELU, 1=SiLU
};

extern "C" __global__ __aicore__ void activation_gelu_silu_kernel(
    GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ActivationTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelActivationGeluSilu op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum, tilingData.actType);
    op.Process();
}

#else
namespace {
const int activation_gelu_silu_kernel_stub = 0;
}
#endif  // HARNESS_HAS_ASCEND
