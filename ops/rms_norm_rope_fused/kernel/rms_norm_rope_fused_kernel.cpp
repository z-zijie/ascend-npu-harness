// AscendC kernel for fused RMSNorm + RoPE
// Target arch: dav-2201.
//
// Algorithm (per (B,S) row):
//   Step 1 — RMSNorm over H:
//     rms = sqrt(mean(x^2) + eps)
//     y[i] = x[i] / rms * weight[i]
//   Step 2 — RoPE on first rotary_dim elements:
//     For half-pair j in [0, rotary_dim/2):
//       idx = j*2
//       y[idx]   = x[idx]*cos[j] - x[idx+1]*sin[j]
//       y[idx+1] = x[idx+1]*cos[j] + x[idx]*sin[j]
//   Step 3 — Copy rope portion back, keep rest as-is

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "rms_norm_rope_fused_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelRMSNormRoPE {
public:
    __aicore__ inline KernelRMSNormRoPE() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR cos, GM_ADDR sin,
                                 GM_ADDR out,
                                 uint32_t B, uint32_t S, uint32_t H,
                                 uint32_t rotaryDim, uint32_t tileNum, float eps)
    {
        ASSERT(tileNum != 0);
        this->B = B; this->S = S; this->H = H;
        this->rotaryDim = rotaryDim;
        this->eps       = eps;
        this->tileNum   = tileNum;

        // Total rows = B * S. Distribute across cores.
        uint32_t totalRows = B * S;
        this->rowsPerBlock = totalRows / AscendC::GetBlockNum();
        if (this->rowsPerBlock == 0) { this->rowsPerBlock = 1; }

        this->tileLength = H / tileNum / BUFFER_NUM;
        if (this->tileLength == 0) { this->tileLength = 1; }

        uint32_t rowOffset = this->rowsPerBlock * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + rowOffset * H,
                            this->rowsPerBlock * H);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight), H);
        cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cos), rotaryDim / 2);
        sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sin), rotaryDim / 2);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out) + rowOffset * H,
                              this->rowsPerBlock * H);

        pipe.InitBuffer(inQueueX,   BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueW,   BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1,    BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2,    BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY,  BUFFER_NUM, this->tileLength * sizeof(float));
        // Buffers for cos/sin (small, one per half-pair)
        pipe.InitBuffer(cosBuf, 1, (rotaryDim / 2) * sizeof(float));
        pipe.InitBuffer(sinBuf, 1, (rotaryDim / 2) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Preload cos and sin once (they're shared across all rows)
        {
            AscendC::LocalTensor<float> cLocal = cosBuf.AllocTensor<float>();
            AscendC::LocalTensor<float> sLocal = sinBuf.AllocTensor<float>();
            AscendC::DataCopy(cLocal, cosGm, rotaryDim / 2);
            AscendC::DataCopy(sLocal, sinGm, rotaryDim / 2);
            cosBuf.EnQue(cLocal);
            sinBuf.EnQue(sLocal);
        }

        // Process each (B,S) row assigned to this block
        for (uint32_t r = 0; r < this->rowsPerBlock; r++) {
            // ---- Step 1: Compute RMS over full H ----
            float sum_x2 = 0.0f;
            int32_t loopCount = this->tileNum * BUFFER_NUM;
            for (int32_t i = 0; i < loopCount; i++) {
                CopyInX(i, r);
                sum_x2 += ReduceSquare(i);
            }
            float rms = AscendC::Sqrt(sum_x2 / static_cast<float>(H) + eps);
            float invRms = 1.0f / rms;

            // ---- Step 2: RMSNorm + RoPE on first rotary_dim ----
            // Two sub-steps handled by checking whether the tile overlaps with rotary_dim
            for (int32_t i = 0; i < loopCount; i++) {
                CopyInXNormW(i, r);
                uint32_t tileStart = i * this->tileLength;
                uint32_t tileEnd   = tileStart + this->tileLength;

                if (tileEnd <= rotaryDim) {
                    // Entire tile is within rotary_dim: do RMSNorm then RoPE
                    RMSNormTile(i, invRms);
                    ApplyRoPE(i);
                } else if (tileStart >= rotaryDim) {
                    // Entire tile is beyond rotary_dim: just RMSNorm
                    RMSNormTile(i, invRms);
                } else {
                    // Tile straddles rotary_dim boundary — split handling
                    // For simplicity: RMSNorm entire tile, RoPE on prefix within rotary_dim
                    RMSNormTile(i, invRms);
                    ApplyRoPE(i);  // RoPE only touches indices < rotaryDim internally
                }
                CopyOut(i, r);
            }
        }
    }

private:
    // ---- Step 1: Load x tile and reduce sum of squares ----
    __aicore__ inline void CopyInX(int32_t progress, uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        uint32_t offset = row * H + progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline float ReduceSquare(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> x2     = tmpBuf1.AllocTensor<float>();
        AscendC::Mul(x2, xLocal, xLocal, this->tileLength);
        float s2 = AscendC::ReduceSum<float>(x2, nullptr, this->tileLength);
        inQueueX.FreeTensor(xLocal);
        tmpBuf1.FreeTensor(x2);
        return s2;
    }

    // ---- Step 2: Load x and weight tile for normalization ----
    __aicore__ inline void CopyInXNormW(int32_t progress, uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
        uint32_t offset = row * H + progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], this->tileLength);
        AscendC::DataCopy(wLocal, weightGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueW.EnQue(wLocal);
    }

    // RMSNorm: y = x / rms * weight  (using invRms)
    __aicore__ inline void RMSNormTile(int32_t progress, float invRms)
    {
        (void)progress;
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> wLocal = inQueueW.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        uint32_t n = this->tileLength;
        // y = x * invRms
        AscendC::Muls(yLocal, xLocal, invRms, n);
        // y = y * weight
        AscendC::Mul(yLocal, yLocal, wLocal, n);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueW.FreeTensor(wLocal);
    }

    // RoPE: pairwise rotation using cos/sin on first rotary_dim elements
    __aicore__ inline void ApplyRoPE(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> yLocal  = outQueueY.DeQue<float>();
        AscendC::LocalTensor<float> cosVals = cosBuf.DeQue<float>();
        AscendC::LocalTensor<float> sinVals = sinBuf.DeQue<float>();
        AscendC::LocalTensor<float> tmp     = tmpBuf1.AllocTensor<float>();

        // Rotate pairs: for j in [0, min(rotaryDim/2, tileLength/2))
        uint32_t numPairs = this->tileLength / 2;
        if (numPairs > rotaryDim / 2) { numPairs = rotaryDim / 2; }

        for (uint32_t j = 0; j < numPairs; j++) {
            float y_even = yLocal.GetValue(2 * j);
            float y_odd  = yLocal.GetValue(2 * j + 1);
            float c      = cosVals.GetValue(j);
            float s      = sinVals.GetValue(j);
            // RoPE: [y_even_new, y_odd_new] = [y_even*c - y_odd*s, y_odd*c + y_even*s]
            float new_even = y_even * c - y_odd * s;
            float new_odd  = y_odd  * c + y_even * s;
            yLocal.SetValue(2 * j,     new_even);
            yLocal.SetValue(2 * j + 1, new_odd);
        }

        outQueueY.EnQue<float>(yLocal);
        tmpBuf1.FreeTensor(tmp);
        cosBuf.EnQue(cosVals);
        sinBuf.EnQue(sinVals);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t row)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        uint32_t offset = row * H + progress * this->tileLength;
        AscendC::DataCopy(outGm[offset], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX, inQueueW;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpBuf1, tmpBuf2;
    AscendC::TQue<AscendC::TPosition::VECIN,  1> cosBuf, sinBuf;
    AscendC::GlobalTensor<float> xGm, weightGm, cosGm, sinGm, outGm;
    uint32_t B = 0, S = 0, H = 0, rotaryDim = 0;
    uint32_t rowsPerBlock = 0, tileNum = 0, tileLength = 0;
    float eps = 1e-5f;
};

struct __tiling__ RMSNormRopeTilingData {
    uint32_t B;
    uint32_t S;
    uint32_t H;
    uint32_t rotaryDim;
    uint32_t tileNum;
    float    eps;
};

extern "C" __global__ __aicore__ void rms_norm_rope_fused_kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR cos, GM_ADDR sin, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RMSNormRopeTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelRMSNormRoPE op;
    op.Init(x, weight, cos, sin, out,
            tilingData.B, tilingData.S, tilingData.H,
            tilingData.rotaryDim, tilingData.tileNum, tilingData.eps);
    op.Process();
}

#else
namespace harness { namespace rms_norm_rope_fused {
const int rms_norm_rope_fused_kernel_stub = 0;
}}  // namespace harness::rms_norm_rope_fused
#endif  // HARNESS_HAS_ASCEND
