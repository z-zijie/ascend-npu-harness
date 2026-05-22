// =============================================================================
// AscendC Kernel: Flash Attention Forward (Fused)
// Target: dav-2201
//
// Implements tile-based flash attention with online softmax.
// Uses AscendC::MatMul (Cube Core) for Q@K^T and P@V,
// and AscendC vector ops (Exp, Sub, Mul, Add, etc.) for softmax.
//
// Algorithm (per head, per batch, per Q tile):
//   Init:  m = -inf, l = 0, acc = 0
//   For each KV tile:
//     S = Q @ K^T * scale
//     Apply causal mask if needed
//     m_new = max(m, row_max(S))
//     P = exp(S - m_new)
//     l_new = l * exp(m - m_new) + row_sum(P)
//     acc = acc * exp(m - m_new) + P @ V
//     m = m_new, l = l_new
//   O = acc / l
// =============================================================================

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "flash_attention_fwd_fused_tiling.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_Q    = 32;   // query tile size
constexpr int32_t BLOCK_K    = 64;   // key tile size
constexpr float   NEG_INF    = -1.0e30f;

class KernelFlashAttention {
public:
    __aicore__ inline KernelFlashAttention() {}
    __aicore__ inline void Init(GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR O,
                                 uint32_t B, uint32_t H,
                                 uint32_t S_q, uint32_t S_k,
                                 uint32_t D, uint32_t D_v,
                                 float scale, uint32_t maskType)
    {
        this->B     = B;
        this->H     = H;
        this->S_q   = S_q;
        this->S_k   = S_k;
        this->D     = D;
        this->D_v   = D_v;
        this->scale = scale;
        this->maskType = maskType;  // 0=none, 1=causal

        // Total Q tiles per batch-head: each B*H pair has (S_q/BLOCK_Q) Q tiles
        uint32_t numQTiles = (S_q + BLOCK_Q - 1) / BLOCK_Q;
        uint32_t totalWork = B * H * numQTiles;
        this->workPerBlock = totalWork / AscendC::GetBlockNum();
        if (this->workPerBlock == 0) { this->workPerBlock = 1; }

        // Global tensor wrappers
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Q), B * H * S_q * D);
        kGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(K), B * H * S_k * D);
        vGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(V), B * H * S_k * D_v);
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(O), B * H * S_q * D_v);

        // UB buffer allocation
        // Q tile: BLOCK_Q * D
        pipe.InitBuffer(qBuf,    BUFFER_NUM, BLOCK_Q * D     * sizeof(float));
        // K tile: BLOCK_K * D
        pipe.InitBuffer(kBuf,    BUFFER_NUM, BLOCK_K * D     * sizeof(float));
        // V tile: BLOCK_K * D_v
        pipe.InitBuffer(vBuf,    BUFFER_NUM, BLOCK_K * D_v   * sizeof(float));
        // Score tile S: BLOCK_Q * BLOCK_K
        pipe.InitBuffer(sBuf,    BUFFER_NUM, BLOCK_Q * BLOCK_K * sizeof(float));
        // P tile (exp scores): BLOCK_Q * BLOCK_K
        pipe.InitBuffer(pBuf,    BUFFER_NUM, BLOCK_Q * BLOCK_K * sizeof(float));
        // Accumulator O: BLOCK_Q * D_v
        pipe.InitBuffer(accBuf,  BUFFER_NUM, BLOCK_Q * D_v   * sizeof(float));
        // PV intermediate: BLOCK_Q * D_v
        pipe.InitBuffer(pvBuf,   BUFFER_NUM, BLOCK_Q * D_v   * sizeof(float));
        // Per-query m and l: BLOCK_Q scalars each
        pipe.InitBuffer(mBuf,    1, BLOCK_Q * sizeof(float));
        pipe.InitBuffer(lBuf,    1, BLOCK_Q * sizeof(float));
        // Temporary workspaces
        pipe.InitBuffer(tmp1,    BUFFER_NUM, BLOCK_Q * BLOCK_K * sizeof(float));
        pipe.InitBuffer(tmp2,    BUFFER_NUM, std::max(BLOCK_Q, BLOCK_K) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t startWork = this->workPerBlock * AscendC::GetBlockIdx();
        uint32_t endWork   = startWork + this->workPerBlock;
        uint32_t numQTiles = (S_q + BLOCK_Q - 1) / BLOCK_Q;

        for (uint32_t wid = startWork; wid < endWork; wid++) {
            uint32_t bhIdx   = wid / numQTiles;   // which (B,H) pair
            uint32_t qtIdx   = wid % numQTiles;    // which Q tile within (B,H)
            uint32_t b       = bhIdx / H;
            uint32_t h       = bhIdx % H;
            uint32_t qStart  = qtIdx * BLOCK_Q;
            uint32_t curQ    = BLOCK_Q;
            if (qStart + curQ > S_q) { curQ = S_q - qStart; }

            ProcessOneQueryTile(b, h, qStart, curQ);
        }
    }

private:
    // Process one query tile with all KV tiles (online softmax)
    __aicore__ inline void ProcessOneQueryTile(uint32_t b, uint32_t h,
                                                uint32_t qStart, uint32_t curQ)
    {
        // ---- Initialize m, l, acc ----
        {
            AscendC::LocalTensor<float> mInit = mBuf.AllocTensor<float>();
            AscendC::LocalTensor<float> lInit = lBuf.AllocTensor<float>();
            for (uint32_t qi = 0; qi < curQ; qi++) {
                mInit.SetValue(qi, NEG_INF);
                lInit.SetValue(qi, 0.0f);
            }
            mBuf.EnQue(mInit);
            lBuf.EnQue(lInit);
        }
        {
            AscendC::LocalTensor<float> accInit = accBuf.AllocTensor<float>();
            AscendC::Dup(accInit, 0.0f, curQ * D_v);
            accBuf.EnQue(accInit);
        }

        // ---- Load Q tile ----
        AscendC::LocalTensor<float> qLocal = qBuf.AllocTensor<float>();
        uint32_t qOffset = ((b * H + h) * S_q + qStart) * D;
        AscendC::DataCopy(qLocal, qGm[qOffset], curQ * D);
        qBuf.EnQue(qLocal);

        uint32_t numKTiles = (S_k + BLOCK_K - 1) / BLOCK_K;

        // ---- Inner loop: iterate over KV tiles ----
        for (uint32_t kvIdx = 0; kvIdx < numKTiles; kvIdx++) {
            uint32_t kvStart = kvIdx * BLOCK_K;
            uint32_t curK    = BLOCK_K;
            if (kvStart + curK > S_k) { curK = S_k - kvStart; }

            ProcessKVTile(b, h, qStart, kvStart, curQ, curK);
        }

        // ---- Finalize: O = acc / l, write back ----
        FinalizeAndWrite(b, h, qStart, curQ);
    }

    __aicore__ inline void ProcessKVTile(uint32_t b, uint32_t h,
                                          uint32_t qStart, uint32_t kvStart,
                                          uint32_t curQ, uint32_t curK)
    {
        // -- Step 1: Load K tile --
        AscendC::LocalTensor<float> kLocal = kBuf.AllocTensor<float>();
        uint32_t kOffset = ((b * H + h) * S_k + kvStart) * D;
        AscendC::DataCopy(kLocal, kGm[kOffset], curK * D);
        kBuf.EnQue(kLocal);

        // -- Step 2: Compute S = Q @ K^T * scale --
        AscendC::LocalTensor<float> qLocal2 = qBuf.DeQue<float>();
        AscendC::LocalTensor<float> kLocal2 = kBuf.DeQue<float>();
        AscendC::LocalTensor<float> sLocal  = sBuf.AllocTensor<float>();

        AscendC::MatMul<float, float, float, float>(
            sLocal, qLocal2, kLocal2,
            {curQ, D, curK},
            false, true,   // A not trans, B transposed: Q [Q,D] @ K^T [D,K] -> S [Q,K]
            curQ, D, curK);

        // Scale: S = S * scale
        AscendC::Muls(sLocal, sLocal, scale, curQ * curK);

        // -- Step 3: Apply causal mask --
        if (maskType == 1) {
            ApplyCausalMask(sLocal, qStart, kvStart, curQ, curK);
        }

        // -- Step 4: Online softmax update --
        AscendC::LocalTensor<float> mOld = mBuf.DeQue<float>();
        AscendC::LocalTensor<float> lOld = lBuf.DeQue<float>();
        AscendC::LocalTensor<float> accOld = accBuf.DeQue<float>();

        AscendC::LocalTensor<float> pLocal = pBuf.AllocTensor<float>();
        AscendC::LocalTensor<float> mNew   = tmp2.AllocTensor<float>();
        AscendC::LocalTensor<float> lNew   = tmp1.AllocTensor<float>();
        AscendC::LocalTensor<float> sTmp   = tmp1.AllocTensor<float>();  // reuse buffer

        // For each query in the tile, compute m_new, exp(S-m_new), l_new
        for (uint32_t qi = 0; qi < curQ; qi++) {
            // Row max: m_old_row = mOld[qi]; find max over curK scores
            float mOldQi = mOld.GetValue(qi);
            float mNewQi = mOldQi;
            for (uint32_t kj = 0; kj < curK; kj++) {
                float sv = sLocal.GetValue(qi * curK + kj);
                if (sv > mNewQi) { mNewQi = sv; }
            }

            // Compute exp_scale = exp(m_old - m_new)
            float expScale = AscendC::Exp(mOldQi - mNewQi);

            // l_new = l_old * exp_scale + sum(exp(S - m_new))
            float lOldQi = lOld.GetValue(qi);
            float lNewQi = lOldQi * expScale;
            for (uint32_t kj = 0; kj < curK; kj++) {
                float sv = sLocal.GetValue(qi * curK + kj);
                float pv = AscendC::Exp(sv - mNewQi);
                pLocal.SetValue(qi * curK + kj, pv);
                lNewQi += pv;
            }

            // Rescale accumulator: acc *= exp_scale
            for (uint32_t dv = 0; dv < D_v; dv++) {
                float oldAcc = accOld.GetValue(qi * D_v + dv);
                accOld.SetValue(qi * D_v + dv, oldAcc * expScale);
            }

            mNew.SetValue(qi, mNewQi);
            lNew.SetValue(qi, lNewQi);
        }

        // -- Step 5: Load V tile and compute acc += P @ V --
        AscendC::LocalTensor<float> vLocal = vBuf.AllocTensor<float>();
        uint32_t vOffset = ((b * H + h) * S_k + kvStart) * D_v;
        AscendC::DataCopy(vLocal, vGm[vOffset], curK * D_v);
        vBuf.EnQue(vLocal);

        AscendC::LocalTensor<float> pLocal2 = pLocal;  // already have P
        AscendC::LocalTensor<float> vLocal2 = vBuf.DeQue<float>();
        AscendC::LocalTensor<float> pvLocal = pvBuf.AllocTensor<float>();

        AscendC::MatMul<float, float, float, float>(
            pvLocal, pLocal2, vLocal2,
            {curQ, curK, D_v},
            false, false,
            curQ, curK, D_v);

        // acc_new = acc_old + P @ V
        AscendC::Add(accOld, accOld, pvLocal, curQ * D_v);

        // Enque updated m, l, acc for next KV tile iteration
        mBuf.EnQue(mNew);
        lBuf.EnQue(lNew);
        accBuf.EnQue(accOld);
        qBuf.EnQue(qLocal2);    // re-enque Q for next KV tile

        // Free temporary tensors
        pvBuf.FreeTensor(pvLocal);
        vBuf.FreeTensor(vLocal2);
        sBuf.FreeTensor(sLocal);
        kBuf.FreeTensor(kLocal2);

        // Note: pLocal2 aliases pLocal which was allocated, don't double-free
        pBuf.FreeTensor(pLocal);  // but FreeTensor on the handle, not the alias
    }

    __aicore__ inline void ApplyCausalMask(AscendC::LocalTensor<float>& sLocal,
                                            uint32_t qStart, uint32_t kvStart,
                                            uint32_t curQ, uint32_t curK)
    {
        for (uint32_t qi = 0; qi < curQ; qi++) {
            uint32_t globalQi = qStart + qi;
            for (uint32_t kj = 0; kj < curK; kj++) {
                uint32_t globalKj = kvStart + kj;
                if (globalKj > globalQi) {
                    sLocal.SetValue(qi * curK + kj, NEG_INF);
                }
            }
        }
    }

    __aicore__ inline void FinalizeAndWrite(uint32_t b, uint32_t h,
                                             uint32_t qStart, uint32_t curQ)
    {
        AscendC::LocalTensor<float> accFinal = accBuf.DeQue<float>();
        AscendC::LocalTensor<float> lFinal   = lBuf.DeQue<float>();
        AscendC::LocalTensor<float> mFinal   = mBuf.DeQue<float>();
        AscendC::LocalTensor<float> oLocal   = pvBuf.AllocTensor<float>();

        // O = acc / l (per row)
        for (uint32_t qi = 0; qi < curQ; qi++) {
            float invL = 1.0f / lFinal.GetValue(qi);
            for (uint32_t dv = 0; dv < D_v; dv++) {
                float val = accFinal.GetValue(qi * D_v + dv) * invL;
                oLocal.SetValue(qi * D_v + dv, val);
            }
        }

        // Write output tile to global memory
        uint32_t oOffset = ((b * H + h) * S_q + qStart) * D_v;
        AscendC::DataCopy(oGm[oOffset], oLocal, curQ * D_v);

        pvBuf.FreeTensor(oLocal);
        accBuf.FreeTensor(accFinal);
        lBuf.FreeTensor(lFinal);
        mBuf.FreeTensor(mFinal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> qBuf, kBuf, vBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accBuf;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> sBuf, pBuf, pvBuf, tmp1, tmp2;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> mBuf, lBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, oGm;
    uint32_t B = 0, H = 0, S_q = 0, S_k = 0, D = 0, D_v = 0;
    uint32_t workPerBlock = 0, maskType = 0;
    float scale = 1.0f;
};

struct __tiling__ FlashAttentionTilingData {
    uint32_t B;
    uint32_t H;
    uint32_t S_q;
    uint32_t S_k;
    uint32_t D;
    uint32_t D_v;
    float    scale;
    uint32_t maskType;  // 0=none, 1=causal
};

extern "C" __global__ __aicore__ void flash_attention_fwd_fused_kernel(
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR O,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FlashAttentionTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelFlashAttention op;
    op.Init(Q, K, V, O,
            tilingData.B, tilingData.H,
            tilingData.S_q, tilingData.S_k,
            tilingData.D, tilingData.D_v,
            tilingData.scale, tilingData.maskType);
    op.Process();
}

#else
namespace harness { namespace flash_attention_fwd_fused {
const int flash_attention_fwd_fused_kernel_stub = 0;
}}  // namespace harness::flash_attention_fwd_fused
#endif  // HARNESS_HAS_ASCEND
