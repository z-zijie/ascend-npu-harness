// AscendC kernel for Matrix Multiply + Bias (matmul_bias)
// C = A @ B + bias   where A=[M,K], B=[K,N], bias=[N], C=[M,N]
// Target arch: dav-2201.
// Uses AscendC::MatMul on the Cube Core for the matrix multiplication,
// followed by AscendC::Add for bias broadcast.

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "matmul_bias_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulBias {
public:
    __aicore__ inline KernelMatmulBias() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR bias, GM_ADDR C,
                                 uint32_t M, uint32_t K, uint32_t N,
                                 uint32_t tileM, uint32_t tileK, uint32_t tileN)
    {
        ASSERT(tileM != 0 && tileK != 0 && tileN != 0);
        this->M = M; this->K = K; this->N = N;
        this->tileM = tileM; this->tileK = tileK; this->tileN = tileN;

        // Distribute M dimension across cores
        this->mBlocksPerCore = M / (tileM * AscendC::GetBlockNum());
        if (this->mBlocksPerCore == 0) { this->mBlocksPerCore = 1; }
        uint32_t mOffset = this->mBlocksPerCore * tileM * AscendC::GetBlockIdx();

        // Global buffers for A [M,K], B [K,N], bias [N], C [M,N]
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A) + mOffset * K,
                            this->mBlocksPerCore * tileM * K);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B), K * N);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), N);
        cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C) + mOffset * N,
                            this->mBlocksPerCore * tileM * N);

        // UB buffers for tiles
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileM * tileK * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileK * tileN * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, tileN * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, tileM * tileN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Iterate over M tiles assigned to this core
        for (uint32_t mBlock = 0; mBlock < this->mBlocksPerCore; mBlock++) {
            uint32_t globalM = (this->mBlocksPerCore * AscendC::GetBlockIdx() + mBlock) * tileM;

            // Iterate over K tiles (inner product accumulation dimension)
            for (uint32_t kBlock = 0; kBlock < (this->K + tileK - 1) / tileK; kBlock++) {
                uint32_t curK = tileK;
                if ((kBlock + 1) * tileK > this->K) { curK = this->K - kBlock * tileK; }

                // Iterate over N tiles
                for (uint32_t nBlock = 0; nBlock < (this->N + tileN - 1) / tileN; nBlock++) {
                    uint32_t curN = tileN;
                    if ((nBlock + 1) * tileN > this->N) { curN = this->N - nBlock * tileN; }

                    int32_t loopCount = BUFFER_NUM;
                    for (int32_t i = 0; i < loopCount; i++) {
                        CopyIn(i, mBlock, kBlock, nBlock, globalM, curK, curN);
                        Compute(i, mBlock, kBlock, nBlock, curM(mBlock), curK, curN);
                        if (kBlock == 0) {
                            AddBias(i, curN, nBlock);
                        }
                        CopyOut(i, mBlock, nBlock, globalM, curN);
                    }
                }
            }
        }
    }

private:
    __aicore__ uint32_t curM(uint32_t mBlock) const {
        uint32_t curM_ = tileM;
        if ((mBlock + 1) * tileM > this->M) { curM_ = this->M - mBlock * tileM; }
        return curM_;
    }

    __aicore__ inline void CopyIn(int32_t progress,
                                   uint32_t mBlock, uint32_t kBlock, uint32_t nBlock,
                                   uint32_t globalM, uint32_t curK, uint32_t curN)
    {
        AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();

        // A tile: [tileM, curK] from row globalM, col kBlock*tileK
        uint32_t aSrcOff = globalM * K + kBlock * tileK;
        AscendC::DataCopy(aLocal, aGm[aSrcOff], tileM * curK);

        // B tile: [curK, curN] from row kBlock*tileK, col nBlock*tileN
        uint32_t bSrcOff = kBlock * tileK * N + nBlock * tileN;
        AscendC::DataCopy(bLocal, bGm[bSrcOff], curK * curN);

        // Bias tile: [curN] — load once per (mBlock, nBlock) pair
        if (kBlock == 0) {
            AscendC::LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();
            uint32_t biasOff = nBlock * tileN;
            AscendC::DataCopy(biasLocal, biasGm[biasOff], curN);
            inQueueBias.EnQue(biasLocal);
        }

        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void Compute(int32_t progress,
                                    uint32_t mBlock, uint32_t kBlock, uint32_t nBlock,
                                    uint32_t curM, uint32_t curK, uint32_t curN)
    {
        (void)progress; (void)mBlock; (void)kBlock; (void)nBlock;
        AscendC::LocalTensor<float> aLocal = inQueueA.DeQue<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();

        // MatMul on Cube Core: C_tile[M,N] = A_tile[M,K] @ B_tile[K,N]
        // Using AscendC::MatMul template with float types
        AscendC::MatMul<float, float, float, float>(
            cLocal, aLocal, bLocal,
            {curM, curK, curN},
            false, false,   // A not transposed, B not transposed
            curM, curK, curN);

        outQueueC.EnQue<float>(cLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
    }

    __aicore__ inline void AddBias(int32_t progress, uint32_t curN, uint32_t nBlock)
    {
        (void)progress; (void)nBlock;
        AscendC::LocalTensor<float> cLocal    = outQueueC.DeQue<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> cBiased   = outQueueC.AllocTensor<float>();

        // Add bias to each row of C: C[i,:] += bias[:]
        // For each row, add bias vector
        for (uint32_t m = 0; m < tileM; m++) {
            // Use vector Add: c[m*curN : (m+1)*curN] += bias[0:curN]
            AscendC::Add(cLocal[m * curN], cLocal[m * curN], biasLocal, curN);
        }
        // Copy result to output
        AscendC::DataCopy(cBiased, cLocal, tileM * curN);
        outQueueC.EnQue<float>(cBiased);
        outQueueC.FreeTensor(cLocal);  // free the dequeued version
        inQueueBias.FreeTensor(biasLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress,
                                    uint32_t mBlock, uint32_t nBlock,
                                    uint32_t globalM, uint32_t curN)
    {
        (void)progress;
        AscendC::LocalTensor<float> cLocal = outQueueC.DeQue<float>();
        uint32_t dstOff = globalM * N + nBlock * tileN;
        AscendC::DataCopy(cGm[dstOff], cLocal, tileM * curN);
        outQueueC.FreeTensor(cLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueA, inQueueB, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::GlobalTensor<float> aGm, bGm, biasGm, cGm;
    uint32_t M = 0, K = 0, N = 0;
    uint32_t tileM = 16, tileK = 16, tileN = 16;
    uint32_t mBlocksPerCore = 0;
};

struct __tiling__ MatmulBiasTilingData {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t tileM;
    uint32_t tileK;
    uint32_t tileN;
};

extern "C" __global__ __aicore__ void matmul_bias_kernel(
    GM_ADDR A, GM_ADDR B, GM_ADDR bias, GM_ADDR C,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MatmulBiasTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelMatmulBias op;
    op.Init(A, B, bias, C,
            tilingData.M, tilingData.K, tilingData.N,
            tilingData.tileM, tilingData.tileK, tilingData.tileN);
    op.Process();
}

#else
namespace harness { namespace matmul_bias {
const int matmul_bias_kernel_stub = 0;
}}  // namespace harness::matmul_bias
#endif  // HARNESS_HAS_ASCEND
