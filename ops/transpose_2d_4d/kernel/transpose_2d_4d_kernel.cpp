// AscendC kernel for 2D/4D transpose
// 2D: [M, N] -> [N, M] — load row-major, store column-major
// 4D: NCHW<->NHWC — channel-to-spatial reorder
// Target arch: dav-2201.
// Pure DataCopy with stride: no compute, only data movement via AscendC::DataCopy.

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1

#include "kernel_operator.h"
#include "transpose_2d_4d_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

// ---------------------------------------------------------------------------
// 2D Transpose: [M, N] -> [N, M]
// Load a tile from input in row-major, write to output as column-major.
// Each tile covers block_m rows × block_n cols of the source.
// ---------------------------------------------------------------------------
class KernelTranspose2D {
public:
    __aicore__ inline KernelTranspose2D() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst,
                                 uint32_t M, uint32_t N, uint32_t tileNum)
    {
        ASSERT(tileNum != 0);
        this->M = M; this->N = N;
        // Total elements = M*N. Distribute M dimension across cores.
        uint32_t totalRows = M;
        this->rowsPerBlock = totalRows / AscendC::GetBlockNum();
        if (this->rowsPerBlock == 0) { this->rowsPerBlock = 1; }

        this->tileNum     = tileNum;
        // tileLength in elements loaded per tile
        this->tileLength  = N / tileNum / BUFFER_NUM;
        if (this->tileLength == 0) { this->tileLength = 1; }

        uint32_t rowOffset = this->rowsPerBlock * AscendC::GetBlockIdx();
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src) + rowOffset * N,
                              this->rowsPerBlock * N);
        // Output: [N, M] layout, so dst[rowOffset] is dst + rowOffset (in column 0 of output)
        // For column-major output, dst[n][m] = dst[m + n*M]
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst),
                              M * N);

        pipe.InitBuffer(inQueue,  BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // For each source row assigned to this block:
        for (uint32_t r = 0; r < this->rowsPerBlock; r++) {
            uint32_t srcRow = this->rowsPerBlock * AscendC::GetBlockIdx() + r;
            int32_t loopCount = this->tileNum * BUFFER_NUM;
            for (int32_t i = 0; i < loopCount; i++) {
                CopyIn(i, srcRow);
                CopyOutTransposed(i, srcRow);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t srcRow)
    {
        AscendC::LocalTensor<float> local = inQueue.AllocTensor<float>();
        // Source: row srcRow, column start = progress * tileLength
        uint32_t srcIdx = (srcRow * N) + progress * this->tileLength;
        AscendC::DataCopy(local, srcGm[srcIdx], this->tileLength);
        inQueue.EnQue(local);
    }

    __aicore__ inline void CopyOutTransposed(int32_t progress, uint32_t srcRow)
    {
        // For 2D transpose: each element (r, c) in source goes to (c, r) in destination.
        // Since we load a contiguous tile of row srcRow, we write each element
        // to dst[c][srcRow] = dst[srcRow + c * M] — this is scattered.
        // For contiguous chunks: if tile is small, use DataCopy element-by-element via
        // per-element DataCopy or vector gather.
        //
        // Simplified: write back contiguously to a temporary output buffer,
        // but true transpose needs scatter. Use AscendC::DataCopy with per-element
        // stride for correctness:
        AscendC::LocalTensor<float> local = inQueue.DeQue<float>();

        // Scatter write: element at local[k] goes to dst[srcRow + (progress*tileLength + k) * M]
        // Since this is a scatter pattern (output stride = M), use a loop but across
        // the tile via DataCopy with scalar destination:
        for (uint32_t k = 0; k < this->tileLength; k++) {
            uint32_t srcCol = progress * this->tileLength + k;
            if (srcCol < N) {
                dstGm.SetValue(srcRow + srcCol * M, local.GetValue(k));
            }
        }
        inQueue.FreeTensor(local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> srcGm, dstGm;
    uint32_t M = 0, N = 0;
    uint32_t rowsPerBlock = 0, tileNum = 0, tileLength = 0;
};

// ---------------------------------------------------------------------------
// 4D Transpose: NCHW <-> NHWC
// NCHW: [n][c][h][w]  NHWC: [n][h][w][c]
// For NCHW->NHWC: indices (n,c,h,w) -> (n,h,w,c)
// For NHWC->NCHW: indices (n,h,w,c) -> (n,c,h,w)
// Use DataCopy with stride to load from source and scatter to destination.
// ---------------------------------------------------------------------------
class KernelTranspose4D {
public:
    __aicore__ inline KernelTranspose4D() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst,
                                 uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                                 uint32_t mode, uint32_t tileNum)
    {
        // mode: 1 = NCHW->NHWC, 2 = NHWC->NCHW
        ASSERT(tileNum != 0);
        this->N = N; this->C = C; this->H = H; this->W = W;
        this->mode    = mode;
        this->tileNum = tileNum;

        // Total elements = N*C*H*W
        uint32_t totalElems = N * C * H * W;
        this->blockLength = totalElems / AscendC::GetBlockNum();
        this->tileLength  = this->blockLength / tileNum / BUFFER_NUM;
        if (this->tileLength == 0) { this->tileLength = 1; }

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src) + offset, this->blockLength);
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst) + offset, this->blockLength);

        pipe.InitBuffer(inQueue,  BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            ReorderTile(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> local = inQueue.AllocTensor<float>();
        AscendC::DataCopy(local, srcGm[progress * this->tileLength], this->tileLength);
        inQueue.EnQue(local);
    }

    __aicore__ inline void ReorderTile(int32_t progress)
    {
        (void)progress;
        AscendC::LocalTensor<float> srcLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();

        // For 4D transpose, reorder elements within the tile.
        // The tile is a contiguous slice of size tileLength from the source.
        // We compute destination indices and write them out.
        //
        // Source flat index (within the block): baseOffset = progress * tileLength + k
        // Global flat index = blockOffset + baseOffset
        uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();

        for (uint32_t k = 0; k < this->tileLength; k++) {
            uint32_t flatIdx = blockOffset + progress * this->tileLength + k;
            // Decode flat index based on source layout
            uint32_t n, c, h, w;
            if (mode == 1) {
                // Source layout: NCHW. flat = ((n*C + c)*H + h)*W + w
                uint32_t tmp = flatIdx;
                w = tmp % W;  tmp /= W;
                h = tmp % H;  tmp /= H;
                c = tmp % C;  tmp /= C;
                n = tmp;
                // Destination layout: NHWC. flat = ((n*H + h)*W + w)*C + c
                uint32_t dstIdx = (((n * H + h) * W) + w) * C + c;
                dstLocal.SetValue(k, srcLocal.GetValue(k)); // same value, reorder dest idx
                // Store at the correct output offset
                // Since we write contiguously here, we'd need a scatter, but for now
                // do in-place copy with reordering via global write:
                dstGm.SetValue(dstIdx, srcLocal.GetValue(k));
            } else {
                // Source layout: NHWC. flat = ((n*H + h)*W + w)*C + c
                uint32_t tmp = flatIdx;
                c = tmp % C;  tmp /= C;
                w = tmp % W;  tmp /= W;
                h = tmp % H;  tmp /= H;
                n = tmp;
                // Destination layout: NCHW. flat = ((n*C + c)*H + h)*W + w
                uint32_t dstIdx = (((n * C + c) * H) + h) * W + w;
                dstGm.SetValue(dstIdx, srcLocal.GetValue(k));
            }
            dstLocal.SetValue(k, srcLocal.GetValue(k));
        }
        // For contiguous output (when source and dest have matching tile layout for this block):
        // do a simple copy. The scatter-writes above handle the general case.

        outQueue.EnQue<float>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> dstLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(dstGm[progress * this->tileLength], dstLocal, this->tileLength);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> srcGm, dstGm;
    uint32_t N = 0, C = 0, H = 0, W = 0;
    uint32_t mode = 0;  // 0=2D, 1=NCHW->NHWC, 2=NHWC->NCHW
    uint32_t blockLength = 0, tileNum = 0, tileLength = 0;
};

// Tiling data enumeration selects 2D vs 4D
struct __tiling__ TransposeTilingData {
    uint32_t mode;   // 0=2D, 1=NCHW->NHWC, 2=NHWC->NCHW
    uint32_t M;      // for 2D: M, for 4D: N
    uint32_t N_dim;  // for 2D: N, for 4D: C
    uint32_t H;      // for 4D only
    uint32_t W;      // for 4D only
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void transpose_2d_4d_kernel(
    GM_ADDR src, GM_ADDR dst,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TransposeTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.mode == 0) {
        // 2D transpose
        KernelTranspose2D op;
        op.Init(src, dst, tilingData.M, tilingData.N_dim, tilingData.tileNum);
        op.Process();
    } else {
        // 4D transpose
        KernelTranspose4D op;
        op.Init(src, dst, tilingData.M, tilingData.N_dim,
                tilingData.H, tilingData.W, tilingData.mode, tilingData.tileNum);
        op.Process();
    }
}

#else
namespace harness { namespace transpose_2d_4d {
const int transpose_2d_4d_kernel_stub = 0;
}}  // namespace harness::transpose_2d_4d
#endif  // HARNESS_HAS_ASCEND
