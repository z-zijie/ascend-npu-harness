#!/usr/bin/env python3
"""Export CANN-Bench source-dir projects from official task files.

Supported operators: Exp, ForeachAddcdivScalar.
Based on the working `cann-bench/examples/direct_launch_example` project structure.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


HARNESS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = HARNESS_ROOT.parent
DEFAULT_REFERENCE_DIR = WORKSPACE_ROOT / "cann-bench" / "examples" / "direct_launch_example"


def load_operator_name(task_dir: Path) -> str:
    proto_path = task_dir / "proto.yaml"
    if not proto_path.exists():
        raise FileNotFoundError(f"missing proto.yaml: {proto_path}")
    text = proto_path.read_text(encoding="utf-8")
    in_operator = False
    for line in text.splitlines():
        if re.match(r"^operator:\s*$", line):
            in_operator = True
            continue
        if in_operator and re.match(r"^\S", line):
            break
        match = re.match(r"^\s+name:\s*([A-Za-z0-9_]+)\s*$", line)
        if in_operator and match:
            return match.group(1)
    raise ValueError(f"invalid proto.yaml, missing operator.name: {proto_path}")


def validate_task_dir(task_dir: Path) -> None:
    """Validate that the task directory has required files."""
    required = ["proto.yaml", "cases.yaml", "desc.md", "golden.py"]
    missing = [name for name in required if not (task_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"task dir {task_dir} missing files: {', '.join(missing)}")


def clean_or_create_out_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"output dir exists, pass --force to replace: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def copy_reference_scaffold(reference_dir: Path, out_dir: Path) -> None:
    if not reference_dir.exists():
        raise FileNotFoundError(f"reference CANN-Bench example not found: {reference_dir}")

    files = [
        "build.sh",
        "setup.py",
        "CMakeLists.txt",
        "requirements.txt",
        ".gitignore",
        "csrc/extension.cpp",
        "csrc/CMakeLists.txt",
        "csrc/ops/CMakeLists.txt",
    ]
    dirs = ["cmake", "scripts"]

    for rel in files:
        src = reference_dir / rel
        dst = out_dir / rel
        if not src.exists():
            raise FileNotFoundError(f"reference file missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for rel in dirs:
        src = reference_dir / rel
        dst = out_dir / rel
        if not src.exists():
            raise FileNotFoundError(f"reference directory missing: {src}")
        shutil.copytree(src, dst)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


# ===========================================================================
# Exp operator templates
# ===========================================================================

EXP_INIT_PY = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CANN-Bench Exp operator package."""

__version__ = "1.0.0"

import torch

try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "Cannot import _C. Build and install the generated cann_bench wheel first."
    ) from e


def exp(
    x: torch.Tensor,
    base: float = -1.0,
    scale: float = 1.0,
    shift: float = 0.0,
) -> torch.Tensor:
    return torch.ops.cann_bench.exp(x, float(base), float(scale), float(shift))
'''

EXP_OP_CMAKE = '''# Auto-generated CANN-Bench direct launch Exp operator registration.

set(EXP_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/exp_kernel.cpp
)

set(EXP_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/exp_plugin.cpp
)

register_direct_launch_op(
    "${EXP_KERNEL_SRCS}"
    op_kernel
    "${EXP_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
'''

EXP_LAUNCH_H = '''#ifndef EXP_LAUNCH_H
#define EXP_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_exp_tiling_params(int64_t totalLength);

extern "C" {
void launch_exp_kernel_float(
    GM_ADDR x,
    GM_ADDR z,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scale,
    float shift,
    float logBase,
    int32_t hasScale,
    int32_t hasShift,
    int32_t hasLogBase,
    int32_t isBaseOne,
    void* stream);

void launch_exp_kernel_half(
    GM_ADDR x,
    GM_ADDR z,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scale,
    float shift,
    float logBase,
    int32_t hasScale,
    int32_t hasShift,
    int32_t hasLogBase,
    int32_t isBaseOne,
    void* stream);

void launch_exp_kernel_bfloat16(
    GM_ADDR x,
    GM_ADDR z,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scale,
    float shift,
    float logBase,
    int32_t hasScale,
    int32_t hasShift,
    int32_t hasLogBase,
    int32_t isBaseOne,
    void* stream);
}

#endif // EXP_LAUNCH_H
'''

EXP_KERNEL_CPP = '''#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

template <typename T>
class KernelExp {
public:
    __aicore__ inline KernelExp() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR z,
        int64_t totalLength,
        int64_t blockLength,
        uint32_t tileSize,
        float scale,
        float shift,
        float logBase,
        int32_t hasScale,
        int32_t hasShift,
        int32_t hasLogBase,
        int32_t isBaseOne)
    {
        xGm_.SetGlobalBuffer((__gm__ T*)x + blockLength * AscendC::GetBlockIdx());
        zGm_.SetGlobalBuffer((__gm__ T*)z + blockLength * AscendC::GetBlockIdx());
        pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, tileSize * sizeof(T));
        pipe_.InitBuffer(outQueueZ_, PIPELINE_DEPTH, tileSize * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFloatBuf_, tileSize * sizeof(float));
            pipe_.InitBuffer(zFloatBuf_, tileSize * sizeof(float));
        }

        int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
        if (currentBlockLength > blockLength) currentBlockLength = blockLength;
        if (currentBlockLength < 0) currentBlockLength = 0;
        elementNumPerTile_ = tileSize;
        tileNum_ = currentBlockLength / elementNumPerTile_;
        tailTileElementNum_ = currentBlockLength - tileNum_ * elementNumPerTile_;
        scale_ = scale;
        shift_ = shift;
        logBase_ = logBase;
        hasScale_ = hasScale;
        hasShift_ = hasShift;
        hasLogBase_ = hasLogBase;
        isBaseOne_ = isBaseOne;
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < tileNum_; ++i) {
            CopyIn(i * elementNumPerTile_, elementNumPerTile_);
            Compute(elementNumPerTile_);
            CopyOut(i * elementNumPerTile_, elementNumPerTile_);
        }
        if (tailTileElementNum_ > 0) {
            CopyIn(tileNum_ * elementNumPerTile_, tailTileElementNum_);
            Compute(tailTileElementNum_);
            CopyOut(tileNum_ * elementNumPerTile_, tailTileElementNum_);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX_.template AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeF32(AscendC::LocalTensor<float> xLocal, AscendC::LocalTensor<float> zLocal, int64_t count)
    {
        if (isBaseOne_ != 0) {
            AscendC::Duplicate(zLocal, 1.0f, count);
            return;
        }
        if (hasScale_ == 0 && hasShift_ == 0 && hasLogBase_ == 0) {
            AscendC::Exp(zLocal, xLocal, count);
            return;
        }
        if (hasScale_ != 0) {
            AscendC::Muls(zLocal, xLocal, scale_, count);
        } else {
            AscendC::Muls(zLocal, xLocal, 1.0f, count);
        }
        if (hasShift_ != 0) {
            AscendC::Adds(zLocal, zLocal, shift_, count);
        }
        if (hasLogBase_ != 0) {
            AscendC::Muls(zLocal, zLocal, logBase_, count);
        }
        AscendC::Exp(zLocal, zLocal, count);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = inQueueX_.template DeQue<T>();
        auto zLocal = outQueueZ_.template AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            ComputeF32(xLocal, zLocal, count);
        } else {
            auto xF = xFloatBuf_.template Get<float>();
            auto zF = zFloatBuf_.template Get<float>();
            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, count);
            ComputeF32(xF, zF, count);
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT
                : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(zLocal, zF, roundMode, count);
        }
        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto zLocal = outQueueZ_.template DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(zGm_[offset], zLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_, zGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueZ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_, zFloatBuf_;
    int64_t elementNumPerTile_ = 0;
    int64_t tileNum_ = 0;
    int64_t tailTileElementNum_ = 0;
    float scale_ = 1.0f;
    float shift_ = 0.0f;
    float logBase_ = 1.0f;
    int32_t hasScale_ = 0;
    int32_t hasShift_ = 0;
    int32_t hasLogBase_ = 0;
    int32_t isBaseOne_ = 0;
};

template <typename T>
__global__ __aicore__ __vector__ void exp_kernel(
    GM_ADDR x,
    GM_ADDR z,
    int64_t totalLength,
    int64_t blockLength,
    uint32_t tileSize,
    float scale,
    float shift,
    float logBase,
    int32_t hasScale,
    int32_t hasShift,
    int32_t hasLogBase,
    int32_t isBaseOne)
{
    KernelExp<T> op;
    op.Init(x, z, totalLength, blockLength, tileSize, scale, shift, logBase,
        hasScale, hasShift, hasLogBase, isBaseOne);
    op.Process();
}

std::tuple<int64_t, int64_t, int64_t> calc_exp_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static uint32_t FIXED_TILE_ELEMS = 8192;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
    numBlocks = std::max<int64_t>(1, std::min<int64_t>(coreNum, numBlocks));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, blockLength, static_cast<int64_t>(FIXED_TILE_ELEMS));
}

extern "C" {

void launch_exp_kernel_float(
    GM_ADDR x,
    GM_ADDR z,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scale,
    float shift,
    float logBase,
    int32_t hasScale,
    int32_t hasShift,
    int32_t hasLogBase,
    int32_t isBaseOne,
    void* stream)
{
    exp_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize,
        scale, shift, logBase, hasScale, hasShift, hasLogBase, isBaseOne);
}

void launch_exp_kernel_half(
    GM_ADDR x,
    GM_ADDR z,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scale,
    float shift,
    float logBase,
    int32_t hasScale,
    int32_t hasShift,
    int32_t hasLogBase,
    int32_t isBaseOne,
    void* stream)
{
    exp_kernel<half><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize,
        scale, shift, logBase, hasScale, hasShift, hasLogBase, isBaseOne);
}

void launch_exp_kernel_bfloat16(
    GM_ADDR x,
    GM_ADDR z,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scale,
    float shift,
    float logBase,
    int32_t hasScale,
    int32_t hasShift,
    int32_t hasLogBase,
    int32_t isBaseOne,
    void* stream)
{
    exp_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, z, totalLength, blockLength, tileSize,
        scale, shift, logBase, hasScale, hasShift, hasLogBase, isBaseOne);
}

}
'''

EXP_PLUGIN_CPP = '''#include <ATen/Operators.h>
#include <cmath>
#include <tuple>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/exp_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("exp(Tensor x, float base, float scale, float shift) -> Tensor");
}

torch::Tensor exp_meta(
    const torch::Tensor& x,
    double base,
    double scale,
    double shift)
{
    (void)base;
    (void)scale;
    (void)shift;
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("exp", exp_meta);
}

torch::Tensor exp_npu(
    const torch::Tensor& x,
    double base,
    double scale,
    double shift)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto x_c = x.contiguous();
    auto dtype = x_c.scalar_type();
    TORCH_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
        "Exp only supports float32, float16, and bfloat16");

    auto z = torch::empty_like(x_c);
    int64_t totalLength = x_c.numel();
    if (totalLength == 0) {
        return z;
    }

    float scaleF = static_cast<float>(scale);
    float shiftF = static_cast<float>(shift);
    float logBaseF = base > 0.0 ? static_cast<float>(std::log(base)) : 1.0f;
    int32_t hasScale = std::fabs(scaleF - 1.0f) > 1.0e-7f ? 1 : 0;
    int32_t hasShift = std::fabs(shiftF) > 1.0e-7f ? 1 : 0;
    int32_t hasLogBase = std::fabs(logBaseF - 1.0f) > 1.0e-7f ? 1 : 0;
    int32_t isBaseOne = base > 0.0 && std::fabs(static_cast<float>(base) - 1.0f) <= 1.0e-7f ? 1 : 0;

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t numBlocks;
    int64_t blockLength;
    int64_t tileSize64;
    std::tie(numBlocks, blockLength, tileSize64) = calc_exp_tiling_params(totalLength);
    uint32_t tileSize = static_cast<uint32_t>(tileSize64);
    auto xPtr = (GM_ADDR)x_c.data_ptr();
    auto zPtr = (GM_ADDR)z.data_ptr();

    auto acl_call = [=]() -> int {
        if (dtype == torch::kFloat32) {
            launch_exp_kernel_float(xPtr, zPtr, totalLength, numBlocks, blockLength, tileSize,
                scaleF, shiftF, logBaseF, hasScale, hasShift, hasLogBase, isBaseOne, stream);
        } else if (dtype == torch::kFloat16) {
            launch_exp_kernel_half(xPtr, zPtr, totalLength, numBlocks, blockLength, tileSize,
                scaleF, shiftF, logBaseF, hasScale, hasShift, hasLogBase, isBaseOne, stream);
        } else {
            launch_exp_kernel_bfloat16(xPtr, zPtr, totalLength, numBlocks, blockLength, tileSize,
                scaleF, shiftF, logBaseF, hasScale, hasShift, hasLogBase, isBaseOne, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Exp", acl_call);
    return z;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("exp", exp_npu);
}

} // namespace cann_bench
'''

EXP_README = '''# Generated CANN-Bench Exp Project

Generated by `ascend-npu-harness/scripts/export_cannbench.py`.

This source-dir project uses CANN-Bench's direct launch route. The Exp operator
is registered as `torch.ops.cann_bench.exp` and launches an Ascend C kernel
directly from the torch-npu plugin layer.

Build and evaluate from the workspace root on `ascend91093` after sourcing
`/etc/profile` and CANN:

```bash
cd generated/cannbench/exp
bash build.sh
cd ../../../cann-bench
PYTHONPATH=src python -m kernel_eval.cli eval --source-dir ../generated/cannbench/exp --operator Exp --device-id 0
```
'''


# ===========================================================================
# ForeachAddcdivScalar operator templates
# ===========================================================================

FOREACH_ADDCDIV_SCALAR_INIT_PY = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CANN-Bench ForeachAddcdivScalar operator package."""

__version__ = "1.0.0"

from typing import List
import torch

try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "Cannot import _C. Build and install the generated cann_bench wheel first."
    ) from e


def foreach_addcdiv_scalar(
    x1: List[torch.Tensor],
    x2: List[torch.Tensor],
    x3: List[torch.Tensor],
    scalar: float = 1.0,
) -> List[torch.Tensor]:
    return torch.ops.cann_bench.foreach_addcdiv_scalar(x1, x2, x3, float(scalar))
'''

FOREACH_ADDCDIV_SCALAR_OP_CMAKE = '''# Auto-generated CANN-Bench direct launch ForeachAddcdivScalar operator registration.

set(FOREACH_ADDCDIV_SCALAR_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/foreach_addcdiv_scalar_kernel.cpp
)

set(FOREACH_ADDCDIV_SCALAR_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/foreach_addcdiv_scalar_plugin.cpp
)

register_direct_launch_op(
    "${FOREACH_ADDCDIV_SCALAR_KERNEL_SRCS}"
    op_kernel
    "${FOREACH_ADDCDIV_SCALAR_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
'''

FOREACH_ADDCDIV_SCALAR_LAUNCH_H = '''#ifndef FOREACH_ADDCDIV_SCALAR_LAUNCH_H
#define FOREACH_ADDCDIV_SCALAR_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_foreach_addcdiv_scalar_tiling_params(int64_t totalLength);

extern "C" {

void launch_foreach_addcdiv_scalar_kernel_float(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR y,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scalar,
    void* stream);

void launch_foreach_addcdiv_scalar_kernel_half(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR y,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scalar,
    void* stream);

void launch_foreach_addcdiv_scalar_kernel_bfloat16(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR y,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scalar,
    void* stream);

}

#endif // FOREACH_ADDCDIV_SCALAR_LAUNCH_H
'''

FOREACH_ADDCDIV_SCALAR_KERNEL_CPP = '''#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

template <typename T>
class KernelForeachAddcdivScalar {
public:
    __aicore__ inline KernelForeachAddcdivScalar() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR x3,
        GM_ADDR y,
        int64_t totalLength,
        int64_t blockLength,
        uint32_t tileSize,
        float scalar)
    {
        int64_t offset = blockLength * AscendC::GetBlockIdx();
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1 + offset);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2 + offset);
        x3Gm_.SetGlobalBuffer((__gm__ T*)x3 + offset);
        yGm_.SetGlobalBuffer((__gm__ T*)y + offset);

        pipe_.InitBuffer(inQueueX1_, PIPELINE_DEPTH, tileSize * sizeof(T));
        pipe_.InitBuffer(inQueueX2_, PIPELINE_DEPTH, tileSize * sizeof(T));
        pipe_.InitBuffer(inQueueX3_, PIPELINE_DEPTH, tileSize * sizeof(T));
        pipe_.InitBuffer(outQueueY_, PIPELINE_DEPTH, tileSize * sizeof(T));

        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(x1FloatBuf_, tileSize * sizeof(float));
            pipe_.InitBuffer(x2FloatBuf_, tileSize * sizeof(float));
            pipe_.InitBuffer(x3FloatBuf_, tileSize * sizeof(float));
            pipe_.InitBuffer(yFloatBuf_, tileSize * sizeof(float));
        }

        int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
        if (currentBlockLength > blockLength) currentBlockLength = blockLength;
        if (currentBlockLength < 0) currentBlockLength = 0;
        elementNumPerTile_ = tileSize;
        tileNum_ = currentBlockLength / elementNumPerTile_;
        tailTileElementNum_ = currentBlockLength - tileNum_ * elementNumPerTile_;
        scalar_ = scalar;
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < tileNum_; ++i) {
            CopyIn(i * elementNumPerTile_, elementNumPerTile_);
            Compute(elementNumPerTile_);
            CopyOut(i * elementNumPerTile_, elementNumPerTile_);
        }
        if (tailTileElementNum_ > 0) {
            CopyIn(tileNum_ * elementNumPerTile_, tailTileElementNum_);
            Compute(tailTileElementNum_);
            CopyOut(tileNum_ * elementNumPerTile_, tailTileElementNum_);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        auto x1Local = inQueueX1_.template AllocTensor<T>();
        auto x2Local = inQueueX2_.template AllocTensor<T>();
        auto x3Local = inQueueX3_.template AllocTensor<T>();

        AscendC::DataCopyPad(x1Local, x1Gm_[offset], copyParams, padParams);
        AscendC::DataCopyPad(x2Local, x2Gm_[offset], copyParams, padParams);
        AscendC::DataCopyPad(x3Local, x3Gm_[offset], copyParams, padParams);

        inQueueX1_.EnQue(x1Local);
        inQueueX2_.EnQue(x2Local);
        inQueueX3_.EnQue(x3Local);
    }

    __aicore__ inline void ComputeF32(
        AscendC::LocalTensor<float> x1Local,
        AscendC::LocalTensor<float> x2Local,
        AscendC::LocalTensor<float> x3Local,
        AscendC::LocalTensor<float> yLocal,
        int64_t count)
    {
        // y = x2 / x3
        AscendC::Div(yLocal, x2Local, x3Local, count);
        // y = scalar * y
        AscendC::Muls(yLocal, yLocal, scalar_, count);
        // y = x1 + y
        AscendC::Add(yLocal, x1Local, yLocal, count);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto x1Local = inQueueX1_.template DeQue<T>();
        auto x2Local = inQueueX2_.template DeQue<T>();
        auto x3Local = inQueueX3_.template DeQue<T>();
        auto yLocal = outQueueY_.template AllocTensor<T>();

        if constexpr (std::is_same<T, float>::value) {
            ComputeF32(x1Local, x2Local, x3Local, yLocal, count);
        } else {
            auto x1F = x1FloatBuf_.template Get<float>();
            auto x2F = x2FloatBuf_.template Get<float>();
            auto x3F = x3FloatBuf_.template Get<float>();
            auto yF = yFloatBuf_.template Get<float>();

            AscendC::Cast(x1F, x1Local, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(x2F, x2Local, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(x3F, x3Local, AscendC::RoundMode::CAST_NONE, count);

            ComputeF32(x1F, x2F, x3F, yF, count);

            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT
                : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(yLocal, yF, roundMode, count);
        }

        outQueueY_.EnQue(yLocal);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);
        inQueueX3_.FreeTensor(x3Local);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        auto yLocal = outQueueY_.template DeQue<T>();
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> x1Gm_, x2Gm_, x3Gm_, yGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX1_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX2_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX3_;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1FloatBuf_, x2FloatBuf_, x3FloatBuf_, yFloatBuf_;
    int64_t elementNumPerTile_ = 0;
    int64_t tileNum_ = 0;
    int64_t tailTileElementNum_ = 0;
    float scalar_ = 1.0f;
};

template <typename T>
__global__ __aicore__ __vector__ void foreach_addcdiv_scalar_kernel(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR y,
    int64_t totalLength,
    int64_t blockLength,
    uint32_t tileSize,
    float scalar)
{
    KernelForeachAddcdivScalar<T> op;
    op.Init(x1, x2, x3, y, totalLength, blockLength, tileSize, scalar);
    op.Process();
}

std::tuple<int64_t, int64_t, int64_t> calc_foreach_addcdiv_scalar_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    // UB usage per element: 4 queues * PIPELINE_DEPTH(2) * sizeof(T)
    //   + 4 float cast bufs * sizeof(float)  (only for fp16/bf16, but pre-allocate for worst case)
    // Worst case: 8 * 4 + 4 * 4 = 48 bytes/elem; best: 8 * 2 + 4 * 4 = 32 bytes/elem
    // Use conservative 48 bytes/elem with half-UB safety margin.
    constexpr static int64_t BYTES_PER_ELEM = 48;
    constexpr static int64_t MAX_TILE_ELEMS = 4096;
    constexpr static int64_t MIN_TILE_ELEMS = 256;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
    numBlocks = std::max<int64_t>(1, std::min<int64_t>(coreNum, numBlocks));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileElems = static_cast<int64_t>(ubSize / 2) / BYTES_PER_ELEM;
    if (tileElems < MIN_TILE_ELEMS) tileElems = MIN_TILE_ELEMS;
    if (tileElems > MAX_TILE_ELEMS) tileElems = MAX_TILE_ELEMS;
    return std::make_tuple(numBlocks, blockLength, tileElems);
}

extern "C" {

void launch_foreach_addcdiv_scalar_kernel_float(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR y,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scalar,
    void* stream)
{
    foreach_addcdiv_scalar_kernel<float><<<numBlocks, nullptr, stream>>>(
        x1, x2, x3, y, totalLength, blockLength, tileSize, scalar);
}

void launch_foreach_addcdiv_scalar_kernel_half(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR y,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scalar,
    void* stream)
{
    foreach_addcdiv_scalar_kernel<half><<<numBlocks, nullptr, stream>>>(
        x1, x2, x3, y, totalLength, blockLength, tileSize, scalar);
}

void launch_foreach_addcdiv_scalar_kernel_bfloat16(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR x3,
    GM_ADDR y,
    int64_t totalLength,
    int64_t numBlocks,
    int64_t blockLength,
    uint32_t tileSize,
    float scalar,
    void* stream)
{
    foreach_addcdiv_scalar_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(
        x1, x2, x3, y, totalLength, blockLength, tileSize, scalar);
}

}
'''

FOREACH_ADDCDIV_SCALAR_PLUGIN_CPP = '''#include <ATen/Operators.h>
#include <cmath>
#include <tuple>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/foreach_addcdiv_scalar_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("foreach_addcdiv_scalar(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[]");
}

std::vector<at::Tensor> foreach_addcdiv_scalar_meta(
    c10::List<at::Tensor> x1,
    c10::List<at::Tensor> x2,
    c10::List<at::Tensor> x3,
    double scalar)
{
    (void)scalar;
    std::vector<at::Tensor> y;
    y.reserve(x1.size());
    for (int64_t i = 0; i < static_cast<int64_t>(x1.size()); ++i) {
        y.push_back(at::empty_like(x1[i]));
    }
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("foreach_addcdiv_scalar", foreach_addcdiv_scalar_meta);
}

std::vector<at::Tensor> foreach_addcdiv_scalar_npu(
    c10::List<at::Tensor> x1,
    c10::List<at::Tensor> x2,
    c10::List<at::Tensor> x3,
    double scalar)
{
    TORCH_CHECK(x1.size() == x2.size() && x2.size() == x3.size(),
        "foreach_addcdiv_scalar: input tensor lists must have the same length");

    at::Tensor first = x1[0];
    c10::Device device = first.device();
    const c10::OptionalDeviceGuard guard(device);

    std::vector<at::Tensor> y;
    y.reserve(x1.size());

    float scalarF = static_cast<float>(scalar);

    for (int64_t i = 0; i < static_cast<int64_t>(x1.size()); ++i) {
        at::Tensor t1 = x1[i]; t1 = t1.contiguous();
        at::Tensor t2 = x2[i]; t2 = t2.contiguous();
        at::Tensor t3 = x3[i]; t3 = t3.contiguous();

        TORCH_CHECK(t1.sizes() == t2.sizes() && t2.sizes() == t3.sizes(),
            "foreach_addcdiv_scalar: tensors at index ", i, " must have the same shape");

        auto dtype = t1.scalar_type();
        TORCH_CHECK(
            dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
            "foreach_addcdiv_scalar only supports float32, float16, and bfloat16");

        auto out = at::empty_like(t1);
        int64_t totalLength = t1.numel();

        if (totalLength == 0) {
            y.push_back(out);
            continue;
        }

        int64_t numBlocks;
        int64_t blockLength;
        int64_t tileSize64;
        std::tie(numBlocks, blockLength, tileSize64) =
            calc_foreach_addcdiv_scalar_tiling_params(totalLength);

        auto x1Ptr = (GM_ADDR)t1.data_ptr();
        auto x2Ptr = (GM_ADDR)t2.data_ptr();
        auto x3Ptr = (GM_ADDR)t3.data_ptr();
        auto yPtr = (GM_ADDR)out.data_ptr();
        uint32_t tileSize = static_cast<uint32_t>(tileSize64);

        auto stream = c10_npu::getCurrentNPUStream().stream(false);

        auto acl_call = [=]() -> int {
            if (dtype == torch::kFloat32) {
                launch_foreach_addcdiv_scalar_kernel_float(
                    x1Ptr, x2Ptr, x3Ptr, yPtr, totalLength, numBlocks, blockLength,
                    tileSize, scalarF, stream);
            } else if (dtype == torch::kFloat16) {
                launch_foreach_addcdiv_scalar_kernel_half(
                    x1Ptr, x2Ptr, x3Ptr, yPtr, totalLength, numBlocks, blockLength,
                    tileSize, scalarF, stream);
            } else {
                launch_foreach_addcdiv_scalar_kernel_bfloat16(
                    x1Ptr, x2Ptr, x3Ptr, yPtr, totalLength, numBlocks, blockLength,
                    tileSize, scalarF, stream);
            }
            return 0;
        };
        at_npu::native::OpCommand::RunOpApi("ForeachAddcdivScalar", acl_call);

        y.push_back(out);
    }

    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("foreach_addcdiv_scalar", foreach_addcdiv_scalar_npu);
}

} // namespace cann_bench
'''

FOREACH_ADDCDIV_SCALAR_README = '''# Generated CANN-Bench ForeachAddcdivScalar Project

Generated by `ascend-npu-harness/scripts/export_cannbench.py`.

This source-dir project uses CANN-Bench's direct launch route. The
ForeachAddcdivScalar operator is registered as
`torch.ops.cann_bench.foreach_addcdiv_scalar` and launches an Ascend C kernel
directly from the torch-npu plugin layer.

Formula: y[i] = x1[i] + scalar * x2[i] / x3[i]

Build and evaluate from the workspace root on `ascend91093` after sourcing
`/etc/profile` and CANN:

```bash
cd generated/cannbench/foreach_addcdiv_scalar
bash build.sh
cd ../../../cann-bench
PYTHONPATH=src python -m kernel_eval.cli eval --source-dir ../generated/cannbench/foreach_addcdiv_scalar --operator ForeachAddcdivScalar --device-id 0
```
'''


# ===========================================================================
# Generation functions
# ===========================================================================

def generate_exp(out_dir: Path) -> None:
    write(out_dir / "cann_bench" / "__init__.py", EXP_INIT_PY)
    write(out_dir / "README.md", EXP_README)

    op_dir = out_dir / "csrc" / "ops" / "exp"
    write(op_dir / "CMakeLists.txt", EXP_OP_CMAKE)
    write(op_dir / "op_kernel" / "exp_launch.h", EXP_LAUNCH_H)
    write(op_dir / "op_kernel" / "exp_kernel.cpp", EXP_KERNEL_CPP)
    write(op_dir / "op_plugin" / "exp_plugin.cpp", EXP_PLUGIN_CPP)


def generate_foreach_addcdiv_scalar(out_dir: Path) -> None:
    write(out_dir / "cann_bench" / "__init__.py", FOREACH_ADDCDIV_SCALAR_INIT_PY)
    write(out_dir / "README.md", FOREACH_ADDCDIV_SCALAR_README)

    op_dir = out_dir / "csrc" / "ops" / "foreach_addcdiv_scalar"
    write(op_dir / "CMakeLists.txt", FOREACH_ADDCDIV_SCALAR_OP_CMAKE)
    write(op_dir / "op_kernel" / "foreach_addcdiv_scalar_launch.h",
          FOREACH_ADDCDIV_SCALAR_LAUNCH_H)
    write(op_dir / "op_kernel" / "foreach_addcdiv_scalar_kernel.cpp",
          FOREACH_ADDCDIV_SCALAR_KERNEL_CPP)
    write(op_dir / "op_plugin" / "foreach_addcdiv_scalar_plugin.cpp",
          FOREACH_ADDCDIV_SCALAR_PLUGIN_CPP)


GENERATORS = {
    "Exp": generate_exp,
    "ForeachAddcdivScalar": generate_foreach_addcdiv_scalar,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a CANN-Bench direct launch source-dir project")
    parser.add_argument("--task-dir", required=True, type=Path,
                        help="CANN-Bench task directory")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Generated source-dir output path")
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Replace output directory if it exists")
    args = parser.parse_args(argv)

    task_dir = args.task_dir.resolve()
    out_dir = args.out_dir.resolve()
    reference_dir = args.reference_dir.resolve()

    validate_task_dir(task_dir)
    operator_name = load_operator_name(task_dir)

    if operator_name not in GENERATORS:
        supported = ", ".join(sorted(GENERATORS))
        print(f"ERROR: operator {operator_name!r} is not yet supported by this exporter.",
              file=sys.stderr)
        print(f"Supported operators: {supported}", file=sys.stderr)
        return 1

    clean_or_create_out_dir(out_dir, args.force)
    copy_reference_scaffold(reference_dir, out_dir)
    GENERATORS[operator_name](out_dir)

    print(f"Exported {operator_name} CANN-Bench direct launch source-dir: {out_dir}")
    print("Next:")
    print(f"  cd {out_dir}")
    print("  bash build.sh")
    print("  cd <workspace>/cann-bench")
    print(f"  PYTHONPATH=src python -m kernel_eval.cli eval --source-dir {out_dir}"
          f" --operator {operator_name} --device-id 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
