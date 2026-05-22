# LayerNorm 需求背景

## 运行环境
- 芯片平台：Ascend 950DT / 950PR
- CANN 版本：8.0.RC1

## 调用方式
- 调用方式：图模式
- 调用框架：PyTorch + torch_npu

## ACLNN 接口
```cpp
aclnnStatus aclnnLayerNormGetWorkspaceSize(const aclTensor* x, const aclTensor* gamma,
                                           const aclTensor* beta, const aclIntArray* normalizedShape,
                                           double eps, const aclTensor* out,
                                           uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnLayerNorm(void* workspace, uint64_t workspaceSize,
                           aclOpExecutor* executor, aclrtStream stream);
```

## 性能目标
- 带宽利用率 >= 80%
- 延迟 <= 100us (典型 shape)
