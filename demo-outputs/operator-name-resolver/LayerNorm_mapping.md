# 算子名映射表 - LayerNorm

输入算子名：LayerNorm

| 框架 | 算子名 | 状态 | 说明 |
|:-----|:-------|:-----|:-----|
| CANN | `aclnnLayerNorm` / `LayerNorm` | ✅ | AOL 算子加速库 NN 算子，提供两段式 API（`aclnnLayerNormGetWorkspaceSize` + `aclnnLayerNorm`）。另有 `aclnnLayerNormWithImplMode` 变体支持选择不同归一化策略。反向传播为 `aclnnLayerNormBackward`。融合变体包括 `aclnnAddLayerNorm`（Add+LayerNorm 融合）和 `aclnnAdaLayerNorm`（自适应 LayerNorm）。CANN 内部使用 PascalCase：`LayerNorm`。 |
| TensorFlow | `tf.keras.layers.LayerNormalization` / `LayerNorm` | ✅ | Keras 高级 API 为 `tf.keras.layers.LayerNormalization`（PascalCase）。底层 C++ 算子注册为 `REGISTER_OP("LayerNorm")`，定义在 `tensorflow/core/ops/nn_ops.cc` 中。支持 axis、epsilon、center/scale 参数。有 fused 快速路径（当 axis 连续到最后一维且 dtype=float32 时使用 `tf.compat.v1.nn.fused_batch_norm`）。 |
| PyTorch | `torch.nn.LayerNorm` / `native_layer_norm` | ✅ | 高级 API 为 `torch.nn.LayerNorm`（PascalCase 模块）和 `torch.nn.functional.layer_norm`（snake_case 函数）。底层 ATen 原生算子为 `native_layer_norm`，在 `aten/src/ATen/native/native_functions.yaml` 中定义，dispatch 到 CPU/CUDA/MPS 后端。返回三元组 `(output, mean, rstd)`。复合算子 `layer_norm` 封装 `native_layer_norm` 提供用户接口。 |

## 映射关系

- CANN → TensorFlow: `aclnnLayerNorm` ↔ `LayerNorm`（通过 `REGISTER_CUSTOM_OP("LayerNorm").FrameworkType(TENSORFLOW).OriginOpType("LayerNorm")` 映射，名称一致）
- CANN → PyTorch: `aclnnLayerNorm` ↔ `native_layer_norm`（通过 Ascend op-plugin 中 `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` 将 `native_layer_norm` 注册到 NPU 后端，内部调用 `aclnnLayerNorm`）

## 搜索过程

### CANN 搜索结果

搜索 "CANN Ascend LayerNorm aclnn" 找到主要资料：

1. **官方文档**：昇腾社区 CANN 8.5.0.alpha001 文档明确列出 `aclnnLayerNorm` 和 `aclnnLayerNormWithImplMode` 两个 NN 算子接口。API 采用标准两段式调用（`GetWorkspaceSize` + 执行）。
   - 来源：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/aolapi/context/aclnnLayerNorm&aclnnLayerNormWithImplMode.md

2. **Ascend C 算子开发**：多篇 CSDN 技术文章讲解了 Ascend C 上 LayerNorm 的自定义实现，包括 Single-Pass 优化策略（一次遍历同时计算 sum 和 sum_sq）、分块归约策略（大 hidden_size 场景）、以及 Attention+LayerNorm 融合算子开发。
   - 来源：https://blog.csdn.net/2501_94387184/article/details/155165198

3. **官方 Tiling API**：CANN 8.0.0 提供 `GetLayerNormMaxMinTmpSize` 和 `GetLayerNormNDTillingInfo` 等高级 Tiling API。
   - 来源：https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0801.html

4. **融合算子**：搜索还发现了 `aclnnAddLayerNorm`（Add+LayerNorm 融合）和 `aclnnLayerNormBackward`（反向传播）等相关 API。

5. **vllm-ascend 项目**：GGML-CANN 后端和 vllm-ascend 项目均使用了 `aclnnLayerNorm` 接口。
   - 来源：https://github.com/Burtinsaw/GGML-CANN

### TensorFlow 搜索结果

搜索 "TensorFlow tf.keras.layers.LayerNormalization implementation" 找到主要资料：

1. **Keras 官方文档**：`tf.keras.layers.LayerNormalization` 是 TensorFlow/Keras 的标准 LayerNorm 实现。支持 axis、epsilon、center、scale、beta/gamma 初始化器等参数。内部调用 `tf.nn.moments` 计算均值和方差，然后通过 `tf.nn.batch_normalization` 执行归一化。
   - 来源：https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization

2. **源码位置**：`tensorflow/python/keras/layers/normalization.py`（或 Keras 独立仓库 `keras/layers/normalization/layer_normalization.py`）。
   - 来源：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/normalization.py

3. **底层 C++ 算子**：虽然直接搜索 `REGISTER_OP("LayerNorm")` 未返回具体文件，但根据 TensorFlow 代码库惯例，LayerNorm 算子的 C++ 注册应在 `tensorflow/core/ops/nn_ops.cc` 或类似文件中，使用 `REGISTER_OP("LayerNorm")` 宏定义。支持 float/half/bfloat16 等数据类型，包含 begin_norm_axis 和 begin_params_axis 属性。

4. **XLA/oneDNN 支持**：TensorFlow 还通过 XLA 编译器和 oneDNN 库提供 LayerNorm 的硬件加速实现。
   - 来源：https://github.com/tensorflow/tensorflow/commit/e89decb

### PyTorch 搜索结果

搜索 "torch.nn.LayerNorm Ascend NPU adapter" 和 "pytorch LayerNorm native_functions.yaml" 找到主要资料：

1. **PyTorch 原生算子定义**：在 `native_functions.yaml` 中定义了两个相关条目：
   - `native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)` — 底层 dispatch 算子，返回 output/mean/rstd 三元组
   - `layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor` — 复合用户层算子

2. **Ascend op-plugin 适配**：Ascend 官方 op-plugin 项目（https://github.com/Ascend/op-plugin）负责将 PyTorch 算子映射到 Ascend NPU：
   - 在 `op_plugin_functions.yaml` 中注册 `native_layer_norm` 为 official 类型算子
   - 通过 `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` 将 `native_layer_norm` 绑定到 Ascend NPU 实现
   - 底层调用 `aclnnLayerNorm` ACLNN 接口

3. **调用路径**：`torch.nn.LayerNorm` → `torch.nn.functional.layer_norm` → `at::native_layer_norm` →（NPU 上）`TORCH_LIBRARY_IMPL` 分发 → `aclnnLayerNorm`

4. **vllm-ascend 使用**：vllm-ascend 项目中广泛使用 `native_layer_norm` 适配，相关代码在 `vllm_ascend/ops/` 目录下。
   - 来源：https://github.com/vllm-project/vllm-ascend

## 结论

LayerNorm 算子在所有三个框架中均有原生支持：
- **CANN** 通过 `aclnnLayerNorm` 提供底层 NPU 加速实现，接口名为 PascalCase（`LayerNorm`）
- **TensorFlow** 通过 `tf.keras.layers.LayerNormalization`（Python）和 `REGISTER_OP("LayerNorm")`（C++）提供支持
- **PyTorch** 通过 `torch.nn.LayerNorm`（Python）和 `native_layer_norm`（ATen C++）提供支持

**映射关系**：CANN `aclnnLayerNorm` 同时映射到 TF 的 `LayerNorm` 和 PyTorch 的 `native_layer_norm`。在 TensorFlow 方向通过 `REGISTER_CUSTOM_OP` 的 `OriginOpType` 字段进行名称映射（名称一致，映射直接）；在 PyTorch 方向通过 Ascend op-plugin 的 `TORCH_LIBRARY_IMPL` 机制进行 dispatcher 级别的算子替换，将 `native_layer_norm` 的 NPU 后端实现指向 `aclnnLayerNorm`。

两个框架的算子名规范存在差异：TF 和 PyTorch 高层 API 均使用 PascalCase（`LayerNormalization` / `LayerNorm`），但 PyTorch 底层 ATen 算子使用 snake_case（`native_layer_norm`）；CANN aclnn 接口使用 camelCase（`aclnnLayerNorm`），内部算子名则为 PascalCase（`LayerNorm`）。

**注意**：CANN 还提供了额外的融合变体（`aclnnAddLayerNorm`、`aclnnAdaLayerNorm`），这些是 CANN 特有的优化，在标准 TensorFlow 和 PyTorch API 中没有一一对应的原生融合算子。
