# 07-module_layer

本篇用于逐步整理 PyTorch 中常见的 `nn` 网络层，按“Layer 总览 -> 单个 layer 的作用、参数、输入输出形状、代码实践、易错点”的顺序持续补充。

速查目录：

- `Layer 总览`：说明 layer 和 `nn.Module` 的关系，以及学习单个 layer 时应关注哪些问题。
- `单个 layer 的整理格式`：约定后续每个网络层章节的固定结构，方便复习和继续追加内容。
- 后续章节：每学习完一个常见 layer，就在本篇新增一个对应章节。

## Layer 总览

在 PyTorch 中，常见网络层通常都定义在 `torch.nn` 下，例如 `nn.Linear`、`nn.Conv2d`、`nn.ReLU`、`nn.BatchNorm2d`、`nn.MaxPool2d`。这些 layer 本质上都是 `nn.Module` 或与 `Module` 体系配合使用的组件。

学习一个 layer 时，不应只记住它的名字，还要理解它在模型中的位置：

- 它解决什么问题，例如特征变换、空间卷积、非线性激活、归一化、池化或正则化。
- 它是否包含可学习参数，例如 `Linear` 和 `Conv2d` 有权重，`ReLU` 通常没有权重。
- 它对输入输出形状有什么要求，尤其是 batch 维、通道维、特征维和空间维。
- 它在 `forward` 中如何被调用，以及常和哪些 layer 组合使用。
- 它有哪些常见易错点，例如维度不匹配、通道数写错、训练和推理行为不同。

可以把单个 layer 理解为模型中的一个计算单元：外层 `Module` 负责组织网络结构，而具体 layer 负责完成某一步张量计算。

```python
import torch
import torch.nn as nn


layer = nn.Linear(in_features=4, out_features=2)

x = torch.randn(3, 4)  # [batch_size, in_features]
y = layer(x)           # [batch_size, out_features]

print(y.shape)
```

这段代码中，`nn.Linear` 是一个具体 layer；它接收形状为 `[3, 4]` 的输入，把每个样本的 4 个特征映射成 2 个输出特征。

## 单个 layer 的整理格式

后续每学习完一个 layer，就按下面结构整理成一个二级章节：

```text
## Layer 名称

### 作用

说明这个 layer 用来做什么，通常放在模型的什么位置。

### 常用参数

解释最常用、最容易写错的参数。

### 输入输出形状

写清楚输入张量和输出张量的维度变化。

### 代码示例

给出最小但完整的 PyTorch 示例。

### 易错点

整理维度、参数、训练推理差异等常见问题。
```

这个格式不是为了限制内容，而是为了保证每个 layer 的笔记都能围绕“怎么用、输入输出是什么、哪里容易错”来展开。
