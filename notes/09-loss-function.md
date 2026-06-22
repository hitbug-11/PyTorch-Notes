# 09-loss_function

本篇整理深度学习中最基础的两类损失函数，按“Loss Function 总览 -> 极大似然估计与深度学习参数更新 -> MSELoss -> CrossEntropyLoss -> 小结”的顺序组织。阅读时可以先理解 loss function 和极大似然估计的关系，再分别掌握回归任务中的均方误差和分类任务中的交叉熵。

速查目录：

- `Loss Function 总览`：说明 loss function 的作用，并区分 regression 和 classification 两类基础任务。
- `极大似然估计与深度学习参数更新`：说明为什么很多 loss 可以从极大似然估计角度理解，以及深度学习为什么通常使用梯度下降类方法更新参数。
- `MSELoss`：介绍回归问题中的均方误差，推导它和高斯噪声假设下极大似然估计的关系，并说明 PyTorch 用法。
- `CrossEntropyLoss`：介绍分类问题中的交叉熵，推导它和多分类负对数似然的关系，并说明 PyTorch 用法。
- `小结`：归纳 MSE 和 Cross Entropy 的适用任务、统计假设和常见易错点。

## Loss Function 总览

Loss function 用来衡量模型输出和真实标签之间的差距。训练神经网络时，我们先通过前向传播得到预测结果，再用 loss function 把预测误差变成一个标量，最后通过反向传播计算这个标量对模型参数的梯度。

常见训练流程可以概括为：

1. 输入 $x$。
2. 模型 $f_\theta(x)$ 得到预测结果。
3. loss function 根据预测结果和真实标签计算 loss。
4. `backward` 计算梯度。
5. `optimizer` 根据梯度更新参数。

本篇先只关注两类最基础任务：

| 任务类型 | 目标形式 | 常用损失函数 | 典型问题 |
| --- | --- | --- | --- |
| Regression | 连续数值 | `nn.MSELoss` | 房价预测、温度预测、坐标回归 |
| Classification | 离散类别 | `nn.CrossEntropyLoss` | 图像分类、文本分类、多类别识别 |

这两个 loss 看起来只是误差计算公式，但它们背后都可以从统计学中的极大似然估计理解。

## 极大似然估计与深度学习参数更新

极大似然估计，Maximum Likelihood Estimation，简称 MLE，核心思想是：选择一组参数，使已经观察到的训练数据在这组参数下出现的概率尽可能大。

假设训练集为：

$$
D = \{(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)\}
$$

模型参数为 $\theta$。如果样本之间独立，则整份训练数据的似然函数可以写成：

$$
L(\theta) = \prod_{i=1}^{N} p(y_i \mid x_i; \theta)
$$

MLE 的目标是：

$$
\hat{\theta} = \arg\max_{\theta} L(\theta)
$$

由于连乘容易数值下溢，也不方便求导，通常会对似然取对数：

$$
\log L(\theta) = \sum_{i=1}^{N} \log p(y_i \mid x_i; \theta)
$$

最大化对数似然等价于最小化负对数似然：

$$
\arg\max_{\theta} \sum_{i=1}^{N} \log p(y_i \mid x_i; \theta)
=
\arg\min_{\theta} -\sum_{i=1}^{N} \log p(y_i \mid x_i; \theta)
$$

很多深度学习 loss 的统计来源就是负对数似然。也就是说，loss function 并不只是随意定义的误差公式，它经常对应一个概率建模假设。

### 和统计课中 MLE 的区别

在概率论与数理统计课中，MLE 常见做法是：

1. 写出似然函数 $L(\theta)$ 或对数似然函数 $\log L(\theta)$。
2. 对参数 $\theta$ 求导。
3. 令导数等于 0。
4. 解方程得到参数的解析解。

例如简单高斯分布均值估计中，令导数为 0 可以得到样本均值就是均值参数的 MLE。

但深度学习里的参数更新通常不是这样做。深度学习模型一般满足几个特点：

- 参数量巨大，可能有百万、千万甚至更多参数。
- 模型由多层线性变换和非线性激活组成，函数形式非常复杂。
- loss surface 通常是非凸的，可能存在大量局部极小值、鞍点和平坦区域。
- 即使能写出梯度方程，也几乎不可能直接解出所有参数的闭式解。
- 数据通常按 mini-batch 训练，训练目标本身也在用采样近似整份数据的目标。

因此深度学习通常采用数值优化方式：

```python
loss = criterion(pred, target)
loss.backward()
optimizer.step()
```

其中：

- `loss.backward()` 使用自动求导计算 loss 对模型参数的梯度。
- `optimizer.step()` 使用 SGD、Adam 等优化算法沿着降低 loss 的方向更新参数。

所以需要区分两个层面：

- loss 的设计可以来自极大似然估计。
- 参数的求解通常依靠梯度下降类数值优化，而不是手动求导后令导数为 0 解方程。

## MSELoss：回归问题的均方误差

`MSELoss`，Mean Squared Error Loss，均方误差，常用于回归任务。它衡量预测值和真实连续值之间的平方差。

对单个样本，模型预测为：

$$
\hat{y}_i = f_\theta(x_i)
$$

真实值为 $y_i$，平方误差为：

$$
(y_i - \hat{y}_i)^2
$$

对整个 batch 或训练集求平均，就得到均方误差：

$$
\mathrm{MSE}
=
\frac{1}{N} \sum_{i=1}^{N} (y_i - f_\theta(x_i))^2
$$

平方误差的特点是：误差越大，惩罚增长越快。因此 MSE 会更重视较大的预测偏差。

### 和极大似然估计的关系

MSE 可以从高斯噪声假设下的极大似然估计推导出来。

假设回归目标满足：

$$
y_i = f_\theta(x_i) + \varepsilon_i
$$

其中 $f_\theta(x_i)$ 是模型预测值，$\varepsilon_i$ 是观测噪声。进一步假设噪声独立同分布，并且服从均值为 0、方差为 $\sigma^2$ 的高斯分布：

$$
\varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

那么在给定输入 $x_i$ 和参数 $\theta$ 时，目标值 $y_i$ 的条件分布为：

$$
y_i \mid x_i; \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2)
$$

对应的条件概率密度为：

$$
p(y_i \mid x_i; \theta)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp \left(
-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}
\right)
$$

如果样本独立，则整份数据的似然为：

$$
L(\theta)
=
\prod_{i=1}^{N} p(y_i \mid x_i; \theta)
$$

取对数：

$$
\log L(\theta)
=
\sum_{i=1}^{N} \log p(y_i \mid x_i; \theta)
$$

代入高斯密度：

$$
\log L(\theta)
=
\sum_{i=1}^{N}
\left[
-\frac{1}{2}\log(2\pi\sigma^2)
-
\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}
\right]
$$

取负对数似然：

$$
-\log L(\theta)
=
\sum_{i=1}^{N}
\left[
\frac{1}{2}\log(2\pi\sigma^2)
+
\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}
\right]
$$

如果 $\sigma^2$ 被看作固定常数，那么下面两项不会改变最优的 $\theta$：

- $\frac{1}{2}\log(2\pi\sigma^2)$ 是常数项。
- $\frac{1}{2\sigma^2}$ 只是平方误差前的正比例系数。

因此：

$$
\arg\min_{\theta} -\log L(\theta)
=
\arg\min_{\theta} \sum_{i=1}^{N} (y_i - f_\theta(x_i))^2
$$

这说明：在“观测误差服从独立同分布高斯噪声”的假设下，最大化似然等价于最小化平方误差。PyTorch 中的 `MSELoss` 就是这一思想在回归任务中的常见实现。

### PyTorch 用法

`nn.MSELoss` 的常用形式是：

```python
torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")
```

参数说明：

| 参数 | 含义 | 说明 |
| --- | --- | --- |
| `reduction` | 如何把逐元素 loss 汇总成最终输出 | 常用值为 `"mean"`、`"sum"`、`"none"`。默认 `"mean"` 会对所有元素取平均。 |
| `size_average` | 旧版平均控制参数 | 已逐步弃用，实际使用中优先看 `reduction`。 |
| `reduce` | 旧版是否汇总 loss 的参数 | 已逐步弃用，实际使用中优先看 `reduction`。 |

输入和目标的形状应相同：

```text
input:  [batch_size, ...]
target: [batch_size, ...]
```

如果模型输出是 `[batch_size, 1]`，真实值也应该是 `[batch_size, 1]`。

### 使用示例

下面是一个最小回归训练片段：

```python
import torch
import torch.nn as nn
import torch.optim as optim


model = nn.Linear(in_features=3, out_features=1)
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(8, 3)       # [batch_size, in_features]
target = torch.randn(8, 1)  # [batch_size, 1]

pred = model(x)             # [batch_size, 1]
loss = criterion(pred, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss.item())
```

这段代码中：

- `model(x)` 得到连续值预测。
- `criterion(pred, target)` 计算预测值和真实值之间的均方误差。
- `loss.backward()` 计算参数梯度。
- `optimizer.step()` 根据梯度更新模型参数。

## CrossEntropyLoss：分类问题的交叉熵

`CrossEntropyLoss` 是多分类任务中最常用的损失函数。它衡量模型预测的类别概率分布与真实类别之间的差距。

在分类任务中，模型通常不会直接输出概率，而是输出每个类别对应的 raw score，这些值称为 logits。假设一个样本有 $C$ 个类别，模型输出为：

$$
z_i = [z_{i,1}, z_{i,2}, \ldots, z_{i,C}]
$$

通过 softmax 可以把 logits 转换成概率分布：

$$
p_\theta(y_i = c \mid x_i)
=
\frac{\exp(z_{i,c})}{\sum_{j=1}^{C}\exp(z_{i,j})}
$$

如果真实类别是 $y_i$，那么我们希望模型分配给真实类别的概率越大越好。

### 和极大似然估计的关系

对多分类问题，标签 $y_i$ 来自类别集合：

$$
\{1, 2, \ldots, C\}
$$

可以把标签看作服从 categorical distribution。模型给出的 softmax 概率就是每个类别在当前输入下出现的条件概率：

$$
p_\theta(y_i = c \mid x_i)
=
\frac{\exp(z_{i,c})}{\sum_{j=1}^{C}\exp(z_{i,j})}
$$

交叉熵的通用形式，是衡量目标分布 $q_i$ 和模型预测分布 $p_i$ 之间的差距。对第 $i$ 个样本，若目标分布为：

$$
q_i = [q_{i,1}, q_{i,2}, \ldots, q_{i,C}]
$$

模型预测分布为：

$$
p_i = [p_{i,1}, p_{i,2}, \ldots, p_{i,C}]
$$

则单个样本的交叉熵为：

$$
\mathrm{CE}(q_i, p_i)
=
-\sum_{c=1}^{C} q_{i,c} \log p_{i,c}
$$

对整份训练集，交叉熵损失可以写成：

$$
\mathrm{Loss}
=
-\sum_{i=1}^{N}
\sum_{c=1}^{C} q_{i,c} \log p_{i,c}
$$

把 softmax 概率代入 $p_{i,c}$，得到：

$$
\mathrm{Loss}
=
-\sum_{i=1}^{N}
\sum_{c=1}^{C}
q_{i,c}
\log
\frac{\exp(z_{i,c})}{\sum_{j=1}^{C}\exp(z_{i,j})}
$$

从极大似然估计角度看，最小化交叉熵就是最大化目标分布加权后的对数似然：

$$
\arg\min_{\theta}
\left(
-\sum_{i=1}^{N}\sum_{c=1}^{C} q_{i,c}\log p_{i,c}
\right)
=
\arg\max_{\theta}
\sum_{i=1}^{N}\sum_{c=1}^{C} q_{i,c}\log p_{i,c}
$$

也就是说，交叉熵不是单纯看预测类别是否正确，而是直接惩罚模型给目标分布中高概率类别分配过低概率。目标分布 $q_i$ 和预测分布 $p_i$ 越接近，交叉熵越小。

在 PyTorch 中，当 `target` 是类别索引时，`CrossEntropyLoss` 等价于把 `LogSoftmax` 和 `NLLLoss` 组合在一起：

$$
\mathrm{CrossEntropyLoss}(\mathrm{logits}, \mathrm{target})
=
\mathrm{NLLLoss}(\mathrm{LogSoftmax}(\mathrm{logits}), \mathrm{target})
$$

因此使用 `nn.CrossEntropyLoss` 时，输入应直接传 logits，不需要手动先做 `softmax`。

### PyTorch 用法

`nn.CrossEntropyLoss` 的常用形式是：

```python
torch.nn.CrossEntropyLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
)
```

参数说明：

| 参数 | 含义 | 说明 |
| --- | --- | --- |
| `weight` | 类别权重 | 形状为 `[num_classes]` 的张量。类别不平衡时，可以给少数类更大权重。 |
| `ignore_index` | 忽略某个标签值 | 常用于分割或序列任务中忽略 padding 标签。只适用于 target 是类别索引的情况。 |
| `reduction` | 如何汇总 loss | 常用值为 `"mean"`、`"sum"`、`"none"`。默认 `"mean"`。 |
| `label_smoothing` | 标签平滑系数 | 取值范围通常为 `[0.0, 1.0]`。`0.0` 表示不做标签平滑。 |
| `size_average` | 旧版平均控制参数 | 已逐步弃用，实际使用中优先看 `reduction`。 |
| `reduce` | 旧版是否汇总 loss 的参数 | 已逐步弃用，实际使用中优先看 `reduction`。 |

常规多分类任务中，输入和目标形状通常是：

```text
input:  [batch_size, num_classes]
target: [batch_size]
```

注意：

- `input` 是 logits，不要求每一行和为 1，也不要求每个值在 `[0, 1]`。
- `target` 通常是类别索引，不是 one-hot 编码。
- `target` 的 dtype 应该是 `torch.long`。

### 使用示例

下面是一个最小多分类训练片段：

```python
import torch
import torch.nn as nn
import torch.optim as optim


num_features = 4
num_classes = 3

model = nn.Linear(in_features=num_features, out_features=num_classes)
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(8, num_features)               # [batch_size, num_features]
target = torch.tensor([0, 2, 1, 1, 0, 2, 2, 1]) # [batch_size], dtype=torch.long

logits = model(x)                              # [batch_size, num_classes]
loss = criterion(logits, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss.item())
```

这段代码中：

- `model(x)` 输出的是 logits。
- `criterion(logits, target)` 内部会完成 `LogSoftmax + NLLLoss`。
- `target` 是类别索引，例如 `0`、`1`、`2`。
- 不要在传入 `criterion` 前对 `logits` 手动调用 `softmax`。

### 常见易错点

1. 不要先手动 `softmax`

   错误写法：

   ```python
   probs = torch.softmax(logits, dim=1)
   loss = criterion(probs, target)
   ```

   `nn.CrossEntropyLoss` 需要的是 logits。手动 `softmax` 后再传入，会改变数值计算方式，也可能影响梯度稳定性。

2. target 通常不是 one-hot

   常规分类中，正确写法是：

   ```python
   target = torch.tensor([0, 2, 1], dtype=torch.long)
   ```

   而不是：

   ```python
   target = torch.tensor([
       [1, 0, 0],
       [0, 0, 1],
       [0, 1, 0],
   ])
   ```

3. target 类型应是 `torch.long`

   如果类别索引用浮点数表示，通常会报错或导致语义不符合预期：

   ```python
   target = target.long()
   ```

## 小结

`MSELoss` 和 `CrossEntropyLoss` 分别对应深度学习中最基础的回归任务和分类任务。

| Loss | 任务 | 统计假设 | MLE 关系 | PyTorch 输入 |
| --- | --- | --- | --- | --- |
| `MSELoss` | 回归 | 误差服从独立同分布高斯噪声 | 最大化高斯似然等价于最小化平方误差 | `pred` 和 `target` 形状相同 |
| `CrossEntropyLoss` | 多分类 | 标签服从 categorical distribution | 最大化真实类别概率等价于最小化负对数似然 | logits 为 `[N, C]`，target 为 `[N]` 类别索引 |

学习 loss function 时，应同时关注三件事：

- 它适合什么任务。
- 它背后的概率假设或优化目标是什么。
- 它在 PyTorch 中要求什么输入形状和标签格式。

一句话记忆：

```text
回归先看 MSE：高斯噪声假设下的负对数似然。
分类先看 Cross Entropy：softmax 分类模型下真实类别的负对数似然。
```
