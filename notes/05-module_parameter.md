# 05-module_parameter

本篇围绕 `nn.Parameter` 解释 PyTorch 如何识别、保存、遍历和更新模型参数。核心主线是：

```text
Parameter 是什么
    -> 如何被 Module 注册到 _parameters
    -> 多个 Parameter 如何用容器管理
    -> model.parameters() 如何取出参数
    -> 优化器如何更新参数
```

## Parameter 是什么

`torch.nn.Parameter` 可以理解为“带有模型参数身份的 Tensor”。它本质上仍然能像 Tensor 一样参与加法、矩阵乘法、求梯度等计算；特殊之处在于：当一个 `Parameter` 被赋值为 `nn.Module` 的属性时，PyTorch 会自动把它登记到这个模块的 `_parameters` 中。

普通 Tensor 和 `Parameter` 的关键区别不在于能不能计算，而在于会不会被 `Module` 当作模型参数管理。

```python
import torch
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_weight = torch.randn(2, 3)
        self.param_weight = nn.Parameter(torch.randn(2, 3))


model = MyModule()

for name, param in model.named_parameters():
    print(name, param.shape)
```

输出示例：

```text
param_weight torch.Size([2, 3])
```

`tensor_weight` 虽然也是 Tensor，但不会出现在 `named_parameters()` 中；`param_weight` 是 `nn.Parameter`，因此会被模型识别为参数。

下面手动实现一个接近 `nn.Linear` 的层，进一步说明 `Parameter` 如何参与计算：

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        output = x @ self.weight.t()
        if self.bias is not None:
            output += self.bias
        return output
```

这段代码中：

- `self.weight` 的形状是 `[out_features, in_features]`，表示线性层的权重矩阵。
- `self.bias` 的形状是 `[out_features]`，表示每个输出特征对应一个偏置。
- `x @ self.weight.t()` 是矩阵乘法，等价于手动实现全连接层的核心计算。
- `weight` 和 `bias` 都是 `nn.Parameter`，会被 `model.parameters()` 找到，并在训练中由优化器更新。

例如：

```python
linear = MyLinear(in_features=3, out_features=2)

for name, param in linear.named_parameters():
    print(name, param.shape, param.requires_grad)
```

输出示例：

```text
weight torch.Size([2, 3]) True
bias torch.Size([2]) True
```

如果输入是 4 个样本，每个样本 3 个特征：

```python
x = torch.randn(4, 3)
output = linear(x)
print(output.shape)
```

输出示例：

```text
torch.Size([4, 2])
```

这说明 `Parameter` 既是计算图中的 Tensor，又是 `Module` 可以自动管理的可训练参数。

## 参数如何注册

`Module`、`Parameter` 和 `_parameters` 的关系可以这样理解：

- `Module` 管结构：模型由哪些层组成，前向传播怎么走。
- `Parameter` 管数值：这些层里面哪些 Tensor 是需要学习的。
- `_parameters` 是每个 `Module` 内部保存 `Parameter` 的字典。

```text
nn.Parameter 是参数对象本身
module._parameters 是 Module 内部保存这些参数对象的字典
```

例如：

```python
class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 3))
```

执行下面这行代码时：

```python
self.weight = nn.Parameter(torch.randn(2, 3))
```

`nn.Module.__setattr__` 会发现右侧是 `nn.Parameter`，于是把它登记到当前模块的 `_parameters` 中。可以近似理解为：

```python
self._parameters["weight"] = self.weight
```

所以这个模块内部大致是：

```text
MyLinear._parameters = {
    "weight": Parameter(...)
}
```

因此：

- `Parameter` 是真正参与训练和更新的可学习 Tensor。
- `_parameters` 是 `Module` 内部保存这些 `Parameter` 的字典。
- `model.parameters()` 会遍历这些字典，把里面的 `Parameter` 取出来。

## 参数存放位置

某个模块的 `_parameters` 只保存“当前模块自己直接拥有的参数”，不会把子模块的参数全都塞进当前模块。

以 `TinnyCNN` 为例：

```python
class TinnyCNN(nn.Module):
    def __init__(self, cls_num=2):
        super(TinnyCNN, self).__init__()
        self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3))
        self.fc = nn.Linear(36, cls_num)
```

`TinnyCNN` 自己没有直接写：

```python
self.weight = nn.Parameter(...)
```

所以最外层的 `TinnyCNN._parameters` 通常是空的。它真正直接管理的是两个子模块：

```text
TinnyCNN._modules = {
    "convolution_layer": Conv2d(...),
    "fc": Linear(...)
}
```

参数分别存放在子模块自己的 `_parameters` 中：

```text
TinnyCNN.convolution_layer._parameters = {
    "weight": Parameter(...),
    "bias": Parameter(...)
}

TinnyCNN.fc._parameters = {
    "weight": Parameter(...),
    "bias": Parameter(...)
}
```

从模型整体看，`TinnyCNN` 的参数树是：

```text
TinnyCNN
├── convolution_layer
│   ├── weight: Parameter, shape = [1, 1, 3, 3]
│   └── bias: Parameter, shape = [1]
└── fc
    ├── weight: Parameter, shape = [2, 36]
    └── bias: Parameter, shape = [2]
```

所以，不能简单地以为“最外层模型的 `_parameters` 就包含全部参数”。更准确的理解是：每个 `Module` 都有自己的 `_parameters`，外层模型通过递归遍历整个 `Module` 树，才能拿到所有层的参数。

## 参数容器

除了单个 `Parameter`，PyTorch 也提供了参数容器：`nn.ParameterList` 和 `nn.ParameterDict`。它们和 `ModuleList`、`ModuleDict` 的思路类似，只不过管理对象从子模块变成了参数。

需要参数容器的典型场景是：模型里有一组需要训练的矩阵或向量，但它们本身不是完整网络层。如果直接把多个 `nn.Parameter` 放进普通 Python `list` 或 `dict`，`Module` 不会自动递归注册里面的元素；使用 `ParameterList` 或 `ParameterDict`，这些参数才能被 `model.parameters()` 找到。

`ParameterDict` 适合按名字选择参数：

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterDict({
            "left": nn.Parameter(torch.randn(5, 10)),
            "right": nn.Parameter(torch.randn(5, 10)),
        })

    def forward(self, x, choice):
        x = self.params[choice].mm(x)
        return x
```

查看参数名时，会带上容器名称和 key：

```python
model = MyModule()

for name, param in model.named_parameters():
    print(name, param.shape)
```

输出示例：

```text
params.left torch.Size([5, 10])
params.right torch.Size([5, 10])
```

`ParameterList` 适合按顺序保存一组参数，可以像列表一样迭代，也可以用整数下标访问：

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(10, 10))
            for _ in range(10)
        ])

    def forward(self, x):
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x
```

查看参数名时，`ParameterList` 会用数字下标作为名字的一部分：

```python
model = MyModule()

for name, param in model.named_parameters():
    print(name, param.shape)
```

输出示例：

```text
params.0 torch.Size([10, 10])
params.1 torch.Size([10, 10])
...
params.9 torch.Size([10, 10])
```

简单总结：

- 子模块用 `ModuleList`、`ModuleDict`。
- 可训练参数用 `ParameterList`、`ParameterDict`。
- 普通 Python `list`、`dict` 只保存对象，不负责把里面的 `Parameter` 注册给 `Module`。

## 参数如何取出

`nn.Module` 通过注册机制知道自己有哪些子模块和参数，所以可以提供 `parameters()`、`named_parameters()`、`state_dict()` 这些函数。

### `parameters()`

`parameters()` 返回模型中所有参数的迭代器。它的核心逻辑可以理解为：递归遍历当前 `Module` 以及所有子 `Module`，逐个查看每个模块自己的 `_parameters`，然后把里面的 `Parameter` 一个个取出来。

```python
for param in model.parameters():
    print(param.shape)
```

以 `TinnyCNN` 为例，`model.parameters()` 大致会按下面的思路工作：

```text
遍历 TinnyCNN._parameters
遍历 TinnyCNN._modules["convolution_layer"]._parameters
遍历 TinnyCNN._modules["fc"]._parameters
```

可以用伪代码理解为：

```text
遍历 model 自身以及所有子 Module:
    遍历当前 module._parameters.values():
        逐个返回 param
```

实际拿到的参数就是：

```text
convolution_layer.weight
convolution_layer.bias
fc.weight
fc.bias
```

`parameters()` 只返回参数本身，不返回参数名字。如果想知道参数来自哪一层，应使用 `named_parameters()`。

### `named_parameters()`

`named_parameters()` 返回参数名和参数对象，更适合调试和检查模型。

```python
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
```

输出示例：

```text
convolution_layer.weight torch.Size([1, 1, 3, 3]) True
convolution_layer.bias torch.Size([1]) True
fc.weight torch.Size([2, 36]) True
fc.bias torch.Size([2]) True
```

### `state_dict()`

`state_dict()` 返回模型的状态字典，里面包含参数，也包含需要保存的 buffer。

```python
print(model.state_dict().keys())
```

输出示例：

```text
odict_keys(['convolution_layer.weight', 'convolution_layer.bias', 'fc.weight', 'fc.bias'])
```

保存模型参数时，最常见的方式是保存 `state_dict`：

```python
torch.save(model.state_dict(), "tinnycnn.pth")
```

## 参数如何更新

优化器需要知道“应该更新哪些参数”。因此创建优化器时，通常会把 `model.parameters()` 传进去：

```python
import torch.optim as optim


optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
)
```

这里可以拆成两部分看：

- `model.parameters()`：告诉优化器要更新哪些 `Parameter`。
- `lr`、`momentum`、`weight_decay`：告诉优化器如何更新参数，这些属于超参数。

训练时，反向传播会把梯度保存到参数的 `.grad` 中，`optimizer.step()` 再根据这些梯度更新参数值。

```python
outputs = model(inputs)
loss = criterion(outputs, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 相关概念

学习 PyTorch 时，容易把“权重”“参数”“超参数”混在一起。可以这样区分：

| 概念 | 含义 | 是否由训练自动学习 | 例子 |
| --- | --- | --- | --- |
| 权重 | 某一层中的可学习矩阵或卷积核 | 是 | `Conv2d.weight`、`Linear.weight` |
| 参数 | 模型中所有可学习量的统称 | 是 | 权重、偏置 |
| 超参数 | 训练前由人指定的配置 | 否 | `lr`、`momentum`、`weight_decay`、`batch_size` |

权重通常是参数的一部分；参数还包括偏置等其他可学习量；超参数不会被反向传播自动学习，而是控制训练过程。

## 打印参数

下面的示例都基于这个模型：

```python
model = TinnyCNN(cls_num=2)
```

### 打印模型结构

命令：

```python
print(model)
```

输出示例：

```text
TinnyCNN(
  (convolution_layer): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1))
  (fc): Linear(in_features=36, out_features=2, bias=True)
)
```

这个命令适合快速查看模型由哪些子模块组成。

### 打印参数名

命令：

```python
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
```

输出示例：

```text
convolution_layer.weight torch.Size([1, 1, 3, 3]) True
convolution_layer.bias torch.Size([1]) True
fc.weight torch.Size([2, 36]) True
fc.bias torch.Size([2]) True
```

这个命令最适合调试模型参数，因为它能同时看到参数名、维度和 `requires_grad` 状态。

### 打印参数形状

命令：

```python
for param in model.parameters():
    print(param.shape)
```

输出示例：

```text
torch.Size([1, 1, 3, 3])
torch.Size([1])
torch.Size([2, 36])
torch.Size([2])
```

这个命令适合快速确认优化器能拿到哪些参数，但它不会显示参数名。

### 打印完整参数值

命令：

```python
for name, param in model.named_parameters():
    print(name)
    print(param)
```

输出示例：

```text
convolution_layer.weight
Parameter containing:
tensor([[[[ 0.1824, -0.0941,  0.2165],
          [-0.3152,  0.0528,  0.1087],
          [ 0.2713, -0.2016,  0.0449]]]], requires_grad=True)

convolution_layer.bias
Parameter containing:
tensor([0.1172], requires_grad=True)

fc.weight
Parameter containing:
tensor([[ 0.0231, -0.0814,  0.1128,  ..., -0.0416,  0.0589, -0.0972],
        [-0.0665,  0.1043, -0.0187,  ...,  0.1201, -0.0324,  0.0716]],
       requires_grad=True)

fc.bias
Parameter containing:
tensor([ 0.0842, -0.0527], requires_grad=True)
```

完整参数值来自随机初始化，每次运行通常不同。调试时更常看参数名、形状和 `requires_grad`，不一定需要打印全部数值。

## 小结

- `Parameter` 是带有模型参数身份的 Tensor，赋值给 `Module` 属性后会被登记到 `_parameters`。
- 普通 Tensor 可以参与计算，但不会自动出现在 `named_parameters()` 中。
- 每个 `Module` 都有自己的 `_parameters`，子模块参数保存在子模块自己的 `_parameters` 中。
- 多个独立参数应使用 `ParameterList` 或 `ParameterDict` 管理，避免普通 `list`、`dict` 中的参数无法被模型注册。
- `model.parameters()` 会递归遍历整个 `Module` 树，逐层取出 `_parameters` 中的参数。
- 优化器通过 `model.parameters()` 获取需要更新的参数；`lr`、`momentum`、`weight_decay` 等是控制更新方式的超参数。
- 打印模型参数时，最常用的是 `print(model)`、`named_parameters()` 和 `state_dict().keys()`。
