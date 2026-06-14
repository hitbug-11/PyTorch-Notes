# 05-module_parameter

本篇用于复习和速查 `nn.Parameter` 的核心机制，按“定义 -> 注册 -> 存放 -> 容器 -> 取出 -> 打印 -> 保存 -> 载入 -> 更新”的顺序展开。

速查目录：

- `Parameter 是什么`：区分普通 Tensor 和 `nn.Parameter`，并用手写 `MyLinear` 说明参数如何参与计算。
- `参数如何注册`：解释 `Parameter` 被赋值给 `Module` 属性后，如何进入 `_parameters`。
- `参数存放位置`：用 `TinnyCNN` 说明最外层模块和子模块各自保存哪些参数。
- `参数容器`：说明 `ParameterList`、`ParameterDict` 适合管理多组独立参数。
- `参数如何取出`：梳理 `parameters()`、`named_parameters()`、`state_dict()` 的作用。
- `打印参数`：汇总常用打印命令和输出示例。
- `参数如何保存`：说明如何保存模型参数和训练检查点。
- `参数如何载入`：说明如何用 `torch.load()` 和 `load_state_dict()` 载入已保存参数。
- `参数如何更新`：说明优化器如何通过 `model.parameters()` 获取并更新参数。
- `相关概念`：区分权重、参数、超参数。
- `小结`：归纳参数注册、查看、保存、载入和更新的关键点。

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

## 参数如何保存

保存模型参数时，最常见的方式是保存 `state_dict`：

```python
torch.save(model.state_dict(), "tinnycnn.pth")
```

这种文件只保存模型参数和 buffer，不保存模型类定义，也不保存 `forward()` 逻辑。后续载入时，需要先重新创建同结构模型，再把参数加载进去。

如果保存的不只是模型参数，而是训练检查点，文件里通常会包含模型参数、优化器状态和训练轮数：

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "checkpoint.pth")
```

两种保存方式的区别：

| 保存对象 | 适合场景 | 载入方式 |
| --- | --- | --- |
| `model.state_dict()` | 只保存模型参数，用于推理或迁移学习 | `model.load_state_dict(torch.load(...))` |
| checkpoint 字典 | 保存训练进度，用于中断后继续训练 | 分别取出 `model_state_dict`、`optimizer_state_dict`、`epoch` |

一般建议优先保存 `state_dict` 或结构清晰的 checkpoint 字典，不建议直接保存整个模型对象。保存整个模型会把代码结构和序列化绑定得更紧，后续类名、文件路径或代码结构变化时更容易出问题。

## 参数如何载入

保存参数后，最常见的载入方式是：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinnyCNN(cls_num=2).to(device)

state_dict = torch.load(
    "tinnycnn.pth",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()
```

这里要区分两个步骤：

- `torch.load()`：从文件中读取保存下来的对象，常见结果是一个 `state_dict` 字典。
- `model.load_state_dict()`：把这个字典中的参数和 buffer 拷贝到当前模型对象中。

完整流程通常是：先重新创建同结构模型，再载入参数。

```python
import torch
import torch.nn as nn


class TinnyCNN(nn.Module):
    def __init__(self, cls_num=2):
        super().__init__()
        self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3))
        self.fc = nn.Linear(36, cls_num)

    def forward(self, x):
        x = self.convolution_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinnyCNN(cls_num=2).to(device)

state_dict = torch.load(
    "tinnycnn.pth",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

x = torch.randn(4, 1, 8, 8).to(device)
with torch.no_grad():
    output = model(x)
```

这个例子中，`tinnycnn.pth` 保存的是：

```python
torch.save(model.state_dict(), "tinnycnn.pth")
```

所以载入时也应该先创建 `TinnyCNN(cls_num=2)`，再调用 `load_state_dict()`。参数文件只保存参数数值，不保存 `forward()` 逻辑；如果模型结构不一致，参数名或参数形状就可能对不上。

`torch.load()` 常用参数：

| 参数 | 含义 | 说明 |
| --- | --- | --- |
| `f` | 文件路径或文件对象 | 要读取的 `.pth`、`.pt` 等文件。 |
| `map_location` | 设备映射 | 控制参数载入到哪里。训练或推理时常用 `map_location=device` 直接映射到当前设备；排查兼容问题时也常用 `"cpu"` 先载入到 CPU。 |
| `weights_only` | 是否只按权重安全载入 | 推荐载入 `state_dict` 时显式写 `weights_only=True`，减少反序列化不可信对象的风险。 |

`load_state_dict()` 常用参数：

| 参数 | 含义 | 说明 |
| --- | --- | --- |
| `state_dict` | 参数字典 | 通常来自 `torch.load()`，里面的 key 应该和当前模型 `state_dict().keys()` 对应。 |
| `strict` | 是否严格匹配 | 默认 `True`，要求参数名完全一致；为 `False` 时允许缺失或多余 key，常用于迁移学习或只加载部分参数。 |
| `assign` | 是否保留 state_dict 中 Tensor 属性 | 默认 `False`，通常不需要手动改。若设为 `True`，一般应在载入参数之后再创建优化器。 |

### 严格载入

默认情况下，`load_state_dict()` 使用 `strict=True`：

```python
model.load_state_dict(state_dict)
```

这要求保存文件中的 key 和当前模型的 key 完全匹配。例如当前模型需要：

```text
convolution_layer.weight
convolution_layer.bias
fc.weight
fc.bias
```

那么保存文件里也应该有这些 key，而且对应 Tensor 的形状也要一致。

如果模型结构改过，比如 `fc` 输出类别数从 2 改成 10，那么 `fc.weight` 和 `fc.bias` 的形状就不匹配，严格载入会报错。

### 非严格载入

如果只想载入部分参数，可以使用 `strict=False`：

```python
incompatible_keys = model.load_state_dict(state_dict, strict=False)

print(incompatible_keys.missing_keys)
print(incompatible_keys.unexpected_keys)
```

返回值中：

- `missing_keys`：当前模型需要，但参数文件里没有的 key。
- `unexpected_keys`：参数文件里有，但当前模型用不到的 key。

这种方式常用于迁移学习。例如只载入卷积特征提取层，重新训练分类头：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinnyCNN(cls_num=10).to(device)
state_dict = torch.load("tinnycnn.pth", map_location=device, weights_only=True)

current_state = model.state_dict()

filtered_state = {
    name: value
    for name, value in state_dict.items()
    if name in current_state and value.shape == current_state[name].shape
}

model.load_state_dict(filtered_state, strict=False)
```

这里通过参数名和形状筛选，只把能对上的参数载入当前模型。分类头形状不一致的参数会被跳过，由当前模型重新初始化。

### 载入到 GPU

如果要在 GPU 上推理或继续训练，常见写法是先确定 `device`，再把模型和载入的参数都放到同一个设备上：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinnyCNN(cls_num=2).to(device)

state_dict = torch.load(
    "tinnycnn.pth",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

x = torch.randn(4, 1, 8, 8).to(device)
with torch.no_grad():
    output = model(x)
```

需要注意：模型参数和输入数据必须在同一个设备上。也就是说，模型 `.to(device)` 后，输入 batch 也要 `.to(device)`，否则 forward 时会出现 CPU/GPU 设备不一致的错误。

如果不确定 checkpoint 原来保存在哪个设备，或者当前机器没有 GPU，可以先载入到 CPU：

```python
state_dict = torch.load(
    "tinnycnn.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
```

这种 CPU 载入方式更稳妥，适合跨机器、跨设备迁移；之后如果需要 GPU，再执行 `model.to(device)`。

### 载入训练检查点

如果载入的是“参数如何保存”一节中的 checkpoint 字典，需要先取出对应字段：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinnyCNN(cls_num=2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

这类写法适合中断后继续训练。如果只是做推理，通常只需要保存和载入 `model.state_dict()`。

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

## 小结

- `Parameter` 是带有模型参数身份的 Tensor，赋值给 `Module` 属性后会被登记到 `_parameters`。
- 普通 Tensor 可以参与计算，但不会自动出现在 `named_parameters()` 中。
- 每个 `Module` 都有自己的 `_parameters`，子模块参数保存在子模块自己的 `_parameters` 中。
- 多个独立参数应使用 `ParameterList` 或 `ParameterDict` 管理，避免普通 `list`、`dict` 中的参数无法被模型注册。
- `model.parameters()` 会递归遍历整个 `Module` 树，逐层取出 `_parameters` 中的参数。
- 打印参数时，最常用的是 `print(model)`、`named_parameters()` 和 `state_dict().keys()`。
- 保存参数时，推荐保存 `model.state_dict()`；如果要继续训练，则保存包含模型参数、优化器状态和 epoch 的 checkpoint 字典。
- 载入参数时，用 `torch.load(..., map_location=device)` 读取到目标设备，再用 `model.load_state_dict()` 恢复到同结构模型中。
- 载入参数前要先创建模型结构；参数文件只保存数值，不保存模型的 `forward()` 逻辑。
- 优化器通过 `model.parameters()` 获取需要更新的参数；`lr`、`momentum`、`weight_decay` 等是控制更新方式的超参数。

## 参考资料

- [PyTorch torch.save](https://docs.pytorch.org/docs/2.12/generated/torch.save.html)
- [PyTorch torch.load](https://docs.pytorch.org/docs/2.12/generated/torch.load.html)
- [PyTorch Module.load_state_dict](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)
