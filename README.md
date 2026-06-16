# PyTorch-Notes

这是一个用于记录 PyTorch 学习过程的笔记仓库。内容主要包括 PyTorch 基础概念、常用 API、数据处理流程、典型模型架构、训练代码框架以及学习过程中值得沉淀的实践经验。

仓库中的笔记会按章节整理到 `notes/` 目录中，方便后续复习、补充和同步到 GitHub。

## 目录结构

```text
PyTorch-Notes/
├── README.md
├── notes/
│   ├── 01-dataset.md
│   ├── 02-dataloader.md
│   ├── 03-transforms.md
│   ├── 04-module_overview.md
│   ├── 05-module_parameter.md
│   ├── 06-module_containers.md
│   ├── 07-module_layer.md
│   ├── 08-hooks.md
│   └── assets/
└── .gitignore
```

- `README.md`：仓库说明和笔记索引。
- `notes/`：按章节整理的 Markdown 学习笔记。
- `notes/assets/`：笔记中引用的图片、流程图等辅助资源。

## 笔记索引

| 章节 | 内容 |
| --- | --- |
| [Dataset](notes/01-dataset.md) | 介绍 `torch.utils.data.Dataset` 的作用、自定义数据集写法，以及 `txt`、文件夹、`csv`、`.npz` 等常见数据组织方式。 |
| [DataLoader](notes/02-dataloader.md) | 介绍 `torch.utils.data.DataLoader` 如何组织 batch、控制采样顺序、合并样本并配合训练循环使用。 |
| [Transforms](notes/03-transforms.md) | 介绍 `torchvision.transforms` 在数据预处理、数据增强、类型转换和标准化中的作用。 |
| [04-module_overview](notes/04-module_overview.md) | 介绍 `nn.Module` 的模型组织方式、子模块注册、TinnyCNN 示例和前向调用机制。 |
| [05-module_parameter](notes/05-module_parameter.md) | 介绍 `Parameter`、`_parameters`、参数容器、`parameters()`、优化器参数传递和模型参数打印方法。 |
| [06-module_containers](notes/06-module_containers.md) | 介绍 `Sequential`、`ModuleList`、`ModuleDict` 的使用场景、注册机制、前向调用方式和 AlexNet 中的容器应用。 |
| [07-module_layer](notes/07-module_layer.md) | 用于持续整理 PyTorch 常见 `nn` 网络层，按作用、参数、输入输出形状、代码实践和易错点逐章补充。 |
| [08-hooks](notes/08-hooks.md) | 介绍 PyTorch hook 机制，比较常用 Module hook 和 Tensor hook，并按作用、用法和示例说明各类 hook 的使用场景。 |

## 内容范围

本仓库计划持续整理以下内容：

- PyTorch 张量、自动求导、模型定义和训练流程。
- `Dataset`、`DataLoader`、数据预处理和数据增强。
- 常见网络层、损失函数、优化器和学习率调度。
- CNN、RNN、Transformer 等典型模型结构的 PyTorch 实现框架。
- 训练、验证、保存模型、加载模型和调试过程中的常见问题。

## 使用方式

直接阅读 `notes/` 目录下的 Markdown 文件即可。代码示例以学习和理解为主，重点说明关键 API、张量形状、前向传播流程和典型用法。

如果某一章节包含可视化资源或流程图，相关文件会放在 `notes/assets/` 中，并由对应 Markdown 文档引用。

## 更新约定

- 每个主题尽量整理为一篇独立的 Markdown 文档。
- 新增章节后，及时更新 README 中的笔记索引。
- 示例代码保持简洁、清晰，优先服务于概念理解。
- 仓库会与 GitHub 远程仓库同步更新。
