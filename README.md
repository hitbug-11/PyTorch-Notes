# PyTorch-Notes

这是一个用于记录 PyTorch 学习过程的笔记仓库。内容主要包括 PyTorch 基础概念、常用 API、数据处理流程、典型模型架构、训练代码框架以及学习过程中值得沉淀的实践经验。

仓库中的笔记会按章节整理到 `notes/` 目录中，方便后续复习、补充和同步到 GitHub。

## 目录结构

```text
PyTorch-Notes/
├── README.md
├── notes/
│   ├── dataset.md
│   └── assets/
└── .gitignore
```

- `README.md`：仓库说明和笔记索引。
- `notes/`：按章节整理的 Markdown 学习笔记。
- `notes/assets/`：笔记中引用的图片、流程图等辅助资源。

## 笔记索引

| 章节 | 内容 |
| --- | --- |
| [Dataset](notes/dataset.md) | 介绍 `torch.utils.data.Dataset` 的作用、自定义数据集写法，以及 `txt`、文件夹、`csv`、`.npz` 等常见数据组织方式。 |

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
