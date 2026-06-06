# Dataset

## Dataset 的功能

`torch.utils.data.Dataset` 是 PyTorch 提供的数据集抽象基类。实际项目中通常需要继承 `Dataset`，把磁盘上的图片、标签和其他元信息整理成模型可读取的样本。一个自定义 `Dataset` 至少要实现两个方法：

- `__getitem__(self, index)`：根据索引读取一个样本。通常先根据 `index` 找到图片路径和标签，再从磁盘读取图片，执行预处理或数据增强，最后返回一个样本，例如 `(image, label)`。
- `__len__(self)`：返回数据集样本数量。`DataLoader` 会依赖这个长度计算采样范围。如果返回 0，常见原因是路径写错、标注文件为空、目录结构不符合代码预期。

`Dataset` 负责和磁盘建立联系：它知道样本在哪里、标签在哪里、如何读取单个样本。`DataLoader` 负责把多个样本组织成 batch，并处理 shuffle、sampler、多进程读取等批量加载逻辑。

![Dataset 与 DataLoader 的关系](assets/dataset-dataloader-flow.svg)

上图可以理解为：磁盘数据先由 `Dataset` 读取并预处理，`DataLoader` 再多次调用 `Dataset.__getitem__`，把单个样本组装成 `Batch Data`，最后送入模型。采样规则体现在传入 `__getitem__` 的索引中，后续可以通过 `DataLoader` 的 `shuffle` 或 `sampler` 控制随机采样、均衡采样、有偏采样等策略。

本节先用 COVID-19 X 光分类数据集说明三种典型的“图片路径 + 标签”组织方式，再补充一种常见的数组打包方式 `.npz`：

- 数据划分和标签写在 `txt` 文件中。
- 数据划分和标签通过文件夹结构体现。
- 数据划分和标签写在 `csv` 文件中。
- 图像和标签提前打包在 `.npz` 文件中。

前三种方式的共同目标都是构建 `self.img_info` 这样的列表：

```python
self.img_info = [
    (path_img_0, label_0),
    (path_img_1, label_1),
]
```

这样 `__getitem__` 只需要根据索引取出路径和标签，再读取图片并返回。

## Dataset 的通用模板

前三个案例的主体结构基本一致，差异主要集中在 `_get_img_info()` 如何收集图片路径和标签。`.npz` 方式则通常不需要维护路径列表，而是直接维护数组。

```python
from torch.utils.data import Dataset
from PIL import Image


class COVID19Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception(f"data_dir:{self.root_dir} is a empty dir! Please checkout your path to images!")
        return len(self.img_info)

    def _get_img_info(self):
        raise NotImplementedError
```

这里的关键点是：

- `root_dir` 表示图片根目录或数据集根目录。
- `transform` 用于图像预处理，例如 resize、转 tensor、normalize。
- `img_info` 是 `Dataset` 内部维护的样本索引表。
- `__getitem__` 返回的是一个样本；如果配合 `DataLoader(batch_size=2)`，两个样本会被自动拼成一个 batch。

## 案例一：数据划分及标签在 txt 中

这种方式把图片放在图片目录中，把训练集、验证集的样本清单分别写入 `train.txt` 和 `valid.txt`。代码来自 `PyTorch-Tutorial-2nd-1.0.0/code/chapter-2/02_COVID_19_cls.py`。

### 数据集目录结构

```text
covid-19-demo/
├── imgs/
│   ├── covid-19/
│   │   ├── ryct.2020200028.fig1a.jpeg
│   │   └── auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg
│   └── no-finding/
│       ├── 00001215_000.png
│       └── 00001215_001.png
└── labels/
    ├── train.txt
    └── valid.txt
```

`train.txt` 示例：

```text
covid-19/ryct.2020200028.fig1a.jpeg 0 1
no-finding/00001215_001.png 1 0
```

这里每行表示一个样本。示例代码使用第 1 列作为相对图片路径，使用第 3 列作为类别标签：

```python
self.img_info = [
    (os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
    for i in txt_data
]
```

### Dataset 核心代码

```python
class COVID19Dataset(Dataset):
    def __init__(self, root_dir, txt_path, transform=None):
        self.root_dir = root_dir
        self.txt_path = txt_path
        self.transform = transform
        self.img_info = []
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception(f"data_dir:{self.root_dir} is a empty dir! Please checkout your path to images!")
        return len(self.img_info)

    def _get_img_info(self):
        with open(self.txt_path, "r") as f:
            txt_data = f.read().strip().split("\n")

        self.img_info = [
            (os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
            for i in txt_data
        ]
```

### 适用场景与注意点

- 适合数据划分已经固定的实验：训练集和验证集分别由 `train.txt`、`valid.txt` 控制。
- 图片实际目录可以保持不变，改变划分时只需要改标注文件。
- 代码和标注文件的列含义必须一致。当前示例假设 `i.split()[2]` 是标签，如果标签在第 2 列，就要同步修改索引。
- `root_dir` 应传入图片根目录，例如 `covid-19-demo/imgs`，不是整个数据集根目录。

使用方式：

```python
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

img_dir = "covid-19-demo/imgs"
path_txt_train = "covid-19-demo/labels/train.txt"

transforms_func = transforms.Compose([
    transforms.Resize((8, 8)),
    transforms.ToTensor(),
])

train_data = COVID19Dataset(
    root_dir=img_dir,
    txt_path=path_txt_train,
    transform=transforms_func,
)
train_loader = DataLoader(dataset=train_data, batch_size=2)
```

如果原图是灰度图，`Image.open(path_img).convert("L")` 后再 `ToTensor()`，单张图片形状通常是 `[1, H, W]`。`DataLoader` 组 batch 后形状变为 `[B, 1, H, W]`。

## 案例二：数据划分及标签在文件夹中体现

这种方式不再单独写标注文件，而是通过目录名表达训练集、验证集和类别。代码来自 `PyTorch-Tutorial-2nd-1.0.0/code/chapter-3/01_dataset.py` 中的 `COVID19Dataset_2`。

### 数据集目录结构

```text
covid-19-dataset-2/
├── train/
│   ├── covid-19/
│   │   └── ryct.2020200028.fig1a.jpeg
│   └── no-finding/
│       └── 00001215_001.png
└── valid/
    ├── covid-19/
    │   └── auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg
    └── no-finding/
        └── 00001215_000.png
```

在这个结构中：

- `train/` 和 `valid/` 表示数据划分。
- `covid-19/` 和 `no-finding/` 表示类别名称。
- 图片路径本身不直接包含数值标签，需要把类别名映射成整数标签。

### Dataset 核心代码

```python
class COVID19Dataset_2(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []
        self.str_2_int = {"no-finding": 0, "covid-19": 1}
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception(f"data_dir:{self.root_dir} is a empty dir! Please checkout your path to images!")
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpeg"):
                    path_img = os.path.join(root, file)
                    sub_dir = os.path.basename(root)
                    label_int = self.str_2_int[sub_dir]
                    self.img_info.append((path_img, label_int))
```

### 适用场景与注意点

- 适合图像分类数据集，尤其是目录已经按类别整理好的情况。
- 每次构造数据集时，`root_dir` 直接传入某个划分目录，例如 `covid-19-dataset-2/train` 或 `covid-19-dataset-2/valid`。
- 类别目录名必须出现在 `str_2_int` 中，否则会在 `self.str_2_int[sub_dir]` 处报错。
- 当前代码只识别以 `"png"` 或 `"jpeg"` 结尾的文件，不会读取 `"jpg"`。如果数据集中有 `.jpg`，需要扩展判断条件。

使用方式：

```python
root_dir_train = "covid-19-dataset-2/train"
root_dir_valid = "covid-19-dataset-2/valid"

train_set = COVID19Dataset_2(root_dir_train)
valid_set = COVID19Dataset_2(root_dir_valid)
```

这种方式的优点是直观，不需要额外标注文件；缺点是数据划分和标签绑定在目录结构中，调整训练集、验证集时要移动文件或重新组织目录。

## 案例三：数据划分及标签在 csv 中

这种方式把所有图片放在一个目录中，用一份 `csv` 同时记录图片名、标签和所属划分。代码来自 `PyTorch-Tutorial-2nd-1.0.0/code/chapter-3/01_dataset.py` 中的 `COVID19Dataset_3`。

### 数据集目录结构

```text
covid-19-dataset-3/
├── imgs/
│   ├── auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg
│   ├── ryct.2020200028.fig1a.jpeg
│   ├── 00001215_000.png
│   └── 00001215_001.png
└── dataset-meta-data.csv
```

`dataset-meta-data.csv` 示例：

```csv
img-name,label,set-type,etc
ryct.2020200028.fig1a.jpeg,1,train,
00001215_001.png,0,train,
00001215_000.png,0,valid,
auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg,1,valid,
```

在这个结构中：

- `img-name` 表示图片文件名。
- `label` 表示整数标签。
- `set-type` 表示样本属于 `train` 还是 `valid`。

### Dataset 核心代码

```python
class COVID19Dataset_3(Dataset):
    def __init__(self, root_dir, path_csv, mode, transform=None):
        self.root_dir = root_dir
        self.path_csv = path_csv
        self.mode = mode
        self.transform = transform
        self.img_info = []
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception(f"data_dir:{self.root_dir} is a empty dir! Please checkout your path to images!")
        return len(self.img_info)

    def _get_img_info(self):
        df = pd.read_csv(self.path_csv)
        df.drop(df[df["set-type"] != self.mode].index, inplace=True)
        df.reset_index(inplace=True)

        for idx in range(len(df)):
            path_img = os.path.join(self.root_dir, df.loc[idx, "img-name"])
            label_int = int(df.loc[idx, "label"])
            self.img_info.append((path_img, label_int))
```

### 适用场景与注意点

- 适合样本元信息较多的场景，例如除了图片名和标签，还要记录病人 ID、来源、检查时间、额外属性等。
- 训练集和验证集不再靠目录区分，而是靠 `mode` 过滤：`COVID19Dataset_3(root_dir, path_csv, "train")` 只读取 `set-type == "train"` 的样本。
- `df.reset_index(inplace=True)` 很重要。`drop` 删除行后，原始索引可能不连续；后面用 `df.loc[idx, ...]` 按 `0, 1, 2, ...` 访问时，需要先重置索引。
- 这种方式需要 `pandas`，相比 `txt` 和文件夹方式多引入一个依赖。

使用方式：

```python
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

root_dir = "covid-19-dataset-3/imgs"
path_csv = "covid-19-dataset-3/dataset-meta-data.csv"

normalize = transforms.Normalize([0.5], [0.5])
transforms_train = transforms.Compose([
    transforms.Resize((4, 4)),
    transforms.ToTensor(),
    normalize,
])

train_set = COVID19Dataset_3(root_dir, path_csv, "train", transform=transforms_train)
train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)

for i, (inputs, target) in enumerate(train_loader):
    print(i, inputs.shape, target.shape)
```

这里 `inputs` 的形状通常是 `[B, 1, 4, 4]`，`target` 的形状通常是 `[B]`。`B` 由 `batch_size` 决定。

## 案例四：数据和标签在 npz 中

`.npz` 是 NumPy 提供的压缩打包格式，可以把多个数组存到同一个文件中。对于已经预处理好的小型或中型数据集，可以把图像数组和标签数组分别保存为 `images`、`labels`，再由 `Dataset` 按索引读取。

这种方式和前三种有一个关键区别：前三种通常在 `__getitem__` 中根据路径从磁盘读取图片；`.npz` 通常在初始化时读取数组，然后在 `__getitem__` 中直接按下标取数组。

### 数据集目录结构

```text
covid-19-npz/
├── train.npz
└── valid.npz
```

`train.npz` 内部可以包含：

```text
images: [N, H, W] 或 [N, C, H, W]
labels: [N]
```

其中：

- `N` 表示样本数量。
- `H, W` 表示图像高度和宽度。
- `C` 表示通道数。灰度图通常是 `1`，RGB 图通常是 `3`。
- `labels` 中的每个元素是对应样本的整数标签。

### Dataset 核心代码

```python
import numpy as np
import torch
from torch.utils.data import Dataset


class COVID19NPZDataset(Dataset):
    def __init__(self, path_npz, transform=None):
        self.path_npz = path_npz
        self.transform = transform

        data = np.load(self.path_npz)
        self.images = data["images"]
        self.labels = data["labels"]
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])

        image = torch.as_tensor(image, dtype=torch.float32)

        if image.ndim == 2:
            image = image.unsqueeze(0)

        if image.max().item() > 1:
            image = image / 255.0

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)
```

使用方式：

```python
from torch.utils.data import DataLoader

train_set = COVID19NPZDataset("covid-19-npz/train.npz")
valid_set = COVID19NPZDataset("covid-19-npz/valid.npz")

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
```

如果 `images` 的形状是 `[N, H, W]`，单个样本取出后是 `[H, W]`，代码中会用 `unsqueeze(0)` 补成 `[1, H, W]`。如果 `images` 本来就是 `[N, C, H, W]`，取出的单个样本已经是 `[C, H, W]`，不需要再补通道维度。

### 适用场景与注意点

- 适合数据已经提前预处理成数组的情况，例如医学影像切片、表格特征、仿真数据、较小的图像数据集。
- 目录结构简单，训练集和验证集可以分别保存为 `train.npz`、`valid.npz`。
- 读取逻辑比图片路径方式更短，不需要 `PIL.Image.open()`。
- 如果 `.npz` 文件很大，初始化时一次性加载数组可能占用较多内存。超大数据集更适合使用 HDF5、LMDB，或继续使用路径索引方式按需读取。
- `torchvision.transforms` 中有些变换面向 PIL 图像，有些可以直接处理 Tensor。使用 `.npz` 时要确认 `transform` 的输入类型是否匹配。

## 四种 Dataset 组织方式的关键差异

| 组织方式 | 数据划分在哪里 | 标签在哪里 | 构造 Dataset 时需要什么 | `_get_img_info()` 的核心逻辑 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| `txt` 文件 | `train.txt`、`valid.txt` | `txt` 每行的指定列 | 图片根目录、对应 `txt` 路径 | 逐行读取 `txt`，解析相对路径和标签 | 划分固定、标注文件简单 |
| 文件夹 | `train/`、`valid/` 目录 | 类别子目录名 | 当前划分的根目录 | 遍历目录，用父目录名映射标签 | 图像分类目录已整理好 |
| `csv` 文件 | `set-type` 字段 | `label` 字段 | 图片根目录、`csv` 路径、`mode` | 读取表格，按 `mode` 过滤，再解析图片名和标签 | 元信息较多、划分灵活 |
| `.npz` 文件 | 通常由 `train.npz`、`valid.npz` 区分 | `labels` 数组 | `.npz` 文件路径 | 通常不需要 `_get_img_info()`，直接读取 `images` 和 `labels` 数组 | 数据已预处理成数组 |

从代码结构看，`txt`、文件夹、`csv` 三种方式的 `__getitem__` 和 `__len__` 基本相同，真正变化的是“如何从磁盘结构或标注文件中建立 `img_info`”。`.npz` 方式则把样本提前整理成数组，变化点是“如何从数组中取出单个样本”。因此写自定义 `Dataset` 时，可以先想清楚三个问题：

- 每个样本的图片路径从哪里来？
- 每个样本的标签从哪里来？
- 训练集、验证集、测试集的划分从哪里来？

只要这三个问题明确，就可以选择合适的数据组织方式：路径型数据通常整理成 `(path_img, label)` 列表，数组型数据通常直接保存为 `images` 和 `labels`。

## 常见错误

- 路径传错：例如 `txt` 方式中 `root_dir` 应该指向 `imgs/`，如果传成数据集总目录，拼出来的图片路径会多一层或少一层。
- 标签列读错：`i.split()[2]` 表示读取第 3 列，必须和标注文件格式一致。
- 类别映射漏写：文件夹方式中，目录名必须能在 `str_2_int` 字典里找到。
- 图片后缀不匹配：如果代码只判断 `.png` 和 `.jpeg`，数据里的 `.jpg` 不会被加入数据集。
- 没有使用 `transform`：如果直接返回 `PIL.Image.Image`，模型不能直接训练；通常至少需要 `transforms.ToTensor()`。
- 数据集长度为 0：大概率是路径、文件名、后缀、划分字段或过滤条件不匹配。
- `.npz` 维度不符合模型输入：例如灰度图保存为 `[N, H, W]` 时，要在 `__getitem__` 中补成 `[1, H, W]`。
- `.npz` 数值范围不合适：如果图像数组是 `uint8` 的 `0-255`，通常要转成 `float32` 并缩放到 `0-1`。
