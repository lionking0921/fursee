# 洞见兽颜：基于YOLO与DINOv3的兽装身份检索与聚类框架

[山东大学计算机科学与技术学院](http://www.cs.en.qd.sdu.edu.cn/)

[吴君迪](https://scholar.google.com/citations?user=_v-8x6IAAAAJ)

[ :scroll: [`论文`](https://arxiv.org/abs/2606.22872)] [ :book:[`BibTeX引用`](#引用Fursee)] [ :earth_americas: [`English Version`](https://github.com/lionking0921/fursee/blob/main/README.md)]

## 概览

洞见兽颜（兽脸识别）系统，采用3阶段工作方式进行自动化的兽装身份检索与聚类。此框架先使用YOLO探测兽头，再把裁剪后的兽头图片送入由ArcFace调优后的DINOv3模型提取特征向量，最后使用DBSCAN聚类算法进行无监督聚类。兽脸识别模型在兽装识别与分类任务中的性能优于GPT5.5、Claude Opus 4.8和Qwen3.7-Plus模型。

![Attention Heatmap](https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/Fursee_F2.jpg)

该项目的工作流程如下:

1. **探测**: 使用特制的YOLO模型裁剪出输入图片中的所有`福瑞`目标。
2. **提取特征向量**: 对于每张裁剪出来的兽头，输入一个训练好的DINOv3模型进行特征向量提取。
3. **搜索或聚类**: 特征向量将被存储在一个轻量级的 `.fvdb`向量数据库中并用于后续的分类、身份检索或Top-K相似度排行任务。

![Workflow](https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/Fursee_F1.jpg)

### 特性

- 从图片中裁剪出兽头。
- 使用`db.py`构建并更新本地的特征向量数据库。
- 使用DBSCAN聚类算法将图片按照角色进行分类。
- 为某个或某些特定角色，挑选出属于他（们）的所有返图。
- 对于一张或多张参考图，搜索与其最相似的K张图片。
- 支持使用多GPU加速特征提取。

## 安装

### 依赖

- 需要安装Python并且版本不低于3.12
- 需要安装Conda
- 使用GPU推理，需要显卡硬件支持并且安装CUDA，CUDA版本不低于12.1

### 配置环境

1. **创建并激活虚拟环境:**

```bash
conda create -n fursee python=3.12
conda deactivate
conda activate fursee
```

2. **克隆仓库并安装相应的库:**

```bash
git clone https://github.com/lionking0921/fursee.git
cd fursee
pip install -r requirements.txt
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 下载模型

由于仓库大小有限，模型文件并不存在于本仓库中，您可通过下列任意一种方式下载权重:

- **通过**[夸克网盘](https://pan.quark.cn/s/c6f4b595d67e)**下载。**
- **加入QQ群**[1050477921](https://qm.qq.com/q/W4eKZHXDqu)**从群文件里下载。**

下载模型权重后，请将其解压到`fursee`项目目录之下。整个项目的文件架构如下所示：

```text
./fursee
├── db.py                    # 用于创建或更新特征向量数据库
├── classify.py              # 聚类脚本，对特征向量数据库中的数据进行无监督聚类
├── identify.py              # 为某个或某些特定角色，挑选出属于他（们）的所有返图
├── similar.py               # 对于一张或多张参考图，搜索与其最相似的K张图片
├── requirements.txt         # Python环境依赖
├── fursee_models/           # 模型权重文件夹
├── input/                   # 默认输入文件夹
├── output/                  # 输出文件夹
├── buffer/                  # 用于存放裁剪后的图片以及特征向量数据库的文件夹
└── utils/                   # 项目基础脚本文件
```

模型权重文件夹`fursee_models/`中的文件架构如下:

```text
fursee_models/
├── cut.pt                  # YOLO探测模型
├── model.safetensors       # DINOv3模型权重
├── config.json
└── preprocessor_config.json
```

## 运行兽脸识别

### 输入文件夹

默认情况下兽脸识别系统从以下这些文件夹中读取输入图片：

```text
input/images/       	# 主图片文件夹
input/id_targets/  		# 用于身份检索的图片，系统将找出属于这只/这些兽的所有图片
input/sim_targets/   	# 用于相似度搜索的图片
```

支持的图片格式包括 `.jpg`，`.jpeg`，`.png`，`.webp`，`.bmp`，以及 `.tiff`。

### 创建数据库

在运行任何任务之前，需要先构建向量数据库。**请先运行以下指令构建数据库。**

#### 冷构建

冷构建将销毁先前构建的数据库并建立新的数据库。如果您的图片已置于`input/images`目录下，则可直接运行：

```bash
python db.py cold
```

系统将执行以下操作：

1. 从`input/images`目录读取图片，检测兽头并把它们裁剪出来，裁剪后的图片保存在`buffer/`文件夹内。
2. 对于裁剪后的图片提取其特征向量并保存在`buffer/features.fvdb`中。

如果您的图片不在`input/images`目录下，请执行：

```bash
python db.py cold --input-folder PATH_TO_YOUR_IMAGES
```

#### 热追加

追加新图至数据库，如果新增加的图片已位于`input/images`文件夹，可执行此命令：

```bash
python db.py append
```

如不在`input/images`目录下，请执行：

```bash
python db.py append --input-folder PATH_TO_YOUR_IMAGES
```

上述命令会将新图无冲突地加入现有数据库中。

### 图片聚类

**使用场景：**按角色进行返图自动分类。

请执行：

```bash
python classify.py
```

结果保存在：

```text
output/classify/
        ├── centroids		# 每个角色的缩略图
        ├── class_0
			……				# 每个角色各自对应唯一的文件夹，存储包含该角色的所有图片
        ├── class_n
        └── special_noise	# 未被分类的图片（噪声）存于此文件夹
```

> [!TIP]
> 若一张图片中包含多个角色，则这张图片会被同时分发到图中角色所对应的文件夹里。

> [!WARNING]
> 一个角色至少需要两张图片才能形成一个类，否则将归类为噪声。

### 身份检索

**使用场景：** 在海量返图中找出属于某个或某些角色的返图。

把你要检索的角色的照片置于`input/id_targets/`文件夹下（支持多张图片），然后运行：

```bash
python identify.py
```

结果保存在：

```text
output/identify/
```

每个匹配成功的角色，将分别存放于其各自的文件夹。

> [!TIP]
> 如果查询图片中的角色不在数据库中，则不会为该角色返回任何图片。

> [!WARNING]
> 如果一张查询图片中包含多个角色，那么只取用在图片中所占面积最大的角色。

### 相似度搜索

**使用场景：** Top-K搜索功能旨在根据给定的查询图像，从海量数据库中检索出相似度最高的K张图像。

把你要进行相似度搜索的角色的照片置于 `input/sim_targets/`文件夹下（支持多张图片），然后运行：

```bash
python similar.py --k 2
```

结果按搜索的图片名保存于不同的文件夹中：

```text
output/similar/
        ├── query_a/
            ├── 001_98.21.jpg
            ├── 002_96.54.jpg
        ├── query_b/
            ├── 001_97.10.jpg
            ├── 002_93.80.jpg
```

每个输出文件名均包含其排序序号与相似度百分比。

> [!WARNING]
> 如果一张查询图片中包含多个角色，那么只取用在图片中所占面积最大的角色。

### 重置

**使用场景：** 快速清空相对应的文件夹里的内容。

该脚本支持以下清理范围：

- **`output`：**清空输出目录（`output/classify`、`output/identify`、`output/similar`）。
- **`buffer`：**清空缓存目录。
- **`all`：**清空程序使用的全部目录，包含输入缓存与输出结果。

在命令行运行该脚本，并传入目标范围作为参数。

```bash
python reset.py <scope>
```

**例如：**

清空`output`文件夹，可执行：

```bash
python reset.py output
```

> [!CAUTION]
> 请谨慎使用`reset.py`！该操作不可逆，数据不可恢复。

## 模型性能

检索与聚类实验证明，本文所提出的框架在全部评价指标上均优于GPT5.5、Claude Opus 4.8以及Qwen3.7-Plus等主流多模态模型，在兽装检索与聚类任务上取得了具备竞争力的效果。

### 检索任务实验结果

| 模型              | 命中率     |
| ----------------- | ---------- |
| GPT5.5            | 70.00%     |
| Claude Opus 4.8   | 85.00%     |
| Qwen3.7-Plus      | 46.67%     |
| **Fursee (Ours)** | **93.33%** |

### 聚类任务实验结果

| Model             | 原始准确率 | 原始召回率 | 原始F1分数 | 最终F1分数 |
| ----------------- | ---------- | ---------- | ---------- | ---------- |
| GPT5.5            | 0.7601     | 0.5834     | 0.6077     | 0.6064     |
| Claude Opus 4.8   | 0.7907     | 0.3983     | 0.4762     | 0.3956     |
| Qwen3.7-Plus      | 0.8779     | 0.6905     | 0.7442     | 0.7043     |
| **Fursee (Ours)** | **0.8986** | **0.8720** | **0.8760** | **0.8755** |

## 证书

该项目采用Fursee证书，详情请阅读[LICENSE.md](https://github.com/lionking0921/fursee/blob/main/LICENSE.md)。

## 联系方式

感谢关注本项目！ 如有任何问题或想要交流探讨，欢迎联系作者。
如果您认为该项目对您有帮助，可以通过[B站充电](https://space.bilibili.com/451573384)赞助作者。

<div align="center">
<a href="https://github.com/lionking0921"><img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/github.png" width="3%" alt="Jundi Wu GitHub"></a>
<img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/transparent.png" width="3%" alt="space">
<a href="https://qm.qq.com/q/sNf0kXUdXk"><img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/QQ.png" width="3%" alt="Jundi Wu QQ"></a>
<img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/transparent.png" width="3%" alt="space">
<a href="https://space.bilibili.com/451573384"><img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/bilibili.png" width="3%" alt="Jundi Wu Bilibili"></a>
<img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/transparent.png" width="3%" alt="space">
<a href="https://v.douyin.com/h9RFABiVMuI"><img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/douyin.png" width="3%" alt="Jundi Wu Douyin"></a>
</div>

## 引用Fursee

若您在研究工作中使用Fursee，请引用以下BibTeX条目。

```text
@misc{wu2026fursee,
  title={Fursee: Hybrid YOLO-DINOv3 Framework for Fursuit Identity Retrieval and Clustering},
  author={Wu, Jundi},
  year={2026},
  eprint={2606.22872},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2606.22872},
}
```
