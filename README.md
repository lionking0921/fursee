# Fursee: Hybrid YOLO-DINOv3 Framework for Fursuit Identity Retrieval and Clustering

[School of Computer Science and Technology, Shandong University](http://www.cs.en.qd.sdu.edu.cn/)

[Jundi Wu](https://scholar.google.com/citations?user=_v-8x6IAAAAJ)

[ :scroll: [`Paper`](https://arxiv.org/abs/2606.22872)] [ :book:[`BibTeX`](#citing-fursee)] [ :earth_asia: [`中文版`](https://github.com/lionking0921/fursee/blob/main/README-zh.md)]

## Overview

Fursee is a three-stage pipeline for fursuit identity retrieval and clustering. It utilizes YOLO for detection, ArcFace-optimized DINOv3 for embeddings, and DBSCAN for unsupervised grouping. Outperforms GPT-5.5, Claude Opus 4.8, and Qwen3.7-Plus on fursuit benchmarks.

![Attention Heatmap](https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/Fursee_F2.jpg)

The project uses a staged vision pipeline:

1. **Detection**: a YOLO model crops detected `furry` targets from input images.
2. **Embedding**: a DINOv3 model converts crops into normalized feature vectors.
3. **Search or clustering**: feature vectors are stored in a lightweight `.fvdb` vector database and used for classification, Top-K similarity search, or identity lookup.

![Workflow](https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/Fursee_F1.jpg)

### Features

- Detect and crop furry faces from images.
- Build and update a local feature database with `db.py`.
- Cluster images into identity-like classes with DBSCAN.
- Identify images that belong to one or more reference targets.
- Search for the Top-K images most similar to one or more reference images.
- Supports CUDA acceleration and multi-GPU feature extraction when available.

## Installation

### Requirements

- Python 3.12 or higher
- Conda
- CUDA-compatible GPU with CUDA 12.1 or higher

### Environment Setup

1. **Create and activate a virtual environment:**

```bash
conda create -n fursee python=3.12
conda deactivate
conda activate fursee
```

2. **Clone the repository and install the package:**

```bash
git clone https://github.com/lionking0921/fursee.git
cd fursee
pip install -r requirements.txt
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Download Model

Model weights are not distributed with this repository. You can choose either of the two ways to fetch the models:

- **Download via** [Quark Cloud Disk](https://pan.quark.cn/s/c6f4b595d67e)**.**
- **Download by joining our QQ group** [1050477921](https://qm.qq.com/q/W4eKZHXDqu) to download the weights **from the group files.**

After downloading the weight files, please extract them to the `fursee` project directory. The structure of the entire project is shown below.

```text
./fursee
├── db.py                    # Build/update the feature DB
├── classify.py              # Cluster images using an existing feature DB
├── identify.py              # Identify images matching target identities
├── similar.py               # Top-K similarity search from one or more reference images
├── requirements.txt         # Python dependencies
├── fursee_models/           # Local detection and embedding model assets
├── input/                   # Default input image folders
├── output/                  # Generated results
├── buffer/                  # Cropped images and feature database cache
└── utils/                   # Detection, embedding, clustering, and vector DB utilities
```

Fursee expects local model assets in the `fursee_models/` directory:

```text
fursee_models/
├── cut.pt                  # YOLO detection model
├── model.safetensors       # DINOv3 model weights
├── config.json
└── preprocessor_config.json
```

## Run Fursee

### Input Folders

By default, FurSee reads images from these folders:

```text
input/images/       	# Main image collection
input/id_targets/  		# One or more reference images for identity search
input/sim_targets/   	# One or more reference images for similarity search
```

Supported image formats include `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, and `.tiff`.

### Create Database

A vector database is required before carrying out any task. **Please run the flowing commands to construct the database first.**

#### Cold start

Cold start mode will clear the previous database and then build the new database. If the original images are placed in the `input/images` directory, run the following command:

```bash
python db.py cold
```

This will:

1. Detect and crop targets from `input/images/` into `buffer/`.
2. Extract embeddings into `buffer/features.fvdb`.

If the photos are from a separate folder, please run:

```bash
python db.py cold --input-folder PATH_TO_YOUR_IMAGES
```

#### Append mode

To add new images to the existing database, run this command if the new images are located in the `input/images` directory:

```bash
python db.py append
```

Otherwise, run:

```bash
python db.py append --input-folder PATH_TO_YOUR_IMAGES
```

The commands above will append new feature vectors to database without conflicts.

### Classify / Cluster Images

**Usage scenarios:** Automatically categorize furry images by role.

Run:

```bash
python classify.py
```

Results are copied to:

```text
output/classify/
        ├── centroids		# Thumbnails for each class
        ├── class_0
			……				# Images grouped by class
        ├── class_n
        └── special_noise	# Unclassified images
```

> [!TIP]
> If an image contains multiple characters, the image will be distributed simultaneously to the folders corresponding to each character present in the image.

> [!WARNING]
> A character requires at least two images to form a class. Otherwise, it will be classified as noise.

### Identity Retrieval

**Usage scenarios:** Users can quickly find photos of specific character without scrolling through thousands of images. 

Place one or more target reference images in `input/id_targets/`, then run:

```bash
python identify.py
```

Results are copied to:

```text
output/identify/
```

Each matched identity is placed in its own folder.

> [!TIP]
> If the character in the query image is not present in the database, no images will be returned for that character.

> [!WARNING]
> When a query image includes multiple characters, only the one with the largest area in the image is selected.

### Similarity Search

**Usage scenarios:** The Top-K search function is designed to retrieve the K most similar images from a massive database based on a given query image.

Place one or more reference images in `input/sim_targets/`, then run:

```bash
python similar.py --k 2
```

Results are grouped by query image name:

```text
output/similar/
        ├── query_a/
            ├── 001_98.21.jpg
            ├── 002_96.54.jpg
        ├── query_b/
            ├── 001_97.10.jpg
            ├── 002_93.80.jpg
```

Each output filename includes its rank and similarity percentage.

> [!WARNING]
> When a query image includes multiple characters, only the one with the largest area in the image is selected.

### Reset

**Usage scenarios:** Quickly clear the contents of the corresponding folder.

The script supports the following cleaning scopes:

- **`output`:** Clears the output directories (`output/classify`, `output/identify`, `output/similar`).
- **`buffer`:** Clears the buffer directory.
- **`all`:** Clears all directories used by the application, including input buffers and output results.

Run the script from the command line, passing the desired scope as an argument.

```bash
python reset.py <scope>
```

**Example:**

To clear the `output` directories, run:

```bash
python reset.py output
```

> [!CAUTION]
> Use `reset.py` carefully! This operation is irreversible. All erased data is unrecoverable.

## Model Performance

Retrieval and clustering experiments verify that our pipeline outperforms mainstream multimodal models including GPT5.5, Claude Opus 4.8 and Qwen3.7-Plus on all evaluation metrics, achieving competitive performance for fursuit head retrieval and grouping.

### Retrieval Experiment Results

| Model             | Hit Rate   |
| ----------------- | ---------- |
| GPT5.5            | 70.00%     |
| Claude Opus 4.8   | 85.00%     |
| Qwen3.7-Plus      | 46.67%     |
| **Fursee (Ours)** | **93.33%** |

### Clustering Experiment Results

| Model             | $Precision_{raw}$ | $Recall_{raw}$ | $F1_{raw}$ | $F1_{final}$ |
| ----------------- | ------------- | ---------- | ---------- | ---------- |
| GPT5.5            | 0.7601        | 0.5834     | 0.6077     | 0.6064     |
| Claude Opus 4.8   | 0.7907        | 0.3983     | 0.4762     | 0.3956     |
| Qwen3.7-Plus      | 0.8779        | 0.6905     | 0.7442     | 0.7043     |
| **Fursee (Ours)** | **0.8986**    | **0.8720** | **0.8760** | **0.8755** |

## License

This project is licensed under the Fursee License - see the [LICENSE.md](https://github.com/lionking0921/fursee/blob/main/LICENSE.md) file for details.

## Contact

Thanks for checking out this project!
Feel free to reach out to the author for any questions or discussions.
If you find this project helpful, consider supporting the author via a [Bilibili Recharge](https://space.bilibili.com/451573384)!

<div align="center">
<a href="https://github.com/lionking0921"><img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/github.png" width="3%" alt="Jundi Wu GitHub"></a>
<img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/transparent.png" width="3%" alt="space">
<a href="https://qm.qq.com/q/sNf0kXUdXk"><img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/QQ.png" width="3%" alt="Jundi Wu QQ"></a>
<img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/transparent.png" width="3%" alt="space">
<a href="https://space.bilibili.com/451573384"><img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/bilibili.png" width="3%" alt="Jundi Wu Bilibili"></a>
<img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/transparent.png" width="3%" alt="space">
<a href="https://v.douyin.com/h9RFABiVMuI"><img src="https://raw.githubusercontent.com/lionking0921/wjdpicture/refs/heads/main/images/douyin.png" width="3%" alt="Jundi Wu Douyin"></a>
</div>

## Citing Fursee

If you use Fursee in your research, please use the following BibTeX entry.

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
