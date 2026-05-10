# MAE-OVD

> **MAE-Powered Open-Vocabulary Vision-Language Model for Anti-UAV Detection**
>
> **[ICARM 2026](https://icarm2026.org/)** | [ArXiv](https://arxiv.org/) | [BibTeX](#citation)

[![Conference](https://img.shields.io/badge/Conference-ICARM%202026-blue)](https://icarm2026.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-orange.svg)](https://www.python.org/)

---

## Table of Contents

- [Highlights](#highlights)
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Model Zoo](#model-zoo)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgement](#acknowledgement)

---

## Highlights

- **Novel MAE-Driven Framework**: First work to apply Masked Autoencoding for anti-UAV open-vocabulary detection
- **Degradation-Resistant**: Effectively handles motion blur, tiny objects, low SNR, and background clutter
- **Lightweight Inference**: Decoupled train-test pipeline enables efficient deployment
- **State-of-the-Art**: Outperforms existing OVD methods across all five UAV categories

---

## Overview

Open-vocabulary detection (OVD) is valuable for anti-unmanned aerial vehicle (UAV) applications, as natural-language referring enables rapid, precise instance-level localization. However, existing OVD methods face significant challenges in UAV views due to:

- Motion blur from fast-moving targets
- Tiny objects at long distances
- Low signal-to-noise ratio
- Heavy background clutter

MAE-OVD addresses these challenges with three key innovations:

| Component | Description |
|-----------|-------------|
| **MAE-Driven Backbone** | Reconstructs partially corrupted UAV observations and mines degradation-resistant global structural cues |
| **BiTL-PAN** | Bi-directional Text-Linked Path Aggregation Network for multi-scale text-visual interaction |
| **IFSD** | Implicit Feature Semantic Distillation module to purify alignment responses |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAE-OVD Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Train Phase:                                                                │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────────────────┐  │
│  │   MAE    │   │   Text      │   │  BiTL-   │   │        IFSD         │  │
│  │  Branch  │──▶│  Encoder    │──▶│   PAN    │──▶│  (Semantic          │  │
│  │(Auxiliary)│  │  (Frozen)   │   │          │   │   Purification)     │  │
│  └──────────┘   └──────────────┘   └──────────┘   └─────────────────────┘  │
│                                                          │                  │
│                                                          ▼                  │
│                                                ┌─────────────────────┐      │
│                                                │      Box Head       │      │
│                                                │    (Frozen/Head)    │      │
│                                                └─────────────────────┘      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Test Phase (Lightweight Inference):                                         │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────────────────┐  │
│  │  Image   │   │  BiTL-IFSD  │   │      Box Head (Frozen)            │  │
│  │ Encoder  │──▶│   Aligner   │──▶│    Zero-Shot Localization         │  │
│  └──────────┘   └──────────────┘   └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **MAE-Driven Backbone**: Degradation-resistant representation learning via masked autoencoding
2. **BiTL-PAN**: U-shaped bidirectional interaction hub with IR-IP (Instance-Refined Implicit Prompting)
3. **IFSD**: Semantic purification hub for suppressing distractors (birds, clouds, buildings)

---

## Installation

### Prerequisites

| Component | Version |
|-----------|--------|
| Ubuntu | 18.04 / 20.04 |
| Python | 3.8+ |
| PyTorch | 2.1.2 |
| CUDA | 11.x |

### Using Conda (Recommended)

```bash
# Clone the conda environment from an existing YOLO-World environment
conda create -n mae-ovd --clone yolo_world_vlt
conda activate mae-ovd

# Install the package
cd MAE-OVD
pip install -e .
```

### Manual Installation

```bash
# Create a new conda environment
conda create -n mae-ovd python=3.8
conda activate mae-ovd

# Install PyTorch
pip install torch==2.1.2 torchvision==0.16.2

# Install dependencies
pip install transformers tokenizers numpy opencv-python
pip install supervision>=0.19.0 openmim
pip install mmcv-lite==2.0.1 mmdet==3.3.0 mmengine==0.10.4 mmcv==2.1.0

# Install this package
pip install -e .

# Optional: Install development dependencies
pip install pytest onnx onnxruntime onnxsim
```

---

## Quick Start

### 1. Pretraining

```bash
# Default configuration
python tools/train_pretrain.py --config configs/pretrain/default.py

# YOLO-World based configuration
python tools/train_pretrain.py --config configs/pretrain/yolo_world_bitl_finetune.py

# With custom hyperparameters
python tools/train_pretrain.py \
    --config configs/pretrain/default.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 2e-4
```

### 2. Testing

```bash
# Activate environment
conda activate mae-ovd
cd MAE-OVD

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_bitl_pan.py -v
pytest tests/test_imd.py -v
```

### 3. ONNX Export

```bash
python tools/export_onnx.py \
    --checkpoint work_dirs/pretrain/default/epoch_100.pth \
    --output mae_ovd.onnx
```

---

## Dataset Preparation

### COCO Dataset

```bash
# Download COCO 2017
mkdir -p datasets/coco
cd datasets/coco

# Download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### Update Configuration

Modify `configs/pretrain/default.py`:

```python
data_root = "/path/to/your/datasets/coco"
ann_file = "annotations/instances_train2017.json"
data_prefix = dict(img="train2017/")
```

---

## Training

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_epochs` | 100 | Total training epochs |
| `base_lr` | 2e-4 | Base learning rate |
| `batch_size` | 16 | Batch size per GPU |
| `mae_mask_ratio` | 0.4 | MAE masking ratio |
| `mae_lambda` | 0.1 | MAE loss weight |
| `weight_decay` | 0.05 | Weight decay |

### MAE Pretraining Strategy

- **Warmup**: First 1 epoch with MAE activation
- **Joint Training**: MAE branch activated with probability 0.35 per batch
- **Decay**: Last 2 epochs disable MAE branch and Mosaic augmentation

---

## Model Zoo

| Model | Backbone | Description |
|-------|----------|-------------|
| MAE-OVD-S | YOLOv8-S | Small model for real-time applications |
| MAE-OVD-L | YOLOv8-L | Large model for best accuracy |

### UAV-Anti-UAV Benchmark Results (COCO-style, 5 categories)

| Category | MAE-OVD-S | MAE-OVD-L | Best Previous |
|----------|-----------|-----------|---------------|
| Fixed_wing (AvgAP) | 0.313 | **0.376** | 0.294 (GLIP) |
| FPV (AvgAP) | 0.077 | **0.297** | 0.136 (M-YOLO-W-S) |
| Multi_rotor (AvgAP) | 0.122 | **0.384** | 0.284 (GroundingDINO) |
| VTOL (AvgAP) | 0.397 | **0.693** | 0.608 (GroundingDINO) |
| Helicopter (AvgAP) | 0.356 | **0.588** | 0.429 (GLIP) |

See the [paper](./docs/root.tex) for detailed experimental results.

---

## Project Structure

```
MAE-OVD/
├── mae_ovd/                    # Main package
│   ├── models/
│   │   ├── backbone/           # Visual encoders
│   │   │   ├── lightweight_encoder.py   # Lightweight backbone
│   │   │   ├── text_encoder.py          # CLIP text encoder
│   │   │   └── yolo_world_backbone.py   # YOLO-World backbone wrapper
│   │   ├── bitl_pan/           # BiTL-PAN modules
│   │   │   ├── bitl_pan.py     # Main BiTL-PAN implementation
│   │   │   ├── ir_ip.py        # Instance-Refined Implicit Prompting
│   │   │   └── t_ssg.py        # Text-guided Spatial Gating
│   │   ├── imd/                # IFSD modules
│   │   │   ├── ifsd.py         # Implicit Feature Semantic Distillation
│   │   │   ├── fusion.py       # Feature Fusion
│   │   │   ├── grounding.py    # Semantic Grounding
│   │   │   └── template_extractor.py  # Template Extraction
│   │   ├── mae/                # MAE decoder
│   │   │   └── mae_decoder.py  # MAE reconstruction decoder
│   │   └── yolo_world_det_head.py  # YOLO-World detection head
│   ├── losses/                 # Loss functions
│   ├── datasets/               # Dataset interfaces
│   └── utils/                  # Utilities
├── configs/                     # Configuration files
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── LICENSE                    # Apache 2.0 License
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{maeovd2026icarm,
  author    = {Changhong Fu and Zhangchi Guo and Liangliang Yao and
               Haobo Zuo and Mengyuan Li and Yipeng Feng},
  title     = {MAE-OVD: MAE-Powered Open-Vocabulary Vision-Language Model for Anti-UAV Detection},
  booktitle = {International Conference on Advanced Robotics and Mechatronics (ICARM)},
  year      = {2026},
  address   = {TBD},
  publisher = {IEEE}
}
```

---

## License

This project is released under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgement

This project is built upon [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and utilizes the [CLIP](https://github.com/openai/CLIP) text encoder. We thank the original authors for their contributions.

**Funding**: This work was supported by the National Natural Science Foundation of China (62173249) and Natural Science Foundation of Shanghai (20ZR1460100).

---

## Contact

For questions, issues, or collaborations, please contact:

- **Changhong Fu**: [changhongfu@tongji.edu.cn](mailto:changhongfu@tongji.edu.cn)
- **Zhangchi Guo**: [Open an Issue](https://github.com/vision4robotics/MAE-OVD/issues)

---

<p align="center">
  <img src="https://visitor-badge.la玄?ime=vision4robotics/MAE-OVD" alt="Visitor Badge" />
</p>
