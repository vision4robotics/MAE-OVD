# MAE-OVD Configuration

## Directory Structure

- **backbone/**: Lightweight backbone configuration (channels: F3=64, F4=128, F5=256)
- **pretrain/**: Pretraining configuration (YOLO-World style, detection data)
- **default.py**: Default pretraining settings with MAE-OVD components

## Component Configuration

### Backbone
- Input channels: (64, 128, 256) for F3, F4, F5
- Text embedding dimension: 256
- YOLO-World backbone wrapper available

### MAE Pretraining
- Mask ratio: 0.4 (joint detector training), 0.5 (MAE-focused pretraining)
- MAE loss weight λ: starts at 0.1, max 0.2
- MAE activation probability: 0.35 per batch
- Warmup epochs: 1

### BiTL-PAN
- In channels: (64, 128, 256)
- Text dimension: 256

### IFSD
- Projected dimension: 256
- Aligned size: 20×20
- Text dimension: 256

## Usage

**Environment**: Activate conda environment before training:
```bash
conda activate mae-ovd
```

**Pretraining**:
```bash
python tools/train_pretrain.py --config configs/pretrain/default.py
```

## Data Configuration

Set your data paths in config files:
- `data_root`: Root directory of dataset
- `ann_file`: Path to annotation file
- `data_prefix`: Image path prefix
