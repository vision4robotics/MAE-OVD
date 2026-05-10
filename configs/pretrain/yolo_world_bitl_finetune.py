# MAE-OVD Pretraining Configuration
# Based on YOLO-World backbone with frozen bbox_head, BiTL-PAN fine-tuning
#
# Core Idea:
#   - Reuse YOLO-World pretrained backbone (YOLOv8CSPDarknet) and bbox_head weights
#   - Freeze backbone (optional) and bbox_head, use bbox_head as "detection Oracle"
#   - Train only BiTL-PAN + TextEncoder + IFSD (Fusion/Grounding/TemplateExtractor)
#   - Loss: L_BCE(M_raw, GT_mask) + λ_det·L_det(frozen_bbox_head) + λ_mae·L_mae
#
# Usage:
#   conda activate mae-ovd
#   cd /path/to/MAE-OVD
#   python tools/train_pretrain.py --config configs/pretrain/yolo_world_bitl_finetune.py

# ====================================================================
# YOLO-World paths (Relative to YOLO-World root)
# ====================================================================
yolo_world_root = "../YOLO-World"  # Relative path to YOLO-World
yolo_world_ckpt = "work_dirs/v2_s_stageB_coco_finetune_mae_stage3_full/epoch_25.pth"

# ====================================================================
# Mode flags
# ====================================================================
use_yolo_world_backbone = True
backbone_frozen = True

use_yolo_world_det_head = True
yolo_det_head_frozen = True

# ====================================================================
# Channel configuration (YOLO-World-v2-s, widen=0.5)
# ====================================================================
backbone_out_channels = (128, 256, 512)
bitl_pan_in_channels = (128, 256, 512)
text_dim = 256

imd_fused_channels = 896
fusion_size = 20
template_size = 7

# ====================================================================
# Detection loss (supervised via frozen bbox_head)
# ====================================================================
use_detection_branch = False
use_yolo_det_loss = True
yolo_det_loss_weight = 1.5
yolo_det_use_l1 = True
yolo_det_l1_weight = 0.1

# ====================================================================
# BCE mask loss
# ====================================================================
loss_bce_reduction = "mean"
use_f_target_aux = False
lambda_f_target_aux = 0.1

# ====================================================================
# Template Extractor spatial consistency loss
# ====================================================================
template_extractor_loss_weight = 0.5

# ====================================================================
# MAE auxiliary branch
# ====================================================================
use_mae = True
mae_mask_ratio = 0.5
mae_lambda = 0.2
mae_prob_branch = 0.5
mae_spark_style = True
mae_patch_size = 16
train_image_size = (320, 320)

# ====================================================================
# Training hyperparameters
# ====================================================================
max_epochs = 50
base_lr = 1e-4
train_batch_size_per_gpu = 8
max_samples = 20000
val_interval = 5
log_interval = 50
optimizer_type = "AdamW"
weight_decay = 0.05
clip_grad_norm = 10.0

# ====================================================================
# Data (Update paths according to your environment)
# ====================================================================
data_root = "datasets/coco2017"  # Relative path, update to your data location
ann_file = "annotations/instances_train2017.json"
data_prefix = dict(img="train2017/")
num_workers = 4
clip_model_name = "third_party/openai/clip-vit-base-patch32"

# ====================================================================
# Checkpoint
# ====================================================================
work_dir = "work_dirs/pretrain/yolo_world_bitl"
load_from = None
resume = False
