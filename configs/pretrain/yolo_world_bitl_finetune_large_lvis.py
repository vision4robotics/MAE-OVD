# MAE-OVD Pretraining Configuration
# YOLO-World-v2-l backbone with full LVIS dataset training
#
# Usage:
#   conda activate mae-ovd
#   cd /path/to/MAE-OVD
#   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
#     tools/train_pretrain.py --yolo-world-mode \
#     --config configs/pretrain/yolo_world_bitl_finetune_large_lvis.py --launcher pytorch
#
# Full LVIS: ~118K images, 1203 categories

# ====================================================================
# Training hyperparameters
# ====================================================================
max_epochs = 50
base_lr = 1e-4
train_batch_size_per_gpu = 8
max_samples = None                          # Use full dataset
val_interval = 5
log_interval = 50
optimizer_type = "AdamW"
weight_decay = 0.05
clip_grad_norm = 10.0

# ====================================================================
# YOLO-World paths (Relative to YOLO-World root)
# ====================================================================
yolo_world_root = "../YOLO-World"  # Relative path to YOLO-World
yolo_world_ckpt = "weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"

# ====================================================================
# YOLO-World-v2-l architecture parameters
# ====================================================================
yolo_world_arch = "P5"
yolo_world_deepen_factor = 1.0
yolo_world_widen_factor = 1.0
yolo_world_last_stage_out_channels = 512

# ====================================================================
# Mode flags
# ====================================================================
use_yolo_world_backbone = True
backbone_frozen = True

use_yolo_world_det_head = True
# Note: Box Head is trainable during training (Paper §4 end-to-end optimization)

# ====================================================================
# Channel configuration (YOLO-World-v2-l)
# ====================================================================
# Paper §4.1: YOLO-World v2 with YOLOv8-L backbone
backbone_out_channels = (256, 512, 512)   # F3, F4, F5
bitl_pan_in_channels = (256, 512, 512)    # BiTL-PAN aligned with backbone
text_dim = 256

imd_fused_channels = 1280
fusion_size = 20
template_size = 7

# ====================================================================
# Detection loss (Box Head + BiTL-PAN joint optimization)
# Paper §4: Box head is optimized end-to-end
# ====================================================================
use_detection_branch = False
use_yolo_det_loss = True
yolo_det_loss_weight = 1.5
yolo_det_use_l1 = True
yolo_det_l1_weight = 0.1
yolo_det_head_in_channels = (256, 512, 512)
yolo_det_widen_factor = 1.0

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
# MAE auxiliary branch (Paper §4)
# ====================================================================
use_mae = True
mae_mask_ratio = 0.4          # 0.4 for joint training, 0.5 for MAE-focused pretraining
mae_lambda = 0.1             # Base λ value
mae_lambda_max = 0.2         # Warmup max λ
mae_prob_branch = 0.35        # MAE activation probability
mae_spark_style = True
mae_patch_size = 16
train_image_size = (640, 640)  # Paper §4.1: 640×640 input
# MAE scheduling
mae_warmup_epochs = 1
mae_decay_start_epoch = max_epochs - 2
disable_mosaic_epoch = max_epochs - 2

# ====================================================================
# Data - Full LVIS training set (Update paths according to your environment)
# ====================================================================
data_root = "datasets/lvis"  # Relative path, update to your data location
ann_file = "lvis_v1_train.json"
data_prefix = dict(img="train2017/")
num_workers = 8
clip_model_name = "third_party/openai/clip-vit-base-patch32"

# ====================================================================
# Checkpoint
# ====================================================================
work_dir = "work_dirs/pretrain/yolo_world_bitl_l_lvis_full"
load_from = None
resume = False
