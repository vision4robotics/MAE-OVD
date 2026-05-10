# MAE-OVD Pretraining Configuration (Default)
# Lightweight backbone with MAE-driven pretraining for anti-UAV detection
#
# Training Strategy:
#   - MAE-driven backbone for degradation-resistant representation learning
#   - BiTL-PAN for multi-scale text-visual interaction
#   - IFSD for semantic purification
#   - Decoupled train-test pipeline
#
# Usage:
#   conda activate mae-ovd
#   cd /path/to/MAE-OVD
#   python tools/train_pretrain.py --config configs/pretrain/default.py

# ====================================================================
# Backbone configuration (consistent with mae_ovd.models.backbone)
# ====================================================================
backbone_out_channels = (64, 128, 256)  # F3, F4, F5
text_dim = 256
fusion_size = 20  # F_fused unified 20×20; F_proj channel 256
template_size = 7  # IFSD output X: 256×7×7

# ====================================================================
# Model components (aligned with mae_ovd.models)
# ====================================================================
backbone_type = "LightweightEncoder"
bitl_pan_in_channels = (64, 128, 256)
imd_fused_channels = 448  # 64 + 128 + 256

# ====================================================================
# Training (YOLO-World style, aligned with MMDetection)
# ====================================================================
max_epochs = 100
base_lr = 2e-4
train_batch_size_per_gpu = 16
max_samples = 20000
val_interval = 2
log_interval = 100
optimizer_type = "AdamW"
weight_decay = 0.05

# ====================================================================
# Data (Update paths according to your environment)
# ====================================================================
data_root = "datasets/coco2017"  # Relative path, update to your data location
ann_file = "annotations/instances_train2017.json"
data_prefix = dict(img="train2017/")
num_workers = 4
clip_model_name = "third_party/openai/clip-vit-base-patch32"

# ====================================================================
# Loss
# ====================================================================
loss_bce_reduction = "mean"
use_f_target_aux = False
lambda_f_target_aux = 0.1

# ====================================================================
# Detection branch
# ====================================================================
use_detection_branch = True
detection_loss_weight = 1.0
box_head_frozen = True
detection_use_l1 = True
detection_l1_weight = 0.1

# ====================================================================
# MAE auxiliary branch (Paper §4: MAE-driven degradation-resistant learning)
# ====================================================================
use_mae = True
mae_mask_ratio = 0.4          # Random mask ratio (0.4 for joint training, 0.5 for MAE-focused pretraining)
mae_lambda = 0.1              # Base λ value
mae_lambda_max = 0.2          # Warmup max λ
mae_prob_branch = 0.35        # MAE activation probability
mae_spark_style = True
mae_patch_size = 16
train_image_size = (320, 320)

# ====================================================================
# MAE scheduling
# ====================================================================
mae_warmup_epochs = 1         # First epoch warmup
mae_decay_start_epoch = 98    # Last 2 epochs decay (max_epochs=100)
disable_mosaic_epoch = 98     # Last 2 epochs disable Mosaic

# ====================================================================
# Checkpoint
# ====================================================================
work_dir = "work_dirs/pretrain/default"
load_from = None
resume = False
