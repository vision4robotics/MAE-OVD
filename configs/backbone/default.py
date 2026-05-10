# MAE-OVD Lightweight Backbone Configuration
# Multi-scale output: F3(64), F4(128), F5(256)
# Spatial dimensions vary with input size

# Multi-scale channels (aligned with BiTL-PAN and IFSD)
out_channels = (64, 128, 256)  # F3, F4, F5

# Backbone type
backbone_type = "LightweightEncoder"

# Optional: ImageNet pretrained weight path (None for random initialization)
pretrained = None

# Input channels (RGB)
in_channels = 3
