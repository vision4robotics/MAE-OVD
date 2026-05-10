"""
MAE Decoder: Multi-scale feature fusion -> patch-level pixel prediction.
Input: multi_scale_feats=[F3, F4, F5] (optional tuple/list),
       each scale resized to unified resolution (1/4 of F5), concatenated and projected to unified channels, then upsampled for reconstruction.
Supports paper equations (1)-(3): masked features, MAE loss, total loss weighting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MAEDecoder(nn.Module):
    """
    Multi-scale fusion MAE decoder.
    Input: multi_scale_feats: list/tuple of [F3, F4, F5], each (N, C_i, H_i, W_i)
          - F3: (N, 64, H/8,  W/8)
          - F4: (N, 128, H/16, W/16)
          - F5: (N, 256, H/32, W/32)
    Output: (N, patch_size**2 * 3, H_p, W_p), H_p=W_p=image_size/patch_size
    Pipeline:
        1. Resize each scale to F5 spatial dimensions
        2. Concatenate channels -> project to unified channel dim (default 256)
        3. Dynamically upsample to target resolution
    """

    def __init__(
        self,
        in_channels_list=(64, 128, 256),
        proj_dim: int = 256,
        patch_size: int = 16,
        in_ch: int = None,  # Alias: single-scale input channels (backward compatibility)
    ):
        super().__init__()
        self.patch_size = patch_size
        # Compatible with in_ch parameter (single-scale input, for backward compatibility)
        if in_ch is not None:
            self.in_channels_list = (in_ch // 4, in_ch // 2, in_ch)
        else:
            self.in_channels_list = list(in_channels_list)
        total_in_ch = sum(self.in_channels_list)
        self.proj_dim = proj_dim

        # Multi-scale -> unified channels
        self.proj = nn.Conv2d(total_in_ch, proj_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(proj_dim)
        self.relu = nn.ReLU(inplace=True)

        # Intermediate layers + final prediction
        self.decoder = nn.Sequential(
            nn.Conv2d(proj_dim, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, patch_size ** 2 * 3, 1),
        )

    def forward(self, multi_scale_feats: Tensor, target_size: tuple = None) -> Tensor:
        """
        Args:
            multi_scale_feats: list/tuple [F3, F4, F5] or single Tensor
            target_size: Target resolution (H_p, W_p), auto-inferred from patch_size and image size by default
        Returns:
            pred: (N, patch_size**2*3, H_p, W_p)
        """
        if isinstance(multi_scale_feats, (list, tuple)):
            feats = list(multi_scale_feats)
            if len(feats) != 3:
                raise ValueError(f"MAEDecoder expects 3 scales, got {len(feats)}")
            # Unify to F5 resolution (smallest scale)
            target_size = feats[2].shape[2:]  # F5's (H5, W5)
            resized = [
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                for f in feats
            ]
            x = torch.cat(resized, dim=1)
        else:
            x = multi_scale_feats  # type: ignore

        x = self.proj(x)
        x = self.bn(x)
        x = self.relu(x)  # (N, proj_dim, H5, W5)
        
        # Dynamically upsample to target size
        # Target size determined by image_size / patch_size
        if target_size is None:
            target_size = x.shape[2:]  # fallback to F5 size
        
        pred = self.decoder(x)  # (N, patch_size**2*3, H5, W5)
        
        # Upsample to target size if needed
        if pred.shape[2:] != target_size:
            pred = F.interpolate(pred, size=target_size, mode="nearest")
        
        return pred
