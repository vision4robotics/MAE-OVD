"""
YOLOWorldImageBackbone: Wrapper for YOLO-World's YOLOv8CSPDarknet, loading pretrained weights,
providing the same interface as LightweightEncoder: forward(x) -> [F3, F4, F5].

YOLO-World-v2-s output channels (P5 arch, widen=0.5, last_stage=1024):
  F3 (P3): 128ch, 1/8 downsampling
  F4 (P4): 256ch, 1/16 downsampling
  F5 (P5): 512ch, 1/32 downsampling

Different from LightweightEncoder(64,128,256), this module outputs (128, 256, 512),
requiring corresponding in_channels adjustments in downstream modules like BiTL-PAN and FeatureFusion.
"""
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# YOLO-World-v2-s default backbone parameters
_DEFAULT_ARCH = "P5"
_DEFAULT_LAST_STAGE_OUT = 1024
_DEFAULT_DEEPEN = 0.33
_DEFAULT_WIDEN = 0.5
_DEFAULT_OUT_CHANNELS = (128, 256, 512)  # Corresponding to P3/P4/P5 (widen=0.5)


class YOLOWorldImageBackbone(nn.Module):
    """
    YOLO-World image backbone (YOLOv8CSPDarknet), without text_model.
    Loads weights from YOLO-World checkpoint with 'backbone.image_model.*' prefix.

    Args:
        yolo_world_root (str): YOLO-World project root directory for sys.path insertion.
        checkpoint_path (str | None): .pth path; None for random initialization (debugging).
        frozen (bool): Whether to freeze all parameters (recommended for pretraining).
        arch / deepen_factor / widen_factor / last_stage_out_channels:
            Backbone structure parameters consistent with YOLO-World config.
    """

    out_channels: Tuple[int, ...]  # (c3, c4, c5)

    def __init__(
        self,
        yolo_world_root: str = "../YOLO-World",  # Relative path to YOLO-World
        checkpoint_path: Optional[str] = None,
        frozen: bool = True,
        arch: str = _DEFAULT_ARCH,
        deepen_factor: float = _DEFAULT_DEEPEN,
        widen_factor: float = _DEFAULT_WIDEN,
        last_stage_out_channels: int = _DEFAULT_LAST_STAGE_OUT,
        out_channels: Tuple[int, ...] = _DEFAULT_OUT_CHANNELS,
    ):
        super().__init__()
        # Explicitly specify output channels from config, compatible with v2-s / v2-l scales.
        self.out_channels = out_channels

        # Add YOLO-World to sys.path (idempotent)
        mmyolo_path = os.path.join(yolo_world_root, "third_party", "mmyolo")
        for p in [yolo_world_root, mmyolo_path]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from mmyolo.models.backbones.csp_darknet import YOLOv8CSPDarknet

        self.backbone = YOLOv8CSPDarknet(
            arch=arch,
            last_stage_out_channels=last_stage_out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            out_indices=(2, 3, 4),  # P3, P4, P5
            norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
            act_cfg=dict(type="SiLU", inplace=True),
        )

        if checkpoint_path is not None:
            self._load_from_yolo_world_ckpt(checkpoint_path)
            print(f"[YOLOWorldImageBackbone] Loaded from {checkpoint_path}")

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            print("[YOLOWorldImageBackbone] Backbone frozen.")

        self._frozen = frozen

    # ------------------------------------------------------------------
    def _load_from_yolo_world_ckpt(self, ckpt_path: str) -> None:
        """Load backbone.image_model.* weights from YOLO-World checkpoint."""
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("state_dict", raw)

        prefix = "backbone.image_model."
        backbone_state = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }
        if not backbone_state:
            raise ValueError(
                f"Weights with 'backbone.image_model.*' prefix not found. "
                f"Please verify checkpoint path: {ckpt_path}"
            )
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"  [warn] backbone missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  [warn] backbone unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """If frozen=True, keep backbone in eval mode to fix BN statistics."""
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            [F3, F4, F5]: List of 3, channels (128, 256, 512),
                          spatial sizes H/8, H/16, H/32.
        """
        outs = self.backbone(x)  # tuple of (P3, P4, P5)
        return list(outs)
