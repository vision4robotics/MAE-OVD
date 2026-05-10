"""
MAE-OVD Box Head module (aligned with Paper §4 Training Strategy).

Paper §4 states: the image encoder, IR-IP, BiTL-PAN, IFSD,
and Box head are optimized end-to-end.

Box Head is the core detection head of MAE-OVD, jointly optimized with BiTL-PAN during training
for effective utilization of BiTL-IFSD.

Architecture (YOLO-World-v2):
  - Multi-scale input: [F3(256), F4(512), F5(512)] (BiTL-PAN output, aligned with v2-l)
  - reg_preds[i]: Conv2d × 2 + Conv2d(1×1) → DFL raw dist (B, 4*reg_max, H_i, W_i)
  - DFL decoding → ltrb offsets → xyxy absolute coordinates
  - Take predicted bbox at GT bbox center, compute GIoU + L1 loss

Note: Box Head parameters require_grad=True (trainable), gradients backpropagate to BiTL-PAN.
"""
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bbox_giou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute per-sample GIoU (batch-level).
    pred/gt: (B, 4) xyxy format, absolute pixel coordinates.
    Returns: (B,) GIoU values ∈ [-1, 1].
    """
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx1, gy1, gx2, gy2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]

    # Intersection
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    # Areas
    pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    gt_area = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = pred_area + gt_area - inter + eps

    iou = inter / union

    # Smallest enclosing box
    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)
    c_area = (cx2 - cx1).clamp(0) * (cy2 - cy1).clamp(0) + eps

    giou = iou - (c_area - union) / c_area
    return giou


class YOLOWorldDetHead(nn.Module):
    """
    MAE-OVD Box Head, jointly optimized with BiTL-PAN (aligned with Paper §4).

    Paper §4 states: the image encoder, IR-IP, BiTL-PAN, IFSD,
    and Box head are optimized end-to-end.

    All modules (including Box Head) are trainable, gradients backpropagate to BiTL-PAN.

    Usage:
        det_head = YOLOWorldDetHead(yolo_world_root=..., checkpoint_path=...)
        loss = det_head(img_feats, gt_bboxes_norm, img_size)
        loss.backward()  # Gradients flow back to img_feats (from BiTL-PAN)

    Args:
        yolo_world_root: YOLO-World project root directory.
        checkpoint_path: YOLO-World checkpoint path (optional, for weight initialization).
        strides: Downsample strides for each scale, aligned with in_channels order.
        reg_max: Number of DFL distribution bins, default 16 (consistent with YOLO-World-v2).
        use_l1: Whether to add L1 bbox loss.
        l1_weight: L1 loss weight.
    """

    def __init__(
        self,
        yolo_world_root: str = "../YOLO-World",  # Relative path to YOLO-World
        checkpoint_path: Optional[str] = None,
        strides: Tuple[int, ...] = (8, 16, 32),
        reg_max: int = 16,
        head_in_channels: Tuple[int, ...] = (256, 512, 1024),
        widen_factor: float = 0.5,
        use_l1: bool = True,
        l1_weight: float = 0.1,
    ):
        super().__init__()
        self.strides = strides
        self.reg_max = reg_max
        self.use_l1 = use_l1
        self.l1_weight = l1_weight

        # Add YOLO-World to sys.path
        mmyolo_path = os.path.join(yolo_world_root, "third_party", "mmyolo")
        for p in [yolo_world_root, mmyolo_path]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from mmyolo.utils import register_all_modules as reg_mmyolo
        from mmdet.utils import register_all_modules as reg_mmdet
        reg_mmdet(init_default_scope=False)
        reg_mmyolo(init_default_scope=False)
        # Direct import and instantiation (bypass registry issues)
        from yolo_world.models.dense_heads.yolo_world_head import YOLOWorldHeadModule

        # head_in_channels / widen_factor from config, compatible with v2-s / v2-l
        self.head_module = YOLOWorldHeadModule(
            num_classes=80,
            in_channels=list(head_in_channels),
            widen_factor=widen_factor,
            featmap_strides=[8, 16, 32],
            reg_max=reg_max,
            embed_dims=512,
            use_bn_head=True,
            act_cfg=dict(type="SiLU", inplace=True),
            norm_cfg=dict(type="BN", eps=0.001, momentum=0.03),
        )

        if checkpoint_path is not None:
            self._load_from_ckpt(checkpoint_path)
            print(f"[YOLOWorldDetHead] Loaded bbox_head from {checkpoint_path}")

        # Box Head trainable (Paper §4 requires end-to-end optimization)
        # Note: Maintains trainable state after loading pretrained weights for joint optimization with BiTL-PAN
        self.head_module.train()
        print("[YOLOWorldDetHead] Box head is TRAINABLE (jointly optimized with BiTL-PAN per §4).")

        # DFL projection vector [0, 1, ..., reg_max-1]
        self.register_buffer(
            "proj",
            torch.arange(reg_max, dtype=torch.float32),
            persistent=False,
        )

    def _load_from_ckpt(self, ckpt_path: str) -> None:
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("state_dict", raw)
        prefix = "bbox_head.head_module."
        head_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not head_state:
            raise ValueError(f"Bbox head weights not found with 'bbox_head.head_module.*' prefix: {ckpt_path}")
        missing, unexpected = self.head_module.load_state_dict(head_state, strict=False)
        if missing:
            print(f"  [warn] head_module missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  [warn] head_module unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    def train(self, mode: bool = True):
        super().train(mode)
        # Box Head trainable, maintain normal train mode
        self.head_module.train()
        return self

    def _decode_reg_single(
        self,
        feat: torch.Tensor,
        reg_pred: nn.Module,
        stride: int,
    ) -> torch.Tensor:
        """
        Single scale: feat (B,C,H,W) → reg_pred → DFL decoding → xyxy (B,H,W,4) absolute coordinates.
        """
        B, _, H, W = feat.shape
        raw = reg_pred(feat)  # (B, 4*reg_max, H, W), gradients backpropagate to feat

        # DFL decoding
        raw = raw.reshape(B, 4, self.reg_max, H * W).permute(0, 3, 1, 2)  # (B,H*W,4,reg_max)
        ltrb = raw.softmax(3).matmul(self.proj.view(-1, 1)).squeeze(-1)    # (B,H*W,4)
        ltrb = ltrb.reshape(B, H, W, 4) * stride                           # Convert to pixel units

        # Build cell-center grid (absolute coordinates)
        gy = torch.arange(H, device=feat.device, dtype=feat.dtype)
        gx = torch.arange(W, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # (H,W)
        cx = (grid_x + 0.5) * stride  # (H,W)
        cy = (grid_y + 0.5) * stride

        l, t, r, b_ = ltrb[..., 0], ltrb[..., 1], ltrb[..., 2], ltrb[..., 3]  # (B,H,W)
        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b_
        return torch.stack([x1, y1, x2, y2], dim=-1)  # (B,H,W,4) xyxy absolute coordinates

    def forward(
        self,
        img_feats: List[torch.Tensor],
        gt_bboxes_norm: torch.Tensor,
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Compute multi-scale detection loss (GIoU + optional L1), taking predicted bbox at GT bbox center.

        Args:
            img_feats: [F3, F4, F5], BiTL-PAN outputs (channels 128/256/512).
            gt_bboxes_norm: (B, 4) normalized xyxy, [0,1].
            img_size: (H, W) input image resolution for denormalizing GT.

        Returns:
            Scalar loss, gradients flow to img_feats (propagating to BiTL-PAN).
        """
        H_img, W_img = img_size
        B = gt_bboxes_norm.shape[0]

        # Denormalize GT bbox to pixel coordinates (B, 4)
        gt_px = gt_bboxes_norm.clone()
        gt_px[:, [0, 2]] *= W_img
        gt_px[:, [1, 3]] *= H_img
        gt_cx = (gt_px[:, 0] + gt_px[:, 2]) / 2  # (B,)
        gt_cy = (gt_px[:, 1] + gt_px[:, 3]) / 2

        reg_preds = self.head_module.reg_preds  # ModuleList, length 3

        total_loss = gt_bboxes_norm.new_zeros(1)
        n_valid = 0

        for scale_idx, (feat, stride) in enumerate(zip(img_feats, self.strides)):
            _, _, H, W = feat.shape
            xyxy_map = self._decode_reg_single(feat, reg_preds[scale_idx], stride)
            # xyxy_map: (B, H, W, 4)

            # GT center at current stride grid coordinates, clamp to valid range
            gc_x = (gt_cx / stride).long().clamp(0, W - 1)  # (B,)
            gc_y = (gt_cy / stride).long().clamp(0, H - 1)

            # Extract predicted bbox at GT center grid point
            batch_idx = torch.arange(B, device=feat.device)
            pred_box = xyxy_map[batch_idx, gc_y, gc_x]  # (B, 4)

            # GIoU loss
            giou = _bbox_giou(pred_box, gt_px)
            loss_giou = (1 - giou).mean()
            total_loss = total_loss + loss_giou

            if self.use_l1:
                loss_l1 = F.l1_loss(pred_box, gt_px, reduction="mean")
                total_loss = total_loss + self.l1_weight * loss_l1

            n_valid += 1

        return total_loss / max(n_valid, 1)

    def get_pred_bbox_at_gt_center(
        self,
        img_feats: List[torch.Tensor],
        gt_bboxes_norm: torch.Tensor,
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Take predicted bbox at GT bbox center grid point for inference/visualization (no gradient).

        Args:
            img_feats: [F3, F4, F5], BiTL-PAN outputs.
            gt_bboxes_norm: (B, 4) normalized xyxy [0,1].
            img_size: (H, W) input image resolution.

        Returns:
            pred_bbox: (B, 4) pixel coordinates xyxy, same batch as gt_bboxes_norm.
        """
        H_img, W_img = img_size
        B = gt_bboxes_norm.shape[0]
        gt_px = gt_bboxes_norm.clone()
        gt_px[:, [0, 2]] *= W_img
        gt_px[:, [1, 3]] *= H_img
        gt_cx = (gt_px[:, 0] + gt_px[:, 2]) / 2
        gt_cy = (gt_px[:, 1] + gt_px[:, 3]) / 2
        reg_preds = self.head_module.reg_preds
        preds = []
        for scale_idx, (feat, stride) in enumerate(zip(img_feats, self.strides)):
            _, _, H, W = feat.shape
            xyxy_map = self._decode_reg_single(feat, reg_preds[scale_idx], stride)
            gc_x = (gt_cx / stride).long().clamp(0, W - 1)
            gc_y = (gt_cy / stride).long().clamp(0, H - 1)
            batch_idx = torch.arange(B, device=feat.device)
            pred_box = xyxy_map[batch_idx, gc_y, gc_x]  # (B, 4)
            preds.append(pred_box)
        return preds[-1]  # Use last scale prediction (can be changed to multi-scale average)

    def get_pred_bbox_at_cell(
        self,
        img_feats: List[torch.Tensor],
        scale_idx: int,
        grid_y: int,
        grid_x: int,
    ) -> torch.Tensor:
        """
        Take predicted bbox at specified scale and grid point (used when no GT, from M_raw peak grid point).

        Args:
            img_feats: [F3, F4, F5], BiTL-PAN outputs.
            scale_idx: 0/1/2 corresponding to F3/F4/F5.
            grid_y, grid_x: Grid coordinates (row, col), must be within corresponding scale H,W.

        Returns:
            pred_bbox: (B, 4) pixel coordinates xyxy.
        """
        feat = img_feats[scale_idx]
        stride = self.strides[scale_idx]
        reg_pred = self.head_module.reg_preds[scale_idx]
        xyxy_map = self._decode_reg_single(feat, reg_pred, stride)  # (B, H, W, 4)
        B = feat.shape[0]
        batch_idx = torch.arange(B, device=feat.device)
        return xyxy_map[batch_idx, grid_y, grid_x]  # (B, 4)
