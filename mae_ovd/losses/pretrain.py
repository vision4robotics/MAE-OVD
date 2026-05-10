"""
Pretraining Loss (aligned with Paper §6.1): BCE between M_raw and Binary Mask from GT Box (20×20);
Optional F_target auxiliary supervision implemented via same BCE in train_pretrain.
"""
import torch
import torch.nn.functional as F


def bbox_to_mask(bbox, grid_size=(20, 20), normalized=True, device=None, dtype=None):
    """
    Convert GT bbox to binary mask on grid_size (for pretraining BCE).
    bbox: (B, 4) or (B, 1, 4), format x1,y1,x2,y2; if normalized=True range is [0,1].
    grid_size: (H, W), e.g., (20, 20)
    return: (B, 1, H, W), binary mask, 1 inside target region (or soft weight), 0 outside.
    """
    H, W = grid_size
    if bbox.dim() == 3:
        bbox = bbox.squeeze(1)  # (B, 4)
    B = bbox.size(0)
    if device is None:
        device = bbox.device
    if dtype is None:
        dtype = bbox.dtype

    # Grid center positions in normalized coordinates
    # Cell (i, j) center: x = (j+0.5)/W, y = (i+0.5)/H
    if not normalized:
        # If bbox is pixel coordinates, need img_h, img_w for normalization; assumed normalized here
        pass
    x1 = bbox[:, 0:1]  # (B, 1)
    y1 = bbox[:, 1:2]
    x2 = bbox[:, 2:3]
    y2 = bbox[:, 3:4]

    j = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)  # (1, 1, W)
    i = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)  # (1, H, 1)
    cx = (j + 0.5) / W # (1, 1, W)
    cy = (i + 0.5) / H # (1, H, 1)

    x1_b = x1.view(B, 1, 1)
    x2_b = x2.view(B, 1, 1)
    y1_b = y1.view(B, 1, 1)
    y2_b = y2.view(B, 1, 1)
    in_x = (cx >= x1_b) & (cx <= x2_b)  # (B, 1, W)
    in_y = (cy >= y1_b) & (cy <= y2_b)  # (B, H, 1)
    mask = (in_y.float() * in_x.float()).clamp(0, 1)  # (B, H, W)
    return mask.unsqueeze(1)  # (B, 1, H, W)


def pretrain_loss(m_raw, gt_mask, reduction="mean", pos_weight=None):
    """
    BCE between M_raw and GT mask; using BCEWithLogitsLoss for numerical stability.

    Args:
        m_raw: (B, 1, H, W) Raw similarity logits (grounding output before sigmoid)
        gt_mask: (B, 1, H, W) Binary mask generated from GT Box, values 0 or 1
        reduction: Loss aggregation method, default "mean"
        pos_weight: Positive sample weight for handling class imbalance.
                    If None, auto-computed from gt_mask (typically 50-200x).

    Important: GT mask positive samples typically only 0.5-3%, without pos_weight
               M_raw will be pushed to negative infinity, losing semantic localization ability!
    """
    if pos_weight is None:
        # Auto-compute positive sample weight: background ratio / positive ratio
        # This makes positive and negative samples contribute roughly equally to loss
        neg_count = (gt_mask == 0).sum().item()
        pos_count = (gt_mask == 1).sum().item()
        if pos_count > 0:
            pos_weight = neg_count / pos_count
            # Clamp weight range to avoid extreme values
            pos_weight = min(max(pos_weight, 10.0), 500.0)
        else:
            pos_weight = 10.0

    # pos_weight must be converted to tensor
    if not isinstance(pos_weight, torch.Tensor):
        pos_weight = gt_mask.new_tensor([pos_weight])

    return F.binary_cross_entropy_with_logits(
        m_raw, gt_mask, reduction=reduction, pos_weight=pos_weight
    )


def _giou_loss_normalized(pred_bbox, gt_bbox, eps=1e-7):
    """
    GIoU loss for normalized bbox (1 - GIoU).
    pred_bbox, gt_bbox: (B, 4) x1, y1, x2, y2, range [0, 1].
    """
    px1, py1, px2, py2 = pred_bbox[:, 0], pred_bbox[:, 1], pred_bbox[:, 2], pred_bbox[:, 3]
    gx1, gy1, gx2, gy2 = gt_bbox[:, 0], gt_bbox[:, 1], gt_bbox[:, 2], gt_bbox[:, 3]
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    area_i = iw * ih
    area_u = area_p + area_g - area_i
    iou = area_i / (area_u + eps)
    cx1 = torch.minimum(px1, gx1)
    cy1 = torch.minimum(py1, gy1)
    cx2 = torch.maximum(px2, gx2)
    cy2 = torch.maximum(py2, gy2)
    area_c = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
    giou = iou - (area_c - area_u) / (area_c + eps)
    return (1.0 - giou).mean()


def detection_loss_for_x(pred_bbox, gt_bbox, use_l1=False, l1_weight=0.1, eps=1e-7):
    """
    Pretraining: detection loss between pred_bbox from frozen Box Head and GT bbox.
    pred_bbox, gt_bbox: (B, 4) x1, y1, x2, y2 normalized [0, 1].
    return: L_giou + optional L1
    """
    loss_giou = _giou_loss_normalized(pred_bbox, gt_bbox, eps=eps)
    if not use_l1:
        return loss_giou
    l1 = (pred_bbox - gt_bbox).abs().mean()
    return loss_giou + l1_weight * l1
