"""
Patch masking and MAE loss utilities (based on YOLO-World/yolo_world/models/utils/patch_mask_utils.py).
Used for MAE auxiliary branch: patch masking, mask maps, SparK feature zeroing, MSE on masked regions only.
Extended features:
  - Semantic-aware masking: adjust mask ratio based on category frequency
  - Frequency-aware masking: protect high-frequency details based on frequency domain analysis
  - Cross-pyramid unified masking: same mask upsampled to each scale
"""
from typing import Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def patchify(imgs: Tensor, patch_size: int = 16) -> Tensor:
    """
    Split image into non-overlapping patches (no einsum, ONNX friendly).
    imgs: (N, 3, H, W), H, W must be divisible by patch_size
    returns: (N, L, patch_size**2 * 3), L = (H // patch_size) * (W // patch_size)
    """
    p = patch_size
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    # (N, 3, h, p, w, p) -> (N, h, w, 3, p, p) -> (N, h*w, p*p*3)
    x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(imgs.shape[0], h * w, p * p * 3)
    return x


def random_masking_2d(
    num_patches_h: int,
    num_patches_w: int,
    mask_ratio: float,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """
    Random masking per sample; 1 = masked (SparK: features zeroed at masked positions).
    returns: mask (N, H_p, W_p)
    """
    L = num_patches_h * num_patches_w
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(batch_size, L, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.ones(batch_size, L, device=device)
    mask[:, :len_keep] = 0
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask = mask.reshape(batch_size, num_patches_h, num_patches_w)
    return mask


def make_masked_image(
    imgs: Tensor,
    mask: Tensor,
    patch_size: int,
    fill_value: Optional[float] = None,
) -> Tensor:
    """
    Fill masked regions (mask==1) with mean or fill_value.
    imgs: (N, 3, H, W), mask: (N, H_p, W_p), 1 = masked
    returns: (N, 3, H, W)
    """
    N, _, H, W = imgs.shape
    if fill_value is None:
        fill_value_t = imgs.reshape(N, 3, -1).mean(dim=2, keepdim=True).unsqueeze(3)
    else:
        fill_value_t = torch.full(
            (N, 3, 1, 1), fill_value, device=imgs.device, dtype=imgs.dtype
        )
    mask_pixel = mask.unsqueeze(1).float()
    mask_pixel = torch.nn.functional.interpolate(
        mask_pixel, size=(H, W), mode="nearest"
    )
    masked_imgs = imgs * (1 - mask_pixel) + fill_value_t * mask_pixel
    return masked_imgs


def apply_spark_mask(
    img_feats: Tuple[Tensor, ...],
    mask: Tensor,
    strides: Tuple[int, ...] = (8, 16, 32),
    img_size: Union[int, Tuple[int, int]] = 320,
    mask_tokens: Optional[list] = None,
) -> Tuple[Tensor, ...]:
    """
    SparK style: zero or replace masked positions with learnable tokens.
    img_feats: Multi-scale features (F3, F4, F5), each (N, C_i, H_i, W_i)
    mask: (N, H_p, W_p), 1 = masked
    mask_tokens: Optional, each (1, C_i, 1, 1); if provided: feat*(1-mask_s) + token*mask_s
    """
    N, H_p, W_p = mask.shape
    if isinstance(img_size, (list, tuple)):
        img_h, img_w = img_size[0], img_size[1]
    else:
        img_h = img_w = img_size
    out = []
    for i, feat in enumerate(img_feats):
        _, _, h, w = feat.shape
        mask_s = torch.nn.functional.interpolate(
            mask.unsqueeze(1).float(),
            size=(h, w),
            mode="nearest",
        )
        if mask_tokens is not None and i < len(mask_tokens):
            tok = mask_tokens[i].to(feat.dtype).to(feat.device)
            out.append(feat * (1.0 - mask_s) + tok * mask_s)
        else:
            out.append(feat * (1.0 - mask_s))
    return tuple(out)


def compute_mae_loss_on_masked_patches(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    norm_pix: bool = False,
) -> Tensor:
    """
    Compute MSE only on masked patches (mask==1).
    pred: (N, L, P*P*3) or (N, P*P*3, H_p, W_p), normalized to (N, L, D)
    target: (N, L, P*P*3), from patchify(imgs)
    mask: (N, H_p, W_p) or (N, L), 1 = masked
    """
    # Save original dimension info
    if pred.dim() == 4:
        N, D, H_pred, W_pred = pred.shape
        L_pred = H_pred * W_pred
        # Convert pred to (N, L, D) format
        pred = pred.reshape(N, D, L_pred).permute(0, 2, 1)  # (N, L, D)
    elif pred.dim() == 3:
        N, L_pred, D = pred.shape
        H_pred, W_pred = None, None
    else:
        raise ValueError(f"pred must be 3D or 4D, got {pred.dim()}D with shape {pred.shape}")
    
    if mask.dim() == 3:
        mask = mask.reshape(mask.shape[0], -1)
    
    L_target = mask.shape[1]
    
    # Fix: pred and mask/target patch counts may differ (e.g., 20×20 vs 40×40)
    # Upsample pred to match target/mask patch count
    if L_pred != L_target:
        H_target = int(math.sqrt(L_target))
        W_target = L_target // H_target
        # pred: (N, L, D) -> (N, D, H, W) -> resize -> (N, D, H_target, W_target) -> (N, L_target, D)
        pred_4d = pred.permute(0, 2, 1).reshape(N, D, H_pred, W_pred)
        pred_up = F.interpolate(pred_4d, size=(H_target, W_target), mode="nearest")
        pred = pred_up.reshape(N, D, -1).permute(0, 2, 1)
    
    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True) + 1e-6
        target = (target - mean) / var ** 0.5
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum().clamp(min=1)
    return loss


def semantic_aware_masking(
    mask_ratio_base: float,
    category_freq: str = "common",
    background_ratio: float = 0.6,
    device: torch.device = None,
) -> float:
    """
    Semantic masking strategy that adjusts mask ratio based on category frequency (aligned with paper).

    MAE-OVD paper:
      - rare: mask_ratio = 0.4
      - common: mask_ratio = 0.1
      - frequent: mask_ratio = 0.0
      - background: mask_ratio = 0.6

    Args:
        mask_ratio_base: Base mask ratio (used when category_freq is "common")
        category_freq: Category frequency type, "rare" | "common" | "frequent"
        background_ratio: Mask ratio for background regions
        device: Optional, device for returned mask_ratio

    Returns:
        Adjusted mask ratio float
    """
    freq_to_ratio = {
        "rare": 0.4,
        "common": 0.1,
        "frequent": 0.0,
    }
    # Background mask ratio independent of category_freq
    if category_freq == "background":
        return background_ratio
    return freq_to_ratio.get(category_freq, mask_ratio_base)


def frequency_aware_masking(
    img: Tensor,
    mask_ratio: float = 0.5,
    threshold: float = 0.5,
    device: torch.device = None,
) -> Tensor:
    """
    Frequency-aware masking strategy based on frequency domain analysis.

    Identifies high-frequency detail regions (edges, textures) through DCT/FFT analysis,
    reduces masking probability in these regions to protect small objects or detail-rich areas.

    Args:
        img: (N, 3, H, W) Input image
        mask_ratio: Overall masking ratio
        threshold: High-frequency energy threshold, below this is considered low-frequency (sky, background)
        device: Output device

    Returns:
        mask: (N, H_p, W_p) Binary mask, 1=masked
    """
    N, C, H, W = img.shape
    patch_size = 16
    if H % patch_size != 0 or W % patch_size != 0:
        # Cannot handle non-16x dimensions, return random mask
        H_p, W_p = H // patch_size, W // patch_size
        return random_masking_2d(H_p, W_p, mask_ratio, N, device or img.device)

    H_p, W_p = H // patch_size, W // patch_size
    if device is None:
        device = img.device

    masks = []
    for i in range(N):
        img_i = img[i]  # (3, H, W)
        gray = img_i.mean(dim=0)  # (H, W)

        # 2D DCT computation (simplified approximation: gradient energy as frequency proxy)
        # Compute gradient energy for each 16×16 patch as "frequency" proxy
        patch_energy = F.avg_pool2d(
            gray.unsqueeze(0).unsqueeze(1),  # (1,1,H,W)
            kernel_size=patch_size,
            stride=patch_size,
        ).squeeze(0).squeeze(0)  # (H_p, W_p)

        # Compute gradient energy (Sobel as high-frequency approximation)
        grad_x = (gray[:, 2:] - gray[:, :-2]).abs()
        grad_y = (gray[2:, :] - gray[:-1, :]).abs()
        # Pad to patch size
        grad_x_pad = F.pad(grad_x, (1, 1, 0, 0), mode="replicate")[:, :H]
        grad_y_pad = F.pad(grad_y, (0, 0, 1, 1), mode="replicate")[:H, :]

        patch_grad = F.avg_pool2d(
            grad_x_pad.add(grad_y_pad).unsqueeze(0).unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size,
        ).squeeze(0).squeeze(0)

        # High-frequency regions (high gradient energy) reduce masking probability
        energy_mean = patch_grad.mean()
        energy_std = patch_grad.std() + 1e-6
        prob_map = ((patch_grad - energy_mean) / energy_std).sigmoid()

        # Low-frequency regions (prob < threshold) more likely to be masked
        mask_prob = mask_ratio + (1 - mask_ratio) * (1 - prob_map)
        mask_prob = mask_prob.clamp(0.1, 0.9)

        # Sample mask
        noise = torch.rand(H_p * W_p, device=device)
        len_keep = int(H_p * W_p * (1 - mask_ratio))
        ids_shuffle = torch.argsort(noise)
        mask = torch.ones(H_p * W_p, device=device)
        mask[ids_shuffle[:len_keep]] = 0
        ids_restore = torch.argsort(ids_shuffle)
        mask = torch.gather(mask, dim=0, index=ids_restore)
        masks.append(mask.reshape(1, H_p, W_p))

    return torch.cat(masks, dim=0)


def propagate_mask_across_pyramid(
    mask: Tensor,
    target_sizes: list,
) -> list:
    """
    Upsample unified mask to multiple feature pyramid levels.

    Paper equation (1): Mask maintains consistent visibility prior across all pyramid levels

    Args:
        mask: (N, H_p, W_p) Unified mask (patch level), 1=masked
        target_sizes: list of (H, W) tuples, target sizes for each level

    Returns:
        masks_per_scale: list of Tensors, each (N, 1, H_i, W_i) kept 4D for broadcasting
    """
    masks = []
    for target_size in target_sizes:
        mask_up = F.interpolate(
            mask.unsqueeze(1).float(),
            size=target_size,
            mode="nearest",
        )
        # Keep 4D: (N, 1, H, W) instead of squeezing to (N, H, W)
        # This correctly broadcasts to (N, C, H, W) features
        masks.append(mask_up)
    return masks


def apply_multi_scale_spark_mask(
    img_feats: Tuple[Tensor, ...],
    mask: Tensor,
    mask_tokens: Optional[list] = None,
) -> Tuple[Tensor, ...]:
    """
    Cross-pyramid multi-scale SparK masking (aligned with paper equation (1)).

    Difference from apply_spark_mask:
      - This function uses propagate_mask_across_pyramid to ensure consistent masking across levels
      - Supports optional learnable mask_tokens

    Args:
        img_feats: Multi-scale features (F3, F4, F5), each (N, C_i, H_i, W_i)
        mask: (N, H_p, W_p), Unified patch-level mask
        mask_tokens: Optional, each (1, C_i, 1, 1)

    Returns:
        tuple of Tensor: Masked features for each level
    """
    target_sizes = [f.shape[2:] for f in img_feats]
    multi_scale_masks = propagate_mask_across_pyramid(mask, target_sizes)

    out = []
    for i, (feat, mask_s) in enumerate(zip(img_feats, multi_scale_masks)):
        if mask_tokens is not None and i < len(mask_tokens):
            tok = mask_tokens[i].to(feat.dtype).to(feat.device)
            out.append(feat * (1.0 - mask_s) + tok * mask_s)
        else:
            out.append(feat * (1.0 - mask_s))
    return tuple(out)
