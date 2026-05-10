"""
IFSD: Implicit Feature Semantic Distillation (aligned with Paper §3.3, §4.2)

IFSD serves as the semantic purification hub:
  1. Receives multi-scale text-enhanced features F1Y, F2Y, F3Y from BiTL-PAN
  2. Projects to unified channels C=256 and resolution H×W=20×20
  3. Computes alignment response M_raw = F_fused · Y_proj^T
  4. Purifies response through task-aware masking and staged optimization

Paper equations (11)-(12):
  M_raw = F_fused · Y_proj^T
  M_A = σ(M_raw)

Purpose: Suppress semantically similar distractors (birds, clouds, buildings), focusing response on actual object regions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitFeatureSemanticDistillation(nn.Module):
    """
    Implicit Feature Semantic Distillation module (IFSD).

    Pipeline:
      1. Multi-scale feature projection: F1Y, F2Y, F3Y -> unified channels (proj_dim) + unified resolution (aligned_size)
      2. Feature fusion: Concat -> Conv1x1 -> F_fused
      3. Text projection: Y -> Linear -> Y_proj
      4. Alignment response: M_raw = F_fused · Y_proj^T, M_A = sigmoid(M_raw)
      5. Semantic distillation: F_target = F_fused * M_A (optional output)
    """

    def __init__(
        self,
        channels=(64, 128, 256),
        proj_dim=256,
        aligned_size=20,
        text_dim=256,
    ):
        super().__init__()
        self.channels = list(channels)
        self.proj_dim = proj_dim
        self.aligned_size = aligned_size
        self.text_dim = text_dim

        # Per-scale projection layers: F_iY -> proj_dim
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(c, proj_dim, kernel_size=1)
            for c in channels
        ])

        # Fusion projection: sum(proj_layers) -> F_fused
        self.fusion_proj = nn.Conv2d(proj_dim * len(channels), proj_dim, kernel_size=1)

        # Text projection: Y -> Y_proj (proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

    def forward(self, f1y, f2y, f3y, y):
        """
        Args:
            f1y: (B, C1, H1, W1) - Shallow text-enhanced features
            f2y: (B, C2, H2, W2) - Mid-level text-enhanced features
            f3y: (B, C3, H3, W3) - Deep text-enhanced features
            y: (B, text_dim) Refined text embedding
        Returns:
            m_raw: (B, 1, aligned_size, aligned_size) Raw alignment response (unnormalized)
            m_a: (B, 1, aligned_size, aligned_size) Normalized response
            f_target: (B, proj_dim*3, aligned_size, aligned_size) Response-weighted features
        """
        target_size = (self.aligned_size, self.aligned_size)

        # 1. Multi-scale projection + upsample to unified resolution
        proj_feats = []
        for i, (feat, proj_layer) in enumerate(zip([f1y, f2y, f3y], self.proj_layers)):
            p = proj_layer(feat)
            p_up = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            proj_feats.append(p_up)

        # 2. Feature fusion
        f_fused = torch.cat(proj_feats, dim=1)  # (B, proj_dim*3, H, W)
        f_fused = self.fusion_proj(f_fused)  # (B, proj_dim, H, W)

        # 3. Text projection
        y_proj = self.text_proj(y)  # (B, proj_dim)

        # 4. Compute alignment response (equations 11-12)
        B, C, H, W = f_fused.shape
        f_flat = f_fused.view(B, C, -1)  # (B, C, H*W)
        y_proj_ = y_proj.unsqueeze(2)  # (B, proj_dim, 1)

        # matmul for inner product: (B, 1, H*W)
        m_raw_flat = torch.bmm(y_proj_.transpose(1, 2), f_flat)  # (B, 1, H*W)
        m_raw = m_raw_flat.view(B, 1, H, W)
        m_a = torch.sigmoid(m_raw)

        # 5. Semantic distillation: response map guided features
        f_target = f_fused * m_a  # (B, proj_dim, H, W)

        return m_raw, m_a, f_target


class TaskAwareIFSD(nn.Module):
    """
    IFSD with task-aware masking (aligned with Paper §3.3 task-aware masking).

    Additional features compared to IFSD:
      - Hard/soft masking mechanism: generate task mask based on M_A peak regions
      - Staged optimization: keep full response in early training, strengthen masking in later training
    """

    def __init__(
        self,
        channels=(64, 128, 256),
        proj_dim=256,
        aligned_size=20,
        text_dim=256,
        use_task_mask=True,
        mask_threshold=0.5,
    ):
        super().__init__()
        self.use_task_mask = use_task_mask
        self.mask_threshold = mask_threshold
        self.ifsd = ImplicitFeatureSemanticDistillation(
            channels=channels,
            proj_dim=proj_dim,
            aligned_size=aligned_size,
            text_dim=text_dim,
        )

    def forward(self, f1y, f2y, f3y, y, epoch=None, max_epochs=None):
        """
        Args:
            epoch: Current training epoch (for dynamic masking strength adjustment)
            max_epochs: Total training epochs
        """
        m_raw, m_a, f_target = self.ifsd(f1y, f2y, f3y, y)

        if self.use_task_mask and epoch is not None and max_epochs is not None:
            # Staged optimization: enhance masking in later stages
            progress = epoch / max_epochs
            if progress > 0.5:  # Later stage
                threshold = self.mask_threshold * (0.5 + 0.5 * progress)
                # Hard mask: preserve high-response regions
                mask_hard = (m_a > threshold).float()
                # Soft mask: preserve peak regions, suppress edges
                f_target = f_target * mask_hard

        return m_raw, m_a, f_target
