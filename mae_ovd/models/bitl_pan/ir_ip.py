"""
IR-IP: Instance-Refined Implicit Prompting (aligned with paper equations 4-6)

Input: F3 (B, C3, H3, W3), y (B, text_dim=256).
Output: Y (B, text_dim), F'_3 (B, C3, H3, W3).

Paper equations:
  Eq(4):  v̄ = GAP(F₃)
  Eq(5):  v̄' = Linear(v̄)
  Eq(6):  Y = LayerNorm(Att(y, v̄', v̄')) ⊕ y
          F'_3 = Broadcast(v̄ ⊗ Linear(y))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IRIP(nn.Module):
    """
    Instance-Refined Implicit Prompting.

    Eq(4):  v̄ = GAP(F₃) → Global average pooling
    Eq(5):  v̄' = Linear(v̄) → Project to feature dimension
    Eq(6):  Y = LayerNorm(Att(y, v̄', v̄')) ⊕ y → Text refinement + residual
            F'_3 = Broadcast(v̄ ⊗ Linear(y)) → Visual-text feature fusion
    """

    def __init__(self, feat_channels=256, text_dim=256):
        super().__init__()
        self.feat_channels = feat_channels
        self.text_dim = text_dim
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Attention scaling factor
        self.scale = feat_channels ** -0.5

        # Q projection: text_dim -> feat_channels
        self.q_proj = nn.Linear(text_dim, feat_channels)
        # K/V projection: feat_channels -> feat_channels
        self.k_proj = nn.Linear(feat_channels, feat_channels)
        
        # Project feat_channels back to text_dim (for generating Y)
        self.feat_to_text = nn.Linear(feat_channels, text_dim)

        # Eq(6): Y = LayerNorm(Att(y, v̄', v̄')) ⊕ y
        self.layer_norm = nn.LayerNorm(text_dim)

        # Eq(6) F'_3: v̄ ⊗ Linear(y)
        self.y_to_feat = nn.Linear(text_dim, feat_channels)

    def forward(self, f3, y):
        """
        Args:
            f3: (B, C3, H3, W3) Deep scale features
            y: (B, text_dim) Text embedding
        Returns:
            Y: (B, text_dim) Refined text embedding (used in TSSG within BiTL-PAN)
            f3_prime: (B, C3, H3, W3) Visual-text fusion features (used in top-down path)
        """
        B, C3, H3, W3 = f3.shape

        # Eq(4): v̄ = GAP(F₃)
        v_bar = self.gap(f3).view(B, C3)  # (B, C3)

        # Eq(5): v̄' = Linear(v̄)
        v_bar_proj = self.k_proj(v_bar)  # (B, C3)

        # Eq(6) top: Y = LayerNorm(Att(y, v̄', v̄')) ⊕ y
        # Q = y (B, text_dim) -> Q_proj -> (B, C3)
        # K = V = v_bar_proj (B, C3)
        # Att(Q, K, V) = softmax(QK^T/√d) * V
        
        q = self.q_proj(y)  # (B, C3)
        # Scaled dot-product attention
        attn_score = torch.sum(q * v_bar_proj, dim=1, keepdim=True) * self.scale  # (B, 1)
        v_attended = attn_score * v_bar_proj  # (B, C3)

        # Project back to text_dim and residual connect with original y
        v_attended_proj = self.feat_to_text(v_attended)  # (B, text_dim)
        Y = self.layer_norm(v_attended_proj + y)

        # Eq(6) bottom: F'_3 = Broadcast(v̄ ⊗ Linear(y))
        # v̄ ⊗ Linear(y): element-wise multiply, then broadcast to (B, C3, H3, W3)
        y_feat = self.y_to_feat(y)  # (B, C3)
        fused = v_bar * y_feat  # (B, C3)
        f3_prime = fused.view(B, C3, 1, 1).expand(B, C3, H3, W3)  # (B, C3, H3, W3)

        return Y, f3_prime


class IRIPSimple(nn.Module):
    """
    IR-IP Simplified version: maintains compatibility with original implementation while aligning with paper equations.

    Eq(4):  v̄ = GAP(F₃)
    Eq(5):  v̄' = Linear(v̄)
    Eq(6):  Y = LayerNorm(Att(y, v̄', v̄')) ⊕ y, F'_3 = Broadcast(v̄ ⊗ Linear(y))
    """

    def __init__(self, feat_channels=256, text_dim=256):
        super().__init__()
        self.feat_channels = feat_channels
        self.text_dim = text_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scale = feat_channels ** -0.5
        # Q projection
        self.q_proj = nn.Linear(text_dim, feat_channels)
        # K/V projection
        self.k_proj = nn.Linear(feat_channels, feat_channels)
        # LayerNorm
        self.layer_norm = nn.LayerNorm(text_dim)
        # F'_3 computation
        self.y_to_feat = nn.Linear(text_dim, feat_channels)

    def forward(self, f3, y):
        """
        Args:
            f3: (B, C3, H3, W3)
            y: (B, text_dim)
        Returns:
            Y: (B, text_dim)
            f3_prime: (B, C3, H3, W3)
        """
        B, C3, H3, W3 = f3.shape

        # Eq(4): v̄ = GAP(F₃)
        v_bar = self.gap(f3).view(B, C3)  # (B, C3)

        # Eq(5): v̄' = Linear(v̄)
        v_bar_proj = self.k_proj(v_bar)  # (B, C3)

        # Eq(6) top: Y = LayerNorm(Att(y, v̄', v̄')) ⊕ y
        q = self.q_proj(y)  # (B, C3)
        # Scaled dot-product attention: (B, C3) @ (B, C3)^T @ (B, C3) = (B, C3)
        # Simplified scalar attention for ONNX compatibility
        attn_score = torch.sum(q * v_bar_proj, dim=1, keepdim=True) * self.scale  # (B, 1)
        v_attended = attn_score * v_bar_proj  # (B, C3)

        # Project back to text_dim and LayerNorm
        # Due to possible C3 != text_dim, use residual connection
        # Simplified: sum v_attended and add to y
        v_attended_reduced = v_attended.sum(dim=1, keepdim=True)  # (B, 1)
        y_norm = self.layer_norm(y + v_attended_reduced)  # (B, text_dim)
        Y = y_norm

        # Eq(6) bottom: F'_3 = Broadcast(v̄ ⊗ Linear(y))
        y_feat = self.y_to_feat(y)  # (B, C3)
        fused = v_bar * y_feat  # (B, C3) - element-wise multiply
        f3_prime = fused.view(B, C3, 1, 1).expand(B, C3, H3, W3)  # (B, C3, H3, W3)

        return Y, f3_prime
