"""
Semantic Grounding: M_raw = F_proj·Y_proj^T, M_A = Sigmoid(M_raw), F_target = F_fused * M_A.
Uses matmul implementation, ONNX friendly, no einsum.
"""
import torch
import torch.nn as nn


class SemanticGrounding(nn.Module):
    """Similarity heatmap M_raw = F_proj·Y_proj^T (matmul), M_A = Sigmoid(M_raw), F_target = F_fused * M_A."""

    def __init__(self):
        super().__init__()

    def forward(self, f_proj, y_proj, f_fused):
        """
        f_proj: (B, 256, H, W), y_proj: (B, 256), f_fused: (B, 448, H, W)
        return: M_raw (B, 1, H, W), M_A (B, 1, H, W), F_target (B, 448, H, W)
        """
        B, C, H, W = f_proj.shape
        # F_proj as (B, 256, H*W), Y_proj (B, 256) -> (B, 256, 1)
        # M_raw = Y_proj^T @ F_proj per batch: (B, 1, H*W)
        f_flat = f_proj.view(B, C, -1)  # (B, 256, H*W)
        y_proj_ = y_proj.unsqueeze(2)   # (B, 256, 1)
        m_raw_flat = torch.bmm(y_proj_.transpose(1, 2), f_flat)  # (B, 1, H*W)
        m_raw = m_raw_flat.view(B, 1, H, W)
        m_a = torch.sigmoid(m_raw)
        f_target = f_fused * m_a
        return m_raw, m_a, f_target
