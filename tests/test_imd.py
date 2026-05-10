"""
Unit tests: IFSD submodules forward shape, output X (B, 256, 7, 7).
"""
import pytest
import torch

from mae_ovd.models.imd import FeatureFusion, SemanticGrounding, TemplateExtractor


def test_fusion_import():
    m = FeatureFusion(channels=(64, 128, 256), fused_size=20, proj_dim=256)
    assert m is not None


def test_grounding_import():
    m = SemanticGrounding()
    assert m is not None


def test_template_extractor_import():
    m = TemplateExtractor(in_channels=448, out_channels=256, template_size=7)
    assert m is not None


def test_fusion_forward():
    """Fusion: F3/F4/F5 resize→concat→F_proj; Y→Y_proj; output shapes aligned with paper."""
    B, H, W = 2, 16, 16
    fusion = FeatureFusion(channels=(64, 128, 256), fused_size=20, proj_dim=256)
    f3 = torch.randn(B, 64, H, W)
    f4 = torch.randn(B, 128, H // 2, W // 2)
    f5 = torch.randn(B, 256, H // 4, W // 4)
    y = torch.randn(B, 256)
    f_fused, f_proj, y_proj = fusion(f3, f4, f5, y)
    assert f_fused.shape == (B, 448, 20, 20)
    assert f_proj.shape == (B, 256, 20, 20)
    assert y_proj.shape == (B, 256)


def test_grounding_forward():
    """Grounding: M_raw=matmul, M_A=Sigmoid, F_target=F_fused*M_A."""
    B, H, W = 2, 20, 20
    grounding = SemanticGrounding()
    f_proj = torch.randn(B, 256, H, W)
    y_proj = torch.randn(B, 256)
    f_fused = torch.randn(B, 448, H, W)
    m_raw, m_a, f_target = grounding(f_proj, y_proj, f_fused)
    assert m_raw.shape == (B, 1, H, W)
    assert m_a.shape == (B, 1, H, W)
    assert f_target.shape == (B, 448, H, W)
    assert m_a.min() >= 0 and m_a.max() <= 1


def test_template_extractor_forward():
    """TemplateExtractor: Output X (B, 256, 7, 7)."""
    B, H, W = 2, 20, 20
    ext = TemplateExtractor(in_channels=448, out_channels=256, template_size=7)
    f_target = torch.randn(B, 448, H, W)
    m_a = torch.softmax(torch.randn(B, 1, H, W).view(B, 1, -1), dim=2).view(B, 1, H, W)
    x = ext(f_target, m_a)
    assert x.shape == (B, 256, 7, 7)


def test_imd_full_pipeline():
    """IFSD full pipeline: F^y_3,F^y_4,F^y_5 + Y -> Fusion -> Grounding -> TemplateExtractor -> X (B,256,7,7)."""
    B = 2
    fusion = FeatureFusion(channels=(64, 128, 256), fused_size=20, proj_dim=256)
    grounding = SemanticGrounding()
    extractor = TemplateExtractor(in_channels=448, out_channels=256, template_size=7)

    fy3 = torch.randn(B, 64, 20, 20)
    fy4 = torch.randn(B, 128, 20, 20)
    fy5 = torch.randn(B, 256, 20, 20)
    y = torch.randn(B, 256)

    f_fused, f_proj, y_proj = fusion(fy3, fy4, fy5, y)
    m_raw, m_a, f_target = grounding(f_proj, y_proj, f_fused)
    x = extractor(f_target, m_a)

    assert x.shape == (B, 256, 7, 7), "IFSD output template X should be (B, 256, 7, 7)"
