"""
Unit tests: BiTL-PAN forward shape, IR-IP/T-SSG interfaces.
Tests: parameter conventions, data flow, residual connections, gradients, ONNX compatibility.
"""
import os
import pytest
import torch

from mae_ovd.models.bitl_pan import BiTLPAN, IRIP, TSSG


# ---------- Basic Import and Shape Tests ----------

def test_ir_ip_import():
    m = IRIP(feat_channels=256, text_dim=256)
    assert m is not None


def test_t_ssg_import():
    m = TSSG(c_i=256, c_j=256, text_dim=256)
    assert m is not None


def test_bitl_pan_import():
    m = BiTLPAN(in_channels=(64, 128, 256), text_dim=256)
    assert m is not None


def test_ir_ip_forward_shape():
    """IR-IP: Input F5 (B,256,H5,W5), y (B,256); Output Y (B,256), F'_5 (B,256,H5,W5)."""
    B, C, H5, W5 = 2, 256, 10, 10
    irip = IRIP(feat_channels=C, text_dim=256)
    f5 = torch.randn(B, C, H5, W5)
    y = torch.randn(B, 256)
    Y, f5_prime = irip(f5, y)
    assert Y.shape == (B, C)
    assert f5_prime.shape == (B, C, H5, W5)


def test_t_ssg_forward_shape():
    """T-SSG: Input f_i (B,c_i,H,W), f_j (B,c_j,Hj,Wj), y (B,256); Output F^y_i (B,c_i,H,W)."""
    B, c_i, c_j, H, W = 2, 128, 256, 20, 20
    tssg = TSSG(c_i=c_i, c_j=c_j, text_dim=256)
    f_i = torch.randn(B, c_i, H, W)
    f_j = torch.randn(B, c_j, 10, 10)  # Different spatial size, internally interpolated
    y = torch.randn(B, 256)
    out = tssg(f_i, f_j, y)
    assert out.shape == (B, c_i, H, W)


def test_bitl_pan_forward_shape():
    """BiTL-PAN: Input [F3,F4,F5], y; Output (F^y_3, F^y_4, F^y_5) with same spatial dimensions as input."""
    B = 2
    # Convention: F3(64)/F4(128)/F5(256), spatial sizes 40,20,10 with F4 as baseline
    F3 = torch.randn(B, 64, 40, 40)
    F4 = torch.randn(B, 128, 20, 20)
    F5 = torch.randn(B, 256, 10, 10)
    y = torch.randn(B, 256)
    pan = BiTLPAN(in_channels=(64, 128, 256), text_dim=256)
    fy_3, fy_4, fy_5 = pan([F3, F4, F5], y)
    assert fy_3.shape == F3.shape
    assert fy_4.shape == F4.shape
    assert fy_5.shape == F5.shape


# ---------- Parameter Conventions ----------

def test_plan_channel_convention():
    """Backbone outputs F3:64ch, F4:128ch, F5:256ch; text y 256-dim; IR-IP outputs Y ∈ R^{B×256}."""
    B, H4, W4 = 2, 20, 20
    # F4 as 20×20 baseline, F3 typically 2x, F5 typically 0.5x
    H3, W3 = 40, 40
    H5, W5 = 10, 10
    F3 = torch.randn(B, 64, H3, W3)
    F4 = torch.randn(B, 128, H4, W4)
    F5 = torch.randn(B, 256, H5, W5)
    y = torch.randn(B, 256)
    irip = IRIP(feat_channels=256, text_dim=256)
    Y, F5_prime = irip(F5, y)
    assert Y.shape == (B, 256)
    assert F5_prime.shape == (B, 256, H5, W5)
    pan = BiTLPAN(in_channels=(64, 128, 256), text_dim=256)
    fy_3, fy_4, fy_5 = pan([F3, F4, F5], y)
    assert fy_3.shape == (B, 64, H3, W3)
    assert fy_4.shape == (B, 128, H4, W4)
    assert fy_5.shape == (B, 256, H5, W5)


# ---------- IR-IP Deep Tests ----------

def test_ir_ip_gap_and_f5_prime_structure():
    """IR-IP: F'_5 equals y broadcast multiplied with F5, independent of GAP; Y depends on GAP(F5) and y."""
    B, C, H, W = 1, 256, 8, 8
    irip = IRIP(feat_channels=C, text_dim=256)
    f5 = torch.randn(B, C, H, W)
    y = torch.randn(B, 256)
    Y, f5_prime = irip(f5, y)
    # F'_5 should equal (y_proj broadcast) * f5, i.e., f5_prime same shape as f5 with element-wise proportion
    y_proj = irip.y_to_feat(y).view(B, C, 1, 1)
    expected_f5_prime = y_proj * f5
    torch.testing.assert_close(f5_prime, expected_f5_prime)


def test_ir_ip_gradient_flow():
    """IR-IP: Gradients can backpropagate to f5 and y."""
    B, C, H, W = 2, 256, 10, 10
    irip = IRIP(feat_channels=C, text_dim=256)
    f5 = torch.randn(B, C, H, W, requires_grad=True)
    y = torch.randn(B, 256, requires_grad=True)
    Y, f5_prime = irip(f5, y)
    loss = Y.sum() + f5_prime.sum()
    loss.backward()
    assert f5.grad is not None and f5.grad.shape == f5.shape
    assert y.grad is not None and y.grad.shape == y.shape


def test_ir_ip_batch_one_and_various_spatial():
    """IR-IP: B=1 and various spatial sizes (H5,W5) work correctly."""
    irip = IRIP(feat_channels=256, text_dim=256)
    for B in (1, 4):
        for H5, W5 in [(5, 5), (20, 20), (10, 15)]:
            f5 = torch.randn(B, 256, H5, W5)
            y = torch.randn(B, 256)
            Y, f5_prime = irip(f5, y)
            assert Y.shape == (B, 256)
            assert f5_prime.shape == (B, 256, H5, W5)


# ---------- T-SSG Deep Tests ----------

def test_t_ssg_residual_formula():
    """T-SSG: Gating formula F^y_i = (F_i * γ * M_ij) + F_i (residual), element-wise verification."""
    B, c_i, c_j, H, W = 2, 64, 128, 16, 16
    tssg = TSSG(c_i=c_i, c_j=c_j, text_dim=256)
    f_i = torch.randn(B, c_i, H, W)
    f_j = torch.randn(B, c_j, H, W)
    y = torch.randn(B, 256)
    out = tssg(f_i, f_j, y)
    # Manually compute M_ij, γ, verify out = (f_i * γ * M_ij) + f_i
    f_j_aligned = torch.nn.functional.interpolate(f_j, size=(H, W), mode="bilinear", align_corners=False)
    cat_f = torch.cat([f_i, f_j_aligned], dim=1)
    m_ij = torch.sigmoid(tssg.spatial_conv(cat_f))
    gamma = tssg.channel_fc(y).view(B, c_i, 1, 1)
    expected = (f_i * gamma * m_ij) + f_i
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_t_ssg_same_size_no_interpolate_path():
    """T-SSG: F_i and F_j same size uses same branch, output shape correct."""
    B, c_i, c_j, H, W = 2, 256, 256, 10, 10
    tssg = TSSG(c_i=c_i, c_j=c_j, text_dim=256)
    f_i = torch.randn(B, c_i, H, W)
    f_j = torch.randn(B, c_j, H, W)
    y = torch.randn(B, 256)
    out = tssg(f_i, f_j, y)
    assert out.shape == (B, c_i, H, W)


def test_t_ssg_gradient_flow():
    """T-SSG: Gradients backpropagate to f_i, f_j, y."""
    B, c_i, c_j, H, W = 2, 128, 256, 8, 8
    tssg = TSSG(c_i=c_i, c_j=c_j, text_dim=256)
    f_i = torch.randn(B, c_i, H, W, requires_grad=True)
    f_j = torch.randn(B, c_j, 10, 10, requires_grad=True)
    y = torch.randn(B, 256, requires_grad=True)
    out = tssg(f_i, f_j, y)
    out.sum().backward()
    assert f_i.grad is not None
    assert f_j.grad is not None
    assert y.grad is not None


# ---------- BiTL-PAN Data Flow and Completeness ----------

def test_bitl_pan_batch_one():
    """BiTL-PAN: Output shape correct for B=1."""
    F3 = torch.randn(1, 64, 20, 20)
    F4 = torch.randn(1, 128, 10, 10)
    F5 = torch.randn(1, 256, 5, 5)
    y = torch.randn(1, 256)
    pan = BiTLPAN(in_channels=(64, 128, 256), text_dim=256)
    fy_3, fy_4, fy_5 = pan([F3, F4, F5], y)
    assert fy_3.shape == (1, 64, 20, 20)
    assert fy_4.shape == (1, 128, 10, 10)
    assert fy_5.shape == (1, 256, 5, 5)


def test_bitl_pan_20x20_f4_baseline():
    """With F4 as 20×20 baseline, F3 ~40×40, F5 ~10×10."""
    B = 2
    F3 = torch.randn(B, 64, 40, 40)
    F4 = torch.randn(B, 128, 20, 20)
    F5 = torch.randn(B, 256, 10, 10)
    y = torch.randn(B, 256)
    pan = BiTLPAN(in_channels=(64, 128, 256), text_dim=256)
    fy_3, fy_4, fy_5 = pan([F3, F4, F5], y)
    assert fy_3.shape == (B, 64, 40, 40)
    assert fy_4.shape == (B, 128, 20, 20)
    assert fy_5.shape == (B, 256, 10, 10)


def test_bitl_pan_gradient_flow():
    """BiTL-PAN: Gradients can backpropagate from output to img_feats and txt_embed."""
    B = 2
    F3 = torch.randn(B, 64, 20, 20, requires_grad=True)
    F4 = torch.randn(B, 128, 10, 10, requires_grad=True)
    F5 = torch.randn(B, 256, 5, 5, requires_grad=True)
    y = torch.randn(B, 256, requires_grad=True)
    pan = BiTLPAN(in_channels=(64, 128, 256), text_dim=256)
    fy_3, fy_4, fy_5 = pan([F3, F4, F5], y)
    (fy_3.sum() + fy_4.sum() + fy_5.sum()).backward()
    assert F3.grad is not None
    assert F4.grad is not None
    assert F5.grad is not None
    assert y.grad is not None


def test_bitl_pan_data_flow_ir_ip_then_td_then_bu():
    """Data flow order: 1) IR-IP → Y,F'_5  2) Top-down F5→F4→F3  3) Bottom-up updates F^y_4, F^y_5."""
    # Verify by checking if output changes when input changes (e.g., changing F5 must change fy_5)
    B = 2
    F3 = torch.randn(B, 64, 8, 8)
    F4 = torch.randn(B, 128, 8, 8)
    F5_a = torch.randn(B, 256, 4, 4)
    F5_b = F5_a + 1.0
    y = torch.randn(B, 256)
    pan = BiTLPAN(in_channels=(64, 128, 256), text_dim=256)
    pan.eval()
    with torch.no_grad():
        _, _, fy_5_a = pan([F3, F4, F5_a], y)
        _, _, fy_5_b = pan([F3, F4, F5_b], y)
    assert not torch.allclose(fy_5_a, fy_5_b), "F5 change should cause F^y_5 change"


# ---------- ONNX Compatibility (No einsum) ----------

def test_bitl_pan_no_einsum_in_code():
    """BiTL-PAN implementation avoids einsum, uses only Linear/matmul/Conv/Sigmoid/Mul/Add."""
    for name in ("ir_ip.py", "t_ssg.py", "bitl_pan.py"):
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mae_ovd", "models", "bitl_pan", name))
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        # Check for actual einsum calls
        assert "torch.einsum" not in src, f"{name} contains torch.einsum, violates ONNX compatibility"
        assert ".einsum(" not in src, f"{name} contains .einsum(, violates ONNX compatibility"


# ---------- ONNX Export and Inference ----------

class _BiTLPANOnnxWrapper(torch.nn.Module):
    """Wrapper for BiTL-PAN with 4 independent inputs for ONNX export."""

    def __init__(self):
        super().__init__()
        self.pan = BiTLPAN(in_channels=(64, 128, 256), text_dim=256)

    def forward(self, F3, F4, F5, txt_embed):
        return self.pan([F3, F4, F5], txt_embed)


def test_bitl_pan_onnx_export():
    """BiTL-PAN can be exported to ONNX (fixed spatial dimensions), no einsum or incompatible operators."""
    import tempfile
    B, H3, W3 = 2, 40, 40
    H4, W4 = 20, 20
    H5, W5 = 10, 10
    wrapper = _BiTLPANOnnxWrapper()
    wrapper.eval()
    F3 = torch.randn(B, 64, H3, W3)
    F4 = torch.randn(B, 128, H4, W4)
    F5 = torch.randn(B, 256, H5, W5)
    y = torch.randn(B, 256)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (F3, F4, F5, y),
                onnx_path,
                input_names=["F3", "F4", "F5", "txt_embed"],
                output_names=["Fy3", "Fy4", "Fy5"],
                dynamic_axes={
                    "F3": {0: "B"}, "F4": {0: "B"}, "F5": {0: "B"}, "txt_embed": {0: "B"},
                    "Fy3": {0: "B"}, "Fy4": {0: "B"}, "Fy5": {0: "B"},
                },
                opset_version=11,
            )
        assert os.path.isfile(onnx_path) and os.path.getsize(onnx_path) > 0
    finally:
        if os.path.isfile(onnx_path):
            os.unlink(onnx_path)


def test_bitl_pan_onnx_export_and_infer():
    """Export ONNX and infer with onnxruntime, shape consistent with PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime not installed, skipping ONNX inference test")
    import tempfile
    B, H3, W3, H4, W4, H5, W5 = 2, 40, 40, 20, 20, 10, 10
    wrapper = _BiTLPANOnnxWrapper()
    wrapper.eval()
    F3 = torch.randn(B, 64, H3, W3)
    F4 = torch.randn(B, 128, H4, W4)
    F5 = torch.randn(B, 256, H5, W5)
    y = torch.randn(B, 256)
    with torch.no_grad():
        expected = wrapper(F3, F4, F5, y)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (F3, F4, F5, y),
                onnx_path,
                input_names=["F3", "F4", "F5", "txt_embed"],
                output_names=["Fy3", "Fy4", "Fy5"],
                opset_version=11,
            )
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feeds = {
            "F3": F3.numpy(), "F4": F4.numpy(), "F5": F5.numpy(), "txt_embed": y.numpy(),
        }
        outs = sess.run(None, feeds)
        assert len(outs) == 3
        assert outs[0].shape == expected[0].numpy().shape
        assert outs[1].shape == expected[1].numpy().shape
        assert outs[2].shape == expected[2].numpy().shape
    finally:
        if os.path.isfile(onnx_path):
            os.unlink(onnx_path)
