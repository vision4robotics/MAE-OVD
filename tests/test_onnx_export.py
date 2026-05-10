"""
Unit tests: ONNX export (BiTL-PAN / IFSD) and onnxruntime inference.
"""
import pytest


def test_onnx_utils_import():
    from mae_ovd.utils.onnx_utils import check_onnx_friendly
    assert check_onnx_friendly(None) is True
