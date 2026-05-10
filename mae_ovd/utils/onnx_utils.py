"""
ONNX compatibility check: no einsum, no dynamic shape on critical paths, suitable for RK3588 NPU.
"""
import torch.nn as nn


def check_onnx_friendly(module):
    """
    Placeholder: traverse module or trace, check for einsum / dynamic shape etc.
    """
    # TODO: Simple check or torch.jit.trace trial export
    return True
