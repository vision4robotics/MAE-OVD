"""
BiTL-PAN: IR-IP + 3×T-SSG (top-down) + 2×T-SSG (bottom-up).
Input: img_feats=[F3,F4,F5], txt_embed y. Output: F^y_3, F^y_4, F^y_5.

Architecture:
  - IR-IP: outputs Y (text_dim dimension) and F'_3 (c5 dimension)
  - Y directly used for TSSG's channel_fc (input text_dim, output c_i)
  - No _y_proj projection layer needed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ir_ip import IRIP
from .t_ssg import TSSG


class BiTLPAN(nn.Module):
    """Bidirectional semantic enhancement flow: IR-IP → top-down (F5→F4→F3) → bottom-up (F3→F4→F5), T-SSG gating at each step."""

    def __init__(self, in_channels=(64, 128, 256), text_dim=256):
        super().__init__()
        self.in_channels = list(in_channels)
        c3, c4, c5 = in_channels[0], in_channels[1], in_channels[2]
        self.text_dim = text_dim

        self.ir_ip = IRIP(feat_channels=c5, text_dim=text_dim)

        # Top-down: F5(F5, F'_5); F4(F4, F^y_5_up); F3(F3, F^y_4_up)
        self.td_ssg_5 = TSSG(c_i=c5, c_j=c5, text_dim=text_dim)   # F5 layer, Fj=F'_5
        self.td_ssg_4 = TSSG(c_i=c4, c_j=c5, text_dim=text_dim)   # F4, Fj=upsampled F^y_5
        self.td_ssg_3 = TSSG(c_i=c3, c_j=c4, text_dim=text_dim)   # F3, Fj=upsampled F^y_4

        # Bottom-up: update F^y_4(F^y_4, F^y_3_down); update F^y_5(F^y_5, F^y_4_down)
        self.bu_ssg_4 = TSSG(c_i=c4, c_j=c3, text_dim=text_dim)
        self.bu_ssg_5 = TSSG(c_i=c5, c_j=c4, text_dim=text_dim)

    def forward(self, img_feats, txt_embed):
        """
        img_feats: [F3, F4, F5]
        txt_embed: (B, text_dim=256)
        return: (F^y_3, F^y_4, F^y_5)
        """
        F3, F4, F5 = img_feats[0], img_feats[1], img_feats[2]
        y = txt_embed

        # 1. IR-IP(F5, y) → Y (text_dim), F'_5 (c5)
        Y, F5_prime = self.ir_ip(F5, y)
        # Y is text_dim dimension, used for TSSG's channel_fc
        # F5_prime is c5 dimension, used for top-down path

        # 2. Top-down
        fy_5 = self.td_ssg_5(F5, F5_prime, Y)
        _, _, H4, W4 = F4.shape
        fy_5_up = F.interpolate(fy_5, size=(H4, W4), mode="bilinear", align_corners=False)
        fy_4 = self.td_ssg_4(F4, fy_5_up, Y)
        _, _, H3, W3 = F3.shape
        fy_4_up = F.interpolate(fy_4, size=(H3, W3), mode="bilinear", align_corners=False)
        fy_3 = self.td_ssg_3(F3, fy_4_up, Y)

        # 3. Bottom-up
        _, _, H4b, W4b = fy_4.shape
        fy_3_down = F.interpolate(fy_3, size=(H4b, W4b), mode="bilinear", align_corners=False)
        fy_4 = self.bu_ssg_4(fy_4, fy_3_down, Y)
        _, _, H5b, W5b = fy_5.shape
        fy_4_down = F.interpolate(fy_4, size=(H5b, W5b), mode="bilinear", align_corners=False)
        fy_5 = self.bu_ssg_5(fy_5, fy_4_down, Y)

        return fy_3, fy_4, fy_5
