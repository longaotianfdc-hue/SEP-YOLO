import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .prepbn import RepBN, LinearNorm
from .attention import *
from .ast import AdaptiveSparseSA
from .filc import FMFFN
from .semnet import SEFN
from .mona import Mona
from .transMamba import SpectralEnhancedFFN
from .EVSSM import EDFFN
from ..modules.transformer import TransformerEncoderLayer, AIFI
from ..modules.block import C2PSA, PSABlock, ABlock, A2C2f, C3k

__all__ = [ 'MS_GRU']

ln = nn.LayerNorm
linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)
class MSDWConv(nn.Module):
    """
    多尺度深度可分离卷积模块 (Multi-scale Depthwise Convolution)
    将不同膨胀率的深度卷积并行组合，然后融合。
    """

    def __init__(self, channels, act_layer=nn.GELU) -> None:
        super().__init__()
        # 分支1: 基础 3x3 深度卷积 (d=1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, groups=channels),
            act_layer()
        )
        # 分支2: 膨胀 3x3 深度卷积 (d=2)
        # 膨胀率为 2 时，3x3 卷积的感受野等效于 5x5，所需的 padding 为 (3-1)*2 / 2 = 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=True, groups=channels),
            act_layer()
        )

        # 融合层: 1x1 卷积用于融合来自两个分支的特征，保持通道数不变
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1, 1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # 沿通道维度拼接两个分支的输出
        x_cat = torch.cat((x1, x2), dim=1)

        # 通过 1x1 卷积进行特征融合
        return self.fusion_conv(x_cat)
class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
        #     act_layer()
        # )
        self.dwconv=MSDWConv(hidden_features, act_layer=act_layer)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    # def forward(self, x):
    #     x, v = self.fc1(x).chunk(2, dim=1)
    #     x = self.dwconv(x) * v
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x
class PSABlock_CGLU(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = ConvolutionalGLU(c, c * 2, c)

class MS_GRU(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_CGLU(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))