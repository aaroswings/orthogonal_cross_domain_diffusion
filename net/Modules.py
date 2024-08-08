import torch
import torch.nn.functional as F
from torch.nn import Module, Identity, Dropout, Sequential, Linear, SiLU, GroupNorm, Conv2d, PixelShuffle, MultiheadAttention
from torch.nn import init
from einops import repeat
from einops.layers.torch import Rearrange

import math
from typing import Optional

    
class FourierFeatures(Module):
    # Fourier features for angles (phi, theta)
    # For angular features, set scale to 1.
    def __init__(
        self, 
        dim_out: int, 
        std: float = 1., 
        num_features: int = 1, 
        scale: float = math.pi * 2
    ):
        super().__init__()
        if dim_out % 2 != 0:
            raise ValueError("dim_out must be divisible by 2")
        self.num_features = num_features
        self.register_buffer('weight', torch.randn((num_features, dim_out // 2)) * std)
        self.scale = scale

    def forward(self, input):
        if input.dim() != 2:
            raise ValueError("Expected input of (B x k) tensors.")
        if input.size(1) != self.num_features:
            raise ValueError(f"Expected input to have {self.num_features} features to encode.")
        
        f = self.scale * input @ self.weight

        return torch.cat([f.cos(), f.sin()], dim=-1)
    

class ScaleShiftNorm(Module):
    def __init__(
        self, 
        groups: int, 
        dim: int, 
        emb_dim: int, 
        eps: float = 1e-5
    ):
        super().__init__()
        self.norm = GroupNorm(groups, dim, eps=eps)
        self.proj = Sequential(SiLU(), Linear(emb_dim, dim * 2))
    
    def forward(self, x, emb):
        emb = self.proj(emb)[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x
    

class FeatureProjection(Module):
    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        groups: int = 8, 
        scale_shift: bool = False, 
        emb_dim: Optional[int] = None
    ):
        super().__init__()
        if scale_shift: assert emb_dim is not None
        self.scale_shift = scale_shift
        self.proj = Conv2d(dim, dim_out, 3, padding=1, bias=False)
        self.norm = (
            ScaleShiftNorm(groups, dim_out, emb_dim) if scale_shift
            else GroupNorm(groups, dim_out)
        )
        self.actvn = SiLU()
    
    def forward(self, x, emb: Optional[torch.tensor] = None):
        x = self.proj(x)
        x = self.norm(x, emb) if self.scale_shift else self.norm(x)
        x = self.actvn(x)
        return x


class ResBlock(Module):
    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        emb_dim: int, 
        p_dropout: float = 0.0, 
        group_dim: int = 16,
        groups: int = 32
    ):
        super().__init__()
        assert dim % group_dim == 0
        # groups = dim
        # while(dim // groups > group_dim): 
        #     groups = groups // 2
        self.register_buffer('root2', torch.sqrt(torch.tensor(2)))
        self.block1 = FeatureProjection(dim, dim_out, groups, True, emb_dim)
        self.dropout = Dropout(p_dropout) if p_dropout else Identity()
        self.block2 = FeatureProjection(dim, dim_out, groups, False)
        self.skip = Identity() if dim == dim_out else Conv2d(dim, dim_out, 1)

    def forward(self, x, emb = None):
        h = self.block1(x, emb)
        h = self.dropout(h)
        h = self.block2(h)
        return (h + self.skip(x)) / self.root2
    

class Upsample(Module):
    def __init__(
        self, 
        dim: int, 
        dim_out: Optional[int] = None, 
        factor: int = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim_out or dim
        conv = Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = Sequential(
            conv,
            SiLU(),
            PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class Downsample(Sequential):
    def __init__(
        self, dim: int, dim_out: Optional[int] = None, factor: int = 2
    ):
        super().__init__(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = factor, p2 = factor),
            Conv2d(dim * (factor ** 2), dim_out or dim, 1)
        )


class ImageSelfAttention2d(Module): 
     """ 
     Self-attention with a layer norm.
     """ 
     def __init__(
        self, dim: int, num_heads: int, dropout: float
    ): 
        super().__init__()
        self.register_buffer('root2', torch.sqrt(torch.tensor(2)))
        self.norm = GroupNorm(1, dim, affine=False)
        self.attn = MultiheadAttention(dim, num_heads, dropout)

     def forward(self, x): 
        x = self.norm(x)
        b, c, w, h = x.shape
        y = x.reshape(b, w * h, c)
        y, _ = self.attn(y, y, y)
        y = y.reshape(b, c, w, h)

        return y