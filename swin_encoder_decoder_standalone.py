import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

from layers import HyperLinear, HyperSoftmax, Concatenate, HyperConv3d

from functools import reduce, lru_cache
from operator import mul

import warnings


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)




class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., n_divs=1):
        assert in_features % n_divs == 0, f'in_features [{in_features}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % n_divs == 0, \
            f'hidden_features [{hidden_features}] is not divisible by n_divs [{n_divs}]'
        assert out_features % n_divs == 0, f'out_features [{out_features}] is not divisible by n_divs [{n_divs}]'

        Linear = nn.Linear if n_divs == 1 else HyperLinear

        self.fc1 = Linear(in_features, hidden_features, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        Linear = nn.Linear if n_divs == 1 else HyperLinear
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_divs=1, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, n_divs=n_divs)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        Linear = nn.Linear if n_divs == 1 else HyperLinear
        self.reduction = Linear(4 * dim, 2 * dim, bias=False, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.norm = norm_layer(4 * dim)
        self.concat = Concatenate(dim=-1, n_divs=n_divs)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        # x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        # x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        # x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = rearrange(x, 'b d (h p1) (w p2) c -> b d h w (p2 p1 c)', p1=2, p2=2)
        # x = x.view(B, -1, 4 * C)  # B D*H/2*W/2 4*C  - are we using this?

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    """ Patch Expanding Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm, ratio=2, factor=1, n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.factor = factor
        Linear = nn.Linear if n_divs == 1 else HyperLinear
        self.expansion = Linear(dim, self.factor * self.ratio * dim, bias=False, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.norm = norm_layer(dim)
        # self.concat = Concatenate(dim=-1, n_divs=n_divs)
        # self.D, self.H, self.W = None, None, None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C)
        """
        # D, H, W = self.D, self.H, self.W
        #
        # B, L, C = x.shape
        # assert L == D * H * W, "input feature has wrong size"
        # B, D, H, W, C = x.shape

        x = self.norm(x)
        x = self.expansion(x)

        # padding
        # pad_input = (H % self.ratio == 1) or (W % self.ratio == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % self.ratio, 0, H % self.ratio))

        # x = x.view(B, D, H*self.ratio, W*self.ratio, (C//self.ratio) * self.factor)

        x = rearrange(x, 'b d h w (p2 p1 c) -> b d (h p1) (w p2) c', p1=self.ratio, p2=self.ratio)

        # x = x.view(B, -1, C//self.ratio * self.factor)  # B D*H*2*W*2 C/2

        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim//2, norm_layer=norm_layer, n_divs=n_divs)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])

        # self.downsample = downsample
        # if self.downsample is not None:
        #     self.downsample = downsample(dim=dim, norm_layer=norm_layer, n_divs=n_divs)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """

        if self.downsample is not None:
            x = rearrange(x, 'b c d h w -> b d h w c')
            x = self.downsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')

        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        # print(x.shape)
        x = rearrange(x, 'b d h w c -> b c d h w')
        # x = x.view(B, D, H, W, -1)
        #
        # if self.downsample is not None:
        #     x = self.downsample(x)
        # x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None, n_divs=1):
        super().__init__()
        assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'

        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        Conv3d = nn.Conv3d if n_divs == 1 else HyperConv3d
        self.proj = Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                           **{'n_divs': n_divs for n in [n_divs] if n > 1})
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 # pretrained=None,
                 # pretrained2d=True,
                 patch_size=(4, 4, 4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 # out_indices=(0, 1, 2, 3),
                 # frozen_stages=-1,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()

        assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % num_heads[0] == 0, f'embed_dim [{embed_dim}] is not divisible by num_head[0]=={num_heads[0]}'

        # self.pretrained = pretrained
        # self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        # self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                downsample=PatchMerging if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.layers.append(layer)

        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # num_features = [int(embed_dim * 2 ** (i+1)) if (i < self.num_layers - 1)
        #                 else int(embed_dim * 2 ** (i)) for i in range(self.num_layers)]
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        assert np.all(k % v for k, v in zip(num_features, num_heads)), 'embed_dim/ num_features not a multiple of heads'
        self.num_features = num_features

        # add a norm layer for each output
        # self.norm = norm_layer(self.num_features)
        # for i_layer in out_indices:
        #     layer = norm_layer(num_features[i_layer])
        #     layer_name = f'norm{i_layer}'
        #     self.add_module(layer_name, layer)
        self.norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = norm_layer(num_features[i_layer])
            self.norms.append(layer)

    def init_weights(self):
        """Initialize the weights in backbone.

        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        outs = []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            print(i, norm, x.shape)
            x = layer(x.contiguous())
            x = rearrange(x, 'n c d h w -> n d h w c')
            x = norm(x)
            x = rearrange(x, 'n d h w c -> n c d h w')
            outs.append(x)

        # for i in range(self.num_layers):
        #     layer = self.layers[i]
        #     x = layer(x.contiguous())
        #     # print(x.shape)
        #     if i in self.out_indices:
        #         name = f"norm{i}"
        #         # norm = getattr(self, f"norm{i}")
        #         norm = getattr(self, name)
        #         x = rearrange(x, 'n c d h w -> n d h w c')
        #         # C, D, Wh, Ww = x.size(1), x.size(2), x.size(3), x.size(4)
        #         # x = x.flatten(2).transpose(1, 2)
        #         # x = norm(x)
        #         # x = x.transpose(1, 2).view(-1, C, D, Wh, Ww)
        #         print(i, x.shape, name, norm)
        #         x = norm(x)
        #         x = rearrange(x, 'n d h w c -> n c d h w')
        #         outs.append(x)
        return tuple(outs)
        # for layer in self.layers:
        #     x = layer(x.contiguous())
        #
        # x = rearrange(x, 'n c d h w -> n d h w c')
        # x = self.norm(x)
        # x = rearrange(x, 'n d h w c -> n c d h w')
        #
        # return x


class WindowAttentionDecode3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        Linear = nn.Linear if n_divs == 1 else HyperLinear
        # self.qkv = Linear(dim, dim * 3, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.kv = Linear(dim, dim * 2, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})  # from skip
        self.q = Linear(dim, dim * 1, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})  # from input
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]  # make torchscript happy (cannot use tensor as tuple)

        By, Ny, Cy = y.shape
        kv = self.kv(y).reshape(By, Ny, 2, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        attn = attn.type_as(v)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerDecode3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_divs=1, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)

        self.norm_mixed = norm_layer(dim)
        self.attn_mixed = WindowAttentionDecode3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, n_divs=n_divs)
        # # #####
        #
        # self.norm1 = norm_layer(dim)
        # self.attn = WindowAttention3D(
        #     dim, window_size=self.window_size, num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)
        #
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, n_divs=n_divs)

    def get_window(self, x):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            # attn_mask = mask_matrix
        else:
            shifted_x = x
            # attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        return x_windows, (Dp, Hp, Wp, pad_r, pad_b, pad_d1), (window_size, shift_size)

    def self_attn(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def mixed_attn(self, x, y, mask_matrix):
        B, D, H, W, C = x.shape

        x_windows, (Dp, Hp, Wp, pad_r, pad_b, pad_d1), (window_size, shift_size) = self.get_window(x)
        y_windows, _, _ = self.get_window(y)

        # cyclic shift
        if any(i > 0 for i in shift_size):
            attn_mask = mask_matrix
        else:
            attn_mask = None


        # W-MSA/SW-MSA
        attn_windows = self.attn_mixed(x_windows, y_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part1_new(self, x, y, mask_matrix):
        shortcut = x
        # print(shortcut.shape)
        # self attention
        x = self.norm1(x)
        x = self.self_attn(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        # attention with skip connection
        shortcut = x
        # print(x.shape, y.shape, 'before')
        x = self.norm_mixed(y)
        # print(x.shape, y.shape, 'after')
        x = self.mixed_attn(x, y, mask_matrix)
        x = shortcut + self.drop_path(x)
        return x

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, y, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # shortcut = x
        if self.use_checkpoint:
            # x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
            x = checkpoint.checkpoint(self.forward_part1_new, x, y, mask_matrix)
        else:
            # x = self.forward_part1(x, mask_matrix)
            x = self.forward_part1_new(x, y, mask_matrix)
        # x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class EncodeLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim//2, norm_layer=norm_layer, n_divs=n_divs)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])

        # self.downsample = downsample
        # if self.downsample is not None:
        #     self.downsample = downsample(dim=dim, norm_layer=norm_layer, n_divs=n_divs)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """

        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class DecodeLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim=dim*2, norm_layer=norm_layer, n_divs=n_divs)

        # print(depth)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerDecode3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])

        # self.downsample = downsample
        # if self.downsample is not None:
        #     self.downsample = downsample(dim=dim, norm_layer=norm_layer, n_divs=n_divs)

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
            y: Input feature, tensor size (B, C, D, H1, W1).
        """
        if self.upsample is not None:
            x = rearrange(x, 'b c d h w -> b d h w c')
            x = self.upsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')

        B, C, D, H, W = x.shape
        _, _, _, H1, W1= y.shape

        if H1 != H or W1 != W:
            # x = x.view(B, H, W, -1)
            x = x[:, :, :, :H1, :W1].contiguous()
            # x = x.view(B, H1 * W1, -1)
            # H, W = H1, W1

        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        y = rearrange(y, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            # print(x.shape, y.shape)
            x = blk(x, y, attn_mask)
        # print(x.shape)
        x = rearrange(x, 'b d h w c -> b c d h w')
        # x = x.view(B, D, H, W, -1)
        #
        # if self.downsample is not None:
        #     x = self.downsample(x)
        # x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class Head(nn.Module):
    def __init__(self, decode_dim=96, ratio=4, out_chans=None, img_size=(64, 64),
                 in_depth=1, out_depth=1,
                 n_divs=1):
        super().__init__()
        assert decode_dim % n_divs == 0, f'dim [{decode_dim}] is not divisible by n_divs [{n_divs}]'
        self.out_chans = out_chans if out_chans else decode_dim
        self.img_size = img_size  # to_3tuple(img_size)
        self.ratio = ratio
        # self.out_depth = out_depth
        self.expand = PatchExpanding(decode_dim, ratio=ratio, factor=ratio, n_divs=n_divs)

        self.constant_shape = in_depth == out_depth
        self.final = nn.Sequential(
            nn.Linear(in_features=decode_dim * (1 if self.constant_shape else in_depth),
                      out_features=self.out_chans * (1 if self.constant_shape else out_depth)),
            nn.Sigmoid()
        )


    def forward(self, x):  # , H, W):
        # B, C, D, H, W = x.shape

        H1, W1 = self.img_size
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.expand(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        B, C, D, H, W = x.shape

        if H1 != H or W1 != W:
            x = x[:, :, :, H1, :W1].contiguous()

        if self.constant_shape:
            x = rearrange(x, 'b c d h w -> b d h w c')
        else:
            x = rearrange(x, 'b c d h w -> b h w (c d)')
        x = self.final(x)
        if self.constant_shape:
            x = rearrange(x, 'b d h w c -> b c d h w')
        else:
            x = rearrange(x, 'b h w (c d) -> b c d h w', c=self.out_chans)
        return x


class LearnVectorBlock3D(nn.Module):
    def __init__(self, in_channels, featmaps, filter_size, act_layer=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = featmaps

        padding = (filter_size[0] // 2, filter_size[1] // 2, filter_size[2] // 2)
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=featmaps, kernel_size=filter_size, padding=padding),
            act_layer()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

    def extra_repr(self) -> str:
        extra_str = f"in_channels={self.in_channels}, out_channels={self.out_channels}, activation={self.activation}"
        return extra_str


class SwinEncoderDecoderTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int/tuple): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self,
                 img_size=(256, 256),
                 in_depth=1, out_depth=1,
                 patch_size=(1, 4, 4),
                 in_chans=3,
                 out_chans=None,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()

        # assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % num_heads[0] == 0, f'embed_dim [{embed_dim}] is not divisible by num_head[0]=={num_heads[0]}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        n_chans = int(np.ceil(in_chans / n_divs) * n_divs)
        self.learn_vector = nn.Sequential(
            nn.Conv3d(in_channels=in_chans, out_channels=n_chans, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.GELU()
        ) if n_chans != in_chans else nn.Identity()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=n_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        assert np.all(k % v for k, v in zip(num_features, num_heads)), 'embed_dim/ num_features not a multiple of heads'
        self.num_features = num_features

        # build up layers
        self.uplayers = nn.ModuleList()
        self.decode_dim = num_features[::-1]

        for i_layer in range(self.num_layers):
            # print(i_layer)
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            decode_dim = self.decode_dim[i_layer]
            layer = DecodeLayer(
                dim=decode_dim,
                depth=depths[i_layer],  # 1,  #depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                upsample=PatchExpanding if i_layer != 0 else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.uplayers.append(layer)

        self.head = Head(decode_dim=decode_dim, ratio=self.patch_size[-1], out_chans=out_chans, n_divs=n_divs,
                         img_size=self.img_size, in_depth=in_depth, out_depth=out_depth)

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.learn_vector(x)
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        features = []
        # hw = []
        # for i in range(self.num_layers):
        #     layer = self.layers[i]
        #     layer.H, layer.W = H, W
        #     # x, H, W = layer(x, H, W)
        #     x = layer(x)
        #     H, W = layer.H, layer.W
        #     # print(i, H, W, x.shape)
        #     features.insert(0, x)
        #     hw.insert(0, (H, W))

        # outs = []
        for layer in self.layers:
            # print(x.shape)
            # x = layer(x.contiguous())
            x = layer(x)
            features.insert(0, x)
            # x = rearrange(x, 'n c d h w -> n d h w c')
            # x = norm(x)
            # x = rearrange(x, 'n d h w c -> n c d h w')
            # outs.append(x)

        for layer, y in zip(self.uplayers, features):
            # print(x.shape, y.shape)
            x = layer(x, y)

        # for i in range(self.num_layers):
        #     layer = self.uplayers[i]
        #     y = features[i]
        #     H1, W1 = hw[i]
        #     # print(i, H, W, H1, W1)
        #     layer.H, layer.W, layer.H1, layer.W1 = H, W, H1, W1
        #     # x, H, W = layer(x, y, H, W, H1, W1)
        #     x = layer(x, y)
        #     H, W = layer.H, layer.W
        #     # if i == 0:
        #     #     y = features[i]
        #     #     x, H, W = layer(x, y, H, W)
        #     # else:
        #     #     x, H, W = layer(x, H, W)
        # self.head.H, self.head.W = H, W
        # # x = self.head(x, H, W)
        # print(x.shape)
        x = self.head(x)

        return x

n_div_dict = {'sedenion': 16,
              'octonion': 8,
              'quaternion': 4,
              'complex': 2,
              'real': 1}

class HyperSwinEncoderDecoder3D(SwinEncoderDecoderTransformer3D):
    """Hypercomplex Swin Transformer encoder-decoder 3D architecture."""
    def __init__(self, args):
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        heads_ = 8
        n_multiples_in = int(n_divs * np.ceil(args.len_seq_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf < embed_dim:
            warnings.warn(f"args.sf = {args.sf} < embed_dim used [{embed_dim}]")
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))

        super().__init__(depths=tuple([args.nb_layers] * args.stages),
                         num_heads=tuple([8] * args.stages),
                         out_chans=args.len_seq_out,
                         in_chans=args.len_seq_in,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(args.height, args.width),
                         in_depth=args.depth, out_depth=len(args.target_vars),
                         n_divs=n_divs,
                         drop_rate=args.dropout,
                         patch_size=(1, *([args.patch_size] * 2))
                         )