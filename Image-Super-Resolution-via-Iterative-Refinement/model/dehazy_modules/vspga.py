# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @2022 AIR, CAS. (mails.ucas.ac.cn)
#
# Vision transformer with stable prior and global attention for cross-domain image dehazing  (VSPGA)
# @author: yuanzhiqiang <yuanzhiqiang@mails.ucas.ac.cn>
# @date: 2022/05/22

# Based on the code of paper "Vision Transformers for Single Image Dehazing"

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.cuda.amp import autocast
import torch.fft
from inspect import isfunction

def exists(x):
    return x is not None

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class FeatureWiseAffine(nn.Module):

    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class GlobalAttention(nn.Module):
    def __init__(self, dim, window_size, patch_dim = 30, att_g_a = 30, att_g_b = 20):
        super().__init__()

        patch_src_dim = window_size **2 * dim

        self.patch_dim = patch_dim
        self.dim = dim

        self.patch_embed = nn.Sequential(
            nn.Linear(patch_src_dim, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, self.patch_dim, bias=True),
        )

        # att_g parameters
        self.att_g_alpha, self.att_g_beta = nn.Parameter(torch.FloatTensor([0])), nn.Parameter(torch.FloatTensor([0]))
        self.att_g_a, self.att_g_b = att_g_a, att_g_b

        self.eps = 1e-8


    def l2norm(self, X, dim):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + self.eps
        X = torch.div(X, norm)
        return X

    def l1norm(self, X, dim, eps=1e-8):
        """L1-normalize columns of X
        """
        norm = torch.sum(X, dim=dim, keepdim=True) + eps
        X = torch.div(X, norm)
        return X

    def get_mean_std(self, input):
        if len(input.shape) == 4:
            dim = (1, 2, 3)
        elif len(input.shape) == 3:
            dim = (1, 2)
        else:
            raise NotImplementedError

        mean = torch.mean(input, dim=dim, keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=dim, keepdim=True) + self.eps)
        return mean, std

    def forward(self, windows, window_size, H, W):

        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, -1) # ([1, 28, 28, 8, 8, 24]) B, n_patch, n_patch, dim

        # get mean and std
        src_mean, src_std = self.get_mean_std(x)

        # x => B , n_patch^2, n_patch^2
        x = self.patch_embed(x).view(B, -1, self.patch_dim)
        x = self.l2norm(x, dim=-1)

        # get att_g
        att_g = torch.bmm(x, x.permute(0, 2, 1))
        att_g = torch.exp(self.att_g_a * self.att_g_alpha * att_g + self.att_g_beta * self.att_g_b)
        att_g = self.l1norm(att_g, dim=-1)

        # windows = > B x n_pixel x dim, n_patch^2
        reshaped_windows = windows.view(B, (H // window_size) * (W // window_size), -1)

        # global attention
        reshaped_windows = torch.bmm(att_g, reshaped_windows)

        cur_mean, cur_std = self.get_mean_std(x)

        reshaped_windows = (reshaped_windows - cur_mean) / cur_std
        reshaped_windows = reshaped_windows * src_std.squeeze(dim=-1) + src_mean.squeeze(dim=-1)

        # reverse size
        reshaped_windows = reshaped_windows.view(-1, window_size**2, self.dim)

        return reshaped_windows

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

class MultiFreqAttention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None, default_spec_size=224, device=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        self.default_spec_size = default_spec_size

        self.eps = 1e-5

        # pre convs

        # diff spec constructs
        self.group_dims = self.dim

        # conv for high freqs
        self.hf_convs = nn.Sequential(
            nn.Conv2d(self.group_dims, self.group_dims, kernel_size=5, padding=2, groups=self.group_dims, padding_mode='reflect'),
            nn.ReLU(True),
            nn.Conv2d(self.group_dims, self.group_dims, kernel_size=5, padding=2, groups=self.group_dims, padding_mode='reflect')
        )

        # SA block for low freqs
        self.lf_V = nn.Conv2d(self.group_dims, self.group_dims, 1)
        self.lf_QK = nn.Conv2d(self.group_dims, self.group_dims * 2, 1)
        self.lf_attn = WindowAttention(self.group_dims, window_size, num_heads)
        self.lf_attn_global = GlobalAttention(self.group_dims, window_size)

        # Spec Filter for dynamic freqs
        # self.complex_weight= nn.Parameter(torch.randn(self.default_spec_size, self.default_spec_size // 2 + 1, self.group_dims, dtype=torch.float32) * 0.02, requires_grad=True)
        # self.register_parameter("matrixs", self.complex_weight)

        # Fusion Stage
        self.fusion = SKFusion(dim=self.group_dims, height=4, reduction=2, keep_dim=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def feature_norm(self, X, mean_, std_):
        mean = torch.mean(X, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((X - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (X - mean) / std

        out = normalized_input * mean_ + std_
        return out

    def forward(self, X):
        B, C, H, W = X.shape

        # split
        # LF, HF, DF = torch.split(X, [self.group_dims] * 3, dim=1)
        # LF, HF, DF = X, X, X
        LF, HF = X, X

        # ===========================================
        # H conv
        hf_features = self.hf_convs(HF)
        # hf_features = self.feature_norm(hf_features, f_mean, f_std)

        # ===========================================
        # L SA
        lf_V = self.lf_V(LF)
        lf_QK = self.lf_QK(LF)
        lf_QKV = torch.cat([lf_QK, lf_V], dim=1)

        shifted_QKV = self.check_size(lf_QKV, self.shift_size > 0)
        Ht, Wt = shifted_QKV.shape[2:]

        shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
        qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

        attn_windows = self.lf_attn(qkv)

        attn_windows = self.lf_attn_global(attn_windows, self.window_size, Ht, Wt)

        # merge windows
        shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

        # reverse cyclic shift
        out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
        lf_features = out.permute(0, 3, 1, 2)

        # # ===========================================
        # # D Spec Filter
        # DF = DF.permute(0, 2, 3, 1)
        # DF_spec = torch.abs(torch.fft.rfft2(DF, dim=(1, 2), norm='ortho'))
        #
        # # spec upsample
        # DF_spec = F.upsample(DF_spec.permute(0, 3, 1, 2), size=(self.default_spec_size, self.default_spec_size // 2 + 1),
        #                mode='bilinear').permute(0, 2, 3, 1)
        # DF_spec = DF_spec * self.complex_weight
        #
        # # spec downsample
        # DF_spec = F.upsample(DF_spec.permute(0, 3, 1, 2), size=(H, W)).permute(0, 2, 3, 1)
        # df_features = torch.fft.irfft2(DF_spec, s=(H, W), dim=(1, 2), norm='ortho')
        # df_features = df_features.permute(0, 3, 1, 2)

        # ===========================================
        # Fusion
        # cat_features = self.fusion([lf_features.unsqueeze(1), hf_features.unsqueeze(1)])
        # out_features = self.feature_norm(cat_features, f_mean, f_std)
        cat_features = 0.5 * (lf_features + hf_features)


        return cat_features

class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)
            self.attn_global = GlobalAttention(dim, window_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows = self.attn(qkv)

            # global attention
            attn_windows = self.attn_global(attn_windows, self.window_size, Ht, Wt)

            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None,attention_type='Attention',device=None,
                 noise_level_mlp=None
                 ):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim)
        if attention_type == 'Attention':
            self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)
        elif attention_type == 'SpecAttention':
            self.attn = MultiFreqAttention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type, device=device)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

        self.is_bn = norm_layer == nn.BatchNorm2d

        self.noise_level_mlp = noise_level_mlp

        self.noise_func = FeatureWiseAffine(dim, dim, use_affine_level=False)

    def forward(self, x, t):
        identity = x

        if self.is_bn:
            x = self.norm1(x)
            x = self.attn(x)
        else:
            if self.use_attn: x, rescale, rebias = self.norm1(x)
            x = self.attn(x)
            if self.use_attn: x = x * rescale + rebias
        x = identity + x

        x = self.noise_func(x, t)

        identity = x
        if self.is_bn:
            x = self.norm2(x)
            x = self.mlp(x)
        else:
            if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
            x = self.mlp(x)
            if self.use_attn and self.mlp_norm: x = x * rescale + rebias
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None,attention_type='Attention',device=None,
                 with_noise_level_emb=True
                 ):

        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = dim
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(dim),
                nn.Linear(dim, dim * 4),
                Swish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type,
                             attention_type=attention_type,
                             device = device,
                             noise_level_mlp=self.noise_level_mlp
                             )
            for i in range(depth)])

    def forward(self, x, t):
        t = self.noise_level_mlp(t) if exists(
            self.noise_level_mlp) else None

        for blk in self.blocks:
            x = blk(x, t)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8, keep_dim=False, block_lens=3):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        if keep_dim:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(dim, d, kernel_size=3, padding=1, stride=2, bias=False),
                nn.ReLU(),
                nn.Conv2d(d, dim, kernel_size=3, padding=1, stride=2, bias=False)
            )
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(dim, d, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(d, dim * height, 1, bias=False)
            )

        self.softmax = nn.Softmax(dim=1)
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, in_feats):

        if self.keep_dim:
            B, _, C, H, W = in_feats[0].shape

            in_feats = torch.cat(in_feats, dim=1)

            feats_sum = torch.sum(in_feats, dim=1)

            attn = self.mlp(feats_sum)
            attn = self.softmax(self.avg_pool(attn).reshape(B, -1)).unsqueeze(-1).unsqueeze(-1)

            out = feats_sum * attn
        else:
            B, C, H, W = in_feats[0].shape

            in_feats = torch.cat(in_feats, dim=1)

            in_feats = in_feats.view(B, self.height, C, H, W)

            feats_sum = torch.sum(in_feats, dim=1)
            attn = self.mlp(self.avg_pool(feats_sum))
            attn = self.softmax(attn.view(B, self.height, C, 1, 1))

            out = torch.sum(in_feats * attn, dim=1)
        return out


class VSPGA(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d],
                 attention_type='Attention',
                 device=None,
                 with_noise_level_emb=True
                 ):
        super(VSPGA, self).__init__()

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.out_chans = out_chans

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0],
                                 attention_type=attention_type, device=device,
                                 )

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1],
                                 attention_type=attention_type, device=device
                                 )

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2],
                                 attention_type=attention_type, device=device
                                 )

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3],
                                 attention_type=attention_type, device=device
                                 )

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4],
                                 attention_type=attention_type, device=device
                                 )

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x, t):
        x = self.patch_embed(x)
        x = self.layer1(x, t)

        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x, t)

        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x, t)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x, t)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x, t)
        x = self.patch_unembed(x)
        return x

    # @autocast()
    def forward(self, x, t):
        x = self.check_image_size(x)
        feat = self.forward_features(x, t)

        return feat

def UNet(device=None, **kwargs):
    return VSPGA(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        norm_layer=[nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d],
        # attention_type = 'SpecAttention'

    )


if __name__ == "__main__":
    model = vspga_s_bn()
    x = torch.randn((2, 6, 224, 256))
    t = torch.tensor([10, 11]).view(2, -1)

    x = model(x, t)
    print(x.shape)