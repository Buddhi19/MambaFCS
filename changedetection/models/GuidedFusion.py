import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return (avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))

class PyramidFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Multi-scale processing branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.channel_att = ChannelAttention(out_channels * 3)
        self.spatial_att = SpatialAttention()
        
        # Fusion
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        
        # Multi-scale features
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        
        # Concatenate and apply attention
        combined = torch.cat([b1, b3, b5], dim=1)
        channel_weights = self.channel_att(combined)
        weighted = combined * channel_weights
        spatial_weights = self.spatial_att(weighted)
        weighted = weighted * spatial_weights
        
        # Final fusion
        fused = self.final_conv(weighted)
        return fused + residual

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                 padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))

from einops import rearrange
import math


class CrossAttention(nn.Module):
    """
    A simple Transformer-style cross-attention on flattened feature maps.
    This can capture deeper correlations between two sets of features (pre/post).
    """
    def __init__(self, dim, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=True)

        # Dropouts for attention + final projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        """
        x_q: [B, HW, C] - queries (e.g., pre_feat flattened)
        x_kv: [B, HW, C] - keys/values (e.g., post_feat flattened)
        Returns: [B, HW, C] 
        """
        B, N, C = x_q.shape

        # Project Q from x_q
        qkv_q = self.qkv(x_q).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv_q = qkv_q.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

        # Project K,V from x_kv
        qkv_kv = self.qkv(x_kv).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv_kv = qkv_kv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

        # Split out Q, K, V
        q = qkv_q[0]  # [B, num_heads, N, head_dim]
        k = qkv_kv[0] # [B, num_heads, N, head_dim]
        v = qkv_kv[1] # [B, num_heads, N, head_dim]
        # (Note: the 3rd slice in each qkv_ array is for Q, but we only need
        #  Q from x_q and K,V from x_kv in this cross-attention design.)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.attn_drop(attn_scores)

        x_att = attn_scores @ v  # [B, num_heads, N, head_dim]
        x_att = x_att.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # Final linear projection
        x_att = self.proj(x_att)
        x_att = self.proj_drop(x_att)

        return x_att



class ChannelGate(nn.Module):
    """
    ECA‑style channel attention.
    Conv1d expects shape [B, 1, C] so we must transpose BEFORE the conv.
    """
    def __init__(self, C, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(C) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                 # x: [B, C, H, W]
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1)         # [B, C, 1]
        y = self.conv(y.transpose(1, 2))                    # [B, 1, C]
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)   # [B, C, 1, 1]
        return x * y


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        avg_val = torch.mean(x, dim=1, keepdim=True)
        gate = self.conv(torch.cat([max_val, avg_val], dim=1))
        return x * self.sigmoid(gate)


# ────────────────────────────────────────────────────────────────
# FFT helper: converts spatial map → log‑magnitude frequency map
# ────────────────────────────────────────────────────────────────
class FFTBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):                                    # x: [B,C,H,W]
        amp = torch.log1p(torch.abs(torch.fft.fft2(x, norm="ortho")))
        return self.proj(amp.real)                           # preserve H×W


# ────────────────────────────────────────────────────────────────
# Upgraded CrossAttentionFusion with Dual‑Frequency branch
# ────────────────────────────────────────────────────────────────
class FFT_Fusion(nn.Module):
    def __init__(self, in_channels, use_diff=True,
                 cross_attn_heads=4, freq_ratio=1):
        super().__init__()
        self.use_diff = False
        c_freq = int(in_channels * freq_ratio)

        self.FFT_BRANCH = False
        # FFT branches
        fusion_in = 2* in_channels   
        
        if self.FFT_BRANCH:            # pre+post spatial
            self.fft_pre  = FFTBranch(in_channels, c_freq)
            self.fft_post = FFTBranch(in_channels, c_freq)

            fusion_in = 2 * (in_channels + c_freq)               # pre+post spatial+freq
        if self.use_diff:
            fusion_in += in_channels                         # |pre‑post|

        self.reduce_conv = nn.Conv2d(fusion_in, in_channels, 1, bias=False)
        self.reduce_bn   = nn.BatchNorm2d(in_channels)
        self.reduce_relu = nn.ReLU(inplace=True)

        self.ch_gate = ChannelGate(in_channels)
        self.sp_gate = SpatialGate()

    def forward(self, pre_feat, post_feat):
        cat = torch.cat([pre_feat, post_feat], dim=1)

        if self.FFT_BRANCH:
            pre_freq  = self.fft_pre(pre_feat)
            post_freq = self.fft_post(post_feat)

            cat = torch.cat([pre_feat, pre_freq,
                            post_feat, post_freq], dim=1)

        if self.use_diff:
            cat = torch.cat([cat, torch.abs(pre_feat - post_feat)], dim=1)

        fused = self.reduce_relu(self.reduce_bn(self.reduce_conv(cat)))  # [B,C,H,W]

        fused = self.ch_gate(fused)
        fused = self.sp_gate(fused)
        return fused