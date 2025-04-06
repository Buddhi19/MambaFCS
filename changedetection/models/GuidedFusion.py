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


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor
    
    def forward(self, pre_feat, post_feat):
        B, C, H, W = pre_feat.size()
        # Generate query, key, and value
        query = self.query_conv(pre_feat).view(B, -1, H * W).permute(0, 2, 1)  # B, H*W, C
        key = self.key_conv(post_feat).view(B, -1, H * W)  # B, C, H*W
        value = self.value_conv(post_feat).view(B, -1, H * W)  # B, C, H*W
        
        # Compute attention scores and apply softmax
        attention = torch.bmm(query, key)  # B, H*W, H*W
        attention = F.softmax(attention, dim=-1)
        
        # Compute weighted output
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, H*W
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable gamma
        out = self.gamma * out + pre_feat
        return out