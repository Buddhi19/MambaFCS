import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# GCD-DDPM Components
class DDPMModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDPMModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        return self.decoder(self.encoder(x) + t.view(-1, 1, 1, 1))

class DifferenceConditionalEncoder(nn.Module):
    def __init__(self, channels_list):
        super(DifferenceConditionalEncoder, self).__init__()
        self.channels_list = channels_list
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            ) for ch in channels_list
        ])

    def forward(self, pre_features, post_features):
        diff_features = []
        for pre_f, post_f, layer in zip(pre_features, post_features, self.layers):
            diff = layer(post_f - pre_f)
            diff_features.append(diff)
        return diff_features

class NoiseSuppressionSemanticEnhancer(nn.Module):
    def __init__(self):
        super(NoiseSuppressionSemanticEnhancer, self).__init__()
        self.fft_layer = FFTLayer()

    def forward(self, x):
        x_freq = self.fft_layer(x)
        x_filtered = x_freq * self.attention_map(x_freq)
        x_filtered = self.fft_layer.inverse(x_filtered)
        return x_filtered
    
class FFTLayer(nn.Module):
    def __init__(self):
        super(FFTLayer, self).__init__()

    def forward(self, x):
        x_freq = torch.fft.fftn(x, dim=(-2, -1))
        return x_freq

    def inverse(self, x_freq):
        x = torch.fft.ifftn(x_freq, dim=(-2, -1))
        return x