import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleChangeGuidedAttention(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(128, channels, kernel_size=1)
            for channels in channels_list
        ])
        
    def forward(self, features_list, change_map):
        conditioned_features = []
        for i, features in enumerate(features_list):
            # Down sample change_map to match feature size
            _, _, H, W = features.shape
            change_resized = F.interpolate(change_map, size=(H, W), mode='bilinear')
            
            # Compute attention
            attention = torch.sigmoid(self.conv_layers[i](change_resized))
            
            # Condition features
            conditioned_features.append(features * (1 + attention))
        
        return conditioned_features