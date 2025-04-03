import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from RemoteSensing.changedetection.models.Mamba_backbone import Backbone_VSSM
from RemoteSensing.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from RemoteSensing.changedetection.models.ChangeDecoder import ChangeDecoder
from RemoteSensing.changedetection.models.SemanticDecoder import SemanticDecoder
from RemoteSensing.changedetection.models.MultiScaleChangeGuidedAttention import MultiScaleChangeGuidedAttention
from RemoteSensing.changedetection.DiffusionBranch.denoising_diffusion_pytorch import GaussianDiffusion, Unet


class STMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained_diffusion, **kwargs):
        super(STMambaSCD, self).__init__()
        # Use the Unet from pretrained GaussianDiffusion as the encoder
        self.encoder = pretrained_diffusion.model  # Extract Unet from GaussianDiffusion
        
        # Define encoder dimensions based on Unet structure
        # Assuming dim=64, dim_mults=(1, 2, 4, 8), reversed for upsampling
        # Adjust these based on your actual Unet configuration
        self.encoder.dims = [64, 128, 256, 512]  # e.g., features at 256x256, 128x128, 64x64, 32x32
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = True  # Unet uses channel-first (B, C, H, W)

        norm_layer = _NORMLAYERS.get(kwargs['norm_layer'].lower(), nn.LayerNorm)        
        ssm_act_layer = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), nn.SiLU)
        mlp_act_layer = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), nn.GELU)

        # Clean kwargs for decoder initialization
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        # Decoders
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        # Multi-scale attention modules
        self.change_attention_1 = MultiScaleChangeGuidedAttention(
            channels_list=self.encoder.dims
        )
        
        self.change_attention_2 = MultiScaleChangeGuidedAttention(
            channels_list=self.encoder.dims
        )

        # Output convolutions
        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)

    def forward(self, pre_data, post_data):
        # Extract upsampling features from DDPM Unet
        # time=None, but internally set to 0 when extract_features=True
        pre_features = self.encoder(pre_data, time=None, extract_features=True)
        post_features = self.encoder(post_data, time=None, extract_features=True)
        
        # Process through decoders
        output_bcd = self.decoder_bcd(pre_features, post_features)
        pre_features = self.change_attention_1(pre_features, output_bcd)
        post_features = self.change_attention_2(post_features, output_bcd)

        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)

        # Final predictions with upsampling
        output_bcd = self.main_clf_cd(output_bcd)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')
        
        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')

        return output_bcd, output_T1, output_T2
