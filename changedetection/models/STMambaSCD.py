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
    def __init__(self, output_cd, output_clf, pretrained,  **kwargs):
        super(STMambaSCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
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

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.change_attention_1 = MultiScaleChangeGuidedAttention(
                            channels_list=[
                                128,
                                256,
                                512,
                                1024
                            ]
                        )
        
        self.change_attention_2 = MultiScaleChangeGuidedAttention(
                            channels_list=[
                                128,
                                256,
                                512,
                                1024
                            ]
                        )


        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)

        
        self.diffusion = GaussianDiffusion(
            CustomUnet(  # Use the custom UNet
                dim=128,
                channels=6,  # Input channels (3 for pre, 3 for post)
                dim_mults=(2, 4, 8, 16),
                full_attn=(False, False, False, True) 
            ),
            image_size=256,
            timesteps=500,
            min_snr_loss_weight=True,  # Use min-SNR loss weighting
            min_snr_gamma=5,  # Gamma value for min-SNR loss weighting
            offset_noise_strength=0.1  # Add offset noise
        )

        self.attn_proj_list = nn.ModuleList([
            nn.Conv2d(128*i, 128*2*i, kernel_size=1, bias=False) for i in [1, 2, 4, 8]
        ])

    def forward(self, pre_data, post_data):
        pre_data = pre_data.float()
        post_data = post_data.float()
        x = torch.cat([pre_data, post_data], dim=1)  # Shape: [batch, 6, height, width]
        
        batch_size = x.shape[0]
        timesteps = torch.randint(0, 500, (batch_size,)).to(x.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.diffusion.q_sample(x, timesteps, noise=noise)
        
        timesteps = timesteps.float()
        if x_noisy.dtype != torch.float32:
            x_noisy = x_noisy.float()
        

        predicted_noise, attention_maps = self.diffusion.model(x_noisy, timesteps)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        
        # Encoder features
        pre_features = self.encoder(pre_data)  # e.g., [128@256x256, 256@128x128, 512@64x64, 1024@32x32]
        post_features = self.encoder(post_data)
        
        attention_maps_pre = []
        attention_maps_post = []
        assert len(attention_maps) == len(pre_features), "Mismatch between attention maps and feature levels"
        for i, (attn_map, pre_feat, post_feat) in enumerate(zip(attention_maps, pre_features, post_features)):
            target_size = pre_feat.shape[-2:]
            channel_size = pre_feat.shape[1]
            if attn_map.shape[-2:] != target_size:
                attn_map = self.attn_proj_list[i](attn_map)
                attn_map_pre = attn_map[:, :channel_size]
                attn_map_post = attn_map[:, channel_size:]
                attn_map_pre = F.interpolate(attn_map_pre, size=target_size, mode='bilinear', align_corners=False)
                attn_map_post = F.interpolate(attn_map_post, size=target_size, mode='bilinear', align_corners=False)
                attention_maps_pre.append(torch.sigmoid(attn_map_pre))
                attention_maps_post.append(torch.sigmoid(attn_map_post))
            else:
                attn_map_pre = attn_map[:, :channel_size]
                attn_map_post = attn_map[:, channel_size:]
                attention_maps_pre.append(torch.sigmoid(attn_map_pre))
                attention_maps_post.append(torch.sigmoid(attn_map_post))
        
        pre_features = [feat * attn for feat, attn in zip(pre_features, attention_maps_pre)]
        post_features = [feat * attn for feat, attn in zip(post_features, attention_maps_post)]
        # Decoder processing - passing encoder outputs to the decoder
        output_bcd = self.decoder_bcd(pre_features, post_features)
        """
        pre_features -> something with outputbcd
        """
        pre_features = self.change_attention_1(pre_features, output_bcd)
        post_features = self.change_attention_2(post_features, output_bcd)


        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)

        output_bcd = self.main_clf_cd(output_bcd)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')
        
        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')


        return output_bcd, output_T1, output_T2, diffusion_loss


class CustomUnet(Unet):
    def forward(self, x, t):
        # Initial convolution
        x = self.init_conv(x)  # [batch, 128, H, W]
        r = x.clone()  # Residual connection
        
        # Time embedding
        time_emb = self.time_mlp(t)
        
        h = []
        attention_maps = []
        
        # Downsampling path
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            h.append(x)
            
            x = block2(x, time_emb)
            h.append(x)
            
            if attn is not None:
                x = attn(x) + x  # Apply attention with residual
                attention_maps.append(x)  # Collect attention map
            
            x = downsample(x)
        
        # Middle blocks
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time_emb)
        
        # Upsampling path
        for block1, block2, attn, upsample in self.ups:
            skip1 = h.pop()
            skip2 = h.pop()
            
            x = torch.cat((x, skip1), dim=1)
            x = block1(x, time_emb)
            
            x = torch.cat((x, skip2), dim=1)
            x = block2(x, time_emb)
            
            x = upsample(x)
        
        # Final output
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, time_emb)
        final_output = self.final_conv(x)
        
        return final_output, attention_maps