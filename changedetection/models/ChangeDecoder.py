import torch
import torch.nn as nn
import torch.nn.functional as F
from RemoteSensing.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from RemoteSensing.changedetection.models.ResBlockSe import ResBlock, SqueezeExcitation
from RemoteSensing.changedetection.models.GuidedFusion import PyramidFusion, DepthwiseSeparableConv, CrossAttentionFusion, BiDirectionalCrossAttentionFusion

class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-1], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        

        self.st_block_31 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-2], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        

        self.st_block_21 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-3], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        

        self.st_block_11 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-4], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.fuse_layer_1 = BiDirectionalCrossAttentionFusion(in_channels=encoder_dims[-1])
        self.fuse_layer_2 = BiDirectionalCrossAttentionFusion(in_channels=encoder_dims[-2])
        self.fuse_layer_3 = BiDirectionalCrossAttentionFusion(in_channels=encoder_dims[-3])
        self.fuse_layer_4 = BiDirectionalCrossAttentionFusion(in_channels=encoder_dims[-4])

        self.down_sample_1 = DepthwiseSeparableConv(in_channels=encoder_dims[-1], out_channels=encoder_dims[-2])
        self.down_sample_2 = DepthwiseSeparableConv(in_channels=encoder_dims[-2], out_channels=encoder_dims[-3])
        self.down_sample_3 = DepthwiseSeparableConv(in_channels=encoder_dims[-3], out_channels=encoder_dims[-4])
    

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=encoder_dims[-2], out_channels=encoder_dims[-2], stride=1)
        self.smooth_layer_2 = ResBlock(in_channels=encoder_dims[-3], out_channels=encoder_dims[-3], stride=1)
        self.smooth_layer_1 = ResBlock(in_channels=encoder_dims[-4], out_channels=encoder_dims[-4], stride=1)
    

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):
        change_maps = []
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''
        p4 = self.fuse_layer_1(pre_feat_4, post_feat_4)
        p4 = self.st_block_41(p4)

        p4_diff = torch.abs(pre_feat_4 - post_feat_4)
        p4_attention = torch.sigmoid(p4_diff)
        p4 = p4 * p4_attention

        p4 = self.down_sample_1(p4)
        

        '''
            Stage II
        '''
        p3 = self.fuse_layer_2(pre_feat_3, post_feat_3)
        p3 = self._upsample_add(p4, p3)  # Stage number as argument
        p3 = self.smooth_layer_3(p3)
        p3 = self.st_block_31(p3)
        
        # Calculate attention after fusion
        p3_diff = torch.abs(pre_feat_3 - post_feat_3)
        p3_attention = torch.sigmoid(p3_diff)
        p3 = p3 * p3_attention

        p3 = self.down_sample_2(p3)

        '''
            Stage III
        '''
        p2 = self.fuse_layer_3(pre_feat_2, post_feat_2)
        p2 = self._upsample_add(p3, p2)  # Stage number as argument
        p2 = self.smooth_layer_2(p2)
        p2 = self.st_block_21(p2)
        p2_diff = torch.abs(pre_feat_2 - post_feat_2)
        p2_attention = torch.sigmoid(p2_diff)
        p2 = p2 * p2_attention

        p2 = self.down_sample_3(p2)

        '''
            Stage IV
        '''
        p1 = self.fuse_layer_4(pre_feat_1, post_feat_1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)
        p1 = self.st_block_11(p1)
        p1_diff = torch.abs(pre_feat_1 - post_feat_1)
        p1_attention = torch.sigmoid(p1_diff)
        p1 = p1 * p1_attention

        change_maps = [p4_attention, p3_attention, p2_attention, p1_attention]

        return p1, change_maps