import torch
import torch.nn as nn
import torch.nn.functional as F
from RemoteSensing.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute


class MambaGF(nn.Module):
    def __init__(self, **kwargs):
        super(MambaGF, self).__init__()

        self.vssm_T1 = VSSM(
            hidden_dim=kwargs['hidden_dim'], drop_path=0.1, norm_layer=kwargs['norm_layer'], channel_first=kwargs['channel_first'],
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=kwargs['ssm_act_layer'],
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=kwargs['mlp_act_layer'], mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
        )

        self.vssm_T2 = VSSM(
            hidden_dim=kwargs['hidden_dim'], drop_path=0.1, norm_layer=kwargs['norm_layer'], channel_first=kwargs['channel_first'],
            ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=kwargs['ssm_act_layer'],
            ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
            forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=kwargs['mlp_act_layer'], mlp_drop_rate=kwargs['mlp_drop_rate'],
            gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']
        )

        self._LAYER_NORM_1 = nn.LayerNorm(kwargs['hidden_dim'])
        self._LAYER_NORM_2 = nn.LayerNorm(kwargs['hidden_dim'])

        self.cross_attention = nn.MultiheadAttention(embed_dim=kwargs['hidden_dim'], num_heads=kwargs['num_heads'])

        self.SIGMOID = nn.Sigmoid()

    def forward(self, pre_feature, post_feature):
        ori_pre_feature = pre_feature
        ori_post_feature = post_feature

        pre_feature = self.vssm_T1(pre_feature)
        post_feature = self.vssm_T2(post_feature)

        pre_feature = self._LAYER_NORM_1(pre_feature)
        post_feature = self._LAYER_NORM_2(post_feature)

        pre_feature = pre_feature.flatten(2).permute(2, 0, 1)  # (N, C, H, W) -> (HW, N, C)
        post_feature = post_feature.flatten(2).permute(2, 0, 1)  # (N, C, H, W) -> (HW, N, C)

        attn_output, _ = self.cross_attention(pre_feature, post_feature, post_feature)
        attn_output = attn_output.permute(1, 2, 0).view_as(ori_pre_feature)  # (HW, N, C) -> (N, C, H, W)

        guided_feature = pre_feature * self.SIGMOID(attn_output)
        return guided_feature