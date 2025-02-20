import torch
import torch.nn as nn
import torch.nn.functional as F
from RemoteSensing.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute


class MambaGF(nn.Module):
    def __init__(self,in_channels, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(MambaGF, self).__init__()

        self.vssm_T1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.vssm_T2 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self._BATCH_NORM_1 = nn.BatchNorm2d(128)
        self._BATCH_NORM_2 = nn.BatchNorm2d(128)

        self.SIGMOID = nn.Sigmoid()

        self.CONV = nn.Conv2d(kernel_size=1, in_channels=128, out_channels=in_channels*2)

    def forward(self, pre_feature, post_feature):
        ori_pre_feature = pre_feature
        ori_post_feature = post_feature

        pre_feature = self.vssm_T1(pre_feature)
        post_feature = self.vssm_T2(post_feature)

        pre_feature = self._BATCH_NORM_1(pre_feature)
        post_feature = self._BATCH_NORM_2(post_feature)

        post_feature = self.SIGMOID(post_feature)
        guided_feature = torch.mul(pre_feature, post_feature)
        guided_feature = self.CONV(guided_feature)

        return guided_feature