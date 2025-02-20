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
        
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels)
        )
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels*2, 1, 1),
            nn.Sigmoid()
        )
        self.out_Layer = nn.Conv2d(in_channels*4, in_channels*2, 1)

    def forward(self, pre, post):
        # Basic concatenation
        cat = torch.cat([pre, post], dim=1)
        
        # Difference features
        diff = self.diff_conv(torch.abs(pre - post))
        
        # Attention-guided fusion
        attn_map = self.attn(cat)
        attn_fused = attn_map * pre + (1 - attn_map) * post
        
        # Final fusion
        out = torch.cat([cat, diff, attn_fused], dim=1)
        out = self.out_Layer(out)
        return out

        