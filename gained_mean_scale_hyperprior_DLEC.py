
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torchvision.transforms import transforms, functional

# pylint: disable=E0611,E0401
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d

from gained_utils import get_scale_table, ResBlock, NonLocalAttention
# pylint: enable=E0611,E0401
import math

import torch
import torch.nn as nn

from compressai.entropy_models import GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.models import ScaleHyperprior

class GainedMeanScaleHyperprior(ScaleHyperprior):
    
    r"""Cui, Ze, et al. "Asymmetric gained deep image compression with continuous rate adaptation." 
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    https://openaccess.thecvf.com/content/CVPR2021/papers/Cui_Asymmetric_Gained_Deep_Image_Compression_With_Continuous_Rate_Adaptation_CVPR_2021_paper.pdf

    .. code-block:: none
                  ┌───┐  ┌───┐   y     ┌───┐  ┌───┐ z  ┌───┐ z_hat      z_hat  ┌───┐  ┌───┐
            x ──►─┤g_a├──|g u|─►─┬──►──┤h_a├──|g u|─►──┤ Q ├───►───·⋯⋯·───►───|igu|──┤h_s├─┐
                  └───┘  └───┘   │     └───┘  └───┘    └───┘        EB         └───┘  └───┘ │
                                 ▼                                                          │
                               ┌─┴─┐                                                        │
                               │ Q │                                                        ▼
                               └─┬─┘                                                        │
                                 │                                                          │
                           y_hat ▼                                                          │
                                 │                                                          │
                                 ·                                                          │
                                 GC : ◄─────────────────────◄───────────────────────────────┘
                                 ·                 scales_hat
                                 │
                           y_hat ▼
                                 │
                  ┌───┐  ┌───┐   │
        x_hat ──◄─┤g_s├──|igu|───┘
                  └───┘  └───┘
        EB = Entropy bottleneck
        GC = Gaussian conditional
        gu: Gained Unit
        igu: Inverse Gained Unit
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """


    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.t_a_zero_luma = nn.Sequential(
            conv(1, N),
            nn.PReLU(),
        )

        self.t_a_zero_luma_inv = nn.Sequential(
            nn.PReLU(),
            deconv(N, 1),
        )

        self.t_a_zero_chroma = nn.Sequential(
            conv(2, N, kernel_size=3, stride=1),
            nn.PReLU(),
        )

        self.t_a_zero_chroma_inv = nn.Sequential(
            nn.PReLU(),
            conv(N, 2, kernel_size=3, stride=1),
        )

        self.t_a_1by1 = nn.Sequential(
            conv(2*N, N, stride=1, kernel_size=1),
            nn.PReLU(),
        )
        
        self.t_a_1by1_inv = nn.Sequential(
            nn.PReLU(),
            conv(N, 2*N, stride=1, kernel_size=1),
        )

        self.t_a_r = nn.Sequential(
            conv(N, N),
            nn.PReLU(),
            conv(N, N),
            nn.PReLU(),
            conv(N, M),
        )

        self.t_a_r_inv = nn.Sequential(
            deconv(M, N),
            nn.PReLU(),
            deconv(N, N),
            nn.PReLU(),
            deconv(N, N),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        
        # self.lmbda = [0.0004, 0.0009, 0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]

        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130]
        #self.lmbda_chroma = [0.00025, 0.00045, 0.0009, 0.0018, 0.0035, 0.0067, 0.0130, 0.0250]
        #self.alpha = [1, 0.25, 0.1, 0.025]
        # self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800]
        # self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

    

    def t_a(self, luma, chroma):
        t_zero_luma = self.t_a_zero_luma(luma)
        down_sampling = transforms.Compose([
            transforms.Resize((chroma.shape[2] // 2, chroma.shape[3] // 2)),
        ])
        chroma = down_sampling(chroma)
        t_zero_chroma = self.t_a_zero_chroma(chroma)
        t_1by1 = self.t_a_1by1(torch.cat([t_zero_luma, t_zero_chroma], dim=1))
        y = self.t_a_r(t_1by1)
        return y

    def t_a_inv(self, y_hat):
        t_r_inv = self.t_a_r_inv(y_hat)
        t_1by1_inv = self.t_a_1by1_inv(t_r_inv)
        luma, chroma = torch.chunk(t_1by1_inv, chunks = 2, dim = 1)
        t_zero_luma_inv = self.t_a_zero_luma_inv(luma)
        t_zero_chroma_inv = self.t_a_zero_chroma_inv(chroma)
        up_sampling = transforms.Compose([
            transforms.Resize((t_zero_chroma_inv.shape[2] * 2, t_zero_chroma_inv.shape[3] * 2)),
        ])
        t_zero_chroma_inv = up_sampling(t_zero_chroma_inv)
        x_hat = torch.cat([t_zero_luma_inv, t_zero_chroma_inv], dim = 1)
        return x_hat
    

    def forward(self, luma, chroma, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        
        y = self.t_a(luma, chroma)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) #Gain[s]: [M] -> [1,M,1,1]
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2,1)
        
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.t_a_inv(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, luma, chroma, s): #, l):
        # assert s in range(0, self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
        # assert l >=0 and l <= 1, "l should in [0,1]"

        InterpolatedGain = torch.abs(self.Gain[s])#.pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
        InterpolatedHyperGain = torch.abs(self.HyperGain[s])#.pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])#.pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)
        
        # InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
        # InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
        # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
        y = self.t_a(luma, chroma)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z = self.h_a(y)
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2,1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        ######    
        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols", means_hat)
        ######
        return {"strings": [y_strings, z_strings], 
                "shape": z.size()[-2:],
                "ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": gained_y_hat}

    def decompress(self, strings, shape, s): #, l):
        assert isinstance(strings, list) and len(strings) == 2
        # assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels-1})"
        # assert l >= 0 and l <= 1, "l should in [0,1]"
        InterpolatedInverseGain = torch.abs(self.InverseGain[s])#.pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])#.pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2,1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        gained_y_hat = y_hat
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.t_a_inv(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}
    
    #####
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def getY(self, x, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0, {self.levels - 1}), but get s:{s}"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1 - l) * torch.abs(self.Gain[s + 1]).pow(l)

        # 如果x不是64的倍数，就对x做padding
        h, w = x.size(2), x.size(3)
        p = 64  # maximum 6 strides of 2
        new_h = (h + p - 1) // p * p  # padding为64的倍数
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top
        x_padded = F.pad(
            x,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )

        y = self.g_a(x_padded)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_quantized = self.gaussian_conditional.quantize(y, "noise")

        return {"ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": y_quantized}

        # return y, y_quantized

    def getScale(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        gaussian_params = self.h_s(z)
        scales, means = gaussian_params.chunk(2, 1)
        return scales

    def getX(self, y_hat, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1 - l) * torch.abs(self.InverseGain[s + 1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1 - l) * torch.abs(
            self.InverseHyperGain[s + 1]).pow(l)

        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    #####