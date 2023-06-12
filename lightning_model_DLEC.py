# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch import Tensor
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image

# from neuralcompression.models import ScaleHyperprior
# from compressai.models import ScaleHyperprior

def ycbcr_to_rgb(ycbcr_images):
    # YCbCr to RGB 변환
    rgb_images = []
    for ycbcr_image in ycbcr_images:
        ycbcr_image = transforms.ToPILImage()(ycbcr_image)  # Tensor를 PIL 이미지로 변환
        rgb_image = ycbcr_image.convert("RGB")  # YCbCr을 RGB로 변환
        rgb_image = transforms.ToTensor()(rgb_image)  # PIL 이미지를 Tensor로 변환
        rgb_images.append(rgb_image)
    rgb_images = torch.stack(rgb_images, dim=0)  # 이미지들을 하나의 텐서로 결합
    return rgb_images
        
        
def rgb_to_ycbcr(images):
    # RGB to YCbCr 변환
    ycbcr_images = []
    for image in images:
        image = transforms.ToPILImage()(image)  # Tensor를 PIL 이미지로 변환
        ycbcr_image = image.convert("YCbCr")  # RGB를 YCbCr로 변환
        ycbcr_image = transforms.ToTensor()(ycbcr_image)  # PIL 이미지를 Tensor로 변환
        ycbcr_images.append(ycbcr_image)
    ycbcr_images = torch.stack(ycbcr_images, dim=0)  # 이미지들을 하나의 텐서로 결합
    return ycbcr_images


class LightningModel(LightningModule):
    """
    Model and training loop for the scale hyperprior model.

    Combines a pre-defined scale hyperprior model with its training loop
    for use with PyTorch Lightning.

    Args:
        model: the ScaleHyperprior model to train.
        distortion_lambda: A scaling factor for the distortion term
            of the loss.
        learning_rate: passed to the main network optimizer (i.e. the one that
            adjusts the analysis and synthesis parameters).
        aux_learning_rate: passed to the optimizer that learns the quantiles
            used to build the CDF table for the entropy codder.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        aux_learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.aux_learning_rate = aux_learning_rate
        
        
    def forward(self, x, s):
        ycbcr_images = rgb_to_ycbcr(x).to("cuda")
        luma = ycbcr_images[:,0,:,:].unsqueeze(dim=1)
        chroma = ycbcr_images[:,1:,:,:]
        return self.model(luma, chroma, s)
    
    
    def rate_distortion_loss(
        self,
        reconstruction: Tensor,
        latent_likelihoods: Tensor,
        hyper_latent_likelihoods: Tensor,
        original: Tensor,
        lmbda: float,
        alpha: float
    ):
        num_images, _, height, width = original.shape
        num_pixels = num_images * height * width

        bits = (
            latent_likelihoods.log().sum() + hyper_latent_likelihoods.log().sum()
        ) / -math.log(2)
        bpp_loss = bits / num_pixels

        #ycbcr_reconstruction = rgb_to_ycbcr(reconstruction)
        ycbcr_original = rgb_to_ycbcr(original).to("cuda")
        
        luma_recon = reconstruction[:,0,:,:].unsqueeze(dim=1)
        chroma_recon = reconstruction[:,1:,:,:]
        luma_original = ycbcr_original[:,0,:,:].unsqueeze(dim=1)
        chroma_original = ycbcr_original[:,1:,:,:]

        distortion_loss_luma = F.mse_loss(luma_recon, luma_original)
        distortion_loss_chroma = F.mse_loss(chroma_recon, chroma_original)
        
        distortion_loss = ((1 * distortion_loss_luma) + (2 * alpha * distortion_loss_chroma)) / (1 + 2 * alpha)
        combined_loss = lmbda * 255 ** 2 * distortion_loss + bpp_loss

        return bpp_loss, distortion_loss, combined_loss

    def update(self, force=True):
        return self.model.update(force=force)

    def compress(
        self, x: Tensor, s: int
    ) -> Tuple[List[str], List[str], Sequence[int], Sequence[int], Sequence[int]]:
        ycbcr_images = rgb_to_ycbcr(x).to("cuda")
        luma = ycbcr_images[:,0,:,:].unsqueeze(dim=1)
        chroma = ycbcr_images[:,1:,:,:]
        return self.model.compress(luma, chroma, s)

    def decompress(
        self, strings, shape, s
    ):
        return self.model.decompress(
            strings, shape, s
        )

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx not in [0, 1]:
            raise ValueError(
                f"Received unexpected optimizer index {optimizer_idx}"
                " - should be 0 or 1"
            )

        if optimizer_idx == 0:
            # x_hat, y_likelihoods, z_likelihoods = self(batch) # Meta version
            s = random.randint(0, self.model.levels-1)
            out = self(batch, s) ## 어케 저렇게 되지..?
            # out = self.model(batch, self.s) 
            
            bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
                # x_hat, y_likelihoods, z_likelihoods, batch    # Meta version
                out['x_hat'], out['likelihoods']['y'], out['likelihoods']['z'], batch, self.model.lmbda[s], self.model.alpha[s]
            )
            self.log_dict(
                {
                    "bpp_loss": bpp_loss.item(),
                    "distortion_loss": distortion_loss.item(),
                    "psnr": 10 * math.log10(1 * 1 / distortion_loss.item()),
                    "loss": combined_loss.item(),
                },
                sync_dist=True,
            )
            return combined_loss

        else:
            # This is the loss for learning the quantiles of the
            # distribution for the hyperprior.
            # quantile_loss = self.model.quantile_loss()    # Meta version
            quantile_loss = self.model.aux_loss()
            self.log("quantile_loss", quantile_loss.item(), sync_dist=True)
            return quantile_loss

    def validation_step(self, batch, batch_idx):
        # x_hat, y_likelihoods, z_likelihoods = self(batch) # Meta version
        s = random.randint(0, self.model.levels-1) 

        out = self(batch, s)
        #out['x_hat'] = ycbcr_to_rgb(out['x_hat'])
        bpp_loss, distortion_loss, combined_loss = self.rate_distortion_loss(
            # x_hat, y_likelihoods, z_likelihoods, batch    # Meta version
            out['x_hat'], out['likelihoods']['y'], out['likelihoods']['z'], batch, self.model.lmbda[s], self.model.alpha[s]
        )

        self.log_dict(
            {
                "val_loss": combined_loss.item(),
                "val_distortion_loss": distortion_loss.item(),
                "val_psnr": 10 * math.log10(1 * 1 / distortion_loss.item()),
                "val_bpp_loss": bpp_loss.item(),
            },
            sync_dist=True,
        )

    def configure_optimizers(self):
        # Meta version
        # model_param_dict, quantile_param_dict = self.model.collect_parameters()   
        # optimizer = optim.Adam(
            # model_param_dict.values(),
            # lr=self.learning_rate,
        # )
        # aux_optimizer = optim.Adam(
            # quantile_param_dict.values(),
            # lr=self.aux_learning_rate,
        # )
        
        parameters = {
            n
            for n, p in self.model.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
    
        aux_parameters = {
            n
            for n, p in self.model.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }
        params_dict = dict(self.model.named_parameters())

        optimizer = optim.Adam(
            (params_dict[n] for n in sorted(parameters)),
            lr=self.learning_rate,
        )
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=self.aux_learning_rate,
        )

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=20)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6], gamma=0.5)
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            },
            {"optimizer": aux_optimizer},
        )
