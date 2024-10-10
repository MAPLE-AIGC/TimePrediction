# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
import random
from typing import Sequence
import lpips
import torch.nn as nn


        
class ImageLPIPSLoss:
    def __init__(
        self, 
        P_mean=-0.6, 
        P_std=1.6, 
        sigma_data: float = 1.0,
        p_uncond: float = .1,
        text_loss_weight: float = 0.1,
        dist: str = 'lognormal',
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.p_uncond = p_uncond
        self.dist = dist
        
        sigma_max = 80.0
        sigma_min = 0.002
        rho = 7
        val_denoising_step = 1000
        step_indices = torch.arange(val_denoising_step)
        sigmas = (sigma_max ** (1 / rho) + step_indices / (val_denoising_step - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.sigmas = sigmas    
        self.text_loss_weight = text_loss_weight

        #self.loss_fn = lpips.LPIPS(net='vgg')
        self.mse_loss_fn = nn.MSELoss()


    def __call__(self, net, batch):
        # for name, param in net.module.timestep_predictor.named_parameters():
        #     print(f"{name}: {param}")
        device = 'cuda'
        self.mse_loss_fn.to(device)
        images = batch["images"].to(device)
        prompts = batch["prompts"]
        latents = net.module.vae.encode(images).latent_dist.sample()
        
        # (noise - x_0)
        latents_x_0, sigma = net(
            latents=latents,
            prompts=prompts, 
        )

        #lpips_loss = self.loss_fn(images, new_image)
        #lpips_loss = lpips_loss.mean(dim=0)
        mse_loss = self.mse_loss_fn(latents, latents_x_0)
        #print(mse_loss)
        mse_loss = mse_loss.unsqueeze(0)
        
        sigma_penalty = torch.zeros_like(sigma).to(device)
        sigma_penalty += torch.where(sigma > 0.9, (sigma - 0.9) ** 2, torch.zeros_like(sigma))
        sigma_penalty += torch.where(sigma < 0.1, (0.1 - sigma) ** 2, torch.zeros_like(sigma))

        sigma_loss=sigma_penalty.mean()
        # Final loss
        total_loss = mse_loss + 1000 * sigma_loss  # Adjust penalty weight as needed

        #print(total_loss)
        return {
            "mse_loss": mse_loss,
            "sigma_loss": sigma_loss,
            "total_loss": total_loss,
        }
        


