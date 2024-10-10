# Load model directly
import torch 
import json 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
)
from .unet import UNet2DConditionModel
from typing import Sequence
from typing import List



    
class SD1_4_Text_to_Image(torch.nn.Module):
    
    def __init__(
        self,
        pre_train_model: str = None,
    ):
        super(SD1_4_Text_to_Image, self).__init__()
        #1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(pre_train_model, subfolder="vae").eval()

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(pre_train_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pre_train_model, subfolder="text_encoder").eval()

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(pre_train_model, subfolder="unet").eval()

        
        self.scheduler = DDPMScheduler.from_pretrained(pre_train_model, subfolder="scheduler")
        #self.scheduler.set_timesteps(50)
        # Initialize the TimestepPredictor
        self.pool = nn.AdaptiveAvgPool2d((8, 8)).eval()

        # self.transformer = nn.Transformer(batch_first=True, 
        #                                   d_model=768, 
        #                                   num_encoder_layers=4, 
        #                                   num_decoder_layers=4, 
        #                                   dim_feedforward=768*4, 
        #                                   dropout=0.1)

        self.fc = nn.Sequential(
            nn.Linear(1920*64, 512),
            nn.LayerNorm(512),  # 添加批量归一化
            nn.GELU(),
            nn.Linear(512, 1),
        )
        
        for n, p in self.fc.named_parameters():
            p.requires_grad = True
        # for n, p in self.transformer.named_parameters():
        #     p.requires_grad = True
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
    def encode_prompt(self, prompts):
        # generate prompts
        text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to('cuda'))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to('cuda'))[0]
        return text_embeddings,uncond_embeddings
        
    def step(self, timesteps, latents, noise_pred, new_timestep=None, enable_next_timesteps=False, validation=False, variance_noises=None):
        # Initialize LPIPS with VGG architecture
        batch_size=latents.shape[0]
        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps.to(torch.int32)].view(batch_size, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        
        if enable_next_timesteps==False:
            return pred_original_sample
        else:
            
            predicted_variance = None
            prev_t = new_timestep
            prev_t_clamped = torch.where(prev_t >= 0, prev_t, torch.ones_like(prev_t))
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t_clamped.to(torch.int32)].view(batch_size, 1, 1, 1)
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
            current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

            latents_new = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
            if validation:
                variance = 0
                #if timesteps > 0:
                    # device = noise_pred.device
                #variance_noise = torch.randn_like(latents_new)
                variance_noise = variance_noises[timesteps.to(torch.int32)].to('cuda')
                variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

                # we always take the log of variance, so clamp it to ensure it's not 0
                variance = torch.clamp(variance, min=1e-20)
                variance = (variance ** 0.5) * variance_noise
                latents_new = latents_new + variance
            return latents_new
    def generate(
        self,
        latents: torch.Tensor, 
        t: torch.Tensor, 
        prompt_embeddings: torch.Tensor,
        uncond_embeddings: torch.Tensor,
        predict_next_timetsep: bool,
        do_classifier_free_guidance: bool,
    ):
        if do_classifier_free_guidance:
            prompt_embeddings= torch.cat([uncond_embeddings, prompt_embeddings])
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        #with torch.no_grad():
        noise_pred, sample_features = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeddings, return_dict=False)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            sample_features=[sample_feature[0:1, :, :, :] for sample_feature in sample_features]
        # perform guidance
        #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        if predict_next_timetsep:
            #sigma=self.timestep_predictor(sample_features)[0].to(latents.device)
            pooled1 = self.pool(sample_features[0])  # Shape: (bt, 320, 8, 8)
            pooled2 = self.pool(sample_features[1])  # Shape: (bt, 1280, 8, 8)
            pooled3 = self.pool(sample_features[2])  # Shape: (bt, 320, 8, 8)

            #print(pooled1.shape)
            # Flatten tensors
            flattened1 = pooled1.view(pooled1.size(0), -1)  # Shape: (320, 64)
            flattened2 = pooled2.view(pooled2.size(0), -1)  # Shape: (1280, 64)
            flattened3 = pooled3.view(pooled3.size(0), -1)  # Shape: (320, 64)

            #Concatenate along the first dimension
            combined = torch.cat((flattened1, flattened2, flattened3), dim=1)  # Shape: (bt, 1920, 8, 8)
            #batch_size = combined.size(0)
            #combined = combined.view(batch_size, combined.size(1), -1)  # Shape: (bt, 1920, 64)
            #print(combined.shape)
            # Pass through Transformer
            #transformer_output = self.transformer(combined)  # Transformer expects input of shape (batch, seq_len, d_model)
            #print(transformer_output.shape)
            # Average over sequence length dimension
            #transformer_output = transformer_output.mean(dim=1)  # Shape: (bt, d_model)
            #print(transformer_output.shape)
            # Pass through the fully connected layer
            sigma = torch.sigmoid(self.fc(combined))  # Shape: (bt, 1)
            #sigma = sigma.view(1)

            #print(sigma)
            # 逐元素相乘
            next_timesteps = t *sigma.squeeze()
            next_timesteps = (torch.floor(next_timesteps) - next_timesteps).detach() + next_timesteps
            # offsets = torch.tensor([-30, -25, -35])
            # random_offset = offsets[torch.randint(0, len(offsets), (1,))]
            # next_timesteps = t + random_offset.item()
            #print(next_timesteps)
            #next_timesteps=t-20
            return noise_pred,next_timesteps
        else:
            return noise_pred
        
    def forward(
        self, 
        images: torch.Tensor, 
        prompts: torch.Tensor,
    ):
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        text_embeddings, uncond_embeddings = self.encode_prompt(prompts)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        timesteps = timesteps.expand(latents.shape[0]).to(latents.device)
        #print(timesteps)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        #(noise - x_0)

        noise_pred,next_timestep = self.generate(
            latents=noisy_latents,
            t=timesteps,
            prompt_embeddings=text_embeddings, 
            uncond_embeddings=uncond_embeddings,
            predict_next_timetsep=True,
            do_classifier_free_guidance=False
        )

        latents_t_prime = self.step(timesteps,noisy_latents, noise_pred, next_timestep,enable_next_timesteps=True)

        noise_pred_t_prime = self.generate(
            latents=latents_t_prime,
            t=next_timestep,
            prompt_embeddings=text_embeddings, 
            uncond_embeddings=uncond_embeddings,
            predict_next_timetsep=False,
            do_classifier_free_guidance=False
        )

        latents_x_0 = self.step(next_timestep, latents_t_prime, noise_pred_t_prime, enable_next_timesteps=False)
        # combined=torch.randn(1,320*8*8 + 1280*8*8 + 320*8*8).to(latents.device)
        # sigma = torch.sigmoid(self.fc(combined))
        #latents_x_0=noise_pred*next_timestep
        new_image = self.vae.decode(latents_x_0 / self.vae.config.scaling_factor, return_dict=False)[0]
        return new_image
