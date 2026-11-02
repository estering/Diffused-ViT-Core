import math
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from .config import DiffusionConfig, ModelConfig

ENDPOINT_PROB = 0.1
INFERENCE_XI = 0.5


class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device) * -scale
        )
        angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, feature):
        return feature + self.layers(feature)


class FeatureDenoiser(nn.Module):
    def __init__(self, model_config, diffusion_config):
        super().__init__()
        self.time_embedding = TimeEmbedding(diffusion_config.time_embedding_dim)
        self.input_layers = nn.Sequential(
            nn.Linear(
                model_config.feature_dim + diffusion_config.time_embedding_dim,
                diffusion_config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(diffusion_config.hidden_dim, model_config.feature_dim),
        )
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(model_config.feature_dim, model_config.dropout)
                for _ in range(diffusion_config.residual_blocks)
            ]
        )

    def forward(self, feature, timestep):
        timestep_feature = self.time_embedding(timestep)
        feature = torch.cat((feature, timestep_feature), dim=1)
        return self.residual_blocks(self.input_layers(feature))


@dataclass
class DiffusedViTOutput:
    logits: torch.Tensor
    clean_feature: torch.Tensor
    refined_feature: torch.Tensor


class DiffusedViT(nn.Module):
    def __init__(self, model_config, diffusion_config):
        super().__init__()
        self.model_config = model_config
        self.diffusion_config = diffusion_config

        weights = (
            ViT_B_16_Weights.IMAGENET1K_V1
            if self.model_config.pretrained
            else None
        )
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()
        self.denoiser = FeatureDenoiser(self.model_config, self.diffusion_config)
        self.classifier = nn.Sequential(
            nn.Linear(
                self.model_config.feature_dim,
                self.model_config.classifier_hidden_dim,
            ),
            nn.LayerNorm(self.model_config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.model_config.dropout),
            nn.Linear(self.model_config.classifier_hidden_dim, 1),
        )

        betas = torch.linspace(
            self.diffusion_config.beta_start,
            self.diffusion_config.beta_end,
            self.diffusion_config.timesteps,
        )
        alpha_bars = torch.cat(
            (torch.ones(1), torch.cumprod(1.0 - betas, dim=0))
        )
        self.register_buffer("alpha_bars", alpha_bars)

    def _sample_timestep(self, batch, device):
        t = torch.randint(
            1,
            self.diffusion_config.timesteps + 1,
            (batch,),
            device=device,
        )
        endpoint = torch.rand(batch, device=device) < ENDPOINT_PROB
        return torch.where(endpoint, torch.zeros_like(t), t)

    def add_noise(self, clean_feature, timestep):
        noise = torch.randn_like(clean_feature)
        alpha_bar = self.alpha_bars[timestep].unsqueeze(1)
        return (
            alpha_bar.sqrt() * clean_feature
            + (1.0 - alpha_bar).sqrt() * noise
        )

    def forward(self, image):
        clean_feature = self.backbone(image)
        if self.training:
            timestep = self._sample_timestep(
                image.shape[0], image.device
            )
            feature = self.add_noise(clean_feature, timestep)
        else:
            timestep = torch.zeros(
                image.shape[0], dtype=torch.long, device=image.device
            )
            feature = clean_feature

        refined_feature = self.denoiser(feature, timestep)
        logits = self.classifier(refined_feature).squeeze(1)
        return DiffusedViTOutput(logits, clean_feature, refined_feature)
