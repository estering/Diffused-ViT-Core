from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 224
    gaussian_radius: float = 1.0
    rotation_degrees: float = 15.0
    crop_scale_min: float = 0.9


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int
    beta_start: float
    beta_end: float
    time_embedding_dim: int = 128
    hidden_dim: int = 1536
    residual_blocks: int = 3


@dataclass(frozen=True)
class ModelConfig:
    dropout: float
    feature_dim: int = 768
    classifier_hidden_dim: int = 512
    pretrained: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    learning_rate_decay: float
    diffusion_weight: float
    workers: int
    seed: int
