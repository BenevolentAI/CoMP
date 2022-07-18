from dataclasses import dataclass


@dataclass
class ModelConfig:
    model: str
    latent_dim: int
    hidden_dim: int
    num_layers: int
    penalty_scale: float
    cvae_penalty: str
    kl_beta: float
    use_batchnorm: bool
    bandwidth: float
    penalise_z: bool
    rbf_version: int


@dataclass
class TrainConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    gamma: float
    check_val_every_n_epoch: int
