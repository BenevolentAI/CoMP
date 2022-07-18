import torch
import torch.distributions as dist
import torch.nn as nn


def calc_input_dims(tensor_dataset, model_config):
    gene_expression_dim = tensor_dataset.tensors[0].shape[1]
    if model_config.model in ["cvae", "comp", "trvae"]:
        assert len(tensor_dataset.tensors) in [2, 3]
        label_dims = tensor_dataset.tensors[1].shape[1]
        encoder_input_dim = gene_expression_dim + label_dims
        decoder_input_dim = model_config.latent_dim + label_dims
    elif model_config.model in ["vae"]:
        encoder_input_dim = gene_expression_dim
        decoder_input_dim = model_config.latent_dim
    else:
        assert False, f"{model_config.model} is not handled when creating encoder"
    return encoder_input_dim, decoder_input_dim


def _make_batchnorm_layer(input_dim: int, use_batchnorm: bool) -> nn.Module:
    if use_batchnorm:
        return nn.BatchNorm1d(input_dim)
    else:
        return nn.Identity()


class Encoder(nn.Module):
    """Encoder used in all models, consisting of a multilayer perceptron with RELU activations"""

    def __init__(
        self, x_dim, z_dim, hidden_dim, n_layers=1, use_batchnorm=False, bandwidth=0.1
    ):
        super().__init__()
        self.n_layers = n_layers
        if n_layers == 0:
            # Linear Encoder
            self.fc21 = nn.Linear(x_dim, z_dim)  # mean
            self.fc22 = nn.Linear(x_dim, z_dim)  # scale
        else:
            # Non-linear Encoder
            self.fc1 = nn.Linear(x_dim, hidden_dim)
            self.batchnorm1 = _make_batchnorm_layer(hidden_dim, use_batchnorm)
            self.fc21 = nn.Linear(hidden_dim, z_dim)  # mean
            self.fc22 = nn.Linear(hidden_dim, z_dim)  # scale

        # setup the non-linearities
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.bandwidth = bandwidth

        if n_layers > 1:
            self.middle: nn.Module = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        _make_batchnorm_layer(hidden_dim, use_batchnorm),
                        self.relu,
                    )
                    for i in range(n_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()

    def forward(self, x):
        if self.n_layers == 0:
            hidden = x
        else:
            hidden = self.relu(self.batchnorm1(self.fc1(x)))
            hidden = self.middle(hidden)

        z_loc = self.fc21(hidden)
        z_scale = self.bandwidth * torch.ones(z_loc.shape, device=z_loc.device)
        posterior = dist.Normal(z_loc, z_scale)
        penalty_dist = posterior

        return posterior, penalty_dist


class GaussianDecoder(nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        hidden_dim,
        n_layers=1,
        return_hidden=False,
        use_batchnorm=False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.batchnorm1 = _make_batchnorm_layer(hidden_dim, use_batchnorm)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.fc22 = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU()  # To match TrVAE
        self.softplus = nn.Softplus()
        self.return_hidden = return_hidden
        if n_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        _make_batchnorm_layer(hidden_dim, use_batchnorm),
                        self.relu,
                    )
                    for i in range(n_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()

    def forward(self, z):
        hidden_0 = self.relu(self.batchnorm1(self.fc1(z)))
        hidden_1 = self.middle(hidden_0)
        xhat = self.fc21(hidden_1)
        x_sd = 1e-2 + self.softplus(self.fc22(hidden_1)).expand(xhat.shape)
        if self.return_hidden:
            return dist.Normal(xhat, x_sd), hidden_0
        else:
            return dist.Normal(xhat, x_sd)
