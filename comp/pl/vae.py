import logging
import pytorch_lightning as pl
import torch
import torch.distributions as tdist

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class VAE(pl.LightningModule):
    """VAE class with flexible encoder and decoder"""

    def __init__(
        self, encoder, decoder, z_dim, learning_rate=0.001, gamma=1.0, beta=1.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set the encoder and decoder networks
        self.encoder = encoder
        self.decoder = decoder

        # Store the output dimension
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta

    def forward(self, x, c=None, d=None):
        """Compute the approximate posterior for input data x; c and d are ignored.

        Returns:
            an instance of torch.distributions.Normal representing the variational  posterior q(z | x)
        """
        return self.encoder(x)[0]

    def forward_decoder(self, z, c=None, d=None):
        """Compute the decoded reconstruction of x for latent vectors z; c and d are ignored.

        Returns:
            an instance of torch.distributions.Distribution representing the likelihood p(x | z)
        """
        return self.decoder(z)

    def _compute_elbo(self, x):
        z, _ = self.encoder(x)
        z_prior = tdist.Normal(torch.zeros_like(z.loc), torch.ones_like(z.scale))
        x_hat = self.forward_decoder(z.rsample())

        reconstruction = x_hat.log_prob(x).sum()
        kl = tdist.kl.kl_divergence(z, z_prior).sum()
        self.log("log p(x | z)", reconstruction)
        self.log("kl(q|p)", kl)
        elbo = reconstruction - self.beta * kl

        return elbo  # / x.shape[0]

    def training_step(self, batch, batch_idx):
        """Execute one training step"""
        x, *_ = batch
        elbo = self._compute_elbo(x)
        self.log("train_elbo", elbo)
        self.log("train_loss", -elbo)
        return {"loss": -elbo, "elbo": elbo}

    def validation_step(self, batch, batch_idx):
        """Execute one training step"""
        x, *_ = batch
        elbo = self._compute_elbo(x)
        self.log("valid_elbo", elbo)
        self.log("valid_loss", -elbo)
        return {"loss": -elbo, "elbo": elbo}

    def configure_optimizers(self):
        """Fetch the optimiser parameters - required by PyTorch Lightning"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]
