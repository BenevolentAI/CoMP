import logging

import pytorch_lightning as pl
import torch
import torch.distributions as tdist

from comp.nn.loss import GroupwiseMMD, RBF

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class CVAE(pl.LightningModule):
    """Conditional VAE with flexible encoder and decoder"""

    def __init__(
        self,
        encoder,
        decoder,
        z_dim,
        penalty="mmd",
        penalty_scale=0.0,
        learning_rate=0.001,
        gamma=1.0,
        beta=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set the encoder and decoder networks
        self.encoder = encoder
        self.decoder = decoder
        self.mmd_op = GroupwiseMMD(RBF(length_scale=1.0))

        # Store the output dimension
        self.z_dim = z_dim
        self.penalty = penalty
        self.penalty_scale = penalty_scale
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta

    def forward(self, x, c):
        """
        Compute the approximate posterior for input data x with condition labels c

        Returns:
            An instance of torch.distributions.Normal representing the variational
            posterior q(z | x, c).
        """
        inputs = [x, c]
        return self.encoder(torch.cat(inputs, dim=-1))[0]

    def forward_decoder(self, z, c):
        inputs = [z, c]
        return self.decoder(torch.cat(inputs, dim=-1))

    def _compute_loss(self, x, c, stage):
        z = self.forward(x, c)
        z_prior = tdist.Normal(torch.zeros_like(z.loc), torch.ones_like(z.scale) * 0.1)
        z_sample = z.rsample()
        x_hat = self.forward_decoder(z_sample, c)

        reconstruction = x_hat.log_prob(x).mean()
        self.log(f"{stage}_reconstruction", reconstruction)

        kl = tdist.kl.kl_divergence(z, z_prior).mean()
        self.log(f"{stage}_kl", kl)

        elbo = reconstruction - self.beta * kl
        self.log(f"{stage}_elbo", elbo)

        loss = -elbo
        if self.penalty == "mmd":
            mmd_term = self.mmd_op(c, z_sample)
            self.log(f"{stage}_mmd", mmd_term)
            loss = loss + mmd_term * self.penalty_scale
        self.log(f"{stage}_elbo", elbo)
        self.log(f"{stage}_loss", loss)
        return loss

    def _execute_step(self, batch, batch_idx, stage):
        """Execute one training/validation step"""
        # Pass group labels for loss calculation if theu are in the batch
        x, c = batch
        loss = self._compute_loss(x, c, stage=stage)
        return {"loss": loss, "elbo": -loss}

    def training_step(self, batch, batch_idx):
        """Execute one training step"""
        return self._execute_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        """Execute one validation step"""
        return self._execute_step(batch, batch_idx, stage="valid")

    def configure_optimizers(self):
        """Fetch the optimiser parameters - required by PyTorch Lightning"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]
