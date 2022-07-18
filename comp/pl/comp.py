import logging

import pytorch_lightning as pl
import torch
import torch.distributions as dist

LARGE_NEGATIVE = -10000000.0
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def pairwise_label_mask(m: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch_size x n_classes tensor, where each row is a one-hot encoding of
    the class label, to a batch_size x batch_size symmetric tensor indciating whether
    the [i, j]th objects in the batch are in the same class.
    """
    return torch.einsum("ik,jk->ij", m, m)


def make_subset_penalty_v2(
    posterior,
    categories,
    latent,
    penalty_scale=1.0,
):
    """Subset penalty

    Args:
        posterior (torch distribution): The q(z|x) distribution with shape=(batch_size, num_features)
        categories (torch Tensor): The categories tensor with shape=(batch_size, 2).
            The rows are one hot vectors.
        penalty_scale (float): The positive penalty scale factor

    Returns:
        (torch Tensor): The penalty scalar.
    """
    # Computes the cross product of log probs, shape B x B, (i.e. batch sample x source_distribution)
    batch_size = latent.shape[0]
    log_probs = posterior.log_prob(latent.unsqueeze(1))
    mask_temp = pairwise_label_mask(categories)
    mask_own = mask_temp - torch.eye(batch_size, device=latent.device)
    mask_other = 1.0 - mask_temp
    mixture_size_own = mask_own.sum(-1)
    mixture_size_other = batch_size - 1 - mixture_size_own

    log_prob_own = (mask_own * log_probs + (1 - mask_own) * LARGE_NEGATIVE).logsumexp(
        -1
    ) - torch.log(mixture_size_own)
    log_prob_other = (
        mask_other * log_probs + (1 - mask_other) * LARGE_NEGATIVE
    ).logsumexp(-1) - torch.log(mixture_size_other)

    contrastive_penalty = penalty_scale * (log_prob_own - log_prob_other).mean()
    return contrastive_penalty


class COMP(pl.LightningModule):
    """Conditional VAE with flexible encoder and decoder"""

    def __init__(
        self,
        encoder,
        decoder,
        z_dim,
        penalty_scale=1.0,
        learning_rate=0.001,
        gamma=1.0,
        beta=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.z_dim = z_dim
        self.penalty_scale = penalty_scale
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta

        LOGGER.info("Using penalty scale: %f", self.penalty_scale)

        # Set the encoder and decoder networks
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """Compute the approximate posterior for input data x

        Returns:
            an instance of torch.distributions.Normal representing the variational  posterior q(z | x)
        """
        encoder_input = [x, c]
        return self.encoder(torch.cat(encoder_input, dim=-1))[0]

    def forward_decoder(self, z, c):
        inputs = [z, c]
        return self.decoder(torch.cat(inputs, dim=-1))

    def _compute_elbo(
        self,
        x: torch.Tensor,
        categories: torch.Tensor,
        qz: dist.Distribution,
        qz_penalty: dist.Distribution,
    ):
        latents = qz.rsample()
        z_prior = dist.Independent(
            dist.Normal(torch.zeros_like(latents), torch.ones_like(latents)), 1
        )
        x_hat = self.forward_decoder(latents, categories)

        reconstruction = x_hat.log_prob(x)
        if reconstruction.dim() == 2:
            reconstruction = reconstruction.sum(axis=1)
        kl = dist.kl.kl_divergence(qz, z_prior)
        elbo = (reconstruction - self.beta * kl).mean()

        penalty = make_subset_penalty_v2(
            qz_penalty,
            categories,
            latents,
            self.penalty_scale,
        )
        elbo -= penalty
        return elbo, penalty

    def _execute_step(self, batch, batch_idx, stage):
        assert (
            len(batch) == 2
        ), f"Expected batch of [x, categories], got {[x.shape for x in batch]}"
        x, categories = batch

        enc_inputs = batch[:2]
        qz_temps = self.encoder(torch.cat(enc_inputs, dim=-1))
        qz = dist.Independent(qz_temps[0], 1)
        qz_penalty = dist.Independent(qz_temps[1], 1)
        elbo, penalty = self._compute_elbo(
            x, categories=categories, qz=qz, qz_penalty=qz_penalty
        )
        self.log(f"{stage}_elbo", elbo)
        self.log(f"{stage}_loss", -elbo)
        self.log(f"{stage}_penalty", penalty)
        return {"loss": -elbo, "elbo": elbo, "penalty": penalty}

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
