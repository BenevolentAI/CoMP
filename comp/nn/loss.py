from typing import Callable

import torch
import torch.nn as nn


class RBF(nn.Module):
    def __init__(self, length_scale: float):
        super().__init__()
        self._length_scale = length_scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x / self._length_scale, y / self._length_scale)
        return torch.exp(-0.5 * dists)


class MultiScaleRBF(nn.Module):
    def __init__(self, gammas):
        super().__init__()
        self.register_buffer(
            "_gammas",
            torch.unsqueeze(torch.as_tensor(gammas, dtype=torch.float32), dim=1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gamma_dists = torch.matmul(self._gammas, torch.cdist(x, y).reshape(1, -1))
        exps = torch.exp(-0.5 * gamma_dists)
        return torch.sum(exps, axis=0) / len(self._gammas)


def mmd(x1: torch.Tensor, x2: torch.Tensor, kernel: Callable = None) -> torch.Tensor:
    """
    Maximum mean discrepancy (MMD)

    Args:
        x1 (np.ndarray): n x m array representing the first sample
        x2 (np.ndarray): n x m array representing the second sample
        kernel: the kernel function. If not provided, this will use a RBF kernel with length_scale=1

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    kernel = kernel or RBF(length_scale=1.0)
    x1x1 = kernel(x1, x1)
    x1x2 = kernel(x1, x2)
    x2x2 = kernel(x2, x2)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
    return diff


class GroupwiseMMD(nn.Module):
    def __init__(self, kernel: nn.Module):
        super().__init__()
        self.kernel = kernel

    def mmd(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return mmd(x1, x2, self.kernel)

    def forward(self, c: torch.Tensor, z_sample: torch.Tensor):
        d_parts = [(z_sample, c)]
        mmd_term = torch.tensor(0.0, device=z_sample.device)
        for z_group, c_group in d_parts:
            if len(z_group) == 0 or len(c_group) == 0:
                continue
            c0_mask = c_group[:, 0]
            c1_mask = c_group[:, 1]
            if c0_mask.sum() == 0 or c1_mask.sum() == 0:
                continue
            z_0 = z_group[c0_mask.to(torch.bool), :]
            z_1 = z_group[c1_mask.to(torch.bool), :]
            mmd_term += self.mmd(z_0, z_1)
        # average mmd terms over all groupings
        return mmd_term / len(d_parts)
