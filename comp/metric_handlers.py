import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from umap import UMAP

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
UMAP_SEED = 42


def get_umap_params(dataset_name):
    params = dict(seed=UMAP_SEED)
    if dataset_name == "uci-income":
        params.update(a=0.1, b=2.0, n_neighbours=100)
    return params


def calc_metrics(
    output_dir: str,
    dataset: torch.utils.data.TensorDataset,
    sample_metadata_df: pd.DataFrame,
    models: List[Union[torch.nn.Module, Path, str]],
    dataset_name: str,
    model_type: str,
    load_model_fn: Callable[[Path], pl.LightningModule],
    use_cuda: bool,
):
    """
    models can be either a list of models or list of paths to Pytorch Lightning checkpoints
    to load.
    """
    for i, m in enumerate(models):
        if isinstance(m, Path) or isinstance(m, str):
            model = load_model_fn(m)
            model_id = Path(m).stem
        elif isinstance(m, torch.nn.Module):
            model = m
            model_id = f"model-{i}"
        else:
            raise ValueError(
                f"Invalid model reference, must be a torch.nn.Module or Path to checkpoint: {m}"
            )
        LOG.info(f"Calculating outputs and metrics for {model_type} {model_id}")
        is_conditional_model = model_type != "vae"
        features, _ = predict(
            model, dataset, conditional_model=is_conditional_model, use_cuda=use_cuda
        )
        feature_array = features.cpu().numpy()

        latent_df = pd.DataFrame(
            data=feature_array,
            index=sample_metadata_df.index,
            columns=[f"Z{i}" for i in range(features.shape[1])],
        )
        LOG.info("Generated latents for full data set: %s", latent_df.shape)
        latent_file = os.path.join(
            output_dir, "latents", f"latents-{str(model_id)}.parquet"
        )
        LOG.info("Saving loc from latent distribution to %s", latent_file)
        latent_df.to_parquet(latent_file)

        umap_params = get_umap_params(dataset_name)
        LOG.info(f"Computing UMAP projection with params {umap_params}.")
        umap_df = calc_umap(feature_array, sample_metadata_df, **umap_params)
        LOG.info(f"UMAP projection complete.")

        umap_path = os.path.join(output_dir, "umaps", f"umap-{str(model_id)}.parquet")
        LOG.info(f"Saving checkpoint metrics: {str(umap_path)}")
        umap_df.to_parquet(umap_path)


def predict(
    model: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    conditional_model: bool,
    use_cuda: bool,
) -> Tuple[torch.Tensor, float]:
    if use_cuda:
        model.cuda()
    model.eval()
    with torch.no_grad():
        # inputs is a concatentation of input features and conditional one-hot labels
        # if the model is a conditional one, others it just contains the input features.
        assert 0 < len(dataset.tensors) < 3
        if len(dataset.tensors) == 2 and conditional_model:
            x, c = dataset.tensors
        else:
            x = dataset.tensors[0]
            c = None
        qz = model.forward(x, c)
        decoder_result = model.forward_decoder(qz.loc, c)
        if model.decoder.return_hidden:
            logprob = decoder_result[0].log_prob(x)
        else:
            logprob = decoder_result.log_prob(x)
    return qz.loc, logprob.mean().item()


def calc_umap(
    latents_array: np.ndarray,
    metadata_df: pd.DataFrame,
    n_neighbours: int = 15,
    a: Optional[float] = None,
    b: Optional[float] = None,
    seed: int = UMAP_SEED,
) -> pd.DataFrame:
    n_components = 2
    df = pd.DataFrame(
        UMAP(
            n_components=n_components,
            a=a,
            b=b,
            n_neighbors=n_neighbours,
            random_state=seed,
            transform_seed=42,
        ).fit_transform(latents_array),
        index=metadata_df.index,
        columns=[f"umap-{i}" for i in range(n_components)],
        dtype=np.float32,
    )
    return df
