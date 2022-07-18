import logging
import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def extract_labels(metadata_df, use_cuda):
    labels = torch.as_tensor(
        OneHotEncoder()
        .fit_transform(metadata_df["type"].to_numpy().reshape(-1, 1))
        .todense(),
        dtype=torch.float32,
    )
    LOGGER.info(f"Using 'type' labels with shape {labels.shape}")
    if use_cuda:
        return labels.cuda()
    else:
        return labels


def prepare_training_data(
    data_dir,
    batch_size,
    include_labels,
    use_cuda,
):
    """
    Loads data and returns a TensorDataset, train and validation loaders, and the metadata
    associated with each sample.

    Args:
        data_dir (str): Path to the directory to load the data from.
        batch_size (int): Batch size for the data loaders.
        include_labels (bool): Whether to create a dataset with labels.
        use_cuda (bool): Use CUDA for the dataset and loaders.

    Returns:
        torch.TensorDataset, torch.DataLoader, torch.DataLoader, pd.DataFrame
    """
    features_df = pd.read_csv(
        os.path.join(data_dir, "features.tsv"), sep="\t", index_col=0
    )
    metadata_df = pd.read_csv(
        os.path.join(data_dir, "metadata.tsv"), sep="\t", index_col=0
    )
    features = torch.as_tensor(features_df.to_numpy(), dtype=torch.float32)

    LOGGER.info("Using data matrix with shape: %s", features.shape)

    if use_cuda:
        input_tensors = [features.cuda()]
    else:
        input_tensors = [features]
    if include_labels:
        input_tensors.append(extract_labels(metadata_df, use_cuda))
    dataset = TensorDataset(*input_tensors)

    L = len(dataset)
    valid_len = L // 10
    LOGGER.info(f"Creating random train/valid split: {L - valid_len}:{valid_len}")
    trainset, validset = random_split(dataset, [L - valid_len, valid_len])
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    valid_loader = DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return dataset, train_loader, valid_loader, metadata_df
