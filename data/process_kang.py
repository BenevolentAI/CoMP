"""Prepare Kang et al. PBMC scRNA-seq under INFb stimulation dataset"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from data.utils import create_arg_parser, save_output

logging.basicConfig(
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
FILENAMES = dict(
    counts="kang_count.h5ad",
)


def load_kang(input_dir, top_var_number=2000):
    """Load the PBMC scRNA-seq with IFNb simulation dataset.

    Args:
        input_dir (pathlib.Path): The input folder with files given in KANG dict, saved from
            https://github.com/theislab/trVAE_reproducibility
        top_var_number (int): Filtering for the top high variance genes.

    Returns:
        feature_df (pd.DataFrame): n x p data frame containing the gene expression features
        metadata_df (pd.DataFrame): n x m data frame containing metadata. For Kang, this
            contains the perturbation status
    """
    adata = sc.read(input_dir / FILENAMES["counts"])
    LOGGER.info(f"Read data into scanpy, data size: {adata.X.shape}")
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    LOGGER.info(f"Normalised data")
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=top_var_number)
    adata = adata[:, adata.var["highly_variable"]]
    LOGGER.info(
        f"Subsetted data using {top_var_number} most highly variable genes, data size: {adata.X.shape}"
    )
    gex = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)
    adata.obs["type"] = np.where(
        adata.obs["stim"] == "CTRL", "unperturbed", "perturbed"
    )
    adata.obs["cell"] = adata.obs["cell_type"]
    LOGGER.info(f"Outputting gene expression and metadata DataFrames")
    return gex, adata.obs


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    LOGGER.info(f"Processing PBMC scRNA-seq with IFNb simulation dataset")
    data_kang, metadata_kang = load_kang(
        input_dir=input_dir, top_var_number=args.top_var_number
    )

    LOGGER.info(f"data_kang.shape {data_kang.shape}")
    LOGGER.info(f"metadata_kang.shape {metadata_kang.shape}")

    LOGGER.info(f"Saving Tumour/Cell dataset in {output_dir}")
    save_output(output_dir, data_kang, metadata_kang)


if __name__ == "__main__":
    main()
