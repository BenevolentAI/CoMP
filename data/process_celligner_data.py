"""Prepare tumour/cell line dataset from Celligner"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.utils import create_arg_parser, save_output

logging.basicConfig(
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
FILENAMES = dict(
    hgnc_file="hgnc_complete_set_7.24.2018.txt",
    tumor_file="TCGA_mat.tsv",
    cl_file="CCLE_mat.csv",
    info_file="Celligner_info.csv",
)


def load_tumour_cl(input_dir, top_var_number=8000):
    """Load the tumour_cl data

    Args:
        input_dir (pathlib.Path): The input folder with files given in FILENAMES dict.
        top_var_number (int): Filtering for the top high variance genes.

    Returns:
        feature_df (pd.DataFrame): n x p data frame containing the gene expression features
        metadata_df (pd.DataFrame): n x m data frame containing metadata. For tumour_cl, this
            contains the type (CL or tumor), the disease and the disease subtype (if known)
    """
    hgnc_df = pd.read_csv(input_dir / FILENAMES["hgnc_file"], delimiter="\t")
    info_df = pd.read_csv(input_dir / FILENAMES["info_file"], index_col=0)
    # Convert gex tables to float32 to save memory. Doing so within read_csv throws an error.
    tumor_df = (
        pd.read_csv(input_dir / FILENAMES["tumor_file"], delimiter="\t", index_col=0)
        .set_index("Gene")
        .astype(np.float32)
        .T
    )
    tumor_df = tumor_df.loc[:, ~tumor_df.columns.duplicated()]
    cl_df = pd.read_csv(input_dir / FILENAMES["cl_file"], index_col=0).astype(
        np.float32
    )
    cl_df.columns = cl_df.columns.map(lambda s: s.split(" (ENS")[0])
    cl_df = cl_df.loc[:, ~cl_df.columns.duplicated()]

    common_genes = cl_df.columns & tumor_df.columns
    cl_df = cl_df[common_genes]
    assert cl_df.shape[1] == len(
        common_genes
    ), f"{cl_df.shape[1]} != {len(common_genes)}"
    tumor_df = tumor_df[common_genes]
    assert tumor_df.shape[1] == len(common_genes)

    # Filter most varying genes
    if top_var_number is not None:
        LOGGER.info(f"Variance filtering with {top_var_number} top gene features.")
        cl_top = list(
            cl_df.var(axis=0).sort_values(ascending=False).iloc[0:top_var_number].index
        )
        tumor_top = list(
            tumor_df.var(axis=0)
            .sort_values(ascending=False)
            .iloc[0:top_var_number]
            .index
        )
        all_top = list(set(cl_top + tumor_top))
        cl_df = cl_df.loc[:, all_top]
        tumor_df = tumor_df.loc[:, all_top]

    gex_df = pd.concat([tumor_df, cl_df])

    func_genes = set(
        hgnc_df[~hgnc_df.locus_group.isin(["non-coding RNA", "pseudogene"])].symbol
    )
    gex_df = gex_df[gex_df.columns.intersection(func_genes)]
    LOGGER.info(f"No. of selected gene features: {gex_df.shape[1]}")
    info_df = info_df.loc[gex_df.index]
    return gex_df, info_df[["disease", "subtype", "type"]]


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    LOGGER.info(f"Processing Tumour/Cell line dataset from Celligner")
    data_cellinger, metadata_celligner = load_tumour_cl(
        input_dir=input_dir, top_var_number=args.top_var_number
    )

    LOGGER.info(f"data_cellinger.shape {data_cellinger.shape}")
    LOGGER.info(f"metadata_celligner.shape {metadata_celligner.shape}")

    LOGGER.info(f"Saving Tumour/Cell dataset in {output_dir}")
    save_output(output_dir, data_cellinger, metadata_celligner)


if __name__ == "__main__":
    main()
