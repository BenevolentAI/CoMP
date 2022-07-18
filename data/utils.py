"""Utilities for data processing"""
import argparse


def save_output(outdir, data, metadata):
    """Save processed data into output folders. The filenames are hardcoded in the functions in loaders.py.

    Args:
        outdir (pathlib.Path): The output director
        data (pd.DataFrame): The dataset with shape (n_samples, n_features)
        metadata (pd.DataFrame): The metadata with data
    """
    outdir.mkdir(parents=True, exist_ok=True)
    data.to_csv(outdir / "features.tsv", sep="\t")
    metadata.to_csv(outdir / "metadata.tsv", sep="\t")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="The parent datafolder with downloaded data.",
    )
    parser.add_argument(
        "--top-var-number", type=int, default=None, help="No. of features to filter."
    )
    parser.add_argument(
        "--output-dir", type=str, default="/tmp/comp", help="The output directory."
    )
    return parser
