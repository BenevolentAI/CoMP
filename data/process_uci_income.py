"""Prepare UCI income dataset"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing

from data.utils import create_arg_parser, save_output

logging.basicConfig(
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
FILENAMES = dict(
    train_file="adult.data",
    test_file="adult.test",
)


def load_uci(input_dir):
    """Load the UCI income source data.

    Args:
        input_dir (pathlib.Path): The input folder with files 'adult.data' and 'adult.test' saved
            from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

    Returns:
        train, test (pd.DataFrame): The train and test UCI income data
    """
    train_file = input_dir / FILENAMES["train_file"]
    test_file = input_dir / FILENAMES["test_file"]
    features = [
        "age",
        "workclass",
        "final_weight",
        "education",
        "education_label",
        "martial_status",
        "job",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "country",
        "target",
    ]
    train = pd.read_csv(
        train_file, names=features, sep=r"\s*,\s*", engine="python", na_values="?"
    )
    test = pd.read_csv(
        test_file,
        names=features,
        sep=r"\s*,\s*",
        engine="python",
        na_values="?",
        skiprows=1,
    )
    return train, test


def transform_uci_features(df):
    binary_data = pd.get_dummies(df)
    feature_cols = binary_data[binary_data.columns[:-2]]
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(
        scaler.fit_transform(feature_cols), columns=feature_cols.columns
    )
    return data


def process_uci(train_df, test_df, split_train_test=False):
    train_and_test = pd.concat([train_df, test_df]).reset_index(drop=True)
    train_and_test.relationship = np.where(
        train_and_test.relationship == "Wife", "spouse", train_and_test.relationship
    )
    train_and_test.relationship = np.where(
        train_and_test.relationship == "Husband", "spouse", train_and_test.relationship
    )
    train_and_test.dropna(inplace=True)
    test_dim = test_df.shape[0]
    labels = train_and_test["target"]
    labels = labels.replace("<=50K", 0).replace(">50K", 1)
    labels = labels.replace("<=50K.", 0).replace(">50K.", 1)
    if split_train_test:
        train_labels = labels[test_dim:]
        test_labels = labels[:test_dim]
        cens_attrib_train = train_and_test.sex[test_dim:]
        cens_attrib_test = train_and_test.sex[:test_dim]
    else:
        cens_attrib = train_and_test.sex

    train_and_test.drop(["education", "target"], axis=1, inplace=True)
    if split_train_test:
        train = train_and_test[test_dim:]
        test = train_and_test[:test_dim]
        train = transform_uci_features(train)
        train.drop(
            ["sex_Male", "sex_Female", "country_Holand-Netherlands"],
            axis=1,
            inplace=True,
        )
        test = transform_uci_features(test)
        test.drop(["sex_Male", "sex_Female"], axis=1, inplace=True)
        return (
            (train, test),
            (train_labels, test_labels),
            (cens_attrib_train, cens_attrib_test),
        )
    else:
        train = transform_uci_features(train_and_test)
        train.drop(
            ["sex_Male", "sex_Female", "country_Holand-Netherlands"],
            axis=1,
            inplace=True,
        )
        return train, labels, cens_attrib


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    LOGGER.info(f"Processing UCI Income dataset")
    train_df, test_df = load_uci(input_dir=input_dir)
    data_uci, income_target, gender = process_uci(train_df, test_df)
    metadata_uci = pd.concat([income_target, gender], axis=1)
    metadata_uci.columns = ["income", "type"]
    LOGGER.info(f"data_uci.shape {data_uci.shape}")
    LOGGER.info(f"metadata_uci.shape {metadata_uci.shape}")

    LOGGER.info(f"Saving UCI Income dataset in {output_dir}")
    save_output(output_dir, data_uci, metadata_uci)


if __name__ == "__main__":
    main()
