#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="CoMP",
    version="1.0.0",
    packages=find_packages(exclude=["contrib", "docs", "tests", "notebooks"]),
    include_package_data=True,
    description='PyTorch implementation of "Contrastive Mixture of Posteriors for CounterfactualInference, Data Integration and Fairness"',
    keywords="",
    url="",
    zip_safe=True,
)
