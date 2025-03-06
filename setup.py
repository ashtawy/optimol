#!/usr/bin/env python

import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    lines = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        lines = open_file.read().strip()
    return lines


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
    name="optimol",
    version="0.0.1",
    description="optimol",
    author="Hossam Ashtawy",
    author_email="hossam.ashtawy@ensemtx.com",
    url="https://github.com/ashtawy/optimol.git",
    install_requires=read_requirements("requirements.txt"),
    packages=find_packages(),  # Recursively find all packages in src/
    include_package_data=True,  # To include non-Python files specified in MANIFEST.in
    # use this to customize global commands available in the
    # terminal after installing the package
    entry_points={
        "console_scripts": [
            "optimol_train = optimol.train:main",
            "optimol_score = optimol.eval:main",
        ]
    },
)
