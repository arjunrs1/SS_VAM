# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
setup(
    name="ss-vam",
    version="0.1.0",
    description="Self-Supervised Visual Acoustic Matching (NeurIPS 2023)",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/arjunrs1/SS_VAM",
    packages=find_packages(include=["novas", "novas.*", "as_scripts", "as_scripts.*"]),
    py_modules=["trainer"],
    python_requires=">=3.7,<3.11",
    install_requires=[line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
                      if line.strip() and not line.startswith("#")],
)
