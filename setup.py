# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing

import os
from setuptools import setup, find_packages

# ---------------------------------------------------------------------------
# Base runtime dependencies (pure-Python / pre-built wheels only).
# Heavy CUDA extensions (diffvg, xformers, triton) are in extras_require so
# that a plain  `pip install .`  succeeds even without a CUDA build toolchain.
# ---------------------------------------------------------------------------
BASE_REQUIRES = [
    # Configuration / experiment management
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    # Acceleration / distributed training
    "accelerate>=0.28",
    # Image processing
    "Pillow>=9.0",
    "opencv-python>=4.8",
    "scikit-image>=0.21",
    # SVG utilities
    "svgwrite",
    "svgpathtools",
    "cairosvg",
    "svgutils",
    "cssutils",
    "freetype-py",
    "shapely",
    # Numerical / ML utilities
    "numpy>=1.24,<2",
    "scipy",
    "scikit-fmm",
    "einops",
    "timm",
    "tqdm",
    "ftfy",
    "regex",
    "easydict",
    "scikit-learn",
    "matplotlib",
    "wandb",
    "beautifulsoup4",
    "webdataset",
    "torch-tools",
    # HuggingFace stack (version ranges compatible with torch 2.x)
    "transformers>=4.38",
    "safetensors",
    "datasets",
    "diffusers>=0.28",
    "pytorch_lightning>=2.1",
]

EXTRAS_REQUIRE = {
    # Install pydiffvg after building from source:
    #   git clone https://github.com/BachiLi/diffvg && pip install ./diffvg
    # Then install this package with:  pip install ".[diffvg]"
    "diffvg": [
        "pydiffvg",   # built & installed separately from source
    ],
    # xformers accelerates attention in SD-based pipelines.
    # Version must match the installed torch – see https://github.com/facebookresearch/xformers
    "xformers": [
        "xformers",
    ],
    # triton kernel backend (optional, Linux only)
    "triton": [
        "triton",
    ],
    "full": [
        "xformers",
        "triton",
    ],
}

setup(
    name="PyTorch-SVGRender",
    version="0.2.0",
    packages=find_packages(exclude=["lama*", "ImageReward*", "test*"]),
    install_requires=BASE_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.10",
    # Metadata
    author="Ximing Xing, Juncheng Hu et al.",
    author_email="ximingxing@gmail.com",
    description="SVG Differentiable Rendering: Generating vector graphics using neural networks.",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    keywords=["Artificial Intelligence", "AIGC", "Generative Models", "SVG Generation"],
    url="https://github.com/ximinng/PyTorch-SVGRender",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
