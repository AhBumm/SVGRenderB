#!/bin/bash
# Installation script for PyTorch-SVGRender
# Target: Python 3.10 + PyTorch 2.8 + CUDA 12.8
#
# Usage:
#   bash script/install.sh              # base install (no diffvg)
#   INSTALL_DIFFVG=1 bash script/install.sh  # also compile & install diffvg
#   INSTALL_XFORMERS=1 bash script/install.sh  # also install xformers

set -e

# ---------------------------------------------------------------------------
# 1. Create conda environment (Python 3.10)
# ---------------------------------------------------------------------------
eval "$(conda shell.bash hook)"

conda create --name svgrender python=3.10 --yes
conda activate svgrender
echo "Conda environment 'svgrender' created with Python 3.10."

# ---------------------------------------------------------------------------
# 2. Install PyTorch 2.8 + CUDA 12.8 via official pip wheel
#    See: https://pytorch.org/get-started/locally/
# ---------------------------------------------------------------------------
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo "PyTorch (torch 2.8, cu128) installation complete."

# ---------------------------------------------------------------------------
# 3. (Optional) xformers – recommended for Stable Diffusion pipelines.
#    Install only when INSTALL_XFORMERS=1 to avoid hard-failure on machines
#    where the compatible wheel is unavailable.
# ---------------------------------------------------------------------------
if [ "${INSTALL_XFORMERS:-0}" = "1" ]; then
    pip install xformers || echo "Warning: xformers install failed – continuing without it."
    echo "xformers installation attempt done."
fi

# ---------------------------------------------------------------------------
# 4. Base Python dependencies
# ---------------------------------------------------------------------------
pip install hydra-core omegaconf
pip install freetype-py shapely svgutils cairosvg
pip install "opencv-python>=4.8" scikit-image matplotlib wandb beautifulsoup4
pip install "numpy>=1.24,<2" scipy scikit-fmm einops timm
pip install "accelerate>=0.28" "transformers>=4.38" safetensors datasets
pip install easydict scikit-learn "pytorch_lightning>=2.1" webdataset
pip install ftfy regex tqdm
pip install svgwrite svgpathtools cssutils torch-tools
echo "Base Python dependencies installed."

# ---------------------------------------------------------------------------
# 5. Generative-model pipeline dependencies (optional but recommended)
# ---------------------------------------------------------------------------
pip install "diffusers>=0.28"
pip install git+https://github.com/openai/CLIP.git || echo "Warning: CLIP install failed."
echo "Generative pipeline dependencies installed."

# ---------------------------------------------------------------------------
# 6. (Optional) diffvg – requires CUDA toolkit headers and cmake.
#    Set INSTALL_DIFFVG=1 to enable. Common failure cause: CUDA header mismatch.
#    If compilation fails see: https://github.com/BachiLi/diffvg#installation
# ---------------------------------------------------------------------------
if [ "${INSTALL_DIFFVG:-0}" = "1" ]; then
    echo "Installing DiffVG (this may take several minutes)..."

    # System-level build tools (Ubuntu/Debian)
    sudo apt-get update -qq
    sudo apt-get install -y cmake ffmpeg build-essential libjpeg-dev libpng-dev libtiff-dev

    if [ ! -d "diffvg" ]; then
        git clone https://github.com/BachiLi/diffvg.git
    fi
    cd diffvg
    git submodule update --init --recursive
    pip install .
    cd ..
    echo "DiffVG installation complete."
else
    echo "Skipping DiffVG (set INSTALL_DIFFVG=1 to install)."
    echo "After installing DiffVG manually, the 'diffvg' and 'diffvg_tile' methods will be available."
fi

# ---------------------------------------------------------------------------
# Final confirmation
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " PyTorch-SVGRender environment installed successfully!"
echo " To also install diffvg:  INSTALL_DIFFVG=1 bash script/install.sh"
echo " To also install xformers: INSTALL_XFORMERS=1 bash script/install.sh"
echo "========================================================"