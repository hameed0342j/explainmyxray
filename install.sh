#!/bin/bash
# =============================================================
# ExplainMyXray v2 — Installation Script (Linux/macOS)
# =============================================================
# Requirements:
#   - Python 3.10+ (3.11 recommended)
#   - NVIDIA GPU with ≥12 GB VRAM (RTX 3090/4080 Laptop/4080+)
#   - CUDA 12.1+ and cuDNN installed
#   - ~10 GB disk for model + dependencies
# =============================================================

set -e

echo "============================================"
echo " ExplainMyXray v2 — Setup"
echo "============================================"

# ---- Check Python version ----
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required. Found $PYTHON_VERSION"
    exit 1
fi

# ---- Check NVIDIA GPU ----
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Make sure CUDA is installed."
fi

# ---- Create virtual environment ----
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Activated venv: $VENV_DIR"

# ---- Upgrade pip ----
pip install --upgrade pip setuptools wheel

# ---- Install PyTorch with CUDA ----
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ---- Install ML dependencies ----
echo ""
echo "Installing ML dependencies..."
pip install \
    "transformers>=4.52.0" \
    "trl>=0.17.0" \
    "peft>=0.15.0" \
    "accelerate>=1.5.0" \
    "bitsandbytes>=0.45.0" \
    "datasets>=3.5.0" \
    evaluate \
    tensorboard \
    scikit-learn \
    "Pillow>=10.0" \
    matplotlib \
    pandas \
    gdown \
    huggingface_hub \
    jupyter \
    ipykernel

# ---- Register Jupyter kernel ----
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name explainmyxray --display-name "ExplainMyXray v2"

# ---- HuggingFace Login ----
echo ""
echo "============================================"
echo "  HuggingFace Authentication"
echo "============================================"
echo "MedGemma requires accepting the license at:"
echo "  https://huggingface.co/google/medgemma-4b-it"
echo ""
echo "Login with your HuggingFace token:"
huggingface-cli login

# ---- Verify installation ----
echo ""
echo "============================================"
echo "  Verifying installation..."
echo "============================================"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'VRAM: {vram:.1f} GB')
    cc = torch.cuda.get_device_capability()
    print(f'Compute capability: {cc[0]}.{cc[1]}')
    if cc[0] < 8:
        print('WARNING: bfloat16 requires compute capability >= 8.0')
    if vram < 10:
        print('WARNING: ≥12 GB VRAM recommended')
import transformers; print(f'Transformers: {transformers.__version__}')
import peft; print(f'PEFT: {peft.__version__}')
import trl; print(f'TRL: {trl.__version__}')
import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')
print()
print('✅ All dependencies verified!')
"

echo ""
echo "============================================"
echo "  Checking for PadChest dataset..."
echo "============================================"

GDRIVE_FOUND=0

# Check Google Colab mount
if [ -d "/content/drive/MyDrive/Padchest" ]; then
    echo "  FOUND: PadChest dataset at /content/drive/MyDrive/Padchest/"
    GDRIVE_FOUND=1
fi

# Check macOS Google Drive for Desktop (CloudStorage)
if [ "$GDRIVE_FOUND" -eq 0 ] && [ -d "$HOME/Library/CloudStorage" ]; then
    for gdrive_dir in "$HOME/Library/CloudStorage"/GoogleDrive-*/; do
        if [ -d "${gdrive_dir}My Drive/Padchest" ]; then
            echo "  FOUND: PadChest dataset at ${gdrive_dir}My Drive/Padchest/"
            GDRIVE_FOUND=1
            break
        fi
    done
fi

# Check common Linux paths
if [ "$GDRIVE_FOUND" -eq 0 ]; then
    for candidate in \
        "$HOME/Google Drive/My Drive/Padchest" \
        "$HOME/google-drive/My Drive/Padchest" \
        "$HOME/gdrive/My Drive/Padchest" \
        "/mnt/google-drive/My Drive/Padchest" \
        "/mnt/gdrive/My Drive/Padchest"; do
        if [ -d "$candidate" ]; then
            echo "  FOUND: PadChest dataset at $candidate"
            GDRIVE_FOUND=1
            break
        fi
    done
fi

if [ "$GDRIVE_FOUND" -eq 0 ]; then
    echo "  PadChest dataset not auto-detected."
    echo ""
    echo "  RECOMMENDED: Install Google Drive for Desktop to stream the dataset"
    echo "  without downloading ~1TB locally:"
    echo "    → https://www.google.com/drive/download/"
    echo "    Sign in with the Google account that has PadChest"
    echo "    The notebook auto-detects Drive paths — no manual config needed!"
    echo ""
    echo "  ALTERNATIVE: Download PadChest locally (keep under 300 GB)"
    echo "    Set paths manually in Cell 5 of the notebook."
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate the venv:  source venv/bin/activate"
echo "  2. Open the notebook:  jupyter notebook notebooks/ExplainMyXray_v2.ipynb"
echo "  3. Select kernel:      'ExplainMyXray v2'"
echo "  4. Run all cells sequentially"
echo ""
echo "Dataset access (recommended: Google Drive for Desktop):"
echo "  - Install: https://www.google.com/drive/download/"
echo "  - Sign in with the account that has PadChest in My Drive"
echo "  - The notebook auto-detects Drive paths — no manual config needed!"
echo "  - Streams ~1TB dataset on-demand (zero local download)"
echo ""
echo "Fallback (local download, last resort):"
echo "  - Download PadChest to local disk (keep under 300 GB)"
echo "  - Update paths manually in Cell 5 of the notebook"
echo ""
