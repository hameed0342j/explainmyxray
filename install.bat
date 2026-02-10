@echo off
REM =============================================================
REM ExplainMyXray v2 — Installation Script (Windows)
REM =============================================================
REM Requirements:
REM   - Python 3.10+ (3.11 recommended)
REM   - NVIDIA GPU with >=12 GB VRAM (RTX 3090/4080 Laptop/4080+)
REM   - CUDA 12.1+ installed
REM   - ~10 GB disk for model + dependencies
REM =============================================================

echo ============================================
echo  ExplainMyXray v2 — Setup (Windows)
echo ============================================

REM ---- Check Python ----
python --version 2>NUL
if errorlevel 1 (
    echo ERROR: Python not found in PATH. Install Python 3.10+
    pause
    exit /b 1
)

REM ---- Check GPU ----
echo.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>NUL
if errorlevel 1 (
    echo WARNING: nvidia-smi not found. Make sure CUDA is installed.
)

REM ---- Create virtual environment ----
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo Activated venv

REM ---- Upgrade pip ----
pip install --upgrade pip setuptools wheel

REM ---- Install PyTorch with CUDA ----
echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

REM ---- Install ML dependencies ----
echo.
echo Installing ML dependencies...
pip install ^
    "transformers>=4.52.0" ^
    "trl>=0.17.0" ^
    "peft>=0.15.0" ^
    "accelerate>=1.5.0" ^
    "bitsandbytes>=0.45.0" ^
    "datasets>=3.5.0" ^
    evaluate ^
    tensorboard ^
    scikit-learn ^
    "Pillow>=10.0" ^
    matplotlib ^
    pandas ^
    gdown ^
    huggingface_hub ^
    jupyter ^
    ipykernel

REM ---- Register Jupyter kernel ----
echo.
echo Registering Jupyter kernel...
python -m ipykernel install --user --name explainmyxray --display-name "ExplainMyXray v2"

REM ---- HuggingFace Login ----
echo.
echo ============================================
echo   HuggingFace Authentication
echo ============================================
echo MedGemma requires accepting the license at:
echo   https://huggingface.co/google/medgemma-4b-it
echo.
echo Login with your HuggingFace token:
huggingface-cli login

REM ---- Verify installation ----
echo.
echo ============================================
echo   Verifying installation...
echo ============================================
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU'); import transformers; print(f'Transformers: {transformers.__version__}'); import peft; print(f'PEFT: {peft.__version__}'); import trl; print(f'TRL: {trl.__version__}'); print('All dependencies verified!')"

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Activate venv:  venv\Scripts\activate.bat
echo   2. Open notebook:  jupyter notebook notebooks\ExplainMyXray_v2.ipynb
echo   3. Select kernel:  'ExplainMyXray v2'
echo   4. Run all cells sequentially
echo.
echo For PadChest dataset via Google Drive for Desktop:
echo   - Install Google Drive for Desktop
echo   - Dataset will appear at G:\My Drive\Padchest\
echo   - Update Config paths in Cell 5 of the notebook:
echo     gdrive_padchest_csv = "G:/My Drive/Padchest/PADCHEST_chest_x_ray_images_labels_160K.csv"
echo     gdrive_padchest_images = "G:/My Drive/Padchest/images"
echo.
pause
