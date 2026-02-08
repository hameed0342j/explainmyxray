# ExplainMyXray v2 🩻

> AI-powered chest X-ray analysis with **disease localization** — built on MedGemma-4B and the BIMCV PadChest dataset (160K+ images, 174 findings, 104 anatomical locations).

**Target: ≥95% diagnostic accuracy** on multi-label chest X-ray classification with anatomical region overlays.

---

## What It Does

1. Takes a chest X-ray image as input
2. Produces a **structured radiology report**:
   - **FINDINGS**: List of detected conditions (e.g., cardiomegaly, pneumonia, pleural effusion)
   - **LOCATIONS**: Anatomical regions for each finding (e.g., right lower lobe, cardiac)
   - **IMPRESSION**: Clinical summary
3. Visualizes findings with **color-coded bounding box overlays** on the X-ray

---

## Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) — Gemma 3 decoder + medical SigLIP encoder |
| **Fine-tuning** | QLoRA 4-bit NF4 via TRL `SFTTrainer` |
| **LoRA Config** | r=32, α=64, all-linear targets + lm_head/embed_tokens |
| **Precision** | bfloat16 (requires compute capability ≥ 8.0) |
| **Dataset** | BIMCV PadChest — 160K+ CXR images, 174 findings, 104 locations |
| **Training** | Curriculum learning (easy→hard), cosine LR, early stopping |
| **Evaluation** | Per-finding precision/recall/F1, exact + soft match accuracy |

---

## Project Structure

```
ExplainMyXray/
├── install.sh                          # Linux/macOS setup script
├── install.bat                         # Windows setup script (USE THIS)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── architecture_prompt.json            # Napkin AI architecture visualization prompt
├── notebooks/
│   ├── ExplainMyXray_v2.ipynb          # ⭐ MAIN TRAINING NOTEBOOK (run this)
│   ├── model.ipynb                     # Legacy v1 notebook (reference only)
│   ├── test_model.ipynb                # Testing notebook
│   └── train_medgemma.md               # Training documentation
├── src/
│   ├── data/
│   │   ├── dataset.py                  # Dataset loader
│   │   └── collator.py                 # Data collator
│   ├── model/
│   │   └── config.py                   # Model configuration
│   └── inference/
│       └── predictor.py                # Inference pipeline
├── app/
│   ├── api.py                          # FastAPI backend
│   └── frontend.py                     # Streamlit UI
├── scripts/
│   └── simplify_reports.py             # Report simplification
└── ai_error_analysis/
    ├── technical_decisions_guide.md     # Design decisions documentation
    └── training_errors_log.md          # Error tracking log
```

---

## ⚠️ Security — API Keys & Tokens

**NEVER hardcode tokens in code files.** This project uses environment variables for all secrets.

### How to Set Up Your Token

1. Copy the example env file:
   ```cmd
   copy .env.example .env
   ```
2. Open `.env` in a text editor and paste your HuggingFace token:
   ```
   HF_TOKEN=hf_your_actual_token_here
   ```
3. The `.env` file is in `.gitignore` — it will NEVER be pushed to GitHub

**Alternative (simpler):** Just run this once in your terminal and it saves the token globally:
```cmd
huggingface-cli login
```
Then paste your token when prompted. You won't need a `.env` file at all.

---

## Windows RTX 5080 Laptop — Complete Setup Guide (Step by Step)

> **This section is written for you if you have a Windows laptop with an NVIDIA RTX 5080 GPU.**
> Follow every step in order. Copy-paste the commands exactly.

### Step 0: Install Prerequisites (One-Time)

You need these 4 things installed before starting. If you already have them, skip to Step 1.

#### 0a. Install Python 3.11

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11.x** (click the big yellow button)
3. Run the installer
4. **⚠️ IMPORTANT: Check the box "Add Python to PATH" at the bottom of the installer**
5. Click "Install Now"
6. Verify: Open **Command Prompt** (search `cmd` in Start menu) and type:
   ```cmd
   python --version
   ```
   You should see `Python 3.11.x`

#### 0b. Install Git

1. Go to https://git-scm.com/download/win
2. Download and install with default settings
3. Verify in Command Prompt:
   ```cmd
   git --version
   ```

#### 0c. Install NVIDIA CUDA Toolkit 12.4

1. Go to https://developer.nvidia.com/cuda-12-4-0-download-archive
2. Select: Windows → x86_64 → 11 (or 10) → exe (local)
3. Download and install (takes ~5 minutes)
4. Verify in Command Prompt:
   ```cmd
   nvidia-smi
   ```
   You should see your RTX 5080 with ~12 GB listed

#### 0d. Install VS Code

1. Go to https://code.visualstudio.com/
2. Download and install
3. Open VS Code → go to **Extensions** (Ctrl+Shift+X) → search and install:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)

#### 0e. Install Google Drive for Desktop (for the dataset)

1. Go to https://www.google.com/drive/download/
2. Install and sign in with the Google account that has the PadChest dataset
3. After install, your Google Drive files appear at **`G:\My Drive\`**
4. Verify the dataset folder exists at: `G:\My Drive\Padchest\`

#### 0f. Get a HuggingFace Account + Accept MedGemma License

1. Create a free account at https://huggingface.co/join
2. Go to https://huggingface.co/google/medgemma-4b-it
3. Click **"Agree and access repository"** (you must accept the license to download the model)
4. Go to https://huggingface.co/settings/tokens
5. Click **"New token"** → name it anything → select **"Read"** → click **"Generate"**
6. **Copy this token** — you'll need it during setup

---

### Step 1: Clone the Repository

Open **Command Prompt** (search `cmd` in Start menu) and run:

```cmd
cd %USERPROFILE%\Desktop
git clone https://github.com/hameed0342j/explainmyxray.git
cd explainmyxray
```

This downloads the project to your Desktop in a folder called `explainmyxray`.

---

### Step 2: Run the Install Script

Still in Command Prompt, inside the `explainmyxray` folder:

```cmd
install.bat
```

**What this does (automatically):**
- Creates a Python virtual environment (`venv` folder)
- Installs PyTorch with CUDA 12.4 GPU support
- Installs all AI/ML libraries (transformers, peft, trl, bitsandbytes, etc.)
- Registers a Jupyter kernel called "ExplainMyXray v2"
- Asks you to paste your **HuggingFace token** (from Step 0f)

This takes **10-20 minutes** depending on internet speed. Wait for it to say `Setup complete!`.

**If you see errors**, try the manual install instead (see "Manual Install" section below).

---

### Step 3: Open the Notebook in VS Code

1. Open **VS Code**
2. Click **File → Open Folder** → navigate to `Desktop\explainmyxray` → click **Select Folder**
3. In the Explorer panel (left side), open: `notebooks` → **`ExplainMyXray_v2.ipynb`**
4. VS Code will ask you to select a kernel. Click **"Select Kernel"** in the top-right → choose:
   - **ExplainMyXray v2** (this is the venv we created)
   - If you don't see it: click "Python Environments" → select `.\venv\Scripts\python.exe`

---

### Step 4: Configure Dataset Paths (Cell 5)

Before running anything, update the dataset paths in **Cell 5** of the notebook.

Find these lines and change them to your Google Drive paths:

```python
# Change these two lines:
cfg.gdrive_padchest_csv = "G:/My Drive/Padchest/PADCHEST_chest_x_ray_images_labels_160K.csv"
cfg.gdrive_padchest_images = "G:/My Drive/Padchest/images"
```

> **Note:** Use forward slashes `/` not backslashes `\` in the paths, even on Windows.
>
> If your Google Drive letter is different (e.g., `H:` instead of `G:`), check in File Explorer what drive letter Google Drive uses. You can also look at `This PC` to find "Google Drive".

Also make sure this is set:
```python
cfg.use_full_padchest = True     # Use the full 160K dataset
```

---

### Step 5: Run the Notebook (Training)

Run the cells **in order, one at a time**, from top to bottom:

| Cell Range | What It Does | Time |
|------------|-------------|------|
| Cells 1-4 | Install dependencies, imports, GPU check | ~1 min |
| Cell 5 | Configuration (paths, hyperparameters) | instant |
| Cells 6-10 | Load & preprocess PadChest dataset | ~5-10 min |
| Cells 11-12 | Create train/val/test splits | ~1 min |
| Cells 13-14 | Load MedGemma-4B model (downloads ~2.5 GB first time) | ~5 min |
| Cells 15-16 | Configure trainer and START TRAINING | **~8-12 hours** |
| Cells 17-22 | Evaluate model accuracy | ~10 min |
| Cells 23-24 | Save model + run inference on test X-rays | ~5 min |

**⚠️ Training will take 8-12 hours.** You can leave it running overnight. Don't close VS Code.

**How to know it's working**: You will see a progress bar with loss values decreasing over time.

**How to know it's done**: The progress bar reaches 100% and prints training metrics.

---

### Step 6: Run Inference (After Training)

Once training is complete, use the prediction function in **Cell 24**:

```python
# Analyze any chest X-ray:
prediction = predict_xray("path/to/any_chest_xray.png", view="PA")
```

Replace the path with any X-ray image on your computer. For example:
```python
prediction = predict_xray("C:/Users/YourName/Desktop/test_xray.png", view="PA")
```

This will:
- Print a structured radiology report (FINDINGS / LOCATIONS / IMPRESSION)
- Show a side-by-side image: original X-ray on the left, annotated X-ray with colored boxes on the right

---

## Quick Test (Without Full Dataset)

If you just want to test that everything works before running the full 160K training:

1. Open Cell 5 in the notebook
2. Set `use_full_padchest = False`
3. Run all cells — it will train on the included 24 sample images (takes ~5 minutes)
4. This won't give good accuracy, but it verifies your setup works

---

## Manual Install (If install.bat Fails)

Open Command Prompt and run these one at a time:

```cmd
cd %USERPROFILE%\Desktop\explainmyxray

python -m venv venv

venv\Scripts\activate.bat

pip install --upgrade pip setuptools wheel

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install "transformers>=4.52.0" "trl>=0.17.0" "peft>=0.15.0" "accelerate>=1.5.0" "bitsandbytes>=0.45.0" "datasets>=3.5.0" evaluate tensorboard scikit-learn "Pillow>=10.0" matplotlib pandas gdown huggingface_hub jupyter ipykernel

python -m ipykernel install --user --name explainmyxray --display-name "ExplainMyXray v2"

huggingface-cli login
```

When asked for the HuggingFace token, paste the one from Step 0f.

---

## Verify GPU is Working

After install, run this in Command Prompt (with venv activated) to check your GPU:

```cmd
venv\Scripts\activate.bat
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5080
VRAM: 12.0 GB
```

If CUDA shows `False`, your CUDA toolkit isn't installed correctly. Reinstall from Step 0c.

---

## Dataset Setup — BIMCV PadChest

The full PadChest dataset contains **160K+ chest X-ray images** organized in 38 numbered sub-folders.

### Folder Structure

```
Padchest/
├── PADCHEST_chest_x_ray_images_labels_160K.csv    ← labels file
└── images/
    ├── 0/    ← sub-folder with .png X-ray images
    ├── 1/
    ├── 2/
    │   ...
    └── 37/   ← 38 folders total
```

### Where to Put the Dataset

| Setup | CSV Path | Images Path |
|-------|----------|-------------|
| **Windows (Drive for Desktop)** | `G:/My Drive/Padchest/PADCHEST_chest_x_ray_images_labels_160K.csv` | `G:/My Drive/Padchest/images` |
| **Google Colab** (default) | `/content/drive/MyDrive/Padchest/PADCHEST_chest_x_ray_images_labels_160K.csv` | `/content/drive/MyDrive/Padchest/images` |
| **Local download** | Your local path to the CSV | Your local path to the `images/` folder |

Update these in **Cell 5** → `cfg.gdrive_padchest_csv` and `cfg.gdrive_padchest_images`.

---

## Google Colab Setup (Alternative)

If you want to use Google Colab instead of running locally:

1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook** → select `notebooks/ExplainMyXray_v2.ipynb`
3. Click **Runtime → Change runtime type** → select **T4 GPU** (free) or **A100** (Colab Pro)
4. Run Cell 1 — it installs everything and mounts Google Drive automatically
5. Make sure your PadChest dataset is in Google Drive at `My Drive/Padchest/`
6. Run all remaining cells in order

---

## Training Details

### Key Hyperparameters (Cell 5)

| Parameter | Value | What It Does |
|-----------|-------|--------------|
| `lora_r` | 32 | LoRA rank — higher = more learning capacity |
| `lora_alpha` | 64 | Scaling factor (2× rank for strong adaptation) |
| `num_train_epochs` | 5 | Number of passes through all 160K images |
| `learning_rate` | 1e-4 | How fast the model learns (lower = more careful) |
| `gradient_accumulation_steps` | 32 | Simulates batch size 32 even though we fit 1 image at a time |
| `use_curriculum` | True | Trains on easy cases first, hard cases later |

### Training Output

The trained model saves to `./explainmyxray-v2-medgemma-padchest/` — this is a ~200 MB LoRA adapter folder. The full 4B model is NOT saved (only the fine-tuned weights on top of it).

### If Accuracy is Below 95%

Try these changes in Cell 5 (one at a time, retrain each time):

| Change | How | Why |
|--------|-----|-----|
| More epochs | `num_train_epochs = 7` or `10` | Give the model more training time |
| Higher LoRA rank | `lora_r = 64`, `lora_alpha = 128` | More learning capacity (uses ~1 GB more VRAM) |
| Lower learning rate | `learning_rate = 5e-5` | Learn more carefully |
| Longer sequences | `max_seq_length = 768` | Allow longer report generation |
| Higher quality labels | Filter CSV to `Labeling == "Physician"` | Use only doctor-verified annotations |

---

## Troubleshooting

### Common Issues on Windows

| Problem | Solution |
|---------|----------|
| `python` not found | Reinstall Python and check "Add to PATH". Use `py` instead of `python` |
| `nvidia-smi` not found | Install CUDA Toolkit from Step 0c. Restart your computer |
| CUDA shows False in PyTorch | Make sure CUDA 12.4 is installed. Run `nvidia-smi` — the CUDA version shown must be ≥12.4 |
| `bitsandbytes` error on Windows | Run: `pip install bitsandbytes --prefer-binary` |
| Out of memory (OOM) error | Close Chrome and other apps. Or reduce `max_seq_length` to 256 in Cell 5 |
| `ModuleNotFoundError` | Make sure you activated the venv: `venv\Scripts\activate.bat` |
| Notebook kernel not found | In VS Code, click "Select Kernel" → "Python Environments" → pick `.\venv\Scripts\python.exe` |
| HuggingFace access denied | Accept MedGemma license at https://huggingface.co/google/medgemma-4b-it then re-run `huggingface-cli login` |
| Google Drive path not found | Check your Drive letter in File Explorer under "This PC". It might be `H:\` instead of `G:\` |
| Training seems stuck / no progress | It's not stuck — first epoch on 160K images takes ~2 hours. Check the progress bar percentage |
| VS Code says "Jupyter not installed" | Open terminal in VS Code (Ctrl+`) → run `venv\Scripts\activate.bat` then `pip install jupyter ipykernel` |

### How to Activate the Virtual Environment (Every Time)

Every time you open a new Command Prompt or VS Code terminal, you need to activate the environment:

```cmd
cd %USERPROFILE%\Desktop\explainmyxray
venv\Scripts\activate.bat
```

You'll see `(venv)` at the start of the prompt. This means it's activated.

In **VS Code**, the notebook automatically uses the right environment if you selected the kernel in Step 3.

---

## Hardware Requirements

| Environment | GPU | VRAM | Training Time |
|-------------|-----|------|---------------|
| **Windows RTX 5080 (local)** | RTX 5080 | 12 GB | ~8-12 hours |
| Google Colab (free) | T4 | 16 GB | ~10-14 hours |
| Google Colab Pro | A100 | 40 GB | ~3-5 hours |

**VRAM Usage During Training (~9 GB of 12 GB):**

| Component | VRAM |
|-----------|------|
| Base model (4-bit quantized) | ~2.5 GB |
| LoRA adapter weights | ~0.5 GB |
| Optimizer states (AdamW) | ~1.5 GB |
| Activations (batch=1 + gradient checkpointing) | ~3.0 GB |
| CUDA overhead | ~1.5 GB |
| **Total** | **~9.0 GB** |

---

## Files Explained

| File | What It Is | Do You Need to Touch It? |
|------|-----------|--------------------------|
| `notebooks/ExplainMyXray_v2.ipynb` | **The main notebook. Run this to train and use the model.** | Yes — update paths in Cell 5 |
| `install.bat` | Windows automated setup script | Just run it once |
| `install.sh` | Linux/macOS setup script | Only if on Linux/Mac |
| `requirements.txt` | List of Python packages needed | No — install.bat handles this |
| `README.md` | This file | No |
| `architecture_prompt.json` | Visual architecture diagram prompt (for Napkin AI) | No — for documentation only |
| `notebooks/model.ipynb` | Old v1 notebook (PaliGemma-3B) | No — reference only |
| `app/api.py` | FastAPI backend for serving predictions | Optional — for deployment |
| `app/frontend.py` | Streamlit web UI for drag-and-drop X-ray analysis | Optional — for deployment |
| `scripts/simplify_reports.py` | Simplify medical reports for patients | Optional |

---

## TL;DR — The 5-Minute Version

```cmd
:: 1. Clone the repo
git clone https://github.com/hameed0342j/explainmyxray.git
cd explainmyxray

:: 2. Run the installer (takes ~15 min)
install.bat

:: 3. Open VS Code
code .

:: 4. Open notebooks/ExplainMyXray_v2.ipynb
:: 5. Select kernel: "ExplainMyXray v2"
:: 6. Update dataset paths in Cell 5
:: 7. Run all cells (training takes ~8-12 hours)
:: 8. Use predict_xray() in Cell 24 to analyze any X-ray
```

---

## License

MIT License — For educational and research purposes only.

**Not intended for clinical diagnostic use.** This is an AI research tool.
Always consult a qualified radiologist for medical image interpretation.
