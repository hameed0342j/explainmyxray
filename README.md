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
3. Visualizes findings with **bounding box overlays** on the X-ray

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
├── install.bat                         # Windows setup script
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── notebooks/
│   ├── ExplainMyXray_v2.ipynb          # Main training notebook (v2)
│   └── model.ipynb                     # Legacy v1 notebook (reference)
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
└── scripts/
    └── simplify_reports.py             # Report simplification
```

---

## Quick Start

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **NVIDIA GPU** with ≥12 GB VRAM (RTX 3090 / 4080 / 5080+)
- **CUDA 12.1+** and cuDNN installed
- **HuggingFace account** with MedGemma license accepted:
  [https://huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

### Option A: Automated Install

**Linux / macOS:**
```bash
git clone <this-repo>
cd ExplainMyXray
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
git clone <this-repo>
cd ExplainMyXray
install.bat
```

This will:
- Create a Python virtual environment
- Install PyTorch with CUDA support
- Install all ML dependencies (transformers, peft, trl, bitsandbytes, etc.)
- Register a Jupyter kernel named "ExplainMyXray v2"
- Prompt you to log in to HuggingFace

### Option B: Manual Install

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate.bat       # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
huggingface-cli login
```

### Option C: Google Colab

1. Upload `notebooks/ExplainMyXray_v2.ipynb` to Colab
2. Select **GPU runtime** (T4 minimum, A100 recommended)
3. Run Cell 1 — it will install dependencies and mount Google Drive automatically
4. Run all remaining cells sequentially

---

## Dataset Setup — BIMCV PadChest

The full PadChest dataset contains **160K+ chest X-ray images** organized in numbered sub-folders:

```
/content/drive/MyDrive/Padchest/
├── PADCHEST_chest_x_ray_images_labels_160K.csv
└── images/
    ├── 0/    ← sub-folder with .png images
    ├── 1/
    ├── 2/
    │   ...
    └── 37/
```

### Google Colab (default)
The notebook is pre-configured for Colab with Google Drive:
```
CSV:    /content/drive/MyDrive/Padchest/PADCHEST_chest_x_ray_images_labels_160K.csv
Images: /content/drive/MyDrive/Padchest/images/0/ ... /37/
```
Drive is auto-mounted in Cell 1. No path changes needed.

### Windows (Google Drive for Desktop)
Install [Google Drive for Desktop](https://www.google.com/drive/download/), then update `Config` in Cell 5:
```python
cfg.gdrive_padchest_csv = "G:/My Drive/Padchest/PADCHEST_chest_x_ray_images_labels_160K.csv"
cfg.gdrive_padchest_images = "G:/My Drive/Padchest/images"
```

### Local Download
If you have the dataset downloaded locally:
```python
cfg.gdrive_padchest_csv = "/path/to/PADCHEST_chest_x_ray_images_labels_160K.csv"
cfg.gdrive_padchest_images = "/path/to/images"
```

### Quick Test with Sample Data
Set `use_full_padchest = False` in Cell 5 to use the included 24-image sample.

---

## Training

1. Open `notebooks/ExplainMyXray_v2.ipynb`
2. Run cells sequentially (Phase 1 → Phase 6)
3. Training takes ~8-12 hours on RTX 5080 for 5 epochs on full PadChest
4. Model saves LoRA adapters (~200 MB) to `./explainmyxray-v2-medgemma-padchest/`

### Key Hyperparameters (Cell 5)

| Parameter | Value | Why |
|-----------|-------|-----|
| `lora_r` | 32 | Higher rank = more capacity for 95%+ accuracy |
| `lora_alpha` | 64 | 2× rank for stronger adaptation signal |
| `num_train_epochs` | 5 | Enough passes to converge on 160K images |
| `learning_rate` | 1e-4 | Conservative for pretrained model |
| `gradient_accumulation_steps` | 32 | Effective batch = 32 for training stability |
| `use_curriculum` | True | Easy→hard ordering improves convergence |

### If Accuracy < 95%

- Increase `num_train_epochs` to 7–10
- Increase `lora_r` to 64 (uses ~1 GB more VRAM)
- Lower `learning_rate` to 5e-5
- Increase `max_seq_length` to 768
- Filter dataset to physician-labeled rows only (higher quality annotations)

---

## Inference

After training, use the interactive function in Cell 24:

```python
prediction = predict_xray("/path/to/chest_xray.png", view="PA")
```

This produces:
- Printed structured report (FINDINGS / LOCATIONS / IMPRESSION)
- Side-by-side visualization: original X-ray + anatomical overlay with labeled regions

---

## Hardware Requirements

| Environment | GPU | VRAM | Time |
|-------------|-----|------|------|
| Training (Colab) | T4 / A100 | 16+ GB | 6–12 hrs |
| Training (Local) | RTX 3090 / 4080 / 5080 | ≥12 GB | 8–12 hrs |
| Inference | Any NVIDIA GPU | ≥6 GB | ~2 sec/image |

**VRAM Breakdown (12 GB budget):**
- Base model (4-bit): ~2.5 GB
- LoRA adapters: ~0.5 GB
- Optimizer states: ~1.5 GB
- Activations (batch=1 + gradient checkpointing): ~3 GB
- **Total: ~7.5 GB** — fits comfortably in 12 GB

---

## License

MIT License — For educational and research purposes only.

**Not intended for clinical diagnostic use.** This is an AI research tool.
Always consult a qualified radiologist for medical image interpretation.
