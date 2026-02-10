# ExplainMyXray v2 — AI Assistant Prompt

> **Copy-paste this entire prompt to Claude (or any LLM) in your IDE (Windsurf, Cursor, VS Code Copilot, etc.)**
> It gives the AI full context about the project so it can help you set up, debug, and run everything.

---

## PROMPT START — COPY EVERYTHING BELOW THIS LINE

---

You are helping me set up and run a medical AI project called **ExplainMyXray v2**. Read the README.md file in this workspace first, then follow these instructions.

---

### ⛔ CRITICAL RULES — READ BEFORE DOING ANYTHING

**Your ONLY job is to help me run the notebook cell by cell and fix errors. Nothing else.**

#### DO NOT:
1. **DO NOT hallucinate** — never invent file paths, package versions, error messages, or outputs. If you don't know something, say "I don't know" or ask me to check.
2. **DO NOT modify notebook code** unless I explicitly ask you to, or unless fixing a clear bug that prevents execution. The notebook is already written and tested.
3. **DO NOT rewrite, refactor, or "improve" any cell** — run it as-is first. Only suggest changes if it errors out.
4. **DO NOT skip cells or run cells out of order** — run them sequentially: Cell 1, then Cell 2, then Cell 3, etc. Every cell depends on the ones before it.
5. **DO NOT install packages that aren't in the notebook or requirements.txt** — if a package is missing, tell me the exact error and I'll decide what to do.
6. **DO NOT make up solutions** — if a cell errors, show me the exact error traceback and suggest a fix based on the error message (not based on guessing).
7. **DO NOT run multiple cells at once** — one cell at a time, wait for output, confirm it worked, then move to the next.
8. **DO NOT create new files, scripts, or notebooks** — everything you need is already in this project.
9. **DO NOT change model architecture, LoRA config, training hyperparameters, or quantization settings** — these are carefully tuned for 12 GB VRAM.
10. **DO NOT download the dataset locally** unless I explicitly ask. The dataset streams from Google Drive for Desktop (zero local download).
11. **DO NOT suggest using Google Colab** — I am running locally on my Windows laptop with VS Code.
12. **DO NOT add unnecessary complexity** — keep every fix as simple as possible. One-line fixes over multi-file refactors.

#### DO:
1. **DO read README.md first** — it has all the setup instructions.
2. **DO run one cell at a time** and wait for the output before proceeding.
3. **DO report the exact output of each cell** — copy-paste what you see, don't summarize.
4. **DO report errors exactly as they appear** — full traceback, no truncation.
5. **DO ask me to verify things** you can't check yourself (e.g., "Is Google Drive for Desktop running?").
6. **DO keep fixes simple** — the simplest fix that works is the best fix.
7. **DO tell me when a cell succeeds** before moving to the next one.
8. **DO follow the notebook's comments** — each cell has comments explaining what it does.

---

### Project Summary

ExplainMyXray v2 fine-tunes **MedGemma-4B-it** (Google's medical vision-language model) on the **BIMCV PadChest dataset** (160K+ chest X-ray images) to generate structured radiology reports with disease localization. The model identifies 174 radiographic findings and 104 anatomical locations, then visualizes them as color-coded bounding box overlays on the X-ray.

### My Setup

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 4080 Laptop (12 GB VRAM)
- **IDE**: VS Code with Python + Jupyter extensions
- **Dataset**: BIMCV PadChest (~160K images, ~1TB) stored in Google Drive, **streamed via Google Drive for Desktop** — NOT downloaded locally
- **Python**: 3.11 in a virtual environment (venv)
- **Storage**: Limited local disk — dataset streams from Google Drive on-demand

---

### Step-by-Step Workflow (Follow This Exactly)

#### Phase 1: Setup (one-time)
1. **Read README.md** — understand the project structure and prerequisites.
2. **Run `install.bat`** — installs venv, PyTorch, all dependencies, Jupyter kernel. If it fails, give me the exact error and the manual command to fix it. Do NOT try to write your own install script.
3. **Verify GPU works** — run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. Must show `True` and `RTX 4080 Laptop`. If not, tell me the exact output.

#### Phase 2: Run the Notebook (cell by cell)
4. **Open `notebooks/ExplainMyXray_v2.ipynb`** in VS Code.
5. **Select kernel**: "ExplainMyXray v2" or the venv Python (`.\venv\Scripts\python.exe`).
6. **Run Cell 1** (installs + Drive detection). Wait for output. It should print:
   - `✅ All dependencies installed.`
   - `✅ Google Drive for Desktop detected!` (if Drive is set up)
   - If Drive is NOT detected, tell me — I'll check if Drive for Desktop is running.
7. **Run Cell 2** (imports). Should print `✅ All imports successful`.
8. **Run Cell 3** (HuggingFace auth). Should find a cached token. If not, it asks me to login.
9. **Run Cell 4** (GPU config). Should print compute capability and free VRAM.
10. **Run Cell 5** (Config). Should print detected dataset paths and configuration.
    - If paths say "not found" → ask me to verify Google Drive is running.
    - **DO NOT change any hyperparameters** — they are already set.
11. **Run Cells 6-10** (data loading). This loads and preprocesses the CSV. May take 5-10 min.
12. **Run Cells 11-12** (dataset splits). Should print train/val/test sizes.
13. **Run Cells 13-14** (model loading). Downloads MedGemma (~2.5 GB first time). May take 5 min.
14. **Run Cells 15-16** (trainer + START TRAINING). **Training takes ~8-12 hours.** Leave it running.
15. **Run Cells 17-22** (evaluation). After training completes.
16. **Run Cells 23-24** (save + inference). Test with `predict_xray()`.

#### If A Cell Errors:
1. **Show me the full error traceback** — do not truncate or summarize.
2. **Tell me which cell number failed** and what the cell was trying to do.
3. **Suggest ONE simple fix** based on the error message. Do not suggest multiple alternatives.
4. **Wait for me to confirm** before applying any fix.
5. **After fixing, re-run that same cell** — do not skip ahead.

---

### Common Issues & Simple Fixes

| Error | Fix |
|-------|-----|
| `bitsandbytes` error on Windows | `pip install bitsandbytes --prefer-binary` |
| CUDA not found | Check `nvidia-smi`. Reinstall CUDA Toolkit 12.4 |
| HuggingFace access denied | Accept license at https://huggingface.co/google/medgemma-4b-it then `huggingface-cli login` |
| Out of memory (OOM) | Close other apps. Or reduce `max_seq_length` to 256 in Cell 5 |
| `ModuleNotFoundError` | Activate venv: `venv\Scripts\activate.bat` then `pip install <package>` |
| Google Drive path not found | Check Drive for Desktop is running + signed in. Check drive letter in "This PC" |
| First epoch very slow | Normal with Drive streaming. Files cache after first access |
| Drive disconnects | Check internet. Re-sign in to Google Drive for Desktop |

---

### Technical Architecture (Reference Only — DO NOT Modify)

- **Model**: `google/medgemma-4b-it` — loaded with `AutoModelForImageTextToText` + `AutoProcessor`
- **Quantization**: 4-bit NF4 via BitsAndBytes (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, compute dtype `bfloat16`)
- **Fine-tuning**: QLoRA via PEFT — `LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", modules_to_save=["lm_head", "embed_tokens"])`
- **Trainer**: TRL `SFTTrainer` with `SFTConfig` — NOT vanilla HuggingFace Trainer
- **Data format**: Chat conversation: System → User (with image) → Assistant (structured report)
- **Chat template**: `processor.apply_chat_template(messages, tokenize=False)` — do NOT manually format tokens
- **Precision**: BFloat16 (compute capability ≥ 8.0)
- **Gradient checkpointing**: Enabled to fit in 12 GB VRAM
- **Effective batch size**: 1 × 32 gradient accumulation = 32
- **Early stopping**: Patience=5, monitors eval_loss

**DO NOT change any of the above.** These settings are the result of extensive testing and are tuned for this exact GPU.

### Key Files

| File | Purpose | Modify? |
|------|---------|---------|
| `README.md` | Complete setup guide — READ THIS FIRST | NO |
| `notebooks/ExplainMyXray_v2.ipynb` | **Main notebook — run this to train** | NO (just run it) |
| `install.bat` | Windows automated setup script | NO (just run it) |
| `install.sh` | Linux/macOS setup script | NO |
| `.env.example` | Template for env variables (copy to `.env`) | Copy only |
| `requirements.txt` | Python package list | NO |
| `CLAUDE_PROMPT.md` | This prompt file | NO |

### Google Drive for Desktop — Dataset Streaming

The PadChest dataset (~1TB) streams via **Google Drive for Desktop**. Zero local download needed.

- Google Drive for Desktop creates a virtual drive (e.g., `G:` on Windows)
- Files appear local but are fetched from cloud on-demand
- Notebook Cell 1 auto-detects the mount point
- First epoch may be ~2-3x slower (files streaming), subsequent epochs are faster (cached)

**Setup**: Install from https://www.google.com/drive/download/ → sign in → verify `G:\My Drive\Padchest\` exists.

**If auto-detection fails**: Set paths manually in Cell 5 (check your drive letter in "This PC"):
```python
cfg.gdrive_padchest_csv = "G:/My Drive/Padchest/PADCHEST_chest_x_ray_images_labels_160K.csv"
cfg.gdrive_padchest_images = "G:/My Drive/Padchest/images"
```

**Fallback (last resort)**: Download a subset locally (max 300 GB). Set paths in Cell 5 to local folder.

### Environment Variables

- **Option A**: Copy `.env.example` to `.env`, add your HuggingFace token
- **Option B**: Run `huggingface-cli login` once (saves globally)
- **NEVER hardcode tokens in notebook cells**

### Expected Training Behavior

- First epoch: ~2 hours on RTX 4080 Laptop with 160K images
- Loss starts at ~2-4, decreases to <0.5 by epoch 3
- Progress bar shows step count, loss, and learning rate
- Early stopping may trigger before all 5 epochs
- Total: ~8-12 hours for 5 epochs
- Output: LoRA adapter saved to `./explainmyxray-v2-medgemma-padchest/` (~200 MB)

### After Training

```python
# Cell 24:
prediction = predict_xray("C:/path/to/any/chest_xray.png", view="PA")
```

Prints a structured report (FINDINGS / LOCATIONS / IMPRESSION) and shows annotated X-ray visualization.

---

### ⛔ FINAL REMINDER

- **Run cells one at a time. Report exact outputs. Don't hallucinate. Don't modify code. Keep it simple.**
- If something fails, show the error and suggest ONE fix. Don't rewrite the project.
- You are a setup assistant, not a developer. Your job is to run what's already written.

---

## PROMPT END

---

> **Note**: This prompt gives the AI strict guardrails to prevent hallucination and over-engineering. The AI should run the notebook exactly as written, report errors honestly, and suggest minimal fixes. If you hit any error, paste the full traceback and the AI will suggest a targeted fix.
