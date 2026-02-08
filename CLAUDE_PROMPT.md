# ExplainMyXray v2 — AI Assistant Prompt

> **Copy-paste this entire prompt to Claude (or any LLM) in your IDE (Windsurf, Cursor, VS Code Copilot, etc.)**
> It gives the AI full context about the project so it can help you set up, debug, and run everything.

---

## PROMPT START — COPY EVERYTHING BELOW THIS LINE

---

You are helping me set up and run a medical AI project called **ExplainMyXray v2**. Read the README.md file in this workspace first, then follow these instructions.

### Project Summary

ExplainMyXray v2 fine-tunes **MedGemma-4B-it** (Google's medical vision-language model) on the **BIMCV PadChest dataset** (160K+ chest X-ray images) to generate structured radiology reports with disease localization. The model identifies 174 radiographic findings and 104 anatomical locations, then visualizes them as color-coded bounding box overlays on the X-ray.

### My Setup

- **OS**: Windows 11
- **GPU**: NVIDIA RTX 5080 (12 GB VRAM)
- **IDE**: VS Code with Python + Jupyter extensions
- **Dataset**: BIMCV PadChest (~160K images) stored in Google Drive, accessed via Google Drive for Desktop
- **Python**: 3.11 in a virtual environment (venv)

### What I Need You To Do

1. **First, read the README.md** — it has complete setup instructions including prerequisites, install steps, dataset paths, and troubleshooting.

2. **Help me run `install.bat`** — this is the automated setup script. If it fails on any step, debug the error and give me the manual commands to fix it.

3. **Help me configure the notebook** — the main file is `notebooks/ExplainMyXray_v2.ipynb`. I need to:
   - Select the correct Jupyter kernel ("ExplainMyXray v2" or the venv Python)
   - Update dataset paths in **Cell 5** to point to my Google Drive:
     ```python
     cfg.gdrive_padchest_csv = "G:/My Drive/Padchest/PADCHEST_chest_x_ray_images_labels_160K.csv"
     cfg.gdrive_padchest_images = "G:/My Drive/Padchest/images"
     ```
   - My Google Drive letter might be different (check "This PC" in File Explorer)

4. **Help me run the notebook cell by cell** — if any cell throws an error, debug it. Common issues:
   - `bitsandbytes` errors on Windows → `pip install bitsandbytes --prefer-binary`
   - CUDA not found → check `nvidia-smi` and CUDA toolkit installation
   - HuggingFace access denied → need to accept license at https://huggingface.co/google/medgemma-4b-it and run `huggingface-cli login`
   - Out of memory → close other apps, or reduce `max_seq_length` to 256 in Cell 5
   - Module not found → activate venv first: `venv\Scripts\activate.bat`

5. **After training completes (~8-12 hours)**, help me run the evaluation cells and use `predict_xray()` for inference.

### Technical Architecture (For Debugging)

- **Model**: `google/medgemma-4b-it` — loaded with `AutoModelForImageTextToText` + `AutoProcessor`
- **Quantization**: 4-bit NF4 via BitsAndBytes (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, compute dtype `bfloat16`)
- **Fine-tuning**: QLoRA via PEFT — `LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", modules_to_save=["lm_head", "embed_tokens"])`
- **Trainer**: TRL `SFTTrainer` with `SFTConfig` — NOT vanilla HuggingFace Trainer. This handles chat template tokenization automatically
- **Data format**: Each training example is a chat conversation: System message → User message (with image) → Assistant response (structured report)
- **Chat template**: `processor.apply_chat_template(messages, tokenize=False)` — do NOT manually format tokens
- **Precision**: BFloat16 (requires GPU compute capability ≥ 8.0, which RTX 5080 has)
- **Gradient checkpointing**: Enabled to fit in 12 GB VRAM
- **Effective batch size**: 1 × 32 gradient accumulation = 32
- **Early stopping**: Patience=5, monitors eval_loss
- **Dataset**: PadChest CSV → parse labels with `ast.literal_eval` → split findings/locations by "loc " prefix → format as structured report

### Key Files

| File | Purpose |
|------|---------|
| `README.md` | Complete setup guide — READ THIS FIRST |
| `notebooks/ExplainMyXray_v2.ipynb` | **Main notebook — run this to train the model** |
| `install.bat` | Windows automated setup script |
| `install.sh` | Linux/macOS automated setup script |
| `.env.example` | Template for environment variables (copy to `.env`) |
| `requirements.txt` | Python package list |
| `architecture_prompt.json` | Visual architecture diagram (for documentation only) |

### Debugging Rules

1. **Always activate venv first** before running any Python command: `venv\Scripts\activate.bat`
2. **Check GPU**: Run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` — must show `True` and `RTX 5080`
3. **Check imports**: If `ModuleNotFoundError`, install the missing package in the venv: `pip install <package>`
4. **Check VRAM**: If OOM error, run `nvidia-smi` to see what's using GPU memory. Close other apps. Kill zombie Python processes: `taskkill /f /im python.exe`
5. **Check paths**: Windows paths use forward slashes in Python: `"G:/My Drive/..."` not `"G:\My Drive\..."`
6. **Read error messages carefully** — PyTorch and transformers errors are usually descriptive. Search the error on GitHub Issues if stuck.
7. **Don't modify the model architecture code** unless you know what you're doing. The LoRA config, quantization, and trainer setup are carefully tuned for 12 GB VRAM.
8. **If training loss is not decreasing**, the learning rate might be wrong or the data paths are incorrect (model is training on empty/broken images).

### Environment Variables

The project uses `HF_TOKEN` for HuggingFace authentication. Set it up:
- **Option A**: Copy `.env.example` to `.env` and add your token
- **Option B**: Run `huggingface-cli login` once (saves token globally)
- **NEVER hardcode tokens in notebook cells**

### Expected Training Behavior

- First epoch takes ~2 hours on RTX 5080 with 160K images
- Loss should start around 2-4 and decrease to <0.5 by epoch 3
- You'll see a progress bar with step count, loss, and learning rate
- Early stopping may trigger before all 5 epochs if loss plateaus
- Total training: ~8-12 hours for 5 epochs
- Output: LoRA adapter saved to `./explainmyxray-v2-medgemma-padchest/` (~200 MB)

### After Training — How to Use

```python
# In the notebook, Cell 24:
prediction = predict_xray("C:/path/to/any/chest_xray.png", view="PA")
```

This prints a structured report (FINDINGS / LOCATIONS / IMPRESSION) and shows a side-by-side visualization with the original X-ray and annotated overlay.

---

## PROMPT END

---

> **Note**: This prompt gives the AI enough context to help you through the entire setup, training, and debugging process without needing to ask you many clarifying questions. If you hit any error, just paste the full error traceback and the AI will know exactly what to fix.
