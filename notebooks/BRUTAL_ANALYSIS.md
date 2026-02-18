# ExplainMyXray v2 — BRUTAL ANALYSIS

> **Date:** 2026-02-17 | **Notebook:** `ExplainMyXray_v2.ipynb` (RTX 4080 variant)  
> **Target:** Friend's RTX 4080 + 200 GB local PadChest data  
> **Comparison:** vs T4 (Kaggle/Colab) + 1 TB full PadChest via GDrive

---

## 1. BRUTAL VERDICT

**RTX 4080 + 200 GB local data IS BETTER than T4 + 1 TB GDrive — but the notebook has 7 bugs/design flaws that will tank your results if not fixed.**

---

## 2. RTX 4080 (200 GB local) vs T4 (1 TB GDrive) — HEAD TO HEAD

| Factor | RTX 4080 + 200 GB Local | T4 + 1 TB GDrive | Winner |
|--------|-------------------------|-------------------|--------|
| **GPU Raw Power** | 48.7 TFLOPS FP16 (desktop) / 33.8 (laptop) | 8.1 TFLOPS FP16 | **RTX 4080** (4-6x faster compute) |
| **VRAM** | 12 GB GDDR6X | 15 GB GDDR5 (HBM on Colab) | T4 (3 GB more headroom) |
| **Precision** | BF16 native (cc 8.9) | FP16 only (cc 7.5) | **RTX 4080** (BF16 = better dynamic range, no overflow) |
| **Tensor Cores** | 4th Gen (FP8/BF16/TF32) | 3rd Gen (FP16/INT8) | **RTX 4080** (generation ahead) |
| **Memory Bandwidth** | 716.8 GB/s (desktop) | 320 GB/s | **RTX 4080** (2.2x bandwidth) |
| **Data I/O Speed** | NVMe SSD: 3-7 GB/s | GDrive stream: 10-50 MB/s | **RTX 4080** (100-700x faster I/O) |
| **Data Volume** | ~30-40K images (200 GB) | ~160K images (1 TB) | **T4** (4-5x more data) |
| **Finding Coverage** | ~100-120 of 174 findings | All 174 findings | **T4** (complete coverage) |
| **Training Time/Epoch** | ~30-45 min (local read) | ~2-3 hours (streaming) | **RTX 4080** (4-5x faster) |
| **Total 8 Epochs** | ~4-6 hours | ~16-24 hours | **RTX 4080** (4x faster) |
| **Session Limits** | None (your machine) | 12h Colab / 9h Kaggle | **RTX 4080** (no cutoff) |
| **Cost** | Free (own hardware) | Free tier limits / $10/mo Pro | **RTX 4080** |
| **Reproducibility** | Train anytime, resume anytime | Session dies = restart | **RTX 4080** |

### BOTTOM LINE: RTX 4080 + 200 GB wins on 8 of 11 factors.

The ONLY thing T4 + 1 TB has going for it is more data. But here's the key insight:

> **Quality of training > Quantity of data, and 30-40K well-preprocessed PadChest images is PLENTY for fine-tuning a model that already has medical pre-training (MedGemma was pre-trained on 1M+ medical images including CXR).**

MedGemma's SigLIP encoder already knows what chest X-rays look like. You're fine-tuning the decoder to generate structured reports. 30-40K examples with good preprocessing is more than enough. Papers like CheXagent-2 achieved SOTA with even less fine-tuning data on an already-pretrained model.

**The 4-5x compute speedup means you can iterate faster**: try different LoRA ranks, learning rates, and prompt formats. On T4 you get ONE shot per 24-hour session.

---

## 3. CRITICAL BUGS & DESIGN FLAWS (Fix Before Running)

### BUG 1: Image Resolution is Wrong (HIGH)
**Cell 7 / Config:** `image_size: int = 512`

SigLIP encoder in MedGemma was pre-trained on **896x896** images. You're preprocessing to 512x512, then the `AutoProcessor` resize it again internally. This means:
- You're doing a 512x512 resize (lossy) then the processor does ANOTHER resize to 896x896 (upscaling garbage)
- Net effect: ~67% pixel information lost vs feeding the original image

**Fix:** Change `image_size` to 896, OR (better) **skip the resize entirely** and let `AutoProcessor` handle it. The processor knows the correct resolution for SigLIP. Your preprocessing should do: crop → pad → CLAHE → sharpen → convert RGB, then let the processor resize.

### BUG 2: Label Masking Doesn't Mask Prompts (HIGH)
**Cell 17 (collate_fn):**
```python
labels = batch["input_ids"].clone()
pid = processor.tokenizer.pad_token_id
if pid is not None: labels[labels == pid] = -100
labels[labels == 262144] = -100  # image token
batch["labels"] = labels
```

This only masks padding and image tokens. The system prompt ("You are an expert board-certified radiologist...") and user prompt ("Analyze this chest X-ray...") tokens are NOT masked. The model is being trained to predict these prompt tokens — which is a **complete waste of training capacity**. The model should only learn to predict the **assistant response** (the actual report).

**Fix:** Use TRL's `DataCollatorForCompletionOnlyLM` which handles this automatically:
```python
from trl import DataCollatorForCompletionOnlyLM
response_template = "<start_of_turn>model"  # Gemma's assistant turn token
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=processor.tokenizer)
```
Or manually find the assistant turn start index and set `labels[:, :start_idx] = -100`.

### BUG 3: Location Attribution is Broken (MEDIUM)
**Cell 11 (build_assistant_response):**
```python
for f in abn:
    matched = [l for l in lu if l]
    loc_joined = ', '.join(matched[:3])
```

This assigns ALL locations to EVERY finding. If a patient has `cardiomegaly` in the `cardiac silhouette` and `pneumothorax` in the `right lung`, the code generates:
```
- Cardiomegaly (right lung, cardiac silhouette)
- Pneumothorax (right lung, cardiac silhouette)
```
This is **medically wrong**. The model learns incorrect finding-location associations.

**Fix:** Parse `LabelsLocalizationsBySentence` properly — it contains per-sentence finding-location pairs. Extract the pairing, don't flatten and recombine.

### BUG 4: Curriculum Learning Poisons the Test Split (MEDIUM)
**Cell 12-14:** Data is sorted by difficulty BEFORE splitting. This means:
- Train set = first 90% = EASY cases
- Test set = last 5% = ALL THE HARDEST CASES
- Validation = middle 5% = medium difficulty

This makes evaluation metrics **artificially pessimistic** (you're testing on the hardest cases only) and training **artificially easy** (model never sees hard cases in the easy epochs).

**Fix:** Split FIRST, THEN sort within train split:
```python
# split first
random.shuffle(examples_list)
n_train = int(n * 0.90)
train_ex = examples_list[:n_train]
# THEN sort train by difficulty
train_ex.sort(key=lambda x: compute_difficulty_for_example(x))
```

### BUG 5: No Data Augmentation (MEDIUM)
With only 30-40K images (200 GB subset), overfitting is a real risk. The notebook has ZERO augmentation — no rotation, no brightness jitter, no contrast adjustment. 

**Fix:** Add CONSERVATIVE augmentations (medical imaging rules):
- Small rotation: ±3-5° (realistic patient positioning variation)
- Slight brightness/contrast jitter: ±5%
- **DO NOT** horizontal flip (left vs right matters in CXR — dextrocardia is rare)
- **DO NOT** use aggressive augmentations (elastic deform, cutout, etc.)

### BUG 6: Sequence Length May Truncate Complex Reports (LOW)
`max_seq_length=768` might be too short for reports with 5+ findings, each with locations and an impression section. Run this check:
```python
lengths = [len(processor.tokenizer.encode(text)) for text in all_texts]
print(f"Max: {max(lengths)}, P95: {np.percentile(lengths, 95)}, P99: {np.percentile(lengths, 99)}")
```
If P99 > 768, increase to 1024.

### BUG 7: SYSTEM Prompt Trains the Model to Say "PadChest" (LOW)
The system prompt says "from the BIMCV PadChest dataset." During inference on real-world X-rays, the model will still reference PadChest. The system prompt should be dataset-agnostic for a deployable model.

**Fix:** Remove "from the BIMCV PadChest dataset" from SYSTEM_PROMPT.

---

## 4. 200 GB LOCAL DATA — WHAT YOUR FRIEND ACTUALLY HAS

| Metric | Estimated Value |
|--------|----------------|
| Storage | 200 GB of PadChest |
| Average image size | 5-7 MB per PNG (16-bit + variable resolution) |
| Estimated images | **30,000-40,000** (of 160,000 total) |
| Finding coverage | ~100-120 of 174 unique findings |
| Missing findings | Ultra-rare findings with <50 total images in full dataset |
| Training images (90%) | ~27,000-36,000 |
| Validation images (5%) | ~1,500-2,000 |
| Test images (5%) | ~1,500-2,000 |

### Key Advantages of Local Data:
1. **I/O Speed:** NVMe reads at 3-7 GB/s vs GDrive at 10-50 MB/s — training is I/O bound on T4/Colab, never on RTX 4080
2. **No disconnects:** No Google Drive timeouts, no Colab session resets
3. **Deterministic:** Same data every run, no streaming failures
4. **Can train more epochs:** Faster per-epoch means 12-15 epochs is feasible in same wall time as 8 epochs on T4

### Key Disadvantages:
1. **~75% less data** — missing rare findings, less diversity
2. **Need to verify which image folders are present** — PadChest stores images in numbered subdirectories (0-37)
3. **Label distribution skew** — if your 200 GB is non-random (e.g., first N folders), label distribution differs from full dataset

### Mitigation for Less Data:
1. Train more epochs (10-12 instead of 8)
2. Use stronger augmentation
3. Lower LoRA rank to r=32 (less parameters to learn = less data needed)
4. Increase warmup ratio to 0.15 (slower start, less overfitting)
5. Add dropout to 0.1 (regularization)

---

## 5. RECOMMENDED CONFIG CHANGES FOR 200 GB + RTX 4080

```python
@dataclass
class Config:
    model_id: str = "google/medgemma-4b-it"
    
    # CHANGED: Don't resize - let processor handle it
    image_size: int = 896  # Match SigLIP native resolution
    
    # CHANGED: Lower rank for smaller dataset (less overfitting)
    lora_r: int = 32       # Was 64 — too much capacity for 30K images
    lora_alpha: int = 64   # Keep alpha/r ratio = 2
    lora_dropout: float = 0.1  # Was 0.05 — more regularization for smaller data
    
    # CHANGED: More epochs, smaller data = faster
    num_train_epochs: int = 12  # Was 8 — can afford more with fast I/O
    
    # CHANGED: Slightly lower LR to prevent overfitting
    learning_rate: float = 3e-5  # Was 5e-5
    warmup_ratio: float = 0.15   # Was 0.1 — slower start
    
    # CHANGED: Longer sequences for complex reports  
    max_seq_length: int = 1024  # Was 768
    
    # CHANGED: More aggressive early stopping
    # Early stopping patience = 3 (was 5)
    
    # UNCHANGED (these are fine)
    gradient_accumulation_steps: int = 16
    per_device_train_batch_size: int = 1
    label_smoothing_factor: float = 0.05
    lr_scheduler_type: str = "cosine_with_restarts"
```

---

## 6. TRAINING TIME ESTIMATES (RTX 4080 + 200 GB Local)

| Phase | Time Estimate |
|-------|---------------|
| Install dependencies | 2-5 min |
| Load CSV + parse labels | 1-2 min |
| Preprocess 30-40K images | 15-30 min (CPU bound, 4 workers) |
| Load MedGemma + quantize | 2-3 min |
| Training per epoch | ~30-45 min |
| Total 12 epochs | **6-9 hours** |
| Evaluation (test set) | 10-20 min |
| **TOTAL END-TO-END** | **~7-10 hours** |

vs T4 + 1 TB GDrive: **~20-30 hours** (plus session management hell)

---

## 7. WHAT YOUR FRIEND NEEDS TO DO BEFORE RUNNING

1. **Check which folders they have:**
   ```bash
   ls -la /path/to/padchest/images/ | head -40
   # Should see folders like 0/ 1/ 2/ ... up to 37/
   # Count how many folders and images they have
   find /path/to/padchest/images/ -name "*.png" | wc -l
   ```

2. **Verify CSV exists:**
   ```bash
   ls -la /path/to/padchest/PADCHEST_chest_x_ray_images_labels_160K.csv
   ```

3. **Set paths manually in Cell 6** (don't rely on auto-detect):
   ```python
   cfg.padchest_csv = "/absolute/path/to/PADCHEST_chest_x_ray_images_labels_160K.csv"
   cfg.padchest_images = "/absolute/path/to/images"
   ```

4. **Accept MedGemma license** at https://huggingface.co/google/medgemma-4b-it and run `huggingface-cli login`

5. **Install NVIDIA drivers + CUDA 12.x** — verify with `nvidia-smi`

---

# ARCHITECTURE JSON

See: `RTX4080_ARCHITECTURE.json` in the same directory.

---
