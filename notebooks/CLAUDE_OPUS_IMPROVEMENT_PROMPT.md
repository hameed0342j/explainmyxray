# Claude Opus 4.6 — ExplainMyXray v2 Improvement Prompt

> **Copy-paste this ENTIRE prompt to Claude Opus 4.6 to get it to fix and improve the notebook.**

---

## CONTEXT

I have a medical AI notebook (`ExplainMyXray_v2.ipynb`) that fine-tunes **MedGemma-4B-it** on the **BIMCV PadChest** chest X-ray dataset using **QLoRA**. The notebook was originally designed for an RTX 4080 with 1 TB of PadChest data streamed via Google Drive.

**My friend is running it on his system with:**
- RTX 4080 (12 GB VRAM, Ada Lovelace, compute capability 8.9)
- **200 GB of PadChest data stored locally** on NVMe SSD (~30,000-40,000 images of 160K total)
- Linux (not Colab, not Kaggle)
- No session time limits

I need you to **fix 7 identified bugs** and **optimize the notebook** for this smaller-data, faster-I/O setup.

---

## BUGS TO FIX (listed by severity)

### BUG 1 (HIGH): Image resolution mismatch
**Location:** Config class + `preprocess_medical_image()` function
**Problem:** `image_size = 512` preprocesses images to 512x512, but MedGemma's SigLIP encoder expects **896x896**. The `AutoProcessor` will upscale the already-downscaled 512 image, losing ~67% of detail.
**Fix:** Either:
- (A) Change `image_size` to 896 and resize in preprocessing to 896x896 (preserves quality)  
- (B) Better: Don't resize in preprocessing at all — do crop/pad/CLAHE/sharpen but skip the resize. Return the image at its natural resolution and let `AutoProcessor` handle resize to 896x896. This is the most correct approach.

### BUG 2 (HIGH): Label masking doesn't mask prompt tokens
**Location:** `collate_fn` function (Cell 17)
**Problem:** Only pad tokens and image tokens (`262144`) are masked in labels. The system prompt tokens and user prompt tokens are NOT masked, so the model is trained to predict "You are an expert board-certified radiologist..." which wastes training capacity.
**Fix:** Replace the manual collate_fn with TRL's `DataCollatorForCompletionOnlyLM`:
```python
from trl import DataCollatorForCompletionOnlyLM
response_template = "<start_of_turn>model"
collator = DataCollatorForCompletionOnlyLM(
    response_template_ids=processor.tokenizer.encode(response_template, add_special_tokens=False),
    tokenizer=processor.tokenizer
)
```
Or manually mask everything before the assistant response start token with `-100`.

### BUG 3 (MEDIUM): Location attribution is wrong
**Location:** `build_assistant_response()` function (Cell 11)
**Problem:** All locations are paired with every finding. The code does:
```python
matched = [l for l in lu if l]
```
This gives EVERY finding ALL locations, creating medically incorrect training labels (e.g., attributing "cardiac silhouette" to "pneumothorax").
**Fix:** Parse `LabelsLocalizationsBySentence` column to extract per-finding location pairs. This column has the sentence-level finding-location associations. Use those pairs instead of flattening and recombining.

### BUG 4 (MEDIUM): Curriculum learning contaminates test split
**Location:** Cells 12-14
**Problem:** Data is sorted by difficulty BEFORE train/val/test split. Result: train = easy cases, test = hardest cases only. Evaluation metrics are artificially pessimistic.
**Fix:** Split data randomly FIRST, then apply curriculum sorting ONLY to the train split:
```python
random.shuffle(examples)
n_train = int(n * 0.90)
train_ex = sorted(examples[:n_train], key=difficulty_key)
val_ex = examples[n_train:n_train+n_val]
test_ex = examples[n_train+n_val:]
```

### BUG 5 (MEDIUM): No data augmentation with small dataset
**Location:** `preprocess_medical_image()` and `row_to_example()`
**Problem:** With 30-40K images (200 GB subset), overfitting is a real risk. Zero augmentation is used.
**Fix:** Add conservative medical-appropriate augmentations:
- Small rotation: ±3-5° (realistic positioning variation)
- Slight brightness/contrast jitter: ±5-10%
- **DO NOT** horizontal flip (left/right matters — dextrocardia, pleural effusion laterality)
- Apply augmentations only to TRAINING data, not validation/test

### BUG 6 (LOW): System prompt mentions "PadChest dataset"
**Location:** `SYSTEM_PROMPT` (Cell 11)
**Problem:** "from the BIMCV PadChest dataset" in the system prompt means the model will reference PadChest during inference on real-world X-rays.
**Fix:** Remove the dataset reference: "You are an expert board-certified radiologist AI analyzing chest X-rays."

### BUG 7 (LOW): max_seq_length=768 may truncate complex reports
**Location:** Config class
**Fix:** Increase to 1024. Profile actual tokenized lengths first with:
```python
lengths = [len(processor.tokenizer.encode(text)) for text in all_texts]
print(f"P95={np.percentile(lengths,95):.0f}, P99={np.percentile(lengths,99):.0f}, Max={max(lengths)}")
```

---

## OPTIMIZATION REQUESTS FOR 200 GB LOCAL DATA

### 1. Adjust hyperparameters for smaller dataset:
- `lora_r`: 64 → **32** (less capacity needed for fewer examples, reduces overfitting)
- `lora_alpha`: 128 → **64** (keep alpha/r ratio at 2)
- `lora_dropout`: 0.05 → **0.1** (more regularization)
- `learning_rate`: 5e-5 → **3e-5** (slower learning, less overfitting)
- `warmup_ratio`: 0.1 → **0.15** (gentler start)
- `num_train_epochs`: 8 → **12** (can afford more epochs with fast local I/O)
- `max_seq_length`: 768 → **1024**
- Early stopping patience: 5 → **3** (stop faster if overfit)

### 2. Add a data profiling cell after CSV load:
- Count how many images actually exist in the 200 GB
- Show finding distribution (bar chart of top 30 findings)
- Show how many findings have <10 examples (these are noise)
- Optionally filter out findings with <5 examples

### 3. Add training monitoring:
- Print VRAM usage every N steps
- Show sample predictions during training (every eval step)
- Plot training loss vs eval loss live

### 4. Add a quick sanity check cell:
- After model loads, run inference on 1 test image WITHOUT fine-tuning to establish baseline
- Compare pre-finetune vs post-finetune on same image

---

## CONSTRAINTS

1. **Must fit in 12 GB VRAM** — no exceptions. batch_size=1 is mandatory.
2. **Must use bf16** — RTX 4080 has Ada Lovelace tensor cores that support BF16 natively. Do NOT use fp16.
3. **Do not change model** — keep `google/medgemma-4b-it`. Do not switch to MedGemma 1.5 (friend doesn't have the license yet).
4. **Keep the 7-stage preprocessing pipeline** — it's battle-tested. Only change the resize step.
5. **Keep the structured report format** (FINDINGS / LOCATIONS / IMPRESSION).
6. **All data is LOCAL** — remove all Google Drive detection code. Replace with simple path configuration.
7. **Linux paths** — use `/home/...` style paths, not Windows.
8. **Keep it as a single notebook** — no external scripts, no separate files.

---

## DELIVERABLE

Produce the **complete fixed notebook code** with:
1. All 7 bugs fixed
2. Hyperparameters optimized for 200 GB local data
3. Data profiling cell added
4. Training monitoring improvements
5. Pre/post fine-tune comparison cell
6. Clean, well-commented code
7. Google Drive code removed (hardcoded local paths with clear config)

Mark every change with a comment like `# FIX: Bug #N — description` so I can verify.

---

## TECHNICAL REFERENCE

- Model: `google/medgemma-4b-it` (SigLIP 896x896 vision encoder + Gemma 3 4B decoder)
- Dataset: BIMCV PadChest (160K images, 174 findings, 104 locations) — friend has ~200 GB subset
- Fine-tuning: QLoRA NF4, paged_adamw_8bit optimizer, gradient checkpointing
- Framework: Transformers ≥4.52.0, TRL ≥0.17.0, PEFT ≥0.15.0, BitsAndBytes ≥0.45.0
- GPU: RTX 4080 12 GB (cc 8.9, bf16+tf32+fp8 tensor cores)
- Expected training: ~6-9 hours total for 12 epochs on 30-40K images
