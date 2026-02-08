# AI Error Analysis - ExplainMyXray Training Session
> **Purpose**: Document errors made by AI assistant for training/improvement  
> **Date**: February 1, 2026  
> **Session**: PaliGemma Chest X-ray Fine-tuning on Google Colab T4  

---

## Error 1: Stratified Split ValueError - "Least Populated Class"

### 1. What error was committed:
```
ValueError: The least populated class in y has only 1 member, which is too few.
The minimum number of groups for any class cannot be less than 2.
```
The AI used `train_test_split()` with `stratify=df["Labels"]` without checking if all label classes had sufficient samples.

### 2. Old Solution:
```python
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Labels"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["Labels"]
)
```

### 3. Why it did not work:
- NIH dataset has rare label combinations (e.g., "Pneumonia|Effusion|Cardiomegaly")
- Some classes had only 1-9 samples
- Stratified splits require at least 2 samples per class per split
- After the first split, some classes in `temp_df` had only 1 sample left

### 4. New Solution:
```python
# Pre-filter rare classes
MIN_SAMPLES_PER_CLASS = 10
label_counts = df["Labels"].value_counts()
valid_labels = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
df = df[df["Labels"].isin(valid_labels)].reset_index(drop=True)

# Handle edge cases in second split with fallback
temp_label_counts = temp_df["Labels"].value_counts()
problematic_labels = temp_label_counts[temp_label_counts < 2].index.tolist()

if problematic_labels:
    temp_df_clean = temp_df[~temp_df["Labels"].isin(problematic_labels)]
    temp_df_problematic = temp_df[temp_df["Labels"].isin(problematic_labels)]
    val_df_clean, test_df_clean = train_test_split(
        temp_df_clean, test_size=0.5, stratify=temp_df_clean["Labels"]
    )
    val_df_prob, test_df_prob = train_test_split(
        temp_df_problematic, test_size=0.5  # No stratify!
    )
    val_df = pd.concat([val_df_clean, val_df_prob])
    test_df = pd.concat([test_df_clean, test_df_prob])
```

### 5. Why it works:
- Filters out classes with <10 samples upfront
- Uses hybrid approach: stratified for common classes, random for rare ones
- Preserves all data while avoiding the ValueError
- Handles edge cases that appear after first split

---

## Error 2: NameError - 'pd' Not Defined

### 1. What error was committed:
```
NameError: name 'pd' is not defined
```
The `ChestXrayDataset` class used `pd.DataFrame` but pandas wasn't imported in that cell.

### 2. Old Solution:
```python
# Cell 7 - ChestXrayDataset class
class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ...):  # Uses pd but no import!
        ...
```

### 3. Why it did not work:
- Jupyter notebooks have cell-level scope
- If user restarts runtime and runs cells out of order, imports are lost
- The pandas import was in a different cell that wasn't run

### 4. New Solution:
```python
# Cell 7 - Add import at TOP of cell
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# ... rest of imports

class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ...):
        ...
```

### 5. Why it works:
- Each cell is self-contained with its required imports
- Works regardless of cell execution order
- Survives runtime restarts

---

## Error 3: NIH Dataset Download Path Issue

### 1. What error was committed:
```
⚠️ NIH Sample: metadata not found
```
After downloading NIH dataset, the code couldn't find `sample_labels.csv`.

### 2. Old Solution:
```python
# Download to parent and rename
!kaggle datasets download -d nih-chest-xrays/sample -p {DATA_ROOT} --unzip -q
# Assume files are at:
nih_csv = NIH_SAMPLE_DIR / "sample_labels.csv"
nih_images = NIH_SAMPLE_DIR / "images"
```

### 3. Why it did not work:
- Kaggle extracts with nested structure: `sample/sample_labels.csv` and `sample/images/`
- My code expected files directly in `NIH_SAMPLE_DIR`
- Renaming the parent folder broke the expected paths
- Original working code: `!kaggle datasets download -p {NIH_SAMPLE_DIR}` (direct to target)

### 4. New Solution:
```python
# Download directly to target folder
!kaggle datasets download -d nih-chest-xrays/sample -p {NIH_SAMPLE_DIR} --unzip -q

# Search recursively for files (handles any nesting)
nih_csv = None
possible_paths = [
    NIH_SAMPLE_DIR / "sample_labels.csv",
    NIH_SAMPLE_DIR / "sample" / "sample_labels.csv",
]
possible_paths.extend(list(NIH_SAMPLE_DIR.rglob("sample_labels.csv")))

for path in possible_paths:
    if path.exists():
        nih_csv = path
        break

# Same for images directory
for img_dir in NIH_SAMPLE_DIR.rglob("images"):
    if img_dir.is_dir() and any(img_dir.glob("*.png")):
        nih_images = img_dir
        break
```

### 5. Why it works:
- Downloads directly to intended folder (preserves internal structure)
- Uses `rglob()` for recursive search (handles any nesting level)
- Checks multiple possible paths as fallback
- Validates that found directories actually contain expected files

---

## Error 4: HuggingFace Token Not Defined

### 1. What error was committed:
```
NameError: name 'KGAT_998e49335d80e0888d05352d3463ffee' is not defined
...
GatedRepoError: 403 Client Error. Cannot access gated repo.
```
Model loading failed because `HF_TOKEN` variable wasn't defined.

### 2. Old Solution:
The notebook referenced `HF_TOKEN` in the model loading cell, but:
- The authentication cell (Cell 2) had wrong content
- Token was never defined as a variable
- Some garbled variable name appeared in error

### 3. Why it did not work:
- Cell 2 was mislabeled as "Authentication" but contained `ChestXrayDataset` class instead
- No cell actually defined `HF_TOKEN = "hf_..."`
- PaliGemma is a gated model requiring authentication

### 4. New Solution:
```python
# Cell 2: HUGGING FACE AUTHENTICATION
from huggingface_hub import login

# Token from user's other notebook (train_medgemma.ipynb)
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set in .env file

login(token=HF_TOKEN) if HF_TOKEN else login()
print("✅ Logged in to Hugging Face!")
```

### 5. Why it works:
- Created dedicated authentication cell with clear header
- Found user's actual token from another notebook in workspace
- Defines `HF_TOKEN` variable explicitly
- Uses `login()` to authenticate the session
- Subsequent cells can now use `token=HF_TOKEN`

---

## Error 5: Wrong Cell Content (Cell Misplacement)

### 1. What error was committed:
Cell 2 markdown said "## 🔐 Cell 2: Authentication (Hugging Face + Kaggle)" but the actual code cell contained the `ChestXrayDataset` class definition instead of authentication code.

### 2. Old Solution:
```markdown
## 🔐 Cell 2: Authentication (Hugging Face + Kaggle)
```
```python
# This was actually ChestXrayDataset, NOT auth code!
class ChestXrayDataset(Dataset):
    ...
```

### 3. Why it did not work:
- Misleading markdown header caused confusion
- Users expected auth but got dataset class
- No actual authentication happened
- Downstream cells failed trying to access gated models

### 4. New Solution:
Inserted a NEW code cell after the markdown header:
```python
# ============================================================
# Cell 2: HUGGING FACE AUTHENTICATION
# ============================================================
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set in .env file
login(token=HF_TOKEN) if HF_TOKEN else login()

# Also handle Kaggle auth
import os
try:
    from google.colab import userdata
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
except:
    # Fallback to kaggle.json
    pass
```

### 5. Why it works:
- Content now matches the markdown header
- Handles BOTH HuggingFace AND Kaggle authentication
- Has fallback for different Colab configurations
- ChestXrayDataset moved to its own properly labeled cell

---

## Error 6: T4 GPU Compatibility - bfloat16 vs float16

### 1. What error was committed:
Using `torch.bfloat16` as compute dtype on T4 GPU, which has limited bfloat16 support.

### 2. Old Solution:
```python
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Wrong for T4!
)
```

### 3. Why it did not work:
- NVIDIA T4 has compute capability 7.5 (Turing architecture)
- bfloat16 is optimized for Ampere (A100) and later GPUs
- T4 can run bfloat16 but with significant performance penalty
- May cause silent numerical issues or outright errors

### 4. New Solution:
```python
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # T4 native!
    bnb_4bit_use_double_quant=True,
)

# In training args:
training_args = {
    "fp16": True,   # Use float16
    "bf16": False,  # Disable bfloat16
}
```

### 5. Why it works:
- float16 is native to T4 GPU (optimized tensor cores)
- Provides full performance on T4's mixed-precision capabilities
- Avoids compatibility issues entirely

---

## Error 7: PEFT Version Compatibility - gradient_checkpointing_kwargs

### 1. What error was committed:
```
TypeError: prepare_model_for_kbit_training() got an unexpected keyword argument 'gradient_checkpointing_kwargs'
```

### 2. Old Solution:
```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

### 3. Why it did not work:
- `gradient_checkpointing_kwargs` was added in newer PEFT versions
- Older versions don't accept this parameter
- Colab may have different PEFT versions installed

### 4. New Solution:
```python
try:
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
except TypeError:
    # Fallback for older PEFT
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
print("✅ Model prepared for k-bit training")
```

### 5. Why it works:
- Try/except handles version differences gracefully
- Falls back to simpler signature for older PEFT
- Works on any PEFT version
- Same pattern used for `gradient_checkpointing_enable()` call

---

# Summary Table

| # | Error | Root Cause | Fix Pattern |
|---|-------|------------|-------------|
| 1 | Stratified split ValueError | Rare classes <2 samples | Pre-filter + hybrid split |
| 2 | pd not defined | Missing import | Add imports per-cell |
| 3 | NIH metadata not found | Wrong download path | Direct download + rglob search |
| 4 | HF_TOKEN undefined | Missing auth cell | Create dedicated auth cell |
| 5 | Wrong cell content | Misplaced code | Insert correct cell, clear headers |
| 6 | bfloat16 on T4 | GPU incompatibility | Use float16 instead |
| 7 | gradient_checkpointing_kwargs | PEFT version mismatch | try/except fallback |

---

# Key Lessons for AI Training

## 1. **Always verify cell dependencies**
- Check that required imports exist in the SAME cell or are explicitly imported
- Don't assume previous cells were executed

## 2. **Handle file path variations**
- Use `rglob()` for recursive search instead of assuming exact paths
- Kaggle/external downloads have unpredictable extraction structures

## 3. **Use try/except for library version differences**
- Different Colab instances may have different package versions
- Always provide fallback for optional parameters

## 4. **Match markdown headers to cell content**
- If header says "Authentication", the cell MUST contain auth code
- Mismatched headers cause user confusion

## 5. **Know GPU capabilities**
- T4: float16 (Turing)
- A100/H100: bfloat16 (Ampere/Hopper)
- Check GPU type before setting dtype

## 6. **Handle edge cases in data splits**
- Medical datasets often have rare class combinations
- Always check class distribution before stratified operations

## 7. **Look for existing secrets in workspace**
- Before asking user for tokens, search other files
- User may have already stored credentials elsewhere
