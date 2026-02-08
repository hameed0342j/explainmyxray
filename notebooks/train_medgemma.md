# ExplainMyXray - MedGemma Training Notebook
# Fine-tune PaliGemma for Chest X-ray to Patient-Friendly Text

## Cell 1: Install Dependencies
```python
!pip install -q -U transformers datasets peft accelerate bitsandbytes pillow pandas tqdm
!pip install -q -U torch torchvision

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Cell 2: Login to Hugging Face (Required for Gemma)
```python
from huggingface_hub import login

# Get your token from: https://huggingface.co/settings/tokens
# You need to accept the Gemma license first: https://huggingface.co/google/paligemma-3b-pt-224
login(token="YOUR_HF_TOKEN_HERE")
```

## Cell 3: Configuration
```python
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType

# Model Configuration
MODEL_ID = "google/paligemma-3b-pt-224"  # or paligemma-3b-mix-448 for higher resolution
OUTPUT_DIR = "./medgemma_lora_adapters"

# 4-bit Quantization (QLoRA) - Fits on Colab T4
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Training Configuration
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 256
```

## Cell 4: Load Model and Processor
```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model, prepare_model_for_kbit_training

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model in 4-bit...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=BNB_CONFIG,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()
```

## Cell 5: Upload Your Dataset
```python
from google.colab import files
import pandas as pd

# Option 1: Upload CSV directly
print("Upload your CSV file with ImageID and Report columns:")
uploaded = files.upload()

# Get the filename
csv_filename = list(uploaded.keys())[0]
df = pd.read_csv(csv_filename)
print(f"Loaded {len(df)} samples")
print(df.head())

# Option 2: Use sample data (for testing)
# Uncomment below to create sample data
# df = pd.DataFrame({
#     "ImageID": ["sample1.png", "sample2.png"],
#     "Report": ["Normal chest xray", "Cardiomegaly noted"],
#     "SimplifiedReport": ["Your chest looks healthy.", "Your heart appears slightly enlarged."]
# })
```

## Cell 6: Upload Images
```python
import os
from google.colab import files

# Create images directory
os.makedirs("images", exist_ok=True)

print("Upload your X-ray images (PNG/JPG):")
uploaded_images = files.upload()

# Move to images folder
for filename, content in uploaded_images.items():
    with open(f"images/{filename}", "wb") as f:
        f.write(content)

print(f"Uploaded {len(uploaded_images)} images to ./images/")
```

## Cell 7: Create Dataset
```python
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, processor, max_length=256):
        self.df = df.dropna(subset=["ImageID"]).reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_length = max_length
        
        # Use SimplifiedReport if available, else Report
        self.text_col = "SimplifiedReport" if "SimplifiedReport" in df.columns else "Report"
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row["ImageID"]
        if not img_path.exists():
            # Try without extension matching
            matches = list(self.image_dir.glob(f"{row['ImageID'].split('.')[0]}.*"))
            img_path = matches[0] if matches else img_path
            
        image = Image.open(img_path).convert("RGB")
        
        # Prompt and target
        prompt = "describe this chest xray in simple terms for a patient:"
        target = str(row[self.text_col]) if pd.notna(row[self.text_col]) else ""
        
        # Process
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        
        labels = self.processor.tokenizer(
            target,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }

# Create dataset
dataset = ChestXrayDataset(df, "images", processor, MAX_LENGTH)
print(f"Dataset size: {len(dataset)}")

# Split train/val
from torch.utils.data import random_split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
```

## Cell 8: Data Collator
```python
from dataclasses import dataclass
from typing import Dict, List
import torch

@dataclass
class MultimodalCollator:
    processor: any
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        
        # Mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

collator = MultimodalCollator(processor)
```

## Cell 9: Training
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",  # Disable wandb
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
)

print("Starting training...")
trainer.train()
```

## Cell 10: Save LoRA Adapters
```python
# Save only the LoRA adapters (small file!)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"✅ Adapters saved to {OUTPUT_DIR}")

# Check size
import os
total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR))
print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
```

## Cell 11: Download Adapters
```python
from google.colab import files
import shutil

# Zip the adapters folder
shutil.make_archive("medgemma_lora_adapters", "zip", OUTPUT_DIR)

# Download
files.download("medgemma_lora_adapters.zip")
print("✅ Download complete! Extract and use with the inference script.")
```

## Cell 12: Test Inference
```python
from PIL import Image

# Test with a sample image
test_image_path = list(Path("images").glob("*.png"))[0] if list(Path("images").glob("*.png")) else None

if test_image_path:
    test_image = Image.open(test_image_path).convert("RGB")
    
    inputs = processor(
        images=test_image,
        text="describe this chest xray in simple terms for a patient:",
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
        )
    
    result = processor.decode(outputs[0], skip_special_tokens=True)
    print("Generated explanation:")
    print(result)
else:
    print("No test image found")
```

---

## Usage Instructions

1. **Run cells 1-4** to set up the environment and load the model
2. **Upload your dataset** (CSV with ImageID and Report columns) in cell 5
3. **Upload your images** in cell 6
4. **Run cells 7-9** to create the dataset and start training
5. **Download the adapters** from cell 11 (~50-100MB)
6. **Use with the local inference script** in your ExplainMyXray project
