# ExplainMyXray - Technical Decisions Reference Guide
> **Purpose**: Document why each technology, library, approach, and hyperparameter was chosen  
> **Project**: Chest X-ray Report Generation using Fine-tuned Vision-Language Model  
> **Environment**: Google Colab T4 GPU (16GB VRAM)  

---

# Table of Contents
1. [Libraries & Frameworks](#1-libraries--frameworks)
2. [Model Selection](#2-model-selection)
3. [Quantization & Memory Optimization](#3-quantization--memory-optimization)
4. [LoRA Fine-tuning](#4-lora-fine-tuning)
5. [Datasets](#5-datasets)
6. [Data Processing & Splitting](#6-data-processing--splitting)
7. [Training Hyperparameters](#7-training-hyperparameters)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Callbacks & Training Control](#9-callbacks--training-control)

---

# 1. Libraries & Frameworks

## 1.1 Transformers
| Attribute | Value |
|-----------|-------|
| **Terminology** | Transformers (HuggingFace) |
| **Version Used** | ≥4.47.0 |
| **Use Case** | Provides pre-trained models, tokenizers, processors, and training utilities for NLP/Vision tasks |
| **Why Specifically Here** | Required to load PaliGemma model (`PaliGemmaForConditionalGeneration`), handle image+text processing (`AutoProcessor`), and use the `Trainer` class for training loop |
| **Alternatives** | PyTorch Lightning (more manual), JAX/Flax (different ecosystem), TensorFlow Keras (different backend), Custom PyTorch training loop (more code) |

## 1.2 PEFT (Parameter-Efficient Fine-Tuning)
| Attribute | Value |
|-----------|-------|
| **Terminology** | PEFT |
| **Version Used** | ≥0.14.0 |
| **Use Case** | Enables LoRA, QLoRA, and other parameter-efficient training methods |
| **Why Specifically Here** | 3B parameter model won't fit in 16GB T4 VRAM for full fine-tuning. PEFT allows training only 0.2% of parameters while freezing the rest |
| **Alternatives** | Full fine-tuning (needs >48GB VRAM), Adapter-Tuning (similar but older), Prefix-Tuning (less effective for vision), BitFit (trains only biases) |

## 1.3 BitsAndBytes
| Attribute | Value |
|-----------|-------|
| **Terminology** | bitsandbytes |
| **Version Used** | ≥0.45.0 |
| **Use Case** | Enables 4-bit and 8-bit quantization of model weights to reduce memory |
| **Why Specifically Here** | PaliGemma 3B requires ~12GB in FP16. With 4-bit quantization, it fits in ~4GB, leaving room for gradients/activations |
| **Alternatives** | GPTQ (static quantization, less flexible), GGML/GGUF (for CPU inference), AWQ (newer, less mature), No quantization (needs A100 80GB) |

## 1.4 Accelerate
| Attribute | Value |
|-----------|-------|
| **Terminology** | Accelerate (HuggingFace) |
| **Version Used** | ≥1.0.0 |
| **Use Case** | Handles device placement, mixed precision, and distributed training |
| **Why Specifically Here** | Works with `device_map="auto"` to automatically distribute model layers across GPU/CPU. Essential for `Trainer` class |
| **Alternatives** | Manual device placement (`model.to("cuda")`), DeepSpeed (overkill for single GPU), FSDP (for multi-GPU) |

## 1.5 PyTorch
| Attribute | Value |
|-----------|-------|
| **Terminology** | PyTorch |
| **Version Used** | 2.x (Colab default) |
| **Use Case** | Deep learning framework - tensors, autograd, neural network building blocks |
| **Why Specifically Here** | HuggingFace Transformers is built on PyTorch. Required for all model operations |
| **Alternatives** | TensorFlow (different ecosystem), JAX (functional, Google-preferred), MXNet (less popular) |

## 1.6 PIL/Pillow
| Attribute | Value |
|-----------|-------|
| **Terminology** | Pillow (PIL Fork) |
| **Version Used** | ≥10.0.0 |
| **Use Case** | Image loading, manipulation, and preprocessing |
| **Why Specifically Here** | Loads chest X-ray images, converts to RGB, handles grayscale→RGB conversion |
| **Alternatives** | OpenCV (heavier, more features), imageio (less common), torchvision.io (PyTorch native) |

## 1.7 Torchvision
| Attribute | Value |
|-----------|-------|
| **Terminology** | torchvision.transforms |
| **Use Case** | Image augmentation and preprocessing pipelines |
| **Why Specifically Here** | Applies training augmentations (RandomHorizontalFlip, RandomRotation, ColorJitter) to increase data diversity |
| **Alternatives** | Albumentations (more augmentations), imgaug (deprecated), Kornia (GPU augmentations) |

## 1.8 Scikit-learn
| Attribute | Value |
|-----------|-------|
| **Terminology** | sklearn (scikit-learn) |
| **Use Case** | Machine learning utilities - data splitting, metrics |
| **Why Specifically Here** | `train_test_split` with `stratify` parameter ensures each class is proportionally represented in train/val/test |
| **Alternatives** | Manual splitting (loses stratification), PyTorch random_split (no stratification) |

## 1.9 Evaluate
| Attribute | Value |
|-----------|-------|
| **Terminology** | evaluate (HuggingFace) |
| **Use Case** | Standardized metric computation for NLP/Vision tasks |
| **Why Specifically Here** | Computes BLEU and ROUGE scores to evaluate generated reports vs ground truth |
| **Alternatives** | nltk.translate.bleu_score (older), rouge_score package directly, Custom metrics |

---

# 2. Model Selection

## 2.1 PaliGemma-3B-pt-224
| Attribute | Value |
|-----------|-------|
| **Terminology** | PaliGemma (Vision-Language Model) |
| **Model ID** | `google/paligemma-3b-pt-224` |
| **Use Case** | Multimodal model that can see images AND generate text descriptions |
| **Why Specifically Here** | |
| | 1. **Vision-Language**: Can process X-ray images AND generate medical reports |
| | 2. **Size**: 3B parameters - small enough for T4 with quantization |
| | 3. **224px input**: Efficient for medical images (vs 448px version) |
| | 4. **Pre-trained**: Already understands image-text relationships |
| | 5. **Gemma base**: Modern architecture with good text generation |
| **Alternatives** | |
| | - `google/paligemma-3b-pt-448`: Higher resolution but more VRAM |
| | - `llava-v1.6-mistral-7b`: Larger, needs more VRAM |
| | - `Qwen-VL-Chat`: Chinese-focused, less medical data |
| | - `BiomedCLIP`: Vision only, no text generation |
| | - `MedGemma` (if available): Medical-specific but may not exist |

## 2.2 Why NOT Other Models?
| Model | Reason NOT Used |
|-------|-----------------|
| GPT-4V | Closed source, expensive API, no fine-tuning |
| LLaVA-7B | Too large for T4 (7B > 3B) |
| BLIP-2 | Older architecture, less accurate |
| Florence-2 | Microsoft, different API |
| CogVLM | Very large (17B), Chinese-focused |

---

# 3. Quantization & Memory Optimization

## 3.1 4-bit Quantization (QLoRA)
| Attribute | Value |
|-----------|-------|
| **Terminology** | QLoRA (Quantized LoRA) |
| **Config** | `load_in_4bit=True` |
| **Use Case** | Reduce model memory footprint by 4x (FP32→4-bit) |
| **Why Specifically Here** | PaliGemma 3B needs ~12GB in FP16, but T4 only has 16GB. 4-bit brings it to ~4GB, leaving 12GB for training |
| **Alternatives** | 8-bit quantization (2x memory, slightly better quality), FP16 (no quantization, needs A100), FP32 (impossible on consumer GPUs) |

## 3.2 NF4 Quantization Type
| Attribute | Value |
|-----------|-------|
| **Terminology** | NormalFloat4 (NF4) |
| **Config** | `bnb_4bit_quant_type="nf4"` |
| **Use Case** | Quantization scheme optimized for normally-distributed weights |
| **Why Specifically Here** | NF4 preserves more information than standard FP4 because neural network weights are typically normally distributed |
| **Alternatives** | `fp4` (standard 4-bit, less accurate), `int4` (integer quantization, even less accurate) |

## 3.3 Double Quantization
| Attribute | Value |
|-----------|-------|
| **Terminology** | Double Quantization |
| **Config** | `bnb_4bit_use_double_quant=True` |
| **Use Case** | Quantizes the quantization constants themselves |
| **Why Specifically Here** | Saves additional ~0.4GB memory with minimal quality loss. On tight VRAM budget, every bit counts |
| **Alternatives** | `False` (slightly better quality, uses more memory) |

## 3.4 Compute Dtype: FP16
| Attribute | Value |
|-----------|-------|
| **Terminology** | Compute Data Type |
| **Config** | `bnb_4bit_compute_dtype=torch.float16` |
| **Use Case** | Precision used for matrix multiplications during forward/backward pass |
| **Why Specifically Here** | T4 GPU (Turing architecture) has optimized FP16 tensor cores. BF16 is slower on T4 |
| **Alternatives** | `torch.bfloat16` (for A100/H100), `torch.float32` (slowest, most accurate) |

## 3.5 Gradient Checkpointing
| Attribute | Value |
|-----------|-------|
| **Terminology** | Gradient Checkpointing |
| **Config** | `use_gradient_checkpointing=True` |
| **Use Case** | Trade compute for memory - recompute activations during backward pass instead of storing |
| **Why Specifically Here** | Reduces activation memory by ~60%. Essential for fitting 3B model training on 16GB |
| **Alternatives** | No checkpointing (faster but 2-3x more VRAM needed) |

---

# 4. LoRA Fine-tuning

## 4.1 LoRA (Low-Rank Adaptation)
| Attribute | Value |
|-----------|-------|
| **Terminology** | LoRA (Low-Rank Adaptation of Large Language Models) |
| **Use Case** | Train small "adapter" matrices instead of full model weights |
| **Why Specifically Here** | |
| | 1. **Memory**: Only 0.23% of parameters trained → tiny memory footprint |
| | 2. **Speed**: 10x fewer gradients to compute |
| | 3. **No catastrophic forgetting**: Base model frozen, preserves pre-trained knowledge |
| | 4. **Portability**: Adapter is ~50MB vs 6GB full model |
| **Alternatives** | |
| | - Full fine-tuning (needs 48GB+ VRAM) |
| | - Prefix-tuning (prepends learnable tokens) |
| | - Adapter-tuning (adds layers between existing) |
| | - IA3 (scales activations, even smaller) |

## 4.2 LoRA Rank (r=16)
| Attribute | Value |
|-----------|-------|
| **Terminology** | LoRA Rank |
| **Config** | `lora_r=16` |
| **Use Case** | Controls the size of low-rank matrices (higher = more capacity) |
| **Why Specifically Here** | Rank 16 is a sweet spot - enough capacity for medical knowledge, low enough for T4 VRAM. Higher ranks (32, 64) showed diminishing returns in papers |
| **Alternatives** | r=8 (faster, less capacity), r=32 (more capacity, more VRAM), r=64 (overkill for most tasks) |

## 4.3 LoRA Alpha (α=32)
| Attribute | Value |
|-----------|-------|
| **Terminology** | LoRA Alpha (Scaling Factor) |
| **Config** | `lora_alpha=32` |
| **Use Case** | Scales LoRA outputs. Effective learning rate = α/r × lr |
| **Why Specifically Here** | α=32 with r=16 gives scaling of 2x. This amplifies LoRA updates without increasing rank. Common practice: α = 2×r |
| **Alternatives** | α=r (1x scaling), α=4×r (more aggressive updates) |

## 4.4 Target Modules
| Attribute | Value |
|-----------|-------|
| **Terminology** | Target Modules |
| **Config** | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| **Use Case** | Which layers to apply LoRA adapters to |
| **Why Specifically Here** | Attention layers (Q, K, V, O projections) are the most important for adapting model behavior. Research shows attention adaptation is most effective |
| **Alternatives** | |
| | - Add `gate_proj`, `up_proj`, `down_proj` (FFN layers, more parameters) |
| | - All linear layers (maximum adaptation, most VRAM) |
| | - Only q_proj, v_proj (minimal, less effective) |

## 4.5 LoRA Dropout
| Attribute | Value |
|-----------|-------|
| **Terminology** | LoRA Dropout |
| **Config** | `lora_dropout=0.05` |
| **Use Case** | Regularization to prevent overfitting in LoRA layers |
| **Why Specifically Here** | 5% dropout is light regularization. Medical data has patterns → don't need heavy regularization. Higher dropout slows convergence |
| **Alternatives** | 0.0 (no dropout, risk overfitting), 0.1 (standard), 0.2 (heavy regularization) |

## 4.6 Bias Setting
| Attribute | Value |
|-----------|-------|
| **Terminology** | LoRA Bias |
| **Config** | `bias="none"` |
| **Use Case** | Whether to train bias terms in LoRA layers |
| **Why Specifically Here** | Bias terms add minimal capacity but increase complexity. "none" is the default and works well |
| **Alternatives** | "all" (train all biases), "lora_only" (only LoRA layer biases) |

## 4.7 Task Type
| Attribute | Value |
|-----------|-------|
| **Terminology** | Task Type |
| **Config** | `TaskType.CAUSAL_LM` |
| **Use Case** | Tells PEFT what kind of model/task this is |
| **Why Specifically Here** | PaliGemma is a causal (autoregressive) language model - generates tokens left-to-right |
| **Alternatives** | `SEQ_2_SEQ_LM` (for encoder-decoder), `SEQ_CLS` (classification), `TOKEN_CLS` (NER) |

---

# 5. Datasets

## 5.1 Kaggle Chest X-ray Pneumonia
| Attribute | Value |
|-----------|-------|
| **Terminology** | chest-xray-pneumonia dataset |
| **Source** | `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia` |
| **Size** | ~17,568 images |
| **Use Case** | Binary classification dataset (Normal vs Pneumonia) with X-ray images |
| **Why Specifically Here** | |
| | 1. High quality, curated images |
| | 2. Clear labels (Normal/Pneumonia) |
| | 3. Good image resolution |
| | 4. Widely used benchmark |
| **Alternatives** | CheXpert (larger but requires registration), MIMIC-CXR (requires credentialing), ChestX-ray14 (less curated) |

## 5.2 NIH Chest X-ray Sample
| Attribute | Value |
|-----------|-------|
| **Terminology** | NIH ChestX-ray8 Sample |
| **Source** | `kaggle datasets download -d nih-chest-xrays/sample` |
| **Size** | ~5,606 images |
| **Use Case** | Multi-label classification with 14 disease categories |
| **Why Specifically Here** | |
| | 1. Multiple disease labels (more realistic) |
| | 2. Includes label combinations (e.g., "Pneumonia\|Effusion") |
| | 3. Adds variety to training data |
| | 4. Official NIH dataset (trusted source) |
| **Alternatives** | Full NIH ChestX-ray14 (112k images, 42GB), PadChest (Spanish), VinDr-CXR (Vietnamese) |

## 5.3 Why These Two Together?
| Reason | Explanation |
|--------|-------------|
| **Data Diversity** | Pneumonia dataset is binary; NIH adds multi-label complexity |
| **Volume** | Combined ~23k images is substantial for fine-tuning |
| **Quality** | Both are well-curated, peer-reviewed datasets |
| **Accessibility** | Both available on Kaggle without special approval |
| **Size** | Small enough to download quickly and fit in Colab |

---

# 6. Data Processing & Splitting

## 6.1 Report Generation (Text Labels)
| Attribute | Value |
|-----------|-------|
| **Terminology** | Synthetic Report Generation |
| **Use Case** | Convert disease labels to natural language reports |
| **Why Specifically Here** | PaliGemma generates text, not labels. We need text targets for training. Real radiology reports aren't available in Kaggle datasets |
| **Alternatives** | Use MIMIC-CXR (has real reports but requires credentialing), GPT-4 generated reports (expensive), Manual annotation (time-consuming) |

## 6.2 Stratified Split
| Attribute | Value |
|-----------|-------|
| **Terminology** | Stratified Train/Val/Test Split |
| **Config** | `stratify=df["Labels"]` |
| **Split Ratio** | 80% train, 10% val, 10% test |
| **Use Case** | Ensure each split has proportional representation of all classes |
| **Why Specifically Here** | Medical datasets often have class imbalance. Random split might put all rare diseases in test set. Stratification prevents this |
| **Alternatives** | Random split (simple but risky), K-fold cross-validation (more robust but 5x training time), Leave-one-out (too slow) |

## 6.3 Minimum Samples Per Class Filter
| Attribute | Value |
|-----------|-------|
| **Terminology** | Rare Class Filtering |
| **Config** | `MIN_SAMPLES_PER_CLASS = 10` |
| **Use Case** | Remove classes with too few samples for reliable training/evaluation |
| **Why Specifically Here** | Stratified split requires ≥2 samples per class per split. With 80/10/10 split, need ≥10 samples to guarantee ≥1 in each split |
| **Alternatives** | Keep all classes (causes ValueError), Upsample rare classes (introduces bias), Merge rare classes into "Other" (loses granularity) |

## 6.4 Maximum Samples Control
| Attribute | Value |
|-----------|-------|
| **Terminology** | Dataset Size Limiting |
| **Config** | `MAX_SAMPLES = 8000` |
| **Use Case** | Control training time and prevent overfitting |
| **Why Specifically Here** | Full ~23k samples would take ~6+ hours on T4. 8k samples is a good balance: enough for learning, fast enough for iteration |
| **Alternatives** | 5000 (faster, less data), 15000 (better quality, longer), Full dataset (best quality, longest) |

## 6.5 Image Augmentation
| Attribute | Value |
|-----------|-------|
| **Terminology** | Data Augmentation |
| **Config** | `RandomHorizontalFlip(0.3)`, `RandomRotation(5°)`, `ColorJitter(0.1)` |
| **Use Case** | Artificially increase training data diversity |
| **Why Specifically Here** | |
| | 1. **Horizontal flip**: X-rays can be mirrored (anatomy is symmetric) |
| | 2. **Small rotation**: Accounts for patient positioning variation |
| | 3. **Color jitter**: Accounts for different X-ray machines/settings |
| | 4. **Light augmentation**: Medical images shouldn't be heavily distorted |
| **Alternatives** | Heavy augmentation (distorts medical features), No augmentation (faster but less generalization), Mixup/CutMix (experimental for medical) |

---

# 7. Training Hyperparameters

## 7.1 Batch Size
| Attribute | Value |
|-----------|-------|
| **Terminology** | Per-Device Batch Size |
| **Config** | `per_device_train_batch_size=4` |
| **Use Case** | Number of samples processed before gradient update |
| **Why Specifically Here** | 4 is the maximum that fits in T4 VRAM with 3B quantized model. Higher causes OOM |
| **Alternatives** | 1 (safe but slow), 2 (conservative), 8+ (needs A100) |

## 7.2 Gradient Accumulation
| Attribute | Value |
|-----------|-------|
| **Terminology** | Gradient Accumulation Steps |
| **Config** | `gradient_accumulation_steps=8` |
| **Effective Batch** | 4 × 8 = 32 |
| **Use Case** | Simulate larger batch size by accumulating gradients over multiple forward passes |
| **Why Specifically Here** | Large effective batch (32) provides stable gradients. Actual batch of 4 fits in memory. Best of both worlds |
| **Alternatives** | 4 (effective=16, faster), 16 (effective=64, more stable but slower) |

## 7.3 Learning Rate
| Attribute | Value |
|-----------|-------|
| **Terminology** | Learning Rate |
| **Config** | `learning_rate=2e-4` (0.0002) |
| **Use Case** | Step size for gradient updates |
| **Why Specifically Here** | 2e-4 is the standard for LoRA fine-tuning. Higher rates (1e-3) cause instability with quantized models. Lower rates (1e-5) are too slow |
| **Alternatives** | 1e-4 (more conservative), 5e-4 (faster but riskier), 1e-3 (usually too high for LoRA) |

## 7.4 Learning Rate Scheduler
| Attribute | Value |
|-----------|-------|
| **Terminology** | LR Scheduler |
| **Config** | `lr_scheduler_type="cosine"` |
| **Use Case** | Gradually reduce learning rate during training |
| **Why Specifically Here** | Cosine schedule provides smooth decay. Helps model converge to better minimum. Better than linear for fine-tuning |
| **Alternatives** | "linear" (simple decay), "constant" (no decay), "polynomial" (configurable), "cosine_with_restarts" (for longer training) |

## 7.5 Warmup Steps
| Attribute | Value |
|-----------|-------|
| **Terminology** | Learning Rate Warmup |
| **Config** | `warmup_steps=50` |
| **Use Case** | Gradually increase LR from 0 at start of training |
| **Why Specifically Here** | Prevents gradient explosion at start when weights are random. 50 steps is ~1-2% of training |
| **Alternatives** | `warmup_ratio=0.1` (10% of training), warmup_steps=100 (longer warmup), warmup_steps=0 (risky) |

## 7.6 Weight Decay
| Attribute | Value |
|-----------|-------|
| **Terminology** | Weight Decay (L2 Regularization) |
| **Config** | `weight_decay=0.01` |
| **Use Case** | Penalize large weights to prevent overfitting |
| **Why Specifically Here** | 0.01 is standard for transformer fine-tuning. Prevents LoRA weights from growing too large |
| **Alternatives** | 0.0 (no regularization), 0.1 (heavy regularization), 0.001 (light) |

## 7.7 Max Gradient Norm
| Attribute | Value |
|-----------|-------|
| **Terminology** | Gradient Clipping |
| **Config** | `max_grad_norm=1.0` |
| **Use Case** | Clip gradients to prevent exploding gradients |
| **Why Specifically Here** | Quantized models can have unstable gradients. Clipping to 1.0 is standard safety measure |
| **Alternatives** | 0.5 (more aggressive), 2.0 (less clipping), None (risky) |

## 7.8 Number of Epochs
| Attribute | Value |
|-----------|-------|
| **Terminology** | Training Epochs |
| **Config** | `num_train_epochs=10` |
| **Use Case** | Number of complete passes through training data |
| **Why Specifically Here** | 10 epochs with early stopping usually finds optimal point. LoRA converges faster than full fine-tuning |
| **Alternatives** | 3-5 (quick experiment), 20+ (risk overfitting), 1 (just to test) |

## 7.9 Optimizer
| Attribute | Value |
|-----------|-------|
| **Terminology** | Optimizer |
| **Config** | `optim="adamw_torch_fused"` |
| **Use Case** | Algorithm for updating weights based on gradients |
| **Why Specifically Here** | Fused AdamW is 5-10% faster than standard AdamW. Uses kernel fusion to reduce memory transfers |
| **Alternatives** | "adamw_torch" (standard), "adamw_8bit" (less VRAM), "sgd" (simpler, often worse for transformers) |

## 7.10 Precision (FP16)
| Attribute | Value |
|-----------|-------|
| **Terminology** | Mixed Precision Training |
| **Config** | `fp16=True, bf16=False` |
| **Use Case** | Use 16-bit floats for faster computation |
| **Why Specifically Here** | T4 has optimized FP16 tensor cores. BF16 is not fully supported on T4 (Turing architecture) |
| **Alternatives** | bf16=True (for A100/H100), fp32 (slowest, most accurate), fp16=False (no mixed precision) |

## 7.11 Max Sequence Length
| Attribute | Value |
|-----------|-------|
| **Terminology** | Maximum Sequence Length |
| **Config** | `max_length=384` |
| **Use Case** | Maximum tokens in input+output sequence |
| **Why Specifically Here** | Medical reports are short (~100-200 words). 384 tokens is sufficient and saves VRAM. Longer sequences use quadratically more memory |
| **Alternatives** | 256 (shorter reports only), 512 (longer context), 1024 (for detailed reports, more VRAM) |

---

# 8. Evaluation Metrics

## 8.1 BLEU Score
| Attribute | Value |
|-----------|-------|
| **Terminology** | BLEU (Bilingual Evaluation Understudy) |
| **Library** | `evaluate.load("sacrebleu")` |
| **Use Case** | Measures n-gram overlap between generated and reference text |
| **Why Specifically Here** | Standard metric for text generation. Captures how many words/phrases from reference appear in generation |
| **Alternatives** | METEOR (semantic similarity), CIDEr (image captioning), BERTScore (semantic), chrF (character-based) |

## 8.2 ROUGE Score
| Attribute | Value |
|-----------|-------|
| **Terminology** | ROUGE (Recall-Oriented Understudy for Gisting Evaluation) |
| **Library** | `evaluate.load("rouge")` |
| **Variants** | ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence) |
| **Use Case** | Measures recall of reference n-grams in generated text |
| **Why Specifically Here** | ROUGE-L is good for medical reports - captures if key information is included even if wording differs |
| **Alternatives** | Same as BLEU alternatives |

## 8.3 Training Loss
| Attribute | Value |
|-----------|-------|
| **Terminology** | Cross-Entropy Loss |
| **Use Case** | Measures how well model predicts next token |
| **Why Specifically Here** | Standard loss for language modeling. Directly optimized during training. Lower is better |
| **Alternatives** | Label smoothing loss (regularization), Focal loss (for class imbalance) |

## 8.4 Eval Loss (Validation Loss)
| Attribute | Value |
|-----------|-------|
| **Terminology** | Validation Loss |
| **Use Case** | Loss on held-out data, measures generalization |
| **Why Specifically Here** | Used for early stopping and best model selection. If eval_loss increases while train_loss decreases → overfitting |
| **Alternatives** | BLEU/ROUGE on validation set (slower but more meaningful) |

---

# 9. Callbacks & Training Control

## 9.1 Early Stopping
| Attribute | Value |
|-----------|-------|
| **Terminology** | Early Stopping Callback |
| **Config** | `early_stopping_patience=3, early_stopping_threshold=0.001` |
| **Use Case** | Stop training when validation loss stops improving |
| **Why Specifically Here** | Prevents overfitting and saves time. If loss doesn't improve by 0.001 for 3 evaluations, stop |
| **Alternatives** | No early stopping (train all epochs), patience=5 (more tolerance), patience=1 (aggressive) |

## 9.2 Checkpoint Saving
| Attribute | Value |
|-----------|-------|
| **Terminology** | Model Checkpointing |
| **Config** | `save_steps=250, save_total_limit=3` |
| **Use Case** | Save model periodically to resume if training crashes |
| **Why Specifically Here** | Colab can disconnect. Saving every 250 steps (~2 epochs) ensures minimal lost progress. Limit 3 saves disk space |
| **Alternatives** | save_steps=100 (more frequent), save_total_limit=1 (save space), save_strategy="epoch" (per epoch) |

## 9.3 Load Best Model at End
| Attribute | Value |
|-----------|-------|
| **Terminology** | Best Model Loading |
| **Config** | `load_best_model_at_end=True, metric_for_best_model="eval_loss"` |
| **Use Case** | After training, revert to the checkpoint with best validation loss |
| **Why Specifically Here** | Final epoch may not be the best. This ensures you keep the best-performing model |
| **Alternatives** | Keep final model (may be overfit), Use BLEU as metric (slower evaluation) |

## 9.4 DataLoader Workers
| Attribute | Value |
|-----------|-------|
| **Terminology** | DataLoader Parallelism |
| **Config** | `dataloader_num_workers=4` |
| **Use Case** | Number of CPU processes loading data in parallel |
| **Why Specifically Here** | 4 workers keep GPU fed with data. More workers use more RAM. Colab has 2 CPU cores, 4 workers is efficient |
| **Alternatives** | 0 (single process, slowest), 2 (conservative), 8+ (diminishing returns) |

---

# Quick Reference Card

| Category | Choice | Why |
|----------|--------|-----|
| **Model** | PaliGemma-3B-pt-224 | Vision-language, fits T4 |
| **Quantization** | 4-bit NF4 | 4x memory reduction |
| **Fine-tuning** | LoRA r=16, α=32 | 0.2% parameters trained |
| **Compute Dtype** | FP16 | T4 tensor cores |
| **Batch Size** | 4 × 8 = 32 effective | Max that fits |
| **Learning Rate** | 2e-4 | Standard for LoRA |
| **Scheduler** | Cosine | Smooth decay |
| **Epochs** | 10 + early stopping | Prevent overfitting |
| **Split** | 80/10/10 stratified | Class balance |
| **Metrics** | BLEU, ROUGE | Text generation standard |

---

# Files in This Guide
- [training_errors_log.md](training_errors_log.md) - Error analysis from this session
- This file: Technical decisions reference
