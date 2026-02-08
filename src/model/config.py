"""
QLoRA and Model Configuration for MedGemma
"""
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType


def get_bnb_config():
    """4-bit quantization config for Colab T4 compatibility."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config():
    """LoRA adapter configuration for efficient fine-tuning."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# Model configurations
MODEL_CONFIGS = {
    "paligemma-3b": {
        "model_id": "google/paligemma-3b-pt-224",
        "processor_id": "google/paligemma-3b-pt-224",
        "image_size": 224,
        "max_length": 512,
    },
    "paligemma-3b-448": {
        "model_id": "google/paligemma-3b-mix-448",
        "processor_id": "google/paligemma-3b-mix-448",
        "image_size": 448,
        "max_length": 512,
    },
}

DEFAULT_MODEL = "paligemma-3b"
