"""
Offline Inference Engine for MedGemma.
Loads base model + LoRA adapters for local predictions.
"""
import io
from pathlib import Path
from typing import Union
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

from src.model.config import get_bnb_config, MODEL_CONFIGS, DEFAULT_MODEL


class MedGemmaPredictor:
    """
    Inference wrapper for the fine-tuned MedGemma model.
    Loads the base model and LoRA adapters for offline use.
    """
    
    def __init__(
        self,
        adapter_path: str,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        """
        Args:
            adapter_path: Path to saved LoRA adapters
            model_name: Base model configuration name
            device: Device to load model on ("auto", "cuda", "cpu")
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.config = MODEL_CONFIGS[model_name]
        self.device = device
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config["processor_id"]
        )
        
        # Load base model
        quantization_config = get_bnb_config() if load_in_4bit else None
        
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            quantization_config=quantization_config,
            device_map=device,
            torch_dtype=torch.float16,
        )
        
        # Load LoRA adapters
        if Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
            )
            print(f"Loaded LoRA adapters from {adapter_path}")
        else:
            print(f"Warning: Adapter path {adapter_path} not found. Using base model.")
        
        self.model.eval()
    
    def predict(
        self,
        image: Union[str, bytes, Image.Image],
        prompt: str = "describe this chest xray in simple terms for a patient:",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate patient-friendly explanation for an X-ray image.
        
        Args:
            image: Image path, bytes, or PIL Image
            prompt: Instruction prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated explanation text
        """
        # Handle different image input types
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        
        # Decode
        generated_text = self.processor.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        # Remove the prompt from output
        if prompt in generated_text:
            generated_text = generated_text.split(prompt)[-1].strip()
        
        return generated_text


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MedGemma Inference")
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    parser.add_argument("--adapters", default="./medgemma_lora_adapters", help="Path to LoRA adapters")
    args = parser.parse_args()
    
    predictor = MedGemmaPredictor(adapter_path=args.adapters)
    result = predictor.predict(args.image)
    print(f"\n📋 Explanation:\n{result}")
