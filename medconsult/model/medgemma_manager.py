"""
MedGemma model manager: loads google/medgemma-1.5-4b-it for medical inference.
Handles text and image inputs; runs on GPU when available.
"""

import os
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class MedGemmaManager:
    """Singleton: loads and runs MedGemma 1.5 4B for medical inference."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MedGemmaManager, cls).__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_hf_token(self):
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        hf_token_path = Path.home() / ".huggingface" / "token"
        if hf_token_path.exists():
            with open(hf_token_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        print("ERROR: Hugging Face token not found. Please set HF_TOKEN environment variable or login via huggingface-cli.")
        return None

    def load_model(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor

        token = self._get_hf_token()
        model_id = "google/medgemma-1.5-4b-it"

        if self.device == "cpu":
            print("WARNING: GPU not available. Loading model to CPU. This will be slow.")

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        # Automatically download and load model and processor
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
            token=token
        )

        return self.model, self.processor

    def _normalize_image(self, image):
        """Convert image to RGB PIL Image. Accepts PIL Image, file path (str/Path), or numpy array."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        try:
            import numpy as np
            if isinstance(image, np.ndarray):
                return Image.fromarray(image).convert("RGB")
        except ImportError:
            pass
        return image  # Unknown type — pass through and let processor handle/fail

    def generate_response(self, system_prompt, user_message, image=None, max_tokens=1024):
        self.load_model()

        if image is not None:
            image = self._normalize_image(image)

        if image is not None:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": user_message}
            ]
        else:
            user_content = user_message

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Convert chat array to a formatted string using the processor
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if image is not None:
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")
            
        # Move inputs to the correct device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if hasattr(v, 'to')}

        # Generate per specs: max_tokens, temperature 0.1
        generate_kwargs = {
            "max_new_tokens": max_tokens, 
            "do_sample": True,
            "temperature": 0.1
        }
        
        # Fallback to deterministic if CPU without do_sample support
        if self.device == "cpu":
             generate_kwargs["do_sample"] = False
             generate_kwargs.pop("temperature")

        outputs = self.model.generate(**inputs, **generate_kwargs)

        generated_token_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.processor.decode(generated_token_ids, skip_special_tokens=True)
        return generated_text.strip()
