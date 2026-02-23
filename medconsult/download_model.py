import os
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

def download_model():
    token = os.environ.get("HF_TOKEN")
    model_id = "google/medgemma-1.5-4b-it"
    
    print(f"Downloading {model_id}...")
    print("This may take some time depending on your connection speed (~16GB).")
    
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        token=token
    )
    
    print("Download complete!")

if __name__ == "__main__":
    download_model()
