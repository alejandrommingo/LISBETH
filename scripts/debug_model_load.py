
from transformers import AutoConfig
import os

model_name = "PlanTL-GOB-ES/roberta-large-bne"
print(f"Testing access to {model_name}...")
try:
    config = AutoConfig.from_pretrained(model_name)
    print("Success! Config loaded.")
except Exception as e:
    print(f"Failed: {e}")
    # Try listing local cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        print("Cache contents:")
        print(os.listdir(cache_dir))
