
from transformers import AutoTokenizer

model_name = "PlanTL-GOB-ES/roberta-large-bne"
print(f"Loading {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print("Fast tokenizer loaded.")
except Exception as e:
    print(f"Fast tokenizer failed: {e}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("Slow tokenizer loaded.")
    except Exception as e2:
        print(f"Slow tokenizer failed: {e2}")
