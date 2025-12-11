from transformers import AutoTokenizer
model_name = "PlanTL-GOB-ES/roberta-base-bne"
print(f"Attempting to load {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Success loading Base!")
except Exception as e:
    print(f"Failed loading Base: {e}")
