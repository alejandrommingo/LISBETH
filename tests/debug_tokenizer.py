from transformers import AutoTokenizer, RobertaTokenizerFast
import os

model_name = "PlanTL-GOB-ES/roberta-large-bne"
print(f"Attempting to load {model_name}...")

try:
    # Force Fast
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    print("Success loading RobertaTokenizerFast!")
    print(tokenizer)
except Exception as e:
    print(f"Failed loading RobertaTokenizerFast: {e}")

try:
    # Auto loop
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Success loading AutoTokenizer!")
except Exception as e:
    print(f"Failed loading AutoTokenizer: {e}")
