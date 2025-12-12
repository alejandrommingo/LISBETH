
import shutil
import os

path = os.path.expanduser("~/.cache/huggingface/hub/models--PlanTL-GOB-ES--roberta-large-bne")
if os.path.exists(path):
    print(f"Removing corrupted cache: {path}")
    shutil.rmtree(path)
else:
    print("Cache not found.")

path_base = os.path.expanduser("~/.cache/huggingface/hub/models--PlanTL-GOB-ES--roberta-base-bne")
if os.path.exists(path_base):
    print(f"Removing corrupted cache: {path_base}")
    shutil.rmtree(path_base)
else:
    print("Base cache not found.")
