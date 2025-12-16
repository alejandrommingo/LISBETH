import pandas as pd
import numpy as np
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_CSV = "data/phase2/embeddings_occurrences.csv"
OUTPUT_CSV = "data/phase2/embeddings_occurrences.csv" # Overwrite
MODEL_PATH = "models/beto-adapted"

def get_embedding(model, tokenizer, text, keyword):
    """
    Extracts embeddings for the keyword in text using subword mean pooling.
    Returns (penultimate_layer, last4_concat).
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"][0]
    
    # Locate keyword tokens
    # Note: This is a simplified heuristic. Ideally reuse the robustness of Phase 2 pipeline.
    # Keyword might be split.
    # We look for the sequence of tokens matching keyword.
    
    kw_ids = tokenizer.encode(keyword, add_special_tokens=False)
    if not kw_ids:
        return None, None
        
    # Search for sublist
    len_kw = len(kw_ids)
    start_idx = -1
    
    # Naive search
    input_ids_list = input_ids.tolist()
    for i in range(len(input_ids_list) - len_kw + 1):
        if input_ids_list[i:i+len_kw] == kw_ids:
            start_idx = i
            break
            
    if start_idx == -1:
        # Fallback: try case insensitive? Or substring?
        # For this script we try best effort. 
        # If fuzzy match needed, it gets complex. 
        return None, None

    end_idx = start_idx + len_kw
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    # Penultimate: -2
    # Last 4: -1, -2, -3, -4
    
    # Stack layers: (layers, seq_len, hidden_dim)
    # We want specific tokens [start_idx:end_idx]
    
    # Penultimate
    pen_layer = hidden_states[-2][0] # (seq, dim)
    span_emb_pen = pen_layer[start_idx:end_idx].mean(dim=0).cpu().numpy()
    
    # Last 4 concat
    # Concat last 4 layers -> (seq, 4*dim) then mean? Or mean each then concat?
    # Protocol says "mean of tokens" for the vector.
    # Usual strategy: Concat last 4 hidden states at token level -> (seq, 4*dim), then mean over tokens.
    layers_last4 = [hidden_states[i][0] for i in [-1, -2, -3, -4]]
    cat_layers = torch.cat(layers_last4, dim=-1) # (seq, 4*dim)
    span_emb_last4 = cat_layers[start_idx:end_idx].mean(dim=0).cpu().numpy()
    
    return span_emb_pen, span_emb_last4

def main():
    logger.info(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    
    logger.info(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    logger.info("Updating DAPT embeddings...")
    
    updates = 0
    errors = 0
    
    dapt_pen_col = []
    dapt_last4_col = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['context_sentence']
        keyword = row['keyword']
        
        pen, last4 = get_embedding(model, tokenizer, text, keyword)
        
        if pen is not None:
            updates += 1
            dapt_pen_col.append(json.dumps(pen.tolist()))
            dapt_last4_col.append(json.dumps(last4.tolist()))
        else:
            errors += 1
            # Keep original if found? Or failing?
            # If we fail to find keyword (tokenizer diff?), we put empty list to crash/warn Phase 3?
            # Or use original? Original is Garbage.
            # We'll use empty list logic which Phase 3 might filter or error on.
            # Actually, let's look at `row['tokenizer_kw_ids']` if it exists.
            # To be safe, if we fail, we output []
            dapt_pen_col.append("[]")
            dapt_last4_col.append("[]")

    df['embedding_dapt_penultimate'] = dapt_pen_col
    df['embedding_dapt_last4_concat'] = dapt_last4_col
    
    # Update model metadata col
    df['model_dapt'] = "models/beto-adapted"
    
    logger.info(f"Saving to {OUTPUT_CSV}. Updates: {updates}, Errors/Misses: {errors}")
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()
