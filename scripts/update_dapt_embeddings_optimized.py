import pandas as pd
import numpy as np
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_CSV = "data/phase2/embeddings_occurrences.csv"
OUTPUT_CSV = "data/phase2/embeddings_occurrences.csv"
MODEL_PATH = "models/beto-adapted"

class OccurrencesDataset(Dataset):
    def __init__(self, df):
        self.sentences = df['context_sentence'].tolist()
        self.keywords = df['keyword'].tolist()
        self.starts = df['char_start'].tolist()
        self.ends = df['char_end'].tolist()
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        return self.sentences[idx], self.keywords[idx], self.starts[idx], self.ends[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH).to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    model.eval()
    
    logger.info(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    dataset = OccurrencesDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_pen = []
    all_last4 = []
    
    logger.info("Updating DAPT embeddings (Optimized with Offsets)...")
    
    for batch_text, batch_kw, batch_start, batch_end in tqdm(dataloader):
        # Tokenize batch with offsets
        inputs = tokenizer(list(batch_text), return_tensors="pt", padding=True, truncation=True, max_length=512, return_offsets_mapping=True).to(device)
        
        # Move inputs to device (excluding offset_mapping which is not a tensor for model)
        model_inputs = {k: v for k, v in inputs.items() if k != 'offset_mapping'}
        
        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        pen_layer = hidden_states[-2]
        layers_last4 = [hidden_states[i] for i in [-1, -2, -3, -4]]
        cat_layers = torch.cat(layers_last4, dim=-1)
        
        # Iterate batch
        batch_size = len(batch_text)
        offset_mapping = inputs['offset_mapping'].cpu().numpy() # (batch, seq, 2)
        
        for i in range(batch_size):
            # Find keyword in sentence dynamically (more robust than global char_start)
            user_sentence = batch_text[i]
            user_kw = batch_kw[i]
            
            # Case insensitive search
            s_lower = user_sentence.lower()
            k_lower = user_kw.lower()
            
            start_idx_char = s_lower.find(k_lower)
            
            if start_idx_char == -1:
                 # Fallback: exact match failed? Try finding as token logic?
                 # If we can't find the string, we can't extract.
                 all_pen.append("[]")
                 all_last4.append("[]")
                 continue
                 
            end_idx_char = start_idx_char + len(user_kw)
            
            offsets = offset_mapping[i]
            
            token_indices = []
            for t_idx, (o_start, o_end) in enumerate(offsets):
                if o_start == 0 and o_end == 0: continue
                # Overlap with found char range
                if max(start_idx_char, o_start) < min(end_idx_char, o_end):
                    token_indices.append(t_idx)

            if token_indices:
                start_t = token_indices[0]
                end_t = token_indices[-1] + 1
                
                emb_pen = pen_layer[i, start_t:end_t, :].mean(dim=0).cpu().numpy()
                emb_last4 = cat_layers[i, start_t:end_t, :].mean(dim=0).cpu().numpy()
                
                all_pen.append(json.dumps(emb_pen.tolist()))
                all_last4.append(json.dumps(emb_last4.tolist()))
            else:
                all_pen.append("[]")
                all_last4.append("[]")

    df['embedding_dapt_penultimate'] = all_pen
    df['embedding_dapt_last4_concat'] = all_last4
    df['model_dapt'] = "models/beto-adapted"
    
    logger.info(f"Saving to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    main()
