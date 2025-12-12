import argparse
import pandas as pd
import torch
import glob
import os
from src.nlp.model import LisbethModel
from tqdm import tqdm

def extract_embeddings(data_dir, output_file, keywords=None, model_name="PlanTL-GOB-ES/roberta-large-bne", layers=4):
    if keywords is None:
        keywords = ["Yape", "Yapear", "Yapame", "Yapeo", "Plin"]
        
    print(f"Initializing model: {model_name}")
    try:
        model = LisbethModel(model_name=model_name)
    except Exception as e:
        print(f"Primary model failed ({e}). Falling back to Base for demo if needed.")
        model = LisbethModel(model_name="PlanTL-GOB-ES/roberta-base-bne")

    csv_files = glob.glob(os.path.join(data_dir, "yape_*.csv"))
    results = []
    
    print(f"Processing {len(csv_files)} files for keywords: {keywords}...")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if "plain_text" not in df.columns:
                continue
                
            texts = df["plain_text"].dropna().astype(str).tolist()
            dates = df["published_at"].tolist() if "published_at" in df.columns else [None]*len(texts)
            medias = df["newspaper"].tolist() if "newspaper" in df.columns else [None]*len(texts)
            
            for text, date, media in tqdm(zip(texts, dates, medias), total=len(texts), desc=os.path.basename(file)):
                for word in keywords:
                    # Try both original casing and lowercase to maximize recall
                    variants = set([word, word.lower(), word.capitalize()])
                    
                    for variant in variants:
                        # Extract embeddings for EACH variant using DUAL strategy
                        embeddings = model.extract_dual_embedding(text, variant)
                        
                        if embeddings:
                            for res in embeddings:
                                results.append({
                                    "date": date,
                                    "media": media,
                                    "keyword": word, # Normalize to the canonical keyword
                                    "embedding": res['contextual'].tolist(), # Main dynamic embedding
                                    "embedding_static": res['static'].tolist(), # For projection
                                    "original_word": variant,
                                    "context": text[:200] + "..."
                                })
                        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    # Save to Parquet
    if results:
        df_out = pd.DataFrame(results)
        print(f"Saving {len(df_out)} embeddings to {output_file}")
        df_out.to_parquet(output_file, index=False)
    else:
        print("No embeddings found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output", default="data/embeddings_yape.parquet")
    parser.add_argument("--keywords", nargs="+", default=["Yape", "Yapear", "Yapame", "Yapeo", "Plin"], help="List of keywords to extract")
    parser.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne")
    args = parser.parse_args()
    
    extract_embeddings(args.data_dir, args.output, args.keywords, args.model)
