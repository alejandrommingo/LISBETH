import json
import argparse
import pandas as pd
import os
import sys
from src.nlp.model import LisbethModel

def build_anchors(json_path, output_path, model_name="xlm-roberta-base"):
    """
    Reads a JSON file with dimensions and sentences, extracts embeddings for keywords
    in those specific contexts, and saves to Parquet.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
        
    print(f"Loading anchors from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Initializing model: {model_name}...")
    model = LisbethModel(model_name=model_name)
    
    results = []
    
    for dimension, content in data.items():
        description = content.get("description", "")
        anchors = content.get("anchors", [])
        
        print(f"Processing dimension '{dimension}' ({len(anchors)} anchors)...")
        
        for item in anchors:
            keyword = item["keyword"]
            sentence = item["sentence"]
            
            # Extract embedding of 'keyword' from 'sentence'
            # Returns a list of tensors (one per occurrence). Expecting 1 usually.
            embeddings = model.extract_embedding(sentence, keyword)
            
            if embeddings:
                # We take the mean if multiple occurrences (rare in these short sentences)
                # or just the first one. Let's take the first for precision.
                vector_contextual = embeddings[0]
                
                # Hybrid Strategy: Also get the static (Layer 0) embedding
                vector_static = model.get_static_embedding(keyword)
                
                results.append({
                    "dimension": dimension,
                    "keyword": keyword,
                    "sentence": sentence,
                    "embedding_contextual": vector_contextual,
                    "embedding_static": vector_static,
                    "description": description
                })
            else:
                print(f"Warning: Could not extract '{keyword}' from '{sentence}'")

    if not results:
        print("No embeddings extracted.")
        return

    df = pd.DataFrame(results)
    print(f"Extracted {len(df)} anchor embeddings.")
    
    # Save to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Contextual Anchors from JSON")
    parser.add_argument("--json", required=True, help="Path to dimensiones_ancla.json")
    parser.add_argument("--output", required=True, help="Output path for parquet file")
    parser.add_argument("--model", default="xlm-roberta-base", help="Model name")
    
    args = parser.parse_args()
    
    # Add project root to path if needed (rudimentary fix)
    sys.path.append(os.getcwd())
    
    build_anchors(args.json, args.output, args.model)
