
import pandas as pd
import numpy as np
import shutil
import os

def adapt_legacy_data(input_path, output_path, is_anchors=False):
    print(f"Adapting {input_path} -> {output_path}")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} missing.")
        return

    df = pd.read_parquet(input_path)
    
    if is_anchors:
        # Check columns
        if 'embedding_contextual' not in df.columns:
            print("Mapping 'embedding' -> 'embedding_contextual' & 'embedding_static'")
            # Rename if exists, or copy
            if 'embedding' in df.columns:
                df['embedding_contextual'] = df['embedding']
                df['embedding_static'] = df['embedding'] # Hack: Use same for static
            elif 'vector' in df.columns:
                 df['embedding_contextual'] = df['vector']
                 df['embedding_static'] = df['vector']
    else:
        # Corpus data
        if 'embedding_static' not in df.columns and 'embedding' in df.columns:
             print("Duplicating 'embedding' -> 'embedding_static'")
             df['embedding_static'] = df['embedding']
             
    df.to_parquet(output_path)
    print("Saved.")

if __name__ == "__main__":
    # Adapt Spanish Corpus
    adapt_legacy_data('data/embeddings_v2.parquet', 'data/embeddings_spanish_dual.parquet', is_anchors=False)
    
    # Adapt Spanish Anchors (Legacy)
    adapt_legacy_data('data/anchors_embeddings.parquet', 'data/anchors_spanish.parquet', is_anchors=True)
