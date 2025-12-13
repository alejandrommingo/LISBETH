
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to find data if run from scripts/
DATA_PATH = 'data/embeddings_occurrences.parquet'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '../data/embeddings_occurrences.parquet'


def main():
    target_file = DATA_PATH
    if len(sys.argv) > 1:
        target_file = sys.argv[1]

    if not os.path.exists(target_file):
        print(f"ERROR: File not found at {target_file}")
        sys.exit(1)
        
    print(f"Loading {target_file}...")
    try:
        df = pd.read_parquet(target_file)
    except Exception as e:
        print(f"FAILED to load parquet: {e}")
        sys.exit(1)

    
    print(f"Shape: {df.shape}")
    print("\nColumns found:")
    print(df.columns.tolist())
    
    # 2.2 A) Identificación
    req_cols_a = ['occurrence_id', 'run_id', 'model_id', 'model_variant', 'layer_strategy', 'pooling_strategy']
    missing_a = [c for c in req_cols_a if c not in df.columns]
    
    # 2.2 B) Información temporal
    req_cols_b = ['published_at', 'year', 'month', 'year_month']
    missing_b = [c for c in req_cols_b if c not in df.columns]

    # 2.2 C) Fuente
    req_cols_c = ['newspaper', 'source_api', 'url']
    missing_c = [c for c in req_cols_c if c not in df.columns]

    # 2.2 D) Lingüística
    req_cols_d = ['keyword_canonical', 'keyword_matched', 'char_start', 'char_end', 'token_start', 'token_end']
    missing_d = [c for c in req_cols_d if c not in df.columns]

    # 2.2 E) Contexto
    req_cols_e = ['context_sentence']
    missing_e = [c for c in req_cols_e if c not in df.columns]

    # 2.2 F) Embeddings
    req_cols_f = ['embedding_contextual_last4', 'embedding_contextual_penultimate']
    missing_f = [c for c in req_cols_f if c not in df.columns]

    print("\n--- SCHEMA CHECK ---")
    if missing_a: print(f"❌ Missing A: {missing_a}")
    else: print("✅ A OK")
    if missing_b: print(f"❌ Missing B: {missing_b}")
    else: print("✅ B OK")
    if missing_c: print(f"❌ Missing C: {missing_c}")
    else: print("✅ C OK")
    if missing_d: print(f"❌ Missing D: {missing_d}")
    else: print("✅ D OK")
    if missing_e: print(f"❌ Missing E: {missing_e}")
    else: print("✅ E OK")
    if missing_f: print(f"❌ Missing F: {missing_f}")
    else: print("✅ F OK")

    print("\n--- NULL CHECK ---")
    all_req = req_cols_a + req_cols_b + req_cols_c + req_cols_d + req_cols_e + req_cols_f
    existing_req = [c for c in all_req if c in df.columns]
    nulls = df[existing_req].isnull().sum()
    if nulls.sum() > 0:
        print("⚠️ NULLS FOUND:")
        print(nulls[nulls > 0])
    else:
        print("✅ No nulls in required columns")

    print("\n--- CONTENT SUMMARY ---")
    if 'model_variant' in df.columns:
        print(f"Model Variants: {df['model_variant'].unique()}")
    if 'year_month' in df.columns:
        print(f"Months: {df['year_month'].nunique()} ({df['year_month'].min()} to {df['year_month'].max()})")
        
    # Check consistency of embeddings
    print("\n--- EMBEDDING CHECK ---")
    for col in ['embedding_contextual_last4', 'embedding_contextual_penultimate']:
        if col in df.columns:
            # Check length of first element
            first = df[col].iloc[0]
            print(f"{col}: type={type(first)}, len={len(first) if hasattr(first, '__len__') else 'scalar'}")

if __name__ == "__main__":
    main()
