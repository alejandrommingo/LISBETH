import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

def load_embeddings(series):
    return np.stack(series.apply(json.loads).values)

def run_eda(file_path):
    print(f"--- EXPERT EDA REPORT: {file_path} ---\n")
    
    # 1. LOAD & INTEGRITY
    if not os.path.exists(file_path):
        print("CRITICAL: File not found.")
        return

    df = pd.read_csv(file_path)
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Check IDs uniqueness
    n_unique_ids = df['occurrence_id'].nunique()
    print(f"Unique IDs: {n_unique_ids} (All unique: {n_unique_ids == len(df)})")
    
    # Check Nulls
    nulls = df.isnull().sum()
    if nulls.any():
         print("\nWARNING: Null values detected:")
         print(nulls[nulls > 0])
    else:
         print("Data Integrity: No null values found.")

    # 2. DISTRIBUTIONS
    print("\n--- DISTRIBUTIONS ---")
    
    # Keywords
    print("\nTop found keywords:")
    print(df['keyword_found'].value_counts().head(10))
    
    # Newspapers
    print("\nTop Newspapers:")
    print(df['newspaper'].value_counts().head(5))
    
    # Temporal
    df['dt'] = pd.to_datetime(df['published_at'], errors='coerce')
    df['year_month_date'] = df['dt'].dt.to_period('M')
    print("\nMonthly coverage sample (Head/Tail):")
    counts = df.groupby('year_month_date').size()
    print(counts.head(3))
    print("...")
    print(counts.tail(3))
    
    # 3. EMBEDDING ANALYSIS
    print("\n--- EMBEDDING ANALYSIS ---")
    
    # Load first vectors to check dims
    try:
        # Sample for speed if large
        sample_df = df.sample(min(len(df), 500), random_state=42)
        
        emb_base = load_embeddings(sample_df['embedding_baseline_penultimate'])
        emb_dapt = load_embeddings(sample_df['embedding_dapt_penultimate'])
        
        print(f"Baseline Penultimate Shape: {emb_base.shape}")
        print(f"DAPT Penultimate Shape: {emb_dapt.shape}")
        
        # Check for Collapse (Std Dev of norms)
        norms_base = np.linalg.norm(emb_base, axis=1)
        norms_dapt = np.linalg.norm(emb_dapt, axis=1)
        
        print(f"Baseline Mean Norm: {norms_base.mean():.4f} (std: {norms_base.std():.4f})")
        print(f"DAPT Mean Norm:     {norms_dapt.mean():.4f} (std: {norms_dapt.std():.4f})")
        
        if norms_base.std() < 0.01:
            print("WARNING: Baseline embeddings look collapsed (variance almost zero).")
            
        # DAPT Impact Analysis (Cosine Sim between Baseline and DAPT for same row)
        # We need to reshape for pairwise, but here we want row-wise.
        # Dot product / (norm * norm)
        
        # efficient row-wise cosine sim
        dot_product = np.sum(emb_base * emb_dapt, axis=1)
        sims = dot_product / (norms_base * norms_dapt)
        
        print(f"\nMean Cosine Similarity (Baseline vs DAPT): {sims.mean():.4f}")
        print(f"Min Sim: {sims.min():.4f}, Max Sim: {sims.max():.4f}")
        
        if sims.mean() > 0.99:
             print("ALERT: DAPT model is producing nearly identical embeddings to Baseline. Did loading fail silently or is domain adaptation minimal?")
        elif sims.mean() < 0.90:
             print("SUCCESS: DAPT has noticeably shifted the embedding space.")
        else:
             print("NOTE: Moderate shift observed.")
             
    except Exception as e:
        print(f"ERROR analysing embeddings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_eda("data/phase2/embeddings_occurrences.csv")
