import os
import glob
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from src.analysis.temporal import TemporalSegmenter
from src.analysis.dimensionality import SubspaceAnalyzer

def audit_data_integrity():
    print("\n[1. Data Integrity Audit]")
    # Use robust path lookup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    csv_files = glob.glob(os.path.join(data_dir, 'yape_*.csv'))
    
    print(f"Searching in {data_dir}")
    print(f"Found {len(csv_files)} CSV source files.")
    
    total_rows = 0
    total_dupes = 0
    
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f)
            n_rows = len(df)
            n_dupes = 0
            if 'text' in df.columns:
                n_dupes = df.duplicated(subset=['text']).sum()
            elif 'cuerpo' in df.columns:
                n_dupes = df.duplicated(subset=['cuerpo']).sum()
            
            print(f"  {os.path.basename(f)}: Rows={n_rows}, Duplicates={n_dupes} ({n_dupes/n_rows:.1%})")
            total_rows += n_rows
            total_dupes += n_dupes
        except Exception as e:
            print(f"  Error reading {f}: {e}")

def audit_tokenization_and_model():
    print("\n[2. Model & Tokenization Audit]")
    model_name = "PlanTL-GOB-ES/roberta-large-bne" 
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        word = "Yape"
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.encode(word, add_special_tokens=False)
        print(f"  Tokenization of '{word}': {tokens} (IDs: {ids})")
        
        if len(ids) > 1:
            print("  WARNING: 'Yape' is split into multiple subwords. Embedding averaging is happening.")
            
        # Check static embedding generation
        inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # Mocking logic from current model.py (concat last 4)
        hidden_states = outputs.hidden_states
        selected = hidden_states[-4:]
        concat = torch.cat(selected, dim=-1)
        vector = torch.mean(concat[0], dim=0).numpy()
        
        print(f"  Generated Static Embedding Shape: {vector.shape}")
        return vector
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None

def audit_embeddings_validity(static_yape_vector):
    print("\n[3. Embedding Validity Audit]")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(base_dir, 'data', 'embeddings_test_cli.parquet')
    parquet_files = glob.glob(pattern)
    
    if not parquet_files:
        print("  No embedding parquets found.")
        return

    for p in parquet_files:
        try:
            df = pd.read_parquet(p)
            print(f"  Checking {os.path.basename(p)} ({len(df)} rows)")
            
            sample = df.iloc[0]['embedding']
            print(f"  Vector Dimension: {len(sample)}")
            
            if static_yape_vector is not None:
                # Check similarity of random sample vs static yape
                sample_vecs = df['embedding'].sample(min(100, len(df)), random_state=42).values
                sample_matrix = np.vstack(sample_vecs)
                sims = cosine_similarity(sample_matrix, static_yape_vector.reshape(1, -1))
                avg_sim = np.mean(sims)
                print(f"  Avg Similarity of sample vs Static 'Yape': {avg_sim:.4f}")
                if avg_sim < 0.3:
                     print("  CRITICAL WARNING: Low similarity. The embeddings might not represent 'Yape'.")
                else:
                     print("  Validity Check: PASS (Vectors align with target word)")
                     
        except Exception as e:
            print(f"  Error reading {p}: {e}")

def audit_anchors():
    print("\n[4. Anchor Orthogonality & Hybrid Check]")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, 'data', 'anchors_embeddings.parquet')

    if not os.path.exists(path):
        print("  Anchors file not found.")
        return
        
    df = pd.read_parquet(path)
    dimensions = df['dimension'].unique()
    print(f"  Dimensions found: {dimensions}")
    
    # Calculate Centroids
    centroids = {}
    for dim in dimensions:
        data = df[df['dimension'] == dim]
        
        # Static
        if 'embedding_static' in data.columns:
            vecs_s = np.vstack(data['embedding_static'].values)
            cent_s = np.mean(vecs_s, axis=0)
            centroids[f"{dim}_static"] = cent_s / np.linalg.norm(cent_s)
            
        # Contextual
        if 'embedding_contextual' in data.columns:
            vecs_c = np.vstack(data['embedding_contextual'].values)
            cent_c = np.mean(vecs_c, axis=0)
            centroids[f"{dim}_contextual"] = cent_c / np.linalg.norm(cent_c)
            
    # Check Correlations (Contextual)
    print("  --- Cross-Correlation (Contextual Centroids) ---")
    keys = [k for k in centroids.keys() if 'contextual' in k]
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            k1, k2 = keys[i], keys[j]
            sim = np.dot(centroids[k1], centroids[k2])
            print(f"  Sim({k1}, {k2}): {sim:.4f}")
            if sim > 0.85:
                print("    WARNING: High overlap between theoretical dimensions. Concepts aren't distinct.")
                
    # Check Hybrid Efficacy
    print("  --- Hybrid Efficacy (Static vs Contextual) ---")
    for dim in dimensions:
        k_s = f"{dim}_static"
        k_c = f"{dim}_contextual"
        if k_s in centroids and k_c in centroids:
            sim = np.dot(centroids[k_s], centroids[k_c])
            print(f"  Sim({dim}_static, {dim}_contextual): {sim:.4f}")
            if sim > 0.98:
                 print("    WARNING: Context adds almost no information.")

def audit_temporal_sensitivity():
    print("\n[5. Temporal Window Sensitivity]")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob(os.path.join(base_dir, 'data', 'embeddings_*.parquet'))
    files = [f for f in files if 'anchors' not in f]
    
    if not files:
        print("  No embedding files found for sensitivity analysis.")
        return
        
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(pd.read_parquet(f))
        except: pass
        
    if not dfs: return
    
    df_full = pd.concat(dfs, ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'])
    print(f"  Total Data Points: {len(df_full)}")
    
    segmenter = TemporalSegmenter(df_full, date_column='date')
    
    configs = [
        (1, 1, 50), # 1m size, 1m step
        (3, 1, 50), # 3m size, 1m step
    ]
    
    for w_size, w_step, min_c in configs:
        windows = list(segmenter.generate_windows(window_months=w_size, step_months=w_step, min_count=min_c))
        print(f"  Config ({w_size}m size, {w_step}m step, min={min_c}): Generated {len(windows)} valid windows.")
        densities = [w['count'] for w in windows]
        if densities:
            print(f"    Avg Density: {np.mean(densities):.1f}, Min: {np.min(densities)}, Max: {np.max(densities)}")
        else:
            print("    No valid windows found.")

if __name__ == "__main__":
    yape_vec = audit_tokenization_and_model()
    audit_data_integrity()
    audit_embeddings_validity(yape_vec)
    audit_anchors()
    audit_temporal_sensitivity()
