
import os
import glob
import pandas as pd
import numpy as np
from src.analysis.temporal import TemporalSegmenter
from src.analysis.subspaces import SubspaceConstructor
from src.analysis.metrics import SociologicalMetrics
from src.analysis.dimensionality import SubspaceAnalyzer

def run_pipeline():
    print(">>> Phase 3 Pipeline: Subspace Construction & Metrics Calculation <<<")
    
    # 1. LOAD DATA
    print("[1/5] Loading Data...")
    dfs = []
    # Search for parquets in data/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(base_dir, 'data', 'embeddings_*.parquet')
    files = [f for f in glob.glob(pattern) if 'anchors' not in f and 'results' not in f]
    
    if not files:
        print("Error: No embedding files found.")
        return

    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Warning: Failed to read {f}: {e}")
            
    df_full = pd.concat(dfs, ignore_index=True)
    
    # 2. DEDUPLICATE
    print(f"  Raw rows: {len(df_full)}")
    df_full['hash'] = df_full['embedding'].apply(lambda x: hash(x.tobytes()))
    df_full = df_full.drop_duplicates(subset=['hash']).drop('hash', axis=1)
    print(f"  Unique embeddings: {len(df_full)}")
    
    # Ensure date format
    if 'date' not in df_full.columns:
        print("Error: 'date' column missing.")
        return
    df_full['date'] = pd.to_datetime(df_full['date'])
    
    # 3. SEGMENTATION
    print("[2/5] Temporal Segmentation (3-month Rolling Windows)...")
    segmenter = TemporalSegmenter(df_full, date_column='date')
    windows = list(segmenter.generate_windows(window_months=3, step_months=1, min_count=50))
    print(f"  Generated {len(windows)} valid windows.")
    
    if not windows:
        print("Error: No valid windows.")
        return
        
    # 4. SUBSPACE CONSTRUCTION AND ANALYSIS
    print("[3/5] Subspace Construction & Dimensionality Analysis...")
    constructor = SubspaceConstructor()
    analyzer = SubspaceAnalyzer(random_state=42)
    
    # Calculate K evolution
    k_evolution = []
    for w in windows:
        mat = np.vstack(w['data']['embedding'].values)
        if mat.shape[0] > 30:
             k, _, _ = analyzer.horns_parallel_analysis(mat, num_simulations=5)
             k_evolution.append(k)
        else:
             k_evolution.append(0) # Not enough data
             
    # Build subspaces using dynamic K (or fixed K? Let's use max K found or median?)
    # For temporal alignment (Procrustes), it is statistically cleaner to keep K fixed across time.
    # Otherwise alignment between 3D and 4D spaces is mathematically ambiguous (padding with zeros?).
    # We will use the Median K from the evolution as the fixed dimension for alignment.
    valid_ks = [k for k in k_evolution if k > 0]
    median_k = int(np.median(valid_ks)) if valid_ks else 3
    print(f"  Median Intrinsic Dimension detected: k={median_k}. Using this for alignment.")
    
    subspaces = constructor.build_subspaces(windows, fixed_k=median_k, align=True)
    print(f"  Constructed {len(subspaces)} aligned subspaces.")
    
    # 5. METRICS CALCULATION
    print("[4/5] Computing Sociological Metrics (Drift, Entropy, Projections)...")
    metrics_calc = SociologicalMetrics()
    
    # A. Sequential Drift (t vs t-1)
    drift_df = metrics_calc.calculate_drift(subspaces)
    
    # B. Entropy
    entropy_df = metrics_calc.calculate_entropy(subspaces)
    
    # C. Centroid Drift & Eigenvalues Data
    centroid_dists = []
    drift_dates = []
    
    # Store eigenvalues for reporting/plotting variance later
    eigen_data_list = []
    
    for i, s in enumerate(subspaces):
        # Collect eigenvalues
        if s.eigenvalues is not None:
             # Convert to list for serialization
             ev_list = s.eigenvalues.tolist() if isinstance(s.eigenvalues, np.ndarray) else list(s.eigenvalues)
             eigen_data_list.append(ev_list)
        else:
             eigen_data_list.append([])

    # Shifted loop for drift
    for i in range(1, len(subspaces)):
        c_prev = subspaces[i-1].centroid
        c_curr = subspaces[i].centroid
        dist = np.linalg.norm(c_curr - c_prev)
        centroid_dists.append(dist)
        drift_dates.append(subspaces[i].label)
    
    centroid_df = pd.DataFrame({'date': drift_dates, 'centroid_drift': centroid_dists})

    # D. Similarity Matrix (All-vs-All) for Heatmap
    print("    Computing Window-Window Similarity Matrix...")
    n_wins = len(subspaces)
    sim_matrix = np.zeros((n_wins, n_wins))
    labels = [s.label for s in subspaces]
    
    for i in range(n_wins):
        for j in range(n_wins):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                # Cosine similarity of flattened basis? Or Principal Angles?
                # Metrics.calculate_drift uses trace(A' B). Let's use that.
                # Assuming orthogonal bases:
                # Sim = Trace(U_i.T @ U_j) / k  ? No, assumes aligned.
                # Let's use simple subspace projection similarity:
                # Sim = || U_i^T U_j ||_F^2 / k
                term = np.dot(subspaces[i].basis, subspaces[j].basis.T) # shape (k, k)
                sim = np.linalg.norm(term)**2 / subspaces[i].basis.shape[0]
                sim_matrix[i, j] = sim
                
    sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    sim_df.to_csv(os.path.join(base_dir, 'data', 'phase3_sim_matrix.csv'))

    # Projections
    anchors_path = os.path.join(base_dir, 'data', 'anchors_embeddings.parquet')
    if os.path.exists(anchors_path):
        anchors_df = pd.read_parquet(anchors_path)
        print("  Anchors loaded. Computing Orthogonal Projections...")
        # Use new orthogonalization logic
        proj_df = metrics_calc.calculate_projections(subspaces, anchors_df, orthogonalize=True)
    else:
        print("Warning: Anchors file not found. Skipping projections.")
        proj_df = pd.DataFrame({'date': [s.label for s in subspaces]})
        
    # 6. SAVE RESULTS
    print("[5/5] Saving Results...")
    
    # Merge all metrics into one master DataFrame
    # Date is the key.
    # Drift starts at index 1 (t vs t-1).
    
    dates = [s.label for s in subspaces]
    results_master = pd.DataFrame({'date': dates})
    
    # Merge
    results_master = results_master.merge(drift_df, on='date', how='left')
    results_master = results_master.merge(entropy_df, on='date', how='left')
    results_master = results_master.merge(centroid_df, on='date', how='left')
    results_master = results_master.merge(proj_df, on='date', how='left')
    
    # Add K stats
    # Align lengths
    if len(k_evolution) == len(results_master):
        results_master['intrinsic_dimension_k'] = k_evolution
        
    # Add Eigenvalues (Serialized as string to avoid schema issues with variable length lists)
    results_master['eigenvalues'] = [str(x) for x in eigen_data_list] # Store as string representation of list
        
    output_path = os.path.join(base_dir, 'data', 'phase3_results.parquet')
    results_master.to_parquet(output_path)
    print(f"  Success! Results saved to: {output_path}")
    print(results_master.head())

if __name__ == "__main__":
    run_pipeline()
