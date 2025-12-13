import argparse
import pandas as pd
import numpy as np
import os
import logging
from datetime import timedelta
import metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase3")

def load_data(occurrences_path, anchors_path):
    logger.info(f"Loading data from {occurrences_path}")
    df_occ = pd.read_parquet(occurrences_path)
    
    # Ensure datetime
    if "published_at" in df_occ.columns:
        df_occ["published_at"] = pd.to_datetime(df_occ["published_at"])
    
    # Load anchors
    if anchors_path and os.path.exists(anchors_path):
        df_anch = pd.read_parquet(anchors_path)
    else:
        df_anch = None
        
    return df_occ, df_anch

def prepare_anchors(df_anch):
    """
    Orthogonalizes anchors (Löwdin).
    Returns dict: dimension -> vector (center of dimension) OR raw orthogonalized vectors.
    """
    if df_anch is None: return None, None
    
    # Assume 1 anchor per dimension or multiple?
    # User plan implies "Frame Projection".
    # Strategy: Compute center of each dimension, then orthogonalize these centers.
    
    dims = df_anch["dimension"].unique()
    dim_vectors = []
    dim_names = []
    
    for d in dims:
        subset = df_anch[df_anch["dimension"] == d]
        # Get embeddings
        vecs = np.stack(subset["embedding_last4_concat"].values)
        # Mean vector
        center = np.mean(vecs, axis=0)
        dim_vectors.append(center)
        dim_names.append(d)
        
    matrix = np.stack(dim_vectors).T # (d, m)
    matrix_orth = metrics.lowdin_orthogonalization(matrix)
    
    return matrix_orth, dim_names

def run_pipeline(occurrences_path, anchors_path, output_dir, window_days=90, step_days=30):
    os.makedirs(output_dir, exist_ok=True)
    
    df, df_anchors = load_data(occurrences_path, anchors_path)
    frame_matrix, frame_names = prepare_anchors(df_anchors)
    
    # Sort by date
    df = df.sort_values("published_at")
    start_date = df["published_at"].min()
    end_date = df["published_at"].max()
    
    logger.info(f"Time range: {start_date} to {end_date}")
    
    # Results containers
    dim_stats = []
    drift_stats = []
    entropy_stats = []
    proj_stats = []
    
    prev_subspace = None
    
    current_date = start_date
    while current_date + timedelta(days=window_days) <= end_date:
        w_start = current_date
        w_end = current_date + timedelta(days=window_days)
        
        # Filter window
        mask = (df["published_at"] >= w_start) & (df["published_at"] < w_end)
        window_df = df[mask]
        
        n_samples = len(window_df)
        
        # Density check
        if n_samples < 50:
            logger.warning(f"Window {w_start.date()} skipped: density {n_samples} < 50")
            current_date += timedelta(days=step_days)
            continue
            
        # Matrix construction
        X = np.stack(window_df["embedding_last4_concat"].values)
        
        # Metric 1: Dimensionality (Horn)
        k_horn, _, _ = metrics.horn_parallel_analysis(X, n_iter=10)
        # Bootstrap stability? (Optional/Expensive, let's use Horn for now)
        k_selected = min(k_horn, 20) # Cap at 20 for interpretation sanity
        
        # Metric 2: Subspace
        U, S, _ = metrics.compute_subspace(X, k=k_selected)
        
        # Alignment & Drift
        drift_val = 0.0
        if prev_subspace is not None:
            # Align U to Prev
            # Usually we Procrustes U to U_prev, but standard drift is just distance
            # Procrustes is for temporal smoothness if we visualized trajectories
            # Here we just want the distance magnitude
            drift_val = metrics.grassmannian_distance(prev_subspace, U)
            
        prev_subspace = U
        
        # Metric 3: Entropy
        ent = metrics.semantic_entropy(S)
        
        # Metric 4: Projections
        if frame_matrix is not None:
             projs = metrics.project_on_frame(U, frame_matrix)
             for i, p_val in enumerate(projs):
                 proj_stats.append({
                     "window_start": w_start,
                     "dimension": frame_names[i],
                     "projection": p_val
                 })
        
        # Store stats
        dim_stats.append({
            "window_start": w_start,
            "window_end": w_end,
            "n_samples": n_samples,
            "k_horn": k_horn,
            "k_selected": k_selected
        })
        
        drift_stats.append({
            "window_start": w_start,
            "drift": drift_val
        })
        
        entropy_stats.append({
            "window_start": w_start,
            "entropy": ent
        })
        
        logger.info(f"Processed {w_start.date()}: N={n_samples}, k={k_selected}, Drift={drift_val:.4f}")
        
        # Step
        current_date += timedelta(days=step_days)

    # Save outputs
    pd.DataFrame(dim_stats).to_parquet(os.path.join(output_dir, "window_dimensionality.parquet"))
    pd.DataFrame(drift_stats).to_parquet(os.path.join(output_dir, "drift_timeseries.parquet"))
    pd.DataFrame(entropy_stats).to_parquet(os.path.join(output_dir, "entropy_timeseries.parquet"))
    if proj_stats:
        pd.DataFrame(proj_stats).to_parquet(os.path.join(output_dir, "frame_projections.parquet"))
        
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrences", required=True)
    parser.add_argument("--anchors")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    run_pipeline(args.occurrences, args.anchors, args.output)
