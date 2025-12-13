
import argparse
import sys
import os
import pandas as pd
import numpy as np
import logging
from src.phase3.data_loader import Phase3DataLoader
from src.phase3.windowing import RollingWindowSegmenter
from src.phase3.dimensionality import DimensionalitySelector
from src.phase3.subspace import SubspaceConstructor
from src.phase3.metrics import SociologicalMetrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Semantic Subspace Analysis Pipeline")
    parser.add_argument("--data_path", required=True, help="Path to embeddings_occurrences.parquet")
    parser.add_argument("--anchors_path", required=True, help="Path to anchors parquet")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--suffix", default="", help="Suffix for output files")
    
    args = parser.parse_args()
    
    # 1. Load Data
    loader = Phase3DataLoader(args.data_path)
    df_merged = loader.load_occurrences()
    
    if df_merged.empty:
        logger.error("No data available. Exiting.")
        sys.exit(1)
        
    # 2. Windowing
    segmenter = RollingWindowSegmenter(window_months=3, step_months=1, min_count=50)
    windows = list(segmenter.get_windows(df_merged))
    logger.info(f"Generated {len(windows)} valid windows.")
    
    if not windows:
        logger.error("No valid windows found. Exiting.")
        sys.exit(1)

    # 3. Dimensionality Analysis (First Pass)
    logger.info("Running Dimensionality Analysis (Horn's PA)...")
    dim_selector = DimensionalitySelector()
    
    k_stats = []
    
    for w in windows:
        # Construct matrix
        # 'embedding' column contains numpy arrays or bytes
        # loader should have ensured they are usable. 
        # If they are bytes, we need to decode.
        
        # Check first element
        first_emb = w['data'].iloc[0]['embedding']
        if isinstance(first_emb, bytes):
            # Assume float32? need to know shape or dtype.
            # Usually ndarray in parquet.
            pass # Parquet read usually handles it if properly saved.
            
        # Stack
        matrix = np.stack(w['data']['embedding'].values)
        
        k_opt, _, _ = dim_selector.select_k_horns(matrix)
        stability = dim_selector.check_stability_bootstrap(matrix, k_opt)
        
        k_stats.append({
            'window_start': w['start'],
            'window_end': w['end'],
            'count': w['count'],
            'k_optimal': k_opt,
            'stability': stability
        })
        
    df_k = pd.DataFrame(k_stats)
    
    # Decide Consensus K for Alignment
    # Median of K optimal
    median_k = int(df_k['k_optimal'].median())
    logger.info(f"Median Intrinsic Dimension: {median_k}. Using for alignment.")
    
    # 4. Subspace Construction & Alignment (Second Pass)
    logger.info(f"Constructing Subspaces (Fixed K={median_k}) and Aligning...")
    constructor = SubspaceConstructor(fixed_k=median_k)
    
    subspaces = []
    aligned_bases = [] # For saving
    
    # Initialize previous subspace for alignment
    prev_basis = None
    
    for w in windows:
        matrix = np.stack(w['data']['embedding'].values)
        basis, evals = constructor.build(matrix)
        
        # Align
        if prev_basis is not None:
             basis_aligned, R, error = constructor.align(prev_basis, basis)
        else:
             basis_aligned = basis
             error = 0.0
             
        subspaces.append({
            'window_start': w['start'],
            'basis': basis_aligned,
            'eigenvalues': evals,
            'alignment_error': error
        })
        
        prev_basis = basis_aligned
        aligned_bases.append(basis_aligned)

    # 5. Metrics
    logger.info("Computing Sociological Metrics...")
    metrics_calc = SociologicalMetrics()
    
    # Load Anchors
    if os.path.exists(args.anchors_path):
        anchors_df = pd.read_parquet(args.anchors_path)
    else:
        logger.warning(f"Anchors file not found: {args.anchors_path}")
        anchors_df = pd.DataFrame()

    results_list = []
    
    for i, s in enumerate(subspaces):
        # Entropy
        entropy = metrics_calc.calculate_entropy(s['eigenvalues'])
        
        # Drift (vs previous)
        if i > 0:
            drift = metrics_calc.calculate_drift(subspaces[i-1]['basis'], s['basis'])
        else:
            drift = 0.0 # Start
            
        # Frame Projections
        projections = metrics_calc.calculate_frame_projection(s['basis'], anchors_df)
        
        row = {
            'window_start': s['window_start'],
            'entropy': entropy,
            'drift': drift,
            'alignment_error': s['alignment_error']
        }
        # Flatten projections
        for dim, val in projections.items():
            row[f'proj_{dim}'] = val
            
        results_list.append(row)
        
    df_metrics = pd.DataFrame(results_list)
    
    # 6. Save Outputs
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = args.suffix
    
    # Dimensionality
    df_k.to_parquet(os.path.join(args.output_dir, f'window_dimensionality{suffix}.parquet'))
    
    # Metrics (Drift, Entropy, Projections)
    # Split them? Or Master? 
    # Report asked for separate files but master is easier to verify.
    # Let's write the requested separated files + master.
    
    # Drift
    df_metrics[['window_start', 'drift']].to_parquet(os.path.join(args.output_dir, f'semantic_drift_timeseries{suffix}.parquet'))
    
    # Entropy
    df_metrics[['window_start', 'entropy']].to_parquet(os.path.join(args.output_dir, f'semantic_entropy_timeseries{suffix}.parquet'))
    
    # Projections
    # Extract projection cols
    proj_cols = [c for c in df_metrics.columns if c.startswith('proj_')]
    if proj_cols:
        df_metrics[['window_start'] + proj_cols].to_parquet(os.path.join(args.output_dir, f'frame_projections{suffix}.parquet'))
    
    # Subspaces (NPZ)
    np.savez(
        os.path.join(args.output_dir, f'aligned_subspaces{suffix}.npz'), 
        bases=np.array(aligned_bases),
        dates=[pd.to_datetime(s['window_start']).isoformat() for s in subspaces]
    )
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
