import pandas as pd
import json
import logging
from typing import List, Dict, Any
from src.phase3.schemas import Phase3Config, Phase3RunContext

logger = logging.getLogger(__name__)

class Agent11Assembler:
    """
    Agente 11 — ENSAMBLADOR FINAL (CSV + manifests)
    """
    
    def run(self, context: Phase3RunContext, results_buffer: List[Dict]):
        logger.info("Agent 11: Assembling final results...")
        
        # 1. Build Phase 3 Results DataFrame
        df_results = pd.DataFrame(results_buffer)
        
        # 2. Calculate Deltas (DAPT - Baseline)
        self._calculate_deltas(df_results)
        
        # 3. Save phase3_results.csv
        out_csv = Phase3Config.OUTPUT_CSV
        df_results.to_csv(out_csv, index=False)
        logger.info(f"Saved main results to {out_csv}")
        
        # 4. Generate Subspaces Index
        subspaces_data = []
        for idx, row in df_results.iterrows():
            win_start = row['window_start_month']
            win_end = row['window_end_month']
            
            for v in Phase3Config.VARIANTS:
                for s in Phase3Config.STRATEGIES:
                    combo = f"{v}_{s}"
                    # Check columns like k_<v>_<s>
                    if f"subspace_path_{v}_{s}" in row:
                        path_val = row[f"subspace_path_{v}_{s}"]
                        k_val = row[f"k_{v}_{s}"]
                        subspaces_data.append({
                            "window_start_month": win_start,
                            "window_end_month": win_end,
                            "variant": v,
                            "strategy": s,
                            "k_selected": k_val,
                            "subspace_path": path_val
                        })
                        
        df_index = pd.DataFrame(subspaces_data)
        index_path = Phase3Config.MANIFESTS_DIR / "subspaces_index.csv"
        df_index.to_csv(index_path, index=False)
        
        # 5. Generate Run Manifest
        manifest = {
            "timestamp": context.run_timestamp,
            "global_parameters": {
                "WINDOW_MONTHS": Phase3Config.WINDOW_MONTHS,
                "STEP_MONTHS": Phase3Config.STEP_MONTHS,
                "N_MIN_OCCURRENCES": Phase3Config.N_MIN_OCCURRENCES,
                "LOW_DENSITY_FLAG": Phase3Config.LOW_DENSITY_FLAG,
                "VARIANTS": Phase3Config.VARIANTS,
                "STRATEGIES": Phase3Config.STRATEGIES,
                "DIMENSIONS": Phase3Config.DIMENSIONS,
                "SEED": Phase3Config.SEED,
                "B_HORN": Phase3Config.B_HORN,
                "B_BOOT": Phase3Config.B_BOOT,
                "CENTERING": Phase3Config.CENTERING
            },
            "anchors_run_id": context.anchors_run_id,
            "model_fingerprints": {
                "baseline": context.baseline_model_fingerprint,
                "dapt": context.dapt_model_fingerprint
            },
            "valid_windows_count": len(context.valid_windows),
            "k_selection_method": "min(Horn Parallel Analysis, Bootstrap 5th Percentile Stability)",
            "tolerances": {
                "lowdin_orthonormality": "Frobenius < 1e-3",
                "singular_values": "validity check"
            }
        }
        
        manifest_path = Phase3Config.MANIFESTS_DIR / "run_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        logger.info("Agent 11: Final artifacts generated.")

    def _calculate_deltas(self, df: pd.DataFrame):
        # Delta = DAPT - Baseline
        # For each strategy (penultimate, last4_concat)
        for strategy in Phase3Config.STRATEGIES:
            # Entropy
            base_ent = f"entropy_baseline_{strategy}"
            dapt_ent = f"entropy_dapt_{strategy}"
            if base_ent in df and dapt_ent in df:
                df[f"delta_entropy_{strategy}"] = df[dapt_ent] - df[base_ent]
            
            # Drift
            base_drift = f"drift_baseline_{strategy}"
            dapt_drift = f"drift_dapt_{strategy}"
            if base_drift in df and dapt_drift in df:
                df[f"delta_drift_{strategy}"] = df[dapt_drift] - df[base_drift]
                
            # Projections (Centroid & Subspace) for each Dimension
            for dim in Phase3Config.DIMENSIONS:
                # Centroid
                c_base = f"centroid_proj_{dim}_baseline_{strategy}"
                c_dapt = f"centroid_proj_{dim}_dapt_{strategy}"
                if c_base in df and c_dapt in df:
                    df[f"delta_centroid_proj_{dim}_{strategy}"] = df[c_dapt] - df[c_base]
                    
                # Subspace
                s_base = f"subspace_proj_{dim}_baseline_{strategy}"
                s_dapt = f"subspace_proj_{dim}_dapt_{strategy}"
                if s_base in df and s_dapt in df:
                    df[f"delta_subspace_proj_{dim}_{strategy}"] = df[s_dapt] - df[s_base]
