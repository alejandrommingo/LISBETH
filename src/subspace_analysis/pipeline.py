import logging
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Modules (Refactored)
from src.subspace_analysis.schemas import Phase3Config, Phase3RunContext
from src.subspace_analysis.auditor import DataAuditor
from src.subspace_analysis.windowing import WindowPipelineStep
from src.subspace_analysis.anchors import AnchorGenerator
from src.subspace_analysis.subspace import (
    MatrixBuilder, Centerer, KSelector, SubspacePersister
)
from src.subspace_analysis.metrics import MetricCalculator
from src.subspace_analysis.pipeline_assembler import PipelineAssembler

logger = logging.getLogger(__name__)

class Phase3Orchestrator:
    """
    Phase 3 Orchestrator (Controller)
    Responsibility: Execute pipeline, handle errors, validate.
    """
    
    def run(self):
        logger.info("Starting Phase 3 (STRICT) Protocol Orchestrator")
        context = Phase3RunContext()
        
        try:
            # --- 1. Audit ---
            auditor = DataAuditor()
            df_source = auditor.run(Phase3Config.INPUT_CSV)
            
            # --- 2. Windows ---
            win_builder = WindowPipelineStep()
            valid_windows = win_builder.run(df_source)
            context.valid_windows = valid_windows
            
            # --- 3. Anchors ---
            def update_context(anchors_run_id, baseline_fp, dapt_fp):
                context.anchors_run_id = anchors_run_id
                context.baseline_model_fingerprint = baseline_fp
                context.dapt_model_fingerprint = dapt_fp
                
            anchor_gen = AnchorGenerator()
            anchor_gen.run(update_context)
            
            # --- Prepare Subspace & Metric Modules ---
            matrix_builder = MatrixBuilder()
            centerer = Centerer()
            k_selector = KSelector()
            subspace_persister = SubspacePersister()
            metric_calc = MetricCalculator()
            
            results_buffer = []
            
            # State for Time Metrics (Previous U)
            prev_U_state = {} 
            
            # --- Loop Windows ---
            for i, (start_m, end_m) in enumerate(valid_windows):
                logger.info(f"Processing Window {i+1}/{len(valid_windows)}: {start_m} to {end_m}")
                
                months_in_window = pd.date_range(end=end_m, periods=3, freq='MS').strftime("%Y-%m").tolist()
                mask = df_source['year_month'].isin(months_in_window)
                window_df = df_source[mask]
                
                # Row data
                row_res = {
                    "window_start_month": start_m,
                    "window_end_month": end_m,
                    "window_size_months": Phase3Config.WINDOW_MONTHS,
                    "step_months": Phase3Config.STEP_MONTHS,
                    "n_occurrences": len(window_df),
                    "n_documents": window_df[Phase3Config.COL_URL].nunique(),
                    "low_density": len(window_df) < Phase3Config.LOW_DENSITY_FLAG
                }
                
                # Loop Combinations
                combinations = [(v, s) for v in Phase3Config.VARIANTS for s in Phase3Config.STRATEGIES]
                
                # Calculate/Cache Global Means for this Window run (Or globally? Best globally but input df available here)
                # Actually we can do it lazily.
                
                # Loop Combinations
                combinations = [(v, s) for v in Phase3Config.VARIANTS for s in Phase3Config.STRATEGIES]
                
                for variant, strategy in combinations:
                    # 1. Get Window Data (Raw)
                    X_raw, mu_raw = matrix_builder.run(window_df, variant, strategy)
                    
                    # 2. Get Global Mean (for Correction)
                    # We need the column name
                    col_name = f"embedding_{variant}_{strategy}"
                    cache_key = f"mu_global_{variant}_{strategy}"
                    mu_global = None
                    
                    if hasattr(self, cache_key):
                        mu_global = getattr(self, cache_key)
                    else:
                        full_stack = np.vstack([np.array(json.loads(x)) for x in df_source[col_name]])
                        mu_global = np.mean(full_stack, axis=0)
                        setattr(self, cache_key, mu_global)

                    
                    # Loop Conditions
                    for condition in Phase3Config.CONDITIONS: # ["raw", "corrected"]
                        combo_key = f"{variant}_{strategy}_{condition}"
                        
                        # Apply Condition
                        if condition == "corrected":
                            # Corrected: X = X_raw - mu_global
                            # The centroid of this new cloud is mu_raw - mu_global
                            # Xc (centered) is (X_raw - mu_global) - (mu_raw - mu_global) = X_raw - mu_raw.
                            # So Subspace is IDENTICAL.
                            X_input = X_raw - mu_global
                            mu_input = np.mean(X_input, axis=0)
                        else:
                            X_input = X_raw
                            mu_input = mu_raw
                        
                        # Center (PCA proper)
                        Xc = centerer.run(X_input, mu_input)
                        
                        # k
                        k_horn, k_boot, k_sel = k_selector.run(Xc, 
                                                               B_HORN=Phase3Config.B_HORN, 
                                                               B_BOOT=Phase3Config.B_BOOT, 
                                                               seed=Phase3Config.SEED)
                        
                        # Subspace
                        # Note: Subspace U is identical for Raw/Corrected if centered.
                        # But we save it with specific key anyway.
                        sub_path = subspace_persister.run(Xc, mu_input, k_sel, {"start": start_m, "end": end_m}, variant, strategy, suffix=condition)
                        
                        # Store Metrics
                        row_res[f"k_{combo_key}"] = k_sel
                        row_res[f"subspace_path_{combo_key}"] = sub_path
                        
                        # Load & Metrics
                        data_npz = np.load(sub_path)
                        U = data_npz['U']
                        s_vals = data_npz['singular_values']
                        
                        entropy = metric_calc.calculate_entropy(s_vals)
                        row_res[f"entropy_{combo_key}"] = entropy
                        
                        # Drift
                        prev_U = prev_U_state.get(combo_key)
                        if prev_U is not None:
                            drift, proc = metric_calc.calculate_drift_procrustes(prev_U, U)
                        else:
                            drift, proc = np.nan, np.nan
                        
                        row_res[f"drift_{combo_key}"] = drift
                        row_res[f"procrustes_{combo_key}"] = proc
                        
                        prev_U_state[combo_key] = U
                        
                        # Projections
                        # AnchorMap: Raw anchors (orthogonalized). 
                        # For "Corrected", we technically project the Corrected Centroid onto the Raw Anchors?
                        # Or should we project Corrected Centroid onto Corrected Anchors?
                        # Standard practice: Keep Anchors fixed as Reference Frame. Correct the signal (Yape).
                        anchor_map, _ = metric_calc.load_anchors(variant, strategy, condition=condition)
                        
                        # Project Centroid
                        c_projs = metric_calc.calculate_centroid_projection(mu_input, anchor_map)
                        for k, v in c_projs.items():
                            row_res[f"{k}_{combo_key}"] = v
                            
                        # Project Subspace
                        s_projs = metric_calc.calculate_subspace_projection(U, anchor_map)
                        for k, v in s_projs.items():
                            row_res[f"{k}_{combo_key}"] = v
                
                results_buffer.append(row_res)
                
            # --- 4. Assemble ---
            assembler = PipelineAssembler()
            assembler.run(context, results_buffer)
            
            # --- 5. Final Validation (Hard) ---
            self._validate_outputs()
            
            logger.info("Phase 3 Protocol Completed Successfully.")
            
        except Exception as e:
            logger.error(f"Phase 3 ABORTED due to error: {e}", exc_info=True)
            sys.exit(1)

    def _validate_outputs(self):
        logger.info("Running Final Validation (HARD)...")
        if not Phase3Config.OUTPUT_CSV.exists():
            raise RuntimeError(f"Validation FAIL: {Phase3Config.OUTPUT_CSV} missing")
        if not (Phase3Config.ARTIFACTS_DIR / "embeddings_anchors.csv").exists():
             raise RuntimeError("Validation FAIL: embeddings_anchors.csv missing")
             
        if not (Phase3Config.MANIFESTS_DIR / "run_manifest.json").exists():
            raise RuntimeError("Validation FAIL: run_manifest.json missing")
            
        for v in Phase3Config.VARIANTS:
            for s in Phase3Config.STRATEGIES:
                p = Phase3Config.ANCHORS_DIR / f"anchors_{v}_{s}.npz"
                if not p.exists():
                    raise RuntimeError(f"Validation FAIL: missing anchor file {p}")
        
        df = pd.read_csv(Phase3Config.OUTPUT_CSV)
        if df['n_occurrences'].min() < Phase3Config.N_MIN_OCCURRENCES:
             raise RuntimeError(f"Validation FAIL: Found window with < {Phase3Config.N_MIN_OCCURRENCES} occurrences")
             
        for idx, row in df.head(5).iterrows():
            # key: subspace_path_baseline_penultimate_raw
            v = Phase3Config.VARIANTS[0]
            s = Phase3Config.STRATEGIES[0]
            c = Phase3Config.CONDITIONS[0] # raw
            key = f"subspace_path_{v}_{s}_{c}"
            
            if key not in row:
                 logger.warning(f"Validation: Key {key} not found in output. Available: {row.keys()}")
                 continue
                 
            path = row[key]
            if not Path(path).exists():
                 raise RuntimeError(f"Validation FAIL: subspace path invalid {path}")
                  
        logger.info("Validation Passed.")
