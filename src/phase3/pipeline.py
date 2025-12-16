import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, List

# Agents
from src.phase3.schemas import Phase3Config, Phase3RunContext
from src.phase3.agents.auditor import Agent1Auditor
from src.phase3.agents.window_builder import Agent2WindowBuilder
from src.phase3.agents.anchor_generator import Agent3AnchorGenerator
from src.phase3.agents.subspace_builder import (
    Agent4MatrixBuilder, Agent5Centerer, Agent6KSelector, Agent7SubspacePersister
)
from src.phase3.agents.metrics import MetricCalculator
from src.phase3.agents.assembler import Agent11Assembler

logger = logging.getLogger(__name__)

class Phase3Orchestrator:
    """
    Agente 0 — ORQUESTADOR (Controller)
    Responsibility: Execute pipeline, handle errors, validate.
    """
    
    def run(self):
        logger.info("Starting Phase 3 (STRICT) Protocol Orchestrator")
        context = Phase3RunContext()
        
        try:
            # --- 1. Audit ---
            auditor = Agent1Auditor()
            df_source = auditor.run(Phase3Config.INPUT_CSV)
            
            # --- 2. Windows ---
            win_builder = Agent2WindowBuilder()
            valid_windows = win_builder.run(df_source)
            context.valid_windows = valid_windows
            
            # --- 3. Anchors ---
            def update_context(anchors_run_id, baseline_fp, dapt_fp):
                context.anchors_run_id = anchors_run_id
                context.baseline_model_fingerprint = baseline_fp
                context.dapt_model_fingerprint = dapt_fp
                
            anchor_gen = Agent3AnchorGenerator()
            anchor_gen.run(update_context)
            
            # --- Prepare Subspace & Metric Agents ---
            matrix_builder = Agent4MatrixBuilder()
            centerer = Agent5Centerer()
            k_selector = Agent6KSelector()
            subspace_persister = Agent7SubspacePersister()
            metric_calc = MetricCalculator()
            
            # Load anchors for projections once (or per strategy... it depends on variant/strategy)
            # Anchors are specific to variant/strategy.
            
            results_buffer = []
            
            # State for Time Metrics (Previous U)
            # stored as dict: key=(variant, strategy) -> U_prev (d, k)
            prev_U_state = {} 
            
            # --- Loop Windows ---
            for i, (start_m, end_m) in enumerate(valid_windows):
                logger.info(f"Processing Window {i+1}/{len(valid_windows)}: {start_m} to {end_m}")
                
                # Filter data for this window
                # Only need rows where year_month is in [start_m, end_m - 2..] -> No, window logic is in Agent 2.
                # Re-select rows here or pass from Agent 2? Agent 2 returns tuples.
                # Need to replicate logic to slice DF. Efficient enough to do boolean mask.
                
                # Window logic: End month is 'end_m'. 3 months inclusive.
                months_in_window = pd.date_range(end=end_m, periods=3, freq='MS').strftime("%Y-%m").tolist()
                # Wait, pd.date_range(end=...) might not align if end_m isn't end of month?
                # Safer: if end_m="2022-03", we want 2022-01, 2022-02, 2022-03.
                # Re-use manual logic to be safe.
                y, m = map(int, end_m.split('-'))
                # Just string match is risky if we cross years correctly? Agent 2 logic was safe.
                # Let's filter by string "year_month" column which we added in Agent 2?
                # Agent 1 returned cleaned DF, Agent 2 modified it? 
                # Agent 2 modified copy or inplace? Python objects: inplace. 
                # Let's verify Agent 2 code... 'df['year_month'] = ...' -> Yes.
                
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
                
                for variant, strategy in combinations:
                    combo_key = f"{variant}_{strategy}"
                    
                    # Agent 4: Matrix
                    X, mu = matrix_builder.run(window_df, variant, strategy)
                    
                    # Agent 5: Center
                    Xc = centerer.run(X, mu)
                    
                    # Agent 6: k
                    k_horn, k_boot, k_sel = k_selector.run(Xc, 
                                                           B_HORN=Phase3Config.B_HORN, 
                                                           B_BOOT=Phase3Config.B_BOOT, 
                                                           seed=Phase3Config.SEED)
                    
                    # Agent 7: Subspace
                    sub_path = subspace_persister.run(Xc, mu, k_sel, {"start": start_m, "end": end_m}, variant, strategy)
                    
                    # Store Metrics
                    row_res[f"k_{combo_key}"] = k_sel
                    # Optional: k_horn, k_boot
                    row_res[f"subspace_path_{combo_key}"] = sub_path
                    
                    # Load SVD results from Agent 7 logic?
                    # Agent 7 re-runs SVD on Xc. 
                    # Optimization: Agent 7 could return U, s too. 
                    # But adhering to strict separation, we can re-read or Agent 7 returns path, we load?
                    # "Metrics depends on U". 
                    # Let's Modify Agent 7 to return (path, U, s) or just re-read. 
                    # Re-reading is safer for strict persistence.
                    # Or just reuse the math since we have Xc in memory.
                    # I will reuse SVD results if possible? Agent 7 calculates them.
                    # To avoid computing SVD twice (Agent 7 and Agent 8 indirectly?), 
                    # let's modify Agent 7 or just accept overhead?
                    # Actually Agent 8 reads files. "INPUTS: ... paths .npz". 
                    # So I should read from file.
                    
                    # Load from file (Agent 8 Prep)
                    data_npz = np.load(sub_path)
                    U = data_npz['U']
                    s_vals = data_npz['singular_values']
                    
                    # Agent 8: Metrics
                    entropy = metric_calc.calculate_entropy(s_vals)
                    row_res[f"entropy_{combo_key}"] = entropy
                    
                    # Drift / Procrustes
                    prev_U = prev_U_state.get(combo_key)
                    if prev_U is not None:
                        drift, proc = metric_calc.calculate_drift_procrustes(prev_U, U)
                    else:
                        drift, proc = np.nan, np.nan
                    
                    row_res[f"drift_{combo_key}"] = drift
                    row_res[f"procrustes_{combo_key}"] = proc
                    
                    # Update state
                    prev_U_state[combo_key] = U
                    
                    # Agent 9 & 10: Projections
                    # Load Anchors for this combo
                    anchor_map, _ = metric_calc.load_anchors(variant, strategy)
                    
                    # Project Centroid
                    c_projs = metric_calc.calculate_centroid_projection(mu, anchor_map)
                    for k, v in c_projs.items():
                        row_res[f"{k}_{combo_key}"] = v
                        
                    # Project Subspace
                    s_projs = metric_calc.calculate_subspace_projection(U, anchor_map)
                    for k, v in s_projs.items():
                        row_res[f"{k}_{combo_key}"] = v
                
                results_buffer.append(row_res)
                
            # --- 4. Assemble ---
            assembler = Agent11Assembler()
            assembler.run(context, results_buffer)
            
            # --- 5. Final Validation (Hard) ---
            self._validate_outputs()
            
            logger.info("Phase 3 Protocol Completed Successfully.")
            
        except Exception as e:
            logger.error(f"Phase 3 ABORTED due to error: {e}", exc_info=True)
            sys.exit(1)

    def _validate_outputs(self):
        logger.info("Running Final Validation (HARD)...")
        # 1. Check Files
        if not Phase3Config.OUTPUT_CSV.exists():
            raise RuntimeError(f"Validation FAIL: {Phase3Config.OUTPUT_CSV} missing")
        if not (Phase3Config.ARTIFACTS_DIR / "embeddings_anchors.csv").exists():
             raise RuntimeError("Validation FAIL: embeddings_anchors.csv missing")
             
        # Check Manifest
        if not (Phase3Config.MANIFESTS_DIR / "run_manifest.json").exists():
            raise RuntimeError("Validation FAIL: run_manifest.json missing")
            
        # Check Anchors NPZ
        for v in Phase3Config.VARIANTS:
            for s in Phase3Config.STRATEGIES:
                p = Phase3Config.ANCHORS_DIR / f"anchors_{v}_{s}.npz"
                if not p.exists():
                    raise RuntimeError(f"Validation FAIL: missing anchor file {p}")
        
        # Check Results Content
        df = pd.read_csv(Phase3Config.OUTPUT_CSV)
        if df['n_occurrences'].min() < 20:
             raise RuntimeError("Validation FAIL: Found window with < 20 occurrences")
             
        # Check Subspaces exist
        # Sample a few
        for idx, row in df.head(5).iterrows():
            path = row[f"subspace_path_{Phase3Config.VARIANTS[0]}_{Phase3Config.STRATEGIES[0]}"]
            if not Path(path).exists():
                 raise RuntimeError(f"Validation FAIL: subspace path invalid {path}")
                 
        logger.info("Validation Passed.")
