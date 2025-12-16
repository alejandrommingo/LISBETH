import pandas as pd
import json
import logging
import numpy as np
from src.phase3.schemas import Phase3Config

logger = logging.getLogger(__name__)

class Agent1Auditor:
    """
    Agente 1 — AUDITOR DE INPUTS (Data Auditor)
    Responsibility: Validate inputs, schema, NaNs.
    """
    
    def run(self, input_csv_path: str) -> pd.DataFrame:
        logger.info(f"Agent 1: Auditing {input_csv_path}...")
        
        # 1. Read CSV
        try:
            df = pd.read_csv(input_csv_path)
        except Exception as e:
            raise RuntimeError(f"FAIL: Could not read CSV: {e}")
            
        # 2. Verify existence of minimum columns
        required_cols = [
            Phase3Config.COL_OCCURRENCE_ID,
            Phase3Config.COL_PUBLISHED_AT,
            Phase3Config.COL_URL,
            Phase3Config.COL_EMB_BASELINE_PENULTIMATE,
            Phase3Config.COL_EMB_BASELINE_LAST4,
            Phase3Config.COL_EMB_DAPT_PENULTIMATE,
            Phase3Config.COL_EMB_DAPT_LAST4
        ]
        
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise RuntimeError(f"FAIL: Missing required columns: {missing_cols}")
            
        # 3. Parse published_at to datetime UTC
        # If any is NaT -> FAIL
        try:
            df[Phase3Config.COL_PUBLISHED_AT] = pd.to_datetime(df[Phase3Config.COL_PUBLISHED_AT], utc=True, errors='coerce')
        except Exception as e:
             raise RuntimeError(f"FAIL: Date parsing error: {e}")
             
        if df[Phase3Config.COL_PUBLISHED_AT].isna().any():
            raise RuntimeError("FAIL: Found NaT in published_at after parsing.")
            
        # 4. Verify occurrence_id unique
        if df[Phase3Config.COL_OCCURRENCE_ID].duplicated().any():
            dups = df[Phase3Config.COL_OCCURRENCE_ID].duplicated().sum()
            raise RuntimeError(f"FAIL: Found {dups} duplicate occurrence_ids.")
            
        # 5. Check Embeddings (Random Sample of 100)
        embedding_cols = [
            Phase3Config.COL_EMB_BASELINE_PENULTIMATE,
            Phase3Config.COL_EMB_BASELINE_LAST4,
            Phase3Config.COL_EMB_DAPT_PENULTIMATE,
            Phase3Config.COL_EMB_DAPT_LAST4
        ]
        
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        dims_detected = {}
        
        for col in embedding_cols:
            col_dims = []
            # We don't want to parse ALL if dataset is huge, but Agent 4 will parse them later.
            # Here we just audit the sample to fail fast.
            
            for idx, val in sample_df[col].items():
                try:
                    vec = json.loads(val)
                except Exception as e:
                    raise RuntimeError(f"FAIL: JSON parse error in {col} at index {idx}: {e}")
                
                if not vec or not isinstance(vec, list):
                    raise RuntimeError(f"FAIL: Empty or invalid list in {col} at index {idx}")
                
                # Check for NaN/Inf in values
                if not np.all(np.isfinite(vec)):
                     raise RuntimeError(f"FAIL: NaN or Inf found in embedding {col} at index {idx}")
                     
                col_dims.append(len(vec))
            
            # Check length constancy
            if len(set(col_dims)) > 1:
                raise RuntimeError(f"FAIL: Inconsistent dimensions in {col}: {set(col_dims)}")
            
            dims_detected[col] = col_dims[0]

        # 6. Verify dimensional consistency
        # dim(baseline_penultimate) == dim(dapt_penultimate)
        if dims_detected[Phase3Config.COL_EMB_BASELINE_PENULTIMATE] != dims_detected[Phase3Config.COL_EMB_DAPT_PENULTIMATE]:
             raise RuntimeError(f"FAIL: Dimension mismatch Penultimate: Baseline={dims_detected[Phase3Config.COL_EMB_BASELINE_PENULTIMATE]} vs DAPT={dims_detected[Phase3Config.COL_EMB_DAPT_PENULTIMATE]}")
             
        # dim(baseline_last4) == dim(dapt_last4)
        if dims_detected[Phase3Config.COL_EMB_BASELINE_LAST4] != dims_detected[Phase3Config.COL_EMB_DAPT_LAST4]:
             raise RuntimeError(f"FAIL: Dimension mismatch Last4: Baseline={dims_detected[Phase3Config.COL_EMB_BASELINE_LAST4]} vs DAPT={dims_detected[Phase3Config.COL_EMB_DAPT_LAST4]}")

        # 7. Audit Report
        logger.info("Audit Successful.")
        logger.info(f"Range: {df[Phase3Config.COL_PUBLISHED_AT].min()} to {df[Phase3Config.COL_PUBLISHED_AT].max()}")
        logger.info(f"Total Rows: {len(df)}")
        logger.info(f"Dimensions: {dims_detected}")
        
        return df
