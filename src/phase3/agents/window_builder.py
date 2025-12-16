import pandas as pd
import logging
from typing import List, Tuple
from src.phase3.schemas import Phase3Config

logger = logging.getLogger(__name__)

class Agent2WindowBuilder:
    """
    Agente 2 — CONSTRUCTOR DE VENTANAS (Window Builder)
    Responsibility: Create rolling windows and filter valid ones.
    """
    
    def run(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        logger.info("Agent 2: Building windows...")
        
        # 1. Create year_month
        df['year_month'] = df[Phase3Config.COL_PUBLISHED_AT].dt.strftime("%Y-%m")
        unique_months = sorted(df['year_month'].unique())
        
        if not unique_months:
            raise RuntimeError("FAIL: No months found in data.")
            
        min_ym = unique_months[0]
        max_ym = unique_months[-1]
        
        # 2. Continuous monthly index
        # We need a proper date range to ensure we don't skip missing months
        full_range = pd.date_range(start=min_ym, end=max_ym, freq='MS')
        all_months = full_range.strftime("%Y-%m").tolist()
        
        valid_windows = []
        
        # 3. Calculate rolling window 3 months
        # We start looking from the 3rd month (index 2)
        if len(all_months) < 3:
             logger.warning("Data span is less than 3 months. No full windows possible.")
             # Technically this might be a fail if we strictly need 2 windows, 
             # but check is done at the end of this agent.
        
        window_counts = []

        for i in range(2, len(all_months)):
            end_month = all_months[i]
            # Window includes i-2, i-1, i
            window_months = [all_months[i-2], all_months[i-1], all_months[i]]
            start_month = window_months[0]
            
            # Select rows
            mask = df['year_month'].isin(window_months)
            window_df = df[mask]
            
            n_occurrences = len(window_df)
            n_documents = window_df[Phase3Config.COL_URL].nunique()
            
            is_valid = n_occurrences >= Phase3Config.N_MIN_OCCURRENCES
            
            window_counts.append({
                "window_start": start_month,
                "window_end": end_month,
                "n_occurrences": n_occurrences,
                "n_documents": n_documents,
                "valid": is_valid
            })
            
            if is_valid:
                valid_windows.append((start_month, end_month))
                
        # Save diagnostics
        counts_df = pd.DataFrame(window_counts)
        manifest_path = Phase3Config.MANIFESTS_DIR / "window_counts_all.csv"
        counts_df.to_csv(manifest_path, index=False)
        logger.info(f"Saved window counts to {manifest_path}")

        # 6. FAIL Condition: < 2 valid windows
        if len(valid_windows) < 2:
            raise RuntimeError(f"FAIL: Only {len(valid_windows)} valid windows found (minimum 2 required). Checks details in {manifest_path}")
            
        logger.info(f"Agent 2: Found {len(valid_windows)} valid windows.")
        return valid_windows
