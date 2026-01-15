import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import logging

logger = logging.getLogger(__name__)

class RollingWindowSegmenter:
    def __init__(self, window_months: int = 3, step_months: int = 1, min_count: int = 50):
        self.window_months = window_months
        self.step_months = step_months
        self.min_count = min_count

    def get_windows(self, df: pd.DataFrame):
        """
        Yields (window_start, window_end, df_window) tuples.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to segmenter.")
            return

        # Ensure sorted
        df = df.sort_values('published_at')
        
        start_date = df['published_at'].min().normalize()
        end_date = df['published_at'].max().normalize()
        
        # Start at the beginning of the month of the start_date
        current_start = start_date.replace(day=1)
        
        while current_start + relativedelta(months=self.window_months) <= end_date + relativedelta(months=1): 
            # Define window range
            current_end = current_start + relativedelta(months=self.window_months)
            
            # Filter data
            mask = (df['published_at'] >= current_start) & (df['published_at'] < current_end)
            df_window = df[mask].copy()
            
            # Density Check
            count = len(df_window)
            
            # Check unique keywords? Or just raw volume?
            # Report says: "n_t >= n_min" (occurrences).
            # Also "group by keyword canonical". 
            
            if count >= self.min_count:
                yield {
                    'start': current_start,
                    'end': current_end,
                    'data': df_window,
                    'count': count
                }
            else:
                logger.debug(f"Skipping window {current_start.date()} - {current_end.date()}: density {count} < {self.min_count}")
            
            # Advance
            current_start += relativedelta(months=self.step_months)

class WindowPipelineStep:
    """
    Pipeline Step: Window Builder
    Responsibility: Create rolling windows and filter valid ones using Phase3Config.
    """
    
    def run(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        from src.subspace_analysis.schemas import Phase3Config
        logger.info("WindowPipelineStep: Building windows...")
        
        # 1. Create year_month
        df['year_month'] = df[Phase3Config.COL_PUBLISHED_AT].dt.strftime("%Y-%m")
        unique_months = sorted(df['year_month'].unique())
        
        if not unique_months:
            raise RuntimeError("FAIL: No months found in data.")
            
        min_ym = unique_months[0]
        max_ym = unique_months[-1]
        
        # 2. Continuous monthly index
        full_range = pd.date_range(start=min_ym, end=max_ym, freq='MS')
        all_months = full_range.strftime("%Y-%m").tolist()
        
        valid_windows = []
        
        # 3. Calculate rolling window
        win_size = Phase3Config.WINDOW_MONTHS
        
        if len(all_months) < win_size:
             logger.warning(f"Data span ({len(all_months)} months) is less than window ({win_size} months). No full windows possible.")
        
        window_counts = []

        for i in range(win_size - 1, len(all_months)):
            end_month = all_months[i]
            # Window slice
            start_idx = i - (win_size - 1)
            window_months = all_months[start_idx : i + 1]
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

        # 6. FAIL Condition: check min windows
        if len(valid_windows) < Phase3Config.MIN_WINDOWS:
            raise RuntimeError(f"FAIL: Only {len(valid_windows)} valid windows found (minimum {Phase3Config.MIN_WINDOWS} required). Checks details in {manifest_path}")
            
        logger.info(f"WindowPipelineStep: Found {len(valid_windows)} valid windows.")
        return valid_windows
