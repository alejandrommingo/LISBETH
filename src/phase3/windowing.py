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
