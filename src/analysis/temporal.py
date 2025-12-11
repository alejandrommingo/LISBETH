import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np

class TemporalSegmenter:
    def __init__(self, df: pd.DataFrame, date_column: str = "date"):
        """
        Initializes the segmenter with a dataframe containing a datetime column.
        """
        self.df = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            try:
                self.df[date_column] = pd.to_datetime(self.df[date_column])
            except Exception as e:
                raise ValueError(f"Could not convert column '{date_column}' to datetime: {e}")
                
        self.date_col = date_column
        self.min_date = self.df[self.date_col].min()
        self.max_date = self.df[self.date_col].max()

    def generate_windows(self, window_months=3, step_months=1, min_count=50):
        """
        Generator that yields temporal windows.
        
        Args:
            window_months: Size of the window in months.
            step_months: Step size in months.
            min_count: Minimum number of embeddings required to accept the window.
            
        Yields:
            dict: {
                "start_date": datetime,
                "end_date": datetime,
                "data": DataFrame (subset),
                "count": int,
                "label": str (e.g., "2020-01_2020-03")
            }
        """
        current_start = self.min_date.replace(day=1) # Normalize to start of month
        final_date = self.max_date
        
        while current_start <= final_date:
            # Calculate end date (exclusive)
            current_end = current_start + relativedelta(months=window_months)
            
            # Filter data
            mask = (self.df[self.date_col] >= current_start) & (self.df[self.date_col] < current_end)
            window_data = self.df[mask].copy()
            count = len(window_data)
            
            # Check density
            if count >= min_count:
                label = f"{current_start.strftime('%Y-%m')}_{current_end.strftime('%Y-%m')}"
                yield {
                    "start_date": current_start,
                    "end_date": current_end,
                    "data": window_data,
                    "count": count,
                    "label": label
                }
            
            # Move sliding window
            current_start += relativedelta(months=step_months)

    @staticmethod
    def load_parquet(path):
        """Helper to load data from parquet efficiently."""
        if not path.endswith('.parquet'):
            raise ValueError("File must be a parquet file.")
        return pd.read_parquet(path)
