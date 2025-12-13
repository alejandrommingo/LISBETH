import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class Phase3DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_occurrences(self) -> pd.DataFrame:
        """
        Loads the pre-standardized occurrences parquet file.
        Expects columns: published_at, keyword_canonical, embedding_contextual_last4, etc.
        """
        logger.info(f"Loading Standardized Data: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded rows: {len(df)}")
        
        # Ensure 'embedding' column exists for downstream
        if 'embedding' not in df.columns and 'embedding_contextual_last4' in df.columns:
            df['embedding'] = df['embedding_contextual_last4']
            
        # Ensure published_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['published_at']):
            df['published_at'] = pd.to_datetime(df['published_at'])
            
        return df
