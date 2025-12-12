import pandas as pd
from pathlib import Path
import datetime as dt

data_dir = Path("data/raw")
files = sorted(data_dir.glob("yape_*.parquet"))

print("=== INSPECTING HARVEST FILES ===")
for f in files:
    try:
        df = pd.read_parquet(f)
        if df.empty:
            print(f"{f.name}: EMPTY")
            continue
            
        # Check published_at or infer from filename year if parsing fails
        # Assuming published_at exists as datetime or string
        if "published_at" in df.columns:
            # Ensure datetime
            timestamps = pd.to_datetime(df["published_at"], utc=True)
            max_date = timestamps.max()
            min_date = timestamps.min()
            count = len(df)
            print(f"{f.name}: {count} records. Range: {min_date.date()} -> {max_date.date()}")
        else:
            print(f"{f.name}: No published_at column found.")
            
    except Exception as e:
        print(f"{f.name}: ERROR reading - {e}")
