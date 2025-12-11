
import pandas as pd
import numpy as np

try:
    df = pd.read_parquet('data/phase3_results.parquet')
    print("--- INFO ---")
    print(df.info())
    print("\n--- SAMPLE DATES ---")
    print(df['date'].head(10))
    print("\n--- LAST DATES ---")
    print(df['date'].tail(10))
    
    print("\n--- NULLS PER COLUMN ---")
    print(df.isnull().sum())
    
    print("\n--- PROJECTION COLUMNS SAMPLE ---")
    cols = [c for c in df.columns if 'score_' in c]
    if cols:
        print(df[cols].head())
    else:
        print("No projection columns found!")
        
    print("\n--- CHECKING PARSE LOGIC ---")
    # Simulate the parsing logic used in the report
    try:
        parsed_dates = pd.to_datetime(df['date'].apply(lambda x: x.split('_')[0]))
        print("Parsing successful. Range:", parsed_dates.min(), "to", parsed_dates.max())
    except Exception as e:
        print(f"Parsing FAILED: {e}")

except Exception as e:
    print(f"Error loading parquet: {e}")
