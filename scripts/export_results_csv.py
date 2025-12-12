
import pandas as pd
import numpy as np
import json
import os

INPUT_PARQUET = 'data/phase3_results_spanish.parquet'
OUTPUT_CSV = 'data/resultados_finales_spanish_sota.csv'

def export_results():
    if not os.path.exists(INPUT_PARQUET):
        print(f"Error: {INPUT_PARQUET} not found.")
        return

    print(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    # 1. Inspect Columns
    print("Columns:", df.columns.tolist())
    
    # 2. Serialize Complex Columns (Arrays/Lists)
    # CSVs can't handle lists natively, so we JSON-encode them.
    for col in df.columns:
        # Check if type is object (could be list)
        if df[col].dtype == 'object':
            # Check first element to see if it's a list/array
            first_val = df[col].iloc[0] if not df.empty else None
            if isinstance(first_val, (list, np.ndarray)):
                print(f"Serializing column: {col}")
                # Use json.dumps for lists. Handle numpy arrays by converting to list first.
                df[col] = df[col].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else json.dumps(x))

    # 3. Export
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Successfully exported {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    export_results()
