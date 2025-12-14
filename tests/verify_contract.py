import pandas as pd
import sys
import os

def verify_contract(file_path):
    print(f"Verifying CONTRACT for: {file_path}")
    if not os.path.exists(file_path):
        print("FAIL: File not found")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows.")
    
    # Required Columns
    required = [
        "char_start", "char_end", 
        "token_start", "token_end", 
        "source_api", 
        "model_baseline", "model_dapt",
        "published_at", "newspaper"
    ]
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"FAIL: Missing columns: {missing}")
        sys.exit(1)
        
    print("SUCCESS: All contract columns present.")
    
    # Check Nulls in critical
    null_errs = 0
    for c in required:
        n_nulls = df[c].isnull().sum()
        if n_nulls > 0:
            print(f"FAIL: Column {c} has {n_nulls} nulls!")
            null_errs += 1
            
            # Print sample of nulls
            print(df[df[c].isnull()].head(2))
            
    if null_errs > 0:
        sys.exit(1)
        
    print("SUCCESS: No nulls in contract columns.")
    
    # Check types
    # spans should be int-like
    if not pd.api.types.is_numeric_dtype(df['token_start']):
        print("WARNING: token_start is not numeric")
        
    print("Contract Verification PASSED.")

if __name__ == "__main__":
    verify_contract("data/phase2/embeddings_occurrences.csv")
