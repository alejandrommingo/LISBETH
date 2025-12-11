
import pandas as pd
import glob
import os

def merge_csvs(target_year, new_file):
    original_file = f"data/yape_{target_year}.csv"
    if not os.path.exists(original_file):
        print(f"Original {original_file} not found. Renaming new file.")
        if os.path.exists(new_file):
            os.rename(new_file, original_file)
        return

    if not os.path.exists(new_file):
        print(f"New file {new_file} not found. Nothing to merge.")
        return

    print(f"Merging {new_file} into {original_file}...")
    try:
        df_orig = pd.read_csv(original_file)
        df_new = pd.read_csv(new_file)
        
        # Concat
        combined = pd.concat([df_orig, df_new])
        
        # Deduplicate based on URL if exists, else plain_text
        if 'url' in combined.columns:
            combined = combined.drop_duplicates(subset=['url'])
        else:
            combined = combined.drop_duplicates(subset=['plain_text'])
            
        print(f"Original size: {len(df_orig)}, New size: {len(df_new)}, Combined: {len(combined)}")
        
        combined.to_csv(original_file, index=False)
        print("Merged successfully.")
        
    except Exception as e:
        print(f"Merge failed: {e}")

if __name__ == "__main__":
    # Merge 2021 Gap
    merge_csvs("2021", "data/yape_2021_gap.csv")
    
    # Merge 2022 v2
    merge_csvs("2022", "data/yape_2022_v2.csv")
