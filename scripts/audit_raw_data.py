
import pandas as pd
import os
import glob

data_dir = 'data'
files = glob.glob(os.path.join(data_dir, 'yape_*.csv'))

print(f"Found {len(files)} files: {files}")

full_df = pd.DataFrame()

for f in sorted(files):
    if 'yape_yapear' in f: continue # Skip the aggregate test file
    try:
        df = pd.read_csv(f)
        # Try to parse dates. GDELT usually has a date column 
        # Inspect columns first if unsure, but typical is 'date' or 'fecha'
        # based on previous errors, user has 'date' column
        if 'published_at' in df.columns:
            df['file'] = os.path.basename(f)
            full_df = pd.concat([full_df, df])
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not full_df.empty:
    print(f"\nTotal records: {len(full_df)}")
    
    # standardize date
    full_df['dt'] = pd.to_datetime(full_df['published_at'], errors='coerce')
    full_df = full_df.dropna(subset=['dt'])
    
    # Group by Month
    monthly_counts = full_df.groupby(full_df['dt'].dt.to_period('M')).size()
    
    print("\n--- MONTHLY COUNTS ---")
    print(monthly_counts)
    
    # Check for gaps
    min_date = full_df['dt'].min()
    max_date = full_df['dt'].max()
    all_months = pd.period_range(min_date, max_date, freq='M')
    
    missing = all_months.difference(monthly_counts.index)
    if not missing.empty:
        print("\n--- MISSING MONTHS ---")
        print(missing)
    else:
        print("\nNo completely missing months in range.")
        
    print("\n--- LOW DENSITY MONTHS (< 10 news) ---")
    print(monthly_counts[monthly_counts < 10])

else:
    print("No data loaded.")
