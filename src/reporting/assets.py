import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'

def load_data(input_csv):
    df = pd.read_csv(input_csv)
    # Ensure date parsing works for 'YYYY-MM' format
    df['date'] = pd.to_datetime(df['window_end_month'])
    return df.sort_values('date')

def plot_drift(df, output_dir):
    plt.figure()
    
    # Identify Variants/Conditions
    # We want to compare RAW vs CORRECTED for the same Variant/Strategy
    # Let's find one representant (e.g., DAPT/Penultimate)
    
    candidate_bases = set()
    for c in df.columns:
        if "drift_" in c:
            # drift_variant_strategy_condition
            parts = c.split("_")
            # Expected: drift, variant, strategy, condition
            if len(parts) >= 4:
                base = "_".join(parts[1:-1]) # variant_strategy
                candidate_bases.add(base)
    
    if not candidate_bases:
        print("No paired drift data found.")
        return

    # Select best candidate (prefer DAPT)
    base = next((b for b in candidate_bases if "dapt" in b), list(candidate_bases)[0])
    
    raw_col = f"drift_{base}_raw"
    cor_col = f"drift_{base}_corrected"
    
    if raw_col in df.columns and cor_col in df.columns:
        sns.lineplot(data=df, x='date', y=raw_col, marker='o', label='Raw', color='gray', linestyle='--')
        sns.lineplot(data=df, x='date', y=cor_col, marker='o', label='Corrected', color='#d62728')
        plt.title(f'Semantic Drift Comparison ({base})', fontsize=16)
    elif raw_col in df.columns:
        sns.lineplot(data=df, x='date', y=raw_col, marker='o', label='Raw')
        plt.title(f'Semantic Drift (Raw)', fontsize=16)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "semantic_drift_comparison.png"), dpi=300)
    plt.close()
    print(f"Generated {os.path.join(output_dir, 'semantic_drift_comparison.png')}")

def plot_entropy(df, output_dir):
    plt.figure()
    # Similar logic
    candidate_bases = set()
    for c in df.columns:
        if "entropy_" in c:
            parts = c.split("_")
            if len(parts) >= 4:
                base = "_".join(parts[1:-1])
                candidate_bases.add(base)
    
    if not candidate_bases: return
    base = next((b for b in candidate_bases if "dapt" in b), list(candidate_bases)[0])
    
    raw_col = f"entropy_{base}_raw"
    cor_col = f"entropy_{base}_corrected"

    if raw_col in df.columns:
        sns.lineplot(data=df, x='date', y=raw_col, marker='s', label='Raw', color='gray', linestyle='--')
    if cor_col in df.columns:
        sns.lineplot(data=df, x='date', y=cor_col, marker='s', label='Corrected', color='#1f77b4')

    plt.title(f'Entropy Comparison ({base})', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "semantic_entropy_comparison.png"), dpi=300)
    plt.close()

def plot_projections_heatmap(df, output_dir):
    # We generate TWO heatmaps: Raw & Corrected
    # Find active columns
    all_cols = [c for c in df.columns if "centroid_proj_" in c]
    
    conditions = ["raw", "corrected"]
    
    for cond in conditions:
        # Filter cols ending with _cond or _{base}_{cond}
        # Column format: centroid_proj_functional_DAPT_Penultimate_raw
        # Actually in pipeline.py: `f"{k}_{combo_key}"` where k is "centroid_proj_DIM" and combo is "VAR_STRAT_COND"
        # So: centroid_proj_funcional_baseline_penultimate_raw
        
        # We need to find the base variant/strategy
        cols = [c for c in all_cols if c.endswith(f"_{cond}")]
        if not cols: continue
        
        # Pick DAPT if available
        dapt_cols = [c for c in cols if "dapt" in c]
        use_cols = dapt_cols if dapt_cols else cols
        
        # Extract Dimension Labels
        # c looks like: centroid_proj_{dim}_{base}_{cond}
        # We need to extract {dim}.
        # Split by _
        # centroid, proj, [dim], [variant], [strategy], [condition]
        
        labels = []
        final_cols = []
        for c in use_cols:
            parts = c.split("_")
            # Assuming format: centroid_proj_DIMNAME_variant_strategy_condition
            # But DIMNAME might have underscores? "afectiva" usually no.
            # Parts: 0:centroid, 1:proj, 2...N-4:DIM, N-3:Var, N-2:Strat, N-1:Cond
            # Safest is to remove known suffix and prefix.
            # Suffix: _{var}_{strat}_{cond}
            # Prefix: centroid_proj_
            
            # Let's clean it up dynamically
            # Find index of variant?
            # We know cond is last.
            # base is var_strat.
            # To isolate dim, we need to know var_strat string.
            # Let's guess base from one column
            pass
        
        # Simple extraction for now assuming standard dims
        dims = ["funcional", "social", "afectiva"]
        
        selected_matrix_cols = []
        selected_labels = []
        
        # Group by Base?
        # Just take the first valid Base found
        base_found = None
        for c in use_cols:
            if not base_found:
                # Infer base from matching suffix
                suffix = f"_{cond}"
                without_suffix = c[:-len(suffix)] # centroid_proj_dim_var_strat
                # Try to split?
                parts = without_suffix.split("_")
                # last 2 parts are var, strat
                base_found = "_".join(parts[-2:])
        
        if not base_found: continue
        
        # Now collect only cols for this base
        for dim in dims:
            # Construct expected col name
            # keys were: f"{k}_{combo_key}" -> centroid_proj_dim_var_strat_cond
            col_name = f"centroid_proj_{dim}_{base_found}_{cond}"
            if col_name in df.columns:
                selected_matrix_cols.append(col_name)
                selected_labels.append(dim)
        
        if not selected_matrix_cols: continue

        heatmap_data = df.set_index('date')[selected_matrix_cols].T
        heatmap_data.index = selected_labels # Just dim names
        
        plt.figure(figsize=(10, 5))
        sns.heatmap(heatmap_data, cmap="viridis", annot=False)
        plt.title(f'Projections ({cond.upper()}) - {base_found}', fontsize=16)
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"projection_heatmap_{cond}.png"), dpi=300)
        plt.close()
        print(f"Generated projection_heatmap_{cond}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Phase 3 CSV")
    parser.add_argument("--output", required=True, help="Output Assets Directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    try:
        df = load_data(args.input)
    except FileNotFoundError:
        print("Input file not found.")
        sys.exit(1)
        
    if df.empty:
        print("Dataframe empty.")
        sys.exit(1)
    
    print("Generating assets...")
    plot_drift(df, args.output)
    plot_entropy(df, args.output)
    plot_projections_heatmap(df, args.output)
    
    print("Done.")

if __name__ == "__main__":
    main()
