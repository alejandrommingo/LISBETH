import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_CSV = "data/phase3/phase3_results.csv"
OUTPUT_DIR = "academic/methodological_report/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'

def load_data():
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['window_end_month'])
    return df.sort_values('date')

def plot_drift(df):
    plt.figure()
    # Use the best config: DAPT + Last4 Concat
    sns.lineplot(data=df, x='date', y='drift_dapt_last4_concat', marker='o', linewidth=2.5, color='#d62728')
    plt.title('Semantic Drift of "Yape" (Grassmannian Distance)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Distance ($d(S_t, S_{t-1})$)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Annotate high peaks if any
    peak = df.loc[df['drift_dapt_last4_concat'].idxmax()]
    plt.annotate('Max Drift', 
                 xy=(peak['date'], peak['drift_dapt_last4_concat']), 
                 xytext=(peak['date'], peak['drift_dapt_last4_concat'] + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "semantic_drift.png"), dpi=300)
    plt.close()
    print(f"Generated {os.path.join(OUTPUT_DIR, 'semantic_drift.png')}")

def plot_entropy(df):
    plt.figure()
    sns.lineplot(data=df, x='date', y='entropy_dapt_last4_concat', marker='s', linewidth=2.5, color='#1f77b4')
    plt.title('Evolution of Semantic Entropy (Complexity)', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "semantic_entropy.png"), dpi=300)
    plt.close()
    print(f"Generated {os.path.join(OUTPUT_DIR, 'semantic_entropy.png')}")

def plot_projections_heatmap(df):
    # Prepare data for heatmap
    # Pivot or melt? We want Date on X, Dimension on Y
    cols = [
        'centroid_proj_funcional_dapt_last4_concat',
        'centroid_proj_social_dapt_last4_concat', 
        'centroid_proj_afectiva_dapt_last4_concat'
    ]
    labels = ['Functional', 'Social', 'Affective']
    
    # Create valid dataframe for heatmap
    heatmap_data = df.set_index('date')[cols].T
    heatmap_data.index = labels
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap="viridis", annot=False, cbar_kws={'label': 'Projection Intensity'})
    plt.title('Thematic Projection Heatmap', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Dimension')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "projection_heatmap.png"), dpi=300)
    plt.close()
    print(f"Generated {os.path.join(OUTPUT_DIR, 'projection_heatmap.png')}")

def main():
    print("Loading data...")
    df = load_data()
    
    print("Generating assets...")
    plot_drift(df)
    plot_entropy(df)
    plot_projections_heatmap(df)
    
    print("Done.")

if __name__ == "__main__":
    main()
