
import nbformat as nbf
import os

def create_comparison_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- TITLE ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
# Comparative Analysis: Spanish SOTA vs Multilingual Model
## Impact on Semantic Subspaces of 'Yape'
    """))
    
    # --- SETUP CODE ---
    nb.cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Path setup
sys.path.append(os.path.abspath('..')) 
from src.visualization.paper_plots import setup_pub_style, _handle_date_axis

setup_pub_style()

# Load Results
df_es = pd.read_parquet('../data/phase3_results_spanish.parquet')
df_multi = pd.read_parquet('../data/phase3_results_multilingual.parquet')

# Normalize Dates for Plotting
df_es['date_dt'] = pd.to_datetime(df_es['date'].apply(lambda x: x.split('_')[0]))
df_multi['date_dt'] = pd.to_datetime(df_multi['date'].apply(lambda x: x.split('_')[0]))

print(f"Loaded: Spanish ({len(df_es)} windows) vs Multilingual ({len(df_multi)} windows)")
    """))
    
    # --- DRIFT COMPARISON ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 1. Semantic Stability (Drift)
Drift measures how much the meaning changes from one window to the next ($1 - \text{CosineSim}(S_t, S_{t-1})$).
*   High peaks = Sudden shifts in meaning.
*   We expect the Spanish model to be more stable if it captures true semantic shifts rather than noise.
    """))
    
    nb.cells.append(nbf.v4.new_code_cell("""
# Merge on date to align
df_merged = pd.merge(df_es, df_multi, on='date_dt', suffixes=('_es', '_multi'), how='outer')
df_merged = df_merged.sort_values('date_dt')

fig, ax = plt.subplots(figsize=(12, 6))

# X Axis from merged
x_vals = np.arange(len(df_merged))
labels = df_merged['date_dt'].dt.strftime('%b %y')

ax.plot(x_vals, df_merged['drift_es'], marker='o', label='Spanish (SOTA)', color='#2c3e50')
ax.plot(x_vals, df_merged['drift_multi'], marker='x', linestyle='--', label='Multilingual', color='#e74c3c')

ax.set_xticks(x_vals[::3])
ax.set_xticklabels(labels[::3], rotation=45)
ax.set_title("Comparative Semantic Drift")
ax.set_ylabel("Drift Magnitude")
ax.legend()
plt.tight_layout()
plt.show()
    """))
    
    # --- DIMENSIONALITY ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 2. Intrinsic Dimensionality (Complexity)
How *rich* is the meaning?
*   Higher $k$ = More nuances/senses detected.
*   Multilingual models might have higher 'noise' dimensionality due to cross-lingual interference.
    """))
    
    nb.cells.append(nbf.v4.new_code_cell("""
fig, ax = plt.subplots(figsize=(12, 6))

# Use merged DF
if 'intrinsic_dimension_k_es' in df_merged.columns:
    ax.plot(x_vals, df_merged['intrinsic_dimension_k_es'], marker='o', label='Spanish (SOTA)', color='purple')
    ax.plot(x_vals, df_merged['intrinsic_dimension_k_multi'], marker='s', linestyle='--', label='Multilingual', color='orange')
    
    ax.set_xticks(x_vals[::3])
    ax.set_xticklabels(labels[::3], rotation=45)
    ax.set_title("Intrinsic Dimensionality Evolution")
    ax.set_ylabel("Latent Dimensions (k)")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Intrinsic Dimension column not found in results.")
    """))
    
    # --- PROJECTION COMPARISON ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 3. Projection on 'Functional' Dimension
How strongly does Yape project onto the 'Functional' (Utility/Payment) axis?
    """))
    
    nb.cells.append(nbf.v4.new_code_cell("""
fig, ax = plt.subplots(figsize=(12, 6))

col = 'score_centroid_funcional_contextual'
col_es = f'{col}_es'
col_multi = f'{col}_multi'

if col_es in df_merged.columns:
    ax.plot(x_vals, df_merged[col_es], marker='o', label='Spanish (SOTA)', color='#004e66')
    ax.plot(x_vals, df_merged[col_multi], marker='x', linestyle='--', label='Multilingual', color='#d14a2b')
    
    ax.set_xticks(x_vals[::3])
    ax.set_xticklabels(labels[::3], rotation=45)
    ax.set_title("Projection: Functional Dimension (Contextual)")
    ax.set_ylabel("Cosine Similarity")
    ax.legend()
    plt.tight_layout()
    plt.show()
    """))

    os.makedirs('academic', exist_ok=True)
    with open('academic/Reporte_Comparativo_Modelos.ipynb', 'w') as f:
        nbf.write(nb, f)
        
    print("Comparison Notebook created: academic/Reporte_Comparativo_Modelos.ipynb")

if __name__ == "__main__":
    create_comparison_notebook()
