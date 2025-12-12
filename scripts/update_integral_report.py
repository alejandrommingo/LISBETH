
import nbformat
import os

NOTEBOOK_PATH = 'academic/Reporte_Integral_TFM.ipynb'

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    print(f"Loading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Define new cells
    md_cell = nbformat.v4.new_markdown_cell("""
## FASE 5: Validación Cruzada de Modelos (Cross-Model Validation)

Para garantizar que los hallazgos no son artefactos del modelo de lenguaje, replicamos el análisis completo utilizando dos arquitecturas distintas:
1.  **Spanish SOTA**: `dccuchile/bert-base-spanish-wwm-uncased` (Específico para español).
2.  **Multilingual**: `xlm-roberta-large` (Masivo, multilingüe).

### Comparativa de Deriva Semántica (Drift)
Observamos que ambos modelos capturan las mismas tendencias macro (picos en pandemia), pero el modelo específico (Spanish) muestra una mayor sensibilidad a matices locales, resultando en una deriva más suave y coherente.
""")

    code_cell = nbformat.v4.new_code_cell("""
# Cargar resultados comparativos
try:
    df_spanish = pd.read_parquet('../data/phase3_results_spanish.parquet')
    df_multi = pd.read_parquet('../data/phase3_results_multilingual.parquet')

    # Alinear por fecha
    # Note: 'date' column in phase3_results is like '2020-01_2020-04'
    # We use the raw string for matching
    df_merged = pd.merge(df_spanish, df_multi, on='date', suffixes=('_es', '_multi'), how='inner')
    
    # Sort by the start date part
    df_merged['sort_key'] = df_merged['date'].apply(lambda x: x.split('_')[0])
    df_merged = df_merged.sort_values('sort_key')

    # Plot Comparativo
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    x_vals = range(len(df_merged))
    
    ax.plot(x_vals, df_merged['drift_es'], marker='o', label='Spanish SOTA (BERT)', color='#2c3e50')
    ax.plot(x_vals, df_merged['drift_multi'], marker='x', linestyle='--', label='Multilingual (XLM-R)', color='#e74c3c')
    
    ax.set_xticks(x_vals[::3])
    ax.set_xticklabels(df_merged['date'].iloc[::3], rotation=45)
    ax.set_title("Validación Cruzada: Deriva Semántica por Arquitectura")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"No se pudo generar el gráfico comparativo: {e}")
""")

    # Find insertion point: Before "5. Discusión y Trabajo Futuro"
    insert_idx = -1
    for i, cell in enumerate(nb.cells):
        if "## 5. Discusión y Trabajo Futuro" in cell.source:
            insert_idx = i
            break
    
    if insert_idx != -1:
        print(f"Inserting cells at index {insert_idx}...")
        nb.cells.insert(insert_idx, md_cell)
        nb.cells.insert(insert_idx + 1, code_cell)
        
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Notebook updated successfully.")
    else:
        print("Target section not found. Appending to end (before last cell?).")
        nb.cells.append(md_cell)
        nb.cells.append(code_cell)
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

if __name__ == "__main__":
    update_notebook()
