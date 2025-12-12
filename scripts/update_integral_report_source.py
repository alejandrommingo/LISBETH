
import nbformat
import os

NOTEBOOK_PATH = 'academic/Reporte_Integral_TFM.ipynb'

def update_notebook_source():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    print(f"Loading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Update Model Description
    # Find cell describing Phase 2.1
    for cell in nb.cells:
        if "3.  **RoBERTa (Bidireccional)**: **Seleccionado**" in cell.source:
             print("Updating Model Description...")
             new_source = cell.source.replace(
                 "Específicamente `PlanTL-GOB-ES/roberta-large-bne` (SOTA en español).",
                 "Específicamente `dccuchile/bert-base-spanish-wwm-uncased` (BETO, SOTA en español)."
             )
             # Also update justification if needed, but BERT is also bidirectional masked, so the math holds.
             cell.source = new_source

    # 2. Update Data Path
    # Find cell loading 'phase3_results.parquet'
    for cell in nb.cells:
        if "pd.read_parquet('../data/phase3_results.parquet')" in cell.source:
             print("Updating Data Path...")
             cell.source = cell.source.replace(
                 "pd.read_parquet('../data/phase3_results.parquet')",
                 "pd.read_parquet('../data/phase3_results_spanish.parquet')"
             )

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook updated successfully.")

if __name__ == "__main__":
    update_notebook_source()
