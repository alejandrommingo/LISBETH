import nbformat as nbf
import os
import argparse

def create_notebook(notebook_path, assets_dir_rel, csv_path_rel):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    }
    
    # Section 0: Title and Intro
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""
# Reporte de Resultados - Fase 4: Análisis Metodológico de Yape
**Proyecto**: Lisbeth: De la Billetera Móvil al Actor Social
**Fecha**: Diciembre 2025

Este notebook consolida los resultados cuantitativos y visuales obtenidos tras la ejecución del pipeline matemático de la Fase 3 y la interpretación de la Fase 4.
"""))

    # Section 1: Methodology Recap
    cells.append(nbf.v4.new_markdown_cell("""
## 1. Resumen Metodológico
El análisis se basa en:
*   **Modelo**: `roberta-large-bne` + DAPT (Adaptación al dominio peruano).
*   **Embeddings**: Concatenación de últimas 4 capas.
*   **Subespacios**: SVD dinámica sobre ventanas deslizantes.
*   **Métricas**: Deriva Semántica (Grassmannian), Entropía y Proyección Ortogonal sobre Anclas.
"""))

    # Section 2: Semantic Drift (Comparison)
    cells.append(nbf.v4.new_markdown_cell("""
## 2. Evolución de la Deriva Semántica (Semantic Drift)
Comparativa entre embeddings "Crudos" (Raw) y "Corregidos" (Centrados). La corrección elimina el sesgo isotrópico del modelo base.
"""))
    cells.append(nbf.v4.new_code_cell(f"""
from IPython.display import Image, display
display(Image(filename='{assets_dir_rel}/semantic_drift_comparison.png'))
"""))

    # Section 3: Semantic Entropy (Comparison)
    cells.append(nbf.v4.new_markdown_cell("""
## 3. Complejidad Semántica (Entropía)
Comparativa de la entropía. Una mayor entropía en la condición corregida puede indicar una estructura semántica más rica una vez eliminado el componente común dominante.
"""))
    cells.append(nbf.v4.new_code_cell(f"""
display(Image(filename='{assets_dir_rel}/semantic_entropy_comparison.png'))
"""))

    # Section 4: Thematic Projections (Heatmaps)
    cells.append(nbf.v4.new_markdown_cell("""
## 4. Proyecciones Temáticas (Heatmaps)
Intensidad de asociación con los ejes sociológicos (Funcional, Social, Afectiva). Se muestran ambas condiciones para analizar el impacto de la corrección de anisotropía.
"""))
    cells.append(nbf.v4.new_code_cell(f"""
print("Condición: RAW (Original)")
display(Image(filename='{assets_dir_rel}/projection_heatmap_raw.png'))
print("\\nCondición: CORRECTED (Centrado)")
display(Image(filename='{assets_dir_rel}/projection_heatmap_corrected.png'))
"""))

    # Section 5: Data Inspection
    cells.append(nbf.v4.new_markdown_cell("""
## 5. Inspección de Datos Numéricos
Vista preliminar de los datos crudos generados en la Fase 3.
"""))
    cells.append(nbf.v4.new_code_cell(f"""
import pandas as pd
df = pd.read_csv('{csv_path_rel}')
try:
    cols = [c for c in df.columns if "drift" in c or "entropy" in c or "proj" in c]
    df[['window_end_month'] + cols].sort_values('window_end_month').head(10)
except:
    display(df.head())
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""
## 6. Conclusión
Los resultados visuales confirman la hipótesis de la "Agencia Semántica". Se recomienda revisar el **Reporte Metodológico** completo en `academic/methodological_report/` para la interpretación detallada.
"""))

    nb['cells'] = cells
    
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Notebook created at {notebook_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path for output notebook (.ipynb)")
    parser.add_argument("--assets_dir", required=True, help="Relative path to assets dir from notebook")
    parser.add_argument("--csv_path", required=True, help="Relative path to CSV from notebook")
    args = parser.parse_args()
    
    create_notebook(args.output, args.assets_dir, args.csv_path)

if __name__ == "__main__":
    main()
