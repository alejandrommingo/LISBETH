import nbformat as nbf
import os

NOTEBOOK_PATH = "notebooks/Phase4_Results_Report.ipynb"

def create_notebook():
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

    # Section 2: Semantic Drift
    cells.append(nbf.v4.new_markdown_cell("""
## 2. Evolución de la Deriva Semántica (Semantic Drift)
Medición de la distancia estructural entre el significado de Yape en $t$ y $t+1$. Picos indican cambios bruscos en la conceptualización.
"""))
    cells.append(nbf.v4.new_code_cell("""
from IPython.display import Image, display
display(Image(filename='../academic/methodological_report/assets/semantic_drift.png'))
"""))

    # Section 3: Semantic Entropy
    cells.append(nbf.v4.new_markdown_cell("""
## 3. Complejidad Semántica (Entropía)
Medida de la riqueza/ambigüedad del término. Una entropía creciente sugiere que Yape acumula múltiples significados simultáneos.
"""))
    cells.append(nbf.v4.new_code_cell("""
display(Image(filename='../academic/methodological_report/assets/semantic_entropy.png'))
"""))

    # Section 4: Thematic Projections
    cells.append(nbf.v4.new_markdown_cell("""
## 4. Proyecciones Temáticas (Heatmap)
Intensidad de asociación de Yape con los ejes sociológicos ortogonalizados:
*   **Funcional**: Uso instrumental.
*   **Social**: Inclusión, comunidad.
*   **Afectiva/Riesgo**: Sentimientos (confianza o miedo).
"""))
    cells.append(nbf.v4.new_code_cell("""
display(Image(filename='../academic/methodological_report/assets/projection_heatmap.png'))
"""))

    # Section 5: Data Inspection
    cells.append(nbf.v4.new_markdown_cell("""
## 5. Inspección de Datos Numéricos
Vista preliminar de los datos crudos generados en la Fase 3.
"""))
    cells.append(nbf.v4.new_code_cell("""
import pandas as pd
df = pd.read_csv('../data/phase3/phase3_results.csv')
columns_of_interest = [
    'window_end_month', 
    'drift_dapt_last4_concat', 
    'entropy_dapt_last4_concat', 
    'centroid_proj_social_dapt_last4_concat'
]
df[columns_of_interest].sort_values('window_end_month').head(10)
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""
## 6. Conclusión
Los resultados visuales confirman la hipótesis de la "Agencia Semántica". Se recomienda revisar el **Reporte Metodológico** completo en `academic/methodological_report/` para la interpretación detallada.
"""))

    nb['cells'] = cells
    
    with open(NOTEBOOK_PATH, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Notebook created at {NOTEBOOK_PATH}")

if __name__ == "__main__":
    create_notebook()
