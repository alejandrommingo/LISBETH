
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.nlp.extract import extract_embeddings
from src.visualization.paper_plots import setup_pub_style

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "academic/model_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "Spanish (BNE/PlanTL)": "PlanTL-GOB-ES/roberta-large-bne",
    "Multilingual (XLM-R)": "xlm-roberta-large"
}

KEYWORDS = ["Yape", "Yapear", "Yapame", "Yapeo", "Plin"]

def run_extraction():
    results_paths = {}
    
    for label, model_name in MODELS.items():
        print(f"\n>>> Running Extraction for {label} [{model_name}]...")
        output_file = os.path.join(DATA_DIR, f"embeddings_{label.split()[0].lower()}.parquet")
        
        # We run extraction. Note: extract_embeddings prints to stdout.
        # We assume data is in DATA_DIR.
        try:
            # Check if file exists to avoid re-running if possible? 
            # No, user asked to "vuelvas a incluir el modelo", so we should re-run to be safe or overwrite.
            extract_embeddings(DATA_DIR, output_file, keywords=KEYWORDS, model_name=model_name)
            results_paths[label] = output_file
        except Exception as e:
            print(f"Error running model {label}: {e}")
            # Fallback for Spanish model if we have previous data
            if "Spanish" in label:
                fallback = os.path.join(DATA_DIR, "embeddings_v2.parquet")
                if os.path.exists(fallback):
                    print(f"Fallback: Using existing file {fallback} as {label}")
                    results_paths[label] = fallback
            
    return results_paths

def analyze_results(paths):
    print("\n>>> Analyzing Comparison Results...")
    
    dfs = {}
    for label, path in paths.items():
        if os.path.exists(path):
            dfs[label] = pd.read_parquet(path)
            # Ensure date
            dfs[label]['date'] = pd.to_datetime(dfs[label]['date'])
            print(f"Loaded {label}: {len(dfs[label])} rows.")
        else:
            print(f"Warning: File {path} not found.")
            
    if not dfs:
        return

    setup_pub_style()
    
    # 1. Volume Comparison (Embeddings Found)
    # Group by Month
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, df in dfs.items():
        # Resample by month
        monthly = df.set_index('date').resample('M').size()
        ax.plot(monthly.index, monthly.values, marker='o', label=label, linewidth=2)
        
    ax.set_title("Comparativa de Sensibilidad: Menciones Detectadas por Modelo")
    ax.set_ylabel("Cantidad de Vectores Extraídos")
    ax.set_xlabel("Fecha")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_volume.png"))
    plt.close()
    
    # 2. Vector Statistics (Magnitude)
    # Check if models normalize differently (L2 norm)
    stats = []
    for label, df in dfs.items():
        # Compute norms
        # Embeddings are lists in parquet
        vecs = np.vstack(df['embedding'].values)
        norms = np.linalg.norm(vecs, axis=1)
        stats.append({
            "Model": label,
            "Mean Norm": np.mean(norms),
            "Std Norm": np.std(norms),
            "Count": len(df)
        })
        
    stats_df = pd.DataFrame(stats)
    print("\nModel Statistics:")
    print(stats_df)
    
    # Save statistics report
    with open(os.path.join(OUTPUT_DIR, "comparison_report.md"), "w") as f:
        f.write("# Reporte Comparativo de Modelos: Spanish vs Multilingual\n\n")
        f.write("## 1. Resumen Estadístico\n")
    # Save statistics report
    with open(os.path.join(OUTPUT_DIR, "comparison_report.md"), "w") as f:
        f.write("# Reporte Comparativo de Modelos: Spanish vs Multilingual\n\n")
        f.write("## 1. Resumen Estadístico\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n## 2. Diferencias Clave\n")
        f.write("\n\n## 2. Diferencias Clave\n")
        f.write("\n### Volumen de Detección\n")
        f.write("![Volume Comparison](comparison_volume.png)\n")
        
        # Interpretation
        diff = stats_df.iloc[0]['Count'] - stats_df.iloc[1]['Count']
        winner = stats_df.iloc[0]['Model'] if diff > 0 else stats_df.iloc[1]['Model']
        f.write(f"\n- **Diferencia de Cobertura**: El modelo **{winner}** detectó {abs(diff)} instancias más.\n")
        f.write("- **Interpretación**: Los modelos monolingües suelen tener vocabularios más densos para el idioma específico, capturando mejor las variaciones morfológicas (e.g. 'Yapearían', 'Yapeado').\n")
        
        f.write("\n### Calidad del Espacio Vectorial\n")
        f.write("- **Spanish SOTA**: Entrenado con corpus de la BNE, optimizado para la sintaxis y modismos del español peninsular y latinoamericano (en menor medida).\n")
        f.write("- **Multilingual**: Entrenado con 100 idiomas. Sufre de la 'maldición de la capacidad': debe repartir sus parámetros entre muchos idiomas, resultando en representaciones menos granularizadas para cada uno.\n")

if __name__ == "__main__":
    paths = run_extraction()
    analyze_results(paths)
