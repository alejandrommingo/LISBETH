
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def setup_pub_style():
    """
    Configures matplotlib for publication-quality figures (Nature/Science style).
    """
    plt.style.use('seaborn-v0_8-paper')
    
    # Font settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Figure settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    # Colors
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#004e66', '#d14a2b', '#e5b22b', '#5d7667', '#8c9c90'])

# --- 1. GENERAL PLOTS ---

def _handle_date_axis(ax, df, date_col, categorical=True):
    """
    Helper to handle x-axis formatting.
    If categorical=True, plots against range(N) and labels with date strings.
    If categorical=False, assumes x-axis is datetime and uses DateFormatter.
    """
    if categorical:
        # Create ordinal x-axis
        x_vals = np.arange(len(df))
        # Set ticks
        ax.set_xticks(x_vals)
        # Format labels: shorten if needed
        labels = df[date_col].astype(str).tolist()
        # If too many, slice
        # Relaxed limit to 40 to allow typical 2-year monthly/quarterly data to show fully
        if len(labels) > 40:
             # Keep every Nth label
             n = len(labels) // 40 + 1
             for i in range(len(labels)):
                 if i % n != 0:
                     labels[i] = ""
        ax.set_xticklabels(labels, rotation=45, ha='right')
        return x_vals, labels
    else:
        # Standard Date Axis
        date_form = DateFormatter("%b %y")
        ax.xaxis.set_major_formatter(date_form)
        plt.xticks(rotation=45)
        return df[date_col], None

def plot_news_volume(df, date_col='date', count_col='count', output_path=None):
    """Plots bar chart of news volume per window."""
    setup_pub_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    df = df.sort_values(by=date_col)
    
    # Force categorical for volume to see all bars clearly
    x_vals, _ = _handle_date_axis(ax, df, date_col, categorical=True)
    
    ax.bar(x_vals, df[count_col], color='#2c3e50', alpha=0.7)
    ax.set_title('Distribución de Noticias por Ventana Temporal')
    ax.set_ylabel('Cantidad de Noticias/Vectores')
    ax.set_xlabel('Ventana Temporal')
    
    if output_path: plt.savefig(output_path, bbox_inches='tight')
    plt.show()

# --- 2. PHASE 3: SUBSPACE PLOTS ---

def plot_similarity_matrix(sim_df, title="Matriz de Similitud Temporal (Subspace Overlap)", output_path=None):
    """Plots the window-to-window similarity matrix."""
    setup_pub_style()
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df, cmap='viridis', square=True, vmin=0, vmax=1, cbar_kws={'label': 'Similitud ($Tr(U_i^T U_j)$)'})
    
    plt.title(title)
    plt.xlabel('Ventana Temporal Destino')
    plt.ylabel('Ventana Temporal Origen')
    
    # Use categorical indexing implicitly by Heatmap
    if len(sim_df) > 40:
        ticks = np.arange(0, len(sim_df), 3)
        plt.xticks(ticks + 0.5, sim_df.columns[ticks], rotation=90, fontsize=8)
        plt.yticks(ticks + 0.5, sim_df.index[ticks], rotation=0, fontsize=8)
    else:
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
    plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def plot_complexity_evolution(df, date_col='date', k_col='k', drift_col='drift', output_path=None):
    """Plots stacked evolution of Intrinsic Dimension (K) and Semantic Drift."""
    setup_pub_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    df = df.sort_values(by=date_col)
    x_vals, labels = _handle_date_axis(axes[1], df, date_col, categorical=True) # Handle axis on bottom plot
    
    # Apply same xticks to top plot (hide labels)
    axes[0].set_xticks(x_vals)
    axes[0].set_xticklabels([])
    
    # 1. Intrinsic Dimension (K)
    axes[0].plot(x_vals, df[k_col], marker='o', color='purple', linestyle='-', linewidth=2)
    axes[0].set_title("Evolución de la Complejidad Dimensional ($k$ de Horn)")
    axes[0].set_ylabel("Dimensiones Latentes ($k$)")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Semantic Drift
    axes[1].plot(x_vals, df[drift_col], marker='x', color='crimson', label='Drift (Inestabilidad)')
    axes[1].set_title("Inestabilidad Semántica ($1 - CosineSimilarity_{t, t-1}$)")
    axes[1].set_ylabel("Magnitud del Cambio")
    axes[1].fill_between(x_vals, df[drift_col], alpha=0.1, color='crimson')
    
    plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def plot_projection_comparison(df, metric_prefix='score_centroid_', title_prefix='Proyección', output_path=None):
    """
    Plots comparison between Contextual (Usage) and Static (Dictionary) meanings.
    """
    setup_pub_style()
    dims = ['funcional', 'social', 'afectiva']
    colors = {'funcional': '#004e66', 'social': '#5d7667', 'afectiva': '#d14a2b'}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    
    df = df.sort_values(by='date')
    x_vals, labels = _handle_date_axis(axes[0], df, 'date', categorical=True)
    
    # Ensure ALL axes have labels since they are side-by-side
    for ax in axes:
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels, rotation=45, ha='right')

    for i, dim in enumerate(dims):
        ax = axes[i]
        c = colors[dim]
        
        # Contextual Line
        col_ctx = f'{metric_prefix}{dim}_contextual'
        if col_ctx in df.columns:
            ax.plot(x_vals, df[col_ctx], color=c, linestyle='-', marker='o', markersize=4, label='Contextual (Uso Real)')
            
        # Static Line
        col_sta = f'{metric_prefix}{dim}_static'
        if col_sta in df.columns:
            ax.plot(x_vals, df[col_sta], color='gray', linestyle='--', marker='x', markersize=4, alpha=0.6, label='Estático (Definición)')
            
        ax.set_title(f"Dimensión {dim.capitalize()}")
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_ylabel("Similitud Coseno")
            ax.legend()
            
    plt.suptitle(f"{title_prefix} - Comparativa: Uso vs Diccionario", y=1.05)
    plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def plot_semantic_drift(df, date_col='date', drift_col='drift', events=None, output_path=None):
    """
    Plots drift with events.
    """
    setup_pub_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    df = df.sort_values(by=date_col)
    
    x_vals, _ = _handle_date_axis(ax, df, date_col, categorical=True)
    
    ax.plot(x_vals, df[drift_col], color='#2c3e50', linewidth=1.5, marker='o', markersize=3, label=r'Deriva Semántica ($1 - \cos(S_t, S_{t-1})$)')
    ax.fill_between(x_vals, df[drift_col], alpha=0.1, color='#2c3e50')
    
    # Events mapping needs date matching. 
    # Since visual axis is ordinal, we need to find the index closest to the event date.
    if events:
        y_max = df[drift_col].max()
        # Ensure date_col is datetime objects for comparison
        dates_series = pd.to_datetime(df[date_col]) if not isinstance(df[date_col].iloc[0], (pd.Timestamp, float)) else df[date_col]
        
        for date_str, label in events.items():
            date_obj = pd.to_datetime(date_str)
            # Find closest window
            # We look for window start dates.
            # If date_obj is within range [min, max]
            if dates_series.min() <= date_obj <= dates_series.max():
                # Get closest index
                idx = (dates_series - date_obj).abs().idxmin()
                ax.axvline(idx, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
                ax.text(idx, y_max * 0.95, f' {label}', rotation=90, va='top', fontsize=8, color='#c0392b')

    ax.set_ylabel('Inestabilidad Semántica')
    ax.set_xlabel('Tiempo')
    ax.set_title('Evolución Temporal de la Deriva Semántica')
    
    plt.tight_layout()
    plt.show()

def plot_scree_sequence(eigen_data, title="Evolución de la Estructura Dimensional (Scree Plots)", output_path=None):
    """
    Plots the Scree Plot (Explained Variance) for selected time windows.
    """
    setup_pub_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(eigen_data) > 4:
        indices = np.linspace(0, len(eigen_data)-1, 4, dtype=int)
        selection = [eigen_data[i] for i in indices]
    else:
        selection = eigen_data
        
    for item in selection:
        date_label = str(item['date'])
        sv = np.array(item['eigenvalues'])
        variance = (sv ** 2) / np.sum(sv ** 2)
        cum_var = np.cumsum(variance)
        
        ax.plot(range(1, len(cum_var)+1), cum_var, marker='.', label=f'{date_label}')
        
    ax.set_xlabel('Número de Componentes (Dimensiones)')
    ax.set_ylabel('Varianza Explicada Acumulada')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color='gray', linestyle=':', label='90% Varianza')
    ax.legend()
    
    plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight')
    plt.show()
