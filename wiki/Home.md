# 🏠 LISBETH Wiki

Bienvenidos a la documentación técnica de **LISBETH** (Legitimacy & Identity Semantic BERT Embedding Time-series Harvester).

Este proyecto es parte del TFM **"De la billetera móvil al actor social: Análisis computacional de la representación mediática de Yape en el Perú"**.

---

## 🔭 Visión General

**LISBETH** es un sistema híbrido que combina **Sociología Digital** y **Procesamiento de Lenguaje Natural (NLP)** para analizar cómo un producto financiero (Yape) se transforma en un fenómeno cultural. El sistema ingesta noticias históricas, las procesa semánticamente y modela su evolución a lo largo del tiempo.

### Objetivos Clave
1.  **Recolección Exhaustiva**: Recuperar el registro histórico completo de menciones en prensa (2016-2023).
2.  **Modelado Semántico**: Utilizar Modelos de Lenguaje (LLMs/Transformers) para capturar el significado contextual.
3.  **Sociología Computacional**: Cuantificar conceptos abstractos como "legitimidad", "confianza" y "riesgo".

---

## 🏗️ Arquitectura del Sistema

El flujo de trabajo se divide en 4 fases secuenciales:

```mermaid
graph TD
    subgraph Phase 1: Data Harvesting
        A[GDELT Project] -->|Raw Metadata| B(News Harvester)
        C[Google News] -->|Complementary| B
        B -->|Scraping & Cleaning| D[(Raw Corpus JSON/CSV)]
    end

    subgraph Phase 2: NLP Infrastructure
        D --> E[Domain Adaptation (DAPT)]
        E -->|Fine-tuned Roberta| F[Embedding Extraction]
        F -->|Subword Pooling| G[(Vector Database Parquet)]
    end

    subgraph Phase 3: Semantic Analysis
        G --> H[SVD / PCA Reduction]
        H --> I[Time-series Construction]
        I --> J[Metric Calculation]
        J -->|Semantic Drift / Entropy| K[Analytical Tables]
    end

    subgraph Phase 4: Reporting
        K --> L[Academic Notebook]
        L --> M[Paper / TFM Report]
    end
```

---

## 🛠️ Stack Tecnológico

### Core
*   **Lenguaje**: Python 3.12+
*   **Gestión de Dependencias**: `pip`, `venv`

### Data Engineering (Fase 1)
*   **Ingesta**: `requests`, `feedparser`, `trafilatura` (extracción de texto).
*   **Procesamiento**: `pandas`, `orjson` (JSON rápido).
*   **Fuentes**: API GDELT 2.0, Google News RSS.

### NLP & Machine Learning (Fase 2 & 3)
*   **Modelos**: Hugging Face Transformers (`roberta-large-bne`, `xlm-roberta`).
*   **Deep Learning**: PyTorch.
*   **Álgebra Lineal**: `scikit-learn` (PCA, SVD), `numpy`.

### Análisis y Visualización (Fase 4)
*   **Interactive**: Jupyter Lab.
*   **Plotting**: Matplotlib, Seaborn (para gráficos estáticos de alta calidad).

---

## 📚 Navegación

*   **[Guía de Instalación](Setup.md)**: Configura tu entorno de desarrollo.
*   **[Referencia API](API.md)**: Documentación de comandos CLI y módulos internos.
*   **[Guía de Contribución](Guia_Contribucion.md)**: Estándares de código y workflow.
