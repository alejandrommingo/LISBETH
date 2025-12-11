# LISBETH: De la Billetera Móvil al Actor Social
## Análisis Computacional de la Representación Mediática de Yape en el Perú

**TFM - Máster en Big Data y Data Science | UNED**
* **Investigador**: Alejandro Mingo
* **Proyecto**: `LISBETH` (Legitimacy & Identity Semantic BERT Embedding Time-series Harvester)

---

## 📖 Descripción del Proyecto

**Lisbeth** es un sistema de investigación computacional diseñado para analizar cómo la aplicación "Yape" ha trascendido su función financiera para convertirse en un **Actor Social** en la cultura peruana. 

El proyecto combina **Sociología Digital** y **Procesamiento de Lenguaje Natural (NLP)** para rastrear la evolución semántica de la marca en la prensa nacional (2016-2023), identificando cómo los medios construyen y transforman su legitimidad (de la "innovación funcional" a la "solidaridad cotidiana").

---

## 🏗️ Arquitectura y Fases del Proyecto

El desarrollo se estructura en fases secuenciales que transforman datos no estructurados en conocimiento sociológico.

### ✅ Fase 1: Data Harvesting (Recolector de Noticias)
*Infraestructura de recolección masiva y curación de corpus.*

*   **Fuentes Híbridas**: Integración de **GDELT** (histórico profundo), **Google News** y **RSS** directos.
*   **Cobertura**: +30 medios peruanos (El Comercio, La República, Gestión, RPP, etc.).
*   **Capacidades Técnicas**:
    *   **Multi-Keyword Targeting**: Rastreo simultáneo de variantes (`Yape`, `Yapear`, `Yapeo`, `Plin`).
    *   **Daily Chunking**: Algoritmo de segmentación diaria para maximizar la recuperación de datos históricos (superando límites de API).
    *   **WAF Bypass**: Navegación simulada para extraer contenido de sitios protegidos (Client-Side Rendering).
    *   **Relevance Scoring**: Clasificación automática de artículos según la densidad terminológica.

### ✅ Fase 2: Infraestructura NLP
*Adaptación de modelos y vectorización semántica.*

*   **Core Model**: Modelos Transformadores del Estado del Arte (SOTA) en español (`PlanTL-GOB-ES/roberta-large-bne` o `xlm-roberta`).
*   **DAPT (Domain-Adaptive Pretraining)**: Re-entrenamiento del modelo base con el corpus periodístico peruano recolectado para "enseñarle" terminología local y jerga financiera específica.
*   **Subword Mean Pooling**: Estrategia matemática para reconstruir vectores de palabras fragmentadas por el tokenizador (ej: `['Yap', '##ear']` $\rightarrow$ `Yapear`).
*   **Extracción de Embeddings Contextuales**: Generación de representaciones vectoriales densas para cada ocurrencia de la marca, capturando el significado exacto según su contexto de uso.

### 🚧 Fase 3: Análisis de Subespacios Semánticos (En Progreso)
*Modelado matemático de la evolución.*
*   Análisis de Componentes Principales (PCA) y SVD sobre ventanas temporales.
*   Detección de Deriva Semántica (*Semantic Drift*).
*   Proyección de Marcos Teóricos (Confianza, Inclusión, Riesgo).

---

## 🚀 Guía de Uso Rápida

### 1. Instalación
```bash
git clone https://github.com/alejandrommingo/LISBETH.git
cd LISBETH
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Recolección de Datos (Harvester)
Descargar noticias históricas de medios peruanos:
```bash
# Ejemplo: Descargar noticias de 2020 a 2021 sobre Yape
PYTHONPATH=src python -m news_harvester prototype \
    --keyword "Yape" "Yapear" \
    --from 2020-01-01 --to 2021-01-01 \
    --media all \
    --output data/yape_2020.csv
```

### 3. Pipeline NLP
Ejecutar las herramientas de procesamiento de lenguaje:

**A. Adaptación al Dominio (DAPT):**
Entrenar el modelo con el texto descargado para mejorar su comprensión:
```bash
python src/cli.py dapt --data data/corpus.txt --output models/lisbeth-roberta-adapted --epochs 3
```

**B. Extracción de Embeddings:**
Generar la base de datos vectorial para análisis:
```bash
python src/cli.py extract \
    --data_dir data \
    --keywords Yape Yapear Plin \
    --output data/embeddings_final.parquet
```

### 4. Demo Educativa
Explora el funcionamiento interno paso a paso:
```bash
jupyter notebook notebooks/phase2_demo.ipynb
```

---

## 📂 Estructura del Repositorio

```
LISBETH/
├── academic/           # Documentación teórica (TFM Intro, Metdología)
├── data/               # Corpus crudo y Datasets (Ignorados por git)
├── models/             # Checkpoints de modelos NLP (Ignorados por git)
├── notebooks/          # Demos y experimentos (Jupyter)
├── src/
│   ├── data/           # Lógica de scraping y curación
│   ├── nlp/            # Modelos, DAPT y Extracción
│   ├── utils/          # Herramientas auxiliares
│   └── cli.py          # Punto de entrada unificado
├── tests/              # Tests unitarios y de integración
└── README.md           # Documentación del proyecto
```

---

**Estado del Proyecto**: Fase 2 Completada (Diciembre 2025).
