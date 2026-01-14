# LISBETH: De la Billetera Móvil al Actor Social
## Análisis Computacional de la Representación Mediática de Yape en el Perú

**TFM - Máster en Big Data y Data Science | UNED**
* **Investigador**: Alejandro Mingo
* **Proyecto**: `LISBETH` (Legitimacy & Identity Semantic BERT Embedding Time-series Harvester)

---

## 📖 Descripción del Proyecto

**Lisbeth** es un sistema de investigación computacional ("Laboratorio") diseñado para analizar la evolución semántica de la aplicación "Yape" en la prensa peruana (2016-2023). El sistema combina técnicas avanzadas de **NLP (Modelos Transformadores Adaptados al Dominio)** con **Sociología Digital** para cuantificar cómo la marca ha transitado de ser una herramienta financiera a un "Actor Social" legítimo.

El núcleo metodológico reside en la corrección de la **Anisotropía** del espacio vectorial y el análisis de **Subespacios Semánticos** dinámicos, permitiendo medir matemáticamente conceptos abstractos como la "Deriva Semántica" y la "Proyección Sociológica".

---

## 🏗️ Arquitectura y Fases del Proyecto

El sistema se orquesta mediante una CLI maestra: `pipeline_manager.py`.

### ✅ Fase 1: Data Harvesting (Recolector Granular)
*Infraestructura de recolección de noticias resiliente.*

*   **Estrategia "Day x Media"**: A diferencia de scrapers tradicionales que hacen consultas masivas, Lisbeth itera **día por día** y **medio por medio** (ej. "Solo El Comercio el 12/03/2020"). Esto bypass-ea las limitaciones de retorno de GDELT (max 250 registros) y asegura una completitud histórica cercana al 100%.
*   **Fuentes Híbridas**: GDELT (primaria), Google News (backup), RSS (tiempo real).
*   **Resiliencia**:
    *   Manejo de "Soft 404s" y contenido renderizado por JS (Client-Side) mediante selectores CSS específicos por dominio (`src/news_harvester/domains.py`).
    *   Fallback automático a la librería `trafilatura` para extracción de texto limpio.

### ✅ Fase 2: Infraestructura NLP (La "Fábrica de Embeddings")
*Transformación de texto en tensores matemáticos ajustados.*

#### 2.1 Model Management
El sistema soporta cualquier modelo de Hugging Face, pero está optimizado para modelos monolingües en español:
*   **`PlanTL-GOB-ES/roberta-large-bne`**: SOTA (State of the Art) entrenado por la Biblioteca Nacional de España.
*   **`dccuchile/bert-base-spanish-wwm-uncased`** (BETO): Alternativa robusta y ligera.

#### 2.2 DAPT (Domain-Adaptive Pretraining)
Antes de extraer embeddings, el modelo base se somete a un "re-entrenamiento" ligero (**DAPT**) utilizando el corpus recolectado en Fase 1.
*   **Por qué**: Un modelo genérico no entiende que "Yapear" es un verbo o que "Plin" es un competidor, no un sonido.
*   **Parámetros**:
    *   MLM (Masked Language Modeling): Se ocultan aleatoriamente palabras del corpus peruano y el modelo aprende a predecirlas.
    *   Epochs: Configurable (default 3).

#### 2.3 Extracción de Embeddings Contextuales
Para cada mención de la palabra clave (ej. "Yape"):
1.  **Tokenización**: Se localiza la palabra en la oración. Si se fragmenta en sub-tokens (`['Yap', '##ear']`), se aplica **Mean Pooling** para obtener un único vector.
2.  **Layer Strategy**: Se extraen las activaciones ocultas.
    *   **`penultimate`**: La capa anterior a la última (mejor para representaciones geométricas generales).
    *   **`last4_concat`**: Concatenación de las últimas 4 capas (4096 dims para RoBERTa-large), capturando matices sintácticos y semánticos profundos.

### ✅ Fase 3: Análisis de Subespacios (El "Laboratorio Matemático")
*Donde ocurre la magia sociológica.*

#### 3.1 Dual Anisotropy Correction
Los modelos de lenguaje sufren de "Anisotropía": todos los vectores tienden a ocupar un cono estrecho en el espacio, distorsionando las distancias (coseno).
Lisbeth implementa un protocolo estricto de comparación:
1.  **RAW (Crudo)**: Embeddings tal cual salen del modelo.
2.  **CORRECTED (Corregido)**: Se calcula el **Vector Medio Global** ($\mu_{global}$) de todo el corpus y se resta de cada embedding ($v' = v - \mu_{global}$). Esto "centra" la nube de puntos y revela la verdadera estructura semántica interna.

#### 3.2 Subespacios Dinámicos
Se agrupan los embeddings en **Ventanas Deslizantes** (ej. Trimestrales) y se aplica **SVD (Singular Value Decomposition)** para hallar los ejes principales de significado en ese periodo.

#### 3.3 Métricas
*   **Semantic Drift**: Distancia Grassmanniana entre el subespacio del tiempo $t$ y el tiempo $t+1$. Mide cuánto ha cambiado el significado.
*   **Entropía**: Dispersión de los valores singulares. Alta entropía = Significado difuso/polisémico.
*   **Proyección de Anclas**: Se definen vectores teóricos (ej. "Seguridad", "Comunidad") y se mide matemáticamente cuánto se acerca el concepto "Yape" a ellos.

### ✅ Fase 4: Reportes Automáticos
Generación de Notebooks y Gráficos (Heatmaps, Series Temporales) que comparan visualmente las condiciones RAW vs CORRECTED para validar los hallazgos.

---

## 🚀 Guía Exhaustiva de Parámetros y Ejecución

El script `pipeline_manager.py` es el punto de entrada único.

### 0. Configuración Inicial
```bash
# Definir lista de medios (disponible en repo)
cat data/media_list.csv
# name,domain,type
# elcomercio,elcomercio.pe,national
# ...
```

### 1. Descarga de Modelos
Pre-descarga los modelos para evitar latencia o errores de red durante el proceso.
```bash
python pipeline_manager.py phase2 download-models \
    --models "dccuchile/bert-base-spanish-wwm-uncased" "PlanTL-GOB-ES/roberta-large-bne"
```

### 2. Fase 1: Recolección (Harvesting)
**Parámetros Clave**:
*   `--pipeline granular`: (Implícito en lógica interna) Activa el loop "Day x Media".
*   `--media-list`: Ruta al CSV de medios. Si se omite, busca en todo GDELT (menos exhaustivo).
*   `--keyword`: Palabras a rastrear.

```bash
python pipeline_manager.py phase1 \
    --keyword "Yape" "Yapear" \
    --from 2020-01-01 --to 2021-01-01 \
    --media-list data/media_list.csv \
    --output data/raw_news_2020.csv
```

### 3. Fase 2: Procesamiento NLP

#### Paso 3.1: DAPT (Opcional pero Recomendado)
Entrena el modelo base sobre tu data.
*   `--model`: Modelo base de HuggingFace.
*   `--epochs`: 3 suele ser suficiente para adaptación ligera.

```bash
python pipeline_manager.py phase2 dapt \
    --data data/raw_news_2020.csv \
    --output models/lisbeth-adapted-2020 \
    --model "dccuchile/bert-base-spanish-wwm-uncased" \
    --epochs 3
```

#### Paso 3.2: Extracción
Genera el dataset vectorial.
*   `--dapt_model`: Ruta al modelo entrenado en 3.1.
*   `--model`: Modelo base (se usa para generar la línea base comparativa).

```bash
python pipeline_manager.py phase2 extract \
    --data_dir data/raw_news_dir_2020 \
    --output data/embeddings_2020.csv \
    --model "dccuchile/bert-base-spanish-wwm-uncased" \
    --dapt_model models/lisbeth-adapted-2020
```

### 4. Fase 3: Análisis de Subespacios
Ejecuta el cálculo masivo de métricas. No requiere parámetros complejos, ya que la configuración científica (ventanas, anclas, estrategias) se define en `src/phase3/schemas.py` o se infiere.
*   **Output**: Genera una estructura de carpetas `artifacts/` con subespacios `.npz` y un CSV resumen `phase3_results.csv`.

```bash
python pipeline_manager.py phase3 \
    --input data/embeddings_2020.csv \
    --output-dir results/analysis_2020
```

### 5. Fase 4: Reporte
Genera el entregable final.
*   Crea un Notebook de Jupyter (`report.ipynb`) en la carpeta de destino con todas las gráficas pre-cargadas.

```bash
python pipeline_manager.py phase4 \
    --input results/analysis_2020/phase3_results.csv \
    --output_dir results/final_report_2020
```

---

## 📂 Estructura del Repositorio

```
LISBETH/
├── academic/               # Templates de reportes metodológicos
├── data/                   # Datos (Gitignored, salvo media_list.csv)
│   └── media_list.csv      # Catálogo de medios peruanos
├── execution_test/         # Artefactos de validación (Run de prueba)
├── notebooks/              # Demos interactivos
├── models/                 # Modelos (Gitignored)
├── scripts/                # Utilidades (Generator de assets)
├── src/                    # Código Fuente
│   ├── news_harvester/     # Lógica scraping (Domains, Selectors)
│   ├── nlp/                # Lógica DAPT y tensores
│   ├── phase3/             # Matemáticas (SVD, Grassman, Procrustes)
│   └── phase4/             # Reporting logic
├── pipeline_manager.py     # CLI Maestro
└── README.md               # Este archivo
```

---
**Lisbeth v2.0 - Enero 2026**
