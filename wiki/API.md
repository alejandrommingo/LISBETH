# 🔌 Referencia de API y CLI

La interacción principal con LISBETH se realiza a través de su interfaz de línea de comandos (CLI). El proyecto expone dos puntos de entrada principales: uno para NLP/Análisis y otro específico para la recolección de noticias.

---

## 💻 CLI Principal: `src/cli.py`

Este script orquesta las tareas de la **Fase 2 (NLP)**.

Uso general:
```bash
python src/cli.py [COMMAND] [ARGS]
```

### Comandos

#### `dapt` (Domain Adaptive Pretraining)
Re-entrena un modelo base (BERT/RoBERTa) con el corpus específico del dominio peruano.

*   **Argumentos**:
    *   `--data` (Requerido): Ruta al corpus de texto plano (`.txt`).
    *   `--model`: Modelo base de HuggingFace (Default: `PlanTL-GOB-ES/roberta-large-bne`).
    *   `--output`: Directorio donde guardar el modelo adaptado.
    *   `--epochs`: Número de épocas de entrenamiento (Default: 1).

*   **Ejemplo**:
    ```bash
    python src/cli.py dapt --data data/corpus_peru.txt --output models/ys-roberta
    ```

#### `extract`
Genera embeddings contextuales para una lista de keywords específicas.

*   **Argumentos**:
    *   `--data_dir`: Directorio conteniendo archivos CSV con las noticias.
    *   `--keywords`: Lista de palabras clave a vectorizar (ej. `Yape Yapear`).
    *   `--output`: Archivo de salida (formato Parquet recomendado).
    *   `--model`: Ruta o nombre del modelo a utilizar.

*   **Ejemplo**:
    ```bash
    python src/cli.py extract --keywords Yape Plin --output data/vectors.parquet
    ```

---

## 📰 News Harvester CLI

Herramienta dedicada para la **Fase 1 (Recolección)**. Se ejecuta como un módulo de Python.

Uso general:
```bash
python -m src.news_harvester [COMMAND] [ARGS]
```

### Comandos

#### `prototype`
Ejecuta el pipeline completo de recolección: Busca en GDELT -> Filtra -> Descarga HTML -> Procesa -> Guarda.

*   **Argumentos Clave**:
    *   `--keyword`: Palabras clave de búsqueda.
    *   `--from`, `--to`: Rango de fechas (YYYY-MM-DD).
    *   `--media`: Filtro de medios (`all` o lista específica ej. `elcomercio`).
    *   `--format`: `csv` o `parquet`.
    *   `--skip-html`: Solo descarga metadatos, salta la descarga de cuerpos de noticias (útil para pruebas rápidas).

*   **Ejemplo Completo**:
    ```bash
    python -m src.news_harvester prototype \
      --keyword Yape \
      --from 2021-01-01 --to 2021-01-31 \
      --media all \
      --output data/enero_2021.csv
    ```

#### `fetch`
Solo consulta a la API de GDELT y devuelve metadatos crudos, sin descarga de HTML posterior automática (a menos que se especifique flag). Utilizado para debugging o recolección ligera.

---

## 📦 Estructura de Módulos (Python API)

Si deseas importar LISBETH como una librería en tus scripts o notebooks, estas son las clases principales.

### `src.news_harvester`

*   **`collectors.fetch_articles`**: Función core que consulta GDELT con soporte para *daily chunking* (paginación diaria).
*   **`models.NewsRecord`**: Pydantic model que representa una noticia procesada y lista para análisis. Campos: `date`, `medium`, `title`, `body`, `url`.

### `src.nlp`

*   **`model.LisbethModel`**: Wrapper alrededor de `AutoModel` de HuggingFace. Maneja la tokenización, movimiento a GPU y extracción de capas ocultas (pooling strategy).
*   **`extract.extract_embeddings`**: Función de alto nivel que orquesta la lectura de datos, inferencia y guardado.

### `src.analysis`

*   Contiene lógica para la reducción de dimensionalidad (SVD) y métricas semánticas. (En desarrollo activo).
