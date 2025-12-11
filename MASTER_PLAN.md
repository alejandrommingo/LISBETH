# Plan Maestro: Análisis de Yape como Actor Social

Este documento establece la hoja de ruta integral para la investigación "De la billetera móvil al actor social", detallando roles, fases y tareas específicas para garantizar una ejecución robusta y científicamente válida.

## 1. Definición de Roles y Responsabilidades

Para asegurar la calidad en cada etapa del pipeline, definimos los siguientes roles lógicos (que pueden ser ejecutados por una o varias personas/agentes):

### 🛠️ Role: Data Engineer (Ingeniero de Datos)
**Responsabilidad**: Garantizar la disponibilidad, calidad y completitud del corpus de noticias.
*   Mantenimiento de `Lisbeth News Harvester`.
*   Implementación de soporte para múltiples queries.
*   Limpieza y preprocesamiento de texto (normalización, eliminación de ruido).
*   Gestión del almacenamiento de datos y embeddings.

### 🧠 Role: NLP Engineer (Ingeniero de PNL)
**Responsabilidad**: Transformar texto en representaciones matemáticas precisas.
*   Selección y validación de modelos de lenguaje (BERT/RoBERTa).
*   Implementación del pipeline de extracción de embeddings contextuales (Token-level).
*   Fine-tuning de modelos (si fuera necesario por especificidad del dominio).

### 📐 Role: Data Scientist (Científico de Datos)
**Responsabilidad**: Modelado matemático y cálculo de métricas.
*   Implementación de algoritmos de reducción de dimensionalidad (PCA/SVD).
*   Ejecución de Análisis Paralelo de Horn y Bootstrapping para estabilidad.
*   Cálculo de métricas complejas (Deriva Semántica, Entropía, Proyección).

### 🔎 Role: Researcher (Investigador Principal)
**Responsabilidad**: Definición teórica e interpretación de resultados.
*   Definición de "Vectores Ancla" (listas de palabras semilla para cada frame).
*   Validación semántica de los subespacios hallados.
*   Redacción de hallazgos y vinculación con la teoría.

---

## 2. Plan de Ejecución Detallado

### Fase 1: Expansión y Refinamiento de Datos (Data Engineer)
**Objetivo**: Capturar todas las variaciones de la marca para no perder data relevante.

*   [x] **Implementar Multi-Query Support**: Modificar el harvester para aceptar listas de keywords (`["Yape", "Yapear", "Yapeo", "Yapame"]`).
*   [x] **Implementar Daily Chunking**: Modificar el harvester para que **siempre** divida las consultas en intervalos diarios (día a día) para asegurar la máxima exhaustividad, independientemente del rango total.
*   [x] **Solución Ligera para Renderizado JS**: Implementar un mecanismo (ej. Playwright o similar optimizado) para extraer texto de medios con Client-Side Rendering (La República, etc.) que actualmente devuelven "200 OK" pero sin contenido. **Crítico antes de avanzar**.
*   [x] **Recolección Histórica Completa (Re-run)**: Ejecutar barrido **1 Enero 2020 - 1 Enero 2021** con Daily Chunking y soporte JS.

### Fase 2: Infraestructura NLP (NLP Engineer)
**Objetivo**: Convertir el corpus en una base de datos de vectores contextuales robustos.

*   [x] **Selección de Modelo Principal**: Utilizar `PlanTL-GOB-ES/roberta-large-bne` (Encoder Bidireccional SOTA en español).
    *   **Modelo de Contraste**: Usar `bertin-project/bertin-roberta-base-spanish` para validar robustez.
    *   **Nota**: Se descarta GPT (Decoder) por inadecuación para representación semántica bidireccional.
*   [x] **Adaptación al Dominio (DAPT)**: Realizar *Continued Pretraining* del modelo sobre el corpus de prensa peruana para mejorar la representación de terminología local ("Yapear", "Plin").
*   [x] **Pipeline de Tokenización**: Implementar estrategia de **Subword Pooling** (promedio de sub-tokens) para manejar correctamente la fragmentación (ej. `['Yape', '##ar']` -> `Yapear`).
*   [x] **Extracción de Embeddings**: Generar tensores para cada ocurrencia de la marca.
    *   **Estrategia**: Concatenar últimas 4 capas (o investigar penúltima) + Normalización (Whitening/Centering).
*   [x] **Almacenamiento**: Guardar los vectores con metadatos (fecha, medio, oración original) en formato eficiente (ej. Parquet).

### Fase 3: Análisis de Subespacios Semánticos (En Progreso)
**Objetivo**: Modelar la evolución del significado de la marca a lo largo del tiempo mediante técnicas algebraicas.

#### Sub-fase 3.1: Estrategia de Segmentación Temporal (Data Scientist)
**Objetivo**: Preparar los datos para un análisis evolutivo robusto, evitando el ruido de las fluctuaciones diarias.
*   [ ] **Implementar Rolling Windows**: Crear generador de ventanas deslizantes configurables (ej. Tamaño: 3 meses, Paso: 1 mes) para suavizar tendencias.
*   [ ] **Filtrado Dinámico de Vocabulario**: Asegurar que solo términos relevantes y persistentes en la ventana temporal sean considerados (min_frequency per window).
*   [ ] **Validación de Densidad**: Verificar que cada ventana tenga suficiente densidad de "keywords" para un análisis estadísticamente significativo.

#### Sub-fase 3.2: Análisis de Estabilidad y Dimensionalidad (Data Scientist)
**Objetivo**: Determinar matemáticamente cuántas dimensiones ($k$) son necesarias para representar la realidad latente sin sobreajuste.
*   [ ] **Análisis Paralelo de Horn**: Implementar test de permutación para distinguir señal de ruido aleatorio.
*   [ ] **Bootstrapping de Estabilidad**: Evaluar la robustez de los autovalores mediante remuestreo con reemplazo.
*   [ ] **Selección de $k$ Óptimo**: Definir criterio de corte automático para cada ventana temporal.

## 3.3 Subspace Construction (Data Scientist)
- [ ] **Architecture Refactor**: Split into `scripts/run_phase3_pipeline.py` (CLI) and `notebooks/phase3_analysis.ipynb` (Viewer).
- [ ] Implement SVD decomposition on centered embeddings <!-- id: 45 -->
- [ ] Implement Orthogonal Procrustes for temporal alignment <!-- id: 46 -->
- [ ] Validate alignment stability with synthetic data <!-- id: 47 -->

## 3.4 Sociological Metrics (Researcher + Data Scientist)
- [ ] **Methodology Upgrade**: Implement **Gram-Schmidt Orthogonalization** for Anchors (`metrics.py`).
- [ ] Calculate Semantic Drift (Cosine Distance $t$ vs $t+1$) <!-- id: 47 -->
- [ ] Calculate Theoretical Projections (Heatmap of Basis vs Orthogonal Anchors) <!-- id: 48 -->
- [ ] Calculate Semantic Entropy (Volume of meaning) <!-- id: 49 -->
la "ambigüedad" o "riqueza" del significado.
*   [ ] **Proyección de Marcos (Frame Projection)**: Proyectar los vectores de la marca sobre los ejes definidos por las Anclas Contextuales (Confianza, Inclusión, Riesgo) extraídas en Fase 2.

### Fase 4: Interpretación y Visualización (Researcher + Data Scientist)
**Objetivo**: Traducir números a narrativa sociológica.

*   [ ] **Definición de Anclas**: Implementar **Estrategia Híbrida**:
    *   *Baseline*: Vectores estáticos (Capa 0) de las keywords.
    *   *Contextual*: Centroides de "Oraciones Prototípicas" (del archivo `data/dimensiones_ancla.json`) procesadas por el modelo (últimas 4 capas).
*   [ ] **Validación ("Ground Truth")**: Etiquetado manual de un subconjunto de datos para validar alineamiento de frames y valencia.
*   [ ] **Visualización Evolutiva**: Graficar métricas con bandas de error (IC).
*   [ ] **Redacción de Resultados**: Integrar hallazgos validando hipótesis.

---

## 3. Próximos Pasos Inmediatos (Sprint Actual)

1.  **Data Engineer**: Actualizar `Lisbeth` para soportar múltiples keywords (ej. "Yape", "Yapear").
2.  **Researcher**: Validar la lista de variantes de la marca a rastrear.
