# Metodología de Investigación: Representación Mediática de Yape

Este documento define la estrategia metodológica para analizar la evolución de la representación de **Yape** en la prensa peruana, basándose en el marco teórico de "La marca como actor social".

Se propone un enfoque de **Análisis de Subespacios Semánticos** utilizando Modelos de Lenguaje Contextuales (BERT/RoBERTa). A diferencia del enfoque clásico de centroides, este método modela la identidad de la marca como un espacio multidimensional dinámico, capaz de capturar la polisemia y la complejidad del discurso.

## 1. Objetivo General
Analizar cómo los medios peruanos han construido y transformado la identidad de Yape a través de la evolución de sus **subespacios semánticos** en el tiempo, evaluando dimensiones latentes de significado.

## 2. Pipeline de Procesamiento

### Fase 1: Recolección y Preprocesamiento
*   **Corpus**: Noticias de 30+ medios peruanos (2016-Presente), agrupadas por ventanas temporales (ej. mensual o trimestral).
*   **Tokenización**: Identificación de todas las menciones de la marca y sus variantes morfológicas: "Yape", "Yapear", "Yapame", "Yapeo".

### Fase 2: Extracción de Vectores Contextuales (Token Embeddings)
En lugar de vectorizar el documento completo, nos centraremos en el **uso específico** de la marca.
*   **Justificación**: El vector de un documento promedio diluye el significado de la marca en el tema general.
*   **Selección de Modelo**:
    *   **Principal**: `PlanTL-GOB-ES/roberta-large-bne`. Justificación: Arquitectura bidireccional (Encoder) con mayor capacidad de abstracción (1024 dim) y SOTA en español.
    *   **Adaptación (DAPT)**: Se realizará *Domain-Adaptive Pretraining* sobre el corpus de prensa recolectado para adaptar el modelo al idiolecto peruano sin supervisión.
    *   **Contraste**: `bertin-project/bertin-roberta-base-spanish`.
*   **Tratamiento de Subpalabras (Subword Pooling)**: Dado que el tokenizer BPE puede fragmentar "Yapear" en `['Yape', '##ar']`, se implementará una estrategia de **Mean Pooling** de los sub-tokens para reconstruir el vector de la palabra completa antes de la extracción.
*   **Estrategia de Capas**: Se priorizará la **Concatenación de las últimas 4 capas** oculta, o el uso de la penúltima capa, sujeto a validación de anisotropía.
*   **Resultado**: Nube de puntos $X_t$, normalizada (Whitening/Centering).

### Fase 3: Construcción de Subespacios (PCA/SVD)
Para modelar la estructura semántica en el tiempo $t$, aplicamos reducción de dimensionalidad sobre la nube de puntos $X_t$.
*   **Segmentación Temporal**: Se utilizarán **Ventanas Deslizantes** (Rolling Windows, ej. 3 meses con paso de 1 mes) para garantizar densidad de puntos y suavizar transiciones abruptas por falta de datos.
*   **Definición del Subespacio**: El subespacio $S_t$ definido por los $k$ primeros componentes principales (SVD).
*   **Selección de Dimensionalidad ($k$)**:
    *   **Análisis Paralelo de Horn**: Para determinar el umbral de ruido.
    *   **Bootstrap de Subespacios**: Re-muestreo de los datos de cada ventana para estimar la estabilidad de los eigenvectores y construir intervalos de confianza.

## 3. Métricas de Análisis Propuestas

Al trabajar con subespacios, las métricas evolucionan para medir propiedades geométricas y estructurales.


### 3.1. Alineamiento Teórico (Subspace Projection)
*Pregunta: ¿Qué tanto del significado de Yape se explica por los conceptos teóricos?*

*   **Definición**: Se construyen vectores ancla teóricos ($V_{funcional}$, $V_{afectiva}$, $V_{social}$) basándose en el archivo `data/dimensiones_ancla.json`.
*   **Estrategia Híbrida de Cálculo**:
    1.  **Enfoque Estático (Baseline)**: Extracción de embeddings de la Capa 0 (Word Embedding estático) de las palabras clave.
    2.  **Enfoque Contextual (Propuesto)**: Extracción de embeddings contextuales (Concatenación últimas 4 capas) de las palabras clave insertadas en sus "oraciones prototípicas" (definidas en el JSON). El centroide de estos vectores contextuales forma el $V_{dim}$ final.
*   **Interpretación**: Se comparará la proyección del subespacio $S_t$ sobre ambos tipos de vectores ancla para validar la robustez de la dimensión teórica.

### 3.2. Deriva del Subespacio (Grassmannian Distance)
*Pregunta: ¿Cuánto ha cambiado estructuralmente el significado de Yape?*

*   **Definición**: Medida de distancia entre dos subespacios lineales.
*   **Cálculo**: Basado en los **Ángulos Principales** entre el subespacio $S_t$ y $S_{t-1}$. Si los subespacios son idénticos, los ángulos son 0.
*   **Interpretación**: Detecta cambios estructurales profundos (reconfiguración del significado) más allá del simple desplazamiento del centroide.

### 3.3. Complejidad Semántica (Semantic Volume/Entropy)
*Pregunta: ¿Es Yape un concepto monolítico o diverso?*

*   **Cálculo**: Entropía de los valores singulares (eigenvalues) de la descomposición SVD.
*   **Interpretación**:
    *   **Baja Entropía**: Yape se usa de forma muy homogénea (ej. solo como "app de pagos").
    *   **Alta Entropía**: Yape tiene múltiples facetas activas simultáneamente (es "app", es "ayuda", es "verbo", es "riesgo"). Un aumento en la complejidad sugiere que la marca se está convirtiendo en un fenómeno cultural multidimensional.

### 3.4. Valencia en el Subespacio
*Pregunta: ¿Las dimensiones dominantes son positivas o negativas?*

*   **Método**: Proyectar vectores de palabras con carga afectiva clara (listas de léxico positivo/negativo) sobre el primer componente principal de $S_t$.
*   **Interpretación**: Determina si el eje principal de discusión sobre Yape está alineado con el polo positivo o negativo.

### 3.5. Estrategia de Anclaje Híbrido: Ortogonalización Simétrica (Lowdin)
Para definir dimensiones estrictamente ortogonales sin favorecer a ninguna (como hace Gram-Schmidt con la primera), aplicamos **Ortogonalización Simétrica de Löwdin**.

1.  **Matriz de Centroides**: Construimos $\mathbf{C} = [v_F, v_S, v_A]$, donde cada vector es el centroide de sus palabras ancla.
2.  **Matriz de Superposición**: Calculamos $\mathbf{S} = \mathbf{C}^T \mathbf{C}$ (correlaciones entre dimensiones).
3.  **Transformación Ortogonal**: Obtenemos los nuevos anclajes ortogonales $\mathbf{C}_{\perp}$ mediante:
    $$ \mathbf{C}_{\perp} = \mathbf{C} \mathbf{S}^{-1/2} $$
    
**Por qué es mejor**: Esta transformación garantiza que los nuevos ejes ortogonales $\mathbf{C}_{\perp}$ son los **matemáticamente más cercanos posible** a los centroides originales teóricos $\mathbf{C}$ (mínima distancia de Frobenius), preservando el significado de las tres dimensiones equitativamente.

## 4. Ventajas de este Enfoque
1.  **Precisión**: Al usar vectores contextuales del token, aislamos el significado de la marca del ruido del artículo.
2.  **Estructura Latente**: PCA/SVD revela las dimensiones ocultas. Podríamos descubrir que en 2021 surge una dimensión latente que no habíamos teorizado (ej. "Yape como requisito laboral") observando los componentes.
## 5. Validación y Robustez ("Ground Truth")
Para evitar resultados puramente matemáticos sin correlato real, se implementarán:

1.  **Anotación Manual**: Un subconjunto de noticias será etiquetado manualmente por humanos (Funcional/Social/Afectivo + Valencia). Se medirá la capacidad de los subespacios $S_t$ para predecir estas etiquetas (Validación Cruzada).
2.  **Correlación con Eventos**: Se construirá una línea de tiempo de eventos exógenos (Lanzamientos, Caídas de sistema, Bonos del estado) para validar si los cambios bruscos en la Distancia Grassmanniana coinciden con hitos reales del "Actor Social".
