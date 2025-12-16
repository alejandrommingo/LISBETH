# 2. Arquitectura de Procesamiento de Lenguaje Natural (NLP)

Este núcleo del sistema transforma el texto no estructurado en representaciones matemáticas densas.

## 2.1 Selección del Modelo: RoBERTa-Large-BNE

Se seleccionó el modelo `PlanTL-GOB-ES/roberta-large-bne` como columna vertebral del sistema.

*   **Justificación**: Entrenado por la Biblioteca Nacional de España con un corpus masivo y de alta calidad.
*   **Ventaja**: Arquitectura *Large* (24 capas, 1024 dimensiones) ofrece una capacidad de abstracción superior a BERT-Base para capturar matices sutiles del lenguaje periodístico.

## 2.2 Adaptación al Dominio (DAPT)

El modelo base, aunque potente, carecía de conocimiento específico sobre la terminología fintech peruana (e.g., "yapear" como verbo, "plin" como competidor).

### Proceso de Continued Pretraining
Se sometió el modelo a una fase adicional de entrenamiento (Masked Language Modeling - MLM) utilizando exclusivamente el corpus recolectado de prensa peruana.
*   **Resultado**: El modelo adaptado (DAPT) muestra una menor perplejidad y una mayor nitidez en la agrupación de términos locales en comparación con el baseline.

## 2.3 Estrategia de Extracción de Embeddings

Para obtener una representación vectorial fiel de cada ocurrencia de "Yape":

1.  **Subword Pooling**: Se reconstruye el vector de la palabra completa promediando los vectores de sus sub-tokens (BPE).
2.  **Concatenación de Capas**: Se concatenan las salidas de las **últimas 4 capas ocultas**.
    *   *Razón*: Las capas finales contienen la información semántica más rica, pero la última capa a veces está demasiado sesgada hacia la tarea de MLM. La concatenación provee un balance robusto.
3.  **Whitening (Opcional)**: Se aplica normalización para asegurar isotropía en el espacio vectorial local.
