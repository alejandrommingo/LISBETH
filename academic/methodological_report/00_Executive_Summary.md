# 0. Resumen Ejecutivo (Executive Summary)

Este reporte metodológico documenta la implementación técnica y validación científica del sistema **Lisbeth**, diseñado para analizar la evolución de la representación mediática de **Yape** como un "Actor Social" en el ecosistema peruano (2019-2023).

## Objetivos Alcanzados
1.  **Ingeniería de Datos Robusta**: Implementación de un pipeline de extracción tolerante a fallos (GDELT) con *Daily Chunking* y soporte para renderizado JS, logrando una cobertura temporal del 98% del periodo objetivo.
2.  **Modelado Semántico Avanzado**: Desarrollo de un modelo de embeddings contextuales basado en `roberta-large-bne` con adaptación al dominio (DAPT), superando significativamente al baseline en la representación de la jerga local.
3.  **Marco Matemático Validado**: Formalización de un pipeline algebraico (SVD $\to$ Procrustes $\to$ Gram-Schmidt) que garantiza la estabilidad y comparabilidad de los subespacios semánticos en el tiempo.
4.  **Resultados Sociológicos**: Identificación cuantitativa de hitos de cambio semántico (*Drift*) correlacionados con eventos sociopolíticos (Pandemia, Bonos), confirmando la hipótesis de la "agencia semántica" de la marca.

## Estructura del Reporte
*   **Sección 1: Ingeniería de Datos**: Detalles de adquisición y limpieza.
*   **Sección 2: Arquitectura NLP**: Selección de modelo y estrategia DAPT.
*   **Sección 3: Marco Matemático**: Definiciones formales.
*   **Sección 4: Narrativa Visual**: Resultados gráficos (Drift, Entropía, Proyecciones).
*   **Sección 5: Interpretación**: Análisis profundo y conclusiones.
