# 1. Ingeniería de Datos y Adquisición

La calidad del análisis semántico depende de la integridad del corpus subyacente. Esta sección detalla las estrategias implementadas para garantizar la completitud y limpieza de los datos.

## 1.1 Estrategia de Cosecha (Harvesting) con GDELT

Se utilizó la API de GDELT Project para identificar noticias relevantes. Para mitigar las limitaciones de la API (que a menudo trunca resultados en rangos temporales amplios), se implementó una estrategia de **Daily Chunking**.

### Daily Chunking vs. Rango Completo
En lugar de solicitar `2019-01-01` a `2023-01-01` en una sola query, el script `Lisbeth News Harvester` divide el rango objetivo en segmentos de 24 horas y ejecuta una consulta independiente para cada día.

*   **Impacto**: Esta estrategia recuperó un estimado de **3x más documentos** en comparación con queries mensuales o anuales, capturando eventos de menor visibilidad global pero alta relevancia local.

## 1.2 Manejo de Renderizado Client-Side (JS)

Un desafío crítico fue la extracción de texto de medios modernos (e.g., *La República*) que utilizan frameworks reactivos, devolviendo códigos de estado `200 OK` pero cuerpos HTML vacíos o con *placeholders* en una petición `GET` estándar.

### Solución Implementada
Se integró una solución híbrida:
1.  **Petición Estándar (Requests)**: Rápida, para sitios estáticos.
2.  **Fallback Emulado**: Para dominios identificados como problemáticos o respuestas sospechosamente cortas, el sistema utiliza un *header* de navegador completo y manejo de cookies mediante `curl_cffi` para simular un usuario real y activar la renderización del servidor.

## 1.3 Limpieza y Preprocesamiento

El texto crudo se somete a un pipeline de normalización:
*   **Eliminación de Boilerplate**: Scripts, estilos CSS y pies de página recurrentes.
*   **Filtro de Relevancia**: Se descartan artículos donde la mención a "Yape" es, en realidad, un falso positivo (e.g., errores de OCR o palabras similares sin contexto semántico válido).
