# Lisbeth News Harvester

**Lisbeth News Harvester** es una herramienta robusta y escalable diseñada para recolectar, procesar y estructurar noticias de medios peruanos. Su objetivo principal es facilitar el análisis de reputación y tendencias mediante la creación de corpus de texto de alta calidad.

El sistema se centra en **GDELT** como fuente primaria masiva, complementada por **Google News** y **RSS directos**, e incorpora mecanismos avanzados de filtrado, limpieza y puntuación de relevancia.

## Características Principales

*   **Fuentes Masivas**:
    *   **GDELT**: Cobertura histórica con más de 30 medios peruanos (El Comercio, La República, RPP, Ojo Público, etc.).
    *   **Google News & RSS**: Fuentes complementarias para noticias recientes.
*   **Calidad de Datos**:
    *   **Extracción Limpia**: Uso de `trafilatura` para obtener texto plano sin ruido de navegación.
    *   **Puntuación de Relevancia**: Algoritmo que califica (0-100) cada artículo basándose en la presencia de la palabra clave.
        *   **Título**: 40 puntos si aparece la keyword.
        *   **Lead (Primeros 200 caracteres)**: 30 puntos si aparece.
        *   **Cuerpo**: 10 puntos por cada aparición (hasta un máximo de 30 puntos).
        *   *Nota*: El cálculo es insensible a mayúsculas y acentos.
    *   **Resiliencia**: Sistema anti-bloqueo con rotación de User-Agents y recuperación automática de enlaces rotos (403/404) vía **Wayback Machine**.
*   **Filtrado Avanzado**:
    *   Selección de medios por nombre (`--media elcomercio rpp`).
    *   Selección de fuentes de recolección (`--sources gdelt google`).

## Instalación

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/alejandrommingo/LISBETH.git
    cd LISBETH
    ```

2.  **Crear entorno virtual e instalar dependencias**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configuración (Opcional)**:
    El archivo `.env` permite ajustar parámetros globales, aunque el sistema funciona con valores por defecto optimizados.

## Uso: El "Golden Path"

El caso de uso principal es descargar noticias históricas sobre un tema específico (ej. "Yape") de todos los medios peruanos disponibles.

### Comando Prototipo

```bash
PYTHONPATH=src python -m news_harvester prototype \
    --keyword "Yape" \
    --from 2020-03-01 \
    --to 2020-05-01 \
    --media all \
    --output data/yape_dataset.csv
```

### Argumentos Clave

*   `--keyword`: Término de búsqueda (ej. "Yape", "Vizcarra", "BCP").
*   `--from` / `--to`: Rango de fechas (YYYY-MM-DD).
*   `--media`: Filtra por medios específicos.
    *   `all`: Todos los 30+ medios peruanos (Por defecto).
    *   Nombres específicos: `elcomercio`, `larepublica`, `rpp`, `gestion`, `trome`, `ojo`, `publimetro`, `americatv`, `canaln`, `willax`, `ojopublico`, `idl`, etc.
*   `--sources`: Fuentes de recolección.
    *   `gdelt`: (Por defecto) La más completa para histórico.
    *   `google`, `rss`: Útiles para noticias de última hora.

## Estructura de Datos (CSV)

El archivo generado contiene las siguientes columnas:

| Columna | Descripción |
| :--- | :--- |
| `title` | Titular del artículo. |
| `newspaper` | Dominio del medio (ej. `elcomercio.pe`). |
| `published_at` | Fecha y hora de publicación (UTC). |
| `plain_text` | Texto completo extraído y limpio. |
| `relevance_score` | Puntuación (0-100) indicando qué tan centrado está el artículo en la keyword. |
| `source` | Fuente de origen (`GDELT`, `GoogleNews`, `DirectRSS`). |
| `url` | Enlace original. |

## Desarrollo

Para ejecutar las pruebas:
```bash
PYTHONPATH=src pytest tests/
```

Para verificar estilo de código:
```bash
ruff check .
```
