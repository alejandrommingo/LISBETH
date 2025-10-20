# Lisbeth News Harvester

Este proyecto automatiza la descarga de noticias de periódicos peruanos. A continuación se detallan los pasos para crear y usar el entorno virtual de Python, instalar dependencias y ejecutar la herramienta.

## Requisitos previos

- Python 3.12 o superior (se recomienda usar la misma versión utilizada en este repositorio).
- `pip` actualizado.

## Crear el entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell, reemplaza el segundo comando por:

```powershell
.venv\Scripts\Activate.ps1
```

## Instalar dependencias

Con el entorno virtual activo:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
# Dependencias opcionales para desarrollo
python -m pip install -e .[dev]
```

## Estructura sugerida del proyecto

```
Lisbeth/
├── README.md
├── requirements.txt
├── pyproject.toml
└── src/
    └── news_harvester/
        ├── __init__.py
        ├── __main__.py
        ├── cli.py
        ├── config.py
        ├── models.py
        ├── collectors/
        │   ├── __init__.py
        │   └── gdelt.py
        ├── processing/
        │   ├── __init__.py
        │   └── text.py
        └── storage/
            ├── __init__.py
            └── table.py
```

## Uso rápido

### Subcomando `fetch`

Consulta la API pública de GDELT para obtener titulares y, opcionalmente, descargar el HTML de las noticias.

```bash
# Ejemplo: noticias que mencionan "Yape" entre el 1 de marzo y el 1 de abril de 2020
python -m news_harvester fetch \
    --keyword "Yape" \
    --from 2020-03-01 \
    --to 2020-04-01 \
    --download-html
```

Por defecto se filtra por dominios peruanos (`elcomercio.pe`, `gestion.pe`, etc.) y se genera un archivo JSON en `data/`. Utiliza `--output` para personalizar la ruta o `--domains` para ampliar la lista de medios válidos.

### Subcomando `prototype`

Ejecuta el pipeline completo (GDELT → descarga HTML → extracción de texto → tabla) empleando los parámetros por defecto del proyecto (palabra clave "Yape" y rango 2020-03-01 a 2020-05-01).

```bash
python -m news_harvester prototype
```

Parámetros útiles:

- `--keyword`: otra palabra clave.
- `--from` / `--to`: fechas específicas (inclusive) en formato ISO.
- `--format`: `csv` (por defecto) o `parquet`.
- `--output`: ruta de archivo destino si no deseas la predeterminada en `data/`.
- `--skip-html`: salta la descarga de HTML (solo extrae metadatos).

> Nota: GDELT espera palabras clave de al menos 3 caracteres y códigos de país de dos letras (`PE`). Si la API devuelve advertencias, el comando las registrará y continuará con los artículos válidos.

### Estructura del dataset generado

Cada registro se serializa con los campos:

- `title`: titular de la noticia.
- `newspaper`: dominio del medio (`elcomercio.pe`, etc.).
- `url`: enlace canónico del artículo.
- `published_at`: fecha/hora (UTC) inferida de publicación.
- `plain_text`: contenido textual limpio.
- `keyword`: término utilizado en la búsqueda.
El comando de validación inicial generó `data/yape_20200301_20200501.csv` con 17 registros.
## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto para almacenar credenciales o claves API. Ejemplo:

```
NEWS_API_KEY="tu_clave"
```

Carga estas variables en tu código con `python-dotenv`.

## Próximos pasos sugeridos

- Añadir parseadores específicos por dominio para extraer título, resumen y cuerpo sin depender del HTML completo.
- Registrar métricas de scraping y detectar duplicados para controlar la calidad del dataset.
- Incorporar almacenamiento (por ejemplo, SQLite o PostgreSQL) y un modelo de datos para artículos normalizados.
- Automatizar ejecuciones periódicas con `APScheduler` o flujos tipo Airflow/Prefect.
- Crear pipelines de evaluación de calidad (tests adicionales, linters, CI). 
