# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir a LISBETH! Este documento establece las normas y flujos de trabajo para mantener la calidad y consistencia del código.

---

## 🏗️ Flujo de Trabajo (Git Workflow)

Utilizamos un modelo de "Feature Branching" simplificado.

1.  **Main Branch**: `main` contiene la versión estable y desplegable del código.
2.  **Ramas de Desarrollo**: Crea una rama para cada nueva funcionalidad o corrección.
    *   Formato: `feat/nombre-funcionalidad` o `fix/descripcion-bug`.
    *   Ejemplo: `feat/add-google-news-scraper`, `fix/gdelt-timeout`.

### Pasos para contribuir:

1.  Haz un **Fork** (si eres externo) o crea una **rama** desde `main`.
2.  Realiza tus cambios (commits atómicos y descriptivos).
3.  Asegúrate de que los tests pasen.
4.  Abre un **Pull Request (PR)** hacia `main`.

---

## 🎨 Estilo de Código

El código debe ser limpio, legible y tipado.

### Linter & Formatter (`ruff`)
Utilizamos **Ruff** para linting y formateo rápido.

*   Antes de hacer commit, ejecuta:
    ```bash
    ruff check . --fix
    ruff format .
    ```

### Type Hinting
Todo el código nuevo debe incluir anotaciones de tipo (Type Hints) de Python.
*   **Sí**: `def process(data: list[str]) -> int:`
*   **No**: `def process(data):`

### Docstrings
Usa docstrings estilo Google o NumPy para funciones complejas y clases públicas.
```python
def fetch_data(url: str) -> dict:
    """Descarga datos desde una URL.

    Args:
        url: La dirección web objetivo.

    Returns:
        Un diccionario con la respuesta JSON.
    """
```

---

## 🧪 Testing

La fiabilidad es crítica para un proyecto de investigación.

*   **Framework**: `pytest`
*   **Ubicación**: Carpeta `tests/`.
*   **Ejecución**:
    ```bash
    pytest
    ```

**Regla de Oro**: Si añades una nueva funcionalidad crítica (especialmente en `news_harvester` o cálculos matemáticos en `nlp`), debes añadir al menos un test unitario que verifique su funcionamiento básico.

---

## 📝 Convenciones de Commits

Recomendamos usar **Conventional Commits** para mantener un historial limpio:

*   `feat: añade soporte para scraping de RPP`
*   `fix: corrige error de timeout en GDELT`
*   `docs: actualiza instrucciones de instalación`
*   `refactor: mejora estructura de clases en model.py`
