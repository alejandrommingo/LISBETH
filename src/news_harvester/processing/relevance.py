"""Lógica para calcular la relevancia de un artículo respecto a una palabra clave."""

from __future__ import annotations

import unicodedata


def calculate_relevance_score(text: str, title: str, keyword: str | list[str]) -> float:
    """Calcula una puntuación de relevancia (0-100) para un artículo.

    Factores:
    - Presencia en el título (40 puntos).
    - Presencia en el primer párrafo/lead (30 puntos).
    - Frecuencia en el cuerpo (10 puntos por aparición, máx 30).

    Args:
        text: Contenido completo del artículo.
        title: Titular del artículo.
        keyword: Palabra(s) clave a buscar. Puede ser str o lista de str.

    Returns:
        float: Puntuación entre 0.0 y 100.0.
    """
    if not keyword:
        return 0.0

    keywords = [keyword] if isinstance(keyword, str) else keyword
    if not keywords:
        return 0.0

    score = 0.0
    keywords_normalized = [_normalize(k) for k in keywords]
    title_normalized = _normalize(title)
    text_normalized = _normalize(text)

    # 1. Presencia en el título (basta con que aparezca una)
    if any(k in title_normalized for k in keywords_normalized):
        score += 40.0

    # 2. Presencia en el lead (primeros 200 caracteres aprox)
    lead = text_normalized[:200]
    if any(k in lead for k in keywords_normalized):
        score += 30.0

    # 3. Frecuencia en el cuerpo (suma de todas las keywords)
    count = sum(text_normalized.count(k) for k in keywords_normalized)
    
    # 10 puntos por cada aparición adicional (descontando la del lead si ya se contó)
    # Simplificación: contamos todas y topeamos en 30
    frequency_score = min(30.0, count * 10.0)
    score += frequency_score

    return min(100.0, score)


def _normalize(text: str) -> str:
    """Normaliza texto para comparación insensible a mayúsculas y acentos."""
    if not text:
        return ""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ASCII", "ignore")
        .decode("utf-8")
        .casefold()
    )
