"""Conversión de artículos GDELT a registros estructurados."""

from __future__ import annotations

import datetime as dt

from .text import extract_plain_text
from ..collectors.gdelt import Article
from ..models import NewsRecord
from .relevance import calculate_relevance_score

CONTENT_MIN_PARAGRAPH_CHARS = 90


def infer_published_datetime(article: Article) -> dt.datetime:
    """Determina la fecha/hora de publicación más confiable disponible.

    Prioriza ``publish_datetime``; si no existe, usa ``publish_date`` combinada con
    la hora de ``seen_datetime``; finalmente recurre a ``seen_datetime``.
    """

    if article.publish_datetime is not None:
        return article.publish_datetime

    if article.publish_date is not None:
        return dt.datetime.combine(
            article.publish_date,
            article.seen_datetime.timetz(),
            tzinfo=article.seen_datetime.tzinfo,
        )

    return article.seen_datetime


def build_news_record(
    *,
    article: Article,
    keyword: str | list[str] | None = None,
    html: str | None = None,
) -> NewsRecord | None:
    """Crea un ``NewsRecord`` usando el HTML (si está disponible).

    Si no se logra obtener texto con densidad suficiente (por ejemplo, porque
    el artículo no contiene un párrafo sustantivo con la palabra clave),
    devuelve ``None`` para permitir que el pipeline omita el registro.
    """

    # Priorizar argumento html, luego article.raw_html
    html_content = html or article.raw_html or ""

    plain_text = extract_plain_text(
        html_content,
        keyword=keyword,
        min_paragraph_chars=CONTENT_MIN_PARAGRAPH_CHARS,
        require_keyword=bool(keyword),
        strict_mode=False,
        domain=article.domain,
    )

    if not plain_text:
        return None

    relevance = (
        calculate_relevance_score(plain_text, article.title, keyword)
        if keyword
        else 0.0
    )

    # Determinar fuente (si no está explícita en el artículo, asumimos GDELT por defecto)
    # En el futuro, Article podría tener un campo 'source_api'
    source = getattr(article, "source_api", "GDELT")
    
    # Si keyword es una lista, la convertimos a string para el registro
    keyword_str = ", ".join(keyword) if isinstance(keyword, list) else keyword

    return NewsRecord(
        title=article.title,
        newspaper=article.domain,
        url=article.url,
        published_at=infer_published_datetime(article),
        plain_text=plain_text,
        keyword=keyword_str,
        relevance_score=relevance,
        source=source,
    )
