"""Conversión de artículos GDELT a registros estructurados."""

from __future__ import annotations

import datetime as dt

from .text import extract_plain_text
from ..collectors.gdelt import Article
from ..models import NewsRecord

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
    keyword: str | None = None,
    html: str | None = None,
) -> NewsRecord | None:
    """Crea un ``NewsRecord`` usando el HTML (si está disponible).

    Si no se logra obtener texto con densidad suficiente (por ejemplo, porque
    el artículo no contiene un párrafo sustantivo con la palabra clave),
    devuelve ``None`` para permitir que el pipeline omita el registro.
    """

    plain_text = extract_plain_text(
        html or "",
        keyword=keyword,
        min_paragraph_chars=CONTENT_MIN_PARAGRAPH_CHARS,
        require_keyword=bool(keyword),
    )

    if not plain_text.strip():
        return None

    published_at = infer_published_datetime(article)

    return NewsRecord(
        title=article.title or "",
        newspaper=article.domain or article.source_country or "desconocido",
        url=article.url,
        published_at=published_at,
        plain_text=plain_text,
        keyword=keyword,
    )
