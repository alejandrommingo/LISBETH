"""Colector para Google News RSS."""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import List

import feedparser
import httpx
from fake_useragent import UserAgent

from .gdelt import Article

logger = logging.getLogger(__name__)
ua = UserAgent()


def fetch_google_news(
    *,
    keyword: str | list[str],
    start: dt.datetime,
    end: dt.datetime,
    source_country: str | None = "PE",
    timeout: float = 30.0,
) -> List[Article]:
    """Consulta el RSS de Google News y devuelve artículos.

    Nota: Google News RSS no soporta filtrado estricto por fechas históricas
    como GDELT. Devuelve lo más reciente o relevante. Filtramos manualmente
    por fecha si es posible, pero la cobertura histórica es limitada.
    """

    # Construir URL del feed
    # ceid: country, gl: geolocation, hl: host language
    base_url = "https://news.google.com/rss/search"
    
    if isinstance(keyword, str):
        keywords = [keyword]
    else:
        keywords = keyword
        
    # Google News soporta OR
    if len(keywords) > 1:
        keyword_part = f"({' OR '.join(keywords)})"
    else:
        keyword_part = keywords[0]

    query = f"{keyword_part} after:{start.strftime('%Y-%m-%d')} before:{end.strftime('%Y-%m-%d')}"
    if source_country:
        query += " when:1y"  # Intento de ampliar rango, aunque RSS es limitado

    params = {
        "q": query,
        "hl": "es-419",
        "gl": source_country if source_country else "PE",
        "ceid": f"{source_country}:es-419" if source_country else "PE:es-419",
    }

    logger.info("Consultando Google News RSS: %s", params)

    # Usamos httpx para obtener el XML raw, a veces feedparser falla con redes
    headers = {"User-Agent": ua.random}
    try:
        resp = httpx.get(
            base_url,
            params=params,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
    except Exception as exc:
        logger.error("Error al consultar Google News: %s", exc)
        return []

    articles: List[Article] = []

    # Cliente para resolver redirecciones
    resolver_client = httpx.Client(
        timeout=10.0, follow_redirects=True, headers={"User-Agent": ua.random}
    )

    for entry in feed.entries:
        try:
            # Parsear fecha de publicación
            # feedparser devuelve struct_time en entry.published_parsed
            if not hasattr(entry, "published_parsed") or not entry.published_parsed:
                continue

            pub_dt = dt.datetime.fromtimestamp(
                time.mktime(entry.published_parsed), tz=dt.timezone.utc
            )

            # Filtrado manual por fecha (el parámetro de búsqueda no siempre es estricto)
            if not (start <= pub_dt <= end):
                continue

            # Resolver URL real (Google News usa enlaces de redirección)
            original_url = entry.link
            resolved_url = _resolve_url(resolver_client, original_url)

            article = Article(
                title=entry.title,
                url=resolved_url,
                domain=_extract_domain(resolved_url),
                seen_datetime=dt.datetime.now(dt.timezone.utc),
                seen_date=dt.date.today(),
                language="es",
                source_country=source_country,
                publish_datetime=pub_dt,
                publish_date=pub_dt.date(),
            )
            articles.append(article)

        except Exception as exc:
            logger.warning("Error procesando entrada de Google News: %s", exc)
            continue

    resolver_client.close()
    return articles


def _resolve_url(client: httpx.Client, url: str) -> str:
    """Resuelve la URL final tras las redirecciones de Google."""
    try:
        # Google News usa un JS redirect a veces, o 302.
        # Hacemos un HEAD o GET rápido.
        resp = client.head(url, follow_redirects=True)
        return str(resp.url)
    except Exception:
        # Si falla HEAD, intentamos GET
        try:
            resp = client.get(url, follow_redirects=True)
            return str(resp.url)
        except Exception:
            return url


def _extract_domain(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc
    except Exception:
        return ""
