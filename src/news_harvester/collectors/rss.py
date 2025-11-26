"""Colector para feeds RSS directos de periódicos peruanos."""

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


def fetch_from_rss(
    *,
    feeds: List[str],
    keyword: str,
    start: dt.datetime,
    end: dt.datetime,
    timeout: float = 30.0,
) -> List[Article]:
    """Consulta una lista de feeds RSS y devuelve artículos filtrados.

    Args:
        feeds: Lista de URLs de feeds RSS.
        keyword: Palabra clave para filtrar (búsqueda simple en título/resumen).
        start: Fecha inicial (inclusive).
        end: Fecha final (inclusive).
        timeout: Timeout para la descarga del feed.

    Returns:
        Lista de objetos Article.
    """

    articles: List[Article] = []
    headers = {"User-Agent": ua.random}

    # Normalización básica para filtrado
    keyword_cf = keyword.casefold()

    for feed_url in feeds:
        try:
            logger.info("Consultando RSS: %s", feed_url)
            # Usamos httpx para descargar el contenido raw, más robusto que feedparser directo
            resp = httpx.get(
                feed_url, headers=headers, timeout=timeout, follow_redirects=True
            )
            resp.raise_for_status()

            feed = feedparser.parse(resp.text)

            for entry in feed.entries:
                # 1. Filtrado por fecha
                if not hasattr(entry, "published_parsed") or not entry.published_parsed:
                    continue

                pub_dt = dt.datetime.fromtimestamp(
                    time.mktime(entry.published_parsed), tz=dt.timezone.utc
                )

                if not (start <= pub_dt <= end):
                    continue

                # 2. Filtrado por keyword (simple)
                # Buscamos en título y resumen/descripción
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")

                content_to_check = f"{title} {summary}".casefold()

                if keyword_cf not in content_to_check:
                    continue

                # 3. Construcción del artículo
                url = entry.link
                domain = _extract_domain(url)

                article = Article(
                    title=title,
                    url=url,
                    domain=domain,
                    seen_datetime=dt.datetime.now(dt.timezone.utc),
                    seen_date=dt.date.today(),
                    language="es",
                    source_country="PE",
                    publish_datetime=pub_dt,
                    publish_date=pub_dt.date(),
                    source_api="DirectRSS",
                )
                articles.append(article)

        except Exception as exc:
            logger.warning("Error procesando feed %s: %s", feed_url, exc)
            continue

    return articles


def _extract_domain(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc
    except Exception:
        return ""
