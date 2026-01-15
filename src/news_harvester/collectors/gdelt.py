"""Cliente de conveniencia para la API de documentos de GDELT 2.0."""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import httpx
from fake_useragent import UserAgent
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_SEEN_FORMAT = "%Y%m%dT%H%M%SZ"
GDELT_COMPACT_DATETIME_FORMAT = "%Y%m%d%H%M%S"
GDELT_COMPACT_DATE_FORMAT = "%Y%m%d"
GDELT_COMPACT_DATE_FORMAT = "%Y%m%d"
# Random User Agent

logger = logging.getLogger(__name__)
ua = UserAgent()


class GDELTError(RuntimeError):
    """Error genérico cuando GDELT devuelve una respuesta inesperada."""


@dataclass(slots=True)
class Article:
    """Representa un artículo retornado por GDELT."""

    title: str
    url: str
    domain: str
    seen_datetime: dt.datetime
    seen_date: dt.date
    language: str | None = None
    source_country: str | None = None
    publish_datetime: dt.datetime | None = None
    publish_date: dt.date | None = None
    publish_date: dt.date | None = None
    raw_html: str | None = None
    source_api: str = "GDELT"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "seen_datetime": self.seen_datetime.isoformat(),
            "seen_date": self.seen_date.isoformat(),
            "language": self.language,
            "source_country": self.source_country,
            "publish_datetime": self.publish_datetime.isoformat()
            if self.publish_datetime
            else None,
            "publish_date": self.publish_date.isoformat()
            if self.publish_date
            else None,
            "raw_html": self.raw_html,
            "source_api": self.source_api,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, str]) -> "Article":
        seen_datetime_raw = payload.get("seendatetime") or payload.get("seendate")
        if not seen_datetime_raw:
            raise GDELTError("Respuesta de GDELT incompleta: falta seendate")

        seen_datetime = _parse_datetime(seen_datetime_raw)
        seen_date = seen_datetime.date()

        publish_datetime_raw = payload.get("publishdatetime")
        publish_datetime = None
        if publish_datetime_raw:
            publish_datetime = _parse_datetime(
                publish_datetime_raw, suppress_errors=True
            )

        publish_date_raw = payload.get("publishdate")
        publish_date = None
        if publish_date_raw:
            try:
                publish_date = _parse_date(publish_date_raw)
            except ValueError:  # pragma: no cover
                publish_date = None

        return cls(
            title=payload.get("title", ""),
            url=payload["url"],
            domain=payload.get("domain", ""),
            seen_datetime=seen_datetime,
            seen_date=seen_date,
            language=payload.get("language"),
            source_country=payload.get("sourcecountry"),
            publish_datetime=publish_datetime,
            publish_date=publish_date,
        )


def _parse_datetime(value: str, *, suppress_errors: bool = False) -> dt.datetime | None:
    value = value.strip()
    candidate_formats = (
        GDELT_SEEN_FORMAT,
        GDELT_COMPACT_DATETIME_FORMAT,
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    )

    for fmt in candidate_formats:
        try:
            dt_obj = dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
        return dt_obj

    if suppress_errors:
        return None
    raise GDELTError(f"No se pudo interpretar la fecha/hora: {value!r}")


def _parse_date(value: str) -> dt.date:
    value = value.strip()
    candidate_formats = (
        GDELT_COMPACT_DATE_FORMAT,
        "%Y-%m-%d",
    )
    candidates = {value, value[:8], value[:10]}
    for candidate in candidates:
        for fmt in candidate_formats:
            try:
                return dt.datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"Formato de fecha no soportado: {value!r}")



from curl_cffi import requests as cffi_requests

def _ensure_client(
    client: cffi_requests.Session | None, timeout: float
) -> tuple[cffi_requests.Session, bool]:
    if client is not None:
        return client, False
    # Impersonate Chrome to bypass WAFs
    return cffi_requests.Session(impersonate="chrome110", timeout=timeout), True


def fetch_articles(
    *,
    keyword: str | list[str],
    start: dt.datetime,
    end: dt.datetime,
    source_country: str | None = "PE",
    domains: Sequence[str] | None = None,
    max_records: int = 250,
    timeout: float = 30.0,
    client: cffi_requests.Session | None = None,
) -> List[Article]:
    # ... (rest of docstring) ...
    
    # ... (validation logic same as before) ...
    if start >= end:
        raise ValueError("`start` debe ser anterior a `end`.")
    if max_records <= 0 or max_records > 250:
        raise ValueError("`max_records` debe estar en el rango 1-250.")

    domains_set: set[str] | None = {d.lower() for d in domains} if domains else None
    
    # Construcción de la query (same as before)
    if isinstance(keyword, str):
        keywords = [keyword]
    else:
        keywords = keyword

    cleaned_keywords = []
    for k in keywords:
        k = k.strip()
        if " " in k and not k.startswith('"'):
            cleaned_keywords.append(f'"{k}"')
        else:
            cleaned_keywords.append(k)
    
    if len(cleaned_keywords) > 1:
        keyword_part = f"({' OR '.join(cleaned_keywords)})"
    else:
        keyword_part = cleaned_keywords[0]

    query_parts = [keyword_part]
    if source_country:
        query_parts.append(f"sourceCountry:{source_country.upper()}")
    
    if domains:
        # Optimization: Filter by domain at source
        domain_parts = [f"domain:{d}" for d in domains]
        if len(domain_parts) > 1:
            query_parts.append(f"({' OR '.join(domain_parts)})")
        else:
            query_parts.append(domain_parts[0])

    query = " ".join(query_parts)

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": start.strftime(GDELT_COMPACT_DATETIME_FORMAT),
        "enddatetime": end.strftime(GDELT_COMPACT_DATETIME_FORMAT),
    }

    client_obj, created = _ensure_client(client, timeout)
    articles: list[Article] = []
    offset = 0

    try:
        while True:
            req_params = params | ({"offset": str(offset)} if offset else {})
            # curl_cffi uses params same as requests
            response = client_obj.get(GDELT_BASE_URL, params=req_params)
            response.raise_for_status()
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                snippet = response.text[:200]
                raise GDELTError(
                    f"Respuesta no válida de GDELT (no es JSON). Fragmento: {snippet!r}"
                ) from exc
            batch = payload.get("articles", [])
            if not isinstance(batch, list):
                raise GDELTError("Estructura inesperada en la respuesta de GDELT")
            if not batch:
                break

            for item in batch:
                if not isinstance(item, dict):
                    continue
                try:
                    article = Article.from_payload(item)
                except Exception as exc:
                    logger.warning("No fue posible parsear un artículo: %s", exc)
                    continue

                if domains_set and article.domain.lower() not in domains_set:
                    continue
                articles.append(article)

            if len(batch) < max_records:
                break
            offset += max_records
    finally:
        if created:
            client_obj.close()

    return articles


def download_article_bodies(
    articles: Iterable[Article],
    *,
    delay_seconds: float = 1.0,
    timeout: float = 20.0,
    client: cffi_requests.Session | None = None,
) -> None:
    """Descarga el HTML de cada artículo y lo adjunta en ``raw_html``.

    Modifica los objetos ``Article`` in-place. Si ocurre un error, el HTML se
    deja como ``None`` y se continúa.
    """

    client_obj, created = _ensure_client(client, timeout)

    # Configuración de reintentos para solicitudes individuales
    # curl_cffi raises requests.exceptions
    retry_decorator = retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception), # Catch-all for simplicity with cffi
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    try:
        for article in articles:
            try:
                # Función interna para permitir decoración con tenacity
                @retry_decorator
                def _fetch_one(url: str) -> str:
                    # curl_cffi handles headers/impersonation in Session
                    resp = client_obj.get(url)
                    resp.raise_for_status()
                    return resp.text

                article.raw_html = _fetch_one(article.url)

            except Exception as exc:
                logger.error(
                    "Error al descargar %s: %s. Intentando Wayback Machine...",
                    article.url,
                    exc,
                )
                # Wayback fallback might need standard httpx or curl_cffi too
                # For simplicity, let's use the same client
                article.raw_html = _try_wayback_machine(client_obj, article)
            else:
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
    finally:
        if created:
            client_obj.close()


def _try_wayback_machine(client: cffi_requests.Session, article: Article) -> str | None:
    """Intenta recuperar una instantánea de Wayback Machine cercana a la fecha de publicación."""
    try:
        # Formato timestamp para API: YYYYMMDD
        timestamp = article.seen_date.strftime("%Y%m%d")
        api_url = f"http://archive.org/wayback/available?url={article.url}&timestamp={timestamp}"
        
        logger.info("Consultando Wayback API: %s", api_url)
        # curl_cffi session
        resp = client.get(api_url, timeout=10.0)
        
        if resp.status_code != 200:
            logger.warning("Wayback API devolvió status %s", resp.status_code)
            return None
            
        data = resp.json()
        snapshots = data.get("archived_snapshots", {})
        closest = snapshots.get("closest")

        if closest and closest.get("available"):
            snapshot_url = closest.get("url")
            logger.info("Instantánea encontrada en Wayback Machine: %s", snapshot_url)
            # Descargar la instantánea
            wb_resp = client.get(snapshot_url, timeout=30.0)
            wb_resp.raise_for_status()
            return wb_resp.text
        else:
            logger.warning("Wayback Machine no tiene instantáneas para %s", article.url)

    except Exception as exc:
        logger.warning(
            "Fallo al consultar Wayback Machine para %s: %s", article.url, exc
        )

    return None
