"""Cliente de conveniencia para la API de documentos de GDELT 2.0."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import httpx

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_SEEN_FORMAT = "%Y%m%dT%H%M%SZ"
GDELT_COMPACT_DATETIME_FORMAT = "%Y%m%d%H%M%S"
GDELT_COMPACT_DATE_FORMAT = "%Y%m%d"
DEFAULT_USER_AGENT = "lisbeth-news-harvester/0.1.0 (https://github.com/)"

logger = logging.getLogger(__name__)


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
    raw_html: str | None = None

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
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "raw_html": self.raw_html,
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
            publish_datetime = _parse_datetime(publish_datetime_raw, suppress_errors=True)

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


def _ensure_client(client: httpx.Client | None, timeout: float) -> tuple[httpx.Client, bool]:
    if client is not None:
        return client, False
    return httpx.Client(timeout=timeout, headers={"User-Agent": DEFAULT_USER_AGENT}, follow_redirects=True), True


def fetch_articles(
    *,
    keyword: str,
    start: dt.datetime,
    end: dt.datetime,
    source_country: str | None = "PER",
    domains: Sequence[str] | None = None,
    max_records: int = 250,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> List[Article]:
    """Consulta la API de GDELT y devuelve artículos filtrados por dominio.

    Parámetros
    ----------
    keyword:
        Palabra o frase a buscar (se encapsula entre comillas automáticamente).
    start / end:
        Intervalo temporal en UTC.
    domains:
        Lista de dominios permitidos; si es ``None`` no se filtra.
    max_records:
        Límite (<= 250) por página; GDELT permite usar ``offset`` para paginar.
    timeout:
        Tiempo máximo de espera por petición en segundos.
    client:
        Cliente HTTP reutilizable para facilitar pruebas.
    """

    if start >= end:
        raise ValueError("`start` debe ser anterior a `end`.")
    if max_records <= 0 or max_records > 250:
        raise ValueError("`max_records` debe estar en el rango 1-250.")

    domains_set: set[str] | None = {d.lower() for d in domains} if domains else None
    keyword_term = keyword.strip()
    if " " in keyword_term:
        query_parts = [f'"{keyword_term}"']
    else:
        query_parts = [keyword_term]
    if source_country:
        query_parts.append(f"sourceCountry:{source_country.upper()}")
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
            response = client_obj.get(GDELT_BASE_URL, params=req_params)
            response.raise_for_status()
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                snippet = response.text[:200]
                raise GDELTError(
                    "Respuesta no válida de GDELT (no es JSON). Fragmento: "
                    f"{snippet!r}"
                ) from exc
            batch = payload.get("articles", [])
            if not isinstance(batch, list):  # pragma: no cover - defensa
                raise GDELTError("Estructura inesperada en la respuesta de GDELT")
            if not batch:
                break

            for item in batch:
                if not isinstance(item, dict):  # pragma: no cover
                    continue
                try:
                    article = Article.from_payload(item)
                except Exception as exc:  # pragma: no cover
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
    client: httpx.Client | None = None,
) -> None:
    """Descarga el HTML de cada artículo y lo adjunta en ``raw_html``.

    Modifica los objetos ``Article`` in-place. Si ocurre un error, el HTML se
    deja como ``None`` y se continúa.
    """

    client_obj, created = _ensure_client(client, timeout)

    try:
        for article in articles:
            try:
                response = client_obj.get(article.url)
                response.raise_for_status()
            except httpx.HTTPError:
                article.raw_html = None
            else:
                article.raw_html = response.text

            if delay_seconds > 0:
                time.sleep(delay_seconds)
    finally:
        if created:
            client_obj.close()
