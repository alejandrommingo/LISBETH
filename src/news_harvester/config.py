"""Configuración centralizada del recolector."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .domains import PERUVIAN_MEDIA


class Settings(BaseSettings):
    """Define la configuración del proyecto usando variables de entorno."""

    newspaper_sources: List[str] = Field(
        default_factory=lambda: [
            "el_comercio",
            "la_republica",
            "gestion",
        ]
    )

    target_domains: list[str] = Field(
        default_factory=lambda: list(PERUVIAN_MEDIA.values()),
        description="Lista de dominios de interés para filtrar en GDELT.",
    )

    source_country: str = Field(
        "PE",
        description="Código de país ISO Alpha-2 para filtrar en GDELT (ej. PE para Perú).",
    )
    gdelt_max_records: int = Field(
        250,
        ge=1,
        le=250,
        description="Número máximo de registros por página desde la API de GDELT.",
    )
    prototype_start: dt.date = Field(
        dt.date(2020, 3, 1), description="Fecha inicial por defecto para el prototipo."
    )
    prototype_end: dt.date = Field(
        dt.date(2020, 5, 1),
        description="Fecha final por defecto (inclusive) para el prototipo.",
    )

    # Fuentes RSS directas
    PERUVIAN_RSS_FEEDS: list[str] = [
        "https://elcomercio.pe/arc/outboundfeeds/rss/",
        "https://larepublica.pe/rss/",
        "https://rpp.pe/feed",
        "https://gestion.pe/arc/outboundfeeds/rss/",
        "https://peru21.pe/arc/outboundfeeds/rss/",
        "https://trome.com/arc/outboundfeeds/rss/",
        "https://ojo.pe/arc/outboundfeeds/rss/",
        "https://diariocorreo.pe/arc/outboundfeeds/rss/",
        "https://publimetro.pe/arc/outboundfeeds/rss/",
    ]
    request_timeout: float = Field(
        30.0, ge=1.0, description="Tiempo máximo de espera por solicitud en segundos."
    )
    request_delay_seconds: float = Field(
        1.0,
        ge=0.0,
        description="Retardo entre descargas de HTML para evitar saturar a los periódicos.",
    )
    output_dir: Path = Field(
        Path("data"), description="Directorio donde se guardarán los resultados."
    )
    model_config = SettingsConfigDict(env_prefix="LISBETH_", case_sensitive=False)
