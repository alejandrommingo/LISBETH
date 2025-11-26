"""Modelos de datos estructurados para Lisbeth."""

from __future__ import annotations

import datetime as dt
from typing import cast

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


class NewsRecord(BaseModel):
    """Representa un registro tabular listo para análisis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(..., description="Titular de la noticia")
    newspaper: str = Field(..., description="Dominio o nombre del periódico de origen")
    url: AnyHttpUrl = Field(..., description="URL canónica del artículo")
    published_at: dt.datetime = Field(
        ..., description="Fecha y hora exactas de publicación (tz-aware)"
    )
    plain_text: str = Field(..., description="Contenido plano del artículo")
    keyword: str | None = Field(
        None, description="Palabra clave utilizada en la búsqueda"
    )
    relevance_score: float = Field(0.0, description="Puntuación de relevancia (0-100)")
    source: str = Field(
        "GDELT", description="Fuente de origen (GDELT, GoogleNews, etc.)"
    )

    @property
    def published_date(self) -> dt.date:
        """Alias conveniente para la fecha calendaria."""
        published_at = cast(dt.datetime, self.published_at)
        return published_at.date()
