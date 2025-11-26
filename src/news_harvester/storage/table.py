"""Persistencia tabular de registros de noticias."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from ..models import NewsRecord

OutputFormat = Literal["csv", "parquet"]


def _records_to_dataframe(records: Iterable[NewsRecord]) -> pd.DataFrame:
    rows = [record.model_dump() for record in records]
    return pd.DataFrame(rows)


def write_records(
    records: Iterable[NewsRecord],
    *,
    output_path: Path,
    format: OutputFormat = "csv",
    **pandas_kwargs,
) -> Path:
    """Escribe los registros en la ruta indicada y devuelve la ruta resultante."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = _records_to_dataframe(records)

    if df.empty:
        # Crear archivo vacío con cabeceras
        empty_df = pd.DataFrame(
            columns=[
                "title",
                "newspaper",
                "url",
                "published_at",
                "plain_text",
                "keyword",
            ]
        )
        if format == "csv":
            empty_df.to_csv(output_path, index=False, **pandas_kwargs)
        else:
            empty_df.to_parquet(output_path, index=False, **pandas_kwargs)
        return output_path

    if format == "csv":
        df.to_csv(output_path, index=False, **pandas_kwargs)
    elif format == "parquet":
        df.to_parquet(output_path, index=False, **pandas_kwargs)
    else:  # pragma: no cover - protección frente a uso incorrecto
        raise ValueError("Formato no soportado: {format}")

    return output_path
