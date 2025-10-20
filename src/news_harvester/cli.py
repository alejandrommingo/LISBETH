"""Punto de entrada de línea de comandos para Lisbeth News Harvester."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Sequence

import orjson
from dotenv import load_dotenv

from .collectors import Article, GDELTError, download_article_bodies, fetch_articles
from .config import Settings
from .models import NewsRecord
from .processing.records import build_news_record
from .storage import write_records


def _parse_iso_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - validación de argparse
        raise argparse.ArgumentTypeError(
            "Use el formato YYYY-MM-DD para las fechas"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news_harvester",
        description="Herramientas para recolectar noticias peruanas.",
    )
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Descarga metadatos (y opcionalmente HTML) desde la API GDELT",
    )
    fetch_parser.add_argument("--keyword", required=True, help="Palabra clave a buscar.")
    fetch_parser.add_argument(
        "--from",
        dest="date_from",
        type=_parse_iso_date,
        required=True,
        help="Fecha inicial (YYYY-MM-DD) inclusive.",
    )
    fetch_parser.add_argument(
        "--to",
        dest="date_to",
        type=_parse_iso_date,
        required=True,
        help="Fecha final (YYYY-MM-DD) inclusive.",
    )
    fetch_parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Lista de dominios permitidos; usa la configuración por defecto si se omite.",
    )
    fetch_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Número máximo de artículos por página (1-250).",
    )
    fetch_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Archivo JSON destino. Por defecto se guarda en el directorio configurado.",
    )
    fetch_parser.add_argument(
        "--download-html",
        action="store_true",
        help="Descarga el HTML de cada artículo después de consultarlo en GDELT.",
    )

    prototype_parser = subparsers.add_parser(
        "prototype",
        help="Ejecuta el prototipo completo (GDELT -> HTML -> texto -> tabla)",
    )
    prototype_parser.add_argument(
        "--keyword",
        default="Yape",
        help="Palabra clave a buscar (por defecto: Yape).",
    )
    prototype_parser.add_argument(
        "--from",
        dest="date_from",
        type=_parse_iso_date,
        default=None,
        help="Fecha inicial (YYYY-MM-DD). Usa la configuración por defecto si se omite.",
    )
    prototype_parser.add_argument(
        "--to",
        dest="date_to",
        type=_parse_iso_date,
        default=None,
        help="Fecha final (YYYY-MM-DD). Usa la configuración por defecto si se omite.",
    )
    prototype_parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Formato de salida para la tabla (csv o parquet).",
    )
    prototype_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta de archivo destino. Si se omite, se genera una en el directorio configurado.",
    )
    prototype_parser.add_argument(
        "--skip-html",
        action="store_true",
        help="No descargar los cuerpos HTML y generar el texto vacío (útil para pruebas rápidas).",
    )
    return parser


def _load_environment() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    load_dotenv(env_path)
    return Settings()


def _date_range_to_datetimes(date_from: dt.date, date_to: dt.date) -> tuple[dt.datetime, dt.datetime]:
    start_dt = dt.datetime.combine(date_from, dt.time.min, tzinfo=dt.timezone.utc)
    end_dt = dt.datetime.combine(date_to, dt.time(hour=23, minute=59, second=59), tzinfo=dt.timezone.utc)
    return start_dt, end_dt


def _save_articles(articles: Sequence[Article], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [article.to_dict() for article in articles]
    output_path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = _load_environment()

    if args.command == "prototype":
        return run_prototype(args, settings)

    if args.command != "fetch":
        parser.print_help()
        return

    start_dt, end_dt = _date_range_to_datetimes(args.date_from, args.date_to)

    domains = args.domains or settings.target_domains
    max_records = args.max_records or settings.gdelt_max_records

    articles = fetch_articles(
        keyword=args.keyword,
        start=start_dt,
        end=end_dt,
        source_country=settings.source_country,
        domains=domains,
        max_records=max_records,
        timeout=settings.request_timeout,
    )

    if args.download_html and articles:
        download_article_bodies(
            articles,
            delay_seconds=settings.request_delay_seconds,
            timeout=settings.request_timeout,
        )

    output_path = (
        args.output
        if args.output is not None
        else settings.output_dir
        / f"{args.keyword.lower()}_{args.date_from:%Y%m%d}_{args.date_to:%Y%m%d}.json"
    )

    _save_articles(articles, output_path)

    print(
        f"Se guardaron {len(articles)} artículos en {output_path}."
    )


def run_prototype(args: argparse.Namespace, settings: Settings) -> None:
    keyword: str = args.keyword
    date_from = args.date_from or settings.prototype_start
    date_to = args.date_to or settings.prototype_end

    start_dt, end_dt = _date_range_to_datetimes(date_from, date_to)

    try:
        articles = fetch_articles(
            keyword=keyword,
            start=start_dt,
            end=end_dt,
            source_country=settings.source_country,
            domains=settings.target_domains,
            max_records=settings.gdelt_max_records,
            timeout=settings.request_timeout,
        )
    except GDELTError as exc:
        print(f"Error al consultar GDELT: {exc}")
        return

    if not args.skip_html and articles:
        download_article_bodies(
            articles,
            delay_seconds=settings.request_delay_seconds,
            timeout=settings.request_timeout,
        )

    records: list[NewsRecord] = []
    skipped_without_content = 0
    for article in articles:
        record = build_news_record(
            article=article,
            keyword=keyword,
            html=article.raw_html,
        )
        if record is None:
            skipped_without_content += 1
            continue
        records.append(record)

    if args.output is not None:
        output_path = args.output
    else:
        output_suffix = f"{keyword.lower()}_{date_from:%Y%m%d}_{date_to:%Y%m%d}.{args.format}"
        output_path = settings.output_dir / output_suffix

    write_records(records, output_path=output_path, format=args.format)

    message = f"Prototipo completado: {len(records)} registros almacenados en {output_path}."
    if skipped_without_content:
        message += f" Se omitieron {skipped_without_content} artículos sin párrafos relevantes."
    print(message)


if __name__ == "__main__":  # pragma: no cover
    main()
