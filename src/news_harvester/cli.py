"""Punto de entrada de línea de comandos para Lisbeth News Harvester."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Sequence

import orjson
from dotenv import load_dotenv

from .collectors import Article, GDELTError, download_article_bodies, fetch_articles
from .collectors.google import fetch_google_news
from .collectors.rss import fetch_from_rss
from .config import Settings
from .domains import PERUVIAN_MEDIA, MEDIA_RSS_FEEDS


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
    fetch_parser.add_argument(
        "--keyword", 
        required=True, 
        nargs="+",
        help="Palabra(s) clave a buscar."
    )
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

    harvest_parser = subparsers.add_parser(
        "harvest",
        help="Ejecuta la recolección completa (GDELT/Google/RSS -> HTML -> texto -> tabla)",
    )
    harvest_parser.add_argument(
        "--keyword",
        default=["Yape"],
        nargs="+",
        help="Palabra(s) clave a buscar (por defecto: Yape).",
    )
    harvest_parser.add_argument(
        "--from",
        dest="date_from",
        type=_parse_iso_date,
        default=None,
        help="Fecha inicial (YYYY-MM-DD). Usa la configuración por defecto si se omite.",
    )
    harvest_parser.add_argument(
        "--to",
        dest="date_to",
        type=_parse_iso_date,
        default=None,
        help="Fecha final (YYYY-MM-DD). Usa la configuración por defecto si se omite.",
    )
    harvest_parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Formato de salida para la tabla (csv o parquet).",
    )
    harvest_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta de archivo destino. Si se omite, se genera una en el directorio configurado.",
    )
    harvest_parser.add_argument(
        "--no-fetch-html",
        action="store_true",
        help="No descargar los cuerpos HTML. Los artículos sin texto serán descartados.",
    )
    harvest_parser.add_argument(
        "--sources",
        nargs="+",
        default=["gdelt"],
        choices=["gdelt", "google", "rss"],
        help="Fuentes a utilizar (por defecto: gdelt)",
    )
    harvest_parser.add_argument(
        "--media",
        nargs="+",
        default=["all"],
        help="Medios a filtrar por nombre (ej: elcomercio rpp). 'all' incluye todos.",
    )
    return parser


def _load_environment() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    load_dotenv(env_path)
    return Settings()


def _date_range_to_datetimes(
    date_from: dt.date, date_to: dt.date
) -> tuple[dt.datetime, dt.datetime]:
    start_dt = dt.datetime.combine(date_from, dt.time.min, tzinfo=dt.timezone.utc)
    end_dt = dt.datetime.combine(
        date_to, dt.time(hour=23, minute=59, second=59), tzinfo=dt.timezone.utc
    )
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

    if args.command == "harvest":
        return run_harvest(args, settings)

    if args.command != "fetch":
        parser.print_help()
        return

    start_dt, end_dt = _date_range_to_datetimes(args.date_from, args.date_to)

    domains = args.domains or settings.target_domains
    max_records = args.max_records or settings.gdelt_max_records

    # args.keyword es ahora una lista
    keywords = args.keyword

    # Daily Chunking Logic
    current_date = start_dt
    all_articles: list[Article] = []

    print(f"Iniciando recolección con Daily Chunking ({start_dt.date()} -> {end_dt.date()})...")

    while current_date <= end_dt:
        # Definir el final del día actual (23:59:59)
        day_end = current_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        if day_end > end_dt:
            day_end = end_dt
        
        print(f"  Consultando {current_date.date()}...")
        try:
            daily_articles = fetch_articles(
                keyword=keywords,
                start=current_date,
                end=day_end,
                source_country=settings.source_country,
                domains=domains,
                max_records=max_records,
                timeout=settings.request_timeout,
            )
            all_articles.extend(daily_articles)
            print(f"    -> Encontrados: {len(daily_articles)}")
        except Exception as e:
            print(f"    -> Error en {current_date.date()}: {e}")

        # Avanzar al siguiente día
        current_date += dt.timedelta(days=1)
        # Resetear hora a 00:00:00 para el siguiente día
        current_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)

    articles = all_articles
    print(f"Total artículos encontrados: {len(articles)}")

    if args.download_html and articles:
        download_article_bodies(
            articles,
            delay_seconds=settings.request_delay_seconds,
            timeout=settings.request_timeout,
        )

    # Generar nombre de archivo basado en keywords
    keyword_slug = "_".join(k.lower() for k in keywords[:3])
    if len(keywords) > 3:
        keyword_slug += "_etc"

    output_path = (
        args.output
        if args.output is not None
        else settings.output_dir
        / f"{keyword_slug}_{args.date_from:%Y%m%d}_{args.date_to:%Y%m%d}.json"
    )

    _save_articles(articles, output_path)

    print(f"Se guardaron {len(articles)} artículos en {output_path}.")


def run_harvest(args: argparse.Namespace, settings: Settings) -> None:
    keywords: list[str] = args.keyword
    date_from = args.date_from or settings.prototype_start
    date_to = args.date_to or settings.prototype_end

    start_dt, end_dt = _date_range_to_datetimes(date_from, date_to)
    articles: list[Article] = []

    selected_sources = args.sources
    print(f"Fuentes seleccionadas: {selected_sources}")

    # Resolver dominios
    target_domains = []
    if "all" in args.media:
        # "all" ahora significa SIN FILTRO DE DOMINIO (confiamos en sourceCountry:PE)
        target_domains = None
        print("Medios seleccionados: TODOS (Sin filtro de dominio, solo país)")
    else:
        for media_name in args.media:
            if media_name in PERUVIAN_MEDIA:
                target_domains.append(PERUVIAN_MEDIA[media_name])
            else:
                print(f"Advertencia: Medio '{media_name}' no reconocido.")
        if not target_domains:
            print(
                "Advertencia: No se seleccionaron dominios válidos. Usando TODOS por defecto."
            )
            target_domains = None
        else:
            print(f"Medios seleccionados: {args.media} ({target_domains})")
    
    keyword_display = ", ".join(keywords)

    # Daily Chunking Logic
    current_date = start_dt
    
    # Prepare output path
    if args.output is not None:
        output_path = args.output
    else:
        keyword_slug = "_".join(k.lower() for k in keywords[:3])
        if len(keywords) > 3:
            keyword_slug += "_etc"
        output_suffix = (
            f"{keyword_slug}_{date_from:%Y%m%d}_{date_to:%Y%m%d}.{args.format}"
        )
        output_path = settings.output_dir / output_suffix
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear file if it exists (fresh start)
    # Si retomamos, habría que leerlo, pero para este prototipo limpiamos.
    if output_path.exists():
        output_path.unlink()

    print(f"Iniciando recolección con Daily Chunking ({start_dt.date()} -> {end_dt.date()})...")
    print(f"Guardando resultados incrementalmente en: {output_path}")

    # Metrics
    metrics = {
        "n_candidates": 0,
        "n_download_ok": 0, # Note: download_article_bodies doesn't return stats easily, skipping exact check here unless Refactored
        "n_records_saved": 0,
        "n_skipped_no_text": 0
    }

    import pandas as pd

    while current_date <= end_dt:
        day_end = current_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        if day_end > end_dt:
            day_end = end_dt
        
        print(f"  Consultando {current_date.date()}...", flush=True)
        
        daily_articles: list[Article] = []
        try:
            # GDELT
            if "gdelt" in selected_sources:
                batch = fetch_articles(
                    keyword=keywords,
                    start=current_date,
                    end=day_end,
                    source_country=settings.source_country,
                    domains=target_domains,
                    max_records=settings.gdelt_max_records,
                    timeout=settings.request_timeout,
                )
                for a in batch:
                    a.source_api = "GDELT"
                daily_articles.extend(batch)
            
            # GOOGLE NEWS (Complemento)
            if "google" in selected_sources:
                try:
                    google_batch = fetch_google_news(
                        keyword=keywords,
                        start=current_date,
                        end=day_end,
                        source_country=settings.source_country
                    )
                    for a in google_batch:
                        a.source_api = "GOOGLE"
                    daily_articles.extend(google_batch)
                except Exception as e_google:
                    print(f"    -> Error Google News: {e_google}")

            # RSS (Medios locales directos)
            # Solo tiene sentido consultar RSS si estamos procesando el día de hoy
            # De lo contrario, estaríamos consultando lo mismo para cada día histórico (ineficiente e incorrecto)
            today = dt.datetime.now(dt.timezone.utc).date()
            if "rss" in selected_sources and current_date.date() == today:
                 try:
                    # Collect all feeds
                    feed_urls = list(MEDIA_RSS_FEEDS.values())
                    rss_batch = fetch_from_rss(
                        feeds=feed_urls,
                        keyword=keywords,
                        start=current_date,
                        end=day_end
                    )
                    
                    for a in rss_batch:
                        a.source_api = "RSS"
                    daily_articles.extend(rss_batch)
                 except Exception as e_rss:
                     print(f"    -> Error RSS: {e_rss}")


            # Eliminar duplicados básicos por URL
            unique_articles = {}
            for a in daily_articles:
                unique_articles[a.url] = a
            daily_articles = list(unique_articles.values())

            metrics["n_candidates"] += len(daily_articles)

            if daily_articles:
                print(f"    -> Encontrados (total fuentes): {len(daily_articles)}", flush=True)
                
                if not args.no_fetch_html:
                    download_article_bodies(
                        daily_articles,
                        delay_seconds=settings.request_delay_seconds,
                        timeout=settings.request_timeout,
                    )
                    # Rough estimate
                    metrics["n_download_ok"] += sum(1 for a in daily_articles if a.raw_html)
                
                # Process and Filter
                daily_records = []
                for article in daily_articles:
                    record = build_news_record(
                        article=article,
                        keyword=keywords,
                        html=article.raw_html,
                    )
                    if record:
                        daily_records.append(record)
                    else:
                        metrics["n_skipped_no_text"] += 1
                
                if daily_records:
                    # Append to file
                    data_rows = []
                    for r in daily_records:
                        row = r.model_dump()
                        # Convert AnyHttpUrl to string for Parquet compatibility
                        row["url"] = str(row["url"])
                        data_rows.append(row)

                    df = pd.DataFrame(data_rows)
                    
                    if args.format == "csv":
                        mode = "w" if metrics["n_records_saved"] == 0 else "a"
                        header = (metrics["n_records_saved"] == 0)
                        df.to_csv(output_path, mode=mode, header=header, index=False)
                    elif args.format == "parquet":
                        if metrics["n_records_saved"] == 0:
                            df.to_parquet(output_path, engine="pyarrow", index=False)
                        else:
                            existing_df = pd.read_parquet(output_path)
                            combined = pd.concat([existing_df, df], ignore_index=True)
                            combined.to_parquet(output_path, engine="pyarrow", index=False)
                    
                    saved_count = len(daily_records)
                    metrics["n_records_saved"] += saved_count
                    print(f"    -> Guardados: {saved_count} (Total: {metrics['n_records_saved']})", flush=True)
            else:
                 print(f"    -> Sin resultados.", flush=True)

        except Exception as e:
            print(f"    -> Error en {current_date.date()}: {e}", flush=True)

        current_date += dt.timedelta(days=1)
        current_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"Proceso completado.")
    print("Métricas de Ejecución:")
    print(orjson.dumps(metrics, option=orjson.OPT_INDENT_2).decode())

    message = (
        f"Harvest completado: {metrics['n_records_saved']} registros almacenados en {output_path}."
    )
    if metrics["n_skipped_no_text"]:
        message += f" Se omitieron {metrics['n_skipped_no_text']} artículos vacíos/irrelevantes."
    print(message)


if __name__ == "__main__":  # pragma: no cover
    main()
