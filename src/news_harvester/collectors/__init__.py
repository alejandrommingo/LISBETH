"""Colecciones de fuentes de noticias externas."""

from .gdelt import Article, GDELTError, fetch_articles, download_article_bodies

__all__ = [
    "Article",
    "GDELTError",
    "fetch_articles",
    "download_article_bodies",
]
