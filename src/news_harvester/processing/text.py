"""Limpieza de HTML para obtener texto plano."""

from __future__ import annotations

import re
import unicodedata

import trafilatura
from bs4 import BeautifulSoup

_REMOVABLE_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "header",
    "footer",
    "nav",
    "form",
}

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTIPLE_BREAKS_RE = re.compile(r"\n{2,}")

_DEFAULT_MIN_PARAGRAPH_CHARS = 0

_LINE_NOISE_EQUALS = {
    "01/05/2020 09h20",
    "audiencias vecinales",
    "blog",
    "blogs",
    "buenas prácticas",
    "cargando siguiente contenido",
    "clictómano",
    "club el comercio",
    "club del suscriptor",
    "columnistas",
    "contenido de",
    "copiar enlace",
    "corresponsales escolares",
    "daniel san román",
    "daniel san roman",
    "derechos arco",
    "día 1",
    "edición impresa",
    "editorial",
    "economía",
    "empresas",
    "españa",
    "estilos",
    "finanzas personales",
    "fotogalerías",
    "g de gestión",
    "gestión de servicios",
    "gestión tv",
    "inmobiliarias",
    "internacional",
    "juegos",
    "lo último",
    "lujo",
    "mag.",
    "management & empleo",
    "mercados",
    "méxico",
    "mix",
    "moda",
    "mundo",
    "no te pierdas",
    "notas contratadas",
    "opinión",
    "pasar al contenido principal",
    "perú",
    "peru quiosco",
    "política",
    "política de cookies",
    "política de privacidad",
    "política integrada de gestión",
    "políticas de privacidad",
    "politica de cookies",
    "politica de privacidad",
    "portada",
    "pregunta de hoy",
    "preguntas frecuentes",
    "privacy manager",
    "provecho",
    "¿quiénes somos?",
    "quiénes somos",
    "saltar intro",
    "siguiente artículo",
    "siguiente noticia",
    "tags relacionados",
    "te puede interesar",
    "tecnología",
    "tendencias",
    "terminos y condiciones",
    "términos y condiciones",
    "términos y condiciones de uso",
    "tu dinero",
    "últimas noticias",
    "ultimas noticias",
    "únete",
    "únete a el comercio",
    "unete a el comercio",
    "viajes",
    "videos",
}

_SECTION_HEADERS = {
    "tags relacionados",
    "no te pierdas",
    "contenido de",
    "videos recomendados",
    "te puede interesar",
}

_TERMINAL_HEADERS = {
    "no te pierdas",
    "contenido de",
    "videos recomendados",
    "te puede interesar",
}

_LINE_NOISE_PREFIXES = (
    "suscríbete",
    "síguenos en",
    "compartir en",
    "nota relacionada",
    "relacionado:",
    "ver también",
    "lee también",
    "publicidad",
    "clictómano |",
)

_DOMAIN_LINE_RE = re.compile(r"^[\w.-]+\.[a-z]{2,}$")


def _is_all_caps(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) < 3:
        return False
    return all(ch.isupper() for ch in letters)


def _is_short_navigation_item(line: str, normalized: str) -> bool:
    if any(ch in line for ch in ".?!;:") and normalized not in _LINE_NOISE_EQUALS:
        return False

    candidate = re.sub(r"[^\w\s]", " ", line)
    candidate = _WHITESPACE_RE.sub(" ", candidate).strip()
    if not candidate:
        return True

    words = candidate.split()
    if len(words) > 4:
        return False

    return all(len(word) <= 20 for word in words)


def extract_plain_text(
    html: str,
    *,
    keyword: str | list[str] | None = None,
    min_paragraph_chars: int | None = None,
    require_keyword: bool = False,
    strict_mode: bool = True,
) -> str:
    """Convierte HTML en un texto plano normalizado.

    Intenta usar trafilatura primero. Si falla o devuelve vacío, usa el método
    legacy basado en BeautifulSoup.
    """

    if not html or not html.strip():
        return ""

    keywords = [keyword] if isinstance(keyword, str) else (keyword or [])
    keywords_cf = [k.casefold() for k in keywords] if keywords else []

    # Intento principal con trafilatura
    try:
        traf_text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            target_language="es",
        )
        if traf_text:
            # Trafilatura devuelve texto limpio, pero aplicamos normalización básica
            normalized = unicodedata.normalize("NFC", traf_text)
            # Verificación de keyword si es requerida
            if require_keyword and keywords_cf:
                normalized_cf = normalized.casefold()
                if not any(k in normalized_cf for k in keywords_cf):
                    # Si trafilatura extrajo texto pero no tiene ninguna keyword,
                    # podría ser un falso positivo.
                    pass
                else:
                    return normalized.strip()
            else:
                return normalized.strip()
    except Exception:
        pass

    # --- FALLBACK LEGACY ---
    soup = BeautifulSoup(html, "lxml")
    for tag_name in _REMOVABLE_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    text = soup.get_text(separator="\n")
    text = unicodedata.normalize("NFC", text)

    skip_section = False
    filtered_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            if filtered_lines and filtered_lines[-1] == "":
                continue
            if filtered_lines:
                filtered_lines.append("")
            continue

        line = _WHITESPACE_RE.sub(" ", line)
        normalized = line.casefold()

        if normalized in _LINE_NOISE_EQUALS:
            if normalized in _SECTION_HEADERS:
                if normalized in _TERMINAL_HEADERS:
                    break
                skip_section = True
            continue

        if any(normalized.startswith(prefix) for prefix in _LINE_NOISE_PREFIXES):
            skip_section = True
            continue

        if _DOMAIN_LINE_RE.match(normalized):
            skip_section = True
            continue

        if skip_section:
            if _is_short_navigation_item(line, normalized):
                continue
            skip_section = False

        if _is_all_caps(line) and len(line) <= 80:
            continue

        filtered_lines.append(line)

    paragraphs: list[str] = []
    current: list[str] = []
    for line in filtered_lines:
        if line == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    if not paragraphs:
        return ""

    threshold = (
        _DEFAULT_MIN_PARAGRAPH_CHARS
        if min_paragraph_chars is None
        else max(0, min_paragraph_chars)
    )
    
    filtered_paragraphs: list[str] = []
    keyword_paragraph_found = False

    for paragraph in paragraphs:
        normalized_paragraph = _WHITESPACE_RE.sub(" ", paragraph).strip()
        if not normalized_paragraph:
            continue

        contains_keyword = False
        if keywords_cf:
             contains_keyword = any(k in normalized_paragraph.casefold() for k in keywords_cf)

        if threshold > 0 and len(normalized_paragraph) < threshold:
            # En modo estricto, descartamos párrafos cortos.
            # En modo laxo, si contiene la keyword, lo dejamos pasar aunque sea corto.
            if strict_mode:
                continue
            if not contains_keyword:
                continue

        filtered_paragraphs.append(normalized_paragraph)
        if contains_keyword:
            keyword_paragraph_found = True

    if require_keyword and keywords_cf and not keyword_paragraph_found:
        return ""

    if not filtered_paragraphs:
        return ""

    text = "\n\n".join(filtered_paragraphs)
    text = _MULTIPLE_BREAKS_RE.sub("\n", text)

    return text.strip()
