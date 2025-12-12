"""Limpieza de HTML para obtener texto plano con estrategia de fallback."""

from __future__ import annotations

import re
import unicodedata
import logging

import trafilatura
from bs4 import BeautifulSoup
from readability import Document
import justext

from ..domains import DOMAIN_SELECTORS

# Configure logging
logger = logging.getLogger(__name__)

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


# --- EXTRACTION LADDER METHODS ---

def _extract_trafilatura(html: str) -> str | None:
    try:
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            target_language="es",
        )
    except Exception:
        return None

def _extract_readability(html: str) -> str | None:
    try:
        doc = Document(html)
        # summary() devuelve HTML simplificado, necesitamos pasarlo a texto
        # Usamos BeautifulSoup para limpiar ese HTML simplificado
        clean_html = doc.summary()
        soup = BeautifulSoup(clean_html, "lxml")
        return soup.get_text(separator="\n")
    except Exception:
        return None

def _extract_justext(html: str) -> str | None:
    try:
        paragraphs = justext.justext(html, justext.get_stoplist("Spanish"))
        text = "\n".join([p.text for p in paragraphs if not p.is_boilerplate])
        return text if text.strip() else None
    except Exception:
        return None

def _extract_selectors(html: str, domain: str) -> str | None:
    if not domain or domain not in DOMAIN_SELECTORS:
        return None
    
    selectors = DOMAIN_SELECTORS[domain]
    try:
        soup = BeautifulSoup(html, "lxml")
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(separator="\n")
    except Exception:
        pass
    
    return None

def _extract_legacy_heuristics(html: str) -> str | None:
    """Fallback actual basado en reglas personalizadas para Lisbeth."""
    try:
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
            return None

        return "\n\n".join(paragraphs)
    except Exception:
        return None


def _filter_paragraphs(
    text: str,
    threshold: int,
    keywords_cf: list[str],
    strict_mode: bool
) -> str:
    if not text:
        return ""
    
    # Normalizar espacios y saltos antes de filtrar
    lines = text.splitlines()
    filtered: list[str] = []
    
    for line in lines:
        normalized_line = _WHITESPACE_RE.sub(" ", line).strip()
        if not normalized_line:
            continue
            
        if threshold > 0:
            # Check length
            if len(normalized_line) < threshold:
                contains_keyword = False
                if keywords_cf:
                     contains_keyword = any(k in normalized_line.casefold() for k in keywords_cf)
                
                if strict_mode:
                    continue
                if not contains_keyword:
                    continue
        
        filtered.append(normalized_line)
        
    return "\n\n".join(filtered)


def extract_plain_text(
    html: str,
    *,
    keyword: str | list[str] | None = None,
    min_paragraph_chars: int | None = None,
    require_keyword: bool = False,
    strict_mode: bool = True,
    domain: str | None = None,
) -> str:
    """Convierte HTML en un texto plano normalizado usando una estrategia escalonada."""

    if not html or not html.strip():
        return ""

    keywords = [keyword] if isinstance(keyword, str) else (keyword or [])
    keywords_cf = [k.casefold() for k in keywords] if keywords else []
    
    threshold = (
        _DEFAULT_MIN_PARAGRAPH_CHARS
        if min_paragraph_chars is None
        else max(0, min_paragraph_chars)
    )

    def _process_candidate(text: str | None) -> str | None:
        if not text:
            return None
        
        # 1. Aplicar filtrado de párrafos (restore legacy logic)
        filtered_text = _filter_paragraphs(text, threshold, keywords_cf, strict_mode)
        if not filtered_text:
            return None
            
        normalized = unicodedata.normalize("NFC", filtered_text).strip()
        
        # 2. Quality Gate
        # - Longitud mínima total (para evitar stubs)
        if len(normalized) < 200: 
            return None
            
        # - Keyword obligatorio
        if require_keyword and keywords_cf:
            normalized_cf = normalized.casefold()
            if not any(k in normalized_cf for k in keywords_cf):
                return None
                
        return normalized

    # Escalera de extracción
    # 1. Trafilatura
    candidate = _extract_trafilatura(html)
    processed = _process_candidate(candidate)
    if processed: return processed

    # 2. Readability
    candidate = _extract_readability(html)
    processed = _process_candidate(candidate)
    if processed: return processed
        
    # 3. jusText
    candidate = _extract_justext(html)
    processed = _process_candidate(candidate)
    if processed: return processed

    # 4. Selectores Específicos
    if domain:
        candidate = _extract_selectors(html, domain)
        processed = _process_candidate(candidate)
        if processed: return processed

    # 5. Legacy Heuristics
    candidate = _extract_legacy_heuristics(html)
    processed = _process_candidate(candidate)
    if processed: return processed

    # Fallo total
    return ""
