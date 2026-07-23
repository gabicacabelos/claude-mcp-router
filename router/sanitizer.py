"""
Limpieza local de texto — $0 APIs, determinista, sin pérdida de información.

- strip_html: elimina scripts, estilos, tags y boilerplate de payloads web
- clean_text: normaliza espacios, líneas en blanco y caracteres invisibles

Un payload web limpio puede pesar 40-60% menos en tokens sin perder contenido.
"""

import html as _html
import re

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".toml",
    ".sql", ".sh", ".bash", ".rs", ".go", ".java", ".c", ".cpp", ".h",
    ".xml", ".proto", ".graphql", ".env", ".ini", ".cfg",
}

_RE_SCRIPT = re.compile(r"<(script|style|noscript|svg|iframe|head)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_BLOCK = re.compile(
    r"</?(p|div|br|li|ul|ol|tr|td|th|table|h[1-6]|section|article|header|footer|nav|main|blockquote)\b[^>]*>",
    re.I,
)
_RE_TAG = re.compile(r"<[^>]{1,500}>")
_RE_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_RE_MULTI_BLANK = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def looks_like_html(text: str) -> bool:
    head = text[:4000]
    if re.search(r"<!DOCTYPE\s+html|<html\b|<body\b|<head\b", head, re.I):
        return True
    # Densidad de tags de cierre como heurística para fragmentos
    return text.count("</") > 15 and text.count("<div") > 3


def strip_html(text: str) -> str:
    """Convierte HTML a texto plano legible. Local, sin dependencias."""
    text = _RE_SCRIPT.sub(" ", text)
    text = _RE_COMMENT.sub(" ", text)
    text = _RE_BLOCK.sub("\n", text)
    text = _RE_TAG.sub(" ", text)
    text = _html.unescape(text)
    return clean_text(text, is_code=False)


def clean_text(text: str, is_code: bool = False) -> str:
    """
    Normalización conservadora:
    - Siempre: line endings, caracteres zero-width, trailing whitespace, máx 1 línea en blanco
    - Solo prosa (no código): colapsa runs de espacios internos
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_ZERO_WIDTH.sub("", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _RE_MULTI_BLANK.sub("\n\n", text)
    if not is_code:
        # No tocar bloques de código markdown (indentación significativa)
        parts = text.split("```")
        for i in range(0, len(parts), 2):  # índices pares = fuera de code fences
            parts[i] = _RE_MULTI_SPACE.sub(" ", parts[i])
        text = "```".join(parts)
    return text.strip()


def is_code_file(file_path: str) -> bool:
    from pathlib import Path
    return Path(file_path).suffix.lower() in _CODE_EXTENSIONS


def sanitize_file_content(content: str, file_path: str) -> tuple[str, bool]:
    """
    Pipeline completo para contenido de archivo.
    Returns: (texto_limpio, era_html)

    La extensión manda sobre la heurística: un archivo de código NUNCA se
    procesa como HTML. Un scraper, un template o un test con fixtures contienen
    literales '<html>'/'<div>' y la heurística los daba por página web,
    destruyendo el código (tags borrados). La fidelidad exacta es la promesa
    central: ante la duda, no tocar.
    """
    if is_code_file(file_path):
        return clean_text(content, is_code=True), False
    if looks_like_html(content):
        return strip_html(content), True
    return clean_text(content, is_code=False), False
