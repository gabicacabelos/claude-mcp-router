"""
Detección de tiers para el router de contexto.

Tier 0 — NO comprimir: código, stack traces, JSON payloads, claves
Tier 1 — Compresión ligera (~40%): docs cortos, contexto de proyecto
Tier 2 — Compresión fuerte (~70%): documentos grandes > 10k tokens
Tier 3 — Delegar a modelo barato: tareas de boilerplate puro
"""

import re
from enum import Enum
from pathlib import Path
from typing import Optional
import tiktoken


class Tier(Enum):
    ZERO = 0    # Sin compresión — integridad crítica
    ONE = 1     # Compresión ligera
    TWO = 2     # Compresión fuerte
    THREE = 3   # Delegación completa a modelo barato


TIER0_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml",
    ".sql", ".env", ".sh", ".bash",
    ".rs", ".go", ".java", ".c", ".cpp", ".h",
    ".xml", ".proto", ".graphql",
}

TIER0_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"at \w+\.\w+\(.*:\d+\)",
    r"Error:\s",
    r"BEGIN TRANSACTION",
    r"-----BEGIN [A-Z ]+ KEY-----",
    r'"[a-zA-Z_]+"\s*:\s*\{',
    r"0x[0-9a-fA-F]{8,}",
]

TIER3_PATTERNS = [
    r"Escrib[ei] un? (email|mensaje|respuesta) (estándar|genéric)",
    r"Genera[r]? \d+ (variaciones|versiones|ejemplos)",
    r"Traducir? (al|a) (inglés|español|portugués)",
    r"Resumir? en (una|1|dos|2) (línea|oraci)",
]


def _estimate_tokens(text: str) -> int:
    """Estimación rápida de tokens usando tiktoken (cl100k_base)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


class TierDetector:
    """
    Determina el tier de procesamiento para un fragmento de texto.
    Sin llamadas a API — 100% local y determinístico.
    """

    def __init__(
        self,
        tier1_max_tokens: int = 2000,
        tier2_min_tokens: int = 10000,
    ):
        self.tier1_max_tokens = tier1_max_tokens
        self.tier2_min_tokens = tier2_min_tokens
        self._tier0_patterns = [re.compile(p) for p in TIER0_PATTERNS]
        self._tier3_patterns = [re.compile(p, re.IGNORECASE) for p in TIER3_PATTERNS]

    def detect(self, text: str, file_path: Optional[str] = None) -> Tier:
        """
        Detecta el tier apropiado para el texto dado.

        Args:
            text: Contenido a analizar
            file_path: Ruta del archivo (opcional, ayuda con la extensión)

        Returns:
            Tier enum con la decisión
        """
        if file_path:
            ext = Path(file_path).suffix.lower()
            if ext in TIER0_EXTENSIONS:
                return Tier.ZERO

        for pattern in self._tier0_patterns:
            if pattern.search(text):
                return Tier.ZERO

        for pattern in self._tier3_patterns:
            if pattern.search(text):
                return Tier.THREE

        token_count = _estimate_tokens(text)

        if token_count <= self.tier1_max_tokens:
            return Tier.ONE
        elif token_count >= self.tier2_min_tokens:
            return Tier.TWO
        else:
            return Tier.ONE

    def estimate_tokens(self, text: str) -> int:
        """Expone la estimación de tokens para logging."""
        return _estimate_tokens(text)
