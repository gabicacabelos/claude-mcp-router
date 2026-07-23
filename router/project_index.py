"""
Índice de proyecto: búsqueda BM25 cross-archivo, incremental y 100% local.

`smart_read` responde "¿dónde dice X en ESTE archivo?". La pregunta real del
usuario suele ser "¿dónde se maneja X en el PROYECTO?" — y ahí Claude cae en
Grep, que es literal y sin estado.

Este índice:
  - Es incremental: re-indexa SOLO los archivos cuyo hash cambió (reusa la
    infraestructura de invalidación por hash que ya existe en el ledger).
  - Es determinista: BM25 puro-Python sobre el mismo tokenizer del ranker.
    Devuelve fragmentos EXACTOS, nunca resúmenes.
  - Mejora con el uso: el índice vive en el ledger y persiste cross-sesión y
    cross-cliente. Grep arranca de cero cada vez; esto no.
"""

import logging
import math
import time
from collections import Counter
from pathlib import Path

from .ranker import rank_chunks, tokenize
from .sanitizer import sanitize_file_content

logger = logging.getLogger(__name__)

# Extensiones indexables por default: código y docs. Deliberadamente acotado —
# indexar todo el disco es la forma más rápida de inflar la DB sin dar valor.
INDEXABLE_SUFFIXES = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".cpp", ".h", ".cs", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".sql", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json",
    ".md", ".rst", ".txt",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist",
    "build", ".next", ".nuxt", "target", ".pytest_cache", ".mypy_cache",
    "vendor", ".idea", ".vscode", "coverage", ".tox", "site-packages",
}

MAX_FILE_BYTES = 400_000   # archivos gigantes no rinden en un índice léxico
MAX_FILES = 2000           # techo duro: mantiene la indexación en segundos


def _iter_candidates(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in INDEXABLE_SUFFIXES:
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield p


def build_index(ledger, root: str | Path, max_files: int = MAX_FILES) -> dict:
    """
    Indexa (o re-indexa) un proyecto. Solo toca lo que cambió:
      - archivo nuevo o con hash distinto → se re-tokeniza
      - archivo sin cambios               → se saltea (costo ~0)
      - archivo borrado                   → se saca del índice
    """
    root_p = Path(root).resolve()
    root_key = str(root_p)
    t0 = time.time()
    indexed = skipped = 0
    seen: set[str] = set()

    for p in _iter_candidates(root_p):
        if indexed + skipped >= max_files:
            break
        path_key = str(p)
        seen.add(path_key)
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        content, _ = sanitize_file_content(raw, path_key)
        h = ledger.hash(content)
        if ledger.indexed_hash(path_key) == h:
            skipped += 1
            continue
        ledger.put_indexed(path_key, root_key, h, tokenize(content))
        indexed += 1

    stale = [p for p in ledger.indexed_paths(root_key) if p not in seen]
    removed = ledger.drop_indexed(stale)

    return {
        "root": root_key,
        "indexed": indexed,
        "unchanged": skipped,
        "removed": removed,
        "total_docs": len(ledger.indexed_paths(root_key)),
        "took_s": round(time.time() - t0, 2),
    }


def search(ledger, root: str | Path, query: str, top_k: int = 5,
           fragments_per_file: int = 1) -> list[dict]:
    """
    BM25 a nivel documento para elegir los archivos, y después el ranker por
    chunks (ya existente) para extraer el fragmento exacto dentro de cada uno.
    Dos etapas: barato para descartar, preciso para mostrar.
    """
    root_key = str(Path(root).resolve())
    docs = ledger.indexed_docs(root_key)
    if not docs:
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    n_docs = len(docs)
    avgdl = max(1.0, sum(len(t) for _, t in docs) / n_docs)
    df: Counter = Counter()
    for _, toks in docs:
        for t in set(toks):
            df[t] += 1

    k1, b = 1.5, 0.75
    scored = []
    for path, toks in docs:
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for q in q_tokens:
            n = df.get(q, 0)
            if n == 0:
                continue
            idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
            f = tf.get(q, 0)
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        if s > 0:
            scored.append((s, path))

    scored.sort(reverse=True)
    results = []
    for score, path in scored[:top_k]:
        entry = {"file": path, "score": round(score, 4), "fragments": []}
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
            content, _ = sanitize_file_content(raw, path)
            top, _engine = rank_chunks(content, query, top_k=fragments_per_file)
            entry["fragments"] = [
                {"lines": f"{c.start_line}-{c.end_line}", "text": c.text}
                for c in top
            ]
        except Exception as e:
            entry["fragments_error"] = str(e)[:120]
        results.append(entry)
    return results
