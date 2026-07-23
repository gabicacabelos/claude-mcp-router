"""
Mini-RAG local: chunking + ranking de relevancia 100% local, $0 APIs.

Motor híbrido:
  1. fastembed (embeddings ONNX locales, bge-small) — si está instalado
  2. BM25 puro-Python — fallback determinista, cero dependencias

Devuelve los top-k chunks más relevantes a una query, con números de línea.
Nunca lanza excepciones hacia afuera: si todo falla, degrada a los primeros chunks.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

CHUNK_TARGET_CHARS = 1600   # ~400 tokens por chunk
CHUNK_MAX_CHARS = 2600      # corte duro si no aparece un límite natural
CHUNK_OVERLAP_LINES = 3

# Líneas que son buenos puntos de corte (inicio de sección/símbolo)
_BOUNDARY = re.compile(
    r"^\s*(#{1,6}\s|def\s|class\s|function\s|async\s+def\s|export\s|const\s+\w+\s*=|"
    r"public\s|private\s|=== |--- |\*\*\*|<h[1-6])"
)

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_STOPWORDS = frozenset(
    "the a an of in on at to for and or is are was were be been it this that with as by from "
    "el la los las un una de del en y o es son fue que con como por para se su al lo si no".split()
)


def tokenize(text: str) -> list[str]:
    """Tokeniza con split de snake_case y camelCase. Lowercase, sin stopwords."""
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        for part in raw.split("_"):
            for sub in _CAMEL.split(part):
                s = sub.lower()
                if len(s) > 1 and s not in _STOPWORDS:
                    out.append(s)
    return out


@dataclass
class Chunk:
    text: str
    start_line: int  # 1-based inclusive
    end_line: int    # 1-based inclusive
    score: float = 0.0
    tokens: list[str] = field(default_factory=list, repr=False)


def chunk_text(text: str) -> list[Chunk]:
    """Divide texto en chunks por líneas, cortando en límites naturales cuando puede."""
    lines = text.split("\n")
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_chars = 0
    start = 1

    def flush(end_line: int):
        nonlocal buf, buf_chars, start
        if buf and any(l.strip() for l in buf):
            chunks.append(Chunk(text="\n".join(buf), start_line=start, end_line=end_line))
        # overlap: retroceder unas líneas para no cortar contexto
        overlap = buf[-CHUNK_OVERLAP_LINES:] if len(buf) > CHUNK_OVERLAP_LINES else []
        start = end_line - len(overlap) + 1
        buf = list(overlap)
        buf_chars = sum(len(l) + 1 for l in buf)

    for i, line in enumerate(lines, start=1):
        boundary = _BOUNDARY.match(line) or not line.strip()
        if buf_chars >= CHUNK_TARGET_CHARS and boundary:
            flush(i - 1)
        elif buf_chars >= CHUNK_MAX_CHARS:
            flush(i - 1)
        buf.append(line)
        buf_chars += len(line) + 1

    if buf and any(l.strip() for l in buf):
        chunks.append(Chunk(text="\n".join(buf), start_line=start, end_line=len(lines)))
    return chunks


# ─── BM25 ────────────────────────────────────────────────────────────────────

def _bm25(query_tokens: list[str], chunks: list[Chunk]) -> list[float]:
    k1, b = 1.5, 0.75
    n_docs = len(chunks)
    if n_docs == 0:
        return []
    for c in chunks:
        if not c.tokens:
            c.tokens = tokenize(c.text)
    avgdl = max(1.0, sum(len(c.tokens) for c in chunks) / n_docs)
    df: Counter = Counter()
    for c in chunks:
        for t in set(c.tokens):
            df[t] += 1
    scores: list[float] = []
    for c in chunks:
        tf = Counter(c.tokens)
        dl = len(c.tokens)
        s = 0.0
        for q in query_tokens:
            n = df.get(q, 0)
            if n == 0:
                continue
            idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
            f = tf.get(q, 0)
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


# ─── fastembed (opcional) ────────────────────────────────────────────────────

_EMBEDDER = None
_EMBED_TRIED = False


def _get_embedder():
    """Lazy-load de fastembed. Si no está instalado o falla, queda deshabilitado."""
    global _EMBEDDER, _EMBED_TRIED
    if _EMBED_TRIED:
        return _EMBEDDER
    _EMBED_TRIED = True
    try:
        from fastembed import TextEmbedding  # type: ignore
        _EMBEDDER = TextEmbedding("BAAI/bge-small-en-v1.5")
        logger.info("fastembed activo — ranking híbrido BM25+embeddings")
    except Exception as e:
        logger.info(f"fastembed no disponible ({type(e).__name__}) — ranking BM25 puro")
        _EMBEDDER = None
    return _EMBEDDER


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed_scores(query: str, chunks: list[Chunk]) -> list[float] | None:
    emb = _get_embedder()
    if emb is None:
        return None
    try:
        q_vec = list(emb.embed([query]))[0]
        c_vecs = list(emb.embed([c.text[:2000] for c in chunks]))
        return [_cosine(q_vec, v) for v in c_vecs]
    except Exception as e:
        logger.warning(f"fastembed falló en runtime: {e} — usando BM25")
        return None


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


# ─── API pública ─────────────────────────────────────────────────────────────

def rank_chunks(text: str, query: str, top_k: int = 4) -> tuple[list[Chunk], str]:
    """
    Devuelve (top_k chunks ordenados por posición en el archivo, motor_usado).
    Determinista con BM25; híbrido si fastembed está disponible.
    """
    chunks = chunk_text(text)
    if not chunks:
        return [], "none"
    if len(chunks) <= top_k:
        return chunks, "all"

    q_tokens = tokenize(query)
    bm25 = _normalize(_bm25(q_tokens, chunks))
    emb = _embed_scores(query, chunks)
    engine = "bm25"
    if emb is not None:
        emb_n = _normalize(emb)
        combined = [0.45 * b + 0.55 * e for b, e in zip(bm25, emb_n)]
        engine = "hybrid"
    else:
        combined = bm25

    for c, s in zip(chunks, combined):
        c.score = round(s, 4)

    top = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
    # Si nada matcheó (todo score 0), devolver los primeros chunks como degradación
    if all(c.score == 0.0 for c in top):
        top = chunks[:top_k]
        engine += "_fallback_head"
    # Reordenar por posición para que Claude lea en orden natural
    top.sort(key=lambda c: c.start_line)
    return top, engine


_OUTLINE = re.compile(
    r"^\s*(#{1,6}\s+[^\W─═-].*|def\s+\w+.*|class\s+\w+.*|async\s+def\s+\w+.*|function\s+\w+.*|"
    r"export\s+(default\s+)?(function|class|const)\s+\w+.*|"
    r"CREATE\s+(TABLE|INDEX|VIEW)\s+\w+.*|\[[\w.-]+\]|@app\.\w+.*|@mcp\.tool.*)",
    re.IGNORECASE,
)


def build_outline(text: str, max_items: int = 60) -> list[str]:
    """Mapa estructural del archivo: headings, defs, clases — con número de línea."""
    out: list[str] = []
    for i, line in enumerate(text.split("\n"), start=1):
        if _OUTLINE.match(line):
            out.append(f"L{i}:{line.strip()[:110]}")
            if len(out) >= max_items:
                break
    return out
