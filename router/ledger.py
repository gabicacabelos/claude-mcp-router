"""
Ledger de ingesta: memoria persistente de qué archivos leyó Claude, cross-sesión
y cross-cliente (Desktop, Code, Cowork comparten este proceso/DB).

El desperdicio #1 en uso diario real es re-leer los mismos archivos en cada
sesión nueva. El ledger lo elimina:
  - archivo sin cambios → respuesta "unchanged" con outline (~50 tokens vs miles)
  - archivo modificado  → SOLO el diff unificado contra el snapshot guardado

SQLite síncrono a propósito: operaciones <1ms, sin complejidad async.
"""

import difflib
import hashlib
import json
import logging
import sqlite3
import time
from array import array
from pathlib import Path

logger = logging.getLogger(__name__)

SNAPSHOT_MAX_CHARS = 400_000   # no guardar snapshots de archivos gigantes
DIFF_MAX_RATIO = 0.6           # si el diff pesa >60% del archivo, no rinde: devolver normal
PURGE_AFTER_DAYS = 30          # entradas sin lecturas hace >30 días se eliminan
DB_MAX_BYTES = 100 * 1024 * 1024  # 100MB: por encima se descartan los snapshots más viejos


class FileLedger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS file_ledger (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                snapshot TEXT,
                outline TEXT,
                tokens INTEGER,
                first_seen REAL,
                last_seen REAL,
                reads INTEGER DEFAULT 1
            )
        """)
        # Cache de ranking por query: (contenido, query, top_k) → line ranges.
        # Keyed por el hash del contenido sanitizado → si el archivo cambia,
        # cambia el hash y la entrada vieja queda inalcanzable (invalidación automática).
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                file_hash TEXT NOT NULL,
                query_norm TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                ranges TEXT NOT NULL,
                created REAL,
                PRIMARY KEY (file_hash, query_norm, top_k)
            )
        """)
        # Cache de vectores de chunk (fastembed) por file_hash — evita re-embeber
        # todos los chunks en cada query. Un vector por fila (float32 empaquetado).
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                file_hash TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (file_hash, chunk_idx)
            )
        """)
        self._db.commit()
        try:
            self.purge()
        except Exception as e:
            logger.warning(f"purga del ledger falló: {e}")

    def purge(self) -> dict:
        """
        Mantenimiento automático (corre al iniciar el servidor):
        1. Elimina entradas sin lecturas hace > PURGE_AFTER_DAYS
        2. Si la DB supera DB_MAX_BYTES, vacía los snapshots más antiguos
           (se conserva hash+outline: 'unchanged' sigue funcionando, solo se pierde el diff)
        """
        cutoff = time.time() - PURGE_AFTER_DAYS * 86400
        cur = self._db.execute("DELETE FROM file_ledger WHERE last_seen < ?", (cutoff,))
        expired = cur.rowcount
        self._db.commit()

        snapshots_dropped = 0
        db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        if db_size > DB_MAX_BYTES:
            rows = self._db.execute(
                "SELECT path, LENGTH(snapshot) FROM file_ledger WHERE snapshot IS NOT NULL ORDER BY last_seen ASC"
            ).fetchall()
            to_free = db_size - int(DB_MAX_BYTES * 0.8)  # bajar al 80% del límite
            freed = 0
            for path, size in rows:
                if freed >= to_free:
                    break
                self._db.execute("UPDATE file_ledger SET snapshot=NULL WHERE path=?", (path,))
                freed += size or 0
                snapshots_dropped += 1
            self._db.commit()
            self._db.execute("VACUUM")

        # Cachés derivados: si el file_hash ya no vive en el ledger, sus entradas
        # de query/vectores son inalcanzables → se limpian para no inflar el disco.
        self._db.execute("DELETE FROM query_cache WHERE file_hash NOT IN (SELECT hash FROM file_ledger)")
        self._db.execute("DELETE FROM chunk_vectors WHERE file_hash NOT IN (SELECT hash FROM file_ledger)")
        self._db.commit()

        if expired or snapshots_dropped:
            logger.info(f"Ledger: {expired} entradas expiradas, {snapshots_dropped} snapshots liberados")
        return {"expired": expired, "snapshots_dropped": snapshots_dropped}

    @staticmethod
    def hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_query(query: str) -> str:
        """Normaliza una query para cache: colapsa espacios y case-fold."""
        return " ".join(query.split()).casefold()

    def get(self, path: str) -> dict | None:
        row = self._db.execute(
            "SELECT hash, snapshot, outline, tokens, first_seen, last_seen, reads FROM file_ledger WHERE path=?",
            (path,),
        ).fetchone()
        if not row:
            return None
        return {
            "hash": row[0], "snapshot": row[1], "outline": json.loads(row[2] or "[]"),
            "tokens": row[3], "first_seen": row[4], "last_seen": row[5], "reads": row[6],
        }

    def record(self, path: str, content: str, outline: list[str], tokens: int) -> None:
        h = self.hash(content)
        snapshot = content if len(content) <= SNAPSHOT_MAX_CHARS else None
        now = time.time()
        self._db.execute(
            """
            INSERT INTO file_ledger (path, hash, snapshot, outline, tokens, first_seen, last_seen, reads)
            VALUES (?,?,?,?,?,?,?,1)
            ON CONFLICT(path) DO UPDATE SET
                hash=excluded.hash, snapshot=excluded.snapshot, outline=excluded.outline,
                tokens=excluded.tokens, last_seen=excluded.last_seen, reads=reads+1
            """,
            (path, h, snapshot, json.dumps(outline, ensure_ascii=False), tokens, now, now),
        )
        self._db.commit()

    def touch(self, path: str) -> None:
        """Actualiza last_seen y contador sin re-escribir snapshot."""
        self._db.execute(
            "UPDATE file_ledger SET last_seen=?, reads=reads+1 WHERE path=?",
            (time.time(), path),
        )
        self._db.commit()

    def diff(self, old: str, new: str, context_lines: int = 2) -> str | None:
        """
        Diff unificado old→new. Devuelve None si el diff no rinde
        (muy grande respecto al archivo) — en ese caso conviene contenido normal.
        """
        d = "\n".join(
            difflib.unified_diff(
                old.split("\n"), new.split("\n"),
                fromfile="antes", tofile="ahora", lineterm="", n=context_lines,
            )
        )
        if not d:
            return ""
        if len(d) > len(new) * DIFF_MAX_RATIO:
            return None
        return d

    # ─── Cache de ranking por query (#1) ──────────────────────────────────────

    def get_query_cache(self, file_hash: str, query_norm: str, top_k: int) -> list[tuple[int, int]] | None:
        """Line ranges cacheados para (file_hash, query, top_k). None si no hay hit."""
        row = self._db.execute(
            "SELECT ranges FROM query_cache WHERE file_hash=? AND query_norm=? AND top_k=?",
            (file_hash, query_norm, top_k),
        ).fetchone()
        if not row:
            return None
        return [(int(a), int(b)) for a, b in json.loads(row[0])]

    def put_query_cache(self, file_hash: str, query_norm: str, top_k: int,
                        ranges: list[tuple[int, int]]) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO query_cache (file_hash, query_norm, top_k, ranges, created) "
            "VALUES (?,?,?,?,?)",
            (file_hash, query_norm, top_k, json.dumps([[int(a), int(b)] for a, b in ranges]), time.time()),
        )
        self._db.commit()

    # ─── Cache de vectores de embeddings (#2) ─────────────────────────────────

    def get_chunk_vectors(self, file_hash: str) -> list[list[float]] | None:
        """Vectores de chunk cacheados para file_hash (en orden), o None si no hay."""
        rows = self._db.execute(
            "SELECT vector FROM chunk_vectors WHERE file_hash=? ORDER BY chunk_idx",
            (file_hash,),
        ).fetchall()
        if not rows:
            return None
        return [list(array("f", r[0])) for r in rows]

    def put_chunk_vectors(self, file_hash: str, vectors: list) -> None:
        """Persiste los vectores de chunk (float32 empaquetado) reemplazando los previos."""
        self._db.execute("DELETE FROM chunk_vectors WHERE file_hash=?", (file_hash,))
        self._db.executemany(
            "INSERT INTO chunk_vectors (file_hash, chunk_idx, vector) VALUES (?,?,?)",
            [(file_hash, i, array("f", [float(x) for x in v]).tobytes()) for i, v in enumerate(vectors)],
        )
        self._db.commit()

    def check_files(self, files: list[dict]) -> list[dict]:
        """
        Para checkpoint/resume: compara hashes guardados en el checkpoint contra
        el estado actual en disco. files = [{"path":..., "hash":...}]
        Returns lista con estado por archivo: unchanged | changed | deleted | unknown
        """
        out = []
        for f in files:
            p = Path(f["path"])
            if not p.is_file():
                out.append({"path": f["path"], "state": "deleted"})
                continue
            try:
                current = self.hash(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                out.append({"path": f["path"], "state": "unknown"})
                continue
            state = "unchanged" if current == f.get("hash") else "changed"
            out.append({"path": f["path"], "state": state})
        return out

    def stats(self) -> dict:
        row = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(reads),0), COALESCE(SUM(tokens),0) FROM file_ledger"
        ).fetchone()
        size_kb = round(Path(self.db_path).stat().st_size / 1024, 1) if Path(self.db_path).exists() else 0
        return {"files_tracked": row[0], "total_reads": row[1], "tokens_tracked": row[2], "db_size_kb": size_kb}

    def close(self) -> None:
        self._db.close()
