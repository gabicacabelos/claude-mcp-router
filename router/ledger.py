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
from pathlib import Path

logger = logging.getLogger(__name__)

SNAPSHOT_MAX_CHARS = 400_000   # no guardar snapshots de archivos gigantes
DIFF_MAX_RATIO = 0.6           # si el diff pesa >60% del archivo, no rinde: devolver normal


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
        self._db.commit()

    @staticmethod
    def hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

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
        return {"files_tracked": row[0], "total_reads": row[1], "tokens_tracked": row[2]}

    def close(self) -> None:
        self._db.close()
