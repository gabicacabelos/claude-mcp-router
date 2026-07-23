"""
Inbox: cola de órdenes cruzadas entre clientes de Claude.

Cowork, Claude Code y Desktop no pueden comandarse entre sí en tiempo real —
pero comparten este disco. El inbox es el buzón asíncrono: un cliente deja una
orden (opcionalmente vinculada a un checkpoint con todo el contexto de la tarea),
otro cliente la consume, la ejecuta y reporta el resultado.

Flujo típico:
  [Cowork]  inbox send to=code message="migrar los tests a pytest" checkpoint="refactor-auth"
  [Code]    inbox check to=code   → ve la orden + el resumen del checkpoint
  [Code]    checkpoint resume "refactor-auth"  → contexto completo en ~300 tokens
  [Code]    inbox complete id=1 result="tests migrados, 34/34 verdes"
  [Cowork]  inbox history        → ve el resultado
"""

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DONE_RETENTION_DAYS = 30


class Inbox:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS inbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                to_client TEXT NOT NULL DEFAULT 'any',
                from_client TEXT NOT NULL DEFAULT 'unknown',
                message TEXT NOT NULL,
                checkpoint TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                created_at REAL NOT NULL,
                done_at REAL
            )
        """)
        self._db.commit()
        try:
            cutoff = time.time() - DONE_RETENTION_DAYS * 86400
            self._db.execute("DELETE FROM inbox WHERE status='done' AND done_at < ?", (cutoff,))
            self._db.commit()
        except Exception as e:
            logger.warning(f"purga de inbox falló: {e}")

    def send(self, message: str, to_client: str = "any", from_client: str = "unknown",
             checkpoint: str | None = None) -> int:
        cur = self._db.execute(
            "INSERT INTO inbox (to_client, from_client, message, checkpoint, created_at) VALUES (?,?,?,?,?)",
            (to_client.lower().strip() or "any", from_client.lower().strip() or "unknown",
             message, checkpoint, time.time()),
        )
        self._db.commit()
        return cur.lastrowid

    def check(self, to_client: str | None = None) -> list[dict]:
        """Órdenes pendientes para un cliente (incluye las dirigidas a 'any')."""
        if to_client:
            rows = self._db.execute(
                "SELECT id, to_client, from_client, message, checkpoint, created_at FROM inbox "
                "WHERE status='pending' AND (to_client=? OR to_client='any') ORDER BY created_at",
                (to_client.lower().strip(),),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, to_client, from_client, message, checkpoint, created_at FROM inbox "
                "WHERE status='pending' ORDER BY created_at"
            ).fetchall()
        return [
            {"id": r[0], "to": r[1], "from": r[2], "message": r[3], "checkpoint": r[4],
             "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(r[5]))}
            for r in rows
        ]

    def complete(self, order_id: int, result: str | None = None) -> bool:
        cur = self._db.execute(
            "UPDATE inbox SET status='done', result=?, done_at=? WHERE id=? AND status='pending'",
            (result, time.time(), order_id),
        )
        self._db.commit()
        return cur.rowcount > 0

    def history(self, limit: int = 10) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, to_client, from_client, message, result, done_at FROM inbox "
            "WHERE status='done' ORDER BY done_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "to": r[1], "from": r[2], "message": r[3], "result": r[4],
             "done": time.strftime("%Y-%m-%d %H:%M", time.localtime(r[5])) if r[5] else None}
            for r in rows
        ]

    def close(self) -> None:
        self._db.close()
