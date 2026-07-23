"""
Inbox: cola de órdenes cruzadas entre clientes de Claude.

Cowork, Claude Code, Desktop y Claude Design no pueden comandarse entre sí en
tiempo real — pero comparten este disco. El inbox es el buzón asíncrono: un
cliente deja una orden (opcionalmente vinculada a un checkpoint con todo el
contexto de la tarea y a assets de handoff), otro cliente la consume, la
ejecuta y reporta el resultado.

Flujo típico (código ↔ diseño):
  [Code]    inbox send to=design message="hacé el hero de la landing"
                       assets=["/proj/brief.md","https://.../wireframe.png"]
  [Design]  inbox check to=design  → ve la orden, el checkpoint y los assets
  [Design]  inbox complete id=1 result="mockup listo"
                       assets=["https://.../hero-v1.fig","/exports/hero.png"]
  [Code]    inbox history          → ve el resultado + los assets devueltos
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DONE_RETENTION_DAYS = 30

# Clientes reconocidos del "pack". El inbox acepta cualquier string como
# destino (texto libre), pero estos son los roles de primera clase que
# los clientes chequean por convención al arrancar una sesión.
KNOWN_CLIENTS = ("cowork", "code", "desktop", "design", "any")


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
                assets TEXT,
                result_assets TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                created_at REAL NOT NULL,
                done_at REAL
            )
        """)
        self._migrate()
        self._db.commit()
        try:
            cutoff = time.time() - DONE_RETENTION_DAYS * 86400
            self._db.execute("DELETE FROM inbox WHERE status='done' AND done_at < ?", (cutoff,))
            self._db.commit()
        except Exception as e:
            logger.warning(f"purga de inbox falló: {e}")

    def _migrate(self) -> None:
        """Agrega columnas nuevas a DBs viejas sin romperlas (ALTER idempotente)."""
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(inbox)").fetchall()}
        for col in ("assets", "result_assets"):
            if col not in cols:
                try:
                    self._db.execute(f"ALTER TABLE inbox ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError as e:
                    logger.warning(f"migración inbox ({col}) falló: {e}")

    @staticmethod
    def _dump_assets(assets) -> str | None:
        """Normaliza una lista de assets (rutas/URLs) a JSON, o None."""
        if not assets:
            return None
        if isinstance(assets, str):
            assets = [assets]
        clean = [str(a).strip() for a in assets if str(a).strip()]
        return json.dumps(clean, ensure_ascii=False) if clean else None

    @staticmethod
    def _load_assets(raw) -> list[str]:
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    def find_pending_duplicate(self, to_client: str, message: str) -> dict | None:
        """
        Busca una orden PENDIENTE con el mismo destino exacto y el mismo mensaje
        (normalizado: espacios colapsados, sin distinguir mayúsculas). Evita que
        una orden se acumule dos veces en el buzón mientras la anterior sigue sin
        resolver. No hace match contra 'any' cruzado — el destino debe ser idéntico.
        """
        to_client = (to_client or "any").lower().strip()
        norm = " ".join(message.split()).casefold()
        rows = self._db.execute(
            "SELECT id, to_client, from_client, message, checkpoint, assets, created_at FROM inbox "
            "WHERE status='pending' AND to_client=?",
            (to_client,),
        ).fetchall()
        for r in rows:
            if " ".join(r[3].split()).casefold() == norm:
                return {
                    "id": r[0], "to": r[1], "from": r[2], "message": r[3], "checkpoint": r[4],
                    "assets": self._load_assets(r[5]),
                    "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(r[6])),
                }
        return None

    def send(self, message: str, to_client: str = "any", from_client: str = "unknown",
             checkpoint: str | None = None, assets=None, allow_duplicate: bool = False) -> int:
        to_client = to_client.lower().strip() or "any"
        if not allow_duplicate:
            dup = self.find_pending_duplicate(to_client, message)
            if dup:
                logger.info(f"inbox.send: orden duplicada evitada — reusando id {dup['id']} (to='{to_client}')")
                return dup["id"]
        cur = self._db.execute(
            "INSERT INTO inbox (to_client, from_client, message, checkpoint, assets, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (to_client, from_client.lower().strip() or "unknown",
             message, checkpoint, self._dump_assets(assets), time.time()),
        )
        self._db.commit()
        return cur.lastrowid

    def check(self, to_client: str | None = None) -> list[dict]:
        """Órdenes pendientes para un cliente (incluye las dirigidas a 'any')."""
        if to_client:
            rows = self._db.execute(
                "SELECT id, to_client, from_client, message, checkpoint, assets, created_at FROM inbox "
                "WHERE status='pending' AND (to_client=? OR to_client='any') ORDER BY created_at",
                (to_client.lower().strip(),),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, to_client, from_client, message, checkpoint, assets, created_at FROM inbox "
                "WHERE status='pending' ORDER BY created_at"
            ).fetchall()
        return [
            {"id": r[0], "to": r[1], "from": r[2], "message": r[3], "checkpoint": r[4],
             "assets": self._load_assets(r[5]),
             "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(r[6]))}
            for r in rows
        ]

    def complete(self, order_id: int, result: str | None = None, assets=None) -> bool:
        cur = self._db.execute(
            "UPDATE inbox SET status='done', result=?, result_assets=?, done_at=? "
            "WHERE id=? AND status='pending'",
            (result, self._dump_assets(assets), time.time(), order_id),
        )
        self._db.commit()
        return cur.rowcount > 0

    def history(self, limit: int = 10) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, to_client, from_client, message, result, result_assets, done_at FROM inbox "
            "WHERE status='done' ORDER BY done_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "to": r[1], "from": r[2], "message": r[3], "result": r[4],
             "result_assets": self._load_assets(r[5]),
             "done": time.strftime("%Y-%m-%d %H:%M", time.localtime(r[6])) if r[6] else None}
            for r in rows
        ]

    def close(self) -> None:
        self._db.close()
