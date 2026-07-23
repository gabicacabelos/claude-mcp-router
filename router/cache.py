"""
Caché local con SQLite + SHA-256.

Evita llamar a Gemini cuando el contenido ya fue comprimido antes.
La clave es SHA-256(contenido) — solo se re-comprime si el archivo cambió.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./cache/router_cache.db"
DEFAULT_TTL_STATIC = 86400
DEFAULT_TTL_DYNAMIC = 3600


class RouterCache:
    """
    Caché async basado en SQLite + SHA-256.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        ttl_static: int = DEFAULT_TTL_STATIC,
        ttl_dynamic: int = DEFAULT_TTL_DYNAMIC,
    ):
        self.db_path = db_path
        self.ttl_static = ttl_static
        self.ttl_dynamic = ttl_dynamic
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Inicializa la DB y crea tablas si no existen. Idempotente: no re-conecta."""
        if self._db is not None:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl INTEGER NOT NULL,
                hits INTEGER DEFAULT 0,
                provider TEXT DEFAULT 'gemini'
            )
        """)
        await self._db.commit()
        logger.info(f"Cache inicializada en {self.db_path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @staticmethod
    def hash(content: str) -> str:
        """SHA-256 del contenido. Determinístico."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> Optional[str]:
        """Retorna valor cacheado si existe y no expiró."""
        if not self._db:
            return None

        async with self._db.execute(
            "SELECT value, created_at, ttl FROM cache WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        value, created_at, ttl = row
        age = time.time() - created_at

        if age > ttl:
            await self._db.execute("DELETE FROM cache WHERE key = ?", (key,))
            await self._db.commit()
            return None

        await self._db.execute(
            "UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,)
        )
        await self._db.commit()

        logger.debug(f"Cache HIT: {key[:8]}... (age: {age:.0f}s)")
        return value

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
        provider: str = "gemini",
    ) -> None:
        if not self._db:
            return

        effective_ttl = ttl or self.ttl_static

        await self._db.execute(
            """
            INSERT OR REPLACE INTO cache (key, value, created_at, ttl, provider)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, value, time.time(), effective_ttl, provider),
        )
        await self._db.commit()
        logger.debug(f"Cache SET: {key[:8]}... (ttl: {effective_ttl}s)")

    async def get_stats(self) -> dict:
        if not self._db:
            return {}

        async with self._db.execute(
            "SELECT COUNT(*), SUM(hits), SUM(CASE WHEN hits > 0 THEN 1 ELSE 0 END) FROM cache"
        ) as cursor:
            row = await cursor.fetchone()

        total_entries, total_hits, entries_with_hits = row or (0, 0, 0)

        return {
            "total_entries": total_entries or 0,
            "total_hits": total_hits or 0,
            "entries_with_hits": entries_with_hits or 0,
            "db_path": self.db_path,
            "db_size_kb": round(
                os.path.getsize(self.db_path) / 1024, 1
            ) if os.path.exists(self.db_path) else 0,
        }

    async def purge_expired(self) -> int:
        """Elimina entradas expiradas."""
        if not self._db:
            return 0
        now = time.time()
        async with self._db.execute(
            "DELETE FROM cache WHERE (created_at + ttl) < ?", (now,)
        ) as cursor:
            deleted = cursor.rowcount
        await self._db.commit()
        if deleted:
            logger.info(f"Cache: {deleted} entradas expiradas eliminadas")
        return deleted
