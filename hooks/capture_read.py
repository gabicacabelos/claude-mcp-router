#!/usr/bin/env python3
"""
Captura pasiva de lecturas nativas — hook PostToolUse de Claude Code.

El problema que resuelve: si Claude usa su `Read` nativo en vez de
router_smart_read, el ledger no se entera y la memoria cross-sesión queda con
agujeros. Este hook cierra el circuito: el valor se acumula AUNQUE Claude nunca
elija las tools del MCP.

REGLA DE ORO — este script corre en el camino crítico de la terminal del usuario:
  1. Un solo INSERT en una tabla de paso tonta (sin hash, sin leer el archivo).
  2. Timeout duro: si la DB está ocupada, se DESCARTA en silencio.
     Perder una captura es gratis; congelar la terminal del usuario, no.
  3. Nunca escribe a stdout/stderr ni devuelve exit != 0 (rompería el flujo).

Instalación (opt-in) en .claude/settings.json del usuario:

  {
    "hooks": {
      "PostToolUse": [
        {
          "matcher": "Read",
          "hooks": [
            {
              "type": "command",
              "command": "python /ruta/a/claude-continuity-mcp/hooks/capture_read.py"
            }
          ]
        }
      ]
    }
  }

El hook recibe por stdin el JSON del evento; de ahí sale el path leído.
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Presupuesto total del hook. Por encima de esto preferimos perder la captura.
BUSY_TIMEOUT_MS = 100

DB_PATH = Path(__file__).resolve().parent.parent / "cache" / "ledger.db"


def _extract_path(event: dict) -> str | None:
    """El path leído, desde el shape del evento PostToolUse."""
    ti = event.get("tool_input") or {}
    for key in ("file_path", "path", "notebook_path"):
        v = ti.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw:
            return
        event = json.loads(raw)
    except Exception:
        return  # entrada rara: no es asunto del hook romper nada

    path = _extract_path(event)
    if not path:
        return

    # Solo registramos archivos que existen; el drain del MCP hará el resto.
    try:
        if not Path(path).is_file():
            return
    except Exception:
        return

    if not DB_PATH.exists():
        return  # el MCP nunca corrió acá: nada que alimentar

    con = None
    try:
        con = sqlite3.connect(str(DB_PATH), timeout=BUSY_TIMEOUT_MS / 1000)
        con.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        con.execute(
            "INSERT INTO raw_reads (path, client, seen_at) VALUES (?,?,?)",
            (path, os.environ.get("CLAUDE_CLIENT", "code"), time.time()),
        )
        con.commit()
    except Exception:
        # DB ocupada/bloqueada/tabla inexistente → se descarta en silencio.
        pass
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
