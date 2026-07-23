#!/usr/bin/env python3
"""
INDIVIDRA MCP — Context Ingestion & Bulk Offload Engine  (v2.0.0)

3 herramientas de alto impacto:
  router_smart_read    → Lectura quirúrgica de archivos grandes (Mini-RAG local, $0)
  router_bulk_process  → Offload masivo de tareas repetitivas a modelos gratuitos
  router_status        → Estado, métricas honestas y diagnóstico de proveedores

Filosofía v2:
  - El ahorro real está en NO meter archivos de 20k tokens al contexto de Claude,
    no en resumirlos con pérdida usando modelos débiles.
  - smart_read es determinista y local: devuelve los chunks EXACTOS del archivo.
  - bulk_process delega trabajo repetitivo (clasificar/extraer sobre N items)
    donde los modelos gratis sí rinden, con failover transparente.
  - Todas las salidas van minificadas: ni un token regalado.

─────────────────────────────────────────────
claude_desktop_config.json (o `claude mcp add --scope user`):
─────────────────────────────────────────────
{
  "mcpServers": {
    "individra-router": {
      "command": "python",
      "args": ["C:/ruta/a/individra-mcp-router/server.py"],
      "env": {"GROQ_API_KEY": "...", "OPENROUTER_API_KEY": "..."}
    }
  }
}
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import httpx
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from mcp.server.fastmcp import FastMCP

from router.cache import RouterCache
from router.circuit_breaker import CircuitBreaker
from router.inbox import Inbox
from router.ledger import FileLedger
from router.providers import CheapLLM, GROQ_API_URL, GROQ_MODEL, OPENROUTER_API_URL, get_free_models
from router.ranker import build_outline, chunk_text, rank_chunks
from router.sanitizer import sanitize_file_content

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────

_server_dir = Path(__file__).parent
load_dotenv(_server_dir / ".env", override=True)

_config_path = _server_dir / "router_config.yaml"
CONFIG: dict = {}
if _config_path.exists():
    with open(_config_path) as f:
        CONFIG = yaml.safe_load(f) or {}

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("individra-mcp")

cache = RouterCache(
    db_path=str(_server_dir / "cache" / "router_cache.db"),
    ttl_static=CONFIG.get("cache", {}).get("ttl_static_seconds", 86400),
    ttl_dynamic=CONFIG.get("cache", {}).get("ttl_dynamic_seconds", 3600),
)
circuit_breaker = CircuitBreaker(
    failure_threshold=CONFIG.get("circuit_breaker", {}).get("failure_threshold", 4),
    reset_timeout_seconds=CONFIG.get("circuit_breaker", {}).get("reset_timeout_seconds", 180),
)
llm = CheapLLM(circuit_breaker=circuit_breaker)
ledger = FileLedger(db_path=str(_server_dir / "cache" / "ledger.db"))
inbox = Inbox(db_path=str(_server_dir / "cache" / "inbox.db"))
_checkpoints_dir = _server_dir / "checkpoints"

_stats = {
    "start_time": time.time(),
    "smart_reads": 0,
    "tokens_file_total": 0,      # tokens de los archivos originales pedidos
    "tokens_delivered": 0,       # tokens que efectivamente entraron al contexto de Claude
    "bulk_items_processed": 0,
    "bulk_items_failed": 0,
    "cache_hits": 0,
    "api_calls": 0,
    "unchanged_hits": 0,
    "diff_reads": 0,
}

VERSION = "2.3.0"

# Umbral: archivos por debajo se devuelven enteros (el overhead de RAG no rinde)
FULL_RETURN_MAX_TOKENS = 1500
BULK_ITEM_MAX_CHARS = 12000
BULK_MAX_CONCURRENCY = 4


def _tokens(text: str) -> int:
    return len(text) // 4


def _j(obj) -> str:
    """JSON minificado — política global: ni un token regalado."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _ledger_safe_record(key: str, content: str, tokens: int, outline: list[str] | None = None) -> None:
    """Registra en el ledger sin romper el flujo si SQLite falla."""
    try:
        ledger.record(key, content, outline if outline is not None else build_outline(content), tokens)
    except Exception as e:
        logger.warning(f"ledger.record falló: {e}")


# ─────────────────────────────────────────────
# FastMCP + instructions (auto-activación)
# ─────────────────────────────────────────────

INSTRUCTIONS = """Suite de ingesta de contexto y procesamiento masivo con memoria cross-sesión. Usala PROACTIVAMENTE para proteger tu ventana de contexto:
1. router_smart_read: para leer un archivo grande (>15KB) o buscar algo puntual en cualquier archivo, pasá `query` con lo que buscás — devuelve solo los fragmentos exactos relevantes con números de línea (ranking local, sin pérdida). Sin `query` devuelve el mapa estructural. MEMORIA: si el archivo ya fue leído en una sesión anterior y no cambió, devuelve solo el outline (~50 tokens); si cambió, devuelve SOLO el diff. Usá `force_full=true` si necesitás el contenido completo igual.
2. router_bulk_process: para tareas repetitivas sobre muchos archivos o textos (clasificar, extraer campos, resumir N items) — procesa en paralelo con modelos gratuitos externos y devuelve un JSON consolidado. No usar para código crítico ni razonamiento complejo.
3. router_checkpoint: al cerrar una tarea larga o cuando el contexto se está llenando, guardá un checkpoint (action=save) con resumen, decisiones y pendientes. Al arrancar una sesión sobre trabajo previo, action=resume lo restaura en ~300 tokens e indica qué archivos cambiaron desde entonces.
4. router_inbox: buzón de órdenes entre clientes (Cowork/Code/Desktop/Design). Si el usuario dice "dejale esta tarea a Claude Code", "pasale el diseño a Claude Design", "que Design haga el mockup" o similar: action=send con la orden, un checkpoint vinculado y `assets` (rutas/URLs de brief, wireframe, export .fig/.png) para el handoff código↔diseño. AL INICIO de sesiones de trabajo, chequeá órdenes pendientes con action=check; al ejecutarlas marcá complete con el resultado (y `assets` devueltos si generaste algo, ej. el export de un mockup).
5. router_status: métricas de ahorro y diagnóstico de proveedores.
Todas las salidas vienen en JSON minificado."""

mcp = FastMCP("individra_router_mcp", instructions=INSTRUCTIONS)


# ─────────────────────────────────────────────
# Modelos de entrada
# ─────────────────────────────────────────────

class SmartReadInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    file_path: str = Field(..., description="Ruta absoluta al archivo", min_length=1)
    query: Optional[str] = Field(
        default=None,
        description="Qué buscás en el archivo (ej: '¿dónde se configuran los webhooks?'). Si se omite y el archivo es grande, se devuelve el mapa estructural.",
    )
    top_k: int = Field(default=4, ge=1, le=10, description="Cantidad de fragmentos a devolver")
    force_full: bool = Field(
        default=False,
        description="True = ignorar la memoria de lecturas previas y devolver contenido completo/mapa",
    )


class BulkProcessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de rutas de archivo o textos crudos a procesar",
    )
    instruction: str = Field(
        ...,
        min_length=10,
        description="Qué hacer con cada item (ej: 'extraé remitente, fecha y monto de esta factura')",
    )
    output_schema: Optional[str] = Field(
        default=None,
        description="Esquema JSON deseado por item (ej: '{\"sender\":str,\"amount\":float}')",
    )
    mode: Literal["map", "map_reduce"] = Field(
        default="map",
        description="map=resultado por item | map_reduce=además consolida todo en un resumen final",
    )


class CheckpointInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    action: Literal["save", "resume", "list"] = Field(
        ...,
        description="save=guardar estado de la sesión | resume=restaurar el último (o `name`) | list=ver checkpoints disponibles",
    )
    name: Optional[str] = Field(
        default=None,
        description="Nombre del checkpoint (default: 'latest'). Usá nombres por proyecto/tarea, ej: 'refactor-auth'",
    )
    summary: Optional[str] = Field(default=None, description="[save] Resumen del estado de la tarea (2-5 oraciones)")
    decisions: Optional[list[str]] = Field(default=None, description="[save] Decisiones tomadas")
    open_items: Optional[list[str]] = Field(default=None, description="[save] Pendientes / próximos pasos")
    files: Optional[list[str]] = Field(
        default=None,
        description="[save] Rutas de archivos relevantes a la tarea — al hacer resume se reporta cuáles cambiaron",
    )


class InboxInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    action: Literal["send", "check", "complete", "history"] = Field(
        ...,
        description="send=dejar una orden para otro cliente | check=ver órdenes pendientes | complete=marcar hecha con resultado | history=últimas completadas",
    )
    message: Optional[str] = Field(default=None, description="[send] La orden/instrucción a dejar")
    to: Optional[str] = Field(default=None, description="[send/check] Cliente destino: 'code', 'cowork', 'desktop', 'design' o 'any' (default)")
    from_client: Optional[str] = Field(default=None, description="[send] Quién deja la orden: 'cowork', 'code', 'desktop', 'design'")
    checkpoint: Optional[str] = Field(
        default=None,
        description="[send] Nombre de checkpoint vinculado — el receptor lo puede resumir para ver el contexto completo de la tarea",
    )
    assets: Optional[list[str]] = Field(
        default=None,
        description="[send/complete] Assets de handoff: rutas de archivos o URLs que acompañan la orden o el resultado "
                    "(brief, wireframe, export .fig/.png, specs de diseño). Clave para el ida y vuelta código ↔ diseño.",
    )
    order_id: Optional[int] = Field(default=None, description="[complete] ID de la orden a marcar como hecha")
    result: Optional[str] = Field(default=None, description="[complete] Resultado/resumen de lo ejecutado, visible para quien dejó la orden")


class StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deep: bool = Field(default=False, description="True = testear conectividad real con los proveedores")


# ─────────────────────────────────────────────
# Tool 1: smart_read — Ingesta quirúrgica (local, $0, sin pérdida)
# ─────────────────────────────────────────────

@mcp.tool(
    name="router_smart_read",
    annotations={
        "title": "Lectura Quirúrgica de Archivos (Mini-RAG local)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def router_smart_read(params: SmartReadInput) -> str:
    """
    Lee un archivo protegiendo la ventana de contexto de Claude. 100% local y determinista.

    - Archivo chico (≤~6KB): lo devuelve entero, limpio y minificado.
    - Archivo grande + query: ranking local (BM25/embeddings) → devuelve SOLO los
      fragmentos exactos relevantes, con números de línea. Cero pérdida de fidelidad.
    - Archivo grande sin query: devuelve el mapa estructural (outline con líneas).
    - MEMORIA cross-sesión: si el archivo ya fue leído antes y no cambió,
      devuelve "unchanged" + outline (~50 tokens). Si cambió, devuelve SOLO el
      diff unificado. force_full=true para saltear la memoria.
    - HTML: se limpia localmente (scripts/tags fuera) antes de procesar.
    """
    fp = Path(params.file_path)
    if not fp.exists() or not fp.is_file():
        return _j({"status": "error", "reason": f"no existe: {params.file_path}"})
    try:
        raw = fp.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return _j({"status": "error", "reason": str(e)[:200]})

    content, was_html = sanitize_file_content(raw, str(fp))
    tok_raw, tok_clean = _tokens(raw), _tokens(content)
    _stats["smart_reads"] += 1
    _stats["tokens_file_total"] += tok_raw

    base = {"file": str(fp), "tokens_file": tok_raw}
    if was_html:
        base["html_stripped"] = True
        base["tokens_after_clean"] = tok_clean

    # ─── Memoria cross-sesión (ledger) — solo cuando no hay query ni force_full ───
    key = str(fp.resolve())
    entry = None
    try:
        entry = ledger.get(key)
    except Exception as e:
        logger.warning(f"ledger.get falló: {e}")

    if entry and not params.query and not params.force_full:
        new_hash = FileLedger.hash(content)
        if entry["hash"] == new_hash:
            ledger.touch(key)
            _stats["unchanged_hits"] += 1
            outline = entry["outline"] or build_outline(content)
            delivered = _tokens("\n".join(outline))
            _stats["tokens_delivered"] += delivered
            return _j({
                "status": "unchanged",
                **base,
                "last_seen": time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["last_seen"])),
                "reads": entry["reads"] + 1,
                "outline": outline,
                "hint": "sin cambios desde la última lectura — usá `query` para fragmentos o `force_full=true` para el contenido completo",
            })
        if entry["snapshot"]:
            diff = ledger.diff(entry["snapshot"], content)
            outline = build_outline(content)
            ledger.record(key, content, outline, tok_clean)
            if diff is not None:
                _stats["diff_reads"] += 1
                delivered = _tokens(diff)
                _stats["tokens_delivered"] += delivered
                return _j({
                    "status": "diff",
                    **base,
                    "since": time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["last_seen"])),
                    "tokens_delivered": delivered,
                    "saved_vs_full_pct": round((1 - delivered / max(1, tok_clean)) * 100, 1),
                    "diff": diff if diff else "(sin diferencias tras sanitizado)",
                    "hint": "solo se muestran los cambios — `force_full=true` para el archivo completo",
                })
            # diff demasiado grande → seguir al flujo normal (ya re-registrado)
            entry = None

    # Chico → entero
    if tok_clean <= FULL_RETURN_MAX_TOKENS:
        _stats["tokens_delivered"] += tok_clean
        _ledger_safe_record(key, content, tok_clean)
        return _j({"status": "full", **base, "content": content})

    # Grande + query → chunks exactos rankeados localmente
    if params.query:
        _ledger_safe_record(key, content, tok_clean)
        top, engine = rank_chunks(content, params.query, top_k=params.top_k)
        delivered = sum(_tokens(c.text) for c in top)
        _stats["tokens_delivered"] += delivered
        return _j({
            "status": "chunks",
            **base,
            "query": params.query,
            "engine": engine,
            "tokens_delivered": delivered,
            "saved_vs_full_pct": round((1 - delivered / max(1, tok_clean)) * 100, 1),
            "chunks": [
                {"lines": f"{c.start_line}-{c.end_line}", "score": c.score, "text": c.text}
                for c in top
            ],
            "note": "fragmentos EXACTOS del archivo — para más contexto repetir con otra query o top_k mayor",
        })

    # Grande sin query → mapa estructural
    outline = build_outline(content)
    _ledger_safe_record(key, content, tok_clean, outline=outline)
    head = "\n".join(content.split("\n")[:25])
    _stats["tokens_delivered"] += _tokens(head) + _tokens("\n".join(outline))
    return _j({
        "status": "map",
        **base,
        "total_lines": content.count("\n") + 1,
        "total_chunks": len(chunk_text(content)),
        "head": head,
        "outline": outline,
        "hint": "archivo grande — llamá de nuevo con `query` para obtener los fragmentos relevantes",
    })


# ─────────────────────────────────────────────
# Tool 2: bulk_process — Offload masivo con failover transparente
# ─────────────────────────────────────────────

@mcp.tool(
    name="router_bulk_process",
    annotations={
        "title": "Procesamiento Masivo en Modelos Gratuitos",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def router_bulk_process(params: BulkProcessInput) -> str:
    """
    Procesa N archivos o textos en paralelo contra modelos gratuitos (Groq → OpenRouter)
    y devuelve un único JSON consolidado. Claude no ve los contenidos originales:
    solo el resultado final estructurado.

    Ideal para: clasificar lotes, extraer campos de facturas/emails/transcripciones,
    resumir muchos documentos, mapear datos no estructurados a un esquema.
    NO usar para: código crítico, decisiones arquitecturales, razonamiento complejo.

    Failover transparente: si un proveedor cae, rota al siguiente; si todos fallan
    para un item, el item se reporta como fallido sin romper el lote.
    Caché SHA-256: mismo item + misma instrucción = resultado instantáneo sin API.
    """
    await cache.init()

    schema_part = f"\nRespondé EXACTAMENTE con este esquema JSON:\n{params.output_schema}" if params.output_schema else ""
    sem = asyncio.Semaphore(BULK_MAX_CONCURRENCY)

    def load_item(item: str) -> tuple[str, str]:
        """Returns (nombre, contenido). Si parece ruta y existe, lee el archivo."""
        if len(item) < 500:
            p = Path(item)
            try:
                if p.is_file():
                    raw = p.read_text(encoding="utf-8", errors="replace")
                    clean, _ = sanitize_file_content(raw, str(p))
                    return p.name, clean
            except Exception:
                pass
        return f"text_{abs(hash(item)) % 10000}", item

    async def process(idx: int, item: str) -> dict:
        name, content = load_item(item)
        if len(content) > BULK_ITEM_MAX_CHARS:
            content = content[:BULK_ITEM_MAX_CHARS] + "\n[...truncado]"

        prompt = (
            f"{params.instruction}{schema_part}\n"
            "Respondé SOLO con el resultado, sin introducción ni explicación.\n"
            f"---\n{content}"
        )
        cache_key = RouterCache.hash(prompt)
        cached = await cache.get(cache_key)
        if cached:
            _stats["cache_hits"] += 1
            return {"i": idx, "src": name, "out": cached, "provider": "cache"}

        async with sem:
            result, provider = await llm.call(
                prompt, max_tokens=1500, json_mode=bool(params.output_schema)
            )
        if result is None:
            _stats["bulk_items_failed"] += 1
            return {"i": idx, "src": name, "error": "todos los proveedores fallaron"}

        _stats["api_calls"] += 1
        _stats["bulk_items_processed"] += 1
        _stats["tokens_file_total"] += _tokens(content)
        await cache.set(cache_key, result, provider=provider)
        return {"i": idx, "src": name, "out": result, "provider": provider}

    results = await asyncio.gather(*[process(i, item) for i, item in enumerate(params.items)])
    ok = [r for r in results if "out" in r]
    failed = [r for r in results if "error" in r]

    response = {
        "status": "ok" if ok else "failed",
        "processed": len(ok),
        "failed": len(failed),
        "results": results,
    }

    if params.mode == "map_reduce" and len(ok) > 1:
        joined = "\n".join(f"[{r['src']}]: {r['out']}" for r in ok)[:BULK_ITEM_MAX_CHARS]
        reduce_prompt = (
            f"Los siguientes son resultados por-item de la tarea: '{params.instruction}'.\n"
            "Consolidalos en un único resumen/estructura global. Respondé SOLO con el resultado.\n"
            f"---\n{joined}"
        )
        reduced, provider = await llm.call(reduce_prompt, max_tokens=2000)
        if reduced:
            _stats["api_calls"] += 1
            response["reduced"] = reduced
            response["reduce_provider"] = provider
        else:
            response["reduced"] = None
            response["reduce_note"] = "consolidación falló — usar resultados por item"

    delivered = _tokens(_j(response))
    _stats["tokens_delivered"] += delivered
    return _j(response)


# ─────────────────────────────────────────────
# Tool 3: checkpoint — handoff de contexto entre sesiones y clientes
# ─────────────────────────────────────────────

def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60] or "latest"


@mcp.tool(
    name="router_checkpoint",
    annotations={
        "title": "Checkpoint/Resume de Sesión (cross-cliente)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def router_checkpoint(params: CheckpointInput) -> str:
    """
    Traspaso de contexto entre sesiones y clientes (Desktop, Code, Cowork comparten esto).

    save: guardá resumen, decisiones, pendientes y archivos relevantes al cerrar
          una tarea larga o cuando el contexto se llena. Persiste en disco como
          JSON legible (checkpoints/{name}.json) — el usuario puede editarlo.
    resume: una sesión nueva restaura todo en ~300 tokens, e indica qué archivos
            cambiaron en disco desde el checkpoint (via hash) — sin re-leerlos.
    list: checkpoints disponibles con fecha y resumen.
    """
    _checkpoints_dir.mkdir(exist_ok=True)

    if params.action == "save":
        if not params.summary:
            return _j({"status": "error", "reason": "save requiere `summary`"})
        name = _safe_name(params.name or "latest")
        files_entry = []
        for f in params.files or []:
            p = Path(f)
            h = None
            if p.is_file():
                try:
                    h = FileLedger.hash(p.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass
            files_entry.append({"path": f, "hash": h})
        data = {
            "name": name,
            "saved_at": time.strftime("%Y-%m-%d %H:%M"),
            "ts": time.time(),
            "summary": params.summary,
            "decisions": params.decisions or [],
            "open_items": params.open_items or [],
            "files": files_entry,
        }
        path = _checkpoints_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        return _j({"status": "saved", "name": name, "path": str(path), "files_tracked": len(files_entry)})

    if params.action == "list":
        items = []
        for p in sorted(_checkpoints_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                items.append({"name": d.get("name", p.stem), "saved_at": d.get("saved_at"), "summary": (d.get("summary") or "")[:120]})
            except Exception:
                continue
        return _j({"status": "ok", "checkpoints": items})

    # resume
    if params.name:
        path = _checkpoints_dir / f"{_safe_name(params.name)}.json"
        if not path.exists():
            return _j({"status": "error", "reason": f"checkpoint '{params.name}' no existe — usar action=list"})
    else:
        candidates = sorted(_checkpoints_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not candidates:
            return _j({"status": "empty", "reason": "no hay checkpoints guardados"})
        path = candidates[0]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return _j({"status": "error", "reason": f"checkpoint corrupto: {e}"})

    file_states = ledger.check_files(data.get("files", []))
    changed = [f["path"] for f in file_states if f["state"] == "changed"]
    return _j({
        "status": "resumed",
        "name": data.get("name"),
        "saved_at": data.get("saved_at"),
        "summary": data.get("summary"),
        "decisions": data.get("decisions"),
        "open_items": data.get("open_items"),
        "files": file_states,
        "hint": (
            f"archivos modificados desde el checkpoint: {changed} — leelos con router_smart_read para ver solo los diffs"
            if changed else "ningún archivo relevante cambió desde el checkpoint"
        ),
    })


# ─────────────────────────────────────────────
# Tool 4: inbox — órdenes cruzadas entre clientes
# ─────────────────────────────────────────────

@mcp.tool(
    name="router_inbox",
    annotations={
        "title": "Inbox de Órdenes entre Clientes (Cowork ↔ Code ↔ Desktop ↔ Design)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def router_inbox(params: InboxInput) -> str:
    """
    Buzón asíncrono entre clientes de Claude (Cowork, Code, Desktop y Claude
    Design). Los chats no pueden comandarse en tiempo real, pero comparten este
    disco: acá un cliente deja órdenes y otro las consume, ejecuta y reporta.

    send: dejá una orden (ej. desde Cowork para Claude Code, o entre Code y
          Design), opcionalmente vinculada a un checkpoint y con `assets`
          (rutas/URLs de brief, wireframe, export .fig/.png) para el handoff.
    check: al arrancar una sesión, mirá si hay órdenes pendientes para vos.
    complete: marcá la orden hecha con un resumen del resultado y, si aplica,
          `assets` devueltos (ej. Design entrega el export del mockup).
    history: qué se completó, con qué resultado y qué assets volvieron.
    """
    try:
        if params.action == "send":
            if not params.message:
                return _j({"status": "error", "reason": "send requiere `message`"})
            if params.checkpoint:
                cp_path = _checkpoints_dir / f"{_safe_name(params.checkpoint)}.json"
                if not cp_path.exists():
                    return _j({"status": "error", "reason": f"checkpoint '{params.checkpoint}' no existe — guardalo primero con router_checkpoint"})
            oid = inbox.send(
                message=params.message,
                to_client=params.to or "any",
                from_client=params.from_client or "unknown",
                checkpoint=params.checkpoint,
                assets=params.assets,
            )
            return _j({
                "status": "sent", "id": oid, "to": params.to or "any",
                "checkpoint": params.checkpoint,
                "assets": params.assets or [],
                "note": "el destinatario la verá con router_inbox action=check",
            })

        if params.action == "check":
            orders = inbox.check(to_client=params.to)
            # adjuntar resumen del checkpoint vinculado para dar contexto sin otra llamada
            for o in orders:
                if o.get("checkpoint"):
                    cp_path = _checkpoints_dir / f"{_safe_name(o['checkpoint'])}.json"
                    if cp_path.exists():
                        try:
                            cp = json.loads(cp_path.read_text(encoding="utf-8"))
                            o["checkpoint_summary"] = (cp.get("summary") or "")[:200]
                        except Exception:
                            pass
            hint = (
                "ejecutá las órdenes; usá router_checkpoint action=resume con el checkpoint vinculado para el contexto completo; al terminar marcá complete con el resultado"
                if orders else "sin órdenes pendientes"
            )
            return _j({"status": "ok", "pending": len(orders), "orders": orders, "hint": hint})

        if params.action == "complete":
            if not params.order_id:
                return _j({"status": "error", "reason": "complete requiere `order_id`"})
            ok = inbox.complete(params.order_id, result=params.result, assets=params.assets)
            if not ok:
                return _j({"status": "error", "reason": f"orden {params.order_id} no existe o ya estaba completada"})
            return _j({"status": "completed", "id": params.order_id, "result_assets": params.assets or []})

        # history
        return _j({"status": "ok", "completed": inbox.history()})

    except Exception as e:
        logger.warning(f"inbox error: {e}")
        return _j({"status": "error", "reason": str(e)[:200]})


# ─────────────────────────────────────────────
# Tool 5: status — métricas honestas + diagnóstico
# ─────────────────────────────────────────────

@mcp.tool(
    name="router_status",
    annotations={
        "title": "Estado y Diagnóstico del MCP",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def router_status(params: StatusInput) -> str:
    """
    Métricas de la sesión (tokens que NO entraron al contexto de Claude, cache, llamadas)
    y estado de circuit breakers. Con deep=true testea conectividad real de proveedores.
    """
    await cache.init()
    saved = max(0, _stats["tokens_file_total"] - _stats["tokens_delivered"])
    status = {
        "version": VERSION,
        "uptime_s": round(time.time() - _stats["start_time"]),
        "providers": {
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        },
        "circuit_breakers": circuit_breaker.get_status() or None,
        "session": {
            "smart_reads": _stats["smart_reads"],
            "unchanged_hits": _stats["unchanged_hits"],
            "diff_reads": _stats["diff_reads"],
            "bulk_processed": _stats["bulk_items_processed"],
            "bulk_failed": _stats["bulk_items_failed"],
            "api_calls": _stats["api_calls"],
            "cache_hits": _stats["cache_hits"],
            "tokens_source_total": _stats["tokens_file_total"],
            "tokens_delivered_to_claude": _stats["tokens_delivered"],
            "tokens_kept_out_of_context": saved,
        },
        "ledger": ledger.stats(),
        "cache": await cache.get_stats(),
    }

    if params.deep:
        diag = {}
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        GROQ_API_URL,
                        headers={"Authorization": f"Bearer {groq_key}"},
                        json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 5},
                        timeout=15.0,
                    )
                diag["groq"] = {"http": r.status_code, "model": GROQ_MODEL}
            except Exception as e:
                diag["groq"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        else:
            diag["groq"] = {"error": "GROQ_API_KEY no configurada"}

        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_key:
            discovered = await get_free_models()
            diag["openrouter"] = {"discovered": len(discovered), "models": []}
            for model in discovered[:3]:
                try:
                    async with httpx.AsyncClient() as client:
                        r = await client.post(
                            OPENROUTER_API_URL,
                            headers={"Authorization": f"Bearer {or_key}"},
                            json={"model": model, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 5},
                            timeout=20.0,
                        )
                    diag["openrouter"]["models"].append({"m": model, "http": r.status_code})
                except Exception as e:
                    diag["openrouter"]["models"].append({"m": model, "error": str(e)[:100]})
        else:
            diag["openrouter"] = {"error": "OPENROUTER_API_KEY no configurada"}

        try:
            from router.ranker import _get_embedder
            diag["fastembed"] = "activo" if _get_embedder() else "no instalado (BM25 puro)"
        except Exception:
            diag["fastembed"] = "no instalado (BM25 puro)"

        diag["python"] = sys.version.split()[0]
        status["diagnostics"] = diag

    return _j(status)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"INDIVIDRA MCP v{VERSION} — Context Ingestion & Bulk Offload Engine ✓")
    mcp.run()
