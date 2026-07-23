#!/usr/bin/env python3
"""
INDIVIDRA MCP — Memoria y continuidad para Claude entre sesiones y clientes  (v3.0.0)

4 herramientas de alto impacto — 100% locales, sin API keys:
  router_smart_read  → Lectura quirúrgica de archivos grandes con memoria cross-sesión
  router_checkpoint  → Handoff de contexto entre sesiones y clientes (~300 tokens)
  router_inbox       → Órdenes asíncronas entre clientes (Cowork/Code/Desktop/Design)
  router_status      → Estado y métricas honestas de la sesión

Filosofía:
  - Lo que ningún cliente de Claude hace: recordar qué archivos ya leíste y devolver
    solo los diffs. El valor es la continuidad, no un "ahorro mágico de tokens".
  - smart_read es determinista y local: devuelve los chunks EXACTOS del archivo,
    nunca resúmenes con pérdida generados por un modelo débil.
  - checkpoint + inbox comparten estado en disco entre todos tus clientes.
  - Todas las salidas van minificadas: ni un token regalado.

Nota: el procesamiento masivo en modelos gratuitos (antes router_bulk_process) se
separó a su propio repo, `individra-bulk-offload`, porque es otro producto y
dependía de servicios externos que diluían este núcleo 100% local.

─────────────────────────────────────────────
claude_desktop_config.json (o `claude mcp add --scope user`):
─────────────────────────────────────────────
{
  "mcpServers": {
    "claude-continuity": {
      "command": "python",
      "args": ["C:/ruta/a/claude-continuity-mcp/server.py"]
    }
  }
}
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from mcp.server.fastmcp import FastMCP

from router.inbox import Inbox
from router.ledger import FileLedger
from router.ranker import build_outline, chunk_text, rank_chunks
from router.sanitizer import sanitize_file_content
from router import rules as project_rules
from router import project_index

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────

_server_dir = Path(__file__).parent

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("individra-mcp")

ledger = FileLedger(db_path=str(_server_dir / "cache" / "ledger.db"))
inbox = Inbox(db_path=str(_server_dir / "cache" / "inbox.db"))

def _code_staleness(boot_time: float) -> dict | None:
    """
    Un proceso MCP no recarga módulos solo: si server.py o router/*.py cambian
    en disco después de que este proceso arrancó (por un git pull/checkout, o
    porque otro cliente editó el código), este proceso sigue corriendo la
    versión vieja hasta que lo reinicien. git push no reinicia nada — esto
    hace observable esa desincronización sin tener que inferirla a mano
    (revisar qué tools desaparecieron, probar comportamiento nuevo a ciegas, etc).
    """
    try:
        watched = [_server_dir / "server.py", *sorted((_server_dir / "router").glob("*.py"))]
        newest_path, newest_mtime = None, 0.0
        for p in watched:
            if p.exists():
                m = p.stat().st_mtime
                if m > newest_mtime:
                    newest_path, newest_mtime = p, m
        if newest_mtime > boot_time:
            return {
                "stale": True,
                "changed_file": newest_path.name if newest_path else None,
                "changed_ago_s": round(time.time() - newest_mtime),
                "hint": "el código en disco cambió después de que este proceso arrancó — reiniciá el MCP (reconectar en /mcp o reiniciar el cliente) para aplicar los cambios",
            }
        return {"stale": False}
    except Exception as e:
        logger.warning(f"chequeo de staleness falló: {e}")
        return None
_checkpoints_dir = _server_dir / "checkpoints"

_stats = {
    "start_time": time.time(),
    "smart_reads": 0,
    "tokens_file_total": 0,      # tokens de los archivos originales pedidos
    "tokens_delivered": 0,       # tokens que efectivamente entraron al contexto de Claude
    "unchanged_hits": 0,
    "diff_reads": 0,
}

VERSION = "3.3.0"

# Umbral: archivos por debajo se devuelven enteros (el overhead de RAG no rinde)
FULL_RETURN_MAX_TOKENS = 1500


# ─── Config en caliente (router_config.json, opcional) ───────────────────────
# Si el archivo NO existe aplican estos defaults (zero-config). Si existe, se
# re-parsea solo cuando cambia el mtime → ajustar un tunable NO pide reiniciar.

_CONFIG_PATH = _server_dir / "router_config.json"
_CONFIG_DEFAULTS = {
    "full_return_max_tokens": FULL_RETURN_MAX_TOKENS,
    "default_top_k": 4,
    "diff_max_ratio": 0.6,
    "cache_enabled": True,
}
_config_cache: dict = {"mtime": None, "cfg": dict(_CONFIG_DEFAULTS)}


def _load_config() -> dict:
    """Config live-editable: stat + re-parse solo si cambió el mtime."""
    try:
        if not _CONFIG_PATH.exists():
            if _config_cache["mtime"] is not None:
                _config_cache["mtime"] = None
                _config_cache["cfg"] = dict(_CONFIG_DEFAULTS)
            return _config_cache["cfg"]
        m = _CONFIG_PATH.stat().st_mtime
        if m != _config_cache["mtime"]:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            cfg = dict(_CONFIG_DEFAULTS)
            for k in _CONFIG_DEFAULTS:
                if k in data:
                    cfg[k] = data[k]
            _config_cache["mtime"] = m
            _config_cache["cfg"] = cfg
            logger.info(f"router_config.json recargado: {cfg}")
    except Exception as e:
        logger.warning(f"router_config.json inválido: {e} — usando la última config buena")
    return _config_cache["cfg"]


# ─── Staleness throttled para todas las tools ────────────────────────────────
# El chequeo de mtimes corre como máximo una vez cada STALE_CHECK_INTERVAL_S;
# entre medio se sirve el resultado cacheado (overhead casi nulo por llamada).

STALE_CHECK_INTERVAL_S = 5.0
_stale_cache: dict = {"ts": 0.0, "result": None}


def _stale_throttled() -> dict | None:
    now = time.time()
    if now - _stale_cache["ts"] >= STALE_CHECK_INTERVAL_S:
        _stale_cache["result"] = _code_staleness(_stats["start_time"])
        _stale_cache["ts"] = now
    return _stale_cache["result"]


def _tokens(text: str) -> int:
    return len(text) // 4


def _j(obj) -> str:
    """
    JSON minificado — política global: ni un token regalado.
    Si el código en disco cambió después del boot, inyecta `code_stale` en el
    payload de CUALQUIER tool (cero sorpresas: el campo se OMITE por completo
    cuando no hay staleness; router_status ya trae su campo propio y se saltea).
    """
    if isinstance(obj, dict) and "code_staleness" not in obj:
        st = _stale_throttled()
        if st and st.get("stale"):
            obj["code_stale"] = {
                "changed_file": st.get("changed_file"),
                "changed_ago_s": st.get("changed_ago_s"),
                "hint": "código nuevo en disco sin aplicar — reconectá el MCP para aplicarlo",
            }
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

INSTRUCTIONS = """Memoria y continuidad para Claude entre sesiones y clientes — 100% local, sin API keys. Usala PROACTIVAMENTE para proteger tu ventana de contexto y no re-explorar trabajo previo:
1. router_smart_read: para leer un archivo grande (>15KB) o buscar algo puntual en cualquier archivo, pasá `query` con lo que buscás — devuelve solo los fragmentos exactos relevantes con números de línea (ranking local, sin pérdida). Sin `query` devuelve el mapa estructural. MEMORIA: si el archivo ya fue leído en una sesión anterior y no cambió, devuelve solo el outline (~50 tokens); si cambió, devuelve SOLO el diff. Usá `force_full=true` si necesitás el contenido completo igual.
2. router_checkpoint: al cerrar una tarea larga o cuando el contexto se está llenando, guardá un checkpoint (action=save) con resumen, decisiones y pendientes. Al arrancar una sesión sobre trabajo previo, action=resume lo restaura en ~300 tokens e indica qué archivos cambiaron desde entonces.
3. router_inbox: buzón de órdenes entre clientes (Cowork/Code/Desktop/Design). Si el usuario dice "dejale esta tarea a Claude Code", "pasale el diseño a Claude Design", "que Design haga el mockup" o similar: action=send con la orden, un checkpoint vinculado y `assets` (rutas/URLs de brief, wireframe, export .fig/.png) para el handoff código↔diseño. AL INICIO de sesiones de trabajo, chequeá órdenes pendientes con action=check; al ejecutarlas marcá complete con el resultado (y `assets` devueltos si generaste algo, ej. el export de un mockup).
4. router_status: métricas de la sesión (tokens que no entraron al contexto, lecturas, checkpoints, inbox).
5. router_project_search: buscá en TODO el proyecto cuando NO sabés en qué archivo está lo que buscás ("¿dónde se validan los webhooks?"). Índice BM25 local e incremental — devuelve los archivos más relevantes con fragmentos exactos. Preferilo sobre grep cuando la búsqueda es conceptual y no una cadena literal; usá router_smart_read si ya sabés el archivo.
6. router_rules: reglas PERMANENTES del proyecto ("nunca usar Redux", "los tests van en tests/"), distintas del estado de tarea de un checkpoint. Guardá una con action=add cuando el usuario fije una convención durable, o promové una decisión de un checkpoint con action=promote. Viven en .claude-continuity-rules.json en la raíz del proyecto (git-friendly). NO hace falta leerlas: se inyectan solas como `project_rules` en smart_read/resume. sync_to_claudemd=true además las escribe en el CLAUDE.md.
Todas las salidas vienen en JSON minificado."""

mcp = FastMCP("claude_continuity_mcp", instructions=INSTRUCTIONS)


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
    top_k: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="Cantidad de fragmentos a devolver (default: default_top_k de la config, 4 si no hay config)",
    )
    force_full: bool = Field(
        default=False,
        description="True = ignorar la memoria de lecturas previas y devolver contenido completo/mapa",
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


class RulesInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    action: Literal["add", "list", "remove", "promote"] = Field(
        ...,
        description="add=nueva regla permanente | list=reglas del proyecto | remove=eliminar por id | promote=promover una decisión de un checkpoint a regla",
    )
    project_dir: str = Field(
        ..., min_length=1,
        description="Raíz del proyecto: ahí vive .claude-continuity-rules.json (git-friendly) y el CLAUDE.md si se sincroniza",
    )
    text: Optional[str] = Field(default=None, description="[add] El texto literal de la regla (ej: 'nunca usar Redux')")
    from_client: Optional[str] = Field(default=None, description="[add/promote] Quién la decide: 'code', 'cowork', 'desktop', 'design' o el nombre del humano")
    checkpoint: Optional[str] = Field(default=None, description="[add] Checkpoint de procedencia | [promote] checkpoint del que se promueve la decisión")
    decision_index: Optional[int] = Field(default=None, ge=0, description="[promote] Índice (0-based) de la decisión a promover; si el checkpoint tiene una sola, se asume")
    rule_id: Optional[int] = Field(default=None, description="[remove] id de la regla a eliminar")
    sync_to_claudemd: bool = Field(
        default=False,
        description="True = además escribir/actualizar la sección delimitada de reglas en el CLAUDE.md del proyecto (alimenta la memoria nativa de Claude Code)",
    )


class ProjectSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project_dir: str = Field(..., min_length=1, description="Raíz del proyecto a buscar/indexar")
    query: Optional[str] = Field(
        default=None,
        description="Qué buscás en TODO el proyecto (ej: '¿dónde se validan los webhooks?'). Si se omite, solo (re)indexa.",
    )
    top_k: int = Field(default=5, ge=1, le=15, description="Cantidad de archivos a devolver")
    fragments_per_file: int = Field(default=1, ge=1, le=4, description="Fragmentos exactos por archivo")
    reindex: bool = Field(default=False, description="True = forzar el barrido de indexación antes de buscar")


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
    cfg = _load_config()
    # Promueve lo que el hook de captura pasiva haya dejado en la tabla de paso.
    # El costo lo paga el MCP acá, nunca el hook (que corre en la terminal).
    try:
        ledger.drain_raw_reads()
    except Exception as e:
        logger.warning(f"drain de captura pasiva falló: {e}")
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
    # Reglas del proyecto inyectadas (piggyback): Claude las ve al tocar un
    # archivo del proyecto sin una llamada aparte. Se omite si no hay reglas.
    _rules = _rules_for_paths([str(fp)])
    if _rules:
        base["project_rules"] = _rules

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
            diff = ledger.diff(entry["snapshot"], content, max_ratio=cfg["diff_max_ratio"])
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
    if tok_clean <= cfg["full_return_max_tokens"]:
        _stats["tokens_delivered"] += tok_clean
        _ledger_safe_record(key, content, tok_clean)
        return _j({"status": "full", **base, "content": content})

    # Grande + query → chunks exactos rankeados localmente (con cache por hash+query)
    if params.query:
        _ledger_safe_record(key, content, tok_clean)
        top_k = params.top_k or cfg["default_top_k"]
        cache_on = bool(cfg["cache_enabled"]) and not params.force_full
        new_hash = FileLedger.hash(content)
        q_norm = FileLedger.normalize_query(params.query)

        # Cache HIT: query repetida sobre archivo sin cambios → releer esas líneas,
        # sin re-chunkear ni re-rankear. Invalidación automática por hash.
        if cache_on:
            cached = None
            try:
                cached = ledger.get_query_cache(new_hash, q_norm, top_k)
            except Exception as e:
                logger.warning(f"query cache get falló: {e}")
            if cached:
                lines = content.split("\n")
                chunks_out = [
                    {"lines": f"{s}-{e}", "text": "\n".join(lines[s - 1:e])}
                    for s, e in cached
                ]
                delivered = sum(_tokens(c["text"]) for c in chunks_out)
                _stats["tokens_delivered"] += delivered
                return _j({
                    "status": "chunks",
                    **base,
                    "query": params.query,
                    "engine": "cache",
                    "cache_hit": True,
                    "tokens_delivered": delivered,
                    "saved_vs_full_pct": round((1 - delivered / max(1, tok_clean)) * 100, 1),
                    "chunks": chunks_out,
                    "note": "fragmentos EXACTOS del archivo (cache) — para más contexto repetir con otra query o top_k mayor",
                })

        top, engine = rank_chunks(content, params.query, top_k=top_k,
                                  file_hash=new_hash,
                                  vector_store=ledger if cfg["cache_enabled"] else None)
        if cache_on:
            try:
                ledger.put_query_cache(new_hash, q_norm, top_k,
                                       [(c.start_line, c.end_line) for c in top])
            except Exception as e:
                logger.warning(f"query cache put falló: {e}")
        delivered = sum(_tokens(c.text) for c in top)
        _stats["tokens_delivered"] += delivered
        return _j({
            "status": "chunks",
            **base,
            "query": params.query,
            "engine": engine,
            "cache_hit": False,
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
# Tool 2: checkpoint — handoff de contexto entre sesiones y clientes
# ─────────────────────────────────────────────

def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60] or "latest"


def _cold_start_digest() -> dict:
    """
    Arranque en frío: no hay checkpoints, pero el ledger y el inbox SÍ tienen
    rastro de actividad. Agregación pura desde SQLite — determinista, sin LLM.
    `mode` lo declara explícitamente: un rastro de archivos NO es un checkpoint
    intencional y no contiene decisiones de arquitectura.
    """
    try:
        ledger.drain_raw_reads()  # el digest en frío se nutre también de lo capturado pasivamente
    except Exception as e:
        logger.warning(f"digest: drain falló: {e}")
    try:
        recent = ledger.recent_files(10)
    except Exception as e:
        logger.warning(f"digest: recent_files falló: {e}")
        recent = []
    try:
        orders = inbox.history(5)
    except Exception as e:
        logger.warning(f"digest: inbox.history falló: {e}")
        orders = []
    changed = [f["path"] for f in recent if f["state"] == "changed"]
    digest = {
        "status": "resumed",
        "mode": "reconstructed_activity",
        "note": "no hay checkpoints guardados — esto es actividad RECONSTRUIDA del ledger/inbox "
                "(rastro de archivos leídos y órdenes completadas), NO un checkpoint intencional: "
                "no contiene decisiones ni contexto de tarea",
        "recent_files": recent,
        "recent_orders": orders,
        "hint": (
            f"archivos con cambios desde la última lectura: {changed} — leelos con router_smart_read para ver solo los diffs"
            if changed else
            "podés retomar desde los archivos recientes; guardá checkpoints al cerrar tareas para resumes con contexto real"
        ),
    }
    _rules = _rules_for_paths([f["path"] for f in recent])
    if _rules:
        digest["project_rules"] = _rules
    return digest


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
            # Arranque en frío: en vez de "empty", digest determinista de actividad
            return _j(_cold_start_digest())
        path = candidates[0]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return _j({"status": "error", "reason": f"checkpoint corrupto: {e}"})

    file_states = ledger.check_files(data.get("files", []))
    changed = [f["path"] for f in file_states if f["state"] == "changed"]
    out = {
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
    }
    _rules = _rules_for_paths([f["path"] for f in file_states])
    if _rules:
        out["project_rules"] = _rules
    return _j(out)


# ─────────────────────────────────────────────
# Tool 3: inbox — órdenes cruzadas entre clientes
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
          Antes de enviar, SIEMPRE chequeá con action=check/history si ya hay
          una orden pendiente o reciente para lo mismo, para no acumular
          duplicados — igual el server rechaza duplicados exactos (mismo
          destino + mismo mensaje ya pendiente) devolviendo status=duplicate
          con el id existente en vez de crear uno nuevo.
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
            to = params.to or "any"
            dup = inbox.find_pending_duplicate(to, params.message)
            if dup:
                return _j({
                    "status": "duplicate", "id": dup["id"], "to": to,
                    "note": f"ya existe una orden pendiente idéntica (id={dup['id']}, creada {dup['created']}) sin resolver — no se creó una nueva. "
                            "Si es intencional, cambiá el mensaje o pedile al destinatario que la complete/vos cancelala primero.",
                })
            oid = inbox.send(
                message=params.message,
                to_client=to,
                from_client=params.from_client or "unknown",
                checkpoint=params.checkpoint,
                assets=params.assets,
            )
            return _j({
                "status": "sent", "id": oid, "to": to,
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
# Tool 4: status — métricas honestas de la sesión
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
    Métricas de la sesión: tokens que NO entraron al contexto de Claude, lecturas,
    hits de memoria (unchanged/diff), estado del ledger e inbox. Incluye
    `code_staleness`: si server.py o router/*.py cambiaron en disco después de
    que ESTE proceso arrancó (git push no reinicia nada — un proceso MCP no
    recarga módulos solo), lo marca `stale=true` con el archivo y hace cuánto
    cambió. Si aparece stale=true, reiniciá el MCP antes de confiar en el
    comportamiento nuevo. Con deep=true suma diagnóstico local (fastembed/python).
    """
    saved = max(0, _stats["tokens_file_total"] - _stats["tokens_delivered"])
    status = {
        "version": VERSION,
        "uptime_s": round(time.time() - _stats["start_time"]),
        "session": {
            "smart_reads": _stats["smart_reads"],
            "unchanged_hits": _stats["unchanged_hits"],
            "diff_reads": _stats["diff_reads"],
            "tokens_source_total": _stats["tokens_file_total"],
            "tokens_delivered_to_claude": _stats["tokens_delivered"],
            "tokens_kept_out_of_context": saved,
        },
        "ledger": ledger.stats(),
        "inbox_pending": len(inbox.check()),
        "code_staleness": _code_staleness(_stats["start_time"]),
    }
    try:
        pending_raw = ledger.pending_raw_reads()
        if pending_raw:
            status["passive_capture_pending"] = pending_raw
    except Exception:
        pass

    if params.deep:
        diag = {}
        try:
            from router.ranker import _get_embedder
            diag["fastembed"] = "activo" if _get_embedder() else "no instalado (BM25 puro)"
        except Exception:
            diag["fastembed"] = "no instalado (BM25 puro)"
        diag["python"] = sys.version.split()[0]
        status["diagnostics"] = diag

    return _j(status)


# ─────────────────────────────────────────────
# Tool 5: rules — reglas permanentes del proyecto, con procedencia
# ─────────────────────────────────────────────

def _rules_for_paths(paths: list[str]) -> list[str]:
    """Reglas de los proyectos a los que pertenecen estos paths (dedup por archivo de reglas)."""
    seen: set = set()
    collected: list[dict] = []
    for p in paths:
        try:
            rf = project_rules.find_rules_file(p)
        except Exception:
            continue
        if rf and rf not in seen:
            seen.add(rf)
            collected.extend(project_rules.load_rules(rf.parent))
    return project_rules.inject_texts(collected)


@mcp.tool(
    name="router_rules",
    annotations={
        "title": "Reglas Permanentes del Proyecto (con procedencia)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def router_rules(params: RulesInput) -> str:
    """
    Reglas PERMANENTES del proyecto ("nunca usar Redux"), distintas del estado
    de tarea de un checkpoint. Cada una con procedencia: quién, cuándo, y de qué
    checkpoint nació. Viven en .claude-continuity-rules.json en la raíz del
    proyecto — git-friendly, editable a mano, viaja con el repo.

    No hace falta llamar para LEERLAS: se inyectan solas como `project_rules`
    en smart_read y resume de archivos del proyecto.

    add: nueva regla (texto literal, sin síntesis). Dedup automático.
    promote: convierte una decisión de un checkpoint en regla permanente.
    sync_to_claudemd=true: además mantiene una sección delimitada en el
    CLAUDE.md del proyecto (alimenta la memoria nativa de Claude Code).
    """
    pdir = Path(params.project_dir)
    if not pdir.is_dir():
        return _j({"status": "error", "reason": f"project_dir no existe: {params.project_dir}"})

    if params.action == "add":
        if not params.text:
            return _j({"status": "error", "reason": "add requiere `text`"})
        rule, created = project_rules.add_rule(
            pdir, params.text, params.from_client or "unknown", params.checkpoint)
        out = {"status": "added" if created else "duplicate", "rule": rule,
               "rules_file": str(project_rules.rules_path(pdir))}
        if not created:
            out["hint"] = "ya existía una regla equivalente — se devuelve la existente"
        if params.sync_to_claudemd:
            out["claudemd"] = str(project_rules.sync_to_claudemd(pdir))
        return _j(out)

    if params.action == "list":
        return _j({
            "status": "ok",
            "rules": project_rules.load_rules(pdir),
            "rules_file": str(project_rules.rules_path(pdir)),
            "hint": "estas reglas se inyectan solas en smart_read/resume de archivos de este proyecto",
        })

    if params.action == "remove":
        if params.rule_id is None:
            return _j({"status": "error", "reason": "remove requiere `rule_id`"})
        if not project_rules.remove_rule(pdir, params.rule_id):
            return _j({"status": "error", "reason": f"no existe regla id={params.rule_id}"})
        out = {"status": "removed", "rule_id": params.rule_id}
        if params.sync_to_claudemd:
            out["claudemd"] = str(project_rules.sync_to_claudemd(pdir))
        return _j(out)

    # promote — decisión de checkpoint → regla permanente (procedencia incluida)
    if not params.checkpoint:
        return _j({"status": "error", "reason": "promote requiere `checkpoint`"})
    cp_path = _checkpoints_dir / f"{_safe_name(params.checkpoint)}.json"
    if not cp_path.exists():
        return _j({"status": "error", "reason": f"checkpoint '{params.checkpoint}' no existe — usar router_checkpoint action=list"})
    try:
        cp = json.loads(cp_path.read_text(encoding="utf-8"))
    except Exception as e:
        return _j({"status": "error", "reason": f"checkpoint corrupto: {e}"})
    decisions = cp.get("decisions") or []
    if not decisions:
        return _j({"status": "error", "reason": "el checkpoint no tiene `decisions` para promover"})
    idx = params.decision_index
    if idx is None:
        if len(decisions) != 1:
            return _j({"status": "error", "reason": "hay varias decisiones — pasá `decision_index`",
                       "decisions": decisions})
        idx = 0
    if idx >= len(decisions):
        return _j({"status": "error", "reason": f"decision_index {idx} fuera de rango ({len(decisions)} decisiones)",
                   "decisions": decisions})
    rule, created = project_rules.add_rule(
        pdir, decisions[idx], params.from_client or "unknown", params.checkpoint)
    out = {"status": "promoted" if created else "duplicate", "rule": rule,
           "rules_file": str(project_rules.rules_path(pdir))}
    if params.sync_to_claudemd:
        out["claudemd"] = str(project_rules.sync_to_claudemd(pdir))
    return _j(out)


# ─────────────────────────────────────────────
# Tool 6: project_search — búsqueda BM25 cross-archivo, incremental
# ─────────────────────────────────────────────

@mcp.tool(
    name="router_project_search",
    annotations={
        "title": "Búsqueda en Todo el Proyecto (BM25 local, incremental)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def router_project_search(params: ProjectSearchInput) -> str:
    """
    Busca en TODO el proyecto y devuelve los archivos más relevantes con los
    fragmentos EXACTOS de cada uno. Para "¿dónde se maneja X?" cuando no sabés
    en qué archivo está — smart_read es por-archivo, esto es cross-archivo.

    Índice incremental: la primera llamada indexa; las siguientes solo
    re-indexan los archivos cuyo hash cambió (los sin cambios cuestan ~0), y
    el índice persiste cross-sesión y cross-cliente.

    Ventaja sobre grep: ranking por relevancia (no coincidencia literal), y
    entiende snake_case/camelCase. Determinista, 100% local.
    """
    root = Path(params.project_dir)
    if not root.is_dir():
        return _j({"status": "error", "reason": f"project_dir no existe: {params.project_dir}"})

    stats = None
    try:
        # Sin query (o con reindex) el barrido es explícito. Con query, igual se
        # refresca: es incremental, así que sobre un proyecto ya indexado es barato.
        stats = project_index.build_index(ledger, root)
    except Exception as e:
        logger.warning(f"indexación falló: {e}")
        if not params.query:
            return _j({"status": "error", "reason": f"indexación falló: {str(e)[:160]}"})

    if not params.query:
        return _j({"status": "indexed", **(stats or {}),
                   "hint": "volvé a llamar con `query` para buscar en todo el proyecto"})

    try:
        results = project_index.search(ledger, root, params.query,
                                       top_k=params.top_k,
                                       fragments_per_file=params.fragments_per_file)
    except Exception as e:
        return _j({"status": "error", "reason": f"búsqueda falló: {str(e)[:160]}"})

    delivered = sum(_tokens(fr["text"]) for r in results for fr in r.get("fragments", []))
    _stats["tokens_delivered"] += delivered
    out = {
        "status": "results",
        "query": params.query,
        "root": str(root.resolve()),
        "docs_indexed": (stats or {}).get("total_docs"),
        "matches": len(results),
        "tokens_delivered": delivered,
        "results": results,
    }
    if stats:
        out["index"] = {k: stats[k] for k in ("indexed", "unchanged", "removed", "took_s") if k in stats}
    if not results:
        out["hint"] = ("ninguna coincidencia léxica — BM25 necesita compartir palabras con el texto; "
                       "probá otros términos o usá router_smart_read si ya sabés el archivo")
    _rules = _rules_for_paths([str(root)])
    if _rules:
        out["project_rules"] = _rules
    return _j(out)


# ─────────────────────────────────────────────
# MCP Prompts — flujos invocables por el USUARIO (slash commands)
# La continuidad no puede depender solo de que el modelo elija las tools:
# estos prompts la disparan a pedido del humano, sin azar.
# ─────────────────────────────────────────────

@mcp.prompt(name="resume", description="Retomar el trabajo previo: restaura el último checkpoint (o reconstruye la actividad reciente) y chequea órdenes pendientes")
def prompt_resume() -> str:
    return (
        "Retomá el trabajo previo de este proyecto:\n"
        "1. Llamá router_checkpoint con action='resume' (sin name, para el más reciente). "
        "Si la respuesta trae mode='reconstructed_activity', tratála como rastro de archivos y órdenes — "
        "NO como decisiones de arquitectura.\n"
        "2. Llamá router_inbox con action='check' y to=<este cliente> para ver órdenes pendientes.\n"
        "3. Resumile al usuario en 3-5 líneas: dónde quedó el trabajo, qué archivos cambiaron en disco "
        "desde entonces, y qué órdenes hay pendientes. Cerrá proponiendo el siguiente paso concreto."
    )


@mcp.prompt(name="handoff", description="Cerrar la sesión con un traspaso: guarda checkpoint y deja la orden en el inbox de otro cliente")
def prompt_handoff(to: str = "", message: str = "") -> str:
    destino = to.strip() or "<preguntale al usuario: code, cowork, desktop o design>"
    orden = message.strip() or "<preguntale al usuario qué tiene que hacer el receptor>"
    return (
        f"Cerrá esta sesión con un handoff a '{destino}':\n"
        "1. Guardá router_checkpoint action='save' con: name corto y descriptivo, summary de lo hecho "
        "en esta sesión, decisions tomadas, open_items pendientes y files relevantes (rutas absolutas).\n"
        f"2. Dejá la orden con router_inbox action='send', to='{destino}', message='{orden}', "
        "checkpoint=<el nombre que guardaste> y assets=[rutas/URLs] si hay material de handoff.\n"
        "3. Confirmale al usuario: id de la orden, nombre del checkpoint vinculado y qué va a ver "
        "el cliente receptor cuando arranque."
    )


@mcp.prompt(name="inbox", description="Chequear el inbox y ejecutar las órdenes pendientes de este cliente")
def prompt_inbox() -> str:
    return (
        "Chequeá y ejecutá las órdenes del inbox:\n"
        "1. Llamá router_inbox action='check' con to=<este cliente>.\n"
        "2. Para cada orden pendiente: si tiene checkpoint vinculado, restauralo primero con "
        "router_checkpoint action='resume' name=<checkpoint> para el contexto completo; revisá los assets si trae.\n"
        "3. Ejecutá lo pedido. Ante acciones difíciles de revertir (push, borrar, publicar), verificá el "
        "estado real antes y reportá con evidencia.\n"
        "4. Al terminar cada orden, marcala con router_inbox action='complete', order_id y un result "
        "detallado (hashes, conteos, rutas). Si generaste archivos, devolvelos en assets.\n"
        "5. Cerrá con un resumen al usuario y verificá que el inbox quede sin pendientes."
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"INDIVIDRA MCP v{VERSION} — memoria y continuidad para Claude ✓")
    mcp.run()
