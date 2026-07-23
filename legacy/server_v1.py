#!/usr/bin/env python3
"""
INDIVIDRA MCP Router — Servidor FastMCP principal

Herramientas expuestas a Claude:
  router_compress_context  → Comprime texto grande antes de enviarlo a Claude
  router_route_task        → Delega tarea completa a modelo barato
  router_smart_read        → Lee archivo con compresión inteligente por tier
  router_status            → Estado del sistema: circuit breakers, caché, tokens ahorrados

─────────────────────────────────────────────
Configuración en claude_desktop_config.json:
─────────────────────────────────────────────
{
  "mcpServers": {
    "individra-router": {
      "command": "python",
      "args": ["C:/Users/TU_USUARIO/Escritorio/INDIVIDRA/individra-mcp-router/server.py"],
      "env": {
        "GEMINI_API_KEY": "tu_clave",
        "GROQ_API_KEY": "tu_clave"
      }
    }
  }
}
─────────────────────────────────────────────
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Literal, Optional

import httpx

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

from router.cache import RouterCache
from router.circuit_breaker import CircuitBreaker
from router.classifier import classify_intent
from router.compressor import ContextCompressor
from router.tiers import TierDetector, Tier

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────

# Buscar .env en el mismo directorio que server.py
_server_dir = Path(__file__).parent
load_dotenv(_server_dir / ".env", override=True)  # .env siempre tiene prioridad sobre el entorno del sistema

_config_path = _server_dir / "router_config.yaml"
with open(_config_path) as f:
    CONFIG = yaml.safe_load(f)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("individra-router")

# ─────────────────────────────────────────────
# Componentes
# ─────────────────────────────────────────────

# Siempre ruta absoluta relativa al script — evita errores de permisos con rutas relativas
_db_path = str(_server_dir / "cache" / "router_cache.db")

cache = RouterCache(
    db_path=_db_path,
    ttl_static=CONFIG["cache"].get("ttl_static_seconds", 86400),
    ttl_dynamic=CONFIG["cache"].get("ttl_dynamic_seconds", 3600),
)

circuit_breaker = CircuitBreaker(
    failure_threshold=CONFIG["circuit_breaker"]["failure_threshold"],
    reset_timeout_seconds=CONFIG["circuit_breaker"]["reset_timeout_seconds"],
)

compressor = ContextCompressor(circuit_breaker=circuit_breaker)

tier_detector = TierDetector(
    tier1_max_tokens=CONFIG["tiers"]["tier1_max_tokens"],
    tier2_min_tokens=CONFIG["tiers"]["tier2_min_tokens"],
)

_session_stats = {
    "start_time": time.time(),
    "tokens_original": 0,
    "tokens_compressed": 0,
    "cache_hits": 0,
    "api_calls_gemini": 0,
    "api_calls_openrouter": 0,
    "tasks_routed": 0,
}

# ─────────────────────────────────────────────
# FastMCP
# ─────────────────────────────────────────────

mcp = FastMCP("individra_router_mcp")


# ─────────────────────────────────────────────
# Modelos Pydantic
# ─────────────────────────────────────────────

class CompressInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(..., description="Texto a comprimir (mínimo 50 chars)", min_length=50)
    level: Literal["light", "medium", "heavy"] = Field(
        default="medium",
        description="light=60-70% del original | medium=40-60% | heavy=25-40%",
    )
    context_type: Literal["static", "dynamic"] = Field(
        default="static",
        description="static=TTL 24h (docs del proyecto) | dynamic=TTL 1h (contenido cambiante)",
    )
    use_cache: bool = Field(default=True, description="False para forzar nueva compresión")


class RouteTaskInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(..., description="Tarea a delegar al modelo barato", min_length=10)
    context: Optional[str] = Field(default=None, description="Contexto adicional (se comprime si es grande)")


class SmartReadInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    file_path: str = Field(..., description="Ruta absoluta al archivo a leer", min_length=1)
    force_compress: bool = Field(default=False, description="Forzar compresión independientemente del tier")


class ReadManyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(..., description="Rutas absolutas a los archivos a leer", min_length=1)
    level: Literal["light", "medium", "heavy"] = Field(default="medium")
    use_cache: bool = Field(default=True)


class ExtractActionsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(..., description="Texto a analizar (notas, emails, transcripciones)", min_length=50)
    output_format: Literal["json", "markdown"] = Field(
        default="markdown",
        description="markdown para lectura humana | json para procesamiento automático",
    )


# ─────────────────────────────────────────────
# Herramientas
# ─────────────────────────────────────────────

@mcp.tool(
    name="router_compress_context",
    annotations={
        "title": "Comprimir Contexto con Gemini",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def router_compress_context(params: CompressInput) -> str:
    """
    Comprime un texto grande usando Gemini Flash Lite antes de pasarlo a Claude.

    Flujo:
      1. Detecta tier (Tier 0 = no comprimir, código/stack traces)
      2. Verifica caché SHA-256 — retorna al instante si ya fue comprimido
      3. Llama a Gemini (retry automático con backoff en caso de 429)
      4. Fallback a OpenRouter si Gemini falla
      5. Guarda en caché para futuras llamadas

    Usar para: documentos de proyecto, READMEs, wikis, especificaciones largas.
    NO usar para: código fuente, stack traces, JSON payloads.
    """
    await cache.init()

    tier = tier_detector.detect(params.text)
    token_count = tier_detector.estimate_tokens(params.text)

    if tier == Tier.ZERO:
        return json.dumps({
            "status": "skipped",
            "reason": "Tier 0 — contenido de integridad crítica (código/configuración)",
            "text": params.text,
            "tokens_original": token_count,
            "tokens_result": token_count,
            "savings_pct": 0,
        }, ensure_ascii=False)

    if token_count < 200:
        return json.dumps({
            "status": "skipped",
            "reason": f"Texto corto ({token_count} tokens) — compresión no rentable",
            "text": params.text,
            "tokens_original": token_count,
            "tokens_result": token_count,
            "savings_pct": 0,
        }, ensure_ascii=False)

    cache_key = RouterCache.hash(params.text)

    if params.use_cache:
        cached = await cache.get(cache_key)
        if cached:
            cached_tokens = tier_detector.estimate_tokens(cached)
            savings_pct = round((1 - cached_tokens / token_count) * 100, 1)
            _session_stats["cache_hits"] += 1
            _session_stats["tokens_original"] += token_count
            _session_stats["tokens_compressed"] += cached_tokens
            return json.dumps({
                "status": "cache_hit",
                "text": cached,
                "tokens_original": token_count,
                "tokens_result": cached_tokens,
                "savings_pct": savings_pct,
                "provider": "cache",
            }, ensure_ascii=False)

    level = params.level
    if tier == Tier.TWO and level == "medium":
        level = "heavy"

    compressed, provider, success = await compressor.compress(params.text, level=level)

    if success:
        compressed_tokens = tier_detector.estimate_tokens(compressed)
        savings_pct = round((1 - compressed_tokens / token_count) * 100, 1)

        if params.use_cache:
            ttl = (
                CONFIG["cache"]["ttl_static_seconds"]
                if params.context_type == "static"
                else CONFIG["cache"]["ttl_dynamic_seconds"]
            )
            await cache.set(cache_key, compressed, ttl=ttl, provider=provider)

        _session_stats["tokens_original"] += token_count
        _session_stats["tokens_compressed"] += compressed_tokens
        if provider == "gemini":
            _session_stats["api_calls_gemini"] += 1
        elif provider == "openrouter":
            _session_stats["api_calls_openrouter"] += 1

        logger.info(f"Compresión: {token_count} → {compressed_tokens} tokens ({savings_pct}% ahorro) via {provider}")

        return json.dumps({
            "status": "compressed",
            "text": compressed,
            "tokens_original": token_count,
            "tokens_result": compressed_tokens,
            "savings_pct": savings_pct,
            "provider": provider,
            "level": level,
        }, ensure_ascii=False)

    return json.dumps({
        "status": "failed",
        "reason": "Todos los proveedores fallaron — usar texto original",
        "text": params.text,
        "tokens_original": token_count,
        "tokens_result": token_count,
        "savings_pct": 0,
    }, ensure_ascii=False)


@mcp.tool(
    name="router_route_task",
    annotations={
        "title": "Delegar Tarea a Modelo Barato",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def router_route_task(params: RouteTaskInput) -> str:
    """
    Delega una tarea completa a Gemini Flash Lite o modelos gratuitos de OpenRouter.

    Usar para: emails template, traducciones, variaciones de texto, resúmenes cortos.
    NO usar para: debugging de código, decisiones arquitecturales, análisis técnico profundo.
    """
    await cache.init()

    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        intent, confidence = await classify_intent(params.task)
        if intent == "code_task" and confidence > 0.7:
            return json.dumps({
                "status": "rejected",
                "reason": f"Tarea de código detectada (confianza {confidence:.0%}) — usar Claude directamente",
                "task": params.task,
            }, ensure_ascii=False)

    full_prompt = params.task
    if params.context:
        ctx_tokens = tier_detector.estimate_tokens(params.context)
        if ctx_tokens > 1000:
            compressed_ctx, _, success = await compressor.compress(params.context, level="medium")
            context_to_use = compressed_ctx if success else params.context
        else:
            context_to_use = params.context
        full_prompt = f"Contexto:\n{context_to_use}\n\nTarea:\n{params.task}"

    groq_key = os.getenv("GROQ_API_KEY", "")
    result_text = None
    provider_used = "none"

    if groq_key and circuit_breaker.can_call("groq"):
        try:
            async with __import__("httpx").AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": full_prompt}],
                        "max_tokens": 4096,
                        "temperature": 0.7,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
            data = response.json()
            result_text = data["choices"][0]["message"].get("content")
            if result_text:
                provider_used = "groq/llama-3.1-8b-instant"
                circuit_breaker.record_success("groq")
                _session_stats["api_calls_gemini"] += 1  # reusa el contador existente
            else:
                circuit_breaker.record_failure("groq")
        except Exception as e:
            logger.warning(f"Groq route_task falló: {e}")
            circuit_breaker.record_failure("groq")

    if result_text is None:
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key:
            for model in ["baidu/cobuddy:free", "qwen/qwen-2.5-7b-instruct:free", "meta-llama/llama-3.2-3b-instruct:free"]:
                try:
                    async with __import__("httpx").AsyncClient() as client:
                        response = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {openrouter_key}", "HTTP-Referer": "https://individratec.com"},
                            json={"model": model, "messages": [{"role": "user", "content": full_prompt}], "max_tokens": 4096},
                            timeout=45.0,
                        )
                        response.raise_for_status()
                    data = response.json()
                    result_text = data["choices"][0]["message"]["content"]
                    provider_used = model
                    _session_stats["api_calls_openrouter"] += 1
                    break
                except Exception as e:
                    logger.warning(f"OpenRouter {model} falló: {e}")
                    continue

    if result_text is None:
        return json.dumps({
            "status": "failed",
            "reason": "Todos los proveedores fallaron. Ejecutar tarea con Claude directamente.",
        }, ensure_ascii=False)

    _session_stats["tasks_routed"] += 1

    return json.dumps({
        "status": "success",
        "result": result_text,
        "provider": provider_used,
        "note": "Generado por modelo externo — revisar antes de usar en contexto crítico",
    }, ensure_ascii=False)


@mcp.tool(
    name="router_smart_read",
    annotations={
        "title": "Leer Archivo con Compresión Inteligente",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def router_smart_read(params: SmartReadInput) -> str:
    """
    Lee un archivo y aplica compresión inteligente según tipo y tamaño.

    Tier 0 (código, .json, .yaml, configs) → retorna sin modificar
    Tier 1 (docs cortos < 2000 tokens) → retorna sin modificar
    Tier 2+ (docs > 10000 tokens) → compresión fuerte (~70%)

    Caché SHA-256: el mismo archivo no se comprime dos veces si no cambió.
    """
    await cache.init()

    file_path = Path(params.file_path)
    if not file_path.exists():
        return json.dumps({"status": "error", "reason": f"Archivo no encontrado: {params.file_path}"})

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return json.dumps({"status": "error", "reason": f"No se pudo leer: {e}"})

    tier = tier_detector.detect(content, file_path=str(file_path))
    token_count = tier_detector.estimate_tokens(content)

    if tier == Tier.ZERO or (token_count < 500 and not params.force_compress):
        return json.dumps({
            "status": "raw",
            "file": str(file_path),
            "tier": tier.value,
            "content": content,
            "tokens": token_count,
        }, ensure_ascii=False)

    cache_key = RouterCache.hash(content)
    cached = await cache.get(cache_key)
    if cached:
        cached_tokens = tier_detector.estimate_tokens(cached)
        _session_stats["cache_hits"] += 1
        return json.dumps({
            "status": "cache_hit",
            "file": str(file_path),
            "tier": tier.value,
            "content": cached,
            "tokens_original": token_count,
            "tokens_result": cached_tokens,
            "savings_pct": round((1 - cached_tokens / token_count) * 100, 1),
        }, ensure_ascii=False)

    level = "heavy" if (tier == Tier.TWO or params.force_compress) else "medium"
    compressed, provider, success = await compressor.compress(content, level=level)

    if success:
        compressed_tokens = tier_detector.estimate_tokens(compressed)
        await cache.set(cache_key, compressed, provider=provider)
        _session_stats["tokens_original"] += token_count
        _session_stats["tokens_compressed"] += compressed_tokens
        return json.dumps({
            "status": "compressed",
            "file": str(file_path),
            "tier": tier.value,
            "content": compressed,
            "tokens_original": token_count,
            "tokens_result": compressed_tokens,
            "savings_pct": round((1 - compressed_tokens / token_count) * 100, 1),
            "provider": provider,
        }, ensure_ascii=False)

    return json.dumps({
        "status": "raw_fallback",
        "file": str(file_path),
        "tier": tier.value,
        "content": content,
        "tokens": token_count,
        "note": "Compresión falló — se retorna contenido original",
    }, ensure_ascii=False)


@mcp.tool(
    name="router_read_many",
    annotations={
        "title": "Leer Múltiples Archivos con Compresión Paralela",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def router_read_many(params: ReadManyInput) -> str:
    """
    Lee múltiples archivos en paralelo, aplica tier detection y compresión a cada uno,
    y entrega un único bloque de contexto consolidado listo para Claude.

    Tier 0 (código, configs) → raw, integridad garantizada
    Tier 1 (< 500 tokens)   → raw, compresión no rentable
    Tier 2+ (docs largos)   → comprimido via Groq/OpenRouter

    Graceful degradation: archivos no encontrados se reportan en metadata
    sin interrumpir el procesamiento de los archivos válidos.
    """
    await cache.init()

    errors = []
    valid_paths = []

    for path_str in params.paths:
        p = Path(path_str)
        if not p.exists():
            errors.append({"path": path_str, "error": "archivo no encontrado"})
        elif not p.is_file():
            errors.append({"path": path_str, "error": "no es un archivo"})
        else:
            valid_paths.append(p)

    if not valid_paths:
        return json.dumps({
            "status": "error",
            "reason": "Ningún archivo válido encontrado",
            "errors": errors,
        }, ensure_ascii=False)

    async def read_file(p: Path) -> tuple:
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            return p, content, None
        except Exception as e:
            return p, "", str(e)

    read_results = await asyncio.gather(*[read_file(p) for p in valid_paths])

    for p, _, err in read_results:
        if err:
            errors.append({"path": str(p), "error": err})

    async def process_file(p: Path, content: str) -> dict:
        tier = tier_detector.detect(content, file_path=str(p))
        token_count = tier_detector.estimate_tokens(content)

        if tier == Tier.ZERO or token_count < 500:
            return {
                "name": p.name, "path": str(p), "status": "raw",
                "tier": tier.value, "content": content,
                "tokens_original": token_count, "tokens_result": token_count,
                "savings_pct": 0, "provider": "none",
            }

        cache_key = RouterCache.hash(content)
        if params.use_cache:
            cached = await cache.get(cache_key)
            if cached:
                cached_tokens = tier_detector.estimate_tokens(cached)
                _session_stats["cache_hits"] += 1
                return {
                    "name": p.name, "path": str(p), "status": "cache_hit",
                    "tier": tier.value, "content": cached,
                    "tokens_original": token_count, "tokens_result": cached_tokens,
                    "savings_pct": round((1 - cached_tokens / token_count) * 100, 1),
                    "provider": "cache",
                }

        compressed, provider, success = await compressor.compress(content, level=params.level)
        if success:
            compressed_tokens = tier_detector.estimate_tokens(compressed)
            if params.use_cache:
                await cache.set(cache_key, compressed, provider=provider)
            _session_stats["tokens_original"] += token_count
            _session_stats["tokens_compressed"] += compressed_tokens
            return {
                "name": p.name, "path": str(p), "status": "compressed",
                "tier": tier.value, "content": compressed,
                "tokens_original": token_count, "tokens_result": compressed_tokens,
                "savings_pct": round((1 - compressed_tokens / token_count) * 100, 1),
                "provider": provider,
            }

        return {
            "name": p.name, "path": str(p), "status": "raw_fallback",
            "tier": tier.value, "content": content,
            "tokens_original": token_count, "tokens_result": token_count,
            "savings_pct": 0, "provider": "none",
        }

    file_results = await asyncio.gather(*[
        process_file(p, content)
        for p, content, err in read_results if not err
    ])

    context_parts = []
    total_original = 0
    total_result = 0

    for fr in file_results:
        header = (
            f"=== {fr['name']} | tier:{fr['tier']} | "
            f"{fr['status']} | {fr['tokens_result']} tokens ==="
        )
        context_parts.append(f"{header}\n{fr['content']}")
        total_original += fr["tokens_original"]
        total_result += fr["tokens_result"]

    total_savings = round((1 - total_result / total_original) * 100, 1) if total_original > 0 else 0

    return json.dumps({
        "status": "ok",
        "files_processed": len(file_results),
        "files_skipped": len(errors),
        "tokens_original": total_original,
        "tokens_result": total_result,
        "savings_pct": total_savings,
        "errors": errors or None,
        "context": "\n\n".join(context_parts),
    }, ensure_ascii=False)


@mcp.tool(
    name="router_extract_actions",
    annotations={
        "title": "Extraer Decisiones y Action Items de Texto",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def router_extract_actions(params: ExtractActionsInput) -> str:
    """
    Extrae decisiones, action items y preguntas abiertas de texto no estructurado.
    Funciona con notas de reunión, hilos de email, transcripciones de calls.

    Si el texto supera 3000 tokens, lo comprime primero antes de enviar a Groq.
    Output en markdown (para lectura) o JSON (para procesamiento automático).
    """
    await cache.init()

    groq_key = os.getenv("GROQ_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    estimated_tokens = len(params.text) // 4
    text_to_process = params.text

    if estimated_tokens > 3000:
        compressed, _, success = await compressor.compress(params.text, level="medium")
        if success:
            text_to_process = compressed

    extraction_prompt = f"""Analizá el siguiente texto y extraé información estructurada.

Respondé ÚNICAMENTE con un objeto JSON válido con exactamente estas claves:
{{
  "decisions": ["decisión concreta 1", "decisión concreta 2"],
  "action_items": [
    {{"task": "descripción de la tarea", "owner": "nombre o null", "deadline": "fecha o null"}}
  ],
  "open_questions": ["pregunta sin resolver 1"],
  "executive_summary": "máximo 3 oraciones describiendo el contenido"
}}

Reglas:
- decisions: solo hechos acordados, no intenciones vagas
- action_items: solo tareas concretas y accionables
- open_questions: temas sin resolución que requieren seguimiento
- Si no hay elementos en una categoría, dejá el array vacío []

Texto:
{text_to_process}"""

    result_data = None
    provider_used = "none"

    if groq_key:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": extraction_prompt}],
                        "max_tokens": 2048,
                        "temperature": 0.05,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=25.0,
                )
                response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content")
            if content:
                result_data = json.loads(content)
                provider_used = "groq"
        except Exception as e:
            logger.warning(f"Groq extract_actions falló: {e}")

    if result_data is None and openrouter_key:
        for model in ["qwen/qwen-2.5-7b-instruct:free", "meta-llama/llama-3.2-3b-instruct:free"]:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_key}",
                            "HTTP-Referer": "https://individratec.com",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": extraction_prompt}],
                            "max_tokens": 2048,
                            "temperature": 0.05,
                        },
                        timeout=45.0,
                    )
                    response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content")
                if content:
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        result_data = json.loads(match.group())
                        provider_used = model
                        break
            except Exception as e:
                logger.warning(f"OpenRouter {model} extract_actions falló: {e}")
                continue

    if result_data is None:
        return json.dumps({
            "status": "error",
            "reason": "Todos los proveedores fallaron — intentar con texto más corto",
        }, ensure_ascii=False)

    if params.output_format == "markdown":
        md = []
        if result_data.get("executive_summary"):
            md.append(f"## Resumen\n{result_data['executive_summary']}\n")
        if result_data.get("decisions"):
            md.append("## Decisiones tomadas")
            for d in result_data["decisions"]:
                md.append(f"- {d}")
            md.append("")
        if result_data.get("action_items"):
            md.append("## Action items")
            for ai in result_data["action_items"]:
                owner = f" — **{ai['owner']}**" if ai.get("owner") else ""
                deadline = f" `{ai['deadline']}`" if ai.get("deadline") else ""
                md.append(f"- [ ] {ai['task']}{owner}{deadline}")
            md.append("")
        if result_data.get("open_questions"):
            md.append("## Preguntas abiertas")
            for q in result_data["open_questions"]:
                md.append(f"- {q}")

        return json.dumps({
            "status": "ok",
            "provider": provider_used,
            "tokens_input": estimated_tokens,
            "output": "\n".join(md),
        }, ensure_ascii=False)

    return json.dumps({
        "status": "ok",
        "provider": provider_used,
        "tokens_input": estimated_tokens,
        "output": result_data,
    }, ensure_ascii=False)


@mcp.tool(
    name="router_status",
    annotations={
        "title": "Estado del Router MCP",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def router_status() -> str:
    """
    Estado actual del router: circuit breakers, tokens ahorrados en la sesión,
    estadísticas del caché y proveedores configurados.
    """
    await cache.init()
    cache_stats = await cache.get_stats()

    session_duration = round(time.time() - _session_stats["start_time"])
    tokens_saved = _session_stats["tokens_original"] - _session_stats["tokens_compressed"]
    savings_pct = (
        round(tokens_saved / _session_stats["tokens_original"] * 100, 1)
        if _session_stats["tokens_original"] > 0
        else 0
    )

    status = {
        "router_version": "1.0.0",
        "mode": CONFIG.get("mode", "personal"),
        "session_duration_seconds": session_duration,
        "providers_configured": {
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        },
        "circuit_breakers": circuit_breaker.get_status(),
        "session_stats": {
            "tokens_original": _session_stats["tokens_original"],
            "tokens_after_compression": _session_stats["tokens_compressed"],
            "tokens_saved": tokens_saved,
            "savings_pct": savings_pct,
            "cache_hits": _session_stats["cache_hits"],
            "api_calls_gemini": _session_stats["api_calls_gemini"],
            "api_calls_openrouter": _session_stats["api_calls_openrouter"],
            "tasks_routed": _session_stats["tasks_routed"],
        },
        "cache": cache_stats,
    }

    return json.dumps(status, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# Herramienta de diagnóstico
# ─────────────────────────────────────────────

@mcp.tool(
    name="router_diagnose",
    annotations={
        "title": "Diagnosticar Conectividad de Proveedores",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def router_diagnose() -> str:
    """
    Testea la conectividad real con Gemini y OpenRouter.
    Devuelve el error exacto si algo falla. Usar para diagnóstico.
    """
    import httpx
    import sys

    results = {}

    # Test Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        results["groq"] = {"status": "error", "detail": "GROQ_API_KEY no configurada"}
    else:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": "Reply with just the word OK"}],
                        "max_tokens": 10,
                        "temperature": 0.1,
                    },
                    timeout=15.0,
                )
            if response.status_code == 200:
                data = response.json()
                reply = data["choices"][0]["message"].get("content", "")
                results["groq"] = {"status": "ok", "response": reply.strip(), "model": "llama-3.1-8b-instant"}
            else:
                results["groq"] = {
                    "status": "error",
                    "http_code": response.status_code,
                    "detail": response.text[:500],
                }
        except Exception as e:
            results["groq"] = {"status": "exception", "type": type(e).__name__, "detail": str(e)[:300]}

    # Test OpenRouter — prueba en cascada hasta encontrar uno que funcione
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        results["openrouter"] = {"status": "error", "detail": "OPENROUTER_API_KEY no configurada"}
    else:
        _or_test_models = [
            "qwen/qwen-2.5-7b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ]
        _or_result = {"status": "error", "detail": "Todos los modelos fallaron"}
        for _or_model in _or_test_models:
            try:
                async with httpx.AsyncClient() as client:
                    _or_resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_key}",
                            "HTTP-Referer": "https://individratec.com",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": _or_model,
                            "messages": [{"role": "user", "content": "Reply with just the word OK"}],
                            "max_tokens": 10,
                        },
                        timeout=20.0,
                    )
                if _or_resp.status_code == 200:
                    _or_data = _or_resp.json()
                    _or_reply = _or_data["choices"][0]["message"].get("content")
                    if _or_reply:
                        _or_result = {"status": "ok", "response": _or_reply.strip(), "model": _or_model}
                        break
                    else:
                        _or_result = {"status": "warn", "detail": f"{_or_model}: content=None (model filter)", "model": _or_model}
                elif _or_resp.status_code == 429:
                    _or_result = {"status": "warn", "http_code": 429, "detail": f"{_or_model}: rate limit transitorio", "model": _or_model}
                    break  # 429 es transitorio — no seguir probando
                else:
                    _or_result = {"status": "error", "http_code": _or_resp.status_code, "detail": f"{_or_model}: {_or_resp.text[:200]}"}
            except Exception as _or_e:
                _or_result = {"status": "exception", "type": type(_or_e).__name__, "detail": str(_or_e)[:200], "model": _or_model}
        results["openrouter"] = _or_result

    # Info del entorno
    results["env"] = {
        "python_version": sys.version,
        "groq_key_prefix": groq_key[:8] + "..." if groq_key else "N/A",
        "openrouter_key_prefix": openrouter_key[:8] + "..." if openrouter_key else "N/A",
    }

    return json.dumps(results, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("INDIVIDRA MCP Router iniciado ✓")
    mcp.run()
