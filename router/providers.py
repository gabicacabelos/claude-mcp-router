"""
Pool de proveedores gratuitos con failover transparente.

Cascada: Groq (LPU, rápido) → OpenRouter free models (rotación automática).
Nunca lanza excepciones: si todo falla devuelve (None, "none").

Los free tiers mueren y cambian constantemente — por eso la lista de modelos
de OpenRouter se recorre en orden hasta encontrar uno vivo, y el circuit
breaker evita insistir con proveedores caídos.
"""

import asyncio
import logging
import os
import time

import httpx

from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_INPUT_CHARS = 14000  # ~3500 tokens, margen bajo el límite de 6k TPM

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Fallback estático — solo se usa si el descubrimiento dinámico falla.
# El catálogo :free de OpenRouter rota constantemente: la fuente de verdad
# es get_free_models(), que consulta la API y se cachea 6 horas.
OPENROUTER_FREE_MODELS = [
    "qwen/qwen-2.5-7b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

# Familias preferidas para el ranking del descubrimiento (mejores primero)
_FAMILY_PREFERENCE = ["qwen3", "qwen", "llama-3.3", "llama", "nemotron", "gpt-oss", "gemma", "mistral", "deepseek"]

_free_cache: list[str] = []
_free_cache_ts: float = 0.0
FREE_MODELS_TTL = 6 * 3600


async def get_free_models() -> list[str]:
    """
    Descubre en runtime qué modelos :free están vivos en OpenRouter
    (pricing prompt=0 y completion=0). Cache de 6h. Nunca lanza excepción:
    si la API falla, devuelve el último cache o el fallback estático.
    """
    global _free_cache, _free_cache_ts
    if _free_cache and time.time() - _free_cache_ts < FREE_MODELS_TTL:
        return _free_cache
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(OPENROUTER_MODELS_URL, timeout=15.0)
        r.raise_for_status()
        data = r.json().get("data", [])
        free = [
            m["id"] for m in data
            if m.get("pricing", {}).get("prompt") == "0"
            and m.get("pricing", {}).get("completion") == "0"
            and m.get("id")
        ]

        def rank(mid: str) -> int:
            low = mid.lower()
            for i, fam in enumerate(_FAMILY_PREFERENCE):
                if fam in low:
                    return i
            return len(_FAMILY_PREFERENCE)

        free.sort(key=rank)
        if free:
            _free_cache = free[:8]
            _free_cache_ts = time.time()
            logger.info(f"OpenRouter free models descubiertos: {_free_cache}")
            return _free_cache
        logger.warning("OpenRouter no reporta modelos free — usando fallback estático")
    except Exception as e:
        logger.warning(f"Descubrimiento de modelos free falló ({type(e).__name__}) — usando fallback")
    return _free_cache or OPENROUTER_FREE_MODELS

_HEADERS_OR_EXTRA = {"HTTP-Referer": "https://individratec.com", "X-Title": "INDIVIDRA MCP"}


async def _try_groq(prompt: str, max_tokens: int, json_mode: bool, timeout: float) -> str | None:
    key = os.getenv("GROQ_API_KEY", "")
    if not key or len(prompt) > GROQ_MAX_INPUT_CHARS:
        return None
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=timeout,
            )
        if resp.status_code == 429:
            logger.warning("Groq 429 — pasando a OpenRouter")
            return None
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content")
        return content.strip() if content else None
    except Exception as e:
        logger.warning(f"Groq error: {type(e).__name__}: {str(e)[:150]}")
        return None


async def _try_openrouter(prompt: str, max_tokens: int, timeout: float) -> tuple[str, str] | None:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None
    for model in await get_free_models():
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    OPENROUTER_API_URL,
                    headers={"Authorization": f"Bearer {key}", **_HEADERS_OR_EXTRA},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                    },
                    timeout=timeout,
                )
            if resp.status_code == 429:
                # Rate limit global de la cuenta free — esperar una vez y reintentar el mismo modelo
                await asyncio.sleep(8)
                continue
            if resp.status_code != 200:
                logger.warning(f"OpenRouter {model}: HTTP {resp.status_code} — siguiente modelo")
                continue
            content = resp.json()["choices"][0]["message"].get("content")
            if content:
                return content.strip(), model
        except Exception as e:
            logger.warning(f"OpenRouter {model}: {type(e).__name__}: {str(e)[:120]}")
            continue
    return None


class CheapLLM:
    """Interfaz única al pool de modelos gratuitos, con circuit breaker."""

    def __init__(self, circuit_breaker: CircuitBreaker | None = None):
        self.cb = circuit_breaker or CircuitBreaker(failure_threshold=4, reset_timeout_seconds=180)

    async def call(
        self,
        prompt: str,
        max_tokens: int = 2048,
        json_mode: bool = False,
        timeout: float = 30.0,
    ) -> tuple[str | None, str]:
        """
        Returns: (respuesta | None, proveedor). NUNCA lanza excepción.
        """
        if self.cb.can_call("groq"):
            result = await _try_groq(prompt, max_tokens, json_mode, timeout)
            if result:
                self.cb.record_success("groq")
                return result, f"groq/{GROQ_MODEL}"
            if len(prompt) <= GROQ_MAX_INPUT_CHARS and os.getenv("GROQ_API_KEY"):
                self.cb.record_failure("groq")

        if self.cb.can_call("openrouter"):
            result = await _try_openrouter(prompt, max_tokens, timeout + 15)
            if result:
                self.cb.record_success("openrouter")
                text, model = result
                return text, model
            if os.getenv("OPENROUTER_API_KEY"):
                self.cb.record_failure("openrouter")

        return None, "none"

    def status(self) -> dict:
        return self.cb.get_status()
