"""
Compresor de contexto: Groq (primario, textos cortos/medios) → OpenRouter free (fallback).

Circuit breaker + exponential backoff incluidos.
Groq free tier: ~6000 TPM en llama-3.1-8b-instant → solo se usa si el texto es < 3500 tokens.
OpenRouter actúa de primary para textos grandes y de fallback cuando Groq rate-limit.
"""

import asyncio
import logging
import os

import httpx

from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# ─── Groq ───────────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"   # LPU ultra-rápido, free tier
GROQ_MAX_INPUT_TOKENS = 3500          # Margen conservador sobre el límite de 6k TPM

# ─── OpenRouter (fallback) ───────────────────────────────────────────────────
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_FREE_MODELS = [
    "qwen/qwen-2.5-7b-instruct:free",          # Qwen — alta disponibilidad
    "meta-llama/llama-3.2-3b-instruct:free",   # Llama 3.2 — fallback sólido
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "baidu/cobuddy:free",                      # Baidu — puede devolver content=None
]

COMPRESS_SYSTEM_PROMPT = """Sos un compresor técnico de contexto para sistemas LLM.

Tu tarea: condensar el siguiente texto preservando TODA la información técnica relevante.

Reglas:
- Eliminar frases redundantes, relleno y explicaciones obvias
- Conservar: nombres, funciones, endpoints, errores, decisiones técnicas, números, versiones
- NO agregar información nueva ni interpretar
- Formato: prosa densa o bullet points, lo que sea más compacto
- Respondé SOLO con el texto comprimido, sin introducción

"""


def _build_prompt(text: str, level: str) -> str:
    levels = {
        "light": "Objetivo: 60-70% del tamaño original.",
        "medium": "Objetivo: 40-60% del tamaño original.",
        "heavy": "Objetivo: 25-40% del tamaño original. Máxima densidad.",
    }
    return COMPRESS_SYSTEM_PROMPT + levels.get(level, levels["medium"]) + f"\n\nTexto:\n{text}"


async def _compress_with_groq(
    text: str,
    level: str,
    api_key: str,
    estimated_tokens: int = 0,
    max_retries: int = 2,
) -> str | None:
    """Comprime con Groq LPU. Solo se invoca si el texto entra dentro del límite de TPM."""
    if estimated_tokens > GROQ_MAX_INPUT_TOKENS:
        logger.info(f"Groq skip — texto demasiado largo ({estimated_tokens} tokens > {GROQ_MAX_INPUT_TOKENS})")
        return None

    prompt = _build_prompt(text, level)

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GROQ_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 4096,
                        "temperature": 0.1,
                    },
                    timeout=20.0,
                )
                response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"].get("content")
            if not content:
                logger.warning("Groq retornó content=None")
                return None
            logger.info(f"Compresión via Groq ({GROQ_MODEL})")
            return content.strip()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 10.0 * (attempt + 1)
                logger.warning(f"Groq 429 — retry {attempt + 1}/{max_retries} en {wait:.0f}s")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                continue
            logger.error(f"Groq HTTP {e.response.status_code}: {e.response.text[:300]}")
            return None
        except Exception as e:
            logger.error(f"Groq error ({type(e).__name__}): {e}")
            return None

    return None


async def _compress_with_openrouter(
    text: str,
    level: str,
    api_key: str,
    max_retries: int = 2,
) -> str | None:
    if not api_key:
        return None

    prompt = _build_prompt(text, level)

    for model in OPENROUTER_FREE_MODELS:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        OPENROUTER_API_URL,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "HTTP-Referer": "https://individratec.com",
                            "X-Title": "INDIVIDRA MCP Router",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 4096,
                            "temperature": 0.1,
                        },
                        timeout=45.0,
                    )
                    response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"].get("content")
                if not content:
                    logger.warning(f"OpenRouter {model} retornó content=None — siguiente modelo")
                    break  # saltar al siguiente modelo
                logger.info(f"Compresión via OpenRouter ({model})")
                return content.strip()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    await asyncio.sleep(10.0)
                    continue
                logger.warning(f"OpenRouter {model} falló ({e.response.status_code})")
                break
            except Exception as e:
                logger.warning(f"OpenRouter {model} error: {e}")
                break

    return None


class ContextCompressor:
    """
    Compresor con cascada: Gemini → OpenRouter → pass-through.
    """

    def __init__(self, circuit_breaker: CircuitBreaker | None = None):
        self.cb = circuit_breaker or CircuitBreaker(
            failure_threshold=5, reset_timeout_seconds=300
        )
        self._groq_key = os.getenv("GROQ_API_KEY", "")
        self._openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    async def compress(self, text: str, level: str = "medium") -> tuple[str, str, bool]:
        """
        Comprime texto. Cascada: Groq (textos ≤ 3500 tokens) → OpenRouter → pass-through.

        Returns:
            (texto_resultado, proveedor_usado, fue_comprimido)
        """
        # Estimación rápida de tokens para decidir si Groq puede manejar el texto
        estimated_tokens = len(text) // 4

        if self._groq_key and self.cb.can_call("groq"):
            result = await _compress_with_groq(
                text, level, self._groq_key, estimated_tokens=estimated_tokens
            )
            if result:
                self.cb.record_success("groq")
                return result, "groq", True
            elif estimated_tokens <= GROQ_MAX_INPUT_TOKENS:
                # Solo cuenta como falla si no fue skip por tamaño
                self.cb.record_failure("groq")

        if self._openrouter_key and self.cb.can_call("openrouter"):
            result = await _compress_with_openrouter(text, level, self._openrouter_key)
            if result:
                self.cb.record_success("openrouter")
                return result, "openrouter", True
            else:
                self.cb.record_failure("openrouter")

        logger.warning("Todos los proveedores fallaron — retornando texto original")
        return text, "none", False

    def get_provider_status(self) -> dict:
        return self.cb.get_status()
