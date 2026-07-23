"""
Clasificador de intent usando Groq (LPU ultra-rápido).

Groq se usa SOLO para prompts cortos de clasificación (50-100 tokens).
NO enviar documentos completos — límite free de Groq es 6k TPM.

Intents:
  compress_context  → Contexto de proyecto, comprimir con Gemini
  route_to_cheap    → Tarea delegable a modelo barato
  pass_through      → Enviar directo a Claude sin modificar
  code_task         → Tarea de código (Claude es mejor)
"""

import json
import logging
import os
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

Intent = Literal[
    "compress_context",
    "route_to_cheap",
    "pass_through",
    "code_task",
]

CLASSIFICATION_PROMPT = """Clasificá el siguiente texto en UNA de estas categorías:

- compress_context: Es contexto/documentación de proyecto resumible sin perder información crítica
- route_to_cheap: Es una tarea de generación simple (email genérico, lista, traducción)
- pass_through: Es una pregunta técnica compleja, análisis o decisión arquitectural
- code_task: Involucra código, debugging, refactoring o análisis de código

Respondé SOLO con el JSON: {{"intent": "<categoría>", "confidence": 0.0-1.0}}

Texto a clasificar (primeros 200 chars):
{text_preview}"""


async def classify_intent(
    text: str,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> tuple[Intent, float]:
    """
    Clasifica el intent del texto usando Groq.

    Solo envía los primeros 200 chars para no consumir TPM.

    Returns:
        Tuple (intent, confidence) o ("pass_through", 0.5) en error
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        logger.warning("GROQ_API_KEY no configurada — usando pass_through")
        return "pass_through", 0.5

    text_preview = text[:200].replace("\n", " ").strip()
    prompt = CLASSIFICATION_PROMPT.format(text_preview=text_preview)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 60,
                    "temperature": 0.1,
                },
                timeout=timeout,
            )
            response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        intent = parsed.get("intent", "pass_through")
        confidence = float(parsed.get("confidence", 0.5))

        valid_intents = {"compress_context", "route_to_cheap", "pass_through", "code_task"}
        if intent not in valid_intents:
            logger.warning(f"Intent desconocido: {intent} — usando pass_through")
            return "pass_through", 0.5

        logger.debug(f"Intent: {intent} (confianza: {confidence:.2f})")
        return intent, confidence

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("Groq rate limit — usando pass_through")
        else:
            logger.error(f"Groq HTTP error {e.response.status_code}")
        return "pass_through", 0.5

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"No se pudo parsear respuesta de Groq: {e}")
        return "pass_through", 0.5

    except Exception as e:
        logger.error(f"Error inesperado en clasificación: {e}")
        return "pass_through", 0.5
