"""
Circuit Breaker por proveedor.

Estados:
  CLOSED   → Normal. Las requests pasan.
  OPEN     → Fallo detectado. Bloquea requests por reset_timeout segundos.
  HALF_OPEN → Período de sondeo. Deja pasar 1 request para probar.

Transiciones:
  CLOSED    → OPEN      después de `failure_threshold` fallos consecutivos
  OPEN      → HALF_OPEN después de `reset_timeout` segundos
  HALF_OPEN → CLOSED    si la request de sondeo tiene éxito
  HALF_OPEN → OPEN      si la request de sondeo falla
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ProviderState:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0


class CircuitBreaker:
    """
    Circuit breaker independiente por proveedor (Gemini, Groq, OpenRouter).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 300,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout_seconds
        self._states: Dict[str, ProviderState] = {}

    def _get_state(self, provider: str) -> ProviderState:
        if provider not in self._states:
            self._states[provider] = ProviderState()
        return self._states[provider]

    def can_call(self, provider: str) -> bool:
        state = self._get_state(provider)

        if state.state == CircuitState.CLOSED:
            return True

        if state.state == CircuitState.OPEN:
            elapsed = time.time() - state.last_failure_time
            if elapsed >= self.reset_timeout:
                logger.info(f"Circuit breaker [{provider}]: OPEN → HALF_OPEN")
                state.state = CircuitState.HALF_OPEN
                return True
            return False

        return True  # HALF_OPEN: dejar pasar sondeo

    def record_success(self, provider: str) -> None:
        state = self._get_state(provider)
        state.success_count += 1
        state.failure_count = 0

        if state.state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker [{provider}]: HALF_OPEN → CLOSED ✓")
            state.state = CircuitState.CLOSED

    def record_failure(self, provider: str) -> None:
        state = self._get_state(provider)
        state.failure_count += 1
        state.last_failure_time = time.time()

        if state.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker [{provider}]: HALF_OPEN → OPEN")
            state.state = CircuitState.OPEN
            return

        if state.failure_count >= self.failure_threshold:
            if state.state != CircuitState.OPEN:
                logger.warning(
                    f"Circuit breaker [{provider}]: CLOSED → OPEN "
                    f"({state.failure_count} fallos consecutivos)"
                )
                state.state = CircuitState.OPEN

    def get_status(self) -> Dict[str, dict]:
        result = {}
        for provider, state in self._states.items():
            elapsed = (
                time.time() - state.last_failure_time
                if state.last_failure_time > 0
                else None
            )
            result[provider] = {
                "state": state.state.value,
                "failures": state.failure_count,
                "successes": state.success_count,
                "seconds_since_last_failure": round(elapsed, 1) if elapsed else None,
            }
        return result

    def reset(self, provider: str) -> None:
        if provider in self._states:
            self._states[provider] = ProviderState()
            logger.info(f"Circuit breaker [{provider}]: reset manual")
