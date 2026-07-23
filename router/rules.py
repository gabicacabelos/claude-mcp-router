"""
Reglas de proyecto con procedencia — la memoria que sobrevive a todo.

Una regla es una decisión PERMANENTE del proyecto ("nunca usar Redux", "los
tests van en tests/"), distinta del estado de tarea que captura el checkpoint.
Cada regla lleva procedencia: quién la decidió (cliente/humano), cuándo, y el
checkpoint donde nació si fue promovida desde una decisión.

Diseño deliberado:
- Archivo JSON legible en la RAÍZ DEL PROYECTO del usuario (no en el dir del
  MCP): se versiona en git junto al código, viaja con el repo, sobrevive a
  cualquier cambio de API de Anthropic y es editable a mano.
- Determinista: la regla es texto literal escrito por el humano o promovido
  desde `decisions` de un checkpoint. Nunca síntesis con pérdida vía LLM.
- Distribución sin fricción: las reglas se inyectan en los payloads de
  smart_read y resume (piggyback sobre llamadas que ya ocurren) — no hace
  falta una llamada aparte para leerlas.
- Sync opcional a CLAUDE.md: alimenta la memoria nativa de Claude Code en una
  sección delimitada, en vez de competir con ella.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RULES_FILENAME = ".claude-continuity-rules.json"
CLAUDEMD_START = "<!-- INDIVIDRA RULES START -->"
CLAUDEMD_END = "<!-- INDIVIDRA RULES END -->"
MAX_RULES_INJECTED = 10  # tope de reglas inyectadas en payloads (disciplina de tokens)


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def rules_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / RULES_FILENAME


def load_rules(project_dir: str | Path) -> list[dict]:
    """Reglas del proyecto, [] si no hay archivo o está corrupto (nunca lanza)."""
    p = rules_path(project_dir)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        rules = data.get("rules", []) if isinstance(data, dict) else []
        return [r for r in rules if isinstance(r, dict) and r.get("text")]
    except Exception as e:
        logger.warning(f"rules file corrupto en {p}: {e}")
        return []


def _save_rules(project_dir: str | Path, rules: list[dict]) -> Path:
    p = rules_path(project_dir)
    payload = {
        "_comment": "Reglas permanentes del proyecto (claude-continuity-mcp). "
                    "Editable a mano; commitealo al repo para compartirlas con el equipo.",
        "rules": rules,
    }
    p.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def add_rule(project_dir: str | Path, text: str, source_client: str = "unknown",
             checkpoint: str | None = None) -> tuple[dict, bool]:
    """
    Agrega una regla con procedencia. Devuelve (regla, created).
    Si ya existe una regla con el mismo texto normalizado, devuelve la existente
    (created=False) — las reglas no se acumulan duplicadas.
    """
    rules = load_rules(project_dir)
    norm = _normalize(text)
    for r in rules:
        if _normalize(r["text"]) == norm:
            return r, False
    rule = {
        "id": max((r.get("id", 0) for r in rules), default=0) + 1,
        "text": text.strip(),
        "source_client": (source_client or "unknown").lower().strip(),
        "checkpoint": checkpoint,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    rules.append(rule)
    _save_rules(project_dir, rules)
    return rule, True


def remove_rule(project_dir: str | Path, rule_id: int) -> bool:
    rules = load_rules(project_dir)
    kept = [r for r in rules if r.get("id") != rule_id]
    if len(kept) == len(rules):
        return False
    _save_rules(project_dir, kept)
    return True


def find_rules_file(start: str | Path) -> Path | None:
    """
    Busca el archivo de reglas subiendo desde `start` (archivo o directorio)
    hasta la raíz del filesystem — mismo patrón de descubrimiento que .gitignore.
    """
    p = Path(start)
    if p.is_file():
        p = p.parent
    for d in [p, *p.parents]:
        candidate = d / RULES_FILENAME
        if candidate.is_file():
            return candidate
    return None


def rules_for_file(file_path: str | Path) -> list[dict]:
    """Reglas del proyecto al que pertenece un archivo (walk-up). [] si no hay."""
    found = find_rules_file(file_path)
    if found is None:
        return []
    return load_rules(found.parent)


def sync_to_claudemd(project_dir: str | Path) -> Path:
    """
    Escribe las reglas en una sección delimitada del CLAUDE.md del proyecto.
    Alimenta la memoria nativa de Claude Code sin pisar el contenido del usuario:
    solo se reemplaza lo que está entre los marcadores; el resto queda intacto.
    """
    rules = load_rules(project_dir)
    lines = [CLAUDEMD_START,
             "<!-- Sección generada por claude-continuity-mcp (router_rules). No editar a mano: se regenera en cada sync. -->",
             "## Reglas del proyecto", ""]
    for r in rules:
        prov = f" _(decidida por {r['source_client']}, {r['created']}"
        prov += f", checkpoint `{r['checkpoint']}`)_" if r.get("checkpoint") else ")_"
        lines.append(f"- {r['text']}{prov}")
    lines.append(CLAUDEMD_END)
    block = "\n".join(lines)

    md = Path(project_dir) / "CLAUDE.md"
    if md.is_file():
        content = md.read_text(encoding="utf-8")
        if CLAUDEMD_START in content and CLAUDEMD_END in content:
            pre = content.split(CLAUDEMD_START)[0]
            post = content.split(CLAUDEMD_END, 1)[1]
            content = pre + block + post
        else:
            content = content.rstrip("\n") + "\n\n" + block + "\n"
    else:
        content = block + "\n"
    md.write_text(content, encoding="utf-8")
    return md


def inject_texts(rules: list[dict]) -> list[str]:
    """Textos de reglas para inyectar en payloads, con tope de disciplina de tokens."""
    return [r["text"] for r in rules[:MAX_RULES_INJECTED]]
