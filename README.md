# INDIVIDRA MCP — Memoria persistente y compartida para Claude

MCP server que le da a Claude algo que hoy no tiene: **memoria que persiste entre sesiones y se comparte entre todos sus clientes** (Code, Desktop, Cowork, Design). Recuerda qué archivos ya leíste y devuelve solo los diffs, retoma tareas donde las dejaste, y coordina órdenes entre clientes. 100% local, determinista, sin API keys.

## Por qué existe

Claude es amnésico entre sesiones y ciego entre clientes: lo que leíste en Claude Code no existe para Claude Desktop, y cada sesión nueva re-lee los mismos archivos como si fuera la primera vez. Este MCP es el mismo proceso local compartido por todos tus clientes, con estado persistente en disco — es la memoria que Claude no tiene. El ahorro de tokens que produce es real, pero es una consecuencia, no la promesa: lo que diferencia al proyecto es la **continuidad**.

## Qué es — y qué NO es

**Es:**

- Continuidad: lo leído/decidido en una sesión sigue disponible en la próxima, y en cualquier otro cliente.
- Una capa para que archivos de 20.000 tokens no entren enteros al contexto de Claude cuando solo necesitás 400.
- Determinista donde importa: `smart_read` devuelve fragmentos *exactos* del archivo (ranking local BM25/embeddings) y diffs *exactos*, nunca resúmenes con pérdida generados por un modelo débil.

**NO es:**

- Un "ahorrador mágico de tokens" para tu chat diario. Si tu sesión es conversación normal, esto no te ayuda — y ninguna herramienta lo hará.
- Un compresor de código. Comprimir código con LLMs baratos degrada las respuestas de Claude y termina costando más en reintentos. Nunca se hizo acá.

> **Nota:** el procesamiento masivo en modelos gratuitos (antes `router_bulk_process`) se separó a su propio repo, [`individra-bulk-offload`](https://github.com/gabicacabelos/individra-bulk-offload), porque es otro producto: dependía de servicios externos que diluían este núcleo 100% local.

## Las 4 herramientas

### `router_smart_read` — Lectura quirúrgica con memoria (local, $0, sin APIs)

```
smart_read(file_path="docs/manual.md", query="¿dónde se configuran los webhooks?")
→ los 2-4 fragmentos exactos relevantes, con números de línea
```

- Archivo chico (≤ ~6KB): lo devuelve entero, limpio.
- Archivo grande + `query`: chunking + ranking híbrido (fastembed si está instalado, BM25 puro-Python si no) → solo los fragmentos relevantes. Ahorro típico: 70-90% del archivo, **cero pérdida de fidelidad**.
- Archivo grande sin `query`: mapa estructural (outline con líneas) para decidir qué pedir.
- HTML: se limpia localmente (scripts, tags, boilerplate fuera) antes de procesar.

**Memoria cross-sesión (diff reads):** cada lectura queda registrada (hash + snapshot en SQLite local). En cualquier sesión futura, en cualquier cliente:

- Archivo sin cambios → `{"status":"unchanged","outline":[...]}` — ~50-100 tokens en vez de miles. En tests: 90 tokens vs 10.195 del archivo.
- Archivo modificado → **solo el diff unificado** contra el snapshot (99%+ de ahorro medido).
- `force_full=true` saltea la memoria cuando necesitás el contenido completo.

### `router_checkpoint` — Handoff de contexto entre sesiones y clientes

```
checkpoint(action="save", name="refactor-auth",
           summary="Migrado el login a JWT, falta el refresh token",
           decisions=["usar RS256"], open_items=["tests de expiración"],
           files=["src/auth.py"])
```

Al cerrar una tarea larga (o cuando el contexto se llena), Claude guarda el estado como JSON legible en `checkpoints/` — editable por vos, compartible con tu equipo. Una sesión nueva —en Desktop, Code o Cowork, da igual— hace `action="resume"` y recupera todo en ~300 tokens, **incluyendo qué archivos cambiaron en disco desde el checkpoint** (comparación por hash, sin re-leerlos). `action="list"` muestra los checkpoints disponibles.

### `router_inbox` — Órdenes cruzadas entre clientes

```
# En Cowork:
inbox(action="send", to="code", message="migrá los tests a pytest",
      checkpoint="refactor-auth")

# En Claude Code, al arrancar:
inbox(action="check", to="code")
→ la orden + el resumen del checkpoint vinculado
inbox(action="complete", order_id=1, result="34/34 tests verdes")

# De vuelta en Cowork:
inbox(action="history")  → ves el resultado
```

Los chats de Claude no pueden comandarse entre sí en tiempo real — pero comparten este disco. El inbox es el buzón asíncrono: dejás una orden desde un cliente (vinculada a un checkpoint para que el receptor tenga todo el contexto de lo que estaban haciendo), el otro la consume al arrancar, la ejecuta y reporta el resultado. Decile a Cowork *"dejale esta tarea a Claude Code"* y listo — las `instructions` del servidor hacen que cada cliente chequee su buzón al empezar a trabajar.

**Clientes del pack:** `cowork`, `code`, `desktop` y `design` (Claude Design). Cualquiera puede darle órdenes a cualquiera — el destino es texto libre, así que también podés inventar roles propios. Para que un cliente reciba órdenes solo necesita tener este MCP cargado y chequear su buzón con `action="check"`.

**Handoff código ↔ diseño (`assets`):** las órdenes y los resultados pueden llevar `assets` — una lista de rutas de archivos o URLs (brief, wireframe, export `.fig`/`.png`, specs). Así el ida y vuelta entre desarrollo y diseño viaja con el material, no solo el texto:

```
# Claude Code le pide un mockup a Claude Design, con el material:
inbox(action="send", to="design", from_client="code",
      message="hero de la landing, dark + acento cyan",
      checkpoint="landing-v2",
      assets=["/proj/brief.md", "https://.../wireframe.png"])

# Claude Design, al arrancar, ve la orden + el brief + el wireframe:
inbox(action="check", to="design")
inbox(action="complete", order_id=7, result="mockup listo, 2 variantes",
      assets=["https://figma.com/.../hero", "/exports/hero-v1.png"])

# Claude Code ve el resultado y el export devuelto:
inbox(action="history")  → result + result_assets
```

Y al revés: Claude Design puede dejarle a `code` una orden con el export final para que lo implemente. El inbox es bidireccional entre todos los clientes.

### `router_status` — Métricas honestas de la sesión

La métrica principal es `tokens_kept_out_of_context`: tokens de las fuentes originales que **no** entraron a la ventana de Claude. También reporta lecturas, hits de memoria (unchanged/diff), estado del ledger y órdenes pendientes en el inbox. Con `deep=true` suma diagnóstico local (fastembed/python).

## Ahorro real (números honestos)

| Escenario | Ahorro | Fidelidad |
|---|---|---|
| Re-leer un archivo ya conocido, sin cambios | 98-99% | 100% (outline + memoria) |
| Re-leer un archivo conocido que cambió | 90-99% | 100% (diff exacto) |
| Retomar una tarea en sesión/cliente nuevo (resume) | ~300 tokens vs re-explorar todo | 100% |
| Buscar algo puntual en un archivo de 20k tokens | 80-95% | 100% (fragmentos exactos) |
| Ingesta de HTML/web sucio | 40-60% | 100% (solo se quita ruido) |
| Chat conversacional normal | ~0% | — |

El overhead fijo es ~500 tokens de definiciones de tools por sesión. Si no procesás archivos grandes ni lotes, ese es tu costo neto: sé honesto con vos mismo sobre tu caso de uso.

## Instalación

```bash
git clone <repo> && cd individra-mcp-router
pip install -r requirements.txt
# Opcional (ranking semántico local para smart_read): pip install fastembed
```

No necesita ninguna API key ni archivo `.env`: las cuatro herramientas son 100% locales.

### Registrarlo en todas tus sesiones de Claude

**Claude Desktop / Cowork** — `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "individra-router": {
      "command": "python",
      "args": ["C:/ruta/a/individra-mcp-router/server.py"]
    }
  }
}
```

**Claude Code** (alcance usuario = disponible en todos los proyectos):

```bash
claude mcp add individra-router --scope user -- python /ruta/a/server.py
```

### Activación automática

El servidor declara `instructions` que se inyectan en el system prompt de cada sesión, indicándole a Claude cuándo usar cada tool sin que se lo pidas. Para reforzarlo, agregá una línea a tu `CLAUDE.md` global:

```
Para leer archivos >15KB o buscar algo puntual en ellos usá router_smart_read con query.
Al cerrar tareas largas guardá un router_checkpoint; al retomar, resumílo.
```

## Arquitectura

```
smart_read ──▶ sanitizer (HTML/texto, local) ──▶ ledger (¿ya lo vi?)
                    │                              ├─ sin cambios → outline (~50 tok)
                    │                              └─ cambió → diff unificado
                    └──▶ ranker (con query)
                          ├─ fastembed (bge-small, ONNX local) si está
                          └─ BM25 puro-Python (fallback, 0 deps)
checkpoint ──▶ checkpoints/*.json (legible/editable) + verificación de hashes al resumir
inbox ──────▶ cola SQLite compartida (órdenes entre Cowork/Code/Desktop/Design + assets de handoff + resultados)
```

Todo local: el ledger y el inbox son SQLite en disco; los checkpoints son JSON legibles/editables. Sin red, sin API keys, sin dependencias externas para el núcleo.

## Limitaciones conocidas

- BM25 es léxico: una query sin palabras en común con el texto degrada a los primeros chunks del archivo. Instalar `fastembed` mejora las queries semánticas (requiere onnxruntime compatible con tu versión de Python).
- No hay memoria semántica de "decisiones de proyecto" todavía (ej: "nunca usar Redux"): el checkpoint captura estado de tarea, no reglas permanentes. Es una extensión natural pendiente.

## Licencia

MIT — ver `LICENSE`.
