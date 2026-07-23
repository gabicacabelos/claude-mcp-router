# INDIVIDRA MCP — Context Ingestion & Bulk Offload Engine

## Qué es

Un servidor MCP (Model Context Protocol) que se conecta a cualquier cliente de Claude — Code, Desktop, Cowork, Claude Design — y les da algo que ninguno tiene de fábrica: memoria persistente en disco, compartida entre todos ellos. No es un modelo, no es un agente, no reemplaza a Claude: es infraestructura local que se sienta entre Claude y tus archivos para decidir qué entra realmente a la ventana de contexto.

Corre en tu máquina. Las funciones núcleo (lectura, memoria, checkpoints, coordinación entre clientes) son 100% locales, determinísticas y no requieren ninguna API key.

## El problema que resuelve

Claude es amnésico entre sesiones y ciego entre clientes. Si ayer le pediste a Claude Code que lea `auth.py`, hoy en una sesión nueva —o en Claude Desktop, o en Cowork— vuelve a leerlo entero. Mismo archivo, mismos tokens, otra vez. En un proyecto activo eso significa releer miles de tokens por día en contenido que no cambió una sola línea.

El segundo problema es la fragmentación: cada cliente de Claude trabaja aislado. Lo que se decidió en una sesión de Code no existe para la sesión de Cowork del mismo proyecto. No hay forma de dejarle una tarea a otro cliente y que la retome con contexto — cada handoff implica re-explicar todo desde cero.

El tercer problema es más simple: hay tareas repetitivas y de bajo riesgo (clasificar 50 emails, extraer campos de 30 facturas) que no necesitan el modelo más caro disponible, pero terminan pasando por Claude igual porque no hay una forma prolija de derivarlas.

## Qué ofrece — cinco herramientas

**`smart_read`** — lectura quirúrgica con memoria. Un archivo grande no entra entero al contexto: se lee con ranking local (BM25 o embeddings si están instalados) y devuelve solo los fragmentos exactos relevantes, con número de línea. La parte distintiva es la memoria cross-sesión: cada lectura queda registrada con su hash. La próxima vez que cualquier cliente pida ese mismo archivo, si no cambió devuelve un outline de ~90 tokens en vez de los 10.195 originales (99% menos, fidelidad 100%, porque no resume con un modelo — es determinista). Si cambió, devuelve solo el diff.

**`router_checkpoint`** — guarda el estado de una tarea larga (resumen, decisiones tomadas, pendientes, archivos involucrados) como JSON legible en disco. Cualquier sesión futura, en cualquier cliente, lo retoma con `action="resume"` en ~300 tokens — y de paso le informa qué archivos cambiaron en disco desde entonces, sin releerlos.

**`router_inbox`** — el buzón asíncrono entre clientes. Los chats de Claude no pueden comandarse en tiempo real entre sí, pero comparten este disco: un cliente deja una orden (opcionalmente con un checkpoint vinculado y con `assets` — rutas o URLs de material de apoyo), otro la consume al arrancar, la ejecuta y reporta el resultado. Cualquier cliente puede dejarle órdenes a cualquier otro, incluyendo el handoff código ↔ diseño: Claude Code le manda un brief y un wireframe a Claude Design, Design responde con el export del mockup.

**`router_bulk_process`** — offload de trabajo repetitivo (clasificación, extracción estructurada, resúmenes por lote) a modelos gratuitos externos (Groq, OpenRouter free) con failover automático y caché. Claude nunca ve los archivos originales; solo el JSON consolidado. Para código crítico o razonamiento complejo, esta herramienta se aparta a propósito — eso sigue siendo trabajo de Claude.

**`router_status`** — métricas honestas de cuántos tokens de las fuentes originales nunca entraron al contexto, y diagnóstico real de qué proveedores están vivos.

## La novedad para la comunidad

Hay muchas herramientas de "ahorro de contexto" que en el fondo son un LLM barato resumiendo tu archivo antes de que Claude lo vea — lo cual introduce pérdida y, en código, termina costando más en reintentos por errores de esa compresión. Este proyecto evita esa trampa a propósito: donde importa la fidelidad, `smart_read` es puramente determinista (hashing, diff, ranking léxico o por embeddings), nunca un resumen generado.

Pero el punto que no existe en ningún otro MCP público es la combinación de las tres piezas: **memoria que persiste entre sesiones**, **memoria que persiste entre clientes distintos de Claude**, y **un canal para que esos clientes se coordinen entre sí sin que el usuario tenga que oficiar de mensajero**. Hoy, si trabajás con Claude Code para programar y Claude Desktop o Cowork para gestión, son dos mundos que no se hablan. Este MCP los conecta con un disco compartido: un checkpoint guardado en Code aparece disponible en Cowork; una orden dejada en Cowork la ve Code al arrancar; y ahora, con la incorporación de Claude Design como cuarto cliente del pack, el mismo mecanismo sirve para el ida y vuelta entre desarrollo y diseño — brief y wireframe viajan como `assets` de la orden, el mockup exportado vuelve de la misma forma.

Es open source (MIT), sin costo para las funciones núcleo, y pensado para instalarse una vez y quedar activo en todas las sesiones de Claude que uses — no es una herramienta que se invoca a mano, sino infraestructura que Claude aprende a usar sola gracias a las `instructions` que el propio servidor inyecta.

## Números medidos

| Escenario | Ahorro | Fidelidad |
|---|---|---|
| Releer archivo sin cambios | 98–99% | 100% |
| Releer archivo modificado | 90–99% | 100% (diff exacto) |
| Retomar tarea en cliente nuevo (checkpoint resume) | ~300 tokens vs re-explorar todo | 100% |
| Búsqueda puntual en archivo de 20k tokens | 80–95% | 100% |
| Lote de 30 extracciones vía bulk_process | ~95%+ | la del modelo gratis usado |
| Chat conversacional normal | ~0% | — |

Repo: `github.com/gabicacabelos/claude-mcp-router`
