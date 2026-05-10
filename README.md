# INDIVIDRA MCP Router

> Reduce el consumo de tokens de Claude en 40-70% comprimiendo contexto con modelos gratuitos (Groq + OpenRouter) antes de enviarlo a Claude.

Funciona como un servidor [MCP](https://modelcontextprotocol.io) que se integra directamente con **Claude Desktop**. Sin proxies, sin intermediarios — todo corre en tu máquina.

---

## ¿Qué problema resuelve?

Claude tiene un contexto limitado y cada token cuesta. Cuando trabajás con documentos largos, wikis, especificaciones o logs, la mayoría del contenido es redundante. Este router comprime ese contenido automáticamente usando modelos gratuitos (Groq Llama, OpenRouter) antes de pasárselo a Claude.

**Resultado típico:** un documento de 5000 tokens se convierte en ~2500 tokens con la misma información técnica relevante.

---

## Arquitectura

```
Claude Desktop
    │
    ▼
MCP Router (local, Python)
    │
    ├─▶ Groq llama-3.1-8b-instant   ← primario (textos ≤ 3500 tokens, LPU ultrarrápido)
    ├─▶ OpenRouter free models       ← fallback (textos grandes o cuando Groq rate-limit)
    │     ├─ qwen/qwen-2.5-7b-instruct:free
    │     ├─ meta-llama/llama-3.2-3b-instruct:free
    │     └─ deepseek/deepseek-r1-distill-llama-70b:free
    └─▶ Pass-through                 ← último recurso (texto original sin modificar)
```

**Componentes internos:**
- **Tier detection** — código, JSON, YAML y stack traces nunca se comprimen (integridad crítica)
- **Caché SHA-256** — el mismo documento no se comprime dos veces (SQLite local, TTL 24h)
- **Circuit breaker** — si un proveedor falla 5 veces, se bypasea 5 minutos automáticamente
- **Intent classifier** — usa Groq para detectar si una tarea es de código antes de delegarla

---

## Requisitos

- Python 3.10+
- [Claude Desktop](https://claude.ai/download)
- API keys gratuitas (ver abajo)

---

## Instalación

### 1. Clonar el repo

```bash
git clone https://github.com/tu-usuario/individra-mcp-router.git
cd individra-mcp-router
```

### 2. Obtener API keys gratuitas

| Proveedor | Link | Límite free |
|---|---|---|
| **Groq** | https://console.groq.com/keys | 14,400 req/día, 6k TPM |
| **OpenRouter** | https://openrouter.ai/keys | Variable por modelo |

> Gemini es opcional. Si tenés una key válida, podés agregarla en `.env` y el router la intentará como tercer proveedor.

### 3. Ejecutar el instalador

```bash
python install.py
```

El instalador:
1. Instala dependencias Python (`pip install -r requirements.txt`)
2. Crea `.env` a partir de `.env.example`
3. Agrega `individra-router` a `claude_desktop_config.json` automáticamente
4. Verifica que `server.py` compile correctamente

### 4. Editar `.env` con tus claves

```env
GROQ_API_KEY=tu_clave_aqui
OPENROUTER_API_KEY=tu_clave_aqui
```

### 5. Reiniciar Claude Desktop

Cerrar completamente (desde la bandeja del sistema) y volver a abrir.

### 6. Verificar

Escribí en Claude:

```
router_diagnose()
```

Deberías ver `"status": "ok"` para Groq y OpenRouter.

---

## Herramientas disponibles en Claude

| Herramienta | Descripción |
|---|---|
| `router_compress_context` | Comprime texto largo antes de pasarlo a Claude |
| `router_route_task` | Delega una tarea completa a un modelo gratuito |
| `router_smart_read` | Lee un archivo con compresión automática según tipo y tamaño |
| `router_status` | Estado del sistema: circuit breakers, tokens ahorrados, caché |
| `router_diagnose` | Testea conectividad con cada proveedor y devuelve el error exacto |

---

## Lógica de tiers

| Tier | Tipo de contenido | Acción |
|---|---|---|
| 0 | Código, JSON, YAML, configs, stack traces | Sin compresión — integridad crítica |
| 1 | Texto < 2000 tokens | Sin compresión — ya es corto |
| 2 | Texto 2000–10000 tokens | Compresión media (~50%) |
| 3 | Texto > 10000 tokens | Compresión fuerte (~70%) |

---

## Troubleshooting

**El router no aparece en Claude**
→ Reiniciar Claude Desktop completamente (no solo la ventana).

**Error "Module not found"**
→ `pip install -r requirements.txt`

**Groq 429**
→ Normal en free tier si hacés muchas compresiones seguidas. El router hace retry automático y cae al fallback de OpenRouter. Se recupera solo en ~60 segundos.

**`router_diagnose()` muestra error en Gemini**
→ Algunos planes/regiones de Google AI Studio tienen cuota 0 en los modelos gratuitos. El router funciona perfectamente sin Gemini usando Groq + OpenRouter.

---

## Configuración manual (sin instalador)

Si preferís configurar manualmente, agregá esto a tu `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "individra-router": {
      "command": "python",
      "args": ["/ruta/absoluta/a/individra-mcp-router/server.py"],
      "env": {
        "GROQ_API_KEY": "tu_clave",
        "OPENROUTER_API_KEY": "tu_clave"
      }
    }
  }
}
```

Rutas del config por sistema operativo:
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

---

## Contribuciones

PRs bienvenidos. Si encontrás un modelo free de OpenRouter que funcione bien para compresión técnica, abrí un issue con el nombre del modelo y resultados de prueba.

---

Construido por [INDIVIDRA](https://individratec.com) — Automatización con IA para empresas B2B.
