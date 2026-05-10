#!/usr/bin/env python3
"""
INDIVIDRA MCP Router — Instalador automático

Qué hace:
  1. Instala dependencias Python
  2. Crea .env a partir de .env.example
  3. Configura claude_desktop_config.json automáticamente
  4. Verifica que server.py compila correctamente

Uso:
  python install.py
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BANNER = """
╔══════════════════════════════════════════════╗
║   INDIVIDRA MCP Router — Instalador v1.0.0   ║
║   Token saver para Claude Desktop             ║
╚══════════════════════════════════════════════╝
"""

PROJECT_DIR = Path(__file__).parent.resolve()
SERVER_SCRIPT = PROJECT_DIR / "server.py"


def print_step(step: str) -> None:
    print(f"\n→ {step}")

def ok(msg: str) -> None:
    print(f"   ✓ {msg}")

def err(msg: str) -> None:
    print(f"   ✗ {msg}")


def get_claude_config_path() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "Claude" / "claude_desktop_config.json"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def install_dependencies() -> bool:
    print_step("Instalando dependencias Python...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_DIR / "requirements.txt"), "--quiet"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        ok("Dependencias instaladas")
        return True
    else:
        err(f"Error:\n{result.stderr[:400]}")
        return False


def setup_env() -> bool:
    env_file = PROJECT_DIR / ".env"
    env_example = PROJECT_DIR / ".env.example"

    if env_file.exists():
        ok(".env ya existe — no se sobreescribe")
        return True

    if not env_example.exists():
        err(".env.example no encontrado")
        return False

    shutil.copy(env_example, env_file)
    ok(f".env creado")
    print(f"\n   ⚠️  ACCIÓN REQUERIDA:")
    print(f"   Editá el archivo: {env_file}")
    print(f"   Reemplazá GEMINI_API_KEY y GROQ_API_KEY con tus claves reales")
    return True


def configure_claude_desktop() -> bool:
    config_path = get_claude_config_path()
    print_step(f"Configurando Claude Desktop...")
    print(f"   Ruta: {config_path}")

    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Leer claves del .env
    env_file = PROJECT_DIR / ".env"
    env_vars = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip()

    # En Windows usar forward slashes para el path
    server_path = str(SERVER_SCRIPT).replace("\\", "/")

    config["mcpServers"]["individra-router"] = {
        "command": sys.executable,
        "args": [server_path],
        "env": {
            "GEMINI_API_KEY": env_vars.get("GEMINI_API_KEY", ""),
            "GROQ_API_KEY": env_vars.get("GROQ_API_KEY", ""),
            "OPENROUTER_API_KEY": env_vars.get("OPENROUTER_API_KEY", ""),
        }
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    ok('Servidor "individra-router" agregado a Claude Desktop')
    return True


def verify_syntax() -> bool:
    print_step("Verificando sintaxis del servidor...")
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(SERVER_SCRIPT)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        ok("server.py — sintaxis OK")
        return True
    else:
        err(f"Error de sintaxis:\n{result.stderr}")
        return False


def main():
    print(BANNER)
    print(f"Directorio del proyecto: {PROJECT_DIR}")

    steps = [
        ("Instalar dependencias", install_dependencies),
        ("Crear .env", setup_env),
        ("Configurar Claude Desktop", configure_claude_desktop),
        ("Verificar sintaxis", verify_syntax),
    ]

    success_count = 0
    for name, func in steps:
        try:
            if func():
                success_count += 1
        except Exception as e:
            err(f"Error en '{name}': {e}")

    print("\n" + "─" * 48)
    print(f"Resultado: {success_count}/{len(steps)} pasos OK")

    if success_count == len(steps):
        print("""
✓ Instalación completa

Próximos pasos:
  1. Editá .env con tus claves API reales
  2. Reiniciá Claude Desktop completamente
  3. Verificá escribiendo en Claude: router_status()
""")
    else:
        print("\n⚠  Instalación parcial — revisar errores arriba\n")


if __name__ == "__main__":
    main()
