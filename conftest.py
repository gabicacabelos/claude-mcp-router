import pathlib
import sys

# Garantiza que el paquete `router` sea importable al correr pytest desde
# cualquier directorio (la raíz del repo debe estar en sys.path).
sys.path.insert(0, str(pathlib.Path(__file__).parent))
