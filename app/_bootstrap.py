"""Coloca a raiz do repositório e os pacotes de projects/*/src no sys.path.

Importe no topo de Home.py e de cada página: `import _bootstrap  # noqa: F401`.
Os caminhos são os mesmos que o conftest.py dá aos testes, então uma página que importa o
pacote de um projeto se comporta igual no pytest e no Streamlit Cloud.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_paths = [str(ROOT), *(str(p) for p in sorted((ROOT / "projects").glob("*/src")))]
sys.path[:0] = [p for p in _paths if p not in sys.path]
