"""Deixa os pacotes de projects/*/src importáveis nos testes (cada pacote tem nome único)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for src in sorted((ROOT / "projects").glob("*/src")):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
