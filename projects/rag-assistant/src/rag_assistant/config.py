"""Corpus: leis do Planalto (textos oficiais), caminhos e parâmetros."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
# Cache do HTML bruto (gitignored). Dado não confiável: só é lido como texto, nunca executado.
RAW_DIR = Path(os.environ.get("RAG_RAW_DIR", PROJECT_DIR / "data" / "raw" / "planalto"))

LAWS = [
    {"id": "lgpd", "name": "Lei Geral de Proteção de Dados", "short": "LGPD", "number": "13.709/2018",
     "url": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm", "file": "l13709.htm"},
    {"id": "marco-civil", "name": "Marco Civil da Internet", "short": "Marco Civil", "number": "12.965/2014",
     "url": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2014/lei/l12965.htm", "file": "l12965.htm"},
    {"id": "lai", "name": "Lei de Acesso à Informação", "short": "LAI", "number": "12.527/2011",
     "url": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12527.htm", "file": "l12527.htm"},
    {"id": "licitacoes", "name": "Nova Lei de Licitações e Contratos", "short": "Lei 14.133", "number": "14.133/2021",
     "url": "https://www.planalto.gov.br/ccivil_03/_ato2019-2022/2021/lei/L14133.htm", "file": "L14133.htm"},
    {"id": "cdc", "name": "Código de Defesa do Consumidor (texto compilado)", "short": "CDC", "number": "8.078/1990",
     "url": "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm", "file": "l8078compilado.htm"},
]
