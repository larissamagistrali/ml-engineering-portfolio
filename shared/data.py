"""Acesso a dados do app (arquivo idêntico nos repos).

O Streamlit Cloud tem ~1 GB de memória: versionamos só amostras/agregados em Parquet e
consultamos com DuckDB. Bases grandes ficam fora do Git.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"


def project_dir(project: str) -> Path:
    return PROJECTS / project


def data_path(project: str, filename: str) -> Path:
    """Caminho de um arquivo em projects/<project>/data/."""
    return PROJECTS / project / "data" / filename


def use_project(project: str) -> None:
    """Põe projects/<project>/src no sys.path para importar o pacote do projeto.

    Convenção: o pacote é `projects/<nome-com-hifen>/src/<nome_com_underscore>/`, com nome
    único por projeto (evita colisão de módulos entre páginas).
    """
    src = str(PROJECTS / project / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def query(sql: str, tables: dict[str, str] | None = None, params: list | None = None) -> pd.DataFrame:
    """Executa SQL (DuckDB) sobre arquivos Parquet.

    `tables` mapeia nome-da-view -> caminho do Parquet, ex.::

        query("select segmento, count(*) n from clientes group by 1",
              {"clientes": str(data_path("customer-segmentation", "rfm.parquet"))})
    """
    con = duckdb.connect()
    try:
        for name, path in (tables or {}).items():
            if not name.isidentifier():
                raise ValueError(f"nome de view inválido: {name!r}")
            literal = str(path).replace("'", "''")  # CREATE VIEW não aceita parâmetros
            con.execute(f"create view {name} as select * from read_parquet('{literal}')")
        return con.execute(sql, params or []).df()
    finally:
        con.close()


@st.cache_resource(show_spinner=False)
def load_joblib(path: str) -> Any:
    """Carrega um modelo já treinado (cache por processo)."""
    import joblib

    return joblib.load(path)
