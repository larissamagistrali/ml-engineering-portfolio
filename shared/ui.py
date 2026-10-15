"""Estrutura padrão de página (arquivo idêntico nos repos).

Toda página de projeto segue o mesmo roteiro, em abas:
Problema · Dados · Abordagem · Resultado · Demo interativa.

Uso típico em app/pages/1_Projeto.py::

    import _bootstrap  # noqa: F401
    from shared import ui

    ui.setup_page("Churn", "📉")
    ui.header("Churn", "Quem vai cancelar?", tags=["scikit-learn", "SHAP"])
    problema, dados, abordagem, resultado, demo = ui.tabs()
    with problema:
        st.markdown("...")
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import plotly.graph_objects as go
import streamlit as st

from . import theme

_TAGS = re.compile(r"<[^>]+>")
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|$~])")

SECTIONS = ("Problema", "Dados", "Abordagem", "Resultado", "Demo interativa")
DataKind = Literal["real", "synthetic", "mixed"]

_NOTICE = {
    "real": ("✅", "Dados reais"),
    "synthetic": ("⚠️", "Dados sintéticos (demonstração)"),
    "mixed": ("⚠️", "Dados parcialmente sintéticos"),
}


def setup_page(title: str, icon: str, layout: Literal["wide", "centered"] = "wide") -> None:
    """Deve ser a primeira chamada Streamlit da página."""
    st.set_page_config(page_title=title, page_icon=icon, layout=layout)
    theme.apply_plotly_theme()


def header(title: str, subtitle: str, tags: Iterable[str] = (), code_path: str | None = None) -> None:
    st.title(title)
    st.markdown(f"**{subtitle}**")
    meta = list(tags)
    if code_path:
        meta.append(f"código em `{code_path}`")
    if meta:
        st.caption(" · ".join(meta))


def tabs() -> tuple:
    """As cinco abas padrão, na ordem de SECTIONS."""
    return st.tabs(list(SECTIONS))


def data_notice(kind: DataKind, source: str, detail: str = "") -> None:
    """Aviso honesto sobre a origem dos dados (ícone + texto, nunca só cor)."""
    icon, label = _NOTICE[kind]
    msg = f"{icon} **{label}** · fonte: {source}"
    if detail:
        msg += f"\n\n{detail}"
    (st.success if kind == "real" else st.warning)(msg)


def metric_row(items: Sequence[tuple[str, str] | tuple[str, str, str]]) -> None:
    """Linha de métricas: (rótulo, valor) ou (rótulo, valor, ajuda)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, *rest = item
        col.metric(label, value, help=rest[0] if rest else None)


def plot(fig: go.Figure, key: str | None = None) -> None:
    """Renderiza um gráfico Plotly com o tema do portfólio (theme=None evita o override do Streamlit).

    O título sai da figura e vira texto do Streamlit, que quebra linha em qualquer largura;
    no Plotly ele seria cortado no celular e disputaria espaço com a legenda.
    """
    title = _TAGS.sub("", re.sub(r"<br\s*/?>", " ", fig.layout.title.text or "")).strip()
    if title:
        escaped = _MD_SPECIAL.sub(r"\\\1", title)  # "R$" não pode virar LaTeX, "*" não vira itálico
        st.markdown(f"**{escaped}**")
        fig.update_layout(title_text=None)
    st.plotly_chart(theme.style(fig), theme=None, key=key)


def project_grid(projects: Sequence[Mapping[str, str]], columns: int = 2) -> None:
    """Cartões de projeto para a Home.

    Cada item: title, summary, stack, page (ex.: "pages/1_Churn.py"), icon, data_kind
    ("real" | "synthetic" | "mixed") e, opcionalmente, dataset.
    """
    cols = st.columns(columns)
    for i, p in enumerate(projects):
        with cols[i % columns].container(border=True):
            st.subheader(f"{p.get('icon', '')} {p['title']}".strip())
            st.write(p["summary"])
            icon, label = _NOTICE[p.get("data_kind", "synthetic")]  # type: ignore[index]
            line = f"{icon} {label}"
            if p.get("dataset"):
                line += f" · {p['dataset']}"
            st.caption(line)
            st.caption(f"Stack: {p['stack']}")
            st.page_link(p["page"], label="Abrir projeto →")
