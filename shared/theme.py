"""Tema visual único dos três portfólios (arquivo idêntico nos repos).

Paleta categórica validada para daltonismo, em ordem fixa (nunca ciclar); uma rampa
sequencial (azul) e uma divergente (azul <-> vermelho com meio cinza). As cores de
status são reservadas: nunca usá-las como "série 4".
"""
from __future__ import annotations

from collections.abc import Iterable

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Ordem fixa: azul, laranja, água, amarelo, magenta, verde, violeta, vermelho.
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
# Magnitude contínua: um matiz só, do claro ao escuro.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Polaridade: dois matizes com meio cinza neutro.
DIVERGING = [[0.0, "#256abf"], [0.5, "#f0efec"], [1.0, "#e34948"]]
# Estado. Sempre acompanhar de ícone + texto, nunca só a cor.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
TEMPLATE_NAME = "portfolio"


def color_map(categories: Iterable[str]) -> dict[str, str]:
    """Cor por entidade, na ordem canônica recebida (a cor segue a entidade, não o ranking).

    Passe sempre a lista completa e estável de categorias, não a lista filtrada, para que
    um filtro não repinte as séries restantes. Da nona categoria em diante: cinza neutro
    (agrupe em "Outros").
    """
    cats = list(categories)
    return {
        c: CATEGORICAL[i] if i < len(CATEGORICAL) else INK_MUTED for i, c in enumerate(cats)
    }


def _build_template() -> go.layout.Template:
    axis = dict(
        automargin=True,  # a margem cresce para caber rótulos longos (a base é só 8px)
        linecolor=BASELINE,
        gridcolor=GRID,
        zeroline=False,
        ticks="",
        title=dict(font=dict(size=12, color=INK_SECONDARY)),
        tickfont=dict(size=12, color=INK_MUTED),
    )
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT, color=INK_SECONDARY, size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=CATEGORICAL,
            colorscale=dict(sequential=[[i / 6, c] for i, c in enumerate(SEQUENTIAL)], diverging=DIVERGING),
            xaxis={**axis, "showgrid": False},
            yaxis={**axis, "showgrid": True, "gridwidth": 1},
            # Legenda presa ao topo do contêiner: o Plotly abre a margem para quantas linhas ela tiver
            # (o título não fica na figura; ui.plot o renderiza como texto do Streamlit).
            legend=dict(orientation="h", yref="container", y=1, yanchor="top", xanchor="left", x=0, title_text=""),
            margin=dict(l=8, r=8, t=16, b=8),
            title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
            hoverlabel=dict(bgcolor=SURFACE, font=dict(family=FONT, color=INK, size=13), bordercolor=BASELINE),
            bargap=0.25,
        ),
        data=dict(
            # Marcas finas, linhas de 2px, 2px de respiro da cor da superfície entre preenchimentos.
            scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=2, color=SURFACE)))],
            bar=[go.Bar(marker=dict(line=dict(width=2, color=SURFACE)))],
        ),
    )


def apply_plotly_theme() -> None:
    """Registra o template 'portfolio' e o define como padrão do Plotly."""
    pio.templates[TEMPLATE_NAME] = _build_template()
    pio.templates.default = TEMPLATE_NAME


def style(fig: go.Figure) -> go.Figure:
    """Aplica o tema a uma figura (idempotente) e a devolve.

    Use via `ui.plot`, que tira o título da figura: título dentro do Plotly não quebra linha e
    disputaria o topo com a legenda.
    """
    if TEMPLATE_NAME not in pio.templates:
        apply_plotly_theme()
    # Sem título na legenda (o Plotly Express põe o nome da coluna): os itens já se explicam, e o
    # título na mesma linha impede a legenda horizontal de quebrar direito em telas estreitas.
    fig.update_layout(template=TEMPLATE_NAME, legend_title_text="")
    return fig

apply_plotly_theme()
