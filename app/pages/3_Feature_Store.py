"""Página: feature store de recompra com Feast. Lê só os exports (o Feast não é importado aqui)."""
import json

import _bootstrap  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared import data, theme, ui

PROJECT = "feature-store"
ui.setup_page("Feature store (Feast)", "🛒")
ui.header(
    "Feature store de recompra (Feast)",
    "Qual feature valia no momento da decisão? Quanto um join ingênuo vaza o futuro?",
    tags=["Feast 0.67", "point-in-time join", "SQLite online store", "pandas"],
    code_path=f"projects/{PROJECT}",
)
problema, dados, abordagem, resultado, demo = ui.tabs()
ROOT = data.project_dir(PROJECT) / "data"


@st.cache_data
def load_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / name)


registry = load_json("registry.json")
metrics = load_json("pit_vs_naive_metrics.json")
online = load_json("online_lookup.json")
examples = load_parquet("pit_examples.parquet")
FEATS = ["n_orders", "n_items_total", "avg_basket_size", "avg_days_between_orders", "reorder_ratio",
         "dept_purchases", "dept_share", "times_bought", "prev_purchase_gap_days"]

with problema:
    st.markdown(
        "Em modelos com histórico, a feature usada no treino precisa ser a que existia **no instante da decisão**. "
        "Se o join pega o valor mais recente de cada cliente, o modelo aprende com o futuro e parece melhor do que é. "
        "Esta página mostra o contrato das features e quantifica esse vazamento."
    )

with dados:
    ui.data_notice(
        "synthetic",
        "gerado por `feature_store_demo.synthetic`, com o schema do Instacart Market Basket Analysis",
        "O Instacart é Kaggle-only. 3.000 clientes, 27 mil pedidos e 122 mil itens sintéticos (semente 42). "
        "Os números ilustram o método de point-in-time, não comportamento real de consumidores.",
    )
    st.markdown("Alvo: o cliente compra este produto **no pedido seguinte**. As features são calculadas com os pedidos "
                "até o pedido atual. Os produtos avaliados são os já comprados antes pelo cliente.")

with abordagem:
    st.subheader("Catálogo de features")
    st.caption(f"Projeto Feast `{registry['project']}`. Entidades: " +
               ", ".join(f"{e['name']} ({e['join_key']})" for e in registry["entities"]))
    for v in registry["feature_views"]:
        with st.expander(f"{v['name']} · chave {', '.join(v['join_keys'])} · TTL {v['ttl_days']} dias"):
            st.markdown(v["description"] or "")
            st.caption(f"fonte: `{v['source_file']}` · timestamp: `{v['timestamp_field']}`")
            st.dataframe(pd.DataFrame(v["features"]), hide_index=True, width="stretch")
    for s in registry["feature_services"]:
        st.markdown(f"**FeatureService `{s['name']}`**: {s['description']}  \n"
                    f"Features: " + ", ".join(f"`{f}`" for f in s["features"]))

with resultado:
    ui.metric_row([
        ("ROC-AUC (point-in-time)", f"{metrics['auc']['point_in_time']['roc_auc_teste']:.3f}",
         "Teste nos 30% finais do tempo, treino nos 70% iniciais."),
        ("ROC-AUC (join ingênuo)", f"{metrics['auc']['ingenuo']['roc_auc_teste']:.3f}",
         "Mesmo modelo, mesmo teste, features do último estado."),
        ("Ganho falso (ROC-AUC)", f"+{metrics['auc_gap_roc']:.3f}", "Quanto o vazamento infla a métrica."),
    ])
    lk = metrics["leakage"]
    st.subheader("Quanto o join ingênuo vaza")
    st.markdown(
        f"- Pedidos do cliente: em média **{lk['mean_n_orders_pit']:.1f}** no momento da decisão, mas "
        f"**{lk['mean_n_orders_ingenuo']:.1f}** no join ingênuo (inclui pedidos futuros).\n"
        f"- Linhas com valor diferente: `n_orders` e `n_items_total` em 100%; `dept_purchases` em "
        f"{lk['share_rows_different_value']['dept_purchases']:.0%}; `times_bought` em "
        f"{lk['share_rows_different_value']['times_bought']:.0%}.\n"
        f"- PR-AUC: {metrics['auc']['point_in_time']['pr_auc_teste']:.3f} (ponto no tempo) contra "
        f"{metrics['auc']['ingenuo']['pr_auc_teste']:.3f} (ingênuo). A classe positiva é rara "
        f"({metrics['label_rate']:.1%})."
    )
    fig = go.Figure(go.Bar(
        x=["point-in-time", "ingênuo"],
        y=[metrics["auc"]["point_in_time"]["roc_auc_teste"], metrics["auc"]["ingenuo"]["roc_auc_teste"]],
        marker=dict(color=[theme.CATEGORICAL[0], theme.CATEGORICAL[1]]),
        text=[f"{metrics['auc']['point_in_time']['roc_auc_teste']:.3f}", f"{metrics['auc']['ingenuo']['roc_auc_teste']:.3f}"],
        textposition="outside",
    ))
    fig.add_hline(y=0.5, line=dict(dash="dot", color=theme.BASELINE))
    # Barra codifica comprimento a partir do zero; começar em 0,5 exageraria a diferença.
    fig.update_layout(title="ROC-AUC no teste temporal", yaxis=dict(range=[0, 1.05], title="ROC-AUC"), showlegend=False)
    ui.plot(fig)
    st.caption("A linha pontilhada marca um classificador aleatório (0,5).")

with demo:
    st.subheader("Point-in-time: cliente e momento → valores 'as of'")
    st.markdown("Escolha um cliente e um momento (pedido). A tabela mostra o valor que existia naquele instante e "
                "o valor que o join ingênuo entregaria.")
    cust = st.selectbox("Cliente", sorted(examples["customer_id"].unique().tolist()))
    sub = examples[examples["customer_id"] == cust]
    times = sorted(sub["event_timestamp"].unique())
    idx = st.select_slider("Momento da decisão", options=list(range(len(times))), value=len(times) // 2,
                           format_func=lambda i: pd.Timestamp(times[i]).strftime("%Y-%m-%d %H:%M"))
    at = sub[sub["event_timestamp"] == times[idx]]
    row = at.iloc[0]
    comp = pd.DataFrame({
        "feature": FEATS,
        "as of (point-in-time)": [row[f] for f in FEATS],
        "join ingênuo (último estado)": [row[f"{f}_ingenuo"] for f in FEATS],
    })
    comp["diferente"] = ["⚠️ sim" if (pd.notna(a) and pd.notna(b) and abs(a - b) > 1e-9) else "não"
                         for a, b in zip(comp["as of (point-in-time)"], comp["join ingênuo (último estado)"])]
    st.dataframe(comp.round(3), hide_index=True, width="stretch")
    st.caption(f"Linha: cliente {cust}, produto {int(row['product_id'])}, rótulo (compra no pedido seguinte) = {int(row['label'])}. "
               "⚠️ = o join ingênuo vazou um valor que ainda não existia.")

    st.subheader("Lookup online (SQLite, último valor materializado)")
    options = {f"cliente {e['entity']['customer_id']} · produto {e['entity']['product_id']}": e for e in online["examples"]}
    pick = st.selectbox("Entidade", list(options))
    st.caption(f"Materializado até {online['materialized_until'][:19]}. {online['note']}")
    st.json({"entidade": options[pick]["entity"], "features": options[pick]["features"]}, expanded=True)
