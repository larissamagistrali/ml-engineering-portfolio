"""Página: risco de crédito (PD) com versão do modelo, explicação por feature e contrato da API."""
import json

import _bootstrap  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared import data, theme, ui

PROJECT = "credit-risk-mlops"
ui.setup_page("Risco de crédito (PD)", "💳")
ui.header(
    "Risco de crédito (PD)",
    "Qual a probabilidade de inadimplência no mês seguinte deste cliente?",
    tags=["LightGBM", "MLflow", "FastAPI", "SHAP (TreeSHAP)", "model card"],
    code_path=f"projects/{PROJECT}",
)
problema, dados, abordagem, resultado, demo = ui.tabs()

data.use_project(PROJECT)
from credit_risk_mlops import config  # noqa: E402
from credit_risk_mlops.features import FEATURE_LABELS, FEATURE_NAMES  # noqa: E402
from credit_risk_mlops.serving import ModelLoadError, load_bundle, score_records  # noqa: E402


@st.cache_resource(show_spinner="Carregando o modelo campeão...")
def bundle_or_error():
    try:
        return load_bundle(config.ARTIFACTS_DIR), None
    except ModelLoadError as exc:
        return None, str(exc)


@st.cache_data
def metrics() -> dict:
    return data.read_json(str(data.data_path(PROJECT, config.METRICS_FILE)))


@st.cache_data
def card() -> dict:
    return json.loads((config.ARTIFACTS_DIR / config.CARD_FILE).read_text(encoding="utf-8"))


m, card_obj = metrics(), card()
best = m["candidates"][m["winner"]]

with problema:
    st.markdown(
        "Uma instituição precisa estimar, para cada cliente, a chance de atraso de pagamento no próximo mês. "
        "Esta página mostra um modelo treinado e registrado com MLflow, exposto por uma API FastAPI e "
        "explicado cliente a cliente."
    )

with dados:
    ui.data_notice(
        "real",
        "UCI Machine Learning Repository, *Default of Credit Card Clients* (Yeh & Lien, 2009), CC BY 4.0",
        "Substituto auth-free de Home Credit / Give Me Some Credit (Kaggle). 30.000 clientes de um emissor "
        "em Taiwan, abr-set/2005. Não é o mesmo dado da proposta original e tem 17 anos: os números valem para "
        "demonstrar o pipeline, não para crédito atual.",
    )
    st.markdown(f"Inadimplência observada: **{card_obj['data']['default_rate']:.1%}**. "
                f"Partição {card_obj['data']['split']['train']:,} / {card_obj['data']['split']['valid']:,} / "
                f"{card_obj['data']['split']['test']:,} (treino/validação/teste).".replace(",", "."))
    st.markdown("Sexo e estado civil **não** entram no modelo (atributos protegidos). Categorias não documentadas "
                "de escolaridade e estado civil foram agrupadas em 'outros'.")

with abordagem:
    st.markdown(
        "- **Baseline:** regressão logística com padronização.\n"
        "- **Candidato:** LightGBM com busca aleatória de hiperparâmetros (12 tentativas) e parada antecipada na validação.\n"
        "- **Gate de promoção:** o LightGBM só vira `champion` se superar o baseline em ROC-AUC de validação "
        f"(ganho mínimo de 0,005) e em log-loss.\n"
        "- **Registro:** MLflow Model Registry, alias `champion`. O artefato desta página é exportado do alias."
    )
    st.code(
        "PYTHONPATH=projects/credit-risk-mlops/src python -m credit_risk_mlops.data\n"
        "PYTHONPATH=projects/credit-risk-mlops/src python -m credit_risk_mlops.train",
        language="bash",
    )

with resultado:
    st.subheader("Métricas no teste (6.000 clientes, não usados para escolher o modelo)")
    ui.metric_row([
        ("ROC-AUC", f"{best['test']['roc_auc']:.3f}", "Capacidade de ordenar bons e maus pagadores (0,5 = aleatório)."),
        ("KS", f"{best['test']['ks']:.3f}", "Separação máxima entre as distribuições de bons e maus."),
        ("PR-AUC", f"{best['test']['pr_auc']:.3f}", "Área sob a curva precisão-revocação."),
        ("Brier", f"{best['test']['brier']:.3f}", "Erro quadrático da probabilidade (menor é melhor)."),
        ("ECE", f"{best['test']['ece']:.3f}", "Erro médio de calibração (PD prevista vs. observada)."),
    ])
    base = m["candidates"]["baseline"]["test"]
    comp = pd.DataFrame(
        {
            "métrica": ["ROC-AUC", "KS", "PR-AUC", "Brier", "log-loss", "ECE"],
            "baseline (logística)": [base["roc_auc"], base["ks"], base["pr_auc"], base["brier"], base["log_loss"], base["ece"]],
            "LightGBM": [best["test"]["roc_auc"], best["test"]["ks"], best["test"]["pr_auc"], best["test"]["brier"],
                         best["test"]["log_loss"], best["test"]["ece"]],
        }
    )
    st.dataframe(comp.round(4), hide_index=True, width="stretch")
    st.caption(f"Campeão: {card_obj['model_version']} ({card_obj['algorithm']}). Números de `data/metrics.json`.")

    cal = pd.DataFrame(m["calibration_test"][m["winner"]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="calibração perfeita",
                             line=dict(color=theme.BASELINE, dash="dot")))
    fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["observed"], mode="lines+markers", name="LightGBM (decis)",
                             line=dict(color=theme.CATEGORICAL[0])))
    fig.update_layout(title="Calibração no teste: PD prevista x taxa observada", xaxis_title="PD prevista",
                      yaxis_title="Taxa observada")
    ui.plot(fig)

    st.subheader("Model card (resumo)")
    st.json({k: card_obj[k] for k in ("model_version", "algorithm", "trained_at", "registry", "selection")}, expanded=False)
    st.markdown("**Limitações declaradas:**")
    for item in card_obj["limitations"]:
        st.markdown(f"- {item}")

with demo:
    st.markdown("Ajuste o perfil do cliente. A PD e a explicação são calculadas pelo modelo na hora.")
    left, right = st.columns(2)
    with left:
        limit = st.slider("Limite de crédito (NT$)", 10_000, 500_000, 120_000, step=10_000)
        age = st.slider("Idade", 21, 75, 34)
        edu = st.selectbox("Escolaridade", [1, 2, 3, 4], index=1,
                           format_func=lambda v: {1: "Pós-graduação", 2: "Universidade", 3: "Ensino médio", 4: "Outros"}[v])
        bill = st.slider("Fatura do último mês (NT$)", 0, 300_000, 41_000, step=1_000)
        pay_last = st.number_input("Pagamento do último mês (NT$)", 0, 300_000, 2_500, step=500)
    with right:
        status = st.select_slider(
            "Situação de pagamento do último mês",
            options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            value=0,
            format_func=lambda v: {-2: "sem consumo", -1: "em dia", 0: "uso rotativo"}.get(v, f"{v} mês(es) de atraso"),
        )
        months_delay = st.slider("Meses com atraso nos últimos 6 (aprox.)", 0, 6, 0)

    # Histórico simples: mantém o perfil informado e completa os 5 meses anteriores com o mesmo padrão.
    pay_status = [status] + [(status if i < months_delay else -1) for i in range(5)]
    bill_amt = [bill] * 6
    pay_amt = [pay_last] + [pay_last] * 5
    record = {"limit_bal": limit, "age": age, "education": edu, "pay_status": pay_status,
              "bill_amt": bill_amt, "pay_amt": pay_amt}

    bundle, load_error = bundle_or_error()
    if bundle is None:
        st.error(f"Modelo indisponível: {load_error}. Rode `make train-credit-risk`.")
    else:
        item = score_records(bundle, [record], explain=True, top_k=8)[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("PD estimada", f"{item['pd']:.1%}")
        c2.metric("Faixa de risco", f"{item['risk_band']} · {item['risk_band_label']}")
        c3.metric("Versão do modelo", bundle.version)
        st.caption("Limiares das faixas: A < 10%, B < 20%, C < 35%, D < 55%, E ≥ 55%.")
        for w in item["warnings"]:
            st.warning(f"⚠️ {w}")

        st.subheader("Por que este resultado (contribuição de cada feature, em log-odds)")
        f = pd.DataFrame(item["top_factors"])
        f["cor"] = f["contribution_logit"].map(lambda v: theme.STATUS["critical"] if v > 0 else theme.CATEGORICAL[0])
        fig = go.Figure(go.Bar(
            x=f["contribution_logit"], y=f["label"], orientation="h",
            marker=dict(color=f["cor"]), text=f["direction"].map({"increases_risk": "aumenta risco", "decreases_risk": "reduz risco"}),
            textposition="auto", name="contribuição",
        ))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Contribuição para o log-odds (0 = valor-base)",
                          showlegend=False)
        ui.plot(fig)
        st.caption("Explicação exata do LightGBM (TreeSHAP nativo do LightGBM, `pred_contrib`). Não é causal.")

    st.subheader("Contrato da API")
    req = {"limit_bal": limit, "age": age, "education": edu, "pay_status": pay_status, "bill_amt": bill_amt,
           "pay_amt": pay_amt}
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Request** `POST /predict`")
        st.json(req, expanded=True)
    with col_b:
        st.markdown("**Response** (resumida)")
        if bundle is not None:
            sample = {"model_version": bundle.version,
                      "prediction": {k: item[k] for k in ("pd", "risk_band", "risk_band_label", "warnings")}}
            st.json(sample, expanded=True)
    st.caption("Endpoints: `/health`, `/model-info`, `/predict`, `/predict/batch`. Ver o README do projeto.")
