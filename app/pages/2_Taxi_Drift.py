"""Página: monitoramento de drift de dados, predição e desempenho (NYC TLC, 2024-01 a 2024-06)."""
import json

import _bootstrap  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared import data, theme, ui

PROJECT = "taxi-drift-monitoring"
ui.setup_page("Drift em corridas de táxi", "🚕")
ui.header(
    "Drift em corridas de táxi",
    "O modelo de duração continua confiável seis meses depois? Onde e quando ele piora?",
    tags=["LightGBM", "PSI", "KS", "Evidently", "DuckDB"],
    code_path=f"projects/{PROJECT}",
)
problema, dados, abordagem, resultado, demo = ui.tabs()

ROOT = data.project_dir(PROJECT) / "data"


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / name)


summary = load_json("metrics.json")
perf = load_parquet("performance_by_month.parquet")
drift = load_parquet("drift_by_month.parquet")
sim = load_json("simulation_metrics.json")
months = [m["month"] for m in summary["months"]]
ALL_MONTHS = [summary["reference_month"], *months[1:]]
config_psi_moderate = summary["thresholds"]["psi_moderate"]
MONTH_COLORS = theme.color_map(ALL_MONTHS)

with problema:
    st.markdown(
        "Modelos de previsão degradam quando o mundo muda. Aqui um modelo de duração de corridas é treinado com "
        "janeiro de 2024 e pontuado de fevereiro a junho. Cada mês é comparado à referência em três níveis: "
        "**dados** (as entradas mudaram?), **predição** (a saída mudou?) e **desempenho** (o erro subiu?)."
    )

with dados:
    ui.data_notice(
        "real",
        "NYC TLC, Yellow Taxi Trip Records, 2024-01 a 2024-06 (domínio público, dados abertos da cidade de Nova York)",
        "Amostra de 50.000 corridas por mês após limpeza básica (semente 42). Duração entre 1 e 120 min, distância "
        "entre 0,1 e 60 milhas, valores de passageiro e pagamento válidos. Zonas mapeadas para o distrito (borough).",
    )
    st.markdown(
        f"Referência: **{summary['reference_month']}**. Alvo: duração da corrida (min). Features: distância, "
        "passageiros, hora, dia da semana, distrito de embarque e desembarque, tipo de pagamento. "
        "Nenhuma feature usa o valor da tarifa, que só existe depois da corrida."
    )

with abordagem:
    st.markdown(
        "- **Modelo:** LightGBM treinado só com 80% do mês de janeiro (holdout de 20% para o MAE de referência).\n"
        "- **PSI** por feature (faixas de quantil da referência) e **KS** nas numéricas. PSI ≥ 0,10 moderado, ≥ 0,20 alto.\n"
        "- **Desempenho:** MAE em minutos e MAPE em corridas com pelo menos 3 min. Alerta se o MAE passar de 110% do de referência.\n"
        "- **Relatório Evidently** (drift de dados, predição e regressão) gerado por `make report-taxi`."
    )

with resultado:
    ref_mae = summary["reference_mae"]
    last = perf.iloc[-1]
    ui.metric_row([
        ("MAE de referência", f"{ref_mae:.2f} min", "Erro absoluto médio no holdout de janeiro."),
        ("MAE em junho", f"{last['mae']:.2f} min", f"{last['mae_ratio']:.0%} da referência."),
        ("Maior PSI de feature (junho)", f"{drift[drift['month'] == '2024-06']['psi'].max():.3f}",
         "Entradas quase estáveis: o problema não é a distribuição das features."),
    ])

    st.subheader("Alertas por mês")
    rows = []
    for _, r in perf.iterrows():
        dados_ok = not r.get("alert_data", False)
        perf_ok = not r.get("alert_performance", False)
        rows.append({
            "mês": r["month"],
            "corridas": int(r["n"]),
            "MAE (min)": round(r["mae"], 3),
            "MAE / referência": round(r["mae_ratio"], 3),
            "maior PSI (feature)": round(r["max_feature_psi"], 3),
            "PSI da predição": round(r["psi_prediction"], 4),
            "dados": "🟢 sem alerta" if dados_ok else "🟠 alerta",
            "desempenho": "🟢 sem alerta" if perf_ok else "🔴 MAE acima do limiar",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Alerta de dados: alguma feature com PSI ≥ 0,20. Alerta de desempenho: MAE > 110% da referência.")

    mae_fig = go.Figure()
    mae_fig.add_hline(y=ref_mae, line=dict(color=theme.BASELINE, dash="dot"),
                      annotation_text="referência", annotation_position="top left")
    mae_fig.add_trace(go.Scatter(x=perf["month"], y=perf["mae"], mode="lines+markers", name="MAE",
                                 line=dict(color=theme.CATEGORICAL[0], width=2)))
    mae_fig.update_layout(title="MAE por mês (minutos)", yaxis_title="MAE (min)", xaxis_title="mês", showlegend=False)
    ui.plot(mae_fig)

    st.subheader("PSI por feature e mês")
    pivot = drift.pivot(index="feature", columns="month", values="psi").reindex(columns=months[1:])
    heat = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=[[i / 6, c] for i, c in enumerate(theme.SEQUENTIAL)],
        zmin=0, zmax=0.2, colorbar=dict(title="PSI"), text=[[f"{v:.3f}" for v in row] for row in pivot.values],
        texttemplate="%{text}", hovertemplate="%{y} · %{x}: PSI %{z:.3f}<extra></extra>",
    ))
    heat.update_layout(title="PSI (escala até 0,20; acima disso é alerta alto)", yaxis_title="", xaxis_title="mês")
    ui.plot(heat)
    st.caption("Um matiz só: quanto mais escuro, maior o deslocamento. Limiares: 0,10 moderado, 0,20 alto.")

    st.subheader("Relatório Evidently (referência x mês escolhido)")
    reports = sorted((data.project_dir(PROJECT) / "data" / "evidently").glob("*.html"))
    if reports:
        chosen = st.selectbox("Relatório", [p.name for p in reports])
        report = data.project_dir(PROJECT) / "data" / "evidently" / chosen  # gerado pelo nosso pipeline, conteúdo confiável
        if hasattr(st, "iframe"):
            st.iframe(report, height=900)
        else:  # Streamlit antigo: components.v1.html foi descontinuado nas versões novas
            import streamlit.components.v1 as components

            components.html(report.read_text(encoding="utf-8"), height=900, scrolling=True)
    else:
        st.info("Relatório não gerado. Rode `make report-taxi`.")

    st.subheader("Política de retreino sugerida")
    st.markdown(
        "- **Não retreinar só por PSI.** Neste período o PSI ficou estável e o erro subiu: retreinar com dados "
        "novos é o que resolve a mudança de relação, não ajustar a distribuição.\n"
        "- **Gatilho:** MAE acima de 110% da referência por **2 meses seguidos** (confirmado, não ruído de um mês).\n"
        "- **Gatilho secundário:** PSI ≥ 0,20 em 3 ou mais features ao mesmo tempo, ou PSI da predição ≥ 0,20.\n"
        "- **Ação:** retreinar com a janela mais recente (ex.: últimos 6 meses), validar no mês seguinte e só promover "
        "se o MAE voltar ao patamar de referência. Registrar a versão no MLflow, como no projeto de crédito.\n"
        "- **Verificar rótulos:** o alvo chega com atraso (a duração é conhecida ao final). O monitor usa o erro real "
        "só depois que a corrida termina."
    )

with demo:
    st.markdown("Ajuste os limiares de alerta. Os alertas são recalculados sobre as tabelas do pipeline.")
    c1, c2, c3 = st.columns(3)
    psi_high = c1.slider("Limiar de PSI alto", 0.05, 0.50, float(summary["thresholds"]["psi_high"]), 0.01)
    mae_ratio = c2.slider("Limiar de MAE (x referência)", 1.00, 1.50, float(summary["thresholds"]["mae_ratio_alert"]), 0.01)
    persist = c3.slider("Meses seguidos para retreinar", 1, 3, 2)

    flags = (perf["mae_ratio"] > mae_ratio).tolist()
    streak, trigger = 0, None
    for m, f in zip(perf["month"], flags):
        streak = streak + 1 if f else 0
        if streak >= persist and trigger is None:
            trigger = m
    alt = perf[["month", "mae_ratio", "max_feature_psi"]].copy()
    alt["alerta de desempenho"] = [("🔴 sim" if f else "🟢 não") for f in flags]
    alt["alerta de dados"] = [("🟠 sim" if v >= psi_high else "🟢 não") for v in alt["max_feature_psi"]]
    st.dataframe(alt.round(3), hide_index=True, width="stretch")
    if trigger:
        st.error(f"🔴 Retreino recomendado a partir de **{trigger}** (MAE acima do limiar por {persist} mês(es) seguidos).")
    else:
        st.success("🟢 Nenhum gatilho de retreino com os limiares escolhidos.")

    st.subheader("Simulação (dado sintético, não é o dado real)")
    st.warning(f"⚠️ **{sim['label']}**")
    sim_df = pd.DataFrame(sim["psi_by_feature"])
    sim_df["classe"] = sim_df["psi"].map(lambda v: "🟠 moderado" if v >= config_psi_moderate else "🟢 estável")
    st.dataframe(sim_df, hide_index=True, width="stretch")
    st.markdown(
        f"Com 30% das corridas perturbadas, o PSI de cada feature fica abaixo do limiar de alerta, mas o MAE sobe de "
        f"**{sim['mae_real']:.2f}** para **{sim['mae_with_perturbed_inputs']:.2f}** min. Ou seja: uma perturbação "
        "parcial pode passar despercebida pelo PSI e aparecer só no erro."
    )
