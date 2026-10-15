"""Página inicial: portfólio de Engenharia de ML / MLOps."""
import _bootstrap  # noqa: F401
import streamlit as st

from shared import ui

ui.setup_page("ML Engineering · Portfólio", "🧩")
st.title("Engenharia de ML e MLOps")
st.markdown(
    "**Pipelines reprodutíveis de ML, do dado ao serving e ao monitoramento.** Quatro projetos com treino "
    "versionado (MLflow), feature store (Feast), monitoramento de drift (Evidently) e um assistente de "
    "recuperação com citações sobre leis brasileiras."
)
st.info(
    "**Como o repositório funciona.** O pipeline (dados, treino, exportação) roda localmente, com dependências "
    "pesadas em venvs separados. Esta vitrine só lê artefatos pequenos versionados em `projects/*/data` e "
    "`projects/*/artifacts`: modelos, métricas, tabelas Parquet, relatórios HTML e JSON. Nada é chamado em tempo real "
    "por HTTP, exceto o assistente de RAG, quando há uma chave de API configurada."
)
ui.project_grid([
    {"icon": "💳", "title": "Risco de crédito com MLOps", "summary": "Probabilidade de inadimplência com MLflow, "
     "API FastAPI, model card e explicação por cliente.", "stack": "LightGBM · MLflow · FastAPI · Docker",
     "page": "pages/1_Credit_Risk.py", "data_kind": "real", "dataset": "UCI Default of Credit Card Clients"},
    {"icon": "🚕", "title": "Monitoramento de drift de taxi", "summary": "Drift de dados, de predição e de desempenho "
     "ao longo de 6 meses de corridas de táxi em NYC.", "stack": "LightGBM · Evidently · PSI · DuckDB",
     "page": "pages/2_Taxi_Drift.py", "data_kind": "real", "dataset": "NYC TLC Yellow Taxi 2024"},
    {"icon": "🛒", "title": "Feature store", "summary": "Features de cliente com point-in-time correto, "
     "vazamento quantificado e lookup online.", "stack": "Feast · SQLite · pandas",
     "page": "pages/3_Feature_Store.py", "data_kind": "synthetic", "dataset": "gerado com schema Instacart"},
    {"icon": "⚖️", "title": "Assistente de leis (RAG)", "summary": "Perguntas sobre leis brasileiras com trechos "
     "citados e avaliação de recuperação.", "stack": "BM25 · Claude API (opcional) · Planalto",
     "page": "pages/4_RAG_Assistant.py", "data_kind": "real", "dataset": "textos oficiais do Planalto"},
])
st.caption("Código, dados e métricas: ver o README de cada projeto e o README raiz.")
