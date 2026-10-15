# Engenharia de ML e MLOps — portfólio

Quatro projetos de engenharia de ML: treino com rastreamento e registro de modelo, API de inferência com model card,
monitoramento de drift em produção, feature store com point-in-time correto e um assistente de leis com citações.
A vitrine é um app Streamlit multipágina em `app/`.

**App no ar:** https://SEU-APP.streamlit.app (TODO: preencher após o deploy)

| Projeto | Pergunta que responde | Dados | Stack | Status |
|---|---|---|---|---|
| [credit-risk-mlops](projects/credit-risk-mlops/README.md) | Qual a probabilidade de inadimplência deste cliente? | Real (UCI, substituto de Home Credit) | LightGBM, MLflow, FastAPI, Docker (não executado) | Pipeline, API e app prontos |
| [taxi-drift-monitoring](projects/taxi-drift-monitoring/README.md) | O modelo de duração continua confiável? Onde piora? | Real (NYC TLC, jan–jun/2024) | LightGBM, Evidently, PSI/KS | Pipeline, relatório e app prontos |
| [feature-store](projects/feature-store/README.md) | Qual feature valia no momento da decisão? | Sintético (schema Instacart) | Feast 0.67, SQLite | Feast aplicado, point-in-time verificado |
| [rag-assistant](projects/rag-assistant/README.md) | Qual artigo de lei responde a esta pergunta? | Real (textos do Planalto) | BM25 próprio, Claude API (opcional) | Corpus, avaliação e app prontos; LLM não exercitado com API real |

## Como rodar
```bash
make install      # venv do app (requirements.txt raiz + pytest)
make app          # http://localhost:8501
make test         # smoke test de todas as páginas + testes dos projetos
```
Os pipelines de dados e treino rodam **localmente**, cada um no seu venv (`make venv-<projeto>`):
`make data` (todos os dados) e `make train` (modelos). Os alvos por projeto estão em `make help`.
O app não chama as bibliotecas pesadas (MLflow, Feast, Evidently, FastAPI): lê só artefatos pequenos versionados
(modelo joblib, JSON, Parquet, HTML, índice BM25).

## Deploy no Streamlit Community Cloud
- Arquivo principal: `app/Home.py`. Python 3.11+.
- `requirements.txt` da raiz (sem MLflow, Feast, Evidently ou FastAPI, para caber em ~1 GB de memória).
- Segredos: `ANTHROPIC_API_KEY` (opcional, ativa o modo LLM do assistente) em *Secrets* no Cloud, ou em
  `.streamlit/secrets.toml` localmente (arquivo ignorado pelo Git).

## Estrutura
```
app/                 Home.py, pages/ (uma por projeto), _bootstrap.py
shared/              tema, estrutura de página, acesso a dados (comum aos portfólios)
projects/<nome>/     src/ (pacote Python), data/ (amostras e agregados pequenos), tests/, README.md
tests/               smoke test do app (AppTest em todas as páginas)
```

## Origem dos dados
- O Kaggle exige login, então **Home Credit** e **Give Me Some Credit** foram substituídos pelo
  UCI *Default of Credit Card Clients* (CC BY 4.0). **Instacart** é sintético com o schema original, rotulado como tal.
- NYC TLC e Planalto são dados públicos. Cada projeto cita fonte, licença e limitações no seu README.
- Os números das páginas vêm de execuções reais dos pipelines (arquivos em `data/`), não de valores digitados.
