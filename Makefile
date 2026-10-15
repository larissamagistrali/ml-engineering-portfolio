# Atalhos do repositório. `make help` lista os alvos.
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install app test data train venv-% data-% train-% api-% test-% report-taxi-drift
help:            ## Lista os alvos
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

install:         ## Cria o venv e instala as dependências do app
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt

app:             ## Sobe o app Streamlit
	$(BIN)/streamlit run app/Home.py

test:            ## Roda os testes (inclui o teste de fumaça do app)
	$(BIN)/pytest -q

# --- Alvos por projeto ---------------------------------------------------------
# Cada projeto de dados/treino usa o próprio venv em projects/<p>/.venv (`make venv-<p>`).
# Sem `make venv-<p>`, os alvos caem no python3 do sistema (não recomendado).
PROJ_CREDIT := projects/credit-risk-mlops
PROJ_TAXI   := projects/taxi-drift-monitoring
PROJ_FEAT   := projects/feature-store
PROJ_RAG    := projects/rag-assistant
PY_CREDIT   ?= $(PROJ_CREDIT)/.venv/bin/python
PY_TAXI     ?= $(PROJ_TAXI)/.venv/bin/python
PY_FEAT     ?= $(PROJ_FEAT)/.venv/bin/python
PY_RAG      ?= $(PROJ_RAG)/.venv/bin/python
API_PORT    ?= 8701

data: data-credit-risk data-taxi-drift data-feature-store data-rag   ## Gera todos os dados e artefatos de dados
train: train-credit-risk train-taxi-drift                             ## Treina/registra os modelos (MLflow e referência de drift)

venv-credit-risk:   ## Cria o venv do projeto de crédito (pipeline + MLflow)
	python3 -m venv $(PROJ_CREDIT)/.venv && $(PROJ_CREDIT)/.venv/bin/pip install -r $(PROJ_CREDIT)/requirements-pipeline.txt pytest
venv-taxi-drift:    ## Cria o venv do projeto de drift (LightGBM + Evidently)
	python3 -m venv $(PROJ_TAXI)/.venv && $(PROJ_TAXI)/.venv/bin/pip install -r $(PROJ_TAXI)/requirements.txt pytest
venv-feature-store: ## Cria o venv do projeto de feature store (Feast)
	python3 -m venv $(PROJ_FEAT)/.venv && $(PROJ_FEAT)/.venv/bin/pip install -r $(PROJ_FEAT)/requirements.txt pytest
venv-rag-assistant: ## Cria o venv do projeto RAG (pipeline do corpus)
	python3 -m venv $(PROJ_RAG)/.venv && $(PROJ_RAG)/.venv/bin/pip install -r $(PROJ_RAG)/requirements.txt pytest

# Crédito: dados (UCI) -> treino com MLflow -> exporta model.joblib + model_card.json
data-credit-risk:   ## Baixa e limpa o UCI Default of Credit Card Clients
	PYTHONPATH=$(PROJ_CREDIT)/src $(PY_CREDIT) -m credit_risk_mlops.data
train-credit-risk:  ## Treina baseline e LightGBM, registra no MLflow e exporta o campeão
	PYTHONPATH=$(PROJ_CREDIT)/src $(PY_CREDIT) -m credit_risk_mlops.train
api-credit-risk:    ## Sobe a API de PD (uvicorn) na porta 8701 (variável API_PORT)
	PYTHONPATH=$(PROJ_CREDIT)/src $(PY_CREDIT) -m uvicorn credit_risk_mlops.api:app --port $(API_PORT)
test-credit-risk:   ## Testes do projeto de crédito
	PYTHONPATH=$(PROJ_CREDIT)/src $(PY_CREDIT) -m pytest -q $(PROJ_CREDIT)/tests

# Drift de táxi: amostras TLC -> referência (jan) -> monitor (fev-jun) -> simulação -> relatório
data-taxi-drift:    ## Baixa os 6 meses de corridas (cache em data/raw) e gera as tabelas de drift
	PYTHONPATH=$(PROJ_TAXI)/src $(PY_TAXI) -m taxi_drift_monitoring.data
	PYTHONPATH=$(PROJ_TAXI)/src $(PY_TAXI) -m taxi_drift_monitoring.monitor
	PYTHONPATH=$(PROJ_TAXI)/src $(PY_TAXI) -m taxi_drift_monitoring.simulate
train-taxi-drift:   ## Treina o modelo de referência (jan) e pontua os meses seguintes
	PYTHONPATH=$(PROJ_TAXI)/src $(PY_TAXI) -m taxi_drift_monitoring.monitor
report-taxi:        ## Gera o relatório Evidently de jan vs. junho em data/reports (não versionado)
	PYTHONPATH=$(PROJ_TAXI)/src $(PY_TAXI) -m taxi_drift_monitoring.report 2024-06
report-taxi-commit: ## Regenera o relatório versionado em data/evidently (≈4 MB)
	PYTHONPATH=$(PROJ_TAXI)/src $(PY_TAXI) -m taxi_drift_monitoring.report 2024-06 --commit
test-taxi-drift:    ## Testes do projeto de drift
	PYTHONPATH=$(PROJ_TAXI)/src $(PY_TAXI) -m pytest -q $(PROJ_TAXI)/tests

# Feature store: dados sintéticos (schema Instacart) -> feast apply -> demo PIT x ingênuo -> exports
data-feature-store: ## Gera os dados sintéticos, constrói as features, aplica o Feast e exporta a demo
	PYTHONPATH=$(PROJ_FEAT)/src $(PY_FEAT) -m feature_store_demo.synthetic
	PYTHONPATH=$(PROJ_FEAT)/src $(PY_FEAT) -m feature_store_demo.build
	cd $(PROJ_FEAT)/feature_repo && $(abspath $(PROJ_FEAT))/.venv/bin/feast apply
	PYTHONPATH=$(PROJ_FEAT)/src $(PY_FEAT) -m feature_store_demo.demo
train-feature-store: data-feature-store   ## Alias: a feature store não treina modelo; prepara as features do modelo de recompra
test-feature-store: ## Testes da feature store
	PYTHONPATH=$(PROJ_FEAT)/src $(PY_FEAT) -m pytest -q $(PROJ_FEAT)/tests

# RAG: leis do Planalto -> limpeza -> chunks por artigo -> BM25 -> avaliação de recuperação
data-rag:           ## Baixa as 5 leis, gera o corpus, o índice BM25 e a avaliação de recuperação
	PYTHONPATH=$(PROJ_RAG)/src $(PY_RAG) -m rag_assistant.corpus
	PYTHONPATH=$(PROJ_RAG)/src $(PY_RAG) -m rag_assistant.evaluate
test-rag-assistant: ## Testes do RAG (inclui o caminho LLM com cliente mockado)
	PYTHONPATH=$(PROJ_RAG)/src $(PY_RAG) -m pytest -q $(PROJ_RAG)/tests
