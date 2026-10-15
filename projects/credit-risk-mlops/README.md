# Risco de crédito com MLOps (PD)

## Problema
Estimar a probabilidade de inadimplência no mês seguinte (PD) de cada cliente de cartão de crédito, com um
pipeline reprodutível: treino com rastreamento de experimentos, registro de modelo, API de inferência
versionada, model card e testes.

## Dados
- **Substituto de Home Credit / Give Me Some Credit.** Ambos são Kaggle-only (exigem login), então usei o
  **UCI "Default of Credit Card Clients"** (Yeh & Lien, 2009), que é público e sem autenticação.
- 30.000 clientes de um emissor em Taiwan, abril a setembro de 2005. 23 variáveis, sem dados faltantes.
- Licença **CC BY 4.0**. DOI `10.24432/C55S3H`. Fonte: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Os dados têm 17 anos e um único emissor: os resultados ilustram o pipeline, não o crédito atual.
- **Categorias não documentadas** tratadas e registradas em `data/data_quality.json`: escolaridade 0, 5 e 6 viraram
  "outros" (4); estado civil 0 virou "outros" (3). Os códigos de situação de pagamento -2 e 0 existem sem
  definição oficial e foram mantidos como ordinais.
- **Não há data da solicitação**, então não existe corte temporal. A partição é estratificada 60/20/20 (treino/validação/teste).
- Sexo e estado civil são excluídos das features (atributos protegidos). Sexo entra só na auditoria por grupo.

## Abordagem
1. `data.py`: baixa o zip oficial, lê o `.xls` com `xlrd` (tratado como dado, nunca executado) e grava um Parquet.
2. `features.py`: features derivadas (utilização do limite, tendência da fatura, atrasos, razão pagamento/fatura).
   O mesmo código é usado no treino, na API e no app, para evitar *train/serve skew*.
3. `train.py`: baseline logístico e busca aleatória de 12 configurações de LightGBM com parada antecipada.
   Ambos são registrados no **MLflow Model Registry** (backend SQLite em `mlruns/`, ignorado pelo Git).
   O LightGBM só vira `champion` se superar o baseline em ROC-AUC de validação (ganho mínimo de 0,005) e em log-loss.
   O campeão é recarregado **pelo alias** e exportado para `artifacts/model.joblib` + `artifacts/model_card.json`.
4. `api.py`: FastAPI com `/health`, `/model-info`, `/predict` e `/predict/batch`. Validação com Pydantic
   (limites de valor, 6 meses fixos, `extra="forbid"`), versão do modelo em toda resposta, logs JSON por requisição.
5. Explicação por cliente: contribuição exata por feature em log-odds, com o TreeSHAP nativo do LightGBM
   (`pred_contrib`). Para a logística, coeficiente x valor padronizado. Ambas são exatas (somam o logit).

## Resultados
Números copiados de `data/metrics.json` (uma execução de `make train-credit-risk`, seed 42).

| Modelo | Conjunto | ROC-AUC | KS | PR-AUC | Brier | Log-loss | ECE |
|---|---|---|---|---|---|---|---|
| Logística (baseline) | validação | 0,741 | 0,385 | 0,489 | 0,144 | 0,455 | 0,012 |
| **LightGBM (campeão)** | validação | 0,780 | 0,423 | 0,545 | 0,137 | 0,433 | 0,017 |
| Logística (baseline) | teste | 0,760 | 0,421 | 0,523 | 0,139 | 0,443 | 0,012 |
| **LightGBM (campeão)** | teste | **0,792** | **0,450** | **0,580** | **0,131** | **0,419** | 0,015 |

- Taxa observada no teste: 0,221. PD média prevista pelo campeão: 0,220 (boa calibração global).
- Por faixa no teste, a PD prevista acompanha a taxa observada (ex.: faixa E, PD 0,70 vs. observada 0,72).
- Auditoria no teste (não usada na seleção): ROC-AUC 0,797 (mulheres) e 0,783 (homens); 0,790 até 29 anos,
  0,795 de 30 a 44 e 0,785 acima de 45. A taxa observada difere mais (0,208 mulheres vs. 0,241 homens) do que o ROC-AUC entre os grupos; isso não é um teste de equidade.
- Modelo campeão: `credit-risk-pd:v2`, LightGBM, `num_leaves=7`, `learning_rate=0,05`, 231 árvores.

## Como rodar
```bash
# Pré-requisitos: venv com as dependências do projeto (mlflow, xlrd, lightgbm, shap opcional)
PYTHONPATH=projects/credit-risk-mlops/src python -m credit_risk_mlops.data     # baixa e limpa os dados
PYTHONPATH=projects/credit-risk-mlops/src python -m credit_risk_mlops.train    # treina, registra, exporta
mlflow ui --backend-store-uri sqlite:///projects/credit-risk-mlops/mlruns/mlflow.db   # opcional: ver experimentos
PYTHONPATH=projects/credit-risk-mlops/src uvicorn credit_risk_mlops.api:app --port 8701
curl -s localhost:8701/health
pytest projects/credit-risk-mlops/tests
```

### Contrato da API
`POST /predict` com o corpo abaixo (listas de 6 meses, índice 0 = mês mais recente):

```json
{"limit_bal": 120000, "age": 34, "education": 2,
 "pay_status": [0, 0, 0, 0, 0, 0],
 "bill_amt": [41000, 39500, 38200, 36900, 35000, 33800],
 "pay_amt": [2500, 2300, 2200, 2000, 2000, 1900]}
```

Resposta (saída real desta execução, resumida):

```json
{"model_version": "credit-risk-pd:v2",
 "prediction": {"pd": 0.0651, "risk_band": "A", "risk_band_label": "Muito baixo", "warnings": [],
                "top_factors": [{"feature": "max_delay", "contribution_logit": -0.3236, "direction": "decreases_risk"}]}}
```

Docker: `docker build -t credit-risk-api projects/credit-risk-mlops && docker run -p 8000:8000 credit-risk-api`
(imagem multi-stage, usuário não-root, healthcheck em `/health`).

## Limitações
- **Docker não foi executado** neste ambiente (daemon desligado). O `Dockerfile` está escrito, mas não foi construído nem testado.
- Dados de 2005, de um emissor, sem corte temporal. A avaliação tende a ser otimista para uso real.
- Só clientes que receberam crédito aparecem nos dados: não há inferência de rejeitados (viés de seleção).
- A calibração foi medida, mas as probabilidades não foram recalibradas depois do treino.
- A auditoria por grupo é descritiva e não é um teste de equidade formal.
- `mlruns/` (SQLite e artefatos) fica fora do Git. A API lê só `artifacts/`.
- O app carrega `artifacts/`. Se o artefato não carregar na versão de bibliotecas do ambiente, a página mostra
  a mensagem de erro e não re-treina.
- Dados baixados ficam em `data/raw/` (ignorado pelo Git).
