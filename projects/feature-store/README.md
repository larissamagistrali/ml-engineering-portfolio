# Feature store de recompra (Feast, point-in-time)

> **Dados sintéticos.** O Instacart Market Basket Analysis é Kaggle-only. Os dados são gerados com o **mesmo schema**
> (orders, order_products, products, aisles, departments) por `feature_store_demo/synthetic.py`, com semente fixa.
> Os valores não são reais. Os números mostram o método de point-in-time, não comportamento de consumidores.

## Problema
Em modelos com histórico, a feature usada no treino precisa ser a que existia **no instante da decisão**. Um join
que pega o valor mais recente de cada cliente "vaza" o futuro e infla as métricas. Este projeto constrói a feature
store de um modelo de recompra, faz o join correto e quantifica o vazamento do join ingênuo.

## Dados
- Gerados em `feature_repo/data/raw/` (gitignored): 3.000 clientes, 27.227 pedidos, 122.287 itens de pedido, 2.000 produtos,
  134 corredores, 21 departamentos. Cada pedido tem `order_timestamp` (sintético) para o point-in-time.
- Alvo: **o cliente compra este produto no pedido seguinte**. Só entram produtos já comprados antes pelo cliente.
  Amostra de 60.000 linhas (taxa positiva de 6,3%).

## Abordagem
1. `build.py` gera três tabelas de **estado** com `event_timestamp` (horário do pedido em que o estado passou a valer):
   - `customer_order_stats` (cliente): pedidos, itens, cesta média, intervalo médio, fração de recompras;
   - `customer_department_affinity` (cliente x departamento): itens e participação;
   - `customer_product_history` (cliente x produto): vezes comprado e intervalo entre as duas últimas compras.
2. `feature_repo/`: `feature_store.yaml` (registry e online store em SQLite, offline em arquivo), entidades
   (`customer`, `department`, `product`), três `FeatureView` com TTL de 3.650 dias e o `FeatureService` `recompra_v1`.
3. `feast apply` registra tudo. `demo.py` faz `materialize` e compara:
   - **join point-in-time** (`get_historical_features`): último estado com `event_timestamp <= ts` do rótulo;
   - **join ingênuo**: último estado de cada chave, ignorando o tempo.
4. Avaliação em corte temporal: treino nos 70% iniciais do tempo, teste nos 30% finais, com `HistGradientBoosting`.

**Desvio do Feast encontrado e tratado:** o join de arquivo do Feast 0.67 descarta linhas com a mesma
(cliente, timestamp), que ocorrem quando vários produtos entram no mesmo pedido. Verificado: de 2.000 linhas de entrada,
o Feast devolvia 1.115. A correção soma 1, 2, 3... µs ao timestamp da linha (menor que o intervalo entre pedidos, então
o estado do pedido continua sendo o último válido) e restaura o timestamp original com um `row_id`. Depois disso,
o Feast devolve todas as linhas.

## Resultados
Números de `data/pit_vs_naive_metrics.json` (uma execução, semente 42).

| Join | ROC-AUC (teste temporal) | PR-AUC (teste temporal) |
|---|---|---|
| Point-in-time (Feast) | **0,602** | **0,087** |
| Ingênuo (último estado) | 0,892 | 0,259 |
| **Ganho falso do ingênuo** | **+0,290** | **+0,172** |

- Pedidos do cliente no momento da decisão: **6,4** (PIT) contra **10,4** (ingênuo, inclui pedidos futuros).
- Linhas com valor diferente: `n_orders` e `n_items_total` em 100%; `dept_purchases` em 60%; `times_bought` em 20%.
- **Verificação do point-in-time:** em 300 linhas amostradas, o `times_bought` devolvido pelo Feast bate com um cálculo
  independente (contagem de compras até o pedido k) em **300 de 300**. Está no teste `test_feast_point_in_time_matches_manual_asof`.
- O AUC absoluto do PIT é baixo (0,60), e isso é esperado: no processo gerador sintético, o sinal de recompra é fraco
  (`times_bought` sozinho tem AUC 0,57). O resultado que importa é a **diferença** entre os dois joins.
- **Online:** `materialize` até 2024-05-13 e `get_online_features` por entidade (exemplos em `data/online_lookup.json`).
  O online guarda o último valor materializado por entidade, então não há consulta "as of" online.

## Como rodar
```bash
make venv-feature-store
make data-feature-store     # gera os dados, build, feast apply, demo e exports
make test-feature-store
```
`feast apply` e `materialize` rodam dentro de `feature_repo/`. A página do app lê só `data/*.json` e `data/*.parquet`.

## Limitações
- **Dados sintéticos**, com schema do Instacart. Nada aqui é comportamento real de consumidores.
- **Sem produto-nível agregado:** não há popularidade global do produto como feature, porque exigiria estado com corte
  temporal próprio. Fica como próximo passo.
- **Desambiguação de 1 µs:** contorna o descarte de linhas do Feast 0.67 (ver acima). Não é um recurso oficial da API.
- **Online sem tempo:** o lookup online devolve o último valor materializado; o point-in-time é offline.
- **Escala:** o offline store é arquivo local (pandas). Para volumes grandes, o caminho seria Spark/BigQuery.
- **Registry:** o registry SQLite e os dados de origem ficam em `feature_repo/data/` (gitignored, ≈100 MB). `make data-feature-store` os recria.
