# Drift em corridas de táxi (NYC TLC, 2024)

## Problema
Um modelo de previsão de duração de corridas treinado em janeiro de 2024 continua confiável nos meses seguintes?
O monitoramento separa três perguntas: **os dados mudaram** (drift de dados), **a predição mudou** (drift de
predição) e **o erro subiu** (drift de desempenho).

## Dados
- **NYC TLC, Yellow Taxi Trip Records**, janeiro a junho de 2024, arquivos Parquet oficiais (dados abertos da
  cidade de Nova York, domínio público). Zonas: `taxi_zone_lookup.csv` do mesmo portal.
- Amostra de **50.000 corridas por mês** após limpeza básica (semente 42). Corridas com duração fora de 1 a 120 min,
  distância fora de 0,1 a 60 milhas, passageiros fora de 1 a 6 ou pagamento inválido são descartadas.
  Por mês, de 2,7 a 3,2 milhões de corridas passam na limpeza (`data/work/sampling_log.json`, não versionado).
- Alvo: duração da corrida (min). Features: distância, passageiros, hora e dia da semana do embarque,
  distrito (borough) de embarque e desembarque, tipo de pagamento. A tarifa não entra, pois só existe depois da corrida.
- **Nenhum drift artificial foi injetado no dado real.** A seção de simulação é separada e rotulada.

## Abordagem
1. `data.py`: baixa os arquivos para o cache (`.part` e renomeia) e amostra por mês.
2. `model.py`: LightGBM treinado só com 80% de janeiro. O holdout de 20% dá o MAE de referência.
3. `drift.py`: **PSI** por feature (faixas de quantil da referência; ≥ 0,10 moderado, ≥ 0,20 alto), **KS** nas
   numéricas, PSI da predição e **MAE** (min) com **MAPE** só em corridas de pelo menos 3 min.
   Alerta de desempenho: MAE acima de 110% do de referência.
4. `monitor.py`: pontua fevereiro a junho e grava `performance_by_month.parquet` e `drift_by_month.parquet`.
5. `report.py`: relatório **Evidently 0.7** (`DataDriftPreset` + `RegressionPreset`) de janeiro contra junho.
6. `simulate.py`: **simulação** (30% das corridas de junho com distância x1,5, hora +3h e pagamento "cartão"),
   em arquivo separado.

## Resultados
Números de `data/metrics.json` e `data/simulation_metrics.json` (uma execução, semente 42).

| Mês | MAE (min) | MAE / referência | Maior PSI de feature | PSI da predição | Alerta de desempenho |
|---|---|---|---|---|---|
| 2024-01 (referência, holdout) | 3,356 | 1,000 | — | — | — |
| 2024-02 | 3,478 | 1,036 | 0,026 | 0,001 | não |
| 2024-03 | 3,701 | 1,103 | 0,030 | 0,002 | **sim** (> 1,10) |
| 2024-04 | 3,882 | 1,157 | 0,007 | 0,003 | **sim** |
| 2024-05 | 4,388 | 1,307 | 0,025 | 0,004 | **sim** |
| 2024-06 | 4,286 | 1,277 | 0,017 | 0,004 | **sim** |

- **Os dados quase não mudaram:** todos os PSI de features ficam abaixo de 0,10, ou seja, estáveis. O maior valor
  é o do dia da semana, um efeito de calendário (quantos fins de semana caem em cada mês).
- **O erro subiu:** o MAE cresceu cerca de 28% em maio e junho, com drift de dados desprezível. Isso aponta para
  **drift de conceito**: a relação entre distância/hora e duração mudou (por exemplo, trânsito e velocidade média).
- O MAPE (corridas ≥ 3 min) ficou estável entre 0,244 e 0,255. Por isso o aumento do MAE deve estar concentrado em
  corridas mais longas, onde o erro absoluto é maior. Não foi investigado por segmento.
- **Simulação (rotulada):** com 30% das corridas perturbadas, o PSI fica entre 0,000 e 0,038 (abaixo de qualquer alerta),
  mas o MAE sobe de 4,29 para 4,67 min. Uma perturbação parcial pode passar pelo PSI e aparecer só no erro.
- Relatório Evidently de jan vs. jun versionado em `data/evidently/` (4,0 MB). Os demais meses: `make report-taxi`.

**Política de retreino sugerida** (também na página do app): não retreinar só por PSI; retreinar quando o MAE
passar de 110% da referência por 2 meses seguidos ou quando o PSI for alto em 3 ou mais features. Validar no mês
seguinte antes de promover.

## Como rodar
```bash
make venv-taxi-drift        # venv do projeto (LightGBM, Evidently)
make data-taxi-drift        # baixa os 6 meses (cache em data/raw), amostra, monitora e simula
make report-taxi            # relatório Evidently em data/reports (não versionado)
make report-taxi-commit     # regenera o relatório versionado (≈4 MB)
make test-taxi-drift
```
Para reaproveitar um cache de downloads já existente, aponte `TLC_RAW_DIR=/caminho/do/cache`.

## Limitações
- **Janela curta:** seis meses de 2024. Não há como separar sazonalidade anual de drift.
- **Referência única:** o monitor compara tudo com janeiro. Uma referência móvel pode dar outro quadro.
- **Holdout de referência menor:** o MAE de referência usa 10 mil corridas (20% de janeiro); os meses monitorados usam 50 mil.
  A comparação é razoável, mas tem mais variância no denominador.
- **Não há rótulo em tempo real:** o monitor calcula o erro depois que a corrida termina; em produção, isso chega com atraso.
- **Relatório Evidently:** o HTML embute o JavaScript da biblioteca (≈4 MB), por isso só um relatório é versionado.
- **Simulação** não representa mudança real. Serve só para mostrar a sensibilidade dos alertas.
- Os dados brutos ficam no cache (gitignored). `data/work/` também não é versionado.
