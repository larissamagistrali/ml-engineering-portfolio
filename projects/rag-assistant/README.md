# Assistente de leis brasileiras (RAG com citações)

## Problema
Consultar uma lei exige achar o artigo certo e confiar na fonte. Este assistente recupera o artigo mais relevante,
mostra o trecho com o link oficial e, quando há chave de API, deixa o Claude redigir a resposta **somente** a partir
dos artigos recuperados, com citação numerada e validação das citações.

## Dados
- **Textos oficiais do Planalto** (planalto.gov.br), cinco leis (domínio público):
  LGPD (Lei 13.709/2018), Marco Civil da Internet (Lei 12.965/2014), Lei de Acesso à Informação (Lei 12.527/2011),
  Nova Lei de Licitações (Lei 14.133/2021) e Código de Defesa do Consumidor (Lei 8.078/1990, texto compilado).
- Os HTMLs são baixados para `data/raw/planalto/` (gitignored). Os arquivos têm codificação Windows-1252 e são lidos
  como texto (nunca executados).
- **Regras de limpeza** (documentadas em `parse.py`): remove scripts e estilos; remove marcações editoriais entre
  parênteses ("(Vide Lei …)", "(Redação dada …)", "(Incluído …)", "(Vigência …)"); remove artigos vetados sem texto
  normativo (ex.: "Art. 28. (VETADO)."); preserva sufixos ("Art. 337-A").
- Resultado: **483 artigos indexados** (LGPD 75, Marco Civil 33, LAI 47, Lei 14.133 206, CDC 122).
  Nenhum artigo inteiro é marcado como revogado pelo Planalto; as revogações aparecem dentro de incisos e
  parágrafos, e ficam no texto.

## Abordagem
1. **Chunking por artigo**: cada chunk vai do "Art. N" até o próximo, com lei, número, capítulo e URL.
2. **BM25 próprio** (`bm25.py`, sem embeddings): tokenização sem acento e sem stopwords, k1 = 1,5, b = 0,75.
   O índice serializado (`data/corpus/bm25_index.json`) tem 464 KB.
3. **Geração:**
   - **extrativo (padrão):** sem `ANTHROPIC_API_KEY`, mostra os trechos com citação. Não inventa texto.
   - **LLM:** `anthropic` SDK, modelo `claude-opus-5-5` (troque com `ANTHROPIC_MODEL`), `output_config={"effort": "low"}`,
     thinking omitido (sempre ligado no Opus 5.5), resposta em streaming (`st.write_stream`), fallback server-side
     `fallbacks="default"` com `server-side-fallback-2026-07-01` para Opus e Sonnet. **Haiku não recebe esse parâmetro**.
     Para custo menor, use `ANTHROPIC_MODEL=claude-haiku-5-5`.
   - **Validação pós-geração:** todo `[n]` citado tem que existir entre os trechos recuperados. Sem citação e sem
     recusa explícita ("Não encontrei resposta nos trechos recuperados."), a resposta é marcada. Se o stop_reason for
     `refusal`, o app avisa.
   - **Erros:** a cadeia de exceções tipadas do SDK (`AuthenticationError`, `RateLimitError`, `APIConnectionError`…)
     vira mensagem em português.
4. **Proteções de custo:** modo LLM só com chave; limite de 10 perguntas por sessão; contexto de top-5 trechos.

## Resultados
Avaliação de recuperação com **20 perguntas** escritas à mão, cada uma com o artigo esperado validado contra o texto
oficial. Números de `data/retrieval_eval.json` (k = 5):

| hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---|---|---|
| 0,50 | 0,65 | 0,80 | 0,599 |

- Erros observados (não corrigidos para não ajustar o conjunto de teste): "consumidor" aparece em muitos artigos do CDC
  e puxa o ranking (q16 não entra no top-5); "prática abusiva" e "cláusulas" trazem artigos de outras leis (q17, q20).
  Um filtro por lei ou um reranqueador melhorariam, mas não foram implementados.
- **Geração com LLM não foi exercitada com API real** (sem chave neste ambiente). O caminho foi testado com cliente mockado.

## Como rodar
```bash
make venv-rag-assistant
make data-rag                  # baixa as leis, gera o corpus, o índice e a avaliação
make test-rag-assistant
# modo LLM (opcional):
export ANTHROPIC_API_KEY=...   # ou em .streamlit/secrets.toml (ANTHROPIC_API_KEY = "...")
export ANTHROPIC_MODEL=claude-haiku-5-5   # opcional, menor custo
```

## Limitações
- **Chamada real à API Claude não foi exercitada.** Sem chave neste ambiente e sem o `ant` CLI para `ant auth status`.
  Os testes do modo LLM usam cliente mockado: verificam os parâmetros enviados (fallback, effort, sem thinking, Haiku sem
  fallback), a validação das citações, a recusa e o mapeamento de erros.
- **Corpus de cinco leis**, não o direito brasileiro inteiro. Versões são as do Planalto no momento da coleta.
- **Sem filtro por lei e sem reranqueamento:** a recuperação é só BM25. As falhas estão descritas acima.
- **Avaliação pequena:** 20 perguntas, escritas pelo autor. Não é um benchmark.
- **Resposta jurídica:** o assistente não substitui consulta jurídica.
