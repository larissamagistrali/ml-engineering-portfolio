"""Página: assistente de leis (RAG). Modo extrativo por padrão; modo LLM só com chave configurada."""
import json

import _bootstrap  # noqa: F401
import pandas as pd
import streamlit as st

from shared import data, ui

PROJECT = "rag-assistant"
MAX_QUESTIONS = 10  # limite por sessão (proteção de custo em app público)
TOP_K = 5
ui.setup_page("Assistente de leis (RAG)", "⚖️")
ui.header(
    "Assistente de leis (RAG)",
    "Pergunte sobre a LGPD, o Marco Civil, a LAI, a Lei 14.133 e o CDC. A resposta cita o artigo.",
    tags=["BM25 próprio", "chunking por artigo", "Claude API (opcional)", "Planalto"],
    code_path=f"projects/{PROJECT}",
)
problema, dados, abordagem, resultado, demo = ui.tabs()
ROOT = data.project_dir(PROJECT) / "data"
data.use_project(PROJECT)  # pacote rag_assistant em projects/rag-assistant/src (o app não depende do conftest)


@st.cache_resource(show_spinner="Carregando o índice...")
def load_retriever():
    from rag_assistant.retrieve import Retriever

    return Retriever.from_disk()


@st.cache_data
def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def llm_client():
    """Cliente Anthropic só se o SDK e a chave existirem; senão None (modo extrativo)."""
    try:
        import anthropic
    except ImportError:
        return None, "SDK `anthropic` não instalado: modo extrativo."
    from rag_assistant.generate import get_api_key

    key = get_api_key()
    if not key:
        return None, "Sem `ANTHROPIC_API_KEY`: modo extrativo."
    return anthropic.Anthropic(api_key=key, max_retries=2, timeout=60.0), None


client, mode_reason = llm_client()
mode = "LLM (Claude)" if client else "extrativo"
corpus = load_json("corpus/corpus_summary.json")
evaluation = load_json("retrieval_eval.json")

with problema:
    st.markdown(
        "Quem precisa consultar uma lei quer o texto certo, com a fonte. Este assistente busca o artigo mais "
        "relevante para a pergunta e mostra o trecho com o link oficial. Quando há chave de API, o Claude redige a "
        "resposta, **somente** a partir dos artigos recuperados, com citação numerada."
    )

with dados:
    ui.data_notice(
        "real",
        "textos oficiais do Planalto (planalto.gov.br), cinco leis",
        f"{corpus['chunks_indexados']} artigos indexados. Marcações editoriais como '(Vide Lei …)' são removidas; "
        "artigos vetados sem texto normativo ficam fora do índice.",
    )
    st.markdown("**Fontes:**")
    for f in corpus["fontes"]:
        st.markdown(f"- [{f['lei']} ({f['numero']})]({f['url']})")

with abordagem:
    st.markdown(
        "1. **Coleta:** HTML do Planalto (com decodificação Windows-1252 quando necessário).\n"
        "2. **Limpeza:** remove scripts e marcações; mantém o texto normativo.\n"
        "3. **Chunking:** um chunk por artigo, com lei, número, capítulo e URL como metadados.\n"
        "4. **Recuperação:** BM25 próprio (sem embeddings), top-5 trechos.\n"
        "5. **Geração:** modo extrativo (sem chave) ou Claude com instruções para citar `[n]` e recusar quando o "
        "contexto não responde.\n"
        "6. **Validação:** todo `[n]` citado tem que existir nos trechos; senão a resposta é marcada."
    )
    st.caption(f"Modo ativo agora: **{mode}**" + (f" ({mode_reason})" if mode_reason else ""))

with resultado:
    m = evaluation["metrics"]
    ui.metric_row([
        ("hit@1", f"{m['hit@1']:.0%}", "O artigo esperado é o primeiro resultado."),
        ("hit@3", f"{m['hit@3']:.0%}", "Aparece entre os 3 primeiros."),
        ("hit@5", f"{m['hit@5']:.0%}", "Aparece entre os 5 primeiros (contexto enviado ao modelo)."),
        ("MRR@5", f"{m['mrr@5']:.2f}", "Média do inverso da posição do artigo esperado."),
    ])
    st.caption(f"{m['n_questions']} perguntas escritas à mão com o artigo esperado, validado contra o texto oficial. "
               "Sem ajuste fino do ranqueamento para este conjunto.")
    rows = pd.DataFrame([
        {"id": r["id"], "pergunta": r["question"], "artigo esperado": ", ".join(r["expected"]),
         "posição": (str(r["rank"]) if r["rank"] else "fora do top-5"),
         "resultado": ("🟢 no top-1" if r["hit1"] else ("🟠 no top-5" if r["hit5"] else "🔴 fora"))}
        for r in evaluation["per_question"]
    ])
    st.dataframe(rows, hide_index=True, width="stretch")
    st.markdown(
        "**Falhas típicas:** termos genéricos como 'consumidor' aparecem em muitos artigos do CDC e puxam o ranking; "
        "sem filtro por lei, perguntas parecidas podem ir para outra norma (ex.: prática abusiva na Lei 14.133). "
        "Um filtro por lei ou um reranqueador resolveriam parte disso, mas não foram implementados."
    )

with demo:
    if "history" not in st.session_state:
        st.session_state.history = []
    used = len(st.session_state.history)
    st.caption(f"Modo: **{mode}** · perguntas nesta sessão: {used}/{MAX_QUESTIONS}")
    if mode == "extrativo":
        st.info("Modo extrativo: sem chave de API, a página mostra os artigos recuperados, sem redigir uma resposta.")

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("badge"):
                st.caption(turn["badge"])
            for i, h in enumerate(turn["hits"], start=1):
                with st.expander(f"[{i}] {h['law_short']} · {h['article']} · score {h['score']}"):
                    st.write(h["text"])
                    st.markdown(f"[Texto oficial no Planalto]({h['url']})")

    prompt = st.chat_input(
        "Ex.: qual o prazo para desistir de uma compra feita pela internet?",
        disabled=used >= MAX_QUESTIONS,
    )
    if used >= MAX_QUESTIONS:
        st.warning(f"Limite de {MAX_QUESTIONS} perguntas por sessão atingido. Recarregue a página para continuar.")
    if prompt:
        from rag_assistant.generate import LLMAnswerer, extractive_answer, validate, error_message

        retriever = load_retriever()
        hits = retriever.search(prompt, k=TOP_K)
        hit_dicts = [{"law_short": h.law_short, "article": h.article, "text": h.text, "url": h.url,
                      "score": h.score, "chunk_id": h.chunk_id} for h in hits]
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            badge = None
            if client is None:
                answer = extractive_answer(hits)
                st.markdown(answer)
            else:
                try:
                    answerer = LLMAnswerer(client)
                    answer = st.write_stream(answerer.stream(prompt, hits))
                    v = validate(answer, len(hits))
                    if answerer.last_stop_reason == "refusal":
                        badge = "🔴 O modelo recusou a pergunta (política de segurança). Nenhuma resposta foi gerada."
                    elif v.ok and v.refused:
                        badge = "🟠 Recusa explícita: os trechos recuperados não respondem à pergunta."
                    elif v.ok:
                        badge = f"🟢 Citações validadas: {', '.join(f'[{c}]' for c in v.cited)}."
                    else:
                        badge = "🔴 Validação falhou: " + "; ".join(v.problems) + ". Use as fontes abaixo."
                except Exception as exc:  # noqa: BLE001 (cadeia de exceções do SDK tratada em error_message)
                    answer = f"{error_message(exc)} Veja os trechos recuperados abaixo."
                    st.warning(f"⚠️ {answer}")
                    badge = None
                if badge:
                    st.caption(badge)
            for i, h in enumerate(hit_dicts, start=1):
                with st.expander(f"[{i}] {h['law_short']} · {h['article']} · score {h['score']}"):
                    st.write(h["text"])
                    st.markdown(f"[Texto oficial no Planalto]({h['url']})")
        st.session_state.history.append({"question": prompt, "answer": answer, "badge": badge, "hits": hit_dicts})
