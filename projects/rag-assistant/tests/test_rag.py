"""Testes do RAG: parsing, BM25, validação de citações e caminho LLM com cliente MOCKADO (sem chamada real)."""
import json
from types import SimpleNamespace

import pytest

from rag_assistant.bm25 import BM25, tokenize
from rag_assistant.generate import (
    REFUSAL, LLMAnswerer, extractive_answer, model_supports_fallback, validate,
)
from rag_assistant.parse import chunk_law, html_to_text
from rag_assistant.retrieve import Hit

LAW = {"id": "x", "short": "X", "number": "1/2000", "url": "https://example.test/x"}
HTML = """<html><head><style>p{}</style></head><body>
<p>CAPÍTULO I</p><p>DO OBJETO</p>
<p>Art. 1º Esta lei trata de dados pessoais de forma detalhada para o teste.</p>
<p>Art. 2º Texto do artigo dois que é suficientemente longo para ser mantido.</p>
<p>Art. 3º (VETADO).</p>
<p>Art. 4º Artigo com marcação (Vide Lei nº 9.999) e texto normativo suficiente aqui.</p>
<p>Art. 337-A. Artigo com sufixo e texto normativo suficientemente longo para o corpus.</p>
</body></html>"""


def test_html_to_text_and_chunking():
    chunks = chunk_law(HTML, LAW)
    labels = [c.article for c in chunks]
    assert labels == ["Art. 1", "Art. 2", "Art. 4", "Art. 337-A"]  # vetado sai; sufixo preservado
    assert all(c.chapter == "CAPÍTULO I" for c in chunks)
    art4 = next(c for c in chunks if c.article == "Art. 4")
    assert "(Vide" not in art4.text and "normativo" in art4.text
    assert "CAPÍTULO I" in html_to_text(HTML)


def test_bm25_ranks_relevant_doc_first():
    docs = [tokenize("prazo para recurso contra negativa de acesso"), tokenize("neutralidade de rede pacotes"),
            tokenize("dados pessoais sensíveis origem racial")]
    idx = BM25(docs, ["a", "b", "c"])
    assert idx.search("neutralidade de rede")[0][0] == 1
    assert idx.search("dados pessoais sensíveis")[0][0] == 2


def test_tokenize_removes_accents_and_stopwords():
    assert "licitação" not in tokenize("licitação") and "licitacao" in tokenize("licitação")
    assert "de" not in tokenize("de consumidor")


def test_validate_flags_invented_citation():
    v = validate("A resposta é X [1][7].", n_sources=3)
    assert not v.ok and v.invalid == [7]


def test_validate_accepts_valid_citations_and_explicit_refusal():
    assert validate("Sim [1], e também [2, 3].", n_sources=3).ok
    refusal = validate(REFUSAL, n_sources=3)
    assert refusal.ok and refusal.refused


def test_validate_rejects_uncited_answer():
    assert not validate("Resposta sem fonte nenhuma.", n_sources=2).ok


def test_extractive_answer_has_citations_and_no_generation():
    hits = [Hit("x-art1", "X", "1/2000", "Art. 1", "", "texto", "u", 1.0)]
    text = extractive_answer(hits)
    assert "[1]" in text and "modo extrativo" in text
    assert extractive_answer([]) == REFUSAL


def test_haiku_gets_no_fallback_parameter():
    assert model_supports_fallback("claude-opus-5-5") and model_supports_fallback("claude-sonnet-5-5")
    assert not model_supports_fallback("claude-haiku-5-5")


class _FakeStream:
    def __init__(self, chunks, stop_reason="end_turn"):
        self._chunks = chunks
        self._final = SimpleNamespace(stop_reason=stop_reason)
        self.text_stream = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


class _FakeClient:
    def __init__(self, chunks, stop_reason="end_turn", beta=True):
        self.calls = []
        self._chunks, self._stop = chunks, stop_reason
        make = lambda **kw: (self.calls.append(("beta" if beta else "plain", kw)), _FakeStream(self._chunks, self._stop))[1]  # noqa: E731
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=make))
        self.messages = SimpleNamespace(stream=make)


HITS = [Hit("x-art1", "X", "1/2000", "Art. 1", "", "texto um", "u", 1.0),
        Hit("x-art2", "X", "1/2000", "Art. 2", "", "texto dois", "u", 0.9)]


def test_llm_path_uses_fallback_and_effort_with_mock():
    client = _FakeClient(["Resposta ", "[1]."])
    ans = LLMAnswerer(client, model="claude-opus-5-5")
    text = "".join(ans.stream("pergunta?", HITS))
    kind, kw = client.calls[0]
    assert kind == "beta" and kw["fallbacks"] == "default" and kw["output_config"] == {"effort": "low"}
    assert "thinking" not in kw
    assert validate(text, len(HITS)).ok and ans.last_stop_reason == "end_turn"


def test_llm_haiku_path_has_no_fallback_with_mock():
    client = _FakeClient(["[2]"], beta=False)
    "".join(LLMAnswerer(client, model="claude-haiku-5-5").stream("q?", HITS))
    kind, kw = client.calls[0]
    assert kind == "plain" and "fallbacks" not in kw


def test_refusal_stop_reason_is_surfaced():
    client = _FakeClient([""], stop_reason="refusal")
    ans = LLMAnswerer(client, model="claude-opus-5-5")
    "".join(ans.stream("q?", HITS))
    assert ans.last_stop_reason == "refusal"


def test_real_corpus_retrieval_metrics_match_saved_file():
    """Se o índice e o JSON de avaliação existirem, recalcula as métricas e confere com o arquivo salvo."""
    from rag_assistant import config
    from rag_assistant.evaluate import evaluate
    from rag_assistant.retrieve import load_chunks

    idx_path = config.CORPUS_DIR / "bm25_index.json"
    saved = config.DATA_DIR / "retrieval_eval.json"
    if not (idx_path.exists() and saved.exists()):
        pytest.skip("rode `make data-rag` antes")
    qs = json.loads((config.CORPUS_DIR / "retrieval_questions.json").read_text(encoding="utf-8"))
    res = evaluate(BM25.load(idx_path), load_chunks(), qs)
    assert res["metrics"] == json.loads(saved.read_text(encoding="utf-8"))["metrics"]
