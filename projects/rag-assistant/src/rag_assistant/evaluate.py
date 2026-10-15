"""Avaliação de recuperação: hit@1/3/5 e MRR sobre as perguntas com artigo esperado.

    PYTHONPATH=projects/rag-assistant/src python -m rag_assistant.evaluate
"""
from __future__ import annotations

import json

from . import config
from .bm25 import BM25, tokenize
from .retrieve import load_chunks

QUESTIONS = config.CORPUS_DIR / "retrieval_questions.json"


def evaluate(index: BM25, chunks: list[dict], questions: list[dict], k: int = 5) -> dict:
    rows = []
    for q in questions:
        ranked = index.search(q["question"], k=k)
        ids = [index.doc_ids[i] for i, _ in ranked]
        rank = next((r + 1 for r, cid in enumerate(ids) if cid in q["expected"]), None)
        rows.append({"id": q["id"], "question": q["question"], "expected": q["expected"], "top_ids": ids,
                     "rank": rank, "hit1": rank == 1 if rank else False, "hit3": bool(rank and rank <= 3),
                     "hit5": bool(rank and rank <= 5), "rr": (1 / rank) if rank else 0.0})
    n = len(rows)
    metrics = {
        "n_questions": n, "k": k,
        "hit@1": round(sum(r["hit1"] for r in rows) / n, 4),
        "hit@3": round(sum(r["hit3"] for r in rows) / n, 4),
        "hit@5": round(sum(r["hit5"] for r in rows) / n, 4),
        "mrr@5": round(sum(r["rr"] for r in rows) / n, 4),
    }
    return {"metrics": metrics, "per_question": rows,
            "note": "Perguntas e artigos esperados escritos à mão e validados contra o texto oficial (corpus/retrieval_questions.json)."}


def run() -> dict:
    index = BM25.load(config.CORPUS_DIR / "bm25_index.json")
    chunks = load_chunks()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    result = evaluate(index, chunks, questions)
    (config.DATA_DIR / "retrieval_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result["metrics"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
