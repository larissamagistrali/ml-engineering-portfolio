"""Pipeline do corpus: coleta -> limpeza -> chunks por artigo -> índice BM25 serializado.

    PYTHONPATH=projects/rag-assistant/src python -m rag_assistant.corpus
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .bm25 import BM25, tokenize
from .fetch import fetch_all
from .parse import chunk_law, decode


def build(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.CORPUS_DIR) -> dict:
    files = fetch_all(raw_dir)
    chunks = []
    per_law = {}
    for law in config.LAWS:
        found = chunk_law(decode(files[law["id"]].read_bytes()), law)
        per_law[law["short"]] = {"artigos": len(found), "revogados_excluidos": sum(c.revoked for c in found)}
        chunks.extend(c for c in found if not c.revoked)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    index = BM25([tokenize(c.text + " " + c.article + " " + c.law_short) for c in chunks],
                 [c.chunk_id for c in chunks])
    index.save(out_dir / "bm25_index.json")
    summary = {"leis": per_law, "chunks_indexados": len(chunks), "vocabulario": len(index.idf),
               "fontes": [{"lei": l["short"], "numero": l["number"], "url": l["url"]} for l in config.LAWS]}
    (out_dir / "corpus_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
