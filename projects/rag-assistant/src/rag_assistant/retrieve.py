"""Recuperação: BM25 sobre o índice serializado; devolve os trechos (chunks) com metadados."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config
from .bm25 import BM25, tokenize


@dataclass
class Hit:
    chunk_id: str
    law_short: str
    law_number: str
    article: str
    chapter: str
    text: str
    url: str
    score: float


def load_chunks(path: Path = config.CORPUS_DIR / "chunks.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Retriever:
    def __init__(self, index: BM25, chunks: list[dict]) -> None:
        self.index = index
        self.by_id = {c["chunk_id"]: c for c in chunks}

    @classmethod
    def from_disk(cls) -> "Retriever":
        return cls(BM25.load(config.CORPUS_DIR / "bm25_index.json"), load_chunks())

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not tokenize(query):
            return []
        out = []
        for i, score in self.index.search(query, k=k):
            c = self.by_id[self.index.doc_ids[i]]
            out.append(Hit(c["chunk_id"], c["law_short"], c["law_number"], c["article"], c["chapter"], c["text"],
                           c["url"], round(score, 3)))
        return out
