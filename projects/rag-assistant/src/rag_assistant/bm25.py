"""BM25 próprio (sem dependências pesadas). Tokenização simples em português: minúsculas, sem acento, sem stopwords."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

STOPWORDS = frozenset("""a o os as um uma uns umas de do da dos das em no na nos nas por para pelo pela pelos pelas
com sem sob sobre e ou que se ao aos à às é ser sua seu suas seus ele ela eles elas isso esta este estes estas
esse essa esses essas aquele aquela como mais menos entre quando onde qual quais quem já não nem também até
pode podem deve devem há foi ter tem seja sejam caso cada""".split())
K1, B = 1.5, 0.75


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize(text)) if len(t) > 1 and t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], doc_ids: list[str], k1: float = K1, b: float = B) -> None:
        self.doc_ids = doc_ids
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(1, len(docs))
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.postings.setdefault(t, []).append((i, f))

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.doc_ids)
        for t in set(query_tokens):
            if t not in self.postings:
                continue
            idf = self.idf[t]
            for i, f in self.postings[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = self.score(tokenize(query))
        ranked = sorted(((i, s) for i, s in enumerate(scores) if s > 0), key=lambda x: (-x[1], x[0]))
        return ranked[:k]

    def to_dict(self) -> dict:
        return {
            "k1": self.k1, "b": self.b, "avgdl": round(self.avgdl, 4), "n_docs": len(self.doc_ids),
            "doc_ids": self.doc_ids, "doc_len": self.doc_len,
            "idf": {t: round(v, 5) for t, v in self.idf.items()},
            "postings": {t: [[i, f] for i, f in p] for t, p in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BM25":
        obj = cls.__new__(cls)
        obj.doc_ids = d["doc_ids"]
        obj.k1, obj.b = d["k1"], d["b"]
        obj.doc_len = d["doc_len"]
        obj.avgdl = d["avgdl"]
        obj.idf = d["idf"]
        obj.postings = {t: [(i, f) for i, f in p] for t, p in d["postings"].items()}
        obj.tf = None
        return obj

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
