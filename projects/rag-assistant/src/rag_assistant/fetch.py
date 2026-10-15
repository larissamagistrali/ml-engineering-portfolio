"""Baixa os HTMLs do Planalto para o cache (via `.part`). Reaproveita arquivos já existentes."""
from __future__ import annotations

import urllib.request
from pathlib import Path

from . import config

UA = "Mozilla/5.0 (compatible; ml-engineering-portfolio/1.0; educational RAG demo)"


def fetch_all(raw_dir: Path = config.RAW_DIR) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for law in config.LAWS:
        target = raw_dir / law["file"]
        if not (target.exists() and target.stat().st_size > 0):
            part = target.with_name(target.name + ".part")
            req = urllib.request.Request(law["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as resp, part.open("wb") as fh:  # noqa: S310 (URL fixa)
                fh.write(resp.read())
            part.rename(target)
        out[law["id"]] = target
    return out
