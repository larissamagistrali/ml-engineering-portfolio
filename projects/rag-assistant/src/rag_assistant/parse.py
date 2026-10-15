"""Limpeza do HTML do Planalto e chunking por artigo ("Art. N").

Regras documentadas (aplicadas, nesta ordem):
1. Decodificação: UTF-8 se válido; senão cp1252 (os arquivos do Planalto são Windows-1252).
2. Remove <style>, <script> e comentários; quebras de parágrafo viram linhas.
3. Remove marcações editoriais entre parênteses que começam com "(Vide", "(Incluído", "(Incluída",
   "(Redação dada", "(Vigência", "(Vigência" e "(Regulamento" (não alteram o texto normativo, só apontam
   outras normas). Remove também a parte "Vigência" entre colchetes quando existir.
4. Artigo que o Planalto marca como revogado (texto "Art. N ... (Revogado"): mantido no corpus, mas
   marcado com `revoked=True`, e excluído do índice de busca.
5. Capítulo: última linha "CAPÍTULO ..." vista antes do artigo.
6. Artigos vetados sem parágrafos restantes ("Art. N. (VETADO).") ficam fora do corpus; se têm parágrafos, entram.
7. Sufixos ("Art. 337-A") são preservados como artigos distintos.
"""
from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass

ARTICLE_RE = re.compile(r"^\s*Art\.\s*(\d+)(?:\s*-\s*([A-Z]))?\s*[º°oª]?(?=[\s.\-–])", re.MULTILINE)
VETO_MARK_RE = re.compile(r"\(\s*VETAD[OA]\s*\)\.?", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*(CAPÍTULO\s+[IVXLCDM\dº°o]+)\s*$", re.MULTILINE | re.IGNORECASE)
EDITORIAL_RE = re.compile(
    r"\s*\((?:Vide|Incluíd[oa]|Incluíd[oa]s|Redação dada|Vigência|Regulamento|Renumerad[oa]|Vetad[oa]|Revogad[oa]\s+pel[ao])[^()]*\)",
    re.IGNORECASE,
)
REVOKED_RE = re.compile(r"\(\s*Revogad[oa]", re.IGNORECASE)
BR_RE = re.compile(r"<\s*br\s*/?>|</\s*p\s*>|</\s*div\s*>|</\s*h\d\s*>|</\s*tr\s*>", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    law_id: str
    law_short: str
    law_number: str
    article: str
    article_number: int
    chapter: str
    text: str
    url: str
    revoked: bool

    def to_dict(self) -> dict:
        return asdict(self)


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def html_to_text(markup: str) -> str:
    s = re.sub(r"(?is)<(style|script)[^>]*>.*?</\1>", " ", markup)
    s = re.sub(r"(?s)<!--.*?-->", " ", s)
    s = BR_RE.sub("\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def strip_editorial(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", EDITORIAL_RE.sub("", text)).strip()


def chunk_law(markup: str, law: dict) -> list[Chunk]:
    """Divide a lei em artigos. Cada chunk vai do "Art. N" até o próximo artigo."""
    text = html_to_text(markup)
    # Ignora o sumário/cabeçalho antes do primeiro artigo.
    matches = list(ARTICLE_RE.finditer(text))
    chapters = [(m.start(), m.group(1).upper()) for m in CHAPTER_RE.finditer(text)]
    chunks: list[Chunk] = []
    seen_numbers: set[str] = set()
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        number = int(m.group(1))
        suffix = m.group(2) or ""
        label = f"Art. {number}-{suffix}" if suffix else f"Art. {number}"
        # Artigos repetidos (ex.: sumário em texto corrido) são ignorados; mantém a primeira ocorrência.
        first_line = body.split("\n", 1)[0]
        core = VETO_MARK_RE.sub("", body).strip()  # artigo vetado sem parágrafos restantes não tem texto normativo
        if label in seen_numbers or len(core) < 25:
            continue
        seen_numbers.add(label)
        chapter = ""
        for pos, name in chapters:
            if pos < m.start():
                chapter = name
        revoked = bool(REVOKED_RE.search(first_line[:200]))
        clean = strip_editorial(" ".join(body.split()))  # quebras de linha do HTML viram espaço
        chunks.append(Chunk(
            chunk_id=f"{law['id']}-art{number}{suffix.lower()}",
            law_id=law["id"], law_short=law["short"], law_number=law["number"],
            article=label, article_number=number, chapter=chapter,
            text=clean, url=law["url"], revoked=revoked,
        ))
    return chunks
