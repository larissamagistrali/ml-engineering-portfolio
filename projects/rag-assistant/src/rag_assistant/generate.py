"""Geração de resposta. Dois modos:

- extrativo (padrão, sem chave): mostra os trechos recuperados com citação. Não inventa texto.
- LLM (com ANTHROPIC_API_KEY): Claude responde só com os trechos, cita [n], e a resposta passa por
  validação pós-geração (todo [n] citado tem que existir; recusa explícita quando o contexto não responde).

O SDK `anthropic` é opcional: sem ele, o modo LLM fica desativado com mensagem.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .retrieve import Hit

REFUSAL = "Não encontrei resposta nos trechos recuperados."
DEFAULT_MODEL = "claude-opus-5-5"
CITE_RE = re.compile(r"\[([\d,\s]+)\]")

SYSTEM_PROMPT = f"""Você responde perguntas sobre leis brasileiras usando APENAS os trechos numerados que receber.
Regras:
- Cite cada afirmação com o número do trecho entre colchetes, por exemplo [1] ou [2][3].
- Use somente números de trechos que existam na lista recebida.
- Se os trechos não respondem à pergunta, responda exatamente: "{REFUSAL}"
- Não use conhecimento externo nem complete lacunas. Responda em português, de forma objetiva."""


def cite_label(h: Hit) -> str:
    return f"{h.law_short} ({h.law_number}), {h.article}"


def build_user_message(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{i}] {cite_label(h)} — {h.text}" for i, h in enumerate(hits, start=1)]
    return "Trechos:\n" + "\n\n".join(blocks) + f"\n\nPergunta: {question}"


def extractive_answer(hits: list[Hit]) -> str:
    if not hits:
        return REFUSAL
    lines = ["Trechos recuperados (modo extrativo, sem geração):"]
    lines += [f"- **[{i}]** {cite_label(h)}: {h.text[:280]}{'…' if len(h.text) > 280 else ''}"
              for i, h in enumerate(hits, start=1)]
    return "\n".join(lines)


@dataclass
class Validation:
    ok: bool
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    refused: bool = False
    problems: list[str] = field(default_factory=list)


def validate(text: str, n_sources: int) -> Validation:
    """Checa as citações: todo [n] tem que existir na lista; sem citação só vale como recusa explícita."""
    cited: list[int] = []
    for group in CITE_RE.findall(text):
        cited += [int(x) for x in re.findall(r"\d+", group)]
    invalid = sorted({c for c in cited if c < 1 or c > n_sources})
    refused = REFUSAL in text
    problems = []
    if invalid:
        problems.append(f"citações inexistentes: {invalid}")
    if not cited and not refused:
        problems.append("resposta sem nenhuma citação [n] e sem recusa explícita")
    return Validation(ok=not problems, cited=sorted(set(cited)), invalid=invalid, refused=refused, problems=problems)


def model_supports_fallback(model: str) -> bool:
    """Haiku não tem fallback server-side: ali não se envia o parâmetro."""
    return model.startswith(("claude-opus", "claude-sonnet"))


def get_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:  # st.secrets só existe dentro do Streamlit
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001 (ausência de secrets não é erro)
        return None


def get_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def error_message(exc: Exception) -> str:
    """Mensagem em português para a cadeia de exceções tipadas do SDK."""
    try:
        import anthropic as a
    except ImportError:  # pragma: no cover
        return "SDK anthropic não instalado."
    if isinstance(exc, a.AuthenticationError):
        return "Chave de API inválida. Verifique ANTHROPIC_API_KEY."
    if isinstance(exc, a.PermissionDeniedError):
        return "A chave não tem permissão para este modelo."
    if isinstance(exc, a.NotFoundError):
        return "Modelo não encontrado. Verifique ANTHROPIC_MODEL."
    if isinstance(exc, a.RateLimitError):
        return "Limite de requisições atingido. Tente de novo em instantes."
    if isinstance(exc, a.BadRequestError):
        return f"Requisição recusada pela API: {getattr(exc, 'message', exc)}"
    if isinstance(exc, a.APITimeoutError):
        return "A API demorou demais para responder."
    if isinstance(exc, a.APIConnectionError):
        return "Falha de conexão com a API."
    if isinstance(exc, a.APIStatusError):
        return f"Erro da API (HTTP {exc.status_code})."
    return "Erro inesperado ao gerar a resposta."


class LLMAnswerer:
    """Gera a resposta em streaming e guarda o stop_reason final para a validação."""

    def __init__(self, client: Any, model: str | None = None, max_tokens: int = 4000) -> None:
        self.client = client
        self.model = model or get_model()
        self.max_tokens = max_tokens
        self.last_stop_reason: str | None = None

    def _kwargs(self, question: str, hits: list[Hit]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_message(question, hits)}],
            "output_config": {"effort": "low"},  # thinking é sempre ligado em Opus 5.5: controla-se pelo effort
        }
        return kwargs

    def stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        kwargs = self._kwargs(question, hits)
        if model_supports_fallback(self.model):
            # Fallback server-side padrão da skill: re-roda em outro modelo se houver recusa por política.
            ctx = self.client.beta.messages.stream(
                **kwargs, fallbacks="default", betas=["server-side-fallback-2026-07-01"])
        else:
            ctx = self.client.messages.stream(**kwargs)
        with ctx as s:
            for text in s.text_stream:
                yield text
            final = s.get_final_message()
        self.last_stop_reason = getattr(final, "stop_reason", None)
