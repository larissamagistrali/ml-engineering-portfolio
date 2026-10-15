"""Logs estruturados (uma linha JSON por evento), só com a biblioteca padrão."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

# Atributos padrão do LogRecord que não devem ser copiados como campos extras.
_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **{k: v for k, v in record.__dict__.items() if k not in _STD},
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(level: str = "INFO") -> logging.Logger:
    """Configura o logger `credit_risk_api` para emitir JSON em stdout (idempotente)."""
    logger = logging.getLogger("credit_risk_api")
    logger.setLevel(level)
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger
