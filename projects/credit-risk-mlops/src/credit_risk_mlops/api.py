"""API FastAPI de inferência de PD. Carrega o artefato exportado pelo treino (não acessa o MLflow).

    PYTHONPATH=projects/credit-risk-mlops/src uvicorn credit_risk_mlops.api:app --port 8701
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import config
from .logging_config import configure
from .schemas import BatchRequest, BatchResponse, Customer, HealthResponse, PredictResponse
from .serving import ModelBundle, ModelLoadError, load_bundle, score_records

log = configure()
state: dict[str, Any] = {"bundle": None, "load_error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state["bundle"] = load_bundle()
        log.info("model_loaded", extra={"model_version": state["bundle"].version})
    except ModelLoadError as exc:
        state["load_error"] = str(exc)
        log.error("model_load_failed", extra={"error": str(exc)})
    yield


app = FastAPI(
    title="Credit Risk PD API",
    version="1.0.0",
    description="Probabilidade de inadimplência (PD) no mês seguinte. Dados públicos (UCI, 2005); uso educacional.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["x-request-id"] = rid
    log.info("request", extra={"request_id": rid, "method": request.method, "path": request.url.path,
                               "status": response.status_code, "latency_ms": ms})
    return response


def _bundle() -> ModelBundle:
    if state["bundle"] is None:
        raise HTTPException(status_code=503, detail=f"modelo indisponível: {state['load_error'] or 'não carregado'}")
    return state["bundle"]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> JSONResponse:
    b: ModelBundle | None = state["bundle"]
    body = {"status": "ok" if b else "unavailable", "model_version": b.version if b else None, "model_loaded": b is not None}
    return JSONResponse(body, status_code=200 if b else 503)


@app.get("/model-info", tags=["ops"])
def model_info() -> dict[str, Any]:
    """Model card completo (sem o detalhe de importâncias) para auditoria."""
    b = _bundle()
    card = {k: v for k, v in b.card.items() if k not in ("global_importance_test", "fairness_audit_test")}
    return card


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(customer: Customer, explain: bool = True) -> dict[str, Any]:
    b = _bundle()
    item = score_records(b, [customer.model_dump()], explain=explain)[0]
    return {"model_version": b.version, "prediction": item}


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(body: BatchRequest, explain: bool = False) -> dict[str, Any]:
    b = _bundle()
    items = score_records(b, [c.model_dump() for c in body.customers], explain=explain)
    return {"model_version": b.version, "n": len(items), "predictions": items}
