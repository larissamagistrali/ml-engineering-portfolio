"""Contrato da API (Pydantic): validação com limites e exemplos que aparecem no /docs."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Six = dict(min_length=6, max_length=6)

PayStatus = Annotated[int, Field(ge=-2, le=9)]
Money = Annotated[float, Field(ge=-1_000_000, le=5_000_000)]
Payment = Annotated[float, Field(ge=0, le=5_000_000)]

EXAMPLE_CUSTOMER = {
    "limit_bal": 120000,
    "age": 34,
    "education": 2,
    "pay_status": [0, 0, 0, 0, 0, 0],
    "bill_amt": [41000, 39500, 38200, 36900, 35000, 33800],
    "pay_amt": [2500, 2300, 2200, 2000, 2000, 1900],
}


class Customer(BaseModel):
    """Perfil do cliente. Nas listas, o índice 0 é o mês mais recente e o 5 é o mais antigo (6 meses)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_CUSTOMER})

    limit_bal: float = Field(ge=1_000, le=5_000_000, description="Limite de crédito concedido (NT$).")
    age: int = Field(ge=18, le=100, description="Idade em anos.")
    education: int = Field(ge=1, le=4, description="1 pós-graduação, 2 universidade, 3 ensino médio, 4 outros.")
    pay_status: list[PayStatus] = Field(
        **Six, description="Situação de pagamento por mês: -2 sem consumo, -1 em dia, 0 uso rotativo, 1..9 meses de atraso."
    )
    bill_amt: list[Money] = Field(**Six, description="Valor da fatura por mês (NT$; pode ser negativo por crédito).")
    pay_amt: list[Payment] = Field(**Six, description="Valor pago por mês (NT$).")


class Factor(BaseModel):
    feature: str
    label: str
    value: float
    contribution_logit: float = Field(description="Contribuição em log-odds (TreeSHAP); positivo aumenta a PD.")
    direction: Literal["increases_risk", "decreases_risk"]


class Prediction(BaseModel):
    pd: float = Field(ge=0, le=1, description="Probabilidade de inadimplência no mês seguinte.")
    risk_band: Literal["A", "B", "C", "D", "E"]
    risk_band_label: str
    warnings: list[str] = []
    top_factors: list[Factor] | None = None


class PredictResponse(BaseModel):
    model_version: str
    prediction: Prediction


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    model_version: str
    n: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_version: str | None
    model_loaded: bool
