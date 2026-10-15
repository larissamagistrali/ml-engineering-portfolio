"""Engenharia de features compartilhada por treino, API e app (evita train/serve skew).

Convenção do contrato: as listas `pay_status`, `bill_amt` e `pay_amt` têm 6 posições e o índice 0
é o mês mais recente (setembro/2005 no dataset: PAY_0, BILL_AMT1, PAY_AMT1).
Atributos protegidos (sexo, estado civil) não entram no modelo.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .config import N_MONTHS

FEATURE_SET_VERSION = "1"

BASE_COLUMNS = ["limit_bal", "age", "education"]
PAY_COLUMNS = [f"pay_{i}" for i in range(1, N_MONTHS + 1)]
BILL_COLUMNS = [f"bill_{i}" for i in range(1, N_MONTHS + 1)]
PAYAMT_COLUMNS = [f"payamt_{i}" for i in range(1, N_MONTHS + 1)]
RAW_COLUMNS = BASE_COLUMNS + PAY_COLUMNS + BILL_COLUMNS + PAYAMT_COLUMNS

DERIVED_COLUMNS = [
    "util_1",
    "util_mean",
    "bill_trend",
    "max_delay",
    "n_delays",
    "mean_delay",
    "pay_ratio_1",
    "pay_ratio_mean",
]
FEATURE_NAMES = RAW_COLUMNS + DERIVED_COLUMNS

FEATURE_LABELS = {
    "limit_bal": "Limite de crédito",
    "age": "Idade",
    "education": "Escolaridade",
    "util_1": "Utilização do limite (último mês)",
    "util_mean": "Utilização média do limite (6 meses)",
    "bill_trend": "Tendência da fatura (último vs. 6º mês, % do limite)",
    "max_delay": "Maior atraso nos 6 meses",
    "n_delays": "Nº de meses com atraso",
    "mean_delay": "Atraso médio (meses)",
    "pay_ratio_1": "Pagamento do último mês / fatura anterior",
    "pay_ratio_mean": "Pagamento médio / fatura média (5 meses)",
    **{f"pay_{i}": f"Situação de pagamento, mês -{i - 1}" if i > 1 else "Situação de pagamento, último mês"
       for i in range(1, N_MONTHS + 1)},
    **{f"bill_{i}": f"Fatura, mês -{i - 1}" if i > 1 else "Fatura, último mês" for i in range(1, N_MONTHS + 1)},
    **{f"payamt_{i}": f"Valor pago, mês -{i - 1}" if i > 1 else "Valor pago, último mês"
       for i in range(1, N_MONTHS + 1)},
}


def _safe_ratio(num: pd.Series, den: pd.Series, default: float = 1.0, clip: float = 2.0) -> pd.Series:
    """num/den com default quando não há nada a pagar (den <= 0); limitado a [0, clip]."""
    ratio = num / den.where(den > 0)
    return ratio.fillna(default).clip(0, clip)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Recebe as colunas RAW_COLUMNS e devolve FEATURE_NAMES (mesma ordem usada no treino)."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"colunas ausentes: {missing}")
    df = raw[RAW_COLUMNS].astype(float).copy()
    pay = df[PAY_COLUMNS]
    delays = pay.clip(lower=0)
    limit = df["limit_bal"].where(df["limit_bal"] > 0, np.nan)
    df["util_1"] = (df["bill_1"] / limit).fillna(0.0)
    df["util_mean"] = (df[BILL_COLUMNS].mean(axis=1) / limit).fillna(0.0)
    df["bill_trend"] = ((df["bill_1"] - df[f"bill_{N_MONTHS}"]) / limit).fillna(0.0)
    df["max_delay"] = pay.max(axis=1)
    df["n_delays"] = (pay >= 1).sum(axis=1).astype(float)
    df["mean_delay"] = delays.mean(axis=1)
    # O pagamento do mês t quita a fatura do mês t+1 (anterior); por isso o desalinhamento de índices.
    df["pay_ratio_1"] = _safe_ratio(df["payamt_1"], df["bill_2"])
    paid = df[PAYAMT_COLUMNS[:-1]].sum(axis=1)
    billed = df[BILL_COLUMNS[1:]].clip(lower=0).sum(axis=1)
    df["pay_ratio_mean"] = _safe_ratio(paid, billed)
    return df[FEATURE_NAMES]


def raw_from_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Converte registros do contrato (listas de 6 posições) para o quadro de colunas RAW_COLUMNS."""
    rows = []
    for r in records:
        row: dict[str, Any] = {"limit_bal": r["limit_bal"], "age": r["age"], "education": r["education"]}
        for key, prefix in (("pay_status", "pay"), ("bill_amt", "bill"), ("pay_amt", "payamt")):
            values = r[key]
            if len(values) != N_MONTHS:
                raise ValueError(f"{key} precisa ter {N_MONTHS} posições")
            row.update({f"{prefix}_{i + 1}": v for i, v in enumerate(values)})
        rows.append(row)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)
