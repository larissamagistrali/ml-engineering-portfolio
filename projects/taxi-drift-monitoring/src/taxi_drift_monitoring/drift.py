"""Métricas de drift (PSI e KS) e de desempenho (MAE, MAPE) por mês."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config

EPS = 1e-6


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI com faixas de quantis da referência (faixas únicas; valores atuais fora são levados às bordas)."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variável quase constante na referência
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    levels = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    r = ref.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).clip(lower=EPS)
    c = cur.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).clip(lower=EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_level(value: float) -> tuple[str, str]:
    """Classificação do PSI com ícone e texto (nunca só cor)."""
    if value >= config.PSI_HIGH:
        return "🔴", "alto"
    if value >= config.PSI_MODERATE:
        return "🟠", "moderado"
    return "🟢", "estável"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str] = config.FEATURES) -> pd.DataFrame:
    """PSI (todas as features) e KS p-valor (só numéricas; em categóricas, KS não se aplica)."""
    rows = []
    for f in features:
        if f in config.NUMERIC:
            psi = psi_numeric(ref[f].to_numpy(float), cur[f].to_numpy(float))
            ks = ks_2samp(ref[f], cur[f])
            rows.append({"feature": f, "kind": "numérica", "psi": psi, "ks_stat": float(ks.statistic),
                         "ks_pvalue": float(ks.pvalue)})
        else:
            rows.append({"feature": f, "kind": "categórica", "psi": psi_categorical(ref[f], cur[f]),
                         "ks_stat": np.nan, "ks_pvalue": np.nan})
    return pd.DataFrame(rows)


def performance(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """MAE em minutos; MAPE só em corridas com duração >= 3 min (MAPE explode perto de zero)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    mask = y >= 3
    return {
        "mae": float(np.mean(np.abs(y - pred))),
        "mape_ge3min": float(np.mean(np.abs(y[mask] - pred[mask]) / y[mask])),
        "bias": float(np.mean(pred - y)),
        "median_ae": float(np.median(np.abs(y - pred))),
    }
