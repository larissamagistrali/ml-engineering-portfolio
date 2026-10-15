"""Métricas de discriminação, probabilidade e calibração para modelos de PD."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

from . import config


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """KS = max |TPR - FPR| (separação entre as distribuições de bons e maus pagadores)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(np.abs(tpr - fpr)))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Tabela de confiabilidade com bins de mesma frequência (decis da PD prevista)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE com bins de mesma frequência: média ponderada de |PD média - taxa observada|."""
    t = calibration_table(y, p, n_bins)
    return float(np.average((t["mean_pred"] - t["observed"]).abs(), weights=t["n"]))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_pd": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def risk_band(pd_value: float) -> tuple[str, str]:
    """Faixa de risco (letra, rótulo) para uma PD."""
    for letter, label, upper in config.RISK_BANDS:
        if pd_value < upper:
            return letter, label
    return config.RISK_BANDS[-1][0], config.RISK_BANDS[-1][1]


def band_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Por faixa: nº de clientes, PD média e taxa de inadimplência observada."""
    letters = [risk_band(v)[0] for v in np.asarray(p)]
    df = pd.DataFrame({"band": letters, "y": np.asarray(y), "p": np.asarray(p)})
    out = df.groupby("band").agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean")).reset_index()
    labels = {letter: label for letter, label, _ in config.RISK_BANDS}
    out["label"] = out["band"].map(labels)
    out["share"] = out["n"] / out["n"].sum()
    return out.sort_values("band").reset_index(drop=True)
