"""SIMULAÇÃO (não é o dado real): perturbação controlada para demonstrar o detector de drift.

Pega o mês 2024-06 e altera 30% das corridas: distância x1,5, hora deslocada +3h e pagamento
convertido para 'cartão' (1). Os números saem em arquivo separado e rotulados como simulação.

    PYTHONPATH=projects/taxi-drift-monitoring/src python -m taxi_drift_monitoring.simulate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .data import load_samples
from .drift import feature_drift, performance
from .model import predict, train_reference

SIM_MONTH = "2024-06"
SHARE = 0.30


def perturb(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = out.index[rng.random(len(out)) < SHARE]
    out.loc[idx, "trip_distance"] = out.loc[idx, "trip_distance"] * 1.5
    out.loc[idx, "pickup_hour"] = (out.loc[idx, "pickup_hour"] + 3) % 24
    out.loc[idx, "payment_type"] = 1.0
    return out


def run(samples: pd.DataFrame | None = None) -> dict:
    df = samples if samples is not None else load_samples()
    ref = df[df["month"] == config.REFERENCE_MONTH].reset_index(drop=True)
    model, _, _ = train_reference(ref)
    cur = df[df["month"] == SIM_MONTH].reset_index(drop=True)
    sim = perturb(cur)
    fd = feature_drift(ref, sim)
    fd.insert(0, "scenario", "simulação: 30% com distância x1,5, hora +3h e pagamento cartão")
    perf_real = performance(cur[config.TARGET], predict(model, cur))
    perf_sim = performance(sim[config.TARGET], predict(model, sim))  # alvo inalterado: só as entradas mudam
    out = {
        "label": "SIMULAÇÃO: entradas perturbadas de forma controlada; não representa mudança real observada",
        "base_month": SIM_MONTH,
        "share_perturbed": SHARE,
        "psi_by_feature": fd[["feature", "psi"]].round(4).to_dict(orient="records"),
        "mae_real": perf_real["mae"],
        "mae_with_perturbed_inputs": perf_sim["mae"],
    }
    fd.to_parquet(config.DATA_DIR / "simulation_drift.parquet", compression="zstd", index=False)
    (config.DATA_DIR / "simulation_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
