"""Pipeline de monitoramento: referência = mês 1; pontua os meses 2 a 6 e grava tabelas pequenas.

    PYTHONPATH=projects/taxi-drift-monitoring/src python -m taxi_drift_monitoring.monitor
"""
from __future__ import annotations

import json

import pandas as pd
from scipy.stats import ks_2samp

from . import config
from .data import load_samples
from .drift import feature_drift, performance, psi_numeric
from .model import predict, train_reference


def run(samples: pd.DataFrame | None = None) -> dict:
    df = samples if samples is not None else load_samples()
    ref_all = df[df["month"] == config.REFERENCE_MONTH].reset_index(drop=True)
    model, ho, ho_y = train_reference(ref_all)
    ho_pred = predict(model, ho)
    ref_perf = performance(ho_y, ho_pred)
    ref_pred = pd.Series(ho_pred)

    drift_rows, perf_rows = [], []
    perf_rows.append({"month": config.REFERENCE_MONTH, "role": "referência (holdout 20%)", "n": len(ho), **ref_perf,
                      "psi_prediction": 0.0, "ks_prediction": 0.0})
    for month in config.MONTHS[1:]:
        cur = df[df["month"] == month].reset_index(drop=True)
        fd = feature_drift(ref_all, cur)
        fd.insert(0, "month", month)
        drift_rows.append(fd)
        pred = predict(model, cur)
        p = performance(cur[config.TARGET], pred)
        psi_pred = psi_numeric(ho_pred, pred)
        ks_pred = ks_2samp(ho_pred, pred).statistic
        perf_rows.append({"month": month, "role": "monitorado", "n": len(cur), **p,
                          "psi_prediction": psi_pred, "ks_prediction": float(ks_pred)})

    drift = pd.concat(drift_rows, ignore_index=True)
    perf = pd.DataFrame(perf_rows)
    perf["mae_ratio"] = perf["mae"] / ref_perf["mae"]
    perf["alert_performance"] = perf["mae_ratio"] > config.MAE_ALERT_RATIO
    max_psi = drift.groupby("month")["psi"].max()
    perf["max_feature_psi"] = perf["month"].map(max_psi).fillna(0.0)
    perf["n_features_psi_high"] = perf["month"].map(drift[drift["psi"] >= config.PSI_HIGH].groupby("month").size()).fillna(0).astype(int)
    perf["alert_data"] = perf["n_features_psi_high"] > 0
    perf["alert_prediction"] = perf["psi_prediction"] >= config.PSI_HIGH

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    drift.to_parquet(config.DATA_DIR / "drift_by_month.parquet", compression="zstd", index=False)
    perf.to_parquet(config.DATA_DIR / "performance_by_month.parquet", compression="zstd", index=False)

    summary = {
        "reference_month": config.REFERENCE_MONTH,
        "sample_size_per_month": config.SAMPLE_SIZE,
        "seed": config.SEED,
        "model": "LGBMRegressor(n_estimators=400, lr=0.05, num_leaves=63)",
        "features": config.FEATURES,
        "target": config.TARGET,
        "reference_mae": ref_perf["mae"],
        "thresholds": {"psi_moderate": config.PSI_MODERATE, "psi_high": config.PSI_HIGH,
                       "mae_ratio_alert": config.MAE_ALERT_RATIO},
        "months": perf.drop(columns=["role"]).round(5).to_dict(orient="records"),
        "max_psi_by_month": {k: round(float(v), 4) for k, v in max_psi.items()},
    }
    (config.DATA_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps({"reference_mae": round(s["reference_mae"], 3), "months": [
        {k: m[k] for k in ("month", "n", "mae", "mae_ratio", "max_feature_psi", "psi_prediction", "alert_performance")}
        for m in s["months"]]}, indent=2, ensure_ascii=False))
