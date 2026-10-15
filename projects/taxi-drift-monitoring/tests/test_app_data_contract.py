"""Garante que os artefatos que o app lê existem e têm o formato esperado."""
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def test_metrics_and_tables_exist():
    s = json.loads((DATA / "metrics.json").read_text(encoding="utf-8"))
    assert s["reference_month"] == "2024-01" and len(s["months"]) == 6
    perf = pd.read_parquet(DATA / "performance_by_month.parquet")
    drift = pd.read_parquet(DATA / "drift_by_month.parquet")
    assert {"month", "mae", "mae_ratio", "alert_performance"} <= set(perf.columns)
    assert {"month", "feature", "psi"} <= set(drift.columns)


def test_simulation_is_labelled():
    s = json.loads((DATA / "simulation_metrics.json").read_text(encoding="utf-8"))
    assert s["label"].startswith("SIMULAÇÃO")
