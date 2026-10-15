"""Relatório HTML do Evidently (drift de dados + predição + regressão) de um mês contra a referência.

    PYTHONPATH=projects/taxi-drift-monitoring/src python -m taxi_drift_monitoring.report 2024-06 [--commit]

Sem --commit grava em data/reports/ (gitignored). Com --commit grava em data/evidently/ (versionado,
~3,7 MB: o HTML embute o JavaScript do Evidently; por isso há só um relatório versionado).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset

from . import config
from .data import load_samples
from .model import predict, train_reference

COMMITTED_DIR = config.PROJECT_DIR / "data" / "evidently"


def _dataset(df: pd.DataFrame, dd: DataDefinition) -> Dataset:
    return Dataset.from_pandas(df[config.FEATURES + [config.TARGET, "prediction"]].copy(), data_definition=dd)


def build(month: str, out_dir: Path = config.REPORTS_DIR, samples: pd.DataFrame | None = None) -> Path:
    df = samples if samples is not None else load_samples()
    ref = df[df["month"] == config.REFERENCE_MONTH].reset_index(drop=True)
    cur = df[df["month"] == month].reset_index(drop=True)
    model, _, _ = train_reference(ref)
    ref["prediction"] = predict(model, ref)
    cur["prediction"] = predict(model, cur)
    dd = DataDefinition(
        numerical_columns=config.NUMERIC + [config.TARGET, "prediction"],
        categorical_columns=config.CATEGORICAL,
        regression=[Regression(target=config.TARGET, prediction="prediction")],
    )
    snap = Report([DataDriftPreset(), RegressionPreset()]).run(
        current_data=_dataset(cur, dd), reference_data=_dataset(ref, dd)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"evidently_{config.REFERENCE_MONTH}_vs_{month}.html"
    snap.save_html(str(path))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("month", nargs="?", default="2024-06")
    ap.add_argument("--commit", action="store_true", help="grava em data/evidently/ (versionado)")
    args = ap.parse_args()
    print(build(args.month, COMMITTED_DIR if args.commit else config.REPORTS_DIR))
