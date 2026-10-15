"""Inferência compartilhada por API e app: PD, faixa de risco, explicação por feature e avisos."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_LABELS, FEATURE_NAMES, FEATURE_SET_VERSION, build_features, raw_from_records
from .metrics import risk_band


class ModelLoadError(RuntimeError):
    """Artefato ausente ou incompatível com o código de features atual."""


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


@dataclass
class ModelBundle:
    """Modelo + model card. `model` é um LGBMClassifier ou um Pipeline(StandardScaler, LogisticRegression)."""

    model: Any
    card: dict

    def __post_init__(self) -> None:
        if self.card.get("features") != FEATURE_NAMES or self.card.get("feature_set_version") != FEATURE_SET_VERSION:
            raise ModelLoadError(
                "features do model card diferem das do código (train/serve skew): re-treine o modelo."
            )

    @property
    def version(self) -> str:
        return str(self.card["model_version"])

    def predict_pd(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features[FEATURE_NAMES])[:, 1]

    def contributions(self, features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Contribuição de cada feature em log-odds, relativa ao valor-base: logit(PD) = base + soma.

        LightGBM: TreeSHAP exato nativo (`pred_contrib=True`). Logística: coef x valor padronizado (exato).
        """
        X = features[FEATURE_NAMES]
        if hasattr(self.model, "booster_"):
            raw = self.model.booster_.predict(X, pred_contrib=True)
            return raw[:, -1], pd.DataFrame(raw[:, :-1], columns=FEATURE_NAMES, index=features.index)
        scaler, clf = self.model.named_steps["scale"], self.model.named_steps["clf"]
        z = scaler.transform(X)
        contrib = z * clf.coef_[0]
        return np.full(len(X), float(clf.intercept_[0])), pd.DataFrame(contrib, columns=FEATURE_NAMES, index=features.index)


def load_bundle(directory: Path | None = None) -> ModelBundle:
    directory = directory or config.ARTIFACTS_DIR
    model_path, card_path = directory / config.MODEL_FILE, directory / config.CARD_FILE
    if not model_path.exists() or not card_path.exists():
        raise ModelLoadError(f"artefatos não encontrados em {directory} (rode o treino)")
    try:
        model = joblib.load(model_path)
    except Exception as exc:  # versão de lib incompatível, arquivo corrompido etc.
        raise ModelLoadError(f"falha ao carregar {model_path.name}: {exc}") from exc
    return ModelBundle(model, json.loads(card_path.read_text(encoding="utf-8")))


def input_warnings(raw: pd.DataFrame, card: Mapping[str, Any]) -> list[list[str]]:
    """Avisos por linha quando algum valor está fora da faixa vista no treino."""
    ranges = card.get("input_ranges", {})
    groups = {"limit_bal": ["limit_bal"], "age": ["age"], "bill_amt": [c for c in raw.columns if c.startswith("bill_")],
              "pay_amt": [c for c in raw.columns if c.startswith("payamt_")]}
    out: list[list[str]] = [[] for _ in range(len(raw))]
    for name, cols in groups.items():
        if name not in ranges:
            continue
        lo, hi = ranges[name]
        bad = ((raw[cols] < lo) | (raw[cols] > hi)).any(axis=1).to_numpy()
        for i in np.flatnonzero(bad):
            out[i].append(f"{name} fora da faixa observada no treino ({lo:g} a {hi:g}); estimativa menos confiável")
    return out


def score_records(
    bundle: ModelBundle, records: Sequence[Mapping[str, Any]], explain: bool = True, top_k: int = 5
) -> list[dict[str, Any]]:
    """Pontua registros no formato do contrato da API e devolve dicts prontos para serializar."""
    raw = raw_from_records(records)
    feats = build_features(raw)
    pds = bundle.predict_pd(feats)
    warns = input_warnings(raw, bundle.card)
    contrib = bundle.contributions(feats)[1] if explain else None
    results = []
    for i, p in enumerate(pds):
        letter, label = risk_band(float(p))
        item: dict[str, Any] = {
            "pd": round(float(p), 4),
            "risk_band": letter,
            "risk_band_label": label,
            "warnings": warns[i],
        }
        if contrib is not None:
            row = contrib.iloc[i]
            top = row.reindex(row.abs().sort_values(ascending=False).index)[:top_k]
            item["top_factors"] = [
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "value": round(float(feats.iloc[i][name]), 4),
                    "contribution_logit": round(float(val), 4),
                    "direction": "increases_risk" if val > 0 else "decreases_risk",
                }
                for name, val in top.items()
            ]
        results.append(item)
    return results
