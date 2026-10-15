"""Testes do pipeline de risco de crédito (features, métricas, serving e API)."""
import json

import numpy as np
import pandas as pd
import pytest

from credit_risk_mlops import config
from credit_risk_mlops.features import FEATURE_NAMES, build_features, raw_from_records
from credit_risk_mlops.metrics import evaluate, ks_statistic, risk_band

REC = {
    "limit_bal": 120000, "age": 34, "education": 2,
    "pay_status": [0, 0, 0, 0, 0, 0],
    "bill_amt": [41000, 39500, 38200, 36900, 35000, 33800],
    "pay_amt": [2500, 2300, 2200, 2000, 2000, 1900],
}


def test_features_shape_and_order():
    feats = build_features(raw_from_records([REC]))
    assert list(feats.columns) == FEATURE_NAMES
    assert np.isfinite(feats.to_numpy()).all()


def test_raw_from_records_rejects_wrong_length():
    bad = {**REC, "pay_status": [0, 0]}
    with pytest.raises(ValueError):
        raw_from_records([bad])


def test_ratio_handles_zero_limit_and_no_bill():
    rec = {**REC, "limit_bal": 10000, "bill_amt": [0, 0, 0, 0, 0, 0], "pay_amt": [0] * 6}
    feats = build_features(raw_from_records([rec])).iloc[0]
    assert feats["util_1"] == 0.0
    assert feats["pay_ratio_1"] == 1.0


def test_risk_band_monotone():
    letters = [risk_band(p)[0] for p in np.linspace(0, 1, 101)]
    assert letters == sorted(letters)
    assert risk_band(0.05)[0] == "A" and risk_band(0.99)[0] == "E"


def test_ks_and_metrics_on_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, p)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert ks_statistic(y, p) == pytest.approx(1.0)
    assert 0 <= m["brier"] <= 1


def test_undocumented_categories_are_mapped():
    from credit_risk_mlops.data import clean

    raw = pd.DataFrame({
        "ID": [1, 2], "LIMIT_BAL": [1, 2], "SEX": [1, 2], "EDUCATION": [0, 5], "MARRIAGE": [0, 1], "AGE": [20, 30],
        **{f"PAY_{i}": [0, 0] for i in (0, 2, 3, 4, 5, 6)},
        **{f"BILL_AMT{i}": [0, 0] for i in range(1, 7)}, **{f"PAY_AMT{i}": [0, 0] for i in range(1, 7)},
        "default payment next month": [0, 1],
    })
    raw = raw.rename(columns={"PAY_0": "PAY_0"})
    df, report = clean(raw)
    assert set(df["education"].unique()) <= {1, 2, 3, 4}
    assert set(df["marriage"].unique()) <= {1, 2, 3}
    assert report["education_undocumented_to_4"] == {"0": 1, "5": 1}


# --- artefatos exportados (exigem `make train-credit-risk`) ---
ARTIFACTS = config.ARTIFACTS_DIR / config.MODEL_FILE
needs_artifacts = pytest.mark.skipif(not ARTIFACTS.exists(), reason="rode o treino antes (make train-credit-risk)")


@needs_artifacts
def test_model_card_is_consistent_with_features():
    card = json.loads((config.ARTIFACTS_DIR / config.CARD_FILE).read_text(encoding="utf-8"))
    assert card["features"] == FEATURE_NAMES
    assert card["registry"]["alias"] == "champion"
    assert card["metrics"]["test"]["roc_auc"] > 0.7


@needs_artifacts
def test_contributions_sum_to_logit():
    from credit_risk_mlops.serving import load_bundle

    b = load_bundle()
    feats = build_features(raw_from_records([REC]))
    base, contrib = b.contributions(feats)
    logit = np.log(b.predict_pd(feats) / (1 - b.predict_pd(feats)))
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(), logit, rtol=1e-3, atol=1e-3)


@needs_artifacts
def test_api_contract():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from credit_risk_mlops.api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        r = client.post("/predict", json=REC)
        assert r.status_code == 200
        body = r.json()
        assert body["model_version"].startswith("credit-risk-pd:v")
        assert 0 <= body["prediction"]["pd"] <= 1
        assert len(body["prediction"]["top_factors"]) == 5
        bad = client.post("/predict", json={**REC, "age": 5})
        assert bad.status_code == 422
        extra = client.post("/predict", json={**REC, "sex": 1})
        assert extra.status_code == 422
        batch = client.post("/predict/batch", json={"customers": [REC, REC]})
        assert batch.status_code == 200 and batch.json()["n"] == 2
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422
