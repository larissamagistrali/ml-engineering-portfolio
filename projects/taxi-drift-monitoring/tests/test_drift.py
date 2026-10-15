"""Testes unitários: PSI, KS, limpeza e classificação de alertas (sem baixar dados)."""
import numpy as np
import pandas as pd
import pytest

from taxi_drift_monitoring import config
from taxi_drift_monitoring.data import clean_month
from taxi_drift_monitoring.drift import performance, psi_categorical, psi_level, psi_numeric


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi_numeric(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=5000), rng.normal(loc=1.0, size=5000)) > config.PSI_HIGH


def test_psi_categorical_handles_new_levels():
    ref = pd.Series(["a", "b"] * 50)
    cur = pd.Series(["a", "c"] * 50)
    assert psi_categorical(ref, cur) > 0


def test_psi_level_has_icon_and_text():
    icon, label = psi_level(0.25)
    assert icon and label == "alto"


def test_performance_mae_and_mape_filter():
    y = np.array([2.0, 10.0, 20.0])
    p = np.array([3.0, 10.0, 25.0])
    m = performance(y, p)
    assert m["mae"] == pytest.approx((1 + 0 + 5) / 3)
    assert m["mape_ge3min"] == pytest.approx((0 + 0.25) / 2)


def test_clean_month_drops_invalid_rows():
    pickup = pd.to_datetime(["2024-03-05 10:00", "2024-02-28 10:00", "2024-03-06 10:00"])
    df = pd.DataFrame({
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": pickup + pd.to_timedelta([10, 10, 500], unit="min"),
        "passenger_count": [1, 1, 1], "trip_distance": [2.0, 2.0, 2.0],
        "PULocationID": [161, 161, 161], "DOLocationID": [162, 162, 162], "payment_type": [1, 1, 1],
    })
    out = clean_month(df, "2024-03", {161: "Manhattan", 162: "Manhattan"})
    assert len(out) == 1  # fora do mês, e duração de 500 min são descartadas
    assert out.iloc[0]["pu_borough"] == "Manhattan"
