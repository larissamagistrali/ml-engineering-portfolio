"""Testes da feature store: exports que o app lê e a verificação point-in-time contra cálculo manual."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPO_DATA = ROOT / "feature_repo" / "data"


def test_exports_exist_and_are_small():
    for name in ("registry.json", "pit_vs_naive_metrics.json", "online_lookup.json", "pit_examples.parquet"):
        assert (DATA / name).exists(), name
    total = sum(p.stat().st_size for p in DATA.glob("*") if p.is_file())
    assert total < 5 * 1024 * 1024


def test_registry_has_ttl_and_source_for_every_view():
    reg = json.loads((DATA / "registry.json").read_text(encoding="utf-8"))
    assert len(reg["feature_views"]) == 3
    for v in reg["feature_views"]:
        assert v["ttl_days"] and v["source_file"].endswith(".parquet") and v["timestamp_field"] == "event_timestamp"
        assert v["features"]


def test_leakage_is_quantified_and_positive():
    m = json.loads((DATA / "pit_vs_naive_metrics.json").read_text(encoding="utf-8"))
    assert m["auc_gap_roc"] > 0
    assert m["leakage"]["mean_n_orders_ingenuo"] > m["leakage"]["mean_n_orders_pit"]


def test_pit_examples_are_point_in_time_consistent():
    """Em cada linha exportada, o valor PIT de n_orders não pode passar do número de pedidos até o momento."""
    ex = pd.read_parquet(DATA / "pit_examples.parquet")
    assert (ex["n_orders"] <= ex["n_orders_ingenuo"]).all()


@pytest.mark.skipif(not REPO_DATA.exists(), reason="dados do feature repo ausentes (rode build e feast apply)")
def test_feast_point_in_time_matches_manual_asof():
    pytest.importorskip("feast")
    import numpy as np

    from feature_store_demo.build import load_raw
    from feature_store_demo.demo import DATA as RDATA, REPO, pit_join
    from feast import FeatureStore

    store = FeatureStore(repo_path=str(REPO))
    labels = pd.read_parquet(RDATA / "label_rows.parquet")
    labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"])
    sample = labels.sample(100, random_state=3).reset_index(drop=True)
    got = pit_join(store, sample)
    raw = load_raw(REPO / "data" / "raw")
    items = raw["order_products"].merge(raw["orders"][["order_id", "user_id", "order_number"]], on="order_id")
    expected = [int(((items.user_id == r.customer_id) & (items.product_id == r.product_id)
                     & (items.order_number <= r.order_number)).sum()) for r in sample.itertuples()]
    assert np.array_equal(got["times_bought"].fillna(0).astype(int).to_numpy(), np.array(expected))
