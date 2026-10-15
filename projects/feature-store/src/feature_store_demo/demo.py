"""Demo de feature store: join point-in-time x join ingênuo, quantificação do vazamento, online e exports.

    PYTHONPATH=projects/feature-store/src python -m feature_store_demo.demo

Roda de verdade contra o repo Feast (`feature_repo/`, após `feast apply`). Escreve exports pequenos em
`data/` que o app lê sem importar o Feast.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "feature_repo"
DATA = REPO / "data"
OUT = ROOT / "data"
SERVICE = "recompra_v1"
FEATURES = ["n_orders", "n_items_total", "avg_basket_size", "avg_days_between_orders", "reorder_ratio",
            "dept_purchases", "dept_share", "times_bought", "prev_purchase_gap_days"]
VIEWS = {
    "customer_order_stats": ["customer_id"],
    "customer_department_affinity": ["customer_id", "department_id"],
    "customer_product_history": ["customer_id", "product_id"],
}
N_EXPORT_ROWS = 300


def naive_join(labels: pd.DataFrame) -> pd.DataFrame:
    """Join INGÊNUO: pega o ÚLTIMO estado de cada chave, ignorando o tempo do rótulo (vaza o futuro)."""
    out = labels[["customer_id", "product_id", "department_id", "event_timestamp", "label"]].copy()
    for view, keys in VIEWS.items():
        states = pd.read_parquet(DATA / f"{view}.parquet")
        latest = states.sort_values("event_timestamp").groupby(keys, as_index=False).last()
        cols = [c for c in latest.columns if c not in ("event_timestamp", "created_timestamp")]
        out = out.merge(latest[cols], on=keys, how="left")
    return out


def pit_join(store: FeatureStore, labels: pd.DataFrame) -> pd.DataFrame:
    """Join point-in-time do Feast: para cada linha, o último estado com event_timestamp <= ts do rótulo.

    Desambiguação: o join de arquivo do Feast 0.67 descarta linhas com a mesma (cliente, timestamp), que
    ocorrem quando vários produtos entram no mesmo pedido. Soma-se 1, 2, 3... µs ao timestamp da linha
    (menor que o intervalo entre pedidos, então o estado do pedido continua sendo o último válido). Um
    `row_id` devolve cada linha ao seu timestamp original.
    """
    base = labels[["customer_id", "product_id", "department_id", "event_timestamp", "label"]].copy()
    base["row_id"] = np.arange(len(base))
    offset = base.groupby(["customer_id", "event_timestamp"]).cumcount() + 1
    entity_df = base.copy()
    entity_df["event_timestamp"] = base["event_timestamp"] + pd.to_timedelta(offset, unit="us")
    job = store.get_historical_features(entity_df=entity_df, features=store.get_feature_service(SERVICE))
    out = job.to_df()
    out = out.set_index("row_id").loc[base["row_id"]].reset_index()
    out["event_timestamp"] = base["event_timestamp"].to_numpy()
    return out.drop(columns=["row_id"])


def auc_gap(pit: pd.DataFrame, naive: pd.DataFrame) -> dict:
    """Treina nos primeiros 70% do tempo e testa nos 30% finais, com cada conjunto de features."""
    cut = pit["event_timestamp"].quantile(0.7)
    res = {}
    for name, frame in (("point_in_time", pit), ("ingenuo", naive)):
        f = frame.copy()
        for c in FEATURES:
            f[c] = f[c].astype(float).fillna(0.0)
        tr, te = f[f["event_timestamp"] <= cut], f[f["event_timestamp"] > cut]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)
        clf.fit(tr[FEATURES], tr["label"])
        p = clf.predict_proba(te[FEATURES])[:, 1]
        res[name] = {"roc_auc_teste": round(float(roc_auc_score(te["label"], p)), 4),
                     "pr_auc_teste": round(float(average_precision_score(te["label"], p)), 4),
                     "n_treino": int(len(tr)), "n_teste": int(len(te))}
    return res


def leakage_stats(pit: pd.DataFrame, naive: pd.DataFrame) -> dict:
    same = {}
    for c in ("n_orders", "n_items_total", "times_bought", "dept_purchases"):
        a, b = pit[c].fillna(-1), naive[c].fillna(-1)
        same[c] = round(float((a != b).mean()), 4)
    return {
        "share_rows_different_value": same,
        "mean_n_orders_pit": round(float(pit["n_orders"].mean()), 3),
        "mean_n_orders_ingenuo": round(float(naive["n_orders"].mean()), 3),
        "mean_times_bought_pit": round(float(pit["times_bought"].fillna(0).mean()), 3),
        "mean_times_bought_ingenuo": round(float(naive["times_bought"].fillna(0).mean()), 3),
    }


def online_examples(store: FeatureStore, labels: pd.DataFrame, n: int = 5) -> list[dict]:
    sample = labels.sample(n=n, random_state=42)
    rows = [{"customer_id": int(r.customer_id), "product_id": int(r.product_id), "department_id": int(r.department_id)}
            for r in sample.itertuples()]
    resp = store.get_online_features(features=store.get_feature_service(SERVICE), entity_rows=rows).to_dict()
    out = []
    for i, row in enumerate(rows):
        values = {k: (v[i] if not isinstance(v, list) else v[i]) for k, v in resp.items() if k not in row}
        out.append({"entity": row, "features": {k: (None if v is None else float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                  for k, v in values.items()}})
    return out


def export_registry(store: FeatureStore) -> dict:
    """Registry legível: entidades, feature views (chaves, TTL, fonte, tipos) e o FeatureService."""
    views = []
    for fv in store.list_feature_views():
        views.append({
            "name": fv.name,
            "entities": [e for e in fv.entities],
            "join_keys": [f.name for f in fv.entity_columns],
            "ttl_days": fv.ttl.days if fv.ttl else None,
            "source_file": Path(fv.batch_source.path).name,
            "timestamp_field": fv.batch_source.timestamp_field,
            "created_timestamp_column": fv.batch_source.created_timestamp_column,
            "description": fv.description,
            "features": [{"name": f.name, "dtype": str(f.dtype).replace("Array(", "").rstrip(")"),
                          "description": f.description} for f in fv.features],
        })
    services = [{"name": s.name, "description": s.description,
                 "views": [p.name for p in s.feature_view_projections],
                 "features": [f"{p.name}:{f.name}" for p in s.feature_view_projections for f in p.features]}
                for s in store.list_feature_services()]
    entities = [{"name": e.name, "join_key": e.join_key, "value_type": str(e.value_type).replace("ValueType.", "")}
                for e in store.list_entities()]
    return {"project": store.project, "entities": entities, "feature_views": views, "feature_services": services,
            "source": "exportado via FeatureStore.list_entities / list_feature_views / list_feature_services"}


def run() -> dict:
    store = FeatureStore(repo_path=str(REPO))
    labels = pd.read_parquet(DATA / "label_rows.parquet")
    labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"])

    # Janela de materialização: do início dos dados até o último pedido.
    start = labels["event_timestamp"].min() - timedelta(days=1)
    end = pd.read_parquet(DATA / "customer_order_stats.parquet")["event_timestamp"].max() + timedelta(days=1)
    store.materialize(start_date=start.to_pydatetime(), end_date=end.to_pydatetime())

    pit = pit_join(store, labels)
    naive = naive_join(labels)
    assert len(pit) == len(labels) == len(naive)
    pit["event_timestamp"] = pd.to_datetime(pit["event_timestamp"])

    gap = auc_gap(pit, naive)
    leak = leakage_stats(pit, naive)
    results = {
        "service": SERVICE,
        "label_definition": "compra do produto no pedido seguinte (k+1), features do pedido k",
        "n_rows": int(len(labels)),
        "label_rate": round(float(labels["label"].mean()), 4),
        "auc": gap,
        "auc_gap_roc": round(gap["ingenuo"]["roc_auc_teste"] - gap["point_in_time"]["roc_auc_teste"], 4),
        "auc_gap_pr": round(gap["ingenuo"]["pr_auc_teste"] - gap["point_in_time"]["pr_auc_teste"], 4),
        "leakage": leak,
        "data_kind": "synthetic (schema Instacart; valores gerados, não reais)",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pit_vs_naive_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    # Exemplos para a página: todas as linhas de 40 clientes (point-in-time x ingênuo lado a lado).
    rng = np.random.default_rng(42)
    chosen = rng.choice(pit["customer_id"].unique(), size=40, replace=False)
    export = pit[pit["customer_id"].isin(chosen)].copy()
    naive_renamed = naive[["customer_id", "product_id", "event_timestamp", *FEATURES]].rename(
        columns={c: f"{c}_ingenuo" for c in FEATURES})
    ex = export.merge(naive_renamed, on=["customer_id", "product_id", "event_timestamp"], how="left")
    ex = ex.drop_duplicates(["customer_id", "product_id", "event_timestamp"]).sort_values(["customer_id", "event_timestamp"])
    ex.to_parquet(OUT / "pit_examples.parquet", compression="zstd", index=False)
    # Amostra do offline store (treino) em parquet, pequena
    pit.sample(n=min(2000, len(pit)), random_state=7).sort_values("event_timestamp").to_parquet(
        OUT / "offline_sample.parquet", compression="zstd", index=False)

    (OUT / "registry.json").write_text(json.dumps(export_registry(store), indent=2, ensure_ascii=False), encoding="utf-8")
    online = online_examples(store, labels)
    (OUT / "online_lookup.json").write_text(json.dumps({
        "note": "Lookup online após materialize(): valor mais recente materializado por entidade (não há tempo de consulta).",
        "materialized_until": str(end),
        "examples": online,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
