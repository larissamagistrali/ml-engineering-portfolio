"""Constrói as tabelas de features (estados com event_timestamp) e as linhas de rótulo.

Definição do alvo: para o cliente c, no pedido k (features conhecidas em ts_k), o rótulo é 1 se o
produto p aparecer no pedido k+1. Só entram produtos já comprados até o pedido k.

Estados: cada linha de feature vale a partir do horário do pedido em que foi calculada. O join
point-in-time pega, para cada rótulo em ts_k, o último estado com event_timestamp <= ts_k.

    PYTHONPATH=projects/feature-store/src python -m feature_store_demo.build
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2] / "feature_repo"
RAW = REPO / "data" / "raw"
DATA = REPO / "data"
LABEL_SAMPLE = 60_000
SEED = 42


def load_raw(raw: Path = RAW) -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(raw / f"{n}.parquet") for n in ("orders", "order_products", "products")}


def build_states(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Estados por cliente (ordem, cesta, reorder), por cliente x departamento e por cliente x produto."""
    orders = raw["orders"].sort_values(["user_id", "order_number"]).reset_index(drop=True)
    items = raw["order_products"].merge(raw["products"][["product_id", "department_id"]], on="product_id")
    items = items.merge(orders[["order_id", "user_id", "order_number", "order_timestamp"]], on="order_id")
    items = items.sort_values(["user_id", "order_number", "add_to_cart_order"])

    cust_rows, dept_rows, prod_rows = [], [], []
    for uid, u_orders in orders.groupby("user_id", sort=False):
        u_items = items[items["user_id"] == uid]
        n_items_cum, reordered_cum = 0, 0
        dept_cnt: dict[int, int] = {}
        prod_cnt: dict[int, int] = {}
        prod_last: dict[int, list] = {}
        gaps: list[float] = []
        prev_ts = None
        for _, o in u_orders.iterrows():
            oid, k, ts = o["order_id"], int(o["order_number"]), o["order_timestamp"]
            if prev_ts is not None:
                gaps.append((ts - prev_ts).total_seconds() / 86400)
            prev_ts = ts
            cur = u_items[u_items["order_id"] == oid]
            n_items_cum += len(cur)
            reordered_cum += int(cur["reordered"].sum())
            for d in cur["department_id"]:
                dept_cnt[int(d)] = dept_cnt.get(int(d), 0) + 1
            for p in cur["product_id"]:
                p = int(p)
                prod_cnt[p] = prod_cnt.get(p, 0) + 1
                prod_last.setdefault(p, []).append(ts)
            cust_rows.append({
                "customer_id": int(uid), "event_timestamp": ts,
                "n_orders": k, "n_items_total": int(n_items_cum),
                "avg_basket_size": n_items_cum / k,
                "avg_days_between_orders": float(np.mean(gaps)) if gaps else 0.0,
                "reorder_ratio": reordered_cum / n_items_cum,
            })
            for d, c in dept_cnt.items():
                dept_rows.append({"customer_id": int(uid), "department_id": d, "event_timestamp": ts,
                                  "dept_purchases": c, "dept_share": c / n_items_cum})
            # estado de produto só muda quando o produto é comprado neste pedido
            for p in cur["product_id"].unique():
                p = int(p)
                times = prod_cnt[p]
                gap = (prod_last[p][-1] - prod_last[p][-2]).total_seconds() / 86400 if times > 1 else 0.0
                prod_rows.append({"customer_id": int(uid), "product_id": p, "event_timestamp": ts,
                                  "times_bought": times, "prev_purchase_gap_days": gap})
    cust = pd.DataFrame(cust_rows)
    dept = pd.DataFrame(dept_rows)
    prod = pd.DataFrame(prod_rows)
    for df in (cust, dept, prod):
        df["created_timestamp"] = df["event_timestamp"]
    return {"customer_order_stats": cust, "customer_department_affinity": dept, "customer_product_history": prod}


def build_labels(raw: dict[str, pd.DataFrame], states: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Linhas (cliente, produto, ts_k, rótulo = compra no pedido k+1), amostradas com semente fixa."""
    orders = raw["orders"].sort_values(["user_id", "order_number"])
    items = raw["order_products"].merge(raw["products"][["product_id", "department_id"]], on="product_id")
    items = items.merge(orders[["order_id", "user_id", "order_number", "order_timestamp"]], on="order_id")
    nxt = items[["user_id", "order_number", "product_id"]].copy()
    nxt["order_number"] = nxt["order_number"] - 1  # produto do pedido k+1 associado ao pedido k
    nxt["bought_next"] = 1
    nxt = nxt.drop_duplicates()
    cand = items[["user_id", "order_number", "order_id", "product_id", "department_id", "order_timestamp"]].copy()
    # produtos comprados até o pedido k (histórico acumulado)
    hist = items.sort_values("order_number")[["user_id", "order_number", "product_id"]].drop_duplicates(["user_id", "product_id"])
    rows = []
    for uid, g in orders.groupby("user_id"):
        max_k = int(g["order_number"].max())
        h = hist[hist["user_id"] == uid]
        for k in range(1, max_k):
            known = h[h["order_number"] <= k]["product_id"].unique()
            ts = orders[(orders["user_id"] == uid) & (orders["order_number"] == k)]["order_timestamp"].iloc[0]
            rows.extend((int(uid), int(p), ts, k) for p in known)
    lab = pd.DataFrame(rows, columns=["customer_id", "product_id", "event_timestamp", "order_number"])
    lab = lab.merge(nxt.rename(columns={"user_id": "customer_id"}), on=["customer_id", "order_number", "product_id"], how="left")
    lab["label"] = lab["bought_next"].fillna(0).astype("int8")
    lab = lab.drop(columns=["bought_next"])
    lab = lab.merge(raw["products"][["product_id", "department_id"]], on="product_id", how="left")
    lab = lab.sample(n=min(LABEL_SAMPLE, len(lab)), random_state=SEED).sort_values("event_timestamp").reset_index(drop=True)
    lab["event_timestamp"] = lab["event_timestamp"].astype("datetime64[us]")
    return lab


def run(raw_dir: Path = RAW, out_dir: Path = DATA) -> dict[str, int]:
    raw = load_raw(raw_dir)
    states = build_states(raw)
    labels = build_labels(raw, states)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in states.items():
        df = df.copy()
        df["event_timestamp"] = df["event_timestamp"].astype("datetime64[us]")
        df["created_timestamp"] = df["created_timestamp"].astype("datetime64[us]")
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        counts[name] = len(df)
    labels.to_parquet(out_dir / "label_rows.parquet", index=False)
    counts["label_rows"] = len(labels)
    counts["label_positive_rate"] = round(float(labels["label"].mean()), 4)
    return counts


if __name__ == "__main__":
    print(run())
