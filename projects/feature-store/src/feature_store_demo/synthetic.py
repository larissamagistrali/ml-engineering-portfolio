"""Gerador SINTÉTICO com o schema do Instacart (orders, order_products, products, aisles, departments).

O Instacart Market Basket Analysis é Kaggle-only (exige login), então os dados são gerados aqui com
semente fixa. Os nomes de tabelas e colunas seguem o original; os valores não são reais.

    PYTHONPATH=projects/feature-store/src python -m feature_store_demo.synthetic
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 3000
N_PRODUCTS = 2000
N_AISLES = 134
DEPARTMENTS = [
    "frozen", "other", "bakery", "produce", "alcohol", "international", "beverages", "pets",
    "dry goods pasta", "bulk", "personal care", "meat seafood", "pantry", "breakfast", "canned goods",
    "dairy eggs", "household", "babies", "snacks", "deli", "missing",
]
START = pd.Timestamp("2023-01-01")


def generate(out_dir: Path, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    departments = pd.DataFrame({"department_id": np.arange(1, len(DEPARTMENTS) + 1), "department": DEPARTMENTS})
    aisles = pd.DataFrame({"aisle_id": np.arange(1, N_AISLES + 1),
                           "aisle": [f"corredor {i}" for i in range(1, N_AISLES + 1)]})
    aisle_dept = rng.integers(1, len(DEPARTMENTS) + 1, size=N_AISLES)
    product_aisle = rng.integers(1, N_AISLES + 1, size=N_PRODUCTS)
    products = pd.DataFrame({
        "product_id": np.arange(1, N_PRODUCTS + 1),
        "product_name": [f"produto sintético {i}" for i in range(1, N_PRODUCTS + 1)],
        "aisle_id": product_aisle,
        "department_id": aisle_dept[product_aisle - 1],
    })
    # Popularidade tipo Zipf: poucos produtos concentram as compras.
    pop = 1.0 / np.arange(1, N_PRODUCTS + 1) ** 0.8
    pop = pop / pop.sum()

    orders, items = [], []
    order_id = 0
    for cust in range(1, N_CUSTOMERS + 1):
        n_orders = 3 + rng.poisson(6)
        favorites = rng.choice(N_PRODUCTS, size=25, replace=False, p=pop)
        ts = START + pd.Timedelta(days=int(rng.integers(0, 180)), hours=int(rng.integers(0, 24)))
        bought: set[int] = set()
        for k in range(1, n_orders + 1):
            if k > 1:
                ts = ts + pd.Timedelta(days=float(min(60, max(1, rng.gamma(2.0, 7.0)))))
            order_id += 1
            n_items = 2 + rng.poisson(2.5)
            chosen: list[int] = []
            for _ in range(n_items * 3):
                if len(chosen) >= n_items:
                    break
                src = rng.random()
                if src < 0.55:
                    p = int(rng.choice(favorites)) + 1
                else:
                    p = int(rng.choice(N_PRODUCTS, p=pop)) + 1
                if p not in chosen:
                    chosen.append(p)
            orders.append({"order_id": order_id, "user_id": cust, "order_number": k, "order_timestamp": ts})
            for pos, p in enumerate(chosen, start=1):
                items.append({"order_id": order_id, "product_id": p, "add_to_cart_order": pos,
                              "reordered": int(p in bought)})
            bought.update(chosen)
    orders_df = pd.DataFrame(orders)
    orders_df["days_since_prior_order"] = orders_df.groupby("user_id")["order_timestamp"].diff().dt.total_seconds() / 86400
    orders_df["days_since_prior_order"] = orders_df["days_since_prior_order"].round(2)
    order_products = pd.DataFrame(items)
    tables = {"orders": orders_df, "order_products": order_products, "products": products,
              "aisles": aisles, "departments": departments}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return tables


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2] / "feature_repo" / "data" / "raw"
    t = generate(here)
    print({k: len(v) for k, v in t.items()}, "linhas")
