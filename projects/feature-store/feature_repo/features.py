"""Definições Feast: entidades, feature views batch (com event_timestamp) e o FeatureService de recompra.

Os dados de origem são gerados por `feature_store_demo.build` (sintéticos, schema Instacart).
"""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

DATA = Path(__file__).resolve().parent / "data"
TTL = timedelta(days=3650)  # dados de 2023-2024; o TTL cobre a janela de materialização

customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.INT64, description="Cliente")
department = Entity(name="department", join_keys=["department_id"], value_type=ValueType.INT64,
                    description="Departamento do catálogo")
product = Entity(name="product", join_keys=["product_id"], value_type=ValueType.INT64, description="Produto")


def _source(name: str) -> FileSource:
    return FileSource(
        name=f"{name}_source",
        path=str(DATA / f"{name}.parquet"),
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )


customer_order_stats = FeatureView(
    name="customer_order_stats",
    entities=[customer],
    ttl=TTL,
    schema=[
        Field(name="n_orders", dtype=Int64, description="Pedidos do cliente até o momento"),
        Field(name="n_items_total", dtype=Int64, description="Itens comprados até o momento"),
        Field(name="avg_basket_size", dtype=Float64, description="Itens por pedido"),
        Field(name="avg_days_between_orders", dtype=Float64, description="Intervalo médio entre pedidos (dias)"),
        Field(name="reorder_ratio", dtype=Float64, description="Fração de itens que são recompras"),
    ],
    source=_source("customer_order_stats"),
    description="Estatísticas de pedidos do cliente (estado após cada pedido).",
    tags={"owner": "ml-engineering", "kind": "batch"},
)

customer_department_affinity = FeatureView(
    name="customer_department_affinity",
    entities=[customer, department],
    ttl=TTL,
    schema=[
        Field(name="dept_purchases", dtype=Int64, description="Itens do departamento até o momento"),
        Field(name="dept_share", dtype=Float64, description="Participação do departamento nos itens do cliente"),
    ],
    source=_source("customer_department_affinity"),
    description="Afinidade do cliente por departamento (estado após cada pedido).",
    tags={"owner": "ml-engineering", "kind": "batch"},
)

customer_product_history = FeatureView(
    name="customer_product_history",
    entities=[customer, product],
    ttl=TTL,
    schema=[
        Field(name="times_bought", dtype=Int64, description="Vezes que o cliente comprou o produto até o momento"),
        Field(name="prev_purchase_gap_days", dtype=Float64, description="Dias entre as duas últimas compras do produto"),
    ],
    source=_source("customer_product_history"),
    description="Histórico de compra do cliente por produto (muda só quando o produto é comprado).",
    tags={"owner": "ml-engineering", "kind": "batch"},
)

recompra_v1 = FeatureService(
    name="recompra_v1",
    features=[customer_order_stats, customer_department_affinity, customer_product_history],
    description="Modelo de recompra: 'o cliente compra este produto no próximo pedido?'",
)
