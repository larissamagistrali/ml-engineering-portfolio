"""Constantes do projeto: meses, amostragem, limiares de alerta e caminhos."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
# Cache bruto. Por padrão fica no próprio projeto (gitignored). Pode-se apontar para um cache compartilhado.
RAW_DIR = Path(os.environ.get("TLC_RAW_DIR", PROJECT_DIR / "data" / "raw" / "tlc"))
WORK_DIR = PROJECT_DIR / "data" / "work"  # amostras (gitignored); o app não lê daqui
REPORTS_DIR = PROJECT_DIR / "data" / "reports"  # relatórios Evidently sob demanda (gitignored)

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
REFERENCE_MONTH = "2024-01"
SAMPLE_SIZE = 50_000
SEED = 42

TARGET = "duration_min"
NUMERIC = ["trip_distance", "passenger_count", "pickup_hour", "pickup_dow"]
CATEGORICAL = ["pu_borough", "do_borough", "payment_type"]
FEATURES = NUMERIC + CATEGORICAL

# Limiares (convenção usual do PSI) e de alerta de desempenho.
PSI_MODERATE = 0.10
PSI_HIGH = 0.20
MAE_ALERT_RATIO = 1.10  # alerta se MAE do mês > 110% do MAE de referência

BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR", "Unknown"]
