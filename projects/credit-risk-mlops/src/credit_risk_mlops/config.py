"""Constantes do projeto: caminhos, fonte dos dados, sementes e faixas de risco."""
from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MLRUNS_DIR = PROJECT_DIR / "mlruns"

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
UCI_DOI = "10.24432/C55S3H"
UCI_LICENSE = "CC BY 4.0"
UCI_CITATION = (
    "Yeh, I-C. & Lien, C-H. (2009). The comparisons of data mining techniques for the predictive "
    "accuracy of probability of default of credit card clients. Expert Systems with Applications, "
    "36(2), 2473-2480."
)

SEED = 42
N_MONTHS = 6
TARGET = "default"

CLEAN_FILE = "credit_default.parquet"
EVAL_FILE = "eval_test.parquet"
METRICS_FILE = "metrics.json"
MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"

# Faixas de risco por PD (limite superior exclusivo); validadas no teste (monotonia da taxa observada).
RISK_BANDS: list[tuple[str, str, float]] = [
    ("A", "Muito baixo", 0.10),
    ("B", "Baixo", 0.20),
    ("C", "Moderado", 0.35),
    ("D", "Alto", 0.55),
    ("E", "Muito alto", 1.01),
]
