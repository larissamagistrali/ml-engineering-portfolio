"""Modelo de duração da corrida (regressão LightGBM), treinado só no mês de referência."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from . import config

CATEGORY_LEVELS = {c: config.BOROUGHS for c in ("pu_borough", "do_borough")}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de features com categorias fixas (mesma codificação em todos os meses)."""
    X = df[config.FEATURES].copy()
    for col, levels in CATEGORY_LEVELS.items():
        X[col] = pd.Categorical(X[col], categories=levels)
    X["payment_type"] = X["payment_type"].astype("int16")
    return X


def train_reference(ref: pd.DataFrame, seed: int = config.SEED) -> tuple[LGBMRegressor, pd.DataFrame, pd.Series]:
    """Treina no mês de referência (80%) e devolve o modelo, o holdout (20%) e seu alvo."""
    tr, ho = train_test_split(ref, test_size=0.2, random_state=seed)
    model = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.9, random_state=seed, n_jobs=4, verbose=-1)
    model.fit(encode(tr), tr[config.TARGET])
    return model, ho.reset_index(drop=True), ho[config.TARGET].reset_index(drop=True)


def predict(model: LGBMRegressor, df: pd.DataFrame) -> np.ndarray:
    return model.predict(encode(df))
