"""Partição dos dados e construção dos modelos (baseline logístico e LightGBM).

Mantido sem MLflow/FastAPI para que o app Streamlit consiga re-treinar rapidamente se o `.joblib`
não carregar na versão de bibliotecas do ambiente.
"""
from __future__ import annotations

import inspect
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURE_NAMES, build_features


def split_data(df: pd.DataFrame, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """Treino/validação/teste (60/20/20), estratificado pelo alvo.

    O dataset não tem data da solicitação, então não há como fazer um corte temporal: a partição
    é estratificada e aleatória (fixa pela semente).
    """
    train, rest = train_test_split(df, test_size=0.4, stratify=df[config.TARGET], random_state=seed)
    valid, test = train_test_split(rest, test_size=0.5, stratify=rest[config.TARGET], random_state=seed)
    return {"train": train.reset_index(drop=True), "valid": valid.reset_index(drop=True), "test": test.reset_index(drop=True)}


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return build_features(df), df[config.TARGET].to_numpy()


def make_baseline() -> Pipeline:
    """Regressão logística com padronização (baseline interpretável)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=3000, random_state=config.SEED)),
    ])


SEARCH_SPACE: dict[str, list[Any]] = {
    "learning_rate": [0.02, 0.03, 0.05],
    "num_leaves": [7, 15, 31],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0],
}


def sample_params(n: int, seed: int = config.SEED) -> list[dict[str, Any]]:
    return list(ParameterSampler(SEARCH_SPACE, n_iter=n, random_state=seed))


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict[str, Any],
    X_valid: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    n_estimators: int = 2000,
) -> lgb.LGBMClassifier:
    """Treina um LGBMClassifier; com validação, usa parada antecipada (100 rodadas, log-loss)."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        subsample_freq=1,
        random_state=config.SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=4,
        verbose=-1,
        **params,
    )
    if X_valid is not None and y_valid is not None:
        # lightgbm >= 4.7 trocou `eval_set` por `eval_X`/`eval_y` (o antigo emite DeprecationWarning).
        if "eval_X" in inspect.signature(model.fit).parameters:
            evals: dict[str, Any] = {"eval_X": X_valid[FEATURE_NAMES], "eval_y": y_valid}
        else:
            evals = {"eval_set": [(X_valid[FEATURE_NAMES], y_valid)]}
        model.fit(
            X_train[FEATURE_NAMES],
            y_train,
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
            **evals,
        )
    else:
        model.fit(X_train[FEATURE_NAMES], y_train)
    return model


def retrain_champion(card: dict, df: pd.DataFrame) -> Any:
    """Reconstrói o campeão a partir dos hiperparâmetros do model card e do mesmo split."""
    parts = split_data(df, card.get("seed", config.SEED))
    X, y = xy(parts["train"])
    if card["algorithm"] == "lightgbm":
        return fit_lgbm(X, y, card["hyperparameters"], n_estimators=card["n_estimators"])
    return make_baseline().fit(X[FEATURE_NAMES], y)
