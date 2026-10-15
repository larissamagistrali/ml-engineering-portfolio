"""Pipeline de treino com MLflow: baseline vs LightGBM, registro, alias `champion` e exportação.

    PYTHONPATH=projects/credit-risk-mlops/src python -m credit_risk_mlops.train

Fluxo: dados limpos -> split estratificado 60/20/20 -> baseline logístico -> busca aleatória de
LightGBM (parada antecipada na validação) -> registra os dois no Model Registry -> promove a
`champion` só se passar no gate -> exporta `model.joblib` + `model_card.json` do campeão registrado.
Requer `mlflow` (venv do projeto; não faz parte do requirements.txt do app).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from . import MODEL_NAME, config  # noqa: E402
from .data import dataset_fingerprint, load_clean  # noqa: E402
from .features import FEATURE_NAMES, FEATURE_SET_VERSION  # noqa: E402
from .metrics import band_table, calibration_table, evaluate  # noqa: E402
from .modeling import fit_lgbm, make_baseline, sample_params, split_data, xy  # noqa: E402
from .serving import ModelBundle  # noqa: E402

EXPERIMENT = "credit-risk-pd"
MIN_AUC_GAIN = 0.005  # gate de promoção: ganho mínimo de ROC-AUC (validação) sobre o baseline


def _lib_versions() -> dict[str, str]:
    names = ["lightgbm", "scikit-learn", "mlflow", "pandas", "numpy"]
    return {"python": platform.python_version(), **{n: metadata.version(n) for n in names}}


def setup_mlflow() -> MlflowClient:
    """Tracking local: SQLite + artefatos em `mlruns/` (ignorado pelo Git)."""
    config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.MLRUNS_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(config.MLRUNS_DIR / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


def _importance(bundle: ModelBundle, X: pd.DataFrame) -> list[dict[str, Any]]:
    _, contrib = bundle.contributions(X)
    imp = contrib.abs().mean().sort_values(ascending=False)
    return [{"feature": k, "mean_abs_logit_contribution": round(float(v), 5)} for k, v in imp.head(15).items()]


def _fairness(test: pd.DataFrame, p: np.ndarray) -> dict[str, Any]:
    """Auditoria simples por grupo (sexo e faixa etária) no teste. Sexo NÃO é feature do modelo."""
    df = test[[config.TARGET, "sex", "age"]].copy()
    df["pd"] = p
    df["sexo"] = df["sex"].map({1: "masculino", 2: "feminino"})
    df["faixa_etaria"] = pd.cut(df["age"], [0, 29, 44, 200], labels=["até 29", "30 a 44", "45+"])
    out: dict[str, Any] = {}
    for col in ("sexo", "faixa_etaria"):
        out[col] = [
            {
                "group": str(name),
                "n": int(len(g)),
                "observed_default_rate": round(float(g[config.TARGET].mean()), 4),
                "mean_pd": round(float(g["pd"].mean()), 4),
                "roc_auc": round(float(roc_auc_score(g[config.TARGET], g["pd"])), 4),
            }
            for name, g in df.groupby(col, observed=True)
        ]
    return out


def _input_ranges(df: pd.DataFrame) -> dict[str, list[int]]:
    bills = df[[f"bill_{i}" for i in range(1, 7)]]
    pays = df[[f"payamt_{i}" for i in range(1, 7)]]
    return {
        "limit_bal": [int(df["limit_bal"].min()), int(df["limit_bal"].max())],
        "age": [int(df["age"].min()), int(df["age"].max())],
        "bill_amt": [int(bills.min().min()), int(bills.max().max())],
        "pay_amt": [int(pays.min().min()), int(pays.max().max())],
    }


def build_card(*, key: str, r: dict, base: dict, df: pd.DataFrame, parts: dict[str, pd.DataFrame], Xte: pd.DataFrame,
               yte: np.ndarray, quality: dict, n_trials: int) -> dict[str, Any]:
    """Model card de um candidato (r = resultados do candidato; base = resultados do baseline)."""
    card: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_version": f"{MODEL_NAME}:v{r['version']}",
        "algorithm": "lightgbm" if key == "lightgbm" else "logistic_regression",
        "task": "classificação binária: probabilidade de inadimplência no mês seguinte (PD)",
        "registry": {"backend": "mlflow (sqlite local)", "alias": "champion", "version": r["version"], "run_id": r["run_id"]},
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": config.SEED,
        "libraries": _lib_versions(),
        "data": {
            "name": "UCI Default of Credit Card Clients (Taiwan, 2005)",
            "source": config.UCI_PAGE,
            "license": config.UCI_LICENSE,
            "doi": config.UCI_DOI,
            "n_rows": int(len(df)),
            "default_rate": round(float(df[config.TARGET].mean()), 4),
            "fingerprint_sha256": dataset_fingerprint(df),
            "source_zip_sha256": quality.get("source_zip_sha256"),
            "split": {
                "strategy": "estratificado por inadimplência (o dataset não tem data da solicitação: não há corte temporal)",
                **{k: int(len(v)) for k, v in parts.items()},
            },
        },
        "features": FEATURE_NAMES,
        "feature_set_version": FEATURE_SET_VERSION,
        "excluded_features": {"sex": "atributo protegido", "marriage": "atributo protegido"},
        "input_ranges": _input_ranges(df),
        "hyperparameters": r["params"],
        "n_estimators": r["n_est"],
        "selection": {
            "criterion": f"log-loss na validação (menor é melhor); gate de promoção: ROC-AUC de validação >= baseline + {MIN_AUC_GAIN} "
            "e log-loss menor que o do baseline",
            "n_trials": n_trials,
        },
        "metrics": {"valid": r["valid"], "test": r["test"]},
        "baseline_metrics": {"valid": base["valid"], "test": base["test"]},
        "risk_bands": [{"band": b, "label": lab, "pd_upper_exclusive": min(up, 1.0)} for b, lab, up in config.RISK_BANDS],
        "band_performance_test": band_table(yte, r["pt"]).round(4).to_dict(orient="records"),
        "fairness_audit_test": _fairness(parts["test"], r["pt"]),
        "intended_use": "Demonstração educacional de um pipeline de ML em produção (treino, registro, serving, monitoramento).",
        "out_of_scope": "Qualquer decisão real de crédito. O dataset é de 2005 e de um único emissor em Taiwan.",
        "limitations": [
            "Dados de um único emissor/país/período (abr-set/2005): sem garantia de validade em outros contextos.",
            "Sem data de solicitação: a avaliação é estratificada, não fora do tempo (tende a ser otimista).",
            "Só clientes que receberam crédito aparecem nos dados (viés de seleção; sem inferência de rejeitados).",
            "Idade e escolaridade entram no modelo e podem atuar como proxies; sexo e estado civil foram excluídos.",
            "Probabilidades não foram recalibradas depois do treino; ver ECE e a curva de calibração.",
        ],
    }
    card["global_importance_test"] = _importance(ModelBundle(r["model"], card), Xte)
    return card


def _fit_and_log(key: str, model: Any, params: dict, extra: dict, data: dict, tags: dict, client: MlflowClient) -> dict:
    """Run aninhado do candidato: parâmetros, métricas (valid/test), modelo e registro no Model Registry."""
    Xva, yva, Xte, yte = data["Xva"], data["yva"], data["Xte"], data["yte"]
    pv = model.predict_proba(Xva[FEATURE_NAMES])[:, 1]
    pt = model.predict_proba(Xte[FEATURE_NAMES])[:, 1]
    mva, mte = evaluate(yva, pv), evaluate(yte, pt)
    with mlflow.start_run(run_name=f"candidate-{key}", nested=True, tags={**tags, "candidate": key}) as run:
        mlflow.log_params({**params, **extra})
        mlflow.log_metrics({f"valid_{k}": v for k, v in mva.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in mte.items()})
        flavor = mlflow.lightgbm if key == "lightgbm" else mlflow.sklearn  # cada modelo com o seu flavor nativo
        flavor.log_model(
            model, name="model", signature=infer_signature(Xva[FEATURE_NAMES].head(200), pv[:200]),
            input_example=Xva[FEATURE_NAMES].head(3), registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id
    version = max(client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'"), key=lambda v: int(v.version)).version
    for tag, value in (("candidate", key), ("valid_log_loss", f"{mva['log_loss']:.6f}"), ("valid_roc_auc", f"{mva['roc_auc']:.6f}")):
        client.set_model_version_tag(MODEL_NAME, version, tag, value)
    return {"model": model, "valid": mva, "test": mte, "pv": pv, "pt": pt, "run_id": run_id, "version": version,
            "params": params, "n_est": extra.get("n_estimators_used")}


def train(n_trials: int = 12) -> dict[str, Any]:
    client = setup_mlflow()
    df = load_clean()
    quality_path = config.DATA_DIR / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    parts = split_data(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = xy(parts["train"]), xy(parts["valid"]), xy(parts["test"])
    data = {"Xva": Xva, "yva": yva, "Xte": Xte, "yte": yte}
    tags = {"data_fingerprint": dataset_fingerprint(df)[:16], "feature_set_version": FEATURE_SET_VERSION}

    with mlflow.start_run(run_name="pipeline", tags=tags) as parent:
        mlflow.log_params({"seed": config.SEED, "n_trials": n_trials, "n_rows": len(df), **{f"n_{k}": len(v) for k, v in parts.items()}})

        trials: list[dict[str, Any]] = []
        best: tuple[float, dict, Any] | None = None
        for i, params in enumerate(sample_params(n_trials)):
            model = fit_lgbm(Xtr, ytr, params, Xva, yva)
            m = evaluate(yva, model.predict_proba(Xva[FEATURE_NAMES])[:, 1])
            used = int(model.best_iteration_ or model.n_estimators)
            with mlflow.start_run(run_name=f"lgbm-trial-{i:02d}", nested=True, tags=tags):
                mlflow.log_params({**params, "n_estimators_used": used})
                mlflow.log_metrics({f"valid_{k}": v for k, v in m.items()})
            trials.append({"trial": i, **params, "n_estimators_used": used,
                           **{f"valid_{k}": round(v, 5) for k, v in m.items() if k in ("roc_auc", "ks", "log_loss", "brier")}})
            if best is None or m["log_loss"] < best[0]:
                best = (m["log_loss"], params, model)
        assert best is not None
        _, best_params, best_model = best
        used = int(best_model.best_iteration_ or best_model.n_estimators)

        baseline = make_baseline().fit(Xtr[FEATURE_NAMES], ytr)
        res = {
            "baseline": _fit_and_log("baseline", baseline, {"C": 0.5}, {"algorithm": "logistic_regression"}, data, tags, client),
            "lightgbm": _fit_and_log("lightgbm", best_model, best_params, {"algorithm": "lightgbm", "n_estimators_used": used},
                                     data, tags, client),
        }
        client.set_registered_model_alias(MODEL_NAME, "baseline", res["baseline"]["version"])

        base, cand = res["baseline"], res["lightgbm"]
        passes = (cand["valid"]["roc_auc"] >= base["valid"]["roc_auc"] + MIN_AUC_GAIN
                  and cand["valid"]["log_loss"] < base["valid"]["log_loss"])
        winner = "lightgbm" if passes else "baseline"
        try:
            current_ll = float(client.get_model_version_by_alias(MODEL_NAME, "champion").tags.get("valid_log_loss", "inf"))
        except MlflowException:
            current_ll = float("inf")  # ainda não há campeão
        promoted = res[winner]["valid"]["log_loss"] < current_ll
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, "champion", res[winner]["version"])
        mlflow.log_params({"winner": winner, "promoted": promoted})

        for key, r in res.items():
            card = build_card(key=key, r=r, base=base, df=df, parts=parts, Xte=Xte, yte=yte, quality=quality, n_trials=n_trials)
            client.log_dict(r["run_id"], card, "model_card.json")
        parent_id = parent.info.run_id

    exported = export_champion(client)
    write_reports(res, trials, winner, promoted, yte, parts, parent_id)
    return {"winner": winner, "promoted_to_champion": promoted, "exported": exported}


def export_champion(client: MlflowClient, out_dir: Path | None = None) -> dict[str, str]:
    """Carrega o campeão PELO ALIAS do registry e grava model.joblib + model_card.json."""
    out_dir = out_dir or config.ARTIFACTS_DIR
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    card = json.loads(Path(client.download_artifacts(mv.run_id, "model_card.json")).read_text(encoding="utf-8"))
    if card["registry"]["version"] != mv.version:
        raise RuntimeError("model card não corresponde à versão campeã do registry")
    loader = mlflow.lightgbm if card["algorithm"] == "lightgbm" else mlflow.sklearn
    model = loader.load_model(f"models:/{MODEL_NAME}@champion")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / config.MODEL_FILE, compress=3)
    (out_dir / config.CARD_FILE).write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"model_version": card["model_version"], "algorithm": card["algorithm"]}


def write_reports(res: dict, trials: list, winner: str, promoted: bool, y_test: np.ndarray, parts: dict, parent_id: str) -> None:
    """Tabelas pequenas lidas pelo app: métricas, calibração, faixas, busca de hiperparâmetros e predições de teste."""
    out = {
        "winner": winner,
        "promoted_to_champion": promoted,
        "mlflow_parent_run_id": parent_id,
        "candidates": {k: {"valid": r["valid"], "test": r["test"], "registry_version": r["version"]} for k, r in res.items()},
        "trials": trials,
        "calibration_test": {k: calibration_table(y_test, r["pt"]).round(5).to_dict(orient="records") for k, r in res.items()},
        "bands_test": {k: band_table(y_test, r["pt"]).round(5).to_dict(orient="records") for k, r in res.items()},
        "split_sizes": {k: int(len(v)) for k, v in parts.items()},
    }
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / config.METRICS_FILE).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    ev = pd.DataFrame({"y": y_test.astype("int8"), **{f"pd_{k}": r["pt"].astype("float32") for k, r in res.items()}})
    ev.to_parquet(config.DATA_DIR / config.EVAL_FILE, compression="zstd", index=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Treina, registra (MLflow) e exporta o modelo de PD.")
    ap.add_argument("--n-trials", type=int, default=12, help="tentativas da busca aleatória de LightGBM")
    args = ap.parse_args(argv)
    print(json.dumps(train(args.n_trials), indent=2, ensure_ascii=False))
    print(f"MLflow UI: mlflow ui --backend-store-uri sqlite:///{config.MLRUNS_DIR / 'mlflow.db'}", file=sys.stderr)


if __name__ == "__main__":
    main()
