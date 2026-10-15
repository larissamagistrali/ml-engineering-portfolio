"""Coleta e limpeza do UCI "Default of Credit Card Clients" (Taiwan, abr-set/2005; CC BY 4.0).

    PYTHONPATH=projects/credit-risk-mlops/src python -m credit_risk_mlops.data

Baixa o zip oficial para `data/raw/uci/` (ignorado pelo Git), lê o .xls (arquivo não confiável: só
é lido como dados, nunca executado), padroniza nomes/categorias e grava `data/credit_default.parquet`.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config

RENAME = {
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_1",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    **{f"BILL_AMT{i}": f"bill_{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"payamt_{i}" for i in range(1, 7)},
    "default payment next month": config.TARGET,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path) -> Path:
    """Baixa o zip (para .part e renomeia). Reaproveita se já existir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "default_of_credit_card_clients.zip"
    if target.exists():
        return target
    part = target.with_suffix(".part")
    req = urllib.request.Request(config.UCI_URL, headers={"User-Agent": "ml-engineering-portfolio (educational)"})
    with urllib.request.urlopen(req, timeout=120) as resp, part.open("wb") as out:  # noqa: S310 (URL fixa, https)
        out.write(resp.read())
    part.rename(target)
    return target


def read_raw(zip_path: Path) -> pd.DataFrame:
    """Lê a planilha do zip. Requer `xlrd` (formato .xls antigo)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xls")]
        if len(names) != 1:
            raise ValueError(f"esperava 1 arquivo .xls no zip, achei {names}")
        with zf.open(names[0]) as fh:
            return pd.read_excel(fh, header=1, engine="xlrd")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Padroniza nomes e trata categorias não documentadas; devolve (df, relatório de qualidade)."""
    df = raw.rename(columns=RENAME).drop(columns=["ID"])
    report: dict = {"n_rows_raw": int(len(df)), "n_duplicated_rows_kept": int(df.duplicated().sum())}
    # EDUCATION documentada: 1 pós, 2 universidade, 3 ensino médio, 4 outros. Há 0, 5 e 6 sem definição.
    undocumented_edu = ~df["education"].isin([1, 2, 3, 4])
    report["education_undocumented_to_4"] = {
        str(k): int(v) for k, v in df.loc[undocumented_edu, "education"].value_counts().sort_index().items()
    }
    df.loc[undocumented_edu, "education"] = 4
    # MARRIAGE documentada: 1 casado, 2 solteiro, 3 outros. Há 0 sem definição.
    undocumented_mar = ~df["marriage"].isin([1, 2, 3])
    report["marriage_undocumented_to_3"] = {
        str(k): int(v) for k, v in df.loc[undocumented_mar, "marriage"].value_counts().sort_index().items()
    }
    df.loc[undocumented_mar, "marriage"] = 3
    # PAY_x documentada: -1 em dia, 1..9 meses de atraso. Valores -2 e 0 existem sem definição oficial;
    # mantidos como estão (ordinais). Convenção comum: -2 sem consumo, 0 uso rotativo/pagamento mínimo.
    report["pay_status_values_not_in_documentation"] = {
        "-2": int((df[[f"pay_{i}" for i in range(1, 7)]] == -2).to_numpy().sum()),
        "0": int((df[[f"pay_{i}" for i in range(1, 7)]] == 0).to_numpy().sum()),
    }
    report["n_missing_values"] = int(df.isna().sum().sum())
    report["default_rate"] = round(float(df[config.TARGET].mean()), 4)
    df = df.astype({c: "int32" for c in df.columns})
    return df.reset_index(drop=True), report


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Hash do conteúdo (independe da serialização Parquet)."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def prepare(raw_dir: Path | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or config.DATA_DIR / "raw" / "uci"
    out_dir = out_dir or config.DATA_DIR
    zip_path = download(raw_dir)
    df, report = clean(read_raw(zip_path))
    report["source_zip_sha256"] = sha256_file(zip_path)
    report["dataset_fingerprint_sha256"] = dataset_fingerprint(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / config.CLEAN_FILE, compression="zstd", index=False)
    (out_dir / "data_quality.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return df


def load_clean(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or config.DATA_DIR / config.CLEAN_FILE)


if __name__ == "__main__":
    frame = prepare()
    print(f"{len(frame)} linhas, taxa de inadimplência {frame[config.TARGET].mean():.4f}")
