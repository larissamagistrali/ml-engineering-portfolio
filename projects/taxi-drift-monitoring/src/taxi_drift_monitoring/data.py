"""Download (cache bruto com .part), limpeza e amostragem das corridas do NYC TLC.

    PYTHONPATH=projects/taxi-drift-monitoring/src python -m taxi_drift_monitoring.data

Os arquivos são baixados para `*.part` e renomeados ao final. Se o arquivo final já existe, é reutilizado.
Os arquivos brutos são dados não confiáveis: são lidos como tabela, nunca executados.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

COLUMNS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance",
    "PULocationID", "DOLocationID", "payment_type",
]


def fetch(url: str, target: Path) -> Path:
    """Baixa `url` para `target` (via `.part`) se ainda não existir."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    part = target.with_name(target.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ml-engineering-portfolio (educational; NYC TLC open data)"})
    with urllib.request.urlopen(req, timeout=300) as resp, part.open("wb") as out:  # noqa: S310 (URL fixa, https)
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    part.rename(target)
    return target


def load_zones(raw_dir: Path) -> pd.DataFrame:
    path = fetch(f"{config.BASE_URL}/misc/taxi_zone_lookup.csv", raw_dir / "taxi_zone_lookup.csv")
    zones = pd.read_csv(path)
    zones["Borough"] = zones["Borough"].fillna("Unknown").where(zones["Borough"].isin(config.BOROUGHS), "Unknown")
    return zones[["LocationID", "Borough"]]


def clean_month(df: pd.DataFrame, month: str, borough_map: dict[int, str]) -> pd.DataFrame:
    """Limpeza básica: intervalo de tempo coerente com o mês do arquivo, distâncias e zonas válidas."""
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
    out = pd.DataFrame({
        "pickup": pickup,
        "duration_min": (dropoff - pickup).dt.total_seconds() / 60.0,
        "trip_distance": df["trip_distance"].astype("float64"),
        "passenger_count": df["passenger_count"].astype("float64"),
        "PULocationID": df["PULocationID"].astype("int64"),
        "DOLocationID": df["DOLocationID"].astype("int64"),
        "payment_type": df["payment_type"].astype("float64"),
    })
    keep = (
        out["pickup"].dt.strftime("%Y-%m").eq(month)
        & out["duration_min"].between(1, 120)
        & out["trip_distance"].between(0.1, 60)
        & out["passenger_count"].between(1, 6)
        & out["payment_type"].isin([1, 2, 3, 4, 5, 6])
    )
    out = out.loc[keep].copy()
    out["pickup_hour"] = out["pickup"].dt.hour.astype("int16")
    out["pickup_dow"] = out["pickup"].dt.dayofweek.astype("int16")
    out["pu_borough"] = out["PULocationID"].map(borough_map).fillna("Unknown")
    out["do_borough"] = out["DOLocationID"].map(borough_map).fillna("Unknown")
    out["month"] = month
    return out.drop(columns=["pickup", "PULocationID", "DOLocationID"])


def sample_month(path: Path, month: str, borough_map: dict[int, str], n: int = config.SAMPLE_SIZE,
                 seed: int = config.SEED) -> tuple[pd.DataFrame, dict]:
    """Lê um mês, limpa e amostra `n` corridas com semente fixa. Devolve (amostra, contagens)."""
    table = pq.read_table(path, columns=COLUMNS)
    raw_n = table.num_rows
    cleaned = clean_month(table.to_pandas(), month, borough_map)
    sample = cleaned.sample(n=min(n, len(cleaned)), random_state=seed).sort_index()
    info = {"month": month, "rows_raw": int(raw_n), "rows_clean": int(len(cleaned)), "rows_sample": int(len(sample))}
    return sample.reset_index(drop=True), info


def build_samples(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.WORK_DIR) -> pd.DataFrame:
    zones = load_zones(raw_dir)
    borough_map = dict(zip(zones["LocationID"], zones["Borough"]))
    frames, infos = [], []
    for month in config.MONTHS:
        path = fetch(f"{config.BASE_URL}/trip-data/yellow_tripdata_{month}.parquet", raw_dir / f"yellow_tripdata_{month}.parquet")
        sample, info = sample_month(path, month, borough_map)
        frames.append(sample)
        infos.append(info)
    all_samples = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_samples.to_parquet(out_dir / "samples_2024H1.parquet", compression="zstd", index=False)
    (out_dir / "sampling_log.json").write_text(json.dumps(infos, indent=2), encoding="utf-8")
    return all_samples


def load_samples(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or config.WORK_DIR / "samples_2024H1.parquet")


if __name__ == "__main__":
    s = build_samples()
    print(s.groupby("month").size().to_string())
    print(np.round(s[config.TARGET].describe(), 2).to_string())
