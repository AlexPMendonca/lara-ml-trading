# -*- coding: utf-8 -*-
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
LARA_DIR = REPO_DIR / "lara"
MODEL_DIR = LARA_DIR / "modelos" / "modelo_lara4"
OUT_DIR_AUD = REPO_DIR / "reports" / "auditorias"

DATASET = MODEL_DIR / "dataset_lara4_jan2026_shadow_pv058.csv"
REPORT  = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_SIMPLIFICADO_V1.report.json"
MODEL   = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_SIMPLIFICADO_V1.pkl"

OUT_GLOBAL = OUT_DIR_AUD / "calibracao_v1_global.csv"
OUT_SIDE   = OUT_DIR_AUD / "calibracao_v1_side.csv"


def pf_from_series(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    pos = float(s[s > 0].sum())
    neg = float(-s[s < 0].sum())
    if neg == 0:
        return float("inf") if pos > 0 else np.nan
    return pos / neg


def summarize(df: pd.DataFrame, split: str, mode: str, thr_buy: float, thr_sell: float) -> dict:
    d = df[df["split"] == split].copy()

    if mode == "global":
        keep = d["prob_acerto"] >= thr_buy
    elif mode == "side":
        keep = np.where(
            d["tipo"].eq("BUY"),
            d["prob_acerto"] >= thr_buy,
            d["prob_acerto"] >= thr_sell
        )
        keep = pd.Series(keep, index=d.index)
    else:
        raise ValueError("mode inválido")

    kept = d[keep].copy()

    row = {
        "split": split,
        "mode": mode,
        "thr_buy": float(thr_buy),
        "thr_sell": float(thr_sell),
        "rows_total_split": int(len(d)),
        "rows_kept": int(len(kept)),
        "keep_rate": float(len(kept) / len(d)) if len(d) else np.nan,
    }

    if kept.empty:
        row.update({
            "acerto_mean": np.nan,
            "lucro_total": 0.0,
            "lucro_medio": np.nan,
            "pf": np.nan,
            "buy_rows": 0,
            "sell_rows": 0,
            "buy_pct": np.nan,
            "sell_pct": np.nan,
            "prob_mean": np.nan,
            "confidence_mean": np.nan,
        })
        return row

    row.update({
        "acerto_mean": float(pd.to_numeric(kept["acerto"], errors="coerce").mean()),
        "lucro_total": float(pd.to_numeric(kept["lucro_real"], errors="coerce").sum()),
        "lucro_medio": float(pd.to_numeric(kept["lucro_real"], errors="coerce").mean()),
        "pf": float(pf_from_series(kept["lucro_real"])),
        "buy_rows": int((kept["tipo"] == "BUY").sum()),
        "sell_rows": int((kept["tipo"] == "SELL").sum()),
        "buy_pct": float((kept["tipo"] == "BUY").mean()),
        "sell_pct": float((kept["tipo"] == "SELL").mean()),
        "prob_mean": float(pd.to_numeric(kept["prob_acerto"], errors="coerce").mean()),
        "confidence_mean": float(pd.to_numeric(kept["confidence"], errors="coerce").mean()),
    })
    return row


def pivot_compare(dfres: pd.DataFrame) -> pd.DataFrame:
    train = dfres[dfres["split"] == "train"].copy()
    test = dfres[dfres["split"] == "test"].copy()

    key = ["mode", "thr_buy", "thr_sell"]
    cols = [
        "rows_kept", "keep_rate", "acerto_mean", "lucro_total", "lucro_medio",
        "pf", "buy_rows", "sell_rows", "buy_pct", "sell_pct",
        "prob_mean", "confidence_mean"
    ]

    train = train[key + cols].rename(columns={c: f"{c}_train" for c in cols})
    test  = test[key + cols].rename(columns={c: f"{c}_test" for c in cols})

    out = train.merge(test, on=key, how="outer")

    out["gap_acerto"] = out["acerto_mean_train"] - out["acerto_mean_test"]
    out["gap_keep"]   = out["keep_rate_train"] - out["keep_rate_test"]
    out["gap_pf"]     = out["pf_train"] - out["pf_test"]

    return out


OUT_DIR_AUD.mkdir(parents=True, exist_ok=True)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    if not REPORT.exists():
        raise FileNotFoundError(REPORT)
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    with open(REPORT, "r", encoding="utf-8") as f:
        rep = json.load(f)

    with open(MODEL, "rb") as f:
        model = pickle.load(f)

    features = rep["training_summary"]["features"]
    test_days = set(rep["training_summary"]["dias_teste"])

    df = pd.read_csv(DATASET)
    df["dia"] = df["dia"].astype(str)
    df["tipo"] = df["tipo"].astype(str).str.upper()
    df["acerto"] = pd.to_numeric(df["acerto"], errors="coerce").astype(int)
    df["lucro_real"] = pd.to_numeric(df["lucro_real"], errors="coerce")
    df["split"] = np.where(df["dia"].isin(test_days), "test", "train")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Há NaNs nas features do V1.")

    prob = model.predict_proba(X)[:, 1]
    df["prob_acerto"] = prob
    df["confidence"] = (df["prob_acerto"] - 0.5).abs() * 2.0

    print("\n===== BASE V1 =====")
    print("features =", features)
    print("dias_teste =", sorted(test_days))
    print(df.groupby("split").size().to_string())

    # 1) Sweep global
    rows_global = []
    for thr in np.arange(0.50, 0.82, 0.02):
        thr = round(float(thr), 2)
        rows_global.append(summarize(df, "train", "global", thr, thr))
        rows_global.append(summarize(df, "test",  "global", thr, thr))

    res_global = pd.DataFrame(rows_global)
    cmp_global = pivot_compare(res_global)
    cmp_global.to_csv(OUT_GLOBAL, index=False)

    print("\n===== TOP GLOBAL (ordenado por lucro_total_test, depois pf_test) =====")
    show_global = cmp_global.sort_values(
        ["lucro_total_test", "pf_test", "acerto_mean_test", "rows_kept_test"],
        ascending=[False, False, False, False]
    )
    print(show_global.head(15).to_string(index=False))

    # 2) Sweep por lado
    rows_side = []
    thrs_buy = [round(x, 2) for x in np.arange(0.50, 0.82, 0.04)]
    thrs_sell = [round(x, 2) for x in np.arange(0.50, 0.82, 0.04)]

    for tb in thrs_buy:
        for ts in thrs_sell:
            rows_side.append(summarize(df, "train", "side", tb, ts))
            rows_side.append(summarize(df, "test",  "side", tb, ts))

    res_side = pd.DataFrame(rows_side)
    cmp_side = pivot_compare(res_side)
    cmp_side.to_csv(OUT_SIDE, index=False)

    print("\n===== TOP SIDE (ordenado por lucro_total_test, depois pf_test) =====")
    show_side = cmp_side.sort_values(
        ["lucro_total_test", "pf_test", "acerto_mean_test", "rows_kept_test"],
        ascending=[False, False, False, False]
    )
    print(show_side.head(20).to_string(index=False))

    # 3) Faixa minimamente útil
    filt_global = cmp_global[
        (cmp_global["rows_kept_train"] >= 150) &
        (cmp_global["rows_kept_test"] >= 10)
    ].copy()

    filt_side = cmp_side[
        (cmp_side["rows_kept_train"] >= 150) &
        (cmp_side["rows_kept_test"] >= 10)
    ].copy()

    print("\n===== GLOBAL FILTRADO (train>=150 trades e test>=10 trades) =====")
    if filt_global.empty:
        print("Sem configuração global dentro do filtro.")
    else:
        print(
            filt_global.sort_values(
                ["lucro_total_test", "pf_test", "acerto_mean_test"],
                ascending=[False, False, False]
            ).head(10).to_string(index=False)
        )

    print("\n===== SIDE FILTRADO (train>=150 trades e test>=10 trades) =====")
    if filt_side.empty:
        print("Sem configuração side dentro do filtro.")
    else:
        print(
            filt_side.sort_values(
                ["lucro_total_test", "pf_test", "acerto_mean_test"],
                ascending=[False, False, False]
            ).head(15).to_string(index=False)
        )

    print("\nArquivos gerados:")
    print("-", OUT_GLOBAL)
    print("-", OUT_SIDE)


if __name__ == "__main__":
    main()