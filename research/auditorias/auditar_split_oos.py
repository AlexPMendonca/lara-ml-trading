# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
LARA_DIR = REPO_DIR / "lara"
MODEL_DIR = LARA_DIR / "modelos" / "modelo_lara4"
OUT_DIR_AUD = REPO_DIR / "reports" / "auditorias"

DATASET = MODEL_DIR / "dataset_lara4_jan2026_shadow_pv058.csv"
REPORT = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_SIMPLIFICADO_V1.report.json"

OUT_DAY = OUT_DIR_AUD / "auditoria_split_por_dia.csv"
OUT_FEATURE = OUT_DIR_AUD / "auditoria_split_por_feature.csv"
OUT_DUP = OUT_DIR_AUD / "auditoria_split_duplicatas.csv"
OUT_NN = OUT_DIR_AUD / "auditoria_split_nearest_neighbor.csv"


def psi_numeric(train_s: pd.Series, test_s: pd.Series, bins: int = 10):
    a = pd.to_numeric(train_s, errors="coerce").dropna()
    b = pd.to_numeric(test_s, errors="coerce").dropna()

    if len(a) < 20 or len(b) < 10:
        return np.nan

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(a, qs))

    if len(edges) < 3:
        return np.nan

    a_bins = pd.cut(a, bins=edges, include_lowest=True)
    b_bins = pd.cut(b, bins=edges, include_lowest=True)

    a_pct = a_bins.value_counts(normalize=True, sort=False)
    b_pct = b_bins.value_counts(normalize=True, sort=False)

    idx = a_pct.index.union(b_pct.index)
    a_pct = a_pct.reindex(idx, fill_value=0.0001)
    b_pct = b_pct.reindex(idx, fill_value=0.0001)

    a_pct = a_pct.clip(lower=0.0001)
    b_pct = b_pct.clip(lower=0.0001)

    return float(((b_pct - a_pct) * np.log(b_pct / a_pct)).sum())


def describe_overlap(train_s: pd.Series, test_s: pd.Series):
    a = pd.to_numeric(train_s, errors="coerce").dropna()
    b = pd.to_numeric(test_s, errors="coerce").dropna()

    if a.empty or b.empty:
        return {
            "train_min": np.nan, "train_max": np.nan,
            "test_min": np.nan, "test_max": np.nan,
            "pct_test_inside_train_range": np.nan
        }

    inside = ((b >= a.min()) & (b <= a.max())).mean()

    return {
        "train_min": float(a.min()),
        "train_max": float(a.max()),
        "test_min": float(b.min()),
        "test_max": float(b.max()),
        "pct_test_inside_train_range": float(inside),
    }


def standardized_nearest_neighbor(train_df: pd.DataFrame, test_df: pd.DataFrame, features):
    tr = train_df[features].apply(pd.to_numeric, errors="coerce")
    te = test_df[features].apply(pd.to_numeric, errors="coerce")

    tr = tr.dropna().reset_index(drop=True)
    te = te.dropna().reset_index(drop=True)

    if tr.empty or te.empty:
        return pd.DataFrame()

    mu = tr.mean()
    sd = tr.std(ddof=0).replace(0, 1.0)

    tr_z = (tr - mu) / sd
    te_z = (te - mu) / sd

    tr_arr = tr_z.to_numpy(dtype=float)
    te_arr = te_z.to_numpy(dtype=float)

    rows = []
    for i in range(te_arr.shape[0]):
        d = np.sqrt(((tr_arr - te_arr[i]) ** 2).sum(axis=1))
        nn = float(d.min())
        q10 = float(np.quantile(d, 0.10))
        q50 = float(np.quantile(d, 0.50))
        rows.append({
            "test_row_idx": int(i),
            "nn_dist_min": nn,
            "dist_q10": q10,
            "dist_median": q50,
        })

    return pd.DataFrame(rows)


OUT_DIR_AUD.mkdir(parents=True, exist_ok=True)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET}")
    if not REPORT.exists():
        raise FileNotFoundError(f"Report não encontrado: {REPORT}")

    with open(REPORT, "r", encoding="utf-8") as f:
        rep = json.load(f)

    features = rep["training_summary"]["features"]
    test_days = set(rep["training_summary"]["dias_teste"])
    train_days = set(rep["training_summary"]["dias_treino"])

    df = pd.read_csv(DATASET)
    df["dia"] = df["dia"].astype(str)
    df["tipo"] = df["tipo"].astype(str).str.upper()
    df["split"] = np.where(df["dia"].isin(test_days), "test", "train")
    df["acerto"] = pd.to_numeric(df["acerto"], errors="coerce")
    df["lucro_real"] = pd.to_numeric(df["lucro_real"], errors="coerce")

    print("\n===== CHECAGEM CRONOLÓGICA =====")
    train_days_sorted = sorted(train_days)
    test_days_sorted = sorted(test_days)
    print("Dias treino:", train_days_sorted)
    print("Dias teste :", test_days_sorted)

    chrono_ok = (max(train_days_sorted) < min(test_days_sorted))
    overlap_days = sorted(train_days.intersection(test_days))
    print("Treino termina antes do teste começar? ", chrono_ok)
    print("Interseção treino/teste por dia       : ", overlap_days if overlap_days else "nenhuma")

    print("\n===== CONTAGEM POR DIA =====")
    by_day = (
        df.groupby(["split", "dia"], observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_mean=("acerto", "mean"),
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
            buy_pct=("tipo", lambda s: (s == "BUY").mean()),
            sell_pct=("tipo", lambda s: (s == "SELL").mean()),
        )
        .reset_index()
        .sort_values(["split", "dia"])
    )
    by_day.to_csv(OUT_DAY, index=False)
    print(by_day.to_string(index=False))

    print("\n===== DUPLICATAS EXATAS ENTRE TREINO E TESTE =====")
    sig_cols = ["dia", "tipo"] + [c for c in features if c in df.columns] + ["acerto"]
    sig_cols = [c for c in sig_cols if c in df.columns]

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    sig_no_day = ["tipo"] + [c for c in features if c in df.columns] + ["acerto"]
    sig_no_day = [c for c in sig_no_day if c in df.columns]

    train_sig = train_df[sig_no_day].astype(str).agg("|".join, axis=1)
    test_sig = test_df[sig_no_day].astype(str).agg("|".join, axis=1)

    dup_mask = test_sig.isin(set(train_sig))
    dup_df = test_df.loc[dup_mask, ["dia", "tipo", "acerto"] + [c for c in features if c in test_df.columns]].copy()
    dup_df.to_csv(OUT_DUP, index=False)

    print(f"Linhas teste: {len(test_df)}")
    print(f"Duplicatas exatas treino↔teste (sem usar 'dia'): {int(dup_mask.sum())}")
    if not dup_df.empty:
        print(dup_df.head(20).to_string(index=False))

    print("\n===== DRIFT POR FEATURE (train vs test) =====")
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue

        tr = train_df[feat]
        te = test_df[feat]

        ov = describe_overlap(tr, te)
        psi = psi_numeric(tr, te, bins=10)

        rows.append({
            "feature": feat,
            "psi_train_vs_test": psi,
            "train_mean": pd.to_numeric(tr, errors="coerce").mean(),
            "test_mean": pd.to_numeric(te, errors="coerce").mean(),
            "mean_diff_abs": abs(
                pd.to_numeric(te, errors="coerce").mean()
                - pd.to_numeric(tr, errors="coerce").mean()
            ),
            **ov
        })

    feat_df = pd.DataFrame(rows).sort_values(
        ["psi_train_vs_test", "mean_diff_abs"],
        ascending=[False, False]
    )
    feat_df.to_csv(OUT_FEATURE, index=False)
    print(feat_df.to_string(index=False))

    print("\n===== PROXIMIDADE TESTE ↔ TREINO (nearest neighbor padronizado) =====")
    nn_df = standardized_nearest_neighbor(train_df, test_df, [c for c in features if c in df.columns])
    nn_df.to_csv(OUT_NN, index=False)
    if nn_df.empty:
        print("Sem dados suficientes.")
    else:
        print(nn_df.describe().to_string())

    print("\n===== REGRAS DE LEITURA =====")
    print("1) chrono_ok=True e sem interseção por dia = split cronológico correto")
    print("2) duplicatas exatas > 0 = possível contaminação")
    print("3) PSI < 0.10  ~ drift pequeno")
    print("4) PSI 0.10-0.25 ~ drift moderado")
    print("5) PSI > 0.25  ~ drift forte")
    print("6) nn_dist_min muito perto de zero em muitas linhas = teste muito parecido com treino")
    print(f"\nArquivos gerados: {OUT_DAY}, {OUT_FEATURE}, {OUT_DUP}, {OUT_NN}")


if __name__ == "__main__":
    main()