# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_DIR = Path(__file__).resolve().parents[2]
LARA_DIR = REPO_DIR / "lara"
MODEL_DIR = LARA_DIR / "modelos" / "modelo_lara4"
OUT_DIR_AUD = REPO_DIR / "reports" / "auditorias"

DATASET = MODEL_DIR / "dataset_lara4_jan2026_shadow_pv058.csv"
REPORT = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_SIMPLIFICADO_V1.report.json"

OUT_RANK = OUT_DIR_AUD / "auditoria_features_ranking.csv"
OUT_SIDE_REGIME = OUT_DIR_AUD / "auditoria_test_lado_regime.csv"
OUT_DAY = OUT_DIR_AUD / "auditoria_test_por_dia.csv"
OUT_BUCKET_BUY = OUT_DIR_AUD / "auditoria_bucket_buy_pc.csv"
OUT_BUCKET_SELL = OUT_DIR_AUD / "auditoria_bucket_sell_pv.csv"


def safe_auc(y, x):
    s = pd.to_numeric(x, errors="coerce")
    y = pd.Series(y)

    mask = s.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan

    y2 = y[mask].astype(int)
    x2 = s[mask].astype(float)

    if y2.nunique() < 2 or x2.nunique() < 2:
        return np.nan

    try:
        return float(roc_auc_score(y2, x2))
    except Exception:
        return np.nan


def auc_eff(auc):
    if pd.isna(auc):
        return np.nan
    return max(float(auc), 1.0 - float(auc))


def split_stats(df, split_name):
    d = df[df["split"] == split_name].copy()
    if d.empty:
        return None
    return {
        "split": split_name,
        "rows": int(len(d)),
        "dias": sorted(d["dia"].astype(str).unique().tolist()),
        "acerto_mean": float(d["acerto"].mean()),
        "lucro_medio": float(pd.to_numeric(d["lucro_real"], errors="coerce").mean()),
        "buy_pct": float((d["tipo"].astype(str).str.upper() == "BUY").mean()),
        "sell_pct": float((d["tipo"].astype(str).str.upper() == "SELL").mean()),
    }


def bucket_table(df, feature, n_quant=4):
    d = df.copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d[d[feature].notna()].copy()

    if d.empty or d[feature].nunique() < 2:
        return pd.DataFrame()

    q = min(n_quant, int(d[feature].nunique()))
    if q < 2:
        return pd.DataFrame()

    d["bucket"] = pd.qcut(d[feature], q=q, duplicates="drop")
    out = (
        d.groupby("bucket", observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_mean=("acerto", "mean"),
            lucro_total=("lucro_real", lambda s: pd.to_numeric(s, errors="coerce").sum()),
            lucro_medio=("lucro_real", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            feat_min=(feature, "min"),
            feat_max=(feature, "max"),
        )
        .reset_index()
    )
    return out


OUT_DIR_AUD.mkdir(parents=True, exist_ok=True)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET}")

    if not REPORT.exists():
        raise FileNotFoundError(f"Report não encontrado: {REPORT}")

    with open(REPORT, "r", encoding="utf-8") as f:
        rep = json.load(f)

    features = rep["training_summary"]["features"]
    gains = rep["training_summary"].get("feature_importance_gain", {})
    test_days = set(rep["training_summary"]["dias_teste"])

    df = pd.read_csv(DATASET)
    df["dia"] = df["dia"].astype(str)
    df["tipo"] = df["tipo"].astype(str).str.upper()
    df["split"] = np.where(df["dia"].isin(test_days), "test", "train")
    df["acerto"] = pd.to_numeric(df["acerto"], errors="coerce").astype("Int64")
    df["lucro_real"] = pd.to_numeric(df["lucro_real"], errors="coerce")

    print("\n===== RESUMO GERAL =====")
    for split_name in ["train", "test"]:
        st = split_stats(df, split_name)
        if st:
            print(
                f"{split_name.upper():5s} | rows={st['rows']:3d} | "
                f"acerto={st['acerto_mean']:.3f} | lucro_medio={st['lucro_medio']:.2f} | "
                f"BUY={100*st['buy_pct']:.1f}% | SELL={100*st['sell_pct']:.1f}% | "
                f"dias={st['dias']}"
            )

    print("\n===== RANKING DE FEATURES (sinal unitário) =====")
    rows = []
    for feat in features:
        train = df[df["split"] == "train"].copy()
        test = df[df["split"] == "test"].copy()

        train_auc = safe_auc(train["acerto"], train[feat])
        test_auc = safe_auc(test["acerto"], test[feat])

        train_eff = auc_eff(train_auc)
        test_eff = auc_eff(test_auc)

        train_hit = pd.to_numeric(train.loc[train["acerto"] == 1, feat], errors="coerce").mean()
        train_miss = pd.to_numeric(train.loc[train["acerto"] == 0, feat], errors="coerce").mean()
        test_hit = pd.to_numeric(test.loc[test["acerto"] == 1, feat], errors="coerce").mean()
        test_miss = pd.to_numeric(test.loc[test["acerto"] == 0, feat], errors="coerce").mean()

        dir_train = "hit>miss" if pd.notna(train_hit) and pd.notna(train_miss) and train_hit > train_miss else "hit<miss"
        dir_test = "hit>miss" if pd.notna(test_hit) and pd.notna(test_miss) and test_hit > test_miss else "hit<miss"

        rows.append({
            "feature": feat,
            "gain_report": float(gains.get(feat, np.nan)),
            "auc_train_raw": train_auc,
            "auc_test_raw": test_auc,
            "auc_train_eff": train_eff,
            "auc_test_eff": test_eff,
            "gap_eff_train_test": abs(train_eff - test_eff) if pd.notna(train_eff) and pd.notna(test_eff) else np.nan,
            "train_hit_mean": train_hit,
            "train_miss_mean": train_miss,
            "test_hit_mean": test_hit,
            "test_miss_mean": test_miss,
            "dir_train": dir_train,
            "dir_test": dir_test,
            "direcao_estavel": int(dir_train == dir_test),
        })

    rank = pd.DataFrame(rows).sort_values(
        ["auc_test_eff", "direcao_estavel", "gap_eff_train_test", "gain_report"],
        ascending=[False, False, True, False]
    )
    rank.to_csv(OUT_RANK, index=False)

    print(rank[[
        "feature", "gain_report", "auc_train_eff", "auc_test_eff",
        "gap_eff_train_test", "dir_train", "dir_test", "direcao_estavel"
    ]].to_string(index=False))

    print("\n===== TESTE POR LADO E REGIME =====")
    test = df[df["split"] == "test"].copy()
    side_regime = (
        test.groupby(["tipo", "regime_mercado"], observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_mean=("acerto", "mean"),
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
        )
        .reset_index()
        .sort_values(["tipo", "regime_mercado"])
    )
    side_regime.to_csv(OUT_SIDE_REGIME, index=False)
    print(side_regime.to_string(index=False))

    print("\n===== TESTE POR DIA E LADO =====")
    by_day = (
        test.groupby(["dia", "tipo"], observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_mean=("acerto", "mean"),
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
        )
        .reset_index()
        .sort_values(["dia", "tipo"])
    )
    by_day.to_csv(OUT_DAY, index=False)
    print(by_day.to_string(index=False))

    print("\n===== PC em BUY (teste) =====")
    buy_test = test[test["tipo"] == "BUY"].copy()
    bucket_buy = bucket_table(buy_test, "pc", n_quant=4)
    bucket_buy.to_csv(OUT_BUCKET_BUY, index=False)
    if bucket_buy.empty:
        print("Sem dados suficientes.")
    else:
        print(bucket_buy.to_string(index=False))

    print("\n===== PV em SELL (teste) =====")
    sell_test = test[test["tipo"] == "SELL"].copy()
    bucket_sell = bucket_table(sell_test, "pv", n_quant=4)
    bucket_sell.to_csv(OUT_BUCKET_SELL, index=False)
    if bucket_sell.empty:
        print("Sem dados suficientes.")
    else:
        print(bucket_sell.to_string(index=False))

    print("\n===== LEITURA RÁPIDA =====")
    print("1) auc_test_eff perto de 0.50 = feature sem sinal real no teste")
    print("2) gap grande entre auc_train_eff e auc_test_eff = instabilidade / sobreajuste")
    print("3) direcao_estavel=0 = feature muda de sentido entre treino e teste")
    print("4) buckets monotônicos em BUY/pc e SELL/pv = pc/pv carregam informação útil")
    print(f"\nArquivos gerados: {OUT_RANK}, {OUT_SIDE_REGIME}, {OUT_DAY}, {OUT_BUCKET_BUY}, {OUT_BUCKET_SELL}")


if __name__ == "__main__":
    main()