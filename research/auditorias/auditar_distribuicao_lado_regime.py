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

OUT_SIDE = OUT_DIR_AUD / "auditoria_distribuicao_side.csv"
OUT_REGIME = OUT_DIR_AUD / "auditoria_distribuicao_regime.csv"
OUT_SIDE_REGIME = OUT_DIR_AUD / "auditoria_distribuicao_side_regime.csv"
OUT_COMPARE = OUT_DIR_AUD / "auditoria_distribuicao_compare_train_test.csv"
OUT_BY_DAY = OUT_DIR_AUD / "auditoria_distribuicao_por_dia.csv"


def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if len(s.dropna()) else np.nan


def build_share_table(df, group_cols):
    base = (
        df.groupby(["split"] + group_cols, observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_mean=("acerto", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            lucro_total=("lucro_real", lambda s: pd.to_numeric(s, errors="coerce").sum()),
            lucro_medio=("lucro_real", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
    )

    totals = (
        df.groupby("split", observed=False)
        .agg(rows_split=("acerto", "size"))
        .reset_index()
    )

    out = base.merge(totals, on="split", how="left")
    out["share_split"] = out["rows"] / out["rows_split"]
    return out


def compare_train_test_shares(df, key_cols):
    agg = (
        df.groupby(["split"] + key_cols, observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_mean=("acerto", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            lucro_total=("lucro_real", lambda s: pd.to_numeric(s, errors="coerce").sum()),
            lucro_medio=("lucro_real", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
    )

    totals = (
        df.groupby("split", observed=False)
        .agg(rows_split=("acerto", "size"))
        .reset_index()
    )
    agg = agg.merge(totals, on="split", how="left")
    agg["share_split"] = agg["rows"] / agg["rows_split"]

    idx_cols = key_cols
    train = agg[agg["split"] == "train"].copy().drop(columns=["split"])
    test = agg[agg["split"] == "test"].copy().drop(columns=["split"])

    train = train.rename(columns={
        "rows": "rows_train",
        "acerto_mean": "acerto_train",
        "lucro_total": "lucro_total_train",
        "lucro_medio": "lucro_medio_train",
        "rows_split": "rows_split_train",
        "share_split": "share_train",
    })
    test = test.rename(columns={
        "rows": "rows_test",
        "acerto_mean": "acerto_test",
        "lucro_total": "lucro_total_test",
        "lucro_medio": "lucro_medio_test",
        "rows_split": "rows_split_test",
        "share_split": "share_test",
    })

    cmp = train.merge(test, on=idx_cols, how="outer")

    for c in ["rows_train", "rows_test", "share_train", "share_test",
              "acerto_train", "acerto_test", "lucro_total_train",
              "lucro_total_test", "lucro_medio_train", "lucro_medio_test"]:
        if c in cmp.columns:
            cmp[c] = pd.to_numeric(cmp[c], errors="coerce")

    cmp["share_delta_abs"] = (cmp["share_test"] - cmp["share_train"]).abs()
    cmp["acerto_delta_abs"] = (cmp["acerto_test"] - cmp["acerto_train"]).abs()
    cmp["lucro_medio_delta_abs"] = (cmp["lucro_medio_test"] - cmp["lucro_medio_train"]).abs()

    def flag_share(x):
        if pd.isna(x):
            return "sem_dado"
        if x >= 0.25:
            return "grave"
        if x >= 0.15:
            return "moderado"
        return "baixo"

    cmp["flag_shift_share"] = cmp["share_delta_abs"].apply(flag_share)
    return cmp.sort_values(["flag_shift_share", "share_delta_abs"], ascending=[True, False])


OUT_DIR_AUD.mkdir(parents=True, exist_ok=True)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET}")
    if not REPORT.exists():
        raise FileNotFoundError(f"Report não encontrado: {REPORT}")

    with open(REPORT, "r", encoding="utf-8") as f:
        rep = json.load(f)

    test_days = set(rep["training_summary"]["dias_teste"])

    df = pd.read_csv(DATASET)
    df["dia"] = df["dia"].astype(str)
    df["tipo"] = df["tipo"].astype(str).str.upper()
    df["regime_mercado"] = df["regime_mercado"].astype(str)
    df["split"] = np.where(df["dia"].isin(test_days), "test", "train")
    df["acerto"] = pd.to_numeric(df["acerto"], errors="coerce")
    df["lucro_real"] = pd.to_numeric(df["lucro_real"], errors="coerce")

    print("\n===== RESUMO GERAL =====")
    resumo = (
        df.groupby("split", observed=False)
        .agg(
            rows=("acerto", "size"),
            dias=("dia", "nunique"),
            acerto_mean=("acerto", "mean"),
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
            buy_pct=("tipo", lambda s: (s == "BUY").mean()),
            sell_pct=("tipo", lambda s: (s == "SELL").mean()),
        )
        .reset_index()
    )
    print(resumo.to_string(index=False))

    print("\n===== DISTRIBUIÇÃO POR LADO =====")
    by_side = build_share_table(df, ["tipo"]).sort_values(["split", "tipo"])
    by_side.to_csv(OUT_SIDE, index=False)
    print(by_side.to_string(index=False))

    print("\n===== DISTRIBUIÇÃO POR REGIME =====")
    by_regime = build_share_table(df, ["regime_mercado"]).sort_values(["split", "regime_mercado"])
    by_regime.to_csv(OUT_REGIME, index=False)
    print(by_regime.to_string(index=False))

    print("\n===== DISTRIBUIÇÃO POR LADO × REGIME =====")
    by_side_regime = build_share_table(df, ["tipo", "regime_mercado"]).sort_values(["split", "tipo", "regime_mercado"])
    by_side_regime.to_csv(OUT_SIDE_REGIME, index=False)
    print(by_side_regime.to_string(index=False))

    print("\n===== COMPARAÇÃO TRAIN vs TEST =====")
    cmp_side = compare_train_test_shares(df, ["tipo"])
    cmp_regime = compare_train_test_shares(df, ["regime_mercado"])
    cmp_side_regime = compare_train_test_shares(df, ["tipo", "regime_mercado"])

    print("\n--- LADO ---")
    print(cmp_side.to_string(index=False))

    print("\n--- REGIME ---")
    print(cmp_regime.to_string(index=False))

    print("\n--- LADO × REGIME ---")
    print(cmp_side_regime.to_string(index=False))

    compare_all = {
        "side": cmp_side,
        "regime": cmp_regime,
        "side_regime": cmp_side_regime,
    }
    with pd.ExcelWriter(OUT_COMPARE.with_suffix(".xlsx")) as writer:
        for sheet, dfx in compare_all.items():
            dfx.to_excel(writer, sheet_name=sheet[:31], index=False)

    print("\n===== TESTE POR DIA =====")
    by_day = (
        df[df["split"] == "test"]
        .groupby(["dia", "tipo", "regime_mercado"], observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_mean=("acerto", "mean"),
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
        )
        .reset_index()
        .sort_values(["dia", "tipo", "regime_mercado"])
    )
    by_day.to_csv(OUT_BY_DAY, index=False)
    print(by_day.to_string(index=False))

    print("\n===== REGRAS DE LEITURA =====")
    print("1) share_train vs share_test muito diferentes = desbalanceamento estrutural")
    print("2) flag_shift_share=grave sugere mudança forte de composição do holdout")
    print("3) lado × regime com poucas linhas no treino = baixa robustez")
    print("4) acerto/lucro ruins concentrados em um lado ou regime = candidato quebrando em bloco específico")
    print("5) se o teste tiver combinações ausentes ou quase ausentes no treino, o modelo tende a generalizar mal")

    print(f"\nArquivos gerados: {OUT_SIDE}, {OUT_REGIME}, {OUT_SIDE_REGIME}, {OUT_COMPARE.with_suffix('.xlsx')}, {OUT_BY_DAY}")


if __name__ == "__main__":
    main()