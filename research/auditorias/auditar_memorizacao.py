# -*- coding: utf-8 -*-
import json
import pickle
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
MODEL = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_SIMPLIFICADO_V1.pkl"

OUT_PRED = OUT_DIR_AUD / "auditoria_memorizacao_predicoes.csv"
OUT_CONF = OUT_DIR_AUD / "auditoria_memorizacao_confidence.csv"
OUT_CAL = OUT_DIR_AUD / "auditoria_memorizacao_calibracao.csv"
OUT_NN = OUT_DIR_AUD / "auditoria_memorizacao_nn_test.csv"
OUT_DAY = OUT_DIR_AUD / "auditoria_memorizacao_test_por_dia.csv"
OUT_TOP = OUT_DIR_AUD / "auditoria_memorizacao_top_confianca_test.csv"


def safe_auc(y, p):
    y = pd.Series(y).astype(float)
    p = pd.Series(p).astype(float)
    mask = y.notna() & p.notna()
    y = y[mask]
    p = p[mask]
    if len(y) < 10 or y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def brier_score(y, p):
    y = pd.Series(y).astype(float)
    p = pd.Series(p).astype(float)
    mask = y.notna() & p.notna()
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return np.nan
    return float(np.mean((p - y) ** 2))


def summarize_probs(df, split_name):
    d = df[df["split"] == split_name].copy()
    return {
        "split": split_name,
        "rows": int(len(d)),
        "acerto_real": float(d["acerto"].mean()),
        "prob_mean": float(d["prob_acerto"].mean()),
        "prob_std": float(d["prob_acerto"].std(ddof=0)),
        "confidence_mean": float(d["confidence"].mean()),
        "confidence_std": float(d["confidence"].std(ddof=0)),
        "accuracy_05": float((d["pred_label"] == d["acerto"]).mean()),
        "auc": safe_auc(d["acerto"], d["prob_acerto"]),
        "brier": brier_score(d["acerto"], d["prob_acerto"]),
        "overconfidence_gap": float(d["prob_acerto"].mean() - d["acerto"].mean()),
    }


def make_confidence_table(df):
    d = df.copy()
    bins = [-0.001, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
    labels = [
        "0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
        "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"
    ]
    d["conf_bucket"] = pd.cut(d["confidence"], bins=bins, labels=labels, include_lowest=True)
    out = (
        d.groupby(["split", "conf_bucket"], observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_mean=("acerto", "mean"),
            prob_mean=("prob_acerto", "mean"),
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
        )
        .reset_index()
    )
    return out


def make_calibration_table(df):
    d = df.copy()
    bins = [-0.001, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
    labels = [
        "0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
        "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"
    ]
    d["prob_bucket"] = pd.cut(d["prob_acerto"], bins=bins, labels=labels, include_lowest=True)
    out = (
        d.groupby(["split", "prob_bucket"], observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_real=("acerto", "mean"),
            prob_media=("prob_acerto", "mean"),
            calibration_gap=("prob_acerto", lambda s: np.nan),  # placeholder
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
        )
        .reset_index()
    )
    # preencher gap = prob_media - acerto_real
    out["calibration_gap"] = out["prob_media"] - out["acerto_real"]
    return out


def standardized_nn_distance(train_df, test_df, features):
    tr = train_df[features].apply(pd.to_numeric, errors="coerce").copy()
    te = test_df[features].apply(pd.to_numeric, errors="coerce").copy()

    valid_tr = tr.notna().all(axis=1)
    valid_te = te.notna().all(axis=1)

    tr = tr.loc[valid_tr].reset_index(drop=True)
    te = te.loc[valid_te].reset_index(drop=True)

    if tr.empty or te.empty:
        return pd.DataFrame()

    mu = tr.mean()
    sd = tr.std(ddof=0).replace(0, 1.0)

    tr_arr = ((tr - mu) / sd).to_numpy(dtype=float)
    te_arr = ((te - mu) / sd).to_numpy(dtype=float)

    rows = []
    for i in range(te_arr.shape[0]):
        d = np.sqrt(((tr_arr - te_arr[i]) ** 2).sum(axis=1))
        rows.append({
            "te_valid_idx": int(i),
            "nn_dist_min": float(d.min()),
            "dist_q10": float(np.quantile(d, 0.10)),
            "dist_q50": float(np.quantile(d, 0.50)),
        })
    return pd.DataFrame(rows)


OUT_DIR_AUD.mkdir(parents=True, exist_ok=True)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET}")
    if not REPORT.exists():
        raise FileNotFoundError(f"Report não encontrado: {REPORT}")
    if not MODEL.exists():
        raise FileNotFoundError(f"Modelo PKL não encontrado: {MODEL}")

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
        raise ValueError("Há NaNs nas features. Revise o dataset antes de seguir.")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        pred_raw = model.predict(X)
        prob = pd.to_numeric(pd.Series(pred_raw), errors="coerce").fillna(0.5).clip(0, 1).to_numpy()

    df["prob_acerto"] = prob
    df["pred_label"] = (df["prob_acerto"] >= 0.5).astype(int)
    df["confidence"] = (df["prob_acerto"] - 0.5).abs() * 2.0

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("\n===== RESUMO DO MODELO =====")
    resumo = pd.DataFrame([
        summarize_probs(df, "train"),
        summarize_probs(df, "test"),
    ])
    print(resumo.to_string(index=False))

    print("\n===== CONFIANÇA vs RESULTADO =====")
    conf_tbl = make_confidence_table(df)
    conf_tbl.to_csv(OUT_CONF, index=False)
    print(conf_tbl.to_string(index=False))

    print("\n===== CALIBRAÇÃO POR FAIXA DE PROBABILIDADE =====")
    cal_tbl = make_calibration_table(df)
    cal_tbl.to_csv(OUT_CAL, index=False)
    print(cal_tbl.to_string(index=False))

    # Distância do teste ao treino
    nn = standardized_nn_distance(train_df, test_df, features)
    if not nn.empty:
        test_valid = test_df.reset_index(drop=True).copy()
        test_valid = test_valid.iloc[:len(nn)].copy()
        test_valid = pd.concat([test_valid.reset_index(drop=True), nn.reset_index(drop=True)], axis=1)

        test_valid["nn_bucket"] = pd.qcut(test_valid["nn_dist_min"], q=4, duplicates="drop")
        nn_tbl = (
            test_valid.groupby("nn_bucket", observed=False)
            .agg(
                rows=("acerto", "size"),
                acerto_mean=("acerto", "mean"),
                prob_mean=("prob_acerto", "mean"),
                confidence_mean=("confidence", "mean"),
                lucro_total=("lucro_real", "sum"),
                lucro_medio=("lucro_real", "mean"),
                nn_min=("nn_dist_min", "min"),
                nn_max=("nn_dist_min", "max"),
            )
            .reset_index()
        )
    else:
        test_valid = test_df.copy()
        test_valid["nn_dist_min"] = np.nan
        nn_tbl = pd.DataFrame()

    OUT_PRED.write_text("", encoding="utf-8")
    df.to_csv(OUT_PRED, index=False)

    print("\n===== TESTE POR DISTÂNCIA AO TREINO =====")
    nn_tbl.to_csv(OUT_NN, index=False)
    if nn_tbl.empty:
        print("Sem dados suficientes.")
    else:
        print(nn_tbl.to_string(index=False))

    print("\n===== TESTE POR DIA =====")
    by_day = (
        test_valid.groupby(["dia"], observed=False)
        .agg(
            rows=("acerto", "size"),
            acerto_real=("acerto", "mean"),
            prob_media=("prob_acerto", "mean"),
            confidence_media=("confidence", "mean"),
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
            nn_dist_media=("nn_dist_min", "mean"),
        )
        .reset_index()
        .sort_values("dia")
    )
    by_day.to_csv(OUT_DAY, index=False)
    print(by_day.to_string(index=False))

    print("\n===== TOP CONFIANÇA NO TESTE =====")
    top_test = test_valid.sort_values(["confidence", "prob_acerto"], ascending=[False, False]).head(15).copy()
    cols_top = ["dia", "tipo", "acerto", "lucro_real", "prob_acerto", "confidence", "pc", "pv", "nn_dist_min"]
    cols_top = [c for c in cols_top if c in top_test.columns]
    top_test = top_test[cols_top]
    top_test.to_csv(OUT_TOP, index=False)
    print(top_test.to_string(index=False))

    print("\n===== FLAGS DE LEITURA =====")
    train_acc = float(resumo.loc[resumo["split"] == "train", "accuracy_05"].iloc[0])
    test_acc = float(resumo.loc[resumo["split"] == "test", "accuracy_05"].iloc[0])
    train_gap = float(resumo.loc[resumo["split"] == "train", "overconfidence_gap"].iloc[0])
    test_gap = float(resumo.loc[resumo["split"] == "test", "overconfidence_gap"].iloc[0])

    print(f"gap_accuracy_train_test = {train_acc - test_acc:.4f}")
    print(f"overconfidence_gap_train = {train_gap:.4f}")
    print(f"overconfidence_gap_test  = {test_gap:.4f}")

    # leitura rápida da confiança no teste
    test_conf = conf_tbl[conf_tbl["split"] == "test"].copy()
    test_conf = test_conf[test_conf["rows"] > 0].copy()
    if not test_conf.empty:
        test_conf_sorted = test_conf.sort_values("conf_bucket")
        print("\nResumo confiança-teste:")
        print(test_conf_sorted[["conf_bucket", "rows", "acerto_mean", "prob_mean", "lucro_medio"]].to_string(index=False))

    print("\nRegras:")
    print("1) gap_accuracy_train_test muito alto = forte suspeita de memorização")
    print("2) confidence alta no teste sem acerto melhor = confiança enganosa")
    print("3) calibration_gap positivo alto = modelo superconfiante")
    print("4) exemplos mais próximos do treino deveriam performar melhor se houver padrão transferível")
    print("5) top confiança do teste com muitos erros = modelo memoriza treino e exagera convicção no holdout")

    print(f"\nArquivos gerados: {OUT_PRED}, {OUT_CONF}, {OUT_CAL}, {OUT_NN}, {OUT_DAY}, {OUT_TOP}")


if __name__ == "__main__":
    main()