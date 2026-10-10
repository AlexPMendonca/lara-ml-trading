# -*- coding: utf-8 -*-
"""Treino auditável do modelo Lara 4 (XGBoost) com holdout temporal por dia.

Melhorias desta versão:
- argparse para dataset/output/configuração
- métricas mais ricas (ROC AUC, PR AUC, Brier)
- relatório JSON do treino
- geração automática de modelo.meta.json
- feature importance serializada
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config_utils import DEFAULT_FEATURES, dump_json, write_model_meta

try:
    from joblib import dump as joblib_dump
except ImportError:
    joblib_dump = None

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "modelos" / "modelo_lara4"
DATASET_PATH = MODEL_DIR / "dataset_lara4_jan2026_shadow_pv058.csv"
# Saídas padrão: um CANDIDATO novo, sem sobrescrever o modelo em produção (SIMPLIFICADO_V1)
MODEL_PKL_PATH = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_CANDIDATO.pkl"
MODEL_JSON_PATH = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_CANDIDATO.json"
REPORT_JSON_PATH = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_CANDIDATO.report.json"
FEATURES = ["tendencia_pct", "atr", "vol_norm", "pc", "pv"]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    return value


def carregar_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {dataset_path}")

    print(f"[INFO] Lendo dataset consolidado: {dataset_path}")
    df = pd.read_csv(dataset_path)

    required_columns = {"alvo_acerto", "dia", *FEATURES}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Dataset sem colunas obrigatórias: {missing}")

    df = df.copy()
    df["alvo_acerto"] = pd.to_numeric(df["alvo_acerto"], errors="raise").astype(int)
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    null_mask = df[FEATURES].isna().any(axis=1)
    null_rows = int(null_mask.sum())
    if null_rows:
        print(f"[WARN] Removendo {null_rows} linhas com NaN nas features.")
        df = df.loc[~null_mask].copy()

    if df.empty:
        raise ValueError("Dataset vazio após limpeza.")

    return df


def separar_treino_teste_por_dia(df: pd.DataFrame, n_test_days: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    dias = sorted(df["dia"].astype(str).unique())
    if len(dias) <= n_test_days:
        raise ValueError(f"Poucos dias no dataset ({len(dias)}). Reduza n_test_days ou colete mais dias.")

    dias_treino = dias[:-n_test_days]
    dias_teste = dias[-n_test_days:]

    df_train = df[df["dia"].astype(str).isin(dias_treino)].copy()
    df_test = df[df["dia"].astype(str).isin(dias_teste)].copy()

    print("\n===== DIAS USADOS =====")
    print(f"Dias treino ({len(dias_treino)}): {dias_treino}")
    print(f"Dias teste  ({len(dias_teste)}): {dias_teste}")
    print(f"\n[INFO] Exemplos treino: {len(df_train)}")
    print(f"[INFO] Exemplos teste : {len(df_test)}")

    return df_train, df_test, dias_treino, dias_teste


def montar_X_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURES].astype(float).values
    y = df["alvo_acerto"].astype(int).values
    return X, y


def _compute_sample_weight(df_train: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
    n_total = len(y_train)
    n_pos = int(y_train.sum())
    n_neg = n_total - n_pos

    print("\n===== BALANCEAMENTO (TREINO) =====")
    print(f"Total treino          : {n_total}")
    print(f"Positivos (1/acerto)  : {n_pos} ({(100.0 * n_pos / n_total):.1f}%)")
    print(f"Negativos (0/erro)    : {n_neg} ({(100.0 * n_neg / n_total):.1f}%)")

    if n_pos == 0 or n_neg == 0:
        w_pos = 1.0
        w_neg = 1.0
        print("[AVISO] Apenas uma classe presente no treino. Pesos padrão = 1.0.")
    else:
        w_pos = n_total / (2.0 * n_pos)
        w_neg = n_total / (2.0 * n_neg)

    lados = df_train["tipo"].astype(str).fillna("UNKNOWN") if "tipo" in df_train.columns else pd.Series(["BUY"] * n_total)
    n_buy = int((lados == "BUY").sum())
    n_sell = int((lados == "SELL").sum())

    print("\n===== DISTRIBUIÇÃO LADO (TREINO) =====")
    print(f"BUY   : {n_buy} ({(100.0 * n_buy / n_total):.1f}%)")
    print(f"SELL  : {n_sell} ({(100.0 * n_sell / n_total):.1f}%)")

    if n_buy > 0 and n_sell > 0:
        w_buy_side = n_total / (2.0 * n_buy)
        w_sell_side = n_total / (2.0 * n_sell)
    else:
        w_buy_side = 1.0
        w_sell_side = 1.0

    print("\n===== PESOS USADOS =====")
    print(f"w_pos (classe 1/acerto) : {w_pos:.3f}")
    print(f"w_neg (classe 0/erro)   : {w_neg:.3f}")
    print(f"w_buy_side              : {w_buy_side:.3f}")
    print(f"w_sell_side             : {w_sell_side:.3f}")

    lados_arr = lados.values
    class_factor = np.where(y_train == 1, w_pos, w_neg)
    side_factor = np.where(lados_arr == "BUY", w_buy_side, np.where(lados_arr == "SELL", w_sell_side, 1.0))
    sample_weight = class_factor * side_factor
    mean_w = float(sample_weight.mean()) if n_total > 0 else 1.0
    if mean_w > 0:
        sample_weight = sample_weight / mean_w
    return sample_weight


def treinar_modelo(df_train: pd.DataFrame) -> XGBClassifier:
    X_train, y_train = montar_X_y(df_train)
    sample_weight = _compute_sample_weight(df_train, y_train)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\n[INFO] Treinando modelo XGBoost com sample_weight (acerto + lado)...")
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _safe_metric(metric_func, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        value = metric_func(y_true, y_score)
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return None
        return float(value)
    except Exception:
        return None


def _classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    return _to_builtin(report)


def compute_metrics(model: XGBClassifier, df_part: pd.DataFrame, *, label: str) -> Dict[str, Any]:
    X, y = montar_X_y(df_part)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "label": label,
        "rows": int(len(df_part)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": _safe_metric(roc_auc_score, y, proba),
        "pr_auc": _safe_metric(average_precision_score, y, proba),
        "brier": _safe_metric(brier_score_loss, y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": _classification_report_dict(y, pred),
        "prob_min": float(np.min(proba)),
        "prob_max": float(np.max(proba)),
        "prob_mean": float(np.mean(proba)),
        "prob_std": float(np.std(proba)),
        "prob_quantiles": {
            "q00": float(np.quantile(proba, 0.00)),
            "q10": float(np.quantile(proba, 0.10)),
            "q25": float(np.quantile(proba, 0.25)),
            "q50": float(np.quantile(proba, 0.50)),
            "q75": float(np.quantile(proba, 0.75)),
            "q90": float(np.quantile(proba, 0.90)),
            "q100": float(np.quantile(proba, 1.00)),
        },
    }
    return metrics


def salvar_modelo(model: XGBClassifier, model_pkl_path: Path, model_json_path: Path) -> None:
    model_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    if joblib_dump is not None:
        print(f"\n[INFO] Salvando modelo em PKL: {model_pkl_path}")
        joblib_dump(model, model_pkl_path)
    else:
        print(f"\n[INFO] joblib não disponível. Salvando com pickle: {model_pkl_path}")
        with model_pkl_path.open("wb") as f:
            pickle.dump(model, f)

    booster = model.get_booster()
    print(f"[INFO] Salvando booster em JSON: {model_json_path}")
    booster.save_model(model_json_path)


def _feature_importance(model: XGBClassifier) -> Dict[str, float]:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    out: Dict[str, float] = {}
    for idx, feature in enumerate(FEATURES):
        out[feature] = float(score.get(f"f{idx}", 0.0))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treino auditável do modelo Lara 3 / Etapa F")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Caminho do dataset consolidado.")
    parser.add_argument("--n-test-days", type=int, default=3, help="Quantidade de dias no holdout final.")
    parser.add_argument("--model-pkl", default=str(MODEL_PKL_PATH), help="Saída do modelo PKL.")
    parser.add_argument("--model-json", default=str(MODEL_JSON_PATH), help="Saída do booster JSON.")
    parser.add_argument("--report-json", default=str(REPORT_JSON_PATH), help="Saída do relatório JSON.")
    parser.add_argument("--calibracao", default="nenhuma", help="Calibração declarativa do modelo (ex.: nenhuma, sigmoid).")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_pkl_path = Path(args.model_pkl)
    model_json_path = Path(args.model_json)
    report_json_path = Path(args.report_json)

    df = carregar_dataset(dataset_path)
    df_train, df_test, dias_treino, dias_teste = separar_treino_teste_por_dia(df, n_test_days=args.n_test_days)
    model = treinar_modelo(df_train)

    train_metrics = compute_metrics(model, df_train, label="train")
    test_metrics = compute_metrics(model, df_test, label="test")

    salvar_modelo(model, model_pkl_path, model_json_path)

    training_summary = {
        "dataset_path": str(dataset_path),
        "rows_total": int(len(df)),
        "rows_train": int(len(df_train)),
        "rows_test": int(len(df_test)),
        "dias_treino": dias_treino,
        "dias_teste": dias_teste,
        "features": list(FEATURES),
        "feature_importance_gain": _feature_importance(model),
        "calibracao": args.calibracao,
    }

    report_payload = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "training_summary": training_summary,
    }
    dump_json(report_json_path, report_payload)
    meta_path = write_model_meta(
        model_json_path=model_json_path,
        features=FEATURES,
        calibracao=args.calibracao,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        training_summary=training_summary,
    )

    print("\n===== AVALIAÇÃO RESUMIDA =====")
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    print("\n[OK] Treino concluído com sucesso.")
    print(f"[OK] Modelo PKL : {model_pkl_path}")
    print(f"[OK] Modelo JSON: {model_json_path}")
    print(f"[OK] Relatório  : {report_json_path}")
    print(f"[OK] Meta JSON  : {meta_path}")


if __name__ == "__main__":
    main()

