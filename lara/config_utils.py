# -*- coding: utf-8 -*-
"""Utilitários centrais para elevar a Etapa F a um padrão mais auditável.

Este módulo concentra:
- validação forte do config
- resolução de caminhos relativos ao projeto
- escrita de metadados do modelo
- helpers de serialização JSON
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "config_lara.json"

REQUIRED_TOP_LEVEL_KEYS: Tuple[str, ...] = (
    "symbol",
    "simulacao",
    "limiar_compra_base",
    "limiar_venda_base",
    "sl_points",
    "tp_points",
    "paths",
    "version_tag",
)

REQUIRED_PATH_KEYS: Tuple[str, ...] = ("pasta_logs",)

DEFAULT_FEATURES: List[str] = [
    "preco",
    "sl",
    "tp",
    "tendencia_pct",
    "atr",
    "atr_media50",
    "vol_norm",
    "pc",
    "pv",
    "limiar_compra_ajustado",
    "limiar_venda_ajustado",
]


@dataclass
class ConfigValidationResult:
    normalized: Dict[str, Any]
    warnings: List[str]


def project_path(*parts: str | Path) -> Path:
    path = PROJECT_ROOT
    for part in parts:
        path = path / Path(part)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON deve conter um objeto no topo: {path}")
    return data


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _ensure_bool(cfg: Dict[str, Any], key: str) -> None:
    if not isinstance(cfg.get(key), bool):
        raise TypeError(f"Config '{key}' deve ser booleano.")


def _ensure_number(cfg: Dict[str, Any], key: str, *, min_value: float | None = None) -> None:
    value = cfg.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"Config '{key}' deve ser numérico.")
    if min_value is not None and float(value) < min_value:
        raise ValueError(f"Config '{key}' deve ser >= {min_value}.")


def resolve_config_path(path: str | Path | None = None) -> Path:
    if path is None:
        path = DEFAULT_CONFIG_NAME
    p = Path(path)
    if not p.is_absolute():
        p = project_path(p)
    return p


def validate_and_normalize_config(cfg: Dict[str, Any]) -> ConfigValidationResult:
    warnings: List[str] = []

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Config sem chaves obrigatórias: {missing}")

    if not isinstance(cfg.get("paths"), dict):
        raise TypeError("Config 'paths' deve ser objeto/dict.")

    missing_paths = [k for k in REQUIRED_PATH_KEYS if k not in cfg["paths"]]
    if missing_paths:
        raise KeyError(f"Config paths sem chaves obrigatórias: {missing_paths}")

    _ensure_bool(cfg, "simulacao")
    if "permitir_execucao_real" in cfg:
        _ensure_bool(cfg, "permitir_execucao_real")
    if "usar_modelo_xgb" in cfg:
        _ensure_bool(cfg, "usar_modelo_xgb")

    _ensure_number(cfg, "limiar_compra_base", min_value=0.0)
    _ensure_number(cfg, "limiar_venda_base", min_value=0.0)
    _ensure_number(cfg, "sl_points", min_value=0.0)
    _ensure_number(cfg, "tp_points", min_value=0.0)

    if float(cfg["limiar_compra_base"]) > 1.0:
        warnings.append("limiar_compra_base > 1.0; confirme se a escala está correta.")
    if float(cfg["limiar_venda_base"]) > 1.0:
        warnings.append("limiar_venda_base > 1.0; confirme se a escala está correta.")

    symbol = str(cfg.get("symbol", "")).strip().upper()
    if not symbol:
        raise ValueError("Config 'symbol' vazio.")
    cfg["symbol"] = symbol

    version_tag = str(cfg.get("version_tag", "")).strip()
    if not version_tag:
        raise ValueError("Config 'version_tag' vazio.")

    if cfg.get("usar_modelo_xgb", False):
        model_path = str(cfg.get("caminho_modelo", "")).strip()
        if not model_path:
            raise ValueError("usar_modelo_xgb=true exige 'caminho_modelo'.")
        if "features_esperadas" in cfg:
            feats = cfg["features_esperadas"]
            if not isinstance(feats, list) or not feats:
                raise TypeError("Config 'features_esperadas' deve ser lista não vazia.")
        else:
            cfg["features_esperadas"] = list(DEFAULT_FEATURES)
            warnings.append("features_esperadas ausente; usando DEFAULT_FEATURES.")

    log_dir = Path(str(cfg["paths"]["pasta_logs"]))
    if not log_dir.is_absolute():
        log_dir = project_path(log_dir)
    cfg["paths"]["pasta_logs"] = str(log_dir)

    if cfg.get("usar_modelo_xgb", False):
        model_path = Path(str(cfg["caminho_modelo"]))
        if not model_path.is_absolute():
            model_path = project_path(model_path)
        cfg["caminho_modelo"] = str(model_path)

    return ConfigValidationResult(normalized=cfg, warnings=warnings)


def write_model_meta(
    *,
    model_json_path: Path,
    features: Iterable[str],
    calibracao: str,
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    training_summary: Dict[str, Any],
) -> Path:
    meta_path = model_json_path.with_suffix("").with_suffix(".meta.json")
    payload = {
        "features": list(features),
        "calibracao": calibracao,
        "auc_kfold_mean": test_metrics.get("roc_auc"),
        "auc": test_metrics.get("roc_auc"),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "training_summary": training_summary,
    }
    dump_json(meta_path, payload)
    return meta_path

