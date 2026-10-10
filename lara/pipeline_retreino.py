# -*- coding: utf-8 -*-
"""Pipeline único da Etapa F para seleção de dias + treino auditável.

Versão robusta para operação diária:
- auto-detecta dataset quando o default não existe
- lê overrides do config principal
- escreve o candidato novo sem sobrescrever o baseline legado
- mantém a seleção de dias como auditoria, mas não derruba o treino se a seleção falhar
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SELECT_SCRIPT = ROOT_DIR / "selecionar_dias_treino.py"
TRAIN_SCRIPT = ROOT_DIR / "treinar_modelo.py"
DEFAULT_SELECTION = ROOT_DIR / "dias_treino_selecionados.csv"
DEFAULT_CONFIG = ROOT_DIR / "config_lara.json"


def run_step(cmd: list[str], title: str, *, fatal: bool = True) -> int:
    print(f"\n===== {title} =====")
    print("[CMD]", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=ROOT_DIR)
    if completed.returncode != 0 and fatal:
        raise SystemExit(completed.returncode)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa a Etapa F em padrão auditável.")
    parser.add_argument("--base-dir", default=str(ROOT_DIR / "arquivo_ABC"), help="Pasta dos logs diários.")
    parser.add_argument("--selection-output", default="", help="CSV com dias aprovados para treino.")
    parser.add_argument("--dataset", default="", help="Dataset consolidado do treino.")
    parser.add_argument("--model-pkl", default="", help="Saída do modelo PKL.")
    parser.add_argument("--model-json", default="", help="Saída do booster JSON.")
    parser.add_argument("--report-json", default="", help="Saída do relatório JSON.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config principal do Lara.")
    parser.add_argument("--n-test-days", type=int, default=3, help="Dias de holdout no treino.")
    parser.add_argument("--calibracao", default="nenhuma", help="Calibração declarativa a registrar no meta.json.")
    return parser


def _load_cfg(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception as exc:
        print(f"[WARN] Falha ao ler config {path}: {exc}")
        return {}


def _resolve(root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def choose_dataset(root: Path, explicit: str | None, cfg: dict) -> Path:
    candidates: list[Path] = []
    for raw in [
        explicit,
        cfg.get("etapa_f_dataset"),
        "modelos/modelo_lara4/dataset_lara4_jan2026_shadow_pv058.csv",
        "modelos/modelo_lara3/dataset_lara3_treino.csv",
        "modelos/modelo_lara3/dataset_lara3_treino_LEGADO_PRE2026.csv",
    ]:
        p = _resolve(root, raw)
        if p is not None and p not in candidates:
            candidates.append(p)
    for p in candidates:
        if p.exists():
            print(f"[INFO] Dataset selecionado para a Etapa F: {p}")
            return p
    raise FileNotFoundError("Nenhum dataset compatível encontrado para a Etapa F.")


def derive_outputs(root: Path, dataset_path: Path, explicit: dict[str, str | None], cfg: dict) -> tuple[Path, Path, Path, Path]:
    if dataset_path.name.startswith("dataset_lara4_") or "modelo_lara4" in str(dataset_path):
        base_dir = (root / "modelos" / "modelo_lara4").resolve()
        model_pkl = base_dir / "modelo_lara4_xgb_jan2026_shadow_pv058_CANDIDATO.pkl"
        model_json = base_dir / "modelo_lara4_xgb_jan2026_shadow_pv058_CANDIDATO.json"
        report_json = base_dir / "modelo_lara4_xgb_jan2026_shadow_pv058_CANDIDATO.report.json"
    else:
        base_dir = (root / "modelos" / "modelo_lara3").resolve()
        model_pkl = base_dir / "modelo_lara3_xgb_v2.pkl"
        model_json = base_dir / "modelo_lara3_xgb_v2.json"
        report_json = base_dir / "modelo_lara3_xgb_v2.report.json"

    model_pkl = _resolve(root, explicit.get("model_pkl") or cfg.get("etapa_f_model_pkl")) or model_pkl
    model_json = _resolve(root, explicit.get("model_json") or cfg.get("etapa_f_model_json")) or model_json
    report_json = _resolve(root, explicit.get("report_json") or cfg.get("etapa_f_report_json")) or report_json
    selection_output = _resolve(root, explicit.get("selection_output") or cfg.get("etapa_f_selection_output")) or DEFAULT_SELECTION
    return selection_output, model_pkl, model_json, report_json


def main() -> None:
    args = build_parser().parse_args()
    py = sys.executable
    cfg = _load_cfg(Path(args.config))

    dataset_path = choose_dataset(ROOT_DIR, args.dataset, cfg)
    selection_output, model_pkl, model_json, report_json = derive_outputs(
        ROOT_DIR,
        dataset_path,
        {
            "selection_output": args.selection_output,
            "model_pkl": args.model_pkl,
            "model_json": args.model_json,
            "report_json": args.report_json,
        },
        cfg,
    )

    selection_rc = run_step(
        [py, str(SELECT_SCRIPT), "--base-dir", args.base_dir, "--output", str(selection_output)],
        "SELEÇÃO DE DIAS",
        fatal=False,
    )
    if selection_rc != 0:
        print("[WARN] Seleção de dias falhou; seguindo com o dataset consolidado já validado.")

    run_step(
        [
            py,
            str(TRAIN_SCRIPT),
            "--dataset",
            str(dataset_path),
            "--n-test-days",
            str(args.n_test_days),
            "--calibracao",
            args.calibracao,
            "--model-pkl",
            str(model_pkl),
            "--model-json",
            str(model_json),
            "--report-json",
            str(report_json),
        ],
        "TREINO AUDITÁVEL",
    )
    print("\n[OK] Etapa F concluída com pipeline auditável.")
    print(f"[OK] Selection CSV: {selection_output}")
    print(f"[OK] Modelo PKL   : {model_pkl}")
    print(f"[OK] Modelo JSON  : {model_json}")
    print(f"[OK] Report JSON  : {report_json}")


if __name__ == "__main__":
    main()

