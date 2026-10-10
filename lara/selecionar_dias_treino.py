# -*- coding: utf-8 -*-
"""Seleciona dias válidos para treino do modelo Lara com configuração parametrizada.

Versão robusta:
- aceita labels reais (log_operacoes_v3)
- aceita labels shadow recentes (CALIB_PC027_PV058 ou STRICT) quando não há real
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = ROOT_DIR / "arquivo_ABC"
DEFAULT_OUTPUT = ROOT_DIR / "dias_treino_selecionados.csv"

BASE_REQUIRED = {
    "log_ia_exec": "log_ia_exec_{date}.csv",
    "log_120s": "log_120s_{date}.csv",
}
OPTIONAL_PATTERNS = {
    "validacao_incremental": "validacao_incremental_{date}.csv",
    "reconciliacao": "reconciliacao_v5_2_custom_{date}.csv",
}
MIN_ROWS = {
    "log_ia_exec": 200,
    "log_120s": 20,
    "log_operacoes_v3": 5,
    "log_operacoes_shadow": 5,
}
EXPECTED_COLS = {
    "log_ia_exec": [],
    "log_120s": ["positions"],
    "log_operacoes_v3": ["tipo", "lucro_real"],
    "log_operacoes_shadow": ["direcao", "resultado_pontos", "motivo_saida"],
}


def parse_iso_date(text: str | None) -> Optional[date]:
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def parse_date_from_filename(stem: str, prefix: str) -> Optional[date]:
    try:
        raw = stem.split(prefix, 1)[1]
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except Exception:
        return None


def find_all_dates(base_dir: Path) -> List[date]:
    dates = set()
    for f in base_dir.glob("log_ia_exec_*.csv"):
        day = parse_date_from_filename(f.stem, "log_ia_exec_")
        if day is not None:
            dates.add(day)
    return sorted(dates)


def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[ERRO] Falha ao ler {path.name}: {exc}")
        return None


def choose_label_file(base_dir: Path, day_str: str) -> Tuple[str, Path] | None:
    candidates = [
        ("log_operacoes_v3", base_dir / f"log_operacoes_v3_{day_str}.csv"),
        ("log_operacoes_shadow", base_dir / f"log_operacoes_shadow_v3_{day_str}_STRICT.csv"),
        ("log_operacoes_shadow", base_dir / f"log_operacoes_shadow_v3_{day_str}_CALIB_PC065_PV058.csv"),
        ("log_operacoes_shadow", base_dir / f"log_operacoes_shadow_v3_{day_str}_CALIB_PC027_PV058.csv"),
        ("log_operacoes_shadow", base_dir / f"log_operacoes_shadow_v3_{day_str}.csv"),
    ]
    for kind, path in candidates:
        if path.exists():
            return kind, path
    return None


def check_day(*, base_dir: Path, day: date, start_date: Optional[date], end_date: Optional[date], exclude_dates: set[str]) -> dict:
    day_str = day.isoformat()

    if start_date and day < start_date:
        return {"data": day_str, "usar_no_treino": 0, "status": "DESCARTAR", "motivo": "Antes de START_DATE"}
    if end_date and day > end_date:
        return {"data": day_str, "usar_no_treino": 0, "status": "DESCARTAR", "motivo": "Depois de END_DATE"}
    if day_str in exclude_dates:
        return {"data": day_str, "usar_no_treino": 0, "status": "DESCARTAR", "motivo": "Marcado manualmente"}

    paths_required: dict[str, Path] = {}
    for key, pattern in BASE_REQUIRED.items():
        p = base_dir / pattern.format(date=day_str)
        if not p.exists():
            return {"data": day_str, "usar_no_treino": 0, "status": "DESCARTAR", "motivo": f"Arquivo obrigatório ausente: {p.name}"}
        paths_required[key] = p

    label_choice = choose_label_file(base_dir, day_str)
    if label_choice is None:
        return {"data": day_str, "usar_no_treino": 0, "status": "DESCARTAR", "motivo": "Sem arquivo de labels (real/shadow)"}

    label_kind, label_path = label_choice
    paths_required[label_kind] = label_path

    dataframes: dict[str, pd.DataFrame] = {}
    for key, path in paths_required.items():
        df = load_csv_safe(path)
        if df is None:
            return {"data": day_str, "usar_no_treino": 0, "status": "DESCARTAR", "motivo": f"Falha leitura: {path.name}"}
        dataframes[key] = df

    for key, df in dataframes.items():
        min_rows = MIN_ROWS.get(key, 1)
        if len(df) < min_rows:
            return {"data": day_str, "usar_no_treino": 0, "status": "DESCARTAR", "motivo": f"Poucas linhas em {key}: {len(df)} < {min_rows}"}
        expected = EXPECTED_COLS.get(key, [])
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            return {"data": day_str, "usar_no_treino": 0, "status": "DESCARTAR", "motivo": f"Colunas ausentes em {key}: {missing_cols}"}

    optional_found = []
    for key, pattern in OPTIONAL_PATTERNS.items():
        p = base_dir / pattern.format(date=day_str)
        if p.exists():
            optional_found.append(key)

    return {
        "data": day_str,
        "usar_no_treino": 1,
        "status": "OK",
        "motivo": "Dia apto",
        "rows_ia": int(len(dataframes["log_ia_exec"])),
        "rows_120s": int(len(dataframes["log_120s"])),
        "rows_labels": int(len(dataframes[label_kind])),
        "origem_labels": label_path.name,
        "artefatos_opcionais": ",".join(optional_found) if optional_found else "-",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seleciona dias válidos para treino do Lara.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Pasta dos logs/artefatos diários.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV de saída com os dias aprovados.")
    parser.add_argument("--start-date", default=None, help="Data mínima (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Data máxima (YYYY-MM-DD).")
    parser.add_argument("--exclude-dates", nargs="*", default=[], help="Datas para exclusão manual.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(args.base_dir)
    output = Path(args.output)
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    exclude_dates = set(args.exclude_dates)

    if not base_dir.exists():
        raise SystemExit(f"Base dir não encontrada: {base_dir}")

    all_days = find_all_dates(base_dir)
    rows = [
        check_day(
            base_dir=base_dir,
            day=day,
            start_date=start_date,
            end_date=end_date,
            exclude_dates=exclude_dates,
        )
        for day in all_days
    ]
    df = pd.DataFrame(rows).sort_values("data").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    total = len(df)
    ok = int(df["usar_no_treino"].sum()) if total else 0
    descartados = total - ok
    print(f"[OK] Seleção concluída: {ok} aprovados / {descartados} descartados / {total} dias avaliados.")
    print(f"[OK] CSV salvo em: {output}")
    if total:
        print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()

