# -*- coding: utf-8 -*-
r"""
replay_volatilidade.py

Objetivo
--------
Fazer replay histórico do efeito de volatilidade em cima dos log_ia_exec já existentes,
gerando dois cenários por dia:

- VOL_OFF: sem kill switch e sem ajuste de limiar por volatilidade
- VOL_ON : com kill switch e com ajuste_limiar_vol configurável

Depois, o script:
1) cria uma cópia modificada do log_ia_exec para cada cenário
2) substitui temporariamente o log_ia_exec oficial do dia
3) chama o simular_shadow.py
4) coleta métricas por dia
5) gera resumo agregado OFF vs ON

Importante
----------
Este script NÃO reexecuta o runner principal no passado.
Ele faz um replay contrafactual em cima do log_ia_exec histórico.

Quando ele é válido?
- quando o log_ia_exec já contém pc, pv, vol_norm, bloqueado_por e direção/sinal suficientes
- quando queremos medir especificamente o efeito do bloco simples:
    * vol_norm >= 1.5  -> sobe limiares por ajuste_limiar_vol
    * vol_norm >= 2.5  -> bloqueia por kill_switch_vol

Uso
---
python .\replay_volatilidade.py `
  --project-root . `
  --dias 2026-04-14 2026-04-15 2026-04-16 2026-04-17 2026-04-20 2026-04-22 2026-04-23 2026-04-24 2026-04-27 2026-04-28 `
  --pc 0.70 `
  --pv 0.60 `
  --ajuste-on 0.05

Saídas
------
comparativos_volatilidade_historica\VOL_REPLAY_<timestamp>\
  - resultado_detalhado_por_dia.csv
  - resultado_agregado.csv
  - auditoria_log_ia.csv
  - conclusao.txt
  - logs\VOL_OFF\...
  - logs\VOL_ON\...
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# Helpers
# ============================================================

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def try_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return None
    # tenta lidar com vírgula decimal sem quebrar ponto decimal normal
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def longest_existing(paths: List[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def normalize_blocked_tokens(value: object) -> List[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    for sep in ["|", ";", ","]:
        s = s.replace(sep, "|")
    parts = [p.strip() for p in s.split("|")]
    parts = [p for p in parts if p]
    seen = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return seen


def join_blocked_tokens(tokens: List[str]) -> str:
    tokens = [t for t in tokens if str(t).strip()]
    return "|".join(tokens)


def infer_direction(row: pd.Series) -> str:
    # prioridade para colunas existentes
    for col in ["direcao_calc", "direcao", "side", "sinal", "decisao"]:
        if col in row.index:
            val = str(row.get(col, "")).strip().upper()
            if val in {"BUY", "SELL", "COMPRA", "VENDA"}:
                return "BUY" if val in {"BUY", "COMPRA"} else "SELL"

    pc = try_float(row.get("pc"))
    pv = try_float(row.get("pv"))
    if pc is None and pv is None:
        return ""
    pc = 0.0 if pc is None else pc
    pv = 0.0 if pv is None else pv
    return "BUY" if pc >= pv else "SELL"


def normalize_bool_like(x) -> int:
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "sim"}:
        return 1
    if s in {"0", "false", "f", "no", "nao", "não", ""}:
        return 0
    try:
        return 1 if float(s) != 0 else 0
    except Exception:
        return 0


def maybe_set(df: pd.DataFrame, col: str, values) -> None:
    if col in df.columns:
        df[col] = values


def first_existing(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


# ============================================================
# Métricas do shadow
# ============================================================

PREFERRED_PNL_COLUMNS = [
    "lucro_real",
    "resultado",
    "pnl",
    "profit",
    "resultado_liquido",
    "resultado_pontos",
    "pontos_resultado",
    "pontos",
]


@dataclass
class ScenarioMetrics:
    dia: str
    cenario: str
    arquivo_origem: str
    operacoes: int
    wins: int
    losses: int
    empates: int
    lucro_total: float
    lucro_medio: float
    winrate: float
    profit_factor: float
    max_drawdown: float
    coluna_pnl: str
    observacao: str


def infer_pnl_column(rows: List[dict]) -> Optional[str]:
    if not rows:
        return None
    cols = rows[0].keys()
    for col in PREFERRED_PNL_COLUMNS:
        if col in cols:
            vals = [try_float(r.get(col)) for r in rows]
            if any(v is not None for v in vals):
                return col
    for col in cols:
        vals = [try_float(r.get(col)) for r in rows[:50]]
        valid = [v for v in vals if v is not None]
        if len(valid) >= max(3, len(vals) // 4):
            return col
    return None


def calc_drawdown(cumulative: List[float]) -> float:
    if not cumulative:
        return 0.0
    peak = cumulative[0]
    max_dd = 0.0
    for x in cumulative:
        peak = max(peak, x)
        dd = peak - x
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_metrics(dia: str, cenario: str, csv_path: Path) -> ScenarioMetrics:
    rows = read_csv_rows(csv_path)
    if not rows:
        return ScenarioMetrics(dia, cenario, str(csv_path), 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "", "CSV vazio")

    pnl_col = infer_pnl_column(rows)
    if not pnl_col:
        return ScenarioMetrics(dia, cenario, str(csv_path), 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "", "Sem coluna PnL")

    pnls = []
    for r in rows:
        v = try_float(r.get(pnl_col))
        if v is not None:
            pnls.append(v)

    if not pnls:
        return ScenarioMetrics(dia, cenario, str(csv_path), 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, pnl_col, "Coluna PnL sem valores")

    wins = sum(1 for x in pnls if x > 0)
    losses = sum(1 for x in pnls if x < 0)
    empates = sum(1 for x in pnls if x == 0)
    total = sum(pnls)
    avg = total / len(pnls) if pnls else 0.0
    gross_profit = sum(x for x in pnls if x > 0)
    gross_loss_abs = abs(sum(x for x in pnls if x < 0))
    if gross_loss_abs == 0:
        pf = float("inf") if gross_profit > 0 else 0.0
    else:
        pf = gross_profit / gross_loss_abs

    acc = 0.0
    curve = []
    for x in pnls:
        acc += x
        curve.append(acc)

    return ScenarioMetrics(
        dia=dia,
        cenario=cenario,
        arquivo_origem=str(csv_path),
        operacoes=len(pnls),
        wins=wins,
        losses=losses,
        empates=empates,
        lucro_total=round(total, 6),
        lucro_medio=round(avg, 6),
        winrate=round((wins / len(pnls) * 100.0) if pnls else 0.0, 4),
        profit_factor=(999999.0 if math.isinf(pf) else round(pf, 6)),
        max_drawdown=round(calc_drawdown(curve), 6),
        coluna_pnl=pnl_col,
        observacao="OK",
    )


def aggregate_metrics(metrics: List[ScenarioMetrics], cenario: str) -> dict:
    subset = [m for m in metrics if m.cenario == cenario]
    ops = sum(m.operacoes for m in subset)
    wins = sum(m.wins for m in subset)
    losses = sum(m.losses for m in subset)
    empates = sum(m.empates for m in subset)
    lucro_total = sum(m.lucro_total for m in subset)
    lucro_medio = (lucro_total / ops) if ops else 0.0

    gross_profit = sum(m.lucro_total for m in subset if m.lucro_total > 0)
    gross_loss_abs = abs(sum(m.lucro_total for m in subset if m.lucro_total < 0))
    pf = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (999999.0 if gross_profit > 0 else 0.0)

    acc = 0.0
    curve = []
    for m in subset:
        acc += m.lucro_total
        curve.append(acc)

    return {
        "cenario": cenario,
        "dias": len(subset),
        "operacoes": ops,
        "wins": wins,
        "losses": losses,
        "empates": empates,
        "lucro_total": round(lucro_total, 6),
        "lucro_medio_por_operacao": round(lucro_medio, 6),
        "winrate": round((wins / ops * 100.0) if ops else 0.0, 4),
        "profit_factor_proxy": round(pf, 6),
        "max_drawdown_por_dia_proxy": round(calc_drawdown(curve), 6),
    }


# ============================================================
# Replay do log_ia
# ============================================================

def apply_vol_scenario_to_log(df: pd.DataFrame, scenario: str, pc_base: float, pv_base: float, ajuste_on: float) -> Tuple[pd.DataFrame, dict]:
    out = df.copy()

    # Garantia mínima de colunas
    for col in ["pc", "pv", "vol_norm", "bloqueado_por"]:
        if col not in out.columns:
            out[col] = ""

    if "conf_dir" not in out.columns:
        out["conf_dir"] = ""

    if "direcao_calc" not in out.columns:
        out["direcao_calc"] = ""

    if "limiar_dir_calc" not in out.columns:
        out["limiar_dir_calc"] = ""

    if "limiar_compra_ajustado" not in out.columns:
        out["limiar_compra_ajustado"] = ""

    if "limiar_venda_ajustado" not in out.columns:
        out["limiar_venda_ajustado"] = ""

    # auditoria
    if "vol_replay_scenario" not in out.columns:
        out["vol_replay_scenario"] = ""
    if "vol_replay_changed" not in out.columns:
        out["vol_replay_changed"] = 0
    if "vol_replay_reason" not in out.columns:
        out["vol_replay_reason"] = ""

    # colunas que o simulador pode usar
    for col in ["shadow_lock", "abaixo_limiar_pc_pv"]:
        if col not in out.columns:
            out[col] = 0

    total_rows = len(out)
    vol_ge_15 = 0
    vol_ge_25 = 0
    changed_rows = 0
    kill_rows = 0
    threshold_tight_rows = 0

    new_shadow_lock = []
    new_below = []
    new_blocked = []
    new_conf = []
    new_dir = []
    new_lc = []
    new_lv = []
    new_ld = []
    new_reason = []
    new_changed = []

    # sinais de base preservados quando possível
    orig_shadow = out["shadow_lock"].tolist() if "shadow_lock" in out.columns else [0] * total_rows

    for idx, row in out.iterrows():
        direction = infer_direction(row)
        pc = try_float(row.get("pc")) or 0.0
        pv = try_float(row.get("pv")) or 0.0
        vol = try_float(row.get("vol_norm")) or 0.0

        if vol >= 1.5:
            vol_ge_15 += 1
        if vol >= 2.5:
            vol_ge_25 += 1

        lim_c = pc_base
        lim_v = pv_base
        blocked = normalize_blocked_tokens(row.get("bloqueado_por"))

        # remove traço anterior de kill switch antes de reaplicar cenário
        blocked = [b for b in blocked if b != "kill_switch_vol"]

        reasons = []
        changed = 0

        if scenario == "VOL_ON":
            if vol >= 1.5:
                lim_c = min(0.80, pc_base + ajuste_on)
                lim_v = min(0.80, pv_base + ajuste_on)
                threshold_tight_rows += 1
                changed = 1
                reasons.append("threshold_up")
            if vol >= 2.5:
                blocked.append("kill_switch_vol")
                kill_rows += 1
                changed = 1
                reasons.append("kill_switch")
        elif scenario == "VOL_OFF":
            lim_c = pc_base
            lim_v = pv_base
        else:
            raise ValueError(f"Cenário inválido: {scenario}")

        # conf_dir de replay
        conf_dir = pc if direction == "BUY" else pv if direction == "SELL" else max(pc, pv)

        # se há qualquer bloqueio, o replay trava o sinal
        has_any_block = len(blocked) > 0
        lim_dir = lim_c if direction == "BUY" else lim_v if direction == "SELL" else ""

        # decide lock/below no replay
        if direction == "BUY":
            pass_threshold = pc >= lim_c
        elif direction == "SELL":
            pass_threshold = pv >= lim_v
        else:
            pass_threshold = False

        # regra de replay:
        # - qualquer bloqueio -> shadow_lock=0
        # - sem bloqueio e acima do limiar -> 1
        # - sem bloqueio e abaixo do limiar -> 0 + abaixo_limiar=1
        if has_any_block:
            shadow_lock = 0
            below = 0
        else:
            shadow_lock = 1 if pass_threshold else 0
            below = 0 if pass_threshold else 1

        # se já havia shadow_lock=0 no original e agora ficaria 1,
        # mantemos 1 porque estamos justamente reavaliando o efeito do bloco
        # com base em pc/pv/vol e bloqueado_por do replay.

        if changed:
            changed_rows += 1

        new_shadow_lock.append(shadow_lock)
        new_below.append(below)
        new_blocked.append(join_blocked_tokens(blocked))
        new_conf.append(conf_dir)
        new_dir.append(direction)
        new_lc.append(lim_c)
        new_lv.append(lim_v)
        new_ld.append(lim_dir)
        new_reason.append("|".join(reasons))
        new_changed.append(changed)

    out["shadow_lock"] = new_shadow_lock
    out["abaixo_limiar_pc_pv"] = new_below
    out["bloqueado_por"] = new_blocked
    out["conf_dir"] = new_conf
    out["direcao_calc"] = new_dir
    out["limiar_compra_ajustado"] = new_lc
    out["limiar_venda_ajustado"] = new_lv
    out["limiar_dir_calc"] = new_ld
    out["vol_replay_scenario"] = scenario
    out["vol_replay_changed"] = new_changed
    out["vol_replay_reason"] = new_reason

    audit = {
        "cenario": scenario,
        "rows_total": total_rows,
        "rows_vol_ge_1_5": vol_ge_15,
        "rows_vol_ge_2_5": vol_ge_25,
        "rows_changed": changed_rows,
        "rows_threshold_up": threshold_tight_rows,
        "rows_kill_switch": kill_rows,
        "shadow_lock_1": int((out["shadow_lock"].astype(int) == 1).sum()),
        "shadow_lock_0": int((out["shadow_lock"].astype(int) == 0).sum()),
    }
    return out, audit


# ============================================================
# Chamada do simulador
# ============================================================

def find_generated_shadow_log(project_root: Path, dia: str) -> Optional[Path]:
    folder = project_root / "arquivo_ABC"
    candidates = list(folder.glob(f"log_operacoes_shadow_v3_{dia}*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_simulator(project_root: Path, dia: str, pc: float, pv: float, timeout: int) -> Tuple[int, str, str]:
    sim_script = project_root / "simular_shadow.py"
    if not sim_script.exists():
        raise FileNotFoundError(f"Simulador não encontrado: {sim_script}")

    cmd = [
        sys.executable,
        str(sim_script),
        "--project-dir", str(project_root),
        "--day", dia,
        "--mode", "calib",
        "--pc", str(pc),
        "--pv", str(pv),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def compare_winner(agg_off: dict, agg_on: dict) -> Tuple[str, List[str]]:
    score_off = 0
    score_on = 0
    reasons = []

    def gt(a, b):
        return float(a) > float(b)

    lucro_off = float(agg_off["lucro_total"])
    lucro_on = float(agg_on["lucro_total"])
    pf_off = float(agg_off["profit_factor_proxy"])
    pf_on = float(agg_on["profit_factor_proxy"])
    wr_off = float(agg_off["winrate"])
    wr_on = float(agg_on["winrate"])
    dd_off = float(agg_off["max_drawdown_por_dia_proxy"])
    dd_on = float(agg_on["max_drawdown_por_dia_proxy"])

    if lucro_on > lucro_off:
        score_on += 3
        reasons.append(f"VOL_ON ganhou em lucro total ({lucro_on:.2f} > {lucro_off:.2f})")
    elif lucro_off > lucro_on:
        score_off += 3
        reasons.append(f"VOL_OFF ganhou em lucro total ({lucro_off:.2f} > {lucro_on:.2f})")

    if pf_on > pf_off:
        score_on += 2
        reasons.append(f"VOL_ON ganhou em PF proxy ({pf_on:.3f} > {pf_off:.3f})")
    elif pf_off > pf_on:
        score_off += 2
        reasons.append(f"VOL_OFF ganhou em PF proxy ({pf_off:.3f} > {pf_on:.3f})")

    if wr_on > wr_off:
        score_on += 1
        reasons.append(f"VOL_ON ganhou em winrate ({wr_on:.2f}% > {wr_off:.2f}%)")
    elif wr_off > wr_on:
        score_off += 1
        reasons.append(f"VOL_OFF ganhou em winrate ({wr_off:.2f}% > {wr_on:.2f}%)")

    if dd_on < dd_off:
        score_on += 2
        reasons.append(f"VOL_ON ganhou em drawdown ({dd_on:.2f} < {dd_off:.2f})")
    elif dd_off < dd_on:
        score_off += 2
        reasons.append(f"VOL_OFF ganhou em drawdown ({dd_off:.2f} < {dd_on:.2f})")

    if score_on > score_off:
        return "VOL_ON_SUPERIOR", reasons
    if score_off > score_on:
        return "VOL_OFF_SUPERIOR", reasons
    return "EMPATE_TECNICO", reasons


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Replay histórico do efeito de volatilidade em log_ia_exec + shadow.")
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2] / "lara"), help="Pasta lara/ do projeto")
    ap.add_argument("--dias", nargs="+", required=True, help="Dias no formato YYYY-MM-DD")
    ap.add_argument("--pc", type=float, required=True, help="Limiar base de compra")
    ap.add_argument("--pv", type=float, required=True, help="Limiar base de venda")
    ap.add_argument("--ajuste-on", type=float, default=0.05, help="Ajuste de limiar quando VOL_ON e vol_norm>=1.5")
    ap.add_argument("--timeout", type=int, default=300, help="Timeout por simulação")
    ap.add_argument("--saida-dir", default="comparativos_volatilidade_historica", help="Pasta de saída")
    return ap.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    logs_dir = project_root / "arquivo_ABC"
    out_root = project_root / args.saida_dir / f"VOL_REPLAY_{now_tag()}"
    out_logs = out_root / "logs"
    ensure_dir(out_logs)

    detailed_metrics: List[ScenarioMetrics] = []
    exec_rows: List[dict] = []
    audit_rows: List[dict] = []

    scenarios = ["VOL_OFF", "VOL_ON"]

    for dia in args.dias:
        ia_original = logs_dir / f"log_ia_exec_{dia}.csv"
        if not ia_original.exists():
            print(f"[WARN] log_ia_exec não encontrado para {dia}: {ia_original}")
            exec_rows.append({"dia": dia, "cenario": "", "status": "LOG_IA_NAO_ENCONTRADO", "arquivo": str(ia_original)})
            continue

        # backup do original em memória via cópia física
        backup_original = out_root / "backups" / f"log_ia_exec_{dia}.csv"
        safe_copy(ia_original, backup_original)

        try:
            df_orig = pd.read_csv(ia_original, encoding="utf-8-sig")
        except Exception:
            df_orig = pd.read_csv(ia_original)

        for scenario in scenarios:
            print("=" * 90)
            print(f"Dia {dia} | {scenario}")
            print("=" * 90)

            df_mod, audit = apply_vol_scenario_to_log(
                df=df_orig,
                scenario=scenario,
                pc_base=float(args.pc),
                pv_base=float(args.pv),
                ajuste_on=float(args.ajuste_on),
            )
            audit["dia"] = dia
            audit_rows.append(audit)

            scenario_dir = out_logs / scenario
            ensure_dir(scenario_dir)

            scenario_log_ia = scenario_dir / f"log_ia_exec_{dia}_{scenario}.csv"
            df_mod.to_csv(scenario_log_ia, index=False, encoding="utf-8-sig")

            # substitui temporariamente o oficial pelo cenário
            safe_copy(scenario_log_ia, ia_original)

            ret, stdout, stderr = run_simulator(project_root, dia, args.pc, args.pv, args.timeout)
            stdout_path = scenario_dir / f"{dia}_stdout.txt"
            stderr_path = scenario_dir / f"{dia}_stderr.txt"
            stdout_path.write_text(stdout or "", encoding="utf-8")
            stderr_path.write_text(stderr or "", encoding="utf-8")

            exec_row = {
                "dia": dia,
                "cenario": scenario,
                "status": "OK" if ret == 0 else "FALHA_SIMULADOR",
                "returncode": ret,
                "log_ia_replay": str(scenario_log_ia),
                "stdout_file": str(stdout_path),
                "stderr_file": str(stderr_path),
                "rows_changed": audit["rows_changed"],
                "rows_kill_switch": audit["rows_kill_switch"],
                "rows_vol_ge_1_5": audit["rows_vol_ge_1_5"],
                "rows_vol_ge_2_5": audit["rows_vol_ge_2_5"],
            }

            if ret != 0:
                exec_rows.append(exec_row)
                print(f"[FALHA] simulador retornou código {ret}")
                continue

            gen_shadow = find_generated_shadow_log(project_root, dia)
            if not gen_shadow:
                exec_row["status"] = "SHADOW_NAO_ENCONTRADO"
                exec_rows.append(exec_row)
                print("[FALHA] não encontrei log shadow gerado")
                continue

            shadow_copy = scenario_dir / f"log_operacoes_shadow_v3_{dia}_{scenario}.csv"
            safe_copy(gen_shadow, shadow_copy)

            m = compute_metrics(dia, scenario, shadow_copy)
            detailed_metrics.append(m)

            exec_row.update({
                "shadow_file": str(shadow_copy),
                "operacoes": m.operacoes,
                "lucro_total": m.lucro_total,
                "winrate": m.winrate,
                "profit_factor": m.profit_factor,
                "max_drawdown": m.max_drawdown,
            })
            exec_rows.append(exec_row)

            print(
                f"[OK] {scenario} | rows_changed={audit['rows_changed']} | "
                f"kill={audit['rows_kill_switch']} | ops={m.operacoes} | "
                f"lucro={m.lucro_total:.2f} | wr={m.winrate:.2f}% | PF={m.profit_factor:.3f}"
            )

        # restaura original ao final do dia
        safe_copy(backup_original, ia_original)

    # restaura tudo de novo por segurança
    for dia in args.dias:
        backup_original = out_root / "backups" / f"log_ia_exec_{dia}.csv"
        ia_original = logs_dir / f"log_ia_exec_{dia}.csv"
        if backup_original.exists():
            safe_copy(backup_original, ia_original)

    detailed_rows = [asdict(m) for m in detailed_metrics]
    write_csv_rows(out_root / "resultado_detalhado_por_dia.csv", detailed_rows)
    write_csv_rows(out_root / "execucao_tecnica.csv", exec_rows)
    write_csv_rows(out_root / "auditoria_log_ia.csv", audit_rows)

    agg_off = aggregate_metrics(detailed_metrics, "VOL_OFF")
    agg_on = aggregate_metrics(detailed_metrics, "VOL_ON")
    agg_rows = [agg_off, agg_on]
    write_csv_rows(out_root / "resultado_agregado.csv", agg_rows)

    winner, reasons = compare_winner(agg_off, agg_on)

    lines = []
    lines.append("REPLAY HISTÓRICO DO BLOCO DE VOLATILIDADE")
    lines.append("")
    lines.append(f"Projeto: {project_root}")
    lines.append(f"PC base = {args.pc:.2f}")
    lines.append(f"PV base = {args.pv:.2f}")
    lines.append(f"Ajuste VOL_ON = {args.ajuste_on:.2f}")
    lines.append(f"Dias = {', '.join(args.dias)}")
    lines.append("")
    lines.append("Cenários:")
    lines.append("- VOL_OFF: sem kill switch / sem ajuste de limiar")
    lines.append("- VOL_ON : com kill switch / com ajuste de limiar quando vol_norm >= 1.5")
    lines.append("")
    lines.append("Resultado agregado:")
    lines.append(
        f"VOL_OFF -> lucro_total={agg_off['lucro_total']:.2f} | "
        f"ops={agg_off['operacoes']} | wr={agg_off['winrate']:.2f}% | "
        f"PFproxy={agg_off['profit_factor_proxy']:.3f} | DDproxy={agg_off['max_drawdown_por_dia_proxy']:.2f}"
    )
    lines.append(
        f"VOL_ON  -> lucro_total={agg_on['lucro_total']:.2f} | "
        f"ops={agg_on['operacoes']} | wr={agg_on['winrate']:.2f}% | "
        f"PFproxy={agg_on['profit_factor_proxy']:.3f} | DDproxy={agg_on['max_drawdown_por_dia_proxy']:.2f}"
    )
    lines.append("")
    lines.append(f"Conclusão automática: {winner}")
    lines.append("")
    lines.append("Razões:")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("Leitura correta deste teste:")
    lines.append("- Este teste mede o efeito contrafactual do bloco de volatilidade sobre os log_ia_exec históricos.")
    lines.append("- Ele não recompõe o runner principal candle a candle no passado; ele reaplica a regra de volatilidade nos logs.")
    lines.append("- Se houver divergência relevante, aí sim vale montar uma segunda etapa com rerun completo do pipeline principal.")
    lines.append("")
    lines.append("Audite primeiro:")
    lines.append("- auditoria_log_ia.csv")
    lines.append("- rows_changed")
    lines.append("- rows_kill_switch")
    lines.append("- diferenças em shadow_lock, limiar_dir_calc e bloqueado_por")

    (out_root / "conclusao.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 90)
    print("ARQUIVOS GERADOS")
    print("=" * 90)
    print(out_root)
    print(" - resultado_detalhado_por_dia.csv")
    print(" - resultado_agregado.csv")
    print(" - auditoria_log_ia.csv")
    print(" - execucao_tecnica.csv")
    print(" - conclusao.txt")
    print(" - logs/VOL_OFF")
    print(" - logs/VOL_ON")
    print("=" * 90)


if __name__ == "__main__":
    main()
