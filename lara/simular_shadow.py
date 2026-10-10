# -*- coding: utf-8 -*-
"""
simular_shadow.py — CALIB_MODE (2026-01-20) — TIMEALIGN_FIX (2026-01-21)

Inclui:
- --mode strict|calib
- --pc / --pv overrides
- Sempre grava CSV com cabeçalho (mesmo com 0 trades)

FIX:
- Corrige desalinhamento típico de +3h entre hora do log e preços do MT5
- NÃO altera log_120s (ele é a referência). Usa log_120s apenas para auto-diagnóstico.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import os
import re
from dataclasses import dataclass, fields
from typing import Callable, Dict, List, Tuple

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

DATE_RE = re.compile(r"log_ia_exec_(\d{4}-\d{2}-\d{2})\.csv$")


# ---------------------------
# Dataclass de saída
# ---------------------------
@dataclass
class Trade:
    dia: str
    ticket_virtual: int
    entrada_hora: str
    saida_hora: str
    direcao: str
    preco_entrada: float
    preco_saida: float
    sl_preco: float
    tp_preco: float
    sl_points: float
    tp_points: float
    resultado_pontos: float
    motivo_saida: str
    duracao_s: int
    pc_entrada: float
    pv_entrada: float
    conf_dir_entrada: float
    limiar_dir_entrada: float
    regime_entrada: str
    lc_usado: float
    version_tag: str


# ---------------------------
# Utilitários
# ---------------------------
def _read_json(path: str) -> Dict:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_mt5() -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 não está disponível. Instale o pacote 'MetaTrader5' no Python.")
    if not mt5.initialize():
        err = mt5.last_error()
        raise RuntimeError(
            f"Falha ao inicializar MT5. last_error={err}. Abra o terminal e faça login, depois tente novamente."
        )


def _parse_day_time(dia: str, hhmmss: str) -> dt.datetime:
    """Retorna datetime naive no 'relógio do log' (BRT)."""
    y, m, d = map(int, dia.split("-"))
    parts = str(hhmmss).strip().split(":")
    if len(parts) == 2:
        hh, mm = map(int, parts)
        ss = 0
    else:
        hh, mm, ss = map(int, parts[:3])
    return dt.datetime(y, m, d, hh, mm, ss)


def _load_rates_m1(symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """
    Busca candles M1 no MT5.

    Observação:
    - O campo rates['time'] vem como epoch (segundos).
    - Dependendo do broker/terminal, esse epoch pode se comportar como UTC real
      ou como "relógio do servidor" (efeito +3h quando comparado ao log).
    - O alinhamento final é resolvido por auto-diagnóstico (se log_120s existir).
    """
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"Sem candles M1 retornados para {symbol} no intervalo {start} → {end}. "
            "Verifique se o símbolo está visível no Market Watch."
        )
    df = pd.DataFrame(rates)
    df["time"] = df["time"].astype("int64")
    df = df.sort_values("time").reset_index(drop=True)

    # dt_local será preenchido depois, conforme o modo de alinhamento escolhido
    df["dt_local"] = pd.to_datetime(df["time"], unit="s")  # placeholder (naive)
    return df


def _epoch_from_local_as_utc(t_local: dt.datetime) -> int:
    """
    Interpreta o datetime naive como UTC (sem aplicar timezone do SO).
    Isso remove o deslocamento de +3h quando o epoch do MT5 está ancorado
    no relógio do servidor/terminal.
    """
    return int(calendar.timegm(t_local.timetuple()))


def _epoch_from_local_brt_to_utc(t_local: dt.datetime) -> int:
    """
    Interpreta o datetime naive como BRT e converte para epoch UTC real.
    Usa offset fixo de -03:00 (America/Sao_Paulo atual; sem DST).
    """
    # BRT = UTC-3 => UTC = BRT + 3h
    t_utc = t_local + dt.timedelta(hours=3)
    return int(calendar.timegm(t_utc.timetuple()))


def _set_dt_local_column(df_rates: pd.DataFrame, mode: str) -> None:
    """
    Define df_rates['dt_local'] em "horário do log" (BRT), coerente com o modo.

    mode:
      - "SERVER_CLOCK": epoch do MT5 está ancorado no relógio do servidor (efeito +3h se usar BRT->UTC)
      - "UTC_EPOCH": epoch do MT5 é UTC real (precisa converter UTC->BRT para comparar com o log)
    """
    if mode == "UTC_EPOCH":
        # epoch é UTC real => converter UTC -> BRT (UTC-3) para comparar com hora do log
        # BRT = UTC - 3h
        df_rates["dt_local"] = pd.to_datetime(df_rates["time"], unit="s") - pd.Timedelta(hours=3)
    else:
        # epoch já acompanha o relógio "local do terminal" (BRT) em termos de clock-time
        df_rates["dt_local"] = pd.to_datetime(df_rates["time"], unit="s")


def _find_bar_index(df_rates: pd.DataFrame, t_local: dt.datetime, epoch_func: Callable[[dt.datetime], int]) -> int:
    ts_arr = df_rates["time"].to_numpy(dtype="int64", copy=False)
    target_ts = int(epoch_func(t_local))
    idx = ts_arr.searchsorted(target_ts, side="left")
    if idx <= 0:
        return 0
    if idx >= len(df_rates):
        return len(df_rates) - 1
    return int(idx)



def _infer_direction_from_scores(pc: float | None, pv: float | None) -> str:
    pc = pd.to_numeric(pc, errors="coerce")
    pv = pd.to_numeric(pv, errors="coerce")

    pc = float(pc) if pd.notna(pc) else 0.0
    pv = float(pv) if pd.notna(pv) else 0.0

    if pc > pv:
        return "BUY"
    if pv > pc:
        return "SELL"
    return ""


def _simulate_one_trade(
    df_rates: pd.DataFrame,
    entry_time_local: dt.datetime,
    direction: str,
    sl_points: float,
    tp_points: float,
    point: float,
    eod_time_local: dt.datetime,
    epoch_func: Callable[[dt.datetime], int],
) -> Tuple[dt.datetime, float, float, float, str]:
    i0 = _find_bar_index(df_rates, entry_time_local, epoch_func=epoch_func)
    entry_bar = df_rates.iloc[i0]
    entry_price = float(entry_bar["open"])

    if direction == "BUY":
        sl_price = entry_price - sl_points * point
        tp_price = entry_price + tp_points * point
    else:
        sl_price = entry_price + sl_points * point
        tp_price = entry_price - tp_points * point

    for i in range(i0, len(df_rates)):
        bar = df_rates.iloc[i]
        tbar = bar["dt_local"]
        tbar_dt = tbar.to_pydatetime() if hasattr(tbar, "to_pydatetime") else tbar

        # EOD no horário do log (BRT)
        if tbar_dt > eod_time_local:
            prev = df_rates.iloc[max(i - 1, i0)]
            prev_dt = prev["dt_local"]
            exit_time = prev_dt.to_pydatetime() if hasattr(prev_dt, "to_pydatetime") else prev_dt
            exit_price = float(prev["close"])
            pnl_points = (exit_price - entry_price) / point if direction == "BUY" else (entry_price - exit_price) / point
            return exit_time, entry_price, exit_price, pnl_points, "EOD"

        h = float(bar["high"])
        l = float(bar["low"])

        if direction == "BUY":
            hit_sl = l <= sl_price
            hit_tp = h >= tp_price
        else:
            hit_sl = h >= sl_price
            hit_tp = l <= tp_price

        if hit_sl and hit_tp:
            return tbar_dt, entry_price, sl_price, -sl_points, "SL"
        if hit_sl:
            return tbar_dt, entry_price, sl_price, -sl_points, "SL"
        if hit_tp:
            return tbar_dt, entry_price, tp_price, tp_points, "TP"

    last = df_rates.iloc[-1]
    last_dt = last["dt_local"]
    exit_time = last_dt.to_pydatetime() if hasattr(last_dt, "to_pydatetime") else last_dt
    exit_price = float(last["close"])
    pnl_points = (exit_price - entry_price) / point if direction == "BUY" else (entry_price - exit_price) / point
    return exit_time, entry_price, exit_price, pnl_points, "EOD"


def _parse_block_reasons(s: str) -> List[str]:
    parts = [p.strip().lower() for p in str(s).split("|")]
    return [p for p in parts if p]


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["hora", "pc", "pv", "conf_dir", "bloqueado_por", "regime_mercado", "limiar_confianca"]:
        if col not in df.columns:
            df[col] = None
    return df


def _infer_direction_from_regime(regime: str) -> str:
    r = str(regime).strip().lower()
    if r == "uptrend":
        return "BUY"
    if r == "downtrend":
        return "SELL"
    return ""


def _pick_candidates(
    df_ia: pd.DataFrame,
    cfg: Dict,
    lc: float,
    mode: str,
    pc_override: float | None,
    pv_override: float | None,
    debug: bool = True
) -> pd.DataFrame:
    df = _normalize_cols(df_ia.copy())

    n0 = len(df)

    # 1) regimes válidos
    df = df[df["regime_mercado"].isin(["uptrend", "downtrend"])]
    n1 = len(df)

    # 2) bloqueios permitidos
    if mode == "strict":
        allowed_sets = [set(["shadow_lock"])]
    else:
        allowed_sets = [set(["shadow_lock"]), set(["shadow_lock", "abaixo_limiar_pc_pv"])]

    def _allowed_blocks(s: str) -> bool:
        reasons = set(_parse_block_reasons(s))
        return any(reasons == a for a in allowed_sets)

    df = df[df["bloqueado_por"].fillna("").apply(_allowed_blocks)]
    n2 = len(df)

    # 3) gate PC/PV recalculado
    df["pc"] = pd.to_numeric(df["pc"], errors="coerce")
    df["pv"] = pd.to_numeric(df["pv"], errors="coerce")
    df["conf_dir"] = pd.to_numeric(df["conf_dir"], errors="coerce")

    pc_cfg = float(pc_override) if pc_override is not None else float(cfg.get("limiar_compra_base", 0.5) or 0.5)
    pv_cfg = float(pv_override) if pv_override is not None else float(cfg.get("limiar_venda_base", 0.5) or 0.5)

    direction = df.apply(lambda r: _infer_direction_from_scores(r.get("pc"), r.get("pv")), axis=1)
    df["direcao_calc"] = direction

    is_buy = df["direcao_calc"].eq("BUY")
    is_sell = df["direcao_calc"].eq("SELL")

    # Recalcula a confianca coerente com a direcao recalculada.
    df["conf_dir"] = None
    df.loc[is_buy, "conf_dir"] = df.loc[is_buy, "pc"]
    df.loc[is_sell, "conf_dir"] = df.loc[is_sell, "pv"]
    df["conf_dir"] = pd.to_numeric(df["conf_dir"], errors="coerce")

    df["limiar_dir_calc"] = None
    df.loc[is_buy, "limiar_dir_calc"] = pc_cfg
    df.loc[is_sell, "limiar_dir_calc"] = pv_cfg
    df["limiar_dir_calc"] = pd.to_numeric(df["limiar_dir_calc"], errors="coerce")

    df = df[df["conf_dir"].notna() & df["limiar_dir_calc"].notna()]
    df = df[df["conf_dir"] >= df["limiar_dir_calc"]]
    n3 = len(df)

    if debug and n3 == 0:
        conf_max = None
        try:
            conf_max = pd.to_numeric(df_ia.get("conf_dir", pd.Series(dtype=float)), errors="coerce").max()
        except Exception:
            pass

        top_bp = None
        try:
            bp = df_ia.get("bloqueado_por", pd.Series([""], dtype=str)).fillna("").astype(str)
            top_bp = bp.value_counts().head(8)
        except Exception:
            pass

        print("[WARN] Nenhum candidato para simulação neste dia.")
        print(f"       total={n0} -> regime_ok={n1} -> blocks_ok={n2} -> gate_pc_pv={n3}")
        if conf_max is not None and pd.notna(conf_max):
            print(f"       conf_dir_max_no_log (aprox): {conf_max}")
        print(f"       LC (apenas informativo) = {lc:.4f}")
        print(f"       PC_cfg={pc_cfg:.4f} PV_cfg={pv_cfg:.4f}")
        if top_bp is not None and len(top_bp) > 0:
            print("       Top bloqueado_por (raw):")
            for k, v in top_bp.items():
                print(f"         {v:>5}  {k}")

    return df.reset_index(drop=True)


def _detect_time_mode(
    dia: str,
    df_rates: pd.DataFrame,
    df120: pd.DataFrame | None,
    debug: bool = True
) -> Tuple[str, Callable[[dt.datetime], int]]:
    """
    Decide automaticamente entre:
      - SERVER_CLOCK  (remove efeito +3h)
      - UTC_EPOCH     (UTC real)

    Se log_120s existir, usa preços para decidir.
    Se não existir, assume UTC_EPOCH (padrão mais conservador).
    """
    if df120 is None or df120.empty or "hora" not in df120.columns or "preco" not in df120.columns:
        if debug:
            print("[INFO] log_120s não disponível para auto-diagnóstico. Usando modo UTC_EPOCH (padrão).")
        _set_dt_local_column(df_rates, "UTC_EPOCH")
        return "UTC_EPOCH", _epoch_from_local_brt_to_utc

    # Amostra de pontos do log_120s (até 12)
    sample = df120.dropna(subset=["hora", "preco"]).copy()
    if len(sample) > 12:
        sample = sample.iloc[0:12]

    times_local = [_parse_day_time(dia, h) for h in sample["hora"].astype(str).tolist()]
    prices_ref = sample["preco"].astype(float).to_numpy()

    ts_arr = df_rates["time"].to_numpy(dtype="int64", copy=False)

    def score(epoch_func: Callable[[dt.datetime], int]) -> float:
        diffs: List[float] = []
        for tloc, pref in zip(times_local, prices_ref):
            target_ts = int(epoch_func(tloc))
            idx = ts_arr.searchsorted(target_ts, side="left")
            if idx <= 0:
                idx = 0
            elif idx >= len(ts_arr):
                idx = len(ts_arr) - 1
            pbar = float(df_rates.iloc[int(idx)]["open"])
            diffs.append(abs(pbar - float(pref)))
        if not diffs:
            return float("inf")
        return float(sum(diffs) / len(diffs))

    # Avalia os dois modos
    score_server = score(_epoch_from_local_as_utc)       # trata hora do log como UTC (remove +3h)
    score_utc    = score(_epoch_from_local_brt_to_utc)   # converte BRT->UTC (UTC real)

    if debug:
        print(f"[TIME] score SERVER_CLOCK={score_server:.2f} | UTC_EPOCH={score_utc:.2f}")

    if score_server < score_utc:
        mode = "SERVER_CLOCK"
        _set_dt_local_column(df_rates, mode)
        if debug:
            print("[TIME] Modo escolhido: SERVER_CLOCK (corrige efeito +3h).")
        return mode, _epoch_from_local_as_utc

    mode = "UTC_EPOCH"
    _set_dt_local_column(df_rates, mode)
    if debug:
        print("[TIME] Modo escolhido: UTC_EPOCH (UTC real).")
    return mode, _epoch_from_local_brt_to_utc


def simulate_day(
    project_dir: str,
    dia: str,
    mode: str,
    pc_override: float | None,
    pv_override: float | None,
    dry_run: bool = False,
    debug: bool = True,
) -> str:
    cfg_path = os.path.join(project_dir, "config_lara.json")
    cfg = _read_json(cfg_path)

    logs_dir = os.path.join(project_dir, "arquivo_ABC")
    ia_path = os.path.join(logs_dir, f"log_ia_exec_{dia}.csv")
    if not os.path.exists(ia_path):
        raise FileNotFoundError(f"Não encontrei {ia_path}")

    df_ia = pd.read_csv(ia_path)

    try:
        u = sorted(set(df_ia.get("limiar_confianca", pd.Series([None])).dropna().astype(float).tolist()))
        lc = float(u[0]) if len(u) == 1 else float(cfg.get("limiar_confianca", 0.5))
    except Exception:
        lc = float(cfg.get("limiar_confianca", 0.5))

    sl_points = float(cfg.get("sl_points", 150))
    tp_points = float(cfg.get("tp_points", 150))

    hora_inicio_operacao = str(cfg.get("hora_inicio_operacao", "") or "").strip()
    skip_open_min = int(cfg.get("skip_open_min", 0) or 0)

    cooldown_apenas_apos_loss = bool(cfg.get("cooldown_apenas_apos_loss", False))
    cooldown_barras = int(
        cfg.get("cooldown_apos_loss_barras", cfg.get("cooldown_candles", 3)) or 0
    )

    # simulador trabalha em M1, então 1 barra = 1 minuto
    cooldown_min = cooldown_barras

    symbol = str(cfg.get("symbol", "WING26"))
    p120 = os.path.join(logs_dir, f"log_120s_{dia}.csv")
    df120 = None
    try:
        if os.path.exists(p120):
            df120 = pd.read_csv(p120)
            su = sorted(set(df120.get("symbol", pd.Series([symbol])).dropna().astype(str).tolist()))
            if len(su) == 1 and su[0]:
                symbol = su[0]
    except Exception:
        df120 = None

    version_tag = str(cfg.get("version_tag", ""))

    _ensure_mt5()
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info({symbol}) retornou None. Verifique o símbolo e Market Watch.")
    point = float(getattr(info, "point", 1.0) or 1.0)

    # Busca um range amplo para não ficar refém de timezone do terminal
    start = _parse_day_time(dia, "00:00:00")
    end = _parse_day_time(dia, "23:59:59")

    df_rates = _load_rates_m1(symbol, start, end)

    # Decide automaticamente a base de tempo correta (usando log_120s como referência, sem alterá-lo)
    time_mode, epoch_func = _detect_time_mode(dia=dia, df_rates=df_rates, df120=df120, debug=debug)

    # EOD no relógio do log (BRT)
    eod_time = _parse_day_time(dia, "17:58:00")

    hist_max_raw = df_rates["dt_local"].iloc[-1]
    hist_max = hist_max_raw.to_pydatetime() if hasattr(hist_max_raw, "to_pydatetime") else hist_max_raw

    df_cand = _pick_candidates(df_ia, cfg, lc, mode, pc_override, pv_override, debug=debug)

    out_path = os.path.join(logs_dir, f"log_operacoes_shadow_v3_{dia}.csv")
    if dry_run:
        return out_path

    if df_cand is None or df_cand.empty:
        cols = [f.name for f in fields(Trade)]
        pd.DataFrame(columns=cols).to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    if hora_inicio_operacao:
        start_allowed_time = _parse_day_time(dia, hora_inicio_operacao)
    else:
        start_allowed_time = _parse_day_time(dia, "09:15:00") + dt.timedelta(minutes=skip_open_min)

    trades: List[Trade] = []
    next_allowed_time = start_allowed_time
    ticket = 0

    if debug:
        print(f"[TIME] time_mode={time_mode} | symbol={symbol}")
        print(f"[CFG] hora_inicio_operacao={hora_inicio_operacao!r} | skip_open_min={skip_open_min}")
        print(f"[CFG] cooldown_apenas_apos_loss={cooldown_apenas_apos_loss} | cooldown_barras={cooldown_barras} | cooldown_min={cooldown_min}")
        print(f"[CFG] start_allowed_time={start_allowed_time}")

    for _, row in df_cand.iterrows():
        t_signal = _parse_day_time(dia, str(row["hora"]))

        # Se, por algum motivo, o sinal cair depois do histórico disponível, ignora
        if t_signal > hist_max:
            continue
        if t_signal < next_allowed_time:
            continue

        regime = str(row.get("regime_mercado", ""))
        direction = str(row.get("direcao_calc", "")).strip().upper()
        if direction not in ("BUY", "SELL"):
            direction = _infer_direction_from_scores(row.get("pc"), row.get("pv"))

        if direction not in ("BUY", "SELL"):
            continue

        exit_time, entry_price, exit_price, pnl_points, motivo = _simulate_one_trade(
            df_rates=df_rates,
            entry_time_local=t_signal,
            direction=direction,
            sl_points=sl_points,
            tp_points=tp_points,
            point=point,
            eod_time_local=eod_time,
            epoch_func=epoch_func,
        )

        ticket += 1
        dur = int((exit_time - t_signal).total_seconds())
        sl_price = (entry_price - sl_points * point) if direction == "BUY" else (entry_price + sl_points * point)
        tp_price = (entry_price + tp_points * point) if direction == "BUY" else (entry_price - tp_points * point)

        pc = float(pd.to_numeric(row.get("pc", None), errors="coerce") or 0.0)
        pv = float(pd.to_numeric(row.get("pv", None), errors="coerce") or 0.0)
        conf_dir = float(pd.to_numeric(row.get("conf_dir", None), errors="coerce") or 0.0)
        limiar_dir_calc = float(pd.to_numeric(row.get("limiar_dir_calc", None), errors="coerce") or 0.0)

        trades.append(
            Trade(
                dia=dia,
                ticket_virtual=ticket,
                entrada_hora=t_signal.strftime("%H:%M:%S"),
                saida_hora=exit_time.strftime("%H:%M:%S"),
                direcao=direction,
                preco_entrada=round(float(entry_price), 2),
                preco_saida=round(float(exit_price), 2),
                sl_preco=round(float(sl_price), 2),
                tp_preco=round(float(tp_price), 2),
                sl_points=sl_points,
                tp_points=tp_points,
                resultado_pontos=round(float(pnl_points), 2),
                motivo_saida=motivo,
                duracao_s=dur,
                pc_entrada=pc,
                pv_entrada=pv,
                conf_dir_entrada=conf_dir,
                limiar_dir_entrada=limiar_dir_calc,
                regime_entrada=regime,
                lc_usado=float(lc),
                version_tag=version_tag,
            )
        )

        if cooldown_apenas_apos_loss:
            if float(pnl_points) < 0:
                next_allowed_time = exit_time + dt.timedelta(minutes=cooldown_min)
            else:
                next_allowed_time = exit_time
        else:
            next_allowed_time = exit_time + dt.timedelta(minutes=cooldown_min)

    df_out = pd.DataFrame([t.__dict__ for t in trades])
    if df_out.empty:
        cols = [f.name for f in fields(Trade)]
        df_out = pd.DataFrame(columns=cols)

    df_out.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def list_days(logs_dir: str) -> List[str]:
    days: List[str] = []
    for fn in os.listdir(logs_dir):
        m = DATE_RE.search(fn)
        if m:
            days.append(m.group(1))
    return sorted(set(days))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=".", help="Pasta raiz do Lara (onde está o config_lara.json)")
    ap.add_argument("--day", default="", help="Dia no formato YYYY-MM-DD (opcional)")
    ap.add_argument("--all", action="store_true", help="Processa todos os dias com log_ia_exec_*.csv")

    ap.add_argument("--mode", default="strict", choices=["strict", "calib"],
                    help="strict=apenas shadow_lock; calib=shadow_lock ou shadow_lock+abaixo_limiar_pc_pv")
    ap.add_argument("--pc", type=float, default=None, help="Override PC (limiar_compra_base) para o gate (útil no modo calib)")
    ap.add_argument("--pv", type=float, default=None, help="Override PV (limiar_venda_base) para o gate (útil no modo calib)")
    ap.add_argument("--no-debug", action="store_true", help="Desliga prints de diagnóstico")

    args = ap.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    logs_dir = os.path.join(project_dir, "arquivo_ABC")

    if not os.path.isdir(logs_dir):
        print(f"[ERRO] Não encontrei logs_dir: {logs_dir}")
        return 2

    if not args.day and not args.all:
        print("[ERRO] Use --day YYYY-MM-DD ou --all")
        return 2

    debug = not args.no_debug

    try:
        if args.day:
            out = simulate_day(
                project_dir=project_dir,
                dia=args.day,
                mode=args.mode,
                pc_override=args.pc,
                pv_override=args.pv,
                debug=debug
            )
            print(f"[OK] Gerado: {out}")
        else:
            for dia in list_days(logs_dir):
                out = simulate_day(
                    project_dir=project_dir,
                    dia=dia,
                    mode=args.mode,
                    pc_override=args.pc,
                    pv_override=args.pv,
                    debug=debug
                )
                print(f"[OK] {dia}: {out}")
        return 0
    except Exception as e:
        print(f"[ERRO] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



