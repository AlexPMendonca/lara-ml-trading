import re
import math
import json
import shutil
from pathlib import Path
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_DIR / "lara"
LOG_DIR = PROJECT_DIR / "arquivo_ABC"  # logs diários gerados pelo motor (não versionados)
OUT_DIR = REPO_DIR / "reports" / "backtest" / "bateria_pc058_pv058_10d"

PC_BASE = 0.58
PV_BASE = 0.58

SL_DEFAULT = 150.0
TP_DEFAULT = 150.0

EMA_PAIRS = [(20, 100), (9, 21)]


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def normalizar_colunas(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def smart_num_value(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if not s:
        return np.nan

    s = s.replace("\ufeff", "")
    s = re.sub(r"[^\d,\.\-\+eE]", "", s)

    if s in ("", "+", "-", ".", ","):
        return np.nan

    if "e" in s.lower() and "," not in s:
        try:
            return float(s)
        except Exception:
            return np.nan

    if "," in s and "." in s:
        # Decide pelo último separador.
        if s.rfind(",") > s.rfind("."):
            # padrão brasileiro: 135.000,50
            s = s.replace(".", "").replace(",", ".")
        else:
            # padrão internacional: 135,000.50
            s = s.replace(",", "")

    elif "," in s:
        # vírgula decimal
        if s.count(",") > 1:
            partes = s.split(",")
            s = "".join(partes[:-1]) + "." + partes[-1]
        else:
            s = s.replace(",", ".")

    elif "." in s:
        if s.count(".") > 1:
            partes = s.split(".")
            s = "".join(partes[:-1]) + "." + partes[-1]
        else:
            antes, depois = s.split(".")
            antes_limpo = antes.replace("+", "").replace("-", "")

            # Caso típico de preço brasileiro sem decimal: 135.000 -> 135000.
            # Mas preserva probabilidade 0.580 -> 0.58.
            if (
                depois.isdigit()
                and antes_limpo.isdigit()
                and len(depois) == 3
                and len(antes_limpo) <= 3
            ):
                try:
                    raw = float(s)
                    if abs(raw) >= 10:
                        s = antes + depois
                except Exception:
                    pass

    try:
        return float(s)
    except Exception:
        return np.nan


def to_num(series):
    return pd.Series([smart_num_value(x) for x in series], index=series.index, dtype="float64")


def achar_coluna(df, candidatos, obrigatoria=False, nome=""):
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    for cand in candidatos:
        key = cand.lower().strip()
        if key in cols_lower:
            return cols_lower[key]

    for cand in candidatos:
        key = cand.lower().strip()
        for c in df.columns:
            cl = str(c).lower().strip()

            # Evita confundir probabilidade com limiar ajustado.
            if key in ("compra", "venda", "buy", "sell") and "limiar" in cl:
                continue

            if key in cl:
                return c

    if obrigatoria:
        raise ValueError(f"Coluna obrigatória não encontrada: {nome or candidatos}")

    return None


def carregar_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=";", engine="python")

    return normalizar_colunas(df)


def extrair_data(nome):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(nome))
    return m.group(1) if m else None


def preparar_tempo(df, day):
    time_col = achar_coluna(
        df,
        ["timestamp", "data_hora", "datetime", "datahora", "hora", "time", "dt"],
        obrigatoria=True,
        nome="tempo"
    )

    df = df.copy()
    raw = df[time_col].astype(str).str.strip()

    time_only = raw.str.match(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$").mean() > 0.60

    if day and time_only:
        parsed = pd.to_datetime(day + " " + raw, errors="coerce")
    else:
        try:
            parsed = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=False)
        except TypeError:
            parsed = pd.to_datetime(raw, errors="coerce", dayfirst=False)

        # Se quase tudo falhar, tenta forçar como hora do dia informado.
        if parsed.isna().mean() > 0.80 and day:
            parsed = pd.to_datetime(day + " " + raw, errors="coerce")

    df["_ts"] = parsed
    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

    return df, time_col


def parse_hhmm(s):
    h, m = str(s).split(":")[:2]
    return time(int(h), int(m))


def max_drawdown(series):
    if len(series) == 0:
        return 0.0

    res = pd.Series(series).astype(float).fillna(0)
    equity = pd.concat([pd.Series([0.0]), res.cumsum()], ignore_index=True)
    pico = equity.cummax()
    dd = equity - pico
    return float(dd.min())


def profit_factor(series):
    res = pd.Series(series).astype(float).fillna(0)
    ganhos = float(res[res > 0].sum())
    perdas = float(res[res < 0].sum())

    if perdas < 0:
        return ganhos / abs(perdas)

    if ganhos > 0:
        return 999.0

    return 0.0


# =============================================================================
# PREPARAÇÃO DOS DADOS
# =============================================================================

def preparar_120s(path_120s, day):
    df = carregar_csv(path_120s)
    df, time_col = preparar_tempo(df, day)

    close_col = achar_coluna(
        df,
        ["close", "fechamento", "preco", "preço", "ultimo", "último", "last", "valor"],
        obrigatoria=False
    )

    high_col = achar_coluna(
        df,
        ["high", "maxima", "máxima", "max"],
        obrigatoria=False
    )

    low_col = achar_coluna(
        df,
        ["low", "minima", "mínima", "min"],
        obrigatoria=False
    )

    numeric_cols = []
    for c in df.columns:
        if c == "_ts":
            continue

        converted = to_num(df[c])
        if converted.notna().sum() > max(5, len(df) * 0.50):
            numeric_cols.append(c)

    if close_col is None:
        if numeric_cols:
            close_col = numeric_cols[-1]
        else:
            raise ValueError(f"Não achei coluna de preço em {path_120s}")

    df["_close"] = to_num(df[close_col])

    if high_col is not None:
        df["_high"] = to_num(df[high_col])
    else:
        df["_high"] = df["_close"]

    if low_col is not None:
        df["_low"] = to_num(df[low_col])
    else:
        df["_low"] = df["_close"]

    df = df.dropna(subset=["_close"]).copy()
    df = df.sort_values("_ts").reset_index(drop=True)
    df["_bar_index"] = np.arange(len(df))

    for curta, longa in EMA_PAIRS:
        df[f"_ema{curta}"] = df["_close"].ewm(span=curta, adjust=False).mean()
        df[f"_ema{longa}"] = df["_close"].ewm(span=longa, adjust=False).mean()

    df["_range"] = (df["_high"] - df["_low"]).abs()
    df["_range_media50"] = df["_range"].rolling(50, min_periods=10).mean()
    df["_range_ratio"] = df["_range"] / df["_range_media50"].replace(0, np.nan)

    keep = [
        "_ts", "_bar_index", "_close", "_high", "_low",
        "_range", "_range_media50", "_range_ratio",
        "_ema20", "_ema100", "_ema9", "_ema21"
    ]

    keep = [c for c in keep if c in df.columns]

    return df[keep].copy(), {
        "time_col_120s": time_col,
        "close_col_120s": close_col,
        "high_col_120s": high_col,
        "low_col_120s": low_col,
        "rows_120s": len(df),
    }


def preparar_ia(path_ia, mercado_120s, day):
    df = carregar_csv(path_ia)
    df, time_col = preparar_tempo(df, day)

    pc_col = achar_coluna(
        df,
        ["pc", "prob_compra", "probabilidade_compra", "p_compra", "prob_buy", "p_buy", "proba_compra"],
        obrigatoria=True,
        nome="PC/probabilidade compra"
    )

    pv_col = achar_coluna(
        df,
        ["pv", "prob_venda", "probabilidade_venda", "p_venda", "prob_sell", "p_sell", "proba_venda"],
        obrigatoria=True,
        nome="PV/probabilidade venda"
    )

    preco_col = achar_coluna(
        df,
        ["preco", "preço", "price", "ultimo", "último", "close", "valor"],
        obrigatoria=False
    )

    sl_col = achar_coluna(df, ["sl", "stop", "stop_loss", "stoploss"], obrigatoria=False)
    tp_col = achar_coluna(df, ["tp", "take", "take_profit", "takeprofit"], obrigatoria=False)

    tendencia_col = achar_coluna(
        df,
        ["tendencia_pct", "tendência_pct", "trend_pct", "tendencia", "tendência"],
        obrigatoria=False
    )

    atr_col = achar_coluna(df, ["atr"], obrigatoria=False)
    atr_media_col = achar_coluna(df, ["atr_media50", "atr_media", "atr_medio", "atr_mean"], obrigatoria=False)
    vol_norm_col = achar_coluna(df, ["vol_norm", "volatilidade_norm", "volnorm"], obrigatoria=False)

    df["_pc"] = to_num(df[pc_col])
    df["_pv"] = to_num(df[pv_col])

    if preco_col:
        df["_preco_ia"] = to_num(df[preco_col])
    else:
        df["_preco_ia"] = np.nan

    if sl_col:
        df["_sl"] = to_num(df[sl_col]).fillna(SL_DEFAULT)
    else:
        df["_sl"] = SL_DEFAULT

    if tp_col:
        df["_tp"] = to_num(df[tp_col]).fillna(TP_DEFAULT)
    else:
        df["_tp"] = TP_DEFAULT

    if tendencia_col:
        df["_tendencia_pct"] = to_num(df[tendencia_col])
    else:
        df["_tendencia_pct"] = np.nan

    if atr_col:
        df["_atr"] = to_num(df[atr_col])
    else:
        df["_atr"] = np.nan

    if atr_media_col:
        df["_atr_media50"] = to_num(df[atr_media_col])
    else:
        df["_atr_media50"] = np.nan

    if vol_norm_col:
        df["_vol_norm"] = to_num(df[vol_norm_col])
    else:
        df["_vol_norm"] = np.nan

    mercado_ref_cols = [
        "_ts", "_bar_index", "_close", "_high", "_low",
        "_range", "_range_media50", "_range_ratio",
        "_ema20", "_ema100", "_ema9", "_ema21"
    ]
    mercado_ref_cols = [c for c in mercado_ref_cols if c in mercado_120s.columns]

    mercado_ref = mercado_120s[mercado_ref_cols].copy()

    df = pd.merge_asof(
        df.sort_values("_ts"),
        mercado_ref.sort_values("_ts"),
        on="_ts",
        direction="backward"
    )

    df["_preco"] = df["_preco_ia"].fillna(df["_close"])
    df = df.dropna(subset=["_pc", "_pv", "_preco", "_bar_index"]).copy()
    df["_bar_index"] = df["_bar_index"].astype(int)

    return df, {
        "time_col_ia": time_col,
        "pc_col": pc_col,
        "pv_col": pv_col,
        "preco_col": preco_col,
        "sl_col": sl_col,
        "tp_col": tp_col,
        "tendencia_col": tendencia_col,
        "atr_col": atr_col,
        "atr_media_col": atr_media_col,
        "vol_norm_col": vol_norm_col,
        "rows_ia": len(df),
    }


def achar_ultimos_10_dias():
    arquivos_ia = sorted(LOG_DIR.glob("log_ia_exec_*.csv"))
    dias = []

    for f in arquivos_ia:
        day = extrair_data(f.name)
        if not day:
            continue

        f120 = LOG_DIR / f"log_120s_{day}.csv"
        if f120.exists():
            dias.append(day)

    return sorted(set(dias))[-10:]


# =============================================================================
# REGRAS DE FILTRO
# =============================================================================

def direcao_sinal(row, pc_thr, pv_thr):
    pc = float(row["_pc"])
    pv = float(row["_pv"])

    if pc >= pc_thr and pc >= pv:
        return "BUY"

    if pv >= pv_thr and pv > pc:
        return "SELL"

    return None


def calcula_gap_ok(row, lado, gap_min):
    if gap_min is None or gap_min <= 0:
        return True

    pc = float(row["_pc"])
    pv = float(row["_pv"])

    if lado == "BUY":
        return (pc - pv) >= gap_min

    if lado == "SELL":
        return (pv - pc) >= gap_min

    return False


def tendencia_bloqueia(row, lado, cfg):
    modo = cfg.get("trend_mode")
    if not modo:
        return False

    curta = cfg.get("ema_curta", 20)
    longa = cfg.get("ema_longa", 100)
    zona = float(cfg.get("trend_zone", 0.0) or 0.0)

    c_curta = f"_ema{curta}"
    c_longa = f"_ema{longa}"

    if c_curta not in row.index or c_longa not in row.index:
        return False

    ema_c = row.get(c_curta, np.nan)
    ema_l = row.get(c_longa, np.nan)

    if pd.isna(ema_c) or pd.isna(ema_l) or ema_l == 0:
        return False

    diff = (float(ema_c) - float(ema_l)) / float(ema_l)

    tendencia_alta = diff > zona
    tendencia_baixa = diff < -zona

    if modo == "both":
        if lado == "BUY" and tendencia_baixa:
            return True
        if lado == "SELL" and tendencia_alta:
            return True

    elif modo == "sell_only":
        if lado == "SELL" and tendencia_alta:
            return True

    elif modo == "buy_only":
        if lado == "BUY" and tendencia_baixa:
            return True

    return False


def vol_ratio(row):
    atr = row.get("_atr", np.nan)
    atr_media = row.get("_atr_media50", np.nan)

    if pd.notna(atr) and pd.notna(atr_media) and float(atr_media) > 0:
        return float(atr) / float(atr_media)

    range_ratio = row.get("_range_ratio", np.nan)
    if pd.notna(range_ratio):
        return float(range_ratio)

    vol_norm = row.get("_vol_norm", np.nan)
    if pd.notna(vol_norm):
        return float(vol_norm)

    return np.nan


def volatilidade_alta(row, factor):
    r = vol_ratio(row)
    if pd.isna(r):
        return False

    return float(r) > float(factor)


def calcular_sl_tp(entry, sl_raw, tp_raw, lado):
    sl_raw = float(sl_raw) if pd.notna(sl_raw) else SL_DEFAULT
    tp_raw = float(tp_raw) if pd.notna(tp_raw) else TP_DEFAULT

    # Se vier como preço absoluto, usa como preço.
    if sl_raw > 10000:
        sl_price = sl_raw
    else:
        sl_price = entry - sl_raw if lado == "BUY" else entry + sl_raw

    if tp_raw > 10000:
        tp_price = tp_raw
    else:
        tp_price = entry + tp_raw if lado == "BUY" else entry - tp_raw

    return sl_price, tp_price


def horario_bloqueia(ts, session_start, session_end, cfg):
    skip_open = cfg.get("skip_open_min")
    skip_final = cfg.get("skip_final_min")
    time_start = cfg.get("time_start")
    time_end = cfg.get("time_end")

    if skip_open is not None:
        if ts < session_start + timedelta(minutes=int(skip_open)):
            return True

    if skip_final is not None:
        if ts > session_end - timedelta(minutes=int(skip_final)):
            return True

    if time_start is not None:
        if ts.time() < parse_hhmm(time_start):
            return True

    if time_end is not None:
        if ts.time() > parse_hhmm(time_end):
            return True

    return False


# =============================================================================
# SIMULAÇÃO
# =============================================================================

def simular_operacoes(ia, mercado_120s, cfg):
    ops = []

    bloqueios = {
        "sem_sinal": 0,
        "em_operacao": 0,
        "gap": 0,
        "horario": 0,
        "tendencia": 0,
        "volatilidade": 0,
        "vol_cooldown": 0,
        "loss_cooldown": 0,
        "max_daily_loss": 0,
        "max_consecutive_losses": 0,
    }

    pc_thr = float(cfg.get("pc", PC_BASE))
    pv_thr = float(cfg.get("pv", PV_BASE))

    ocupado_ate = None
    vol_cd_until_bar = -1
    loss_cd_until_bar = -1

    daily_equity = 0.0
    consecutive_losses = 0
    stop_day = False

    session_start = ia["_ts"].min()
    session_end = ia["_ts"].max()

    mercado_120s = mercado_120s.sort_values("_ts").reset_index(drop=True)
    ia = ia.sort_values("_ts").reset_index(drop=True)

    for _, row in ia.iterrows():
        ts = row["_ts"]
        bar_idx = int(row["_bar_index"])

        # Atualiza cooldown de volatilidade por candle/sinal, mesmo sem operação.
        if cfg.get("vol_mode") == "cooldown":
            factor = cfg.get("vol_factor", 2.0)
            cd_bars = int(cfg.get("vol_cooldown_bars", 3))

            if volatilidade_alta(row, factor):
                vol_cd_until_bar = max(vol_cd_until_bar, bar_idx + cd_bars)

        if stop_day:
            if cfg.get("max_daily_loss") is not None:
                bloqueios["max_daily_loss"] += 1
            else:
                bloqueios["max_consecutive_losses"] += 1
            continue

        if ocupado_ate is not None and ts <= ocupado_ate:
            bloqueios["em_operacao"] += 1
            continue

        if bar_idx <= loss_cd_until_bar:
            bloqueios["loss_cooldown"] += 1
            continue

        if cfg.get("vol_mode") == "cooldown" and bar_idx <= vol_cd_until_bar:
            bloqueios["vol_cooldown"] += 1
            continue

        if horario_bloqueia(ts, session_start, session_end, cfg):
            bloqueios["horario"] += 1
            continue

        lado = direcao_sinal(row, pc_thr, pv_thr)

        if lado is None:
            bloqueios["sem_sinal"] += 1
            continue

        gap_min = cfg.get("gap_min")
        if not calcula_gap_ok(row, lado, gap_min):
            bloqueios["gap"] += 1
            continue

        if tendencia_bloqueia(row, lado, cfg):
            bloqueios["tendencia"] += 1
            continue

        if cfg.get("vol_mode") == "block":
            factor = cfg.get("vol_factor", 2.0)
            if volatilidade_alta(row, factor):
                bloqueios["volatilidade"] += 1
                continue

        entry = float(row["_preco"])
        sl_price, tp_price = calcular_sl_tp(entry, row["_sl"], row["_tp"], lado)

        futuro = mercado_120s[mercado_120s["_ts"] > ts].copy()

        if futuro.empty:
            continue

        exit_ts = futuro.iloc[-1]["_ts"]
        exit_bar_idx = int(futuro.iloc[-1]["_bar_index"])
        exit_price = float(futuro.iloc[-1]["_close"])
        motivo = "EOD"

        for _, bar in futuro.iterrows():
            high = float(bar["_high"])
            low = float(bar["_low"])

            if lado == "BUY":
                bateu_sl = low <= sl_price
                bateu_tp = high >= tp_price

                if bateu_sl and bateu_tp:
                    exit_ts = bar["_ts"]
                    exit_bar_idx = int(bar["_bar_index"])
                    exit_price = sl_price
                    motivo = "SL_AMBIGUO"
                    break
                elif bateu_sl:
                    exit_ts = bar["_ts"]
                    exit_bar_idx = int(bar["_bar_index"])
                    exit_price = sl_price
                    motivo = "SL"
                    break
                elif bateu_tp:
                    exit_ts = bar["_ts"]
                    exit_bar_idx = int(bar["_bar_index"])
                    exit_price = tp_price
                    motivo = "TP"
                    break

            else:
                bateu_sl = high >= sl_price
                bateu_tp = low <= tp_price

                if bateu_sl and bateu_tp:
                    exit_ts = bar["_ts"]
                    exit_bar_idx = int(bar["_bar_index"])
                    exit_price = sl_price
                    motivo = "SL_AMBIGUO"
                    break
                elif bateu_sl:
                    exit_ts = bar["_ts"]
                    exit_bar_idx = int(bar["_bar_index"])
                    exit_price = sl_price
                    motivo = "SL"
                    break
                elif bateu_tp:
                    exit_ts = bar["_ts"]
                    exit_bar_idx = int(bar["_bar_index"])
                    exit_price = tp_price
                    motivo = "TP"
                    break

        if lado == "BUY":
            resultado = exit_price - entry
        else:
            resultado = entry - exit_price

        resultado = round(float(resultado), 2)

        daily_equity += resultado

        if resultado < 0:
            consecutive_losses += 1
            loss_cd_bars = cfg.get("loss_cooldown_bars")
            if loss_cd_bars is not None:
                loss_cd_until_bar = exit_bar_idx + int(loss_cd_bars)
        else:
            consecutive_losses = 0

        max_daily_loss = cfg.get("max_daily_loss")
        if max_daily_loss is not None and daily_equity <= -abs(float(max_daily_loss)):
            stop_day = True

        max_seq = cfg.get("max_consecutive_losses")
        if max_seq is not None and consecutive_losses >= int(max_seq):
            stop_day = True

        ocupado_ate = exit_ts

        ops.append({
            "entrada_ts": ts,
            "saida_ts": exit_ts,
            "entrada_bar": bar_idx,
            "saida_bar": exit_bar_idx,
            "lado": lado,
            "preco_entrada": entry,
            "preco_saida": exit_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "motivo_saida": motivo,
            "resultado_pts": resultado,
            "equity_dia_apos_trade": round(daily_equity, 2),
            "pc": round(float(row["_pc"]), 6),
            "pv": round(float(row["_pv"]), 6),
            "gap": round(abs(float(row["_pc"]) - float(row["_pv"])), 6),
            "vol_ratio": round(vol_ratio(row), 6) if pd.notna(vol_ratio(row)) else np.nan,
            "ema20": round(float(row["_ema20"]), 5) if "_ema20" in row.index and pd.notna(row["_ema20"]) else np.nan,
            "ema100": round(float(row["_ema100"]), 5) if "_ema100" in row.index and pd.notna(row["_ema100"]) else np.nan,
            "ema9": round(float(row["_ema9"]), 5) if "_ema9" in row.index and pd.notna(row["_ema9"]) else np.nan,
            "ema21": round(float(row["_ema21"]), 5) if "_ema21" in row.index and pd.notna(row["_ema21"]) else np.nan,
        })

    return pd.DataFrame(ops), bloqueios


# =============================================================================
# CENÁRIOS
# =============================================================================

def add_scenario(cenarios, name, **kwargs):
    cfg = {
        "name": name,
        "pc": PC_BASE,
        "pv": PV_BASE,
        "trend_mode": None,
        "ema_curta": None,
        "ema_longa": None,
        "trend_zone": 0.0,
        "vol_mode": None,
        "vol_factor": None,
        "vol_cooldown_bars": None,
        "gap_min": None,
        "time_start": None,
        "time_end": None,
        "skip_open_min": None,
        "skip_final_min": None,
        "loss_cooldown_bars": None,
        "max_daily_loss": None,
        "max_consecutive_losses": None,
    }
    cfg.update(kwargs)
    cenarios.append(cfg)


def montar_cenarios():
    cenarios = []

    # BASE
    add_scenario(cenarios, "BASE_PC058_PV058")

    # 1) PV mais seletivo
    for pv in [0.60, 0.62, 0.65, 0.70]:
        add_scenario(cenarios, f"PV{int(pv*100):03d}_SELL_MAIS_SELETIVO", pv=pv)

    # 2) GAP mínimo
    for gap in [0.03, 0.05, 0.08, 0.10]:
        add_scenario(cenarios, f"GAP{int(gap*100):03d}", gap_min=gap)

    # 3) Tendência nos dois lados
    for curta, longa in [(20, 100), (9, 21)]:
        add_scenario(
            cenarios,
            f"TEND_BOTH_EMA{curta}_{longa}",
            trend_mode="both",
            ema_curta=curta,
            ema_longa=longa,
            trend_zone=0.0
        )

    # 4) Tendência apenas no SELL
    for curta, longa in [(20, 100), (9, 21)]:
        add_scenario(
            cenarios,
            f"TEND_SELL_ONLY_EMA{curta}_{longa}",
            trend_mode="sell_only",
            ema_curta=curta,
            ema_longa=longa,
            trend_zone=0.0
        )

        for pv in [0.60, 0.62]:
            add_scenario(
                cenarios,
                f"TEND_SELL_ONLY_EMA{curta}_{longa}_PV{int(pv*100):03d}",
                pv=pv,
                trend_mode="sell_only",
                ema_curta=curta,
                ema_longa=longa,
                trend_zone=0.0
            )

    # 5) Zona neutra de tendência apenas no SELL
    for zone_pct in [0.0005, 0.0010, 0.0015]:
        label = int(zone_pct * 10000)
        add_scenario(
            cenarios,
            f"TEND_SELL_EMA20_100_ZONA{label:03d}",
            trend_mode="sell_only",
            ema_curta=20,
            ema_longa=100,
            trend_zone=zone_pct
        )
        add_scenario(
            cenarios,
            f"TEND_BOTH_EMA20_100_ZONA{label:03d}",
            trend_mode="both",
            ema_curta=20,
            ema_longa=100,
            trend_zone=zone_pct
        )

    # 6) Volatilidade com limite maior
    for factor in [1.75, 2.00, 2.50]:
        add_scenario(
            cenarios,
            f"VOL_BLOCK_{str(factor).replace('.', '_')}X",
            vol_mode="block",
            vol_factor=factor
        )

    # 7) Volatilidade com cooldown
    for factor in [2.00, 2.50]:
        for cd in [3, 5]:
            add_scenario(
                cenarios,
                f"VOL_CD_{str(factor).replace('.', '_')}X_{cd}BARS",
                vol_mode="cooldown",
                vol_factor=factor,
                vol_cooldown_bars=cd
            )

    # 8) Filtro por horário
    add_scenario(cenarios, "SEM_ABERTURA_15MIN", skip_open_min=15)
    add_scenario(cenarios, "SEM_ABERTURA_30MIN", skip_open_min=30)
    add_scenario(cenarios, "SEM_FINAL_30MIN", skip_final_min=30)
    add_scenario(cenarios, "JANELA_0930_1700", time_start="09:30", time_end="17:00")
    add_scenario(cenarios, "JANELA_1000_1700", time_start="10:00", time_end="17:00")
    add_scenario(cenarios, "JANELA_1000_1600", time_start="10:00", time_end="16:00")

    # 9) Cooldown após stop
    for cd in [1, 2, 3, 5]:
        add_scenario(cenarios, f"LOSS_CD_{cd}BARS", loss_cooldown_bars=cd)

    # 10) Limite de perda diária
    for loss in [300, 450, 600]:
        add_scenario(cenarios, f"MAX_DAILY_LOSS_{loss}", max_daily_loss=loss)

    add_scenario(cenarios, "MAX_3_STOPS_CONSECUTIVOS", max_consecutive_losses=3)

    # 11) Combinações enxutas
    add_scenario(cenarios, "COMBO_PV060_LOSSCD2", pv=0.60, loss_cooldown_bars=2)
    add_scenario(cenarios, "COMBO_PV062_GAP005", pv=0.62, gap_min=0.05)
    add_scenario(cenarios, "COMBO_PV060_GAP005", pv=0.60, gap_min=0.05)
    add_scenario(cenarios, "COMBO_PV060_SEM_ABERTURA15", pv=0.60, skip_open_min=15)
    add_scenario(cenarios, "COMBO_PV060_VOLCD2X3", pv=0.60, vol_mode="cooldown", vol_factor=2.0, vol_cooldown_bars=3)
    add_scenario(cenarios, "COMBO_PV062_TEND_SELL_EMA20_100", pv=0.62, trend_mode="sell_only", ema_curta=20, ema_longa=100)
    add_scenario(cenarios, "COMBO_PV060_GAP005_LOSSCD2", pv=0.60, gap_min=0.05, loss_cooldown_bars=2)
    add_scenario(cenarios, "COMBO_PV062_GAP003_TENDSELL", pv=0.62, gap_min=0.03, trend_mode="sell_only", ema_curta=20, ema_longa=100)
    add_scenario(cenarios, "COMBO_GAP005_LOSSCD2", gap_min=0.05, loss_cooldown_bars=2)
    add_scenario(cenarios, "COMBO_SEM_ABERTURA15_LOSSCD2", skip_open_min=15, loss_cooldown_bars=2)
    add_scenario(cenarios, "COMBO_VOLCD2X3_LOSSCD2", vol_mode="cooldown", vol_factor=2.0, vol_cooldown_bars=3, loss_cooldown_bars=2)
    add_scenario(cenarios, "COMBO_MAXLOSS450_LOSSCD2", max_daily_loss=450, loss_cooldown_bars=2)

    return cenarios


# =============================================================================
# RESUMOS
# =============================================================================

def resumir_ops(ops, cenario, dia="TOTAL"):
    if ops is None or ops.empty:
        return {
            "cenario": cenario,
            "dia": dia,
            "operacoes": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "resultado_pts": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pts": 0.0,
            "buy_ops": 0,
            "buy_resultado": 0.0,
            "sell_ops": 0,
            "sell_resultado": 0.0,
            "avg_trade": 0.0,
        }

    res = ops["resultado_pts"].astype(float)
    wins = int((res > 0).sum())
    losses = int((res < 0).sum())

    buy = ops[ops["lado"] == "BUY"]
    sell = ops[ops["lado"] == "SELL"]

    return {
        "cenario": cenario,
        "dia": dia,
        "operacoes": int(len(ops)),
        "wins": wins,
        "losses": losses,
        "winrate": round(wins / len(ops), 4) if len(ops) else 0.0,
        "resultado_pts": round(float(res.sum()), 2),
        "profit_factor": round(float(profit_factor(res)), 4),
        "max_drawdown_pts": round(float(max_drawdown(res)), 2),
        "buy_ops": int(len(buy)),
        "buy_resultado": round(float(buy["resultado_pts"].sum()), 2) if not buy.empty else 0.0,
        "sell_ops": int(len(sell)),
        "sell_resultado": round(float(sell["resultado_pts"].sum()), 2) if not sell.empty else 0.0,
        "avg_trade": round(float(res.mean()), 4) if len(res) else 0.0,
    }


def calcular_ranking(df_total, df_dia):
    base = df_total[df_total["cenario"] == "BASE_PC058_PV058"].iloc[0].to_dict()

    df = df_total.copy()

    df["delta_resultado_vs_base"] = df["resultado_pts"] - float(base["resultado_pts"])
    df["delta_pf_vs_base"] = df["profit_factor"] - float(base["profit_factor"])
    df["delta_drawdown_vs_base"] = df["max_drawdown_pts"] - float(base["max_drawdown_pts"])
    df["delta_ops_vs_base"] = df["operacoes"] - int(base["operacoes"])

    positivos = (
        df_dia
        .assign(dia_positivo=lambda x: x["resultado_pts"] > 0)
        .groupby("cenario")["dia_positivo"]
        .sum()
        .reset_index()
        .rename(columns={"dia_positivo": "dias_positivos"})
    )

    dias_total = (
        df_dia
        .groupby("cenario")["dia"]
        .nunique()
        .reset_index()
        .rename(columns={"dia": "dias_testados"})
    )

    df = df.merge(positivos, on="cenario", how="left")
    df = df.merge(dias_total, on="cenario", how="left")

    df["dias_positivos"] = df["dias_positivos"].fillna(0).astype(int)
    df["dias_testados"] = df["dias_testados"].fillna(0).astype(int)

    # Score balanceado:
    # - valoriza lucro
    # - valoriza PF
    # - penaliza drawdown muito negativo
    # - valoriza dias positivos
    # - não premia demais cenários com poucas operações
    df["score"] = (
        df["resultado_pts"].astype(float)
        + df["profit_factor"].replace(999, 5).astype(float) * 250
        + df["max_drawdown_pts"].astype(float) * 0.25
        + df["dias_positivos"].astype(float) * 150
        + np.minimum(df["operacoes"].astype(float), float(base["operacoes"])) * 0.01
    )

    df["aprovacao_preliminar"] = np.where(
        (df["resultado_pts"] > float(base["resultado_pts"])) &
        (df["profit_factor"] >= float(base["profit_factor"])) &
        (df["max_drawdown_pts"] >= float(base["max_drawdown_pts"]) * 1.05),
        "FORTE",
        np.where(
            (df["resultado_pts"] > float(base["resultado_pts"])) &
            (df["profit_factor"] >= float(base["profit_factor"])),
            "PROMISSOR",
            "NAO_SUPEROU_BASE"
        )
    )

    df = df.sort_values(
        ["score", "resultado_pts", "profit_factor", "max_drawdown_pts"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    df.insert(0, "rank", np.arange(1, len(df) + 1))

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dias = achar_ultimos_10_dias()
    if not dias:
        raise SystemExit(
            f"[ERRO] Nenhum par log_ia_exec_*.csv / log_120s_*.csv encontrado em {LOG_DIR}.\n"
            "Esses logs são gerados pelo motor durante o pregão e não são versionados."
        )
    cenarios = montar_cenarios()

    print("=" * 100)
    print("BATERIA DE TESTES PC058/PV058 - 10 PREGÕES")
    print("=" * 100)
    print(f"Projeto: {PROJECT_DIR}")
    print(f"Logs:    {LOG_DIR}")
    print(f"Saída:   {OUT_DIR}")
    print()
    print(f"PC base fixo: {PC_BASE}")
    print(f"PV base fixo: {PV_BASE}")
    print(f"Total de cenários: {len(cenarios)}")
    print()
    print("Dias usados:")
    for d in dias:
        print(" -", d)

    if len(dias) < 10:
        print()
        print(f"[ALERTA] Foram encontrados apenas {len(dias)} dias com log_ia_exec e log_120s.")

    todas_ops = []
    resumos_dia = []
    bloqueios_all = []
    diagnostico = []

    for day in dias:
        path_ia = LOG_DIR / f"log_ia_exec_{day}.csv"
        path_120s = LOG_DIR / f"log_120s_{day}.csv"

        print()
        print("=" * 100)
        print(f"DIA {day}")
        print("=" * 100)

        try:
            mercado, diag_120s = preparar_120s(path_120s, day)
            ia, diag_ia = preparar_ia(path_ia, mercado, day)

            diag = {"dia": day}
            diag.update(diag_120s)
            diag.update(diag_ia)
            diag["hora_inicio_ia"] = ia["_ts"].min()
            diag["hora_fim_ia"] = ia["_ts"].max()
            diag["hora_inicio_120s"] = mercado["_ts"].min()
            diag["hora_fim_120s"] = mercado["_ts"].max()
            diagnostico.append(diag)

            print(f"[OK] IA rows: {len(ia)} | 120s rows: {len(mercado)}")
            print(f"[OK] IA horário: {ia['_ts'].min()} -> {ia['_ts'].max()}")

            for i, cfg in enumerate(cenarios, start=1):
                nome = cfg["name"]

                ops, bloqueios = simular_operacoes(ia, mercado, cfg)

                if not ops.empty:
                    ops.insert(0, "dia", day)
                    ops.insert(1, "cenario", nome)
                    todas_ops.append(ops)

                resumo = resumir_ops(ops, nome, dia=day)
                resumos_dia.append(resumo)

                b = {"dia": day, "cenario": nome}
                b.update(bloqueios)
                bloqueios_all.append(b)

                if i % 10 == 0 or i == len(cenarios):
                    print(f"[OK] {i}/{len(cenarios)} cenários processados")

        except Exception as e:
            diagnostico.append({"dia": day, "erro": str(e)})
            print(f"[ERRO] {day}: {e}")

    if todas_ops:
        df_ops = pd.concat(todas_ops, ignore_index=True)
    else:
        df_ops = pd.DataFrame()

    df_dia = pd.DataFrame(resumos_dia)
    df_bloq = pd.DataFrame(bloqueios_all)
    df_diag = pd.DataFrame(diagnostico)
    df_cenarios = pd.DataFrame(cenarios)

    resumo_total = []

    for cfg in cenarios:
        nome = cfg["name"]
        ops_c = df_ops[df_ops["cenario"] == nome].copy() if not df_ops.empty else pd.DataFrame()
        resumo_total.append(resumir_ops(ops_c, nome, dia="TOTAL"))

    df_total = pd.DataFrame(resumo_total)
    df_rank = calcular_ranking(df_total, df_dia)

    # Salvar arquivos
    df_ops.to_csv(OUT_DIR / "operacoes_todos_cenarios.csv", index=False, encoding="utf-8-sig")
    df_dia.to_csv(OUT_DIR / "resumo_por_dia_todos_cenarios.csv", index=False, encoding="utf-8-sig")
    df_total.to_csv(OUT_DIR / "resumo_total_todos_cenarios.csv", index=False, encoding="utf-8-sig")
    df_rank.to_csv(OUT_DIR / "ranking_todos_cenarios.csv", index=False, encoding="utf-8-sig")
    df_bloq.to_csv(OUT_DIR / "bloqueios_todos_cenarios.csv", index=False, encoding="utf-8-sig")
    df_diag.to_csv(OUT_DIR / "diagnostico_dias_usados.csv", index=False, encoding="utf-8-sig")
    df_cenarios.to_csv(OUT_DIR / "cenarios_testados.csv", index=False, encoding="utf-8-sig")

    # Resumo em TXT
    txt_path = OUT_DIR / "TOP_15_RANKING.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TOP 15 - BATERIA DE TESTES PC058/PV058\n")
        f.write("=" * 100 + "\n\n")
        f.write(df_rank.head(15)[[
            "rank", "cenario", "operacoes", "winrate", "resultado_pts",
            "profit_factor", "max_drawdown_pts", "dias_positivos",
            "buy_resultado", "sell_resultado", "delta_resultado_vs_base",
            "aprovacao_preliminar", "score"
        ]].to_string(index=False))
        f.write("\n")

    print()
    print("=" * 100)
    print("TOP 15 - RANKING")
    print("=" * 100)
    print(df_rank.head(15)[[
        "rank", "cenario", "operacoes", "winrate", "resultado_pts",
        "profit_factor", "max_drawdown_pts", "dias_positivos",
        "buy_resultado", "sell_resultado",
        "delta_resultado_vs_base", "aprovacao_preliminar", "score"
    ]].to_string(index=False))

    print()
    print("=" * 100)
    print("BASE")
    print("=" * 100)
    print(df_total[df_total["cenario"] == "BASE_PC058_PV058"].to_string(index=False))

    print()
    print("=" * 100)
    print("ARQUIVOS GERADOS")
    print("=" * 100)
    for f in sorted(OUT_DIR.glob("*")):
        print(f"[OK] {f}")

    # Compacta automaticamente
    zip_base = OUT_DIR
    zip_path = shutil.make_archive(str(zip_base), "zip", OUT_DIR)

    print()
    print("=" * 100)
    print("ZIP GERADO")
    print("=" * 100)
    print(zip_path)

    print()
    print("[CONCLUÍDO] Bateria de testes finalizada.")


if __name__ == "__main__":
    main()
