# -*- coding: utf-8 -*-
"""
eod_logfix.py — pós-processamento de logs no fim do dia

Objetivos:
A) Padronizar horário (corrigir casos de -3h / +3h no campo 'hora', sem distorcer o horário correto)
B) Padronizar SL/TP em pontos no log_operacoes_v3 (se detectar valores em "preço")
C) Criar/atualizar coluna 'acerto' (lucro_real > 0) no log_operacoes_v3
D) Deduplicar validacao_incremental por PID (mantém última ocorrência do dia)

Uso:
  python eod_logfix.py --pasta_logs arquivo_ABC --data YYYY-MM-DD
Se --data não for passado, usa a data de hoje (timezone local).
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime as dt

import pandas as pd


def _hoje_iso():
    return dt.now().strftime("%Y-%m-%d")


def _load_csv(p: Path):
    if not p.exists():
        print(f"[INFO] Arquivo não encontrado, pulando: {p}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[ERRO] Falha ao ler {p}: {e}")
        return None


def _save_csv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"[OK] Salvo: {p} ({len(df)} linhas)")


def _time_to_seconds(h: int, m: int, s: int) -> int:
    return int(h) * 3600 + int(m) * 60 + int(s)


def _seconds_to_hms(sec: int) -> str:
    sec = int(sec) % 86400
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_hora_to_seconds(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    # Caso comum: HH:MM:SS / HH:MM
    if ":" in s and ("-" not in s and "T" not in s and " " not in s):
        parts = s.split(":")
        try:
            hh = int(parts[0])
            mm = int(parts[1]) if len(parts) > 1 else 0
            ss = int(parts[2]) if len(parts) > 2 else 0
            if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
                return _time_to_seconds(hh, mm, ss)
        except Exception:
            pass

    # Caso: data+hora (ISO, etc.)
    try:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            return None
        return _time_to_seconds(int(t.hour), int(t.minute), int(t.second))
    except Exception:
        return None


def _circular_diff(a: int, b: int) -> int:
    """Diferença (a-b) em segundos, no menor caminho circular (±12h)."""
    return int(((a - b + 43200) % 86400) - 43200)


def _reference_from_log_ia(pasta_logs: Path, data_ymd: str):
    """Retorna (ref_median, lo, hi) em segundos do dia, com base no log_ia_exec do dia."""
    p = pasta_logs / f"log_ia_exec_{data_ymd}.csv"
    df = _load_csv(p)
    if df is None or "hora" not in df.columns:
        return None, None, None

    secs = df["hora"].map(_parse_hora_to_seconds).dropna().astype(int)
    if len(secs) == 0:
        return None, None, None

    ref = int(secs.median())
    lo = int(secs.quantile(0.01))  # ~início do dia
    hi = int(secs.quantile(0.99))  # ~fim do dia
    return ref, lo, hi


def _normalize_hora_column(
    df: pd.DataFrame,
    col: str = "hora",
    ref_sec: int | None = None,
    lo_sec: int | None = None,
    hi_sec: int | None = None,
) -> pd.DataFrame:
    if col not in df.columns:
        return df

    secs = df[col].map(_parse_hora_to_seconds)
    secs_valid = secs.dropna().astype(int)
    if len(secs_valid) == 0:
        return df

    # fallback se não houver referência externa
    if ref_sec is None:
        ref_sec = int(secs_valid.median())
    if lo_sec is None:
        lo_sec = min(int(secs_valid.min()), 8 * 3600)   # pelo menos 08:00
    if hi_sec is None:
        hi_sec = max(int(secs_valid.max()), 19 * 3600)  # pelo menos 19:00

    def _in_range(sec: int) -> bool:
        return int(lo_sec) <= int(sec) <= int(hi_sec)

    def _best_sec(sec):
        if sec is None or (isinstance(sec, float) and pd.isna(sec)):
            return None
        sec = int(sec)

        cands = [sec, (sec + 10800) % 86400, (sec - 10800) % 86400]

        # Regra de ouro:
        # Só aplica ±3h se o horário ORIGINAL estiver FORA do intervalo típico do dia
        # e algum candidato cair DENTRO desse intervalo. Evita “empurrar” horários corretos.
        if not _in_range(sec):
            in_range = [c for c in cands if _in_range(c)]
            if len(in_range) == 1:
                return in_range[0]
            if len(in_range) > 1:
                # escolhe o mais próximo da referência
                diffs = [abs(_circular_diff(c, ref_sec)) for c in in_range]
                return in_range[int(pd.Series(diffs).idxmin())]

        # Caso geral: não mexe
        return sec

    secs_norm = secs.map(_best_sec)
    df[col] = secs_norm.map(lambda v: _seconds_to_hms(v) if v is not None and not pd.isna(v) else "")

    df["_hora_sec"] = secs_norm.fillna(-1).astype(int)
    df = df.sort_values(["_hora_sec"]).drop(columns=["_hora_sec"])
    return df



def _normalize_close_time_by_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta o campo 'hora' das linhas de FECHAMENTO usando 'duracao_operacao' quando disponível.
    Cenário típico: 'hora' do fechamento foi registrada no loop (ex.: +8s), mas a duração foi
    calculada a partir do histórico de deals (ex.: 2s). Nesse caso, para bater com o MT5,
    definimos: hora_fechamento = hora_abertura + duracao_operacao.

    Regras de segurança:
    - exige colunas: 'pid', 'resultado', 'hora', 'duracao_operacao'
    - exige que exista 1 linha de abertura e 1 linha de fechamento para o mesmo pid
    - só ajusta se a diferença for > 5 segundos
    """
    req = {"pid", "resultado", "hora", "duracao_operacao"}
    if not req.issubset(set(df.columns)):
        return df

    # normaliza duração numérica
    dur = pd.to_numeric(df["duracao_operacao"], errors="coerce")

    # segundos de hora
    sec = df["hora"].map(_parse_hora_to_seconds)

    # índices por pid
    try:
        pids = pd.to_numeric(df["pid"], errors="coerce")
    except Exception:
        pids = df["pid"]

    df = df.copy()
    tol = 5  # segundos

    for pid_val in pd.Series(pids).dropna().unique():
        mask = pids == pid_val
        sub = df.loc[mask]
        if len(sub) < 2:
            continue

        # Abertura: primeira ocorrência com resultado == 'aberta'
        open_idx = sub.index[sub["resultado"].astype(str).str.lower().eq("aberta")].tolist()
        close_idx = sub.index[sub["resultado"].astype(str).str.lower().eq("fechada")].tolist()
        if len(open_idx) == 0 or len(close_idx) == 0:
            continue

        oi = open_idx[0]
        # usa a última linha de fechamento (mais segura)
        ci = close_idx[-1]

        open_sec = sec.get(oi, None)
        close_sec = sec.get(ci, None)
        dsec = dur.get(ci, None)

        if open_sec is None or close_sec is None or dsec is None or pd.isna(open_sec) or pd.isna(close_sec) or pd.isna(dsec):
            continue

        # filtros de duração razoável
        try:
            dsec_i = int(round(float(dsec)))
        except Exception:
            continue
        if dsec_i <= 0 or dsec_i > 12 * 3600:
            continue

        target = int(open_sec) + dsec_i
        if target < 0:
            continue
        if target >= 86400:
            target = target % 86400

        # ajusta só se estiver claramente diferente
        if abs(_circular_diff(int(close_sec), int(target))) > tol:
            df.at[ci, "hora"] = _seconds_to_hms(target)

    return df

def _normalize_sl_tp_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza sl/tp em pontos:
    - Se detectar valores muito altos (ex.: > 5000), assume que é preço e substitui por referência em pontos.
    - Referência = mediana dos valores "baixos" (1..2000) encontrados no próprio CSV.
    """
    for col in ("sl", "tp"):
        if col not in df.columns:
            continue

        v = pd.to_numeric(df[col], errors="coerce")
        low = v[(v > 0) & (v <= 2000)]
        if len(low) == 0:
            continue

        ref = float(low.median())
        hi_mask = v > 5000
        if hi_mask.any():
            v.loc[hi_mask] = ref

        def _fmt(x):
            if pd.isna(x):
                return ""
            try:
                xf = float(x)
                if abs(xf - round(xf)) < 1e-9:
                    return str(int(round(xf)))
                return str(round(xf, 4))
            except Exception:
                return str(x)

        df[col] = v.map(_fmt)

    return df


def fix_log_operacoes(pasta_logs: Path, data_ymd: str):
    p = pasta_logs / f"log_operacoes_v3_{data_ymd}.csv"
    df = _load_csv(p)
    if df is None:
        return

    ref, lo, hi = _reference_from_log_ia(pasta_logs, data_ymd)
    df = _normalize_hora_column(df, "hora", ref_sec=ref, lo_sec=lo, hi_sec=hi)
    df = _normalize_close_time_by_duration(df)
    df = _normalize_sl_tp_points(df)

    if "lucro_real" in df.columns:
        df["lucro_real"] = pd.to_numeric(df["lucro_real"], errors="coerce")
        df["acerto"] = (df["lucro_real"] > 0).astype("Int64")
    else:
        print("[WARN] 'lucro_real' não existe em log_operacoes_v3; não vou gerar 'acerto'.")

    _save_csv(df, p)


def fix_validacao_incremental(pasta_logs: Path, data_ymd: str):
    p = pasta_logs / f"validacao_incremental_{data_ymd}.csv"
    df = _load_csv(p)
    if df is None:
        return

    ref, lo, hi = _reference_from_log_ia(pasta_logs, data_ymd)
    df = _normalize_hora_column(df, "hora", ref_sec=ref, lo_sec=lo, hi_sec=hi)
    df = _normalize_close_time_by_duration(df)

    if "pid" not in df.columns:
        print("[WARN] 'pid' não existe na validacao_incremental; não é possível deduplicar.")
        _save_csv(df, p)
        return

    df["pid"] = df["pid"].astype(str)
    df = df.drop_duplicates(subset=["pid"], keep="last")
    _save_csv(df, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta_logs", default="arquivo_ABC", help="Diretório onde estão os CSVs do dia")
    ap.add_argument("--data", default=None, help="YYYY-MM-DD do pregão (default = hoje)")
    args = ap.parse_args()

    data_ymd = args.data or _hoje_iso()
    pasta = Path(args.pasta_logs)

    print(f"[EOD] Pasta logs: {pasta} | Data: {data_ymd}")
    fix_log_operacoes(pasta, data_ymd)
    fix_validacao_incremental(pasta, data_ymd)
    print("[EOD] Finalizado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
