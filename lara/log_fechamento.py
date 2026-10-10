# -*- coding: utf-8 -*-
"""
log_fechamento — helpers mínimos para logging de fechamento no Lara 3.3.1

Objetivo:
- Garantir preenchimento de: motivo_saida, lucro_real, duracao_operacao, fechado_por_ia
- Sem alterar a estratégia/entradas: apenas pós-fechamento (SL/TP/IA/EOD)

Como usar (resumo):
1) Coloque este arquivo ao lado do seu motor_lara.py e logging_utils.py
2) No motor_lara.py, importe:
       from log_fechamento import finalize_close_log_from_deals
3) Dentro de close_position_market(...), close_position_by_ia(...)
   e no fechamento forçado EOD, chame:
       finalize_close_log_from_deals(env, position_id=pid, symbol=symbol,
                                     tipo=tipo, sl=sl, tp=tp,
                                     closed_by="IA-EXIT" ou "EOD" ou "MANUAL")
   * closed_by deve ser uma das strings: "IA-EXIT", "EOD" ou "MANUAL".
4) O helper buscará os deals no histórico e preencherá os campos críticos.
"""

import datetime as dt
from typing import Optional, Tuple

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# Espera-se que exista um módulo com o logger de operações v3
try:
    from logging_utils import log_operacoes_v3
except Exception as e:
    log_operacoes_v3 = None

def _ensure_history_selected(start: dt.datetime, end: dt.datetime) -> bool:
    if mt5 is None:
        return False
    try:
        return bool(mt5.history_select(start, end))
    except Exception:
        return False

def _classify_reason_from_deal(deal) -> Optional[str]:
    """Tenta mapear o motivo de fechamento a partir do 'deal.reason'."""
    if mt5 is None or deal is None:
        return None
    try:
        r = getattr(deal, "reason", None)
        # Alguns builds/nodes podem não ter todas as constantes, então usamos getattr c/ fallback
        if r == getattr(mt5, "DEAL_REASON_SL", -999):
            return "SL"
        if r == getattr(mt5, "DEAL_REASON_TP", -999):
            return "TP"
    except Exception:
        pass
    return None

def _fetch_in_out_deals(position_id: int, symbol: str, lookback_days: int = 3) -> Tuple[Optional[object], Optional[object]]:
    """Retorna (deal_in, deal_out) do histórico, se encontrados."""
    if mt5 is None:
        return None, None
    now = dt.datetime.now()
    start = now - dt.timedelta(days=lookback_days)
    if not _ensure_history_selected(start, now):
        return None, None
    try:
        deals = mt5.history_deals_get(start, now) or []
    except Exception:
        deals = []
    deal_in, deal_out = None, None
    for d in deals:
        try:
            if d.position_id == position_id and d.symbol == symbol:
                # entry: 0 = IN, 1 = OUT (DEAL_ENTRY_IN/OUT)
                if d.entry == getattr(mt5, "DEAL_ENTRY_IN", 0) and deal_in is None:
                    deal_in = d
                elif d.entry == getattr(mt5, "DEAL_ENTRY_OUT", 1) and deal_out is None:
                    deal_out = d
        except Exception:
            continue
    return deal_in, deal_out

def _dt_from_timestamp(ts: Optional[int]) -> Optional[dt.datetime]:
    if ts is None:
        return None
    try:
        return dt.datetime.fromtimestamp(ts)
    except Exception:
        return None

def finalize_close_log_from_deals(env,
                                  position_id: int,
                                  symbol: str,
                                  tipo: str,
                                  sl: float,
                                  tp: float,
                                  closed_by: str) -> None:
    """
    Consolida e envia uma linha completa para o log de operações v3 no momento do fechamento.

    Parâmetros:
    - env: objeto de ambiente que contenha env.cfg (dict) e, idealmente, último snapshot de métricas (regime, atr etc.)
    - position_id: ID da posição (pid) a ser conciliada no histórico
    - symbol: símbolo (ex: 'WINZ25')
    - tipo: 'BUY' ou 'SELL'
    - sl / tp: valores usados na abertura (p/ registro)
    - closed_by: 'IA-EXIT', 'EOD' ou 'MANUAL'
    """
    if log_operacoes_v3 is None:
        # Sem o logger não há o que fazer aqui
        return

    # Busca deals IN/OUT e calcula lucro, tempo de close e duração
    deal_in, deal_out = _fetch_in_out_deals(position_id, symbol, lookback_days=5)
    lucro_real = None
    close_dt = None
    duracao_seg = None

    if deal_out is not None:
        try:
            lucro_real = float(getattr(deal_out, "profit", 0.0))
        except Exception:
            lucro_real = 0.0
        close_dt = _dt_from_timestamp(getattr(deal_out, "time", None))

    if deal_in is not None and close_dt is not None:
        open_dt = _dt_from_timestamp(getattr(deal_in, "time", None))
        if open_dt is not None:
            duracao_seg = (close_dt - open_dt).total_seconds()

    # Motivo saída: prioriza SL/TP do MetaTrader; caso não detecte, usa closed_by
    motivo_mt5 = _classify_reason_from_deal(deal_out)
    motivo_saida = motivo_mt5 if motivo_mt5 in ("SL", "TP") else (closed_by or "MANUAL")

    # Fechado por IA?
    fechado_por_ia = "sim" if closed_by == "IA-EXIT" else "nao"

    # Hora para o log (fallback = now)
    hora_txt = (close_dt or dt.datetime.now()).strftime("%H:%M:%S")

    # Tenta obter version_tag do cfg
    version_tag = None
    try:
        version_tag = env.cfg.get("version_tag")
    except Exception:
        pass

    # Monta campos mínimos (demais campos o seu logger já preenche/conhece)
    extra = {
        "motivo_saida": motivo_saida,
        "lucro_real": lucro_real if lucro_real is not None else 0.0,
        "duracao_operacao": duracao_seg if duracao_seg is not None else 0.0,
        "fechado_por_ia": fechado_por_ia,
        "version_tag": version_tag,
    }

    # Dispara o logger oficial v3 (assumindo assinatura flexível por kwargs)
    try:
        log_operacoes_v3(
            hora=hora_txt,
            pid=position_id,
            tipo=tipo,
            sl=sl,
            tp=tp,
            **extra
        )
    except TypeError:
        # Caso sua função não aceite kwargs livres, tente versão reduzida e/ou adapte no logging_utils
        log_operacoes_v3(
            hora=hora_txt,
            pid=position_id,
            tipo=tipo,
            sl=sl,
            tp=tp,
            motivo_saida=extra["motivo_saida"],
            lucro_real=extra["lucro_real"],
            duracao_operacao=extra["duracao_operacao"],
            fechado_por_ia=extra["fechado_por_ia"],
            version_tag=extra["version_tag"],
        )
