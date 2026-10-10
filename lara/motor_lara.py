# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import math
import json
import os
import sys
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from log_fechamento import finalize_close_log_from_deals
from auditoria_cooldown import (
    ensure_cd_audit_state,
    register_closed_trade_for_audit,
    build_cooldown_audit_fields,
)

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

from logging_utils import (
    log_ia_exec,
    log_operacao,
    log_120s,
    log_validacao_incremental,
    ensure_logs_dir,
)
from config_utils import load_json, resolve_config_path, validate_and_normalize_config

# Offset (em segundos) aplicado em timestamps de deals/history do MT5, se detectado.
MT5_TIME_OFFSET_SEC = 0

VERSION_TAG = "Lara_3.3.1_XGB_2026-01-19_PC_PV_GATE"

# Ordem padrão das features usadas no TREINO do modelo v2
DEFAULT_XGB_FEATURES = [
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


def _mt5_time_local_hora(ts: int) -> str:
    """
    Converte timestamp MT5 (segundos) para HH:MM:SS no horário local,
    aplicando offset detectado (MT5_TIME_OFFSET_SEC) para padronizar.
    """
    try:
        _ts = int(ts)
        if MT5_TIME_OFFSET_SEC:
            _ts = _ts + int(MT5_TIME_OFFSET_SEC)
        return dt.datetime.fromtimestamp(_ts).strftime("%H:%M:%S")
    except Exception:
        return dt.datetime.now().strftime("%H:%M:%S")


def _real_trading_enabled(cfg: Dict[str, Any]) -> bool:
    """Hard safety: real trading only when simulacao==False AND permitir_execucao_real==True."""
    try:
        if bool(cfg.get("simulacao", True)):
            return False
        return bool(cfg.get("permitir_execucao_real", False))
    except Exception:
        return False


class Env:
    def __init__(self, cfg: Dict[str, Any]):
        self._last_ia_at = 0.0
        self._last_tel_at = 0.0

        self.cfg = cfg
        self.symbol = cfg["symbol"]
        self.pause = float(cfg.get("pause_seg", 2.0))

        self.logs_dir = Path(cfg["paths"]["pasta_logs"])
        ensure_logs_dir(self.logs_dir)

        self._cooldown_until: Optional[float] = None

        self._xgb_loaded = False
        self._xgb_model = None
        self._xgb_features: Optional[List[str]] = None
        self._modelo_bloqueio: Optional[str] = None
        self._xgb_calibracao: str = "nenhuma"
        self._xgb_meta: Dict[str, Any] = {}

        self.logged_deals: set[int] = set()

        # pid_cache: guarda dados do momento da abertura para reconciliação posterior
        # Ex: {"pc":..., "pv":..., "direcao":..., "conf_dir":..., "limiar_dir":..., "ind":{...}}
        self.pid_cache: Dict[int, Dict[str, Any]] = {}

        # open recon throttle
        self._last_open_recon_at = 0.0

        # store detected offset
        self.mt5_time_offset_sec = 0

        # opens already logged
        self.logged_open_pids: set[int] = set()

    def connect(self) -> bool:
        if mt5 is None:
            print("[ERRO] MetaTrader5 indisponível. pip install MetaTrader5")
            return False
        if not mt5.initialize():
            print("[ERRO] initialize MT5:", mt5.last_error())
            return False
        ai = mt5.account_info()
        if ai is None:
            print("[ERRO] Conta não conectada no MT5.")
            return False
        print(f"[OK] MT5 conectado: {ai.login} | TradeAllowed={ai.trade_allowed}")

        # Detecta diferença de fuso nos timestamps do MT5 (deals/history).
        global MT5_TIME_OFFSET_SEC
        self.mt5_time_offset_sec = 0
        try:
            ts_srv = mt5.time_current()
            if ts_srv:
                local_from_srv = dt.datetime.fromtimestamp(int(ts_srv))
                delta = (dt.datetime.now() - local_from_srv).total_seconds()
                if abs(delta) >= 1800:
                    MT5_TIME_OFFSET_SEC = int(round(delta / 3600.0)) * 3600
                    self.mt5_time_offset_sec = MT5_TIME_OFFSET_SEC
                    print(f"[INFO] Offset MT5 detectado: {MT5_TIME_OFFSET_SEC} s (aplicado em deals/history)")
                else:
                    MT5_TIME_OFFSET_SEC = 0
                    self.mt5_time_offset_sec = 0
        except Exception:
            MT5_TIME_OFFSET_SEC = 0
            self.mt5_time_offset_sec = 0

        return True

    def ensure_symbol(self, symbol: str) -> bool:
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"[ERRO] Símbolo não encontrado: {symbol}")
            return False
        if not info.visible and not mt5.symbol_select(symbol, True):
            print(f"[ERRO] Não foi possível selecionar {symbol}")
            return False
        print(f"[OK] Símbolo OK: {symbol} | point={info.point}")
        return True

    def trading_window_open(self) -> bool:
        now = dt.datetime.now().time()
        return now < dt.time(18, 0, 0)

    def is_eod_window(self) -> bool:
        now = dt.datetime.now().time()
        return dt.time(17, 58, 0) <= now <= dt.time(18, 0, 0)

    def get_tick(self) -> Optional[Dict[str, Any]]:
        t = mt5.symbol_info_tick(self.symbol)
        return None if t is None else {"bid": float(t.bid), "ask": float(t.ask), "time": int(t.time)}

    def get_rates(self, timeframe=None, count=300) -> Optional[List[Dict[str, Any]]]:
        if mt5 is None:
            return None
        if timeframe is None:
            timeframe = getattr(mt5, "TIMEFRAME_M1", 1)
        r = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        return None if r is None else list(r)

    def set_cooldown_candles(self, candles: int):
        self._cooldown_until = time.time() + 60.0 * max(1, int(candles))

    def cooldown_restante(self) -> int:
        if self._cooldown_until is None:
            return 0
        return max(0, int(self._cooldown_until - time.time()))


# --------- calibração das probabilidades do XGB ---------
def _aplicar_calibracao(p: float, calib: str) -> float:
    if calib is None:
        calib = ""
    calib = str(calib).lower().strip()

    if calib in ("nenhuma", "none", "raw", ""):
        pass
    elif calib == "sigmoid":
        p = 1.0 / (1.0 + math.exp(-8.0 * (p - 0.5)))

    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    return p


# --------- indicadores básicos ---------
def ema(values: List[float], period: int) -> List[float]:
    if period <= 1 or len(values) == 0:
        return values[:]
    k = 2.0 / (period + 1.0)
    out, e = [], values[0]
    for v in values:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def true_range(h: float, l: float, prev_close: float) -> float:
    return max(h - l, abs(h - prev_close), abs(l - prev_close))


def compute_indicators(env: Env, cfg: Dict[str, Any]) -> Optional[Dict[str, float]]:
    rates = env.get_rates(mt5.TIMEFRAME_M1, count=300)
    if not rates or len(rates) < 120:
        return None

    closes = [float(r["close"]) for r in rates]
    highs = [float(r["high"]) for r in rates]
    lows = [float(r["low"]) for r in rates]

    ema_c = ema(closes, int(cfg["ema_curta"]))[-1]
    ema_l = ema(closes, int(cfg["ema_longa"]))[-1]

    trs = []
    for i in range(1, len(rates)):
        trs.append(true_range(highs[i], lows[i], closes[i - 1]))

    atr_period = int(cfg["atr_periodo"])
    if len(trs) < atr_period:
        return None
    atr = sum(trs[-atr_period:]) / atr_period

    atr_media_ref = int(cfg["atr_media_ref"])
    if len(trs) < atr_media_ref:
        return None
    atr50 = sum(trs[-atr_media_ref:]) / atr_media_ref

    tendencia_pct = (ema_c - ema_l) / ema_l if ema_l != 0 else 0.0
    vol_norm = (atr / atr50) if atr50 > 0 else 0.0
    regime = "uptrend" if ema_c > ema_l else "downtrend" if ema_c < ema_l else "side"

    return dict(
        ema_curta=float(ema_c),
        ema_longa=float(ema_l),
        tendencia_pct=float(tendencia_pct),
        atr=float(atr),
        atr_media50=float(atr50),
        vol_norm=float(vol_norm),
        regime_mercado=regime,
        preco_atual=float(closes[-1]),
    )


def _model_is_fresh(path: str, max_days: int) -> bool:
    try:
        mtime = os.path.getmtime(path)
        age_days = (time.time() - mtime) / 86400.0
        return age_days <= max_days
    except Exception:
        return False


def _load_xgb(env: Env, cfg: Dict[str, Any]) -> None:
    if env._xgb_loaded:
        return

    if not cfg.get("usar_modelo_xgb", False):
        env._modelo_bloqueio = "modelo_desligado"
        env._xgb_loaded = True
        return

    if xgb is None:
        env._modelo_bloqueio = "xgboost_nao_instalado"
        env._xgb_loaded = True
        return

    path = cfg.get("caminho_modelo", "")
    if not path or not os.path.exists(path):
        env._modelo_bloqueio = "arquivo_inexistente"
        env._xgb_loaded = True
        return

    if not _model_is_fresh(path, int(cfg.get("modelo_max_idade_dias", 14))):
        env._modelo_bloqueio = "modelo_antigo"
        env._xgb_loaded = True
        return

    meta: Dict[str, Any] = {}
    meta_path = path.replace(".json", ".meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            auc = float(meta.get("auc_kfold_mean", meta.get("auc", 0.0)))
            if auc < float(cfg.get("qualidade_minima_auc", 0.55)):
                env._modelo_bloqueio = "qualidade_insuficiente"
                env._xgb_loaded = True
                return

            feats = meta.get("features", None)
            if isinstance(feats, list) and feats:
                env._xgb_features = [str(f) for f in feats]
            env._xgb_meta = meta
            print(f"[XGB] Meta carregado: auc={auc:.3f} | calibracao={meta.get('calibracao', 'nenhuma')} | features={len(env._xgb_features or [])}")
        except Exception as exc:
            print(f"[WARN][XGB] Falha ao ler meta.json: {exc}")
            meta = {}

    calib_meta = meta.get("calibracao")
    calib_cfg = cfg.get("calibracao")
    env._xgb_calibracao = calib_meta if calib_meta is not None else (calib_cfg or "nenhuma")

    try:
        booster = xgb.Booster()
        booster.load_model(path)
        env._xgb_model = booster
        env._xgb_loaded = True
        env._modelo_bloqueio = None
    except Exception:
        env._modelo_bloqueio = "erro_carregar_booster"
        env._xgb_loaded = True


def ajustar_limiares_por_vol(cfg: Dict[str, Any], ind: Dict[str, float]) -> Tuple[float, float]:
    base_c = float(cfg["limiar_compra_base"])
    base_v = float(cfg["limiar_venda_base"])
    aj = float(cfg.get("ajuste_limiar_vol", 0.0))

    if not cfg.get("usar_kill_switch", True):
        return base_c, base_v

    if ind["vol_norm"] >= 1.5:
        return min(0.80, base_c + aj), min(0.80, base_v + aj)

    return base_c, base_v


def _features_from_ind(env: Env, cfg: Dict[str, Any], ind: Dict[str, float]) -> Dict[str, float]:
    """
    Monta features no formato do treino v2.
    Observação: como o modelo atual entrega 1 score (confiança/acerto),
    usamos heurística para pc/pv nesta função (compatibilidade de treino).
    """
    tmin = float(cfg.get("tendencia_pct_min", 0.003))
    if ind["tendencia_pct"] >= tmin and ind["regime_mercado"] == "uptrend":
        pc_h, pv_h = 0.60, 0.40
    elif ind["tendencia_pct"] <= -tmin and ind["regime_mercado"] == "downtrend":
        pc_h, pv_h = 0.40, 0.60
    else:
        pc_h, pv_h = 0.50, 0.50

    lim_c, lim_v = ajustar_limiares_por_vol(cfg, ind)

    preco = float(ind["preco_atual"])
    sl_points = float(cfg.get("sl_points", 0.0))
    tp_points = float(cfg.get("tp_points", 0.0))

    return {
        "preco": preco,
        "sl": sl_points,
        "tp": tp_points,
        "tendencia_pct": float(ind["tendencia_pct"]),
        "atr": float(ind["atr"]),
        "atr_media50": float(ind["atr_media50"]),
        "vol_norm": float(ind["vol_norm"]),
        "pc": float(pc_h),
        "pv": float(pv_h),
        "limiar_compra_ajustado": float(lim_c),
        "limiar_venda_ajustado": float(lim_v),
    }


def _xgb_predict_p(env: Env, cfg: Dict[str, Any], ind: Dict[str, float]) -> Optional[float]:
    _load_xgb(env, cfg)
    if env._modelo_bloqueio:
        return None
    if env._xgb_model is None or xgb is None:
        return None

    feats_dict = _features_from_ind(env, cfg, ind)
    feat_order = list(env._xgb_features) if env._xgb_features else list(DEFAULT_XGB_FEATURES)

    for f in feat_order:
        if f not in feats_dict:
            env._modelo_bloqueio = "features_incompativeis"
            return None

    row = [float(feats_dict[f]) for f in feat_order]

    try:
        dm = xgb.DMatrix([row], feature_names=feat_order)
        p = float(env._xgb_model.predict(dm)[0])
    except Exception:
        env._modelo_bloqueio = "erro_predict"
        return None

    p = _aplicar_calibracao(p, getattr(env, "_xgb_calibracao", None))
    return p


def infer_probas(env: Env, cfg: Dict[str, Any], ind: Dict[str, float]) -> Tuple[float, float, str]:
    """
    Retorna (pc, pv, motivo).

    Modelo atual entrega 1 score p.
    Compatibilizacao direcional:
    - uptrend   -> pc=p,   pv=1-p
    - downtrend -> pc=1-p, pv=p
    - side      -> pc=0.5, pv=0.5
    """
    motivo = ""
    regime = str(ind.get("regime_mercado", "")).lower().strip()
    pc, pv = 0.50, 0.50

    p = None
    if cfg.get("usar_modelo_xgb", False):
        p = _xgb_predict_p(env, cfg, ind)
        if p is None:
            motivo = env._modelo_bloqueio or "modelo_invalido"
        else:
            p = float(max(0.0, min(1.0, p)))
            if regime == "uptrend":
                pc, pv = p, 1.0 - p
            elif regime == "downtrend":
                pc, pv = 1.0 - p, p
            else:
                pc, pv = 0.50, 0.50

    if p is None:
        tmin = float(cfg.get("tendencia_pct_min", 0.003))
        if ind["tendencia_pct"] >= tmin and regime == "uptrend":
            pc, pv = 0.60, 0.40
        elif ind["tendencia_pct"] <= -tmin and regime == "downtrend":
            pc, pv = 0.40, 0.60
        else:
            pc, pv = 0.50, 0.50

    return float(pc), float(pv), motivo

def _retcode_ok(retcode: Any) -> bool:
    try:
        rc = int(retcode)
    except Exception:
        return False
    ok_set = {
        int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
        int(getattr(mt5, "TRADE_RETCODE_PLACED", 10008)),
        int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
    }
    return rc in ok_set


def _retcode_name(retcode: Any) -> str:
    try:
        rc = int(retcode)
    except Exception:
        return str(retcode)
    try:
        for k in dir(mt5):
            if k.startswith("TRADE_RETCODE_"):
                try:
                    if int(getattr(mt5, k)) == rc:
                        return k
                except Exception:
                    continue
    except Exception:
        pass
    return str(rc)


def _digits_for_symbol(symbol: str) -> int:
    try:
        info = mt5.symbol_info(symbol)
        return max(0, int(getattr(info, "digits", 0) or 0))
    except Exception:
        return 2


def _normalize_price(symbol: str, value: float) -> float:
    return round(float(value), _digits_for_symbol(symbol))


def _normalize_volume(symbol: str, volume: float) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return float(volume)
    vmin = float(getattr(info, "volume_min", 0.0) or 0.0)
    vmax = float(getattr(info, "volume_max", 0.0) or 0.0)
    step = float(getattr(info, "volume_step", 0.0) or 0.0)
    v = float(volume)
    if vmax > 0:
        v = min(v, vmax)
    if vmin > 0:
        v = max(v, vmin)
    if step > 0:
        units = round(v / step)
        v = units * step
        v = round(v, max(0, len(str(step).split(".")[-1].rstrip("0"))))
        if vmin > 0:
            v = max(v, vmin)
        if vmax > 0:
            v = min(v, vmax)
    return float(v)


def _min_stop_distance(symbol: str) -> float:
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.0
        point = float(getattr(info, "point", 0.0) or 0.0)
        level = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        freeze = float(getattr(info, "trade_freeze_level", 0.0) or 0.0)
        return max(level, freeze) * point
    except Exception:
        return 0.0


def _filling_candidates(symbol: str) -> List[int]:
    candidates: List[int] = []
    try:
        info = mt5.symbol_info(symbol)
        fm = getattr(info, "filling_mode", None) if info is not None else None
        if fm is not None:
            try:
                candidates.append(int(fm))
            except Exception:
                pass
    except Exception:
        pass
    for name in ("ORDER_FILLING_FOK", "ORDER_FILLING_IOC", "ORDER_FILLING_RETURN"):
        if hasattr(mt5, name):
            try:
                candidates.append(int(getattr(mt5, name)))
            except Exception:
                pass
    out: List[int] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out or [0]


def _sl_tp(symbol: str, order_type: int, price: float, sl_points: float, tp_points: float) -> Tuple[float, float]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return float(price), float(price)
    point = float(getattr(info, "point", 0.0) or 0.0)
    min_dist = _min_stop_distance(symbol)
    if order_type == mt5.ORDER_TYPE_BUY:
        sl = price - sl_points * point
        tp = price + tp_points * point
        if min_dist > 0:
            sl = min(sl, price - min_dist)
            tp = max(tp, price + min_dist)
    else:
        sl = price + sl_points * point
        tp = price - tp_points * point
        if min_dist > 0:
            sl = max(sl, price + min_dist)
            tp = min(tp, price - min_dist)
    return _normalize_price(symbol, sl), _normalize_price(symbol, tp)


def _build_market_request(symbol: str, volume: float, order_type: int, magic: int, price: float, sl: float | None = None, tp: float | None = None, position: int | None = None, comment: str = "") -> Dict[str, Any]:
    req: Dict[str, Any] = dict(
        action=mt5.TRADE_ACTION_DEAL,
        symbol=symbol,
        volume=_normalize_volume(symbol, volume),
        type=order_type,
        price=_normalize_price(symbol, price),
        deviation=20,
        magic=int(magic),
        comment=comment[:31] if comment else "LARA",
    )
    if sl is not None:
        req["sl"] = _normalize_price(symbol, sl)
    if tp is not None:
        req["tp"] = _normalize_price(symbol, tp)
    if position is not None:
        req["position"] = int(position)
    if hasattr(mt5, "ORDER_TIME_GTC"):
        req["type_time"] = getattr(mt5, "ORDER_TIME_GTC")
    return req


def _order_check_ok(req: Dict[str, Any]) -> Tuple[bool, int]:
    if not hasattr(mt5, "order_check"):
        return True, 0
    try:
        chk = mt5.order_check(req)
    except Exception:
        return False, -1
    if chk is None:
        return False, -1
    ret = int(getattr(chk, "retcode", 0) or 0)
    if ret in (0,):
        return True, ret
    return _retcode_ok(ret), ret


def _send_order_with_fallback(req: Dict[str, Any], symbol: str) -> Tuple[bool, int, Optional[int]]:
    last_ret = -1
    for filling in _filling_candidates(symbol):
        trial = dict(req)
        trial["type_filling"] = filling
        ok_chk, ret_chk = _order_check_ok(trial)
        if not ok_chk and ret_chk not in (0, -1):
            last_ret = ret_chk
            print(f"[ORDER_CHECK] {symbol} filling={filling} retcode={ret_chk} ({_retcode_name(ret_chk)})")
            continue
        res = mt5.order_send(trial)
        if res is None:
            last_ret = -1
            print(f"[ORDER_SEND] {symbol} filling={filling} retornou None")
            continue
        ret = int(getattr(res, "retcode", -1) or -1)
        if _retcode_ok(ret):
            return True, ret, getattr(res, "order", None)
        last_ret = ret
        print(f"[ORDER_SEND] {symbol} filling={filling} retcode={ret} ({_retcode_name(ret)})")
    return False, last_ret, None


def send_order_market(
    symbol: str,
    volume: float,
    order_type: int,
    magic: int,
    sl_points: float,
    tp_points: float,
) -> Tuple[bool, int, Optional[int]]:
    if mt5 is None:
        return False, -1, None
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        return False, -1, None
    price = float(tick.ask) if order_type == mt5.ORDER_TYPE_BUY else float(tick.bid)
    sl, tp = _sl_tp(symbol, order_type, price, sl_points, tp_points)
    req = _build_market_request(
        symbol=symbol,
        volume=volume,
        order_type=order_type,
        magic=magic,
        price=price,
        sl=sl,
        tp=tp,
        comment="LARA_OPEN",
    )
    return _send_order_with_fallback(req, symbol)


def close_position_market(symbol: str) -> Tuple[bool, int]:
    if mt5 is None:
        return False, -1
    pos = mt5.positions_get(symbol=symbol)
    if not pos:
        return True, 0
    pos = pos[0]
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    volume = float(pos.volume)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, -1
    price = float(tick.bid) if order_type == mt5.ORDER_TYPE_SELL else float(tick.ask)
    req = _build_market_request(
        symbol=symbol,
        volume=volume,
        order_type=order_type,
        magic=int(pos.magic) if hasattr(pos, "magic") else 0,
        price=price,
        position=int(pos.ticket),
        comment="LARA_CLOSE",
    )
    ok, ret, _ = _send_order_with_fallback(req, symbol)
    return ok, ret


def dry_run_order_checks(symbol: str, volume: float, magic: int, sl_points: float, tp_points: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"symbol": symbol, "checks": []}
    if mt5 is None:
        out["error"] = "MetaTrader5 indisponível"
        return out
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        out["error"] = f"Sem tick para {symbol}"
        return out
    for label, order_type, price in [
        ("BUY", mt5.ORDER_TYPE_BUY, float(tick.ask)),
        ("SELL", mt5.ORDER_TYPE_SELL, float(tick.bid)),
    ]:
        sl, tp = _sl_tp(symbol, order_type, price, sl_points, tp_points)
        req = _build_market_request(
            symbol=symbol,
            volume=volume,
            order_type=order_type,
            magic=magic,
            price=price,
            sl=sl,
            tp=tp,
            comment=f"LARA_{label}_CHECK",
        )
        fill_results = []
        for filling in _filling_candidates(symbol):
            trial = dict(req)
            trial["type_filling"] = filling
            ok_chk, ret_chk = _order_check_ok(trial)
            fill_results.append({
                "label": label,
                "type_filling": int(filling),
                "ok": bool(ok_chk),
                "retcode": int(ret_chk),
                "retcode_name": _retcode_name(ret_chk),
                "volume": trial.get("volume"),
                "price": trial.get("price"),
                "sl": trial.get("sl"),
                "tp": trial.get("tp"),
            })
        out["checks"].append({"side": label, "results": fill_results})
    return out


def can_open_new_position(symbol: str) -> bool:
    pos = mt5.positions_get(symbol=symbol)
    return (not pos) or (len(pos) == 0)


def _wait_position_pid(env: Env, timeout: float = 2.0, poll: float = 0.1) -> Optional[int]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            pos = mt5.positions_get(symbol=env.symbol)
            if pos:
                return int(pos[0].ticket)
        except Exception:
            pass
        time.sleep(poll)
    return None


def decide_and_maybe_send_order(env: Env, cfg: Dict[str, Any]) -> None:
    ind = compute_indicators(env, cfg)
    if not ind:
        time.sleep(env.pause)
        return

    pc, pv, motivo_xgb = infer_probas(env, cfg, ind)
    lim_c, lim_v = ajustar_limiares_por_vol(cfg, ind)

    # Mantém para LOG apenas (não governa o gate do BLOCO_B)
    _lc = cfg.get("limiar_confianca", None)
    limiar_conf = float(_lc) if _lc is not None else float(max(lim_c, lim_v))

    bloqueado_por = ""

    def add_block(reason: str) -> None:
        nonlocal bloqueado_por
        if not reason:
            return
        if not bloqueado_por:
            bloqueado_por = reason
        elif reason not in bloqueado_por.split("|"):
            bloqueado_por = f"{bloqueado_por}|{reason}"

    if motivo_xgb:
        add_block(motivo_xgb)

    if cfg.get("usar_kill_switch", True) and ind["vol_norm"] >= 2.5:
        add_block("kill_switch_vol")

    if cfg.get("usar_filtro_tendencia", True) and (
        (ind["regime_mercado"] == "uptrend" and ind["tendencia_pct"] < cfg.get("tendencia_pct_min", 0.003))
        or (ind["regime_mercado"] == "downtrend" and abs(ind["tendencia_pct"]) < cfg.get("tendencia_pct_min", 0.003))
    ):
        add_block("filtro_tendencia")

    cd_rest = env.cooldown_restante()
    if cd_rest > 0:
        add_block("cooldown")

    regime = ind.get("regime_mercado", "side")

    # Side selection guiado por score. Regime permanece apenas como contexto/filtro.
    direcao = None
    conf = None
    limiar_dir = None

    try:
        _pc = float(pc)
        _pv = float(pv)
        _lim_c = float(lim_c)
        _lim_v = float(lim_v)
    except Exception:
        _pc = _pv = _lim_c = _lim_v = float("nan")

    if _pc > _pv and _pc >= _lim_c:
        direcao = "BUY"
        conf = _pc
        limiar_dir = _lim_c
    elif _pv > _pc and _pv >= _lim_v:
        direcao = "SELL"
        conf = _pv
        limiar_dir = _lim_v
    else:
        add_block("sem_sinal")

    if conf is not None and limiar_dir is not None and conf < limiar_dir:
        add_block("abaixo_limiar_pc_pv")

    if not _real_trading_enabled(cfg):
        add_block("shadow_lock")

    # --- Observabilidade adicional do gate direcional (PC/PV) ---
    # Preenche SEM alterar a logica operacional:
    # a decisao final continua a mesma, inclusive sem_sinal/shadow_lock.
    if direcao is not None:
        direcao_log = direcao
        conf_dir_log = round(float(conf), 6) if conf is not None else -1
        limiar_dir_log = round(float(limiar_dir), 6) if limiar_dir is not None else -1
    else:
        try:
            if _pc > _pv:
                direcao_log = "BUY"
                conf_dir_log = round(float(_pc), 6)
                limiar_dir_log = round(float(_lim_c), 6)
            elif _pv > _pc:
                direcao_log = "SELL"
                conf_dir_log = round(float(_pv), 6)
                limiar_dir_log = round(float(_lim_v), 6)
            else:
                direcao_log = "NONE"
                conf_dir_log = -1
                limiar_dir_log = -1
        except Exception:
            direcao_log = "NONE"
            conf_dir_log = -1
            limiar_dir_log = -1

    now = time.time()
    if now - getattr(env, "_last_ia_at", 0.0) >= float(cfg.get("ia_exec_interval_s", 15)):
        signal_elegivel = (
            direcao_log in ("BUY", "SELL")
            and conf_dir_log not in ("", -1, None)
            and limiar_dir_log not in ("", -1, None)
            and float(conf_dir_log) >= float(limiar_dir_log)
        )

        ia_row = dict(
            hora=dt.datetime.now().strftime("%H:%M:%S"),
            regime_mercado=ind["regime_mercado"],
            direcao=direcao_log,
            conf_dir=conf_dir_log,
            limiar_dir=limiar_dir_log,
            tendencia_pct=round(ind["tendencia_pct"], 6),
            atr=round(ind["atr"], 2),
            atr_media50=round(ind["atr_media50"], 2),
            vol_norm=round(ind["vol_norm"], 3),
            pc=round(pc, 6),
            pv=round(pv, 6),
            limiar_compra_ajustado=float(lim_c),
            limiar_venda_ajustado=float(lim_v),
            limiar_confianca=round(limiar_conf, 6),
            bloqueado_por=bloqueado_por if bloqueado_por else "NONE",
            cooldown_restante=cd_rest,
            version_tag=env.cfg.get("version_tag", VERSION_TAG),
        )

        ia_row.update(
            build_cooldown_audit_fields(
                ctx=env,
                now_time=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                signal_elegivel=signal_elegivel,
                bloqueado_por_existente=ia_row.get("bloqueado_por", "NONE"),
                candle_seconds=120,
            )
        )

        log_ia_exec(env.logs_dir, ia_row)
        env._last_ia_at = now

    if bloqueado_por:
        return

    if not can_open_new_position(env.symbol):
        return

    if not _real_trading_enabled(cfg):
        return

    tick = env.get_tick()
    if (not tick) or tick.get("bid", 0.0) <= 0.0 or tick.get("ask", 0.0) <= 0.0:
        return

    side = direcao
    order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL

    ok, ret, order_id = send_order_market(
        env.symbol,
        float(cfg["lote"]),
        order_type,
        int(cfg["magic"]),
        float(cfg["sl_points"]),
        float(cfg["tp_points"]),
    )
    if ok:
        env.set_cooldown_candles(int(cfg.get("cooldown_candles", 3)))

    pid_logged = _wait_position_pid(env, timeout=2.0) if ok else None
    pid_val = int(pid_logged) if pid_logged is not None else (int(order_id) if order_id else 0)

    try:
        if pid_val:
            env.pid_cache[int(pid_val)] = {
                "pc": round(pc, 6),
                "pv": round(pv, 6),
                "direcao": side,
                "conf_dir": conf_dir_log,
                "limiar_dir": limiar_dir_log,
                "ind": dict(ind),
            }
            env.logged_open_pids.add(int(pid_val))
    except Exception:
        pass

    log_operacao(
        env.logs_dir,
        dict(
            hora=dt.datetime.now().strftime("%H:%M:%S"),
            pid=int(pid_val),
            tipo=side,
            direcao=side,
            conf_dir=conf_dir_log,
            limiar_dir=limiar_dir_log,
            preco=float(ind["preco_atual"]),
            sl=float(cfg["sl_points"]),
            tp=float(cfg["tp_points"]),
            resultado="aberta" if ok else f"erro_{ret}",
            fechado_por_ia="",
            regime_mercado=ind["regime_mercado"],
            tendencia_pct=round(ind["tendencia_pct"], 6),
            atr=round(ind["atr"], 2),
            atr_media50=round(ind["atr_media50"], 2),
            vol_norm=round(ind["vol_norm"], 3),
            pc=round(pc, 6),
            pv=round(pv, 6),
            limiar_compra_ajustado=float(lim_c),
            limiar_venda_ajustado=float(lim_v),
            limiar_confianca=round(limiar_conf, 6),
            motivo_saida="",
            lucro_real="",
            duracao_operacao="",
            version_tag=env.cfg.get("version_tag", VERSION_TAG),
        ),
    )


def _deal_time_to_dt(ts) -> dt.datetime:
    """Converte timestamps de deal/history para datetime local, aplicando offset se detectado."""
    try:
        _ts = int(ts.time) if hasattr(ts, "time") else int(ts)
    except Exception:
        _ts = 0
    t = dt.datetime.fromtimestamp(_ts)
    try:
        if MT5_TIME_OFFSET_SEC:
            t = t + dt.timedelta(seconds=int(MT5_TIME_OFFSET_SEC))
    except Exception:
        pass
    return t


def _find_entry_time_for_pid(env: Env, pid: int, day_start: dt.datetime, t_out: dt.datetime) -> dt.datetime:
    try:
        deals = mt5.history_deals_get(day_start, t_out) or []
        best = None
        for d in deals:
            if getattr(d, "symbol", "") != env.symbol:
                continue
            if int(getattr(d, "position_id", 0) or 0) != pid:
                continue
            if getattr(d, "entry", None) == getattr(mt5, "DEAL_ENTRY_IN", 0):
                t = _deal_time_to_dt(d)
                if best is None or t < best:
                    best = t
        return best or t_out
    except Exception:
        return t_out


def _wait_profit_for_pid(env: Env, pid: int, timeout: float = 8.0, poll: float = 0.2):
    try:
        if not mt5:
            return None
        start = dt.datetime.now()
        day_start = dt.datetime(start.year, start.month, start.day, 0, 0, 0)
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            deals = mt5.history_deals_get(day_start, dt.datetime.now()) or []
            for d in deals:
                if getattr(d, "symbol", "") != env.symbol:
                    continue
                if int(getattr(d, "position_id", 0) or 0) != int(pid):
                    continue
                if getattr(d, "entry", None) != getattr(mt5, "DEAL_ENTRY_OUT", 1):
                    continue

                reason = getattr(d, "reason", None)
                comment = (getattr(d, "comment", "") or "").lower()
                REASONS = {
                    getattr(mt5, "DEAL_REASON_SL", 0): "SL",
                    getattr(mt5, "DEAL_REASON_TP", 0): "TP",
                    getattr(mt5, "DEAL_REASON_EXPERT", 0): "IA-EXIT",
                    getattr(mt5, "DEAL_REASON_CLIENT", 0): "MANUAL",
                }
                motivo = REASONS.get(reason, "")
                if not motivo:
                    if "sl" in comment:
                        motivo = "SL"
                    elif "tp" in comment:
                        motivo = "TP"
                    elif "expert" in comment or "ea" in comment:
                        motivo = "IA-EXIT"
                    elif "manual" in comment or "close" in comment:
                        motivo = "MANUAL"
                    else:
                        motivo = "OUT"

                lucro = float(getattr(d, "profit", 0.0) or 0.0)
                t_out = _deal_time_to_dt(d)
                t_in = _find_entry_time_for_pid(env, int(pid), day_start, t_out)
                duracao = max(0.0, (t_out - t_in).total_seconds())
                return round(lucro, 2), motivo, round(duracao, 2)
            time.sleep(poll)
    except Exception:
        return None
    return None


def maybe_close_by_ia(env: Env, cfg: Dict[str, Any]) -> None:
    ind = compute_indicators(env, cfg)
    if not ind:
        return

    if not _real_trading_enabled(cfg):
        return

    pc, pv, _ = infer_probas(env, cfg, ind)
    lim_c, lim_v = ajustar_limiares_por_vol(cfg, ind)

    pos = mt5.positions_get(symbol=env.symbol)
    if not pos:
        return
    pos = pos[0]
    side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"

    regime = ind.get("regime_mercado", "side")
    close_it = False

    # Log do gatilho de fechamento (direção oposta)
    direcao_trigger = ""
    conf_dir = ""
    limiar_dir = ""

    if side == "BUY":
        direcao_trigger = "SELL"
        conf_dir = round(float(pv), 6)
        limiar_dir = round(float(lim_v), 6)
        if regime == "downtrend" and float(pv) >= float(lim_v):
            close_it = True
    else:
        direcao_trigger = "BUY"
        conf_dir = round(float(pc), 6)
        limiar_dir = round(float(lim_c), 6)
        if regime == "uptrend" and float(pc) >= float(lim_c):
            close_it = True

    if not close_it:
        return

    ok, ret = close_position_market(env.symbol)

    _info = _wait_profit_for_pid(env, int(pos.ticket), timeout=8.0)
    if _info is None:
        return

    _lucro_real, _motivo_saida, _duracao = _info

    _lc = cfg.get("limiar_confianca", None)
    limiar_conf = float(_lc) if _lc is not None else float(max(lim_c, lim_v))

    log_operacao(
        env.logs_dir,
        dict(
            hora=dt.datetime.now().strftime("%H:%M:%S"),
            pid=int(pos.ticket),
            tipo="close_" + side,
            direcao=direcao_trigger,
            conf_dir=conf_dir,
            limiar_dir=limiar_dir,
            preco=float(ind["preco_atual"]),
            sl=float(cfg["sl_points"]),
            tp=float(cfg["tp_points"]),
            resultado="fechada" if ok else f"erro_close_{ret}",
            fechado_por_ia="sim",
            regime_mercado=ind["regime_mercado"],
            tendencia_pct=round(ind["tendencia_pct"], 6),
            atr=round(ind["atr"], 2),
            atr_media50=round(ind["atr_media50"], 2),
            vol_norm=round(ind["vol_norm"], 3),
            pc=round(pc, 6),
            pv=round(pv, 6),
            limiar_compra_ajustado=float(lim_c),
            limiar_venda_ajustado=float(lim_v),
            limiar_confianca=round(limiar_conf, 6),
            motivo_saida=_motivo_saida,
            lucro_real=round(_lucro_real, 2),
            duracao_operacao=round(_duracao, 2),
            version_tag=env.cfg.get("version_tag", VERSION_TAG),
        ),
    )

    log_validacao_incremental(
        env.logs_dir,
        dict(
            hora=dt.datetime.now().strftime("%H:%M:%S"),
            pid=int(pos.ticket),
            tipo="fechada",
            resultado="ok" if ok else f"erro_close_{ret}",
            lucro_real=round(_lucro_real, 2),
            motivo_saida=_motivo_saida,
            version_tag=env.cfg.get("version_tag", VERSION_TAG),
        ),
    )


def telemetry_flush(env: Env) -> None:
    try:
        intv = float(getattr(env, "cfg", {}).get("telemetry_interval_s", 120))
        if time.time() - getattr(env, "_last_tel_at", 0.0) < intv:
            return
        t = mt5.symbol_info_tick(env.symbol)
        price = float(getattr(t, "bid", 0.0) or 0.0) if t else 0.0
        log_120s(
            env.logs_dir,
            dict(
                hora=dt.datetime.now().strftime("%H:%M:%S"),
                symbol=env.symbol,
                preco=price,
                positions=len(mt5.positions_get(symbol=env.symbol) or []),
                version_tag=env.cfg.get("version_tag", VERSION_TAG),
            ),
        )
        env._last_tel_at = time.time()
    except Exception:
        pass


def eod_close_all(env: Env, cfg: Dict[str, Any]) -> None:
    if not _real_trading_enabled(cfg):
        return
    pos = mt5.positions_get(symbol=env.symbol)
    if pos:
        close_position_market(env.symbol)


def reconcile_deals(env: Env, cfg: Dict[str, Any]) -> None:
    """
    Loga FECHAMENTOS (DEAL_ENTRY_OUT) do dia no log_operacoes_v3 e log_validacao_incremental,
    com limiar_*_ajustado consistente (quando possível).
    """
    try:
        if not mt5:
            return

        now = dt.datetime.now()
        day_start = dt.datetime(now.year, now.month, now.day, 0, 0, 0)
        deals = mt5.history_deals_get(day_start, now) or []

        ind = None
        try:
            ind = compute_indicators(env, cfg)
        except Exception:
            ind = None

        if ind:
            try:
                lim_c_adj, lim_v_adj = ajustar_limiares_por_vol(cfg, ind)
            except Exception:
                lim_c_adj = float(cfg.get("limiar_compra_base", 0.55))
                lim_v_adj = float(cfg.get("limiar_venda_base", 0.45))
        else:
            lim_c_adj = float(cfg.get("limiar_compra_base", 0.55))
            lim_v_adj = float(cfg.get("limiar_venda_base", 0.45))

        magic_cfg = int(cfg.get("magic", 0) or 0)

        for d in deals:
            try:
                if getattr(d, "symbol", "") != env.symbol:
                    continue
                if getattr(d, "entry", None) != getattr(mt5, "DEAL_ENTRY_OUT", 1):
                    continue

                # filtro strict por magic quando definido
                try:
                    mag = int(getattr(d, "magic", 0) or 0)
                    if magic_cfg and mag != magic_cfg:
                        continue
                except Exception:
                    if magic_cfg:
                        continue

                ticket = int(getattr(d, "ticket", 0))
                if ticket in env.logged_deals:
                    continue

                reason = getattr(d, "reason", None)
                comment = (getattr(d, "comment", "") or "").lower()
                REASONS = {
                    getattr(mt5, "DEAL_REASON_SL", 0): "SL",
                    getattr(mt5, "DEAL_REASON_TP", 0): "TP",
                    getattr(mt5, "DEAL_REASON_EXPERT", 0): "IA-EXIT",
                    getattr(mt5, "DEAL_REASON_CLIENT", 0): "MANUAL",
                }
                motivo = REASONS.get(reason, "")
                if not motivo:
                    if "sl" in comment:
                        motivo = "SL"
                    elif "tp" in comment:
                        motivo = "TP"
                    elif "expert" in comment or "ea" in comment:
                        motivo = "IA-EXIT"
                    elif "manual" in comment or "close" in comment:
                        motivo = "MANUAL"
                    else:
                        motivo = "OUT"

                lucro = round(float(getattr(d, "profit", 0.0) or 0.0), 2)
                pid = int(getattr(d, "position_id", 0) or 0)

                t_out = _deal_time_to_dt(d)
                t_in = _find_entry_time_for_pid(env, pid, day_start, t_out)
                duracao = round(max(0.0, (t_out - t_in).total_seconds()), 2)

                register_closed_trade_for_audit(
                    ctx=env,
                    exit_time=t_out,
                    result_pts=lucro,
                    result_label=("WIN" if lucro > 0 else "LOSS" if lucro < 0 else "EOD"),
                    cooldown_barras=int(cfg.get("cooldown_barras", 3)),
                    candle_seconds=120,
                    cooldown_apenas_apos_loss=bool(cfg.get("cooldown_apenas_apos_loss", True)),
                )

                pc_val = pv_val = ""
                if ind:
                    try:
                        _pc, _pv, _ = infer_probas(env, cfg, ind)
                        pc_val, pv_val = round(_pc, 6), round(_pv, 6)
                    except Exception:
                        pass

                # tenta pegar do cache da abertura (preferível)
                direcao_val = ""
                conf_dir_val = ""
                limiar_dir_val = ""
                if pid in env.pid_cache:
                    try:
                        _c = env.pid_cache.get(pid) or {}
                        if pc_val == "":
                            pc_val = _c.get("pc", "")
                        if pv_val == "":
                            pv_val = _c.get("pv", "")
                        direcao_val = _c.get("direcao", "") or ""
                        conf_dir_val = _c.get("conf_dir", "") or ""
                        limiar_dir_val = _c.get("limiar_dir", "") or ""
                    except Exception:
                        pass

                close_side = (
                    "close_BUY"
                    if getattr(d, "type", None) == getattr(mt5, "DEAL_TYPE_SELL", 1)
                    else "close_SELL"
                )

                # fallback se não tinha no cache: assume direção da posição (close_BUY => posição BUY)
                if not direcao_val:
                    direcao_val = "BUY" if close_side == "close_BUY" else "SELL"

                if conf_dir_val == "" or limiar_dir_val == "":
                    try:
                        if direcao_val == "BUY":
                            conf_dir_val = pc_val
                            limiar_dir_val = round(float(lim_c_adj), 6)
                        else:
                            conf_dir_val = pv_val
                            limiar_dir_val = round(float(lim_v_adj), 6)
                    except Exception:
                        pass

                log_validacao_incremental(
                    env.logs_dir,
                    dict(
                        hora=_mt5_time_local_hora(int(getattr(d, "time", 0) or 0)),
                        pid=pid,
                        tipo="fechada",
                        resultado="ok",
                        lucro_real=lucro,
                        motivo_saida=motivo,
                        version_tag=env.cfg.get("version_tag", VERSION_TAG),
                    ),
                )

                log_operacao(
                    env.logs_dir,
                    dict(
                        hora=_mt5_time_local_hora(int(getattr(d, "time", 0) or 0)),
                        pid=pid,
                        tipo=close_side,
                        direcao=direcao_val,
                        conf_dir=conf_dir_val,
                        limiar_dir=limiar_dir_val,
                        preco=float(getattr(d, "price", 0.0) or 0.0),
                        sl=float(cfg.get("sl_points", 0)),
                        tp=float(cfg.get("tp_points", 0)),
                        resultado="fechada",
                        fechado_por_ia="sim" if motivo == "IA-EXIT" else "",
                        regime_mercado=(ind or {}).get("regime_mercado", "") if ind else "",
                        tendencia_pct=round((ind or {}).get("tendencia_pct", 0.0), 6) if ind else "",
                        atr=round((ind or {}).get("atr", 0.0), 2) if ind else "",
                        atr_media50=round((ind or {}).get("atr_media50", 0.0), 2) if ind else "",
                        vol_norm=round((ind or {}).get("vol_norm", 0.0), 3) if ind else "",
                        pc=pc_val,
                        pv=pv_val,
                        limiar_compra_ajustado=float(lim_c_adj),
                        limiar_venda_ajustado=float(lim_v_adj),
                        motivo_saida=motivo,
                        lucro_real=lucro,
                        duracao_operacao=duracao,
                        version_tag=env.cfg.get("version_tag", VERSION_TAG),
                    ),
                )

                env.logged_deals.add(ticket)
            except Exception:
                continue
    except Exception:
        return


def reconcile_aberturas(env: Env, cfg: Dict[str, Any]) -> None:
    """
    Garante que toda ABERTURA (DEAL_ENTRY_IN) do dia apareça no log_operacoes_v3 como resultado='aberta'.
    Apenas LOG (não envia/fecha ordens).
    """
    if not mt5:
        return

    # Throttle
    try:
        now_ts = time.time()
        if now_ts - float(getattr(env, "_last_open_recon_at", 0.0) or 0.0) < 10.0:
            return
        env._last_open_recon_at = now_ts
    except Exception:
        pass

    # Seed do CSV do dia (para evitar duplicar)
    try:
        if not env.logged_open_pids:
            import csv



            p = env.logs_dir / f"log_operacoes_v3_{dt.date.today().isoformat()}.csv"
            if p.exists():
                with open(p, "r", newline="", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        if str(row.get("resultado", "")).lower().startswith("aberta"):
                            try:
                                env.logged_open_pids.add(int(float(row.get("pid", 0) or 0)))
                            except Exception:
                                pass
    except Exception:
        pass

    try:
        now = dt.datetime.now()
        day_start = dt.datetime(now.year, now.month, now.day, 0, 0, 0)
        deals = mt5.history_deals_get(day_start, now)
        if not deals:
            return
    except Exception:
        return

    ind = None
    try:
        ind = compute_indicators(env, cfg)
    except Exception:
        ind = None

    # Limiar ajustado por vol (se tiver ind), senão base
    try:
        if ind:
            lim_c_adj, lim_v_adj = ajustar_limiares_por_vol(cfg, ind)
        else:
            lim_c_adj = float(cfg.get("limiar_compra_base", ""))
            lim_v_adj = float(cfg.get("limiar_venda_base", ""))
    except Exception:
        lim_c_adj = float(cfg.get("limiar_compra_base", ""))
        lim_v_adj = float(cfg.get("limiar_venda_base", ""))

    magic_cfg = int(cfg.get("magic", 0) or 0)

    for d in deals:
        try:
            if getattr(d, "symbol", "") != env.symbol:
                continue
            if getattr(d, "entry", None) != getattr(mt5, "DEAL_ENTRY_IN", 0):
                continue

            # filtro strict por magic quando definido (evita "manual" entrar)
            try:
                mag = int(getattr(d, "magic", 0) or 0)
                if magic_cfg and mag != magic_cfg:
                    continue
            except Exception:
                if magic_cfg:
                    continue

            pid = int(getattr(d, "position_id", 0) or 0)
            if not pid or pid in env.logged_open_pids:
                continue

            deal_type = int(getattr(d, "type", -1))
            if deal_type == getattr(mt5, "DEAL_TYPE_BUY", 0):
                tipo = "BUY"
            elif deal_type == getattr(mt5, "DEAL_TYPE_SELL", 1):
                tipo = "SELL"
            else:
                continue

            hora = _mt5_time_local_hora(int(getattr(d, "time", 0) or 0))
            price = float(getattr(d, "price", 0.0) or 0.0)

            sl = float(cfg.get("sl_points", "")) if cfg.get("sl_points", "") != "" else ""
            tp = float(cfg.get("tp_points", "")) if cfg.get("tp_points", "") != "" else ""

            pc_val = pv_val = ""
            try:
                if pid in env.pid_cache:
                    pc_val = env.pid_cache[pid].get("pc", "")
                    pv_val = env.pid_cache[pid].get("pv", "")
                elif ind:
                    _pc, _pv, _ = infer_probas(env, cfg, ind)
                    pc_val, pv_val = round(_pc, 6), round(_pv, 6)
            except Exception:
                pass

            # direcao/conf_dir/limiar_dir para abertura
            direcao_val = tipo
            conf_dir_val = pc_val if tipo == "BUY" else pv_val
            limiar_dir_val = round(float(lim_c_adj), 6) if tipo == "BUY" else round(float(lim_v_adj), 6)

            log_operacao(
                env.logs_dir,
                dict(
                    hora=hora,
                    pid=int(pid),
                    tipo=tipo,
                    direcao=direcao_val,
                    conf_dir=conf_dir_val,
                    limiar_dir=limiar_dir_val,
                    preco=price,
                    sl=sl,
                    tp=tp,
                    resultado="aberta",
                    fechado_por_ia="",
                    regime_mercado=(ind or {}).get("regime_mercado", "") if ind else "",
                    tendencia_pct=round((ind or {}).get("tendencia_pct", 0.0), 6) if ind else "",
                    atr=round((ind or {}).get("atr", 0.0), 2) if ind else "",
                    atr_media50=round((ind or {}).get("atr_media50", 0.0), 2) if ind else "",
                    vol_norm=round((ind or {}).get("vol_norm", 0.0), 3) if ind else "",
                    pc=pc_val,
                    pv=pv_val,
                    limiar_compra_ajustado=float(lim_c_adj) if lim_c_adj != "" else "",
                    limiar_venda_ajustado=float(lim_v_adj) if lim_v_adj != "" else "",
                    limiar_confianca=float(cfg.get("limiar_confianca", "")) if cfg.get("limiar_confianca", "") != "" else "",
                    motivo_saida="",
                    lucro_real="",
                    duracao_operacao="",
                    version_tag=env.cfg.get("version_tag", VERSION_TAG),
                ),
            )

            env.logged_open_pids.add(pid)
        except Exception:
            continue


def run_loop(env: Env, cfg: Dict[str, Any]) -> None:
    while env.trading_window_open():
        try:
            if env.is_eod_window():
                eod_close_all(env, cfg)
                time.sleep(env.pause)
                continue

            maybe_close_by_ia(env, cfg)
            decide_and_maybe_send_order(env, cfg)
        except Exception as e:
            print("[ERRO] run_loop:", e)
            time.sleep(env.pause)
        finally:
            try:
                reconcile_aberturas(env, cfg)
            except Exception:
                pass
            try:
                reconcile_deals(env, cfg)
            except Exception:
                pass
            telemetry_flush(env)
            time.sleep(env.pause)


def preflight(env: Env) -> bool:
    if not env.connect():
        return False
    if not env.ensure_symbol(env.symbol):
        return False
    t = env.get_tick()
    if not t:
        print("[ERRO] Sem tick do símbolo.")
        return False
    try:
        ai = mt5.account_info() if mt5 else None
        ti = mt5.terminal_info() if mt5 and hasattr(mt5, "terminal_info") else None
        if ai is not None:
            print(f"[CHECK] Conta={getattr(ai, 'login', '?')} trade_allowed={getattr(ai, 'trade_allowed', '?')} trade_mode={getattr(ai, 'trade_mode', '?')}")
        if ti is not None:
            print(f"[CHECK] Terminal trade_allowed={getattr(ti, 'trade_allowed', '?')} connected={getattr(ti, 'connected', '?')}")
    except Exception:
        pass
    print(f"[CHECK] Pré-flight OK. Último preço ~ {t['bid']}/{t['ask']}")
    return True


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = resolve_config_path(path)
    cfg = load_json(cfg_path)
    validated = validate_and_normalize_config(cfg)
    for warning in validated.warnings:
        print(f"[WARN][CONFIG] {warning}")
    return validated.normalized


def main() -> int:
    if len(sys.argv) >= 2:
        cfg_path = sys.argv[1]
    else:
        cfg_path = os.environ.get("LARA_CONFIG", "config_lara.json")

    if not os.path.exists(cfg_path):
        print(f"[ERRO] Config não encontrado: {cfg_path}")
        return 2

    cfg = load_config(cfg_path)
    env = Env(cfg)

    if not preflight(env):
        return 3

    print(f"[INFO] Rodando: {cfg.get('version_tag', VERSION_TAG)}")
    run_loop(env, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

