# -*- coding: utf-8 -*-
import argparse
import importlib
import time
from pathlib import Path

from config_utils import resolve_config_path, load_json, validate_and_normalize_config

CANDIDATES = [
    "motor_lara",
]


def _try_import():
    last = None
    for module_name in CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Falhou import motor_lara: {last}")


def _load_config(config_path: str | Path):
    path = resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config não encontrado: {path}")
    cfg = load_json(path)
    validated = validate_and_normalize_config(cfg)
    for warning in validated.warnings:
        print(f"[WARN][CONFIG] {warning}")
    return validated.normalized, str(path)


def maybe_send_test_order(api, cfg):
    if not cfg.get("enviar_ordem_teste_no_inicio", False):
        print("[TEST] Flag false; pulando teste.")
        return
    real_enabled = bool(getattr(api, "_real_trading_enabled", lambda _cfg: False)(cfg))
    if not real_enabled:
        print("[TEST] Ordem de teste bloqueada: config ainda está em shadow/simulação.")
        return
    if not hasattr(api, "send_order_market") or not hasattr(api, "close_position_market"):
        print("[WARN] Sem funções de ordem no motor_lara; pulando teste.")
        return
    symbol = str(cfg.get("symbol", "WINZ25"))
    volume = float(cfg.get("lote", 1.0))
    magic = int(cfg.get("magic", 33001))
    sl_pts = float(cfg.get("sl_points", 150))
    tp_pts = float(cfg.get("tp_points", 150))
    try:
        print(f"[TEST] BUY {volume} {symbol} (SL={sl_pts}, TP={tp_pts})")
        ok, ret, order_id = api.send_order_market(symbol, volume, api.mt5.ORDER_TYPE_BUY, magic, sl_pts, tp_pts)  # type: ignore[attr-defined]
        print("[TEST] BUY:", ok, "id=", order_id, "retcode=", ret)
        if ok:
            time.sleep(2.0)
            ok2, ret2 = api.close_position_market(symbol)  # type: ignore[attr-defined]
            print("[TEST] CLOSE:", ok2, "retcode=", ret2)
    except Exception as exc:
        print("[WARN] Falha no teste:", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap do Lara 3.3.1 XGB com validação forte do config.")
    parser.add_argument("--config", default="config_lara.json", help="Caminho do config JSON.")
    args = parser.parse_args()

    cfg, cfg_path = _load_config(args.config)
    api = _try_import()
    env = api.Env(cfg) if hasattr(api, "Env") else None

    if hasattr(api, "preflight"):
        if not api.preflight(env):
            print("[ABORT] preflight falhou")
            return 3
    elif hasattr(api, "ensure_symbol"):
        try:
            api.ensure_symbol(cfg.get("symbol", "WINZ25"))
        except Exception as exc:
            print("[ABORT] ensure_symbol falhou:", exc)
            return 3

    maybe_send_test_order(api, cfg)
    print(f"[RUN] Lara 3.3.1-XGB | symbol={cfg.get('symbol')} | cfg={cfg_path}")
    print(f"[RUN] version_tag={cfg.get('version_tag')} | simulacao={cfg.get('simulacao')} | permitir_execucao_real={cfg.get('permitir_execucao_real', False)}")

    if hasattr(api, "run_loop"):
        api.run_loop(env, cfg)
        return 0

    print("[ABORT] Falta run_loop no motor_lara.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())

