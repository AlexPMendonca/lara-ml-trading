# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "config_lara.json"


def fail(msg: str) -> int:
    print(f"[ERRO] {msg}")
    return 1


def main() -> int:
    try:
        import MetaTrader5 as mt5
    except Exception as exc:
        return fail(f"Biblioteca MetaTrader5 indisponível: {exc}")

    import motor_lara as impl

    if not CFG.exists():
        return fail(f"Config ausente: {CFG}")
    cfg = json.loads(CFG.read_text(encoding="utf-8"))

    if not mt5.initialize():
        return fail(f"initialize() falhou: {mt5.last_error()}")

    ai = mt5.account_info()
    ti = mt5.terminal_info() if hasattr(mt5, "terminal_info") else None
    if ai is None:
        return fail("Conta não conectada no MT5.")

    symbol = str(cfg.get("symbol", "")).strip().upper()
    if not symbol:
        return fail("Config sem symbol.")

    if not mt5.symbol_select(symbol, True):
        return fail(f"symbol_select falhou para {symbol}")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return fail(f"Sem tick para {symbol}")

    print(f"[OK] Conta={getattr(ai, 'login', '?')} trade_allowed={getattr(ai, 'trade_allowed', '?')} trade_mode={getattr(ai, 'trade_mode', '?')}")
    if ti is not None:
        print(f"[OK] Terminal connected={getattr(ti, 'connected', '?')} trade_allowed={getattr(ti, 'trade_allowed', '?')}")
    print(f"[OK] Symbol={symbol} bid={getattr(tick, 'bid', '?')} ask={getattr(tick, 'ask', '?')}")

    checks = impl.dry_run_order_checks(
        symbol=symbol,
        volume=float(cfg.get("lote", 1.0)),
        magic=int(cfg.get("magic", 33175)),
        sl_points=float(cfg.get("sl_points", 150)),
        tp_points=float(cfg.get("tp_points", 150)),
    )

    if checks.get("error"):
        return fail(str(checks["error"]))

    total_ok = 0
    for side in checks.get("checks", []):
        print(f"\n[CHECK] {side.get('side')}")
        for row in side.get("results", []):
            status = "OK" if row.get("ok") else "FALHA"
            print(
                f"  filling={row.get('type_filling')} status={status} retcode={row.get('retcode')} ({row.get('retcode_name')}) "
                f"vol={row.get('volume')} price={row.get('price')} sl={row.get('sl')} tp={row.get('tp')}"
            )
            if row.get("ok"):
                total_ok += 1

    if total_ok <= 0:
        return fail("Nenhum order_check passou. Não vire para demo assim.")

    print("\n[OK] Pré-flight demo concluído. Existe ao menos um filling válido no order_check.")
    print("[OBS] Este script não envia ordem; valida a estrutura de envio antes da demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

