# -*- coding: utf-8 -*-
"""
teste_ordem.py

Envia UMA ordem de teste a mercado (1 contrato) e fecha alguns segundos depois,
usando as mesmas funções de envio do motor_lara.

Travas de segurança:
- Só roda em conta DEMO (aborta em conta real).
- Aborta se já existir posição aberta no símbolo.

Uso (dentro da pasta lara):
    python teste_ordem.py           -> compra e fecha
    python teste_ordem.py venda     -> vende e fecha
"""
import json
import sys
import time
from pathlib import Path

import MetaTrader5 as mt5
import motor_lara as impl

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "config_lara.json").read_text(encoding="utf-8"))

SYMBOL = str(cfg["symbol"]).upper()
VOLUME = 1.0
MAGIC = 99999          # diferente do robô, para identificar no histórico
SL_PTS = float(cfg.get("sl_points", 150))
TP_PTS = float(cfg.get("tp_points", 150))
ESPERA_S = 5


def main() -> int:
    lado = sys.argv[1].lower() if len(sys.argv) > 1 else "compra"
    if lado not in ("compra", "venda"):
        print("[ERRO] Use: python teste_ordem.py [compra|venda]")
        return 1

    if not mt5.initialize():
        print("[ERRO] initialize falhou:", mt5.last_error())
        return 1

    conta = mt5.account_info()
    if conta is None:
        print("[ERRO] Conta não conectada.")
        return 1
    if conta.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        print(f"[ABORTADO] Conta {conta.login} NÃO é demo. Teste cancelado.")
        return 1

    if not mt5.symbol_select(SYMBOL, True):
        print(f"[ERRO] symbol_select falhou: {SYMBOL}")
        return 1

    if mt5.positions_get(symbol=SYMBOL):
        print(f"[ABORTADO] Já existe posição aberta em {SYMBOL}. Feche antes de testar.")
        return 1

    tipo = mt5.ORDER_TYPE_BUY if lado == "compra" else mt5.ORDER_TYPE_SELL
    print(f"[TESTE] Conta demo {conta.login} | {lado.upper()} {VOLUME} {SYMBOL} | SL={SL_PTS} TP={TP_PTS}")

    ok, ret, ordem = impl.send_order_market(SYMBOL, VOLUME, tipo, MAGIC, SL_PTS, TP_PTS)
    print(f"[ENVIO] ok={ok} retcode={ret} ordem={ordem}")
    if not ok:
        print("[ERRO] Ordem não executada. Veja o retcode acima.")
        return 1

    pos = mt5.positions_get(symbol=SYMBOL)
    if pos:
        p = pos[0]
        print(f"[POSIÇÃO] ticket={p.ticket} preço={p.price_open} sl={p.sl} tp={p.tp}")

    print(f"[TESTE] Aguardando {ESPERA_S}s para fechar...")
    time.sleep(ESPERA_S)

    ok2, ret2 = impl.close_position_market(SYMBOL)
    print(f"[FECHAMENTO] ok={ok2} retcode={ret2}")

    if mt5.positions_get(symbol=SYMBOL):
        print("[ATENÇÃO] Ainda há posição aberta! Feche manualmente no MT5.")
        return 1

    print("[OK] Teste concluído: ordem aberta e fechada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
