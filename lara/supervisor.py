# -*- coding: utf-8 -*-
"""
supervisor.py 

Correções nesta versão:
1) **eod_logfix.py** agora é chamado com flags corretas:
   `--data YYYY-MM-DD --pasta_logs <pasta>` (antes passava a data como argumento posicional).
2) **Pós-EOD**: o supervisor **não reinicia** o runner no mesmo dia; ele **hiberna até o próximo dia útil às 09:15 BRT**.
3) **Janela operacional** usando hora local do sistema (mesmo formato do restante do código). Inicia exatamente às 09:15 em dia útil;
   se iniciar depois das 09:15, começa imediatamente.
"""

import datetime as dt
import time
import subprocess
from pathlib import Path
import json
import sys
import os
import importlib
from atualizar_contrato_win import atualizar_contrato_cfg
 
try:
    import MetaTrader5 as mt5
    
except Exception:
    mt5 = None

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_lara.py"
ETAPA_F_PIPELINE = ROOT / "pipeline_retreino.py"
PREZIP_CHECK = ROOT / "prezip_check.py"
CFG_CANDIDATES = [
    ROOT / "config_lara.json",
]


def now_local():
    return dt.datetime.now()


def is_business_day(d: dt.date) -> bool:
    # Segunda(0) .. Sexta(4); sem calendário de feriados
    return d.weekday() < 5


def next_business_day(d: dt.date) -> dt.date:
    nd = d + dt.timedelta(days=1)
    while not is_business_day(nd):
        nd += dt.timedelta(days=1)
    return nd


def sleep_until(target: dt.datetime):
    """Dorme em blocos até o instante target (hora local)."""
    while True:
        now = now_local()
        delta = (target - now).total_seconds()
        if delta <= 0:
            return
        # dormir em blocos de até 60s para reavaliar condições
        time.sleep(60 if delta > 60 else delta)


def carregar_cfg():
    cfg = {}
    for p in CFG_CANDIDATES:
        if p.exists():
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                pass
    symbol = cfg.get("symbol") or "WINZ25"
    pasta_logs = (cfg.get("paths") or {}).get("pasta_logs") or "arquivo_ABC"
    etapa_f_enabled = bool(cfg.get("etapa_f_auto_run", True))
    etapa_f_n_test_days = int(cfg.get("etapa_f_n_test_days", 3) or 3)
    etapa_f_calibracao = str(cfg.get("etapa_f_calibracao", "nenhuma") or "nenhuma")
    etapa_f_run_prezip = bool(cfg.get("etapa_f_run_prezip", True))
    return {
        "symbol": symbol,
        "pasta_logs": pasta_logs,
        "etapa_f_auto_run": etapa_f_enabled,
        "etapa_f_n_test_days": etapa_f_n_test_days,
        "etapa_f_calibracao": etapa_f_calibracao,
        "etapa_f_run_prezip": etapa_f_run_prezip,
    }


def conectar_mt5():
    if mt5 is None:
        print("[MT5] Biblioteca MetaTrader5 não disponível neste ambiente.")
        return False
    if mt5.initialize():
        return True
    time.sleep(1)
    return mt5.initialize()


def selecionar_simbolo(symbol: str) -> bool:
    if mt5 is None:
        return False
    if not mt5.symbol_select(symbol, True):
        print(f"[MT5] symbol_select falhou: {symbol}")
        return False
    return True


def fechar_todas_as_posicoes(symbol: str) -> None:
    if mt5 is None:
        print("[EOD] mt5=None; não é possível zerar posições.")
        return
    if not conectar_mt5():
        print("[EOD] initialize() falhou; não consegui zerar posições.")
        return
    if not selecionar_simbolo(symbol):
        print("[EOD] symbol_select falhou; abortando zeragem.")
        return

    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        print(f"[EOD] Já estava zerado em {symbol}.")
        return

    print(f"[EOD] Encontradas {len(positions)} posições abertas em {symbol}. Fechando...")
    try:
        impl = importlib.import_module("motor_lara")
    except Exception as exc:
        impl = None
        print(f"[EOD] Aviso: fallback legado de fechamento, import do integration falhou: {exc}")

    while True:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            print(f"[EOD] Zeragem concluída em {symbol}.")
            return
        pos = positions[0]
        ticket = getattr(pos, "ticket", 0)
        vol = getattr(pos, "volume", 0)
        if impl is not None and hasattr(impl, "close_position_market"):
            ok, rc = impl.close_position_market(symbol)
            print(f"[EOD] Fechamento robusto pos={ticket} vol={vol} -> ok={ok} retcode={rc}")
        else:
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            price = float(getattr(tick, 'bid', 0.0) or 0.0) if close_type == mt5.ORDER_TYPE_SELL else float(getattr(tick, 'ask', 0.0) or 0.0)
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": vol,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 30,
                "magic": 33175,
                "comment": "EOD_FORCE_FLAT",
                "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 1),
            }
            res = mt5.order_send(req)
            rc = getattr(res, "retcode", None) if res is not None else None
            print(f"[EOD] Fechamento legado pos={ticket} vol={vol} -> retcode={rc}")
        time.sleep(0.5)


def rodar_logfix_dedup(data_iso: str, pasta_logs: str):
    script = ROOT / "eod_logfix.py"
    if not script.exists():
        print("[EOD] eod_logfix.py não encontrado. Pulando dedup.")
        return
    cmd = [sys.executable, str(script), "--data", data_iso, "--pasta_logs", pasta_logs]
    print("[EOD] Rodando dedup:", " ".join(cmd))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out.stdout:
            print("[EOD] dedup stdout:\n" + out.stdout)
        if out.stderr:
            print("[EOD] dedup stderr:\n" + out.stderr)
    except subprocess.TimeoutExpired:
        print("[EOD] eod_logfix timeout (300s).")
    except Exception as e:
        print(f"[EOD] Erro ao rodar dedup: {e}")


def rodar_etapa_f(data_iso: str, cfg: dict):
    if not cfg.get("etapa_f_auto_run", True):
        print("[ETAPA F] Execução automática desabilitada no config.")
        return
    if not ETAPA_F_PIPELINE.exists():
        print(f"[ETAPA F] Pipeline não encontrado: {ETAPA_F_PIPELINE}")
        return

    base_path = Path(str(cfg.get("pasta_logs", "arquivo_ABC")))
    if not base_path.is_absolute():
        base_path = (ROOT / base_path).resolve()

    cmd = [
        sys.executable,
        str(ETAPA_F_PIPELINE),
        "--base-dir",
        str(base_path),
        "--n-test-days",
        str(cfg.get("etapa_f_n_test_days", 3)),
        "--calibracao",
        str(cfg.get("etapa_f_calibracao", "nenhuma")),
    ]
    print("[ETAPA F] Rodando pipeline automático:", " ".join(cmd))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if out.stdout:
            print("[ETAPA F] stdout:\n" + out.stdout)
        if out.stderr:
            print("[ETAPA F] stderr:\n" + out.stderr)
        if out.returncode != 0:
            print(f"[ETAPA F] Pipeline terminou com erro retcode={out.returncode}")
        else:
            print(f"[ETAPA F] Pipeline concluído para {data_iso}.")
    except subprocess.TimeoutExpired:
        print("[ETAPA F] Timeout (1800s).")
    except Exception as e:
        print(f"[ETAPA F] Erro ao rodar pipeline: {e}")


def rodar_prezip_check(data_iso: str, pasta_logs: str):
    if not PREZIP_CHECK.exists():
        print(f"[PREZIP] Script não encontrado: {PREZIP_CHECK}")
        return
    base_path = Path(pasta_logs)
    if not base_path.is_absolute():
        base_path = (ROOT / base_path).resolve()
    cmd = [sys.executable, str(PREZIP_CHECK), "--day", data_iso, "--base", str(base_path)]
    print("[PREZIP] Rodando checagem automática:", " ".join(cmd))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out.stdout:
            print("[PREZIP] stdout:\n" + out.stdout)
        if out.stderr:
            print("[PREZIP] stderr:\n" + out.stderr)
    except subprocess.TimeoutExpired:
        print("[PREZIP] Timeout (300s).")
    except Exception as e:
        print(f"[PREZIP] Erro ao rodar checagem: {e}")


def wait_for_operating_window():
    """
    Janela operacional:
      - Se fim de semana: dorme até próxima segunda 09:15.
      - Se antes das 09:15: dorme até 09:15.
      - Se após 17:58: dorme até o próximo dia útil 09:15.
      - Entre 09:15 e 17:58: inicia imediatamente.
    """
    while True:
        now = now_local()
        today = now.date()

        # Horário de EOD fixo (17:58)
        eod_1758 = dt.datetime(today.year, today.month, today.day, 17, 58, 0)

        # 1) Fim de semana → próxima segunda 09:15
        if not is_business_day(today):
            nd = next_business_day(today)
            target = dt.datetime(nd.year, nd.month, nd.day, 9, 15, 0)
            print("[HIBERNANDO] Fim de semana. Próximo dia útil:", nd.isoformat(), "às 09:15.")
            sleep_until(target)
            continue

        # 2) Antes das 09:15 → dormir até 09:15 de hoje
        if now.time() < dt.time(9, 15, 0):
            target = dt.datetime(today.year, today.month, today.day, 9, 15, 0)
            print("[HIBERNANDO] Ainda não é 09:15. Vou dormir até", target.isoformat())
            sleep_until(target)
            continue

        # 3) Já passou do EOD (17:58) → próximo dia útil 09:15
        if now >= eod_1758:
            nd = next_business_day(today)
            target = dt.datetime(nd.year, nd.month, nd.day, 9, 15, 0)
            print("[HIBERNANDO] Passou do EOD. Vou dormir até", target.isoformat())
            total_sec = int(max(0, (target - now_local()).total_seconds()))
            hh = total_sec // 3600
            mm = (total_sec % 3600) // 60
            print(f"[HIBERNANDO] {hh:02d}:{mm:02d} horas até {target.isoformat()}. Dormindo...")
            sleep_until(target)
            continue

        # 4) Dentro da janela (>=09:15 e <17:58) → inicia o robô
        print("[ATIVO] Dia útil e dentro da janela (>=09:15 e <17:58). Iniciando robô.")
        return


def rodar_pregao_com_eod():
    """
    Executa o runner principal durante o pregão, força zeragem às 17:58
    e roda o eod_logfix para o dia corrente.
    """
    # 1) Antes de tudo, garante que o contrato WIN está atualizado
    try:
        atualizar_contrato_cfg()  # usa config_lara.json da própria pasta
    except Exception as e:
        print(f"[ROLLOVER] Erro ao atualizar contrato do índice: {e}")

    # 2) Carrega a configuração já com o symbol possivelmente atualizado
    cfg = carregar_cfg()
    symbol = cfg["symbol"]
    pasta_logs = cfg["pasta_logs"]

    if not RUNNER.exists():
        print(f"[ERRO] Runner não encontrado: {RUNNER}")
        time.sleep(60)
        return

    env = dict(os.environ)
    env["LARA_SUPERVISOR"] = "1"

    cmd = [sys.executable, str(RUNNER)]
    print("[RUNNER] Iniciando:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)

    try:
        while True:
            time.sleep(5)
            now = now_local()
            eod_cut = now.replace(hour=17, minute=58, second=0, microsecond=0)

            if now >= eod_cut:
                print("[EOD] 17:58 alcançado. Forçando zeragem, encerrando runner e rodando pós-pregão...")
                try:
                    fechar_todas_as_posicoes(symbol)
                except Exception as e:
                    print(f"[EOD] Zeragem falhou: {e}")

                print("[EOD] Encerrando runner...")
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("[EOD] Runner não terminou em 30s. Enviando kill.")
                    proc.kill()
                    proc.wait(timeout=10)

                data_iso = now.date().isoformat()
                rodar_logfix_dedup(data_iso, pasta_logs)
                rodar_etapa_f(data_iso, cfg)
                if cfg.get("etapa_f_run_prezip", True):
                    rodar_prezip_check(data_iso, pasta_logs)

                print("[EOD] Sessão encerrada.")
                break

            ret = proc.poll()
            if ret is not None:
                print(f"[RUNNER] Finalizado prematuramente (retcode={ret}). Voltando à hibernação.")
                break
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


def main():
    print("[SUPERVISOR] Iniciado. Modo 24/7 com hibernação + EOD automático + Etapa F automática.")
    while True:
        # só retorna se está dentro de 09:15–17:58 de um dia útil
        wait_for_operating_window()
        # quando volta (EOD ou runner caiu), o loop reavalia janela
        rodar_pregao_com_eod()
        # ao sair do pregão, a próxima iteração cairá na ramificação "Passou do EOD"
        # e irá hibernar até o próximo dia útil 09:15 (hora local)


if __name__ == "__main__":
    main()

