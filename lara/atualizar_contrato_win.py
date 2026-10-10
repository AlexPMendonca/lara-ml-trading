# -*- coding: utf-8 -*-
"""
atualizar_contrato_win.py

Script para fazer a troca automática do contrato do mini índice (WIN)
no arquivo config_lara.json.

Nova regra:
- Considera rolagem ANTECIPADA: troca para o próximo contrato
  3 DIAS ÚTEIS antes do vencimento do contrato atual.
"""

from __future__ import annotations
import datetime as dt
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config_lara.json"

# Meses pares de vencimento do índice
CONTRACT_MONTHS = [2, 4, 6, 8, 10, 12]
MONTH_TO_CODE = {
    2:  "G",  # Fevereiro
    4:  "J",  # Abril
    6:  "M",  # Junho
    8:  "Q",  # Agosto
    10: "V",  # Outubro
    12: "Z",  # Dezembro
}

DIAS_ROLAGEM_UTEIS = 2  # rolar 3 dias ÚTEIS antes do vencimento


def subtrair_dias_uteis(data: dt.date, dias: int) -> dt.date:
    """
    Subtrai 'dias' dias ÚTEIS (segunda a sexta) da data informada.
    Se dias = 3, volta 3 dias úteis ignorando sábado/domingo.
    """
    if dias < 0:
        raise ValueError("dias deve ser não-negativo")
    atual = data
    restantes = dias
    while restantes > 0:
        atual -= dt.timedelta(days=1)
        if atual.weekday() < 5:  # 0=segunda .. 4=sexta
            restantes -= 1
    return atual


def calcular_vencimento_mes(ano: int, mes: int) -> dt.date:
    """
    Retorna a data de vencimento do mini índice para (ano, mes):
    quarta-feira mais próxima do dia 15.
    """
    base = dt.date(ano, mes, 15)
    # 0 = segunda, 1 = terça, 2 = quarta, ...
    weekday = base.weekday()

    # quarta = 2
    delta_prev = (weekday - 2) % 7
    prev_wed = base - dt.timedelta(days=delta_prev)
    next_wed = prev_wed + dt.timedelta(days=7)

    # Escolhe a quarta mais próxima do dia 15
    if abs((prev_wed - base).days) <= abs((next_wed - base).days):
        venc = prev_wed
    else:
        venc = next_wed

    if venc.weekday() != 2:
        raise RuntimeError(f"Data de vencimento não é quarta-feira: {venc}")

    return venc


def _montar_tabela_contratos(data_ref: dt.date) -> list[tuple[dt.date, str]]:
    """
    Monta uma lista ordenada [(data_venc, symbol), ...] cobrindo
    o ano anterior, o ano atual e o próximo ano em relação a data_ref.
    """
    contratos: list[tuple[dt.date, str]] = []
    ano_ref = data_ref.year
    for ano in (ano_ref - 1, ano_ref, ano_ref + 1):
        for mes in CONTRACT_MONTHS:
            venc = calcular_vencimento_mes(ano, mes)
            letra = MONTH_TO_CODE[mes]
            symbol = f"WIN{letra}{ano % 100:02d}"
            contratos.append((venc, symbol))

    # Ordena por data de vencimento
    contratos.sort(key=lambda x: x[0])
    return contratos


def calcular_contrato_vigente(data_ref: dt.date | None = None) -> tuple[str, dt.date]:
    """
    Dada uma data de referência (hoje, se None),
    retorna (symbol_WIN, data_vencimento) do contrato a ser negociado
    considerando rolagem ANTECIPADA de DIAS_ROLAGEM_UTEIS antes do vencimento.

    Regra:
      - Para cada contrato com vencimento E_k, definimos data de rolagem R_k = E_k - N dias úteis.
      - Em qualquer dia d, usamos o PRIMEIRO contrato para o qual d < R_k.
      - Se d >= R_k para todos, usamos o último da lista.
    """
    if data_ref is None:
        data_ref = dt.date.today()

    contratos = _montar_tabela_contratos(data_ref)
    # Lista de datas de rolagem (quando DEIXAMOS de usar esse contrato e passamos ao próximo)
    datas_rolagem = [subtrair_dias_uteis(venc, DIAS_ROLAGEM_UTEIS) for venc, _ in contratos]

    for (venc, symbol), data_roll in zip(contratos, datas_rolagem):
        if data_ref < data_roll:
            return symbol, venc

    # Se passou de todas as datas de rolagem, ficamos com o último
    venc_ult, symbol_ult = contratos[-1]
    return symbol_ult, venc_ult


def atualizar_contrato_cfg(cfg_path: Path | None = None) -> None:
    """
    Lê config_lara.json, calcula o contrato WIN vigente
    (com rolagem antecipada) e atualiza o campo 'symbol' se necessário.
    """
    if cfg_path is None:
        cfg_path = CFG_PATH

    if not cfg_path.exists():
        print(f"[ROLLOVER] Arquivo de config não encontrado: {cfg_path}")
        return

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ROLLOVER] Erro ao ler config {cfg_path}: {e}")
        return

    symbol_atual = cfg.get("symbol")
    hoje = dt.date.today()
    novo_symbol, vencimento = calcular_contrato_vigente(hoje)

    if symbol_atual == novo_symbol:
        print(f"[ROLLOVER] {hoje} -> Contrato já está atualizado: {novo_symbol} (venc.: {vencimento})")
        return

    print(f"[ROLLOVER] {hoje} -> Atualizando contrato do índice:")
    print(f"           Antigo: {symbol_atual}")
    print(f"           Novo:   {novo_symbol} (venc.: {vencimento})")

    cfg["symbol"] = novo_symbol

    try:
        cfg_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"[ROLLOVER] Config atualizada com sucesso em {cfg_path}")
    except Exception as e:
        print(f"[ROLLOVER] Erro ao salvar config: {e}")


if __name__ == "__main__":
    # Permite rodar manualmente:
    #   python atualizar_contrato_win.py
    atualizar_contrato_cfg()

