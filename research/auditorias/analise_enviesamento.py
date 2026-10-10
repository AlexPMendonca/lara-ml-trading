# -*- coding: utf-8 -*-
"""
analise_enviesamento.py

Objetivo (atualizado para SHADOW):
- Interpretar 'pc' como probabilidade de acerto / confiança (p_acerto) do modelo.
- Direção (BUY/SELL) é inferida pelo 'regime_mercado' (uptrend/downtrend), quando disponível.
- Em SHADOW, pode não existir log_operacoes_v3_{data}.csv; nesse caso, analisa apenas IA.

Uso:
    python analise_enviesamento.py <pasta_arquivo_ABC> <data YYYY-MM-DD>

Exemplo:
    python analise_enviesamento.py "C:/.../arquivo_ABC" 2025-12-12
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np


def _is_blank(x) -> bool:
    try:
        return (x is None) or (str(x).strip() == "")
    except Exception:
        return True


def analisar_dia(base_dir: str, data: str, thr_simetrico: float = 0.55) -> None:
    base = Path(base_dir)

    arq_ia = base / f"log_ia_exec_{data}.csv"
    arq_ops = base / f"log_operacoes_v3_{data}.csv"

    if not arq_ia.exists():
        print(f"[ERRO] Arquivo não encontrado: {arq_ia}")
        return

    df_ia = pd.read_csv(arq_ia)

    df_ops = None
    if not arq_ops.exists():
        print(f"[AVISO] {arq_ops.name} não encontrado (provável SHADOW). Vou analisar apenas a IA.")
    else:
        try:
            df_ops = pd.read_csv(arq_ops)
        except Exception:
            df_ops = None
            print(f"[AVISO] Falha ao ler {arq_ops.name}. Vou analisar apenas a IA.")

    print("=" * 80)
    print(f"ANÁLISE (SHADOW/REAL) - DIA {data}")
    print("=" * 80)

    n = len(df_ia)
    print(f"\n[IA] Linhas em log_ia_exec: {n}")

    # segurança: garante colunas esperadas
    if "pc" not in df_ia.columns:
        print("[ERRO] Falta coluna 'pc' em log_ia_exec (esperado p_acerto/confiança).")
        return

    p_acerto = df_ia["pc"].astype(float)

    # Limiar preferencial (ordem):
    # 1) coluna limiar_confianca (se existir no log)
    # 2) max(limiar_compra_ajustado, limiar_venda_ajustado) (se existirem)
    # 3) thr_simetrico
    thr_col = None
    if "limiar_confianca" in df_ia.columns:
        try:
            thr_col = df_ia["limiar_confianca"].astype(float)
        except Exception:
            thr_col = None

    thr_real = None
    if {"limiar_compra_ajustado", "limiar_venda_ajustado"}.issubset(df_ia.columns):
        try:
            lc = df_ia["limiar_compra_ajustado"].astype(float)
            lv = df_ia["limiar_venda_ajustado"].astype(float)
            thr_real = np.maximum(lc, lv)
        except Exception:
            thr_real = None

    # escolhe o limiar efetivo para análise de gatilhos
    if thr_col is not None:
        thr_eff = thr_col
        thr_desc = "limiar_confianca (do log)"
    elif thr_real is not None:
        thr_eff = thr_real
        thr_desc = "max(limiar_compra_ajustado, limiar_venda_ajustado)"
    else:
        thr_eff = float(thr_simetrico)
        thr_desc = f"limiar simétrico {thr_simetrico:.2f}"

    # bloqueio: considera não-bloqueado se NaN ou string vazia
    if "bloqueado_por" in df_ia.columns:
        b = df_ia["bloqueado_por"]
        nb_mask = b.isna() | (b.astype(str).str.strip() == "")
        bloqueios = b[~nb_mask].astype(str).str.strip()
    else:
        nb_mask = np.ones(n, dtype=bool)
        bloqueios = pd.Series([], dtype=str)

    df_nb = df_ia[nb_mask].copy()
    p_nb = df_nb["pc"].astype(float)

    # direção pelo regime (se existir)
    if "regime_mercado" in df_ia.columns:
        reg = df_ia["regime_mercado"].astype(str).str.lower()
    else:
        reg = pd.Series(["side"] * n)

    direcao = np.where(reg == "uptrend", "BUY", np.where(reg == "downtrend", "SELL", "SIDE"))

    if isinstance(thr_eff, (int, float)):
        thr_nb = float(thr_eff)
        thr_all = float(thr_eff)
        thr_nb_arr = float(thr_eff)
        thr_vals_nb = thr_nb_arr
        thr_vals_all = thr_all
    else:
        # vetor
        thr_vals_all = np.array(thr_eff, dtype=float)
        thr_vals_nb = thr_vals_all[nb_mask.values] if hasattr(nb_mask, "values") else thr_vals_all[nb_mask]

    # gatilhos (em shadow): p_acerto >= thr_eff e direção != SIDE e não bloqueado
    dir_nb = direcao[nb_mask.values] if hasattr(nb_mask, "values") else direcao[nb_mask]
    gatilhos = (p_nb.values >= thr_vals_nb) & (dir_nb != "SIDE")

    print("\n[IA] Pc interpretado como p_acerto (confiança):")
    print(f"  média={p_acerto.mean():.3f} | mediana={p_acerto.median():.3f} | p10={p_acerto.quantile(0.1):.3f} | p90={p_acerto.quantile(0.9):.3f}")
    print(f"  min={p_acerto.min():.3f} | max={p_acerto.max():.3f}")

    print(f"\n[IA] Limiar usado para 'gatilho' nesta análise: {thr_desc}")
    if isinstance(thr_eff, (int, float)):
        print(f"  valor={float(thr_eff):.3f}")
    else:
        print(f"  média={np.nanmean(thr_vals_all):.3f} | min={np.nanmin(thr_vals_all):.3f} | max={np.nanmax(thr_vals_all):.3f}")

    n_nb = len(df_nb)
    n_g = int(gatilhos.sum())
    print(f"\n[IA] Não bloqueados: {n_nb} ({(100.0*n_nb/n if n else 0.0):.1f}%)")
    print(f"[IA] Gatilhos (p_acerto>=limiar e direção!=SIDE): {n_g} ({(100.0*n_g/n_nb if n_nb else 0.0):.1f}% dos não bloqueados)")

    # gatilhos por direção
    if n_nb > 0:
        g_buy = int(((dir_nb == "BUY") & gatilhos).sum())
        g_sell = int(((dir_nb == "SELL") & gatilhos).sum())
        print(f"  Gatilhos BUY:  {g_buy}")
        print(f"  Gatilhos SELL: {g_sell}")

    # Regimes
    if "regime_mercado" in df_ia.columns:
        print("\n[IA] Regime de mercado (distribuição):")
        print(df_ia["regime_mercado"].value_counts(normalize=True).mul(100).round(1).to_string())

    # Bloqueios
    if len(bloqueios) > 0:
        print("\n[IA] Bloqueios (top):")
        print(bloqueios.value_counts().head(10).to_string())

    # -------- Operações (se existirem) --------
    if df_ops is None:
        return

    print("\n" + "-" * 80)
    print(f"[TRADES] Analisando {arq_ops.name}")
    print("-" * 80)

    df_tr = df_ops.copy()
    if "tipo" not in df_tr.columns or "lucro_real" not in df_tr.columns:
        print("[AVISO] log_operacoes_v3 não tem colunas 'tipo' e/ou 'lucro_real'.")
        return

    df_tr = df_tr[df_tr["tipo"].isin(["close_BUY", "close_SELL"])]
    df_tr = df_tr[df_tr["lucro_real"].notna()]

    if df_tr.empty:
        print("[TRADES] Nenhuma operação fechada no dia.")
        return

    df_tr["lado"] = np.where(df_tr["tipo"] == "close_BUY", "BUY", "SELL")
    df_tr["acerto"] = (df_tr["lucro_real"] > 0).astype(int)

    resumo = (
        df_tr.groupby("lado")
        .agg(
            n_trades=("lado", "size"),
            lucro_total=("lucro_real", "sum"),
            lucro_medio=("lucro_real", "mean"),
            acuracia=("acerto", "mean"),
        )
        .reset_index()
    )

    print("\n[TRADES] Resumo por lado (BUY x SELL):")
    for _, row in resumo.iterrows():
        lado = row["lado"]
        n_trades = int(row["n_trades"])
        lucro_total = float(row["lucro_total"])
        lucro_medio = float(row["lucro_medio"])
        acur = float(row["acuracia"]) * 100
        print(
            f"  {lado:4s} | trades = {n_trades:2d} | "
            f"lucro_total = {lucro_total:7.1f} | "
            f"lucro_medio = {lucro_medio:6.2f} | "
            f"acurácia = {acur:5.1f}%"
        )

    lucro_dia = float(df_tr["lucro_real"].sum())
    acur_dia = float(df_tr["acerto"].mean()) * 100
    print(f"\n[TRADES] Dia {data}: lucro_total = {lucro_dia:.1f} | acurácia global = {acur_dia:.1f}%")

    n_buy_trades = int((df_tr["lado"] == "BUY").sum())
    n_sell_trades = int((df_tr["lado"] == "SELL").sum())
    if n_buy_trades + n_sell_trades > 0:
        frac_buy = 100.0 * n_buy_trades / (n_buy_trades + n_sell_trades)
        frac_sell = 100.0 * n_sell_trades / (n_buy_trades + n_sell_trades)
        print(
            f"\n[ALERTA] Distribuição de trades no dia:"
            f"\n  BUY  = {n_buy_trades} ({frac_buy:5.1f}%)"
            f"\n  SELL = {n_sell_trades} ({frac_sell:5.1f}%)"
        )
        if frac_sell >= 80.0 or frac_buy >= 80.0:
            print("  -> VIÉS FORTE detectado: um dos lados responde por >= 80% das operações.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso:")
        print("  python analise_enviesamento.py <pasta_arquivo_ABC> <data YYYY-MM-DD>")
        sys.exit(1)

    base_dir = sys.argv[1]
    data = sys.argv[2]
    analisar_dia(base_dir, data)
