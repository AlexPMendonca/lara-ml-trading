import importlib.util
import itertools
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_DIR / "lara"
BASE_SCRIPT = Path(__file__).resolve().parent / "bateria_cenarios.py"
OUT_DIR = REPO_DIR / "reports" / "backtest" / "otimizacao_combinacoes_lara_10d"

TOP_N_OPS = 30


def carregar_base():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Arquivo base não encontrado: {BASE_SCRIPT}\n"
            "Rode primeiro o script bateria_cenarios.py"
        )

    spec = importlib.util.spec_from_file_location("bt", BASE_SCRIPT)
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    return bt


def nome_pc_pv(pc, pv):
    return f"PC{int(round(pc*100)):03d}_PV{int(round(pv*100)):03d}"


def add_scenario(cenarios, name, **kwargs):
    cfg = {
        "name": name,
        "pc": 0.58,
        "pv": 0.58,
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


def merge_mods(base_cfg, mods):
    cfg = dict(base_cfg)
    name_parts = [cfg["name"]]

    for mod in mods:
        name_parts.append(mod["label"])
        for k, v in mod["params"].items():
            cfg[k] = v

    cfg["name"] = "__".join(name_parts)
    return cfg


def dedup_cenarios(cenarios):
    vistos = set()
    saida = []

    for c in cenarios:
        chave = (
            c.get("pc"),
            c.get("pv"),
            c.get("gap_min"),
            c.get("loss_cooldown_bars"),
            c.get("max_daily_loss"),
            c.get("max_consecutive_losses"),
            c.get("skip_open_min"),
            c.get("skip_final_min"),
            c.get("time_start"),
            c.get("time_end"),
            c.get("trend_mode"),
            c.get("ema_curta"),
            c.get("ema_longa"),
            c.get("trend_zone"),
            c.get("vol_mode"),
            c.get("vol_factor"),
            c.get("vol_cooldown_bars"),
        )

        if chave in vistos:
            continue

        vistos.add(chave)
        saida.append(c)

    return saida


def montar_cenarios():
    cenarios = []

    # Pares prováveis de PC/PV.
    # Inclui a base, variações conservadoras e variações levemente agressivas.
    core_pairs = [
        (0.58, 0.58),
        (0.58, 0.60),
        (0.58, 0.62),
        (0.58, 0.65),
        (0.58, 0.70),

        (0.60, 0.58),
        (0.60, 0.60),
        (0.60, 0.62),
        (0.60, 0.65),

        (0.62, 0.58),
        (0.62, 0.60),
        (0.62, 0.62),
        (0.62, 0.65),

        (0.65, 0.58),
        (0.65, 0.60),
        (0.65, 0.62),

        (0.70, 0.58),
        (0.70, 0.60),

        # What-if mais flexível, para ver se existe ganho oculto.
        (0.55, 0.55),
        (0.55, 0.58),
        (0.58, 0.55),
    ]

    base_cfgs = []
    for pc, pv in core_pairs:
        label = nome_pc_pv(pc, pv)
        cfg = {
            "name": label,
            "pc": pc,
            "pv": pv,
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
        base_cfgs.append(cfg)
        cenarios.append(cfg)

    # Garante nome exato da base para comparação.
    cenarios[0]["name"] = "BASE_PC058_PV058"

    gap_mods = [
        {"family": "gap", "label": "GAP003", "params": {"gap_min": 0.03}},
        {"family": "gap", "label": "GAP005", "params": {"gap_min": 0.05}},
        {"family": "gap", "label": "GAP008", "params": {"gap_min": 0.08}},
    ]

    loss_cd_mods = [
        {"family": "loss_cd", "label": "LOSSCD1", "params": {"loss_cooldown_bars": 1}},
        {"family": "loss_cd", "label": "LOSSCD2", "params": {"loss_cooldown_bars": 2}},
        {"family": "loss_cd", "label": "LOSSCD3", "params": {"loss_cooldown_bars": 3}},
        {"family": "loss_cd", "label": "LOSSCD5", "params": {"loss_cooldown_bars": 5}},
    ]

    risk_mods = [
        {"family": "risk", "label": "MAXLOSS300", "params": {"max_daily_loss": 300}},
        {"family": "risk", "label": "MAXLOSS450", "params": {"max_daily_loss": 450}},
        {"family": "risk", "label": "MAXLOSS600", "params": {"max_daily_loss": 600}},
        {"family": "risk", "label": "MAX3STOPS", "params": {"max_consecutive_losses": 3}},
        {"family": "risk", "label": "MAX4STOPS", "params": {"max_consecutive_losses": 4}},
    ]

    horario_mods = [
        {"family": "horario", "label": "NOOPEN15", "params": {"skip_open_min": 15}},
        {"family": "horario", "label": "NOOPEN30", "params": {"skip_open_min": 30}},
        {"family": "horario", "label": "NOFINAL30", "params": {"skip_final_min": 30}},
        {"family": "horario", "label": "J0930_1700", "params": {"time_start": "09:30", "time_end": "17:00"}},
        {"family": "horario", "label": "J1000_1700", "params": {"time_start": "10:00", "time_end": "17:00"}},
        {"family": "horario", "label": "J1000_1600", "params": {"time_start": "10:00", "time_end": "16:00"}},
    ]

    trend_mods = [
        {"family": "trend", "label": "TSELL_EMA20_100", "params": {"trend_mode": "sell_only", "ema_curta": 20, "ema_longa": 100, "trend_zone": 0.0}},
        {"family": "trend", "label": "TSELL_EMA20_100_Z005", "params": {"trend_mode": "sell_only", "ema_curta": 20, "ema_longa": 100, "trend_zone": 0.0005}},
        {"family": "trend", "label": "TSELL_EMA20_100_Z010", "params": {"trend_mode": "sell_only", "ema_curta": 20, "ema_longa": 100, "trend_zone": 0.0010}},
        {"family": "trend", "label": "TBOTH_EMA20_100_Z005", "params": {"trend_mode": "both", "ema_curta": 20, "ema_longa": 100, "trend_zone": 0.0005}},
        {"family": "trend", "label": "TSELL_EMA9_21", "params": {"trend_mode": "sell_only", "ema_curta": 9, "ema_longa": 21, "trend_zone": 0.0}},
    ]

    vol_mods = [
        {"family": "vol", "label": "VOLBLOCK2X", "params": {"vol_mode": "block", "vol_factor": 2.0}},
        {"family": "vol", "label": "VOLBLOCK25X", "params": {"vol_mode": "block", "vol_factor": 2.5}},
        {"family": "vol", "label": "VOLCD2X3", "params": {"vol_mode": "cooldown", "vol_factor": 2.0, "vol_cooldown_bars": 3}},
        {"family": "vol", "label": "VOLCD2X5", "params": {"vol_mode": "cooldown", "vol_factor": 2.0, "vol_cooldown_bars": 5}},
        {"family": "vol", "label": "VOLCD25X3", "params": {"vol_mode": "cooldown", "vol_factor": 2.5, "vol_cooldown_bars": 3}},
    ]

    single_mods = gap_mods + loss_cd_mods + risk_mods + horario_mods + trend_mods + vol_mods

    # 1) PC/PV + cada filtro isolado.
    for base in base_cfgs:
        for mod in single_mods:
            cenarios.append(merge_mods(base, [mod]))

    # 2) PC/PV + cooldown após loss + outro filtro.
    # Esta é a família mais promissora, porque LOSSCD2 já apareceu bem.
    outros_para_loss = gap_mods + risk_mods + horario_mods + trend_mods + vol_mods
    for base in base_cfgs:
        for loss_mod in loss_cd_mods:
            for mod in outros_para_loss:
                if mod["family"] == loss_mod["family"]:
                    continue
                cenarios.append(merge_mods(base, [loss_mod, mod]))

    # 3) PC/PV + GAP + horário.
    for base in base_cfgs:
        for gap_mod in gap_mods:
            for hmod in horario_mods:
                cenarios.append(merge_mods(base, [gap_mod, hmod]))

    # 4) PC/PV + limite de risco + horário.
    for base in base_cfgs:
        for rmod in risk_mods:
            for hmod in horario_mods:
                cenarios.append(merge_mods(base, [rmod, hmod]))

    # 5) PC/PV + GAP + cooldown + horário: combinações fortes, mas ainda plausíveis.
    # Para não explodir, usa GAP003/GAP005 e LOSSCD2/LOSSCD3.
    gap_core = [gap_mods[0], gap_mods[1]]
    loss_core = [loss_cd_mods[1], loss_cd_mods[2]]
    horario_core = [horario_mods[0], horario_mods[2], horario_mods[3], horario_mods[4]]
    for base in base_cfgs:
        for gap_mod in gap_core:
            for loss_mod in loss_core:
                for hmod in horario_core:
                    cenarios.append(merge_mods(base, [gap_mod, loss_mod, hmod]))

    # 6) Combos específicos que queremos garantir.
    combos_manuais = [
        {"name": "CHALLENGER_PV060_LOSSCD2", "pc": 0.58, "pv": 0.60, "loss_cooldown_bars": 2},
        {"name": "CHALLENGER_PV060_LOSSCD2_MAXLOSS450", "pc": 0.58, "pv": 0.60, "loss_cooldown_bars": 2, "max_daily_loss": 450},
        {"name": "CHALLENGER_PV060_LOSSCD2_NOOPEN15", "pc": 0.58, "pv": 0.60, "loss_cooldown_bars": 2, "skip_open_min": 15},
        {"name": "CHALLENGER_PV060_LOSSCD2_NOFINAL30", "pc": 0.58, "pv": 0.60, "loss_cooldown_bars": 2, "skip_final_min": 30},
        {"name": "CHALLENGER_PV060_LOSSCD2_VOLCD2X3", "pc": 0.58, "pv": 0.60, "loss_cooldown_bars": 2, "vol_mode": "cooldown", "vol_factor": 2.0, "vol_cooldown_bars": 3},
        {"name": "CHALLENGER_PV060_LOSSCD2_GAP003", "pc": 0.58, "pv": 0.60, "loss_cooldown_bars": 2, "gap_min": 0.03},
        {"name": "CHALLENGER_PV060_LOSSCD2_GAP005", "pc": 0.58, "pv": 0.60, "loss_cooldown_bars": 2, "gap_min": 0.05},
        {"name": "CHALLENGER_PV062_LOSSCD2", "pc": 0.58, "pv": 0.62, "loss_cooldown_bars": 2},
        {"name": "CHALLENGER_PC060_PV060_LOSSCD2", "pc": 0.60, "pv": 0.60, "loss_cooldown_bars": 2},
        {"name": "CHALLENGER_PC060_PV062_LOSSCD2", "pc": 0.60, "pv": 0.62, "loss_cooldown_bars": 2},
    ]

    for c in combos_manuais:
        add_scenario(cenarios, **c)

    cenarios = dedup_cenarios(cenarios)

    return cenarios


def resumir_ops(bt, ops, cenario, dia):
    return bt.resumir_ops(ops, cenario, dia=dia)


def classificar(row, base):
    resultado = float(row["resultado_pts"])
    pf = float(row["profit_factor"])
    dd = float(row["max_drawdown_pts"])
    ops = int(row["operacoes"])

    base_resultado = float(base["resultado_pts"])
    base_pf = float(base["profit_factor"])
    base_dd = float(base["max_drawdown_pts"])

    if resultado > 0 and pf > 1.0 and dd > base_dd and ops >= 80:
        return "FORTE"

    if resultado > base_resultado and pf > base_pf and dd > base_dd:
        return "MELHOROU_BASE"

    if resultado > base_resultado and dd > base_dd:
        return "MELHOROU_RISCO"

    if resultado > base_resultado:
        return "MELHOROU_RESULTADO"

    return "NAO_APROVADO"


def montar_ranking(df_total, df_dia):
    base = df_total[df_total["cenario"] == "BASE_PC058_PV058"].iloc[0].to_dict()

    df = df_total.copy()

    dias_pos = (
        df_dia
        .assign(dia_positivo=lambda x: x["resultado_pts"] > 0)
        .groupby("cenario")["dia_positivo"]
        .sum()
        .reset_index()
        .rename(columns={"dia_positivo": "dias_positivos"})
    )

    dias_testados = (
        df_dia
        .groupby("cenario")["dia"]
        .nunique()
        .reset_index()
        .rename(columns={"dia": "dias_testados"})
    )

    df = df.merge(dias_pos, on="cenario", how="left")
    df = df.merge(dias_testados, on="cenario", how="left")

    df["dias_positivos"] = df["dias_positivos"].fillna(0).astype(int)
    df["dias_testados"] = df["dias_testados"].fillna(0).astype(int)

    df["delta_resultado_vs_base"] = df["resultado_pts"] - float(base["resultado_pts"])
    df["delta_pf_vs_base"] = df["profit_factor"] - float(base["profit_factor"])
    df["delta_drawdown_vs_base"] = df["max_drawdown_pts"] - float(base["max_drawdown_pts"])
    df["delta_ops_vs_base"] = df["operacoes"] - int(base["operacoes"])

    df["classificacao"] = df.apply(lambda r: classificar(r, base), axis=1)

    # Score mais robusto:
    # resultado positivo pesa mais,
    # drawdown melhor pesa,
    # PF pesa,
    # dias positivos pesam,
    # mas não força cenários com operações muito baixas.
    df["score"] = (
        df["resultado_pts"].astype(float)
        + df["profit_factor"].replace(999, 5).astype(float) * 500
        + df["max_drawdown_pts"].astype(float) * 0.35
        + df["dias_positivos"].astype(float) * 250
        + np.minimum(df["operacoes"].astype(float), 500) * 0.05
    )

    df = df.sort_values(
        ["classificacao", "score", "resultado_pts", "profit_factor", "max_drawdown_pts"],
        ascending=[True, False, False, False, False]
    )

    # Reordena para FORTE primeiro.
    ordem = {
        "FORTE": 1,
        "MELHOROU_BASE": 2,
        "MELHOROU_RISCO": 3,
        "MELHOROU_RESULTADO": 4,
        "NAO_APROVADO": 5,
    }
    df["_ordem"] = df["classificacao"].map(ordem).fillna(9)
    df = df.sort_values(
        ["_ordem", "score", "resultado_pts", "profit_factor", "max_drawdown_pts"],
        ascending=[True, False, False, False, False]
    ).drop(columns=["_ordem"]).reset_index(drop=True)

    df.insert(0, "rank", np.arange(1, len(df) + 1))

    return df


def main():
    bt = carregar_base()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dias = bt.achar_ultimos_10_dias()
    if not dias:
        raise SystemExit(
            f"[ERRO] Nenhum log diário encontrado em {bt.LOG_DIR}.\n"
            "Esses logs são gerados pelo motor durante o pregão e não são versionados."
        )
    cenarios = montar_cenarios()

    print("=" * 110)
    print("OTIMIZAÇÃO DE COMBINAÇÕES PROVÁVEIS - LARA - 10 PREGÕES")
    print("=" * 110)
    print(f"Projeto: {PROJECT_DIR}")
    print(f"Script base: {BASE_SCRIPT}")
    print(f"Saída: {OUT_DIR}")
    print(f"Total de cenários: {len(cenarios)}")
    print()
    print("Dias usados:")
    for d in dias:
        print(" -", d)

    if len(dias) < 10:
        print()
        print(f"[ALERTA] Foram encontrados apenas {len(dias)} dias.")

    df_cenarios = pd.DataFrame(cenarios)
    df_cenarios.to_csv(OUT_DIR / "cenarios_otimizacao.csv", index=False, encoding="utf-8-sig")

    resumos_dia = []
    bloqueios_all = []
    diagnostico = []

    cache_dados = {}

    for day in dias:
        path_ia = bt.LOG_DIR / f"log_ia_exec_{day}.csv"
        path_120s = bt.LOG_DIR / f"log_120s_{day}.csv"

        print()
        print("=" * 110)
        print(f"DIA {day}")
        print("=" * 110)

        try:
            mercado, diag_120s = bt.preparar_120s(path_120s, day)
            ia, diag_ia = bt.preparar_ia(path_ia, mercado, day)

            cache_dados[day] = (ia, mercado)

            diag = {"dia": day}
            diag.update(diag_120s)
            diag.update(diag_ia)
            diag["hora_inicio_ia"] = ia["_ts"].min()
            diag["hora_fim_ia"] = ia["_ts"].max()
            diag["hora_inicio_120s"] = mercado["_ts"].min()
            diag["hora_fim_120s"] = mercado["_ts"].max()
            diagnostico.append(diag)

            print(f"[OK] IA rows: {len(ia)} | 120s rows: {len(mercado)}")
            print(f"[OK] Horário IA: {ia['_ts'].min()} -> {ia['_ts'].max()}")

            total = len(cenarios)

            for i, cfg in enumerate(cenarios, start=1):
                nome = cfg["name"]

                ops, bloqueios = bt.simular_operacoes(ia, mercado, cfg)

                resumo = resumir_ops(bt, ops, nome, day)
                resumos_dia.append(resumo)

                b = {"dia": day, "cenario": nome}
                b.update(bloqueios)
                bloqueios_all.append(b)

                if i % 250 == 0 or i == total:
                    print(f"[OK] {i}/{total} cenários processados")

        except Exception as e:
            print(f"[ERRO] {day}: {e}")
            diagnostico.append({"dia": day, "erro": str(e)})

    df_dia = pd.DataFrame(resumos_dia)
    df_bloq = pd.DataFrame(bloqueios_all)
    df_diag = pd.DataFrame(diagnostico)

    resumo_total = []

    for cfg in cenarios:
        nome = cfg["name"]
        parte = df_dia[df_dia["cenario"] == nome].copy()

        if parte.empty:
            continue

        total_ops = int(parte["operacoes"].sum())
        wins = int(parte["wins"].sum())
        losses = int(parte["losses"].sum())
        resultado = float(parte["resultado_pts"].sum())

        # Para PF agregado aproximado, usa a soma dos resultados por dia como base menor.
        # Depois os TOP cenários terão operações completas salvas para auditoria.
        buy_resultado = float(parte["buy_resultado"].sum())
        sell_resultado = float(parte["sell_resultado"].sum())

        # Aproximação via dia: melhor que nada para ranking inicial.
        # Como cada operação tem SL/TP próximo, o PF por operações será reavaliado nos TOP.
        pf_medio_ponderado = np.average(
            parte["profit_factor"].astype(float),
            weights=np.maximum(parte["operacoes"].astype(float), 1)
        ) if total_ops > 0 else 0.0

        dd_total = float(parte["max_drawdown_pts"].min())

        resumo_total.append({
            "cenario": nome,
            "dia": "TOTAL",
            "operacoes": total_ops,
            "wins": wins,
            "losses": losses,
            "winrate": round(wins / total_ops, 4) if total_ops else 0.0,
            "resultado_pts": round(resultado, 2),
            "profit_factor": round(float(pf_medio_ponderado), 4),
            "max_drawdown_pts": round(dd_total, 2),
            "buy_ops": int(parte["buy_ops"].sum()),
            "buy_resultado": round(buy_resultado, 2),
            "sell_ops": int(parte["sell_ops"].sum()),
            "sell_resultado": round(sell_resultado, 2),
            "avg_trade": round(resultado / total_ops, 4) if total_ops else 0.0,
        })

    df_total = pd.DataFrame(resumo_total)

    # Garante que a base existe.
    if df_total[df_total["cenario"] == "BASE_PC058_PV058"].empty:
        raise RuntimeError("BASE_PC058_PV058 não encontrada no resumo total.")

    df_rank = montar_ranking(df_total, df_dia)

    df_dia.to_csv(OUT_DIR / "resumo_por_dia_otimizacao.csv", index=False, encoding="utf-8-sig")
    df_total.to_csv(OUT_DIR / "resumo_total_otimizacao.csv", index=False, encoding="utf-8-sig")
    df_rank.to_csv(OUT_DIR / "ranking_otimizacao.csv", index=False, encoding="utf-8-sig")
    df_bloq.to_csv(OUT_DIR / "bloqueios_otimizacao.csv", index=False, encoding="utf-8-sig")
    df_diag.to_csv(OUT_DIR / "diagnostico_otimizacao.csv", index=False, encoding="utf-8-sig")

    # Reprocessa operações completas dos TOP cenários para auditoria.
    top_cenarios = df_rank.head(TOP_N_OPS)["cenario"].tolist()
    cfg_by_name = {c["name"]: c for c in cenarios}
    ops_top = []

    print()
    print("=" * 110)
    print(f"REPROCESSANDO OPERAÇÕES DOS TOP {TOP_N_OPS} CENÁRIOS")
    print("=" * 110)

    for day, (ia, mercado) in cache_dados.items():
        for nome in top_cenarios:
            cfg = cfg_by_name[nome]
            ops, _ = bt.simular_operacoes(ia, mercado, cfg)
            if not ops.empty:
                ops.insert(0, "dia", day)
                ops.insert(1, "cenario", nome)
                ops_top.append(ops)

    if ops_top:
        df_ops_top = pd.concat(ops_top, ignore_index=True)
    else:
        df_ops_top = pd.DataFrame()

    df_ops_top.to_csv(OUT_DIR / "operacoes_top30_cenarios.csv", index=False, encoding="utf-8-sig")

    txt_path = OUT_DIR / "TOP_30_OTIMIZACAO.txt"
    cols_print = [
        "rank", "cenario", "classificacao", "operacoes", "winrate",
        "resultado_pts", "profit_factor", "max_drawdown_pts",
        "dias_positivos", "buy_resultado", "sell_resultado",
        "delta_resultado_vs_base", "delta_drawdown_vs_base", "score"
    ]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TOP 30 - OTIMIZAÇÃO DE COMBINAÇÕES LARA - 10 PREGÕES\n")
        f.write("=" * 120 + "\n\n")
        f.write(df_rank.head(30)[cols_print].to_string(index=False))
        f.write("\n\nBASE:\n")
        f.write(df_total[df_total["cenario"] == "BASE_PC058_PV058"].to_string(index=False))

    print()
    print("=" * 110)
    print("TOP 30 - RANKING FINAL")
    print("=" * 110)
    print(df_rank.head(30)[cols_print].to_string(index=False))

    print()
    print("=" * 110)
    print("BASE")
    print("=" * 110)
    print(df_total[df_total["cenario"] == "BASE_PC058_PV058"].to_string(index=False))

    print()
    print("=" * 110)
    print("ARQUIVOS GERADOS")
    print("=" * 110)
    for f in sorted(OUT_DIR.glob("*")):
        print(f"[OK] {f}")

    zip_base = OUT_DIR
    zip_path = shutil.make_archive(str(zip_base), "zip", OUT_DIR)

    print()
    print("=" * 110)
    print("ZIP GERADO")
    print("=" * 110)
    print(zip_path)

    print()
    print("[CONCLUÍDO] Otimização finalizada.")


if __name__ == "__main__":
    main()
