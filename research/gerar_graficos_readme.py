# -*- coding: utf-8 -*-
"""Gera os gráficos usados no README a partir dos relatórios do projeto.

Uso (de qualquer pasta):
    python research/gerar_graficos_readme.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "lara" / "modelos" / "modelo_lara4"
REPORT = MODEL_DIR / "modelo_lara4_xgb_jan2026_shadow_pv058_SIMPLIFICADO_V1.report.json"
CALIB = REPO / "reports" / "auditorias" / "auditoria_memorizacao_calibracao.csv"
CURVA = REPO / "reports" / "backtest" / "otimizacao_combinacoes_lara_10d" / "curva_melhor_vs_base.csv"
OUT = REPO / "docs" / "img"

# Paleta categórica validada (slots 1 e 2) + tinta de texto
AZUL, LARANJA = "#2a78d6", "#eb6834"
TXT, TXT2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def estilo(ax, titulo: str, subtitulo: str) -> None:
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRADE)
    ax.tick_params(colors=TXT2, length=0, labelsize=10)
    ax.yaxis.grid(True, color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(titulo, loc="left", fontsize=13, color=TXT, fontweight="bold", pad=24)
    ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=10, color=TXT2)


def salvar(fig, nome: str) -> None:
    fig.patch.set_facecolor(FUNDO)
    fig.tight_layout()
    fig.savefig(OUT / nome, dpi=150, facecolor=FUNDO)
    plt.close(fig)
    print("[OK]", OUT / nome)


def grafico_treino_teste() -> None:
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    metricas = ["ROC AUC", "Acurácia"]
    treino = [r["train_metrics"]["roc_auc"], r["train_metrics"]["accuracy"]]
    teste = [r["test_metrics"]["roc_auc"], r["test_metrics"]["accuracy"]]
    n_tr, n_te = r["train_metrics"]["rows"], r["test_metrics"]["rows"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(metricas))
    w = 0.34
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], treino, w, color=AZUL, label=f"Treino ({n_tr} operações)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], teste, w, color=LARANJA, label=f"Teste fora da amostra ({n_te} operações)")
    ax.axhline(0.5, color=TXT2, linestyle="--", linewidth=1)
    ax.text(1.55, 0.51, "acaso (0,50)", color=TXT2, fontsize=9, ha="right")
    for barras in (b1, b2):
        for b in barras:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}".replace(".", ","),
                    ha="center", color=TXT, fontsize=10)
    ax.set_xticks(list(x), metricas)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    estilo(ax, "O modelo decora o treino e não generaliza", "XGBoost, holdout temporal dos últimos 3 pregões")
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9, labelcolor=TXT2)
    salvar(fig, "treino_vs_teste.png")


def grafico_calibracao() -> None:
    c = pd.read_csv(CALIB)
    c["meio"] = c["prob_bucket"].str.slice(0, 4).astype(float) + 0.05
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot([0, 1], [0, 1], color=TXT2, linestyle="--", linewidth=1)
    ax.text(0.97, 0.9, "calibração perfeita", color=TXT2, fontsize=9, ha="right", rotation=33)
    for split, cor, nome in (("train", AZUL, "Treino"), ("test", LARANJA, "Teste")):
        s = c[(c["split"] == split) & (c["rows"] > 0)]
        ax.plot(s["meio"], s["acerto_real"], color=cor, linewidth=2, marker="o", markersize=7,
                markeredgecolor=FUNDO, markeredgewidth=2, label=nome)
        ultimo = s.iloc[-1]
        ax.annotate(nome, (ultimo["meio"], ultimo["acerto_real"]), xytext=(8, 0), textcoords="offset points",
                    color=TXT, fontsize=10, va="center")
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Probabilidade prevista pelo modelo", color=TXT2, fontsize=10)
    ax.set_ylabel("Taxa de acerto real", color=TXT2, fontsize=10)
    estilo(ax, "Confiança alta no teste não vira acerto", "Acerto real por faixa de probabilidade prevista")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TXT2)
    salvar(fig, "calibracao.png")


def grafico_backtest() -> None:
    d = pd.read_csv(CURVA)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rotulos = {"base": "Configuração base", "melhor": "Melhor combinação (escolhida nos mesmos dias)"}
    for col, cor in (("base", LARANJA), ("melhor", AZUL)):
        ax.plot(d["dia"], d[col].cumsum(), color=cor, linewidth=2, marker="o", markersize=6,
                markeredgecolor=FUNDO, markeredgewidth=2, label=rotulos[col])
        ax.annotate(f"{d[col].sum():+,.0f} pts".replace(",", "."), (len(d) - 1, d[col].sum()), xytext=(8, 0),
                    textcoords="offset points", color=TXT, fontsize=10, va="center")
    ax.axhline(0, color=TXT2, linewidth=1)
    ax.set_xticks(range(len(d)), [f"{x[8:10]}/{x[5:7]}" for x in d["dia"]])
    ax.set_xlim(-0.3, len(d) + 0.9)
    ax.set_ylabel("Resultado acumulado (pontos brutos)", color=TXT2, fontsize=10)
    n = int(open(CURVA.with_name("n_cenarios.txt")).read())
    estilo(ax, "Otimização de filtros em 10 pregões (in-sample)", f"{n:,} combinações testadas".replace(",", ".") + ", sem custos operacionais")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TXT2)
    salvar(fig, "backtest_otimizacao.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    grafico_treino_teste()
    grafico_calibracao()
    grafico_backtest()
