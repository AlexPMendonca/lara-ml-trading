# -*- coding: utf-8 -*-
"""
logging_utils.py — LOGFIX_DUP_IAEXIT + CAMPOS_GATE_DIR (2026-01-19)
- Evita duplicidade de fechamento IA-EXIT (chave: pid+motivo_saida no dia)
- Define fechado_por_ia="sim" para IA-EXIT e "nao" para TP/SL quando ausente
- Faz forward-fill de atr_media50 a partir do último valor do dia (fallback: atr)
- Formata probabilidades pc/pv com 3 casas decimais
- NOVO: adiciona campos de auditoria do gate direcional:
    log_ia_exec: direcao, conf_dir, limiar_dir
    log_operacoes_v3: direcao, conf_dir, limiar_dir
- NOVO: auto-upgrade de header (se CSV do dia já existir com header antigo)
"""
import csv
import datetime as dt
from pathlib import Path
from typing import List, Optional


def ensure_logs_dir(logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)


def _csv_path(logs_dir: Path, prefix: str) -> Path:
    return logs_dir / f"{prefix}_{dt.date.today().isoformat()}.csv"


def _read_header(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header and isinstance(header, list):
                return [str(x) for x in header]
    except Exception:
        return None
    return None


def _rewrite_with_new_header(path: Path, new_header: List[str]) -> None:
    """
    Reescreve o CSV inteiro com um header novo, preservando os dados existentes.
    Colunas novas recebem "".
    Colunas antigas extras (que não estão no new_header) são preservadas (mantidas no header final).
    """
    old_header = _read_header(path) or []
    if not old_header:
        # arquivo vazio ou inválido -> recria
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(new_header)
        return

    # Preserva colunas antigas extras e acrescenta as novas faltantes
    final_header = list(old_header)
    for col in new_header:
        if col not in final_header:
            final_header.append(col)

    # Lê todos os registros existentes
    rows = []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            dr = csv.DictReader(f)
            for row in dr:
                rows.append(row)
    except Exception:
        rows = []

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(final_header)
        for row in rows:
            w.writerow([row.get(k, "") for k in final_header])

    tmp.replace(path)


def _ensure_header(path: Path, header: List[str]) -> List[str]:
    """
    Garante que exista header. Se já existir e estiver “menor” (sem colunas novas),
    faz upgrade automático preservando dados.
    Retorna o header efetivo do arquivo (após upgrade).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        return header

    existing = _read_header(path)
    if not existing:
        # arquivo corrompido/vazio: recria
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        return header

    # Se faltar qualquer coluna do header desejado, faz upgrade
    missing = [c for c in header if c not in existing]
    if missing:
        _rewrite_with_new_header(path, header)
        # retorna o header efetivo pós-upgrade
        return _read_header(path) or header

    return existing


def _append_row(path: Path, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _fmt_prob(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return x if (x is not None) else ""


def _last_nonempty_from_csv(path: Path, field: str):
    """Return last non-empty value for 'field' from CSV 'path' (today's file)."""
    if not path.exists():
        return None
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = list(csv.DictReader(f))
            for row in reversed(r):
                v = row.get(field, "")
                if v not in ("", None):
                    return v
    except Exception:
        return None
    return None


def log_ia_exec(logs_dir: Path, d: dict):
    p = _csv_path(logs_dir, "log_ia_exec")

    # Header com campos novos do gate direcional
    h = [
        "hora",
        "pc",
        "pv",
        "regime_mercado",
        "direcao",
        "conf_dir",
        "limiar_dir",
        "tendencia_pct",
        "atr",
        "atr_media50",
        "vol_norm",
        "limiar_compra_ajustado",
        "limiar_venda_ajustado",
        "limiar_confianca",
        "bloqueado_por",
        "cooldown_restante",
        "version_tag",
    ]

    # normalizar probabilidades
    if "pc" in d:
        d["pc"] = _fmt_prob(d["pc"])
    if "pv" in d:
        d["pv"] = _fmt_prob(d["pv"])
    if "conf_dir" in d:
        d["conf_dir"] = _fmt_prob(d["conf_dir"])

    # garante header (com upgrade automático) e escreve
    header_eff = _ensure_header(p, h)
    _append_row(p, [d.get(k, "") for k in header_eff])


def log_operacao(logs_dir: Path, d: dict):
    """
    Gera uma linha em log_operacoes_v3_YYYY-MM-DD.csv garantindo:
    - Sem duplicidade para IA-EXIT (pid + motivo_saida)
    - fechado_por_ia default coerente
    - atr_media50 preenchido (ffill / atr)
    - pc/pv com 3 casas
    - NOVO: direcao/conf_dir/limiar_dir
    """
    p = _csv_path(logs_dir, "log_operacoes_v3")

    h = [
        "hora",
        "pid",
        "tipo",
        "direcao",
        "conf_dir",
        "limiar_dir",
        "preco",
        "sl",
        "tp",
        "resultado",
        "fechado_por_ia",
        "regime_mercado",
        "tendencia_pct",
        "atr",
        "atr_media50",
        "vol_norm",
        "pc",
        "pv",
        "limiar_compra_ajustado",
        "limiar_venda_ajustado",
        "limiar_confianca",
        "motivo_saida",
        "lucro_real",
        "duracao_operacao",
        "version_tag",
    ]

    # Defaults/inteligência de campos
    motivo = str(d.get("motivo_saida", "")).upper()
    if not d.get("fechado_por_ia"):
        if "IA" in motivo:
            d["fechado_por_ia"] = "sim"
        elif motivo in ("TP", "SL"):
            d["fechado_por_ia"] = "nao"

    # preencher atr_media50 se vazio -> último valor do dia -> atr
    atr50 = d.get("atr_media50", "")
    if atr50 in ("", None):
        last = _last_nonempty_from_csv(p, "atr_media50")
        d["atr_media50"] = last if last not in ("", None) else d.get("atr", "")

    # formatar probabilidades
    if "pc" in d:
        d["pc"] = _fmt_prob(d["pc"])
    if "pv" in d:
        d["pv"] = _fmt_prob(d["pv"])
    if "conf_dir" in d:
        d["conf_dir"] = _fmt_prob(d["conf_dir"])

    # Header sempre garantido (com upgrade automático)
    header_eff = _ensure_header(p, h)

    # Guardar chaves existentes (para IA-EXIT dedup)
    dedup_key = (str(d.get("pid", "")), motivo)
    if "IA" in motivo and p.exists():
        try:
            with open(p, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get("pid", ""), str(row.get("motivo_saida", "")).upper()) == dedup_key:
                        # Já temos um IA-EXIT para esse PID hoje -> não escrever de novo
                        return
        except Exception:
            # Em caso de erro de leitura, não bloquear o log
            pass

    # Escrever linha (com header efetivo)
    _append_row(p, [d.get(k, "") for k in header_eff])


def log_120s(logs_dir: Path, d: dict):
    p = _csv_path(logs_dir, "log_120s")
    h = ["hora", "symbol", "preco", "positions", "version_tag"]
    header_eff = _ensure_header(p, h)
    _append_row(p, [d.get(k, "") for k in header_eff])


def log_validacao_incremental(logs_dir: Path, d: dict):
    """
    Registra resultado resumido, evitando duplicidade por (pid, motivo_saida) no dia.
    """
    p = _csv_path(logs_dir, "validacao_incremental")
    h = ["hora", "pid", "tipo", "resultado", "lucro_real", "motivo_saida", "version_tag"]
    header_eff = _ensure_header(p, h)

    motivo = str(d.get("motivo_saida", "")).upper()
    key = (str(d.get("pid", "")), motivo)
    if p.exists():
        try:
            with open(p, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get("pid", ""), str(row.get("motivo_saida", "")).upper()) == key:
                        return
        except Exception:
            pass

    _append_row(p, [d.get(k, "") for k in header_eff])
