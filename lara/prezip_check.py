# -*- coding: utf-8 -*-
"""Checagem rápida antes de gerar ZIP diário."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def check(path: Path, name: str):
    if not path.exists():
        print(f"[MISSING] {name}: {path}")
        return None

    df = pd.read_csv(path)
    print(f"[OK] {name}: rows={len(df)} cols={len(df.columns)}")

    if "hora" in df.columns and len(df):
        print(f"     hora: {df['hora'].iloc[0]} -> {df['hora'].iloc[-1]}")

    if "preco" in df.columns:
        zeros = int((pd.to_numeric(df["preco"], errors="coerce") == 0).sum())
        if zeros:
            print(f"     [WARN] preco=0: {zeros}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Checagem rápida dos logs do dia antes do ZIP.")
    parser.add_argument("--day", required=True, help="Data no formato YYYY-MM-DD")
    parser.add_argument("--base", default="arquivo_ABC", help="Pasta de logs")
    args = parser.parse_args()

    base = Path(args.base)
    day = args.day

    ia = base / f"log_ia_exec_{day}.csv"
    hb = base / f"log_120s_{day}.csv"
    check(ia, "log_ia_exec")
    check(hb, "log_120s")


if __name__ == "__main__":
    main()
