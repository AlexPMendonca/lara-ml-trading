from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def step(msg: str) -> None:
    print("\n" + "=" * 74)
    print(msg)
    print("=" * 74)


def fail(msg: str) -> None:
    print(f"\n[FALHA] {msg}")
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.exists():
        fail(f"Arquivo obrigatório não encontrado: {path}")


def run_checked(label: str, args: list[str], cwd: Path) -> None:
    print(f"[RUN] {label}")
    print("      " + " ".join(args))
    result = subprocess.run(args, cwd=str(cwd))
    if result.returncode != 0:
        fail(f"{label} retornou código {result.returncode}")


def format_threshold_tag(value: float) -> str:
    return f"{round(value * 100):03d}"


def get_new_version_tag(
    current_version_tag: str,
    day: str,
    day_index: int | None,
    auto_increment_day_index: bool,
) -> str:
    if not current_version_tag or not current_version_tag.strip():
        fail("config_lara.json está sem version_tag.")

    regex = r"^(?P<prefix>.+)_D(?P<idx>\d+)_(?P<date>\d{4}-\d{2}-\d{2})$"
    match = re.match(regex, current_version_tag)

    if not match:
        fail(f"Não consegui interpretar o version_tag atual: {current_version_tag}")

    prefix = match.group("prefix")
    curr_idx = int(match.group("idx"))
    curr_date = match.group("date")

    if day_index is not None:
        new_idx = int(day_index)
    elif auto_increment_day_index:
        new_idx = curr_idx if curr_date == day else curr_idx + 1
    else:
        if curr_date == day:
            new_idx = curr_idx
        else:
            fail(
                f"O version_tag atual está em {curr_date} e você pediu {day}. "
                f"Informe --day-index ou use --auto-increment-day-index."
            )

    return f"{prefix}_D{new_idx}_{day}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True, help="Dia no formato YYYY-MM-DD")
    parser.add_argument("--pc", type=float, default=0.74, help="Threshold de compra")
    parser.add_argument("--pv", type=float, default=0.58, help="Threshold de venda")
    parser.add_argument("--project-dir", default=".", help="Pasta raiz do projeto Lara")
    parser.add_argument("--day-index", type=int, default=None, help="Índice D manual")
    parser.add_argument(
        "--auto-increment-day-index",
        action="store_true",
        help="Incrementa o D automaticamente se a data mudar",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    logs_dir = project_dir / "arquivo_ABC"
    zip_dir = project_dir / "ZIP_DIA"
    config_path = project_dir / "config_lara.json"
    ia_path = logs_dir / f"log_ia_exec_{args.day}.csv"
    m120_path = logs_dir / f"log_120s_{args.day}.csv"
    base_shadow = logs_dir / f"log_operacoes_shadow_v3_{args.day}.csv"

    pc_tag = format_threshold_tag(args.pc)
    pv_tag = format_threshold_tag(args.pv)

    strict_path = logs_dir / f"log_operacoes_shadow_v3_{args.day}_STRICT.csv"
    calib_path = logs_dir / f"log_operacoes_shadow_v3_{args.day}_CALIB_PC{pc_tag}_PV{pv_tag}.csv"
    zip_path = zip_dir / f"Lara_4.0_{args.day}.zip"

    eod_script = project_dir / "eod_logfix.py"
    shadow_script = project_dir / "simular_shadow.py"
    prezip_script = project_dir / "prezip_check.py"

    step("1) Pré-checagem de arquivos base")
    require_file(config_path)
    require_file(ia_path)
    require_file(m120_path)
    require_file(eod_script)
    require_file(shadow_script)

    print(f"[OK] {config_path.name}")
    print(f"[OK] {ia_path.name}")
    print(f"[OK] {m120_path.name}")

    step("2) Backup e atualização automática do version_tag")
    backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_backup = project_dir / f"config_lara_BACKUP_PRE_FECHAMENTO_{backup_suffix}.json"
    shutil.copy2(config_path, config_backup)
    print(f"[OK] Backup criado: {config_backup}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    current_version_tag = str(config.get("version_tag", "")).strip()
    new_version_tag = get_new_version_tag(
        current_version_tag=current_version_tag,
        day=args.day,
        day_index=args.day_index,
        auto_increment_day_index=args.auto_increment_day_index,
    )

    config["version_tag"] = new_version_tag
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[OK] version_tag anterior: {current_version_tag}")
    print(f"[OK] version_tag novo    : {new_version_tag}")

    step("3) Ajuste EOD do dia")
    run_checked(
        "eod_logfix.py",
        [
            sys.executable,
            str(eod_script),
            "--pasta_logs",
            str(logs_dir),
            "--data",
            args.day,
        ],
        cwd=project_dir,
    )

    step("4) Gerar STRICT")
    run_checked(
        "simular_shadow.py STRICT",
        [
            sys.executable,
            str(shadow_script),
            "--project-dir",
            str(project_dir),
            "--day",
            args.day,
            "--mode",
            "strict",
        ],
        cwd=project_dir,
    )

    require_file(base_shadow)
    shutil.copy2(base_shadow, strict_path)
    require_file(strict_path)
    print(f"[OK] STRICT salvo em: {strict_path}")

    step("5) Gerar CALIB")
    run_checked(
        "simular_shadow.py CALIB",
        [
            sys.executable,
            str(shadow_script),
            "--project-dir",
            str(project_dir),
            "--day",
            args.day,
            "--mode",
            "calib",
            "--pc",
            str(args.pc),
            "--pv",
            str(args.pv),
        ],
        cwd=project_dir,
    )

    require_file(base_shadow)
    shutil.copy2(base_shadow, calib_path)
    require_file(calib_path)
    print(f"[OK] CALIB salvo em: {calib_path}")

    step("6) Conferência final dos artefatos")
    artifacts = [config_path, ia_path, m120_path, strict_path, calib_path]
    for file_path in artifacts:
        require_file(file_path)
        stat = file_path.stat()
        print(f"{file_path} | {stat.st_size} bytes | {datetime.fromtimestamp(stat.st_mtime)}")

    if prezip_script.exists():
        step("7) Rodar prezip_check")
        run_checked(
            "prezip_check.py",
            [
                sys.executable,
                str(prezip_script),
                "--day",
                args.day,
            ],
            cwd=project_dir,
        )

    step("8) Gerar ZIP")
    zip_dir.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in artifacts:
            arcname = file_path.relative_to(project_dir)
            zf.write(file_path, arcname)

    require_file(zip_path)
    stat = zip_path.stat()
    print(f"{zip_path} | {stat.st_size} bytes | {datetime.fromtimestamp(stat.st_mtime)}")
    print(f"SHA256: {sha256_file(zip_path)}")

    step("9) Resumo")
    print("[OK] Fechamento concluído com sucesso.")
    print(f"[OK] Dia         : {args.day}")
    print(f"[OK] version_tag : {new_version_tag}")
    print(f"[OK] STRICT      : {strict_path}")
    print(f"[OK] CALIB       : {calib_path}")
    print(f"[OK] ZIP         : {zip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())