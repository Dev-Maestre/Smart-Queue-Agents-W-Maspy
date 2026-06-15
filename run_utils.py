from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


RUNS_BASE = "runs_output"
LATEST_RUN_FILE = "latest_run.txt"


def get_run_dir(tag: str) -> Path:
    base_dir = Path(__file__).parent
    runs_base = base_dir / RUNS_BASE
    runs_base.mkdir(parents=True, exist_ok=True)

    env_run = os.environ.get("RUN_DIR")
    if env_run:
        run_dir = Path(env_run)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_base / f"{tag}_{ts}"

    run_dir.mkdir(parents=True, exist_ok=True)
    (runs_base / LATEST_RUN_FILE).write_text(str(run_dir), encoding="utf-8")
    return run_dir


def write_run_readme(run_dir: Path, params: dict):
    lines = []
    lines.append("# Execucao - parametros")
    lines.append("")
    for key in sorted(params.keys()):
        lines.append(f"- {key}: {params[key]}")
    lines.append("")
    config_path = Path(__file__).parent / "config.py"
    if config_path.exists():
        lines.append("## Config utilizada")
        lines.append("")
        lines.append("```python")
        lines.extend(config_path.read_text(encoding="utf-8").splitlines())
        lines.append("```")
        lines.append("")
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def get_latest_run_dir() -> Path | None:
    base_dir = Path(__file__).parent
    latest = base_dir / RUNS_BASE / LATEST_RUN_FILE
    if latest.exists():
        return Path(latest.read_text(encoding="utf-8").strip())
    return None

