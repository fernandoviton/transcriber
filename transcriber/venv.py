"""Hand off to the project's .venv if it exists and we're not already running in it.

This keeps `py -m transcriber ...` and Transcriber.pyw using the project's installed
dependencies, including a native Python environment on Windows ARM64.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"


def _in_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == VENV.resolve()
    except OSError:
        return False


def relaunch(args: list[str], gui: bool = False) -> None:
    """If .venv exists and we're elsewhere, rerun `args` with its Python and exit."""
    if _in_venv() or os.environ.get("TRANSCRIBER_NO_VENV"):
        return
    python = VENV / "Scripts" / ("pythonw.exe" if gui else "python.exe")
    if not python.exists():
        return
    env = dict(os.environ, TRANSCRIBER_NO_VENV="1",  # never loop
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    if gui:
        subprocess.Popen([str(python), *args], cwd=ROOT, env=env)
        sys.exit(0)
    try:
        result = subprocess.run([str(python), *args], env=env)
    except KeyboardInterrupt:  # the child got Ctrl+C too and handles it
        sys.exit(130)
    sys.exit(result.returncode)
