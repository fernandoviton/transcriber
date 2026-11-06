"""Settings from environment variables, optionally loaded from a .env file at the repo root."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "sessions"
PROFILES = ROOT / "profiles"

# Rough list prices in USD per audio minute, used only for the running cost estimate.
ASR_PRICE_PER_MIN = {
    "gpt-4o-transcribe-diarize": 0.006,
    "gpt-transcribe": 0.006,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
    "whisper-1": 0.006,
}


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# Keeps ffmpeg/copilot from flashing console windows when running under pythonw (the GUI).
NO_WINDOW = getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0)


def default_asr() -> str:
    return get("TRANSCRIBER_ASR") or "whisper"


def default_summarizer() -> str:
    return get("TRANSCRIBER_SUMMARIZER", "local")


def default_summaries() -> str:
    return get("TRANSCRIBER_SUMMARIES", "later")


ASR_BACKENDS = ("whisper", "local", "openai-diarize", "openai")
SUMMARIZERS = ("local", "copilot", "openai")
# auto: summarize each part as soon as it's transcribed. later: only transcribe; summarize on request.
SUMMARY_MODES = ("later", "auto")
REQUIRED_PROFILE_FILES = ("summarize.md", "recap.md", "glossary.md")


def check_ready(asr: str, summarizer: str | None) -> None:
    """Fail now, not an hour into a recording, if a backend can't run. summarizer=None: not summarizing."""
    import importlib.util
    import shutil

    problems = []
    if asr not in ASR_BACKENDS:
        problems.append(f"unknown ASR backend '{asr}' (choose from {', '.join(ASR_BACKENDS)})")
    if summarizer is not None and summarizer not in SUMMARIZERS:
        problems.append(f"unknown summarizer '{summarizer}' (choose from {', '.join(SUMMARIZERS)})")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            problems.append(f"{tool} not found on PATH (winget install Gyan.FFmpeg)")
    if (asr.startswith("openai") or summarizer == "openai") and not get("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set (add it to .env), but an OpenAI backend is selected")
    if asr == "local" and importlib.util.find_spec("onnx_asr") is None:
        problems.append("the local speech model isn't installed in this Python (onnx-asr). "
                        "Create the .venv as described under Setup in README.md")
    if summarizer == "copilot" and not shutil.which("copilot"):
        problems.append("GitHub Copilot CLI `copilot` not found on PATH")
    if problems:
        raise SystemExit("Can't start:\n  - " + "\n  - ".join(problems))
