"""ffmpeg helpers: probe duration and cut a part into small mono MP3 chunks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config

CHUNK_SEC = 600  # 10 min: ~5 MB as 64 kbps mono MP3, well under API upload limits

# Cut low rumble (table knocks, HVAC), then even out loudness so quiet, far-away speakers
# are boosted and close/loud ones are tamed. Max gain 10x keeps silence from becoming hiss.
LEVELING = "highpass=f=80,dynaudnorm=f=500:g=31:p=0.9:m=10"


def _filters() -> list[str]:
    return ["-af", LEVELING] if config.get("TRANSCRIBER_NORMALIZE", "1").lower() not in ("0", "false", "no", "off") else []


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise SystemExit(f"{tool} not found on PATH. Install ffmpeg (winget install Gyan.FFmpeg).")
    return path


def duration(path: Path) -> float:
    result = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, creationflags=config.NO_WINDOW,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Can't read audio file {path.name}: {result.stderr.strip()[-300:] or 'unknown format'}")


def make_chunks(src: Path, out_dir: Path, chunk_sec: int = CHUNK_SEC) -> list[tuple[Path, float]]:
    """Return [(chunk_path, start_seconds)], creating any chunk that doesn't exist yet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = duration(src)
    chunks = []
    start = 0.0
    index = 0
    while start < total - 1.0:  # skip a sub-second tail
        chunk = out_dir / f"chunk-{index:03d}.mp3"
        if not chunk.exists():
            tmp = chunk.with_name(f".{chunk.name}")
            result = subprocess.run(
                [_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", f"{start:.3f}", "-t", str(chunk_sec), "-i", str(src),
                 "-vn", "-ac", "1", *_filters(), "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "64k",
                 "-f", "mp3", str(tmp)],
                capture_output=True, text=True, creationflags=config.NO_WINDOW,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg couldn't read {src.name}: {result.stderr.strip()[-500:]}")
            tmp.replace(chunk)
        chunks.append((chunk, start))
        start += chunk_sec
        index += 1
    return chunks
