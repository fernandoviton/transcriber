"""Speech-to-text backends and per-part transcription with per-chunk caching.

Backends (pick with TRANSCRIBER_ASR or `--asr`):
  whisper         Whisper large-v3-turbo (whisper.cpp) on this machine: free, private, ~8x faster than
                  real time. Glossary names go in as hints. Timestamps, no speakers. The default.
  local           NVIDIA Parakeet v3 (via onnx-asr) on this machine: free, private, ~8x faster than
                  real time on the native ARM64 .venv. Timestamps, no speakers, no glossary hints.
  openai-diarize  OpenAI gpt-4o-transcribe-diarize: timestamps + speaker letters. Needs OPENAI_API_KEY.
  openai          OpenAI gpt-transcribe (or TRANSCRIBER_OPENAI_ASR_MODEL): glossary names go in as
                  keyword hints, but there are no timestamps or speakers inside a 10-minute chunk.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import audio, config
from .session import Session

BACKENDS = config.ASR_BACKENDS


def glossary_terms(glossary_md: str) -> list[str]:
    """Names from glossary bullets like '- Project Atlas: migration project' (comments are skipped)."""
    terms = []
    for line in glossary_md.splitlines():
        match = re.match(r"\s*[-*]\s+\**([^:*<>]+)", line)
        if match:
            terms.append(match.group(1).strip())
    return terms


def glossary_hint_from(terms: list[str], limit: int = 600) -> str:
    return (("Names and terms: " + ", ".join(terms) + ".") if terms else "")[:limit]


# ---- OpenAI ------------------------------------------------------------------

def _openai_post(path: Path, fields: dict) -> dict:
    import httpx

    try:
        return _openai_post_raw(path, fields)
    except httpx.TransportError as exc:
        raise RuntimeError(f"Can't reach OpenAI ({exc.__class__.__name__}). No internet? This session uses "
                           "OpenAI transcription; switch it to local (Settings…, or `set --asr local`) "
                           "to work offline.") from exc


def _openai_post_raw(path: Path, fields: dict) -> dict:
    import httpx

    key = config.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env). Or use --asr local.")
    base = config.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    for attempt in range(4):
        with path.open("rb") as f:
            response = httpx.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                data=fields,
                files={"file": (path.name, f, "audio/mpeg")},
                timeout=900,
            )
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3 \
                and "insufficient_quota" not in response.text:  # out of credits won't fix itself
            time.sleep(5 * 2 ** attempt)
            continue
        if response.status_code >= 400:
            raise RuntimeError(f"OpenAI transcription failed ({response.status_code}): {response.text[:500]}")
        return response.json()
    raise RuntimeError("unreachable")


def _openai_diarize(path: Path, terms: list[str]) -> dict:
    model = config.get("TRANSCRIBER_OPENAI_DIARIZE_MODEL", "gpt-4o-transcribe-diarize")
    fields = {"model": model, "response_format": "diarized_json", "chunking_strategy": "auto"}
    language = config.get("TRANSCRIBER_LANGUAGE", "en")
    if language:
        fields["language"] = language
    data = _openai_post(path, fields)
    segments = [{"start": float(s.get("start", 0)), "end": float(s.get("end", 0)),
                 "speaker": s.get("speaker"), "text": (s.get("text") or "").strip()}
                for s in data.get("segments", [])]
    if not segments and data.get("text"):
        segments = [{"start": 0.0, "end": audio.duration(path), "speaker": None, "text": data["text"].strip()}]
    return {"model": model, "segments": segments}


def _openai_plain(path: Path, terms: list[str]) -> dict:
    model = config.get("TRANSCRIBER_OPENAI_ASR_MODEL", "gpt-transcribe")
    fields: dict = {"model": model, "response_format": "json"}
    if terms:
        fields["prompt"] = "A recording of a group conversation. " + glossary_hint_from(terms)
    language = config.get("TRANSCRIBER_LANGUAGE", "en")
    if model == "gpt-transcribe":
        # gpt-transcribe takes literal keyword hints, and `languages[]` instead of `language`.
        if terms:
            fields["keywords[]"] = terms[:100]
        if language:
            fields["languages[]"] = [language]
    elif language:
        fields["language"] = language
    data = _openai_post(path, fields)
    text = (data.get("text") or "").strip()
    return {"model": model, "segments": [{"start": 0.0, "end": audio.duration(path), "speaker": None, "text": text}]}


# ---- local -------------------------------------------------------------------

_local_model = None
LOCAL_MODEL = "nemo-parakeet-tdt-0.6b-v3"


def _decode(path: Path):
    import subprocess

    import numpy as np

    result = subprocess.run(
        [audio._require("ffmpeg"), "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, creationflags=config.NO_WINDOW,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't decode {path.name}: {result.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768


def _local(path: Path, terms: list[str]) -> dict:
    """NVIDIA Parakeet (ONNX) on this machine's CPU, ~10x faster than real time on native ARM64.
    It takes no spelling hints, so the glossary only helps at the summary step."""
    global _local_model
    try:
        import onnx_asr
    except ImportError as exc:
        raise SystemExit("onnx-asr is not installed. See Setup in README.md.") from exc
    name = config.get("TRANSCRIBER_LOCAL_MODEL", LOCAL_MODEL)
    if _local_model is None:
        import onnxruntime as ort

        # Disable spin-waiting to leave CPU time for recording.
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(config.get("TRANSCRIBER_LOCAL_THREADS", "6"))
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        # First use downloads the model (~700 MB) to the Hugging Face cache.
        model = onnx_asr.load_model(name, quantization=config.get("TRANSCRIBER_LOCAL_QUANT", "int8") or None,
                                    sess_options=options)
        _local_model = model.with_vad(onnx_asr.load_vad("silero", sess_options=options))
    segments = _local_model.recognize(_decode(path), sample_rate=16000)
    return {"model": f"local:{name}",
            "segments": [{"start": float(s.start), "end": float(s.end), "speaker": None, "text": s.text.strip()}
                         for s in segments if s.text.strip()]}


_BACKEND_FN = {"openai-diarize": _openai_diarize, "openai": _openai_plain, "local": _local}


# ---- local whisper (whisper.cpp) -------------------------------------------------

WHISPER_DIR = config.ROOT / "tools" / "whisper"
WHISPER_BIN_URL = "https://github.com/ggml-org/whisper.cpp/releases/download/b5130/whisper-bin-win-cpu-arm64.zip"
WHISPER_MODEL = "ggml-large-v3-turbo-q8_0.bin"
WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name}"
WHISPER_VAD = "ggml-silero-v5.1.2.bin"
WHISPER_VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/" + WHISPER_VAD


def _download(url: str, dest: Path) -> None:
    """Download to dest via a .part file; an interrupted download resumes where it stopped next time."""
    import httpx

    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    try:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120) as response:
            if response.status_code == 416:  # .part is already complete
                total = have
            else:
                response.raise_for_status()
                if response.status_code != 206:
                    have = 0  # server ignored the range: start over
                total = have + int(response.headers.get("content-length", 0))
                with tmp.open("ab" if have else "wb") as f:
                    for block in response.iter_bytes(1 << 20):
                        f.write(block)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Couldn't download {url} ({exc}). The first run of a local model needs "
                           "internet to fetch it; run again to resume. After that it works offline.") from exc
    if total and tmp.stat().st_size != total:
        raise RuntimeError(f"Download of {dest.name} stopped early ({tmp.stat().st_size} of {total} bytes). "
                           "Run again to resume.")
    tmp.replace(dest)


def _whisper_files() -> tuple[Path, Path, Path]:
    """whisper-cli.exe, the model and the VAD model under tools/whisper, downloaded on first use."""
    import zipfile

    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    cli = WHISPER_DIR / "bin" / "Release" / "whisper-cli.exe"
    if not cli.exists():
        archive = WHISPER_DIR / "bin.zip"
        _download(WHISPER_BIN_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(WHISPER_DIR / "bin")
        archive.unlink()
    name = config.get("TRANSCRIBER_WHISPER_MODEL", WHISPER_MODEL)
    model = WHISPER_DIR / name
    if not model.exists():
        _download(WHISPER_MODEL_URL.format(name=name), model)
    vad = WHISPER_DIR / WHISPER_VAD
    if not vad.exists():
        _download(WHISPER_VAD_URL, vad)
    return cli, model, vad


def _whisper(path: Path, terms: list[str]) -> dict:
    """OpenAI's Whisper (large-v3-turbo) via whisper.cpp on this machine, ~8x faster than real time.
    Glossary names go in as the prompt, which nudges spelling of project names and terms. Voice
    activity detection skips silence, which is where Whisper tends to invent text."""
    import subprocess

    cli, model, vad = _whisper_files()
    wav = path.with_suffix(".whisper-in.wav")
    out = path.with_suffix(".whisper-out")
    decoded = subprocess.run([audio._require("ffmpeg"), "-v", "error", "-y", "-i", str(path), "-ac", "1",
                              "-ar", "16000", str(wav)], capture_output=True, creationflags=config.NO_WINDOW)
    if decoded.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't decode {path.name}: {decoded.stderr.decode(errors='replace')[-300:]}")
    args = [str(cli), "-m", str(model), "-f", str(wav), "-l", config.get("TRANSCRIBER_LANGUAGE", "en") or "auto",
            "-t", config.get("TRANSCRIBER_WHISPER_THREADS", "8"), "--vad", "-vm", str(vad),
            "-oj", "-of", str(out), "-np"]
    if terms:
        # Whisper reads at most ~224 prompt tokens; ~700 characters of names stays under that.
        args += ["--prompt", glossary_hint_from(terms, limit=700), "--carry-initial-prompt"]
    try:
        result = subprocess.run(args, capture_output=True, creationflags=config.NO_WINDOW)
        json_file = Path(f"{out}.json")
        if result.returncode != 0 or not json_file.exists():
            raise RuntimeError(f"whisper failed ({result.returncode}): "
                               f"{result.stderr.decode(errors='replace')[-500:]}")
        data = json.loads(json_file.read_text(encoding="utf-8", errors="replace"))
        json_file.unlink()
    finally:
        wav.unlink(missing_ok=True)
    segments = [{"start": s["offsets"]["from"] / 1000, "end": s["offsets"]["to"] / 1000, "speaker": None,
                 "text": s["text"].strip()}
                for s in data.get("transcription", []) if s.get("text", "").strip()]
    return {"model": f"whisper:{model.stem.removeprefix('ggml-')}", "segments": segments}


_BACKEND_FN["whisper"] = _whisper


# ---- per part ------------------------------------------------------------------

def transcribe_part(session: Session, part: Path, backend: str, echo=print) -> list[dict]:
    """Transcribe one part, reusing any chunk already done. Returns part-relative segments."""
    if backend not in _BACKEND_FN:
        raise SystemExit(f"Unknown ASR backend '{backend}'. Choose from: {', '.join(BACKENDS)}")
    terms = glossary_terms(session.profile_file("glossary.md"))
    chunks = audio.make_chunks(part, session.path / "work" / part.stem)
    segments: list[dict] = []
    for index, (chunk, offset) in enumerate(chunks):
        cache = chunk.with_suffix(f".{backend}.json")
        if cache.exists():
            result = json.loads(cache.read_text(encoding="utf-8"))
        else:
            minutes = audio.duration(chunk) / 60
            echo(f"  transcribing {part.name} chunk {index + 1}/{len(chunks)} ({minutes:.1f} min, {backend})")
            began = time.monotonic()
            result = _BACKEND_FN[backend](chunk, terms)
            cache.write_text(json.dumps(result, indent=1), encoding="utf-8")
            model = result["model"]
            cost = round(minutes * config.ASR_PRICE_PER_MIN.get(model, 0.0), 4)
            session.log(event="transcribed", part=part.name, chunk=index, model=model,
                        audio_min=round(minutes, 2), seconds=round(time.monotonic() - began, 1), est_cost_usd=cost)
        for seg in result["segments"]:
            speaker = seg.get("speaker")
            segments.append({
                "start": round(offset + seg["start"], 2),
                "end": round(offset + seg["end"], 2),
                # Diarization labels are only consistent inside one chunk, so tag them with the chunk.
                "speaker": f"{speaker}{index + 1}" if speaker else None,
                "text": seg["text"],
            })
    return segments


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def to_markdown(segments: list[dict], offset: float, paragraph_sec: float = 45) -> str:
    """Readable transcript with session-time stamps. Consecutive lines from one speaker are merged;
    lines without speakers (local / plain OpenAI) are grouped into ~45-second paragraphs."""
    lines: list[str] = []
    last_speaker = object()
    para_start = 0.0
    for seg in segments:
        if not seg["text"]:
            continue
        stamp = clock(offset + seg["start"])
        who = seg["speaker"]
        same = who == last_speaker and (who is not None or seg["start"] - para_start < paragraph_sec)
        if lines and same:
            lines[-1] += " " + seg["text"]
        else:
            lines.append(f"[{stamp}] {who + ': ' if who else ''}{seg['text']}")
            para_start = seg["start"]
        last_speaker = who
    return "\n\n".join(lines)


def write_transcript(session: Session, part: Path, segments: list[dict], offset: float) -> None:
    jsonl = session.transcript_path(part, ".jsonl")
    jsonl.write_text("".join(json.dumps(s) + "\n" for s in segments), encoding="utf-8")
    session.transcript_path(part, ".md").write_text(
        f"# Transcript: {part.name}\n\n{to_markdown(segments, offset)}\n", encoding="utf-8")


def load_segments(session: Session, part: Path) -> list[dict] | None:
    path = session.transcript_path(part, ".jsonl")
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
