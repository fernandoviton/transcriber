"""LLM summaries: one per part, plus a whole-session recap.

Backends (TRANSCRIBER_SUMMARIZER or `--summarizer`):
  local    llama.cpp's llama-server on this machine with a GGUF model (default Qwen3-30B-A3B-Instruct,
           downloaded to tools/llama on first use). Free, private, works offline. TRANSCRIBER_LLM_MODEL.
  copilot  pipes the prompt into `copilot -s` (GitHub Copilot CLI). No API key needed.
           Optional: TRANSCRIBER_COPILOT_MODEL, TRANSCRIBER_COPILOT_AGENT.
  openai   OpenAI chat completions. Needs OPENAI_API_KEY. Model: TRANSCRIBER_SUMMARY_MODEL.
"""

from __future__ import annotations

import atexit
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

from . import config
from .session import Session
from .transcribe import _download, clock, to_markdown

NO_TOOLS = ("Reply with the finished Markdown only. Do not use tools, read or write files, "
            "or add any preamble or closing remarks.")


def _copilot(prompt: str, workdir: Path) -> str:
    exe = shutil.which("copilot")
    if not exe:
        raise SystemExit("GitHub Copilot CLI (`copilot`) not found. Install it or use --summarizer openai.")
    cmd = [exe, "-s", "--disable-builtin-mcps", "--stream", "off"]
    if model := config.get("TRANSCRIBER_COPILOT_MODEL"):
        cmd += ["--model", model]
    if agent := config.get("TRANSCRIBER_COPILOT_AGENT"):
        cmd += ["--agent", agent]
    # Run in an empty folder so the CLI doesn't pick up this repo's files or instructions.
    workdir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                            cwd=workdir, timeout=1800, creationflags=config.NO_WINDOW)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"copilot failed ({result.returncode}): {(result.stderr or result.stdout)[-800:]}\n"
                           "Copilot summaries need internet. Transcripts are still saved; summarize later, "
                           "or switch the summarizer to local (Settings…).")
    return result.stdout.strip()


def _openai(prompt: str, workdir: Path) -> str:
    import httpx

    try:
        return _openai_raw(prompt, workdir)
    except httpx.TransportError as exc:
        raise RuntimeError(f"Can't reach OpenAI ({exc.__class__.__name__}). No internet? Transcripts are "
                           "saved; summarize later.") from exc


def _openai_raw(prompt: str, workdir: Path) -> str:
    import httpx

    key = config.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env). Or use --summarizer copilot.")
    base = config.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = config.get("TRANSCRIBER_SUMMARY_MODEL", "gpt-5-mini")
    response = httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                          timeout=900)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI summary failed ({response.status_code}): {response.text[:500]}")
    return response.json()["choices"][0]["message"]["content"].strip()


_BACKEND_FN = {"copilot": _copilot, "openai": _openai}


# ---- local (llama.cpp) -------------------------------------------------------------

LLAMA_DIR = config.ROOT / "tools" / "llama"
LLAMA_BIN_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11321/llama-b11321-bin-win-cpu-arm64.zip"
# Mixture-of-experts: 30B parameters of knowledge, ~3B active per word, so it runs like a small model.
# Q4_0 is the 4-bit format llama.cpp speeds up most on ARM CPUs.
LLAMA_MODEL = "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
_server: tuple[subprocess.Popen, int] | None = None


def _llama_files() -> tuple[Path, Path]:
    """llama-server.exe and the model under tools/llama, downloaded on first use.
    TRANSCRIBER_LLM_MODEL: a .gguf file name from an unsloth/<name>-GGUF repo, or a full URL."""
    import zipfile

    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    exe = next(LLAMA_DIR.glob("bin/**/llama-server.exe"), None)
    if exe is None:
        archive = LLAMA_DIR / "bin.zip"
        _download(LLAMA_BIN_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(LLAMA_DIR / "bin")
        archive.unlink()
        exe = next(LLAMA_DIR.glob("bin/**/llama-server.exe"))
    choice = config.get("TRANSCRIBER_LLM_MODEL", LLAMA_MODEL)
    name = choice.rsplit("/", 1)[-1]
    model = LLAMA_DIR / name
    if not model.exists():
        repo = re.sub(r"-(UD-)?(I?Q\d\w*|BF16|F16)\.gguf$", "-GGUF", name)
        _download(choice if choice.startswith("http")
                  else f"https://huggingface.co/unsloth/{repo}/resolve/main/{name}", model)
    return exe, model


def _llama_port() -> int:
    """Start llama-server (once per run; the model takes a while to load) and return its port."""
    global _server
    import httpx

    if _server and _server[0].poll() is None:
        return _server[1]
    exe, model = _llama_files()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = LLAMA_DIR / "server.log"
    process = subprocess.Popen(
        [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "--jinja",
         "-c", config.get("TRANSCRIBER_LLM_CONTEXT", "32768"), "-np", "1", "-fa", "on",
         # Separate inference and batch thread counts allow tuning for the available CPU.
         "-t", config.get("TRANSCRIBER_LLM_THREADS", "6"),
         "-tb", config.get("TRANSCRIBER_LLM_BATCH_THREADS", "8")],
        stdout=subprocess.DEVNULL, stderr=log.open("w", encoding="utf-8"), creationflags=config.NO_WINDOW)
    _server = (process, port)
    atexit.register(stop_local)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _server = None
            tail = log.read_text(encoding="utf-8", errors="replace")[-800:]
            raise RuntimeError(f"The local summarizer (llama-server) didn't start:\n{tail}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                return port
        except httpx.HTTPError:
            pass
        time.sleep(1)
    stop_local()
    raise RuntimeError(f"The local summarizer took over 10 minutes to load. See {log}")


def stop_local() -> None:
    """Shut down llama-server to give its memory back (the model uses ~17 GB)."""
    global _server
    if _server:
        process, _ = _server
        _server = None
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def _local(prompt: str, workdir: Path) -> str:
    import httpx

    port = _llama_port()
    response = httpx.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}],
              # Qwen's recommended settings for its instruct models.
              "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 4000},
        timeout=3600)
    if response.status_code >= 400:
        detail = response.text[:500]
        if "context" in detail.lower():
            detail += ("\nThe transcript is too long for the local model's memory window. Use shorter "
                       "parts, or raise TRANSCRIBER_LLM_CONTEXT (uses more RAM).")
        raise RuntimeError(f"Local summary failed ({response.status_code}): {detail}")
    return response.json()["choices"][0]["message"]["content"].strip()


_BACKEND_FN = {"local": _local, "copilot": _copilot, "openai": _openai}


def run_llm(session: Session, prompt: str, backend: str, what: str) -> str:
    if backend not in _BACKEND_FN:
        raise SystemExit(f"Unknown summarizer '{backend}'. Choose from: {', '.join(_BACKEND_FN)}")
    began = time.monotonic()
    text = _BACKEND_FN[backend](prompt, session.path / "work" / "llm")
    session.log(event="summarized", what=what, backend=backend, prompt_chars=len(prompt),
                seconds=round(time.monotonic() - began, 1))
    return _strip_fence(text)


def _strip_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def _glossary_block(session: Session) -> str:
    glossary = session.profile_file("glossary.md")
    return f"## Glossary (correct spellings; the transcript may mishear these)\n\n{glossary}\n\n" if glossary else ""


NOTHING = "- (nothing)"
SPEAKER_NOTE = ("Speaker tags like A1 or B3 come from automatic speaker detection. They reset every 10 minutes "
                "and are often wrong, so don't rely on them for who said what. Lines with no tag have no speaker info.")


def part_prompt(session: Session, index: int, segments: list[dict], offset: float, end: float,
                earlier: list[str], notes: list[str] | None = None) -> str:
    context = "\n\n".join(earlier) if earlier else "(This is the first part of the session.)"
    if notes:
        stretches = ", ".join(n.splitlines()[0].removeprefix("### ") for n in notes)
        source = ("## Notes on this part, in time order\n\nA helper read the transcript a few minutes at a "
                  "time and wrote these notes.\n\n" + "\n\n".join(notes) + "\n\n"
                  f"## Your task\n\nWrite the summary for part {index + 1} following the instructions at the "
                  f"top. It must cover the WHOLE part: all {len(notes)} stretches of notes ({stretches}), "
                  "not just the last one. Combine related notes into one bullet and drop minor details. "
                  "Never mention off-topic chat (table talk, small talk, food, drinks, breaks, books, TV), "
                  "not even to say it happened.")
    else:
        source = f"## Transcript\n\n{SPEAKER_NOTE}\n\n{to_markdown(segments, offset)}"
    return (
        f"{session.profile_file('summarize.md')}\n\n"
        f"{_glossary_block(session)}"
        f"## Earlier in this session (context only; do not repeat it)\n\n{context}\n\n"
        f"# Part {index + 1}, session time {clock(offset)}-{clock(end)}\n\n"
        f"{source}\n\n"
        f"---\n{NO_TOOLS}\n"
    )


def notes_prompt(session: Session, segments: list[dict], offset: float, start: float, stop: float) -> str:
    return (
        "You are taking notes on a long recording so someone can summarize it later. Their summary "
        "instructions are below so you know what matters.\n\n"
        f"## Their summary instructions (context only)\n\n{session.profile_file('summarize.md')}\n\n"
        f"{_glossary_block(session)}"
        f"## Transcript stretch, session time {clock(start)}-{clock(stop)}\n\n{SPEAKER_NOTE}\n\n"
        f"{to_markdown(segments, offset)}\n\n"
        "## Your task\n\n"
        "Don't write the summary. Write plain notes on everything in this stretch that the instructions say "
        "matters: events, names, decisions, items, open questions. Use up to 12 short bullets in time order, "
        "each ending with a rough timestamp like (~0:12). No headline, no headings, no open-threads section: "
        "bullets only. Only state what the transcript supports; mark guesses with 'unclear:'.\n\n"
        "Skip off-topic chat completely (table talk, small talk, jokes, food, drinks, breaks, books, TV, "
        "other games): don't describe it and don't mention that it happened. If nothing in this stretch "
        f"matters, reply with exactly: {NOTHING}\n\n"
        f"---\n{NO_TOOLS}\n"
    )


def summarize_part(session: Session, index: int, segments: list[dict], offset: float, end: float,
                   earlier: list[str], backend: str, what: str, echo=print) -> str:
    """Cloud models read a whole part at once. The local model misses the start of long transcripts
    and slows down a lot as the input grows, so it first takes notes in short chunks, then summarizes those."""
    chunk = float(config.get("TRANSCRIBER_LLM_CHUNK_MINUTES", "10")) * 60
    if backend != "local" or chunk <= 0 or end - offset <= chunk * 1.5:
        return run_llm(session, part_prompt(session, index, segments, offset, end, earlier), backend, what)
    notes = []
    groups: list[list[dict]] = [[]]
    for seg in (s for s in segments if s["text"]):
        if groups[-1] and seg["start"] - groups[-1][0]["start"] >= chunk and end - offset - seg["start"] > chunk / 2:
            groups.append([])
        groups[-1].append(seg)
    for piece in filter(None, groups):
        start, stop = offset + piece[0]["start"], offset + piece[-1]["end"]
        echo(f"    notes {clock(start)}-{clock(stop)}")
        text = run_llm(session, notes_prompt(session, piece, offset, start, stop), backend,
                       f"{what} notes {clock(start)}")
        if not ("(nothing)" in text and len(text) < 40):  # skip stretches with nothing worth noting
            notes.append(f"### {clock(start)}-{clock(stop)}\n\n{text}")
    notes_file = session.path / "work" / "notes" / f"{Path(what).stem}.md"
    notes_file.parent.mkdir(parents=True, exist_ok=True)
    notes_file.write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    if not notes:
        return "_No relevant discussion in this part._"
    return run_llm(session, part_prompt(session, index, segments, offset, end, earlier, notes), backend, what)


def recap_prompt(session: Session, part_summaries: list[str]) -> str:
    joined = "\n\n".join(part_summaries)
    return (
        f"{session.profile_file('recap.md')}\n\n"
        f"{_glossary_block(session)}"
        f"## Part summaries, in order\n\n{joined}\n\n"
        f"---\n{NO_TOOLS}\n"
    )
