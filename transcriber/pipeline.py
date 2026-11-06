"""Bring a session up to date: transcribe new parts, summarize them, rebuild session.md."""

from __future__ import annotations

import threading
from pathlib import Path
from datetime import datetime

from . import audio, config
from .session import Session
from .summarize import recap_prompt, run_llm, stop_local, summarize_part
from .transcribe import clock, load_segments, transcribe_part, write_transcript

_lock = threading.Lock()  # one pipeline run at a time (recording triggers runs in the background)


def process(session: Session, asr: str | None = None, summarizer: str | None = None,
            final: bool = False, redo_summaries: bool = False, summarize: bool | None = None,
            redo_parts: set[str] | None = None, echo=print) -> None:
    """Transcribe new parts; summarize them if `summarize` (default: asked for a recap / redo, or the
    session's summaries are 'auto'); with `final`, also write the whole-session recap.
    `redo_parts` (file names or stems) re-summarizes just those parts."""
    if summarize is None:
        summarize = final or redo_summaries or bool(redo_parts) or session.summarize_automatically
    with _lock:
        try:
            _process(session, asr, summarizer, final, redo_summaries, summarize, redo_parts or set(), echo)
        finally:
            stop_local()  # free the local model's memory between runs


def _process(session: Session, asr: str | None, summarizer: str | None, final: bool,
             redo_summaries: bool, summarize: bool, redo_parts: set[str], echo) -> None:
    asr = asr or session.config["asr"]
    summarizer = summarizer or session.config["summarizer"]
    config.check_ready(asr, summarizer if summarize else None)
    parts = session.parts()
    if not parts:
        echo("No audio parts yet.")
        return
    unknown = redo_parts - {p.name for p in parts} - {p.stem for p in parts}
    if unknown:
        raise SystemExit(f"No such part: {', '.join(sorted(unknown))}")
    parts, durations = readable_parts(session, echo)
    offsets = [sum(durations[:i]) for i in range(len(parts))]
    summaries: list[str] = []
    for index, part in enumerate(parts):
        offset, end = offsets[index], offsets[index] + durations[index]
        segments = load_segments(session, part)
        if segments is None:
            try:
                segments = transcribe_part(session, part, asr, echo)
            except SystemExit:
                raise  # setup problems (missing tool/model) affect every part
            except Exception as exc:
                echo(f"  skipping {part.name}: transcription failed ({exc})")
                continue
            write_transcript(session, part, segments, offset)
            echo(f"  transcript ready: transcript/{part.stem}.md")
        if not summarize:
            continue
        summary_file = session.summary_path(part)
        if redo_summaries or part.name in redo_parts or part.stem in redo_parts or not summary_file.exists():
            heading = f"## Part {index + 1} ({clock(offset)}-{clock(end)})"
            if not any(s["text"] for s in segments):
                body = "_No speech detected._"
            else:
                echo(f"  summarizing {part.name} ({summarizer})")
                body = summarize_part(session, index, segments, offset, end, summaries,
                                      summarizer, what=part.name, echo=echo)
            summary_file.write_text(f"{heading}\n\n{body}\n", encoding="utf-8")
            write_session_md(session)
            echo(f"  summary ready: summary/{part.stem}.md")
        summaries.append(summary_file.read_text(encoding="utf-8").strip())
    if final and summarize and summarizer == "local":
        echo("  (no session recap: the local model is good at single parts but not whole sessions; "
             "write one yourself or use a cloud summarizer)")
    elif final and summarize:
        echo(f"  writing session recap ({summarizer})")
        recap = run_llm(session, recap_prompt(session, summaries), summarizer, what="recap")
        (session.path / "recap.md").write_text(recap + "\n", encoding="utf-8")
    write_session_md(session)
    if summarize:
        echo(f"Up to date: {session.path / 'session.md'}")
    else:
        echo("Transcripts up to date. Summaries are set to 'later': click Summarize + recap "
             "(or run `summarize`) when you're ready.")


def readable_parts(session: Session, echo=None) -> tuple[list[Path], list[float]]:
    """Parts ffprobe can read, with their durations. Broken files (e.g. a recording cut off by a
    crash) are skipped with a warning so they don't stop the rest of the session."""
    parts, durations = [], []
    for part in session.parts():
        try:
            durations.append(audio.duration(part))
            parts.append(part)
        except RuntimeError as exc:
            if echo:
                echo(f"  skipping {part.name}: {exc}")
    return parts, durations


def write_session_md(session: Session) -> None:
    parts, durations = readable_parts(session)
    total = sum(durations)
    recap_file = session.path / "recap.md"
    recap = (f"## Session recap\n\n{recap_file.read_text(encoding='utf-8').strip()}\n\n"
             if recap_file.exists() else "")
    summaries = [session.summary_path(p).read_text(encoding="utf-8").strip()
                 for p in parts if session.summary_path(p).exists()]
    body = "\n\n".join(summaries) or "_No parts summarized yet._"
    text = (
        f"# {session.name}\n\n"
        f"_Profile: {session.config['profile']} · {len(parts)} part(s) · {clock(total)} of audio · "
        f"updated {datetime.now():%Y-%m-%d %H:%M}. Full transcripts are in `transcript/`._\n\n"
        f"{recap}"
        f"# By part\n\n{body}\n"
    )
    (session.path / "session.md").write_text(text, encoding="utf-8")
