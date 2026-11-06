"""Command line: py -m transcriber <command> ..."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from . import audio, config
from .pipeline import process
from .session import AUDIO_EXT, Session
from .transcribe import BACKENDS, clock


ASR_HELP = {
    "whisper": "this laptop (Whisper): free, private, uses glossary names as hints, ~8x real time",
    "local": "this laptop (Parakeet): free, private, no glossary hints, ~8x real time",
    "openai-diarize": "OpenAI, with timestamps + speaker labels (~$0.36/audio hour)",
    "openai": "OpenAI gpt-transcribe, uses glossary names as hints, no timestamps",
}
SUMMARIZER_HELP = {
    "local": "this laptop (Qwen3 30B via llama.cpp): free, private, offline, slower",
    "copilot": "GitHub Copilot CLI (no API key, needs internet)",
    "openai": "OpenAI chat model (uses OPENAI_API_KEY)",
}
SUMMARIES_HELP = {
    "later": "only transcribe; summarize when you ask",
    "auto": "summarize each part as soon as it's transcribed",
}


def _ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"{question}{suffix}: ").strip()
        except EOFError:
            raise SystemExit(f"\nNeed an answer for '{question}' (or pass it as an option).")
        if answer or default:
            return answer or default


def _choose(question: str, options: list[str], default: str, help_text: dict | None = None) -> str:
    print(f"\n{question}")
    for number, option in enumerate(options, 1):
        detail = f"  - {help_text[option]}" if help_text and option in help_text else ""
        print(f"  {number}. {option}{' (default)' if option == default else ''}{detail}")
    while True:
        answer = _ask("Pick a number or name", default)
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer in options:
            return answer
        print(f"  '{answer}' isn't one of the options.")


def _profile_summary(profile: str) -> str:
    """First heading of the profile's summarize.md, e.g. 'summarize one part of a meeting'."""
    path = config.PROFILES / profile / "summarize.md"
    first = path.read_text(encoding="utf-8").splitlines()[0] if path.is_file() else ""
    return first.lstrip("# ").removeprefix("Instructions: ").strip()


def cmd_new(args) -> None:
    interactive = sys.stdin.isatty()
    if not args.name:
        if not interactive:
            raise SystemExit("Give the session a name: new \"<name>\"")
        args.name = _ask("Session name (e.g. weekly planning)")
    if not args.profile:
        profiles = sorted(p.name for p in config.PROFILES.iterdir() if p.is_dir())
        if not profiles:
            raise SystemExit(f"No profiles in {config.PROFILES}")
        default = "meeting" if "meeting" in profiles else profiles[0]
        args.profile = (_choose("Profile (what to listen for and how to summarize):", profiles, default,
                                {p: _profile_summary(p) for p in profiles})
                        if interactive else default)
    if not args.asr:
        args.asr = (_choose("Transcription:", list(config.ASR_BACKENDS), config.default_asr(), ASR_HELP)
                    if interactive else config.default_asr())
    if not args.summarizer:
        args.summarizer = (_choose("Summarizer:", list(config.SUMMARIZERS), config.default_summarizer(),
                                   SUMMARIZER_HELP)
                           if interactive else config.default_summarizer())
    if not args.summaries:
        args.summaries = (_choose("Summaries:", list(config.SUMMARY_MODES), config.default_summaries(),
                                  SUMMARIES_HELP)
                          if interactive else config.default_summaries())

    session = Session.create(args.name, args.profile, args.asr, args.summarizer, args.summaries)
    print(f"\nCreated {session.path}")
    print(f"  profile={args.profile}  asr={args.asr}  summarizer={args.summarizer}  summaries={args.summaries}")
    print(f"\nNext: drop audio files into\n  {session.audio_dir}\n"
          "(they're processed in file-name order), then run:\n"
          f"  py -m transcriber process \"{session.path.name}\"     (transcribe)\n"
          f"  py -m transcriber summarize \"{session.path.name}\"   (summaries + recap)\n"
          f"Or record live: py -m transcriber record \"{session.path.name}\"")
    if not args.no_open and hasattr(os, "startfile"):
        os.startfile(session.audio_dir)


def cmd_record(args) -> None:
    from .record import record

    session = Session.open(args.session)
    record(session, device=args.device, part_minutes=args.part_minutes, auto_process=not args.no_process,
           asr=args.asr, summarizer=args.summarizer)


def cmd_import(args) -> None:
    session = Session.open(args.session)
    if not args.no_process:
        summarizer = args.summarizer or session.config["summarizer"]
        config.check_ready(args.asr or session.config["asr"],
                           summarizer if args.final or session.summarize_automatically else None)
    sources = [Path(f) for f in args.files]
    for src in sources:
        if not src.is_file():
            raise SystemExit(f"File not found: {src}")
        if src.suffix.lower() not in AUDIO_EXT:
            raise SystemExit(f"Not an audio file I know: {src}")
    for src in sources:
        dest = session.audio_dir / session.next_part_name(src.suffix, src.name)
        shutil.copy2(src, dest)
        print(f"Imported {src.name} -> audio/{dest.name} ({clock(audio.duration(dest))})")
    if not args.no_process:
        process(session, asr=args.asr, summarizer=args.summarizer, final=args.final)


def cmd_process(args) -> None:
    session = Session.open(args.session)
    process(session, asr=args.asr, summarizer=args.summarizer, final=args.final,
            redo_summaries=args.redo_summaries)


def cmd_summarize(args) -> None:
    session = Session.open(args.session)
    process(session, asr=args.asr, summarizer=args.summarizer, final=not args.part, summarize=True,
            redo_summaries=args.redo, redo_parts=set(args.part or ()))


def cmd_set(args) -> None:
    session = Session.open(args.session)
    session.update(profile=args.profile, asr=args.asr, summarizer=args.summarizer, summaries=args.summaries)
    cfg = session.config
    print(f"{session.path.name}: profile={cfg['profile']}  asr={cfg['asr']}  "
          f"summarizer={cfg['summarizer']}  summaries={cfg['summaries']}")
    if args.asr:
        print("  (parts already transcribed keep their transcript; new parts use the new setting)")


def cmd_status(args) -> None:
    session = Session.open(args.session)
    cfg = session.config
    print(f"{session.path}\n  profile={cfg['profile']}  asr={cfg['asr']}  summarizer={cfg['summarizer']}"
          f"  summaries={cfg['summaries']}\n")
    total = 0.0
    for part in session.parts():
        try:
            length = audio.duration(part)
        except RuntimeError:
            print(f"  {part.name:40s} {'unreadable':>8s}  (skipped)")
            continue
        total += length
        done_t = "transcribed" if session.transcript_path(part).exists() else "-"
        done_s = "summarized" if session.summary_path(part).exists() else "-"
        print(f"  {part.name:40s} {clock(length):>8s}  {done_t:12s} {done_s}")
    events = session.events()
    cost = sum(e.get("est_cost_usd", 0) for e in events)
    minutes = sum(e.get("audio_min", 0) for e in events if e.get("event") == "transcribed")
    print(f"\n  audio {clock(total)} · transcribed {minutes:.0f} min · est. transcription cost ${cost:.2f}")
    print(f"  recap: {'yes' if (session.path / 'recap.md').exists() else 'not yet'}")


def cmd_devices(args) -> None:
    from .record import list_devices

    list_devices()


def cmd_gui(args) -> None:
    from .gui import main as gui_main

    gui_main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcriber", description="Record, transcribe and summarize long sessions.")
    sub = parser.add_subparsers(dest="command", required=True)

    def backends(p):
        p.add_argument("--asr", choices=BACKENDS, help="speech-to-text backend (default: the session's)")
        p.add_argument("--summarizer", choices=config.SUMMARIZERS, help="LLM backend (default: the session's)")

    p = sub.add_parser("new", help="create a session (asks for anything you don't pass)")
    p.add_argument("name", nargs="?")
    p.add_argument("--profile", help="folder under profiles/")
    p.add_argument("--no-open", action="store_true", help="don't open the audio folder in Explorer")
    p.add_argument("--summaries", choices=config.SUMMARY_MODES,
                   help="later: only transcribe until you run `summarize`; auto: summarize as you go")
    backends(p)
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("record", help="record live; processes each part as it finishes")
    p.add_argument("session", nargs="?", help="session folder or part of its name (default: latest)")
    p.add_argument("--device", help="input device number or name (see `devices`)")
    p.add_argument("--part-minutes", type=float, default=60, help="minutes per part (default: 60)")
    p.add_argument("--no-process", action="store_true", help="only record")
    backends(p)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("import", help="add recorded files (e.g. iPhone voice memos) as parts")
    p.add_argument("files", nargs="+")
    p.add_argument("--session", help="session folder or part of its name (default: latest)")
    p.add_argument("--no-process", action="store_true")
    p.add_argument("--final", action="store_true", help="also write the session recap")
    backends(p)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("process", help="transcribe anything not done yet (and summarize if summaries=auto)")
    p.add_argument("session", nargs="?")
    p.add_argument("--final", action="store_true", help="also summarize and write the whole-session recap")
    p.add_argument("--redo-summaries", action="store_true", help="re-summarize every part (e.g. after editing the profile)")
    backends(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("summarize", help="transcribe what's left, summarize every part, write the recap")
    p.add_argument("session", nargs="?")
    p.add_argument("--redo", action="store_true", help="re-summarize parts that already have a summary")
    p.add_argument("--part", action="append", metavar="NAME",
                   help="re-summarize just this part (e.g. part-03); repeatable; skips the recap")
    backends(p)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("set", help="change a session's settings, e.g. set test-mic --asr local")
    p.add_argument("session", nargs="?")
    p.add_argument("--profile")
    p.add_argument("--summaries", choices=config.SUMMARY_MODES)
    backends(p)
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("status", help="show parts, progress and estimated cost")
    p.add_argument("session", nargs="?")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("devices", help="list microphones")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("gui", help="open the Transcriber window")
    p.set_defaults(func=cmd_gui)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        args.func(args)
    except RuntimeError as exc:  # backend failures: show the message, not a traceback
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
