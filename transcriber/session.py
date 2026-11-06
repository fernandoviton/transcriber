"""A session is a folder under sessions/ holding audio parts and everything derived from them.

    sessions/<name>/
      session.json          profile, backends, summaries (auto|later), created time
      audio/                one file per part (recorded hour or imported file), in name order
      work/<part>/          10-minute chunks + cached per-chunk transcription results
      transcript/<part>.jsonl + .md
      summary/<part>.md     summary of that part
      recap.md              whole-session recap (written by `summarize`)
      session.md            the file to read: recap + every part summary
      log.jsonl             what ran, how long, estimated cost
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from . import config

AUDIO_EXT = {".wav", ".m4a", ".mp3", ".mp4", ".aac", ".flac", ".ogg", ".webm", ".caf"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "session"


def check_profile(profile: str) -> None:
    """A profile needs all its files; the instruction files can't be empty (the glossary can)."""
    folder = config.PROFILES / profile
    if not folder.is_dir():
        known = ", ".join(sorted(p.name for p in config.PROFILES.iterdir() if p.is_dir()))
        raise SystemExit(f"Profile '{profile}' not found in {config.PROFILES}. Known: {known}")
    problems = []
    for name in config.REQUIRED_PROFILE_FILES:
        path = folder / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif name != "glossary.md" and not path.read_text(encoding="utf-8").strip():
            problems.append(f"{name} is empty")
    if problems:
        raise SystemExit(f"Profile '{profile}' ({folder}) is incomplete: {'; '.join(problems)}")


class Session:
    REQUIRED_KEYS = ("name", "profile", "asr", "summarizer")

    def __init__(self, path: Path) -> None:
        self.path = path
        settings = path / "session.json"
        if not settings.is_file():
            raise SystemExit(f"Not a session folder (no session.json): {path}")
        try:
            self.config: dict = json.loads(settings.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{settings} is not valid JSON: {exc}") from exc
        missing = [k for k in self.REQUIRED_KEYS if not self.config.get(k)]
        if missing:
            raise SystemExit(f"{settings} is missing required setting(s): {', '.join(missing)}")
        if self.config["asr"] not in config.ASR_BACKENDS:
            raise SystemExit(f"{settings}: unknown asr '{self.config['asr']}' "
                             f"(choose from {', '.join(config.ASR_BACKENDS)})")
        if self.config["summarizer"] not in config.SUMMARIZERS:
            raise SystemExit(f"{settings}: unknown summarizer '{self.config['summarizer']}' "
                             f"(choose from {', '.join(config.SUMMARIZERS)})")
        if self.config.setdefault("summaries", "auto") not in config.SUMMARY_MODES:  # older sessions: auto
            raise SystemExit(f"{settings}: unknown summaries '{self.config['summaries']}' "
                             f"(choose from {', '.join(config.SUMMARY_MODES)})")
        if not self.audio_dir.is_dir():
            raise SystemExit(f"Session is missing its audio folder: {self.audio_dir}")
        check_profile(self.config["profile"])
        for sub in ("work", "transcript", "summary"):  # derived output; safe to recreate
            (path / sub).mkdir(exist_ok=True)

    # ---- creation / lookup -------------------------------------------------

    @classmethod
    def create(cls, name: str, profile: str, asr: str, summarizer: str, summaries: str | None = None) -> "Session":
        summaries = summaries or config.default_summaries()
        check_profile(profile)
        config.check_ready(asr, summarizer if summaries == "auto" else None)
        folder = f"{datetime.now():%Y-%m-%d}-{slug(name)}"
        path = config.SESSIONS / folder
        if path.exists():
            raise SystemExit(f"Session already exists: {path}")
        for sub in ("audio", "work", "transcript", "summary"):
            (path / sub).mkdir(parents=True)
        data = {"name": name, "profile": profile, "asr": asr, "summarizer": summarizer, "summaries": summaries,
                "created": datetime.now().isoformat(timespec="seconds")}
        (path / "session.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        return cls(path)

    def update(self, **changes) -> None:
        """Change settings (profile, asr, summarizer, summaries) and save session.json."""
        new = {**self.config, **{k: v for k, v in changes.items() if v}}
        if new["asr"] not in config.ASR_BACKENDS or new["summarizer"] not in config.SUMMARIZERS \
                or new["summaries"] not in config.SUMMARY_MODES:
            raise SystemExit(f"Invalid settings: {changes}")
        check_profile(new["profile"])
        (self.path / "session.json").write_text(json.dumps(new, indent=2), encoding="utf-8")
        self.config = new

    @property
    def summarize_automatically(self) -> bool:
        return self.config["summaries"] == "auto"

    @classmethod
    def open(cls, ref: str | None) -> "Session":
        if ref and Path(ref).is_dir():
            return cls(Path(ref).resolve())  # an explicit folder must be a valid session
        candidates = sorted(
            (p for p in config.SESSIONS.glob("*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        if ref:
            candidates = [p for p in candidates if ref.lower() in p.name.lower()]
            if len(candidates) > 1:
                names = "\n  ".join(p.name for p in candidates)
                raise SystemExit(f"'{ref}' matches more than one session; be more specific:\n  {names}")
        if not candidates:
            raise SystemExit(f"No session found{f' matching {ref!r}' if ref else ''}. Create one with `new`.")
        session = cls(candidates[-1])
        if not ref:
            print(f"Using latest session: {session.path.name}", file=sys.stderr)
        return session

    # ---- paths -------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.config.get("name", self.path.name)

    @property
    def audio_dir(self) -> Path:
        return self.path / "audio"

    @property
    def profile_dir(self) -> Path:
        return config.PROFILES / self.config["profile"]

    def profile_file(self, name: str) -> str:
        """Read a profile file. Checked again here because it may have been deleted since opening."""
        path = self.profile_dir / name
        if not path.is_file():
            raise SystemExit(f"Profile '{self.config['profile']}' is missing {name}: {path}")
        return path.read_text(encoding="utf-8").strip()

    def parts(self) -> list[Path]:
        """Finished audio parts in natural name order ('Recording 9' before 'Recording 10').
        Files starting with '.' are still being written."""
        def natural(p: Path):
            return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())]
        return sorted((p for p in self.audio_dir.iterdir()
                       if p.suffix.lower() in AUDIO_EXT and not p.name.startswith(".")), key=natural)

    def next_part_name(self, suffix: str, original: str = "") -> str:
        number = len(self.parts()) + 1
        extra = f"-{slug(Path(original).stem)}" if original else ""
        return f"part-{number:02d}{extra}{suffix.lower()}"

    def transcript_path(self, part: Path, ext: str = ".jsonl") -> Path:
        return self.path / "transcript" / f"{part.stem}{ext}"

    def summary_path(self, part: Path) -> Path:
        return self.path / "summary" / f"{part.stem}.md"

    def log(self, **event) -> None:
        event = {"at": datetime.now().isoformat(timespec="seconds"), **event}
        with (self.path / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def events(self) -> list[dict]:
        path = self.path / "log.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
