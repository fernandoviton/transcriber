"""Live recording into rolling parts. Each finished part is processed in the background."""

from __future__ import annotations

import math
import queue
import threading
import time
import wave

import numpy as np

from . import config
from .pipeline import process
from .session import Session

RATE = 16000


def input_devices() -> list[tuple[int, str]]:
    """(index, name) for each microphone, default first. Skips duplicate names Windows lists per host API."""
    import sounddevice as sd

    default_in = sd.default.device[0]
    seen, result = set(), []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and dev["name"].strip() and dev["name"] not in seen:
            seen.add(dev["name"])
            result.append((index, dev["name"]))
    result.sort(key=lambda d: d[0] != default_in)
    return result


def list_devices() -> None:
    import sounddevice as sd

    default_in = sd.default.device[0]
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mark = "*" if index == default_in else " "
            print(f"{mark} {index:3d}  {dev['name']}  ({dev['max_input_channels']} ch)")
    print("\n* = default input. Use `record --device <number or part of the name>`.")


def _meter(dbfs: float) -> str:
    filled = max(0, min(20, int((dbfs + 60) / 3)))  # -60 dBFS .. 0 dBFS
    return "#" * filled + "." * (20 - filled)


def record(session: Session, device=None, part_minutes: float = 60, auto_process: bool = True,
           asr: str | None = None, summarizer: str | None = None,
           stop: threading.Event | None = None, echo=None, on_level=None, on_part=None) -> None:
    """Record until Ctrl+C or until `stop` is set. `echo(msg)` gets log lines and
    `on_level(text)` a status line about once a second (both default to the console).

    If `on_part(final)` is given, it's called whenever a part is saved (final=True after Stop) and
    the caller does the processing, so this returns right after Stop. Otherwise each part is
    processed in a background thread and this waits for the final pass before returning."""
    import sys

    import sounddevice as sd

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if auto_process:
        config.check_ready(asr or session.config["asr"],
                           (summarizer or session.config["summarizer"]) if session.summarize_automatically else None)
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    stop = stop or threading.Event()
    blocks: "queue.Queue[np.ndarray]" = queue.Queue()
    stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", device=device,
                            callback=lambda data, *_: blocks.put(data.copy()))
    part_frames = int(part_minutes * 60 * RATE)
    workers: list[threading.Thread] = []

    if echo is None:
        def echo(msg: str) -> None:
            print(f"\r{' ' * 100}\r{msg}", flush=True)
    if on_level is None:
        def on_level(text: str) -> None:
            print(f"\r{text}   ", end="", flush=True)

    def run_background() -> None:
        try:
            process(session, asr=asr, summarizer=summarizer, echo=echo)
        except Exception as exc:  # keep recording; the next run retries
            echo(f"  processing failed (will retry after the next part): {exc}")

    def open_part():
        name = session.next_part_name(".wav")
        tmp = session.audio_dir / f".{name}"  # hidden until finished, so processing skips it
        w = wave.open(str(tmp), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        return w, tmp, name

    def close_part(w, tmp, name, frames) -> bool:
        w.close()
        if frames < RATE * 5:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(session.audio_dir / name)
        echo(f"Saved {name} ({frames / RATE / 60:.1f} min)")
        return True

    device_name = sd.query_devices(device, "input")["name"]
    echo(f"Recording session '{session.name}' from: {device_name}")
    echo(f"New part every {part_minutes:g} min. Stop to finish.\n")
    w, tmp, name = open_part()
    frames = 0
    started = time.monotonic()
    level_window: list[np.ndarray] = []
    quiet_since = time.monotonic()
    last_draw = 0.0
    stream.start()
    try:
        while not stop.is_set():
            try:
                data = blocks.get(timeout=1)
            except queue.Empty:
                continue
            w.writeframes(data.tobytes())
            frames += len(data)
            level_window.append(data)
            now = time.monotonic()
            if now - last_draw >= 1.0:
                samples = np.concatenate(level_window).astype(np.float32)
                level_window.clear()
                rms = math.sqrt(float(np.mean(samples ** 2))) if samples.size else 0.0
                dbfs = 20 * math.log10(max(rms, 1e-9) / 32768)
                if dbfs > -50:
                    quiet_since = now
                warn = "  <- very quiet, check the mic" if now - quiet_since > 30 else ""
                elapsed = int(now - started)
                part_time = int(frames / RATE)

                def hms(s: int) -> str:
                    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"

                on_level(f"● REC {name}  {hms(part_time)} (total {hms(elapsed)})  "
                         f"[{_meter(dbfs)}] {dbfs:5.1f} dBFS{warn}")
                last_draw = now
            if frames >= part_frames:
                if close_part(w, tmp, name, frames) and auto_process:
                    if on_part:
                        on_part(False)
                    else:
                        t = threading.Thread(target=run_background, daemon=True)
                        t.start()
                        workers.append(t)
                w, tmp, name = open_part()
                frames = 0
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        while not blocks.empty():
            data = blocks.get()
            w.writeframes(data.tobytes())
            frames += len(data)
        close_part(w, tmp, name, frames)

    echo("Stopped.")
    if auto_process and on_part:
        on_part(True)
    elif auto_process:
        for t in workers:
            t.join()
        auto = session.summarize_automatically
        echo("Finishing: transcribing the last part" + (" and writing the session recap..." if auto else "..."))
        process(session, asr=asr, summarizer=summarizer, final=auto, echo=echo)
