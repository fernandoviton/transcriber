"""Small tkinter window over the same pipeline as the CLI.

    py -m transcriber gui        (or double-click Transcriber.pyw for no console)
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import audio, config
from .pipeline import process
from .session import AUDIO_EXT, Session
from .transcribe import clock

ASR_LABELS = {
    "whisper": "Local – Whisper, uses glossary (free, private)",
    "local": "Local – Parakeet, no glossary (free, private)",
    "openai-diarize": "OpenAI + speakers (~$0.36/h)",
    "openai": "OpenAI, glossary hints",
}
SUMMARIZER_LABELS = {"local": "Local – Qwen3 30B (free, private, offline)", "copilot": "Copilot CLI (internet)",
                     "openai": "OpenAI (internet)"}
SUMMARIES_LABELS = {
    "later": "Later – transcribe only; I'll click Summarize + recap",
    "auto": "As I go – summarize each part when transcribed",
}


def _open(path: Path) -> None:
    if hasattr(os, "startfile"):
        os.startfile(path)


def _key(labels: dict, label: str) -> str:
    return next(k for k, v in labels.items() if v == label)


class SessionDialog(tk.Toplevel):
    """New session, or (with `session`) change an existing session's settings."""

    def __init__(self, parent: tk.Tk, session: Session | None = None) -> None:
        super().__init__(parent)
        self.editing = session
        self.title(f"Settings – {session.name}" if session else "New session")
        self.transient(parent)
        self.resizable(False, False)
        self.result: Session | None = None
        profiles = sorted(p.name for p in config.PROFILES.iterdir() if p.is_dir())
        cfg = session.config if session else {
            "profile": "meeting" if "meeting" in profiles else (profiles or [""])[0],
            "asr": config.default_asr(), "summarizer": config.default_summarizer(),
            "summaries": config.default_summaries()}

        self.name = tk.StringVar(value=session.name if session else "")
        self.profile = tk.StringVar(value=cfg["profile"])
        self.asr = tk.StringVar(value=ASR_LABELS[cfg["asr"]])
        self.summarizer = tk.StringVar(value=SUMMARIZER_LABELS[cfg["summarizer"]])
        self.summaries = tk.StringVar(value=SUMMARIES_LABELS[cfg["summaries"]])

        form = ttk.Frame(self, padding=14)
        form.pack(fill="both")
        name = ttk.Entry(form, textvariable=self.name, width=50, state="disabled" if session else "normal")
        rows = [
            ("Name", name),
            ("Profile", ttk.Combobox(form, textvariable=self.profile, values=profiles, state="readonly")),
            ("Transcription", ttk.Combobox(form, textvariable=self.asr, values=list(ASR_LABELS.values()),
                                           state="readonly")),
            ("Summaries", ttk.Combobox(form, textvariable=self.summaries,
                                       values=list(SUMMARIES_LABELS.values()), state="readonly")),
            ("Summarizer", ttk.Combobox(form, textvariable=self.summarizer,
                                        values=list(SUMMARIZER_LABELS.values()), state="readonly")),
        ]
        for row, (label, widget) in enumerate(rows):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))
            widget.grid(row=row, column=1, sticky="ew", pady=4)
        if session:
            ttk.Label(form, foreground="#555", text="Changes apply to parts not yet transcribed / summarized.\n"
                      "Use Redo summaries to re-summarize after changing the profile.").grid(
                row=len(rows), column=0, columnspan=2, sticky="w", pady=(6, 0))
        buttons = ttk.Frame(form)
        buttons.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Save" if session else "Create", command=self.save).pack(side="right", padx=6)
        (rows[1] if session else rows[0])[1].focus_set()
        self.bind("<Return>", lambda _: self.save())
        self.bind("<Escape>", lambda _: self.destroy())
        self.grab_set()

    def save(self) -> None:
        settings = {"profile": self.profile.get(), "asr": _key(ASR_LABELS, self.asr.get()),
                    "summarizer": _key(SUMMARIZER_LABELS, self.summarizer.get()),
                    "summaries": _key(SUMMARIES_LABELS, self.summaries.get())}
        try:
            if self.editing:
                self.editing.update(**settings)
                self.result = self.editing
            else:
                name = self.name.get().strip()
                if not name:
                    messagebox.showwarning("New session", "Give the session a name.", parent=self)
                    return
                self.result = Session.create(name, **settings)
        except SystemExit as exc:
            messagebox.showerror("Can't save session", str(exc), parent=self)
            return
        self.destroy()


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Transcriber")
        root.geometry("1100x720")
        root.minsize(850, 520)
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.session: Session | None = None
        self.sessions: list[tuple[Path, Session | None, str]] = []
        # Processing runs one job at a time on its own thread, independent of recording,
        # so you can start the next recording while the previous one is still being transcribed.
        self.jobs: list[dict] = []  # pending, in order
        self.jobs_cv = threading.Condition()
        self.current_job: dict | None = None
        threading.Thread(target=self.job_worker, daemon=True).start()
        self.recording: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.durations: dict[tuple, float | None] = {}
        self.preview_mtime = 0.0

        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")

        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        # ---- left: sessions
        left = ttk.Frame(paned, padding=(0, 0, 6, 0))
        paned.add(left, weight=1)
        ttk.Label(left, text="Sessions", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.session_list = tk.Listbox(left, activestyle="none", exportselection=False, font=("Segoe UI", 10))
        self.session_list.pack(fill="both", expand=True, pady=6)
        self.session_list.bind("<<ListboxSelect>>", lambda _: self.on_select())
        row = ttk.Frame(left)
        row.pack(fill="x")
        self.new_button = ttk.Button(row, text="New session…", command=self.new_session)
        self.new_button.pack(side="left")
        ttk.Button(row, text="Refresh", command=self.load_sessions).pack(side="left", padx=6)

        # ---- right: selected session
        right = ttk.Frame(paned, padding=(6, 0, 0, 0))
        paned.add(right, weight=4)
        self.title = ttk.Label(right, text="Select or create a session", font=("Segoe UI", 13, "bold"))
        self.title.pack(anchor="w")
        self.info = ttk.Label(right, text="", foreground="#555")
        self.info.pack(anchor="w", pady=(0, 6))

        self.parts = ttk.Treeview(right, columns=("length", "transcribed", "summarized"), height=5)
        self.parts.heading("#0", text="Audio part")
        self.parts.heading("length", text="Length")
        self.parts.heading("transcribed", text="Transcribed")
        self.parts.heading("summarized", text="Summarized")
        self.parts.column("#0", width=380)
        for column in ("length", "transcribed", "summarized"):
            self.parts.column(column, width=90, anchor="center")
        self.parts.pack(fill="x")
        self.totals = ttk.Label(right, text="", foreground="#555")
        self.totals.pack(anchor="w", pady=(2, 6))

        actions = ttk.Frame(right)
        actions.pack(fill="x")
        self.session_buttons = [
            ttk.Button(actions, text="Open audio folder", command=lambda: self.open_sub("audio")),
            ttk.Button(actions, text="Add files…", command=self.add_files),
            ttk.Button(actions, text="Transcribe", command=lambda: self.run_process(final=False)),
            ttk.Button(actions, text="Summarize + recap", command=lambda: self.run_process(final=True)),
            ttk.Button(actions, text="Redo summaries", command=lambda: self.run_process(final=True, redo=True)),
            ttk.Button(actions, text="Redo selected", command=self.redo_selected),
            ttk.Button(actions, text="Open session.md", command=lambda: self.open_sub("session.md")),
            ttk.Button(actions, text="Open folder", command=lambda: self.open_sub("")),
            ttk.Button(actions, text="Settings…", command=self.edit_settings),
        ]
        for button in self.session_buttons:
            button.pack(side="left", padx=(0, 6))

        rec = ttk.LabelFrame(right, text="Record live", padding=8)
        rec.pack(fill="x", pady=8)
        ttk.Label(rec, text="Mic").pack(side="left")
        self.devices: list[tuple[int, str]] = []
        self.device = tk.StringVar()
        self.device_box = ttk.Combobox(rec, textvariable=self.device, state="readonly", width=42)
        self.device_box.pack(side="left", padx=6)
        ttk.Label(rec, text="New part every").pack(side="left", padx=(8, 0))
        self.part_minutes = tk.StringVar(value="60")
        ttk.Spinbox(rec, from_=5, to=180, increment=5, textvariable=self.part_minutes, width=5).pack(side="left", padx=4)
        ttk.Label(rec, text="min").pack(side="left")
        self.record_button = ttk.Button(rec, text="● Record", command=self.toggle_record)
        self.record_button.pack(side="left", padx=10)
        self.level = ttk.Label(right, text="", font=("Consolas", 10))
        self.level.pack(anchor="w")

        status = ttk.Frame(right)
        status.pack(fill="x", pady=(4, 0))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=140)
        self.progress.pack(side="left")
        self.status = ttk.Label(status, text="Idle", foreground="#555")
        self.status.pack(side="left", padx=8)

        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True, pady=(4, 0))
        self.preview = self._text_tab(tabs, "Summary (session.md)", ("Segoe UI", 10))
        self.log = self._text_tab(tabs, "Log", ("Consolas", 9))
        self.tabs = tabs

        self.load_devices()
        self.load_sessions()
        self.set_buttons()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(200, self.pump)
        root.after(1500, self.poll)

    # ---- helpers -------------------------------------------------------------

    def _text_tab(self, tabs: ttk.Notebook, title: str, font) -> tk.Text:
        frame = ttk.Frame(tabs)
        text = tk.Text(frame, wrap="word", font=font, padx=10, pady=8, relief="flat", state="disabled")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        tabs.add(frame, text=title)
        return text

    def write_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def echo(self, message: str) -> None:
        """Thread-safe log line (called from worker threads)."""
        self.events.put(("log", message))

    def set_buttons(self) -> None:
        has = self.session is not None
        for button in self.session_buttons:
            button.state(["!disabled"] if has else ["disabled"])
        if self.recording:
            self.record_button.configure(text="■ Stop" if self.stop_event and not self.stop_event.is_set()
                                         else "Finishing…")
            self.record_button.state(["disabled"] if self.stop_event.is_set() else ["!disabled"])
            self.session_list.configure(state="disabled")
            self.new_button.state(["disabled"])
        else:
            self.record_button.configure(text="● Record")
            self.record_button.state(["!disabled"] if has and self.devices else ["disabled"])
            self.session_list.configure(state="normal")
            self.new_button.state(["!disabled"])

    def duration(self, path: Path) -> float | None:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime)
        if key not in self.durations:
            try:
                self.durations[key] = audio.duration(path)
            except Exception:
                self.durations[key] = None
        return self.durations[key]

    # ---- sessions --------------------------------------------------------------

    def load_sessions(self, select: Path | None = None) -> None:
        current = select or (self.session.path if self.session else None)
        config.SESSIONS.mkdir(exist_ok=True)
        self.sessions = []
        for path in sorted((p for p in config.SESSIONS.iterdir() if p.is_dir()), reverse=True):
            try:
                self.sessions.append((path, Session(path), ""))
            except SystemExit as exc:
                self.sessions.append((path, None, str(exc)))
        self.session_list.delete(0, "end")
        for index, (path, session, error) in enumerate(self.sessions):
            self.session_list.insert("end", f"{'⚠ ' if error else ''}{path.name}")
            if error:
                self.session_list.itemconfigure(index, foreground="#b00020")
            if path == current:
                self.session_list.selection_set(index)
        if current and not self.session_list.curselection() and self.sessions:
            self.session = None
        self.on_select()

    def on_select(self) -> None:
        selection = self.session_list.curselection()
        if not selection:
            self.show(None)
            return
        path, session, error = self.sessions[selection[0]]
        if error:
            self.show(None, error_title=path.name, error=error)
        else:
            self.show(Session(path))  # re-read in case session.json changed

    def show(self, session: Session | None, error_title: str = "", error: str = "") -> None:
        self.session = session
        self.preview_mtime = 0.0
        if session is None:
            self.title.configure(text=error_title or "Select or create a session")
            self.info.configure(text=error or "", foreground="#b00020" if error else "#555")
            self.parts.delete(*self.parts.get_children())
            self.totals.configure(text="")
            self.set_preview("")
        else:
            cfg = session.config
            self.title.configure(text=session.name)
            self.info.configure(foreground="#555", text=(
                f"Profile: {cfg['profile']}   ·   Transcription: {ASR_LABELS.get(cfg['asr'], cfg['asr'])}   ·   "
                f"Summaries: {'as I go' if cfg['summaries'] == 'auto' else 'later'} "
                f"({SUMMARIZER_LABELS.get(cfg['summarizer'], cfg['summarizer'])})   ·   {session.path.name}"))
            self.refresh_session()
        self.set_buttons()

    def refresh_session(self) -> None:
        session = self.session
        if session is None or not session.path.exists():
            return
        rows, total = [], 0.0
        for part in session.parts():
            length = self.duration(part)
            total += length or 0
            rows.append((part.name, clock(length) if length is not None else "unreadable",
                         "✓" if session.transcript_path(part).exists() else "",
                         "✓" if session.summary_path(part).exists() else ""))
        existing = [(self.parts.item(i, "text"), *self.parts.item(i, "values")) for i in self.parts.get_children()]
        if existing != [tuple(r) for r in rows]:
            self.parts.delete(*self.parts.get_children())
            for name, *values in rows:
                self.parts.insert("", "end", text=name, values=values)
        events = session.events()
        cost = sum(e.get("est_cost_usd", 0) for e in events)
        recap = "recap written" if (session.path / "recap.md").exists() else "no recap yet"
        hint = "   ·   Drop recordings into the audio folder, then Transcribe." if not rows else ""
        self.totals.configure(text=f"{len(rows)} part(s)   ·   {clock(total)} of audio   ·   "
                                   f"est. transcription cost ${cost:.2f}   ·   {recap}{hint}")
        md = session.path / "session.md"
        mtime = md.stat().st_mtime if md.exists() else -1.0
        if mtime != self.preview_mtime:
            self.preview_mtime = mtime
            self.set_preview(md.read_text(encoding="utf-8") if md.exists()
                             else "Nothing transcribed yet.\n\nAdd audio (Open audio folder / Add files…) "
                                  "and click Transcribe, or record live below. Click Summarize + recap "
                                  "when you want the summaries.")

    def set_preview(self, text: str) -> None:
        position = self.preview.yview()[0]
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")
        self.preview.yview_moveto(position)

    def new_session(self) -> None:
        dialog = SessionDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.result:
            self.load_sessions(select=dialog.result.path)
            self.write_log(f"Created {dialog.result.path}")
            self.open_sub("audio")

    def edit_settings(self) -> None:
        if not self.session:
            return
        dialog = SessionDialog(self.root, self.session)
        self.root.wait_window(dialog)
        if dialog.result:
            cfg = dialog.result.config
            self.write_log(f"Settings saved for {dialog.result.name}: asr={cfg['asr']}  "
                           f"summaries={cfg['summaries']}  summarizer={cfg['summarizer']}  profile={cfg['profile']}")
            self.show(dialog.result)

    def open_sub(self, sub: str) -> None:
        if not self.session:
            return
        target = self.session.path / sub if sub else self.session.path
        if not target.exists():
            messagebox.showinfo("Not there yet", f"{target.name} doesn't exist yet. Process the session first.")
            return
        _open(target)

    def add_files(self) -> None:
        if not self.session:
            return
        pattern = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXT))
        files = filedialog.askopenfilenames(title="Add recordings", filetypes=[("Audio", pattern), ("All files", "*.*")])
        if not files:
            return
        session = self.session

        def work(echo) -> None:
            for file in files:
                src = Path(file)
                if src.suffix.lower() not in AUDIO_EXT:
                    echo(f"Skipped {src.name}: not an audio file I know")
                    continue
                dest = session.audio_dir / session.next_part_name(src.suffix, src.name)
                echo(f"Copying {src.name} -> audio/{dest.name}")
                shutil.copy2(src, dest)
            echo("Added. Click Transcribe when ready.")

        self.enqueue({"label": f"Adding {len(files)} file(s) to {session.name}", "work": work})

    # ---- background processing queue ----------------------------------------------

    def enqueue(self, job: dict) -> None:
        """Queue work for the processing thread. Main thread only.
        A process job is dropped if an equal-or-bigger one for the same session is already waiting."""
        with self.jobs_cv:
            key = job.get("key")
            if key and any(j.get("key") and j["key"][0] == key[0] and j["key"][1] >= key[1]
                           and j["key"][2] >= key[2] for j in self.jobs):
                self.write_log(f"(already queued: {job['label']})")
                return
            self.jobs.append(job)
            self.jobs_cv.notify()
        self.write_log(f"Queued: {job['label']}")
        self.update_status()

    def enqueue_process(self, session: Session, final: bool, redo: bool = False,
                        parts: tuple[str, ...] = ()) -> None:
        if parts:
            what = f"redo summary of {', '.join(parts)}"
        else:
            what = "redo summaries + recap" if redo else ("summarize + recap" if final else "transcribe")
        self.enqueue({
            "label": f"{session.name}: {what}",
            "key": None if parts else (session.path, final, redo),
            "work": lambda echo: process(session, final=final, redo_summaries=redo,
                                         redo_parts=set(parts), echo=echo),
        })

    def redo_selected(self) -> None:
        if not self.session:
            return
        parts = tuple(self.parts.item(i, "text") for i in self.parts.selection())
        if not parts:
            messagebox.showinfo("Redo selected", "Select one or more parts in the list first "
                                "(Ctrl+click or Shift+click to pick several).")
            return
        self.enqueue_process(self.session, final=False, parts=parts)

    def job_worker(self) -> None:
        while True:
            with self.jobs_cv:
                while not self.jobs:
                    self.jobs_cv.wait()
                job = self.jobs.pop(0)
            self.events.put(("job_start", job))

            def echo(message: str) -> None:
                self.events.put(("progress", message))

            try:
                job["work"](echo)
            except SystemExit as exc:  # validation errors from the pipeline
                self.events.put(("error", f"{job['label']}: {exc}"))
            except Exception as exc:
                self.events.put(("error", f"{job['label']}: {type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("job_done", job))

    def update_status(self, detail: str = "") -> None:
        waiting = len(self.jobs)
        more = f"   ({waiting} more queued)" if waiting else ""
        if self.current_job:
            text = f"Working on {self.current_job['label']}"
            if detail:
                text += f" — {detail.strip()}"
            self.status.configure(text=text + more, foreground="#000")
        else:
            self.status.configure(text="Idle" + more, foreground="#555")

    def run_process(self, final: bool, redo: bool = False) -> None:
        if not self.session:
            return
        if not self.session.parts():
            messagebox.showinfo("No audio", "Add recordings to the audio folder first.")
            return
        self.tabs.select(1)
        self.enqueue_process(self.session, final, redo)

    # ---- recording -------------------------------------------------------------------

    def load_devices(self) -> None:
        try:
            from .record import input_devices

            self.devices = input_devices()
        except Exception as exc:
            self.devices = []
            self.write_log(f"No microphones: {exc}")
        self.device_box.configure(values=[f"{name}" for _, name in self.devices])
        if self.devices:
            self.device.set(self.devices[0][1])

    def toggle_record(self) -> None:
        if self.recording:
            if self.stop_event:
                self.stop_event.set()
                self.set_buttons()
            return
        if not self.session or not self.devices:
            return
        try:
            minutes = float(self.part_minutes.get())
            assert minutes >= 1
        except (ValueError, AssertionError):
            messagebox.showwarning("Record", "Minutes per part must be a number, 1 or more.")
            return
        device = next(i for i, name in self.devices if name == self.device.get())
        session = self.session
        self.stop_event = threading.Event()
        stop = self.stop_event

        def on_part(final: bool) -> None:  # called on the recording thread
            self.events.put(("part_saved", session, final))

        def work() -> None:
            from .record import record

            try:
                record(session, device=device, part_minutes=minutes, stop=stop, echo=self.echo,
                       on_level=lambda text: self.events.put(("level", text)), on_part=on_part)
            except SystemExit as exc:
                self.events.put(("error", str(exc)))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("recording_done",))

        self.tabs.select(1)
        self.write_log(f"— Recording ({minutes:g}-min parts) —")
        self.recording = threading.Thread(target=work, daemon=True)
        self.recording.start()
        self.set_buttons()

    # ---- event loop ----------------------------------------------------------------------

    def pump(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self.write_log(event[1])
                elif kind == "level":
                    self.level.configure(text=event[1])
                elif kind == "error":
                    self.write_log(f"Error: {event[1]}")
                    messagebox.showerror("Transcriber", event[1])
                elif kind == "progress":
                    self.write_log(event[1])
                    self.update_status(event[1])
                elif kind == "job_start":
                    self.current_job = event[1]
                    self.progress.start(12)
                    self.write_log(f"— {event[1]['label']} —")
                    self.update_status()
                elif kind == "job_done":
                    self.current_job = None
                    self.progress.stop()
                    self.refresh_session()
                    self.update_status()
                elif kind == "part_saved":
                    _, session, final = event
                    self.enqueue_process(session, final=final and session.summarize_automatically)
                elif kind == "recording_done":
                    self.recording = None
                    self.stop_event = None
                    self.level.configure(text="")
                    self.refresh_session()
                    self.set_buttons()
        except queue.Empty:
            pass
        self.root.after(200, self.pump)

    def poll(self) -> None:
        """Pick up files dropped into the audio folder and new summaries as they're written."""
        try:
            self.refresh_session()
        except Exception as exc:
            self.write_log(f"Refresh failed: {exc}")
        self.root.after(2000, self.poll)

    def on_close(self) -> None:
        if self.recording or self.current_job or self.jobs:
            if not messagebox.askyesno("Transcriber", "Work is still running (recording or processing). "
                                                      "Quit anyway? A recording in progress would be lost; "
                                                      "unfinished processing picks up next time you click Process."):
                return
        self.root.destroy()


def main() -> int:
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens; must precede Tk()
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0
