# Transcriber

Record or import meeting audio, transcribe it, and get a short Markdown summary:
one per part (each recorded hour or each file you add), plus a recap for the whole meeting.
Summaries capture discussion, decisions, action items, and open questions.

## Setup

- ffmpeg on PATH (`winget install Gyan.FFmpeg`).
- Windows ARM64 and a **native ARM64** Python with tkinter (python.org's ARM64 installer).
  The bundled whisper.cpp and llama.cpp downloads target Windows ARM64; other platforms
  need compatible binaries. Avoid an emulated x64 Python for local transcription.
- Create the project's `.venv` once (uv, or plain `-m venv` + pip):

  ```powershell
  uv venv --python "$env:LOCALAPPDATA\Programs\Python\Python311-arm64\python.exe" .venv
  uv pip install --python .venv\Scripts\python.exe -r requirements.txt
  ```

  After that, `py -m transcriber ...` and `Transcriber.pyw` switch to `.venv` automatically.
- Transcription runs **locally** with OpenAI's Whisper model (large-v3-turbo, via whisper.cpp).
  Names and project terms from the profile's glossary are given to it as spelling hints. The first run
  downloads it into `tools\whisper` (~850 MB); after that it works offline.
- Summaries default to a **local** Qwen3 model through llama.cpp. Its first run downloads
  a large model and requires about 17 GB of memory. Set `TRANSCRIBER_SUMMARIZER=copilot`
  to use an installed GitHub Copilot CLI instead, or `openai` with an API key.
  Cloud summarizers need internet and receive transcript text.
- By default new sessions only transcribe; run summaries separately when ready
  (see "Summaries: now or later" below).
- Optional: copy `.env.example` to `.env` and add `OPENAI_API_KEY` to use cloud transcription instead.

## Use

### Easiest: the window

Double-click **`Transcriber.pyw`** (or run `py -m transcriber gui`).

1. **New session…**: pick a name and profile. The audio folder opens.
2. Either drag your recordings into that folder (they show up in the list within a few seconds), or pick a mic under **Record live** and press **Record**. Press **Stop** when you're done.
3. Press **Transcribe** (after recording, this happens automatically). Works offline.
4. Press **Summarize + recap** (internet is needed for a cloud summarizer).
5. Read the result in the **Summary** tab, or open `session.md`.

**Settings…** changes a session's transcription, summaries mode, summarizer or profile, e.g. to
switch an older OpenAI session to local so it works offline.

Processing runs in the background, one job at a time. The status line under **Record live** shows
what it's doing and how many jobs are waiting. You can keep recording (or start the next
recording) while earlier parts are still being transcribed.

**Mic tips for a table of people:** in Windows Sound settings set the mic's input volume high and
turn **Voice focus** (Windows Studio Effects) off, since it filters out anyone not right in front
of the laptop. Put the mic in the middle of the table; if it has a pickup-pattern switch, use omni.
Volume is leveled before transcription, so quiet speakers get boosted (see `TRANSCRIBER_NORMALIZE`).

The window runs the same steps as the commands below.

### Command line

Every session starts with `new`. Then either add recordings you already have, or record live.

#### A. You already have recordings (e.g. iPhone voice memos)

```powershell
py -m transcriber new                 # 1. answer the questions; the session's audio\ folder opens
                                      # 2. drag your recordings into that audio\ folder
py -m transcriber process             # 3. transcribe everything (offline is fine)
py -m transcriber summarize           # 4. summarize each part; cloud backends also write a recap
```

#### B. Record live on this laptop

```powershell
py -m transcriber new                 # 1. answer the questions
py -m transcriber record              # 2. records; each hour is transcribed as soon as it ends
                                      # 3. Ctrl+C when done: finishes transcribing the last part
py -m transcriber summarize           # 4. summarize each part; cloud backends also write a recap
```

Use `py -m transcriber devices` and `record --device <number>` to pick a microphone (e.g. a USB
speakerphone). `record --part-minutes 10` makes 10-minute parts instead of 1-hour ones.

### Summaries: now or later

Each session has a **summaries** setting:

- `later` (default): `process`, `record` and the **Transcribe** button only transcribe. Summaries
  and the recap are written when you run `summarize` / press **Summarize + recap**.
- `auto`: each part is summarized as soon as it's transcribed. Cloud backends also write a
  recap when recording stops and need internet while summarizing.

Pick it when creating the session (`new --summaries auto`), or change it later with
**Settings…** or `py -m transcriber set <session> --summaries auto`.

### How your audio is split

- **One file = one part = one summary section.** A long file is not split into hourly
  summaries: a single 3-hour recording gets one summary for all 3 hours (plus the recap).
  For hourly summaries, record in 1-hour files (live `record` already does this).
- Behind the scenes, every file is transcribed in 10-minute pieces, so any length works.
  You don't need to split anything yourself.
- Files are processed in name order, with numbers compared as numbers ("Recording 9" before
  "Recording 10"). `.m4a`, `.wav`, `.mp3` and most other audio formats work.

### Re-running is safe

- `process` and `summarize` only do what's missing. Add another file later and run them again:
  only the new file is transcribed and summarized.
- `summarize` rewrites the recap each time, so it always covers every part.
  The local summarizer skips the recap: it's good at single parts but not at whole sessions,
  so write that yourself or use a cloud summarizer (`copilot`/`openai`) for it.
- After editing a profile, `summarize --redo` redoes the summaries without transcribing again.
  To try a change on one part first, use `summarize --part part-03` (repeatable, skips the recap),
  or select parts in the app and click **Redo selected**.

### Results

Open `sessions\<date>-<name>\session.md`: the recap first (if there is one), then a summary for each part.
Full transcripts are in `transcript\`.

### Other commands and options

```powershell
py -m transcriber status                              # parts, progress, estimated cost
py -m transcriber import C:\path\memo1.m4a memo2.m4a  # copy files in and transcribe (instead of dragging)
py -m transcriber set test-mic --asr whisper          # change a session's settings
```

- Commands use the latest session unless you name one (any unique part of its folder name),
  e.g. `py -m transcriber summarize weekly-planning`.
- `new "weekly planning" --profile meeting --asr whisper --summaries later --no-open` skips the questions.
- `process --final` is the same as `summarize`.

## Customize: profiles

A profile is a folder in `profiles/` with three plain-English files:

| File | Used for |
|---|---|
| `summarize.md` | Instructions (not output) for summarizing each part: what to keep, what to skip, the format. |
| `recap.md` | Instructions (not output) for the whole-session recap written at the end. |
| `glossary.md` | Names and terms. The whole file goes to the summarizer, which fixes misheard names. With `--asr whisper` (the default) or `openai`, the names (bullets like `- Project Atlas: migration project`) are also given to the transcriber as spelling hints. Put the most-said names first: Whisper only reads the first ~700 characters of names. |

The built-in `meeting` profile is the default. An optional `rpg` profile summarizes tabletop
RPG sessions, with a glossary of fictional characters and setting terms. Select it with
`new "example session" --profile rpg`, or choose it when creating a session in the window.

Keep personal names, client details, and
confidential terminology in a separate local profile, for example `profiles/private-team/`.
Custom profile folders other than the built-in `meeting` and `rpg` profiles are ignored by
Git by default. Copy a built-in profile to the new
folder and customize its glossary.

## Settings (.env or environment)

| Variable | Default | Meaning |
|---|---|---|
| `TRANSCRIBER_ASR` | `whisper` | `whisper` (on this laptop, uses glossary hints), `local` (Parakeet on this laptop, no hints), `openai-diarize` (cloud, timestamps + speakers), `openai` (cloud gpt-transcribe, uses glossary keywords). None of them label speakers except `openai-diarize`. |
| `TRANSCRIBER_SUMMARIZER` | `local` | `local` (llama.cpp), `copilot`, or `openai` |
| `TRANSCRIBER_SUMMARIES` | `later` | Default summaries mode for new sessions: `later` or `auto` |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | | OpenAI (or compatible) endpoint |
| `TRANSCRIBER_LANGUAGE` | `en` | Spoken language; empty = auto-detect |
| `TRANSCRIBER_WHISPER_MODEL` | `ggml-large-v3-turbo-q8_0.bin` | Whisper model file from [whisper.cpp's list](https://huggingface.co/ggerganov/whisper.cpp). |
| `TRANSCRIBER_WHISPER_THREADS` | `8` | CPU threads for Whisper |
| `TRANSCRIBER_LOCAL_MODEL` | `nemo-parakeet-tdt-0.6b-v3` | Parakeet model for `--asr local` (any [onnx-asr](https://github.com/istupakov/onnx-asr) name) |
| `TRANSCRIBER_LOCAL_THREADS` | `6` | CPU threads for Parakeet |
| `TRANSCRIBER_LLM_MODEL` | `Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf` | GGUF model for the local summarizer |
| `TRANSCRIBER_NORMALIZE` | `1` | Even out volume before transcription (boosts quiet, far-away voices). Your original audio is untouched. `0` = off |
| `TRANSCRIBER_COPILOT_MODEL`, `TRANSCRIBER_COPILOT_AGENT` | | Model / agent passed to `copilot` |
| `TRANSCRIBER_SUMMARY_MODEL` | `gpt-5-mini` | Model for `--summarizer openai` |

A session remembers the backends it was created with; `--asr` / `--summarizer` override them per run.

## Privacy

With the default `whisper` transcription and `local` summarizer, audio and transcripts stay
on the local machine after model downloads. With cloud transcription, 10-minute audio chunks
go to OpenAI. With a cloud summarizer, transcript text goes to Copilot or OpenAI.
Choose backends appropriate for the meeting's confidentiality requirements.

Recordings, session output, exported transcripts, `.env` files, and downloaded tools are
excluded from Git. The repository includes only application code, generic instructions, and
fictional examples. Do not commit real meeting content or private profile details.
