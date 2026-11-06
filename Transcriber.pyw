"""Double-click to open the Transcriber window without a console."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transcriber.venv import relaunch  # noqa: E402

relaunch([str(Path(__file__).resolve())], gui=True)

from transcriber.gui import main  # noqa: E402

main()
