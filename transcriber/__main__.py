import sys

from .venv import relaunch

relaunch(["-m", "transcriber", *sys.argv[1:]])

from .cli import main  # noqa: E402

sys.exit(main())
