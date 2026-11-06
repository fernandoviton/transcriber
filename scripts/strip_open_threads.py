"""Remove "**Open threads:**" blocks from Markdown files.

A block is the "**Open threads:**" line plus the list right under it. The list ends at the
first empty line or the first line that isn't a list item (or an indented continuation).
Matching ignores case and the colon's position, so "**Open Threads:**" and "**Open threads**:"
both match.

Usage:
    py scripts\\strip_open_threads.py FILE_OR_FOLDER [...] [--dry-run]

Folders are searched for *.md files (including subfolders). Files are edited in place.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADING = re.compile(r"^\s*\*\*\s*open threads\s*:?\s*\*\*\s*:?\s*$", re.IGNORECASE)
LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def strip(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    out: list[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        if not HEADING.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        removed += 1
        i += 1
        while i < len(lines) and lines[i].strip() and (LIST_ITEM.match(lines[i]) or lines[i][:1] in " \t"):
            i += 1
        # Avoid leaving a double blank line where the block was.
        while out and not out[-1].strip() and i < len(lines) and not lines[i].strip():
            i += 1
    result = "\n".join(out).rstrip() + "\n" if out else ""
    return result, removed


def md_files(paths: list[str]):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(p.rglob("*.md"))
        elif p.is_file():
            yield p
        else:
            print(f"not found: {p}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove **Open threads:** blocks from Markdown files.")
    parser.add_argument("paths", nargs="+", help="Markdown files or folders")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without editing")
    args = parser.parse_args()
    total = 0
    for path in md_files(args.paths):
        text = path.read_text(encoding="utf-8")
        new, removed = strip(text)
        if removed:
            total += removed
            print(f"{path}: {removed} block(s){' (dry run)' if args.dry_run else ''}")
            if not args.dry_run:
                path.write_text(new, encoding="utf-8")
    print(f"Done: {total} block(s) {'found' if args.dry_run else 'removed'}.")


if __name__ == "__main__":
    main()
