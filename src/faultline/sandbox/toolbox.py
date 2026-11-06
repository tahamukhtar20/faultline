"""Tool actions, run inside a sandbox container with `python -I` so workspace code cannot shadow it.

Standard library only. Errors go to stderr with exit code 2.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

WORKSPACE = Path("/workspace")
DATABASE = "/data/orders.db"
MAX_ROWS = 50


class ToolError(Exception):
    pass


def resolve(relative: str) -> Path:
    candidate = (WORKSPACE / relative).resolve()
    if not candidate.is_relative_to(WORKSPACE):
        raise ToolError("path escapes the workspace")
    return candidate


def read(argument: str) -> str:
    path_text, _, line_range = argument.partition(":")
    path = resolve(path_text)
    if not path.is_file():
        raise ToolError(f"not a file: {path_text}")
    lines = path.read_text(errors="replace").splitlines()
    if not line_range:
        return "\n".join(f"{number}: {line}" for number, line in enumerate(lines, start=1))
    try:
        start, _, end = line_range.partition("-")
        first, last = int(start), int(end or start)
    except ValueError:
        raise ToolError("use path, path:line or path:first-last") from None
    selected = lines[first - 1 : last]
    return "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start=first))


def listing() -> str:
    files: list[str] = []
    for root, directories, names in os.walk(WORKSPACE):
        directories[:] = sorted(d for d in directories if not d.startswith((".", "__")))
        files.extend(str((Path(root) / name).relative_to(WORKSPACE)) for name in sorted(names))
    return "\n".join(files)


def search(text: str) -> str:
    if not text:
        return listing()
    matches = []
    for relative in listing().splitlines():
        content = (WORKSPACE / relative).read_text(errors="replace")
        for number, line in enumerate(content.splitlines(), start=1):
            if text in line:
                matches.append(f"{relative}:{number}: {line}")
    return "\n".join(matches) if matches else "no matches"


def patch(argument: str) -> str:
    try:
        payload = json.loads(argument)
    except ValueError as error:
        raise ToolError(f"patch argument is not JSON: {error}") from None
    if not isinstance(payload, dict):
        raise ToolError("patch argument must be a JSON object")
    path_text, old, new = payload.get("path"), payload.get("old"), payload.get("new")
    if not (isinstance(path_text, str) and isinstance(old, str) and isinstance(new, str)):
        raise ToolError("patch requires string path, old and new fields")
    path = resolve(path_text)
    if not path.is_file():
        raise ToolError(f"not a file: {path_text}")
    content = path.read_text()
    count = content.count(old) if old else 0
    if count != 1:
        raise ToolError(f"patch anchor must occur exactly once; found {count}")
    path.write_text(content.replace(old, new, 1))
    return f"patched {path_text}"


def query(sql: str) -> str:
    if not Path(DATABASE).exists():
        return "database does not exist yet"
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True, timeout=5)
    connection.execute("PRAGMA query_only = ON")
    try:
        cursor = connection.execute(sql)
    except sqlite3.Error as error:
        raise ToolError(f"query failed: {error}") from None
    rows = cursor.fetchmany(MAX_ROWS)
    header = [column[0] for column in cursor.description or ()]
    return "\n".join(" | ".join(str(value) for value in row) for row in [header, *rows])


COMMANDS = {"read": read, "search": search, "patch": patch, "query": query}


def main() -> int:
    command, argument = sys.argv[1], sys.stdin.read()
    try:
        print(COMMANDS[command](argument))
    except (ToolError, OSError, UnicodeError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
