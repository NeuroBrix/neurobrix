"""File tools: read_file, write_file, edit_file, list_dir, grep.

All paths go through the sandbox jail. grep is pure-Python `re` over
jailed files — deterministic, no external binary. Handlers return the
exact text the model sees; errors are returned as text too (the model
recovers), except jail violations which raise and stop the turn.
"""

import re
from typing import List

from neurobrix.agent.sandbox import Sandbox, SandboxViolation
from neurobrix.agent.tools import ToolRegistry, ToolSpec

_READ_LINE_LIMIT = 2000
_GREP_MATCH_BOUND = 200
_GREP_FILE_BOUND = 2000  # files scanned before bailing out


def register_fs_tools(registry: ToolRegistry, sandbox: Sandbox) -> None:
    def read_file(path: str, offset: int = 0, limit: int = _READ_LINE_LIMIT) -> str:
        target = sandbox.resolve(path)
        if not target.is_file():
            return f"ERROR: not a file: {path}"
        lines = target.read_text(errors="replace").splitlines()
        offset, limit = max(int(offset), 0), max(int(limit), 1)
        window = lines[offset : offset + limit]
        if not window:
            return f"[empty range: file has {len(lines)} lines]"
        numbered = "\n".join(
            f"{i + offset + 1:6d}\t{line}" for i, line in enumerate(window)
        )
        suffix = (
            f"\n[... {len(lines) - offset - len(window)} more lines]"
            if offset + len(window) < len(lines)
            else ""
        )
        return numbered + suffix

    def write_file(path: str, content: str) -> str:
        target = sandbox.resolve(path)
        sandbox.approve("write_file", str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content.encode())} bytes to {target.relative_to(sandbox.workdir)}"

    def edit_file(path: str, old_string: str, new_string: str) -> str:
        target = sandbox.resolve(path)
        sandbox.approve("edit_file", str(target))
        if not target.is_file():
            return f"ERROR: not a file: {path}"
        text = target.read_text(errors="replace")
        count = text.count(old_string)
        if count == 0:
            return "ERROR: old_string not found — read the file and match exactly"
        if count > 1:
            return f"ERROR: old_string matches {count} times — provide a unique match"
        target.write_text(text.replace(old_string, new_string, 1))
        return f"edited {target.relative_to(sandbox.workdir)}"

    def list_dir(pattern: str = "**/*") -> str:
        entries: List[str] = []
        for match in sorted(sandbox.workdir.glob(pattern)):
            try:
                resolved = sandbox.resolve(str(match))
            except SandboxViolation:
                continue  # symlink pointing out of the jail — invisible
            rel = str(resolved.relative_to(sandbox.workdir))
            entries.append(rel + "/" if resolved.is_dir() else rel)
            if len(entries) >= _GREP_FILE_BOUND:
                entries.append(f"[... truncated at {_GREP_FILE_BOUND} entries]")
                break
        return "\n".join(entries) if entries else "[no matches]"

    def grep(pattern: str, glob: str = "**/*") -> str:
        try:
            needle = re.compile(pattern)
        except re.error as exc:
            return f"ERROR: bad regex: {exc}"
        hits: List[str] = []
        scanned = 0
        for path in sorted(sandbox.workdir.glob(glob)):
            try:
                resolved = sandbox.resolve(str(path))
            except SandboxViolation:
                continue
            if not resolved.is_file():
                continue
            scanned += 1
            if scanned > _GREP_FILE_BOUND:
                hits.append(f"[... stopped after scanning {_GREP_FILE_BOUND} files]")
                break
            rel = resolved.relative_to(sandbox.workdir)
            try:
                for lineno, line in enumerate(
                    resolved.read_text(errors="replace").splitlines(), 1
                ):
                    if needle.search(line):
                        hits.append(f"{rel}:{lineno}: {line.strip()}")
                        if len(hits) >= _GREP_MATCH_BOUND:
                            hits.append(f"[... truncated at {_GREP_MATCH_BOUND} matches]")
                            return "\n".join(hits)
            except OSError:
                continue
        return "\n".join(hits) if hits else "[no matches]"

    _path_prop = {"type": "string", "description": "Path inside the workdir"}
    registry.register(ToolSpec(
        "read_file",
        "Read a file (line-numbered). Use offset/limit for large files.",
        {
            "type": "object",
            "properties": {
                "path": _path_prop,
                "offset": {"type": "integer", "description": "First line index (0-based)"},
                "limit": {"type": "integer", "description": "Max lines to return"},
            },
            "required": ["path"],
        },
        read_file,
    ))
    registry.register(ToolSpec(
        "write_file",
        "Create or overwrite a file with the given content.",
        {
            "type": "object",
            "properties": {"path": _path_prop, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        write_file,
    ))
    registry.register(ToolSpec(
        "edit_file",
        "Replace one exact occurrence of old_string with new_string in a file.",
        {
            "type": "object",
            "properties": {
                "path": _path_prop,
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        edit_file,
    ))
    registry.register(ToolSpec(
        "list_dir",
        "List files and directories matching a glob pattern (default: everything).",
        {
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "Glob, e.g. src/**/*.py"}},
            "required": [],
        },
        list_dir,
    ))
    registry.register(ToolSpec(
        "grep",
        "Search file contents with a regex; returns file:line: match lines.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex"},
                "glob": {"type": "string", "description": "File glob to scan (default **/*)"},
            },
            "required": ["pattern"],
        },
        grep,
    ))
