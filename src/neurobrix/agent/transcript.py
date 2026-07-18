"""Incremental transcript recording — THE R29 artifact of agent runs.

Two synchronized files under one directory:
  transcript.jsonl — machine-readable event stream
  transcript.md    — human-readable session log (what the maintainer reads)

Events are flushed as they happen: a crashed or interrupted session still
leaves a faithful record.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class Transcript:
    def __init__(
        self,
        directory: str,
        meta: Optional[Dict[str, Any]] = None,
        echo: bool = False,
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.dir / "transcript.jsonl").open("a")
        self._md = (self.dir / "transcript.md").open("a")
        self._echo = echo
        self._t0 = time.monotonic()
        if meta:
            self.event("meta", **meta)
            self._md_write("# Agent session\n")
            for key, value in meta.items():
                self._md_write(f"- **{key}**: {value}")
            self._md_write("")

    def event(self, kind: str, **data: Any) -> None:
        record = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **data}
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    # ── typed events (each also renders the markdown view) ───────────

    def turn(self, index: int) -> None:
        self.event("turn", index=index)
        self._md_write(f"\n## Turn {index}\n")

    def model_text(self, text: str) -> None:
        self.event("model_text", text=text)
        self._md_write("**Assistant:**\n\n```\n" + text + "\n```\n")

    def tool_call(self, name: str, arguments: Dict[str, Any]) -> None:
        self.event("tool_call", name=name, arguments=arguments)
        args = json.dumps(arguments, ensure_ascii=False)
        if len(args) > 400:
            args = args[:400] + "…"
        self._md_write(f"→ **{name}** {args}")

    def tool_result(self, name: str, output: str, elapsed_s: float) -> None:
        self.event("tool_result", name=name, output=output, elapsed_s=round(elapsed_s, 3))
        shown = output if len(output) <= 2000 else output[:2000] + "\n[… truncated in md; full output in jsonl]"
        self._md_write(f"\n```\n{shown}\n```\n_({elapsed_s:.2f}s)_\n")

    def note(self, text: str) -> None:
        self.event("note", text=text)
        self._md_write(f"_{text}_")

    def finish(self, stop_reason: str, final_answer: str, turns: int) -> None:
        self.event("finish", stop_reason=stop_reason, final_answer=final_answer, turns=turns)
        self._md_write(f"\n## Finished — {stop_reason} after {turns} turn(s)\n")
        self._md_write("**Final answer:**\n\n" + (final_answer or "_(none)_") + "\n")
        self._jsonl.close()
        self._md.close()

    def _md_write(self, line: str) -> None:
        self._md.write(line + "\n")
        self._md.flush()
        if self._echo:
            print(line)
