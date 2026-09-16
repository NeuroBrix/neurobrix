#!/usr/bin/env python
"""The instrument for a code request: EXECUTE what the model wrote.

A text that reads well and computes wrong passes every statistic an engine can
run about itself. So the answer is not read for plausibility — the function is
extracted, imported into a fresh interpreter, and run against cases it never
saw. Cases are supplied as a JSON list of [args, expected].

    python tools/run_the_code_it_wrote.py answer.txt --function merge_intervals \\
        --cases cases.json > verdict.json

Exit 0 only when every case returns what it must.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract(answer: str, function: str) -> str:
    """The code the answer carries: a fenced block if there is one, else the
    whole text. Fences are stripped, prose outside them dropped."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.S)
    if blocks:
        for b in blocks:
            if f"def {function}" in b:
                return b
        return blocks[0]
    i = answer.find(f"def {function}")
    return answer[i:] if i >= 0 else answer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answer")
    ap.add_argument("--function", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--code-out")
    a = ap.parse_args()
    answer = Path(a.answer).read_text()
    code = extract(answer, a.function)
    if a.code_out:
        Path(a.code_out).write_text(code)
    cases = json.loads(Path(a.cases).read_text())
    runner = (
        f"{code}\n\n"
        "import json, sys\n"
        f"_cases = json.loads(sys.argv[1])\n"
        "_out = []\n"
        "for args, expected in _cases:\n"
        "    try:\n"
        f"        got = {a.function}(*args)\n"
        "        try:\n"
        "            got = [list(x) for x in got]\n"
        "        except Exception:\n"
        "            pass\n"
        "        _out.append({'args': args, 'expected': expected, 'got': got, 'ok': got == expected})\n"
        "    except Exception as exc:\n"
        "        _out.append({'args': args, 'expected': expected, 'error': f'{type(exc).__name__}: {exc}', 'ok': False})\n"
        "print(json.dumps(_out))\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(runner)
        path = fh.name
    r = subprocess.run([sys.executable, path, json.dumps(cases)], capture_output=True, text=True, timeout=60)
    rec = {"function": a.function, "answer_chars": len(answer), "code_chars": len(code),
           "extracted_a_definition": f"def {a.function}" in code}
    if r.returncode != 0:
        rec["verdict"] = "FAIL"
        rec["error"] = (r.stderr or "").strip()[-600:]
    else:
        results = json.loads(r.stdout.strip().splitlines()[-1])
        rec["results"] = results
        rec["passed"] = sum(1 for x in results if x["ok"])
        rec["of"] = len(results)
        rec["verdict"] = "PASS" if rec["passed"] == rec["of"] else "FAIL"
    print(json.dumps(rec, indent=1))
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
