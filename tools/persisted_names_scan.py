"""Persisted names that still carry the old vocabulary: where code writes or reads them, and which containers hold them.

The container format's next step renames every persisted name that carries a word the public vocabulary refuses
(tools/public_vocabulary.py, the same pattern): a key written into a container, a census table or a certificate, a
cache file name, an environment variable, a command-line flag. A persisted name never changes inside a release
lot: the inventory that prepares the format step must see EVERY occurrence, so this scan reads what the code
spells, not what it does.

Two readings:

  source ROOT...       every string literal of every Python file (comments and docstrings apart: a docstring is a
                       literal, it is reported as kind `docstring`), and every key and string value of every
                       .json / .jsonl / .yml / .yaml file under the roots. One TSV row per occurrence:
                       path, line, kind, name (the whole identifier the word sits in), the literal.
  containers ROOT...   every *.nbx container under the roots, opened as the zip it is: only its .json members are
                       read (weights are never opened). One JSON object per container: per member, the keys that
                       carry a refused word with their counts, the string values that carry one, and the member names that
                       carry one.

Usage: python tools/persisted_names_scan.py source src nbxtensor/src > hits.tsv
       python tools/persisted_names_scan.py containers <models root> > containers.jsonl
"""
import ast
import io
import json
import re
import sys
import tokenize
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_vocabulary import BANNED  # noqa: E402

DATA_SUFFIXES = (".json", ".jsonl", ".yml", ".yaml")
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules"}
# the whole name a refused word sits in: an identifier, an environment variable, a dotted or slashed path, a flag
NAME = re.compile(r"[A-Za-z0-9_.\-/]*")
JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"(\s*:)?')
YAML_KEY = re.compile(r"^\s*(?:-\s+)?([^\s:#'\"][^:#]*?|'[^']*'|\"[^\"]*\"):(?:\s|$)")


def name_around(text: str, start: int, end: int) -> str:
    """The whole name the match [start, end) of `text` sits in."""
    left = start
    while left > 0 and NAME.fullmatch(text[left - 1]):
        left -= 1
    right = end
    while right < len(text) and NAME.fullmatch(text[right]):
        right += 1
    return text[left:right].strip("./-")


def _hits(path: str, line: int, kind: str, literal: str) -> list:
    flat = literal.replace("\t", " ").replace("\n", "\\n")
    return [(path, line, kind, name_around(literal, m.start(), m.end()), flat[:200])
            for m in BANNED.finditer(literal)]


def _docstring_lines(text: str) -> set:
    """The first line of every docstring of a module (module, class, function)."""
    lines = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                lines.add(first.lineno)
    return lines


def python_hits(path: str, text: str) -> list:
    """Every refused word inside a string literal of a Python file (comments are not literals); a docstring is told
    apart from a literal by its kind."""
    try:
        docs = _docstring_lines(text)
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        raise ValueError(f"{path}: not a Python file this scan can read ({exc}); scan it by hand") from exc
    found = []
    for tok in tokens:
        if tok.type == tokenize.STRING:
            found += _hits(path, tok.start[0], "docstring" if tok.start[0] in docs else "string", tok.string)
    return found


def data_hits(path: str, text: str) -> list:
    """Every refused word in a key or a string value of a JSON, JSON-lines or YAML file, by line."""
    found = []
    yaml = path.endswith((".yml", ".yaml"))
    for n, line in enumerate(text.splitlines(), 1):
        if yaml:
            if line.lstrip().startswith("#"):
                continue
            body = line.split(" #", 1)[0]
            key = YAML_KEY.match(body)
            if key:
                found += _hits(path, n, "key", key.group(1))
                found += _hits(path, n, "value", body[key.end():])
            else:
                found += _hits(path, n, "value", body)
            continue
        for m in JSON_STRING.finditer(line):
            found += _hits(path, n, "key" if m.group(2) else "value", m.group(1))
    return found


def source(roots: list) -> list:
    rows = []
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*")
                                                       if p.is_file() and not SKIP_DIRS & set(p.parts))
        for f in files:
            if f.suffix == ".py":
                rows += python_hits(str(f), f.read_text(encoding="utf-8"))
            elif f.name.endswith(DATA_SUFFIXES):
                rows += data_hits(str(f), f.read_text(encoding="utf-8"))
    return rows


BANNED_BYTES = re.compile(BANNED.pattern.encode(), re.IGNORECASE)
CHUNK = 1 << 24


def member_words(stream, chunk: int = CHUNK) -> tuple:
    """(keys, values): the JSON strings of one member that carry a refused word, counted, read as a byte stream in
    chunks (a graph can weigh more than a gigabyte: it is never parsed whole). A string holding a refused word is cut
    out between the quotes around the word; it is a key when a colon follows it."""
    keys, values = Counter(), Counter()
    buf, skip = b"", 0
    while True:
        data = stream.read(chunk)
        final = not data
        buf += data
        cut = len(buf) if final else max(0, len(buf) - 4096)
        last_right = -1
        for m in BANNED_BYTES.finditer(buf, skip):
            if m.start() >= cut:
                break                                     # read again with the next chunk, whole
            if m.start() <= last_right:
                continue                                  # a second word of a string already counted
            left, right = buf.rfind(b'"', 0, m.start()), buf.find(b'"', m.end())
            if left < 0 or right < 0:
                continue                                  # not inside a string (a bare word: no JSON string)
            text = buf[left + 1:right].decode("utf-8", "replace")
            (keys if buf[right + 1:right + 64].lstrip().startswith(b":") else values)[text[:200]] += 1
            last_right = right
        if final:
            return keys, values
        # the next chunk starts at the last quote before the cut, unless the string it opens was counted already:
        # a string the cut splits is read whole next time, and nothing before `skip` is counted twice
        quote = buf.rfind(b'"', 0, cut)
        start = quote if quote > last_right else cut
        skip, buf = cut - start, buf[start:]


def container(path: Path) -> dict:
    """The refused words one container holds, member by member; weights are never read."""
    out = {"container": str(path), "members": {}, "json_members": 0}
    with zipfile.ZipFile(path) as z:
        out["member_names"] = [n for n in z.namelist() if BANNED.search(n)]
        for info in z.infolist():
            if not info.filename.endswith(".json"):
                continue
            out["json_members"] += 1
            with z.open(info) as stream:
                keys, values = member_words(stream)
            if keys or values:
                out["members"][info.filename] = {"keys": dict(keys), "values": dict(values.most_common(50)),
                                                 "distinct_values": len(values)}
    return out


def containers(roots: list):
    for root in roots:
        root = Path(root)
        for p in ([root] if root.is_file() else sorted(root.rglob("*.nbx"))):
            yield container(p)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] not in ("source", "containers"):
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "source":
        rows = source(argv[1:])
        print("path\tline\tkind\tname\tliteral")
        for r in rows:
            print("\t".join(map(str, r)))
        print(f"persisted names scan: {len(rows)} occurrences", file=sys.stderr)
    else:
        for c in containers(argv[1:]):
            print(json.dumps(c, sort_keys=True))
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
