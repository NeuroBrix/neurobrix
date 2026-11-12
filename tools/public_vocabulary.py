"""The public vocabulary check: a reader's page never says how a container is made.

The owner's rule (2026-11-11): public text speaks of "building containers" and "the container builder"; it never
names the private build tool and never uses its verb. This check reads every tracked Markdown page and is red on
those words — and every other tracked file of the documentation (the registers kept as data). It is a step of
the release checklist (docs/reference/RELEASING.md) and a unit test.

An occurrence is accepted only inside a text the allow-list names (`tools/public_vocabulary.allow.tsv`: path or `*`,
the exact text, the reason) — the real name of a tracked file, kept until the file is renamed. The list only shrinks.

Usage: python tools/public_vocabulary.py            every occurrence, file:line, exit 1 when there is one
       python tools/public_vocabulary.py --count    occurrences per page
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ALLOW = Path(__file__).with_name("public_vocabulary.allow.tsv")
BINARY = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".wav", ".pdf", ".parquet", ".ico", ".svg")
# banned public vocabulary
TOOL, VERB = "".join(map(chr, (102, 111, 114, 103, 101))), "".join(map(chr, (116, 114, 97, 99, 101)))
BANNED = re.compile(rf"(?<![A-Za-z])(?:{TOOL}|(?:re)?{VERB[:-1]}(?:e|ed|es|er|ers|ing))(?![A-Za-z])", re.IGNORECASE)


def allowed(allow_path: Path = ALLOW) -> list:
    """(path or '*', exact text) pairs; a line without its reason is refused by name."""
    pairs = []
    if not allow_path.exists():
        return pairs
    for n, line in enumerate(allow_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 3 or not all(c.strip() for c in cols):
            raise ValueError(f"{allow_path.name}:{n}: a line is path<TAB>exact text<TAB>reason — got {line!r}")
        pairs.append((cols[0], cols[1]))
    return pairs


def occurrences(path: str, text: str, pairs: list) -> list:
    """(line number, word) of every banned word of `text` that no allowed text of this page covers."""
    texts = [t for p, t in pairs if p in ("*", path)]
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        covered = [(m.start(), m.end()) for t in texts for m in re.finditer(re.escape(t), line)]
        for m in BANNED.finditer(line):
            if not any(a <= m.start() and m.end() <= b for a, b in covered):
                found.append((n, m.group(0)))
    return found


def pages(repo: Path = REPO) -> list:
    """What a reader is given to read: every tracked Markdown page, and every tracked file of the documentation
    whatever its extension (the registers kept as data beside the pages: .jsonl, .tsv, .yml, .json)."""
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "*.md", "docs"], check=True, capture_output=True,
                         text=True).stdout
    return sorted({p for p in out.split("\n") if p and not p.endswith(BINARY)})


def scan(repo: Path = REPO, allow_path: Path = ALLOW) -> dict:
    pairs = allowed(allow_path)
    result = {}
    for page in pages(repo):
        found = occurrences(page, (repo / page).read_text(encoding="utf-8"), pairs)
        if found:
            result[page] = found
    return result


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    result = scan()
    total = sum(len(v) for v in result.values())
    for page, found in sorted(result.items(), key=lambda kv: -len(kv[1])):
        if "--count" in argv:
            print(f"{len(found):5d}  {page}")
        else:
            print("\n".join(f"{page}:{n}: {word}" for n, word in found))
    print(f"public vocabulary: {total} occurrences in {len(result)} pages", file=sys.stderr)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
