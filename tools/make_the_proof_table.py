#!/usr/bin/env python3
"""Make the SHIPPED element-wise tile proof table, once, on a machine of the class (the project lead, 24/11/2026).

The library proves each element-wise variant at its first launch in a worker process (nbxtensor `ops/ew_proof.py`),
and keeps the proof in the replay cache. A model that fills its card leaves no room for that worker's context
(deepseek-moe-16b-chat on two full V100 32 GB, register row 444): the cure is a shipped table, made here by running
each model's first launch with a SPARE card of the same class in sight. The library's worker runs on the freest
of the cards the run was given, and the run is given the spare card explicitly (`neurobrix run --proof-card`), so
the spare card hosts the proofs while the model runs on its own cards.

    python tools/make_the_proof_table.py <models file> <draft dir> [--proof-card N] [--cards 0,1] [--hardware ID]

`--proof-card` is OPTIONAL. Without it (a one-card run: a Mac, one board) the proofs run on the LAUNCH CARD, as the
library's own rule says (`nbxtensor.ops.ew_proof_child.proof_device`: a run that declared no card, or a process that
sees one device, has the launch's own card alone): the run is not given `--proof-card`, CUDA_VISIBLE_DEVICES is the
model's cards (`--cards`) or the machine's as given, and each model's line says which card proved.

The models file: one model per line, `<cache entry>` alone (its request is the census's own, derived from the
container: `container_request.derived_request`, the request `certified_census.census_requests` certifies) or
`<cache entry> <request flags>` in shell syntax; `#` starts a comment, a blank line is skipped
(tools/proof_table_models.example.txt).

For each model, in order:
  * a FRESH home under the homes directory (`--homes`, default `<draft dir>.homes`): its own NEUROBRIX_HOME,
    replay cache, TRITON_CACHE_DIR and TMPDIR; the catalogue is read where this machine keeps it (NEUROBRIX_CACHE);
    the machine's license acceptances, when it has any, are copied in (a gated model is not asked again);
  * one first launch through the entry point (`python -m neurobrix run --model M <request> --triton --proof-card K`),
    with CUDA_VISIBLE_DEVICES naming the model's cards (`--cards`) and the proof card, and K the proof card's visible
    ordinal in that list: the run declares its plan's cards and K to the library
    (`cli/engine_choice.run_cards` -> `nbxtensor.ops.ew_proof_child.declare_proof_cards`), the only cards the proof's
    worker may run on; a plan that places the model on the proof card is refused by name by the run;
  * the run's kept proofs merged into the draft (`python -m nbxtensor.ops.ew_proof export <home>/replay_cache/
    ew_tile_proof.json <draft>/<vendor>/<profile>/ew_tile_proof.json`), stamp by stamp; a disagreement stops the
    tool, named;
  * one line: the card that proved (the proof card and its visible ordinal, or the launch card), rc, the variants
    the run proved (its `[ew-proof] this run proved N` line), seconds, refusals (its
    `NOT proven` lines).
At the end: each draft table's record count per stamp, and the stamps.

The proof card reaches the library through the run's own option (`--proof-card`), never through the order of
CUDA_VISIBLE_DEVICES (qa 24/11): the library chooses only among the cards the run declared, the freest by NVML (no
context opened on a card to ask). The tool writes nothing
outside its draft and its homes, and never touches the engine's config directory. A spare card needs a machine with
the proof card beside the model's cards; a one-device host (a Mac) runs without `--proof-card`, on the launch card.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
PROVED = re.compile(r"\[ew-proof\] this run proved (\d+) element-wise variant")
NOT_PROVEN = "NOT proven"


class TableRefused(SystemExit):
    """A step the table cannot take past: a models file line, a card list, or an export that disagrees."""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """The command line: the models file, the draft directory, the proof card, the model's cards, the homes."""
    ap = argparse.ArgumentParser(description="Make the shipped element-wise tile proof table on a spare card.")
    ap.add_argument("models", type=Path, help="one model per line: <cache entry> [request flags]")
    ap.add_argument("draft", type=Path, help="the draft certified directory (<vendor>/<profile>/ew_tile_proof.json)")
    ap.add_argument("--proof-card", type=int, default=None,
                    help="the spare card the proofs run on (board index); absent: the launch card, the library's rule")
    ap.add_argument("--cards", default=None,
                    help="the model's cards, comma-separated board indices (default: CUDA_VISIBLE_DEVICES as given, "
                         "which must name the proof card when one is given)")
    ap.add_argument("--hardware", default=None, help="passed to `neurobrix run --hardware` (default: auto-detected)")
    ap.add_argument("--homes", type=Path, default=None, help="where the fresh homes are made (default <draft>.homes)")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds one first launch may take")
    a = ap.parse_args(argv)
    if a.proof_card is not None and a.proof_card < 0:
        raise TableRefused(f"--proof-card {a.proof_card}: a board index is 0 or more")
    a.card_list = parse_cards(a.cards, a.proof_card)
    if a.homes is None:
        a.homes = a.draft.parent / f"{a.draft.name}.homes"
    return a


def parse_cards(cards: Optional[str], proof_card: Optional[int]) -> Optional[List[str]]:
    """The model's cards as listed (None: CUDA_VISIBLE_DEVICES as given). The proof card among them is refused by
    name: it would host the model, and the proofs would have no spare card."""
    if cards is None:
        return None
    out = [c.strip() for c in cards.split(",") if c.strip()]
    if not out or not all(c.isdigit() for c in out):
        raise TableRefused(f"--cards {cards!r}: comma-separated board indices, at least one")
    if proof_card is not None and str(proof_card) in out:
        raise TableRefused(f"--cards {cards!r} holds the proof card {proof_card}: the proofs need a card the model "
                           "does not run on")
    if len(set(out)) != len(out):
        raise TableRefused(f"--cards {cards!r} names a card twice")
    return out


def visible_devices(card_list: Optional[List[str]], proof_card: Optional[int],
                    given: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """(CUDA_VISIBLE_DEVICES of a model's run, the proof card's visible ordinal in it): its cards and the proof card;
    without `--cards`, the value given, which must name the proof card. The ordinal is what the run is told
    (`--proof-card`): the order of the list does not choose the proof's card. Without a proof card: the model's cards,
    else the value given as it is (None: unset, the machine's own), and no ordinal: the proofs run on the launch card."""
    if proof_card is None:
        return (",".join(card_list) if card_list is not None else given), None
    if card_list is not None:
        entries = card_list + [str(proof_card)]
    else:
        entries = [e.strip() for e in (given or "").split(",") if e.strip()]
        if str(proof_card) not in entries:
            raise TableRefused(f"CUDA_VISIBLE_DEVICES={given!r} does not name the proof card {proof_card}: give "
                               "--cards, or a visible set holding it")
    return ",".join(entries), entries.index(str(proof_card))


def run_command(model: str, request: List[str], proof_ordinal: Optional[int], hardware: Optional[str]) -> List[str]:
    """The first launch of one model: `python -m neurobrix run --model M <request> --triton --proof-card K`, K the
    proof card's visible ordinal (the run declares it to the library with its plan's cards), `--hardware` when given.
    Without an ordinal, no `--proof-card`: the library proves on the launch card."""
    from one_name import cli_name
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", cli_name(model), *request, "--triton"]
    if proof_ordinal is not None:
        cmd += ["--proof-card", str(int(proof_ordinal))]
    return cmd + (["--hardware", hardware] if hardware else [])


def read_models(path: Path) -> List[Tuple[str, Optional[List[str]]]]:
    """[(cache entry, request flags or None)] in file order: None is the census's own request, derived later from
    the container. A line whose flags do not parse, or a model listed twice, is refused by name with its line."""
    out, seen = [], set()
    for no, raw in enumerate(path.read_text().splitlines(), 1):
        try:
            words = shlex.split(raw, comments=True)       # a `#` inside quotes is the request's, not a comment
        except ValueError as exc:
            raise TableRefused(f"{path}:{no}: {exc}: {raw!r}") from exc
        if not words:
            continue
        model, flags = words[0], words[1:]
        if model.startswith("-"):
            raise TableRefused(f"{path}:{no}: a line starts with the model's cache entry, not a flag: {raw!r}")
        if model in seen:
            raise TableRefused(f"{path}:{no}: {model} is listed twice")
        seen.add(model)
        out.append((model, flags or None))
    if not out:
        raise TableRefused(f"{path}: no model listed")
    return out


def census_request(model: str) -> List[str]:
    """The census's own request of a model: derived from its container as the certification census derived it."""
    sys.path.insert(0, str(REPO / "tools"))
    import container_request
    return list(container_request.derived_request(model))


def fresh_home(homes: Path, model: str) -> dict:
    """A new home for one model's run (its directory named by the model and the UTC second, never reused): the
    environment entries that put the engine's home, its replay cache, Triton's cache and the temporary files in it."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = homes / f"{model}-{stamp}"
    if root.exists():
        raise TableRefused(f"{root} exists: a home is made fresh, never reused")
    for sub in ("home", "home/replay_cache", "triton_cache", "tmp"):
        (root / sub).mkdir(parents=True)
    return {"root": root, "NEUROBRIX_HOME": str(root / "home"),
            "NEUROBRIX_REPLAY_CACHE": str(root / "home" / "replay_cache"),
            "TRITON_CACHE_DIR": str(root / "triton_cache"), "TMPDIR": str(root / "tmp")}


def run_env(home: dict, visible: Optional[str], catalogue: str, base: Optional[dict] = None) -> dict:
    """The environment of one model's first launch: this machine's, the fresh home's entries over it, the visible
    cards (None: the machine's own, left as they are), the catalogue, this tree's engine first on the path."""
    env = dict(os.environ if base is None else base)
    env.update({k: v for k, v in home.items() if k != "root"})
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    env["NEUROBRIX_CACHE"] = catalogue
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src")] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                                               if p])
    return env


def kept_destination(kept: Path, draft: Path) -> Path:
    """The draft table a run's kept proofs merge into: <draft>/<vendor>/<profile>/ew_tile_proof.json, read from the
    kept file's own stamps; kept proofs of two device classes are refused by name."""
    doc = json.loads(kept.read_text())
    classes = {(b["stamp"]["vendor"], b["stamp"]["profile"]) for b in doc.get("stamps", {}).values()}
    if len(classes) != 1:
        raise TableRefused(f"{kept}: its proofs name {len(classes)} device class(es) {sorted(classes)}, one expected")
    vendor, profile = classes.pop()
    return draft / vendor / profile / "ew_tile_proof.json"


def summary_of(log_text: str) -> Tuple[Optional[int], int]:
    """(variants the run proved, from its `this run proved N` line; None when it printed none, refusals: its
    `NOT proven` lines)."""
    found = PROVED.findall(log_text)
    return (int(found[-1]) if found else None), sum(NOT_PROVEN in line for line in log_text.splitlines())


def one_model(a, model: str, flags: Optional[List[str]], catalogue: str, licenses: Optional[Path]) -> str:
    """One model: a fresh home, its first launch, the export of its kept proofs into the draft; its line."""
    request = flags if flags is not None else census_request(model)
    home = fresh_home(a.homes, model)
    if licenses is not None and licenses.is_file():
        shutil.copy2(licenses, Path(home["NEUROBRIX_HOME"]) / licenses.name)
    visible, proof_ordinal = visible_devices(a.card_list, a.proof_card, os.environ.get("CUDA_VISIBLE_DEVICES"))
    cmd = run_command(model, request, proof_ordinal, a.hardware)
    log = home["root"] / "run.log"
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write("$ " + ("" if visible is None else f"CUDA_VISIBLE_DEVICES={visible} ") + f"{shlex.join(cmd)}\n")
        fh.flush()
        try:
            rc = subprocess.run(cmd, env=run_env(home, visible, catalogue), cwd=str(home["root"]), stdout=fh,
                                stderr=subprocess.STDOUT, timeout=a.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    seconds = time.time() - t0
    proved, refused = summary_of(log.read_text(errors="replace"))
    kept = Path(home["NEUROBRIX_REPLAY_CACHE"]) / "ew_tile_proof.json"
    merged = "no kept proofs"
    if kept.is_file():
        dest = kept_destination(kept, a.draft)
        ex = subprocess.run([sys.executable, "-m", "nbxtensor.ops.ew_proof", "export", str(kept), str(dest)],
                            env=run_env(home, visible, catalogue), capture_output=True, text=True)
        if ex.returncode != 0:
            raise TableRefused(f"{model}: the export into {dest} refused (rc {ex.returncode}): "
                               f"{(ex.stderr or ex.stdout).strip()[-1200:]}")
        merged = ex.stdout.strip()
    return (f"[proof-table] {model}: proved on {proving_card(a.proof_card, proof_ordinal)}, rc {rc}, proved {'?' if proved is None else proved}, {seconds:.0f} s, "
            f"refused {refused}; {merged}; log {log}")


def proving_card(proof_card: Optional[int], proof_ordinal: Optional[int]) -> str:
    """The card that proved, in words: the proof card (board index, visible ordinal), or the launch card."""
    if proof_card is None:
        return "the launch card (no --proof-card: the library's rule, the launch's own card)"
    return f"the proof card {proof_card} (visible ordinal {proof_ordinal})"


def draft_summary(draft: Path) -> List[str]:
    """Each draft table's records per stamp, and the stamp."""
    lines = []
    for table in sorted(draft.glob("*/*/ew_tile_proof.json")):
        doc = json.loads(table.read_text())
        for key, block in sorted(doc.get("stamps", {}).items()):
            lines.append(f"[proof-table] {table.relative_to(draft)}: stamp {key}: {len(block.get('variants', {}))} "
                         f"record(s); {json.dumps(block.get('stamp'), sort_keys=True)}")
    return lines or [f"[proof-table] {draft}: no table"]


def main(argv: Optional[List[str]] = None) -> int:
    """Read the models, run each one's first launch in a fresh home, merge its proofs, say the draft."""
    a = parse_args(argv)
    models = read_models(a.models)
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "tools"))
    from neurobrix.core.paths import cache_dir, engine_home
    catalogue = str(cache_dir())
    licenses = engine_home() / "license_acceptances.json"
    a.draft.mkdir(parents=True, exist_ok=True)
    a.homes.mkdir(parents=True, exist_ok=True)
    for model, flags in models:
        print(one_model(a, model, flags, catalogue, licenses), flush=True)
    for line in draft_summary(a.draft):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
