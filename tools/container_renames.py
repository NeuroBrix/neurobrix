"""A renamed container keeps its history: the dated records written under its former name.

The owner's rule of 2026-09-28 03:06: a model's name is the model's own name as its maker
publishes it. A publication-format suffix of its Hugging Face repository (`_diffusers`,
`-Diffusers`) says where and in which format the weights are published, not what the model is;
the repository is recorded as the SOURCE (Forge's registry `hf_repo`, the manifest's
`origin.repository`), never as the name. Five catalogue containers carried such a suffix and were
renamed to the model's own name.

A dated record (a campaign's `result.json`, a catalogue pass's `meet.json`, a regression row, a
compiled index of last proofs) is never rewritten: it says what was true on the day it was
written, under the name the container had then. A tool that JOINS such a record to today's
container reads the record's name through `current_name` — one table, here, instead of one alias
per tool. Each line is the owner's decision, dated; nothing is inferred from a name's shape.
"""
from __future__ import annotations

#: former container name -> current container name (renamed 2026-09-28, owner's rule of 03:06).
RENAMED = {
    "SANA-Video_2B_720p_diffusers": "SANA-Video_2B_720p",
    "Wan2.1-I2V-14B-480P-Diffusers": "Wan2.1-I2V-14B-480P",
    "Wan2.1-T2V-1.3B-Diffusers": "Wan2.1-T2V-1.3B",
    "Wan2.1-VACE-1.3B-diffusers": "Wan2.1-VACE-1.3B",
    "Wan2.2-I2V-A14B-Diffusers": "Wan2.2-I2V-A14B",
}


def current_name(recorded: str) -> str:
    """The container a record written under `recorded` belongs to today. A name that was never
    renamed is its own current name."""
    return RENAMED.get(recorded, recorded)


def by_current_name(mapping: dict) -> dict:
    """A name-keyed record (`{container: value}`) keyed by today's names. Where one container
    holds an entry under both names, the one written under its current name is the newer and
    wins; the other is not merged into it."""
    out = {}
    for name, value in mapping.items():
        now = current_name(name)
        if now != name and now in mapping:
            continue
        out[now] = value
    return out
