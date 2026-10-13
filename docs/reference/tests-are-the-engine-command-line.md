# Tests are the engine's command line

Owner's rule, 2026-10-12 19:50. Applies to every machine and every agent of the project.

## 0. Chain of command

The supervisors pilot, not the agent. An inbox entry marked URGENT is executed before any other action, at
once, and its acknowledgement is the first line of the report. Orders are executed in the order they are
given; an agent that chooses to do something else first is at fault. Context hygiene is part of the order:
inbox rotation, a memory of at most 20 files / 60 KB, a short report, the sizes written before the turn ends.

## 1. A model is tested the way a user runs it

The only test of a model is the engine's entry point with the appropriate command and flags — model, prompt,
resolution, steps, frames, temperature, input image or audio file — and the output looked at:

    python -m neurobrix run --model <name> --prompt "..." --steps N --height H --width W [--input-image F]
        [--audio F] [--triton | --sequential | --triton-sequential] --certified-only --output <artefact>

No custom test script. A script is allowed only to chain many identical commands, or to measure a dimension
outside the engine's workflow; it lives in the campaign directory, never in the repository, and it invents no
execution path. The gate harness (`tools/regression_matrix.py`) orchestrates only that exact command and writes
it in every result row, so that any row can be replayed by hand. Every test script on every machine is
inventoried; those that do not respect this are deleted.

## 2. "Validated several times, broken today"

Before a validated model is announced broken, its artefact is compared with its last dated proof, produced by
the SAME command with the same flags, and the comparison says which of the three changed: the container (its
sha256), the engine (its commit), or the request / the test. If it is the test, the defect is the test. Nothing
is fixed by breaking something else: every fix is proven at the command line on the fixed model AND on a
validated neighbouring model before it lands.

## 3. Code hygiene

A modification is complete or does not exist: no unplugged fragment, no old code branch left beside the new one
(several fixes were written without being plugged in, and the engine kept running the obsolete path); the modified
file is cleaned. Every modification is committed and pushed at once to both remotes: git history is the only way
back.

Related: `docs/reference/nothing-is-lost.md`, `docs/reference/confidentiality-and-economy.md`,
`docs/reference/validated-models.md`.
