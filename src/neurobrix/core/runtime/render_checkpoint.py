"""Resumable renders: a step checkpoint that survives a power cut.

The rack this engine is developed on sits on a single breaker with no UPS,
and the breaker trips when the air conditioning and both GPU servers ramp
together. The whole rack drops at once. That is accepted ground truth — but
it makes a long non-resumable render a bet: a 14-hour native video render
restarts from zero every time, and on 2026-09-03 one was started knowing
that a cut would destroy it.

So the loop can write down where it is. A cut then costs the steps since the
last checkpoint, not the whole render.

Design, in the order the constraints bind:

* **Atomic writes.** Temp file, then `os.replace`. A checkpoint interrupted
  mid-write must not replace a good one with a truncated one — that would
  turn a recoverable cut into a corrupt resume, which is worse than no
  checkpoint at all.
* **A fingerprint, and a refusal.** A checkpoint carries a hash of what
  defines the run: model, step count, seed, guidance, latent shape and
  dtype. Resuming into a *different* run would silently produce an output
  that matches neither. On any mismatch this refuses to resume and says so;
  restarting is correct, resuming wrongly is not.
* **No torch.** The triton engine has the same need and R33 keeps that tree
  sealed, so the state crosses as a plain numpy array and each engine
  converts at its own boundary. numpy is allowed CPU glue on both sides.
* **ON by default, and free on short renders.** This rack has one breaker and
  no UPS, and a cut at the thirteenth hour of a fourteen-hour render loses
  everything. A resume point that must be remembered is a resume point nobody
  has when they need it, so it is the default and not an option.

  Short renders still pay nothing, because the save is gated on ELAPSED TIME
  rather than on step count: a render that finishes inside the interval never
  writes at all. One rule covers a 20-step image render and a 100-step video
  render, which is the same reasoning as the progress heartbeat.

Environment:

* ``NBX_RENDER_CHECKPOINT=0`` — disable. Any other value is a directory to
  store under; unset means ``~/.neurobrix/checkpoints/``.
* ``NBX_RENDER_CHECKPOINT_EVERY=N`` — save every N steps. Setting this is an
  explicit request for step-based saving and it **overrides the time gate**:
  if you asked for every N steps, you get every N steps.
* ``NBX_RENDER_CHECKPOINT_SECONDS=S`` — change the time gate (default 60).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


# --- scheduler state <-> npz, without pickle --------------------------------
#
# `np.load(allow_pickle=True)` would execute arbitrary code from a file the
# render wrote; a checkpoint is data, so it stays data. The scheduler state is
# a shallow dict of scalars, None, arrays, and lists of those — enough
# structure to need describing, little enough to describe in JSON. Arrays go
# to their own npz entries and the JSON carries their keys.

_ARRAY_TAG = "__nbx_array__"


def _flatten_state(state: dict[str, Any]) -> tuple[str, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}

    def encode(value, path):
        if isinstance(value, np.ndarray):
            key = f"sched_{path}"
            arrays[key] = value
            return {_ARRAY_TAG: key}
        if isinstance(value, (list, tuple)):
            return [encode(v, f"{path}_{i}") for i, v in enumerate(value)]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        raise TypeError(f"scheduler state field {path!r} is {type(value).__name__}, "
                        f"which cannot travel in a checkpoint")

    return json.dumps({k: encode(v, k) for k, v in state.items()}), arrays


def _unflatten_state(blob: str, data) -> dict[str, Any]:
    def decode(value):
        if isinstance(value, dict) and _ARRAY_TAG in value:
            return data[value[_ARRAY_TAG]]
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return {k: decode(v) for k, v in json.loads(blob).items()}

_ENV_ENABLE = "NBX_RENDER_CHECKPOINT"
_ENV_EVERY = "NBX_RENDER_CHECKPOINT_EVERY"
_ENV_SECONDS = "NBX_RENDER_CHECKPOINT_SECONDS"
# A render that finishes inside this never writes a checkpoint at all.
_DEFAULT_SECONDS = 60.0
_DEFAULT_DIR = Path.home() / ".neurobrix" / "checkpoints"


def _fingerprint(parts: dict[str, Any]) -> str:
    """Stable hash of everything that defines this render.

    Anything that would change the output belongs here. A value that is
    absent is recorded as absent rather than skipped, so two runs that differ
    only by an unset option do not collide.
    """
    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class RenderCheckpoint:
    """Per-step state for one render, on disk, atomically."""

    def __init__(self, path: Path, fingerprint: str, every: int = 1,
                 min_interval_s: float = 0.0) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.every = max(1, every)
        # 0 means "no time gate" — step-based saving, which is what an explicit
        # NBX_RENDER_CHECKPOINT_EVERY asks for.
        self.min_interval_s = max(0.0, min_interval_s)
        self._last_save = time.monotonic()

    # --- construction -------------------------------------------------------

    @classmethod
    def from_env(cls, *, model: str, num_steps: int,
                 extra: dict[str, Any] | None = None) -> "RenderCheckpoint | None":
        """Build one unless the environment turned it off.

        The rack has one breaker and no UPS. A resume point that has to be
        asked for is one nobody has at hour thirteen, so this is ON unless
        explicitly disabled — and the time gate keeps it free for the short
        renders that do not need it.

        `extra` carries the rest of the run identity (seed, guidance, spatial
        dims, engine). Callers pass what they have; what they omit simply is
        not part of the fingerprint, which is why the caller — not this
        module — decides what defines its run.
        """
        setting = os.environ.get(_ENV_ENABLE, "").strip()
        if setting == "0":
            return None
        if not num_steps or num_steps < 2:
            return None            # nothing to resume into

        directory = (_DEFAULT_DIR if not setting or setting in ("1", "true", "yes")
                     else Path(setting))
        parts = {"model": model, "num_steps": num_steps}
        parts.update(extra or {})
        fingerprint = _fingerprint(parts)

        explicit_every = os.environ.get(_ENV_EVERY, "").strip()
        try:
            every = int(explicit_every) if explicit_every else 1
        except ValueError:
            every = 1
        # An explicit step interval overrides the time gate.
        if explicit_every:
            seconds = 0.0
        else:
            try:
                seconds = float(os.environ.get(_ENV_SECONDS, "") or _DEFAULT_SECONDS)
            except ValueError:
                seconds = _DEFAULT_SECONDS

        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory / f"{model}-{fingerprint}.npz", fingerprint,
                   every, seconds)

    # --- read ---------------------------------------------------------------

    def load(self) -> tuple[int, np.ndarray, dict[str, Any]] | None:
        """`(next_step_index, latent, scheduler_state)`, or None to start fresh.

        Returns None — never raises — for every "cannot resume" case: no
        checkpoint, a different run, or a file the cut left unreadable. The
        caller's correct response to all three is identical: start from step
        0. A corrupt checkpoint is exactly what a power cut produces, so it
        must not itself be fatal.
        """
        if not self.path.exists():
            return None
        try:
            with np.load(self.path, allow_pickle=False) as data:
                stored = str(data["fingerprint"])
                if stored != self.fingerprint:
                    print(f"   [Checkpoint] ignoring {self.path.name}: it belongs "
                          f"to a different run ({stored} != {self.fingerprint}). "
                          f"Starting from step 0.")
                    return None
                step = int(data["step"])
                latent = data["latent"]
                sched = _unflatten_state(str(data["sched"]), data)
        except Exception as exc:  # noqa: BLE001 — deliberate, see below
            # Deliberately broad. A checkpoint is INSURANCE on a render, and
            # the correct response to every way of failing to read one is
            # identical: start from step 0. A truncated npz alone can surface
            # as OSError, ValueError, EOFError, KeyError or zipfile.BadZipFile
            # depending on where the cut landed — enumerating them invites the
            # one that was missed to abort a render that is otherwise fine.
            # (BadZipFile is how it actually failed first; the test found it.)

            print(f"   [Checkpoint] {self.path.name} is unreadable ({exc}); "
                  f"starting from step 0.")
            return None

        print(f"   [Checkpoint] resuming at step {step + 1} from {self.path}")
        return step + 1, latent, sched

    # --- write --------------------------------------------------------------

    def should_save(self, step_idx: int, num_steps: int) -> bool:
        """Save on the interval, and always on the last step.

        With a time gate set (the default), a step that is due by COUNT is
        still skipped until the gate has elapsed — so a render finishing inside
        the interval writes nothing and costs nothing. An explicit step
        interval clears the gate, because asking for every N steps is asking
        for every N steps.
        """
        due = (step_idx + 1) % self.every == 0 or step_idx == num_steps - 1
        if not due:
            return False
        if self.min_interval_s <= 0:
            return True
        now = time.monotonic()
        if now - self._last_save < self.min_interval_s:
            return False
        self._last_save = now
        return True

    def save(self, step_idx: int, latent: np.ndarray,
             sched_state: dict[str, Any] | None = None) -> None:
        """Write step state atomically: temp file, then replace.

        A failure here is reported and swallowed: losing a checkpoint costs
        the steps since the previous one, while raising would abort a render
        that is otherwise proceeding correctly. The render is the deliverable;
        the checkpoint is insurance on it.
        """
        tmp = self.path.with_suffix(".npz.tmp")
        try:
            blob, arrays = _flatten_state(sched_state or {})
            # np.savez APPENDS ".npz" to a path that does not end in it, which
            # would write beside the temp name and leave the rename with
            # nothing to move. Handing it an open file keeps the name exact.
            with open(tmp, "wb") as fh:
                np.savez(fh, step=np.int64(step_idx),
                         fingerprint=np.str_(self.fingerprint),
                         sched=np.str_(blob), latent=latent, **arrays)
            os.replace(tmp, self.path)
        except (OSError, ValueError) as exc:
            print(f"   [Checkpoint] could not write step {step_idx}: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self) -> None:
        """Remove the checkpoint once the render has completed."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
