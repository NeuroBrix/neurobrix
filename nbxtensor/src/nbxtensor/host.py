"""What the application that runs nbxtensor provides to it: the facts and policies the library does not own.

nbxtensor computes; the application decides. The decisions still read here: the hardware vendor's files (tiles,
budgets, the matrix unit) and where they live, the dtype policy that picks an attention's and a matrix unit's operand
dtypes, and where a backend build may live. Each is moving to a fact the library reads itself or a setting with a
default (docs/nbxtensor/API.md section 11); what left already: the memory limit (`memory.set_limit`), the certified
store (`ops.autotune_settings.set_store`), the device's memory (`backend.device_facts`), the launch observer
(`ops.launch_observer`). The application provides each service ONCE, by name (`provide`); the library reads it by name
(`need`). A service no application provided is refused by name, never guessed.

Each service is a callable. An application provides a function that reads its own state at call time, so a value it
changes later is the one read.

Torch-free, numpy-free, and nothing here imports the application.
"""
from __future__ import annotations

from typing import Callable, Dict

#: the services this library reads, and what each one answers — the contract an application fulfils
SERVICES: Dict[str, str] = {
    "vendors_dir": "() -> the directory of the application's own vendor files (pathlib.Path); optional: absent, the library's device files alone",
}

#: hooks an application MAY provide; absent, the library runs as the value here describes (never a guessed fact)
_ABSENT: dict = {}

_PROVIDED: Dict[str, Callable] = {}


def provide(name: str, fn: Callable) -> None:
    """The application provides service `name` (one of `SERVICES`). Providing it again replaces it."""
    if name not in SERVICES:
        raise KeyError(f"nbxtensor.host: {name!r} is not a service this library reads (known: {', '.join(sorted(SERVICES))})")
    if not callable(fn):
        raise TypeError(f"nbxtensor.host: service {name!r} must be callable, got {type(fn).__name__}")
    _PROVIDED[name] = fn


def need(name: str) -> Callable:
    """Service `name` as the application provided it; refused by name when none did."""
    try:
        return _PROVIDED[name]
    except KeyError:
        if name in _ABSENT:
            return _ABSENT[name]
        if name not in SERVICES:
            raise KeyError(f"nbxtensor.host: {name!r} is not a service this library reads") from None
        raise RuntimeError(f"ZERO FALLBACK: nbxtensor needs the service {name!r} ({SERVICES[name]}) and the application "
                           f"running it provided none (nbxtensor.host.provide)") from None


def provided() -> Dict[str, Callable]:
    """The services provided so far (a copy)."""
    return dict(_PROVIDED)
