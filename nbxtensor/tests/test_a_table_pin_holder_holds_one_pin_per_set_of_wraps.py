"""`pointer_tables.TablePins` holds per-call pointer tables' pins for a cache that outlives the call (an expert store):
a table whose wraps are already held ends its own pins at once (the held pin keeps the refcount), a table over new
wraps is held, and `end` ends every held pin. No card: a pin is an object with `_resident_bases` and `__exit__`.

What this test does if the code is wrong: red when a held table's pins are ended at the call (the store's arena would
be re-wrapped at a new address: NaN, 2026-11-08), when a duplicate pin is held (the held list grows by one per decoded
token), or when `end` leaves a pin entered (its wrap keeps an unbound expert's pages resident)."""
from types import SimpleNamespace

from nbxtensor.memory.pointer_tables import TablePins, end_call_pins


class _Pin:
    def __init__(self, *bases):
        self._resident_bases, self.exited = list(bases), False

    def __exit__(self, *a):
        self.exited = True


def test_held_once_per_set_of_wraps_and_ended_together():
    h = TablePins()
    first, again, wider = _Pin(1, 2), _Pin(2), _Pin(2, 3)
    end_call_pins(SimpleNamespace(pins=[first]), h)
    end_call_pins(SimpleNamespace(pins=[again]), h)
    end_call_pins(SimpleNamespace(pins=[wider]), h)
    assert not first.exited and again.exited and not wider.exited
    h.end()
    assert first.exited and wider.exited


def test_without_a_holder_the_call_ends_its_own_pins():
    pin = _Pin(7)
    t = SimpleNamespace(pins=[pin], release_pins=lambda: pin.__exit__())
    end_call_pins(t)
    assert pin.exited
