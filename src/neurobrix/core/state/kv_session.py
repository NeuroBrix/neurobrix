"""When a request is a serving session's turn — the engine's decision; the session record itself is nbxtensor's
(`nbxtensor.memory.kv_session`, docs/design/prism_kv.md §5bis)."""
from __future__ import annotations

import functools

from nbxtensor.memory.kv_session import forgetting_on_failure


def session_turn(execute):
    """Decorates a flow handler's `execute(self, kv_session)`: the request's session (`self.ctx.kv_session`, None for
    every other request) is handed in, and a failed turn leaves the session holding nothing."""
    @functools.wraps(execute)
    def run(self):
        kv_session = getattr(self.ctx, "kv_session", None)
        return forgetting_on_failure(kv_session, lambda: execute(self, kv_session))
    return run
