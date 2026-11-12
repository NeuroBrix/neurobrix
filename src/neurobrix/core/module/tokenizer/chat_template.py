"""The environment a chat template is rendered in: the one its vendor's reference renders it in.

Contract. `environment()` gives a jinja2 environment configured as transformers configures its own
(src/transformers/utils/chat_template_utils.py):
- sandboxed and immutable (a template is data carried by a container: attribute access to Python's objects and
  in-place mutation of what it is given are refused with jinja2's SecurityError);
- `trim_blocks` and `lstrip_blocks`; the `loopcontrols` extension (`{% break %}`, `{% continue %}`);
- the reference's `tojson` (json.dumps, `ensure_ascii` False by default, never HTML-escaped, and the reference's
  arguments `ensure_ascii`, `indent`, `separators`, `sort_keys`);
- the two globals a template may call, `raise_exception` (the template's own refusal, a TemplateError) and
  `strftime_now`.
It refuses by name when jinja2 is not installed. It renders nothing itself; the tokenizer renders through it.
The reference's `generation` tag (it marks assistant spans for training masks) is not provided: a template using it
fails to parse with jinja2's own message.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import partial

# The reference's filter: json.dumps with ensure_ascii False unless the template asks; keys in their given order
_tojson = partial(json.dumps, ensure_ascii=False)


def environment():
    try:
        from jinja2 import exceptions, ext, sandbox
    except ImportError as e:
        raise RuntimeError(f"ZERO FALLBACK: jinja2 required for chat template.\nInstall with: pip install jinja2 ({e})") from e

    def raise_exception(message):
        raise exceptions.TemplateError(message)

    env = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals.update(raise_exception=raise_exception, strftime_now=lambda fmt: datetime.now().strftime(fmt))
    return env
