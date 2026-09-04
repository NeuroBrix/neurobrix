"""The category vocabulary belongs to the registry, not to the CLI.

Reported by the hub walkthrough of 2026-09-03, against 0.5.2 from PyPI:

    $ neurobrix hub --category TTS
    error: argument --category/-c: invalid choice: 'TTS'
      (choose from IMAGE, VIDEO, AUDIO, SPEECH, LLM, ...)

The CLI carried a *copy* of the taxonomy in an argparse `choices=` list. The
copy drifted: it still offered `AUDIO` and `SPEECH`, which the registry had
retired and answers with a 400, and it had no value at all for `TTS`, `STT`
or `AUDIO_LLM`. **11 of the 45 published models could not be reached by
category** — the three audio modalities, which is exactly the support users
were concluding did not exist.

A hardcoded copy of someone else's vocabulary drifts by construction, so the
fix is not to correct the list: it is to stop keeping one. The value is
validated by the registry, which publishes its vocabulary on every
rejection.

These pins hold both halves — the CLI accepts anything the registry serves,
and a rejection is reported as a rejection rather than as a network failure.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from io import BytesIO

import pytest




# The ten categories the registry serves, recorded from its own 400 body on
# 2026-09-03. Used to pin CLI ACCEPTANCE offline; the live check below is what
# catches the registry adding an eleventh.
REGISTRY_CATEGORIES = [
    "LLM", "CODE", "VLM", "MULTIMODAL", "TTS", "STT",
    "AUDIO_LLM", "IMAGE", "UPSCALER", "VIDEO",
]

# The three that were unreachable, called out because they are the reason the
# walkthrough was commissioned.
PREVIOUSLY_UNREACHABLE = ["TTS", "STT", "AUDIO_LLM"]


def _parser():
    from neurobrix.cli import create_parser

    return create_parser()


def test_the_cli_keeps_no_copy_of_the_taxonomy():
    """The structural pin: no `choices=` on --category, ever again.

    Correcting the list would fix today and drift tomorrow. Not having a list
    is what makes the drift impossible."""
    import inspect

    import neurobrix.cli as cli

    source = inspect.getsource(cli)
    start = source.index("'--category'")
    window = source[start:start + 600]
    assert "choices=" not in window, (
        "--category has a hardcoded choices= list again. The registry owns "
        "this vocabulary; a copy here drifts and silently hides models."
    )


@pytest.mark.parametrize("category", REGISTRY_CATEGORIES)
def test_every_registry_category_is_accepted(category, monkeypatch):
    """Every category the registry serves must reach the request.

    Parsing must not reject it, which is what argparse `choices` used to do
    before the request was ever made."""
    parser = _parser()
    args = parser.parse_args(["hub", "--category", category])
    assert args.category == category


@pytest.mark.parametrize("category", PREVIOUSLY_UNREACHABLE)
def test_the_three_audio_categories_specifically(category):
    """Named separately: these three were the 11 unreachable models."""
    parser = _parser()
    assert parser.parse_args(["hub", "-c", category]).category == category


def test_lowercase_is_still_accepted():
    """The old list carried lowercase variants; dropping it must not remove
    that convenience — the command upper-cases before the request."""
    parser = _parser()
    assert parser.parse_args(["hub", "-c", "tts"]).category == "tts"


# --- the rejection is a rejection, not a network failure --------------------

def _http_error(code: int, body: dict) -> urllib.error.HTTPError:
    payload = json.dumps(body).encode()
    return urllib.error.HTTPError(
        url="https://neurobrix.es/api/models", code=code, msg="Bad Request",
        hdrs=None, fp=BytesIO(payload),  # type: ignore[arg-type]
    )


def test_an_invalid_category_reports_the_registry_reason(monkeypatch, capsys):
    """It answered, and it answered usefully. Reporting that as 'cannot
    connect' sends someone to check their firewall over a typo."""
    from neurobrix.cli.commands import registry as reg

    def raise_400(*_a, **_kw):
        raise _http_error(400, {
            "error": "unknown category 'AUDIO'",
            "validCategories": REGISTRY_CATEGORIES,
        })

    monkeypatch.setattr(urllib.request, "urlopen", raise_400)

    class Args:
        registry = None
        category = "AUDIO"
        search = None
        installed = False

    with pytest.raises(SystemExit) as excinfo:
        reg.cmd_hub(Args())

    out = capsys.readouterr().out
    assert excinfo.value.code == 2, "a bad request is a usage error, not a transport error"
    assert "unknown category 'AUDIO'" in out, "must print the registry's reason"
    assert "TTS" in out and "AUDIO_LLM" in out, "must print the valid vocabulary"
    assert "Cannot connect" not in out, "must not blame the network"


def test_a_real_transport_failure_still_says_cannot_connect(monkeypatch, capsys):
    """The opposite error must not be swallowed by the new branch."""
    from neurobrix.cli.commands import registry as reg

    def raise_urlerror(*_a, **_kw):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(urllib.request, "urlopen", raise_urlerror)

    class Args:
        registry = None
        category = None
        search = None
        installed = False

    with pytest.raises(SystemExit) as excinfo:
        reg.cmd_hub(Args())

    out = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "Cannot connect to registry" in out


@pytest.mark.network
def test_the_recorded_vocabulary_still_matches_the_live_registry():
    """Catches the registry adding or retiring a category.

    Skipped without network; when it runs and fails, the fix is to update
    REGISTRY_CATEGORIES here — the CLI itself needs no change, which is the
    whole point of the design."""
    try:
        request = urllib.request.Request(
            "https://neurobrix.es/api/models?category=__invalid__",
            headers={"User-Agent": "neurobrix-tests"},
        )
        urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode())
        live = body.get("validCategories")
        assert live, "the registry no longer publishes validCategories on a 400"
        assert sorted(live) == sorted(REGISTRY_CATEGORIES), (
            f"registry vocabulary changed: {sorted(live)}"
        )
    except urllib.error.URLError:
        pytest.skip("registry unreachable")
    else:
        pytest.fail("the registry accepted an invalid category")
