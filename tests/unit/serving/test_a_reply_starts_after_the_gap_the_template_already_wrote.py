"""A reply starts after the gap its chat template already wrote: the template's own copy of the gap is removed once,
at the reply's boundary, and the history re-renders to the prompt the model saw.

Before (the Dell chats of 2026-11-22, deepseek-moe-16b-chat and DeepSeek-Coder-V2-Lite-Instruct): the template ends the
generation prompt with `Assistant:` and writes `'Assistant: ' + content` for a past turn, so the model's first token
carries the space (`ĠHello`); the reply decoded alone was `' Hello...'`, stored verbatim, the history then read
`Assistant:  Hello` (two spaces), and the third turn answered with two spaces. One shared function
(`core/runtime/reply_boundary`) reads the gap from the template by rendering one history turn with a sentinel content, and
removes only the part of it the reply starts with: the display, the stored history, the API's content and the
streamer's first delta come out the same.

What this stands on: the DeepSeek template's text as the container carries it (tokenizer_config.json), rendered by the
engine's own renderer, and a table tokenizer (ids -> pieces); no model, no device.

What this test does if the code were wrong: the space stays on the reply (red), a blanket strip eats a reply's own
leading space under a template with no gap (red), a reply with no space loses a character (red), or the streamer's
text differs from the displayed text (red).
"""
import pytest

pytest.importorskip("jinja2")

from types import SimpleNamespace as NS  # noqa: E402

from neurobrix.core.module.tokenizer.sp_tokenizer import HFTokenizer  # noqa: E402
from neurobrix.core.runtime.reply_boundary import reply_gap, without_reply_gap  # noqa: E402
from neurobrix.serving.text_stream import TextDeltaDecoder  # noqa: E402

# deepseek-moe-16b-chat, modules/tokenizer/tokenizer_config.json: chat_template, bos_token, eos_token (verbatim)
DEEPSEEK = ("{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
            "{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}"
            "{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}"
            "{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}")
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
# a template that writes a marker for a past turn that its generation prompt does not (Qwen3's way)
MARKED = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.role == 'assistant' %}<think>\n\n</think>\n\n"
          "{% endif %}{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

PIECES = {1: " Hello", 2: "!", 3: " How", 4: " can", 5: " I", 6: " help", 7: "?", 10: "Hello", 99: EOS}
SPECIAL = {99}


class _Tok(HFTokenizer):
    """The engine's renderer over a template, and a decode that reads pieces from a table."""

    def __init__(self, template, bos="", eos=""):
        self._chat_template, self._bos_token, self._eos_token = template, bos, eos

    def decode(self, ids, skip_special_tokens=False):
        return "".join(PIECES[i] for i in ids if not (skip_special_tokens and i in SPECIAL))


def _engine_text(tok, ids):
    """What the engine returns as a turn's text (serving/engine._extract_llm_output)."""
    from neurobrix.serving.engine import InferenceEngine
    me = NS(_executor=NS(modules={"tokenizer": tok}))
    return InferenceEngine._extract_llm_output(me, {"output_tokens": ids})["text"]


REPLY = [1, 2, 3, 4, 5, 6, 7]          # ' Hello! How can I help?', the first token carrying the space
USER = [{"role": "user", "content": "Hi"}]


def test_the_deepseek_templates_gap_is_one_space_read_from_the_template():
    assert reply_gap(_Tok(DEEPSEEK, BOS, EOS)) == " "


def test_a_deepseek_reply_that_starts_with_the_gap_comes_back_without_it_and_the_history_is_the_prompt_the_model_saw():
    tok = _Tok(DEEPSEEK, BOS, EOS)
    text = _engine_text(tok, REPLY + [99])
    assert text == "Hello! How can I help?"
    prompt = tok.apply_chat_template(USER, tokenize=False, add_generation_prompt=True)
    assert prompt == f"{BOS}User: Hi\n\nAssistant:"
    saw = prompt + " Hello! How can I help?" + EOS                     # the prompt and the reply's ids, as the model held them
    history = tok.apply_chat_template(USER + [{"role": "assistant", "content": text}], tokenize=False,
                                      add_generation_prompt=False)
    assert history == saw                                              # one space, never two
    assert "Assistant: Hello" in history and "Assistant:  Hello" not in history
    again = tok.apply_chat_template(USER + [{"role": "assistant", "content": text}, {"role": "user", "content": "Again"}],
                                    tokenize=False, add_generation_prompt=True)
    assert again.startswith(saw)


def test_a_template_whose_generation_prompt_ends_after_its_gap_removes_nothing():
    tok = _Tok(CHATML)
    assert reply_gap(tok) == ""
    assert _engine_text(tok, REPLY) == " Hello! How can I help?"       # the reply's own leading space is the model's
    assert _engine_text(tok, [10, 2]) == "Hello!"


def test_a_marker_a_template_writes_only_in_history_is_not_a_gap():
    assert reply_gap(_Tok(MARKED)) == ""


def test_a_reply_whose_first_token_carries_no_space_under_a_space_gap_removes_nothing():
    tok = _Tok(DEEPSEEK, BOS, EOS)
    assert _engine_text(tok, [10, 2, 3, 4, 5, 6, 7]) == "Hello! How can I help?"


def test_only_the_common_prefix_of_the_gap_and_the_reply_is_removed_once_never_a_blanket_strip():
    assert without_reply_gap("  Yes", " ") == " Yes"
    assert without_reply_gap("Yes ", " ") == "Yes "
    assert without_reply_gap(" Yes", "") == " Yes"
    assert without_reply_gap("\n\nYes", "\n\n") == "Yes" and without_reply_gap("\nYes", "\n\n") == "Yes"
    assert without_reply_gap("", " ") == ""


def test_the_streamers_text_is_the_displayed_text_and_its_first_delta_is_the_first_piece_of_it():
    tok = _Tok(DEEPSEEK, BOS, EOS)
    shown = _engine_text(tok, REPLY + [99])
    d = TextDeltaDecoder(tok, gap=reply_gap(tok))
    events = [d.push(i) for i in REPLY]
    assert events[0] == {"text": "Hello", "rewind": 0}                 # not ' Hello'
    assert d.text == shown and all(e["rewind"] == 0 for e in events)
    assert "".join(e["text"] for e in events) == shown


def test_the_openai_turn_reader_releases_the_same_content():
    from neurobrix.serving.openai_api.turn import TurnReader
    tok = _Tok(DEEPSEEK, BOS, EOS)
    reader = TurnReader(tok, gap=reply_gap(tok))
    for i in REPLY:
        reader.push(i)
    reader.finish()
    assert reader.turn.content == _engine_text(tok, REPLY) == "Hello! How can I help?"


def test_a_raw_completion_is_not_a_reply_to_a_template_prompt_and_keeps_its_space():
    from neurobrix.serving.openai_api.turn import TurnReader
    tok = _Tok(DEEPSEEK, BOS, EOS)
    reader = TurnReader(tok)                                           # no gap: the completion route passes none
    for i in REPLY:
        reader.push(i)
    reader.finish()
    assert reader.turn.content == " Hello! How can I help?"


def test_a_tokenizer_with_no_chat_template_has_no_gap():
    class _Plain:
        def decode(self, ids, skip_special_tokens=False):
            return ""
    assert reply_gap(_Plain()) == "" and reply_gap(None) == ""
    t = _Tok(DEEPSEEK)
    t._chat_template = None
    assert reply_gap(t) == ""


def test_the_cli_text_of_a_chat_mode_run_drops_the_gap_and_a_non_chat_run_keeps_it():
    from neurobrix.core.runtime.output_dispatch import _extract_text
    tok = _Tok(DEEPSEEK, BOS, EOS)
    outputs = {"global.output_tokens": REPLY}

    def executor(chat, defaults=None):
        return NS(modules={"tokenizer": tok}, variable_resolver=NS(resolved={} if chat is None else {"global.chat_mode": chat}),
                  pkg=NS(defaults=defaults or {}))
    assert _extract_text(outputs, executor(True)) == "Hello! How can I help?"
    assert _extract_text(outputs, executor(None, {"chat_mode": True})) == "Hello! How can I help?"
    assert _extract_text(outputs, executor(False, {"chat_mode": True})) == " Hello! How can I help?"
    assert _extract_text(outputs, executor(None, {"chat_mode": False})) == " Hello! How can I help?"
    import pytest
    with pytest.raises(RuntimeError, match="declares no chat_mode"):      # neither the request nor the container says
        _extract_text(outputs, executor(None))
