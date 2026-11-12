# Tool-call dialects — two more: GLM's key/value form and Mistral's `[TOOL_CALLS]name[ARGS]`

**Responsibility.** A dialect parser turns one assistant turn into the tool calls its chat template declares. The
grammar envelope makes the model write exactly the form that parser reads back. Both are chosen by the markers in
the embedded template (R15), never by a model's name.

**Why now.** The next coding models are GLM-4.7-Flash and Devstral Small 2, captured in that order (decided between
the agents on 2026-11-12). Neither writes a dialect the server reads today. Without them, `neurobrix serve` refuses
agent mode for both models, and NBX Code cannot drive them.

## What the vendors write (read at the source, 2026-11-12)

| | GLM-4.7-Flash (`zai-org/GLM-4.7-Flash`, `chat_template.jinja`) | Devstral Small 2 (`mistralai/Devstral-Small-2-24B-Instruct-2512`, `chat_template.jinja`) |
|---|---|---|
| one call | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>` | `[TOOL_CALLS]NAME[ARGS]{JSON object}` |
| a value | a string written bare; anything else `tojson(ensure_ascii=False)` | the whole arguments object `tojson` (`'{}'` when empty) |
| several calls | blocks back to back, no separator | `[TOOL_CALLS]` again for each call, no separator |
| end of the block | `</tool_call>` | none: the calls run to the end of the turn (`eos_token`) |
| the markers in the tokenizer | `<tool_call>` 154843, `</tool_call>` 154844, `<arg_key>` 154847, `</arg_key>` 154848, `<arg_value>` 154849, `</arg_value>` 154850: added tokens, `special: false` | `[TOOL_CALLS]` 9, `[ARGS]` 32: `special: true` (also `[CALL_ID]` 33, which the template never writes) |
| results | `<|observation|><tool_response>…</tool_response>` | `[TOOL_RESULTS]…[/TOOL_RESULTS]` |

The template instructs GLM in its own words: "output the function name and arguments within the following XML
format", followed by the one-call line above.

## The contract change: inner markers are kept

Today a parser declares `start_marker`, `start_markers` and `end_marker`. The streaming reader (`TurnReader`) and the
whole-turn decode (`engine.py`, `MarkerDecoder`) keep exactly those special tokens and skip every other one.

Devstral's `[ARGS]` is a special token inside the block. Skipped, `[TOOL_CALLS]read[ARGS]{"path": "a"}` decodes as
`[TOOL_CALLS]read{"path": "a"}`, and the name no longer has an end.

So `ToolCallParser` gains `inner_markers: tuple = ()` and one method, `kept_markers()`: the openings, the end marker
and the inner markers. Both decode sites call `kept_markers()` in place of the list they assemble today, so no site
keeps a marker list of its own. GLM declares its four argument markers too. They are not special in its tokenizer, so
this changes nothing for it today, but the parser does not depend on that fact.

## Selection (`select_parser`), in this order

1. `<arg_key>` in the template → **glm**. It must come before the `<tool_call>` check, because GLM's template contains
   `<tool_call>` too and would otherwise get the JSON parser.
2. `<function=` → xml (unchanged).
3. `<tool_call>` → json (unchanged).
4. `[TOOL_CALLS]` and `[ARGS]` → **mistral_args**.
5. `<|tool_call|>` → json_list (unchanged).

Checked: Qwen3-Coder's template has no `<arg_key>`, and GLM-4.7-Flash's has no `<function=`. Older Mistral templates
write `[TOOL_CALLS][{...}]` with no `[ARGS]`. That form is not in this lot: `select_parser` returns None for it, and
the mode gate refuses agent mode by name, as it does today.

## The two parsers

**glm** (`parsers/glm.py`). Blocks are `<tool_call>(.*?)</tool_call>`. The name is the text before the first
`<arg_key>`, stripped of whitespace. The pairs are `<arg_key>K</arg_key>`, then optional whitespace, then
`<arg_value>V</arg_value>`. A key is stripped; a value is kept exactly as written.

The block is malformed when:
- the name is empty or contains whitespace;
- a key is repeated;
- anything other than whitespace remains between the pairs.

Outside any block, an opening with no close, or a stray `</tool_call>`, `<arg_key>` or `<arg_value>`, is also
malformed, never prose. That is the lesson the xml parser records: a broken call read as an answer ends the task in
the middle.

Values come out as text, as in the xml dialect. `typed_arguments` (routes) already reads a value as JSON where the
tool's schema declares a type other than string. GLM writes non-strings as JSON, so the round trip is exact.

**mistral_args** (`parsers/mistral.py`). The block runs from the first `[TOOL_CALLS]` to the end of the turn. Text
before it is prose. The block is split on `[TOOL_CALLS]`, and each piece is `NAME[ARGS]JSON`. The name is non-empty,
with no whitespace and no `[`. The JSON is a single object, with only whitespace after it.

Any piece that fails makes the whole block malformed, as in json_list. The loop re-asks once, so nothing is
half-taken.

## The envelope (grammar side, shared with the Dell)

- **glm.** Repeat, with no separator: `<tool_call>`, then the name, then for each required parameter in its declared
  order `<arg_key>` K `</arg_key>` `<arg_value>` V `</arg_value>`, then `</tool_call>`.
  - A string value is `until "</arg_value>"`, ended by its token where the tokenizer writes it as one id (the form
    `reasoning_close` already uses).
  - An enum or const value is the bare choice, as in xml.
  - Any other value is the schema's JSON, then the closing marker.
- **mistral_args.** Repeat, with no separator: `[TOOL_CALLS]`, then the name, then `[ARGS]`, then the schema's JSON.
  Only the end of the turn after it.
- **Each marker** is a token step where it is one id (`marker()`), as today.

**One limit of the grammar engine, said by name.** nbxgrammar's choice is decided by a literal prefix, and it refuses
a choice where one literal is a prefix of another (`schema.rs`: "a choice whose literal is a prefix of another's").

The existing dialects end each name with text: `"get_weather", ` in json, `get_weather>` in xml. Here the name is
followed by a marker token, so the choice's literals are the bare names. Tools named `read` and `read_file` would
make the description fail to compile.

The envelope refuses this case before compiling, naming both tools and the dialect:
"`read` is a prefix of `read_file`; the dialect writes a name bare before a marker token". This is never a guess.

The general fix belongs to the grammar engine: a choice whose alternatives may share a prefix, decided by the next
token. It is a proposal to the Dell, not part of this lot. NBX Code's own tool names (`read`, `create`, `replace`,
`insert`, `find`, `search`, `shell`, `plan`) share no prefix.

**What the code added to this section.**
- A GLM enum or const value is also written bare before a marker token (`</arg_value>`), so the same rule refuses two
  values where one is a prefix of the other ("'c' is a prefix of 'celsius'").
- Where a marker is text rather than one id, it joins the literal before it, and the literals differ by their bytes.
- nbxgrammar refuses an empty `seq` and an empty `lit`. So a tool with no required argument whose markers are text
  has its literal end one byte early, and its `then` is that last byte.
- The xml dialect's check on raw string parameters (`_free_text`) is shared with GLM's.

**Found on the way, in the same branch (b7d960e92).** The engine rendered chat templates in a plain jinja2
environment. Its `tojson` HTML-escapes and sorts keys, refuses `ensure_ascii=False` (GLM's template uses it), and runs
unsandboxed. Templates now render in the vendor reference's environment (`core/module/tokenizer/chat_template.py`).
Without that fix, GLM's second agent turn could not be rendered at all.

## Tests (each seen red before green)

- `test_parsers.py`. Per new dialect:
  - the vendor's own template, rendered with jinja2 (already a dependency) on an assistant message with two calls
    (string, integer, list and nested values), parses back to the same calls (round trip through the real template,
    the template text kept as a test fixture with its source and licence line: GLM MIT, Devstral Apache-2.0);
  - prose before the calls;
  - each malformed case above;
  - the selection order (a GLM-like template gets glm, not json).
- `test_a_forced_tool_call_is_written_as_its_parser_reads_it.py`. The canonical call of each new dialect is accepted
  step by step, with its markers as token steps, and read back by its parser. What the envelope forbids is refused:
  - prose first;
  - an undeclared name;
  - another function when one is named;
  - a missing required argument;
  - the prefix pair refused by name.
- `MarkerDecoder` with a fake tokenizer where `[ARGS]` is special: kept when the parser declares it, dropped when it
  does not (the red).
- Live, later: when the Dell has captured GLM-4.7-Flash, a served run of NBX Code with `--calls required`, then the
  ten tasks.

## What this does not do

- The reasoning blocks: GLM's `<think>` is `select_reasoning`'s concern; its detection for GLM is checked in the same
  lot, but not changed unless it fails.
- The older Mistral list form.
- `[CALL_ID]`.
- Rendering the results (the template's own job).
