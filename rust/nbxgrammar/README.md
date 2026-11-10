# nbxgrammar

NeuroBrix's grammar engine for constrained decoding. It compiles a JSON Schema, or a description of a tool-call
format, into a matcher that gives at every decoding step the mask of the tokens a model may write next — so the output
can only be what the schema or the format allows. It is the engine behind `response_format: json_schema` and forced
`tool_choice` in [NeuroBrix](https://github.com/NeuroBrix/neurobrix)'s OpenAI-compatible server.

Written in Rust with the standard library only, loaded from Python with `ctypes` (no binding library). One wheel per
platform serves every Python 3.

```bash
pip install nbxgrammar
```

```python
import json, numpy as np, nbxgrammar as G

grammar = G.Grammar.json_schema(json.dumps({"type": "object", "properties": {"city": {"type": "string"}},
                                            "required": ["city"]}))
vocab = G.Vocab(token_bytes, eos_ids)        # each token id's bytes (None for a special token), the end-of-sequence ids
matcher = G.Matcher(grammar)
mask = matcher.mask(vocab)                    # int32 words, ceil(vocab / 32): bit v % 32 of word v // 32 = token v allowed
matcher.advance(vocab, token_id)              # the token the sampler chose
```

## What it enforces

- **JSON Schema, a subset**: objects (`properties`, `required`; no other property is generated), arrays (`items`,
  `minItems`, `maxItems`), strings, numbers, integers, booleans, null, `enum`, `const`, `anyOf` whose alternatives start
  differently, `$ref` into `$defs`. Any other constraining keyword is refused by name (a `ValueError`), never ignored;
  annotations (`title`, `description`, `default`, `examples`, ...) are read as annotations.
- **Descriptions**: literals, schemas, sequences, choices by a literal prefix, repetitions with a separator, free text up
  to a terminator (`until`, optionally ended by a special token), and exact token ids — what a chat template's tool-call
  format is made of.
- **Whitespace**: at most one space where JSON allows whitespace inside a value, never a newline, and none around a
  JSON root.

Measured on Qwen3's vocabulary (151 936 tokens, Apple M4 Pro): a mask in about 15 µs median inside JSON, 73 µs in free
text.

## Building from source

The crate pins its toolchain (`rust-toolchain.toml`). `pip install .` builds the library with `cargo` and places it in
the package; `cargo test --release` runs the oracle tests (every mask against a brute force over the vocabulary, every
completed output against an independent validator).

## License

Apache License 2.0 ([LICENSE](LICENSE)); see [NOTICE](NOTICE). Contributions are made under the
[NeuroBrix Contributor Assignment Agreement](https://github.com/NeuroBrix/neurobrix/blob/main/docs/legal/contributor-assignment-agreement.md)
([CONTRIBUTING.md](CONTRIBUTING.md)).

Project: [neurobrix.org](https://neurobrix.org) · documentation: [docs.neurobrix.org](https://docs.neurobrix.org) ·
contact@neurobrix.org
