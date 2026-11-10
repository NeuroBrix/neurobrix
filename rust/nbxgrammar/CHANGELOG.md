# Changelog

All notable changes to nbxgrammar are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - unreleased

### Added
- **The grammar engine**: a JSON Schema subset and grammar descriptions (literals, schemas, sequences, choices,
  repetitions, free text up to a terminator or a special token, token ids) compiled into a matcher; per-step token
  masks over a tokenizer's byte vocabulary, string-safe and free-text tokens cached once per vocabulary; any constraint
  it cannot enforce refused by name.
- **The Python package**: `Grammar.json_schema`, `Grammar.description`, `Vocab`, `Matcher.mask / advance / accepting`,
  through the library's C ABI with `ctypes`; one wheel per platform (`py3-none-<platform>`) for Linux x86_64 and
  aarch64, macOS arm64 and x86_64, Windows x86_64.
