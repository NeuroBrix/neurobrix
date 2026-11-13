# Building applications with NeuroBrix (a guide for coding agents)

This page is written for coding agents (Claude Code and others) and for the developers who drive them. It describes the
surface an application should use to put NeuroBrix models inside a product, and the rules that keep that integration small.

## The surface

The application surface of NeuroBrix is its **command line**: one command per task, the same for every model family, the
result written to a file. An application calls it as a subprocess and reads the file. There is no per-model pipeline to
write, no PyTorch to install, and no CUDA or framework version to match.

Since 0.6.3, a model kept loaded by `neurobrix serve --http` also answers the **OpenAI API** over HTTP (below), so an
application already written against an OpenAI client can point it at the local daemon. An in-process Python API is on the
roadmap; until it exists, use the commands and the HTTP API below; they are the stable contract.

## Setup (once per machine)

```bash
pip install neurobrix
python -m nbxtensor install-metal      # macOS on Apple Silicon only: NeuroBrix's Triton build for Apple GPUs
neurobrix doctor --json                # {"ok": true, "problems": []} when the engine can run
```

`neurobrix doctor` names every missing piece with the command that installs it. An agent should run it before anything else
and stop on `"ok": false`, showing the user the problems it lists.

## Finding and installing models

```bash
neurobrix hub --json                                # models available on the hub (neurobrix.org/models)
neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0 # download + unpack; the hub identifier, as listed by `hub`
neurobrix list --json                               # installed models: name, family, size, license
neurobrix remove TinyLlama/TinyLlama-1.1B-Chat-v1.0 # free the disk
```

A model has **one name**, its hub name (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`): `import` records it beside the installed
model, and `--model`, `remove`, `list` and the HTTP API's `model` field all take it. `list --json` (schema 2) gives it as
`name`, and the installed directory as `entry`. The short directory name is refused for a model whose name is recorded,
naming the hub name to use. A model installed before 0.6.3 has no record yet: run `neurobrix import <org>/<name>` once for it
(it downloads nothing). Some models ask for their license to be accepted: add `--accept-license` to `import` once the user
has agreed to it.

## One command per task

The family is read from the model, so `run` takes the inputs that family needs and writes the result to `--output`:

| task | command | output |
|---|---|---|
| text generation, chat, code | `neurobrix run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "..." --max-tokens 256 --output answer.txt` | text file |
| speech-to-text | `neurobrix run --model openai/whisper-large-v3-turbo --audio meeting.wav --output transcript.txt` | text file |
| speech understanding | `neurobrix run --model mistralai/Voxtral-Mini-3B-2507 --audio meeting.wav --prompt "Summarize this call." --output answer.txt` | text file |
| text-to-speech | `neurobrix run --model hexgrad/Kokoro-82M --prompt "Hello." --speaker af_heart --output speech.wav` | WAV |
| vision-language | `neurobrix run --model zai-org/GLM-4.1V-9B-Thinking --input-image photo.jpg --prompt "What is in this image?" --output answer.txt` | text file |
| image generation | `neurobrix run --model ostris/Flex.1-alpha --prompt "a lighthouse at dusk" --steps 20 --seed 42 --output image.png` | PNG |
| video generation | `neurobrix run --model zai-org/CogVideoX-2b --prompt "..." --num-frames 17 --seed 42 --output clip.mp4` | MP4 |
| upscaling | `neurobrix upscale --model xinntao/Real-ESRGAN-RealESRGAN_x4plus --input photo.png --output photo_x4.png` | PNG |

`neurobrix run --help` lists every option (`--seed`, `--temperature`, `--steps`, `--height`, `--width`, `--system`, ...);
the [CLI reference](reference/cli.md) is generated from the same parser.
The exit code is 0 on success and non-zero on failure, with the reason on the terminal.

## Keeping a model warm

Each `run` loads the model, runs it and exits. For an application that calls the same model many times, `serve` keeps it
loaded in a background daemon:

```bash
neurobrix serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0     # returns once the model is loaded; exit code 1 on failure
neurobrix run --prompt "..."                        # answered by the daemon: --model may be omitted
neurobrix chat                                      # interactive multi-turn chat (llm family)
neurobrix stop                                      # unload it and free the memory
```

What an agent driving a daemon should know:

- **Check the start line.** `serve` prints `(<family>, warm serving)` or `(<family>, cold serving)`. Warm: the weights
  stay on the device between requests. Cold: the plan Prism chose for this machine loads them per request, so the
  daemon saves the setup but not the loading. See [Serving and chat](serving.md#warm-and-cold-serving).
- **One daemon, one model.** While it runs, a `run` of another model is refused; `neurobrix stop` first, or give each
  daemon its own instance with `NBX_SOCKET_PATH=<path>` (the same value for `serve` and its clients).
- **Text answers from the daemon are printed, not written to `--output`.** On the daemon path, the daemon writes image,
  video and audio results to `--output`, and `run` prints text on stdout; read stdout there.
- **One inference at a time.** A request that arrives while another runs waits for it.
- **The conversation lives in the daemon, and so does its KV cache.** `chat` keeps one shared history per daemon; each
  turn prefills only the tokens the cache does not already hold. A turn that would pass the session's ceiling (the plan's
  `session_reach`, shown by `run --explain-plan`) is refused by name; nothing is summarised or dropped. `/new` (the
  `new_chat` method) clears it. For stateless calls use `run`; to test a multi-turn exchange without a daemon,
  `neurobrix run --turns turns.json` plays a JSON list of user messages as one session.
- **The idle timeout** (`--timeout`, 1800 s by default) stops a daemon that has no open connection.
- **Programs can talk to the socket directly** (length-prefixed JSON, with operation ids, streaming and cancellation):
  [the daemon wire protocol](reference/wire-protocol.md).

## The OpenAI-compatible API

```bash
neurobrix serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --http     # http://127.0.0.1:8765/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
reply = client.chat.completions.create(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                       messages=[{"role": "user", "content": "What is a KV cache?"}])
print(reply.choices[0].message.content)
```

- **Same daemon, same queue.** The HTTP API serves the model the daemon loaded, on the same inference thread as `chat` and
  `run`. The `model` field must be that model's one name (`GET /v1/models` gives it); another name answers 404.
- **Endpoints follow the family**: `/v1/chat/completions` and `/v1/completions` for language models (tool calls parsed
  from the model's chat template, a thinking model's reasoning in `reasoning_content`, streaming as Server-Sent Events),
  `/v1/images/generations`, `/v1/audio/speech`, `/v1/audio/transcriptions`. An endpoint the loaded family does not serve
  answers 409 naming why.
- **Unsupported parameters are refused, not ignored**: `stop`, `logprobs`, `n` > 1, a forced `tool_choice`, ... answer 400
  naming the parameter. Too many waiting requests answer 429. Handle both; do not retry a 400 unchanged.
- **Loopback by default.** Another address needs `--api-key-file`; requests then send `Authorization: Bearer <key>`.

Every endpoint and setting: [Serving and chat](serving.md#openai-compatible-http-api).

## Rules for agents

1. **Call the CLI; do not reimplement a model.** No PyTorch, no `transformers`, no per-model pipeline code in the
   application: one `neurobrix run` per task.
2. **Run `neurobrix doctor --json` first** and stop on `"ok": false`.
3. **Use `list --json` and `hub --json` to discover models**; never hard-code a size or a family the engine can report.
   Name a model by its hub name (`<org>/<name>`) everywhere.
4. **Read results from `--output`.** Use a temporary directory per call, and check the exit code.
5. **The first `import` downloads gigabytes.** Tell the user before calling it, and do it once (setup), not per request.
6. **Same code on Linux, Windows and macOS.** The commands are the same everywhere; only `install-metal` is macOS-specific.

## Example: dictate (voice recording → transcript → summary)

A complete application, written by Claude Code from an empty folder with only this repository and PyPI as sources, and run on
an Apple M4 Pro with NeuroBrix 0.6.2 installed from PyPI (its model names written here as 0.6.3's one name, the hub name):

```python
#!/usr/bin/env python3
"""dictate — voice recording -> transcript -> short summary, with NeuroBrix as the inference engine.

    python dictate.py meeting.wav            # writes meeting.md
"""
import subprocess, sys, tempfile
from pathlib import Path

STT_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def neurobrix(*args: str) -> str:
    """Run one `neurobrix run` and return the text it wrote to --output."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        subprocess.run(["neurobrix", "run", *args, "--output", str(out)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out.read_text().strip()


def main(audio: Path) -> Path:
    transcript = neurobrix("--model", STT_MODEL, "--audio", str(audio))
    summary = neurobrix("--model", LLM_MODEL, "--max-tokens", "96",
                        "--prompt", f"Summarize this in two sentences:\n\n{transcript}")
    note = audio.with_suffix(".md")
    note.write_text(f"# {audio.stem}\n\n## Summary\n\n{summary}\n\n## Transcript\n\n{transcript}\n")
    return note


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
```

Setup for it:

```bash
pip install neurobrix
python -m nbxtensor install-metal            # macOS only
neurobrix import openai/whisper-large-v3-turbo --accept-license
neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0 --accept-license
```

Two models from two different families, one helper function, no framework code.
