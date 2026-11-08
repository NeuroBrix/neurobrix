# Building applications with NeuroBrix (a guide for coding agents)

This page is written for coding agents (Claude Code and others) and for the developers who drive them. It describes the
surface an application should use to put NeuroBrix models inside a product, and the rules that keep that integration small.

## The surface

Today the application surface of NeuroBrix is its **command line**: one command per task, the same for every model family,
the result written to a file. An application calls it as a subprocess and reads the file. There is no per-model pipeline to
write, no PyTorch to install, and no CUDA or framework version to match.

A Python API and a local server with a JSON interface are on the roadmap. Until they exist, use the commands below; they are
the stable contract.

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
neurobrix hub --json                                # models available on the hub (neurobrix.es/models)
neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0 # download + unpack; the hub identifier, as listed by `hub`
neurobrix list --json                               # installed models: name, family, size, license
neurobrix remove TinyLlama-1.1B-Chat-v1.0           # free the disk
```

Two names exist for a model: the **hub identifier** you pass to `import` (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) and the
**model name** every other command takes (`TinyLlama-1.1B-Chat-v1.0`, the `name` field of `list --json`). Some models ask
for their license to be accepted: add `--accept-license` to `import` once the user has agreed to it.

## One command per task

The family is read from the model, so `run` takes the inputs that family needs and writes the result to `--output`:

| task | command | output |
|---|---|---|
| text generation, chat, code | `neurobrix run --model TinyLlama-1.1B-Chat-v1.0 --prompt "..." --max-tokens 256 --output answer.txt` | text file |
| speech-to-text | `neurobrix run --model whisper-large-v3-turbo --audio meeting.wav --output transcript.txt` | text file |
| speech understanding | `neurobrix run --model Voxtral-Mini-3B-2507 --audio meeting.wav --prompt "Summarize this call." --output answer.txt` | text file |
| text-to-speech | `neurobrix run --model Kokoro-82M --prompt "Hello." --output speech.wav` | WAV |
| vision-language | `neurobrix run --model GLM-4.1V-9B-Thinking --input-image photo.jpg --prompt "What is in this image?" --output answer.txt` | text file |
| image generation | `neurobrix run --model Flex.1-alpha --prompt "a lighthouse at dusk" --steps 20 --seed 42 --output image.png` | PNG |
| video generation | `neurobrix run --model CogVideoX-2b --prompt "..." --num-frames 17 --seed 42 --output clip.mp4` | MP4 |
| upscaling | `neurobrix upscale --model Real-ESRGAN-RealESRGAN_x4plus --input photo.png --output photo_x4.png` | PNG |

`neurobrix run --help` lists every option (`--seed`, `--temperature`, `--steps`, `--height`, `--width`, `--system`, ...).
The exit code is 0 on success and non-zero on failure, with the reason on the terminal.

## Keeping a model warm

Each `run` loads the model, runs it and exits. For an application that calls the same model many times, `serve` keeps it
loaded in a background daemon:

```bash
neurobrix serve --model TinyLlama-1.1B-Chat-v1.0     # returns once the model is loaded; exit code 1 on failure
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
- **The conversation lives in the daemon.** `chat` keeps one shared history per daemon and re-sends all of it to the
  model at every turn; `/new` (the `new_chat` method) clears it. For stateless calls use `run`.
- **The idle timeout** (`--timeout`, 1800 s by default) stops a daemon that has no open connection.
- **Programs can talk to the socket directly** (length-prefixed JSON, with operation ids, streaming and cancellation):
  [the daemon wire protocol](reference/wire-protocol.md).

## Rules for agents

1. **Call the CLI; do not reimplement a model.** No PyTorch, no `transformers`, no per-model pipeline code in the
   application: one `neurobrix run` per task.
2. **Run `neurobrix doctor --json` first** and stop on `"ok": false`.
3. **Use `list --json` and `hub --json` to discover models**; never hard-code a size or a family the engine can report.
4. **Read results from `--output`.** Use a temporary directory per call, and check the exit code.
5. **The first `import` downloads gigabytes.** Tell the user before calling it, and do it once (setup), not per request.
6. **Same code on Linux, Windows and macOS.** The commands are the same everywhere; only `install-metal` is macOS-specific.

## Example: dictate (voice recording → transcript → summary)

A complete application, written by Claude Code from an empty folder with only this repository and PyPI as sources, and run on
an Apple M4 Pro with NeuroBrix 0.6.2 installed from PyPI:

```python
#!/usr/bin/env python3
"""dictate — voice recording -> transcript -> short summary, with NeuroBrix as the inference engine.

    python dictate.py meeting.wav            # writes meeting.md
"""
import subprocess, sys, tempfile
from pathlib import Path

STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "TinyLlama-1.1B-Chat-v1.0"


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
