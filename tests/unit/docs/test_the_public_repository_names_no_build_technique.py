"""The public repository never says HOW a container is made (RED_LINES line 29): no build-technique verb, no name of the
build tool or of its folder, in any tracked text file. The public wording is "built", "recorded at build", "the
build-time value", "rebuild", "our own container build".

Every tracked text file is read (`git ls-files`; binary files, .nbx caches and the private docs/internal/ are not).
Three stems are matched in any letter case: the verb (the regex below: trace, traced, tracer, tracers, traces,
tracing, and their re- forms, as whole words), the build tool's name (as a substring; "forget" and its family are
other English words, admitted by name), and its folder's name (as a substring). A hit is admitted only inside a
phrase the list below declares for its file, each with its reason, in three kinds:

  (a) the container FORMAT key: the build-time value of a symbolic dimension that graph.json carries, read and
      written as the format names it;
  (b) the LIBRARY's allocation trace (nbxtensor.memory.trace_replay, the carried traces): a record of one real
      run's allocations, not a container build;
  (c) a trace of a RUN, not of a build: a traceback switch, a profiler or malloc or per-op memory trace, a
      settrace hook, a compile-stage recorder, a walk back through the graph's ops, an English idiom.

Two lines are admitted although they name the build tool (kind (d)): they are names on the machines, outside this
repository (a workshop configuration role, a gitignored pointer directory of the worktrees), and renaming them is a
change of every machine's configuration, not a text edit. They are listed by name so that they stay in sight.

What this test would do if a file said it again: red, by file:line and the word (injection seen red: a comment
"the graph is traced at build" added to src/neurobrix/core/runtime/symexpr.py -> named here).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()

VERB = re.compile(r"\b(re)?trac(e|ed|er|ers|es|ing)\b", re.IGNORECASE)
TOOL_NAME = "f" + "orge"                 # spelled in two halves so that this file does not carry it whole
FOLDER_NAME = "NeuroBrix" + "_System"
NAMES = re.compile(re.escape(TOOL_NAME) + "|" + re.escape(FOLDER_NAME), re.IGNORECASE)
#: English words that hold the tool's name as letters and are other words, admitted by name (the whole letter run)
ENGLISH = ("forget", "forgets", "forgetting", "forgetful")
#: words that hold the verb's letters and are not it; the regex does not match them (whole words only) — named so
#: that the next reader knows they were judged, and held by test_the_patterns_see_every_form_and_only_those
NOT_THE_VERB = ("traceback", "Traceback", "settrace", "tracemalloc", "pytrace", "trace_replay", "untraced")

#: (path or "*", phrase that must hold the hit on its line, reason)
EXCEPTIONS = (
    # (a) the container format key
    ("*", '"trace"', "(a) the container format key: the build-time value of a symbolic dimension in graph.json"),
    ("*", "'trace'", "(a) the container format key, read in an f-string"),
    ("src/neurobrix/kernels/classification.py", '"trace"', "(a)-like: aten's own op name, the matrix trace"),
    ("tools/streamed_component_vs_whole.py", '("build", "trace")',
     "(a)-like: the build mode's old spelling, a hidden CLI alias kept until nbx_version 0.2"),
    ("tools/symbol_collision_census.py", '"trace": built',
     "(a) the old key written beside \"built\" until nbx_version 0.2"),
    ("docs/reference/vacuous-gates-register.md", "node's `trace`", "(a) the format key, quoted as the format names it"),
    ("tools/public_vocabulary.allow.tsv", "`trace`", "(a) the allow-list's own entry for the quoted format key"),
    # (b) the library's allocation trace
    ("src/neurobrix/pytorch/prism/facts_probe.py", "recorded allocation trace", "(b) nbxtensor.memory.trace_replay"),
    ("src/neurobrix/triton/prism/facts_probe.py", "recorded allocation trace", "(b) nbxtensor.memory.trace_replay"),
    ("tests/unit/kernels/test_the_chunk_is_measured_on_a_real_trace.py", "trace",
     "(b) the library's carried allocation trace, replayed by the chunk probe"),
    ("tests/unit/kernels/test_the_chunk_is_measured_on_a_real_trace.py", "traces", "(b) the same, its list"),
    # (c) a trace of a run
    ("src/neurobrix/cli/__init__.py", "trace=False", "(c) the traceback switch of _fail"),
    ("src/neurobrix/cli/__init__.py", "trace: bool", "(c) the traceback switch of _fail"),
    ("src/neurobrix/cli/__init__.py", "elif trace and", "(c) the traceback switch of _fail"),
    ("src/neurobrix/config/vendors/nvidia/ampere.yml", "# trace of Open-Sora-v2's VAE", "(c) a per-op memory trace of a run"),
    ("src/neurobrix/config/vendors/nvidia/hopper.yml", "# trace of Open-Sora-v2's VAE", "(c) a per-op memory trace of a run"),
    ("src/neurobrix/config/vendors/nvidia/volta.yml", "# trace of Open-Sora-v2's VAE", "(c) a per-op memory trace of a run"),
    ("src/neurobrix/core/optim/policy.py", "traces back", "(c) English: a finding's provenance"),
    ("src/neurobrix/core/runtime/graph/compiled_sequence.py", "NaN/range trace", "(c) the NaN debug trace of a run"),
    ("src/neurobrix/core/runtime/graph/compiled_sequence.py", "NaN/Inf TRACING", "(c) the NaN debug trace of a run"),
    ("src/neurobrix/core/runtime/graph/compiled_sequence.py", "ZERO-TRACE", "(c) the zero-output debug trace of a run"),
    ("src/neurobrix/core/runtime/graph/compiled_sequence.py", "trace back through view-like ops",
     "(c) a walk back through the graph's ops"),
    ("src/neurobrix/core/runtime/graph/moe_fusion.py", "trace back from the index ops", "(c) a walk back through the ops"),
    ("src/neurobrix/core/runtime/graph_executor.py", "Trace back through passthrough ops", "(c) a walk back through the ops"),
    ("src/neurobrix/core/runtime/graph_executor.py", "malloc trace", "(c) a malloc trace of a run"),
    ("src/neurobrix/triton/promotion.py", "backward-traced", "(c) a walk back through the ops"),
    ("src/neurobrix/triton/promotion.py", "backward-tracing", "(c) a walk back through the ops"),
    ("src/neurobrix/triton/promotion.py", "Backward-trace", "(c) a walk back through the ops"),
    ("src/neurobrix/core/runtime/graph/expert_store.py", "self.trace", "(c) the decode's routing record (ExpertTrace)"),
    ("src/neurobrix/triton/expert_store.py", "self.trace", "(c) the decode's routing record (ExpertTrace)"),
    ("src/neurobrix/triton/expert_store.py", "(`trace`)", "(c) the decode's routing record (ExpertTrace)"),
    ("src/neurobrix/core/strategies/expert_store.py", "store.trace", "(c) the decode's routing record (ExpertTrace)"),
    ("src/neurobrix/core/strategies/expert_store.py", "the trace reads the routing back", "(c) the routing record"),
    ("src/neurobrix/core/runtime/routing_reuse.py", "traces", "(c) the decode calls' routing records"),
    ("src/neurobrix/core/runtime/routing_reuse.py", "traced layers", "(c) the layers the routing record holds"),
    ("tests/unit/strategies/test_the_card_residency_is_armed_where_a_kernel_reads_data_ptr.py", "trace=None",
     "(c) the routing record"),
    ("tests/unit/strategies/test_the_card_residency_is_armed_where_a_kernel_reads_data_ptr.py", "trace=trace",
     "(c) the routing record"),
    ("tests/unit/strategies/test_the_card_residency_is_armed_where_a_kernel_reads_data_ptr.py", "trace=object()",
     "(c) the routing record"),
    ("tests/unit/triton/test_the_engine_expert_store_keeps_only_decisions.py", "s.trace", "(c) the routing record"),
    ("src/neurobrix/core/strategies/layer_streaming.py", "per-second trace", "(c) a memory trace of a run"),
    ("src/neurobrix/decode_progress.py", "phase trace", "(c) the run's phase trace"),
    ("src/neurobrix/pytorch/prism/runtime_widths.py", "allocator trace", "(c) an allocator trace of a run"),
    ("src/neurobrix/triton/prism/runtime_widths.py", "allocator trace", "(c) an allocator trace of a run"),
    ("tests/unit/core/test_loading_paths_refuse_instead_of_skipping.py", "no trace.", "(c) English idiom"),
    ("tests/unit/kernels/test_f32_to_bf16_bits_streams.py", "peak traced", "(c) tracemalloc's peak"),
    ("tests/unit/kernels/test_index_add_deterministic.py", "traces to Triton codegen", "(c) English: is attributed to"),
    ("tests/unit/kernels/test_the_certifier_prices_a_key_by_its_phases.py", "malloc/phase traces", "(c) a run's traces"),
    ("tests/unit/kernels/test_the_certifier_prices_a_key_by_its_phases.py", "trace showed", "(c) a run's trace"),
    ("tests/unit/kernels/test_the_conv_bias_rides_in_place.py", "malloc trace", "(c) a malloc trace of a run"),
    ("tests/unit/module/test_added_tokens_are_split_in_linear_time.py", "tracer", "(c) a sys.settrace hook"),
    ("tests/unit/prism/test_lifecycle_budget_is_executed.py", "no trace of it", "(c) English idiom"),
    ("tests/unit/prism/test_the_kv_is_priced_once_and_no_buffer_is_added_to_a_walked_live_set.py", "per-op V100 trace",
     "(c) a per-op memory trace of a run"),
    ("tests/unit/runtime/test_a_component_says_when_it_is_released.py", "per-second memory trace",
     "(c) a memory trace of a run"),
    ("tests/unit/tools/test_a_waiting_cell_says_what_it_waits_for.py", "without a trace", "(c) English idiom"),
    ("tools/address_origin.py", "trace (`NBX_MALLOC_TRACE", "(c) the malloc trace a run writes"),
    ("tools/address_origin.py", "end of the trace", "(c) the malloc trace a run writes"),
    ("tools/address_origin.py", "<trace.tsv>", "(c) the malloc trace a run writes"),
    ("tools/census_inputs.py", "trace function", "(c) a sys.settrace hook"),
    ("tools/check_measurement_environment.py", "traced not to the code", "(c) English: attributed"),
    ("tools/gate_scope_probe.py", "tracer", "(c) a sys.settrace hook"),
    ("tools/intel_spirv_census.py", "tracer", "(c) the kernel compiler's stage recorder"),
    ("tools/intel_spirv_census.py", "traced", "(c) the kernel compiler's stage recorder"),
    ("tools/metal_compile_census.py", "tracer", "(c) the kernel compiler's stage recorder"),
    ("tools/metal_compile_census.py", "traced", "(c) the kernel compiler's stage recorder"),
    ("tools/metal_msl_path_equivalence.py", "tracing the stage", "(c) the kernel compiler's stage recorder"),
    ("tools/metal_msl_path_equivalence.py", "tracer captures nothing", "(c) the kernel compiler's stage recorder"),
    ("tools/kernel_reference_bank.py", "TRACE", "(c) the launcher's launch trace"),
    ("tools/llama_fp64_oracle.py", "trace", "(c) the oracle's residual-norm record of a forward"),
    ("tools/precision_zoo_campaign.py", "left no trace", "(c) English idiom"),
    ("tools/regression_matrix.py", "with no trace", "(c) English idiom"),
    ("tools/render_resume_stepdiff.py", "traced", "(c) a wrapper that records a run's checkpoint saves"),
    ("benchmarks/profile_dtype_overhead.py", "CUDA trace", "(c) a CUDA profiler trace"),
    ("benchmarks/profile_dtype_overhead.py", "profiler trace", "(c) a CUDA profiler trace"),
    ("benchmarks/profile_dtype_overhead.py", "benchmarks/traces", "(c) the profiler traces' directory"),
    ("benchmarks/profile_dtype_overhead.py", "Export trace", "(c) a CUDA profiler trace"),
    ("benchmarks/profile_dtype_overhead.py", "Trace saved", "(c) a CUDA profiler trace"),
    ("benchmarks/profile_hf_baseline.py", "CUDA trace", "(c) a CUDA profiler trace"),
    ("benchmarks/profile_hf_baseline.py", "benchmarks/traces", "(c) the profiler traces' directory"),
    ("benchmarks/profile_hf_baseline.py", "Trace saved", "(c) a CUDA profiler trace"),
    # (d) names on the machines, outside this repository
    ("tools/certify_the_catalogue.py", f'interpreter("{TOOL_NAME}")',
     "(d) a role of each machine's private atelier.json `python` table; renaming it is a configuration change"),
    ("tools/precision_zoo_campaign.py", f'FROZEN_POINTERS = (".nbx_registry", "{TOOL_NAME}")',
     "(d) a gitignored pointer directory every frozen worktree carries; renaming it is a workshop change"),
)


def _tracked_texts():
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip(f"{ROOT} is not a git checkout (an sdist): no tracked list to read")
    paths = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True,
                           text=True).stdout.split("\0")
    for p in paths:
        if not p or p == SELF or p.startswith("docs/internal/") or p.endswith(".nbx") or "/.nbx" in p:
            continue
        f = ROOT / p
        if not f.is_file():
            continue
        data = f.read_bytes()
        if b"\0" in data:
            continue                                  # a binary file
        try:
            yield p, data.decode("utf-8")
        except UnicodeDecodeError:
            continue


def _letters_around(line: str, start: int, end: int) -> str:
    while start > 0 and line[start - 1].isalpha():
        start -= 1
    while end < len(line) and line[end].isalpha():
        end += 1
    return line[start:end]


def hits(path: str, text: str, exceptions=EXCEPTIONS) -> list:
    """(line number, word) of every hit of `text` that no declared phrase of this file holds."""
    phrases = [ph for p, ph, _ in exceptions if p in ("*", path)]
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        found = list(VERB.finditer(line)) + list(NAMES.finditer(line))
        if not found:
            continue
        held = [(m.start(), m.end()) for ph in phrases for m in re.finditer(re.escape(ph), line)]
        for m in found:
            if m.group(0).lower() == TOOL_NAME and _letters_around(line, m.start(), m.end()).lower() in ENGLISH:
                continue
            if not any(a <= m.start() and m.end() <= b for a, b in held):
                out.append((n, m.group(0)))
    return out


def test_the_patterns_see_every_form_and_only_those():
    t = "trac"
    seen = f"{t}e. {t}ed, re{t}e {t.upper()}ER {t}ers {t}es {t}ing re{t}ed D-RE{t.upper()}E-X {TOOL_NAME.title()} " \
           f"x{FOLDER_NAME.lower()}/y"
    assert [w.lower() for _, w in hits("p.py", seen, ())] == [
        f"{t}e", f"{t}ed", f"re{t}e", f"{t}er", f"{t}ers", f"{t}es", f"{t}ing", f"re{t}ed", f"re{t}e",
        TOOL_NAME, FOLDER_NAME.lower()]
    other = " ".join(NOT_THE_VERB + ENGLISH) + " Forgets the_forgetting"
    assert hits("p.py", other, ()) == []


def test_a_phrase_holds_only_its_own_file_and_its_own_span():
    t = "trac" + "e"
    ex = (("a.py", f"malloc {t}", "r"),)
    assert hits("a.py", f"the malloc {t} of a run", ex) == []
    assert hits("b.py", f"the malloc {t} of a run", ex) == [(1, t)]
    assert hits("a.py", f"the malloc {t}, then the graph was {t}d", ex) == [(1, t + "d")]


def test_every_declared_phrase_is_still_on_its_file():
    texts = dict(_tracked_texts())
    stale = [(p, ph) for p, ph, _ in EXCEPTIONS if p != "*" and ph not in texts.get(p, "")]
    assert stale == [], f"declared exceptions their file no longer holds: remove them: {stale}"


def test_every_exception_says_why():
    assert all(reason.strip() for _, _, reason in EXCEPTIONS)


def test_the_public_repository_names_no_build_technique():
    found = [f"{p}:{n}: {w}" for p, text in _tracked_texts() for n, w in hits(p, text)]
    assert found == [], "a tracked file says how a container is made (reword it: built, recorded at build, the " \
                        "build-time value, rebuild):\n" + "\n".join(found[:40])
