"""The public repository never says HOW a container is made (RED_LINES line 29): no build-technique verb, no name of the
build tool or of its folder, in any tracked text file. The public wording is "built", "recorded at build", "the
build-time value", "rebuild", "our own container build".

Every tracked text file is read (`git ls-files`; binary files, .nbx caches, the private docs/internal/ and the
census fingerprint files are not). Three stems are matched as SUBSTRINGS, in any letter case, joined forms included:
the verb (trace, tracing: so traced, tracer, retrace, AItracing, retrace_zoo, nbx_tracer all hit), the build tool's
name ("forget" and its family are other English words, admitted by name), and its folder's name. A hit inside a
bare word (trace, traced, tracer, traces, tracing and their re- forms) is admitted only inside a phrase the list
below declares for its file; a hit inside a joined identifier (letters, digits, underscores) only when that
identifier is declared BY NAME, each with its reason. Both lists say which of three kinds a hit is:

  (a) the container FORMAT key: the build-time value of a symbolic dimension that graph.json carries, read and
      written as the format names it;
  (b) the LIBRARY's allocation trace (nbxtensor.memory.trace_replay, the carried traces): a record of one real
      run's allocations, not a container build;
  (c) a trace of a RUN, not of a build: a traceback switch, a profiler or malloc or per-op memory trace, a
      settrace hook, a compile-stage recorder, a walk back through the graph's ops, an English idiom.

No line names the build tool: the workshop role is "factory" and the worktree pointer is gone (no exception).

What this test would do if a file said it again: red, by file:line and the word (injections seen red: a comment
"the graph is traced at build" added to src/neurobrix/core/runtime/symexpr.py; a tracked page naming
tools/retrace_zoo.py, test_a_joined_name_is_red_in_a_tracked_file below).
"""
from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()

STEM = re.compile("trac" + "(e|ing)", re.IGNORECASE)                  # a substring: joined forms hit too
BARE = re.compile(r"(re)?trac(e|ed|er|ers|es|ing)", re.IGNORECASE)      # the bare words, held by a phrase
TOOL_NAME = "f" + "orge"                 # spelled in two halves so that this file does not carry it whole
FOLDER_NAME = "NeuroBrix" + "_System"
NAMES = re.compile(re.escape(TOOL_NAME) + "|" + re.escape(FOLDER_NAME), re.IGNORECASE)
#: English words that hold the tool's name as letters and are other words, admitted by name (the whole letter run)
ENGLISH = ("forget", "forgets", "forgetting", "forgetful")
#: the census fingerprints: the hashed names of the functions a derivation ran, a file format the census reads back
#: byte for byte; renaming a function there unmakes the record. The only path-wide exemption.
FINGERPRINTS = "src/neurobrix/config/census/*/*/*.inputs.jsonl"
_A = "(a) a name of the build-time value the container format carries (graph.json, manifest), not of a technique"
_B = "(b) the library's allocation record (nbxtensor.memory.trace_replay): one real run's allocations"
_C = "(c) a trace of a RUN: a debug switch, a profiler or malloc or launch trace, a settrace hook, a traceback"
_W = "(c) a walk back through the graph's ops"
#: joined identifiers, BY NAME: identifier -> reason
DECLARED = {
    "trace_value": _A + ": the format key", "trace_resolution": _A + ": the manifest key",
    "traced_shape": _A + ": a segment spec key", "trace_seq_len": _A, "trace_size": _A, "off_trace_size": _A,
    "off_trace_in": _A, "off_trace_out": _A, "trace_in": _A, "trace_out": _A, "trace_t_in": _A, "trace_t_out": _A,
    "traced_buffer": _A, "traced_data": _A, "traced_grid_h": _A, "traced_grid_w": _A,
    "k_trace": _A + ": a census row key", "trace_value_is_clear": _A + ": a stimulus row key",
    "traced_island_uids": "(a) the past name the census fingerprints record for that function",
    "trace_replay": _B, "load_trace": _B, "TRACES_DIR": _B,
    "test_the_library_carries_a_trace_that_names_its_card": _B, "test_the_carried_trace_replays_on_a_card": _B,
    "test_several_traces_are_named_never_picked": _B, "test_a_trace_without_its_card_is_refused": _B,
    "test_the_first_launch_probe_measures_on_the_trace": _B,
    "ExpertTrace": "(c) the decode's routing record",
    "test_a_trace_notes_decode_calls_only_and_writes_once": "(c) the decode's routing record",
    "phase_trace": _C + ": the run's phase ledger module", "NBX_PHASE_TRACE": _C, "NBX_MALLOC_TRACE": _C,
    "NBX_LAUNCH_TRACE": _C, "nbx_bank_launch_trace": _C, "NBX_DEVICE_TRACE": _C, "DEVTRACE": _C, "OP_TRACE": _C,
    "VALUE_TRACE": _C, "TID_TRACE": _C, "NBX_TRACE_TIDS": _C, "_trace_tids": _C, "_trace_tids_env": _C,
    "NBX_TRACE_NAN": _C, "NBX_TRITON_TRACE_NAN": _C, "_TRACE_NAN": _C, "trace_nan": _C, "_maybe_trace_nan": _C,
    "_trace_nan_on": _C, "_trace_nan_seen": _C, "NBX_TRACE_ZEROS": _C, "_TRACE_ZEROS": _C, "trace_zeros": _C,
    "LIVE_WATERMARK_TRACE": _C, "NBX_LIVE_WATERMARK_TRACE": _C, "NBX_WEIGHTS_LOAD_TRACE": _C,
    "_StageTracer": _C + ": the kernel compiler's stage recorder", "_tracer": _C, "settrace": _C, "gettrace": _C,
    "untraced": _C + ": a frame the settrace hook leaves alone",
    "tracemalloc": _C, "get_traced_memory": _C, "export_chrome_trace": _C, "trace_path": _C, "trace_": _C,
    "trace_hf_sana_": _C, "traceback": _C, "Traceback": _C, "tracebacks": _C, "__traceback__": _C,
    "with_traceback": _C, "traceback_tail": _C, "stacktrace": _C, "backtrace": _C, "pytrace": _C,
    "_trace_to_mul": _W, "_trace_expert_blocks": _W, "_trace_routing_ops": _W,
    "TRACE_COMPARISON_REPORT": "(c) an ignore rule for an old debug report's file name, never tracked",
}

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
)


def _tracked_texts(root: Path = ROOT):
    if not (root / ".git").exists() or shutil.which("git") is None:
        pytest.skip(f"{root} is not a git checkout (an sdist): no tracked list to read")
    paths = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True,
                           text=True).stdout.split("\0")
    for p in paths:
        if (not p or p == SELF or p.startswith("docs/internal/") or p.endswith(".nbx") or "/.nbx" in p
                or fnmatch.fnmatch(p, FINGERPRINTS)):
            continue
        f = root / p
        if not f.is_file():
            continue
        data = f.read_bytes()
        if b"\0" in data:
            continue                                  # a binary file
        try:
            yield p, data.decode("utf-8")
        except UnicodeDecodeError:
            continue


def _around(line: str, start: int, end: int, keep) -> tuple:
    while start > 0 and keep(line[start - 1]):
        start -= 1
    while end < len(line) and keep(line[end]):
        end += 1
    return start, end


def _ident(c: str) -> bool:
    return c.isalnum() or c == "_"


def hits(path: str, text: str, exceptions=EXCEPTIONS, declared=DECLARED) -> list:
    """(line number, word) of every hit of `text` that no declared phrase of this file, nor a declared identifier, holds."""
    phrases = [ph for p, ph, _ in exceptions if p in ("*", path)]
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        spans = {_around(line, m.start(), m.end(), _ident) for m in STEM.finditer(line)}
        found = []
        for a, z in sorted(spans):
            word = line[a:z]
            if BARE.fullmatch(word):
                found.append((a, z, word))
            elif word not in declared:
                out.append((n, word))
        for m in NAMES.finditer(line):
            a, z = _around(line, m.start(), m.end(), str.isalpha)
            if line[a:z].lower() not in ENGLISH:
                found.append((m.start(), m.end(), m.group(0)))
        if not found:
            continue
        held = [(m.start(), m.end()) for ph in phrases for m in re.finditer(re.escape(ph), line)]
        out += [(n, w) for a, z, w in found if not any(x <= a and z <= y for x, y in held)]
    return out


def test_the_patterns_see_every_form_and_only_those():
    t = "trac"
    seen = f"{t}e. {t}ed, re{t}e {t.upper()}ER {t}ers {t}es {t}ing re{t}ed D-RE{t.upper()}E-X {TOOL_NAME.title()} " \
           f"x{FOLDER_NAME.lower()}/y"
    assert [w.lower() for _, w in hits("p.py", seen, (), {})] == [
        f"{t}e", f"{t}ed", f"re{t}e", f"{t}er", f"{t}ers", f"{t}es", f"{t}ing", f"re{t}ed", f"re{t}e",
        TOOL_NAME, FOLDER_NAME.lower()]
    assert hits("p.py", " ".join(ENGLISH) + " Forgets the_forgetting", (), {}) == []


def test_a_joined_form_hits_unless_declared_by_name():
    """qa 1290-1's mutants: a stem glued to letters, digits or underscores is a hit like a bare word."""
    t = "trac"
    for mutant in (f"AI{t}ing", f"re{t}e_zoo", f"nbx_{t}er", f"tools/re{t}e_zoo.py", f"NBX_{t.upper()}E_X"):
        assert hits("p.md", f"see {mutant} here", (), {}) != [], mutant
    assert hits("p.md", f"see {t}e_value here", (), {f"{t}e_value": "r"}) == []
    assert hits("p.md", f"see {t}e_values here", (), {f"{t}e_value": "r"}) == [(1, f"{t}e_values")]


def test_a_joined_name_is_red_in_a_tracked_file(tmp_path):
    """The scan end to end, on a scratch repository: a tracked page naming a joined build-tool file is named."""
    if shutil.which("git") is None:
        pytest.skip("no git here")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "page.md").write_text("The gate ran `tools/re" + "trace_zoo.py --src <worktree>`.\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "page.md"], check=True)
    found = [(p, n, w) for p, text in _tracked_texts(tmp_path) for n, w in hits(p, text)]
    assert found == [("page.md", 1, "re" + "trace_zoo")]


def test_a_phrase_holds_only_its_own_file_and_its_own_span():
    t = "trac" + "e"
    ex = (("a.py", f"malloc {t}", "r"),)
    assert hits("a.py", f"the malloc {t} of a run", ex) == []
    assert hits("b.py", f"the malloc {t} of a run", ex) == [(1, t)]
    assert hits("a.py", f"the malloc {t}, then the graph was {t}d", ex) == [(1, t + "d")]


def test_every_declared_phrase_and_name_is_still_used():
    texts = dict(_tracked_texts())
    stale = [(p, ph) for p, ph, _ in EXCEPTIONS if p != "*" and ph not in texts.get(p, "")]
    assert stale == [], f"declared exceptions their file no longer holds: remove them: {stale}"
    words = {w for text in texts.values() for w in re.findall(r"[A-Za-z0-9_]+", text)}
    unused = sorted(set(DECLARED) - words)
    assert unused == [], f"declared identifiers no tracked file holds any more: remove them: {unused}"


def test_every_exception_says_why():
    assert all(reason.strip() for _, _, reason in EXCEPTIONS)
    assert all(reason.strip() for reason in DECLARED.values())


def test_the_public_repository_names_no_build_technique():
    found = [f"{p}:{n}: {w}" for p, text in _tracked_texts() for n, w in hits(p, text)]
    assert found == [], "a tracked file says how a container is made (reword it: built, recorded at build, the " \
                        "build-time value, rebuild):\n" + "\n".join(found[:40])
