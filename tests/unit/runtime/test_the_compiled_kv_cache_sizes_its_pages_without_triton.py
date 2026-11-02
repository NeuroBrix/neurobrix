"""The compiled (ATen) branch sizes its paged KV cache without importing triton: the page tiles are a size, read from
nbxtensor.ops.tiling_sizes (torch- and triton-free), not from the Triton kernel module. Seen on 2026-11-02: the
compiled TinyLlama failed in an environment with torch and no triton (`core/module/cache/factory.paged_params`
imported `nbxtensor.ops.paged_decode_attn`, whose first line is `import triton`) — the benchmark's torch side runs
there."""
import subprocess
import sys
import textwrap


def test_the_compiled_paged_cache_params_import_no_triton():
    script = textwrap.dedent("""
        import sys
        class _NoTriton:
            def find_spec(self, name, path=None, target=None):
                if name == "triton" or name.startswith("triton."):
                    raise ImportError("triton is not installed in this environment")
                return None
        sys.meta_path.insert(0, _NoTriton())
        from types import SimpleNamespace
        from neurobrix.core.module.cache.factory import paged_params
        tokens, async_release = paged_params(4, 64, 64, "float32", ("cpu", "arm64"),
                                             SimpleNamespace(memory_bytes=64 << 20))
        assert tokens > 0 and async_release is False
        assert not any(m == "triton" or m.startswith("triton.") for m in sys.modules), "triton was imported"
        print("ok")
    """)
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), p.stderr[-1500:]


def test_the_kernel_module_reads_the_same_tiles():
    from nbxtensor.ops import tiling_sizes
    from nbxtensor.ops.paged_decode_attn import PAGED_DECODE_JUDGED_TILE, PAGED_DECODE_TILE_CANDIDATES
    assert PAGED_DECODE_TILE_CANDIDATES is tiling_sizes.PAGED_DECODE_TILE_CANDIDATES
    assert PAGED_DECODE_JUDGED_TILE == tiling_sizes.PAGED_DECODE_JUDGED_TILE
