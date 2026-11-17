"""Every Python source of the engine compiles with warnings as errors (the library's sources are its own repository's)."""
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREES = (ROOT / "src" / "neurobrix",)


def test_every_engine_source_compiles_without_a_warning():
    failures = []
    count = 0
    for tree in TREES:
        for path in sorted(tree.rglob("*.py")):
            count += 1
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    compile(path.read_bytes(), str(path), "exec")
                except (SyntaxError, Warning) as exc:
                    failures.append(f"{path.relative_to(ROOT)}: {exc}")
    assert count > 0, "no source found under src/neurobrix"
    assert not failures, "\n".join(failures)
