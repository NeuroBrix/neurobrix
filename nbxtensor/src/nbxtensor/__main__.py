"""`python -m nbxtensor doctor`: which GPU backend this machine runs, and for every NVIDIA, AMD or Intel GPU on its PCI
bus whether the runtime library and the Triton backend it needs are installed — with the vendor's install steps when
one is missing (`backend.install_help`); on an Apple-silicon Mac, whether Triton for Metal is installed.
`python -m nbxtensor install-metal [--dry-run]`: install Triton for the Apple GPU from this version's release
(`backend.install_metal`)."""
import sys


def doctor() -> int:
    from nbxtensor.backend import install_help, runtime
    from nbxtensor.backend.metal_backend import is_apple_silicon, triton_metal_available
    if is_apple_silicon() and not triton_metal_available():
        print("Apple GPU found; Triton for Metal not installed — run: python -m nbxtensor install-metal")
        return 1                     # nothing can run yet: a script reads it from the exit status
    try:
        backend = runtime.gpu_backend()
        print(f"backend in force: {backend}" + (" (the Apple GPU, through Metal)" if backend == "metal" else ""))
    except RuntimeError as exc:
        backend = None
        print(f"backend in force: none — {str(exc).splitlines()[0]}")
    if backend != "metal" or install_help.pci_gpus():     # a Mac has no PCI GPU to report: nothing to install there
        print(install_help.report())
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["doctor"]:
        return doctor()
    if argv[:1] == ["install-metal"] and set(argv[1:]) <= {"--dry-run"}:
        import nbxtensor
        from nbxtensor.backend import install_metal
        return install_metal.main(nbxtensor.__version__, dry_run="--dry-run" in argv[1:])
    print("usage: python -m nbxtensor doctor | install-metal [--dry-run]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
