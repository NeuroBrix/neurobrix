"""`python -m nbxtensor doctor`: which GPU backend this machine runs, and for every NVIDIA, AMD or Intel GPU on its PCI
bus whether the runtime library and the Triton backend it needs are installed — with the vendor's install steps when
one is missing (`backend.install_help`)."""
import sys


def doctor() -> int:
    from nbxtensor.backend import install_help, runtime
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
    if argv[:1] != ["doctor"]:
        print("usage: python -m nbxtensor doctor", file=sys.stderr)
        return 2
    return doctor()


if __name__ == "__main__":
    raise SystemExit(main())
