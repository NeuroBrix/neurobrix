"""Helper of the two-card device tests: runs `python -m neurobrix <argv>` in THIS process, then prints, on one line,
the cards this process holds a context on (`CARDS <pci bus ids>`), read from the driver's tool. Run as a script."""
import os
import runpy
import subprocess
import sys


def cards_of_this_process():
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=30).stdout.splitlines()
    return sorted(r.split(",")[0].strip() for r in rows if r.strip().endswith(", " + str(os.getpid())))


if __name__ == "__main__":
    sys.argv = ["neurobrix"] + sys.argv[1:]
    try:
        runpy.run_module("neurobrix", run_name="__main__")
    except SystemExit as stop:
        if stop.code not in (0, None):
            raise
    finally:
        print("CARDS " + " ".join(cards_of_this_process()), flush=True)
