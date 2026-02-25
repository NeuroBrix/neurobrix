"""
neurobrix serve — Start persistent model serving daemon.

Loads model weights into VRAM once, keeps them warm.
Runs in background by default — terminal returns immediately after model loads.

Usage:
    neurobrix serve --model TinyLlama-1.1B-Chat-v1.0 --hardware v100-32g
    neurobrix serve --model deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink
    neurobrix serve --model TinyLlama-1.1B-Chat-v1.0 --hardware v100-32g --foreground
"""

import sys


def cmd_serve(args):
    """Start the serving daemon."""
    from neurobrix.serving.client import DaemonClient
    from neurobrix.serving.server import ServingDaemon

    # Check if already running
    if DaemonClient.is_running():
        pid = DaemonClient.get_pid()
        print(f"[Serve] Daemon already running (PID {pid})")
        print(f"[Serve] Use 'neurobrix chat' to connect, or 'neurobrix stop' to shutdown.")
        sys.exit(1)

    # Determine execution mode
    if getattr(args, 'seq_aten', False):
        mode = "native"
    elif getattr(args, 'triton', False):
        mode = "triton"
    else:
        mode = "compiled"

    timeout = getattr(args, 'timeout', 1800) or 1800
    foreground = getattr(args, 'foreground', False)

    print(f"[Serve] Starting NeuroBrix Serving Daemon")
    print(f"[Serve] Model: {args.model}")
    print(f"[Serve] Hardware: {args.hardware}")
    print(f"[Serve] Mode: {mode}")
    print(f"[Serve] Idle timeout: {timeout}s")

    daemon = ServingDaemon(
        model_name=args.model,
        hardware_id=args.hardware,
        mode=mode,
        idle_timeout=float(timeout),
    )

    daemon.start(foreground=foreground)


def cmd_stop(args):
    """Stop the serving daemon with escalation: socket → SIGTERM → SIGKILL."""
    import os
    import signal
    import time
    from neurobrix.serving.client import DaemonClient
    from neurobrix.serving.protocol import SOCKET_PATH, PID_PATH

    if not DaemonClient.is_running():
        print("[Stop] No daemon running.")
        # Clean up stale files
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if PID_PATH.exists():
            PID_PATH.unlink()
        sys.exit(0)

    pid = DaemonClient.get_pid()
    print(f"[Stop] Sending shutdown to daemon (PID {pid})...")

    # Step 1: Try clean socket shutdown (3s timeout)
    socket_ok = False
    try:
        client = DaemonClient()
        client.connect()
        # Set socket timeout so we don't block forever
        if client._sock is not None:
            client._sock.settimeout(3.0)
        client.shutdown()
        client.close()
        socket_ok = True
    except Exception:
        pass

    if socket_ok:
        # Wait up to 5s for clean exit
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not DaemonClient.is_running():
                print("[Stop] Daemon stopped.")
                return
            time.sleep(0.2)

    # Step 2: SIGTERM (3s timeout)
    if pid is not None:
        print("[Stop] Sending SIGTERM...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            _cleanup_daemon_files(SOCKET_PATH, PID_PATH)
            print("[Stop] Daemon already exited.")
            return

        deadline = time.time() + 3.0
        while time.time() < deadline:
            if not DaemonClient.is_running():
                print("[Stop] Daemon stopped.")
                return
            time.sleep(0.2)

    # Step 3: SIGKILL (force)
    if pid is not None and DaemonClient.is_running():
        print("[Stop] Daemon unresponsive — sending SIGKILL...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        # Wait briefly for kernel to reap
        time.sleep(0.5)

    # Clean up files (daemon can't clean up after SIGKILL)
    _cleanup_daemon_files(SOCKET_PATH, PID_PATH)

    if DaemonClient.is_running():
        print(f"[Stop] WARNING: Daemon still alive after SIGKILL. Manual kill required: kill -9 {pid}")
        sys.exit(1)
    else:
        print("[Stop] Daemon stopped.")


def _cleanup_daemon_files(socket_path, pid_path):
    """Remove daemon socket and PID files."""
    for p in (socket_path, pid_path):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
