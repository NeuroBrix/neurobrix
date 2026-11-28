"""
neurobrix serve — Start persistent model serving daemon.

Loads model weights into VRAM once, keeps them warm.
Runs in background by default — terminal returns immediately after model loads.

Usage:
    neurobrix serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --hardware v100-32g
    neurobrix serve --model deepseek-ai/deepseek-moe-16b-chat --hardware c4140-4xv100-custom-nvlink
    neurobrix serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --hardware v100-32g --foreground
"""

import logging
import sys

log = logging.getLogger(__name__)


def _resolve_serve_mode(args) -> str:
    """Resolve and validate the execution mode for `neurobrix serve`."""
    from neurobrix.core.runtime import engines

    flags = [
        getattr(args, 'compiled', False),
        getattr(args, 'sequential', False),
        getattr(args, 'triton', False),
        getattr(args, 'triton_sequential', False),
    ]
    if sum(bool(f) for f in flags) > 1:
        log.error("Error: only one execution mode flag may be passed "
                  "(--compiled / --sequential / --triton / --triton-sequential).")
        sys.exit(1)
    if getattr(args, 'sequential', False):
        mode = "sequential"
    elif getattr(args, 'triton_sequential', False):
        mode = "triton_sequential"
    elif getattr(args, 'triton', False):
        mode = "triton"
    elif getattr(args, 'compiled', False):
        mode = "compiled"
    else:
        mode, said = engines.default_mode()
        log.info(f"[Serve] {said}")
    try:
        engines.require_engine(mode)
    except (engines.PyTorchBranchNotInstalledError, engines.TritonBranchNotInstalledError) as exc:
        log.error(f"Error: {exc}")
        sys.exit(1)
    return mode


def _wait_daemon_exit(timeout_s: float) -> bool:
    """Poll until the serving daemon stops or `timeout_s` elapses."""
    import time
    from neurobrix.serving.client import DaemonClient

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not DaemonClient.is_running():
            return True
        time.sleep(0.2)
    return False


def _send_socket_shutdown() -> bool:
    """Request a clean socket shutdown from the running daemon."""
    from neurobrix.serving.client import DaemonClient

    try:
        client = DaemonClient()
        client.connect()
        sock = getattr(client, "_sock", None)
        if sock is not None:
            sock.settimeout(3.0)
        client.shutdown()
        client.close()
        return True
    except Exception:
        return False


def _query_running_daemon() -> dict:
    """Return status dict of the running serving daemon, or empty dict on error."""
    from neurobrix.serving.client import DaemonClient

    try:
        client = DaemonClient()
        client.connect()
        st = client.status()
        client.close()
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def cmd_serve(args):
    """Start the serving daemon."""
    from neurobrix.serving.client import DaemonClient
    from neurobrix.serving.server import ServingDaemon, validate_keep_alive

    # Check if already running
    from neurobrix.cli.utils import resolve_model_arg
    args.model = resolve_model_arg(args.model)

    if DaemonClient.is_running():
        pid = DaemonClient.get_pid()
        log.error(f"Error: a daemon is already running (PID {pid}): neurobrix chat talks to it, neurobrix stop ends it")
        sys.exit(1)

    mode = _resolve_serve_mode(args)

    timeout = getattr(args, 'timeout', 1800) or 1800
    foreground = getattr(args, 'foreground', False)
    from neurobrix.core.state.price import validate_max_context
    try:
        max_context = validate_max_context(getattr(args, 'max_context', None))
        keep_alive = validate_keep_alive(getattr(args, 'keep_alive', None))
    except ValueError as exc:
        log.error(f"Error: {exc}")
        sys.exit(1)

    # Resolve hardware: explicit profile or auto-detect
    # The default profile is resolved by the DAEMON, in its own process (ServingDaemon._resolve_hardware): its
    # detection initialises the CUDA driver, and a CUDA context created before the fork of a background daemon is
    # unusable in the child (CUDA error: initialization error; the plan then fell to cpu_execution, measured
    # 2026-11-08 on card 0). Same profile, resolved where it is used.
    hardware_id = args.hardware or None

    log.debug(f"[Serve] {args.model} · hardware {hardware_id if args.hardware else 'auto-detect'} · mode {mode} · "
              f"idle timeout {timeout}s")

    http = None
    _http_flags = [f for f, v in (("--http-host", getattr(args, 'http_host', None)),
                                  ("--http-port", getattr(args, 'http_port', None)),
                                  ("--api-key-file", getattr(args, 'api_key_file', None))) if v is not None]
    if getattr(args, 'http', False):
        http = {"host": args.http_host, "port": args.http_port, "api_key_file": args.api_key_file}
    elif _http_flags:
        log.error(f"Error: {', '.join(_http_flags)} configure the HTTP API: add --http to open it.")
        sys.exit(1)

    daemon = ServingDaemon(
        model_name=args.model,
        hardware_id=hardware_id,
        mode=mode,
        idle_timeout=float(timeout),
        http=http,
        max_context=max_context,
        keep_alive=keep_alive,
    )

    daemon.start(foreground=foreground, ready_fd=getattr(args, 'ready_fd', None),
                 ready_file=getattr(args, 'ready_file', None))




def stop_daemon(quiet: bool = False) -> bool:
    """Stop the running serving daemon with escalation: socket shutdown (clean VRAM unload)
    -> SIGTERM -> SIGKILL. Returns True if no daemon is running afterward, False if it could
    not be stopped."""
    import os
    import signal
    import time
    from neurobrix.serving.client import DaemonClient
    from neurobrix.serving.protocol import SOCKET_PATH, PID_PATH, IS_WINDOWS, INSTANCE

    if not DaemonClient.is_running():
        dead = DaemonClient.get_pid()
        if not quiet:
            if dead is not None:
                from neurobrix.serving.protocol import LOG_PATH
                print(f"No daemon is running: the daemon (PID {dead}) ended without its shutdown — it was killed "
                      f"(a SIGKILL leaves no line in its log; the kernel's out-of-memory killer is the usual sender). "
                      f"Its last lines: {LOG_PATH}")
            else:
                print("No daemon is running.")
        [f.unlink() for f in INSTANCE.instance_files() if f.exists()]
        return True

    pid = DaemonClient.get_pid()
    log.debug(f"[Stop] Sending shutdown to daemon (PID {pid})...")

    if _send_socket_shutdown() and _wait_daemon_exit(5.0):
        if not quiet:
            print("Stopped the serving daemon.")
        return True

    if pid is not None:
        log.debug("[Stop] Sending termination signal...")
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            _cleanup_daemon_files(SOCKET_PATH, PID_PATH)
            if not quiet:
                print("The serving daemon had already exited.")
            return True
        if _wait_daemon_exit(3.0):
            if not quiet:
                print("Stopped the serving daemon.")
            return True

    # Step 3: Force kill (platform-adaptive)
    if pid is not None and DaemonClient.is_running():
        import subprocess as _sp
        log.warning("Warning: the daemon did not answer; it is killed")
        if IS_WINDOWS:
            try:
                _sp.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            except FileNotFoundError:
                os.kill(pid, signal.SIGTERM)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.5)

    _cleanup_daemon_files(SOCKET_PATH, PID_PATH)

    if DaemonClient.is_running():
        if IS_WINDOWS:
            log.error(f"Error: the daemon is still alive; kill it by hand: taskkill /F /PID {pid}")
        else:
            log.error(f"Error: the daemon is still alive after SIGKILL; kill it by hand: kill -9 {pid}")
        return False
    if not quiet:
        print("Stopped the serving daemon.")
    return True


def ensure_serving(
    model_name: str,
    *,
    hardware_id=None,
    mode=None,
    timeout: float = 1800.0,
    max_context=None,
) -> str:
    """Ensure a serving daemon is running with `model_name` (and `mode`/`hardware_id` if explicitly requested).

    Returns:
      * "reused"  — daemon already serves `model_name` (warm reuse, zero reload);
      * "swapped" — previous daemon cleanly shut down (VRAM unloaded) and new daemon started;
      * "started" — no daemon was running; new background daemon auto-started.
    """
    from neurobrix.serving.client import DaemonClient
    from neurobrix.serving.server import ServingDaemon
    from neurobrix.core.runtime import engines

    from neurobrix.core.state.price import validate_max_context
    max_context = validate_max_context(max_context)
    hardware_id = hardware_id or None

    if mode is not None:
        engines.require_engine(mode)

    daemon_running = DaemonClient.is_running()
    current_model = None
    if daemon_running:
        st = _query_running_daemon()
        current_model = st.get("model")
        if (current_model == model_name
                and (mode is None or st.get("mode") == mode)
                and (hardware_id is None or st.get("hardware") == hardware_id)
                and (max_context is None or st.get("max_context") == max_context)):
            return "reused"

    if mode is None:
        resolved_mode, _said = engines.default_mode()
        log.info(f"[Serve] {_said}")
    else:
        resolved_mode = mode

    # Validate engine availability BEFORE stopping any healthy running daemon.
    engines.require_engine(resolved_mode)

    action = "started"
    if daemon_running:
        log.info(
            f"[Serve] Auto-swapping daemon from {current_model or 'unknown'} to {model_name} "
            f"(unloading VRAM cleanly)..."
        )
        if not stop_daemon(quiet=True):
            raise RuntimeError(
                f"Cannot swap to {model_name}: the running serving daemon did not stop cleanly. "
                f"Stop it manually with: neurobrix stop"
            )
        action = "swapped"

    log.info(f"[Serve] Auto-starting daemon for {model_name} (mode={resolved_mode})...")
    daemon_kwargs = {"max_context": max_context} if max_context is not None else {}
    daemon = ServingDaemon(
        model_name=model_name,
        hardware_id=hardware_id,
        mode=resolved_mode,
        idle_timeout=float(timeout),
        **daemon_kwargs,
    )
    daemon.start(foreground=False, exit_parent=False)
    return action


def cmd_stop(args):
    """Stop the serving daemon with escalation: socket -> SIGTERM -> force kill."""
    if not stop_daemon(quiet=False):
        sys.exit(1)


def _cleanup_daemon_files(socket_path, pid_path):
    """Remove the instance's files: socket and pid, and a named Windows
    instance's port file (Studio request 8)."""
    from neurobrix.serving.protocol import INSTANCE
    for p in set((socket_path, pid_path)) | set(INSTANCE.instance_files()):
        try:
            if p is not None and p.exists():
                p.unlink()
        except OSError:
            pass
