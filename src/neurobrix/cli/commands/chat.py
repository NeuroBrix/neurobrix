"""
neurobrix chat — Interactive multi-turn chat with running daemon.

Connects to ServingDaemon via Unix socket. Supports:
    /new     — Start new conversation
    /context — Show token usage
    /status  — Show engine status
    /quit    — Exit chat

Usage:
    neurobrix chat
    neurobrix chat --temperature 0.7
"""

import logging
import sys

log = logging.getLogger(__name__)



def cmd_chat(args):
    """Interactive chat REPL connected to serving daemon."""
    from neurobrix.serving.client import DaemonClient

    if getattr(args, "agent", False):
        from neurobrix.cli.commands.agent import agent_defaults, cmd_agent_repl

        cmd_agent_repl(args, agent_defaults())
        return

    # Check daemon is running
    if not DaemonClient.is_running():
        log.error("Error: no daemon is running; start one first: neurobrix serve --model <name>")
        sys.exit(1)

    # Connect
    client = DaemonClient()
    try:
        client.connect()
    except RuntimeError as e:
        log.error(f"Error: cannot connect to the daemon: {(str(e).strip().splitlines() or [''])[0]}")
        sys.exit(1)

    # Get status for display
    try:
        status = client.status()
    except RuntimeError as e:
        log.error(f"Error: the daemon gave no status: {(str(e).strip().splitlines() or [''])[0]}")
        client.close()
        sys.exit(1)

    model_name = status.get("model", "unknown")
    family = status.get("family", "unknown")

    if family != "llm":
        log.error(f"Error: the daemon serves a '{family}' model, not an 'llm'; use neurobrix run for it")
        client.close()
        sys.exit(1)

    # Build generation kwargs from CLI args
    gen_kwargs = {}
    if getattr(args, 'max_tokens', None) is not None:
        gen_kwargs["max_tokens"] = args.max_tokens
    if getattr(args, 'temperature', None) is not None:
        gen_kwargs["temperature"] = args.temperature
    if getattr(args, 'repetition_penalty', None) is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty

    print(f"{model_name} — /new /context /status /quit")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print()          # ends the prompt's line
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]

                if cmd in ("/quit", "/exit", "/q"):
                    break

                elif cmd == "/new":
                    client.new_chat()
                    print("[New conversation started]")
                    continue

                elif cmd == "/context":
                    try:
                        st = client.status()
                        ctx = st.get("context", {})
                        if ctx:
                            turns = ctx.get("turns", 0)
                            tokens = ctx.get("tokens", 0)
                            max_ctx = ctx.get("max_context", 0)
                            usage = ctx.get("usage_pct", 0)
                            cached = ctx.get("cached_tokens", 0)
                            print(f"[Context] {turns} messages, {tokens} tokens, "
                                  f"{usage}% of {max_ctx} max, {cached} held in the KV cache")
                        else:
                            print("[Context] No active conversation")
                    except RuntimeError as e:
                        log.error(f"Error: {(str(e).strip().splitlines() or [''])[0]}")
                    continue

                elif cmd == "/status":
                    try:
                        st = client.status()
                        print(f"[Status] Model: {st.get('model')}")
                        print(f"[Status] Family: {st.get('family')}")
                        print(f"[Status] Mode: {st.get('mode')}")
                        print(f"[Status] VRAM: {st.get('vram_used_gb', '?')} GB")
                    except RuntimeError as e:
                        log.error(f"Error: {(str(e).strip().splitlines() or [''])[0]}")
                    continue

                else:
                    print(f"[Unknown command: {cmd}]")
                    print("Commands: /new /context /status /quit")
                    continue

            # Send chat message
            try:
                result = client.chat(user_input, **gen_kwargs)
                text = result.get("text", "")
                if text:
                    print(f"Assistant: {text}")
                else:
                    print("[No response generated]")
            except RuntimeError as e:
                log.error(f"Error: {(str(e).strip().splitlines() or [''])[0]}")

    except KeyboardInterrupt:
        print()

    finally:
        client.close()
        log.debug("[Chat] Disconnected")
