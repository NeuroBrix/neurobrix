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

import sys


def cmd_chat(args):
    """Interactive chat REPL connected to serving daemon."""
    from neurobrix.serving.client import DaemonClient

    # Check daemon is running
    if not DaemonClient.is_running():
        print("[Chat] No daemon running.")
        print("[Chat] Start one first:")
        print("  neurobrix serve --model <name> --hardware <profile>")
        sys.exit(1)

    # Connect
    client = DaemonClient()
    try:
        client.connect()
    except RuntimeError as e:
        print(f"[Chat] Connection failed: {e}")
        sys.exit(1)

    # Get status for display
    try:
        status = client.status()
    except RuntimeError as e:
        print(f"[Chat] Failed to get status: {e}")
        client.close()
        sys.exit(1)

    model_name = status.get("model", "unknown")
    family = status.get("family", "unknown")

    if family != "llm":
        print(f"[Chat] Model family is '{family}', not 'llm'.")
        print(f"[Chat] Use 'neurobrix run' for non-LLM models.")
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

    print(f"\nNeuroBrix Chat — {model_name}")
    print(f"Commands: /new /context /status /quit")
    print(f"{'─' * 50}\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
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
                    print("[New conversation started]\n")
                    continue

                elif cmd == "/context":
                    try:
                        st = client.status()
                        ctx = st.get("context", {})
                        if ctx:
                            turns = ctx.get("turns", 0)
                            tokens = ctx.get("estimated_tokens", 0)
                            max_ctx = ctx.get("max_context", 0)
                            usage = ctx.get("usage_pct", 0)
                            print(f"[Context] {turns} messages, ~{tokens} tokens, "
                                  f"{usage}% of {max_ctx} max\n")
                        else:
                            print("[Context] No active conversation\n")
                    except RuntimeError as e:
                        print(f"[Error] {e}\n")
                    continue

                elif cmd == "/status":
                    try:
                        st = client.status()
                        print(f"[Status] Model: {st.get('model')}")
                        print(f"[Status] Family: {st.get('family')}")
                        print(f"[Status] Mode: {st.get('mode')}")
                        print(f"[Status] VRAM: {st.get('vram_used_gb', '?')} GB\n")
                    except RuntimeError as e:
                        print(f"[Error] {e}\n")
                    continue

                else:
                    print(f"[Unknown command: {cmd}]")
                    print(f"Commands: /new /context /status /quit\n")
                    continue

            # Send chat message
            try:
                result = client.chat(user_input, **gen_kwargs)
                if result.get("summarized"):
                    print(f"\n[Context compressed — older turns summarized]")
                text = result.get("text", "")
                if text:
                    print(f"\nAssistant: {text}\n")
                else:
                    print(f"\n[No response generated]\n")
            except RuntimeError as e:
                print(f"\n[Error] {e}\n")

    except KeyboardInterrupt:
        print("\n")

    finally:
        client.close()
        print("[Chat] Disconnected")
