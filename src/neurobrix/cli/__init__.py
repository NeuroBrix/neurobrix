"""
NeuroBrix CLI — Package entry point.

Commands: run, info, inspect, validate, import, list, remove, clean, hub
"""

# Pre-import startup: configure OpenMP / MKL thread count from physical
# core count BEFORE any torch / numpy import below. PyTorch initialises
# its intra-op thread pool the first time ATen runs, and OpenMP reads
# `OMP_NUM_THREADS` at libgomp load time. Calling
# `torch.set_num_threads` after import is a no-op for the already-
# initialised thread pool. Doctrine R34: derived from `os.cpu_count()`
# at startup as a default; the Prism `apply_cpu_config` path may
# override with the hardware-profile value later. Users who set
# `OMP_NUM_THREADS` themselves keep their override (setdefault only).
# P-RUNTIME-HYBRID-DEVICE-DISPATCH 2026-05-12 — required for
# MKL/oneDNN to parallelise compute on the CPU portion of hybrid
# CPU+GPU placements (and on pure-CPU profiles).
import os as _os_startup
_cpu_count = _os_startup.cpu_count() or 1
_os_startup.environ.setdefault("OMP_NUM_THREADS", str(_cpu_count))
_os_startup.environ.setdefault("MKL_NUM_THREADS", str(_cpu_count))

import sys
import argparse
from pathlib import Path

from neurobrix import __version__
from neurobrix.cli.utils import REGISTRY_URL


def create_parser():
    """Create the main argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        prog='neurobrix',
        description='NeuroBrix - Universal Deep Learning Inference Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference (family read from manifest - not required!)
  neurobrix run --model Flex.1-alpha --hardware c4140-v100x4-nvlink \\
      --prompt "cyberpunk city at night" --steps 4

  # Show system info
  neurobrix info --models --hardware

  # Browse registry
  neurobrix hub

  # Inspect a .nbx file
  neurobrix inspect ~/.neurobrix/cache/Flex.1-alpha/model.nbx

For more information: https://neurobrix.es
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'NeuroBrix v{__version__}'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )

    # ========================================
    # RUN command
    # ========================================
    run_parser = subparsers.add_parser(
        'run',
        help='Run inference using NBX Engine',
        description='Run inference using Prism ExecutionPlan + NBX Engine. Family is read from manifest.'
    )
    run_parser.add_argument('--model', default=None, help='Model name (auto-detected from running daemon if omitted)')
    run_parser.add_argument('--hardware', default=None, help='Hardware profile ID (e.g., "v100-32g"). Auto-detected if omitted.')
    run_parser.add_argument('--prompt', default=None, help='Text prompt for generation')
    run_parser.add_argument('--audio', default=None, help='Input audio file path (for speech-to-text models)')
    run_parser.add_argument('--steps', type=int, default=None, help='Number of inference steps')
    run_parser.add_argument('--cfg', type=float, default=None, help='Guidance scale')
    run_parser.add_argument('--height', type=int, default=None, help='Output height in pixels')
    run_parser.add_argument('--width', type=int, default=None, help='Output width in pixels')
    run_parser.add_argument('--output', help='Output file path (default: output.png)')
    run_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    run_parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature (0 = greedy)')
    run_parser.add_argument('--repetition-penalty', type=float, default=None, dest='repetition_penalty',
                            help='Repetition penalty (1.0 = none, 1.1-1.5 recommended)')
    run_parser.add_argument('--set', action='append', metavar='KEY=VALUE',
                            help='Set arbitrary runtime variable (e.g., --set global.cfg=7.5)')
    # Execution modes (mutually exclusive). When no flag is passed the
    # default is --compiled. See CLAUDE.md "Execution Modes" section.
    run_parser.add_argument('--compiled', action='store_true',
                            help='Default mode: PyTorch fused graph + cuDNN/cuBLAS '
                                 '(CompiledSequence). Production PyTorch performance. '
                                 'Used when no execution flag is passed; this flag '
                                 'makes the choice explicit.')
    run_parser.add_argument('--sequential', action='store_true',
                            help='PyTorch eager op-by-op (no fusion, ATen dispatcher). '
                                 'Useful for debugging individual ATen ops. '
                                 'Slower than --compiled.')
    run_parser.add_argument('--triton', action='store_true',
                            help='Triton-pure compiled mode: NeuroBrix Triton kernels '
                                 'with arena + closures + fused kernels (TritonSequence). '
                                 'Production NeuroBrix Triton mode. Should match or beat '
                                 '--compiled on target shapes (project bet: custom '
                                 'kernels > cuDNN).')
    run_parser.add_argument('--triton-sequential', action='store_true', dest='triton_sequential',
                            help='Triton-pure eager mode: Triton kernels op-by-op (no '
                                 'fusion, TritonSequentialDispatcher). Equivalent of '
                                 '--sequential but for the Triton backend. Useful for '
                                 'debugging individual Triton kernels.')
    run_parser.add_argument('--max-tokens', type=int, default=None, dest='max_tokens',
                            help='Maximum number of tokens to generate (LLM only)')

    # ── 9-family inputs ──
    run_parser.add_argument('--input-image', default=None, dest='input_image',
                            help='Input image path (vlm, img2img, inpainting, upscaler, multimodal)')
    run_parser.add_argument('--mask-image', default=None, dest='mask_image',
                            help='Inpainting mask image path (image inpainting mode)')
    run_parser.add_argument('--reference-image', default=None, dest='reference_image',
                            help='Style reference image path (image)')
    run_parser.add_argument('--reference-audio', default=None, dest='reference_audio',
                            help='Voice clone reference audio path (tts)')
    run_parser.add_argument('--speaker', default=None,
                            help='Speaker preset id (tts)')
    run_parser.add_argument('--video', default=None, dest='video',
                            help='Input video path (video v2v)')
    run_parser.add_argument('--num-frames', type=int, default=None, dest='num_frames',
                            help='Number of frames to generate (video)')
    run_parser.add_argument('--fps', type=int, default=None,
                            help='Output frame rate (video)')
    run_parser.add_argument('--system', default=None,
                            help='System prompt for chat-style models (llm, vlm, audio_llm)')
    run_parser.add_argument('--mode', default=None,
                            choices=['auto', 'text', 'chat', 'image', 'audio', 'video',
                                     't2i', 'img2img', 'inpainting', 't2v', 'i2v', 'v2v'],
                            help='Execution mode. Required for multimodal (text|image). '
                                 'Auto-deduced for other families from inputs provided.')

    chat_group = run_parser.add_mutually_exclusive_group()
    chat_group.add_argument('--chat', action='store_true', default=None, dest='chat_mode',
                            help='Force chat template formatting')
    chat_group.add_argument('--no-chat', action='store_false', dest='chat_mode',
                            help='Force raw text completion')

    # ========================================
    # INFO command
    # ========================================
    info_parser = subparsers.add_parser(
        'info',
        help='Display system information',
        description='Show NeuroBrix system status, available models, and hardware'
    )
    info_parser.add_argument('--models', action='store_true', help='List available models')
    info_parser.add_argument('--hardware', action='store_true', help='Show hardware profiles')
    info_parser.add_argument('--system', action='store_true', help='Show system configuration')

    # ========================================
    # INSPECT command
    # ========================================
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Inspect a .nbx file',
        description='Show contents and metadata of a .nbx container'
    )
    inspect_parser.add_argument('nbx_path', help='Path to .nbx file')
    inspect_parser.add_argument('--topology', action='store_true', help='Show topology details')
    inspect_parser.add_argument('--weights', action='store_true', help='Show weight statistics')

    # ========================================
    # IMPORT command
    # ========================================
    import_parser = subparsers.add_parser(
        'import',
        help='Download model from NeuroBrix registry',
        description='Download a .nbx model from neurobrix.es and extract to local cache',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurobrix import pixart/sigma-xl-1024
  neurobrix import deepseek/janus-pro-7b
        """
    )
    import_parser.add_argument('model_ref', help='Model reference: org/name (e.g., sana/1600m-1024)')
    import_parser.add_argument('--registry', default=None, help=f'Registry URL (default: {REGISTRY_URL})')
    import_parser.add_argument('--force', action='store_true', help='Re-download even if already cached')
    import_parser.add_argument('--no-keep', action='store_true', dest='no_keep',
                               help='Delete .nbx from store after extraction (saves disk space)')

    # ========================================
    # LIST command
    # ========================================
    list_parser = subparsers.add_parser(
        'list',
        help='List installed models',
        description='Show installed models (cache) and downloaded archives (store)'
    )
    list_parser.add_argument('--store', action='store_true',
                             help='Show .nbx files in store (~/.neurobrix/store/)')

    # ========================================
    # REMOVE command
    # ========================================
    remove_parser = subparsers.add_parser(
        'remove',
        help='Remove a model (cache, store, or both)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Remove a model from cache, store, or both.',
        epilog="""
Examples:
  neurobrix remove 1600m-1024             # Cache only (default)
  neurobrix remove 1600m-1024 --store     # Store only (.nbx archive)
  neurobrix remove 1600m-1024 --all       # Both cache and store
        """
    )
    remove_parser.add_argument('model_name', help='Model name to remove (e.g., 1600m-1024)')
    remove_parser.add_argument('--store', action='store_true',
                               help='Remove from store only (keep cache)')
    remove_parser.add_argument('--all', action='store_true',
                               help='Remove from both cache and store')

    # ========================================
    # CLEAN command
    # ========================================
    clean_parser = subparsers.add_parser(
        'clean',
        help='Wipe all downloaded models (store and/or cache)',
        description='Completely remove all models from ~/.neurobrix/store/ and/or ~/.neurobrix/cache/'
    )
    clean_parser.add_argument('--store', action='store_true', help='Delete all .nbx files from store/')
    clean_parser.add_argument('--cache', action='store_true', help='Delete all extracted models from cache/')
    clean_parser.add_argument('--all', action='store_true', help='Delete both store and cache')
    clean_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt')

    # ========================================
    # HUB command — Browse registry
    # ========================================
    hub_parser = subparsers.add_parser(
        'hub',
        help='Browse models on the NeuroBrix registry',
        description='List all models available for download on neurobrix.es',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurobrix hub                        # List all models
  neurobrix hub --category IMAGE       # Filter by category
  neurobrix hub --search sana          # Search by name/tag
  neurobrix hub --category LLM -s chat # LLM models matching "chat"
        """
    )
    hub_parser.add_argument('--category', '-c', default=None,
                            choices=['IMAGE', 'VIDEO', 'AUDIO', 'SPEECH', 'LLM', 'UPSCALER',
                                     'image', 'video', 'audio', 'speech', 'llm', 'upscaler'],
                            help='Filter by model category')
    hub_parser.add_argument('--search', '-s', default=None, help='Search models by name, tag, or description')
    hub_parser.add_argument('--registry', default=None, help=f'Registry URL (default: {REGISTRY_URL})')

    # ========================================
    # SERVE command
    # ========================================
    serve_parser = subparsers.add_parser(
        'serve',
        help='Start persistent model serving daemon',
        description='Load model weights into VRAM and serve requests via Unix socket.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurobrix serve --model TinyLlama-1.1B-Chat-v1.0 --hardware v100-32g
  neurobrix serve --model TinyLlama-1.1B-Chat-v1.0 --hardware v100-32g --timeout 3600
        """
    )
    serve_parser.add_argument('--model', required=True, help='Model name')
    serve_parser.add_argument('--hardware', default=None, help='Hardware profile ID. Auto-detected if omitted.')
    serve_parser.add_argument('--timeout', type=int, default=1800,
                              help='Idle timeout in seconds (default: 1800)')
    serve_parser.add_argument('--foreground', action='store_true',
                              help='Run in foreground (block terminal, for debugging)')
    # Execution modes (mutually exclusive). When no flag is passed the
    # default is --compiled. Mirror of run_parser; see CLAUDE.md
    # "Execution Modes" section for the contract.
    serve_parser.add_argument('--compiled', action='store_true',
                              help='Default mode: PyTorch fused graph + cuDNN/cuBLAS')
    serve_parser.add_argument('--sequential', action='store_true',
                              help='PyTorch eager op-by-op (no fusion, debug)')
    serve_parser.add_argument('--triton', action='store_true',
                              help='Triton-pure compiled mode (NeuroBrix Triton kernels)')
    serve_parser.add_argument('--triton-sequential', action='store_true', dest='triton_sequential',
                              help='Triton-pure eager op-by-op (debug Triton)')

    # ========================================
    # CHAT command
    # ========================================
    chat_parser = subparsers.add_parser(
        'chat',
        help='Interactive chat with running daemon',
        description='Connect to a running serving daemon for multi-turn LLM chat.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurobrix chat
  neurobrix chat --temperature 0.7
  neurobrix chat --repetition-penalty 1.2

Slash commands (inside chat):
  /new      Start new conversation
  /context  Show token usage
  /status   Show engine status
  /quit     Exit chat
        """
    )
    chat_parser.add_argument('--max-tokens', type=int, default=None,
                             dest='max_tokens',
                             help='Max tokens per response (default: from model config)')
    chat_parser.add_argument('--temperature', type=float, default=None,
                             help='Sampling temperature')
    chat_parser.add_argument('--repetition-penalty', type=float, default=None,
                             dest='repetition_penalty',
                             help='Repetition penalty (1.0 = none)')

    # ========================================
    # STOP command
    # ========================================
    subparsers.add_parser(
        'stop',
        help='Stop the serving daemon and free VRAM',
        description='Gracefully stop the running daemon, unload weights, and free GPU memory.',
    )

    # ========================================
    # DOCTOR command
    # ========================================
    subparsers.add_parser(
        'doctor',
        help='Diagnose installation issues (PATH, etc.)',
        description='Check that neurobrix is correctly installed and on PATH.',
    )

    # ========================================
    # VALIDATE command
    # ========================================
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate NBX file integrity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
Validate .nbx archive structure and integrity.

Validation levels:
  structure  - Check files exist in ZIP
  schema     - Parse JSON and validate required fields
  coherence  - Cross-reference weights_index with shards (default)
  deep       - Read safetensors headers, verify checksums
''',
    )
    validate_parser.add_argument('nbx_files', nargs='+', type=Path, help='Path(s) to .nbx file(s)')
    validate_parser.add_argument('--level', '-l', default='coherence',
                                 choices=['structure', 'schema', 'coherence', 'deep'],
                                 help='Validation depth (default: coherence)')
    validate_parser.add_argument('--strict', action='store_true', help='Exit on first failure')
    validate_parser.add_argument('--json', action='store_true', help='Output results as JSON')
    validate_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed info')

    # ========================================
    # UPSCALE command — image super-resolution
    # ========================================
    upscale_parser = subparsers.add_parser(
        'upscale',
        help='Upscale an image using a super-resolution model',
        description='Image super-resolution. Loads an input image, '
                    'runs it through an upscaler model, writes the '
                    'high-resolution result.'
    )
    upscale_parser.add_argument('--model', required=True,
                                help='Upscaler model name (e.g. '
                                     'swin2SR-realworld-sr-x4-64-bsrgan-psnr)')
    upscale_parser.add_argument('--input', required=True,
                                help='Input image path (PNG/JPEG)')
    upscale_parser.add_argument('--output', required=True,
                                help='Output image path (PNG)')
    upscale_parser.add_argument('--mode', default='compiled',
                                choices=['compiled', 'sequential',
                                         'triton', 'triton-sequential'],
                                help='Execution mode (default: compiled)')
    upscale_parser.add_argument('--hardware', default=None,
                                help='Hardware profile ID '
                                     '(auto-detected if omitted)')

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.command == 'run':
            from neurobrix.cli.commands.run import cmd_run
            cmd_run(args)
        elif args.command == 'import':
            from neurobrix.cli.commands.registry import cmd_import
            cmd_import(args)
        elif args.command == 'list':
            from neurobrix.cli.commands.registry import cmd_list
            cmd_list(args)
        elif args.command == 'remove':
            from neurobrix.cli.commands.registry import cmd_remove
            cmd_remove(args)
        elif args.command == 'clean':
            from neurobrix.cli.commands.registry import cmd_clean
            cmd_clean(args)
        elif args.command == 'hub':
            from neurobrix.cli.commands.registry import cmd_hub
            cmd_hub(args)
        elif args.command == 'info':
            from neurobrix.cli.commands.info import cmd_info
            cmd_info(args)
        elif args.command == 'inspect':
            from neurobrix.cli.commands.info import cmd_inspect
            cmd_inspect(args)
        elif args.command == 'serve':
            from neurobrix.cli.commands.serve import cmd_serve
            cmd_serve(args)
        elif args.command == 'chat':
            from neurobrix.cli.commands.chat import cmd_chat
            cmd_chat(args)
        elif args.command == 'stop':
            from neurobrix.cli.commands.serve import cmd_stop
            cmd_stop(args)
        elif args.command == 'validate':
            from neurobrix.cli.commands.validate import cmd_validate
            cmd_validate(args)
        elif args.command == 'upscale':
            from neurobrix.cli.commands.upscale import cmd_upscale
            cmd_upscale(args)
        elif args.command == 'doctor':
            from neurobrix.cli._path_helper import print_path_diagnostics
            print_path_diagnostics()
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
