"""Allow running neurobrix as: python -m neurobrix"""
import sys

# If invoked with no args, hint about PATH
if len(sys.argv) == 1:
    from neurobrix.cli._path_helper import check_cli_on_path
    check_cli_on_path()

from neurobrix.cli import main

# The command's return value IS the process status. It used to be discarded — `main()` was
# called and its result dropped — so every `return cmd_xxx(args)` in `cli/__init__.py` was
# unobservable and the process exited 0 whatever the command decided. `autotune certify`
# computes `return 0 if not bad and not summary["failed"] else 1`, with a comment explaining
# why `unreachable` is deliberately excluded from that status; none of it could be seen.
# Measured 2026-09-23: the Apple campaign's addmm family printed `"failed": 1` in its own
# summary while the shell read rc=0, and the batched runner recorded the family COMPLETE with
# a census key uncertified. A status that cannot express failure is worse than no status,
# because everything downstream reads it as success.
#
# Only an INT becomes a status. A command returning a string would otherwise exit 1 and print
# that string as an error, turning a success into a failure on the way past.
_rc = main()
sys.exit(_rc if isinstance(_rc, int) else 0)
