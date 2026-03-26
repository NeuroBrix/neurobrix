"""Allow running neurobrix as: python -m neurobrix"""
import sys

# If invoked with no args, hint about PATH
if len(sys.argv) == 1:
    from neurobrix.cli._path_helper import check_cli_on_path
    check_cli_on_path()

from neurobrix.cli import main
main()
