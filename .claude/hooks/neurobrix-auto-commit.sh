#!/bin/bash
# PostToolUse hook: Track dirty source files (NO auto-commit)
#
# OLD BEHAVIOR (BROKEN): auto-committed every file edit with --no-verify
#   → bypassed CHANGELOG enforcement, created 100+ noise commits,
#   → defeated release-signal counter, version never bumped
#
# NEW BEHAVIOR: tracks which files are dirty in a temp file.
# Claude makes proper grouped commits with:
#   - Meaningful commit messages (type: description)
#   - CHANGELOG.md updates (enforced by pre-commit hook)
#   - Post-commit hook auto-bumps alpha version + stamps CHANGELOG

input=$(cat)
FILE=$(echo "$input" | jq -r '.tool_input.file_path // ""')

# Only track neurobrix source files
case "$FILE" in
  */src/neurobrix/*.py) ;;
  *) exit 0 ;;
esac

case "$FILE" in
  *__pycache__*|*.pyc|*/test_*.py) exit 0 ;;
esac

# Increment dirty counter (release-signal reads this)
COUNTER_FILE="/tmp/.neurobrix_dirty_counter"
COUNT=0
if [ -f "$COUNTER_FILE" ]; then
  COUNT=$(cat "$COUNTER_FILE")
fi
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNTER_FILE"

exit 0
