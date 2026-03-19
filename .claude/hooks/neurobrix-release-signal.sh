#!/bin/bash
# PostToolUse hook: Remind to commit when enough source files edited
#
# Reads the dirty counter from neurobrix-auto-commit.sh.
# Fires a reminder at 10 edits, repeats every 20 after that.
#
# The counter resets when Claude makes a real commit (detected by
# checking if git has uncommitted changes in src/neurobrix/).

input=$(cat)
TOOL=$(echo "$input" | jq -r '.tool_name // ""')
FILE=$(echo "$input" | jq -r '.tool_input.file_path // ""')

# Only track Edit/Write to neurobrix source files
case "$FILE" in
  */src/neurobrix/*.py) ;;
  *) exit 0 ;;
esac

case "$FILE" in
  *__pycache__*|*.pyc|*/test_*.py) exit 0 ;;
esac

COUNTER_FILE="/tmp/.neurobrix_dirty_counter"

# If no dirty files tracked, nothing to signal
if [ ! -f "$COUNTER_FILE" ]; then
  exit 0
fi

COUNT=$(cat "$COUNTER_FILE")

# Check if working tree is clean (user committed) → reset counter
DIRTY=$(git -C /home/mlops/NeuroBrix_System diff --name-only -- 'src/neurobrix/*.py' 2>/dev/null | head -1)
if [ -z "$DIRTY" ]; then
  # Also check staged
  STAGED=$(git -C /home/mlops/NeuroBrix_System diff --cached --name-only -- 'src/neurobrix/*.py' 2>/dev/null | head -1)
  if [ -z "$STAGED" ]; then
    echo "0" > "$COUNTER_FILE"
    exit 0
  fi
fi

# Signal at threshold
THRESHOLD=10
if [ "$COUNT" -eq "$THRESHOLD" ]; then
  echo "" >&2
  echo "╔═══════════════════════════════════════════════════════════╗" >&2
  echo "║  COMMIT SIGNAL: $COUNT source files edited without commit     ║" >&2
  echo "║                                                         ║" >&2
  echo "║  Time to commit:                                        ║" >&2
  echo "║  1. Update CHANGELOG.md [Unreleased] section            ║" >&2
  echo "║  2. git add + commit (type: description)                ║" >&2
  echo "║     → post-commit hook auto-bumps alpha version         ║" >&2
  echo "║     → CHANGELOG [Unreleased] stamped with new version   ║" >&2
  echo "║  3. git push origin main                                ║" >&2
  echo "╚═══════════════════════════════════════════════════════════╝" >&2
elif [ "$((COUNT % 20))" -eq 0 ] && [ "$COUNT" -gt "$THRESHOLD" ]; then
  echo "COMMIT REMINDER: $COUNT uncommitted source edits." >&2
fi

exit 0
