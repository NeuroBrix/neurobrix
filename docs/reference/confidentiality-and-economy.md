# Confidentiality and economy of the assisted work

Owner's rule, 2026-10-12. Applies to every assistant session on every machine of the project.

## Confidentiality

The work is confidential; this repository is public. Therefore:

* No feedback, bug report or survey is ever sent to an assistant tool's vendor from a session on this
  project. The only traffic an assistant tool may emit is its own model calls. Telemetry, error
  reports and feedback prompts are disabled in the tool's settings on every machine.
* A commit message never carries a tool-generated trailer or a session link: no `Co-Authored-By:`
  line naming an assistant, no `Claude-Session:` or similar URL. The message ends with its body.
* No assistant session is ever started from a shell command inside a worktree or a script
  (a non-interactive or scripted session). Such a session is invisible in the owner's terminal and
  billed outside the owner's view. Every model call goes through the one session the owner sees, or
  through a sub-agent that session declared in its report with the reason for it.

## Economy

Measured on 2026-10-12 on one machine: 155 messages in two hours, about 300 000 output tokens and
20.8 million context tokens re-read. The method, not the work, consumed the budget. Hence:

1. A sub-agent is launched only in extreme necessity, its reason written in the report before the
   launch. The default is zero.
2. A turn is never spent checking that a work is advancing. Every launched work has a shell waiter
   that wakes the session at its end, and nothing before.
3. A script of any length is written once into the campaign directory and relaunched by its path,
   never pasted into a tool call.
4. One command per step, outputs truncated (`tail`, `head`), no re-reading of a file already read.
5. The report and the session file are rewritten on disk, never repeated in the conversation.
6. Anything that is not visible in the owner's terminal is closed.

Related: `docs/reference/nothing-is-lost.md` (every launched work has its line, its deliverable path
and its end watcher).
