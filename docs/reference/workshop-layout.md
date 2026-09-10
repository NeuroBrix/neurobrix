# Workshop layout — where things live on a measurement machine

Established 2026-09-10, after an audit found 55 git worktrees, 59 `nbx_*`
directories, five loose files and 68 GB of unlabelled artefacts spread across
the home directory of a rig whose root filesystem was 90 % full.

The rule is not tidiness for its own sake. A machine whose working tree cannot
be described cannot have its measurements attributed: the campaign running that
day was measuring an engine thirteen commits behind its own remote, from a
directory whose name said nothing about it, and nobody could tell from the
filesystem.

## The root holds three kinds of thing, and nothing else

```
/home/mlops/
├── NeuroBrix_System/     the repository — the only clone
├── ml/  venvs/  bench_venvs/     the virtualenvs
├── (the NFS mounts) and the tool caches
└── nbx/                  THE single root for all work
```

No loose file. No working directory. If something is neither the repository, a
virtualenv, nor a mount, it belongs under `nbx/`.

## Under `nbx/`, one directory per nature of thing

| directory | holds | removal condition |
|---|---|---|
| `nbx/worktrees/` | measurement worktrees, one per arm | when its commit is reachable from a remote branch — proven, then `git worktree remove` |
| `nbx/builds/` | `.nbx` containers produced locally | when the container is published to the hub, or superseded by a retrace |
| `nbx/stage/` | staging for builds and uploads | at the end of the operation that created it — never survives its campaign |
| `nbx/logs/` | logs not attached to a campaign | after 30 days, or once the finding they support is committed |
| `nbx/stats/` | measurement tables, timings, counters | when the table is committed to the repository or archived to the NAS |
| `nbx/campaigns/` | one dated directory per campaign | see below |
| `nbx/tmp/` | scratch that may be deleted at any moment | anytime, without asking |

Every one of those directories carries a `README.md` stating exactly that: what
it is for, and the condition under which its content may be removed. A directory
without that file is not part of the discipline.

## A campaign is a directory, and it cleans up after itself

Every campaign creates `nbx/campaigns/<YYYY_MM_DD>_<name>/`, writes **everything**
there — logs, arms, artefacts, the flight record — and at closure either cleans
it or archives it to the NAS with checksums.

**A campaign that ends without cleaning has not ended.** Its directory is the
first thing its closure report must account for.

## Nothing durable lives in a system temporary path

`/tmp` is wiped at boot and this rack has no UPS. Anything that must survive a
power cut — a build, a result, a staging area, a diagnostic script that has been
quoted in a report — is written under `nbx/` or into the repository, never under
`/tmp`, `/var/tmp` or a shell's scratch directory.

## The NAS is where closed work goes

`/mnt/nvme_ai_work` (1.6 TB) holds archived campaigns and artefacts. Anything
moved there goes with its checksums, and the inventory that authorised the move
is written before the move, not after.

Root-filesystem space is the scarce resource: 437 GB total, shared with the
engine cache. The NAS is not.

## The gate

`tests/unit/workshop/test_workshop_layout.py` fails when

* a loose file appears at the workshop root,
* a directory that is neither the repository, a virtualenv, a mount nor `nbx/`
  appears at the workshop root,
* a git worktree points anywhere outside `nbx/worktrees/`,
* a directory under `nbx/` has no `README.md`.

It is inert on any machine that has no `nbx/` root, so it does not fire on a
laptop or in CI — it holds the discipline exactly where the discipline applies.
