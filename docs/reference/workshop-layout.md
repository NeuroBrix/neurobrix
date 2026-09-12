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

## The hub is on this network, and a publish never leaves it

**Established 2026-09-12.** `neurobrix.es` sits behind Cloudflare, and Spanish
operators apply a court order — Juzgado de lo Mercantil nº 6 de Barcelona,
18 December 2024, in force through the 2026/27 season — to Cloudflare's shared
addresses during football matches. Thousands of legitimate sites fall with it
every weekend. On a Saturday evening the name resolved from this rack to a
blocking device presenting a self-signed certificate (`CN=core1.netops.test`,
`O=Widgits Pty Ltd`, valid to 2124), and every publish failed certificate
verification — correctly — while the hub's real addresses, three metres away,
served `CN=neurobrix.es`.

The refusal was right and stays: **bypassing an interception is deciding alone
that it is benign.** The tool could not know the interceptor was a court. What it
also could not do is keep taking the road the court watches.

- **The rack declares its own entry point** in `nbx/env.sh`, sourced by
  `~/.profile` (not only `~/.bashrc`, which returns before the end for a
  non-interactive shell) and by every chain: `NEUROBRIX_REGISTRY=http://10.0.0.39:3000`.
  Both the engine (`neurobrix hub`, `neurobrix import`) and the build toolchain
  read it; the public name is what remains when nothing is declared, because a
  user anywhere else reaches the hub through it.
- **It was verified to be the same instance, not assumed**: the internal listing
  returns the same record ids as the public one and 47 models at `?limit=200`.
  The first read of it returned 12 — the page size — and was taken for a
  different instance for several minutes. So every publish command now asks the
  registry for its listing at entry and refuses one that does not answer in the
  hub's shape: the case that catches is an address answering HTTP 200 with
  *something*.
- **The object store was already internal** (`10.0.0.36:9000`): the API hands
  out presigned URLs on the LAN. Only the API calls were leaving the building.
- **Stated, not hidden: the internal entry point is plain HTTP on port 3000.**
  The admin token travels in clear between `10.0.0.40` and `10.0.0.39` on this
  LAN. The owner chose this path knowing it. A certificate on the internal
  listener would remove the caveat and nothing here would need to change.

And the retry policy learned the one failure it must never retry: a certificate
that does not verify carries no HTTP response, so it read as transient and would
have been re-offered three times. It now stops at once, recognised through four
layers of wrapping.

## One pool carries everything — and the rule that was inferred from it, withdrawn

**Established 2026-09-12, on the host, by the owner.** There is no separate
storage box. `192.168.100.1` is the hypervisor's own 100 Gbps link (a Mellanox
card — the OUI `b8:59:9f` is the card's maker, not a machine). Everything this
rack writes sits on **one ZFS pool, `data`**: the `hf_snapshots` and `models`
exports, **and the zvols of every VM**, including the object store at
`10.0.0.36` and the hub at `10.0.0.39`. The pool is rotational, at 74% capacity
and 40% fragmentation. That topology is a fact and it stays.

**The rule "one pool writer at a time" is WITHDRAWN (owner, 2026-09-12 22:49),
with its reason, so that it does not come back under another name.** It was
inferred from a single incident whose cause was confounded. It was born two
hours after a double mains cut, on a MinIO that had marked its own drive as
hung (`/minio/health/cluster` answering 503) and was refusing three writes in
four; during the incident the host's I/O was near zero and the load was
falling — the pool was not saturated, the store was stuck in the state it had
put itself in. The owner's host reboot cleared it instantly (9.78 MB/s and zero
refusals on the first probe after). That is a wedged state, not a capacity
limit. And the fact that weighs most: **this machine had run three or four
simultaneous copies for months without an incident.** An inference made in a
panic does not hold against months of practice. Both sides got it wrong — the
one who wrote the rule and the one who endorsed it — and it is exactly the
shape the vacuous-gates register describes: a cause placed on one point where
two different rules produced the same symptom. Register entry 44.

**What stays, because it was measured separately**: `SlowDownWrite` had already
fired on 2026-09-07, on a burst at 548 MB/s. That was an instantaneous-rate
problem, not a concurrency problem, and adaptive pacing settled it — uploads
start at **40 MB/s** and halve when the store asks; the store keeps the last
word. Do not raise the start without a measurement.

**What replaces the rule is not another rule; it is the sensor.**
`tools/export_quiet.py` reads bytes off the export — never `df`, never the load
average — and refuses to launch a heavy write while the export serves under
40 MB/s. If the pool is well, everything runs in parallel as it always did;
if it degrades, the sensor sees it and the work waits. A measurement in place
of a policy.

**And the corrected rule for the cards, which the wrong one had idled**: pool
writes are gated by the sensor; **the cards never stop while there is
certification or bench work to do.** A disk stream in the background and four
cards certifying in front is this machine's normal state. Applying the pool
rule to the GPUs left four V100s at zero for hours behind a 118 GB upload —
certification reads the local cache, writes a few kilobytes of JSON, and never
touches the pool.

**The durable repair is not on this machine**: move the store's and the hub's
disks to `nvme_pool`, which sits at 48% and carries nothing. Proposed to the
owner the same evening.

**What NOT to infer from a MAC address.** The wrong topology — a separate NAS
box, the store on another machine — came from reading three MAC prefixes as
three machines. A prefix identifies the maker of a network card. It says
nothing about what is behind the card, and it said nothing here: two of the
three were VMs on the host and the third was the host's own link. Register
entry 43.

### When the store says `SlowDownWrite` on every write

Read `http://10.0.0.36:9000/minio/health/cluster` before anything else. `/live`
and `/ready` answer 200 while the process is alive; **`/cluster` answers 503 when
MinIO has marked its drive as hung**, and from then on every write returns
`SlowDownWrite: Resource requested is unwritable` until the drive recovers or the
service restarts. Measured 2026-09-12: the network was 10 GbE to the store and
200 GbE to the pool with 0.45 ms RTT, `eno1` was emitting 7 kB/s, and the upload
crawled at 398 kB/s — the link was never the bottleneck, the store was refusing.

The store is **VM 230 `storage-MNIO`** on the hypervisor `optimus`
(`192.168.100.1` on the private link), disk `vm-storage:vm-230-disk-1`, 4 TB.
The hub is **VM 101 `NeuroBrix`**. When the guest is wedged in I/O its SSH port
gives no banner and `qm guest cmd 230 ping` does not answer; a `qm stop` then
needs SIGKILL and a `qm start` right after can fail with *"timeout waiting on
systemd"* while the old scope tears down. On the night this was learned the
owner rebooted the host, which brought everything back at once: exports
serving 424 MB/s O_DIRECT, `/cluster` 200, the hub reporting 47.

The order of checks that would have saved an hour: `/minio/health/cluster`
first, then `ethtool`/RTT, then the pool from the host. A 503 there is the
diagnosis; everything else is confirmation.

### Measured the same night: one GPU reader beside the upload cost the store nothing

While the withdrawn rule was still in force it was applied as "nothing heavy at
all while an upload runs", and the GPUs sat idle behind a 53-minute upload. Then
it was tested with the instrument at hand — the upload's own rate, sampled every
minute — by running one GPU proof (CogVideoX-5b-I2V, a 21.5 GB container read
from the cache once, then compute) beside the Wan2.2 upload:

| minute | upload | `SlowDownWrite` | GPU 2 |
|---:|---:|---:|---|
| +1 | 36.7 MB/s | 0 | 92 %, 21.2 GB (loading) |
| +2 | 30.5 MB/s | 0 | 100 %, 23.1 GB (computing) |

A dip of six MB/s during the load, no refusal, and the proof returned PROVEN.
The measurement stands; the conclusion drawn from it that night ("keep the
writers serial") was the withdrawn rule restated, and it is not kept.
