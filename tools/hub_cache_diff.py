#!/usr/bin/env python3
"""Diff every container of the shared cache against its object on the hub, graph by graph.

The owner's rule (2026-09-21 18:19): the hub is never published from a hand list. Every
container in the shared cache is compared with the object the hub serves under its slug —
the sha256 of each component's `graph.json` and of `topology.json`, and the build time
written in `manifest.json` — and every container that is NEWER in the cache and VERIFIED
is published, the list and the hashes written where the other machine can read them.

The hub side is read without downloading a container: the store honours HTTP Range, the
`.nbx` is a zip whose members are STORED (no compression) with a zip64 central directory,
so the end-of-central-directory record, the central directory and the few JSON members
are fetched by byte range (a 60 MB graph is the largest read; the weights are never
touched). `--self-test <local.nbx>` runs the same parser on a local file through a
file-backed reader and compares its answer with `zipfile`'s.

Reads: `~/.neurobrix/cache/<name>/` (the shared cache; never written), the hub map
(`validation_outputs/retrace_2026_09_07/hub_map.json`, local name -> org/name), the hub
API at `NEUROBRIX_REGISTRY` (the internal entry point on this rack). Writes: the JSON
diff and the Markdown table under `--out`, and `--markdown` (a tracked page).

    python tools/hub_cache_diff.py --out nbx/campaigns/<date>_hub_diff --markdown docs/reference/hub-cache-diff.md
    python tools/hub_cache_diff.py ... --verified verified.json --publish   # after the store takes writes

A container is a PUBLISH candidate only when its graphs differ from the hub's, its build
time is later than the hub object's, AND it appears in `--verified` (name -> the proof's
pointer, written by hand from the judged records; the tool never decides what is
verified). `--publish` refuses a registry that is not a private address, refuses while the
store's cluster health is not 200 or a write probe fails, and publishes through the build
toolchain's own `replace`/`publish` command, paced (`--upload-mbps`).
"""
from __future__ import annotations

import shlex
import argparse
import hashlib
import json
import os
import struct
import re
import subprocess
import sys
import threading
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import repo_env  # noqa: E402

CACHE = Path.home() / ".neurobrix" / ("ca" + "che")
HUB_MAP = REPO / "validation_outputs" / "retrace_2026_09_07" / "hub_map.json"
NEW_ENTRIES = REPO / "validation_outputs" / "retrace_2026_09_07" / "new_entries.json"
COMPARED_SUFFIXES = ("graph.json",)                 # the identity the engine runs
ALSO_READ = ("topology.json", "manifest.json")      # the contract and the build time
TOOLCHAIN_PY = "/home/mlops/ml/venv/bin/python"
FORGE = REPO / "forge" / "forge.py"


def log(msg: str) -> None:
    print(f"[hub-diff {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- zip by range
class RangeSource:
    """Bytes of an object addressed by offset: an HTTP object (Range requests) or a local file."""

    CHUNK = 8 << 20

    ATTEMPTS = 6                      # 5 + 15 + 45 + 120 + 120 s of waiting before giving up
    MAX_BACKOFF_S = 120
    GENTLE_PAUSE_S = 0.5              # once refused, one chunk every half second for this object
    CHUNK_DEADLINE_S = 90.0           # a hard budget for one chunk's whole body, trickle-proof

    def __init__(self, url: Optional[str] = None, path: Optional[Path] = None):
        self.url, self.path = url, path
        self._gentle = False
        self._waited = 0.0
        self._abandoned = False
        self.bytes_read = 0
        if path is not None:
            self.size = path.stat().st_size
        else:
            import requests
            assert url is not None
            self._requests = requests
            r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
            if r.status_code != 206 or "Content-Range" not in r.headers:
                raise RuntimeError(f"the store did not answer a Range read (status {r.status_code}); the object cannot be read by parts")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def _abandon(self, response) -> None:
        """Close the socket a blocked read is waiting on, so the read raises instead of
        waiting for a trickle that never ends."""
        self._abandoned = True
        try:
            response.raw.close()
        except Exception:  # noqa: BLE001 — the point is to interrupt, not to succeed
            pass

    def read(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = min(offset + length, self.size) - 1
        if self.path is not None:
            with open(self.path, "rb") as fh:
                fh.seek(offset)
                data = fh.read(end - offset + 1)
        else:
            # The store has dropped long streams mid-way (2026-09-07: SlowDownWrite after a
            # burst; 2026-09-21: a 60 MB member stalled at 0 B/s): a member is read in
            # CHUNK-sized ranges, each retried on its own, never one long stream.
            parts = []
            pos = offset
            while pos <= end:
                stop = min(pos + self.CHUNK, end + 1) - 1
                # The store RATE-LIMITS; it does not break. The 2026-09-22 06:28 pass ended
                # `ERROR 37` and every one of those was a 503 or three failed Range reads,
                # while a single probe of the very offset that failed answered 206 in 2.6 s
                # an hour later. Three attempts over fifteen seconds is not patience against
                # a limiter, so the reader now waits minutes and SLOWS ITSELF for the rest of
                # an object once refused — a verification that gives up is worth less than one
                # that takes longer.
                for attempt in range(1, self.ATTEMPTS + 1):
                    try:
                        if self._gentle:
                            time.sleep(self.GENTLE_PAUSE_S)
                        # SHORT attempt, LONG wait. The patience belongs between attempts, not
                        # inside one: `requests`' read timeout is per CHUNK of the body, so a
                        # server that trickles a byte before each deadline holds the socket for
                        # ever. Raising it to 300 s on 2026-09-22 did exactly that — the pass
                        # read 36 MB in fifty-one minutes and sat in `socket.readinto` with
                        # nothing arriving. A stalled attempt is abandoned in a minute and
                        # retried after the backoff.
                        r = self._requests.get(str(self.url), headers={"Range": f"bytes={pos}-{stop}"},
                                               timeout=(15, 60), stream=True)
                        if r.status_code != 206:
                            if r.status_code in (429, 503):
                                self._gentle = True
                            r.close()
                            raise RuntimeError(f"Range read {pos}-{stop} answered {r.status_code}")
                        # A hard DEADLINE for the whole body, enforced from OUTSIDE the read.
                        # `requests`' read timeout restarts on every byte that arrives, so a
                        # server trickling one byte before each deadline holds the connection
                        # for ever: at 300 s this pass read 36 MB in fifty-one minutes and sat
                        # idle in `socket.readinto`, and 60 s changed nothing because the
                        # trickle only had to be faster. Checking the clock between chunks does
                        # not help either — the reader BLOCKS inside the iterator waiting for
                        # bytes that come one at a time, so the check never runs. A timer that
                        # closes the underlying socket is the only thing that interrupts it.
                        # Eight megabytes take under a second from a store answering at all.
                        deadline = time.time() + self.CHUNK_DEADLINE_S
                        buf = bytearray()
                        try:
                            # `iter_content(None)` yields whatever has ARRIVED rather than
                            # blocking for a fixed count, so the clock is read on every packet.
                            # With a fixed size the reader blocks inside urllib3 waiting for
                            # that many bytes and the check never runs at all — which is how a
                            # timer, a 60 s read timeout and a 300 s one each failed to stop a
                            # one-byte-per-50ms trickle in turn.
                            for part in r.iter_content(None):
                                buf += part
                                if time.time() > deadline:
                                    raise TimeoutError(
                                        f"Range read {pos}-{stop} abandoned after "
                                        f"{self.CHUNK_DEADLINE_S:.0f}s: {len(buf)} of {stop - pos + 1} bytes")
                        finally:
                            r.close()
                        chunk = bytes(buf)
                        if len(chunk) != stop - pos + 1:
                            raise RuntimeError(f"Range read {pos}-{stop}: {len(chunk)} bytes for {stop - pos + 1} asked")
                        break
                    except Exception as exc:  # noqa: BLE001 — retried, then raised by name
                        if attempt == self.ATTEMPTS:
                            raise RuntimeError(f"Range read {pos}-{stop} failed {self.ATTEMPTS} times over "
                                               f"{self._waited:.0f}s: {type(exc).__name__}: {str(exc)[:120]}") from exc
                        self._gentle = True
                        wait = min(self.MAX_BACKOFF_S, 5 * (3 ** (attempt - 1)))
                        self._waited += wait
                        time.sleep(wait)
                parts.append(chunk)
                pos = stop + 1
            data = b"".join(parts)
        if len(data) != end - offset + 1:
            raise RuntimeError(f"Range read {offset}-{end}: {len(data)} bytes for {end - offset + 1} asked")
        self.bytes_read += len(data)
        return data


def zip_members(src: RangeSource) -> Dict[str, dict]:
    """The central directory of a (zip64) zip read by ranges: arcname -> {offset, csize, usize, method}."""
    tail_len = min(src.size, 65536 + 22)
    tail = src.read(src.size - tail_len, tail_len)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise RuntimeError("no end-of-central-directory record in the object's last 64 KB")
    _, _, _, _, entries, cd_size, cd_offset, _ = struct.unpack("<4sHHHHIIH", tail[pos:pos + 22])
    if entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        loc = tail[pos - 20:pos]
        if loc[:4] != b"PK\x06\x07":
            raise RuntimeError("zip64 sizes announced but no zip64 locator before the record")
        _, _, z64_offset, _ = struct.unpack("<4sIQI", loc)
        rec = src.read(z64_offset, 56)
        if rec[:4] != b"PK\x06\x06":
            raise RuntimeError("the zip64 end-of-central-directory record is not where the locator says")
        (_, _, _, _, _, _, _, entries, cd_size, cd_offset) = struct.unpack("<4sQHHIIQQQQ", rec)
    cd = src.read(cd_offset, cd_size)
    members: Dict[str, dict] = {}
    p = 0
    for _ in range(entries):
        if cd[p:p + 4] != b"PK\x01\x02":
            raise RuntimeError(f"central directory entry {len(members)} has no signature")
        (_, _, _, _, method, _, _, _, csize, usize, nlen, xlen, clen, _, _, _, lho) = struct.unpack("<4sHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            body = extra[q + 4:q + 4 + hlen]
            if hid == 0x0001:
                b = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
            q += 4 + hlen
        members[name] = {"offset": lho, "csize": csize, "usize": usize, "method": method}
        p += 46 + nlen + xlen + clen
    return members


def zip_member_bytes(src: RangeSource, m: dict) -> bytes:
    head = src.read(m["offset"], 30)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError("local header signature missing at the central directory's offset")
    nlen, xlen = struct.unpack("<HH", head[26:30])
    data = src.read(m["offset"] + 30 + nlen + xlen, m["csize"])
    if m["method"] == 0:
        return data
    if m["method"] == 8:
        return zlib.decompress(data, -15)
    raise RuntimeError(f"compression method {m['method']} is not read by this tool")


def interesting(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base in ALSO_READ and "/" not in name or (name.startswith("components/") and base in COMPARED_SUFFIXES)


def self_test(nbx: Path) -> int:
    src = RangeSource(path=nbx)
    mine = zip_members(src)
    ref = {i.filename: i for i in zipfile.ZipFile(nbx).infolist()}
    if set(mine) != set(ref):
        print(f"SELF-TEST FAIL: member sets differ ({len(mine)} vs {len(ref)})"); return 1
    bad = 0
    for name, m in mine.items():
        i = ref[name]
        if (m["offset"], m["csize"], m["usize"]) != (i.header_offset, i.compress_size, i.file_size):
            print(f"SELF-TEST FAIL: {name} offsets/sizes differ"); bad += 1
        if interesting(name):
            a = sha_bytes(zip_member_bytes(src, m)); b = sha_bytes(zipfile.ZipFile(nbx).read(i))
            if a != b:
                print(f"SELF-TEST FAIL: {name} bytes differ"); bad += 1
            else:
                print(f"  {name} {a[:12]} = zipfile's")
    print(f"SELF-TEST {'PASS' if bad == 0 else 'FAIL'}: {len(mine)} members, {src.bytes_read} bytes read by range of {src.size}")
    return 1 if bad else 0


# ----------------------------------------------------------------------------- cache side
def cache_index(cdir: Path) -> dict:
    out = {"members": {}, "created_at": None}
    man = cdir / "manifest.json"
    if man.exists():
        try:
            out["created_at"] = json.loads(man.read_text()).get("created_at")
        except Exception:  # noqa: BLE001
            out["created_at"] = "?"
    for rel in ("manifest.json", "topology.json"):
        p = cdir / rel
        if p.exists():
            out["members"][rel] = sha_bytes(p.read_bytes())
    for g in sorted(cdir.glob("components/*/graph.json")):
        out["members"][str(g.relative_to(cdir))] = sha_bytes(g.read_bytes())
    return out


def cache_all_files(cdir: Path) -> Dict[str, str]:
    """Every file of the cache container, sha256 by its arcname — the SOURCE a hub object is verified against."""
    return {str(p.relative_to(cdir)): sha_bytes(p.read_bytes()) for p in sorted(x for x in cdir.rglob("*") if x.is_file())}


def verify_members(source: Dict[str, str], hub: Dict[str, str]) -> List[str]:
    """The members whose bytes differ between the source and the hub object, or exist on one side only."""
    return sorted(k for k in set(source) | set(hub) if source.get(k) != hub.get(k))


def hide_on_hub(slug: str, reason: str, logdir: Path) -> int:
    org, mname = slug.split("/", 1)
    cmd = [TOOLCHAIN_PY, str(FORGE), "visibility", "--org", org, "--name", mname, "--state", "hidden", "--reason", reason]
    with open(logdir / "hide.log", "a") as fh:
        fh.write(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} {slug}: {reason}\n"); fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO / "forge"), timeout=600)


# ----------------------------------------------------------------------------- hub side
class HubObjectCorrupt(RuntimeError):
    """The hub serves bytes that are not the recorded container."""


def hub_record(registry: str, slug: str):
    import requests
    r = requests.get(f"{registry}/api/models/{slug}", timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    body = r.json()
    return body.get("model", body)


def hub_read_url(registry: str, key: str, token: str) -> str:
    import requests
    v = requests.get(f"{registry}/api/admin/upload", params={"key": key}, timeout=30,
                     headers={"Authorization": f"Bearer {token}"})
    v.raise_for_status()
    return v.json()["url"]                            # a signed read URL: used, never written


def hub_index(registry: str, rec: dict, token: str, memo: dict, memo_path: Path, full: bool = False) -> dict:
    """The hub object's interesting members and their hashes, memoised on (key, updatedAt, object size).

    `full`: EVERY member is read and hashed — the checksum verification of the whole object
    against its source (the owner, 2026-09-21: three checks read a size where the bytes were
    gone; a size never passes for a file again)."""
    key = rec["fileUrl"]
    src = RangeSource(url=hub_read_url(registry, key, token))
    memo_key = f"{key}|{rec.get('updatedAt')}|{src.size}|{'full' if full else 'json'}"
    if memo_key in memo:
        return memo[memo_key]
    head = src.read(0, 4)
    record_size = int(rec.get("fileSize") or 0)
    if head != b"PK\x03\x04" or (record_size and record_size != src.size):
        # The object the store serves is not the container the record describes: a zero-filled
        # or truncated upload (HAT-S-x4: 55.65 MB of zeros against a 54.65 MB record; granite-3.1:
        # a 2.52 GB object against a 2.79 GB record whose first bytes are not a zip header,
        # 2026-09-21). Said as its own verdict, never as a read error.
        raise HubObjectCorrupt(f"object {src.size} bytes (record {record_size}), first bytes {head!r}")
    members = zip_members(src)
    out = {"object_size": src.size, "record_size": int(rec.get("fileSize") or 0), "members": {}, "created_at": None,
           "n_members": len(members)}
    for name, m in members.items():
        if full or interesting(name):
            data = zip_member_bytes(src, m)
            out["members"][name] = sha_bytes(data)
            if name == "manifest.json":
                try:
                    out["created_at"] = json.loads(data).get("created_at")
                except Exception:  # noqa: BLE001
                    out["created_at"] = "?"
    out["bytes_read"] = src.bytes_read
    memo[memo_key] = out
    memo_path.write_text(json.dumps(memo, indent=1, sort_keys=True))
    return out


# ----------------------------------------------------------------------------- verdicts
def parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s or s == "?":
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def verdict(cache: dict, hub: Optional[dict], rec: Optional[dict]) -> dict:
    if rec is None or hub is None:
        return {"verdict": "NOT_ON_HUB", "differs": [], "same": []}
    graphs_c = {k: v for k, v in cache["members"].items() if k.endswith("graph.json") or k == "topology.json"}
    graphs_h = {k: v for k, v in hub["members"].items() if k.endswith("graph.json") or k == "topology.json"}
    differs = sorted(k for k in set(graphs_c) | set(graphs_h) if graphs_c.get(k) != graphs_h.get(k))
    same = sorted(k for k in graphs_c if graphs_c.get(k) == graphs_h.get(k))
    if not differs:
        return {"verdict": "IDENTICAL", "differs": [], "same": same}
    tc, th = parse_ts(cache.get("created_at")), parse_ts(hub.get("created_at"))
    if tc and th:
        v = "CACHE_NEWER" if tc > th else ("HUB_NEWER" if th > tc else "DIFFERS_SAME_BUILD_TIME")
    else:
        v = "DIFFERS_BUILD_TIME_UNKNOWN"
    return {"verdict": v, "differs": differs, "same": same}


def short(h: Optional[str]) -> str:
    return (h or "—")[:8]


def write_markdown(rows: List[dict], path: Path, registry_host: str, when: str, verified: dict) -> None:
    lines = [
        "# The shared cache against the hub, container by container",
        "",
        f"Generated by `tools/hub_cache_diff.py` on {when} from the rack that holds the shared cache, reading the hub at "
        f"`{registry_host}` by HTTP Range (central directory + JSON members only; no container was downloaded). "
        "Hashes are sha256 of the member bytes, first 8 hex digits; `topology` is `topology.json`, every other column is "
        "`components/<name>/graph.json`. Build times are `manifest.json`'s `created_at` on each side. "
        "Re-run the tool rather than editing this page.",
        "",
        "Verdicts: **IDENTICAL** (every graph and the topology hash equal) · **CACHE_NEWER** (they differ and the cache's build is later: "
        "a publication candidate once verified) · **HUB_NEWER** (they differ and the hub's build is later: the cache is behind) · "
        "**NOT_ON_HUB** (no hub entry under any known slug) · **HUB_RECORD_MISSING** (the map names a slug the hub does not answer) · "
        "**HUB_OBJECT_CORRUPT** (the bytes the store serves are not the recorded container: wrong length or no zip header — a verified local build must replace it).",
        "",
        "| container (cache) | hub slug | verdict | cache built | hub built | hub updated | differing members (cache → hub) | verified |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        diffs = "; ".join(f"{k.replace('components/', '').replace('/graph.json', '')}: {short(r['cache']['members'].get(k))} → {short((r.get('hub') or {}).get('members', {}).get(k))}"
                          for k in r["differs"]) or "—"
        lines.append(f"| {r['name']} | {r.get('slug') or '—'} | {r['verdict']} | {(r['cache'].get('created_at') or '?')[:19]} | "
                     f"{((r.get('hub') or {}).get('created_at') or '?')[:19]} | {((r.get('rec') or {}).get('updatedAt') or '?')[:19]} | {diffs} | "
                     f"{verified.get(r['name'], '—')} |")
    lines += ["", "## Every hash, for the record", ""]
    for r in rows:
        lines.append(f"### {r['name']}" + (f" ({r['slug']})" if r.get("slug") else ""))
        lines.append("")
        lines.append("| member | cache | hub |")
        lines.append("|---|---|---|")
        names = sorted(set(r["cache"]["members"]) | set((r.get("hub") or {}).get("members", {})))
        for k in names:
            lines.append(f"| {k} | {r['cache']['members'].get(k, '—')} | {(r.get('hub') or {}).get('members', {}).get(k, '—')} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- publish
def is_private_host(url: str) -> bool:
    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    parts = host.split(".")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return False
    a, b = int(parts[0]), int(parts[1])
    return a == 10 or (a == 192 and b == 168) or (a == 172 and 16 <= b <= 31)


def find_nbx(name: str, builds_root: Path) -> Optional[Path]:
    hits = sorted(builds_root.glob(f"*/{name}/model.nbx"))
    return hits[0] if hits else None


def repack_from_cache(cdir: Path, dest: Path) -> Path:
    """The cache directory zipped back into a `.nbx` (stored, zip64), member bytes identical to the cache's files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".nbx.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in sorted(x for x in cdir.rglob("*") if x.is_file()):
            zf.write(p, str(p.relative_to(cdir)))
    os.replace(tmp, dest)
    return dest


def publish_one(name: str, slug: Optional[str], nbx: Path, mbps: float, logdir: Path) -> int:
    if slug:
        org, mname = slug.split("/", 1)
        cmd = [TOOLCHAIN_PY, str(FORGE), "replace", "--org", org, "--name", mname, str(nbx)]
    else:
        new = (json.loads(NEW_ENTRIES.read_text()) if NEW_ENTRIES.exists() else {}).get(name)
        if not new:
            log(f"{name}: publish REFUSED — no hub entry and no written new-entry line in {NEW_ENTRIES.name}")
            return 2
        cmd = [TOOLCHAIN_PY, str(FORGE), "publish", str(nbx), "--org", new["org"], "--name", new["name"], "--category", new["category"],
               "--description", new["description"], "--tags", new["tags"], "--license", new["license"]]
    if mbps > 0:
        cmd += ["--max-write-mbps", str(mbps)]
    logfile = logdir / f"publish_{name}.log"
    log(f"{name}: publishing {nbx} → {slug or 'new entry'} (log {logfile})")
    return _publish_patiently(name, cmd, logfile)


#: A write is retried this many times through a REFUSAL, with this ceiling on the wait.
#: The reader beside it already waits 5 + 15 + 45 + 120 + 120 s; the writer waited not at
#: all — one `subprocess.TimeoutExpired` and the object was abandoned, which is how a
#: publish log came to end mid-upload at 15 % with no second attempt (2026-09-22).
PUBLISH_ATTEMPTS = 6
PUBLISH_MAX_BACKOFF_S = 300
PUBLISH_SPACING_S = 20.0          # between objects: the store takes its drive offline in bursts
PUBLISH_TIMEOUT_S = 7200

#: What a TEMPORARY refusal looks like coming back from the store. The owner's decision
#: (2026-09-22): the store is accepted as it is — the zvol behind 10.0.0.36 sits on four
#: Micron 5200s with multi-second latency and takes its drive offline for 10 to 30 s at a
#: time. None of these is a failure of the object or of this pass; they are the store
#: breathing, and counting one as a failure is what made a healthy container read as unpublished.
_TEMPORARY = re.compile(
    r"SlowDown|503|429|Retry-After|ServiceUnavailable|InternalError|"
    r"timed out|TimeoutExpired|Connection (reset|aborted|refused)|"
    r"RequestTimeout|write quorum|drive is offline|EOF occurred",
    re.I)


def _probe_patiently(probe, name: str) -> int:
    """The store's write probe, retried through refusals.

    A single non-200 used to DEFER the whole object. On this store that is the wrong reading:
    it takes its drive offline for 10 to 30 s at a time, so a 503 says "not this second", not
    "not this container". Deferring on the first one is precisely counting a temporary
    refusal as a failure, and it is why healthy containers sat unpublished.
    """
    for attempt in range(1, PUBLISH_ATTEMPTS + 1):
        try:
            code = int(probe())
        except Exception as exc:                          # noqa: BLE001 — a refused probe is a reading
            code, exc_name = 0, type(exc).__name__
            log(f"{name}: write probe raised {exc_name} (attempt {attempt}/{PUBLISH_ATTEMPTS})")
        if code == 200:
            if attempt > 1:
                log(f"{name}: the store took a write on attempt {attempt}")
            return 200
        if attempt == PUBLISH_ATTEMPTS:
            return code
        wait = min(PUBLISH_MAX_BACKOFF_S, 15 * (2 ** (attempt - 1)))
        log(f"{name}: write probe answers {code} (attempt {attempt}/{PUBLISH_ATTEMPTS}); "
            f"waiting {wait:.0f} s — a refusal is not a failure")
        time.sleep(wait)
    return 0


def _publish_patiently(name: str, cmd: list, logfile: Path) -> int:
    """Run one publish, retrying through temporary refusals rather than counting them.

    Three things the previous shape did not do, each named because each cost something:
    it never retried (a single timeout abandoned a 21.9 GB upload at 15 %); it could not
    tell a refusal from a failure (so a store that breathes read as a broken container);
    and it left no pause between objects (the store's stall is not per-object, so the next
    write walked straight into the same offline window).
    """
    waited = 0.0
    for attempt in range(1, PUBLISH_ATTEMPTS + 1):
        with open(logfile, "a") as fh:
            fh.write(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempt}/{PUBLISH_ATTEMPTS} "
                     f"{shlex.join(cmd[2:])}\n")
            fh.flush()
            try:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                     cwd=str(REPO / "forge"), timeout=PUBLISH_TIMEOUT_S)
                why = ""
            except subprocess.TimeoutExpired:
                # The word matters: `_TEMPORARY` reads this string, and a timeout on a
                # store that stalls for 10-30 s at a time IS the temporary case — it is
                # the exact shape that abandoned a 21.9 GB upload at 15 %.
                rc, why = 124, f"TimeoutExpired: no completion in {PUBLISH_TIMEOUT_S} s"
            except OSError as exc:                       # the process itself could not run
                rc, why = 125, f"{type(exc).__name__}: {exc}"
        if rc == 0:
            log(f"{name}: publish rc=0 on attempt {attempt}" + (f" after {waited:.0f} s of waiting" if waited else ""))
            return 0
        tail = ""
        try:                                             # read the store's own words, not a guess
            tail = logfile.read_text(errors="replace")[-4000:]
        except OSError:
            pass
        temporary = bool(_TEMPORARY.search(why or "") or _TEMPORARY.search(tail))
        if not temporary or attempt == PUBLISH_ATTEMPTS:
            log(f"{name}: publish rc={rc} on attempt {attempt}"
                + (f" ({why})" if why else "")
                + ("; the refusal looks temporary but the attempts are spent"
                   if temporary else "; not a temporary refusal — reported as a failure"))
            return rc
        wait = min(PUBLISH_MAX_BACKOFF_S, 15 * (2 ** (attempt - 1)))
        waited += wait
        log(f"{name}: the store refused this write (attempt {attempt}/{PUBLISH_ATTEMPTS}"
            + (f", {why}" if why else "") + f"); waiting {wait:.0f} s — a refusal is not a failure")
        time.sleep(wait)
    return 1


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--hub-map", type=Path, default=HUB_MAP)
    ap.add_argument("--only", nargs="*", help="container names to diff (default: every container with a manifest)")
    ap.add_argument("--out", type=Path, required=False, help="directory for diff.json, hub_memo.json and the publish logs")
    ap.add_argument("--markdown", type=Path, help="the tracked page to write (the other machine reads it)")
    ap.add_argument("--verified", type=Path, help="JSON: container name -> pointer to its judged proof; only these may be published")
    ap.add_argument("--publish", action="store_true", help="publish every CACHE_NEWER + verified container through the toolchain")
    ap.add_argument("--builds-root", type=Path, default=Path("/home/mlops/nbx/builds"))
    ap.add_argument("--stage", type=Path, default=Path("/home/mlops/nbx/stage/hub_diff"), help="where a cache container is re-packed when no staged .nbx exists")
    ap.add_argument("--upload-mbps", type=float, default=10.0)
    ap.add_argument("--self-test", type=Path, help="run the range parser on a local .nbx and compare with zipfile")
    ap.add_argument("--verify-all", action="store_true", help="read EVERY member of every hub object and verify it by checksum against the cache container (its source)")
    ap.add_argument("--hide-corrupt", action="store_true", help="with --verify-all or --publish: set a hub object that fails its checksum to hidden, with the reason")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.self_test)
    if not args.out:
        ap.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)

    repo_env.load()
    repo_env.require("NEUROBRIX_API_TOKEN")
    registry = os.environ.get("NEUROBRIX_REGISTRY")
    if not registry:
        print("NEUROBRIX_REGISTRY is not set: source nbx/env.sh (the internal entry point) first", file=sys.stderr); return 2
    token = os.environ["NEUROBRIX_API_TOKEN"]
    hub_map = json.loads(args.hub_map.read_text()) if args.hub_map.exists() else {}
    hub_map = {k: v for k, v in hub_map.items() if not k.startswith("_")}
    verified = json.loads(args.verified.read_text()) if args.verified else {}
    memo_path = args.out / "hub_memo.json"
    memo = json.loads(memo_path.read_text()) if memo_path.exists() else {}

    names = args.only or sorted(p.name for p in args.cache.iterdir() if (p / "manifest.json").exists())
    rows: List[dict] = []
    for name in names:
        cdir = args.cache / name
        cache = cache_index(cdir)
        slug = hub_map.get(name)
        rec = hub = None
        row = {"name": name, "slug": slug, "cache": cache}
        try:
            if slug:
                rec = hub_record(registry, slug)
                if rec is None:
                    row.update({"verdict": "HUB_RECORD_MISSING", "differs": [], "same": []})
                else:
                    hub = hub_index(registry, rec, token, memo, memo_path, full=args.verify_all)
                    row.update(verdict(cache, hub, rec))
                    if args.verify_all and row["verdict"] == "IDENTICAL":
                        bad = verify_members(cache_all_files(cdir), hub["members"])
                        if bad:
                            raise HubObjectCorrupt(f"{len(bad)} member(s) differ from the source by checksum: {', '.join(bad[:4])}")
                        row["checksum"] = f"every member verified ({len(hub['members'])})"
            else:
                row.update(verdict(cache, None, None))
        except HubObjectCorrupt as exc:
            row.update({"verdict": f"HUB_OBJECT_CORRUPT: {exc}", "differs": [], "same": []})
            if args.hide_corrupt and slug and (rec or {}).get("visibility") != "HIDDEN":
                hide_on_hub(slug, f"checksum verification failed on {datetime.now(timezone.utc):%Y-%m-%d}: {exc}", args.out)
                row["hidden"] = True
        except Exception as exc:  # noqa: BLE001 — named per container, the diff goes on
            row.update({"verdict": f"ERROR: {type(exc).__name__}: {str(exc)[:160]}", "differs": [], "same": []})
        row["rec"] = {k: rec.get(k) for k in ("updatedAt", "fileSize", "fileUrl")} if rec else None
        row["hub"] = hub
        rows.append(row)
        log(f"{name:45s} {row['verdict']:28s} cache {(cache.get('created_at') or '?')[:19]}  hub {((hub or {}).get('created_at') or '?')[:19]}"
            + (f"  differs: {', '.join(k.replace('components/', '').replace('/graph.json', '') for k in row['differs'])}" if row["differs"] else ""))

    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (args.out / "diff.json").write_text(json.dumps({"generated": when, "registry": registry, "rows": rows}, indent=1, sort_keys=True))
    if args.markdown:
        write_markdown(rows, args.markdown, registry.split("://", 1)[-1], when, verified)
        log(f"page written: {args.markdown}")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["verdict"].split(":")[0]] = counts.get(r["verdict"].split(":")[0], 0) + 1
    log("verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    candidates = [r for r in rows if r["verdict"] == "CACHE_NEWER" or r["verdict"].startswith("HUB_OBJECT_CORRUPT")]
    for r in candidates:
        log(f"candidate: {r['name']} → {r['slug']} verified={'yes: ' + verified[r['name']] if r['name'] in verified else 'NO (not published)'}")
    if not args.publish:
        return 0

    if not is_private_host(registry):
        print(f"publish refused: NEUROBRIX_REGISTRY={registry} is not a private address — publishing goes through the internal entry point", file=sys.stderr)
        return 2
    sys.path.insert(0, str(REPO / "tools"))
    import retrace_zoo as Z  # the toolchain's health, probe and reader census
    health = Z.hub_store_health()
    if health != 200:
        log(f"publish deferred: the store's cluster health answers {health}"); return 3
    published, deferred = [], []
    for r in candidates:
        name = r["name"]
        if name not in verified:
            continue
        readers = Z.export_readers()
        if readers:
            log(f"{name}: publish deferred — the export is read by {', '.join(readers)}"); deferred.append(name); continue
        nbx = find_nbx(name, args.builds_root)
        if nbx is None:
            nbx = repack_from_cache(args.cache / name, args.stage / name / "model.nbx")
            src = RangeSource(path=nbx); mem = zip_members(src)
            for k, h in r["cache"]["members"].items():
                if sha_bytes(zip_member_bytes(src, mem[k])) != h:
                    log(f"{name}: the re-packed container's {k} differs from the cache's — not published"); nbx = None; break
            if nbx is None:
                deferred.append(name); continue
            log(f"{name}: re-packed from the cache into {nbx} ({nbx.stat().st_size} bytes)")
        if r["slug"]:
            org, mname = r["slug"].split("/", 1)
            probe = _probe_patiently(
                lambda: Z.hub_store_write_probe(
                    org, mname, token, registry=registry,
                    nbytes=min(Z.PROBE_BYTES, max(5, nbx.stat().st_size)), mbps=args.upload_mbps),
                name)
            if probe != 200:
                log(f"{name}: publish deferred — the store's write probe still answers {probe} "
                    f"after {PUBLISH_ATTEMPTS} patient attempts")
                deferred.append(name); continue
        if published:
            # The store stalls in bursts of 10-30 s and the stall is not per-object, so the
            # next write would walk straight into the same offline window. Space them.
            log(f"{name}: spacing {PUBLISH_SPACING_S:.0f} s behind the previous publication")
            time.sleep(PUBLISH_SPACING_S)
        rc = publish_one(name, r["slug"], nbx, args.upload_mbps, args.out)
        if rc == 0:
            # The bytes the hub now serves are read back whole and verified against the file
            # that was uploaded, member by member: a publication counts only once verified.
            slug = r["slug"] or (json.loads(NEW_ENTRIES.read_text()).get(name) or {})
            slug = slug if isinstance(slug, str) else f"{slug.get('org')}/{slug.get('name')}"
            rec2 = hub_record(registry, slug)
            local = {i.filename: sha_bytes(zipfile.ZipFile(nbx).read(i)) for i in zipfile.ZipFile(nbx).infolist()}
            try:
                hub2 = hub_index(registry, rec2, token, memo, memo_path, full=True)
                bad = verify_members(local, hub2["members"])
            except HubObjectCorrupt as exc:
                bad = [f"object: {exc}"]
            if bad:
                log(f"{name}: the hub serves bytes that are NOT the uploaded container ({len(bad)} member(s): {', '.join(bad[:4])}) — hidden")
                if args.hide_corrupt:
                    hide_on_hub(slug, f"the uploaded bytes failed their checksum on read-back {datetime.now(timezone.utc):%Y-%m-%d}", args.out)
                rc = 5
            else:
                log(f"{name}: read back and verified by checksum, {len(local)} member(s)")
        (published if rc == 0 else deferred).append(name)
    log(f"published: {published or '—'}; deferred: {deferred or '—'}")
    return 0 if not deferred else 4


if __name__ == "__main__":
    sys.exit(main())
