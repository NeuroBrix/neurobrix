#!/usr/bin/env python3
"""Render the catalogue ledger as a page, from the same records as the report.

The markdown report (`certified_catalogue_report.py`) is the artefact the repo
keeps; this is the same data laid out to be read by someone who is not going to
open a terminal. Both read the campaign records directly, so neither can drift
from the other — the rows here are GENERATED, never transcribed, because a
hand-copied table is a table that goes stale silently.

Usage:  python tools/certified_catalogue_page.py /path/to/out.html
"""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_catalogue_report as R

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("volta_ledger.html")
REPO = Path(__file__).resolve().parents[1]
snap = REPO / "validation_outputs" / "certified_catalogue_2026_09_10" / "hub_snapshot.txt"
hub = R._hub(snap)
cells = R._campaign_cells(Path("/home/mlops/nbx/campaigns"))
total, per_file = R._certified_totals()
by_name = {n.lower(): c for n, c in cells.items()}
for r in hub:
    slug = r["slug"].lower()
    r["cell"] = by_name.get(slug) or by_name.get(R.ALIASES.get(slug, ""))
    c = r["cell"]
    r["known"] = c["cost_s"] if (c and not c.get("failed")) else None
hub.sort(key=lambda r: (r["known"] is None, r["known"] or 0, r["gb"]))

def nm(): return '<span class="nm" title="not measured — deliberately empty">—</span>'


def _blocker(r):
    """The debt that explains why this row has no measurement, if one is
    written down. A cell that ran and crashed, and a model with a known blocker
    from an earlier flight, both say so — "not measured" is true for them and
    incomplete, and the incomplete half is the part someone acts on."""
    slug = r["slug"].lower()
    for name, debt in R.FAILED_CELLS.items():
        if name.lower() in (slug, R.ALIASES.get(slug, "")):
            return debt
    return None


rows = []
for i, r in enumerate(hub, 1):
    c = r["cell"]
    debt = _blocker(r)
    if debt:
        state = "failed" if (c and c.get("failed")) else "blocked"
        rows.append(
            f'<tr class="debt"><td class="num idx">{i}</td>'
            f'<td class="model"><code>{html.escape(r["hub"])}</code></td>'
            f'<td><span class="fam">{html.escape(r["family"].lower())}</span></td>'
            f'<td class="num quiet">{r["gb"]:.1f}</td>'
            f'<td class="debtcell" colspan="8"><b>{state}</b> '
            f'<code>{html.escape(debt)}</code></td></tr>')
        continue
    if c and not c.get("failed"):
        n = c["choice_contradicted_n"] or 0
        cells_html = [
            f'<td class="num">{c["keys"]}</td>',
            f'<td class="num">{c["certified"]}</td>',
            f'<td class="num">{c["differ"]}</td>',
            f'<td class="num quiet">{c["near_tie"]}</td>',
            f'<td class="num">{f"<b class=fnd>{n}</b>" if n else "0"}</td>',
            f'<td class="num quiet">{c["excluded"]}</td>',
            f'<td class="num gain">×{c["gain"]:.2f}</td>',
            f'<td class="num quiet">{c["cost_s"]:.0f} s</td>',
        ]
        cls = ' class="measured"'
    else:
        cells_html = [f'<td class="num">{nm()}</td>'] * 8
        cls = ''
    rows.append(
        f'<tr{cls}><td class="num idx">{i}</td>'
        f'<td class="model"><code>{html.escape(r["hub"])}</code></td>'
        f'<td><span class="fam">{html.escape(r["family"].lower())}</span></td>'
        f'<td class="num quiet">{r["gb"]:.1f}</td>' + "".join(cells_html) + '</tr>')

# The headline counts from the `*_count` fields; the cards below are the
# record's SAMPLE, capped at twenty. Summing the samples reported 52 where the
# counts say otherwise — the same defect this document was corrected for once
# already, and it recurs because a list is the thing in front of you.
findings, n_findings = [], 0
for r in hub:
    c = r["cell"]
    if not c or c.get("failed") or not c.get("choice_contradicted_n"):
        continue
    n_findings += c["choice_contradicted_n"]
    sample = c["choice_contradicted"] or []
    if c["choice_contradicted_n"] > len(sample):
        findings.append(
            f'<li class="more"><div class="f-head"><code class="f-model">'
            f'{html.escape(r["hub"])}</code><span class="f-margin">'
            f'{c["choice_contradicted_n"]} keys</span></div>'
            f'<div class="f-scale">the {len(sample)} cards beside this one are '
            f'the record\'s sample, not the whole list</div></li>')
    for f in sample:
        key = f.get("key", "")
        short = key.split("::", 1)[-1]
        kern = key.split("::", 1)[0].split(".")[-1]
        findings.append(
            f'<li><div class="f-head"><code class="f-model">{html.escape(r["hub"])}</code>'
            f'<span class="f-margin">{f.get("margin",0)*100:.0f} % margin</span></div>'
            f'<code class="f-key">{html.escape(kern)}<span class="f-tuple">{html.escape(short)}</span></code>'
            f'<div class="f-scale">{f.get("delta_ms",0)*1000:.1f} µs on a '
            f'{f.get("best_ms",0)*1000:.1f} µs kernel</div></li>')

measured = [r for r in hub if r["cell"] and not r["cell"].get("failed")]
pop = "".join(f'<li><code>{html.escape(r["hub"])}</code>'
              f'<span class="pop-k">{r["cell"]["keys"]} keys</span>'
              f'<span class="pop-g">×{r["cell"]["gain"]:.2f}</span></li>' for r in measured)
files = "".join(f'<li><code>{html.escape(n)}</code><span>{v:,}</span></li>'
                for n, v in sorted(per_file.items(), key=lambda kv: -kv[1]))

TPL = (Path(__file__).resolve().parent / "templates" / "certified_catalogue_page.html").read_text()
OUT.write_text(TPL
    .replace("{{ROWS}}", "\n".join(rows))
    .replace("{{FINDINGS}}", "\n".join(findings))
    .replace("{{POP}}", pop)
    .replace("{{FILES}}", files)
    .replace("{{TOTAL}}", f"{total:,}")
    .replace("{{NFILES}}", str(len(per_file)))
    .replace("{{NMEAS}}", str(len(measured)))
    .replace("{{NROWS}}", str(len(hub)))
    .replace("{{NUNMEAS}}", str(len(hub) - len(measured)))
    .replace("{{NFIND}}", str(n_findings)))
print(f"wrote {OUT} — {len(rows)} rows, {len(findings)} findings")
