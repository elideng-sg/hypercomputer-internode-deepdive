# Design: g3doc port of the internode-deepdive guide

**Date:** 2026-07-27
**Status:** approved

## Goal

Publish this guide as internal g3doc documentation, without giving up the GitHub
repo as a working, runnable lab environment.

The deliverable is a self-contained `g3doc/` tree inside this repo. A reader with
a google3 client copies that one directory into their client and mails a single
CL. Nothing in the conversion requires google3 access, because this machine has
none (see Constraints).

## Constraints

This machine is a GCE VM (`e2-standard-32`, project `hdlab-elideng`), Ubuntu
24.04 — not a corp workstation. `g4`, `g4d`, `fig`, `hg`, `citc`, `blaze`,
`prodaccess` and `gcert` are all absent and are not installable from public apt;
`/google` does not exist. `source.corp.google.com` resolves but redirects to
`login.corp.google.com`, which is the public login page, not access.

Two consequences that shape the whole design:

1. **The final CL is mailed elsewhere.** Everything up to that point — tree
   layout, diagram rendering, link rewriting, verification — happens here and
   lands in a PR.
2. **The g3doc renderer cannot be tested here.** So the design does not depend
   on renderer features. Diagrams are pre-rendered to SVG rather than trusting
   inline Mermaid support.

## Source inventory

Measured against `main` at `bb5a06a`.

| Source | Count | Disposition |
| :--- | :--- | :--- |
| `docs/**.md` guide chapters + toolkit (excl. `superpowers/`) | 31 | Port |
| `labs/lab-NN-*/README.md` | 21 | Port as lab pages |
| `reference/*.md` | 7 | Port as appendix |
| `assets/**` capture files | 174 | Port verbatim |
| Mermaid blocks across all `.md` | 99 | Pre-render to SVG |
| `docs/superpowers/{specs,plans}` | 6 | **Not ported** |
| `labs/**/*.{sh,py,yaml,json}`, `manifests/`, `scripts/` | ~50 | Stay in place as code |

Mermaid diagram types in use: `flowchart` (58), `graph` (28), `xychart-beta`
(8), `sequenceDiagram` (3), `pie` (1), `gantt` (1).

## Target layout

```
g3doc/
├─ index.md                        ← from README.md, status/CI noise trimmed
├─ _toc.yaml                       ← nav (see Open question)
├─ overview.md                     ← from docs/00-guide-overview.md
├─ part1-single-node/
│  ├─ index.md                     ← NEW part intro + child table
│  └─ 4 chapter pages
├─ part2-inter-node/               ← index.md + 2 chapters + scaling.md
├─ part3-clustering-execution/     ← index.md + 4
├─ part4-platform-reference-arch/  ← index.md + 4
├─ part5-operations-diagnostics/   ← index.md + 5
├─ part6-architecture-gcp-integration/ ← index.md + 4
├─ toolkit/                        ← index.md + T1–T6
├─ reference/                      ← index.md + 7 appendix pages
├─ labs/                           ← index.md + lab-01…lab-21/index.md
├─ images/                         ← 99 rendered SVG + 6 existing PNG/SVG
└─ assets/                         ← 174 capture files
```

Expected page total: **69** markdown pages (24 chapters + 6 toolkit + 7
reference + 21 labs + 9 directory `index.md` + root `index.md` + `overview.md`).

### Two structural changes from the repo shape

**Numeric filename prefixes drop.** `01-gpu-microarchitecture.md` becomes
`gpu-microarchitecture.md`. Ordering is `_toc.yaml`'s job. The `01:` / `15 —`
prefixes stay in the H1 titles, so the guide's pervasive "doc-06" prose
references still read correctly.

**`docs/15-scaling-shape-of-the-cliff.md` moves into part2** as `scaling.md`.
It sits orphaned at top level today, but `00-guide-overview.md` describes it as
Part II's scaling bridge, so the nav should say so.

### Labs: one copy of every script

Each lab README becomes `g3doc/labs/lab-NN/index.md`. Runnable files
(`run.sh`, `*.py`, manifests) **stay in `labs/`** and are linked by tree path,
not embedded. Lab pages already inline the snippets that matter, so nothing is
lost and there is exactly one copy of each script.

## Diagram pipeline

Verified working on this machine before committing to the approach:
`@mermaid-js/mermaid-cli` installed with `PUPPETEER_SKIP_DOWNLOAD=true` against
system Chrome at `/usr/bin/google-chrome`, with
`--no-sandbox --disable-dev-shm-usage`. Test renders of `xychart-beta` (the
riskiest type) and a `classDef`-styled `flowchart` both produced non-empty SVG
with title text present and `#1a73e8` fills intact.

`tools/render-diagrams.sh` walks each source `.md`, extracts every ` ```mermaid `
block, renders it to `g3doc/images/<page-slug>-<n>.svg`, and replaces the block
with an image reference.

Every diagram in this repo is already preceded by an italic `*Figure: …*`
caption. That caption becomes the alt text, so the SVGs are described rather
than bare:

```markdown
![the 16-rank ring — 14 NVLink hops stay inside each node; the 2 ring hops
that cross the node boundary (red) are TCP over gVNIC](../images/nccl-collectives-1.svg)
```

The script is idempotent and re-runnable, so a changed diagram is a re-render,
not a hand edit. It reads from the original `docs/`/`labs/` sources, which stay
Mermaid — that remains the editable source of truth for diagrams.

## Link rewriting

~250 relative links need attention, in three classes:

1. **Directory links** (~40, e.g. `](../lab-06-2node-nccl-collectives/)`) —
   g3doc resolves against the source tree and will not infer an index. Each gets
   an explicit `index.md` target.
2. **Cross-tree hops** — `docs/` ↔ `labs/` ↔ `reference/` links re-resolve for
   the new directory depth, plus the de-numbered filenames.
3. **Two link bugs already on `main`**, fixed rather than faithfully ported:
   - `../part5-operations-diagnostics/20-perf-monitoring-day2.md` (9 refs) —
     real file is `20-performance-monitoring-day2-ops.md`.
   - the `16-diagnostic-method.md` refs from lab pages that resolve to the wrong
     depth.

A link-check step asserts every relative target in `g3doc/` resolves to a real
file. This is the build gate: a broken link fails the port.

## Out of scope

`docs/superpowers/specs/` and `docs/superpowers/plans/` (6 files, ~130KB) are
artifacts about *building* the guide, not the guide itself. They stay
GitHub-only.

No prose is rewritten. This is a structural port: content, measurements, and the
measured-vs-reference honesty framing carry over unchanged. Internal
identifiers already in the prose (`hdlab-elideng`, cluster and zone names, GKE
versions) stay as literals; converting them to `go/` links is a separate,
later call.

## Verification

Before the port is called done:

- All 99 Mermaid blocks produced a non-empty SVG (count asserted, not spot-checked).
- Link checker reports zero unresolved relative targets under `g3doc/`.
- Page count matches the expected 69.
- No ` ```mermaid ` block remains in `g3doc/`.
- Every `assets/` reference from a g3doc page resolves to a ported file.

## Open question for the reviewer

**Nav file name.** This design assumes g3doc nav via `_toc.yaml`. Some
setups use `_book.yaml`. This cannot be checked from here — glance at a
neighboring `g3doc/` directory before mailing the CL. It is a one-file fix
either way, and is called out in the handoff notes.

## Deliverable

A draft PR containing:

- the `g3doc/` tree
- `tools/render-diagrams.sh` and the link checker
- `g3doc-port.md` handoff notes: the `_toc.yaml` assumption, and the steps to
  copy the tree into a client
