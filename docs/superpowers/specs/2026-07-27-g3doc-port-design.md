# Design: g3doc port to `gpu-ai-infra-field-guide`

**Date:** 2026-07-27
**Status:** approved

## Goal

Publish this guide as internal g3doc documentation, without giving up this
repo as a working, runnable lab environment.

The port lands in a **new, separate GitHub repo — `gpu-ai-infra-field-guide`
(private)** — not in this one. A reader with a google3 client clones it, copies
`g3doc/` into their client, and mails a single CL. Nothing in the conversion
requires google3 access, because this machine has none (see Constraints).

### Why a separate repo

Two reasons, both raised by the repo owner:

1. **Avoid conflict.** This repo is actively developed across many branches
   (`feat/*`, `docs/*`); dropping a large generated `g3doc/` tree into it would
   collide with in-flight work and make every future content branch also a
   g3doc branch.
2. **The name no longer fits.** `hypercomputer-internode-deepdive` describes
   Part II alone. The content now spans GPU microarchitecture, intra-node
   NVLink, the inter-node fabric, clustering/JobSet/Kueue, NVIDIA platform
   reference architectures, ops/diagnostics, and GCP architecture integration,
   plus a T1–T6 tool layer and 21 live labs. `gpu-ai-infra-field-guide` covers
   that span, names no single layer or vendor, and has room for further parts.

### Repo split of responsibility

| Repo | Owns |
| :--- | :--- |
| `hypercomputer-internode-deepdive` (existing, public) | The runnable lab environment: `labs/` scripts, `manifests/`, `scripts/`, and the authored `docs/` prose. Stays the place labs are developed and re-run. |
| `gpu-ai-infra-field-guide` (new, private) | The publishable guide: the rendered `g3doc/` tree, the Mermaid-bearing `src/` it was rendered from, the `tools/` that do it, and `assets/`. Self-contained — can regenerate itself without the other repo. |

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
| Mermaid blocks across all `.md` | 147 | Pre-render to SVG |
| `docs/superpowers/{specs,plans}` | 6 | **Not ported** |
| `labs/**/*.{sh,py,yaml,json}`, `manifests/`, `scripts/` | ~50 | **Not ported** — stay in the lab repo as runnable code |

Mermaid diagram types in use: `flowchart` (106), `graph` (28), `xychart-beta`
(8), `sequenceDiagram` (3), `pie` (1), `gantt` (1).

> **Count correction.** An earlier draft of this spec said 99 diagrams. That
> number came from a stale local `main` ref (`ca76dbe`) that predated the PR #16
> architecture-overview / step-flow diagram fan-out. Counted against
> `origin/main` (`bb5a06a`) — which is what this work branches from — the real
> total is **147**. All verification assertions use 147.

## Target layout

The new repo root:

```
gpu-ai-infra-field-guide/
├─ README.md          ← what this repo is, and the copy-into-client steps
├─ g3doc/             ← THE DELIVERABLE: copy this dir into google3
├─ src/               ← Mermaid-bearing markdown, imported from the lab repo
├─ tools/             ← render-diagrams.sh, linkcheck.sh
├─ assets/            ← 174 capture files
└─ VERIFICATION.md    ← provenance log, carried over
```

`src/` mirrors the lab repo's authored prose (`docs/`, `labs/*/README.md`,
`reference/`) and stays Mermaid — it is the editable source of truth for
diagrams. `tools/render-diagrams.sh` reads `src/` and writes `g3doc/`. That is
what makes this repo self-contained: a diagram edit is a `src/` edit plus a
re-render, with no dependency on the lab repo.

Content flows one way: prose is authored in the lab repo, imported into `src/`,
then rendered into `g3doc/`. Re-importing later is a copy over `src/` followed
by a re-render — deliberately a simple, repeatable step rather than a submodule
or subtree, both of which are more machinery than a periodic refresh warrants.

And `g3doc/` itself:

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
├─ images/                         ← 147 rendered SVG + 6 existing PNG/SVG
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
(`run.sh`, `*.py`, manifests) are **not copied** — they stay in the lab repo,
which remains the place labs are run. Lab pages reference them by name and
already inline the snippets that matter, so there is exactly one copy of every
script and no risk of a stale duplicate.

Note the tradeoff this accepts: a lab page's reference to `run.sh` is prose, not
a resolvable link, so the link checker cannot verify it. Cross-repo links would
be checkable but would break if the lab repo is renamed or made private. Given
the guide's readers are internal and the scripts are for re-running labs rather
than reading inline, one authoritative copy is worth the unverifiable reference.

## Diagram pipeline

Verified working on this machine before committing to the approach:
`@mermaid-js/mermaid-cli` installed with `PUPPETEER_SKIP_DOWNLOAD=true` against
system Chrome at `/usr/bin/google-chrome`, with
`--no-sandbox --disable-dev-shm-usage`. Test renders of `xychart-beta` (the
riskiest type) and a `classDef`-styled `flowchart` both produced non-empty SVG
with title text present and `#1a73e8` fills intact.

`tools/render-diagrams.sh` walks each `src/**.md`, extracts every ` ```mermaid `
block, renders it to `g3doc/images/<page-slug>-<n>.svg`, and replaces the block
with an image reference in the corresponding `g3doc/` page.

Alt text comes from the italic caption line that precedes almost every diagram.
Measured distribution of the 147 blocks:

| Preceding line | Count | Alt-text source |
| :--- | :--- | :--- |
| `*Figure: …*` | 98 | the caption, `Figure:` prefix stripped |
| other italic caption (`*Where this fits: …*`, `*The real place …*`) | 46 | the caption verbatim |
| no italic caption | 3 | the nearest preceding heading + diagram type |

The three uncaptioned blocks are `16-diagnostic-method.md:83`,
`18-internode-comms-troubleshooting.md:77`, and
`lab-19-storage-data-path/README.md:19`. The renderer must not assume a caption
exists — it falls back to the heading rather than emitting empty alt text.

```markdown
![the 16-rank ring — 14 NVLink hops stay inside each node; the 2 ring hops
that cross the node boundary (red) are TCP over gVNIC](../images/nccl-collectives-1.svg)
```

The script is idempotent and re-runnable, so a changed diagram is a re-render,
not a hand edit. `src/` stays Mermaid and is the editable source of truth;
`g3doc/` is generated output. Since `g3doc/` is what gets copied into a client,
it is committed rather than gitignored — the CL author should not need to run
Chrome to produce the deliverable.

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
artifacts about *building* the guide, not the guide itself. They stay in the lab
repo only — including this spec.

`labs/**` scripts, `manifests/`, and `scripts/` are not copied (see *Labs*
above). The lab repo keeps them and keeps its current name; nothing about this
port renames or restructures it.

No prose is rewritten. This is a structural port: content, measurements, and the
measured-vs-reference honesty framing carry over unchanged. Internal
identifiers already in the prose (`hdlab-elideng`, cluster and zone names, GKE
versions) stay as literals; converting them to `go/` links is a separate,
later call.

## Verification

Before the port is called done:

- All 147 Mermaid blocks produced a non-empty SVG (count asserted, not spot-checked).
- No rendered image has empty alt text.
- Link checker reports zero unresolved relative targets under `g3doc/`.
- Page count matches the expected 69.
- No ` ```mermaid ` block remains in `g3doc/`.
- Every `assets/` reference from a g3doc page resolves to a ported file.
- The new repo is **private**, and `git log` shows the tree pushed to it.
- A fresh clone of the new repo can re-run `tools/render-diagrams.sh` and
  reproduce `g3doc/` — i.e. it really is self-contained.

## Open question for the reviewer

**Nav file name.** This design assumes g3doc nav via `_toc.yaml`. Some
setups use `_book.yaml`. This cannot be checked from here — glance at a
neighboring `g3doc/` directory before mailing the CL. It is a one-file fix
either way, and is called out in the handoff notes.

## Deliverable

**A new private GitHub repo, `elideng-sg/gpu-ai-infra-field-guide`**, created via
`gh` (v2.63.2, installed to `~/.local/bin` during this work; authenticated as
`elideng-sg`), containing:

- `g3doc/` — the 69-page rendered tree with `_toc.yaml`
- `src/` — the Mermaid-bearing sources
- `tools/` — `render-diagrams.sh` and `linkcheck.sh`
- `assets/` — the 174 captures
- `README.md` — what the repo is, the `_toc.yaml` assumption, and the
  copy-into-client steps
- `VERIFICATION.md` — the provenance log, carried over

Work happens on a branch and lands via PR in the new repo, so the tree is
reviewable before it becomes `main`.

This repo (`hypercomputer-internode-deepdive`) receives **only this spec** — no
`g3doc/` tree, no structural change, nothing that conflicts with the in-flight
`feat/*` and `docs/*` branches.
