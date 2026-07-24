# Integration Roadmap: coordinating the three enrichment tracks

**Date:** 2026-07-23 (rev 1)
**Author:** elideng-sg (with Claude Code)
**Status:** Active — read before implementing any track
**Parent design:** [`2026-07-21-hypercomputer-internode-deepdive-design.md`](2026-07-21-hypercomputer-internode-deepdive-design.md)
**Tracks coordinated:**
- **[Scaling]** [`2026-07-23-three-node-scaling-enrichment-design.md`](2026-07-23-three-node-scaling-enrichment-design.md) — labs 12–13, doc-15
- **[Ops]** [`2026-07-23-diagnostics-troubleshooting-operations-track-design.md`](2026-07-23-diagnostics-troubleshooting-operations-track-design.md) — Part V, labs 14–17, docs 16–20
- **[Arch]** [`2026-07-23-architecture-gcp-integration-track-design.md`](2026-07-23-architecture-gcp-integration-track-design.md) — Part VI, labs 18–21, docs 21–24

---

## 1. Why this doc exists

The three enrichment tracks were specced **independently** and each is individually shippable, but they **plan to edit the same existing files**. Implemented naively they would double-edit, duplicate, or conflict — and two of them have a real **cross-track build dependency**. This roadmap resolves both:

1. **Shared-file ownership** — each file touched by ≥2 tracks gets a single *owning* track; other tracks only *contribute* content through the owner's structure.
2. **Build order** — the sequence that respects the dependencies (Ops' triage hub before Scaling's fault entry; Arch's TCPX pool before Scaling's enabled re-run).

This doc does **not** re-open any track's design; it only de-conflicts their shared surface.

---

## 2. Shared-file ownership map

Files edited by more than one track. **Owner** creates/re-bases the file's structure; **contributors** add only their labelled rows/entries/links into that structure.

| File | Owner | Contributors | De-confliction rule |
| :--- | :--- | :--- | :--- |
| `README.md` | **Track 0** (this re-baseline) | Scaling, Ops, Arch | Track 0 re-bases to the 6-part / 2-cluster skeleton **once**; each track then only *appends* its lab rows + status lines. No track re-writes the structure. |
| `docs/00-guide-overview.md` | **Track 0** | Ops, Arch | Same as README: Track 0 sets the 6-part spine + cluster table + part-flow mermaid; tracks append their part/lab entries only. |
| `docs/part2-inter-node/06-nccl-collectives.md` | **Scaling** | Arch | Scaling owns the 24-GPU busbw row + curve framing; Arch adds a **single link** to lab-18 as the "enabled fabric" counterpart. No prose overlap. |
| `docs/part2-inter-node/05-nic-rdma-gpudirect.md` | **Arch** | Ops | Arch owns the TCPX/lab-18 forward-link; Ops adds the lab-15 "debug the path" cross-link. |
| `docs/part3-clustering-execution/10-observability-debugging.md` | **Ops** | Scaling | **Ops owns the fault-catalog generalization** (into doc-16). Scaling's node-loss fault becomes **one entry** contributed into that catalog, not a parallel edit. ⇒ build Ops' doc-16 framing first (see §3). |
| `labs/lab-10-observability-fleet-debug/` | **Ops** | Scaling | Same: Ops restructures lab-10 to point at Part V; Scaling adds the node-loss signature as one catalogued entry. |
| `labs/lab-06-2node-nccl-collectives/` | **Scaling** | Arch | Scaling adds the 24-GPU third point; Arch adds the "gVNIC baseline vs lab-18 TCPX" link. |
| `reference/nccl-tunables.md` | **Ops** | Scaling | Ops owns filling the stub from all lab usage; Scaling contributes the `NCCL_ALGO`/`NCCL_PROTO` tunables it varied in lab-12b. Single populated file, not two. |
| `reference/tool-cheatsheets.md` | **Ops** | — | Ops fills all four "Commands to be filled" sections. |
| `reference/xid-table.md` | **Ops** | — | Ops fills the XID catalog. |
| `reference/reference-arch-cheatsheet.md` | **Arch** | — | Arch adds GCP-integration reference patterns. |
| `VERIFICATION.md` | **append-only** | all | No owner needed — every track appends its own dated, cluster-labelled provenance rows. Chronological; no conflict. |

**Files with a single track (no coordination needed):** doc-07/08/09 + lab-07/08/09 (Scaling only); doc-15 (Scaling only); docs 16–20 + labs 14–17 + fault/grafana manifests (Ops only); docs 21–24 + labs 18–21 + tcpx/storage/serving manifests + `scripts/provision_tcpx_pool.sh` (Arch only).

**`scripts/lib_capture.sh`:** the **Scaling** track parametrizes it (`LAB_NODEPOOL`, `KUBE_CONTEXT`); Ops and Arch **reuse** it as-is. If Scaling's lab work hasn't landed when Ops/Arch start, whichever track lands first makes the parametrization change and the others rebase onto it. Owner-of-record: **Scaling**.

---

## 3. Cross-track build order

The tracks are independently *shippable*, but two dependencies make the order matter:

- **D1 — triage hub before node-loss:** Ops generalizes lab-10 into the `doc-16` triage framework; Scaling's node-loss fault (lab-13b) is meant to slot into that framework as one catalogued signature. ⇒ **Ops' doc-16 + lab-10 restructure should land before (or with) Scaling's doc-10/lab-10 edits.**
- **D2 — TCPX pool before the enabled curve:** Arch's lab-18 provisions the multi-network TCPX pool and measures gVNIC→TCPX. Scaling's 8/16/24 sweep can then be **re-run on the enabled fabric** for the "enabled curve" cross-run. ⇒ **Arch's lab-18 should land before Scaling's enabled-curve re-run** (the *baseline* gVNIC curve has no such dependency and can go first).

Recommended sequence (each step still independently reviewable/shippable):

1. **Track 0 — integration & re-baseline (this pass):** README + doc-00 to the 6-part / 2-cluster skeleton; this roadmap; parent-spec forward-pointer. *No cluster needed.*
2. **Ops quick-wins + reference backbone:** fix the lab-01 garbage-memory reading and the lab-05 `ip` inconsistency; fill `reference/xid-table.md`, `nccl-tunables.md`, `tool-cheatsheets.md`. *Cheap, high-leverage, unblocks cross-links.*
3. **Ops doc-16 triage hub + lab-15 inter-node comms** (highest-value scenario; establishes the fault-catalog structure D1 depends on).
4. **Scaling lab-12a baseline curve** (8/16/24 gVNIC) + doc-06/09 third-point + doc-15 skeleton. Parametrizes `lib_capture.sh`.
5. **Arch lab-18 TCPX** (flagship; measures the cliff closing) + doc-21. *Satisfies D2.*
6. **Scaling lab-12b/c + lab-13** (ring/tree, training scaling, gang, node-loss → slots into Ops' doc-16 per D1) + optional **enabled-curve re-run on the TCPX pool** (per D2).
7. **Ops lab-14 / lab-16 / lab-17** (single-GPU health, cluster/job failure, perf monitoring & day-2) + docs 17–20.
8. **Arch lab-19 / lab-20 / lab-21** (storage, e2e pipeline, inference serving) + docs 22–24.
9. **Finalize:** doc-15 + Part V/VI wiring; each track appends its README/doc-00 rows into the Track-0 skeleton; VERIFICATION.md complete.

Steps 2–3 (Ops) and 4 (Scaling baseline) have no ordering constraint between them beyond D1's touch-point on lab-10; run in parallel where convenient. Step 5 (Arch TCPX) gates only the *enabled* half of step 6.

---

## 4. Discipline (inherited by all tracks)

- Every addition keeps the repo's honesty rule: no fabric/number claimed that wasn't read off a live run; curated/synthetic signatures labelled as such.
- Flex-safe throughout: no node drain/delete; the TCPX pool respects the A3-Flex cap-of-3 stop-line; holders untouched.
- All provenance to `VERIFICATION.md`, cluster/context named.
- When a track edits a **shared** file, it edits **only its owned/contributed portion** per §2 and leaves a comment or commit note referencing this roadmap so the next track sees the coordination.
