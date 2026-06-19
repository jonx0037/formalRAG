# Handoff brief (SCAFFOLD) — `rank-fusion-rrf`

> **Status: working scaffold, not a finished spec.** This is the Phase-A skeleton from
> `STARTER-PROMPT.md`. The section outline, theorem list, viz intent, and notebook design below are
> a starting point to refine *together* before any prose is drafted. Open questions for Jonathan are
> collected at the end — those decisions shape the rest. Nothing here is committed to the page yet.

## Placeholder table (proposed)

| Field | Proposed value | Note |
|---|---|---|
| `title` | Rank Fusion: Reciprocal Rank Fusion and the Geometry of Rank Aggregation | matches the stub |
| `subtitle` | Why combining ranked lists by *position* beats combining by score — and how close a cheap heuristic gets to the optimal consensus | draft |
| `slug` | `rank-fusion-rrf` | exists as stub |
| `domain` | `ranking-fusion` | matches stub |
| `difficulty` | `advanced` | locked — full treatment incl. Kemeny NP-hardness |
| `pipelineStage` | `fuse` | new |
| `financeCaseStudy` | `true` | locked — stub flips from `false` |
| `modality` | `[text, audio]` | locked — 10-K text ⊕ earnings-call (audio) transcripts |
| `prerequisites` | `[bm25-binary-independence-model]` | already the graph edge; add to frontmatter |
| `reference notebooks` | `notebooks/bm25/` | the exemplar |

## Positioning

BM25 gives a *lexical* ranking; a dense dual-encoder gives a *semantic* one. They disagree, live on
incompatible score scales (BM25 unbounded; cosine ∈ [−1, 1]), and each misses what the other
catches. Rank fusion is the hybrid-retrieval step that combines them. RRF is the dominant method in
practice — and a clean doorway into the real mathematics of **rank aggregation**: positional voting
rules, the Kendall-τ geometry of permutations, the Kemeny consensus, and the gap between a cheap
heuristic and the NP-hard optimum. The finance thread: fuse a BM25 ranking over 10-K text with a
dense ranking over earnings-call passages; the fused list surfaces a disclosure neither leg ranks first.

## Section outline (draft — 9–11 H2s, mirrors the BM25 cadence)

1. **Overview & motivation** — two retrievers disagree; why you can't just add their scores.
2. **Score fusion and why it breaks** — CombSUM / CombMNZ; sensitivity to score scale and
   distribution; a worked case where unbounded BM25 scores swamp cosine.
3. **From scores to positions: Reciprocal Rank Fusion** — definition; the reciprocal-rank decay
   $1/(k+r)$; $k$ as the top-heaviness knob.
4. **RRF as a positional voting rule** — social-choice framing; Borda count; the $k\to\infty$ limit
   *is* Borda; $k\to 0$ is winner-take-the-top.
5. **The geometry of rankings** — permutations as points; Kendall-τ distance (discordant pairs) as a
   metric; Spearman footrule; the Diaconis–Graham inequality $K \le F \le 2K$.
6. **The Kemeny consensus** — the median permutation under Kendall-τ; the (extended) Condorcet
   property; NP-hardness via minimum feedback arc set.
7. **A computable surrogate** — footrule-optimal aggregation via min-cost bipartite matching; the
   2-approximation of Kemeny (Dwork et al.); where RRF sits relative to both (heuristic, no guarantee).
8. **Evaluation** — does fusion actually help? NDCG over the finance corpus; hybrid beats either leg.
9. **Finance case study** — BM25-over-10-K ⊕ dense-over-transcripts; the surfaced disclosure.
10. **Honest caveats** (`rigorFlag`) — see below.
11. **Implementation** — the notebook pillar.

## Theorems / propositions to prove (the rigorous core)

- **Prop (scale invariance).** RRF depends only on ranks, so it is invariant to any strictly
  monotone reweighting of either retriever's scores; CombSUM is not. *(short, but the crux of why RRF.)*
- **Prop (RRF → Borda).** As $k\to\infty$, the RRF order converges to the Borda order. A limit
  theorem in the BM25 spirit, made executable.
- **Thm (Diaconis–Graham).** $K(\sigma,\tau) \le F(\sigma,\tau) \le 2\,K(\sigma,\tau)$ relating
  Kendall-τ and Spearman footrule — full proof; this underpins §7.
- **Thm (footrule 2-approximates Kemeny; Dwork et al.).** Footrule-optimal aggregation is
  poly-time (min-cost perfect matching) and within $2\times$ the Kemeny optimum. State + proof sketch
  leaning on Diaconis–Graham.
- **Remark (Kemeny is NP-hard).** Via minimum feedback arc set on tournaments — cite, don't reprove.

## `rigorFlag` (draft)

> RRF's $k=60$ is an empirically chosen constant (Cormack et al. 2009), not derived from any
> optimality criterion; RRF is a positional heuristic with **no** approximation guarantee to the
> Kemeny-optimal consensus — which is itself NP-hard to compute. The rigorous results here are about
> Kendall-τ/footrule geometry and the footrule 2-approximation, *not* about RRF's optimality; RRF's
> effectiveness is empirical.

## Three-pillar sketch

**Math (KaTeX):** the outline + theorems above, geometric-first — permutations and Kendall-τ
introduced visually before the algebra.

**Viz — "Fusion Laboratory" (D3):**
- Two input ranked lists over a shared finance corpus (List A: BM25/lexical; List B: dense/semantic),
  shown side by side; a third column is the live fused ranking.
- A **$k$ slider** (RRF constant): rows animate/reorder as $k$ changes; per-document hover shows the
  $1/(k+r)$ contribution from each list (mirrors BM25's per-term contribution tooltip).
- A method toggle: **RRF vs Borda vs CombSUM** — CombSUM visibly collapses onto the BM25 order when
  its scores are scaled up, dramatizing scale-invariance.
- Optional second panel: Kendall-τ distance from the fused list to each input (and to the Kemeny
  consensus for the small corpus), showing fusion reduces total disagreement.
- Corpus + both input rankings + $k$ are **mirrored to the decimal** from the `.py`.

**Python (notebook pillar):** `notebooks/rank-fusion-rrf/rank_fusion_rrf.py` (+ `01_…ipynb`).
- Reuse the lexical leg from `notebooks/bm25/bm25.py`; build the dense leg from a *deterministic*
  toy embedding — **locked:** seeded random projection of bag-of-words → cosine, so the notebook is
  self-contained, CPU-only, < 60 s, with no model download (not injected precomputed rankings).
- Functions: `rrf_fuse(lists, k=60)`, `borda`, `combsum`, `kendall_tau`, `spearman_footrule`,
  `footrule_aggregate` (scipy `linear_sum_assignment`), `kemeny_bruteforce` (exact, small $n$),
  `ndcg_at_k` (import from bm25).
- Harness `assert`s (each a pedagogical claim):
  - RRF order unchanged when BM25 scores are scaled ×1000; CombSUM order **changes**.
  - $K \le F \le 2K$ across seeded random permutation pairs (Diaconis–Graham).
  - RRF order at large $k$ equals Borda order (the limit theorem).
  - Kemeny-distance of the footrule aggregate $\le 2\times$ that of the brute-force Kemeny optimum.
  - NDCG(RRF) > max(NDCG(lexical), NDCG(dense)) on the finance corpus + qrels (hybrid beats either leg).

## Cross-site links (VERIFIED against live sibling repos — 2026-06-19)

Searched both siblings' `src/content/topics/`. **No dedicated Kendall-τ / Spearman rank-correlation
topic exists on either site, and formalML has no learning-to-rank topic.** Do not invent slugs for
these. What is real:

- **`formalmlPrereqs: rank-tests`** — formalML's "Rank Tests & Permutation Inference" (Wilcoxon,
  Mann-Whitney, Kruskal-Wallis, permutation inference; `status: published`). Honest relationship: the
  Kendall-τ *distance* that grounds the geometry of rank aggregation is the same concordant/discordant
  pair counting as the Mann-Whitney U / Wilcoxon rank statistics — Kendall-τ is itself a U-statistic.
  Cite the shared combinatorial machinery; do **not** claim `rank-tests` covers rank correlation (it
  doesn't). This is the natural UP-link.
- **Optional `formalstatisticsPrereqs: order-statistics-and-quantiles`** (formalStatistics, published)
  — ranks are a deterministic function of order statistics; supplies the rank-vector representation.
  Include only if a section actually leans on it, else omit to avoid a thin link.
- **Learning-to-rank: no cross-site target exists.** The in-site neighbor is the planned
  `ranking-fusion` LTR topic (LambdaRank/LambdaMART) — a `connections` edge, not a cross-site one.
- formalCalculus: none. Reverse links from siblings are added per-sibling (worktree off `origin/main`)
  once this ships.

## References (Chicago 17th, Notes & Bibliography — core URLs/DOIs verified 2026-06-19)

- Cormack, Gordon V., Charles L. A. Clarke, and Stefan Büttcher. "Reciprocal Rank Fusion Outperforms
  Condorcet and Individual Rank Learning Methods." *SIGIR '09*. https://doi.org/10.1145/1571941.1572114 — RRF origin; the $k=60$ constant.
- Fox, Edward A., and Joseph A. Shaw. "Combination of Multiple Searches." *TREC-2*, NIST SP 500-215,
  pp. 243–252, 1994. https://trec.nist.gov/pubs/trec2/papers/txt/23.txt — CombSUM/CombMNZ.
- Dwork, Cynthia, Ravi Kumar, Moni Naor, and D. Sivakumar. "Rank Aggregation Methods for the Web."
  *WWW '01*. https://doi.org/10.1145/371920.372165 — Kemeny, footrule, NP-hardness, 2-approximation.
- Diaconis, Persi, and R. L. Graham. "Spearman's Footrule as a Measure of Disarray." *JRSS-B* 39, no. 2
  (1977): 262–268. https://doi.org/10.1111/j.2517-6161.1977.tb01624.x — the $K \le F \le 2K$ inequality.
- (optional, `documentation`) a production hybrid-search RRF doc — Elasticsearch / Qdrant / Weaviate — for the systems-aware angle.

## Decisions (locked 2026-06-19)

1. **Finance thread: ON.** `financeCaseStudy: true`; worked example = BM25-over-10-K ⊕
   dense-over-earnings-call-transcripts; `modality: [text, audio]`.
2. **Ambition: full rank-aggregation treatment** — Kendall-τ geometry, Diaconis–Graham, footrule
   2-approximation, Kemeny NP-hardness. Real theorems, not just a formula.
3. **Difficulty: `advanced`.**
4. **Dense leg in code: deterministic toy embedding** — seeded random projection of bag-of-words →
   cosine; self-contained, CPU-only, no model download.
5. **Cross-site: verified** (see the Cross-site section) — `rank-tests` on formalML; no LTR slug.

## Resolved at authoring (2026-06-19)
- **Hybrid-search `documentation` reference: INCLUDE.** Elasticsearch's RRF reference (the
  canonical industry implementation; popularized the `k=60` default). Verify the live URL when
  writing the MDX references.
- **`order-statistics-and-quantiles` cross-link: INCLUDE.** §5 is written to lean on the
  order-statistic representation of a ranking (the rank vector as a deterministic function of the
  order statistics), so the formalStatistics up-link is substantive rather than thin.

## Realized notebook design (verified — `notebooks/rank-fusion-rrf/`)

The corpus was engineered so each leg buries one qrel-3 prize at #3 and RRF recovers the ideal
order; the canonical numbers the MDX prose and `FusionLaboratory.tsx` must mirror to the decimal:

- Query `interest rate exposure`, N = 6. `interest` is the rare term (df 2, high IDF); `rate`/`exposure` common.
- **Lexical (BM25):** `filing-onpoint, filing-hedging, transcript-rate, filing-fx, news-macro, transcript-ops`
- **Dense (cosine):** `transcript-rate, filing-hedging, filing-onpoint, filing-fx, news-macro, transcript-ops`
- **Fused (RRF, k=60):** `filing-onpoint, transcript-rate, filing-hedging, filing-fx, news-macro, transcript-ops`
- **NDCG@10:** lexical 0.980, dense 0.980, **RRF 1.000** (hybrid beats both).
- **Kemeny consensus:** `filing-hedging` (every leg's #2) is lifted to #1 — RRF ranks it #3. A
  verified, concrete instance of *RRF ≠ the optimal consensus* for the `rigorFlag`/§6 contrast.
- **Kendall-τ(lexical, dense) = 3** discordant pairs; footrule aggregate matches the lexical order here.
- **Honest note (toy dense leg):** the seeded random-projection cosine leg is a self-contained
  stand-in for a trained bi-encoder, not a genuine semantic model (by Johnson–Lindenstrauss it is a
  length-normalized lexical-cosine ranking with no IDF). The rank-aggregation mathematics is
  independent of how the second list is produced — say so plainly in the prose; do not oversell "semantic."
