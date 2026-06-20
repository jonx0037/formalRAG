# Brief — The Inverted Index and Safe Dynamic Pruning (`inverted-index-dynamic-pruning`)

Domain: probabilistic-ir · difficulty: intermediate · pipelineStage: retrieve · financeCaseStudy: true (text, pdf).
Prereqs: `vector-space-model-tfidf`, `bm25-binary-independence-model`.

## Positioning

The one classical-IR topic where the math is about *algorithms*, not relevance. BM25 scores a document;
this topic is how you return the top-k from millions without scoring them all — and do it *exactly*.
The hook: dynamic pruning is provably exact (same top-k as a full scan) yet has no asymptotic speed
guarantee. That exact/approximate distinction is the series' anti-hype stance made concrete.

## Section outline (H2)

1. The inverted index (postings, df, length array; DAAT vs TAAT).
2. Exhaustive document-at-a-time scoring (the ground truth).
3. WAND — per-term upper bounds, the threshold, the pivot.
4. The safety theorem — WAND returns the exact top-k (proof).
5. BlockMax-WAND — tighter local bounds, deeper skipping.
6. The honest limit (no asymptotic guarantee), finance case study, implementation.

## Math (all verified in `notebooks/inverted-index-dynamic-pruning/inverted_index_dynamic_pruning.py`)

- **Def 1 — inverted index.** term → sorted `[(doc, tf)]`; df, length array.
- **Def 2 — exhaustive DAAT top-k.** min-heap of size k; score every matched doc.
- **Def 3 — WAND pivot.** `UB_t = max_d contribution_t(d)`; threshold θ = k-th best so far;
  pivot = first prefix with `Σ UB ≥ θ`; skip docs before pivot doc.
- **Thm 1 — safety.** WAND returns the exact top-k. Proof: θ monotone non-decreasing; for any skipped d,
  `score(d) ≤ Σ_{t∋d} UB_t < θ ≤ θ*`, contradicting `score(d) ≥ θ*`. Needs only a *valid* UB + *monotone* θ.
- **Prop 1 — BlockMax-WAND.** Per-block maxima are valid (so exact) and ≤ global UB (so prune ≥ WAND);
  when the block-max bound < θ the full score is skipped. Safety proof reused verbatim.

**rigorFlag (executable):** no asymptotic guarantee — a flat-score query prunes nothing (WAND evals ==
exhaustive); block size is tuned; UB validity must be re-established for any non-BM25 scorer.

## Verified numbers (the viz mirrors postings + UBs to the decimal)

- BM25 replicated verbatim from bm25.py (idf "bm25", k1=1.5, b=0.75) → exhaustive #1 = filing-onpoint
  on the shared finance corpus (consistency with the BM25 topic).
- Pruning at N=5000, k=10: exhaustive **2870**, WAND **792**, BlockMax-WAND **209** (7% of collection).
- Adversarial flat-score query: WAND **40** == exhaustive **40** (no pruning).
- Worked corpus (N=10, avgdl=9), k=3: exhaustive 10, WAND 6, BMW 6; k=5: 10, 7, 7. Top-3:
  d0-margin 3.001, d1-hedge 2.338, d9-boiler 1.845.
- Safety verified on worked + finance corpora (all k) + 150 random instances; threshold monotone.

## Viz — `src/components/viz/InvertedIndexPruningVisualizer.tsx`

Method selector (exhaustive / WAND / BlockMax-WAND) + top-k slider. Replays all three algorithms in TS
over the mirrored postings (live counts match the notebook at every k).

- **Panel A — postings grid:** rows = query terms, cols = docs, cell = contribution heat, UB_t at row end;
  columns never fully scored under the selected method are dimmed.
- **Panel B — documents fully scored:** exhaustive vs WAND vs BMW bars (the payload); same top-k for all.
- **Panel C — top-k ranking.**

## Notebook — `notebooks/inverted-index-dynamic-pruning/`

- `inverted_index_dynamic_pruning.py` — owns the numbers; exhaustive/WAND/BMW + 6 asserting tests +
  `viz_constants()`. Run: `uv run --with numpy python notebooks/inverted-index-dynamic-pruning/inverted_index_dynamic_pruning.py`.
- `01_inverted_index_dynamic_pruning.ipynb` — narrative, imports the `.py`, generated via nbformat.
  Run: `uv run --with numpy --with jupyter jupyter execute notebooks/inverted-index-dynamic-pruning/01_inverted_index_dynamic_pruning.ipynb`.

## Cross-site

- formalStatistics `order-statistics-and-quantiles` — the threshold is the k-th order statistic of the
  score stream, maintained incrementally. (A genuine link; this is otherwise a systems/algorithms topic
  with no further sibling up-links.)

## References (DOIs verified via `curl -sI`)

- Broder et al. 2003 (WAND) — `10.1145/956863.956944`.
- Ding & Suel 2011 (BlockMax-WAND) — `10.1145/2009916.2010048`.
- Mallia et al. 2017 (VBMW) — `10.1145/3077136.3080780`.
- Turtle & Flood 1995 (DAAT/TAAT) — `10.1016/0306-4573(95)00020-H`.
- Manning, Raghavan & Schütze 2008, ch. 1–2, 5 — IR-book URL.
- PISA — documentation.
