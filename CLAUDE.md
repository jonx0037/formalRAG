# CLAUDE.md — formalRAG

The mathematics of retrieval-augmented generation. Fourth site in the **formal** series
(formalML, formalCalculus, formalStatistics). Live at https://www.formalrag.com.
General preferences (pnpm, uv, Chicago citations, American English, geometric-first,
git workflow) live in `~/CLAUDE.md` — not repeated here.

## Positioning

Rigor-first, *systems-aware*: formalize RAG's genuine mathematics (retrieval/embedding geometry,
ANN algorithms, probabilistic IR, ranking, evaluation statistics, information theory) **and** the
systems layer where it has real math. Deliberately NOT a "how to build a RAG app" tutorial.
The author's production multimodal-financial-rag system + *Applied NLP for Finance* are woven in
as a recurring finance case-study thread, culminating in a capstone.

## Tech Stack

Inherited verbatim from formalML: Astro 6 + MDX, React 19 + D3.js, Tailwind 4, KaTeX
(`remark-math` + `rehype-katex`), Pagefind search, pnpm, Vercel. Node ≥ 22.12.

## Commands

```bash
pnpm dev                  # Dev server at localhost:4321
pnpm build                # Production build + Pagefind. The script already bakes in
                          #   NODE_OPTIONS=--max-old-space-size=8192 (just run `pnpm build`), and
                          #   astro.config sets minify:false on Vercel (rehype-katex heap pressure).
                          #   Raise the heap in package.json as equation-dense topics ship (formalML is at 20480).
pnpm exec astro sync      # Validate the content schema + ALL frontmatter fast (no full render) —
                          #   use after schema or cross-site frontmatter changes.
pnpm validate             # validateConnections.ts. Roadmap nodes in curriculum-graph.json with no
                          #   MDX yet are NOTICES, not errors; cross-site refs are checked separately.
pnpm audit:cross-site     # Cross-repo reciprocity validator (needs sibling repos adjacent / FORMAL_*_PATH).
# Python pillar (per-topic, no shared venv):
uv run --with numpy --with scipy --with rank-bm25 python notebooks/<topic>/<topic>.py
```

## Content schema (`src/content.config.ts`) — departures from formalML

- `domain`: 10 RAG-specific keys (retrieval-foundations, probabilistic-ir, embedding-geometry,
  ann-indexing, vector-quantization, neural-retrieval, ranking-fusion, retrieval-evaluation,
  generation-grounding, rag-information-theory).
- Cross-site links go **UP into formalML** — the inverse of the siblings. The `crossSiteRef.site`
  enum includes `formalml`, and there are `formalmlPrereqs`/`formalmlConnections` arrays the
  siblings don't have.
- New fields: `financeCaseStudy` (bool), `modality` (text|pdf|audio|chart|news), `pipelineStage`
  (ingest…evaluate), `rigorFlag` (honest caveat string). `references.type` adds `documentation`.

## Content & verification conventions

- Three pillars per topic: rigorous math (KaTeX) + interactive D3 viz + working Python.
- **Notebook pillar contract** (`notebooks/README.md`): each topic ships
  `notebooks/<slug>/<slug_underscored>.py` — the canonical, tested, importable reference that *owns
  the numbers* — plus `01_<slug_underscored>.ipynb`, a narrative notebook that imports the `.py` and walks the
  topic section by section. Both must exit 0 before shipping; commit the `.ipynb` without stored
  outputs, and **normalize** a hand-written `.ipynb` (nbformat — add cell ids, clear outputs) or
  `jupyter execute` warns (a future hard error). Reliable path: emit the `.ipynb` from a throwaway
  `uv run --with nbformat` generator (sequential `cell-N` ids, `outputs: []`, `execution_count: null`),
  then `jupyter execute` to verify exit 0. `notebooks/bm25/` is the exemplar. The full per-topic
  workflow lives in `STARTER-PROMPT.md` (repo root) — keep it current as conventions evolve.
- **A dependent topic's `.py` IMPORTS its prereq's `.py`, never reimplements it** — add the prereq's
  **hyphenated dir** to the path, then import its **underscored module**:
  `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vector-quantization-lloyd-max"))` then
  `from vector_quantization_lloyd_max import ...` (the `rank_fusion_rrf`→`bm25` precedent). Reuse the
  prereq's exact dataset and **re-derive any shared baseline** rather than hardcoding it, so a
  cross-topic comparison (PQ's recall vs the flat-VQ ceiling it re-derives) is provably one cloud.
  When a prereq is **itself** a dependent topic, add **every** ancestor's hyphenated dir and import each
  underscored module (OPQ & IVF import both `product-quantization` *and* `vector-quantization-lloyd-max` —
  the grand-prereq supplies primitives the direct prereq doesn't re-export, e.g. `finance_dataset`). If a
  topic's chosen scope adds a genuine dependency (Full IVFADC needs `product-quantization` for the residual
  PQ), add the **DAG edge + frontmatter `prerequisites` entry**, not just the node-status flip.
- **Learned-rotation (OPQ) transpose checkpoint:** the VQ/PQ track applies rotations as `(X−mu) @ R.T`
  (R's *rows* = basis). A learned Orthogonal Procrustes R-step must return `R = V @ U.T` from
  `SVD(Xc.T @ Q) = U Σ V.T`, so `apply_rotation` gives `Xc @ R.T = Xc @ (U V.T)` — the intended rotated
  data. A wrong transpose makes distortion **increase**, so a monotone-descent assert + a cross-check
  against `scipy.linalg.orthogonal_procrustes` pin the orientation by construction.
- **Reusing a prereq's search in a layered/restricted structure (HNSW): write a fresh twin, then
  cross-check it byte-for-byte.** NSW's `greedy_search` indexes a flat `list[set]` adjacency and
  `KeyError`s on a node absent from a layer, so HNSW's per-layer beam can't reuse it verbatim — write a
  near-copy `search_layer` over a per-layer **dict** with `layer_adj.get(c, ())` (seeded from a *set* of
  entry points). Prove the twin is faithful with `test_search_layer_matches_flat_on_single_layer`: force
  every node to level 0 and assert `search_layer ≡ greedy_search` on the same flat adjacency — same
  **ids AND ndist**. That single-layer collapse is the cleanest correctness anchor for any "prereq +
  hierarchy" topic. Measure a scaling law (HNSW's `top_level ≈ log_M n`) from the **draws alone**
  (`max` of n geometric level draws), never a 100-graph-build sweep — it's microseconds and seed-cheap.
  And state the apex honestly: at finite n the realized top layer holds `~n·M^-top_level` nodes, so
  assert the geometric law **at** the apex plus the O(1) **extrapolation** to the `round(log_M n)` level,
  not a hardcoded "≤ k nodes".
- **Cross-index head-to-head = build every index on the prereq's exact dataset + ONE shared truth.**
  HNSW vs flat-NSW vs IVF runs all three on the *same* `nsw_dataset` cloud with a single
  `_true_topk(...)` shared across every recall call, compared by **distance computations per query**
  (graph `ndist`; IVF `candidate_fraction·n + nlist`). The **robust** assertion is intra-family
  (`test_hnsw_beats_flat_nsw_at_equal_cost`: the hierarchy reaches a recall at ≤ the flat graph's cost);
  the cross-family verdict (HNSW vs IVF) is stated as **one synthetic cloud, not a universal ranking** —
  pin the inequality to the *observed* winner after running, per the headline-flip rule.
- **Filtered/incremental ANN = node removal from the prereq graph; the over-fetch laws are the exact
  spine, percolation the honest floor.** Deletion (tombstone, scan `k/(1−δ)` to collect k live) and
  predicate filtering (post-filter, scan `k/s`) are the SAME negative-binomial law with `s ↔ 1−δ` —
  build the viz to *show* it: one hyperbola `scan = k/r` vs pass-rate `r`, both measurement sets riding
  it. Extend the fresh-twin rule to a per-node predicate: `search_layer_filtered` adds ONLY a `live[]`
  guard (admit to results iff `live[nb]`, still traverse failing nodes) and collapses to the prereq's
  `search_layer` exactly when all-live — the same byte-for-byte anchor (ids AND ndist).
- **Network-percolation constant gotcha + connectivity ≠ navigability.** The random-deletion
  giant-component threshold is `p_c = 1/(M−1)` for a degree-M-**regular** graph (κ=⟨k²⟩/⟨k⟩=M), NOT the
  Erdős–Rényi `1/M` (κ=M+1, Poisson) — quoting `1/M` for HNSW's near-regular graph is the wrong-ensemble
  error. VERIFY the theorem on a configuration-model regular graph (where it's exact), then MEASURE on
  HNSW's real layer-0 (where it only approximates), and flag that a giant component existing ≠ greedy
  search *finding* the target: recall dies far INSIDE the connected regime, so percolation is the floor,
  never the binding constraint. The pre/post/in-filter "winner" is a selectivity **crossover**, not a
  ranking (pre cheapest+exact at low s; post cheapest at high s; in-filter degenerates toward a full
  scan at very low s), and the intuitive "correlated removal fragments earlier" runs **FALSE** — a
  spatially coherent (modality) predicate keeps the induced subgraph CONNECTED far below where a random
  predicate site-percolates apart. Pin every direction to the sim.
- `rigorFlag` is load-bearing: flag celebrated-but-heuristic results (HNSW scaling, MMR's missing
  1−1/e guarantee, BM25's empirically-tuned k₁/b). Honesty is the differentiator.
- **`pnpm build` passing ≠ math correct.** KaTeX is non-strict: parse errors render as
  `.katex-error` spans and the build still exits 0. Always open the topic page and check.
- **Verify with `browser_evaluate`, not screenshots.** Playwright/Preview MCP screenshots drift to
  `/` on this setup; instead assert DOM state (`.katex` count, slider/ranking presence). Viz uses
  `client:visible`, so the SSR DOM (KaTeX counts, baked readouts) is present on load, but to test
  *interactivity* (toggling a panel, dragging a slider) you must `scrollIntoView` the component and
  wait ~0.5–1 s for hydration first — otherwise clicks no-op against un-hydrated markup. A
  React-controlled range slider ignores a synthesized `input` event (state won't change); drive it with
  a **real keyboard interaction** (`focus()` then `ArrowRight`) and assert the readout updated.
  Hydration tell-tales: these labs render their SVG **declaratively in JSX** (not D3), so the
  `<rect>/<circle>` content IS in the SSR DOM — rect presence is **not** a hydration tell. Use the
  lab's `katex.render()` formula output (a post-load `.katex` count bump), and wait ~1–1.5 s (0.5 s
  was too short for a panel switch) before the **first** click — assert on the katex bump first.
  **Scope DOM assertions to the lab container** — the topic page has other SVGs (DAG/connection graphs)
  that inflate document-wide `circle`/`path`/`text` counts. Hydration is per-load: a fresh navigation
  needs another `scrollIntoView` before any click, **tab/panel switches included** (not just sliders).
- Pagefind UI assets 404 in `astro dev` (generated only by `postbuild`) — expected, harmless.
- **Don't hyperlink prose forward-references to unbuilt topics** — the link 404s until that topic
  ships. Link only to slugs that already have MDX; name a future topic in prose without a link.
  (Stale placeholder links can lurk in *published* topics too — e.g. a "Johnson–Lindenstrauss" link
  was once parked at `/topics/the-retrieval-problem` until a later sweep corrected it.)
- `pnpm dev` may not land on **4321** — with other `formal*` servers up it picks 4322/4323/…; read
  the dev log for the actual port (a `curl :4321` can hit a *different* project and falsely report
  ready). Stop only your own server with `lsof -ti tcp:<port> | xargs kill`, never `pkill -f astro`.
- `astro check` reports ~12 pre-existing type errors in the copied viz components
  (DAGGraph/CurriculumGraph/Figure), inherited from formalML — not regressions. Keep NEW code clean.
- **Viz ↔ Python invariant:** `BM25ScoringLaboratory.tsx`'s corpus mirrors `notebooks/bm25/bm25.py`
  to the decimal, and the topic claims they match. Change one → change both. Numbers the viz needs
  but the corpus *doesn't determine* (e.g. a full-document L2 norm that includes filler terms) go in
  a `viz_constants()` function in the `.py` that prints them in the harness, then are mirrored to the
  `.tsx` — never recomputed in TS (`vector_space_model_tfidf.py` / `probability_ranking_principle.py`).
- **Cast numpy scalars in `viz_constants()` prints** (`round(float(v), 3)`, `int(...)`) — otherwise arrays
  render as `np.float64(...)`/`np.int64(...)` and dirty the values you mirror into the `.tsx`.
- **Pedagogical claims are tests:** the Python harness asserts the limit theorems and the
  length-hijack flip. Don't let prose drift from the verified numbers.
- **Build + RUN a headline flip before writing it into prose/viz — it can be false under the topic's
  exact weighting.** TF-IDF's length-hijack flip surfaces the *wrong* doc under exact IDF `log(N/df)`
  because a corpus-universal term (`rate`, in all docs) gets IDF 0; the fix was to teach the exact
  form (the self-information theorem) but *score* with smoothed `log(1+N/df)`, flagging smoothing as
  a convention. Reusing a shared corpus across topics does **not** guarantee the same flip holds
  under each topic's scoring variant. **Smoothing/normalization parameters must scale to the toy
  corpus's short docs** — Dirichlet μ≈document length (≈5), not the production μ≈1000–2000, or smoothing
  swamps every doc model toward the collection and the ranking signal vanishes.
- **An *asymptotic* headline can be false at the scale you first simulate.** Kleinberg's α=d navigability
  optimum failed at n=900 (2-D lattice: α=0 beat α=2 — the wrong story); it needs large n. The fix was the
  1-D ring (optimal α=1) at n=20000, whose translation symmetry makes the long-range-offset distribution
  shared across nodes → construction is **O(n)**, not O(n²). Simulate at the scale where the separation
  appears, and pick a construction cheap enough to afford it.
- **A viz that *demonstrates* a phenomenon needs baked toy data that actually exhibits it — assert the
  contrast in the harness, don't assume.** The Lloyd lab's first cloud (3 well-separated blobs) converged
  to one optimum from every seed, falsifying its "Reseed shows local optima" story; 4 corner blobs at k=3
  fixed it, locked by `test_toy_local_optima` (global < near-miss < stuck). In a ranking demo where
  scores can all be **zero** (a pruned/empty SPLADE query), top-1 is decided by insertion order, so a
  "retrieved correctly" assert passes spuriously — require top-1 **and a positive score**
  (`top_id == gold and top_score > 0`). A learned-sparse demo also needs **document-side** expansion,
  not just identity activations, or the sparsity (FLOPS/L0) trade-off is flat and shows nothing.
- **vMF cluster toys: two `vMF(μ,κ)` draws have expected cosine ≈ `A_d(κ)²`.** A κ that "looks tight"
  is near-orthogonal in expectation at high d (κ=12, d=64 → cosine ~0.03), so a same-cluster "hard
  negative" isn't actually hard. Size κ so same-cluster cosine clearly beats inter-cluster and assert it
  — InfoNCE's finance toy needed κ_sector≈60 at d=32, not κ=12 at d=64.
- **Rank-ceiling / expressivity toys (DPR, embedding-dimension-lower-bounds): one item per cluster
  forces rank = cluster count, making the ceiling demonstrable.** Build `S = Q Pᵀ` with ONE document
  per company → `rank(S) = #companies` exactly; a naive same-company corpus gives a *gradual* recall
  climb with no clean ceiling. Size the within-cluster κ so same-cluster items stay genuinely near
  (κ_sector≈60 at d=32), so resolving them needs nearly all the rank and recovery sits **below** full
  rank (recall@1 recovered at d=6 < rank=8). Assert the **contrast** (recall@1 monotone in d, <0.5 at
  d=1, =1.0 at d≥recover, exact reconstruction at d=rank), not the decimals — `viz_constants()` bakes
  those; a `/tmp` design-agent estimate drifts from the shipped `.py` (D_RECOVER moved 5→6), the `.py`
  owns the numbers. Reuse the prereq's **loss** as a byte-for-byte anchor (`inbatch_loss_via_gram` ==
  imported `info_nce_loss_batch`, <1e-12), the same twin rule as a reused search routine.
  Its **sign-rank successor** (`embedding-dimension-lower-bounds`): exact sign-rank is intractable
  (∃ℝ-complete / NP-hard) — don't compute it, demonstrate the gap (full rank by eigenvalues + a
  convex-position rank-3 realization) plus a **Forster** spectral lower bound (`sign-rank ≥
  √(mn)/‖M‖`, `=√N` for a Hadamard pattern); assert the contrast. **Vectorize** any free-embedding /
  qrel realizability optimizer's multi-positive row loss (one shared per-row `NegSum`) — a per-query
  Python loop over a `C(n,2)`-query all-pairs corpus busts the <60 s budget (57 s→4 s here).
  Its **late-interaction successor** (`late-interaction-learned-sparse`): to show architecture B
  (multi-vector MaxSim) escapes a limit architecture A (single vector) hits, run BOTH through the
  **SAME optimizer** — B's, with the degenerate setting *being* A (`realize_qrel_maxsim` m=1 vs m=2),
  never A's separately-tuned one — or you compare optimizers, not architectures; anchor the degenerate
  case by proving it scoring-identical (MaxSim at m=1 == the imported DPR dot product, <1e-12). MaxSim's
  max-pool backprop is a gather/scatter through each query token's argmax doc token (`take_along_axis`
  + `np.add.at`). When no clean theorem exists (multi-vector sign-rank is open, LIMIT defers it), state
  the escape as a **demonstrated proposition**, not a theorem, and pin the rigorFlag.
  Its **multi-vector-ANN successor** (`multi-vector-ann-retrieval`, PLAID): the topic *is* "reuse the two
  ANN prereqs" — cluster ALL tokens with the IVF `coarse_quantizer` (on UNIT-normalized tokens: L2
  k-means == cosine clustering **at the objective level** — assert the distance identity
  `‖a−b‖²=2−2⟨a,b⟩`, NOT an assignment equality, since Lloyd cell means aren't unit-norm), residual-PQ
  each token (IVFADC at token level), approximate MaxSim by the centroid, prune, then rerank survivors
  with the **imported** `maxsim_score`. **Cheap-prune + lossy-rerank cascade gotchas:** (1) recall-
  monotone-in-`keep` is a theorem ONLY under an **exact** rerank (superset of survivors → exact top-k
  can only gain true neighbors); the deployed **lossy-PQ** rerank can DIP (a false positive with a high
  *approximate* score displaces a true neighbor) — that dip *is* the score-vs-ranking gap, so assert
  monotonicity on an `exact=True` index, *demonstrate* the lossy frontier. (2) The Cauchy–Schwarz error
  `|⟨q,d⟩−⟨q,c(d)⟩|=|⟨q,r⟩|≤‖q‖‖r‖` (lift to the doc score via the 1-Lipschitz max) is a bound on
  **scores, not the ranking** — state it as the load-bearing rigorFlag, it's *why* the exact rerank
  exists. (3) The collapse anchor (probe-all + prune-nothing + exact rerank == brute MaxSim, recall 1.0
  + identical ordering) needs an **`exact=True` index mode** storing original tokens — PQ is lossy and
  can't reach 1.0; bake BOTH frontier curves (exact → the brute line = collapse anchor; PQ plateaus
  below, the gap = compression loss). **Storage panel:** bake **representative ColBERT-scale** numbers
  (d=128, 32 tokens, K=2¹⁶) so the "32× raw multi-vector → ~1× a single-vector index" story matches the
  late-interaction lab — NOT the small retrieval toy's params (which give a confusing <1× artifact). For
  a K-slider viz, bake only the per-K **centroids** (k-means isn't a closed form) and recompute the
  grid/error/bound in TS from them.
  Its **capstone successor** (`capstone-multimodal-financial-rag`): a SYNTHESIS topic — its novelty is
  **composition, not a new primitive** (imports BM25, dense, MaxSim/PLAID, IVF, PQ, RRF, the over-fetch
  law; reimplements none). Its declared graph prereqs included **5 UNBUILT upper-layer nodes** (graphrag,
  conformal-factuality, context-selection-DPP, cross-modal-alignment, retrieval-distillation); scope to
  the **published** stack, re-point the edges to published nodes, and **name-not-link** the
  generation/grounding/eval layer. **Fusion gain is structurally impossible if the legs are a quality
  ladder** — three monotone approximations of ONE scalar truth (pooled-cosine < centroid-MaxSim < …) make
  the best leg dominate and fusion only adds noise (gain < 0). Make each leg a **partial VIEW** (disjoint
  token windows: lexical/dense/late-interaction see different tokens) so each recalls neighbors the others
  miss — the real multimodal story, and the only thing that gives RRF something to recombine. **The
  dominated-leg flip needs CO-ENDORSEMENT:** under RRF `c=60` a lone top vote (`1/61≈0.0164`) is weaker
  than two mid votes (`~2/63≈0.031`), so the naive "one noisy leg drowns the good one" instance does NOT
  flip — the false positive must be ranked decently by BOTH legs. Build+RUN it; the obvious counterexample
  is false. **Cascade FKG direction:** positive stage dependence ⇒ the independent product `∏rᵢ` is a
  conservative **LOWER** bound on true recall (not upper — correlated failures pile onto already-hard
  queries, losing fewer DISTINCT docs); verify with a bivariate-normal survival copula across BOTH signs,
  and use **illustrative middling retentions** (0.6, 0.5) for the demo — the MEASURED `r₁≈1` leaves no room
  for FKG to bite (vacuous ±0.001 gaps). The over-fetch `1/∏rᵢ` is the **algebraic reciprocal** of the
  composite retention (one neg-binomial on `∏rᵢ`), NOT a product of L physical scan-counts. **Water-filling
  needs real per-doc COST ASYMMETRY** (lexical 4 : dense 16 : late-interaction 256 comps) or uniform is
  already optimal and WF==uniform (vacuous); but cost-asymmetry + a saturating curve pushes cheap legs to
  the depth ceiling — **drop the retention=1 grid point** so capped channels sit ABOVE the water level
  (clean KKT, marginals level). The "no equal-marginal without the log" claim is **FALSE**: `∂R/∂cᵢ =
  gᵢ'·(R/gᵢ)` shares the common `R` that cancels; the log buys **separability + a global optimum**, not the
  existence of equal-marginal. Ground truth = MaxSim `brute_topk` (NEUTRAL: no candidate-gen leg is the
  oracle, so the legs complement and the full-budget collapse reaches recall 1.0). DOI gotcha: cascade
  ranking (Wang/Lin/Metzler, SIGIR 2011) is **`10.1145/2009916.2009934`** (the `…2010022` variant 404s);
  RRF is Cormack et al. SIGIR 2009 `10.1145/1571941.1572114`.
  Its **evaluation-root successor** (`set-metrics-precision-recall-map-mrr`): the DAG root of the
  eval/IT layer — it DEFINES the recall@k the published stack only ever measured. IMPORT the three legs
  (bm25, dense, late-interaction-via-centroid-MaxSim) over the capstone's `token_corpus` + neutral
  `brute_topk` truth to score REAL rankings — re-derive the legs, never import the **downstream**
  capstone. **Notebook import graph ≠ pedagogical DAG:** the `.py` carries the deep multi-vector import
  chain to source real rankings, yet the single frontmatter prereq stays `the-retrieval-problem` (a
  reader needs only ranking+relevance, not BM25/PLAID) — do NOT add the legs as `prerequisites`.
  **Recall denominator = `|R|`** (textbook), NOT the capstone's `min(k,|R|)` (cascade-retention); they
  COINCIDE at the capstone's `k=|R|=10`, so the topic still grounds that recall@10, and the `|R|` form is
  what makes recall@N=1 and **AP = area under the PR curve** (`AP = Σ(R_k−R_{k−1})P_k = (1/|R|)Σ i/posᵢ`,
  a Riemann sum — the formalcalculus `riemann-integral` up-link) hold. AP divides by **`|R|`, not the
  count found** (a 1-of-3-found ranking scores 1/3, not 1.0 — the classic AP bug). **MRR needs its OWN
  known-item qrel** (`|R|=1`, the top-1 oracle doc): `|R|=10` makes RR degenerate, and the clean collapse
  anchor `|R|=1 ⇒ AP=RR ⇒ MAP=MRR` only exists there — ship BOTH qrel regimes off the one truth. **The
  metric-choice flip is a PAIRWISE REVERSAL, not an argmax swap** — argmax(MAP)==argmax(MRR) can hold
  (one leg tops both) while a *pair* reverses (dense beats lexical on MAP 0.735>0.619 but loses on MRR
  0.472<0.522); detect `d_map·d_mrr<0`, RUN it (natural at seed 0, no constructed fallback needed). The
  **metrics-as-estimators** thesis is the load-bearing rigorFlag (MAP=sample mean, SE=σ̂/√n, CI; concentration-
  inequalities/point-estimation/confidence-intervals up-links; significance-testing/bootstrap as forward
  connections). **Panel-D significance separation must be the PROJECTED closed form** `1.96·σ̂/√n` (full-
  sample σ̂, recomputable in TS) — NOT a query-order-dependent subsample, which gives a DIFFERENT
  separation n (projected 12 vs subset 19) and breaks the viz↔python invariant. Bake `REL_RANKS`/`KI_RANK`
  integers + rng/k-means scalars (`MAP`/`MRR`/`std`/`SE_SCALING`); recompute P/R/PR/AP/interpolation/CI
  closed-form in TS. Refs verified: Buckley–Voorhees "Evaluating Evaluation Measure Stability" SIGIR 2000
  `10.1145/345508.345543`; Sanderson FnTIR 2010 `10.1561/1500000009`; Efron–Tibshirani bootstrap
  `10.1201/9780429246593`; `ir_measures` (https://ir-measur.es/) for the AP-convention zoo.
  Its **graded-relevance successor** (`ndcg-discount-geometry`): generalize binary→graded by IMPORTING
  `set_metrics_corpus`/`LEGS` and the generic estimator fns (`metric_summary`, `projected_ci`) — feed
  per-query NDCG through the SAME `metric_summary`, don't re-derive. **Grade by GLOBAL oracle-score
  TERTILES restricted to the top-K** (`maxsim_matrix(queries,docs)` is the exact oracle `brute_topk`
  argsorts, so its top-K == `qrels_set`): assign {1,2,3} to the K relevant docs, 0 elsewhere, so
  `{grade≥1}≡qrels_set` EXACTLY (the nesting anchor) WHILE IDCG varies per query — a rank-band grading
  makes every query's grade profile identical (IDCG constant), score-tertiles give per-query variation
  AND nesting; tertiles also balance the grade counts into thirds. **The bm25.ndcg_at_k TWIN is exact:**
  bm25's `r/log2(i+2)` over a 0-indexed enumerate == `1/log2(i+1)` 1-indexed, linear gain,
  `sorted(qrels.values())[:k]` IDCG — so DEFAULT the signature to `(gain_linear, discount_log2)` and the
  twin is the no-extra-arg reduction (`<1e-12`); pass `gain_exponential` explicitly for the featured
  modern NDCG. **The rearrangement inequality IS the rigorous backbone** (IDCG optimality is a THEOREM,
  not a definition): `DCG = ⟨gains-in-rank-order, descending-discounts⟩`, maximized by descending gains,
  ascending = the strict minimizer when grades differ — assert MAX over random perms + the strict
  ascending<ideal. **Quality-ladder ⇒ NO aggregate leg flip** (the capstone learning recurs: late>dense>
  lexical under every convention), so the headline is CONSTRUCTED convention flips + a per-query reversal
  count (5 here): gain flip (1 perfect g=3 + 3 marginal g=1; `2^g−1` exp→"headline"@rank1 wins, linear→
  "broad"@top-3 wins) and discount flip (3 equal docs; steep geometric p=0.5→"top_heavy" wins, heavy-tail
  log2→"deep" wins). Build+RUN — the obvious aggregate flip is vacuous. **Discount geometry:** head-mass
  in top-K geometric(0.85)≈0.80 > 1/i ≈0.55 > log2 ≈0.19 (light vs heavy tail), marginal value
  `disc(i)−disc(i+1)` positive AND decreasing, and RBP's `E[docs]=1/(1−p)` is the closed user model log2
  LACKS — assert the inequalities, not decimals. **Worked-query picker must PREFER a query containing a
  grade-3** (median-NDCG alone landed on a no-grade-3 query, hiding the exp-gain effect). Two contrasting
  estimator pairs: the **clearest** separates within Q (n=15), the **closest** (lexical/dense, gap 0.05)
  needs the EXTRAPOLATED `n≈185` — give the separation fn an `n_max` param to project past Q. NDCG@k
  truncation can be inconsistent (Wang et al., COLT 2013 — PMLR v30, no DOI), the load-bearing cutoff
  rigorFlag. Refs verified: Järvelin–Kekäläinen TOIS 2002 `10.1145/582415.582418`; Moffat–Zobel RBP TOIS
  2008 `10.1145/1416950.1416952`.
- **Rotation/Procrustes transpose checkpoint:** the VQ/PQ track applies rotations as `(X - mu) @ R.T`
  with R's **rows** = basis vectors (`pca_align`/`balanced_rotation` in `product_quantization.py`). A
  learned-rotation step (OPQ's non-parametric Orthogonal Procrustes update) must therefore return
  `R = V @ U.T` from `SVD(Xc.T @ Q) = U Σ V.T`, so `apply_rotation` yields `Xc @ R.T = Xc @ (U V.T)` —
  the intended rotated data. The wrong transpose makes distortion **increase**, so a monotone-descent
  assert plus a cross-check against `scipy.linalg.orthogonal_procrustes` pin the orientation by
  construction — add a new test of each when introducing a learned rotation.
- **InfoNCE / CPC MI-bound proof direction:** the lower bound `I(q;d⁺) ≥ log(N+1) − L` comes from the
  LLN (the sum of N negative density-ratios → N) **plus dropping the `+1`** in `log(1 + N/r₀)`, NOT a
  Jensen step on "the sum is N in expectation" — Jensen on the concave log gives an *upper* bound on L,
  the wrong direction. Verify the inequality direction numerically (Gaussian joint, Bayes-optimal critic)
  before writing the proof; the `log(N+1)` ceiling/saturation is the load-bearing rigorFlag.
- **Verify reference DOIs** with `curl -sI https://doi.org/<doi>` — the `location:` header in the 302
  alone confirms journal/volume/issue/pages (a HEAD request: no redirect-following, no paywalled GET).
  For conference papers also confirm the **venue + title** via content negotiation
  (`curl -sL -H "Accept: application/vnd.citationstyles.csl+json" https://doi.org/<doi>`) — the 302
  location doesn't catch a wrong venue (Filtered-DiskANN is **WWW 2023**, not the NeurIPS 2023 Big-ANN
  competition; ACORN is SIGMOD/PACMMOD 2024).
- The curriculum is the full 50-topic DAG in `src/data/curriculum.ts` + `curriculum-graph.json`;
  unauthored topics live in `tracks[].planned` and as `status: draft` MDX stubs.
- `status` gates only **listings** (homepage / `/topics` / `/paths`) + prereq availability;
  `getStaticPaths` builds a page for *every* topic, so the full MDX renders at its URL regardless of
  status (a draft "stub" shows a notice only because its *body* is one).
  Gotcha: a merged topic left at `status: draft` while its graph node is `published` still renders at
  its URL but is **hidden from every listing** — verify the flip landed at ship.
- **Sync local `main` first.** `gh pr merge` updates *origin*, not local `main` — before branching each
  topic run `git fetch origin && git checkout main && git merge --ff-only origin/main`. The tell that
  you're on a stale base: a "published" topic shows as a draft stub with an orphaned `__pycache__/` and
  no `.py`/`.ipynb` (its source was merged on a branch you don't have yet). An **Explore subagent will
  confidently report a prereq's `.py`/`.mdx` API surface that isn't on your local disk** in this state —
  verify the files exist (`ls` / `git ls-files`) and `git fetch && git log origin/main` to confirm origin
  is ahead BEFORE trusting the reported surface or branching.
  And commit CLAUDE.md learnings **inside the topic PR or a dedicated chore PR** — a post-merge local
  `docs: learnings` commit strands on the topic branch after the topic PR merges (the DPR one had to
  be cherry-picked).
- **Multiple topics in one session = feature branches off `main`.** They merge in any order *only if*
  each depends solely on already-published prereqs. If a batch topic lists a **sibling** as prereq (e.g.
  pseudo-relevance-feedback needs query-likelihood), sequence them: re-sync `main` only **after** its
  in-batch prereq merges, then branch the dependent off it. **If you can't wait for the prereq to reach
  `main`** (you don't control the merge), branch the dependent off the **prereq branch** — but then the
  dependent PR's base IS that branch, so merging it lands in the prereq branch (bundling both topics),
  NOT `main`, and GitHub's merge dialog names the prereq branch (it reads like merging the prereq); the
  prereq PR then carries both topics to `main` in one merge. Each removes its title
  from a track's `planned[]` array, so the **2nd+ merge needs a trivial one-line `curriculum.ts`
  `planned[]` conflict resolution** (the `curriculum-graph.json` node-status flips auto-merge; but a
  DAG *edge* re-source is a real content edit — keep it on one branch). PRs also get an automated
  `gemini-code-assist` review — fetch its nits with `gh api repos/jonx0037/formalRAG/pulls/<n>/comments`
  (inline comments carry the severity badges; the `/reviews` body is often empty), and address the
  medium-priority robustness/perf/a11y ones before merging. It reliably flags **unguarded denominators**
  (`avgdl`, `|d|+μ`, query length, Σ-of-weights) and empty-collection cases in the notebook `.py` (incl.
  `k≤0` on a recall fn and an empty matrix before `np.linalg.svd`) — add those guards up front. In the viz `.tsx` it reliably flags **transient state-length mismatches** (a
  slider that grows `points` before the reset effect refreshes `assignments` → a crash on `C[labels[i]]`)
  and stale refs in d3 drag handlers — guard array-index lookups (`C[labels[i]]`, `colors[a[i] ?? 0]`)
  and compute drag-end distortion from live points/centroids, not a render-lagging ref. It also flags
  recall/`topk` denominators (`hits/(nq·topk)`), `np.argpartition(d, topk)` when `topk>n` (cap
  `topk=min(topk, n)`), and **tuple-arity mismatches in a fallback `return`** (a 6- vs 7-tuple path — a real
  HIGH-severity catch). It also flags **list-comprehension membership filters over sets** (→ native
  `s1.intersection(s2)` / `s1 - s2` — both snapshot, so an in-loop `del dict[k]` (intersection over a
  dict's keys) or set `.discard(x)` stays safe). It also flags **unused imports** (and a **function a
  refactor orphans** — it flags dead functions, not just imports) and a hand-rolled sigmoid
  `1 / (1 + np.exp(-z))` (→ `scipy.special.expit`, which avoids an overflow `RuntimeWarning`). It also
  flags **loop-invariant recomputation**: hoist an `n`-independent `(mean, std)` out of an `n`-loop
  (a large `n_max` extrapolation), and precompute per-leg/per-item arrays ONCE before an
  `itertools.combinations` loop, not once per pair. But
  **decline with a posted rationale** the nits that would (a) break a byte-for-byte search twin — caching
  a beam's `worst` is an O(1) heap-peek, no gain, and diverges the twin from its `search_layer` source —
  or (b) SSR a KaTeX formula in a `client:visible` lab via `renderToString`: the island never SSRs, and
  every lab shares the `useEffect`+ref idiom (consistency beats a marginal CLS win). Gemini posts inline
  ~1–3 min after the push; `mergeable` flips to `UNKNOWN` transiently right then. A separate **Vercel
  Agent Review** check also runs but is often `NEUTRAL` (*skipped — insufficient credit*), which is NOT a
  failure; gemini stays the inline reviewer, and `mergeStateStatus` shows `UNSTABLE` transiently while the
  preview redeploys after a push. Gemini also flags a **nested `arr.map(r => r.map(...))` that returns a
  bare array** (wrap each row in `<g key={i}>` — heatmap/grid labs hit this). (`jupyter execute` does
  *not* write outputs back, so re-running to verify won't dirty the output-free `.ipynb`.)
- Cross-link `learning-theory` does NOT exist as a formalML slug → use `vc-dimension` /
  `generalization-bounds`. `maximum-likelihood` / `exponential-families` are **formalStatistics** slugs,
  not formalML; `shannon-entropy`, `kl-divergence`, `representation-learning`, `concentration-inequalities`
  are the formalML info-theory/representation slugs. formalML's Eckart–Young / low-rank slugs are
  **`svd`** (full E–Y–M proof) and **`pca-low-rank`** — `singular-value-decomposition` and
  `matrix-factorization` do NOT exist; `pca-dimensionality-reduction` is an **in-site** formalRAG slug.
  Confirm with `ls ../<sibling>/src/content/topics/<slug>.mdx`.

## Cross-site & sibling repos

- The siblings are **structurally divergent** — there is no uniform edit:
  formalML (full machinery + `audit-cross-site-links.mjs`; cross-site data feeds the /paths graph
  + audit, not page rendering), formalStatistics (`site: z.string()`, auto-renders in TopicLayout),
  formalCalculus (`z.literal` per site, `ConnectionsSection`, `connections.ts`/`topicMeta.ts`).
- When editing a sibling, use a **git worktree off its `origin/main`** (`git worktree add … origin/main`)
  — they're usually mid-work on their own dirty branches; never branch in place. Feature branch → PR.
- Reverse links from the siblings into formalRAG are added per-sibling as the linked topics ship
  (check each sibling's git history); formalCalculus is deferred until a real
  formalRAG↔formalCalculus link exists.
- `pnpm audit:cross-site` writes **gitignored** artifacts (`docs/plans/audit-output/`,
  `cross-site-audit-report.md`, `deferred-reciprocals.md`) — never commit them. It exits 0 even with
  warnings: a new topic's `formalml*/formalstatistics*` up-links surface as **missing reciprocals**
  (and, since the audit treats this repo as the `formalml` slot, the up-link as `self-site`) — the
  expected deferred state, not an error, once you've confirmed the target slugs exist on the siblings.

## Deploy

- Production: www.formalrag.com. Vercel project `formalrag`, team `jonathan-aaron-rocha`.
- The Vercel CLI needs `--scope jonathan-aaron-rocha` (no default in non-interactive mode) and a
  lowercase project name (the dir `formalRAG` is rejected — link with `--project formalrag`).
- **Push-to-deploy is live** (GitHub↔Vercel connected): every push to `main` auto-deploys to
  production; every PR/branch push gets a preview deployment (the Vercel bot comments the URL on the
  PR). Just merge to `main` — no CLI deploy needed; use the CLI only for an out-of-band deploy.

## Do NOT

- Use npm or generate `package-lock.json`.
- Commit `.vercel/`, `.claude/settings.local.json`, `dist/`, or `.astro/` (gitignored).
- Change a verified BM25 number in the harness or viz without updating the other.
- Put schema/web-dev metadata (field names, file paths, build commands) in reader-facing prose.

## Editorial voice

Informed peer, not lecturer; collaborative "we"; introduce all notation; no "simply/obviously".
American English. Match formalML's `svd.mdx` register. (Full voice guide: formalML's CLAUDE.md.)
