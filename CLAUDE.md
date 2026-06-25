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
  a **real keyboard interaction** — a synthesized in-page `KeyboardEvent` ALSO silently no-ops; `focus()` the
  slider via `browser_evaluate`, then press the key with the Playwright `browser_press_key` tool (a real OS key),
  and assert the readout updated.
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
  The same rule binds frontmatter **`connections[]` harder**: `pnpm validate` ERRORS on a
  `connections[]` entry whose topic has no MDX yet (unlike `curriculum-graph.json` roadmap nodes,
  which are only notices), so list only BUILT topics there and name forward topics in prose alone.
- `pnpm dev` may not land on **4321** — with other `formal*` servers up it picks 4322/4323/…; read
  the dev log for the actual port (a `curl :4321` can hit a *different* project and falsely report
  ready). Stop only your own server with `lsof -ti tcp:<port> | xargs kill`, never `pkill -f astro`.
- **The Bash tool's working directory PERSISTS across calls** — after any `cd` into a subdir (e.g. a notebook
  dir), repo-root-relative commands (grep/sed/ls, not just throwaway generators) silently target the wrong
  path; use ABSOLUTE paths or execute commands in a subshell, e.g., `(cd subdir && command)`.
- `astro check` reports ~12 pre-existing type errors in the copied viz components
  (DAGGraph/CurriculumGraph/Figure), inherited from formalML — not regressions. Keep NEW code clean.
  Preflight the notebook `.py` with `uv run --with pyflakes python -m pyflakes notebooks/<topic>/<topic_underscored>.py`
  before pushing — it catches unused imports/vars (the gemini nit class) faster than a build or a PR round-trip.
- **Viz ↔ Python invariant:** `BM25ScoringLaboratory.tsx`'s corpus mirrors `notebooks/bm25/bm25.py`
  to the decimal, and the topic claims they match. Change one → change both. Numbers the viz needs
  but the corpus *doesn't determine* (e.g. a full-document L2 norm that includes filler terms) go in
  a `viz_constants()` function in the `.py` that prints them in the harness, then are mirrored to the
  `.tsx` — never recomputed in TS (`vector_space_model_tfidf.py` / `probability_ranking_principle.py`).
- **Cast numpy scalars in `viz_constants()` prints** (`round(float(v), 3)`, `int(...)`) — otherwise arrays
  render as `np.float64(...)`/`np.int64(...)` and dirty the values you mirror into the `.tsx`.
- **Bake only REPRODUCIBLE numbers.** A randomized numerical routine baked into `viz_constants()` drifts
  run-to-run and silently breaks the viz↔python invariant — seed it. `scipy.sparse.linalg.eigsh`/`svds`/`eigs`
  use a RANDOM start vector by default, so pass a fixed `v0=np.random.default_rng(0).standard_normal(n)` when
  baking their output (gemini flags this); the all-ones vector is an exact 0-eigenvector of the modularity
  matrix `B`, a poor `v0`.
- **A baked-number change ripples THREE ways, not two:** a review fix that shifts a `viz_constants()` value
  (e.g. re-baking a witness −0.7038→−0.7105) must update the `.py` print, the `.tsx` const, AND any MDX **prose**
  that quotes it in words — grep the topic across both `.tsx` and `.mdx` for the old value before pushing.
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
  Its **significance/calibration/drift successor** (`significance-testing-calibration`): three pillars,
  ONE shared corpus/`.py`/viz, bound by "compare two distributions of per-query quantities" (system-vs-
  system, score-vs-truth, now-vs-then). IMPORT `ndcg_corpus` (a SUPERSET of `set_metrics_corpus` — adds
  `oracle_scores`/`grades`, so one cached build gives binary labels AND graded relevance) + `per_query_ap`/
  `per_query_ndcg`/`metric_summary`/`projected_ci`/`projected_separation_n`/`projected_ndcg_separation_n`
  and the scoring PRIMITIVES (`dual_encoder_score`/`maxsim_matrix`/`bm25_rank`/`normalize`) for the
  calibration scores. **Get the published unpaired n for FREE** by calling the prereqs' own
  `projected_*_separation_n` (guarantees the headline reuses the exact 185, not a re-derivation).
  **Significance:** pairing cancels shared difficulty (`var(d)=varA+varB−2cov`, ratio 0.59–0.69 here) —
  the paired CI excludes 0 at far fewer queries; resolve the NDCG cliffhanger as the LADDER **185 (crude
  overlap) → 116 (rigorous 80%-power) → 57 (single-realization)** and note power_n > single-realization-n
  because a one-time CI clearing 0 is ~50% power. **Build+RUN the honest twist:** pairing tightens WITHOUT
  manufacturing significance — MAP lexical/dense IS sig at n=40 (p≈4e-4) while NDCG lexical/dense is NOT
  yet (p≈0.107); ship both. Permutation (sign-flip, exact ≤18 else one MC stream) ≈ t ≈ bootstrap;
  `paired_t_test`==`scipy.ttest_rel` <1e-9 (twin). Multiple comparisons: the closest NDCG pair stays
  not-sig under every correction while the genuine pairs survive (no binary FLIP exists on this corpus —
  the honest claim is the correction CEMENTS the marginal pair, 0.107→0.32 Bonferroni, not that it flips a
  verdict). **Brier decomposition gotcha:** the 3-term `rel−res+unc` identity is exact only for the
  BIN-QUANTIZED Brier (each forecast → its bin mean), NOT the raw Brier — assert against `brier_binned`,
  report `brier` separately (binning loses resolution so `brier_binned ≥ brier`, but don't assert that sign
  universally). **Calibration:** raw cosine/MaxSim are wildly over-confident (dense/late ECE 0.43/0.39),
  the reason RRF fuses RANKS not scores; Platt (strictly monotone, a>0) preserves ranking/AUC/NDCG EXACTLY
  (<1e-12 — the orthogonality backbone), isotonic (PAVA) lowers ECE more but can tie. **Per-query
  normalization does NOT uniformly reduce ECE** (lexical it INCREASES) — don't headline it; the robust
  claims are dense/late ECE>0.1 + all-leg MCE>0.05 (lexical bulk-calibrated by its BM25 zero-mass, bad in
  tail) and recalibration lowers ECE, guarded by a held-out split. Calibration is a POOLED cross-query
  object (10 docs/query too sparse per-query); quantile bins so BM25's zero-mass doesn't empty a bin. AUC =
  hand Mann–Whitney via `scipy.stats.rankdata`; Platt via `scipy.optimize` + `scipy.special.expit` (avoid
  the hand-sigmoid overflow gemini flags). **Drift:** synthesize a time axis (no temporal dim) — Gaussian
  noise σ on the dense pooled vectors, oracle grades FIXED (model decay); KS hand staircase-sup ==
  `scipy.ks_2samp` <1e-12; PSI = symmetrized KL (Jeffreys) `=KL(p‖q)+KL(q‖p)` exact on shared guarded
  proportions. **PSI smoothing gotcha:** an EPS-on-proportions floor explodes PSI to ~13 at severe shift
  (small-n + empty-bin artifact) — use additive **Laplace smoothing (alpha=0.5)** for realistic finance-
  range PSI and a clean 0.25 crossing; bins=5 on 40 queries (10 needs large n, the rigorFlag). **Null must
  be matched-n** (bootstrap-resample Q-vs-Q, NOT a 20/20 split) and AVERAGED over seeds (one small-sample
  draw is noisy — itself the PSI-threshold rigorFlag); KS is the binning-free rigorous detector. **Silent-
  decay headline reframed:** "distribution beats mean" is hard at n=40 (KS needs D>0.30, which moves the
  mean too), so make it "drift monitoring must be PAIRED" — a small uniform decay overlapping unpaired CIs
  is caught by the paired test (p≈0.002), tying drift back to the significance pillar; scan small σ for the
  first qualifying level (deterministic). **Input-vs-outcome:** covariate shift (re-weight query mix, model
  fixed) fires input-PSI while the paired outcome on a FIXED query set is EXACTLY 0 (same model) — so input
  drift alone can't diagnose decay; compute the contrast on the fixed set, not the windowed distributions
  (which both shift). Graph prereqs re-pointed to **ndcg-discount-geometry only** (dropped unbuilt
  `cross-encoders-reranking` + marginal `high-dimensional-geometry`, capstone precedent); name-not-link the
  unbuilt `conformal-factuality`/`selective-generation-abstention`. Up-links: formalstatistics
  `hypothesis-testing`/`confidence-intervals-and-duality`/`bootstrap`/`multiple-testing-and-false-discovery`,
  formalml `concentration-inequalities`/`kl-divergence`/`always-valid-inference`/`conformal-prediction`,
  formalcalculus `riemann-integral`/`radon-nikodym` (all confirmed via `ls`). Refs verified: Smucker–Allan–
  Carterette CIKM 2007 `10.1145/1321440.1321528`; Good (permutation/bootstrap, clean DOI beats Noreen's
  none) `10.1007/b138696`; Niculescu-Mizil–Caruana ICML 2005 `10.1145/1102351.1102430`; Zadrozny–Elkan KDD
  2002 `10.1145/775047.775151`; Gama et al. CSUR 2014 `10.1145/2523813`; Massey KS JASA 1951
  `10.1080/01621459.1951.10500769`; Guo et al. ICML 2017 PMLR v70 (no DOI).
  Its **LLM-as-judge successor** (`llm-as-judge-ragas`): carries metrics-as-estimators into the GENERATION
  layer via a **synthetic noisy judge over the oracle** (the vMF-oracle move applied to "relevance" once
  more) — faithfulness = mean of Bernoulli verdicts = a BIASED estimator of latent prevalence π;
  Rogan–Gladen `π̂=(p_obs+sp−1)/(se+sp−1)` debiases, variance `∝1/J²` (inverse-square Youden `J=se+sp−1`)
  explodes as J→0 and is **undefined below the Youden line**. IMPORT significance-testing-calibration's whole
  calibration suite (`reliability_diagram`/`expected_calibration_error`/`platt_scale`/`apply_platt`/
  `isotonic_calibrate`/`apply_isotonic`/`auc_pooled`/`brier_score`) + `paired_t_test`/`permutation_test` +
  `get_corpus`, and set-metrics `precision_at_k` (the collapse anchor: a **perfect judge's faithfulness ==
  imported `precision_at_k`** <1e-12). **Rogan–Gladen unbiasedness is exact ONLY for a HOMOGENEOUS judge**
  (constant se/sp) and BEFORE the [0,1] clip — recovery/unbiasedness tests use a bias-free judge (all β=0);
  the **feature-shifted judges** (verbosity=token-dispersion z-score, position=rank shown, self=dense-leg
  membership) drive the bias/calibration/flip stories, where one audited se/sp imperfectly corrects
  heterogeneous bias. **κ-paradox needs FEATURE-SPREAD judges for a non-degenerate reliability diagram** — a
  homogeneous judge emits only 2 distinct confidences and quantile-binning collapses to one bin; Panel C uses
  lenient/balanced/strict (all β>0), with the bias-free judge ONLY as the swap-test control. Build+RUN the
  paradox: equal `p_o=0.85` → κ `0.70` vs `0.318`, AC1 stable `0.70`/`0.81` (assert **AC1 spread ≪ κ spread**,
  not equality). The ICC assertion is the **algebraic SS identity** `SS_total=SS_q+SS_j+SS_e` (exact <1e-9),
  not a variance-component reconstruction; ICC(2,1) is the two-way-random absolute-agreement form; the
  **judge-variance floor** `σ²_j/J` survives Q→∞, and the budget lever (more judges beats more queries) is a
  **precision/variance** statement needing genuine judge heterogeneity (σ²_j>0). **Swap test = paired
  first-slot(+0.5) vs last-slot(−0.5) confidence diff** — indexing the early/late position term by each
  claim's *actual* rank gives the WRONG sign (build-and-run caught a negative bias); reuses
  `paired_t_test`/`permutation_test`, β_pos=0 control does NOT reject. Ranking flip used the **constructed
  two-system toy** (legs inflate monotonically → no natural reorder; NDCG `constructed_*_flip` precedent).
  Dawid–Skene EM recovers planted per-judge se/sp with NO gold labels (±0.06), **align hard labels to majority
  vote** to fix the label-permutation symmetry (non-convex likelihood, majority-vote init); `cohen_kappa` twin
  vs `sklearn.metrics.cohen_kappa_score` (<1e-9; sklearn already a dep via JL/matryoshka). **Forward
  connections to UNBUILT topics must NOT sit in frontmatter `connections[]`** — `pnpm validate` ERRORS on them
  (unlike `curriculum-graph.json` roadmap nodes, which are notices); name `faithfulness-groundedness`/
  `conformal-factuality` in prose only. Graph: prereq edge **re-pointed** `set-metrics→` ⇒
  `significance-testing-calibration→` (heavy calibration reuse; set-metrics/ndcg stay transitive ancestors +
  `connections`). No formalML EM slug exists → Dawid–Skene EM up-links to **formalstatistics
  `maximum-likelihood`**. Refs verified: RAGAS Es et al. EACL 2024 `10.18653/v1/2024.eacl-demo.16` (arXiv
  2309.15217); Zheng et al. LLM-as-judge NeurIPS 2023 arXiv 2306.05685; G-Eval EMNLP 2023 arXiv 2303.16634;
  Cohen 1960 `10.1177/001316446002000104`; Feinstein–Cicchetti 1990 `10.1016/0895-4356(90)90158-L`;
  Rogan–Gladen 1978 `10.1093/oxfordjournals.aje.a112510`; Dawid–Skene 1979 `10.2307/2346806`; Shrout–Fleiss
  1979 `10.1037/0033-2909.86.2.420`; **Gwet 2008 `10.1348/000711006X126600`** (the `…2044-8317…` DOI 404s);
  Hayes–Krippendorff 2007 `10.1080/19312450709336664`.
  Its **conformal-factuality successor** (`conformal-factuality`): the TERMINUS of the eval layer — turns
  the calibrated judge confidence into a distribution-free GUARANTEE. REUSE (don't re-prove) formalML
  `conformal-prediction` Theorem 1 (split-conformal coverage, `⌈(1−α)(n+1)⌉`-th order-stat threshold);
  DEVELOP FRESH **Conformal Risk Control** (Angelopoulos et al. 2024, the monotone-loss generalization
  formalML lacks). IMPORT `get_corpus`+the calibration suite (`platt_scale`/`apply_platt`/`isotonic_*`/
  `expected_calibration_error`/`auc_pooled`) from significance-testing-calibration, and `K`/`judge_confidence`/
  `oracle_faithfulness`/`doc_length_feature`/`candidate_ids`/`JUDGE_PERFECT` from llm-as-judge-ragas; the
  **collapse anchor** is a perfect judge's back-off retained fraction == imported `precision_at_k` <1e-12
  (lifts the prereq's mean-verdict anchor to the conformal retained set). **The score is `s=1−c̃` (1 − calibrated
  confidence); orientation is load-bearing** (`s=c̃` retains the least-confident). **Conformal VALIDITY is
  calibration-agnostic** — coverage holds for ANY score; recalibration buys EFFICIENCY (retention), not
  validity (the cleanest framing of the prereq→topic edge). **Judge must OVERLAP or every trade-off is
  vacuous:** JUDGE_BALANCED separates faithful/unfaithful confidence PERFECTLY on this corpus (base
  sens/fpr logit gap 3.17 > feature-shift span ~2.35 → AUC 1.0, conf collapses to {0,1}, split_false≡0,
  CRC has nothing to control). Use a custom **lenient** judge `dict(sens0=0.80,spec0=0.62,b_len=1.3,b_pos=0.7,
  b_self=0.6)` (AUC≈0.90, classes overlap, false claims leak — split_false runs to 0.23@α=0.02, the
  recall≠precision bridge). **CRC loss MUST be the fixed-denominator `L=(1/k)·#{retained∧unfaithful}`** (each
  indicator only switches off as the cut rises → non-increasing, B=1); the **fraction-of-retained loss is
  NON-monotone** (denominator shrinks: 0.1→1.0 as the cut rises) and silently voids CRC — reproduce the
  counterexample numerically before adopting the monotone one. CRC controls `E[L]≤α` in EXPECTATION not a
  single realization (0.205@α=0.20 overshoot is honest; RCPS Bates et al. 2021 is the δ-level alternative).
  **Drift = TRUE covariate shift, not symmetric logit noise.** Mean-zero noise on test-half confidence
  logits is concept drift (added variance) — importance weighting can't repair it (build-and-run gave
  break_sigma=None, weighted +0.02). The textbook Tibshirani result needs a covariate shift with a KNOWN
  likelihood ratio: tilt a real covariate (`doc_length_feature`) by `w=exp(−βv)`, resample the test set ∝ w;
  split coverage collapses 0.95→0.27 as β grows, weighted conformal restores it (0.93@β=1, →1.0). **The
  weighted twin is EXACT:** uniform weights + `w_test=mean(w)` ⇒ `pᵢ=1/(n+1)` ⇒ weighted quantile ==
  `s[⌈(1−α)(n+1)⌉−1]` (split conformal) <1e-12. Frontmatter `prerequisites` = BOTH
  `[llm-as-judge-ragas, significance-testing-calibration]` (matches the two pre-existing graph edges — no
  edge add/re-point, just status planned→published + empty the retrieval-evaluation `planned[]`). Cross-site:
  `formalmlPrereqs` conformal-prediction/concentration-inequalities, `formalmlConnections` always-valid-inference,
  `formalstatisticsPrereqs` hypothesis-testing/confidence-intervals-and-duality, `formalstatisticsConnections`
  multiple-testing-and-false-discovery (a conformal set is an inverted test / distribution-free CI; CRC↔FDR
  via Learn-Then-Test). TheoremBlock supports `type="algorithm"|"corollary"|"remark"` (not just def/thm/prop).
  Refs verified: CRC arXiv 2208.02814; Mohri–Hashimoto (per-claim back-off) arXiv 2402.10978; C-RAG (Kang et al.)
  arXiv 2402.03181; Tibshirani et al. covariate-shift arXiv 1904.06019; RCPS Bates et al. `10.1145/3478535`;
  Angelopoulos–Bates gentle intro arXiv 2107.07511; Lei et al. (split-conformal) JASA `10.1080/01621459.2017.1307116`.
  Its **information-theory-keystone successor** (`pmi-retrieval-value`): the ROOT of the rag-information-theory
  layer (entry from the eval layer; unblocks BOTH `faithfulness-groundedness` and `retriever-as-noisy-channel`,
  whose DAG edges already exist — ship = node `planned→published` + drop the title from `curriculum.ts`
  tracks[8].planned, no edge add). Measures "what retrieval adds, in bits" via a SYNTHETIC ANSWER MODEL over
  the imported dense-retrieval finance geometry (`dpr_finance_matrix`: 4 sectors × 2 companies, one filing per
  company → the company filing IS the answer prototype). **The prior MUST be the RAG marginal**
  `p(a|q)=Σ_d p(d|q)p(a|q,d)` (Lewis et al. 2020), NOT a free softmax — only then do the three MI forms
  (joint-sum, expected-KL, entropy-reduction `H(A|Q)−H(A|Q,D)`) agree (measured 0 to printed precision; a free
  prior gave 0.887 vs 0.957). Build the hot path as `prior = pdq @ post` and pin `answer_prior` to it with a
  `<1e-12` test (keeps the user-written canonical def consistent without a Gemini "unused" flag). **REUSE the
  prereq geometry but BUILD a topic-specific query set** — `dpr_finance_matrix`'s own queries are κ=350
  company-tight → prior entropy ≈0 → the bits headline goes vacuous; draw SECTOR-ambiguous queries around the
  sector mean at κ_query≈30 (the tuned-query exception the contract allows), gold = nearest in-sector company,
  `test_prior_genuinely_uncertain` (H(A|Q)>0.5 bit) guards it. `TAU=TAU_DOC=0.2` (additive-logits posterior
  `softmax((⟨q,μ_a⟩+⟨d,μ_a⟩)/τ)`); τ≳0.5 flips the distractor sign positive. **PROP 1 (the headline):** a
  relevant filing gives `pmi(a*;d|q)>0`, a same-sector distractor `<0` (costs bits at the truth) while its
  `KL(post‖prior)>0` — assert the sign split + margin, not decimals (mean_rel +0.98 > 0 > mean_distr −1.11,
  96.9% negative). **InfoNCE COR gotcha — assert ONLY the ceiling/saturation, never "bound ≤ measured MI":** the
  empirical `log(N+1)−L` from a finite batch can EXCEED any single MI estimate (bound 1.84 > retrieval-channel
  I(Q;D) 0.82 here) and the two are different MIs — so the robust, load-bearing claim is `bound ≤ log(N+1)` (the
  CLAUDE.md InfoNCE rigorFlag), shown as a bound curve rising to its rising ceiling; import `info_nce_loss_batch`,
  never reimplement. **Bits ≠ recall via SATURATED recall, NOT a KL-movement ordering:** on this easy corpus the
  gold is always rank-1 so `recall@k≡1.0`/`MAP≡1.0` (import `recall_at_k`/`average_precision` from set-metrics to
  show it) while per-query bits `H(A|Q)−H(A|Q,D)` vary 0.68–1.60 — assert recall_min==recall_max==1 AND
  bits_max−bits_min>0.5. (An alternative "max-KL doc ≠ top-ranked doc" framing is muddy — KL belief-movement
  rewards a surprising WRONG far-sector doc — so don't headline it.) **Saturation (PROP 2):** belief movement
  KL(new‖old) of a 2nd IDENTICAL filing (0.09) ≪ standalone (0.67) ≈ a 2nd DIFFERENT filing (0.44); chain rule
  cited, demonstrated not proven; forward-edge `context-selection-submodular-dpp` named in prose only (unbuilt).
  Cross-site (all slugs `ls`-verified): `formalmlPrereqs` shannon-entropy + kl-divergence; `formalcalculusPrereqs`
  radon-nikodym (pmi = log of an RN derivative) + riemann-integral; `formalmlConnections` rate-distortion +
  information-bottleneck; `formalstatisticsConnections` exponential-families + maximum-likelihood. formalML has NO
  `mutual-information`/`channel-capacity` slug → name in prose. In-site `connections[]` (all published):
  dense-retrieval, set-metrics, infonce-contrastive-objective, probability-ranking-principle,
  query-likelihood-language-models, capstone. Viz Panel A bakes the worked query's full 8×8 POST_ALL so the
  doc-slider scans every candidate; TS recomputes entropy/pmi/histogram bins closed-form. Refs verified (`curl -sI`):
  Shannon 1948 `10.1002/j.1538-7305.1948.tb01338.x` (→IEEE Xplore, BSTJ 27); Cover–Thomas 2006 `10.1002/047174882X`;
  Church–Hanks 1990 PMI-origin is PRE-DOI → `https://aclanthology.org/J90-1003/` (the guessed `10.1162/...` 404s);
  van den Oord CPC arXiv 1807.03748; Poole et al. (variational MI bounds) arXiv 1905.06922; Lewis et al. RAG
  arXiv 2005.11401; MacKay 2003 (Fano, for the noisy-channel forward edge) inference.org.uk/itila.
  Its **noisy-channel successor** (`retriever-as-noisy-channel`): reads the PMI answer model as a COMMUNICATION
  CHANNEL — IMPORT `query_distributions`/`answer_posterior`/`entropy`/`kl`/`cond_mi_breakdown`/`_distractor_id`/
  `_corpus` (never reimplement); ship = node `planned→published` + drop the title from `curriculum.ts` the
  rag-information-theory track's `planned[]` (NO DAG edge changes — the `pmi→noisy-channel→context-selection`
  edges already exist; the unblocked successor is `context-selection-submodular-dpp`, NOT faithfulness — that's a
  separate pmi→faithfulness path). **The PMI corpus is "too easy" for a naive channel story and FALSIFIES the
  obvious headlines — build-and-run before writing:** (1) the QUERY ALONE identifies the gold company (queries
  drawn near the gold prototype) and (2) the GOLD FILING IS RANK-1 for all 32 queries. So a noise-MIX toward
  uniform on `p(d|q)` *ADDS* MI (it raises cross-document disagreement, which IS I(A;D|Q)) — wrong direction; and
  the marginal RAG decoder is PERFECT (recall@k≡1, no error to bound). **Use the right two channel models, mapped
  to the subtitle:** RECALL=ERASURE (w.p. recall read the context, w.p. 1−recall erase to a NON-INFORMATIVE
  UNIFORM belief — the RAG premise) gives the clean monotone story AND the exact BEC identity `I_ε = recall·I₀`
  (robust to ANY Q-measurable fallback: conditioned on Q the fallback carries 0 conditional MI, so even a
  query-only fallback satisfies it — but uniform-on-erasure is the one whose ERROR rises, since a query-only
  fallback is itself a good decoder here and would make erasure HELP). PRECISION=SUBSTITUTION (swap gold→same-
  sector distractor w.p. ε) gives realized error 0→1. **The single rigorous truth: Fano bounds the model's BAYES
  error `E[1−max p]`, NEVER the realized error directly** — realized-vs-Bayes IS the calibration gap (the load-
  bearing rigorFlag + the bridge to significance-testing-calibration/faithfulness). **Fano floor `(H−1)/log₂K` is
  VACUOUS below 1 bit** (well-retrieving channel) — at the clean point H(A|Q,D)=0.876<1 so the floor is 0; it
  only activates once the residual crosses 1 bit (erasure: at recall≈0.92). So the demo MUST degrade the channel;
  don't headline a floor at the clean operating point. **Confident-wrong is the precision headline:** substitution
  drives realized→1 while H(A|Q,D) stays LOW (0.60→0.87, the distractor posterior is just as sharp, only wrong),
  so the Fano floor stays pinned at 0 — an entropy bound is BLIND to confident contamination. Assert the CONTRASTS
  (BEC `I=recall·I₀` <1e-12; collapse-to-PMI bit-identical at recall=1; Fano-is-a-theorem `bayes≥tight≥loose`; the
  `H(A|Q)=H(A|Q,D)+I` Jensen-gap identity; monotone degradation; floor-activates; gap-widens; recall@k saturated +
  precision=1/k + entropy rising toward H(A|Q)=2.06), not decimals; `viz_constants()` owns the numbers (incl. a
  WORKED_BELIEF the lab erodes toward uniform — add it and re-run rather than borrowing the PMI lab's PRIOR). TS
  recomputes binary-entropy/Fano-loose/Fano-tight(bisection)/BEC-BSC-capacity closed-form. Cross-site (all `ls`/
  `curl`-verified): `formalmlPrereqs` shannon-entropy+kl-divergence; `formalmlConnections` rate-distortion+
  information-bottleneck (formalML has NO channel-capacity/mutual-information slug → name Fano/capacity/the
  noisy-channel theorem in prose); `formalcalculusPrereqs` radon-nikodym+riemann-integral; `formalstatistics
  Connections` hypothesis-testing (Fano=the multi-hypothesis testing converse) + maximum-likelihood (the MAP
  decoder). NEW ref verified (`curl -sL` CSL): Berger–Lafferty "Information Retrieval as Statistical Translation"
  SIGIR 1999 `10.1145/312624.312681` (the historical IR-as-noisy-channel root); Fano 1961 "Transmission of
  Information" MIT Press is a pre-DOI book (cite url-less). Gemini posted a clean review (retracted its own one
  comment that falsely claimed `entropy`/`kl` were redefined — they're imported). GOTCHA: a persisted shell `cd`
  into a notebook dir broke a later relative-path generator write — use ABSOLUTE paths in throwaway generators.
  Its **long-context successor** (`retrieval-vs-long-context`): the ROOT of the generation-grounding layer;
  ship = node `planned→published` + drop the title from `curriculum.ts` generation-grounding `planned[]` (NO
  DAG edge changes — the `the-retrieval-problem→` and `→context-selection-submodular-dpp` edges already exist).
  Single frontmatter prereq stays **`the-retrieval-problem`** (import-graph≠DAG): the `.py` IMPORTS the deep
  chain (dense `dual_encoder_score`/`dpr_finance_matrix`, pmi `answer_posterior`/`answer_posterior_two`/
  `saturation_table`, noisy-channel `bayes_error`, set-metrics `recall_at_k`/`precision_at_k`) only to SOURCE
  numbers; the rest are `connections[]` (the-retrieval-problem/retriever-as-noisy-channel/pmi-retrieval-value/
  set-metrics/dense-retrieval-dual-encoders, all have MDX). **The interior optimum is a build-and-run TRAP —
  ship the MONOTONE fallback.** A finite **attention budget** (softmax weights Σ=1) over a single-prototype-
  per-company geometry makes extra relevant passages REDUNDANT (the imported `saturation_table`: redundant≪
  novel), so a unit-budget AVERAGE has a FLAT left arm — no rising "more relevant helps" arm. A genuine
  interior `k*>1` needs IMPERFECT retrieval (recall@1<1, κ_query≈15) AND is **seed-fragile** (rise +0.002–0.028,
  flips to k*=1 on seeds 1/21). So the ROBUST, seed-independent headline is **monotone "more context is not
  better"**: at GOOD retrieval (κ_query=60, recall@1=1.0) Q(k) peaks at k=1 and declines (k*=1 every seed,
  Q 0.71→0.51); the interior optimum is a poor-retrieval REMARK shown as a dashed contrast, never asserted. The
  multi-doc posterior `softmax((⟨q,μ_a⟩+Σ_j w_j⟨d_j,μ_a⟩)/τ)` GENERALIZES `answer_posterior_two`: collapse
  anchors are `answer_posterior` at k=1,w=[1] and `answer_posterior_two` at k=2,w=[1,1] (<1e-12) — keep `weights`
  an EXPLICIT arg so degenerate weights recover the imports. Use a SEPARATE **`TAU_GEN`≈0.3** for the curves
  (the imported `TAU`=0.2 saturates one passage); and a SOFT **`τ_attn`≈0.45** — relevance-aware attention with
  good retrieval gives distractors ~0 weight (no dilution, no decline), so the budget must spread for distractors
  to bite. Two-mechanism decline: **redundancy** (imported saturation, flat while precision≈1) + **dilution**
  (distractors steal budget, precision↓, entropy H(A|ctx)↑). **conf≡Q under good retrieval** — the confident-wrong
  gap is ZERO (the honest CONTRAST with noisy-channel: dilution raises ENTROPY, not confident-wrongness), so bake
  rising **entropy** as the distortion axis, not a gap. **"Quality tracks precision, not recall":** recall_R(1)=
  1/R<1<recall_R(n)=1 climbs by SET coverage even when recall@1=1.0, so the |R|>1 set escapes "too easy"
  independent of retrieval difficulty. Attention cost `(kL)²` is the THEOREM (`cost(2k)/cost(k)=4`); FlashAttention
  is O(n) **MEMORY** not FLOPs — the load-bearing rigorFlag. Positional U = EMPIRICAL (Liu et al.) baked as a soft
  erasure (BEC per rank); the Q-dip is SHALLOW (gold is 1 of n_ctx), so plot the dramatic weight-U + the subtle Q
  consequence. Rate-distortion verdict: focused retrieval Pareto-dominates stuffing (`cost(n)≫cost(k*)`, `Q(n)≤Q*`);
  forward edge `context-selection-submodular-dpp` PROSE-ONLY (unbuilt). **TS GOTCHA:** a panel's local SVG height
  `const H` SHADOWS a module-level entropy array named `H` → `poly(H,…)` passed the number 232; rename the array
  (`ENTROPY`) — a real `tsc` catch. Cross-site (all `ls`/`curl`-verified): `formalmlPrereqs` shannon-entropy+
  kl-divergence; `formalmlConnections` rate-distortion+information-bottleneck (NO formalML attention/transformer
  slug → Θ(n²) prose+Vaswani only); `formalcalculusPrereqs` mean-value-taylor (interior-optimum 1st/2nd-order test);
  `formalcalculusConnections` convex-optimization+riemann-integral; `formalstatisticsConnections` exponential-families
  (the vMF-softmax answer model). Refs verified: Liu et al. "Lost in the Middle" TACL 2024 `10.1162/tacl_a_00638`
  (arXiv 2307.03172); Vaswani "Attention Is All You Need" arXiv 1706.03762; Dao et al. FlashAttention arXiv
  2205.14135 (the memory caveat); Lewis RAG 2005.11401; Shannon/Cover–Thomas reused.
  Its **context-selection successor** (`context-selection-submodular-dpp`): the SHOWPIECE of the
  rag-information-theory layer; ship = node `planned→published` + drop the title from `curriculum.ts`
  the layer's `planned[]` (NO DAG edge changes — the THREE prereq edges + the outbound
  `→multi-hop-iterative-retrieval` edge already exist; the unblocked successor is multi-hop-iterative-retrieval).
  **Frontmatter `prerequisites` = the THREE graph edges** (retriever-as-noisy-channel/chunking-as-segmentation/
  retrieval-vs-long-context), matching the pre-existing edges — do NOT over-apply the rvlc single-prereq
  precedent (that held only because rvlc had ONE inbound edge; this node has three). The `.py` IMPORTS
  `rvlc_corpus`/`answer_posterior_topk` (the bridge), pmi `saturation_table`/`answer_posterior`/`entropy`,
  set-metrics `recall_at_k`, vmf `normalize`/`sample_vmf`, dense `dual_encoder_score`; `chunking-as-segmentation`
  is **NOT imported** (its `mean_resultant_length(emb,i,j)` collides with the vMF closed form) — conceptual
  prereq only. **The `sim≥0` convention is load-bearing for facility-location submodularity** — use a SEPARATE
  `cov_sim_matrix` = clipped cosine `max(0,cos)` for facility/MMR (monotone + the `(s_e−m)⁺` hinge needs
  nonneg), keep raw cosine `sim_matrix` (the Gram) for the DPP volume geometry. **The easy-corpus trap is
  real:** `rvlc_corpus` is "too easy" (top-k precision≈1, no room for diversity on answer quality) — build a
  topic-specific `selection_corpus` with the redundancy trap: `N_GENERIC` sector-generic near-duplicates
  (ambiguous gold/peer, HIGHEST relevance, the cluster top-k wastes the budget on) + ONE lower-relevance
  **disambiguator** near the gold prototype (the only passage that resolves the answer) + distractors; query
  drawn only MILDLY toward gold (`gold_mix≈0.12`) so generics OUT-RANK the disambiguator (top-k skips it) yet A
  is the answer. **DROP a `b_specific`** — including the peer's disambiguator lets diversity get tricked into
  re-ambiguity. Use `PAYOFF_TAU≈0.15` (sharper than the imported 0.30) so the answer is decisive; read every
  method through the imported `answer_posterior_topk` with an EQUAL budget so any Q delta is attributable to
  selection alone. **Headline PINNED** (build-and-run): Q topk 0.38 < mmr 0.44 < facility 0.54 ≈ dpp 0.55;
  **entropy RISES for diversity** (it spreads mass onto the truth across MORE companies), so Q (mass on truth)
  is the headline, NOT entropy — don't assert entropy drops (a wrong-signed first guess). **Info gain is NOT
  submodular even on the corpus** (witness ≈ −0.70 vs facility ≈ 0) — assert the VIOLATION + the constructed
  XOR proof (Δ(D2|∅)=0 < Δ(D2|{D1})=1, supermodular), NOT submodularity; this is the honesty hinge (info gain
  submodular only under conditional independence, Krause–Guestrin). **DPP greedy MAP maximizes the MONOTONE
  surrogate `log det(I+L_S)`** (raw `log det(L_S)` is non-monotone + −∞ on singular near-duplicates; ridge
  guards the raw slogdet for the geometry asserts) — ONE `greedy_select` reused by facility location AND the
  log-det DPP; lazy-greedy (Minoux) == standard (the speedup anchor). **NWF panel needs a CONSTRUCTED
  worst-case** — greedy == OPT on the smooth finance pool (ratio 1.0, vacuous), so add a rectangular 0/1
  max-coverage instance (A/B/C over 12 facets) where greedy 9 < OPT 12 at k=2, both above the 0.632·OPT floor
  7.59, to make the bound a VISIBLE gap. `det(S_S)=Vol²=∏σ²` (Gram identity); factorization
  `det(L_S)=(∏q²)det(S_S)` asserted <1e-9. Collapse anchors: greedy@1=argmax, MMR(λ=1)=top-k, DPP@1=max-quality,
  degenerate-weight `answer_posterior_topk`==imported `answer_posterior` <1e-12. **Repels-duplicates test must
  use a NEAR-duplicate distinct index** (not the same index twice — the `I+L` surrogate gives a small positive
  gain for a repeated index); unit-quality `L` isolates the pure volume claim. `pipelineStage: 'select'`
  introduced (was unused). Cross-site (all `ls`-verified): `formalmlPrereqs` shannon-entropy+kl-divergence;
  `formalmlConnections` svd (det=volume=∏σ²)+information-bottleneck; `formalcalculusConnections`
  convex-optimization (multilinear/Lovász extension)+riemann-integral (NO `formalcalculusPrereqs` — the
  mean-value-taylor/interior-optimum link was rvlc's, not load-bearing here); `formalstatisticsConnections`
  exponential-families. Refs verified (`curl -sI`): Nemhauser–Wolsey–Fisher 1978 `10.1007/BF01588971`;
  Nemhauser–Wolsey (best-algorithms/oracle bound) `10.1287/moor.3.3.177`; Carbonell–Goldstein MMR SIGIR 1998
  `10.1145/290941.291025`; Kulesza–Taskar DPP FnTML 2012 `10.1561/2200000044` (arXiv 1207.6083); Krause–Guestrin
  UAI 2005 arXiv 1207.1394; Minoux 1978 `10.1007/BFb0006528`; Lin–Bilmes ACL 2011 `aclanthology.org/P11-1052`;
  Chen et al. NeurIPS 2018 arXiv 1709.05135; Feige (1−1/e tight) JACM 1998 `10.1145/285055.285059`; Cover–Thomas reused.
  Its **multi-hop successor** (`multi-hop-iterative-retrieval`): the FIRST topic where retrieval becomes a
  SEARCH (single frontmatter prereq = the one graph edge `context-selection-submodular-dpp`; ship = node
  `planned→published` + drop the title from `curriculum.ts` rag-information-theory `planned[]`, NO edge changes —
  the `context-selection→multi-hop→graphrag` edges already exist; unblocks `graphrag-community-detection`). The
  `.py` IMPORTS the chain (vmf `normalize`/`sample_vmf`, dense `dual_encoder_score`, pmi `answer_posterior`/
  `entropy`/`kl`, set-metrics `recall_at_k`, rvlc `rvlc_corpus`/`answer_posterior_topk`, context-selection
  `greedy_select`/`info_gain_fn`/`info_gain_xor_witness`/`submodularity_witness`) **plus the CAPSTONE as a
  SIBLING numeric source** for the FKG/over-fetch laws (`cascade_recall`/`over_fetch_factor`/`dependence_sweep`) —
  capstone stays in `connections[]`, NOT `prerequisites` (import graph ≠ DAG); importing it is cheap (its corpus is
  lazily cached behind `_corpus()`). **The corpus is the "mention" geometry, NOT a pure geodesic** (my first design):
  a bridge filing of company X that names Y is `cos α·u_X + sin α·u_Y`, and `reformulate(q,d)=normalize(d−⟨d,q⟩q)`
  extracts EXACTLY the `sin α` mention → points the next retrieval at Y. **Build-and-run TUNING crux** (all from a
  `_diagnostics()` sweep, never guessed): (1) `KAPPA_NODE=200` is FAR too loose at d=32 (orthogonal noise ~0.4 swamps
  the mention) → use `κ≈8000` (cosine 0.996) so reformulation is clean; (2) **private per-chain companies** (drawn
  around distinct sector means, NOT shared rvlc protos) — sharing a company as a pure-answer in one chain AND a
  bridge-source in another breaks the loop (the pure-company passages outrank the bridge); (3) `α=40°` not 45/50 —
  the bridge cosine `cos α` must clear the WORST same-sector distractor (`max off-diag` company cosine), which at
  `KAPPA_COMPANY=18` is ~0.53 < cos40=0.766 (cos45=0.707 was too close, recall dropped); (4) `TAU_HOP=0.35` (~2σ
  above the d=32 equatorial noise `1/√d`). **The stopping rule is RESIDUAL-based, NOT belief-movement** — and this
  IS the supermodular climax made operational: the KL belief-movement is TINY at the bridge (0.08) and HUGE at the
  answer (9.2), so a myopic "stop when belief stops moving" rule would halt at the worthless-looking bridge and never
  reach the answer. Stop instead when the read filing opens no new direction (`‖d−⟨d,q⟩q‖ < REFORM_EPS`); the
  bridge/terminal residual gap is empirical ([0.53,0.79] vs [0.05,0.43]) so `eps=0.47` sits in it (tuned,
  seed-dependent — the rigorFlag; `hops_taken` PINNED). **The vMF answer model does NOT give a clean numeric XOR**
  (the answer doc aligns with its own company regardless of the bridge), so carry the supermodular climax via the
  IMPORTED `info_gain_xor_witness` (exact general proof, violation=1.0) + the OPERATIONAL fact the corpus DOES
  exhibit: `answer_in_single_pool=0` (single-shot can't reach the near-orthogonal answer) vs reachable after one
  reformulation — do NOT try to force a vMF Δ(ans|bridge)>Δ(ans|∅) numeric witness. **Measured per-hop recall is ~1**
  on the clean corpus → the FKG/compounding demo MUST use illustrative middling retentions (`DEMO_R1,R2=0.6,0.5`,
  the capstone's), or the product law/FKG is vacuous. Collapse anchors: 1-hop belief == imported `answer_posterior`
  <1e-12; single-hop recall == imported `recall_at_k`; greedy hop == imported `greedy_select` argmax; chain recall ==
  imported `cascade_recall`. `pipelineStage: 'retrieve'` (iterated retrieval). Viz Panel A is a 1-D "cosine to the
  current query" axis (bridge above τ, answer below, jumps to ~1 after reformulation) — recompute reachability `cos>τ`
  + `ρ^(1/k)` + `∏r` closed-form in TS; bake the cosines/recall/trajectory/FKG sweep. Cross-site (all `ls`/`curl`
  verified): `formalmlPrereqs` shannon-entropy+kl-divergence; `formalmlConnections` random-walks (a trajectory is a
  walk on the retrieval graph)+information-bottleneck+rate-distortion; `formalstatisticsConnections` hypothesis-testing
  (the SPRT stopping analogy)+exponential-families; `formalcalculusConnections` stability-dynamics (Bellman
  fixed-point/value-iteration contraction)+convex-optimization. NO sibling slug for MDP/Bellman/dynamic-programming/
  optimal-stopping/SPRT/Wald/martingale → name in prose. graphrag-community-detection UNBUILT → prose only, NOT in
  `connections[]`. Refs verified (`curl -sI`): HotpotQA `aclanthology.org/D18-1259`; MDR Xiong arXiv 2009.12756 (ICLR
  2021, no DOI); IRCoT `aclanthology.org/2023.acl-long.557`; Self-Ask (names "compositionality gap")
  `aclanthology.org/2023.findings-emnlp.378`; FLARE `aclanthology.org/2023.emnlp-main.495`; Self-RAG arXiv 2310.11511;
  Wald SPRT 1945 `10.1214/aoms/1177731118`; Bellman 1957 (book, no DOI); FKG 1971 `10.1007/BF01651330`; Cover–Thomas reused.
  Its **graphrag successor** (`graphrag-community-detection`, the TERMINUS of the rag-information-theory track — with it the
  track is COMPLETE, 0 planned): the topic where retrieval becomes a PARTITION, not a path. Single frontmatter prereq
  `multi-hop-iterative-retrieval` (the one inbound edge); ship = node `planned→published` + drop the title from
  `curriculum.ts` rag-information-theory `planned[]` (now `[]`), NO edge changes. `pipelineStage: index` (the community
  graph + summaries are precomputed OFFLINE, not a per-query retrieve). numpy/scipy ONLY (no networkx — the topic OWNS
  modularity/spectral-bipartition/Louvain/Leiden/SBM; `eigsh` matrix-free `LinearOperator` for `B=A−kkᵀ/2m` so a sparse
  SBM never densifies). IMPORTS dense `dpr_finance_matrix`/`normalize`/set-metrics `recall_at_k` for the geometry; builds
  the entity graph FRESH (a co-occurrence graph, not multi-hop's trajectory — do NOT import multi-hop). **Entity-graph
  tuning crux (build-and-run R4):** reuse `dpr_finance_matrix` BUT at **dim=64, κ_sector=200** (NOT the dense d=32/κ=60) so
  the 5 sector means are near-orthogonal → Leiden recovers the sectors at overlap 1.0 across ALL seeds; d=32 merged two
  sectors (overlap 0.75, only 4 communities). Edge weight `max(0, cos−0.15)`; at seed 7 the graph is connected with ~18
  cross-sector bridges. **SBM detectability recovery (R1, the headline surprise):** recover via B's MOST EXTREME
  eigenvector (`eigsh which='BE'`, then `argmax|λ|`), NOT largest-algebraic — a DISASSORTATIVE block (c_out>c_in, e.g.
  (1,7)) lives in B's most-NEGATIVE eigenvalue, and the plain adjacency leading vector localizes on high-degree nodes near
  threshold. The threshold is asymptotic, so ASSERT the clear above/below CONTRAST (n=3000: SNR 3.68→overlap 0.93,
  0.20→0.01) and in the grid sign test SKIP the finite-n smeared band `0.5≤SNR≤2.5`, never the exact crossing (the
  "asymptotic headline false at first scale" rule). **ΔQ twin:** `modularity_gain` is the FULL move
  `add_gain(target) − add_gain(current\{i})`, NOT the isolated-node add gain — the first build failed asserting the
  isolated form == definitional `Q(after)−Q(before)`. **Resolution limit:** assert DETERMINISTIC `Q(pairs)>Q(singles)` on a
  30-clique ring + the REVERSE on a 6-clique ring (√(2m)≈25.7 dwarfs clique-size-5), Louvain finds 15 of 30 — NOT a brute
  over 2ⁿ. The **γ slider must span 0.05→8** (merge→4 communities, a 0.3–4.0 plateau→5 sectors, fragment→25 singletons); a
  moderate γ range is FLAT on a clean 5-block graph and teaches nothing. **Louvain-disconnected witness (R2):** a
  disconnected community (two triangles, no inter-edge, one label) is a Louvain LOCAL OPTIMUM (`louvain_local_move`-stable,
  no single-node move improves Q) that `leiden_refine` splits into connected pieces — the HONEST "fixed-point" framing, NOT
  "Louvain produces a disconnected community generically" (assert full Louvain from singletons recovers the correct
  partition on the same graph; the connectivity GUARANTEE is what differs). Aggregation `H^T A H` preserves modularity
  (the multi-level soundness anchor). Cross-site (all `ls`/`curl` verified): `formalmlPrereqs` svd+spectral-theorem (the
  relaxed `Q=¼ₘ sᵀBs` Rayleigh quotient) + shannon-entropy+kl-divergence (the info-limit); `formalmlConnections`
  graph-laplacians (strongest — B is the modularity Laplacian, Fiedler/Cheeger)+clustering+random-walks;
  `formalstatisticsConnections` maximum-likelihood (SBM=latent-block MLE)+exponential-families+hypothesis-testing
  (structure-vs-ER null, the threshold = the no-power point); `formalcalculusConnections` eigenvalues-eigenvectors+
  convex-optimization (relax-and-round). formalML has NO spectral-clustering/mutual-information/markov-chains slug → name in
  prose. **Bundled** the `pmi-retrieval-value` + `embedding-dimension-lower-bounds` `draft→published` MDX flips (the "verify
  the flip landed at ship" gotcha — both graph nodes were already `published`, the MDX status stuck at `draft`, so they
  rendered at their URLs but were hidden from every listing). Refs verified (`curl -sI` + CSL): Newman–Girvan PRE 2004
  `10.1103/PhysRevE.69.026113`; Newman PNAS 2006 `10.1073/pnas.0601602103`; Blondel (Louvain) JSTAT 2008
  `10.1088/1742-5468/2008/10/P10008`; Traag (Leiden) SciRep 2019 `10.1038/s41598-019-41695-z`; Fortunato–Barthélemy PNAS
  2007 `10.1073/pnas.0605965104`; Decelle PRE 2011 `10.1103/PhysRevE.84.066106`; Massoulié STOC 2014
  `10.1145/2591796.2591857`; Mossel–Neeman–Sly PTRF `10.1007/s00440-014-0576-6`; Brandes "On Modularity Clustering" IEEE
  TKDE **2008** `10.1109/TKDE.2007.190689` (published 2008 despite the 2007 in the DOI suffix — confirm via CSL);
  Reichardt–Bornholdt PRE 2006 `10.1103/PhysRevE.74.016110`; Edge et al. GraphRAG arXiv 2404.16130; Cover–Thomas reused.
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
  medium-priority robustness/perf/a11y ones before merging. (The consumer `gemini-code-assist` app is
  being SUNSET — new org installs blocked 2026-06-18, all reviews cease 2026-07-17; after that the
  inline-review step won't run, so don't block a merge waiting on it.) It reliably flags **unguarded denominators**
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
  `itertools.combinations` loop, not once per pair. It also flags **biased / edge-cased numerical samplers**
  (a submodularity/Monte-Carlo witness restricted to `n≥k`, or permutation-cuts that can draw `A==B`) → prefer a
  **partition sampler**: pick the test element `e` first, then assign each remaining item to {A and B}/{B only}/
  {neither} (works for any `n≥1`, no degenerate cut). But
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
