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
  Hydration tell-tales: the React JSX (buttons, list, *empty* `<svg>`) is in the SSR DOM, but
  D3-drawn `<rect>/<circle>` children and `katex.render()` output (a post-load `.katex` count bump)
  appear only *after* hydration — assert on those before clicking.
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
  fixed it, locked by `test_toy_local_optima` (global < near-miss < stuck).
- **Rotation/Procrustes transpose checkpoint:** the VQ/PQ track applies rotations as `(X - mu) @ R.T`
  with R's **rows** = basis vectors (`pca_align`/`balanced_rotation` in `product_quantization.py`). A
  learned-rotation step (OPQ's non-parametric Orthogonal Procrustes update) must therefore return
  `R = V @ U.T` from `SVD(Xc.T @ Q) = U Σ V.T`, so `apply_rotation` yields `Xc @ R.T = Xc @ (U V.T)` —
  the intended rotated data. The wrong transpose makes distortion **increase**, so a monotone-descent
  assert plus a cross-check against `scipy.linalg.orthogonal_procrustes` pin the orientation by
  construction — add a new test of each when introducing a learned rotation.
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
- **Sync local `main` first.** `gh pr merge` updates *origin*, not local `main` — before branching each
  topic run `git fetch origin && git checkout main && git merge --ff-only origin/main`. The tell that
  you're on a stale base: a "published" topic shows as a draft stub with an orphaned `__pycache__/` and
  no `.py`/`.ipynb` (its source was merged on a branch you don't have yet).
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
  (`avgdl`, `|d|+μ`, query length, Σ-of-weights) and empty-collection cases in the notebook `.py` — add
  those guards up front. In the viz `.tsx` it reliably flags **transient state-length mismatches** (a
  slider that grows `points` before the reset effect refreshes `assignments` → a crash on `C[labels[i]]`)
  and stale refs in d3 drag handlers — guard array-index lookups (`C[labels[i]]`, `colors[a[i] ?? 0]`)
  and compute drag-end distortion from live points/centroids, not a render-lagging ref. It also flags
  recall/`topk` denominators (`hits/(nq·topk)`), `np.argpartition(d, topk)` when `topk>n` (cap
  `topk=min(topk, n)`), and **tuple-arity mismatches in a fallback `return`** (a 6- vs 7-tuple path — a real
  HIGH-severity catch). It also flags **list-comprehension membership filters over sets** (→ native
  `s1.intersection(s2)` / `s1 - s2` — both snapshot, so an in-loop `del`/`discard` stays safe). But
  **decline with a posted rationale** the nits that would (a) break a byte-for-byte search twin — caching
  a beam's `worst` is an O(1) heap-peek, no gain, and diverges the twin from its `search_layer` source —
  or (b) SSR a KaTeX formula in a `client:visible` lab via `renderToString`: the island never SSRs, and
  every lab shares the `useEffect`+ref idiom (consistency beats a marginal CLS win). Gemini posts inline
  ~1–3 min after the push; `mergeable` flips to `UNKNOWN` transiently right then. (`jupyter execute` does
  *not* write outputs back, so re-running to verify won't dirty the output-free `.ipynb`.)
- Cross-link `learning-theory` does NOT exist as a formalML slug → use `vc-dimension` /
  `generalization-bounds`.

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
