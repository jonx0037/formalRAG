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
- **Verify reference DOIs** with `curl -sI https://doi.org/<doi>` — the `location:` header in the 302
  alone confirms journal/volume/issue/pages (a HEAD request: no redirect-following, no paywalled GET).
- The curriculum is the full 50-topic DAG in `src/data/curriculum.ts` + `curriculum-graph.json`;
  unauthored topics live in `tracks[].planned` and as `status: draft` MDX stubs.
- **Sync local `main` first.** `gh pr merge` updates *origin*, not local `main` — before branching each
  topic run `git fetch origin && git checkout main && git merge --ff-only origin/main`. The tell that
  you're on a stale base: a "published" topic shows as a draft stub with an orphaned `__pycache__/` and
  no `.py`/`.ipynb` (its source was merged on a branch you don't have yet).
- **Multiple topics in one session = feature branches off `main`.** They merge in any order *only if*
  each depends solely on already-published prereqs. If a batch topic lists a **sibling** as prereq (e.g.
  pseudo-relevance-feedback needs query-likelihood), sequence them: re-sync `main` only **after** its
  in-batch prereq merges, then branch the dependent off it. Each removes its title
  from a track's `planned[]` array, so the **2nd+ merge needs a trivial one-line `curriculum.ts`
  `planned[]` conflict resolution** (the `curriculum-graph.json` node-status flips auto-merge; but a
  DAG *edge* re-source is a real content edit — keep it on one branch). PRs also get an automated
  `gemini-code-assist` review — fetch its nits with `gh api repos/jonx0037/formalRAG/pulls/<n>/comments`
  (inline comments carry the severity badges; the `/reviews` body is often empty), and address the
  medium-priority robustness/perf/a11y ones before merging. It reliably flags **unguarded denominators**
  (`avgdl`, `|d|+μ`, query length, Σ-of-weights) and empty-collection cases in the notebook `.py` — add
  those guards up front. (`jupyter execute` does *not* write outputs back, so re-running to verify won't
  dirty the committed output-free `.ipynb`.)
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
