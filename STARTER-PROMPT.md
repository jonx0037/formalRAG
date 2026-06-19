# STARTER-PROMPT.md — authoring a formalRAG topic

This is the **running playbook** for taking one curriculum topic from nothing to a shipped,
verified page. It is a living document: when a convention changes, update *this file* — do not
fork it per topic. Each topic copies the placeholder table, fills it in, and works the two phases
in order.

formalRAG is a rigor-first, systems-aware site on the mathematics of retrieval-augmented
generation (live at https://www.formalrag.com). Every topic gets **three pillars**: rigorous
KaTeX math, an interactive D3 viz, and a working Python notebook pillar. We author end-to-end in
Claude Code; project conventions and the do-NOT list live in [`CLAUDE.md`](CLAUDE.md), general
preferences (pnpm, uv, Chicago citations, American English, geometric-first, git workflow) in
`~/CLAUDE.md`. Prose is self-approved per standing authorization — flag only genuine objections.

---

## How to use this file

1. **Pick the next topic** from `src/data/curriculum.ts` / `src/data/curriculum-graph.json`. Confirm
   its prerequisites are published (or decide to backfill them first).
2. **Fill the placeholder table** below. Everything topic-specific lives there; the two phases are
   invariant across topics.
3. **Run Phase A** (draft & verify the math + notebook). Deliverables: a brief at
   `docs/plans/formalrag-<slug>-brief.md` and the verified notebook pair in `notebooks/<slug>/`.
4. **Run Phase B** (implement & ship the MDX + viz + graph + PR). Each pillar is verified against
   the notebook's numbers before it is considered done.
5. **Stop and ask** at any of the checkpoints flagged below.

### Placeholder table (fill per topic)

| Placeholder | Example | Where it goes |
|---|---|---|
| `{{TITLE}}` | `Rank Fusion: Reciprocal Rank Fusion and the Geometry of Rank Aggregation` | frontmatter `title` |
| `{{SUBTITLE}}` | one sentence on the core idea | frontmatter `subtitle` |
| `{{SLUG}}` | `rank-fusion-rrf` (lowercase, hyphenated, no suffix) | filename, `notebooks/<slug>/` |
| `{{DOMAIN}}` | `ranking-fusion` | frontmatter `domain` (one of the 10 below) |
| `{{DIFFICULTY}}` | `intermediate` | `foundational` \| `intermediate` \| `advanced` |
| `{{PIPELINE_STAGE}}` | `fuse` | `ingest` \| `index` \| `retrieve` \| `fuse` \| `rerank` \| `select` \| `generate` \| `evaluate` |
| `{{FINANCE_CASE_STUDY}}` | `true` | `financeCaseStudy` (bool) |
| `{{MODALITIES}}` | `[text, news]` | subset of `text` \| `pdf` \| `audio` \| `chart` \| `news` |
| `{{PREREQUISITES}}` | `[bm25-binary-independence-model]` | in-site topic slugs (must exist as MDX) |
| `{{RIGOR_FLAG}}` | the honest caveat (see Phase A) | frontmatter `rigorFlag` |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/bm25/` | shipped notebooks to model voice/layout on |

The ten domains: `retrieval-foundations`, `probabilistic-ir`, `embedding-geometry`, `ann-indexing`,
`vector-quantization`, `neural-retrieval`, `ranking-fusion`, `retrieval-evaluation`,
`generation-grounding`, `rag-information-theory`.

---

## Phase A — Draft & verify (math + notebook)

Produce the mathematics and the notebook pillar *before* touching MDX or viz. The notebook is the
source of truth that the later pillars are checked against.

### A1. The three-pillar contract

- **Math (KaTeX):** definitions, theorems with *full* proofs (never "it can be shown"), and a
  concrete motivating example before each definition.
- **Viz (D3):** one or more interactive components; each shows a parameter the reader can
  manipulate and a thing they should learn from manipulating it.
- **Python (notebook pillar):** the `notebooks/README.md` contract —
  - `notebooks/<slug>/<slug_underscored>.py` — canonical, tested, importable, CPU-only, < 60 s. Deps
    declared in the module docstring as the `uv run --with … python …` line. A harness of
    `assert`-based tests encoding **every pedagogical claim the topic makes** (limit theorems,
    monotonicity, the worked-example flip, a cross-check against a reference library where one
    exists). This file **owns every number the viz mirrors**.
  - `notebooks/<slug>/01_<slug_underscored>.ipynb` — narrative notebook that imports the `.py` and
    walks the topic section by section, so claims render as executed output. Model it on
    `notebooks/bm25/01_bm25.ipynb`. Commit without stored outputs.
  - Both exit 0:
    ```bash
    uv run --with <deps> python notebooks/<slug>/<slug>.py
    uv run --with <deps> --with jupyter jupyter execute notebooks/<slug>/01_<slug>.ipynb
    ```

### A2. Editorial voice (non-negotiable)

Informed peer, not lecturer. Collaborative "we" (we define, we observe); "you" only for direct
reader instructions ("try dragging the slider"). Introduce **all** notation on first use, even
$\lVert \mathbf{x} \rVert$. No "simply", "obviously", "it's easy to see". American English. Match
the register of formalML's `svd.mdx`. **Geometric-first:** introduce visually and concretely, then
the algebra; foundational topics stop at geometric intuition.

### A3. The schema fields, and what they mean

Required: `title`, `subtitle`, `status` (`draft` while authoring → `published` at ship),
`difficulty`, `tags`, `domain`, `abstract`, `prerequisites`. formalRAG-specific:

- `pipelineStage` — the second navigation axis, orthogonal to `domain` (where in the
  ingest→evaluate pipeline this topic lives).
- `financeCaseStudy` — `true` if the topic carries the recurring finance thread (earnings calls,
  10-K filings, the production multimodal-financial-RAG system). Set the `modality` array to match
  (e.g. `[text, audio]` for a transcript example).
- **`rigorFlag` is load-bearing.** Honesty is the differentiator: name the celebrated-but-heuristic
  parts plainly — a tuned magic constant, a missing optimality guarantee, a false independence
  assumption. (BM25: empirically-tuned $k_1/b$; HNSW: scaling is empirical; MMR: no $1-1/e$
  guarantee.) Never launder a heuristic as a theorem.
- `notebookPath` points at the `.ipynb`; `githubUrl` at the MDX on GitHub.
- `references[].type` ∈ `paper | book | course | blog | video | documentation` (`documentation`
  for FAISS/Qdrant/ColBERT/RAGAS-style docs that back the code pillar). **Every reference needs a
  `url`** — DOI for papers/books, proceedings or arXiv otherwise.

### A4. Cross-site links — mind the direction

formalRAG links **UP into formalML** (its deepest prerequisite layer) *and* down into the
calculus/statistics foundations — the inverse of the sibling sites. Each entry is
`{ topic, site, relationship }` where `site ∈ {formalml, formalcalculus, formalstatistics}` and
`relationship` is ≥ 40 chars of real prose:

- `formalmlPrereqs` / `formalstatisticsPrereqs` / `formalcalculusPrereqs` — what this topic *needs*.
- `…Connections` — what this topic *informs* (rarer, forward links).

**Verify the target slug exists on the sibling before citing it — do not invent.** (CLAUDE.md flags
that `learning-theory` is *not* a formalML slug; use `vc-dimension` / `generalization-bounds`.)
Reverse links from the siblings back into formalRAG are added per-sibling, via a worktree off that
sibling's `origin/main`, as the linked topics ship.

### A5. Phase A deliverables

- `docs/plans/formalrag-<slug>-brief.md` — the implementation spec: section outline, theorem
  statements + proofs, viz design intent (component-level, not React/D3 code), notebook-cell map,
  cross-site prereq list, references. Per the scaffold-don't-draft preference, build the outline and
  the math collaboratively before drafting full prose.
- `notebooks/<slug>/` — the verified `.py` + `.ipynb`, both exiting 0.

---

## Phase B — Implement & ship (MDX + viz + graph + PR)

The notebook is now immutable and is the source of truth. Build the page against it.

### B1. Required reading (in order)

`CLAUDE.md` → `docs/plans/formalrag-<slug>-brief.md` → `notebooks/<slug>/01_<slug>.ipynb`. Then
study the BM25 topic as the structural exemplar: `src/content/topics/bm25-binary-independence-model.mdx`
(frontmatter anatomy, `TheoremBlock`/`NamedSection`/`RigorFlag`/`FinanceCaseStudy` usage, the
`<Viz client:visible />` embed) and `src/components/viz/BM25ScoringLaboratory.tsx`.

### B2. Build order

1. Author the MDX from the brief — pull definitions/theorems/proofs from the notebook, adapt only
   formatting for `TheoremBlock`. Set the full frontmatter (A3) with `status: draft`.
2. Build each viz component. **Viz ↔ Python invariant:** any corpus/weights/numbers the viz shows
   are mirrored *to the decimal* from the topic's `.py`, with a comment citing the asserting test.
   Change one → change both.
3. Add the topic to `src/data/curriculum.ts` / `curriculum-graph.json` (move it out of `planned`).
   Wire `prerequisites` + `connections`; both endpoints must resolve to real MDX.
4. Add cross-site arrays (A4). Defer a sibling's reciprocal link until that sibling topic ships.
5. Flip `status: published`, set `datePublished`, `estimatedReadTime`, `githubUrl`, `notebookPath`.

### B3. Validation gates (run in order)

```bash
pnpm exec astro sync       # schema + ALL frontmatter (fast, no full render)
pnpm validate              # validateConnections.ts — prereq/connection/graph integrity
pnpm audit:cross-site      # reciprocity (needs sibling repos adjacent / FORMAL_*_PATH)
pnpm dev                   # then open the topic page
```

**`pnpm build` / exit 0 ≠ math correct.** KaTeX is non-strict: parse errors render as
`.katex-error` spans and the build still exits 0. **Open the page and verify the DOM with
`browser_evaluate`, not screenshots** (screenshots drift to `/` on this setup). Assert: zero
`.katex-error` spans, the expected `.katex` count, and that each viz mounted (slider/ranking
present). Re-run the notebook one final time; confirm the page's worked-example numbers equal the
notebook's printed output.

### B4. Ship

Per the git workflow: feature branch → commit → PR for review before `main` (deploys trigger from
`main`). No worktrees for MDX/content — author on the branch directly. PR body: what shipped (viz
count, prereq additions), the validation-gate results, and any honest caveats the `rigorFlag`
encodes.

---

## Stop and ask if

- A proof needs a lemma/prerequisite not on a published in-site or sibling topic the reader can be
  assumed to have read.
- A viz needs a dataset or parameter regime not yet specified.
- A cross-site target slug can't be verified to exist — never fabricate one.
- A verification number disagrees with the notebook, or a `.katex-error` appears and the fix is not
  obvious.
- You're tempted to change a verified number in the harness *or* the viz without updating the other.
- The topic would duplicate or contradict a published topic's scope.

---

*Living document. When a convention, command, or schema field changes, update this file and
`CLAUDE.md` together. The real test that this prompt is complete is authoring the next topic with it
and hitting no missing step.*
