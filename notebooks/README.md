# Notebooks — the Python pillar

Every formalRAG topic ships a **notebook pillar** alongside its math and its interactive
viz. The pillar is two coupled artifacts per topic, living in `notebooks/<slug>/`:

| File | Role | Rule |
|---|---|---|
| `<slug>.py` | **Canonical reference implementation.** Importable, deterministic, CPU-only, runs end-to-end in well under 60 s. | Owns every number the topic or the viz depends on. Carries a verification harness of `assert`-based tests that encode **each pedagogical claim the topic makes** (limit theorems, monotonicity, the worked-example flip, cross-checks against a reference library). Running it as a script *is* the regression test for the topic. |
| `01_<slug_underscored>.ipynb` | **Narrative notebook.** Imports the `.py` and walks the topic section by section in markdown cells, running the harness so each claim renders as executed output. | Never redefines the math — it calls into `<slug>.py`. The topic's `notebookPath` frontmatter points here. Commit it **without** stored outputs (we execute to verify, not to snapshot). |

The directory name uses hyphens (`notebooks/rank-fusion-rrf/`); the notebook filename uses
underscores (`01_rank_fusion_rrf.ipynb`). The `.py` file uses the hyphenless slug
(`rank_fusion_rrf.py`) so it imports cleanly.

## Why two artifacts

The `.py` is machine-verifiable and importable; the `.ipynb` is readable and teaches. Splitting
them lets the tested numbers live in one place while the prose narrative imports them — so prose
can never silently drift from verified output. `notebooks/bm25/` is the reference exemplar: model
new topics on its cell layout, import pattern, and harness structure.

## The Viz ↔ Python invariant

Any corpus, weights, or worked-example numbers a topic's D3 viz displays are **mirrored to the
decimal** from the topic's `.py`, and the topic claims they match. The `.py` is the source of
truth: change a number there, then update the viz (and re-run the notebook) — never the reverse.
For BM25, `BM25ScoringLaboratory.tsx`'s corpus mirrors `bm25/bm25.py`, and the same
`b = 0 → padded-transcript-#1, b = 0.75 → on-point-filing-#1` flip is asserted in the harness.

## Running

No shared venv — dependencies are declared per topic in the `.py` module docstring and supplied
ad hoc with `uv run --with …`:

```bash
# Run the reference implementation (asserts every claim, prints the demos):
uv run --with numpy --with scipy --with rank-bm25 python notebooks/bm25/bm25.py

# Execute the narrative notebook top to bottom to verify it still runs (no outputs saved):
uv run --with numpy --with scipy --with rank-bm25 --with jupyter \
    jupyter execute notebooks/bm25/01_bm25.ipynb
```

Both must exit 0 before a topic ships. See `../STARTER-PROMPT.md` for where the notebook pillar
sits in the full topic-authoring workflow.
