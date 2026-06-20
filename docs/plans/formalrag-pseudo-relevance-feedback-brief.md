# Brief — Relevance Feedback and Query Expansion: Rocchio and RM3 (`pseudo-relevance-feedback`)

Domain: probabilistic-ir · difficulty: intermediate · pipelineStage: retrieve · financeCaseStudy: true (text).
Prereqs: `bm25-binary-independence-model`, `query-likelihood-language-models`. **Completes the probabilistic-IR track.**

## Positioning

The closing-the-loop topic: use the documents already retrieved to repair the query (vocabulary mismatch),
then search again. Two families — Rocchio (vector space) and RM1/RM3 (language model, builds on the
query-likelihood KL view). The honest hinge: pseudo-relevance feedback *assumes* the top-k are relevant, so
a little feedback helps and too much **drifts**. No per-query guarantee.

## Section outline (H2)

1. Relevance feedback and the vocabulary-mismatch problem.
2. Rocchio: the centroid update (Def 1, Prop 1 + proof).
3. The relevance model RM1 (Def 2).
4. RM3: interpolating with the original query (Def 3, Prop 2 limits).
5. Query drift (rigorFlag), finance case study, implementation.

## Math (all verified in `notebooks/pseudo-relevance-feedback/pseudo_relevance_feedback.py`)

- **Def 1 — Rocchio.** `q' = α q + β·centroid(D_r) − γ·centroid(D_nr)`; PRF sets γ=0, D_r = top-k.
- **Prop 1 — centroid optimality.** The fixed-norm query maximizing mean-sim(D_r) − mean-sim(D_nr) points
  along `centroid(D_r) − centroid(D_nr)` (Cauchy–Schwarz on `q'·(d̄_r − d̄_nr)`). α q anchors to the query.
- **Def 2 — RM1.** `P(w|R) = Σ_d P(w|d) P(d|q)`, `P(d|q) ∝ P(q|d)` (query likelihood); proper distribution.
- **Def 3 — RM3.** `P_RM3 = (1−α) P_ml(·|q) + α P_RM1`; re-score by cross-entropy `Σ_w P_RM3(w) log P(w|d)` (KL view).
- **Prop 2 — RM3 limits.** α=0 → original query-likelihood ranking; α=1 → pure relevance model.

**rigorFlag (executable):** PRF assumes top-k relevant (often false) → drift; α/β/γ, #feedback docs, #expansion
terms tuned; re-ranking smoothing must scale to doc length; improves *average* effectiveness, no per-query guarantee.

## Verified numbers (the viz mirrors the per-feedback-size table)

- Query `rate guidance`, relevant = {r1,r2,r3,r4} (r2,r3 are synonym-only: outlook/forecast, no "guidance").
- Config: MU=5, ALPHA=0.5, N_TERMS=10, K=4.
- RM3 recall@4 vs #feedback: 0.50 (0) → **1.00** (1–2) → 0.75 (3) → 0.50 (4) → 0.75 (5). Improve **then drift**.
- RM1 expansion at n_fb=2 (clean): forecast 0.243, outlook 0.243, rate 0.137, guidance 0.122, charge 0.031.
  At n_fb=4 (polluted): forecast/outlook drop to 0.193, off-topic budget/costs (0.045) enter.
- Rocchio recall@4: 0.75 (0) → 1.00 (1–2); q' = a q + b·centroid, raises mean cosine to feedback docs.
- α=0 recovers the unexpanded query-likelihood ranking (recall 0.50).

## Viz — `src/components/viz/QueryExpansionFeedbackLab.tsx`

One feedback-size slider (0–5) indexing the mirrored `viz_constants()` table (no RM1 arithmetic in TS).

- **Panel A — recall@4 vs feedback size:** the improve-then-drift bar curve (current n_fb highlighted).
- **Panel B — RM1 expansion terms:** weights, colored bridge (outlook/forecast) vs off-topic (segment/budget/tax).
- **Panel C — re-ranked top-6:** relevant docs highlighted; recall@4 readout.

## Notebook — `notebooks/pseudo-relevance-feedback/`

- `pseudo_relevance_feedback.py` — owns the numbers; Rocchio + RM1/RM3 + miniature query-likelihood + 5
  asserting tests + `viz_constants()`. Run: `uv run --with numpy python notebooks/pseudo-relevance-feedback/pseudo_relevance_feedback.py`.
- `01_pseudo_relevance_feedback.ipynb` — narrative, imports the `.py`, generated via nbformat.
  Run: `uv run --with numpy --with jupyter jupyter execute notebooks/pseudo-relevance-feedback/01_pseudo_relevance_feedback.ipynb`.

## Cross-site (all confirmed on disk)

- formalML `clustering` (Rocchio nearest-centroid), `kl-divergence` (RM3 cross-entropy re-scoring).
- formalStatistics `point-estimation` (RM1 is a point estimate of the expanded query; its contaminated-sample bias is drift).

## References (DOIs verified via `curl -sI`)

- Lavrenko & Croft 2001 (RM1) — `10.1145/383952.383972`.
- Lv & Zhai 2009 (RM3 / PRF estimation comparison) — `10.1145/1645953.1646259`.
- Salton & Buckley 1990 (Rocchio relevance feedback) — `10.1002/(SICI)1097-4571(199006)41:4<288::AID-ASI8>3.0.CO;2-H`.
- Manning, Raghavan & Schütze 2008, ch. 9 — IR-book URL.
- Pyserini / Anserini RM3 — documentation.

## Note

Also re-adds the reciprocal `query-likelihood-language-models → pseudo-relevance-feedback` connection and
inline link (deferred when QL shipped before PRF existed).
