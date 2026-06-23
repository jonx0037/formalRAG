# Brief — NDCG: Graded Relevance and Discount Geometry

**Slug / node:** `ndcg-discount-geometry` · **domain** retrieval-evaluation · **stage** evaluate ·
**difficulty** intermediate · **financeCaseStudy** true · **modality** [text, audio]
**Prereq (only):** `set-metrics-precision-recall-map-mrr` (PR #44, on `main`).

## Thesis

Set metrics treat relevance as binary; a perfect 10-K disclosure and a vaguely-related transcript
snippet both count as "relevant." NDCG keeps the *degree* of match (graded relevance) and the
*position* it lands (a rank discount), and normalizes by the best achievable ordering. The
distinctive math is the **discount geometry** — the shape of `1/log₂(i+1)`, its heavy tail vs the
geometric/RBP discount's clean user model — and the **rearrangement inequality**, which is *why* the
ideal ranking (sort by descending gain) maximizes DCG. The estimator thread from set-metrics carries
over verbatim: per-query NDCG → mean → SE → CI → bootstrap.

## What is reused (imported, never reimplemented)

- `set_metrics_corpus`, `LEGS`/`LEG_NAMES`, the cached per-leg `rankings`, `qrels_set` — from
  `set_metrics_precision_recall_map_mrr`.
- `metric_summary`, `projected_ci` — generic estimator machinery, reused on NDCG samples.
- `maxsim_matrix` (the exact oracle, = what `brute_topk` calls), `TOPK` — to grade documents.
- `ndcg_at_k` from `bm25` — the **byte-for-byte twin** (linear gain + log₂ discount).

## The one new construction — graded relevance

`graded_qrels`: per query, the oracle top-K docs (= `qrels_set`) get grades {1,2,3} by **global
tertiles** of their exact-MaxSim oracle score; everything else is grade 0. So
`{doc : grade ≥ 1} ≡ qrels_set` **exactly** (asserted nesting / shared-baseline anchor), while IDCG
varies per query.

## Sections (MDX)

1. **From sets to grades** — binary throws away degree-of-match (finance: on-point disclosure vs
   tangential snippet); construct graded qrels from the oracle.
2. **Gain and discount: the two design choices** — `gain_linear(g)=g` vs `gain_exp(g)=2^g−1`;
   `1/log₂(i+1)` and alternatives. Both are conventions (rigorFlag).
3. **DCG, IDCG, NDCG** — the inner-product form `DCG = ⟨gains-in-rank-order, discounts⟩`; normalize by
   IDCG; NDCG ∈ [0,1].
4. **The ideal ranking is optimal: the rearrangement inequality** — descending discounts paired with
   descending gains maximize the inner product → IDCG; ascending = the minimizer. The rigorous heart,
   with proof. (Twin: reduces to bm25's `ndcg_at_k` under linear gain.)
5. **Discount geometry** — weight mass in the head, marginal value `d(i)−d(i+1)`, log's heavy tail vs
   geometric/RBP's light tail + its closed user model (`E[docs examined] = 1/(1−p)`); swap sensitivity
   `Δ = Δgain·Δdiscount` (name LambdaRank, **no link** — unbuilt).
6. **NDCG as an estimator** — carry the set-metrics thread; the convention/verdict flip (Phase-A
   discovery); when two legs' CIs separate.
7. **Convention sensitivity & consistency** — gain/discount choices flip the verdict (built + run);
   Wang et al. (2013) consistency + the truncation caveat. Honest rigorFlags.

## Theorems made executable (asserts)

- **Twin:** `ndcg_at_k(·, gain_linear, log₂) == bm25.ndcg_at_k`, `<1e-12`.
- **Rearrangement:** `IDCG ≥ DCG(any permutation)`; ascending-gain order `< IDCG` strictly when grades
  differ. → NDCG ∈ [0,1], =1 iff top-k ideal.
- **Nesting:** `{grade ≥ 1} ≡ qrels_set` per query.
- **Discount geometry:** geometric concentrates more head-mass than log at equal k; marginal value
  positive & decreasing; `RBP E[docs]=1/(1−p)`.
- **Estimator:** per-query NDCG SE `~ 1/√n` (reused resampler); bootstrap ≈ analytic; two-leg CI
  overlap → separation `n`.
- **Convention flip:** constructed gain-flip and discount-flip toys must reverse the verdict; report
  natural per-query reversals on the corpus (the legs are a quality ladder, so an *aggregate* leg flip
  may be vacuous — pin the headline to what actually reverses).

## Viz — `NDCGLaboratory.tsx` (4 panels, numbers baked from `viz_constants()`)

- **A** Graded relevance & ideal ranking: worked query, per-leg grade-in-rank-order vs ideal, live
  DCG/IDCG/NDCG.
- **B** Discount geometry: log₂ / geometric(p) / 1/i curves; RBP `p` slider; head-mass & marginal-value
  bars.
- **C** Convention flip: toggle linear↔exp gain and log↔geometric discount; per-leg MNDCG bars
  reorder; constructed-toy inset.
- **D** NDCG as estimator: query-count slider; per-query NDCG strip; MNDCG 95% CI shrinking `1/√n`;
  two-leg overlap/separation. Mirrors set-metrics Panel D.

## rigorFlags

Gain function is a convention (exp = common modern default, not "correct"); the log₂ discount is a
**heuristic** with no closed user model (unlike RBP's geometric); NDCG@k truncation can be
inconsistent (Wang 2013); grades are oracle-score tertiles, a neutral stand-in for editorial
judgments; MNDCG is a sample mean — gaps within CI overlap aren't real; normal-approx CI is
approximate (paired/skewed/bounded) → bootstrap cross-check, paired significance test deferred.

## References

- Järvelin & Kekäläinen (2002), *Cumulated gain-based evaluation of IR techniques*, ACM TOIS — the
  original (N)DCG with linear gain.
- Burges et al. / LETOR — the `2^g−1` exponential gain default.
- Moffat & Zobel (2008), *Rank-biased precision*, ACM TOIS — the geometric discount with a user model.
- Wang, Wang, Li, He, Liu (2013), *A Theoretical Analysis of NDCG Type Ranking Measures*, COLT — the
  consistency result + truncation caveat (verify DOI).

## Cross-site (verify slugs first)

Mirror set-metrics' estimator up-links: `formalstatisticsPrereqs` (`point-estimation`,
`confidence-intervals-and-duality`), `formalstatisticsConnections` (`hypothesis-testing`,
`bootstrap`). Optional `formalmlConnections` for consistency↔Bayes-optimal ranking if a slug fits.
Connections: link `set-metrics-precision-recall-map-mrr`, `rank-fusion-rrf` (published); name-not-link
`lambdarank-lambdamart-listwise`, `significance-testing-calibration`.
