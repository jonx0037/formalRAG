# Brief — The Probability Ranking Principle (`probability-ranking-principle`)

Implementation spec. Math + notebook verified before MDX/viz (Phase A complete). Domain
`probabilistic-ir`, difficulty `intermediate`, **theory-first** (`financeCaseStudy: false`, a
finance-*flavored* worked example only). Prerequisite `the-retrieval-problem` — **independent of the
vector space model** (the DAG edge VSM→PRP is corrected to the-retrieval-problem→PRP). Child:
`bm25-binary-independence-model` (already published).

## Positioning

The decision-theoretic root of probabilistic IR. BM25 *opens* by assuming the PRP ("rank by the
probability of relevance, then by its log-odds"); this topic earns that opening. The whole result is
one move — the exchange argument — and the proof uses **no independence assumption**, only additivity
of cost and linearity of expectation.

## Section outline (H2)

Overview & motivation (+ `<ExchangeArgumentLaboratory client:visible />`) → The decision-theoretic
setup → (prefix-sum proposition) → The exchange argument (lemma + theorem) → Cost models: when it holds
and what it specializes to → A finance-flavored example → Honest caveats → Implementation.

## Math (all verified in `notebooks/probability-ranking-principle/probability_ranking_principle.py`)

- **Def 1–3** — relevance r.v. `R ~ Bernoulli(p_d)`; orderings/cutoffs; the **additive linear cost
  model** (additivity flagged as the load-bearing assumption).
- **Prop 1** — `E[#rel in top-k] = Σ_{i≤k} p_{π(i)}`, by linearity of expectation (no independence).
- **Lemma 1 (exchange)** — swapping an out-of-order adjacent pair weakly raises the prefix sum at every
  cutoff, strictly at the cutoff between them; the unaffected cutoffs cancel *because* cost is additive.
- **Thm 1 (PRP)** — sort by decreasing `p` ⇒ optimal at every cutoff at once (exchange argument →
  bubble-sort completion; inversions strictly decrease to zero).
- **Prop 2** — precision@k / recall@k corollary (prefix-sum rearrangement).
- **Prop 3** — rank by `p` ≡ rank by odds `p/(1−p)` ≡ rank by log-odds (monotone invariance). **This is
  the sentence BM25 consumes.**
- **Remark / rigorFlag** — additivity breaks under interdependent relevance (near-duplicates) →
  diversity/MMR, which the PRP does NOT cover; `p` is estimated, not observed; single static batch.

## Verified numbers (the viz mirrors these to the decimal)

- P(R) = [0.82, 0.61, 0.55, 0.30, 0.12] (filing-onpoint, news-macro, transcript-pad, filing-fx,
  transcript-short).
- PRP-optimal cumulative Σ = [0.82, 1.43, 1.98, 2.28, 2.40]; scrambled start (2,4,0,3,1) =
  [0.55, 0.67, 1.49, 1.79, 2.40], 6 inversions.
- Worked-example flip: PRP expected-relevant@3 = **1.98** vs length-biased = **1.46**.

## Viz — `src/components/viz/ExchangeArgumentLaboratory.tsx`

Reorder the 5 documents with ▲▼ buttons; a step chart races the current cumulative expected-relevant
curve toward the PRP-optimal dashed envelope. A "one adjacent swap" button performs one bubble-sort
step (swaps the first out-of-order pair); readouts show E[#rel@3] current vs optimal and inversions
remaining; a badge announces optimality when inversions hit 0. Numbers mirror `viz_constants()`.

## Notebook — `notebooks/probability-ranking-principle/`

`probability_ranking_principle.py` (canonical, owns numbers, 8 asserts incl. the theorem brute-forced
over all 120 permutations) + `01_probability_ranking_principle.ipynb` (narrative, exits 0). Deps:
`uv run --with numpy` (+ stdlib `itertools`).

## Cross-site (all confirmed on disk)

`formalmlPrereqs: naive-bayes` (Bayes-optimal decision rule); `formalstatisticsPrereqs:
expectation-moments` (linearity of expectation), `point-estimation` (decision theory / loss / risk);
`formalstatisticsConnections: likelihood-ratio-tests-and-np` (Neyman–Pearson, same exchange shape). No
`formalcalculusPrereqs` (no suitable rearrangement/permutation slug — not forced).

## References

Robertson 1977 (the PRP); Robertson & Spärck-Jones 1976; Manning–Raghavan–Schütze (IIR, ch. 11);
Robertson & Zaragoza 2009; Carbonell & Goldstein 1998 (MMR / the additivity break). DOIs
resolve-checked before publish.
