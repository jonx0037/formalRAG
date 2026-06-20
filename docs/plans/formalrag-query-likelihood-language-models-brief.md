# Brief — Query-Likelihood Language Models and Smoothing (`query-likelihood-language-models`)

Domain: probabilistic-ir · difficulty: intermediate · pipelineStage: retrieve · financeCaseStudy: true (audio, text).
Prereqs: `the-retrieval-problem`, `vector-space-model-tfidf`.

## Positioning

The third model on the shared lexical corpus (`the-retrieval-problem → VSM → BM25 → query-likelihood`).
Where VSM measures geometric similarity and BM25 ranks by probability of *relevance*, query likelihood
ranks by the probability the document's language model *generated* the query. The MLE `P(t|d)=tf/|d|`
builds length normalization into the probability, so QL sidesteps the length hijack — but pays with the
zero-frequency catastrophe, which smoothing repairs. Smoothing is the whole substance of the topic.

## Section outline (H2)

1. The query-likelihood model (multinomial unigram).
2. The zero-frequency catastrophe.
3. Two smoothings — Jelinek–Mercer and Dirichlet.
4. Dirichlet smoothing is a Bayesian posterior mean (Thm 1).
5. The KL-divergence view (Thm 2).
6. The dual role of smoothing — an IDF-like effect (Thm 3).
7. Length adaptivity, the length-hijack contrast, finance case study, caveats, implementation.

## Math (all verified in `notebooks/query-likelihood-language-models/query_likelihood_language_models.py`)

- **Def 1 — QL.** `score(q,d)=Σ_t c(t,q) log P(t|M_d)`, `P_ml(t|d)=tf/|d|`.
- **Prop 1 — zero-frequency catastrophe.** Any missing query term → `P_ml=0` → `−∞`. Kills 3/6 docs here.
- **Def 2 — smoothings.** JM `(1−λ)P_ml+λP(t|C)`; Dirichlet `(tf+μP(t|C))/(|d|+μ)`.
- **Thm 1 — Dirichlet = posterior mean.** Conjugate `Dirichlet(α_t=μP(t|C))` prior + multinomial likelihood
  → `Dirichlet(α_t+tf_t)` posterior; mean `=(tf+μP(t|C))/(|d|+μ)` since `Σα=μ`, `Σtf=|d|`.
- **Thm 2 — KL view.** `−KL(θ_q‖θ_d)=H(θ_q)+(1/|q|)score(q,d)`; `H(θ_q)` constant → rank-equivalent. Hook for RM3.
- **Thm 3 (Zhai–Lafferty) — IDF-like effect.** JM log score = matched-term sum
  `Σ c(t,q) log(1+((1−λ)/λ)P_ml/P(t|C))` + doc-independent constant `|q|logλ + Σ c(t,q)logP(t|C)`;
  per-term weight `↓` as `P(t|C) ↑` (`w(x)=log(1+κ/x)`, `w'<0`).
- **Prop 2 (rigorFlag context) — Dirichlet length normalization.** Effective weight `μ/(|d|+μ)` strictly
  decreasing in `|d|`; explicit length penalty `|q|log(μ/(|d|+μ))`. JM has none.

**rigorFlag:** unigram independence false; λ, μ tuned (μ≈1000–2000); IDF-like is inverse-*collection*-frequency,
emergent from smoothing, not a derived IDF.

## Verified numbers (the viz mirrors `P(t|C)`, `coll_len`, `|d|`, query-term tf to the decimal)

- `N=6`, `|C|=316`. `P(t|C)`: interest **0.025316**, rate **0.034810**, exposure **0.018987**.
- MLE (3 docs `−∞`): filing-onpoint **−6.74 (#1)**, filing-boiler −7.92, transcript-pad −13.37.
- JM λ=0.5: filing-onpoint **−8.16 (#1)**, filing-boiler −9.06, filing-fx −9.58 (all 6 finite).
- Dirichlet μ=2000: filing-onpoint **−10.93 (#1)**, filing-fx −10.95, filing-boiler −10.96 (tight pack).
- KL rank-equivalence holds on worked example + 200 strict random instances.

## Viz — `src/components/viz/QueryLikelihoodSmoothingLab.tsx`

Method selector (none / JM / Dirichlet) + one slider (λ or μ) + panel-A document picker.

- **Panel A — document model bars:** for a chosen doc, smoothed `P(t|d)` (bar), MLE (dashed marker),
  `P(t|C)` (line). Pick the FX filing (no "interest") to watch a zero MLE lift to the collection prob.
- **Panel B — effective smoothing weight:** Dirichlet `μ/(|d|+μ)` per doc, decreasing in `|d|` (length
  adaptivity); JM flat at λ; none = 0.
- **Panel C — live ranking (payload):** `log P(q|d)`; "no smoothing" sends 3 docs to `−∞`; smoothing
  makes all 6 finite with the concise on-point filing #1.

## Notebook — `notebooks/query-likelihood-language-models/`

- `query_likelihood_language_models.py` — owns the numbers; 7 asserting tests + demos + `viz_constants()`.
  Run: `uv run --with numpy python notebooks/query-likelihood-language-models/query_likelihood_language_models.py`.
- `01_query_likelihood_language_models.ipynb` — narrative, imports the `.py`, generated via nbformat.
  Run: `uv run --with numpy --with jupyter jupyter execute notebooks/query-likelihood-language-models/01_query_likelihood_language_models.ipynb`.

## Cross-site (all confirmed on disk; reciprocals deferred to siblings)

- formalStatistics `maximum-likelihood` (MLE), `bayesian-foundations-and-prior-selection` (Dirichlet posterior mean).
- formalML `kl-divergence` (KL view), `shannon-entropy` (query entropy / cross-entropy).

## References (DOIs verified via `curl -sI`)

- Ponte & Croft 1998 — `10.1145/290941.291008`.
- Zhai & Lafferty 2004 (smoothing study) — `10.1145/984321.984322`.
- Lafferty & Zhai 2001 (KL / risk minimization) — `10.1145/383952.383970`.
- Manning, Raghavan & Schütze 2008, ch. 12 — IR-book URL.
- Pyserini QL/QLD — documentation.
