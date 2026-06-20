# Brief — The Vector Space Model and TF-IDF (`vector-space-model-tfidf`)

Implementation spec for the topic. Math + notebook verified before MDX/viz (Phase A complete).
Domain `probabilistic-ir`, difficulty `foundational`, `financeCaseStudy: true`, prerequisite
`the-retrieval-problem`. Child topic: `bm25-binary-independence-model` (already published).

## Positioning

Sits between `the-retrieval-problem` (delivered cosine + the off-sphere magnitude story) and
`bm25-binary-independence-model` (assumes a tf transform, the shared IDF factor, and the length
problem). The differentiator is the **information-theoretic reading of IDF** — it is exactly the
self-information of a term's presence — which BM25 later re-derives probabilistically from the BIM.
Honest framing: cosine VSM is a calibrated geometry, not a probability; only the IDF = self-information
identity (and the concavity/cosine-invariance facts) are genuine theorems.

## Section outline (H2)

Overview & motivation (+ `<VectorSpaceLaboratory client:visible />`) → Documents and queries as
term-weight vectors → Term frequency and sublinear scaling → IDF as self-information → The TF-IDF weight
→ Cosine normalization and the length problem → Where TF-IDF falls short: the bridge to BM25 → Finance
case study → Honest caveats → Implementation.

## Math (all verified in `notebooks/vector-space-model-tfidf/vector_space_model_tfidf.py`)

- **Thm 1 — IDF = self-information.** `idf_t = −log(df_t/N) = log(N/df_t)`, the pointwise Shannon surprise
  of a term's presence under a uniform-document draw. Exact (any base; base-2 ⇒ bits). Corollaries:
  `df=N ⇒ 0 bits` (illustrated by `rate`, in all 6 docs), singleton ⇒ `log N`. Asserted by
  `test_idf_is_self_information`.
- **Prop 1 — sublinear tf is increasing & concave.** `g(x)=1+log x`, `g'=1/x>0`, `g''=−1/x²<0`.
  Asserted by `test_sublinear_tf_monotone_concave` (strict, on a dense grid).
- **Prop 2 — cosine kills pure magnitude.** `cos(q, c·w_d)=cos(q,w_d)` while raw dot scales by `c`.
  Asserted by `test_cosine_normalization_invariant`.
- **Prop 3 (HEURISTIC, rigorFlag) — the two gaps that motivate BM25.** (a) sublinear tf is *unbounded*
  (no saturation ceiling) — asserted vs BM25's capped factor by `test_sublinear_unbounded_vs_bm25_bounded`;
  (b) cosine length-norm is document-global and untunable → BM25's saturating tf-factor + tunable `b`.

**Two IDF forms (a deliberate, flagged choice).** Theorem 1 is exact for the *textbook* `log(N/df)`;
scoring uses the *smoothed* `log(1+N/df)` so universal terms (`rate`) keep a small positive weight and
the length-hijack flip cleanly surfaces the on-point filing — the same exact-vs-smoothed split BM25 makes
with its RSJ weight. Smoothing is a convention, named in the rigorFlag.

## Verified numbers (the viz mirrors these to the decimal)

- Self-information (textbook, bits): interest **0.263**, rate **0.000**, exposure **0.585**.
- Smoothed scoring IDF (nats): interest 0.788, rate 0.693, exposure 0.916.
- Raw tf-idf dot (length hijack): **transcript-pad 3.873 (#1)**, filing-onpoint 2.706, filing-boiler 1.942.
- Cosine (the flip): **filing-onpoint 0.313 (#1)**, filing-fx 0.248, filing-boiler 0.219.
- scikit-learn `TfidfVectorizer` agrees on the top doc (filing-onpoint).

## Viz — `src/components/viz/VectorSpaceLaboratory.tsx`

Self-contained, `client:visible`, D3 + live KaTeX, numbers mirrored from the `.py` with a comment citing
the asserting test. Controls: **sublinear-tf toggle**, **cosine-normalization toggle**, live weight-formula
KaTeX line.
- **Panel A — self-information bars:** `idf_t` per query term in *bits* (textbook), making Thm 1 visceral
  (`rate` = 0 bits).
- **Panel B — tf-scaling curve:** raw `y=x` (dashed) vs `1+log x` (solid), *no ceiling line* — concave but
  unbounded (Prop 3a).
- **Panel C — live re-ranking (payload):** cosine OFF → `transcript-pad` hijacks #1; cosine ON →
  `filing-onpoint` snaps to #1. A badge forward-links to BM25 ("all-or-nothing — BM25 makes this a dial").

## Notebook — `notebooks/vector-space-model-tfidf/`

`vector_space_model_tfidf.py` (canonical, owns numbers, 6 asserts) + `01_vector_space_model_tfidf.ipynb`
(narrative, imports the `.py`, exits 0, no stored outputs). Deps:
`uv run --with numpy --with scikit-learn` (+ `--with jupyter` for the notebook). Corpus reused verbatim
from `notebooks/bm25/bm25.py`.

## Cross-site (all confirmed on disk)

`formalmlPrereqs: shannon-entropy` (self-information); `formalcalculusPrereqs: linear-algebra` (vectors,
dot product, norm, cosine), `mean-value-taylor` (concavity via the second derivative);
`formalstatisticsPrereqs: discrete-distributions` (count data underlying tf/df).

## References

Manning–Raghavan–Schütze (IIR, ch. 6–7); Spärck Jones 1972 (IDF); Robertson 2004 (IDF as
self-information); Salton & Buckley 1988 (SMART weighting / cosine); scikit-learn TfidfVectorizer docs.
DOIs resolve-checked before publish.
