# Brief — `conformal-factuality`: Distribution-Free Correctness Guarantees for Generation

Phase-A implementation spec. The notebook (`notebooks/conformal-factuality/conformal_factuality.py`)
is verified and OWNS every number; this brief records the final design and the values the MDX/viz mirror.

## Thesis & arc

The eval layer's through-line — *a metric is an estimator* — reaches its terminus: this topic turns the
calibrated, debiased per-claim judge confidence (from `llm-as-judge-ragas`) into a **distribution-free
correctness guarantee**. Conformal prediction converts *any* nonconformity score into finite-sample
coverage under **exchangeability alone**. We run split conformal for a recall guarantee, then **Conformal
Risk Control** for the guarantee that matters — the false-claim rate among what we emit — then show drift
breaks exchangeability and weighted conformal repairs it. Finance thread: a RAG-over-filings system that
**backs off** unsupported claims at a guaranteed error rate an auditor could sign off on.

## What is reused vs. fresh

- **Reused (link up, do NOT re-prove):** formalML `conformal-prediction` owns exchangeability (Def 1),
  split-conformal (Def 2), **Theorem 1** marginal coverage (threshold = ⌈(1−α)(n+1)⌉-th smallest score;
  two-sided ≤ 1−α+1/(n+1)), conditional-coverage **impossibility** (Thm 4), the covariate-shift remark.
- **Fresh:** the faithfulness nonconformity score; per-claim back-off; the **CRC theorem** on a monotone
  false-claim loss; the perfect-judge **collapse anchor** onto imported `precision_at_k`; covariate-shift
  break + weighted-conformal repair; the finance abstention case study.

## Core math (TheoremBlocks, in order)

1. **Def — nonconformity score.** Calibrated judge confidence `c̃∈[0,1]`; score `s = 1 − c̃`. Faithful →
   high confidence → low score. (Pitfall: `s = c̃` inverts the retained set.)
2. **Algorithm — per-claim back-off.** `Ĉ = {i : c̃_i ≥ τ}`; sort by descending confidence, drop the
   bottom until all retained clear the cut. Mohri–Hashimoto / C-RAG shape.
3. **Calibration procedure.** Hold out n_cal claims with labels; restrict to truly-faithful (Y=1); score
   `S_i = 1−c̃_i`; `τ̂ = ⌈(1−α)(n₁+1)⌉`-th smallest (cap +∞ when rank>n₁). `confidence_threshold = 1−τ̂`.
4. **Corollary — recall (Theorem 1 verbatim).** `P(retained | faithful) ≥ 1−α`. **α controls the drop
   rate of genuinely-faithful claims, NOT hallucination leakage** — the bridge to CRC. (Build-and-run
   confirms: under the lenient judge the *uncontrolled* false-claim rate runs to 0.23 at α=0.02.)
5. **Theorem — CRC (Angelopoulos et al. 2024).** Monotone non-increasing loss `L_i(λ) ≤ B`,
   `λ̂ = inf{λ : (n/(n+1))R̂_n(λ) + B/(n+1) ≤ α}` ⇒ `E[L_{n+1}(λ̂)] ≤ α`. Keep both finite-sample
   corrections. Split conformal = the indicator-loss special case (B=1).
   - **Monotone loss (load-bearing):** `L_i(λ) = (1/k)·#{retained ∧ unfaithful}` (denominator = fixed
     slot count k). Each indicator only switches off as λ rises ⇒ non-increasing, ∈[0,1], B=1. CRC thus
     controls the **per-slot** false-claim rate. **Reject** the fraction-of-retained loss — numerically
     non-monotone (counterexample baked: 0.1 → 1.0 as the cut rises).
6. **Proposition — perfect-judge collapse anchor.** sens=spec=1 ⇒ scores separate (faithful 0, unfaithful
   ~1): (a) zero false-claim at every grid α; (b) per-query retained fraction == imported
   `oracle_faithfulness` == `precision_at_k`, `<1e-12`.
7. **Remark + Def — covariate shift & weighted repair.** A deployment that over-represents low-verbosity
   claims (tilt by the **known** likelihood ratio `w(x)=exp(−βx)`, x = the doc-verbosity covariate) breaks
   exchangeability; the unweighted threshold under-covers. Weighted split-conformal (Tibshirani et al.
   2019): normalized weights with the test point's own mass in the denominator; `τ̂^w` = weighted
   (1−α)-quantile restores coverage at the shifted distribution (marginal, not conditional).
8. **RigorFlag:** marginal-not-conditional (Thm 4 impossibility); exchangeability fragility (intra-answer
   dependence, drift); **CRC is E[loss]≤α in expectation, not high-probability** (one realization can
   exceed — α=0.20 realizes 0.205; RCPS / Bates et al. 2021 is the δ-level alternative, named not built);
   "faithful" = faithful under the **synthetic judge oracle** (inherits the prereq's caveat); weighted
   repair assumes the likelihood ratio is **known** (must be estimated in practice).

## Notebook API (verified, `conformal_factuality.py`)

Imports: `LEG_NAMES, precision_at_k` (set-metrics); `get_corpus, platt_scale, apply_platt,
isotonic_calibrate, apply_isotonic, expected_calibration_error, auc_pooled` (significance-testing);
`K, judge_confidence, oracle_faithfulness, doc_length_feature, candidate_ids, JUDGE_PERFECT`
(llm-as-judge). Judge = custom **lenient** `dict(sens0=0.80, spec0=0.62, b_len=1.3, b_pos=0.7,
b_self=0.6)` (AUC≈0.90, overlapping classes — JUDGE_BALANCED separates perfectly and is vacuous).

Functions: `pooled_confidence`, `calibrated_confidence`, `split_masks`, `split_conformal_threshold`,
`confidence_threshold`, `back_off_retained`, `false_claim_loss` (monotone), `fraction_loss` (the rejected
naive loss), `loss_matrix`, `conformal_risk_control_threshold`, `claim_covariates`,
`weighted_conformal_threshold`, `mc_realized_coverage`, `crc_threshold_and_risk`, `risk_coverage_curve`,
`covariate_shift_curve`, `worked_answer`, `judge_calibration_quality`, `viz_constants`.

Tests (all pass, <2s): shares-one-corpus; quantile-rank + +∞ guards; **perfect-judge collapse** (==
precision_at_k <1e-12, zero false at all α); back-off removes least-confident; **monotone loss**; **naive
fraction-loss non-monotone counterexample**; **CRC monotone + risk≤α**; **coverage≈1−α**; **weighted
twin == split <1e-12**; **covariate-shift breaks then weighted restores**; guards.

## Verified numbers the viz bakes (mirror TO THE DECIMAL)

- **Shared:** n_queries 40, n_docs 120, k 10, α 0.10, leg dense, judge lenient. calib_quality:
  ece_raw 0.0928, ece_platt 0.0622, auc_raw = auc_platt 0.8981 (Platt preserves AUC exactly).
- **Panel A:** 137 sorted calib faithful scores (0.0282 … 0.8889); tau_by_α {0.02:0.8766, 0.05:0.618,
  0.10:0.4377, 0.15:0.3447, 0.20:0.2904, 0.30:0.2064, 0.40:0.1058}; mc_coverage means {0.983, 0.949,
  0.902, 0.855, 0.805, 0.701, 0.601} ≈ target 1−α (validity demonstrated).
- **Panel B:** worked query 20, precision@k 0.5; conf [0.9705, 0.0362, 0.9367, 0.7965, 0.0456, 0.8051,
  0.9336, 0.3537, 0.8912, 0.104]; y [1,0,1,0,0,1,1,0,1,0]. (Claim 3: an unfaithful doc endorsed at 0.80 —
  the high-confidence hallucination that leaks until τ rises.)
- **Panel C (frontier):** split_false {0.230, 0.160, 0.115, 0.105, 0.100, 0.090, 0.040} (UNCONTROLLED)
  vs crc_false {0.000, 0.010, 0.085, 0.140, 0.205, 0.300, 0.390} (≤α, the 0.205@0.20 overshoot is the
  expectation-not-realization caveat); crc_retention {0.0, 0.30, 0.595, 0.715, 0.810, 0.910, 1.0}.
- **Panel D (covariate shift):** β grid (0,0.5,1,1.5,2,3); split coverage {0.948, 0.851, 0.764, 0.646,
  0.480, 0.266} collapses; weighted {0.948, 0.880, 0.934, 1.0, 1.0, 1.0} restores; break_beta 0.5.

## Viz (4 panels) — `ConformalFactualityLaboratory.tsx`

React island, `client:visible`, declarative JSX SVG, KaTeX per panel, baked constants + invariant comment.
α slider shared A↔B↔C; β slider for D. TS recomputes ONLY: Panel A quantile index/q̂ as α moves; Panel B
sort+filter+retained-faithful mean at the shared τ. Everything corpus-derived is baked.

## MDX frontmatter & graph

`prerequisites: [llm-as-judge-ragas, significance-testing-calibration]`; `domain retrieval-evaluation`;
`pipelineStage evaluate`; `difficulty advanced`; `financeCaseStudy true`; `modality [text, pdf]`.
`connections[]` (BUILT only): llm-as-judge-ragas, significance-testing-calibration,
set-metrics-precision-recall-map-mrr. Cross-site: `formalmlPrereqs` conformal-prediction,
concentration-inequalities; `formalmlConnections` always-valid-inference; `formalstatisticsPrereqs`
hypothesis-testing, confidence-intervals-and-duality; `formalstatisticsConnections`
multiple-testing-and-false-discovery. Name unbuilt in-site (faithfulness-groundedness,
selective-generation) in prose only. Graph: empty the retrieval-evaluation `planned[]`; flip node
status planned→published (two prereq edges already exist; no forward edges).

## References (verify DOIs with `curl -sI`)

CRC arXiv 2208.02814 · Mohri–Hashimoto arXiv 2402.10978 · C-RAG (Kang et al.) arXiv 2402.03181 ·
Tibshirani et al. covariate-shift arXiv 1904.06019 · RCPS Bates et al. JACM 2021 DOI 10.1145/3478535 ·
Angelopoulos–Bates gentle intro arXiv 2107.07511 · (upstream cite: Lei et al. JASA 2018
10.1080/01621459.2017.1307116; Vovk–Gammerman–Shafer; Foygel Barber et al. 2021).
