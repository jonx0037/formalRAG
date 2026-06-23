"""LLM-as-judge and faithfulness — RAGAS metrics as a FAMILY OF ESTIMATORS built from a noisy instrument.

The reference implementation for the formalRAG `llm-as-judge-ragas` topic, the fourth node of the
evaluation layer. Its prerequisites built one through-line: a retrieval metric is an ESTIMATOR — a
sample mean of per-query scores with a standard error (set-metrics), graded (NDCG), and tested,
calibrated, and monitored for drift (significance-testing-calibration). Every prior topic measured
retrieval against a FIXED relevance label (the synthetic vMF MaxSim oracle).

Generation has no fixed label. The production answer is to hire an LLM as a JUDGE (RAGAS, G-Eval,
MT-Bench), and the judge is a noisy, biased measurement INSTRUMENT. This topic carries the
metrics-as-estimators thread into the generation layer: each RAGAS metric (faithfulness here) is an
estimator built from noisy judge verdicts, and the judge itself has sensitivity, specificity,
systematic bias, and an irreducible variance floor.

THE UNIFYING THESIS. An LLM judge is a measurement instrument with its own bias and variance; every
RAGAS metric is an estimator built from its verdicts, and reading it correctly means CORRECTING for
the instrument — debiasing the verdict (Rogan-Gladen), agreeing on the protocol (chance-corrected
reliability), calibrating the confidence (the prereq's suite), and pricing the noise floor (ICC).

We IMPORT the corpus, the binary truth, the estimator machinery, and the ENTIRE calibration + paired-
test suite from the prereqs (which themselves import the published retrieval stack); we never
reimplement them. The judge is SYNTHETIC — a Bernoulli rater of the imported oracle ground truth,
whose sensitivity/specificity we control and whose biases (verbosity, position, self-preference) are
keyed to real document features — which is exactly what lets us PROVE the estimator theorems exactly
and DEMONSTRATE the corrections.

rigorFlag (primary): the judge here is a SYNTHETIC oracle with fixed, KNOWN sensitivity and
specificity; a real LLM judge has neither fixed nor known error rates — they drift with the prompt,
the model, the domain, and the candidate's position — so every correction below is conditional on
error rates that in practice must themselves be estimated from a labeled audit set. Rogan-Gladen
debiasing is unbiased only for a HOMOGENEOUS judge with known rates and BEFORE clipping to [0,1]; its
variance is inflated by 1/(se+sp-1)^2, which explodes as the judge approaches uselessness (a
near-random judge cannot be debiased, only amplified). Cohen's kappa is NOT pure agreement — under
skewed marginals high agreement can score kappa near zero (Feinstein-Cicchetti), so we read Gwet's
AC1 alongside it. The judge-variance component is an irreducible floor: more queries shrink query
sampling error but never the per-item judge noise. RAGAS faithfulness is itself a RATIO of two
LLM-extracted counts, so its denominator is noisy too. Dawid-Skene recovers error rates with no gold
labels only under conditional independence, with a non-convex likelihood identifiable up to label
permutation.

Run:  uv run --with numpy --with scipy --with scikit-learn \\
        python notebooks/llm-as-judge-ragas/llm_as_judge_ragas.py
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np
from scipy.special import expit
from sklearn.metrics import cohen_kappa_score

# --------------------------------------------------------------------------- #
# Import the three prereqs + the published stack. Add EVERY ancestor's hyphenated dir to the path
# (importing the NDCG/significance corpus pulls the whole multi-vector subtree at import time), then
# the direct prereq significance-testing-calibration (which re-exports get_corpus and the calibration
# + paired-test suite). We IMPORT the corpus, the binary truth, the estimator machinery, and the whole
# calibration/paired-test suite; we never reimplement them, and never import the downstream topics.
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "vector-quantization-lloyd-max",
    "product-quantization",
    "ivf-voronoi-partitioning",
    "bm25",
    "dense-retrieval-dual-encoders",
    "late-interaction-learned-sparse",
    "multi-vector-ann-retrieval",
    "set-metrics-precision-recall-map-mrr",
    "ndcg-discount-geometry",
    "significance-testing-calibration",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from multi_vector_ann_retrieval import TOPK                          # noqa: E402
from set_metrics_precision_recall_map_mrr import LEG_NAMES, precision_at_k   # noqa: E402
from significance_testing_calibration import (                       # noqa: E402
    get_corpus, paired_t_test, permutation_test,
    reliability_diagram, expected_calibration_error, max_calibration_error,
    brier_score, platt_scale, apply_platt,
    isotonic_calibrate, apply_isotonic, auc_pooled,
)

SEED = 0
ALPHA = 0.05
EPS = 1e-12
K = TOPK                                       # 10 candidate "claims" per answer (the top-k retrieved)
N_JUDGES = 5

# Judge presets: a logit-space bias model. (sens0, spec0) are the BASE sensitivity/specificity; the
# beta_* shift the logit of the per-claim endorsement probability by document features (verbosity =
# token dispersion, position = rank shown, self = the judge endorsing its own family's docs). A
# HOMOGENEOUS judge sets every beta to 0 (constant rates — where Rogan-Gladen is exactly unbiased).
JUDGE_PERFECT = dict(sens0=1.0, spec0=1.0, b_len=0.0, b_pos=0.0, b_self=0.0)   # collapse anchor
JUDGE_HOMOG = dict(sens0=0.85, spec0=0.80, b_len=0.0, b_pos=0.0, b_self=0.0)   # unbiased-RG regime
JUDGE_LENIENT = dict(sens0=0.90, spec0=0.70, b_len=0.9, b_pos=0.6, b_self=0.6)  # over-endorses (biased)
JUDGE_BALANCED = dict(sens0=0.82, spec0=0.84, b_len=0.4, b_pos=0.3, b_self=0.2)  # mild bias, feature spread
JUDGE_STRICT = dict(sens0=0.78, spec0=0.93, b_len=0.1, b_pos=0.1, b_self=0.0)   # under-endorses
JUDGE_NOISY = dict(sens0=0.56, spec0=0.56, b_len=0.0, b_pos=0.0, b_self=0.0)    # near coin flip
JUDGE_PANEL = (JUDGE_LENIENT, JUDGE_STRICT, JUDGE_HOMOG, JUDGE_BALANCED,
               dict(sens0=0.88, spec0=0.76, b_len=0.6, b_pos=0.2, b_self=0.4))


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _is_perfect(params: dict) -> bool:
    return (params["sens0"] >= 1.0 and params["spec0"] >= 1.0
            and params["b_len"] == 0.0 and params["b_pos"] == 0.0 and params["b_self"] == 0.0)


# =========================================================================== #
# Movement 0 — the binary truth, the document features, the candidate "claims", the synthetic judge.
# =========================================================================== #

def binary_truth(corpus: dict) -> np.ndarray:
    """Y[q, d] in {0, 1}: 1 iff doc d is in `qrels_set[q]` (the oracle's top-K MaxSim neighbours). The
    estimand ground truth — a claim is 'truly supported' iff the retrieved doc is genuinely relevant."""
    nq, nd = corpus["n_queries"], corpus["n_docs"]
    Y = np.zeros((nq, nd), dtype=int)
    for q in range(nq):
        for d in corpus["qrels_set"][q]:
            Y[q, int(d)] = 1
    return Y


def doc_length_feature(corpus: dict) -> np.ndarray:
    """len_z[d]: a z-scored 'verbosity' proxy = the dispersion of a document's own token directions
    (mean pairwise 1 - cosine among its tokens). A longer / more wide-ranging answer covers more ground;
    the lenient judge over-endorses it regardless of truth. GUARD: a single-token doc -> dispersion 0."""
    docs = corpus["docs"]                                  # (n_docs, m_d, d) unit-norm vMF tokens
    disp = np.empty(docs.shape[0])
    for i, doc in enumerate(docs):
        m = doc.shape[0]
        if m <= 1:
            disp[i] = 0.0
            continue
        G = doc @ doc.T                                    # pairwise cosines
        off = (G.sum() - np.trace(G)) / (m * (m - 1))      # mean off-diagonal cosine
        disp[i] = 1.0 - off
    sd = float(np.std(disp))
    return (disp - float(np.mean(disp))) / sd if sd > 0 else np.zeros_like(disp)


def self_feature(corpus: dict, leg: str) -> np.ndarray:
    """self_mask[q, d] in {0, 1}: 1 iff doc d is a 'house favourite' for query q — here, a doc the DENSE
    leg ranks in its own top-K. Models self-preference: a judge from the dense family over-endorses
    documents its own retriever liked. Survives order-averaging (the swap test cannot remove it)."""
    nq, nd = corpus["n_queries"], corpus["n_docs"]
    mask = np.zeros((nq, nd), dtype=float)
    for q in range(nq):
        for d in corpus["rankings"]["dense"][q][:K]:
            mask[q, int(d)] = 1.0
    return mask


def candidate_ids(corpus: dict, leg: str, k: int = K) -> np.ndarray:
    """The (n_queries, k) matrix of candidate doc ids = `leg`'s top-k ranking per query — the 'claims'
    the judge rates. Imported from the prereq's cached rankings (never re-ranked here)."""
    return np.array([corpus["rankings"][leg][q][:k] for q in range(corpus["n_queries"])], dtype=int)


def oracle_faithfulness(corpus: dict, leg: str, k: int = K) -> np.ndarray:
    """The ESTIMAND, per query: the true supported fraction of the k candidates = precision@k of `leg`
    against `qrels_set` (mean of Y over the candidates). Equals the IMPORTED `precision_at_k` exactly —
    the collapse anchor a perfect judge reproduces."""
    return np.array([precision_at_k(corpus["rankings"][leg][q], corpus["qrels_set"][q], k)
                     for q in range(corpus["n_queries"])])


def _effective_probs(corpus: dict, leg: str, params: dict, k: int = K):
    """Per-candidate endorsement probability and truth. Returns (p1, y) each (n_queries, k): p1 = the
    judge's probability of saying 'supported' (sens_eff where truly relevant, fpr_eff where not), after
    the verbosity/position/self logit shifts; y = the binary truth over the candidates."""
    Y = binary_truth(corpus)
    len_z = doc_length_feature(corpus)
    smask = self_feature(corpus, leg)
    cand = candidate_ids(corpus, leg, k)
    nq = corpus["n_queries"]
    y = np.array([[Y[q, cand[q, i]] for i in range(k)] for q in range(nq)], dtype=int)
    len_c = np.array([[len_z[cand[q, i]] for i in range(k)] for q in range(nq)])
    self_c = np.array([[smask[q, cand[q, i]] for i in range(k)] for q in range(nq)])
    pos_term = 0.5 - (np.arange(k) / (k - 1) if k > 1 else np.zeros(k))      # earlier rank -> + shift
    shift = params["b_len"] * len_c + params["b_pos"] * pos_term[None, :] + params["b_self"] * self_c
    sens_eff = expit(_logit(params["sens0"]) + shift)
    fpr_eff = expit(_logit(1.0 - params["spec0"]) + shift)
    p1 = np.where(y == 1, sens_eff, fpr_eff)
    return p1, y


def judge_verdicts(corpus: dict, leg: str, params: dict, rng, k: int = K) -> np.ndarray:
    """(n_queries, k) boolean verdict matrix — vectorized Bernoulli draws from the per-candidate
    endorsement probability. A PERFECT judge (sens=spec=1, no bias) returns the truth deterministically
    (no sampling), so the collapse anchor is exact."""
    p1, y = _effective_probs(corpus, leg, params, k)
    if _is_perfect(params):
        return y.astype(bool)
    return rng.random(p1.shape) < p1


def judged_faithfulness(corpus: dict, leg: str, params: dict, rng, k: int = K) -> np.ndarray:
    """f_hat_q = mean verdict over the k claims — the BIASED RAGAS faithfulness estimator, per query."""
    return judge_verdicts(corpus, leg, params, rng, k).mean(axis=1)


def judge_confidence(corpus: dict, leg: str, params: dict, k: int = K):
    """Pool the judge's CONTINUOUS endorsement probability and the binary truth over all (query, claim)
    pairs: returns (conf[N], labels[N]) with N = n_queries * k. Feeds the imported calibration suite
    verbatim — is the judge's stated confidence a probability?"""
    p1, y = _effective_probs(corpus, leg, params, k)
    return p1.ravel(), y.ravel().astype(float)


# =========================================================================== #
# PILLAR 1 — the judge as a noisy instrument: confusion rates and Rogan-Gladen debiasing.
# =========================================================================== #

def observed_positive_rate(verdicts: np.ndarray) -> float:
    """p_obs = mean verdict — the naive (uncorrected) faithfulness, a biased estimate of prevalence."""
    return float(np.mean(verdicts)) if verdicts.size else 0.0


def confusion_rates(verdicts: np.ndarray, truth: np.ndarray) -> dict:
    """Empirical sensitivity and specificity of a verdict matrix against the binary truth. GUARD: no
    positives (or negatives) -> the corresponding rate is the 0.5 sentinel (direction undefined)."""
    v = np.asarray(verdicts).astype(bool).ravel()
    y = np.asarray(truth).astype(bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    sens = float((v & y).sum() / n_pos) if n_pos else 0.5
    spec = float(((~v) & (~y)).sum() / n_neg) if n_neg else 0.5
    return {"sens": sens, "spec": spec, "n_pos": n_pos, "n_neg": n_neg}


def rogan_gladen(p_obs: float, sens: float, spec: float, clip: bool = True) -> float:
    """The Rogan-Gladen (1978) prevalence correction: pi_hat = (p_obs + spec - 1)/(sens + spec - 1),
    inverting p_obs = pi*sens + (1-pi)(1-spec). Unbiased for a HOMOGENEOUS judge with known rates,
    BEFORE the [0,1] clip (clipping reintroduces bias near the boundary — rigorFlag). GUARD: the Youden
    index J = sens+spec-1 ~ 0 (a coin-flip judge) -> nan (non-identifiable)."""
    J = sens + spec - 1.0
    if abs(J) < 1e-9:
        return float("nan")
    pi = (p_obs + spec - 1.0) / J
    return float(min(max(pi, 0.0), 1.0)) if clip else float(pi)


def rogan_gladen_variance(p_obs: float, sens: float, spec: float, n: int) -> float:
    """Delta-method (here EXACT, since the map is affine in p_obs) variance of the corrected estimate:
    Var = p_obs(1-p_obs) / (n * J^2), J = sens+spec-1. The 1/J^2 = inverse-square Youden inflation that
    diverges as the judge degrades toward chance. GUARD: J ~ 0 or n <= 0 -> nan."""
    J = sens + spec - 1.0
    if abs(J) < 1e-9 or n <= 0:
        return float("nan")
    return float(p_obs * (1.0 - p_obs) / (n * J * J))


def system_effective_rates(corpus: dict, leg: str, params: dict, k: int = K) -> tuple[float, float]:
    """The judge's EXPECTED sensitivity and specificity ON THIS SYSTEM'S candidates (the average of the
    per-claim sens_eff over truly-relevant claims, and of 1 - fpr_eff over truly-irrelevant claims).
    These differ across systems because the systems' candidate documents have different feature
    profiles — which is why one audited (sens, spec) imperfectly corrects a heterogeneous judge."""
    p1, y = _effective_probs(corpus, leg, params, k)
    pos, neg = y == 1, y == 0
    sens = float(np.mean(p1[pos])) if pos.any() else 0.5
    spec = float(np.mean(1.0 - p1[neg])) if neg.any() else 0.5
    return sens, spec


def corrected_faithfulness(corpus: dict, leg: str, params: dict, rng, k: int = K) -> dict:
    """The naive vs Rogan-Gladen-corrected faithfulness for `leg`, with the system's empirical (audited)
    confusion rates: {p_obs (naive), sens_hat, spec_hat, youden, pi_corrected, pi_oracle, var_inflation}."""
    v = judge_verdicts(corpus, leg, params, rng, k)
    truth = binary_truth(corpus)
    y = np.array([[truth[q, candidate_ids(corpus, leg, k)[q, i]] for i in range(k)]
                  for q in range(corpus["n_queries"])])
    cr = confusion_rates(v, y)
    p_obs = observed_positive_rate(v)
    J = cr["sens"] + cr["spec"] - 1.0
    return {"p_obs": p_obs, "sens_hat": cr["sens"], "spec_hat": cr["spec"], "youden": J,
            "pi_corrected": rogan_gladen(p_obs, cr["sens"], cr["spec"]),
            "pi_oracle": float(np.mean(oracle_faithfulness(corpus, leg, k))),
            "var_inflation": (1.0 / (J * J)) if abs(J) > 1e-9 else float("nan")}


# =========================================================================== #
# PILLAR 2 — agreement is not accuracy: Cohen's kappa, the paradox, Gwet's AC1, Krippendorff, the floor.
# =========================================================================== #

def observed_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """p_o = fraction of items the two raters agree on."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(a == b)) if a.size else 0.0


def expected_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """p_e = chance agreement = sum over categories of the product of the two raters' marginals."""
    a, b = np.asarray(a), np.asarray(b)
    n = a.size
    if n == 0:
        return 0.0
    pe = 0.0
    for c in (0, 1):
        pe += (np.mean(a == c)) * (np.mean(b == c))
    return float(pe)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's chance-corrected agreement kappa = (p_o - p_e)/(1 - p_e). GUARD: p_e >= 1 -> 0.0. Twin of
    sklearn.metrics.cohen_kappa_score (asserted < 1e-9)."""
    po, pe = observed_agreement(a, b), expected_agreement(a, b)
    return float((po - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 0.0


def _kappa_from_counts(t: tuple[int, int, int, int]) -> dict:
    """Cohen's kappa AND Gwet's AC1 from a 2x2 table (a, b, c, d) = (both+, A+/B-, A-/B+, both-)."""
    a, b, c, d = t
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)          # Cohen chance term
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-12 else 0.0
    q = ((a + b) + (a + c)) / (2.0 * n)                             # mean '+' prevalence over raters
    pe_g = 2.0 * q * (1.0 - q)                                      # Gwet chance term (binary)
    ac1 = (po - pe_g) / (1.0 - pe_g) if (1.0 - pe_g) > 1e-12 else 0.0
    return {"po": po, "pe": pe, "kappa": kappa, "pe_gwet": pe_g, "ac1": ac1}


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet's AC1 = (p_o - p_e)/(1 - p_e) with p_e = 2q(1-q), q the overall '+' prevalence (the binary
    special case of (1/(Q-1)) sum_k pi_k(1-pi_k)). Stable under the skewed marginals that break kappa."""
    a, b = np.asarray(a), np.asarray(b)
    n = a.size
    if n == 0:
        return 0.0
    po = observed_agreement(a, b)
    q = (float(np.mean(a == 1)) + float(np.mean(b == 1))) / 2.0
    pe = 2.0 * q * (1.0 - q)
    return float((po - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 0.0


def krippendorff_alpha_binary(ratings: np.ndarray) -> float:
    """Krippendorff's alpha for binary nominal data via the coincidence matrix, allowing missing values
    (nan). alpha = 1 - (n_total - 1) * D_o / D_e_unnormalized, the standard nominal form. Returns 1.0 for
    perfect agreement; GUARD: < 2 pairable values -> nan."""
    R = np.asarray(ratings, dtype=float)                           # (n_items, n_raters), nan = missing
    o = np.zeros((2, 2))
    for row in R:
        vals = row[~np.isnan(row)].astype(int)
        m = vals.size
        if m < 2:
            continue
        for j in range(m):
            for l in range(m):
                if j != l:
                    o[vals[j], vals[l]] += 1.0 / (m - 1)
    n = o.sum()
    if n < 2:
        return float("nan")
    nc = o.sum(axis=1)                                             # category totals
    Do = sum(o[c, k] for c in range(2) for k in range(2) if c != k)
    De = sum(nc[c] * nc[k] for c in range(2) for k in range(2) if c != k) / (n - 1)
    return float(1.0 - Do / De) if De > 1e-12 else 1.0


def kappa_paradox_tables() -> dict:
    """Two 2x2 tables with IDENTICAL observed agreement p_o = 0.85 but very different Cohen kappa
    (skew-driven), while Gwet's AC1 stays stable. The non-vacuous paradox headline — values RUN here."""
    balanced = _kappa_from_counts((40, 10, 5, 45))                 # p_o = 0.85, balanced marginals
    skewed = _kappa_from_counts((80, 10, 5, 5))                    # p_o = 0.85, skewed marginals
    return {"balanced": balanced, "skewed": skewed,
            "po_gap": abs(balanced["po"] - skewed["po"]),
            "kappa_gap": abs(balanced["kappa"] - skewed["kappa"]),
            "ac1_gap": abs(balanced["ac1"] - skewed["ac1"])}


def judge_panel_ratings(corpus: dict, leg: str, panel=JUDGE_PANEL, seed: int = SEED) -> np.ndarray:
    """(n_queries, n_judges) matrix of per-query faithfulness, one column per panel judge — independent
    rng sub-streams (deterministic). The object whose variance decomposes into query vs judge effects."""
    ss = np.random.SeedSequence(seed)
    streams = [np.random.default_rng(s) for s in ss.spawn(len(panel))]
    cols = [judged_faithfulness(corpus, leg, p, streams[j]) for j, p in enumerate(panel)]
    return np.column_stack(cols)


def variance_components(ratings: np.ndarray) -> dict:
    """Two-way (query x judge, single observation per cell) ANOVA decomposition of the ratings matrix.
    Returns the sums of squares (SS_total = SS_query + SS_judge + SS_error, an EXACT identity), the mean
    squares, the random-effects variance components, and ICC(2,1) — the two-way random-effects,
    absolute-agreement reliability (Shrout-Fleiss). GUARD: degenerate shape -> zeros."""
    X = np.asarray(ratings, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return {"ss_total": 0.0, "ss_query": 0.0, "ss_judge": 0.0, "ss_error": 0.0,
                "var_query": 0.0, "var_judge": 0.0, "var_error": 0.0, "icc21": 0.0}
    grand = float(np.mean(X))
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)
    ss_total = float(np.sum((X - grand) ** 2))
    ss_query = float(k * np.sum((row_means - grand) ** 2))
    ss_judge = float(n * np.sum((col_means - grand) ** 2))
    ss_error = ss_total - ss_query - ss_judge
    ms_query = ss_query / (n - 1)
    ms_judge = ss_judge / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    var_query = max((ms_query - ms_error) / k, 0.0)
    var_judge = max((ms_judge - ms_error) / n, 0.0)
    var_error = max(ms_error, 0.0)
    denom = ms_query + (k - 1) * ms_error + k * (ms_judge - ms_error) / n
    icc21 = (ms_query - ms_error) / denom if denom > 1e-12 else 0.0
    return {"ss_total": ss_total, "ss_query": ss_query, "ss_judge": ss_judge, "ss_error": ss_error,
            "var_query": var_query, "var_judge": var_judge, "var_error": var_error,
            "icc21": float(icc21)}


def precision_floor_vs_n(vc: dict, n_grid=(10, 20, 40, 80, 160, 320, 640), n_judges: int = 1) -> list[dict]:
    """The standard error of the mean faithfulness vs query count Q at a fixed number of judges J:
    SE^2 = var_query/Q + var_judge/J + var_error/(Q*J). The query component shrinks as 1/Q; the
    judge-variance term var_judge/J is a FLOOR no number of queries can lower."""
    vq, vj, ve = vc["var_query"], vc["var_judge"], vc["var_error"]
    rows = []
    for Q in n_grid:
        se_query = math.sqrt(vq / Q + ve / (Q * n_judges))
        se_floor = math.sqrt(vj / n_judges)
        rows.append({"n": int(Q), "se_query": se_query, "se_floor": se_floor,
                     "se_total": math.sqrt(vq / Q + vj / n_judges + ve / (Q * n_judges))})
    return rows


def budget_lever(vc: dict, budget: int = 200) -> dict:
    """The nested-variance lever at a fixed call budget (Q*J = budget): compare {Q=40, J=5} to
    {Q=budget, J=1}. The single-judge arm drives query error to near zero but is stuck at the FULL judge
    variance var_judge/1; the multi-judge arm averages the judge term down to var_judge/5. With genuine
    judge heterogeneity (var_judge large) the multi-judge arm has the lower SE — more judges beats more
    queries for precision. Returns both SEs and the winner (pinned to the observed comparison)."""
    vq, vj, ve = vc["var_query"], vc["var_judge"], vc["var_error"]

    def se(Q, J):
        return math.sqrt(vq / Q + vj / J + ve / (Q * J))

    se_multi = se(40, 5)
    se_single = se(budget, 1)
    return {"se_multi_5j": se_multi, "se_single_1j": se_single,
            "multi_judge_wins": se_multi < se_single, "budget": int(budget)}


# =========================================================================== #
# PILLAR 3 — judge calibration (reuse the prereq's suite) and the paired swap test for position bias.
# =========================================================================== #

def judge_ece_table(corpus: dict, leg: str, params: dict, n_bins: int = 10) -> dict:
    """ECE/MCE/Brier of the judge's stated confidence, raw vs Platt vs isotonic — REUSING the prereq's
    calibration suite verbatim on the judge confidences. Reports AUC raw vs Platt (a strictly monotone
    recalibration leaves it unchanged — calibration is orthogonal to ranking)."""
    conf, y = judge_confidence(corpus, leg, params)
    a, b = platt_scale(conf, y)
    platt = apply_platt(conf, a, b)
    iso = apply_isotonic(conf, isotonic_calibrate(conf, y))
    out = {}
    for name, p in (("raw", conf), ("platt", platt), ("isotonic", iso)):
        out[name] = {"ece": expected_calibration_error(p, y, n_bins),
                     "mce": max_calibration_error(p, y, n_bins),
                     "brier": brier_score(p, y)}
    out["platt_params"] = {"a": a, "b": b}
    out["auc_raw"] = auc_pooled(conf, y)
    out["auc_platt"] = auc_pooled(platt, y)
    return out


def swap_test_bias(corpus: dict, leg: str, params: dict, k: int = K) -> dict:
    """The paired swap test for position bias. For each claim, compute the judge's confidence shown at
    its actual position i and at the mirrored position k-1-i, holding the verbosity/self features fixed;
    the paired difference (shown-early minus shown-late) isolates the position effect. Reuses the
    prereq's paired_t_test and permutation_test. Detects bias iff b_pos > 0; a b_pos=0 control does not
    reject. Returns {bias, t_p, perm_p, n}."""
    Y = binary_truth(corpus)
    len_z = doc_length_feature(corpus)
    smask = self_feature(corpus, leg)
    cand = candidate_ids(corpus, leg, k)
    nq = corpus["n_queries"]
    diffs = []
    for q in range(nq):
        for i in range(k):
            d = cand[q, i]
            base = params["b_len"] * len_z[d] + params["b_self"] * smask[q, d]
            y = Y[q, d]
            base_logit = _logit(params["sens0"]) if y == 1 else _logit(1.0 - params["spec0"])
            ce = expit(base_logit + base + params["b_pos"] * 0.5)     # claim shown FIRST
            cl = expit(base_logit + base + params["b_pos"] * -0.5)    # claim shown LAST
            diffs.append(ce - cl)
    d = np.array(diffs)
    tt = paired_t_test(d)
    perm = permutation_test(d)
    return {"bias": float(np.mean(d)) if d.size else 0.0, "t_p": tt["p"], "perm_p": perm["p"],
            "n": int(d.size)}


# =========================================================================== #
# PILLAR 4 — Dawid-Skene latent-class EM: estimate each judge's error rates with NO gold labels.
# =========================================================================== #

def planted_judge_panel(n_items: int = 400, sens=(0.92, 0.80, 0.70, 0.85, 0.78),
                        spec=(0.88, 0.90, 0.75, 0.82, 0.80), prior: float = 0.5,
                        seed: int = SEED) -> dict:
    """Generate a panel of judge verdicts with KNOWN per-judge sensitivity/specificity over n_items with
    a controlled true-label prevalence — so Dawid-Skene's recovery can be checked against the plant.
    Returns {ratings (n_items, n_judges), truth, sens, spec}."""
    rng = np.random.default_rng(seed)
    z = (rng.random(n_items) < prior).astype(int)                  # latent true labels
    J = len(sens)
    R = np.empty((n_items, J), dtype=int)
    for j in range(J):
        p1 = np.where(z == 1, sens[j], 1.0 - spec[j])
        R[:, j] = (rng.random(n_items) < p1).astype(int)
    return {"ratings": R, "truth": z, "sens": np.array(sens), "spec": np.array(spec)}


def dawid_skene_em(ratings: np.ndarray, n_iter: int = 100, tol: float = 1e-6, seed: int = SEED) -> dict:
    """The Dawid-Skene (1979) EM for the latent-class observer-error model with NO gold labels. E-step:
    posterior over each item's latent class; M-step: closed-form class prior and per-judge confusion
    matrices. Init by majority vote (breaks symmetry); the hard labels are ALIGNED to the majority vote
    to fix the label-permutation ambiguity. Returns {labels_hard, class_prior, sens, spec, n_iter_run,
    loglik}. GUARD: < 2 judges or degenerate -> majority vote."""
    R = np.asarray(ratings, dtype=int)
    n, J = R.shape
    C = 2
    if J < 2 or n == 0:
        maj = (R.mean(axis=1) >= 0.5).astype(int) if n else np.array([], dtype=int)
        return {"labels_hard": maj, "class_prior": np.array([0.5, 0.5]),
                "sens": np.full(J, 0.5), "spec": np.full(J, 0.5), "n_iter_run": 0, "loglik": 0.0}
    # one-hot the verdicts: V[j] is (n, C)
    V = [np.eye(C)[R[:, j]] for j in range(J)]
    # init T by majority vote (soft)
    maj = (R.mean(axis=1) >= 0.5).astype(int)
    T = np.full((n, C), 1e-6)
    T[np.arange(n), maj] = 1.0
    T /= T.sum(axis=1, keepdims=True)
    loglik = 0.0
    run = 0
    for run in range(1, n_iter + 1):
        # M-step
        prior = T.mean(axis=0)
        theta = []
        for j in range(J):
            num = T.T @ V[j]                                       # (C true, C observed)
            theta.append(num / np.clip(num.sum(axis=1, keepdims=True), 1e-12, None))
        # E-step (log domain)
        logT = np.log(np.clip(prior, 1e-12, None))[None, :].repeat(n, axis=0)
        for j in range(J):
            logT = logT + V[j] @ np.log(np.clip(theta[j].T, 1e-12, None))
        m = logT.max(axis=1, keepdims=True)
        Tn = np.exp(logT - m)
        Tn /= Tn.sum(axis=1, keepdims=True)
        new_ll = float(np.sum(m + np.log(np.sum(np.exp(logT - m), axis=1, keepdims=True))))
        if abs(new_ll - loglik) < tol and run > 1:
            T, loglik = Tn, new_ll
            break
        T, loglik = Tn, new_ll
    labels = T.argmax(axis=1)
    # align latent class 1 with the majority-vote '+' (resolve label-permutation symmetry)
    if np.mean(labels == maj) < 0.5:
        labels = 1 - labels
        T = T[:, ::-1]
        theta = [th[::-1][:, ::-1] for th in theta]
    sens = np.array([theta[j][1, 1] for j in range(J)])
    spec = np.array([theta[j][0, 0] for j in range(J)])
    return {"labels_hard": labels, "class_prior": T.mean(axis=0), "sens": sens, "spec": spec,
            "n_iter_run": int(run), "loglik": float(loglik)}


# =========================================================================== #
# Headline flips — RUN before they are written into prose/viz (the obvious version is often vacuous).
# =========================================================================== #

def correction_flips_ranking(corpus: dict, params: dict = JUDGE_LENIENT, seed: int = SEED) -> dict:
    """FLIP (a): does Rogan-Gladen correction REVERSE the ranking of two systems? The SAME lenient judge
    inflates the system whose candidates are longer/earlier MORE; raw faithfulness can rank a worse
    system above a better one, while the corrected (and oracle) order disagrees. Scans leg pairs, returns
    the first natural flip with the corrected order matching the oracle; falls back to a constructed toy."""
    rng = np.random.default_rng(seed)
    rows = {leg: corrected_faithfulness(corpus, leg, params, np.random.default_rng(seed)) for leg in LEG_NAMES}
    for a, b in itertools.combinations(LEG_NAMES, 2):
        ra, rb = rows[a], rows[b]
        raw = ra["p_obs"] - rb["p_obs"]
        corr = ra["pi_corrected"] - rb["pi_corrected"]
        orac = ra["pi_oracle"] - rb["pi_oracle"]
        if raw * corr < 0 and corr * orac > 0 and abs(raw) > 1e-6 and abs(orac) > 1e-6:
            return {"kind": "natural", "pair": (a, b), "raw_gap": raw, "corr_gap": corr,
                    "oracle_gap": orac, "rows": {a: ra, b: rb}}
    return {"kind": "constructed", **_constructed_correction_flip()}


def _constructed_correction_flip() -> dict:
    """A hand-built two-system toy guaranteeing the raw-vs-corrected ranking flip (mirrors NDCG's
    constructed flips). System A: lower true faithfulness but a judge that over-endorses it (high p_obs);
    System B: higher true faithfulness, less inflated. Corrected order matches the truth; raw does not."""
    # (p_obs, sens, spec, pi_oracle) chosen so raw favours A but RG-corrected favours B (= oracle).
    A = {"p_obs": 0.80, "sens_hat": 0.95, "spec_hat": 0.55, "pi_oracle": 0.50}
    B = {"p_obs": 0.74, "sens_hat": 0.90, "spec_hat": 0.85, "pi_oracle": 0.66}
    A["pi_corrected"] = rogan_gladen(A["p_obs"], A["sens_hat"], A["spec_hat"])
    B["pi_corrected"] = rogan_gladen(B["p_obs"], B["sens_hat"], B["spec_hat"])
    return {"pair": ("system_A", "system_B"), "raw_gap": A["p_obs"] - B["p_obs"],
            "corr_gap": A["pi_corrected"] - B["pi_corrected"], "oracle_gap": A["pi_oracle"] - B["pi_oracle"],
            "rows": {"system_A": A, "system_B": B}}


# =========================================================================== #
# viz_constants — every number LLMJudgeLaboratory.tsx mirrors (cast numpy scalars).
# =========================================================================== #

def viz_constants() -> None:
    corpus = get_corpus()
    nq = corpus["n_queries"]

    print("\n=== shared constants ===")
    print(f"N_QUERIES = {nq}  N_DOCS = {corpus['n_docs']}  K = {K}  N_JUDGES = {N_JUDGES}  SEED = {SEED}")

    print("\n=== Panel A — the noisy instrument (Rogan-Gladen) ===")
    rng = np.random.default_rng(SEED)
    for leg in LEG_NAMES:
        cf = corrected_faithfulness(corpus, leg, JUDGE_LENIENT, np.random.default_rng(SEED))
        print(f"  {leg:16s} p_obs={round(cf['p_obs'], 4)} sens_hat={round(cf['sens_hat'], 4)} "
              f"spec_hat={round(cf['spec_hat'], 4)} youden={round(cf['youden'], 4)} "
              f"pi_corr={round(cf['pi_corrected'], 4)} pi_oracle={round(cf['pi_oracle'], 4)} "
              f"var_infl={round(cf['var_inflation'], 3)}")
    print("  RG_TABLE (p_obs, sens, spec) -> pi_hat  [the closed-form anchor TS recomputes]:")
    for (pi, se, sp) in [(0.5, 0.9, 0.9), (0.3, 0.8, 0.85), (0.7, 0.95, 0.6)]:
        pobs = pi * se + (1 - pi) * (1 - sp)
        print(f"    pi={pi} se={se} sp={sp} -> p_obs={round(pobs, 4)} pi_hat={round(rogan_gladen(pobs, se, sp, clip=False), 4)}")

    print("\n=== Panel B — inter-rater agreement + the kappa paradox + the variance floor ===")
    kp = kappa_paradox_tables()
    print(f"  PARADOX balanced: po={round(kp['balanced']['po'], 4)} kappa={round(kp['balanced']['kappa'], 4)} ac1={round(kp['balanced']['ac1'], 4)}")
    print(f"  PARADOX skewed:   po={round(kp['skewed']['po'], 4)} kappa={round(kp['skewed']['kappa'], 4)} ac1={round(kp['skewed']['ac1'], 4)}")
    print(f"  GAPS po={round(kp['po_gap'], 4)} kappa={round(kp['kappa_gap'], 4)} ac1={round(kp['ac1_gap'], 4)}")
    ratings = judge_panel_ratings(corpus, "dense")
    vc = variance_components(ratings)
    print(f"  VAR_COMPONENTS var_query={round(vc['var_query'], 5)} var_judge={round(vc['var_judge'], 5)} "
          f"var_error={round(vc['var_error'], 5)} icc21={round(vc['icc21'], 4)}")
    print(f"  SS identity: total={round(vc['ss_total'], 5)} "
          f"sum={round(vc['ss_query'] + vc['ss_judge'] + vc['ss_error'], 5)}")
    print("  PRECISION_FLOOR_VS_N (J=1): [n, se_query, se_floor, se_total]")
    for r in precision_floor_vs_n(vc, n_judges=1):
        print(f"    {r['n']:4d}  {round(r['se_query'], 5)}  {round(r['se_floor'], 5)}  {round(r['se_total'], 5)}")
    bl = budget_lever(vc)
    print(f"  BUDGET_LEVER se(40q,5j)={round(bl['se_multi_5j'], 5)} se({bl['budget']}q,1j)={round(bl['se_single_1j'], 5)} "
          f"multi_wins={bl['multi_judge_wins']}")

    print("\n=== Panel C — judge-confidence calibration (DENSE leg, three judge profiles) ===")
    n_pairs = corpus["n_queries"] * K
    base_rate = float(np.mean(binary_truth(corpus)[np.arange(corpus["n_queries"])[:, None],
                                                   candidate_ids(corpus, "dense", K)]))
    print(f"  N_PAIRS={n_pairs}  BASE_RATE={round(base_rate, 4)}")
    for jname, jp in (("lenient", JUDGE_LENIENT), ("balanced", JUDGE_BALANCED), ("strict", JUDGE_STRICT)):
        t = judge_ece_table(corpus, "dense", jp)
        sb = swap_test_bias(corpus, "dense", jp)
        print(f"  [{jname}] ECE raw={round(t['raw']['ece'], 4)} platt={round(t['platt']['ece'], 4)} "
              f"iso={round(t['isotonic']['ece'], 4)}  MCE raw={round(t['raw']['mce'], 4)}  "
              f"AUC raw={round(t['auc_raw'], 6)} platt={round(t['auc_platt'], 6)}  "
              f"platt(a,b)=({round(t['platt_params']['a'], 4)},{round(t['platt_params']['b'], 4)})  "
              f"SWAP bias={round(sb['bias'], 4)} t_p={sb['t_p']:.3e}")
    ctrl = swap_test_bias(corpus, "dense", JUDGE_HOMOG)
    print(f"  SWAP_CONTROL (bias-free, b_pos=0): bias={round(ctrl['bias'], 4)} t_p={ctrl['t_p']:.3e}")
    print("  RELIABILITY_BINS [conf, acc, n] (dense x judge x cond; TS recomputes ECE/MCE):")
    for jname, jp in (("lenient", JUDGE_LENIENT), ("balanced", JUDGE_BALANCED), ("strict", JUDGE_STRICT)):
        conf, y = judge_confidence(corpus, "dense", jp)
        a, b = platt_scale(conf, y)
        conds = {"raw": conf, "platt": apply_platt(conf, a, b),
                 "isotonic": apply_isotonic(conf, isotonic_calibrate(conf, y))}
        for cname, p in conds.items():
            compact = [[round(bb["conf"], 4), round(bb["acc"], 4), bb["n"]]
                       for bb in reliability_diagram(p, y, 10)]
            print(f"    {jname}/{cname}: {compact}")

    print("\n=== Panel D — Dawid-Skene EM (recover error rates with no gold labels) + ranking flip ===")
    pl = planted_judge_panel()
    ds = dawid_skene_em(pl["ratings"])
    print(f"  planted sens={[round(float(v), 3) for v in pl['sens']]}")
    print(f"  EM      sens={[round(float(v), 3) for v in ds['sens']]}")
    print(f"  planted spec={[round(float(v), 3) for v in pl['spec']]}")
    print(f"  EM      spec={[round(float(v), 3) for v in ds['spec']]}")
    maj = (pl["ratings"].mean(axis=1) >= 0.5).astype(int)
    print(f"  EM label acc={round(float(np.mean(ds['labels_hard'] == pl['truth'])), 4)} "
          f"majority acc={round(float(np.mean(maj == pl['truth'])), 4)} "
          f"n_iter={ds['n_iter_run']} prior={[round(float(v), 3) for v in ds['class_prior']]}")
    flip = correction_flips_ranking(corpus)
    print(f"  RANK_FLIP ({flip['kind']}) pair={flip['pair']} raw_gap={round(flip['raw_gap'], 4)} "
          f"corr_gap={round(flip['corr_gap'], 4)} oracle_gap={round(flip['oracle_gap'], 4)}")
    for name, r in flip["rows"].items():
        print(f"    {name}: p_obs={round(r['p_obs'], 4)} sens={round(r['sens_hat'], 4)} "
              f"spec={round(r['spec_hat'], 4)} pi_corr={round(r['pi_corrected'], 4)} pi_oracle={round(r['pi_oracle'], 4)}")


# =========================================================================== #
# Verification harness — every pedagogical claim is an assert.
# =========================================================================== #

def test_perfect_judge_collapse() -> None:
    # a perfect judge's faithfulness == the imported precision_at_k per query, exactly (the collapse anchor).
    c = get_corpus()
    rng = np.random.default_rng(SEED)
    for leg in LEG_NAMES:
        jf = judged_faithfulness(c, leg, JUDGE_PERFECT, rng)
        oracle = oracle_faithfulness(c, leg)
        assert np.max(np.abs(jf - oracle)) < 1e-12, (leg, float(np.max(np.abs(jf - oracle))))


def test_truth_nests_qrels() -> None:
    # the binary truth has exactly |R| positives per query and equals qrels_set.
    c = get_corpus()
    Y = binary_truth(c)
    for q in range(c["n_queries"]):
        assert int(Y[q].sum()) == len(c["qrels_set"][q]), q
        assert set(np.flatnonzero(Y[q]).tolist()) == set(int(d) for d in c["qrels_set"][q]), q


def test_rogan_gladen_recovers_pi_exact() -> None:
    # known (pi, sens, spec): the unclipped RG estimator recovers pi to machine precision; and at
    # sens=spec=1 it reduces to p_obs (no correction).
    for pi, se, sp in [(0.5, 0.9, 0.9), (0.2, 0.8, 0.85), (0.7, 0.95, 0.6), (0.33, 0.7, 0.7)]:
        pobs = pi * se + (1 - pi) * (1 - sp)
        assert abs(rogan_gladen(pobs, se, sp, clip=False) - pi) < 1e-12, (pi, se, sp)
    assert abs(rogan_gladen(0.42, 1.0, 1.0, clip=False) - 0.42) < 1e-12


def test_rogan_gladen_unbiased_in_expectation() -> None:
    # for a HOMOGENEOUS judge with known constant rates, averaging RG over many judge draws ~ the oracle
    # prevalence (unbiased in expectation), within a Monte-Carlo band.
    c = get_corpus()
    leg = "dense"
    se, sp = JUDGE_HOMOG["sens0"], JUDGE_HOMOG["spec0"]
    oracle = float(np.mean(oracle_faithfulness(c, leg)))
    ss = np.random.SeedSequence(SEED)
    ests = []
    for s in ss.spawn(400):
        v = judge_verdicts(c, leg, JUDGE_HOMOG, np.random.default_rng(s))
        ests.append(rogan_gladen(observed_positive_rate(v), se, sp, clip=False))
    mean_est = float(np.mean(ests))
    se_mc = float(np.std(ests, ddof=1)) / math.sqrt(len(ests))
    assert abs(mean_est - oracle) < 4.0 * se_mc + 0.02, (mean_est, oracle, se_mc)


def test_variance_explodes_toward_coinflip() -> None:
    # the 1/J^2 inflation is strictly increasing as sens+spec -> 1, matches the closed form, and RG is
    # non-identifiable (nan) exactly at the Youden line.
    infl = [rogan_gladen_variance(0.5, s, s, 100) for s in (0.95, 0.8, 0.7, 0.6)]
    js = [2 * s - 1 for s in (0.95, 0.8, 0.7, 0.6)]
    for v, J in zip(infl, js):
        assert abs(v - 0.25 / (100 * J * J)) < 1e-12, (v, J)
    assert infl[0] < infl[1] < infl[2] < infl[3], infl
    assert math.isnan(rogan_gladen(0.5, 0.5, 0.5)), "coin-flip judge must be non-identifiable"
    assert math.isnan(rogan_gladen_variance(0.5, 0.5, 0.5, 100))


def test_naive_faithfulness_is_biased() -> None:
    # a lenient (over-endorsing) judge inflates faithfulness above the oracle; RG correction pulls it back.
    c = get_corpus()
    rng = np.random.default_rng(SEED)
    cf = corrected_faithfulness(c, "dense", JUDGE_LENIENT, rng)
    assert cf["p_obs"] > cf["pi_oracle"], cf
    assert abs(cf["pi_corrected"] - cf["pi_oracle"]) < abs(cf["p_obs"] - cf["pi_oracle"]), cf


def test_cohen_kappa_twin() -> None:
    # the hand kappa matches sklearn.metrics.cohen_kappa_score (the reused-routine twin), AND the
    # counts-based kappa matches the vector-based one (two independent derivations).
    c = get_corpus()
    ss = np.random.SeedSequence(SEED)
    a = judge_verdicts(c, "dense", JUDGE_LENIENT, np.random.default_rng(ss.spawn(1)[0])).ravel().astype(int)
    b = judge_verdicts(c, "dense", JUDGE_STRICT, np.random.default_rng(ss.spawn(1)[0])).ravel().astype(int)
    assert abs(cohen_kappa(a, b) - cohen_kappa_score(a, b)) < 1e-9, (cohen_kappa(a, b), cohen_kappa_score(a, b))
    # 2x2 counts vs vectors
    tab = (int(np.sum((a == 1) & (b == 1))), int(np.sum((a == 1) & (b == 0))),
           int(np.sum((a == 0) & (b == 1))), int(np.sum((a == 0) & (b == 0))))
    assert abs(_kappa_from_counts(tab)["kappa"] - cohen_kappa(a, b)) < 1e-9


def test_kappa_paradox() -> None:
    # two tables with EQUAL observed agreement but a large kappa gap, while AC1 stays stable.
    kp = kappa_paradox_tables()
    assert kp["po_gap"] < 1e-9, kp
    assert kp["kappa_gap"] > 0.3, kp
    assert kp["ac1_gap"] < 0.5 * kp["kappa_gap"], kp                # AC1 spread << kappa spread (robust)


def test_krippendorff_sane() -> None:
    # perfect agreement -> alpha 1; a small hand panel is finite and in range.
    perfect = np.array([[1, 1], [0, 0], [1, 1], [0, 0]], dtype=float)
    assert abs(krippendorff_alpha_binary(perfect) - 1.0) < 1e-9
    c = get_corpus()
    ratings = (judge_panel_ratings(c, "dense") >= 0.5).astype(float)   # binarize per-query faithfulness
    a = krippendorff_alpha_binary(ratings)
    assert np.isfinite(a) and -1.0 <= a <= 1.0, a


def test_icc_variance_identity() -> None:
    # the two-way ANOVA sums-of-squares identity is exact; ICC(2,1) is a valid reliability in [-?,1].
    c = get_corpus()
    vc = variance_components(judge_panel_ratings(c, "dense"))
    assert abs(vc["ss_total"] - (vc["ss_query"] + vc["ss_judge"] + vc["ss_error"])) < 1e-9, vc
    assert vc["icc21"] <= 1.0 + 1e-9, vc


def test_judge_variance_is_a_floor() -> None:
    # with heterogeneous judges (var_judge > 0), the query-sampling SE shrinks in Q while the judge floor
    # is constant; the total SE plateaus above a positive floor as Q grows.
    c = get_corpus()
    vc = variance_components(judge_panel_ratings(c, "dense"))
    assert vc["var_judge"] > 0.0, vc
    rows = precision_floor_vs_n(vc, n_judges=1)
    se_q = [r["se_query"] for r in rows]
    se_f = [r["se_floor"] for r in rows]
    assert all(se_q[i] > se_q[i + 1] for i in range(len(se_q) - 1)), se_q     # query SE strictly down
    assert max(se_f) - min(se_f) < 1e-12, se_f                                 # floor flat
    assert rows[-1]["se_total"] > min(se_f) - 1e-12                            # plateaus above the floor


def test_budget_lever_multi_judge_wins() -> None:
    # at equal call budget, the 5-judge arm has lower SE than the 1-judge arm (more judges beats more
    # queries for precision), given genuine judge heterogeneity.
    c = get_corpus()
    vc = variance_components(judge_panel_ratings(c, "dense"))
    bl = budget_lever(vc)
    assert bl["multi_judge_wins"], bl


def test_judge_overconfident_then_recalibrated() -> None:
    # the raw judge confidence is miscalibrated (ECE > 0.1 on the lenient leg); Platt AND isotonic lower it.
    c = get_corpus()
    t = judge_ece_table(c, "dense", JUDGE_LENIENT)
    assert t["raw"]["ece"] > 0.1, t["raw"]["ece"]
    assert t["platt"]["ece"] < t["raw"]["ece"], t
    assert t["isotonic"]["ece"] < t["raw"]["ece"], t


def test_platt_preserves_judge_ranking() -> None:
    # Platt (a > 0, strictly monotone) leaves the pooled AUC of the judge confidence EXACTLY unchanged.
    c = get_corpus()
    conf, y = judge_confidence(c, "dense", JUDGE_LENIENT)
    a, b = platt_scale(conf, y)
    assert a > 0.0, a
    assert abs(auc_pooled(conf, y) - auc_pooled(apply_platt(conf, a, b), y)) < 1e-12


def test_swap_detects_injected_bias() -> None:
    # the paired swap test detects an injected position bias (b_pos > 0 -> bias > 0, p < 0.05) but a
    # b_pos=0 control does NOT reject.
    c = get_corpus()
    sb = swap_test_bias(c, "dense", JUDGE_LENIENT)
    assert sb["bias"] > 0.0 and sb["t_p"] < 0.05 and sb["perm_p"] < 0.05, sb
    ctrl = swap_test_bias(c, "dense", JUDGE_HOMOG)
    assert abs(ctrl["bias"]) < 1e-9 and ctrl["t_p"] > 0.05, ctrl


def test_dawid_skene_recovers_error_rates() -> None:
    # with no gold labels, EM recovers the planted per-judge sens/spec within tolerance, and its hard
    # labels are at least as accurate as majority vote.
    pl = planted_judge_panel()
    ds = dawid_skene_em(pl["ratings"])
    assert np.max(np.abs(ds["sens"] - pl["sens"])) < 0.12, (ds["sens"], pl["sens"])
    assert np.max(np.abs(ds["spec"] - pl["spec"])) < 0.12, (ds["spec"], pl["spec"])
    maj = (pl["ratings"].mean(axis=1) >= 0.5).astype(int)
    assert np.mean(ds["labels_hard"] == pl["truth"]) >= np.mean(maj == pl["truth"]) - 1e-9


def test_dawid_skene_guards() -> None:
    # degenerate inputs return majority vote without crashing.
    out = dawid_skene_em(np.array([[1], [0], [1]]))                # single judge
    assert out["n_iter_run"] == 0 and out["labels_hard"].tolist() == [1, 0, 1]


def test_correction_flips_ranking() -> None:
    # the raw-vs-corrected ranking flip is real: raw and corrected disagree in sign, and the corrected
    # order matches the oracle.
    c = get_corpus()
    flip = correction_flips_ranking(c)
    assert flip["raw_gap"] * flip["corr_gap"] < 0, flip                # raw vs corrected disagree
    assert flip["corr_gap"] * flip["oracle_gap"] > 0, flip             # corrected matches oracle


def test_guards() -> None:
    # confusion-rate empty-class sentinels and RG non-identifiability.
    cr = confusion_rates(np.array([[True, True]]), np.array([[1, 1]]))   # no negatives
    assert cr["spec"] == 0.5, cr
    assert math.isnan(rogan_gladen(0.5, 0.5, 0.5))


def _run_all() -> None:
    print("llm_as_judge_ragas — verifying every claim:")
    test_perfect_judge_collapse()
    test_truth_nests_qrels()
    test_rogan_gladen_recovers_pi_exact()
    test_rogan_gladen_unbiased_in_expectation()
    test_variance_explodes_toward_coinflip()
    test_naive_faithfulness_is_biased()
    test_cohen_kappa_twin()
    test_kappa_paradox()
    test_krippendorff_sane()
    test_icc_variance_identity()
    test_judge_variance_is_a_floor()
    test_budget_lever_multi_judge_wins()
    test_judge_overconfident_then_recalibrated()
    test_platt_preserves_judge_ranking()
    test_swap_detects_injected_bias()
    test_dawid_skene_recovers_error_rates()
    test_dawid_skene_guards()
    test_correction_flips_ranking()
    test_guards()
    print("all llm-as-judge tests passed")
    viz_constants()


if __name__ == "__main__":
    _run_all()
