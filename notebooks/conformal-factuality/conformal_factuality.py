"""Conformal factuality — turning a calibrated judge confidence into a distribution-free guarantee.

The reference implementation for the formalRAG `conformal-factuality` topic, the fifth and final node
of the evaluation layer. Its prerequisites built one through-line: a retrieval metric is an ESTIMATOR
(set-metrics), graded (NDCG), tested/calibrated/monitored-for-drift (significance-testing-calibration),
and — once generation enters — read through a noisy, biased INSTRUMENT, the LLM judge, whose verdicts we
debias and whose confidence we calibrate (llm-as-judge-ragas). Every prior topic produced a NUMBER with
a standard error. This topic produces a GUARANTEE.

THE UNIFYING THESIS. Conformal prediction converts ANY nonconformity score into a finite-sample coverage
guarantee under exchangeability alone — no model of when the LLM hallucinates, no distributional
assumption. We take the calibrated per-claim judge confidence built upstream as the score, run SPLIT
CONFORMAL for a recall guarantee (we rarely drop a genuinely-faithful claim), then CONFORMAL RISK CONTROL
(Angelopoulos et al. 2024) for the guarantee that actually matters — the false-claim rate among what we
emit — which split conformal's 0/1 coverage loss cannot deliver. A RAG-over-filings system that BACKS OFF
unsupported claims at a guaranteed error rate is the finance thread: abstain rather than hallucinate a
figure, at a rate an auditor could sign off on.

Two facts organize the code. (1) VALIDITY is calibration-agnostic: coverage holds for the raw judge
confidence too; recalibration (the prereq's Platt/isotonic) buys EFFICIENCY (more claims retained at the
same guarantee), not validity. (2) The guarantee is MARGINAL and rests on EXCHANGEABILITY, which DRIFT
breaks — so the last movement injects covariate shift, watches coverage fall below 1 - alpha, and repairs
it with weighted conformal (Tibshirani et al. 2019).

We IMPORT the corpus, the calibrated-judge machinery, the calibration suite, and the precision@k anchor
from the prereqs (which themselves import the published retrieval stack); we never reimplement them. The
judge is SYNTHETIC (the prereq's Bernoulli rater of the imported vMF/MaxSim oracle), which is exactly what
lets us PROVE the collapse anchor exactly and DEMONSTRATE the guarantees.

rigorFlag (primary): every guarantee here is MARGINAL, not conditional — averaged over the claim/query
draw, with no per-query or per-domain promise (distribution-free CONDITIONAL coverage is impossible at
finite informative-set size, formalML's Foygel-Barber et al. 2021). It rests on EXCHANGEABILITY of the
calibration and test claims, which intra-answer dependence and query drift stress and covariate shift
breaks; the weighted repair assumes a KNOWN likelihood ratio. Conformal Risk Control controls E[loss] in
EXPECTATION over the calibration draw, not with high probability (the delta-level alternative is RCPS,
Bates et al. 2021, named not built). And "faithful" means faithful UNDER THE SYNTHETIC JUDGE ORACLE: the
whole edifice inherits llm-as-judge-ragas's synthetic-oracle caveat — distribution-free w.r.t. the data,
conditional on the judge.

Run:  uv run --with numpy --with scipy --with scikit-learn \\
        python notebooks/conformal-factuality/conformal_factuality.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Import the two prereqs + the published stack. Add EVERY ancestor's hyphenated dir to the path (importing
# the judge/calibration corpus pulls the whole retrieval subtree at import time), then the two direct
# prereqs llm-as-judge-ragas (the calibrated judge confidence + the precision@k collapse anchor) and
# significance-testing-calibration (the calibration suite). We never reimplement them, never import a
# downstream topic.
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
    "llm-as-judge-ragas",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from set_metrics_precision_recall_map_mrr import LEG_NAMES, precision_at_k          # noqa: E402
from significance_testing_calibration import (                                       # noqa: E402
    get_corpus, platt_scale, apply_platt, isotonic_calibrate, apply_isotonic,
    expected_calibration_error, auc_pooled,
)
from llm_as_judge_ragas import (                                                      # noqa: E402
    K, judge_confidence, oracle_faithfulness, doc_length_feature, candidate_ids, JUDGE_PERFECT,
)

SEED = 0
ALPHA = 0.10                       # the central miscoverage / risk level (the slider's default)
WORKED_LEG = "dense"               # the worked answer for Panel B (a valid LEG_NAMES key)
# A realistic LENIENT judge: informative (AUC ~ 0.90) but over-endorses (spec0 < sens0), so its faithful
# and unfaithful confidences OVERLAP and unsupported claims leak through at high confidence — the gap a
# recall guarantee cannot see and conformal risk control must close. (JUDGE_BALANCED separates the two
# classes perfectly on this corpus, which would make every conformal trade-off vacuous.)
JUDGE = dict(sens0=0.80, spec0=0.62, b_len=1.3, b_pos=0.7, b_self=0.6)
CALIB_FRAC = 0.5                   # split-conformal calibration fraction (whole answers)
ALPHA_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)   # the risk-coverage sweep
LAMBDA_GRID = tuple(round(x, 4) for x in np.linspace(0.0, 1.0, 51))  # confidence-cut grid for CRC
SHIFT_BETAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)             # covariate-shift strengths (known tilt exp(-beta*v))
SHIFT_SEEDS = (0, 1, 2, 3, 4)                             # average coverage over shifted-test resamples
B_LOSS = 1.0                       # the false-claim loss is bounded in [0, 1]


_CORPUS: dict | None = None


def corpus(seed: int = SEED) -> dict:
    """Module-scope cache: the ONE NDCG/judge corpus all movements share — no second corpus."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = get_corpus(seed)
    return _CORPUS


# =========================================================================== #
# Movement 0 — the score: the calibrated per-claim judge confidence, split into calib / test answers.
# =========================================================================== #

def pooled_confidence(corpus: dict, leg: str = WORKED_LEG, params: dict = JUDGE, k: int = K):
    """The judge's per-claim confidence + truth, pooled over all (query, claim) pairs, with the query
    index of each pair. Reshaping by (n_queries, k) recovers the per-answer structure. IMPORTED verbatim
    from llm-as-judge-ragas — `conf` is the judge's stated P(supported), `y` the binary oracle truth."""
    conf, y = judge_confidence(corpus, leg, params, k)
    qidx = np.repeat(np.arange(corpus["n_queries"]), k)
    return conf, y, qidx


def calibrated_confidence(conf: np.ndarray, y: np.ndarray, calib_mask: np.ndarray,
                          method: str = "platt") -> np.ndarray:
    """Recalibrate the judge confidence into a probability, FITTING the recalibrator on the calibration
    answers only and applying it everywhere. 'identity' returns the raw confidence (validity holds for it
    too — recalibration buys efficiency, not validity). GUARD: an empty calibration half -> identity."""
    conf = np.asarray(conf, dtype=float)
    if method == "identity" or not np.any(calib_mask):
        return conf
    sc, yc = conf[calib_mask], np.asarray(y, dtype=float)[calib_mask]
    if method == "platt":
        a, b = platt_scale(sc, yc)
        return apply_platt(conf, a, b)
    if method == "isotonic":
        iso = isotonic_calibrate(sc, yc)
        return apply_isotonic(conf, iso)
    raise ValueError(f"unknown method {method}")


def split_masks(qidx: np.ndarray, n_queries: int, calib_frac: float = CALIB_FRAC):
    """The whole-answer calib/test split: queries 0..half-1 calibrate, the rest test. Splitting on whole
    answers (not individual claims) keeps an answer's k claims together, matching the prereq's held-out
    idiom."""
    half = max(1, int(round(n_queries * calib_frac)))
    calib = qidx < half
    return calib, ~calib, half


# =========================================================================== #
# Movement 1 — split conformal: the nonconformity score, the quantile threshold, the recall guarantee.
# =========================================================================== #

def split_conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal quantile threshold: the ceil((1-alpha)(n+1))-th smallest calibration score
    (formalML Theorem 1). GUARDS: an empty calibration set, or a rank exceeding n (small n / small alpha),
    returns +inf — the vacuous-but-valid 'cannot certify, retain everything' cap."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        return float("inf")
    rank = math.ceil((1.0 - alpha) * (n + 1))
    if rank > n:
        return float("inf")
    return float(s[rank - 1])


def confidence_threshold(scores_faithful: np.ndarray, alpha: float) -> float:
    """The back-off confidence cut tau = 1 - q_hat, where q_hat is the split-conformal quantile of the
    faithful-claim scores s = 1 - c. Retain a claim iff its calibrated confidence c >= tau. A +inf score
    quantile maps to a -inf cut (retain everything)."""
    return 1.0 - split_conformal_threshold(scores_faithful, alpha)


def back_off_retained(p_claims: np.ndarray, tau: float) -> dict:
    """Per-claim back-off for one answer: sort by descending confidence and remove the least-confident
    claims until every retained claim has calibrated confidence >= tau. Returns the retained / removed
    index sets and the abstain flag (the retained set is empty)."""
    p = np.asarray(p_claims, dtype=float)
    retained = set(np.flatnonzero(p >= tau).tolist())     # set membership, not a list-comp filter
    removed = set(range(p.size)) - retained
    return {"retained": np.array(sorted(retained), dtype=int),
            "removed": np.array(sorted(removed), dtype=int),
            "n_retained": len(retained), "abstain": len(retained) == 0, "tau": float(tau)}


# =========================================================================== #
# Movement 2 — conformal risk control: the monotone false-claim loss and the CRC threshold.
# =========================================================================== #

def false_claim_loss(p_claims: np.ndarray, y_claims: np.ndarray, lam: float, k: int = K) -> float:
    """The MONOTONE false-claim loss for one answer at confidence cut lam: (1/k) * #{retained AND
    unfaithful}. The denominator is the FIXED slot count k, so each retained-and-false indicator can only
    switch OFF as lam rises -> L is non-increasing in lam, bounded in [0, 1] (B = 1). This is what CRC
    controls. (The fraction-of-retained loss divides by the SHRINKING retained count and is NOT monotone
    -- see `fraction_loss` and `test_fraction_loss_not_monotone`.)"""
    p = np.asarray(p_claims, dtype=float)
    y = np.asarray(y_claims)
    retained = p >= lam
    return float(np.sum(retained & (y == 0)) / max(k, 1))


def fraction_loss(p_claims: np.ndarray, y_claims: np.ndarray, lam: float) -> float:
    """The NAIVE false-claim loss: #{retained AND unfaithful} / #retained. Intuitive but UNSOUND for CRC:
    its denominator shrinks with lam, so the ratio is NOT monotone (an abstained answer scores 0)."""
    p = np.asarray(p_claims, dtype=float)
    y = np.asarray(y_claims)
    retained = p >= lam
    nret = int(retained.sum())
    if nret == 0:
        return 0.0
    return float(np.sum(retained & (y == 0)) / nret)


def loss_matrix(P: np.ndarray, Y: np.ndarray, lambdas, k: int = K) -> np.ndarray:
    """The (n_answers, n_lambda) matrix of the monotone false-claim loss over the confidence-cut grid.
    Vectorized: L[q, j] = (1/k) * #{retained-and-unfaithful in answer q at cut lambdas[j]}. Non-increasing
    along each row."""
    lams = np.asarray(lambdas, dtype=float)
    P = np.asarray(P, dtype=float)
    Y = np.asarray(Y)
    retained = P[:, :, None] >= lams[None, None, :]       # (n_answers, k, n_lambda)
    false = (Y == 0)[:, :, None]
    return np.sum(retained & false, axis=1) / max(k, 1)   # (n_answers, n_lambda)


def conformal_risk_control_threshold(L: np.ndarray, lambdas, alpha: float,
                                     B: float = B_LOSS) -> float:
    """The CRC threshold (Angelopoulos, Bates, Fisch, Lei & Schuster 2024): the smallest cut lam with
    (n/(n+1)) * Rhat_n(lam) + B/(n+1) <= alpha, where Rhat_n is the mean calibration loss. Since the loss
    is non-increasing in lam, the adjusted risk is non-increasing, so the inf is the first qualifying grid
    point. GUARDS: empty calibration -> the most conservative cut; nothing clears the bound -> likewise."""
    lams = np.asarray(lambdas, dtype=float)
    n = L.shape[0]
    if n == 0:
        return float(lams[-1])
    Rhat = L.mean(axis=0)
    adjusted = (n / (n + 1.0)) * Rhat + B / (n + 1.0)
    ok = np.flatnonzero(adjusted <= alpha)
    return float(lams[ok[0]]) if ok.size else float(lams[-1])


# =========================================================================== #
# Movement 3 — covariate shift breaks exchangeability; weighted conformal repairs it.
# =========================================================================== #

def claim_covariates(corpus: dict, leg: str = WORKED_LEG, k: int = K) -> np.ndarray:
    """Per-claim covariate x = the document's verbosity (the z-scored token-dispersion feature the judge's
    length bias keys on), pooled over (query, claim). The judge's confidence depends on it, so it is a
    genuine covariate whose distribution a deployment shift can tilt while the score-given-covariate
    relationship stays fixed."""
    return doc_length_feature(corpus)[candidate_ids(corpus, leg, k)].ravel()


def weighted_conformal_threshold(scores: np.ndarray, weights: np.ndarray, alpha: float,
                                 w_test: float | None = None) -> float:
    """The weighted split-conformal quantile (Tibshirani et al. 2019): the smallest score s with
    (sum_{S_i <= s} w_i) / (sum_i w_i + w_test) >= 1 - alpha, the test point's own mass w_test sitting at
    +inf. With uniform weights and w_test = mean(w) this is EXACTLY the unweighted quantile
    s[ceil((1-alpha)(n+1))-1] (the collapse twin). GUARD: nonpositive total weight -> unweighted."""
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if s.size == 0 or w.sum() <= 0:
        return split_conformal_threshold(s, alpha)
    if w_test is None:
        w_test = float(np.mean(w))
    order = np.argsort(s, kind="stable")
    s_sorted, w_sorted = s[order], w[order]
    total = w.sum() + w_test
    cum = np.cumsum(w_sorted) / total
    ok = np.flatnonzero(cum >= (1.0 - alpha))
    return float(s_sorted[ok[0]]) if ok.size else float("inf")


# =========================================================================== #
# Movement 4 — the empirical verifications: coverage, CRC risk, the risk-coverage frontier, drift.
# =========================================================================== #

def mc_realized_coverage(corpus: dict, leg: str = WORKED_LEG, params: dict = JUDGE, alpha: float = ALPHA,
                         n_resplits: int = 300, method: str = "platt", seed: int = SEED, k: int = K) -> dict:
    """Monte-Carlo validity check for the split-conformal RECALL guarantee: over many random whole-answer
    calib/test partitions, the fraction of truly-faithful TEST claims retained should be ~ 1 - alpha. ONE
    rng stream; the judge confidence and qidx are hoisted out of the resplit loop."""
    conf, y, qidx = pooled_confidence(corpus, leg, params, k)      # hoisted invariants
    faithful = y == 1
    nq = corpus["n_queries"]
    half = max(1, int(round(nq * CALIB_FRAC)))
    rng = np.random.default_rng(seed)
    covs = []
    for _ in range(n_resplits):
        calib_q = rng.permutation(nq)[:half]
        cmask = np.isin(qidx, calib_q)
        p = calibrated_confidence(conf, y, cmask, method)
        s = 1.0 - p
        tau = split_conformal_threshold(s[cmask & faithful], alpha)
        test_sf = (~cmask) & faithful
        if not np.any(test_sf):
            continue
        covs.append(float(np.mean(s[test_sf] <= tau)))
    covs = np.asarray(covs)
    mean = float(covs.mean()) if covs.size else 0.0
    std = float(covs.std()) if covs.size else 0.0
    target = 1.0 - alpha
    se = std / math.sqrt(covs.size) if covs.size else 0.0
    return {"mean": mean, "std": std, "target": target,
            "within_mc": bool(abs(mean - target) < 3.0 * se + 0.02)}


def crc_threshold_and_risk(corpus: dict, leg: str = WORKED_LEG, params: dict = JUDGE, alpha: float = ALPHA,
                           method: str = "platt", k: int = K) -> dict:
    """Run CRC on the calibration answers, then measure the achieved risk on the held-out test answers.
    Returns the CRC cut, the achieved per-slot false-claim rate on test (should be <= alpha in
    expectation), and the test retention (mean fraction of claims kept)."""
    conf, y, qidx = pooled_confidence(corpus, leg, params, k)
    cmask, tmask, half = split_masks(qidx, corpus["n_queries"])
    p = calibrated_confidence(conf, y, cmask, method)
    nq = corpus["n_queries"]
    P, Y = p.reshape(nq, k), y.reshape(nq, k)
    calib_q, test_q = np.arange(half), np.arange(half, nq)
    L_cal = loss_matrix(P[calib_q], Y[calib_q], LAMBDA_GRID, k)
    lam = conformal_risk_control_threshold(L_cal, LAMBDA_GRID, alpha)
    L_test = loss_matrix(P[test_q], Y[test_q], (lam,), k)
    retained = P[test_q] >= lam
    return {"lambda": lam, "risk_test": float(L_test.mean()),
            "retention_test": float(np.mean(retained))}


def _split_eval(P: np.ndarray, Y: np.ndarray, calib_q, test_q, alpha: float, k: int):
    """Helper: split-conformal at level `alpha` on the faithful calib claims, evaluated on test answers.
    Returns achieved recall (faithful claims retained), per-slot false-claim rate, retention, abstain
    rate, and the confidence cut tau."""
    s = 1.0 - P
    faith_cal = (Y[calib_q] == 1)
    tau = confidence_threshold(s[calib_q][faith_cal], alpha)        # cut on faithful calib scores
    Pt, Yt = P[test_q], Y[test_q]
    retained = Pt >= tau
    faith_test = (Yt == 1)
    recall = float(np.mean(retained[faith_test])) if np.any(faith_test) else 1.0
    false_rate = float(np.mean(np.sum(retained & (Yt == 0), axis=1) / max(k, 1)))
    retention = float(np.mean(retained))
    abstain = float(np.mean(~np.any(retained, axis=1)))
    return {"tau": tau, "recall": recall, "false_rate": false_rate,
            "retention": retention, "abstain": abstain}


def risk_coverage_curve(corpus: dict, leg: str = WORKED_LEG, params: dict = JUDGE,
                        alpha_grid=ALPHA_GRID, method: str = "platt", k: int = K) -> list[dict]:
    """The risk-coverage frontier across the alpha grid. For each alpha: the SPLIT-conformal recall point
    (recall guaranteed, false-claim rate UNCONTROLLED) and the CRC point (false-claim rate guaranteed
    <= alpha). The contrast is the topic's headline: only CRC's curve respects the y = alpha line."""
    conf, y, qidx = pooled_confidence(corpus, leg, params, k)
    cmask, _, half = split_masks(qidx, corpus["n_queries"])
    p = calibrated_confidence(conf, y, cmask, method)
    nq = corpus["n_queries"]
    P, Y = p.reshape(nq, k), y.reshape(nq, k)
    calib_q, test_q = np.arange(half), np.arange(half, nq)
    L_cal_full = loss_matrix(P[calib_q], Y[calib_q], LAMBDA_GRID, k)  # reuse across alphas (hoisted)
    rows = []
    for a in alpha_grid:
        sp = _split_eval(P, Y, calib_q, test_q, a, k)
        lam = conformal_risk_control_threshold(L_cal_full, LAMBDA_GRID, a)
        ret_crc = P[test_q] >= lam
        crc_false = float(np.mean(np.sum(ret_crc & (Y[test_q] == 0), axis=1) / max(k, 1)))
        rows.append({"alpha": float(a),
                     "split_recall": sp["recall"], "split_false": sp["false_rate"],
                     "split_retention": sp["retention"], "split_abstain": sp["abstain"],
                     "crc_lambda": float(lam), "crc_false": crc_false,
                     "crc_retention": float(np.mean(ret_crc))})
    return rows


def covariate_shift_curve(corpus: dict, leg: str = WORKED_LEG, params: dict = JUDGE, alpha: float = ALPHA,
                          betas=SHIFT_BETAS, seeds=SHIFT_SEEDS, method: str = "platt", k: int = K) -> list[dict]:
    """Coverage vs covariate-shift strength beta. The deployment over-represents low-verbosity claims
    (the harder ones for this judge): the test distribution Q tilts the calibration distribution P by the
    KNOWN likelihood ratio w(x) = exp(-beta * x). The threshold is fit on the unshifted calibration half;
    coverage is measured on a Q-resampled test set. The UNWEIGHTED (split) threshold falls below 1 - alpha
    as beta grows (exchangeability broken); the WEIGHTED-conformal threshold, using the known w, restores
    it. Averaged over resample seeds."""
    conf, y, qidx = pooled_confidence(corpus, leg, params, k)
    v = claim_covariates(corpus, leg, k)
    cmask, _, _ = split_masks(qidx, corpus["n_queries"])
    p = calibrated_confidence(conf, y, cmask, method)
    s = 1.0 - p
    faithful = y == 1
    cal, test = cmask & faithful, (~cmask) & faithful
    s_cal, v_cal = s[cal], v[cal]
    s_test, v_test = s[test], v[test]
    rows = []
    for beta in betas:
        w_cal = np.exp(-beta * v_cal)
        w_test = np.exp(-beta * v_test)
        tau = split_conformal_threshold(s_cal, alpha)                                   # unweighted
        tau_w = weighted_conformal_threshold(s_cal, w_cal, alpha, w_test=float(np.mean(w_test)))
        probs = w_test / w_test.sum() if w_test.sum() > 0 else None
        split_covs, weighted_covs = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(s_test.size, size=s_test.size, replace=True, p=probs)      # Q-sample
            qs = s_test[idx]
            split_covs.append(float(np.mean(qs <= tau)))
            weighted_covs.append(float(np.mean(qs <= tau_w)))
        rows.append({"beta": float(beta), "target": 1.0 - alpha,
                     "split": float(np.mean(split_covs)), "weighted": float(np.mean(weighted_covs))})
    return rows


# =========================================================================== #
# Movement 5 — the worked answer for Panel B, and the judge-calibration efficiency note.
# =========================================================================== #

def worked_answer(corpus: dict, leg: str = WORKED_LEG, params: dict = JUDGE,
                  method: str = "platt", k: int = K) -> dict:
    """Pick a single illustrative TEST answer for the back-off panel: a query with both faithful and
    unfaithful retained claims and a mid-range faithfulness, so the back-off cut visibly matters. Returns
    its per-claim calibrated confidence and truth, plus the precision@k anchor for that query."""
    conf, y, qidx = pooled_confidence(corpus, leg, params, k)
    cmask, _, half = split_masks(qidx, corpus["n_queries"])
    p = calibrated_confidence(conf, y, cmask, method)
    nq = corpus["n_queries"]
    P, Y = p.reshape(nq, k), y.reshape(nq, k)
    test_q = np.arange(half, nq)
    # score each candidate query by how mixed it is (both classes present), prefer ~half faithful
    best_q, best_score = test_q[0], -1.0
    for q in test_q:
        n_faith = int(Y[q].sum())
        if 0 < n_faith < k:
            mix = 1.0 - abs(n_faith / k - 0.5)        # closest to a 50/50 mix wins
            if mix > best_score:
                best_q, best_score = int(q), mix
    pk = precision_at_k(corpus["rankings"][leg][best_q], corpus["qrels_set"][best_q], k)
    return {"query": int(best_q),
            "conf": [round(float(c), 4) for c in P[best_q]],
            "y": [int(t) for t in Y[best_q]],
            "precision_at_k": round(float(pk), 4)}


def judge_calibration_quality(corpus: dict, leg: str = WORKED_LEG, params: dict = JUDGE, k: int = K) -> dict:
    """The 'is the judge's confidence a probability?' readout that motivates recalibration: pooled ECE of
    the raw vs Platt-recalibrated judge confidence, and the AUC (unchanged by the strictly-monotone Platt
    map -- recalibration improves efficiency without touching the ranking)."""
    conf, y, qidx = pooled_confidence(corpus, leg, params, k)
    cmask, _, _ = split_masks(qidx, corpus["n_queries"])
    a, b = platt_scale(conf[cmask], y[cmask])
    platt = apply_platt(conf, a, b)
    return {"ece_raw": float(expected_calibration_error(conf, y)),
            "ece_platt": float(expected_calibration_error(platt, y)),
            "auc_raw": float(auc_pooled(conf, y)),
            "auc_platt": float(auc_pooled(platt, y))}


# =========================================================================== #
# viz_constants — every number the D3 lab bakes, mirrored TO THE DECIMAL into ConformalFactualityLaboratory.tsx.
# =========================================================================== #

def viz_constants() -> dict:
    """Print (and return) the baked panel data for the viz. Every numpy scalar is cast (round(float),
    int) so the printed values are clean to mirror into the .tsx."""
    c = corpus()
    nq = c["n_queries"]

    # ---- Panel A: calibration-quantile / coverage ----
    conf, y, qidx = pooled_confidence(c, WORKED_LEG, JUDGE)
    cmask, _, _ = split_masks(qidx, nq)
    p = calibrated_confidence(conf, y, cmask, "platt")
    s = 1.0 - p
    faithful = y == 1
    calib_scores = np.sort(s[cmask & faithful])
    tau_by_alpha = {f"{a:.2f}": round(split_conformal_threshold(calib_scores, a), 4) for a in ALPHA_GRID}
    mc = {f"{a:.2f}": mc_realized_coverage(c, WORKED_LEG, JUDGE, a) for a in ALPHA_GRID}
    mc_cov = {a: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)} for a, v in mc.items()}

    # ---- Panel B: per-claim back-off on a worked answer ----
    wa = worked_answer(c, WORKED_LEG, JUDGE)

    # ---- Panel C: risk-coverage frontier + CRC-vs-split ----
    rc = risk_coverage_curve(c, WORKED_LEG, JUDGE)
    rc_round = [{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()} for r in rc]

    # ---- Panel D: covariate shift breaks coverage, weighted repairs ----
    dc = covariate_shift_curve(c, WORKED_LEG, JUDGE, ALPHA)
    dc_round = [{kk: round(vv, 4) for kk, vv in r.items()} for r in dc]
    break_beta = next((r["beta"] for r in dc if r["split"] < r["target"] - 0.02
                       and r["weighted"] >= r["target"] - 0.02), None)

    cal = judge_calibration_quality(c, WORKED_LEG, JUDGE)

    out = {
        "shared": {"n_queries": int(nq), "n_docs": int(c["n_docs"]), "k": int(K),
                   "alpha": ALPHA, "worked_leg": WORKED_LEG, "judge": "lenient",
                   "alpha_grid": list(ALPHA_GRID),
                   "calib_quality": {kk: round(vv, 4) for kk, vv in cal.items()}},
        "panelA": {"calib_scores": [round(float(x), 4) for x in calib_scores],
                   "tau_by_alpha": tau_by_alpha, "mc_coverage": mc_cov},
        "panelB": wa,
        "panelC": rc_round,
        "panelD": {"curve": dc_round, "break_beta": break_beta},
    }

    print("=" * 78)
    print("conformal-factuality viz_constants  (mirror TO THE DECIMAL into the .tsx)")
    print("=" * 78)
    print(f"shared: {out['shared']}")
    print("-" * 78)
    print(f"Panel A calib_scores (n={len(out['panelA']['calib_scores'])}): {out['panelA']['calib_scores']}")
    print(f"Panel A tau_by_alpha:  {out['panelA']['tau_by_alpha']}")
    print(f"Panel A mc_coverage:   {out['panelA']['mc_coverage']}")
    print("-" * 78)
    print(f"Panel B worked answer query={out['panelB']['query']}  precision@k={out['panelB']['precision_at_k']}")
    print(f"Panel B conf: {out['panelB']['conf']}")
    print(f"Panel B y:    {out['panelB']['y']}")
    print("-" * 78)
    print("Panel C risk-coverage frontier (alpha, split_recall, split_false, split_retention, "
          "crc_false, crc_retention):")
    for r in out["panelC"]:
        print(f"   a={r['alpha']:.2f}  split_recall={r['split_recall']:.3f}  split_false={r['split_false']:.3f}"
              f"  split_ret={r['split_retention']:.3f}  crc_false={r['crc_false']:.3f}"
              f"  crc_ret={r['crc_retention']:.3f}")
    print("-" * 78)
    print(f"Panel D covariate-shift coverage (break_beta={out['panelD']['break_beta']}):")
    for r in out["panelD"]["curve"]:
        print(f"   beta={r['beta']:.2f}  split={r['split']:.3f}  weighted={r['weighted']:.3f}  "
              f"target={r['target']:.3f}")
    print("=" * 78)
    return out


# =========================================================================== #
# Test harness — every pedagogical claim is an assert.
# =========================================================================== #

def test_shares_one_corpus() -> None:
    """The judge confidence here IS the imported prereq array — one corpus, no re-derivation."""
    c = corpus()
    conf, y, _ = pooled_confidence(c, WORKED_LEG, JUDGE)
    conf2, y2 = judge_confidence(c, WORKED_LEG, JUDGE, K)
    assert np.allclose(conf, conf2) and np.array_equal(y, y2)
    assert conf.size == c["n_queries"] * K


def test_split_conformal_quantile_rank() -> None:
    """The threshold is exactly the ceil((1-alpha)(n+1))-th order statistic, with the +inf guards."""
    s = np.linspace(0.0, 1.0, 100)
    for a in (0.05, 0.1, 0.2):
        rank = math.ceil((1.0 - a) * (len(s) + 1))
        expected = float(np.sort(s)[rank - 1]) if rank <= len(s) else float("inf")
        assert split_conformal_threshold(s, a) == expected
    assert split_conformal_threshold(np.array([]), 0.1) == float("inf")     # empty
    assert split_conformal_threshold(np.array([0.3]), 0.01) == float("inf")  # rank > n


def test_perfect_judge_collapse() -> None:
    """The collapse anchor. A PERFECT judge separates faithful (s~0) from unfaithful (s~1) cleanly, so:
    (a) for any reasonable cut the retained set is exactly the faithful claims -> zero false-claim rate at
    every grid alpha; (b) the per-query retained fraction equals the imported oracle_faithfulness ==
    precision_at_k to < 1e-12."""
    c = corpus()
    for leg in LEG_NAMES:
        conf, y, qidx = pooled_confidence(c, leg, JUDGE_PERFECT)
        nq = c["n_queries"]
        P, Y = conf.reshape(nq, K), y.reshape(nq, K)
        oracle = oracle_faithfulness(c, leg)
        # (b) a clean cut at 0.5 recovers the faithful set exactly, every query
        for q in range(nq):
            bo = back_off_retained(P[q], 0.5)
            retained_frac = bo["n_retained"] / K
            assert abs(retained_frac - oracle[q]) < 1e-12
            assert np.all(Y[q][bo["retained"]] == 1)        # no unfaithful claim retained
        # (a) zero false-claim at every grid alpha (n_faithful large -> rank never overflows to +inf)
        faithful = y == 1
        calib = qidx < max(1, int(round(nq * CALIB_FRAC)))
        for a in ALPHA_GRID:
            tau = confidence_threshold((1.0 - conf)[calib & faithful], a)
            false_kept = int(np.sum((conf >= tau) & (y == 0)))
            assert false_kept == 0


def test_back_off_removes_least_confident() -> None:
    """Back-off retains exactly {p >= tau}; the min retained confidence exceeds the max removed; an
    all-low-confidence answer abstains."""
    p = np.array([0.9, 0.7, 0.4, 0.2, 0.95])
    bo = back_off_retained(p, 0.5)
    assert set(bo["retained"].tolist()) == {0, 1, 4}
    assert set(bo["removed"].tolist()) == {2, 3}
    assert p[bo["retained"]].min() >= 0.5 > p[bo["removed"]].max()
    assert back_off_retained(np.array([0.1, 0.2, 0.3]), 0.5)["abstain"]


def test_false_claim_loss_monotone() -> None:
    """The fixed-denominator loss is non-increasing along the confidence-cut grid for every answer."""
    c = corpus()
    conf, y, _ = pooled_confidence(c, WORKED_LEG, JUDGE)
    nq = c["n_queries"]
    P, Y = conf.reshape(nq, K), y.reshape(nq, K)
    L = loss_matrix(P, Y, LAMBDA_GRID, K)
    assert np.all(np.diff(L, axis=1) <= 1e-12)


def test_fraction_loss_not_monotone() -> None:
    """The naive fraction-of-retained loss CAN rise as the cut rises -- the reason CRC needs the
    fixed-denominator loss. We exhibit the counterexample numerically."""
    # 10 claims: one false at confidence 0.55, nine true clustered just below it at 0.50..0.54.
    p = np.array([0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.504, 0.503, 0.502, 0.501])
    y = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    lo = fraction_loss(p, y, 0.50)     # retain all: 1 false / 10 = 0.1
    hi = fraction_loss(p, y, 0.545)    # retain only the false claim: 1 / 1 = 1.0
    assert lo < hi                      # NON-monotone: loss rose as the cut rose


def test_crc_monotone_and_bound() -> None:
    """The CRC empirical risk is non-increasing in the cut; lambda_hat is the inf clearing the corrected
    bound; the achieved test risk is <= alpha within Monte-Carlo tolerance (the marginal CRC guarantee)."""
    c = corpus()
    res = crc_threshold_and_risk(c, WORKED_LEG, JUDGE, ALPHA)
    # monotone empirical risk
    conf, y, qidx = pooled_confidence(c, WORKED_LEG, JUDGE)
    cmask, _, half = split_masks(qidx, c["n_queries"])
    p = calibrated_confidence(conf, y, cmask, "platt")
    P, Y = p.reshape(c["n_queries"], K), y.reshape(c["n_queries"], K)
    L = loss_matrix(P[:half], Y[:half], LAMBDA_GRID, K)
    assert np.all(np.diff(L.mean(axis=0)) <= 1e-12)
    assert res["risk_test"] <= ALPHA + 0.05           # marginal guarantee, MC slack


def test_realized_coverage_near_target() -> None:
    """Split-conformal recall coverage tracks 1 - alpha (the exchangeable, un-drifted regime)."""
    c = corpus()
    for a in (0.05, 0.10, 0.20):
        mc = mc_realized_coverage(c, WORKED_LEG, JUDGE, a)
        assert mc["within_mc"], (a, mc)


def test_weighted_collapses_to_split_under_no_shift() -> None:
    """With uniform weights and w_test = mean(w), the weighted quantile is EXACTLY the unweighted
    split-conformal threshold (the twin anchor for the weighted machinery)."""
    s = np.linspace(0.0, 1.0, 50)
    w = np.ones_like(s)
    for a in (0.05, 0.1, 0.2):
        assert abs(weighted_conformal_threshold(s, w, a) - split_conformal_threshold(s, a)) < 1e-12


def test_covariate_shift_breaks_then_weighted_restores() -> None:
    """Under covariate shift the unweighted coverage falls below 1 - alpha while the weighted-conformal
    coverage (known likelihood ratio) is restored to the target -- the Panel D headline, run not assumed."""
    c = corpus()
    curve = covariate_shift_curve(c, WORKED_LEG, JUDGE, ALPHA)
    target = curve[0]["target"]
    broken = [r for r in curve if r["split"] < target - 0.02]
    assert broken, "expected some shift level to break split coverage"
    worst = max(broken, key=lambda r: r["beta"])
    assert worst["weighted"] > worst["split"]                       # weighting helps where split fails
    assert worst["weighted"] >= target - 0.03                       # and restores toward the target


def test_guards() -> None:
    """The denominator / empty-input guards hold."""
    assert split_conformal_threshold(np.array([]), 0.1) == float("inf")
    assert weighted_conformal_threshold(np.array([0.1, 0.2]), np.array([0.0, 0.0]), 0.1) \
        == split_conformal_threshold(np.array([0.1, 0.2]), 0.1)
    assert false_claim_loss(np.array([0.1, 0.2]), np.array([0, 0]), 0.5) == 0.0   # all removed
    assert back_off_retained(np.array([]), 0.5)["abstain"]


def _run_all() -> None:
    tests = [test_shares_one_corpus, test_split_conformal_quantile_rank, test_perfect_judge_collapse,
             test_back_off_removes_least_confident, test_false_claim_loss_monotone,
             test_fraction_loss_not_monotone, test_crc_monotone_and_bound,
             test_realized_coverage_near_target, test_weighted_collapses_to_split_under_no_shift,
             test_covariate_shift_breaks_then_weighted_restores, test_guards]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print("all conformal-factuality tests passed\n")


if __name__ == "__main__":
    _run_all()
    viz_constants()
