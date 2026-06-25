"""Selective Generation: When a RAG System Should Abstain — the canonical reference.

Run (CPU-only, < 60 s):
    uv run --with numpy --with scipy --with scikit-learn \
        python notebooks/selective-generation-abstention/selective_generation_abstention.py

The terminal node of the generation-grounding layer. Its two predecessors measured and CERTIFIED an
answer at the CLAIM level — `faithfulness-groundedness` (precision/recall over an answer's claims) and
`conformal-factuality` (which claims to keep, at a guaranteed false-claim rate). Both back off WITHIN an
answer: they decide *which claims to keep*. This topic asks the ANSWER-LEVEL question the back-off
frontier set up — *whether to answer at all*.

WHAT IS NEW HERE (vs the imports). The per-query EMIT-or-ABSTAIN gate and the objects that govern it:
  * Chow's rule — the cost-optimal reject threshold: abstain iff P(wrong) > c_abs / c_err (Chow 1970);
  * the risk-coverage curve and its area AURC — selective risk vs the fraction of queries answered, with
    units of QUERIES, not claims (the answer-level mirror of AP = area-under-PR);
  * the achievable-vs-oracle gap — the excess AURC a real (imperfect) confidence signal pays, which
    shrinks to zero only as the signal's AUC -> 1 (the gap IS the signal's quality / calibration);
  * an ANSWER-LEVEL conformal selective-risk guarantee — the imported conformal-risk-control machinery
    applied to a NEW, answer-level monotone loss (the unconditional wrong-emission rate); and
  * the two-stage composition: claim-level conformal back-off -> answer-level abstain when the certified
    answer is too THIN, plus an abstain-vs-err COST model (defer to a human analyst rather than guess).

Everything else is IMPORTED from the published stack and never reimplemented: the corpus, the answer
generator, the noisy lenient judge, the calibration suite, the conformal threshold, the per-claim
back-off. The two pedagogical prerequisites are `faithfulness-groundedness` (the two-sided measurement
this gate sits on) and `significance-testing-calibration` (the calibration that makes a confidence
score a probability); everything else (conformal-factuality, llm-as-judge-ragas, set-metrics) is an
in-site `connections[]` sibling whose functions we reuse to source numbers (import graph != the DAG).

rigorFlag (primary). (1) CHOW OPTIMALITY ASSUMES A CALIBRATED, KNOWN POSTERIOR. "Abstain iff P(wrong) >
c_abs/c_err" is Bayes-optimal only if the answer score IS the true P(correct). The lenient judge's score
is not — so the REALIZED selective risk differs from the BAYES selective risk, and the difference is
exactly a CALIBRATION gap (we read it off with the imported ECE / Brier on the answer score vs the
oracle). The Chow cutoff is only as good as the calibration of the score it cuts. (2) The conformal
selective-risk guarantee is MARGINAL and holds IN EXPECTATION over the calibration draw, not per
realization (exchangeability; a single split can overshoot); and it controls the UNCONDITIONAL
wrong-emission rate, while the conditional selective risk (the risk-coverage y-axis) carries an extra
1/coverage factor and is NON-monotone — the answer-level echo of conformal-factuality's
false-claim-loss vs fraction-loss distinction. (3) AURC and its achievable-vs-oracle gap are
single-corpus point estimates with no standard error here (a query bootstrap, via the significance
prereq, would give a CI). (4) Inherits the SYNTHETIC-JUDGE-ORACLE caveat: "correct" means correct under
the geometric support oracle and the synthetic judge — distribution-free w.r.t. the data, conditional on
the judge.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Import the published stack. Add EVERY ancestor's hyphenated dir to the path (importing the
# faithfulness / judge / calibration / conformal modules pulls the whole retrieval + evaluation subtree),
# then import the UNDERSCORED modules. We never reimplement an imported routine and never import a
# downstream topic (there is none — this is the terminal node).
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "the-retrieval-problem",
    "infonce-contrastive-objective",
    "vector-quantization-lloyd-max",
    "product-quantization",
    "ivf-voronoi-partitioning",
    "bm25",
    "vector-space-model-tfidf",
    "query-likelihood-language-models",
    "probability-ranking-principle",
    "dense-retrieval-dual-encoders",
    "late-interaction-learned-sparse",
    "multi-vector-ann-retrieval",
    "set-metrics-precision-recall-map-mrr",
    "ndcg-discount-geometry",
    "significance-testing-calibration",
    "llm-as-judge-ragas",
    "conformal-factuality",
    "pseudo-relevance-feedback",
    "query-transformation-hyde",
    "pmi-retrieval-value",
    "faithfulness-groundedness",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from faithfulness_groundedness import (                                           # noqa: E402
    _corpus, build_panel, panel_confidence, calibrated_panel_conf, split_panel,
    crc_backoff, abstention_frontier, answer_faithfulness,
    ALPHA, CALIB_FRAC, LAMBDA_GRID, PANEL_NCLAIMS,
)
from conformal_factuality import (                                                # noqa: E402
    conformal_risk_control_threshold, split_conformal_threshold,
    weighted_conformal_threshold, back_off_retained,
)
from significance_testing_calibration import (                                    # noqa: E402
    auc_pooled, expected_calibration_error, brier_score,
)
from llm_as_judge_ragas import JUDGE_PERFECT                                      # noqa: E402


# --------------------------------------------------------------------------- #
# Module constants — tuned by a build-and-run `_diagnostics()` sweep, never guessed. The panel is the
# imported faithfulness panel at a per-query answer; we choose its faithfulness rate so the base
# correct-rate is non-degenerate (the easy-corpus trap), and a lenient over-endorsing judge so the
# answer score orders correctness IMPERFECTLY (the perfectly-orders trap).
# --------------------------------------------------------------------------- #
PANEL_SEED = 3                     # the imported build_panel default seed (one answer per query)
SEL_R = 0.72                       # the panel's per-slot faithfulness rate (base correct-rate knob): tuned
                                   # so base correct ~0.59 AND the imported lenient judge's answer score has
                                   # AUC ~0.77 (informative but imperfect -> a real achievable-vs-oracle gap,
                                   # and selective generation beats BOTH always-emit and always-abstain)
FAITH_FLOOR = 0.75                 # an answer is "correct" iff >= this fraction of its claims is supported

# the abstain-vs-err cost model (Chow). A wrong emitted financial answer costs far more than deferring
# to a human analyst, so c_err >> c_abs -> an interior emit cutoff.
C_ERR = 5.0
C_ABS = 1.0

MIN_CLAIMS = 3                     # two-stage thinness gate: a certified answer with fewer retained claims
                                   # than this is too thin to be useful -> abstain even if faithful
THRESH_GRID = tuple(round(x, 4) for x in np.linspace(0.0, 1.0, 101))  # the score-threshold sweep

# numpy 2.x renamed np.trapz -> np.trapezoid; bind whichever exists so the area is version-safe.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# =========================================================================== #
# Movement 0 — the panel: one answer per query (the imported faithfulness panel), its calibrated judge
# confidence, and the per-answer (score, correct) pair the whole topic runs over.
# =========================================================================== #

def answer_correct(answer: dict, faith_floor: float = FAITH_FLOOR) -> int:
    """The binary ANSWER-LEVEL target: 1 iff the answer is faithful enough to emit. PROPOSED definition:
    `answer_faithfulness(answer) >= faith_floor` (a threshold of the imported precision-over-claims). The
    floor is the anti-easy-corpus knob (Trap a) AND the quantity the perfect-judge score orders exactly
    (so the perfect-judge collapse to the oracle curve holds). Coverage / responsiveness is NOT folded in
    here — a faithful-but-THIN answer is caught by the two-stage thinness gate in Movement 4, matching the
    faithfulness hand-off ('when the certified-faithful answer is too thin to be useful, abstain')."""
    return int(answer_faithfulness(answer) >= faith_floor)


def answer_score(conf_row: np.ndarray, agg: str = "mean") -> float:
    """The per-answer CONFIDENCE the emit/abstain gate keys on: aggregate the calibrated per-claim judge
    confidences for one answer into one scalar proxy for P(answer correct). PROPOSED agg='mean' (the
    Chow-posterior proxy); 'min' (weakest-link) is the conservative alternative that widens the gap to the
    oracle. This is the ACHIEVABLE ordering signal — imperfect, because the judge is noisy and lenient."""
    c = np.asarray(conf_row, dtype=float)
    if c.size == 0:
        return 0.0
    if agg == "mean":
        return float(c.mean())
    if agg == "min":
        return float(c.min())
    raise ValueError(f"unknown agg {agg}")


def selective_arrays(panel: dict, conf: np.ndarray, agg: str = "mean"):
    """The per-answer (score, correct) arrays the risk-coverage curve, the Chow rule and the conformal
    threshold all consume. `score` orders answers by confidence; `correct` is the oracle emit target."""
    scores = np.array([answer_score(conf[q], agg) for q in range(len(panel["answers"]))])
    correct = np.array([answer_correct(a) for a in panel["answers"]], dtype=int)
    return scores, correct


_SETUP: dict | None = None


def _setup() -> dict:
    """Module-scope cache: the panel, its raw + calibrated judge confidence, the calib/test split, and the
    per-answer (score, correct) pair — built once, deterministically (the panel seed is fixed at PANEL_SEED
    to preserve the viz<->python invariant, so there is no seed argument to drift)."""
    global _SETUP
    if _SETUP is None:
        corpus = _corpus()
        panel = build_panel(corpus, n_claims=PANEL_NCLAIMS, r=SEL_R, seed=PANEL_SEED)
        raw = panel_confidence(panel)
        calib, test, half = split_panel(panel["Y"].shape[0])
        conf = calibrated_panel_conf(panel, raw, calib, method="platt")
        scores, correct = selective_arrays(panel, conf)
        _SETUP = {"corpus": corpus, "panel": panel, "raw": raw, "conf": conf,
                  "calib": calib, "test": test, "half": half,
                  "scores": scores, "correct": correct}
    return _SETUP


# =========================================================================== #
# Movement 1 — Chow's rule: the cost-optimal per-query emit/abstain decision.
# =========================================================================== #

def chow_threshold(c_err: float, c_abs: float) -> float:
    """Chow's optimal reject rule (1970). The expected cost of EMITTING is c_err * P(wrong); of ABSTAINING
    is c_abs. Emit iff c_err * P(wrong) <= c_abs, i.e. iff P(correct) >= 1 - c_abs/c_err. Returns that EMIT
    CUTOFF on P(correct), clipped to [0,1]. Degenerate costs: c_abs=0 (free abstention) -> cutoff 1.0
    (emit only when certain ~ always abstain); c_abs >= c_err -> cutoff <= 0 (always emit)."""
    if c_err <= 0:
        return 0.0
    return float(np.clip(1.0 - c_abs / c_err, 0.0, 1.0))


def chow_decision(scores: np.ndarray, emit_cutoff: float) -> np.ndarray:
    """Apply the Chow rule per query: emit (1) iff the answer score >= the emit cutoff, else abstain (0)."""
    return (np.asarray(scores, dtype=float) >= emit_cutoff).astype(int)


# =========================================================================== #
# Movement 2 — the risk-coverage curve and its area (AURC). Selective risk vs the fraction of queries
# answered, with units of QUERIES (the answer-level mirror of AP = area-under-PR).
# =========================================================================== #

def selective_risk_coverage(scores: np.ndarray, correct: np.ndarray) -> list[dict]:
    """The ANSWER-LEVEL risk-coverage curve. Emit the k most-confident answers (k = 0..N); at each k report
    coverage = k/N and selective_risk = (# wrong among the emitted)/k. Units are QUERIES, not claims.
    coverage=0 carries selective_risk 0 by convention (the curve's origin). At coverage=1 the selective
    risk is the unconditional base error rate (the collapse anchor)."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n = scores.size
    if n == 0:
        return [{"coverage": 0.0, "selective_risk": 0.0, "k": 0}]
    order = np.argsort(-scores, kind="stable")              # most confident first
    wrong = (correct[order] == 0).astype(int)
    rows = [{"coverage": 0.0, "selective_risk": 0.0, "k": 0}]
    cum_wrong = 0
    for k in range(1, n + 1):
        cum_wrong += int(wrong[k - 1])
        rows.append({"coverage": k / n, "selective_risk": cum_wrong / k, "k": k})
    return rows


def aurc(rc_rows: list[dict]) -> float:
    """Area Under the Risk-Coverage curve: the trapezoidal integral of selective risk over coverage in
    [0,1] (the answer-level analog of average precision = area-under-PR). LOWER is better — a good
    confidence signal abstains on exactly the answers that would be wrong, keeping selective risk near 0
    until coverage is forced high."""
    cov = np.array([r["coverage"] for r in rc_rows], dtype=float)
    risk = np.array([r["selective_risk"] for r in rc_rows], dtype=float)
    if cov.size < 2:
        return 0.0
    return float(_trapz(risk, cov))


def oracle_rc_curve(correct: np.ndarray) -> list[dict]:
    """The ORACLE (best-possible) risk-coverage curve: order answers by TRUE correctness, so every correct
    answer is emitted before any wrong one. Selective risk stays 0 until coverage exceeds the base accuracy,
    then climbs as the policy is forced to emit wrong answers. The lower envelope any confidence signal can
    reach; the gap from the achievable curve up to it is the signal's quality."""
    correct = np.asarray(correct, dtype=int)
    n = correct.size
    if n == 0:
        return [{"coverage": 0.0, "selective_risk": 0.0, "k": 0}]
    order = np.argsort(-correct, kind="stable")            # correct (1) first
    wrong = (correct[order] == 0).astype(int)
    rows = [{"coverage": 0.0, "selective_risk": 0.0, "k": 0}]
    cum_wrong = 0
    for k in range(1, n + 1):
        cum_wrong += int(wrong[k - 1])
        rows.append({"coverage": k / n, "selective_risk": cum_wrong / k, "k": k})
    return rows


def rc_gap(scores: np.ndarray, correct: np.ndarray) -> dict:
    """The headline of Movement 2: the achievable AURC (ordering by the judge score), the oracle AURC
    (ordering by truth), their gap (the EXCESS AURC, Geifman-Uziel-El-Yaniv 2019), and the pooled AUC of
    score-vs-correct. The gap shrinks to 0 as AUC -> 1 — a perfect signal IS the oracle. This ties the
    risk-coverage story to the imported `auc_pooled`."""
    ach = aurc(selective_risk_coverage(scores, correct))
    orc = aurc(oracle_rc_curve(correct))
    auc = float(auc_pooled(np.asarray(scores, dtype=float), np.asarray(correct, dtype=float)))
    return {"aurc_achievable": ach, "aurc_oracle": orc, "gap": ach - orc, "auc": auc}


# =========================================================================== #
# Movement 3 — the answer-level conformal selective-risk guarantee. The IMPORTED conformal-risk-control
# machinery applied to a NEW, answer-level monotone loss (the unconditional wrong-emission rate).
# =========================================================================== #

def wrong_emission_loss(scores: np.ndarray, correct: np.ndarray, t: float) -> float:
    """The MONOTONE answer-level loss at score threshold t: (# emitted-and-wrong)/N, the FIXED-N
    denominator. Each answer's wrong-and-emitted indicator can only switch OFF as t rises (raising the bar
    un-emits), so the loss is non-increasing in t, bounded in [0,1] (B=1). This is what conformal risk
    control controls — the answer-level twin of conformal-factuality's monotone false_claim_loss."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n = scores.size
    if n == 0:
        return 0.0
    return float(np.sum((scores >= t) & (correct == 0)) / n)


def selective_risk_loss(scores: np.ndarray, correct: np.ndarray, t: float) -> float:
    """The NAIVE conditional selective risk at threshold t: (# emitted-and-wrong)/(# emitted). Intuitive
    (it is the risk-coverage y-axis) but UNSOUND for conformal risk control: its denominator SHRINKS as t
    rises, so it is NOT monotone — the answer-level echo of conformal-factuality's fraction_loss
    counterexample. An all-abstain threshold scores 0."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    emit = scores >= t
    n_emit = int(emit.sum())
    if n_emit == 0:
        return 0.0
    return float(np.sum(emit & (correct == 0)) / n_emit)


def selective_loss_matrix(scores: np.ndarray, correct: np.ndarray, thresholds) -> np.ndarray:
    """The (N_answers, n_threshold) matrix of the monotone wrong-emission loss: L[i,j] = 1 iff answer i is
    EMITTED at thresholds[j] (score >= t) AND wrong, else 0. Per-row 0/1 (one answer); the mean over rows
    is `wrong_emission_loss` at each threshold. Non-increasing along each row in t. Mirrors the shape
    conformal-factuality's `loss_matrix` produces, so the imported CRC threshold consumes it directly."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    lams = np.asarray(thresholds, dtype=float)
    emit = scores[:, None] >= lams[None, :]                 # (N, n_thresh)
    wrong = (correct == 0)[:, None]
    return (emit & wrong).astype(float)


def selective_conformal_threshold(scores_cal: np.ndarray, correct_cal: np.ndarray,
                                  alpha: float, thresholds=LAMBDA_GRID) -> float:
    """The answer-level conformal selective-risk threshold: build the monotone wrong-emission loss matrix on
    the calibration answers, then call the IMPORTED `conformal_risk_control_threshold` to pick the smallest
    score cut whose adjusted expected wrong-emission rate is <= alpha. The machinery is imported; only the
    LOSS is new and answer-level. The smallest qualifying cut is the highest-coverage threshold that still
    controls the risk at alpha."""
    L = selective_loss_matrix(scores_cal, correct_cal, thresholds)
    return float(conformal_risk_control_threshold(L, thresholds, alpha))


def claim_vs_answer_curves(panel: dict, conf: np.ndarray, scores: np.ndarray,
                           correct: np.ndarray) -> dict:
    """The distinctness object: the IMPORTED claim-level frontier (`abstention_frontier` — retained-claim
    precision/coverage swept by the per-claim cut, units = CLAIMS) beside the new answer-level
    risk-coverage curve (units = ANSWERS / queries). Same panel, different objects — claim-retention is not
    answer-coverage, and the two areas differ."""
    claim = abstention_frontier(panel, conf)
    answer = selective_risk_coverage(scores, correct)
    return {"claim": claim, "answer": answer}


# =========================================================================== #
# Movement 4 — the two-stage composition (claim back-off -> answer abstain) and the cost operating point.
# =========================================================================== #

def post_backoff_signal(panel: dict, conf: np.ndarray, alpha: float = ALPHA,
                        calib_frac: float = CALIB_FRAC) -> np.ndarray:
    """The answer-level THINNESS signal that is the OUTPUT of claim-level back-off: run the IMPORTED
    conformal claim back-off to get its threshold lambda_hat (`crc_backoff`), then per answer count the
    claims it certifies as faithful (`back_off_retained`). A 'certified-faithful answer too thin to be
    useful' is one with a small retained count — the second reason to abstain."""
    crc = crc_backoff(panel, conf, alpha=alpha, calib_frac=calib_frac)
    lam = crc["lambda_hat"]
    return np.array([back_off_retained(conf[q], lam)["n_retained"] for q in range(conf.shape[0])], dtype=int)


def expected_selective_cost(scores: np.ndarray, correct: np.ndarray, t: float,
                            c_err: float, c_abs: float) -> float:
    """The realized expected cost at score threshold t: per query, c_err if emitted-and-wrong, c_abs if
    abstained, 0 if emitted-and-correct; averaged. Swept over thresholds it is the U-shaped cost curve whose
    minimum sits at the Chow cutoff when the score is calibrated. (Distinct from
    probability_ranking_principle.expected_cost, a ranking-set false-positive/false-negative cost.)"""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=int)
    if scores.size == 0:
        return 0.0
    emit = scores >= t
    cost = np.where(emit, np.where(correct == 1, 0.0, c_err), c_abs)
    return float(cost.mean())


def cost_curve(scores: np.ndarray, correct: np.ndarray, c_err: float, c_abs: float,
               thresholds=THRESH_GRID) -> list[dict]:
    """The expected cost as a function of the emit threshold — the U-curve marked at the Chow minimum."""
    return [{"t": float(t), "cost": expected_selective_cost(scores, correct, t, c_err, c_abs)}
            for t in thresholds]


def two_stage_decision(panel: dict, conf: np.ndarray, c_err: float = C_ERR, c_abs: float = C_ABS,
                       min_claims: int = MIN_CLAIMS, alpha: float = ALPHA, agg: str = "mean") -> dict:
    """Compose the two stages. Stage 1: claim-level conformal back-off certifies the faithful claims (the
    imported `crc_backoff` / `back_off_retained`). Stage 2: the answer-level Chow gate — abstain iff the
    answer is too RISKY (score below the Chow emit cutoff) OR too THIN (fewer than min_claims certified
    claims). Returns the per-query decision arrays and the realized cost."""
    scores, correct = selective_arrays(panel, conf, agg)
    emit_cutoff = chow_threshold(c_err, c_abs)
    retained = post_backoff_signal(panel, conf, alpha)
    risky = scores < emit_cutoff
    thin = retained < min_claims
    emit = (~risky) & (~thin)
    cost = float(np.where(emit, np.where(correct == 1, 0.0, c_err), c_abs).mean())
    return {"emit": emit.astype(int), "abstain_risky": risky.astype(int), "abstain_thin": thin.astype(int),
            "scores": scores, "correct": correct, "retained": retained,
            "emit_cutoff": emit_cutoff, "cost": cost}


def finance_operating_point(panel: dict, conf: np.ndarray, c_err: float = C_ERR,
                            c_abs: float = C_ABS) -> dict:
    """The case study: a RAG over filings defers to a human analyst rather than emit an answer it cannot
    certify. Reports the defer rate, the residual error among emitted answers, and the realized cost against
    the always-emit and always-abstain baselines (the operating point an auditor signs off on)."""
    ts = two_stage_decision(panel, conf, c_err, c_abs)
    emit = ts["emit"].astype(bool)
    correct = ts["correct"]
    n_emit = int(emit.sum())
    residual = float(np.sum(emit & (correct == 0)) / n_emit) if n_emit else 0.0
    always_emit = float(np.where(correct == 1, 0.0, c_err).mean())
    always_abstain = float(c_abs)
    return {"defer_rate": float(1.0 - emit.mean()), "residual_error": residual,
            "cost": ts["cost"], "cost_always_emit": always_emit, "cost_always_abstain": always_abstain,
            "n_emit": n_emit, "n": int(emit.size)}


def calibration_gap(scores: np.ndarray, correct: np.ndarray, n_bins: int = 8) -> dict:
    """The rigorFlag made numeric: how far the answer score is from a true P(correct). Chow's rule treats
    `score` as P(correct); we measure its ECE and Brier against the oracle `correct`. A nonzero ECE is why
    the realized cost-minimizing threshold need not equal the Chow cutoff — the calibration gap."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=float)
    return {"ece": float(expected_calibration_error(s, y, n_bins)),
            "brier": float(brier_score(s, y)),
            "auc": float(auc_pooled(s, y))}


# =========================================================================== #
# Build-and-run diagnostics — read these separations BEFORE trusting any headline. The constants are set
# from what this prints (the base correct-rate, the score AUC, the cost-curve interior, the RC slope).
# =========================================================================== #

def _diagnostics() -> None:
    s = _setup()
    scores, correct = s["scores"], s["correct"]
    n = scores.size
    print(f"panel: {n} answers (one per query), {PANEL_NCLAIMS} claims each, faithfulness rate r={SEL_R}")
    print(f"base correct-rate (faith >= {FAITH_FLOOR}): {correct.mean():.3f}  "
          f"(want 0.15 < . < 0.85; base error {1 - correct.mean():.3f})")
    gap = rc_gap(scores, correct)
    print(f"score AUC vs correct: {gap['auc']:.3f}  (want 0.55 < . < 0.98 so achievable sits ABOVE oracle)")
    print(f"AURC achievable {gap['aurc_achievable']:.3f} vs oracle {gap['aurc_oracle']:.3f}  "
          f"-> excess (E-AURC) {gap['gap']:.3f}  (want > 0)")
    rc = selective_risk_coverage(scores, correct)
    risks = [r["selective_risk"] for r in rc]
    print(f"RC selective-risk spread: {min(risks):.3f}..{max(risks):.3f}  (want max-min > 0.1, a real slope)")
    emit_cutoff = chow_threshold(C_ERR, C_ABS)
    cc = cost_curve(scores, correct, C_ERR, C_ABS)
    costs = [r["cost"] for r in cc]
    t_star = cc[int(np.argmin(costs))]["t"]
    print(f"Chow cutoff (c_err={C_ERR}, c_abs={C_ABS}): {emit_cutoff:.3f}; empirical cost argmin t={t_star:.3f}; "
          f"cost {min(costs):.3f} vs always-emit {expected_selective_cost(scores, correct, 0.0, C_ERR, C_ABS):.3f} "
          f"/ always-abstain {C_ABS:.3f}")
    cg = calibration_gap(scores, correct)
    print(f"calibration gap of the score: ECE {cg['ece']:.3f}, Brier {cg['brier']:.3f}")
    conf_thr = selective_conformal_threshold(scores[s["calib"]], correct[s["calib"]], ALPHA)
    print(f"answer-level conformal threshold @ alpha={ALPHA}: t={conf_thr:.3f}")
    fop = finance_operating_point(s["panel"], s["conf"])
    print(f"two-stage operating point: defer {fop['defer_rate']:.3f}, residual error {fop['residual_error']:.3f}, "
          f"cost {fop['cost']:.3f} (always-emit {fop['cost_always_emit']:.3f})")


# =========================================================================== #
# Demo + viz_constants.
# =========================================================================== #

def _r(v, n: int = 4) -> float:
    return round(float(v), n)


def selective_demo() -> dict:
    """The headline numbers printed."""
    s = _setup()
    scores, correct = s["scores"], s["correct"]
    gap = rc_gap(scores, correct)
    emit_cutoff = chow_threshold(C_ERR, C_ABS)
    conf_thr = selective_conformal_threshold(scores[s["calib"]], correct[s["calib"]], ALPHA)
    realized = wrong_emission_loss(scores[s["test"]], correct[s["test"]], conf_thr)
    fop = finance_operating_point(s["panel"], s["conf"])
    print(f"panel: {scores.size} answers, base correct-rate {correct.mean():.2f}")
    print(f"MOVEMENT 1 (Chow): with c_err={C_ERR}, c_abs={C_ABS}, emit iff P(correct) >= {emit_cutoff:.2f}")
    print(f"MOVEMENT 2 (risk-coverage): achievable AURC {gap['aurc_achievable']:.3f} > oracle "
          f"{gap['aurc_oracle']:.3f}; excess {gap['gap']:.3f} closes as AUC ({gap['auc']:.2f}) -> 1")
    print(f"MOVEMENT 3 (conformal): answer-level threshold t={conf_thr:.2f} controls the test wrong-emission "
          f"rate at alpha={ALPHA} — realized {realized:.3f}")
    print(f"MOVEMENT 4 (two-stage + cost): defer {fop['defer_rate']:.2f}, residual error among emitted "
          f"{fop['residual_error']:.3f}, cost {fop['cost']:.3f} vs always-emit {fop['cost_always_emit']:.3f}")
    return {"gap": gap, "emit_cutoff": emit_cutoff, "conf_thr": conf_thr, "fop": fop}


def viz_constants() -> None:
    """Print every MEASURED number SelectiveGenerationAbstentionLaboratory.tsx mirrors to the decimal. The
    TS recomputes only CLOSED FORM from these baked inputs (the live operating point, the cost curve and
    its Chow minimum, the conformal quantile q-hat, the two-stage decision). Every number is a seeded,
    deterministic computation, so the baked values are reproducible."""
    s = _setup()
    scores, correct = s["scores"], s["correct"]
    gap = rc_gap(scores, correct)
    emit_cutoff = chow_threshold(C_ERR, C_ABS)

    print("// ----- shared constants -----")
    print(f"const N_QUERIES = {scores.size};")
    print(f"const BASE_ERROR_RATE = {_r(1 - correct.mean(), 3)};")
    print(f"const ALPHA = {ALPHA};")
    print(f"const C_ERR = {C_ERR};")
    print(f"const C_ABS = {C_ABS};")
    print(f"const EMIT_CUTOFF = {_r(emit_cutoff, 3)};   // Chow cutoff = 1 - c_abs/c_err")
    print(f"const MIN_CLAIMS = {MIN_CLAIMS};")

    print("// ----- Panel A + B: the per-answer (score, correct) cloud (the viz scans it live) -----")
    print(f"const SCORES = {[_r(v, 3) for v in scores]};")
    print(f"const CORRECT = {[int(v) for v in correct]};")

    print("// ----- Panel A: the risk-coverage curves + AURC -----")
    rc = selective_risk_coverage(scores, correct)
    orc = oracle_rc_curve(correct)
    rc_pts = [{"c": _r(r["coverage"], 3), "risk": _r(r["selective_risk"], 3)} for r in rc]
    orc_pts = [{"c": _r(r["coverage"], 3), "risk": _r(r["selective_risk"], 3)} for r in orc]
    print(f"const RC_ACHIEVABLE = {rc_pts};")
    print(f"const RC_ORACLE = {orc_pts};")
    print(f"const AURC = {{'achievable': {_r(gap['aurc_achievable'], 3)}, 'oracle': {_r(gap['aurc_oracle'], 3)}, "
          f"'gap': {_r(gap['gap'], 3)}, 'auc': {_r(gap['auc'], 3)}}};")

    print("// ----- Panel B: the Chow cost curve (U-shaped, minimum at the cutoff) -----")
    cc = cost_curve(scores, correct, C_ERR, C_ABS)
    cc_pts = [{"t": _r(r["t"], 3), "cost": _r(r["cost"], 3)} for r in cc[::2]]
    costs = [r["cost"] for r in cc]
    print(f"const COST_CURVE = {cc_pts};")
    print(f"const COST = {{'at_chow': {_r(min(costs), 3)}, "
          f"'always_emit': {_r(expected_selective_cost(scores, correct, 0.0, C_ERR, C_ABS), 3)}, "
          f"'always_abstain': {_r(C_ABS, 3)}}};")

    print("// ----- Panel C: the two-stage worked decision (score, retained, decision per query) -----")
    ts = two_stage_decision(s["panel"], s["conf"])
    rows = [{"score": _r(scores[q], 3), "correct": int(correct[q]), "retained": int(ts["retained"][q]),
             "emit": int(ts["emit"][q])} for q in range(scores.size)]
    print(f"const TWO_STAGE = {rows};")
    fop = finance_operating_point(s["panel"], s["conf"])
    print(f"const OPERATING = {{'defer_rate': {_r(fop['defer_rate'], 3)}, "
          f"'residual_error': {_r(fop['residual_error'], 3)}, 'cost': {_r(fop['cost'], 3)}, "
          f"'cost_always_emit': {_r(fop['cost_always_emit'], 3)}}};")

    print("// ----- Panel D: the answer-level conformal guarantee (calib scores + realized risk) -----")
    cal_scores = np.sort(scores[s["calib"]])
    conf_thr = selective_conformal_threshold(scores[s["calib"]], correct[s["calib"]], ALPHA)
    realized = wrong_emission_loss(scores[s["test"]], correct[s["test"]], conf_thr)
    cov_at = float((scores[s["test"]] >= conf_thr).mean())
    print(f"const CALIB_SCORES = {[_r(v, 3) for v in cal_scores]};   // ascending calibration scores")
    print(f"const CONFORMAL = {{'threshold': {_r(conf_thr, 3)}, 'realized_risk': {_r(realized, 3)}, "
          f"'coverage': {_r(cov_at, 3)}, 'alpha': {ALPHA}}};")
    cg = calibration_gap(scores, correct)
    print(f"const CAL_GAP = {{'ece': {_r(cg['ece'], 3)}, 'brier': {_r(cg['brier'], 3)}}};")


# =========================================================================== #
# Verification harness — each assert is a pedagogical claim the topic makes.
# =========================================================================== #

def test_coverage_one_is_base_error() -> None:
    """Collapse anchor: at full coverage (emit everything) the selective risk equals the unconditional base
    error rate — the risk-coverage curve's right endpoint."""
    s = _setup()
    rc = selective_risk_coverage(s["scores"], s["correct"])
    full = rc[-1]
    assert abs(full["coverage"] - 1.0) < 1e-12, "the last point must be full coverage"
    assert abs(full["selective_risk"] - (1.0 - s["correct"].mean())) < 1e-12, "full-coverage risk != base error"
    print(f"[ok] collapse: full-coverage selective risk == base error rate {full['selective_risk']:.3f}")


def test_perfect_judge_collapses_to_oracle() -> None:
    """Collapse anchor: a PERFECT judge's score (mean confidence == mean truth == faithfulness) orders the
    answers exactly by the quantity that defines `correct`, so the achievable risk-coverage curve equals the
    ORACLE curve and the score AUC is 1 — the achievable-vs-oracle gap closes to zero."""
    s = _setup()
    panel = s["panel"]
    conf = panel_confidence(panel, JUDGE_PERFECT)
    assert np.array_equal(conf, panel["Y"].astype(float)), "perfect judge must return the truth"
    scores, correct = selective_arrays(panel, conf)
    g = rc_gap(scores, correct)
    assert abs(g["gap"]) < 1e-12, f"perfect-judge achievable AURC must equal oracle AURC: {g}"
    assert abs(g["auc"] - 1.0) < 1e-12, f"perfect-judge score AUC must be 1: {g}"
    print(f"[ok] collapse: perfect judge -> achievable AURC == oracle ({g['aurc_oracle']:.3f}), AUC 1.0")


def test_chow_threshold_degenerate_costs() -> None:
    """Degenerate-cost anchor: free abstention (c_abs=0) -> emit cutoff 1.0 (effectively always abstain);
    abstaining at least as costly as erring (c_abs >= c_err) -> emit cutoff 0.0 (always emit). The decision
    arrays collapse accordingly."""
    s = _setup()
    scores = s["scores"]
    assert chow_threshold(5.0, 0.0) == 1.0, "free abstention should give cutoff 1.0"
    assert chow_threshold(5.0, 5.0) == 0.0, "c_abs == c_err should give cutoff 0.0"
    assert chow_threshold(5.0, 9.0) == 0.0, "c_abs > c_err should give cutoff 0.0 (always emit)"
    assert chow_decision(scores, 0.0).sum() == scores.size, "cutoff 0 must emit every answer"
    assert chow_decision(scores, 1.0 + 1e-9).sum() == 0, "cutoff above 1 must abstain on every answer"
    print("[ok] degenerate costs: c_abs=0 -> always abstain; c_abs>=c_err -> always emit")


def test_chow_interior_beats_baselines() -> None:
    """Chow interior: with c_err > c_abs > 0 the emit cutoff is strictly interior, and the realized expected
    cost at the empirical optimum beats BOTH always-emit and always-abstain — abstention pays."""
    s = _setup()
    scores, correct = s["scores"], s["correct"]
    cutoff = chow_threshold(C_ERR, C_ABS)
    assert 0.0 < cutoff < 1.0, f"the Chow cutoff should be interior: {cutoff}"
    costs = [expected_selective_cost(scores, correct, r["t"], C_ERR, C_ABS) for r in cost_curve(scores, correct, C_ERR, C_ABS)]
    best = min(costs)
    always_emit = expected_selective_cost(scores, correct, 0.0, C_ERR, C_ABS)
    assert best < always_emit - 1e-9, f"selective cost should beat always-emit: {best} vs {always_emit}"
    assert best < C_ABS - 1e-9, f"selective cost should beat always-abstain: {best} vs {C_ABS}"
    print(f"[ok] Chow interior: cutoff {cutoff:.2f}; cost {best:.3f} < always-emit {always_emit:.3f} / "
          f"always-abstain {C_ABS:.3f}")


def test_aurc_is_riemann_area() -> None:
    """AURC == area-under-RC, the answer-level mirror of AP == area-under-PR: a hand trapezoidal sum on a
    tiny synthetic (score, correct) equals `aurc`."""
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 0, 1, 1, 0])
    rc = selective_risk_coverage(scores, correct)
    cov = np.array([r["coverage"] for r in rc]); risk = np.array([r["selective_risk"] for r in rc])
    hand = float(_trapz(risk, cov))
    assert abs(aurc(rc) - hand) < 1e-12, "aurc must equal the trapezoidal area"
    assert abs(rc[-1]["selective_risk"] - 0.4) < 1e-12, "base error of the toy is 2/5"
    print(f"[ok] AURC == Riemann area under the RC curve ({aurc(rc):.3f})")


def test_achievable_above_oracle_not_vacuous() -> None:
    """The score must be INFORMATIVE but IMPERFECT (0.55 < AUC < 0.98), so the achievable risk-coverage
    curve sits strictly ABOVE the oracle — the excess AURC is a real, non-vacuous gap (the signal quality
    the rest of the topic is about)."""
    s = _setup()
    g = rc_gap(s["scores"], s["correct"])
    assert 0.55 < g["auc"] < 0.98, f"the judge score should order correctness imperfectly: AUC {g['auc']}"
    assert g["gap"] > 1e-3, f"the achievable AURC should exceed the oracle: gap {g['gap']}"
    print(f"[ok] achievable above oracle: AUC {g['auc']:.3f}, excess AURC {g['gap']:.3f} > 0")


def test_correctness_nondegenerate() -> None:
    """Anti-easy-corpus: the base correct-rate is interior (some answers should be emitted, some abstained)
    and the risk-coverage curve genuinely slopes (max - min selective risk > 0.1)."""
    s = _setup()
    correct = s["correct"]
    assert 0.15 < correct.mean() < 0.85, f"base correct-rate should be interior: {correct.mean()}"
    risks = [r["selective_risk"] for r in selective_risk_coverage(s["scores"], correct)]
    assert max(risks) - min(risks) > 0.1, f"the RC curve should slope: spread {max(risks) - min(risks)}"
    print(f"[ok] non-degenerate: base correct {correct.mean():.2f}, RC spread {max(risks) - min(risks):.2f}")


def test_wrong_emission_loss_monotone() -> None:
    """The answer-level wrong-emission loss is MONOTONE non-increasing in the threshold (raising the bar can
    only un-emit), while the conditional selective risk is NOT monotone — the answer-level echo of
    conformal-factuality's false_claim_loss (monotone) vs fraction_loss (non-monotone)."""
    s = _setup()
    scores, correct = s["scores"], s["correct"]
    grid = np.linspace(0.0, 1.0, 51)
    mono = np.array([wrong_emission_loss(scores, correct, t) for t in grid])
    assert np.all(np.diff(mono) <= 1e-12), "wrong-emission loss must be non-increasing in the threshold"
    cond = np.array([selective_risk_loss(scores, correct, t) for t in grid])
    assert np.any(np.diff(cond) > 1e-9), "conditional selective risk should be non-monotone (counterexample)"
    print("[ok] monotone wrong-emission loss; non-monotone conditional selective risk (the CRC distinction)")


def test_selective_conformal_controls_and_reuses() -> None:
    """The answer-level conformal threshold (1) is computed by the IMPORTED CRC machinery, equal to the
    hand-derived first qualifying grid cut (a reuse anchor, not a fork), and (2) controls the test
    wrong-emission rate near alpha."""
    s = _setup()
    scores, correct = s["scores"], s["correct"]
    sc, cc = scores[s["calib"]], correct[s["calib"]]
    L = selective_loss_matrix(sc, cc, LAMBDA_GRID)
    n = L.shape[0]; Rhat = L.mean(axis=0)
    adjusted = (n / (n + 1.0)) * Rhat + 1.0 / (n + 1.0)
    ok = np.flatnonzero(adjusted <= ALPHA)
    hand = float(LAMBDA_GRID[ok[0]]) if ok.size else float(LAMBDA_GRID[-1])
    got = selective_conformal_threshold(sc, cc, ALPHA)
    assert abs(got - hand) < 1e-12, f"conformal threshold must match the hand-derived cut: {got} vs {hand}"
    realized = wrong_emission_loss(scores[s["test"]], correct[s["test"]], got)
    assert realized <= ALPHA + 0.12, f"the conformal threshold should control the test risk: {realized}"
    print(f"[ok] conformal reuse: threshold {got:.3f} == hand-derived; test wrong-emission {realized:.3f} "
          f"<= alpha {ALPHA}+slack")


def test_weighted_collapses_to_split() -> None:
    """Reuse twin: with uniform weights the IMPORTED weighted-conformal quantile equals the unweighted
    split-conformal quantile of the answer-level nonconformity scores (s = 1 - score), to machine
    precision — the imported quantile machinery, not a fork."""
    s = _setup()
    nonconf = 1.0 - s["scores"][s["calib"]]
    w = np.ones_like(nonconf)
    assert abs(weighted_conformal_threshold(nonconf, w, ALPHA) - split_conformal_threshold(nonconf, ALPHA)) < 1e-12
    print("[ok] reuse twin: weighted-conformal == split-conformal under uniform weights (<1e-12)")


def test_answer_level_distinct_from_claim_level() -> None:
    """The distinctness thesis, made executable: the IMPORTED claim-level frontier (`abstention_frontier` —
    units of CLAIMS) and the new answer-level risk-coverage curve (units of ANSWERS) are different objects —
    different x-grids and different areas. Claim-retention is not answer-coverage."""
    s = _setup()
    cv = claim_vs_answer_curves(s["panel"], s["conf"], s["scores"], s["correct"])
    claim_x = sorted({_r(r["retention"], 3) for r in cv["claim"]})
    answer_x = sorted({_r(r["coverage"], 3) for r in cv["answer"]})
    assert claim_x != answer_x, "claim-retention grid and answer-coverage grid must differ (different objects)"
    claim_area = float(_trapz([r["faithfulness"] for r in cv["claim"]],
                                [r["retention"] for r in cv["claim"]]))
    assert abs(claim_area - aurc(cv["answer"])) > 1e-6, "claim-level area != answer-level AURC (different units)"
    print("[ok] distinct: claim-level frontier (claims) != answer-level risk-coverage (answers)")


def test_two_stage_thinness_gate() -> None:
    """The two-stage composition: some answers are abstained for THINNESS (fewer than min_claims certified
    claims) independent of the Chow risk gate — the second reason to abstain (a faithful but too-thin
    answer), exactly the faithfulness hand-off."""
    s = _setup()
    ts = two_stage_decision(s["panel"], s["conf"])
    assert ts["emit"].sum() < ts["emit"].size, "the two-stage gate should abstain on at least one answer"
    assert ts["retained"].min() >= 0, "retained counts are non-negative"
    # emit requires BOTH not-risky and not-thin
    assert np.all(ts["emit"] == ((1 - ts["abstain_risky"]) * (1 - ts["abstain_thin"]))), \
        "emit must be the AND of the two gates"
    print(f"[ok] two-stage: emit {int(ts['emit'].sum())}/{ts['emit'].size}; thin-abstained "
          f"{int(ts['abstain_thin'].sum())}, risky-abstained {int(ts['abstain_risky'].sum())}")


def test_finance_operating_point_pays() -> None:
    """The finance operating point: selective generation lowers the expected cost below always-emit, by
    deferring the risky answers to a human analyst."""
    s = _setup()
    fop = finance_operating_point(s["panel"], s["conf"])
    assert 0.0 < fop["defer_rate"] < 1.0, f"defer rate should be interior: {fop}"
    assert fop["cost"] <= fop["cost_always_emit"] + 1e-9, f"selective cost should not exceed always-emit: {fop}"
    print(f"[ok] finance: defer {fop['defer_rate']:.2f}, residual error {fop['residual_error']:.3f}, "
          f"cost {fop['cost']:.3f} <= always-emit {fop['cost_always_emit']:.3f}")


def test_viz_constants_reproducible() -> None:
    """Bake-only-reproducible: two builds of the (score, correct) cloud and the risk-coverage curve are
    bit-identical (the panel and its calibrated confidence are fully seeded; no random start leaks in)."""
    global _SETUP
    _SETUP = None
    a = _setup()
    sa, ca = a["scores"].copy(), a["correct"].copy()
    _SETUP = None
    b = _setup()
    assert np.array_equal(sa, b["scores"]) and np.array_equal(ca, b["correct"]), "the (score, correct) cloud drifts"
    assert selective_risk_coverage(sa, ca) == selective_risk_coverage(b["scores"], b["correct"]), "RC drifts"
    print("[ok] reproducible: the (score, correct) cloud and RC curve are bit-identical across builds")


def test_guards() -> None:
    """Defensive guards (the gemini-prone cases): an empty panel of answers, empty score/correct arrays and
    a zero c_err return sane sentinels rather than dividing by zero."""
    assert aurc(selective_risk_coverage(np.array([]), np.array([]))) == 0.0
    assert wrong_emission_loss(np.array([]), np.array([]), 0.5) == 0.0
    assert selective_risk_loss(np.array([]), np.array([]), 0.5) == 0.0
    assert chow_threshold(0.0, 1.0) == 0.0, "a zero error cost must not divide by zero"
    assert answer_score(np.array([])) == 0.0
    print("[ok] guards: empty arrays / zero c_err are safe")


def _run_all() -> None:
    test_coverage_one_is_base_error()
    test_perfect_judge_collapses_to_oracle()
    test_chow_threshold_degenerate_costs()
    test_chow_interior_beats_baselines()
    test_aurc_is_riemann_area()
    test_achievable_above_oracle_not_vacuous()
    test_correctness_nondegenerate()
    test_wrong_emission_loss_monotone()
    test_selective_conformal_controls_and_reuses()
    test_weighted_collapses_to_split()
    test_answer_level_distinct_from_claim_level()
    test_two_stage_thinness_gate()
    test_finance_operating_point_pays()
    test_viz_constants_reproducible()
    test_guards()


if __name__ == "__main__":
    print("selective_generation_abstention: diagnostics\n")
    _diagnostics()
    print("\nrunning tests\n")
    _run_all()
    print("\nDemo:")
    selective_demo()
    print("\nviz_constants (mirror into SelectiveGenerationAbstentionLaboratory.tsx):")
    viz_constants()
    print("\nall checks passed.")
