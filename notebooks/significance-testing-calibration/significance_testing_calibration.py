"""Significance testing, score calibration, and drift detection — comparing TWO distributions.

The reference implementation for the formalRAG `significance-testing-calibration` topic, the third
node of the evaluation layer. Its prerequisites defined a metric as an ESTIMATOR of one population
mean (set-metrics) and generalized that to graded relevance (NDCG), and BOTH left the same question
open: their Panel D used a crude OVERLAPPING-CI read to ask "is the gap real?", finding the closest
leg pair (lexical vs dense, NDCG gap ~0.05) needs an extrapolated n~185 queries to separate. That
read is conservative and not equivalent to a test. This topic closes the question with the right
instrument — a PAIRED significance test — and generalizes the SAME two-distribution comparison to two
more production-evaluation questions.

THE UNIFYING THESIS. A metric is an estimator (prereqs); here we ask whether TWO distributions of
per-query quantities DIFFER. Three instances, one instrument:
  - significance: system A vs system B on the same queries     (d_q = metric_A(q) - metric_B(q))
  - drift:        the system now vs the system then            (d = score_now - score_then)
  - calibration:  the predicted score vs the realized relevance (per-bin confidence - accuracy)
Pairing is the single geometric fact binding them: subtracting query-by-query cancels shared per-query
difficulty (var(d) = var(A) + var(B) - 2cov(A,B) << the unpaired variance, because easy queries are
easy for every leg), so the paired test resolves a gap at far fewer queries than the overlapping-CI read.

We IMPORT the corpus, legs, estimator machinery, and graded relevance from the two prereqs (which
themselves import the published retrieval stack); we never reimplement them, and we re-derive raw
per-(query, doc) SCORES locally from the imported scoring primitives (the calibration pillar needs
scores, not just rankings). We do NOT import the downstream capstone.

rigorFlag (primary): production drift monitoring is ONLINE — you re-test as each window arrives — so a
fixed-n KS/PSI p-value is invalid under continuous peeking (the multiple-looks problem), and PSI's
0.1/0.25 traffic-light thresholds are heuristics that sidestep, not solve, peeking; the principled fix
is anytime-valid inference (confidence sequences / e-values), a CONNECTION named here, not built.
Secondary: the per-query differences are bounded and skewed, so the paired t-test's normality is only
approximate (hence the permutation and bootstrap cross-checks, which are exact under exchangeability);
calibration's reliability diagram is necessarily a POOLED cross-query object (10 docs/query is too
sparse per query), so cross-query score incomparability is itself the miscalibration; the isotonic-
vs-Platt ECE edge is sensitive to in-sample overfit (hence the held-out split). SYNTHETIC vMF tokens
and a MaxSim oracle mean "relevance" is "top-k under the exact end-scorer", not human judgment.

Run:  uv run --with numpy --with scipy \\
        python notebooks/significance-testing-calibration/significance_testing_calibration.py
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from scipy.special import expit

# --------------------------------------------------------------------------- #
# Import the two prereqs + the published stack. Add EVERY ancestor's hyphenated dir to the path
# (importing the NDCG/set-metrics corpus pulls the whole multi-vector subtree at import time), then the
# two direct prereqs. We IMPORT the corpus, legs, estimator machinery, graded relevance, and the
# scoring PRIMITIVES (for re-deriving raw scores); we never reimplement them, never import the capstone.
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
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize                       # noqa: E402
from dense_retrieval_dual_encoders import dual_encoder_score         # noqa: E402
from late_interaction_learned_sparse import maxsim_matrix            # noqa: E402
from bm25 import bm25_rank                                           # noqa: E402
from multi_vector_ann_retrieval import TOPK, N_DOCS                  # noqa: E402
from set_metrics_precision_recall_map_mrr import (                   # noqa: E402
    LEG_NAMES, per_query_ap, metric_summary, projected_ci, projected_separation_n,
)
from ndcg_discount_geometry import (                                 # noqa: E402
    ndcg_corpus, per_query_ndcg, ndcg_at_k, gain_exponential, discount_log2,
    projected_ndcg_separation_n,
)

SEED = 0
ALPHA = 0.05
POWER = 0.80
EPS = 1e-9                                  # the log/division guard for PSI & KL
WORKED_PAIR = ("lexical", "dense")          # the closest-gap pair, both metrics (the cliffhanger pair)
Z_ALPHA = float(st.norm.ppf(1 - ALPHA / 2))  # 1.96, the projection multiplier (recomputed in TS)


_CORPUS: dict | None = None


def get_corpus(seed: int = SEED) -> dict:
    """Module-scope cache: the NDCG corpus (a SUPERSET of the set-metrics corpus — it adds oracle
    scores and graded relevance) is built ONCE. All three pillars share it; no second corpus."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = ndcg_corpus(seed)
    return _CORPUS


# Per-query metric samples both prereqs expose; the raw material every paired comparison differences.
def ap_samples(corpus: dict, leg: str) -> np.ndarray:
    """Per-query AP (set regime) for `leg` — the MAP samples."""
    return per_query_ap(corpus, leg, "qrels_set")


def ndcg_samples(corpus: dict, leg: str) -> np.ndarray:
    """Per-query NDCG@10 (exponential gain, log2 discount) for `leg` — the mean-NDCG samples."""
    return per_query_ndcg(corpus, leg, TOPK, gain_exponential, discount_log2)


# =========================================================================== #
# PILLAR 1 — Significance: the paired test, distribution-free tests, multiple comparisons, power.
# =========================================================================== #

def paired_diffs(samples_a: np.ndarray, samples_b: np.ndarray) -> np.ndarray:
    """d_q = a_q - b_q over the SAME queries — the one primitive every pillar differences. The paired
    design: subtracting query-by-query cancels the shared per-query difficulty both legs see."""
    a, b = np.asarray(samples_a, dtype=float), np.asarray(samples_b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    return a - b


def paired_t_test(d: np.ndarray) -> dict:
    """The paired t-test on the difference vector: t = mean(d) / (std(d)/sqrt(n)), df = n-1, two-sided
    p, Cohen's d_z = mean(d)/std(d), and the t-CI on the mean difference. GUARD: n<=1 or std=0 -> t=0,
    p=1. Equals scipy.stats.ttest_rel(a, b) by construction (the reused-routine twin)."""
    d = np.asarray(d, dtype=float)
    n = d.size
    mean = float(np.mean(d)) if n else 0.0
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    if se <= 0.0 or n <= 1:
        return {"n": int(n), "mean": mean, "std": sd, "se": se, "t": 0.0, "df": max(n - 1, 0),
                "p": 1.0, "cohen_dz": 0.0, "ci_lo": mean, "ci_hi": mean}
    t = mean / se
    df = n - 1
    p = float(2.0 * st.t.sf(abs(t), df))
    tcrit = float(st.t.ppf(1 - ALPHA / 2, df))
    return {"n": int(n), "mean": mean, "std": sd, "se": se, "t": float(t), "df": int(df),
            "p": p, "cohen_dz": mean / sd, "ci_lo": mean - tcrit * se, "ci_hi": mean + tcrit * se}


def permutation_test(d: np.ndarray, n_perm: int = 20000, seed: int = 0, exact_max_n: int = 18) -> dict:
    """The Fisher randomization / sign-flip test (Smucker, Allan & Carterette, CIKM 2007 — the
    IR-canonical test). Under H0 the two systems are exchangeable per query, so each d_q is equally
    likely +-|d_q|. Build the null distribution of mean(d) by sign flips; two-sided p = tail area.
    EXACT enumeration of all 2^n sign vectors when n <= exact_max_n, else ONE Monte-Carlo rng stream
    (the (#>=obs + 1)/(B + 1) estimator). Exact under exchangeability — no normality assumption."""
    d = np.asarray(d, dtype=float)
    n = d.size
    obs = float(abs(np.mean(d))) if n else 0.0
    if n == 0:
        return {"p": 1.0, "observed": 0.0, "null_mean": 0.0, "null_std": 0.0, "exact": True}
    if n <= exact_max_n:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))   # (2^n, n)
        null_signed = (signs @ d) / n                                      # signed null (mean is 0)
        p = float(np.mean(np.abs(null_signed) >= obs - 1e-12))
        return {"p": p, "observed": float(np.mean(d)), "null_mean": float(np.mean(null_signed)),
                "null_std": float(np.std(null_signed, ddof=1)), "exact": True}
    rng = np.random.default_rng(seed)
    flips = rng.choice((1.0, -1.0), size=(n_perm, n))
    null = np.abs(flips @ d) / n
    p = float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))
    return {"p": p, "observed": float(np.mean(d)), "null_mean": 0.0,
            "null_std": float(np.std(flips @ d / n, ddof=1)), "exact": False}


def paired_bootstrap_test(d: np.ndarray, b: int = 20000, seed: int = 0) -> dict:
    """The paired bootstrap test: resample the per-query differences with replacement, center by the
    observed mean to impose H0, and read the two-sided tail. Assumption-light cross-check of the paired
    t and permutation p (the d_q are bounded and skewed). Reuses the prereq's bootstrap resampling idiom."""
    d = np.asarray(d, dtype=float)
    n = d.size
    if n == 0:
        return {"p": 1.0, "se": 0.0}
    obs = float(abs(np.mean(d)))
    rng = np.random.default_rng(seed)
    boots = np.mean(rng.choice(d, size=(b, n), replace=True), axis=1)      # vectorized resample
    centered = np.abs(boots - np.mean(d))
    p = float((np.sum(centered >= obs - 1e-12) + 1) / (b + 1))
    return {"p": p, "se": float(np.std(boots, ddof=1))}


def _metric_samples(corpus: dict, leg: str, metric: str) -> np.ndarray:
    return ap_samples(corpus, leg) if metric == "map" else ndcg_samples(corpus, leg)


def pairwise_tests(corpus: dict, metric: str = "map", test: str = "paired_t") -> list[dict]:
    """The C(3,2) = 3 pairwise leg comparisons, each with its raw p (and gap, Cohen's d_z). metric in
    {'map','ndcg'}; test in {'paired_t','permutation','bootstrap'}. The family the corrections act on."""
    rows = []
    for a, b in itertools.combinations(LEG_NAMES, 2):
        d = paired_diffs(_metric_samples(corpus, a, metric), _metric_samples(corpus, b, metric))
        tt = paired_t_test(d)
        if test == "permutation":
            p = permutation_test(d)["p"]
        elif test == "bootstrap":
            p = paired_bootstrap_test(d)["p"]
        else:
            p = tt["p"]
        rows.append({"pair": (a, b), "gap": tt["mean"], "cohen_dz": tt["cohen_dz"],
                     "t": tt["t"], "p": float(p)})
    return rows


def holm_bonferroni(pvals: list[float]) -> dict:
    """Bonferroni (p_i * m) and Holm step-down adjusted p-values, with reject-at-alpha flags. Holm is
    uniformly more powerful than Bonferroni while controlling the same family-wise error rate."""
    m = len(pvals)
    bonf = [min(1.0, p * m) for p in pvals]
    order = sorted(range(m), key=lambda i: pvals[i])
    holm = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)            # enforce monotonic step-down
        holm[i] = running
    return {"bonferroni": bonf, "holm": holm,
            "reject_bonf": [v < ALPHA for v in bonf], "reject_holm": [v < ALPHA for v in holm]}


def bh_fdr(pvals: list[float], q: float = ALPHA) -> dict:
    """Benjamini-Hochberg FDR: the largest k with p_(k) <= (k/m) q rejects the k smallest p-values; the
    BH-adjusted p-values are the step-up cumulative minima. Controls the false-discovery rate, not FWER."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, pvals[i] * m / (rank + 1))
        adj[i] = val
        prev = val
    return {"bh": adj, "reject": [v < q for v in adj]}


def power_required_n(d: np.ndarray, alpha: float = ALPHA, power: float = POWER) -> int:
    """The rigorous required query count: n = ceil( ((z_{1-a/2} + z_{1-b}) / d_z)^2 ) for a target
    power 1-b at the OBSERVED paired effect size d_z = mean(d)/std(d). The principled replacement for
    the prereqs' crude CI-separation n. GUARD: d_z = 0 -> infinite (return a large sentinel)."""
    sd = float(np.std(np.asarray(d, dtype=float), ddof=1))
    mean = float(np.mean(np.asarray(d, dtype=float)))
    if sd <= 0.0 or mean == 0.0:
        return 10 ** 9
    dz = abs(mean / sd)
    za, zb = float(st.norm.ppf(1 - alpha / 2)), float(st.norm.ppf(power))
    return int(math.ceil(((za + zb) / dz) ** 2))


def paired_separation_n(d: np.ndarray, n_max: int = 100000) -> int | None:
    """Smallest n at which the PAIRED projected 95% CI on the mean difference EXCLUDES 0 — the standard-
    normal projection 1.96*std/sqrt(n) with the observed std held fixed (recomputable in TS). This is a
    single-realization read (one CI clearing 0 is ~50% power), strictly below the 80%-power required n.
    GUARD: mean=0 -> None. Closed form n* = (1.96/d_z)^2, clamped to >= 2."""
    d = np.asarray(d, dtype=float)
    mean, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    if mean == 0.0 or sd <= 0.0:
        return None
    nstar = int(math.ceil((Z_ALPHA * sd / abs(mean)) ** 2))
    nstar = max(nstar, 2)
    return nstar if nstar <= n_max else None


# =========================================================================== #
# PILLAR 2 — Calibration: reliability diagram, ECE/MCE, Brier decomposition, Platt & isotonic.
# =========================================================================== #

def leg_raw_scores_query(corpus: dict, leg: str, q: int) -> np.ndarray:
    """The RAW per-doc score vector of `leg` on query q, re-derived from the IMPORTED scoring
    primitives (NOT the ranking): dense = dual-encoder cosine P @ z_q; late_interaction = MaxSim over
    centroid-substituted docs; lexical = BM25 (0 for docs sharing no quantized token — the zero mass)."""
    if leg == "dense":
        return np.asarray(dual_encoder_score(corpus["q_vecs"][q], corpus["doc_vecs"]), dtype=float)
    if leg == "late_interaction":
        Q = corpus["queries"][q]
        Dc = corpus["C"][corpus["doc_cids"]]
        return np.asarray(maxsim_matrix(Q[None, ...], Dc)[0], dtype=float)
    if leg == "lexical":
        s = np.zeros(corpus["n_docs"], dtype=float)
        for doc_id_str, score in bm25_rank(corpus["q_text"][q], corpus["bm25_index"]):
            s[int(doc_id_str)] = float(score)
        return s
    raise ValueError(f"unknown leg {leg}")


def leg_scores(corpus: dict, leg: str, qrels_key: str = "qrels_set"):
    """Pool every (query, doc) raw score and its binary relevance label across all queries. Returns
    (scores[N], labels[N], qidx[N]) with N = n_queries * n_docs; qidx tags each pair's query (for the
    per-query normalization and the held-out split). Calibration is necessarily POOLED — 10 docs/query
    is far too sparse for a per-query reliability diagram."""
    qr = corpus[qrels_key]
    scores, labels, qidx = [], [], []
    for q in range(corpus["n_queries"]):
        s = leg_raw_scores_query(corpus, leg, q)
        scores.append(s)
        lab = np.zeros(corpus["n_docs"], dtype=float)
        for d in qr[q]:
            lab[int(d)] = 1.0
        labels.append(lab)
        qidx.append(np.full(corpus["n_docs"], q, dtype=int))
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(qidx)


def to_unit(scores: np.ndarray, qidx: np.ndarray | None = None, mode: str = "minmax_global") -> np.ndarray:
    """Map scores into [0,1] for the reliability-diagram axis. 'minmax_global': one global min-max (a
    monotone squash that does NOT calibrate — the deliberately-uncalibrated baseline). 'minmax_perquery':
    min-max WITHIN each query before pooling (removes cross-query incomparability — the diagnosis of
    WHY raw is miscalibrated). GUARD: a degenerate (constant) range -> 0.5."""
    s = np.asarray(scores, dtype=float)
    if mode == "minmax_global":
        lo, hi = float(np.min(s)), float(np.max(s))
        return np.full_like(s, 0.5) if hi - lo < EPS else (s - lo) / (hi - lo)
    if mode == "minmax_perquery":
        assert qidx is not None
        out = np.empty_like(s)
        for q in np.unique(qidx):
            m = qidx == q
            lo, hi = float(np.min(s[m])), float(np.max(s[m]))
            out[m] = 0.5 if hi - lo < EPS else (s[m] - lo) / (hi - lo)
        return out
    raise ValueError(f"unknown mode {mode}")


def reliability_diagram(scores01: np.ndarray, labels: np.ndarray, n_bins: int = 10,
                        strategy: str = "quantile") -> list[dict]:
    """Per-bin {lo, hi, n, conf, acc}: conf = mean [0,1]-score in bin, acc = empirical relevance rate.
    Equal-frequency (quantile) bins so BM25's zero mass doesn't empty bins; perfect calibration is the
    diagonal conf == acc. The geometric object the ECE Riemann-sums and the viz bakes. GUARD: empty
    bins are dropped."""
    s, y = np.asarray(scores01, dtype=float), np.asarray(labels, dtype=float)
    if strategy == "quantile":
        edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)                       # collapse duplicate edges (mass points)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(s, edges[1:-1], right=False)
    bins = []
    for b in range(len(edges) - 1):
        m = idx == b
        nb = int(np.sum(m))
        if nb == 0:
            continue
        bins.append({"lo": float(edges[b]) if b > 0 else float(np.min(s)),
                     "hi": float(edges[b + 1]) if b < len(edges) - 2 else float(np.max(s)),
                     "n": nb, "conf": float(np.mean(s[m])), "acc": float(np.mean(y[m]))})
    return bins


def expected_calibration_error(scores01: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) |acc_b - conf_b| — the bin-count-weighted L1 area between the reliability
    curve and the diagonal, a Riemann sum over the score partition (the calibration analog of AP-as-area)."""
    bins = reliability_diagram(scores01, labels, n_bins)
    N = sum(b["n"] for b in bins)
    if N == 0:
        return 0.0
    return float(sum(b["n"] / N * abs(b["acc"] - b["conf"]) for b in bins))


def max_calibration_error(scores01: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """MCE = max_b |acc_b - conf_b| — the worst bin (the L-infinity companion to ECE; the dangerous bin
    for an abstention gate)."""
    bins = reliability_diagram(scores01, labels, n_bins)
    return float(max((abs(b["acc"] - b["conf"]) for b in bins), default=0.0))


def brier_score(prob: np.ndarray, labels: np.ndarray) -> float:
    """The Brier score: mean squared error of the probability against the 0/1 label, mean((p - y)^2)."""
    p, y = np.asarray(prob, dtype=float), np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else 0.0


def brier_decomposition(prob: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy's reliability-resolution-uncertainty decomposition over the bins:
    BS_binned = reliability - resolution + uncertainty. Reliability (down under recalibration) is the
    squared sibling of ECE; resolution (must be preserved) is bin-accuracy spread around the base rate;
    uncertainty = p_bar(1 - p_bar) is irreducible. The three-term identity is EXACT for the calibration-
    BINNED Brier (each forecast replaced by its bin's mean confidence); the gap to the raw Brier is the
    within-bin forecast spread quantization loses (we report both). Returns the three terms + brier_binned
    (which the identity reconstructs to <1e-12) + brier (the raw score)."""
    p, y = np.asarray(prob, dtype=float), np.asarray(labels, dtype=float)
    N = p.size
    if N == 0:
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0,
                "brier_binned": 0.0, "brier": 0.0, "identity": 0.0}
    base = float(np.mean(y))
    # assign each point its bin's mean confidence, so the 3-term identity reconstructs EXACTLY.
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1], right=False)
    rel = res = 0.0
    conf_per_point = np.empty(N)
    for b in np.unique(idx):
        m = idx == b
        nb = int(np.sum(m))
        conf_b, acc_b = float(np.mean(p[m])), float(np.mean(y[m]))
        conf_per_point[m] = conf_b
        rel += nb / N * (conf_b - acc_b) ** 2
        res += nb / N * (acc_b - base) ** 2
    unc = base * (1.0 - base)
    brier_binned = float(np.mean((conf_per_point - y) ** 2))
    return {"reliability": float(rel), "resolution": float(res), "uncertainty": float(unc),
            "brier_binned": brier_binned, "brier": brier_score(p, y), "identity": float(rel - res + unc)}


def platt_scale(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling sigmoid(a*s + b) by logistic-regression cross-entropy (scipy.optimize on the
    raw scores). Returns (a, b); a > 0 (the recalibrator is strictly monotone, hence ranking-preserving).
    Uses scipy.special.expit to avoid an overflow warning."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    sd = float(np.std(s)) or 1.0
    sn = (s - float(np.mean(s))) / sd                 # standardize for a well-conditioned fit

    def nll(theta):
        a, b = theta
        z = a * sn + b
        # stable cross-entropy via logaddexp: -[y*z - log(1+e^z)]
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    a_n, b_n = float(res.x[0]), float(res.x[1])
    # un-standardize: a*s + b = a_n*(s-mean)/sd + b_n  ->  a = a_n/sd, b = b_n - a_n*mean/sd
    a = a_n / sd
    b = b_n - a_n * float(np.mean(s)) / sd
    return a, b


def apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    """The calibrated probability sigmoid(a*s + b) (expit for numerical stability)."""
    return expit(a * np.asarray(scores, dtype=float) + b)


def _pava(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the non-decreasing least-squares fit to y in the given order. Returns a
    same-length array of fitted block means (each block constant)."""
    vals, wts, lens = [], [], []
    for yi in y:
        vals.append(float(yi)); wts.append(1.0); lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-15:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            v1, w1, l1 = vals.pop(), wts.pop(), lens.pop()
            nw = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / nw); wts.append(nw); lens.append(l1 + l2)
    out = []
    for v, ln in zip(vals, lens):
        out.extend([v] * ln)
    return np.array(out)


def isotonic_calibrate(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Isotonic (PAVA) recalibration: the monotone non-decreasing step function minimizing squared error
    against the labels in score order. Returns {x, y} step points (x = the score thresholds, y = the
    fitted probabilities), compressed to block boundaries — the curve the viz plots. Nonparametric and
    exactly monotone; can introduce ties (flat blocks), so it is the lower-ECE but not strictly
    order-preserving recalibrator (Platt is the strict one)."""
    s, ylab = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    order = np.argsort(s, kind="stable")
    xs, ys = s[order], ylab[order]
    fit = _pava(ys)
    # compress consecutive equal fitted values to (threshold, value) step points
    x_steps, y_steps = [float(xs[0])], [float(fit[0])]
    for i in range(1, len(xs)):
        if abs(fit[i] - y_steps[-1]) > 1e-12:
            x_steps.append(float(xs[i])); y_steps.append(float(fit[i]))
    return {"x": x_steps, "y": y_steps}


def apply_isotonic(scores: np.ndarray, iso: dict) -> np.ndarray:
    """Evaluate the fitted isotonic step function: each score takes the fitted value of the largest
    threshold <= it (right-continuous), clamped to the end blocks."""
    x = np.asarray(iso["x"], dtype=float)
    y = np.asarray(iso["y"], dtype=float)
    pos = np.searchsorted(x, np.asarray(scores, dtype=float), side="right") - 1
    pos = np.clip(pos, 0, len(y) - 1)
    return y[pos]


def auc_pooled(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pooled Mann-Whitney AUC = P(score(pos) > score(neg)) with ties at 0.5. A strictly monotone score
    transform leaves it EXACTLY unchanged (the orthogonality of calibration and ranking). GUARD: no
    pos or no neg -> 0.5."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = st.rankdata(s)
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ranking_preserved_platt(corpus: dict, leg: str, a: float, b: float) -> bool:
    """Per query, does Platt recalibration leave the doc ordering identical? sigmoid(a*s+b) with a>0 is
    strictly increasing, so argsort(-recal) == argsort(-raw) exactly, every query."""
    for q in range(corpus["n_queries"]):
        s = leg_raw_scores_query(corpus, leg, q)
        r_raw = np.argsort(-s, kind="stable")
        r_cal = np.argsort(-apply_platt(s, a, b), kind="stable")
        if not np.array_equal(r_raw, r_cal):
            return False
    return True


def ece_table(corpus: dict, leg: str, n_bins: int = 10) -> dict:
    """ECE/MCE under each condition for `leg`: raw (global min-max), per-query normalized, Platt,
    isotonic — the headline ordering, RUN not assumed."""
    s, y, qidx = leg_scores(corpus, leg)
    raw01 = to_unit(s, qidx, "minmax_global")
    perq01 = to_unit(s, qidx, "minmax_perquery")
    a, b = platt_scale(s, y)
    platt = apply_platt(s, a, b)
    iso = isotonic_calibrate(s, y)
    iso_p = apply_isotonic(s, iso)
    out = {}
    for name, p in (("raw", raw01), ("perquery", perq01), ("platt", platt), ("isotonic", iso_p)):
        out[name] = {"ece": expected_calibration_error(p, y, n_bins),
                     "mce": max_calibration_error(p, y, n_bins),
                     "brier": brier_score(p, y)}
    out["platt_params"] = {"a": a, "b": b}
    out["auc_raw"] = auc_pooled(s, y)
    out["auc_platt"] = auc_pooled(platt, y)
    return out


def heldout_ece(corpus: dict, leg: str, n_bins: int = 10) -> dict:
    """Fit recalibrators on queries 0..k-1, evaluate ECE on the held-out half — guards the ECE drop
    against an in-sample-only artifact (isotonic can overfit). Returns held-out raw vs platt vs isotonic ECE."""
    s, y, qidx = leg_scores(corpus, leg)
    half = corpus["n_queries"] // 2
    tr, te = qidx < half, qidx >= half
    a, b = platt_scale(s[tr], y[tr])
    iso = isotonic_calibrate(s[tr], y[tr])
    raw_te = to_unit(s, qidx, "minmax_global")[te]
    return {"raw": expected_calibration_error(raw_te, y[te], n_bins),
            "platt": expected_calibration_error(apply_platt(s[te], a, b), y[te], n_bins),
            "isotonic": expected_calibration_error(apply_isotonic(s[te], iso), y[te], n_bins)}


# =========================================================================== #
# PILLAR 3 — Drift: two-sample KS, PSI = symmetrized KL, the synthetic-drift toy, input vs outcome.
# =========================================================================== #

KNOB_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)     # the embedding-degradation noise grid (sigma)
DRIFT_BINS = 5                                       # PSI bins (~8/bin on 40 queries; 10 needs large n)


def perturbed_dense_ndcg(corpus: dict, sigma: float, seed: int = 0) -> np.ndarray:
    """Re-derive the dense leg's per-query NDCG@10 with the pooled query/doc vectors degraded by
    isotropic Gaussian noise of scale sigma (re-normalized to the sphere), scored against the UNCHANGED
    MaxSim oracle grades. The monotone model-decay knob: sigma=0 reproduces the published dense NDCG."""
    rng = np.random.default_rng(seed)
    qv, dv = corpus["q_vecs"], corpus["doc_vecs"]
    qp = normalize(qv + sigma * rng.standard_normal(qv.shape))
    dp = normalize(dv + sigma * rng.standard_normal(dv.shape))
    out = np.empty(corpus["n_queries"])
    for q in range(corpus["n_queries"]):
        s = dual_encoder_score(qp[q], dp)
        ranking = np.argsort(-s, kind="stable").tolist()
        out[q] = ndcg_at_k(ranking, corpus["grades"][q], TOPK, gain_exponential, discount_log2)
    return out


def empirical_cdf(sample: np.ndarray):
    """The right-continuous empirical CDF as (sorted support, cumulative proportions) — the staircase
    the KS test sup-distances and the viz draws."""
    s = np.sort(np.asarray(sample, dtype=float))
    n = s.size
    return s, (np.arange(1, n + 1) / n if n else np.array([]))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sample Kolmogorov-Smirnov: D = sup_x |F_a(x) - F_b(x)|, the largest vertical gap between the
    two empirical-CDF staircases (attained at a pooled sample point), plus the asymptotic p-value and
    the argmax location. The hand sup is cross-checked against scipy.stats.ks_2samp (the twin)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    grid = np.sort(np.concatenate([a, b]))
    Fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    Fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    gaps = np.abs(Fa - Fb)
    j = int(np.argmax(gaps))
    sci = st.ks_2samp(a, b)
    return {"stat": float(gaps[j]), "pvalue": float(sci.pvalue), "at": float(grid[j]),
            "scipy_stat": float(sci.statistic)}


def histogram_props(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10, alpha: float = 0.5):
    """Bin both samples on the REFERENCE quantile edges (the PSI convention), clamping out-of-range
    current points into the end bins. Returns (p, q, edges) proportion vectors that sum to 1, with
    additive (Laplace) smoothing alpha per bin — the standard credit-risk PSI practice that keeps an
    empty bin from sending the log to infinity at small samples (the alternative, a tiny epsilon floor,
    over-reacts to one missing observation on only forty queries). alpha=0 recovers raw proportions."""
    ref, cur = np.asarray(ref, dtype=float), np.asarray(cur, dtype=float)
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:                                   # degenerate (constant reference)
        edges = np.array([ref.min() - 1e-9, ref.max() + 1e-9])
    nb = edges.size - 1
    inner = edges[1:-1]
    pi = np.bincount(np.digitize(ref, inner, right=False), minlength=nb)[:nb].astype(float)
    qi = np.bincount(np.digitize(cur, inner, right=False), minlength=nb)[:nb].astype(float)
    p = (pi + alpha) / (ref.size + alpha * nb)
    q = (qi + alpha) / (cur.size + alpha * nb)
    return p, q, edges


def population_stability_index(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """PSI = sum_b (p_b - q_b) ln(p_b / q_b) over the reference-quantile bins — the credit-risk drift
    standard (traffic light: <0.1 stable, 0.1-0.25 moderate, >0.25 significant). Every bin is floored by
    EPS before the log and ratio (empty bins would give ln(0)/division-by-zero)."""
    p, q, _ = histogram_props(ref, cur, n_bins)
    p = np.maximum(p, EPS); q = np.maximum(q, EPS)
    return float(np.sum((p - q) * np.log(p / q)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) = sum_b p_b ln(p_b/q_b) over a shared binning; p, q floored by EPS. Used to show that
    PSI is the SYMMETRIZED KL (Jeffreys divergence): PSI = KL(p||q) + KL(q||p)."""
    p = np.maximum(np.asarray(p, dtype=float), EPS)
    q = np.maximum(np.asarray(q, dtype=float), EPS)
    return float(np.sum(p * np.log(p / q)))


def drift_windows(corpus: dict, knob: float, seed: int = 0, mode: str = "degrade"):
    """Construct a reference and a current window of per-query NDCG@10. mode='degrade': reference =
    unperturbed dense leg, current = the dense leg degraded by Gaussian noise sigma=knob (model decay,
    oracle grades FIXED). mode='null': both windows are the unperturbed leg split by a random query
    partition (same distribution, finite samples — the genuine null)."""
    ref = ndcg_samples(corpus, "dense")
    if mode == "degrade":
        return ref, perturbed_dense_ndcg(corpus, knob, seed)
    if mode == "null":
        # matched-n genuine null: resample the SAME distribution with replacement (H0 holds, finite-
        # sample noise present) — a fair n=Q-vs-Q comparison to the degrade windows.
        rng = np.random.default_rng(seed)
        return ref, rng.choice(ref, size=ref.size, replace=True)
    raise ValueError(f"unknown mode {mode}")


def drift_summary(corpus: dict, knob_levels=KNOB_LEVELS, seed: int = 0, n_bins: int = DRIFT_BINS) -> list[dict]:
    """Per knob level (degrade mode): PSI, KS statistic + p-value, mean NDCG, the mean shift in SE
    units, and the traffic-light band. The null-vs-drift table the viz bakes and the headline reads."""
    ref = ndcg_samples(corpus, "dense")
    se = float(np.std(ref, ddof=1)) / math.sqrt(ref.size)
    rows = []
    for sigma in knob_levels:
        cur = perturbed_dense_ndcg(corpus, sigma, seed)
        psi = population_stability_index(ref, cur, n_bins)
        ks = ks_two_sample(ref, cur)
        dmean = float(np.mean(cur) - np.mean(ref))
        light = "red" if psi > 0.25 else ("amber" if psi > 0.1 else "green")
        rows.append({"sigma": float(sigma), "psi": psi, "ks_stat": ks["stat"], "ks_p": ks["pvalue"],
                     "mean_ndcg": float(np.mean(cur)), "mean_shift_se": dmean / se if se > 0 else 0.0,
                     "light": light})
    return rows


def psi_monotone_curve(corpus: dict, knob_levels=KNOB_LEVELS, seeds=(0, 1, 2, 3, 4), n_bins: int = DRIFT_BINS):
    """Seed-averaged PSI and KS statistic across the knob grid — kills the per-draw jitter so the
    monotonicity assert is robust (the headline numbers bake a single seed)."""
    ref = ndcg_samples(corpus, "dense")
    psi, ks = [], []
    for sigma in knob_levels:
        ps = [population_stability_index(ref, perturbed_dense_ndcg(corpus, sigma, s), n_bins) for s in seeds]
        ds = [ks_two_sample(ref, perturbed_dense_ndcg(corpus, sigma, s))["stat"] for s in seeds]
        psi.append(float(np.mean(ps))); ks.append(float(np.mean(ds)))
    return {"sigma": list(knob_levels), "psi": psi, "ks": ks}


SILENT_SIGMAS = (0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2)   # the small-degradation scan for the silent case


def silent_decay(corpus: dict, seed: int = 0) -> dict:
    """The 'drift monitoring must be PAIRED' construction — the significance pillar's paired-vs-unpaired
    power gap, now applied over time. A SMALL uniform degradation drops per-query NDCG slightly but
    consistently on the SAME eval queries. The aggregate monitor (two independent CIs on mean-then vs
    mean-now) cannot distinguish them — the intervals OVERLAP, a dashboard stays green — yet the PAIRED
    test on the per-query (now - then) differences rejects, because pairing cancels the shared per-query
    difficulty. We scan a few small sigmas and return the smallest where the contrast holds (deterministic);
    KS on the marginal distribution is reported too (it, like the unpaired mean, misses a small uniform
    shift, which is why the paired test is the right monitor here)."""
    ref = ndcg_samples(corpus, "dense")
    sa = metric_summary(ref)
    chosen = None
    for sigma in SILENT_SIGMAS:
        cur = perturbed_dense_ndcg(corpus, sigma, seed)
        sb = metric_summary(cur)
        overlap = sa["ci_lo"] <= sb["ci_hi"] and sb["ci_lo"] <= sa["ci_hi"]
        pt = paired_t_test(paired_diffs(cur, ref))
        if overlap and pt["p"] < ALPHA:
            chosen = (sigma, cur, sb, overlap, pt)
            break
    if chosen is None:                                   # fallback: the largest scanned sigma
        sigma = SILENT_SIGMAS[-1]
        cur = perturbed_dense_ndcg(corpus, sigma, seed)
        sb = metric_summary(cur)
        chosen = (sigma, cur, sb,
                  sa["ci_lo"] <= sb["ci_hi"] and sb["ci_lo"] <= sa["ci_hi"],
                  paired_t_test(paired_diffs(cur, ref)))
    sigma, cur, sb, overlap, pt = chosen
    ks = ks_two_sample(ref, cur)
    return {"ref": ref, "cur": cur, "sigma": float(sigma), "unpaired_overlap": bool(overlap),
            "paired_p": pt["p"], "ks_p": ks["pvalue"], "ks_stat": ks["stat"],
            "mean_shift": sb["mean"] - sa["mean"], "mean_shift_se": (sb["mean"] - sa["mean"]) / sa["se"],
            "se": sa["se"]}


def _top1_dense_score(corpus: dict, q: int) -> float:
    """The top-1 dense retrieval score for query q — an INPUT signal observable WITHOUT relevance labels
    (what a production monitor actually sees)."""
    return float(np.max(dual_encoder_score(corpus["q_vecs"][q], corpus["doc_vecs"])))


def input_vs_outcome_drift(corpus: dict, seed: int = 0, n_bins: int = DRIFT_BINS) -> dict:
    """Data-drift vs model-decay. COVARIATE SHIFT: re-weight the query MIX (reference = easy half by
    NDCG, current = hard half) with the model UNCHANGED — the INPUT distribution (top-1 dense score,
    label-free) shifts (input PSI fires) but the conditional quality is unchanged, so a paired NDCG
    comparison on a FIXED query set shows no decay. MODEL DECAY (degradation): the paired NDCG on the
    fixed set drops. The honest lesson: input drift alone cannot diagnose decay; you need a labelled
    paired outcome test (the significance pillar)."""
    nd = ndcg_samples(corpus, "dense")
    order = np.argsort(nd)
    half = corpus["n_queries"] // 2
    hard, easy = order[:half], order[half:]
    inp_ref = np.array([_top1_dense_score(corpus, q) for q in easy])
    inp_cur = np.array([_top1_dense_score(corpus, q) for q in hard])
    # covariate shift: input distribution differs, but the model is identical -> no paired decay on a fixed set.
    fixed = np.arange(corpus["n_queries"])
    paired_covariate = paired_diffs(nd[fixed], nd[fixed])               # identically zero (same model)
    # model decay on the SAME fixed set: degraded current vs reference.
    deg = perturbed_dense_ndcg(corpus, 0.5, seed)
    paired_decay = paired_diffs(deg[fixed], nd[fixed])
    return {
        "input_psi_covariate": population_stability_index(inp_ref, inp_cur, n_bins),
        "outcome_paired_covariate_mean": float(np.mean(paired_covariate)),
        "outcome_paired_decay_mean": float(np.mean(paired_decay)),
        "decay_t_p": paired_t_test(paired_decay)["p"],
    }


def jeffreys_from_props(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) + KL(q||p) on already-binned proportions (the same EPS-guarded p, q PSI uses) — the
    symmetrized-KL identity, exact to floating point against PSI."""
    return kl_divergence(p, q) + kl_divergence(q, p)


# =========================================================================== #
# viz_constants — every number SignificanceTestingLaboratory.tsx mirrors (cast numpy scalars).
# =========================================================================== #

def viz_constants() -> None:
    corpus = get_corpus()
    nq = corpus["n_queries"]
    a_leg, b_leg = WORKED_PAIR

    print("\n=== shared constants ===")
    print(f"N_QUERIES = {nq}  WORKED_PAIR = {WORKED_PAIR}  ALPHA = {ALPHA}  Z_ALPHA = {round(Z_ALPHA, 4)}")

    print("\n=== Panel A — the paired test (worked pair), both metrics ===")
    for metric in ("map", "ndcg"):
        sa, sb = _metric_samples(corpus, a_leg, metric), _metric_samples(corpus, b_leg, metric)
        d = paired_diffs(sa, sb)
        tt = paired_t_test(d)
        se_unp = math.sqrt(np.var(sa, ddof=1) / nq + np.var(sb, ddof=1) / nq)
        if metric == "map":
            unp_n = projected_separation_n(corpus, a_leg, b_leg, "qrels_set", n_max=100000)
        else:
            unp_n = projected_ndcg_separation_n(corpus, a_leg, b_leg, n_max=100000)
        print(f"  [{metric}] mean_d={round(tt['mean'], 4)} std_d={round(tt['std'], 4)} "
              f"se_paired={round(tt['se'], 4)} se_unpaired={round(se_unp, 4)} "
              f"t={round(tt['t'], 3)} p={tt['p']:.3e} d_z={round(tt['cohen_dz'], 4)}")
        print(f"       PAIRED_SEP_N={paired_separation_n(d)} UNPAIRED_OVERLAP_N={unp_n} "
              f"POWER_N={power_required_n(d)}")
        print(f"       PER_Q_DIFF={[round(float(v), 3) for v in d]}")
        print(f"       corr(A,B)={round(float(np.corrcoef(sa, sb)[0, 1]), 3)} "
              f"var_ratio={round(float(np.var(d, ddof=1) / (np.var(sa, ddof=1) + np.var(sb, ddof=1))), 3)}")

    print("\n=== Panel B — distribution-free p (worked pair, MAP) + 3-way correction grid ===")
    d_map = paired_diffs(ap_samples(corpus, a_leg), ap_samples(corpus, b_leg))
    perm = permutation_test(d_map)
    boot = paired_bootstrap_test(d_map)
    print(f"  worked MAP: t_p={paired_t_test(d_map)['p']:.3e} perm_p={perm['p']:.3e} boot_p={boot['p']:.3e}")
    print(f"  PERM_NULL_STD={round(perm['null_std'], 4)} OBSERVED={round(perm['observed'], 4)}")
    for metric in ("map", "ndcg"):
        rows = pairwise_tests(corpus, metric, "paired_t")
        pv = [r["p"] for r in rows]
        hb = holm_bonferroni(pv)
        bh = bh_fdr(pv)
        print(f"  [{metric}] pairs={[r['pair'] for r in rows]}")
        print(f"       raw_p={[f'{p:.3e}' for p in pv]}")
        print(f"       bonf={[round(v, 4) for v in hb['bonferroni']]} holm={[round(v, 4) for v in hb['holm']]} "
              f"bh={[round(v, 4) for v in bh['bh']]}")
        print(f"       reject_raw={[p < ALPHA for p in pv]} reject_bonf={hb['reject_bonf']} "
              f"reject_bh={bh['reject']}")

    print("\n=== Panel C — calibration (reliability, ECE/MCE, Platt/isotonic, ranking invariance) ===")
    s0, y0, _ = leg_scores(corpus, "dense")
    print(f"  N_PAIRS={s0.size}  N_POS={int(np.sum(y0))}  BASE_RATE={round(float(np.mean(y0)), 4)}")
    for leg in LEG_NAMES:
        tbl = ece_table(corpus, leg)
        print(f"  {leg:16s} ECE raw={round(tbl['raw']['ece'], 4)} perq={round(tbl['perquery']['ece'], 4)} "
              f"platt={round(tbl['platt']['ece'], 4)} iso={round(tbl['isotonic']['ece'], 4)}")
        print(f"      MCE raw={round(tbl['raw']['mce'], 4)} platt={round(tbl['platt']['mce'], 4)}  "
              f"platt(a,b)=({round(tbl['platt_params']['a'], 4)},{round(tbl['platt_params']['b'], 4)})  "
              f"AUC raw={round(tbl['auc_raw'], 6)} platt={round(tbl['auc_platt'], 6)} "
              f"preserved={ranking_preserved_platt(corpus, leg, tbl['platt_params']['a'], tbl['platt_params']['b'])}")
        ho = heldout_ece(corpus, leg)
        print(f"      HELD-OUT ECE raw={round(ho['raw'], 4)} platt={round(ho['platt'], 4)} iso={round(ho['isotonic'], 4)}")
    bd = brier_decomposition(apply_platt(s0, *platt_scale(s0, y0)), y0)
    print(f"  dense Brier decomp: rel={round(bd['reliability'], 5)} res={round(bd['resolution'], 5)} "
          f"unc={round(bd['uncertainty'], 5)} brier_binned={round(bd['brier_binned'], 5)} "
          f"brier={round(bd['brier'], 5)} identity_gap={abs(bd['brier_binned'] - bd['identity']):.2e}")
    print("  RELIABILITY_BINS [conf, acc, n] per leg x {raw, platt, isotonic} (TS recomputes ECE):")
    for leg in LEG_NAMES:
        s, y, qidx = leg_scores(corpus, leg)
        a, b = platt_scale(s, y)
        conds = {"raw": to_unit(s, qidx, "minmax_global"),
                 "platt": apply_platt(s, a, b),
                 "isotonic": apply_isotonic(s, isotonic_calibrate(s, y))}
        for cname, p in conds.items():
            compact = [[round(bb["conf"], 4), round(bb["acc"], 4), bb["n"]]
                       for bb in reliability_diagram(p, y, 10)]
            print(f"    {leg}/{cname}: {compact}")

    print("\n=== Panel D — drift detection (PSI/KS over the degradation knob; silent decay; input vs outcome) ===")
    for r in drift_summary(corpus):
        print(f"  sigma={r['sigma']:.2f}  PSI={round(r['psi'], 4)} ({r['light']})  KS_D={round(r['ks_stat'], 4)} "
              f"KS_p={r['ks_p']:.3e}  meanNDCG={round(r['mean_ndcg'], 4)}  shift={round(r['mean_shift_se'], 2)}SE")
    null_psis = [population_stability_index(*drift_windows(corpus, 0.0, seed=s, mode="null"), DRIFT_BINS)
                 for s in range(20)]
    null_kps = [ks_two_sample(*drift_windows(corpus, 0.0, seed=s, mode="null"))["pvalue"] for s in range(20)]
    print(f"  NULL (matched-n resample, mean over 20 seeds): PSI={round(float(np.mean(null_psis)), 4)} "
          f"KS_p={round(float(np.mean(null_kps)), 4)}")
    sd = silent_decay(corpus)
    print(f"  SILENT DECAY (sigma={sd['sigma']}): mean_shift={round(sd['mean_shift'], 4)} "
          f"({round(sd['mean_shift_se'], 2)}SE)  unpaired_overlap={sd['unpaired_overlap']}  "
          f"paired_p={sd['paired_p']:.3e}  KS_p={sd['ks_p']:.3e} (KS misses the small uniform shift)")
    iv = input_vs_outcome_drift(corpus)
    print(f"  INPUT vs OUTCOME: input_PSI(covariate)={round(iv['input_psi_covariate'], 4)} "
          f"outcome_paired(covariate)={round(iv['outcome_paired_covariate_mean'], 4)} "
          f"outcome_paired(decay)={round(iv['outcome_paired_decay_mean'], 4)} decay_p={iv['decay_t_p']:.3e}")
    # the symmetrized-KL identity
    p, q, _ = histogram_props(*drift_windows(corpus, 0.35, mode="degrade"), DRIFT_BINS)
    print(f"  PSI==Jeffreys check: PSI={round(float(np.sum((np.maximum(p,EPS)-np.maximum(q,EPS))*np.log(np.maximum(p,EPS)/np.maximum(q,EPS)))), 6)} "
          f"Jeffreys={round(jeffreys_from_props(p, q), 6)}")
    print("  REF_NDCG (dense, the 'then' window):")
    print(f"    {[round(float(v), 3) for v in ndcg_samples(corpus, 'dense')]}")
    print("  CUR_NDCG (dense degraded at each sigma, the 'now' window — TS draws hist/ECDF, computes KS gap):")
    for sigma in KNOB_LEVELS[1:]:
        print(f"    sigma={sigma}: {[round(float(v), 3) for v in perturbed_dense_ndcg(corpus, sigma, 0)]}")


# =========================================================================== #
# Verification harness — every pedagogical claim is an assert.
# =========================================================================== #

def test_pairing_reduces_variance() -> None:
    # var(d) < var(A) + var(B) for every leg pair, both metrics (positive cross-query correlation).
    c = get_corpus()
    for metric in ("map", "ndcg"):
        for a, b in itertools.combinations(LEG_NAMES, 2):
            sa, sb = _metric_samples(c, a, metric), _metric_samples(c, b, metric)
            d = paired_diffs(sa, sb)
            ratio = float(np.var(d, ddof=1) / (np.var(sa, ddof=1) + np.var(sb, ddof=1)))
            assert ratio < 1.0, (metric, a, b, ratio)
            assert float(np.corrcoef(sa, sb)[0, 1]) > 0.0, (metric, a, b)


def test_paired_se_below_unpaired() -> None:
    c = get_corpus()
    nq = c["n_queries"]
    for metric in ("map", "ndcg"):
        for a, b in itertools.combinations(LEG_NAMES, 2):
            sa, sb = _metric_samples(c, a, metric), _metric_samples(c, b, metric)
            se_paired = paired_t_test(paired_diffs(sa, sb))["se"]
            se_unp = math.sqrt(np.var(sa, ddof=1) / nq + np.var(sb, ddof=1) / nq)
            assert se_paired < se_unp, (metric, a, b, se_paired, se_unp)


def test_paired_separates_sooner_than_overlap() -> None:
    # the headline: the paired CI excludes 0 at FAR fewer queries than the prereqs' overlapping-CI read.
    c = get_corpus()
    a, b = WORKED_PAIR
    d_map = paired_diffs(ap_samples(c, a), ap_samples(c, b))
    unp_map = projected_separation_n(c, a, b, "qrels_set", n_max=100000)
    assert paired_separation_n(d_map) < unp_map, (paired_separation_n(d_map), unp_map)
    d_nd = paired_diffs(ndcg_samples(c, a), ndcg_samples(c, b))
    unp_nd = projected_ndcg_separation_n(c, a, b, n_max=100000)
    assert paired_separation_n(d_nd) < unp_nd, (paired_separation_n(d_nd), unp_nd)


def test_ttest_rel_twin() -> None:
    # the reused-routine twin: our paired_t_test == scipy.stats.ttest_rel, <1e-9.
    c = get_corpus()
    for metric in ("map", "ndcg"):
        for a, b in itertools.combinations(LEG_NAMES, 2):
            sa, sb = _metric_samples(c, a, metric), _metric_samples(c, b, metric)
            mine = paired_t_test(paired_diffs(sa, sb))
            ref = st.ttest_rel(sa, sb)
            assert abs(mine["t"] - float(ref.statistic)) < 1e-9, (metric, a, b)
            assert abs(mine["p"] - float(ref.pvalue)) < 1e-9, (metric, a, b)


def test_permutation_approximates_t() -> None:
    # permutation p ~ t-test p across the family; and exact enumeration == Monte-Carlo on a small slice.
    c = get_corpus()
    for a, b in itertools.combinations(LEG_NAMES, 2):
        d = paired_diffs(ap_samples(c, a), ap_samples(c, b))
        tp = paired_t_test(d)["p"]
        pp = permutation_test(d)["p"]
        assert abs(pp - tp) < 0.03 or (0.4 <= (pp + 1e-12) / (tp + 1e-12) <= 2.5), (a, b, pp, tp)
    small = paired_diffs(ap_samples(c, "lexical"), ap_samples(c, "dense"))[:12]
    p_exact = permutation_test(small, exact_max_n=18)["p"]
    p_mc = permutation_test(small, n_perm=40000, seed=3, exact_max_n=0)["p"]
    assert abs(p_exact - p_mc) < 0.03, (p_exact, p_mc)


def test_bootstrap_approximates_t() -> None:
    c = get_corpus()
    for a, b in itertools.combinations(LEG_NAMES, 2):
        d = paired_diffs(ndcg_samples(c, a), ndcg_samples(c, b))
        tp = paired_t_test(d)["p"]
        bp = paired_bootstrap_test(d, seed=4)["p"]
        assert abs(bp - tp) < 0.03 or (0.4 <= (bp + 1e-12) / (tp + 1e-12) <= 2.5), (a, b, bp, tp)


def test_multiple_comparison_changes_verdict() -> None:
    # on the NDCG family the closest pair (lexical/dense) is NOT significant raw AND under every
    # correction, while the other two ARE under every correction — the correction's effect is real.
    c = get_corpus()
    rows = pairwise_tests(c, "ndcg", "paired_t")
    pv = [r["p"] for r in rows]
    hb, bh = holm_bonferroni(pv), bh_fdr(pv)
    close_i = next(i for i, r in enumerate(rows) if set(r["pair"]) == set(WORKED_PAIR))
    assert pv[close_i] > ALPHA, (rows[close_i], pv[close_i])
    assert not hb["reject_bonf"][close_i] and not hb["reject_holm"][close_i] and not bh["reject"][close_i]
    for i in range(len(rows)):
        if i != close_i:
            assert pv[i] < ALPHA and hb["reject_bonf"][i] and bh["reject"][i], rows[i]


def test_map_pair_significant_ndcg_pair_not() -> None:
    # the honest twist: pairing tightens WITHOUT manufacturing significance — MAP lexical/dense IS
    # significant at n=40 (p<0.05); the same NDCG pair is NOT yet (p>0.05).
    c = get_corpus()
    a, b = WORKED_PAIR
    p_map = paired_t_test(paired_diffs(ap_samples(c, a), ap_samples(c, b)))["p"]
    p_nd = paired_t_test(paired_diffs(ndcg_samples(c, a), ndcg_samples(c, b)))["p"]
    assert p_map < ALPHA, p_map
    assert p_nd > ALPHA, p_nd


def test_power_n_exceeds_single_realization_n() -> None:
    # 80%-power required n > the one-time CI-excludes-0 n (a single realized CI clearing 0 is ~50% power).
    c = get_corpus()
    a, b = WORKED_PAIR
    d = paired_diffs(ndcg_samples(c, a), ndcg_samples(c, b))
    assert power_required_n(d) > paired_separation_n(d), (power_required_n(d), paired_separation_n(d))


def test_duality_ci_excludes_zero_iff_reject() -> None:
    # test-CI duality: the paired CI on the mean difference excludes 0 IFF the paired test rejects at alpha.
    c = get_corpus()
    for metric in ("map", "ndcg"):
        for a, b in itertools.combinations(LEG_NAMES, 2):
            tt = paired_t_test(paired_diffs(_metric_samples(c, a, metric), _metric_samples(c, b, metric)))
            excl = (tt["ci_lo"] > 0) or (tt["ci_hi"] < 0)
            assert excl == (tt["p"] < ALPHA), (metric, a, b, tt["ci_lo"], tt["ci_hi"], tt["p"])


def test_correction_adjusts_pvalues_up() -> None:
    pv = [0.001, 0.02, 0.2]
    hb, bh = holm_bonferroni(pv), bh_fdr(pv)
    assert all(hb["bonferroni"][i] >= pv[i] - 1e-12 for i in range(3))
    assert all(hb["holm"][i] >= pv[i] - 1e-12 for i in range(3))
    assert all(bh["bh"][i] >= pv[i] - 1e-12 for i in range(3))
    assert hb["holm"][0] <= hb["bonferroni"][0] + 1e-12          # Holm no more conservative than Bonferroni


def test_calibration_raw_miscalibrated() -> None:
    # RUN the contrast: the smooth cosine/MaxSim legs are strongly OVERCONFIDENT (large ECE), and every
    # leg is miscalibrated somewhere (a meaningful worst bin). Lexical BM25 is well-calibrated in the
    # bulk (its zero mass sits at conf~0/acc~low) but bad in the tail — so the robust per-leg claim is
    # MCE, the aggregate claim is ECE on the dense/late legs.
    c = get_corpus()
    eces = {leg: ece_table(c, leg)["raw"]["ece"] for leg in LEG_NAMES}
    for leg in LEG_NAMES:
        tbl = ece_table(c, leg)
        assert tbl["raw"]["mce"] > 0.05, (leg, tbl["raw"]["mce"])
        assert 0.0 <= tbl["raw"]["ece"] <= tbl["raw"]["mce"] + 1e-12 <= 1.0 + 1e-12, leg
    assert eces["dense"] > 0.1 and eces["late_interaction"] > 0.1, eces      # cosine/MaxSim overconfident
    assert float(np.mean(list(eces.values()))) > 0.1, eces


def test_recalibration_lowers_ece() -> None:
    # Platt AND isotonic cut ECE below raw, every leg (recalibration works).
    c = get_corpus()
    for leg in LEG_NAMES:
        tbl = ece_table(c, leg)
        assert tbl["platt"]["ece"] < tbl["raw"]["ece"], (leg, tbl["platt"]["ece"], tbl["raw"]["ece"])
        assert tbl["isotonic"]["ece"] < tbl["raw"]["ece"], (leg, tbl["isotonic"]["ece"], tbl["raw"]["ece"])


def test_platt_preserves_ranking_exactly() -> None:
    # the orthogonality backbone: a strictly monotone recalibration leaves the ranking, AUC, and NDCG
    # EXACTLY unchanged (<1e-12) — calibration is orthogonal to the ranking metrics the prereqs measured.
    c = get_corpus()
    for leg in LEG_NAMES:
        s, y, _ = leg_scores(c, leg)
        a, b = platt_scale(s, y)
        assert a > 0.0, (leg, a)                                    # strictly increasing
        assert ranking_preserved_platt(c, leg, a, b), leg
        assert abs(auc_pooled(s, y) - auc_pooled(apply_platt(s, a, b), y)) < 1e-12, leg
        # per-query NDCG identical (the ranking the prereq metric sees is unchanged)
        for q in range(0, c["n_queries"], 7):
            raw_rank = np.argsort(-leg_raw_scores_query(c, leg, q), kind="stable").tolist()
            cal_rank = np.argsort(-apply_platt(leg_raw_scores_query(c, leg, q), a, b), kind="stable").tolist()
            nd_raw = ndcg_at_k(raw_rank, c["grades"][q], TOPK, gain_exponential, discount_log2)
            nd_cal = ndcg_at_k(cal_rank, c["grades"][q], TOPK, gain_exponential, discount_log2)
            assert abs(nd_raw - nd_cal) < 1e-12, (leg, q)


def test_brier_decomposition_identity() -> None:
    c = get_corpus()
    for leg in LEG_NAMES:
        s, y, _ = leg_scores(c, leg)
        p = apply_platt(s, *platt_scale(s, y))
        bd = brier_decomposition(p, y)
        # the three-term identity reconstructs the calibration-binned Brier EXACTLY.
        assert abs(bd["brier_binned"] - bd["identity"]) < 1e-9, (leg, bd)
        assert bd["reliability"] >= 0 and bd["resolution"] >= 0 and bd["uncertainty"] >= 0, (leg, bd)


def test_heldout_ece_improves() -> None:
    # the ECE drop is not an in-sample-only artifact: held-out recalibrated ECE beats held-out raw.
    c = get_corpus()
    for leg in LEG_NAMES:
        ho = heldout_ece(c, leg)
        assert ho["platt"] < ho["raw"], (leg, ho)


def test_drift_null_silent() -> None:
    # under the matched-n resampled null (same distribution) the detector does NOT fire, in expectation:
    # mean PSI sits in the green band and the KS test keeps its nominal (non-rejecting) behaviour. A
    # single small-sample draw is noisy (PSI's 0.1/0.25 thresholds assume large n — the rigorFlag), so we
    # average over seeds, the honest statement of "silent under H0".
    c = get_corpus()
    psis, kps = [], []
    for s in range(20):
        ref, cur = drift_windows(c, 0.0, seed=s, mode="null")
        psis.append(population_stability_index(ref, cur, DRIFT_BINS))
        kps.append(ks_two_sample(ref, cur)["pvalue"])
    assert float(np.mean(psis)) < 0.1, ("mean null PSI should be small", float(np.mean(psis)))
    assert float(np.mean(kps)) > 0.30, ("mean null KS p should be far from 0", float(np.mean(kps)))


def test_psi_ks_monotone_in_knob() -> None:
    # seed-averaged PSI and KS rise with the degradation knob, and PSI crosses the 0.25 line.
    c = get_corpus()
    curve = psi_monotone_curve(c)
    psi, ks = curve["psi"], curve["ks"]
    assert all(psi[i + 1] >= psi[i] - 1e-9 for i in range(len(psi) - 1)), psi
    assert all(ks[i + 1] >= ks[i] - 1e-9 for i in range(len(ks) - 1)), ks
    assert psi[0] < 0.1 < 0.25 < psi[-1], psi                       # spans green -> red


def test_silent_decay_paired_beats_unpaired() -> None:
    # the headline contrast: a small uniform drift the AGGREGATE two-CI comparison cannot distinguish
    # (overlapping intervals) IS caught by the PAIRED drift test on the same eval queries — drift
    # monitoring, like system comparison, must be paired.
    c = get_corpus()
    sd = silent_decay(c)
    assert sd["unpaired_overlap"], ("aggregate CIs should overlap", sd["sigma"])
    assert sd["paired_p"] < ALPHA, ("paired test should reject", sd["paired_p"])
    # the degradation is genuine (mean drops) but small relative to the aggregate sampling error.
    assert sd["mean_shift"] < 0.0, sd["mean_shift"]


def test_input_vs_outcome_drift() -> None:
    # covariate shift moves the INPUT distribution (input PSI fires) with NO paired outcome decay, while
    # degradation drops the paired outcome — input drift alone cannot diagnose decay.
    c = get_corpus()
    iv = input_vs_outcome_drift(c)
    assert iv["input_psi_covariate"] > 0.1, iv
    assert abs(iv["outcome_paired_covariate_mean"]) < 1e-12, iv          # same model -> exactly zero
    assert iv["outcome_paired_decay_mean"] < 0.0 and iv["decay_t_p"] < 0.05, iv


def test_psi_is_symmetrized_kl() -> None:
    # PSI == KL(p||q) + KL(q||p) (Jeffreys) on the shared binning, <1e-12.
    c = get_corpus()
    for sigma in (0.1, 0.35, 0.5):
        p, q, _ = histogram_props(*drift_windows(c, sigma, mode="degrade"), DRIFT_BINS)
        pg, qg = np.maximum(p, EPS), np.maximum(q, EPS)
        psi = float(np.sum((pg - qg) * np.log(pg / qg)))
        assert abs(psi - jeffreys_from_props(p, q)) < 1e-12, (sigma, psi)


def test_ks_hand_matches_scipy() -> None:
    # the reused-routine twin: the hand staircase-sup == scipy.stats.ks_2samp statistic, <1e-12.
    c = get_corpus()
    for sigma in (0.05, 0.2, 0.5):
        ref, cur = drift_windows(c, sigma, mode="degrade")
        ks = ks_two_sample(ref, cur)
        assert abs(ks["stat"] - ks["scipy_stat"]) < 1e-12, (sigma, ks)


def test_psi_kl_guards() -> None:
    # the EPS guard keeps PSI/KL finite on empty bins, and a distribution against itself is zero.
    a = np.linspace(0.0, 1.0, 40)
    assert abs(population_stability_index(a, a, 5)) < 1e-9
    assert abs(kl_divergence(np.array([0.5, 0.5]), np.array([0.5, 0.5]))) < 1e-12
    cur = np.linspace(2.0, 3.0, 40)                                     # disjoint support -> empty ref bins
    far = population_stability_index(a, cur, 4)
    assert np.isfinite(far) and far > 0.0, far


def test_shares_one_corpus() -> None:
    # the per-query samples here ARE the imported prereq's arrays (no second corpus built).
    c = get_corpus()
    assert np.allclose(ap_samples(c, "dense"), per_query_ap(c, "dense", "qrels_set"))
    assert np.allclose(ndcg_samples(c, "late_interaction"),
                       per_query_ndcg(c, "late_interaction", TOPK, gain_exponential, discount_log2))


def _run_all() -> None:
    print("significance_testing_calibration — verifying every claim:")
    test_pairing_reduces_variance()
    test_paired_se_below_unpaired()
    test_paired_separates_sooner_than_overlap()
    test_ttest_rel_twin()
    test_permutation_approximates_t()
    test_bootstrap_approximates_t()
    test_multiple_comparison_changes_verdict()
    test_map_pair_significant_ndcg_pair_not()
    test_power_n_exceeds_single_realization_n()
    test_duality_ci_excludes_zero_iff_reject()
    test_correction_adjusts_pvalues_up()
    test_calibration_raw_miscalibrated()
    test_recalibration_lowers_ece()
    test_platt_preserves_ranking_exactly()
    test_brier_decomposition_identity()
    test_heldout_ece_improves()
    test_drift_null_silent()
    test_psi_ks_monotone_in_knob()
    test_silent_decay_paired_beats_unpaired()
    test_input_vs_outcome_drift()
    test_psi_is_symmetrized_kl()
    test_ks_hand_matches_scipy()
    test_psi_kl_guards()
    test_shares_one_corpus()
    print("all significance/calibration/drift tests passed")
    viz_constants()


if __name__ == "__main__":
    _run_all()
