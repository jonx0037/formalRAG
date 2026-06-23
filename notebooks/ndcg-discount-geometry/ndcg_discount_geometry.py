"""NDCG — graded relevance, the discount geometry, and the ideal ranking as a rearrangement.

The reference implementation for the formalRAG `ndcg-discount-geometry` topic, the second node of the
evaluation layer. Its prerequisite, `set-metrics-precision-recall-map-mrr`, defined the set-metric
family over BINARY relevance (a doc is in R_q or not) and reframed every metric as an ESTIMATOR. NDCG
generalizes that on two axes at once:

  - GRADED relevance: a document carries a grade g in {0,1,2,3}, not a 0/1 flag. A perfect 10-K
    disclosure (g=3) and a tangential transcript snippet (g=1) are both "relevant" to set metrics; NDCG
    keeps the difference.
  - a DISCOUNT geometry: a hit at rank i is worth gain(g) * discount(i), with the celebrated
    discount(i) = 1/log2(i+1). DCG is the rank-discounted gain; IDCG is its maximum over orderings;
    NDCG = DCG / IDCG lands in [0,1].

We do NOT reimplement the corpus, the legs, or the estimator machinery — we IMPORT them from the
set-metrics module (which itself imports the published retrieval stack). The one genuinely new
construction is graded relevance, re-derived from the SAME exact-MaxSim oracle the set-metrics binary
qrels came from: per query, the oracle top-K docs (= `qrels_set`) get grades {1,2,3} by GLOBAL tertiles
of their oracle score; everything else is grade 0. So `{doc : grade >= 1}` equals the set-metrics
binary relevant set EXACTLY (the nesting / shared-baseline anchor), while IDCG still varies per query.

Two rigorous anchors:
  1. TWIN. Under linear gain (gain(g)=g) and the log2 discount, `ndcg_at_k` equals the IMPORTED
     `bm25.ndcg_at_k` to < 1e-12 — the reused-routine cross-check (the same "write a twin, prove it
     identical" rule the search-topic track uses).
  2. REARRANGEMENT. DCG = <gains-in-rank-order, descending-discounts>. By the rearrangement inequality,
     the inner product is maximized by sorting gains DESCENDING (the ideal ranking) and minimized by
     sorting them ASCENDING — so IDCG is exactly the ideal and NDCG in [0,1], =1 iff the top-k is ideal.

rigorFlag (primary): the GAIN function and the DISCOUNT are CONVENTIONS, not derived. Exponential gain
(2^g - 1) is the common modern default (Burges/LETOR) but is a choice; the log2 discount is a HEURISTIC
with no closed user model — unlike the geometric (rank-biased precision) discount p^(i-1), whose user
model gives E[docs examined] = 1/(1-p). The choice can flip the verdict (built and RUN below, never
assumed). Secondary: NDCG@k truncation can be inconsistent (Wang et al. 2013, COLT); grades are oracle-
score tertiles, a neutral stand-in for human editorial judgments; MNDCG is a sample mean with standard
error, so a gap within CI overlap is not yet real (the proper paired test is the next topic).

Run:  uv run --with numpy --with scipy \\
        python notebooks/ndcg-discount-geometry/ndcg_discount_geometry.py
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Import the prereq + the published stack. Add EVERY ancestor's hyphenated dir to the path (importing
# the set-metrics corpus pulls the whole multi-vector subtree at import time), then the direct prereq.
# We IMPORT the corpus, legs, and estimator machinery; we never reimplement them.
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
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bm25 import ndcg_at_k as bm25_ndcg_at_k                      # noqa: E402  (the twin baseline)
from late_interaction_learned_sparse import maxsim_matrix         # noqa: E402  (the exact oracle scorer)
from multi_vector_ann_retrieval import TOPK, N_DOCS              # noqa: E402
from set_metrics_precision_recall_map_mrr import (               # noqa: E402
    set_metrics_corpus, LEG_NAMES, metric_summary, projected_ci,
)

SEED = 0
RBP_P = 0.85                       # default rank-biased-precision persistence (the geometric discount)
GRADE_TERTILES = (1.0 / 3.0, 2.0 / 3.0)   # global oracle-score quantiles splitting grades 1|2|3


# =========================================================================== #
# Movement 0 — the shared corpus, the exact oracle, and GRADED relevance (the one new construction).
# =========================================================================== #

def _grade_from_score(score: float, t1: float, t2: float) -> int:
    """Grade a top-K document by its exact-MaxSim oracle score against two GLOBAL thresholds:
    3 (highly relevant) >= t2, 2 (relevant) >= t1, else 1 (marginally relevant). Top-K only — every
    graded doc is at least grade 1, so {grade >= 1} is exactly the set-metrics binary top-K set."""
    return 3 if score >= t2 else (2 if score >= t1 else 1)


def ndcg_corpus(seed: int = SEED) -> dict:
    """The set-metrics corpus (IMPORTED, not rebuilt by hand) augmented with the exact oracle score
    matrix and GRADED relevance. `maxsim_matrix(queries, docs)` is the very matrix `brute_topk` argsorts
    to make `qrels_set`, so grading its top-K is consistent with the binary truth by construction.
    Adds: `oracle_scores` (n_q x n_docs), `grades` (list of {doc_id: grade}), `t1`/`t2` (the tertiles),
    `ideal_grades` (per-query sorted grade multiset, the IDCG basis)."""
    corpus = set_metrics_corpus(seed)
    S = maxsim_matrix(corpus["queries"], corpus["docs"])           # exact MaxSim over ALL tokens
    nq, K = corpus["n_queries"], corpus["r_size"]
    topk_ids = [np.argsort(-S[q], kind="stable")[:K] for q in range(nq)]   # == brute_topk ordering
    topk_scores = np.array([S[q, topk_ids[q]] for q in range(nq)])         # (nq, K)
    flat = topk_scores.ravel()
    t1, t2 = (float(np.quantile(flat, GRADE_TERTILES[0])), float(np.quantile(flat, GRADE_TERTILES[1])))
    grades, ideal = [], []
    for q in range(nq):
        g = {int(topk_ids[q][j]): _grade_from_score(float(topk_scores[q][j]), t1, t2) for j in range(K)}
        grades.append(g)
        ideal.append(sorted(g.values(), reverse=True))
    corpus.update({"oracle_scores": S, "grades": grades, "t1": t1, "t2": t2, "ideal_grades": ideal})
    return corpus


_CORPUS: dict | None = None


def _corpus(seed: int = SEED) -> dict:
    """Module-scope cache: the corpus, oracle, and grades are built ONCE (the <60s budget)."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = ndcg_corpus(seed)
    return _CORPUS


def _ranking(corpus: dict, leg: str, q: int) -> list[int]:
    """The cached full ranking of `leg` on query `q` (built by the imported set-metrics corpus)."""
    return corpus["rankings"][leg][q]


# =========================================================================== #
# Movement 1 — the two design choices: gain functions and discount functions (both CONVENTIONS).
# =========================================================================== #

def gain_linear(g: int) -> float:
    """Linear gain: the grade itself (Jarvelin-Kekalainen 2002). Treats g=2 as worth twice g=1."""
    return float(g)


def gain_exponential(g: int) -> float:
    """Exponential gain 2^g - 1 (Burges/LETOR, the common modern default): sharply over-weights the top
    grades — g=3 is worth 7, g=1 worth 1 — so surfacing one perfect doc beats several marginal ones."""
    return float(2 ** g - 1)


def discount_log2(i: int) -> float:
    """The celebrated logarithmic discount 1/log2(i+1) at 1-indexed rank i. rank 1 -> 1, rank 2 ->
    1/log2(3) ~ 0.6309. A HEURISTIC: slow (heavy-tailed) decay, no closed user model (rigorFlag)."""
    return 1.0 / math.log2(i + 1)


def discount_geometric(i: int, p: float = RBP_P) -> float:
    """The geometric (rank-biased-precision) discount p^(i-1): a clean user model — the reader examines
    rank i with probability p^(i-1) and stops with prob 1-p, so E[docs examined] = 1/(1-p). Lighter tail
    than log2 (concentrates weight in the head as p shrinks)."""
    return p ** (i - 1)


def discount_reciprocal(i: int) -> float:
    """The reciprocal discount 1/i (the harmonic / MRR-like geometry): steeper than log2, heavier than a
    small-p geometric. Included as a third point of comparison for the discount-shape panel."""
    return 1.0 / i


# =========================================================================== #
# Movement 2 — DCG, IDCG, NDCG, and the inner-product / rearrangement view.
# =========================================================================== #

def dcg_of_gain_sequence(gains_in_order, discount=discount_log2) -> float:
    """DCG of a gain sequence already in rank order: sum_i gain_i * discount(i). This is the inner
    product <gains-in-rank-order, discount-vector> the rearrangement inequality acts on."""
    return sum(g * discount(i) for i, g in enumerate(gains_in_order, start=1))


def dcg_at_k(ranking, grades: dict, k: int = TOPK, gain=gain_linear, discount=discount_log2) -> float:
    """DCG@k = sum over the top-k ranked docs of gain(grade) * discount(rank). Absent docs contribute
    grade 0. GUARD: k <= 0 -> 0.0."""
    if k <= 0:
        return 0.0
    return sum(gain(grades.get(d, 0)) * discount(i) for i, d in enumerate(ranking[:k], start=1))


def ideal_dcg_at_k(grades: dict, k: int = TOPK, gain=gain_linear, discount=discount_log2) -> float:
    """IDCG@k = DCG of the IDEAL ordering: sort the grades descending, take the top-k, discount in place.
    By the rearrangement inequality this is the maximum DCG@k over every ordering of the documents."""
    ideal = sorted(grades.values(), reverse=True)[:k]
    return sum(gain(g) * discount(i) for i, g in enumerate(ideal, start=1))


def ndcg_at_k(ranking, grades: dict, k: int = TOPK, gain=gain_linear, discount=discount_log2) -> float:
    """NDCG@k = DCG@k / IDCG@k in [0,1]; 1 iff the top-k is an ideal ordering. GUARD: IDCG <= 0 (no
    relevant docs) -> 0.0. DEFAULTS (linear gain, log2 discount) make this the exact twin of
    `bm25.ndcg_at_k`; pass `gain_exponential` for the common modern NDCG."""
    idcg = ideal_dcg_at_k(grades, k, gain, discount)
    if idcg <= 0.0:
        return 0.0
    return dcg_at_k(ranking, grades, k, gain, discount) / idcg


# =========================================================================== #
# Movement 3 — discount geometry: head-mass, marginal value, the RBP user model.
# =========================================================================== #

def discount_weight_in_head(discount, k: int = TOPK, n: int = N_DOCS) -> float:
    """The fraction of total discount weight (over n positions) that sits in the top-k — the geometry's
    'how much does this measure care about the head' read. A light-tailed (geometric, small p) discount
    concentrates here; the heavy-tailed log2 spreads weight deep. GUARD: empty total -> 0.0."""
    total = sum(discount(i) for i in range(1, n + 1))
    head = sum(discount(i) for i in range(1, k + 1))
    return head / total if total > 0 else 0.0


def marginal_value(discount, i: int) -> float:
    """The value lost by being bumped from rank i to rank i+1: discount(i) - discount(i+1). Positive
    (discounts decrease) and itself decreasing in i — the cost of a demotion is steep at the top, flat
    deep down. This Delta is one factor of the swap sensitivity Delta = Delta_gain * Delta_discount."""
    return discount(i) - discount(i + 1)


def rbp_expected_docs_examined(p: float = RBP_P) -> float:
    """The rank-biased-precision user model's expected number of documents examined, 1/(1-p): the closed
    interpretation the log2 discount lacks. GUARD: p in [0,1)."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must be in [0, 1), got {p}")
    return 1.0 / (1.0 - p)


# =========================================================================== #
# Movement 4 — per-query NDCG and NDCG-as-ESTIMATOR (the set-metrics thread, generalized).
# =========================================================================== #

def per_query_ndcg(corpus: dict, leg: str, k: int = TOPK,
                   gain=gain_linear, discount=discount_log2) -> np.ndarray:
    """The per-query NDCG@k — the i.i.d.-across-queries samples whose mean is the leg's mean NDCG and
    whose spread is the estimator's variance (fed to the IMPORTED `metric_summary`)."""
    return np.array([ndcg_at_k(_ranking(corpus, leg, q), corpus["grades"][q], k, gain, discount)
                     for q in range(corpus["n_queries"])])


def mean_ndcg(corpus: dict, leg: str, k: int = TOPK, gain=gain_linear, discount=discount_log2) -> float:
    """Mean NDCG@k = (1/Q) sum_q NDCG_q — the sample mean of per-query NDCG."""
    return float(np.mean(per_query_ndcg(corpus, leg, k, gain, discount)))


def ndcg_se_scaling(corpus: dict, leg: str, sub_sizes=(5, 10, 20, 40, 80), trials: int = 1000,
                    gain=gain_exponential, discount=discount_log2, seed: int = 0) -> list[dict]:
    """The SE ~ 1/sqrt(n) demonstration for NDCG (the set-metrics SE-scaling experiment, on NDCG
    samples): treat the Q observed per-query NDCG as a sample, resample WITH replacement at each size n,
    and report empirical std of the mean against pop_std/sqrt(n). se_root_n = empirical_se*sqrt(n) ~ const."""
    nd = per_query_ndcg(corpus, leg, TOPK, gain, discount)
    pop_std = float(np.std(nd, ddof=1))
    rng = np.random.default_rng(seed)
    rows = []
    for n in sub_sizes:
        means = np.array([np.mean(rng.choice(nd, size=n, replace=True)) for _ in range(trials)])
        emp = float(np.std(means, ddof=1))
        rows.append({"n": int(n), "empirical_se": emp, "theory_se": pop_std / math.sqrt(n),
                     "se_root_n": emp * math.sqrt(n)})
    return rows


def bootstrap_ndcg_se(corpus: dict, leg: str, b: int = 2000, gain=gain_exponential,
                      discount=discount_log2, seed: int = 0) -> float:
    """Bootstrap SE of mean NDCG: resample the Q per-query NDCG with replacement b times, take the std —
    the assumption-light cross-check of the analytic std/sqrt(n) (per-query NDCG are bounded and skewed)."""
    nd = per_query_ndcg(corpus, leg, TOPK, gain, discount)
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(rng.choice(nd, size=nd.size, replace=True)) for _ in range(b)])
    return float(np.std(boots, ddof=1))


def _ndcg_mean_std(corpus: dict, leg: str, gain=gain_exponential, discount=discount_log2) -> tuple[float, float]:
    nd = per_query_ndcg(corpus, leg, TOPK, gain, discount)
    return float(np.mean(nd)), float(np.std(nd, ddof=1))


def projected_ndcg_separation_n(corpus: dict, leg_a: str, leg_b: str, gain=gain_exponential,
                                discount=discount_log2, n_max: int | None = None) -> int | None:
    """Smallest n at which the two legs' projected 95% CIs no longer overlap (None if never within n_max).
    n_max defaults to the Q queries we actually have; pass a larger cap to EXTRAPOLATE how many i.i.d.
    queries a small gap would need (the observed per-query std held fixed). The (mean, std) are hoisted
    out of the loop — they do not depend on n — so a large n_max stays cheap."""
    cap = n_max or corpus["n_queries"]
    mean_a, std_a = _ndcg_mean_std(corpus, leg_a, gain, discount)
    mean_b, std_b = _ndcg_mean_std(corpus, leg_b, gain, discount)
    for n in range(2, cap + 1):
        la, ua = projected_ci(mean_a, std_a, n)
        lb, ub = projected_ci(mean_b, std_b, n)
        if not (la <= ub and lb <= ua):
            return n
    return None


def two_leg_ndcg_comparison(corpus: dict, leg_a: str, leg_b: str,
                            gain=gain_exponential, discount=discount_log2) -> dict:
    """The 'is the NDCG gap real?' setup at full Q: each leg's `metric_summary`, the gap, and whether the
    95% CIs overlap. Per-query NDCG are PAIRED (same queries), so this naive test is conservative."""
    nda, ndb = (per_query_ndcg(corpus, leg_a, TOPK, gain, discount),
                per_query_ndcg(corpus, leg_b, TOPK, gain, discount))
    sa, sb = metric_summary(nda), metric_summary(ndb)
    overlap = sa["ci_lo"] <= sb["ci_hi"] and sb["ci_lo"] <= sa["ci_hi"]
    return {"leg_a": leg_a, "leg_b": leg_b, "summary_a": sa, "summary_b": sb,
            "gap": sa["mean"] - sb["mean"], "ci_overlap": overlap}


# =========================================================================== #
# Movement 5 — the headline FLIP: the convention changes the verdict (built and RUN, never assumed).
# =========================================================================== #

def leg_ndcg_table(corpus: dict, k: int = TOPK) -> dict:
    """Per-leg mean NDCG@k under the four convention combinations the viz toggles between."""
    return {leg: {
        "lin_log": mean_ndcg(corpus, leg, k, gain_linear, discount_log2),
        "exp_log": mean_ndcg(corpus, leg, k, gain_exponential, discount_log2),
        "exp_geo": mean_ndcg(corpus, leg, k, gain_exponential, lambda i: discount_geometric(i, RBP_P)),
        "lin_geo": mean_ndcg(corpus, leg, k, gain_linear, lambda i: discount_geometric(i, RBP_P)),
    } for leg in LEG_NAMES}


def _pairwise_reversal(score_a: dict, score_b: dict):
    """A leg pair (x, y) ordered one way by score_a and the OTHER by score_b (opposite-sign gaps) — the
    honest 'convention changes the verdict' claim even when one leg happens to top both tables."""
    for x, y in itertools.combinations(LEG_NAMES, 2):
        da, db = score_a[x] - score_a[y], score_b[x] - score_b[y]
        if da * db < 0:
            return {"pair": (x, y), "a_winner": x if da > 0 else y, "b_winner": x if db > 0 else y}
    return None


def convention_flips_verdict(corpus: dict, k: int = TOPK) -> dict:
    """Does the verdict depend on the gain/discount convention on THIS corpus? Reports the per-leg table,
    the argmax winner under each convention, any aggregate PAIRWISE reversal, and the count of PER-QUERY
    reversals between linear-vs-exponential gain (legs are a quality ladder, so an aggregate flip may not
    appear — the per-query count and the constructed toys are the robust demonstration). RUN before prose."""
    tbl = leg_ndcg_table(corpus, k)
    winners = {conv: max(tbl, key=lambda lg: tbl[lg][conv]) for conv in ("lin_log", "exp_log", "exp_geo")}
    gain_reversal = _pairwise_reversal({lg: tbl[lg]["lin_log"] for lg in LEG_NAMES},
                                       {lg: tbl[lg]["exp_log"] for lg in LEG_NAMES})
    disc_reversal = _pairwise_reversal({lg: tbl[lg]["exp_log"] for lg in LEG_NAMES},
                                       {lg: tbl[lg]["exp_geo"] for lg in LEG_NAMES})
    # per-query: count (leg pair, query) instances where linear and exponential gain disagree on order.
    # Precompute each leg's per-query NDCG once per convention (not once per pair).
    ndcg_lin = {leg: per_query_ndcg(corpus, leg, k, gain_linear, discount_log2) for leg in LEG_NAMES}
    ndcg_exp = {leg: per_query_ndcg(corpus, leg, k, gain_exponential, discount_log2) for leg in LEG_NAMES}
    pq_gain = 0
    for x, y in itertools.combinations(LEG_NAMES, 2):
        dla = ndcg_lin[x] - ndcg_lin[y]
        dea = ndcg_exp[x] - ndcg_exp[y]
        pq_gain += int(np.sum(dla * dea < -1e-12))
    return {"table": tbl, "winners": winners, "gain_reversal": gain_reversal,
            "disc_reversal": disc_reversal, "per_query_gain_reversals": pq_gain,
            "aggregate_flips": (gain_reversal is not None) or (disc_reversal is not None)}


def constructed_gain_flip() -> dict:
    """A minimal toy proving the GAIN convention flips the verdict. One query, |R| = 4: one perfect doc
    (g=3) and three marginal docs (g=1). Leg 'headline' puts the g=3 doc at rank 1 (the rest deep); leg
    'broad' puts the three g=1 docs in the top 3 and the g=3 doc at rank 4. Under exponential gain
    (g=3 -> 7) 'headline' wins; under linear gain (g=3 -> 3) 'broad' wins."""
    grades = {0: 3, 1: 1, 2: 1, 3: 1}                       # docs 0..3 relevant, 9..7 are filler (g=0)
    headline = [0, 9, 8, 7, 1, 2, 3]                        # perfect doc first, marginals buried
    broad = [1, 2, 3, 0, 9, 8, 7]                           # three marginals first, perfect doc at 4
    out = {}
    for gname, gain in (("exp", gain_exponential), ("lin", gain_linear)):
        h = ndcg_at_k(headline, grades, 4, gain, discount_log2)
        b = ndcg_at_k(broad, grades, 4, gain, discount_log2)
        out[gname] = {"headline": h, "broad": b, "winner": "headline" if h > b else "broad"}
    out["flips"] = out["exp"]["winner"] != out["lin"]["winner"]
    return out


def constructed_discount_flip() -> dict:
    """A minimal toy proving the DISCOUNT convention flips the verdict. One query, |R| = 3, all g=1. Leg
    'top_heavy' finds one relevant doc at rank 1 (the other two beyond the cutoff); leg 'deep' finds all
    three at ranks 2-4. Under a steep geometric (p=0.5) 'top_heavy' wins (rank 1 dominates); under the
    heavy-tailed log2 'deep' wins (three accumulated hits beat one)."""
    grades = {0: 1, 1: 1, 2: 1}
    top_heavy = [0, 7, 8, 9, 6, 5]                          # one hit at rank 1, docs 1,2 absent in top-4
    deep = [7, 0, 1, 2, 8, 9]                               # three hits at ranks 2,3,4
    out = {}
    for dname, disc in (("geo", lambda i: discount_geometric(i, 0.5)), ("log", discount_log2)):
        t = ndcg_at_k(top_heavy, grades, 4, gain_exponential, disc)
        d = ndcg_at_k(deep, grades, 4, gain_exponential, disc)
        out[dname] = {"top_heavy": t, "deep": d, "winner": "top_heavy" if t > d else "deep"}
    out["flips"] = out["geo"]["winner"] != out["log"]["winner"]
    return out


# =========================================================================== #
# Worked-example query selection (a representative, non-degenerate query for Panels A & B).
# =========================================================================== #

def pick_worked_query(corpus: dict, leg: str = "dense") -> int:
    """Choose ONE worked query for Panels A & B: among queries whose ideal profile contains a grade-3
    doc (so the exponential-gain effect is visible), the one whose `leg` NDCG (exponential gain) is
    closest to that leg's MEDIAN — a representative, non-degenerate case. Deterministic."""
    nd = per_query_ndcg(corpus, leg, TOPK, gain_exponential, discount_log2)
    med = float(np.median(nd))
    has3 = [q for q in range(corpus["n_queries"]) if 3 in corpus["ideal_grades"][q]]
    pool = has3 or list(range(corpus["n_queries"]))
    return min(pool, key=lambda q: abs(nd[q] - med))


def grades_in_rank_order(corpus: dict, leg: str, q: int, k: int = TOPK) -> list[int]:
    """The grade of each of the top-k docs as `leg` ranks them (the gain sequence the viz discounts)."""
    g = corpus["grades"][q]
    return [int(g.get(d, 0)) for d in _ranking(corpus, leg, q)[:k]]


# =========================================================================== #
# viz_constants — every number NDCGLaboratory.tsx mirrors to the decimal (cast numpy scalars).
# =========================================================================== #

def viz_constants() -> None:
    corpus = _corpus()
    nq, K = corpus["n_queries"], corpus["r_size"]
    wq = pick_worked_query(corpus, "dense")

    print("\n=== shared constants ===")
    print(f"N_DOCS = {corpus['n_docs']}  R_SIZE = {K}  N_QUERIES = {nq}  WORKED_Q = {wq}")
    print(f"GRADE_TERTILES t1={round(corpus['t1'], 4)}  t2={round(corpus['t2'], 4)}  RBP_P={RBP_P}")
    # global grade distribution across all queries (the relevance profile)
    allg = [g for q in range(nq) for g in corpus["grades"][q].values()]
    dist = {gr: int(np.sum(np.array(allg) == gr)) for gr in (1, 2, 3)}
    print(f"GRADE_DIST (over all {len(allg)} relevant judgments) = {dist}")

    print("\n=== Panel A & B — worked query: grades in each leg's rank order + the ideal ===")
    print(f"  IDEAL_GRADES(q={wq}) = {corpus['ideal_grades'][wq]}")
    for leg in LEG_NAMES:
        gro = grades_in_rank_order(corpus, leg, wq, K)
        nd_lin = ndcg_at_k(_ranking(corpus, leg, wq), corpus["grades"][wq], K, gain_linear, discount_log2)
        nd_exp = ndcg_at_k(_ranking(corpus, leg, wq), corpus["grades"][wq], K, gain_exponential, discount_log2)
        print(f"  {leg:16s} GRADES_IN_RANK_ORDER={gro}  NDCG_lin={round(nd_lin, 4)}  NDCG_exp={round(nd_exp, 4)}")

    print("\n=== Panel B — discount geometry (closed form; TS recomputes curves & head-mass) ===")
    for name, disc in (("log2", discount_log2), ("recip", discount_reciprocal),
                       ("geo_p085", lambda i: discount_geometric(i, RBP_P))):
        head = discount_weight_in_head(disc, K, corpus["n_docs"])
        mv1, mv5 = marginal_value(disc, 1), marginal_value(disc, 5)
        print(f"  {name:9s} head_mass@{K}={round(head, 4)}  marg(1)={round(mv1, 4)}  marg(5)={round(mv5, 4)}")
    print(f"  RBP E[docs examined] @p={RBP_P}: {round(rbp_expected_docs_examined(RBP_P), 4)}")

    print("\n=== Panel C — per-leg mean NDCG under each convention (bars reorder) ===")
    tbl = leg_ndcg_table(corpus, K)
    for leg in LEG_NAMES:
        v = tbl[leg]
        print(f"  {leg:16s} lin_log={round(v['lin_log'], 4)}  exp_log={round(v['exp_log'], 4)}"
              f"  exp_geo={round(v['exp_geo'], 4)}  lin_geo={round(v['lin_geo'], 4)}")
    gf, df = constructed_gain_flip(), constructed_discount_flip()
    print(f"  CONSTRUCTED gain-flip: exp->{gf['exp']['winner']} (h={round(gf['exp']['headline'], 3)},"
          f" b={round(gf['exp']['broad'], 3)}) | lin->{gf['lin']['winner']}"
          f" (h={round(gf['lin']['headline'], 3)}, b={round(gf['lin']['broad'], 3)})  flips={gf['flips']}")
    print(f"  CONSTRUCTED discount-flip: geo->{df['geo']['winner']} (t={round(df['geo']['top_heavy'], 3)},"
          f" d={round(df['geo']['deep'], 3)}) | log->{df['log']['winner']}"
          f" (t={round(df['log']['top_heavy'], 3)}, d={round(df['log']['deep'], 3)})  flips={df['flips']}")

    print("\n=== Panel D — NDCG as estimator (exponential gain, log2 discount) ===")
    for leg in LEG_NAMES:
        nd = per_query_ndcg(corpus, leg, K, gain_exponential, discount_log2)
        s = metric_summary(nd)
        print(f"  {leg:16s} SUMMARY={{'mean': {round(s['mean'], 4)}, 'std': {round(s['std'], 4)}, "
              f"'se': {round(s['se'], 4)}, 'ci': [{round(s['ci_lo'], 4)}, {round(s['ci_hi'], 4)}]}}")
        print(f"      PER_Q_NDCG={[round(float(v), 3) for v in nd]}")
    print("  SE_SCALING (late_interaction; n, empirical_se, theory_se, se*sqrt(n)):")
    for row in ndcg_se_scaling(corpus, "late_interaction"):
        print(f"    n={row['n']:3d}  emp={round(row['empirical_se'], 4)}  thy={round(row['theory_se'], 4)}"
              f"  se*sqrt(n)={round(row['se_root_n'], 4)}")
    # two contrasting pairs (chosen after running): the CLOSEST (needs many queries) and the CLEAREST.
    pairs = list(itertools.combinations(LEG_NAMES, 2))
    gaps = {pr: abs(mean_ndcg(corpus, pr[0], K, gain_exponential, discount_log2)
                    - mean_ndcg(corpus, pr[1], K, gain_exponential, discount_log2)) for pr in pairs}
    close, clear = min(gaps, key=gaps.get), max(gaps, key=gaps.get)
    tl = two_leg_ndcg_comparison(corpus, close[0], close[1])
    sep_in_q = projected_ndcg_separation_n(corpus, close[0], close[1])
    sep_extrap = projected_ndcg_separation_n(corpus, close[0], close[1], n_max=5000)  # how many it WOULD need
    sep_clear = projected_ndcg_separation_n(corpus, clear[0], clear[1])
    print(f"  CLOSEST_PAIR={close}  gap={round(tl['gap'], 4)}  full_n_overlap={tl['ci_overlap']}"
          f"  separate_within_Q={sep_in_q}  separate_at_n(extrapolated)={sep_extrap}")
    print(f"  CLEAREST_PAIR={clear}  gap={round(gaps[clear], 4)}  separate_at_n={sep_clear}")

    print("\n=== headline flip — convention changes the verdict (pairwise + per-query, RUN) ===")
    flip = convention_flips_verdict(corpus, K)
    print(f"  winners: lin_log={flip['winners']['lin_log']}  exp_log={flip['winners']['exp_log']}"
          f"  exp_geo={flip['winners']['exp_geo']}")
    print(f"  aggregate gain_reversal={flip['gain_reversal']}  disc_reversal={flip['disc_reversal']}")
    print(f"  per_query_gain_reversals={flip['per_query_gain_reversals']}  aggregate_flips={flip['aggregate_flips']}")


# =========================================================================== #
# Verification harness — every pedagogical claim is an assert.
# =========================================================================== #

def test_grades_nest_binary_qrels() -> None:
    # {doc : grade >= 1} is EXACTLY the set-metrics binary top-K relevant set (the shared-baseline anchor),
    # and every query has exactly K grades, each in {1,2,3}.
    c = _corpus()
    for q in range(c["n_queries"]):
        g = c["grades"][q]
        assert len(g) == c["r_size"], (q, len(g))
        assert {d for d, gr in g.items() if gr >= 1} == c["qrels_set"][q], q
        assert all(gr in (1, 2, 3) for gr in g.values()), (q, g)


def test_ndcg_matches_bm25_twin() -> None:
    # THE twin: under linear gain + log2 discount, ndcg_at_k == the IMPORTED bm25.ndcg_at_k, <1e-12,
    # on every leg/query and several cutoffs (the reused-routine cross-check).
    c = _corpus()
    for leg in LEG_NAMES:
        for q in range(0, c["n_queries"], 4):
            r, g = _ranking(c, leg, q), c["grades"][q]
            for k in (1, 5, 10, 25):
                mine = ndcg_at_k(r, g, k, gain_linear, discount_log2)
                theirs = bm25_ndcg_at_k(r, g, k)
                assert abs(mine - theirs) < 1e-12, (leg, q, k, mine, theirs)


def test_ndcg_bounds_and_perfect() -> None:
    c = _corpus()
    for leg in LEG_NAMES:
        for q in range(0, c["n_queries"], 3):
            for gain in (gain_linear, gain_exponential):
                v = ndcg_at_k(_ranking(c, leg, q), c["grades"][q], TOPK, gain, discount_log2)
                assert 0.0 <= v <= 1.0 + 1e-12, (leg, q, v)
    # an ideal ranking (relevant docs in descending-grade order, up front) scores exactly 1.
    grades = {0: 3, 1: 2, 2: 1}
    ideal = [0, 1, 2, 9, 8, 7]
    for gain in (gain_linear, gain_exponential):
        for disc in (discount_log2, discount_reciprocal, lambda i: discount_geometric(i, RBP_P)):
            assert abs(ndcg_at_k(ideal, grades, 3, gain, disc) - 1.0) < 1e-12


def test_rearrangement_inequality() -> None:
    # THE rigorous backbone: IDCG (gains sorted DESCENDING vs descending discounts) is the MAX over
    # orderings; ASCENDING gains is the MIN, strictly below IDCG whenever the grades are not all equal.
    rng = np.random.default_rng(0)
    c = _corpus()
    for q in range(0, c["n_queries"], 5):
        grades = sorted(c["grades"][q].values(), reverse=True)
        gvec = np.array([gain_exponential(g) for g in grades], dtype=float)
        dvec = np.array([discount_log2(i) for i in range(1, len(gvec) + 1)], dtype=float)
        # IDCG/worst via the inner-product helper; pin it to the elementwise np.sum form once.
        ideal = dcg_of_gain_sequence(list(np.sort(gvec)[::-1]), discount_log2)
        assert abs(ideal - float(np.sum(np.sort(gvec)[::-1] * dvec))) < 1e-12, q
        for _ in range(40):
            perm = rng.permutation(gvec)
            assert float(np.sum(perm * dvec)) <= ideal + 1e-12, q
        worst = dcg_of_gain_sequence(list(np.sort(gvec)), discount_log2)   # ascending gains = anti-sorted = minimum
        assert worst <= ideal + 1e-12
        if len(set(grades)) > 1:
            assert worst < ideal - 1e-9, (q, worst, ideal)     # strict when grades differ
    # IDCG equals dcg_at_k of the ideal ordering (the two definitions agree).
    for q in range(0, c["n_queries"], 7):
        g = c["grades"][q]
        ideal_ranking = sorted(g, key=lambda d: -g[d]) + [d for d in range(c["n_docs"]) if d not in g]
        assert abs(dcg_at_k(ideal_ranking, g, TOPK, gain_exponential, discount_log2)
                   - ideal_dcg_at_k(g, TOPK, gain_exponential, discount_log2)) < 1e-12, q


def test_discount_geometry() -> None:
    # a light-tailed geometric concentrates MORE weight in the head than the heavy-tailed log2; the
    # reciprocal sits between. Marginal value is positive and decreasing for every discount.
    n, k = N_DOCS, TOPK
    log_head = discount_weight_in_head(discount_log2, k, n)
    geo_head = discount_weight_in_head(lambda i: discount_geometric(i, RBP_P), k, n)
    recip_head = discount_weight_in_head(discount_reciprocal, k, n)
    assert geo_head > recip_head > log_head, (geo_head, recip_head, log_head)
    for disc in (discount_log2, discount_reciprocal, lambda i: discount_geometric(i, RBP_P)):
        mv = [marginal_value(disc, i) for i in range(1, 25)]
        assert all(m > -1e-12 for m in mv), mv                              # positive (decreasing discount)
        assert all(mv[i + 1] <= mv[i] + 1e-12 for i in range(len(mv) - 1)), mv  # itself decreasing
    assert abs(rbp_expected_docs_examined(0.8) - 5.0) < 1e-12               # 1/(1-p)


def test_ndcg_se_scales_as_inv_sqrt_n() -> None:
    c = _corpus()
    rows = ndcg_se_scaling(c, "late_interaction", trials=2000, seed=1)
    for row in rows:
        assert 0.70 <= row["empirical_se"] / max(row["theory_se"], 1e-12) <= 1.35, row
    emp = [row["empirical_se"] for row in rows]
    assert all(emp[i + 1] < emp[i] for i in range(len(emp) - 1)), emp        # strictly decreasing
    root = [row["se_root_n"] for row in rows]
    assert max(root) / min(root) < 1.40, root                                # se*sqrt(n) ~ const


def test_bootstrap_se_matches_analytic() -> None:
    c = _corpus()
    for leg in ("dense", "late_interaction"):
        analytic = metric_summary(per_query_ndcg(c, leg, TOPK, gain_exponential, discount_log2))["se"]
        boot = bootstrap_ndcg_se(c, leg, b=3000, seed=2)
        assert abs(boot - analytic) / max(analytic, 1e-12) < 0.18, (leg, boot, analytic)


def test_two_leg_separation_contrast() -> None:
    # the CLEAREST pair separates within the Q queries we have (a resolvable gap); the CLOSEST pair's
    # gap is so small it stays inside CI overlap at full Q but WOULD separate at a finite (larger) n —
    # the 'how many queries to resolve this NDCG gap' hook the significance topic resolves.
    c = _corpus()
    pairs = list(itertools.combinations(LEG_NAMES, 2))
    gaps = {pr: abs(mean_ndcg(c, pr[0], TOPK, gain_exponential, discount_log2)
                    - mean_ndcg(c, pr[1], TOPK, gain_exponential, discount_log2)) for pr in pairs}
    close, clear = min(gaps, key=gaps.get), max(gaps, key=gaps.get)
    sep_clear = projected_ndcg_separation_n(c, clear[0], clear[1])
    assert sep_clear is not None and 2 < sep_clear <= c["n_queries"], (clear, sep_clear)
    sep_extrap = projected_ndcg_separation_n(c, close[0], close[1], n_max=20000)
    assert sep_extrap is not None and sep_extrap > 2, (close, sep_extrap)   # a finite (if large) n exists


def test_convention_flip_constructed() -> None:
    # RUN, not assumed: the constructed gain-flip and discount-flip toys MUST reverse the verdict.
    gf, df = constructed_gain_flip(), constructed_discount_flip()
    assert gf["flips"], gf
    assert gf["exp"]["winner"] == "headline" and gf["lin"]["winner"] == "broad", gf
    assert df["flips"], df
    assert df["geo"]["winner"] == "top_heavy" and df["log"]["winner"] == "deep", df


def test_convention_flip_corpus_runs() -> None:
    # the corpus analysis must complete and report a real demonstration: either an aggregate pairwise
    # reversal OR at least one per-query reversal (the legs are a quality ladder, so we do not REQUIRE an
    # aggregate flip — we require the analysis to surface the convention's effect honestly).
    flip = convention_flips_verdict(_corpus())
    assert flip["aggregate_flips"] or flip["per_query_gain_reversals"] > 0, flip


def test_ideal_grades_consistent() -> None:
    # ideal_grades is the per-query sorted grade multiset and matches the grade dict's values.
    c = _corpus()
    for q in range(0, c["n_queries"], 6):
        assert c["ideal_grades"][q] == sorted(c["grades"][q].values(), reverse=True), q


def test_guards() -> None:
    assert dcg_at_k([1, 2, 3], {1: 2}, 0) == 0.0                      # k <= 0
    assert ndcg_at_k([1, 2, 3], {}, 5) == 0.0                         # no relevant docs -> IDCG 0
    try:
        rbp_expected_docs_examined(1.0)
        assert False, "p=1 should raise"
    except ValueError:
        pass


def _run_all() -> None:
    print("ndcg_discount_geometry — verifying every claim:")
    test_grades_nest_binary_qrels()
    test_ndcg_matches_bm25_twin()
    test_ndcg_bounds_and_perfect()
    test_rearrangement_inequality()
    test_discount_geometry()
    test_ndcg_se_scales_as_inv_sqrt_n()
    test_bootstrap_se_matches_analytic()
    test_two_leg_separation_contrast()
    test_convention_flip_constructed()
    test_convention_flip_corpus_runs()
    test_ideal_grades_consistent()
    test_guards()
    print("all NDCG tests passed")
    viz_constants()


if __name__ == "__main__":
    _run_all()
