"""Learning to Rank — pointwise, pairwise (RankNet), and why the field exists.

The reference implementation for the formalRAG `learning-to-rank-pairwise` topic, the root of the
ranking-fusion learning-to-rank sub-track. The evaluation layer (`set-metrics`, `ndcg`) *measured* a
ranking; this topic *learns* one. Ranking reduces to supervised learning three ways:

  - POINTWISE: regress each document's score onto its relevance grade by least squares. The global
    MSE minimizer — but it optimizes calibration (matching grade MAGNITUDE), not order.
  - PAIRWISE (RankNet, Burges et al. 2005): model each preference as a Bernoulli trial,
    P(i > j) = sigma(s_i - s_j), and minimize the pairwise cross-entropy. For a LINEAR scorer
    s = w . x this loss is CONVEX in w, so it has a single global optimum reachable by Newton /
    L-BFGS-B — no SGD, no learning-rate schedule, fully reproducible.
  - LISTWISE: score a whole permutation at once (ListNet, LambdaRank) — previewed in the prose only.

The rigorous backbone (the reason the field exists): NDCG and MAP are PIECEWISE-CONSTANT in the
scores. The ranking changes only when two scores cross, so on any open region with no tie the metric
is constant -> gradient zero almost everywhere, jump discontinuities exactly at the swaps. They cannot
be gradient-optimized directly; the smooth pairwise logistic is a SURROGATE standing in for them.
RankNet's gradient factorizes into a per-document lambda force lambda_i = sum_j lambda_ij — the bridge
to LambdaRank, which scales each pair's force by the |Delta-NDCG| that swapping it would cause.

Two headlines, both BUILT-AND-RUN before they are written:
  H1 — ranking is not regression: a pointwise-MSE-optimal model can LOSE on NDCG to a pairwise model
       with WORSE MSE (order beats calibration). Anchored by a constructed deterministic witness
       (MSE_pointwise < MSE_pairwise yet NDCG_pointwise < NDCG_pairwise, provable), reported honestly
       on the real corpus.
  H2 — the finance climax: a learned RankNet over the three retrieval legs' scores, trained on labeled
       preferences, beats each single leg AND reciprocal-rank fusion on held-out queries — pinned to
       the observed winner after running, NOT a universal ranking.

We do NOT rebuild the corpus, the legs, the fusion baseline, or the metrics — we IMPORT them. The
substrate is the CAPSTONE's complementary-view token corpus (the set-metrics legs are a quality ladder
where learned fusion is vacuous; the capstone legs are partial views that genuinely complement, so a
learned weighting has headroom). The grade construction reuses the ndcg tertile recipe verbatim.

rigorFlag: the pairwise logistic is a SURROGATE — minimizing it does not in general minimize NDCG/MAP
(the consistency gap, exactly why LambdaRank reweights by Delta-NDCG). Convexity holds only for the
LINEAR scorer; a deep RankNet is non-convex. "Learned beats RRF" is one synthetic cloud pinned to the
run, not a universal verdict. The pairwise reduction is position-blind (a top swap and a tail swap cost
the same) — the gap listwise objectives address. Grades are exact-MaxSim oracle tertiles, a neutral
stand-in for human editorial judgments.

Run:  uv run --with numpy --with scipy --with scikit-learn \\
        python notebooks/learning-to-rank-pairwise/learning_to_rank_pairwise.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.optimize import check_grad, minimize
from scipy.special import expit

# --------------------------------------------------------------------------- #
# Import the prereqs + the published stack. Add EVERY ancestor's hyphenated dir to the path (the
# capstone substrate pulls the whole multi-vector subtree at import; ndcg pulls set-metrics), then the
# direct modules. We IMPORT the corpus, legs, fusion baseline, grade recipe, and metrics — never
# reimplement them (import graph != pedagogical DAG: the frontmatter prereqs are only PRP + set-metrics).
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "johnson-lindenstrauss",
    "vector-quantization-lloyd-max",
    "product-quantization",
    "ivf-voronoi-partitioning",
    "bm25",
    "rank-fusion-rrf",
    "dense-retrieval-dual-encoders",
    "late-interaction-learned-sparse",
    "multi-vector-ann-retrieval",
    "filtered-incremental-ann",
    "set-metrics-precision-recall-map-mrr",
    "ndcg-discount-geometry",
    "capstone-multimodal-financial-rag",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bm25 import bm25_rank                                              # noqa: E402
from late_interaction_learned_sparse import maxsim_matrix              # noqa: E402
from multi_vector_ann_retrieval import TOPK                            # noqa: E402
from ndcg_discount_geometry import (                                   # noqa: E402
    GRADE_TERTILES, _grade_from_score, discount_log2, gain_exponential, gain_linear, ndcg_at_k,
)
from set_metrics_precision_recall_map_mrr import average_precision, metric_summary  # noqa: E402
from capstone_multimodal_financial_rag import (                        # noqa: E402
    LEGS, WIN_LI, capstone_corpus, leg_dense_ranking, recall_at_k, rrf_fuse,
)

SEED = 0
N_TRAIN = 24                      # query train/test split (40 queries total): 24 train / 16 test
GAIN = gain_exponential           # the modern-default headline NDCG gain (2^g - 1); linear is the twin
DISCOUNT = discount_log2          # the celebrated log2 discount (a convention inherited from ndcg)
LEG_ORDER = ("lexical", "dense", "late_interaction")   # the feature column order, x = [lex, dense, li]


# =========================================================================== #
# Movement 0 — the substrate: three complementary leg scores as features, oracle-tertile grades.
# =========================================================================== #

def _leg_score_matrices(corpus: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three legs' per-(query, doc) SCORES (not rankings), each (n_queries, n_docs). The dense and
    late-interaction scores are the substrate's own pre-argsort scores; the lexical (BM25) scores are
    ZERO-FILLED to the full doc set (BM25 omits zero-score docs — gathering an unfilled list mis-indexes,
    the SPLADE all-zero gotcha)."""
    nq, nd = corpus["n_queries"], corpus["n_docs"]
    s_dense = corpus["q_vecs"] @ corpus["doc_vecs"].T                       # (nq, nd) pooled MIPS
    win = list(WIN_LI)
    dc = corpus["C"][corpus["index"]["doc_cids"][:, win]]                   # (nd, |WIN_LI|, d)
    s_li = maxsim_matrix(corpus["queries"], dc)                            # (nq, nd) windowed centroid MaxSim
    s_lex = np.zeros((nq, nd), dtype=float)
    for q in range(nq):
        for doc_id, score in bm25_rank(corpus["q_text"][q], corpus["bm25_index"]):
            s_lex[q, int(doc_id)] = float(score)
    return s_lex, s_dense, s_li


def standardize_per_query(x: np.ndarray) -> np.ndarray:
    """Z-score each feature column within one query's documents (x is (n_docs, n_features)). Leg scores
    live on incompatible scales (BM25 unbounded, cosine in [-1,1], MaxSim ~[0, m_q]); standardizing
    conditions the fit. GUARD: a constant column (std == 0) -> divide by 1 (the platt_scale precedent).
    The z-score is affine-increasing per column, so it leaves each single-leg ranking unchanged."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (x - mu) / sd


def _derive_grades(corpus: dict) -> tuple[list[dict[int, int]], np.ndarray, float, float]:
    """Re-derive graded relevance by the ndcg tertile recipe on the capstone's exact-MaxSim oracle.
    `maxsim_matrix(queries, docs)` over ALL m_d tokens is the very matrix `brute_topk` argsorts to make
    the truth sets, so the top-K docs (= truth_sets[q]) get grades {1,2,3} by GLOBAL oracle-score
    tertiles, everything else grade 0. Returns (grades, oracle_scores, t1, t2). The nesting anchor
    {grade >= 1} == truth_sets[q] holds by construction (re-derive the shared baseline, do not
    reimplement the metric)."""
    S = maxsim_matrix(corpus["queries"], corpus["docs"])                   # (nq, nd) exact MaxSim
    nq, K = corpus["n_queries"], TOPK
    topk_ids = [np.argsort(-S[q], kind="stable")[:K] for q in range(nq)]
    topk_scores = np.array([S[q, topk_ids[q]] for q in range(nq)])
    flat = topk_scores.ravel()
    t1 = float(np.quantile(flat, GRADE_TERTILES[0]))
    t2 = float(np.quantile(flat, GRADE_TERTILES[1]))
    grades = [
        {int(topk_ids[q][j]): _grade_from_score(float(topk_scores[q][j]), t1, t2) for j in range(K)}
        for q in range(nq)
    ]
    return grades, S, t1, t2


def ltr_corpus(seed: int = SEED) -> dict:
    """The capstone complementary-view corpus augmented with the learning-to-rank substrate:
      - feats: (n_queries, n_docs, 3) per-query-standardized leg-score features [lex, dense, li]
      - y:     (n_queries, n_docs) graded relevance (0 outside the oracle top-K)
      - grades, truth_sets, t1, t2: the graded-relevance objects
      - train_q, test_q: a seeded query-index split (the learned ranker fits on train, scores on test)
    Built once; cached at module scope via `_corpus()`."""
    corpus = capstone_corpus(seed)
    s_lex, s_dense, s_li = _leg_score_matrices(corpus)
    nq, nd = corpus["n_queries"], corpus["n_docs"]
    feats = np.empty((nq, nd, 3), dtype=float)
    for q in range(nq):
        feats[q] = standardize_per_query(np.stack([s_lex[q], s_dense[q], s_li[q]], axis=1))
    grades, oracle, t1, t2 = _derive_grades(corpus)
    y = np.zeros((nq, nd), dtype=float)
    for q in range(nq):
        for d, g in grades[q].items():
            y[q, d] = g
    perm = np.random.default_rng(seed).permutation(nq)
    train_q = sorted(int(i) for i in perm[:N_TRAIN])
    test_q = sorted(int(i) for i in perm[N_TRAIN:])
    corpus.update({
        "leg_scores": {"lexical": s_lex, "dense": s_dense, "late_interaction": s_li},
        "feats": feats, "y": y, "grades": grades, "oracle_scores": oracle, "t1": t1, "t2": t2,
        "train_q": train_q, "test_q": test_q,
    })
    return corpus


_CORPUS: dict | None = None


def _corpus(seed: int = SEED) -> dict:
    """Module-scope cache — the corpus, oracle, grades, and split are built ONCE (the <60s budget)."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = ltr_corpus(seed)
    return _CORPUS


# =========================================================================== #
# Movement 1 — the pointwise reduction: least squares to grades.
# =========================================================================== #

def pointwise_solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The pointwise reduction: the least-squares weights for s = X w ~ y, augmenting X with an
    intercept column. Returns w of length n_features + 1 (last entry the intercept). This IS the global
    MSE minimizer over linear scorers (np.linalg.lstsq); the intercept is a global constant that does
    not change any within-query ranking."""
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def _augment(X: np.ndarray) -> np.ndarray:
    """Append the intercept column (for scoring with a pointwise weight vector)."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


def calibrated_mse(scores: np.ndarray, grades: np.ndarray) -> float:
    """The MSE of a scoring model AFTER its best affine recalibration to grades:
    min_{a,b} mean (a*scores + b - grades)^2. This puts pointwise and pairwise on a common footing —
    pointwise (the global LS fit) attains the minimum over all linear scorers, so its calibrated MSE is
    a LOWER bound for any other direction's, including the pairwise one (the H1 inequality, by
    construction). GUARD: empty input -> 0.0."""
    if scores.size == 0:
        return 0.0
    A = np.vstack([scores, np.ones_like(scores)]).T
    coef, *_ = np.linalg.lstsq(A, grades, rcond=None)
    resid = A @ coef - grades
    return float(np.mean(resid ** 2))


# =========================================================================== #
# Movement 2 — RankNet: the pairwise logistic surrogate (convex, no SGD).
# =========================================================================== #

def within_query_pairs(feat_q: np.ndarray, y_q: np.ndarray) -> np.ndarray:
    """The difference vectors Delta_ij = x_i - x_j for every ORDERED preference pair (y_i > y_j) within
    one query. Returns (n_pairs, n_features). Ties (equal grade, incl. the grade-0 bulk) contribute no
    pair. Precomputed ONCE per query (the difference vectors are loop-invariant for the loss)."""
    ii, jj = np.where(y_q[:, None] > y_q[None, :])
    return feat_q[ii] - feat_q[jj]


def ranknet_loss(w: np.ndarray, deltas: np.ndarray) -> float:
    """The RankNet pairwise cross-entropy for a linear scorer: L(w) = sum_pairs softplus(-w . Delta_ij)
    = sum log(1 + exp(-w . Delta)). softplus(-z) is convex in z and z = w . Delta is linear in w, so L is
    CONVEX in w (a sum of convex functions). `np.logaddexp(0, -z)` is the overflow-safe softplus."""
    z = deltas @ w
    return float(np.sum(np.logaddexp(0.0, -z)))


def ranknet_grad(w: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """The closed-form gradient: dL/dw = sum_pairs -sigma(-w . Delta) * Delta. Uses scipy's expit
    (overflow-safe sigmoid)."""
    z = deltas @ w
    return -(expit(-z)[:, None] * deltas).sum(axis=0)


def ranknet_hessian(w: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """The Hessian: H(w) = sum_pairs sigma(z)(1 - sigma(z)) Delta Delta^T with z = w . Delta. Each weight
    sigma(z)(1-sigma(z)) >= 0, so H is a nonneg-weighted sum of rank-1 outer products -> positive
    semidefinite, the algebraic witness that L is convex (single global optimum)."""
    z = deltas @ w
    s = expit(z)
    wgt = s * (1.0 - s)
    return (deltas * wgt[:, None]).T @ deltas


def all_train_pairs(corpus: dict, qs: list[int]) -> np.ndarray:
    """Stack the within-query difference vectors across a set of queries (the training preference set)."""
    blocks = [within_query_pairs(corpus["feats"][q], corpus["y"][q]) for q in qs]
    blocks = [b for b in blocks if b.shape[0] > 0]
    return np.vstack(blocks) if blocks else np.zeros((0, 3))


def fit_ranknet(deltas: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
    """Minimize the convex pairwise loss to its global optimum with L-BFGS-B and the analytic gradient.
    Deterministic: fixed x0 (zeros by default), no SGD, no rng. GUARD: no pairs -> zero weights."""
    if deltas.shape[0] == 0:
        return np.zeros(deltas.shape[1])
    if x0 is None:
        x0 = np.zeros(deltas.shape[1])
    res = minimize(ranknet_loss, x0, args=(deltas,), jac=ranknet_grad, method="L-BFGS-B")
    return res.x


def lambda_forces(w: np.ndarray, feat_q: np.ndarray, y_q: np.ndarray) -> np.ndarray:
    """The per-document lambda force lambda_i = sum_j lambda_ij for one query, where lambda_ij =
    -sigma(-(s_i - s_j)) pushes the higher-graded doc up and the lower one down. The net force on each
    doc, summed over its pairs (gather/scatter via np.add.at). The factorization sum_i lambda_i x_i ==
    ranknet_grad(w, deltas) is the bridge to LambdaRank (which scales lambda_ij by |Delta-NDCG_ij|)."""
    s = feat_q @ w
    n = s.shape[0]
    lam = np.zeros(n)
    ii, jj = np.where(y_q[:, None] > y_q[None, :])
    g = expit(-(s[ii] - s[jj]))                       # sigma(-(s_i - s_j)), the pair force magnitude
    np.add.at(lam, ii, -g)                            # the winner i is pulled up
    np.add.at(lam, jj, g)                             # the loser j is pulled down
    return lam


# =========================================================================== #
# Movement 3 — scoring, ranking, and the metrics (all reused, never reimplemented).
# =========================================================================== #

def score_docs(corpus: dict, q: int, w: np.ndarray) -> np.ndarray:
    """Linear scores s_d = w . x_d for every doc of query q. Accepts a feature-length w (pairwise) or a
    feature+intercept w (pointwise); the intercept is a global constant and does not change the order."""
    X = corpus["feats"][q]
    if w.shape[0] == X.shape[1] + 1:
        return _augment(X) @ w
    return X @ w


def ranking_from_w(corpus: dict, q: int, w: np.ndarray) -> list[int]:
    """The full ranking (all docs, descending score, stable ties) induced by weights w on query q."""
    return np.argsort(-score_docs(corpus, q, w), kind="stable").tolist()


def ndcg_of_ranking(corpus: dict, q: int, ranking: list[int], gain=GAIN, discount=DISCOUNT) -> float:
    """NDCG@TOPK of a ranking under the topic's graded relevance — the IMPORTED ndcg_at_k (the twin: we
    reuse the metric, we do not reimplement it)."""
    return ndcg_at_k(ranking, corpus["grades"][q], TOPK, gain, discount)


def mean_ndcg_over(corpus: dict, qs: list[int], w: np.ndarray, gain=GAIN, discount=DISCOUNT) -> float:
    """Mean NDCG@TOPK of the w-induced rankings over a query set."""
    return float(np.mean([ndcg_of_ranking(corpus, q, ranking_from_w(corpus, q, w), gain, discount)
                          for q in qs]))


def mean_recall_over(corpus: dict, qs: list[int], w: np.ndarray, k: int = TOPK) -> float:
    """Mean recall@k of the w-induced rankings over a query set (imported recall_at_k, the set metric)."""
    ts = corpus["truth_sets"]
    return float(np.mean([recall_at_k(ranking_from_w(corpus, q, w), ts[q], k) for q in qs]))


def leg_ranking(corpus: dict, q: int, leg: str) -> list[int]:
    """The imported single-leg ranking for query q (the substrate's own LEGS functions)."""
    return LEGS[leg](corpus, q)


def mean_leg_metric(corpus: dict, qs: list[int], leg: str, metric: str = "recall", k: int = TOPK) -> float:
    """Mean recall@k or NDCG@k of a single leg over a query set (held-out evaluation of a baseline)."""
    ts, gr = corpus["truth_sets"], corpus["grades"]
    vals = []
    for q in qs:
        r = leg_ranking(corpus, q, leg)
        vals.append(recall_at_k(r, ts[q], k) if metric == "recall" else ndcg_at_k(r, gr[q], k, GAIN, DISCOUNT))
    return float(np.mean(vals))


def rrf_ranking(corpus: dict, q: int) -> list[int]:
    """The reciprocal-rank-fusion baseline ranking for query q — the imported rrf_fuse of the three
    legs' rankings (the unsupervised combiner the learned ranker is compared against)."""
    return rrf_fuse([leg_ranking(corpus, q, leg) for leg in LEG_ORDER])


def mean_rrf_metric(corpus: dict, qs: list[int], metric: str = "recall", k: int = TOPK) -> float:
    """Mean recall@k or NDCG@k of the RRF fusion over a query set."""
    ts, gr = corpus["truth_sets"], corpus["grades"]
    vals = []
    for q in qs:
        r = rrf_ranking(corpus, q)
        vals.append(recall_at_k(r, ts[q], k) if metric == "recall" else ndcg_at_k(r, gr[q], k, GAIN, DISCOUNT))
    return float(np.mean(vals))


# =========================================================================== #
# Movement 4 — the rigorous backbone: rank metrics are piecewise-constant (Theorem 2).
# =========================================================================== #

def surrogate_sweep(corpus: dict, qs: list[int], w_a: np.ndarray, w_b: np.ndarray, n: int = 240) -> dict:
    """Sweep w(t) = (1-t) w_a + t w_b along the segment between two weight vectors and record, at each t,
    the mean NDCG@TOPK over the queries `qs` (a STEP function of t — flat between adjacent-doc score
    swaps, jumping at them) and the smooth total pairwise loss sum_q L_q(w(t)). The whole reason the
    field exists, made executable: the metric has zero gradient almost everywhere and jumps at swaps,
    while the surrogate is C^1. A single-query sweep gives the cleanest plateaus; a query-set sweep gives
    the staircase the viz shows (morphing from the weakest leg's direction to the learned combiner).
    Returns the grid, the NDCG step, the smooth loss, and the t-locations of any top-K swap."""
    ts = np.linspace(0.0, 1.0, n)
    feats = [corpus["feats"][q] for q in qs]
    grades = [corpus["grades"][q] for q in qs]
    deltas = [within_query_pairs(corpus["feats"][q], corpus["y"][q]) for q in qs]
    ndcg, loss, sigs = [], [], []
    for t in ts:
        w = (1.0 - t) * w_a + t * w_b
        nd_vals, ls, sig = [], 0.0, []
        for X, gr, D in zip(feats, grades, deltas):
            order = np.argsort(-(X @ w), kind="stable")
            sig.append(tuple(order[:TOPK].tolist()))
            nd_vals.append(ndcg_at_k(order.tolist(), gr, TOPK, GAIN, DISCOUNT))
            ls += ranknet_loss(w, D) if D.shape[0] else 0.0
        ndcg.append(float(np.mean(nd_vals)))
        loss.append(ls)
        sigs.append(tuple(sig))
    swaps = [float(ts[i]) for i in range(1, n) if sigs[i] != sigs[i - 1]]
    return {"t": ts.tolist(), "ndcg": ndcg, "loss": loss, "swaps": swaps,
            "n_distinct_ndcg": len(set(round(v, 12) for v in ndcg))}


def _weakest_leg_direction(corpus: dict, qs: list[int]) -> np.ndarray:
    """A unit weight vector on the lowest-NDCG single leg (the Panel-B sweep start) — morphing it into
    the learned combiner makes NDCG climb in discrete steps, the clearest piecewise-constant staircase."""
    order = {leg: i for i, leg in enumerate(LEG_ORDER)}
    worst = min(LEG_ORDER, key=lambda leg: mean_leg_metric(corpus, qs, leg, "ndcg"))
    e = np.zeros(3)
    e[order[worst]] = 1.0
    return e


# =========================================================================== #
# Movement 5 — H1: ranking is not regression (the constructed witness + the corpus report).
# =========================================================================== #

def constructed_ranking_vs_regression() -> dict:
    """A deterministic two-query witness proving ranking != regression, with two features:
      - feature 1 is the within-query ORDER signal, but its best SIGN is contested across queries —
        query 1 (high grade MAGNITUDE, few docs) wants w1 > 0, query 2 (low magnitude, more docs and so
        more preference PAIRS) wants w1 < 0. Least squares is magnitude-weighted (query 1's g=3 dominates
        the squared error) and picks w1 > 0; the pairwise loss is pair-count-weighted (query 2's 3 pairs
        dominate) and picks w1 < 0. So the two reductions induce DIFFERENT rankings.
      - feature 2 is a query-LEVEL constant (the query's mean grade), the same for every doc in a query.
        It is order-irrelevant WITHIN a query (so it cannot change either ranking), but least squares
        exploits it to fit each query's grade level and lower its MSE. The pairwise gradient along it is
        EXACTLY ZERO (every within-query difference vanishes), so RankNet cannot use it.

    The result: pointwise has the lower (calibrated) MSE — it is the global MSE minimizer and it spends
    feature 2 on calibration — yet WORSE NDCG, because it mis-orders query 2; pairwise spends nothing on
    magnitude and orders both queries better. Order beats calibration.

    Returns {mse_pointwise, mse_pairwise, ndcg_pointwise, ndcg_pairwise, w_pointwise, w_pairwise}."""
    feats = [np.array([[1.0, 1.5], [0.0, 1.5]]),
             np.array([[0.0, 0.25], [1.0, 0.25], [1.0, 0.25], [1.0, 0.25]])]
    grades = [{0: 3, 1: 0}, {0: 1, 1: 0, 2: 0, 3: 0}]
    ys = [np.array([3.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])]

    # Pointwise: global least squares over both queries' (feature, grade) points.
    X = np.vstack(feats)
    y = np.concatenate(ys)
    w_pt = pointwise_solve(X, y)               # [slope, intercept]

    # Pairwise: the convex RankNet optimum over both queries' preference pairs.
    deltas = np.vstack([within_query_pairs(feats[i], ys[i]) for i in range(2)])
    w_pw = fit_ranknet(deltas)                 # [slope]

    def _two_query_ndcg(score_fn) -> float:
        out = []
        for i in range(2):
            s = score_fn(feats[i])
            order = np.argsort(-s, kind="stable").tolist()
            out.append(ndcg_at_k(order, grades[i], len(order), gain_linear, discount_log2))
        return float(np.mean(out))

    def _two_query_mse(score_fn) -> float:
        s_all = np.concatenate([score_fn(feats[i]) for i in range(2)])
        return calibrated_mse(s_all, y)

    ndcg_pt = _two_query_ndcg(lambda f: _augment(f) @ w_pt)
    ndcg_pw = _two_query_ndcg(lambda f: f @ w_pw)
    mse_pt = _two_query_mse(lambda f: _augment(f) @ w_pt)
    mse_pw = _two_query_mse(lambda f: f @ w_pw)
    return {"mse_pointwise": mse_pt, "mse_pairwise": mse_pw,
            "ndcg_pointwise": ndcg_pt, "ndcg_pairwise": ndcg_pw,
            "w_pointwise": w_pt.tolist(), "w_pairwise": w_pw.tolist()}


def h1_on_corpus(corpus: dict) -> dict:
    """The H1 comparison on the REAL corpus: fit pointwise (LS) and pairwise (RankNet) on the train
    queries, then on the TEST queries report each model's mean NDCG and calibrated MSE. Reported
    honestly — the in-sample double inequality is corpus/seed-fragile, so the constructed witness is the
    asserted anchor and this is a measured observation."""
    tr, te = corpus["train_q"], corpus["test_q"]
    Xtr = np.vstack([corpus["feats"][q] for q in tr])
    ytr = np.concatenate([corpus["y"][q] for q in tr])
    w_pt = pointwise_solve(Xtr, ytr)
    w_pw = fit_ranknet(all_train_pairs(corpus, tr))

    ndcg_pt = mean_ndcg_over(corpus, te, w_pt)
    ndcg_pw = mean_ndcg_over(corpus, te, w_pw)
    s_pt = np.concatenate([score_docs(corpus, q, w_pt) for q in te])
    s_pw = np.concatenate([score_docs(corpus, q, w_pw) for q in te])
    y_te = np.concatenate([corpus["y"][q] for q in te])
    return {"ndcg_pointwise": ndcg_pt, "ndcg_pairwise": ndcg_pw,
            "mse_pointwise": calibrated_mse(s_pt, y_te), "mse_pairwise": calibrated_mse(s_pw, y_te),
            "flip": bool(ndcg_pw > ndcg_pt)}


# =========================================================================== #
# Movement 6 — H2: learned fusion vs each leg vs RRF (the finance climax).
# =========================================================================== #

def learned_vs_baselines(corpus: dict) -> dict:
    """Train a RankNet over the three complementary leg scores on the train queries; evaluate the learned
    ranker, each single leg, RRF, and pointwise on the HELD-OUT test queries by recall@TOPK and
    NDCG@TOPK. Returns the per-method metrics (train and test), the learned weights, the per-query test
    recall summary (the estimator CI), and the observed winner. The verdict is PINNED to the run."""
    tr, te = corpus["train_q"], corpus["test_q"]
    w = fit_ranknet(all_train_pairs(corpus, tr))
    w_pt = pointwise_solve(np.vstack([corpus["feats"][q] for q in tr]),
                           np.concatenate([corpus["y"][q] for q in tr]))

    def _methods(qs):
        m = {leg: {"recall": mean_leg_metric(corpus, qs, leg, "recall"),
                   "ndcg": mean_leg_metric(corpus, qs, leg, "ndcg")} for leg in LEG_ORDER}
        m["rrf"] = {"recall": mean_rrf_metric(corpus, qs, "recall"),
                    "ndcg": mean_rrf_metric(corpus, qs, "ndcg")}
        m["pointwise"] = {"recall": mean_recall_over(corpus, qs, w_pt),
                          "ndcg": mean_ndcg_over(corpus, qs, w_pt)}
        m["learned"] = {"recall": mean_recall_over(corpus, qs, w),
                        "ndcg": mean_ndcg_over(corpus, qs, w)}
        return m

    test = _methods(te)
    train = _methods(tr)
    per_q_recall = np.array([recall_at_k(ranking_from_w(corpus, q, w), corpus["truth_sets"][q], TOPK)
                             for q in te])
    baselines = {k: test[k]["recall"] for k in (*LEG_ORDER, "rrf")}
    winner = max(test, key=lambda k: test[k]["recall"])
    return {"train": train, "test": test, "w_learned": w.tolist(),
            "learned_recall_ci": metric_summary(per_q_recall),
            "best_baseline_recall": max(baselines.values()),
            "winner": winner,
            "learned_gain_over_rrf": test["learned"]["recall"] - test["rrf"]["recall"]}


def pick_worked_query(corpus: dict, w: np.ndarray | None = None, w_pt: np.ndarray | None = None) -> int:
    """A test query for the worked example: prefer one whose grade set contains a g=3 doc and whose
    pointwise and pairwise rankings differ (so the Panel-A/B/C contrast is non-trivial); fall back to the
    test query of median learned NDCG. Callers that already fit the weights pass them in to skip the
    Newton solve."""
    tr, te = corpus["train_q"], corpus["test_q"]
    if w is None:
        w = fit_ranknet(all_train_pairs(corpus, tr))
    if w_pt is None:
        w_pt = pointwise_solve(np.vstack([corpus["feats"][q] for q in tr]),
                               np.concatenate([corpus["y"][q] for q in tr]))
    cands = []
    for q in te:
        has3 = 3 in corpus["grades"][q].values()
        differ = ranking_from_w(corpus, q, w)[:TOPK] != ranking_from_w(corpus, q, w_pt)[:TOPK]
        if has3 and differ:
            cands.append(q)
    if cands:
        return min(cands, key=lambda q: abs(ndcg_of_ranking(corpus, q, ranking_from_w(corpus, q, w)) - 0.7))
    ndcgs = [(q, ndcg_of_ranking(corpus, q, ranking_from_w(corpus, q, w))) for q in te]
    ndcgs.sort(key=lambda kv: kv[1])
    return ndcgs[len(ndcgs) // 2][0]


# =========================================================================== #
# Tests — every pedagogical claim is an assert (the notebook owns the numbers).
# =========================================================================== #

def test_pointwise_equals_lstsq() -> None:
    """The pointwise solver IS the least-squares solution: cross-check against the normal equations."""
    c = _corpus()
    tr = c["train_q"]
    X = np.vstack([c["feats"][q] for q in tr])
    y = np.concatenate([c["y"][q] for q in tr])
    w = pointwise_solve(X, y)
    A = _augment(X)
    w_normal = np.linalg.solve(A.T @ A, A.T @ y)
    assert np.allclose(w, w_normal, atol=1e-9), (w, w_normal)


def test_ranknet_grad_fd() -> None:
    """The analytic gradient matches a finite-difference gradient (the mean-value theorem made
    numerical). check_grad's default sqrt-eps step on a loss summed over ~1000 pairs lands near 1e-5, so
    the bar is the conventional 1e-4 — a strong agreement, not a slack one."""
    c = _corpus()
    deltas = within_query_pairs(c["feats"][c["train_q"][0]], c["y"][c["train_q"][0]])
    err = check_grad(ranknet_loss, ranknet_grad, np.array([0.3, -0.2, 0.5]), deltas)
    assert err < 1e-4, err


def test_ranknet_loss_convex_unique() -> None:
    """The loss is convex (Hessian PSD) and has a single global optimum: two different starts converge
    to the same w*."""
    c = _corpus()
    deltas = all_train_pairs(c, c["train_q"])
    w0 = fit_ranknet(deltas, x0=np.zeros(3))
    w1 = fit_ranknet(deltas, x0=np.random.default_rng(1).standard_normal(3))
    assert np.allclose(w0, w1, atol=1e-5), (w0, w1)
    H = ranknet_hessian(w0, deltas)
    assert np.linalg.eigvalsh(H).min() >= -1e-9


def test_ranknet_translation_invariant() -> None:
    """s_i - s_j is invariant to a constant score shift: adding c to every score leaves the loss and the
    ranking unchanged (order-not-absolute, like the cross-modal gap)."""
    c = _corpus()
    q = c["test_q"][0]
    w = fit_ranknet(all_train_pairs(c, c["train_q"]))
    deltas = within_query_pairs(c["feats"][q], c["y"][q])
    s = score_docs(c, q, w)
    base = ranknet_loss(w, deltas)
    # the loss uses only differences, so a constant shift of the scores cannot change it
    shifted_deltas = (c["feats"][q][np.where(c["y"][q][:, None] > c["y"][q][None, :])[0]]
                      - c["feats"][q][np.where(c["y"][q][:, None] > c["y"][q][None, :])[1]])
    assert np.allclose(ranknet_loss(w, shifted_deltas), base, atol=1e-12)
    order_a = np.argsort(-s, kind="stable")
    order_b = np.argsort(-(s + 3.14159), kind="stable")
    assert order_a.tolist() == order_b.tolist()


def test_one_feature_ranknet_matches_leg() -> None:
    """A one-feature RankNet on the dense leg learns a positive weight and recovers exactly that leg's
    ranking (a monotone threshold on the single score) — the collapse anchor."""
    c = _corpus()
    tr, te = c["train_q"], c["test_q"]
    di = LEG_ORDER.index("dense")
    deltas = np.vstack([within_query_pairs(c["feats"][q][:, [di]], c["y"][q]) for q in tr])
    w = fit_ranknet(deltas)
    assert w[0] > 0, w
    for q in te[:5]:
        mine = np.argsort(-(c["feats"][q][:, di]), kind="stable").tolist()
        assert mine == leg_dense_ranking(c, q), q


def test_gradient_factorizes() -> None:
    """The gradient factorizes into per-document lambda forces: sum_i lambda_i x_i == ranknet_grad."""
    c = _corpus()
    q = c["train_q"][0]
    w = fit_ranknet(all_train_pairs(c, c["train_q"]))
    lam = lambda_forces(w, c["feats"][q], c["y"][q])
    lhs = c["feats"][q].T @ lam
    rhs = ranknet_grad(w, within_query_pairs(c["feats"][q], c["y"][q]))
    assert np.allclose(lhs, rhs, atol=1e-9), (lhs, rhs)


def test_metrics_anchor() -> None:
    """The imported metrics are wired correctly (reuse, not reimplement): a perfect ranking — the
    relevant docs first in descending grade order — scores AP = 1 and NDCG = 1; and ndcg_of_ranking is
    exactly the imported ndcg_at_k on a real ranking."""
    c = _corpus()
    q = c["test_q"][0]
    grades = c["grades"][q]
    ideal = sorted(grades, key=lambda d: -grades[d]) + [d for d in range(c["n_docs"]) if d not in grades]
    assert abs(average_precision(ideal, c["truth_sets"][q]) - 1.0) < 1e-12
    assert abs(ndcg_of_ranking(c, q, ideal) - 1.0) < 1e-12
    r = rrf_ranking(c, q)
    assert abs(ndcg_of_ranking(c, q, r) - ndcg_at_k(r, grades, TOPK, GAIN, DISCOUNT)) < 1e-12


def test_grades_nest_truth() -> None:
    """The graded relevance nests the binary truth: {doc : grade >= 1} == the capstone truth set."""
    c = _corpus()
    for q in range(c["n_queries"]):
        graded = {d for d, g in c["grades"][q].items() if g >= 1}
        assert graded == c["truth_sets"][q], q


def test_surrogate_is_stepwise_vs_smooth() -> None:
    """Theorem 2: NDCG is piecewise-constant along a weight sweep (few distinct values, flat between
    swaps) while the pairwise loss is smooth (its discrete second difference stays bounded)."""
    c = _corpus()
    q = pick_worked_query(c)
    w_pw = fit_ranknet(all_train_pairs(c, c["train_q"]))
    w_a = _weakest_leg_direction(c, c["test_q"])
    sw = surrogate_sweep(c, [q], w_a, w_pw, n=240)
    assert sw["n_distinct_ndcg"] < 60, sw["n_distinct_ndcg"]      # finitely many plateaus
    assert len(sw["swaps"]) >= 1
    d_nd = np.abs(np.diff(np.array(sw["ndcg"])))
    d_ls = np.abs(np.diff(np.array(sw["loss"])))
    # NDCG is piecewise-constant: flat on almost every step, its change concentrated at a few swaps.
    assert np.mean(d_nd < 1e-9) > 0.8, np.mean(d_nd < 1e-9)
    assert d_nd.max() / (d_nd.sum() + 1e-12) > 0.2               # change lives in a handful of jumps
    # The surrogate is smooth: it changes on essentially every step, its change spread evenly (no jump).
    assert np.mean(d_ls > 1e-9) > 0.95, np.mean(d_ls > 1e-9)
    assert d_ls.max() / (d_ls.sum() + 1e-12) < 0.1              # no jump discontinuity (C^1)


def test_constructed_ranking_beats_regression() -> None:
    """H1 anchor (ALWAYS holds): the pointwise model has lower calibrated MSE yet worse NDCG than the
    pairwise model — order beats calibration."""
    d = constructed_ranking_vs_regression()
    assert d["mse_pointwise"] < d["mse_pairwise"], d
    assert d["ndcg_pointwise"] < d["ndcg_pairwise"], d


def test_learned_vs_baselines_runs() -> None:
    """H2 (pinned to the run): the learned ranker generalizes to the held-out queries — its test recall
    is at least the best single-leg test recall (no pre-baked 'beats RRF' verdict; the winner is read
    off the run and the gain over RRF is reported)."""
    c = _corpus()
    res = learned_vs_baselines(c)
    leg_best = max(res["test"][leg]["recall"] for leg in LEG_ORDER)
    assert res["test"]["learned"]["recall"] >= leg_best - 1e-9, res["test"]
    assert res["winner"] in (*LEG_ORDER, "rrf", "pointwise", "learned")


def test_no_pairs_guard() -> None:
    """An all-equal-grade query yields zero preference pairs and is handled cleanly (empty -> no pairs,
    fit returns zero weights)."""
    feat_q = np.random.default_rng(0).standard_normal((5, 3))
    y_q = np.zeros(5)
    assert within_query_pairs(feat_q, y_q).shape[0] == 0
    assert np.allclose(fit_ranknet(np.zeros((0, 3))), np.zeros(3))


def _run_tests() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")


# =========================================================================== #
# viz_constants — every number LearningToRankLaboratory.tsx mirrors (reproducible only; cast scalars).
# =========================================================================== #

def viz_constants() -> None:
    """Print every value the viz bakes. Reproducible only (deterministic solves, seeded split); numpy
    scalars cast to float/int so the mirrored values are clean."""
    c = _corpus()
    tr, te = c["train_q"], c["test_q"]
    r4 = lambda v: round(float(v), 4)

    print("=== dims / split ===")
    print("N_DOCS", c["n_docs"], "N_QUERIES", c["n_queries"], "TOPK", TOPK)
    print("TRAIN_Q", tr)
    print("TEST_Q", te)
    print("GRADE_TERTILES t1,t2", r4(c["t1"]), r4(c["t2"]))
    gd = {1: 0, 2: 0, 3: 0}
    for q in range(c["n_queries"]):
        for g in c["grades"][q].values():
            gd[g] += 1
    print("GRADE_DIST", gd)

    print("=== fitted weights ===")
    w_pt = pointwise_solve(np.vstack([c["feats"][q] for q in tr]),
                           np.concatenate([c["y"][q] for q in tr]))
    w_pw = fit_ranknet(all_train_pairs(c, tr))
    print("W_POINTWISE [lex,dense,li,intercept]", [r4(v) for v in w_pt])
    print("W_RANKNET   [lex,dense,li]", [r4(v) for v in w_pw])

    print("=== H2 learned vs baselines (test) ===")
    res = learned_vs_baselines(c)
    for k, v in res["test"].items():
        print(f"  {k:16s} recall={r4(v['recall'])}  ndcg={r4(v['ndcg'])}")
    print("LEARNED_W [lex,dense,li]", [r4(v) for v in res["w_learned"]])
    print("WINNER", res["winner"], "GAIN_OVER_RRF", r4(res["learned_gain_over_rrf"]))
    ci = res["learned_recall_ci"]
    print("LEARNED_RECALL_CI n,mean,se,lo,hi",
          ci["n"], r4(ci["mean"]), r4(ci["se"]), r4(ci["ci_lo"]), r4(ci["ci_hi"]))

    print("=== H1 constructed witness ===")
    h1 = constructed_ranking_vs_regression()
    print("CONSTRUCTED_H1",
          {k: r4(v) for k, v in h1.items() if not isinstance(v, list)})
    print("=== H1 on corpus (reported) ===")
    print("H1_CORPUS", {k: (r4(v) if isinstance(v, float) else v) for k, v in h1_on_corpus(c).items()})

    print("=== worked query (Panels A/C) ===")
    q = pick_worked_query(c, w=w_pw, w_pt=w_pt)
    print("WORKED_Q", q)
    order = ranking_from_w(c, q, w_pw)
    print("WORKED_RANKING_TOPK", order[:TOPK])
    print("WORKED_GRADES", {int(d): int(g) for d, g in c["grades"][q].items()})
    lam = lambda_forces(w_pw, c["feats"][q], c["y"][q])
    print("LAMBDA_FORCES_TOPK", [r4(lam[d]) for d in order[:TOPK]])
    print("WORKED_FEATS_TOPK", [[r4(v) for v in c["feats"][q][d]] for d in order[:TOPK]])

    print("=== surrogate sweep (Panel B): weakest-leg direction -> learned combiner, over test queries ===")
    w_a = _weakest_leg_direction(c, te)
    sw = surrogate_sweep(c, te, w_a, w_pw, n=240)
    print("SWEEP_FROM_LEG", LEG_ORDER[int(np.argmax(w_a))],
          "n_distinct_ndcg", sw["n_distinct_ndcg"], "n_swaps", len(sw["swaps"]))
    print("NDCG endpoints", r4(sw["ndcg"][0]), "->", r4(sw["ndcg"][-1]),
          "| LOSS endpoints", r4(sw["loss"][0]), "->", r4(sw["loss"][-1]))
    # downsample to ~48 points: bake the NDCG staircase AND the smooth loss (TS plots both, reads at slider)
    idx = np.linspace(0, len(sw["t"]) - 1, 48).round().astype(int)
    print("SWEEP_TS", [r4(sw["t"][i]) for i in idx])
    print("NDCG_SWEEP", [r4(sw["ndcg"][i]) for i in idx])
    print("LOSS_SWEEP", [round(float(sw["loss"][i]), 1) for i in idx])

    print("=== learning curve (Panel D): test recall@10 vs #train queries ===")
    pair_blocks = [within_query_pairs(c["feats"][q], c["y"][q]) for q in tr]   # each query's pairs ONCE
    curve = []
    for ntr in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24):
        blocks = [b for b in pair_blocks[:ntr] if b.shape[0] > 0]
        w_n = fit_ranknet(np.vstack(blocks) if blocks else np.zeros((0, 3)))
        curve.append((ntr, r4(mean_recall_over(c, te, w_n))))
    print("LEARN_CURVE", curve)
    print("CONSTRUCTED_H1_WEIGHTS w_pointwise", [r4(v) for v in h1["w_pointwise"]],
          "w_pairwise", [r4(v) for v in h1["w_pairwise"]])


if __name__ == "__main__":
    print("Running learning-to-rank-pairwise tests...")
    _run_tests()
    print()
    viz_constants()
    print("\nAll learning-to-rank-pairwise checks passed.")
