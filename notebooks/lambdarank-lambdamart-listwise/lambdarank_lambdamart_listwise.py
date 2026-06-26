"""LambdaRank, LambdaMART, and listwise objectives — making the ranking loss position-aware.

The reference implementation for the formalRAG `lambdarank-lambdamart-listwise` topic, the SECOND node
of the ranking-fusion learning-to-rank sub-track. The predecessor (`learning-to-rank-pairwise`) reduced
ranking to a pairwise logistic surrogate (RankNet) whose gradient factorizes into a per-document force
`lambda_i = sum_j lambda_ij`. That force is POSITION-BLIND: a swap at ranks 1-2 and a swap at ranks
99-100 carry the same `sigma(s_j - s_i)` magnitude. This topic makes the objective position-aware three
ways, and is honest about what each one is:

  - LAMBDARANK (Burges-Ragno-Le 2006): scale each pair force by the |Delta-NDCG_ij| that swapping the
    two documents would cause. The discount geometry makes |Delta-NDCG| large for top-of-list swaps and
    small for tail swaps, so the gradient is CONCENTRATED at the head. The catch (the honest core): the
    weight depends on the current ranking, so the lambda field is NOT the gradient of any scalar loss —
    GLOBALLY. Within a single ranking cell (no swaps) the weights are constant and lambda IS a gradient
    (a weighted-RankNet loss); across a swap the field is DISCONTINUOUS, so no C^1 global potential
    exists. LambdaRank is a heuristic gradient that empirically ascends NDCG (Donmez-Svore-Burges 2009
    local optimality, named not proved).
  - LAMBDAMART (Wu-Burges-Svore-Gao 2010): feed those lambdas as pseudo-residuals to gradient-boosted
    regression trees (Friedman 2001). The trees buy a NONLINEAR scorer the linear RankNet cannot express
    — demonstrated on a constructed XOR-interaction instance where every linear scorer is rank-capped
    below 1 while a depth-2 tree reaches NDCG = 1.
  - LISTWISE (ListNet, Cao et al. 2007; ListMLE, Xia et al. 2008): replace the pairwise heuristic with a
    proper PROBABILISTIC loss over the whole permutation. The Plackett-Luce likelihood gives ListMLE =
    -log P(pi*|s), which IS convex in the scores (logsumexp minus linear) — a genuine loss with a single
    optimum, the principled contrast to LambdaRank's non-integrable field.

The arc: pairwise (a loss, position-blind) -> LambdaRank (position-aware, NOT a global loss) -> listwise
(a proper convex loss). We do NOT rebuild the corpus, legs, grades, RankNet, or metrics — we IMPORT them
from `learning-to-rank-pairwise` (which itself imports the capstone substrate + ndcg grade recipe). The
import graph is not the pedagogical DAG: the frontmatter prereqs are only the two graph edges
(learning-to-rank-pairwise + ndcg-discount-geometry).

Headlines, all BUILT-AND-RUN before written:
  - Delta-NDCG closed form == physical-swap NDCG difference (<1e-12), with UNTRUNCATED DCG so the
    two-term identity is exact for every pair.
  - Uniform-weight LambdaRank == the imported `lambda_forces` bit-for-bit (the collapse anchor / the
    bridge back to RankNet).
  - The lambda field is conservative for RankNet and NON-integrable for LambdaRank: the spectator-pair
    force is DISCONTINUOUS across a swap (a 3-doc witness), while within a cell both Jacobians are
    symmetric.
  - ListMLE / ListNet are convex (PSD Hessian, two-start) with FD-checked gradients; ListMLE -> 0 as a
    perfectly-ordered score is scaled up (a limit, not a finite-fit equality).
  - LambdaRank's |Delta-NDCG| weight is strictly DECREASING in the pair's ranks (equal grade gap), so its
    gradient mass is top-concentrated where RankNet's is spread by pair count (provable, seed-free).
  - LambdaMART beats every linear scorer on a constructed XOR instance (NDCG 1 vs a sub-1 ceiling);
    on the real near-linear leg features the gain is small / within CI, reported honestly.

rigorFlag: LambdaRank optimizes NO scalar loss — it is a heuristic gradient (locally a weighted-RankNet
loss, globally non-integrable). Its empirical NDCG win over RankNet on this forgiving corpus is small /
within CI; the seed-free claim is the STRUCTURAL top-concentration of the gradient. Listwise convexity
holds only for the LINEAR scorer (a deep/tree model is non-convex). LambdaMART's escape of the linear
ceiling is shown on a constructed instance; on the real leg features trees roughly tie linear. Grades are
exact-MaxSim oracle tertiles, a neutral stand-in for human editorial judgments.

Run:  uv run --with numpy --with scipy --with scikit-learn \\
        python notebooks/lambdarank-lambdamart-listwise/lambdarank_lambdamart_listwise.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.optimize import check_grad, minimize
from scipy.special import expit, logsumexp, softmax
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------------------------------- #
# Import the predecessor + the published stack. Mirror the learning-to-rank-pairwise sys.path block (the
# capstone substrate pulls the whole multi-vector subtree at import; ndcg pulls set-metrics) and append
# `learning-to-rank-pairwise` itself — we import its corpus, RankNet, lambda forces, scoring/ranking/metric
# helpers, and the ndcg gain/discount/NDCG closed forms. Import graph != DAG.
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
    "learning-to-rank-pairwise",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ndcg_discount_geometry import (                                   # noqa: E402
    ideal_dcg_at_k, marginal_value, ndcg_at_k,
)
from set_metrics_precision_recall_map_mrr import metric_summary        # noqa: E402
from capstone_multimodal_financial_rag import recall_at_k             # noqa: E402
from learning_to_rank_pairwise import (                               # noqa: E402
    DISCOUNT, GAIN, LEG_ORDER, TOPK, SEED,
    _corpus as _ltr_corpus, all_train_pairs, fit_ranknet, lambda_forces, mean_leg_metric,
    mean_recall_over, mean_rrf_metric, ranking_from_w,
)

N_ROUNDS = 30                  # default LambdaMART boosting rounds
NU = 0.1                       # boosting shrinkage / step size (Friedman's nu)
TREE_DEPTH = 3                 # DecisionTreeRegressor(max_depth=3, random_state=0) — the nonlinearity
N_NEG = 10                     # grade-0 hard negatives per query in the listwise candidate set
CAND = TOPK + N_NEG            # candidate-set size for ListMLE / ListNet (the K graded docs + N_NEG negs)


# =========================================================================== #
# Movement 0 — the substrate (imported verbatim from learning-to-rank-pairwise).
# =========================================================================== #

def _corpus(seed: int = SEED) -> dict:
    """The learning-to-rank-pairwise corpus, reused wholesale: feats (n_q, n_docs, 3) standardized leg
    scores, y / grades (oracle-tertile graded relevance), oracle_scores, truth_sets, train_q / test_q.
    Re-derive nothing — the import IS the shared baseline."""
    return _ltr_corpus(seed)


# =========================================================================== #
# Movement 1 — the Delta-NDCG closed form (the LambdaRank weight).
# =========================================================================== #

def _grade_gain_vector(y_q: np.ndarray, gain=GAIN) -> np.ndarray:
    """gain(grade) for every doc of a query (y_q holds float grades in {0,1,2,3})."""
    return np.array([gain(int(round(float(g)))) for g in y_q], dtype=float)


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    """The 1-indexed rank position of each document under a score vector (stable descending argsort).
    rank[d] = 1 means doc d is ranked first."""
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(scores.shape[0], dtype=int)
    rank[order] = np.arange(1, scores.shape[0] + 1)
    return rank


def delta_ndcg_swap(grades_q: dict, ranking: list[int], p: int, q_pos: int,
                    gain=GAIN, discount=DISCOUNT) -> float:
    """The closed-form |Delta-NDCG| of swapping the documents at 1-indexed rank positions p and q_pos:
        |Delta-NDCG| = |G(g_p) - G(g_q)| * |D(p) - D(q)| / IDCG,
    where g_p is the grade of the doc currently at rank p (absent docs grade 0). IDCG is the UNTRUNCATED
    ideal DCG (k = full list), so the two-term identity is exact for EVERY pair — including pairs that
    straddle the top-k cutoff, where a truncated DCG would silently drop a term. Reuses the imported
    gain/discount/ideal_dcg_at_k."""
    n = len(ranking)
    g_p = gain(grades_q.get(ranking[p - 1], 0))
    g_q = gain(grades_q.get(ranking[q_pos - 1], 0))
    idcg = ideal_dcg_at_k(grades_q, n, gain, discount)
    if idcg <= 0.0:
        return 0.0
    return abs((g_p - g_q) * (discount(p) - discount(q_pos))) / idcg


def delta_ndcg_brute(grades_q: dict, ranking: list[int], p: int, q_pos: int,
                     gain=GAIN, discount=DISCOUNT) -> float:
    """The same |Delta-NDCG| computed by PHYSICALLY swapping the two docs and re-running the imported
    ndcg_at_k at the full (untruncated) cutoff. The twin anchor for delta_ndcg_swap (<1e-12)."""
    n = len(ranking)
    before = ndcg_at_k(ranking, grades_q, n, gain, discount)
    swapped = list(ranking)
    swapped[p - 1], swapped[q_pos - 1] = swapped[q_pos - 1], swapped[p - 1]
    after = ndcg_at_k(swapped, grades_q, n, gain, discount)
    return abs(after - before)


# =========================================================================== #
# Movement 2 — LambdaRank: the Delta-NDCG-weighted lambda field.
# =========================================================================== #

def ranknet_field(s: np.ndarray, y_q: np.ndarray) -> np.ndarray:
    """The RankNet per-document force as a function of a raw per-query SCORE vector s (not weights w):
    lambda_i = sum_j lambda_ij with lambda_ij = -sigma(-(s_i - s_j)) over ordered pairs y_i > y_j. This is
    exactly the imported lambda_forces' inner computation (lambda = dL/ds), lifted to score space so the
    integrability question can be asked in R^n. With s = feat_q @ w it equals lambda_forces(w, feat_q, y_q)."""
    n = s.shape[0]
    lam = np.zeros(n)
    ii, jj = np.where(y_q[:, None] > y_q[None, :])
    g = expit(-(s[ii] - s[jj]))
    np.add.at(lam, ii, -g)
    np.add.at(lam, jj, g)
    return lam


def lambdarank_field(s: np.ndarray, y_q: np.ndarray, grades_q: dict,
                     gain=GAIN, discount=DISCOUNT) -> np.ndarray:
    """The LambdaRank force in SCORE space: lambda_ij = -sigma(-(s_i - s_j)) * |Delta-NDCG_ij|, where the
    weight uses the rank positions induced BY s. The ranking-dependence of the weight is exactly what
    makes this field non-integrable. Reduces to ranknet_field when every |Delta-NDCG| is forced to 1."""
    n = s.shape[0]
    lam = np.zeros(n)
    ii, jj = np.where(y_q[:, None] > y_q[None, :])
    g = expit(-(s[ii] - s[jj]))
    rank = _rank_positions(s)
    idcg = ideal_dcg_at_k(grades_q, n, gain, discount)
    G = _grade_gain_vector(y_q, gain)
    D = np.array([discount(int(rank[d])) for d in range(n)])
    wgt = np.abs((G[ii] - G[jj]) * (D[ii] - D[jj])) / (idcg if idcg > 0 else 1.0)
    np.add.at(lam, ii, -g * wgt)
    np.add.at(lam, jj, g * wgt)
    return lam


def lambdarank_forces(w: np.ndarray, feat_q: np.ndarray, y_q: np.ndarray, grades_q: dict,
                      gain=GAIN, discount=DISCOUNT, uniform: bool = False) -> np.ndarray:
    """The per-document LambdaRank force for a LINEAR scorer s = feat_q @ w. With uniform=True the weights
    collapse to 1 and this is bit-for-bit the imported lambda_forces (the collapse anchor / RankNet bridge)."""
    s = feat_q @ w
    if uniform:
        return ranknet_field(s, y_q)
    return lambdarank_field(s, y_q, grades_q, gain, discount)


def lambdarank_gradient(w: np.ndarray, feat_q: np.ndarray, y_q: np.ndarray, grades_q: dict,
                        gain=GAIN, discount=DISCOUNT) -> np.ndarray:
    """The model gradient direction sum_i lambda_i x_i for a linear scorer — the descent direction
    LambdaRank steps along. NOTE: this is NOT the gradient of any scalar loss (Movement 3); it is the
    factorized lambda field pushed through the features."""
    lam = lambdarank_forces(w, feat_q, y_q, grades_q, gain, discount)
    return feat_q.T @ lam


def fit_lambdarank(corpus: dict, qs: list[int], n_iters: int = 40, lr: float = 0.5,
                   x0: np.ndarray | None = None, gain=GAIN, discount=DISCOUNT) -> np.ndarray:
    """Fit a linear LambdaRank scorer by gradient DESCENT on the (non-integrable) lambda field: each
    iteration re-scores, re-ranks, recomputes the |Delta-NDCG| weights, sums the per-doc forces into a
    model gradient sum_i lambda_i x_i, and steps w against it. Deterministic (fixed x0 = zeros, no SGD).
    There is no loss to monitor — the field is the only object — so we run a fixed schedule."""
    w = np.zeros(3) if x0 is None else np.array(x0, dtype=float)
    for _ in range(n_iters):
        grad = np.zeros(3)
        for q in qs:
            grad += lambdarank_gradient(w, corpus["feats"][q], corpus["y"][q], corpus["grades"][q],
                                        gain, discount)
        w = w - lr * grad / max(len(qs), 1)
    return w


def gradient_concentration_by_rank(corpus: dict, q: int, w: np.ndarray,
                                   gain=GAIN, discount=DISCOUNT) -> dict:
    """Bucket the magnitude of each pair's force by the TOP rank it touches (min of the two rank
    positions), for RankNet (uniform weight) and LambdaRank (|Delta-NDCG| weight). Returns per-bucket
    mass and the top-3 share for each — the illustration of the seed-free claim that LambdaRank piles
    gradient on the head while RankNet spreads it by pair count."""
    s = corpus["feats"][q] @ w
    y_q, grades_q = corpus["y"][q], corpus["grades"][q]
    n = s.shape[0]
    ii, jj = np.where(y_q[:, None] > y_q[None, :])
    g = expit(-(s[ii] - s[jj]))
    rank = _rank_positions(s)
    idcg = ideal_dcg_at_k(grades_q, n, gain, discount)
    G = _grade_gain_vector(y_q, gain)
    D = np.array([discount(int(rank[d])) for d in range(n)])
    dndcg = np.abs((G[ii] - G[jj]) * (D[ii] - D[jj])) / (idcg if idcg > 0 else 1.0)
    top_rank = np.minimum(rank[ii], rank[jj])
    nbuck = TOPK
    rn = np.zeros(nbuck)
    lr = np.zeros(nbuck)
    for b in range(1, nbuck + 1):
        m = top_rank == b
        rn[b - 1] = float(np.sum(g[m]))
        lr[b - 1] = float(np.sum(g[m] * dndcg[m]))
    rn_share = float(rn[:3].sum() / (rn.sum() + 1e-12))
    lr_share = float(lr[:3].sum() / (lr.sum() + 1e-12))
    return {"ranknet_mass": rn.tolist(), "lambdarank_mass": lr.tolist(),
            "ranknet_top3_share": rn_share, "lambdarank_top3_share": lr_share}


def weight_decreasing_in_rank(gain=GAIN, discount=DISCOUNT) -> dict:
    """The seed-free structural fact: for an EQUAL grade gap, the |Delta-NDCG| swap weight is strictly
    decreasing in the pair's ranks because |D(p) - D(q)| shrinks as the ranks grow (the marginal_value
    geometry from ndcg). Returns the adjacent-pair weights at the head vs the tail (proportional to the
    discount marginal), which strictly order head > tail."""
    head = abs(marginal_value(discount, 1))     # |D(1) - D(2)|
    tail = abs(marginal_value(discount, 9))     # |D(9) - D(10)|
    return {"head_1_2": float(head), "tail_9_10": float(tail), "strictly_decreasing": bool(head > tail)}


# =========================================================================== #
# Movement 3 — is lambda a gradient? The integrability obstruction (the rigorFlag core).
# =========================================================================== #

def numerical_jacobian(field_fn, s: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian J[i,j] = d lambda_i / d s_j of a score-space field. Within a single
    ranking cell (eps small enough not to cross a swap) the Jacobian of a gradient field is symmetric
    (Clairaut). The perturbation must stay inside the cell — pass a well-separated s."""
    n = s.shape[0]
    J = np.zeros((n, n))
    for j in range(n):
        sp = s.copy(); sp[j] += eps
        sm = s.copy(); sm[j] -= eps
        J[:, j] = (field_fn(sp) - field_fn(sm)) / (2.0 * eps)
    return J


def swap_discontinuity_witness(field_fn, s: np.ndarray, a: int, b: int, eps: float = 1e-4) -> float:
    """Evaluate the force on doc a just below and just above the s_a = s_b swap hyperplane (move s_a by
    +-eps around s_b) and return the jump |lambda_a(above) - lambda_a(below)|. For a conservative field
    (RankNet) the force is continuous (jump -> 0); for LambdaRank a SPECTATOR pair's weight jumps when a
    and b exchange rank, so lambda_a is discontinuous (a fixed positive jump). Needs >= 3 docs with
    distinct grades — the discontinuity lives in a third doc's pair, not the swapping pair itself."""
    below = s.copy(); below[a] = s[b] - eps
    above = s.copy(); above[a] = s[b] + eps
    return float(abs(field_fn(above)[a] - field_fn(below)[a]))


def closed_loop_circulation(field_fn, s0: np.ndarray, a: int, b: int,
                            half: float = 0.5, n_steps: int = 800) -> float:
    """Line integral oint lambda . ds around a rectangular loop in the (s_a, s_b) plane (other scores fixed
    at s0) that crosses the s_a = s_b swap boundary. For a conservative field (RankNet) the circulation is
    ~0; for the non-integrable LambdaRank field it is bounded away from 0. The loop is centered on the
    boundary point where s_a = s_b. (BUILD-AND-RUN gated — the airtight anchor is the discontinuity
    witness; this is the illustrative line integral.)"""
    c = 0.5 * (s0[a] + s0[b])
    corners = [(c - half, c - half), (c + half, c - half), (c + half, c + half), (c - half, c + half)]
    circ = 0.0
    for k in range(4):
        (xa, ya), (xb, yb) = corners[k], corners[(k + 1) % 4]
        for t in range(n_steps):
            f0 = (t + 0.5) / n_steps
            sa = xa + (xb - xa) * f0
            sb = ya + (yb - ya) * f0
            s = s0.copy(); s[a] = sa; s[b] = sb
            lam = field_fn(s)
            circ += lam[a] * (xb - xa) / n_steps + lam[b] * (yb - ya) / n_steps
    return float(circ)


def _toy_query() -> dict:
    """A minimal 3-document toy with DISTINCT grades {2, 1, 0} for the integrability stage. Doc 0 (grade
    2) and doc 1 (grade 1) are the swapping pair; doc 2 (grade 0) is the spectator whose pair with doc 0
    carries the discontinuous weight. Returns y_q, grades_q, and a base score vector well inside a cell."""
    y_q = np.array([2.0, 1.0, 0.0])
    grades_q = {0: 2, 1: 1, 2: 0}
    s_base = np.array([1.6, 1.0, 0.0])      # doc 0 above doc 1 above doc 2 (a clean cell interior)
    return {"y_q": y_q, "grades_q": grades_q, "s_base": s_base}


def _toy_fields(toy: dict):
    """Bind the two score-space fields to the toy's grades (closures field_fn(s))."""
    y_q, grades_q = toy["y_q"], toy["grades_q"]
    rn = lambda s: ranknet_field(s, y_q)
    lr = lambda s: lambdarank_field(s, y_q, grades_q)
    return rn, lr


# =========================================================================== #
# Movement 4 — listwise objectives: Plackett-Luce (ListMLE) and top-one (ListNet), both CONVEX.
# =========================================================================== #

def candidate_set(corpus: dict, q: int, n_neg: int = N_NEG) -> list[int]:
    """The per-query candidate set for the listwise losses: the K graded docs plus the n_neg HARDEST
    grade-0 negatives (highest oracle score among the non-relevant). Running Plackett-Luce over the full
    110-doc grade-0 tail is meaningless and tail-dominated; restricting to a small candidate set keeps the
    likelihood informative. Deterministic."""
    grades_q = corpus["grades"][q]
    rel = sorted(grades_q.keys())
    oracle = corpus["oracle_scores"][q]
    negs = [int(d) for d in np.argsort(-oracle) if d not in grades_q][:n_neg]
    return rel + negs


def optimal_permutation(corpus: dict, q: int, docs: list[int]) -> list[int]:
    """The ideal permutation pi* of a candidate set: grade DESCENDING, ties broken by oracle score
    descending, then doc id — a deterministic TOTAL order (resolving the grade-tie non-uniqueness)."""
    grades_q = corpus["grades"][q]
    oracle = corpus["oracle_scores"][q]
    return sorted(docs, key=lambda d: (-grades_q.get(d, 0), -float(oracle[d]), d))


def plackett_luce_logprob(scores: np.ndarray, perm_local: list[int]) -> float:
    """log P(pi | s) under the Plackett-Luce model: sum_r [ s_{pi(r)} - logsumexp(s over the remaining
    suffix) ]. `perm_local` indexes into `scores`. The MLE objective whose negative is ListMLE."""
    total = 0.0
    for r in range(len(perm_local)):
        suffix = scores[perm_local[r:]]
        total += scores[perm_local[r]] - logsumexp(suffix)
    return float(total)


def listmle_loss_scores(scores: np.ndarray, perm_local: list[int]) -> float:
    """ListMLE as a function of SCORES: -log P(pi* | s) = sum_r [ logsumexp(suffix) - s_{pi(r)} ]. Each
    term is convex (logsumexp) minus linear, so the loss is convex in s (a 1-direction null space along a
    global shift)."""
    return -plackett_luce_logprob(scores, perm_local)


def listmle_loss(w: np.ndarray, feat_cand: np.ndarray, perm_local: list[int]) -> float:
    """ListMLE for a linear scorer s = feat_cand @ w over one query's candidate set."""
    return listmle_loss_scores(feat_cand @ w, perm_local)


def listmle_grad(w: np.ndarray, feat_cand: np.ndarray, perm_local: list[int]) -> np.ndarray:
    """Closed-form ListMLE gradient. d/ds: at stage r the suffix softmax p_r contributes +p_r and the
    chosen item -1; accumulate per doc, then chain through s = feat_cand @ w (grad_w = feat^T dL/ds)."""
    s = feat_cand @ w
    n = s.shape[0]
    dL_ds = np.zeros(n)
    for r in range(len(perm_local)):
        suffix = perm_local[r:]
        p = softmax(s[suffix])
        dL_ds[suffix] += p
        dL_ds[perm_local[r]] -= 1.0
    return feat_cand.T @ dL_ds


def listmle_hessian(w: np.ndarray, feat_cand: np.ndarray, perm_local: list[int]) -> np.ndarray:
    """ListMLE Hessian in w: sum_r feat_suffix^T (diag(p_r) - p_r p_r^T) feat_suffix, a sum of softmax
    covariances (PSD). The algebraic convexity witness."""
    s = feat_cand @ w
    H = np.zeros((w.shape[0], w.shape[0]))
    for r in range(len(perm_local)):
        suffix = perm_local[r:]
        p = softmax(s[suffix])
        Xs = feat_cand[suffix]
        cov = np.diag(p) - np.outer(p, p)
        H += Xs.T @ cov @ Xs
    return H


def listnet_top1_target(corpus: dict, q: int, docs: list[int], gain=GAIN) -> np.ndarray:
    """The ListNet top-one target distribution softmax(gain(grade)) over the candidate set — the
    probability each doc is ranked first, tied to the metric's gain convention (2^g - 1)."""
    grades_q = corpus["grades"][q]
    g = np.array([gain(grades_q.get(d, 0)) for d in docs], dtype=float)
    return softmax(g)


def listnet_loss(w: np.ndarray, feat_cand: np.ndarray, target_p: np.ndarray) -> float:
    """ListNet top-one cross-entropy: -sum_i p_i log softmax(s)_i = -sum_i p_i (s_i - logsumexp(s)).
    Convex in s (linear + logsumexp)."""
    s = feat_cand @ w
    return float(-np.sum(target_p * (s - logsumexp(s))))


def listnet_grad(w: np.ndarray, feat_cand: np.ndarray, target_p: np.ndarray) -> np.ndarray:
    """Closed-form ListNet gradient: feat^T (softmax(s) - p)."""
    s = feat_cand @ w
    return feat_cand.T @ (softmax(s) - target_p)


def listnet_hessian(w: np.ndarray, feat_cand: np.ndarray, target_p: np.ndarray) -> np.ndarray:
    """ListNet Hessian: feat^T (diag(softmax) - softmax softmax^T) feat (PSD; target-independent)."""
    s = feat_cand @ w
    p = softmax(s)
    return feat_cand.T @ (np.diag(p) - np.outer(p, p)) @ feat_cand


def _listwise_blocks(corpus: dict, qs: list[int]) -> list[tuple]:
    """Per-query (feat_cand, perm_local, target_p) blocks for the listwise fitters (candidate features in
    a fixed order; pi* and the top-one target expressed as local indices)."""
    blocks = []
    for q in qs:
        docs = candidate_set(corpus, q)
        feat_cand = corpus["feats"][q][docs]
        pos = {d: i for i, d in enumerate(docs)}
        perm_local = [pos[d] for d in optimal_permutation(corpus, q, docs)]
        target_p = listnet_top1_target(corpus, q, docs)
        blocks.append((feat_cand, perm_local, target_p))
    return blocks


def fit_listmle(corpus: dict, qs: list[int], x0: np.ndarray | None = None) -> np.ndarray:
    """Fit a linear ListMLE scorer (convex; L-BFGS-B with the analytic gradient over all train candidate
    sets). Deterministic from x0 = zeros."""
    blocks = _listwise_blocks(corpus, qs)

    def loss(w):
        return sum(listmle_loss(w, fc, pl) for fc, pl, _ in blocks)

    def grad(w):
        return sum(listmle_grad(w, fc, pl) for fc, pl, _ in blocks)

    x0 = np.zeros(3) if x0 is None else x0
    return minimize(loss, x0, jac=grad, method="L-BFGS-B").x


def fit_listnet(corpus: dict, qs: list[int], x0: np.ndarray | None = None) -> np.ndarray:
    """Fit a linear ListNet (top-one) scorer (convex; L-BFGS-B with the analytic gradient)."""
    blocks = _listwise_blocks(corpus, qs)

    def loss(w):
        return sum(listnet_loss(w, fc, tp) for fc, _, tp in blocks)

    def grad(w):
        return sum(listnet_grad(w, fc, tp) for fc, _, tp in blocks)

    x0 = np.zeros(3) if x0 is None else x0
    return minimize(loss, x0, jac=grad, method="L-BFGS-B").x


# =========================================================================== #
# Movement 5 — LambdaMART: boosting the lambda field into regression trees.
# =========================================================================== #

def lambdamart_fit(corpus: dict, qs: list[int], n_rounds: int = N_ROUNDS, nu: float = NU,
                   max_depth: int = TREE_DEPTH, gain=GAIN, discount=DISCOUNT) -> list:
    """Gradient-boost the per-document LambdaRank lambda into a forest. Each round: re-score every train
    doc with the current ensemble, compute the LambdaRank force lambda_i, fit a
    DecisionTreeRegressor(max_depth, random_state=0) to the pseudo-residual -lambda_i (the ascent
    direction; lambda = dL/ds, so -lambda raises NDCG), and append nu * tree. Deterministic (seeded
    trees, no SGD). Returns the list of fitted trees."""
    scores = {q: np.zeros(corpus["n_docs"]) for q in qs}
    feats = {q: corpus["feats"][q] for q in qs}
    trees = []
    for _ in range(n_rounds):
        X_all, target = [], []
        for q in qs:
            lam = lambdarank_field(scores[q], corpus["y"][q], corpus["grades"][q], gain, discount)
            X_all.append(feats[q])
            target.append(-lam)
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        tree.fit(np.vstack(X_all), np.concatenate(target))
        trees.append(tree)
        for q in qs:
            scores[q] = scores[q] + nu * tree.predict(feats[q])
    return trees


def lambdamart_scores(trees: list, feat_q: np.ndarray, nu: float = NU) -> np.ndarray:
    """The ensemble score sum_t nu * tree_t.predict(X) for one query's docs. An empty ensemble -> zeros
    (the 0-rounds == base anchor)."""
    if not trees:
        return np.zeros(feat_q.shape[0])
    return nu * np.sum([t.predict(feat_q) for t in trees], axis=0)


def lambdamart_ranking(trees: list, corpus: dict, q: int, nu: float = NU) -> list[int]:
    """The LambdaMART ranking for query q (stable descending argsort of the ensemble scores)."""
    return np.argsort(-lambdamart_scores(trees, corpus["feats"][q], nu), kind="stable").tolist()


def _toy_xor() -> dict:
    """A constructed query where relevance is an XOR interaction of two features no linear scorer can
    order. Four archetypes, two docs each: (+,+) and (-,-) are relevant (grade 3); (+,-) and (-,+) are
    not (grade 0). A linear scorer w.(a,b) must rank both relevant above both irrelevant, i.e.
    min(w1+w2, -w1-w2) > max(w1-w2, -w1+w2), i.e. -|w1+w2| > |w1-w2| — impossible. A depth-2 tree splits
    on a then b and realizes the partition exactly. A third (constant) feature pads to the 3-feature
    shape. Returns feats (n,3), y, grades."""
    base = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0],
                     [1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]])
    feats = np.hstack([base, np.zeros((8, 1))])          # third feature constant (linear-useless)
    grades = {0: 3, 1: 3, 2: 3, 3: 3}                    # (+,+) and (-,-) docs are relevant
    y = np.array([3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    return {"feats": feats, "y": y, "grades": grades}


def _toy_corpus(toy: dict) -> dict:
    """Wrap an 8-doc toy as a one-query corpus so the shared fitters / metrics apply."""
    return {"feats": [toy["feats"]], "y": [toy["y"]], "grades": [toy["grades"]],
            "n_docs": toy["feats"].shape[0], "n_queries": 1}


def constructed_nonlinear_toy() -> dict:
    """Build the XOR query, fit the best linear LambdaRank scorer and a LambdaMART forest on it, and
    return their NDCG. THE assertion: ndcg_lambdamart > ndcg_best_linear (the trees escape the rank
    ceiling no linear scorer can). The construction is seed-free; the fit is seeded (random_state=0)."""
    toy = _toy_xor()
    c = _toy_corpus(toy)
    grades, n = toy["grades"], toy["feats"].shape[0]
    w_lin = fit_lambdarank(c, [0], n_iters=200, lr=0.5)
    rank_lin = np.argsort(-(toy["feats"] @ w_lin), kind="stable").tolist()
    ndcg_lin = ndcg_at_k(rank_lin, grades, n, GAIN, DISCOUNT)
    trees = lambdamart_fit(c, [0], n_rounds=20, nu=0.3, max_depth=2)
    rank_lm = lambdamart_ranking(trees, c, 0, nu=0.3)
    ndcg_lm = ndcg_at_k(rank_lm, grades, n, GAIN, DISCOUNT)
    return {"ndcg_lambdamart": float(ndcg_lm), "ndcg_best_linear": float(ndcg_lin),
            "w_best_linear": w_lin.tolist()}


def lambdamart_mean_ndcg(trees: list, corpus: dict, qs: list[int], nu: float = NU,
                         gain=GAIN, discount=DISCOUNT) -> float:
    """Mean NDCG@TOPK of a LambdaMART forest over a query set (held-out evaluation)."""
    return float(np.mean([ndcg_at_k(lambdamart_ranking(trees, corpus, q, nu), corpus["grades"][q],
                                    TOPK, gain, discount) for q in qs]))


def lambdamart_mean_recall(trees: list, corpus: dict, qs: list[int], nu: float = NU,
                           k: int = TOPK) -> float:
    """Mean recall@k of a LambdaMART forest over a query set."""
    ts = corpus["truth_sets"]
    return float(np.mean([recall_at_k(lambdamart_ranking(trees, corpus, q, nu), ts[q], k) for q in qs]))


def boost_curve(corpus: dict, checkpoints=(0, 1, 2, 5, 10, 15, 20, 25, 30)) -> list[tuple]:
    """Test NDCG@TOPK of the LambdaMART forest at increasing round counts (the boosting climb-and-plateau
    curve). Fits to the max checkpoint once, snapshotting at each checkpoint."""
    tr, te = corpus["train_q"], corpus["test_q"]
    cps = sorted(set(checkpoints))
    curve = []
    if cps[0] == 0:
        curve.append((0, lambdamart_mean_ndcg([], corpus, te)))
    for r in [c for c in cps if c > 0]:
        trees = lambdamart_fit(corpus, tr, n_rounds=r)
        curve.append((r, round(lambdamart_mean_ndcg(trees, corpus, te), 4)))
    return curve


# =========================================================================== #
# Movement 6 — the method comparison (held-out NDCG / recall for all five, CI on the headline).
# =========================================================================== #

def method_comparison(corpus: dict) -> dict:
    """Fit RankNet, LambdaRank, ListNet, ListMLE, and LambdaMART on the train queries; evaluate each on
    the HELD-OUT test queries by NDCG@TOPK and recall@TOPK, alongside the RRF and best-leg references. The
    learned-method headline carries a per-query NDCG estimator CI (metric_summary). The verdict is PINNED
    to the run — no pre-baked winner."""
    tr, te = corpus["train_q"], corpus["test_q"]
    w_rn = fit_ranknet(all_train_pairs(corpus, tr))
    w_lr = fit_lambdarank(corpus, tr)
    w_ln = fit_listnet(corpus, tr)
    w_lm_trees = lambdamart_fit(corpus, tr)
    w_lmle = fit_listmle(corpus, tr)

    def per_query_ndcg(rank_fn):
        return np.array([ndcg_at_k(rank_fn(q), corpus["grades"][q], TOPK, GAIN, DISCOUNT) for q in te])

    linear = {"ranknet": w_rn, "lambdarank": w_lr, "listnet": w_ln, "listmle": w_lmle}
    test = {}
    ci = {}
    for name, w in linear.items():
        pq = per_query_ndcg(lambda q, w=w: ranking_from_w(corpus, q, w))
        test[name] = {"ndcg": float(pq.mean()),
                      "recall": mean_recall_over(corpus, te, w)}
        ci[name] = metric_summary(pq)
    pq_lm = per_query_ndcg(lambda q: lambdamart_ranking(w_lm_trees, corpus, q))
    test["lambdamart"] = {"ndcg": float(pq_lm.mean()),
                          "recall": lambdamart_mean_recall(w_lm_trees, corpus, te)}
    ci["lambdamart"] = metric_summary(pq_lm)
    test["rrf"] = {"ndcg": mean_rrf_metric(corpus, te, "ndcg"),
                   "recall": mean_rrf_metric(corpus, te, "recall")}
    best_leg = max(LEG_ORDER, key=lambda leg: mean_leg_metric(corpus, te, leg, "ndcg"))
    test["best_leg"] = {"ndcg": mean_leg_metric(corpus, te, best_leg, "ndcg"),
                        "recall": mean_leg_metric(corpus, te, best_leg, "recall")}
    winner = max((m for m in test), key=lambda m: test[m]["ndcg"])
    return {"test": test, "ci": ci, "winner": winner, "best_leg": best_leg,
            "w_lambdarank": w_lr.tolist(), "w_ranknet": w_rn.tolist()}


def pick_worked_query(corpus: dict, w_lr: np.ndarray | None = None) -> int:
    """A test query with a grade-3 doc for Panel A (the LambdaRank/RankNet top-concentration contrast).
    Falls back to the median-NDCG test query."""
    tr, te = corpus["train_q"], corpus["test_q"]
    if w_lr is None:
        w_lr = fit_lambdarank(corpus, tr)
    cands = [q for q in te if 3 in corpus["grades"][q].values()]
    pool = cands if cands else te
    scored = sorted(pool, key=lambda q: ndcg_at_k(ranking_from_w(corpus, q, w_lr),
                                                  corpus["grades"][q], TOPK, GAIN, DISCOUNT))
    return scored[len(scored) // 2]


# =========================================================================== #
# Tests — every pedagogical claim is an assert (the notebook owns the numbers).
# =========================================================================== #

def test_delta_ndcg_closed_form_matches_brute() -> None:
    """The Delta-NDCG closed form == the physical-swap NDCG difference (untruncated), over many
    (query, p, q) including pairs that straddle the top-k cutoff."""
    c = _corpus()
    w = fit_lambdarank(c, c["train_q"])
    for q in c["test_q"][:4]:
        ranking = ranking_from_w(c, q, w)
        grades_q = c["grades"][q]
        for (p, qq) in [(1, 2), (2, 3), (1, 10), (9, 11), (5, 50), (3, 100)]:
            cf = delta_ndcg_swap(grades_q, ranking, p, qq)
            br = delta_ndcg_brute(grades_q, ranking, p, qq)
            assert abs(cf - br) < 1e-12, (q, p, qq, cf, br)


def test_lambdarank_collapses_to_ranknet() -> None:
    """Uniform-weight LambdaRank == the imported lambda_forces bit-for-bit (the collapse anchor)."""
    c = _corpus()
    w = fit_ranknet(all_train_pairs(c, c["train_q"]))
    for q in c["train_q"][:5]:
        mine = lambdarank_forces(w, c["feats"][q], c["y"][q], c["grades"][q], uniform=True)
        ref = lambda_forces(w, c["feats"][q], c["y"][q])
        assert np.allclose(mine, ref, atol=1e-12), q


def test_lambdarank_field_equals_lambda_forces() -> None:
    """The score-space ranknet_field equals lambda_forces when s = feat_q @ w (the lift to R^n)."""
    c = _corpus()
    w = fit_ranknet(all_train_pairs(c, c["train_q"]))
    q = c["train_q"][0]
    s = c["feats"][q] @ w
    assert np.allclose(ranknet_field(s, c["y"][q]), lambda_forces(w, c["feats"][q], c["y"][q]), atol=1e-12)


def test_weight_strictly_decreasing_in_rank() -> None:
    """The seed-free structural claim: for an equal grade gap the |Delta-NDCG| weight is strictly larger
    for a head swap (ranks 1-2) than a tail swap (ranks 9-10)."""
    d = weight_decreasing_in_rank()
    assert d["strictly_decreasing"], d
    assert d["head_1_2"] > 10 * d["tail_9_10"], d        # an order of magnitude, not a hair


def test_gradient_top_concentrated() -> None:
    """LambdaRank concentrates gradient mass at the head: its top-3 share of the per-pair force exceeds
    RankNet's (uniform weight) on the worked query."""
    c = _corpus()
    w = fit_lambdarank(c, c["train_q"])
    q = pick_worked_query(c, w)
    gc = gradient_concentration_by_rank(c, q, w)
    assert gc["lambdarank_top3_share"] > gc["ranknet_top3_share"], gc


def test_within_cell_jacobian_symmetric_both() -> None:
    """Within a no-swap cell BOTH fields have a symmetric Jacobian — LambdaRank's weights are constant
    there, so it is LOCALLY the gradient of a weighted-RankNet loss (the subtlety the non-integrability
    is GLOBAL)."""
    toy = _toy_query()
    rn, lr = _toy_fields(toy)
    s = toy["s_base"]
    Jr = numerical_jacobian(rn, s)
    Jl = numerical_jacobian(lr, s)
    assert np.allclose(Jr, Jr.T, atol=1e-6), Jr
    assert np.allclose(Jl, Jl.T, atol=1e-6), Jl


def test_lambdarank_field_discontinuous_across_swap() -> None:
    """The non-integrability witness: across the s_a = s_b swap, the LambdaRank force on doc a JUMPS by a
    fixed amount (a spectator pair's weight is discontinuous), while the RankNet force is continuous. The
    clean separation is the eps-scaling: RankNet's across-boundary difference is O(eps) (the smooth drift
    of a continuous field -> shrinks with eps), LambdaRank's is Theta(1) (eps-stable -> a true jump). A
    discontinuous field is no C^1 gradient — LambdaRank optimizes no scalar loss."""
    toy = _toy_query()
    rn, lr = _toy_fields(toy)
    s = toy["s_base"]
    rn_coarse = swap_discontinuity_witness(rn, s, 0, 1, eps=1e-4)
    rn_fine = swap_discontinuity_witness(rn, s, 0, 1, eps=1e-5)
    assert rn_fine < 0.2 * rn_coarse, (rn_coarse, rn_fine)        # O(eps): ~10x smaller -> continuous
    lr_coarse = swap_discontinuity_witness(lr, s, 0, 1, eps=1e-4)
    lr_fine = swap_discontinuity_witness(lr, s, 0, 1, eps=1e-5)
    assert lr_coarse > 1e-2 and lr_fine > 1e-2, (lr_coarse, lr_fine)
    assert abs(lr_coarse - lr_fine) < 1e-3, (lr_coarse, lr_fine)  # eps-stable -> a genuine discontinuity


def test_field_circulation() -> None:
    """The line integral oint lambda . ds around a swap-crossing loop is ~0 for the conservative RankNet
    field (path-independent — it IS a gradient) and bounded away from 0 for the non-integrable LambdaRank
    field (the classical signature of a non-conservative vector field)."""
    toy = _toy_query()
    rn, lr = _toy_fields(toy)
    s = toy["s_base"]
    assert abs(closed_loop_circulation(rn, s, 0, 1)) < 1e-3
    assert abs(closed_loop_circulation(lr, s, 0, 1)) > 1e-2


def test_listmle_grad_fd() -> None:
    """ListMLE analytic gradient matches finite differences (<1e-4)."""
    c = _corpus()
    q = c["train_q"][0]
    docs = candidate_set(c, q)
    feat_cand = c["feats"][q][docs]
    pos = {d: i for i, d in enumerate(docs)}
    perm_local = [pos[d] for d in optimal_permutation(c, q, docs)]
    err = check_grad(listmle_loss, listmle_grad, np.array([0.3, -0.2, 0.5]), feat_cand, perm_local)
    assert err < 1e-4, err


def test_listnet_grad_fd() -> None:
    """ListNet analytic gradient matches finite differences (<1e-4)."""
    c = _corpus()
    q = c["train_q"][0]
    docs = candidate_set(c, q)
    feat_cand = c["feats"][q][docs]
    target_p = listnet_top1_target(c, q, docs)
    err = check_grad(listnet_loss, listnet_grad, np.array([0.3, -0.2, 0.5]), feat_cand, target_p)
    assert err < 1e-4, err


def test_listwise_convex_unique() -> None:
    """Both listwise losses are convex (PSD Hessian, eigenvalue >= -1e-9 with the global-shift null space)
    and have a single optimum: two starts converge to the same w*."""
    c = _corpus()
    q = c["train_q"][0]
    docs = candidate_set(c, q)
    feat_cand = c["feats"][q][docs]
    pos = {d: i for i, d in enumerate(docs)}
    perm_local = [pos[d] for d in optimal_permutation(c, q, docs)]
    target_p = listnet_top1_target(c, q, docs)
    wm = np.array([0.2, -0.3, 0.4])
    assert np.linalg.eigvalsh(listmle_hessian(wm, feat_cand, perm_local)).min() >= -1e-9
    assert np.linalg.eigvalsh(listnet_hessian(wm, feat_cand, target_p)).min() >= -1e-9
    tr = c["train_q"]
    a = fit_listmle(c, tr, x0=np.zeros(3))
    b = fit_listmle(c, tr, x0=np.random.default_rng(1).standard_normal(3))
    assert np.allclose(a, b, atol=1e-4), (a, b)
    an = fit_listnet(c, tr, x0=np.zeros(3))
    bn = fit_listnet(c, tr, x0=np.random.default_rng(2).standard_normal(3))
    assert np.allclose(an, bn, atol=1e-4), (an, bn)


def test_listmle_zero_in_limit() -> None:
    """ListMLE -> 0 as a perfectly-ordered score is scaled up (a LIMIT — the loss is convex but not zero
    at any finite fit)."""
    m = 6
    perm_local = list(range(m))
    ideal = np.arange(m, 0, -1, dtype=float)             # strictly descending in pi* order
    assert listmle_loss_scores(20.0 * ideal, perm_local) < 1e-3
    assert listmle_loss_scores(1.0 * ideal, perm_local) > 1e-1   # not zero at a finite scale


def test_optimal_permutation_grade_desc() -> None:
    """pi* is grade-descending with an oracle tiebreak; its Plackett-Luce logprob at the oracle scores
    beats a swapped permutation."""
    c = _corpus()
    q = c["train_q"][0]
    docs = candidate_set(c, q)
    perm = optimal_permutation(c, q, docs)
    grades_q = c["grades"][q]
    gs = [grades_q.get(d, 0) for d in perm]
    assert gs == sorted(gs, reverse=True), gs
    pos = {d: i for i, d in enumerate(docs)}
    pl = [pos[d] for d in perm]
    scores = np.array([float(c["oracle_scores"][q][d]) for d in docs])
    swapped = pl.copy(); swapped[0], swapped[-1] = swapped[-1], swapped[0]
    assert plackett_luce_logprob(scores, pl) >= plackett_luce_logprob(scores, swapped)


def test_lambdamart_zero_rounds_equals_base() -> None:
    """An empty LambdaMART ensemble scores all-zero, so its ranking is the stable base order — the
    0-rounds == base anchor."""
    c = _corpus()
    q = c["test_q"][0]
    assert np.allclose(lambdamart_scores([], c["feats"][q]), 0.0)
    assert lambdamart_ranking([], c, q) == list(range(c["n_docs"]))


def test_lambdamart_beats_linear_on_xor() -> None:
    """The constructed XOR instance: LambdaMART (depth-2 trees) strictly beats the best linear scorer's
    NDCG (which is rank-capped below 1)."""
    d = constructed_nonlinear_toy()
    assert d["ndcg_best_linear"] < 1.0 - 1e-6, d
    assert d["ndcg_lambdamart"] > d["ndcg_best_linear"] + 1e-6, d


def test_method_comparison_runs() -> None:
    """All five learned methods evaluate on the held-out queries and beat the best single leg by NDCG
    (pinned to the run — no pre-baked winner)."""
    c = _corpus()
    res = method_comparison(c)
    for m in ("ranknet", "lambdarank", "listnet", "listmle", "lambdamart"):
        assert res["test"][m]["ndcg"] >= res["test"]["best_leg"]["ndcg"] - 1e-9, (m, res["test"])
    assert res["winner"] in res["test"]


def _run_tests() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")


# =========================================================================== #
# viz_constants — every number LambdaRankLaboratory.tsx mirrors (reproducible only; cast scalars).
# =========================================================================== #

def viz_constants() -> None:
    """Print every value the viz bakes. Reproducible only (deterministic solves, seeded trees); numpy
    scalars cast to float/int so the mirrored values are clean."""
    c = _corpus()
    tr, te = c["train_q"], c["test_q"]
    r4 = lambda v: round(float(v), 4)

    print("=== dims / split ===")
    print("N_DOCS", c["n_docs"], "N_QUERIES", c["n_queries"], "TOPK", TOPK, "CAND", CAND)
    print("TRAIN_Q", tr)
    print("TEST_Q", te)

    print("=== Panel A: RankNet vs LambdaRank lambda forces / top-concentration ===")
    w_lr = fit_lambdarank(c, tr)
    q = pick_worked_query(c, w_lr)
    print("WORKED_Q", q)
    order = ranking_from_w(c, q, w_lr)
    print("WORKED_RANKING_TOPK", order[:TOPK])
    print("WORKED_GRADES_TOPK", [int(c["grades"][q].get(d, 0)) for d in order[:TOPK]])
    lam_rn = ranknet_field(c["feats"][q] @ w_lr, c["y"][q])
    lam_lr = lambdarank_field(c["feats"][q] @ w_lr, c["y"][q], c["grades"][q])
    print("RANKNET_LAMBDA_TOPK", [r4(lam_rn[d]) for d in order[:TOPK]])
    print("LAMBDARANK_LAMBDA_TOPK", [r4(lam_lr[d]) for d in order[:TOPK]])
    gc = gradient_concentration_by_rank(c, q, w_lr)
    print("RANKNET_MASS_BY_RANK", [r4(v) for v in gc["ranknet_mass"]])
    print("LAMBDARANK_MASS_BY_RANK", [r4(v) for v in gc["lambdarank_mass"]])
    print("TOP3_SHARE ranknet,lambdarank", r4(gc["ranknet_top3_share"]), r4(gc["lambdarank_top3_share"]))
    wd = weight_decreasing_in_rank()
    print("WEIGHT_HEAD_1_2, TAIL_9_10", r4(wd["head_1_2"]), r4(wd["tail_9_10"]))

    print("=== Panel B: integrability (3-doc toy) ===")
    toy = _toy_query()
    rn, lr = _toy_fields(toy)
    s = toy["s_base"]
    print("TOY_GRADES", toy["grades_q"], "TOY_S_BASE", [r4(v) for v in s])
    print("SWAP_JUMP_RANKNET", round(float(swap_discontinuity_witness(rn, s, 0, 1, eps=1e-5)), 6))
    print("SWAP_JUMP_LAMBDARANK", r4(swap_discontinuity_witness(lr, s, 0, 1, eps=1e-5)))
    print("CIRC_RANKNET", r4(closed_loop_circulation(rn, s, 0, 1)))
    print("CIRC_LAMBDARANK", r4(closed_loop_circulation(lr, s, 0, 1)))
    Jr = numerical_jacobian(rn, s)
    Jl = numerical_jacobian(lr, s)
    print("JAC_RANKNET_OFFDIAG_ASYM", r4(np.max(np.abs(Jr - Jr.T))))
    print("JAC_LAMBDARANK_OFFDIAG_ASYM", r4(np.max(np.abs(Jl - Jl.T))))

    print("=== Panel C: listwise (candidate set, pi*, convex bowl) ===")
    docs = candidate_set(c, q)
    perm = optimal_permutation(c, q, docs)
    print("CAND_DOCS", docs)
    print("OPT_PERM", perm)
    feat_cand = c["feats"][q][docs]
    pos = {d: i for i, d in enumerate(docs)}
    perm_local = [pos[d] for d in perm]
    target_p = listnet_top1_target(c, q, docs)
    w_lmle = fit_listmle(c, tr)
    w_ln = fit_listnet(c, tr)
    print("W_LISTMLE", [r4(v) for v in w_lmle], "W_LISTNET", [r4(v) for v in w_ln])
    print("PL_LOGPROB_STAR", r4(plackett_luce_logprob(feat_cand @ w_lmle, perm_local)))
    print("LISTNET_TARGET_TOP5", [r4(v) for v in target_p[:5]])
    # a coarse 2-D loss bowl over (w_dense, w_li) at w_lex = optimum, for the convexity contour
    grid = np.linspace(-3.0, 3.0, 13)
    bowl_mle, bowl_net = [], []
    for a in grid:
        row_mle, row_net = [], []
        for b in grid:
            wv = np.array([w_lmle[0], a, b])
            row_mle.append(round(listmle_loss(wv, feat_cand, perm_local), 3))
            wn = np.array([w_ln[0], a, b])
            row_net.append(round(listnet_loss(wn, feat_cand, target_p), 3))
        bowl_mle.append(row_mle); bowl_net.append(row_net)
    print("BOWL_GRID", [r4(v) for v in grid])
    print("LISTMLE_BOWL", bowl_mle)
    print("LISTNET_BOWL", bowl_net)

    print("=== Panel D: LambdaMART boosting + method comparison ===")
    print("BOOST_CURVE", boost_curve(c))
    cx = constructed_nonlinear_toy()
    print("CONSTRUCTED_LAMBDAMART ndcg_lm,ndcg_lin",
          r4(cx["ndcg_lambdamart"]), r4(cx["ndcg_best_linear"]))
    res = method_comparison(c)
    for m in ("ranknet", "lambdarank", "listnet", "listmle", "lambdamart", "rrf", "best_leg"):
        v = res["test"][m]
        print(f"  {m:12s} ndcg={r4(v['ndcg'])}  recall={r4(v['recall'])}")
    print("WINNER", res["winner"], "BEST_LEG", res["best_leg"])
    hci = res["ci"][res["winner"]] if res["winner"] in res["ci"] else res["ci"]["lambdarank"]
    print("HEADLINE_CI n,mean,se,lo,hi",
          hci["n"], r4(hci["mean"]), r4(hci["se"]), r4(hci["ci_lo"]), r4(hci["ci_hi"]))
    print("W_LAMBDARANK", [r4(v) for v in res["w_lambdarank"]],
          "W_RANKNET", [r4(v) for v in res["w_ranknet"]])


if __name__ == "__main__":
    print("Running lambdarank-lambdamart-listwise tests...")
    _run_tests()
    print()
    viz_constants()
    print("\nAll lambdarank-lambdamart-listwise checks passed.")
