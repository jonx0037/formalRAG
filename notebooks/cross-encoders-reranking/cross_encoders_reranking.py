"""Cross-encoders and the reranking cascade — the reference implementation for the formalRAG
`cross-encoders-reranking` topic.

The dual-encoder topic kept invoking the cross-encoder as the rank-free, un-precomputable
counterpoint; this module SPENDS that power. A cross-encoder scores a FUSED pair s(q, d) = h([q; d]),
abandoning the separable inner product that made dual retrieval precomputable. Three movements, every
pedagogical claim an `assert`:

  MOVEMENT 1 — JOINT ENCODING BREAKS THE RANK CEILING, AND THE NONLINEARITY IS WHY. A dual encoder's
    score matrix S = Q G^T has rank <= d (the prereq's Eckart-Young ceiling). The SUBTLE, load-bearing
    fact: a LEARNED BILINEAR scorer q^T W d does NOT escape it — S = Q W G^T = (Q W) G^T is just a dual
    encoder with reparametrized queries, still rank <= d. Only a genuinely NONLINEAR joint scorer
    escapes. We demonstrate it on the signed-identity target imported from the embedding-dimension
    topic (full rank n): the best rank-d dual encoder and the best learned bilinear both plateau at a
    strictly positive reconstruction error for d < n, while a random-ReLU-feature cross-encoder on the
    one-hot pair [e_i; e_j] reconstructs it to machine precision at every d.
    (`random_relu_features`, `fit_cross_encoder_ridge`, `cross_encoder_score`, `bilinear_score`,
    `fit_bilinear`, `recon_error_curve`.)

  MOVEMENT 2 — THE RETRIEVE-THEN-RERANK CASCADE AND THE RECALL PINCH. The cross-encoder is linear in
    what it scores, so it cannot score the whole corpus; it reranks a cheap first stage's top-K. Cost
    is c_ret + K c_ce against the brute |C| c_ce. Under a known-item qrel (one gold doc per query — the
    finance toy), an ORACLE rerank of the top-K makes recall@1 equal the candidate pool's recall@K: it
    LIFTS recall@1 from the first stage's recall@1 up to its recall@K and CAPS it there. The cascade
    can never recover a gold the first stage dropped. (`stage1_ranking`, `candidate_pool`,
    `oracle_rerank_recall`, `cascade_recall_vs_k`, cost via the imported `over_fetch_factor`.)

  MOVEMENT 3 — MONOTONICITY, AND WHEN RERANKING HURTS. An oracle rerank is recall-monotone in K (a
    larger top-K is a superset). A LOSSY cross-encoder can DIP recall@1 BELOW the first stage by being
    confidently wrong, and — the sharpest contrast — MORE over-fetch makes the dip WORSE. We model the
    lossy reranker two ways: oracle scores plus seeded Gaussian corruption, and a constructed
    confident-wrong reranker that deterministically promotes a same-sector distractor.
    (`lossy_rerank_recall`, `confident_wrong_rerank_recall`, the dip witness.)

  CLOSING REMARK (not a movement) — expressivity is paid for in samples: the cross-encoder's capacity
    overfits where the dual encoder's inner-product inductive bias generalizes. Stated, up-linked to
    learning theory, not simulated here.

Honest caveats (rigorFlag territory): the "cross-encoder" is a deterministic random-ReLU-feature
scorer with a closed-form ridge head, a seeded stand-in for h([q; d]), NOT a trained transformer;
universal approximation is asymptotic (width -> infinity), so the rank-ceiling escape is DEMONSTRATED,
not proved. The recall pinch is an exact identity only under a known-item (|R| = 1) qrel; with several
relevant documents it weakens to an inequality. Monotonicity in K is a theorem for an EXACT (oracle)
reranker only; the lossy reranker is a model of an imperfect cross-encoder, not a trained model's
error distribution. The cascade's accuracy gain is measured on the synthetic vMF finance cloud reused
from the InfoNCE / DPR topics, restricted to a rank-3 first stage so it is imperfect-but-recoverable.

Imports its prerequisite (`dense-retrieval-dual-encoders`) and several import-only numeric siblings
(`embedding-dimension-lower-bounds` for the signed-identity target, `set-metrics-...` for recall@k,
`capstone-...` for the over-fetch law); it never reimplements them. `viz_constants()` prints what
`CrossEncoderRerankingLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/cross-encoders-reranking/cross_encoders_reranking.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# Established cross-topic pattern: add EACH ancestor's HYPHENATED dir to the path, import the
# UNDERSCORED module. The prerequisite is dense-retrieval-dual-encoders (the single DAG edge); it
# transitively needs the three DPR ancestors. set-metrics / multi-vector / capstone / embedding-
# dimension are import-only numeric siblings (connections, not prerequisites — the import graph is
# NOT the pedagogical DAG), reused so we never reimplement a recall denominator or an over-fetch law.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",                 # normalize, sample_vmf (DPR ancestor)
    "the-retrieval-problem",                    # dot, rank (DPR ancestor)
    "infonce-contrastive-objective",            # finance vMF geometry (DPR ancestor)
    "dense-retrieval-dual-encoders",            # THE prerequisite
    "embedding-dimension-lower-bounds",         # signed_identity — the M1 high-rank target
    "set-metrics-precision-recall-map-mrr",     # recall_at_k / precision_at_k (list-based)
    "capstone-multimodal-financial-rag",        # cascade_recall / over_fetch_factor
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dense_retrieval_dual_encoders import (                                 # noqa: E402
    dpr_finance_matrix, cross_encoder_oracle, score_matrix, best_rank_d,
    realize_dual_encoder, relative_frobenius_error, topk_recall,
)
from embedding_dimension_lower_bounds import signed_identity               # noqa: E402
from set_metrics_precision_recall_map_mrr import recall_at_k               # noqa: E402
from capstone_multimodal_financial_rag import over_fetch_factor            # noqa: E402


# =========================================================================== #
# Movement 1 — the rank ceiling, the bilinear foil, and the nonlinear escape.
# =========================================================================== #

def bilinear_score(Q: np.ndarray, W: np.ndarray, P: np.ndarray) -> np.ndarray:
    """The LEARNED-BILINEAR score matrix S = Q W P^T (query embeddings Q in R^{nq x d}, learned
    interaction W in R^{d x d}, document embeddings P in R^{np x d}). Despite the learned W, this is a
    product through a d-dimensional bottleneck: S = (Q W) P^T, a dual encoder with reparametrized
    queries, so rank(S) <= d. Learning a metric is NOT fusing the pair."""
    return np.atleast_2d(Q) @ np.asarray(W, dtype=float) @ np.atleast_2d(P).T


def fit_bilinear(Q: np.ndarray, P: np.ndarray, M: np.ndarray, lam: float = 1e-8) -> np.ndarray:
    """The best learned interaction W minimizing ||Q W P^T - M||_F (ridge-regularized least squares
    via the vectorized normal equations: vec(Q W P^T) = (P kron Q) vec(W)). The resulting S = Q W P^T
    is still rank <= d, so this is the strongest a frozen-embedding bilinear can do — and it cannot
    beat the rank-d ceiling. GUARD: lam > 0 keeps the normal equations well-posed."""
    Q = np.atleast_2d(Q)
    P = np.atleast_2d(P)
    d = Q.shape[1]
    A = np.kron(P, Q)                                   # (nq*np) x (d*d): rows = vec of e_q e_d^T blocks
    y = M.reshape(-1, order="F")                        # column-major vec to match np.kron layout
    w = np.linalg.solve(A.T @ A + lam * np.eye(d * d), A.T @ y)
    return w.reshape(d, d, order="F")


def random_relu_features(X: np.ndarray, n_feat: int, seed: int, gamma: float = 1.0) -> np.ndarray:
    """A random nonlinear lift of stacked pairs X (n_pairs x input_dim) into
    phi(X) = relu(X @ R + b), with R ~ N(0, gamma / input_dim) and b ~ U[-1, 1], drawn from ONE
    seeded rng stream (the Monte-Carlo one-stream rule, so the feature map is bit-reproducible). The
    ReLU is the nonlinearity that lifts the realizable score-matrix rank above any inner dimension."""
    X = np.atleast_2d(X)
    in_dim = X.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((in_dim, n_feat)) * np.sqrt(gamma / in_dim)
    b = rng.uniform(-1.0, 1.0, size=n_feat)
    return np.maximum(0.0, X @ R + b)


def fit_cross_encoder_ridge(X: np.ndarray, y: np.ndarray, n_feat: int, seed: int,
                            gamma: float = 1.0, lam: float = 1e-6) -> dict:
    """Fit the random-ReLU cross-encoder: closed-form ridge readout w = (Phi^T Phi + lam I)^-1 Phi^T y
    on the random features Phi = phi(X). Deterministic given (n_feat, seed, gamma, lam). Returns the
    fitted scorer {R, b, w, in_dim, n_feat, seed, gamma} consumed by `cross_encoder_score`. This is a
    finite, seeded stand-in for a trained joint scorer h([q; d]); the closed-form head keeps every
    baked number bit-reproducible."""
    X = np.atleast_2d(X)
    in_dim = X.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((in_dim, n_feat)) * np.sqrt(gamma / in_dim)
    b = rng.uniform(-1.0, 1.0, size=n_feat)
    Phi = np.maximum(0.0, X @ R + b)
    w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(n_feat), Phi.T @ np.asarray(y, dtype=float))
    return {"R": R, "b": b, "w": w, "in_dim": in_dim, "n_feat": n_feat, "seed": seed, "gamma": gamma}


def cross_encoder_score(scorer: dict, X: np.ndarray) -> np.ndarray:
    """The cross-encoder's score for each stacked pair X (n_pairs x in_dim): phi(X) @ w. Each pair is
    scored JOINTLY through the shared nonlinear features — NOT a factorization Q G^T — so the realized
    score matrix's rank is not capped by any embedding dimension."""
    X = np.atleast_2d(X)
    Phi = np.maximum(0.0, X @ scorer["R"] + scorer["b"])
    return Phi @ scorer["w"]


def onehot_pairs(n: int):
    """All n^2 one-hot id pairs [e_i; e_j] in R^{2n} (row-major over (i, j)) and the (i, j) index
    arrays. The cross-encoder reads each (query id, doc id) pair jointly; the dual/bilinear encoders
    must instead embed the ids into d dimensions and pay the rank ceiling."""
    eye = np.eye(n)
    rows, ii, jj = [], [], []
    for i in range(n):
        for j in range(n):
            rows.append(np.concatenate([eye[i], eye[j]]))
            ii.append(i)
            jj.append(j)
    return np.array(rows), np.array(ii), np.array(jj)


def reconstruct_cross_encoder_matrix(M: np.ndarray, n_feat: int, seed: int,
                                     gamma: float = 1.0, lam: float = 1e-6) -> np.ndarray:
    """Fit the random-ReLU cross-encoder on the one-hot pairs of an n x n target M and return its
    realized score matrix. With enough features it interpolates M to machine precision — the rank-free
    counterpoint to the dual encoder, demonstrated on a concrete target."""
    n = M.shape[0]
    X, ii, jj = onehot_pairs(n)
    y = M[ii, jj]
    scorer = fit_cross_encoder_ridge(X, y, n_feat, seed, gamma, lam)
    return cross_encoder_score(scorer, X).reshape(n, n)


def best_bilinear_ceiling(M: np.ndarray, d: int) -> np.ndarray:
    """The best a learned bilinear q^T W d can do at inner dimension d: with the OPTIMAL embeddings
    (the SVD's singular vectors) and W = diag(top-d singular values), Q W P^T = U_d Sigma_d V_d^T =
    best_rank_d(M, d). So the best bilinear IS the truncated SVD — identical to the best dual encoder,
    and no learned interaction beats it. Returns that matrix (asserted == best_rank_d)."""
    M = np.asarray(M, dtype=float)
    d = max(1, min(int(d), min(M.shape)))
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return bilinear_score(U[:, :d], np.diag(s[:d]), Vt[:d].T)


def recon_error_curve(M: np.ndarray, dims, n_feat: int, seed: int,
                      gamma: float = 1.0, lam: float = 1e-10):
    """For each inner dimension d, the relative-Frobenius reconstruction error of the rank-d CEILING
    (the best rank-d dual encoder = the best learned bilinear = truncated SVD, all identical) and the
    d-INDEPENDENT error of the nonlinear cross-encoder. Returns a list of {d, ceiling, cross}. The
    ceiling plateaus at a positive error for d < rank(M); the cross-encoder is flat near zero."""
    M = np.asarray(M, dtype=float)
    cross_err = relative_frobenius_error(M, reconstruct_cross_encoder_matrix(M, n_feat, seed, gamma, lam))
    rows = []
    for d in dims:
        d = int(d)
        ceiling = relative_frobenius_error(M, best_rank_d(M, d))
        rows.append({"d": d, "ceiling": round(ceiling, 6), "cross": round(cross_err, 8)})
    return rows


def stack_pairs(Q: np.ndarray, P: np.ndarray):
    """Every (query, document) embedding pair stacked as [q; d] in R^{2*dim}, row-major over
    (query, doc). Returns (X, ii, jj) — the cross-encoder's joint input on the finance corpus, the
    real-embedding analogue of the one-hot pairs of Movement 1."""
    Q, P = np.atleast_2d(Q), np.atleast_2d(P)
    rows, ii, jj = [], [], []
    for i in range(Q.shape[0]):
        for j in range(P.shape[0]):
            rows.append(np.concatenate([Q[i], P[j]]))
            ii.append(i)
            jj.append(j)
    return np.array(rows), np.array(ii), np.array(jj)


def cross_encoder_finance_scores(Q: np.ndarray, P: np.ndarray, truth: np.ndarray,
                                 n_feat: int = 256, seed: int = 11) -> np.ndarray:
    """Fit the random-ReLU cross-encoder on the finance [q; d] pairs (relevance target 1 on the gold
    pair, 0 elsewhere) and return its (nq x np) score matrix. The genuine joint scorer h([q; d]) on
    the finance geometry — it realizes the relevance exactly (no rank ceiling), separating the
    same-sector hard negatives the dual encoder confuses. (Fit on the full relevance: the train/test
    coincidence is the rigorFlag's expressivity-not-generalization caveat.)"""
    X, ii, jj = stack_pairs(Q, P)
    y = (np.asarray(truth)[ii] == jj).astype(float)
    scorer = fit_cross_encoder_ridge(X, y, n_feat, seed, gamma=1.0, lam=1e-8)
    return cross_encoder_score(scorer, X).reshape(Q.shape[0], P.shape[0])


# =========================================================================== #
# Movement 2 / 3 — the cascade: first-stage retrieval, candidate pool, rerank.
# =========================================================================== #

def stage1_ranking(S1: np.ndarray) -> np.ndarray:
    """The first stage's per-query ranking: argsort each row of the first-stage score matrix S1
    (descending, stable). Returns an (nq x np) array of document indices, best first."""
    S1 = np.atleast_2d(S1)
    return np.argsort(-S1, axis=1, kind="stable")


def candidate_pool(order: np.ndarray, K: int) -> np.ndarray:
    """The top-K candidate pool of each query (the first K columns of the stage-1 ranking). K is
    capped at the corpus size. The reranker only ever sees — and can only ever surface — these."""
    K = max(1, min(int(K), order.shape[1]))
    return order[:, :K]


def oracle_rerank_recall(order: np.ndarray, truth: np.ndarray, K: int) -> float:
    """recall@1 after an ORACLE rerank of the top-K: a perfect scorer surfaces the gold document iff
    it is in the pool, so recall@1_after = fraction of queries whose gold survived stage 1 = the
    first stage's recall@K. The exact recall pinch (known-item qrel)."""
    pool = candidate_pool(order, K)
    truth = np.asarray(truth)
    hits = sum(1 for i in range(pool.shape[0]) if truth[i] in pool[i])
    return hits / pool.shape[0]


def stage1_recall_at_k(order: np.ndarray, truth: np.ndarray, K: int) -> float:
    """The first stage's recall@K: the fraction of queries whose gold document is in the top-K. Equal
    to `oracle_rerank_recall` by construction (the pinch); kept separate so the identity is asserted,
    not assumed. Cross-checks against the imported list-based set-metrics `recall_at_k` per query."""
    pool = candidate_pool(order, K)
    truth = np.asarray(truth)
    hits = sum(1 for i in range(pool.shape[0]) if truth[i] in pool[i])
    return hits / pool.shape[0]


def rerank_by_scores(order: np.ndarray, rerank_scores: np.ndarray, truth: np.ndarray, K: int) -> float:
    """recall@1 after reranking each query's top-K pool by a supplied (nq x np) score matrix: within
    the pool, take the argmax of the rerank scores and check it is the gold. The single engine behind
    the oracle, lossy, and confident-wrong rerankers — they differ only in the scores passed in."""
    pool = candidate_pool(order, K)
    truth = np.asarray(truth)
    hits = 0
    for i in range(pool.shape[0]):
        cand = pool[i]
        best = cand[int(np.argmax(rerank_scores[i, cand]))]
        hits += int(best == truth[i])
    return hits / pool.shape[0]


def lossy_scores(S_oracle: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """A lossy cross-encoder's scores: the oracle-quality score matrix plus seeded Gaussian
    corruption N(0, sigma^2). sigma = 0 recovers the oracle exactly. A model of an imperfect reranker
    (NOT a trained model's error), the controllable knob behind the confident-wrong dip."""
    rng = np.random.default_rng(seed)
    return np.asarray(S_oracle, dtype=float) + sigma * rng.standard_normal(S_oracle.shape)


def confident_wrong_scores(S_oracle: np.ndarray, truth: np.ndarray, sector_of_passage: np.ndarray,
                           boost: float) -> np.ndarray:
    """A constructed confident-wrong reranker: take the oracle scores but deterministically PROMOTE,
    for each query, one same-sector distractor (the nearest non-gold document of the same sector) by
    `boost`. A guaranteed dip witness — the reranker is confidently wrong on exactly the hard
    negatives the first stage already struggles with."""
    S = np.array(S_oracle, dtype=float)
    truth = np.asarray(truth)
    sector_of_passage = np.asarray(sector_of_passage)
    for i in range(S.shape[0]):
        gold = truth[i]
        same = [j for j in range(S.shape[1]) if j != gold
                and sector_of_passage[j] == sector_of_passage[gold]]
        if same:
            # the same-sector distractor the oracle already ranks highest (the most tempting one)
            distractor = same[int(np.argmax(S[i, same]))]
            S[i, distractor] = S[i, gold] + boost
    return S


def cascade_recall_vs_k(order: np.ndarray, truth: np.ndarray, S_oracle: np.ndarray,
                        sigma: float, lossy_seed: int, k_grid) -> list:
    """For each over-fetch depth K: the first stage's recall@1 (flat), its recall@K, the oracle-rerank
    recall@1 (== recall@K), and a lossy-rerank recall@1. The recall-vs-K frontier the laboratory
    plots."""
    r1_stage1 = stage1_recall_at_k(order, truth, 1)
    L = lossy_scores(S_oracle, sigma, lossy_seed)
    rows = []
    for K in k_grid:
        rows.append({
            "K": int(K),
            "stage1_r1": round(r1_stage1, 4),
            "stage1_rk": round(stage1_recall_at_k(order, truth, K), 4),
            "oracle": round(oracle_rerank_recall(order, truth, K), 4),
            "lossy": round(rerank_by_scores(order, L, truth, K), 4),
        })
    return rows


def rerank_buckets(order: np.ndarray, truth: np.ndarray, rerank_scores: np.ndarray, K: int) -> dict:
    """Per-query outcome buckets of a rerank against the first stage at depth K: `fixed` (stage-1
    wrong, reranker right — the hard-negative win), `kept` (both right), `broke` (stage-1 right,
    reranker wrong — the confident-wrong dip), `missed` (both wrong, or gold not in pool). The net
    lift is (#fixed - #broke) / nq, which a lossy reranker can drive NEGATIVE."""
    pool = candidate_pool(order, K)
    truth = np.asarray(truth)
    counts = {"fixed": 0, "kept": 0, "broke": 0, "missed": 0}
    for i in range(pool.shape[0]):
        cand = pool[i]
        s1_top = int(order[i, 0])
        re_top = cand[int(np.argmax(rerank_scores[i, cand]))]
        s1_ok = s1_top == truth[i]
        re_ok = re_top == truth[i]
        if not s1_ok and re_ok:
            counts["fixed"] += 1
        elif s1_ok and re_ok:
            counts["kept"] += 1
        elif s1_ok and not re_ok:
            counts["broke"] += 1
        else:
            counts["missed"] += 1
    counts["net_lift"] = round((counts["fixed"] - counts["broke"]) / pool.shape[0], 4)
    return counts


# =========================================================================== #
# Module constants the viz panels step through (tuned by _diagnostics()).
# =========================================================================== #

# Movement 1 — the signed-identity expressivity target and the cross-encoder surrogate.
SIGN_N = 6                       # signed_identity(6): full rank 6, the same target as the prereq's
                                 #   test_cross_encoder_has_no_rank_constraint (a 6x6 full-rank matrix)
CE_N_FEAT = 256                  # random-ReLU features (enough to interpolate a 6x6 target)
CE_SEED = 11                     # ONE rng stream for the feature map (Monte-Carlo rule)
CE_GAMMA = 1.0
CE_LAM = 1e-10                   # tiny ridge: the readout interpolates the target (the rank-free limit)
M1_DIMS = tuple(range(1, SIGN_N + 1))

# Movement 2 / 3 — the finance cascade (reuse the DPR corpus; restrict the first stage).
D_STAGE1 = 3                     # first-stage dual-encoder rank (tuned: recall@1 imperfect, recall@K=1)
K_GRID = (1, 2, 3, 4, 6, 8)      # over-fetch depths for the recall-vs-K frontier
LOSSY_SIGMA = 0.25               # a decent-but-imperfect reranker for the Panel B frontier line
SIGMA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)   # Panel C reranker-quality sweep (0 = oracle)
LOSSY_SEED = 23
CW_BOOST = 1.0                   # confident-wrong promotion margin (guaranteed dip)
DIP_K = 8                        # the depth at which the dip witness / buckets are read

# The cost model — a headline corpus so the K-rerank speedup is legible (the toy has only 8 docs).
C_RETRIEVE = 1.0                 # first-stage MIPS unit cost
C_CE = 25.0                      # per-pair cross-encoder cost (>> retrieval — the whole tension)
CORPUS_HEADLINE = 1_000_000      # passages the brute cross-encoder would have to score


# =========================================================================== #
# Cached setup — the finance cascade instance, built once.
# =========================================================================== #

_CACHE: dict = {}


def _setup() -> dict:
    """Build (and cache) the finance cascade instance: the DPR corpus, the full-rank score matrix
    (the oracle scorer, recall@1 = 1), and the restricted rank-D_STAGE1 first stage with its ranking.
    Everything downstream reads this one instance."""
    if _CACHE:
        return _CACHE
    Q, P, truth, sector = dpr_finance_matrix()
    S_full = score_matrix(Q, P)                          # full-dim dual encoder: the oracle (recall@1=1)
    S1 = best_rank_d(S_full, D_STAGE1)                   # the restricted, imperfect first stage
    order = stage1_ranking(S1)
    _CACHE.update({"Q": Q, "P": P, "truth": truth, "sector": sector,
                   "S_full": S_full, "S1": S1, "order": order})
    return _CACHE


# =========================================================================== #
# Diagnostics — build-and-run separations to read BEFORE trusting any headline.
# =========================================================================== #

def _diagnostics() -> None:
    """Print the separations that set D_STAGE1, LOSSY_SIGMA, CW_BOOST, CE_N_FEAT. Read these before
    pinning any assert (the CLAUDE.md build-and-run rule — several 'obvious' claims here are false)."""
    s = _setup()
    truth, sector = s["truth"], s["sector"]
    S_full, order = s["S_full"], s["order"]
    nq, npass = S_full.shape

    print("  [diag] corpus:", nq, "queries,", npass, "documents; intrinsic rank",
          int(np.linalg.matrix_rank(S_full)))

    print("  [diag] first-stage recall@1 / recall@K by restriction dim d (want interior d: r1 in"
          " (0.3,0.85) AND recall@K=1 at small K):")
    for d in range(1, npass + 1):
        S1d = best_rank_d(S_full, d)
        od = stage1_ranking(S1d)
        rks = [round(stage1_recall_at_k(od, truth, K), 3) for K in K_GRID]
        print(f"         d={d}: r@1={stage1_recall_at_k(od, truth, 1):.3f}  r@K{list(K_GRID)}={rks}")

    print(f"  [diag] chosen D_STAGE1={D_STAGE1}: stage-1 r@1={stage1_recall_at_k(order, truth, 1):.3f}"
          "  oracle-rerank r@1 by K:")
    for K in K_GRID:
        print(f"         K={K}: oracle={oracle_rerank_recall(order, truth, K):.3f}"
              f"  == recall@K={stage1_recall_at_k(order, truth, K):.3f}")

    print("  [diag] lossy rerank recall@1 by (sigma, K) — want a dip below stage-1 r@1, worse at"
          " larger K:")
    for sigma in (0.25, 0.5, 1.0, 2.0):
        row = []
        L = lossy_scores(S_full, sigma, LOSSY_SEED)
        for K in (3, 5, 8):
            row.append((K, round(rerank_by_scores(order, L, truth, K), 3)))
        print(f"         sigma={sigma}: {row}")

    cw = confident_wrong_scores(S_full, truth, sector, CW_BOOST)
    print(f"  [diag] confident-wrong rerank r@1 at K={DIP_K}:"
          f" {rerank_by_scores(order, cw, truth, DIP_K):.3f} (vs stage-1"
          f" {stage1_recall_at_k(order, truth, 1):.3f}); buckets="
          f"{rerank_buckets(order, truth, cw, DIP_K)}")

    M = signed_identity(SIGN_N)
    ce = reconstruct_cross_encoder_matrix(M, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    print(f"  [diag] M1 signed_identity({SIGN_N}): rank={int(np.linalg.matrix_rank(M))},"
          f" cross-encoder recon err={relative_frobenius_error(M, ce):.2e},"
          f" cross-encoder matrix rank={int(np.linalg.matrix_rank(np.round(ce, 6)))}")
    curve = recon_error_curve(M, M1_DIMS, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    print("  [diag] M1 recon-error curve (rank-d ceiling plateaus, cross ~0):")
    for r in curve:
        print(f"         d={r['d']}: ceiling={r['ceiling']:.4f} cross={r['cross']:.2e}")
    Sce = cross_encoder_finance_scores(s["Q"], s["P"], truth)
    print(f"  [diag] finance cross-encoder recall@1={topk_recall(Sce, truth, 1):.3f}"
          f" (vs rank-{D_STAGE1} stage {stage1_recall_at_k(order, truth, 1):.3f})")


# =========================================================================== #
# Verification harness — each assert is a pedagogical claim the topic makes.
# =========================================================================== #

def test_dual_encoder_rank_capped_by_dim() -> None:
    """Movement 1: a dual encoder's score matrix S = Q G^T has rank <= d (reuses the prereq's
    `score_matrix`; the ceiling the cross-encoder breaks)."""
    rng = np.random.default_rng(0)
    for d in (1, 2, 4, 7):
        Q = rng.standard_normal((20, d))
        G = rng.standard_normal((12, d))
        assert np.linalg.matrix_rank(score_matrix(Q, G)) <= d, f"rank(Q G^T) exceeded d={d}"
    print("  [ok] Movement 1: dual encoder score matrix has rank <= d (the prereq ceiling)")


def test_bilinear_stays_rank_d() -> None:
    """Movement 1, the hinge: a LEARNED bilinear q^T W d does NOT escape the ceiling — S = Q W P^T =
    (Q W) P^T is a product through a d-bottleneck, so rank(S) <= d for ARBITRARY W; and at W = I it is
    byte-for-byte the prereq's dual-encoder score (the collapse anchor)."""
    rng = np.random.default_rng(1)
    for d in (1, 2, 4, 6):
        Q = rng.standard_normal((18, d))
        P = rng.standard_normal((10, d))
        for _ in range(3):
            W = rng.standard_normal((d, d))             # arbitrary, full-rank, non-symmetric
            assert np.linalg.matrix_rank(bilinear_score(Q, W, P)) <= d, \
                f"learned bilinear escaped the rank ceiling at d={d}"
        # W = I collapses to the dual encoder (the prereq's score_matrix), bit-for-bit.
        assert np.allclose(bilinear_score(Q, np.eye(d), P), score_matrix(Q, P), atol=1e-12), \
            "bilinear at W=I must equal the dual-encoder score"
    print("  [ok] Movement 1: a learned bilinear stays rank <= d (and W=I == the dual encoder)")


def test_bilinear_best_equals_dual_ceiling() -> None:
    """Movement 1: the best learned bilinear IS the truncated SVD — with the SVD's singular vectors as
    embeddings and W = diag(singular values), Q W P^T == best_rank_d(M, d) to machine precision. A
    learned interaction matrix cannot beat the rank-d ceiling; the dual and bilinear ceilings coincide
    exactly (so the laboratory draws ONE ceiling curve)."""
    M = signed_identity(SIGN_N)
    for d in range(1, SIGN_N + 1):
        assert np.allclose(best_bilinear_ceiling(M, d), best_rank_d(M, d), atol=1e-9), \
            f"best bilinear != truncated SVD at d={d}"
    print("  [ok] Movement 1: best learned bilinear == truncated SVD (dual/bilinear ceilings coincide)")


def test_cross_encoder_breaks_rank_ceiling() -> None:
    """Movement 1, the headline contrast: on the full-rank signed-identity target, the rank-d ceiling
    (best dual = best bilinear = truncated SVD) has strictly positive reconstruction error for d < n,
    while the nonlinear cross-encoder reconstructs it to near machine precision at every d. Assert the
    CONTRAST, never the decimals."""
    M = signed_identity(SIGN_N)
    assert int(np.linalg.matrix_rank(M)) == SIGN_N, "signed_identity should be full rank n"
    curve = recon_error_curve(M, M1_DIMS, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    for r in curve:
        if r["d"] < SIGN_N:
            assert r["ceiling"] > 1e-3, f"rank-d ceiling should be positive at d={r['d']}<n"
        assert r["cross"] < 1e-4, f"the cross-encoder should reconstruct it at any d (d={r['d']})"
    # the ceiling really binds (large gap) at the smallest d
    assert curve[0]["ceiling"] - curve[0]["cross"] > 0.5, "the rank gap should be visible at d=1"
    print(f"  [ok] Movement 1: cross-encoder breaks the rank ceiling — ceiling plateaus "
          f"(d<{SIGN_N}), cross ~0 at every d")


def test_cross_encoder_raises_matrix_rank() -> None:
    """Movement 1: the realized cross-encoder score matrix has rank GREATER than any small dual-
    encoder dimension — its expressivity is not a factorization. On signed_identity(n) it recovers
    full rank n."""
    M = signed_identity(SIGN_N)
    ce = reconstruct_cross_encoder_matrix(M, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    assert int(np.linalg.matrix_rank(np.round(ce, 6))) == SIGN_N, \
        "the cross-encoder's realized matrix should reach full rank n"
    print(f"  [ok] Movement 1: cross-encoder realizes a full-rank (rank {SIGN_N}) score matrix")


def test_cross_encoder_realizes_finance_relevance() -> None:
    """Movement 1 on finance: the genuine cross-encoder h([q; d]) realizes the finance relevance
    exactly (recall@1 = 1.0), separating the same-sector hard negatives the restricted rank-3 dual
    first stage confuses (recall@1 = 0.781). The expressivity, on the production geometry."""
    s = _setup()
    Q, P, truth, order = s["Q"], s["P"], s["truth"], s["order"]
    Sce = cross_encoder_finance_scores(Q, P, truth)
    ce_r1 = topk_recall(Sce, truth, 1)
    stage1 = stage1_recall_at_k(order, truth, 1)
    assert ce_r1 > stage1 + 1e-9, f"cross-encoder ({ce_r1}) should beat the rank-3 stage ({stage1})"
    assert abs(ce_r1 - 1.0) < 1e-12, f"cross-encoder should realize the finance relevance, got {ce_r1}"
    print(f"  [ok] Movement 1 (finance): cross-encoder recall@1 = {ce_r1:.3f} "
          f"realizes the relevance the rank-{D_STAGE1} stage ({stage1:.3f}) cannot")


def test_oracle_reproduces_prereq_no_rank_constraint() -> None:
    """Movement 1 collapse anchor: the imported `cross_encoder_oracle` realizes any target with no
    rank constraint while a dual encoder below the rank cannot — re-running the prereq's exact
    `test_cross_encoder_has_no_rank_constraint` logic via the IMPORTED oracle (never reimplemented)."""
    rng = np.random.default_rng(4)
    M = rng.standard_normal((6, 6))
    r = int(np.linalg.matrix_rank(M))
    assert r == 6, "test setup: target should be full rank 6"
    for d in range(1, r):
        Qd, Gd = realize_dual_encoder(M, d)
        assert relative_frobenius_error(M, Qd @ Gd.T) > 1e-6, \
            f"dual encoder should NOT realize a full-rank target at d={d}<{r}"
    assert relative_frobenius_error(M, cross_encoder_oracle(M)) < 1e-12, \
        "the imported cross-encoder oracle should realize any target exactly"
    print("  [ok] Movement 1 anchor: imported cross_encoder_oracle reproduces the prereq's "
          "no-rank-constraint contrast")


def test_degenerate_surrogate_matches_oracle() -> None:
    """Movement 1 twin anchor: with enough features the random-ReLU cross-encoder interpolates the
    target, matching the imported `cross_encoder_oracle` to near machine precision — the finite
    surrogate's degenerate (interpolating) limit IS the oracle."""
    M = signed_identity(SIGN_N)
    ce = reconstruct_cross_encoder_matrix(M, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    assert relative_frobenius_error(cross_encoder_oracle(M), ce) < 1e-6, \
        "interpolating cross-encoder should match the oracle"
    print("  [ok] Movement 1 twin: interpolating cross-encoder == cross_encoder_oracle")


def test_recall_pinch_identity() -> None:
    """Movement 2, the exact identity: an oracle rerank of the top-K makes recall@1 equal the first
    stage's recall@K (known-item qrel). Asserted to < 1e-12 at every K, and cross-checked against the
    imported list-based set-metrics `recall_at_k` per query."""
    s = _setup()
    order, truth = s["order"], s["truth"]
    for K in K_GRID:
        oracle = oracle_rerank_recall(order, truth, K)
        rk = stage1_recall_at_k(order, truth, K)
        assert abs(oracle - rk) < 1e-12, f"pinch broken at K={K}: oracle {oracle} != recall@K {rk}"
        # cross-check recall@K against the imported set-metrics (|R|=1, so recall is 0/1 per query)
        per_q = np.mean([recall_at_k(list(order[i]), {int(truth[i])}, K) for i in range(len(truth))])
        assert abs(per_q - rk) < 1e-12, f"set-metrics recall@K disagrees at K={K}"
    print("  [ok] Movement 2: recall pinch — oracle-rerank recall@1 == stage-1 recall@K (all K)")


def test_cascade_ceiling() -> None:
    """Movement 2 corollary: ANY reranker's recall@1 is <= the first stage's recall@K — it cannot
    recover a gold the first stage dropped from the pool. Checked for the oracle, lossy, and
    confident-wrong rerankers."""
    s = _setup()
    order, truth, S_full, sector = s["order"], s["truth"], s["S_full"], s["sector"]
    L = lossy_scores(S_full, LOSSY_SIGMA, LOSSY_SEED)
    CW = confident_wrong_scores(S_full, truth, sector, CW_BOOST)
    for K in K_GRID:
        ceil = stage1_recall_at_k(order, truth, K)
        for name, sc in (("oracle", S_full), ("lossy", L), ("confident_wrong", CW)):
            r1 = rerank_by_scores(order, sc, truth, K)
            assert r1 <= ceil + 1e-12, f"{name} exceeded the cascade ceiling at K={K}: {r1} > {ceil}"
    print("  [ok] Movement 2: every reranker's recall@1 <= stage-1 recall@K (the cascade ceiling)")


def test_cost_model() -> None:
    """Movement 2, the cost law: cascade cost = c_ret + K c_ce against brute |C| c_ce, and the
    over-fetch is the imported `over_fetch_factor` reciprocal of the first stage's recall@K."""
    s = _setup()
    order, truth = s["order"], s["truth"]
    for K in K_GRID:
        cascade = C_RETRIEVE + K * C_CE
        brute = CORPUS_HEADLINE * C_CE
        assert cascade < brute, f"cascade should be cheaper than brute at K={K}"
        r = stage1_recall_at_k(order, truth, K)
        if r > 0:
            assert abs(over_fetch_factor([r]) - 1.0 / r) < 1e-12, "over-fetch != 1/recall@K"
    print("  [ok] Movement 2: cost = c_ret + K c_ce << |C| c_ce; over-fetch = 1/recall@K (imported)")


def test_full_K_collapses_to_brute() -> None:
    """Movement 2 collapse anchor: at K = |C| (rerank the whole corpus) the oracle rerank equals the
    brute-force argmax over the oracle score matrix — both recall@1 = 1.0, and the surfaced document
    is the global argmax for every query."""
    s = _setup()
    order, truth, S_full = s["order"], s["truth"], s["S_full"]
    npass = S_full.shape[1]
    oracle_full = oracle_rerank_recall(order, truth, npass)
    brute = topk_recall(S_full, truth, 1)
    assert abs(oracle_full - 1.0) < 1e-12, f"rerank-everything recall@1 should be 1.0, got {oracle_full}"
    assert abs(brute - 1.0) < 1e-12, f"brute argmax recall@1 should be 1.0, got {brute}"
    # the surfaced doc is the global argmax (rerank everything by S_full == argmax over the corpus)
    for i in range(len(truth)):
        re_top = int(np.argmax(S_full[i, order[i, :npass]]))
        assert order[i, :npass][re_top] == int(np.argmax(S_full[i])), "rerank-all != global argmax"
    print("  [ok] Movement 2 anchor: rerank everything (K=|C|) == brute argmax (recall@1 = 1.0)")


def test_oracle_rerank_monotone_in_k() -> None:
    """Movement 3, the theorem (superset argument): oracle-rerank recall@1 is non-decreasing in K, a
    larger pool being a superset the oracle can only gain the gold from."""
    s = _setup()
    order, truth = s["order"], s["truth"]
    vals = [oracle_rerank_recall(order, truth, K) for K in K_GRID]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1)), \
        f"oracle rerank not monotone in K: {vals}"
    print(f"  [ok] Movement 3: oracle rerank is recall-monotone in K {[round(v, 3) for v in vals]}")


def test_lossy_rerank_can_dip() -> None:
    """Movement 3, the demonstrated proposition: a lossy / confident-wrong cross-encoder can DIP
    recall@1 BELOW the first stage, and more over-fetch makes the dip WORSE (the anti-monotone
    direction, the sharpest contrast to the oracle's monotonicity). The constructed confident-wrong
    reranker is the guaranteed witness."""
    s = _setup()
    order, truth, S_full, sector = s["order"], s["truth"], s["S_full"], s["sector"]
    stage1 = stage1_recall_at_k(order, truth, 1)
    CW = confident_wrong_scores(S_full, truth, sector, CW_BOOST)
    dip = rerank_by_scores(order, CW, truth, DIP_K)
    assert dip < stage1 - 1e-9, f"confident-wrong rerank should dip below stage-1 {stage1}, got {dip}"
    # more over-fetch is worse (or no better): the anti-monotone direction
    cw_small = rerank_by_scores(order, CW, truth, 2)
    cw_large = rerank_by_scores(order, CW, truth, DIP_K)
    assert cw_large <= cw_small + 1e-12, \
        f"larger K should not help a confident-wrong reranker: K=2 {cw_small} vs K={DIP_K} {cw_large}"
    print(f"  [ok] Movement 3: lossy rerank dips below stage-1 ({dip:.3f} < {stage1:.3f}); "
          f"more over-fetch is worse")


def test_lossy_vs_oracle_gap_nonvacuous() -> None:
    """Movement 3: the lossy reranker is strictly worse than the oracle at the dip depth (the dip
    story is not vacuous)."""
    s = _setup()
    order, truth, S_full, sector = s["order"], s["truth"], s["S_full"], s["sector"]
    oracle = oracle_rerank_recall(order, truth, DIP_K)
    CW = confident_wrong_scores(S_full, truth, sector, CW_BOOST)
    lossy = rerank_by_scores(order, CW, truth, DIP_K)
    assert oracle - lossy > 1e-3, f"oracle {oracle} and lossy {lossy} too close — vacuous"
    print(f"  [ok] Movement 3: oracle {oracle:.3f} strictly beats confident-wrong {lossy:.3f}")


def test_finance_first_stage_imperfect_but_recoverable() -> None:
    """The cascade sweet spot (build-and-run guard against the too-easy trap): the restricted first
    stage is imperfect at recall@1 but fully recoverable at some K (recall@K = 1), so the cascade has
    something to lift; the FULL-dimension dual encoder, by contrast, is already perfect (the trap)."""
    s = _setup()
    order, truth, S_full = s["order"], s["truth"], s["S_full"]
    r1 = stage1_recall_at_k(order, truth, 1)
    assert 0.3 < r1 < 0.95, f"first-stage recall@1 should be imperfect-but-decent, got {r1}"
    assert any(abs(stage1_recall_at_k(order, truth, K) - 1.0) < 1e-12 for K in K_GRID), \
        "first stage should be fully recoverable at some K (recall@K = 1)"
    # the too-easy trap: the full-dim dual encoder is already perfect (so we MUST restrict it)
    full_order = stage1_ranking(S_full)
    assert abs(stage1_recall_at_k(full_order, truth, 1) - 1.0) < 1e-12, \
        "full-dim dual encoder should be perfect (documents the too-easy trap)"
    print(f"  [ok] cascade window: stage-1 recall@1 = {r1:.3f} (imperfect), recoverable at some K; "
          f"full-dim is perfect (the trap)")


def test_viz_constants_reproducible() -> None:
    """The bake-only-reproducible guard: the random-ReLU surrogate + every baked curve is bit-
    identical across two builds (the one-rng-stream + closed-form-ridge payoff)."""
    M = signed_identity(SIGN_N)
    a = recon_error_curve(M, M1_DIMS, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    b = recon_error_curve(M, M1_DIMS, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    assert a == b, "recon-error curve is not reproducible across builds"
    s = _setup()
    c1 = cascade_recall_vs_k(s["order"], s["truth"], s["S_full"], LOSSY_SIGMA, LOSSY_SEED, K_GRID)
    c2 = cascade_recall_vs_k(s["order"], s["truth"], s["S_full"], LOSSY_SIGMA, LOSSY_SEED, K_GRID)
    assert c1 == c2, "cascade curve is not reproducible across builds"
    print("  [ok] reproducible: the surrogate and all baked curves are bit-identical across builds")


def test_guards() -> None:
    """Edge cases: K capped at the corpus size; K=0 floored to 1; an empty rerank pool is well-formed.
    (The gemini unguarded-denominator class.)"""
    s = _setup()
    order, truth = s["order"], s["truth"]
    npass = order.shape[1]
    assert candidate_pool(order, 999).shape[1] == npass, "K should cap at the corpus size"
    assert candidate_pool(order, 0).shape[1] == 1, "K=0 should floor to 1"
    assert oracle_rerank_recall(order, truth, 999) == oracle_rerank_recall(order, truth, npass), \
        "K beyond the corpus should equal K = |C|"
    print("  [ok] guards: K capped at |C|, K=0 floored to 1")


def _run_all() -> None:
    test_dual_encoder_rank_capped_by_dim()
    test_bilinear_stays_rank_d()
    test_bilinear_best_equals_dual_ceiling()
    test_cross_encoder_breaks_rank_ceiling()
    test_cross_encoder_raises_matrix_rank()
    test_cross_encoder_realizes_finance_relevance()
    test_oracle_reproduces_prereq_no_rank_constraint()
    test_degenerate_surrogate_matches_oracle()
    test_recall_pinch_identity()
    test_cascade_ceiling()
    test_cost_model()
    test_full_K_collapses_to_brute()
    test_oracle_rerank_monotone_in_k()
    test_lossy_rerank_can_dip()
    test_lossy_vs_oracle_gap_nonvacuous()
    test_finance_first_stage_imperfect_but_recoverable()
    test_viz_constants_reproducible()
    test_guards()


# =========================================================================== #
# Demo — the headline numbers, printed.
# =========================================================================== #

def cross_encoder_demo() -> dict:
    """The headlines: the cross-encoder breaks the rank ceiling a learned bilinear cannot; an oracle
    rerank pinches recall@1 to the pool's recall@K; a lossy reranker dips below the first stage."""
    out = {}
    M = signed_identity(SIGN_N)
    curve = recon_error_curve(M, M1_DIMS, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    print(f"  MOVEMENT 1: signed_identity({SIGN_N}) reconstruction error by dim d "
          f"(rank-d ceiling / cross-encoder):")
    for r in curve:
        print(f"             d={r['d']}: ceiling={r['ceiling']:.3f}  cross={r['cross']:.1e}")
    out["m1_curve"] = curve

    s = _setup()
    order, truth, S_full, sector = s["order"], s["truth"], s["S_full"], s["sector"]
    print(f"  MOVEMENT 2: first stage = rank-{D_STAGE1} dual encoder; recall@1 = "
          f"{stage1_recall_at_k(order, truth, 1):.3f}")
    rows = cascade_recall_vs_k(order, truth, S_full, LOSSY_SIGMA, LOSSY_SEED, K_GRID)
    for row in rows:
        print(f"             K={row['K']}: stage1 r@K={row['stage1_rk']:.3f}  "
              f"oracle rerank r@1={row['oracle']:.3f} (== recall@K)  lossy r@1={row['lossy']:.3f}")
    out["cascade"] = rows

    CW = confident_wrong_scores(S_full, truth, sector, CW_BOOST)
    buckets = rerank_buckets(order, truth, CW, DIP_K)
    print(f"  MOVEMENT 3: confident-wrong rerank at K={DIP_K}: buckets={buckets} "
          f"(net lift {buckets['net_lift']:+.3f}, below stage-1)")
    out["dip_buckets"] = buckets
    return out


# =========================================================================== #
# Viz constants — printed for CrossEncoderRerankingLaboratory.tsx to mirror.
# =========================================================================== #

def _r(v, n=4):
    return round(float(v), n)


def viz_constants() -> None:
    """Print every MEASURED number the laboratory mirrors to the decimal. TS recomputes only CLOSED
    FORM: the cost arithmetic c_ret + K c_ce, the over-fetch reciprocal, the rerank bucketing, and the
    net lift. Every measured number (the reconstruction curves, the recall-vs-K frontier, the dip
    witness) is baked here. numpy scalars are cast to avoid np.float64(...) pollution."""
    s = _setup()
    order, truth, S_full, sector = s["order"], s["truth"], s["S_full"], s["sector"]
    nq, npass = S_full.shape

    print("  // ----- shared constants -----")
    print(f"  const N_QUERIES = {int(nq)};")
    print(f"  const N_PASSAGES = {int(npass)};")
    print(f"  const D_STAGE1 = {D_STAGE1};")
    print(f"  const SIGN_N = {SIGN_N};")
    print(f"  const C_RETRIEVE = {_r(C_RETRIEVE, 2)};")
    print(f"  const C_CE = {_r(C_CE, 2)};")
    print(f"  const CORPUS_HEADLINE = {int(CORPUS_HEADLINE)};")
    print(f"  const K_GRID = {list(int(k) for k in K_GRID)};")

    print("  // ----- Panel A: the rank-ceiling reconstruction curves (signed_identity target) -----")
    M = signed_identity(SIGN_N)
    curve = recon_error_curve(M, M1_DIMS, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    print(f"  const RECON_CURVE = {curve};")
    print(f"  const CEILING_RANK = {SIGN_N};   // dual/bilinear hit 0 only at d = rank")
    sv = np.linalg.svd(M, compute_uv=False)
    print(f"  const SIGN_SINGULAR_VALUES = {[_r(v) for v in sv]};")
    ce = reconstruct_cross_encoder_matrix(M, CE_N_FEAT, CE_SEED, CE_GAMMA, CE_LAM)
    print(f"  const CROSS_MATRIX_RANK = {int(np.linalg.matrix_rank(np.round(ce, 6)))};")

    print("  // ----- Panel B: the cascade recall-vs-K frontier -----")
    rows = cascade_recall_vs_k(order, truth, S_full, LOSSY_SIGMA, LOSSY_SEED, K_GRID)
    print(f"  const CASCADE = {rows};")
    print(f"  const STAGE1_R1 = {_r(stage1_recall_at_k(order, truth, 1))};")

    print("  // ----- Panel C: the per-query rerank buckets by reranker quality (sigma sweep) -----")
    print(f"  const DIP_K = {DIP_K};")
    print(f"  const SIGMA_GRID = {[_r(g, 2) for g in SIGMA_GRID]};   // 0 = oracle")
    by_sigma = []
    for g in SIGMA_GRID:
        L = lossy_scores(S_full, g, LOSSY_SEED)
        b = rerank_buckets(order, truth, L, DIP_K)
        by_sigma.append({"sigma": _r(g, 2), **{k: int(b[k]) for k in
                        ("fixed", "kept", "broke", "missed")}, "net_lift": b["net_lift"]})
    print(f"  const BUCKETS_BY_SIGMA = {by_sigma};")
    CW = confident_wrong_scores(S_full, truth, sector, CW_BOOST)
    print(f"  const BUCKETS_CONFIDENT_WRONG = {rerank_buckets(order, truth, CW, DIP_K)};")
    # the dip witness: a query stage-1 got right that the confident-wrong reranker breaks
    pool = candidate_pool(order, DIP_K)
    witness = None
    for i in range(nq):
        s1_top = int(order[i, 0])
        re_top = int(pool[i][int(np.argmax(CW[i, pool[i]]))])
        if s1_top == int(truth[i]) and re_top != int(truth[i]):
            witness = {"query": int(i), "gold": int(truth[i]), "stage1_top": s1_top,
                       "reranked_top": re_top, "sector_gold": int(sector[truth[i]]),
                       "sector_wrong": int(sector[re_top])}
            break
    print(f"  const DIP_WITNESS = {witness};")


if __name__ == "__main__":
    print("Cross-encoders & the reranking cascade — diagnostics\n")
    _diagnostics()
    print("\nVerification harness:")
    _run_all()
    print("\nDemo:")
    cross_encoder_demo()
    print("\nViz constants (mirrored to the decimal in CrossEncoderRerankingLaboratory.tsx):")
    viz_constants()
    print("\nAll checks passed.")
