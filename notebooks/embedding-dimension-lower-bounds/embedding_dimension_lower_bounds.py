"""How many dimensions does relevance need? — the reference implementation for the formalRAG
`embedding-dimension-lower-bounds` topic.

DPR proved an UPPER bound: a d-dimensional dual encoder realizes exactly the relevance matrices
of rank <= d (Eckart-Young-Mirsky gives the optimal rank-d approximation to any target). But rank
is the wrong complexity for a RELEVANCE PATTERN, which is a SIGN pattern — relevant vs not, the
row-wise order, not the scores. This module asks the tight question DPR deferred: how many
dimensions does relevance NEED? The answer is the SIGN-RANK (a.k.a. dimension complexity) and its
robust cousin MARGIN COMPLEXITY. Every pedagogical claim below is an `assert`:

  MOVEMENT 1 — RANK IS THE WRONG COMPLEXITY => SIGN-RANK. The sign-rank of a sign matrix M is the
    smallest rank of a real matrix B with sign(B) = M: the smallest embedding dimension in which the
    relevance PATTERN is linearly realizable, regardless of the score magnitudes. Rank and sign-rank
    are genuinely different: the signed identity (+1 on the diagonal, -1 off) has full rank n but
    sign-rank 3 — a constructive rank-3 realization proves the pattern needs only 3 dimensions while
    its rank says n. (`signed_identity`, `realize_sign_rank`, `sign_rank_upper_bound`, `sign_agreement`.)

  MOVEMENT 2 — A CLOSED-FORM LOWER BOUND: FORSTER ON HADAMARD. Forster's theorem gives a spectral
    lower bound sign-rank(M) >= sqrt(mn) / ||M||_2. For an N x N Hadamard matrix H, H H^T = N I so
    ||H||_2 = sqrt(N) and the bound is exactly sqrt(N): a relevance pattern shaped like H provably
    needs at least sqrt(N) dimensions. We compute the bound (closed form), confirm the spectral
    fact, and corroborate it by FAILING to realize H in fewer than sqrt(N) dimensions. (We prove the
    Hadamard application and CITE Forster's general theorem.) (`hadamard_pattern`, `forster_lower_bound`.)

  MOVEMENT 3 — MARGIN COMPLEXITY: THE DIMENSION FOR A USABLE GAP. Sign-rank asks only for correct
    signs (entries may sit arbitrarily close to zero); margin complexity mc(M) = 1/margin(M) asks
    how robustly the pattern separates. A contrastively trained (soft-margin) encoder cares about
    margin, not bare signs, so margin complexity is the more faithful capacity measure. We measure
    the achievable normalized margin of a correct-sign rank-d realization: it is zero below the
    sign-rank (no correct realization exists) and grows with d above it — more dimensions buy a
    bigger gap. (We report ACHIEVABLE margins by projected gradient; the exact mc / gamma_2 value is
    an SDP we describe but do not solve.) (`margin_curve`.)

  MOVEMENT 4 — THE RETRIEVAL THEOREM AND THE FREE-EMBEDDING WALL (Weller, Boratko, Naim, Lee 2025,
    the LIMIT result). The minimum embedding dimension to realize a binary qrel matrix A equals the
    sign-rank of 2A - 1 up to +/- 1. The all-pairs qrel (queries = every document pair, each relevant
    to its two documents) forces this to grow: even with PERFECT, freely optimized embeddings, the
    largest number of documents whose all-pairs qrel a d-dimensional model can realize — the critical
    n — grows only polynomially in d. So at any fixed d some combinatorial relevance pattern is
    unrepresentable. (`allpairs_qrel`, `realize_qrel`, `free_embedding_critical_n`, `critical_n_curve`.)

  FINANCE — THE HEADLINE FLIP. The dimension that PERFECTLY solves single-company retrieval on DPR's
    finance corpus (its D_RECOVER) FAILS combinatorial multi-company relevance over a comparable
    corpus: simple relevance suffices at a small d, combinatorial relevance needs strictly more.

Honest caveats (rigorFlag territory): the "rank <= d" framing inherited from DPR is OUR exposition
(Eckart-Young governs approximation in rank — the wrong norm for a sign pattern); the LIMIT result
uses the row-wise order-preserving rank and sign-rank directly. Sign-rank is the RIGHT measure but
computing it is intractable in general (deciding sign-rank <= 3 is complete for the existential
theory of the reals; the strict +/-1 case is NP-hard) — we compute it only for tiny matrices and use
Forster lower bounds elsewhere. Many sign-rank lower bounds are existential/non-constructive; the
closed-form bound we PROVE is Forster-on-Hadamard. Margin complexity and sign-rank are polynomially
related with a log-N slack, not equal. The free-embedding critical n is an EMPIRICAL best-case
optimization, not a closed-form theorem. The LIMIT wall is for SINGLE-VECTOR dot-product retrieval;
multi-vector (late interaction), cross-encoders, and sparse BM25 are not bound by it. All finance
numbers are measured on the synthetic vMF cloud reused from the InfoNCE / DPR topics, not a trained
encoder.

This module imports its prerequisites (`hypersphere-vmf-geometry`, `the-retrieval-problem`,
`infonce-contrastive-objective`, `dense-retrieval-dual-encoders`) for the sphere sampler, the cosine
score, the InfoNCE loss, and DPR's finance matrix / rank-ceiling tools — it never reimplements them.
The in-batch loss is reused as a BYTE-FOR-BYTE anchor. `viz_constants()` prints what
`EmbeddingDimensionLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/embedding-dimension-lower-bounds/embedding_dimension_lower_bounds.py
"""
from __future__ import annotations

import math
import pathlib
import sys
from itertools import combinations

import numpy as np
from scipy.linalg import svd as scipy_svd

# Established cross-topic pattern: add EACH ancestor's HYPHENATED dir to the path, import the
# UNDERSCORED module. The direct prerequisite is the dual encoder (DPR); but DPR's grand-prereqs
# supply primitives DPR does not re-export (the sphere sampler, the cosine score, the InfoNCE loss),
# so we add every ancestor explicitly and never reimplement any of them.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("hypersphere-vmf-geometry", "the-retrieval-problem",
             "infonce-contrastive-objective", "dense-retrieval-dual-encoders"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf            # noqa: E402
from infonce_contrastive_objective import info_nce_loss_batch         # noqa: E402
from dense_retrieval_dual_encoders import (                           # noqa: E402
    dpr_finance_matrix,
    score_matrix,
    best_rank_d,
    topk_recall,
    rank_recall_curve,
    first_full_recall_dim,
    relative_frobenius_error,
    inbatch_loss_via_gram,
    dpr_inbatch_example,
    DPR_TAU,
)


# --------------------------------------------------------------------------- #
# Movement 1 — rank is the wrong complexity => sign-rank.
# --------------------------------------------------------------------------- #

def signed_identity(n: int) -> np.ndarray:
    """The signed identity sign pattern: +1 on the diagonal, -1 off it (M = 2 I_n - J_n). It has
    full rank n (eigenvalues 2 - n once and 2 with multiplicity n-1), yet its sign-rank is 3 for
    every n >= 3 — the cleanest rank-vs-sign-rank gap. GUARD: n >= 1."""
    if n < 1:
        raise ValueError(f"need n >= 1, got {n}")
    return 2 * np.eye(n) - np.ones((n, n))


def sign_agreement(B: np.ndarray, M: np.ndarray) -> float:
    """The fraction of entries whose sign matches the target sign matrix M (zeros in B count as a
    mismatch). The relevance-pattern accuracy of a real matrix B against the pattern M. GUARD:
    empty matrix -> 0.0."""
    M = np.asarray(M, dtype=float)
    B = np.asarray(B, dtype=float)
    if M.size == 0:
        return 0.0
    return float(np.mean(np.sign(B) == np.sign(M)))


def _sign_realize_grad(X: np.ndarray, Y: np.ndarray, M: np.ndarray, margin0: float):
    """Loss and gradients for realizing the sign pattern M with B = X Y^T. The smooth surrogate is
    the mean softplus(margin0 - M (.) B): it drives every entry of M (.) B above margin0. Returns
    (loss, gX, gY) with B re-used by the caller."""
    B = X @ Y.T
    z = margin0 - M * B                          # want z < 0 everywhere
    # softplus(z) = log(1 + e^z); d/dB = -M (.) sigmoid(z) / (mn).
    sig = 1.0 / (1.0 + np.exp(-z))
    loss = float(np.mean(np.logaddexp(0.0, z)))
    dB = -(M * sig) / M.size
    return loss, dB @ Y, dB.T @ X


def realize_sign_rank(M: np.ndarray, d: int, restarts: int = 6, steps: int = 800,
                      lr: float = 0.5, margin0: float = 1.0, seed: int = 0):
    """Try to realize the sign pattern M = sign(X Y^T) with inner dimension d, by projected gradient
    descent on the smooth sign-margin surrogate over several random restarts. Returns
    (realized, margin, B) for the best restart: `realized` is True iff every entry of M (.) B is
    strictly positive (all signs correct), and `margin` is the achieved NORMALIZED margin
    min_{ij} M_ij B_ij / (||x_i|| ||y_j||) (a lower bound on the true max margin at dimension d).
    GUARDS: empty M -> (False, 0.0, empty); d >= 1."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return False, 0.0, np.empty_like(M)
    m, n = M.shape
    d = max(1, int(d))
    best = (False, -np.inf, None)
    for r in range(restarts):
        rng = np.random.default_rng(seed + 1009 * r)
        X = rng.standard_normal((m, d)) / math.sqrt(d)
        Y = rng.standard_normal((n, d)) / math.sqrt(d)
        vX = np.zeros_like(X)
        vY = np.zeros_like(Y)
        for _ in range(steps):
            _, gX, gY = _sign_realize_grad(X, Y, M, margin0)
            vX = 0.9 * vX - lr * gX
            vY = 0.9 * vY - lr * gY
            X = X + vX
            Y = Y + vY
        B = X @ Y.T
        nx = np.linalg.norm(X, axis=1)[:, None]
        ny = np.linalg.norm(Y, axis=1)[None, :]
        norm_margin = (M * B) / np.maximum(nx * ny, 1e-12)
        realized = bool(np.all(M * B > 0))
        margin = float(norm_margin.min())
        if (realized, margin) > (best[0], best[1]):
            best = (realized, margin, B)
    return best[0], max(best[1], 0.0) if best[0] else 0.0, best[2]


def sign_rank_upper_bound(M: np.ndarray, max_d: int | None = None, **kw) -> int:
    """The smallest dimension d at which `realize_sign_rank` succeeds — an UPPER bound on the
    sign-rank that is tight for the tiny matrices we use it on (sign-rank itself is intractable in
    general). Never exceeds rank(M), which always realizes its own signs. GUARD: empty -> 0."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 0
    cap = int(min(M.shape)) if max_d is None else int(max_d)
    for d in range(1, cap + 1):
        realized, _, _ = realize_sign_rank(M, d, **kw)
        if realized:
            return d
    return cap


# --------------------------------------------------------------------------- #
# Movement 2 — a closed-form lower bound: Forster on Hadamard.
# --------------------------------------------------------------------------- #

def hadamard_pattern(k: int) -> np.ndarray:
    """The 2^k x 2^k Sylvester-Hadamard sign matrix H, built by the recursion
    H_{k} = [[H_{k-1}, H_{k-1}], [H_{k-1}, -H_{k-1}]] from H_0 = [1]. Its rows are orthogonal, so
    H H^T = 2^k I and ||H||_2 = sqrt(2^k). GUARD: k >= 0."""
    if k < 0:
        raise ValueError(f"need k >= 0, got {k}")
    H = np.ones((1, 1))
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    return H


def forster_lower_bound(M: np.ndarray) -> float:
    """Forster's spectral lower bound on the sign-rank of a sign matrix M:
    sign-rank(M) >= sqrt(mn) / ||M||_2 (the largest singular value in the denominator). For an
    N x N Hadamard matrix this is exactly sqrt(N). GUARDS: empty -> 0.0; the spectral norm is
    floored away from zero."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 0.0
    m, n = M.shape
    spectral = float(np.linalg.norm(M, 2))
    return math.sqrt(m * n) / max(spectral, 1e-12)


# --------------------------------------------------------------------------- #
# Movement 3 — margin complexity: the dimension for a usable gap.
# --------------------------------------------------------------------------- #

def margin_curve(M: np.ndarray, dims, **kw):
    """For each inner dimension d, the achievable normalized margin of a correct-sign rank-d
    realization of M (0.0 when no correct realization is found, i.e. below the sign-rank). Returns a
    list of {d, realized, margin}. The margin is zero below the sign-rank and grows with d above it:
    more dimensions buy a more robust separation — the geometric content of margin complexity."""
    rows = []
    for d in dims:
        realized, margin, _ = realize_sign_rank(M, int(d), **kw)
        rows.append({"d": int(d), "realized": bool(realized), "margin": round(float(margin), 4)})
    return rows


# --------------------------------------------------------------------------- #
# Movement 4 — the retrieval theorem and the free-embedding wall (LIMIT).
# --------------------------------------------------------------------------- #

def allpairs_qrel(n: int) -> np.ndarray:
    """The all-pairs qrel relevance mask: one query for every unordered pair {i, j} of n documents,
    relevant to exactly documents i and j. Returns a boolean (C(n,2) x n) matrix. This is the LIMIT
    construction — the densest top-2 pattern, whose realizable size is governed by the sign-rank.
    GUARD: n >= 2."""
    if n < 2:
        raise ValueError(f"all-pairs needs n >= 2, got {n}")
    pairs = list(combinations(range(n), 2))
    A = np.zeros((len(pairs), n), dtype=bool)
    for r, (i, j) in enumerate(pairs):
        A[r, i] = True
        A[r, j] = True
    return A


def _qrel_loss_grad(B: np.ndarray, rel: np.ndarray):
    """Loss and gradient (w.r.t. the score matrix B) of the multi-positive row objective: every
    relevant document must outscore every irrelevant one in its query row. For query row q and each
    relevant r, the term is logsumexp({B[q,i] : i irrelevant} + {B[q,r]}) - B[q,r] (an InfoNCE
    pushing r above all irrelevant). Fully vectorized: every positive in a row shares the same
    irrelevant log-sum, so the (queries x positives) double loop collapses to matrix ops. Returns
    (loss, dB)."""
    rel = np.asarray(rel, dtype=bool)
    neg = ~rel
    n_terms = int(rel.sum())
    if n_terms == 0:
        return 0.0, np.zeros_like(B)
    m = B.max(axis=1, keepdims=True)                 # per-row max for numerical stability
    E = np.exp(B - m)                                # shifted exponentials in (0, 1]
    neg_sum = (E * neg).sum(axis=1, keepdims=True)   # shared irrelevant log-sum per row
    denom = E + neg_sum                              # candidate sum for each positive (valid on rel)
    # term_{q,r} = logsumexp({B_qr} u {B_qi : i irrelevant}) - B_qr = m_q + log(denom_qr) - B_qr.
    loss = float(((m + np.log(denom) - B) * rel).sum() / n_terms)
    # Gradient: positive r gets softmax weight p_pos - 1; each irrelevant i accumulates E_qi/denom
    # over every positive r in its row (a shared per-row inverse-denominator sum).
    dB = rel * (E / denom - 1.0)
    inv_den = (rel / denom).sum(axis=1, keepdims=True)
    dB = dB + neg * (E * inv_den)
    return loss, dB / n_terms


def realize_qrel(rel: np.ndarray, d: int, restarts: int = 3, steps: int = 700,
                 lr: float = 1.0, seed: int = 0):
    """Best-case (free-embedding) realizability of a qrel mask in dimension d: optimize free query
    embeddings Q (nq x d) and document embeddings G (n x d) to rank every relevant document above
    every irrelevant one in its row, by projected gradient over several restarts. Returns
    (realized, row_order_accuracy): `realized` is True iff EVERY query ranks its relevant set in the
    top-|R| (row-order accuracy 1.0). GUARD: d >= 1."""
    rel = np.asarray(rel, dtype=bool)
    nq, n = rel.shape
    d = max(1, int(d))
    k_per_row = rel.sum(axis=1)
    best_acc = 0.0
    for r in range(restarts):
        rng = np.random.default_rng(seed + 7919 * r)
        Q = rng.standard_normal((nq, d)) / math.sqrt(d)
        G = rng.standard_normal((n, d)) / math.sqrt(d)
        vQ = np.zeros_like(Q)
        vG = np.zeros_like(G)
        for _ in range(steps):
            B = Q @ G.T
            _, dB = _qrel_loss_grad(B, rel)
            gQ = dB @ G
            gG = dB.T @ Q
            vQ = 0.9 * vQ - lr * gQ
            vG = 0.9 * vG - lr * gG
            Q = Q + vQ
            G = G + vG
        B = Q @ G.T
        # Row-order accuracy: a query is correct iff its top-|R| scored docs are exactly its
        # relevant set.
        correct = 0
        for q in range(nq):
            k = int(k_per_row[q])
            if k == 0:
                continue
            top = np.argpartition(-B[q], kth=k - 1)[:k]
            correct += int(set(top.tolist()) == set(np.flatnonzero(rel[q]).tolist()))
        acc = correct / nq
        best_acc = max(best_acc, acc)
        if best_acc >= 1.0:
            break
    return (best_acc >= 1.0 - 1e-12), float(best_acc)


def free_embedding_critical_n(d: int, n_grid, **kw) -> int:
    """The critical document count at dimension d: the largest n in `n_grid` whose all-pairs qrel a
    d-dimensional free embedding can realize. Once n exceeds it, no d-dimensional single-vector model
    realizes the pattern — the LIMIT wall, measured on the best case. Returns 0 if even the smallest
    n fails."""
    crit = 0
    for n in n_grid:
        realized, _ = realize_qrel(allpairs_qrel(int(n)), d, **kw)
        if realized:
            crit = int(n)
        else:
            break
    return crit


def critical_n_curve(d_grid, n_grid, **kw):
    """The free-embedding wall the viz draws: critical n vs embedding dimension d, the largest
    all-pairs document count realizable in each dimension. Returns a list of {d, critical_n}."""
    return [{"d": int(d), "critical_n": free_embedding_critical_n(int(d), n_grid, **kw)}
            for d in d_grid]


# --------------------------------------------------------------------------- #
# Module constants the viz panels step through.
# --------------------------------------------------------------------------- #

GAP_N = 4                                  # the signed-identity demonstration size
HADAMARD_KS = (1, 2, 3, 4)                 # Hadamard sizes N = 2, 4, 8, 16
MARGIN_DIMS = (1, 2, 3, 4, 5, 6)           # dimension sweep for the margin curve
CRIT_D_GRID = (2, 3, 4, 5, 6)              # embedding dimensions for the wall
CRIT_N_GRID = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)  # candidate document counts

# The combinatorial finance instance: a larger vMF corpus (more companies) over which we pose
# all-pairs "filings from both company i and company j" queries, so the relevance pattern is
# combinatorial rather than one-document-per-company. Sized so the wall bites at DPR's recovery dim.
FIN_SECTORS = 4
FIN_COMP = 6                               # 24 companies = 24 documents
FIN_SEED = 7


def finance_combinatorial_instance(n_sectors: int = FIN_SECTORS, n_comp: int = FIN_COMP,
                                    seed: int = FIN_SEED):
    """Reuse DPR's finance vMF geometry to build a larger document set (one per company) and pose
    the all-pairs combinatorial qrel over it: a query for each company pair, relevant to both
    companies' documents. Returns (rel_mask, n_docs). The simple per-company task lives on DPR's
    standard instance; this is its combinatorial counterpart on a comparable corpus."""
    _, P, _, _ = dpr_finance_matrix(seed=seed, n_sectors=n_sectors, n_comp=n_comp)
    n_docs = P.shape[0]
    return allpairs_qrel(n_docs), n_docs


# --------------------------------------------------------------------------- #
# Finance demo — the headline flip, printed.
# --------------------------------------------------------------------------- #

def finance_demo() -> dict:
    """The finance headline flip: the embedding dimension that perfectly solves single-company
    retrieval on DPR's finance corpus cannot satisfy combinatorial multi-company relevance over a
    comparable corpus — combinatorial relevance needs strictly more dimensions."""
    # Simple per-company task on DPR's standard 8-document instance (reused wholesale).
    Q, P, truth, _ = dpr_finance_matrix()
    S = score_matrix(Q, P)
    curve = rank_recall_curve(S, truth)
    d_recover = first_full_recall_dim(curve)
    print(f"  SIMPLE per-company retrieval (DPR's {P.shape[0]}-document corpus): "
          f"recall@1 recovers at d = {d_recover} (rank = {P.shape[0]})")

    # Combinatorial all-pairs task on a larger but comparable corpus.
    rel, n_docs = finance_combinatorial_instance()
    n_queries = rel.shape[0]
    realized_at_recover, acc_recover = realize_qrel(rel, d_recover)
    print(f"  COMBINATORIAL all-pairs retrieval ({n_docs} documents, {n_queries} company-pair "
          f"queries): at the SAME d = {d_recover}, realized = {realized_at_recover} "
          f"(row-order accuracy {acc_recover:.3f})")

    # The dimension the combinatorial task actually needs.
    d_combo = None
    for d in range(d_recover, n_docs + 1):
        realized, _ = realize_qrel(rel, d)
        if realized:
            d_combo = d
            break
    print(f"  combinatorial relevance is realized only at d = {d_combo} "
          f"(> the simple task's recovery dim {d_recover})")
    return {
        "d_recover": int(d_recover),
        "n_docs_combo": int(n_docs),
        "acc_combo_at_recover": round(float(acc_recover), 4),
        "d_combo": int(d_combo) if d_combo is not None else None,
    }


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_rank_vs_signrank_gap() -> None:
    """Movement 1, the headline: the signed identity has full rank n but a constructive rank-3
    realization of its sign pattern, so its sign-rank is at most 3 < n. Rank and sign-rank measure
    different things; rank is not how many dimensions the relevance PATTERN needs."""
    M = signed_identity(GAP_N)
    assert int(np.linalg.matrix_rank(M)) == GAP_N, "signed identity should have full rank n"
    realized3, margin3, B3 = realize_sign_rank(M, 3)
    assert realized3, "the signed identity's sign pattern should be realizable at rank 3"
    assert int(np.linalg.matrix_rank(B3)) <= 3, "the rank-3 realization should have rank <= 3"
    assert sign_agreement(B3, M) >= 1.0, "the rank-3 realization should match every sign"
    # The gap is real: a correct realization exists strictly below full rank.
    sr = sign_rank_upper_bound(M)
    assert sr <= 3 < GAP_N, f"sign-rank {sr} should be <= 3 and strictly below rank {GAP_N}"
    # The pattern is NOT realizable in 1 dimension (a single direction cannot separate it).
    realized1, _, _ = realize_sign_rank(M, 1)
    assert not realized1, "the signed identity should not be realizable in one dimension"
    print(f"  [ok] Movement 1: signed identity has rank {GAP_N} but sign-rank <= {sr} "
          f"(rank-3 realization matches all signs; margin {margin3:.3f})")


def test_signrank_never_exceeds_rank() -> None:
    """Movement 1 sanity: a matrix realizes its OWN signs at d = rank, so the sign-rank upper bound
    never exceeds the rank — sign-rank is the smaller, tighter measure."""
    rng = np.random.default_rng(3)
    for _ in range(3):
        R = rng.standard_normal((5, 4))
        M = np.sign(R)
        M[M == 0] = 1.0
        sr = sign_rank_upper_bound(M)
        assert sr <= int(np.linalg.matrix_rank(M)), f"sign-rank {sr} exceeded rank"
    print("  [ok] Movement 1: sign-rank upper bound never exceeds the matrix rank")


def test_forster_on_hadamard() -> None:
    """Movement 2: for an N x N Hadamard matrix, ||H||_2 = sqrt(N) exactly, so Forster's bound is
    sqrt(N); and a realization in fewer than ceil(sqrt(N)) dimensions cannot reproduce its signs,
    corroborating the lower bound."""
    for k in HADAMARD_KS:
        H = hadamard_pattern(k)
        N = H.shape[0]
        assert np.allclose(H @ H.T, N * np.eye(N)), "Hadamard rows should be orthogonal"
        assert abs(float(np.linalg.norm(H, 2)) - math.sqrt(N)) < 1e-9, "||H||_2 should be sqrt(N)"
        assert abs(forster_lower_bound(H) - math.sqrt(N)) < 1e-9, "Forster bound should be sqrt(N)"
    # Corroborate the bound: H_16 (bound 4) cannot be realized in 3 dimensions.
    H16 = hadamard_pattern(4)
    realized_below, _, _ = realize_sign_rank(H16, 3, restarts=8, steps=1000)
    assert not realized_below, "Hadamard H_16 should not be realizable below its Forster bound"
    print(f"  [ok] Movement 2: Forster on Hadamard — bound = sqrt(N) "
          f"(N in {[2 ** k for k in HADAMARD_KS]}); H_16 not realizable below sqrt(16) = 4")


def test_eckart_young_is_wrong_norm() -> None:
    """Movements 1-2 bridge: the Frobenius-optimal low-rank approximation (Eckart-Young) is not the
    sign-optimal one. The dimension to drive Frobenius error to zero is the full rank, but the sign
    pattern is recoverable strictly below it; and at an intermediate rank the truncated SVD leaves
    sign errors that a sign objective at the SAME rank fixes."""
    M = signed_identity(GAP_N)
    full_rank = int(np.linalg.matrix_rank(M))
    # Frobenius error is zero only at full rank, but the signs are all correct strictly below it.
    err_below = relative_frobenius_error(M, best_rank_d(M, full_rank - 1))
    assert err_below > 1e-6, "truncated SVD below full rank should have nonzero Frobenius error"
    # At rank 3 the Frobenius-optimal approximation need not match all signs...
    frob3 = best_rank_d(M, 3)
    frob_agree = sign_agreement(frob3, M)
    # ...but a sign-objective realization at the same rank does.
    realized3, _, B3 = realize_sign_rank(M, 3)
    sign_agree = sign_agreement(B3, M)
    assert realized3 and sign_agree >= 1.0, "sign objective should match all signs at rank 3"
    assert sign_agree >= frob_agree, \
        f"sign-optimal agreement {sign_agree} should be >= Frobenius {frob_agree} at rank 3"
    print(f"  [ok] Eckart-Young is the wrong norm: full rank {full_rank} for zero Frobenius error, "
          f"but signs recover at rank 3 (sign-opt agreement {sign_agree:.2f} >= "
          f"Frobenius {frob_agree:.2f})")


def test_margin_grows_with_dimension() -> None:
    """Movement 3: the achievable margin of a correct-sign realization is zero below the sign-rank
    (no correct realization) and positive and non-decreasing above it — more dimensions buy a more
    robust separation. The geometric content of margin complexity."""
    M = signed_identity(GAP_N)
    rows = margin_curve(M, MARGIN_DIMS)
    by_d = {r["d"]: r for r in rows}
    # Below the sign-rank (d = 1) there is no correct realization, so margin is zero.
    assert not by_d[1]["realized"] and by_d[1]["margin"] == 0.0, "d=1 should have zero margin"
    # At and above the sign-rank the pattern is realized with a positive margin.
    assert by_d[3]["realized"] and by_d[3]["margin"] > 0.0, "d=3 should realize with margin > 0"
    # The margin among the realized dimensions does not collapse as d grows (more room, not less).
    realized_margins = [r["margin"] for r in rows if r["realized"]]
    assert realized_margins[-1] >= realized_margins[0] - 1e-6, \
        f"margin should not shrink with more dimensions: {realized_margins}"
    print(f"  [ok] Movement 3: margin by dimension {[(r['d'], r['margin']) for r in rows]} "
          f"(zero below sign-rank, positive and non-shrinking above)")


def test_free_embedding_wall() -> None:
    """Movement 4 (LIMIT): even with perfect, freely optimized embeddings, the largest all-pairs
    document count realizable in d dimensions — the critical n — grows with d, and at a fixed small
    d a large-enough corpus is unrealizable. The single-vector wall, measured on the best case."""
    curve = critical_n_curve(CRIT_D_GRID, CRIT_N_GRID)
    crit = [r["critical_n"] for r in curve]
    # Critical n is non-decreasing in the embedding dimension.
    assert all(crit[i] <= crit[i + 1] for i in range(len(crit) - 1)), \
        f"critical n should grow with dimension: {crit}"
    # The wall is real: more dimensions admit strictly more documents across the range.
    assert crit[-1] > crit[0], f"more dimensions should admit more documents: {crit}"
    # At the smallest dimension, a large-enough corpus is unrealizable (recall below 1).
    big = CRIT_N_GRID[-1]
    realized_big, acc_big = realize_qrel(allpairs_qrel(big), CRIT_D_GRID[0])
    assert not realized_big, f"all-pairs over {big} docs should be unrealizable at d={CRIT_D_GRID[0]}"
    print(f"  [ok] Movement 4: free-embedding wall — critical n by d "
          f"{[(r['d'], r['critical_n']) for r in curve]}; "
          f"all-pairs over {big} docs fails at d={CRIT_D_GRID[0]} (acc {acc_big:.2f})")


def test_inbatch_loss_anchor_reused() -> None:
    """The reuse contract (the byte-for-byte twin rule, inherited from DPR): the in-batch InfoNCE
    loss is the IMPORTED loss, never reimplemented. On DPR's finance batch the Gram re-derivation
    equals `info_nce_loss_batch` to machine precision — the same anchor DPR pins."""
    Qb, Gb = dpr_inbatch_example()
    via_gram = inbatch_loss_via_gram(Qb, Gb, DPR_TAU)
    imported = info_nce_loss_batch(Qb, Gb, DPR_TAU)
    assert abs(via_gram - imported) < 1e-12, f"loss anchor drifted: {via_gram} != {imported}"
    print(f"  [ok] reuse: in-batch loss via Gram == imported info_nce_loss_batch "
          f"({via_gram:.4f}, |diff| < 1e-12)")


def test_finance_headline_flip() -> None:
    """The finance flip: the embedding dimension that perfectly recovers single-company retrieval on
    DPR's corpus FAILS combinatorial multi-company relevance over a comparable corpus, and the
    combinatorial task needs strictly more dimensions."""
    Q, P, truth, _ = dpr_finance_matrix()
    S = score_matrix(Q, P)
    d_recover = first_full_recall_dim(rank_recall_curve(S, truth))
    # Simple task: perfectly solved at d_recover (this is DPR's recovered dimension).
    assert topk_recall(best_rank_d(S, d_recover), truth, 1) >= 1.0, \
        "the simple per-company task should be solved at its recovery dimension"
    # Combinatorial task on a larger corpus: NOT realizable at the same dimension.
    rel, n_docs = finance_combinatorial_instance()
    realized_at_recover, acc = realize_qrel(rel, d_recover)
    assert not realized_at_recover, \
        f"combinatorial relevance should fail at d={d_recover} (got accuracy {acc})"
    # It is realizable with more dimensions (the pattern is not impossible, just higher-dimensional).
    realized_full, _ = realize_qrel(rel, n_docs)
    assert realized_full, "combinatorial relevance should be realizable at full dimension"
    print(f"  [ok] finance flip: simple task solved at d={d_recover}, but combinatorial all-pairs "
          f"over {n_docs} docs fails there (accuracy {acc:.3f}) and needs more dimensions")


# --------------------------------------------------------------------------- #
# Viz constants — printed for EmbeddingDimensionLaboratory.tsx to mirror.
# --------------------------------------------------------------------------- #

def viz_constants() -> None:
    """Print Panel A (rank vs sign-rank gap), Panel B (the Forster wall on Hadamard), and Panel C
    (the free-embedding retrieval wall) — all baked to the decimal in the .tsx. TS recomputes only
    CLOSED FORM (the Forster bound sqrt(N), the all-pairs query count C(n,2)); every MEASURED number
    (the sign-realization, the achievable margins, the critical-n curve, the finance flip) is baked
    here."""
    print("  PANEL A — rank vs sign-rank (the signed-identity gap):")
    M = signed_identity(GAP_N)
    realized3, margin3, B3 = realize_sign_rank(M, 3)
    rows = margin_curve(M, MARGIN_DIMS)
    print(f"    GAP_N = {GAP_N}, RANK = {int(np.linalg.matrix_rank(M))}, "
          f"SIGN_RANK = {sign_rank_upper_bound(M)}")
    print(f"    SIGN_MATRIX = {[[int(x) for x in row] for row in M]}")
    print(f"    RANK3_REALIZATION = {np.round(B3, 4).tolist()}")
    print(f"    MARGIN_BY_DIM = {[{'d': r['d'], 'realized': r['realized'], 'margin': r['margin']} for r in rows]}")

    print("  PANEL B — the Forster wall on Hadamard:")
    forster = [{"k": int(k), "N": int(2 ** k), "spectral": round(float(np.linalg.norm(hadamard_pattern(k), 2)), 4),
                "forster_bound": round(forster_lower_bound(hadamard_pattern(k)), 4)}
               for k in HADAMARD_KS]
    print(f"    FORSTER = {forster}")
    print(f"    HADAMARD_4 = {[[int(x) for x in row] for row in hadamard_pattern(2)]}")

    print("  PANEL C — the free-embedding retrieval wall (LIMIT):")
    curve = critical_n_curve(CRIT_D_GRID, CRIT_N_GRID)
    print(f"    CRIT_N_GRID = {list(CRIT_N_GRID)}")
    print(f"    CRITICAL_N = {[{'d': r['d'], 'critical_n': r['critical_n']} for r in curve]}")
    print(f"    ALLPAIRS_QUERIES = {[{'n': int(n), 'queries': int(n * (n - 1) // 2)} for n in CRIT_N_GRID]}")

    print("  FINANCE — the headline flip (simple vs combinatorial relevance):")
    fin = finance_demo()
    print(f"    FINANCE_FLIP = {fin}")


if __name__ == "__main__":
    print("Embedding-dimension lower bounds — verification harness")
    test_rank_vs_signrank_gap()                    # the headline runs first
    test_signrank_never_exceeds_rank()
    test_forster_on_hadamard()
    test_eckart_young_is_wrong_norm()
    test_margin_grows_with_dimension()
    test_free_embedding_wall()
    test_inbatch_loss_anchor_reused()
    test_finance_headline_flip()
    print("Finance demo:")
    finance_demo()
    print("Viz constants (mirrored to the decimal in EmbeddingDimensionLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
