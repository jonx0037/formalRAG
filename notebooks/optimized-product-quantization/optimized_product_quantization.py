"""Optimized product quantization (OPQ) and score-aware quantization (ScaNN) — the
reference implementation for the formalRAG `optimized-product-quantization` topic.

The previous topic ended on a cliffhanger. On a deliberately variance-imbalanced cloud, plain
product quantization has reconstruction distortion ~85.6; a NAIVE rotation that only aligns to
the principal axes barely helps (~83.3, it starves the low-variance subspaces); but a rotation
that BALANCES variance across subspaces cuts distortion to ~15.4. "Choose the rotation that
minimizes PQ distortion" is exactly OPTIMIZED product quantization — the first half of this
topic. The second half changes the LOSS, not the rotation: for maximum-inner-product search
(MIPS), the error that matters is in the inner product seen by high-scoring queries, not uniform
reconstruction. Score-aware (anisotropic) quantization — the idea behind ScaNN — weights the
quantization residual's component PARALLEL to the datapoint more heavily than the orthogonal one.

This module imports its prerequisite (`product_quantization`, which itself imports the Lloyd-Max
core) and never reimplements it, so OPQ is provably "PQ with a learned rotation" on the SAME
clouds, and re-derives the prereq's raw/pca/balanced baselines rather than hardcoding them.
It establishes, and verifies, five facts and the honest caveats they rest on:

  1. ROTATION IS A FREE ISOMETRY. An orthogonal R preserves every distance and inner product, so
     rotating data AND queries leaves the nearest-neighbor / MIPS problem unchanged. PQ is the
     R = I special case, so OPQ distortion <= PQ distortion always. (`apply_rotation`,
     `is_orthogonal`, `test_orthogonal_preserves_distances`, `test_rotation_isometry_keeps_recall`)
  2. PARAMETRIC OPQ BALANCES THE PRODUCT OF VARIANCES. Under a Gaussian/high-rate bound, optimal R
     decorrelates (PCA) AND equalizes the PRODUCT of eigenvalues (the determinant) across
     subspaces — NOT the SUM the prereq's heuristic balances. (`parametric_opq`,
     `test_parametric_balances_determinant`)
  3. NON-PARAMETRIC OPQ IS MONOTONE BLOCK COORDINATE DESCENT. Alternate (i) fix R, train PQ on the
     rotated data (Lloyd per subspace), (ii) fix the codebooks, update R by the closed-form
     Orthogonal Procrustes solution. Each step is its subproblem's global optimum, so distortion
     is monotone non-increasing — to a LOCAL optimum, not a global one. (`procrustes_update`,
     `nonparametric_opq`, `test_procrustes_minimizes_frobenius`, `validate_against_procrustes`,
     `test_opq_trajectory_monotone`, `test_opq_beats_balanced_heuristic`)
  4. SCORE-AWARE LOSS PENALIZES PARALLEL ERROR. Split the residual r = x - x_hat into its component
     PARALLEL to x and the orthogonal remainder. For aligned (high-score) queries the inner-product
     error is dominated by the parallel part, so weighting it by eta > 1 preserves the inner
     products that matter. (`residual_decomposition`, `anisotropic_loss`, `anisotropic_codeword`,
     `test_anisotropic_penalizes_parallel`)
  5. ANISOTROPIC ENCODING LOWERS HIGH-SCORE INNER-PRODUCT ERROR. On the finance cloud, re-encoding
     each vector against a shared codebook under the anisotropic metric lowers the inner-product
     error on each query's true top-k pairs versus the Euclidean encoding. DIRECTION, not a magic
     number. (`score_aware_demo`)

Honest caveats (rigorFlag territory, all asserted as DIRECTIONS only): parametric optimality holds
only under the Gaussian/high-rate bound (otherwise it is a principled INITIALIZER); the
non-parametric loop reaches only an initialization-dependent LOCAL optimum; the rotation never
changes true neighbors, so the only win is lower quantization distortion at a fixed rate; ScaNN's
gain is a direction (eta / threshold T are tuned, the clean eta formula is a large-d limit, the
additive objective assumes constant-norm data).

Every pedagogical claim is an `assert` below; `viz_constants()` prints what
`OptimizedProductQuantizationLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python \
        notebooks/optimized-product-quantization/optimized_product_quantization.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist

# OPQ = PQ with a learned rotation. Import the verified PQ core (which re-exports the Lloyd-Max
# cloud and k-means), the established two-hop cross-topic pattern: add each prereq's HYPHENATED
# directory to the path, import its UNDERSCORED module. We import PQ for everything OPQ rotates,
# and the Lloyd-Max module directly for the primitives PQ does not re-export (the same dual import
# product_quantization.py itself uses for vector_quantization_lloyd_max).
_PQ_DIR = pathlib.Path(__file__).resolve().parents[1] / "product-quantization"
if str(_PQ_DIR) not in sys.path:
    sys.path.insert(0, str(_PQ_DIR))
_VQ_DIR = pathlib.Path(__file__).resolve().parents[1] / "vector-quantization-lloyd-max"
if str(_VQ_DIR) not in sys.path:
    sys.path.insert(0, str(_VQ_DIR))

from product_quantization import (  # noqa: E402
    balanced_rotation,
    pca_align,
    pq_decode,
    pq_distortion,
    pq_encode,
    rotation_distortion_study,
    subspace_slices,
    train_pq,
    variance_imbalanced_cloud,
)
from vector_quantization_lloyd_max import (  # noqa: E402
    assign,
    best_codebook,
    finance_dataset,
)

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Rotation algebra. The whole module follows the prereq's convention: a rotation
# R has ROWS = basis vectors, and rotated data is (X - mu) @ R.T (project onto
# the rows). is_orthogonal checks R R.T = I.
# --------------------------------------------------------------------------- #

def apply_rotation(X: np.ndarray, R: np.ndarray, mu: np.ndarray | None = None) -> np.ndarray:
    """Center (if mu given) and rotate: (X - mu) @ R.T. GUARD: R must be (d, d) with d = X cols."""
    d = X.shape[1]
    if R.shape != (d, d):
        raise ValueError(f"rotation R must be ({d},{d}), got {R.shape}")
    Xc = X if mu is None else X - mu
    return Xc @ R.T


def is_orthogonal(R: np.ndarray, atol: float = 1e-9) -> bool:
    """True iff R R.T = I to atol — the property every OPQ rotation must preserve."""
    d = R.shape[0]
    return bool(np.linalg.norm(R @ R.T - np.eye(d)) < atol)


# --------------------------------------------------------------------------- #
# Parametric OPQ: PCA, then allocate principal axes to subspaces to balance the
# PRODUCT of per-subspace variances (the determinant) — the high-rate-optimal
# invariant. Contrast with the prereq's balanced_rotation, which balances the SUM.
# --------------------------------------------------------------------------- #

def parametric_opq(X: np.ndarray, m: int) -> np.ndarray:
    """OPQ's parametric (Gaussian) solution. PCA-align, then greedily assign each next-largest
    eigen-axis to the subspace with the smallest current SUM OF LOG-VARIANCES, so the per-subspace
    variance PRODUCTS (determinants) equalize — the AM-GM + Fischer optimum. The prereq's
    balanced_rotation instead equalizes the raw variance SUM (the trace); under the high-rate
    bound the determinant is the right quantity. Returns a rotation R (d, d) (rows = basis)."""
    d = X.shape[1]
    if d % m != 0:
        raise ValueError(f"dimension d={d} is not divisible by m={m} subspaces")
    R_pca, mu = pca_align(X)
    var = apply_rotation(X, R_pca, mu).var(axis=0)
    nll = -np.log(np.maximum(var, _EPS))           # per-axis -log variance (large for small var)
    order = np.argsort(var)                         # ascending variance = descending nll (LPT order)
    w = d // m
    buckets: list[list[int]] = [[] for _ in range(m)]
    load = np.zeros(m)                              # running SUM of nll per subspace (balance this)
    for ax in order:                                # longest-processing-time: biggest nll first
        full = np.array([len(b) for b in buckets]) >= w
        j = int(np.argmin(load + full * 1e18))      # least-loaded subspace, skipping full ones
        buckets[j].append(ax)
        load[j] += float(nll[ax])
    perm = np.concatenate([np.array(b, dtype=int) for b in buckets])
    return R_pca[perm]                             # permute the PCA axes into product-balanced blocks


def subspace_logvar_spread(X: np.ndarray, R: np.ndarray, m: int) -> float:
    """max - min over subspaces of the SUM of log-variances (= log of the variance PRODUCT /
    determinant). Zero means perfectly product-balanced; large means one subspace hoards the
    variance. The quantity parametric_opq drives down. GUARD: floor variances before log."""
    var = apply_rotation(X, R, X.mean(axis=0)).var(axis=0)
    logsums = [float(np.log(np.maximum(var[a:b], _EPS)).sum()) for a, b in subspace_slices(X.shape[1], m)]
    return max(logsums) - min(logsums)


# --------------------------------------------------------------------------- #
# Non-parametric OPQ: alternating optimization. The R-step is the Orthogonal
# Procrustes problem, solved in closed form by an SVD.
# --------------------------------------------------------------------------- #

def procrustes_update(Xc: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """The R-step. Given CENTERED data Xc (n, d) and fixed reconstruction targets Q (n, d) in the
    rotated frame, return the rotation R (rows = basis) minimizing the reconstruction Frobenius
    error of apply_rotation(Xc, R) = Xc @ R.T against Q. We want min over orthogonal S of
    ||Xc S - Q||_F with S = R.T; the Orthogonal Procrustes solution is S = U V.T from
    SVD(Xc.T @ Q) = U Sigma V.T, hence R = S.T = V U.T (this matches the OPQ paper's R = V U.T for
    column-stacked data). GUARDS: reject empty / mismatched inputs; SVD on a finite matrix."""
    if Xc.size == 0 or Q.size == 0:
        raise ValueError("procrustes_update needs non-empty Xc and Q")
    if Xc.shape != Q.shape:
        raise ValueError(f"Xc {Xc.shape} and Q {Q.shape} must match")
    M = Xc.T @ Q                                   # (d, d) cross-covariance
    U, _, Vt = np.linalg.svd(M)
    S = U @ Vt                                      # argmin_S ||Xc S - Q||_F  (== scipy's R)
    return S.T                                      # rotated data = Xc @ R.T = Xc @ S


def nonparametric_opq(X: np.ndarray, m: int, k_star: int, n_iter: int = 15, seed: int = 0):
    """Alternating optimization (block coordinate descent), warm-started from parametric_opq.
    Each iteration: (R-step) hold the codebooks, move R to the Procrustes optimum; (codebook step)
    hold R, retrain PQ on the rotated data. The codebook step takes the BETTER of the retrained
    codebooks and the kept ones, so the recorded distortion is provably monotone non-increasing
    regardless of any Lloyd local optimum. Returns (R, codebooks, trajectory) where trajectory[t]
    is the distortion after the codebook step of iteration t. GUARDS: n_iter >= 1; center once."""
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    mu = X.mean(axis=0)
    Xc = X - mu
    R = parametric_opq(X, m)                        # principled warm start
    cb = train_pq(apply_rotation(Xc, R), m, k_star, seed=seed)
    trajectory = [pq_distortion(apply_rotation(Xc, R), cb)]
    for _ in range(n_iter - 1):
        # R-step: targets Q = the current reconstruction in the rotated frame, then move R.
        Xr = apply_rotation(Xc, R)
        Q = pq_decode(pq_encode(Xr, cb), cb)
        R = procrustes_update(Xc, Q)
        # codebook step under the new rotation: never accept an increase vs the kept codebooks
        # (cache both distortions so pq_distortion runs exactly twice, not three times, per step).
        Xr = apply_rotation(Xc, R)
        cb_new = train_pq(Xr, m, k_star, seed=seed)
        dist_new, dist_old = pq_distortion(Xr, cb_new), pq_distortion(Xr, cb)
        if dist_new <= dist_old:
            cb = cb_new
        trajectory.append(min(dist_new, dist_old))
    return R, cb, np.array(trajectory)


# --------------------------------------------------------------------------- #
# Score-aware (anisotropic) quantization — the ScaNN idea. The residual's
# component PARALLEL to the datapoint dominates inner-product error for aligned
# queries, so weighting it by eta > 1 preserves high-score inner products.
# --------------------------------------------------------------------------- #

def residual_decomposition(x: np.ndarray, x_hat: np.ndarray):
    """Split the residual r = x - x_hat into the component PARALLEL to x (projection onto the
    datapoint direction, the MIPS-relevant axis) and the ORTHOGONAL remainder. Returns
    (r_par, r_orth) with r_par + r_orth = r and r_par . r_orth = 0. GUARD: ||x||^2 < eps (a
    zero datapoint has no defined direction) -> r_par = 0, r_orth = r."""
    r = x - x_hat
    nx2 = float(x @ x)
    if nx2 < _EPS:
        return np.zeros_like(r), r
    r_par = ((r @ x) / nx2) * x
    return r_par, r - r_par


def anisotropic_loss(x: np.ndarray, x_hat: np.ndarray, eta: float) -> float:
    """Score-aware loss eta * ||r_par||^2 + ||r_orth||^2. eta = 1 recovers isotropic ||x - x_hat||^2;
    eta > 1 up-weights the parallel (inner-product-relevant) residual. GUARD: eta >= 0."""
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    r_par, r_orth = residual_decomposition(x, x_hat)
    return float(eta * (r_par @ r_par) + (r_orth @ r_orth))


def anisotropic_codeword(x: np.ndarray, candidates: np.ndarray, eta: float) -> int:
    """Index of the codeword minimizing anisotropic_loss(x, c, eta) (lowest index breaks ties),
    versus the isotropic argmin_c ||x - c||^2. GUARD: candidates must be non-empty."""
    if len(candidates) == 0:
        raise ValueError("anisotropic_codeword needs a non-empty candidate set")
    losses = np.array([anisotropic_loss(x, c, eta) for c in candidates])
    return int(np.argmin(losses))


def _unit_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row (ScaNN's constant-norm / unit-sphere regime). GUARD: zero rows."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, _EPS)


def anisotropic_distances(X: np.ndarray, C: np.ndarray, eta: float) -> np.ndarray:
    """Vectorized score-aware distances (n, k): entry (i, j) = ||r||^2 + (eta-1)||r_par||^2 for
    r = x_i - c_j, using ||r_par||^2 = (<r, x>)^2 / ||x||^2 = (||x||^2 - <x, c>)^2 / ||x||^2.
    eta = 1 gives plain squared Euclidean. GUARD: floor ||x||^2 before dividing; eta >= 0."""
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    xn2 = np.maximum(np.einsum("ij,ij->i", X, X), _EPS)          # (n,) ||x||^2
    cn2 = np.einsum("ij,ij->i", C, C)                            # (k,) ||c||^2
    xc = X @ C.T                                                 # (n,k) <x,c>
    sq_eucl = xn2[:, None] - 2.0 * xc + cn2[None, :]             # ||r||^2
    par = (xn2[:, None] - xc) ** 2 / xn2[:, None]               # ||r_par||^2
    return sq_eucl + (eta - 1.0) * par


def anisotropic_lloyd(X: np.ndarray, k: int, eta: float, n_iter: int = 20, seed: int = 0):
    """Train a k-codeword codebook under the score-aware (anisotropic) objective — ScaNN's actual
    quantizer, not just an anisotropic re-assignment. Warm-start from isotropic k-means, then
    alternate: ASSIGN each x to its anisotropic-nearest codeword, and UPDATE each codeword to the
    closed-form score-aware optimum c* = (|S| I + (eta-1) U^T U)^{-1} (eta * sum_S x), where U are
    the unit-normalized cluster points (the M_x x = eta x identity collapses the right-hand side).
    eta = 1 reproduces Lloyd (c* = mean). Returns the codebook C (k, d). GUARDS: eta >= 1; empty
    clusters keep their codeword; the normal-equation matrix is SPD for eta >= 1."""
    if eta < 1.0:
        raise ValueError(f"anisotropic_lloyd needs eta >= 1, got {eta}")
    d = X.shape[1]
    _, C, _ = best_codebook(X, k, seed=seed, restarts=3)         # isotropic warm start
    eye = np.eye(d)
    for _ in range(n_iter):
        labels = anisotropic_distances(X, C, eta).argmin(axis=1)
        for j in range(k):
            members = X[labels == j]
            if len(members) == 0:
                continue                                         # keep the codeword (no members)
            U = _unit_rows(members)
            A = len(members) * eye + (eta - 1.0) * (U.T @ U)
            C[j] = np.linalg.solve(A, eta * members.sum(axis=0))
    return C


def _highscore_ip_mse(Qu, Xu, Xq, true_scores, topk):
    """Mean squared inner-product error (<q,x> - <q,x_q>)^2 over each query's true top-k pairs —
    the high-score pairs that actually surface in a MIPS ranking. GUARDS: cap topk at the database
    size (np.argpartition would raise otherwise); empty pair set -> 0.0."""
    topk = min(topk, true_scores.shape[1])
    approx = Qu @ Xq.T                              # (nq, n) approximate scores
    se, count = 0.0, 0
    for qi in range(Qu.shape[0]):
        top = np.argpartition(true_scores[qi], -topk)[-topk:]
        diff = true_scores[qi, top] - approx[qi, top]
        se += float(np.sum(diff * diff))
        count += len(top)
    return se / count if count else 0.0


def _mips_recall(Qu, Xu, Xq, true_scores, topk):
    """recall@topk of MIPS when database scores are approximated by <q, x_q>. GUARDS: cap topk at
    the database size (np.argpartition would raise otherwise); empty query set or topk <= 0 -> 0.0."""
    topk = min(topk, true_scores.shape[1])
    denom = Qu.shape[0] * topk
    if denom <= 0:
        return 0.0
    approx = Qu @ Xq.T
    hits = 0
    for qi in range(Qu.shape[0]):
        truth = set(np.argpartition(true_scores[qi], -topk)[-topk:].tolist())
        got = set(np.argpartition(approx[qi], -topk)[-topk:].tolist())
        hits += len(truth & got)
    return hits / denom


def score_aware_demo(seed: int = 0, eta: float = 8.0, k: int = 32, topk: int = 10) -> dict:
    """On the unit-normalized finance cloud, train TWO codebooks of the same size — isotropic
    k-means and the score-aware anisotropic quantizer (anisotropic_lloyd) — and compare the
    inner-product error on each query's true top-k pairs, plus MIPS recall@topk. The anisotropic
    codebook optimizes the loss that matters for MIPS, so high-score inner products survive
    quantization better. DIRECTION only; numbers feed the viz. Returns a dict of both metrics."""
    X, _, queries = finance_dataset()
    Xu, Qu = _unit_rows(X), _unit_rows(queries)
    _, C_iso, _ = best_codebook(Xu, k, seed=seed, restarts=3)         # isotropic k-means
    C_aniso = anisotropic_lloyd(Xu, k, eta, seed=seed)                # score-aware quantizer
    Xiso = C_iso[assign(Xu, C_iso)[0]]
    Xaniso = C_aniso[anisotropic_distances(Xu, C_aniso, eta).argmin(axis=1)]
    true_scores = Qu @ Xu.T
    return {
        "eta": eta, "k": k, "topk": topk,
        "iso_ip_mse": _highscore_ip_mse(Qu, Xu, Xiso, true_scores, topk),
        "aniso_ip_mse": _highscore_ip_mse(Qu, Xu, Xaniso, true_scores, topk),
        "iso_recall": _mips_recall(Qu, Xu, Xiso, true_scores, topk),
        "aniso_recall": _mips_recall(Qu, Xu, Xaniso, true_scores, topk),
    }


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_orthogonal_preserves_distances() -> None:
    """A rotation is an isometry: for orthogonal R, ||xR - yR|| == ||x - y|| and inner products
    are preserved — so the retrieval geometry is untouched, the rotation is free."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    R = parametric_opq(X, 4)
    assert is_orthogonal(R), "parametric_opq did not return an orthogonal matrix"
    Xr = apply_rotation(X, R)
    d_before = cdist(X, X)
    d_after = cdist(Xr, Xr)
    assert np.allclose(d_before, d_after, atol=1e-9), "rotation changed pairwise distances"
    assert np.allclose(X @ X.T, Xr @ Xr.T, atol=1e-9), "rotation changed inner products"
    print("  [ok] rotation is an isometry: distances and inner products preserved")


def test_procrustes_minimizes_frobenius() -> None:
    """procrustes_update returns the GLOBAL minimizer of ||Xc R.T - Q||_F: no random orthogonal
    perturbation of it lowers the reconstruction error."""
    rng = np.random.default_rng(1)
    Xc = rng.standard_normal((60, 6))
    Q = rng.standard_normal((60, 6))
    R = procrustes_update(Xc, Q)
    assert is_orthogonal(R), "Procrustes solution is not orthogonal"
    best = np.linalg.norm(apply_rotation(Xc, R) - Q)
    for s in range(20):
        P = parametric_opq(rng.standard_normal((30, 6)), 3)  # arbitrary orthogonal perturbations
        worse = np.linalg.norm(apply_rotation(Xc, R @ P) - Q)
        assert worse >= best - 1e-9, f"perturbed rotation beat the Procrustes optimum ({worse} < {best})"
    print(f"  [ok] Procrustes is the global R-step optimum (||Xc R.T - Q||_F = {best:.4f})")


def validate_against_procrustes() -> None:
    """Cross-check the R-step against scipy.linalg.orthogonal_procrustes (no opaque library): both
    reach the same minimal reconstruction error, to 1e-8. scipy returns S = R.T."""
    rng = np.random.default_rng(2)
    Xc = rng.standard_normal((50, 5))
    Q = rng.standard_normal((50, 5))
    R = procrustes_update(Xc, Q)
    S_scipy, _ = orthogonal_procrustes(Xc, Q)        # argmin_S ||Xc S - Q||_F
    ours = np.linalg.norm(apply_rotation(Xc, R) - Q)
    theirs = np.linalg.norm(Xc @ S_scipy - Q)
    assert abs(ours - theirs) <= 1e-8 * max(theirs, 1.0), f"objective {ours:.6f} != scipy {theirs:.6f}"
    assert np.allclose(R.T, S_scipy, atol=1e-7), "our R.T differs from scipy's S"
    print(f"  [ok] cross-check: R-step matches scipy.orthogonal_procrustes ({ours:.4f})")


def test_parametric_balances_determinant() -> None:
    """Parametric OPQ equalizes the PRODUCT of per-subspace variances: its log-variance spread
    across subspaces is far smaller than raw contiguous blocks (the SUM-balancing heuristic is a
    weaker target — we only require OPQ to beat the raw blocks here)."""
    X = variance_imbalanced_cloud(n=600, d=8, seed=1)
    raw_spread = subspace_logvar_spread(X, np.eye(8), 4)
    opq_spread = subspace_logvar_spread(X, parametric_opq(X, 4), 4)
    assert opq_spread < raw_spread, f"parametric spread {opq_spread:.3f} not below raw {raw_spread:.3f}"
    print(f"  [ok] parametric OPQ balances the determinant: log-var spread "
          f"{raw_spread:.2f} (raw) -> {opq_spread:.2f} (OPQ)")


def test_opq_trajectory_monotone() -> None:
    """Block coordinate descent: the alternating-optimization distortion trajectory is monotone
    non-increasing (each step solves its subproblem to the global optimum)."""
    X = variance_imbalanced_cloud(n=600, d=8, seed=1)
    _, _, traj = nonparametric_opq(X, 4, 16, n_iter=12, seed=1)
    assert np.all(np.diff(traj) <= 1e-9 * traj[0]), f"OPQ distortion not monotone: {np.round(traj, 3)}"
    print(f"  [ok] OPQ alternating optimization is monotone: D {traj[0]:.2f} -> {traj[-1]:.2f} "
          f"over {len(traj)} steps")


def test_opq_beats_balanced_heuristic() -> None:
    """Non-parametric OPQ matches or beats the prereq's variance-balancing HEURISTIC on the SAME
    re-derived cloud (and crushes raw PQ). Baselines are re-derived via the prereq, never
    hardcoded — provably one cloud."""
    X = variance_imbalanced_cloud(n=600, d=8, seed=1)
    base = rotation_distortion_study(X, 4, 16, seed=1)   # {raw, random, pca_only, balanced}
    _, _, traj = nonparametric_opq(X, 4, 16, n_iter=15, seed=1)
    opq = float(traj[-1])
    assert opq < base["raw"], f"OPQ {opq:.2f} did not beat raw {base['raw']:.2f}"
    assert opq <= base["balanced"] * 1.02, f"OPQ {opq:.2f} worse than heuristic {base['balanced']:.2f}"
    print(f"  [ok] OPQ {opq:.2f} <= balanced heuristic {base['balanced']:.2f} << raw {base['raw']:.2f} "
          f"(pca_only {base['pca_only']:.2f})")


def test_anisotropic_penalizes_parallel() -> None:
    """The score-aware mechanism, in isolation: of two reconstructions with EQUAL Euclidean error,
    the one with the smaller PARALLEL residual has the smaller anisotropic loss (eta > 1) AND lower
    inner-product error for aligned (high-score) queries. The parallel component is what moves the
    inner product."""
    x = np.array([1.0, 0.0])
    err = 0.3
    x_par = x - np.array([err, 0.0])                  # residual entirely PARALLEL to x
    x_orth = x - np.array([0.0, err])                 # residual entirely ORTHOGONAL, equal norm
    rp, _ = residual_decomposition(x, x_par)
    _, ro = residual_decomposition(x, x_orth)
    assert np.isclose(np.linalg.norm(x - x_par), np.linalg.norm(x - x_orth)), "set up equal-norm residuals"
    eta = 5.0
    assert anisotropic_loss(x, x_orth, eta) < anisotropic_loss(x, x_par, eta), \
        "anisotropic loss should PREFER the reconstruction with smaller parallel residual"
    # Monte-Carlo over aligned unit queries (score >= T): the orthogonal-residual reconstruction
    # (parallel preserved) has lower inner-product error.
    rng = np.random.default_rng(0)
    ang = rng.uniform(-np.pi / 4, np.pi / 4, 4000)    # queries within 45 deg of x (high score)
    Qd = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    ip_par = np.mean((Qd @ (x - x_par)) ** 2)
    ip_orth = np.mean((Qd @ (x - x_orth)) ** 2)
    assert ip_orth < ip_par, f"aligned-query IP error: orthogonal-residual {ip_orth:.4f} !< parallel {ip_par:.4f}"
    print(f"  [ok] score-aware: preserving the parallel component lowers high-score IP error "
          f"({ip_orth:.4f} < {ip_par:.4f})")


def test_anisotropic_encoding_lowers_highscore_error() -> None:
    """On the finance cloud, the score-aware (anisotropic) codebook lowers the inner-product error
    on each query's true top-k pairs versus an isotropic k-means codebook of the same size, and
    does not lower MIPS recall. DIRECTION, not a magic number."""
    d = score_aware_demo()
    assert d["aniso_ip_mse"] < d["iso_ip_mse"], \
        f"anisotropic IP-MSE {d['aniso_ip_mse']:.5f} not below isotropic {d['iso_ip_mse']:.5f}"
    assert d["aniso_recall"] >= d["iso_recall"], \
        f"anisotropic recall {d['aniso_recall']:.3f} below isotropic {d['iso_recall']:.3f}"
    print(f"  [ok] score-aware codebook lowers high-score IP error: "
          f"{d['iso_ip_mse']:.5f} -> {d['aniso_ip_mse']:.5f} "
          f"({100 * (d['iso_ip_mse'] - d['aniso_ip_mse']) / d['iso_ip_mse']:.0f}% less, recall "
          f"{d['iso_recall']:.3f} -> {d['aniso_recall']:.3f}, eta={d['eta']})")


def test_rotation_isometry_keeps_recall() -> None:
    """Applying ANY orthogonal R to both the database and the queries leaves the true nearest
    neighbors unchanged — the invariance OPQ exploits to rotate freely."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 8))
    queries = rng.standard_normal((25, 8))
    R = random_orthogonal(8, seed=4)
    nn_before = cdist(queries, X, "sqeuclidean").argmin(axis=1)
    nn_after = cdist(apply_rotation(queries, R), apply_rotation(X, R), "sqeuclidean").argmin(axis=1)
    assert np.array_equal(nn_before, nn_after), "rotation changed the true nearest neighbors"
    print("  [ok] rotating data AND queries preserves every true nearest neighbor")


def random_orthogonal(d: int, seed: int = 0) -> np.ndarray:
    """A Haar-distributed orthogonal matrix via QR of a Gaussian (rows = basis)."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return Q


# --------------------------------------------------------------------------- #
# Viz constants — the exact numbers OptimizedProductQuantizationLaboratory.tsx mirrors.
# --------------------------------------------------------------------------- #

def viz_constants() -> None:
    """Print the rotation quartet + per-subspace variances (Panel A), the convergence trajectory
    (Panel B), and the score-aware decomposition + eta sweep (Panel C) — all baked to the decimal
    in the .tsx."""
    X = variance_imbalanced_cloud(n=600, d=8, seed=1)
    m, k_star = 4, 16
    base = rotation_distortion_study(X, m, k_star, seed=1)
    R_opq, _, traj = nonparametric_opq(X, m, k_star, n_iter=12, seed=1)

    # Panel A: the four distortions on the one cloud, and per-subspace variance bars.
    print("  PANEL A — rotation distortions on the variance-imbalanced cloud (n=600,d=8,m=4,k*=16):")
    print(f"    DISTORTIONS raw={base['raw']:.4f} pca_only={base['pca_only']:.4f} "
          f"balanced={base['balanced']:.4f} opq={traj[-1]:.4f}")
    R_bal = balanced_rotation(X, m)
    for name, R in [("raw", np.eye(8)), ("pca_only", pca_align(X)[0]),
                    ("balanced", R_bal), ("opq", R_opq)]:
        var = apply_rotation(X, R, X.mean(axis=0)).var(axis=0)
        bars = [float(np.round(var[a:b].sum(), 4)) for a, b in subspace_slices(8, m)]
        print(f"    SUBSPACE_VAR[{name}] = {bars}")

    # Panel B: the monotone convergence trajectory.
    print("  PANEL B — alternating-optimization distortion trajectory:")
    print(f"    TRAJECTORY = {[float(np.round(t, 4)) for t in traj]}")
    print(f"    BALANCED_REF = {base['balanced']:.4f}  (heuristic line OPQ crosses below)")

    # Panel C: score-aware residual decomposition + an eta sweep where the choice SWAPS from the
    # Euclidean-closest codeword (cand0) to the parallel-preserving one (cand1) as eta rises.
    print("  PANEL C — score-aware (anisotropic) quantization:")
    x = np.array([1.0, 0.0])
    cands = np.array([[0.8, 0.1], [1.0, 0.28], [0.6, 0.0], [0.85, 0.35]])
    rp, ro = residual_decomposition(x, cands[0])
    print(f"    X = {x.tolist()}  CANDS = {cands.tolist()}")
    print(f"    R_PAR(x, cand0) = {np.round(rp, 4).tolist()}  R_ORTH = {np.round(ro, 4).tolist()}")
    for eta in (1.0, 2.0, 4.0, 8.0):
        idx = anisotropic_codeword(x, cands, eta)
        losses = [round(anisotropic_loss(x, c, eta), 4) for c in cands]
        print(f"    ETA={eta:>4}: chosen_codeword={idx}  losses={losses}")
    demo = score_aware_demo()
    print(f"    FINANCE iso_ip_mse={demo['iso_ip_mse']:.5f} aniso_ip_mse={demo['aniso_ip_mse']:.5f} "
          f"iso_recall={demo['iso_recall']:.4f} aniso_recall={demo['aniso_recall']:.4f}")


if __name__ == "__main__":
    print("Optimized PQ (OPQ) / score-aware quantization (ScaNN) verification harness")
    test_orthogonal_preserves_distances()
    test_procrustes_minimizes_frobenius()
    validate_against_procrustes()
    test_parametric_balances_determinant()
    test_opq_trajectory_monotone()
    test_opq_beats_balanced_heuristic()
    test_anisotropic_penalizes_parallel()
    test_anisotropic_encoding_lowers_highscore_error()
    test_rotation_isometry_keeps_recall()
    print("Viz constants (mirrored to the decimal in OptimizedProductQuantizationLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
