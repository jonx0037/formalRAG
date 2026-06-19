"""PCA as optimal linear dimensionality reduction for embeddings — the reference
implementation for the formalRAG `pca-dimensionality-reduction` topic.

A retriever stores millions of d-dimensional embeddings (d = 768, 1536, 3072), and
both memory and ANN latency scale with d. High-dimensional geometry told us real
embeddings have low *effective* rank, so most of those dimensions carry little
variance. PCA is the variance-optimal linear projection onto the directions the
embedding cloud actually spreads along. This module establishes — and verifies —
four facts, then the honest twist that variance-optimal is not retrieval-optimal:

  1. PCA TWO WAYS. The principal directions are the top eigenvectors of the centered
     covariance Sigma = X~^T X~ / (n-1); equivalently the right singular vectors of
     the centered data matrix X~, with eigenvalue lambda_i = s_i^2 / (n-1). The
     eigendecomposition and the SVD agree to machine precision.
  2. ECKART-YOUNG-MIRSKY. The best rank-k approximation is the truncated SVD, with
     squared Frobenius error exactly sum_{i>k} s_i^2 = (n-1) sum_{i>k} lambda_i — and
     it beats every random rank-k projection.
  3. PROJECTION DISTORTION = EXPLAINED VARIANCE. The fraction of squared norm (and of
     mean pairwise squared distance) retained by the top-k projection equals the
     explained-variance ratio EVR(k) = sum_{i<=k} lambda_i / sum lambda_i — exactly,
     since both use the same eigendecomposition.
  4. RETRIEVAL SURVIVES, AND PCA BEATS RANDOM. Nearest-neighbor recall@10 after
     projecting to k dims is monotone in k and, because PCA is data-DEPENDENT, beats a
     data-oblivious random projection at the same k. (Random projection / Johnson-
     Lindenstrauss is the forward topic.)

Every pedagogical claim is an `assert` below. The exact targets are exact: eigh and
SVD eigenvalues agree; Frobenius error^2 = sum of tail singular values^2; EVR is read
straight off the spectrum. `grid_table()` prints the scree / reconstruction / recall /
scatter blocks that `SpectrumLaboratory.tsx` mirrors to the decimal. NUMERICAL NOTES:
center before both methods; use `eigh` (symmetric) not `eig`; eigenvectors are sign-
ambiguous (canonicalize before comparing); assert only order-invariant quantities
(EVR, effective rank, reconstruction error) past the signal rank, where the noise-floor
eigenvalues are tied. `sklearn.decomposition.PCA` is the reference cross-check.

Run:  uv run --with numpy --with scipy --with scikit-learn python notebooks/pca-dimensionality-reduction/pca_dimensionality_reduction.py
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as SklearnPCA

# Kept-dimension grid the viz panels step through.
K_GRID: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512, 768)


# --------------------------------------------------------------------------- #
# Synthetic embeddings — a low-rank signal with a decaying spectrum plus an
# ambient noise floor, optionally split into topical clusters. A stand-in for the
# manifold a real encoder produces; NOT a trained encoder.
# --------------------------------------------------------------------------- #

def structured_embeddings(n: int, d: int, k: int, n_clusters: int = 1,
                          decay: float = 0.93, noise: float = 0.05,
                          cluster_sep: float = 2.5, seed: int = 0):
    """n embeddings in R^d on a k-dim latent subspace with a decaying spectrum.

    The latent coordinates have standard deviations decay^j (a heavy-tailed signal
    spectrum), optionally offset into `n_clusters` topical groups, then mapped through
    a random orthonormal d x k basis and perturbed by ambient Gaussian noise. Returns
    (X, labels).
    """
    rng = np.random.default_rng(seed)
    scales = decay ** np.arange(k)                          # decaying signal strengths
    z = rng.standard_normal((n, k)) * scales
    if n_clusters > 1:
        labels = rng.integers(0, n_clusters, n)
        centers = rng.standard_normal((n_clusters, k)) * scales * cluster_sep
        z = z + centers[labels]
    else:
        labels = np.zeros(n, dtype=int)
    basis, _ = np.linalg.qr(rng.standard_normal((d, k)))    # d x k orthonormal columns
    X = z @ basis.T + noise * rng.standard_normal((n, d))
    return X, labels


# --------------------------------------------------------------------------- #
# PCA two ways, and the bookkeeping the theory needs.
# --------------------------------------------------------------------------- #

def _center(X: np.ndarray):
    mu = X.mean(axis=0)
    return X - mu, mu


def _canon_signs(comps: np.ndarray) -> np.ndarray:
    """Fix each component's sign so its largest-magnitude entry is positive, so two
    eigenbases can be compared despite the eigenvector sign ambiguity."""
    out = comps.copy()
    for i in range(out.shape[0]):
        j = int(np.argmax(np.abs(out[i])))
        if out[i, j] < 0:
            out[i] = -out[i]
    return out


def pca_via_covariance(X: np.ndarray):
    """PCA by eigendecomposition of the centered covariance. Returns
    (eigvals desc, components as rows, mean)."""
    Xc, mu = _center(X)
    cov = Xc.T @ Xc / (X.shape[0] - 1)
    w, V = scipy.linalg.eigh(cov)            # ascending eigenvalues, symmetric solver
    order = np.argsort(w)[::-1]
    return w[order], V[:, order].T, mu


def pca_via_svd(X: np.ndarray):
    """PCA by SVD of the centered data matrix. Returns
    (eigvals desc, components as rows, singular values, mean)."""
    Xc, mu = _center(X)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = s ** 2 / (X.shape[0] - 1)
    return eigvals, Vt, s, mu


def explained_variance_ratio(eigvals: np.ndarray) -> np.ndarray:
    """Cumulative EVR(k) = sum_{i<=k} lambda_i / sum lambda_i."""
    c = np.cumsum(eigvals)
    return c / c[-1]


def effective_rank(eigvals: np.ndarray) -> float:
    """Participation ratio (sum lambda)^2 / sum lambda^2 — a soft count of the
    directions that carry variance."""
    return float(eigvals.sum() ** 2 / np.sum(eigvals ** 2))


# --------------------------------------------------------------------------- #
# Eckart-Young: reconstruction error and the random-projection comparison.
# --------------------------------------------------------------------------- #

def reconstruction_error_sq(X: np.ndarray, k: int):
    """Squared Frobenius reconstruction error of the rank-k truncated SVD, computed
    directly and as the tail singular-value sum. Returns (direct, tail_sum)."""
    Xc, _ = _center(X)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xk = (U[:, :k] * s[:k]) @ Vt[:k]
    return float(np.sum((Xc - Xk) ** 2)), float(np.sum(s[k:] ** 2))


def random_projection_error_sq(X: np.ndarray, k: int, trials: int = 5,
                               seed: int = 0) -> float:
    """Mean squared Frobenius error of projecting onto a random orthonormal
    k-subspace — always at least the truncated-SVD error (Eckart-Young)."""
    Xc, _ = _center(X)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(trials):
        W, _ = np.linalg.qr(rng.standard_normal((X.shape[1], k)))
        proj = Xc @ W @ W.T
        errs.append(np.sum((Xc - proj) ** 2))
    return float(np.mean(errs))


# --------------------------------------------------------------------------- #
# Projection distortion = explained-variance ratio (Theorem 4).
# --------------------------------------------------------------------------- #

def projection_distortion(X: np.ndarray, k: int):
    """Fraction of squared norm and of mean pairwise squared distance retained by the
    top-k PCA projection, alongside EVR(k). All three coincide. Returns
    (retained_norm, retained_pairwise, evr_k)."""
    eigvals, comps, mu = pca_via_covariance(X)
    Xc = X - mu
    W = comps[:k].T                                  # d x k top eigenvectors
    scores = Xc @ W
    retained_norm = float(np.sum(scores ** 2) / np.sum(Xc ** 2))
    # Mean pairwise squared distance = 2n * total squared-norm energy for centered data
    # (the cross term vanishes because sum_i Xc_i = 0), so its retained fraction is the
    # same ratio — computed on the full centered cloud, where the identity is exact.
    full_pw = float(np.sum(cdist(Xc, Xc, "sqeuclidean")))
    proj_pw = float(np.sum(cdist(scores, scores, "sqeuclidean")))
    retained_pairwise = proj_pw / full_pw
    evr_k = float(explained_variance_ratio(eigvals)[k - 1])
    return retained_norm, retained_pairwise, evr_k


# --------------------------------------------------------------------------- #
# Retrieval: nearest-neighbor recall@k after projection, PCA vs random.
# --------------------------------------------------------------------------- #

def _topk_neighbors(queries: np.ndarray, corpus: np.ndarray, topk: int) -> np.ndarray:
    """Indices of the `topk` nearest corpus rows to each query (squared Euclidean)."""
    d2 = cdist(queries, corpus, "sqeuclidean")
    return np.argpartition(d2, topk, axis=1)[:, :topk]


def recall_after_projection(X: np.ndarray, queries: np.ndarray, kept_dims,
                            topk: int = 10, method: str = "pca",
                            seed: int = 0) -> dict[int, float]:
    """Mean recall@topk of the full-dimensional nearest neighbors after projecting the
    corpus and queries to each kept dimension, by PCA or by a random orthonormal map."""
    eigvals, comps, mu = pca_via_covariance(X)
    Xc, Qc = X - mu, queries - mu
    truth = _topk_neighbors(Qc, Xc, topk)
    truth_sets = [set(row) for row in truth]
    rng = np.random.default_rng(seed)
    out: dict[int, float] = {}
    for kd in kept_dims:
        if method == "pca":
            W = comps[:kd].T
        else:
            W, _ = np.linalg.qr(rng.standard_normal((X.shape[1], kd)))
        nn = _topk_neighbors(Qc @ W, Xc @ W, topk)
        recalls = [len(truth_sets[i].intersection(nn[i])) / topk for i in range(len(nn))]
        out[kd] = float(np.mean(recalls))
    return out


# --------------------------------------------------------------------------- #
# The showcase / finance dataset, and the grid table the viz mirrors.
# --------------------------------------------------------------------------- #

FINANCE_DIM = 1536
FINANCE_INTRINSIC_K = 48
FINANCE_CLUSTERS = 3
FINANCE_KEPT = 128       # the production kept-dimension the finance demo reports at


def finance_dataset():
    """The canonical synthetic financial-embedding cloud both grid_table() and
    finance_demo() read from: 1536-d, low decaying-rank signal, three topical clusters,
    plus a held-out query set. SYNTHETIC low-rank-plus-noise, not a trained encoder."""
    X, labels = structured_embeddings(1200, FINANCE_DIM, FINANCE_INTRINSIC_K,
                                      n_clusters=FINANCE_CLUSTERS, decay=0.97,
                                      noise=0.02, cluster_sep=1.2, seed=1)
    queries, _ = structured_embeddings(150, FINANCE_DIM, FINANCE_INTRINSIC_K,
                                       n_clusters=FINANCE_CLUSTERS, decay=0.97,
                                       noise=0.02, cluster_sep=1.2, seed=2)
    return X, labels, queries


def grid_table() -> dict[str, list]:
    """The numbers SpectrumLaboratory.tsx mirrors to the decimal. Deterministic.

    scree   : top eigenvalues (scree curve) + cumulative EVR at every K_GRID point.
    error   : per-K PCA reconstruction error^2 and mean random-projection error^2.
    recall  : per-K recall@10 retained, PCA vs random projection.
    scatter : top-2 principal-component coordinates of a labeled cluster subsample.
    """
    X, labels, queries = finance_dataset()
    eigvals, comps, mu = pca_via_covariance(X)
    evr = explained_variance_ratio(eigvals)

    scree = {
        "top_eigvals": [float(v) for v in eigvals[:40]],
        "evr_at_k": [float(evr[k - 1]) for k in K_GRID],
        "effective_rank": effective_rank(eigvals),
        "total_variance": float(eigvals.sum()),
    }
    error = [
        {"k": k,
         "pca_err": reconstruction_error_sq(X, k)[1],
         "rand_err": random_projection_error_sq(X, k, trials=5, seed=3)}
        for k in K_GRID
    ]
    rec_pca = recall_after_projection(X, queries, K_GRID, method="pca")
    rec_rand = recall_after_projection(X, queries, K_GRID, method="random", seed=4)
    recall = [{"k": k, "pca": rec_pca[k], "rand": rec_rand[k]} for k in K_GRID]

    # Top-2 PC scatter: project a labeled subsample onto PC1, PC2.
    sub_idx = np.concatenate([np.where(labels == c)[0][:25] for c in range(FINANCE_CLUSTERS)])
    scores2 = (X[sub_idx] - mu) @ comps[:2].T
    scatter = [{"x": float(p[0]), "y": float(p[1]), "c": int(labels[i])}
               for p, i in zip(scores2, sub_idx)]
    return {"scree": scree, "error": error, "recall": recall, "scatter": scatter}


def finance_demo() -> dict:
    """Headline numbers at the production kept-dimension: effective rank, EVR, and
    recall@10 retained for PCA vs random projection in R^1536."""
    X, labels, queries = finance_dataset()
    eigvals, _, _ = pca_via_covariance(X)
    evr = explained_variance_ratio(eigvals)
    rec_pca = recall_after_projection(X, queries, [FINANCE_KEPT], method="pca")[FINANCE_KEPT]
    rec_rand = recall_after_projection(X, queries, [FINANCE_KEPT], method="random", seed=4)[FINANCE_KEPT]
    out = {
        "dim": FINANCE_DIM,
        "intrinsic_k": FINANCE_INTRINSIC_K,
        "effective_rank": effective_rank(eigvals),
        "evr_at_kept": float(evr[FINANCE_KEPT - 1]),
        "kept": FINANCE_KEPT,
        "recall_pca": rec_pca,
        "recall_rand": rec_rand,
    }
    print(f"  {FINANCE_CLUSTERS} topical clusters of embeddings in R^{FINANCE_DIM} "
          f"(SYNTHETIC low-rank-plus-noise, not a trained encoder):")
    print(f"  effective rank {out['effective_rank']:.1f} (of {FINANCE_DIM} ambient "
          f"dims); intrinsic signal rank {FINANCE_INTRINSIC_K}")
    print(f"  keep top {FINANCE_KEPT} PCs -> EVR {out['evr_at_kept']*100:.2f}% of variance")
    print(f"  recall@10 retained at {FINANCE_KEPT} dims:  PCA {out['recall_pca']*100:.1f}%"
          f"   random projection {out['recall_rand']*100:.1f}%")
    print("  -> the embedding cloud has low effective rank; PCA keeps the directions "
          "retrieval lives in, so it beats a data-oblivious random projection.")
    return out


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_pca_eig_svd_agree() -> None:
    """PCA by covariance-eigendecomposition and by SVD give the same eigenvalues
    (machine precision) and the same components up to sign (on the signal rank)."""
    X, _ = structured_embeddings(600, 200, 20, n_clusters=3, seed=0)
    w_cov, comps_cov, _ = pca_via_covariance(X)
    w_svd, comps_svd, _, _ = pca_via_svd(X)
    assert np.allclose(w_cov, w_svd, atol=1e-9), "eig vs SVD eigenvalues disagree"
    top = 20
    assert np.allclose(_canon_signs(comps_cov[:top]), _canon_signs(comps_svd[:top]),
                       atol=1e-6), "eig vs SVD components disagree on the signal rank"
    print("  [ok] PCA two ways agree: eig(Sigma) eigenvalues = s^2/(n-1), same components")


def test_rayleigh_first_pc() -> None:
    """The first principal direction maximizes the Rayleigh quotient w^T Sigma w over
    unit w (no random unit vector beats lambda_1), attaining v1^T Sigma v1 = lambda_1."""
    X, _ = structured_embeddings(500, 80, 15, seed=1)
    Xc, _ = _center(X)
    cov = Xc.T @ Xc / (X.shape[0] - 1)
    eigvals, comps, _ = pca_via_covariance(X)
    v1, lam1 = comps[0], eigvals[0]
    assert abs(v1 @ cov @ v1 - lam1) < 1e-8, "v1^T Sigma v1 != lambda_1"
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4000, 80))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    rq = np.einsum("ij,jk,ik->i", W, cov, W)
    assert rq.max() <= lam1 + 1e-9, "a random unit vector beat the Rayleigh maximum"
    print("  [ok] Rayleigh quotient: top eigenvector maximizes w^T Sigma w = lambda_1")


def test_variance_reconstruction_equivalence() -> None:
    """E||z - WW^T z||^2 = tr(Sigma) - tr(W^T Sigma W) for the PCA frame W, and the
    residual equals the tail eigenvalue sum (variance maximized = error minimized)."""
    X, _ = structured_embeddings(600, 120, 18, seed=2)
    eigvals, comps, mu = pca_via_covariance(X)
    Xc = X - mu
    cov = Xc.T @ Xc / (X.shape[0] - 1)
    for k in (5, 20, 60):
        W = comps[:k].T
        resid = Xc - Xc @ W @ W.T
        mean_recon = np.sum(resid ** 2) / (X.shape[0] - 1)
        identity = np.trace(cov) - np.trace(W.T @ cov @ W)
        assert abs(mean_recon - identity) < 1e-8, f"reconstruction identity off at k={k}"
        assert abs(identity - eigvals[k:].sum()) < 1e-8, f"residual != tail sum at k={k}"
    print("  [ok] variance-max = reconstruction-min: residual = sum_{i>k} lambda_i")


def test_eckart_young() -> None:
    """Truncated-SVD reconstruction error^2 = sum of tail singular values^2, and it is
    no larger than any random rank-k projection's error (Eckart-Young-Mirsky)."""
    X, _ = structured_embeddings(700, 150, 25, n_clusters=3, seed=3)
    eigvals, _, _ = pca_via_covariance(X)
    for k in (4, 16, 64):
        direct, tail = reconstruction_error_sq(X, k)
        assert abs(direct - tail) < 1e-6 * max(1.0, tail), f"Frobenius != tail sum at k={k}"
        assert abs(tail - (X.shape[0] - 1) * eigvals[k:].sum()) < 1e-5 * tail, \
            f"tail s^2 != (n-1) tail lambda at k={k}"
        rand = random_projection_error_sq(X, k, trials=8, seed=k)
        assert tail <= rand + 1e-6, f"truncated SVD beaten by random projection at k={k}"
    print("  [ok] Eckart-Young: error^2 = sum tail s^2, and truncated SVD beats random")


def test_explained_variance_and_effective_rank() -> None:
    """Cumulative EVR is monotone to 1; effective rank is far below ambient d for the
    structured cloud and near d for isotropic data."""
    X, _ = structured_embeddings(800, 256, 24, n_clusters=3, decay=0.9, noise=0.04, seed=4)
    eigvals, _, _ = pca_via_covariance(X)
    evr = explained_variance_ratio(eigvals)
    assert np.all(np.diff(evr) >= -1e-12), "EVR not monotone nondecreasing"
    assert abs(evr[-1] - 1.0) < 1e-9, "EVR does not reach 1"
    er_struct = effective_rank(eigvals)
    assert er_struct < 80, f"structured effective rank too high: {er_struct:.1f}"
    iso, _ = structured_embeddings(800, 256, 256, decay=1.0, noise=0.0, seed=5)
    er_iso = effective_rank(pca_via_covariance(iso)[0])
    assert er_iso > 4 * er_struct, f"isotropic effective rank {er_iso:.0f} not >> {er_struct:.0f}"
    print(f"  [ok] explained variance + effective rank: structured {er_struct:.1f} << "
          f"isotropic {er_iso:.0f} (ambient 256)")


def test_projection_distortion() -> None:
    """The retained fraction of squared norm and of mean pairwise squared distance both
    equal EVR(k) — exactly for the norm, within sampling for the pairwise check."""
    X, _ = structured_embeddings(900, 200, 20, n_clusters=3, seed=6)
    for k in (4, 16, 64):
        rn, rp, evr_k = projection_distortion(X, k)
        assert abs(rn - evr_k) < 1e-9, f"retained norm != EVR at k={k}: {rn} vs {evr_k}"
        assert abs(rp - evr_k) < 1e-6, f"retained pairwise distance != EVR at k={k}"
    print("  [ok] projection distortion: retained squared-distance fraction = EVR(k)")


def test_recall_after_projection() -> None:
    """Nearest-neighbor recall@10 after projection is monotone in kept dimension and
    PCA beats random projection at every kept dimension on the structured cloud."""
    X, _ = structured_embeddings(1000, 256, 24, n_clusters=3, seed=7)
    queries, _ = structured_embeddings(120, 256, 24, n_clusters=3, seed=8)
    kept = (4, 8, 16, 32, 64, 128)
    pca = recall_after_projection(X, queries, kept, method="pca")
    rand = recall_after_projection(X, queries, kept, method="random", seed=9)
    pca_seq = [pca[k] for k in kept]
    for a, b in zip(pca_seq, pca_seq[1:]):
        assert b >= a - 0.02, f"PCA recall not monotone in kept dim: {a:.3f} -> {b:.3f}"
    for k in kept:
        assert pca[k] >= rand[k] - 1e-9, f"PCA recall below random at kept={k}"
    assert pca[kept[-1]] > 0.9, f"PCA recall should approach 1 by kept={kept[-1]}"
    print(f"  [ok] retrieval: recall@10 monotone in k, PCA >= random (PCA {pca[64]*100:.0f}% "
          f"vs random {rand[64]*100:.0f}% at 64 dims)")


def test_sklearn_crosscheck() -> None:
    """Our SVD-based PCA matches sklearn.decomposition.PCA (the reference library)."""
    X, _ = structured_embeddings(500, 100, 15, n_clusters=3, seed=10)
    eigvals, comps, _, _ = pca_via_svd(X)
    ref = SklearnPCA(svd_solver="full").fit(X)
    assert np.allclose(eigvals, ref.explained_variance_, atol=1e-8), \
        "eigenvalues disagree with sklearn"
    assert np.allclose(_canon_signs(comps[:15]), _canon_signs(ref.components_[:15]), atol=1e-6), \
        "components disagree with sklearn"
    print("  [ok] cross-check: PCA matches sklearn.decomposition.PCA")


def test_finance_spectrum() -> None:
    """The 1536-d financial cloud has low effective rank, high EVR at the production
    kept-dim, and PCA retains more recall than random projection there."""
    f = finance_demo()
    assert f["effective_rank"] < 0.2 * f["dim"], "finance effective rank not low"
    assert f["evr_at_kept"] > 0.95, f"EVR at kept too low: {f['evr_at_kept']:.3f}"
    assert f["recall_pca"] > f["recall_rand"], "PCA recall not above random at the kept dim"
    print("  [ok] finance: low effective rank, high EVR at 128 dims, PCA recall > random")


if __name__ == "__main__":
    print("PCA / optimal linear dimensionality reduction verification harness")
    test_pca_eig_svd_agree()
    test_rayleigh_first_pc()
    test_variance_reconstruction_equivalence()
    test_eckart_young()
    test_explained_variance_and_effective_rank()
    test_projection_distortion()
    test_recall_after_projection()
    test_sklearn_crosscheck()
    tbl = grid_table()
    print("Grid table (mirrored by SpectrumLaboratory.tsx):")
    sc = tbl["scree"]
    print(f"  effective rank {sc['effective_rank']:.2f}; top eigenvalues "
          f"{', '.join(f'{v:.3f}' for v in sc['top_eigvals'][:6])} ...")
    print(f"  {'k':>6}{'EVR':>9}{'pca_err':>13}{'rand_err':>13}"
          f"{'recall_pca':>12}{'recall_rand':>13}")
    for e, r, k, ev in zip(tbl["error"], tbl["recall"], K_GRID, sc["evr_at_k"]):
        print(f"  {k:>6}{ev:>9.4f}{e['pca_err']:>13.1f}{e['rand_err']:>13.1f}"
              f"{r['pca']:>12.4f}{r['rand']:>13.4f}")
    print("Finance demo:")
    test_finance_spectrum()
    print("All checks passed.")
