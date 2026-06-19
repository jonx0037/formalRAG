"""Random projections and the Johnson-Lindenstrauss lemma — the reference
implementation for the formalRAG `johnson-lindenstrauss` topic.

PCA reduces an embedding cloud's dimension by reading the cloud's own covariance:
data-DEPENDENT and variance-optimal. Johnson-Lindenstrauss reduces it the opposite
way — multiply by a *random* matrix that has never seen the data — and still preserves
every pairwise distance to within (1 +/- eps). This module establishes, and verifies,
the chain that makes that work:

  1. NORM PRESERVATION IN EXPECTATION. For a random Gaussian matrix A in R^{k x d} with
     i.i.d. N(0,1) entries, the map f(x) = (1/sqrt(k)) A x preserves squared norm in
     expectation: E||f(x)||^2 = ||x||^2. The same holds for Rademacher (+/-1) and sparse
     Achlioptas matrices.
  2. CONCENTRATION (the engine). For the Gaussian map, ||f(x)||^2 / ||x||^2 ~ chi^2_k / k
     EXACTLY, for any ambient d. So it concentrates about 1 at rate exp(-c k eps^2)
     (Laurent-Massart chi-square tail) — independent of d.
  3. THE JL LEMMA (union bound). Apply (2) to the C(n,2) difference vectors of n points;
     a union bound shows k >= 4 ln(n) / (eps^2/2 - eps^3/3) suffices for ALL pairwise
     squared distances to survive to (1 +/- eps). The target dimension depends on
     ln(n) and eps, NOT on d.
  4. DIMENSION INDEPENDENCE, AND DATA-OBLIVIOUS vs DATA-DEPENDENT. The distortion
     distribution at fixed k is invariant to d and worsens with n (more pairs). Random
     projection preserves pairwise DISTANCES, but exact nearest-neighbor recall is a
     stricter ask: on a tightly-clustered low-rank cloud, +/-eps distance distortion
     reshuffles the top-10, so recall is the price of being data-oblivious — and
     data-dependent PCA keeps far more of it at the same k. The honest contrast with
     the prerequisite topic.

Every pedagogical claim is an `assert` below. NUMERICAL NOTES: the chi^2_k/k identity is
exact only for the Gaussian map (Rademacher/sparse match its first two moments and
concentrate the same way); the JL bound is worst-case, so on a structured cloud far
smaller k already works (we show both the guaranteed k and the empirical k). All
randomness is seeded. `grid_table()` prints the distortion / threshold / families /
recall blocks that `RandomProjectionLaboratory.tsx` mirrors to the decimal.
`sklearn.random_projection` is the reference cross-check.

Run:  uv run --with numpy --with scipy --with scikit-learn python notebooks/johnson-lindenstrauss/johnson_lindenstrauss.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection

# Target-dimension grid the viz panels step through.
K_GRID: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)


# --------------------------------------------------------------------------- #
# Synthetic embeddings — a low-rank signal with a decaying spectrum plus an
# ambient noise floor, optionally split into topical clusters. The SAME generator
# the PCA topic uses, so the data-dependent-vs-oblivious comparison is apples to
# apples. A stand-in for the manifold a real encoder produces; NOT a trained encoder.
# --------------------------------------------------------------------------- #

def structured_embeddings(n: int, d: int, k: int, n_clusters: int = 1,
                          decay: float = 0.93, noise: float = 0.05,
                          cluster_sep: float = 2.5, seed: int = 0):
    """n embeddings in R^d on a k-dim latent subspace with a decaying spectrum.

    Latent coordinates have standard deviations decay^j, optionally offset into
    `n_clusters` topical groups, mapped through a random orthonormal d x k basis and
    perturbed by ambient Gaussian noise. Returns (X, labels).
    """
    rng = np.random.default_rng(seed)
    scales = decay ** np.arange(k)
    z = rng.standard_normal((n, k)) * scales
    if n_clusters > 1:
        labels = rng.integers(0, n_clusters, n)
        centers = rng.standard_normal((n_clusters, k)) * scales * cluster_sep
        z = z + centers[labels]
    else:
        labels = np.zeros(n, dtype=int)
    basis, _ = np.linalg.qr(rng.standard_normal((d, k)))
    X = z @ basis.T + noise * rng.standard_normal((n, d))
    return X, labels


# --------------------------------------------------------------------------- #
# The three random-projection families. Each returns an embedding map f: R^d -> R^k
# applied as Y = X @ A.T with A scaled so E||f(x)||^2 = ||x||^2.
# --------------------------------------------------------------------------- #

def _draw(gen: np.random.Generator, family: str, d: int, k: int, s: int = 3) -> np.ndarray:
    """Draw one k x d projection matrix of the named family from generator `gen`,
    scaled by 1/sqrt(k). Monte-Carlo sweeps MUST draw every matrix from a single `gen`
    stream: per-seed default_rng(s) over consecutive s leaves opening draws weakly
    correlated and inflates the sampled variance with d (a real NumPy gotcha)."""
    if family == "gaussian":
        return gen.standard_normal((k, d)) / np.sqrt(k)
    if family == "rademacher":
        return gen.choice((-1.0, 1.0), size=(k, d)) / np.sqrt(k)
    if family == "sparse":
        probs = (1.0 / (2 * s), 1.0 - 1.0 / s, 1.0 / (2 * s))
        vals = (-np.sqrt(s), 0.0, np.sqrt(s))
        return gen.choice(vals, size=(k, d), p=probs) / np.sqrt(k)
    raise ValueError(f"unknown projection family: {family}")


def gaussian_projection(d: int, k: int, seed: int = 0) -> np.ndarray:
    """k x d Gaussian projection matrix, scaled by 1/sqrt(k). Entries i.i.d. N(0,1)."""
    return _draw(np.random.default_rng(seed), "gaussian", d, k)


def rademacher_projection(d: int, k: int, seed: int = 0) -> np.ndarray:
    """k x d Rademacher projection: entries +/-1 with equal probability, scaled by
    1/sqrt(k). Database-friendly — no multiplications, just sign flips and adds."""
    return _draw(np.random.default_rng(seed), "rademacher", d, k)


def sparse_achlioptas_projection(d: int, k: int, s: int = 3, seed: int = 0) -> np.ndarray:
    """k x d sparse Achlioptas projection. Entries are sqrt(s) * {+1 w.p. 1/(2s),
    0 w.p. 1 - 1/s, -1 w.p. 1/(2s)}, scaled by 1/sqrt(k). With s=3, two thirds of the
    matrix is zero. E[entry^2] = 1 so squared norm is preserved in expectation."""
    return _draw(np.random.default_rng(seed), "sparse", d, k, s=s)


PROJECTORS = {
    "gaussian": gaussian_projection,
    "rademacher": rademacher_projection,
    "sparse": sparse_achlioptas_projection,
}


def project(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Apply a projection matrix A (k x d) to data X (n x d): returns n x k."""
    return X @ A.T


# --------------------------------------------------------------------------- #
# Distortion bookkeeping. The JL guarantee is on SQUARED distances:
#   (1 - eps) ||u - v||^2 <= ||f(u) - f(v)||^2 <= (1 + eps) ||u - v||^2.
# --------------------------------------------------------------------------- #

def pairwise_sq_distortion(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-pair ratio ||y_i - y_j||^2 / ||x_i - x_j||^2 over all C(n,2) pairs. A value
    of 1 is perfect preservation; the JL band is [1 - eps, 1 + eps]."""
    orig = pdist(X, "sqeuclidean")
    proj = pdist(Y, "sqeuclidean")
    return proj / orig


def jl_min_dim(n: int, eps: float) -> int:
    """The classical JL target dimension k >= 4 ln(n) / (eps^2/2 - eps^3/3). This is the
    worst-case guarantee; it matches sklearn.johnson_lindenstrauss_min_dim."""
    return int(np.ceil(4.0 * np.log(n) / (eps ** 2 / 2.0 - eps ** 3 / 3.0)))


def laurent_massart_upper(k: int, x: float) -> float:
    """The Laurent-Massart upper-tail threshold t such that P(chi^2_k/k - 1 >= t) <=
    exp(-x): t = 2 sqrt(x/k) + 2 x/k."""
    return 2.0 * np.sqrt(x / k) + 2.0 * x / k


# --------------------------------------------------------------------------- #
# Retrieval: nearest-neighbor recall@k after projection, random vs PCA.
# --------------------------------------------------------------------------- #

def _topk_neighbors(queries: np.ndarray, corpus: np.ndarray, topk: int) -> np.ndarray:
    d2 = cdist(queries, corpus, "sqeuclidean")
    return np.argpartition(d2, topk, axis=1)[:, :topk]


def _pca_components(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of the centered data (d x k) — the data-dependent
    projection PCA would use, for the oblivious-vs-dependent recall comparison."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def recall_after_projection(X: np.ndarray, queries: np.ndarray, kept_dims,
                            topk: int = 10, method: str = "gaussian",
                            seed: int = 0) -> dict[int, float]:
    """Mean recall@topk of the full-dimensional nearest neighbors after projecting
    corpus and queries to each kept dimension, by a random projection family or by PCA."""
    truth = _topk_neighbors(queries, X, topk)
    truth_sets = [set(row) for row in truth]
    out: dict[int, float] = {}
    for kd in kept_dims:
        if method == "pca":
            W = _pca_components(X, kd)
            Xp, Qp = X @ W, queries @ W
        else:
            A = PROJECTORS[method](X.shape[1], kd, seed=seed)
            Xp, Qp = project(X, A), project(queries, A)
        nn = _topk_neighbors(Qp, Xp, topk)
        recalls = [len(truth_sets[i].intersection(nn[i])) / topk for i in range(len(nn))]
        out[kd] = float(np.mean(recalls))
    return out


# --------------------------------------------------------------------------- #
# The showcase / finance dataset, and the grid table the viz mirrors.
# --------------------------------------------------------------------------- #

FINANCE_DIM = 1536
FINANCE_INTRINSIC_K = 48
FINANCE_CLUSTERS = 3
FINANCE_N = 500          # corpus size the distortion histogram is measured on
FINANCE_EPS = 0.2        # target distortion the guaranteed dimension is computed for
FINANCE_KEPT = 128       # the production target dimension the demo reports recall at


def finance_dataset():
    """The canonical synthetic financial-embedding cloud grid_table() and finance_demo()
    read from: 1536-d, low decaying-rank signal, three topical clusters, plus a held-out
    query set. SYNTHETIC low-rank-plus-noise, not a trained encoder."""
    X, labels = structured_embeddings(FINANCE_N, FINANCE_DIM, FINANCE_INTRINSIC_K,
                                      n_clusters=FINANCE_CLUSTERS, decay=0.97,
                                      noise=0.02, cluster_sep=1.2, seed=1)
    queries, _ = structured_embeddings(150, FINANCE_DIM, FINANCE_INTRINSIC_K,
                                       n_clusters=FINANCE_CLUSTERS, decay=0.97,
                                       noise=0.02, cluster_sep=1.2, seed=2)
    return X, labels, queries


def _distortion_summary(ratios: np.ndarray) -> dict:
    """Five-number-ish summary of a pairwise squared-distortion distribution."""
    return {
        "mean": float(ratios.mean()),
        "std": float(ratios.std()),
        "p01": float(np.percentile(ratios, 1)),
        "p99": float(np.percentile(ratios, 99)),
        "max_abs_dev": float(np.max(np.abs(ratios - 1.0))),
    }


def grid_table() -> dict[str, list]:
    """The numbers RandomProjectionLaboratory.tsx mirrors to the decimal. Deterministic.

    distortion : per-k Gaussian pairwise squared-distortion summary (band tightening).
    hist       : a histogram of the distortion ratios at two representative k values.
    threshold  : guaranteed JL dimension jl_min_dim(n, eps) over an eps grid.
    families   : distortion std for Gaussian vs Rademacher vs sparse at a fixed k.
    recall     : per-k recall@10 retained, random projection vs PCA.
    """
    X, labels, queries = finance_dataset()

    distortion, hist = [], {}
    HIST_BINS, HIST_RANGE = 30, (0.3, 1.7)     # the band the viz histogram panel renders
    for kd in K_GRID:
        A = gaussian_projection(FINANCE_DIM, kd, seed=10)
        ratios = pairwise_sq_distortion(X, project(X, A))
        distortion.append({"k": kd, **_distortion_summary(ratios)})
        counts, edges = np.histogram(ratios, bins=HIST_BINS, range=HIST_RANGE)
        hist[kd] = {"counts": [int(c) for c in counts],
                    "edges": [float(e) for e in edges]}

    threshold = [{"eps": round(eps, 2), "k": jl_min_dim(FINANCE_N, eps)}
                 for eps in (0.1, 0.15, 0.2, 0.3, 0.5)]

    families = []
    for name in ("gaussian", "rademacher", "sparse"):
        A = PROJECTORS[name](FINANCE_DIM, 64, seed=11)
        ratios = pairwise_sq_distortion(X, project(X, A))
        families.append({"family": name, **_distortion_summary(ratios)})

    rec_rand = recall_after_projection(X, queries, K_GRID, method="gaussian", seed=12)
    rec_pca = recall_after_projection(X, queries, K_GRID, method="pca")
    recall = [{"k": k, "rand": rec_rand[k], "pca": rec_pca[k]} for k in K_GRID]

    return {"distortion": distortion, "hist": hist, "threshold": threshold,
            "families": families, "recall": recall}


def finance_demo() -> dict:
    """Headline numbers: the guaranteed JL dimension for eps, the empirical distortion at
    the production target dimension, and recall@10 random vs PCA in R^1536."""
    X, labels, queries = finance_dataset()
    A = gaussian_projection(FINANCE_DIM, FINANCE_KEPT, seed=10)
    ratios = pairwise_sq_distortion(X, project(X, A))
    rec_rand = recall_after_projection(X, queries, [FINANCE_KEPT], method="gaussian", seed=12)[FINANCE_KEPT]
    rec_pca = recall_after_projection(X, queries, [FINANCE_KEPT], method="pca")[FINANCE_KEPT]
    out = {
        "dim": FINANCE_DIM,
        "n": FINANCE_N,
        "eps": FINANCE_EPS,
        "k_guaranteed": jl_min_dim(FINANCE_N, FINANCE_EPS),
        "kept": FINANCE_KEPT,
        "max_abs_dev_at_kept": float(np.max(np.abs(ratios - 1.0))),
        "mean_dev_at_kept": float(np.mean(np.abs(ratios - 1.0))),
        "recall_rand": rec_rand,
        "recall_pca": rec_pca,
    }
    n_pairs = FINANCE_N * (FINANCE_N - 1) // 2
    print(f"  {FINANCE_N} embeddings in R^{FINANCE_DIM} (SYNTHETIC low-rank-plus-noise, "
          f"not a trained encoder):")
    print(f"  JL guarantee for eps={FINANCE_EPS} (ALL {n_pairs} pairs): need k >= {out['k_guaranteed']} "
          f"(worst-case; depends on ln(n)={np.log(FINANCE_N):.2f}, NOT on d={FINANCE_DIM})")
    print(f"  at a practical k={FINANCE_KEPT}: the TYPICAL pair distorts only "
          f"{out['mean_dev_at_kept']*100:.1f}%, but the worst of {n_pairs} pairs distorts "
          f"{out['max_abs_dev_at_kept']*100:.1f}% — the union bound is real")
    print(f"  recall@10 retained at {FINANCE_KEPT} dims:  random projection "
          f"{out['recall_rand']*100:.1f}%   vs   PCA {out['recall_pca']*100:.1f}%")
    print("  -> JL preserves *typical* distances at modest k, but the worst-case guarantee "
          "needs k>>; and exact recall is the oblivious price PCA avoids.")
    return out


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_expected_norm_preservation() -> None:
    """E||f(x)||^2 = ||x||^2 for all three families: averaged over many random matrices,
    the squared-norm ratio is 1 (the unbiasedness every JL proof starts from)."""
    x = np.random.default_rng(0).standard_normal(300)
    sq = float(x @ x)
    for fam in ("gaussian", "rademacher", "sparse"):
        gen = np.random.default_rng({"gaussian": 1, "rademacher": 2, "sparse": 3}[fam])
        ratios = []
        for _ in range(600):
            fx = project(x, _draw(gen, fam, 300, 50))
            ratios.append(float(fx @ fx) / sq)
        assert abs(np.mean(ratios) - 1.0) < 0.02, f"{fam}: E||f(x)||^2 != ||x||^2"
    print("  [ok] norm preservation: E||f(x)||^2 = ||x||^2 for Gaussian, Rademacher, sparse")


def test_chi_square_distribution() -> None:
    """For the Gaussian map, k ||f(x)||^2 / ||x||^2 ~ chi^2_k exactly: empirical mean ~ k,
    variance ~ 2k, and the Laurent-Massart upper-tail bound is never violated."""
    d, k = 256, 64
    x = np.random.default_rng(1).standard_normal(d)
    sq = float(x @ x)
    gen = np.random.default_rng(7)
    vals = np.empty(6000)
    for i in range(6000):
        fx = project(x, _draw(gen, "gaussian", d, k))
        vals[i] = k * float(fx @ fx) / sq            # ~ chi^2_k exactly
    assert abs(vals.mean() - k) < 0.05 * k, f"chi^2 mean {vals.mean():.1f} != k={k}"
    assert abs(vals.var() - 2 * k) < 0.15 * (2 * k), f"chi^2 var {vals.var():.1f} != 2k={2*k}"
    ratio = vals / k                                   # ~ chi^2_k / k
    for xparam in (1.0, 2.0, 3.0):
        thresh = laurent_massart_upper(k, xparam)
        emp = float(np.mean(ratio - 1.0 >= thresh))
        assert emp <= np.exp(-xparam) + 0.01, \
            f"Laurent-Massart tail violated at x={xparam}: {emp:.4f} > {np.exp(-xparam):.4f}"
    print(f"  [ok] concentration: k||f(x)||^2/||x||^2 ~ chi^2_k (mean {vals.mean():.1f}, "
          f"var {vals.var():.1f}); Laurent-Massart tail holds")


def test_jl_lemma_holds() -> None:
    """At the guaranteed k = jl_min_dim(n, eps), a single Gaussian projection preserves
    ALL pairwise squared distances to (1 +/- eps) with high probability over seeds."""
    n, eps, d = 300, 0.3, 1200
    X, _ = structured_embeddings(n, d, 40, n_clusters=3, seed=2)
    k = jl_min_dim(n, eps)
    assert k < d, f"guaranteed k={k} not below ambient d={d} for this (n, eps)"
    successes = 0
    trials = 12
    for s in range(trials):
        ratios = pairwise_sq_distortion(X, project(X, gaussian_projection(d, k, seed=100 + s)))
        if np.max(np.abs(ratios - 1.0)) <= eps:
            successes += 1
    assert successes >= trials - 1, \
        f"JL lemma failed too often at k={k}: {successes}/{trials} within eps={eps}"
    print(f"  [ok] JL lemma: at guaranteed k={k}, all C({n},2) pairs within eps={eps} "
          f"in {successes}/{trials} seeds")


def test_dimension_independence() -> None:
    """The distortion of a FIXED vector, ||f(x)||^2/||x||^2 ~ chi^2_k/k, has no d in it:
    its spread is sqrt(2/k) whatever the ambient dimension. And the worst pairwise
    distortion over a cloud worsens with n (a union bound over more pairs), not with d."""
    k = 64
    target_std = np.sqrt(2.0 / k)                      # std of chi^2_k / k
    devs_by_d = []
    for d in (256, 1024, 4096):
        x = np.random.default_rng(3).standard_normal(d)
        sq = float(x @ x)
        gen = np.random.default_rng(20)
        ratios = np.empty(3000)
        for i in range(3000):
            fx = project(x, _draw(gen, "gaussian", d, k))
            ratios[i] = float(fx @ fx) / sq
        devs_by_d.append(ratios.std())
        assert abs(ratios.std() - target_std) < 0.12 * target_std, \
            f"d={d}: distortion std {ratios.std():.3f} != sqrt(2/k)={target_std:.3f}"
    spread = (max(devs_by_d) - min(devs_by_d)) / np.mean(devs_by_d)
    assert spread < 0.12, f"fixed-vector distortion std varies with d (spread {spread:.2f})"

    d = 1024
    max_devs = []
    for n in (100, 400, 1600):
        devs = []
        for s in range(4):
            X, _ = structured_embeddings(n, d, 40, n_clusters=3, seed=4 + s)
            ratios = pairwise_sq_distortion(X, project(X, gaussian_projection(d, k, seed=21 + s)))
            devs.append(float(np.max(np.abs(ratios - 1.0))))
        max_devs.append(float(np.mean(devs)))
    assert max_devs[0] <= max_devs[1] <= max_devs[2] + 1e-9, \
        f"mean max distortion not increasing in n: {max_devs}"
    print(f"  [ok] dimension independence: fixed-vector std ~ sqrt(2/k)={target_std:.3f} across d "
          f"({', '.join(f'{v:.3f}' for v in devs_by_d)}); worst pair grows with n "
          f"({', '.join(f'{v:.2f}' for v in max_devs)})")


def test_projection_families_agree() -> None:
    """Gaussian, Rademacher, and sparse Achlioptas concentrate at the same sqrt(2/k)
    rate — the equivalence that makes any of them a valid JL map. (Mean = 1 by
    unbiasedness is pinned separately; sparse's matrix-level mean fluctuates more on a
    low-rank cloud, an honest cost of zeroing two thirds of the matrix.)"""
    d, k = 800, 64
    X, _ = structured_embeddings(400, d, 40, n_clusters=3, seed=5)
    orig = pdist(X, "sqeuclidean")
    stats = {}
    for fam in ("gaussian", "rademacher", "sparse"):
        gen = np.random.default_rng(30)
        pooled = np.concatenate([pdist(project(X, _draw(gen, fam, d, k)), "sqeuclidean") / orig
                                 for _ in range(20)])
        stats[fam] = (pooled.mean(), pooled.std())
    g_std = stats["gaussian"][1]
    for fam in ("gaussian", "rademacher", "sparse"):
        m, sd = stats[fam]
        assert abs(m - 1.0) < 0.05, f"{fam} mean distortion {m:.4f} not ~1"
        assert abs(sd - g_std) < 0.20 * g_std, f"{fam} concentration spread {sd:.3f} != gaussian {g_std:.3f}"
    print(f"  [ok] families concentrate alike: gaussian/rademacher/sparse std "
          f"{', '.join(f'{stats[n][1]:.3f}' for n in ('gaussian','rademacher','sparse'))}")


def test_recall_after_projection() -> None:
    """Data-oblivious vs data-dependent: random-projection recall@10 climbs with k but
    stays far below data-dependent PCA at every k — preserving distances to +/-eps is not
    enough to preserve the exact top-10 on a tightly-clustered low-rank cloud."""
    X, _ = structured_embeddings(900, 512, 40, n_clusters=3, seed=6)
    queries, _ = structured_embeddings(120, 512, 40, n_clusters=3, seed=7)
    kept = (4, 16, 64, 256)
    rand = recall_after_projection(X, queries, kept, method="gaussian", seed=8)
    pca = recall_after_projection(X, queries, kept, method="pca")
    for k in kept:
        assert pca[k] >= rand[k], f"PCA recall below random at k={k} (data-dependent should win)"
    assert pca[64] - rand[64] > 0.3, "PCA should dominate random at moderate k"
    assert rand[256] > rand[4] + 0.1, f"random recall did not climb with k: {rand}"
    assert pca[256] > 0.85, "PCA recovers retrieval by k=256"
    print(f"  [ok] retrieval: PCA >> random at every k (e.g. {pca[64]*100:.0f}% vs "
          f"{rand[64]*100:.0f}% at k=64); random climbs with k but pays the oblivious price")


def test_sklearn_crosscheck() -> None:
    """Our jl_min_dim matches sklearn's johnson_lindenstrauss_min_dim, and sklearn's
    GaussianRandomProjection achieves the same distortion distribution as ours."""
    for n, eps in ((100, 0.1), (500, 0.2), (10000, 0.3)):
        # Same formula; we ceil (the true minimum that suffices), sklearn truncates, so
        # they agree to within one dimension.
        assert abs(jl_min_dim(n, eps) - int(johnson_lindenstrauss_min_dim(n, eps=eps))) <= 1, \
            f"jl_min_dim disagrees with sklearn at n={n}, eps={eps}"
    X, _ = structured_embeddings(400, 800, 40, n_clusters=3, seed=9)
    ours = pairwise_sq_distortion(X, project(X, gaussian_projection(800, 64, seed=40)))
    grp = GaussianRandomProjection(n_components=64, random_state=40).fit_transform(X)
    theirs = pdist(grp, "sqeuclidean") / pdist(X, "sqeuclidean")
    assert abs(ours.std() - theirs.std()) < 0.25 * ours.std(), \
        "our distortion spread disagrees with sklearn GaussianRandomProjection"
    print("  [ok] cross-check: jl_min_dim = sklearn; distortion matches GaussianRandomProjection")


def test_finance_distortion() -> None:
    """The 1536-d finance cloud: the guaranteed JL dimension is large (worst-case) but a
    far smaller production k already holds every pair within eps. Recall is the honest
    caveat — data-dependent PCA dominates the oblivious random projection at the same k."""
    f = finance_demo()
    assert f["k_guaranteed"] > f["kept"], "guaranteed k should exceed the practical kept dim"
    assert f["mean_dev_at_kept"] < f["eps"], \
        f"typical pair distortion {f['mean_dev_at_kept']:.3f} should be within eps={f['eps']} at the practical k"
    assert f["max_abs_dev_at_kept"] > f["eps"], \
        f"worst-pair distortion {f['max_abs_dev_at_kept']:.3f} should exceed eps at the practical k (union bound)"
    assert f["recall_pca"] > f["recall_rand"] + 0.3, "PCA should dominate random recall at the kept dim"
    assert f["recall_pca"] > 0.85, f"PCA recall too low: {f['recall_pca']:.3f}"
    print("  [ok] finance: guaranteed k >> practical k; typical pair < eps < worst pair; PCA recall >> oblivious")


if __name__ == "__main__":
    print("Johnson-Lindenstrauss / random projection verification harness")
    test_expected_norm_preservation()
    test_chi_square_distribution()
    test_jl_lemma_holds()
    test_dimension_independence()
    test_projection_families_agree()
    test_recall_after_projection()
    test_sklearn_crosscheck()
    tbl = grid_table()
    print("Grid table (mirrored by RandomProjectionLaboratory.tsx):")
    print(f"  {'k':>6}{'mean':>9}{'std':>9}{'p01':>9}{'p99':>9}{'maxdev':>9}"
          f"{'recall_rand':>13}{'recall_pca':>12}")
    for drow, rrow in zip(tbl["distortion"], tbl["recall"]):
        print(f"  {drow['k']:>6}{drow['mean']:>9.4f}{drow['std']:>9.4f}"
              f"{drow['p01']:>9.4f}{drow['p99']:>9.4f}{drow['max_abs_dev']:>9.4f}"
              f"{rrow['rand']:>13.4f}{rrow['pca']:>12.4f}")
    print("  guaranteed JL dimension by eps:",
          ", ".join(f"eps={t['eps']}->{t['k']}" for t in tbl["threshold"]))
    print("  family std (k=64):",
          ", ".join(f"{r['family']} {r['std']:.4f}" for r in tbl["families"]))
    print("Finance demo:")
    test_finance_distortion()
    print("All checks passed.")
