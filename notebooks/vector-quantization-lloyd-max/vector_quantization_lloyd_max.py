"""Vector quantization and the Lloyd–Max optimality conditions — the reference
implementation for the formalRAG `vector-quantization-lloyd-max` topic.

A retriever that stores millions of d-dimensional float32 embeddings pays for every
dimension twice: once in memory, once in per-query distance work. Vector quantization
attacks the memory directly — replace each embedding by the index of the nearest entry
in a small learned codebook, so a vector costs log2(k) bits instead of 32*d. This module
establishes, and verifies, what makes such a quantizer *optimal*, then the honest twist
that "optimal" means only *locally* optimal:

  1. NEAREST-NEIGHBOR CONDITION (optimal encoder). For a fixed codebook, the distortion-
     minimizing encoder assigns each point to its nearest codeword, so the optimal cells
     are the Voronoi regions of the codebook. (`assign`)
  2. CENTROID CONDITION (optimal decoder). For a fixed partition, the distortion-
     minimizing codeword is the conditional mean of its cell, c_i = E[X | X in R_i]. For
     a finite sample that is the cell's sample mean. (`update`)
  3. LLOYD = MONOTONE DESCENT TO A FIXED POINT. Alternating the two conditions never
     increases distortion (D^0 >= D^1 >= ... >= 0), so it converges; on a finite sample it
     reaches a both-conditions fixed point in finitely many steps — a LOCAL, not global,
     optimum. (`lloyd`)
  4. k-MEANS IS LLOYD. Under the empirical measure (uniform over a finite sample) and
     squared-Euclidean distortion, Lloyd's algorithm IS k-means and the distortion IS the
     within-cluster sum of squares. (`kmeans_objective`)

Then the curse of dimensionality, made quantitative: ZADOR'S LAW D*(k) ~ C(d) k^(-2/d).
We verify the *scaling exponent* on low-dimensional uniform sources where the high-rate
regime is reachable (`test_zador_scaling`) — NOT the constant C(d), which is known in
closed form only in special cases. The finance demo then shows the same exponent biting:
on a high-effective-rank cloud, distortion falls only slowly with the codebook size.

Every pedagogical claim is an `assert` below. `viz_constants()` prints the 2-D toy cloud,
the seed codebooks, and the distortion histories that `VectorQuantizationLaboratory.tsx`
mirrors to the decimal. The viz recomputes Lloyd in TS for interactivity, but the seed
cloud, the seed codebooks, and the converged distortions are baked from here — never
re-sampled in TS. `scipy.cluster.vq.kmeans2` is the reference cross-check.

NUMERICAL NOTES: assignment ties (cell boundaries) are broken by lowest index, matching
np.argmin and the TS `Math.min` scan; an empty cell (no points nearest a codeword) has no
mean, so we repair it deterministically by the FURTHEST-POINT rule — reseed the orphaned
codeword onto the worst-served point — which keeps the distortion monotone and the run
reproducible. All randomness flows through np.random.default_rng(seed).

Run:  uv run --with numpy --with scipy python notebooks/vector-quantization-lloyd-max/vector_quantization_lloyd_max.py
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

# Codebook sizes the rate–distortion panel / finance demo step through.
RD_K_GRID: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)


# --------------------------------------------------------------------------- #
# Synthetic sources. The high-dimensional finance cloud reuses the low-rank-
# plus-noise generator from the PCA/geometry topics (the recurring thread); the
# 2-D toy cloud feeds the interactive laboratory. Both SYNTHETIC, not encoders.
# --------------------------------------------------------------------------- #

def structured_embeddings(n: int, d: int, k: int, n_clusters: int = 1,
                          decay: float = 0.93, noise: float = 0.05,
                          cluster_sep: float = 2.5, seed: int = 0):
    """n embeddings in R^d on a k-dim latent subspace with a decaying spectrum,
    optionally offset into `n_clusters` topical groups. Returns (X, labels). The same
    generator the PCA topic uses, so the finance thread is literally the same cloud."""
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


def toy_cloud_2d(seed: int = 0):
    """Sixty points in [0,10]^2 as FOUR Gaussian blobs near the corners — the cloud the
    laboratory draws and viz_constants() bakes. Four blobs with the default k=3 is the
    classic local-optimum demonstration: Lloyd must merge two blobs into one cell, and
    merging an ADJACENT pair (the global optimum) or a DIAGONAL pair (a higher-distortion
    local optimum) depends on the initialization. Returns (X (60,2), labels (60,))."""
    rng = np.random.default_rng(seed)
    centers = np.array([[2.3, 7.7], [7.7, 7.7], [7.7, 2.3], [2.3, 2.3]])
    spread = 0.8
    per = 15
    pts, labs = [], []
    for c in range(4):
        pts.append(centers[c] + spread * rng.standard_normal((per, 2)))
        labs.append(np.full(per, c))
    X = np.clip(np.vstack(pts), 0.0, 10.0)
    return X, np.concatenate(labs)


# --------------------------------------------------------------------------- #
# The two Lloyd–Max half-steps and the alternating algorithm.
# --------------------------------------------------------------------------- #

def assign(X: np.ndarray, C: np.ndarray):
    """E-step / nearest-neighbor condition. Returns (labels (n,), distortion float),
    where labels[i] = argmin_j ||x_i - c_j||^2 (np.argmin -> first minimum, i.e. the
    lowest-index tie-break the viz mirrors) and distortion = sum_i min_j ||x_i - c_j||^2."""
    if X.shape[0] == 0:
        raise ValueError("empty point set")
    d2 = cdist(X, C, "sqeuclidean")
    labels = d2.argmin(axis=1)
    distortion = float(d2[np.arange(X.shape[0]), labels].sum())
    return labels, distortion


def within_cluster_sse(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    """Distortion D = sum_i ||x_i - c_{labels_i}||^2 — the k-means objective."""
    diff = X - C[labels]
    return float(np.sum(diff * diff))


def update(X: np.ndarray, labels: np.ndarray, k: int, C_prev: np.ndarray) -> np.ndarray:
    """M-step / centroid condition. Returns C (k,d) with c_j = mean of the points in
    cell j. EMPTY-CELL GUARD: a codeword that captured no points has no mean (a 0/0),
    so it is reseeded onto the worst-served point — the data point with the largest
    squared distance to its own current codeword (furthest-point repair). Deterministic
    (so TS can mirror it) and non-increasing in distortion (so monotonicity survives)."""
    n, d = X.shape
    C = np.zeros((k, d))
    counts = np.bincount(labels, minlength=k)
    for j in range(k):
        if counts[j] > 0:
            C[j] = X[labels == j].mean(axis=0)
    empty = np.where(counts == 0)[0]
    if empty.size > 0:
        resid = np.sum((X - C_prev[labels]) ** 2, axis=1)   # per-point distortion
        order = np.argsort(resid)[::-1]                      # worst-served first
        taken: set[int] = set()
        ptr = 0
        for j in empty:
            while ptr < n and int(order[ptr]) in taken:
                ptr += 1
            if ptr < n:
                idx = int(order[ptr])
                C[j] = X[idx]
                taken.add(idx)
                ptr += 1
            else:                                            # fewer points than empties
                C[j] = C_prev[j]
    return C


def lloyd_step(X: np.ndarray, C: np.ndarray, k: int):
    """One full Lloyd iteration: assign then update then re-assign, so the reported
    distortion is the objective at the post-update codebook. Returns
    (C_next, labels, distortion)."""
    labels, _ = assign(X, C)
    C_next = update(X, labels, k, C)
    labels, distortion = assign(X, C_next)
    return C_next, labels, distortion


def lloyd(X: np.ndarray, k: int, C0: np.ndarray, max_iter: int = 100,
          tol: float = 1e-9):
    """Iterate the two conditions from initialization C0 to a fixed point. Returns
    (C, labels, history) where history[t] is the distortion of the codebook at the start
    of iteration t (post-assignment) — a monotone non-increasing sequence. Stops when the
    codewords stop moving (max shift < tol) or after max_iter. Guards n==0, k<1, k>n."""
    n = X.shape[0]
    if n == 0:
        raise ValueError("empty point set")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) exceeds n_points ({n}); no valid k-cell partition")
    C = np.array(C0, dtype=float).copy()
    if C.shape[0] != k:
        raise ValueError(f"initial codebook has {C.shape[0]} codewords, expected k={k}")
    if C.shape[1] != X.shape[1]:
        raise ValueError(f"dimension mismatch: X has d={X.shape[1]}, C0 has d={C.shape[1]}")
    labels, dist = assign(X, C)
    history = [dist]
    for _ in range(max_iter):
        C_new = update(X, labels, k, C)
        shift = float(np.max(np.linalg.norm(C_new - C, axis=1)))
        C = C_new
        labels, dist = assign(X, C)
        history.append(dist)
        if shift < tol:
            break
    return C, labels, history


def kmeans_objective(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    """The k-means objective — identically the within-cluster SSE, which is the VQ
    distortion under the empirical measure. Same function, named for the equivalence."""
    return within_cluster_sse(X, C, labels)


# --------------------------------------------------------------------------- #
# Initializations and the local-optimum sweep.
# --------------------------------------------------------------------------- #

def random_init(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Forgy initialization: k distinct data points as the initial codebook."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].copy()


def kmeans_pp_init(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-means++ seeding: D^2-weighted sampling spreads the initial codewords out,
    giving an O(log k) expected-approximation guarantee — a heuristic, not a fix."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    C = np.empty((k, d))
    C[0] = X[rng.integers(n)]
    d2 = np.sum((X - C[0]) ** 2, axis=1)
    for j in range(1, k):
        total = d2.sum()
        if total <= 0:                                   # all remaining points coincide
            C[j] = X[rng.integers(n)]
        else:
            C[j] = X[rng.choice(n, p=d2 / total)]
        d2 = np.minimum(d2, np.sum((X - C[j]) ** 2, axis=1))
    return C


def lloyd_multi_seed(X: np.ndarray, k: int, n_seeds: int = 12, init: str = "random",
                     base_seed: int = 0, max_iter: int = 100):
    """Run Lloyd from n_seeds initializations; return a list of (final_distortion, iters).
    The spread of final distortions IS the local-optimum / initialization sensitivity."""
    out = []
    for s in range(n_seeds):
        C0 = (kmeans_pp_init(X, k, seed=base_seed + s) if init == "kmeans++"
              else random_init(X, k, seed=base_seed + s))
        _, _, hist = lloyd(X, k, C0, max_iter=max_iter)
        out.append((hist[-1], len(hist)))
    return out


def best_codebook(X: np.ndarray, k: int, seed: int = 0, restarts: int = 3):
    """Train a codebook by Lloyd with k-means++ restarts, keeping the lowest-distortion
    run. Returns (distortion, C, labels)."""
    best = None
    for r in range(restarts):
        C0 = kmeans_pp_init(X, k, seed=seed + r)
        C, labels, hist = lloyd(X, k, C0)
        if best is None or hist[-1] < best[0]:
            best = (hist[-1], C, labels)
    return best


# --------------------------------------------------------------------------- #
# Quantization as compression, and retrieval recall under the codebook.
# --------------------------------------------------------------------------- #

def quantize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Replace each row of X by its nearest codeword — the lossy reconstruction Q(X)."""
    labels, _ = assign(X, C)
    return C[labels]


def _topk(queries: np.ndarray, corpus: np.ndarray, topk: int) -> np.ndarray:
    d2 = cdist(queries, corpus, "sqeuclidean")
    return np.argpartition(d2, topk, axis=1)[:, :topk]


def recall_at_k_quantized(X, queries, C, topk: int = 10) -> float:
    """Mean recall@topk when the corpus is replaced by its codeword reconstruction Q(X)
    (queries kept exact) versus the exact nearest neighbors — a symmetric-distance proxy
    for how much retrieval the codebook preserves."""
    truth = [set(r) for r in _topk(queries, X, topk)]
    approx = _topk(queries, quantize(X, C), topk)
    return float(np.mean([len(truth[i] & set(approx[i])) / topk for i in range(len(approx))]))


# --------------------------------------------------------------------------- #
# The finance dataset and the rate–distortion table.
# --------------------------------------------------------------------------- #

FINANCE_DIM = 256
FINANCE_INTRINSIC_K = 16
FINANCE_CLUSTERS = 6
FINANCE_N = 800


def finance_dataset():
    """The synthetic financial-document cloud the codebook is trained on: 256-d, low
    decaying-rank signal, six topical groups (rates / credit / FX / equity / macro /
    earnings), plus a held-out query set. SYNTHETIC low-rank-plus-noise, not a trained
    encoder. Returns (X, labels, queries)."""
    X, labels = structured_embeddings(FINANCE_N, FINANCE_DIM, FINANCE_INTRINSIC_K,
                                      n_clusters=FINANCE_CLUSTERS, decay=0.94,
                                      noise=0.03, cluster_sep=1.5, seed=1)
    queries, _ = structured_embeddings(120, FINANCE_DIM, FINANCE_INTRINSIC_K,
                                       n_clusters=FINANCE_CLUSTERS, decay=0.94,
                                       noise=0.03, cluster_sep=1.5, seed=2)
    return X, labels, queries


def finance_rate_distortion(k_grid=RD_K_GRID, seed: int = 7):
    """For each codebook size k: rate log2(k) bits/vector, distortion (within-cluster SSE),
    distortion per vector, the normalized D/D0 against the single-codeword baseline, and
    recall@10 retained. Returns (rows, baseline_D0)."""
    X, _, queries = finance_dataset()
    n = X.shape[0]
    mean_vec = X.mean(axis=0, keepdims=True)
    d0 = float(np.sum((X - mean_vec) ** 2))             # one-codeword distortion = total SS
    rows = []
    for k in k_grid:
        dist, C, _ = best_codebook(X, k, seed=seed, restarts=2)
        rows.append({
            "k": k,
            "bits": float(np.log2(k)),
            "distortion": dist,
            "per_vector": dist / n,
            "normalized": dist / d0,
            "recall10": recall_at_k_quantized(X, queries, C, topk=10),
        })
    return rows, d0


def finance_demo() -> dict:
    """Headline rate–distortion numbers for the finance codebook, plus the compression
    framing (codeword index vs raw float32 vector)."""
    rows, d0 = finance_rate_distortion()
    raw_bits = FINANCE_DIM * 32
    top = rows[-1]
    print(f"  {FINANCE_CLUSTERS} topical groups of embeddings in R^{FINANCE_DIM} "
          f"(SYNTHETIC low-rank-plus-noise, not a trained encoder):")
    print(f"  raw vector = {FINANCE_DIM} x float32 = {raw_bits} bits; "
          f"codeword index at k={top['k']} = {top['bits']:.0f} bits "
          f"-> {raw_bits / top['bits']:.0f}x smaller code")
    print(f"  {'k':>5}{'bits':>7}{'D/vector':>12}{'D/D0':>9}{'recall@10':>11}")
    for r in rows:
        print(f"  {r['k']:>5}{r['bits']:>7.0f}{r['per_vector']:>12.4f}"
              f"{r['normalized']:>9.4f}{r['recall10']:>11.3f}")
    print("  -> distortion falls with k but SLOWLY (the cloud's high effective rank is "
          "Zador's d in k^(-2/d)); a single full-space codebook is the wall product "
          "quantization breaks by splitting the vector into subspaces.")
    return {"rows": rows, "d0": d0, "raw_bits": raw_bits}


# --------------------------------------------------------------------------- #
# Zador's law: the k^(-2/d) scaling, verified where the high-rate regime is reachable.
# --------------------------------------------------------------------------- #

def uniform_quantizer_distortion(d: int, k: int, n: int = 4000, seed: int = 0,
                                 restarts: int = 4) -> float:
    """Per-point distortion of the best (over restarts) k-codeword quantizer of n uniform
    points in [0,1]^d. Uniform sources are where Zador's high-rate asymptotics are clean."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    best = np.inf
    for r in range(restarts):
        C0 = kmeans_pp_init(X, k, seed=seed * 131 + r)
        _, _, hist = lloyd(X, k, C0, max_iter=200)
        best = min(best, hist[-1] / n)
    return float(best)


def zador_slope(d: int, ks=(8, 16, 32, 64, 128), n: int = 4000, seed: int = 0) -> float:
    """Least-squares slope of log(distortion) vs log(k) for a uniform source in R^d. Zador
    predicts -2/d. Returns the fitted slope."""
    logd = np.array([np.log(uniform_quantizer_distortion(d, k, n=n, seed=seed)) for k in ks])
    logk = np.log(np.array(ks, dtype=float))
    return float(np.polyfit(logk, logd, 1)[0])


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_distortion_monotone() -> None:
    """Across every Lloyd iteration the distortion is non-increasing (Theorem 3) — on the
    toy cloud and on the high-dimensional finance cloud, from several initializations."""
    Xt, _ = toy_cloud_2d()
    Xf, _, _ = finance_dataset()
    for X, k, init in [(Xt, 3, "random"), (Xt, 5, "kmeans++"),
                       (Xf, 16, "kmeans++"), (Xf, 32, "random")]:
        C0 = (kmeans_pp_init(X, k, seed=3) if init == "kmeans++"
              else random_init(X, k, seed=3))
        _, _, hist = lloyd(X, k, C0)
        h = np.array(hist)
        assert np.all(np.diff(h) <= 1e-9 * (h[0] + 1.0)), \
            f"distortion not monotone for k={k}, init={init}: {hist}"
    print("  [ok] Lloyd monotonically decreases distortion to a fixed point")


def test_convergence_fixed_point() -> None:
    """At convergence one more assignment changes no labels and one more update moves no
    codeword (within tol): the returned configuration satisfies BOTH conditions at once."""
    X, _, _ = finance_dataset()
    C, labels, _ = lloyd(X, 24, kmeans_pp_init(X, 24, seed=5), max_iter=200, tol=1e-12)
    labels2, _ = assign(X, C)
    assert np.array_equal(labels, labels2), "assignment not stable at the fixed point"
    C2 = update(X, labels, 24, C)
    assert float(np.max(np.linalg.norm(C2 - C, axis=1))) < 1e-9, \
        "codewords still move at the claimed fixed point"
    print("  [ok] converges to a both-conditions fixed point (labels + codewords stable)")


def test_nearest_neighbor_condition() -> None:
    """At convergence every point is assigned to its NEAREST codeword: no reassignment of
    any single point lowers its distortion (Theorem 1 / the encoder is at its optimum)."""
    X, _, _ = finance_dataset()
    C, labels, _ = lloyd(X, 20, kmeans_pp_init(X, 20, seed=6), max_iter=200, tol=1e-12)
    d2 = cdist(X, C, "sqeuclidean")
    own = d2[np.arange(X.shape[0]), labels]
    assert np.all(own <= d2.min(axis=1) + 1e-9), "a point is not assigned to its nearest codeword"
    print("  [ok] nearest-neighbor condition: optimal encoder is Voronoi assignment")


def test_centroid_condition() -> None:
    """At convergence each (non-empty) codeword EQUALS the sample mean of its cell to
    machine precision: no codeword move lowers distortion (Theorem 2 / decoder optimum)."""
    X, _, _ = finance_dataset()
    C, labels, _ = lloyd(X, 18, kmeans_pp_init(X, 18, seed=7), max_iter=200, tol=1e-12)
    for j in range(C.shape[0]):
        members = X[labels == j]
        if len(members) > 0:
            assert np.allclose(C[j], members.mean(axis=0), atol=1e-9), \
                f"codeword {j} is not its cell's conditional mean"
    print("  [ok] centroid condition: optimal codeword is the cell's conditional mean")


def test_kmeans_objective_equals_sse() -> None:
    """The k-means objective IS the within-cluster SSE IS the VQ distortion — identically,
    not just numerically — and equals the assign()-reported distortion at any codebook."""
    X, _, _ = finance_dataset()
    C, labels, hist = lloyd(X, 12, kmeans_pp_init(X, 12, seed=8))
    assert kmeans_objective(X, C, labels) == within_cluster_sse(X, C, labels), \
        "k-means objective != within-cluster SSE"
    _, dist = assign(X, C)
    assert abs(dist - within_cluster_sse(X, C, labels)) < 1e-6, \
        "assign() distortion != SSE at the converged labels"
    print("  [ok] k-means objective = within-cluster SSE = VQ distortion")


def test_local_optima_sensitivity() -> None:
    """Different initializations converge to DIFFERENT final distortions — Lloyd finds a
    LOCAL optimum — and k-means++ seeding gives a lower mean final than random seeding."""
    X, _, _ = finance_dataset()
    rand = lloyd_multi_seed(X, 32, n_seeds=12, init="random", base_seed=0)
    kpp = lloyd_multi_seed(X, 32, n_seeds=12, init="kmeans++", base_seed=0)
    rand_finals = np.array([d for d, _ in rand])
    kpp_finals = np.array([d for d, _ in kpp])
    assert rand_finals.std() > 0, "all random inits reached the same optimum (no spread)"
    assert rand_finals.max() > rand_finals.min() + 1e-6, "no local-optimum spread"
    assert kpp_finals.mean() <= rand_finals.mean() + 1e-9, \
        "k-means++ did not improve on random initialization"
    print(f"  [ok] local optima: random finals spread "
          f"[{rand_finals.min():.1f}, {rand_finals.max():.1f}], "
          f"k-means++ mean {kpp_finals.mean():.1f} <= random mean {rand_finals.mean():.1f}")


def test_toy_local_optima() -> None:
    """The baked toy-cloud seeds reach three DISTINCT distortions — global < near-miss <<
    diagonal merge — so the laboratory's 'Reseed' story (init determines the optimum) is
    true for the exact cloud the viz bakes, not just asserted in prose."""
    X, _ = toy_cloud_2d()
    finals = []
    for C0 in (SEED_DEFAULT, *SEED_ALTERNATES):
        _, _, hist = lloyd(X, C0.shape[0], C0, tol=1e-12)
        finals.append(hist[-1])
    d_default, d_near, d_diag = finals
    assert d_default < d_near < d_diag, f"toy seeds not strictly ordered: {finals}"
    assert d_diag > 1.5 * d_default, "diagonal merge not dramatically worse than the global optimum"
    print(f"  [ok] toy local optima: global {d_default:.1f} < near-miss {d_near:.1f} "
          f"<< diagonal {d_diag:.1f} (Reseed story holds on the baked cloud)")


def test_empty_cluster_guard() -> None:
    """A codebook that orphans a codeword neither divides by zero nor increases distortion:
    the furthest-point repair returns finite codewords and the run stays monotone."""
    X, _ = toy_cloud_2d()
    # Force an empty cell: place a 4th codeword far outside the cloud so no point is nearest.
    C0 = np.array([[2.3, 7.7], [7.7, 7.7], [5.0, 2.3], [100.0, 100.0]])
    labels, _ = assign(X, C0)
    assert np.bincount(labels, minlength=4)[3] == 0, "test setup failed to orphan a codeword"
    C1 = update(X, labels, 4, C0)
    assert np.all(np.isfinite(C1)), "empty-cell repair produced non-finite codewords"
    _, _, hist = lloyd(X, 4, C0)
    h = np.array(hist)
    assert np.all(np.diff(h) <= 1e-9 * (h[0] + 1.0)), "empty-cell repair broke monotonicity"
    print("  [ok] empty-cell guard: furthest-point repair is finite and non-increasing")


def test_zador_scaling() -> None:
    """The optimal-quantizer distortion scales as k^(-2/d): on uniform sources the fitted
    log-log slope matches -2/d (the EXPONENT, not Zador's constant) for d = 1 and d = 2."""
    s1 = zador_slope(1, ks=(8, 16, 32, 64, 128, 256), seed=0)
    s2 = zador_slope(2, ks=(8, 16, 32, 64, 128, 256), seed=0)
    assert abs(s1 - (-2.0)) < 0.25, f"d=1 slope {s1:.3f} not near -2 (Zador k^(-2))"
    assert abs(s2 - (-1.0)) < 0.25, f"d=2 slope {s2:.3f} not near -1 (Zador k^(-1))"
    print(f"  [ok] Zador scaling: slope(d=1)={s1:.3f}~-2, slope(d=2)={s2:.3f}~-1 (k^(-2/d))")


def test_rate_distortion_monotone() -> None:
    """The finance codebook distortion is non-increasing in the rate log2(k): a larger
    codebook never raises distortion (refinement), and recall@10 is non-decreasing."""
    rows, _ = finance_rate_distortion()
    dist = np.array([r["distortion"] for r in rows])
    rec = np.array([r["recall10"] for r in rows])
    assert np.all(np.diff(dist) <= 1e-6 * dist[0]), f"distortion not monotone in k: {dist}"
    assert rec[-1] >= rec[0] - 1e-9, "recall@10 fell as the codebook grew"
    print(f"  [ok] rate–distortion: D falls {dist[0]:.1f}->{dist[-1]:.1f}, "
          f"recall@10 rises {rec[0]:.2f}->{rec[-1]:.2f} as bits grow")


def validate_against_kmeans2() -> None:
    """From the SAME initialization our Lloyd reaches the same distortion as
    scipy.cluster.vq.kmeans2 — Lloyd's algorithm IS k-means / generalized Lloyd."""
    X, _, _ = finance_dataset()
    C0 = kmeans_pp_init(X, 16, seed=11)
    C_ours, labels_ours, _ = lloyd(X, 16, C0, max_iter=300, tol=1e-12)
    C_sp, labels_sp = kmeans2(X, C0, iter=300, minit="matrix")
    d_ours = within_cluster_sse(X, C_ours, labels_ours)
    d_sp = within_cluster_sse(X, C_sp, labels_sp)
    assert abs(d_ours - d_sp) <= 1e-6 * d_ours, \
        f"our distortion {d_ours:.4f} != scipy kmeans2 {d_sp:.4f}"
    print(f"  [ok] cross-check: distortion {d_ours:.2f} matches scipy.cluster.vq.kmeans2")


# --------------------------------------------------------------------------- #
# Viz constants — the exact numbers VectorQuantizationLaboratory.tsx mirrors.
# --------------------------------------------------------------------------- #

# Seed codebooks for the 2-D toy cloud (four corner blobs, k=3). The "Reseed" story cycles
# through three visibly different outcomes of the SAME data under different initializations:
#   DEFAULT  -> 285.2  the GLOBAL optimum (best adjacent merge), a clear multi-step descent
#   ALT1     -> 313.9  a NEAR-MISS local optimum (a different adjacent merge)
#   ALT2     -> 541.1  a badly-stuck DIAGONAL merge, ~90% worse than the global
# Verified by viz_constants() / test_toy_local_optima before baking into the .tsx.
SEED_DEFAULT = np.array([[4.6, 5.2], [5.4, 4.8], [5.0, 5.6]])   # clumped center -> global opt
SEED_ALTERNATES = (
    np.array([[2.0, 6.0], [3.0, 5.0], [2.5, 4.0]]),    # clumped lower-left -> near-miss local
    np.array([[5.0, 5.0], [2.3, 7.7], [7.7, 2.3]]),    # center codeword merges a diagonal pair
)


def viz_constants() -> None:
    """Print the toy cloud, the seed codebooks, and the distortion histories the
    laboratory bakes. The TS lab recomputes Lloyd live, but these seed inputs and the
    converged endpoints are the contract — mirrored to the decimal, never re-sampled."""
    X, labels = toy_cloud_2d()
    print(f"  INITIAL_POINTS ({X.shape[0]} pts in [0,10]^2), [x, y, blob]:")
    print("   " + "; ".join(f"[{p[0]:.4f},{p[1]:.4f},{labels[i]}]" for i, p in enumerate(X[:6]))
          + f"; ... ({X.shape[0]} total)")
    seeds = [("DEFAULT", SEED_DEFAULT)] + [(f"ALT{i+1}", s) for i, s in enumerate(SEED_ALTERNATES)]
    for name, C0 in seeds:
        C, lab, hist = lloyd(X, C0.shape[0], C0, tol=1e-12)
        print(f"  {name} seed {np.round(C0, 4).tolist()}")
        print(f"    history {[round(h, 4) for h in hist]}")
        print(f"    converged D = {hist[-1]:.4f} in {len(hist)} iters")


def _print_grid() -> None:
    rows, d0 = finance_rate_distortion()
    print(f"  baseline D0 (single codeword) = {d0:.1f}")
    print(f"  {'k':>5}{'bits':>7}{'D':>12}{'D/D0':>9}{'recall@10':>11}")
    for r in rows:
        print(f"  {r['k']:>5}{r['bits']:>7.0f}{r['distortion']:>12.1f}"
              f"{r['normalized']:>9.4f}{r['recall10']:>11.3f}")


if __name__ == "__main__":
    print("Vector quantization / Lloyd–Max verification harness")
    test_distortion_monotone()
    test_convergence_fixed_point()
    test_nearest_neighbor_condition()
    test_centroid_condition()
    test_kmeans_objective_equals_sse()
    test_local_optima_sensitivity()
    test_toy_local_optima()
    test_empty_cluster_guard()
    test_zador_scaling()
    test_rate_distortion_monotone()
    validate_against_kmeans2()
    print("Finance demo (codebook / rate–distortion):")
    finance_demo()
    print("Rate–distortion grid (mirrored detail):")
    _print_grid()
    print("Viz constants (mirrored to the decimal in VectorQuantizationLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
