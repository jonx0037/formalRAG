"""Product quantization and asymmetric distance computation — the reference
implementation for the formalRAG `product-quantization` topic.

The previous topic left a cliffhanger: a single flat codebook of k=256 codewords over a
256-d finance cloud reached only ~53% recall@10, and a flat codebook cannot do better
cheaply — it needs k=2^B centroids for B bits, which is capped at k<=n training points and
intractable past ~16-20 bits. Product quantization breaks that wall. Split each vector into
m disjoint subvectors, quantize each subspace independently with its own small codebook of
k* centroids (that is just Lloyd's k-means per subspace — imported from the previous topic),
and represent a vector by the tuple of m sub-centroid indices: m*log2(k*) bits. This module
establishes, and verifies, four facts and one honest caveat:

  1. ADDITIVE DISTORTION DECOMPOSITION. ||x - Q(x)||^2 = sum_j ||x^j - c^j_{q_j}||^2, because
     squared Euclidean distance separates over disjoint coordinate blocks. So total distortion
     is the SUM of per-subspace distortions and each subspace is minimized independently by
     Lloyd. EXACT. (`pq_distortion`, `pq_subspace_distortions`)
  2. EFFECTIVE CODEBOOK (k*)^m AT STORAGE m*k*. PQ represents (k*)^m distinct codewords while
     storing only m*k* centroids — m=8, k*=256 gives 2^64 codewords from 2048 centroids.
  3. ASYMMETRIC DISTANCE IS EXACT AND A TABLE LOOKUP. With the query q kept un-quantized,
     ||q - Q(x)||^2 = sum_j LUT[j][q_j(x)] where LUT[j][i] = ||q^j - c^j_i||^2 is an m x k*
     table built once per query; each database distance is then m lookups, O(m) vs O(d). EXACT.
     (`adc_table`, `adc_distance`)
  4. SCALABILITY, NOT EQUAL-BIT SUPERIORITY. At any bit budget a flat codebook can TRAIN, the
     flat (unconstrained) codebook matches or BEATS PQ — PQ pays a product-separability
     constraint. PQ's win is reaching bit budgets a flat codebook cannot, so its recall climbs
     far past the flat ceiling. (`flat_vs_pq_equal_bits`, `scalability_frontier`)

The honest caveat (and the next topic): PQ assumes the subspaces are independent. Cross-subspace
correlation (and unequal per-subspace variance under equal bit allocation) is the loss it pays,
and a variance-balancing rotation lowers distortion — which is exactly what OPTIMIZED product
quantization learns. We verify the DIRECTION of that effect, not a magic number.

Every pedagogical claim is an `assert` below; `viz_constants()` prints what
`ProductQuantizationLaboratory.tsx` mirrors to the decimal. The per-subspace quantizer, the
empty-cell furthest-point repair, the synthetic finance cloud, and the kmeans2 cross-check are
all IMPORTED from the Lloyd-Max topic, so PQ is provably just Lloyd run in product.

Run:  uv run --with numpy --with scipy python notebooks/product-quantization/product_quantization.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

# PQ = Lloyd per subspace. Import the verified Lloyd core (and the SAME finance cloud) from
# the prerequisite topic, the established cross-topic pattern (see rank_fusion_rrf.py -> bm25).
_VQ_DIR = pathlib.Path(__file__).resolve().parents[1] / "vector-quantization-lloyd-max"
if str(_VQ_DIR) not in sys.path:
    sys.path.insert(0, str(_VQ_DIR))
from vector_quantization_lloyd_max import (  # noqa: E402
    assign,
    best_codebook,
    finance_dataset,
    within_cluster_sse,
)


# --------------------------------------------------------------------------- #
# Subspace splitting. The empty-sub-cell guard is inherited from the imported
# Lloyd (furthest-point repair) — each subspace is just another Lloyd run.
# --------------------------------------------------------------------------- #

def subspace_slices(d: int, m: int):
    """The m contiguous column ranges of width d//m. GUARD: d must divide by m."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if d % m != 0:
        raise ValueError(f"dimension d={d} is not divisible by m={m} subspaces")
    w = d // m
    return [(j * w, (j + 1) * w) for j in range(m)]


def split_subspaces(X: np.ndarray, m: int):
    """Split X (n,d) into m blocks of width d//m. Returns a list of m arrays (n, d//m)."""
    return [X[:, a:b] for a, b in subspace_slices(X.shape[1], m)]


# --------------------------------------------------------------------------- #
# Train, encode, decode. Training is m independent Lloyd runs (kmeans++ restarts).
# --------------------------------------------------------------------------- #

def train_pq(X: np.ndarray, m: int, k_star: int, seed: int = 0, restarts: int = 2):
    """Train m sub-codebooks by the imported Lloyd. Returns a list of m arrays (k_star, d//m).
    GUARDS: d % m (subspace_slices), k_star < 1, k_star > n (no valid partition — the same
    bound the imported lloyd enforces, checked here for a clear per-subspace message)."""
    n = X.shape[0]
    if k_star < 1:
        raise ValueError(f"k_star must be >= 1, got {k_star}")
    if k_star > n:
        raise ValueError(f"k_star ({k_star}) exceeds n_points ({n}); no valid sub-partition")
    codebooks = []
    for j, sub in enumerate(split_subspaces(X, m)):
        _, C_j, _ = best_codebook(sub, k_star, seed=seed + 101 * j, restarts=restarts)
        codebooks.append(C_j)
    return codebooks


def pq_encode(X: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """PQ codes (n, m): column j is the nearest sub-centroid index in subspace j."""
    subs = split_subspaces(X, len(codebooks))
    codes = np.empty((X.shape[0], len(codebooks)), dtype=np.int64)
    for j, (sub, C_j) in enumerate(zip(subs, codebooks)):
        codes[:, j], _ = assign(sub, C_j)
    return codes


def pq_decode(codes: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """Reconstruct Q(X): concatenate the chosen sub-centroids. Returns (n, d)."""
    return np.hstack([codebooks[j][codes[:, j]] for j in range(len(codebooks))])


def pq_distortion(X: np.ndarray, codebooks: list[np.ndarray]) -> float:
    """Total reconstruction distortion sum_i ||x_i - Q(x_i)||^2."""
    diff = X - pq_decode(pq_encode(X, codebooks), codebooks)
    return float(np.sum(diff * diff))


def pq_subspace_distortions(X: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """The m per-subspace distortions; they SUM to pq_distortion (the additive identity)."""
    out = np.empty(len(codebooks))
    for j, sub in enumerate(split_subspaces(X, len(codebooks))):
        labels, _ = assign(sub, codebooks[j])
        out[j] = within_cluster_sse(sub, codebooks[j], labels)
    return out


# --------------------------------------------------------------------------- #
# Asymmetric (ADC) and symmetric (SDC) distance computation.
# --------------------------------------------------------------------------- #

def adc_table(q: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """Asymmetric LUT for one query q (d,): table (m, k_star), table[j,i] = ||q^j - c^j_i||^2."""
    m = len(codebooks)
    slices = subspace_slices(q.shape[0], m)
    table = np.empty((m, codebooks[0].shape[0]))
    for j, (a, b) in enumerate(slices):
        table[j] = cdist(q[a:b][None, :], codebooks[j], "sqeuclidean")[0]
    return table


def adc_distance(codes: np.ndarray, table: np.ndarray) -> np.ndarray:
    """ADC estimate of ||q - Q(x)||^2 for all db codes (n,m): sum_j table[j, codes[:,j]].
    Exact vs brute force on the decoded Q(X) — the LUT memoizes the m*k* sub-distances."""
    return np.stack([table[j, codes[:, j]] for j in range(table.shape[0])]).sum(axis=0)


def sdc_table(codebooks: list[np.ndarray]) -> np.ndarray:
    """Symmetric LUT (m, k_star, k_star): ||c^j_a - c^j_b||^2 for every sub-centroid pair."""
    m, k = len(codebooks), codebooks[0].shape[0]
    table = np.empty((m, k, k))
    for j in range(m):
        table[j] = cdist(codebooks[j], codebooks[j], "sqeuclidean")
    return table


def sdc_distance(query_codes: np.ndarray, db_codes: np.ndarray, sdc: np.ndarray) -> np.ndarray:
    """SDC estimate (query quantized too): sum_j sdc[j, query_codes[j], db_codes[:,j]]."""
    return np.stack([sdc[j, query_codes[j], db_codes[:, j]]
                     for j in range(db_codes.shape[1])]).sum(axis=0)


# --------------------------------------------------------------------------- #
# Retrieval: recall@k under ADC and SDC (mirrors the Lloyd topic's recall scheme).
# --------------------------------------------------------------------------- #

def _topk(d2_row: np.ndarray, topk: int) -> np.ndarray:
    if topk >= len(d2_row):                       # argpartition needs kth < len
        return np.arange(len(d2_row))
    return np.argpartition(d2_row, topk)[:topk]


def true_topk(queries: np.ndarray, X: np.ndarray, topk: int) -> list[set]:
    d2 = cdist(queries, X, "sqeuclidean")
    return [set(np.argpartition(r, topk)[:topk].tolist()) for r in d2]


def recall_adc(X: np.ndarray, queries: np.ndarray, codebooks: list[np.ndarray],
               topk: int = 10) -> float:
    """Mean recall@topk with the query kept exact (ADC) versus the exact neighbors."""
    if topk < 1 or len(queries) == 0:
        raise ValueError("topk must be >= 1 and queries non-empty")
    codes = pq_encode(X, codebooks)
    truth = true_topk(queries, X, topk)
    hits = 0
    for qi in range(len(queries)):
        approx = _topk(adc_distance(codes, adc_table(queries[qi], codebooks)), topk)
        hits += len(truth[qi] & set(approx.tolist()))
    return hits / (len(queries) * topk)


def recall_sdc(X: np.ndarray, queries: np.ndarray, codebooks: list[np.ndarray],
               topk: int = 10) -> float:
    """Mean recall@topk with the query quantized too (SDC)."""
    if topk < 1 or len(queries) == 0:
        raise ValueError("topk must be >= 1 and queries non-empty")
    codes = pq_encode(X, codebooks)
    qcodes = pq_encode(queries, codebooks)
    sdc = sdc_table(codebooks)
    truth = true_topk(queries, X, topk)
    hits = 0
    for qi in range(len(queries)):
        approx = _topk(sdc_distance(qcodes[qi], codes, sdc), topk)
        hits += len(truth[qi] & set(approx.tolist()))
    return hits / (len(queries) * topk)


def adc_sdc_estimation_error(X, queries, codebooks):
    """Mean squared error of the ADC and SDC distance ESTIMATES against the true ||q-x||^2,
    over the query x db grid. ADC keeps the query exact, so its mean error is lower — but only
    IN THE MEAN (individual pairs can reverse). Returns (mean_sq_err_adc, mean_sq_err_sdc)."""
    codes = pq_encode(X, codebooks)
    qcodes = pq_encode(queries, codebooks)
    sdc = sdc_table(codebooks)
    se_adc, se_sdc, count = 0.0, 0.0, 0
    for qi in range(len(queries)):
        true = cdist(queries[qi][None, :], X, "sqeuclidean")[0]
        est_adc = adc_distance(codes, adc_table(queries[qi], codebooks))
        est_sdc = sdc_distance(qcodes[qi], codes, sdc)
        se_adc += float(np.sum((est_adc - true) ** 2))
        se_sdc += float(np.sum((est_sdc - true) ** 2))
        count += len(true)
    return se_adc / count, se_sdc / count


# --------------------------------------------------------------------------- #
# Bit accounting, the flat-vs-PQ comparison, and the scalability frontier.
# --------------------------------------------------------------------------- #

def pq_bits(m: int, k_star: int) -> float:
    """Code length in bits: m * log2(k_star) (k_star a power of two for an integer count)."""
    return m * float(np.log2(k_star))


def effective_codebook_size(m: int, k_star: int) -> int:
    """(k_star)^m distinct codewords — an exact Python big int (never compute in JS)."""
    return k_star ** m


def flat_vs_pq_equal_bits(X, queries, bits_grid=(8, 10, 12), seed=7):
    """At each bit budget B where a flat k=2^B codebook is trainable (2^B <= n), compare a
    flat codebook against a PQ config with m=2 at the SAME B bits. HONEST RESULT: flat is
    unconstrained, so flat distortion <= PQ distortion at equal trainable bits. Returns rows."""
    n = X.shape[0]
    rows = []
    for B in bits_grid:
        k_flat = 2 ** B
        if k_flat > n:
            continue
        _, C_flat, _ = best_codebook(X, k_flat, seed=seed, restarts=2)
        flat_D = pq_distortion(X, [C_flat])                 # m=1 PQ == flat VQ
        k_star = 2 ** (B // 2)                               # m=2 at the same B bits
        pq_cb = train_pq(X, 2, k_star, seed=seed, restarts=2)
        rows.append({
            "bits": B, "k_flat": k_flat, "flat_D": flat_D,
            "m": 2, "k_star": k_star, "pq_D": pq_distortion(X, pq_cb),
            "flat_recall": recall_adc(X, queries, [C_flat]),
            "pq_recall": recall_adc(X, queries, pq_cb),
        })
    return rows


def scalability_frontier(X, queries, m_grid=(1, 2, 4, 8), k_star=256, seed=7):
    """For each m at fixed k_star: bits, stored centroids, effective codebook size (big int),
    distortion, and recall_adc. m=1 IS the flat codebook (the ~53% ceiling re-derived on the
    SAME cloud); higher m reaches bit budgets a flat codebook cannot, so recall climbs past it.
    Returns rows."""
    rows = []
    for m in m_grid:
        cb = train_pq(X, m, k_star, seed=seed, restarts=2)
        rows.append({
            "m": m, "bits": int(pq_bits(m, k_star)), "stored": m * k_star,
            "effective": effective_codebook_size(m, k_star),
            "distortion": pq_distortion(X, cb), "recall": recall_adc(X, queries, cb),
        })
    return rows


# --------------------------------------------------------------------------- #
# The independence / OPQ teaser: a variance-balancing rotation lowers distortion.
# DIRECTION verified, not a magic number — see test_rotation_lowers_distortion.
# --------------------------------------------------------------------------- #

def variance_imbalanced_cloud(n: int = 600, d: int = 8, decay: float = 0.6, seed: int = 0):
    """A cloud whose coordinate variances decay (std = decay^j), so the m contiguous subspaces
    carry very UNEQUAL variance. Under equal bits per subspace, PQ over-quantizes the low-
    variance tail and under-quantizes the high-variance head — exactly the imbalance a
    rotation fixes. Returns X (n,d)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) * (decay ** np.arange(d))


def pca_align(X: np.ndarray):
    """Center and rotate X onto its principal axes (diagonalize the covariance). Returns the
    rotation R (d,d) with rows = eigenvectors (descending eigenvalue) and the mean."""
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = Xc.T @ Xc / (X.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return V[:, order].T, mu


def balanced_rotation(X: np.ndarray, m: int):
    """OPQ's parametric idea: PCA-align, then permute the principal axes so each of the m
    subspaces receives a near-equal share of total variance (greedy: assign each next-largest
    eigen-axis to the currently lowest-variance subspace). Returns a rotation R (d,d)."""
    R_pca, _ = pca_align(X)
    Xr = (X - X.mean(axis=0)) @ R_pca.T
    var = Xr.var(axis=0)
    order = np.argsort(var)[::-1]                 # axes by descending variance
    d, w = X.shape[1], X.shape[1] // m
    buckets, load = [[] for _ in range(m)], np.zeros(m)
    for ax in order:
        j = int(np.argmin(load + (np.array([len(b) for b in buckets]) >= w) * 1e18))
        buckets[j].append(ax)
        load[j] += var[ax]
    perm = np.concatenate([np.array(b, dtype=int) for b in buckets])
    return R_pca[perm]                            # permute the PCA axes into balanced blocks


def random_rotation(d: int, seed: int = 0):
    """A Haar-distributed orthogonal matrix via QR of a Gaussian. Returns R (d,d), R@R.T = I."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return Q


def rotation_distortion_study(X, m, k_star, seed=0):
    """PQ distortion on the cloud under: raw / a random rotation / PCA-align alone / the
    balanced rotation. The point: a NAIVE rotation (PCA-align alone) can HURT by concentrating
    variance into one subspace, while the balanced rotation HELPS — OPQ optimizes this. Returns
    a dict of distortions. *** RUN-DEPENDENT: directions confirmed empirically below. ***"""
    R_rand = random_rotation(X.shape[1], seed)
    R_pca, mu = pca_align(X)
    R_bal = balanced_rotation(X, m)
    out = {}
    for name, Xr in [("raw", X), ("random", X @ R_rand.T),
                     ("pca_only", (X - mu) @ R_pca.T), ("balanced", (X - mu) @ R_bal.T)]:
        out[name] = pq_distortion(Xr, train_pq(Xr, m, k_star, seed=seed, restarts=2))
    return out


def subspace_correlation_proxy(X: np.ndarray, m: int) -> float:
    """A scalar proxy for cross-subspace dependence: the fraction of total covariance energy
    sitting in the OFF-block entries (between different subspaces). Zero iff subspaces are
    uncorrelated — the loss PQ pays, which a decorrelating rotation reduces."""
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (X.shape[0] - 1)
    slices = subspace_slices(X.shape[1], m)
    block = np.zeros_like(cov, dtype=bool)
    for a, b in slices:
        block[a:b, a:b] = True
    total = float(np.sum(cov ** 2))
    return float(np.sum(cov[~block] ** 2) / total) if total > 0 else 0.0


# --------------------------------------------------------------------------- #
# Toy cloud (viz) and the finance demo (imported cloud).
# --------------------------------------------------------------------------- #

TOY_M = 2
TOY_KSTAR = 4


def toy_pq_cloud(seed: int = 0):
    """48 points in R^4 = two independent 2-D subspaces with DIFFERENT blob structure: a loose
    4-blob layout in dims (0,1) and a tighter 4-blob layout in dims (2,3). The laboratory draws
    each subspace as its own 2-D Voronoi, and the per-subspace distortions differ (loose > tight)."""
    rng = np.random.default_rng(seed)
    n = 48
    cA = np.array([[2.5, 7.5], [7.5, 7.5], [7.5, 2.5], [2.5, 2.5]])
    cB = np.array([[3.0, 5.0], [7.0, 5.0], [5.0, 8.0], [5.0, 2.0]])
    aidx, bidx = rng.integers(0, 4, n), rng.integers(0, 4, n)
    XA = cA[aidx] + 1.15 * rng.standard_normal((n, 2))
    XB = cB[bidx] + 0.6 * rng.standard_normal((n, 2))
    return np.clip(np.hstack([XA, XB]), 0.0, 10.0)


def finance_demo() -> dict:
    """PQ on the SAME 256-d finance cloud the Lloyd topic used. Print the compression framing,
    the re-derived flat 8-bit ceiling, and the PQ recall at 8 bytes (64 bits)."""
    X, _, queries = finance_dataset()
    raw_bits = X.shape[1] * 32
    frontier = scalability_frontier(X, queries)
    flat = next(r for r in frontier if r["m"] == 1)
    pq64 = next(r for r in frontier if r["m"] == 8)
    print(f"  {X.shape[0]} embeddings in R^{X.shape[1]} (SYNTHETIC, the Lloyd-Max cloud):")
    print(f"  raw vector = {raw_bits} bits; PQ m=8,k*=256 code = {pq64['bits']} bits "
          f"({pq64['bits']//8} bytes) -> {raw_bits/pq64['bits']:.0f}x smaller, "
          f"{pq64['stored']} stored centroids vs 2^64 effective codewords")
    print(f"  {'m':>3}{'bits':>6}{'stored':>8}{'recall@10':>11}{'D':>12}")
    for r in frontier:
        print(f"  {r['m']:>3}{r['bits']:>6}{r['stored']:>8}{r['recall']:>11.3f}{r['distortion']:>12.1f}")
    print(f"  -> flat ceiling (m=1, 8 bits): recall {flat['recall']:.3f}; "
          f"PQ at 64 bits: recall {pq64['recall']:.3f} — a budget a flat codebook (k=2^64) "
          f"cannot reach (k <= n).")
    return {"frontier": frontier, "flat": flat, "pq64": pq64, "raw_bits": raw_bits}


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_additive_decomposition() -> None:
    """EXACT: ||x - Q(x)||^2 = sum of per-subspace distortions, on the toy and finance clouds."""
    Xt = toy_pq_cloud()
    cb_t = train_pq(Xt, TOY_M, TOY_KSTAR, seed=0)
    assert abs(pq_distortion(Xt, cb_t) - pq_subspace_distortions(Xt, cb_t).sum()) < 1e-9, \
        "toy: total distortion != sum of subspace distortions"
    Xf, _, _ = finance_dataset()
    cb_f = train_pq(Xf, 8, 256, seed=7)
    assert abs(pq_distortion(Xf, cb_f) - pq_subspace_distortions(Xf, cb_f).sum()) < 1e-6, \
        "finance: additive decomposition fails"
    print("  [ok] additive decomposition: ||x-Q(x)||^2 = sum_j ||x^j-c^j||^2 (exact)")


def test_adc_equals_bruteforce() -> None:
    """EXACT: adc_distance == ||q - Q(x)||^2 computed directly on the decoded Q(X), to 1e-9."""
    X, _, queries = finance_dataset()
    cb = train_pq(X, 8, 256, seed=7)
    codes = pq_encode(X, cb)
    Q = pq_decode(codes, cb)
    for qi in range(8):
        adc = adc_distance(codes, adc_table(queries[qi], cb))
        brute = cdist(queries[qi][None, :], Q, "sqeuclidean")[0]
        assert np.allclose(adc, brute, atol=1e-7), "ADC != brute-force distance to Q(x)"
    print("  [ok] ADC is exact: sum_j LUT[j][q_j] == ||q - Q(x)||^2 (table is a memoized sum)")


def test_effective_codebook_size() -> None:
    """Pure counting (Python big ints): m=8,k*=256 -> 2^64 codewords from 2048 stored centroids."""
    assert effective_codebook_size(8, 256) == 2 ** 64, "256^8 != 2^64"
    assert 8 * 256 == 2048, "stored centroid count wrong"
    assert int(pq_bits(8, 256)) == 64, "bit count wrong"
    print("  [ok] effective codebook 256^8 = 2^64 from 8*256 = 2048 stored centroids")


def test_adc_error_below_sdc_error() -> None:
    """ADC's distance-estimate error is lower than SDC's IN THE MEAN (not per-pair): keeping the
    query exact removes the query-quantization term. Direction asserted, magnitude not baked."""
    X, _, queries = finance_dataset()
    cb = train_pq(X, 8, 256, seed=7)
    mse_adc, mse_sdc = adc_sdc_estimation_error(X, queries[:40], cb)
    assert mse_adc < mse_sdc, f"ADC mean error {mse_adc:.3f} not below SDC {mse_sdc:.3f}"
    print(f"  [ok] ADC mean sq-error {mse_adc:.2f} < SDC {mse_sdc:.2f} (query kept exact)")


def test_adc_recall_ge_sdc_recall() -> None:
    """recall_adc >= recall_sdc on the finance cloud (query exact beats query quantized)."""
    X, _, queries = finance_dataset()
    cb = train_pq(X, 8, 256, seed=7)
    r_adc, r_sdc = recall_adc(X, queries, cb), recall_sdc(X, queries, cb)
    assert r_adc >= r_sdc - 1e-9, f"recall_adc {r_adc:.3f} < recall_sdc {r_sdc:.3f}"
    print(f"  [ok] recall_adc {r_adc:.3f} >= recall_sdc {r_sdc:.3f}")


def test_flat_ge_pq_at_equal_bits() -> None:
    """HONESTY: at every trainable bit budget, the unconstrained flat codebook's distortion is
    <= PQ's (PQ pays the product constraint). If this ever flips it is a Lloyd local optimum,
    not a real win — we do NOT flip the inequality."""
    X, _, queries = finance_dataset()
    rows = flat_vs_pq_equal_bits(X, queries, bits_grid=(4, 6, 8))
    assert rows, "no trainable equal-bit budgets found"
    for r in rows:
        assert r["flat_D"] <= r["pq_D"] * (1 + 1e-6), \
            f"flat distortion {r['flat_D']:.1f} > PQ {r['pq_D']:.1f} at {r['bits']} bits"
    print(f"  [ok] honest: flat VQ distortion <= PQ at equal trainable bits "
          f"({', '.join(str(r['bits']) for r in rows)} bits)")


def test_pq_recall_beats_flat_at_pq_budget() -> None:
    """THE HEADLINE: PQ at 64 bits (m=8,k*=256, 2048 stored centroids) reaches recall@10 far
    above the flat 8-bit ceiling (m=1) on the SAME cloud — a budget a flat codebook (k=2^64)
    cannot reach. The flat ceiling is re-derived here, not hardcoded."""
    X, _, queries = finance_dataset()
    rows = scalability_frontier(X, queries)
    flat = next(r for r in rows if r["m"] == 1)["recall"]
    pq64 = next(r for r in rows if r["m"] == 8)["recall"]
    assert pq64 > flat + 0.2, f"PQ@64bit recall {pq64:.3f} not >> flat@8bit {flat:.3f}"
    print(f"  [ok] headline: PQ@64bit recall {pq64:.3f} >> flat@8bit ceiling {flat:.3f}")


def test_rotation_lowers_distortion() -> None:
    """A variance-balancing rotation lowers PQ distortion on a variance-imbalanced cloud, and a
    NAIVE PCA-align-only rotation can do worse — the gap OPQ optimizes. Direction only."""
    X = variance_imbalanced_cloud(n=600, d=8, seed=1)
    study = rotation_distortion_study(X, 4, 16, seed=1)
    assert study["balanced"] < study["raw"] * (1 - 1e-6), \
        f"balanced rotation {study['balanced']:.3f} did not beat raw {study['raw']:.3f}"
    print(f"  [ok] rotation/OPQ teaser: balanced {study['balanced']:.3f} < raw {study['raw']:.3f} "
          f"(pca_only {study['pca_only']:.3f}, random {study['random']:.3f})")


def test_rate_distortion_monotone_pq() -> None:
    """Refinement: more PQ bits (more subspaces at fixed k*) never raise distortion and never
    lower recall on the finance cloud."""
    X, _, queries = finance_dataset()
    rows = scalability_frontier(X, queries)
    dist = np.array([r["distortion"] for r in rows])
    rec = np.array([r["recall"] for r in rows])
    assert np.all(np.diff(dist) <= 1e-6 * dist[0]), f"distortion not monotone in bits: {dist}"
    assert np.all(np.diff(rec) >= -1e-9), f"recall not monotone in bits: {rec}"
    print(f"  [ok] rate-distortion monotone: D {dist[0]:.0f}->{dist[-1]:.0f}, "
          f"recall {rec[0]:.2f}->{rec[-1]:.2f} as bits grow")


def validate_against_kmeans2() -> None:
    """Per-subspace cross-check: a sub-codebook trained by the imported Lloyd matches
    scipy.cluster.vq.kmeans2 from the same init — PQ is just Lloyd per block, no faiss."""
    X, _, _ = finance_dataset()
    sub = split_subspaces(X, 8)[0]
    from vector_quantization_lloyd_max import kmeans_pp_init, lloyd
    C0 = kmeans_pp_init(sub, 64, seed=3)
    C_ours, lab_ours, _ = lloyd(sub, 64, C0, max_iter=300, tol=1e-12)
    C_sp, lab_sp = kmeans2(sub, C0, iter=300, minit="matrix")
    d_ours = within_cluster_sse(sub, C_ours, lab_ours)
    d_sp = within_cluster_sse(sub, C_sp, lab_sp)
    assert abs(d_ours - d_sp) <= 1e-6 * d_ours, f"subspace Lloyd {d_ours:.3f} != kmeans2 {d_sp:.3f}"
    print(f"  [ok] cross-check: per-subspace Lloyd matches scipy kmeans2 ({d_ours:.2f})")


# --------------------------------------------------------------------------- #
# Viz constants — the exact numbers ProductQuantizationLaboratory.tsx mirrors.
# --------------------------------------------------------------------------- #

def viz_constants() -> None:
    """Print the toy cloud, the two trained sub-codebooks, the additive distortions, the
    scalability frontier, and one ADC demo — all baked to the decimal in the .tsx."""
    X = toy_pq_cloud()
    cb = train_pq(X, TOY_M, TOY_KSTAR, seed=0)
    print(f"  TOY_POINTS_4D ({X.shape[0]} pts in R^4 = two 2-D subspaces), first 4 rows:")
    for p in X[:4]:
        print("    [" + ", ".join(f"{v:.4f}" for v in p) + "]")
    for j in range(TOY_M):
        print(f"  SUB_CODEBOOK_{'AB'[j]} (k*={TOY_KSTAR}): {np.round(cb[j], 4).tolist()}")
    sub_d = pq_subspace_distortions(X, cb)
    print(f"  per-subspace distortion = {np.round(sub_d, 4).tolist()}, total = {sub_d.sum():.4f}")
    # ADC demo: query = first point's exact vector; db = a different point's code.
    q = X[0]
    codes = pq_encode(X, cb)
    table = adc_table(q, cb)
    dbi = 17
    code = codes[dbi]
    adc = float(table[0, code[0]] + table[1, code[1]])
    true = float(cdist(q[None, :], pq_decode(codes[dbi][None, :], cb), "sqeuclidean")[0, 0])
    print(f"  ADC_DEMO query=pt0, db=pt{dbi} code={code.tolist()}: "
          f"table[0,{code[0]}]={table[0,code[0]]:.4f} + table[1,{code[1]}]={table[1,code[1]]:.4f} "
          f"= {adc:.4f}  (true ||q-Q(x)||^2 = {true:.4f})")


if __name__ == "__main__":
    print("Product quantization / ADC verification harness")
    test_additive_decomposition()
    test_adc_equals_bruteforce()
    test_effective_codebook_size()
    test_adc_error_below_sdc_error()
    test_adc_recall_ge_sdc_recall()
    test_flat_ge_pq_at_equal_bits()
    test_pq_recall_beats_flat_at_pq_budget()
    test_rotation_lowers_distortion()
    test_rate_distortion_monotone_pq()
    validate_against_kmeans2()
    print("Finance demo (PQ rate / scalability):")
    finance_demo()
    print("Viz constants (mirrored to the decimal in ProductQuantizationLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
