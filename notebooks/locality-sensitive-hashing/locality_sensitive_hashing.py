"""Locality-sensitive hashing — the reference implementation for the formalRAG
`locality-sensitive-hashing` topic.

IVF cut the space into Voronoi cells you *probe*; the graph topics walked a *graph*. LSH is the third
ANN family and the only one with a sharp, distribution-free theory of its own: hash so that *near*
points collide *more often* than far ones, and the collision probability — not a heuristic — is what
the index is built on. The centerpiece is random-hyperplane SimHash, which is signed random
projection, so this module imports the prerequisite's projection machinery rather than reinventing it.
The mathematics has three movements, and every pedagogical claim below is an `assert`:

  MOVEMENT 1 — THE FAMILY AND ITS COLLISION PROBABILITY (owns the numbers). A hash family is
    (r1, r2, p1, p2)-sensitive if it collides with probability >= p1 inside radius r1 and <= p2 beyond
    r2. For SimHash — one bit per random hyperplane, b(x) = 1[<x, h> >= 0] with h ~ N(0, I) — the
    collision probability of two vectors at angle theta is EXACTLY 1 - theta/pi (Goemans-Williamson /
    Charikar): a hyperplane separates x and y iff it falls in the wedge between them, which has measure
    theta/pi. We verify the law empirically across the angle range. (`random_hyperplanes`,
    `simhash_signature`, `collision_prob_theory`, `collision_prob_empirical`, `collision_curve`)

  MOVEMENT 2 — AMPLIFICATION: AND/OR AND THE S-CURVE. One bit is a weak filter. Concatenate k bits
    (an AND-construction: a table-collision needs all k to agree, probability p^k) and union L
    independent tables (an OR-construction: collide if ANY table matches, probability 1 - (1 - p^k)^L).
    The composite g(p) = 1 - (1 - p^k)^L is the canonical LSH S-curve; (k, L) place and sharpen its
    threshold, trading recall against the candidate-set size. The single-hash case k = L = 1 collapses
    g to the identity p — the byte-for-byte anchor that the amplified family IS the base family there.
    (`s_curve`, `amplified_collision_from_matches`, `s_curve_table`)

  MOVEMENT 3 — THE rho EXPONENT, SUBLINEAR QUERY TIME, AND THE HEAD-TO-HEAD. Tuning (k, L) to the gap
    (p1, p2) gives an (c, r)-ANN data structure with query time O(n^rho) and space O(n^{1+rho}), where
    rho = ln(1/p1) / ln(1/p2) < 1 whenever p1 > p2 — the headline theoretical result, sublinear by a
    data-INDEPENDENT family. rho shrinks as the approximation factor c (the r2/r1 gap) widens. Then the
    track's cross-index head-to-head: LSH vs IVF vs flat-NSW vs HNSW on ONE shared cloud and ONE shared
    ground truth, by distance computations per query. To compare angular SimHash against the Euclidean
    indexes honestly we L2-normalize the cloud: for unit vectors ||x - y||^2 = 2 - 2cos, so the
    Euclidean top-k ranking IS the angular ranking and the ground truth is shared by construction.
    (`rho_exponent`, `rho_curve`, `build_lsh_index`, `lsh_query`, `lsh_recall`, `lsh_frontier`,
    `lsh_head_to_head`)

Honest caveats (rigorFlag territory, asserted as DIRECTIONS): 1 - theta/pi is exact only for the
random-hyperplane family (p-stable / E2LSH is a different law for Euclidean distance); the O(n^rho)
guarantee is worst-case for the (c, r)-ANN DECISION problem under a data-INDEPENDENT family, so
realized recall depends on the data distribution and SimHash's rho is NOT optimal (data-dependent and
cross-polytope LSH beat it); and the head-to-head winner is one verdict on one synthetic normalized
cloud, where a data-AWARE index is expected to dominate the oblivious hash at equal cost — a statement
about this cloud, not a universal ranking.

This module imports its prerequisite (`johnson_lindenstrauss`) for the random hyperplanes and reuses
the ANN track's shared cloud + ground truth (`navigable_small_world_graphs`) and head-to-head harness
(`hnsw`) — it never reimplements them. `viz_constants()` prints what
`LocalitySensitiveHashingLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy --with scikit-learn python notebooks/locality-sensitive-hashing/locality_sensitive_hashing.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# SimHash = sign of a random projection, so the random hyperplanes ARE the prerequisite's projection
# rows; the shared low-rank cloud + brute-force ground truth come from the NSW topic (rooted in
# high-dimensional-geometry.structured_data), and the cross-index frontier reuses HNSW's head_to_head.
# Established cross-topic pattern: add each prereq's HYPHENATED dir to the path, import its UNDERSCORED
# module.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("johnson-lindenstrauss", "navigable-small-world-graphs", "hnsw"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from johnson_lindenstrauss import gaussian_projection  # noqa: E402
from navigable_small_world_graphs import _true_topk, nsw_dataset  # noqa: E402
from hnsw import head_to_head  # noqa: E402


# --------------------------------------------------------------------------- #
# Movement 1 — random-hyperplane SimHash and its collision probability.
# --------------------------------------------------------------------------- #

def random_hyperplanes(d: int, k: int, seed: int = 0) -> np.ndarray:
    """k random hyperplane normals as the rows of a (k, d) matrix, drawn N(0, I) via the prerequisite's
    `gaussian_projection`. The 1/sqrt(k) scale JL applies is irrelevant here — SimHash keeps only the
    SIGN of <x, h>, which the scaling does not change. GUARD: d, k >= 1."""
    if d < 1 or k < 1:
        raise ValueError(f"d and k must be >= 1, got d={d}, k={k}")
    return gaussian_projection(d, k, seed=seed)


def simhash_signature(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """The SimHash code of each row of X under the hyperplanes H (k, d): bit j is 1[<x, h_j> >= 0].
    Returns an (n, k) uint8 array of bits. Accepts a single vector (1-D) as a 1-row matrix."""
    X2 = np.atleast_2d(X)
    return (X2 @ H.T >= 0).astype(np.uint8)


def collision_prob_theory(theta: float) -> float:
    """SimHash collision probability for two vectors at angle theta (radians): 1 - theta/pi. A single
    random hyperplane separates them iff it lands in the wedge of angular measure theta/pi."""
    return 1.0 - theta / np.pi


def _unit_pair_at_angle(theta: float, d: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors in R^d separated by exactly angle theta: a random unit a, and b = cos(theta) a
    + sin(theta) r for r a random unit vector orthogonal to a."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(d)
    a /= np.linalg.norm(a)
    r = rng.standard_normal(d)
    r -= (r @ a) * a
    r /= np.linalg.norm(r)
    b = np.cos(theta) * a + np.sin(theta) * r
    return a, b


def _per_plane_match(u: np.ndarray, v: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Per-hyperplane agreement of the two vectors' SimHash bits: a boolean (n_planes,) array, True
    where the plane does NOT separate u and v. Its mean is the empirical collision probability."""
    su = (u @ H.T >= 0)
    sv = (v @ H.T >= 0)
    return su == sv


def collision_prob_empirical(u: np.ndarray, v: np.ndarray, n_planes: int = 4000, seed: int = 0) -> float:
    """Empirical SimHash collision probability of u and v over n_planes random hyperplanes."""
    H = random_hyperplanes(u.shape[0], n_planes, seed=seed)
    return float(_per_plane_match(u, v, H).mean())


def collision_curve(theta_fracs, d: int = 128, n_planes: int = 6000, seed: int = 0):
    """Empirical vs theoretical collision probability across angles theta = frac * pi. Returns a list
    of {theta_frac, theta, empirical, theory} dicts — the data Panel A plots against 1 - theta/pi."""
    rows = []
    for i, frac in enumerate(theta_fracs):
        theta = float(frac * np.pi)
        u, v = _unit_pair_at_angle(theta, d, seed=seed + i)
        rows.append({
            "theta_frac": round(float(frac), 3),
            "theta": round(theta, 4),
            "empirical": round(collision_prob_empirical(u, v, n_planes, seed=100 + i), 4),
            "theory": round(collision_prob_theory(theta), 4),
        })
    return rows


# --------------------------------------------------------------------------- #
# Movement 2 — amplification: the AND/OR construction and the S-curve.
# --------------------------------------------------------------------------- #

def s_curve(p, k: int, L: int):
    """The composite collision probability of an (AND-k, OR-L) amplified family: g(p) = 1 - (1-p^k)^L.
    Vectorized over an array p. GUARD: k, L >= 1."""
    if k < 1 or L < 1:
        raise ValueError(f"k and L must be >= 1, got k={k}, L={L}")
    p = np.asarray(p, dtype=float)
    return 1.0 - (1.0 - p ** k) ** L


def amplified_collision_from_matches(match: np.ndarray, k: int, L: int) -> float:
    """Empirical composite collision of an (AND-k, OR-L) family, computed from a 1-D boolean array of
    independent per-plane agreements. Reshape into T = len(match)//(k*L) trials of L tables of k
    planes; a table collides iff all k of its planes agree, the family iff ANY of its L tables does.
    Returns the fraction of trials in which the family collides.

    THE COLLAPSE ANCHOR: at k = L = 1 each trial is one plane and the family-collision indicator is the
    plane-agreement itself, so this returns match.mean() exactly — the amplified family IS the base
    SimHash there, byte for byte. GUARD: needs at least k*L planes."""
    if k < 1 or L < 1:
        raise ValueError(f"k and L must be >= 1, got k={k}, L={L}")
    block = k * L
    trials = match.shape[0] // block
    if trials < 1:
        raise ValueError(f"need at least k*L={block} planes, got {match.shape[0]}")
    grid = match[: trials * block].reshape(trials, L, k)
    table_collides = grid.all(axis=2)          # (trials, L): every plane in the table agreed
    family_collides = table_collides.any(axis=1)  # (trials,): at least one table matched fully
    return float(family_collides.mean())


def s_curve_table(k_values, L_values, p_grid):
    """The S-curve g(p) = 1 - (1-p^k)^L sampled over p_grid for each (k, L) config — what Panel B
    draws. Returns {'p': [...], 'curves': [{'k','L','g': [...]}, ...]}."""
    return {
        "p": [round(float(p), 3) for p in p_grid],
        "curves": [
            {"k": int(k), "L": int(L), "g": [round(float(g), 4) for g in s_curve(np.asarray(p_grid), k, L)]}
            for k, L in zip(k_values, L_values)
        ],
    }


# --------------------------------------------------------------------------- #
# Movement 3 — the rho exponent and sublinear query time.
# --------------------------------------------------------------------------- #

def rho_exponent(p1: float, p2: float) -> float:
    """The LSH exponent rho = ln(1/p1) / ln(1/p2). With p1 > p2 in (0, 1), rho < 1: query time O(n^rho)
    is sublinear. GUARD: both in (0, 1) and p1 > p2 (else the ratio is undefined or >= 1)."""
    if not (0.0 < p2 < 1.0) or not (0.0 < p1 < 1.0):
        raise ValueError(f"p1, p2 must be in (0, 1), got p1={p1}, p2={p2}")
    if p1 <= p2:
        raise ValueError(f"need p1 > p2 for a near>far family, got p1={p1}, p2={p2}")
    return float(np.log(1.0 / p1) / np.log(1.0 / p2))


def rho_curve(theta1_frac: float, c_grid):
    """rho as a function of the approximation factor c, for SimHash at near-angle theta1 = theta1_frac
    * pi and far-angle theta2 = min(c * theta1, ~pi). Returns a list of {c, theta2_frac, p1, p2, rho}.
    rho falls as c grows: a wider near/far gap is an easier ANN problem."""
    theta1 = theta1_frac * np.pi
    p1 = collision_prob_theory(theta1)
    rows = []
    for c in c_grid:
        theta2 = min(c * theta1, 0.999 * np.pi)   # cap below pi so p2 > 0
        p2 = collision_prob_theory(theta2)
        rows.append({
            "c": round(float(c), 3),
            "theta2_frac": round(float(theta2 / np.pi), 3),
            "p1": round(float(p1), 4),
            "p2": round(float(p2), 4),
            "rho": round(rho_exponent(p1, p2), 4),
        })
    return rows


# --------------------------------------------------------------------------- #
# The LSH index: L tables of k-bit SimHash, query by union of colliding buckets.
# --------------------------------------------------------------------------- #

def build_lsh_index(X: np.ndarray, k: int, L: int, seed: int = 0):
    """Build L hash tables, each bucketing X by a k-bit SimHash signature. Returns (tables, planes)
    where tables[t] maps a packed-signature key -> list of point indices and planes[t] is that table's
    (k, d) hyperplane matrix. GUARDS: k, L >= 1."""
    if k < 1 or L < 1:
        raise ValueError(f"k and L must be >= 1, got k={k}, L={L}")
    d = X.shape[1]
    planes = [random_hyperplanes(d, k, seed=seed * 1009 + t) for t in range(L)]
    tables = []
    for t in range(L):
        sig = simhash_signature(X, planes[t])     # (n, k) uint8
        buckets: dict[bytes, list[int]] = {}
        for i in range(sig.shape[0]):
            buckets.setdefault(sig[i].tobytes(), []).append(i)
        tables.append(buckets)
    return tables, planes


def lsh_query(X: np.ndarray, q: np.ndarray, tables, planes, topk: int = 10):
    """Query the LSH index: gather the candidate union from the buckets q collides with across all L
    tables, dedupe, compute the EXACT distance to each candidate, and return the top-k. Cost, in the
    head-to-head's "distance computations per query" currency, is the L*k hyperplane projections to
    hash the query (the analogue of IVF's nlist coarse comparisons) PLUS one exact distance per
    candidate examined. Returns (result indices nearest-first, cost). GUARD: topk >= 1."""
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    hash_work = sum(pl.shape[0] for pl in planes)     # L * k projections, the coarse hashing cost
    cand: set[int] = set()
    for t in range(len(tables)):
        key = (q @ planes[t].T >= 0).astype(np.uint8).tobytes()
        cand.update(tables[t].get(key, ()))
    if not cand:
        return [], hash_work
    idx = np.fromiter(cand, dtype=int, count=len(cand))
    d2 = ((X[idx] - q) ** 2).sum(axis=1)
    order = np.argsort(d2)[: min(topk, idx.shape[0])]
    return idx[order].tolist(), int(idx.shape[0] + hash_work)


def lsh_recall(X, queries, tables, planes, topk: int = 10, truth=None):
    """Mean recall@topk and mean cost per query (the L*k hashing work plus one exact distance per
    candidate examined) for an LSH index. GUARDS: non-empty queries, topk >= 1."""
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if truth is None:
        truth = _true_topk(queries, X, topk)
    hits, work = 0, 0
    for qi in range(len(queries)):
        idxs, ndist = lsh_query(X, queries[qi], tables, planes, topk=topk)
        hits += len(truth[qi] & set(idxs))
        work += ndist
    return hits / (len(queries) * topk), work / len(queries)


def lsh_frontier(X, queries, k: int, L_grid, topk: int = 10, truth=None, seed: int = 0):
    """The LSH speed/recall frontier: at fixed bits-per-table k, sweep the number of tables L. More
    tables widen the candidate union (more recall, more cost). Returns a list of {L, recall, cost}."""
    if truth is None:
        truth = _true_topk(queries, X, topk)
    rows = []
    for L in L_grid:
        tables, planes = build_lsh_index(X, k, L, seed=seed)
        rec, work = lsh_recall(X, queries, tables, planes, topk, truth=truth)
        rows.append({"L": int(L), "recall": float(rec), "cost": float(work)})
    return rows


def _normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row onto the unit sphere (so Euclidean and angular rankings coincide)."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def lsh_head_to_head(X, queries, topk: int = 10, k: int = 14,
                     L_grid=(1, 2, 4, 8, 16, 32, 64), seed: int = 0):
    """LSH vs IVF vs flat-NSW vs HNSW on ONE cloud, ONE ground truth. L2-normalize so SimHash's angular
    ranking equals the Euclidean ground truth the graph/IVF indexes are scored against; HNSW's
    head_to_head re-derives the SAME _true_topk on the SAME normalized cloud, so the truth is shared by
    construction. Returns head_to_head's dict with an added 'lsh' frontier and 'lsh_k'."""
    Xn, Qn = _normalize(X), _normalize(queries)
    truth = _true_topk(Qn, Xn, topk)
    h = head_to_head(Xn, Qn, topk=topk, seed=seed)
    h["lsh"] = lsh_frontier(Xn, Qn, k, L_grid, topk=topk, truth=truth, seed=seed)
    h["lsh_k"] = int(k)
    return h


# --------------------------------------------------------------------------- #
# Module constants the viz panels step through.
# --------------------------------------------------------------------------- #

THETA_FRACS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
S_CURVE_K = (4, 8, 16)
S_CURVE_L = (4, 8, 4)
P_GRID = tuple(round(0.05 * i, 3) for i in range(0, 21))     # 0.0 .. 1.0
RHO_THETA1_FRAC = 0.15
C_GRID = (1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
TOPK = 10
LSH_K = 14
LSH_L_GRID = (1, 2, 4, 8, 16, 32, 64)


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_collision_probability_matches_theory() -> None:
    """SimHash's collision probability is 1 - theta/pi across the angle range: the empirical collision
    rate over thousands of random hyperplanes matches the closed form to within Monte-Carlo error."""
    rows = collision_curve(THETA_FRACS, d=128, n_planes=8000, seed=0)
    for r in rows:
        assert abs(r["empirical"] - r["theory"]) < 0.02, \
            f"theta/pi={r['theta_frac']}: empirical {r['empirical']} vs 1-theta/pi {r['theory']}"
    # the law is exactly linear in theta: collision at theta=0 is ~1 and decreases monotonically
    emp = [r["empirical"] for r in rows]
    assert all(emp[i] > emp[i + 1] for i in range(len(emp) - 1)), f"collision not falling in theta: {emp}"
    print(f"  [ok] collision probability = 1 - theta/pi: empirical {emp} vs "
          f"theory {[r['theory'] for r in rows]}")


def test_single_hash_collapse() -> None:
    """The byte-for-byte anchor: the amplified (AND-k, OR-L) family at k = L = 1 IS the base SimHash.
    From one shared array of per-plane agreements, amplified_collision_from_matches(., 1, 1) equals the
    base collision rate match.mean() exactly — and the S-curve g(p) at k=L=1 is the identity p."""
    u, v = _unit_pair_at_angle(0.3 * np.pi, d=96, seed=7)
    H = random_hyperplanes(96, 5000, seed=11)
    match = _per_plane_match(u, v, H)
    base = float(match.mean())
    amp = amplified_collision_from_matches(match, k=1, L=1)
    assert amp == base, f"collapse broken: amplified(k=1,L=1) {amp} != base {base}"
    ident = s_curve(np.array([0.0, 0.37, 0.5, 0.83, 1.0]), k=1, L=1)
    assert np.allclose(ident, [0.0, 0.37, 0.5, 0.83, 1.0]), f"g(p; 1, 1) != p: {ident}"
    print(f"  [ok] single-hash collapse: amplified(k=1,L=1) == base SimHash ({amp:.4f}); g(p;1,1)=p")


def test_s_curve_sharpens() -> None:
    """The empirical amplified collision matches g(p) = 1 - (1-p^k)^L, and (k, L) sharpen the threshold:
    at a base p in the transition band, raising k lowers the composite (a stricter AND) while raising L
    raises it (a more forgiving OR)."""
    u, v = _unit_pair_at_angle(0.25 * np.pi, d=96, seed=3)   # base p = 1 - 0.25 = 0.75
    p = collision_prob_theory(0.25 * np.pi)
    H = random_hyperplanes(96, 240000, seed=5)
    match = _per_plane_match(u, v, H)
    for k, L in ((4, 4), (8, 8), (16, 4)):
        emp = amplified_collision_from_matches(match, k, L)
        pred = float(s_curve(np.array([p]), k, L)[0])
        assert abs(emp - pred) < 0.03, f"(k={k},L={L}): empirical {emp:.3f} vs g(p) {pred:.3f}"
    g_lowk = float(s_curve(np.array([p]), 4, 4)[0])
    g_highk = float(s_curve(np.array([p]), 16, 4)[0])
    assert g_highk < g_lowk, f"raising k should lower the composite at p={p}: {g_highk} !< {g_lowk}"
    g_lowL = float(s_curve(np.array([p]), 8, 4)[0])
    g_highL = float(s_curve(np.array([p]), 8, 16)[0])
    assert g_highL > g_lowL, f"raising L should raise the composite at p={p}: {g_highL} !> {g_lowL}"
    print(f"  [ok] S-curve: empirical amplified collision tracks 1-(1-p^k)^L; k sharpens, L lifts")


def test_rho_below_one() -> None:
    """rho = ln(1/p1)/ln(1/p2) < 1 for a genuine near>far family (p1 > p2), so query time O(n^rho) is
    sublinear; and rho falls as the approximation factor c widens the gap. The guard rejects p2 -> 1
    (ln(1/p2) -> 0) rather than dividing by zero."""
    rows = rho_curve(RHO_THETA1_FRAC, C_GRID)
    rhos = [r["rho"] for r in rows]
    assert all(0.0 < r < 1.0 for r in rhos), f"rho not in (0,1): {rhos}"
    assert all(rhos[i] > rhos[i + 1] for i in range(len(rhos) - 1)), f"rho not falling in c: {rhos}"
    try:
        rho_exponent(0.9, 1.0)            # p2 = 1 -> ln(1/p2) = 0
    except ValueError:
        pass
    else:
        raise AssertionError("rho_exponent must reject p2 = 1 (zero denominator)")
    print(f"  [ok] rho exponent: {rhos} all < 1 and falling as c grows (sublinear, easier with a wider gap)")


def test_lsh_recall_cost_tradeoff() -> None:
    """The LSH frontier is monotone: more tables L never lose recall and always examine more
    candidates, and with enough tables recall climbs well off the floor — the OR-amplification recall
    knob, paid for in candidate-set size."""
    X, queries = nsw_dataset()
    Xn, Qn = _normalize(X), _normalize(queries)
    truth = _true_topk(Qn, Xn, TOPK)
    rows = lsh_frontier(Xn, Qn, LSH_K, LSH_L_GRID, topk=TOPK, truth=truth, seed=0)
    rec = [r["recall"] for r in rows]
    cost = [r["cost"] for r in rows]
    assert all(rec[i] <= rec[i + 1] + 1e-9 for i in range(len(rec) - 1)), f"recall not monotone in L: {rec}"
    assert all(cost[i] <= cost[i + 1] + 1e-9 for i in range(len(cost) - 1)), f"cost not monotone in L: {cost}"
    assert rec[-1] > rec[0] + 0.2, f"recall did not climb with L: {rec}"
    print(f"  [ok] LSH frontier monotone: recall {[round(r,3) for r in rec]} at cost "
          f"{[round(c,1) for c in cost]} (L={list(LSH_L_GRID)})")


def test_head_to_head() -> None:
    """One normalized cloud, one shared ground truth, four indexes compared by distance computations
    per query. The DIRECTION pinned to the observed run: on this low-rank cloud the data-AWARE indexes
    (HNSW/IVF) reach high recall at far lower cost than the data-OBLIVIOUS LSH — the honest price of a
    distribution-free hash, exactly the rigorFlag's claim. (A statement about THIS cloud.)"""
    X, queries = nsw_dataset()
    h = lsh_head_to_head(X, queries, topk=TOPK, k=LSH_K, L_grid=LSH_L_GRID, seed=0)
    target = 0.9
    lsh_cost = min((r["cost"] for r in h["lsh"] if r["recall"] >= target), default=None)
    hnsw_cost = min((r["cost"] for r in h["hnsw"] if r["recall"] >= target), default=None)
    ivf_cost = min((r["cost"] for r in h["ivf"] if r["recall"] >= target), default=None)
    assert None not in (hnsw_cost, ivf_cost, lsh_cost), "an index never reached the target recall — adjust the grid"
    # Pinned to the observed run (the headline-flip rule): with honest cost accounting (the L*k hashing
    # work counted, as IVF counts its nlist coarse comparisons), the data-AWARE indexes dominate the
    # data-OBLIVIOUS hash on this low-rank cloud. A statement about THIS cloud, not a universal ranking.
    assert hnsw_cost < ivf_cost < lsh_cost, \
        f"expected HNSW < IVF < LSH at recall {target}: {hnsw_cost:.1f}, {ivf_cost:.1f}, {lsh_cost:.1f}"
    print(f"  [ok] head-to-head @recall>={target}: HNSW {hnsw_cost:.1f} < IVF {ivf_cost:.1f} < "
          f"LSH {lsh_cost:.1f} dist-comps/query (data-aware beats the oblivious hash on this cloud)")


# --------------------------------------------------------------------------- #
# Viz constants — printed for LocalitySensitiveHashingLaboratory.tsx to mirror.
# --------------------------------------------------------------------------- #

def viz_constants() -> None:
    """Print the collision-probability curve (Panel A), the S-curves (Panel B), and the rho curve +
    head-to-head frontiers (Panel C) — all baked to the decimal in the .tsx."""
    print("  PANEL A — collision probability vs angle (SimHash):")
    cc = collision_curve(THETA_FRACS, d=128, n_planes=8000, seed=0)
    print(f"    COLLISION = {[{'tf': r['theta_frac'], 'emp': r['empirical'], 'thy': r['theory']} for r in cc]}")

    print("  PANEL B — S-curves g(p) = 1 - (1-p^k)^L:")
    sc = s_curve_table(S_CURVE_K, S_CURVE_L, P_GRID)
    print(f"    P_GRID = {sc['p']}")
    for cur in sc["curves"]:
        print(f"    SCURVE k={cur['k']} L={cur['L']} = {cur['g']}")

    print("  PANEL C — rho exponent and the head-to-head:")
    rc = rho_curve(RHO_THETA1_FRAC, C_GRID)
    print(f"    RHO_THETA1_FRAC = {RHO_THETA1_FRAC}")
    print(f"    RHO_CURVE = {[{'c': r['c'], 'p1': r['p1'], 'p2': r['p2'], 'rho': r['rho']} for r in rc]}")
    X, queries = nsw_dataset()
    h = lsh_head_to_head(X, queries, topk=TOPK, k=LSH_K, L_grid=LSH_L_GRID, seed=0)
    print(f"    H2H_N = {h['n']}  H2H_TOPK = {h['topk']}  H2H_NLIST = {h['nlist']}  LSH_K = {h['lsh_k']}")
    for name in ("lsh", "ivf", "hnsw"):
        key = "L" if name == "lsh" else ("nprobe" if name == "ivf" else "ef")
        print(f"    H2H_{name.upper()} = {[{key: r[key], 'recall': round(r['recall'], 4), 'cost': round(r['cost'], 1)} for r in h[name]]}")


if __name__ == "__main__":
    print("Locality-sensitive hashing verification harness")
    test_head_to_head()                       # the headline runs first (pin to the observed run)
    test_collision_probability_matches_theory()
    test_single_hash_collapse()
    test_s_curve_sharpens()
    test_rho_below_one()
    test_lsh_recall_cost_tradeoff()
    print("Viz constants (mirrored to the decimal in LocalitySensitiveHashingLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
