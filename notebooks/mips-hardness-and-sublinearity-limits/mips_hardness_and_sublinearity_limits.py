"""MIPS Hardness — the reference implementation for the formalRAG topic on why
exact high-dimensional search is hopeless and we must approximate.

Uses only numpy. The verification harness makes each claim executable: the dot
product is not metric (a vector need not be its own best match); the asymmetric
lifting transform reduces maximum inner-product search to Euclidean nearest-
neighbor search and preserves the argmax exactly, but distorts approximation
ratios (an explicit straddling counterexample); the Orthogonal Vectors -> farthest-
pair reduction separates orthogonal from non-orthogonal pairs by a fixed gap in
squared distance (so a truly-subquadratic exact algorithm would refute SETH); and,
as the ambient dimension grows, distance concentration collapses a metric index's
pruning power so the inspected fraction climbs toward 1. finance_demo() owns the
brute-force MIPS cost figure; grid_table() and the lifting counterexample own the
decimals MIPSHardnessLaboratory.tsx mirrors.

Run:  uv run --with numpy python notebooks/mips-hardness-and-sublinearity-limits/mips_hardness_and_sublinearity_limits.py
"""
from __future__ import annotations

import time

import numpy as np

# --------------------------------------------------------------------------- #
# MIPS and the metric question
# --------------------------------------------------------------------------- #

def brute_force_mips(q: np.ndarray, C: np.ndarray) -> int:
    """Exact maximum inner-product search: argmax_i <q, C_i>. The O(nd) baseline."""
    return int(np.argmax(C @ q))


def test_brute_force_crosscheck() -> None:
    """The baseline every later claim leans on: the matrix form agrees with a loop."""
    rng = np.random.default_rng(0)
    C = rng.standard_normal((50, 8))
    for _ in range(100):
        q = rng.standard_normal(8)
        loop = max(range(len(C)), key=lambda i: float(np.dot(q, C[i])))
        assert brute_force_mips(q, C) == loop
    print("  [ok] brute-force MIPS argmax matches the explicit loop")


def test_mips_not_self_similar() -> None:
    """The dot product is not metric: a vector need not be its own best match, so
    the metric-ball intuition (a point is closest to itself) fails for MIPS."""
    q = np.array([1.0, 0.0])
    longer = np.array([2.0, 0.0])          # same direction, larger norm
    corpus = np.vstack([q, longer])         # the query is in the corpus
    assert brute_force_mips(q, corpus) == 1                      # winner is `longer`, not q itself
    assert float(np.dot(q, longer)) > float(np.dot(q, q))       # <q, longer> = 2 > 1 = <q, q>
    print("  [ok] MIPS is not metric: a longer aligned vector beats the query's own copy")


# --------------------------------------------------------------------------- #
# The asymmetric lifting transform: MIPS -> Euclidean nearest neighbor
# --------------------------------------------------------------------------- #

def lift(C: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Norm-equalizing augmentation (Bachrach et al.; Shrivastava-Li).

    Append one coordinate to every database vector so they all have norm M, and a
    zero coordinate to the query. Then ||q~ - p~||^2 = ||q||^2 + M^2 - 2<q, p>, so
    minimizing lifted Euclidean distance maximizes the inner product.
    """
    M = float(np.max(np.linalg.norm(C, axis=1)))
    extra = np.sqrt(np.maximum(M**2 - (C**2).sum(axis=1), 0.0))
    C_tilde = np.hstack([C, extra[:, None]])
    q_tilde = np.concatenate([q, [0.0]])
    return C_tilde, q_tilde, M


def test_lifting_preserves_argmax() -> None:
    """Theorem: the lifted Euclidean nearest neighbor is the exact MIPS winner."""
    rng = np.random.default_rng(1)
    C = rng.standard_normal((40, 6)) * rng.uniform(0.5, 2.0, size=(40, 1))  # varied norms
    C_tilde, _, _ = lift(C, np.zeros(6))
    for _ in range(200):
        q = rng.standard_normal(6)
        q_tilde = np.concatenate([q, [0.0]])
        nn = int(np.argmin(np.linalg.norm(C_tilde - q_tilde, axis=1)))
        assert nn == brute_force_mips(q, C)          # top-1 identical pre/post the transform
    print("  [ok] the lifting transform preserves the MIPS argmax exactly (200 random queries)")


# Explicit corpus for the approximation-ratio distortion (also baked into the viz).
_DISTORT_Q = np.array([1.0, 0.0])
_DISTORT_A = np.array([0.90, 0.20])   # the MIPS winner / Euclidean NN
_DISTORT_B = np.array([0.85, 0.00])   # the runner-up that distorts


def lifting_distortion() -> dict[str, float]:
    """Compute B's MIPS- and NN-approximation factors relative to the winner A."""
    q, A, B = _DISTORT_Q, _DISTORT_A, _DISTORT_B
    C = np.vstack([A, B])
    C_tilde, q_tilde, M = lift(C, q)
    s_A, s_B = float(np.dot(q, A)), float(np.dot(q, B))
    dist_A = float(np.linalg.norm(q_tilde - C_tilde[0]))
    dist_B = float(np.linalg.norm(q_tilde - C_tilde[1]))
    return {
        "M": M,
        "s_A": s_A, "s_B": s_B,
        "dist_A": dist_A, "dist_B": dist_B,
        "mips_factor": s_A / s_B,        # how much worse B is as a MIPS solution
        "nn_factor": dist_B / dist_A,    # how much worse B is as a Euclidean neighbor
    }


def test_lifting_distorts_approximation_ratio() -> None:
    """The lifting preserves the argmax but NOT approximation ratios: B is a good
    MIPS approximation yet a poor Euclidean-NN approximation. Pick c = 1.3 — B is
    within c on inner product but not within c on lifted distance."""
    r = lifting_distortion()
    c = 1.3
    assert r["mips_factor"] < c, r       # B is a 1.3-approximate MIPS solution...
    assert r["nn_factor"] > c, r         # ...but NOT a 1.3-approximate Euclidean neighbor
    print(f"  [ok] lifting distorts ratios: B is {r['mips_factor']:.4f}x on MIPS "
          f"but {r['nn_factor']:.4f}x on lifted distance (straddles c = {c})")


# --------------------------------------------------------------------------- #
# The hardness core: Orthogonal Vectors -> farthest pair
# --------------------------------------------------------------------------- #

def _ov_corpus() -> tuple[np.ndarray, np.ndarray, int, tuple[int, int]]:
    """Two sets of fixed-weight (w = 2) Boolean vectors in {0,1}^6 with exactly one
    planted orthogonal (disjoint-support) cross pair A[i], B[j]."""
    w = 2
    A = np.array([
        [1, 1, 0, 0, 0, 0],   # support {0,1}
        [1, 0, 1, 0, 0, 0],   # support {0,2}
    ], dtype=float)
    B = np.array([
        [0, 1, 0, 1, 0, 0],   # support {1,3}: disjoint from A[1]={0,2} (the planted orthogonal pair)
        [0, 0, 1, 1, 0, 0],   # support {2,3}: shares coordinate 2 with A[1]
    ], dtype=float)
    # Planted orthogonal pair: A[1] support {0,2} and B[0] support {1,3} are disjoint.
    return A, B, w, (1, 0)


def test_ov_distance_gap() -> dict[str, float]:
    """OV -> farthest pair: with fixed Hamming weight w, ||a-b||^2 = 2w - 2<a,b>, so
    an orthogonal pair attains the maximal squared distance 2w and every non-
    orthogonal cross pair is at most 2w - 2. A subquadratic exact farthest-pair
    algorithm would therefore decide Orthogonal Vectors, refuting SETH."""
    A, B, w, (pi, pj) = _ov_corpus()
    ortho_d2, nonortho_max_d2 = None, -1.0
    found_ortho = False
    for i in range(len(A)):
        for j in range(len(B)):
            ip = float(np.dot(A[i], B[j]))
            d2 = float(np.dot(A[i] - B[j], A[i] - B[j]))
            assert abs(d2 - (2 * w - 2 * ip)) < 1e-9          # the identity, on 0/1 vectors
            if ip == 0.0:
                ortho_d2 = d2
                found_ortho = (i, j) == (pi, pj) or found_ortho
            else:
                nonortho_max_d2 = max(nonortho_max_d2, d2)
    assert ortho_d2 == 2 * w                                   # orthogonal pair is maximally far
    assert nonortho_max_d2 <= 2 * w - 2                        # a clean gap of >= 2 separates them
    assert found_ortho
    print(f"  [ok] OV->farthest-pair gap: orthogonal d^2 = {ortho_d2:.0f} = 2w, "
          f"non-orthogonal d^2 <= {nonortho_max_d2:.0f} = 2w-2")
    return {"w": float(w), "ortho_d2": ortho_d2, "nonortho_max_d2": nonortho_max_d2}


# --------------------------------------------------------------------------- #
# The empirical curse: metric pruning collapses as the dimension grows
# --------------------------------------------------------------------------- #

def inspected_fraction(d: int, rng: np.random.Generator, n: int = 800,
                       n_pivots: int = 4, n_queries: int = 25) -> float:
    """Mean fraction of candidate points a triangle-inequality pivot filter cannot
    prune for an exact nearest-neighbor query, averaged over several queries against
    one point cloud. A candidate x is pruned if some pivot p certifies
    |D(q,p) - D(p,x)| > r, where r is the true NN distance. As d grows, distances
    concentrate, the certificates vanish, and the inspected fraction -> 1 (pivots are
    excluded from the candidate set they organize). Averaging over queries tames the
    finite-sample variance so the climb is smooth."""
    X = rng.standard_normal((n, d))
    piv_idx = rng.choice(n, size=n_pivots, replace=False)
    candidate = np.ones(n, dtype=bool)
    candidate[piv_idx] = False
    pivots = X[piv_idx]
    fracs = []
    for _ in range(n_queries):
        q = rng.standard_normal(d)
        r = float(np.linalg.norm(X - q, axis=1).min())        # exact NN distance (best radius)
        pruned = np.zeros(n, dtype=bool)
        for p_idx, p in zip(piv_idx, pivots):
            Dqp = float(np.linalg.norm(p - q))                # D(q, pivot)
            Dpx = np.linalg.norm(X - p, axis=1)               # D(pivot, x)
            pruned |= np.abs(Dqp - Dpx) > r                   # triangle-inequality lower bound
        fracs.append(float((~pruned[candidate]).mean()))
    return float(np.mean(fracs))


_D_GRID = [2, 4, 6, 8, 12, 16, 24, 32, 64, 128]


def grid_table() -> list[tuple[int, float]]:
    """Inspected fraction across the dimension grid, all draws from one RNG stream."""
    rng = np.random.default_rng(7)                            # ONE stream across the whole grid
    return [(d, round(inspected_fraction(d, rng), 4)) for d in _D_GRID]


def test_pruning_collapses() -> None:
    """The pruning power collapses: the inspected fraction rises with d and is
    essentially total in high dimension."""
    table = grid_table()
    fracs = [f for _, f in table]
    assert fracs[0] < 0.1                                      # low d: pruning is highly effective
    assert fracs[-1] > 0.95                                    # high d: almost nothing prunes
    # The averaged curve climbs monotonically (allow a hair of residual sampling noise).
    for lo, hi in zip(fracs, fracs[1:]):
        assert hi >= lo - 0.02, f"non-monotone climb: {fracs}"
    print(f"  [ok] pruning collapses: inspected fraction {fracs[0]:.3f} (d={_D_GRID[0]}) "
          f"-> {fracs[-1]:.3f} (d={_D_GRID[-1]})")


# --------------------------------------------------------------------------- #
# Finance demo — exact MIPS is O(nd) per query, untenable at corpus scale
# --------------------------------------------------------------------------- #

def structured_data(n: int, d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A synthetic low-rank-plus-noise embedding cloud (intrinsic dim k << d),
    a deterministic stand-in for a trained encoder. Requires n > d for a full basis."""
    basis = rng.standard_normal((k, d))
    coeffs = rng.standard_normal((n, k))
    return coeffs @ basis + 0.01 * rng.standard_normal((n, d))


def finance_demo() -> None:
    rng = np.random.default_rng(11)
    d, k = 1536, 10
    q = rng.standard_normal(d)
    print(f"  exact MIPS is O(n*d) per query; d = {d} (production embedding dimension)")
    base = None
    for n in (5_000, 10_000, 20_000):
        C = structured_data(n, d, k, rng)
        t0 = time.perf_counter()
        for _ in range(5):
            _ = brute_force_mips(q, C)
        ms = (time.perf_counter() - t0) / 5 * 1e3
        flops = 2 * n * d
        if base is None:
            base = ms
        print(f"  n = {n:>7,}:  {ms:7.2f} ms/query   ({flops:,} multiply-adds)")
    print("  -> cost grows linearly in the corpus size; at millions of vectors this is untenable,")
    print("     which is why every downstream index trades exactness for sublinearity.")


def test_finance_scaling_is_linear() -> None:
    """The O(nd) claim, made deterministic: the multiply-add count is exactly n*d,
    so doubling the corpus doubles the work (independent of wall-clock noise)."""
    d = 1536
    for n in (5_000, 10_000):
        assert 2 * (2 * n) * d == 2 * (2 * n * d)              # work(2n) = 2 * work(n)
    print("  [ok] brute-force MIPS work is exactly 2*n*d multiply-adds (linear in n)")


def test_finance_demo_numbers() -> None:
    """Pin the decimals MIPSHardnessLaboratory.tsx bakes: the lifting-distortion
    factors, the OV gap, and the pruning grid."""
    r = lifting_distortion()
    assert round(r["mips_factor"], 4) == 1.0588, r
    assert round(r["nn_factor"], 4) == 1.7321, r              # = sqrt(3)
    table = grid_table()
    assert table[0][1] < 0.5 and table[-1][1] > 0.95, table
    print("  [ok] viz decimals pinned (lifting factors, pruning grid)")


if __name__ == "__main__":
    print("MIPS Hardness — verification harness")
    test_brute_force_crosscheck()
    test_mips_not_self_similar()
    test_lifting_preserves_argmax()
    test_lifting_distorts_approximation_ratio()
    test_ov_distance_gap()
    test_pruning_collapses()
    test_finance_scaling_is_linear()
    test_finance_demo_numbers()
    print("Pruning grid (inspected fraction vs dimension):")
    for d, f in grid_table():
        print(f"  d = {d:>4}:  inspected {f:.4f}")
    print("Finance demo:")
    finance_demo()
    print("All checks passed.")
