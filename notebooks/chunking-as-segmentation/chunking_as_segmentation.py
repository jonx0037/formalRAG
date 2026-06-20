"""Chunking as a segmentation and optimization problem — the reference implementation
for the formalRAG `chunking-as-segmentation` topic.

Before a document can be retrieved it must be split into chunks. Fixed-size splitting
ignores meaning; the principled alternative is to choose boundaries that maximize
within-chunk coherence, which is a one-dimensional SEGMENTATION problem with an exact
dynamic-programming solution. This module makes that precise and verifies it:

  1. COHERENCE = MEAN RESULTANT LENGTH. For L2-normalized sentence embeddings, the
     within-segment cost sum_t (1 - cos(e_t, mu)) collapses to len - ||sum e_t||, i.e.
     len * (1 - Rbar), where Rbar is the mean resultant length — the von Mises-Fisher
     concentration statistic from the prerequisite topic. Minimizing total cost carves
     the document into tight vMF clusters on the sphere.
  2. THE DP IS OPTIMAL. With an additive segment cost, OPT(j) = min_{i<j} OPT(i) +
     cost(i,j) computes the globally optimal segmentation in O(n^2). It matches brute
     force exactly, and no greedy heuristic (TextTiling-style) or fixed-size split can
     beat its objective value at the same number of segments.
  3. OPTIMAL RECOVERS STRUCTURE. On a document with planted topic shifts, the DP
     recovers the true boundaries better than the greedy and fixed-size baselines, and
     the optimal within-segment cost is monotone decreasing in the number of segments
     (the granularity/over-segmentation tradeoff).

Every pedagogical claim is an `assert` below. The DP optimality is exact (it matches
brute force); TextTiling and fixed-size are heuristics with no optimality guarantee.
`grid_table()` prints the profile / boundaries / coherence blocks that
`ChunkingLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/chunking-as-segmentation/chunking_as_segmentation.py
"""
from __future__ import annotations

import numpy as np

# The granularities (segment counts) the coherence-vs-K curve steps through.
K_GRID: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12)


# --------------------------------------------------------------------------- #
# Synthetic document — a sequence of unit sentence embeddings drawn from a few
# topical von Mises-Fisher clusters, with planted segment boundaries.
# --------------------------------------------------------------------------- #

def synthetic_document(segment_lengths, dim: int = 64, kappa: float = 12.0, seed: int = 0):
    """A document as a sequence of L2-normalized sentence embeddings: each planted
    segment is a vMF-like cluster (a random mean direction perturbed by Gaussian noise
    scaled by 1/sqrt(kappa)). Returns (embeddings (n x dim, unit rows), labels, boundaries),
    where `boundaries` are the internal split indices (segment starts, excluding 0)."""
    rng = np.random.default_rng(seed)
    emb, labels, boundaries, pos = [], [], [], 0
    for s, length in enumerate(segment_lengths):
        mu = rng.standard_normal(dim)
        mu /= np.linalg.norm(mu)
        block = mu + rng.standard_normal((length, dim)) / np.sqrt(kappa)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        emb.append(block)
        labels.extend([s] * length)
        pos += length
        if s < len(segment_lengths) - 1:
            boundaries.append(pos)
    return np.vstack(emb), np.array(labels), boundaries


# --------------------------------------------------------------------------- #
# The additive segment cost and the prefix-sum machinery that makes it O(dim).
# --------------------------------------------------------------------------- #

def _prefix_sums(emb: np.ndarray) -> np.ndarray:
    """Row-cumulative sums with a leading zero row, so the sum over [i, j) is P[j]-P[i]."""
    return np.vstack([np.zeros(emb.shape[1]), np.cumsum(emb, axis=0)])


def segment_cost(P: np.ndarray, i: int, j: int) -> float:
    """Incoherence of the segment [i, j): length - ||sum of unit embeddings|| = len*(1-Rbar).
    Zero iff all sentences point the same way; large when they cancel."""
    return float((j - i) - np.linalg.norm(P[j] - P[i]))


def total_coherence_cost(emb: np.ndarray, boundaries) -> float:
    """Total within-segment incoherence for a segmentation given by internal boundaries."""
    P = _prefix_sums(emb)
    cuts = [0, *boundaries, len(emb)]
    return float(sum(segment_cost(P, a, b) for a, b in zip(cuts, cuts[1:])))


def mean_resultant_length(emb: np.ndarray, i: int, j: int) -> float:
    """Rbar = ||mean of unit embeddings|| over [i, j) — the vMF concentration statistic."""
    return float(np.linalg.norm(emb[i:j].mean(axis=0)))


# --------------------------------------------------------------------------- #
# Optimal segmentation by dynamic programming, and a brute-force oracle.
# --------------------------------------------------------------------------- #

def segment_dp(emb: np.ndarray, k: int):
    """Globally optimal segmentation into exactly k contiguous segments, by DP over
    (segments-used, prefix-length). Returns (boundaries, total_cost). O(k n^2)."""
    n = len(emb)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and the number of embeddings n={n}, got {k}")
    P = _prefix_sums(emb)
    INF = float("inf")
    D = [[INF] * (n + 1) for _ in range(k + 1)]
    back = [[0] * (n + 1) for _ in range(k + 1)]
    D[0][0] = 0.0
    for s in range(1, k + 1):
        for j in range(s, n + 1):
            for i in range(s - 1, j):
                c = D[s - 1][i] + segment_cost(P, i, j)
                if c < D[s][j]:
                    D[s][j] = c
                    back[s][j] = i
    # reconstruct internal boundaries
    bounds, j = [], n
    for s in range(k, 0, -1):
        i = back[s][j]
        if i > 0:
            bounds.append(i)
        j = i
    return sorted(bounds), D[k][n]


def _compositions(n: int, k: int):
    """All ways to write n as k positive parts (ordered) — the segment lengths."""
    if k == 1:
        yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in _compositions(n - first, k - 1):
            yield (first, *rest)


def brute_force_segment(emb: np.ndarray, k: int):
    """The optimal k-segmentation by exhaustive search — the oracle for small n."""
    P = _prefix_sums(emb)
    best_cost, best_bounds = float("inf"), None
    for comp in _compositions(len(emb), k):
        cuts, pos = [0], 0
        for length in comp:
            pos += length
            cuts.append(pos)
        cost = sum(segment_cost(P, a, b) for a, b in zip(cuts, cuts[1:]))
        if cost < best_cost:
            best_cost, best_bounds = cost, cuts[1:-1]
    return best_bounds, best_cost


# --------------------------------------------------------------------------- #
# Heuristic baselines: TextTiling-style greedy, and fixed-size chunking.
# --------------------------------------------------------------------------- #

def adjacent_dissimilarity(emb: np.ndarray, window: int = 3) -> np.ndarray:
    """1 - cosine between the mean of the `window` sentences before and after each gap.
    Peaks mark likely topic boundaries — the TextTiling signal."""
    n = len(emb)
    if n < 2:
        return np.array([], dtype=float)
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    g = np.zeros(n - 1)
    for t in range(1, n):
        left = emb[max(0, t - window):t].mean(axis=0)
        right = emb[t:t + window].mean(axis=0)
        cos = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
        g[t - 1] = 1.0 - cos
    return g


def texttiling_greedy(emb: np.ndarray, k: int, window: int = 3):
    """Greedy boundaries: the k-1 gaps with the highest adjacent dissimilarity. A local,
    TextTiling-style heuristic — no global optimality."""
    g = adjacent_dissimilarity(emb, window)
    idx = np.argsort(g)[::-1]
    bounds = sorted(int(t + 1) for t in idx[:k - 1])
    return bounds


def fixed_size(n: int, k: int):
    """k roughly-equal segments — semantics-free chunking."""
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and n={n}, got {k}")
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [int(b) for b in edges[1:-1]]


# --------------------------------------------------------------------------- #
# Boundary recovery scoring.
# --------------------------------------------------------------------------- #

def boundary_f1(pred, truth, tol: int = 1) -> float:
    """F1 of predicted vs planted boundaries, a prediction matching a truth within `tol`."""
    if len(pred) == 0 and len(truth) == 0:
        return 1.0
    if len(pred) == 0 or len(truth) == 0:
        return 0.0
    truth_left = set(truth)
    tp = 0
    for p in pred:
        hit = next((t for t in truth_left if abs(p - t) <= tol), None)
        if hit is not None:
            tp += 1
            truth_left.discard(hit)
    prec = tp / len(pred)
    rec = tp / len(truth)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# The showcase / finance dataset, and the grid table the viz mirrors.
# --------------------------------------------------------------------------- #

# A synthetic 10-K-like filing: sections of UNEVEN length (Business, Risk Factors,
# MD&A, Financials, Notes, Controls), each a coherent topical block. Uneven on purpose:
# equal-size chunking only hits boundaries by accident when sections are uniform.
FINANCE_SECTIONS = (4, 12, 3, 10, 8, 5)
VIZ_SECTIONS = (4, 8, 5, 10, 5)       # a shorter, uneven document for the boundary strip viz


def finance_document(seed: int = 1):
    return synthetic_document(FINANCE_SECTIONS, dim=64, kappa=14.0, seed=seed)


def grid_table() -> dict:
    """The numbers ChunkingLaboratory.tsx mirrors to the decimal. Deterministic.

    profile    : adjacent-gap dissimilarity along the (viz) document.
    labels     : planted segment id per sentence (for coloring the strip).
    truth      : planted internal boundaries.
    fixed      : fixed-size boundaries at the true segment count.
    perK       : per-K optimal (DP) and greedy boundaries + coherence cost.
    """
    emb, labels, truth = synthetic_document(VIZ_SECTIONS, dim=64, kappa=14.0, seed=3)
    n, true_k = len(emb), len(VIZ_SECTIONS)
    profile = [float(x) for x in adjacent_dissimilarity(emb)]
    perK = []
    for k in K_GRID:
        dp_b, dp_cost = segment_dp(emb, k)
        gr_b = texttiling_greedy(emb, k)
        perK.append({"k": k,
                     "dp_boundaries": dp_b, "dp_cost": round(dp_cost, 4),
                     "dp_f1": round(boundary_f1(dp_b, truth), 4),
                     "greedy_boundaries": gr_b,
                     "greedy_cost": round(total_coherence_cost(emb, gr_b), 4),
                     "greedy_f1": round(boundary_f1(gr_b, truth), 4)})
    fixed_b = fixed_size(n, true_k)
    return {
        "n": n, "true_k": true_k,
        "labels": [int(x) for x in labels],
        "truth": list(truth),
        "profile": profile,
        "fixed": fixed_b,
        "fixed_f1": round(boundary_f1(fixed_b, truth), 4),
        "perK": perK,
    }


def finance_demo() -> dict:
    """Headline: on a synthetic 10-K filing, the DP optimum recovers the section
    boundaries better than greedy and fixed-size, at the true section count."""
    emb, labels, truth = finance_document()
    k = len(FINANCE_SECTIONS)
    dp_b, dp_cost = segment_dp(emb, k)
    gr_b = texttiling_greedy(emb, k)
    fx_b = fixed_size(len(emb), k)
    out = {
        "n": len(emb), "k": k, "truth": truth,
        "dp_boundaries": dp_b, "greedy_boundaries": gr_b, "fixed_boundaries": fx_b,
        "dp_f1": boundary_f1(dp_b, truth), "greedy_f1": boundary_f1(gr_b, truth),
        "fixed_f1": boundary_f1(fx_b, truth),
        "dp_cost": dp_cost, "greedy_cost": total_coherence_cost(emb, gr_b),
    }
    print(f"  synthetic 10-K filing: {len(emb)} sentences, {k} sections "
          f"(SYNTHETIC vMF blocks, not a real filing):")
    print(f"  boundary F1 vs planted sections:  DP-optimal {out['dp_f1']*100:.0f}%"
          f"   greedy (TextTiling) {out['greedy_f1']*100:.0f}%   fixed-size {out['fixed_f1']*100:.0f}%")
    print(f"  within-chunk incoherence:  DP {out['dp_cost']:.3f}  <=  greedy {out['greedy_cost']:.3f} "
          f"(DP is optimal for the objective)")
    print("  -> coherence is a proxy: optimizing it recovers structure, but the true target is downstream retrieval.")
    return out


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_coherence_is_resultant_length() -> None:
    """The segment cost len - ||sum e_t|| equals sum_t (1 - cos(e_t, mu)) for the segment
    mean direction mu — so minimizing cost maximizes the mean resultant length (vMF tie)."""
    emb, _, _ = synthetic_document((6, 6), dim=48, kappa=10.0, seed=0)
    P = _prefix_sums(emb)
    for i, j in ((0, 6), (3, 11), (0, 12)):
        mu = emb[i:j].mean(axis=0)
        mu_hat = mu / np.linalg.norm(mu)
        direct = float(np.sum(1.0 - emb[i:j] @ mu_hat))
        assert abs(segment_cost(P, i, j) - direct) < 1e-9, "cost != sum of (1 - cos to mean direction)"
        assert abs((1 - segment_cost(P, i, j) / (j - i)) - mean_resultant_length(emb, i, j)) < 1e-9, \
            "1 - cost/len != mean resultant length"
    print("  [ok] coherence = mean resultant length: cost = len*(1 - Rbar) (the vMF concentration)")


def test_dp_matches_brute_force() -> None:
    """The DP segmentation attains the exact brute-force optimum for several k on a small
    document — the optimality theorem, checked against the oracle."""
    emb, _, _ = synthetic_document((4, 3, 5, 3), dim=32, kappa=12.0, seed=1)
    for k in (2, 3, 4, 5):
        _, dp_cost = segment_dp(emb, k)
        _, bf_cost = brute_force_segment(emb, k)
        assert abs(dp_cost - bf_cost) < 1e-9, f"DP != brute force at k={k}: {dp_cost} vs {bf_cost}"
    print("  [ok] DP optimality: matches exhaustive brute force at every k")


def test_dp_beats_heuristics() -> None:
    """At the same number of segments, the DP objective value is no worse than the greedy
    (TextTiling-style) and fixed-size segmentations — because the DP is globally optimal."""
    emb, _, _ = synthetic_document((6, 5, 7, 4, 6), dim=64, kappa=14.0, seed=2)
    k = 5
    _, dp_cost = segment_dp(emb, k)
    gr_cost = total_coherence_cost(emb, texttiling_greedy(emb, k))
    fx_cost = total_coherence_cost(emb, fixed_size(len(emb), k))
    assert dp_cost <= gr_cost + 1e-9, f"greedy beat DP: {gr_cost} < {dp_cost}"
    assert dp_cost <= fx_cost + 1e-9, f"fixed-size beat DP: {fx_cost} < {dp_cost}"
    print(f"  [ok] DP <= heuristics: cost DP {dp_cost:.2f} <= greedy {gr_cost:.2f}, fixed {fx_cost:.2f}")


def test_cost_monotone_in_k() -> None:
    """The optimal within-segment cost is monotone nonincreasing in the number of
    segments — more chunks can only reduce within-chunk incoherence (granularity tradeoff)."""
    emb, _, _ = synthetic_document((6, 5, 7, 5, 6, 5), dim=64, kappa=14.0, seed=4)
    costs = [segment_dp(emb, k)[1] for k in range(2, 12)]
    for a, b in zip(costs, costs[1:]):
        assert b <= a + 1e-9, f"optimal cost not monotone in k: {a:.3f} -> {b:.3f}"
    print(f"  [ok] cost monotone in #segments: {costs[0]:.2f} (k=2) down to {costs[-1]:.2f} (k=11)")


def test_dp_recovers_boundaries() -> None:
    """On a document with planted topic shifts, the DP optimum recovers the true
    boundaries at least as well as greedy and strictly better than fixed-size chunking."""
    emb, _, truth = synthetic_document((4, 10, 5, 12, 4), dim=64, kappa=16.0, seed=5)
    k = 5
    dp_f1 = boundary_f1(segment_dp(emb, k)[0], truth)
    gr_f1 = boundary_f1(texttiling_greedy(emb, k), truth)
    fx_f1 = boundary_f1(fixed_size(len(emb), k), truth)
    assert dp_f1 >= gr_f1, f"DP boundary F1 {dp_f1:.2f} below greedy {gr_f1:.2f}"
    assert dp_f1 > fx_f1, f"DP boundary F1 {dp_f1:.2f} not above fixed-size {fx_f1:.2f}"
    assert dp_f1 > 0.8, f"DP should recover most planted boundaries, got {dp_f1:.2f}"
    print(f"  [ok] boundary recovery: DP F1 {dp_f1*100:.0f}% >= greedy {gr_f1*100:.0f}% > fixed {fx_f1*100:.0f}%")


def test_finance_filing() -> None:
    """The synthetic 10-K: the DP recovers the section structure better than the baselines
    and attains the lowest within-chunk incoherence."""
    f = finance_demo()
    assert f["dp_f1"] >= f["greedy_f1"], "DP F1 below greedy on the filing"
    assert f["dp_f1"] > f["fixed_f1"], "DP F1 not above fixed-size on the filing"
    assert f["dp_cost"] <= f["greedy_cost"] + 1e-9, "DP cost not optimal on the filing"
    print("  [ok] finance: DP recovers sections best, lowest incoherence")


if __name__ == "__main__":
    print("Chunking / optimal segmentation verification harness")
    test_coherence_is_resultant_length()
    test_dp_matches_brute_force()
    test_dp_beats_heuristics()
    test_cost_monotone_in_k()
    test_dp_recovers_boundaries()
    tbl = grid_table()
    print("Grid table (mirrored by ChunkingLaboratory.tsx):")
    print(f"  document: {tbl['n']} sentences, planted boundaries {tbl['truth']}")
    print(f"  {'k':>4}{'dp_cost':>10}{'greedy_cost':>13}{'dp_boundaries':>22}")
    for row in tbl["perK"]:
        assert row["dp_cost"] <= row["greedy_cost"] + 1e-9
        print(f"  {row['k']:>4}{row['dp_cost']:>10.3f}{row['greedy_cost']:>13.3f}"
              f"{str(row['dp_boundaries']):>22}")
    print("Finance demo:")
    test_finance_filing()
    print("All checks passed.")
