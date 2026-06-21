"""Navigable small-world graphs and the mathematics of greedy routing — the reference
implementation for the formalRAG `navigable-small-world-graphs` topic.

The inverted file partitioned the space into cells; this topic replaces the flat partition with a
GRAPH, and search becomes a walk. The mathematics has two movements, and this module establishes,
and verifies, both plus the honest caveats they rest on:

  MOVEMENT 1 — THE MATHEMATICS OF GREEDY ROUTING (Kleinberg).
  1. NAVIGABILITY REQUIRES A DIMENSION-MATCHED LINK DISTRIBUTION. On a 2-D lattice where each node
     has its four grid neighbors plus ONE long-range link drawn with probability proportional to
     r^{-alpha} (r the lattice distance), decentralized greedy routing — always step to the neighbor
     nearest the target — is fast ONLY when alpha equals the lattice dimension (alpha = 2). Too
     uniform (alpha = 0) or too local (alpha large) and routing is far slower. We simulate the
     U-shaped delivery-time curve and confirm its minimum sits at alpha = 2. (`kleinberg_curve`,
     `test_kleinberg_navigability`)

  MOVEMENT 2 — THE NAVIGABLE SMALL-WORLD GRAPH FOR ANN.
  2. THE GRAPH IS BUILT BY INCREMENTAL INSERTION. Each point, as it is inserted, links to its M
     approximate nearest neighbors found by greedy search in the graph so far; early points become
     long-range hubs, so the small-world structure emerges from the construction order, with no
     lattice. (`build_nsw`)
  3. GREEDY HILL-CLIMBING GETS STUCK; A BEAM ESCAPES (the honest catch). Pure greedy search (beam
     width ef = 1) stops at a LOCAL minimum, so its recall is below 1; widening the beam recovers
     the true neighbors monotonically. (`greedy_walk`, `greedy_search`, `test_greedy_local_minimum`,
     `test_recall_monotone_in_ef`)
  4. THE SEARCH IS SUBLINEAR, AND THE GRAPH IS SMALL-WORLD. A query touches far fewer than n
     vectors, and the constructed graph has a short average path length — the long-range hubs make
     the diameter grow slowly, not linearly. (`test_nsw_sublinear_work`, `test_small_world_paths`)

Honest caveats (rigorFlag territory, asserted as DIRECTIONS): Kleinberg's polylogarithmic routing is
a theorem about the idealized lattice with the exact r^{-d} link law, NOT a guarantee for the
heuristic NSW graph, whose navigability is empirical; greedy search returns a local minimum, so
recall depends on the beam width ef, the degree M, and the entry point; and all numbers are measured
on a synthetic low-rank cloud from the prerequisite topic.

This module imports its prerequisite (`high_dimensional_geometry`) for the synthetic cloud and never
reimplements it. Every pedagogical claim is an `assert` below; `viz_constants()` prints what
`NavigableSmallWorldLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/navigable-small-world-graphs/navigable_small_world_graphs.py
"""
from __future__ import annotations

import heapq
import pathlib
import sys

import numpy as np
from scipy.spatial.distance import cdist

# NSW is a graph over embeddings whose navigability is governed by the concentration of distances —
# import the synthetic low-rank cloud from the prerequisite (the established cross-topic pattern: add
# the prereq's HYPHENATED dir to the path, import its UNDERSCORED module).
_HDG_DIR = pathlib.Path(__file__).resolve().parents[1] / "high-dimensional-geometry"
if str(_HDG_DIR) not in sys.path:
    sys.path.insert(0, str(_HDG_DIR))
from high_dimensional_geometry import structured_data  # noqa: E402


# --------------------------------------------------------------------------- #
# Movement 1 — Kleinberg's ring and the mathematics of greedy routing. We use the
# one-dimensional ring (the Watts-Strogatz / Kleinberg-1D model), where the optimal
# exponent equals the dimension, alpha = 1: it constructs in O(n), so n can be large
# enough for the asymptotic navigability separation to appear cleanly.
# --------------------------------------------------------------------------- #

def ring_distance(a: int, b: int, n: int) -> int:
    """Distance on the cycle of n nodes: min(|a-b|, n-|a-b|)."""
    d = abs(a - b)
    return min(d, n - d)


def build_kleinberg_ring(n: int, alpha: float, seed: int = 0) -> np.ndarray:
    """A ring of n nodes; each keeps its two ring neighbors and gains ONE long-range contact at a
    forward offset drawn with probability proportional to (ring distance)^{-alpha} (alpha = 0 is
    uniform; large alpha is nearly local). The ring's translation symmetry makes the offset
    distribution shared across nodes, so construction is O(n). Returns longrange (n,). GUARD:
    n >= 4."""
    if n < 4:
        raise ValueError(f"n must be >= 4, got {n}")
    offsets = np.arange(1, n)
    w = np.minimum(offsets, n - offsets).astype(float) ** (-alpha)  # ring distance ^ -alpha, all > 0
    w /= w.sum()
    off = np.random.default_rng(seed).choice(offsets, size=n, p=w)
    return (np.arange(n) + off) % n


def greedy_route_ring(src: int, tgt: int, longrange: np.ndarray, n: int, max_hops: int = 100000) -> int:
    """Decentralized greedy routing on the ring: from src, step to whichever neighbor (two ring
    neighbors + one long-range contact) is nearest tgt in ring distance, until reaching tgt.
    Returns the hop count; a local step toward tgt always exists, so greedy terminates."""
    u, hops = src, 0
    while u != tgt and hops < max_hops:
        cand = ((u - 1) % n, (u + 1) % n, int(longrange[u]))
        u = min(cand, key=lambda c: ring_distance(c, tgt, n))
        hops += 1
    return hops


def expected_delivery_time(n: int, alpha: float, trials: int = 400, seed: int = 0) -> float:
    """Mean greedy hop count over random source-target pairs on the alpha-augmented ring."""
    longrange = build_kleinberg_ring(n, alpha, seed=seed)
    rng = np.random.default_rng(seed + 1)
    hops = []
    while len(hops) < trials:
        s, t = int(rng.integers(0, n)), int(rng.integers(0, n))
        if s != t:
            hops.append(greedy_route_ring(s, t, longrange, n))
    return float(np.mean(hops))


def kleinberg_curve(n: int, alphas, trials: int = 400, seed: int = 0):
    """Mean greedy delivery time as a function of the long-range exponent alpha — the U-curve whose
    minimum sits at alpha = the ring's dimension (1). Returns a list of row dicts."""
    return [{"alpha": a, "hops": expected_delivery_time(n, a, trials=trials, seed=seed)} for a in alphas]


# --------------------------------------------------------------------------- #
# Movement 2 — the navigable small-world graph over embeddings.
# --------------------------------------------------------------------------- #

def _sqd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff @ diff)


def _true_topk(queries: np.ndarray, X: np.ndarray, topk: int):
    """Exact top-k nearest neighbors (sets of indices) by brute force — the recall ground truth."""
    d2 = cdist(queries, X, "sqeuclidean")
    return [set(np.argpartition(r, topk)[:topk].tolist()) for r in d2]


def nsw_dataset(n: int = 500, nq: int = 40, d: int = 20, k: int = 5, seed: int = 0):
    """A low-rank cloud from the prerequisite topic, split into a database and held-out queries
    drawn from the SAME subspace (one structured_data call, so queries are in-distribution)."""
    data = structured_data(n + nq, d, k, seed=seed)
    return data[:n], data[n:]


def greedy_search(X: np.ndarray, adj, q: np.ndarray, entry: int, ef: int, topk=None):
    """The NSW/HNSW beam search. Maintain a candidate min-heap and a result max-heap of size ef;
    from the entry node, expand the nearest unexplored candidate until it is farther than the worst
    kept result. Returns (result indices nearest-first, n_distance_computations). ef = 1 is pure
    greedy hill-climbing. GUARD: ef >= 1."""
    if ef < 1:
        raise ValueError(f"ef must be >= 1, got {ef}")
    de = _sqd(q, X[entry])
    visited = {entry}
    candidates = [(de, entry)]          # min-heap by distance to q
    results = [(-de, entry)]            # max-heap (negated) of the ef best so far
    ndist = 1
    while candidates:
        d, c = heapq.heappop(candidates)
        if d > -results[0][0] and len(results) >= ef:
            break                       # nearest candidate worse than the worst result -> done
        for nb in adj[c]:
            if nb in visited:
                continue
            visited.add(nb)
            dn = _sqd(q, X[nb])
            ndist += 1
            if dn < -results[0][0] or len(results) < ef:
                heapq.heappush(candidates, (dn, nb))
                heapq.heappush(results, (-dn, nb))
                if len(results) > ef:
                    heapq.heappop(results)
    ordered = [i for _, i in sorted((-nd, i) for nd, i in results)]
    return (ordered[:topk] if topk else ordered), ndist


def build_nsw(X: np.ndarray, M: int = 8, ef_construction: int = 16, seed: int = 0):
    """Incremental NSW construction. Insert the points in a random order; each new point greedy-
    searches the graph built so far and links bidirectionally to its M nearest. Early insertions
    become long-range hubs. Returns (adj, entry) where adj is a list of neighbor sets and entry is
    the first inserted node (a natural hub). GUARDS: M >= 1, ef_construction >= 1."""
    if M < 1 or ef_construction < 1:
        raise ValueError(f"M and ef_construction must be >= 1, got M={M}, ef={ef_construction}")
    n = X.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    adj = [set() for _ in range(n)]
    inserted = []
    for idx in order:
        if inserted:
            cand, _ = greedy_search(X, adj, X[idx], entry=inserted[0], ef=ef_construction)
            for j in sorted(cand, key=lambda j: _sqd(X[idx], X[j]))[:M]:
                adj[idx].add(j)
                adj[j].add(idx)
        inserted.append(int(idx))
    return adj, inserted[0]


def greedy_walk(X: np.ndarray, adj, q: np.ndarray, entry: int):
    """Pure greedy hill-climb (ef = 1): from entry, step to the neighbor nearest q while it
    improves; stop at a local minimum. Returns the path of node indices. Its terminal node is the
    greedy answer — possibly a LOCAL minimum, not the true nearest neighbor."""
    u, du, path = entry, _sqd(q, X[entry]), [entry]
    while True:
        best, bd = u, du
        for nb in adj[u]:
            d = _sqd(q, X[nb])
            if d < bd:
                best, bd = nb, d
        if best == u:
            break
        u, du = best, bd
        path.append(u)
    return path


def nsw_recall(X, queries, adj, entry, ef, topk: int = 10, truth=None):
    """Mean recall@topk and mean distance computations per query under greedy beam search. GUARD:
    non-empty queries, topk >= 1."""
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if truth is None:
        truth = _true_topk(queries, X, topk)
    hits, work = 0, 0
    for qi in range(len(queries)):
        idxs, ndist = greedy_search(X, adj, queries[qi], entry, ef, topk=topk)
        hits += len(truth[qi] & set(idxs))
        work += ndist
    return hits / (len(queries) * topk), work / len(queries)


def recall_vs_ef(X, queries, adj, entry, ef_grid, topk: int = 10):
    """For each beam width ef: recall@topk and mean distance computations — the search's speed/
    recall frontier. Returns a list of row dicts."""
    truth = _true_topk(queries, X, topk)
    rows = []
    for ef in ef_grid:
        recall, work = nsw_recall(X, queries, adj, entry, ef, topk, truth=truth)
        rows.append({"ef": ef, "recall": recall, "work": work})
    return rows


def average_path_length(adj, sample: int = 60, seed: int = 0) -> float:
    """Mean shortest-path length (in hops) over random node pairs, by BFS — the small-world
    diameter proxy. GUARD: unreachable pairs are skipped."""
    n = len(adj)
    rng = np.random.default_rng(seed)
    lengths = []
    for _ in range(sample):
        s = int(rng.integers(0, n))
        dist = {s: 0}
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        t = int(rng.integers(0, n))
        if t in dist and t != s:
            lengths.append(dist[t])
    return float(np.mean(lengths)) if lengths else 0.0


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

RING_N, ALPHA_GRID = 20000, (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
NSW_M, NSW_EFC, EF_GRID = 8, 16, (1, 2, 4, 8, 16, 32, 64)


def test_kleinberg_navigability() -> None:
    """Greedy routing on the ring is fastest when the long-range exponent matches the ring's
    dimension (alpha = 1): a too-uniform (alpha = 0) or too-local (alpha = 2) link law routes far
    slower. The navigability sweet spot. DIRECTION."""
    h0 = expected_delivery_time(RING_N, 0.0, trials=400)
    h1 = expected_delivery_time(RING_N, 1.0, trials=400)
    h2 = expected_delivery_time(RING_N, 2.0, trials=400)
    assert h1 < h0 and h1 < h2, f"alpha=1 hops {h1:.1f} not the minimum (alpha0 {h0:.1f}, alpha2 {h2:.1f})"
    print(f"  [ok] Kleinberg navigability: greedy hops minimized at alpha=1 "
          f"({h1:.1f}) vs alpha=0 ({h0:.1f}), alpha=2 ({h2:.1f})")


def test_greedy_local_minimum() -> None:
    """Pure greedy search (ef = 1) gets stuck in local minima, so its recall is below 1; a wider
    beam recovers the neighbors. The honest catch of graph search."""
    X, queries = nsw_dataset()
    adj, entry = build_nsw(X, NSW_M, NSW_EFC, seed=0)
    r1, _ = nsw_recall(X, queries, adj, entry, ef=1, topk=10)
    assert r1 < 1.0 - 1e-6, f"recall@ef=1 {r1:.4f} is not below 1 — no local-minimum effect to show"
    print(f"  [ok] greedy local minimum: recall@ef=1 = {r1:.3f} < 1 (hill-climbing gets stuck)")


def test_recall_monotone_in_ef() -> None:
    """Recall is non-decreasing in the beam width ef: a wider search never loses a neighbor."""
    X, queries = nsw_dataset()
    adj, entry = build_nsw(X, NSW_M, NSW_EFC, seed=0)
    rec = np.array([r["recall"] for r in recall_vs_ef(X, queries, adj, entry, EF_GRID)])
    assert np.all(np.diff(rec) >= -1e-9), f"recall not monotone in ef: {np.round(rec, 3)}"
    print(f"  [ok] recall monotone in ef: {np.round(rec, 3).tolist()} (ef={list(EF_GRID)})")


def test_nsw_sublinear_work() -> None:
    """At a moderate beam width the search touches far fewer than n vectors — sublinear work,
    the point of the graph index."""
    X, queries = nsw_dataset()
    adj, entry = build_nsw(X, NSW_M, NSW_EFC, seed=0)
    _, work = nsw_recall(X, queries, adj, entry, ef=16, topk=10)
    assert work < 0.5 * X.shape[0], f"ef=16 touches {work:.0f} of {X.shape[0]} — not sublinear enough"
    print(f"  [ok] sublinear work: ef=16 touches {work:.0f} of {X.shape[0]} vectors per query")


def test_small_world_paths() -> None:
    """The constructed graph is a small world: its mean shortest-path length is short — the
    long-range hubs make the diameter grow slowly, well below a fraction of n."""
    X, _ = nsw_dataset()
    adj, _ = build_nsw(X, NSW_M, NSW_EFC, seed=0)
    apl = average_path_length(adj)
    assert 0 < apl < 0.05 * X.shape[0], f"mean path length {apl:.2f} is not small-world for n={X.shape[0]}"
    print(f"  [ok] small-world graph: mean shortest-path length {apl:.2f} hops for n={X.shape[0]} "
          f"(~{np.log(X.shape[0]):.1f} = log n)")


def validate_nsw_high_ef() -> None:
    """At a large beam width, greedy search recovers nearly all true neighbors — the graph search
    is exact in the limit, so the only loss is the greedy approximation that ef controls."""
    X, queries = nsw_dataset()
    adj, entry = build_nsw(X, NSW_M, NSW_EFC, seed=0)
    r1, _ = nsw_recall(X, queries, adj, entry, ef=1, topk=10)
    rbig, _ = nsw_recall(X, queries, adj, entry, ef=128, topk=10)
    assert rbig > 0.9 and rbig > r1, f"recall@ef=128 {rbig:.3f} not near-exact / not above ef=1 {r1:.3f}"
    print(f"  [ok] cross-check: recall climbs {r1:.3f} (ef=1) -> {rbig:.3f} (ef=128, near-exact)")


# --------------------------------------------------------------------------- #
# Toy graph for the laboratory and the viz constants.
# --------------------------------------------------------------------------- #

def toy_nsw_graph(seed: int = 2):
    """A small 2-D cloud and its NSW graph for the laboratory: returns (X2, adj, entry, query, nn,
    walk, found_idx). The query is chosen so pure greedy (ef=1) stops at a local minimum that is NOT
    the true nearest neighbor, which a wider beam finds — the boundary the panel illustrates."""
    rng = np.random.default_rng(seed)
    X2 = np.clip(np.vstack([c + 0.9 * rng.standard_normal((9, 2))
                            for c in ([2, 8], [8, 8], [2, 2], [8, 2], [5, 5])]), 0, 10)
    adj, entry = build_nsw(X2, M=3, ef_construction=8, seed=seed)
    # search for a query whose greedy walk terminus is not its true NN
    best = None
    for _ in range(300):
        q = rng.uniform(1.5, 8.5, size=2)
        nn = int(cdist(q[None, :], X2, "sqeuclidean")[0].argmin())
        walk = greedy_walk(X2, adj, q, entry)
        found = greedy_search(X2, adj, q, entry, ef=16, topk=1)[0][0]
        if best is None:
            best = (q, nn, walk, found)
        if walk[-1] != nn and found == nn:
            return X2, adj, entry, q, nn, walk, found
    return (X2, adj, entry, *best[1:])


def viz_constants() -> None:
    """Print the Kleinberg U-curve (Panel A), the toy NSW graph + greedy walk (Panel B), and the
    recall/ef frontier (Panel C) — all baked to the decimal in the .tsx."""
    print(f"  PANEL A — Kleinberg greedy-routing U-curve (ring, n={RING_N}):")
    for r in kleinberg_curve(RING_N, ALPHA_GRID):
        print(f"    alpha={r['alpha']:>4}: mean_hops={r['hops']:.3f}")

    X2, adj, entry, q, nn, walk, found = toy_nsw_graph()
    edges = sorted({(int(min(u, v)), int(max(u, v))) for u in range(len(adj)) for v in adj[u]})
    walk = [int(w) for w in walk]
    print("  PANEL B — toy NSW graph + greedy walk:")
    print(f"    TOY_POINTS ({len(X2)} pts in R^2) = {[[round(float(v), 3) for v in p] for p in X2]}")
    print(f"    EDGES ({len(edges)}) = {[list(e) for e in edges]}")
    print(f"    ENTRY={int(entry)}  QUERY={[round(float(v), 3) for v in q]}  TRUE_NN={int(nn)}")
    print(f"    GREEDY_WALK (ef=1) = {walk}  (terminus {walk[-1]} {'=' if walk[-1] == nn else '!='} NN)")
    print(f"    BEAM_FOUND (ef=16) = {int(found)}")

    X, queries = nsw_dataset()
    adjf, entryf = build_nsw(X, NSW_M, NSW_EFC, seed=0)
    print(f"  PANEL C — recall / work frontier (n={X.shape[0]}, M={NSW_M}):")
    for r in recall_vs_ef(X, queries, adjf, entryf, EF_GRID):
        print(f"    ef={r['ef']:>3}: recall={r['recall']:.4f}  dist_comps={r['work']:.1f}")


if __name__ == "__main__":
    print("Navigable small-world graphs / greedy routing verification harness")
    test_kleinberg_navigability()
    test_greedy_local_minimum()
    test_recall_monotone_in_ef()
    test_nsw_sublinear_work()
    test_small_world_paths()
    validate_nsw_high_ef()
    print("Viz constants (mirrored to the decimal in NavigableSmallWorldLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
