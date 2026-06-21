"""Hierarchical Navigable Small-World graphs (HNSW) — the reference implementation for the formalRAG
`hnsw` topic.

The prerequisite gave us a single navigable graph and a beam search that walks it; its honest catch is
that the walk starts at an arbitrary entry and the small-world property is empirical. HNSW adds ONE
structural idea — a randomized hierarchy of nested graphs — that turns the arbitrary entry into a
provably logarithmic descent. The mathematics has three movements, and this module establishes and
verifies each plus the honest caveats it rests on:

  MOVEMENT 1 — THE LEVEL-ASSIGNMENT LAW (the provable spine). Each node draws a maximum level
    L = floor(-ln(U) * mL), U ~ Uniform(0,1), mL = 1/ln(M), and is inserted into the graphs at every
    layer 0..L. Then P(L >= l) = M^{-l} exactly, layer occupancy decays geometrically by 1/M, the top
    layer holds O(1) nodes, and the expected maximum level — hence the greedy entry-descent depth —
    grows like log_M(n). This is exact probability, the analogue of the prerequisite's Kleinberg
    theorem. (`assign_level`, `default_mL`, `test_level_tail_law`, `test_layer_geometric_decay`,
    `test_top_layer_is_O1`, `test_max_level_scales_log`)

  MOVEMENT 2 — HEURISTIC NEIGHBOR SELECTION (the centerpiece, and a heuristic). What distinguishes
    HNSW from "link to the M nearest" is Malkov-Yashunin Algorithm 4: scanning candidates by
    increasing distance to the base, admit one only if no already-kept neighbor is closer to it than
    it is to the base. Strict-M-nearest clusters links on one side; the heuristic spreads them and
    preserves the long-range links that keep the graph a small world. No optimality proof.
    (`select_neighbors_heuristic`, `test_heuristic_diversifies`)

  MOVEMENT 3 — HIERARCHICAL CONSTRUCTION AND SEARCH, AND THE IVF HEAD-TO-HEAD. Construction inserts
    each point at its level with the prerequisite's beam restricted to a per-layer adjacency; search
    descends greedily (beam width 1) through the sparse upper layers to refine the entry, then runs a
    width-ef beam at layer 0. The hierarchy's payoff is intra-graph and robust: HNSW reaches a given
    recall at no more distance computations than the flat NSW it layers. We close the arc by building
    HNSW, flat NSW, AND the inverted file (IVF) on the SAME cloud with one shared ground truth, and
    tracing recall-versus-cost frontiers. (`build_hnsw`, `search_layer`, `search_hnsw`, `head_to_head`,
    `test_search_layer_matches_flat_on_single_layer`, `test_recall_monotone_in_ef_hnsw`,
    `test_hnsw_beats_flat_nsw_at_equal_cost`, `test_both_reach_exact_at_full_cost`,
    `test_hnsw_vs_ivf_on_one_cloud`)

Honest caveats (rigorFlag territory, asserted as DIRECTIONS): the level law is exact, but the
end-to-end log-n search cost rests on each layer being navigable, which on real embeddings is
empirical, inherited from the NSW heuristic, not a theorem; the neighbor-selection heuristic has no
optimality proof; recall is empirical and depends on M, ef, mL and the entry; and the head-to-head is
one synthetic low-rank cloud with a shared ground truth, a statement about this cloud, not a universal
ranking.

This module imports its prerequisites (`navigable_small_world_graphs`, `ivf_voronoi_partitioning`) and
their ancestors, and never reimplements them. Every pedagogical claim is an `assert` below;
`viz_constants()` prints what `HNSWLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/hnsw/hnsw.py
"""

from __future__ import annotations

import heapq
import pathlib
import sys

import numpy as np
from scipy.spatial.distance import cdist

# HNSW is "NSW + a hierarchy", and the arc closes against the inverted file — import BOTH prerequisites
# and every ancestor they pull in (the established cross-topic pattern: add each prereq's HYPHENATED
# dir to the path, import its UNDERSCORED module). NSW pulls high-dimensional-geometry for the cloud;
# IVF pulls product-quantization and vector-quantization-lloyd-max.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("navigable-small-world-graphs", "high-dimensional-geometry",
             "ivf-voronoi-partitioning", "product-quantization", "vector-quantization-lloyd-max"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from navigable_small_world_graphs import (  # noqa: E402
    _sqd,
    _true_topk,
    build_nsw,
    greedy_search,
    nsw_dataset,
    nsw_recall,
)
from ivf_voronoi_partitioning import (  # noqa: E402
    candidate_fraction,
    coarse_quantizer,
    inverted_lists,
    ivf_recall,
)


# --------------------------------------------------------------------------- #
# Movement 1 — the level-assignment law.
# --------------------------------------------------------------------------- #

def default_mL(M: int) -> float:
    """The level multiplier mL = 1/ln(M) that makes P(level >= l) = M^{-l}. GUARD: M >= 2 (for M=1,
    ln M = 0 and the law degenerates to all-zero)."""
    if M < 2:
        raise ValueError(f"M must be >= 2 for the level law, got {M}")
    return 1.0 / np.log(M)


def assign_level(rng: np.random.Generator, mL: float) -> int:
    """One geometric level draw L = floor(-ln(U) * mL), U ~ Uniform(0,1), from the SHARED rng stream.
    U is clamped off 0 so -ln(U) is finite. GUARD: mL > 0."""
    if mL <= 0:
        raise ValueError(f"mL must be > 0, got {mL}")
    u = max(float(rng.random()), 1e-12)
    return int(np.floor(-np.log(u) * mL))


# --------------------------------------------------------------------------- #
# Movement 2 — heuristic neighbor selection (Malkov-Yashunin Algorithm 4).
# --------------------------------------------------------------------------- #

def select_neighbors_heuristic(X: np.ndarray, base: int, candidates, M: int):
    """Keep up to M of `candidates`, scanning by increasing distance to `base`: admit a candidate c
    ONLY IF no already-kept neighbor r is closer to c than c is to base (d(c,r) >= d(c,base) for all
    kept r). This preserves diverse / long-range links instead of the M strict-nearest, which cluster
    on one side. Returns the kept node ids. GUARDS: c != base; empty -> []; never disconnect (fall
    back to the single nearest if the rule admits nothing)."""
    cands = [int(c) for c in candidates if int(c) != int(base)]
    if not cands:
        return []
    cands.sort(key=lambda c: _sqd(X[base], X[c]))
    kept: list[int] = []
    for c in cands:
        if len(kept) >= M:
            break
        d_cb = _sqd(X[c], X[base])
        if all(_sqd(X[c], X[r]) >= d_cb for r in kept):
            kept.append(c)
    if not kept:
        kept.append(cands[0])
    return kept


# --------------------------------------------------------------------------- #
# Movement 3 — per-layer beam search, hierarchical construction and search.
# --------------------------------------------------------------------------- #

def search_layer(X: np.ndarray, layer_adj: dict, q: np.ndarray, entry_points, ef: int):
    """One-layer beam search: the prerequisite's NSW beam (candidate min-heap + result max-heap of
    size ef), but over a per-layer adjacency DICT (layer_adj.get(c, ()) so nodes absent from the layer
    contribute nothing) and seeded from a SET of entry points (HNSW hands the upper layer's result set
    down as the lower layer's entries). Returns (ordered node ids nearest-first, n_distance_comps).
    GUARD: ef >= 1; non-empty entry_points."""
    if ef < 1:
        raise ValueError(f"ef must be >= 1, got {ef}")
    visited: set[int] = set()
    candidates: list[tuple[float, int]] = []   # min-heap by distance to q
    results: list[tuple[float, int]] = []      # max-heap (negated) of the ef best so far
    ndist = 0
    for e in entry_points:
        e = int(e)
        if e in visited:
            continue
        visited.add(e)
        de = _sqd(q, X[e])
        ndist += 1
        heapq.heappush(candidates, (de, e))
        heapq.heappush(results, (-de, e))
    while len(results) > ef:
        heapq.heappop(results)
    if not candidates:
        raise ValueError("entry_points must be non-empty")
    while candidates:
        d, c = heapq.heappop(candidates)
        if d > -results[0][0] and len(results) >= ef:
            break                       # nearest candidate worse than the worst result -> done
        for nb in layer_adj.get(c, ()):
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
    return ordered, ndist


def build_hnsw(X: np.ndarray, M: int = 8, ef_construction: int = 16, mL=None, seed: int = 0):
    """Hierarchical construction (Malkov-Yashunin Algorithm 1). Insert points in a random order; each
    draws a level L via assign_level from ONE shared rng. A new point greedy-descends (beam 1) from the
    global entry through the layers above L, then beam-inserts (beam ef_construction) at each layer from
    min(L, top) down to 0, linking to neighbors chosen by select_neighbors_heuristic; per-layer degree
    is capped (Mmax = M per upper layer, Mmax0 = 2M at layer 0) by re-running the heuristic on an
    over-full neighborhood. Returns (layers, entry_point, top_level), where layers[l] is a dict
    {node -> set(neighbors)} holding only the nodes present at level l. GUARDS: M >= 2,
    ef_construction >= 1, n >= 1."""
    if M < 2:
        raise ValueError(f"M must be >= 2, got {M}")
    if ef_construction < 1:
        raise ValueError(f"ef_construction must be >= 1, got {ef_construction}")
    n = X.shape[0]
    if n < 1:
        raise ValueError("X must contain at least one vector")
    mL = default_mL(M) if mL is None else mL
    if mL <= 0:
        raise ValueError(f"mL must be > 0, got {mL}")
    rng = np.random.default_rng(seed)              # ONE stream: permutation AND every level draw
    order = rng.permutation(n)
    layers: list[dict] = []
    entry_point = None
    top_level = -1

    def ensure(level: int) -> None:
        while len(layers) <= level:
            layers.append({})

    for raw in order:
        idx = int(raw)
        L = assign_level(rng, mL)
        ensure(L)
        if entry_point is None:
            for lev in range(L + 1):
                layers[lev][idx] = set()
            entry_point, top_level = idx, L
            continue
        # Phase 1 — greedy beam-1 descent through the layers above the insertion level.
        ep_nodes = {entry_point}
        for lev in range(top_level, L, -1):
            ids, _ = search_layer(X, layers[lev], X[idx], ep_nodes, ef=1)
            ep_nodes = {ids[0]}
        # Phase 2 — insert at every layer from min(L, top_level) down to 0.
        for lev in range(min(L, top_level), -1, -1):
            layers[lev].setdefault(idx, set())
            W, _ = search_layer(X, layers[lev], X[idx], ep_nodes, ef=ef_construction)
            Mmax = 2 * M if lev == 0 else M
            for j in select_neighbors_heuristic(X, idx, W, Mmax):
                layers[lev][idx].add(j)
                layers[lev][j].add(idx)
                if len(layers[lev][j]) > Mmax:    # shrink an over-full neighborhood with the heuristic
                    kept = set(select_neighbors_heuristic(X, j, list(layers[lev][j]), Mmax))
                    for r in list(layers[lev][j]):
                        if r not in kept:
                            layers[lev][j].discard(r)
                            layers[lev][r].discard(j)
            ep_nodes = set(W) if W else ep_nodes
        if L > top_level:                          # a new tallest node becomes the entry point
            for lev in range(top_level + 1, L + 1):
                layers[lev].setdefault(idx, set())
            entry_point, top_level = idx, L
    return layers, entry_point, top_level


def search_hnsw(layers, X: np.ndarray, q: np.ndarray, entry: int, top_level: int, ef: int,
                topk: int = 10):
    """Hierarchical search (Malkov-Yashunin Algorithm 5): greedy beam-1 descent through layers
    top_level..1 refines the entry, then a width-ef beam at layer 0 returns the nearest. Returns
    (top-k node ids, TOTAL distance computations across all layers). GUARDS: topk capped at the
    layer-0 occupancy; ef >= 1."""
    ep = int(entry)
    ndist = 0
    for lev in range(top_level, 0, -1):
        ids, nd = search_layer(X, layers[lev], q, {ep}, ef=1)
        ndist += nd
        ep = ids[0]
    n0 = len(layers[0]) if layers else 0
    k = min(topk, n0) if n0 else topk
    ids, nd = search_layer(X, layers[0], q, {ep}, ef=ef)
    ndist += nd
    return ids[:k], ndist


def hnsw_recall(X, queries, layers, entry, top_level, ef, topk: int = 10, truth=None):
    """Mean recall@topk and mean distance computations per query under hierarchical search. GUARD:
    non-empty queries, topk >= 1."""
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if truth is None:
        truth = _true_topk(queries, X, topk)
    hits, work = 0, 0
    for qi in range(len(queries)):
        idxs, ndist = search_hnsw(layers, X, queries[qi], entry, top_level, ef, topk=topk)
        hits += len(truth[qi] & set(idxs))
        work += ndist
    return hits / (len(queries) * topk), work / len(queries)


def recall_vs_ef_hnsw(X, queries, layers, entry, top_level, ef_grid, topk: int = 10):
    """For each beam width ef: recall@topk and mean distance computations — the hierarchical search's
    speed/recall frontier. Returns a list of row dicts."""
    truth = _true_topk(queries, X, topk)
    rows = []
    for ef in ef_grid:
        recall, work = hnsw_recall(X, queries, layers, entry, top_level, ef, topk, truth=truth)
        rows.append({"ef": ef, "recall": recall, "work": work})
    return rows


def layer_node_counts(layers) -> list[int]:
    """Number of nodes present at each layer, bottom (0) to top — the geometric-decay profile."""
    return [len(layer) for layer in layers]


# --------------------------------------------------------------------------- #
# Scaling: the maximum level depends only on the level draws, not the graph — so log_M(n) scaling is
# cheap to measure (draw n levels, take the max, average over trials).
# --------------------------------------------------------------------------- #

def expected_top_level(n: int, M: int, trials: int = 40, seed: int = 0) -> float:
    """Mean maximum level over `trials` independent populations of n level draws (one shared stream).
    Tracks log_M(n) + O(1) — the expected entry-descent depth."""
    mL = default_mL(M)
    rng = np.random.default_rng(seed)
    tops = []
    for _ in range(trials):
        u = np.maximum(rng.random(n), 1e-12)
        tops.append(int(np.floor(-np.log(u) * mL).max()))
    return float(np.mean(tops))


def scaling_study(n_grid, M: int = 8, trials: int = 40, seed: int = 0):
    """Mean top level vs log_M(n) across a sweep of n — Panel C's scaling sub-panel."""
    rows = []
    for n in n_grid:
        rows.append({"n": int(n),
                     "top_level": expected_top_level(n, M, trials, seed),
                     "log_M_n": float(np.log(n) / np.log(M))})
    return rows


# --------------------------------------------------------------------------- #
# The provably-one-cloud head-to-head: HNSW vs flat NSW vs IVF on the SAME (X, queries).
# --------------------------------------------------------------------------- #

def head_to_head(X, queries, topk: int = 10, seed: int = 0,
                 ef_grid=(1, 2, 4, 8, 16, 32, 64, 128),
                 nprobe_grid=(1, 2, 4, 8, 16, 22), nlist=None,
                 M: int = 8, ef_construction: int = 16):
    """Build HNSW, flat NSW, AND IVF on the SAME (X, queries); re-derive ONE shared ground truth; sweep
    efSearch (HNSW/NSW) and nprobe (IVF) to trace recall-vs-cost frontiers. Cost = mean exact distance
    computations per query (HNSW/NSW: returned ndist; IVF: candidate_fraction*n + nlist coarse comps).
    Returns {'hnsw','nsw','ivf': [rows], 'n','topk','nlist'}. GUARDS: topk<=n; nlist in [1, n]."""
    n = X.shape[0]
    topk = min(topk, n)
    truth = _true_topk(queries, X, topk)

    layers, entry, top_level = build_hnsw(X, M=M, ef_construction=ef_construction, seed=seed)
    hnsw_rows = []
    for ef in ef_grid:
        rec, work = hnsw_recall(X, queries, layers, entry, top_level, ef, topk, truth=truth)
        hnsw_rows.append({"ef": int(ef), "recall": rec, "cost": work})

    adj, nentry = build_nsw(X, M=M, ef_construction=ef_construction, seed=seed)
    nsw_rows = []
    for ef in ef_grid:
        rec, work = nsw_recall(X, queries, adj, nentry, ef, topk, truth=truth)
        nsw_rows.append({"ef": int(ef), "recall": rec, "cost": work})

    if nlist is None:
        nlist = int(round(np.sqrt(n)))
    nlist = int(min(max(nlist, 1), n))
    C = coarse_quantizer(X, nlist, seed=seed)
    _, lists = inverted_lists(X, C)
    ivf_rows = []
    for nprobe in nprobe_grid:
        if nprobe > nlist:
            continue
        rec = ivf_recall(queries, X, C, lists, nprobe, topk, truth=truth)
        frac = candidate_fraction(queries, C, lists, nprobe)
        ivf_rows.append({"nprobe": int(nprobe), "recall": float(rec), "cost": float(frac * n + nlist)})

    return {"hnsw": hnsw_rows, "nsw": nsw_rows, "ivf": ivf_rows,
            "n": n, "topk": topk, "nlist": nlist}


def _cost_at_recall(rows, r_target: float):
    """Min cost among frontier rows that reach recall >= r_target (None if none do)."""
    costs = [r["cost"] for r in rows if r["recall"] >= r_target - 1e-9]
    return min(costs) if costs else None


def _recall_at_cost(rows, budget: float) -> float:
    """Best recall achievable at cost <= budget on a frontier (0.0 if the cheapest row exceeds it)."""
    feasible = [r["recall"] for r in rows if r["cost"] <= budget + 1e-9]
    return max(feasible) if feasible else 0.0


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

HNSW_M, HNSW_EFC, EF_GRID = 8, 16, (1, 2, 4, 8, 16, 32, 64, 128)


def test_level_tail_law() -> None:
    """The level law is exact: P(L >= l) = M^{-l}. A large Monte-Carlo sample matches each tail
    probability within a 3-sigma binomial band. The provable spine."""
    M, N = 8, 300_000
    mL = default_mL(M)
    rng = np.random.default_rng(0)
    u = np.maximum(rng.random(N), 1e-12)
    L = np.floor(-np.log(u) * mL)
    for l in (1, 2, 3):
        emp = float((L >= l).mean())
        theo = M ** (-l)
        tol = 3.0 * np.sqrt(theo * (1 - theo) / N) + 1e-4
        assert abs(emp - theo) < tol, f"P(L>={l}) empirical {emp:.5f} vs theory {theo:.5f} (tol {tol:.5f})"
    print(f"  [ok] level tail law: P(L>=1,2,3) = "
          f"{[round(float((L >= l).mean()), 5) for l in (1, 2, 3)]} ~ "
          f"{[round(8.0 ** -l, 5) for l in (1, 2, 3)]} = M^-l")


def test_layer_geometric_decay() -> None:
    """In a built index, per-layer node counts are non-increasing and thin by roughly the factor 1/M
    from each layer to the next — the geometric occupancy the level law predicts. DIRECTION."""
    X, _ = nsw_dataset()
    layers, _, _ = build_hnsw(X, HNSW_M, HNSW_EFC, seed=0)
    counts = layer_node_counts(layers)
    assert counts[0] == X.shape[0], f"layer 0 must hold all {X.shape[0]} nodes, got {counts[0]}"
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), f"occupancy not monotone: {counts}"
    # the 0->1 thinning ratio should be near 1/M (the dominant, best-sampled transition)
    ratio01 = counts[1] / counts[0]
    assert ratio01 < 1.0 / HNSW_M + 0.06, f"layer 0->1 ratio {ratio01:.3f} not ~1/M={1 / HNSW_M:.3f}"
    print(f"  [ok] geometric decay: layer counts {counts}, 0->1 ratio {ratio01:.3f} ~ 1/M={1 / HNSW_M:.3f}")


def test_top_layer_is_O1() -> None:
    """The apex is a handful of hubs. Occupancy follows the geometric law n*M^{-l} even at the top
    layer, and extrapolated to l = round(log_M n) that count is O(1) — the entry point sits there. (At
    finite n the realized top layer holds ~n/M^top_level nodes; the O(1) apex is the log_M(n) level.)"""
    X, _ = nsw_dataset()
    layers, entry, top_level = build_hnsw(X, HNSW_M, HNSW_EFC, seed=0)
    counts = layer_node_counts(layers)
    n = X.shape[0]
    pred_top = n * HNSW_M ** (-top_level)               # geometric prediction at the realized top
    assert 0.3 * pred_top <= counts[top_level] <= 3.5 * pred_top + 1, \
        f"top-level occupancy {counts[top_level]} not ~ n*M^-{top_level} = {pred_top:.1f}"
    l_star = round(np.log(n) / np.log(HNSW_M))           # the log_M n level, where occupancy -> O(1)
    pred_star = n * HNSW_M ** (-l_star)
    assert 0.2 < pred_star < 5.0, f"predicted occupancy at l*={l_star} is {pred_star:.2f}, not O(1)"
    assert entry in layers[top_level], "entry point is not present at the top level"
    print(f"  [ok] geometric apex: counts {counts}; top level {top_level} ~ n*M^-l ({pred_top:.1f}); "
          f"at l*={l_star} occupancy {pred_star:.2f} = O(1); entry={entry}")


def test_max_level_scales_log() -> None:
    """The expected maximum level — the entry-descent depth — grows like log_M(n): it increases with n
    and tracks log_M(n) within O(1). The headline scaling claim, measured from the level draws."""
    M = 8
    rows = scaling_study((200, 350, 500, 750, 1000), M=M, trials=60, seed=0)
    tops = [r["top_level"] for r in rows]
    assert all(tops[i] <= tops[i + 1] + 1e-9 for i in range(len(tops) - 1)), f"top level not increasing: {tops}"
    for r in rows:
        assert abs(r["top_level"] - r["log_M_n"]) < 1.5, \
            f"n={r['n']}: mean top level {r['top_level']:.2f} not within 1.5 of log_M n {r['log_M_n']:.2f}"
    print(f"  [ok] log_M(n) scaling: mean top level {[round(t, 2) for t in tops]} vs "
          f"log_M n {[round(r['log_M_n'], 2) for r in rows]} (n=200..1000)")


def test_search_layer_matches_flat_on_single_layer() -> None:
    """The correctness anchor. Forced to a single layer, the fresh per-layer beam `search_layer` is the
    prerequisite's `greedy_search` exactly — same indices AND same distance-computation count — so the
    hierarchy is the only new thing, not the search."""
    X, queries = nsw_dataset(n=200, nq=15, seed=1)
    adj, entry = build_nsw(X, M=6, ef_construction=12, seed=1)
    flat = {i: adj[i] for i in range(len(adj))}        # the list-of-sets as a per-layer dict
    for qi in range(len(queries)):
        for ef in (1, 4, 16):
            a_ids, a_nd = greedy_search(X, adj, queries[qi], entry, ef, topk=10)
            b_ids, b_nd = search_layer(X, flat, queries[qi], {entry}, ef)
            assert a_ids == b_ids[:10], f"q{qi} ef={ef}: ids differ\n {a_ids}\n {b_ids[:10]}"
            assert a_nd == b_nd, f"q{qi} ef={ef}: ndist {a_nd} != {b_nd}"
    print("  [ok] search_layer == greedy_search on a single flat layer (ids & ndist identical)")


def test_heuristic_diversifies() -> None:
    """The neighbor-selection heuristic spreads links across directions: on a clustered candidate set it
    keeps a DIFFERENT, more spread-out set than the naive M-nearest, preserving long-range links."""
    X2, _, _, _, _, _, _ = toy_hnsw_graph()
    base, cands, M = _heuristic_demo_inputs(X2)
    naive = sorted([c for c in cands if c != base], key=lambda c: _sqd(X2[base], X2[c]))[:M]
    heur = select_neighbors_heuristic(X2, base, cands, M)
    assert set(heur) != set(naive), f"heuristic set {heur} equals naive {naive} — no diversity to show"
    spread = lambda S: float(np.mean([_sqd(X2[a], X2[b]) for a in S for b in S if a < b])) if len(S) > 1 else 0.0
    assert spread(heur) > spread(naive), f"heuristic spread {spread(heur):.2f} not > naive {spread(naive):.2f}"
    print(f"  [ok] heuristic diversifies: kept {heur} (spread {spread(heur):.2f}) vs naive {naive} "
          f"(spread {spread(naive):.2f})")


def test_recall_monotone_in_ef_hnsw() -> None:
    """Hierarchical recall is non-decreasing in the beam width ef: a wider layer-0 search never loses a
    neighbor."""
    X, queries = nsw_dataset()
    layers, entry, top_level = build_hnsw(X, HNSW_M, HNSW_EFC, seed=0)
    rec = np.array([r["recall"] for r in recall_vs_ef_hnsw(X, queries, layers, entry, top_level, EF_GRID)])
    assert np.all(np.diff(rec) >= -1e-9), f"recall not monotone in ef: {np.round(rec, 3)}"
    print(f"  [ok] recall monotone in ef: {np.round(rec, 3).tolist()} (ef={list(EF_GRID)})")


def test_hnsw_beats_flat_nsw_at_equal_cost() -> None:
    """The robust, intra-graph headline: the hierarchy reaches a target recall at no more distance
    computations than the flat NSW it layers — the whole point of the upper layers is a cheaper entry."""
    X, queries = nsw_dataset()
    h = head_to_head(X, queries)
    for r_target in (0.8, 0.9):
        c_hnsw = _cost_at_recall(h["hnsw"], r_target)
        c_nsw = _cost_at_recall(h["nsw"], r_target)
        assert c_hnsw is not None and c_nsw is not None, f"recall {r_target} unreached (hnsw {c_hnsw}, nsw {c_nsw})"
        assert c_hnsw <= c_nsw + 1e-6, f"at recall {r_target}: HNSW cost {c_hnsw:.1f} > flat NSW {c_nsw:.1f}"
    c_hnsw9, c_nsw9 = _cost_at_recall(h["hnsw"], 0.9), _cost_at_recall(h["nsw"], 0.9)
    print(f"  [ok] HNSW <= flat NSW at equal recall: cost@0.9 HNSW {c_hnsw9:.1f} vs NSW {c_nsw9:.1f} "
          f"(of n={h['n']})")


def test_both_reach_exact_at_full_cost() -> None:
    """At full cost both indexes are near-exact: HNSW recall climbs above 0.95 at a wide beam, and IVF
    is EXACT (recall 1.0) when it probes every cell — the approximate methods lose nothing in the limit."""
    X, queries = nsw_dataset()
    h = head_to_head(X, queries)
    hnsw_best = max(r["recall"] for r in h["hnsw"])
    ivf_full = [r for r in h["ivf"] if r["nprobe"] == h["nlist"]]
    assert hnsw_best > 0.95, f"HNSW best recall {hnsw_best:.3f} not near-exact at full beam"
    assert ivf_full and abs(ivf_full[0]["recall"] - 1.0) < 1e-9, \
        f"IVF at nprobe=nlist not exact: {ivf_full}"
    print(f"  [ok] exact in the limit: HNSW best recall {hnsw_best:.3f}; IVF@full-probe recall "
          f"{ivf_full[0]['recall']:.3f}")


def test_hnsw_vs_ivf_on_one_cloud() -> None:
    """The arc-closing comparison, stated honestly. On THIS one synthetic cloud, with a shared ground
    truth, HNSW reaches high recall at sublinear cost; we report — not universalize — how the inverted
    file compares at a matched cost budget."""
    X, queries = nsw_dataset()
    h = head_to_head(X, queries)
    c_hnsw9 = _cost_at_recall(h["hnsw"], 0.9)
    assert c_hnsw9 is not None and c_hnsw9 < 0.5 * h["n"], \
        f"HNSW recall 0.9 needs {c_hnsw9} comps — not sublinear on n={h['n']}"
    r_ivf = _recall_at_cost(h["ivf"], c_hnsw9)         # IVF recall at HNSW's recall-0.9 cost budget
    r_hnsw = _recall_at_cost(h["hnsw"], c_hnsw9)
    print(f"  [ok] one-cloud head-to-head: at cost {c_hnsw9:.1f} (<{h['n']}), recall HNSW {r_hnsw:.3f} "
          f"vs IVF {r_ivf:.3f} — a statement about this cloud, not a universal ranking")


# --------------------------------------------------------------------------- #
# Toy graph + heuristic demo for the laboratory, and the viz constants.
# --------------------------------------------------------------------------- #

def _toy_cloud(seed: int) -> np.ndarray:
    """A small clustered 2-D cloud (five blobs) — clusters give the neighbor heuristic something to
    diversify and make the layer pyramid legible."""
    rng = np.random.default_rng(seed)
    return np.clip(np.vstack([c + 0.9 * rng.standard_normal((7, 2))
                              for c in ([2, 8], [8, 8], [2, 2], [8, 2], [5, 5])]), 0, 10)


def _heuristic_demo_inputs(X2: np.ndarray):
    """A base node whose nearby candidates cluster on one side, so naive-M-nearest and the heuristic
    differ. base = the most central node; candidates = its 10 nearest; M = 4."""
    center = X2.mean(axis=0)
    base = int(cdist(center[None, :], X2, "sqeuclidean")[0].argmin())
    order = np.argsort(cdist(X2[base][None, :], X2, "sqeuclidean")[0])
    cands = [int(c) for c in order if int(c) != base][:10]
    return base, cands, 4


def heuristic_demo(X2: np.ndarray):
    """Naive M-nearest vs heuristic-pruned neighbors for one base node — Panel B. Returns
    (base, candidates, naive_kept, heuristic_kept)."""
    base, cands, M = _heuristic_demo_inputs(X2)
    naive = sorted(cands, key=lambda c: _sqd(X2[base], X2[c]))[:M]
    heur = select_neighbors_heuristic(X2, base, cands, M)
    return base, cands, naive, heur


def toy_hnsw_graph(seed: int = 7):
    """A small 2-D multi-layer HNSW and one query's descent for the laboratory. Returns (X2, layers,
    entry, top_level, query, descent_path, found_idx), where descent_path is the list of (layer, node)
    the beam-1 descent visits, top-first. Reseeds until the graph has >= 3 layers so the pyramid and a
    multi-layer descent are visible."""
    M = 3
    for s in range(seed, seed + 400):
        X2 = _toy_cloud(s)
        layers, entry, top_level = build_hnsw(X2, M=M, ef_construction=8, seed=s)
        if not 2 <= top_level <= 3:        # a legible 3-4 layer pyramid, not a tall single-node tower
            continue
        rng = np.random.default_rng(s + 1000)
        q = rng.uniform(1.5, 8.5, size=2)
        ep, path = entry, [(top_level, entry)]
        for lev in range(top_level, 0, -1):
            ids, _ = search_layer(X2, layers[lev], q, {ep}, ef=1)
            ep = ids[0]
            path.append((lev - 1, ep))
        found = search_hnsw(layers, X2, q, entry, top_level, ef=16, topk=1)[0][0]
        nn = int(cdist(q[None, :], X2, "sqeuclidean")[0].argmin())
        if found == nn:                                  # a clean descent that lands on the true NN
            return X2, layers, entry, top_level, q, path, found
    return X2, layers, entry, top_level, q, path, found  # last attempt (still a valid 7-tuple)


def viz_constants() -> None:
    """Print the toy multi-layer graph + descent (Panel A), the naive-vs-heuristic neighbor sets
    (Panel B), and the recall/cost frontiers + log_M(n) scaling (Panel C) — all baked to the decimal
    in HNSWLaboratory.tsx."""
    X2, layers, entry, top_level, q, path, found = toy_hnsw_graph()
    print("  PANEL A — toy multi-layer HNSW + one query's beam-1 descent:")
    print(f"    TOY_POINTS ({len(X2)} pts in R^2) = {[[round(float(v), 3) for v in p] for p in X2]}")
    levels = [0] * len(X2)
    for lev in range(len(layers)):
        for node in layers[lev]:
            levels[node] = max(levels[node], lev)
    print(f"    TOY_LEVELS = {levels}")
    for lev in range(len(layers)):
        edges = sorted({(int(min(u, v)), int(max(u, v))) for u in layers[lev] for v in layers[lev][u]})
        print(f"    EDGES_L{lev} ({len(edges)}) = {[list(e) for e in edges]}")
    print(f"    ENTRY={int(entry)}  TOP_LEVEL={int(top_level)}  QUERY={[round(float(v), 3) for v in q]}")
    print(f"    DESCENT_PATH (layer,node) = {[[int(a), int(b)] for a, b in path]}  TRUE_NN/FOUND={int(found)}")

    base, cands, naive, heur = heuristic_demo(X2)
    print("  PANEL B — naive M-nearest vs heuristic neighbor selection (M=4):")
    print(f"    DEMO_BASE={int(base)}  DEMO_CANDIDATES={[int(c) for c in cands]}")
    print(f"    NAIVE_KEPT={[int(c) for c in naive]}  HEURISTIC_KEPT={[int(c) for c in heur]}")

    X, queries = nsw_dataset()
    h = head_to_head(X, queries)
    print(f"  PANEL C — recall/cost frontiers on the SAME cloud (n={h['n']}, M={HNSW_M}, nlist={h['nlist']}):")
    print(f"    FRONTIER_HNSW = {[{'ef': r['ef'], 'recall': round(r['recall'], 4), 'cost': round(r['cost'], 1)} for r in h['hnsw']]}")
    print(f"    FRONTIER_NSW  = {[{'ef': r['ef'], 'recall': round(r['recall'], 4), 'cost': round(r['cost'], 1)} for r in h['nsw']]}")
    print(f"    FRONTIER_IVF  = {[{'nprobe': r['nprobe'], 'recall': round(r['recall'], 4), 'cost': round(r['cost'], 1)} for r in h['ivf']]}")
    print(f"    SCALING = {[{'n': r['n'], 'top_level': round(r['top_level'], 3), 'log_M_n': round(r['log_M_n'], 3)} for r in scaling_study((200, 350, 500, 750, 1000), M=HNSW_M, trials=60)]}")


if __name__ == "__main__":
    print("Hierarchical navigable small-world (HNSW) verification harness")
    test_level_tail_law()
    test_layer_geometric_decay()
    test_top_layer_is_O1()
    test_max_level_scales_log()
    test_search_layer_matches_flat_on_single_layer()
    test_heuristic_diversifies()
    test_recall_monotone_in_ef_hnsw()
    test_hnsw_beats_flat_nsw_at_equal_cost()
    test_both_reach_exact_at_full_cost()
    test_hnsw_vs_ivf_on_one_cloud()
    print("Viz constants (mirrored to the decimal in HNSWLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
