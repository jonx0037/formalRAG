"""Filtered and incremental ANN — the reference implementation for the formalRAG
`filtered-incremental-ann` topic.

HNSW gave us a graph that is natively incremental on INSERT and a beam search that walks it. This
topic confronts the two things production indexes face that the static graph ignores: vectors are
DELETED over time, and queries carry a PREDICATE the index must respect. The unifying idea is that
both are the same operation — removing nodes from a navigable graph — and the mathematics has three
movements, with the over-fetch laws as the exact spine and percolation as an honestly-bounded floor:

  MOVEMENT 1 — INSERTION IS FREE; DELETION IS THE HARD HALF (the exact spine). A tombstoned (soft-
    deleted) node stays in the graph as a routing waypoint but is dropped from results. To return k
    LIVE results when a fraction delta of nodes are dead, the number of candidates scanned is
    negative-binomial with mean k/(1-delta) and variance k*delta/(1-delta)^2 — exact under the
    idealization that live/dead is independent of position in the ranked stream. Hard deletion + a
    neighbor-repair heuristic (re-link each orphan to preserve out-degree M) restores the recall that
    tombstone bloat costs; the repair has no optimality proof. (`measure_overfetch_law`,
    `hard_delete_and_repair`, `recall_after_repair`)

  MOVEMENT 2 — CONNECTIVITY UNDER CHURN = PERCOLATION (the honest floor). For a random graph in which
    every node has degree exactly M, retaining each node with probability p leaves a giant connected
    component iff p > p_c = 1/(M-1) (Molloy-Reed / Cohen et al.; the excess-degree branching ratio is
    M-1). We verify this on a near-regular configuration-model graph, then measure the SAME random
    deletion on HNSW's real layer-0 graph, where the threshold only approximates the configuration-
    model law. The load-bearing caveat: connectivity is necessary but NOT sufficient for navigability
    — greedy search returns a local minimum, so recall fails far INSIDE the connected regime and the
    percolation threshold is never the binding constraint. (`random_regular_graph`, `giant_component`,
    `percolation_sweep`, `connectivity_vs_M`)

  MOVEMENT 3 — PREDICATE SEARCH = QUERY-TIME DELETION (the unification + the empirical headline). A
    filter soft-deletes every failing node FOR THIS QUERY: a tombstone is a persistent global
    predicate, a predicate is a per-query tombstone, so post-filtering obeys the SAME law, mean fetch
    k/s for selectivity s, with a sharp binomial recall cliff once s < k/F under a fetch cap F. Three
    strategies trade off — pre-filter (brute-force the passing subset), post-filter (search then
    drop), in-filter (traverse through failing nodes, collect only passing) — and which wins is a
    measured crossover in selectivity, not a universal ranking. The percolation link returns: the
    induced subgraph on a RANDOM passing set is site percolation, so it fragments once s falls below
    p_c, which is why a naive pre-filter on the restricted graph fails at low selectivity and in-
    filter or a denser graph is needed. (`search_layer_filtered`, `prefilter_search`,
    `postfilter_search`, `infilter_search`, `predicate_recall_vs_selectivity`,
    `postfilter_overfetch_law`, `filter_strategy_frontier`, `predicate_subgraph_connectivity`)

Honest caveats (rigorFlag territory, asserted as DIRECTIONS): the two over-fetch laws are exact only
under positional independence — a batch deletion or a correlated predicate removes a spatially, hence
topologically, clustered set and the means become underestimates; the percolation threshold is exact
only for the single-degree configuration model and bounds connectivity, never recall; and every
recall/cost number is measured on one synthetic low-rank cloud with one synthetic predicate, a
statement about this cloud, not a universal ranking of the strategies.

This module imports its prerequisite (`hnsw`) and its ancestors (`navigable_small_world_graphs`) and
never reimplements them. Every pedagogical claim is an `assert` below; `viz_constants()` prints what
`FilteredIncrementalANNLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/filtered-incremental-ann/filtered_incremental_ann.py
"""

from __future__ import annotations

import heapq
import pathlib
import sys

import numpy as np
from scipy.spatial.distance import cdist

# This topic is "HNSW under churn and predicates" — import the direct prerequisite and every ancestor
# it pulls in (the established cross-topic pattern: add each prereq's HYPHENATED dir to the path,
# import its UNDERSCORED module). HNSW supplies the graph we churn/filter; NSW is the canonical owner
# of the squared-distance, brute-force ground truth, and synthetic cloud primitives HNSW re-exports.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("hnsw", "navigable-small-world-graphs", "high-dimensional-geometry"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hnsw import (  # noqa: E402
    build_hnsw,
    search_hnsw,
    search_layer,
    select_neighbors_heuristic,
)
from navigable_small_world_graphs import (  # noqa: E402
    _sqd,
    _true_topk,
    nsw_dataset,
)


# --------------------------------------------------------------------------- #
# Movement 1 — insertion is free; deletion is the hard half (the over-fetch law).
# --------------------------------------------------------------------------- #

def measure_overfetch_law(X, queries, deltas, topk: int = 10, seed: int = 0, trials: int = 30):
    """The tombstone over-fetch law, isolated from graph navigability. For each dead fraction delta,
    mark a random delta of all nodes tombstoned and scan the EXACT distance-ranked stream from nearest
    until topk LIVE results are collected; the scan count is negative-binomial with mean k/(1-delta).
    Returns rows {delta, scanned, predicted=k/(1-delta), var, var_pred}. GUARD: delta < 1 (else the
    stream never yields k live results); topk >= 1."""
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    n = X.shape[0]
    order = np.argsort(cdist(queries, X, "sqeuclidean"), axis=1)   # (nq, n), nearest first
    rng = np.random.default_rng(seed)
    rows = []
    for delta in deltas:
        if not 0.0 <= delta < 1.0:
            raise ValueError(f"delta must be in [0, 1), got {delta}")
        scanned = []
        for _ in range(trials):
            dead = rng.random(n) < delta
            for qi in range(len(queries)):
                cnt = live = 0
                for node in order[qi]:
                    cnt += 1
                    if not dead[node]:
                        live += 1
                        if live >= topk:
                            break
                if live >= topk:
                    scanned.append(cnt)
        arr = np.asarray(scanned, dtype=float)
        rows.append({"delta": float(delta),
                     "scanned": float(arr.mean()),
                     "predicted": float(topk / (1.0 - delta)),
                     "var": float(arr.var()),
                     "var_pred": float(topk * delta / (1.0 - delta) ** 2)})
    return rows


def _copy_layers(layers):
    """A deep-enough copy of the HNSW layer structure (list of {node -> set(neighbors)})."""
    return [{u: set(nb) for u, nb in layer.items()} for layer in layers]


def hard_delete_and_repair(layers, X, deleted, M: int):
    """Remove `deleted` nodes from every layer and REPAIR the holes: each surviving neighbor of a
    deleted node is re-linked, via the prerequisite's `select_neighbors_heuristic`, to the deleted
    node's other survivors (a 2-hop bridge) and pruned back to the degree cap. A heuristic — it
    preserves degree and local connectivity, with no optimality proof. Returns (layers, entry,
    top_level) rebuilt from the survivors. GUARD: empty `deleted` is a no-op copy."""
    dead = {int(d) for d in deleted}
    new_layers = _copy_layers(layers)
    for lev, layer in enumerate(new_layers):
        mmax = 2 * M if lev == 0 else M
        # record, per surviving neighbor, the bridge candidates a deletion orphans it from
        bridge: dict[int, set] = {}
        for d in [k for k in layer if k in dead]:
            survivors = [u for u in layer[d] if u not in dead]
            for u in survivors:
                bridge.setdefault(u, set()).update(w for w in survivors if w != u)
        # excise the deleted nodes and their incident edges
        for d in [k for k in layer if k in dead]:
            for u in layer[d]:
                if u in layer:
                    layer[u].discard(d)
            del layer[d]
        # repair each orphaned survivor
        for u, pool in bridge.items():
            if u not in layer:
                continue
            cand = {c for c in (set(layer[u]) | pool) if c in layer and c != u}
            for w in select_neighbors_heuristic(X, u, list(cand), mmax):
                layer[u].add(w)
                layer[w].add(u)
            if len(layer[u]) > mmax:                       # prune an over-full neighborhood
                keep = set(select_neighbors_heuristic(X, u, list(layer[u]), mmax))
                for r in [r for r in layer[u] if r not in keep]:
                    layer[u].discard(r)
                    layer[r].discard(u)
    while len(new_layers) > 1 and not new_layers[-1]:       # drop emptied top layers
        new_layers.pop()
    top_level = max((lev for lev, layer in enumerate(new_layers) if layer), default=0)
    entry = next(iter(new_layers[top_level])) if new_layers[top_level] else \
        next(iter(new_layers[0]), 0)
    return new_layers, int(entry), int(top_level)


def _topk_live(q, X, live_idx, topk: int):
    """Exact top-k nearest among the live subset (a set of original indices). GUARD: empty subset ->
    empty set; topk capped at the subset size before argpartition."""
    if len(live_idx) == 0:
        return set()
    d = cdist(q[None, :], X[live_idx], "sqeuclidean")[0]
    k = min(topk, len(live_idx))
    sel = np.argpartition(d, k - 1)[:k]
    return set(int(live_idx[i]) for i in sel)


def recall_after_repair(X, queries, M: int = 8, ef_construction: int = 16, delta: float = 0.3,
                        ef: int = 32, topk: int = 10, seed: int = 0):
    """Build a clean HNSW, delete a random fraction delta, and compare recall against the LIVE ground
    truth under (a) tombstoning the full index and (b) hard-delete + repair. Returns
    {rec_tombstoned, rec_repaired, n_deleted, n_remaining, layer0_after}. GUARD: 0 <= delta < 1."""
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    layers, entry, top_level = build_hnsw(X, M=M, ef_construction=ef_construction, seed=seed)
    n = X.shape[0]
    rng = np.random.default_rng(seed + 7)
    dead = rng.random(n) < delta
    deleted = set(np.where(dead)[0].tolist())
    live_idx = np.where(~dead)[0]
    live_set = set(live_idx.tolist())
    truth_live = [_topk_live(queries[qi], X, live_idx, topk) for qi in range(len(queries))]

    hits_t = 0                                             # naive tombstone: top-k from the beam, drop
    for qi in range(len(queries)):                         # dead (no over-fetch) -> loses ~delta of slots
        ids, _ = search_hnsw(layers, X, queries[qi], entry, top_level, ef, topk=topk)
        live_ret = [i for i in ids if i in live_set][:topk]
        hits_t += len(truth_live[qi] & set(live_ret))
    rec_tomb = hits_t / (len(queries) * topk)

    rlayers, rentry, rtop = hard_delete_and_repair(layers, X, deleted, M)
    hits_r = 0
    for qi in range(len(queries)):
        ids, _ = search_hnsw(rlayers, X, queries[qi], rentry, rtop, ef, topk=topk)
        hits_r += len(truth_live[qi] & set(ids))
    rec_rep = hits_r / (len(queries) * topk)

    return {"rec_tombstoned": float(rec_tomb), "rec_repaired": float(rec_rep),
            "n_deleted": int(len(deleted)), "n_remaining": int(n - len(deleted)),
            "layer0_after": int(len(rlayers[0]))}


# --------------------------------------------------------------------------- #
# Movement 2 — connectivity under churn = percolation (the honest floor).
# --------------------------------------------------------------------------- #

def random_regular_graph(n: int, deg: int, seed: int = 0):
    """A near-regular configuration-model graph by stub matching: n*deg half-edges paired at random,
    skipping self- and multi-edges (so realized degrees sit just below deg). The clean substrate on
    which the random-deletion threshold p_c = 1/(deg-1) holds. Returns {node -> set(neighbors)}.
    GUARDS: n >= 2, deg >= 1."""
    if n < 2 or deg < 1:
        raise ValueError(f"need n >= 2 and deg >= 1, got n={n}, deg={deg}")
    rng = np.random.default_rng(seed)
    stubs = np.repeat(np.arange(n), deg)
    rng.shuffle(stubs)
    adj: dict[int, set] = {i: set() for i in range(n)}
    for i in range(0, len(stubs) - 1, 2):
        a, b = int(stubs[i]), int(stubs[i + 1])
        if a != b and b not in adj[a]:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def degree_stats(adj):
    """Mean degree <k>, second moment ratio kappa = <k^2>/<k>, and the configuration-model random-
    deletion threshold p_c = 1/(kappa - 1) (Cohen et al. 2000). GUARD: empty graph -> zeros."""
    degs = np.array([len(nb) for nb in adj.values()], dtype=float)
    if degs.size == 0 or degs.mean() == 0:
        return {"mean_deg": 0.0, "kappa": 0.0, "pc": float("inf")}
    kappa = float((degs ** 2).mean() / degs.mean())
    pc = float("inf") if kappa <= 1.0 else 1.0 / (kappa - 1.0)
    return {"mean_deg": float(degs.mean()), "kappa": kappa, "pc": pc}


def giant_component(adj, keep) -> int:
    """Size of the largest connected component among the `keep` node set, traversing only edges to
    other kept nodes (BFS). GUARD: empty keep -> 0."""
    keep = set(int(k) for k in keep)
    if not keep:
        return 0
    seen: set[int] = set()
    best = 0
    for start in keep:
        if start in seen:
            continue
        comp = 0
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp += 1
            for v in adj.get(u, ()):
                if v in keep and v not in seen:
                    seen.add(v)
                    stack.append(v)
        best = max(best, comp)
    return best


def percolation_sweep(adj, p_grid, trials: int = 20, seed: int = 0):
    """Random node retention: keep each node independently with probability p and record the giant-
    component fraction (largest component / n) averaged over trials. Returns rows {p, giant_frac}."""
    n = len(adj)
    nodes = np.array(list(adj.keys()))
    rng = np.random.default_rng(seed)
    rows = []
    for p in p_grid:
        fracs = []
        for _ in range(trials):
            keep = nodes[rng.random(n) < p]
            fracs.append(giant_component(adj, keep) / n)
        rows.append({"p": float(p), "giant_frac": float(np.mean(fracs))})
    return rows


def connectivity_vs_M(M_grid, X, retain_p: float = 0.2, trials: int = 12, seed: int = 0):
    """Build HNSW at each M and fragment its layer-0 graph by random retention at a FIXED p; a denser
    graph (larger M) keeps a larger giant component. Returns rows {M, mean_deg, giant_frac}."""
    rows = []
    for M in M_grid:
        layers, _, _ = build_hnsw(X, M=M, ef_construction=2 * M, seed=seed)
        adj = layers[0]
        stats = degree_stats(adj)
        gf = percolation_sweep(adj, [retain_p], trials=trials, seed=seed)[0]["giant_frac"]
        rows.append({"M": int(M), "mean_deg": round(stats["mean_deg"], 2), "giant_frac": float(gf)})
    return rows


# --------------------------------------------------------------------------- #
# Movement 3 — predicate search = query-time deletion (the unification + headline).
# --------------------------------------------------------------------------- #

def assign_modalities(X, n_modalities: int = 5, seed: int = 0):
    """Assign each vector to the nearest of n_modalities random anchors, so each modality is a SPATIAL
    cluster — the production-realistic case (text/news/pdf/chart/audio occupy different regions of a
    shared embedding space) where a modality filter is a CORRELATED removal. Returns labels (n,)."""
    rng = np.random.default_rng(seed)
    anchors = X[rng.choice(X.shape[0], n_modalities, replace=False)]
    return cdist(X, anchors, "sqeuclidean").argmin(axis=1).astype(int)


def search_layer_filtered(X, layer_adj, q, entry_points, ef: int, live):
    """The in-filter twin of the prerequisite's `search_layer`: identical beam, but a node enters the
    RESULT heap only if it is live[node]; failing nodes are still TRAVERSED (their neighbors expanded)
    so the predicate subgraph stays reachable. When every node is live this reduces to `search_layer`
    exactly — same ids AND ndist (the correctness anchor). Returns (ordered live ids, ndist). GUARD:
    ef >= 1; non-empty entry_points."""
    if ef < 1:
        raise ValueError(f"ef must be >= 1, got {ef}")
    visited: set[int] = set()
    candidates: list[tuple[float, int]] = []
    results: list[tuple[float, int]] = []
    ndist = 0
    for e in entry_points:
        e = int(e)
        if e in visited:
            continue
        visited.add(e)
        de = _sqd(q, X[e])
        ndist += 1
        heapq.heappush(candidates, (de, e))
        if live[e]:
            heapq.heappush(results, (-de, e))
    while len(results) > ef:
        heapq.heappop(results)
    if not candidates:
        raise ValueError("entry_points must be non-empty")
    while candidates:
        d, c = heapq.heappop(candidates)
        worst = -results[0][0] if results else float("inf")
        if d > worst and len(results) >= ef:
            break
        for nb in layer_adj.get(c, ()):
            if nb in visited:
                continue
            visited.add(nb)
            dn = _sqd(q, X[nb])
            ndist += 1
            worst = -results[0][0] if results else float("inf")
            if dn < worst or len(results) < ef:
                heapq.heappush(candidates, (dn, nb))
                if live[nb]:
                    heapq.heappush(results, (-dn, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
    ordered = [i for _, i in sorted((-nd, i) for nd, i in results)]
    return ordered, ndist


def prefilter_search(X, q, live_idx, topk: int = 10):
    """Brute-force the passing subset only: exact, costs one distance per live vector. Returns
    (topk live ids nearest-first, ndist = n_live). GUARD: empty subset -> ([], 0)."""
    if len(live_idx) == 0:
        return [], 0
    d = cdist(q[None, :], X[live_idx], "sqeuclidean")[0]
    k = min(topk, len(live_idx))
    sel = np.argsort(d)[:k]
    return [int(live_idx[i]) for i in sel], int(len(live_idx))


def postfilter_search(layers, X, q, entry, top_level, ef: int, live, topk: int = 10):
    """Search the full index at beam width ef, then drop failing nodes from the result list — the
    over-fetch governed by ef. Returns (topk live ids, ndist). GUARD: empty index -> ([], 0)."""
    if not layers:
        return [], 0
    ids, ndist = search_hnsw(layers, X, q, entry, top_level, ef, topk=ef)
    live_ret = [i for i in ids if live[i]][:topk]
    return live_ret, ndist


def infilter_search(layers, X, q, entry, top_level, ef: int, live, topk: int = 10):
    """Predicate-aware traversal: beam-1 descent through the upper layers (predicate-agnostic routing),
    then `search_layer_filtered` at layer 0. Returns (topk live ids, ndist). GUARD: empty index ->
    ([], 0)."""
    if not layers:
        return [], 0
    ep = int(entry)
    ndist = 0
    for lev in range(top_level, 0, -1):
        ids, nd = search_layer(X, layers[lev], q, {ep}, ef=1)
        ndist += nd
        ep = ids[0]
    ids, nd = search_layer_filtered(X, layers[0], q, {ep}, ef, live)
    ndist += nd
    return ids[:topk], ndist


def _live_truth(queries, X, live_idx, topk: int):
    """Per-query exact top-k among the live subset — the predicate-restricted ground truth."""
    return [_topk_live(queries[qi], X, live_idx, topk) for qi in range(len(queries))]


def predicate_recall_vs_selectivity(layers, X, queries, masks, entry, top_level,
                                    ef: int = 64, topk: int = 10):
    """For each predicate mask (a boolean array over nodes, with selectivity s = mask.mean()), the
    mean recall@topk and mean distance computations of pre/post/in-filter against the live ground
    truth. Returns {"pre": [...], "post": [...], "in": [...]} of rows {s, recall, ndist}. GUARD:
    recall denominator uses k = min(topk, n_live); empty subset rows report recall 0."""
    out = {"pre": [], "post": [], "in": []}
    for mask in masks:
        live = np.asarray(mask, dtype=bool)
        live_idx = np.where(live)[0]
        s = float(live.mean())
        nlive = int(live_idx.size)
        k = min(topk, nlive)
        truth = _live_truth(queries, X, live_idx, topk)
        for name, fn in (("pre", prefilter_search), ("post", postfilter_search),
                         ("in", infilter_search)):
            hits = work = 0
            for qi in range(len(queries)):
                if name == "pre":
                    ids, nd = prefilter_search(X, queries[qi], live_idx, topk)
                else:
                    ids, nd = fn(layers, X, queries[qi], entry, top_level, ef, live, topk)
                hits += len(truth[qi] & set(ids))
                work += nd
            denom = len(queries) * k if k > 0 else 1
            out[name].append({"s": round(s, 4), "recall": hits / denom, "ndist": work / len(queries)})
    return out


def postfilter_overfetch_law(X, queries, s_grid, topk: int = 10, seed: int = 0, trials: int = 30):
    """The post-filter over-fetch law, isolated on the exact ranked stream (the per-query analogue of
    the tombstone law): for selectivity s, scan the distance-ranked stream until topk PASSING results
    are collected; mean scan = k/s. Returns rows {s, scanned, predicted=k/s}. GUARD: s > 0."""
    n = X.shape[0]
    order = np.argsort(cdist(queries, X, "sqeuclidean"), axis=1)
    rng = np.random.default_rng(seed)
    rows = []
    for s in s_grid:
        if not 0.0 < s <= 1.0:
            raise ValueError(f"s must be in (0, 1], got {s}")
        scanned = []
        for _ in range(trials):
            passes = rng.random(n) < s
            for qi in range(len(queries)):
                cnt = got = 0
                for node in order[qi]:
                    cnt += 1
                    if passes[node]:
                        got += 1
                        if got >= topk:
                            break
                if got >= topk:
                    scanned.append(cnt)
        arr = np.asarray(scanned, dtype=float)
        rows.append({"s": float(s), "scanned": float(arr.mean()), "predicted": float(topk / s)})
    return rows


def filter_strategy_frontier(layers, X, queries, entry, top_level, s_grid, ef: int = 64,
                             topk: int = 10, seed: int = 0):
    """The headline crossover. At each selectivity s (random predicate), measure recall@topk and
    distance computations per query for pre/post/in-filter on the SAME index and ground truth.
    Returns the predicate_recall_vs_selectivity dict over random masks at the requested s grid."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    masks = [rng.random(n) < s for s in s_grid]
    return predicate_recall_vs_selectivity(layers, X, queries, masks, entry, top_level, ef, topk)


def predicate_subgraph_connectivity(adj, labels, trials: int = 12, seed: int = 0):
    """Does the INDUCED subgraph on the passing set stay connected? Add modalities in ascending size
    (correlated / contiguous passing set) and, at each matched selectivity s, compare the giant
    fraction (largest component / n_live) of the correlated set against a RANDOM passing set of the
    same s (site percolation). Returns rows {s, giant_corr, giant_random, n_mods}. The random series
    is the clean percolation link; the correlated direction is measured, not assumed."""
    n = len(adj)
    rng = np.random.default_rng(seed)
    mods = sorted(set(int(m) for m in labels))
    sizes = {m: int(np.sum(labels == m)) for m in mods}
    order = sorted(mods, key=lambda m: sizes[m])
    cum: set[int] = set()
    rows = []
    for j, m in enumerate(order, 1):
        cum |= set(np.where(labels == m)[0].tolist())
        s = len(cum) / n
        gf_corr = giant_component(adj, cum) / max(len(cum), 1)
        randfracs = []
        for _ in range(trials):
            keep = set(np.where(rng.random(n) < s)[0].tolist())
            randfracs.append(giant_component(adj, keep) / max(len(keep), 1))
        rows.append({"s": round(float(s), 4), "giant_corr": round(float(gf_corr), 4),
                     "giant_random": round(float(np.mean(randfracs)), 4), "n_mods": int(j)})
    return rows


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

M_DEF, EFC_DEF, TOPK = 8, 16, 10
DELTA_GRID = (0.1, 0.25, 0.5, 0.7)
S_GRID = (0.05, 0.1, 0.2, 0.4, 0.8)
REG_N, REG_DEG = 4000, 8


def test_search_layer_filtered_matches_unfiltered_when_all_live() -> None:
    """The correctness anchor (the "fresh twin, cross-check byte-for-byte" rule). With every node
    live, the in-filter beam `search_layer_filtered` is the prerequisite's `search_layer` exactly —
    same ids AND same distance-computation count — so the predicate is the only new thing."""
    X, queries = nsw_dataset(n=200, nq=15, seed=1)
    layers, _, _ = build_hnsw(X, M=6, ef_construction=12, seed=1)
    live = np.ones(X.shape[0], dtype=bool)
    layer0 = layers[0]
    entry = next(iter(layer0))
    for qi in range(len(queries)):
        for ef in (1, 4, 16):
            a_ids, a_nd = search_layer(X, layer0, queries[qi], {entry}, ef)
            b_ids, b_nd = search_layer_filtered(X, layer0, queries[qi], {entry}, ef, live)
            assert a_ids == b_ids, f"q{qi} ef={ef}: ids differ\n {a_ids}\n {b_ids}"
            assert a_nd == b_nd, f"q{qi} ef={ef}: ndist {a_nd} != {b_nd}"
    print("  [ok] search_layer_filtered == search_layer when all live (ids & ndist identical)")


def test_overfetch_law_tombstone() -> None:
    """The exact spine, half one: to return k live results when a fraction delta is tombstoned, the
    mean scan is k/(1-delta). Empirical means match within tolerance and rise monotonically in delta;
    the empirical variance tracks k*delta/(1-delta)^2."""
    X, queries = nsw_dataset()
    rows = measure_overfetch_law(X, queries, DELTA_GRID, topk=TOPK, trials=40)
    for r in rows:
        assert abs(r["scanned"] - r["predicted"]) < 0.08 * r["predicted"] + 0.5, \
            f"delta={r['delta']}: scanned {r['scanned']:.2f} vs k/(1-delta) {r['predicted']:.2f}"
    scans = [r["scanned"] for r in rows]
    assert all(scans[i] < scans[i + 1] for i in range(len(scans) - 1)), f"scan not rising in delta: {scans}"
    print(f"  [ok] tombstone over-fetch law: scanned {[round(r['scanned'], 2) for r in rows]} ~ "
          f"k/(1-delta) {[round(r['predicted'], 2) for r in rows]} (delta={list(DELTA_GRID)})")


def test_hard_delete_repair_restores_recall() -> None:
    """Hard deletion + neighbor repair recovers the recall that tombstone bloat costs: repaired recall
    is at least the tombstoned recall, and the repaired layer-0 has exactly the survivors."""
    X, queries = nsw_dataset()
    r = recall_after_repair(X, queries, M=M_DEF, ef_construction=EFC_DEF, delta=0.3, ef=32, seed=0)
    assert r["layer0_after"] == r["n_remaining"], \
        f"layer-0 holds {r['layer0_after']}, expected {r['n_remaining']} survivors"
    assert r["rec_repaired"] >= r["rec_tombstoned"] - 1e-9, \
        f"repair {r['rec_repaired']:.3f} below tombstoned {r['rec_tombstoned']:.3f}"
    print(f"  [ok] hard-delete + repair: recall tombstoned {r['rec_tombstoned']:.3f} -> repaired "
          f"{r['rec_repaired']:.3f}; layer-0 {r['layer0_after']} = {r['n_remaining']} survivors")


def test_regular_graph_percolation_threshold() -> None:
    """Movement 2's theorem, verified where it is exact. On a near-regular configuration-model graph
    the random-deletion giant component survives well above p_c = 1/(deg-1) and collapses well below
    it — the steep drop straddles the prediction."""
    adj = random_regular_graph(REG_N, REG_DEG, seed=0)
    stats = degree_stats(adj)
    pc = stats["pc"]
    above = percolation_sweep(adj, [min(3.0 * pc, 0.9)], trials=8, seed=1)[0]["giant_frac"]
    below = percolation_sweep(adj, [0.4 * pc], trials=8, seed=1)[0]["giant_frac"]
    assert above > 0.4, f"giant fraction {above:.3f} not large at 3*p_c (p_c={pc:.3f})"
    assert below < 0.1, f"giant fraction {below:.3f} not small at 0.4*p_c (p_c={pc:.3f})"
    print(f"  [ok] regular-graph percolation: kappa={stats['kappa']:.2f}, p_c=1/(kappa-1)={pc:.3f}; "
          f"giant frac {below:.3f} (below) -> {above:.3f} (above)")


def test_giant_component_monotone_in_M() -> None:
    """A denser graph survives more deletion: at a fixed low retention the layer-0 giant fraction is
    non-decreasing in M — the lever an operator pulls to buy churn robustness."""
    X, _ = nsw_dataset()
    rows = connectivity_vs_M((4, 8, 16, 32), X, retain_p=0.15, trials=10, seed=0)
    gf = [r["giant_frac"] for r in rows]
    assert all(gf[i] <= gf[i + 1] + 1e-9 for i in range(len(gf) - 1)), f"giant frac not rising in M: {gf}"
    print(f"  [ok] connectivity rises with M: giant frac {[round(g, 3) for g in gf]} "
          f"at M={[r['M'] for r in rows]} (retain p=0.15)")


def test_postfilter_overfetch_law() -> None:
    """The exact spine, half two — the SAME law as the tombstone law with s <-> (1-delta): to return
    k results passing a predicate of selectivity s, the mean fetch is k/s. Means match and rise as s
    falls."""
    X, queries = nsw_dataset()
    rows = postfilter_overfetch_law(X, queries, S_GRID, topk=TOPK, trials=40)
    for r in rows:
        assert abs(r["scanned"] - r["predicted"]) < 0.1 * r["predicted"] + 0.5, \
            f"s={r['s']}: scanned {r['scanned']:.2f} vs k/s {r['predicted']:.2f}"
    print(f"  [ok] post-filter over-fetch law: scanned {[round(r['scanned'], 1) for r in rows]} ~ "
          f"k/s {[round(r['predicted'], 1) for r in rows]} (s={list(S_GRID)})")


def test_postfilter_recall_cliff() -> None:
    """The binomial recall cliff: at a FIXED beam width, post-filter recall collapses as selectivity
    falls (a fixed over-fetch budget cannot fill k passing results once s is small), while pre-filter
    stays exact. The headline's motivation."""
    X, queries = nsw_dataset()
    layers, entry, top_level = build_hnsw(X, M=M_DEF, ef_construction=EFC_DEF, seed=0)
    fr = filter_strategy_frontier(layers, X, queries, entry, top_level, S_GRID, ef=64, topk=TOPK, seed=0)
    post = [r["recall"] for r in fr["post"]]               # ordered by S_GRID ascending
    pre = [r["recall"] for r in fr["pre"]]
    assert post[0] < post[-1] - 0.1, f"no post-filter cliff: recall {post[0]:.3f} (low s) vs {post[-1]:.3f} (high s)"
    assert min(pre) > 0.99, f"pre-filter not exact: {pre}"
    print(f"  [ok] post-filter recall cliff: post recall {[round(p, 3) for p in post]} vs pre "
          f"{[round(p, 3) for p in pre]} (s={list(S_GRID)})")


def test_filter_strategy_crossover() -> None:
    """The empirical headline, measured on this cloud (run first; the inequalities are pinned to the
    observed run). Pre-filter is exact and cheapest at the LOWEST selectivity (a tiny subset to scan);
    post-filter is cheapest at the HIGHEST selectivity (tiny over-fetch); in-filter recovers the recall
    post-filter loses at low s. A crossover, not a universal ranking."""
    X, queries = nsw_dataset()
    layers, entry, top_level = build_hnsw(X, M=M_DEF, ef_construction=EFC_DEF, seed=0)
    fr = filter_strategy_frontier(layers, X, queries, entry, top_level, S_GRID, ef=64, topk=TOPK, seed=0)
    lo, hi = 0, len(S_GRID) - 1
    # at the lowest selectivity pre-filter is exact and cheaper than post-filter's wasted traversal
    assert fr["pre"][lo]["ndist"] < fr["post"][lo]["ndist"], \
        f"pre not cheapest at low s: pre {fr['pre'][lo]['ndist']:.1f} vs post {fr['post'][lo]['ndist']:.1f}"
    # at the lowest selectivity in-filter keeps more recall than post-filter (which falls off the cliff)
    assert fr["in"][lo]["recall"] >= fr["post"][lo]["recall"] - 1e-9, \
        f"in-filter not >= post at low s: in {fr['in'][lo]['recall']:.3f} vs post {fr['post'][lo]['recall']:.3f}"
    # at the highest selectivity post-filter is cheapest (no wasted over-fetch)
    assert fr["post"][hi]["ndist"] <= fr["pre"][hi]["ndist"], \
        f"post not cheapest at high s: post {fr['post'][hi]['ndist']:.1f} vs pre {fr['pre'][hi]['ndist']:.1f}"
    print(f"  [ok] strategy crossover (this cloud): @s={S_GRID[lo]} pre exact@{fr['pre'][lo]['ndist']:.0f} "
          f"comps, in recall {fr['in'][lo]['recall']:.3f} >= post {fr['post'][lo]['recall']:.3f}; "
          f"@s={S_GRID[hi]} post {fr['post'][hi]['ndist']:.0f} <= pre {fr['pre'][hi]['ndist']:.0f} comps")


def test_predicate_subgraph_percolation() -> None:
    """The percolation link for predicates, on the real layer-0 graph. The induced subgraph on a
    RANDOM passing set fragments as selectivity falls (its giant fraction at the lowest s is well
    below one) — which is why a naive pre-filter on the restricted graph fails at low selectivity and
    in-filter or a denser graph is required."""
    X, _ = nsw_dataset()
    layers, _, _ = build_hnsw(X, M=M_DEF, ef_construction=EFC_DEF, seed=0)
    labels = assign_modalities(X, n_modalities=5, seed=0)
    rows = predicate_subgraph_connectivity(layers[0], labels, trials=12, seed=0)
    rand = [r["giant_random"] for r in rows]               # ascending s
    assert rand[0] < 0.9, f"random induced subgraph not fragmented at low s: giant frac {rand[0]:.3f}"
    assert rand[-1] > rand[0], f"random giant fraction not rising with s: {rand}"
    print(f"  [ok] predicate subgraph percolation: random giant frac {[round(r, 3) for r in rand]} "
          f"vs correlated {[round(r['giant_corr'], 3) for r in rows]} (s={[r['s'] for r in rows]})")


# --------------------------------------------------------------------------- #
# Toy graph for the laboratory and the viz constants.
# --------------------------------------------------------------------------- #

def toy_filtered_graph(seed: int = 3):
    """A small 2-D, five-blob cloud (each blob a modality) and its HNSW for the laboratory. Returns
    (X2, layers, levels, entry, top_level, labels, deleted, query, found_live), with `deleted` a ~30%
    tombstone set and `found_live` the true live nearest neighbor. Reseeds for a legible 2-3 layer
    pyramid."""
    M = 3
    pts_per = 7
    for s in range(seed, seed + 300):
        rng = np.random.default_rng(s)
        X2 = np.clip(np.vstack([c + 0.8 * rng.standard_normal((pts_per, 2))
                                for c in ([2, 8], [8, 8], [2, 2], [8, 2], [5, 5])]), 0, 10)
        layers, entry, top_level = build_hnsw(X2, M=M, ef_construction=8, seed=s)
        if not 2 <= top_level <= 3:
            continue
        labels = np.repeat(np.arange(5), pts_per).astype(int)        # blob-major order => modality
        rng2 = np.random.default_rng(s + 50)
        deleted = sorted(np.where(rng2.random(len(X2)) < 0.3)[0].tolist())
        q = rng2.uniform(1.5, 8.5, size=2)
        live_idx = np.array([i for i in range(len(X2)) if i not in set(deleted)])
        found = sorted(_topk_live(q, X2, live_idx, 1))
        levels = [0] * len(X2)
        for lev in range(len(layers)):
            for node in layers[lev]:
                levels[node] = max(levels[node], lev)
        return X2, layers, levels, int(entry), int(top_level), labels, deleted, q, found
    return X2, layers, levels, int(entry), int(top_level), labels, deleted, q, found


def viz_constants() -> None:
    """Print the toy graph + the over-fetch laws (Panel A), the percolation sweeps (Panel B), and the
    filter-strategy frontier + predicate-subgraph connectivity (Panel C) — all baked to the decimal in
    FilteredIncrementalANNLaboratory.tsx."""
    X2, layers, levels, entry, top_level, labels, deleted, q, found = toy_filtered_graph()
    edges0 = sorted({(int(min(u, v)), int(max(u, v))) for u in layers[0] for v in layers[0][u]})
    print("  PANEL A — toy graph + tombstone over-fetch:")
    print(f"    TOY_POINTS ({len(X2)} pts) = {[[round(float(v), 3) for v in p] for p in X2]}")
    print(f"    TOY_LEVELS = {levels}")
    print(f"    TOY_MODALITIES = {[int(m) for m in labels]}")
    print(f"    EDGES_L0 ({len(edges0)}) = {[list(e) for e in edges0]}")
    print(f"    ENTRY={entry}  QUERY={[round(float(v), 3) for v in q]}  DELETED={deleted}  FOUND_LIVE={found}")

    X, queries = nsw_dataset()
    of = measure_overfetch_law(X, queries, DELTA_GRID, topk=TOPK, trials=40)
    print(f"    OVERFETCH_LAW = {[{'delta': r['delta'], 'scanned': round(r['scanned'], 2), 'predicted': round(r['predicted'], 2)} for r in of]}")

    print("  PANEL B — percolation (random deletion):")
    reg = random_regular_graph(REG_N, REG_DEG, seed=0)
    reg_stats = degree_stats(reg)
    p_grid = [round(0.04 * i, 3) for i in range(0, 13)]
    reg_sweep = percolation_sweep(reg, p_grid, trials=10, seed=1)
    print(f"    REG_PC = {round(reg_stats['pc'], 4)}  REG_KAPPA = {round(reg_stats['kappa'], 3)}  REG_DEG = {REG_DEG}")
    print(f"    PERCOLATION_REGULAR = {[{'p': r['p'], 'giant_frac': round(r['giant_frac'], 4)} for r in reg_sweep]}")
    layers_full, _, _ = build_hnsw(X, M=M_DEF, ef_construction=EFC_DEF, seed=0)
    hnsw_stats = degree_stats(layers_full[0])
    hnsw_sweep = percolation_sweep(layers_full[0], p_grid, trials=20, seed=1)
    print(f"    HNSW_PC = {round(hnsw_stats['pc'], 4)}  HNSW_KAPPA = {round(hnsw_stats['kappa'], 3)}  HNSW_MEAN_DEG = {round(hnsw_stats['mean_deg'], 2)}")
    print(f"    PERCOLATION_HNSW = {[{'p': r['p'], 'giant_frac': round(r['giant_frac'], 4)} for r in hnsw_sweep]}")
    cvm = connectivity_vs_M((4, 8, 16, 32), X, retain_p=0.15, trials=10, seed=0)
    print(f"    CONNECTIVITY_VS_M = {[{'M': r['M'], 'mean_deg': r['mean_deg'], 'giant_frac': round(r['giant_frac'], 4)} for r in cvm]}")

    print("  PANEL C — predicate search (random predicate, on the n=500 index):")
    layers2, entry, top_level = build_hnsw(X, M=M_DEF, ef_construction=EFC_DEF, seed=0)
    fr = filter_strategy_frontier(layers2, X, queries, entry, top_level, S_GRID, ef=64, topk=TOPK, seed=0)
    for name in ("pre", "post", "in"):
        print(f"    FRONTIER_{name.upper()} = {[{'s': r['s'], 'recall': round(r['recall'], 4), 'ndist': round(r['ndist'], 1)} for r in fr[name]]}")
    pfl = postfilter_overfetch_law(X, queries, S_GRID, topk=TOPK, trials=40)
    print(f"    POSTFILTER_LAW = {[{'s': r['s'], 'scanned': round(r['scanned'], 1), 'predicted': round(r['predicted'], 1)} for r in pfl]}")
    labels = assign_modalities(X, n_modalities=5, seed=0)
    sub = predicate_subgraph_connectivity(layers2[0], labels, trials=12, seed=0)
    print(f"    SUBGRAPH_CONNECTIVITY = {sub}")


if __name__ == "__main__":
    print("Filtered and incremental ANN verification harness")
    test_filter_strategy_crossover()          # the headline runs first (pin to the observed run)
    test_search_layer_filtered_matches_unfiltered_when_all_live()
    test_overfetch_law_tombstone()
    test_hard_delete_repair_restores_recall()
    test_regular_graph_percolation_threshold()
    test_giant_component_monotone_in_M()
    test_postfilter_overfetch_law()
    test_postfilter_recall_cliff()
    test_predicate_subgraph_percolation()
    print("Viz constants (mirrored to the decimal in FilteredIncrementalANNLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
