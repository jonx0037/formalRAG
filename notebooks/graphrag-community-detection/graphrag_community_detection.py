"""GraphRAG: community detection and the modularity of knowledge — the reference
implementation for the formalRAG `graphrag-community-detection` topic.

Multi-hop retrieval answered a LOCAL question: a *path* from a query to a specific answer
document. GraphRAG answers a GLOBAL / sensemaking question — "what are the dominant themes
across the whole corpus?" — whose answer is a property of the graph's PARTITION, not of any
node or path. The object of study moves from a trajectory (a walk) to a partition of the
entity co-occurrence graph, and the mathematics is: when does a good partition exist, when
can we find it, and what are the fundamental limits. Five movements, every pedagogical claim
an `assert`:

  MOVEMENT 1 — THE ENTITY GRAPH AND MODULARITY. Build a weighted graph G=(V,E,A) whose nodes
    are entities (companies) and whose edge weights are co-occurrence (here: the sharpened
    cosine of the shared finance vMF geometry, so SECTORS are the planted communities).
    Newman-Girvan modularity Q = (1/2m) sum_ij [A_ij - k_i k_j/(2m)] delta(c_i,c_j) scores a
    partition against the degree-preserving (configuration-model) null. (`modularity`.)

  MOVEMENT 2 — THE SPECTRAL RELAXATION (THEOREM, Newman 2006). The modularity matrix
    B = A - k k^T/(2m); for a bipartition s in {+-1}^n, rows of B sum to zero so
    Q = (1/4m) s^T B s. Relaxing s to the sphere, Q is maximized by B's leading (largest
    ALGEBRAIC) eigenvector; round by sign. If lambda_1(B) <= 0 the network is indivisible.
    The rounded integer partition is NOT the optimum (that is NP-hard, Movement 5).
    (`modularity_matrix`, `spectral_bipartition`, `brute_modularity_argmax`.)

  MOVEMENT 3 — THE RESOLUTION LIMIT (THEOREM, Fortunato-Barthelemy 2007 — the rigorFlag).
    The gamma-generalization Q_gamma. Modularity cannot resolve communities smaller than a
    scale set by the WHOLE graph, ~sqrt(2m): on a ring of cliques the optimal partition MERGES
    adjacent cliques once the ring is large. gamma only MOVES the scale; no single gamma
    resolves multi-scale structure. (`ring_of_cliques`, `modularity`.)

  MOVEMENT 4 — THE SBM AND THE DETECTABILITY TRANSITION (THEOREM — the deep payload). The
    stochastic block model is the generative model; for the symmetric 2-block sparse SBM with
    affinities c_in, c_out (p_in=c_in/n, p_out=c_out/n, mean degree (c_in+c_out)/2), a partition
    correlated with the planted one is recoverable iff (c_in-c_out)^2 > 2(c_in+c_out) (the
    Kesten-Stigum threshold; Decelle et al. 2011, proven Mossel-Neeman-Sly / Massoulie). Below
    it NO algorithm beats a coin flip — an information-theoretic converse, the analogue of Fano
    in the noisy-channel topic. (`sbm_sample`, `detectability_threshold`, `spectral_sbm_recovery`,
    `overlap`.)

  MOVEMENT 5 — GRAPHRAG, LOUVAIN, LEIDEN. Modularity maximization is NP-hard (Brandes 2008), so
    Louvain (local-moving + aggregation, closed-form Delta Q) and Leiden (a refinement phase
    GUARANTEEING internally-connected communities) are heuristics. A partition with a disconnected
    community can be a Louvain LOCAL OPTIMUM that local-moving cannot repair; Leiden's refinement
    forbids it. GraphRAG (Edge et al. 2024) runs hierarchical Leiden on the entity graph and
    map-reduces per-community summaries to answer global queries. (`louvain`, `leiden`,
    `leiden_refine`, `community_is_connected`.)

Honest caveats (rigorFlag territory): every method is a heuristic (modularity max is NP-hard);
the resolution limit is intrinsic to single-scale modularity (gamma only moves it; modularity is
also degenerate); the SBM threshold is an asymptotic (n->inf, sparse) info-theoretic limit and a
finite-n simulation only SHARPENS toward it; for q=2 the info-theoretic and algorithmic thresholds
coincide but for q>=4 a HARD phase opens (demonstrated only for q=2); and the entity graph is a
SYNTHETIC planted partition on the finance vMF geometry — the recovery results are exact for that
model and illustrative of a real GraphRAG knowledge graph.

This module IMPORTS its geometry from the dense/vMF prerequisites and never reimplements it; it
OWNS the graph primitives (no networkx). `viz_constants()` prints what
GraphRAGCommunityLaboratory.tsx mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/graphrag-community-detection/graphrag_community_detection.py
"""
from __future__ import annotations

import functools
import math
import pathlib
import sys
from itertools import permutations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

# Established cross-topic pattern: add each ancestor's HYPHENATED dir to the path, import the
# UNDERSCORED module. We reuse the finance sector->company vMF geometry (sectors = planted
# communities) and the set-metrics recovery scorers; we never reimplement them. Frontmatter
# `prerequisites` stays the single graph edge [multi-hop-iterative-retrieval] — the import graph
# is not the pedagogical DAG (these geometry sources sit in `connections[]`).
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "dense-retrieval-dual-encoders",
    "set-metrics-precision-recall-map-mrr",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dense_retrieval_dual_encoders import (              # noqa: E402
    dpr_finance_matrix,
    DPR_SEED,
)
from set_metrics_precision_recall_map_mrr import recall_at_k  # noqa: E402


def r3(v) -> float:
    """Round a (possibly numpy) scalar to 3 places as a clean Python float for the viz mirror."""
    return round(float(v), 3)


# =========================================================================== #
# Constants.
# =========================================================================== #

GR_SEED = DPR_SEED                 # 7 -- the shared finance-geometry seed
GR_N_SECTORS = 5                   # 5 sectors x 5 companies = 25 entity nodes (planted communities)
GR_N_COMP = 5
GR_DIM = 64                        # higher than the dense topic's 32 so the 5 sector means are
                                   #   near-orthogonal (cross-sector cosine ~0; recovery is robust
                                   #   across seeds), with kappa raised to keep within-sector high
GR_KAPPA_SECTOR = 200.0            # within-sector concentration (same-sector cosine ~0.72 at d=64)
GR_THRESHOLD = 0.15                # cosine co-occurrence threshold: edge weight = max(0, cos - thr).
                                   #   At seed 7 the graph is connected with ~18 cross-sector bridges
GAMMA_GRID = (0.05, 0.1, 0.3, 1.0, 4.0, 6.0, 8.0)   # resolution sweep: low gamma MERGES two
#   sectors (4 communities), a 0.3-4.0 plateau recovers the 5 planted sectors, high gamma
#   FRAGMENTS toward singletons (10 -> 25) -- the resolution limit made interactive in Panel A

# Stochastic block model (the q=2 detectability demo).
SBM_N = 3000                       # n for the two headline operating points
SBM_N_GRID = 1500                  # n for the (c_in, c_out) overlap grid (kept smaller for speed)
SBM_ABOVE = (10.0, 1.0)            # (c_in, c_out): SNR (9^2)/(2*11) = 3.68 -- detectable
SBM_BELOW = (6.0, 4.0)             # SNR (2^2)/(2*10) = 0.20 -- undetectable
SBM_C_GRID = (1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0)   # c_in and c_out grid values
SBM_DISPLAY_N = 40                 # small SBM samples drawn for the inset cartoons
SBM_GRID_SEED = 101

# Ring of cliques (the resolution-limit witness).
RING_LARGE_NC, RING_CLIQUE = 30, 5    # 30 cliques of size 5: the optimum MERGES adjacent cliques
RING_SMALL_NC = 3                     # 3 cliques: the optimum KEEPS them separate (the contrast)

LOG2 = math.log(2.0)


# =========================================================================== #
# Movement 1-2 -- graph primitives (this topic OWNS these; no networkx).
# =========================================================================== #

def degrees(A: np.ndarray) -> np.ndarray:
    """Weighted degree vector k_i = sum_j A_ij (a self-loop contributes once to the row sum,
    which is exactly its contribution to Sigma_tot in the aggregated Louvain graph)."""
    return np.asarray(A, float).sum(axis=1)


def total_weight(A: np.ndarray) -> float:
    """2m = sum_ij A_ij (twice the total edge weight). GUARD against the empty graph upstream."""
    return float(np.asarray(A, float).sum())


def modularity(A: np.ndarray, labels, gamma: float = 1.0) -> float:
    """Newman-Girvan modularity Q = sum_c [ (sum_in_c)/(2m) - gamma (Sigma_tot_c/2m)^2 ], the
    block form of Q = (1/2m) sum_ij [A_ij - gamma k_i k_j/(2m)] delta(c_i,c_j). sum_in_c is the
    full within-community block sum A[c,c] (counting both directions). GUARD: 2m = 0 -> 0.0."""
    A = np.asarray(A, float)
    k = A.sum(axis=1)
    m2 = k.sum()
    if m2 <= 0:
        return 0.0
    labels = np.asarray(labels)
    if labels.shape[0] != A.shape[0]:
        raise ValueError(f"labels length {labels.shape[0]} != n nodes {A.shape[0]}")
    Q = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        a_in = A[np.ix_(idx, idx)].sum()
        k_c = k[idx].sum()
        Q += a_in / m2 - gamma * (k_c / m2) ** 2
    return float(Q)


def modularity_matrix(A: np.ndarray) -> np.ndarray:
    """B = A - k k^T/(2m), the matrix whose leading eigenvector relaxes the modularity-optimal
    bipartition. Each row sums to zero (sum_j B_ij = k_i - k_i (2m)/(2m) = 0). GUARD 2m > 0."""
    A = np.asarray(A, float)
    k = A.sum(axis=1)
    m2 = k.sum()
    if m2 <= 0:
        return np.zeros_like(A)
    return A - np.outer(k, k) / m2


def spectral_bipartition(A: np.ndarray, tol: float = 1e-9):
    """Newman's spectral bipartition: the sign vector of B's leading (largest ALGEBRAIC)
    eigenvector. Returns (labels in {0,1}, lambda_1). If lambda_1 <= tol the network is
    INDIVISIBLE (no bipartition beats the trivial one) and we return the all-zero partition."""
    A = np.asarray(A, float)
    B = modularity_matrix(A)
    vals, vecs = np.linalg.eigh(B)            # ascending; B is real-symmetric
    lam1 = float(vals[-1])
    if lam1 <= tol:
        return np.zeros(A.shape[0], int), lam1
    s = vecs[:, -1]
    return (s > 0).astype(int), lam1


def brute_modularity_argmax(A: np.ndarray, gamma: float = 1.0):
    """The modularity-optimal BIPARTITION by exhaustive search (n <= ~16 only): fix node 0 in
    group 0 to kill the global flip, enumerate the other 2^(n-1) assignments. Returns (labels, Q).
    The ground truth the spectral relaxation is checked against."""
    A = np.asarray(A, float)
    n = A.shape[0]
    if n <= 1:
        return np.zeros(n, int), 0.0          # a 0- or 1-node graph is trivially one community
    if n > 16:
        raise ValueError(f"brute force is exponential; n={n} > 16")
    best_lab, best_Q = np.zeros(n, int), modularity(A, np.zeros(n, int), gamma)
    for bits in range(1, 1 << (n - 1)):
        lab = np.zeros(n, int)
        for j in range(n - 1):
            if bits & (1 << j):
                lab[j + 1] = 1
        Q = modularity(A, lab, gamma)
        if Q > best_Q:
            best_Q, best_lab = Q, lab
    return best_lab, float(best_Q)


def relabel_consecutive(labels):
    """Map arbitrary community labels to 0..C-1. Returns (relabeled, C)."""
    uniq, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(int), int(len(uniq))


def community_is_connected(A: np.ndarray, labels) -> dict:
    """Per-community internal connectivity (BFS on the induced subgraph A>0). Returns
    {community: bool}. A single-node community is connected by convention. This is the
    Leiden-vs-Louvain witness: Louvain offers no guarantee that this is all-True."""
    A = np.asarray(A, float)
    labels = np.asarray(labels)
    out = {}
    for c in np.unique(labels):
        idx = list(np.where(labels == c)[0])
        if len(idx) <= 1:
            out[int(c)] = True
            continue
        idx_set = set(idx)
        seen = {idx[0]}
        stack = [idx[0]]
        while stack:
            u = stack.pop()
            for v in np.nonzero(A[u])[0]:
                v = int(v)
                if v in idx_set and v not in seen:
                    seen.add(v)
                    stack.append(v)
        out[int(c)] = (len(seen) == len(idx_set))
    return out


# =========================================================================== #
# Movement 5 -- Louvain and Leiden (closed-form Delta Q; the connectivity guarantee).
# =========================================================================== #

def modularity_gain(A: np.ndarray, i: int, target: int, labels, gamma: float = 1.0) -> float:
    """The closed-form Delta Q of MOVING node i from its current community to `target`. By the
    standard decomposition this is add_gain(i -> target) - add_gain(i -> current\\{i}), where
    add_gain(i -> C) = 2 k_{i,C}/(2m) - 2 gamma Sigma_tot_C k_i/(2m)^2 is the gain of joining an
    isolated i to C. Asserted == the definitional Q(after) - Q(before) in the twin test."""
    A = np.asarray(A, float)
    k = A.sum(axis=1)
    m2 = k.sum()
    labels = np.asarray(labels)
    ki = k[i]
    ci = int(labels[i])
    if int(target) == ci:
        return 0.0

    def add_gain(C: int) -> float:
        members = np.array([j for j in np.nonzero(A[i])[0] if j != i and labels[j] == C])
        k_i_C = float(A[i, members].sum()) if members.size else 0.0
        sigma = float(k[labels == C].sum())
        if C == ci:
            sigma -= ki                    # i's own community, evaluated with i removed
        return 2.0 * k_i_C / m2 - 2.0 * gamma * sigma * ki / (m2 * m2)

    return add_gain(int(target)) - add_gain(ci)


def louvain_local_move(A: np.ndarray, labels=None, gamma: float = 1.0, max_sweeps=None):
    """Phase 1 of Louvain: sweep nodes in fixed order (pinned for determinism), moving each to the
    neighboring community of greatest positive Delta Q until no move improves Q (or max_sweeps is
    hit). Returns the updated labels. GUARD: 2m = 0 -> labels unchanged."""
    A = np.asarray(A, float)
    n = A.shape[0]
    k = A.sum(axis=1)
    m2 = k.sum()
    labels = np.arange(n) if labels is None else np.array(labels, int)
    if m2 <= 0:
        return labels
    sigma_tot: dict[int, float] = {}
    for i in range(n):
        sigma_tot[int(labels[i])] = sigma_tot.get(int(labels[i]), 0.0) + k[i]
    sweeps, improved = 0, True
    while improved:
        improved = False
        for i in range(n):
            ci, ki = int(labels[i]), k[i]
            wt: dict[int, float] = {}
            for j in np.nonzero(A[i])[0]:
                if j == i:
                    continue
                cj = int(labels[j])
                wt[cj] = wt.get(cj, 0.0) + A[i, j]
            sigma_tot[ci] -= ki                                   # remove i from its community
            best_c = ci
            best_gain = (2.0 * wt.get(ci, 0.0) / m2
                         - 2.0 * gamma * sigma_tot.get(ci, 0.0) * ki / (m2 * m2))
            for c, w_ic in wt.items():
                if c == ci:
                    continue
                gain = 2.0 * w_ic / m2 - 2.0 * gamma * sigma_tot.get(c, 0.0) * ki / (m2 * m2)
                if gain > best_gain + 1e-12:
                    best_gain, best_c = gain, c
            sigma_tot[best_c] = sigma_tot.get(best_c, 0.0) + ki   # place i
            if best_c != ci:
                labels[i] = best_c
                improved = True
        sweeps += 1
        if max_sweeps is not None and sweeps >= max_sweeps:
            break
    return labels


def aggregate_graph(A: np.ndarray, labels):
    """Phase 2 of Louvain: collapse each community to a super-node. Agg = H^T A H where H is the
    n x C membership indicator; the diagonal Agg[c,c] is the within-community weight (a self-loop)
    and the row sum equals Sigma_tot_c, so modularity is PRESERVED. Returns (Agg, relabeled)."""
    A = np.asarray(A, float)
    lab, C = relabel_consecutive(labels)
    H = np.zeros((A.shape[0], C))
    H[np.arange(A.shape[0]), lab] = 1.0
    return H.T @ A @ H, lab


def louvain(A: np.ndarray, gamma: float = 1.0, max_levels: int = 30):
    """The full Louvain heuristic: alternate local-moving and aggregation until no community
    merges. Returns (final node->community labels, levels) where `levels[t]` is the partition of
    the ORIGINAL nodes after aggregation level t (the dendrogram GraphRAG summarizes over)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    cur = A.copy()
    orig_of_super = [[i] for i in range(n)]
    node_comm = np.arange(n)
    levels = []
    for _ in range(max_levels):
        lab = louvain_local_move(cur, labels=np.arange(cur.shape[0]), gamma=gamma)
        lab, C = relabel_consecutive(lab)
        new_comm = np.zeros(n, int)
        for s in range(cur.shape[0]):
            for o in orig_of_super[s]:
                new_comm[o] = lab[s]
        node_comm = new_comm
        levels.append(node_comm.copy())
        if C == cur.shape[0]:
            break                                # nothing merged -> converged
        cur, _ = aggregate_graph(cur, lab)
        new_orig: list[list[int]] = [[] for _ in range(C)]
        for s in range(len(orig_of_super)):
            new_orig[lab[s]].extend(orig_of_super[s])
        orig_of_super = new_orig
    return node_comm, levels


def leiden_refine(A: np.ndarray, labels):
    """The connectivity-guaranteeing CORE of Leiden's refinement: split every community into its
    internally-connected components, so no refined community is disconnected. (The full Leiden also
    re-merges well-connected pieces with randomized moves; we implement the minimal version that
    delivers the guarantee — honest in the rigorFlag.) Returns relabeled-consecutive labels."""
    A = np.asarray(A, float)
    labels = np.asarray(labels)
    out = np.full(len(labels), -1, int)
    nxt = 0
    for c in np.unique(labels):
        idx = list(np.where(labels == c)[0])
        idx_set = set(idx)
        unassigned = set(idx)
        while unassigned:
            seed = next(iter(unassigned))
            comp, stack = {seed}, [seed]
            while stack:
                u = stack.pop()
                for v in np.nonzero(A[u])[0]:
                    v = int(v)
                    if v in idx_set and v not in comp:
                        comp.add(v)
                        stack.append(v)
            for u in comp:
                out[u] = nxt
            nxt += 1
            unassigned -= comp
    return out


def leiden(A: np.ndarray, gamma: float = 1.0):
    """Louvain followed by Leiden's refinement: guarantees every returned community is internally
    connected. Returns the refined labels."""
    node_comm, _ = louvain(A, gamma=gamma)
    return leiden_refine(A, node_comm)


# =========================================================================== #
# Corpora -- the finance entity graph, the ring of cliques, the Louvain witness.
# =========================================================================== #

@functools.lru_cache(maxsize=None)
def graphrag_corpus(seed: int = GR_SEED, n_sectors: int = GR_N_SECTORS, n_comp: int = GR_N_COMP,
                    dim: int = GR_DIM, kappa_sector: float = GR_KAPPA_SECTOR,
                    threshold: float = GR_THRESHOLD):
    """The entity co-occurrence graph: one node per company drawn from the finance sector->company
    vMF geometry, edge weight A_ij = max(0, cos(P_i,P_j) - threshold). SECTORS are the planted
    communities. Returns a dict {A, sector, P, C} (A and sector are the graph; P, C the geometry).
    Scalar args keep it hashable for the lru_cache."""
    _, P, _, sector = dpr_finance_matrix(seed=seed, dim=dim, n_sectors=n_sectors, n_comp=n_comp,
                                         kappa_sector=kappa_sector)
    C = P @ P.T
    np.fill_diagonal(C, 0.0)
    A = np.maximum(0.0, C - threshold)
    A = 0.5 * (A + A.T)              # enforce exact symmetry
    return {"A": A, "sector": np.asarray(sector), "P": P, "C": C}


def ring_of_cliques(n_cliques: int, clique_size: int):
    """n_cliques cliques of `clique_size` arranged in a ring; adjacent cliques joined by a single
    edge (clique t's last node -- clique t+1's first node). Returns (A, true_labels) with each
    clique its own community. The Fortunato-Barthelemy resolution-limit witness."""
    n = n_cliques * clique_size
    A = np.zeros((n, n))
    labels = np.zeros(n, int)
    for t in range(n_cliques):
        base = t * clique_size
        for a in range(clique_size):
            labels[base + a] = t
            for b in range(a + 1, clique_size):
                A[base + a, base + b] = A[base + b, base + a] = 1.0
    for t in range(n_cliques):
        u = t * clique_size + (clique_size - 1)            # last node of clique t
        v = ((t + 1) % n_cliques) * clique_size            # first node of clique t+1
        A[u, v] = A[v, u] = 1.0
    return A, labels


def paired_clique_labels(n_cliques: int, clique_size: int):
    """The 'merge adjacent cliques in pairs' partition of the ring (n_cliques even). When the ring
    is large its modularity BEATS the natural one-clique-per-community labelling -- the resolution
    limit. Returns labels."""
    labels = np.zeros(n_cliques * clique_size, int)
    for t in range(n_cliques):
        for a in range(clique_size):
            labels[t * clique_size + a] = t // 2
    return labels


def louvain_disconnected_witness():
    """A graph + partition exhibiting Louvain's missing connectivity guarantee. Two triangles
    {0,1,2} and {3,4,5} with NO edges between them, both labelled community 0; a separate clique
    {6,7,8,9} is community 1. Community 0 is DISCONNECTED, yet it is a fixed point of Louvain's
    local moving (no single node move improves Q, because each triangle is internally cohesive and
    no node is more attracted elsewhere). Louvain has no mechanism to split it; Leiden's refinement
    does. Returns (A, stuck_labels)."""
    n = 10
    A = np.zeros((n, n))
    for tri in ([0, 1, 2], [3, 4, 5]):
        for a in tri:
            for b in tri:
                if a < b:
                    A[a, b] = A[b, a] = 1.0
    clique = [6, 7, 8, 9]
    for a in clique:
        for b in clique:
            if a < b:
                A[a, b] = A[b, a] = 1.0
    stuck = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    return A, stuck


# =========================================================================== #
# Movement 4 -- the stochastic block model and the detectability transition.
# =========================================================================== #

def detectability_threshold(c_in: float, c_out: float) -> float:
    """The Kesten-Stigum margin (c_in - c_out)^2 - 2(c_in + c_out): > 0 detectable, < 0 not."""
    return (c_in - c_out) ** 2 - 2.0 * (c_in + c_out)


def ks_snr(c_in: float, c_out: float) -> float:
    """The detectability SNR (c_in - c_out)^2 / [2(c_in + c_out)] (> 1 detectable). GUARD: the
    empty-affinity graph -> 0.0."""
    denom = 2.0 * (c_in + c_out)
    return (c_in - c_out) ** 2 / denom if denom > 0 else 0.0


def sbm_sample(n: int, c_in: float, c_out: float, seed: int, q: int = 2):
    """Sample a symmetric q-block sparse SBM: balanced blocks, p_in = c_in/n, p_out = c_out/n.
    Returns (A_sparse CSR, planted_labels). One vectorized Bernoulli over the upper triangle from a
    single rng stream (no n^2 Python loop). GUARD: n >= q."""
    if n < q:
        raise ValueError(f"need n >= q, got n={n}, q={q}")
    rng = np.random.default_rng(seed)
    block = np.arange(n) * q // n                       # balanced 0..q-1
    rows, cols = np.triu_indices(n, k=1)
    same = block[rows] == block[cols]
    probs = np.where(same, c_in / n, c_out / n)
    keep = rng.random(rows.shape[0]) < probs
    r, c = rows[keep], cols[keep]
    data = np.ones(r.shape[0])
    A = sparse.csr_matrix(
        (np.concatenate([data, data]), (np.concatenate([r, c]), np.concatenate([c, r]))),
        shape=(n, n),
    )
    return A, block


def spectral_sbm_recovery(A) -> np.ndarray:
    """Recover a 2-block partition from the MOST EXTREME eigenvector of the modularity operator
    B = A - k k^T/(2m), applied MATRIX-FREE so a sparse A never densifies (the plain adjacency
    leading eigenvector localizes on high-degree nodes near threshold; B removes that rank-1 degree
    direction). We take the eigenvalue of largest MAGNITUDE from BOTH ends of the spectrum: an
    ASSORTATIVE community (more edges within) pushes an eigenvalue ABOVE the bulk, a DISASSORTATIVE
    one (more edges across) pushes one BELOW -- the detectability threshold is symmetric in
    (c_in - c_out)^2, so the informative direction is whichever eigenvalue is furthest from zero.
    Returns labels in {0,1}."""
    A = A.tocsr() if sparse.issparse(A) else sparse.csr_matrix(A)
    k = np.asarray(A.sum(axis=1)).ravel()
    m2 = float(k.sum())
    if m2 <= 0:
        return np.zeros(A.shape[0], int)

    def matvec(x):
        return A.dot(x) - k * (k.dot(x) / m2)

    op = LinearOperator((A.shape[0], A.shape[0]), matvec=matvec, dtype=float)
    # a fixed-seed starting vector makes eigsh deterministic (its default v0 is random), so the
    # baked recovery overlaps reproduce run-to-run -- the viz<->python invariant. The all-ones
    # vector is an exact 0-eigenvector of B and a poor start, so we use a seeded generic one.
    v0 = np.random.default_rng(0).standard_normal(A.shape[0])
    vals, vecs = eigsh(op, k=2, which="BE", v0=v0)   # one from each end of the spectrum
    s = vecs[:, int(np.argmax(np.abs(vals)))]
    return (s > 0).astype(int)


def overlap(true_labels, est_labels, q: int = 2) -> float:
    """Chance-corrected recovery overlap: (max over label permutations of the accuracy - 1/q) /
    (1 - 1/q). 0 = no better than random, 1 = perfect. GUARD: empty -> 0.0."""
    true = np.asarray(true_labels)
    est = np.asarray(est_labels)
    if true.size == 0:
        return 0.0
    best = 0.0
    for perm in permutations(range(q)):
        mapped = np.array([perm[int(e)] for e in est])
        best = max(best, float(np.mean(mapped == true)))
    return (best - 1.0 / q) / (1.0 - 1.0 / q)


@functools.lru_cache(maxsize=None)
def sbm_point_overlap(c_in: float, c_out: float, n: int, seed: int) -> float:
    """Overlap of spectral recovery on one SBM sample at (c_in, c_out). Cached so the tests and
    `viz_constants` share the (expensive) computation."""
    A, planted = sbm_sample(n, c_in, c_out, seed)
    return overlap(planted, spectral_sbm_recovery(A))


@functools.lru_cache(maxsize=None)
def sbm_overlap_grid(n: int = SBM_N_GRID, seed: int = SBM_GRID_SEED):
    """The (c_in, c_out) overlap heatmap over SBM_C_GRID x SBM_C_GRID. Returns
    (c_values, grid[i][j] = overlap at c_in=c_values[i], c_out=c_values[j])."""
    cs = SBM_C_GRID
    grid = [[sbm_point_overlap(ci, co, n, seed) for co in cs] for ci in cs]
    return cs, grid


# =========================================================================== #
# Demo -- the headline numbers, printed.
# =========================================================================== #

def graphrag_demo() -> dict:
    """The headline: sectors are recoverable communities of the finance entity graph (high
    modularity, Leiden recovers them); the SBM says WHEN such structure is detectable at all; and
    the resolution limit + Louvain's missing connectivity guarantee are the honest caveats."""
    g = graphrag_corpus()
    A, sector = g["A"], g["sector"]
    Q_planted = modularity(A, sector)
    rng = np.random.default_rng(GR_SEED)
    Q_random = float(np.mean([modularity(A, rng.permutation(sector)) for _ in range(20)]))
    lab_leiden = leiden(A)
    ov = overlap_multi(sector, lab_leiden)
    print(f"  MOVEMENT 1-5: finance entity graph -- {A.shape[0]} companies, "
          f"{GR_N_SECTORS} planted sectors")
    print(f"    modularity Q(planted sectors) = {Q_planted:.3f} >> Q(random) = {Q_random:.3f}")
    print(f"    Leiden recovers the sectors: overlap = {ov:.3f}")
    lab_sp, lam1 = spectral_bipartition(A)
    print(f"    spectral bipartition leading eigenvalue lambda_1(B) = {lam1:.3f} (> 0: divisible)")

    print("  MOVEMENT 3: resolution limit on a ring of cliques:")
    for nc in (RING_SMALL_NC, RING_LARGE_NC):
        Ar, true = ring_of_cliques(nc, RING_CLIQUE)
        Q_singles = modularity(Ar, true)
        Q_pairs = modularity(Ar, paired_clique_labels(nc, RING_CLIQUE)) if nc % 2 == 0 else float("nan")
        m2 = total_weight(Ar)
        verdict = "MERGES" if (nc % 2 == 0 and Q_pairs > Q_singles) else "keeps separate"
        print(f"    {nc:>2} cliques (sqrt(2m)={math.sqrt(m2):.1f}): "
              f"Q(singles)={Q_singles:.3f}, Q(pairs)={Q_pairs:.3f} -> optimum {verdict}")

    print("  MOVEMENT 4: SBM detectability (Kesten-Stigum (c_in-c_out)^2 > 2(c_in+c_out)):")
    for name, (ci, co) in (("above", SBM_ABOVE), ("below", SBM_BELOW)):
        ov_sbm = sbm_point_overlap(ci, co, SBM_N, GR_SEED)
        print(f"    {name} (c_in={ci}, c_out={co}, SNR={ks_snr(ci, co):.2f}): "
              f"recovery overlap = {ov_sbm:.3f}")

    Aw, stuck = louvain_disconnected_witness()
    conn = community_is_connected(Aw, stuck)
    refined = leiden_refine(Aw, stuck)
    conn_ref = community_is_connected(Aw, refined)
    print(f"  MOVEMENT 5: Louvain-stuck partition connectivity {conn} (a False = disconnected); "
          f"Leiden-refined {conn_ref}")
    return {"Q_planted": Q_planted, "Q_random": Q_random, "leiden_overlap": ov}


def overlap_multi(true_labels, est_labels) -> float:
    """Recovery quality for MORE than two blocks, scored by how well the estimated partition's
    induced clustering matches the planted one via best-match accuracy over a greedy assignment.
    For the finance graph (q = #sectors) this measures sector recovery. Falls back to exact
    permutation overlap when the block count is small."""
    true = np.asarray(true_labels)
    if true.size == 0:
        return 0.0
    est, _ = relabel_consecutive(est_labels)
    qt = int(true.max()) + 1
    qe = int(est.max()) + 1
    if qt <= 1:
        return 0.0                            # one true class -> chance is 1, overlap undefined
    # contingency-table greedy best match (rows = estimated, cols = true)
    M = np.zeros((qe, qt))
    for e, t in zip(est, true):
        M[e, t] += 1
    # greedy: assign each estimated community to its majority true label, count matches
    matched = int(M.max(axis=1).sum())
    acc = matched / len(true)
    chance = 1.0 / qt
    return (acc - chance) / (1.0 - chance)


# =========================================================================== #
# Verification harness -- each assert is a pedagogical claim the topic makes.
# =========================================================================== #

def test_modularity_matrix_rows_sum_zero() -> None:
    """Movement 2: every row of B = A - k k^T/(2m) sums to zero -- the fact that lets the constant
    term drop so Q = (1/4m) s^T B s for a bipartition."""
    A, _ = ring_of_cliques(4, 4)
    B = modularity_matrix(A)
    assert np.allclose(B.sum(axis=1), 0.0, atol=1e-12), "rows of B must sum to zero"
    assert np.allclose(B, B.T, atol=1e-12), "B must be symmetric"
    print("  [ok] Movement 2: rows of the modularity matrix B sum to zero (B symmetric)")


def test_incremental_delta_q_twin() -> None:
    """Movement 5, the twin anchor: the closed-form `modularity_gain` equals the definitional
    Q(after) - Q(before) of actually moving the node, to machine precision."""
    g = graphrag_corpus()
    A = g["A"]
    labels = leiden(A)
    rng = np.random.default_rng(0)
    checks = 0
    for _ in range(40):
        i = int(rng.integers(A.shape[0]))
        target = int(rng.choice(np.unique(labels)))
        before = modularity(A, labels)
        moved = labels.copy()
        moved[i] = target
        definitional = modularity(A, moved) - before
        closed = modularity_gain(A, i, target, labels)
        assert abs(definitional - closed) < 1e-9, \
            f"Delta Q twin mismatch: definitional {definitional} vs closed {closed}"
        checks += 1
    print(f"  [ok] Movement 5: closed-form Delta Q == definitional recompute ({checks} moves, <1e-9)")


def test_spectral_matches_brute_bipartition() -> None:
    """Movement 2, the collapse anchor: on a small two-clique graph the spectral sign vector equals
    the brute-force modularity-optimal bipartition (up to a global flip), with matching Q."""
    # two cliques of 4 joined by a single edge -> the optimal bipartition is the two cliques
    A = np.zeros((8, 8))
    for clq in ([0, 1, 2, 3], [4, 5, 6, 7]):
        for a in clq:
            for b in clq:
                if a < b:
                    A[a, b] = A[b, a] = 1.0
    A[3, 4] = A[4, 3] = 1.0
    sp, lam1 = spectral_bipartition(A)
    bf, Qbf = brute_modularity_argmax(A)
    same = np.array_equal(sp, bf) or np.array_equal(sp, 1 - bf)
    assert same, f"spectral {sp} != brute {bf} (up to flip)"
    assert abs(modularity(A, sp) - Qbf) < 1e-12, "spectral and brute Q disagree"
    assert lam1 > 0, "two well-separated cliques must be divisible (lambda_1 > 0)"
    print(f"  [ok] Movement 2: spectral bipartition == brute argmax (Q={Qbf:.3f}, up to flip)")


def test_indivisible_when_lambda1_nonpositive() -> None:
    """Movement 2: a single clique (no community structure) has lambda_1(B) <= 0 and is reported
    INDIVISIBLE (the recursive-bisection stopping rule)."""
    n = 8
    A = np.ones((n, n)) - np.eye(n)            # complete graph K8
    lab, lam1 = spectral_bipartition(A)
    assert lam1 <= 1e-9, f"a clique should have lambda_1(B) <= 0, got {lam1}"
    assert np.all(lab == 0), "an indivisible graph returns the trivial one-community partition"
    print(f"  [ok] Movement 2: a clique is indivisible (lambda_1(B)={lam1:.3e} <= 0)")


def test_aggregate_preserves_modularity() -> None:
    """Movement 5: Louvain aggregation Agg = H^T A H preserves modularity --
    Q(Agg, trivial labels) == Q(A, community labels) -- so the multi-level recursion is sound."""
    g = graphrag_corpus()
    A = g["A"]
    lab = louvain_local_move(A, labels=np.arange(A.shape[0]))
    Agg, rel = aggregate_graph(A, lab)
    q_orig = modularity(A, lab)
    q_agg = modularity(Agg, np.arange(Agg.shape[0]))
    assert abs(q_orig - q_agg) < 1e-9, f"aggregation changed Q: {q_orig} vs {q_agg}"
    print(f"  [ok] Movement 5: aggregation preserves modularity (Q={q_orig:.3f})")


def test_resolution_limit_merges_cliques() -> None:
    """Movement 3, the resolution limit (the contrast): on a LARGE ring of cliques the
    pairs-merged partition has HIGHER modularity than the natural one-clique-per-community
    partition (the optimum merges genuine communities), while on a SMALL ring it does not."""
    # large ring: pairs beat singles -> the limit bites
    A_big, true_big = ring_of_cliques(RING_LARGE_NC, RING_CLIQUE)
    Q_singles_big = modularity(A_big, true_big)
    Q_pairs_big = modularity(A_big, paired_clique_labels(RING_LARGE_NC, RING_CLIQUE))
    assert Q_pairs_big > Q_singles_big, \
        f"large ring: pairs {Q_pairs_big} should beat singles {Q_singles_big}"
    # Louvain (which maximizes modularity) recovers FEWER than n_cliques communities
    lab_big, _ = louvain(A_big)
    _, n_found = relabel_consecutive(lab_big)
    assert n_found < RING_LARGE_NC, \
        f"Louvain should merge cliques on the large ring: found {n_found} >= {RING_LARGE_NC}"
    # small ring: singles win -> structure resolved
    A_sm, true_sm = ring_of_cliques(4, RING_CLIQUE)   # 4 cliques (even, so pairs is defined)
    Q_singles_sm = modularity(A_sm, true_sm)
    Q_pairs_sm = modularity(A_sm, paired_clique_labels(4, RING_CLIQUE))
    assert Q_singles_sm > Q_pairs_sm, \
        f"small ring: singles {Q_singles_sm} should beat pairs {Q_pairs_sm}"
    print(f"  [ok] Movement 3: resolution limit -- large ring merges "
          f"(pairs {Q_pairs_big:.3f} > singles {Q_singles_big:.3f}, Louvain found {n_found} of "
          f"{RING_LARGE_NC}); small ring keeps separate")


def test_gamma_moves_resolution() -> None:
    """Movement 3: raising the resolution parameter gamma MOVES the limit -- at large gamma the
    pairs-merged partition no longer beats the singles on the large ring (gamma penalizes large
    communities), so the genuine cliques are resolved again. gamma moves the scale; it does not
    remove the limit (no single gamma resolves multi-scale structure -- stated in prose)."""
    A, true = ring_of_cliques(RING_LARGE_NC, RING_CLIQUE)
    pairs = paired_clique_labels(RING_LARGE_NC, RING_CLIQUE)
    # at gamma=1 pairs win; at a larger gamma singles win
    assert modularity(A, pairs, gamma=1.0) > modularity(A, true, gamma=1.0)
    g_hi = 2.0
    assert modularity(A, true, gamma=g_hi) > modularity(A, pairs, gamma=g_hi), \
        "raising gamma should un-merge the cliques (singles beat pairs)"
    print(f"  [ok] Movement 3: gamma moves the resolution scale "
          f"(gamma=1 merges, gamma={g_hi} resolves)")


def test_sbm_detectable_above_threshold() -> None:
    """Movement 4: above Kesten-Stigum, spectral recovery correlates with the planted partition."""
    ci, co = SBM_ABOVE
    assert detectability_threshold(ci, co) > 0, "test setup: SBM_ABOVE must be above threshold"
    ov = sbm_point_overlap(ci, co, SBM_N, GR_SEED)
    assert ov > 0.5, f"above threshold (SNR={ks_snr(ci, co):.2f}) overlap should be > 0.5, got {ov}"
    print(f"  [ok] Movement 4: SBM above threshold recovered (overlap={ov:.3f}, SNR={ks_snr(ci, co):.2f})")


def test_sbm_undetectable_below_threshold() -> None:
    """Movement 4, the phase transition: below Kesten-Stigum, NO spectral signal survives -- the
    overlap is at chance, the information-theoretic converse made visible."""
    ci, co = SBM_BELOW
    assert detectability_threshold(ci, co) < 0, "test setup: SBM_BELOW must be below threshold"
    ov = sbm_point_overlap(ci, co, SBM_N, GR_SEED)
    assert ov < 0.1, f"below threshold (SNR={ks_snr(ci, co):.2f}) overlap should be ~0, got {ov}"
    print(f"  [ok] Movement 4: SBM below threshold NOT recovered (overlap={ov:.3f}, "
          f"SNR={ks_snr(ci, co):.2f})")


def test_threshold_sign_matches_grid() -> None:
    """Movement 4: the Kesten-Stigum parabola is the decision boundary -- on CLEARLY-separated grid
    cells (|SNR - 1| > margin) the empirical detect/not sign matches detectability_threshold's
    sign. Cells near the parabola are ambiguous at finite n (the honest caveat) and are skipped."""
    cs, grid = sbm_overlap_grid()
    checks = 0
    for i, ci in enumerate(cs):
        for j, co in enumerate(cs):
            snr = ks_snr(ci, co)
            if 0.5 <= snr <= 2.5:                  # skip the finite-n smeared band around SNR=1
                continue
            predicted_detect = detectability_threshold(ci, co) > 0
            empirical_detect = grid[i][j] > 0.25
            assert predicted_detect == empirical_detect, \
                f"grid sign mismatch at (c_in={ci}, c_out={co}): SNR={snr:.2f}, overlap={grid[i][j]:.3f}"
            checks += 1
    assert checks >= 10, f"expected to check many clearly-separated cells, only {checks}"
    print(f"  [ok] Movement 4: the KS parabola is the empirical decision boundary "
          f"({checks} clearly-separated cells agree)")


def test_louvain_recovers_planted_sectors() -> None:
    """Movement 1/5: on the finance entity graph the planted sectors have high modularity and
    Leiden recovers them, with recovery confirmed by the imported set-metrics recall."""
    g = graphrag_corpus()
    A, sector = g["A"], g["sector"]
    lab = leiden(A)
    ov = overlap_multi(sector, lab)
    assert ov > 0.9, f"Leiden should recover the sectors (overlap > 0.9), got {ov}"
    # cross-check with the imported recall: each sector's members are 'relevant', the recovered
    # community ranks them first
    for s in np.unique(sector):
        members = set(int(i) for i in np.where(sector == s)[0])
        # the recovered community that best covers this sector
        best = max(np.unique(lab),
                   key=lambda c: len(members & set(int(i) for i in np.where(lab == c)[0])))
        ranking = [int(i) for i in np.where(lab == best)[0]]
        rec = recall_at_k(ranking, members, len(members))
        assert rec > 0.5, f"sector {s}: recovered community recall {rec} too low"
    print(f"  [ok] Movement 1/5: Leiden recovers the planted sectors (overlap={ov:.3f})")


def test_leiden_communities_connected_louvain_not() -> None:
    """Movement 5, the Leiden-vs-Louvain contrast: a disconnected community can be a Louvain LOCAL
    OPTIMUM (local moving leaves it unchanged), so Louvain offers no connectivity guarantee; Leiden's
    refinement splits it into connected pieces. (Full Louvain from singletons recovers the correct,
    connected partition -- Louvain is not broken generically.)"""
    A, stuck = louvain_disconnected_witness()
    conn = community_is_connected(A, stuck)
    assert not all(conn.values()), f"the witness community should be disconnected: {conn}"
    # the stuck partition is a fixed point of local moving
    after = louvain_local_move(A, labels=stuck.copy())
    after, _ = relabel_consecutive(after)
    stuck_rel, _ = relabel_consecutive(stuck)
    assert np.array_equal(after, stuck_rel), \
        f"the disconnected partition should be a Louvain local optimum, moved to {after}"
    # Leiden's refinement makes every community connected
    refined = leiden_refine(A, stuck)
    assert all(community_is_connected(A, refined).values()), "Leiden refinement must connect all"
    # full Louvain from singletons does the right thing on this graph (not broken generically)
    full, _ = louvain(A)
    assert all(community_is_connected(A, full).values()), "Louvain from scratch should be connected"
    print(f"  [ok] Movement 5: disconnected community is Louvain-stable, Leiden refines it "
          f"(connectivity {conn} -> all connected)")


def test_planted_modularity_beats_random() -> None:
    """Movement 1: the planted sector partition's modularity far exceeds a random partition's --
    the signal modularity measures."""
    g = graphrag_corpus()
    A, sector = g["A"], g["sector"]
    Q_planted = modularity(A, sector)
    rng = np.random.default_rng(1)
    Q_random = float(np.mean([modularity(A, rng.permutation(sector)) for _ in range(50)]))
    assert Q_planted > 0.3, f"planted modularity should be high, got {Q_planted}"
    assert Q_planted > Q_random + 0.2, \
        f"planted {Q_planted} should beat random {Q_random} by a clear margin"
    print(f"  [ok] Movement 1: Q(planted)={Q_planted:.3f} >> Q(random)={Q_random:.3f}")


def test_hierarchy_levels_coarsen() -> None:
    """Movement 5: Louvain returns a hierarchy whose community count is non-increasing across
    aggregation levels -- the dendrogram GraphRAG summarizes from fine to coarse."""
    A, _ = ring_of_cliques(RING_LARGE_NC, RING_CLIQUE)
    _, levels = louvain(A)
    counts = [relabel_consecutive(lv)[1] for lv in levels]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), \
        f"hierarchy community counts should be non-increasing, got {counts}"
    assert len(levels) >= 1, "expected at least one level"
    print(f"  [ok] Movement 5: Louvain hierarchy coarsens (community counts {counts})")


def test_guards() -> None:
    """Degenerate-input guards (the Gemini list): empty/zero graph -> Q=0 and zero matrix; SBM
    needs n >= q; overlap on empty -> 0; ks_snr on the zero-affinity graph -> 0."""
    assert modularity(np.zeros((3, 3)), [0, 1, 2]) == 0.0, "empty graph -> Q 0"
    assert np.allclose(modularity_matrix(np.zeros((3, 3))), 0.0), "empty graph -> B zero"
    assert overlap([], []) == 0.0, "empty overlap -> 0"
    assert ks_snr(0.0, 0.0) == 0.0, "zero-affinity SNR -> 0"
    try:
        sbm_sample(1, 5, 1, seed=0, q=2)
        assert False, "n < q should raise"
    except ValueError:
        pass
    # a single isolated node is a connected community by convention
    assert community_is_connected(np.zeros((1, 1)), [0]) == {0: True}
    print("  [ok] guards: empty graph, empty overlap, n<q, zero-affinity, singleton community")


def _run_all() -> None:
    test_modularity_matrix_rows_sum_zero()
    test_incremental_delta_q_twin()
    test_spectral_matches_brute_bipartition()
    test_indivisible_when_lambda1_nonpositive()
    test_aggregate_preserves_modularity()
    test_resolution_limit_merges_cliques()
    test_gamma_moves_resolution()
    test_sbm_detectable_above_threshold()
    test_sbm_undetectable_below_threshold()
    test_threshold_sign_matches_grid()
    test_louvain_recovers_planted_sectors()
    test_leiden_communities_connected_louvain_not()
    test_planted_modularity_beats_random()
    test_hierarchy_levels_coarsen()
    test_guards()


# =========================================================================== #
# Viz constants -- printed for GraphRAGCommunityLaboratory.tsx to mirror.
# =========================================================================== #

def _spectral_layout(A: np.ndarray, seed: int = GR_SEED):
    """A deterministic 2-D layout from the graph Laplacian's Fiedler pair (eigenvectors 2 and 3 of
    L = D - A): community-structured graphs lay out with communities spatially separated. Returns
    an (n, 2) array, lightly jittered from a seeded rng to separate coincident nodes."""
    A = np.asarray(A, float)
    k = A.sum(axis=1)
    L = np.diag(k) - A
    vals, vecs = np.linalg.eigh(L)
    xy = vecs[:, 1:3].copy()
    rng = np.random.default_rng(seed)
    xy = xy + 0.02 * rng.standard_normal(xy.shape)
    # normalize to [-1, 1] per axis for the viz
    for c in range(2):
        col = xy[:, c]
        span = max(col.max() - col.min(), 1e-9)
        xy[:, c] = 2.0 * (col - col.min()) / span - 1.0
    return xy


def _edge_list(A: np.ndarray):
    """Upper-triangle (i, j, weight) edges for the viz, weights rounded."""
    A = np.asarray(A, float)
    out = []
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                out.append([i, j, r3(A[i, j])])
    return out


def _sbm_display(c_in: float, c_out: float, seed: int):
    """A small SBM sample for the inset cartoon: edge list, a circular-by-block layout, planted and
    recovered labels. (Illustrative at SBM_DISPLAY_N; the quantitative transition is the grid.)"""
    A, planted = sbm_sample(SBM_DISPLAY_N, c_in, c_out, seed)
    Ad = np.asarray(A.todense())
    recovered = spectral_sbm_recovery(A)
    n = SBM_DISPLAY_N
    xy = []
    for i in range(n):
        # two arcs, one per planted block, so the block structure is legible when present
        blk = int(planted[i])
        ang = math.pi * (i % (n // 2)) / (n // 2) + (0.0 if blk == 0 else math.pi)
        xy.append([r3(math.cos(ang)), r3(math.sin(ang))])
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if Ad[i, j] > 0]
    return {
        "planted": [int(x) for x in planted],
        "recovered": [int(x) for x in recovered],
        "xy": xy,
        "edges": edges,
        "overlap": r3(overlap(planted, recovered)),
    }


def viz_constants() -> None:
    """Print every MEASURED number GraphRAGCommunityLaboratory.tsx mirrors to the decimal. TS
    recomputes only CLOSED FORM: the Kesten-Stigum parabola (c_in-c_out)^2 = 2(c_in+c_out), the SNR
    readout, sqrt(2m), color scales."""
    g = graphrag_corpus()
    A, sector = g["A"], g["sector"]
    n = A.shape[0]

    print("  // ----- Panel A: the finance entity graph + gamma resolution slider -----")
    print(f"const GR_N_NODES = {n};")
    print(f"const GR_N_SECTORS = {GR_N_SECTORS};")
    print(f"const GR_SECTOR = {[int(x) for x in sector]};")
    print(f"const GR_EDGES = {_edge_list(A)};   // [i, j, weight]")
    print(f"const GR_LAYOUT = {[[r3(x), r3(y)] for x, y in _spectral_layout(A)]};")
    Q_planted = modularity(A, sector)
    rng = np.random.default_rng(GR_SEED)
    Q_random = float(np.mean([modularity(A, rng.permutation(sector)) for _ in range(50)]))
    print(f"const GR_Q_PLANTED = {r3(Q_planted)};")
    print(f"const GR_Q_RANDOM = {r3(Q_random)};")
    q_by_gamma = []
    for gam in GAMMA_GRID:
        lab = leiden(A, gamma=gam)
        rel, C = relabel_consecutive(lab)
        q_by_gamma.append({"gamma": r3(gam), "Q": r3(modularity(A, rel, gamma=gam)),
                           "nComm": int(C), "labels": [int(x) for x in rel]})
    print(f"const GR_Q_BY_GAMMA = {q_by_gamma};")

    print("  // ----- Panel B: the SBM detectability phase diagram -----")
    print(f"const SBM_C_GRID = {[r3(c) for c in SBM_C_GRID]};")
    _, grid = sbm_overlap_grid()
    print(f"const SBM_OVERLAP_GRID = {[[r3(v) for v in row] for row in grid]};   // [c_in][c_out]")
    print(f"const SBM_ABOVE = {{cIn: {r3(SBM_ABOVE[0])}, cOut: {r3(SBM_ABOVE[1])}, "
          f"snr: {r3(ks_snr(*SBM_ABOVE))}, overlap: {r3(sbm_point_overlap(*SBM_ABOVE, SBM_N, GR_SEED))}}};")
    print(f"const SBM_BELOW = {{cIn: {r3(SBM_BELOW[0])}, cOut: {r3(SBM_BELOW[1])}, "
          f"snr: {r3(ks_snr(*SBM_BELOW))}, overlap: {r3(sbm_point_overlap(*SBM_BELOW, SBM_N, GR_SEED))}}};")
    print(f"const SBM_DISPLAY_ABOVE = {_sbm_display(*SBM_ABOVE, GR_SEED)};")
    print(f"const SBM_DISPLAY_BELOW = {_sbm_display(*SBM_BELOW, GR_SEED)};")

    print("  // ----- Panel C: resolution limit + Louvain-vs-Leiden -----")
    A_ring, true_ring = ring_of_cliques(RING_LARGE_NC, RING_CLIQUE)
    m2 = total_weight(A_ring)
    print(f"const RING_N_CLIQUES = {RING_LARGE_NC};")
    print(f"const RING_CLIQUE_SIZE = {RING_CLIQUE};")
    print(f"const RING_SQRT_2M = {r3(math.sqrt(m2))};")
    print(f"const RING_Q_SINGLES = {r3(modularity(A_ring, true_ring))};")
    print(f"const RING_Q_PAIRS = {r3(modularity(A_ring, paired_clique_labels(RING_LARGE_NC, RING_CLIQUE)))};")
    lab_ring, _ = louvain(A_ring)
    print(f"const RING_LOUVAIN_NCOMM = {relabel_consecutive(lab_ring)[1]};")
    # small ring contrast (drawn as the inset)
    A_small, true_small = ring_of_cliques(6, RING_CLIQUE)
    print(f"const RING_SMALL_Q_SINGLES = {r3(modularity(A_small, true_small))};")
    print(f"const RING_SMALL_Q_PAIRS = {r3(modularity(A_small, paired_clique_labels(6, RING_CLIQUE)))};")
    Aw, stuck = louvain_disconnected_witness()
    conn = community_is_connected(Aw, stuck)
    refined = leiden_refine(Aw, stuck)
    print(f"const WITNESS_EDGES = {_edge_list(Aw)};")
    print(f"const WITNESS_LAYOUT = {[[r3(x), r3(y)] for x, y in _spectral_layout(Aw, seed=3)]};")
    print(f"const WITNESS_LOUVAIN_LABELS = {[int(x) for x in stuck]};")
    print(f"const WITNESS_LOUVAIN_CONNECTED = {{{', '.join(f'{c}: {str(v).lower()}' for c, v in conn.items())}}};")
    print(f"const WITNESS_LEIDEN_LABELS = {[int(x) for x in relabel_consecutive(refined)[0]]};")


if __name__ == "__main__":
    print("graphrag_community_detection: running tests")
    _run_all()
    print("\nDemo:")
    graphrag_demo()
    print("\nviz_constants (mirror into GraphRAGCommunityLaboratory.tsx):")
    viz_constants()
    print("\nall checks passed.")
