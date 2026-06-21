"""Voronoi partitioning and the inverted-file (IVF) index, with IVFADC — the reference
implementation for the formalRAG `ivf-voronoi-partitioning` topic.

The quantization track built a codebook (Lloyd), factored it (product quantization), and learned
its rotation (optimized PQ). All of those compress a vector so a distance is cheap to estimate, but
they still SCAN every database vector. The inverted-file index is the non-exhaustive half: a coarse
quantizer partitions the space into Voronoi cells, each vector is filed under its nearest centroid,
and a query is compared only against the vectors in the few cells nearest to it. This module imports
its prerequisite quantization code (never reimplementing it) and establishes, and verifies, the two
movements of the index plus the honest caveats they rest on:

  MOVEMENT 1 — THE PARTITION (IVF).
  1. CANDIDATE-SET REDUCTION. With the database split into `nlist` Voronoi cells by k-means and a
     query probing its `nprobe` nearest cells, the expected number of vectors scanned is the total
     occupancy of those cells, which for balanced cells is about (nprobe/nlist) * n. With
     nlist ~ sqrt(n) and small nprobe this is a sqrt(n)-scale speedup. (`candidate_fraction`)
  2. EXHAUSTIVE PROBE IS EXACT. Probing all `nlist` cells with the exact distance recovers exact
     nearest-neighbor search: recall is 1. (`ivf_recall`, `test_full_probe_is_exact`)
  3. THE BOUNDARY EFFECT (the honest catch). A query's true nearest neighbor can lie in a cell whose
     centroid is NOT the query's nearest centroid, so recall at nprobe = 1 is strictly below 1;
     probing more cells (multi-probe) recovers it monotonically. (`test_boundary_effect`,
     `test_recall_monotone_in_nprobe`)

  MOVEMENT 2 — THE RESIDUAL (IVFADC).
  4. RESIDUAL VARIANCE REDUCTION. The coarse quantizer removes the between-cell variance, so the
     residual r = x - c_{i(x)} has strictly smaller total variance than x. Product-quantizing the
     RESIDUAL therefore spends the same bits on a smaller-variance signal. (`variance_reduction`)
  5. IVFADC BEATS FLAT PQ AT EQUAL BITS. At the same product-quantization budget, encoding the
     residual (IVFADC) reaches higher recall than encoding the raw vector (flat PQ), because the
     residual is cheaper to quantize. DIRECTION, not a magic number. (`ivfadc_recall`,
     `test_ivfadc_beats_flat_pq_equal_bits`)

Honest caveats (rigorFlag territory, asserted as DIRECTIONS): the sqrt(n) speedup is a balanced-cell
heuristic, not a worst-case guarantee — k-means cells are imbalanced, so list lengths and query time
vary; the partition is approximate, trading recall for speed via nprobe; and all numbers are measured
on the same synthetic finance cloud the previous topics used.

Every pedagogical claim is an `assert` below; `viz_constants()` prints what
`InvertedFileIndexLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/ivf-voronoi-partitioning/ivf_voronoi_partitioning.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.spatial.distance import cdist

# IVF files vectors under a Lloyd coarse quantizer and refines with product quantization. Import the
# verified k-means core AND the product quantizer (which itself imports the Lloyd core) — the
# established two-hop cross-topic pattern: add each prereq's HYPHENATED dir to the path, import its
# UNDERSCORED module.
_PQ_DIR = pathlib.Path(__file__).resolve().parents[1] / "product-quantization"
if str(_PQ_DIR) not in sys.path:
    sys.path.insert(0, str(_PQ_DIR))
_VQ_DIR = pathlib.Path(__file__).resolve().parents[1] / "vector-quantization-lloyd-max"
if str(_VQ_DIR) not in sys.path:
    sys.path.insert(0, str(_VQ_DIR))

from product_quantization import (  # noqa: E402
    adc_distance,
    adc_table,
    pq_bits,
    pq_encode,
    recall_adc,
    train_pq,
    true_topk,
)
from vector_quantization_lloyd_max import (  # noqa: E402
    assign,
    best_codebook,
    finance_dataset,
)


# --------------------------------------------------------------------------- #
# The coarse quantizer and the inverted lists.
# --------------------------------------------------------------------------- #

def coarse_quantizer(X: np.ndarray, nlist: int, seed: int = 0) -> np.ndarray:
    """Train the coarse quantizer: nlist Voronoi centroids over the database by Lloyd's k-means
    (the imported best_codebook). Returns the centroids (nlist, d). GUARD: nlist >= 1, <= n."""
    if nlist < 1:
        raise ValueError(f"nlist must be >= 1, got {nlist}")
    if nlist > X.shape[0]:
        raise ValueError(f"nlist ({nlist}) exceeds n_points ({X.shape[0]})")
    _, C, _ = best_codebook(X, nlist, seed=seed, restarts=3)
    return C


def inverted_lists(X: np.ndarray, C: np.ndarray):
    """File each vector under its nearest centroid. Returns (labels (n,), lists), where lists[i] is
    the array of database indices assigned to cell i (possibly empty)."""
    labels, _ = assign(X, C)
    lists = [np.where(labels == i)[0] for i in range(C.shape[0])]
    return labels, lists


def nearest_cells(q: np.ndarray, C: np.ndarray, nprobe: int) -> np.ndarray:
    """The nprobe nearest centroid indices to query q, nearest first (lowest-index tie-break).
    GUARD: cap nprobe at the number of cells so np.argpartition cannot raise."""
    nprobe = min(max(nprobe, 1), C.shape[0])
    d = cdist(q[None, :], C, "sqeuclidean")[0]
    idx = np.argpartition(d, nprobe - 1)[:nprobe]
    return idx[np.argsort(d[idx])]


# --------------------------------------------------------------------------- #
# IVF search with the exact distance, the candidate fraction, and the frontier.
# --------------------------------------------------------------------------- #

def ivf_recall(queries: np.ndarray, X: np.ndarray, C: np.ndarray, lists, nprobe: int,
               topk: int = 10, truth=None) -> float:
    """Mean recall@topk: probe the nprobe nearest cells and rank their members by the EXACT
    distance. nprobe = nlist recovers exact search (recall 1). GUARD: empty probed candidate set."""
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if truth is None:
        truth = true_topk(queries, X, topk)
    hits = 0
    for qi in range(len(queries)):
        probe = nearest_cells(queries[qi], C, nprobe)
        members = [lists[i] for i in probe if len(lists[i]) > 0]
        if not members:
            continue
        cand = np.concatenate(members)
        d = cdist(queries[qi][None, :], X[cand], "sqeuclidean")[0]
        kk = min(topk, len(cand))
        top = cand[np.argpartition(d, kk - 1)[:kk]]
        hits += len(truth[qi] & set(top.tolist()))
    return hits / (len(queries) * topk)


def candidate_fraction(queries: np.ndarray, C: np.ndarray, lists, nprobe: int) -> float:
    """Mean fraction of the database scanned at this nprobe: total occupancy of the probed cells
    over n. For balanced cells this is about nprobe/nlist; k-means cells are imbalanced, so it
    drifts. GUARDS: empty query set or empty database -> 0.0."""
    if len(queries) == 0:
        return 0.0
    sizes = np.array([len(l) for l in lists])
    n = int(sizes.sum())
    if n == 0:
        return 0.0
    fracs = [sizes[nearest_cells(q, C, nprobe)].sum() / n for q in queries]
    return float(np.mean(fracs))


def recall_vs_nprobe(queries, X, C, lists, nprobe_grid, topk: int = 10):
    """For each nprobe: recall@topk (exact distance) and the mean fraction of the database scanned.
    The speed/recall frontier the index trades along. Returns a list of row dicts."""
    truth = true_topk(queries, X, topk)
    rows = []
    for nprobe in nprobe_grid:
        rows.append({
            "nprobe": nprobe,
            "recall": ivf_recall(queries, X, C, lists, nprobe, topk, truth=truth),
            "frac": candidate_fraction(queries, C, lists, nprobe),
        })
    return rows


# --------------------------------------------------------------------------- #
# IVFADC: product-quantize the RESIDUAL after the coarse partition.
# --------------------------------------------------------------------------- #

def residuals(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The coarse-quantization residual r_i = x_i - c_{labels_i}: what the coarse centroid leaves."""
    return X - C[labels]


def variance_reduction(X: np.ndarray, C: np.ndarray, labels: np.ndarray):
    """Total variance (trace of the covariance) of the raw vectors versus the residuals. The coarse
    quantizer removes the between-cell variance, so the residual variance is strictly smaller — the
    reason product-quantizing the residual is cheaper. Returns (raw_var, residual_var)."""
    raw = float(np.var(X, axis=0).sum())
    res = float(np.var(residuals(X, C, labels), axis=0).sum())
    return raw, res


def build_ivfadc(X, C, labels, m, k_star, seed: int = 0):
    """IVFADC index: product-quantize the RESIDUALS (one shared sub-codebook set, the standard
    coarse-residual scheme). Returns (codebooks, codes) where codes[i] is the PQ code of x_i's
    residual relative to ITS OWN cell centroid."""
    res = residuals(X, C, labels)
    codebooks = train_pq(res, m, k_star, seed=seed)
    codes = pq_encode(res, codebooks)
    return codebooks, codes


def ivfadc_recall(queries, X, C, lists, codebooks, codes, nprobe, topk: int = 10, truth=None):
    """Mean recall@topk under IVFADC: probe the nprobe nearest cells and rank their members by the
    ASYMMETRIC distance between the query's residual q - c_i and the stored residual codes. nprobe =
    nlist is exhaustive ADC over residuals. GUARDS: non-empty queries, topk >= 1, empty probed set."""
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if truth is None:
        truth = true_topk(queries, X, topk)
    hits = 0
    for qi in range(len(queries)):
        probe = nearest_cells(queries[qi], C, nprobe)
        cand_idx, cand_d = [], []
        for i in probe:
            members = lists[i]
            if len(members) == 0:
                continue
            table = adc_table(queries[qi] - C[i], codebooks)   # query residual w.r.t. THIS cell
            cand_idx.append(members)
            cand_d.append(adc_distance(codes[members], table))
        if not cand_idx:
            continue
        cand = np.concatenate(cand_idx)
        d = np.concatenate(cand_d)
        kk = min(topk, len(cand))
        top = cand[np.argpartition(d, kk - 1)[:kk]]
        hits += len(truth[qi] & set(top.tolist()))
    return hits / (len(queries) * topk)


# --------------------------------------------------------------------------- #
# Toy 2-D cloud for the laboratory (the Voronoi partition and the boundary effect).
# --------------------------------------------------------------------------- #

TOY_NLIST = 6


def toy_ivf_cloud(seed: int = 1):
    """A 2-D cloud in [0,10]^2 — six loose Gaussian blobs — whose k-means cells make the boundary
    effect visible: at least one query whose true nearest neighbor sits in a cell other than the
    query's nearest centroid. Returns X (n, 2)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[2.0, 8.0], [5.0, 8.5], [8.0, 7.5],
                        [2.5, 2.5], [5.5, 2.0], [8.0, 3.0]])
    pts = [c + 1.1 * rng.standard_normal((10, 2)) for c in centers]
    return np.clip(np.vstack(pts), 0.0, 10.0)


def find_boundary_query(X, C, lists, seed: int = 3):
    """Search the toy cloud for a query exhibiting the boundary effect: its nearest centroid's cell
    does NOT contain its true nearest neighbor, but a deeper probe does. Returns (q, nn_index,
    nprobe_needed) — deterministic given the seeds. Falls back to the first query if none found."""
    rng = np.random.default_rng(seed)
    q0 = None
    for _ in range(400):
        q = rng.uniform(1.5, 8.5, size=2)
        nn = int(cdist(q[None, :], X, "sqeuclidean")[0].argmin())
        order = nearest_cells(q, C, C.shape[0])
        # cell that actually contains the NN, by rank among the query's nearest cells
        nn_cell = next(r for r, i in enumerate(order) if nn in lists[i])
        if q0 is None:
            q0 = (q, nn, nn_cell + 1)
        if nn_cell >= 1:                         # NN not in the single nearest cell -> boundary effect
            return q, nn, nn_cell + 1
    return q0


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

NLIST, NPROBE_GRID, PQ_M, PQ_KSTAR = 32, (1, 2, 4, 8, 16, 32), 8, 256


def test_full_probe_is_exact() -> None:
    """Probing all nlist cells with the exact distance IS exact nearest-neighbor search: recall 1."""
    X, _, queries = finance_dataset()
    C = coarse_quantizer(X, NLIST, seed=0)
    _, lists = inverted_lists(X, C)
    r_full = ivf_recall(queries, X, C, lists, NLIST, topk=10)
    assert abs(r_full - 1.0) < 1e-9, f"exhaustive probe recall {r_full:.4f} != 1"
    print(f"  [ok] exhaustive probe (nprobe=nlist={NLIST}) is exact: recall {r_full:.3f}")


def test_recall_monotone_in_nprobe() -> None:
    """Recall is non-decreasing in nprobe: probing more cells never loses a neighbor."""
    X, _, queries = finance_dataset()
    C = coarse_quantizer(X, NLIST, seed=0)
    _, lists = inverted_lists(X, C)
    rec = np.array([r["recall"] for r in recall_vs_nprobe(queries, X, C, lists, NPROBE_GRID)])
    assert np.all(np.diff(rec) >= -1e-9), f"recall not monotone in nprobe: {np.round(rec, 3)}"
    print(f"  [ok] recall monotone in nprobe: {np.round(rec, 3).tolist()} (nprobe={list(NPROBE_GRID)})")


def test_boundary_effect() -> None:
    """The honest catch: at nprobe = 1 some queries miss neighbors that live across a Voronoi
    boundary, so recall is strictly below the exhaustive recall of 1."""
    X, _, queries = finance_dataset()
    C = coarse_quantizer(X, NLIST, seed=0)
    _, lists = inverted_lists(X, C)
    r1 = ivf_recall(queries, X, C, lists, 1, topk=10)
    assert r1 < 1.0 - 1e-6, f"recall@nprobe=1 {r1:.4f} is not below 1 — no boundary effect to show"
    print(f"  [ok] boundary effect: recall@nprobe=1 = {r1:.3f} < 1 (neighbors across cell borders)")


def test_candidate_reduction() -> None:
    """At nprobe = 1 the index scans only a small fraction of the database, and the fraction grows
    monotonically to 1 at nprobe = nlist."""
    X, _, queries = finance_dataset()
    C = coarse_quantizer(X, NLIST, seed=0)
    _, lists = inverted_lists(X, C)
    fracs = np.array([candidate_fraction(queries, C, lists, p) for p in NPROBE_GRID])
    assert fracs[0] < 0.25, f"nprobe=1 scans {fracs[0]:.3f} of the DB — not a meaningful reduction"
    assert np.all(np.diff(fracs) >= -1e-9), f"candidate fraction not monotone: {np.round(fracs, 3)}"
    assert abs(fracs[-1] - 1.0) < 1e-9, f"nprobe=nlist must scan all: {fracs[-1]:.4f}"
    print(f"  [ok] candidate reduction: scan {fracs[0]:.3f} of DB at nprobe=1 -> {fracs[-1]:.2f} at "
          f"nprobe={NLIST}")


def test_residual_variance_reduction() -> None:
    """The coarse quantizer removes between-cell variance, so the residual has strictly smaller
    total variance than the raw vector — why product-quantizing the residual is cheaper."""
    X, _, _ = finance_dataset()
    C = coarse_quantizer(X, NLIST, seed=0)
    labels, _ = inverted_lists(X, C)
    raw, res = variance_reduction(X, C, labels)
    assert res < raw, f"residual variance {res:.2f} not below raw {raw:.2f}"
    print(f"  [ok] residual variance reduction: total variance {raw:.1f} (raw) -> {res:.1f} "
          f"(residual), {100 * (1 - res / raw):.0f}% removed")


def test_ivfadc_beats_flat_pq_equal_bits() -> None:
    """At the same product-quantization bit budget, encoding the residual (IVFADC, exhaustive ADC)
    reaches at least the recall of encoding the raw vector (flat PQ), because the residual is cheaper
    to quantize. DIRECTION only."""
    X, _, queries = finance_dataset()
    C = coarse_quantizer(X, NLIST, seed=0)
    labels, lists = inverted_lists(X, C)
    codebooks, codes = build_ivfadc(X, C, labels, PQ_M, PQ_KSTAR, seed=7)
    r_ivfadc = ivfadc_recall(queries, X, C, lists, codebooks, codes, NLIST, topk=10)
    flat_cb = train_pq(X, PQ_M, PQ_KSTAR, seed=7)
    r_flat = recall_adc(X, queries, flat_cb, topk=10)
    assert r_ivfadc >= r_flat - 1e-9, f"IVFADC recall {r_ivfadc:.3f} below flat PQ {r_flat:.3f}"
    print(f"  [ok] IVFADC beats flat PQ at equal bits ({int(pq_bits(PQ_M, PQ_KSTAR))}-bit codes): "
          f"recall {r_flat:.3f} (flat) -> {r_ivfadc:.3f} (IVFADC)")


def validate_against_bruteforce() -> None:
    """Cross-check the coarse search plumbing: IVF with exhaustive probe returns the same recall as
    a direct brute-force top-k over the whole database — no hidden indexing approximation."""
    X, _, queries = finance_dataset()
    C = coarse_quantizer(X, NLIST, seed=0)
    _, lists = inverted_lists(X, C)
    truth = true_topk(queries, X, 10)
    # brute force recall against itself is 1 by construction; confirm IVF exhaustive matches it.
    r_ivf = ivf_recall(queries, X, C, lists, NLIST, topk=10, truth=truth)
    assert abs(r_ivf - 1.0) < 1e-9, f"IVF exhaustive {r_ivf:.4f} != brute force 1.0"
    print(f"  [ok] cross-check: IVF exhaustive probe == brute-force exact search (recall {r_ivf:.3f})")


# --------------------------------------------------------------------------- #
# Viz constants — the exact numbers InvertedFileIndexLaboratory.tsx mirrors.
# --------------------------------------------------------------------------- #

def viz_constants() -> None:
    """Print the toy Voronoi cloud + boundary query (Panel A), the recall/fraction frontier
    (Panel B), and the residual-variance reduction + IVFADC-vs-flat-PQ recall (Panel C) — all baked
    to the decimal in the .tsx."""
    # Panel A: the 2-D cloud, its nlist centroids, cell labels, and a boundary-effect query.
    Xt = toy_ivf_cloud()
    Ct = coarse_quantizer(Xt, TOY_NLIST, seed=2)
    lab_t, lists_t = inverted_lists(Xt, Ct)
    q, nn, nprobe_needed = find_boundary_query(Xt, Ct, lists_t)
    fmt = lambda a: [[round(float(v), 3) for v in row] for row in a]
    print("  PANEL A — toy Voronoi partition (boundary effect):")
    print(f"    TOY_POINTS ({Xt.shape[0]} pts in R^2) = {fmt(Xt)}")
    print(f"    TOY_CENTROIDS (nlist={TOY_NLIST}) = {fmt(Ct)}")
    print(f"    TOY_LABELS = {lab_t.tolist()}")
    print(f"    BOUNDARY_QUERY = {[round(float(v), 3) for v in q]}  TRUE_NN_INDEX = {nn}  "
          f"NPROBE_TO_FIND_NN = {nprobe_needed}")
    print(f"    QUERY_CELL_ORDER = {nearest_cells(q, Ct, TOY_NLIST).tolist()}")

    # Panel B: the recall / candidate-fraction frontier on the finance cloud.
    X, _, queries = finance_dataset()
    C = coarse_quantizer(X, NLIST, seed=0)
    _, lists = inverted_lists(X, C)
    print(f"  PANEL B — recall / scan frontier (n={X.shape[0]}, nlist={NLIST}):")
    for r in recall_vs_nprobe(queries, X, C, lists, NPROBE_GRID):
        print(f"    nprobe={r['nprobe']:>3}: recall={r['recall']:.4f}  frac_scanned={r['frac']:.4f}")

    # Panel C: residual variance reduction and IVFADC vs flat PQ at equal bits.
    labels, _ = inverted_lists(X, C)
    raw, res = variance_reduction(X, C, labels)
    codebooks, codes = build_ivfadc(X, C, labels, PQ_M, PQ_KSTAR, seed=7)
    r_ivfadc = ivfadc_recall(queries, X, C, lists, codebooks, codes, NLIST, topk=10)
    r_flat = recall_adc(X, queries, train_pq(X, PQ_M, PQ_KSTAR, seed=7), topk=10)
    print("  PANEL C — IVFADC residual encoding:")
    print(f"    RAW_VARIANCE={raw:.4f} RESIDUAL_VARIANCE={res:.4f} "
          f"REMOVED_FRAC={1 - res / raw:.4f}")
    print(f"    EQUAL_BITS={int(pq_bits(PQ_M, PQ_KSTAR))} FLAT_PQ_RECALL={r_flat:.4f} "
          f"IVFADC_RECALL={r_ivfadc:.4f}")


if __name__ == "__main__":
    print("IVF / IVFADC verification harness")
    test_full_probe_is_exact()
    test_recall_monotone_in_nprobe()
    test_boundary_effect()
    test_candidate_reduction()
    test_residual_variance_reduction()
    test_ivfadc_beats_flat_pq_equal_bits()
    validate_against_bruteforce()
    print("Viz constants (mirrored to the decimal in InvertedFileIndexLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
