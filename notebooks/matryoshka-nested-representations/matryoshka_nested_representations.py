"""Matryoshka representations: jointly trained nested subspaces — the reference
implementation for the formalRAG `matryoshka-nested-representations` topic.

A Matryoshka embedding is trained so that every prefix z[:m] is itself a usable
representation: store one 1536-d vector, retrieve with its first 96 dims when speed
matters and its full width when accuracy does. The training objective is a weighted sum
of per-granularity losses, sum_m c_m * loss(z[:m]). This module makes the geometry
precise and verifies it:

  1. LINEAR MRL IS PCA. In the linear, squared-reconstruction setting the jointly
     optimal nested basis is exactly PCA's eigenvalue-ordered basis. PCA's top-m
     subspaces are NESTED and each is the Eckart-Young rank-m optimum, so one ordered
     basis simultaneously minimizes every prefix term — hence the weighted sum, for ANY
     positive granularity weights. Matryoshka generalizes PCA's nested-subspace
     optimality from "variance" to an arbitrary task loss.
  2. PREFIXES DEGRADE GRACEFULLY — IF NESTED. Prefix-m retrieval recall is monotone in
     m for the nested (PCA-ordered) embedding. A random rotation of the SAME embedding
     preserves full-d distances exactly but scrambles the prefixes, so truncating it
     destroys recall. That gap is why the nesting must be TRAINED, not assumed.
  3. ADAPTIVE (FUNNEL) RETRIEVAL. Shortlist with a cheap short prefix, rerank the
     shortlist at full width: recall close to exhaustive full-d search at a fraction of
     the scoring cost.

Every pedagogical claim is an `assert` below. The linear MRL = PCA optimality is exact
(Eckart-Young + nestedness); the "joint training beats independent heads" and the
nonlinear/contrastive instantiation are empirical (cited in the topic, not claimed as
theorems here). `grid_table()` prints the recall / reconstruction / funnel blocks that
`MatryoshkaLaboratory.tsx` mirrors to the decimal. `sklearn.decomposition.PCA` is the
reference cross-check.

Run:  uv run --with numpy --with scipy --with scikit-learn python notebooks/matryoshka-nested-representations/matryoshka_nested_representations.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as SklearnPCA

# Matryoshka granularities for the finance cloud (d = 1536, halving down to 24).
GRANULARITIES: tuple[int, ...] = (24, 48, 96, 192, 384, 768, 1536)


# --------------------------------------------------------------------------- #
# Synthetic embeddings — the SAME low-rank-plus-noise generator the PCA and JL
# topics use, so the nested-subspace story is told on a comparable cloud.
# --------------------------------------------------------------------------- #

def structured_embeddings(n: int, d: int, k: int, n_clusters: int = 1,
                          decay: float = 0.93, noise: float = 0.05,
                          cluster_sep: float = 2.5, seed: int = 0):
    """n embeddings in R^d on a k-dim latent subspace with a decaying spectrum.
    Returns (X, labels)."""
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


# --------------------------------------------------------------------------- #
# The nested basis (PCA = linear MRL) and a distance-preserving non-nested foil.
# --------------------------------------------------------------------------- #

def pca_basis(X: np.ndarray, d: int | None = None):
    """Eigenvalue-ordered PCA basis of the centered data: the linear-MRL optimum.
    Returns (V, singular_values, mean) with V the (D x d) top components as columns."""
    mu = X.mean(axis=0)
    Xc = X - mu
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = d or Vt.shape[0]
    return Vt[:d].T, s[:d], mu


def pca_embedding(X: np.ndarray, V: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Project onto the nested basis: scores Z = (X - mu) V, ordered by variance."""
    return (X - mu) @ V


def random_rotation(d: int, seed: int = 0) -> np.ndarray:
    """A Haar-ish random orthogonal d x d matrix (QR of a Gaussian). Rotating the
    embedding preserves every full-d distance but scrambles which information lands in
    the early coordinates — a representation with the SAME geometry and NO usable nesting."""
    Q, R = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return Q * np.sign(np.diag(R))            # fix signs so Q is a proper rotation/reflection


# --------------------------------------------------------------------------- #
# Reconstruction: prefix-m error and the Eckart-Young rank-m optimum.
# --------------------------------------------------------------------------- #

def prefix_recon_error(X: np.ndarray, V: np.ndarray, m: int) -> float:
    """Squared Frobenius reconstruction error of the rank-m prefix projector V[:,:m].
    Reassociate as (Xc @ Vm) @ Vm.T rather than forming the D x D projector Vm @ Vm.T:
    O(n D m) and no D x D allocation (a ~16x saving at D=1536, m=96)."""
    Xc = X - X.mean(axis=0)
    Vm = V[:, :m]
    return float(np.sum((Xc - (Xc @ Vm) @ Vm.T) ** 2))


def eckart_young_rankm(X: np.ndarray, m: int) -> float:
    """The best achievable rank-m reconstruction error = sum of tail singular values^2."""
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return float(np.sum(s[m:] ** 2))


def joint_mrl_loss(X: np.ndarray, V: np.ndarray, granularities, weights) -> float:
    """The Matryoshka objective in the linear-reconstruction case:
    sum_m c_m * ||Xc - Xc V[:,:m] V[:,:m]^T||_F^2."""
    return float(sum(w * prefix_recon_error(X, V, m) for m, w in zip(granularities, weights)))


# --------------------------------------------------------------------------- #
# Retrieval: prefix-m recall, and adaptive (funnel) retrieval.
# --------------------------------------------------------------------------- #

def _topk(queries: np.ndarray, corpus: np.ndarray, topk: int) -> np.ndarray:
    return np.argpartition(cdist(queries, corpus, "sqeuclidean"), topk, axis=1)[:, :topk]


def prefix_recall(Z: np.ndarray, Zq: np.ndarray, truth_sets, m: int, topk: int = 10) -> float:
    """recall@topk using only the first m embedding coordinates, against the full-width
    nearest neighbors `truth_sets`."""
    nn = _topk(Zq[:, :m], Z[:, :m], topk)
    return float(np.mean([len(truth_sets[i] & set(nn[i])) / topk for i in range(len(nn))]))


def full_truth(Z: np.ndarray, Zq: np.ndarray, topk: int = 10):
    """Nearest neighbors in the full-width embedding — the retrieval ground truth."""
    return [set(row) for row in _topk(Zq, Z, topk)]


def funnel_retrieval(Z: np.ndarray, Zq: np.ndarray, truth_sets, m_short: int,
                     topk: int = 10, shortlist: int = 50) -> tuple[float, float]:
    """Shortlist by a cheap m_short-dim prefix, then rerank the shortlist at full width.
    Returns (recall@topk, cost_fraction) where cost_fraction is scoring cost relative to
    an exhaustive full-width scan."""
    n, d = Z.shape
    short_nn = _topk(Zq[:, :m_short], Z[:, :m_short], shortlist)
    recalls = []
    for i in range(len(Zq)):
        cand = short_nn[i]
        d2 = np.sum((Z[cand] - Zq[i]) ** 2, axis=1)
        rerank = cand[np.argsort(d2)[:topk]]      # rerank set is small; full sort is cheap
        recalls.append(len(truth_sets[i] & set(rerank)) / topk)
    cost_fraction = (n * m_short + shortlist * d) / (n * d)
    return float(np.mean(recalls)), float(cost_fraction)


# --------------------------------------------------------------------------- #
# The showcase / finance dataset, and the grid table the viz mirrors.
# --------------------------------------------------------------------------- #

FINANCE_DIM = 1536
FINANCE_INTRINSIC_K = 48
FINANCE_CLUSTERS = 3
FINANCE_SHORT = 96         # the adaptive-retrieval shortlist prefix the demo reports
FUNNEL_M = 48              # fixed prefix width for the funnel recall/cost Pareto sweep


FINANCE_N = 2000           # > FINANCE_DIM, so the full nested basis spans all 1536 dims


def finance_dataset():
    """The canonical synthetic financial-embedding cloud. n > d so PCA yields a full
    1536-dim nested basis. SYNTHETIC low-rank-plus-noise, not a trained encoder."""
    X, labels = structured_embeddings(FINANCE_N, FINANCE_DIM, FINANCE_INTRINSIC_K,
                                      n_clusters=FINANCE_CLUSTERS, decay=0.97,
                                      noise=0.02, cluster_sep=1.2, seed=1)
    queries, _ = structured_embeddings(150, FINANCE_DIM, FINANCE_INTRINSIC_K,
                                       n_clusters=FINANCE_CLUSTERS, decay=0.97,
                                       noise=0.02, cluster_sep=1.2, seed=2)
    return X, labels, queries


def grid_table() -> dict[str, list]:
    """The numbers MatryoshkaLaboratory.tsx mirrors to the decimal. Deterministic.

    recall : per-granularity recall@10, nested (PCA-ordered) vs rotated (non-nested).
    recon  : per-granularity reconstruction error, nested basis vs Eckart-Young optimum.
    funnel : recall vs cost fraction for funnel retrieval at each shortlist prefix.
    """
    X, labels, queries = finance_dataset()
    V, s, mu = pca_basis(X)
    Z, Zq = pca_embedding(X, V, mu), pca_embedding(queries, V, mu)
    R = random_rotation(FINANCE_DIM, seed=7)
    Zr, Zqr = Z @ R, Zq @ R
    truth = full_truth(Z, Zq)

    Vrand = random_rotation(FINANCE_DIM, seed=13)        # a random orthonormal basis of R^1536
    recall = [{"m": m,
               "nested": prefix_recall(Z, Zq, truth, m),
               "rotated": prefix_recall(Zr, Zqr, truth, m)}
              for m in GRANULARITIES]
    recon = [{"m": m,
              "nested": prefix_recon_error(X, V, m),
              "optimum": eckart_young_rankm(X, m),
              "random": prefix_recon_error(X, Vrand, m)}
             for m in GRANULARITIES]
    # Adaptive-retrieval Pareto: fix a cheap prefix, sweep the shortlist size.
    funnel = []
    for sl in (10, 15, 25, 50, 100, 200):
        rec, cost = funnel_retrieval(Z, Zq, truth, FUNNEL_M, shortlist=sl)
        funnel.append({"shortlist": sl, "recall": rec, "cost": cost})
    return {"recall": recall, "recon": recon, "funnel": funnel}


def finance_demo() -> dict:
    """Headline numbers: prefix recall nested vs rotated at the shortlist width, and the
    funnel operating point (shortlist with a short prefix, rerank at full 1536)."""
    X, labels, queries = finance_dataset()
    V, s, mu = pca_basis(X)
    Z, Zq = pca_embedding(X, V, mu), pca_embedding(queries, V, mu)
    R = random_rotation(FINANCE_DIM, seed=7)
    Zr, Zqr = Z @ R, Zq @ R
    truth = full_truth(Z, Zq)
    rec_nested = prefix_recall(Z, Zq, truth, FINANCE_SHORT)
    rec_rot = prefix_recall(Zr, Zqr, truth, FINANCE_SHORT)
    funnel_rec, funnel_cost = funnel_retrieval(Z, Zq, truth, FINANCE_SHORT, shortlist=50)
    out = {
        "dim": FINANCE_DIM, "short": FINANCE_SHORT,
        "recall_nested": rec_nested, "recall_rotated": rec_rot,
        "funnel_recall": funnel_rec, "funnel_cost": funnel_cost,
    }
    print(f"  {FINANCE_N} embeddings in R^{FINANCE_DIM} (SYNTHETIC low-rank-plus-noise, not a trained encoder):")
    print(f"  prefix-{FINANCE_SHORT} recall@10:  nested (Matryoshka/PCA-ordered) {rec_nested*100:.1f}%"
          f"   vs rotated (same geometry, no nesting) {rec_rot*100:.1f}%")
    print(f"  funnel: shortlist on the {FINANCE_SHORT}-dim prefix, rerank at full {FINANCE_DIM} -> "
          f"recall@10 {funnel_rec*100:.1f}% at {funnel_cost*100:.1f}% of exhaustive scoring cost")
    print("  -> nesting is what makes a prefix usable; without it the same vectors retrieve no better than chance early on.")
    return out


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_linear_mrl_equals_pca() -> None:
    """In the linear-reconstruction setting the PCA basis achieves the Eckart-Young
    rank-m optimum at EVERY granularity simultaneously (nestedness), so it minimizes the
    joint Matryoshka loss; a random orthonormal basis cannot match it."""
    X, _ = structured_embeddings(600, 200, 30, n_clusters=3, seed=0)
    V, _, _ = pca_basis(X)
    grans = (4, 8, 16, 32, 64)
    for m in grans:
        nested = prefix_recon_error(X, V, m)
        opt = eckart_young_rankm(X, m)
        assert abs(nested - opt) < 1e-6 * max(1.0, opt), \
            f"PCA prefix-{m} not the rank-{m} optimum: {nested} vs {opt}"
    # a random orthonormal basis is strictly worse on the joint loss
    Q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((200, 200)))
    Vrand = Q[:, :V.shape[1]]
    weights = (1.0,) * len(grans)
    assert joint_mrl_loss(X, V, grans, weights) <= joint_mrl_loss(X, Vrand, grans, weights) + 1e-6, \
        "random basis beat PCA on the joint Matryoshka loss"
    print("  [ok] linear MRL = PCA: one ordered basis hits every rank-m optimum, minimizing the joint loss")


def test_weight_invariance() -> None:
    """The jointly optimal nested basis is PCA for ANY positive granularity weights,
    because PCA attains each per-m optimum — so the joint loss equals the (unimprovable)
    weighted sum of the per-m optima, whatever the weights."""
    X, _ = structured_embeddings(500, 160, 24, n_clusters=3, seed=2)
    V, _, _ = pca_basis(X)
    grans = (8, 16, 32, 64, 128)
    opt_sum = lambda w: sum(wm * eckart_young_rankm(X, m) for m, wm in zip(grans, w))
    rng = np.random.default_rng(3)
    for _ in range(5):
        w = tuple(rng.uniform(0.2, 5.0, len(grans)))
        assert abs(joint_mrl_loss(X, V, grans, w) - opt_sum(w)) < 1e-5 * opt_sum(w), \
            "PCA joint loss != weighted sum of per-m optima — not weight-invariant"
    print("  [ok] weight invariance: PCA is the joint optimum for every positive granularity weighting")


def test_prefix_recall_monotone() -> None:
    """For the nested (PCA-ordered) embedding, prefix-m recall@10 is monotone
    nondecreasing in m and approaches the full-width recall (here, 1)."""
    X, _ = structured_embeddings(800, 256, 32, n_clusters=3, seed=4)
    Q, _ = structured_embeddings(120, 256, 32, n_clusters=3, seed=5)
    V, _, mu = pca_basis(X)
    Z, Zq = pca_embedding(X, V, mu), pca_embedding(Q, V, mu)
    truth = full_truth(Z, Zq)
    grans = (8, 16, 32, 64, 128, 256)
    rec = [prefix_recall(Z, Zq, truth, m) for m in grans]
    for a, b in zip(rec, rec[1:]):
        assert b >= a - 0.02, f"prefix recall not monotone: {a:.3f} -> {b:.3f}"
    assert rec[-1] > 0.99, f"full-width recall should be ~1, got {rec[-1]:.3f}"
    print(f"  [ok] prefix recall monotone in m, -> {rec[-1]*100:.0f}% at full width")


def test_nested_beats_rotated() -> None:
    """Nesting must be TRAINED: a random rotation preserves full-d distances exactly
    (full-width recall unchanged) but its prefixes retrieve far worse than the nested
    embedding's at small m."""
    X, _ = structured_embeddings(800, 256, 32, n_clusters=3, seed=6)
    Q, _ = structured_embeddings(120, 256, 32, n_clusters=3, seed=7)
    V, _, mu = pca_basis(X)
    Z, Zq = pca_embedding(X, V, mu), pca_embedding(Q, V, mu)
    R = random_rotation(256, seed=8)
    Zr, Zqr = Z @ R, Zq @ R
    truth = full_truth(Z, Zq)
    # full-width recall identical (rotation preserves distances)
    assert abs(prefix_recall(Z, Zq, truth, 256) - prefix_recall(Zr, Zqr, truth, 256)) < 1e-9, \
        "rotation changed full-width recall"
    # but the small-prefix recall collapses for the rotated embedding
    assert prefix_recall(Z, Zq, truth, 32) > prefix_recall(Zr, Zqr, truth, 32) + 0.3, \
        "nested prefix did not beat rotated prefix at small m"
    print(f"  [ok] nesting matters: prefix-32 recall nested {prefix_recall(Z,Zq,truth,32)*100:.0f}% "
          f"vs rotated {prefix_recall(Zr,Zqr,truth,32)*100:.0f}% (full width identical)")


def test_funnel_retrieval() -> None:
    """Adaptive retrieval: shortlist on a short prefix, rerank at full width — recall near
    exhaustive full-width search at a fraction of the scoring cost."""
    X, _ = structured_embeddings(1000, 512, 40, n_clusters=3, seed=9)
    Q, _ = structured_embeddings(120, 512, 40, n_clusters=3, seed=10)
    V, _, mu = pca_basis(X)
    Z, Zq = pca_embedding(X, V, mu), pca_embedding(Q, V, mu)
    truth = full_truth(Z, Zq)
    rec, cost = funnel_retrieval(Z, Zq, truth, m_short=64, shortlist=50)
    assert rec > 0.9, f"funnel recall too low: {rec:.3f}"
    assert cost < 0.5, f"funnel should be much cheaper than exhaustive: cost {cost:.3f}"
    print(f"  [ok] funnel: recall@10 {rec*100:.0f}% at {cost*100:.0f}% of exhaustive cost")


def test_sklearn_crosscheck() -> None:
    """Our PCA nested basis matches sklearn.decomposition.PCA (the reference library)."""
    X, _ = structured_embeddings(500, 120, 20, n_clusters=3, seed=11)
    V, s, _ = pca_basis(X)
    ref = SklearnPCA(svd_solver="full").fit(X)
    # compare reconstruction error per m (sign/rotation-invariant)
    for m in (4, 16, 48):
        ours = prefix_recon_error(X, V, m)
        Vr = ref.components_[:m].T
        Xc = X - X.mean(0)
        theirs = float(np.sum((Xc - Xc @ Vr @ Vr.T) ** 2))
        assert abs(ours - theirs) < 1e-6 * max(1.0, theirs), f"sklearn recon disagrees at m={m}"
    print("  [ok] cross-check: nested basis reconstruction matches sklearn.decomposition.PCA")


def test_finance_funnel() -> None:
    """The 1536-d finance cloud: nesting makes the short prefix usable, and the funnel
    retrieves near full-width recall at a fraction of the cost."""
    f = finance_demo()
    assert f["recall_nested"] > f["recall_rotated"] + 0.3, "nested prefix should dominate rotated"
    assert f["funnel_recall"] > 0.9, f"funnel recall too low: {f['funnel_recall']:.3f}"
    assert f["funnel_cost"] < 0.4, f"funnel should be cheap: {f['funnel_cost']:.3f}"
    print("  [ok] finance: nested prefix >> rotated, funnel keeps recall at a fraction of cost")


if __name__ == "__main__":
    print("Matryoshka / nested-subspace verification harness")
    test_linear_mrl_equals_pca()
    test_weight_invariance()
    test_prefix_recall_monotone()
    test_nested_beats_rotated()
    test_funnel_retrieval()
    test_sklearn_crosscheck()
    tbl = grid_table()
    print("Grid table (mirrored by MatryoshkaLaboratory.tsx):")
    print(f"  {'m':>6}{'nested_recall':>15}{'rotated_recall':>16}{'nested_recon':>14}{'optimum':>12}")
    for rrow, crow in zip(tbl["recall"], tbl["recon"]):
        print(f"  {rrow['m']:>6}{rrow['nested']:>15.4f}{rrow['rotated']:>16.4f}"
              f"{crow['nested']:>14.1f}{crow['optimum']:>12.1f}")
    print(f"  funnel Pareto (prefix {FUNNEL_M}, shortlist -> recall @ cost):",
          ", ".join(f"{r['shortlist']}->{r['recall']*100:.0f}%@{r['cost']*100:.1f}%" for r in tbl["funnel"]))
    print("Finance demo:")
    test_finance_funnel()
    print("All checks passed.")
