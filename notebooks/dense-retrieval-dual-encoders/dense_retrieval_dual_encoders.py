"""Dense retrieval and dual encoders — the reference implementation for the formalRAG
`dense-retrieval-dual-encoders` topic.

InfoNCE told us how a dual encoder is TRAINED. This module asks what that architecture can
REPRESENT and what it COSTS — three movements, every pedagogical claim an `assert`:

  MOVEMENT 1 — SEPARABILITY => PRECOMPUTABILITY => MIPS. A dual encoder scores a query q and a
    document d by the SEPARABLE inner product s(q, d) = <E_Q(q), E_P(d)>. Stack the corpus as
    P (one row per document); then for every query, the top-k by score is exactly the top-k of
    the single matrix-vector product P @ E_Q(q) — maximum-inner-product search over a matrix
    that does NOT depend on q and is therefore precomputable offline. Retrieval at serving time
    is one query encode plus one MIPS. This is what a cross-encoder (which fuses [q; d]) gives
    up. (`dual_encoder_score`, `precompute_passages`, `mips_retrieve`.)

  MOVEMENT 2 — THE RANK-d EXPRESSIVITY CEILING. Stack the scores into a query-by-document
    matrix S = Q G^T with inner (embedding) dimension d. Then rank(S) <= d, and conversely a
    target relevance matrix M is EXACTLY realizable by a d-dimensional dual encoder iff
    rank(M) <= d (the thin-SVD construction). When rank(M) > d the best realizable approximation
    in Frobenius norm is the truncated SVD — the Eckart-Young-Mirsky theorem, which we CITE
    rather than reprove. Retrieval accuracy collapses when d falls below the relevance pattern's
    intrinsic rank and saturates above it. (`score_matrix`, `best_rank_d`, `realize_dual_encoder`,
    `topk_recall`, `dpr_finance_matrix`, `cross_encoder_oracle`.)

  MOVEMENT 3 — IN-BATCH NEGATIVES = THE B^2-FROM-2B GRAM TRICK. For a batch of B (query,
    positive) pairs, stack Q, G in R^{B x d} and form the Gram matrix S = Q G^T in R^{B x B}:
    row i is an (N+1 = B)-way InfoNCE problem with the positive on the diagonal and the other
    B-1 in-batch documents as negatives. So 2B encoder forward passes (B queries + B documents)
    yield B(B-1) = Theta(B^2) negative comparisons at no extra encoding cost. The loss itself is
    IMPORTED from the InfoNCE topic, never reimplemented; we re-derive it as a row-wise
    logsumexp-minus-diagonal mean and assert byte-for-byte agreement. (`inbatch_gram`,
    `inbatch_loss_via_gram`, `symmetric_inbatch_loss`, `negative_pair_count`.)

Honest caveats (rigorFlag territory): the rank-d theorem is an UPPER bound on the dimension that
SUFFICES, not the tight sign-rank / margin-complexity measure of the dimension relevance NEEDS
(deferred to the embedding-dimension topic); the in-batch negatives are SHARED across the batch
(correlated, not the i.i.d. draws the InfoNCE bound assumes) and include FALSE negatives; the
separability => MIPS reduction is a property of the FACTORIZATION, exactly what a cross-encoder
forgoes; and the encoder here is a deterministic SYNTHETIC von Mises-Fisher / projected-GD
stand-in reused from the InfoNCE topic, not a trained transformer.

This module imports its prerequisites (`hypersphere-vmf-geometry`, `the-retrieval-problem`,
`infonce-contrastive-objective`) for the sphere sampler, the cosine score and corpus ranking,
and the InfoNCE loss + finance vMF geometry — it never reimplements them. `viz_constants()`
prints what `DenseDualEncoderLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/dense-retrieval-dual-encoders/dense_retrieval_dual_encoders.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.special import logsumexp

# Established cross-topic pattern: add EACH ancestor's HYPHENATED dir to the path, import the
# UNDERSCORED module. The dual encoder is trained by the InfoNCE topic (its loss + finance vMF
# geometry), which itself builds on the sphere sampler of hypersphere-vMF and the cosine/rank
# primitives of the-retrieval-problem; InfoNCE re-exports some but not all of those, so we add
# every ancestor explicitly and never reimplement any of them.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("hypersphere-vmf-geometry", "the-retrieval-problem", "infonce-contrastive-objective"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf            # noqa: E402
from the_retrieval_problem import dot, cosine, rank                   # noqa: E402
from infonce_contrastive_objective import (                           # noqa: E402
    info_nce_loss,
    info_nce_loss_batch,
    finance_dataset,
)


# --------------------------------------------------------------------------- #
# Movement 1 — separability => precomputability => MIPS.
# --------------------------------------------------------------------------- #

def dual_encoder_score(z_q: np.ndarray, Z_p: np.ndarray) -> np.ndarray:
    """The separable dual-encoder scores of one query against a corpus matrix P: the row vector
    [<z_q, P_j>]_j = P @ z_q. The query and each document interact through a SINGLE inner
    product — the property the whole movement turns on."""
    Z_p = np.atleast_2d(Z_p)
    return Z_p @ np.asarray(z_q, dtype=float)


def precompute_passages(corpus: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Stack a corpus {id: E_P(d)} into the offline document matrix P (one row per document) and
    the parallel id list. This is the work a dual encoder does ONCE, before any query arrives.
    GUARD: an empty corpus yields ([], an empty matrix)."""
    ids = list(corpus.keys())
    if not ids:
        return [], np.empty((0, 0))
    P = np.array([corpus[i] for i in ids], dtype=float)
    return ids, P


def mips_retrieve(z_q: np.ndarray, ids: list[str], P: np.ndarray, k: int) -> list[str]:
    """Top-k retrieval as maximum-inner-product search over the precomputed matrix P: one
    matrix-vector product P @ z_q, then the k largest scores. GUARDS: empty corpus -> []; k is
    capped at the corpus size. Ties are broken by the argsort, matching `rank`'s stable order."""
    n = len(ids)
    if n == 0:
        return []
    k = min(k, n)
    s = dual_encoder_score(z_q, P)
    order = np.argsort(-s, kind="stable")[:k]
    return [ids[int(j)] for j in order]


# --------------------------------------------------------------------------- #
# Movement 2 — the rank-d expressivity ceiling (Eckart-Young-Mirsky cited).
# --------------------------------------------------------------------------- #

def score_matrix(Q: np.ndarray, G: np.ndarray) -> np.ndarray:
    """The realizable query-by-document score matrix S = Q G^T of a dual encoder with query
    embeddings Q (|Q| x d) and document embeddings G (|C| x d). Its rank is at most d."""
    return np.atleast_2d(Q) @ np.atleast_2d(G).T


def best_rank_d(S: np.ndarray, d: int) -> np.ndarray:
    """The best rank-d approximation of S in Frobenius (and spectral) norm: the truncated SVD
    S_d = sum_{i<d} sigma_i u_i v_i^T (Eckart-Young-Mirsky). GUARDS: empty S -> empty; d is
    clamped to [1, min(S.shape)]."""
    S = np.asarray(S, dtype=float)
    if S.size == 0:
        return np.empty_like(S)
    d = max(1, min(int(d), min(S.shape)))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    return (U[:, :d] * s[:d]) @ Vt[:d]


def realize_dual_encoder(M: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor a target matrix M into d-dimensional dual-encoder embeddings (Q, G) with
    Q G^T = best_rank_d(M, d): from the thin SVD M = U S V^T, set Q = U_d sqrt(S_d) and
    G = V_d sqrt(S_d). When rank(M) <= d this reconstructs M EXACTLY (the realizability
    direction); when rank(M) > d it returns the optimal rank-d surrogate. GUARDS: empty M ->
    empty (Q, G); d clamped."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        dd = max(1, int(d))
        return np.zeros((M.shape[0], dd)), np.zeros((M.shape[1], dd))
    d = max(1, min(int(d), min(M.shape)))
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    root = np.sqrt(s[:d])
    Q = U[:, :d] * root
    G = (Vt[:d].T) * root
    return Q, G


def relative_frobenius_error(S: np.ndarray, S_hat: np.ndarray) -> float:
    """||S - S_hat||_F / ||S||_F, guarded against the all-zero matrix."""
    S = np.asarray(S, dtype=float)
    denom = float(np.linalg.norm(S))
    return float(np.linalg.norm(S - S_hat) / max(denom, 1e-12))


def cross_encoder_oracle(M: np.ndarray) -> np.ndarray:
    """A stand-in for a cross-encoder h([q; d]): a scorer that can realize ANY target score
    matrix M, with no rank constraint — it just returns M. The contrast to a dual encoder, whose
    realizable matrices are exactly those of rank <= d. (The cross-encoder topic develops the
    real architecture; here it is only the rank-free counterpoint.)"""
    return np.asarray(M, dtype=float)


def topk_recall(S: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Recall@k of a score matrix: the fraction of query rows whose relevant document
    truth[i] appears among the row's k highest scores. GUARDS: no queries -> 0.0; k <= 0 -> 0.0;
    k capped at the number of documents."""
    S = np.atleast_2d(S)
    nq, nd = S.shape
    if nq == 0 or nd == 0 or k <= 0:
        return 0.0
    k = min(int(k), nd)
    truth = np.asarray(truth)
    # argpartition for the top-k of each row (descending via negation).
    top = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    hits = sum(1 for i in range(nq) if truth[i] in top[i])
    return hits / nq


# --------------------------------------------------------------------------- #
# Movement 3 — in-batch negatives = the B^2-from-2B Gram trick.
# --------------------------------------------------------------------------- #

def inbatch_gram(Q: np.ndarray, G: np.ndarray, tau: float) -> np.ndarray:
    """The B x B in-batch logit matrix S = (Q G^T)/tau: row i is query i scored against every
    in-batch document, with its positive on the diagonal. Inputs are L2-normalized defensively
    so the logits are temperature-scaled cosines. GUARD: tau > 0."""
    if not (tau > 1e-8):
        raise ValueError(f"temperature tau must be > 1e-8, got {tau}")
    Q = normalize(np.atleast_2d(Q))
    G = normalize(np.atleast_2d(G))
    return (Q @ G.T) / tau


def inbatch_loss_via_gram(Q: np.ndarray, G: np.ndarray, tau: float) -> float:
    """The in-batch InfoNCE loss re-derived from the Gram matrix as the row-wise
    logsumexp(S) - diag(S), averaged over rows. This MUST equal the imported
    `info_nce_loss_batch` (the byte-for-byte reuse anchor); we never reimplement the loss as a
    competing definition, only re-read the Gram matrix that the architecture produces."""
    S = inbatch_gram(Q, G, tau)
    return float(np.mean(logsumexp(S, axis=1) - np.diag(S)))


def symmetric_inbatch_loss(Q: np.ndarray, G: np.ndarray, tau: float) -> float:
    """The symmetric in-batch loss 1/2 (L(q->d) + L(d->q)): query-to-document plus the
    document-to-query direction (the same loss on the transposed Gram matrix). Both directions
    are the imported InfoNCE batch loss."""
    return 0.5 * (info_nce_loss_batch(Q, G, tau) + info_nce_loss_batch(G, Q, tau))


def negative_pair_count(B: int) -> tuple[int, int]:
    """The cost law: a batch of B (query, document) pairs costs 2B encoder forward passes and
    yields B(B-1) in-batch negative comparisons. Returns (encoder_passes, negative_pairs).
    GUARD: B >= 2 (an in-batch negative needs at least two rows)."""
    if B < 2:
        raise ValueError(f"in-batch negatives need B >= 2, got {B}")
    return 2 * B, B * (B - 1)


# --------------------------------------------------------------------------- #
# Module constants the viz panels step through.
# --------------------------------------------------------------------------- #

# Finance dual encoder (synthetic vMF stand-in, reused from the InfoNCE topic's geometry):
# each SECTOR is a vMF cluster, each COMPANY a tight vMF sub-cluster, and the corpus holds ONE
# document per company so the document matrix P is exactly (#sectors x #companies) directions,
# forcing the score matrix's intrinsic rank to the company count. Queries are tight draws around
# their own company; the relevant document is that company's. The within-sector concentration
# kappa_sector keeps same-sector companies genuinely near (cosine ~ 0.4 at d = 32 — the
# CLAUDE.md vMF-sizing rule), so separating companies WITHIN a sector needs nearly all of the
# rank and the rank ceiling becomes visible.
DPR_DIM = 32
DPR_N_SECTORS = 4
DPR_N_COMP = 2                    # companies per sector -> 8 documents = the intrinsic rank
DPR_QPC = 4                       # queries per company -> 32 queries
DPR_KAPPA_SECTOR = 60.0
DPR_KAPPA_COMPANY = 350.0
DPR_SEED = 7
DPR_TAU = 0.05                    # temperature for the in-batch loss demo

CORPUS_GRID = (10, 100, 1000, 10000, 100000)   # Panel A: query-time cost vs corpus size
BATCH_GRID = (2, 4, 8, 16, 64)                 # Panel C: the counting law
# A realistic headline scale for the cost panel's labels (Wikipedia-scale passage index).
CORPUS_HEADLINE = 21_000_000
EMB_DIM_HEADLINE = 768


def dpr_finance_matrix(seed: int = DPR_SEED, dim: int = DPR_DIM, n_sectors: int = DPR_N_SECTORS,
                       n_comp: int = DPR_N_COMP, queries_per_company: int = DPR_QPC,
                       kappa_sector: float = DPR_KAPPA_SECTOR,
                       kappa_company: float = DPR_KAPPA_COMPANY):
    """Build the synthetic finance retrieval instance from the InfoNCE sectors->companies vMF
    geometry. Returns (Q, P, truth, sector_of_passage):
      Q     : (n_queries x dim) query embeddings, n_queries = n_sectors*n_comp*queries_per_company
      P     : (n_passages x dim) document embeddings, ONE per company (n_passages = n_sectors*n_comp)
      truth : (n_queries,) the index into P of each query's relevant company document
      sector_of_passage : (n_passages,) the sector of each document (for the block heatmap).
    The score matrix Q P^T has intrinsic rank n_passages (distinct company directions), so it is
    the object whose rank ceiling Movement 2 measures."""
    rng = np.random.default_rng(seed)
    sector_mu = normalize(rng.standard_normal((n_sectors, dim)))
    passages, sector_of_passage = [], []
    queries, truth = [], []
    for si, mu in enumerate(sector_mu):
        members = sample_vmf(n_comp, mu, kappa_sector, seed=seed + 11 * si + 1)
        for ci in range(n_comp):
            c_mu = normalize(members[ci])
            p_idx = len(passages)
            passages.append(c_mu)
            sector_of_passage.append(si)
            qs = sample_vmf(queries_per_company, c_mu, kappa_company,
                            seed=seed + 211 + 7 * si + ci)
            for q in np.atleast_2d(qs):
                queries.append(normalize(q))
                truth.append(p_idx)
    return (np.array(queries), np.array(passages),
            np.array(truth), np.array(sector_of_passage))


def rank_recall_curve(S: np.ndarray, truth: np.ndarray, dims=None):
    """For each embedding dimension d, the recall@1 / recall@3 of the BEST rank-d dual encoder
    (the truncated-SVD reconstruction of S) and its relative Frobenius reconstruction error.
    Returns a list of {d, r1, r3, recon_err}. This is the rank ceiling made measurable."""
    S = np.atleast_2d(S)
    r = min(S.shape)
    if dims is None:
        dims = range(1, r + 1)
    rows = []
    for d in dims:
        Sd = best_rank_d(S, d)
        rows.append({
            "d": int(d),
            "r1": round(topk_recall(Sd, truth, 1), 4),
            "r3": round(topk_recall(Sd, truth, 3), 4),
            "recon_err": round(relative_frobenius_error(S, Sd), 4),
        })
    return rows


def first_full_recall_dim(curve) -> int:
    """The smallest embedding dimension d at which recall@1 reaches 1.0 — the dimension that
    'recovers' the relevance pattern. Returns the last d if it never reaches 1.0."""
    for row in curve:
        if row["r1"] >= 1.0:
            return row["d"]
    return curve[-1]["d"]


def dpr_inbatch_example(seed: int = DPR_SEED):
    """A batch of B = n_passages (query, document) pairs drawn from the finance geometry: query i
    is the first query of company i, its positive is company i's document. Returns (Qb, Gb), the
    rows of the B x B Gram matrix the viz displays (diagonal = positives, off-diagonal = in-batch
    negatives, with same-sector pairs the hardest)."""
    Q, P, truth, _ = dpr_finance_matrix(seed=seed)
    B = P.shape[0]
    # The first query belonging to each company (companies appear in passage order).
    Qb = np.array([Q[np.argmax(truth == j)] for j in range(B)])
    return Qb, P


# --------------------------------------------------------------------------- #
# Finance demo — the headline numbers, printed.
# --------------------------------------------------------------------------- #

def finance_demo() -> dict:
    """The finance headline: a separable dual encoder turns retrieval into one MIPS lookup over a
    precomputed matrix; that matrix has an intrinsic rank set by the company count; and a batch of
    B pairs buys B(B-1) negatives from 2B encodings."""
    Q, P, truth, sec = dpr_finance_matrix()
    S = score_matrix(Q, P)
    rank_S = int(np.linalg.matrix_rank(S))
    n_q, n_p = S.shape
    print(f"  finance dual encoder: {n_q} queries, {n_p} documents (one per company), "
          f"dim = {DPR_DIM}; score matrix S = Q P^T (synthetic vMF, not a trained encoder)")
    print(f"  intrinsic rank(S) = {rank_S} = #companies "
          f"({DPR_N_SECTORS} sectors x {DPR_N_COMP} companies)")

    # Movement 1: serving one query is one MIPS over the precomputed P.
    corpus = {f"doc-{j}": P[j] for j in range(n_p)}
    ids, P_pre = precompute_passages(corpus)
    z_q = Q[0]
    top1_mips = mips_retrieve(z_q, ids, P_pre, k=1)[0]
    top1_scan = rank(z_q, corpus, dot, descending=True)[0]
    print(f"  MOVEMENT 1: query 0 -> MIPS top-1 = {top1_mips} = full-scan top-1 = {top1_scan} "
          f"(= its company doc-{truth[0]})")

    # Movement 2: the rank ceiling.
    curve = rank_recall_curve(S, truth)
    d_rec = first_full_recall_dim(curve)
    print(f"  MOVEMENT 2: recall@1 by embedding dim d = "
          f"{[(r['d'], r['r1']) for r in curve]}")
    print(f"             recovers full recall@1 at d = {d_rec}; "
          f"exact reconstruction at d = rank(S) = {curve[-1]['d']} "
          f"(recon_err = {curve[-1]['recon_err']})")

    # Movement 3: the counting law.
    print(f"  MOVEMENT 3: in-batch negatives from 2B encodings:")
    out = {"rank_S": rank_S, "d_recover": d_rec}
    for B in BATCH_GRID:
        passes, negs = negative_pair_count(B)
        out[f"negs_B{B}"] = negs
        print(f"             B = {B:>3}: {passes:>4} encoder passes -> {negs:>5} negative pairs")
    Qb, Gb = dpr_inbatch_example()
    out["inbatch_loss"] = round(inbatch_loss_via_gram(Qb, Gb, DPR_TAU), 4)
    print(f"             in-batch InfoNCE loss on the B = {Gb.shape[0]} finance batch "
          f"(tau = {DPR_TAU}) = {out['inbatch_loss']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_mips_reduction_equivalence() -> None:
    """Movement 1: the separable score makes top-k retrieval exactly MIPS over the precomputed
    document matrix. The MIPS top-k equals the full-scan ranking equals the cosine ranking on
    the (unit) finance documents, and query 0's top-1 is its own company's document."""
    Q, P, truth, _ = dpr_finance_matrix()
    n_p = P.shape[0]
    corpus = {f"doc-{j}": P[j] for j in range(n_p)}
    ids, P_pre = precompute_passages(corpus)
    for qi in (0, 1, 17, 31):
        z_q = Q[qi]
        mips = mips_retrieve(z_q, ids, P_pre, k=n_p)
        scan = rank(z_q, corpus, dot, descending=True)
        assert mips == scan, f"MIPS order != full-scan order for query {qi}: {mips} vs {scan}"
        # On unit vectors the dot and cosine rankings coincide (the retrieval-problem keystone).
        cos_order = rank(z_q, corpus, cosine, descending=True)
        assert mips == cos_order, f"MIPS order != cosine order for query {qi}"
        assert mips[0] == f"doc-{truth[qi]}", \
            f"query {qi} top-1 {mips[0]} != its company doc-{truth[qi]}"
    # The precomputed matrix does not depend on the query: one matvec answers any query.
    s_matvec = dual_encoder_score(Q[0], P_pre)
    s_loop = np.array([dot(Q[0], P_pre[j]) for j in range(n_p)])
    assert np.allclose(s_matvec, s_loop), "P @ z_q must equal the per-document inner products"
    print(f"  [ok] Movement 1: retrieval = MIPS over the precomputed matrix "
          f"(MIPS == full-scan == cosine order; top-1 is the query's company)")


def test_rank_d_realizability() -> None:
    """Movement 2 (both directions): a product Q G^T with inner dimension d has rank <= d; and a
    target of rank r is realized EXACTLY by a d-dimensional dual encoder iff d >= r, with a
    strictly positive error for d < r."""
    rng = np.random.default_rng(0)
    # (=>) rank(Q G^T) <= d.
    for d in (1, 3, 8):
        Q = rng.standard_normal((20, d))
        G = rng.standard_normal((12, d))
        assert np.linalg.matrix_rank(score_matrix(Q, G)) <= d, f"rank(Q G^T) exceeded d={d}"
    # (<=) a rank-r target: exact iff d >= r, strictly positive error below.
    r = 5
    A = rng.standard_normal((20, r))
    Bm = rng.standard_normal((12, r))
    M = A @ Bm.T                                   # rank exactly r (generically)
    assert np.linalg.matrix_rank(M) == r, "test setup: target should have rank r"
    for d in range(1, 10):
        Qd, Gd = realize_dual_encoder(M, d)
        err = relative_frobenius_error(M, Qd @ Gd.T)
        if d >= r:
            assert err < 1e-9, f"rank-{r} target not realized exactly at d={d}: err={err}"
        else:
            assert err > 1e-6, f"rank-{r} target should be unrealizable at d={d}<r: err={err}"
    print(f"  [ok] Movement 2: rank(Q G^T) <= d, and a rank-{r} target is realized exactly "
          f"iff d >= {r}")


def test_eckart_young_crosscheck() -> None:
    """Movement 2: `best_rank_d` is the truncated SVD, cross-checked against numpy AND scipy, and
    its Frobenius error equals the tail-energy sqrt(sum_{i>=d} sigma_i^2) (Eckart-Young-Mirsky)."""
    rng = np.random.default_rng(1)
    S = rng.standard_normal((15, 9))
    s_np = np.linalg.svd(S, compute_uv=False)
    s_sp = scipy_svd(S, compute_uv=False)
    assert np.allclose(s_np, s_sp), "numpy and scipy singular values disagree"
    for d in range(1, 9):
        Sd = best_rank_d(S, d)
        assert np.linalg.matrix_rank(Sd) <= d, f"best_rank_d returned rank > {d}"
        err = float(np.linalg.norm(S - Sd))
        tail = float(np.sqrt(np.sum(s_np[d:] ** 2)))
        assert abs(err - tail) < 1e-9, f"Frobenius error {err} != tail energy {tail} at d={d}"
    print("  [ok] Movement 2: best_rank_d = truncated SVD (numpy == scipy); "
          "Frobenius error = sqrt(sum of dropped singular values^2)")


def test_rank_ceiling_recall() -> None:
    """Movement 2, the demonstrable phenomenon: on the finance score matrix the rank is the
    company count; recall@1 of the best rank-d dual encoder is non-decreasing in d, collapses
    below full at the smallest d, recovers to 1.0 at a dimension below full rank, and the
    reconstruction is exact at d = rank. Pinned to the observed run."""
    Q, P, truth, _ = dpr_finance_matrix()
    S = score_matrix(Q, P)
    n_p = P.shape[0]
    assert int(np.linalg.matrix_rank(S)) == n_p, "score-matrix rank should equal the company count"
    curve = rank_recall_curve(S, truth)
    r1 = [row["r1"] for row in curve]
    # recall@1 is non-decreasing in d (a richer dual encoder never retrieves worse here).
    assert all(r1[i] <= r1[i + 1] + 1e-9 for i in range(len(r1) - 1)), \
        f"recall@1 not monotone in d: {r1}"
    # The ceiling is real: the smallest d cannot resolve companies, full rank can.
    assert r1[0] < 0.5, f"smallest d should collapse recall@1, got {r1[0]}"
    d_rec = first_full_recall_dim(curve)
    assert d_rec < n_p, f"recall@1 should recover BELOW full rank, got d_recover={d_rec}=rank"
    assert curve[-1]["r1"] >= 1.0, f"full-rank recall@1 should be 1.0, got {curve[-1]['r1']}"
    # Exact reconstruction at d = rank.
    assert curve[-1]["recon_err"] < 1e-9, \
        f"reconstruction should be exact at d=rank, err={curve[-1]['recon_err']}"
    # The full-dimension dual encoder (the raw 32-d embeddings) also gets every query right.
    assert topk_recall(S, truth, 1) >= 1.0, "full-dimension recall@1 should be 1.0"
    print(f"  [ok] Movement 2: rank ceiling — recall@1 by d = {r1}; "
          f"recovers at d = {d_rec} < rank = {n_p}; exact reconstruction at d = rank")


def test_inbatch_equals_imported_infonce() -> None:
    """Movement 3, the byte-for-byte reuse anchor: the Gram re-derivation of the in-batch loss
    equals the IMPORTED `info_nce_loss_batch` (we reuse InfoNCE's loss, never reimplement it),
    and also equals the mean of the imported single-positive losses."""
    rng = np.random.default_rng(2)
    d, B, tau = 16, 7, 0.1
    Q = normalize(rng.standard_normal((B, d)))
    G = normalize(rng.standard_normal((B, d)))
    via_gram = inbatch_loss_via_gram(Q, G, tau)
    imported = info_nce_loss_batch(Q, G, tau)
    assert abs(via_gram - imported) < 1e-12, f"Gram loss {via_gram} != imported {imported}"
    # And the imported single-positive loss, row by row (each row's positive on the diagonal).
    manual = float(np.mean([
        info_nce_loss(Q[i], G[i], np.delete(G, i, axis=0), tau) for i in range(B)
    ]))
    assert abs(via_gram - manual) < 1e-10, f"Gram loss {via_gram} != single-positive mean {manual}"
    print(f"  [ok] Movement 3: in-batch loss via Gram == imported info_nce_loss_batch "
          f"== mean single-positive loss ({via_gram:.4f})")


def test_counting_law() -> None:
    """Movement 3: a batch of B pairs costs 2B encoder passes and yields B(B-1) negative
    comparisons — quadratic negatives from linear encoding. The negatives also equal the count
    of off-diagonal entries of the B x B Gram matrix."""
    for B in BATCH_GRID:
        passes, negs = negative_pair_count(B)
        assert passes == 2 * B, f"encoder passes should be 2B, got {passes} for B={B}"
        assert negs == B * (B - 1), f"negatives should be B(B-1), got {negs} for B={B}"
        # Cross-check against the actual off-diagonal count of a Gram matrix.
        S = np.zeros((B, B))
        off_diagonal = S.size - np.trace(np.ones((B, B)))
        assert negs == int(off_diagonal), "B(B-1) should be the off-diagonal entry count"
    # B < 2 has no in-batch negative.
    try:
        negative_pair_count(1)
        assert False, "B=1 should raise"
    except ValueError:
        pass
    print(f"  [ok] Movement 3: counting law — B pairs -> 2B passes, B(B-1) negatives "
          f"(B in {BATCH_GRID})")


def test_symmetric_loss() -> None:
    """Movement 3: the symmetric loss averages the query->document and document->query
    directions, the latter being the imported batch loss on the transposed Gram matrix."""
    rng = np.random.default_rng(3)
    d, B, tau = 12, 6, 0.15
    Q = normalize(rng.standard_normal((B, d)))
    G = normalize(rng.standard_normal((B, d)))
    sym = symmetric_inbatch_loss(Q, G, tau)
    l_qd = info_nce_loss_batch(Q, G, tau)
    l_dq = info_nce_loss_batch(G, Q, tau)
    assert abs(sym - 0.5 * (l_qd + l_dq)) < 1e-12, "symmetric loss != mean of the two directions"
    # The d->q loss is the q->d row-loss on the transposed Gram matrix.
    via_gram_t = float(np.mean(logsumexp(inbatch_gram(G, Q, tau), axis=1)
                               - np.diag(inbatch_gram(G, Q, tau))))
    assert abs(l_dq - via_gram_t) < 1e-12, "d->q loss != transposed-Gram row loss"
    print(f"  [ok] Movement 3: symmetric loss = 1/2 (L(q->d) + L(d->q)) "
          f"= 1/2 ({l_qd:.4f} + {l_dq:.4f}) = {sym:.4f}")


def test_cross_encoder_has_no_rank_constraint() -> None:
    """Movements 1-2 contrast: a full-rank target relevance matrix is unrealizable by any dual
    encoder below its rank (the best rank-d surrogate has strictly positive error), but a
    cross-encoder — which fuses the pair and is not a factorization — realizes it exactly."""
    rng = np.random.default_rng(4)
    M = rng.standard_normal((6, 6))
    r = np.linalg.matrix_rank(M)
    assert r == 6, "test setup: target should be full rank 6"
    for d in range(1, r):
        Qd, Gd = realize_dual_encoder(M, d)
        assert relative_frobenius_error(M, Qd @ Gd.T) > 1e-6, \
            f"dual encoder should NOT realize a full-rank target at d={d}<{r}"
    assert relative_frobenius_error(M, cross_encoder_oracle(M)) < 1e-12, \
        "the cross-encoder oracle should realize any target exactly"
    print(f"  [ok] contrast: a full-rank (rank {r}) target is unrealizable by a dual encoder "
          f"with d < {r}, but exact for the cross-encoder (no rank constraint)")


def test_finance_dataset_is_reused() -> None:
    """The reuse contract: the InfoNCE finance vMF dataset is imported and usable (one focal
    query with same-sector hard negatives), the same geometry the dual-encoder matrix is built
    from — we never rebuild the contrastive geometry from scratch."""
    data = finance_dataset()
    assert {"z_q", "z_pos", "neg_docs", "same_sector"} <= set(data), "InfoNCE dataset shape changed"
    assert data["same_sector"].any(), "expected same-sector hard negatives in the imported dataset"
    cos_pos = float(data["z_pos"] @ data["z_q"])
    cos_hard = float((data["neg_docs"] @ data["z_q"])[data["same_sector"]].max())
    assert cos_pos > cos_hard, "the positive should be nearer than the hardest same-sector negative"
    print(f"  [ok] reuse: imported InfoNCE finance geometry (positive cosine {cos_pos:.3f} > "
          f"hardest same-sector {cos_hard:.3f})")


# --------------------------------------------------------------------------- #
# Viz constants — printed for DenseDualEncoderLaboratory.tsx to mirror.
# --------------------------------------------------------------------------- #

def _company_display_matrix(seed: int = DPR_SEED):
    """An n_passages x n_passages display heatmap: row j is the FIRST query of company j scored
    against every company document, so the diagonal is the positive (own company) and the block
    structure (same-sector companies near, cross-sector far) is legible. Returns (M, sector)."""
    Qb, P = dpr_inbatch_example(seed)
    M = Qb @ P.T
    _, _, _, sector = dpr_finance_matrix(seed=seed)
    return M, sector


def viz_constants() -> None:
    """Print the cost model (Panel A), the rank ceiling (Panel B), and the in-batch Gram trick
    (Panel C) — all baked to the decimal in the .tsx. TS recomputes only CLOSED FORM: the per-
    query cost arithmetic, B(B-1) and 2B, the heatmap color scale, and the truncated-SVD matmul
    from the baked (U, s, Vt). Every MEASURED number (singular values, recall-by-dim, the score
    and Gram matrices) is baked here."""
    Q, P, truth, sec = dpr_finance_matrix()
    S = score_matrix(Q, P)
    n_q, n_p = S.shape

    print("  PANEL A — query-time cost vs corpus size (closed-form arithmetic, grid baked):")
    print(f"    CORPUS_HEADLINE = {CORPUS_HEADLINE}, EMB_DIM_HEADLINE = {EMB_DIM_HEADLINE}")
    cost = [{"corpus": int(c), "bi_passes": 1, "cross_passes": int(c)} for c in CORPUS_GRID]
    print(f"    COST_VS_CORPUS = {cost}")

    print("  PANEL B — the rank ceiling of the finance score matrix:")
    print(f"    DPR_DIM = {DPR_DIM}, DPR_N_QUERIES = {int(n_q)}, DPR_N_PASSAGES = {int(n_p)}")
    print(f"    INTRINSIC_RANK = {int(np.linalg.matrix_rank(S))}")
    sv = np.linalg.svd(S, compute_uv=False)
    print(f"    SINGULAR_VALUES = {[round(float(v), 4) for v in sv]}")
    curve = rank_recall_curve(S, truth)
    print(f"    RANK_RECALL = {curve}")
    print(f"    D_RECOVER = {first_full_recall_dim(curve)}")
    # The display heatmap (n_p x n_p) and its SVD, so TS can reconstruct at any d (closed form).
    M, sector = _company_display_matrix()
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    print(f"    DISPLAY_MATRIX = {np.round(M, 4).tolist()}")
    print(f"    DISPLAY_SECTOR = {[int(x) for x in sector]}")
    print(f"    DISPLAY_U = {np.round(U, 4).tolist()}")
    print(f"    DISPLAY_S = {[round(float(v), 4) for v in s]}")
    print(f"    DISPLAY_VT = {np.round(Vt, 4).tolist()}")

    print("  PANEL C — the in-batch Gram trick (counts are closed-form in TS):")
    counts = [{"B": int(B), "passes": 2 * int(B), "negatives": int(B) * (int(B) - 1)}
              for B in BATCH_GRID]
    print(f"    COUNTING = {counts}")
    Qb, Gb = dpr_inbatch_example()
    print(f"    GRAM_TAU = {DPR_TAU}")
    print(f"    GRAM_DISPLAY = {np.round(Qb @ Gb.T, 4).tolist()}   # cosines (pre-temperature)")
    print(f"    GRAM_SECTOR = {[int(x) for x in sector]}")     # same company order as DISPLAY_SECTOR
    print(f"    INBATCH_LOSS = {round(inbatch_loss_via_gram(Qb, Gb, DPR_TAU), 4)}")


if __name__ == "__main__":
    print("Dense retrieval & dual encoders — verification harness")
    test_rank_ceiling_recall()                     # the headline runs first
    test_mips_reduction_equivalence()
    test_rank_d_realizability()
    test_eckart_young_crosscheck()
    test_inbatch_equals_imported_infonce()
    test_counting_law()
    test_symmetric_loss()
    test_cross_encoder_has_no_rank_constraint()
    test_finance_dataset_is_reused()
    print("Finance demo:")
    finance_demo()
    print("Viz constants (mirrored to the decimal in DenseDualEncoderLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
