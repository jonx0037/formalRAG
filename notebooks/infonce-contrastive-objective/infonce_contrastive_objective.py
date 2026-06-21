"""Contrastive learning for retrieval — the reference implementation for the formalRAG
`infonce-contrastive-objective` topic.

The retrieval-problem defined relevance as a similarity functional and hypersphere-vMF
told us dense embeddings live on the unit sphere S^{d-1}; neither said where those
embeddings come from. They are TRAINED. InfoNCE is the loss that teaches a dual encoder
to place a query near its relevant document and far from everything else. This module
reads one loss three ways, and every pedagogical claim below is an `assert`:

  MOVEMENT 1 — THE OBJECTIVE. With a query q, its positive d+, and N negatives, all
    L2-normalized so the score s = <q, d>/tau is a temperature-scaled cosine, InfoNCE is
    the (N+1)-way cross-entropy that must pick the positive:
        L = -E[ log( e^{s+} / (e^{s+} + sum_i e^{s_i^-}) ) ].
    (`scores`, `softmax_temperature`, `info_nce_loss`, `info_nce_loss_batch`.)

  MOVEMENT 2 — A MUTUAL-INFORMATION LOWER BOUND (CPC; van den Oord, Li, Vinyals 2018).
    Minimizing InfoNCE maximizes a lower bound on the mutual information between query and
    positive: I(q; d+) >= log(N+1) - L. The bound is CEILINGED at log(N+1), so more
    negatives raise the ceiling and the information story SATURATES at small batch — the
    honest reason large batches help. Verified on a tractable Gaussian joint where
    I = -1/2 ln(1 - rho^2) is exact, using the Bayes-OPTIMAL critic (the density ratio) so
    the test isolates the bound's behavior, not an encoder's. (`gaussian_mi`,
    `mi_bound_curve`.)

  MOVEMENT 3 — ALIGNMENT AND UNIFORMITY ON THE SPHERE (Wang & Isola 2020). As the number
    of negatives grows, the loss splits into ALIGNMENT (positives pulled together,
    E||f(x)-f(y)||^2) and UNIFORMITY (every embedding pushed toward the uniform
    distribution on S^{d-1}, log E e^{-t||f(x)-f(y)||^2}). The uniformity optimum is the
    EXACT uniform sphere law of the hypersphere topic — the kappa -> 0 limit of vMF — and
    temperature plays the role of an inverse vMF concentration tau ~ 1/kappa. (`alignment`,
    `uniformity`, `optimize_align_unif`, `optimize_infonce`, `config_stats`.)

  MOVEMENT 4 — TEMPERATURE AND THE HARD-NEGATIVE GRADIENT. The gradient is a softmax-
    weighted repulsion over the negatives: the weight on negative i is its softmax share
    p_i = e^{s_i/tau}/sum_j e^{s_j/tau}, so the hardest (nearest) negative dominates, and
    tau controls how sharply — small tau concentrates almost all the push on a single
    negative. The gradient structure is exact; the "right" tau is empirical. (`grad_logits`,
    `negative_weights`, `negative_weight_entropy`, `top1_negative_mass`.)

Honest caveats (rigorFlag territory, asserted as DIRECTIONS): the MI bound is loose and
ceilinged at log(N+1); the alignment/uniformity split is ASYMPTOTIC in the negative count,
not a finite-sample identity; and the temperature trade-off (hard-negative focus vs
tolerance to true-but-similar positives) makes the optimal tau empirical, not chosen by a
theorem. The contrastive setup also assumes TRUE negatives — in retrieval, sampled
negatives include false negatives — and cosine logits inherit the hypersphere's
isotropy assumption. The laboratory's numbers are a deterministic SYNTHETIC vMF stand-in,
not a trained dual encoder.

This module imports its prerequisites (`hypersphere-vmf-geometry`, `the-retrieval-problem`)
for the sphere samplers, the uniform/vMF laws, the mean-resultant uniformity measure, and
the cosine score — it never reimplements them. `viz_constants()` prints what
`InfoNCEContrastiveLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/infonce-contrastive-objective/infonce_contrastive_objective.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.special import logsumexp

# Established cross-topic pattern: add each prereq's HYPHENATED dir to the path, import its
# UNDERSCORED module. The sphere samplers, the uniform/vMF laws, and the mean-resultant
# uniformity measure come from hypersphere-vMF; the cosine score and corpus ranking from
# the-retrieval-problem. We never reimplement them.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("hypersphere-vmf-geometry", "the-retrieval-problem"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import (  # noqa: E402
    normalize,
    sample_uniform_sphere,
    sample_vmf,
    mean_resultant_length,
    mle_mu,
    kappa_hat_exact,
    _mean_pairwise_cosine,
)
from the_retrieval_problem import cosine, rank  # noqa: E402


# --------------------------------------------------------------------------- #
# Movement 1 — the InfoNCE objective.
# --------------------------------------------------------------------------- #

def scores(z_q: np.ndarray, Z_k: np.ndarray) -> np.ndarray:
    """The similarity scores of one query against M keys: the cosine <z_q, z_k> of unit
    vectors (the retrieval-problem's score, now the training signal). Rows of Z_k and z_q
    are assumed L2-normalized; we normalize defensively."""
    z_q = normalize(np.atleast_2d(z_q))[0]
    Z_k = normalize(np.atleast_2d(Z_k))
    return Z_k @ z_q


def softmax_temperature(s: np.ndarray, tau: float) -> np.ndarray:
    """Numerically stable softmax of s/tau. GUARD: tau must be a positive temperature."""
    if not (tau > 1e-8):
        raise ValueError(f"temperature tau must be > 1e-8, got {tau}")
    s = np.asarray(s, dtype=float) / tau
    return np.exp(s - logsumexp(s))


def info_nce_loss(z_q: np.ndarray, z_pos: np.ndarray, Z_neg: np.ndarray,
                  tau: float) -> float:
    """Single-positive InfoNCE: the (N+1)-way cross-entropy that must pick the positive,
    L = -log( e^{s+/tau} / (e^{s+/tau} + sum_i e^{s_i^-/tau}) ). GUARDS: tau > 0,
    at least one negative. Computed with logsumexp (never a raw sum of exponentials)."""
    if not (tau > 1e-8):
        raise ValueError(f"temperature tau must be > 1e-8, got {tau}")
    Z_neg = np.atleast_2d(Z_neg)
    if Z_neg.shape[0] < 1:
        raise ValueError("need at least one negative")
    s_pos = float(scores(z_q, z_pos[None, :])[0])
    s_neg = scores(z_q, Z_neg)
    logits = np.concatenate(([s_pos], s_neg)) / tau
    return float(logsumexp(logits) - logits[0])


def info_nce_loss_batch(Zq: np.ndarray, Zk: np.ndarray, tau: float) -> float:
    """In-batch-negative InfoNCE over a batch of B (query, positive) pairs: Zq[i] is query
    i, Zk[i] its positive, and every other Zk[j != i] is a negative for query i. The mean
    over rows of the (N+1=B)-way cross-entropy. GUARDS: tau > 0, B >= 2 (need >= 1
    negative). Uses the BxB logit matrix and a row-wise logsumexp."""
    if not (tau > 1e-8):
        raise ValueError(f"temperature tau must be > 1e-8, got {tau}")
    Zq = normalize(np.atleast_2d(Zq))
    Zk = normalize(np.atleast_2d(Zk))
    B = Zq.shape[0]
    if B < 2:
        raise ValueError(f"in-batch negatives need B >= 2 rows, got {B}")
    S = (Zq @ Zk.T) / tau                       # (B, B); diagonal = positives
    row_lse = logsumexp(S, axis=1)
    pos = np.diag(S)
    return float(np.mean(row_lse - pos))


# --------------------------------------------------------------------------- #
# Movement 2 — InfoNCE is a mutual-information lower bound (CPC).
# --------------------------------------------------------------------------- #

def gaussian_mi(rho: float) -> float:
    """Exact mutual information (nats) of the bivariate-normal joint with correlation rho:
    I(q; k) = -1/2 ln(1 - rho^2). GUARD: |rho| < 1."""
    if not (abs(rho) < 1.0):
        raise ValueError(f"need |rho| < 1, got {rho}")
    return -0.5 * math.log(1.0 - rho * rho)


def rho_for_mi(target_nats: float) -> float:
    """The correlation rho whose Gaussian joint has mutual information `target_nats`:
    rho = sqrt(1 - e^{-2 I})."""
    return math.sqrt(1.0 - math.exp(-2.0 * target_nats))


def estimate_infonce_bound(rho: float, n_neg: int, n_batches: int,
                           seed: int = 0) -> float:
    """Estimate the InfoNCE MI lower bound log(N+1) - L on the Gaussian joint with the
    BAYES-OPTIMAL critic (the density ratio f(q,k) = p(k|q)/p(k)), isolating the bound's
    behavior rather than an encoder's. q ~ N(0,1), positive k+ ~ p(k|q) = N(rho q, 1-rho^2),
    n_neg negatives ~ p(k) = N(0,1). The log critic is, up to a per-batch constant that
    cancels in the softmax, log f(q,k) = -1/2 (k - rho q)^2/(1-rho^2) + 1/2 k^2. ONE RNG
    stream (per-seed default_rng over consecutive seeds would inflate the variance)."""
    if n_neg < 1:
        raise ValueError(f"need n_neg >= 1, got {n_neg}")
    var = 1.0 - rho * rho
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n_batches)
    k_pos = rho * q + math.sqrt(var) * rng.standard_normal(n_batches)
    k_neg = rng.standard_normal((n_batches, n_neg))           # ~ p(k) = N(0,1)
    K = np.concatenate((k_pos[:, None], k_neg), axis=1)       # (B, N+1), col 0 = positive
    logf = -0.5 * (K - rho * q[:, None]) ** 2 / var + 0.5 * K ** 2
    loss = logsumexp(logf, axis=1) - logf[:, 0]
    L = float(np.mean(loss))
    return math.log(n_neg + 1) - L


def mi_bound_curve(target_nats: float, n_neg_grid, n_batches: int = 40000,
                   seed: int = 0):
    """The MI-bound curve the viz draws: for each negative count, the achieved bound
    log(N+1) - L against its ceiling log(N+1) and the true MI. Returns a list of
    {n_neg, candidates, ceiling, bound, mi_true}."""
    rho = rho_for_mi(target_nats)
    rows = []
    for j, n_neg in enumerate(n_neg_grid):
        bound = estimate_infonce_bound(rho, n_neg, n_batches, seed=seed + j)
        rows.append({
            "n_neg": int(n_neg),
            "candidates": int(n_neg + 1),
            "ceiling": round(math.log(n_neg + 1), 4),
            "bound": round(bound, 4),
            "mi_true": round(target_nats, 4),
        })
    return rows


# --------------------------------------------------------------------------- #
# Movement 3 — alignment and uniformity on the sphere (Wang & Isola 2020).
# --------------------------------------------------------------------------- #

def alignment(Zx: np.ndarray, Zy: np.ndarray) -> float:
    """The alignment loss E||f(x) - f(y)||^2 over positive pairs (rows of Zx paired with
    rows of Zy). Small when positives map close together."""
    return float(np.mean(np.sum((Zx - Zy) ** 2, axis=1)))


def uniformity(Z: np.ndarray, t: float = 2.0) -> float:
    """The uniformity loss log E_{i != j} e^{-t ||z_i - z_j||^2} (Wang & Isola's Gaussian
    potential). Lower = more spread out; minimized by the uniform distribution on the
    sphere. GUARD: need n >= 2 points."""
    n = Z.shape[0]
    if n < 2:
        raise ValueError(f"uniformity needs n >= 2 points, got {n}")
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    w = np.exp(-t * d2)
    off = w.sum() - np.trace(w)                  # exclude i == j
    return float(math.log(off / (n * (n - 1))))


def _align_unif_grad(Z: np.ndarray, pair: np.ndarray, t: float, lam: float):
    """Euclidean gradient of L_align + lam * L_unif at the points Z, with `pair` the
    partner index of each row (an involution). Returns (loss, grad)."""
    n = Z.shape[0]
    # Alignment: L = (1/n) sum_i ||z_i - z_{pair(i)}||^2 ; d/dz_i = (2/n)(z_i - z_pair(i)).
    diff = Z - Z[pair]
    L_align = float(np.mean(np.sum(diff ** 2, axis=1)))
    g_align = (2.0 / n) * diff
    # Uniformity: L = log( S / (n(n-1)) ), S = sum_{i!=j} w_ij, w_ij = e^{-t||z_i-z_j||^2}.
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    w = np.exp(-t * d2)
    np.fill_diagonal(w, 0.0)
    S = w.sum()
    L_unif = float(math.log(S / (n * (n - 1))))
    rowsum = w.sum(axis=1, keepdims=True)
    g_unif = (-4.0 * t / S) * (rowsum * Z - w @ Z)
    return L_align + lam * L_unif, g_align + lam * g_unif


def _info_nce_grad(Z: np.ndarray, pair: np.ndarray, tau: float):
    """Euclidean gradient of the in-batch InfoNCE loss (each point is a query whose positive
    is its partner and whose negatives are all other points), w.r.t. the points Z. Returns
    (loss, grad). Derived from the softmax weights Q[i,j] = softmax_{j != i}(<z_i,z_j>/tau)."""
    n = Z.shape[0]
    S = (Z @ Z.T) / tau
    np.fill_diagonal(S, -np.inf)
    row_lse = logsumexp(S, axis=1)
    pos = S[np.arange(n), pair]
    loss = float(np.mean(row_lse - pos))
    Q = np.exp(S - row_lse[:, None])             # (n, n), zero on the diagonal
    # grad_a = (1/(n*tau)) ( -2 z_{pair(a)} + (Q + Q^T) @ Z )_a  (pairing is an involution).
    grad = (1.0 / (n * tau)) * (-2.0 * Z[pair] + (Q + Q.T) @ Z)
    return loss, grad


def _sphere_step(Z: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
    """One projected (Riemannian) gradient-descent step on the sphere: project the gradient
    onto the tangent space, step, and renormalize."""
    g_tan = grad - np.sum(grad * Z, axis=1, keepdims=True) * Z
    return normalize(Z - lr * g_tan)


def _make_pairs(m: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """An initial configuration of 2m points on S^{d-1} and the partner-index involution:
    rows 2i and 2i+1 are a positive pair."""
    Z = sample_uniform_sphere(2 * m, d, seed=seed)
    pair = np.empty(2 * m, dtype=int)
    pair[0::2] = np.arange(1, 2 * m, 2)
    pair[1::2] = np.arange(0, 2 * m, 2)
    return Z, pair


def optimize_align_unif(m: int = 24, d: int = 3, t: float = 2.0, lam: float = 1.0,
                        steps: int = 400, lr: float = 0.5, seed: int = 0):
    """Minimize L_align + lam * L_unif directly by projected GD on the sphere. Returns the
    final configuration Z and its (alignment, uniformity)."""
    Z, pair = _make_pairs(m, d, seed)
    for _ in range(steps):
        _, g = _align_unif_grad(Z, pair, t, lam)
        Z = _sphere_step(Z, g, lr)
    return Z, pair


def optimize_infonce(m: int = 24, d: int = 3, tau: float = 0.2, steps: int = 400,
                     lr: float = 0.5, seed: int = 1):
    """Minimize the in-batch InfoNCE loss by projected GD on the sphere. Returns the final
    configuration Z and the partner involution."""
    Z, pair = _make_pairs(m, d, seed)
    for _ in range(steps):
        _, g = _info_nce_grad(Z, pair, tau)
        Z = _sphere_step(Z, g, lr)
    return Z, pair


def config_stats(Z: np.ndarray, pair: np.ndarray, t: float = 2.0) -> dict[str, float]:
    """Rotation-invariant summaries of a configuration: alignment of the pairs, uniformity,
    and the mean resultant length ||mean(Z)|| (-> 0 means uniform on the sphere, the
    kappa -> 0 vMF law of the hypersphere topic)."""
    return {
        "alignment": round(alignment(Z, Z[pair]), 4),
        "uniformity": round(uniformity(Z, t), 4),
        "mean_resultant": round(float(np.linalg.norm(Z.mean(axis=0))), 4),
    }


def align_unif_vs_tau(taus, m: int = 24, d: int = 3, t: float = 2.0, steps: int = 400,
                      lr: float = 0.5, seed: int = 1):
    """The temperature trade-off the viz draws: train InfoNCE at each tau and report the
    resulting (alignment, uniformity) and the equivalent vMF concentration kappa ~ 1/tau.
    Lower tau buys uniformity (and tight, high-kappa clusters) at the cost of alignment."""
    rows = []
    for tau in taus:
        Z, pair = optimize_infonce(m=m, d=d, tau=tau, steps=steps, lr=lr, seed=seed)
        s = config_stats(Z, pair, t)
        rows.append({
            "tau": round(float(tau), 3),
            "kappa_equiv": round(1.0 / tau, 2),
            "alignment": s["alignment"],
            "uniformity": s["uniformity"],
            "mean_resultant": s["mean_resultant"],
        })
    return rows


# --------------------------------------------------------------------------- #
# Movement 4 — temperature and the hard-negative gradient.
# --------------------------------------------------------------------------- #

def grad_logits(s: np.ndarray, pos: int, tau: float) -> np.ndarray:
    """The InfoNCE gradient w.r.t. the raw similarity scores s (the positive at index
    `pos`): dL/ds_pos = -(1 - p_pos)/tau and dL/ds_i = +p_i/tau for negatives, where
    p = softmax(s/tau). The exact, provable gradient structure."""
    p = softmax_temperature(s, tau)
    g = p / tau
    g[pos] -= 1.0 / tau
    return g


def negative_weights(s_neg: np.ndarray, tau: float) -> np.ndarray:
    """The gradient weights over the negatives alone: p_i = softmax(s_neg/tau). As tau -> 0
    the mass concentrates on the hardest (highest-similarity) negative."""
    return softmax_temperature(s_neg, tau)


def negative_weight_entropy(s_neg: np.ndarray, tau: float) -> float:
    """Shannon entropy (nats) of the negative gradient-weight distribution. Increases toward
    log(N) as tau grows (uniform); shrinks toward 0 as tau -> 0 (one hard negative)."""
    p = negative_weights(s_neg, tau)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def top1_negative_mass(s_neg: np.ndarray, tau: float) -> float:
    """The largest single gradient weight among the negatives, max_i p_i. -> 1 as tau -> 0."""
    return float(np.max(negative_weights(s_neg, tau)))


def temperature_curve(s_neg: np.ndarray, taus):
    """The temperature panel the viz draws: top-1 mass and weight entropy of a FIXED set of
    negative similarities as tau varies. Returns a list of {tau, top1, entropy}."""
    return [
        {"tau": round(float(tau), 3),
         "top1": round(top1_negative_mass(s_neg, tau), 4),
         "entropy": round(negative_weight_entropy(s_neg, tau), 4)}
        for tau in taus
    ]


# --------------------------------------------------------------------------- #
# Module constants the viz panels step through.
# --------------------------------------------------------------------------- #

TAU_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
N_NEG_GRID = (1, 3, 7, 15, 31, 63, 127, 255)
MI_TARGET_NATS = 2.0
UNIF_T = 2.0
# A fixed, legible spread of negative cosine-similarities for the gradient-weight panel:
# one genuinely hard negative near the query, then a tail of easier ones. The positive sits
# just above the hardest negative. These are the synthetic geometry the viz draws.
POS_COS = 0.92
NEG_COS = (0.86, 0.71, 0.58, 0.44, 0.31, 0.17, 0.02, -0.15)


# --------------------------------------------------------------------------- #
# Finance case study: contrastive fine-tuning of a financial dual encoder.
#
# Each SECTOR is a vMF cluster on the sphere; each COMPANY a tight sub-cluster inside its
# sector. A query and its positive are two draws from the same company; the HARD negatives
# are same-sector, different-company documents (geometrically near but irrelevant); the
# easy negatives are other-sector documents. SYNTHETIC vMF stand-in, not a trained encoder.
# --------------------------------------------------------------------------- #

# Dimensions and concentrations chosen so the geometry actually exhibits the phenomenon
# (the "baked data must exhibit it" rule): two vMF(mu, kappa) draws have expected cosine
# A_d(kappa)^2, so kappa_sector is set high enough that same-sector companies are genuinely
# NEAR (cosine ~ 0.4), kappa_company higher still so query<->positive is nearest, and the
# random sector means are near-orthogonal so other-sector documents sit near cosine 0.
FIN_DIM = 32
FIN_SECTORS = ("rates", "credit", "fx", "equity")
FIN_COMPANIES_PER_SECTOR = 4
FIN_KAPPA_SECTOR = 60.0          # tight enough that a sector is a real neighborhood
FIN_KAPPA_COMPANY = 350.0        # tighter still: a company's documents cluster hard


def finance_dataset(seed: int = 7):
    """Build the synthetic financial contrastive batch. Returns a dict with the query and
    positive of a focal company, the same-sector hard negatives, the other-sector easy
    negatives, and the full in-batch negative set (one positive per company)."""
    rng = np.random.default_rng(seed)
    sector_mu = normalize(rng.standard_normal((len(FIN_SECTORS), FIN_DIM)))
    companies = {}                                # (sector, c) -> company mean direction
    for si, mu in enumerate(sector_mu):
        members = sample_vmf(FIN_COMPANIES_PER_SECTOR, mu, FIN_KAPPA_SECTOR,
                             seed=seed + 11 * si + 1)
        for ci in range(FIN_COMPANIES_PER_SECTOR):
            companies[(si, ci)] = normalize(members[ci])
    # The focal company (rates / company 0): query + positive are two draws from it.
    focal = (0, 0)
    qp = sample_vmf(2, companies[focal], FIN_KAPPA_COMPANY, seed=seed + 101)
    z_q, z_pos = qp[0], qp[1]
    # One representative positive document per OTHER company = the in-batch negatives.
    neg_docs, neg_sector = [], []
    for (si, ci), mu in companies.items():
        if (si, ci) == focal:
            continue
        doc = sample_vmf(1, mu, FIN_KAPPA_COMPANY, seed=seed + 211 + 7 * si + ci)[0]
        neg_docs.append(doc)
        neg_sector.append(si)
    neg_docs = np.array(neg_docs)
    neg_sector = np.array(neg_sector)
    same_sector = neg_sector == focal[0]
    return {
        "z_q": z_q, "z_pos": z_pos,
        "neg_docs": neg_docs, "neg_sector": neg_sector, "same_sector": same_sector,
        "focal_sector": focal[0],
    }


def finance_hard_negative_share(data, tau: float) -> dict[str, float]:
    """The share of the InfoNCE gradient that lands on the same-sector (hard) negatives at
    temperature tau, and the single largest hard-negative weight. The gradient weight on
    each negative is its softmax share over the negatives."""
    s_neg = data["neg_docs"] @ data["z_q"]
    w = negative_weights(s_neg, tau)
    same = data["same_sector"]
    return {
        "hard_share": float(w[same].sum()),
        "max_hard_weight": float(w[same].max()),
        "n_hard": int(same.sum()),
        "n_neg": int(len(s_neg)),
    }


def finance_demo() -> dict:
    """The finance headline: at small tau the same-sector hard negatives capture a far
    larger share of the InfoNCE gradient than at large tau, and the single hardest
    same-sector negative dominates. Made executable below."""
    data = finance_dataset()
    s_neg = data["neg_docs"] @ data["z_q"]
    cos_pos = float(data["z_pos"] @ data["z_q"])
    print(f"  focal company query vs positive cosine = {cos_pos:.4f}; "
          f"{data['same_sector'].sum()} same-sector hard negatives, "
          f"{(~data['same_sector']).sum()} other-sector easy negatives "
          f"(synthetic vMF, not a trained encoder)")
    print(f"  hardest same-sector negative cosine = {s_neg[data['same_sector']].max():.4f}, "
          f"hardest other-sector cosine = {s_neg[~data['same_sector']].max():.4f}")
    out = {"cos_pos": cos_pos}
    print(f"  {'tau':>8}{'hard-neg gradient share':>26}{'max hard weight':>18}")
    for tau in (0.05, 0.2, 1.0):
        sh = finance_hard_negative_share(data, tau)
        out[f"hard_share_{tau}"] = sh["hard_share"]
        out[f"max_hard_{tau}"] = sh["max_hard_weight"]
        print(f"  {tau:>8.2f}{sh['hard_share']:>26.4f}{sh['max_hard_weight']:>18.4f}")
    print("  -> lower tau routes more of the gradient onto the same-sector hard negatives.")
    return out


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_loss_is_cross_entropy_and_invariances() -> None:
    """Movement 1: InfoNCE is the (N+1)-way cross-entropy picking the positive. Adding a
    constant to all scores leaves the loss unchanged; as tau -> infinity the loss -> the
    uniform-classifier value log(N+1); a positive far above the negatives drives it to 0."""
    rng = np.random.default_rng(0)
    d, N = 16, 7
    z_q = normalize(rng.standard_normal(d))
    z_pos = normalize(rng.standard_normal(d))
    Z_neg = normalize(rng.standard_normal((N, d)))
    base = info_nce_loss(z_q, z_pos, Z_neg, tau=0.1)
    # Cross-entropy cross-check: -log softmax(logits)[positive].
    logits = np.concatenate(([z_pos @ z_q], Z_neg @ z_q)) / 0.1
    ce = float(-(logits[0] - logsumexp(logits)))
    assert abs(base - ce) == 0.0 or abs(base - ce) < 1e-12, f"{base} != cross-entropy {ce}"
    # tau -> large: scores wash out, loss -> log(N+1).
    big = info_nce_loss(z_q, z_pos, Z_neg, tau=1e6)
    assert abs(big - math.log(N + 1)) < 1e-3, f"tau->inf loss {big} != log(N+1) {math.log(N+1)}"
    # A positive far above the negatives -> loss -> 0.
    z_close = normalize(z_q + 1e-6 * rng.standard_normal(d))
    far_neg = normalize(-np.tile(z_q, (N, 1)) + 1e-6 * rng.standard_normal((N, d)))
    tiny = info_nce_loss(z_q, z_close, far_neg, tau=0.05)
    assert tiny < 1e-3, f"separable case loss should be ~0, got {tiny}"
    print(f"  [ok] InfoNCE = (N+1)-way cross-entropy; tau->inf -> log(N+1)={math.log(N+1):.4f}; "
          f"separable -> {tiny:.2e}")


def test_batch_matches_single() -> None:
    """The in-batch loss is the mean of the single-positive losses with each row's positive
    on the diagonal and every other column a negative."""
    rng = np.random.default_rng(1)
    d, B, tau = 12, 6, 0.15
    Zq = normalize(rng.standard_normal((B, d)))
    Zk = normalize(rng.standard_normal((B, d)))
    batch = info_nce_loss_batch(Zq, Zk, tau)
    manual = np.mean([
        info_nce_loss(Zq[i], Zk[i], np.delete(Zk, i, axis=0), tau) for i in range(B)
    ])
    assert abs(batch - manual) < 1e-10, f"batch {batch} != manual mean {manual}"
    print(f"  [ok] in-batch InfoNCE = mean of single-positive losses ({batch:.4f})")


def test_mi_lower_bound_holds_and_saturates() -> None:
    """Movement 2 (CPC): the bound log(N+1) - L never exceeds the true MI, tightens
    monotonically as the negative count grows, and is PROVABLY loose (saturated below the
    truth) at small N where the ceiling log(N+1) < I_true. Pinned to the observed run."""
    rho = rho_for_mi(MI_TARGET_NATS)
    i_true = gaussian_mi(rho)
    assert abs(i_true - MI_TARGET_NATS) < 1e-9
    rows = mi_bound_curve(MI_TARGET_NATS, N_NEG_GRID, n_batches=60000, seed=0)
    bounds = [r["bound"] for r in rows]
    # The bound never exceeds the truth (a small Monte-Carlo slack is allowed).
    for r in rows:
        assert r["bound"] <= r["mi_true"] + 0.03, \
            f"bound {r['bound']} exceeded I_true {r['mi_true']} at n_neg={r['n_neg']}"
        assert r["bound"] <= r["ceiling"] + 1e-9, \
            f"bound {r['bound']} exceeded its ceiling {r['ceiling']} at n_neg={r['n_neg']}"
    # Tightens monotonically in the negative count.
    assert all(bounds[i] <= bounds[i + 1] + 0.02 for i in range(len(bounds) - 1)), \
        f"bound not monotone in N: {bounds}"
    # Provably loose at the smallest N (ceiling below the truth).
    assert rows[0]["ceiling"] < i_true, "test setup: smallest ceiling should be below I_true"
    assert rows[0]["bound"] < i_true - 0.3, \
        f"bound should be loose at n_neg={rows[0]['n_neg']}: {rows[0]['bound']} vs I_true {i_true:.4f}"
    # Approaches the truth once the ceiling clears it.
    assert bounds[-1] > i_true - 0.25, f"bound should approach I_true at large N: {bounds[-1]}"
    print(f"  [ok] MI bound (I_true={i_true:.3f}): {bounds} -> tightens to I_true, "
          f"saturated at small N (ceiling {rows[0]['ceiling']} < {i_true:.3f})")


def test_uniformity_minimized_by_uniform_sphere() -> None:
    """Movement 3, the clean sub-claim: the uniform distribution on the sphere has lower
    uniformity loss than a concentrated (high-kappa vMF) cloud or a near-collapsed clump."""
    d = 8
    uni = sample_uniform_sphere(200, d, seed=2)
    mu = normalize(np.ones(d))
    tight = sample_vmf(200, mu, kappa=80.0, seed=3)
    clump = sample_vmf(200, mu, kappa=2000.0, seed=4)
    u_uni, u_tight, u_clump = uniformity(uni), uniformity(tight), uniformity(clump)
    assert u_uni < u_tight < u_clump, \
        f"uniform should minimize uniformity loss: {u_uni} !< {u_tight} !< {u_clump}"
    print(f"  [ok] uniformity minimized by the uniform sphere: "
          f"uniform {u_uni:.3f} < vMF(80) {u_tight:.3f} < vMF(2000) {u_clump:.3f}")


def test_alignment_uniformity_decomposition() -> None:
    """Movement 3 (Wang & Isola): minimizing InfoNCE drives BOTH alignment down and the
    configuration toward uniformity (mean resultant -> 0), landing near the configuration
    found by minimizing alignment + uniformity directly. Robust, rotation-invariant
    comparison; pinned to the observed run."""
    m, d = 24, 3
    Z0, pair0 = _make_pairs(m, d, seed=5)
    init = config_stats(Z0, pair0, UNIF_T)
    Za, pa = optimize_align_unif(m=m, d=d, t=UNIF_T, lam=1.0, steps=500, lr=0.5, seed=5)
    Zi, pi = optimize_infonce(m=m, d=d, tau=0.2, steps=500, lr=0.5, seed=6)
    sa, si = config_stats(Za, pa, UNIF_T), config_stats(Zi, pi, UNIF_T)
    # Both optimizers pull the positive pairs together (alignment far below the random init).
    assert sa["alignment"] < 0.5 * init["alignment"], f"align+unif did not align: {sa} vs {init}"
    assert si["alignment"] < 0.5 * init["alignment"], f"InfoNCE did not align: {si} vs {init}"
    # Both land near-uniform: mean resultant close to 0 (the kappa -> 0 vMF law).
    assert sa["mean_resultant"] < 0.15, f"align+unif not uniform: {sa}"
    assert si["mean_resultant"] < 0.15, f"InfoNCE not uniform: {si}"
    # The two converge to comparable alignment and uniformity (the decomposition).
    assert abs(sa["alignment"] - si["alignment"]) < 0.15, f"alignment mismatch: {sa} vs {si}"
    assert abs(sa["uniformity"] - si["uniformity"]) < 0.20, f"uniformity mismatch: {sa} vs {si}"
    print(f"  [ok] alignment/uniformity: init {init} -> align+unif {sa} ~ InfoNCE {si} "
          f"(both align AND spread uniformly)")


def test_temperature_tradeoff_directions() -> None:
    """Movement 3 corollary: lower temperature buys uniformity (a more spread, higher-kappa-
    equivalent configuration) at the cost of alignment. Directions pinned to the observed
    run across the temperature grid."""
    rows = align_unif_vs_tau((0.1, 0.2, 0.5), m=24, d=3, t=UNIF_T, steps=400, lr=0.5, seed=6)
    aligns = [r["alignment"] for r in rows]      # tau increasing
    unifs = [r["uniformity"] for r in rows]
    # Lower tau -> tighter alignment (smaller) and lower uniformity loss (more spread).
    assert aligns[0] <= aligns[-1] + 1e-6, f"lower tau should not worsen alignment: {aligns}"
    assert unifs[0] <= unifs[-1] + 1e-6, f"lower tau should not worsen uniformity: {unifs}"
    print(f"  [ok] temperature trade-off: alignment {aligns}, uniformity {unifs} "
          f"(tau = 0.1, 0.2, 0.5)")


def test_gradient_structure() -> None:
    """Movement 4: the exact gradient w.r.t. the scores, dL/ds_pos = -(1-p_pos)/tau and
    dL/ds_i = +p_i/tau, checked against a finite-difference of the loss."""
    rng = np.random.default_rng(8)
    N, tau = 6, 0.2
    s = rng.standard_normal(N + 1)               # index 0 is the positive
    pos = 0

    def loss_of(sv):
        return float(logsumexp(sv / tau) - sv[pos] / tau)

    g = grad_logits(s, pos, tau)
    p = softmax_temperature(s, tau)
    assert abs(g[pos] - (-(1.0 - p[pos]) / tau)) < 1e-9, "positive-score gradient wrong"
    for i in range(N + 1):
        if i != pos:
            assert abs(g[i] - p[i] / tau) < 1e-9, f"negative-score gradient wrong at {i}"
    # Finite-difference check.
    eps = 1e-6
    for i in range(N + 1):
        sp, sm = s.copy(), s.copy()
        sp[i] += eps
        sm[i] -= eps
        fd = (loss_of(sp) - loss_of(sm)) / (2 * eps)
        assert abs(fd - g[i]) < 1e-5, f"finite-diff mismatch at {i}: {fd} vs {g[i]}"
    print("  [ok] gradient structure: dL/ds_pos = -(1-p_pos)/tau, dL/ds_i = +p_i/tau (FD-checked)")


def test_temperature_concentrates_gradient() -> None:
    """Movement 4: over a fixed spread of negative similarities, the top-1 gradient weight
    DECREASES in tau and the weight entropy INCREASES in tau (toward log N, the uniform
    ceiling); at the smallest tau the hardest negative carries the dominant mass."""
    s_neg = np.array(NEG_COS)
    rows = temperature_curve(s_neg, TAU_GRID)
    top1 = [r["top1"] for r in rows]             # tau increasing
    ent = [r["entropy"] for r in rows]
    assert all(top1[i] >= top1[i + 1] - 1e-9 for i in range(len(top1) - 1)), \
        f"top-1 mass should fall as tau grows: {top1}"
    assert all(ent[i] <= ent[i + 1] + 1e-9 for i in range(len(ent) - 1)), \
        f"entropy should rise as tau grows: {ent}"
    assert ent[-1] < math.log(len(s_neg)) + 1e-9, "entropy cannot exceed log N"
    assert ent[-1] > ent[0] + 0.5, f"entropy should rise substantially: {ent}"
    assert top1[0] > 0.5, f"smallest tau should concentrate on the hardest negative: {top1[0]}"
    assert int(np.argmax(negative_weights(s_neg, TAU_GRID[0]))) == int(np.argmax(s_neg)), \
        "the dominant weight at small tau should be the hardest (max-similarity) negative"
    print(f"  [ok] temperature concentrates gradient: top-1 {top1} (down), entropy {ent} "
          f"(up, ceiling log N = {math.log(len(s_neg)):.3f})")


def test_finance_hard_negatives_dominate() -> None:
    """Finance headline: at small temperature the same-sector hard negatives capture a far
    larger share of the gradient than at large temperature, and the hardest same-sector
    document carries the single largest weight. Pinned to the observed run."""
    data = finance_dataset()
    lo = finance_hard_negative_share(data, tau=0.05)
    hi = finance_hard_negative_share(data, tau=1.0)
    assert lo["hard_share"] > hi["hard_share"] + 0.3, \
        f"small tau should raise the hard-negative gradient share: {lo} vs {hi}"
    # At small tau the dominant weight is a same-sector (hard) negative.
    s_neg = data["neg_docs"] @ data["z_q"]
    w = negative_weights(s_neg, 0.05)
    assert data["same_sector"][int(np.argmax(w))], \
        "the dominant gradient weight at small tau should be a same-sector hard negative"
    print(f"  [ok] finance: hard-negative gradient share {lo['hard_share']:.3f} at tau=0.05 "
          f"vs {hi['hard_share']:.3f} at tau=1.0")


# --------------------------------------------------------------------------- #
# Viz constants — printed for InfoNCEContrastiveLaboratory.tsx to mirror.
# --------------------------------------------------------------------------- #

def viz_constants() -> None:
    """Print the gradient-weight geometry (Panel A), the MI bound vs ceiling (Panel B), and
    the alignment/uniformity trade-off (Panel C) — all baked to the decimal in the .tsx.
    TS recomputes only CLOSED FORM: softmax weights from NEG_COS, the ceiling log(N+1), and
    kappa ~ 1/tau. Every MEASURED number (MI bound, alignment, uniformity, finance share) is
    baked here."""
    print("  PANEL A — gradient weights over negatives (closed-form in TS from these):")
    print(f"    POS_COS = {round(POS_COS, 3)}")
    print(f"    NEG_COS = {[round(float(c), 3) for c in NEG_COS]}")
    print(f"    TAU_GRID = {[round(float(t), 3) for t in TAU_GRID]}")
    tc = temperature_curve(np.array(NEG_COS), TAU_GRID)
    print(f"    TEMP_CURVE = {[{'tau': r['tau'], 'top1': r['top1'], 'entropy': r['entropy']} for r in tc]}")

    print("  PANEL B — MI lower bound vs the log(N+1) ceiling:")
    mb = mi_bound_curve(MI_TARGET_NATS, N_NEG_GRID, n_batches=60000, seed=0)
    print(f"    MI_TARGET_NATS = {round(MI_TARGET_NATS, 4)}")
    print(f"    MI_BOUND = {[{'n_neg': r['n_neg'], 'ceiling': r['ceiling'], 'bound': r['bound']} for r in mb]}")

    print("  PANEL C — alignment / uniformity: the decomposition and the tau trade-off:")
    print(f"    UNIF_T = {round(UNIF_T, 3)}")
    # The convergence headline: a random init, then the SAME endpoint reached by minimizing
    # alignment + uniformity directly and by minimizing InfoNCE — both collapse alignment and
    # spread to uniformity, landing together in the (uniformity, alignment) plane.
    Z0, p0 = _make_pairs(24, 3, seed=5)
    Za, pa = optimize_align_unif(m=24, d=3, t=UNIF_T, lam=1.0, steps=500, lr=0.5, seed=5)
    Zi, pi = optimize_infonce(m=24, d=3, tau=0.2, steps=500, lr=0.5, seed=6)
    conv = [{"label": "init", **config_stats(Z0, p0, UNIF_T)},
            {"label": "align+unif", **config_stats(Za, pa, UNIF_T)},
            {"label": "infonce", **config_stats(Zi, pi, UNIF_T)}]
    print(f"    CONVERGE = {conv}")
    au = align_unif_vs_tau(TAU_GRID, m=24, d=3, t=UNIF_T, steps=400, lr=0.5, seed=6)
    print(f"    ALIGN_UNIF = {[{'tau': r['tau'], 'kappa_equiv': r['kappa_equiv'], 'alignment': r['alignment'], 'uniformity': r['uniformity'], 'mean_resultant': r['mean_resultant']} for r in au]}")

    print("  FINANCE — same-sector hard-negative gradient share:")
    data = finance_dataset()
    fin = [{"tau": round(float(t), 3), **{k: round(v, 4) for k, v in
            ((kk, vv) for kk, vv in finance_hard_negative_share(data, t).items()
             if kk in ("hard_share", "max_hard_weight"))}}
           for t in (0.05, 0.2, 1.0)]
    print(f"    FIN_SHARE = {fin}")


if __name__ == "__main__":
    print("InfoNCE contrastive-objective verification harness")
    test_mi_lower_bound_holds_and_saturates()      # the headline runs first
    test_loss_is_cross_entropy_and_invariances()
    test_batch_matches_single()
    test_uniformity_minimized_by_uniform_sphere()
    test_alignment_uniformity_decomposition()
    test_temperature_tradeoff_directions()
    test_gradient_structure()
    test_temperature_concentrates_gradient()
    print("Finance demo:")
    finance_demo()
    test_finance_hard_negatives_dominate()
    print("Viz constants (mirrored to the decimal in InfoNCEContrastiveLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
