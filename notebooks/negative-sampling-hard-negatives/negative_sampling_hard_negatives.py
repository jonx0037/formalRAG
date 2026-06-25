"""Hard-negative mining and debiased contrastive training — the reference implementation for the
formalRAG `negative-sampling-hard-negatives` topic.

InfoNCE told us the contrastive gradient on a negative IS its softmax weight, so the hardest negative
dominates and random negatives are near-orthogonal noise. This module SPENDS that fact and pays its
hidden cost. Four movements, every pedagogical claim an `assert`:

  MOVEMENT 1 — WHY HARD NEGATIVES: THE GRADIENT WEIGHTS BY SIMILARITY. The InfoNCE gradient on a
    negative is its softmax weight p_i = e^{s_i/tau} / sum_j e^{s_j/tau} (the prereq's exact gradient
    theorem). Random negatives sit near-orthogonal to the query at d=32, so their weights are nearly
    uniform and tiny — a diffuse, low-information gradient. Same-sector HARD negatives sit a fraction of
    a radian away, so the weight concentrates and the per-step gradient (and the InfoNCE loss) is far
    larger. We measure the gradient-weight entropy, the top-1 mass, and the effective number of
    negatives carrying gradient, random vs mined, on the finance geometry — reducing to the imported
    `negative_weights` / `negative_weight_entropy` / `top1_negative_mass`.
    (`random_negative_cosines`, `hard_negative_cosines`, `gradient_weight_comparison`,
    `batch_loss_comparison`, `gradient_concentration_curve`.)

  MOVEMENT 2 — THE FALSE-NEGATIVE PROBLEM. Movement 1's recipe says "sample near the anchor" — but near
    the anchor is exactly where unlabeled true positives live. A label-unaware miner that takes the
    nearest candidates therefore surfaces accidental positives at a rate tau+ that RISES as the mining
    radius tightens, while uniform random sampling hits only the global class prior. On the labeled
    finance pool (four queries per company, so same-company duplicates exist) the nearest neighbors are
    same-company filings: genuine false negatives. (`mine_nearest`, `false_negative_rate`,
    `tau_plus_curve`, `class_prior_tau_plus`.) The substrate is load-bearing: a one-document-per-company
    corpus has tau+ identically zero and the phenomenon vanishes.

  MOVEMENT 3 — THE DEBIASED CONTRASTIVE ESTIMATOR (Chuang, Robinson, Lin, Torralba, Jegelka 2020).
    THE RIGOROUS SPINE. The unlabeled sampling law decomposes p = tau+ p+ + tau- p-, so the
    true-negative expectation is recoverable from unlabeled samples plus the positive distribution:
        E_{p-}[g] = ( E_p[g] - tau+ E_{p+}[g] ) / tau-,    g(x) = e^{s(anchor,x)/tau},
    with a max(., e^{-1/tau}) floor that keeps the corrected denominator positive in finite samples.
    The plug-in estimator is asymptotically unbiased; the biased in-batch mean is not. We assert the
    decomposition as an EXACT identity at the full pool (< 1e-9) and as a closer-than-biased,
    convergent estimator under sampling. Robinson et al.'s (2021) beta-reweighting q_beta propto e^{beta s}
    concentrates the estimator onto harder negatives, and at beta = 1/tau it IS the InfoNCE weighting —
    the bridge back to Movement 1. (`estimator_g`, `oracle_true_negative_mean`, `biased_negative_mean`,
    `positive_mean`, `debiased_negative_mean`, `debiased_convergence_curve`, `beta_reweight_weights`,
    `beta_reweighted_negative_mean`.)

  MOVEMENT 4 — ANCE: THE ASYNCHRONOUS INDEX AND STALENESS (Xiong et al. 2021). THE SYSTEMS-MATH. To
    mine GLOBAL hard negatives rather than in-batch ones, ANCE retrieves them from an approximate
    nearest-neighbor index — which goes STALE as the encoder drifts during training. We model the drift
    deterministically as a non-isometric interpolation of the embedding space toward a seeded target,
    freeze the index at the last refresh, and measure staleness as the top-k overlap between the frozen
    index's mined set and the fresh encoder's. Staleness decays with steps-since-refresh; a stale index
    surfaces stale negatives and drops the now-relevant document; and the refresh interval R trades
    staleness against re-encode cost (1/R per step). The drift MUST be non-isometric: a refreshed
    (co-rotated) index has zero staleness, so the curve exists only because the index lags.
    (`drift_target`, `drift_alpha`, `drifted_encoder`, `mine_with_index`, `staleness_overlap`,
    `staleness_curve`, `refresh_interval_tradeoff`.)

Honest caveats (rigorFlag territory, asserted as DIRECTIONS not decimals): only Movement 3's debiased
estimator is a THEOREM — and even it needs the class prior tau+, which is UNKNOWN in retrieval (here
read off the toy's labels; in practice estimated), while the e^{-1/tau} floor is a finite-sample
safeguard outside the asymptotic identity. Robinson's beta is an EMPIRICAL knob with no theorem to pick
it (too large and it re-imports the false negatives debiasing removes). ANCE's staleness has NO
convergence bound: the drift is a seeded interpolation surrogate for SGD, not trained dynamics, and the
overlap functional and its monotonicity are modeling choices. The ANN index is an exact cosine-argsort
stand-in (no quantization/graph error). The whole laboratory is the deterministic synthetic vMF finance
cloud reused from the InfoNCE / DPR topics — sectors are vMF clusters, companies tight sub-clusters —
not a trained dual encoder.

Imports its prerequisite (`infonce-contrastive-objective`) and the numeric siblings it reuses
(`dense-retrieval-dual-encoders` for the labeled multi-document finance pool, the vMF samplers from
`hypersphere-vmf-geometry`); it never reimplements them. `viz_constants()` prints what
`NegativeSamplingLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/negative-sampling-hard-negatives/negative_sampling_hard_negatives.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# Established cross-topic pattern: add EACH ancestor's HYPHENATED dir to the path, import the
# UNDERSCORED module. The prerequisite is infonce-contrastive-objective (the single DAG edge); it
# transitively needs the two retrieval-problem / hypersphere ancestors. dense-retrieval-dual-encoders
# is an import-only numeric sibling (a connection, not a prerequisite — the import graph is NOT the
# pedagogical DAG): it supplies the labeled, multi-document-per-company finance pool the false-negative
# story needs. We never reimplement a gradient weight, a vMF sampler, or a recall denominator.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",                 # normalize, sample_vmf, mle_mu, kappa_hat_exact
    "the-retrieval-problem",                    # cosine, rank (InfoNCE ancestor)
    "infonce-contrastive-objective",            # THE prerequisite: gradient weights + finance geometry
    "dense-retrieval-dual-encoders",            # the labeled multi-doc-per-company pool (sibling)
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize                                   # noqa: E402
from infonce_contrastive_objective import (                                      # noqa: E402
    negative_weights, info_nce_loss, TAU_GRID,
)
from dense_retrieval_dual_encoders import dpr_finance_matrix, topk_recall        # noqa: E402


# =========================================================================== #
# Movement 1 — why hard negatives: the gradient weights by similarity.
# =========================================================================== #

def _cos_to_anchor(emb: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Cosine of every row of the (unit-norm) embedding matrix to the (unit-norm) anchor — a single
    matrix-vector product. The training signal of the whole topic."""
    return np.atleast_2d(emb) @ np.asarray(anchor, dtype=float)


def random_negative_cosines(pool: dict, anchor_idx: int, n: int, seed: int) -> np.ndarray:
    """Cosines to the anchor of `n` UNIFORMLY sampled TRUE negatives (pool items of a different
    company), from one seeded rng stream. The diffuse, mostly near-orthogonal negatives a random
    sampler draws — the baseline Movement 1 mines beyond."""
    emb, company = pool["emb"], pool["company"]
    others = np.where(company != company[anchor_idx])[0]
    rng = np.random.default_rng(seed)
    n = max(1, min(int(n), len(others)))
    pick = rng.choice(others, size=n, replace=False)
    return _cos_to_anchor(emb[pick], emb[anchor_idx])


def hard_negative_cosines(pool: dict, anchor_idx: int, n: int) -> np.ndarray:
    """Cosines to the anchor of the `n` NEAREST TRUE negatives (different-company pool items, ranked by
    similarity). Label-aware here — these are genuine hard negatives (same-sector, different company);
    Movement 2 drops the label-awareness and shows what a real miner surfaces instead."""
    emb, company = pool["emb"], pool["company"]
    others = np.where(company != company[anchor_idx])[0]
    cos = _cos_to_anchor(emb[others], emb[anchor_idx])
    n = max(1, min(int(n), len(others)))
    top = others[np.argsort(-cos, kind="stable")[:n]]
    return _cos_to_anchor(emb[top], emb[anchor_idx])


def negative_set(pool: dict, anchor_idx: int) -> tuple:
    """An anchor's full set of TRUE negatives (different company): their cosines to the anchor and the
    same-sector (HARD) mask. This is the mixed batch a real in-batch step sees — a few same-sector hard
    negatives among many near-orthogonal cross-sector ones."""
    emb, company, sector = pool["emb"], pool["company"], pool["sector"]
    others = np.where(company != company[anchor_idx])[0]
    cos = _cos_to_anchor(emb[others], emb[anchor_idx])
    same_sector = sector[others] == sector[anchor_idx]
    return cos, same_sector


def hard_gradient_share(cos: np.ndarray, same_sector: np.ndarray, tau: float) -> float:
    """The share of the InfoNCE negative gradient that lands on the same-sector HARD negatives, via the
    IMPORTED `negative_weights`. Because the per-negative weight is e^{s/tau}, the few hard negatives
    carry a share far above their count fraction — and that share grows as tau falls. The generalization
    of the prereq's `finance_hard_negative_share` to a mined batch."""
    w = negative_weights(cos, tau)
    return float(w[same_sector].sum())


def hard_share_curve(cos: np.ndarray, same_sector: np.ndarray, taus) -> list:
    """The same-sector gradient share across a temperature grid — the curve Panel A draws as the tau
    slider moves (TS recomputes the softmax share closed-form from the baked cosines and mask)."""
    return [
        {"tau": round(float(tau), 3), "hard_share": round(hard_gradient_share(cos, same_sector, tau), 4)}
        for tau in taus
    ]


def batch_loss_comparison(pool: dict, anchor_idx: int, n: int, tau: float, seed: int,
                          trials: int = 64) -> dict:
    """The InfoNCE loss of the anchor against a HARD-mined batch vs the average over random batches
    (positive = the anchor's own company document), via the IMPORTED `info_nce_loss`. A hard batch is
    harder to classify, so its loss — and thus the per-step gradient magnitude — is larger: the learning
    signal hard mining buys. The random side averages over `trials` draws (one rng stream) so the
    contrast is not an accident of a single lucky batch."""
    emb, company, doc = pool["emb"], pool["company"], pool["doc_of_company"]
    others = np.where(company != company[anchor_idx])[0]
    z_q, z_pos = emb[anchor_idx], doc[company[anchor_idx]]
    cos = _cos_to_anchor(emb[others], z_q)
    n = max(1, min(int(n), len(others)))
    hard = others[np.argsort(-cos, kind="stable")[:n]]
    rng = np.random.default_rng(seed)
    rand = [info_nce_loss(z_q, z_pos, emb[rng.choice(others, size=n, replace=False)], tau)
            for _ in range(trials)]
    return {
        "hard_loss": round(info_nce_loss(z_q, z_pos, emb[hard], tau), 4),
        "random_loss": round(float(np.mean(rand)), 4),
    }


# =========================================================================== #
# Movement 2 — the false-negative problem.
# =========================================================================== #

def mine_nearest(pool: dict, anchor_idx: int, k: int) -> np.ndarray:
    """The LABEL-UNAWARE miner: the indices of the k nearest OTHER pool items by cosine (an exact
    cosine-argsort stand-in for an ANN index). It does not know which neighbors are accidental
    positives — that ignorance is the whole point of Movement 2. GUARDS: k floored to 1, capped at the
    number of other items; empty pool -> empty."""
    emb = pool["emb"]
    n = emb.shape[0]
    if n <= 1:
        return np.empty(0, dtype=int)
    k = max(1, min(int(k), n - 1))
    cos = _cos_to_anchor(emb, emb[anchor_idx])
    order = np.argsort(-cos, kind="stable")
    order = order[order != anchor_idx]
    return order[:k]


def false_negative_rate(pool: dict, anchor_idx: int, k: int, kind: str, seed: int = 0) -> float:
    """The fraction of an anchor's k selected "negatives" that are actually SAME-company (accidental
    positives — false negatives). `kind="mined"` takes the k nearest; `kind="random"` takes k uniform
    draws from the other items. GUARD: no other items -> 0.0."""
    emb, company = pool["emb"], pool["company"]
    n = emb.shape[0]
    others = np.where(np.arange(n) != anchor_idx)[0]
    if len(others) == 0:
        return 0.0
    k = max(1, min(int(k), len(others)))
    if kind == "mined":
        sel = mine_nearest(pool, anchor_idx, k)
    elif kind == "random":
        rng = np.random.default_rng(seed)
        sel = rng.choice(others, size=k, replace=False)
    else:
        raise ValueError(f"kind must be 'mined' or 'random', got {kind!r}")
    return float(np.mean(company[sel] == company[anchor_idx]))


def class_prior_tau_plus(pool: dict) -> float:
    """The global class prior tau+: the average over anchors of the fraction of OTHER pool items that
    share the anchor's company. This is the rate a uniform random sampler hits accidental positives, and
    the tau+ the debiased estimator needs. GUARD: a pool with no same-company duplicates -> 0.0 (the
    substrate cannot exhibit false negatives — the build-and-run guard)."""
    company = pool["company"]
    n = len(company)
    if n <= 1:
        return 0.0
    rates = []
    for i in range(n):
        others = company[np.arange(n) != i]
        rates.append(float(np.mean(others == company[i])))
    return float(np.mean(rates))


def tau_plus_curve(pool: dict, k_grid, seed: int = 0) -> list:
    """For each mining depth k: the mean false-negative rate of the MINED set vs a RANDOM set, averaged
    over all anchors (random uses one rng stream offset per anchor). The mined rate is high and falls
    toward the prior as k grows; the random rate hugs the class prior. The curve Panel B draws."""
    company = pool["company"]
    n = len(company)
    prior = class_prior_tau_plus(pool)
    rows = []
    for k in k_grid:
        mined = np.mean([false_negative_rate(pool, i, k, "mined") for i in range(n)])
        rand = np.mean([false_negative_rate(pool, i, k, "random", seed=seed + i) for i in range(n)])
        rows.append({"k": int(k), "mined": round(float(mined), 4),
                     "random": round(float(rand), 4), "prior": round(prior, 4)})
    return rows


# =========================================================================== #
# Movement 3 — the debiased contrastive estimator (the rigorous spine).
# =========================================================================== #

def estimator_g(pool: dict, anchor_idx: int, tau: float) -> tuple:
    """The per-item contrastive statistic g(x) = e^{cos(anchor, x)/tau} over all OTHER pool items, plus
    the boolean is_positive mask (same company as the anchor). The object the three expectations average.
    GUARD: tau > 1e-8."""
    if not (tau > 1e-8):
        raise ValueError(f"temperature tau must be > 1e-8, got {tau}")
    emb, company = pool["emb"], pool["company"]
    n = emb.shape[0]
    others = np.where(np.arange(n) != anchor_idx)[0]
    cos = _cos_to_anchor(emb[others], emb[anchor_idx])
    g = np.exp(cos / tau)
    is_pos = company[others] == company[anchor_idx]
    return g, is_pos


def oracle_true_negative_mean(g: np.ndarray, is_pos: np.ndarray) -> float:
    """E_{p-}[g]: the mean of g over the TRUE negatives only (label-aware oracle). The target the
    debiased estimator recovers without labels."""
    neg = g[~is_pos]
    return float(np.mean(neg)) if len(neg) else 0.0


def positive_mean(g: np.ndarray, is_pos: np.ndarray) -> float:
    """E_{p+}[g]: the mean of g over the positive (same-company) items — the positive-distribution
    expectation the correction subtracts."""
    pos = g[is_pos]
    return float(np.mean(pos)) if len(pos) else 0.0


def biased_negative_mean(g: np.ndarray) -> float:
    """E_p[g]: the mean of g over ALL sampled items, accidental positives included. This is what a
    label-unaware in-batch estimator computes — biased away from the true-negative expectation."""
    return float(np.mean(g)) if len(g) else 0.0


def debiased_negative_mean(g_pool: np.ndarray, pos_mean: float, tau_plus: float, tau: float) -> float:
    """The Chuang et al. debiased estimator of E_{p-}[g] from an unlabeled sample g_pool (drawn from p)
    and a positive-distribution mean pos_mean:
        ( E_p[g] - tau+ * pos_mean ) / (1 - tau+),   floored at e^{-1/tau}.
    GUARDS: 0 <= tau+ < 1 (the 1 - tau+ denominator), tau > 1e-8. At tau+ = 0 it collapses to the biased
    mean (the no-contamination case)."""
    if not (tau > 1e-8):
        raise ValueError(f"temperature tau must be > 1e-8, got {tau}")
    if not (0.0 <= tau_plus < 1.0):
        raise ValueError(f"class prior tau+ must lie in [0, 1), got {tau_plus}")
    e_p = float(np.mean(g_pool)) if len(g_pool) else 0.0
    val = (e_p - tau_plus * pos_mean) / (1.0 - tau_plus)
    return float(max(val, np.exp(-1.0 / tau)))


def debiased_convergence_curve(pool: dict, anchor_idx: int, tau: float, tau_plus: float,
                               n_grid, trials: int, seed: int) -> list:
    """For each sample size N: the mean absolute error of the BIASED in-batch mean and the DEBIASED
    estimator against the true-negative oracle, averaged over `trials` random draws of N items from the
    unlabeled pool (one rng stream). The biased error plateaus at the contamination bias; the debiased
    error converges toward zero. The convergence Panel C draws."""
    g, is_pos = estimator_g(pool, anchor_idx, tau)
    oracle = oracle_true_negative_mean(g, is_pos)
    pos_mean = positive_mean(g, is_pos)
    rng = np.random.default_rng(seed)
    rows = []
    for N in n_grid:
        N = int(N)
        biased_err, debiased_err = [], []
        for _ in range(trials):
            draw = rng.choice(g, size=N, replace=True)
            biased_err.append(abs(float(np.mean(draw)) - oracle))
            debiased_err.append(abs(debiased_negative_mean(draw, pos_mean, tau_plus, tau) - oracle))
        rows.append({"N": N, "biased_mae": round(float(np.mean(biased_err)), 4),
                     "debiased_mae": round(float(np.mean(debiased_err)), 4)})
    return rows


def beta_reweight_weights(s_neg: np.ndarray, beta: float) -> np.ndarray:
    """Robinson et al.'s hardness reweighting q_beta(x-) propto e^{beta * s(anchor, x-)}: a stable
    softmax of beta * s_neg. beta = 0 is uniform; beta -> infinity concentrates on the hardest negative.
    At beta = 1/tau this is EXACTLY the InfoNCE negative weighting (the bridge back to Movement 1)."""
    s = np.asarray(s_neg, dtype=float) * float(beta)
    s = s - np.max(s)
    w = np.exp(s)
    return w / w.sum()


def beta_reweighted_negative_mean(g: np.ndarray, s_neg: np.ndarray, beta: float) -> float:
    """The beta-reweighted estimate of the negative expectation: sum_i q_beta(x_i-) g(x_i-). Tilts the
    average toward the harder negatives, where the informative gradient lives."""
    w = beta_reweight_weights(s_neg, beta)
    return float(np.dot(w, g))


# =========================================================================== #
# Movement 4 — ANCE: the asynchronous index and staleness (the systems-math).
# =========================================================================== #

def drift_target(dim: int, seed: int) -> np.ndarray:
    """A single seeded linear target T ~ N(0, 1/dim) the encoder drifts TOWARD during training. One rng
    stream (the Monte-Carlo rule), so the whole drift schedule is bit-reproducible."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dim, dim)) / np.sqrt(dim)


def drift_alpha(t: float, tau_drift: float) -> float:
    """The interpolation schedule alpha_t = 1 - e^{-t/tau_drift}: 0 at the refresh step, rising toward 1
    as training proceeds. A saturating surrogate for the encoder's SGD trajectory."""
    return float(1.0 - np.exp(-float(t) / float(tau_drift)))


def drifted_encoder(X: np.ndarray, T: np.ndarray, alpha: float) -> np.ndarray:
    """The encoder at drift level alpha applied to embeddings X: normalize( (1 - alpha) X + alpha X T^T ).
    NON-ISOMETRIC by construction (the X T^T term is not an orthogonal map), so rankings genuinely change
    as alpha grows — which is exactly why a frozen index goes stale. At alpha = 0 it is the identity."""
    X = np.atleast_2d(X)
    Y = (1.0 - alpha) * X + alpha * (X @ np.asarray(T, dtype=float).T)
    return normalize(Y)


def mine_with_index(index_emb: np.ndarray, query_emb: np.ndarray, k: int) -> np.ndarray:
    """The top-k document indices each query mines from an index: argsort each query's scores against
    index_emb (descending, stable). GUARDS: k floored to 1, capped at the index size; empty index ->
    empty rows."""
    index_emb = np.atleast_2d(index_emb)
    query_emb = np.atleast_2d(query_emb)
    nd = index_emb.shape[0]
    if nd == 0:
        return np.empty((query_emb.shape[0], 0), dtype=int)
    k = max(1, min(int(k), nd))
    S = query_emb @ index_emb.T
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


def _gold_recall_at_1(index_emb: np.ndarray, query_emb: np.ndarray, gold: np.ndarray) -> float:
    """The fraction of queries whose gold document is the index's top-1 — how well a (possibly stale)
    index still surfaces the relevant document. Reuses the imported `topk_recall` over the query-by-index
    score matrix (never a reimplemented recall denominator)."""
    S = np.atleast_2d(query_emb) @ np.atleast_2d(index_emb).T
    return topk_recall(S, np.asarray(gold), 1)


def staleness_overlap(query0: np.ndarray, docs0: np.ndarray, T: np.ndarray,
                      t: float, refresh_t: float, k: int, tau_drift: float) -> float:
    """The mean top-k overlap between the FROZEN index (documents encoded at the last refresh, drift
    alpha(refresh_t)) and the FRESH index (documents at the current drift alpha(t)), both scored against
    the CURRENT (fresh) queries. 1.0 when t == refresh_t; decays as the encoder drifts away from the
    frozen index."""
    a_now, a_ref = drift_alpha(t, tau_drift), drift_alpha(refresh_t, tau_drift)
    q_now = drifted_encoder(query0, T, a_now)
    d_fresh = drifted_encoder(docs0, T, a_now)
    d_stale = drifted_encoder(docs0, T, a_ref)
    fresh = mine_with_index(d_fresh, q_now, k)
    stale = mine_with_index(d_stale, q_now, k)
    overlaps = [len(set(fresh[i]) & set(stale[i])) / fresh.shape[1] for i in range(fresh.shape[0])]
    return float(np.mean(overlaps)) if overlaps else 0.0


def staleness_curve(query0: np.ndarray, docs0: np.ndarray, gold: np.ndarray, T: np.ndarray,
                    t_grid, k: int, tau_drift: float, refresh_t: float = 0.0) -> list:
    """For each step t since the refresh: the staleness overlap, the FROZEN index's gold recall@1, and
    the FRESH index's gold recall@1 (the latter is the ceiling the stale index falls below). The decay
    Panel D draws."""
    a_ref = drift_alpha(refresh_t, tau_drift)
    d_stale = drifted_encoder(docs0, T, a_ref)
    rows = []
    for t in t_grid:
        a_now = drift_alpha(t, tau_drift)
        q_now = drifted_encoder(query0, T, a_now)
        d_fresh = drifted_encoder(docs0, T, a_now)
        rows.append({
            "t": int(t),
            "overlap": round(staleness_overlap(query0, docs0, T, t, refresh_t, k, tau_drift), 4),
            "stale_gold_r1": round(_gold_recall_at_1(d_stale, q_now, gold), 4),
            "fresh_gold_r1": round(_gold_recall_at_1(d_fresh, q_now, gold), 4),
        })
    return rows


def refresh_interval_tradeoff(query0: np.ndarray, docs0: np.ndarray, T: np.ndarray,
                              r_grid, horizon: int, k: int, tau_drift: float) -> list:
    """For each refresh interval R: the average staleness over a training horizon (the index is rebuilt
    every R steps, so the steps-since-refresh cycles 0..R-1) and the amortized re-encode cost 1/R per
    step. Smaller R is fresher but costlier; larger R is cheaper but staler — the systems tradeoff."""
    rows = []
    for R in r_grid:
        R = max(1, int(R))
        overlaps = []
        for t in range(horizon):
            since = t % R                       # steps since the last refresh
            refresh_t = t - since
            overlaps.append(staleness_overlap(query0, docs0, T, t, refresh_t, k, tau_drift))
        rows.append({"R": R, "avg_staleness": round(float(np.mean(overlaps)), 4),
                     "reencode_cost": round(1.0 / R, 4)})
    return rows


# =========================================================================== #
# Module constants the viz panels step through (tuned by _diagnostics()).
# =========================================================================== #

# Movement 1 — the gradient-weight comparison.
M1_ANCHOR = 0                    # the worked anchor query
M1_N_NEG = 6                     # equal-size random vs hard batches (concentration, not set size)
M1_SEED = 3                      # the rng stream for the random batch
GRAD_TAU = 0.2                   # the headline temperature for the weight comparison

# Movement 2 — the false-negative curve.
K_MINE_GRID = (1, 2, 3, 5, 8, 12, 16)    # mining depths for the tau+ curve
M2_SEED = 0

# Movement 3 — the debiased estimator.
M3_TAU = 0.2                     # the temperature defining g = e^{cos/tau}
M3_ANCHOR = 0
N_GRID_M3 = (4, 8, 16, 32, 64, 128, 256, 512)   # sample sizes for the convergence curve
M3_TRIALS = 400                  # Monte-Carlo draws per N (one rng stream)
M3_SEED = 7
BETA_GRID = (0.0, 1.0, 1.0 / M3_TAU)            # uniform, mild, InfoNCE-equivalent (beta = 1/tau)

# Movement 4 — ANCE staleness. A richer document corpus so top-k overlap is a meaningful fraction.
M4_N_COMP = 8                    # companies per sector -> 32 documents (the mineable index)
M4_QPC = 3                       # queries per company
M4_N_QUERIES = 24                # subsample of the query pool used for the staleness measurement
DRIFT_TAU = 10.0                 # the interpolation time-constant (steps)
DRIFT_SEED = 99
K_STALE = 8                      # mining depth for the staleness overlap
T_GRID = (0, 1, 2, 4, 8, 16, 32, 64)            # steps since refresh
R_GRID = (1, 2, 4, 8, 16, 32)                   # refresh intervals for the tradeoff
HORIZON = 32                     # training steps over which the refresh tradeoff is averaged


# =========================================================================== #
# Cached setup — the labeled finance pool + the ANCE corpus, built once.
# =========================================================================== #

_CACHE: dict = {}


def _setup() -> dict:
    """Build (and cache) the two finance instances every movement reads:
      - `pool`: the labeled mining pool — the DPR query embeddings (four per company, so same-company
        duplicates exist), with company and sector labels and the per-company gold document. The
        substrate the false-negative and debiasing movements need.
      - the ANCE corpus: a richer document index (M4_N_COMP companies) and a subsampled query set with
        gold labels, the embeddings the staleness movement drifts.
    """
    if _CACHE:
        return _CACHE
    Q, P, truth, sector_of_passage = dpr_finance_matrix()
    pool = {
        "emb": Q,                                   # 32 query embeddings (4 per company)
        "company": truth,                           # company = gold passage index in [0, 8)
        "sector": sector_of_passage[truth],         # sector of each query's company
        "doc_of_company": P,                        # P[c] = company c's (gold) document
    }
    Qm, Pm, truth_m, _ = dpr_finance_matrix(n_comp=M4_N_COMP, queries_per_company=M4_QPC)
    stride = max(1, Qm.shape[0] // M4_N_QUERIES)
    q_idx = np.arange(Qm.shape[0])[::stride][:M4_N_QUERIES]
    _CACHE.update({
        "pool": pool,
        "tau_plus": class_prior_tau_plus(pool),
        "ance_docs": Pm,                            # the mineable document index (32 docs)
        "ance_queries": Qm[q_idx],                  # the subsampled query set
        "ance_gold": truth_m[q_idx],                # each query's gold document index
        "drift_T": drift_target(Pm.shape[1], DRIFT_SEED),
    })
    return _CACHE


# =========================================================================== #
# Diagnostics — build-and-run separations to read BEFORE trusting any headline.
# =========================================================================== #

def _diagnostics() -> None:
    """Print the separations that set the constants. Read these before pinning any assert (the CLAUDE.md
    build-and-run rule — several 'obvious' claims here are false on the toy geometry)."""
    s = _setup()
    pool = s["pool"]
    n = pool["emb"].shape[0]
    print(f"  [diag] pool: {n} queries, {len(np.unique(pool['company']))} companies, "
          f"{len(np.unique(pool['sector']))} sectors; class prior tau+ = {s['tau_plus']:.4f}")

    cos, same = negative_set(pool, M1_ANCHOR)
    print(f"  [diag] M1 anchor {M1_ANCHOR}: {int(same.sum())} same-sector hard negs of {len(cos)} "
          f"(fraction {np.mean(same):.3f}); hard cos {np.round(np.sort(cos[same])[::-1], 3)}")
    print(f"  [diag] M1 hard gradient share @ tau={GRAD_TAU}: {hard_gradient_share(cos, same, GRAD_TAU):.4f} "
          f"(>> count fraction {np.mean(same):.3f}); share curve {hard_share_curve(cos, same, (1.0, 0.2, 0.05))}")
    print(f"  [diag] M1 batch loss (hard vs random): "
          f"{batch_loss_comparison(pool, M1_ANCHOR, M1_N_NEG, GRAD_TAU, M1_SEED)}")

    print("  [diag] M2 false-negative rate by mining depth k (mined vs random vs prior):")
    for r in tau_plus_curve(pool, K_MINE_GRID, M2_SEED):
        print(f"         k={r['k']:>2}: mined={r['mined']:.3f}  random={r['random']:.3f}  "
              f"prior={r['prior']:.3f}")

    g, is_pos = estimator_g(pool, M3_ANCHOR, M3_TAU)
    print(f"  [diag] M3 @ tau={M3_TAU}: E_p+={positive_mean(g, is_pos):.3f}  "
          f"E_p={biased_negative_mean(g):.3f}  E_p-={oracle_true_negative_mean(g, is_pos):.3f}")
    print("  [diag] M3 convergence (biased MAE flat, debiased MAE -> 0):")
    for r in debiased_convergence_curve(pool, M3_ANCHOR, M3_TAU, s["tau_plus"],
                                        N_GRID_M3, M3_TRIALS, M3_SEED):
        print(f"         N={r['N']:>4}: biased={r['biased_mae']:.3f}  debiased={r['debiased_mae']:.3f}")

    q0, d0, gold, T = s["ance_queries"], s["ance_docs"], s["ance_gold"], s["drift_T"]
    print(f"  [diag] M4 corpus: {d0.shape[0]} docs, {q0.shape[0]} queries; "
          f"fresh gold recall@1 = {_gold_recall_at_1(d0, q0, gold):.3f}")
    print("  [diag] M4 staleness by steps-since-refresh (overlap, stale vs fresh gold r@1):")
    for r in staleness_curve(q0, d0, gold, T, T_GRID, K_STALE, DRIFT_TAU):
        print(f"         t={r['t']:>2}: overlap={r['overlap']:.3f}  stale_r@1={r['stale_gold_r1']:.3f}"
              f"  fresh_r@1={r['fresh_gold_r1']:.3f}")
    print("  [diag] M4 refresh tradeoff (R, avg staleness, re-encode cost 1/R):")
    for r in refresh_interval_tradeoff(q0, d0, T, R_GRID, HORIZON, K_STALE, DRIFT_TAU):
        print(f"         R={r['R']:>2}: avg_staleness={r['avg_staleness']:.3f}  cost={r['reencode_cost']:.3f}")


# =========================================================================== #
# Verification harness — each assert is a pedagogical claim the topic makes.
# =========================================================================== #

def test_hard_negatives_dominate_gradient() -> None:
    """Movement 1, the headline: in a mixed negative batch the few same-sector HARD negatives carry a
    share of the InfoNCE gradient far above their count fraction (the weight is e^{s/tau}), and a
    hard-mined batch yields a larger loss — more learning signal per step — than a random batch. Assert
    the DIRECTIONS, never decimals. Holds for the worked anchor and on average across anchors."""
    s = _setup()
    pool = s["pool"]
    cos, same = negative_set(pool, M1_ANCHOR)
    share = hard_gradient_share(cos, same, GRAD_TAU)
    frac = float(np.mean(same))
    assert share > frac + 0.2, f"hard negatives should carry a disproportionate gradient share: {share} vs {frac}"
    # the same on average over every anchor (not an accident of the worked one).
    shares, fracs = [], []
    for i in range(pool["emb"].shape[0]):
        c, m = negative_set(pool, i)
        shares.append(hard_gradient_share(c, m, GRAD_TAU))
        fracs.append(float(np.mean(m)))
    assert np.mean(shares) > np.mean(fracs) + 0.2, "hard share should exceed the count fraction on average"
    loss = batch_loss_comparison(pool, M1_ANCHOR, M1_N_NEG, GRAD_TAU, M1_SEED)
    assert loss["hard_loss"] > loss["random_loss"], f"a hard batch should give a larger loss: {loss}"
    print(f"  [ok] Movement 1: hard negatives carry {share:.3f} of the gradient on {frac:.3f} of the "
          f"batch; hard-batch loss {loss['hard_loss']} > random {loss['random_loss']}")


def test_hard_share_rises_as_tau_falls() -> None:
    """Movement 1, the temperature story: the same-sector gradient share rises as tau falls (a sharper
    softmax routes more of the gradient onto the hardest negatives), spanning from near the count
    fraction at large tau toward 1 at small tau. Reuses the imported `negative_weights` temperature
    behavior; the collapse to the prereq is `hard_gradient_share` == one call of it."""
    s = _setup()
    cos, same = negative_set(s["pool"], M1_ANCHOR)
    curve = hard_share_curve(cos, same, sorted(TAU_GRID, reverse=True))   # tau high -> low
    shares = [r["hard_share"] for r in curve]
    assert all(shares[i] <= shares[i + 1] + 1e-9 for i in range(len(shares) - 1)), \
        f"hard share should rise as tau falls: {shares}"
    assert shares[-1] > shares[0] + 0.2, f"the temperature should move the share substantially: {shares}"
    # collapse anchor: the share is exactly the imported weights summed over the hard mask.
    direct = float(negative_weights(cos, GRAD_TAU)[same].sum())
    assert abs(hard_gradient_share(cos, same, GRAD_TAU) - direct) < 1e-12, \
        "hard_gradient_share must equal the imported negative_weights summed over the hard mask"
    print(f"  [ok] Movement 1: hard gradient share rises as tau falls ({shares[0]} -> {shares[-1]})")


def test_mining_induces_false_negatives() -> None:
    """Movement 2, the hinge: the label-unaware miner's nearest neighbors are contaminated by accidental
    positives at a far higher rate than random sampling, and the mined rate is highest at the smallest
    mining depth (the nearest neighbors ARE same-company). Assert the contrast and the direction."""
    s = _setup()
    pool = s["pool"]
    curve = tau_plus_curve(pool, K_MINE_GRID, M2_SEED)
    small = curve[0]                                  # k = 1
    assert small["mined"] > small["random"] + 0.1, f"mining should induce more false negatives: {small}"
    assert small["mined"] > 0.5, f"the nearest neighbor should usually be same-company: {small}"
    # mined contamination falls toward the prior as the net widens; random hugs the prior throughout.
    assert curve[-1]["mined"] <= curve[0]["mined"] + 1e-9, "mined rate should not rise with k"
    print(f"  [ok] Movement 2: mining induces false negatives — mined tau+ {small['mined']:.3f} vs "
          f"random {small['random']:.3f} at k=1")


def test_random_rate_matches_prior() -> None:
    """Movement 2: a uniform random sampler hits accidental positives at exactly the class prior
    (unbiased), unlike the miner. Cross-checks `false_negative_rate(kind='random')` against
    `class_prior_tau_plus` over the full mining-depth grid."""
    s = _setup()
    pool = s["pool"]
    prior = s["tau_plus"]
    curve = tau_plus_curve(pool, K_MINE_GRID, M2_SEED)
    for r in curve:
        assert abs(r["random"] - prior) < 0.05, \
            f"random false-negative rate {r['random']} should track the prior {prior} at k={r['k']}"
    print(f"  [ok] Movement 2: random sampling hits the class prior tau+ = {prior:.4f} (unbiased)")


def test_substrate_can_exhibit_false_negatives() -> None:
    """Movement 2, the build-and-run guard: the chosen substrate HAS same-company duplicates, so the
    class prior is strictly positive and the phenomenon is measurable. A one-document-per-company corpus
    would have tau+ identically 0 and the whole movement would be vacuous."""
    s = _setup()
    assert s["tau_plus"] > 0.0, "the mining pool must have same-company duplicates (tau+ > 0)"
    # the multi-document substrate: every company appears more than once.
    company = s["pool"]["company"]
    _, counts = np.unique(company, return_counts=True)
    assert counts.min() > 1, "the false-negative substrate needs >1 document per company"
    print(f"  [ok] Movement 2 substrate: tau+ = {s['tau_plus']:.4f} > 0, "
          f"{int(counts.min())}-{int(counts.max())} queries per company")


def test_debiased_identity_at_full_pool() -> None:
    """Movement 3, the rigorous spine asserted to machine precision: with the EMPIRICAL class prior the
    decomposition E_{p-}[g] = (E_p[g] - tau+ E_{p+}[g]) / (1 - tau+) is an EXACT identity at the full
    pool — the debiased estimator equals the true-negative oracle to < 1e-9, for every anchor."""
    s = _setup()
    pool = s["pool"]
    n = pool["emb"].shape[0]
    for anchor in range(n):
        g, is_pos = estimator_g(pool, anchor, M3_TAU)
        # the per-anchor empirical prior makes the algebra exact.
        tau_plus = float(np.mean(is_pos))
        oracle = oracle_true_negative_mean(g, is_pos)
        pos_mean = positive_mean(g, is_pos)
        debiased = debiased_negative_mean(g, pos_mean, tau_plus, M3_TAU)
        assert abs(debiased - oracle) < 1e-9, \
            f"debiased identity broken at anchor {anchor}: {debiased} != oracle {oracle}"
    print("  [ok] Movement 3: debiased estimator == true-negative oracle at the full pool (< 1e-9, all anchors)")


def test_debiased_closer_than_biased_under_sampling() -> None:
    """Movement 3: under finite sampling the debiased estimator is closer to the true-negative oracle than
    the biased in-batch mean at every N, the debiased error CONVERGES toward zero, and the biased error
    stays bounded away from zero (the contamination bias does not vanish with more samples)."""
    s = _setup()
    curve = debiased_convergence_curve(s["pool"], M3_ANCHOR, M3_TAU, s["tau_plus"],
                                       N_GRID_M3, M3_TRIALS, M3_SEED)
    for r in curve:
        assert r["debiased_mae"] < r["biased_mae"], \
            f"debiased should beat biased at N={r['N']}: {r}"
    assert curve[-1]["debiased_mae"] < curve[0]["debiased_mae"], "debiased error should converge"
    assert curve[-1]["biased_mae"] > 0.1, "biased error should stay bounded away from zero (the bias)"
    assert curve[-1]["debiased_mae"] < curve[-1]["biased_mae"] / 2.0, \
        "at large N the debiased error should be well below the biased bias floor"
    print(f"  [ok] Movement 3: debiased < biased at every N; debiased {curve[0]['debiased_mae']} -> "
          f"{curve[-1]['debiased_mae']} converges, biased plateaus at {curve[-1]['biased_mae']}")


def test_debiased_tau_plus_zero_equals_biased() -> None:
    """Movement 3 collapse anchor: with no assumed contamination (tau+ = 0) the debiased estimator is
    exactly the biased in-batch mean — debiasing only acts through a positive class prior."""
    s = _setup()
    g, _ = estimator_g(s["pool"], M3_ANCHOR, M3_TAU)
    assert abs(debiased_negative_mean(g, 5.0, 0.0, M3_TAU) - biased_negative_mean(g)) < 1e-12, \
        "debiased at tau+=0 must equal the biased mean (independent of the positive mean)"
    print("  [ok] Movement 3 anchor: debiased(tau+=0) == biased in-batch mean")


def test_beta_recovers_infonce_weights() -> None:
    """Movement 3 twin anchor: Robinson's beta-reweighting at beta = 1/tau IS the imported InfoNCE
    `negative_weights(s, tau)` to < 1e-12 — the hardness knob and the temperature are the same dial, the
    bridge back to Movement 1. And beta = 0 is the uniform average."""
    rng = np.random.default_rng(0)
    s_neg = rng.uniform(-0.2, 0.9, size=12)
    for tau in (0.05, 0.2, 0.5):
        assert np.allclose(beta_reweight_weights(s_neg, 1.0 / tau), negative_weights(s_neg, tau), atol=1e-12), \
            f"beta=1/tau must equal the InfoNCE weighting at tau={tau}"
    uniform = beta_reweight_weights(s_neg, 0.0)
    assert np.allclose(uniform, np.full_like(s_neg, 1.0 / len(s_neg)), atol=1e-12), \
        "beta=0 must be the uniform average"
    print("  [ok] Movement 3 twin: beta=1/tau == imported negative_weights (< 1e-12); beta=0 == uniform")


def test_beta_concentrates_on_hard_negatives() -> None:
    """Movement 3: raising beta tilts the reweighted negative mean UP toward the harder (higher-g)
    negatives, and concentrates the weights (lower entropy). The hardness knob does what its name says."""
    s = _setup()
    g, _ = estimator_g(s["pool"], M3_ANCHOR, M3_TAU)
    s_neg = np.log(g) * M3_TAU                       # recover the cosines: s = tau * log g
    means = [beta_reweighted_negative_mean(g, s_neg, b) for b in (0.0, 1.0, 1.0 / M3_TAU)]
    assert means[0] <= means[1] <= means[2] + 1e-9, f"larger beta should weight harder negatives: {means}"
    ent = [float(-np.sum(w * np.log(w))) for w in
           (beta_reweight_weights(s_neg, b) for b in (0.0, 1.0 / M3_TAU))]
    assert ent[1] < ent[0], "larger beta should concentrate the reweighting (lower entropy)"
    print(f"  [ok] Movement 3: beta concentrates on hard negatives (mean {means[0]:.2f} -> {means[2]:.2f})")


def test_debiased_guards() -> None:
    """Movement 3 guards (the gemini unguarded-denominator class): the 1 - tau+ denominator rejects
    tau+ >= 1 and tau+ < 0; the temperature rejects tau <= 1e-8; and the e^{-1/tau} floor binds when the
    raw correction would go non-positive."""
    g = np.array([1.0, 1.2, 0.9])
    for bad in (1.0, 1.5, -0.1):
        try:
            debiased_negative_mean(g, 1.0, bad, M3_TAU)
            raise AssertionError(f"tau+={bad} should have raised")
        except ValueError:
            pass
    try:
        estimator_g(_setup()["pool"], 0, 1e-9)
        raise AssertionError("tau <= 1e-8 should have raised")
    except ValueError:
        pass
    # force the floor: a huge positive mean drives the raw correction below e^{-1/tau}.
    floored = debiased_negative_mean(np.array([1.0, 1.0]), 1e6, 0.5, M3_TAU)
    assert abs(floored - np.exp(-1.0 / M3_TAU)) < 1e-12, "the e^{-1/tau} floor should bind"
    print("  [ok] Movement 3 guards: tau+ in [0,1), tau > 1e-8, and the e^{-1/tau} floor all hold")


def test_zero_drift_index_is_fresh() -> None:
    """Movement 4 collapse anchor: at the refresh step (t = refresh_t, drift alpha equal) the frozen
    index IS the fresh index — staleness overlap is exactly 1.0 and the stale gold recall equals the
    fresh gold recall. No drift, no staleness."""
    s = _setup()
    q0, d0, gold, T = s["ance_queries"], s["ance_docs"], s["ance_gold"], s["drift_T"]
    assert abs(staleness_overlap(q0, d0, T, 0.0, 0.0, K_STALE, DRIFT_TAU) - 1.0) < 1e-12, \
        "overlap at t = refresh_t must be 1.0"
    row0 = staleness_curve(q0, d0, gold, T, (0,), K_STALE, DRIFT_TAU)[0]
    assert row0["overlap"] == 1.0 and row0["stale_gold_r1"] == row0["fresh_gold_r1"], \
        "at t=0 the stale and fresh indices must agree exactly"
    print("  [ok] Movement 4 anchor: zero drift -> staleness overlap 1.0, stale == fresh")


def test_isometry_refreshed_index_is_vacuous() -> None:
    """Movement 4, the anti-trap witness (why the drift must be non-isometric AND the index must lag): a
    pure ROTATION applied consistently to both the queries and the documents — a refreshed index that
    happens to be an orthogonal re-encoding — leaves every mined set unchanged (overlap 1.0). Staleness
    exists only because the FROZEN index lags a NON-isometric encoder, not from drift per se."""
    s = _setup()
    q0, d0 = s["ance_queries"], s["ance_docs"]
    rng = np.random.default_rng(1)
    R, _ = np.linalg.qr(rng.standard_normal((d0.shape[1], d0.shape[1])))   # a random orthogonal map
    base = mine_with_index(d0, q0, K_STALE)
    rot = mine_with_index(normalize(d0 @ R), normalize(q0 @ R), K_STALE)
    overlaps = [len(set(base[i]) & set(rot[i])) / base.shape[1] for i in range(base.shape[0])]
    assert abs(float(np.mean(overlaps)) - 1.0) < 1e-12, \
        "an isometric (refreshed) re-encoding must not change any mined set"
    print("  [ok] Movement 4 anti-trap: an isometric refreshed index has zero staleness (overlap 1.0)")


def test_staleness_decays_and_hurts_recall() -> None:
    """Movement 4, the systems story: as steps-since-refresh grow the frozen index's overlap with the
    fresh encoder decays monotonically and substantially, and the stale index's gold recall@1 falls well
    below the fresh index's (which stays high). Assert the directions, not decimals."""
    s = _setup()
    q0, d0, gold, T = s["ance_queries"], s["ance_docs"], s["ance_gold"], s["drift_T"]
    curve = staleness_curve(q0, d0, gold, T, T_GRID, K_STALE, DRIFT_TAU)
    overlaps = [r["overlap"] for r in curve]
    assert all(overlaps[i] >= overlaps[i + 1] - 1e-9 for i in range(len(overlaps) - 1)), \
        f"staleness overlap should decay monotonically: {overlaps}"
    assert overlaps[-1] < overlaps[0] - 0.3, f"the decay should be substantial: {overlaps}"
    last = curve[-1]
    assert last["stale_gold_r1"] < last["fresh_gold_r1"] - 0.3, \
        f"a stale index should hurt gold recall: {last}"
    assert curve[0]["fresh_gold_r1"] > 0.9, "the fresh index should retrieve the gold (the ceiling)"
    print(f"  [ok] Movement 4: staleness decays {overlaps[0]} -> {overlaps[-1]}; stale gold r@1 "
          f"{last['stale_gold_r1']} << fresh {last['fresh_gold_r1']}")


def test_refresh_tradeoff_monotone() -> None:
    """Movement 4: the refresh interval R trades staleness against cost — a larger R is staler (lower
    average overlap) and cheaper (lower re-encode cost 1/R). Refresh-every-step (R=1) is the no-staleness,
    max-cost corner (the collapse anchor)."""
    s = _setup()
    q0, d0, T = s["ance_queries"], s["ance_docs"], s["drift_T"]
    rows = refresh_interval_tradeoff(q0, d0, T, R_GRID, HORIZON, K_STALE, DRIFT_TAU)
    stale = [r["avg_staleness"] for r in rows]
    cost = [r["reencode_cost"] for r in rows]
    assert all(stale[i] >= stale[i + 1] - 1e-9 for i in range(len(stale) - 1)), \
        f"larger R should not be fresher: {stale}"
    assert all(cost[i] >= cost[i + 1] - 1e-9 for i in range(len(cost) - 1)), \
        f"larger R should be cheaper: {cost}"
    assert abs(rows[0]["avg_staleness"] - 1.0) < 1e-9 and abs(rows[0]["reencode_cost"] - 1.0) < 1e-9, \
        "R=1 should be the no-staleness, full-cost corner"
    print(f"  [ok] Movement 4: refresh tradeoff — staleness {stale} vs cost {cost} (R=1 is the corner)")


def test_viz_constants_reproducible() -> None:
    """The bake-only-reproducible guard: every baked curve is bit-identical across two builds (the
    one-rng-stream + seeded-drift payoff)."""
    s = _setup()
    pool = s["pool"]
    a = tau_plus_curve(pool, K_MINE_GRID, M2_SEED)
    b = tau_plus_curve(pool, K_MINE_GRID, M2_SEED)
    assert a == b, "tau+ curve is not reproducible"
    c1 = debiased_convergence_curve(pool, M3_ANCHOR, M3_TAU, s["tau_plus"], N_GRID_M3, M3_TRIALS, M3_SEED)
    c2 = debiased_convergence_curve(pool, M3_ANCHOR, M3_TAU, s["tau_plus"], N_GRID_M3, M3_TRIALS, M3_SEED)
    assert c1 == c2, "debiased convergence curve is not reproducible"
    q0, d0, gold, T = s["ance_queries"], s["ance_docs"], s["ance_gold"], s["drift_T"]
    d1 = staleness_curve(q0, d0, gold, T, T_GRID, K_STALE, DRIFT_TAU)
    d2 = staleness_curve(q0, d0, gold, T, T_GRID, K_STALE, DRIFT_TAU)
    assert d1 == d2, "staleness curve is not reproducible"
    print("  [ok] reproducible: every baked curve is bit-identical across builds")


def test_guards() -> None:
    """Edge cases (the gemini unguarded-denominator class): empty / singleton pools, k capped and
    floored, an empty index — no ZeroDivisionError, sane sentinels throughout."""
    empty = {"emb": np.empty((0, 4)), "company": np.array([], dtype=int)}
    assert mine_nearest(empty, 0, 3).size == 0
    assert class_prior_tau_plus(empty) == 0.0
    single = {"emb": np.ones((1, 4)) / 2.0, "company": np.array([0])}
    assert mine_nearest(single, 0, 3).size == 0
    assert false_negative_rate(single, 0, 3, "mined") == 0.0
    s = _setup()
    pool = s["pool"]
    n = pool["emb"].shape[0]
    assert len(mine_nearest(pool, 0, 999)) == n - 1, "k should cap at the number of other items"
    assert len(mine_nearest(pool, 0, 0)) == 1, "k=0 should floor to 1"
    assert mine_with_index(np.empty((0, 4)), np.ones((2, 4)), 3).shape == (2, 0), "empty index -> empty rows"
    print("  [ok] guards: empty/singleton pools, k capped/floored, empty index all handled")


def _run_all() -> None:
    test_hard_negatives_dominate_gradient()
    test_hard_share_rises_as_tau_falls()
    test_mining_induces_false_negatives()
    test_random_rate_matches_prior()
    test_substrate_can_exhibit_false_negatives()
    test_debiased_identity_at_full_pool()
    test_debiased_closer_than_biased_under_sampling()
    test_debiased_tau_plus_zero_equals_biased()
    test_beta_recovers_infonce_weights()
    test_beta_concentrates_on_hard_negatives()
    test_debiased_guards()
    test_zero_drift_index_is_fresh()
    test_isometry_refreshed_index_is_vacuous()
    test_staleness_decays_and_hurts_recall()
    test_refresh_tradeoff_monotone()
    test_viz_constants_reproducible()
    test_guards()


# =========================================================================== #
# Demo — the headline numbers, printed.
# =========================================================================== #

def negative_sampling_demo() -> dict:
    """The headlines: hard negatives concentrate the gradient; mining them contaminates the batch with
    accidental positives; the debiased estimator recovers the true-negative expectation; and an ANCE
    index goes stale as the encoder drifts."""
    out = {}
    s = _setup()
    pool = s["pool"]
    cos, same = negative_set(pool, M1_ANCHOR)
    share = hard_gradient_share(cos, same, GRAD_TAU)
    loss = batch_loss_comparison(pool, M1_ANCHOR, M1_N_NEG, GRAD_TAU, M1_SEED)
    print(f"  MOVEMENT 1: @ tau={GRAD_TAU} the {int(same.sum())}/{len(cos)} hard negatives carry "
          f"{share:.3f} of the gradient; hard-batch loss {loss['hard_loss']} > random {loss['random_loss']}")
    out["m1"] = {"hard_share": share, **loss}

    curve = tau_plus_curve(pool, K_MINE_GRID, M2_SEED)
    print(f"  MOVEMENT 2: false-negative rate at k=1 — mined {curve[0]['mined']} vs random "
          f"{curve[0]['random']} (class prior {s['tau_plus']:.3f})")
    out["m2"] = curve

    conv = debiased_convergence_curve(pool, M3_ANCHOR, M3_TAU, s["tau_plus"], N_GRID_M3, M3_TRIALS, M3_SEED)
    print(f"  MOVEMENT 3: estimator MAE vs oracle — biased plateaus at {conv[-1]['biased_mae']}, "
          f"debiased converges {conv[0]['debiased_mae']} -> {conv[-1]['debiased_mae']}")
    out["m3"] = conv

    q0, d0, gold, T = s["ance_queries"], s["ance_docs"], s["ance_gold"], s["drift_T"]
    stale = staleness_curve(q0, d0, gold, T, T_GRID, K_STALE, DRIFT_TAU)
    print(f"  MOVEMENT 4: ANCE staleness — overlap {stale[0]['overlap']} -> {stale[-1]['overlap']}, "
          f"stale gold r@1 falls to {stale[-1]['stale_gold_r1']} (fresh {stale[-1]['fresh_gold_r1']})")
    out["m4"] = stale
    return out


# =========================================================================== #
# Viz constants — printed for NegativeSamplingLaboratory.tsx to mirror.
# =========================================================================== #

def _r(v, n=4):
    return round(float(v), n)


def viz_constants() -> None:
    """Print every MEASURED number the laboratory mirrors to the decimal. TS recomputes only CLOSED
    FORM: Panel A's softmax weights / entropy / top-1 from the baked cosines, Panel C's debiased bar
    (E_p - tau+ E_p+)/(1 - tau+) live as the tau+ slider moves, and Panel D's cost arithmetic. Every
    measured number (the false-negative curve, the convergence curve, the staleness decay) is baked
    here. numpy scalars are cast to avoid np.float64(...) pollution."""
    s = _setup()
    pool = s["pool"]
    n = pool["emb"].shape[0]

    print("  // ----- shared constants -----")
    print(f"  const N_POOL = {int(n)};")
    print(f"  const N_COMPANIES = {int(len(np.unique(pool['company'])))};")
    print(f"  const CLASS_PRIOR_TAU_PLUS = {_r(s['tau_plus'])};")
    print(f"  const GRAD_TAU = {_r(GRAD_TAU, 2)};")
    print(f"  const M3_TAU = {_r(M3_TAU, 2)};")

    print("  // ----- Panel A: the mixed negative set + hard gradient share (TS recomputes the softmax) -----")
    cos, same = negative_set(pool, M1_ANCHOR)
    order = np.argsort(-cos, kind="stable")              # show the batch hardest-first
    print(f"  const NEG_COS = {[_r(v) for v in cos[order]]};")
    print(f"  const NEG_SAME_SECTOR = {[bool(b) for b in same[order]]};")
    print(f"  const HARD_COUNT_FRACTION = {_r(float(np.mean(same)))};")
    print(f"  const HARD_SHARE_CURVE = {hard_share_curve(cos, same, sorted(TAU_GRID, reverse=True))};")
    s_rand = random_negative_cosines(pool, M1_ANCHOR, M1_N_NEG, M1_SEED)
    s_hard = hard_negative_cosines(pool, M1_ANCHOR, M1_N_NEG)
    print(f"  const RANDOM_BATCH_COS = {[_r(v) for v in s_rand]};")
    print(f"  const HARD_BATCH_COS = {[_r(v) for v in s_hard]};")
    print(f"  const BATCH_LOSS = {batch_loss_comparison(pool, M1_ANCHOR, M1_N_NEG, GRAD_TAU, M1_SEED)};")

    print("  // ----- Panel B: the false-negative rate as the mining radius tightens -----")
    print(f"  const K_MINE_GRID = {list(int(k) for k in K_MINE_GRID)};")
    print(f"  const TAU_PLUS_CURVE = {tau_plus_curve(pool, K_MINE_GRID, M2_SEED)};")

    print("  // ----- Panel C: biased / true / debiased convergence (TS recomputes the debiased bar) -----")
    g, is_pos = estimator_g(pool, M3_ANCHOR, M3_TAU)
    print(f"  const E_POS_MEAN = {_r(positive_mean(g, is_pos))};")
    print(f"  const E_BIASED_MEAN = {_r(biased_negative_mean(g))};   // E_p[g] over all items")
    print(f"  const E_ORACLE_MEAN = {_r(oracle_true_negative_mean(g, is_pos))};   // E_p-[g] true negatives")
    print(f"  const MI_FLOOR = {_r(float(np.exp(-1.0 / M3_TAU)))};   // the e^(-1/tau) clamp")
    print(f"  const N_GRID = {list(int(N) for N in N_GRID_M3)};")
    print(f"  const DEBIASED_CONVERGENCE = "
          f"{debiased_convergence_curve(pool, M3_ANCHOR, M3_TAU, s['tau_plus'], N_GRID_M3, M3_TRIALS, M3_SEED)};")

    print("  // ----- Panel D: ANCE staleness vs refresh interval (TS recomputes the cost knee) -----")
    q0, d0, gold, T = s["ance_queries"], s["ance_docs"], s["ance_gold"], s["drift_T"]
    print(f"  const ANCE_N_DOCS = {int(d0.shape[0])};")
    print(f"  const ANCE_N_QUERIES = {int(q0.shape[0])};")
    print(f"  const K_STALE = {int(K_STALE)};")
    print(f"  const DRIFT_TAU = {_r(DRIFT_TAU, 2)};")
    print(f"  const T_GRID = {list(int(t) for t in T_GRID)};")
    print(f"  const STALENESS_CURVE = {staleness_curve(q0, d0, gold, T, T_GRID, K_STALE, DRIFT_TAU)};")
    print(f"  const R_GRID = {list(int(R) for R in R_GRID)};")
    print(f"  const REFRESH_TRADEOFF = "
          f"{refresh_interval_tradeoff(q0, d0, T, R_GRID, HORIZON, K_STALE, DRIFT_TAU)};")


if __name__ == "__main__":
    print("Hard-negative mining & debiased contrastive training — diagnostics\n")
    _diagnostics()
    print("\nVerification harness:")
    _run_all()
    print("\nDemo:")
    negative_sampling_demo()
    print("\nViz constants (mirrored to the decimal in NegativeSamplingLaboratory.tsx):")
    viz_constants()
    print("\nAll checks passed.")
