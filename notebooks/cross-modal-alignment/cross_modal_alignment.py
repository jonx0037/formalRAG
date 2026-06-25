"""Cross-modal contrastive alignment and the modality gap — the reference implementation for
the formalRAG `cross-modal-alignment` topic.

InfoNCE taught a single dual encoder to place a query near its answer; dense retrieval read
that geometry as a maximum-inner-product lookup with a rank-d ceiling. A MULTIMODAL system
trains TWO encoders — a text tower and a chart tower — with the same symmetric contrastive
loss (CLIP; Radford et al. 2021), so a text query can retrieve a chart of the same company.
Liang et al. (2022) observed something unsettling about the result: even after training, the
two modalities occupy DISJOINT cones on the sphere — a measurable "modality gap" between the
text centroid and the chart centroid that contrastive training shrinks but does not close.
This module reads that gap three ways, and every pedagogical claim below is an `assert`:

  MOVEMENT 1 — WHAT THE GAP IS: an orthogonal (Pythagorean) decomposition (THEOREM). For
    paired unit embeddings z_text,i and z_chart,i, the per-pair difference d_i = z_text,i -
    z_chart,i has gap vector g = mean_i(d_i) = mean(z_text) - mean(z_chart) and modality gap
    ||g||. The alignment loss decomposes EXACTLY:
        L_align = mean ||d_i||^2 = ||g||^2 + mean ||d_i - g||^2 = gap^2 + dispersion,
    a Frobenius-Pythagoras split of the difference matrix D into its coherent rank-1 part
    1 g^T (the gap) and its orthogonal complement (dispersion). L_align IS the imported
    `alignment` byte-for-byte; gap^2 <= L_align follows because a projection is a contraction.
    (`cross_modal_corpus`, `decompose_alignment`, `modality_gap`.)

  MOVEMENT 2 — THE GAP IS A CALIBRATION ARTIFACT, INVISIBLE TO MIPS RANKING (THEOREM). Cross-
    modal scores S[i,j] = <t_i, c_j>. Offset every chart embedding by the shared gap,
    c_j -> c_j + alpha g; then S'[i,j] = S[i,j] + alpha <t_i, g>, and the added term is a
    PER-QUERY constant (independent of j). So the argsort over j, and hence recall@k, are
    EXACTLY invariant to the gap under maximum-inner-product search. The gap only shifts
    ABSOLUTE similarities — a temperature-scaled softmax, a fixed threshold, calibration.
    Honest split (rigorFlag): cosine retrieval RENORMALIZES c_j + alpha g, which breaks the
    per-query-constant cancellation, so cosine ranking is NOT gap-invariant and IMPROVES as
    the offset removes the gap (alpha -> 1). (`offset_keys`, `mips_recall_curve`,
    `cosine_recall_curve`, `mips_offset_invariance_witness`.)

  MOVEMENT 3 — WHERE THE GAP COMES FROM: the cone effect and temperature (MEASURED, rigorFlag).
    A vMF toy seeds two modality cones at "initialization" (separation beta). "Training" is a
    deterministic full-batch projected-GD descent on the SYMMETRIC CLIP loss (the imported
    `symmetric_inbatch_loss`, exact). We do NOT claim training preserves the gap — full-batch
    GD closes it at moderate temperature. The robust, seed-stable claim is the MONOTONE
    DIRECTION: lower temperature preserves a larger residual gap, with beta = 0 as the no-gap
    control. (`train_cross_modal`, `residual_gap_vs_tau`.)

Honest caveats (rigorFlag territory): the M1 decomposition and the M2 MIPS-invariance are exact
theorems; the cosine improvement under offset is a MEASURED curve (pinned to seeds). M3 is a
deterministic full-batch surrogate, NOT SGD — the persistent gap in real CLIP-style systems is
an SGD/finite-step/initialization phenomenon our surrogate does not model, and "closing the gap
hurts downstream" (Liang et al.) is NOT reproduced or asserted here. The numbers are a synthetic
vMF stand-in, not trained encoders.

This module imports its prerequisites and siblings — never reimplementing the sphere samplers,
the alignment/uniformity losses, the symmetric InfoNCE loss, or the MIPS recall. `viz_constants()`
prints what `CrossModalAlignmentLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/cross-modal-alignment/cross_modal_alignment.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.special import logsumexp

# Established cross-topic pattern: add each ancestor's HYPHENATED dir to the path, import its
# UNDERSCORED module. We reuse the sphere samplers (hypersphere), the alignment/uniformity and
# symmetric-loss machinery and the projected-GD step (infonce/dense), and the MIPS recall
# (dense) — never reimplementing them.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",            # normalize, sample_vmf, mean_resultant_length, mle_mu
    "the-retrieval-problem",               # cosine (transitive)
    "infonce-contrastive-objective",       # alignment, uniformity, info_nce_loss_batch, _sphere_step
    "dense-retrieval-dual-encoders",       # score_matrix, topk_recall, symmetric_inbatch_loss
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import (                                       # noqa: E402
    normalize, sample_vmf, mean_resultant_length, mle_mu,
)
from infonce_contrastive_objective import (                                  # noqa: E402
    alignment, uniformity, info_nce_loss_batch, _sphere_step, UNIF_T,
)
from dense_retrieval_dual_encoders import (                                  # noqa: E402
    score_matrix, topk_recall, symmetric_inbatch_loss,
)


# --------------------------------------------------------------------------- #
# Module constants — the synthetic geometry the panels step through.
#
# Two vMF(mu, kappa) draws have expected cosine A_d(kappa)^2, so at d=32 a kappa_sector=60
# makes same-sector companies genuinely NEAR (cosine ~0.55) yet resolvable, near-orthogonal
# sectors keep cross-sector documents near cosine 0, and kappa_view sets how tight each
# modality's view of a company is. beta is the SINGLE gap knob: 0 = no modality tilt (the
# no-gap control), larger = two disjoint cones. (The d=64 octave washes the cones out —
# A_64(60)^2 ~ 0.36 — so we stay at d <= 32, the CLAUDE.md kappa/d rule.)
# --------------------------------------------------------------------------- #

CM_SEED = 13
CM_DIM = 32                       # Movements 1 & 2: decomposition + the hard ranking corpus
CM_N_SECTORS = 4
CM_N_COMP = 6                     # 24 companies; same-sector companies genuinely compete
CM_KAPPA_SECTOR = 60.0
CM_KAPPA_VIEW = 120.0             # each view a vMF draw around its company direction
CM_BETA = 0.5                     # the gap knob (0 = no gap)
CM_GAP_SEED = 101                 # the two global modality axes m_text, m_chart
CM_K = 1                          # recall@k

CM_BETA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)        # Panel A: the decomposition sweep
CM_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)         # Panel B: offset in multiples of g

# Movement 3 — the training sandbox. Lower d keeps the residual-gap signal alive under
# full-batch GD (at d=32 the descent closes the gap more aggressively).
CM3_DIM = 16
CM3_N_COMP = 4                    # 16 companies
CM3_KAPPA_VIEW = 80.0
CM3_BETA = 0.5
CM_TRAIN_STEPS = 900
CM_TRAIN_LR = 0.5
CM_TAU_GRID = (0.04, 0.07, 0.1, 0.2, 0.3, 0.4)                 # straddles close/persist

CORPUS_HEADLINE = 1_000_000       # legible corpus size for the calibration narrative


# --------------------------------------------------------------------------- #
# Movement 1 — the two-view corpus and the orthogonal decomposition.
# --------------------------------------------------------------------------- #

def cross_modal_corpus(seed: int = CM_SEED, dim: int = CM_DIM, n_sectors: int = CM_N_SECTORS,
                       n_comp: int = CM_N_COMP, kappa_sector: float = CM_KAPPA_SECTOR,
                       kappa_view: float = CM_KAPPA_VIEW, beta: float = CM_BETA,
                       gap_seed: int = CM_GAP_SEED):
    """Two modality views (text, chart) per company, reusing the sectors->companies vMF geometry
    of `dpr_finance_matrix` but DOUBLED into two modality-tilted views (the established "reuse
    the geometry, build a topic-specific dataset" move — dpr_finance_matrix gives one document
    per company, we need two). Each company has a direction c_mu; its text view is a vMF draw
    around c_mu tilted toward the global text axis m_text by beta, its chart view a separate
    draw tilted toward m_chart. Returns (Z_text, Z_chart, sector_of_company, company_dirs); the
    truth pairing is the diagonal i <-> i."""
    rng = np.random.default_rng(seed)
    sector_mu = normalize(rng.standard_normal((n_sectors, dim)))
    grng = np.random.default_rng(gap_seed)
    m_text = normalize(grng.standard_normal(dim))     # the two cone axes, shared by all companies
    m_chart = normalize(grng.standard_normal(dim))
    Z_text, Z_chart, sector_of_company, company_dirs = [], [], [], []
    for si, mu in enumerate(sector_mu):
        members = sample_vmf(n_comp, mu, kappa_sector, seed=seed + 11 * si + 1)
        for ci in range(n_comp):
            c_mu = normalize(members[ci])
            company_dirs.append(c_mu)
            sector_of_company.append(si)
            t_raw = sample_vmf(1, c_mu, kappa_view, seed=seed + 211 + 7 * si + ci)[0]
            c_raw = sample_vmf(1, c_mu, kappa_view, seed=seed + 379 + 7 * si + ci)[0]
            Z_text.append(normalize((1.0 - beta) * t_raw + beta * m_text))
            Z_chart.append(normalize((1.0 - beta) * c_raw + beta * m_chart))
    return (np.array(Z_text), np.array(Z_chart),
            np.array(sector_of_company), np.array(company_dirs))


def gap_vector(Z_text: np.ndarray, Z_chart: np.ndarray) -> np.ndarray:
    """The modality-gap vector g = mean(z_text) - mean(z_chart): the difference of the two
    cone centroids, equivalently the mean of the per-pair differences d_i = z_text,i - z_chart,i."""
    return Z_text.mean(axis=0) - Z_chart.mean(axis=0)


def modality_gap(Z_text: np.ndarray, Z_chart: np.ndarray) -> float:
    """The modality gap ||g||: the distance between the text and chart cone centroids."""
    return float(np.linalg.norm(gap_vector(Z_text, Z_chart)))


def decompose_alignment(Z_text: np.ndarray, Z_chart: np.ndarray) -> dict[str, float]:
    """The orthogonal (Pythagorean) decomposition of the cross-modal alignment. With the
    difference matrix D (row i = d_i = z_text,i - z_chart,i) and gap vector g = mean(D),
    split D = 1 g^T + (D - 1 g^T) into a coherent rank-1 part and its orthogonal complement.
    Returns L_align = mean||d_i||^2 (== the imported `alignment`), gap2 = ||g||^2,
    dispersion = mean||d_i - g||^2, and the Frobenius cross term <coherent, dispersion>_F
    (exactly 0 — the orthogonality witness)."""
    D = Z_text - Z_chart
    g = D.mean(axis=0)
    coherent = np.ones((D.shape[0], 1)) * g[None, :]      # 1 g^T
    disp = D - coherent
    return {
        "L_align": float(np.mean(np.sum(D ** 2, axis=1))),
        "gap2": float(g @ g),
        "dispersion": float(np.mean(np.sum(disp ** 2, axis=1))),
        "ortho": float(np.sum(coherent * disp)),          # <coherent, dispersion>_F
    }


def decomposition_curve(beta_grid=CM_BETA_GRID, **kw):
    """Panel A: how the gap^2 / dispersion split of L_align grows with the modality tilt beta.
    Returns a list of {beta, L_align, gap2, dispersion, gap}."""
    rows = []
    for beta in beta_grid:
        Zt, Zc, _, _ = cross_modal_corpus(beta=beta, **kw)
        d = decompose_alignment(Zt, Zc)
        rows.append({
            "beta": round(float(beta), 3),
            "L_align": round(d["L_align"], 4),
            "gap2": round(d["gap2"], 4),
            "dispersion": round(d["dispersion"], 4),
            "gap": round(math.sqrt(d["gap2"]), 4),
        })
    return rows


def cone_projection(Z_text: np.ndarray, Z_chart: np.ndarray):
    """A 2D PCA projection of the two cones for Panel A's scatter: project the stacked
    embeddings onto the top-2 principal axes of the centered cloud. Returns (Pt, Pc, axis2d)
    with Pt, Pc the (n, 2) projected text/chart coordinates."""
    X = np.vstack([Z_text, Z_chart])
    if X.size == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((Z_text.shape[1], 2))
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    axis = Vt[:2].T                                       # (d, 2)
    n = Z_text.shape[0]
    P = Xc @ axis
    return P[:n], P[n:], axis


# --------------------------------------------------------------------------- #
# Movement 2 — gap-invariance of MIPS ranking; the gap is a calibration artifact.
# --------------------------------------------------------------------------- #

def offset_keys(Z_chart: np.ndarray, g: np.ndarray, alpha: float) -> np.ndarray:
    """Shift every chart embedding by alpha times the gap vector, c_j -> c_j + alpha g. At
    alpha = 1 the chart centroid moves onto the text centroid (the gap is removed). NOT
    renormalized — that is the point for the MIPS-invariance theorem; cosine renormalizes."""
    return Z_chart + alpha * g[None, :]


def mips_recall_curve(Z_text, Z_chart, truth, alphas=CM_ALPHA_GRID, k: int = CM_K):
    """Recall@k under MAXIMUM-INNER-PRODUCT scoring as the gap-removing offset alpha varies.
    The theorem: this is EXACTLY constant (a per-query constant cannot change the argsort)."""
    g = gap_vector(Z_text, Z_chart)
    return [topk_recall(score_matrix(Z_text, offset_keys(Z_chart, g, a)), truth, k) for a in alphas]


def cosine_recall_curve(Z_text, Z_chart, truth, alphas=CM_ALPHA_GRID, k: int = CM_K):
    """Recall@k under COSINE scoring (offset keys renormalized to the sphere). The honest
    contrast: cosine is NOT gap-invariant — renormalizing c_j + alpha g breaks the per-query
    cancellation, so recall moves and improves as the offset removes the gap."""
    g = gap_vector(Z_text, Z_chart)
    return [topk_recall(score_matrix(Z_text, normalize(offset_keys(Z_chart, g, a))), truth, k)
            for a in alphas]


def mips_offset_invariance_witness(Z_text, Z_chart, alpha: float = 1.0):
    """The exact-argsort invariance anchor: under the offset, the change in every score is the
    per-query constant alpha <t_i, g>, and the per-row argsort is bit-identical. Returns
    (max_deviation, argsort_identical)."""
    g = gap_vector(Z_text, Z_chart)
    S0 = score_matrix(Z_text, Z_chart)
    S1 = score_matrix(Z_text, offset_keys(Z_chart, g, alpha))
    added = S1 - S0
    expected = alpha * (Z_text @ g)                       # (n,) per-query constant
    max_dev = float(np.max(np.abs(added - expected[:, None])))
    same = all(np.array_equal(np.argsort(-S0[i], kind="stable"),
                              np.argsort(-S1[i], kind="stable")) for i in range(S0.shape[0]))
    return max_dev, same


def cross_modal_cosine(Z_text, Z_chart) -> float:
    """Mean diagonal cross-modal cosine <t_i, c_i> — the absolute similarity a fixed relevance
    threshold reads. The gap DEPRESSES this (the two cones sit apart), so a threshold calibrated
    within a modality over-rejects cross-modal pairs; removing the gap raises it."""
    return float(np.mean(np.sum(Z_text * Z_chart, axis=1)))


def threshold_pass_rate(Z_text, Z_chart, theta: float, alpha: float = 0.0) -> float:
    """Fraction of true text<->chart pairs whose (renormalized) cosine clears a relevance
    threshold theta after a gap-removing offset alpha. The calibration consequence of the gap:
    at alpha = 0 (gap present) many true pairs fall below theta; at alpha = 1 (gap removed)
    the pass rate rises, with the RANKING (MIPS recall) untouched throughout."""
    g = gap_vector(Z_text, Z_chart)
    Cc = normalize(offset_keys(Z_chart, g, alpha))
    cos = np.sum(Z_text * Cc, axis=1)
    return float(np.mean(cos >= theta))


# --------------------------------------------------------------------------- #
# Movement 3 — the cone effect and temperature (a deterministic training surrogate).
# --------------------------------------------------------------------------- #

def _raw_symmetric_loss(Z_text: np.ndarray, Z_chart: np.ndarray, tau: float) -> float:
    """The symmetric CLIP loss WITHOUT the imported loss's defensive renormalization — used only
    by the finite-difference gradient check, since the FD must perturb the ambient vectors and
    measure the Euclidean gradient (the shipped trainer feeds this Euclidean gradient to the
    imported `_sphere_step`, which does the tangent projection). At unit inputs this equals the
    imported `symmetric_inbatch_loss` exactly."""
    S = (Z_text @ Z_chart.T) / tau
    diag = np.diag(S)
    return 0.5 * (float(np.mean(logsumexp(S, axis=1) - diag)) + float(np.mean(logsumexp(S, axis=0) - diag)))


def _symmetric_cross_modal_grad(Z_text: np.ndarray, Z_chart: np.ndarray, tau: float):
    """Euclidean gradient of the SYMMETRIC cross-modal InfoNCE (CLIP) loss
    L = 1/2 (L_{t->c} + L_{c->t}) w.r.t. the two embedding blocks, with the diagonal as the
    positives. With S = (T C^T)/tau, P_row = softmax over j (columns), P_col = softmax over i
    (rows of the transpose), and M = P_row + P_col,
        grad_T = (1/(2 n tau)) (M C - 2 C),   grad_C = (1/(2 n tau)) (M^T T - 2 T).
    The loss returned equals the imported `symmetric_inbatch_loss` by construction. Returns
    (loss, grad_T, grad_C)."""
    n = Z_text.shape[0]
    S = (Z_text @ Z_chart.T) / tau
    lse_row = logsumexp(S, axis=1)
    lse_col = logsumexp(S, axis=0)
    diag = np.diag(S)
    loss = 0.5 * (float(np.mean(lse_row - diag)) + float(np.mean(lse_col - diag)))
    P_row = np.exp(S - lse_row[:, None])                  # rows sum to 1 (over j)
    P_col = np.exp(S - lse_col[None, :])                  # columns sum to 1 (over i)
    M = P_row + P_col
    grad_T = (1.0 / (2.0 * n * tau)) * (M @ Z_chart - 2.0 * Z_chart)
    grad_C = (1.0 / (2.0 * n * tau)) * (M.T @ Z_text - 2.0 * Z_text)
    return loss, grad_T, grad_C


def train_cross_modal(Z_text0, Z_chart0, tau: float, steps: int = CM_TRAIN_STEPS,
                      lr: float = CM_TRAIN_LR):
    """Deterministic full-batch projected gradient descent on the symmetric CLIP loss, reusing
    the imported sphere step `_sphere_step`. Returns the trained (Z_text, Z_chart)."""
    Zt, Zc = Z_text0.copy(), Z_chart0.copy()
    for _ in range(steps):
        _, gt, gc = _symmetric_cross_modal_grad(Zt, Zc, tau)
        Zt = _sphere_step(Zt, gt, lr)
        Zc = _sphere_step(Zc, gc, lr)
    return Zt, Zc


def residual_gap_vs_tau(taus=CM_TAU_GRID, dim: int = CM3_DIM, n_comp: int = CM3_N_COMP,
                        kappa_view: float = CM3_KAPPA_VIEW, beta: float = CM3_BETA,
                        steps: int = CM_TRAIN_STEPS, lr: float = CM_TRAIN_LR):
    """Panel C: the cone-effect / temperature link. Build the gap'd corpus once, then train at
    each temperature and report the residual gap, alignment, and union uniformity. The robust,
    seed-stable claim is the MONOTONE DIRECTION: lower tau preserves a larger residual gap.
    Returns a list of {tau, gap_before, gap_after, alignment, uniformity}."""
    Zt0, Zc0, _, _ = cross_modal_corpus(dim=dim, n_comp=n_comp, kappa_view=kappa_view, beta=beta)
    g0 = modality_gap(Zt0, Zc0)
    rows = []
    for tau in taus:
        Zt, Zc = train_cross_modal(Zt0, Zc0, tau, steps=steps, lr=lr)
        rows.append({
            "tau": round(float(tau), 3),
            "gap_before": round(g0, 4),
            "gap_after": round(modality_gap(Zt, Zc), 4),
            "alignment": round(alignment(Zt, Zc), 4),
            "uniformity": round(uniformity(np.vstack([Zt, Zc]), UNIF_T), 4),
        })
    return rows


# --------------------------------------------------------------------------- #
# Finance case study: text filings vs charts of the same companies.
# --------------------------------------------------------------------------- #

def finance_demo() -> dict:
    """The finance headline: a desk embeds 10-K text and price charts of the same companies
    with two towers. A modality gap opens between the text cone and the chart cone — yet the
    cross-modal RANKING (which chart a text query retrieves) is invisible to it under MIPS,
    while a fixed similarity THRESHOLD is corrupted by it. Made executable below."""
    Zt, Zc, sector, _ = cross_modal_corpus()
    truth = np.arange(Zt.shape[0])
    gap = modality_gap(Zt, Zc)
    dec = decompose_alignment(Zt, Zc)
    mips = mips_recall_curve(Zt, Zc, truth)
    cos = cosine_recall_curve(Zt, Zc, truth)
    cm_cos_gap = cross_modal_cosine(Zt, Zc)
    g = gap_vector(Zt, Zc)
    cm_cos_nogap = cross_modal_cosine(Zt, normalize(offset_keys(Zc, g, 1.0)))
    print(f"  {Zt.shape[0]} companies across {len(set(sector))} sectors, two views each "
          f"(text filing, price chart); synthetic vMF, not trained encoders")
    print(f"  modality gap ||g|| = {gap:.4f}; L_align = {dec['L_align']:.4f} "
          f"= gap^2 {dec['gap2']:.4f} + dispersion {dec['dispersion']:.4f} "
          f"(coherent {100 * dec['gap2'] / max(dec['L_align'], 1e-12):.0f}% of the misalignment)")
    print(f"  MIPS recall@1 under a gap-removing offset alpha = {[round(r, 3) for r in mips]} "
          f"(FLAT — the gap is invisible to ranking)")
    print(f"  cosine recall@1 under the same offset      = {[round(r, 3) for r in cos]} "
          f"(MOVES — peaks as the gap is removed)")
    print(f"  mean cross-modal cosine: {cm_cos_gap:.4f} with the gap -> {cm_cos_nogap:.4f} "
          f"without (a fixed relevance threshold reads this — the calibration cost)")
    return {
        "gap": gap, "L_align": dec["L_align"], "gap2": dec["gap2"],
        "mips_flat": float(max(mips) - min(mips)), "cos_gain": float(max(cos) - cos[0]),
        "cm_cos_gap": cm_cos_gap, "cm_cos_nogap": cm_cos_nogap,
    }


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_alignment_decomposition_is_orthogonal() -> None:
    """Movement 1 (THEOREM): L_align decomposes orthogonally into gap^2 + dispersion. L_align
    equals the imported `alignment` byte-for-byte; the coherent and dispersion parts are
    Frobenius-orthogonal; the identity is exact."""
    Zt, Zc, _, _ = cross_modal_corpus()
    dec = decompose_alignment(Zt, Zc)
    imported = alignment(Zt, Zc)
    assert abs(dec["L_align"] - imported) < 1e-12, f"L_align != imported alignment: {dec['L_align']} vs {imported}"
    assert abs(dec["L_align"] - (dec["gap2"] + dec["dispersion"])) < 1e-9, \
        f"decomposition not exact: {dec['L_align']} != {dec['gap2']} + {dec['dispersion']}"
    assert abs(dec["ortho"]) < 1e-9, f"coherent/dispersion not orthogonal: <.,.>_F = {dec['ortho']}"
    print(f"  [ok] L_align {dec['L_align']:.4f} = gap^2 {dec['gap2']:.4f} + dispersion "
          f"{dec['dispersion']:.4f} (orthogonal, <.,.>_F = {dec['ortho']:.2e}; == imported alignment)")


def test_gap_bounded_by_alignment() -> None:
    """Movement 1 corollary: gap^2 <= L_align (a projection is a contraction), with EQUALITY iff
    every per-pair difference d_i is identical (a rigid translation between the cones)."""
    Zt, Zc, _, _ = cross_modal_corpus()
    dec = decompose_alignment(Zt, Zc)
    assert dec["gap2"] <= dec["L_align"] + 1e-12, f"gap^2 {dec['gap2']} > L_align {dec['L_align']}"
    assert dec["dispersion"] > 1e-3, "generic corpus should have positive dispersion (strict inequality)"
    # Equality on a constructed rigid translation: chart = text shifted by a constant vector.
    rng = np.random.default_rng(0)
    T = normalize(rng.standard_normal((20, 16)))
    shift = rng.standard_normal(16)
    C = T + shift                                         # every d_i = -shift (identical)
    rigid = decompose_alignment(T, C)
    assert abs(rigid["gap2"] - rigid["L_align"]) < 1e-9, \
        f"rigid translation should saturate the bound: {rigid['gap2']} != {rigid['L_align']}"
    assert rigid["dispersion"] < 1e-9, f"rigid translation should have zero dispersion: {rigid['dispersion']}"
    print(f"  [ok] gap^2 {dec['gap2']:.4f} <= L_align {dec['L_align']:.4f} (strict; dispersion "
          f"{dec['dispersion']:.4f}); rigid translation saturates it (dispersion {rigid['dispersion']:.2e})")


def test_mips_ranking_is_gap_invariant() -> None:
    """Movement 2 (THE HEADLINE THEOREM): under MIPS the gap-removing offset changes every
    score by the per-query constant alpha <t_i, g>, leaving the argsort bit-identical, so
    recall@k is EXACTLY constant across the offset sweep."""
    Zt, Zc, _, _ = cross_modal_corpus()
    truth = np.arange(Zt.shape[0])
    max_dev, same = mips_offset_invariance_witness(Zt, Zc, alpha=1.0)
    assert max_dev < 1e-12, f"score change is not a per-query constant: max dev {max_dev}"
    assert same, "argsort changed under the offset (it must not)"
    mips = mips_recall_curve(Zt, Zc, truth)
    assert max(mips) - min(mips) < 1e-12, f"MIPS recall is not flat under the offset: {mips}"
    assert 0.0 < mips[0] < 1.0, f"corpus must be HARD (recall in (0,1)), got {mips[0]}"
    print(f"  [ok] MIPS recall@1 flat at {mips[0]:.4f} across alpha (max dev {max_dev:.2e}, "
          f"argsort identical) — the gap is invisible to ranking")


def test_cosine_is_not_gap_invariant() -> None:
    """Movement 2, the honest split: cosine retrieval renormalizes the offset keys, so it is
    NOT gap-invariant — recall moves and IMPROVES as the offset (alpha -> 1) removes the gap,
    and the absolute cross-modal cosine a threshold reads rises when the gap is removed."""
    Zt, Zc, _, _ = cross_modal_corpus()
    truth = np.arange(Zt.shape[0])
    cos = cosine_recall_curve(Zt, Zc, truth)
    assert max(cos) - min(cos) > 0.05, f"cosine recall should move under the offset: {cos}"
    assert cos[-1] >= cos[0] - 1e-9 and max(cos) > cos[0] + 1e-9, \
        f"removing the gap should not hurt and should help cosine recall: {cos}"
    # The calibration reading: mean cross-modal cosine rises once the gap is removed.
    g = gap_vector(Zt, Zc)
    cm_gap = cross_modal_cosine(Zt, Zc)
    cm_nogap = cross_modal_cosine(Zt, normalize(offset_keys(Zc, g, 1.0)))
    assert cm_nogap > cm_gap + 1e-3, f"removing the gap should raise cross-modal cosine: {cm_gap} -> {cm_nogap}"
    print(f"  [ok] cosine recall@1 moves {[round(r, 3) for r in cos]} (peaks as gap removed); "
          f"cross-modal cosine {cm_gap:.4f} -> {cm_nogap:.4f} (calibration)")


def test_symmetric_loss_matches_import() -> None:
    """Movement 3 anchor: the trainer's loss is the imported `symmetric_inbatch_loss`
    byte-for-byte, and one direction is the imported `info_nce_loss_batch` — the CLIP loss is
    reused, never reimplemented as a competing definition."""
    Zt, Zc, _, _ = cross_modal_corpus(dim=CM3_DIM, n_comp=CM3_N_COMP, kappa_view=CM3_KAPPA_VIEW)
    tau = 0.1
    loss, _, _ = _symmetric_cross_modal_grad(Zt, Zc, tau)
    imported = symmetric_inbatch_loss(Zt, Zc, tau)
    assert abs(loss - imported) < 1e-12, f"trainer loss != imported symmetric loss: {loss} vs {imported}"
    one_dir = float(np.mean(logsumexp((Zt @ Zc.T) / tau, axis=1) - np.diag((Zt @ Zc.T) / tau)))
    assert abs(one_dir - info_nce_loss_batch(Zt, Zc, tau)) < 1e-12, "one-direction != imported info_nce_loss_batch"
    print(f"  [ok] trainer loss {loss:.4f} == imported symmetric_inbatch_loss (one-direction == info_nce_loss_batch)")


def test_cross_modal_gradient_finite_difference() -> None:
    """Movement 3 anchor: the symmetric CLIP gradient w.r.t. both blocks matches a central
    finite difference of the raw (ambient) loss — the Euclidean gradient the trainer projects
    with `_sphere_step`."""
    rng = np.random.default_rng(3)
    n, d, tau = 7, 8, 0.15
    Zt = normalize(rng.standard_normal((n, d)))
    Zc = normalize(rng.standard_normal((n, d)))
    _, gT, gC = _symmetric_cross_modal_grad(Zt, Zc, tau)
    eps = 1e-6
    for (grad, which) in ((gT, "text"), (gC, "chart")):
        for i in range(n):
            for k in range(d):
                Zp, Zm, Wp, Wm = Zt.copy(), Zt.copy(), Zc.copy(), Zc.copy()
                if which == "text":
                    Zp[i, k] += eps; Zm[i, k] -= eps
                    fp = _raw_symmetric_loss(Zp, Zc, tau); fm = _raw_symmetric_loss(Zm, Zc, tau)
                else:
                    Wp[i, k] += eps; Wm[i, k] -= eps
                    fp = _raw_symmetric_loss(Zt, Wp, tau); fm = _raw_symmetric_loss(Zt, Wm, tau)
                fd = (fp - fm) / (2 * eps)
                assert abs(fd - grad[i, k]) < 1e-5, f"{which} grad FD mismatch at ({i},{k}): {fd} vs {grad[i, k]}"
    print("  [ok] symmetric CLIP gradient matches finite differences of the raw loss (both blocks)")


def test_beta_zero_is_no_gap_control() -> None:
    """Movement 1/3 control: with no modality tilt (beta = 0) the gap is only Monte-Carlo
    residual (the per-company differences are mean-zero), so it is far below the beta = 0.5 gap;
    and identical views collapse the alignment and the gap to exactly zero."""
    Zt0, Zc0, _, _ = cross_modal_corpus(beta=0.0)
    Zt5, Zc5, _, _ = cross_modal_corpus(beta=0.5)
    g0, g5 = modality_gap(Zt0, Zc0), modality_gap(Zt5, Zc5)
    assert g0 < 0.35, f"beta=0 gap should be small (no systematic tilt), got {g0}"
    assert g5 > 3 * g0, f"beta=0.5 gap should dominate the beta=0 residual: {g5} vs {g0}"
    # Identical views: the exact collapse anchor.
    dec = decompose_alignment(Zt5, Zt5)
    assert dec["L_align"] < 1e-12 and dec["gap2"] < 1e-12, f"identical views must give zero gap/align: {dec}"
    print(f"  [ok] no-gap control: beta=0 gap {g0:.4f} << beta=0.5 gap {g5:.4f}; "
          f"identical views -> L_align {dec['L_align']:.2e}, gap^2 {dec['gap2']:.2e}")


def test_training_lowers_alignment() -> None:
    """Movement 3 sanity: full-batch contrastive training reduces the cross-modal alignment loss
    (pulls paired views together) from its initialization."""
    Zt0, Zc0, _, _ = cross_modal_corpus(dim=CM3_DIM, n_comp=CM3_N_COMP, kappa_view=CM3_KAPPA_VIEW)
    before = alignment(Zt0, Zc0)
    Zt, Zc = train_cross_modal(Zt0, Zc0, tau=0.2)
    after = alignment(Zt, Zc)
    assert after < before, f"training should lower alignment: {before} -> {after}"
    print(f"  [ok] training lowers alignment: {before:.4f} -> {after:.4f}")


def test_temperature_preserves_larger_residual_gap() -> None:
    """Movement 3 (MEASURED, the robust direction): lower temperature preserves a LARGER
    residual modality gap after training; at moderate/high tau full-batch GD closes it. We
    assert the MONOTONE DIRECTION (never the decimals, and never "training preserves the gap")."""
    rows = residual_gap_vs_tau()                          # taus ascending
    after = [r["gap_after"] for r in rows]
    assert all(after[i] >= after[i + 1] - 0.03 for i in range(len(after) - 1)), \
        f"residual gap should be non-increasing as tau rises: {after}"
    assert after[0] > after[-1] + 0.1, f"low tau should leave a clearly larger residual gap: {after}"
    assert after[-1] < 0.3, f"high tau should close the gap substantially: {after[-1]}"
    print(f"  [ok] residual gap monotone in temperature (tau {[r['tau'] for r in rows]}): "
          f"{after} — lower tau preserves a larger gap; high tau closes it")


def test_empty_and_degenerate_guards() -> None:
    """Defensive guards: a single-company corpus has no off-diagonal negatives (uniformity and
    the loss need n >= 2), and tiny corpora still decompose."""
    Zt, Zc, _, _ = cross_modal_corpus(n_sectors=1, n_comp=2)
    dec = decompose_alignment(Zt, Zc)
    assert dec["L_align"] >= 0.0 and dec["gap2"] >= 0.0
    # decompose handles a 2-row corpus; gap-invariance witness still exact.
    max_dev, same = mips_offset_invariance_witness(Zt, Zc, alpha=2.0)
    assert max_dev < 1e-12 and same
    print("  [ok] degenerate guards: tiny corpus decomposes; MIPS invariance still exact")


def _run_all() -> None:
    test_mips_ranking_is_gap_invariant()                 # the headline runs first
    test_alignment_decomposition_is_orthogonal()
    test_gap_bounded_by_alignment()
    test_cosine_is_not_gap_invariant()
    test_symmetric_loss_matches_import()
    test_cross_modal_gradient_finite_difference()
    test_beta_zero_is_no_gap_control()
    test_training_lowers_alignment()
    test_temperature_preserves_larger_residual_gap()
    test_empty_and_degenerate_guards()


# --------------------------------------------------------------------------- #
# Diagnostics — tuning visibility (not asserts).
# --------------------------------------------------------------------------- #

def _diagnostics() -> None:
    """Print the tuning numbers behind the parameter choices (kappa/d separation, the
    decomposition sweep, the temperature ladder) — the build-and-run record."""
    Zt, Zc, sector, comp = cross_modal_corpus()
    print("  -- M1/M2 corpus (d=32, n_comp=6):")
    print(f"     gap = {modality_gap(Zt, Zc):.4f}; decomposition = {decompose_alignment(Zt, Zc)}")
    # Same-sector vs cross-sector company separation (the kappa/d check).
    G = comp @ comp.T
    same = np.array([[sector[i] == sector[j] and i != j for j in range(len(sector))]
                     for i in range(len(sector))])
    cross = np.array([[sector[i] != sector[j] for j in range(len(sector))]
                      for i in range(len(sector))])
    print(f"     same-sector company cosine ~ {G[same].mean():.3f}, cross-sector ~ {G[cross].mean():.3f}")
    print(f"     view concentration A_d(kappa_view) ~ {mean_resultant_length(CM_DIM, CM_KAPPA_VIEW):.3f}")
    # The empirical cone centers via the imported vMF mean-direction MLE.
    mt, _ = mle_mu(Zt); mc, _ = mle_mu(Zc)
    print(f"     estimated cone-axis separation (1 - <mu_text_hat, mu_chart_hat>) = {1 - mt @ mc:.3f}")
    print("  -- decomposition sweep over beta:")
    for r in decomposition_curve():
        print(f"     beta={r['beta']:.1f}: L_align={r['L_align']:.3f} gap^2={r['gap2']:.3f} "
              f"dispersion={r['dispersion']:.3f} gap={r['gap']:.3f}")
    print("  -- offset sweep (alpha): MIPS flat, cosine moves:")
    truth = np.arange(Zt.shape[0])
    print(f"     MIPS   = {[round(r, 3) for r in mips_recall_curve(Zt, Zc, truth)]}")
    print(f"     cosine = {[round(r, 3) for r in cosine_recall_curve(Zt, Zc, truth)]}")
    print("  -- temperature ladder (M3 sandbox, d=16):")
    for r in residual_gap_vs_tau():
        print(f"     tau={r['tau']:.2f}: gap {r['gap_before']:.3f} -> {r['gap_after']:.3f}, "
              f"align={r['alignment']:.3f}, uniformity={r['uniformity']:.3f}")


# --------------------------------------------------------------------------- #
# Viz constants — printed for CrossModalAlignmentLaboratory.tsx to mirror.
# --------------------------------------------------------------------------- #

def viz_constants() -> None:
    """Print every MEASURED number the laboratory mirrors to the decimal. TS recomputes only
    CLOSED FORM: Panel A's gap^2/dispersion split from the baked clouds, Panel B's MIPS recall
    (argsort-invariant, so it does not move) from the baked scores, Panel C reads the baked
    temperature ladder. numpy scalars are cast to avoid np.float64(...) pollution."""
    Zt, Zc, sector, _ = cross_modal_corpus()
    truth = np.arange(Zt.shape[0])

    print("  PANEL A — decomposition over beta + a 2D cone projection at beta=0.5:")
    print(f"    DECOMP = {decomposition_curve()}")
    Pt, Pc, _ = cone_projection(Zt, Zc)
    print(f"    PROJ_TEXT  = {[[round(float(x), 3) for x in p] for p in Pt]}")
    print(f"    PROJ_CHART = {[[round(float(x), 3) for x in p] for p in Pc]}")
    print(f"    SECTOR = {[int(s) for s in sector]}")

    print("  PANEL B — MIPS-flat vs cosine-moves under the gap-removing offset:")
    print(f"    ALPHA_GRID = {[round(float(a), 3) for a in CM_ALPHA_GRID]}")
    print(f"    MIPS_RECALL = {[round(float(r), 4) for r in mips_recall_curve(Zt, Zc, truth)]}")
    print(f"    COSINE_RECALL = {[round(float(r), 4) for r in cosine_recall_curve(Zt, Zc, truth)]}")
    # Panel B recomputes MIPS recall live in TS from the baked score matrix + truth; bake them.
    print(f"    SCORES = {[[round(float(x), 4) for x in row] for row in score_matrix(Zt, Zc)]}")
    print(f"    GAP_VEC_SCORE = {[round(float(x), 4) for x in (Zt @ gap_vector(Zt, Zc))]}")
    print(f"    TRUTH = {[int(t) for t in truth]}")

    print("  PANEL C — cone effect & temperature (M3 sandbox, d=16):")
    print(f"    TAU_GRID = {[round(float(t), 3) for t in CM_TAU_GRID]}")
    print(f"    RESIDUAL_GAP = {residual_gap_vs_tau()}")


if __name__ == "__main__":
    print("Cross-modal alignment verification harness")
    _run_all()
    print("Finance demo:")
    finance_demo()
    print("Diagnostics:")
    _diagnostics()
    print("Viz constants (mirrored to the decimal in CrossModalAlignmentLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
