"""High-dimensional geometry and the concentration of distances — the reference
implementation for the formalRAG `high-dimensional-geometry` topic.

A RAG system retrieves by distance or similarity in R^d, with d in the hundreds to
low thousands. Our intuition for "near" and "far" comes from d = 2, 3 and is wrong
in high dimensions. This module establishes — and verifies — three concentration
phenomena, then the systems-aware twist that keeps the topic honest:

  1. THIN SHELL. For x ~ N(0, I_d), ||x||^2 has mean d and variance 2d, so
     ||x|| / sqrt(d) -> 1: the mass of a high-d Gaussian sits on a thin shell of
     radius sqrt(d), nowhere near the origin where its density peaks.
  2. NEAR-ORTHOGONALITY. Two independent random unit vectors have inner product of
     mean 0 and variance 1/d, so the angle between them concentrates at 90 degrees.
  3. DISTANCE CONCENTRATION (the curse). For i.i.d. coordinates the relative
     variance Var(D^2)/E[D^2]^2 -> 0, so all pairwise distances concentrate; the
     nearest and farthest neighbor of a query become indistinguishable
     (Beyer et al. 1999) and exact nearest-neighbor search loses meaning.
  4. INTRINSIC DIMENSION (why retrieval still works). The curse assumes data that
     fill R^d. Real embeddings lie near a low-dimensional manifold, and contrast is
     governed by the *intrinsic* dimension k, not the ambient d. We recover k with
     the TwoNN estimator (Facco et al. 2017) and show structured data keeps its
     contrast at d = 1536 while i.i.d. data loses it.

Every pedagogical claim the topic makes is an `assert` below. The closed-form
targets are exact: for x ~ N(0, I_d), E||x||^2 = d and Var||x||^2 = 2d; for v
uniform on the sphere, Var<u, v> = 1/d; for Gaussian data, the relative variance of
squared distance is exactly 2/d. `grid_table()` prints the numbers that
`ConcentrationLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/high-dimensional-geometry/high_dimensional_geometry.py
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2

# The dimension grid every panel of the viz steps through.
GRID: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)


# --------------------------------------------------------------------------- #
# Samplers — three data models. Each is seeded and deterministic.
# --------------------------------------------------------------------------- #

def sample_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n points from the standard Gaussian N(0, I_d)."""
    return np.random.default_rng(seed).standard_normal((n, d))


def sample_sphere(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n points uniform on the unit sphere S^{d-1} (normalized Gaussians)."""
    x = np.random.default_rng(seed).standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def sample_cube(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n points uniform on the cube [0, 1]^d."""
    return np.random.default_rng(seed).random((n, d))


# --------------------------------------------------------------------------- #
# Distances and the relative contrast (Beyer et al.).
# --------------------------------------------------------------------------- #

def _query_distances(data: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Euclidean distances from a single query point to every data point."""
    return np.linalg.norm(data - query, axis=1)


def relative_contrast(d: int, n: int = 200, trials: int = 30, seed: int = 0,
                      sampler=sample_gaussian) -> float:
    """Mean relative contrast (D_max - D_min) / D_min of query-to-point distances.

    This is the quantity Beyer et al. (1999) show tends to 0 in high dimensions:
    once it vanishes, the nearest and farthest neighbors are indistinguishable.
    Averaged over `trials` independent (query + data) draws for a stable readout.
    """
    vals = []
    for t in range(trials):
        pts = sampler(n + 1, d, seed=seed + t)
        dist = _query_distances(pts[1:], pts[0])
        dmin = float(dist.min())
        if dmin > 0:
            vals.append((float(dist.max()) - dmin) / dmin)
    return float(np.mean(vals)) if vals else float("nan")  # nan: degenerate (e.g. d=1 sphere)


def squared_distance_relative_variance(d: int, n: int = 4000, seed: int = 0) -> float:
    """Var(D^2) / E[D^2]^2 over pairs of Gaussian points — the proved core.

    For X, Y ~ N(0, I_d) this equals exactly 2/d (D^2 ~ 2 * chi2_d, so E = 2d,
    Var = 8d, ratio = 2/d): the relative variance vanishes, hence all distances
    concentrate at their common mean.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, d))
    b = rng.standard_normal((n, d))
    d2 = np.sum((a - b) ** 2, axis=1)
    return float(np.var(d2) / np.mean(d2) ** 2)


# --------------------------------------------------------------------------- #
# Norm concentration (the thin shell) and near-orthogonality.
# --------------------------------------------------------------------------- #

def norm_concentration_stats(d: int, n: int = 4000, seed: int = 0) -> dict[str, float]:
    """Empirical moments of ||x||^2 / d and the shell radius ||x|| / sqrt(d)."""
    x = sample_gaussian(n, d, seed)
    sqnorm = np.sum(x ** 2, axis=1)            # ~ chi2_d
    shell = np.sqrt(sqnorm) / math.sqrt(d)     # -> 1, with vanishing spread
    return {
        "mean_sqnorm_over_d": float(np.mean(sqnorm / d)),   # -> 1
        "var_sqnorm_over_d": float(np.var(sqnorm / d)),     # -> 2/d
        "shell_mean": float(np.mean(shell)),                # -> 1
        "shell_std": float(np.std(shell)),                  # -> 0
    }


def inner_product_stats(d: int, n: int = 4000, seed: int = 0) -> dict[str, float]:
    """Empirical moments of <u, v> for independent random unit vectors."""
    u = sample_sphere(n, d, seed)
    v = sample_sphere(n, d, seed + 1)
    ip = np.sum(u * v, axis=1)
    return {
        "mean": float(np.mean(ip)),            # -> 0
        "var": float(np.var(ip)),              # -> 1/d
        "mean_abs": float(np.mean(np.abs(ip))),  # -> ~ sqrt(2/(pi d))
    }


# --------------------------------------------------------------------------- #
# Volume: the d-ball and its thin outer shell.
# --------------------------------------------------------------------------- #

def ball_volume(d: int) -> float:
    """Volume of the unit ball in R^d: pi^{d/2} / Gamma(d/2 + 1)."""
    return math.exp((d / 2) * math.log(math.pi) - gammaln(d / 2 + 1))


def shell_fraction(d: int, eps: float) -> float:
    """Fraction of the unit ball's volume within `eps` of the surface: 1-(1-eps)^d."""
    return 1.0 - (1.0 - eps) ** d


# --------------------------------------------------------------------------- #
# Intrinsic dimension: structured (low-rank) vs i.i.d. data, and TwoNN.
# --------------------------------------------------------------------------- #

def iid_data(n: int, d: int, seed: int = 0) -> np.ndarray:
    """i.i.d. Gaussian data that genuinely fills R^d (intrinsic dim = ambient d)."""
    return sample_gaussian(n, d, seed)


def structured_data(n: int, d: int, k: int, noise: float = 0.01,
                    seed: int = 0) -> np.ndarray:
    """n points near a random k-dim subspace of R^d, plus small ambient noise.

    A self-contained stand-in for real embeddings: a k-dimensional Gaussian latent
    isometrically embedded into R^d (orthonormal rows), so pairwise distances are
    governed by the intrinsic dimension k, not the ambient d. NOT a trained encoder.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k))                  # k-dim latent
    basis_cols, _ = np.linalg.qr(rng.standard_normal((d, k)))  # d x k orthonormal
    return z @ basis_cols.T + noise * rng.standard_normal((n, d))


def twonn_intrinsic_dim(x: np.ndarray, discard_fraction: float = 0.1) -> float:
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017).

    For data locally uniform on a k-manifold, the ratio mu = r2/r1 of each point's
    second- to first-nearest-neighbor distance satisfies P(mu > t) = t^{-k}, so
    -log(1 - F(mu)) = k * log(mu): a line through the origin whose slope is k. We
    discard the top `discard_fraction` of ratios (the unstable tail) before fitting.
    """
    sq = np.sum(x ** 2, axis=1)
    gram = x @ x.T
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * gram, 0.0)
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(np.partition(d2, 1, axis=1)[:, :2])
    dist.sort(axis=1)
    r1, r2 = dist[:, 0], dist[:, 1]
    mu = r2 / r1
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1.0)])
    m = len(mu)
    f_emp = np.arange(1, m + 1) / m
    cut = int((1.0 - discard_fraction) * m)
    log_mu = np.log(mu[:cut])
    y = -np.log(1.0 - f_emp[:cut])
    return float(np.sum(log_mu * y) / np.sum(log_mu * log_mu))  # slope through origin


# --------------------------------------------------------------------------- #
# The grid table the viz mirrors, and the finance demo.
# --------------------------------------------------------------------------- #

def grid_table() -> list[dict[str, float]]:
    """Per-dimension summary statistics — the numbers ConcentrationLaboratory.tsx
    mirrors to the decimal. Deterministic (fixed seeds)."""
    rows = []
    for d in GRID:
        ip = inner_product_stats(d, n=4000, seed=1)
        nc = norm_concentration_stats(d, n=4000, seed=2)
        rows.append({
            "d": d,
            "contrast": relative_contrast(d, n=200, trials=30, seed=3),
            "ip_var": ip["var"],
            "ip_var_theory": 1.0 / d,
            "shell_mean": nc["shell_mean"],
            "shell_std": nc["shell_std"],
            "shell_frac_10pct": shell_fraction(d, 0.10),
            "ball_volume": ball_volume(d),
        })
    return rows


# Finance case study: ~1536-d financial-document embeddings. The naive worry is
# that cosine is meaningless at 1536 dimensions; the resolution is that structured
# (low-intrinsic-dimension) data keeps its contrast where i.i.d. data destroys it.
FINANCE_DIM = 1536
FINANCE_INTRINSIC_K = 10


def finance_demo() -> dict[str, float]:
    """Structured vs i.i.d. data at the production embedding dimension."""
    d, k = FINANCE_DIM, FINANCE_INTRINSIC_K
    struct = structured_data(800, d, k, noise=0.01, seed=4)
    noise = iid_data(800, d, seed=5)
    out = {
        "structured_contrast": relative_contrast(d, n=200, trials=30, seed=6,
                                                 sampler=lambda n, dd, seed: structured_data(n, dd, k, 0.01, seed)),
        "iid_contrast": relative_contrast(d, n=200, trials=30, seed=7),
        "structured_twonn": twonn_intrinsic_dim(struct),
        "iid_twonn": twonn_intrinsic_dim(noise),
    }
    print(f"  embeddings in R^{d} (intrinsic k = {k} for the structured set):")
    print(f"  {'set':<14}{'relative contrast':<20}{'TwoNN intrinsic dim':<22}")
    print(f"  {'structured':<14}{out['structured_contrast']:<20.4f}{out['structured_twonn']:<22.2f}")
    print(f"  {'i.i.d. (R^d)':<14}{out['iid_contrast']:<20.4f}{out['iid_twonn']:<22.2f}")
    print("  -> structured data keeps its contrast (NN is meaningful) and TwoNN "
          "recovers the low intrinsic dim; i.i.d. data loses contrast.")
    return out


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_chi_square_moments() -> None:
    """E||x||^2 = d and Var||x||^2 = 2d, matching scipy's chi2 exactly, and the
    empirical Gaussian norm agrees."""
    for d in (1, 2, 10, 100, 1000):
        assert math.isclose(chi2.mean(d), d), f"chi2 mean != d at d={d}"
        assert math.isclose(chi2.var(d), 2 * d), f"chi2 var != 2d at d={d}"
        nc = norm_concentration_stats(d, n=8000, seed=0)
        assert abs(nc["mean_sqnorm_over_d"] - 1.0) < 0.05, f"E||x||^2/d != 1 at d={d}"
    print("  [ok] chi-square moments: E||x||^2 = d, Var||x||^2 = 2d (scipy + empirical)")


def test_norm_concentration() -> None:
    """The shell radius ||x||/sqrt(d) concentrates at 1: its spread shrinks
    monotonically as d grows (the thin shell)."""
    stds = [norm_concentration_stats(d, n=6000, seed=0)["shell_std"] for d in GRID]
    for a, b in zip(stds, stds[1:]):
        assert b < a + 1e-3, f"shell spread not shrinking: {a:.4f} -> {b:.4f}"
    assert stds[-1] < 0.05, f"shell still wide at d={GRID[-1]}: {stds[-1]:.4f}"
    # var(||x||^2/d) tracks the theoretical 2/d.
    for d in (50, 200, 1000):
        v = norm_concentration_stats(d, n=20000, seed=1)["var_sqnorm_over_d"]
        assert abs(v - 2.0 / d) < 0.5 * (2.0 / d), f"var||x||^2/d off 2/d at d={d}"
    print(f"  [ok] thin shell: ||x||/sqrt(d) spread shrinks to {stds[-1]:.4f} at d={GRID[-1]}")


def test_near_orthogonality() -> None:
    """<u, v> for random unit vectors concentrates at 0 with variance 1/d, and the
    mean angle approaches 90 degrees (mean |<u,v>| shrinks)."""
    prev_abs = None
    for d in GRID:
        s = inner_product_stats(d, n=8000, seed=0)
        assert abs(s["mean"]) < 0.05, f"E<u,v> not ~0 at d={d}"
        if d >= 5:  # 1/d variance estimate is reliable once d is not tiny
            assert abs(s["var"] - 1.0 / d) < 0.25 * (1.0 / d), f"Var<u,v> off 1/d at d={d}"
        if prev_abs is not None:
            assert s["mean_abs"] < prev_abs + 1e-3, "mean |<u,v>| not shrinking"
        prev_abs = s["mean_abs"]
    print("  [ok] near-orthogonality: Var<u,v> = 1/d, angle -> 90 degrees")


def test_distance_concentration() -> None:
    """Relative contrast (D_max - D_min)/D_min vanishes as d grows (Beyer et al.
    1999), for all three data models. Anchored on the stable tail d >= 20 (the
    metric is huge and noisy at tiny d, where D_min can be arbitrarily small)."""
    tail = [d for d in GRID if d >= 20]
    for sampler, name in ((sample_gaussian, "gaussian"),
                          (sample_cube, "cube"),
                          (sample_sphere, "sphere")):
        c = [relative_contrast(d, sampler=sampler, seed=10) for d in tail]
        for a, b in zip(c, c[1:]):
            assert b < a + 1e-2, f"contrast not falling ({name}): {a:.3f} -> {b:.3f}"
        assert c[-1] < 0.2, f"contrast still high ({name}) at d={tail[-1]}: {c[-1]:.3f}"
        assert c[0] > 3 * c[-1], f"contrast drop too small ({name}): {c[0]:.3f} -> {c[-1]:.3f}"
    print("  [ok] distance concentration: relative contrast -> 0 (gaussian/cube/sphere)")


def test_relative_variance_vanishes() -> None:
    """Var(D^2)/E[D^2]^2 -> 0 as 2/d (the proved core), monotonically."""
    rv = [squared_distance_relative_variance(d, seed=0) for d in GRID]
    for a, b in zip(rv, rv[1:]):
        assert b < a + 1e-3, f"relative variance not falling: {a:.4f} -> {b:.4f}"
    for d in (50, 200, 1000):
        v = squared_distance_relative_variance(d, n=20000, seed=2)
        assert abs(v - 2.0 / d) < 0.3 * (2.0 / d), f"rel var off 2/d at d={d}: {v:.5f}"
    print("  [ok] relative variance of D^2 vanishes as 2/d (the proved core)")


def test_volume_concentration() -> None:
    """The unit ball's volume -> 0 and its mass flees to the surface:
    shell_fraction(d, 0.1) increases to 1."""
    vols = [ball_volume(d) for d in GRID]
    assert vols[-1] < vols[5] and vols[-1] < 1e-6, "ball volume should collapse"
    fracs = [shell_fraction(d, 0.10) for d in GRID]
    for a, b in zip(fracs, fracs[1:]):
        assert b >= a - 1e-12, "shell fraction should be nondecreasing in d"
    assert fracs[-1] > 0.999, f"outer shell should hold ~all mass: {fracs[-1]:.4f}"
    print(f"  [ok] volume concentration: ball volume -> 0, outer 10% shell holds "
          f"{fracs[-1]*100:.1f}% of mass at d={GRID[-1]}")


def test_intrinsic_dimension() -> None:
    """Contrast is governed by intrinsic, not ambient, dimension. TwoNN recovers k
    at moderate d; at d=1536 structured data keeps contrast where i.i.d. loses it."""
    # TwoNN recovers a known intrinsic dimension.
    est_k = twonn_intrinsic_dim(structured_data(1000, 200, k=10, noise=0.01, seed=0))
    assert 8.0 <= est_k <= 13.0, f"TwoNN should recover k=10, got {est_k:.2f}"
    # On data that fills its space, TwoNN matches the ambient dimension.
    est_amb = twonn_intrinsic_dim(iid_data(1000, 10, seed=1))
    assert 8.0 <= est_amb <= 12.0, f"TwoNN should match ambient d=10, got {est_amb:.2f}"
    # The consequence at the production embedding dimension.
    d, k = FINANCE_DIM, FINANCE_INTRINSIC_K
    c_struct = relative_contrast(d, seed=2,
                                 sampler=lambda n, dd, seed: structured_data(n, dd, k, 0.01, seed))
    c_iid = relative_contrast(d, seed=3)
    assert c_struct > 5 * c_iid, f"structured should keep contrast: {c_struct:.3f} vs {c_iid:.3f}"
    k_struct = twonn_intrinsic_dim(structured_data(800, d, k, 0.01, seed=4))
    k_iid = twonn_intrinsic_dim(iid_data(800, d, seed=5))
    assert k_struct < 0.5 * k_iid, f"structured TwoNN {k_struct:.1f} not << iid {k_iid:.1f}"
    print(f"  [ok] intrinsic dimension governs contrast: TwoNN recovers k "
          f"({est_k:.1f}~10); at d={d}, structured contrast {c_struct:.3f} >> iid {c_iid:.3f}")


if __name__ == "__main__":
    print("High-dimensional geometry verification harness")
    test_chi_square_moments()
    test_norm_concentration()
    test_near_orthogonality()
    test_distance_concentration()
    test_relative_variance_vanishes()
    test_volume_concentration()
    test_intrinsic_dimension()
    print("Grid table (mirrored by ConcentrationLaboratory.tsx):")
    print(f"  {'d':>5}{'contrast':>12}{'Var<u,v>':>12}{'1/d':>10}"
          f"{'shell_std':>12}{'shell10%':>11}")
    for r in grid_table():
        print(f"  {int(r['d']):>5}{r['contrast']:>12.4f}{r['ip_var']:>12.5f}"
              f"{r['ip_var_theory']:>10.5f}{r['shell_std']:>12.4f}{r['shell_frac_10pct']:>11.4f}")
    print("Finance demo:")
    finance_demo()
    print("All checks passed.")
