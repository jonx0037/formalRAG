"""Normalization, the hypersphere, and von Mises-Fisher geometry — the reference
implementation for the formalRAG `hypersphere-vmf-geometry` topic.

Dense retrievers L2-normalize their embeddings, so the working space is the unit
hypersphere S^{d-1}, and cosine similarity *is* the geometry of that sphere. This
module establishes — and verifies — four facts, building straight on the
near-orthogonality result of the `high-dimensional-geometry` topic:

  1. COSINE = DISTANCE ON THE SPHERE. For unit vectors, ||x - y||^2 = 2 - 2<x, y>,
     so ranking by cosine similarity and ranking by (negated) Euclidean distance are
     the SAME ranking. A retriever may train on one and search with the other.
  2. THE EQUATORIAL BAND. For v uniform on S^{d-1} and a fixed axis u, the projection
     t = <u, v> has density proportional to (1 - t^2)^((d-3)/2), with mean 0 and
     variance EXACTLY 1/d — the same 1/d we met as Var<u, v> in high-dimensional
     geometry, now read as a density. The mass crowds the equator of every axis.
  3. THE von MISES-FISHER DISTRIBUTION. The natural exponential family / maximum-
     entropy law on the sphere: f(x; mu, kappa) = C_d(kappa) exp(kappa mu^T x), whose
     normalizing constant C_d(kappa) = kappa^{d/2-1} / [(2 pi)^{d/2} I_{d/2-1}(kappa)]
     is a modified Bessel function PRECISELY because the surface integral reduces, via
     the equatorial slice of fact 2, to the Bessel integral. Its mean resultant length
     is rho = A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa).
  4. ESTIMATION. The MLE of the mean direction is mu_hat = R / ||R|| (R = sum of the
     data), and kappa_hat solves A_d(kappa) = r_bar where r_bar = ||R|| / n. The
     Banerjee et al. (2005) closed form kappa_hat ~ r_bar (d - r_bar^2) / (1 - r_bar^2)
     is an APPROXIMATION to that implicit Bessel-ratio root; we verify its error
     against the numerically-solved root rather than treating it as exact.

Every pedagogical claim the topic makes is an `assert` below. The closed-form targets
are exact: Var<u, v> = 1/d and E[t^4] = 3/(d(d+2)) for the uniform-sphere coordinate;
C_d(kappa) integrates the vMF density to 1; A_d(kappa) is monotone in kappa from 0 to
1. `grid_table()` prints the numbers that `HypersphereLaboratory.tsx` mirrors to the
decimal. NUMERICAL NOTE: I_nu(kappa) overflows for large kappa, so the normalizer and
the A_d ratio are computed with `scipy.special.ive` (the e^{-kappa}-scaled Bessel),
where the scale factors cancel and the values stay finite even at d=1536.

Run:  uv run --with numpy --with scipy python notebooks/hypersphere-vmf-geometry/hypersphere_vmf_geometry.py
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betainc, gammaln, ive, logsumexp

# The dimension grid the equatorial panel of the viz steps through. d >= 2 (the
# coordinate marginal needs a sphere). 768 and 1536 are real embedding dimensions.
D_GRID: tuple[int, ...] = (2, 3, 5, 10, 20, 50, 100, 200, 500, 768, 1536)

# The concentration grid the vMF panel steps through, reported at a fixed display
# dimension so every readout is a baked number (no client-side Bessel ratio). The
# display dimension is moderate (d=100) so the density's tilt is legible across the
# grid — at an embedding dimension the same law holds, only far narrower.
KAPPA_GRID: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
                                 100.0, 200.0, 500.0, 1000.0)
VMF_DISPLAY_DIM = 100


# --------------------------------------------------------------------------- #
# The sphere: normalization, samplers, and the cosine <-> distance identity.
# --------------------------------------------------------------------------- #

def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row onto the unit sphere S^{d-1}."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sample_uniform_sphere(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n points uniform on S^{d-1} (normalized standard Gaussians)."""
    x = np.random.default_rng(seed).standard_normal((n, d))
    return normalize(x)


def coordinate_marginal_samples(n: int, d: int, seed: int = 0) -> np.ndarray:
    """The projection t = <u, v> of uniform v onto a fixed axis u = e_1: the first
    coordinate of a uniform-sphere sample. Its law is the equatorial marginal."""
    return sample_uniform_sphere(n, d, seed)[:, 0]


def squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise squared Euclidean distance ||x - y||^2."""
    return np.sum((x - y) ** 2, axis=-1)


# --------------------------------------------------------------------------- #
# The von Mises-Fisher distribution: normalizer, mean resultant, sampler.
# --------------------------------------------------------------------------- #

def log_surface_area(d: int) -> float:
    """log of the surface area of S^{d-1}: S_{d-1} = 2 pi^{d/2} / Gamma(d/2)."""
    return math.log(2.0) + (d / 2.0) * math.log(math.pi) - gammaln(d / 2.0)


def log_vmf_norm_const(d: int, kappa: float) -> float:
    """log C_d(kappa) for f(x) = C_d(kappa) exp(kappa mu^T x) on S^{d-1}.

    C_d(kappa) = kappa^{d/2-1} / [(2 pi)^{d/2} I_{d/2-1}(kappa)]. Computed with the
    exponentially-scaled Bessel ive(nu, k) = I_nu(k) e^{-k} so log I_nu(k) =
    log ive(nu, k) + k never overflows. kappa = 0 is the uniform law, C = 1/S_{d-1}.
    """
    if kappa <= 0.0:
        return -log_surface_area(d)
    nu = d / 2.0 - 1.0
    iv = ive(nu, kappa)
    if iv > 0.0:
        log_iv = math.log(iv) + kappa
    else:
        # ive underflows to 0 for large order / small argument (e.g. d = 1536 with
        # small kappa); fall back to the leading small-argument term of log I_nu.
        log_iv = nu * math.log(kappa / 2.0) - gammaln(nu + 1.0)
    return nu * math.log(kappa) - (d / 2.0) * math.log(2.0 * math.pi) - log_iv


def mean_resultant_length(d: int, kappa: float) -> float:
    """rho = A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa) = ||E[x]|| for x ~ vMF.

    Computed by the downward continued fraction R_nu = 1/(2 nu/kappa + R_{nu+1}) that
    falls out of the Bessel recurrence I_{nu-1} - I_{nu+1} = (2 nu/kappa) I_nu. This
    evaluates no Bessel function directly, so it is stable at embedding dimensions
    where I_nu(kappa) itself underflows (e.g. d=1536). A_d increases from A_d(0)=0 to
    A_d(inf)=1; rho mu = E[x].
    """
    if kappa <= 0.0:
        return 0.0
    nu0 = d / 2.0                                   # want R_{nu0} = I_{d/2}/I_{d/2-1}
    steps = int(2.0 * kappa + nu0 + 100.0)          # deep enough to kill the tail seed
    nu = nu0 + steps
    r = 0.0
    while nu >= nu0:                                 # process down to and including nu0
        r = 1.0 / (2.0 * nu / kappa + r)
        nu -= 1.0
    return float(r)


def _wood_axis_component(n: int, d: int, kappa: float, rng) -> np.ndarray:
    """Sample n values of W = mu^T x for x ~ vMF(mu, kappa) via Wood's (1994)
    rejection scheme on the tilted equatorial marginal (1-W^2)^((d-3)/2) e^{kappa W}."""
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1) * math.log(1.0 - x0 ** 2)
    out = np.empty(n)
    filled = 0
    while filled < n:
        m = n - filled
        z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=m)
        u = rng.random(m)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        accept = kappa * w + (d - 1) * np.log(1.0 - x0 * w) - c >= np.log(u)
        k = int(accept.sum())
        out[filled:filled + k] = w[accept]
        filled += k
    return out


def sample_vmf(n: int, mu: np.ndarray, kappa: float, seed: int = 0) -> np.ndarray:
    """n points from vMF(mu, kappa) on S^{d-1} (Wood 1994).

    Draw the axis component W = mu^T x by rejection, draw a uniform direction in the
    tangent subspace orthogonal to mu, and combine x = W mu + sqrt(1 - W^2) xi.
    """
    rng = np.random.default_rng(seed)
    mu = normalize(np.asarray(mu, dtype=float).reshape(-1))
    d = mu.shape[0]
    if d < 2:
        raise ValueError("von Mises-Fisher sampling requires dimension d >= 2.")
    if kappa <= 0.0:
        return sample_uniform_sphere(n, d, seed)
    w = _wood_axis_component(n, d, kappa, rng)
    # Tangent directions: Gaussian, project out the mu component, normalize.
    xi = rng.standard_normal((n, d))
    xi = xi - (xi @ mu)[:, None] * mu[None, :]
    xi = normalize(xi)
    return w[:, None] * mu[None, :] + np.sqrt(np.maximum(1.0 - w ** 2, 0.0))[:, None] * xi


# --------------------------------------------------------------------------- #
# Estimation: the MLE of mu and kappa, exact and approximate.
# --------------------------------------------------------------------------- #

def mle_mu(x: np.ndarray) -> tuple[np.ndarray, float]:
    """The MLE of the mean direction and the mean resultant length r_bar = ||R||/n,
    where R = sum_i x_i. Returns (mu_hat = R/||R||, r_bar)."""
    R = x.sum(axis=0)
    norm_R = float(np.linalg.norm(R))
    return R / norm_R, norm_R / x.shape[0]


def kappa_hat_banerjee(d: int, rbar: float) -> float:
    """Banerjee et al. (2005) closed-form approximation to the kappa MLE.

    kappa_hat ~ r_bar (d - r_bar^2) / (1 - r_bar^2). An APPROXIMATION to the implicit
    root of A_d(kappa) = r_bar, not the exact MLE — see `kappa_hat_exact`.
    """
    return rbar * (d - rbar ** 2) / (1.0 - rbar ** 2)


def kappa_hat_exact(d: int, rbar: float) -> float:
    """The exact kappa MLE: the root of A_d(kappa) = r_bar, found with Brent's method.
    The reference the Banerjee approximation is checked against."""
    if rbar <= 0.0:
        return 0.0
    if rbar >= 1.0:
        return float("inf")
    hi = max(1.0, kappa_hat_banerjee(d, rbar)) * 4.0 + 10.0
    while mean_resultant_length(d, hi) < rbar and hi < 1e12:
        hi *= 2.0
    if mean_resultant_length(d, hi) < rbar:        # r_bar ~ 1: the MLE diverges; cap
        return hi
    return float(brentq(lambda k: mean_resultant_length(d, k) - rbar, 1e-9, hi,
                        xtol=1e-10, rtol=1e-12))


# --------------------------------------------------------------------------- #
# The grid table the viz mirrors, and the finance demo.
# --------------------------------------------------------------------------- #

def equator_band_fraction(d: int, eps: float = 0.1) -> float:
    """Exact fraction of the uniform sphere within eps of the equator |<u,v>| <= eps.

    t^2 ~ Beta(1/2, (d-1)/2), so P(|t| <= eps) = I_{eps^2}(1/2, (d-1)/2), the
    regularized incomplete beta. Increases to 1 as d grows.
    """
    return float(betainc(0.5, (d - 1) / 2.0, eps ** 2))


def coordinate_marginal_stats(d: int, n: int = 40000, seed: int = 0) -> dict[str, float]:
    """Empirical moments of the uniform-sphere coordinate t against the exact targets
    Var(t) = 1/d and E[t^4] = 3/(d(d+2))."""
    t = coordinate_marginal_samples(n, d, seed)
    return {
        "mean": float(np.mean(t)),                 # -> 0
        "var": float(np.var(t)),                   # -> 1/d
        "var_theory": 1.0 / d,
        "fourth": float(np.mean(t ** 4)),          # -> 3/(d(d+2))
        "fourth_theory": 3.0 / (d * (d + 2)),
        "band_frac": equator_band_fraction(d, 0.1),
    }


def grid_table() -> dict[str, list[dict[str, float]]]:
    """The numbers HypersphereLaboratory.tsx mirrors to the decimal. Deterministic.

    `equator`: per-dimension coordinate-marginal variance (empirical vs 1/d) and the
    equatorial band fraction. `vmf`: per-kappa mean resultant rho = A_d(kappa) and the
    kappa round-trip (Banerjee vs exact) at the display dimension d = VMF_DISPLAY_DIM.
    """
    equator = []
    for d in D_GRID:
        s = coordinate_marginal_stats(d, n=40000, seed=1)
        equator.append({
            "d": d,
            "proj_var": s["var"],
            "proj_var_theory": s["var_theory"],
            "equator_band_frac": s["band_frac"],
        })
    dv = VMF_DISPLAY_DIM
    vmf = []
    for kappa in KAPPA_GRID:
        rho = mean_resultant_length(dv, kappa)
        vmf.append({
            "kappa": kappa,
            "rho": rho,
            "kappa_hat_banerjee": kappa_hat_banerjee(dv, rho) if rho > 0 else 0.0,
            "kappa_hat_exact": kappa_hat_exact(dv, rho) if rho > 0 else 0.0,
        })
    return {"equator": equator, "vmf": vmf}


# Finance case study: two topical clusters of ~1536-d financial-document embeddings,
# one tight ("interest-rate risk"), one loose ("supply-chain disruption"). The naive
# worry is that cosine is meaningless at 1536 dimensions; the resolution is that a
# topical cluster is a CONCENTRATED vMF whose tightness kappa is measurable, and that
# two such clusters are separable. SYNTHETIC vMF stand-in, not a trained encoder.
FINANCE_DIM = 1536
FINANCE_KAPPA_TIGHT = 900.0
FINANCE_KAPPA_LOOSE = 300.0


def _mean_pairwise_cosine(x: np.ndarray) -> float:
    """Mean over distinct pairs of <x_i, x_j>. For unit rows this is
    (||sum x||^2 - n) / (n(n-1))."""
    n = x.shape[0]
    s = float(np.sum(x.sum(axis=0) ** 2))
    return (s - n) / (n * (n - 1))


def finance_demo() -> dict[str, float]:
    """Tight vs loose vMF clusters at the production embedding dimension."""
    d = FINANCE_DIM
    rng = np.random.default_rng(20)
    mu_tight = normalize(rng.standard_normal(d))
    mu_loose = normalize(rng.standard_normal(d))   # ~orthogonal to mu_tight in R^1536
    tight = sample_vmf(800, mu_tight, FINANCE_KAPPA_TIGHT, seed=21)
    loose = sample_vmf(800, mu_loose, FINANCE_KAPPA_LOOSE, seed=22)

    _, rbar_t = mle_mu(tight)
    _, rbar_l = mle_mu(loose)
    inter = float(np.mean(tight @ loose.T))         # mean inter-cluster cosine
    out = {
        "mu_separation_cos": float(mu_tight @ mu_loose),
        "rbar_tight": rbar_t,
        "rbar_loose": rbar_l,
        "kappa_hat_tight": kappa_hat_exact(d, rbar_t),
        "kappa_hat_loose": kappa_hat_exact(d, rbar_l),
        "kappa_hat_tight_banerjee": kappa_hat_banerjee(d, rbar_t),
        "kappa_hat_loose_banerjee": kappa_hat_banerjee(d, rbar_l),
        "intra_cos_tight": _mean_pairwise_cosine(tight),
        "intra_cos_loose": _mean_pairwise_cosine(loose),
        "inter_cos": inter,
    }
    print(f"  two topical clusters of embeddings in R^{d} (SYNTHETIC vMF, not a "
          f"trained encoder):")
    print(f"  {'cluster':<10}{'kappa_true':>11}{'r_bar':>9}{'kappa_hat':>11}"
          f"{'intra-cos':>11}")
    print(f"  {'tight':<10}{FINANCE_KAPPA_TIGHT:>11.0f}{out['rbar_tight']:>9.4f}"
          f"{out['kappa_hat_tight']:>11.1f}{out['intra_cos_tight']:>11.4f}")
    print(f"  {'loose':<10}{FINANCE_KAPPA_LOOSE:>11.0f}{out['rbar_loose']:>9.4f}"
          f"{out['kappa_hat_loose']:>11.1f}{out['intra_cos_loose']:>11.4f}")
    print(f"  inter-cluster mean cosine = {out['inter_cos']:.4f} "
          f"(mean directions are {out['mu_separation_cos']:+.4f}-aligned)")
    print("  -> the tighter cluster (higher kappa) has a higher mean resultant and a "
          "tighter cosine distribution, and both clusters separate from each other.")
    return out


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_cosine_distance_identity() -> None:
    """Prop 1: for unit vectors ||x - y||^2 = 2 - 2<x, y>, and ranking by cosine
    similarity equals ranking by (negated) Euclidean distance."""
    rng = np.random.default_rng(0)
    for d in (3, 10, 100, 1536):
        x = sample_uniform_sphere(64, d, seed=d)
        y = sample_uniform_sphere(64, d, seed=d + 1)
        lhs = squared_distance(x, y)
        rhs = 2.0 - 2.0 * np.sum(x * y, axis=1)
        assert np.allclose(lhs, rhs, atol=1e-10), f"cosine-distance identity off at d={d}"
    # Ranking equivalence against a fixed query.
    for d in (10, 768):
        q = sample_uniform_sphere(1, d, seed=7)[0]
        cand = sample_uniform_sphere(50, d, seed=8)
        cos = cand @ q
        dist = squared_distance(cand, np.broadcast_to(q, cand.shape))
        assert np.array_equal(np.argsort(-cos), np.argsort(dist)), \
            f"cosine and distance rankings disagree at d={d}"
        assert int(np.argmax(cos)) == int(np.argmin(dist)), "argmax cos != argmin dist"
    print("  [ok] cosine = distance on the sphere: ||x-y||^2 = 2-2<x,y>, same ranking")


def test_coordinate_marginal() -> None:
    """Thm 1: the uniform-sphere coordinate t has mean 0, Var(t) = 1/d, and
    E[t^4] = 3/(d(d+2)) (the (1-t^2)^((d-3)/2) law); the equatorial band fills as d grows."""
    for d in (3, 10, 50, 200, 1536):
        s = coordinate_marginal_stats(d, n=60000, seed=2)
        assert abs(s["mean"]) < 0.02, f"E[t] not ~0 at d={d}: {s['mean']:.4f}"
        assert abs(s["var"] - 1.0 / d) < 0.1 * (1.0 / d), f"Var(t) off 1/d at d={d}"
        assert abs(s["fourth"] - s["fourth_theory"]) < 0.15 * s["fourth_theory"], \
            f"E[t^4] off 3/(d(d+2)) at d={d}"
    fracs = [equator_band_fraction(d, 0.1) for d in D_GRID]
    for a, b in zip(fracs, fracs[1:]):
        assert b >= a - 1e-12, "equatorial band fraction should be nondecreasing in d"
    assert fracs[-1] > 0.999, f"band should fill at d={D_GRID[-1]}: {fracs[-1]:.4f}"
    print(f"  [ok] equatorial law: Var(t)=1/d, E[t^4]=3/(d(d+2)), band -> "
          f"{fracs[-1]*100:.1f}% at d={D_GRID[-1]}")


def test_vmf_normalization() -> None:
    """Def 1 / Thm 2: vMF samples are unit-norm with mean direction mu, and the
    closed-form C_d(kappa) integrates the density to 1 (the Bessel-integral reduction)."""
    mu = normalize(np.arange(1.0, 11.0))           # a fixed direction in R^10
    x = sample_vmf(20000, mu, kappa=20.0, seed=3)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-9), "vMF samples not unit-norm"
    mu_hat, _ = mle_mu(x)
    assert float(mu_hat @ mu) > 0.999, f"vMF mean direction off: cos={mu_hat @ mu:.5f}"
    # The density integrates to 1: C_d(kappa) * S_{d-1} * E_unif[e^{kappa t}] = 1, with
    # E_unif[e^{kappa t}] = C_marg * int_{-1}^{1} e^{kappa t}(1-t^2)^((d-3)/2) dt.
    for d, kappa in ((4, 3.0), (8, 6.0), (16, 10.0)):
        integ = quad(lambda t: math.exp(kappa * t) * (1 - t * t) ** ((d - 3) / 2.0),
                     -1.0, 1.0)[0]
        log_c_marg = gammaln(d / 2.0) - 0.5 * math.log(math.pi) - gammaln((d - 1) / 2.0)
        e_unif = math.exp(log_c_marg) * integ
        total = math.exp(log_vmf_norm_const(d, kappa) + log_surface_area(d)) * e_unif
        assert abs(total - 1.0) < 1e-6, f"vMF density integrates to {total:.6f} != 1 at d={d}"
    print("  [ok] vMF normalization: unit-norm samples, mean dir = mu, C_d(kappa) "
          "integrates to 1 (Bessel reduction)")


def test_mean_resultant() -> None:
    """Prop 2: rho = A_d(kappa) = I_{d/2}/I_{d/2-1} is monotone in kappa from 0 to 1,
    matches the empirical ||E[x]||, and is finite even at d=1536, kappa=5000."""
    for d in (5, 50):
        rhos = [mean_resultant_length(d, k) for k in (0.0, 1.0, 5.0, 20.0, 100.0, 1000.0)]
        assert rhos[0] == 0.0, "A_d(0) should be 0"
        for a, b in zip(rhos, rhos[1:]):
            assert b > a - 1e-12, "A_d should increase in kappa"
        assert rhos[-1] > 0.95, f"A_d should approach 1 for large kappa at d={d}"
    # Empirical mean resultant matches A_d(kappa).
    mu = normalize(np.ones(20))
    for kappa in (5.0, 25.0, 100.0):
        x = sample_vmf(40000, mu, kappa, seed=11)
        emp = float(np.linalg.norm(x.mean(axis=0)))
        assert abs(emp - mean_resultant_length(20, kappa)) < 0.01, \
            f"empirical resultant off A_d at kappa={kappa}"
    # Large-kappa stability at the production dimension (ive ratio, no overflow).
    a_big = mean_resultant_length(1536, 5000.0)
    assert math.isfinite(a_big) and 0.0 < a_big < 1.0, f"A_d unstable at d=1536: {a_big}"
    print(f"  [ok] mean resultant: A_d(kappa) monotone 0->1, matches ||E[x]||, "
          f"stable at d=1536 (A={a_big:.4f})")


def test_max_entropy() -> None:
    """Thm 3: among densities on the circle with a fixed mean resultant, the
    von Mises law exp(kappa cos theta) maximizes entropy. We compare it to tilted
    same-mean-resultant densities exp(a cos theta + b cos 2theta) and confirm none
    has higher entropy (a numeric Gibbs check supporting the max-entropy claim)."""
    theta = np.linspace(-math.pi, math.pi, 4000, endpoint=False)

    def grid_density(logits: np.ndarray) -> np.ndarray:
        return np.exp(logits - logsumexp(logits))

    def resultant(p: np.ndarray) -> float:
        return float(np.sum(p * np.cos(theta)))

    def entropy(p: np.ndarray) -> float:
        nz = p[p > 0]
        return float(-np.sum(nz * np.log(nz)))

    for kappa in (1.0, 4.0, 10.0):
        q = grid_density(kappa * np.cos(theta))
        m = resultant(q)
        h_q = entropy(q)
        for b in (0.0, 0.4, -0.4, 0.8):
            # Solve for a so the tilted density has the same mean resultant m.
            a = brentq(lambda aa: resultant(grid_density(aa * np.cos(theta)
                                                          + b * np.cos(2 * theta))) - m,
                       -50.0, 200.0)
            p = grid_density(a * np.cos(theta) + b * np.cos(2 * theta))
            assert entropy(p) <= h_q + 1e-9, \
                f"tilted density beats von Mises entropy at kappa={kappa}, b={b}"
    print("  [ok] maximum entropy: von Mises maximizes entropy at fixed mean resultant")


def test_mle_recovery() -> None:
    """Thm 4: mu_hat = R/||R|| recovers mu; the exact kappa MLE inverts A_d=r_bar; the
    Banerjee closed form is a bounded-error APPROXIMATION to that exact root."""
    # Finite-sample mean-direction recovery.
    mu = normalize(np.arange(1.0, 51.0))
    x = sample_vmf(20000, mu, kappa=40.0, seed=12)
    mu_hat, rbar = mle_mu(x)
    assert float(mu_hat @ mu) > 0.999, f"mu_hat off: cos={mu_hat @ mu:.5f}"
    # Population round-trip: feed r_bar = A_d(kappa) and recover kappa exactly.
    rel_errs = []
    for d in (10, 100, 768):
        for kappa in (2.0, 10.0, 50.0, 200.0):
            rbar_pop = mean_resultant_length(d, kappa)
            k_exact = kappa_hat_exact(d, rbar_pop)
            assert abs(k_exact - kappa) < 1e-4 * kappa + 1e-6, \
                f"exact MLE failed to invert A_d at d={d}, kappa={kappa}: {k_exact:.4f}"
            k_ban = kappa_hat_banerjee(d, rbar_pop)
            rel_errs.append(abs(k_ban - kappa) / kappa)
    worst = max(rel_errs)
    assert worst < 0.06, f"Banerjee approximation error too large: {worst:.4f}"
    print(f"  [ok] MLE: mu_hat recovers mu; exact kappa inverts A_d; Banerjee within "
          f"{worst*100:.1f}% of the exact root")


def test_finance_clusters() -> None:
    """The tighter vMF cluster has a higher mean resultant, a higher recovered kappa,
    and a tighter cosine distribution; both clusters separate from each other."""
    f = finance_demo()
    assert f["rbar_tight"] > f["rbar_loose"], "tight cluster should have higher r_bar"
    assert f["kappa_hat_tight"] > f["kappa_hat_loose"], "tight should recover higher kappa"
    assert f["intra_cos_tight"] > f["intra_cos_loose"], "tight should have higher intra-cosine"
    assert f["intra_cos_tight"] > f["inter_cos"] + 0.05, "clusters should separate"
    print("  [ok] finance: tighter cluster is more concentrated and clusters separate")


if __name__ == "__main__":
    print("Hypersphere / von Mises-Fisher verification harness")
    test_cosine_distance_identity()
    test_coordinate_marginal()
    test_vmf_normalization()
    test_mean_resultant()
    test_max_entropy()
    test_mle_recovery()
    print("Grid table (mirrored by HypersphereLaboratory.tsx):")
    tbl = grid_table()
    print(f"  equatorial law:  {'d':>6}{'Var(t) emp':>13}{'1/d':>11}{'band|t|<0.1':>13}")
    for r in tbl["equator"]:
        print(f"  {'':>17}{int(r['d']):>6}{r['proj_var']:>13.5f}"
              f"{r['proj_var_theory']:>11.5f}{r['equator_band_frac']:>13.5f}")
    print(f"  vMF at d={VMF_DISPLAY_DIM}: {'kappa':>8}{'rho=A_d':>11}"
          f"{'kappa_hat (Banerjee)':>22}{'kappa_hat (exact)':>19}")
    for r in tbl["vmf"]:
        print(f"  {'':>13}{r['kappa']:>8.0f}{r['rho']:>11.5f}"
              f"{r['kappa_hat_banerjee']:>22.2f}{r['kappa_hat_exact']:>19.2f}")
    print("Finance demo:")
    test_finance_clusters()
    print("All checks passed.")
