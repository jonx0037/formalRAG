"""Deploying under workload uncertainty: the mixture you do not know.

Run:
    uv run --with numpy --with scipy \
        python notebooks/rag-architecture-workload-uncertainty/rag_architecture_workload_uncertainty.py

The predecessor showed that "which architecture should we deploy" is underdetermined until three
things are fixed -- the workload mixture w, the price ratio rho, and the amortization horizon N --
and then it fixed all three and read the answer off a partition of the simplex.

In production you do not know w. You estimate it from traffic, and the estimate has error. That
error interacts badly with a partition, for a reason worth stating before any of the mathematics:
the map from workload to architecture is PIECEWISE CONSTANT. Estimation error is therefore
harmless in the interior of a cell -- the recommendation does not move at all -- and maximally
harmful at a boundary, where an arbitrarily small misestimate flips it outright.

This is the first topic in the arc where the INPUTS to the decision are random rather than the
outputs, and it is a decision-theory topic rather than a retrieval one. Three rules are compared:

  plug-in    argmax_a U(a, what)                       -- pretend the estimate is the truth
  Bayes      argmax_a E_w[U(a, w)]                     -- average over what you do not know
  minimax    argmin_a max_w [U(a*(w), w) - U(a, w)]    -- hedge against the worst case

It is NOT per-query routing. `adaptive-retrieval-routing` is uncertain about WHICH QUERY this is
and picks a strategy for each one; this is uncertain about WHAT THE TRAFFIC IS and commits to a
single fixed architecture for all of it. A point-mass posterior collapses all three rules onto the
predecessor's `winner()`, and a test asserts it.

Imports rag-architecture-pareto and reimplements nothing. There is no new corpus: the 6 x 6 x 3
table of quality and both costs arrives through the import chain, and everything here is a layer
of uncertainty over a decision layer over that table.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("rag-architecture-pareto",):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rag_architecture_pareto as PAR                                # noqa: E402
from rag_architecture_pareto import (                                # noqa: E402
    ARM_NAMES, CONDITIONS, LAMBDA_HEADLINE, N_HEADLINE, RHO_HEADLINE,
    bridge_family, local_only, one_hot, uniform_within_regime, utility, winner,
)

SEED = 20260920                 # the module's own seed; every draw below is derived from it

# --- the posterior on the workload ------------------------------------------------------------
# A Dirichlet posterior from n observed queries under a symmetric prior: having seen counts
# n * what, the posterior over w is Dirichlet(n * what + alpha). alpha = 1 is the uniform prior on
# the simplex, which is the honest default when nothing is known about the traffic beforehand.
PRIOR_ALPHA = 1.0
N_OBSERVED = (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120)
N_HEADLINE_OBS = 40             # the operating point: a deployment that has seen 40 queries

# Worst-case regret over a posterior is an EXTREME statistic, and a raw max over draws is visibly
# unstable (it drifts upward as the draw count rises, because the max of more samples is larger).
# A high quantile of the regret distribution is the same idea with a variance that does not depend
# on how long the simulation ran -- `test_quantile_is_stable_across_seeds` pins it.
REGRET_Q = 0.95
DRAWS = 2000

_TABLE: dict | None = None


def posterior(w_hat: np.ndarray, n_obs: int, draws: int = DRAWS,
              seed: int = SEED, alpha: float = PRIOR_ALPHA) -> np.ndarray:
    """Draws from the Dirichlet posterior over the true workload, having observed n_obs queries
    whose empirical mixture was w_hat.

    n_obs is what concentrates it: the posterior mean is (n*what + alpha)/(n + C*alpha), so a
    large n pulls the mass onto w_hat and a small n leaves it spread across the simplex.

    GUARDS: w_hat must be a probability vector; n_obs >= 0; draws >= 1.
    """
    w_hat = np.asarray(w_hat, dtype=float)
    if w_hat.shape != (len(CONDITIONS),):
        raise ValueError(f"a workload has {len(CONDITIONS)} entries, got {w_hat.shape}")
    if w_hat.min() < -1e-12 or not np.isclose(w_hat.sum(), 1.0):
        raise ValueError("w_hat must be non-negative and sum to 1")
    if n_obs < 0:
        raise ValueError(f"cannot observe a negative number of queries, got {n_obs}")
    if draws < 1:
        raise ValueError(f"need at least one posterior draw, got {draws}")
    conc = n_obs * w_hat + alpha
    return np.random.default_rng(seed).dirichlet(conc, draws)


_KERNEL: dict[tuple, tuple] = {}


def utility_kernel(lam: float = LAMBDA_HEADLINE, rho: float = RHO_HEADLINE,
                   n_queries: int = N_HEADLINE):
    """(M, const) with U(a, w) = (M w)_a + const_a -- the predecessor's utility written out as the
    affine function of w that it is.

    This is the module's own theorem applied to its own hot path. Utility is linear in w, so

        U(a, w) = sum_k w_k [ Q[a,k] - lam (O[a,k] + rho C[a,k]) ]  -  lam * build_a / N

    and every posterior draw can be evaluated in ONE matrix product rather than one Python call
    apiece. Evaluating 2000 draws by calling the imported `utility` in a loop took the module two
    minutes; this takes milliseconds, and `test_kernel_matches_the_imported_utility` proves the
    two agree to floating point rather than asking to be believed.
    """
    key = (lam, rho, n_queries)
    if key not in _KERNEL:
        t = PAR.measure()
        M = t["Q"] - lam * (t["O"] + rho * t["C"])
        const = -lam * PAR.amortized_build(n_queries)
        _KERNEL[key] = (M, const)
    return _KERNEL[key]


def utilities_over(post: np.ndarray, lam: float = LAMBDA_HEADLINE,
                   rho: float = RHO_HEADLINE, n_queries: int = N_HEADLINE) -> np.ndarray:
    """(draws, arms) utilities for every posterior draw, in one matrix product."""
    M, const = utility_kernel(lam, rho, n_queries)
    return np.asarray(post, dtype=float) @ M.T + const


# =================================================================================================
# Movement 2 -- three rules, not one.
# =================================================================================================

def rule_plug_in(w_hat: np.ndarray, **kw) -> str:
    """Treat the estimate as the truth. This is what every cost-aware evaluation does implicitly
    when it reports a winner for "your workload", and it is the predecessor's `winner` called on
    a number that happens to be an estimate."""
    return winner(np.asarray(w_hat, dtype=float),
                  kw.get("lam", LAMBDA_HEADLINE), kw.get("rho", RHO_HEADLINE),
                  kw.get("n_queries", N_HEADLINE))


def _decide(U: np.ndarray, q: float = REGRET_Q):
    """From ONE (draws, arms) utility matrix: the Bayes arm, the hedge, and the per-arm worst-case
    regret. Every posterior-consuming rule funnels through here, which is what lets `all_three`
    evaluate the matrix once and still be running the same code the individual rules run --
    `test_all_three_matches_the_individual_rules` proves the two routes agree.

    GUARD: q must be a quantile.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"a quantile lives in (0, 1], got {q}")
    reg = np.asarray(U).max(axis=1, keepdims=True) - U
    wc = np.quantile(reg, q, axis=0)
    return int(np.argmax(U.mean(axis=0))), int(np.argmin(wc)), wc


def rule_bayes(post: np.ndarray, **kw) -> str:
    """Maximize utility AVERAGED over what you do not know. Because utility is linear in w, this
    is exactly the plug-in rule applied to the posterior MEAN -- a fact worth stating rather than
    hiding, since it is why Bayes and plug-in agree so often here and why the interesting contrast
    is with minimax rather than with Bayes."""
    return ARM_NAMES[_decide(utilities_over(post, **kw))[0]]


def regret_matrix(post: np.ndarray, **kw) -> np.ndarray:
    """(draws, arms): what each arm gives up against the best arm FOR THAT DRAW. Regret, not
    utility, is the quantity a hedge is built on -- an arm can have poor utility everywhere and
    still be a good hedge if it is never far behind whichever arm happens to win."""
    U = utilities_over(post, **kw)
    return U.max(axis=1, keepdims=True) - U


def worst_case_regret(post: np.ndarray, q: float = REGRET_Q, **kw) -> np.ndarray:
    """Per-arm worst-case regret over the posterior, as the q-th quantile rather than the max.

    The max over draws is an extreme statistic whose value grows with the number of draws, so two
    runs at different `draws` disagree and nothing can be baked from it. A quantile estimates a
    fixed property of the regret distribution instead.
    """
    return _decide(utilities_over(post, **kw), q)[2]


def rule_minimax(post: np.ndarray, q: float = REGRET_Q, **kw) -> str:
    """Minimize the worst-case regret. Where the posterior is concentrated inside one cell this
    returns that cell's arm; where it straddles a boundary it can return an arm that is nobody's
    argmax, because being second-best on both sides beats being best on one."""
    return ARM_NAMES[_decide(utilities_over(post, **kw), q)[1]]


def all_three(w_hat: np.ndarray, n_obs: int, draws: int = DRAWS, seed: int = SEED,
              q: float = REGRET_Q, **kw) -> dict:
    """The three rules on one workload and one observation count."""
    post = posterior(w_hat, n_obs, draws, seed)
    # ONE pass over the draws. Routing each rule separately would evaluate the same 2000-draw
    # matrix three times and throw two of them away -- which would undercut the very kernel
    # optimization this module is built on.
    bayes_i, minimax_i, _wc = _decide(utilities_over(post, **kw), q)
    return {"plug_in": rule_plug_in(w_hat, **kw),
            "bayes": ARM_NAMES[bayes_i],
            "minimax": ARM_NAMES[minimax_i]}


def rules_disagree(w_hat: np.ndarray, n_obs: int, **kw) -> bool:
    r = all_three(w_hat, n_obs, **kw)
    return r["plug_in"] != r["minimax"]


def posterior_mean(w_hat: np.ndarray, n_obs: int, alpha: float = PRIOR_ALPHA) -> np.ndarray:
    """The Dirichlet posterior mean in closed form: (n*what + alpha) / (n + C*alpha). This is the
    ENTIRE content of the Bayes rule, because utility is linear in w -- see
    `test_bayes_is_plug_in_at_the_posterior_mean`."""
    w_hat = np.asarray(w_hat, dtype=float)
    conc = n_obs * w_hat + alpha
    return conc / conc.sum()


def shrinkage(w_hat: np.ndarray, n_obs: int, condition: str = "bridge") -> float:
    """How far the prior pulls one coordinate of the estimate toward uniform. This is the whole
    of the gap between plug-in and Bayes, and it vanishes like 1/n."""
    return float(posterior_mean(w_hat, n_obs)[CONDITIONS.index(condition)])


# =================================================================================================
# Movement 3 -- the hedge, and where it appears.
# =================================================================================================

def boundary_scan(fracs=np.linspace(0.0, 1.0, 41), n_obs: int = N_HEADLINE_OBS, **kw) -> list[dict]:
    """Walk the bridge-share family across the cell boundary the predecessor measured, and record
    what each rule says at every step. Interior points agree; the neighborhood of the boundary is
    where they part."""
    out = []
    for f in fracs:
        w = bridge_family(float(f))
        r = all_three(w, n_obs, **kw)
        out.append({"frac": round(float(f), 4), "plug_in": r["plug_in"],
                    "bayes": r["bayes"], "minimax": r["minimax"],
                    "agree": r["plug_in"] == r["minimax"]})
    return out


def disagreement_rate(n_obs: int = N_HEADLINE_OBS, trials: int = 300, seed: int = SEED,
                      **kw) -> float:
    """How often the plug-in rule and the hedge part company, over workloads drawn uniformly from
    the simplex. A deployment does not choose where on the simplex it sits."""
    rng = np.random.default_rng(seed + 77)
    hits = 0
    for i, w in enumerate(rng.dirichlet(np.ones(len(CONDITIONS)), trials)):
        post = posterior(w, n_obs, DRAWS, seed + 1000 + i)
        hits += rule_plug_in(w, **kw) != rule_minimax(post, **kw)
    return hits / trials


def hedge_is_local(**kw) -> dict:
    """The hedge is not one fixed 'safe' arm: deep inside a cell it IS that cell's arm. Reported
    for a workload deep in each of several cells, so the claim is checkable rather than asserted."""
    probes = {"bridge-heavy": bridge_family(0.85),
              "global-heavy": np.array([0.03] * (len(CONDITIONS) - 1) + [0.85]),
              "local-only": local_only(),
              "uniform": uniform_within_regime()}
    out = {}
    for label, w in probes.items():
        w = np.asarray(w, dtype=float)
        w = w / w.sum()
        r = all_three(w, N_HEADLINE_OBS, **kw)
        out[label] = {"plug_in": r["plug_in"], "minimax": r["minimax"],
                      "agree": r["plug_in"] == r["minimax"]}
    return out


# =================================================================================================
# Movement 4 -- how many queries is enough, and where the question has no answer.
# =================================================================================================

def agreement_n(w_hat: np.ndarray, ns=N_OBSERVED, **kw) -> int | None:
    """The smallest observation count at which the hedge stops disagreeing with the plug-in rule.
    None means it never does over the grid -- which is the honest answer at a boundary, not a
    failure to search far enough."""
    for n in ns:
        if not rules_disagree(np.asarray(w_hat, dtype=float), int(n), **kw):
            return int(n)
    return None


def regret_decay(w_hat: np.ndarray, ns=N_OBSERVED, q: float = REGRET_Q, **kw) -> list[tuple]:
    """(n, worst-case regret of the hedge) -- how much the uncertainty is costing, as data
    accumulates."""
    out = []
    for n in ns:
        post = posterior(np.asarray(w_hat, dtype=float), int(n), DRAWS, SEED)
        out.append((int(n), float(worst_case_regret(post, q, **kw).min())))
    return out


def halving_ratios(curve) -> list[float]:
    """Successive ratios of the regret curve on a doubling grid. A 1/sqrt(n) law would put these
    at 1/sqrt(2) = 0.707; they land nearby but do NOT hold a constant, which is why this module
    asserts the monotone decrease and the band rather than the law."""
    return [round(b / a, 4) for (_, a), (_, b) in zip(curve, curve[1:]) if a > 0]


# =================================================================================================
# Movement 5 -- what it costs to be wrong.
# =================================================================================================

def realized_regret(w_true: np.ndarray, n_obs: int, trials: int = 200, seed: int = SEED,
                    q: float = REGRET_Q, **kw) -> dict:
    """The honest accounting. An operator observes n queries drawn from the TRUE workload, forms
    an estimate, applies a rule, and lives with the arm it picked. This measures what that cost,
    against an oracle that knew w_true all along.

    Note what is and is not being averaged: the posterior is the operator's belief, but the regret
    is scored against the truth, so a rule that hedges pays for the hedge whenever the plug-in
    rule was right.
    """
    w_true = np.asarray(w_true, dtype=float)
    rng = np.random.default_rng(seed + 4242)
    u_true = utility(w_true, kw.get("lam", LAMBDA_HEADLINE), kw.get("rho", RHO_HEADLINE),
                     kw.get("n_queries", N_HEADLINE))
    best = float(u_true.max())
    tot = {"plug_in": 0.0, "bayes": 0.0, "minimax": 0.0}
    for t in range(trials):
        counts = rng.multinomial(n_obs, w_true)
        w_hat = counts / counts.sum() if counts.sum() else w_true.copy()
        post = posterior(w_hat, n_obs, DRAWS, seed + 9000 + t)
        for name, arm in (("plug_in", rule_plug_in(w_hat, **kw)),
                          ("bayes", rule_bayes(post, **kw)),
                          ("minimax", rule_minimax(post, q, **kw))):
            tot[name] += best - float(u_true[ARM_NAMES.index(arm)])
    return {k: round(v / trials, 5) for k, v in tot.items()}


def cost_of_hedging(fracs=(0.34, 0.40, 0.43, 0.46, 0.52), n_obs: int = N_HEADLINE_OBS,
                    trials: int = 200) -> list[dict]:
    """Realized regret of each rule at several points either side of the cell boundary.

    The hedge is NOT free, and the asymmetry is the point. Both Bayes and minimax shrink the
    estimate toward the middle of the simplex, which is a bet that the truth is more central than
    the data says. Below the boundary that bet pays; just above it, where the truth genuinely sits
    in a narrow cell, the plug-in rule is correctly confident and the hedge drags it back out of
    the cell it was right about.
    """
    out = []
    for f in fracs:
        r = realized_regret(bridge_family(float(f)), n_obs, trials=trials)
        out.append({"frac": round(float(f), 3), **r,
                    "hedge_worse": bool(r["minimax"] > r["plug_in"] + 1e-9)})
    return out


# =================================================================================================
# viz_constants -- this function OWNS every number the laboratory displays.
# =================================================================================================

def viz_constants() -> dict:
    r4 = lambda v: round(float(v), 4)                                          # noqa: E731
    scan = boundary_scan(np.linspace(0.0, 1.0, 41))
    decay_b = regret_decay(bridge_family(0.43))
    decay_i = regret_decay(bridge_family(0.85))
    probes = {"boundary": bridge_family(0.43), "interior": bridge_family(0.85),
              "local_only": local_only(), "uniform": uniform_within_regime()}
    return {
        "arms": list(ARM_NAMES), "conditions": list(CONDITIONS),
        "n_observed": list(N_OBSERVED), "n_headline_obs": N_HEADLINE_OBS,
        "regret_q": REGRET_Q, "draws": DRAWS, "prior_alpha": PRIOR_ALPHA,
        "lambda_headline": LAMBDA_HEADLINE, "rho_headline": RHO_HEADLINE,
        "boundary_scan": [[s["frac"], s["plug_in"], s["minimax"], bool(s["agree"])] for s in scan],
        "hedge_is_local": hedge_is_local(),
        "agreement_n": {k: agreement_n(w) for k, w in probes.items()},
        "decay_boundary": [[n, r4(r)] for n, r in decay_b],
        "decay_interior": [[n, r4(r)] for n, r in decay_i],
        "halving_boundary": halving_ratios(decay_b),
        "disagreement_rate": r4(disagreement_rate()),
        "shrinkage": [[n, r4(shrinkage(bridge_family(0.43), n))] for n in N_OBSERVED],
        "realized": {k: realized_regret(w, N_HEADLINE_OBS) for k, w in
                     (("boundary", bridge_family(0.43)), ("interior", bridge_family(0.85)))},
        "cost_of_hedging": cost_of_hedging(),
        # The laboratory recomputes the two POINT rules live and exactly -- plug-in from the
        # kernel, Bayes from the closed-form posterior mean, both because utility is linear. The
        # hedge is the one rule that needs Monte Carlo, so it is the one rule that gets baked:
        # minimax arm index over (bridge share) x (observation count).
        "minimax_grid": [[int(ARM_NAMES.index(rule_minimax(posterior(bridge_family(f), int(n)))))
                          for n in N_OBSERVED]
                         for f in np.round(np.linspace(0.0, 1.0, 41), 4)],
        "grid_fracs": [round(float(f), 4) for f in np.linspace(0.0, 1.0, 41)],
        "kernel_M": [[r4(v) for v in row] for row in utility_kernel()[0]],
        # The amortized build enters as a per-arm constant. At the headline N it is tiny -- only
        # the graph arm carries one at all -- so it is baked at full precision rather than rounded
        # to a row of zeros that would read as a bug.
        "kernel_const": [float(f"{v:.10g}") for v in utility_kernel()[1]],
        # the utility table the browser recomputes everything from: one row per arm per condition
        "utility_by_condition": [[r4(utility(one_hot(c), LAMBDA_HEADLINE)[i])
                                  for c in CONDITIONS] for i in range(len(ARM_NAMES))],
    }


# =================================================================================================
# Collapse anchors -- the uncertainty layer must vanish when there is nothing to be uncertain about.
# =================================================================================================

def test_bayes_is_plug_in_at_the_posterior_mean() -> None:
    """Bayes is NOT a third rule. Utility is linear in w, so E_w[U(a,w)] = U(a, E[w]) exactly --
    and the Bayes rule is therefore the plug-in rule evaluated at the posterior MEAN, which is
    just the estimate shrunk toward the prior. The whole gap between plug-in and Bayes is that
    shrinkage, and it vanishes like 1/n."""
    for w in (local_only(), bridge_family(0.43), uniform_within_regime(), one_hot("noisy")):
        w = np.asarray(w, dtype=float) / np.sum(w)
        for n in (10, 40, 1280):
            post = posterior(w, n)
            lhs = utilities_over(post).mean(axis=0)
            rhs = utility(post.mean(axis=0), LAMBDA_HEADLINE)
            assert np.allclose(lhs, rhs, atol=1e-12), (n, float(np.abs(lhs - rhs).max()))
            assert rule_bayes(post) == rule_plug_in(post.mean(axis=0))
    # ...and the closed-form posterior mean is that same object
    assert np.allclose(posterior_mean(bridge_family(0.43), 10 ** 7),
                       bridge_family(0.43), atol=1e-5)


def test_point_mass_collapses_to_the_predecessor() -> None:
    """With enough data the posterior is a point mass, every rule agrees, and all three ARE the
    predecessor's `winner`. The uncertainty layer has to disappear when the uncertainty does."""
    for w in (local_only(), bridge_family(0.85), uniform_within_regime()):
        w = np.asarray(w, dtype=float) / np.sum(w)
        r = all_three(w, 10 ** 6)
        assert r["plug_in"] == r["bayes"] == r["minimax"] == winner(w), (r, winner(w))
        assert float(worst_case_regret(posterior(w, 10 ** 6)).min()) < 1e-6


def test_zero_regret_for_the_true_argmax() -> None:
    """Regret is measured against the best arm for each draw, so the best arm has none."""
    post = posterior(uniform_within_regime(), 100)
    reg = regret_matrix(post)
    assert np.allclose(reg.min(axis=1), 0.0, atol=1e-12)
    assert (reg >= -1e-12).all()


def test_guards() -> None:
    """Every entry point rejects a malformed workload rather than returning a plausible number."""
    bad = [(np.ones(3) / 3, 10), (np.full(len(CONDITIONS), 0.5), 10),
           (uniform_within_regime(), -1)]
    for w, n in bad:
        try:
            posterior(w, n)
        except ValueError:
            continue
        raise AssertionError(f"posterior accepted a malformed input: {w.shape}, n={n}")
    for q in (0.0, 1.5):
        try:
            worst_case_regret(posterior(uniform_within_regime(), 10), q=q)
        except ValueError:
            continue
        raise AssertionError(f"worst_case_regret accepted q={q}")


# =================================================================================================
# The claims, as tests. What the hedge buys, and what it costs.
# =================================================================================================

def test_quantile_is_stable_across_seeds() -> None:
    """Worst-case regret is estimated by Monte Carlo, so before anything is asserted about it the
    ESTIMATOR has to be stable. A raw max over draws is not -- it grows with the draw count,
    because the maximum of more samples is larger. A high quantile is a fixed property of the
    regret distribution, and this pins its spread across seeds."""
    w = bridge_family(0.43)
    vals = [float(worst_case_regret(posterior(w, N_HEADLINE_OBS, DRAWS, SEED + s)).min())
            for s in range(6)]
    spread = max(vals) - min(vals)
    assert spread < 0.02, (vals, spread)
    # ...and the arm it selects must not flip with the seed either
    arms = {rule_minimax(posterior(w, N_HEADLINE_OBS, DRAWS, SEED + s)) for s in range(6)}
    assert len(arms) == 1, arms


def test_the_hedge_is_local_not_a_safe_default() -> None:
    """The minimax arm is not one fixed cautious choice. Deep inside a cell it IS that cell's arm,
    and the cells are different arms -- so the hedge tracks where the posterior sits rather than
    retreating to something globally bland."""
    local = hedge_is_local()
    assert local["bridge-heavy"]["minimax"] == local["bridge-heavy"]["plug_in"] == "agentic"
    assert local["global-heavy"]["minimax"] == local["global-heavy"]["plug_in"] == "graph"
    assert len({v["minimax"] for v in local.values()}) >= 2, local
    # and it does diverge somewhere, or there would be no topic
    assert any(not v["agree"] for v in local.values()), local


def test_a_boundary_costs_two_orders_of_magnitude_more_data() -> None:
    """THE claim. How much traffic you must observe before the hedge and the plug-in rule agree
    depends entirely on where you sit: deep inside a cell the answer is immediate, and at a
    boundary it is hundreds of times larger. The question 'how many queries is enough' has a
    comfortable answer everywhere except where it matters."""
    n_boundary = agreement_n(bridge_family(0.43))
    n_interior = agreement_n(bridge_family(0.85))
    assert n_interior is not None and n_interior <= 20, n_interior
    assert n_boundary is not None and n_boundary >= 50 * n_interior, (n_boundary, n_interior)
    # the disagreement is a NARROW band, not a general phenomenon -- most of the family agrees
    scan = boundary_scan(np.linspace(0.0, 1.0, 41))
    disagree = [s["frac"] for s in scan if not s["agree"]]
    assert 0 < len(disagree) <= 4, disagree
    assert all(0.3 < f < 0.6 for f in disagree), disagree


def test_regret_decays_monotonically_but_not_as_one_over_root_n() -> None:
    """Uncertainty gets cheaper with data, monotonically. It does NOT obey a clean 1/sqrt(n) law
    at this scale: the halving ratios approach 1/sqrt(2) from ABOVE rather than sitting on it, so
    the asserted claims are the monotone decrease and the tail band -- never a constant."""
    curve = regret_decay(bridge_family(0.43))
    vals = [r for _n, r in curve]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])), vals
    assert vals[0] > 5 * vals[-1], (vals[0], vals[-1])
    ratios = halving_ratios(curve)
    assert all(0.6 < r < 1.01 for r in ratios), ratios
    assert ratios[-1] < ratios[0], ratios            # approaching the law, from above
    assert np.mean(ratios[-3:]) < 0.80, ratios       # the tail is near 1/sqrt(2) = 0.707
    # the interior costs nothing almost immediately -- the contrast that makes the boundary sharp
    inner = [r for _n, r in regret_decay(bridge_family(0.85))]
    assert inner[1] == 0.0 and vals[1] > 0.05, (inner[:2], vals[:2])


def test_the_rules_disagree_often_enough_to_matter() -> None:
    """Over workloads drawn without regard to where the boundaries are -- which is the situation a
    deployment is actually in -- the two rules part company on a small but real fraction."""
    rate = disagreement_rate()
    assert 0.02 < rate < 0.30, rate


def test_hedging_is_not_free() -> None:
    """The honest counterweight, and the reason this is a trade rather than an upgrade. Both
    Bayes and minimax shrink the estimate toward the middle of the simplex; that is a bet, and the
    bet LOSES when the truth genuinely sits in a narrow cell just past a boundary, where the
    plug-in rule was correctly confident. Asserted in both directions."""
    rows = cost_of_hedging()
    helped = [r for r in rows if r["minimax"] < r["plug_in"] - 1e-9]
    hurt = [r for r in rows if r["hedge_worse"]]
    assert helped, rows
    assert hurt, rows
    # The sign structure is the mechanism, and it is systematic rather than noisy: the prior pulls
    # every estimate toward the middle of the simplex, so the hedge helps on the central side of
    # the boundary and hurts on the far side, where the truth sits in the narrow cell.
    assert all(r["frac"] < 0.44 for r in helped), helped
    assert all(r["frac"] > 0.43 for r in hurt), hurt
    # And the trade is roughly even, NOT insurance bought cheaply: the worst loss is comparable to
    # the best gain -- measured slightly LARGER, which is stated rather than rounded away.
    best_gain = max(r["plug_in"] - r["minimax"] for r in helped)
    worst_loss = max(r["minimax"] - r["plug_in"] for r in hurt)
    assert 0.5 < best_gain / worst_loss < 2.0, (best_gain, worst_loss)


def test_kernel_matches_the_imported_utility() -> None:
    """The vectorized kernel is a REWRITE of the predecessor's utility, not a second opinion about
    it. Asserted draw by draw against the imported function, to floating point."""
    for w_hat in (local_only(), bridge_family(0.43), uniform_within_regime()):
        post = posterior(np.asarray(w_hat, dtype=float), 40, draws=64)
        fast = utilities_over(post)
        slow = np.array([utility(w, LAMBDA_HEADLINE, RHO_HEADLINE, N_HEADLINE) for w in post])
        assert np.allclose(fast, slow, atol=1e-12), float(np.abs(fast - slow).max())
    # ...including at a non-default price, so the kernel's parameters are genuinely threaded
    post = posterior(uniform_within_regime(), 25, draws=32)
    fast = utilities_over(post, lam=1e-3, rho=10.0, n_queries=50)
    slow = np.array([utility(w, 1e-3, 10.0, 50) for w in post])
    assert np.allclose(fast, slow, atol=1e-12), float(np.abs(fast - slow).max())



def test_all_three_matches_the_individual_rules() -> None:
    """`all_three` evaluates the posterior ONCE and derives every rule from that single matrix,
    where calling the rules separately would evaluate it three times and discard two. That is an
    optimization, so it needs an anchor: the fast route must return exactly what the slow one
    does, on every workload and observation count tried."""
    for w in (local_only(), bridge_family(0.43), bridge_family(0.85), uniform_within_regime()):
        w = np.asarray(w, dtype=float) / np.sum(w)
        for n in (10, 40, 1280):
            fast = all_three(w, n)
            post = posterior(w, n)
            assert fast["plug_in"] == rule_plug_in(w)
            assert fast["bayes"] == rule_bayes(post)
            assert fast["minimax"] == rule_minimax(post)



def test_laboratory_constants_match_the_module() -> None:
    """The laboratory's baked block is EMITTED from viz_constants(), but a retune can regenerate
    one and not the other. This parses the shipped .tsx back and compares it to a fresh bake --
    JSON blocks by name, and the several-to-a-line scalars by regex, because the predecessor's
    guard missed exactly those."""
    import json
    import re
    tsx = pathlib.Path(__file__).resolve().parents[2] / "src/components/viz/WorkloadUncertaintyLaboratory.tsx"
    if not tsx.exists():
        return
    text, v = tsx.read_text(), viz_constants()
    baked = {}
    for name, expr in re.findall(r"^const ([A-Z_0-9]+)(?:: [^=]+)? = (.+?);\s*$", text, re.M):
        try:
            baked[name] = json.loads(expr.replace(" as const", ""))
        except json.JSONDecodeError:
            continue
    for name, key in (("ARMS", "arms"), ("CONDITIONS", "conditions"), ("KERNEL_M", "kernel_M"),
                      ("KERNEL_CONST", "kernel_const"), ("N_OBSERVED", "n_observed"),
                      ("GRID_FRACS", "grid_fracs"), ("MINIMAX_GRID", "minimax_grid"),
                      ("DECAY_BOUNDARY", "decay_boundary"), ("DECAY_INTERIOR", "decay_interior"),
                      ("HALVING", "halving_boundary"), ("AGREEMENT_N", "agreement_n"),
                      ("COST_OF_HEDGING", "cost_of_hedging"), ("REALIZED", "realized")):
        assert name in baked, f"{name} is no longer baked into the laboratory"
        assert baked[name] == v[key], (name, "laboratory and module disagree")
    for name, key in (("PRIOR_ALPHA", "prior_alpha"), ("REGRET_Q", "regret_q"), ("DRAWS", "draws"),
                      ("N_HEADLINE_OBS", "n_headline_obs"),
                      ("DISAGREEMENT_RATE", "disagreement_rate"),
                      ("LAM", "lambda_headline"), ("RHO", "rho_headline")):
        hit = re.search(rf"\b{name} = ([0-9.eE+-]+)", text)
        assert hit, f"{name} is no longer baked into the laboratory"
        assert float(hit.group(1)) == float(v[key]), (name, float(hit.group(1)), v[key])


def test_topic_prose_matches_the_module() -> None:
    """Numbers the topic states in words are the ones nothing else checks."""
    import re
    mdx = pathlib.Path(__file__).resolve().parents[2] / "src/content/topics/rag-architecture-workload-uncertainty.mdx"
    if not mdx.exists():
        return
    text, v = mdx.read_text(), viz_constants()
    for pattern, value in (
        (r"agree from\s+n\s*=\s*([0-9]+)[^0-9]{0,40}inside a cell", float(v["agreement_n"]["interior"])),
        (r"still paying at\s+n\s*=\s*([0-9]+)", float(v["agreement_n"]["boundary"])),
        (r"differ on\s+([0-9.]+)% of", round(v["disagreement_rate"] * 100, 2)),
        (r"quantile of the regret distribution over\s+([0-9]+)\s+draws", float(v["draws"])),
    ):
        hit = re.search(pattern, text, re.S)
        assert hit, f"the topic no longer states: {pattern}"
        assert float(hit.group(1)) == value, (pattern, float(hit.group(1)), value)



def _run_tests() -> None:
    names = sorted(n for n, v in globals().items() if n.startswith("test_") and callable(v))
    for n in names:
        globals()[n]()
        print(f"  ok  {n}")
    print(f"{len(names)} assertions passed")


if __name__ == "__main__":
    print("THE THREE RULES, across the cell boundary the predecessor measured")
    print(f"  {'bridge share':>13s} {'plug-in':>11s} {'Bayes':>11s} {'minimax':>11s}   agree")
    for f in (0.20, 0.34, 0.40, 0.43, 0.46, 0.52, 0.85):
        r = all_three(bridge_family(f), N_HEADLINE_OBS)
        mark = "" if r["plug_in"] == r["minimax"] else "   <- the hedge parts company"
        print(f"  {f:13.2f} {r['plug_in']:>11s} {r['bayes']:>11s} {r['minimax']:>11s}{mark}")
    print()
    print("Bayes is not a third rule: utility is linear in w, so it IS plug-in at the")
    print("posterior mean -- the estimate shrunk toward the prior. The shrinkage is the gap:")
    for n in (10, 40, 160, 1280):
        print(f"  n={n:5d}  posterior-mean bridge share = {shrinkage(bridge_family(0.43), n):.4f}"
              f"   (the estimate was 0.4300)")
    print()
    print("How much traffic before the hedge and the plug-in rule agree?")
    for label, w in (("deep inside a cell", bridge_family(0.85)),
                     ("uniform", uniform_within_regime()),
                     ("local-only", local_only()),
                     ("ON the boundary", bridge_family(0.43))):
        print(f"  {label:>19s}: n = {agreement_n(w)}")
    print()
    print("what the uncertainty costs, as data accumulates (worst-case regret of the hedge):")
    for n, r in regret_decay(bridge_family(0.43)):
        print(f"  n={n:5d}  {r:.4f}  {'#' * int(round(r * 200))}")
    print(f"  halving ratios {halving_ratios(regret_decay(bridge_family(0.43)))}")
    print(f"  (a clean 1/sqrt(n) law would hold these at {1 / np.sqrt(2):.3f} — they approach it,")
    print("   from above, rather than sitting on it)")
    print()
    print("and what the hedge COSTS — realized regret against the truth, either side of 0.43:")
    print(f"  {'bridge share':>13s} {'plug-in':>9s} {'Bayes':>9s} {'minimax':>9s}")
    for r in cost_of_hedging():
        mark = "   <- hedging LOSES here" if r["hedge_worse"] else ""
        print(f"  {r['frac']:13.2f} {r['plug_in']:9.5f} {r['bayes']:9.5f} {r['minimax']:9.5f}{mark}")
    print()
    _run_tests()
