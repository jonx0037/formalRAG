"""The Arm You Did Not Run: learning an architecture under a switching cost.

Four topics have now chosen a retrieval architecture, each removing one assumption from the one
before it. `rag-architecture-mechanisms` measured what six named patterns buy and where each one
fails. `rag-architecture-pareto` showed that "which should we use?" resolves only after a workload
mixture, a price ratio and an amortization horizon are fixed. `rag-architecture-workload-
uncertainty` unfixed the mixture and made it something you estimate. `rag-architecture-switching-
hysteresis` let it move.

All four share an assumption none of them names: THE TABLE IS KNOWN. Every one of them reads the
6 x 6 x 3 quality-and-cost measurement as ground truth and reasons about which row to pick. A
deployment has no such table. It observes only the arm it actually ran, on the queries that
actually arrived; the other five rows are counterfactual. The table is not an input to the
decision -- it is estimated BY the decision, and every query spent measuring a bad arm is a query
answered badly.

That turns the arc's decision problem into a LEARNING problem, and three things fall out of it
that the arc could not have seen:

  1. Against the best FIXED arm in hindsight, a bandit can achieve NEGATIVE regret -- it beats any
     single architecture held for the whole deployment. That is not a paradox: when the best arm
     changes partway through, no fixed arm is good throughout, so the fixed-arm benchmark every
     previous topic optimized is a weak one.
  2. The previous topic's instrument -- a constant margin a challenger must clear before you move
     -- does not merely need retuning here. It STOPS BEING AN INSTRUMENT: across its whole range
     the regret differences vanish into noise, because a constant carries no notion of how well
     each arm is currently known.
  3. A switching cost is a tax on exploration, and it is levied in proportion to how much
     exploring a policy does. Past a high enough price the cheap rule that rarely moves wins, not
     because it is better informed but because it stopped paying.

Nothing about retrieval is re-measured. The six architectures, their per-query outcomes and their
costs arrive unchanged through the import chain; the only new object is the feedback structure.

Run:  uv run --with numpy --with scipy --with scikit-learn python rag_architecture_bandit_exploration.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in ("mechanisms", "pareto", "workload-uncertainty", "switching-hysteresis"):
    sys.path.insert(0, str(_HERE.parent / f"rag-architecture-{_p}"))

import rag_architecture_mechanisms as MECH                          # noqa: E402
import rag_architecture_pareto as PAR                               # noqa: E402
import rag_architecture_switching_hysteresis as HYS                 # noqa: E402
import rag_architecture_workload_uncertainty as WU                  # noqa: E402
sys.path.insert(0, str(_HERE.parent / "significance-testing-calibration"))
from significance_testing_calibration import paired_t_test           # noqa: E402

SEED = 20260921
N_PER_STEP = HYS.N_PER_STEP          # queries observed per window, from the predecessor
N_DEPLOY = 24                        # deployments averaged over; a single run is one draw
UCB_C = 0.5                          # the exploration constant multiplying sqrt(2 log t / n)
MARGIN_GRID = (-0.20, -0.12, -0.08, -0.05, -0.03, -0.015,
               0.0, 0.015, 0.03, 0.05, 0.10, 0.20)
SWITCH_COSTS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.20)
COST_FINE = tuple(round(0.02 * i, 2) for i in range(61))   # for locating the crossover
COST_HEADLINE = 0.0
T_GRID = (300, 600, 1200, 2400, 4800)     # for the rate movement
STATIONARY_SHARE = 0.50                   # a fixed workload, for the instance-dependent rate

ARM_NAMES = tuple(PAR.ARM_NAMES)
# The LOCAL regime only. The global regime is a different corpus with a different query pool, and
# a bandit that could switch corpora mid-deployment would be measuring two things at once.
LOCAL_CLASSES = tuple(c for c in PAR.CONDITIONS if c != "global")


# =================================================================================================
# The feedback structure -- what a deployment can and cannot see.
# =================================================================================================

_TABLE: dict = {}


def reward_table() -> dict:
    """The per-query outcome of EVERY arm on EVERY local query, plus each arm's per-class cost.

    This is the object the whole topic is about, and the point is that a deployment never sees it.
    It exists here only so that regret can be scored: the simulation shows a policy one column of
    it at a time -- the arm it chose, on the query that arrived -- and the rest is counterfactual.

    `ok[a][i]` is 1 when arm `a` answered query `i` correctly, and comes straight from the imported
    `mechanism_matrix`; nothing is re-measured.
    """
    if _TABLE:
        return _TABLE
    corpus = MECH.local()
    matrix = MECH.mechanism_matrix()
    ok = np.array([matrix[a]["local"]["ok"] for a in ARM_NAMES])          # (6, 100) in {0, 1}
    klass = np.asarray(corpus["klass"])
    by_class = {c: np.flatnonzero(klass == c) for c in LOCAL_CLASSES}
    for c, ids in by_class.items():
        if ids.size == 0:
            raise ValueError(f"the local corpus has no queries of class {c!r}")
    t = PAR.measure()
    ci = {c: PAR.CONDITIONS.index(c) for c in LOCAL_CLASSES}
    lam, rho = WU.LAMBDA_HEADLINE, WU.RHO_HEADLINE
    cost = np.array([[lam * (t["O"][a][ci[c]] + rho * t["C"][a][ci[c]]) for c in LOCAL_CLASSES]
                     for a in range(len(ARM_NAMES))])                     # (6, 5)
    quality = np.array([[ok[a][by_class[c]].mean() for c in LOCAL_CLASSES]
                        for a in range(len(ARM_NAMES))])                  # (6, 5)
    _TABLE.update({"ok": ok, "klass": klass, "by_class": by_class,
                   "cost": cost, "utility": quality - cost})
    return _TABLE


def local_mixture(bridge_share: float) -> np.ndarray:
    """The workload as a mixture over the five local classes, parameterized the way the predecessor
    parameterized it: a bridge share, with the remainder spread evenly over the other four.

    GUARD: a share outside [0, 1] is not a mixture.
    """
    if not 0.0 <= bridge_share <= 1.0:
        raise ValueError(f"a bridge share must lie in [0, 1], got {bridge_share}")
    n = len(LOCAL_CLASSES)
    w = np.full(n, (1.0 - bridge_share) / (n - 1))
    w[LOCAL_CLASSES.index("bridge")] = bridge_share
    return w


def true_utility(bridge_share: float) -> np.ndarray:
    """Every arm's expected utility at a given workload -- the quantity a policy is trying to
    learn, and the one regret is scored against. Never shown to a policy."""
    return reward_table()["utility"] @ local_mixture(bridge_share)


_STREAMS: dict = {}


def query_stream(seed: int = SEED, path: np.ndarray | None = None,
                 n_per: int = N_PER_STEP) -> dict:
    """One deployment's arriving traffic, and what every arm WOULD have scored on it.

    The drift path is the predecessor's: the bridge share ramps while the rest of the mixture
    gives way. `rewards[a][t]` is what arm `a` would have returned on query `t` -- the simulation
    reveals exactly one row of that column to the policy, which is the whole point.

    Memoized per (seed, path, n_per): the stream depends on none of the policy parameters, while
    every sweep below varies those, so re-drawing it per sweep point is the loop-invariant
    recomputation the predecessor had to be refactored to avoid.
    """
    if n_per < 1:
        raise ValueError(f"a window must observe at least one query, got {n_per}")
    path = HYS.drift_path("ramp") if path is None else path
    key = (seed, path.tobytes(), n_per)
    if key in _STREAMS:
        return _STREAMS[key]
    tbl = reward_table()
    rng = np.random.default_rng(seed)
    cls, qid = [], []
    for f in path:
        w = local_mixture(float(f))
        for _ in range(n_per):
            c = int(rng.choice(len(LOCAL_CLASSES), p=w))
            cls.append(c)
            qid.append(int(rng.choice(tbl["by_class"][LOCAL_CLASSES[c]])))
    cls, qid = np.asarray(cls), np.asarray(qid)
    _STREAMS[key] = {"cls": cls, "qid": qid,
                     "rewards": tbl["ok"][:, qid] - tbl["cost"][:, cls],
                     "truth": tbl["utility"][:, cls]}
    return _STREAMS[key]


def run_bandit(stream: dict, kind: str = "ucb", margin: float = 0.0, c: float = UCB_C,
               window: int | None = None, seed: int = 0) -> dict:
    """One deployment under one policy. Returns the switching-cost-FREE regret and the switch
    count separately, because total cost is affine in the price of a switch:

        total(sc) = regret + sc * switches

    so an entire price sweep is arithmetic on one run rather than a re-simulation per price.

    The policies, all seeing only the arm they chose:
      greedy   -- argmax of the running mean; no notion of what it has not measured
      margin   -- the PREDECESSOR's rule: hold unless a challenger beats you by `margin`
      ucb      -- argmax of mean + c*sqrt(2 log t / n): the bonus IS the uncertainty
      thompson -- argmax of a draw from each arm's posterior

    GUARDS: an unknown kind, a non-positive window.
    """
    if kind not in ("greedy", "margin", "ucb", "thompson"):
        raise ValueError(f"unknown policy {kind!r}")
    if window is not None and window < 1:
        raise ValueError(f"a window must be at least one observation, got {window}")
    if kind == "thompson" and window is not None:
        # The posterior draw below takes its mean from the window but its variance from the total
        # pull count, which would understate uncertainty for exactly the arms a window exists to
        # make uncertain. Nothing calls this combination; refuse it rather than ship it latent.
        raise ValueError("windowed Thompson sampling is not implemented coherently; see run_bandit")
    rew, truth = stream["rewards"], stream["truth"]
    n_arms, horizon = rew.shape
    rng = np.random.default_rng(seed)
    n = np.zeros(n_arms)
    s = np.zeros(n_arms)
    hist: list[list[float]] = [[] for _ in range(n_arms)]
    cur, switches = 0, 0
    chosen = np.zeros(horizon, dtype=int)
    # The initialization order is SHUFFLED per deployment, and that is load-bearing rather than
    # cosmetic. Pulling the arms in index order leaves the policy holding arm 5 when the warm-up
    # ends -- and arm 5 is `agentic`, which happens to be the best fixed arm on this corpus. A
    # wide margin would then score well for the sole reason that it never left an arm it was
    # handed for free, which is an artifact of the loop, not a property of hysteresis.
    order = rng.permutation(n_arms)
    for t in range(horizon):
        if t < n_arms:                      # initialization: one pull of every arm
            a = int(order[t])
            switches += int(t > 0)
        else:
            if window is None:
                mu = s / np.maximum(n, 1.0)
            else:
                mu = np.array([np.mean(h[-window:]) if h else 0.0 for h in hist])
            if kind == "greedy":
                a = int(np.argmax(mu))
            elif kind == "margin":
                rival = int(np.argmax(np.where(np.arange(n_arms) == cur, -np.inf, mu)))
                a = rival if mu[rival] - mu[cur] > margin else cur
            elif kind == "ucb":
                eff = np.array([min(len(h), window) if window else n[k]
                                for k, h in enumerate(hist)], dtype=float)
                a = int(np.argmax(mu + c * np.sqrt(2.0 * np.log(t + 1) / np.maximum(eff, 1.0))))
            else:
                a = int(np.argmax(rng.normal(mu, 1.0 / np.sqrt(np.maximum(n, 1.0)))))
            switches += int(a != cur)
        cur = a
        chosen[t] = a
        n[a] += 1.0
        s[a] += rew[a, t]
        hist[a].append(float(rew[a, t]))
    held = truth[chosen, np.arange(horizon)].sum()
    totals = truth.sum(axis=1)              # what each arm would have earned if held throughout
    return {"regret": float(totals.max() - held), "switches": int(switches),
            "pulls": n, "chosen": chosen, "best_fixed": int(np.argmax(totals)),
            "dynamic_regret": float(truth.max(axis=0).sum() - held)}


_POLICY: dict = {}


def over_deployments(kind: str, n_deploy: int = N_DEPLOY, **kw) -> dict:
    """The same policy over several deployments. A single run is one draw of exactly the noise
    this topic is about, so nothing is reported from one.

    Memoized on the full argument set: the sweeps below ask for the same policy repeatedly (the
    price sweep alone re-reads every margin at every price), and a policy's outcome depends on
    none of the things those sweeps vary. Callers only read the result.
    """
    key = (kind, n_deploy, tuple(sorted(kw.items())))
    if key in _POLICY:
        return _POLICY[key]
    runs = [run_bandit(query_stream(SEED + d), kind, seed=SEED + d, **kw) for d in range(n_deploy)]
    reg = np.array([r["regret"] for r in runs])
    _POLICY[key] = {"regret": float(reg.mean()), "std": float(reg.std()),
                    "se": float(reg.std() / np.sqrt(len(reg))),
                    "switches": float(np.mean([r["switches"] for r in runs])),
                    "beats_fixed": int((reg < 0).sum()), "n": len(reg),
                    "per_deployment": reg, "runs": runs}
    return _POLICY[key]


# =================================================================================================
# Movement 2 -- learning the table, and what the fixed-arm benchmark is worth.
# =================================================================================================

def policy_table(n_deploy: int = N_DEPLOY) -> list[dict]:
    """Every policy on the same deployments. Regret is measured against the best FIXED arm in
    hindsight -- the benchmark every previous topic in the arc optimized.

    The result that reframes the arc: an uncertainty-weighted policy goes NEGATIVE against that
    benchmark. It is not beating an oracle; it is beating a benchmark that drift has made weak,
    because no single architecture is the right one for the whole deployment.
    """
    specs = (("greedy", {}), ("margin", {"margin": 0.10}), ("ucb", {}), ("thompson", {}),
             ("ucb", {"window": 400}))
    out = []
    for kind, kw in specs:
        r = over_deployments(kind, n_deploy=n_deploy, **kw)
        label = kind if not kw else f"{kind} " + " ".join(f"{k}={v:g}" for k, v in kw.items())
        out.append({"policy": label, "regret": r["regret"], "std": r["std"], "se": r["se"],
                    "switches": r["switches"], "beats_fixed": r["beats_fixed"], "n": r["n"]})
    return out


def dynamic_vs_static(n_deploy: int = N_DEPLOY) -> dict:
    """What the fixed-arm benchmark leaves on the table. `static_gap` is how far the best single
    arm falls short of choosing the best arm at every moment -- the value the whole arc has been
    declining to collect by committing to one row."""
    gaps, statics = [], []
    for d in range(n_deploy):
        st = query_stream(SEED + d)
        truth = st["truth"]
        statics.append(float(truth.sum(axis=1).max()))
        gaps.append(float(truth.max(axis=0).sum() - truth.sum(axis=1).max()))
    return {"static_gap": float(np.mean(gaps)), "best_fixed_total": float(np.mean(statics)),
            "best_fixed_arm": ARM_NAMES[int(np.argmax(query_stream(SEED)["truth"].sum(axis=1)))]}


# =================================================================================================
# Movement 3 -- the predecessor's instrument, applied where its premise does not hold.
# =================================================================================================

def margin_sweep(n_deploy: int = N_DEPLOY, margins=MARGIN_GRID) -> list[dict]:
    """The predecessor's whole instrument, swept, with the table now unknown.

    The predecessor found a clean interior optimum in this parameter. Here the differences across
    its entire range vanish into deployment-to-deployment noise: the honest reading is not that
    the optimum moved but that there is no longer an optimum to find, because a CONSTANT margin
    carries no information about how well any particular arm is currently known.
    """
    drop = ("per_deployment", "runs")          # per-run detail, never baked
    return [{"margin": float(m), **{k: v for k, v in over_deployments(
        "margin", n_deploy=n_deploy, margin=float(m)).items() if k not in drop}}
        for m in margins]


def margins_within_noise(n_deploy: int = N_DEPLOY) -> dict:
    """How many of the swept margins are indistinguishable from the best one. This is the movement's
    actual claim, and it is a claim about RESOLUTION rather than about a value."""
    rows = margin_sweep(n_deploy)
    best = min(rows, key=lambda r: r["regret"])
    tied = [r["margin"] for r in rows if r["regret"] <= best["regret"] + best["se"]]
    ucb = over_deployments("ucb", n_deploy=n_deploy)
    return {"best_margin": best["margin"], "best_regret": best["regret"], "se": best["se"],
            "n_tied": len(tied), "n_total": len(rows), "tied": tied,
            "spread_ratio": float(best["std"] / ucb["std"]),
            "margin_std": best["std"], "ucb_std": ucb["std"], "ucb_regret": ucb["regret"]}


def paired_against_ucb(n_deploy: int = N_DEPLOY) -> list[dict]:
    """Each policy against UCB on the SAME deployments. Pairing cancels the shared difficulty of a
    deployment -- some traffic draws are simply harder than others -- which is the discipline
    `significance-testing-calibration` established and whose test is imported here rather than
    rewritten. The unpaired standard errors above overlap; the paired differences do not."""
    ref = over_deployments("ucb", n_deploy=n_deploy)["per_deployment"]
    specs = (("greedy", {}), ("margin margin=0.1", {"margin": 0.10}),
             ("thompson", {}), ("ucb window=400", {"window": 400}))
    out = []
    for label, kw in specs:
        kind = label.split()[0]
        arm = over_deployments(kind, n_deploy=n_deploy, **kw)["per_deployment"]
        t = paired_t_test(arm - ref)
        out.append({"policy": label, "diff": t["mean"], "t": t["t"], "p": t["p"],
                    "ci_lo": t["ci_lo"], "ci_hi": t["ci_hi"]})
    return out


# =================================================================================================
# Movement 4 -- a switching cost is a tax on exploration.
# =================================================================================================

def switching_cost_table(costs=SWITCH_COSTS, n_deploy: int = N_DEPLOY) -> list[dict]:
    """Total cost = regret + price * switches, swept over the price.

    A policy's TRAJECTORY does not depend on the price -- only the bill does -- so each policy is
    simulated once and every price is arithmetic on that one run. The same linearity the
    predecessor had to be refactored to exploit.

    The tax is levied in proportion to how much a policy explores, so the ordering by switch count
    is the ordering by how fast each policy's total degrades.
    """
    base = {"greedy": over_deployments("greedy", n_deploy=n_deploy),
            "ucb": over_deployments("ucb", n_deploy=n_deploy),
            "thompson": over_deployments("thompson", n_deploy=n_deploy)}
    margins = {m: over_deployments("margin", n_deploy=n_deploy, margin=float(m))
               for m in MARGIN_GRID}
    out = []
    for sc in costs:
        row = {"cost": float(sc)}
        for k, r in base.items():
            row[k] = r["regret"] + sc * r["switches"]
        best_m = min(margins.items(), key=lambda kv: kv[1]["regret"] + sc * kv[1]["switches"])
        row["margin"] = best_m[1]["regret"] + sc * best_m[1]["switches"]
        row["best_margin"] = float(best_m[0])
        row["winner"] = min(("greedy", "ucb", "thompson", "margin"), key=lambda k: row[k])
        out.append(row)
    return out


def exploration_crossover(costs=SWITCH_COSTS, n_deploy: int = N_DEPLOY) -> dict:
    """The price at which exploring stops paying for itself, and the policy that takes over."""
    rows = switching_cost_table(costs, n_deploy)
    flip = next((r for r in rows if r["winner"] != rows[0]["winner"]), None)
    return {"winner_at_zero": rows[0]["winner"], "crossover_cost": flip and flip["cost"],
            "winner_after": flip and flip["winner"],
            "switches": {k: over_deployments(k, n_deploy=n_deploy)["switches"]
                         for k in ("greedy", "ucb", "thompson")}}


# =================================================================================================
# Movement 5 -- why the famous rate cannot be read off this corpus.
# =================================================================================================

def stationary_stream(seed: int, share: float, horizon: int) -> dict:
    """A NON-drifting deployment: one fixed workload, so the classical stationary theory applies."""
    tbl = reward_table()
    rng = np.random.default_rng(seed)
    w = local_mixture(share)
    cls = rng.choice(len(LOCAL_CLASSES), size=horizon, p=w)
    qid = np.array([int(rng.choice(tbl["by_class"][LOCAL_CLASSES[c]])) for c in cls])
    return {"cls": cls, "qid": qid, "rewards": tbl["ok"][:, qid] - tbl["cost"][:, cls],
            "truth": tbl["utility"][:, cls]}


_RATE: dict = {}


def rate_study(horizons=T_GRID, share: float = STATIONARY_SHARE, n_deploy: int = 12) -> dict:
    """How UCB's regret grows with the horizon, on a FIXED instance.

    This movement is a negative result, and it is the honest one. The celebrated rates -- sqrt(T)
    without a switching cost, T^(2/3) with one -- are MINIMAX rates: worst cases over instances,
    where the gap between arms is allowed to shrink with the horizon so that the problem stays
    hard. A fixed corpus has a fixed gap, and on a fixed gap UCB's regret is logarithmic. So no
    amount of simulating this corpus at larger T will produce those exponents, and a fitted slope
    that happens to land between them is measuring neither.
    """
    key = (tuple(horizons), share, n_deploy)
    if key in _RATE:
        return _RATE[key]
    obs = []
    for horizon in horizons:
        runs = [run_bandit(stationary_stream(SEED + d, share, horizon), "ucb", seed=SEED + d)
                for d in range(n_deploy)]
        obs.append(float(np.mean([r["regret"] for r in runs])))
    lg = np.log(np.asarray(horizons, dtype=float))
    slope = float(np.polyfit(lg, np.log(np.maximum(obs, 1e-9)), 1)[0])
    log_fit = np.polyfit(lg, obs, 1)                    # regret against log T, the instance rate
    resid = obs - np.polyval(log_fit, lg)
    ss = float(1.0 - np.sum(resid**2) / np.sum((obs - np.mean(obs))**2))
    _RATE[key] = {"horizons": list(horizons), "regret": obs, "loglog_slope": slope,
                  "log_r2": ss, "sqrt_slope": 0.5, "two_thirds_slope": 2.0 / 3.0}
    return _RATE[key]


def regret_trajectory(kind: str, n_deploy: int = N_DEPLOY, every: int = 24, **kw) -> list[float]:
    """Cumulative regret against the best fixed arm, averaged over deployments and subsampled.

    It starts NEGATIVE-going for an exploring policy only once the workload has drifted far enough
    that the best fixed arm is the wrong arm for the moment -- which is the whole point of the
    second movement made visible.
    """
    acc = None
    runs = over_deployments(kind, n_deploy=n_deploy, **kw)["runs"]
    for d, r in enumerate(runs):
        st = query_stream(SEED + d)
        truth = st["truth"]
        best_fixed = int(np.argmax(truth.sum(axis=1)))
        step = truth[best_fixed] - truth[r["chosen"], np.arange(truth.shape[1])]
        acc = np.cumsum(step) if acc is None else acc + np.cumsum(step)
    mean = acc / n_deploy
    # Subsample, but ALWAYS include the final query. A bare `[::every]` stops at index 2376 of
    # 2400, so the plotted curve ended ~7 units short of the regret the panel's own readout
    # reports beside it, under an axis labelled with the full horizon.
    idx = list(range(0, len(mean), every))
    if idx[-1] != len(mean) - 1:
        idx.append(len(mean) - 1)
    return [round(float(mean[i]), 3) for i in idx]


def viz_constants() -> dict:
    r3 = lambda v: round(float(v), 3)                                          # noqa: E731
    tbl = reward_table()
    path = HYS.drift_path("ramp")
    cross = exploration_crossover(COST_FINE)
    return {
        "arms": list(ARM_NAMES), "classes": list(LOCAL_CLASSES),
        "n_per_step": N_PER_STEP, "n_deploy": N_DEPLOY, "ucb_c": UCB_C,
        "horizon": int(len(path) * N_PER_STEP), "t_steps": int(len(path)),
        "margin_grid": [float(m) for m in MARGIN_GRID],
        "switch_costs": [float(c) for c in SWITCH_COSTS],
        "path": [r3(f) for f in path],
        # the table a deployment never sees: every arm's outcome on every query
        "ok": [[int(v) for v in row] for row in tbl["ok"]],
        "klass": [LOCAL_CLASSES.index(c) for c in tbl["klass"]],
        "utility": [[r3(v) for v in row] for row in tbl["utility"]],
        "policies": [{k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()}
                     for row in policy_table()],
        "paired": [{k: (float(f"{v:.6g}") if isinstance(v, float) else v)
                    for k, v in row.items()} for row in paired_against_ucb()],
        "margin_sweep": [{k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()}
                         for row in margin_sweep()],
        "margins_tied": {k: (r3(v) if isinstance(v, float) else v)
                         for k, v in margins_within_noise().items()},
        "cost_table": [{k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()}
                       for row in switching_cost_table()],
        "crossover": {"cost": cross["crossover_cost"], "before": cross["winner_at_zero"],
                      "after": cross["winner_after"],
                      "switches": {k: r3(v) for k, v in cross["switches"].items()}},
        "dynamic": {k: (r3(v) if isinstance(v, float) else v)
                    for k, v in dynamic_vs_static().items()},
        "rate": {k: ([r3(x) for x in v] if isinstance(v, list) else r3(v))
                 for k, v in rate_study().items()},
        # trajectories for the regret panel, subsampled; the browser plots these directly
        "trajectories": {lab: regret_trajectory(kind, **kw) for lab, kind, kw in
                         (("greedy", "greedy", {}), ("margin", "margin", {"margin": 0.10}),
                          ("ucb", "ucb", {}), ("thompson", "thompson", {}))},
    }


# =================================================================================================
# Collapse anchors -- the learning problem must reduce to something already known at its edges.
# =================================================================================================

def test_the_reward_stream_is_the_imported_measurement() -> None:
    """Nothing about retrieval is re-measured: every reward is the imported per-query outcome of
    the imported arm on the imported corpus, minus that arm's cost on that query's class."""
    tbl = reward_table()
    st = query_stream(SEED)
    matrix = MECH.mechanism_matrix()
    for a, name in enumerate(ARM_NAMES):
        assert np.array_equal(tbl["ok"][a], matrix[name]["local"]["ok"]), name
    want = tbl["ok"][:, st["qid"]] - tbl["cost"][:, st["cls"]]
    assert np.allclose(st["rewards"], want, atol=0), "the stream is not the imported table"


def test_ucb_with_no_bonus_is_greedy() -> None:
    """The exploration bonus is the ONLY thing separating UCB from a policy that believes its own
    running means. Set the constant to zero and the two are the same policy, step for step."""
    st = query_stream(SEED)
    a = run_bandit(st, "ucb", c=0.0, seed=SEED)
    b = run_bandit(st, "greedy", seed=SEED)
    assert np.array_equal(a["chosen"], b["chosen"]), "ucb(c=0) diverged from greedy"
    assert a["regret"] == b["regret"] and a["switches"] == b["switches"]


def test_an_unreachable_margin_never_switches() -> None:
    """A margin no challenger can clear reduces the policy to 'commit to whatever the warm-up left
    you holding' -- the previous topic's infinite-band anchor, one level up."""
    st = query_stream(SEED)
    r = run_bandit(st, "margin", margin=1e9, seed=SEED)
    assert r["switches"] == len(ARM_NAMES) - 1, r["switches"]
    assert len(set(r["chosen"][len(ARM_NAMES):].tolist())) == 1


def test_total_cost_is_affine_in_the_price_of_a_switch() -> None:
    """The identity the whole cost sweep rests on: a policy's trajectory does not depend on the
    price, so every price is arithmetic on one run rather than a fresh simulation."""
    rows = {r["cost"]: r for r in switching_cost_table()}
    base = over_deployments("ucb")
    for c, row in rows.items():
        assert abs(row["ucb"] - (base["regret"] + c * base["switches"])) < 1e-9, c


def test_a_one_hot_workload_reads_one_column() -> None:
    """The mixture machinery collapses: put all the traffic on one class and every arm's expected
    utility is that class's column of the imported table."""
    tbl = reward_table()
    i = LOCAL_CLASSES.index("bridge")
    assert np.allclose(true_utility(1.0), tbl["utility"][:, i], atol=1e-12)
    w = local_mixture(1.0)
    assert w[i] == 1.0 and np.allclose(np.delete(w, i), 0.0)


def test_guards() -> None:
    """Malformed inputs are refused rather than silently producing a plausible number."""
    for bad in (-0.1, 1.5):
        try:
            local_mixture(bad)
        except ValueError:
            continue
        raise AssertionError(f"local_mixture accepted {bad}")
    st = query_stream(SEED)
    for kw in ({"kind": "nonsense"}, {"kind": "ucb", "window": 0}):
        try:
            run_bandit(st, **kw)
        except ValueError:
            continue
        raise AssertionError(f"run_bandit accepted {kw}")
    try:
        query_stream(SEED, n_per=0)
    except ValueError:
        return
    raise AssertionError("query_stream accepted n_per=0")


# =================================================================================================
# Claim tests -- every headline, and every limit on it.
# =================================================================================================

def test_an_exploring_policy_beats_the_best_fixed_arm() -> None:
    """Movement 2, and the result that reframes the arc. Regret against the best FIXED arm in
    hindsight goes NEGATIVE, on every deployment. No fixed arm is right for the whole deployment
    once the workload drifts, so the benchmark the previous four topics optimized is a weak one.
    Asserted as a unanimous sign, not a value."""
    r = over_deployments("ucb")
    assert r["regret"] < 0, r["regret"]
    assert r["beats_fixed"] == r["n"], (r["beats_fixed"], r["n"])
    assert dynamic_vs_static()["static_gap"] > 0


def test_the_predecessors_instrument_stops_resolving() -> None:
    """Movement 3. The previous topic found a clean interior optimum in the margin; here most of
    the grid is indistinguishable from the best of it. The claim is about RESOLUTION -- there is
    no longer an optimum to find -- so it is asserted as a count of ties, never as a location."""
    m = margins_within_noise()
    assert m["n_tied"] >= 8, m
    assert m["spread_ratio"] > 3.0, m["spread_ratio"]
    assert m["ucb_regret"] < m["best_regret"], m


def test_uncertainty_beats_a_constant_decisively_when_paired() -> None:
    """The unpaired standard errors overlap; pairing cancels the shared difficulty of a
    deployment and the separation is unambiguous. The imported test, not a rewritten one."""
    rows = {r["policy"]: r for r in paired_against_ucb()}
    for k in ("greedy", "margin margin=0.1", "thompson"):
        assert rows[k]["diff"] > 0 and rows[k]["p"] < 1e-4, (k, rows[k])


def test_forgetting_does_not_measurably_help_here() -> None:
    """The honesty test, and a planned claim that did NOT survive. A sliding window is the
    standard remedy for a bandit whose target moves, and it is directionally better here -- but on
    a single slow changeover the difference does not clear the noise at this many deployments.
    Asserted as NOT significant so that a future retune cannot quietly promote it."""
    row = next(r for r in paired_against_ucb() if r["policy"].startswith("ucb window"))
    assert row["p"] > 0.05, row
    assert row["ci_lo"] < 0 < row["ci_hi"], row


def test_a_switching_cost_taxes_exploration_in_proportion() -> None:
    """Movement 4. The tax is levied per switch, so it is levied on exploration, and the ordering
    by how fast each policy's total degrades is the ordering by how much it explores."""
    c = exploration_crossover(COST_FINE)
    sw = c["switches"]
    assert sw["thompson"] > sw["ucb"] > sw["greedy"], sw
    assert c["winner_at_zero"] == "ucb"
    assert c["crossover_cost"] is not None and 0.2 < c["crossover_cost"] < 1.2, c
    assert c["winner_after"] in ("margin", "greedy")
    rows = switching_cost_table()
    for k in ("ucb", "thompson", "greedy"):
        col = [r[k] for r in rows]
        assert all(a <= b + 1e-9 for a, b in zip(col, col[1:])), k     # total rises with price


def test_the_famous_rate_is_not_readable_from_one_corpus() -> None:
    """Movement 5, a negative result and the honest one.

    On a FIXED instance the gap between arms is fixed, and UCB's regret there is logarithmic --
    which the data show emphatically. The sqrt(T) and T^(2/3) rates are MINIMAX: worst cases over
    instances whose gaps shrink with the horizon. So the power-law slope fitted to this corpus is
    measuring a logarithm, and the fact that it lands near 0.5 is exactly the trap: it looks like
    a confirmation of sqrt(T) and is nothing of the kind.
    """
    r = rate_study()
    assert r["log_r2"] > 0.98, r["log_r2"]                       # regret IS linear in log T
    assert abs(r["loglog_slope"] - r["two_thirds_slope"]) > 0.15, r["loglog_slope"]
    assert all(a < b for a, b in zip(r["regret"], r["regret"][1:]))


def test_the_plotted_trajectory_ends_where_the_regret_does() -> None:
    """The regret panel draws a subsampled trajectory under an axis labelled with the full
    horizon, and prints the exact regret in a readout beside it. Those must agree: a bare stride
    stopped 23 queries short and the curve ended ~7 units below its own readout."""
    for label, kind, kw in (("greedy", "greedy", {}), ("ucb", "ucb", {}),
                            ("thompson", "thompson", {})):
        traj = regret_trajectory(kind, **kw)
        exact = over_deployments(kind, **kw)["regret"]
        assert abs(traj[-1] - exact) < 1e-2, (label, traj[-1], exact)


def test_windowed_thompson_is_refused() -> None:
    """A window changes which observations the mean is taken over but not the pull count the
    posterior variance is built from, so the combination understates uncertainty for exactly the
    arms a window exists to make uncertain. Nothing calls it; it is refused rather than latent."""
    try:
        run_bandit(query_stream(SEED), "thompson", window=100)
    except ValueError:
        return
    raise AssertionError("run_bandit accepted windowed Thompson sampling")


def test_the_drift_moves_the_target() -> None:
    """The substrate's own precondition: if the best arm never changed, none of the above would be
    a statement about drift. It changes, and it changes to a different arm than it started on."""
    first, last = true_utility(0.20), true_utility(0.70)
    assert int(np.argmax(first)) != int(np.argmax(last))
    assert ARM_NAMES[int(np.argmax(last))] == dynamic_vs_static()["best_fixed_arm"]


# =================================================================================================

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAB = _ROOT / "src" / "components" / "viz" / "BanditExplorationLaboratory.tsx"
_MDX = _ROOT / "src" / "content" / "topics" / "rag-architecture-bandit-exploration.mdx"


def test_laboratory_constants_match_the_module() -> None:
    """The .tsx block is EMITTED from `viz_constants`, so this checks the emitted file on disk is
    still the one the module would produce today -- the JSON blocks AND the bare scalars, which
    are declared several to a line and quoted directly in the panel notes a reader sees."""
    src = _LAB.read_text()
    v = viz_constants()
    for name, want in (("ARMS", v["arms"]), ("CLASSES", v["classes"]), ("OK", v["ok"]),
                       ("KLASS", v["klass"]), ("UTILITY", v["utility"]),
                       ("POLICIES", v["policies"]), ("PAIRED", v["paired"]),
                       ("MARGIN_SWEEP", v["margin_sweep"]), ("MARGINS_TIED", v["margins_tied"]),
                       ("CROSSOVER", v["crossover"]), ("RATE", v["rate"]),
                       ("TRAJECTORIES", v["trajectories"])):
        m = re.search(rf"^const {name}(?:: [^=]+)? = (.*?)(?: as const)?;$", src, re.M)
        assert m, f"{name} is not in the laboratory"
        assert json.loads(m.group(1)) == json.loads(json.dumps(want)), f"{name} has drifted"
    for name, want in (("N_DEPLOY", N_DEPLOY), ("UCB_C", UCB_C),
                       ("HORIZON", viz_constants()["horizon"])):
        m = re.search(rf"\b{name} = ([0-9.eE+-]+)", src)
        assert m, f"the scalar {name} is not in the laboratory"
        assert abs(float(m.group(1)) - float(want)) < 1e-12, f"{name} has drifted"


def _rows(src: str, header: str) -> list[list[str]]:
    """Data rows of the markdown table whose header line contains `header`, with bold markers,
    padding and the typographic minus sign normalized. MDX wraps prose but never table rows."""
    lines = src.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("|") and header in ln)
    out = []
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        out.append([c.strip().replace("**", "").replace("$", "").replace("−", "-").strip()
                    for c in ln.strip("|").split("|")])
    return out


def test_topic_prose_matches_the_module() -> None:
    """Every number a READER sees is parsed back out of the shipped .mdx and compared to a fresh
    bake -- the tables AND the inline figures, since a retune moves both and only the tables look
    like data."""
    src = _MDX.read_text()
    v = viz_constants()

    pol = {r["policy"]: r for r in v["policies"]}
    name_of = {"greedy": "greedy", "constant margin \\delta = 0.10": "margin margin=0.1",
               "Thompson sampling": "thompson", "UCB": "ucb"}
    rows = _rows(src, "policy | regret | spread | switches")
    assert len(rows) == len(name_of), rows
    for row in rows:
        w = pol[name_of[row[0]]]
        assert abs(float(row[1]) - round(w["regret"], 1)) < 0.05, row
        assert abs(float(row[2]) - round(w["std"], 1)) < 0.05, row
        assert abs(float(row[3]) - round(w["switches"], 1)) < 0.05, row
        assert row[4].replace(" ", "") == f"{w['beats_fixed']}/{w['n']}", row

    sweep = {round(r["margin"], 3): r for r in v["margin_sweep"]}
    for row in _rows(src, "| margin $"):
        w = sweep[round(float(row[0]), 3)]
        assert abs(float(row[1]) - round(w["regret"], 1)) < 0.05, row
        assert abs(float(row[2]) - round(w["se"], 1)) < 0.05, row

    rate = v["rate"]
    for row in _rows(src, "| horizon $"):
        i = rate["horizons"].index(float(row[0]))
        assert abs(float(row[1]) - round(rate["regret"][i], 1)) < 0.05, row

    tied, cross = v["margins_tied"], v["crossover"]
    checks = (
        ("the static gap", r"worth\s*\n?\s*\$([0-9.]+)\$ more", v["dynamic"]["static_gap"], 0.05),
        ("the spread ratio", r"\$([0-9.]+)\\times\$ that of an uncertainty", tied["spread_ratio"], 5e-3),
        ("the spread ratio, restated", r"and \$([0-9.]+)\\times\$ the spread", tied["spread_ratio"], 5e-3),
        ("the crossover price", r"price of \$([0-9.]+)\$ per switch", cross["cost"], 5e-3),
        ("greedy switches", r"switches \$([0-9.]+)\$ times", cross["switches"]["greedy"], 0.05),
        ("ucb switches", r"UCB\s*\n?\s*switches \$([0-9.]+)\$", cross["switches"]["ucb"], 0.05),
        ("the log fit", r"linear with \$R\^2 = ([0-9.]+)\$", rate["log_r2"], 5e-4),
        ("the spurious slope", r"a slope of \$([0-9.]+)\$", rate["loglog_slope"], 5e-4),
    )
    for label, pattern, want, tol in checks:
        m = re.search(pattern, src)
        assert m, f"the prose figure for {label} is missing or reworded"
        assert abs(float(m.group(1)) - float(want)) < tol, f"{label}: prose {m.group(1)} vs {want}"

    m = re.search(r"\*\*(\w+) of the twelve\*\*", src)
    words = {"Ten": 10, "Nine": 9, "Eleven": 11, "Eight": 8, "Seven": 7}
    assert m and words.get(m.group(1)) == tied["n_tied"], (m and m.group(1), tied["n_tied"])


def _run_tests() -> None:
    names = sorted(n for n, v in globals().items() if n.startswith("test_") and callable(v))
    for n in names:
        globals()[n]()
        print(f"  ok  {n}")
    print(f"{len(names)} assertions passed")


# NOTE: this guard stays at the very END of the file. `_run_tests` scans `globals()` at call time,
# so a `def test_*` written BELOW it has not executed yet and is silently skipped -- the printed
# count simply comes out short. If an added test does not RAISE that number, that is the bug.
if __name__ == "__main__":
    tbl = reward_table()
    print("WHAT A DEPLOYMENT CANNOT SEE: every arm's outcome on every query is a "
          f"{tbl['ok'].shape[0]} x {tbl['ok'].shape[1]} table,")
    print("but a deployment observes one entry per query -- the arm it actually ran.")
    print(f"  the best arm at a {0.20:.2f} bridge share: "
          f"{ARM_NAMES[int(np.argmax(true_utility(0.20)))]}")
    print(f"  the best arm at a {0.70:.2f} bridge share: "
          f"{ARM_NAMES[int(np.argmax(true_utility(0.70)))]}   <- the target moves")
    print()
    print("LEARNING THE TABLE — regret against the best FIXED arm in hindsight:")
    print(f"  {'policy':>16s} {'regret':>9s} {'std':>8s} {'switches':>9s}  beats the fixed arm")
    for r in policy_table():
        print(f"  {r['policy']:>16s} {r['regret']:9.1f} {r['std']:8.1f} {r['switches']:9.1f}"
              f"  {r['beats_fixed']}/{r['n']}")
    d = dynamic_vs_static()
    print(f"  the best fixed arm is `{d['best_fixed_arm']}`, and choosing the best arm at every")
    print(f"  moment instead would have been worth {d['static_gap']:.0f} more -- which is why an")
    print("  exploring policy can go NEGATIVE against a benchmark the whole arc optimized.")
    print()
    print("THE PREVIOUS TOPIC'S INSTRUMENT, where its premise no longer holds:")
    print(f"  {'margin':>8s} {'regret':>9s} {'se':>7s} {'switches':>9s}")
    for r in margin_sweep():
        print(f"  {r['margin']:8.3f} {r['regret']:9.1f} {r['se']:7.1f} {r['switches']:9.1f}")
    m = margins_within_noise()
    print(f"  {m['n_tied']} of {m['n_total']} margins are within one standard error of the best of")
    print("  them: there is no longer an optimum to find. A constant cannot express how well any")
    print(f"  particular arm is known, and it pays for that with {m['spread_ratio']:.1f}x the spread.")
    print()
    print("paired against UCB on the SAME deployments (difficulty cancels):")
    for r in paired_against_ucb():
        verdict = "significant" if r["p"] < 0.05 else "NOT significant"
        print(f"  {r['policy']:>16s}  diff {r['diff']:+8.1f}  t {r['t']:+7.2f}  "
              f"p {r['p']:.2e}   {verdict}")
    print()
    print("A SWITCHING COST IS A TAX ON EXPLORATION:")
    print(f"  {'price':>7s} {'greedy':>9s} {'ucb':>9s} {'thompson':>10s} {'margin':>9s}   winner")
    for r in switching_cost_table():
        print(f"  {r['cost']:7.2f} {r['greedy']:9.1f} {r['ucb']:9.1f} {r['thompson']:10.1f} "
              f"{r['margin']:9.1f}   {r['winner']}")
    c = exploration_crossover(COST_FINE)
    print(f"  {c['winner_at_zero']} wins until a switch costs {c['crossover_cost']}, after which the")
    print("  cheap rule that rarely moves wins -- not by knowing more, but by having stopped paying.")
    print()
    print("WHY THE FAMOUS RATE IS NOT READABLE HERE:")
    r = rate_study()
    print(f"  {'T':>7s} {'UCB regret':>11s}")
    for T, v in zip(r["horizons"], r["regret"]):
        print(f"  {T:7d} {v:11.2f}")
    print(f"  regret against log T is linear with R^2 = {r['log_r2']:.4f} — the instance-dependent")
    print(f"  rate. A power law fitted to the same points gives a slope of {r['loglog_slope']:.3f},")
    print("  which looks like a confirmation of sqrt(T) and is a fit to a logarithm. sqrt(T) and")
    print("  T^(2/3) are MINIMAX rates over instances; one corpus has one gap and cannot show them.")
    print()
    _run_tests()


# =================================================================================================
# Drift guards -- the shipped .tsx and .mdx are parsed back and compared to a fresh bake. Both
# were verified to FAIL on injected drift before being trusted: a guard that cannot fail is worse
# than no guard, because it reads as protection.
