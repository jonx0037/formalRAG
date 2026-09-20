"""When to switch: a drifting workload, a switching cost, and why you should wait.

Run:
    uv run --with numpy --with scipy \
        python notebooks/rag-architecture-switching-hysteresis/rag_architecture_switching_hysteresis.py

Three topics have each removed one assumption. The first measured six architectures and found no
row winning both regimes. The second showed the choice depends on a workload mixture, a price
ratio and an amortization horizon, then fixed all three. The third unfixed the mixture: you
estimate it, and a partition punishes estimation error hardest at its boundaries.

One assumption is still standing. All three treat the workload as STATIC -- unknown, perhaps, but
not moving. It moves. And once it moves, switching architecture costs something real, so the
question stops being WHICH and becomes WHEN.

The answer is a control band: keep what you have unless a rival beats it by more than a
threshold. The interesting part is why the band is there. The obvious story is that it amortizes
the switching cost, and the obvious story is WRONG -- at a switching cost of exactly zero the
regret-optimal band is already wide, because most apparent changes in the best architecture are
the ESTIMATE moving rather than the workload. The band is a filter on an estimator first and a
budget for switching second.

Imports the whole chain (workload-uncertainty -> pareto -> mechanisms) and the drift detectors
from significance-testing-calibration, and reimplements none of it. There is no new corpus: what
is new here is a time axis and a policy over it.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("rag-architecture-workload-uncertainty", "significance-testing-calibration"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rag_architecture_workload_uncertainty as WU                   # noqa: E402
from rag_architecture_workload_uncertainty import (                  # noqa: E402
    ARM_NAMES, CONDITIONS, LAMBDA_HEADLINE, bridge_family, utility_kernel,
)
from significance_testing_calibration import (                       # noqa: E402
    ks_two_sample, population_stability_index,
)

SEED = 20260921

# --- the time axis ----------------------------------------------------------------------------
T_STEPS = 60                    # observation windows in the deployment's life
N_PER_STEP = 40                 # queries observed per window -- the previous topic's headline n
DRIFT_LO, DRIFT_HI = 0.20, 0.70  # the bridge share ramps across the boundary at 0.43

# The band is measured in UTILITY, the same unit the arms are compared in: switch only when a
# rival's advantage exceeds it. Band 0 is the myopic rule -- switch whenever the argmax moves.
BAND_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
COST_GRID = (0.0, 0.001, 0.005, 0.02, 0.1)
N_SEEDS = 10
BAND_HEADLINE = 0.10
COST_HEADLINE = 0.0             # deliberately ZERO at the headline: the band earns its place
                                # before any switching cost is charged at all

DETECT_WINDOW = 20              # observations per detector window (reference vs current)
DETECT_BINS = 4                 # PSI bins. At 8 bins on a 20-observation window PSI reads 0.25-0.69
                                # on a workload with NO drift at all -- the small-n binning artifact
                                # the significance-testing topic already flags. KS never false-alarms
                                # at any setting tried, which is that topic's own conclusion that KS
                                # is the binning-free detector, arriving here as a constraint.
PSI_ALARM = 0.25                # the credit-risk traffic light: >0.25 is "significant" drift
KS_ALPHA = 0.05


def drift_path(kind: str = "ramp", steps: int = T_STEPS) -> np.ndarray:
    """The bridge share over time. The path is a MODELING CHOICE, not an observation, so more
    than one shape is carried: conclusions that hold only on a straight line are conclusions
    about straight lines.

      ramp   -- a monotone crossing of the boundary, the cleanest case
      static -- no drift at all, the control: any positive band must never switch twice
      wobble -- a non-monotone path that crosses, retreats and crosses again
    """
    t = np.linspace(0.0, 1.0, steps)
    if kind == "ramp":
        return DRIFT_LO + (DRIFT_HI - DRIFT_LO) * t
    if kind == "static":
        return np.full(steps, (DRIFT_LO + DRIFT_HI) / 2.0)
    if kind == "wobble":
        return 0.43 + 0.18 * np.sin(2.5 * np.pi * t) + 0.06 * t
    raise ValueError(f"unknown drift path {kind!r}")


def _kernel():
    """The affine utility map from the previous topic: U(a, w) = (M w)_a + const_a. Reused from
    the outset rather than retrofitted -- the predecessor's own two-minute lesson."""
    return utility_kernel(LAMBDA_HEADLINE)


def utilities_at(w: np.ndarray) -> np.ndarray:
    """Every arm's utility on one workload, in one matrix-vector product."""
    M, const = _kernel()
    return M @ np.asarray(w, dtype=float) + const


def true_switch_count(path: np.ndarray) -> int:
    """How many times the BEST arm actually changes along the path -- the irreducible number of
    switches an oracle with no estimation error would make after its first choice."""
    arms = [int(np.argmax(utilities_at(bridge_family(float(f))))) for f in path]
    return sum(1 for a, b in zip(arms, arms[1:]) if a != b)


# =================================================================================================
# The policy: keep what you have unless a rival beats it by more than `band`.
# =================================================================================================

# The streams depend on (path, seed) but NOT on the band or the switching cost, while every
# sweep in this module varies exactly those two. Recomputing 60 multinomial draws per band is the
# textbook loop-invariant recomputation, so they are memoized once per (path, seed) instead.
_STREAMS: dict = {}


def policy_streams(path: np.ndarray, seed: int = SEED, n_per: int = N_PER_STEP) -> dict:
    """The two arrays the policy is a pure function of: the ESTIMATED utility of every arm at
    every window (what the operator sees) and the TRUE utility (what they are scored against).

    Everything in this topic -- switches, regret, the optimal band -- is a deterministic function
    of these, with no sampling left in it. That is what lets the laboratory recompute the entire
    band sweep in the browser rather than displaying a baked curve, and what lets
    `test_viz_constants_reproduce_the_policy` check the browser's arithmetic against `simulate`.
    """
    key = (path.tobytes(), seed, n_per)
    if key in _STREAMS:
        return _STREAMS[key]
    rng = np.random.default_rng(seed)
    u_hat, u_true = [], []
    for f in path:
        w_true = np.asarray(bridge_family(float(f)), dtype=float)
        counts = rng.multinomial(n_per, w_true)
        w_hat = counts / counts.sum() if counts.sum() else w_true.copy()
        u_hat.append(utilities_at(w_hat))
        u_true.append(utilities_at(w_true))
    _STREAMS[key] = {"u_hat": np.array(u_hat), "u_true": np.array(u_true)}
    return _STREAMS[key]


def run_policy(u_hat, u_true, band: float, cost: float = COST_HEADLINE) -> dict:
    """The band rule over pre-computed streams -- the exact arithmetic the browser mirrors."""
    cur, switches, regret, held = None, 0, 0.0, []
    for uh, ut in zip(u_hat, u_true):
        best = int(np.argmax(uh))
        if cur is None or uh[best] - uh[cur] > band:
            cur, switches = best, switches + 1
        regret += float(np.max(ut) - ut[cur])
        held.append(cur)
    n = len(u_hat)
    return {"switches": switches, "regret": regret / n,
            "total": regret / n + cost * switches / n, "held": held}

def simulate(path: np.ndarray, band: float = BAND_HEADLINE, cost: float = COST_HEADLINE,
             seed: int = SEED, n_per: int = N_PER_STEP) -> dict:
    """One deployment's life. At each window the operator observes n_per queries, forms the
    empirical mixture, and decides whether to move.

    band = 0 is the MYOPIC rule -- switch whenever the estimated best arm changes -- and is
    exactly the previous topic's plug-in rule applied afresh at every step. Any band > 0 adds
    hysteresis: the incumbent keeps its place unless beaten by a clear margin.

    Regret is scored against the TRUE workload at each step, never the estimate, so a policy is
    charged for its mistakes rather than for its beliefs. GUARDS: band >= 0, cost >= 0, n_per >= 1.
    """
    if band < 0:
        raise ValueError(f"a hysteresis band cannot be negative, got {band}")
    if cost < 0:
        raise ValueError(f"a switching cost cannot be negative, got {cost}")
    if n_per < 1:
        raise ValueError(f"a window must observe at least one query, got {n_per}")
    st = policy_streams(path, seed, n_per)
    return run_policy(st["u_hat"], st["u_true"], band, cost)



def over_seeds(path: np.ndarray, band: float, cost: float = COST_HEADLINE,
               n_seeds: int = N_SEEDS, **kw) -> dict:
    """The same policy over several deployments. A single run is one draw of the estimator's
    noise, and the whole topic is about that noise, so nothing is reported from one."""
    runs = [simulate(path, band, cost, SEED + s, **kw) for s in range(n_seeds)]
    return {"switches": float(np.mean([r["switches"] for r in runs])),
            "regret": float(np.mean([r["regret"] for r in runs])),
            "total": float(np.mean([r["total"] for r in runs]))}


def band_sweep(path: np.ndarray | None = None, bands=BAND_GRID, cost: float = COST_HEADLINE,
               n_seeds: int = N_SEEDS) -> list[dict]:
    """Switches and regret across the band, at a fixed switching cost."""
    path = drift_path("ramp") if path is None else path
    return [{"band": float(b), **over_seeds(path, float(b), cost, n_seeds)} for b in bands]


def optimal_band(path: np.ndarray | None = None, cost: float = COST_HEADLINE,
                 bands=BAND_GRID, n_seeds: int = N_SEEDS) -> float:
    """The band minimizing total cost (regret plus charged switches)."""
    rows = band_sweep(path, bands, cost, n_seeds)
    return float(min(rows, key=lambda r: r["total"])["band"])


def cost_sweep(costs=COST_GRID, path: np.ndarray | None = None) -> list[dict]:
    """How the optimal band responds to the price of switching. The headline lives in the FIRST
    row: at cost exactly zero the optimum is already a wide band, so hysteresis is not paying for
    switches -- it is filtering an estimator."""
    path = drift_path("ramp") if path is None else path
    out = []
    for c in costs:
        b = optimal_band(path, float(c))
        row = over_seeds(path, b, float(c))
        out.append({"cost": float(c), "band": b, "switches": row["switches"],
                    "regret": row["regret"], "total": row["total"]})
    return out


def interior_optimum_per_seed(path: np.ndarray | None = None, bands=BAND_GRID,
                              cost: float = COST_HEADLINE, n_seeds: int = N_SEEDS) -> list[dict]:
    """Per seed, the best band and whether BOTH ends lost. The optimum's LOCATION moves with the
    seed, so only its existence is ever claimed -- the fragile-interior-optimum precedent from
    `retrieval-vs-long-context`, checked here before anything was written rather than after."""
    path = drift_path("ramp") if path is None else path
    out = []
    for s in range(n_seeds):
        regs = [simulate(path, float(b), cost, SEED + s)["regret"] for b in bands]
        i = int(np.argmin(regs))
        out.append({"seed": s, "best_band": float(bands[i]),
                    "interior": bool(0 < i < len(bands) - 1),
                    "beats_myopic": bool(regs[i] < regs[0]),
                    "beats_widest": bool(regs[i] < regs[-1])})
    return out


STATIC_SHARES = (0.20, 0.35, 0.43, 0.45, 0.55, 0.70)


def static_chatter(shares=STATIC_SHARES, bands=(0.0, 0.05, 0.10, 0.20),
                   n_seeds: int = N_SEEDS) -> list[dict]:
    """Switching on workloads that DO NOT MOVE -- the control that isolates the estimator.

    With the drift set to exactly zero, every switch a policy makes is attributable to noise and
    nothing else. Deep inside a cell even the myopic rule settles after its first choice; at the
    boundary it switches on half the windows in the deployment's life, responding to a workload
    that never changed. This is the cleanest statement of the topic's headline, and it is the
    previous topic's distance-to-boundary result arriving on the time axis.
    """
    out = []
    for f in shares:
        path = np.full(T_STEPS, float(f))
        row = {"share": float(f), "true_changes": true_switch_count(path)}
        for b in bands:
            row[f"band_{b}"] = over_seeds(path, float(b), n_seeds=n_seeds)["switches"]
        out.append(row)
    return out


# =================================================================================================
# Movement 5 -- detecting is not deciding.
# =================================================================================================

def detector_reading(path: np.ndarray, seed: int = SEED, n_per: int = N_PER_STEP,
                     window: int = DETECT_WINDOW, bins: int = DETECT_BINS) -> dict:
    """The imported drift detectors on the same observation stream the policy sees: the first
    `window` bridge-share observations as the reference, the last `window` as the current.

    ONE comparison, not a maximum over every overlapping pair. A max over ~40 overlapping windows
    is an extreme statistic that grows with how many comparisons were made -- the same trap the
    previous topic hit with worst-case regret, and it inflated PSI here to 1.16 on a workload with
    no drift whatsoever before it was caught.

    GUARD: a window needs at least two observations.
    """
    if window < 2:
        raise ValueError(f"a detection window needs at least two observations, got {window}")
    if len(path) < 2 * window:
        raise ValueError(f"path of {len(path)} steps is too short for two {window}-step windows")
    rng = np.random.default_rng(seed)
    bi = CONDITIONS.index("bridge")
    obs = []
    for f in path:
        w_true = np.asarray(bridge_family(float(f)), dtype=float)
        counts = rng.multinomial(n_per, w_true)
        obs.append(float(counts[bi] / counts.sum()) if counts.sum() else float(w_true[bi]))
    ref, cur = np.asarray(obs[:window]), np.asarray(obs[-window:])
    ks = ks_two_sample(ref, cur)
    psi = population_stability_index(ref, cur, n_bins=bins)
    return {"ks": float(ks["stat"]), "ks_p": float(ks["pvalue"]), "psi": float(psi),
            "ks_alarm": bool(ks["pvalue"] < KS_ALPHA), "psi_alarm": bool(psi > PSI_ALARM)}


def wobble_windows() -> dict:
    """What the detector actually compares on the non-monotone path, and what it cannot see.

    The reference window (the first `DETECT_WINDOW` observations) and the current window (the
    last) overlap heavily, while the path BETWEEN them dips far below either. That is the whole
    reason KS reports nothing on a path whose best architecture changed twice: the excursion is
    invisible to a comparison of two endpoints, by construction rather than by bad luck.
    """
    w = drift_path("wobble")
    r, c, mid = w[:DETECT_WINDOW], w[-DETECT_WINDOW:], w[DETECT_WINDOW:-DETECT_WINDOW]
    f = lambda a: [round(float(a.min()), 2), round(float(a.max()), 2)]     # noqa: E731
    return {"reference": f(r), "current": f(c), "middle_low": round(float(mid.min()), 2)}


def detection_vs_decision(kinds=("static", "ramp", "wobble"), band: float = BAND_HEADLINE,
                          seed: int = SEED) -> list[dict]:
    """Detectors and policy on the same streams. The two disagree in OPPOSITE directions, which is
    the movement's whole point:

      static -- the detectors are correctly silent and a myopic policy switches dozens of times
                anyway. The policy acts when nothing happened.
      wobble -- the excursion happens in the MIDDLE. The reference and current windows overlap
                heavily while the path between them travels far below both, so a
                reference-versus-current comparison sees almost nothing and the policy
                genuinely had to move twice. The detector misses what did happen, because
                everything that happened, happened where it does not look.

    A detector compares two points in time. A policy lives through everything between them.
    """
    out = []
    for kind in kinds:
        path = drift_path(kind)
        det = detector_reading(path, seed)
        out.append({"kind": kind, "true_changes": true_switch_count(path),
                    "myopic_switches": simulate(path, 0.0, seed=seed)["switches"],
                    "banded_switches": simulate(path, band, seed=seed)["switches"],
                    **det})
    return out


# =================================================================================================
# viz_constants -- this function OWNS every number the laboratory displays.
# =================================================================================================

def viz_constants() -> dict:
    r4 = lambda v: round(float(v), 4)                                          # noqa: E731
    ramp = drift_path("ramp")
    return {
        "arms": list(ARM_NAMES), "conditions": list(CONDITIONS),
        "t_steps": T_STEPS, "n_per_step": N_PER_STEP, "n_seeds": N_SEEDS,
        "band_grid": [float(b) for b in BAND_GRID], "cost_grid": [float(c) for c in COST_GRID],
        "band_headline": BAND_HEADLINE, "cost_headline": COST_HEADLINE,
        "drift_lo": DRIFT_LO, "drift_hi": DRIFT_HI,
        "detect_window": DETECT_WINDOW, "detect_bins": DETECT_BINS,
        "psi_alarm": PSI_ALARM, "ks_alpha": KS_ALPHA,
        "paths": {k: [r4(f) for f in drift_path(k)] for k in ("ramp", "static", "wobble")},
        "true_changes": {k: int(true_switch_count(drift_path(k)))
                         for k in ("ramp", "static", "wobble")},
        # regret runs to ~0.004, so four decimals would leave it with two significant figures and
        # the topic's own table quotes five -- these are the numbers the prose guard compares to
        "band_sweep": [{"band": r["band"], "switches": r4(r["switches"]),
                        "regret": round(float(r["regret"]), 5)} for r in band_sweep()],
        "per_seed": interior_optimum_per_seed(),
        "cost_sweep": [{k: (round(float(v), 5) if isinstance(v, float) else v)
                        for k, v in row.items()} for row in cost_sweep()],
        "static_chatter": [{k: (r4(v) if isinstance(v, float) else v) for k, v in row.items()}
                           for row in static_chatter()],
        "detection": [{k: (r4(v) if isinstance(v, float) else v) for k, v in row.items()}
                      for row in detection_vs_decision()],
        "wobble_windows": wobble_windows(),
        # the two streams the whole policy is a pure function of: the browser recomputes every
        # switch count, regret and optimum from these, so no curve in the lab is a baked picture
        # 5 decimals is the measured FLOOR, not a default: at 4 decimals a near-tie between two
        # arms rounds the wrong way on one of eighty (seed, band) pairs and the browser silently
        # diverges from the module. `test_viz_constants_reproduce_the_policy` pins it.
        "u_true": [[round(float(v), 5) for v in row]
                   for row in policy_streams(ramp)["u_true"]],
        "u_hat_seeds": [[[round(float(v), 5) for v in row]
                         for row in policy_streams(ramp, SEED + s)["u_hat"]]
                        for s in range(N_SEEDS)],
    }


# =================================================================================================
# Collapse anchors -- the time axis must vanish when nothing is moving.
# =================================================================================================

def test_band_zero_is_the_previous_topics_rule() -> None:
    """A zero band is the predecessor's plug-in rule applied afresh at every window: no memory, no
    hysteresis, just the argmax of the current estimate. Asserted as the identical SEQUENCE of
    arms, not merely the same count."""
    path = drift_path("ramp")
    held = simulate(path, 0.0, seed=SEED)["held"]
    rng = np.random.default_rng(SEED)
    expect = []
    for f in path:
        w_true = np.asarray(bridge_family(float(f)), dtype=float)
        counts = rng.multinomial(N_PER_STEP, w_true)
        expect.append(ARM_NAMES.index(WU.rule_plug_in(counts / counts.sum())))
    assert held == expect, (held[:8], expect[:8])


def test_an_infinite_band_never_switches() -> None:
    """With the band above every possible advantage the incumbent is never displaced, so the
    policy is 'commit to the first estimate and live with it' -- one switch, and a regret equal to
    holding that arm the whole way."""
    path = drift_path("ramp")
    r = simulate(path, band=1e9, seed=SEED)
    assert r["switches"] == 1
    assert len(set(r["held"])) == 1
    held = r["held"][0]
    manual = float(np.mean([utilities_at(bridge_family(float(f))).max()
                            - utilities_at(bridge_family(float(f)))[held] for f in path]))
    assert abs(r["regret"] - manual) < 1e-12


def test_a_static_workload_deep_in_a_cell_settles() -> None:
    """The control's control: no drift AND far from a boundary means the estimate is never
    ambiguous, so every policy including the myopic one makes its first choice and stops."""
    for share in (0.70,):
        path = np.full(T_STEPS, share)
        for band in (0.0, BAND_HEADLINE):
            r = over_seeds(path, band)
            assert r["switches"] == 1.0, (share, band, r["switches"])
            assert r["regret"] < 1e-9


def test_guards() -> None:
    """Malformed inputs are rejected rather than silently producing a plausible number."""
    path = drift_path("ramp")
    for kwargs in ({"band": -0.1}, {"cost": -1.0}, {"n_per": 0}):
        try:
            simulate(path, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"simulate accepted {kwargs}")
    for w in (1, len(path)):
        try:
            detector_reading(path, window=w if w == 1 else len(path))
        except ValueError:
            continue
        raise AssertionError(f"detector_reading accepted window={w}")
    try:
        drift_path("nonsense")
    except ValueError:
        return
    raise AssertionError("drift_path accepted an unknown path")


# =================================================================================================
# Claim tests -- every headline in the prose, and every limit on it.
# =================================================================================================

def test_myopic_switching_chatters() -> None:
    """Movement 2. Along a path whose true argmax changes ONCE, a rule that re-decides from each
    window's estimate switches an order of magnitude more often. Asserted as a ratio, not a count,
    so the claim survives a reseed."""
    path = drift_path("ramp")
    truth = true_switch_count(path)
    myopic = over_seeds(path, 0.0)["switches"]
    assert truth == 1, truth
    assert myopic >= 5 * max(truth, 1), (myopic, truth)


def test_the_band_improves_regret_not_only_stability() -> None:
    """Movement 3, and the reason the band is not a trade. Fewer switches would be worth something
    even if decision quality suffered; here it does not suffer, it IMPROVES. Suppressing switches
    that the noise proposed removes decisions that were wrong."""
    path = drift_path("ramp")
    myopic, banded = over_seeds(path, 0.0), over_seeds(path, BAND_HEADLINE)
    assert banded["switches"] < myopic["switches"] / 2, (banded, myopic)
    assert banded["regret"] < myopic["regret"], (banded["regret"], myopic["regret"])


def test_an_interior_optimum_exists_on_every_seed() -> None:
    """Movement 3. Both ends of the grid lose on EVERY seed -- but the optimum's location moves,
    so its location is never asserted. The `retrieval-vs-long-context` interior optimum was
    seed-fragile and had to be demoted to a remark; this one is checked seed by seed instead of
    being taken from the average, which is the only way to tell the two cases apart."""
    rows = interior_optimum_per_seed()
    assert len(rows) == N_SEEDS
    for r in rows:
        assert r["interior"], r
        assert r["beats_myopic"] and r["beats_widest"], r
    assert len({r["best_band"] for r in rows}) > 1, "a single best band would make this a constant"


def test_the_band_earns_its_place_at_zero_switching_cost() -> None:
    """Movement 4 -- the headline, and the half that is not obvious. If hysteresis were about
    amortizing an expensive switch, then making switches FREE should collapse the band to zero.
    It does not: at cost exactly zero the optimal band is already wide. The band's real job is
    filtering an estimator, and the switching cost is a second, smaller reason to keep it."""
    rows = cost_sweep()
    free = next(r for r in rows if r["cost"] == 0.0)
    assert free["band"] > 0.0, free
    assert free["band"] >= BAND_GRID[len(BAND_GRID) // 2], free


def test_the_optimal_band_is_nondecreasing_in_cost() -> None:
    """Movement 4's second half: the switching cost does move the band, in the direction the
    classical control-band result says it should -- just from an already-positive starting point."""
    bands = [r["band"] for r in cost_sweep()]
    assert all(b <= c for b, c in zip(bands, bands[1:])), bands
    assert bands[-1] > bands[0], bands


def test_chatter_is_worst_at_a_boundary_and_vanishes_away_from_one() -> None:
    """The control, and the previous topic's result arriving on the time axis. Drift is exactly
    zero on every row, so every switch is the estimator talking. Deep in a cell the myopic rule
    settles; near the boundary it switches on a large fraction of the windows in the deployment's
    life. The band suppresses the boundary case without touching the settled one."""
    rows = {r["share"]: r for r in static_chatter()}
    for r in rows.values():
        assert r["true_changes"] == 0, r
    deep, edge = rows[0.70], rows[0.43]
    assert deep["band_0.0"] == 1.0, deep
    assert edge["band_0.0"] > 10 * deep["band_0.0"], (edge, deep)
    assert edge["band_0.2"] < edge["band_0.0"] / 5, edge


def test_detection_is_not_decision() -> None:
    """Movement 5, and the distinctness argument against `significance-testing-calibration`. The
    two instruments disagree in OPPOSITE directions on the same streams, which is why neither
    substitutes for the other:

      static -- detectors correctly silent, myopic policy switching dozens of times anyway
      wobble -- the policy genuinely moved twice while a reference-versus-current KS test sees
                nothing, because the path ended near where it started
    """
    rows = {r["kind"]: r for r in detection_vs_decision()}
    static, ramp, wobble = rows["static"], rows["ramp"], rows["wobble"]

    assert not static["ks_alarm"] and not static["psi_alarm"], static
    assert static["true_changes"] == 0 and static["myopic_switches"] > 10, static

    assert ramp["ks_alarm"] and ramp["psi_alarm"], ramp

    assert wobble["true_changes"] >= 2, wobble
    assert not wobble["ks_alarm"], wobble
    assert wobble["ks_p"] > 0.5, wobble["ks_p"]


def test_the_detector_cannot_see_the_middle() -> None:
    """The reason KS reports nothing on the wobbling path, asserted rather than asserted-about.
    The two windows it compares overlap; the path between them leaves both ranges entirely. A
    detector that compares two moments is blind to an excursion that happens between them, and
    that is a structural fact about the comparison, not a small-sample accident."""
    w = wobble_windows()
    r, c = w["reference"], w["current"]
    assert min(r[1], c[1]) - max(r[0], c[0]) > 0.15, (r, c)      # the windows genuinely overlap
    assert w["middle_low"] < r[0] and w["middle_low"] < c[0], w  # the dip leaves both of them
    row = next(d for d in detection_vs_decision() if d["kind"] == "wobble")
    assert row["true_changes"] >= 2 and not row["ks_alarm"], row


def test_a_nonmonotone_path_reaches_the_same_verdict() -> None:
    """The designed-input check. A linear ramp through a boundary is a MODELING CHOICE, so the
    conclusions are re-run on a path that crosses, retreats and crosses back. The band still beats
    the myopic rule on both counts, so the verdict is not an artifact of a straight line."""
    path = drift_path("wobble")
    myopic, banded = over_seeds(path, 0.0), over_seeds(path, BAND_HEADLINE)
    assert banded["switches"] < myopic["switches"], (banded, myopic)
    assert banded["regret"] < myopic["regret"], (banded, myopic)


def test_the_kernel_is_the_previous_topics_utility() -> None:
    """The affine map the whole time axis runs on is the predecessor's, not a reimplementation:
    every arm's utility at every workload agrees with the imported `utility` to floating point."""
    M, const = _kernel()
    for f in (0.0, 0.31, 0.5, 1.0):
        w = np.asarray(bridge_family(float(f)), dtype=float)
        got = M @ w + const
        want = WU.utility(w, LAMBDA_HEADLINE)
        assert np.max(np.abs(got - want)) < 1e-12, (f, np.max(np.abs(got - want)))


def test_viz_constants_reproduce_the_policy() -> None:
    """The laboratory recomputes the band sweep live from `u_hat_seeds`/`u_true` rather than
    drawing a baked curve. `simulate` delegates to `run_policy`, so what is genuinely under
    test here is the ROUNDING: the lab receives six-decimal utilities, and a near-tie between two
    arms could round the wrong way, flip an argmax and desynchronize the browser from the module.
    The assertion is that it does not -- on every seed, at four bands, to the exact arm held."""
    v = viz_constants()
    ut = np.array(v["u_true"])
    for s in range(N_SEEDS):
        uh = np.array(v["u_hat_seeds"][s])
        for band in BAND_GRID:
            got = run_policy(uh, ut, band)
            want = simulate(drift_path("ramp"), band, seed=SEED + s)
            assert got["held"] == want["held"], (s, band)
            assert got["switches"] == want["switches"], (s, band)
            assert abs(got["regret"] - want["regret"]) < 1e-6, (s, band)


# =================================================================================================
# Drift guards -- the shipped .tsx and .mdx are parsed back and compared to a fresh bake.
# Both were verified to FAIL on injected drift before being trusted: a guard that cannot fail is
# worse than no guard, because it reads as protection.
# =================================================================================================

_LAB = (pathlib.Path(__file__).resolve().parents[2]
        / "src" / "components" / "viz" / "SwitchingHysteresisLaboratory.tsx")


def test_laboratory_constants_match_the_module() -> None:
    """Every number the laboratory displays comes from `viz_constants()`. The .tsx block is EMITTED
    from it rather than transcribed, so this guard checks that the emitted file on disk is still
    the one the module would produce today -- both the JSON blocks and the bare scalars, which are
    declared several to a line and are quoted directly in the panel notes a reader actually reads.
    """
    src = _LAB.read_text()
    v = viz_constants()

    for name, want in (("PATHS", v["paths"]), ("U_TRUE", v["u_true"]),
                       ("U_HAT_SEEDS", v["u_hat_seeds"]),
                       ("STATIC_CHATTER", v["static_chatter"]),
                       ("DETECTION", v["detection"]), ("ARMS", v["arms"])):
        m = re.search(rf"^const {name}(?:: [^=]+)? = (.*?)(?: as const)?;$", src, re.M)
        assert m, f"{name} is not in the laboratory"
        assert json.loads(m.group(1)) == json.loads(json.dumps(want)), f"{name} has drifted"

    for name, want in (("T_STEPS", T_STEPS), ("N_PER_STEP", N_PER_STEP), ("N_SEEDS", N_SEEDS),
                       ("DRIFT_LO", DRIFT_LO), ("DRIFT_HI", DRIFT_HI),
                       ("BAND_HEADLINE", BAND_HEADLINE), ("COST_HEADLINE", COST_HEADLINE),
                       ("DETECT_WINDOW", DETECT_WINDOW), ("DETECT_BINS", DETECT_BINS),
                       ("PSI_ALARM", PSI_ALARM), ("KS_ALPHA", KS_ALPHA)):
        m = re.search(rf"\b{name} = ([0-9.eE+-]+)", src)
        assert m, f"the scalar {name} is not in the laboratory"
        assert abs(float(m.group(1)) - float(want)) < 1e-12, f"{name} has drifted"

    # Panel B's note tells the reader how many band widths the MODULE's grid holds, to explain why
    # the panel's own finer sweep lands between them. That number is hand-written in the .tsx, so
    # it needs a guard of its own or it silently rots the next time the grid changes.
    m = re.search(r"BAND_GRID_N = ([0-9]+)", src)
    assert m, "BAND_GRID_N is not in the laboratory"
    assert int(m.group(1)) == len(BAND_GRID), "BAND_GRID_N has drifted from the module's grid"

    # The static-workload table's column keys are built from FLOATS here (`band_{b}`) and read by
    # hardcoded STRING in the .tsx, so a change to the band tuple would leave the laboratory
    # rendering blanks while the JSON comparison above still passed on the data. Pin the join.
    m = re.search(r"const bands = \[([^\]]+)\]", src)
    assert m, "the static-chatter column list is not in the laboratory"
    shown = [x.strip().strip("'\"") for x in m.group(1).split(",")]
    have = {k[len("band_"):] for k in static_chatter()[0] if k.startswith("band_")}
    assert set(shown) <= have, f"the laboratory reads columns the module does not emit: {set(shown) - have}"


_MDX = (pathlib.Path(__file__).resolve().parents[2]
        / "src" / "content" / "topics" / "rag-architecture-switching-hysteresis.mdx")


def _table_rows(src: str, header: str) -> list[list[str]]:
    """Every data row of the markdown table whose header line contains `header`, with bold
    markers and cell padding stripped. MDX wraps prose lines but never table rows."""
    lines = src.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("|") and header in ln)
    out = []
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        out.append([c.strip().replace("**", "").replace("$", "") for c in ln.strip("|").split("|")])
    return out


def test_topic_prose_matches_the_module() -> None:
    """Every number a READER sees is parsed back out of the shipped .mdx and compared to a fresh
    bake -- the tables AND the inline figures, since a retune moves both and only the tables look
    like data. The predecessor topics each shipped a stale figure that survived because it was
    spelled out in a sentence rather than sitting in a table."""
    src = _MDX.read_text()
    v = viz_constants()

    rows = _table_rows(src, "real changes | myopic switches")
    assert len(rows) == len(v["static_chatter"]), (len(rows), len(v["static_chatter"]))
    for row, want in zip(rows, v["static_chatter"]):
        assert float(row[0]) == round(want["share"], 2), row
        assert int(row[1]) == want["true_changes"], row
        assert float(row[2]) == round(want["band_0.0"], 1), (row, want["band_0.0"])

    quoted = {r["band"]: r for r in v["band_sweep"]}
    for row in _table_rows(src, "| band $"):
        w = quoted[float(row[0])]
        assert float(row[1]) == round(w["switches"], 1), row
        assert float(row[2]) == round(w["regret"], 5), (row, w["regret"])

    costs = {r["cost"]: r for r in v["cost_sweep"]}
    for row in _table_rows(src, "price of a switch | optimal band"):
        w = costs[float(row[0])]
        assert float(row[1]) == round(w["band"], 2), row
        assert float(row[2]) == round(w["switches"], 1), row
        assert float(row[3]) == round(w["regret"], 5), row

    det = {r["kind"]: r for r in v["detection"]}
    for row in _table_rows(src, "real changes | myopic | banded"):
        w = det[row[0]]
        assert int(row[1]) == w["true_changes"], row
        assert int(row[2]) == w["myopic_switches"], row
        assert int(row[3]) == w["banded_switches"], row
        assert abs(float(row[4]) - w["ks_p"]) < 5e-5, row
        assert abs(float(row[5]) - w["psi"]) < 5e-4, row

    # the inline figures -- the class that hides from a table-shaped guard. Tolerant of the line
    # wrapping MDX applies to prose but not to table rows.
    myopic0 = next(r for r in v["band_sweep"] if r["band"] == 0.0)
    best = min(v["band_sweep"], key=lambda r: r["regret"])
    free = next(r for r in v["cost_sweep"] if r["cost"] == 0.0)
    for label, pattern, want, tol in (
        ("myopic switch count", r"switches\s+\*\*([0-9.]+)\*\*\s*times", myopic0["switches"], 5e-2),
        ("regret at band 0", r"\$([0-9.]+)\$\s*\n?\s*at \$\\delta = 0\$", myopic0["regret"], 5e-6),
        ("regret at the optimum", r"to \$([0-9.]+)\$ near \$\\delta", best["regret"], 5e-6),
        ("the best band in prose", r"to \$[0-9.]+\$ near \$\\delta = ([0-9.]+)\$", best["band"], 5e-3),
        ("the free-switching band", r"is already \$([0-9.]+)\$, the same band", free["band"], 5e-3),
        ("windows per deployment", r"lives through \$T = ([0-9]+)\$", T_STEPS, 0.5),
        ("queries per window", r"it observes \$([0-9]+)\$ queries", N_PER_STEP, 0.5),
        ("the static myopic peak", r"changes architecture \$?([0-9]+)\$? times regardless",
         max(r["myopic_switches"] for r in v["detection"]), 0.5),
    ):
        m = re.search(pattern, src)
        assert m, f"the prose figure for {label} is missing or reworded"
        assert abs(float(m.group(1)) - float(want)) < tol, f"{label}: prose {m.group(1)} vs {want}"

    # the wobble windows, quoted in prose to explain why the detector is blind there
    w = v["wobble_windows"]
    m = re.search(r"reference window spans\s*\n?\s*\$\[([0-9.]+), ([0-9.]+)\]\$ and the current "
                  r"window \$\[([0-9.]+), ([0-9.]+)\]\$", src)
    assert m, "the wobble window spans are missing or reworded"
    assert [float(m.group(1)), float(m.group(2))] == w["reference"], (m.groups(), w)
    assert [float(m.group(3)), float(m.group(4))] == w["current"], (m.groups(), w)
    m = re.search(r"the share falls to\s*\n?\s*\$([0-9.]+)\$", src)
    assert m and float(m.group(1)) == w["middle_low"], "the wobble dip has drifted"

    # the drift range, and the seed-varying optima the topic lists rather than averages
    m = re.search(r"drifts linearly from \$([0-9.]+)\$ to \$([0-9.]+)\$", src)
    assert m and (float(m.group(1)), float(m.group(2))) == (DRIFT_LO, DRIFT_HI), "drift range"
    listed = {float(x) for x in re.findall(r"\$([0-9.]+)\$(?:,| and) ", src[src.index("argmin lands on"):][:120])}
    listed |= {float(re.search(r"and \$([0-9.]+)\$", src[src.index("argmin lands on"):][:160]).group(1))}
    assert listed == {r["best_band"] for r in v["per_seed"]}, (listed, {r["best_band"] for r in v["per_seed"]})


def test_the_per_seed_claims_the_laboratory_makes() -> None:
    """Panel A shows ONE deployment and says three things about the whole set of them. Each is
    checked here per seed, because a panel that opens on a single run is the easiest place in the
    arc for prose to drift past its evidence -- and the default run is, as it happens, the one
    deployment of ten where the headline band does NOT beat switching freely on regret. That is
    stated in the panel rather than avoided by opening on a flattering seed."""
    path = drift_path("ramp")
    wide = BAND_GRID.index(BAND_HEADLINE)
    beats_on_regret = 0
    for s in range(N_SEEDS):
        sw = [simulate(path, float(b), seed=SEED + s)["switches"] for b in BAND_GRID]
        rg = [simulate(path, float(b), seed=SEED + s)["regret"] for b in BAND_GRID]
        assert all(a >= b for a, b in zip(sw, sw[1:])), (s, sw)        # non-increasing in the band
        assert sw[wide] < sw[0], (s, sw)                               # the headline band switches less
        assert min(rg[1:-1]) < rg[0], (s, rg)                          # SOME interior band wins
        beats_on_regret += rg[wide] < rg[0]
    assert beats_on_regret == N_SEEDS - 1, beats_on_regret             # ... but not the same one


def _run_tests() -> None:
    names = sorted(n for n, v in globals().items() if n.startswith("test_") and callable(v))
    for n in names:
        globals()[n]()
        print(f"  ok  {n}")
    print(f"{len(names)} assertions passed")


# NOTE: this guard stays at the very END of the file. `_run_tests` scans `globals()` at call
# time, so a `def test_*` written BELOW it has not executed yet and is silently skipped -- the
# count just comes out one short. If an added test does not raise the printed number, that is
# the bug.
if __name__ == "__main__":
    ramp = drift_path("ramp")
    print("A WORKLOAD THAT MOVES: the bridge share drifts "
          f"{DRIFT_LO:.2f} -> {DRIFT_HI:.2f} over {T_STEPS} windows.")
    print(f"  the best arm changes {true_switch_count(ramp)}x over the whole path")
    print(f"  a rule that re-decides every window switches "
          f"{over_seeds(ramp, 0.0)['switches']:.1f}x")
    print()
    print("THE CONTROL BAND -- switch only when a rival beats the incumbent by more than `band`:")
    print(f"  {'band':>6s} {'switches':>9s} {'regret':>9s}")
    for r in band_sweep():
        mark = "   <- best" if r["band"] == optimal_band() else ""
        print(f"  {r['band']:6.3f} {r['switches']:9.1f} {r['regret']:9.5f}{mark}")
    print("  fewer switches AND lower regret: the band is not trading quality for stability,")
    print("  it is removing decisions that were the estimator talking.")
    print()
    print("the interior optimum, seed by seed (its LOCATION moves; only its existence is claimed):")
    for r in interior_optimum_per_seed():
        print(f"  seed {r['seed']:2d}  best band {r['best_band']:5.3f}"
              f"  interior {str(r['interior']):>5s}"
              f"  beats myopic {str(r['beats_myopic']):>5s}"
              f"  beats widest {str(r['beats_widest']):>5s}")
    print()
    print("HYSTERESIS IS ABOUT NOISE, NOT COST -- the headline. If the band existed to amortize")
    print("an expensive switch, making switches FREE would collapse it. It does not:")
    print(f"  {'switch cost':>11s} {'optimal band':>12s} {'switches':>9s} {'regret':>9s}")
    for r in cost_sweep():
        mark = "   <- switching is FREE here" if r["cost"] == 0.0 else ""
        print(f"  {r['cost']:11.3f} {r['band']:12.3f} {r['switches']:9.1f} "
              f"{r['regret']:9.5f}{mark}")
    print()
    print("the control: workloads that DO NOT MOVE. Every switch below is noise, nothing else.")
    print(f"  {'share':>6s} {'true':>5s} {'band 0':>8s} {'0.05':>8s} {'0.10':>8s} {'0.20':>8s}")
    for r in static_chatter():
        print(f"  {r['share']:6.2f} {r['true_changes']:5d} {r['band_0.0']:8.1f} "
              f"{r['band_0.05']:8.1f} {r['band_0.1']:8.1f} {r['band_0.2']:8.1f}")
    print("  the boundary between two arms sits near 0.43. Distance to it -- the previous")
    print("  topic's result -- is what decides whether a deployment chatters.")
    print()
    print("DETECTING IS NOT DECIDING. Same streams, two instruments, opposite failures:")
    print(f"  {'path':>7s} {'true':>5s} {'myopic':>7s} {'banded':>7s} {'KS p':>8s} {'KS!':>4s} "
          f"{'PSI':>7s} {'PSI!':>5s}")
    for r in detection_vs_decision():
        print(f"  {r['kind']:>7s} {r['true_changes']:5d} {r['myopic_switches']:7d} "
              f"{r['banded_switches']:7d} {r['ks_p']:8.4f} "
              f"{'YES' if r['ks_alarm'] else 'no':>4s} {r['psi']:7.3f} "
              f"{'YES' if r['psi_alarm'] else 'no':>5s}")
    print("  static: the detectors are right and the policy acts anyway.")
    print("  wobble: the policy is right and the detector sees nothing -- the path ended near")
    print("  where it began. A detector compares two moments; a policy lives through all of them.")
    print()
    _run_tests()
