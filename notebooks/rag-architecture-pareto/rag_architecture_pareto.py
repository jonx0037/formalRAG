"""RAG Architecture Pareto: which architecture to deploy, and what that question depends on.

Run:
    uv run --with numpy --with scipy \
        python notebooks/rag-architecture-pareto/rag_architecture_pareto.py

The predecessor measured six named architectures over two regimes and found no row that wins
both. It reported cost in TWO currencies -- similarity operations and generator calls -- and
refused to collapse them, because the exchange rate between them is exactly what a cost model
should not invent. That refusal left a Pareto set rather than a winner. This is what you do
with it.

The question here is "which architecture should we deploy", and the answer is that the question
is underdetermined. It depends on three things, none of which is a property of any architecture:

  the WORKLOAD MIXTURE   -- what fraction of traffic is each kind of question
  the PRICE RATIO rho    -- what a generator call costs in units of a similarity operation
  the AMORTIZATION N     -- how many queries an offline index build is spread over

Move any one of the three and the winner changes. The topic is the shape of that dependence:
a partition of the mixture simplex, a convex hull the price ratio slides along, and a crossover
in N. Its sharpest result is that one arm -- the same arm each time -- is invisible to both of
the standard ways of picking a winner.

This is NOT per-query routing. `adaptive-retrieval-routing` chooses a strategy per query under a
single scalar cost; here the choice is one fixed architecture for a whole deployment, and cost
is a PARTIAL order rather than a number. rho = 0 recovers that topic's cost model exactly, and
a test asserts it.

Imports the six arms, both regimes and the cost units from rag-architecture-mechanisms and
reimplements none of them. There is no new corpus here: everything new is the decision layer.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("rag-architecture-mechanisms",):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rag_architecture_mechanisms as MECH                         # noqa: E402
from rag_architecture_mechanisms import (                          # noqa: E402
    ARMS, CLASSES, C_POOLED, N_SECTORS, R_PASSAGES, glob, local,
)

# The six conditions a deployment's traffic is a mixture over: the five local query classes and
# the global regime. A workload is a point on this 5-simplex.
CONDITIONS = tuple(CLASSES) + ("global",)
ARM_NAMES = tuple(ARMS)

# --- the price ratio ---------------------------------------------------------------------------
# rho = what one generator call costs, denominated in similarity operations. It is NOT measured
# here and cannot be: it is a property of a deployment's hardware and vendor contract, not of any
# architecture. The topic's job is to show what the answer does as it moves, which is why every
# result below is reported as a function of rho rather than at one value of it.
RHO_GRID = tuple(float(x) for x in np.geomspace(1e-1, 1e5, 61))
RHO_HEADLINE = 1e3              # an LLM call against a pooled dot product, order of magnitude

# lambda is the price of cost in units of quality -- how many operations a point of accuracy is
# worth. lambda = 0 is "quality at any price"; large lambda is "cheapest thing that runs".
LAMBDA_GRID = tuple(float(x) for x in np.geomspace(1e-6, 1e-1, 61))
LAMBDA_HEADLINE = 3e-5          # chosen so the simplex is genuinely contested: at this price ALL
                                # FIVE of the arms that ever win hold territory (corrective .67,
                                # graph .13, hybrid .10, agentic .06, hyde .04). Raise it and the
                                # map collapses onto the cheapest arm, which is the topic's point
                                # rather than a tuning accident -- see winner_cells.

# --- amortization ------------------------------------------------------------------------------
# Only the graph arm holds an index that the others do not: the entity-entity similarity matrix
# its community detection runs on. Denominated in the same unit as every other cost here --
# similarity evaluations -- that build is K(K-1)/2 of them, and the community detection itself
# contributes ZERO, because it is graph combinatorics over an already-built matrix rather than
# similarity evaluation. That is a real cost the unit does not see, and it is flagged rather than
# smuggled in.
N_GRID = tuple(int(x) for x in np.unique(np.geomspace(1, 10_000, 40).astype(int)))
N_HEADLINE = 1000
K_GRID = (25, 100, 500, 2000)

_TABLE: dict | None = None


# =================================================================================================
# Movement 1 -- cost is a pair, and it is not a constant.
# =================================================================================================

def measure() -> dict:
    """Quality, operations and generator calls for every (arm, condition), measured ONCE.

    This is the only place the predecessor's arms are run. Everything after it is arithmetic on
    the three (arms x conditions) matrices, which is why the whole topic recomputes live in a
    browser from 108 numbers.

    The second of those matrices is the one the cheat sheets cannot express. A fixed arm costs
    what it costs; an ADAPTIVE arm's cost is a function of the workload it meets, because the
    thing that makes it adaptive is a gate that fires on some queries and not others.
    """
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    Q = np.zeros((len(ARM_NAMES), len(CONDITIONS)))
    O, C = np.zeros_like(Q), np.zeros_like(Q)
    L, G = local(), glob()
    for ai, a in enumerate(ARM_NAMES):
        for ki, k in enumerate(CONDITIONS):
            c = G if k == "global" else L
            idx = list(range(c["n_queries"])) if k == "global" else [int(i) for i in MECH._class_idx(c, k)]
            if not idx:
                raise ValueError(f"condition {k} has no queries")
            runs = [ARMS[a](c, i) for i in idx]
            Q[ai, ki] = np.mean([int(r["answer"] == c["gold"][i]) for r, i in zip(runs, idx)])
            O[ai, ki] = np.mean([r["ops"] for r in runs])
            C[ai, ki] = np.mean([r["calls"] for r in runs])
    _TABLE = {"Q": Q, "O": O, "C": C, "arms": ARM_NAMES, "conditions": CONDITIONS}
    return _TABLE


def cost_spread(arm: str) -> float:
    """How much an arm's operation count varies across the KIND of question it meets.

    Measured across the five local classes only. Including the global regime would compare two
    corpora of different sizes and report corpus size rather than adaptivity, which is the
    quantity of interest: a fixed arm does the same work whatever it is asked, and an adaptive
    one does not, because the gate that makes it adaptive fires on some questions and not others.
    """
    t = measure()
    row = t["O"][t["arms"].index(arm)][:len(CLASSES)]
    lo = float(row.min())
    return float(row.max() / lo) if lo > 0 else float("inf")


def uniform_within_regime() -> np.ndarray:
    """The mixture the predecessor's headline numbers are implicitly reported at: every local
    class equally likely, and the local regime as a whole weighted against the global one in
    proportion to how many queries each actually holds."""
    L, G = local(), glob()
    per_local = L["n_queries"] / len(CLASSES)
    w = np.array([per_local] * len(CLASSES) + [G["n_queries"]], dtype=float)
    return w / w.sum()


def local_only() -> np.ndarray:
    """The local regime alone -- the mixture the predecessor's LOCAL column reports."""
    w = np.array([1.0] * len(CLASSES) + [0.0])
    return w / w.sum()


def one_hot(condition: str) -> np.ndarray:
    """A deployment that only ever sees one kind of question."""
    w = np.zeros(len(CONDITIONS))
    w[CONDITIONS.index(condition)] = 1.0
    return w


def profile(w: np.ndarray, n_queries: int = N_HEADLINE, k_entities: int | None = None):
    """(quality, operations, calls) per query for every arm, on the workload mixture w.

    GUARD: w must be a probability vector over the conditions.
    """
    w = np.asarray(w, dtype=float)
    if w.shape != (len(CONDITIONS),):
        raise ValueError(f"mixture must have {len(CONDITIONS)} entries, got {w.shape}")
    if w.min() < -1e-12 or not np.isclose(w.sum(), 1.0):
        raise ValueError("mixture must be non-negative and sum to 1")
    t = measure()
    return t["Q"] @ w, t["O"] @ w + amortized_build(n_queries, k_entities), t["C"] @ w


# =================================================================================================
# Amortization -- the offline half of a cost, and where you draw the line.
# =================================================================================================

def build_cost(k_entities: int | None = None) -> np.ndarray:
    """Offline index build per arm, in similarity evaluations -- the same unit as every online
    cost here.

    Only the graph arm builds something the others do not: the entity-entity similarity matrix,
    which is K(K-1)/2 evaluations. Every arm needs the passage embeddings, so that is shared and
    cancels. The community detection on top contributes ZERO in this unit, because it is graph
    combinatorics over an already-built matrix rather than similarity evaluation -- real work
    that this unit cannot see, which is stated rather than quietly counted as free.
    """
    k = local()["K"] if k_entities is None else int(k_entities)
    if k < 2:
        raise ValueError(f"need at least 2 entities to build a graph, got {k}")
    b = np.zeros(len(ARM_NAMES))
    b[ARM_NAMES.index("graph")] = k * (k - 1) / 2.0
    return b


def amortized_build(n_queries: int = N_HEADLINE, k_entities: int | None = None) -> np.ndarray:
    """Build cost charged to each query. N -> infinity recovers the predecessor's online-only
    accounting exactly. GUARD: N >= 1."""
    if n_queries < 1:
        raise ValueError(f"a deployment serves at least one query, got {n_queries}")
    return build_cost(k_entities) / float(n_queries)


def online_ops_at(k_entities: int, arm: str) -> float:
    """A MODEL, not a measurement: how an arm's online operation count scales with corpus size.

    Every passage-scanning arm is linear in the corpus -- it scores R passages per entity. The
    graph arm is not: it scores C community summaries and then the members of one, which at the
    C = sqrt(K) that balances the two terms is O(sqrt(K)). That difference is the whole of the
    amortization story, because the SAVING grows like K while the BUILD grows like K^2, so the
    crossover moves out linearly in K rather than staying put.

    Calibrated to reproduce the measured value at the corpus's own K, so the extrapolation starts
    from the measurement rather than replacing it.
    """
    t = measure()
    k0 = local()["K"]
    measured = float(t["O"][ARM_NAMES.index(arm)] @ local_only())
    if arm == "graph":
        return measured * np.sqrt(k_entities / k0)
    return measured * (k_entities / k0)


def crossover_n(fast: str = "graph", slow: str = "naive", k_entities: int | None = None) -> float:
    """The deployment size N* at which the arm carrying an offline build overtakes the one that
    does not. Below it the build has not paid for itself; above it, it has.

    Returns inf when the arm with the build is never cheaper online, in which case there is no
    crossover to find rather than a crossover at zero.
    """
    k = local()["K"] if k_entities is None else int(k_entities)
    saving = online_ops_at(k, slow) - online_ops_at(k, fast)
    if saving <= 0:
        return float("inf")
    b = build_cost(k)[ARM_NAMES.index(fast)] - build_cost(k)[ARM_NAMES.index(slow)]
    return float(b / saving) if b > 0 else 0.0


def amortization_curve(k_entities: int | None = None, ns=N_GRID) -> list[tuple[int, float, float]]:
    """(N, graph ops/query, naive ops/query) -- the two curves that cross at N*."""
    k = local()["K"] if k_entities is None else int(k_entities)
    b = build_cost(k)
    gi, ni = ARM_NAMES.index("graph"), ARM_NAMES.index("naive")
    return [(int(n), online_ops_at(k, "graph") + b[gi] / n,
             online_ops_at(k, "naive") + b[ni] / n) for n in ns]


# =================================================================================================
# Movement 2 -- domination, and what it is a property of.
# =================================================================================================

def dominates(i: int, j: int, q, o, c) -> bool:
    """Arm i dominates arm j: at least as good on every axis, strictly better on one. Quality is
    to be maximized; both costs are to be minimized."""
    at_least = q[i] >= q[j] - 1e-12 and o[i] <= o[j] + 1e-12 and c[i] <= c[j] + 1e-12
    strictly = q[i] > q[j] + 1e-12 or o[i] < o[j] - 1e-12 or c[i] < c[j] - 1e-12
    return bool(at_least and strictly)


def pareto_set(w: np.ndarray, n_queries: int = N_HEADLINE, k_entities: int | None = None) -> list[str]:
    """The arms no other arm dominates on this workload. With two cost currencies this is a
    PARTIAL order, so the Pareto set is generally larger than it would be under one."""
    q, o, c = profile(w, n_queries, k_entities)
    return [a for i, a in enumerate(ARM_NAMES)
            if not any(dominates(j, i, q, o, c) for j in range(len(ARM_NAMES)) if j != i)]


def dominated_by(arm: str, w: np.ndarray, n_queries: int = N_HEADLINE) -> list[str]:
    """Which arms dominate this one on this workload -- empty if it is Pareto-optimal."""
    q, o, c = profile(w, n_queries)
    i = ARM_NAMES.index(arm)
    return [a for j, a in enumerate(ARM_NAMES) if j != i and dominates(j, i, q, o, c)]


# =================================================================================================
# Movement 3 -- the winner map partitions the simplex.
# =================================================================================================

def scalar_cost(w: np.ndarray, rho: float = RHO_HEADLINE, n_queries: int = N_HEADLINE,
                k_entities: int | None = None) -> np.ndarray:
    """The two currencies collapsed at a price: ops + rho * calls. Choosing rho is choosing an
    exchange rate, and rho = 0 prices generation at nothing, which is the single-cost model the
    routing topic works in."""
    if rho < 0:
        raise ValueError(f"a generator call cannot cost less than nothing, got {rho}")
    _q, o, c = profile(w, n_queries, k_entities)
    return o + rho * c


def utility(w: np.ndarray, lam: float, rho: float = RHO_HEADLINE,
            n_queries: int = N_HEADLINE) -> np.ndarray:
    """quality - lambda * cost, the linear scalarization every cost-aware benchmark computes."""
    q, _o, _c = profile(w, n_queries)
    return q - lam * scalar_cost(w, rho, n_queries)


def winner(w: np.ndarray, lam: float = LAMBDA_HEADLINE, rho: float = RHO_HEADLINE,
           n_queries: int = N_HEADLINE) -> str:
    """The arm a cost-aware evaluation would deploy on this workload at this price."""
    return ARM_NAMES[int(np.argmax(utility(w, lam, rho, n_queries)))]


def simplex_slice(x_condition: str = "bridge", y_condition: str = "global", steps: int = 21,
                  lam: float = LAMBDA_HEADLINE, rho: float = RHO_HEADLINE,
                  n_queries: int = N_HEADLINE) -> list[list[str]]:
    """The winner over a 2-D slice of the workload simplex: two conditions vary, the remaining
    four share what is left equally. Returns winners[y][x]; cells with x + y > 1 are infeasible
    and carry an empty string."""
    if x_condition == y_condition:
        raise ValueError("a slice needs two DIFFERENT conditions")
    xi, yi = CONDITIONS.index(x_condition), CONDITIONS.index(y_condition)
    rest = [i for i in range(len(CONDITIONS)) if i not in (xi, yi)]
    out = []
    for gy in range(steps):
        y = gy / (steps - 1)
        row = []
        for gx in range(steps):
            x = gx / (steps - 1)
            if x + y > 1.0 + 1e-9:
                row.append("")
                continue
            w = np.zeros(len(CONDITIONS))
            w[xi], w[yi] = x, y
            for i in rest:
                w[i] = (1.0 - x - y) / len(rest)
            row.append(winner(w / w.sum(), lam, rho, n_queries))
        out.append(row)
    return out


def winner_cells(lam: float = LAMBDA_HEADLINE, rho: float = RHO_HEADLINE,
                 trials: int = 4000, seed: int = 0, n_queries: int = N_HEADLINE) -> dict:
    """How the simplex divides up: the share of random workloads each arm wins."""
    rng = np.random.default_rng(seed)
    out: dict[str, int] = {}
    for w in rng.dirichlet(np.ones(len(CONDITIONS)), trials):
        a = winner(w, lam, rho, n_queries)
        out[a] = out.get(a, 0) + 1
    return {a: n / trials for a, n in sorted(out.items(), key=lambda kv: -kv[1])}


def bridge_crossover(lam: float = LAMBDA_HEADLINE, rho: float = RHO_HEADLINE,
                     steps: int = 101, n_queries: int = N_HEADLINE) -> float | None:
    """The bridge fraction at which the dominated-in-aggregate arm becomes the one to deploy.
    Returns None if it never does."""
    bi = CONDITIONS.index("bridge")
    rest = [i for i in range(len(CONDITIONS)) if i != bi]
    prev = None
    for s in range(steps):
        f = s / (steps - 1)
        w = np.zeros(len(CONDITIONS))
        w[bi] = f
        for i in rest:
            w[i] = (1.0 - f) / len(rest)
        a = winner(w / w.sum(), lam, rho, n_queries)
        if prev is not None and a == "agentic" and prev != "agentic":
            return float(f)
        prev = a
    return None


# =================================================================================================
# Movement 4 -- the scalarization gap.
# =================================================================================================

def hull_vertices(q: np.ndarray, cost: np.ndarray) -> list[int]:
    """Indices on the upper-left convex hull of the (cost, quality) cloud -- exactly the arms that
    maximize q - lambda*cost for some lambda >= 0.

    Sort by cost ascending, keep the quality staircase (a point costing more and scoring less can
    never be chosen), then discard any staircase point lying below the chord joining its
    neighbors: a linear objective can only ever touch a hull vertex, so an interior point is
    beaten by one of the two ends at every slope.
    """
    order = sorted(range(len(q)), key=lambda i: (cost[i], -q[i]))
    stair: list[int] = []
    best = -np.inf
    for i in order:
        # Ascending in cost, keep a point only if it buys quality no cheaper point already has.
        # The FIRST point always survives: nothing is cheaper, so nothing can dominate it, and
        # the cheapest arm is a hull vertex at lambda -> infinity by construction.
        if q[i] > best + 1e-12:
            stair.append(i)
            best = q[i]
    hull: list[int] = []
    for i in stair:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = ((cost[b] - cost[a]) * (q[i] - q[a]) - (q[b] - q[a]) * (cost[i] - cost[a]))
            if cross >= -1e-15:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def selectable(w: np.ndarray, rhos=RHO_GRID, n_queries: int = N_HEADLINE) -> list[str]:
    """Every arm that is the argmax of some linear price -- any lambda, any rho."""
    q, _o, _c = profile(w, n_queries)
    seen: set[str] = set()
    for rho in rhos:
        for i in hull_vertices(q, scalar_cost(w, rho, n_queries)):
            seen.add(ARM_NAMES[i])
    return sorted(seen)


def scalarization_gap(w: np.ndarray, n_queries: int = N_HEADLINE) -> list[str]:
    """Arms that are Pareto-optimal and yet selected by NO linear price. These are the
    architectures a cost-weighted benchmark cannot recommend at any exchange rate, however the
    weights are tuned -- not because they are bad, but because they sit inside the hull."""
    return sorted(set(pareto_set(w, n_queries)) - set(selectable(w, n_queries=n_queries)))


def gap_frequency(trials: int = 3000, seed: int = 1, n_queries: int = N_HEADLINE) -> dict:
    """How often a randomly drawn workload has a Pareto-optimal arm that no price can select,
    and which arms go missing."""
    rng = np.random.default_rng(seed)
    hits, who = 0, {}
    for w in rng.dirichlet(np.ones(len(CONDITIONS)), trials):
        gap = scalarization_gap(w, n_queries)
        if gap:
            hits += 1
            for a in gap:
                who[a] = who.get(a, 0) + 1
    return {"fraction": hits / trials, "arms": dict(sorted(who.items(), key=lambda kv: -kv[1]))}


# =================================================================================================
# Collapse anchors -- every new object must reduce to the predecessor's at its degenerate setting.
# =================================================================================================

def test_table_reproduces_the_predecessor() -> None:
    """The whole topic is arithmetic on one table, and that table must BE the predecessor's
    measurements rather than a second opinion about them. The local-only mixture reproduces its
    LOCAL column and the global one-hot reproduces its GLOBAL column, exactly."""
    M = MECH.mechanism_matrix()
    ql, _o, _c = profile(local_only(), n_queries=10 ** 12)
    qg, _o2, _c2 = profile(one_hot("global"), n_queries=10 ** 12)
    for i, a in enumerate(ARM_NAMES):
        assert abs(ql[i] - M[a]["local"]["acc"]) < 1e-12, (a, ql[i], M[a]["local"]["acc"])
        assert abs(qg[i] - M[a]["global"]["acc"]) < 1e-12, (a, qg[i], M[a]["global"]["acc"])


def test_one_hot_reproduces_the_per_class_numbers() -> None:
    """A deployment seeing only one kind of question scores exactly what the predecessor reported
    for that class -- the mixture model has no freedom at the corners of the simplex."""
    M = MECH.mechanism_matrix()
    for k in CLASSES:
        q, _o, _c = profile(one_hot(k), n_queries=10 ** 12)
        for i, a in enumerate(ARM_NAMES):
            assert abs(q[i] - M[a]["local"]["per_class"][k]) < 1e-12, (a, k)


def test_infinite_amortization_is_the_online_cost() -> None:
    """N -> infinity charges nothing for the build and recovers the predecessor's accounting,
    which reported online cost alone."""
    M = MECH.mechanism_matrix()
    _q, o, _c = profile(local_only(), n_queries=10 ** 12)
    for i, a in enumerate(ARM_NAMES):
        assert abs(o[i] - M[a]["local"]["ops"]) < 1e-6, (a, o[i], M[a]["local"]["ops"])
    assert float(amortized_build(10 ** 12).max()) < 1e-6


def test_rho_zero_is_the_single_cost_model() -> None:
    """Pricing a generator call at nothing collapses the two currencies to one and recovers the
    cost model the routing topic works in. That the ONLY difference between the two topics'
    cost structures is a price is the point: this one refuses to fix it."""
    w = local_only()
    _q, o, _c = profile(w)
    assert np.allclose(scalar_cost(w, rho=0.0), o, atol=1e-12)
    # ...and the price is not a formality. There are operating points where it decides the
    # verdict outright: at lambda = 1e-3 on this workload, pricing generation at nothing picks
    # the arm that spends four generator calls per query, and pricing it at all picks the arm
    # that spends none. Nothing about either architecture changed between those two answers.
    assert winner(w, lam=1e-3, rho=0.0) == "hyde"
    assert winner(w, lam=1e-3, rho=RHO_HEADLINE) == "graph"
    flips = [lam for lam in (1e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2)
             if len({winner(w, lam, r) for r in (0.0, 1e2, 1e5)}) > 1]
    assert len(flips) >= 3, flips


def test_hull_of_two_arms_is_the_segment() -> None:
    """With two points there is no interior, so both are selectable and the gap is empty."""
    q = np.array([0.2, 0.9])
    cost = np.array([10.0, 100.0])
    assert sorted(hull_vertices(q, cost)) == [0, 1]
    assert hull_vertices(np.array([0.5]), np.array([1.0])) == [0]


def test_cheapest_and_best_are_always_selectable() -> None:
    """The extremes of a linear objective: lambda -> infinity buys the cheapest arm and
    lambda -> 0 buys the best one, so neither can ever be in the gap."""
    for w in (local_only(), uniform_within_regime(), one_hot("bridge")):
        q, _o, _c = profile(w)
        cost = scalar_cost(w)
        sel = {ARM_NAMES[i] for i in hull_vertices(q, cost)}
        assert ARM_NAMES[int(np.argmin(cost))] in sel
        assert ARM_NAMES[int(np.argmax(q))] in sel


# =================================================================================================
# The claims, as tests. What each result depends on, and what it does not.
# =================================================================================================

def test_only_the_adaptive_arm_has_a_workload_dependent_cost() -> None:
    """A fixed pipeline does the same work whatever it is asked. The corrective arm does not,
    because its grader fires on some questions and not others -- so its cost is a function of the
    traffic it meets, and a single cost figure for it is wrong by a factor of ten."""
    # Three tiers, not two. Adaptivity is a spectrum and the spread measures where an arm sits on
    # it: the agentic arm stops hopping early when a filing opens no new direction, so its cost is
    # query-gated too -- just two orders of magnitude less dramatically than a grader that fires.
    exact = sorted(a for a in ARM_NAMES if cost_spread(a) == 1.0)
    nearly = sorted(a for a in ARM_NAMES if 1.0 < cost_spread(a) < 2.0)
    varies = sorted(a for a in ARM_NAMES if cost_spread(a) >= 5.0)
    assert exact == ["graph", "hybrid", "hyde", "naive"], exact
    assert nearly == ["agentic"], nearly
    assert varies == ["corrective"], varies
    assert len(exact) + len(nearly) + len(varies) == len(ARM_NAMES)


def test_domination_is_not_a_property_of_an_architecture() -> None:
    """THE claim. An arm can be beaten on every axis at once by the aggregate numbers and still
    be the only thing worth deploying, because domination is a property of an architecture AND a
    workload. Aggregate evidence does not survive a change of mixture."""
    agg = local_only()
    assert "agentic" not in pareto_set(agg), pareto_set(agg)
    beaten = dominated_by("agentic", agg)
    assert "naive" in beaten and len(beaten) >= 2, beaten
    # ...beaten on every axis, not merely on quality:
    q, o, c = profile(agg)
    ai, ni = ARM_NAMES.index("agentic"), ARM_NAMES.index("naive")
    assert q[ai] < q[ni] and o[ai] > o[ni] and c[ai] > c[ni]
    # ...and yet, on a workload made of the questions it was built for, it wins outright.
    pure = one_hot("bridge")
    assert winner(pure, lam=0.0) == "agentic"
    assert "agentic" in pareto_set(pure)
    f = bridge_crossover()
    assert f is not None and 0.2 < f < 0.9, f


def test_the_winner_map_partitions_the_simplex() -> None:
    """No arm wins everywhere, and the division is not a technicality: several arms hold real
    territory. Raising the price of cost collapses the map toward the cheapest arm, which is why
    a benchmark's cost weighting decides its verdict before any measurement is taken."""
    cheap = winner_cells(lam=LAMBDA_HEADLINE, trials=1500)
    assert len(cheap) >= 3, cheap
    assert max(cheap.values()) < 0.9, cheap
    dear = winner_cells(lam=1e-1, trials=800)
    assert max(dear.values()) > max(cheap.values()), (cheap, dear)
    grid = simplex_slice(steps=11)
    seen = {c for row in grid for c in row if c}
    assert len(seen) >= 2, seen


def test_linear_pricing_cannot_see_every_pareto_arm() -> None:
    """The scalarization gap. A linear objective only ever touches a vertex of the hull, so a
    Pareto-optimal arm lying inside it is recommended by no exchange rate whatsoever -- not
    because it is bad, and not because the weights were tuned wrong.

    On the mixtures the predecessor implicitly reported at, the gap happens to be EMPTY, and
    that is stated rather than hidden: the result is that the gap is common across workloads,
    not that it is everywhere.
    """
    assert scalarization_gap(local_only()) == []
    assert scalarization_gap(uniform_within_regime()) == []
    freq = gap_frequency(trials=800)
    assert 0.1 < freq["fraction"] < 0.5, freq["fraction"]
    assert "agentic" in freq["arms"], freq["arms"]
    # the arm most often invisible is the same arm the aggregate numbers already dominated
    assert max(freq["arms"], key=freq["arms"].get) == "agentic", freq["arms"]


def test_a_gap_arm_is_genuinely_unbuyable() -> None:
    """Directly, on a witness: find a workload with a non-empty gap and confirm the missing arm
    is the argmax for NO (lambda, rho) on a dense grid -- the theorem made numerical."""
    rng = np.random.default_rng(1)
    for _ in range(400):
        w = rng.dirichlet(np.ones(len(CONDITIONS)))
        gap = scalarization_gap(w)
        if not gap:
            continue
        missing = gap[0]
        q, _o, _c = profile(w)
        for rho in RHO_GRID:
            cost = scalar_cost(w, rho)
            for lam in LAMBDA_GRID:
                assert ARM_NAMES[int(np.argmax(q - lam * cost))] != missing, (missing, lam, rho)
        assert missing in pareto_set(w)
        return
    raise AssertionError("no workload with a scalarization gap was found to witness")


def test_amortization_moves_the_frontier() -> None:
    """Whether the arm holding an offline index dominates the one that does not is a question
    about deployment size, not about either architecture. Below the crossover the build has not
    paid for itself and the domination reverses."""
    nstar = crossover_n()
    assert 1.0 < nstar < 100.0, nstar
    big, small = int(nstar * 20) + 1, 1
    assert "naive" in dominated_by("naive", local_only(), n_queries=big) or \
        "graph" in dominated_by("naive", local_only(), n_queries=big)
    _q, o_big, _c = profile(local_only(), n_queries=big)
    _q2, o_small, _c2 = profile(local_only(), n_queries=small)
    gi, ni = ARM_NAMES.index("graph"), ARM_NAMES.index("naive")
    assert o_big[gi] < o_big[ni], (o_big[gi], o_big[ni])       # amortized: graph is cheaper
    assert o_small[gi] > o_small[ni], (o_small[gi], o_small[ni])  # one query: it is not


def test_the_crossover_moves_out_with_corpus_size() -> None:
    """And it is not a constant either. The build is quadratic in the entity count while the
    saving is linear, so the deployment has to be bigger before the index pays for itself --
    the opposite of the intuition that an index matters more as a corpus grows."""
    ns = [crossover_n(k_entities=k) for k in K_GRID]
    assert all(b > a for a, b in zip(ns, ns[1:])), ns
    assert ns[-1] > 10 * ns[0], ns


# =================================================================================================
# viz_constants -- this function OWNS every number the laboratory displays.
#
# The whole topic is arithmetic on one 6 x 6 x 3 table, so the laboratory bakes that table -- 108
# numbers -- and recomputes the profiles, the Pareto sets, the hulls, the winner map and the
# amortization curves LIVE. Nothing derived is baked, which is why nothing derived can drift.
# =================================================================================================

def viz_constants() -> dict:
    t = measure()
    r3 = lambda v: round(float(v), 3)                                          # noqa: E731
    gap = gap_frequency(trials=3000)
    return {
        "arms": list(ARM_NAMES), "conditions": list(CONDITIONS),
        "Q": [[r3(v) for v in row] for row in t["Q"]],
        "O": [[r3(v) for v in row] for row in t["O"]],
        "C": [[r3(v) for v in row] for row in t["C"]],
        "build": [r3(v) for v in build_cost()],
        "k0": int(local()["K"]),
        "n_docs": int(local()["n_docs"]), "n_sectors": int(N_SECTORS),
        "r_passages": int(R_PASSAGES), "c_pooled": int(C_POOLED),
        "lambda_headline": LAMBDA_HEADLINE, "rho_headline": RHO_HEADLINE,
        "n_headline": N_HEADLINE,
        "lambda_range": [LAMBDA_GRID[0], LAMBDA_GRID[-1]],
        "rho_range": [RHO_GRID[0], RHO_GRID[-1]],
        # reference values the browser must reproduce from the table alone
        "cost_spread": {a: r3(cost_spread(a)) for a in ARM_NAMES},
        "pareto_local": pareto_set(local_only()),
        "dominated_agentic": dominated_by("agentic", local_only()),
        "winner_cells": {a: r3(v) for a, v in winner_cells(trials=3000).items()},
        "bridge_crossover": r3(bridge_crossover()),
        "gap_fraction": r3(gap["fraction"]),
        "gap_arms": {a: int(n) for a, n in gap["arms"].items()},
        "crossover_n": r3(crossover_n()),
        "crossover_by_k": [[int(k), r3(crossover_n(k_entities=k))] for k in K_GRID],
        "amortization": [[int(n), r3(g), r3(nv)] for n, g, nv in amortization_curve()],
        # the price flip: same workload, same lambda, two exchange rates, two answers
        "rho_flip": {"lambda": 1e-3,
                     "at_zero": winner(local_only(), 1e-3, 0.0),
                     "at_headline": winner(local_only(), 1e-3, RHO_HEADLINE)},
    }


def test_laboratory_constants_match_the_module() -> None:
    """The laboratory's baked block is emitted from viz_constants() rather than transcribed, but
    a retune can still regenerate one and not the other. This parses the shipped .tsx back and
    compares it to a fresh bake."""
    import json
    import re
    tsx = pathlib.Path(__file__).resolve().parents[2] / "src/components/viz/RagArchitectureParetoLaboratory.tsx"
    if not tsx.exists():                       # the notebook must run outside the site checkout
        return
    text, v = tsx.read_text(), viz_constants()
    baked = {}
    for name, expr in re.findall(r"^const ([A-Z_0-9]+)(?:: [^=]+)? = (.+?);\s*$", text, re.M):
        try:
            baked[name] = json.loads(expr.replace(" as const", ""))
        except json.JSONDecodeError:
            continue
    for name, key in (("ARMS", "arms"), ("CONDITIONS", "conditions"), ("Q", "Q"), ("O", "O"),
                      ("C", "C"), ("BUILD", "build"), ("COST_SPREAD", "cost_spread"),
                      ("WINNER_CELLS", "winner_cells"), ("GAP_ARMS", "gap_arms"),
                      ("CROSSOVER_BY_K", "crossover_by_k"), ("RHO_FLIP", "rho_flip")):
        assert name in baked, f"{name} is no longer baked into the laboratory"
        assert baked[name] == v[key], (name, "laboratory and module disagree")
    # The scalars are quoted directly in the panel notes a reader actually reads, so they drift the
    # same way. They are declared several to a line, so they are matched by name rather than by the
    # whole-line JSON parse above.
    for name, key in (("K0", "k0"), ("N_DOCS", "n_docs"), ("N_SECTORS", "n_sectors"),
                      ("LAM0", "lambda_headline"), ("RHO0", "rho_headline"), ("N0", "n_headline"),
                      ("GAP_FRACTION", "gap_fraction"), ("BRIDGE_CROSSOVER", "bridge_crossover"),
                      ("CROSSOVER_N", "crossover_n")):
        hit = re.search(rf"\b{name} = ([0-9.eE+-]+)", text)
        assert hit, f"{name} is no longer baked into the laboratory"
        assert float(hit.group(1)) == float(v[key]), (name, float(hit.group(1)), v[key])


def test_topic_prose_matches_the_module() -> None:
    """A retune moves numbers quoted in words, and the prose is the one thing nothing else checks.
    Every figure the topic states in a sentence is parsed back out and compared."""
    import re
    mdx = pathlib.Path(__file__).resolve().parents[2] / "src/content/topics/rag-architecture-pareto.mdx"
    if not mdx.exists():
        return
    text, v = mdx.read_text(), viz_constants()
    for pattern, value in (
        (r"varies by a factor of ([\d.]+)", v["cost_spread"]["corrective"]),
        (r"bridge fraction reaches ([\d.]+)", v["bridge_crossover"]),
        (r"on ([\d.]+)% of randomly drawn workloads", round(v["gap_fraction"] * 100, 1)),
        (r"crossover at\s+N[^0-9]{0,8}([0-9.]+)", round(v["crossover_n"], 2)),
        (r"nearly\* flat at\s*\$([0-9.]+)", round(v["cost_spread"]["agentic"], 2)),
    ):
        m = re.search(pattern, text)
        assert m, f"the topic no longer states: {pattern}"
        assert float(m.group(1)) == value, (pattern, float(m.group(1)), value)

    # The winner-map shares the topic quotes in a sentence. These move whenever LAMBDA_HEADLINE,
    # the sampling seed or the corpus does, and nothing else checks them.
    for arm, share in v["winner_cells"].items():
        hit = re.search(rf"`{arm}` (\d+)%", text)
        assert hit, f"the topic no longer quotes a winner share for {arm}"
        assert int(hit.group(1)) == round(share * 100), (arm, hit.group(1), share)

    # ...and the comparison that carries the whole domination argument: beaten on EVERY axis.
    q, o, c = profile(local_only(), n_queries=10 ** 12)
    ai, ni = ARM_NAMES.index("agentic"), ARM_NAMES.index("naive")
    hit = re.search(r"lower quality \(([\d.]+) against ([\d.]+)\), more operations \((\d+) against "
                    r"(\d+)\), and\s+more generator calls \(([\d.]+) against (\d+)\)", text)
    assert hit, "the topic no longer states the agentic-against-naive comparison"
    assert [float(x) for x in hit.groups()] == [
        round(q[ai], 3), round(q[ni], 3), float(round(o[ai])), float(round(o[ni])),
        round(c[ai], 2), float(c[ni])], hit.groups()


def _fmt_table() -> str:
    t = measure()
    rows = [f"  {'arm':11s} " + " ".join(f"{k:>9s}" for k in CONDITIONS)]
    for label, M, fmt in (("quality", t["Q"], "9.3f"), ("ops", t["O"], "9.0f"), ("calls", t["C"], "9.2f")):
        rows.append(f"  -- {label} --")
        for ai, a in enumerate(ARM_NAMES):
            rows.append(f"  {a:11s} " + " ".join(f"{M[ai, ki]:{fmt}}" for ki in range(len(CONDITIONS))))
    return "\n".join(rows)


def _run_tests() -> None:
    names = sorted(n for n, v in globals().items() if n.startswith("test_") and callable(v))
    for n in names:
        globals()[n]()
        print(f"  ok  {n}")
    print(f"{len(names)} assertions passed")


def bridge_family(frac: float) -> np.ndarray:
    """The workload family the laboratory's bridge slider sweeps: a given share on bridge
    questions and the rest spread evenly over the other five kinds."""
    return np.array([frac if c == "bridge" else (1.0 - frac) / (len(CONDITIONS) - 1)
                     for c in CONDITIONS])


def test_the_laboratory_slider_reproduces_the_gap() -> None:
    """Panel D lets a reader slide to a workload and read off which arms no price can buy, and it
    recomputes that in the browser over the SAME rho grid this module sweeps. Pin the witness the
    panel is built to show, so a retune cannot quietly empty it.

    The stronger statement is also the subtler one: an arm off the hull at ONE exchange rate may
    be on it at another, so only the union over rho is the theorem's claim. At a 70% bridge share
    two arms are off the hull at the headline price and only one of them is off it everywhere.
    """
    w = bridge_family(0.70)
    assert pareto_set(w) == ["hybrid", "corrective", "graph", "agentic"], pareto_set(w)
    assert scalarization_gap(w) == ["corrective"], scalarization_gap(w)
    q, o, c = profile(w, N_HEADLINE)
    at_headline = {ARM_NAMES[i] for i in hull_vertices(q, scalar_cost(w, RHO_HEADLINE))}
    off_here = [a for a in pareto_set(w) if a not in at_headline]
    assert off_here == ["hybrid", "corrective"], off_here
    assert "hybrid" in selectable(w) and "corrective" not in selectable(w)
    # ...and the family genuinely has a window where the gap is open, not a single lucky point
    open_at = [f for f in (0.5, 0.6, 0.7, 0.8, 0.9) if scalarization_gap(bridge_family(f))]
    assert len(open_at) >= 4, open_at


if __name__ == "__main__":
    print("THE TABLE — quality and both costs, per (arm, condition)")
    print(_fmt_table())
    print()
    print("cost spread WITHIN the local regime (a single cost figure is wrong by this factor):")
    for a in ARM_NAMES:
        print(f"  {a:11s} {cost_spread(a):6.2f}x")
    print()
    w = local_only()
    print(f"LOCAL-only workload: Pareto set = {pareto_set(w)}")
    print(f"  agentic is dominated by {dominated_by('agentic', w)} — and yet:")
    print(f"  it wins outright once the bridge fraction reaches {bridge_crossover():.2f}")
    print()
    print(f"the winner map over random workloads (lambda={LAMBDA_HEADLINE:g}, rho={RHO_HEADLINE:g}):")
    for a, f in winner_cells().items():
        print(f"  {a:11s} {f:6.1%} of workloads")
    g = gap_frequency()
    print()
    print(f"scalarization gap: {g['fraction']:.1%} of workloads have a Pareto-optimal arm that")
    print(f"  NO linear price can select — most often {max(g['arms'], key=g['arms'].get)} {g['arms']}")
    print()
    print(f"amortization: the graph arm's build pays for itself after N* = {crossover_n():.2f} queries")
    for k, n in [(k, crossover_n(k_entities=k)) for k in K_GRID]:
        print(f"  K={k:5d} entities -> N* = {n:8.1f} queries")
    print()
    _run_tests()
