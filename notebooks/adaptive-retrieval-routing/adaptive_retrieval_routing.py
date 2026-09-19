"""Adaptive Retrieval Routing: choosing a retrieval strategy per query, under cost.

Run:
    uv run --with numpy --with scipy python notebooks/adaptive-retrieval-routing/adaptive_retrieval_routing.py

The published stack ends with a BINARY gate: selective generation decides emit-or-abstain
by Chow's rule. This topic is the K-action generalization. A router picks among several
retrieval strategies -- answer from the query alone, retrieve once, or iterate -- using only
features available BEFORE any retrieval fires, and trades answer quality against the cost of
the arm it picks.

Three results, in increasing order of how much they matter:

  THEOREM 1 (K-action rule). Minimizing the expected loss c_err*(1-Q_a) + c_a pointwise gives
      pi*(x) = argmax_a { c_err * E[Q_a | phi(x)] - c_a }.
      At K=2 over {emit, abstain} this reduces EXACTLY to the imported `chow_decision` at the
      imported `chow_threshold` -- the router generalizes the published gate rather than
      replacing it.

  THEOREM 2 (the frontier is a hull). Sweeping lambda traces a family of deterministic
      policies; randomizing between adjacent ones fills in the segments, so the achievable
      (cost, quality) region is the convex hull of those points. Three nested frontiers:
      the hull of the FIXED arms (what you get with no features) sits below the router's
      achievable frontier, which sits below the oracle's.

  THEOREM 3 (the Jensen gap -- the one that matters). The most a router can win over the best
      single arm is
          Oracle - BestFixed = E[max_a U_a] - max_a E[U_a] >= 0,
      a Jensen gap that is strictly positive IFF the arms' advantage ordering actually varies
      across queries. So: routing is worthless, no matter how good the classifier, when one
      arm is uniformly best. That is the claim worth carrying into an interview, and it is the
      structural analogue of `rc_gap` one level up -- there the gap was between a score and an
      oracle over one arm, here it is between arms.

  PROPOSITION 4 (the honesty hinge). pi-hat uses ESTIMATED E[Q_a | phi]. The excess loss over
      pi* is bounded by the summed L1 estimation error across arms, so the router is no better
      than the calibration of the numbers it compares -- exactly the caveat Chow carried. A
      router tuned and scored on the same queries looks roughly twice as good as it is, which
      is why every number below is reported on a held-out split.

Imports and never reimplements: the answer model, the arms' scorer, the reformulation
operator, the abstention gate, the calibration suite and the metric estimators all come from
published topics.
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

# --- the import chain: every ancestor's HYPHENATED dir, then its UNDERSCORED module ----------
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "dense-retrieval-dual-encoders",
    "pmi-retrieval-value",
    "retrieval-vs-long-context",
    "multi-hop-iterative-retrieval",
    "selective-generation-abstention",
    "significance-testing-calibration",
    "set-metrics-precision-recall-map-mrr",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf          # noqa: E402
from pmi_retrieval_value import answer_posterior, entropy           # noqa: E402
from retrieval_vs_long_context import (                             # noqa: E402
    answer_posterior_topk,
    rvlc_corpus,
    TAU_GEN,
)
from multi_hop_iterative_retrieval import reformulate               # noqa: E402
from selective_generation_abstention import (                       # noqa: E402
    chow_decision,
    chow_threshold,
    C_ABS,
    C_ERR,
)
from significance_testing_calibration import (                      # noqa: E402
    expected_calibration_error,
    paired_t_test,
)

# ---------------------------------------------------------------------------------------------
# Constants. Every tuned value is marked; the rigorFlag names them.
# ---------------------------------------------------------------------------------------------
SEED = 7                      # the shared finance-geometry seed
TAU_ANS = TAU_GEN             # 0.30 -- the imported generation temperature

# Arm resource costs, in units of one retrieval call. The SPREAD is load-bearing: with costs
# close together the router's only available win is quality, and Theorem 2's frontier
# collapses to a point. Tuned so iterating is genuinely expensive (the capstone's
# water-filling lesson: without real cost asymmetry the allocation problem is vacuous).
C_NONE = 0.0                  # answer from the query alone: no retrieval
C_SINGLE = 1.0                # one retrieval
C_ITER = 3.0                  # retrieve, reformulate, retrieve again, and read both

ARMS = ("none", "single", "iterative")
ARM_COST = {"none": C_NONE, "single": C_SINGLE, "iterative": C_ITER}

# Query-class mix. Heterogeneity in REQUIRED EFFORT is what creates the Jensen gap; a corpus
# where one arm is uniformly best makes the whole topic vacuous (Theorem 3 says so explicitly),
# so the corpus is built to contain all three regimes and a test asserts it does.
N_PER_CLASS = 24              # queries per class -> 72 total
KAPPA_KNOWN = 400.0           # class A: query sits on its company prototype; the query-only
                              #          posterior is already sharp, so retrieval buys nothing
KAPPA_LOCAL = 45.0            # class B: query identifies the sector but not the company;
                              #          one retrieval resolves it
ALPHA_BRIDGE_DEG = 40.0       # class C: the mention angle. Tuned -- cos(40 deg) = 0.766 must
                              #          clear the worst same-sector company cosine, or the
                              #          bridge is outranked by an ordinary passage. The
                              #          multi-hop topic records that 45 deg was already too close.
KAPPA_BRIDGE = 300.0          # concentration of the bridge passage around its mention direction
BRIDGE_MIX = 0.72             # class C query: how far toward the SOURCE company it sits. Below
                              #          1.0 the query references two entities, which is what
                              #          makes the class detectable pre-retrieval at all.

CAL_FRAC = 0.5                # calibration/test split. Reporting in-sample would roughly
                              # double the apparent gain (Proposition 4).
LAMBDA_GRID = tuple(round(float(x), 4) for x in np.geomspace(1e-3, 3.0, 40))
RIDGE_GRID = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)   # selected by CV INSIDE the calibration split
CV_FOLDS = 4

_CORPUS: dict | None = None


# ---------------------------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------------------------
def routing_corpus(seed: int = SEED) -> dict:
    """The finance geometry of `rvlc_corpus`, with three query classes over ONE passage set.

    Documents are the imported corpus's passages, plus one BRIDGE passage per source company:
    a filing of company X that mentions company Y, built on the multi-hop mention geometry
    f = cos(alpha) u_X + sin(alpha) u_Y. The bridge is what makes a second hop reach an answer
    a single retrieval provably cannot.

    Three query classes, drawn over the same prototypes:
      known  -- on the company prototype; the query alone identifies the answer
      local  -- near the sector; one retrieval resolves the company
      bridge -- references two entities; the answer is the MENTIONED company, not the nearest one
    """
    base = rvlc_corpus(seed=seed)
    protos = base["protos"]                       # (K, dim) company prototypes
    passages = base["passages"]                   # (n, dim)
    company_of_passage = base["company_of_passage"]
    K, dim = protos.shape
    rng = np.random.default_rng(seed + 991)

    alpha = np.deg2rad(ALPHA_BRIDGE_DEG)
    # Pair each company X with a company Y in a DIFFERENT sector, so the mentioned answer is
    # not reachable by drifting within X's own sector shell.
    sector_of_company = base["sector_of_company"]
    bridge_pairs: list[tuple[int, int]] = []
    for x in range(K):
        candidates = [y for y in range(K) if sector_of_company[y] != sector_of_company[x]]
        bridge_pairs.append((x, int(candidates[(x * 5 + 3) % len(candidates)])))

    bridge_vecs, bridge_src, bridge_dst = [], [], []
    for x, y in bridge_pairs:
        mention = normalize(np.cos(alpha) * protos[x] + np.sin(alpha) * protos[y])
        drawn = sample_vmf(1, mention, KAPPA_BRIDGE, seed=seed + 7717 + 13 * x)[0]
        bridge_vecs.append(normalize(drawn))
        bridge_src.append(x)
        bridge_dst.append(y)
    bridge_vecs = np.array(bridge_vecs)

    docs = np.vstack([passages, bridge_vecs])
    owner = np.concatenate([company_of_passage, np.array(bridge_dst)])
    is_bridge = np.concatenate([np.zeros(len(passages), int), np.ones(len(bridge_vecs), int)])

    Q, truth, klass = [], [], []
    # A: known -- the query is the prototype, to numerical precision.
    for i in range(N_PER_CLASS):
        a = i % K
        v = sample_vmf(1, protos[a], KAPPA_KNOWN, seed=seed + 20001 + i)[0]
        Q.append(normalize(v)); truth.append(a); klass.append("known")
    # B: local -- near the company but loose enough that same-sector companies compete.
    for i in range(N_PER_CLASS):
        a = (i * 3 + 1) % K
        v = sample_vmf(1, protos[a], KAPPA_LOCAL, seed=seed + 30001 + i)[0]
        Q.append(normalize(v)); truth.append(a); klass.append("local")
    # C: bridge -- references X and Y; the ANSWER is Y, reachable only through X's bridge filing.
    for i in range(N_PER_CLASS):
        x, y = bridge_pairs[i % K]
        mixed = BRIDGE_MIX * protos[x] + (1.0 - BRIDGE_MIX) * protos[y]
        v = sample_vmf(1, normalize(mixed), KAPPA_BRIDGE, seed=seed + 40001 + i)[0]
        Q.append(normalize(v)); truth.append(y); klass.append("bridge")

    _ = rng  # geometry is seeded per draw; the generator is kept for provenance only
    return {
        "protos": protos,
        "sector_of_company": sector_of_company,
        "docs": docs,
        "owner": owner,
        "is_bridge": is_bridge,
        "Q": np.array(Q),
        "truth": np.array(truth),
        "klass": np.array(klass),
        "K": int(K),
        "dim": int(dim),
        "n_docs": int(docs.shape[0]),
        "n_queries": int(len(Q)),
    }


def _corpus() -> dict:
    """Module-scope cache: the <60s budget discipline this repo runs on."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = routing_corpus()
    return _CORPUS


# ---------------------------------------------------------------------------------------------
# The three arms. Every arm is the SAME imported answer model with different evidence, so a
# quality difference is attributable to the retrieval strategy and nothing else.
# ---------------------------------------------------------------------------------------------
def _top1(corpus: dict, probe: np.ndarray) -> int:
    return int(np.argmax(corpus["docs"] @ probe))


def arm_none(corpus: dict, q: np.ndarray) -> np.ndarray:
    """Answer from the query alone. A degenerate-weight call of the imported posterior:
    weight 0 on the context zeroes the document term, leaving softmax(<q, protos>/tau)."""
    return answer_posterior_topk(q, corpus["docs"][:1], np.array([0.0]),
                                 corpus["protos"], tau=TAU_ANS)


def arm_single(corpus: dict, q: np.ndarray) -> np.ndarray:
    """Retrieve once, read the top passage. At k=1 with unit weight this IS the imported
    `answer_posterior` -- the collapse anchor pins it."""
    d = corpus["docs"][_top1(corpus, q)]
    return answer_posterior_topk(q, d[None, :], np.array([1.0]),
                                 corpus["protos"], tau=TAU_ANS)


def arm_iterative(corpus: dict, q: np.ndarray, max_hops: int = 2) -> np.ndarray:
    """Retrieve, reformulate off what was read, retrieve again, and read both.

    `reformulate` is imported: q' = normalize(d - <d,q> q), the part of the read filing
    orthogonal to the query -- on the mention geometry that residual is the named company.
    With max_hops=1 this is byte-for-byte `arm_single`, which is the second collapse anchor.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    probe = q
    read: list[np.ndarray] = []
    seen: set[int] = set()
    for _ in range(max_hops):
        i = _top1(corpus, probe)
        if i in seen:
            break
        seen.add(i)
        read.append(corpus["docs"][i])
        probe = reformulate(probe, corpus["docs"][i])
    ctx = np.array(read)
    return answer_posterior_topk(q, ctx, np.ones(len(read)), corpus["protos"], tau=TAU_ANS)


ARM_FN = {"none": arm_none, "single": arm_single, "iterative": arm_iterative}


def arm_quality(corpus: dict) -> np.ndarray:
    """Q[q, a] = posterior mass the arm puts on the gold company. Shape (n_queries, 3)."""
    Qm = np.zeros((corpus["n_queries"], len(ARMS)))
    for qi in range(corpus["n_queries"]):
        q = corpus["Q"][qi]
        gold = int(corpus["truth"][qi])
        for ai, name in enumerate(ARMS):
            Qm[qi, ai] = float(ARM_FN[name](corpus, q)[gold])
    return Qm


# ---------------------------------------------------------------------------------------------
# Router features -- PRE-RETRIEVAL ONLY. A feature that needs the retrieved set makes this a
# reranker, not a router, and quietly charges the cost the router exists to save.
# ---------------------------------------------------------------------------------------------
def router_features(corpus: dict) -> np.ndarray:
    """(n_queries, 4): [H(query-only posterior), nearest cosine, top-2 margin, cross-sector max].

    All three are computable before any retrieval fires: the first is the entropy of
    `arm_none`'s output, which the router computes anyway; the other two are one matrix
    product against the prototypes.

    The last one earns its place; the first three do not separate what matters. A compositional
    query references two entities in DIFFERENT sectors, so it carries real mass outside the
    sector of its own nearest prototype -- while a single-company query, however ambiguous, has
    its competition inside its own sector shell. Entropy, nearest-cosine and the top-two margin
    all place the compositional class BETWEEN the other two rather than apart from them (the
    top-two competitor is a same-sector neighbour, not the mentioned company), and with those
    three the router collapses to always-retrieve-once at moderate lambda and gains exactly
    nothing. The cross-sector maximum is what makes the class visible without paying for a
    retrieval to discover it.

    These features were chosen against this corpus, which the rigorFlag states plainly. The
    result that does NOT depend on that choice is Theorem 3: the Jensen gap bounds what any
    router can win, and a real one captures only part of it.
    """
    feats = np.zeros((corpus["n_queries"], 4))
    sims = corpus["Q"] @ corpus["protos"].T                  # (n_queries, K)
    part = np.sort(sims, axis=1)
    sect = np.asarray(corpus["sector_of_company"])
    for qi in range(corpus["n_queries"]):
        feats[qi, 0] = entropy(arm_none(corpus, corpus["Q"][qi]))
        nearest = int(np.argmax(sims[qi]))
        off = sims[qi][sect != sect[nearest]]
        feats[qi, 3] = float(off.max()) if off.size else 0.0
    feats[:, 1] = part[:, -1]
    feats[:, 2] = part[:, -1] - part[:, -2]
    return feats


def _design(feats: np.ndarray) -> np.ndarray:
    """Quadratic design matrix (15 columns from 4 features). Ridge on this is a closed form --
    no SGD, so every number is bit-reproducible (the repo's bake-only-reproducible rule)."""
    cols = [np.ones(len(feats))] + [feats[:, j] for j in range(feats.shape[1])]
    for i, j in itertools.combinations_with_replacement(range(feats.shape[1]), 2):
        cols.append(feats[:, i] * feats[:, j])
    return np.column_stack(cols)


def fit_quality_model(feats: np.ndarray, Qm: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Closed-form ridge, one column per arm: W such that Qhat = _design(feats) @ W."""
    if ridge <= 0:
        raise ValueError(f"ridge must be > 0, got {ridge}")
    X = _design(feats)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Qm)


def select_ridge_cv(feats: np.ndarray, Qm: np.ndarray, lam: float, costs: np.ndarray,
                    folds: int = CV_FOLDS, grid=RIDGE_GRID, seed: int = SEED) -> float:
    """Choose the ridge by k-fold CV INSIDE the calibration split.

    This matters more than it looks. Picking the penalty by whichever value scored best on the
    test split would be selection on test -- the router would then be reported against data
    that helped choose it, which is the same in-sample inflation Proposition 4 warns about,
    one level up. The folds never see the test half.
    """
    n = len(feats)
    order = np.random.default_rng(seed + 5).permutation(n)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    best_ridge, best_score = float(grid[0]), -np.inf
    for ridge in grid:
        total = 0.0
        for f in range(folds):
            hold = order[bounds[f]:bounds[f + 1]]
            keep = np.setdiff1d(order, hold, assume_unique=False)
            if len(hold) == 0 or len(keep) == 0:
                continue
            W = fit_quality_model(feats[keep], Qm[keep], ridge)
            ch = route(predict_quality(feats[hold], W), lam, costs)
            total += float(utility(Qm[hold], ch, lam, costs).sum())
        if total > best_score:
            best_score, best_ridge = total, float(ridge)
    return best_ridge


def predict_quality(feats: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Predicted per-arm quality, clipped to [0, 1] -- it is a probability mass."""
    return np.clip(_design(feats) @ W, 0.0, 1.0)


# ---------------------------------------------------------------------------------------------
# THEOREM 1 -- the K-action rule, and its exact reduction to the published abstention gate
# ---------------------------------------------------------------------------------------------
def route(Qhat: np.ndarray, lam: float, costs: np.ndarray) -> np.ndarray:
    """pi(x) = argmax_a { Qhat[x, a] - lam * c_a }. Ties break to the cheaper arm."""
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    return np.argmax(Qhat - lam * costs[None, :], axis=1)


def utility(Qm: np.ndarray, choice: np.ndarray, lam: float, costs: np.ndarray) -> np.ndarray:
    """Realized per-query utility Q_{pi(x)}(x) - lam * c_{pi(x)}."""
    idx = np.arange(Qm.shape[0])
    return Qm[idx, choice] - lam * costs[choice]


def chow_as_two_action(scores: np.ndarray, c_err: float = C_ERR,
                       c_abs: float = C_ABS) -> np.ndarray:
    """The K=2 case of `route`, written in the loss form, to show it IS Chow's rule.

    Per-arm expected loss is c_err*(1 - Q_a) + c_a. Emitting has quality `scores` and no
    resource cost; abstaining has quality 0 and bookkeeping cost c_abs - c_err, which is
    exactly what makes its loss c_err*(1-0) + (c_abs - c_err) = c_abs. Minimizing gives
    emit iff c_err*(1-s) < c_abs, i.e. s > 1 - c_abs/c_err = the imported `chow_threshold`.

    Returns 1 for emit, 0 for abstain -- matching the imported `chow_decision`.
    """
    s = np.asarray(scores, dtype=float)
    Qm = np.column_stack([np.zeros_like(s), s])          # [abstain, emit]
    costs = np.array([c_abs - c_err, 0.0])
    loss = c_err * (1.0 - Qm) + costs[None, :]
    return np.argmin(loss, axis=1).astype(int)


# ---------------------------------------------------------------------------------------------
# THEOREM 2 -- the achievable frontier is a convex hull
# ---------------------------------------------------------------------------------------------
def policy_point(Qm: np.ndarray, choice: np.ndarray, costs: np.ndarray) -> tuple[float, float]:
    """(mean cost, mean quality) of a deterministic policy."""
    idx = np.arange(Qm.shape[0])
    return float(costs[choice].mean()), float(Qm[idx, choice].mean())


def upper_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The upper-left boundary of the convex hull: the Pareto frontier under randomization.

    Monotone chain on points sorted by cost, keeping right turns -- the concave envelope.
    """
    pts = sorted(set((round(c, 12), round(q, 12)) for c, q in points))
    if len(pts) <= 2:
        return pts
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # cross product of (p2-p1) x (p-p1); <= 0 means p2 is not above the line
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    # keep only the non-dominated prefix: quality must be non-decreasing in cost
    out: list[tuple[float, float]] = []
    best = -np.inf
    for c, q in hull:
        if q > best:
            out.append((c, q)); best = q
    return out


def fixed_arm_points(Qm: np.ndarray, costs: np.ndarray) -> list[tuple[float, float]]:
    """The three always-this-arm policies -- what is achievable with NO features."""
    return [(float(costs[a]), float(Qm[:, a].mean())) for a in range(Qm.shape[1])]


def router_frontier(Qm: np.ndarray, Qhat: np.ndarray, costs: np.ndarray,
                    lambdas=LAMBDA_GRID) -> list[dict]:
    """Sweep lambda; each value gives a deterministic policy and one (cost, quality) point."""
    rows = []
    for lam in lambdas:
        ch = route(Qhat, lam, costs)
        c, q = policy_point(Qm, ch, costs)
        rows.append({"lam": float(lam), "cost": c, "quality": q,
                     "mix": {ARMS[a]: int((ch == a).sum()) for a in range(len(ARMS))}})
    return rows


def oracle_frontier(Qm: np.ndarray, costs: np.ndarray,
                    lambdas=LAMBDA_GRID) -> list[dict]:
    """The same sweep with PERFECT per-query foresight -- the upper bound on any router."""
    rows = []
    for lam in lambdas:
        ch = np.argmax(Qm - lam * costs[None, :], axis=1)
        c, q = policy_point(Qm, ch, costs)
        rows.append({"lam": float(lam), "cost": c, "quality": q})
    return rows


# ---------------------------------------------------------------------------------------------
# THEOREM 3 -- the Jensen gap: the most any router can win
# ---------------------------------------------------------------------------------------------
def jensen_gap(Qm: np.ndarray, lam: float, costs: np.ndarray) -> dict:
    """Oracle - BestFixed = E[max_a U_a] - max_a E[U_a] >= 0, zero iff one arm is a.s. best."""
    U = Qm - lam * costs[None, :]
    oracle = float(U.max(axis=1).mean())
    per_arm = U.mean(axis=0)
    best_fixed = float(per_arm.max())
    argmaxes = U.argmax(axis=1)
    return {
        "lam": float(lam),
        "oracle": oracle,
        "best_fixed": best_fixed,
        "gap": oracle - best_fixed,
        "best_arm": ARMS[int(per_arm.argmax())],
        "argmax_varies": bool(len(set(argmaxes.tolist())) > 1),
        "arm_win_counts": {ARMS[a]: int((argmaxes == a).sum()) for a in range(len(ARMS))},
    }


def degenerate_gap(Qm: np.ndarray, lam: float, costs: np.ndarray,
                   offsets=(0.0, 0.30, 0.10)) -> dict:
    """The control that gives Theorem 3 its teeth.

    "The advantage ordering does not vary" means arm a beats arm b by the SAME amount on every
    query, so the argmax is constant in x. Construct exactly that: one shared per-query signal
    plus a fixed per-arm offset. Then U_a(x) = base(x) + b_a - lam*c_a differs across arms by a
    constant, the argmax cannot move, and E[max] = max[E] -- the gap is identically zero.

    Note what does NOT work: taking each query's maximum and adding a margin. Quality is a
    probability mass, so that has to be clipped back into [0, 1], and the clipping manufactures
    exact ties whose argmax is decided by index order rather than by value -- the gap is still
    zero but the ordering appears to vary, which is a property of the clip, not the model.
    """
    base = Qm.mean(axis=1)
    lo, hi = float(base.min()), float(base.max())
    span = hi - lo
    base = 0.6 * (base - lo) / span if span > 1e-12 else np.zeros_like(base)
    forced = base[:, None] + np.asarray(offsets, dtype=float)[None, :]
    return jensen_gap(forced, lam, costs)


# ---------------------------------------------------------------------------------------------
# PROPOSITION 4 -- the estimated router, honestly split
# ---------------------------------------------------------------------------------------------
def split_indices(n: int, frac: float = CAL_FRAC, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(round(frac * n))
    return np.sort(idx[:cut]), np.sort(idx[cut:])


def routing_report(lam: float = 0.10) -> dict:
    """The headline table: in-sample vs held-out router gain over the best fixed arm."""
    c = _corpus()
    Qm = arm_quality(c)
    feats = router_features(c)
    costs = np.array([ARM_COST[a] for a in ARMS])
    cal, test = split_indices(c["n_queries"])

    ridge = select_ridge_cv(feats[cal], Qm[cal], lam, costs)
    W = fit_quality_model(feats[cal], Qm[cal], ridge)
    in_sample = utility(Qm[cal], route(predict_quality(feats[cal], W), lam, costs), lam, costs).mean()
    held_out = utility(Qm[test], route(predict_quality(feats[test], W), lam, costs), lam, costs).mean()

    U_test = Qm[test] - lam * costs[None, :]
    best_fixed_test = float(U_test.mean(axis=0).max())
    oracle_test = float(U_test.max(axis=1).mean())

    U_cal = Qm[cal] - lam * costs[None, :]
    best_fixed_cal = float(U_cal.mean(axis=0).max())

    # Calibration of the predictor the router compares: Proposition 4's load-bearing number.
    Qhat_test = predict_quality(feats[test], W)
    ece = float(expected_calibration_error(Qhat_test.ravel(),
                                           (Qm[test].ravel() > 0.5).astype(int), n_bins=8))

    # Is the gain distinguishable from zero at this n? The evaluation track exists precisely so
    # that a number like +0.04 is reported as an estimate with a standard error rather than as
    # a fact. PAIRED, because both policies are scored on the same queries and pairing cancels
    # the shared per-query difficulty.
    best_arm_idx = int(U_test.mean(axis=0).argmax())
    per_query_router = utility(Qm[test], route(Qhat_test, lam, costs), lam, costs)
    per_query_fixed = U_test[:, best_arm_idx]
    tt = paired_t_test(per_query_router - per_query_fixed)
    return {
        "lam": float(lam),
        "in_sample_gain": float(in_sample - best_fixed_cal),
        "held_out_gain": float(held_out - best_fixed_test),
        "oracle_gain": float(oracle_test - best_fixed_test),
        "held_out_fraction_of_oracle": float(
            (held_out - best_fixed_test) / (oracle_test - best_fixed_test)
        ) if oracle_test - best_fixed_test > 1e-12 else 0.0,
        "best_fixed_arm_test": ARMS[int(U_test.mean(axis=0).argmax())],
        "quality_model_ece": ece,
        "paired_t": float(tt["t"]),
        "paired_p": float(tt["p"]),
        "paired_ci": (float(tt["ci_lo"]), float(tt["ci_hi"])),
        "significant_at_05": bool(tt["p"] < 0.05),
        "ridge_cv": float(ridge),
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
    }


# ---------------------------------------------------------------------------------------------
# Baked constants -- this module owns every number the viz displays.
# ---------------------------------------------------------------------------------------------
def frontier_bundle(lam_fit: float = 0.20) -> dict:
    """All three frontiers on the SAME held-out queries, so they are actually comparable.

    The quality model is fitted on the calibration half and every frontier is then evaluated on
    the test half only. Evaluating the router on the queries that fitted it would inflate
    exactly the curve the topic is about, which would be a strange thing for this page to do.
    """
    c = _corpus()
    Qm = arm_quality(c)
    feats = router_features(c)
    costs = np.array([ARM_COST[a] for a in ARMS])
    cal, test = split_indices(c["n_queries"])
    ridge = select_ridge_cv(feats[cal], Qm[cal], lam_fit, costs)
    W = fit_quality_model(feats[cal], Qm[cal], ridge)
    Qte, Fte = Qm[test], feats[test]
    return {
        "fixed": fixed_arm_points(Qte, costs),
        "fixed_hull": upper_hull(fixed_arm_points(Qte, costs)),
        "router": [(r["cost"], r["quality"])
                   for r in router_frontier(Qte, predict_quality(Fte, W), costs)],
        "oracle": [(r["cost"], r["quality"]) for r in oracle_frontier(Qte, costs)],
        "ridge": float(ridge),
        "n_test": int(len(test)),
    }


LAM_HEADLINE = 0.20           # the operating point the lab opens on
LAM_ROUTER_LOSES = 0.10       # the operating point where a real router LOSES; pinned by a test


def _r(v, n: int = 4) -> float:
    return round(float(v), n)


def viz_constants() -> None:
    c = _corpus()
    Qm = arm_quality(c)
    feats = router_features(c)
    costs = np.array([ARM_COST[a] for a in ARMS])
    klass = c["klass"]
    classes = ("known", "local", "bridge")

    print("// --- baked from adaptive_retrieval_routing.py ---")
    print(f"const ARMS = {list(ARMS)};")
    print(f"const ARM_COST = {{ none: {C_NONE}, single: {C_SINGLE}, iterative: {C_ITER} }};")
    print(f"const CLASSES = {list(classes)};")

    qbc = {k: [_r(v, 3) for v in Qm[klass == k].mean(axis=0)] for k in classes}
    print(f"const QUALITY_BY_CLASS = {qbc};")

    acc = {}
    for k in classes:
        m = klass == k
        acc[k] = [int(sum(int(np.argmax(ARM_FN[a](c, c["Q"][qi]))) == int(c["truth"][qi])
                          for qi in np.where(m)[0])) for a in ARMS]
    print(f"const CORRECT_BY_CLASS = {acc};  // out of {N_PER_CLASS} each")

    fbc = {k: [_r(v, 3) for v in feats[klass == k].mean(axis=0)] for k in classes}
    print(f"const FEATURES_BY_CLASS = {fbc};  // [H(prior), max cos, top2 margin, cross-sector]")

    gaps = [{"lam": _r(l, 3), **{kk: _r(vv, 4) for kk, vv in
                                 ((kk, vv) for kk, vv in jensen_gap(Qm, l, costs).items()
                                  if kk in ("oracle", "best_fixed", "gap"))}}
            for l in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)]
    print(f"const JENSEN = {gaps};")

    print(f"const FIXED_ARM_POINTS = {[[_r(a, 3), _r(b, 4)] for a, b in fixed_arm_points(Qm, costs)]};")
    print(f"const N_QUERIES = {c['n_queries']};  const N_PER_CLASS = {N_PER_CLASS};")
    # The full per-query quality matrix, so the lab can recompute the Jensen gap LIVE for any
    # lambda rather than reading a baked curve -- the strongest form of the viz<->python invariant.
    print("const Q_MATRIX = [" + ", ".join(
        "[" + ",".join(f"{_r(v, 4)}" for v in row) + "]" for row in Qm) + "];")
    print(f"const KLASS_OF_QUERY = {[str(k) for k in klass]};")
    cal_i, test_i = split_indices(c["n_queries"])
    print(f"const TEST_IDX = {[int(i) for i in test_i]};")

    rep_rows = []
    for l in (0.02, 0.05, 0.10, 0.20):
        r = routing_report(l)
        rep_rows.append({"lam": _r(l, 3), "inSample": _r(r["in_sample_gain"]),
                         "heldOut": _r(r["held_out_gain"]), "oracle": _r(r["oracle_gain"]),
                         "frac": _r(r["held_out_fraction_of_oracle"], 3),
                         "p": _r(r["paired_p"], 4), "ece": _r(r["quality_model_ece"], 3)})
    print(f"const ROUTER_REPORT = {rep_rows};")

    fb = frontier_bundle(LAM_HEADLINE)
    def _pts(rows):
        out, seen = [], set()
        for a, b in rows:                      # dedupe: the lambda grid saturates at both ends
            k = (_r(a, 3), _r(b, 4))
            if k not in seen:
                seen.add(k); out.append([k[0], k[1]])
        return out
    print(f"const ROUTER_FRONTIER = {_pts(fb['router'])};   // held-out queries only")
    print(f"const ORACLE_FRONTIER = {_pts(fb['oracle'])};   // held-out queries only")
    print(f"const FIXED_HULL = {_pts(fb['fixed_hull'])};")
    print(f"const N_TEST = {fb['n_test']};")
    print(f"const CHOW_THRESHOLD = {_r(chow_threshold(C_ERR, C_ABS), 4)};  // c_err={C_ERR}, c_abs={C_ABS}")
    print(f"const LAM_HEADLINE = {LAM_HEADLINE};")
    print(f"const LAM_ROUTER_LOSES = {LAM_ROUTER_LOSES};")


# ---------------------------------------------------------------------------------------------
# Verification harness. Each pedagogical claim the topic makes is an assert here; running this
# module as a script IS the topic's regression test.
# ---------------------------------------------------------------------------------------------
def test_chow_collapse_is_exact() -> None:
    """THEOREM 1, the collapse anchor: at K=2 the K-action rule IS the published Chow gate."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.0, 1.0, 400)
    mine = chow_as_two_action(scores, C_ERR, C_ABS)
    theirs = np.asarray(chow_decision(scores, chow_threshold(C_ERR, C_ABS))).astype(int)
    assert mine.shape == theirs.shape, (mine.shape, theirs.shape)
    assert np.array_equal(mine, theirs), int((mine != theirs).sum())
    # and across a grid of cost ratios, not just the published pair
    for c_err, c_abs in ((10.0, 1.0), (5.0, 2.0), (3.0, 2.5), (2.0, 1.0)):
        a = chow_as_two_action(scores, c_err, c_abs)
        b = np.asarray(chow_decision(scores, chow_threshold(c_err, c_abs))).astype(int)
        assert np.array_equal(a, b), (c_err, c_abs, int((a != b).sum()))


def test_arm_single_is_imported_posterior() -> None:
    """Collapse anchor: one retrieval at unit weight IS the imported `answer_posterior`."""
    c = _corpus()
    for qi in (0, 17, 40, 71):
        q = c["Q"][qi]
        d = c["docs"][_top1(c, q)]
        mine = arm_single(c, q)
        theirs = answer_posterior(q, d, c["protos"], tau=TAU_ANS)
        assert np.max(np.abs(mine - theirs)) < 1e-12, np.max(np.abs(mine - theirs))


def test_iterative_collapses_to_single() -> None:
    """Collapse anchor: one hop of the iterative arm is byte-for-byte the single-shot arm."""
    c = _corpus()
    for qi in range(0, c["n_queries"], 7):
        a = arm_iterative(c, c["Q"][qi], max_hops=1)
        b = arm_single(c, c["Q"][qi])
        assert np.max(np.abs(a - b)) < 1e-12, (qi, np.max(np.abs(a - b)))


def test_arm_none_is_query_only() -> None:
    """The no-retrieval arm is the zero-weight degenerate case: the document term vanishes."""
    c = _corpus()
    for qi in (3, 30, 60):
        q = c["Q"][qi]
        got = arm_none(c, q)
        logits = (q @ c["protos"].T) / TAU_ANS
        want = np.exp(logits - logits.max()); want = want / want.sum()
        assert np.max(np.abs(got - want)) < 1e-12
        # and it does not depend on which document is passed, since the weight is zero
        other = answer_posterior_topk(q, c["docs"][5:6], np.array([0.0]), c["protos"], tau=TAU_ANS)
        assert np.max(np.abs(got - other)) < 1e-12


def test_corpus_is_not_vacuous() -> None:
    """THEOREM 3's precondition. If one arm were uniformly best the gap would be zero and the
    entire topic would be about nothing -- so the corpus must genuinely contain a reversal."""
    c = _corpus()
    Qm = arm_quality(c)
    am = Qm.argmax(axis=1)
    assert len(set(am.tolist())) > 1, "no arm reversal: the corpus is vacuous"
    klass = c["klass"]
    # the compositional class must be won by iteration, and the easy class by something cheaper
    assert ARMS[int(Qm[klass == "bridge"].mean(axis=0).argmax())] == "iterative"
    assert ARMS[int(Qm[klass == "known"].mean(axis=0).argmax())] != "iterative"


def test_bridge_needs_iteration() -> None:
    """The mechanism, stated as an answer-level fact: a single retrieval NEVER reaches the
    mentioned company, and iterating does so for a clear majority."""
    c = _corpus()
    m = c["klass"] == "bridge"
    hits = {a: sum(int(np.argmax(ARM_FN[a](c, c["Q"][qi]))) == int(c["truth"][qi])
                   for qi in np.where(m)[0]) for a in ARMS}
    assert hits["single"] == 0, hits
    assert hits["none"] == 0, hits
    assert hits["iterative"] >= int(0.5 * m.sum()), hits


def test_jensen_gap_nonnegative_and_positive_here() -> None:
    """THEOREM 3. The gap is a Jensen gap, so it is >= 0 by construction; on this corpus it is
    strictly positive because the advantage ordering varies."""
    c = _corpus()
    Qm = arm_quality(c)
    costs = np.array([ARM_COST[a] for a in ARMS])
    for lam in LAMBDA_GRID:
        g = jensen_gap(Qm, lam, costs)
        assert g["gap"] >= -1e-12, (lam, g["gap"])
    for lam in (0.02, 0.05, 0.10, 0.20):
        g = jensen_gap(Qm, lam, costs)
        assert g["gap"] > 1e-3, (lam, g["gap"])
        assert g["argmax_varies"], lam


def test_gap_vanishes_when_one_arm_dominates() -> None:
    """The control that gives THEOREM 3 its teeth: force one arm uniformly best and the gap
    goes to zero. Routing is worthless there no matter how good the classifier."""
    c = _corpus()
    Qm = arm_quality(c)
    costs = np.array([ARM_COST[a] for a in ARMS])
    for lam in (0.0, 0.02, 0.05, 0.10):
        g = degenerate_gap(Qm, lam, costs)
        assert abs(g["gap"]) < 1e-12, (lam, g["gap"])
        assert not g["argmax_varies"], (lam, g["arm_win_counts"])
    # and the real corpus is NOT degenerate -- the two must not be confusable
    assert jensen_gap(Qm, 0.05, costs)["gap"] > 1e-3


def test_frontier_ordering() -> None:
    """THEOREM 2. Three nested frontiers: no-feature hull <= achievable router <= oracle."""
    c = _corpus()
    Qm = arm_quality(c)
    feats = router_features(c)
    costs = np.array([ARM_COST[a] for a in ARMS])
    W = fit_quality_model(feats, Qm, 1e-3)
    Qhat = predict_quality(feats, W)
    for r, o in zip(router_frontier(Qm, Qhat, costs), oracle_frontier(Qm, costs)):
        lam = r["lam"]
        u_router = r["quality"] - lam * r["cost"]
        u_oracle = o["quality"] - lam * o["cost"]
        assert u_router <= u_oracle + 1e-12, (lam, u_router, u_oracle)
        best_fixed = float((Qm - lam * costs[None, :]).mean(axis=0).max())
        assert u_oracle >= best_fixed - 1e-12, (lam, u_oracle, best_fixed)


def test_displayed_frontier_is_held_out_and_ordered() -> None:
    """The curves the lab draws must be evaluated on the test half, and the router must sit
    between the no-feature hull and the oracle at matched lambda."""
    c = _corpus()
    fb = frontier_bundle(LAM_HEADLINE)
    _, test = split_indices(c["n_queries"])
    assert fb["n_test"] == len(test) < c["n_queries"]
    costs = np.array([ARM_COST[a] for a in ARMS])
    Qte = arm_quality(c)[test]
    for lam, (rc, rq), (oc, oq) in zip(LAMBDA_GRID, fb["router"], fb["oracle"]):
        assert rq - lam * rc <= oq - lam * oc + 1e-12, (lam, rq - lam * rc, oq - lam * oc)
        best_fixed = float((Qte - lam * costs[None, :]).mean(axis=0).max())
        assert oq - lam * oc >= best_fixed - 1e-12, (lam, oq - lam * oc, best_fixed)


def test_upper_hull_is_concave() -> None:
    """The hull really is the concave envelope: slopes are non-increasing in cost."""
    pts = [(0.0, 0.30), (1.0, 0.70), (2.0, 0.80), (3.0, 0.82), (1.5, 0.40)]
    h = upper_hull(pts)
    assert (1.5, 0.40) not in h, h          # dominated, must be dropped
    slopes = [(h[i + 1][1] - h[i][1]) / (h[i + 1][0] - h[i][0]) for i in range(len(h) - 1)]
    for a, b in zip(slopes, slopes[1:]):
        assert b <= a + 1e-12, slopes


def test_in_sample_overstates_the_gain() -> None:
    """PROPOSITION 4, measured. A router scored on the queries that fitted it looks better
    than it is -- at EVERY operating point on this corpus."""
    for lam in (0.02, 0.05, 0.10, 0.20):
        r = routing_report(lam)
        assert r["in_sample_gain"] >= r["held_out_gain"] - 1e-12, (lam, r)


def test_router_can_lose_to_the_best_fixed_arm() -> None:
    """The honest headline, pinned. A positive Jensen gap does NOT guarantee a real router
    wins: at this operating point the estimation error exceeds what routing buys, and the
    router is beaten by always-retrieve-once on held-out queries."""
    r = routing_report(LAM_ROUTER_LOSES)
    assert r["oracle_gain"] > 0.0, r          # an oracle WOULD win here
    assert r["held_out_gain"] < 0.0, r        # the realizable router does not
    assert r["best_fixed_arm_test"] == "single", r


def test_router_wins_somewhere() -> None:
    """...and it is not uniformly useless either, or there would be nothing to teach."""
    r = routing_report(LAM_HEADLINE)
    assert r["held_out_gain"] > 0.0, r
    assert r["held_out_fraction_of_oracle"] > 0.25, r


def test_gain_is_reported_as_an_estimate() -> None:
    """The evaluation track's thesis, applied to this topic's own headline: a routing gain is a
    sample mean over queries and gets a paired test, not a victory lap. Whatever the verdict is,
    it must be COMPUTED -- and at the operating point where the router loses, the paired test
    must not certify a win."""
    for lam in (0.02, 0.05, 0.10, 0.20):
        r = routing_report(lam)
        assert np.isfinite(r["paired_p"]), (lam, r)
        assert 0.0 <= r["paired_p"] <= 1.0, (lam, r)
    losing = routing_report(LAM_ROUTER_LOSES)
    assert not (losing["significant_at_05"] and losing["held_out_gain"] > 0), losing


def test_cv_never_sees_the_test_split() -> None:
    """The ridge is chosen inside the calibration half. Selecting it on the test half would be
    the same inflation one level up, so this pins the split arithmetic."""
    c = _corpus()
    cal, test = split_indices(c["n_queries"])
    assert len(set(cal.tolist()) & set(test.tolist())) == 0
    assert len(cal) + len(test) == c["n_queries"]
    ridge = select_ridge_cv(router_features(c)[cal], arm_quality(c)[cal], 0.2,
                            np.array([ARM_COST[a] for a in ARMS]))
    assert ridge in RIDGE_GRID, ridge


def test_reproducible() -> None:
    """Bake-only-reproducible: two builds of the corpus and the report agree exactly."""
    global _CORPUS
    a = routing_corpus()
    _CORPUS = None
    b = routing_corpus()
    for k in ("Q", "docs", "protos", "truth"):
        assert np.array_equal(a[k], b[k]), k
    _CORPUS = None
    r1 = routing_report(0.2)
    _CORPUS = None
    r2 = routing_report(0.2)
    assert r1 == r2, (r1, r2)


def test_guards() -> None:
    c = _corpus()
    for bad, fn in (
        (lambda: route(np.zeros((2, 3)), -0.1, np.zeros(3)), "negative lambda"),
        (lambda: fit_quality_model(np.zeros((4, 4)), np.zeros((4, 3)), 0.0), "zero ridge"),
        (lambda: arm_iterative(c, c["Q"][0], max_hops=0), "zero hops"),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError(f"guard did not fire: {fn}")


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed.")

# ---------------------------------------------------------------------------------------------
# Diagnostics -- run this BEFORE trusting any headline. Every tuned constant above was set
# from this sweep, not guessed.
# ---------------------------------------------------------------------------------------------
def _diagnostics() -> None:
    c = _corpus()
    Qm = arm_quality(c)
    feats = router_features(c)
    costs = np.array([ARM_COST[a] for a in ARMS])
    klass = c["klass"]

    print("corpus:", c["n_queries"], "queries,", c["n_docs"], "docs,",
          c["K"], "companies, dim", c["dim"])
    print()
    print("MEAN QUALITY BY CLASS AND ARM  (posterior mass on the gold company)")
    print(f"  {'class':8s} {'none':>8s} {'single':>8s} {'iter':>8s}   best")
    for kname in ("known", "local", "bridge"):
        m = klass == kname
        row = Qm[m].mean(axis=0)
        print(f"  {kname:8s} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}   {ARMS[int(row.argmax())]}")
    print()
    print("PER-QUERY ARM ARGMAX (quality only, no cost) -- does the ordering VARY?")
    am = Qm.argmax(axis=1)
    for kname in ("known", "local", "bridge"):
        m = klass == kname
        print(f"  {kname:8s}", {ARMS[a]: int((am[m] == a).sum()) for a in range(3)})
    print(f"  ordering varies overall: {len(set(am.tolist())) > 1}")
    print()
    print("JENSEN GAP ACROSS LAMBDA")
    for lam in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40):
        g = jensen_gap(Qm, lam, costs)
        print(f"  lam={lam:5.2f}  oracle={g['oracle']:.4f}  best_fixed={g['best_fixed']:.4f} "
              f"({g['best_arm']:9s})  gap={g['gap']:+.4f}  wins={g['arm_win_counts']}")
    print()
    print("FEATURES BY CLASS  (must separate PRE-retrieval or the router has no signal)")
    print(f"  {'class':8s} {'H(prior)':>10s} {'max cos':>10s} {'top2 marg':>10s} {'xsector':>10s}")
    for kname in ("known", "local", "bridge"):
        m = klass == kname
        print(f"  {kname:8s} {feats[m, 0].mean():10.3f} {feats[m, 1].mean():10.3f} "
              f"{feats[m, 2].mean():10.3f} {feats[m, 3].mean():10.3f}")
    print()
    print("ROUTER, IN-SAMPLE vs HELD-OUT")
    for lam in (0.02, 0.05, 0.10, 0.20):
        r = routing_report(lam)
        print(f"  lam={lam:5.2f}  in={r['in_sample_gain']:+.4f}  held_out={r['held_out_gain']:+.4f}"
              f"  oracle={r['oracle_gain']:+.4f}  frac_of_oracle={r['held_out_fraction_of_oracle']:+.2f}"
              f"  p={r['paired_p']:.3f}  ridge={r['ridge_cv']:.0e}"
              f"  best_fixed={r['best_fixed_arm_test']}")
    print()
    print("BRIDGE SANITY -- does the ANSWER come out right, per arm?")
    for kname in ("known", "local", "bridge"):
        m = klass == kname
        hits = {a: 0 for a in ARMS}
        for qi in np.where(m)[0]:
            for a in ARMS:
                if int(np.argmax(ARM_FN[a](c, c["Q"][qi]))) == int(c["truth"][qi]):
                    hits[a] += 1
        print(f"  {kname:8s} top-1 answer correct: "
              + "  ".join(f"{a}={hits[a]}/{int(m.sum())}" for a in ARMS))


if __name__ == "__main__":
    _run_all()
    print()
    viz_constants()
