"""Context Selection: submodular coverage, MMR, and determinantal point processes.

The canonical, tested reference for the `context-selection-submodular-dpp` topic -- the showpiece of the
"Information Theory of RAG" layer and the payoff of the arc PMI -> noisy-channel -> retrieval-vs-long-context.
That arc ended on a verdict: answer quality does not improve as you stuff the window, because the extra
passages are redundant (PMI measured it in bits) or are same-sector distractors. The redundancy that
flattened the quality curve IS the diminishing-marginal-returns property of a SUBMODULAR set function. This
module makes that precise and earns the approximation theorem the predecessor was begging for.

Four movements, matching the node title plus a synthesis:

  1. SUBMODULAR COVERAGE. The facility-location objective f(S) = sum_i w_i * max_{j in S} sim(i,j) rewards a
     selection for STANDING IN FOR the whole candidate pool. It is monotone submodular (proved by the
     max-of-similarities argument), so the greedy algorithm that adds the highest-marginal-gain passage at
     each step returns a set worth at least (1 - 1/e) ~ 0.632 of the optimum -- Nemhauser-Wolsey-Fisher 1978.
     We DEMONSTRATE the bound against brute-force OPT on a small pool, and show the marginal gains diminish.
  2. MMR, THE HEURISTIC WITHOUT THE THEOREM. Maximal Marginal Relevance (Carbonell-Goldstein 1998) trades
     relevance against redundancy greedily, but its per-step penalty is taken against the EVOLVING chosen
     set, so it is not greedy on any fixed monotone submodular f and carries no (1 - 1/e) guarantee. We RUN
     a pool where MMR's selected set has strictly lower coverage than the greedy submodular set.
  3. DIVERSITY AS VOLUME (DPP). A determinantal point process puts P(S) propto det(L_S); with L = a Gram
     matrix, det(L_S) is the SQUARED VOLUME of the parallelepiped the selected feature vectors span, so two
     near-duplicates span a near-flat solid of near-zero volume and are almost never drawn together. The
     quality x diversity factorization L = diag(q) S diag(q) gives det(L_S) = (prod q_i^2) det(S_S). MAP is
     NP-hard, but log det(I + L_S) is monotone submodular, so greedy MAP inherits the (1 - 1/e) guarantee.
  4. THE PAYOFF (demonstrated, pinned). On a finance pool where the most query-relevant passages are
     sector-generic near-duplicates (they support the gold company AND its confusable peer equally) and the
     DISAMBIGUATING passage is less query-similar, top-k spends its budget on the redundant cluster and
     leaves the answer split between gold and peer; coverage- and diversity-aware selection reach the
     disambiguator and sharpen the answer. We read each method's answer through the IMPORTED
     answer_posterior_topk (the same von Mises-Fisher softmax the rest of the arc used), so the numbers chain.
     The headline is a comparison ACROSS selection methods at a FIXED budget -- robust across seeds, unlike a
     single interior optimum -- pinned to the observed winner.

The honest hinge (the rigorFlag): the objective we actually want is the answer-information gain
I(A; D_S | Q), but mutual information of a SET with a target is NOT submodular in general (a synergistic XOR
pair violates diminishing returns), so we use facility-location coverage as the clean submodular backbone and
treat info gain as motivating-but-caveated. Submodularity of info gain needs conditional independence
(Krause-Guestrin 2005). Every pedagogical claim is an `assert`; `viz_constants()` prints every number the
React lab mirrors to the decimal.

Run:  uv run --with numpy --with scipy \\
          python notebooks/context-selection-submodular-dpp/context_selection_submodular_dpp.py

NOTE on `chunking-as-segmentation`: it is a conceptual/DAG prerequisite (the candidate passages ARE the
chunks a segmenter would produce) but is deliberately NOT imported -- its `mean_resultant_length(emb, i, j)`
segment-coherence helper has a different signature from the vMF closed form `mean_resultant_length(d, kappa)`
and would only invite confusion. The numeric substrate flows through retrieval-vs-long-context (which
re-exports the rest of the chain) on the one shared finance cloud.

VIZ <-> PYTHON INVARIANT: the constants printed by `viz_constants()` are mirrored to the decimal in
`src/components/viz/ContextSelectionLaboratory.tsx`. Change a number here, re-run, then update the .tsx --
never the reverse. Greedy / brute-force / DPP-MAP / posterior outputs are MODEL OUTPUTS and are BAKED; only
closed forms (the marginal-gain differences, the (1 - 1/e) floor, pixel maps) recompute in TS.
"""

from __future__ import annotations

import math
import pathlib
import sys
from itertools import combinations

import numpy as np
from scipy.special import logsumexp  # never hand-roll softmax/sigmoid (overflow + a Gemini flag)

# --- import the prereq chain (notebook import graph != pedagogical DAG: the frontmatter prereqs are the
#     three graph edges retriever-as-noisy-channel / chunking-as-segmentation / retrieval-vs-long-context;
#     these imports only SOURCE numbers on the one shared finance cloud) ---------------------------------
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "the-retrieval-problem",
    "infonce-contrastive-objective",
    "dense-retrieval-dual-encoders",
    "set-metrics-precision-recall-map-mrr",
    "pmi-retrieval-value",
    "retriever-as-noisy-channel",
    "retrieval-vs-long-context",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf  # noqa: E402
from dense_retrieval_dual_encoders import dual_encoder_score, DPR_SEED, DPR_DIM  # noqa: E402
from pmi_retrieval_value import answer_posterior, entropy, TAU  # noqa: E402
from set_metrics_precision_recall_map_mrr import recall_at_k  # noqa: E402
from retrieval_vs_long_context import rvlc_corpus, answer_posterior_topk  # noqa: E402


# =========================================================================== #
# Constants. The geometry reuses the dense/retrieval-vs-long-context sectors and company prototypes (the same
# vMF finance cloud, seed 7). Each query gets a CANDIDATE POOL deliberately built to exhibit the redundancy
# trap: a tight cluster of sector-generic near-duplicates (ambiguous gold-vs-peer, HIGH query relevance) plus
# a less-relevant disambiguating passage near the gold prototype, plus lower-relevance distractors.
# =========================================================================== #

CSEL_SEED = DPR_SEED                  # 7 -- the shared finance-geometry seed
CSEL_DIM = DPR_DIM                    # 32
N_SECTORS = 4                         # reuse the 4 dense/rvlc sectors
N_GENERIC = 5                         # near-duplicate sector-generic passages per pool (the redundant cluster)
N_DISTRACT = 3                        # low-relevance other-sector distractors per pool
QUERIES_PER_COMPANY = 4              # gold queries per company

KAPPA_GENERIC = 220.0                # tight cluster around the sector mean -> near-duplicates (cosine ~ 0.85)
KAPPA_SPECIFIC = 120.0              # disambiguating passage around the gold prototype (cosine ~ 0.6)
KAPPA_DISTRACT = 45.0               # loose other-sector distractors (clearly lower relevance, near-neutral)
KAPPA_QUERY = 140.0                # query around a center leaning only mildly toward the gold company
QUERY_GOLD_MIX = 0.12              # query center = normalize((1-mix)*sector_mu + mix*proto_gold): mildly gold,
                                   #   so generics out-rank the disambiguator (top-k skips it) yet A is the answer

K_SELECT = 3                        # the selection budget for the payoff / worked pool (3 of 9 candidates)
MMR_LAM = 0.5                       # MMR relevance/diversity knob (1 -> top-k; 0 -> pure diversity)
DPP_RIDGE = 1e-6                    # ridge guarding slogdet on near-duplicate (singular) submatrices
DPP_QSCALE = 0.5                   # quality = exp(DPP_QSCALE * relevance): the L = diag(q) S diag(q) weights

TAU = TAU                           # 0.2 -- imported PMI temperature; the collapse anchors pin against it
PAYOFF_TAU = 0.15                  # the answer-quality readout temperature: sharper than the imported 0.30
                                   #   generation temperature, so the selection task is decisive enough for the
                                   #   disambiguator to matter (the answer is genuinely hard otherwise)

ONE_MINUS_INV_E = 1.0 - 1.0 / math.e   # 0.6321... the Nemhauser-Wolsey-Fisher floor
LOG2 = math.log(2.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax via logsumexp (the prereq idiom; never hand-rolled)."""
    logits = np.asarray(logits, dtype=float)
    return np.exp(logits - logsumexp(logits))


def _sector_means(seed: int = CSEL_SEED, n_sectors: int = N_SECTORS, dim: int = CSEL_DIM) -> np.ndarray:
    """Reproduce the EXACT sector means retrieval-vs-long-context / dense-retrieval used: the first
    default_rng(seed).standard_normal draw, normalized. Keeps every pool on the shared finance cloud."""
    return normalize(np.random.default_rng(seed).standard_normal((n_sectors, dim)))


# =========================================================================== #
#  ★  THE CORE FUNCTIONS -- each encodes a load-bearing design choice. The surrounding machinery, tests, and
#     viz_constants are written against the signatures + return contracts documented here.
# =========================================================================== #

def facility_location_value(S, sim_VxV: np.ndarray, weights: np.ndarray | None = None) -> float:
    """★ The facility-location coverage objective  f(S) = sum_i w_i * max_{j in S} sim(i, j).

    DESIGN CHOICE: the universe over which coverage is summed AND the per-item weights. Summing over the FULL
    candidate pool (every passage must be represented by something selected) is what makes f monotone
    submodular by construction -- the clean backbone where the (1 - 1/e) guarantee is exact. The weights
    w_i >= 0 (Lin-Bilmes) let coverage prefer the query-relevant regions; w_i = 1 recovers plain coverage.

        S          iterable of selected candidate indices (a subset of the pool).
        sim_VxV    (n, n) similarity over the full pool: rows = universe i, cols = candidates j.
        weights    (n,) nonnegative per-item importances; None -> all ones.
    GUARD: empty S -> 0.0 (max over an empty set is 0; every item is uncovered). Returns float.
    """
    sim = np.asarray(sim_VxV, dtype=float)
    n = sim.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    cols = [int(j) for j in S]
    if not cols:
        return 0.0
    best = sim[:, cols].max(axis=1)                 # max_{j in S} sim(i, j) for each universe item i
    return float((w * best).sum())


def greedy_select(value_fn, n: int, k: int, *, lazy: bool = False) -> dict:
    """★ The generic monotone-submodular greedy maximizer -- REUSED by facility location AND the log-det DPP.

    DESIGN CHOICE: one maximizer, two objectives. Facility location and DPP-MAP are both monotone-submodular
    maximizations, so they share this exact loop and differ only in `value_fn`. The marginal-gain trace is
    what the viz plots and what the diminishing-returns test consumes.

        value_fn   callable(list[int]) -> float, the set-value oracle (f(empty) assumed 0).
        n          ground-set size; candidates are range(n).
        k          budget.
        lazy       Minoux lazy-greedy (a max-heap of stale upper-bound gains); on a submodular f it returns
                   the IDENTICAL `selected` to the standard loop with far fewer oracle calls (the speedup
                   anchor). Ties break to the smaller index in BOTH paths.
    Returns {"selected": list[int] (pick order), "gains": list[float], "values": list[float] (running f)}.
    GUARDS: k = min(max(k, 0), n); k <= 0 or n == 0 -> empty trace.
    """
    n = int(n)
    k = min(max(int(k), 0), n)
    selected: list[int] = []
    gains: list[float] = []
    values: list[float] = []
    if k == 0:
        return {"selected": selected, "gains": gains, "values": values}
    cur = 0.0
    if lazy:
        import heapq
        # Upper bound on each element's marginal gain = its gain against the empty set (submodularity makes
        # gains only shrink, so this is a valid, monotone-tightening bound). Tuple (-bound, e): smaller e wins ties.
        heap = [(-(value_fn([e]) - 0.0), e) for e in range(n)]
        heapq.heapify(heap)
        while len(selected) < k and heap:
            while True:
                neg_bound, e = heapq.heappop(heap)
                g = value_fn(selected + [e]) - cur          # the TRUE current marginal gain of e
                if not heap or g >= -heap[0][0] - 1e-12:     # e still beats the next stale bound -> accept
                    selected.append(e)
                    cur += g
                    gains.append(g)
                    values.append(cur)
                    break
                heapq.heappush(heap, (-g, e))                 # otherwise reinsert e with its tightened bound
    else:
        remaining = list(range(n))
        while len(selected) < k and remaining:
            best_e, best_g = None, -math.inf
            for e in remaining:
                g = value_fn(selected + [e]) - cur
                if g > best_g + 1e-15:                        # strict: keep the FIRST (smallest e) on ties
                    best_g, best_e = g, e
            selected.append(best_e)
            remaining.remove(best_e)
            cur += best_g
            gains.append(best_g)
            values.append(cur)
    return {"selected": selected, "gains": gains, "values": values}


def brute_force_opt(value_fn, n: int, k: int) -> dict:
    """★ Exact OPT over all C(n, k) subsets (itertools.combinations) on a SMALL pool -- the only way to
    DEMONSTRATE the (1 - 1/e) guarantee is against a true optimum. Keep n small (the pools here are ~10).
    Returns {"opt_set": list[int], "opt_value": float}. GUARDS: k = min(max(k,0), n); k == 0 -> ({}, 0.0)."""
    n = int(n)
    k = min(max(int(k), 0), n)
    if k == 0:
        return {"opt_set": [], "opt_value": 0.0}
    best_set, best_val = None, -math.inf
    for combo in combinations(range(n), k):
        v = value_fn(list(combo))
        if v > best_val:
            best_val, best_set = v, list(combo)
    return {"opt_set": best_set, "opt_value": float(best_val)}


def mmr_select(rel: np.ndarray, sim_VxV: np.ndarray, k: int, lam: float) -> dict:
    """★ Maximal Marginal Relevance (Carbonell-Goldstein 1998) -- the HEURISTIC with NO (1 - 1/e) guarantee.

    DESIGN CHOICE: MMR optimizes a PER-STEP criterion against the evolving chosen set, not a fixed submodular
    objective, so it carries no approximation guarantee -- the topic's rigorFlag. `lam` is an explicit knob so
    the viz can sweep it and so the no-guarantee test can find a `lam` where MMR's set has strictly lower
    coverage than greedy.

        each step picks   argmax_{j not in S}  [ lam * rel[j] - (1 - lam) * max_{i in S} sim(i, j) ].
        lam = 1 -> pure relevance (== top-k); lam = 0 -> pure diversity.
    Returns {"selected": list[int], "scores": list[float]}. The first pick (S empty -> max over empty sim is
    0) is argmax rel. GUARD: k = min(max(k,0), n).
    """
    rel = np.asarray(rel, dtype=float)
    sim = np.asarray(sim_VxV, dtype=float)
    n = rel.shape[0]
    k = min(max(int(k), 0), n)
    selected: list[int] = []
    scores: list[float] = []
    remaining = list(range(n))
    while len(selected) < k and remaining:
        best_e, best_s = None, -math.inf
        for j in remaining:
            redundancy = max((sim[i, j] for i in selected), default=0.0)
            s = lam * rel[j] - (1.0 - lam) * redundancy
            if s > best_s + 1e-15:
                best_s, best_e = s, j
        selected.append(best_e)
        remaining.remove(best_e)
        scores.append(float(best_s))
    return {"selected": selected, "scores": scores}


def dpp_kernel(quality: np.ndarray, features: np.ndarray) -> np.ndarray:
    """★ The DPP L-ensemble kernel  L = diag(q) S diag(q),  S = features features^T (the Gram / cosine matrix).

    DESIGN CHOICE: build L from an explicit quality x diversity factorization (Kulesza-Taskar) rather than a
    black box, so det(L_S) = (prod_{i in S} q_i^2) det(S_S) is assertable: a relevance product times a
    squared-volume (diversity) term. Using the passage embeddings as `features` ties DPP diversity to the
    same geometry facility location and MMR read.

        quality   (n,) positive relevance weights q_i.
        features  (n, d) UNIT passage vectors.
    Returns (n, n) PSD kernel L.
    """
    q = np.asarray(quality, dtype=float)
    F = np.asarray(features, dtype=float)
    S = F @ F.T
    return (q[:, None] * S) * q[None, :]


def dpp_logdet_value(S, L: np.ndarray, ridge: float = DPP_RIDGE) -> float:
    """The RAW DPP log-probability log det(L_S) (up to the constant normalizer), via slogdet on the principal
    submatrix. GUARD (the near-duplicate trap): near-duplicate passages make L_S singular -> slogdet sign <= 0
    / logdet -inf; a tiny `ridge` on the diagonal keeps it finite for the geometry assertions. Empty S -> 0.0.
    Use `logdet_i_plus` for the MONOTONE submodular surrogate the greedy MAP maximizes."""
    cols = [int(j) for j in S]
    if not cols:
        return 0.0
    sub = np.asarray(L, dtype=float)[np.ix_(cols, cols)]
    sub = sub + ridge * np.eye(sub.shape[0])
    sign, logabsdet = np.linalg.slogdet(sub)
    if sign <= 0:
        return float("-inf")
    return float(logabsdet)


def logdet_i_plus(S, L: np.ndarray) -> float:
    """The MONOTONE submodular surrogate  g(S) = log det(I + L_S)  (Kulesza-Taskar). Unlike raw log det(L_S)
    -- which is submodular but NOT monotone and dives to -inf on singular blocks -- this is finite and
    monotone, so greedy_select can maximize it and inherit the (1 - 1/e) guarantee. Empty S -> 0.0."""
    cols = [int(j) for j in S]
    if not cols:
        return 0.0
    sub = np.asarray(L, dtype=float)[np.ix_(cols, cols)]
    m = sub.shape[0]
    sign, logabsdet = np.linalg.slogdet(np.eye(m) + sub)
    return float(logabsdet)                         # I + PSD is positive definite, so sign is always +1


def dpp_greedy_map(L: np.ndarray, k: int) -> dict:
    """Greedy MAP for the DPP: maximize the monotone submodular g(S) = log det(I + L_S) by REUSING
    greedy_select. (Exact k-DPP MAP is NP-hard; this is the certified greedy surrogate, Chen et al. 2018.)
    k = 1 picks argmax_i log det(1 + L_ii) = argmax_i L_ii = argmax quality_i^2. Returns greedy_select's dict."""
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    return greedy_select(lambda s: logdet_i_plus(s, L), n, k)


# =========================================================================== #
# Machinery written against the contracts above.
# =========================================================================== #

_CORPUS: dict | None = None


def selection_corpus(seed: int = CSEL_SEED, n_sectors: int = N_SECTORS, qpc: int = QUERIES_PER_COMPANY,
                     n_generic: int = N_GENERIC, n_distract: int = N_DISTRACT,
                     gold_mix: float = QUERY_GOLD_MIX, kappa_generic: float = KAPPA_GENERIC,
                     kappa_specific: float = KAPPA_SPECIFIC, kappa_distract: float = KAPPA_DISTRACT,
                     kappa_query: float = KAPPA_QUERY) -> dict:
    """Build the candidate pools on the shared finance cloud. Reuse retrieval-vs-long-context's company
    prototypes (its first two companies per sector are the confusable gold/peer pair) and the exact sector
    means. Each gold query gets its own pool with the redundancy trap baked in.

    Pool composition per query (gold company A, confusable peer B in the same sector s):
      - `n_generic` near-duplicates around the SECTOR MEAN -- ambiguous (they support A and B about equally)
        but the HIGHEST query relevance, so top-k spends its whole budget here.
      - one A-SPECIFIC passage near proto[A] -- the disambiguator: LOWER query relevance (so top-k skips it),
        but it is the only passage that resolves the answer to A.
      - `n_distract` other-sector passages -- clearly low relevance, near-neutral on the A-vs-B answer.
    The query is drawn near a center leaning only MILDLY toward A (gold_mix), so the generics out-rank the
    disambiguator on relevance while A remains the intended, resolvable answer.

    Returns a dict with:
        protos            (K, dim)            the full company prototype set (answers), from rvlc
        sector_mu         (n_sectors, dim)    sector means
        pools             list[dict] per query, each:
            vecs          (m, dim)            pool passage unit vectors
            rel           (m,)                dual-encoder relevance to the query
            facet         (m,)                0 generic, 1 A-specific, 3 distractor
            company_of    (m,)                owning company index (for recall / answer geometry)
            relevant      set[int]            passages that support the gold answer (generics + A-specific)
            a_specific    int                 the disambiguator's pool index
        q                 (n_queries, dim)    query unit vectors
        truth             (n_queries,)        gold company index
        K, dim, n_queries, pool_size, n_sectors
    """
    base = rvlc_corpus(seed=seed)                       # reuse the exact protos / sector structure
    protos = base["protos"]
    sector_of_company = base["sector_of_company"]
    sector_mu = _sector_means(seed, n_sectors, protos.shape[1])
    dim = protos.shape[1]
    # gold/peer pairs: the first two companies of each sector (companies 4*s and 4*s+1 in the rvlc layout).
    pairs = []
    for s in range(n_sectors):
        members = [c for c in range(len(protos)) if sector_of_company[c] == s]
        if len(members) >= 2:
            pairs.append((s, members[0], members[1]))
    pool_size = n_generic + 1 + n_distract
    pools, queries, truth = [], [], []
    qid = 0
    for (s, A, B) in pairs:
        for _ in range(qpc):
            sd = seed + 1000 * qid + 7
            # generic near-duplicates around the sector mean (ambiguous, high relevance)
            generics = sample_vmf(n_generic, sector_mu[s], kappa_generic, seed=sd + 1)
            a_spec = normalize(sample_vmf(1, protos[A], kappa_specific, seed=sd + 2)[0])
            # distractors from OTHER sectors (low relevance, near-neutral on the answer)
            other = [so for so in range(n_sectors) if so != s]
            dvecs = []
            for di in range(n_distract):
                so = other[di % len(other)]
                dvecs.append(normalize(sample_vmf(1, sector_mu[so], kappa_distract, seed=sd + 100 + di)[0]))
            vecs = [normalize(g) for g in np.atleast_2d(generics)] + [a_spec] + dvecs
            vecs = np.array(vecs)
            facet = np.array([0] * n_generic + [1] + [3] * n_distract)
            company_of = np.array([A] * n_generic + [A] + [-1] * n_distract)
            a_idx = n_generic                              # the A-specific disambiguator's pool index
            # query: only mildly toward the gold company A from the sector mean
            q_center = normalize((1.0 - gold_mix) * sector_mu[s] + gold_mix * protos[A])
            q = normalize(sample_vmf(1, q_center, kappa_query, seed=sd + 5)[0])
            rel = dual_encoder_score(q, vecs)              # (m,) inner products
            relevant = set(int(i) for i in range(n_generic)) | {a_idx}
            pools.append({
                "vecs": vecs, "rel": rel, "facet": facet, "company_of": company_of,
                "relevant": relevant, "a_specific": int(a_idx),
            })
            queries.append(q)
            truth.append(int(A))
            qid += 1
    return {
        "protos": protos, "sector_mu": sector_mu,
        "pools": pools, "q": np.array(queries), "truth": np.array(truth),
        "K": int(len(protos)), "dim": int(dim), "n_queries": int(len(pools)),
        "pool_size": int(pool_size), "n_sectors": int(n_sectors),
    }


def _corpus(seed: int = CSEL_SEED) -> dict:
    """Module-scope cache: the pools are built once."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = selection_corpus(seed)
    return _CORPUS


def sim_matrix(vecs: np.ndarray) -> np.ndarray:
    """(m, m) RAW cosine similarity (the Gram matrix) for unit vectors -- the DPP kernel block, whose
    determinant is the squared volume. Can be negative; the DPP geometry wants it that way."""
    F = np.asarray(vecs, dtype=float)
    return F @ F.T


def cov_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    """(m, m) NONNEGATIVE coverage similarity max(0, cos) in [0, 1] -- the coverage/redundancy similarity for
    facility location and MMR (a passage covers only the passages it is positively aligned with). The sim >= 0
    convention is load-bearing: it is exactly what makes facility-location coverage monotone (max over the
    empty set is 0) AND submodular (the (s_e - m)^+ hinge argument needs nonnegative similarities)."""
    return np.clip(sim_matrix(vecs), 0.0, None)


def topk_select(rel: np.ndarray, k: int) -> dict:
    """The baseline: the k highest-relevance candidates. The collapse target for MMR at lam = 1."""
    rel = np.asarray(rel, dtype=float)
    n = rel.shape[0]
    k = min(max(int(k), 0), n)
    order = list(np.argsort(-rel, kind="stable")[:k])
    return {"selected": [int(i) for i in order]}


def _facility_fn(pool: dict):
    """The relevance-weighted facility-location value oracle for one pool: f(S) = sum_i rel_i^+ max sim, over
    the NONNEGATIVE coverage similarity (so the objective is genuinely monotone submodular)."""
    sim = cov_sim_matrix(pool["vecs"])
    w = np.clip(pool["rel"], 0.0, None)                 # nonnegative coverage weights (Lin-Bilmes)
    return lambda S: facility_location_value(S, sim, w)


def _method_selection(pool: dict, k: int, lam: float = MMR_LAM) -> dict:
    """Run all four selectors on one pool, returning each method's selected index list."""
    vecs, rel = pool["vecs"], pool["rel"]
    n = len(vecs)
    cov_sim = cov_sim_matrix(vecs)                       # nonneg [0,1] sim for facility + MMR redundancy
    quality = np.exp(DPP_QSCALE * rel)
    return {
        "topk": topk_select(rel, k)["selected"],
        "mmr": mmr_select(rel, cov_sim, k, lam)["selected"],
        "facility": greedy_select(_facility_fn(pool), n, k)["selected"],
        "dpp": dpp_greedy_map(dpp_kernel(quality, vecs), k)["selected"],
    }


def _answer_quality(corpus: dict, pool: dict, q: np.ndarray, truth: int, selected: list[int],
                    tau: float = PAYOFF_TAU) -> tuple[float, float]:
    """Read one selection's answer through the IMPORTED answer_posterior_topk with an EQUAL attention budget
    over the selected passages (the method's only job was choosing WHICH). Returns (mass on truth, entropy)."""
    if not selected:
        return 0.0, 0.0
    ctx = pool["vecs"][selected]
    w = np.full(len(selected), 1.0 / len(selected))
    post = answer_posterior_topk(q, ctx, w, corpus["protos"], tau)
    return float(post[truth]), entropy(post)


METHODS = ("topk", "mmr", "facility", "dpp")


def selection_payoff(corpus: dict, k: int = K_SELECT, lam: float = MMR_LAM,
                     tau: float = PAYOFF_TAU) -> dict:
    """For each selection method, choose k passages and read the answer through the imported
    answer_posterior_topk, averaged over queries:
        Q       mean posterior mass on the true company
        H       mean answer entropy in bits (lower = sharper)
        cover   mean facility-location coverage fraction f(S)/f(V)
        recall  mean recall_at_k over the gold-supporting set (IMPORTED)
    Returns {method: {Q, H, cover, recall}}.
    """
    acc = {m: {"Q": [], "H": [], "cover": [], "recall": []} for m in METHODS}
    for qi in range(corpus["n_queries"]):
        pool = corpus["pools"][qi]
        q = corpus["q"][qi]
        truth = int(corpus["truth"][qi])
        fval = _facility_fn(pool)
        full = fval(list(range(len(pool["vecs"]))))
        sel = _method_selection(pool, k, lam)
        for m in METHODS:
            s = sel[m]
            Qv, Hv = _answer_quality(corpus, pool, q, truth, s, tau)
            acc[m]["Q"].append(Qv)
            acc[m]["H"].append(Hv)
            acc[m]["cover"].append(fval(s) / full if full > 0 else 0.0)
            acc[m]["recall"].append(recall_at_k(s, pool["relevant"], k))
    return {m: {kk: float(np.mean(v)) for kk, v in d.items()} for m, d in acc.items()}


def payoff_curves(corpus: dict, ks=None, lam: float = MMR_LAM, tau: float = PAYOFF_TAU) -> dict:
    """Q(k), cover(k), H(k) per method across a budget grid -- the Panel-D curves."""
    if ks is None:
        ks = list(range(1, corpus["pool_size"] + 1))
    out = {m: {"Q": [], "cover": [], "H": []} for m in METHODS}
    for k in ks:
        pay = selection_payoff(corpus, k, lam, tau)
        for m in METHODS:
            out[m]["Q"].append(pay[m]["Q"])
            out[m]["cover"].append(pay[m]["cover"])
            out[m]["H"].append(pay[m]["H"])
    return {"ks": list(ks), **out}


def coverage_curve(pool: dict, ks=None) -> dict:
    """Greedy facility-location coverage f(S_k) and its marginal gains over k, for ONE worked pool (Panel A).
    Returns {ks, values, gains}. The gains diminish (the discrete signature of submodularity)."""
    n = len(pool["vecs"])
    if ks is None:
        ks = list(range(1, n + 1))
    g = greedy_select(_facility_fn(pool), n, max(ks))
    values = [g["values"][k - 1] for k in ks]
    gains = [g["gains"][k - 1] for k in ks]
    return {"ks": list(ks), "values": values, "gains": gains}


def nwf_curve(pool: dict, ks=None) -> dict:
    """Greedy value, brute-force OPT, and the (1 - 1/e) OPT floor across budget k for ONE worked pool
    (Panel B). The realized ratio greedy/opt sits in [(1 - 1/e), 1]."""
    fval = _facility_fn(pool)
    n = len(pool["vecs"])
    if ks is None:
        ks = list(range(1, min(n, 6) + 1))             # keep C(n,k) brute force cheap
    greedy_vals, opt_vals = [], []
    g = greedy_select(fval, n, max(ks))
    for k in ks:
        greedy_vals.append(g["values"][k - 1])
        opt_vals.append(brute_force_opt(fval, n, k)["opt_value"])
    return {"ks": list(ks), "greedy": greedy_vals, "opt": opt_vals,
            "floor": [ONE_MINUS_INV_E * o for o in opt_vals],
            "ratio": [gv / ov if ov > 0 else 1.0 for gv, ov in zip(greedy_vals, opt_vals)]}


def nwf_worst_case() -> dict:
    """A small max-coverage instance (0/1 facility-location similarities) where greedy is STRICTLY suboptimal,
    so the (1 - 1/e) guarantee is a visible gap rather than a formality (on the smooth finance pool greedy
    happens to reach OPT). Three candidate sets over 12 facets:
        A = {0..5},  B = {0,1,2,6,7,8},  C = {3,4,5,9,10,11}.
    Greedy grabs the biggest set A (covers 6), then B or C for 9; OPT = B u C = 12. At k = 2 greedy/OPT =
    9/12 = 0.75 -- below OPT, comfortably above the 0.632 floor. Returns greedy/opt/floor/ratio over k = 1..3."""
    u = 12
    sets = {0: {0, 1, 2, 3, 4, 5}, 1: {0, 1, 2, 6, 7, 8}, 2: {3, 4, 5, 9, 10, 11}}
    n = len(sets)
    sim = np.zeros((u, n))                               # rectangular: rows = facets, cols = candidate sets
    for j, members in sets.items():
        for i in members:
            sim[i, j] = 1.0
    fval = lambda S: facility_location_value(S, sim)    # w = ones(u): f(S) = number of facets covered
    ks = [1, 2, 3]
    g = greedy_select(fval, n, max(ks))
    gv = [g["values"][k - 1] for k in ks]
    ov = [brute_force_opt(fval, n, k)["opt_value"] for k in ks]
    return {"ks": ks, "greedy": gv, "opt": ov,
            "floor": [ONE_MINUS_INV_E * o for o in ov],
            "ratio": [a / b if b > 0 else 1.0 for a, b in zip(gv, ov)],
            "set_labels": ["A", "B", "C"], "n_facets": u}


def submodularity_witness(value_fn, n: int, trials: int = 400, seed: int = 0) -> float:
    """Numerically certify diminishing returns: sample A subset B and element e not in B, return the MINIMUM
    over samples of [ Delta(e | A) - Delta(e | B) ]. >= -tol certifies submodular. For facility location this
    is exact (>= 0); reused with an info-gain oracle to show where it can fail.

    Robust for any n >= 1 (a partition sampler, not permutation cuts): pick the test element e first, then
    assign each remaining element to {A and B}, {B only}, or {neither}. This guarantees A subseteq B and
    e not in B with no empty-interval edge case and no A == B bias from the cut positions."""
    if n < 1:
        return 0.0
    rng = np.random.default_rng(seed)
    worst = math.inf
    for _ in range(trials):
        e = int(rng.integers(0, n))
        A, B = [], []
        for i in range(n):
            if i == e:
                continue
            r = int(rng.integers(0, 3))
            if r == 0:                       # in the smaller set A (hence also in B)
                A.append(i)
                B.append(i)
            elif r == 1:                     # in B only (B \ A)
                B.append(i)
        dA = value_fn(A + [e]) - value_fn(A)
        dB = value_fn(B + [e]) - value_fn(B)
        worst = min(worst, dA - dB)
    return float(worst if worst != math.inf else 0.0)


def info_gain_fn(corpus: dict, pool: dict, q: np.ndarray, tau: float = PAYOFF_TAU):
    """An answer-side value oracle: f_IG(S) = H(answer | q) - H(answer | q, S), the bits the selected set
    removes from the answer, built on the IMPORTED answer_posterior_topk. Used to SHOW info-gain
    submodularity holds empirically here -- though it is not guaranteed in general (see the XOR witness)."""
    protos = corpus["protos"]
    # prior = the query-only answer distribution (empty context): softmax(<q, protos>/tau)
    prior = _softmax((q @ protos.T) / tau)
    h_prior = entropy(prior)

    def f(S):
        cols = [int(j) for j in S]
        if not cols:
            return 0.0
        ctx = pool["vecs"][cols]
        w = np.full(len(cols), 1.0 / len(cols))
        post = answer_posterior_topk(q, ctx, w, protos, tau)
        return h_prior - entropy(post)
    return f


def info_gain_xor_witness() -> dict:
    """A constructed counterexample: info gain is NOT submodular in general. Let A = D1 XOR D2 with D1, D2
    independent fair bits. Each observation alone is useless -- I(A; D1) = I(A; D2) = 0 -- but together they
    determine A -- I(A; D1, D2) = 1 bit. So the marginal gain of D2 INCREASES with conditioning:
        Delta(D2 | empty) = I(A; D2)        = 0
        Delta(D2 | {D1})  = I(A; D1,D2) - I(A; D1) = 1 - 0 = 1.
    A submodular gain would have Delta(D2 | {D1}) <= Delta(D2 | empty). It does the opposite (supermodular).
    Returns the two gains and the violation (positive = submodularity violated)."""
    # All quantities are exact in bits for the uniform XOR joint.
    I_A_D2 = 0.0                                   # H(A) - H(A|D2) = 1 - 1 = 0
    I_A_D1 = 0.0
    I_A_D1D2 = 1.0                                 # H(A) - H(A|D1,D2) = 1 - 0 = 1
    delta_empty = I_A_D2
    delta_given_d1 = I_A_D1D2 - I_A_D1
    return {"delta_empty": delta_empty, "delta_given_d1": delta_given_d1,
            "violation": delta_given_d1 - delta_empty}


def worked_pool_index(corpus: dict) -> int:
    """Pick a representative worked pool for the viz: the query whose top-k-vs-dpp answer-quality GAP is
    closest to the median gap (a typical, non-degenerate instance of the payoff)."""
    gaps = []
    for qi in range(corpus["n_queries"]):
        pool = corpus["pools"][qi]
        q = corpus["q"][qi]
        truth = int(corpus["truth"][qi])
        sel = _method_selection(pool, K_SELECT)
        q_top, _ = _answer_quality(corpus, pool, q, truth, sel["topk"])
        q_dpp, _ = _answer_quality(corpus, pool, q, truth, sel["dpp"])
        gaps.append(q_dpp - q_top)
    gaps = np.array(gaps)
    med = float(np.median(gaps))
    return int(np.argmin(np.abs(gaps - med)))


def pool_2d(pool: dict) -> np.ndarray:
    """A deterministic 2D PCA projection of a pool's passage vectors for the geometry panel (Panel C).
    Centers, takes the top-2 right singular vectors, projects. Sign-fixed so the layout is reproducible."""
    F = np.asarray(pool["vecs"], dtype=float)
    Fc = F - F.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
    comp = Vt[:2]
    # sign convention: make the largest-magnitude loading positive, per component
    for r in range(2):
        if comp[r][np.argmax(np.abs(comp[r]))] < 0:
            comp[r] = -comp[r]
    return Fc @ comp.T


# =========================================================================== #
# Tests -- every pedagogical claim is an assertion. Run order: collapse anchors, NWF guarantee, submodularity,
# MMR-no-guarantee, DPP geometry, the payoff headline (run before pinned), then guards.
# =========================================================================== #

def test_greedy_k1_is_argmax() -> None:
    """Greedy at k = 1 picks the single element maximizing f({j}) (for facility location, the best coverer)."""
    pool = _corpus()["pools"][0]
    fval = _facility_fn(pool)
    n = len(pool["vecs"])
    got = greedy_select(fval, n, 1)["selected"][0]
    want = int(np.argmax([fval([j]) for j in range(n)]))
    assert got == want, "greedy@k=1 must equal argmax single-element coverage"


def test_topk_collapse_mmr_lam1() -> None:
    """MMR at lam = 1 (pure relevance) reproduces top-k exactly."""
    pool = _corpus()["pools"][0]
    sim = sim_matrix(pool["vecs"])
    got = mmr_select(pool["rel"], sim, K_SELECT, 1.0)["selected"]
    want = topk_select(pool["rel"], K_SELECT)["selected"]
    assert got == want, "MMR(lam=1) must equal top-k"


def test_dpp_k1_is_max_quality() -> None:
    """DPP greedy MAP at k = 1 picks argmax_i L_ii = argmax quality_i^2 = argmax quality."""
    pool = _corpus()["pools"][0]
    quality = np.exp(DPP_QSCALE * pool["rel"])
    L = dpp_kernel(quality, pool["vecs"])
    got = dpp_greedy_map(L, 1)["selected"][0]
    assert got == int(np.argmax(quality)), "DPP@k=1 must pick the max-quality element"


def test_lazy_equals_standard() -> None:
    """Minoux lazy-greedy returns the IDENTICAL selected set to the standard loop on a submodular f."""
    pool = _corpus()["pools"][0]
    fval = _facility_fn(pool)
    n = len(pool["vecs"])
    a = greedy_select(fval, n, K_SELECT, lazy=False)["selected"]
    b = greedy_select(fval, n, K_SELECT, lazy=True)["selected"]
    assert a == b, f"lazy greedy must match standard: {a} vs {b}"


def test_bridge_degenerate_weight_anchor() -> None:
    """A degenerate single-passage call to the IMPORTED answer_posterior_topk reproduces the imported
    answer_posterior bit-for-bit -- proves every method is read through the SAME readout (the reuse anchor)."""
    c = _corpus()
    pool = c["pools"][0]
    q = c["q"][0]
    d = pool["vecs"][0]
    got = answer_posterior_topk(q, d[None, :], np.array([1.0]), c["protos"], TAU)
    want = answer_posterior(q, d, c["protos"], TAU)
    assert np.allclose(got, want, atol=1e-12), "k=1 weight [1] must equal imported answer_posterior"


def test_nwf_guarantee() -> None:
    """THE centerpiece: on every pool, greedy facility-location coverage is within (1 - 1/e) of brute-force
    OPT, and never exceeds it. Demonstrates Nemhauser-Wolsey-Fisher 1978."""
    c = _corpus()
    for qi in range(c["n_queries"]):
        pool = c["pools"][qi]
        fval = _facility_fn(pool)
        n = len(pool["vecs"])
        for k in (2, 3, 4):
            greedy = greedy_select(fval, n, k)["values"][-1]
            opt = brute_force_opt(fval, n, k)["opt_value"]
            assert greedy >= ONE_MINUS_INV_E * opt - 1e-9, f"NWF floor violated q{qi} k{k}: {greedy} < {ONE_MINUS_INV_E*opt}"
            assert greedy <= opt + 1e-9, f"greedy cannot beat OPT q{qi} k{k}"


def test_nwf_gap_visible() -> None:
    """On the constructed worst-case instance greedy is STRICTLY below OPT at some budget (the (1 - 1/e) bound
    is real, not a formality) yet never below the 0.632 floor -- exactly what Nemhauser-Wolsey-Fisher promise."""
    wc = nwf_worst_case()
    assert min(wc["ratio"]) < 1.0 - 1e-9, "greedy must be strictly suboptimal somewhere (a visible gap)"
    assert all(g >= ONE_MINUS_INV_E * o - 1e-9 for g, o in zip(wc["greedy"], wc["opt"])), "never below the floor"


def test_facility_location_submodular_and_monotone() -> None:
    """Facility location is monotone (greedy values non-decreasing) and submodular (the witness is >= 0)."""
    pool = _corpus()["pools"][0]
    fval = _facility_fn(pool)
    n = len(pool["vecs"])
    vals = greedy_select(fval, n, n)["values"]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), "coverage must be monotone non-decreasing"
    assert submodularity_witness(fval, n) >= -1e-9, "facility location must be submodular (witness >= 0)"


def test_marginal_gains_diminish() -> None:
    """The greedy marginal gains are non-increasing -- the discrete signature of diminishing returns."""
    pool = _corpus()["pools"][0]
    gains = coverage_curve(pool)["gains"]
    assert all(b <= a + 1e-9 for a, b in zip(gains, gains[1:])), "marginal gains must diminish"


def test_info_gain_not_submodular() -> None:
    """The honesty hinge: info gain I(A; D_S | Q) is the objective we WANT, but it is NOT submodular -- not
    even on this finance corpus (the witness goes clearly negative, while facility location's is ~0), and
    provably not in general (the constructed XOR makes the marginal gain INCREASE with conditioning). This is
    exactly why facility-location coverage, not info gain, is the submodular backbone that earns NWF."""
    c = _corpus()
    pool = c["pools"][0]
    f_ig = info_gain_fn(c, pool, c["q"][0])
    n = len(pool["vecs"])
    assert submodularity_witness(_facility_fn(pool), n) >= -1e-9, "facility location IS submodular"
    assert submodularity_witness(f_ig, n) < -0.05, "info gain must VIOLATE submodularity on this corpus"
    xor = info_gain_xor_witness()
    assert xor["violation"] > 0.5, "XOR: info gain is supermodular (not submodular in general)"


def test_mmr_no_guarantee() -> None:
    """MMR carries no (1 - 1/e) guarantee: at a diversity-leaning lam it selects a set with strictly lower
    facility-location coverage than the greedy submodular set on at least one pool."""
    c = _corpus()
    found = False
    for qi in range(c["n_queries"]):
        pool = c["pools"][qi]
        fval = _facility_fn(pool)
        cov_sim = cov_sim_matrix(pool["vecs"])
        n = len(pool["vecs"])
        greedy_cov = fval(greedy_select(fval, n, K_SELECT)["selected"])
        mmr_cov = fval(mmr_select(pool["rel"], cov_sim, K_SELECT, 0.3)["selected"])
        if mmr_cov < greedy_cov - 1e-9:
            found = True
            break
    assert found, "MMR must under-cover greedy on at least one pool (no submodular guarantee)"


def test_dpp_det_is_squared_volume() -> None:
    """det(S_S) equals the squared volume of the parallelepiped the selected unit vectors span = the product
    of the squared singular values of the selection's feature matrix (the Gram-determinant identity)."""
    pool = _corpus()["pools"][0]
    F = pool["vecs"]
    S = sim_matrix(F)
    sel = [0, N_GENERIC, N_GENERIC + 1]                  # a generic + the two specifics: a spread set
    det_gram = np.linalg.det(S[np.ix_(sel, sel)])
    sv = np.linalg.svd(F[sel], compute_uv=False)
    vol_sq = float(np.prod(sv ** 2))
    assert abs(det_gram - vol_sq) < 1e-9, f"det(S_S) must equal squared volume: {det_gram} vs {vol_sq}"


def test_dpp_factorization() -> None:
    """det(L_S) = (prod_{i in S} q_i^2) * det(S_S): the quality x diversity split."""
    pool = _corpus()["pools"][0]
    F = pool["vecs"]
    quality = np.exp(DPP_QSCALE * pool["rel"])
    L = dpp_kernel(quality, F)
    S = sim_matrix(F)
    sel = [0, N_GENERIC, N_GENERIC + 1]
    det_L = np.linalg.det(L[np.ix_(sel, sel)])
    qprod = float(np.prod(quality[sel] ** 2))
    det_S = np.linalg.det(S[np.ix_(sel, sel)])
    assert abs(det_L - qprod * det_S) < 1e-9, "DPP factorization det L_S = (prod q^2) det S_S must hold"


def test_dpp_logdet_surrogate_monotone_submodular() -> None:
    """log det(I + L_S) is monotone (greedy values non-decreasing) and submodular (witness >= -tol), which is
    what lets greedy DPP-MAP reuse greedy_select and inherit the (1 - 1/e) guarantee."""
    pool = _corpus()["pools"][0]
    quality = np.exp(DPP_QSCALE * pool["rel"])
    L = dpp_kernel(quality, pool["vecs"])
    n = L.shape[0]
    vals = greedy_select(lambda s: logdet_i_plus(s, L), n, n)["values"]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), "log det(I+L_S) must be monotone"
    assert submodularity_witness(lambda s: logdet_i_plus(s, L), n) >= -1e-6, "log det(I+L_S) must be submodular"


def test_dpp_repels_duplicates() -> None:
    """A DPP repels redundancy through VOLUME: with unit quality (so log det = pure geometry), appending a
    NEAR-DUPLICATE of an already-selected passage adds far less marginal log-det than appending a genuinely
    different passage -- near-collinear vectors span a near-flat parallelepiped."""
    pool = _corpus()["pools"][0]
    F = pool["vecs"]
    L = dpp_kernel(np.ones(len(F)), F)                  # unit quality -> L = Gram -> log det is pure volume
    sims = sim_matrix(F)
    base = [0]                                          # a generic
    dup = max(range(1, N_GENERIC), key=lambda j: sims[0, j])   # the generic most similar to 0 (near-duplicate)
    far = int(np.argmin(sims[0]))                       # the passage least similar to 0 (most diverse)
    dup_gain = logdet_i_plus(base + [dup], L) - logdet_i_plus(base, L)
    far_gain = logdet_i_plus(base + [far], L) - logdet_i_plus(base, L)
    assert dup_gain < far_gain - 1e-3, "a near-duplicate must add less volume (log-det) than a diverse passage"


def test_dpp_singular_handled() -> None:
    """Near-duplicate passages make L_S singular; the ridge keeps the raw log det finite (no -inf / NaN)."""
    pool = _corpus()["pools"][0]
    F = pool["vecs"].copy()
    F[1] = F[0]                                          # an exact duplicate -> singular Gram block
    quality = np.exp(DPP_QSCALE * pool["rel"])
    L = dpp_kernel(quality, F)
    v = dpp_logdet_value([0, 1], L)
    assert math.isfinite(v), "ridge must keep slogdet finite on a singular submatrix"


def test_diversity_beats_topk_on_coverage() -> None:
    """ROBUST, seed-independent: coverage- and diversity-aware selection cover strictly more of the candidate
    pool than top-k at the same budget. Coverage is the guaranteed contrast (quality is the fragile one)."""
    pay = selection_payoff(_corpus())
    assert pay["facility"]["cover"] > pay["topk"]["cover"] + 1e-6, "facility-location must out-cover top-k"
    assert pay["dpp"]["cover"] > pay["topk"]["cover"] + 1e-6, "DPP must out-cover top-k"


def test_payoff_winner_pinned() -> None:
    """The answer-quality payoff, PINNED to the observed run (the obvious headline can be false on an easy
    corpus, so this asserts the ordering actually observed). At a fixed budget, the submodular coverage and
    DPP selections put substantially more mass on the true company than top-k, and beat the relevance-leaning
    MMR heuristic. Top-k spends the budget on redundant generics; coverage/diversity reach the disambiguator.

    Observed at PAYOFF_TAU=0.15: Q[topk]~0.38 < Q[mmr]~0.43 < Q[facility]~0.54 ~ Q[dpp]~0.55. (Entropy is NOT
    the secondary axis here: diversity raises mass on truth while SPREADING it across more companies, so the
    answer entropy actually rises -- the honest nuance reported in the prose, not asserted as a win.)"""
    pay = selection_payoff(_corpus())
    assert pay["dpp"]["Q"] > pay["topk"]["Q"] + 0.08, "DPP must clearly beat top-k on answer mass"
    assert pay["facility"]["Q"] > pay["topk"]["Q"] + 0.08, "facility-location must clearly beat top-k"
    assert pay["dpp"]["Q"] > pay["mmr"]["Q"] + 0.05, "the submodular/DPP methods beat the MMR heuristic"


def test_guards() -> None:
    """Denominator and degenerate-input guards (the Gemini list)."""
    pool = _corpus()["pools"][0]
    fval = _facility_fn(pool)
    n = len(pool["vecs"])
    assert facility_location_value([], sim_matrix(pool["vecs"])) == 0.0          # empty S
    assert greedy_select(fval, n, 999)["selected"][:0] == []                     # k > n caps (no crash)
    assert len(greedy_select(fval, n, 999)["selected"]) == n                     # capped at n
    assert greedy_select(fval, n, 0)["selected"] == []                          # k <= 0
    assert greedy_select(fval, 0, 3)["selected"] == []                          # empty ground set
    assert recall_at_k([0, 1, 2], set(), 3) == 0.0                               # empty relevant set
    assert dpp_logdet_value([], dpp_kernel(np.ones(n), pool["vecs"])) == 0.0     # empty DPP set
    assert mmr_select(pool["rel"][:1], sim_matrix(pool["vecs"][:1]), 5, 0.5)["selected"] == [0]  # n=1


def _run_all() -> None:
    tests = [
        test_greedy_k1_is_argmax, test_topk_collapse_mmr_lam1, test_dpp_k1_is_max_quality,
        test_lazy_equals_standard, test_bridge_degenerate_weight_anchor,
        test_nwf_guarantee, test_nwf_gap_visible,
        test_facility_location_submodular_and_monotone, test_marginal_gains_diminish,
        test_info_gain_not_submodular, test_mmr_no_guarantee,
        test_dpp_det_is_squared_volume, test_dpp_factorization,
        test_dpp_logdet_surrogate_monotone_submodular, test_dpp_repels_duplicates, test_dpp_singular_handled,
        test_diversity_beats_topk_on_coverage, test_payoff_winner_pinned, test_guards,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")


# =========================================================================== #
# Viz constants -- printed for mirroring into ContextSelectionLaboratory.tsx (cast every scalar).
# =========================================================================== #

def viz_constants() -> None:
    c = _corpus()
    wi = worked_pool_index(c)
    pool = c["pools"][wi]
    q = c["q"][wi]
    truth = int(c["truth"][wi])
    cov = coverage_curve(pool)
    nwf = nwf_worst_case()                               # the visible-gap instance for the guarantee panel
    nwf_fin = nwf_curve(pool)                            # the finance pool (greedy reaches OPT: ratio ~ 1)
    pc = payoff_curves(c)
    pay = selection_payoff(c)
    sel = _method_selection(pool, K_SELECT)
    pts = pool_2d(pool)
    quality = np.exp(DPP_QSCALE * pool["rel"])
    L = dpp_kernel(quality, pool["vecs"])
    S = sim_matrix(pool["vecs"])
    spread = [0, N_GENERIC, N_GENERIC + 1]
    det_L = float(np.linalg.det(L[np.ix_(spread, spread)]))
    qprod = float(np.prod(quality[spread] ** 2))
    det_S = float(np.linalg.det(S[np.ix_(spread, spread)]))
    xor = info_gain_xor_witness()

    def r4(v):
        return round(float(v), 4)

    print("// --- baked from viz_constants() ---")
    print(f"const POOL_SIZE = {c['pool_size']};")
    print(f"const N_GENERIC = {N_GENERIC};")
    print(f"const N_DISTRACT = {N_DISTRACT};")
    print(f"const N_QUERIES = {c['n_queries']};")
    print(f"const K_SELECT = {K_SELECT};")
    print(f"const MMR_LAM = {MMR_LAM};")
    print(f"const ONE_MINUS_INV_E = {r4(ONE_MINUS_INV_E)};   // the NWF floor")
    # Panel A: coverage value + marginal gains (greedy facility location on the worked pool)
    print(f"const COV_KS = {cov['ks']};")
    print(f"const COVERAGE = {[r4(v) for v in cov['values']]};   // f(S_k), rising and concave")
    print(f"const COV_GAINS = {[r4(v) for v in cov['gains']]};   // marginal gains, diminishing")
    # Panel B: NWF guarantee on the constructed worst-case instance (greedy < OPT at k=2, both above the floor)
    print(f"const NWF_KS = {nwf['ks']};")
    print(f"const NWF_GREEDY = {[r4(v) for v in nwf['greedy']]};")
    print(f"const NWF_OPT = {[r4(v) for v in nwf['opt']]};")
    print(f"const NWF_FLOOR = {[r4(v) for v in nwf['floor']]};   // (1-1/e)*OPT")
    print(f"const NWF_RATIO = {[r4(v) for v in nwf['ratio']]};   // greedy/opt, all >= 0.6321")
    print(f"const NWF_FINANCE_RATIO = {[r4(v) for v in nwf_fin['ratio']]};   // finance pool: greedy reaches OPT")
    # Panel C: geometry -- 2D points, facets, per-method selections on the worked pool
    print(f"const POINTS_2D = {[[r4(x) for x in row] for row in pts]};")
    print(f"const FACET = {[int(f) for f in pool['facet']]};   // 0 generic, 1 A-specific (disambiguator), 3 distractor")
    print(f"const REL = {[r4(v) for v in pool['rel']]};")
    print(f"const SEL_TOPK = {[int(i) for i in sel['topk']]};")
    print(f"const SEL_MMR = {[int(i) for i in sel['mmr']]};")
    print(f"const SEL_FACILITY = {[int(i) for i in sel['facility']]};")
    print(f"const SEL_DPP = {[int(i) for i in sel['dpp']]};")
    print(f"const WORKED_TRUTH = {truth};")
    # DPP factorization (worked spread set)
    print(f"const DPP_DET_LS = {round(det_L, 6)};")
    print(f"const DPP_QUALITY_SQ = {round(qprod, 6)};")
    print(f"const DPP_VOLUME_SQ = {round(det_S, 6)};")
    # Panel D: payoff curves + the headline bars at K_SELECT
    print(f"const PAYOFF_KS = {pc['ks']};")
    for m in METHODS:
        print(f"const QK_{m.upper()} = {[r4(v) for v in pc[m]['Q']]};")
    for m in METHODS:
        print(f"const COVERK_{m.upper()} = {[r4(v) for v in pc[m]['cover']]};")
    for m in METHODS:
        r = pay[m]
        print(f"const PAYOFF_{m.upper()} = {{Q: {r4(r['Q'])}, H: {r4(r['H'])}, "
              f"cover: {r4(r['cover'])}, recall: {r4(r['recall'])}}};")
    # The honesty witnesses: facility location IS submodular (~0), info gain is NOT (clearly negative);
    # the constructed XOR is the clean general proof (a synergistic pair makes the marginal gain increase).
    fac_w = submodularity_witness(_facility_fn(pool), len(pool["vecs"]))
    ig_w = submodularity_witness(info_gain_fn(c, pool, q), len(pool["vecs"]))
    print(f"const FACILITY_WITNESS = {r4(fac_w)};   // >= 0: submodular")
    print(f"const INFOGAIN_WITNESS = {r4(ig_w)};   // < 0: NOT submodular on this corpus")
    print(f"const XOR_DELTA_EMPTY = {r4(xor['delta_empty'])};")
    print(f"const XOR_DELTA_GIVEN_D1 = {r4(xor['delta_given_d1'])};")


if __name__ == "__main__":
    print("context_selection_submodular_dpp: running tests")
    _run_all()
    print("\nviz_constants (mirror into the .tsx):")
    viz_constants()
    print("\nall checks passed.")
