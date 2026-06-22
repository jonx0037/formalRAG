"""Capstone — The Mathematics of a Production Multimodal Financial RAG System.

The reference implementation for the formalRAG `capstone-multimodal-financial-rag` topic. The
capstone's novelty is COMPOSITION, not a new primitive: it imports every published retrieval leg
(BM25, dense MIPS, late-interaction MaxSim served by PLAID, IVF, PQ, RRF, the over-fetch law) and
composes them into one finance pipeline, then proves the three laws that govern it end to end. One
EXACT statement (a collapse anchor) plus three MEASURED laws, and the honesty is saying which is
which.

The shared corpus is ONE multi-view token corpus (`token_corpus`, the PLAID cloud): each document is
a bag of unit-norm tokens (the late-interaction view), pooled to a single vector (the dense view),
and quantized to a bag of centroid ids (the lexical view, BM25 over centroid-id strings). The exact
end-scorer is brute-force MaxSim (`brute_topk`), so the ground truth is NEUTRAL — no single
candidate-generation leg is the oracle (the oracle is the full MaxSim scan / reranking everything),
which is what lets the legs genuinely complement and lets the collapse anchor reach recall 1.0.

MOVEMENT 1 — cascade recall composition.
  EXACT (idealized): for independent per-stage survival, end-to-end retention R = prod_i r_i, and the
  front end must inject k/prod r_i true neighbors to leave k survivors — ONE negative-binomial
  over-fetch law (imported `measure_overfetch_law`, mean k/r) applied to the COMPOSITE retention
  prod r_i; prod(1/r_i) = 1/prod r_i is an ALGEBRAIC identity, not a multiplication of L physical
  scan-counts. DEMONSTRATED (FKG/Harris): under the positive dependence real queries exhibit (a query
  hard for stage 1 is hard for the rerank too), R_true >= prod r_i — the independent product is a
  conservative LOWER bound; negative association reverses it; equality at independence. We RUN both
  signs (a bivariate-normal copula) before writing the direction. Anchor: all r_i = 1 -> R = 1.

MOVEMENT 2 — hybrid fusion gain.
  EXACT (set algebra): rho_fused <= |union A_L| / |R| (the coverage ceiling), VACUOUS (= 1) whenever
  one leg already covers R, so it bounds gain from above but does NOT explain the domination flip.
  NOT-A-THEOREM: "rho_fused >= max_leg rho_L" is FALSE — the flip needs the false positive to be
  CO-ENDORSED by both legs (under RRF c=60 a lone top vote ~1/61 loses to two mid votes ~2/63), built
  and RUN as `dominated_leg_demo`. DEMONSTRATED: on the complementary multimodal instance the fused
  recall beats every leg, and the gain grows as the legs de-correlate (lower Kendall-tau) — pinned to
  the run, balance/depth-sensitive, not a theorem. Metric is set-coverage recall, not NDCG.

MOVEMENT 3 — end-to-end budget allocation (the capstone theorem).
  Each cascade stage i has a retention curve g_i(c_i) in a COMMON compute unit
  (distance-computations / query). The SEPARABLE model R(c) = prod_i g_i(c_i) is, by the Movement-1
  FKG caveat, a LOWER bound on the true coupled-pipeline recall (we optimize the bound). In log-recall
  the budget-constrained optimum equalizes the marginal LOG-recall per unit cost across active stages:
  g_i'(c_i)/g_i(c_i) = lambda — water-filling. (The equal-marginal also falls out of the un-logged
  product, since d R / d c_i = g_i' * (R / g_i) shares the common scalar R that cancels; the log's job
  is separability + concavity => the KKT point is the GLOBAL max, not the existence of equal-marginal.)
  EXACT only under: common cost unit, separable budget, log-concave g_i on the operating grid, interior
  optimum. DEMONSTRATED (one cloud): water-filling beats uniform and all-in-one. Anchor: full budget
  everywhere -> every g_i -> 1 -> R -> 1.0, the brute exact pipeline.

rigorFlag: synthetic vMF token clouds, not a trained multimodal encoder — every number illustrates a
mechanism, not a benchmark. Multimodality is SYSTEMS-LEVEL (heterogeneous per-modality legs feeding
fusion); the cross-modal ALIGNMENT math, and the generation/grounding/evaluation layer above
retrieval, are named as future work, not derived.

Run:  uv run --with numpy --with scipy --with scikit-learn \\
        python notebooks/capstone-multimodal-financial-rag/capstone_multimodal_financial_rag.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Import the published stack. Add EVERY ancestor's hyphenated dir to the path, then import the
# underscored module (the two-/three-hop pattern from multi_vector_ann_retrieval.py: importing the
# deepest legs transitively needs the grandparents, so list each explicitly).
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "johnson-lindenstrauss",
    "vector-quantization-lloyd-max",
    "product-quantization",
    "ivf-voronoi-partitioning",
    "bm25",
    "rank-fusion-rrf",
    "late-interaction-learned-sparse",
    "multi-vector-ann-retrieval",
    "filtered-incremental-ann",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize                                   # noqa: E402
from bm25 import build_inverted_index, bm25_rank                                 # noqa: E402
from rank_fusion_rrf import rrf_fuse, kendall_tau, combsum                       # noqa: E402
from ivf_voronoi_partitioning import nearest_cells                              # noqa: E402
from filtered_incremental_ann import measure_overfetch_law                       # noqa: E402
from product_quantization import pq_bits                                         # noqa: E402
from late_interaction_learned_sparse import maxsim_matrix                        # noqa: E402
from multi_vector_ann_retrieval import (                                         # noqa: E402
    TOPK, N_DOCS, COLBERT_DIM, COLBERT_TOKENS, REP_NLIST, REP_PQ_M, REP_PQ_KSTAR,
    token_corpus, plaid_index, stage1_candidates, brute_topk,
)

SEED = 0
DEPTH_GRID = (2, 5, 10, 20, 40, 80)       # per-leg scan depth (docs examined / query), sub-saturation
DEMO_R1, DEMO_R2 = 0.6, 0.5               # illustrative middling retentions for the FKG copula demo
                                          # (the MEASURED r1~1 leaves no room for dependence to bite)

BUDGET_DEMO_FRAC = 0.12                    # the Panel-C demo budget, as a fraction of the all-max cost:
                                          # tight enough that the expensive leg is scanned shallow and
                                          # the marginal-recall-per-cost LEVELS across the active stages

# Per-scored-document cost (distance-computations) of each leg — the REAL asymmetry that makes the
# budget allocation non-trivial: a sparse lexical match touches a few centroid-id terms (|WIN_LEX|);
# a dense leg is one pooled dot product (TOKEN_DIM); late-interaction MaxSim scores every query token
# against every window token (m_q * |WIN_LI| * TOKEN_DIM). Scanning the expensive leg deep costs far
# more, so the optimal scan-depth allocation is genuinely non-uniform.
COST_PER_DOC = {"lexical": 4.0, "dense": 16.0, "late_interaction": 256.0}

# Each leg observes a DIFFERENT slice of the document's m_d=8 tokens — the systems-level multimodal
# story: the lexical (text), dense (pooled), and late-interaction legs each see a partial view, and
# fusion recombines them. Disjoint/overlapping windows make the legs genuinely COMPLEMENTARY (each
# recalls true neighbors the others miss) rather than a quality ladder of one scalar truth.
WIN_LEX = (0, 1, 2, 3)                     # lexical leg: first-half tokens, quantized to centroid ids
WIN_DENSE = (4, 5, 6, 7)                   # dense leg: second-half tokens, pooled (disjoint from lexical)
WIN_LI = (1, 3, 4, 6)                      # late-interaction leg: an interleaved partial token view


# =========================================================================== #
# Movement 0 — the shared multi-view corpus and the heterogeneous legs.
# =========================================================================== #

def _truth_sets(truth):
    """The shared MaxSim ground truth as per-query id sets (set overlap is the recall primitive)."""
    return [set(t.tolist()) for t in truth]


def capstone_corpus(seed: int = SEED) -> dict:
    """ONE shared multi-view finance document set, all three views derived from the same tokens so the
    legs rank the SAME documents (RRF needs that) and the ground truth is a single neutral oracle.

      - token view  (docs: N_DOCS x m_d x d, unit-norm)        -> late-interaction / centroid leg
      - dense view  (doc_vecs: N_DOCS x d, pooled + normalized) -> dense MIPS leg
      - lexical view(centroid-id strings)                        -> BM25 leg (exact match on quantized
                                                                    tokens, the lexical exact-token leg)
    Ground truth = brute-force MaxSim top-k (`brute_topk`), the exact end-scorer — NEUTRAL, so no
    candidate-generation leg is the oracle. SYNTHETIC vMF tokens, not a trained encoder (rigorFlag).
    Built once; cache at module scope via `_corpus()`."""
    docs, queries, topic_mu = token_corpus(seed)
    q_vecs = normalize(queries.mean(axis=1))                 # pooled query (shared across legs)
    index = plaid_index(docs, exact=True, seed=seed)         # centroid ids + lossless tokens for anchor
    C = index["C"]
    doc_cids = index["doc_cids"]                             # (N_DOCS, m_d) centroid id per doc token
    # dense view: pool the SECOND-half tokens only (a partial view disjoint from the lexical leg).
    doc_vecs = normalize(docs[:, WIN_DENSE, :].mean(axis=1))
    # lexical view: bag the FIRST-half tokens' centroid ids; the query is the bag of all its query
    # tokens' nearest centroid ids — exact-match lexical retrieval over the quantized vocabulary.
    doc_text = {i: " ".join(f"c{int(c)}" for c in doc_cids[i, WIN_LEX]) for i in range(N_DOCS)}
    q_cids = [[int(nearest_cells(queries[q][j], C, 1)[0]) for j in range(queries.shape[1])]
              for q in range(queries.shape[0])]
    q_text = [" ".join(f"c{c}" for c in cids) for cids in q_cids]
    bm25_index = build_inverted_index(doc_text)
    truth = brute_topk(queries, docs, TOPK)                  # exact MaxSim over ALL m_d tokens
    return {
        "docs": docs, "queries": queries, "topic_mu": topic_mu,
        "doc_vecs": doc_vecs, "q_vecs": q_vecs, "index": index, "C": C,
        "doc_text": doc_text, "q_text": q_text, "bm25_index": bm25_index,
        "truth": truth, "truth_sets": _truth_sets(truth),
        "n_docs": N_DOCS, "n_queries": queries.shape[0],
    }


_CORPUS: dict | None = None


def _corpus(seed: int = SEED) -> dict:
    """Module-scope cache — the corpus (and its PLAID index / MaxSim truth) is built ONCE and reused
    across every test, the <60s budget discipline (rebuilding per test blows it)."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = capstone_corpus(seed)
    return _CORPUS


def leg_dense_ranking(corpus: dict, q: int) -> list[int]:
    """Dense leg: rank all documents by cosine of the pooled query vector against the pooled doc
    vectors (a single-vector MIPS retriever). Returns a full ranking of doc ids."""
    s = corpus["doc_vecs"] @ corpus["q_vecs"][q]
    return np.argsort(-s, kind="stable").tolist()


def leg_centroid_ranking(corpus: dict, q: int) -> list[int]:
    """Late-interaction (token) leg: cheap centroid-MaxSim (the imported approximate MaxSim) over a
    PARTIAL token window (WIN_LI) — a token-level signal distinct from the pooled cosine and the
    lexical match, blind to the tokens outside its window, and NOT the exact-MaxSim oracle (so it is an
    imperfect, complementary retriever of the truth)."""
    Q = corpus["queries"][q]
    Dc = corpus["C"][corpus["index"]["doc_cids"][:, WIN_LI]]      # (n_docs, |WIN_LI|, d) windowed centroids
    s = maxsim_matrix(Q[None, ...], Dc)[0]
    return np.argsort(-s, kind="stable").tolist()


def leg_lexical_ranking(corpus: dict, q: int) -> list[int]:
    """Lexical leg: BM25 (imported) over the centroid-id strings — exact match on the quantized token
    vocabulary, the analogue of a real system's lexical/learned-sparse leg over discrete terms.
    Returns a full ranking of doc ids (BM25 ties and unseen docs ordered by id at the tail)."""
    ranked = bm25_rank(corpus["q_text"][q], corpus["bm25_index"])  # list[(doc_id_str, score)]
    seen, order = set(), []
    for doc_id_str, score in ranked:
        i = int(doc_id_str)
        order.append(i)
        seen.add(i)
    for i in range(corpus["n_docs"]):           # BM25 omits zero-score docs; append by id for a full perm
        if i not in seen:
            order.append(i)
    return order


LEGS = {
    "lexical": leg_lexical_ranking,
    "dense": leg_dense_ranking,
    "late_interaction": leg_centroid_ranking,
}


def leg_rankings(corpus: dict, q: int) -> dict[str, list[int]]:
    """All three heterogeneous leg rankings for one query (the inputs RRF fuses)."""
    return {name: fn(corpus, q) for name, fn in LEGS.items()}


def recall_at_k(ranking, relevant: set, k: int = TOPK) -> float:
    """Set-coverage recall@k: |top_k(ranking) ∩ relevant| / min(k, |relevant|). GUARDS: empty
    relevant -> 0.0; k capped at len(ranking)."""
    if not relevant:
        return 0.0
    kk = min(k, len(ranking))
    hit = len(set(ranking[:kk]) & relevant)
    return hit / min(k, len(relevant))


def mean_leg_recall(corpus: dict, leg_fn, k: int = TOPK) -> float:
    """Mean recall@k of a single leg against the shared MaxSim truth."""
    ts = corpus["truth_sets"]
    return float(np.mean([recall_at_k(leg_fn(corpus, q), ts[q], k) for q in range(corpus["n_queries"])]))


# =========================================================================== #
# Movement 1 — cascade recall composition.
# =========================================================================== #

def cascade_recall(stage_retentions) -> float:
    """The multiplicative end-to-end retention R = prod_i r_i (EXACT under independent per-stage
    survival). GUARDS: every r_i in (0, 1]; empty -> 1.0 (the no-stage identity)."""
    rs = list(stage_retentions)
    if not rs:
        return 1.0
    for r in rs:
        if not (0.0 < r <= 1.0 + 1e-12):
            raise ValueError(f"stage retention must be in (0, 1], got {r}")
    return float(np.prod(rs))


def over_fetch_factor(stage_retentions) -> float:
    """The composed front-end over-fetch 1/prod r_i = 1 / cascade_recall — the ALGEBRAIC reciprocal of
    the composite retention (NOT a product of L physical scan-counts), so that ONE negative-binomial
    k/r law on the composite retention prod r_i delivers k survivors. GUARD: every r_i > 0
    (unguarded -> infinite over-fetch, the empty-denominator pattern)."""
    R = cascade_recall(stage_retentions)
    if R <= 0.0:
        raise ValueError("over_fetch_factor: composite retention must be > 0")
    return 1.0 / R


def measure_stage_retention(found_ids, relevant: set, k: int = TOPK) -> float:
    """The per-stage empirical retention r_i: fraction of the top-k truth a stage's output set keeps.
    Denominator min(k, |relevant|)-guarded."""
    if not relevant:
        return 0.0
    return len(set(found_ids) & relevant) / min(k, len(relevant))


def measure_cascade_stages(corpus: dict, nprobe: int = 8, keep: int = 40, k: int = TOPK) -> dict:
    """Run the real published cascade on the shared corpus and MEASURE each stage's retention against
    the shared MaxSim truth: S1 candidate generation (token-IVF `stage1_candidates`), S2 fuse/prune
    (RRF of the legs cut to `keep`), S3 rerank (exact MaxSim on survivors). Returns the per-stage mean
    retentions and the measured end-to-end recall, so the product law can be checked on the REAL
    stack, not toy constants. Re-derives the imported `brute_topk` truth (the shared-baseline rule)."""
    ts = corpus["truth_sets"]
    docs = corpus["docs"]
    r1s, r2s, r3s, end = [], [], [], []
    for q in range(corpus["n_queries"]):
        rel = ts[q]
        # S1: token-IVF candidate set.
        cand = stage1_candidates(corpus["queries"][q], corpus["index"], nprobe)
        r1s.append(measure_stage_retention(cand, rel, k))
        # S2: fuse the three legs (restricted to candidates), keep the top-`keep`.
        rk = leg_rankings(corpus, q)
        fused = [d for d in rrf_fuse([rk[name] for name in LEGS]) if d in cand][:keep]
        fused_set = set(fused)
        r2s.append(measure_stage_retention(fused_set, rel, k))
        # S3: exact-MaxSim rerank of the survivors to top-k.
        if fused:
            S = maxsim_matrix(corpus["queries"][q][None, ...], docs[np.array(fused)])[0]
            reranked = [fused[i] for i in np.argsort(-S, kind="stable")[:k]]
        else:
            reranked = []
        r3s.append(measure_stage_retention(reranked, rel, k))
        end.append(recall_at_k(reranked, rel, k))
    return {
        "r1": float(np.mean(r1s)), "r2": float(np.mean(r2s)), "r3": float(np.mean(r3s)),
        "product": float(np.mean(r1s) * np.mean(r2s) * np.mean(r3s)),
        "measured_end": float(np.mean(end)),
    }


def simulate_dependent_cascade(r1: float, r2: float, rho: float, n: int = 40000, seed: int = 0) -> dict:
    """A two-stage bivariate-normal copula that GENERATES dependence between per-stage survival so the
    FKG direction can be VERIFIED, not assumed. Stage i survives iff X_i > tau_i with
    tau_i = Phi^{-1}(1 - r_i), (X_1, X_2) ~ N(0, [[1, rho], [rho, 1]]) — so each marginal survival
    probability is exactly r_i and the joint survival is the bivariate-normal orthant probability.
    Returns {rho, R_true, R_indep = r1*r2}. FKG/Harris: positive rho => R_true >= r1*r2."""
    from scipy.stats import norm
    tau1, tau2 = norm.ppf(1 - r1), norm.ppf(1 - r2)
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    X = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    R_true = float(np.mean((X[:, 0] > tau1) & (X[:, 1] > tau2)))
    return {"rho": float(rho), "R_true": R_true, "R_indep": float(r1 * r2)}


def dependence_sweep(r1: float, r2: float, rhos, seed: int = 0) -> list[dict]:
    """Sweep the copula correlation from negative to positive; each row carries the gap
    R_true - R_indep so the two-sided direction (>= for rho > 0, <= for rho < 0, ~0 at rho = 0) is
    measured and baked."""
    rows = []
    for rho in rhos:
        d = simulate_dependent_cascade(r1, r2, rho, seed=seed)
        d["gap"] = d["R_true"] - d["R_indep"]
        rows.append(d)
    return rows


# =========================================================================== #
# Movement 2 — hybrid fusion gain.
# =========================================================================== #

def fusion_recall(corpus: dict, leg_names, k: int = TOPK) -> float:
    """Mean recall@k of the RRF fusion of the named legs against the shared truth."""
    ts = corpus["truth_sets"]
    out = []
    for q in range(corpus["n_queries"]):
        rk = leg_rankings(corpus, q)
        fused = rrf_fuse([rk[name] for name in leg_names])
        out.append(recall_at_k(fused, ts[q], k))
    return float(np.mean(out))


def fusion_gain(corpus: dict, leg_names=tuple(LEGS), k: int = TOPK) -> dict:
    """The signed fusion gain: mean recall of the fused legs minus the best single leg's mean recall.
    Returns the per-leg recalls, the fused recall, and the gain (> 0 demonstrates the gain on this
    corpus; it is NOT a theorem)."""
    per_leg = {name: mean_leg_recall(corpus, LEGS[name], k) for name in leg_names}
    fused = fusion_recall(corpus, leg_names, k)
    best = max(per_leg.values())
    return {"per_leg": per_leg, "fused": fused, "best_leg": best, "gain": fused - best}


def dominated_leg_demo(k: int = 3) -> dict:
    """The headline FLIP, built and RUN: 'fused recall >= best leg' is FALSE. Under RRF c=60 a lone top
    vote (~1/61) is weaker than two mid votes (~2/63), so a false positive flips the ranking only when
    it is CO-ENDORSED by both legs. Relevant set {r1, r2, r3}; the strong leg ranks them 1,2,3 (recall
    1.0) but also ranks the false positive x1 at 4; the noisy leg ranks x1, x2 at the top and buries
    r3. The fused top-3 then drops r3 in favor of x1, so fused recall = 2/3 < 1.0."""
    R = {"r1", "r2", "r3"}
    strong = ["r1", "r2", "r3", "x1", "x2"]          # strong leg: relevant first, then co-endorses x1
    noisy = ["x1", "x2", "r1", "r2", "r3"]           # noisy leg: junk on top, r3 buried at rank 5
    fused = rrf_fuse([strong, noisy])
    rho_strong = recall_at_k(strong, R, k)
    rho_fused = recall_at_k(fused, R, k)
    # control: hold the SAME noisy leg, but drop the junk from the strong leg so it is NO LONGER
    # co-endorsed. The relevant docs stay co-endorsed (two votes) while x1 is single-voted, so r3
    # survives and recall is 1.0 — co-endorsement is the flip's mechanism, not junk-on-top alone.
    strong_clean = ["r1", "r2", "r3"]
    fused_clean = rrf_fuse([strong_clean, noisy])
    return {
        "fused_top3": fused[:k], "rho_strong": rho_strong, "rho_fused": rho_fused,
        "flips": rho_fused < rho_strong,
        "rho_fused_no_coendorse": recall_at_k(fused_clean, R, k),
    }


def _blend_ranking(base: list[int], other: list[int], frac: float) -> list[int]:
    """Morph `other` toward `base`: with frac = 0 return a clone of `base` (no new coverage), with
    frac = 1 return `other`. Interpolates by taking the first `frac` share of positions from `other`
    and filling the rest in `base` order — a one-knob de-correlation axis."""
    n = len(base)
    cut = int(round(frac * n))
    head = other[:cut]
    head_set = set(head)
    tail = [d for d in base if d not in head_set]
    return head + tail


def decorrelation_sweep(corpus: dict, fracs, k: int = TOPK) -> list[dict]:
    """Sweep one leg (dense) from a clone of another (lexical) toward its own ranking and measure the
    fusion gain and the inter-leg Kendall-tau at each stop. Demonstrates that the gain GROWS as the
    legs de-correlate — asserted as an ENDPOINT ordering (clone gain ~ 0 <= disjoint gain), not as
    universal monotonicity (RRF redistributes non-relevant mass non-monotonically)."""
    ts = corpus["truth_sets"]
    rows = []
    for frac in fracs:
        base_r, blend_r, fused_r, taus = [], [], [], []
        for q in range(corpus["n_queries"]):
            rk = leg_rankings(corpus, q)
            base, other = rk["lexical"], rk["dense"]
            blended = _blend_ranking(base, other, frac)
            taus.append(kendall_tau(base, blended))
            base_r.append(recall_at_k(base, ts[q], k))
            blend_r.append(recall_at_k(blended, ts[q], k))
            fused_r.append(recall_at_k(rrf_fuse([base, blended]), ts[q], k))  # clean 2-leg fusion
        # GLOBAL gain: mean fused recall minus the best single leg's MEAN recall (the standard "fused
        # beats the best leg" notion). At frac = 0 the blended leg is a clone of base, so the gain is 0.
        gain = float(np.mean(fused_r)) - max(float(np.mean(base_r)), float(np.mean(blend_r)))
        rows.append({"frac": float(frac), "kendall_tau": float(np.mean(taus)), "gain": gain})
    return rows


# =========================================================================== #
# Movement 3 — end-to-end budget allocation (water-filling).
# =========================================================================== #

def stage_gain_curves(corpus: dict, k: int = TOPK) -> dict:
    """One retention curve per LEG (lexical, dense, late-interaction — the same three legs Panel B
    fuses), each MEASURED against the shared MaxSim truth and costed in the COMMON unit of documents
    examined per query (scan depth d), so a single budget B is well defined. g_leg(d) = fraction of the
    top-k truth in that leg's top-d — a concave, increasing retention.

    The SEPARABLE product R = prod_leg g_leg is, by the Movement-1 FKG caveat, a conservative LOWER
    bound on the true fused-pipeline recall (the real pipeline UNIONs the legs and does at least as
    well); the water-filling optimizes the bound. Returns {leg: [(cost=d, gain), ...]}."""
    ts = corpus["truth_sets"]
    nq = corpus["n_queries"]
    rankings = {name: [LEGS[name](corpus, q) for q in range(nq)] for name in LEGS}
    curves = {}
    for name in LEGS:
        grid = []
        for d in DEPTH_GRID:
            recs = [measure_stage_retention(rankings[name][q][:d], ts[q], k) for q in range(nq)]
            cost = d * COST_PER_DOC[name]                                     # comps = depth * per-doc cost
            grid.append((cost, max(float(np.mean(recs)), 1e-6)))
        curves[name] = grid
    return curves


def end_to_end_recall(alloc: dict, curves: dict) -> float:
    """R = prod_i g_i(c_i): the separable-model recall at an allocation (a dict stage -> grid index)."""
    return float(np.prod([curves[s][alloc[s]][1] for s in curves]))


def _alloc_cost(alloc: dict, curves: dict) -> float:
    return float(sum(curves[s][alloc[s]][0] for s in curves))


def _marginals(alloc: dict, curves: dict) -> dict:
    """Discrete marginal LOG-recall per unit cost for advancing each stage one grid step (None if the
    stage is already at its top step)."""
    out = {}
    for s, grid in curves.items():
        j = alloc[s]
        if j + 1 >= len(grid):
            out[s] = None
            continue
        (c0, g0), (c1, g1) = grid[j], grid[j + 1]
        out[s] = (math.log(g1) - math.log(g0)) / max(c1 - c0, 1e-9)
    return out


def allocate_budget(curves: dict, B: float) -> dict:
    """Discrete water-filling. Start every stage at its cheapest grid step; greedily advance the stage
    with the largest marginal LOG-recall per unit cost while the next step fits the budget B. Under
    log-concave curves this greedy IS the KKT/equal-marginal optimum (the smallest taken marginal >=
    the largest available next marginal). Returns {stage -> grid index}. GUARD: B large enough for the
    cheapest allocation (else raise)."""
    alloc = {s: 0 for s in curves}
    if _alloc_cost(alloc, curves) > B + 1e-9:
        raise ValueError(f"budget {B} below the minimum allocation cost {_alloc_cost(alloc, curves)}")
    while True:
        marg = _marginals(alloc, curves)
        best_s, best_m = None, -math.inf
        for s, m in marg.items():
            if m is None:
                continue
            nxt_cost = curves[s][alloc[s] + 1][0] - curves[s][alloc[s]][0]
            if _alloc_cost(alloc, curves) + nxt_cost <= B + 1e-9 and m > best_m:
                best_s, best_m = s, m
        if best_s is None:
            break
        alloc[best_s] += 1
    return alloc


def demo_budget(curves: dict) -> float:
    """The Panel-C demo budget: a fixed fraction of the cost of scanning every leg to full depth."""
    return BUDGET_DEMO_FRAC * sum(grid[-1][0] for grid in curves.values())


def uniform_alloc(curves: dict, B: float) -> dict:
    """Naive baseline: split the budget evenly, then take the richest grid step each stage can afford
    with its B/S share."""
    share = B / len(curves)
    alloc = {}
    for s, grid in curves.items():
        j = 0
        for idx, (c, _g) in enumerate(grid):
            if c <= share + 1e-9:
                j = idx
        alloc[s] = j
    return alloc


def all_in_one_alloc(curves: dict, B: float, stage: str) -> dict:
    """Baseline: pour the budget into one stage (its richest affordable step), the others at the floor."""
    alloc = {s: 0 for s in curves}
    floor = sum(curves[s][0][0] for s in curves if s != stage)
    j = 0
    for idx, (c, _g) in enumerate(curves[stage]):
        if floor + c <= B + 1e-9:
            j = idx
    alloc[stage] = j
    return alloc


def marginal_log_recall(alloc: dict, curves: dict) -> dict:
    """The realized marginal log-recall g_i'/g_i at the allocation (backward finite difference per
    stage) — the readout that LEVELS OUT (equalizes) at the water-filling optimum."""
    out = {}
    for s, grid in curves.items():
        j = alloc[s]
        if j == 0:
            (c0, g0), (c1, g1) = grid[0], grid[1]
        else:
            (c0, g0), (c1, g1) = grid[j - 1], grid[j]
        out[s] = (math.log(g1) - math.log(g0)) / max(c1 - c0, 1e-9)
    return out


def collapse_anchor(corpus: dict, k: int = TOPK) -> dict:
    """The one EXACT statement: a full-budget pipeline — probe every cell, keep every candidate, rerank
    by exact MaxSim — recovers brute-force retrieval (recall 1.0). Here, reranking ALL documents by
    exact MaxSim equals the truth by construction."""
    ts = corpus["truth_sets"]
    docs = corpus["docs"]
    recs = []
    for q in range(corpus["n_queries"]):
        S = maxsim_matrix(corpus["queries"][q][None, ...], docs)[0]
        ranked = np.argsort(-S, kind="stable")[:k].tolist()
        recs.append(recall_at_k(ranked, ts[q], k))
    return {"recall": float(np.mean(recs))}


def storage_accounting() -> dict:
    """Representative ColBERT/PLAID-scale storage (the late-interaction lab's d=128, 32 tokens,
    K=2^16): a single-vector index is d*32 bits; raw multi-vector is 32x that; PLAID stores per token a
    centroid id (log2 K bits) plus a PQ residual code (imported `pq_bits`), collapsing ~32x back to
    ~1x a single-vector index. Bakes Panel D's storage bars."""
    single = COLBERT_DIM * 32
    raw_multi = COLBERT_TOKENS * COLBERT_DIM * 32
    per_token = math.ceil(math.log2(REP_NLIST)) + pq_bits(REP_PQ_M, REP_PQ_KSTAR)
    plaid = COLBERT_TOKENS * per_token
    return {"single_bits": int(single), "raw_multi_bits": int(raw_multi), "plaid_bits": int(plaid),
            "raw_mult": round(raw_multi / single, 1), "plaid_mult": round(plaid / single, 1)}


# =========================================================================== #
# viz_constants — every number CapstoneLaboratory.tsx bakes to the decimal.
# =========================================================================== #

def viz_constants() -> None:
    corpus = _corpus()
    print("\n=== Panel A — cascade funnel (Movement 1) ===")
    stages = measure_cascade_stages(corpus, nprobe=8, keep=40)
    print("measured stage retentions:", {kk: round(stages[kk], 3) for kk in ("r1", "r2", "r3")})
    print("cascade product R =", round(stages["product"], 3),
          " measured end-to-end recall =", round(stages["measured_end"], 3))
    print("over-fetch factor 1/R =", round(over_fetch_factor([stages["r1"], stages["r2"], stages["r3"]]), 2))
    rhos = (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9)
    sweep = dependence_sweep(DEMO_R1, DEMO_R2, rhos)   # illustrative middling r_i so FKG visibly bites
    print("dependence sweep (rho, R_true, R_indep, gap):")
    for row in sweep:
        print("  ", round(row["rho"], 2), round(row["R_true"], 3), round(row["R_indep"], 3),
              round(row["gap"], 3))
    of = measure_overfetch_law(corpus["doc_vecs"], corpus["q_vecs"], deltas=(0.0, 0.25, 0.5, 0.75),
                               topk=TOPK, trials=20)
    print("over-fetch law (delta, scanned, predicted=k/(1-delta)):",
          [(round(r["delta"], 2), round(r["scanned"], 1), round(r["predicted"], 1)) for r in of])

    print("\n=== Panel B — hybrid fusion gain (Movement 2) ===")
    fg = fusion_gain(corpus)
    print("per-leg recall:", {kk: round(v, 3) for kk, v in fg["per_leg"].items()})
    print("fused recall =", round(fg["fused"], 3), " best leg =", round(fg["best_leg"], 3),
          " gain =", round(fg["gain"], 3))
    flip = dominated_leg_demo()
    print("dominated-leg flip: fused top3 =", flip["fused_top3"],
          " rho_strong =", round(flip["rho_strong"], 3), " rho_fused =", round(flip["rho_fused"], 3),
          " flips =", flip["flips"])
    ds = decorrelation_sweep(corpus, (0.0, 0.25, 0.5, 0.75, 1.0))
    print("de-correlation sweep (frac, kendall_tau, gain):",
          [(r["frac"], round(r["kendall_tau"], 1), round(r["gain"], 3)) for r in ds])

    print("\n=== Panel C — budget water-filling (Movement 3) ===")
    curves = stage_gain_curves(corpus)
    for s, grid in curves.items():
        print(f"  {s}: " + ", ".join(f"({round(c, 1)},{round(g, 3)})" for c, g in grid))
    B = demo_budget(curves)
    wf = allocate_budget(curves, B)
    un = uniform_alloc(curves, B)
    aio = max(end_to_end_recall(all_in_one_alloc(curves, B, s), curves) for s in curves)
    print("budget B =", round(B, 1))
    print("water-filling alloc idx =", wf, " R =", round(end_to_end_recall(wf, curves), 3),
          " cost =", round(_alloc_cost(wf, curves), 1))
    print("uniform alloc idx     =", un, " R =", round(end_to_end_recall(un, curves), 3),
          "  best all-in-one R =", round(aio, 3))
    print("marginal log-recall at WF optimum:", {s: round(m, 5) for s, m in marginal_log_recall(wf, curves).items()})

    print("\n=== Panel D — storage + collapse anchor ===")
    print("storage:", storage_accounting())
    print("collapse anchor recall =", round(collapse_anchor(corpus)["recall"], 3))


# =========================================================================== #
# Verification harness — every pedagogical claim is an assert.
# =========================================================================== #

def test_cascade_recall_is_product() -> None:
    assert abs(cascade_recall([0.8, 0.5, 0.9]) - 0.36) < 1e-12
    assert cascade_recall([]) == 1.0
    assert cascade_recall([0.7]) == 0.7                    # single stage IS one filter (the prereq)
    assert abs(over_fetch_factor([0.8, 0.5, 0.9]) - 1.0 / 0.36) < 1e-9


def test_overfetch_is_reciprocal_identity() -> None:
    # 1/prod r_i is the ALGEBRAIC reciprocal of the composite retention, NOT a product of L scan-counts.
    rs = [0.6, 0.75, 0.9]
    assert abs(over_fetch_factor(rs) - 1.0 / cascade_recall(rs)) < 1e-12
    # one negative-binomial k/r law on the COMPOSITE retention: scanned ~ k / prod r_i.
    corpus = _corpus()
    R = cascade_recall(rs)
    of = measure_overfetch_law(corpus["doc_vecs"], corpus["q_vecs"], deltas=[1.0 - R], topk=TOPK,
                               trials=20)
    assert abs(of[0]["predicted"] - TOPK / R) < 1e-9
    try:
        over_fetch_factor([0.0, 0.5])                       # unguarded denominator must raise
        raise AssertionError("expected ValueError on zero retention")
    except ValueError:
        pass


def test_independence_recovers_product() -> None:
    d = simulate_dependent_cascade(0.7, 0.6, rho=0.0, n=60000, seed=1)
    assert abs(d["R_true"] - d["R_indep"]) < 0.01           # rho = 0 anchor: R_true ~ r1*r2


def test_dependence_direction() -> None:
    # THE headline-flip test, RUN before the prose direction: positive dependence => independent
    # product is a LOWER bound (R_true >= r1*r2); negative dependence reverses it; monotone in rho.
    rows = dependence_sweep(DEMO_R1, DEMO_R2, (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9), seed=2)
    for r in rows:
        if r["rho"] > 1e-9:
            assert r["gap"] >= -0.005, f"positive dependence must give R_true >= product, got {r}"
        if r["rho"] < -1e-9:
            assert r["gap"] <= 0.005, f"negative dependence must give R_true <= product, got {r}"
    gaps = [r["gap"] for r in rows]
    assert all(gaps[i] <= gaps[i + 1] + 1e-3 for i in range(len(gaps) - 1)), "gap must be monotone in rho"
    # the effect must be REAL, not vacuous: at strong positive dependence the lift is substantial.
    assert rows[-1]["gap"] > 0.03, f"FKG lift must be non-trivial at rho=0.9, got {rows[-1]['gap']}"


def test_measured_stages_compose_as_lower_bound() -> None:
    # On the REAL stack stages are positively dependent, so the measured end-to-end recall sits AT or
    # ABOVE the product (a >= direction test, never ==).
    stages = measure_cascade_stages(_corpus(), nprobe=8, keep=40)
    assert 0.0 < stages["r1"] <= 1.0 and 0.0 < stages["r2"] <= 1.0 and 0.0 < stages["r3"] <= 1.0
    assert stages["measured_end"] >= stages["product"] - 0.05


def test_cascade_collapse_anchor() -> None:
    assert abs(collapse_anchor(_corpus())["recall"] - 1.0) < 1e-9


def test_fusion_gain_under_complementarity() -> None:
    fg = fusion_gain(_corpus())
    assert fg["gain"] > 0.0, f"fused recall must beat the best leg on this corpus, got {fg}"


def test_dominated_leg_flip() -> None:
    flip = dominated_leg_demo(k=3)
    assert flip["flips"] and abs(flip["rho_fused"] - 2.0 / 3.0) < 1e-9, flip
    assert flip["rho_strong"] == 1.0
    assert abs(flip["rho_fused_no_coendorse"] - 1.0) < 1e-9   # no co-endorsement => no flip


def test_decorrelation_endpoints() -> None:
    rows = decorrelation_sweep(_corpus(), (0.0, 1.0))
    assert abs(rows[0]["gain"]) < 1e-9                       # clone endpoint: no new coverage, gain 0
    assert rows[-1]["gain"] >= rows[0]["gain"] - 1e-9        # endpoint ordering, not monotonicity


def test_combsum_scale_sensitivity() -> None:
    # RRF is rank-only (scale-invariant); CombSUM on raw scores is swamped by the larger-range leg.
    a = {"d1": 1.0, "d2": 0.9, "d3": 0.1}
    b = {"d1": 1000.0, "d2": 0.0, "d3": 500.0}              # b's scale dominates a raw sum
    assert combsum([a, b])[0] == "d1" and combsum([a, b])[1] == "d3"  # b wins the order


def test_waterfilling_optimality_and_beats_baselines() -> None:
    curves = stage_gain_curves(_corpus())
    B = demo_budget(curves)
    wf = allocate_budget(curves, B)
    # water-filling STRICTLY beats uniform and CRUSHES all-in-one (a starved factor tanks the product).
    assert end_to_end_recall(wf, curves) > end_to_end_recall(uniform_alloc(curves, B), curves) + 0.02
    for s in curves:
        assert end_to_end_recall(wf, curves) > end_to_end_recall(all_in_one_alloc(curves, B, s), curves) + 0.1
    # discrete equal-marginal (KKT): no remaining single step beats the smallest taken marginal.
    taken = list(marginal_log_recall(wf, curves).values())
    affordable = [m for s, m in _marginals(wf, curves).items()
                  if m is not None and _alloc_cost(wf, curves)
                  + (curves[s][wf[s] + 1][0] - curves[s][wf[s]][0]) <= B + 1e-9]
    if affordable:
        assert min(taken) >= max(affordable) - 1e-6
    # the equalization is REAL, not vacuous: the per-stage marginal log-recall LEVELS at the optimum.
    assert max(taken) / max(min(taken), 1e-9) < 3.0, f"marginals should level at the optimum, got {taken}"


def test_logconcave_on_grid() -> None:
    # the global-optimum premise: each stage's log-gain is concave on the operating grid (nonpositive
    # second difference). Flag (not silently assume) if a real curve has a non-concave step.
    curves = stage_gain_curves(_corpus())
    for s, grid in curves.items():
        lg = [math.log(g) for _c, g in grid]
        cs = [grid[i][0] for i in range(len(grid))]
        slopes = [(lg[i + 1] - lg[i]) / (cs[i + 1] - cs[i]) for i in range(len(grid) - 1)]
        non_concave = [i for i in range(len(slopes) - 1) if slopes[i + 1] > slopes[i] + 1e-6]
        assert not non_concave, f"stage {s} log-gain non-concave at steps {non_concave} (restrict grid / flag)"


def test_equal_marginal_also_from_unlogged_product() -> None:
    # kills the overstated 'no equal-marginal without the log': d R / d c_i = g_i' * (R / g_i), so the
    # common scalar R cancels and the stage ranked top by the un-logged marginal equals the one ranked
    # top by g_i'/g_i (the log gives separability/concavity, not the existence of equal-marginal).
    curves = stage_gain_curves(_corpus())
    alloc = {s: 1 for s in curves}
    R = end_to_end_recall(alloc, curves)
    log_marg, prod_marg = {}, {}
    for s, grid in curves.items():
        (c0, g0), (c1, g1) = grid[0], grid[1]
        gp = (g1 - g0) / (c1 - c0)
        log_marg[s] = gp / g0                                # g_i'/g_i
        prod_marg[s] = gp * (R / g0)                         # d R / d c_i, shares the common R
    assert max(log_marg, key=log_marg.get) == max(prod_marg, key=prod_marg.get)


def test_legs_share_one_corpus_and_truth() -> None:
    # the shared-baseline invariant: all legs rank the SAME doc ids, and the truth is the imported
    # brute_topk re-derived on those docs.
    corpus = _corpus()
    rk = leg_rankings(corpus, 0)
    ids = set(range(corpus["n_docs"]))
    for name, order in rk.items():
        assert set(order) == ids, f"leg {name} must rank every doc id once"
    again = brute_topk(corpus["queries"], corpus["docs"], TOPK)
    assert all(set(a.tolist()) == corpus["truth_sets"][q] for q, a in enumerate(again))


def test_storage_collapse() -> None:
    s = storage_accounting()
    assert s["raw_mult"] == 32.0                             # raw multi-vector is 32x a single vector
    assert s["plaid_mult"] < 2.0                             # PLAID collapses it back to ~1x


def _run_all() -> None:
    test_cascade_recall_is_product()
    test_overfetch_is_reciprocal_identity()
    test_independence_recovers_product()
    test_dependence_direction()
    test_measured_stages_compose_as_lower_bound()
    test_cascade_collapse_anchor()
    test_fusion_gain_under_complementarity()
    test_dominated_leg_flip()
    test_decorrelation_endpoints()
    test_combsum_scale_sensitivity()
    test_waterfilling_optimality_and_beats_baselines()
    test_logconcave_on_grid()
    test_equal_marginal_also_from_unlogged_product()
    test_legs_share_one_corpus_and_truth()
    test_storage_collapse()
    print("all capstone tests passed")
    viz_constants()


if __name__ == "__main__":
    _run_all()
