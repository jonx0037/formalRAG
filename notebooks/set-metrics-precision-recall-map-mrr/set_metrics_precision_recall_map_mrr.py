"""Set metrics — precision, recall, the PR curve, Average Precision, MAP, and MRR as ESTIMATORS.

The reference implementation for the formalRAG `set-metrics-precision-recall-map-mrr` topic. This is
the DAG root of the evaluation layer: the published retrieval stack MEASURES recall@k everywhere (the
capstone calls `recall_at_k` a dozen times) but never DEFINES it. Here the whole set-metric family is
defined, and — the load-bearing move — reframed as ESTIMATORS with variance, which sets up significance
testing downstream.

It grounds the capstone by scoring the SAME three legs over the SAME shared token corpus and the SAME
neutral `brute_topk` MaxSim ground truth the capstone used. We IMPORT the published legs (BM25, the dense
dual encoder, late-interaction MaxSim) and the corpus primitives — we never reimplement them, and we do
NOT import the capstone itself (it is a DOWNSTREAM synthesis node), re-deriving the three leg rankings here.

TWO QREL REGIMES off the one MaxSim truth (a metric should be demonstrated in the regime it was designed for):
  - set regime (`qrels_set`): R_q = the top-k MaxSim neighbours, |R| = TOPK = 10. Drives precision, recall,
    the PR curve, AP, MAP, and the estimator/CI machinery.
  - known-item regime (`qrels_ki`): R_q = {the single top-1 MaxSim doc}, |R| = 1. Drives MRR and the clean
    identity MAP == MRR (with one relevant doc at rank r, AP = 1/r = RR), the topic's collapse anchor.

TWO DELIBERATE CONVENTIONS, flagged so a reader comparing this to the capstone is not confused:
  1. FULL-VIEW legs. The capstone gives each leg a DISJOINT partial token window (WIN_LEX/WIN_DENSE/WIN_LI)
     to manufacture the complementarity its fusion story needs. Here every leg is an honest FULL-view
     retriever of the same truth, so AP/recall gaps reflect genuine quality, not engineered blind spots.
  2. RECALL DENOMINATOR = |R|, not min(k, |R|). The capstone divides by min(k, |R|) (cascade-retention
     semantics: "of k slots, how many true neighbours survived"). The textbook recall divides by |R| — the
     form required for recall@N = 1 and for AP to equal the area under the PR curve. The two COINCIDE at the
     capstone's only operating point (k = |R| = 10), so this topic still defines exactly the recall@10 the
     capstone reports; the |R| form is what generalizes the curve.

rigorFlag (primary): these are sample means over a FINITE, FIXED query set drawn from one synthetic vMF
corpus. MAP and MRR are estimators with standard error std/sqrt(n); the per-query AP are bounded, skewed,
and (across legs) paired, so the normal-approximation CI is itself approximate (hence the bootstrap
cross-check). An observed MAP gap within a couple of standard errors is NOT evidence of a better system —
the proper paired significance test is deferred to the downstream topic; this one only sets up the variance.
Secondary: "AP" is convention-ambiguous (raw area under the sawtooth == the relevant-ranks mean vs the
interpolated monotone envelope, always >= raw); synthetic vMF tokens and a MaxSim oracle mean "relevance"
is "top-k under the exact end-scorer", not human judgment.

Run:  uv run --with numpy --with scipy \\
        python notebooks/set-metrics-precision-recall-map-mrr/set_metrics_precision_recall_map_mrr.py
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Import the published stack. Add EVERY ancestor's hyphenated dir to the path (the multi-vector subtree's
# transitive closure — importing token_corpus pulls late-interaction/IVF/PQ/VQ/vMF at import time), then
# import the underscored modules. We IMPORT the legs; we never reimplement them, and never import the
# capstone (downstream).
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "vector-quantization-lloyd-max",
    "product-quantization",
    "ivf-voronoi-partitioning",
    "bm25",
    "dense-retrieval-dual-encoders",
    "late-interaction-learned-sparse",
    "multi-vector-ann-retrieval",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize                       # noqa: E402
from bm25 import build_inverted_index, bm25_rank                     # noqa: E402
from dense_retrieval_dual_encoders import dual_encoder_score         # noqa: E402
from ivf_voronoi_partitioning import nearest_cells                   # noqa: E402
from late_interaction_learned_sparse import maxsim_matrix            # noqa: E402
from multi_vector_ann_retrieval import (                             # noqa: E402
    TOPK, N_DOCS, token_corpus, plaid_index, brute_topk,
)

SEED = 0
ELEVEN_LEVELS = tuple(round(0.1 * i, 1) for i in range(11))   # 0.0, 0.1, ..., 1.0 (TREC 11-point grid)


# =========================================================================== #
# Movement 0 — the shared corpus, the neutral truth, two qrel regimes, three full-view legs.
# =========================================================================== #

def set_metrics_corpus(seed: int = SEED) -> dict:
    """ONE shared token corpus (the capstone/PLAID cloud), the neutral exact-MaxSim ground truth, and the
    SAME three heterogeneous legs the capstone fuses — re-derived here (NOT imported from the downstream
    capstone) as FULL-view retrievers. Two qrel regimes off the one truth (set: |R| = TOPK; known-item:
    |R| = 1). SYNTHETIC vMF tokens, not a trained encoder (rigorFlag)."""
    docs, queries, _ = token_corpus(seed)
    q_vecs = normalize(queries.mean(axis=1))                 # pooled query vector (dense leg)
    doc_vecs = normalize(docs.mean(axis=1))                  # pooled doc vector  (dense leg) — FULL pool
    index = plaid_index(docs, exact=True, seed=seed)
    C, doc_cids = index["C"], index["doc_cids"]
    # lexical view: bag every doc token's centroid id; query is its tokens' nearest centroid ids.
    doc_text = {i: " ".join(f"c{int(c)}" for c in doc_cids[i]) for i in range(N_DOCS)}
    q_cids = [[int(nearest_cells(queries[q][j], C, 1)[0]) for j in range(queries.shape[1])]
              for q in range(queries.shape[0])]
    q_text = [" ".join(f"c{c}" for c in cids) for cids in q_cids]
    bm25_index = build_inverted_index(doc_text)
    truth = brute_topk(queries, docs, TOPK)                  # exact MaxSim over ALL tokens — neutral oracle
    qrels_set = [set(int(d) for d in t.tolist()) for t in truth]
    qrels_ki = [{int(t[0])} for t in truth]                  # known-item: the single top-1 MaxSim doc
    corpus = {
        "docs": docs, "queries": queries, "doc_vecs": doc_vecs, "q_vecs": q_vecs,
        "C": C, "doc_cids": doc_cids, "q_text": q_text, "bm25_index": bm25_index,
        "truth": truth, "qrels_set": qrels_set, "qrels_ki": qrels_ki,
        "n_docs": N_DOCS, "n_queries": queries.shape[0], "r_size": TOPK,
    }
    corpus["rankings"] = {name: [fn(corpus, q) for q in range(corpus["n_queries"])]
                          for name, fn in LEGS.items()}      # cache full rankings once (the <60s budget)
    return corpus


_CORPUS: dict | None = None


def _corpus(seed: int = SEED) -> dict:
    """Module-scope cache: the corpus, PLAID index, MaxSim truth, and all leg rankings are built ONCE."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = set_metrics_corpus(seed)
    return _CORPUS


def leg_dense_ranking(corpus: dict, q: int) -> list[int]:
    """Dense MIPS leg: rank all docs by the IMPORTED dual-encoder score of the pooled query against the
    pooled doc vectors (P @ z_q). A full permutation of doc ids."""
    s = dual_encoder_score(corpus["q_vecs"][q], corpus["doc_vecs"])
    return np.argsort(-s, kind="stable").tolist()


def leg_late_interaction_ranking(corpus: dict, q: int) -> list[int]:
    """Late-interaction leg, served PLAID-style: the IMPORTED maxsim_matrix over the CENTROID-substituted
    full document C[doc_cids] (token-level signal, strong but imperfect — the residual gap keeps it short
    of the exact-MaxSim oracle, so it is a genuine, complementary retriever, not the truth itself)."""
    Q = corpus["queries"][q]
    Dc = corpus["C"][corpus["doc_cids"]]                      # (n_docs, m_d, d): centroid-substituted docs
    s = maxsim_matrix(Q[None, ...], Dc)[0]
    return np.argsort(-s, kind="stable").tolist()


def leg_lexical_ranking(corpus: dict, q: int) -> list[int]:
    """Lexical BM25 leg: the IMPORTED bm25_rank over the centroid-id strings (exact match on the quantized
    token vocabulary). Zero-score docs appended by id for a full permutation (the capstone's tail-fill)."""
    ranked = bm25_rank(corpus["q_text"][q], corpus["bm25_index"])
    seen: set[int] = set()
    order: list[int] = []
    for doc_id_str, _score in ranked:
        i = int(doc_id_str)
        order.append(i)
        seen.add(i)
    for i in range(corpus["n_docs"]):
        if i not in seen:
            order.append(i)
    return order


LEGS = {
    "lexical": leg_lexical_ranking,
    "dense": leg_dense_ranking,
    "late_interaction": leg_late_interaction_ranking,
}
LEG_NAMES = tuple(LEGS)


def _ranking(corpus: dict, leg: str, q: int) -> list[int]:
    """The cached full ranking of `leg` on query `q`."""
    return corpus["rankings"][leg][q]


# =========================================================================== #
# Movement 1 — set metrics at a cutoff: precision, recall, F1.
# =========================================================================== #

def precision_at_k(ranking, relevant: set, k: int) -> float:
    """P@k = |top_k(ranking) ∩ R| / k — the PURITY of the cutoff. Fixed-k denominator (a short ranking is
    penalized). GUARDS: k <= 0 or empty R -> 0.0."""
    if k <= 0 or not relevant:
        return 0.0
    hit = len(set(ranking[:k]) & relevant)
    return hit / k


def recall_at_k(ranking, relevant: set, k: int) -> float:
    """R@k = |top_k(ranking) ∩ R| / |R| — the COVERAGE of the relevant set. Denominator is |R| (textbook
    recall), NOT min(k, |R|) (the capstone's cascade-retention convention); they coincide at k = |R|.
    GUARDS: empty R or k <= 0 -> 0.0."""
    if not relevant or k <= 0:
        return 0.0
    hit = len(set(ranking[:k]) & relevant)
    return hit / len(relevant)


def f1_at_k(ranking, relevant: set, k: int) -> float:
    """The harmonic mean F1@k = 2PR/(P+R). GUARD: P + R = 0 -> 0.0."""
    p = precision_at_k(ranking, relevant, k)
    r = recall_at_k(ranking, relevant, k)
    return 0.0 if (p + r) <= 0.0 else 2.0 * p * r / (p + r)


def relevant_ranks(ranking, relevant: set) -> list[int]:
    """The sorted 1-indexed positions of the relevant docs within the ranking (the raw material the viz
    bakes and from which every metric is recomputed). Walking in rank order, the result is already sorted."""
    rel = set(relevant)
    return [i for i, d in enumerate(ranking, start=1) if d in rel]


# =========================================================================== #
# Movement 2 — the PR curve and Average Precision (the area-under-the-curve theorem).
# =========================================================================== #

def pr_curve(ranking, relevant: set) -> list[tuple[float, float]]:
    """The raw PR curve as the sequence of (recall, precision) corners, one per relevant hit, prefixed with
    (0, 1). At the i-th relevant hit (at rank pos_i) recall = i/|R| and precision = i/pos_i. These corners
    are exactly the points whose recall-weighted precision is AP."""
    m = len(relevant)
    if m == 0:
        return [(0.0, 1.0)]
    pts = [(0.0, 1.0)]
    for i, pos in enumerate(relevant_ranks(ranking, relevant), start=1):
        pts.append((i / m, i / pos))
    return pts


def average_precision(ranking, relevant: set) -> float:
    """AP = (1/|R|) * Σ_{relevant ranks pos_i} (i / pos_i): the mean of precision-at-each-relevant-rank.
    The denominator is |R| (NOT the number found) — unfound relevant docs contribute 0, which is exactly
    what makes AP the area under the raw PR curve. GUARD: empty R -> 0.0."""
    m = len(relevant)
    if m == 0:
        return 0.0
    return sum(i / pos for i, pos in enumerate(relevant_ranks(ranking, relevant), start=1)) / m


def ap_via_pr_area(ranking, relevant: set) -> float:
    """AP recomputed as the area under the raw PR curve: Σ_i (R_i - R_{i-1}) * P_i over the pr_curve
    corners. Each relevant hit advances recall by exactly 1/|R|, so this equals `average_precision`
    identically — the theorem made executable."""
    pts = pr_curve(ranking, relevant)
    return sum((pts[i][0] - pts[i - 1][0]) * pts[i][1] for i in range(1, len(pts)))


def _hit_precisions_recalls(ranking, relevant: set):
    """(P_i, R_i) at the relevant hits as parallel lists — the shared basis for the interpolation forms."""
    m = len(relevant)
    ranks = relevant_ranks(ranking, relevant)
    P = [i / pos for i, pos in enumerate(ranks, start=1)]
    R = [i / m for i in range(1, len(ranks) + 1)]
    return P, R


def interpolated_ap(ranking, relevant: set) -> float:
    """The area under the INTERPOLATED PR curve — the monotone non-increasing upper envelope
    p_interp(r) = max{P_j : R_j >= r}. Because the envelope dominates the raw precision pointwise, this is
    >= raw AP (an exact inequality), the precise sense of 'interpolation inflates AP'. GUARD: empty R -> 0."""
    if not relevant:
        return 0.0
    P, _R = _hit_precisions_recalls(ranking, relevant)
    m = len(relevant)
    # On each recall sub-interval (R_{k-1}, R_k] of width 1/m, the envelope is max(P[k:]) (points with
    # recall >= the level), so the area is (1/m) * Σ_k max(P[k:]).
    return sum(max(P[k:]) for k in range(len(P))) / m if P else 0.0


def interpolated_pr_11pt(ranking, relevant: set, levels=ELEVEN_LEVELS) -> list[float]:
    """The classic TREC 11-point interpolated precision: at each recall level t, max{P_j : R_j >= t}
    (0 if none). Returns the 11 envelope values — a non-increasing sequence the viz/prose use; their mean
    is the 11-point interpolated average precision."""
    P, R = _hit_precisions_recalls(ranking, relevant)
    out = []
    for t in levels:
        cand = [P[j] for j in range(len(P)) if R[j] >= t - 1e-12]
        out.append(max(cand) if cand else 0.0)
    return out


def reciprocal_rank(ranking, relevant: set) -> float:
    """RR = 1 / (rank of the first relevant doc), 1-indexed; 0.0 if no relevant doc appears."""
    rel = set(relevant)
    for k, d in enumerate(ranking, start=1):
        if d in rel:
            return 1.0 / k
    return 0.0


# =========================================================================== #
# Movement 3 — aggregation across queries: MAP and MRR.
# =========================================================================== #

def per_query_ap(corpus: dict, leg: str, qrels_key: str = "qrels_set") -> np.ndarray:
    """The per-query AP — the i.i.d.-across-queries SAMPLES whose mean is MAP and whose spread is the
    estimator's variance."""
    qr = corpus[qrels_key]
    return np.array([average_precision(_ranking(corpus, leg, q), qr[q])
                     for q in range(corpus["n_queries"])])


def per_query_rr(corpus: dict, leg: str, qrels_key: str = "qrels_ki") -> np.ndarray:
    """The per-query reciprocal rank — the samples whose mean is MRR (known-item qrels by default)."""
    qr = corpus[qrels_key]
    return np.array([reciprocal_rank(_ranking(corpus, leg, q), qr[q])
                     for q in range(corpus["n_queries"])])


def mean_average_precision(corpus: dict, leg: str, qrels_key: str = "qrels_set") -> float:
    """MAP = (1/Q) Σ_q AP_q — the sample mean of per-query AP."""
    return float(np.mean(per_query_ap(corpus, leg, qrels_key)))


def mean_reciprocal_rank(corpus: dict, leg: str, qrels_key: str = "qrels_ki") -> float:
    """MRR = (1/Q) Σ_q RR_q — the sample mean of per-query reciprocal rank."""
    return float(np.mean(per_query_rr(corpus, leg, qrels_key)))


# =========================================================================== #
# Movement 4 — metrics as ESTIMATORS: standard error, the 1/sqrt(n) law, and "is it significant?".
# =========================================================================== #

def metric_summary(samples: np.ndarray) -> dict:
    """A metric-as-estimator summary: the point estimate (sample mean), the sample std (ddof=1), the
    standard error std/sqrt(n), and the 95% normal-approx CI mean ± 1.96*SE. The per-query AP (or RR) are
    the samples; MAP (MRR) is their mean. GUARD: n <= 1 -> se = 0."""
    s = np.asarray(samples, dtype=float)
    n = s.size
    mean = float(np.mean(s)) if n else 0.0
    sd = float(np.std(s, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    return {"n": int(n), "mean": mean, "std": sd, "se": se,
            "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se}


def se_scaling_experiment(corpus: dict, leg: str, sub_sizes=(5, 10, 20, 40, 80),
                          trials: int = 1000, qrels_key: str = "qrels_set", seed: int = 0) -> list[dict]:
    """The SE ~ 1/sqrt(n) demonstration. Treat the Q observed per-query AP as a sample from an (infinite)
    query population; resample WITH replacement at each size n (one rng stream — the infinite-population
    proxy, no finite-population correction) and report the empirical std of those MAP estimates against the
    prediction pop_std/sqrt(n). Returns rows {n, empirical_se, theory_se, se_root_n}; se_root_n =
    empirical_se*sqrt(n) should be ~constant (the scaling law made explicit)."""
    ap = per_query_ap(corpus, leg, qrels_key)
    pop_std = float(np.std(ap, ddof=1))
    rng = np.random.default_rng(seed)
    rows = []
    for n in sub_sizes:
        means = np.array([np.mean(rng.choice(ap, size=n, replace=True)) for _ in range(trials)])
        emp = float(np.std(means, ddof=1))
        rows.append({"n": int(n), "empirical_se": emp, "theory_se": pop_std / math.sqrt(n),
                     "se_root_n": emp * math.sqrt(n)})
    return rows


def bootstrap_map_se(corpus: dict, leg: str, b: int = 2000, qrels_key: str = "qrels_set",
                     seed: int = 0) -> float:
    """Bootstrap SE of MAP: resample the Q per-query AP with replacement b times and take the std of the
    bootstrap MAPs — a second, assumption-light estimate of the SE that cross-checks the analytic
    std/sqrt(n)."""
    ap = per_query_ap(corpus, leg, qrels_key)
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(rng.choice(ap, size=ap.size, replace=True)) for _ in range(b)])
    return float(np.std(boots, ddof=1))


def _ci_overlap(a: dict, b: dict) -> bool:
    """Do two 95% CIs overlap? (a crude 'not yet distinguishable' read — the proper PAIRED test is downstream)."""
    return a["ci_lo"] <= b["ci_hi"] and b["ci_lo"] <= a["ci_hi"]


def two_leg_map_comparison(corpus: dict, leg_a: str = "dense", leg_b: str = "late_interaction",
                           n: int | None = None, qrels_key: str = "qrels_set") -> dict:
    """The motivating 'is the difference real?' setup the downstream significance topic resolves. At a
    query count n (None = all Q), report each leg's metric_summary, the observed MAP gap, and whether the
    two 95% CIs OVERLAP. The per-query AP are PAIRED (same queries), so the naive overlapping-CI test is
    conservative; the paired difference's SE is what the next topic uses."""
    ap_a, ap_b = per_query_ap(corpus, leg_a, qrels_key), per_query_ap(corpus, leg_b, qrels_key)
    if n is not None:
        ap_a, ap_b = ap_a[:n], ap_b[:n]
    sa, sb = metric_summary(ap_a), metric_summary(ap_b)
    return {"leg_a": leg_a, "leg_b": leg_b, "n": int(sa["n"]),
            "summary_a": sa, "summary_b": sb, "gap": sa["mean"] - sb["mean"],
            "ci_overlap": _ci_overlap(sa, sb)}


def _ap_mean_std(corpus: dict, leg: str, qrels_key: str = "qrels_set") -> tuple[float, float]:
    ap = per_query_ap(corpus, leg, qrels_key)
    return float(np.mean(ap)), float(np.std(ap, ddof=1))


def projected_ci(mean: float, std: float, n: int) -> tuple[float, float]:
    """The 95% CI projected to a query count n with the OBSERVED per-query std held fixed:
    mean ± 1.96*std/sqrt(n). The closed form Panel D recomputes in TS from the baked (mean, std)."""
    h = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    return mean - h, mean + h


def projected_overlap(corpus: dict, leg_a: str, leg_b: str, n: int, qrels_key: str = "qrels_set") -> bool:
    """Do the two legs' PROJECTED 95% CIs overlap at query count n? (Closed-form from full-sample mean/std,
    so the viz reproduces it exactly — unlike a query-order-dependent subsample.)"""
    la, ua = projected_ci(*_ap_mean_std(corpus, leg_a, qrels_key), n)
    lb, ub = projected_ci(*_ap_mean_std(corpus, leg_b, qrels_key), n)
    return la <= ub and lb <= ua


def projected_separation_n(corpus: dict, leg_a: str, leg_b: str, qrels_key: str = "qrels_set",
                           n_max: int | None = None) -> int | None:
    """Smallest n at which the two legs' projected 95% CIs no longer overlap — the standard 'how many
    i.i.d. queries to resolve this MAP gap' read. None if they never separate within n_max."""
    cap = n_max or corpus["n_queries"]
    for n in range(2, cap + 1):
        if not projected_overlap(corpus, leg_a, leg_b, n, qrels_key):
            return n
    return None


# =========================================================================== #
# The headline FLIP — metric choice changes the verdict (built and RUN, never assumed).
# =========================================================================== #

def leg_metric_table(corpus: dict) -> dict:
    """Per-leg MAP (set regime) and MRR (known-item regime) — the table whose two argmaxes are compared."""
    return {leg: {"map": mean_average_precision(corpus, leg, "qrels_set"),
                  "mrr": mean_reciprocal_rank(corpus, leg, "qrels_ki")} for leg in LEG_NAMES}


def metric_choice_flips_verdict(corpus: dict) -> dict:
    """Does the verdict depend on the metric on THIS corpus? MAP rewards ranking the whole relevant set
    high; MRR rewards one answer at rank 1 — so two systems can rank one way by MAP and the OTHER way by
    MRR. We detect a PAIRWISE REVERSAL (a pair (a, b) with MAP_a > MAP_b but MRR_a < MRR_b), which is the
    honest "metric choice changes the verdict" claim even when one leg happens to top both. Returns the
    table, the overall argmax winners, the reversing pair (if any), and `flips`. RUN before any prose."""
    tbl = leg_metric_table(corpus)
    map_winner = max(tbl, key=lambda lg: tbl[lg]["map"])
    mrr_winner = max(tbl, key=lambda lg: tbl[lg]["mrr"])
    reversal = None
    for a, b in itertools.combinations(LEG_NAMES, 2):
        d_map = tbl[a]["map"] - tbl[b]["map"]
        d_mrr = tbl[a]["mrr"] - tbl[b]["mrr"]
        if d_map * d_mrr < 0:                            # opposite signs -> the verdict reverses between a, b
            reversal = {"pair": (a, b),
                        "map_winner": a if d_map > 0 else b,
                        "mrr_winner": a if d_mrr > 0 else b}
            break
    return {"table": tbl, "map_winner": map_winner, "mrr_winner": mrr_winner,
            "reversal": reversal, "flips": reversal is not None}


def constructed_metric_flip() -> dict:
    """A minimal, hand-built 3-query / 2-leg toy proving the verdict CAN flip, used only as a fallback if no
    natural flip appears on the corpus. Leg 'precise' puts one relevant doc at rank 1 every time (MRR = 1)
    but buries the rest of the relevant set (low MAP); leg 'thorough' is the reverse. Set regime |R| = 3
    per query; known-item is each query's first relevant doc."""
    # Three queries; relevant set per query is {a, b, c}_q, ranking is a list of doc tags.
    R = [{"a1", "b1", "c1"}, {"a2", "b2", "c2"}, {"a3", "b3", "c3"}]
    ki = [{"a1"}, {"a2"}, {"a3"}]                                 # the known-item answer per query
    # 'precise': answer at rank 1, the other two relevant docs buried at the very bottom.
    precise = [["a1", "x", "y", "z", "b1", "c1"], ["a2", "x", "y", "z", "b2", "c2"],
               ["a3", "x", "y", "z", "b3", "c3"]]
    # 'thorough': the whole relevant set in the top 3 but the known-item answer only at rank 3.
    thorough = [["b1", "c1", "a1", "x", "y", "z"], ["b2", "c2", "a2", "x", "y", "z"],
                ["b3", "c3", "a3", "x", "y", "z"]]
    map_p = float(np.mean([average_precision(precise[q], R[q]) for q in range(3)]))
    map_t = float(np.mean([average_precision(thorough[q], R[q]) for q in range(3)]))
    mrr_p = float(np.mean([reciprocal_rank(precise[q], ki[q]) for q in range(3)]))
    mrr_t = float(np.mean([reciprocal_rank(thorough[q], ki[q]) for q in range(3)]))
    return {"map": {"precise": map_p, "thorough": map_t},
            "mrr": {"precise": mrr_p, "thorough": mrr_t},
            "map_winner": "thorough" if map_t > map_p else "precise",
            "mrr_winner": "precise" if mrr_p > mrr_t else "thorough",
            "flips": ("thorough" if map_t > map_p else "precise") != ("precise" if mrr_p > mrr_t else "thorough")}


# =========================================================================== #
# Worked-example query selection (a mid-range query so the PR sawtooth is illustrative, not trivial).
# =========================================================================== #

def pick_worked_query(corpus: dict, leg: str = "dense") -> int:
    """Choose ONE worked query for Panels A & B: the query whose `leg` AP is closest to that leg's MEDIAN
    AP — a representative, non-degenerate sawtooth (not AP = 0 or 1). Deterministic."""
    ap = per_query_ap(corpus, leg, "qrels_set")
    return int(np.argmin(np.abs(ap - float(np.median(ap)))))


# =========================================================================== #
# viz_constants — every number SetMetricsLaboratory.tsx mirrors to the decimal.
# =========================================================================== #

def viz_constants() -> None:
    corpus = _corpus()
    nq = corpus["n_queries"]
    wq = pick_worked_query(corpus, "dense")

    print("\n=== shared constants ===")
    print(f"N_DOCS = {corpus['n_docs']}  R_SIZE = {corpus['r_size']}  N_QUERIES = {nq}  WORKED_Q = {wq}")

    print("\n=== Panel A & B — worked query's relevant ranks per leg (TS recomputes P/R/PR/AP) ===")
    for leg in LEG_NAMES:
        rr = relevant_ranks(_ranking(corpus, leg, wq), corpus["qrels_set"][wq])
        ap = average_precision(_ranking(corpus, leg, wq), corpus["qrels_set"][wq])
        iap = interpolated_ap(_ranking(corpus, leg, wq), corpus["qrels_set"][wq])
        print(f"  {leg:16s} REL_RANKS={rr}  AP={round(ap, 4)}  interp_AP={round(iap, 4)}")

    print("\n=== Panel C — per-query AP (set) and RR (known-item), and MAP/MRR ===")
    for leg in LEG_NAMES:
        ap = per_query_ap(corpus, leg, "qrels_set")
        rrr = per_query_rr(corpus, leg, "qrels_ki")
        print(f"  {leg:16s} MAP={round(float(np.mean(ap)), 4)}  MRR={round(float(np.mean(rrr)), 4)}"
              f"  AP_std={round(float(np.std(ap, ddof=1)), 4)}")
        print(f"      PER_Q_AP={[round(float(v), 3) for v in ap]}")
        print(f"      PER_Q_RR={[round(float(v), 3) for v in rrr]}")

    print("\n=== Panel C anchor — MAP == MRR under known-item (|R| = 1) ===")
    for leg in LEG_NAMES:
        m_ap = mean_average_precision(corpus, leg, "qrels_ki")     # AP in the KNOWN-ITEM regime
        m_rr = mean_reciprocal_rank(corpus, leg, "qrels_ki")
        print(f"  {leg:16s} MAP_ki={round(m_ap, 6)}  MRR={round(m_rr, 6)}  equal={abs(m_ap - m_rr) < 1e-12}")

    print("\n=== Panel D — metrics as estimators (CI half-width 1.96*pop_std/sqrt(n), recomputed in TS) ===")
    for leg in LEG_NAMES:
        ap = per_query_ap(corpus, leg, "qrels_set")
        s = metric_summary(ap)
        print(f"  {leg:16s} SUMMARY={{'mean': {round(s['mean'], 4)}, 'std': {round(s['std'], 4)}, "
              f"'se': {round(s['se'], 4)}, 'ci': [{round(s['ci_lo'], 4)}, {round(s['ci_hi'], 4)}]}}")
    print("  SE_SCALING (dense; n, empirical_se, theory_se, se*sqrt(n)):")
    for row in se_scaling_experiment(corpus, "dense"):
        print(f"    n={row['n']:3d}  emp={round(row['empirical_se'], 4)}  thy={round(row['theory_se'], 4)}"
              f"  se*sqrt(n)={round(row['se_root_n'], 4)}")
    tl = two_leg_map_comparison(corpus, "dense", "late_interaction")
    sep = projected_separation_n(corpus, "dense", "late_interaction")
    print(f"  TWO_LEG dense-vs-late_interaction: gap@n40={round(tl['gap'], 4)} overlap@n40={tl['ci_overlap']}"
          f"  proj_overlap@n5={projected_overlap(corpus, 'dense', 'late_interaction', 5)}"
          f"  proj_separate_at_n={sep}")

    print("\n=== headline flip — metric choice changes the verdict (pairwise reversal) ===")
    flip = metric_choice_flips_verdict(corpus)
    print(f"  table={{ {', '.join(f'{lg}: (MAP {round(v['map'], 3)}, MRR {round(v['mrr'], 3)})' for lg, v in flip['table'].items())} }}")
    print(f"  argmax: MAP winner={flip['map_winner']}  MRR winner={flip['mrr_winner']}")
    print(f"  REVERSAL={flip['reversal']}  flips={flip['flips']}")
    cf = constructed_metric_flip()
    print(f"  constructed extreme: MAP winner={cf['map_winner']} MRR winner={cf['mrr_winner']} flips={cf['flips']}")


# =========================================================================== #
# Verification harness — every pedagogical claim is an assert.
# =========================================================================== #

def test_precision_recall_bounds() -> None:
    c = _corpus()
    for leg in LEG_NAMES:
        for q in range(0, c["n_queries"], 5):
            r = _ranking(c, leg, q)
            for k in (1, 3, 5, 10, 25, c["n_docs"]):
                p = precision_at_k(r, c["qrels_set"][q], k)
                rec = recall_at_k(r, c["qrels_set"][q], k)
                assert 0.0 <= p <= 1.0 and 0.0 <= rec <= 1.0, (leg, q, k, p, rec)


def test_recall_monotone_precision_not() -> None:
    # Recall is non-decreasing in k; precision is NOT — the headline asymmetry, on a hand ranking.
    R = {0, 3, 4}                                       # |R| = 3 scattered
    ranking = [0, 1, 2, 3, 4, 5, 6, 7]
    recs = [recall_at_k(ranking, R, k) for k in range(1, len(ranking) + 1)]
    precs = [precision_at_k(ranking, R, k) for k in range(1, len(ranking) + 1)]
    assert all(recs[i + 1] >= recs[i] - 1e-12 for i in range(len(recs) - 1)), recs
    assert any(precs[i + 1] < precs[i] - 1e-12 for i in range(len(precs) - 1)), precs  # a precision DIP exists
    # the same on a real leg.
    c = _corpus()
    real = [recall_at_k(_ranking(c, "dense", 0), c["qrels_set"][0], k) for k in range(1, c["n_docs"] + 1)]
    assert all(real[i + 1] >= real[i] - 1e-12 for i in range(len(real) - 1))


def test_recall_at_N_is_one() -> None:
    c = _corpus()
    for leg in LEG_NAMES:
        for q in range(0, c["n_queries"], 7):
            assert abs(recall_at_k(_ranking(c, leg, q), c["qrels_set"][q], c["n_docs"]) - 1.0) < 1e-12


def test_ap_equals_area_under_pr() -> None:
    # THE central identity: AP (relevant-ranks mean) == AP via the PR-curve area (Riemann sum), <1e-12,
    # on a hand ranking AND every real leg/query.
    R = {0, 2, 5}
    ranking = [0, 1, 2, 3, 4, 5, 6]
    assert abs(average_precision(ranking, R) - ap_via_pr_area(ranking, R)) < 1e-12
    c = _corpus()
    for leg in LEG_NAMES:
        for q in range(0, c["n_queries"], 3):
            r = _ranking(c, leg, q)
            assert abs(average_precision(r, c["qrels_set"][q]) - ap_via_pr_area(r, c["qrels_set"][q])) < 1e-12


def test_interpolated_ge_raw_and_monotone() -> None:
    c = _corpus()
    for leg in LEG_NAMES:
        for q in range(0, c["n_queries"], 4):
            r = _ranking(c, leg, q)
            raw = average_precision(r, c["qrels_set"][q])
            interp = interpolated_ap(r, c["qrels_set"][q])
            assert interp >= raw - 1e-12, (leg, q, raw, interp)          # envelope area >= raw area
            env = interpolated_pr_11pt(r, c["qrels_set"][q])
            assert all(env[i + 1] <= env[i] + 1e-12 for i in range(len(env) - 1)), env  # non-increasing


def test_perfect_ranking_ap_one() -> None:
    R = {0, 1, 2}
    ranking = [0, 1, 2, 3, 4, 5]                          # all relevant first
    assert abs(average_precision(ranking, R) - 1.0) < 1e-12
    assert abs(precision_at_k(ranking, R, 3) - 1.0) < 1e-12
    assert abs(recall_at_k(ranking, R, 3) - 1.0) < 1e-12
    assert abs(interpolated_ap(ranking, R) - 1.0) < 1e-12


def test_map_is_mean_of_ap() -> None:
    c = _corpus()
    for leg in LEG_NAMES:
        assert abs(mean_average_precision(c, leg) - float(np.mean(per_query_ap(c, leg)))) < 1e-12


def test_mrr_is_mean_of_rr() -> None:
    c = _corpus()
    for leg in LEG_NAMES:
        assert abs(mean_reciprocal_rank(c, leg) - float(np.mean(per_query_rr(c, leg)))) < 1e-12


def test_rr_is_inverse_first_hit() -> None:
    assert reciprocal_rank([9, 8, 0, 7], {0}) == 1.0 / 3.0
    assert reciprocal_rank([0, 1, 2], {0, 2}) == 1.0
    assert reciprocal_rank([1, 2, 3], {0}) == 0.0          # never appears -> 0


def test_map_equals_mrr_known_item() -> None:
    # THE collapse anchor: with |R| = 1, AP_q = 1/rank = RR_q element-wise, so MAP == MRR exactly.
    c = _corpus()
    for leg in LEG_NAMES:
        ap_ki = np.array([average_precision(_ranking(c, leg, q), c["qrels_ki"][q]) for q in range(c["n_queries"])])
        rr = per_query_rr(c, leg, "qrels_ki")
        assert np.max(np.abs(ap_ki - rr)) < 1e-12, leg
        assert abs(mean_average_precision(c, leg, "qrels_ki") - mean_reciprocal_rank(c, leg, "qrels_ki")) < 1e-12


def test_se_scales_as_inv_sqrt_n() -> None:
    c = _corpus()
    rows = se_scaling_experiment(c, "dense", trials=2000, seed=1)
    for row in rows:
        assert 0.75 <= row["empirical_se"] / max(row["theory_se"], 1e-12) <= 1.30, row  # tracks pop_std/sqrt(n)
    emp = [row["empirical_se"] for row in rows]
    assert all(emp[i + 1] < emp[i] for i in range(len(emp) - 1)), emp                    # strictly decreasing
    root = [row["se_root_n"] for row in rows]
    assert max(root) / min(root) < 1.35, root                                            # se*sqrt(n) ~ const


def test_bootstrap_se_matches_analytic() -> None:
    c = _corpus()
    for leg in ("dense", "lexical"):
        analytic = metric_summary(per_query_ap(c, leg))["se"]
        boot = bootstrap_map_se(c, leg, b=3000, seed=2)
        assert abs(boot - analytic) / max(analytic, 1e-12) < 0.15, (leg, boot, analytic)


def test_two_leg_overlap_motivates_significance() -> None:
    # The motivating non-result: dense and late_interaction (the CLOSE pair) have CIs that OVERLAP at a
    # small query count (you cannot yet tell them apart) but SEPARATE at a finite query count — the
    # how-many-queries-to-distinguish-two-systems hook the downstream significance topic resolves.
    c = _corpus()
    assert projected_overlap(c, "dense", "late_interaction", 5), "small-n projected CIs should overlap"
    sep = projected_separation_n(c, "dense", "late_interaction")
    assert sep is not None and 2 < sep <= c["n_queries"], sep
    assert not two_leg_map_comparison(c, "dense", "late_interaction")["ci_overlap"], "full-n CIs separate"


def test_metric_choice_flips_verdict() -> None:
    # RUN, not assumed: a natural PAIRWISE reversal (a pair ranked one way by MAP, the other by MRR) must
    # exist on the corpus, and the constructed extreme always flips too. The prose claims only what holds.
    flip = metric_choice_flips_verdict(_corpus())
    assert flip["flips"] and flip["reversal"] is not None, flip
    assert constructed_metric_flip()["flips"], "constructed toy must demonstrate the flip"


def test_legs_share_one_corpus_and_truth() -> None:
    c = _corpus()
    ids = set(range(c["n_docs"]))
    for leg in LEG_NAMES:
        for q in (0, c["n_queries"] // 2, c["n_queries"] - 1):
            assert set(_ranking(c, leg, q)) == ids, f"{leg} q{q} not a full permutation"
    again = brute_topk(c["queries"], c["docs"], TOPK)
    assert all(set(int(d) for d in a.tolist()) == c["qrels_set"][q] for q, a in enumerate(again))


def test_metric_guards() -> None:
    assert precision_at_k([1, 2, 3], {1}, 0) == 0.0
    assert recall_at_k([1, 2, 3], set(), 5) == 0.0
    assert average_precision([1, 2, 3], set()) == 0.0
    assert reciprocal_rank([1, 2, 3], set()) == 0.0
    assert f1_at_k([1, 2, 3], set(), 2) == 0.0
    # AP divides by |R|, NOT the number found: a ranking that finds only 1 of 3 relevant docs (at rank 1,
    # the other two ABSENT) scores (1/|R|)(1/1) = 1/3 — NOT 1.0 (which the wrong found-count denominator
    # gives). The classic AP bug, pinned.
    one_of_three = average_precision([0, 9, 8], {0, 1, 2})       # only doc 0 present (rank 1); 1, 2 absent
    assert abs(one_of_three - 1.0 / 3.0) < 1e-12, one_of_three
    # and burying the later relevant docs drags AP below the all-found-first ceiling of 1.0.
    buried = average_precision([0, 9, 8, 7, 1, 2], {0, 1, 2})    # 0 at rank 1, but 1 & 2 buried at 5, 6
    allfound = average_precision([0, 1, 2, 9, 8, 7], {0, 1, 2})  # AP = 1
    assert buried < allfound and abs(allfound - 1.0) < 1e-12, (buried, allfound)


def _run_all() -> None:
    print("set_metrics_precision_recall_map_mrr — verifying every claim:")
    test_precision_recall_bounds()
    test_recall_monotone_precision_not()
    test_recall_at_N_is_one()
    test_ap_equals_area_under_pr()
    test_interpolated_ge_raw_and_monotone()
    test_perfect_ranking_ap_one()
    test_map_is_mean_of_ap()
    test_mrr_is_mean_of_rr()
    test_rr_is_inverse_first_hit()
    test_map_equals_mrr_known_item()
    test_se_scales_as_inv_sqrt_n()
    test_bootstrap_se_matches_analytic()
    test_two_leg_overlap_motivates_significance()
    test_metric_choice_flips_verdict()
    test_legs_share_one_corpus_and_truth()
    test_metric_guards()
    print("all set-metric tests passed")
    viz_constants()


if __name__ == "__main__":
    _run_all()
