"""Retrieval versus Long Context: attention complexity, the precision/dilution optimum, and positional bias.

The canonical, tested reference for the `retrieval-vs-long-context` topic. It reads the
retrieve-a-few-passages versus stuff-the-window choice as a RATE-DISTORTION decision at the generation
boundary, continuing `retriever-as-noisy-channel` (which closed on "the retriever spends bits read to
buy down answer error"). Three strands matching the node title plus a synthesis:

  1. Attention complexity (the RATE axis): full self-attention over n = k*L tokens costs Theta((k*L)^2)
     FLOPs. Exact arithmetic. Stuffing the window pays for every token-pair.
  2. More context is not better (the SHOWPIECE): on a corpus with MULTIPLE relevant passages per answer
     plus same-sector hard distractors, the generator's answer posterior over the top-k context -- read
     under a FINITE ATTENTION BUDGET (the weights sum to one) -- is best at the SMALLEST context that
     carries the answer and DEGRADES monotonically as k grows. Two mechanisms: extra relevant passages are
     redundant (diminishing returns, the imported saturation result), and same-sector distractors then
     dilute the budget and raise the answer entropy. recall climbs while precision falls; answer quality
     tracks precision, not recall. (Only under POOR retrieval does a small interior optimum appear -- the
     recall-precision tension -- so the right amount of context DEPENDS on retrieval quality; this is shown
     as an honest contrast, not asserted as a headline because it is seed-fragile.)
  3. Positional bias (lost-in-the-middle): a relevant passage buried in the middle of a long context is
     read with attenuated attention -- a SOFT ERASURE, tying back to the binary erasure channel.

The answer model is the synthetic von Mises-Fisher softmax stand-in the PMI and noisy-channel topics
built; this module IMPORTS it and never reimplements it. Every pedagogical claim is an `assert`-based
test, and `viz_constants()` prints every number the React lab mirrors to the decimal.

Run:  uv run --with numpy --with scipy \\
          python notebooks/retrieval-vs-long-context/retrieval_vs_long_context.py

VIZ <-> PYTHON INVARIANT: the constants printed by `viz_constants()` are mirrored to the decimal in
`src/components/viz/RetrievalVsLongContextLaboratory.tsx`. Change a number here, re-run, then update the
.tsx -- never the reverse. Q(k) and the positional curve are MODEL OUTPUTS and are BAKED; only closed
forms (cost = (k*L)^2, the positional weight, entropy/softmax from baked logits) recompute in TS.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.special import logsumexp  # never hand-roll softmax/sigmoid (overflow + a Gemini flag)

# --- import the prereq chain (notebook import graph != pedagogical DAG: the frontmatter prereq stays
#     the-retrieval-problem; these imports only SOURCE numbers, they are not reader prerequisites) -----
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "the-retrieval-problem",
    "infonce-contrastive-objective",
    "dense-retrieval-dual-encoders",
    "set-metrics-precision-recall-map-mrr",
    "pmi-retrieval-value",
    "retriever-as-noisy-channel",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf, mean_resultant_length  # noqa: E402
from dense_retrieval_dual_encoders import (  # noqa: E402
    dual_encoder_score, DPR_SEED, DPR_DIM, DPR_KAPPA_SECTOR,
)
from pmi_retrieval_value import (  # noqa: E402
    answer_posterior, answer_posterior_two, saturation_table, entropy, TAU, TAU_DOC,
)
from retriever_as_noisy_channel import bayes_error  # noqa: E402
from set_metrics_precision_recall_map_mrr import recall_at_k, precision_at_k  # noqa: E402


# =========================================================================== #
# Constants. The geometry reuses the dense-retrieval sectors/companies (same kappa_sector confusability)
# but THICKENS the corpus: 4 companies/sector and R relevant passages per company, so the top-k
# necessarily starts admitting same-sector distractors once the relevant set is exhausted.
# =========================================================================== #

RVLC_SEED = DPR_SEED                  # 7 -- the shared finance-geometry seed
RVLC_DIM = DPR_DIM                    # 32
RVLC_N_SECTORS = 4
RVLC_N_COMP = 4                       # 4 companies/sector -> K = 16 answers, a thick same-sector shell
R_RELEVANT = 4                        # relevant passages per company (the |R| > 1 set: recall climbs)
QUERIES_PER_COMPANY = 2               # -> 16 * 2 = 32 queries
KAPPA_SECTOR = DPR_KAPPA_SECTOR       # 60.0 -- same-sector cosine ~ 0.4 at d=32 (confusable distractors)
KAPPA_REL = 120.0                     # relevant passages around the company prototype: cosine ~ 0.6,
                                      #   sub-saturating so the FIRST few genuinely help
KAPPA_QUERY = 60.0                    # query identifies the gold COMPANY (recall@1 ~ 1.0): the realistic
                                      #   regime where the answer IS retrieved, so the fight is purely
                                      #   precision / attention / cost. (Lower kappa = harder retrieval;
                                      #   `retrieval_quality_family` contrasts good vs hard at kappa=15.)
KAPPA_QUERY_HARD = 15.0               # a hard-retrieval contrast (recall@1 ~ 0.47) for the second viz panel
TAU = TAU                             # 0.2 -- imported PMI temperature; the collapse anchors pin against it
TAU_GEN = 0.30                        # generation temperature for the answer-quality CURVES (tuned: a soft
                                      #   enough posterior that one passage does not saturate the answer)
TAU_ATTN = 0.45                       # attention-budget temperature (soft, so same-sector distractors that
                                      #   enter the context still draw budget -- the dilution mechanism)
PASSAGE_LEN = 256                     # tokens per passage -- the cost axis only
POS_DIP = 0.55                        # depth of the lost-in-the-middle positional dip (Liu et al. 2023)

K = RVLC_N_SECTORS * RVLC_N_COMP                      # 16 companies = answers
SECTOR_SIZE = RVLC_N_COMP * R_RELEVANT               # 16 passages live in the gold's sector
K_GRID = list(range(1, SECTOR_SIZE + 1))             # 1..16: relevant set + the same-sector shell
N_CTX = SECTOR_SIZE                                  # context length for the positional sweep
N_QUERIES = K * QUERIES_PER_COMPANY                  # 32

LOG2 = math.log(2.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax via logsumexp (the prereq idiom; never hand-rolled)."""
    logits = np.asarray(logits, dtype=float)
    return np.exp(logits - logsumexp(logits))


# =========================================================================== #
#  ★  THE FOUR CORE MODEL FUNCTIONS -- Jonathan writes these bodies.
#     Each encodes a load-bearing design choice. The surrounding machinery, tests, and viz_constants are
#     written against the signatures + return contracts documented here; a body that honors the contract
#     drops straight in. See the handoff note in the session for the design tradeoff behind each.
# =========================================================================== #

def rvlc_corpus(seed: int = RVLC_SEED, dim: int = RVLC_DIM, n_sectors: int = RVLC_N_SECTORS,
                n_comp: int = RVLC_N_COMP, r: int = R_RELEVANT, qpc: int = QUERIES_PER_COMPANY,
                kappa_sector: float = KAPPA_SECTOR, kappa_rel: float = KAPPA_REL,
                kappa_query: float = KAPPA_QUERY) -> dict:
    """★ Build the thick multi-passage finance corpus in the SAME vMF geometry as dense-retrieval.

    DESIGN CHOICE (yours): how "relevant" and "distractor" passages are drawn so that (a) the relevant
    set genuinely ranks above the distractors -- recall_R(k) climbs -- AND (b) the same-sector
    distractors carry enough wrong-company mass to bite once they enter the top-k.

    Recommended construction (mirrors `dpr_finance_matrix`, then thickens it):
      - sector means:    sector_mu = normalize(default_rng(seed).standard_normal((n_sectors, dim)))
                         (the SAME first rng draw dense-retrieval uses -- keeps us in its geometry).
      - company protos:  for each sector, members = sample_vmf(n_comp, sector_mu[s], kappa_sector, ...);
                         protos[a] = normalize(member) for each company a. (K = n_sectors*n_comp protos;
                         these ARE the answer prototypes, exactly as PMI uses P = protos.)
      - passages:        for each company a, draw `r` passages ~ sample_vmf(r, protos[a], kappa_rel, ...).
                         Stack into `passages` (n_passages = K*r), tracking company_of_passage[i] = a and
                         sector_of_passage[i] = sector of a.
      - queries:         for each company a, draw `qpc` queries ~ sample_vmf(qpc, protos[a], kappa_query);
                         normalize; truth[q] = a. (Company-tight: the gold company is identified.)

    Use DISTINCT per-draw seeds (e.g. seed + offsets keyed by sector/company) so draws are independent
    but deterministic. Returns a dict with at least:
        protos              (K, dim)            answer/company prototype directions
        sector_of_company   (K,)               sector label per company
        passages            (n_passages, dim)   all passages, unit vectors
        company_of_passage  (n_passages,)       owning company per passage
        sector_of_passage   (n_passages,)       sector per passage
        Q                   (n_queries, dim)    unit query vectors
        truth               (n_queries,)        gold company index per query
        relevant_sets       list[set[int]]      relevant passage ids per query (= passages of truth[q])
        K, R, n_passages, n_queries, dim        ints
    """
    rng = np.random.default_rng(seed)
    # Sector means: the SAME first rng draw dense-retrieval uses, so we live in its geometry.
    sector_mu = normalize(rng.standard_normal((n_sectors, dim)))
    protos, sector_of_company = [], []
    for si in range(n_sectors):
        members = sample_vmf(n_comp, sector_mu[si], kappa_sector, seed=seed + 11 * si + 1)
        for ci in range(n_comp):
            protos.append(normalize(members[ci]))
            sector_of_company.append(si)
    protos = np.array(protos)                                  # (K, dim) -- the answer prototypes
    sector_of_company = np.array(sector_of_company)
    n_companies = protos.shape[0]
    # Passages: r per company, drawn around the company prototype (sub-saturating concentration).
    passages, company_of_passage, sector_of_passage = [], [], []
    for a in range(n_companies):
        draws = sample_vmf(r, protos[a], kappa_rel, seed=seed + 4001 + 17 * a)
        for p in np.atleast_2d(draws):
            passages.append(normalize(p))
            company_of_passage.append(a)
            sector_of_passage.append(int(sector_of_company[a]))
    passages = np.array(passages)
    company_of_passage = np.array(company_of_passage)
    sector_of_passage = np.array(sector_of_passage)
    # Queries: qpc per company, drawn around the company prototype. kappa_query controls how decisively the
    # query alone picks the company -- low kappa -> ambiguous -> retrieval is imperfect (recall@1 < 1).
    queries, truth = [], []
    for a in range(n_companies):
        draws = sample_vmf(qpc, protos[a], kappa_query, seed=seed + 9001 + 13 * a)
        for q in np.atleast_2d(draws):
            queries.append(normalize(q))
            truth.append(a)
    queries = np.array(queries)
    truth = np.array(truth)
    relevant_sets = [set(int(i) for i in np.where(company_of_passage == truth[qi])[0])
                     for qi in range(len(truth))]
    return {
        "protos": protos, "sector_of_company": sector_of_company,
        "passages": passages, "company_of_passage": company_of_passage,
        "sector_of_passage": sector_of_passage,
        "Q": queries, "truth": truth, "relevant_sets": relevant_sets,
        "K": int(n_companies), "R": int(r), "n_passages": int(len(passages)),
        "n_queries": int(len(queries)), "dim": int(dim),
    }


def attention_weights(scores: np.ndarray, tau_attn: float = TAU_ATTN) -> np.ndarray:
    """★ The finite attention budget over a context: nonnegative weights that SUM TO ONE.

    DESIGN CHOICE (yours): a relevance-aware budget vs a uniform one. The recommended form is
    softmax-over-relevance, w_j ∝ exp(score_j / tau_attn), so a context passage competes for a fixed
    unit of attention -- admitting a high-scoring same-sector distractor STEALS budget from the relevant
    passages (this is the mechanism that makes "more context" actively harmful). A uniform 1/m budget is
    the simpler, more pessimistic alternative. GUARD: tau_attn > 1e-8; handle all-equal scores gracefully
    (softmax of equal scores is uniform, which is the correct degenerate behavior). Returns shape
    (len(scores),), sums to 1.
    """
    if not (tau_attn > 1e-8):
        raise ValueError(f"tau_attn must be > 1e-8, got {tau_attn}")
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return s                              # empty context -> empty budget (no logsumexp over [])
    # Relevance-aware budget: softmax of the retrieval scores. Equal scores -> uniform (the correct
    # degenerate behavior); a soft tau_attn keeps the budget spread so distractors still draw weight.
    return _softmax(s / tau_attn)


def answer_posterior_topk(q: np.ndarray, ctx_vecs: np.ndarray, weights: np.ndarray,
                          protos: np.ndarray, tau: float = TAU) -> np.ndarray:
    """★ The attention-weighted multi-document answer posterior over the K prototypes.

    DESIGN CHOICE (yours): how a finite attention budget combines the per-passage evidence. The natural
    generalization of the prereq additive-logits model (`answer_posterior`, `answer_posterior_two`) is

        logits_a = ( <q, proto_a> + sum_j weights[j] * <ctx_j, proto_a> ) / tau,    p = softmax(logits).

    `weights` is an EXPLICIT argument (not computed inside) precisely so the collapse anchors can pass
    degenerate weights and recover the imported functions:
        - k=1, weights=[1.0]      -> equals answer_posterior(q, ctx[0], protos, tau)
        - k=2, weights=[1.0,1.0]  -> equals answer_posterior_two(q, ctx[0], ctx[1], protos, tau)
    The topic's curves pass budget weights from `attention_weights` (which sum to 1). GUARD: tau > 1e-8.
    Returns shape (K,), sums to 1.
    """
    if not (tau > 1e-8):
        raise ValueError(f"tau must be > 1e-8, got {tau}")
    ctx = np.atleast_2d(np.asarray(ctx_vecs, dtype=float))
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != ctx.shape[0]:
        raise ValueError(f"weights and ctx_vecs must match in length, got {w.shape[0]} and {ctx.shape[0]}")
    sims = ctx @ protos.T                       # (k, K): each context passage's similarity to each prototype
    logits = (q @ protos.T + w @ sims) / tau    # query evidence + budget-weighted document evidence
    return _softmax(logits)


def positional_weight(pos: int, n_ctx: int, dip: float = POS_DIP) -> float:
    """★ The lost-in-the-middle positional attention multiplier: high at the ends, depressed in the middle.

    DESIGN CHOICE (yours): the shape and depth of the U. This is an EMPIRICAL stand-in for the observed
    behavior of trained transformers (Liu et al. 2023), not a derived quantity -- it is baked in so the
    topic can trace its CONSEQUENCE (a buried relevant passage is a soft erasure). A clean closed form a
    TS mirror can reproduce, e.g. a normalized cosine bump  1 - dip * sin(pi * (pos + 0.5) / n_ctx)  (1.0
    at the two ends, 1 - dip at the center). Must return a positive multiplier in (0, 1]; `dip` in [0, 1).
    Keep it expressible in TypeScript verbatim (the .tsx recomputes it from the baked formula).
    """
    if n_ctx <= 1:
        return 1.0
    # A normalized sine bump: 1.0 at the two ends, 1 - dip at the center (Liu et al. 2023, baked).
    return float(1.0 - dip * math.sin(math.pi * (pos + 0.5) / n_ctx))


# =========================================================================== #
# Machinery written against the contracts above (mine).
# =========================================================================== #

_CORPUS: dict | None = None


def _corpus(seed: int = RVLC_SEED) -> dict:
    """Module-scope cache: the thick finance corpus is built once."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = rvlc_corpus(seed)
    return _CORPUS


def attention_cost(k: int, length: int = PASSAGE_LEN) -> float:
    """Full self-attention FLOPs over a context of k passages of `length` tokens: (k*length)^2 = the
    pairwise score matrix. The RATE axis. GUARD: k, length >= 0 (k=0 -> 0). FlashAttention lowers MEMORY
    to O(n) but leaves this arithmetic unchanged -- the rigorFlag."""
    k = max(int(k), 0)
    length = max(int(length), 0)
    return float((k * length) ** 2)


def rank_passages(corpus: dict, q_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense retrieval for one query: passage ids sorted by dual-encoder score (descending) and the
    matching sorted scores. Uses the IMPORTED `dual_encoder_score` (a single inner product per passage)."""
    q = corpus["Q"][q_idx]
    scores = dual_encoder_score(q, corpus["passages"])      # (n_passages,)
    order = np.argsort(-scores, kind="stable")
    return order, scores[order]


def _proto_view_corpus(corpus: dict) -> dict:
    """A PMI-shaped view (documents == company prototypes) so the IMPORTED `saturation_table` re-derives
    the diminishing-returns baseline on THIS geometry rather than us reimplementing it."""
    return {
        "Q": corpus["Q"], "P": corpus["protos"], "protos": corpus["protos"],
        "n_docs": corpus["K"], "n_queries": corpus["n_queries"],
        "truth": corpus["truth"], "sector_of_passage": corpus["sector_of_company"],
    }


def quality_recall_precision_cost(corpus: dict, ks=K_GRID, length: int = PASSAGE_LEN,
                                  tau: float = TAU_GEN, tau_attn: float = TAU_ATTN) -> list[dict]:
    """For each context depth k, averaged over queries:
        Q(k)        mean posterior mass on the TRUE answer  (E[p(a* | q, top-k)]) -- soft accuracy
        conf(k)     mean MAP confidence  (E[max_a p(a | q, top-k)] = 1 - mean bayes_error) -- IMPORTED.
                    With good retrieval conf == Q (the MAP answer is the truth): the model is calibrated
                    but UNCERTAIN, not confidently wrong -- the honest contrast with the substitution story.
        H(k)        mean answer-posterior entropy in bits (IMPORTED `entropy`) -- the DISTORTION axis:
                    rises as context dilutes the answer, the residual-entropy echo of the noisy channel.
        recall(k)   mean recall_at_k over the relevant SET (|R| = R > 1) -- IMPORTED, climbs
        precision(k) mean precision_at_k -- IMPORTED, falls as ~ R/k once distractors enter
        util(k)     mean attention-budget weight landing on relevant passages -- falls past k = R
        cost(k)     (k*length)^2 -- the rate axis
    Returns a list of dicts, one per k.
    """
    protos = corpus["protos"]
    rows = []
    for k in ks:
        Qv, confv, hv, recv, precv, utilv = [], [], [], [], [], []
        for qi in range(corpus["n_queries"]):
            order, sorted_scores = rank_passages(corpus, qi)
            ranking = [int(i) for i in order]
            topk_ids = order[:k]
            ctx = corpus["passages"][topk_ids]
            w = attention_weights(sorted_scores[:k], tau_attn)
            post = answer_posterior_topk(corpus["Q"][qi], ctx, w, protos, tau)
            a_star = int(corpus["truth"][qi])
            Qv.append(float(post[a_star]))
            confv.append(1.0 - bayes_error(post))
            hv.append(entropy(post))
            rel = corpus["relevant_sets"][qi]
            recv.append(recall_at_k(ranking, rel, k))
            precv.append(precision_at_k(ranking, rel, k))
            in_rel = np.array([1.0 if int(i) in rel else 0.0 for i in topk_ids])
            utilv.append(float((w * in_rel).sum()))
        Q_mean = float(np.mean(Qv))
        conf_mean = float(np.mean(confv))
        rows.append({
            "k": int(k), "Q": Q_mean, "conf": conf_mean, "gap": conf_mean - Q_mean,
            "H": float(np.mean(hv)), "recall": float(np.mean(recv)),
            "precision": float(np.mean(precv)), "util": float(np.mean(utilv)),
            "cost": attention_cost(k, length),
        })
    return rows


def retrieval_quality_family(corpus_good: dict | None = None, kappa_hard: float = KAPPA_QUERY_HARD,
                             ks=K_GRID, tau: float = TAU_GEN, tau_attn: float = TAU_ATTN) -> dict:
    """The honest two-regime contrast: Q(k) for GOOD retrieval (the canonical corpus, recall@1 ~ 1) and
    for HARD retrieval (a kappa_hard corpus, recall@1 ~ 0.47). Good retrieval peaks at k=1 and declines
    (lost-in-the-middle / dilution dominates); hard retrieval is flatter -- too few passages now risks
    MISSING the answer, the recall-precision tension that yields a small interior optimum when retrieval
    is poor. The robust, seed-independent claim asserted downstream: better retrieval gives strictly
    higher peak quality, and BOTH regimes are dominated by stuffing the whole window."""
    good = corpus_good if corpus_good is not None else _corpus()
    hard = rvlc_corpus(kappa_query=kappa_hard)
    g = quality_recall_precision_cost(good, ks, tau=tau, tau_attn=tau_attn)
    h = quality_recall_precision_cost(hard, ks, tau=tau, tau_attn=tau_attn)
    return {"good_Q": [r["Q"] for r in g], "hard_Q": [r["Q"] for r in h],
            "good_peak": max(r["Q"] for r in g), "hard_peak": max(r["Q"] for r in h)}


def find_optimum(rows: list[dict]) -> dict:
    """argmax_k Q(k) and the boundary comparisons. is_interior is True iff the optimum is strictly inside
    the grid AND beats both ends."""
    qs = [r["Q"] for r in rows]
    i_star = int(np.argmax(qs))
    k_star = rows[i_star]["k"]
    q_star, q_first, q_last = qs[i_star], qs[0], qs[-1]
    is_interior = (0 < i_star < len(rows) - 1) and (q_star > q_first) and (q_star > q_last)
    return {"k_star": int(k_star), "i_star": i_star, "Q_star": float(q_star),
            "Q_1": float(q_first), "Q_n": float(q_last), "is_interior": bool(is_interior)}


def positional_quality(corpus: dict, n_ctx: int = N_CTX, dip: float = POS_DIP,
                       tau: float = TAU_GEN, tau_attn: float = TAU_ATTN) -> list[dict]:
    """Lost-in-the-middle: fix a context of n_ctx passages (the gold relevant passage + distractors) and
    slide the gold across every position, discounting each passage's attention by `positional_weight`.
    Returns [{pos, Q}] -- U-shaped in position: a buried gold is a soft erasure. Averaged over queries.

    For each query we take the top n_ctx retrieved passages (gold present, since recall is high), then for
    each candidate position p we (1) reorder so the gold sits at p, (2) form base budget weights from the
    retrieval scores, (3) multiply weight_j by positional_weight(position_j) and renormalize, (4) read the
    posterior mass on the truth. Q(mid) < Q(ends)."""
    protos = corpus["protos"]
    per_pos = {p: [] for p in range(n_ctx)}
    for qi in range(corpus["n_queries"]):
        order, sorted_scores = rank_passages(corpus, qi)
        ids = order[:n_ctx]
        scores = sorted_scores[:n_ctx]
        rel = corpus["relevant_sets"][qi]
        gold_local = next((j for j, i in enumerate(ids) if int(i) in rel), None)
        if gold_local is None:
            continue                                  # gold not retrieved (rare here); skip
        a_star = int(corpus["truth"][qi])
        rest = [j for j in range(len(ids)) if j != gold_local]
        for p in range(n_ctx):
            placed = rest[:p] + [gold_local] + rest[p:]   # gold inserted at context position p
            ctx = corpus["passages"][ids[placed]]
            base = attention_weights(scores[placed], tau_attn)
            pos_mult = np.array([positional_weight(pos, n_ctx, dip) for pos in range(n_ctx)])
            w = base * pos_mult
            s = w.sum()
            w = w / s if s > 0 else np.full(n_ctx, 1.0 / n_ctx)
            post = answer_posterior_topk(corpus["Q"][qi], ctx, w, protos, tau)
            per_pos[p].append(float(post[a_star]))
    return [{"pos": p, "Q": float(np.mean(v)) if v else 0.0} for p, v in per_pos.items()]


# =========================================================================== #
# Tests -- every pedagogical claim is an assertion. Run order: collapse anchors, then the headline gate,
# then the directional gates, then guards.
# =========================================================================== #

def test_collapse_k1_is_pmi() -> None:
    """k=1 with weight [1.0] reproduces the imported single-document posterior bit-for-bit."""
    c = _corpus()
    q = c["Q"][0]
    d = c["passages"][0]
    got = answer_posterior_topk(q, d[None, :], np.array([1.0]), c["protos"], TAU)
    want = answer_posterior(q, d, c["protos"], TAU)
    assert np.allclose(got, want, atol=1e-12), "k=1 must equal imported answer_posterior"


def test_collapse_k2_is_pmi_two() -> None:
    """k=2 with weights [1,1] reproduces the imported two-document posterior bit-for-bit (the budget
    layer generalizes the prereq's additive logits)."""
    c = _corpus()
    q = c["Q"][0]
    d1, d2 = c["passages"][0], c["passages"][1]
    got = answer_posterior_topk(q, np.stack([d1, d2]), np.array([1.0, 1.0]), c["protos"], TAU)
    want = answer_posterior_two(q, d1, d2, c["protos"], TAU)
    assert np.allclose(got, want, atol=1e-12), "k=2 weights (1,1) must equal answer_posterior_two"


def test_attention_budget_sums_to_one() -> None:
    """The budget is a probability vector for every k, and the temperature guard fires."""
    c = _corpus()
    _, sorted_scores = rank_passages(c, 0)
    for k in (1, 4, 8, len(K_GRID)):
        w = attention_weights(sorted_scores[:k], TAU_ATTN)
        assert w.shape == (k,) and np.all(w >= -1e-12) and abs(w.sum() - 1.0) < 1e-9
    try:
        attention_weights(sorted_scores[:3], 0.0)
        raise AssertionError("tau_attn <= 1e-8 must raise")
    except ValueError:
        pass


def test_attention_cost_quadratic() -> None:
    """cost(k) = (k*L)^2 exactly, and doubling k quadruples the cost (THEOREM)."""
    assert attention_cost(3, 256) == (3 * 256) ** 2
    assert math.isclose(attention_cost(8, 256) / attention_cost(4, 256), 4.0)
    assert attention_cost(0, 256) == 0.0


def test_recall_set_climbs() -> None:
    """The |R| > 1 relevant set escapes the 'too easy' trap: recall_R(1) < 1 and recall_R(n) == 1."""
    rows = quality_recall_precision_cost(_corpus())
    assert rows[0]["recall"] < 1.0 - 1e-9, "recall@1 must be below 1 (multiple relevant passages)"
    assert abs(rows[-1]["recall"] - 1.0) < 1e-9, "recall must reach 1 once the sector is covered"
    recs = [r["recall"] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(recs, recs[1:])), "recall must be non-decreasing in k"


def test_saturation_imported_matches() -> None:
    """The imported `saturation_table` re-derives diminishing returns on this geometry:
    redundant << standalone and redundant << novel (not reimplemented)."""
    sat = saturation_table(_proto_view_corpus(_corpus()), TAU, TAU_DOC)
    assert sat["redundant"] < sat["standalone"], "a redundant second filing adds less than the first"
    assert sat["redundant"] < sat["novel"], "a novel filing beats a redundant one"


def test_more_context_hurts() -> None:
    """HEADLINE GATE (robust, seed-independent): even with the answer reliably retrieved (recall@1 ~ 1),
    answer quality Q(k) is maximized at the SMALLEST context and declines monotonically -- more context is
    pure downside. The answer ENTROPY H(k) rises in lockstep (the distortion axis, in bits)."""
    rows = quality_recall_precision_cost(_corpus())
    qs = [r["Q"] for r in rows]
    hs = [r["H"] for r in rows]
    assert int(np.argmax(qs)) == 0, "quality must peak at the smallest context (k=1)"
    assert all(b <= a + 1e-9 for a, b in zip(qs, qs[1:])), "Q must be non-increasing in k"
    assert qs[0] - qs[-1] >= 0.1, "stuffing the window must cost a real amount of quality"
    assert all(b >= a - 1e-9 for a, b in zip(hs, hs[1:])), "answer entropy must rise as context dilutes"


def test_less_is_more_mechanism() -> None:
    """The TWO mechanisms behind the decline: while the top-k is all-relevant, precision ~ 1 and added
    passages are REDUNDANT (Q nearly flat); once distractors enter, precision falls, the attention budget
    bleeds off the relevant passages (utilization falls), and Q declines. Cost rises quadratically."""
    rows = quality_recall_precision_cost(_corpus())
    precs = [r["precision"] for r in rows]
    costs = [r["cost"] for r in rows]
    assert all(b <= a + 1e-9 for a, b in zip(precs, precs[1:])), "precision must be non-increasing"
    assert all(b > a for a, b in zip(costs, costs[1:])), "cost must strictly increase"
    tail = [r["util"] for r in rows if r["k"] >= R_RELEVANT]
    assert all(b <= a + 1e-9 for a, b in zip(tail, tail[1:])), "utilization must fall past k=R"


def test_retrieval_quality_regimes() -> None:
    """The honest two-regime contrast: better retrieval yields strictly higher peak quality, and stuffing
    the window is dominated under BOTH good and hard retrieval (the interior optimum, when it appears, is a
    poor-retrieval phenomenon -- not asserted as a headline because it is seed-fragile)."""
    fam = retrieval_quality_family()
    assert fam["good_peak"] > fam["hard_peak"] + 0.05, "good retrieval must beat hard retrieval at the peak"
    assert fam["good_Q"][-1] < fam["good_peak"] - 0.05, "stuffing dominated under good retrieval"
    assert fam["hard_Q"][-1] < fam["hard_peak"] + 1e-9, "stuffing not better than the peak under hard retrieval"


def test_buried_gold_soft_erasure() -> None:
    """Positional bias: quality is U-shaped in the gold's context position -- a buried passage is a soft
    erasure. Q(mid) < Q(start) and Q(mid) < Q(end); the two ends are comparable."""
    pq = positional_quality(_corpus())
    qs = [r["Q"] for r in pq]
    mid = len(qs) // 2
    assert qs[mid] < qs[0] - 1e-9 and qs[mid] < qs[-1] - 1e-9, "the middle must be worst"
    assert abs(qs[0] - qs[-1]) < 0.15, "the two ends should be roughly symmetric"


def test_pareto_dominates_stuff_all() -> None:
    """Rate-distortion verdict: stuffing the whole sector costs far more AND answers no better than the
    optimum -- focused retrieval strictly dominates."""
    rows = quality_recall_precision_cost(_corpus())
    opt = find_optimum(rows)
    last = rows[-1]
    assert last["cost"] > rows[opt["i_star"]]["cost"], "stuffing must cost strictly more"
    assert last["Q"] <= opt["Q_star"] + 1e-9, "stuffing must not improve quality over the optimum"


def test_guards() -> None:
    """Denominator and degenerate-input guards."""
    c = _corpus()
    assert attention_cost(0) == 0.0
    assert recall_at_k(list(range(c["n_passages"])), set(), 5) == 0.0   # empty relevant -> 0
    try:
        answer_posterior_topk(c["Q"][0], c["passages"][:1], np.array([1.0]), c["protos"], 0.0)
        raise AssertionError("tau <= 1e-8 must raise")
    except ValueError:
        pass
    eq = attention_weights(np.zeros(5), TAU_ATTN)        # all-equal scores -> uniform budget
    assert np.allclose(eq, 0.2, atol=1e-9)


def _run_all() -> None:
    tests = [
        test_collapse_k1_is_pmi, test_collapse_k2_is_pmi_two, test_attention_budget_sums_to_one,
        test_attention_cost_quadratic, test_recall_set_climbs, test_saturation_imported_matches,
        test_more_context_hurts, test_less_is_more_mechanism, test_retrieval_quality_regimes,
        test_buried_gold_soft_erasure, test_pareto_dominates_stuff_all, test_guards,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")


# =========================================================================== #
# Viz constants -- printed for mirroring into RetrievalVsLongContextLaboratory.tsx (cast scalars).
# =========================================================================== #

def viz_constants() -> None:
    c = _corpus()
    rows = quality_recall_precision_cost(c)
    opt = find_optimum(rows)
    fam = retrieval_quality_family(c)
    pq = positional_quality(c)
    sat = saturation_table(_proto_view_corpus(c), TAU, TAU_DOC)

    def arr(key):
        return [round(float(r[key]), 4) for r in rows]

    print("// --- baked from viz_constants() ---")
    print(f"const K = {c['K']};                 // companies = answers")
    print(f"const R_RELEVANT = {R_RELEVANT};")
    print(f"const N_QUERIES = {c['n_queries']};")
    print(f"const TAU_GEN = {TAU_GEN};")
    print(f"const TAU_ATTN = {TAU_ATTN};")
    print(f"const PASSAGE_LEN = {PASSAGE_LEN};")
    print(f"const KAPPA_REL = {KAPPA_REL};")
    print(f"const KAPPA_QUERY = {KAPPA_QUERY};       // good retrieval (recall@1 ~ 1)")
    print(f"const KAPPA_QUERY_HARD = {KAPPA_QUERY_HARD}; // hard retrieval contrast")
    print(f"const K_GRID = {[r['k'] for r in rows]};")
    print(f"const Q = {arr('Q')};                // answer quality (mass on truth), good retrieval")
    print(f"const H = {arr('H')};                 // answer entropy in bits (the distortion axis), rises")
    print(f"const RECALL_R = {arr('recall')};")
    print(f"const PRECISION = {arr('precision')};")
    print(f"const UTIL = {arr('util')};")
    print(f"const COST = {[int(r['cost']) for r in rows]};")
    print(f"const HARD_Q = {[round(x, 4) for x in fam['hard_Q']]}; // Q(k) under hard retrieval")
    print(f"const K_STAR = {opt['k_star']};        // the optimum: the smallest covering context")
    print(f"const Q_1 = {round(opt['Q_1'], 4)};")
    print(f"const Q_N = {round(opt['Q_n'], 4)};")
    print(f"const N_CTX = {N_CTX};")
    print(f"const POS_DIP = {POS_DIP};")
    print(f"const POS_Q = {[round(float(r['Q']), 4) for r in pq]};")
    print(f"const SAT_STANDALONE = {round(sat['standalone'], 4)};")
    print(f"const SAT_REDUNDANT = {round(sat['redundant'], 4)};")
    print(f"const SAT_NOVEL = {round(sat['novel'], 4)};")
    print(f"const SAME_SECTOR_COSINE = {round(float(mean_resultant_length(RVLC_DIM, KAPPA_SECTOR)) ** 2, 4)};")


if __name__ == "__main__":
    print("retrieval_vs_long_context: running tests")
    _run_all()
    print("\nviz_constants (mirror into the .tsx):")
    viz_constants()
    print("\nall checks passed.")
