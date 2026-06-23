"""Pointwise mutual information — what retrieval adds to generation, in BITS.

The reference implementation for the formalRAG `pmi-retrieval-value` topic, the keystone of the
information-theory layer. The evaluation layer scored retrieval by whether the right documents came back
(recall, AP, MAP). This topic scores it by a different question: how many BITS does a retrieved document
add to the generator's answer? Generation begins with an answer prior p(a|q) — uncertain, with entropy
H(A|Q) — and a retrieved document d sharpens it to a posterior p(a|q,d). The pointwise mutual information

        pmi(a; d | q) = log2  p(a|q,d) / p(a|q)

is, for a particular answer a, the log-factor by which the document moved the odds (the log of a
Radon-Nikodym density ratio). Its expectation over the posterior is the per-document information gain
KL(p(.|q,d) || p(.|q)) >= 0, and averaging over documents and queries is the conditional mutual information

        I(A; D | Q) = H(A|Q) - H(A|Q,D) = E_{q,d}[ KL(post || prior) ]  >= 0,

"the bits retrieval adds." Five movements, every pedagogical claim an assert:

  MOVEMENT 0 — the corpus. We REUSE the dense-retrieval finance vMF geometry (`dpr_finance_matrix`: four
    sectors, two companies each, ONE filing per company), so the answer prototypes ARE the company
    document directions. We add a topic-specific, SECTOR-AMBIGUOUS query set (drawn around the sector
    mean at kappa_query, not the company), so the prior genuinely spreads over a sector's companies and
    there are bits to add — a tuned query set is the exception the notebook contract allows when the
    prereq's own queries (drawn company-tight at kappa=350) would make the prior already certain.

  MOVEMENT 1 — the answer model: a retrieval distribution p(d|q), a per-document answer posterior
    p(a|q,d), and the prior as the RAG MARGINAL p(a|q) = sum_d p(d|q) p(a|q,d). The marginal definition
    (not a free softmax) is exactly what makes the MI identity in Movement 2 close.

  MOVEMENT 2 — information gain and the three-way-verified identity. I(A;D|Q) computed three ways
    (expected KL, entropy reduction, joint-distribution sum) agree to < 1e-9. Cite Shannon; up-link
    formalML shannon-entropy and kl-divergence.

  MOVEMENT 3 — a relevant filing adds bits; a distractor costs them. pmi(a*;d|q) > 0 for the gold
    filing, < 0 for a same-sector distractor that drags the posterior toward the wrong company — even
    though the distractor's KL(post||prior) is still > 0 (it moves mass, just wrongly).

  MOVEMENT 4 — diminishing returns. A second copy of a document barely moves belief (small KL); a
    different document moves it a lot. The chain rule of mutual information made visible (forward edge to
    context selection / submodular coverage, named in prose).

  MOVEMENT 5 — the encoder was trained to maximize a lower bound on these bits. InfoNCE is a lower bound
    on I(Q;D) ceilinged at log(N+1): we IMPORT `info_nce_loss_batch` and show the bound saturates at the
    ceiling, the same I(Q;D) whose answer-side payoff Movement 2 measures. And bits-added is a DIFFERENT
    axis from recall@k/AP (imported from set-metrics): a top-ranked document can add little, a low-ranked
    one the decisive bit.

Honest caveats (rigorFlag territory): the answer model p(a|q,d) is a SYNTHETIC vMF/softmax stand-in for a
real LLM generator — an exponential-family readout over a labeled corpus, not a trained language model, so
every "bit" is exact for the model and only illustrative of a deployed generator. Queries are
sector-ambiguous synthetic vMF draws, not natural language. MI estimation from samples is hard and biased;
the InfoNCE bound is loose and ceilinged at log(N+1). The "distractor costs bits" claim is about
pmi(a*;d|q) < 0 for the TRUE answer a* under this model — a property of (answer, document) pairs, not of
relevance labels.

This module imports its prerequisites (`hypersphere-vmf-geometry`, `dense-retrieval-dual-encoders`,
`infonce-contrastive-objective`, `set-metrics-precision-recall-map-mrr`) and never reimplements them.
`viz_constants()` prints what `PMIRetrievalValueLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy \\
        python notebooks/pmi-retrieval-value/pmi_retrieval_value.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.special import logsumexp  # never hand-roll softmax/sigmoid (overflow + a Gemini flag)

# --------------------------------------------------------------------------- #
# Import the published stack. Add each ancestor's HYPHENATED dir to the path, import the UNDERSCORED
# module. We reuse the dense-retrieval finance geometry + dual-encoder score, the InfoNCE batch loss (the
# bound the encoder minimized), and the set-metric estimators; we never reimplement any of them.
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "the-retrieval-problem",
    "infonce-contrastive-objective",
    "dense-retrieval-dual-encoders",
    "set-metrics-precision-recall-map-mrr",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf, mean_resultant_length   # noqa: E402
from dense_retrieval_dual_encoders import (                                          # noqa: E402
    dual_encoder_score, dpr_finance_matrix,
    DPR_SEED, DPR_DIM, DPR_N_SECTORS, DPR_N_COMP,
)
from infonce_contrastive_objective import info_nce_loss_batch                        # noqa: E402
from set_metrics_precision_recall_map_mrr import (                                    # noqa: E402
    recall_at_k, average_precision, metric_summary,
)

# Module constants (verified design point; the corpus shares the dense topic's seed/geometry).
PMI_SEED = DPR_SEED              # 7 — the SAME finance geometry the dense-retrieval topic built
PMI_DIM = DPR_DIM               # 32
PMI_N_SECTORS = DPR_N_SECTORS   # 4
PMI_N_COMP = DPR_N_COMP         # 2 companies per sector -> K = 8 answers/documents
PMI_QPS = 8                     # queries per sector -> 32 sector-ambiguous queries
PMI_KAPPA_QUERY = 30.0          # query concentration around the SECTOR mean (ambiguous at company level)
TAU = 0.2                       # answer-model temperature (the sign separation needs tau <~ 0.2)
TAU_DOC = 0.2                   # retrieval-distribution temperature
LOG2 = math.log(2.0)


# =========================================================================== #
# Movement 0 — the shared corpus: reuse the dpr geometry, add sector-ambiguous queries.
# =========================================================================== #

def pmi_corpus(seed: int = PMI_SEED, dim: int = PMI_DIM, n_sectors: int = PMI_N_SECTORS,
               n_comp: int = PMI_N_COMP, qps: int = PMI_QPS,
               kappa_query: float = PMI_KAPPA_QUERY) -> dict:
    """Reuse the dense-retrieval finance corpus and add a sector-ambiguous query set.

    `dpr_finance_matrix` gives P (one unit-vector filing per company, the answer prototypes) and the
    sector label of each company. We then draw queries around each SECTOR MEAN at kappa_query so a query
    pins the sector but not the company; the gold answer truth[q] is the in-sector company whose prototype
    the query is nearest. The sector means are recovered exactly as `dpr_finance_matrix` seeds them
    (one default_rng(seed) draw), so the queries live in the SAME geometry as the documents."""
    _, P, _, sector_of_passage = dpr_finance_matrix(
        seed=seed, dim=dim, n_sectors=n_sectors, n_comp=n_comp)
    sector_of_passage = np.asarray(sector_of_passage)
    protos = P                                              # company filing directions = answer prototypes
    K = P.shape[0]
    # Recover the sector means dpr_finance_matrix used (its only consumption of the seeded rng).
    rng = np.random.default_rng(seed)
    sector_mu = normalize(rng.standard_normal((n_sectors, dim)))
    queries, truth, q_sector = [], [], []
    for si in range(n_sectors):
        comp_ids = np.where(sector_of_passage == si)[0]     # the companies in this sector
        draws = sample_vmf(qps, sector_mu[si], kappa_query, seed=seed + 911 + si)
        for q in np.atleast_2d(draws):
            qn = normalize(q)
            gold = int(comp_ids[int(np.argmax(protos[comp_ids] @ qn))])
            queries.append(qn)
            truth.append(gold)
            q_sector.append(si)
    return {
        "P": P, "protos": protos, "sector_of_passage": sector_of_passage,
        "Q": np.array(queries), "truth": np.array(truth), "q_sector": np.array(q_sector),
        "K": int(K), "n_docs": int(K), "n_queries": len(queries),
    }


_CORPUS: dict | None = None


def _corpus(seed: int = PMI_SEED) -> dict:
    """Module-scope cache: the finance geometry and the query set are built once."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = pmi_corpus(seed)
    return _CORPUS


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax via logsumexp (no hand-rolled exp/normalize)."""
    logits = np.asarray(logits, dtype=float)
    return np.exp(logits - logsumexp(logits))


# =========================================================================== #
# Movement 1 — the answer model: p(d|q), p(a|q,d), and the prior as the RAG marginal.
# =========================================================================== #

def retrieval_dist(q: np.ndarray, P: np.ndarray, tau_doc: float = TAU_DOC) -> np.ndarray:
    """p(d|q) = softmax(<q, P[d]> / tau_doc) — the dual-encoder MIPS distribution over documents (the
    IMPORTED `dual_encoder_score` is the inner product). GUARD: tau_doc > 1e-8. Returns shape (n_docs,)."""
    if not (tau_doc > 1e-8):
        raise ValueError(f"tau_doc must be > 1e-8, got {tau_doc}")
    return _softmax(dual_encoder_score(q, P) / tau_doc)


def answer_posterior(q: np.ndarray, d_vec: np.ndarray, protos: np.ndarray,
                     tau: float = TAU) -> np.ndarray:
    """p(a | q, d): the answer distribution AFTER document d is retrieved, over the K answer prototypes.

    DESIGN CHOICE (yours to write — see the checkpoint note). The retrieved document should ADD its
    evidence to the query's, so that a relevant filing sharpens the posterior onto its own company and a
    distractor sharpens it onto the wrong one. The natural form is a softmax over the COMBINED query+
    document similarity to each prototype:  logits_a = (<q, proto_a> + <d_vec, proto_a>) / tau.

    GUARD: tau > 1e-8. Returns shape (K,), sums to 1.
    """
    if not (tau > 1e-8):
        raise ValueError(f"tau must be > 1e-8, got {tau}")
    # Additive log-linear: query and document evidence are independent sources summed in log-space, so a
    # relevant filing adds its similarity to its own company's prototype on top of the query's.
    logits = (q @ protos.T + d_vec @ protos.T) / tau
    return _softmax(logits)


def answer_prior(q: np.ndarray, P: np.ndarray, protos: np.ndarray,
                 tau: float = TAU, tau_doc: float = TAU_DOC) -> np.ndarray:
    """p(a | q): the answer distribution BEFORE a specific document is fixed — the RAG MARGINAL

            p(a | q) = sum_d  p(d | q) * p(a | q, d).

    DESIGN CHOICE (yours to write). This must be the marginal of `answer_posterior` against
    `retrieval_dist`, NOT an independent softmax — that marginal definition is exactly what makes
    I(A;D|Q) = H(A|Q) - H(A|Q,D) close to numerical zero (Movement 2). Returns shape (K,), sums to 1.
    """
    pdq = retrieval_dist(q, P, tau_doc)
    post = np.array([answer_posterior(q, P[d], protos, tau) for d in range(len(P))])
    return pdq @ post


def answer_posterior_two(q: np.ndarray, d1: np.ndarray, d2: np.ndarray, protos: np.ndarray,
                         tau: float = TAU) -> np.ndarray:
    """p(a | q, d1, d2): the posterior after TWO documents — both add their evidence to the query's.
    Used by Movement 4 (diminishing returns). Collapses to `answer_posterior` when d2 contributes the
    same logits as d1 only in the sense of stacking; it is a genuine two-document readout. GUARD tau>1e-8."""
    if not (tau > 1e-8):
        raise ValueError(f"tau must be > 1e-8, got {tau}")
    logits = (q @ protos.T + d1 @ protos.T + d2 @ protos.T) / tau
    return _softmax(logits)


def query_distributions(corpus: dict, q_idx: int, tau: float = TAU,
                        tau_doc: float = TAU_DOC) -> dict:
    """For one query: p(d|q), the per-document posteriors post[d] = p(a|q,d) (shape (n_docs, K)), and the
    prior as the marginal pdq @ post (algebraically identical to `answer_prior`, computed once here for the
    hot path; `test_prior_is_marginal` pins the equality)."""
    q = corpus["Q"][q_idx]
    P, protos = corpus["P"], corpus["protos"]
    pdq = retrieval_dist(q, P, tau_doc)
    post = np.array([answer_posterior(q, P[d], protos, tau) for d in range(corpus["n_docs"])])
    prior = pdq @ post
    return {"q": q, "pdq": pdq, "post": post, "prior": prior}


def pmi_pointwise(prior: np.ndarray, post: np.ndarray, a: int) -> float:
    """pmi(a; d | q) = log2 p(a|q,d)/p(a|q): the bits document d moved answer a's odds (sign included)."""
    return float(np.log2(post[a] / prior[a]))


# =========================================================================== #
# Movement 2 — information gain and the three-way-verified MI identity (in bits).
# =========================================================================== #

def entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) = -sum p log2 p in BITS, over the support p > 0 (empty/degenerate -> 0)."""
    p = np.asarray(p, dtype=float)
    m = p > 0
    return float(-(p[m] * np.log2(p[m])).sum()) if m.any() else 0.0


def kl(p: np.ndarray, q_ref: np.ndarray) -> float:
    """KL(p || q_ref) = sum p log2(p/q_ref) in BITS, over p > 0. The prior is a mixture that puts positive
    mass wherever any posterior does, so KL(post||prior) is finite. GUARD: skip p == 0 terms."""
    p = np.asarray(p, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    m = p > 0
    return float((p[m] * (np.log2(p[m]) - np.log2(q_ref[m]))).sum()) if m.any() else 0.0


def cond_mi_breakdown(corpus: dict, tau: float = TAU, tau_doc: float = TAU_DOC) -> dict:
    """I(A;D|Q) computed THREE ways and the entropies, averaged over queries:
      (a) expected KL:        E_q E_{d|q}[ KL(post_d || prior) ]
      (b) entropy reduction:  E_q[ H(prior) - sum_d p(d|q) H(post_d) ] = H(A|Q) - H(A|Q,D)
      (c) joint sum:          E_q sum_{a,d} p(a,d|q) log2[ p(a,d|q) / (p(a|q) p(d|q)) ]
    All three must agree to < 1e-9. Returns the three estimates plus H(A|Q) and H(A|Q,D)."""
    a_vals, b_vals, c_vals, h_prior, h_cond = [], [], [], [], []
    for qi in range(corpus["n_queries"]):
        qd = query_distributions(corpus, qi, tau, tau_doc)
        pdq, post, prior = qd["pdq"], qd["post"], qd["prior"]
        # (a) expected KL of each posterior to the prior.
        a_vals.append(sum(pdq[d] * kl(post[d], prior) for d in range(len(pdq))))
        # (b) entropy reduction.
        hp = entropy(prior)
        hc = sum(pdq[d] * entropy(post[d]) for d in range(len(pdq)))
        b_vals.append(hp - hc)
        h_prior.append(hp)
        h_cond.append(hc)
        # (c) joint-distribution mutual information.
        joint = post * pdq[:, None]                          # joint[d, a] = p(d|q) p(a|q,d)
        marg_a = prior                                       # p(a|q) = sum_d joint[d, a]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = joint / (pdq[:, None] * marg_a[None, :])
            terms = np.where(joint > 0, joint * np.log2(ratio), 0.0)
        c_vals.append(float(terms.sum()))
    return {
        "expected_kl": float(np.mean(a_vals)),
        "entropy_reduction": float(np.mean(b_vals)),
        "joint_sum": float(np.mean(c_vals)),
        "H_A_given_Q": float(np.mean(h_prior)),
        "H_A_given_QD": float(np.mean(h_cond)),
    }


def cond_mi(corpus: dict, tau: float = TAU, tau_doc: float = TAU_DOC) -> float:
    """The conditional mutual information I(A;D|Q) in bits (the entropy-reduction form)."""
    return cond_mi_breakdown(corpus, tau, tau_doc)["entropy_reduction"]


# =========================================================================== #
# Movement 3 — a relevant filing adds bits; a distractor costs them.
# =========================================================================== #

def _distractor_id(corpus: dict, q_idx: int) -> int:
    """The same-sector, different-company filing for query q_idx — the plausible distractor (n_comp = 2,
    so exactly one other company shares the sector)."""
    a_star = int(corpus["truth"][q_idx])
    si = int(corpus["sector_of_passage"][a_star])
    same = [j for j in np.where(corpus["sector_of_passage"] == si)[0] if j != a_star]
    return int(same[0]) if same else a_star


def pmi_at_truth_table(corpus: dict, tau: float = TAU, tau_doc: float = TAU_DOC) -> dict:
    """For every query, pmi at the TRUE answer a* for (i) the gold filing P[a*] and (ii) the same-sector
    distractor filing, plus the distractor's KL(post||prior). The headline: mean pmi_rel > 0 > mean
    pmi_distr, while the distractor still has KL > 0 (it moves belief, toward the wrong company)."""
    pmi_rel, pmi_distr, kl_distr = [], [], []
    P, protos = corpus["P"], corpus["protos"]
    for qi in range(corpus["n_queries"]):
        qd = query_distributions(corpus, qi, tau, tau_doc)
        prior = qd["prior"]
        a_star = int(corpus["truth"][qi])
        d_id = _distractor_id(corpus, qi)
        post_rel = answer_posterior(qd["q"], P[a_star], protos, tau)
        post_dis = answer_posterior(qd["q"], P[d_id], protos, tau)
        pmi_rel.append(pmi_pointwise(prior, post_rel, a_star))
        pmi_distr.append(pmi_pointwise(prior, post_dis, a_star))
        kl_distr.append(kl(post_dis, prior))
    pmi_rel = np.array(pmi_rel)
    pmi_distr = np.array(pmi_distr)
    return {
        "pmi_rel": pmi_rel, "pmi_distr": pmi_distr, "kl_distr": np.array(kl_distr),
        "mean_rel": float(np.mean(pmi_rel)), "mean_distr": float(np.mean(pmi_distr)),
        "frac_distr_negative": float(np.mean(pmi_distr < 0)),
    }


# =========================================================================== #
# Movement 4 — diminishing returns: a redundant second document adds almost nothing.
# =========================================================================== #

def saturation_table(corpus: dict, tau: float = TAU, tau_doc: float = TAU_DOC) -> dict:
    """Belief-movement (KL, bits) of documents added in sequence, averaged over queries:
      standalone   = KL(post(d_gold) || prior)              — the gold filing's first-document value
      redundant    = KL(post(d_gold, d_gold) || post(d_gold)) — a second IDENTICAL filing
      novel        = KL(post(d_gold, d_other) || post(d_gold)) — a DIFFERENT filing
    The diminishing-returns headline: redundant << standalone and redundant << novel."""
    P, protos = corpus["P"], corpus["protos"]
    standalone, redundant, novel = [], [], []
    for qi in range(corpus["n_queries"]):
        qd = query_distributions(corpus, qi, tau, tau_doc)
        prior = qd["prior"]
        a_star = int(corpus["truth"][qi])
        d_other = _distractor_id(corpus, qi)
        post1 = answer_posterior(qd["q"], P[a_star], protos, tau)
        post_rr = answer_posterior_two(qd["q"], P[a_star], P[a_star], protos, tau)
        post_rn = answer_posterior_two(qd["q"], P[a_star], P[d_other], protos, tau)
        standalone.append(kl(post1, prior))
        redundant.append(kl(post_rr, post1))
        novel.append(kl(post_rn, post1))
    return {
        "standalone": float(np.mean(standalone)),
        "redundant": float(np.mean(redundant)),
        "novel": float(np.mean(novel)),
        "standalone_arr": np.array(standalone),
        "redundant_arr": np.array(redundant),
        "novel_arr": np.array(novel),
    }


# =========================================================================== #
# Movement 5 — the encoder maximized a lower bound on these bits; bits != recall.
# =========================================================================== #

def retrieval_qd_mi(corpus: dict, tau_doc: float = TAU_DOC) -> float:
    """The retrieval channel's mutual information I(Q;D) in bits, for uniform queries and the softmax
    retrieval distribution: I(Q;D) = E_q KL(p(d|q) || p_bar(d)), p_bar(d) = mean_q p(d|q). This is the
    true MI the InfoNCE objective lower-bounds."""
    Q, P = corpus["Q"], corpus["P"]
    pdq = np.array([retrieval_dist(q, P, tau_doc) for q in Q])
    p_bar = pdq.mean(axis=0)
    return float(np.mean([kl(pdq[i], p_bar) for i in range(len(Q))]))


def infonce_bound_curve(corpus: dict, tau: float = TAU,
                        sizes: tuple[int, ...] | None = None) -> dict:
    """The InfoNCE lower bound on I(Q;D) as the number of in-batch candidates grows. For a sub-batch of m
    (query, gold-filing) pairs we IMPORT `info_nce_loss_batch` and form bound_m = log2(m) - L_m/ln2, which
    is ceilinged at log2(m) (the load-bearing saturation: more negatives raise the ceiling). The encoder
    minimizing L maximizes this bound on the same I(Q;D) the retrieval channel realizes."""
    P, Q, truth, K = corpus["P"], corpus["Q"], corpus["truth"], corpus["K"]
    # One query per company (the first query whose gold answer is that company) — a clean B = K batch.
    first_q = np.array([Q[int(np.argmax(truth == j))] for j in range(K)])
    if sizes is None:
        sizes = tuple(range(2, K + 1))
    rows = []
    for m in sizes:
        Lm = info_nce_loss_batch(first_q[:m], P[:m], tau)        # nats
        rows.append({"m": int(m), "L": float(Lm),
                     "ceiling_bits": float(math.log2(m)),
                     "bound_bits": float(math.log2(m) - Lm / LOG2)})
    return {"rows": rows, "I_QD_bits": retrieval_qd_mi(corpus)}


def bits_vs_recall_table(corpus: dict, tau: float = TAU, tau_doc: float = TAU_DOC,
                         k: int = 3) -> dict:
    """Bits-added is a different axis from recall. For each query we score the MIPS ranking over P with the
    IMPORTED set-metric estimators (recall@k and AP against the single gold answer) AND compute the bits
    retrieval actually adds for that query, the per-query conditional MI H(prior) - sum_d p(d|q) H(post_d).
    On this corpus the gold filing is always rank 1, so recall@k and AP are SATURATED at 1.0 for every
    query — yet the bits added vary widely: presence (recall) is not contribution (bits). Returns the
    per-query (recall, bits) pairs the Panel-D scatter bakes."""
    P = corpus["P"]
    recalls, aps, bits = [], [], []
    for qi in range(corpus["n_queries"]):
        qd = query_distributions(corpus, qi, tau, tau_doc)
        scores = dual_encoder_score(qd["q"], P)
        ranking = np.argsort(-scores, kind="stable").tolist()
        a_star = int(corpus["truth"][qi])
        recalls.append(recall_at_k(ranking, {a_star}, k))
        aps.append(average_precision(ranking, {a_star}))
        hp = entropy(qd["prior"])
        hc = sum(qd["pdq"][d] * entropy(qd["post"][d]) for d in range(corpus["n_docs"]))
        bits.append(hp - hc)
    recalls = np.array(recalls)
    bits = np.array(bits)
    return {
        "recall_summary": metric_summary(recalls), "map": float(np.mean(aps)),
        "recall_min": float(recalls.min()), "recall_max": float(recalls.max()),
        "bits_min": float(bits.min()), "bits_max": float(bits.max()),
        "bits_std": float(np.std(bits, ddof=1)),
        "recalls": recalls, "bits_arr": bits, "n_queries": corpus["n_queries"],
    }


# =========================================================================== #
# Worked-query selection (a representative sector-ambiguous query for Panels A/B).
# =========================================================================== #

def pick_worked_query(corpus: dict, tau: float = TAU, tau_doc: float = TAU_DOC) -> int:
    """The query whose prior entropy is closest to the MEDIAN prior entropy — a representative, genuinely
    uncertain prior (not the most/least ambiguous), so the prior->posterior morph is illustrative."""
    h = np.array([entropy(query_distributions(corpus, qi, tau, tau_doc)["prior"])
                  for qi in range(corpus["n_queries"])])
    return int(np.argmin(np.abs(h - np.median(h))))


# =========================================================================== #
# viz_constants — every number PMIRetrievalValueLaboratory.tsx mirrors to the decimal.
# =========================================================================== #

def viz_constants() -> None:
    corpus = _corpus()
    wq = pick_worked_query(corpus)
    qd = query_distributions(corpus, wq)
    mi = cond_mi_breakdown(corpus)
    sgn = pmi_at_truth_table(corpus)
    sat = saturation_table(corpus)
    inf = infonce_bound_curve(corpus)
    bvr = bits_vs_recall_table(corpus)

    print("\n=== shared constants ===")
    print(f"K = {corpus['K']}  N_DOCS = {corpus['n_docs']}  N_QUERIES = {corpus['n_queries']}  "
          f"WORKED_Q = {wq}  TAU = {TAU}  TAU_DOC = {TAU_DOC}  KAPPA_QUERY = {PMI_KAPPA_QUERY}")
    print(f"SECTOR_OF_PASSAGE = {[int(s) for s in corpus['sector_of_passage']]}")

    print("\n=== Panel A — prior -> posterior on the worked query (TS recomputes entropy/pmi) ===")
    a_star = int(corpus["truth"][wq])
    d_id = _distractor_id(corpus, wq)
    print(f"  A_STAR = {a_star}  DISTRACTOR = {d_id}  Q_SECTOR = {int(corpus['q_sector'][wq])}")
    print(f"  PRIOR = {[round(float(v), 4) for v in qd['prior']]}")
    print(f"  H_PRIOR = {round(entropy(qd['prior']), 4)} bits")
    for tag, d in (("relevant", a_star), ("distractor", d_id)):
        post = qd["post"][d]
        print(f"  POST_{tag} = {[round(float(v), 4) for v in post]}  "
              f"H = {round(entropy(post), 4)}  bits_removed = {round(entropy(qd['prior']) - entropy(post), 4)}")
    print(f"  PDQ = {[round(float(v), 4) for v in qd['pdq']]}")
    print(f"  POST_ALL (one row per candidate doc d, p(a|q,d)) =")
    for d in range(corpus["n_docs"]):
        print(f"    d{d}: {[round(float(v), 4) for v in qd['post'][d]]}")

    print("\n=== Panel A readout — the three-way-verified I(A;D|Q) ===")
    print(f"  H_A_given_Q = {round(mi['H_A_given_Q'], 4)}  H_A_given_QD = {round(mi['H_A_given_QD'], 4)}")
    print(f"  I_ADQ: expected_kl={round(mi['expected_kl'], 6)}  "
          f"entropy_reduction={round(mi['entropy_reduction'], 6)}  joint_sum={round(mi['joint_sum'], 6)}")
    print(f"  max pairwise disagreement = "
          f"{max(abs(mi['expected_kl'] - mi['entropy_reduction']), abs(mi['entropy_reduction'] - mi['joint_sum'])):.2e}")

    print("\n=== Panel B — pmi at the true answer, relevant vs distractor ===")
    print(f"  MEAN_PMI_REL = {round(sgn['mean_rel'], 4)}  MEAN_PMI_DISTR = {round(sgn['mean_distr'], 4)}  "
          f"FRAC_DISTR_NEG = {round(sgn['frac_distr_negative'], 3)}")
    print(f"  PMI_REL = {[round(float(v), 3) for v in sgn['pmi_rel']]}")
    print(f"  PMI_DISTR = {[round(float(v), 3) for v in sgn['pmi_distr']]}")

    print("\n=== Panel C — diminishing returns (belief movement in bits) ===")
    print(f"  STANDALONE = {round(sat['standalone'], 4)}  REDUNDANT = {round(sat['redundant'], 4)}  "
          f"NOVEL = {round(sat['novel'], 4)}")

    print("\n=== Panel D — InfoNCE bound saturation (left) + bits-vs-recall scatter (right) ===")
    print(f"  INFONCE_ROWS (m, L_nats, ceiling_bits=log2(m), bound_bits=log2(m)-L/ln2):")
    for r in inf["rows"]:
        print(f"    m={r['m']:2d}  L={round(r['L'], 4)}  ceiling={round(r['ceiling_bits'], 4)}  "
              f"bound={round(r['bound_bits'], 4)}")
    print(f"  RECALL@3 in [{round(bvr['recall_min'], 3)}, {round(bvr['recall_max'], 3)}]  "
          f"MAP = {round(bvr['map'], 4)}  (recall SATURATED)")
    print(f"  BITS_ADDED in [{round(bvr['bits_min'], 4)}, {round(bvr['bits_max'], 4)}]  "
          f"std = {round(bvr['bits_std'], 4)}  (bits VARY)")
    print(f"  SCATTER_RECALL = {[round(float(v), 3) for v in bvr['recalls']]}")
    print(f"  SCATTER_BITS = {[round(float(v), 4) for v in bvr['bits_arr']]}")


# =========================================================================== #
# Verification harness — every pedagogical claim is an assert.
# =========================================================================== #

def test_pmi_sign_separation() -> None:
    # THE headline (runs first): a relevant filing adds bits at the true answer, a same-sector distractor
    # COSTS bits there, yet the distractor still moves belief (KL > 0) — toward the wrong company.
    c = _corpus()
    t = pmi_at_truth_table(c)
    assert t["mean_rel"] > 0.0 > t["mean_distr"], t
    assert t["mean_rel"] - t["mean_distr"] > 0.3, t            # a clear separation, not a hair
    assert t["frac_distr_negative"] > 0.5, t                  # most distractors cost bits at a*
    kl_distr = pmi_at_truth_table(c)["kl_distr"]
    assert float(np.mean(kl_distr)) > 0.0, kl_distr           # the distractor still moves the posterior


def test_three_way_mi_agreement() -> None:
    # The rigorous backbone: I(A;D|Q) computed three ways agrees to numerical zero, is >= 0, and H drops.
    c = _corpus()
    b = cond_mi_breakdown(c)
    vals = (b["expected_kl"], b["entropy_reduction"], b["joint_sum"])
    assert max(vals) - min(vals) < 1e-9, b
    assert b["entropy_reduction"] > 0.0, b
    assert b["H_A_given_Q"] > b["H_A_given_QD"], b


def test_prior_is_marginal_of_posterior() -> None:
    # The prior IS the RAG marginal sum_d p(d|q) p(a|q,d): the user-written answer_prior matches the
    # hot-path marginal exactly (this equality is what makes the MI identity close).
    c = _corpus()
    P, protos = c["P"], c["protos"]
    for qi in range(0, c["n_queries"], 3):
        qd = query_distributions(c, qi)
        prior_fn = answer_prior(qd["q"], P, protos, TAU, TAU_DOC)
        assert np.max(np.abs(prior_fn - qd["prior"])) < 1e-12, qi
        assert abs(float(prior_fn.sum()) - 1.0) < 1e-12, qi


def test_prior_genuinely_uncertain() -> None:
    # Guards headline-2 vacuity: if the prior were already peaked there would be no bits to add. The
    # sector-ambiguous queries keep the mean prior entropy substantial and the posterior much lower.
    c = _corpus()
    b = cond_mi_breakdown(c)
    assert b["H_A_given_Q"] > 0.5, b                          # >~ half a bit of answer uncertainty
    assert b["H_A_given_QD"] < b["H_A_given_Q"] - 0.2, b      # retrieval removes a real chunk of it


def test_corpus_reuses_dpr_geometry() -> None:
    # Reuse anchor: the answer prototypes ARE the dense-retrieval finance filings, and the vMF sizing
    # holds — same-sector companies are genuinely near, cross-sector near-orthogonal (kappa rule).
    c = _corpus()
    _, P_ref, _, sec_ref = dpr_finance_matrix(seed=PMI_SEED)
    assert np.allclose(c["P"], P_ref) and np.array_equal(c["sector_of_passage"], sec_ref)
    sec = c["sector_of_passage"]
    same, cross = [], []
    K = c["K"]
    for i in range(K):
        for j in range(i + 1, K):
            (same if sec[i] == sec[j] else cross).append(float(c["P"][i] @ c["P"][j]))
    assert np.mean(same) > np.mean(cross) + 0.2, (np.mean(same), np.mean(cross))
    # A_d sanity: the documented concentration gives a positive mean resultant length.
    assert mean_resultant_length(PMI_DIM, 60.0) > mean_resultant_length(PMI_DIM, 30.0) > 0.0


def test_saturation_diminishing_returns() -> None:
    # A second identical filing barely moves belief; a different filing moves it a lot.
    c = _corpus()
    s = saturation_table(c)
    assert s["redundant"] < s["standalone"], s
    assert s["redundant"] < s["novel"], s
    assert s["redundant"] >= 0.0 and s["standalone"] > 0.0, s


def test_infonce_bound_saturates() -> None:
    # The InfoNCE bound is ceilinged at log2(m): bound_m <= ceiling_m for every sub-batch (the load-bearing
    # saturation), the loss is positive, and adding negatives raises the ceiling. The retrieval channel's
    # actual I(Q;D) is non-negative.
    c = _corpus()
    inf = infonce_bound_curve(c)
    for r in inf["rows"]:
        assert r["L"] > 0.0, r
        assert r["bound_bits"] <= r["ceiling_bits"] + 1e-9, r
    ceilings = [r["ceiling_bits"] for r in inf["rows"]]
    assert all(ceilings[i] <= ceilings[i + 1] + 1e-12 for i in range(len(ceilings) - 1)), ceilings
    assert inf["I_QD_bits"] >= 0.0, inf["I_QD_bits"]


def test_bits_vs_recall_differ() -> None:
    # RUN, not assumed: recall and AP are SATURATED at 1.0 (the gold filing is always retrieved), a
    # degenerate quality axis on this corpus — yet the bits retrieval adds vary widely across queries, so
    # bits-added is a genuinely different axis from recall: presence is not contribution.
    c = _corpus()
    bvr = bits_vs_recall_table(c)
    assert bvr["recall_min"] == bvr["recall_max"] == 1.0, bvr        # recall sees every query as perfect
    assert abs(bvr["map"] - 1.0) < 1e-12, bvr
    assert bvr["bits_max"] - bvr["bits_min"] > 0.5, bvr              # yet the bits added vary by > half a bit
    assert 0.0 <= bvr["recall_summary"]["mean"] <= 1.0, bvr


def test_kl_and_mi_nonneg() -> None:
    c = _corpus()
    for qi in range(0, c["n_queries"], 4):
        qd = query_distributions(c, qi)
        for d in range(c["n_docs"]):
            assert kl(qd["post"][d], qd["prior"]) >= -1e-12
    assert cond_mi(c) >= 0.0


def test_guards() -> None:
    # Degenerate distributions and temperatures.
    assert entropy(np.array([1.0, 0.0, 0.0])) == 0.0
    assert entropy(np.array([])) == 0.0
    assert kl(np.array([1.0, 0.0]), np.array([0.5, 0.5])) >= 0.0
    for bad in (0.0, -1.0, 1e-9):
        try:
            retrieval_dist(np.ones(PMI_DIM), _corpus()["P"], bad)
            assert False, "tau_doc <= 1e-8 should raise"
        except ValueError:
            pass
    # The imported set-metric guards (empty relevant set, k <= 0) still hold here.
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0
    assert average_precision([1, 2, 3], set()) == 0.0


def _run_all() -> None:
    print("pmi_retrieval_value — verifying every claim:")
    test_pmi_sign_separation()             # the headline runs first
    test_three_way_mi_agreement()
    test_prior_is_marginal_of_posterior()
    test_prior_genuinely_uncertain()
    test_corpus_reuses_dpr_geometry()
    test_saturation_diminishing_returns()
    test_infonce_bound_saturates()
    test_bits_vs_recall_differ()
    test_kl_and_mi_nonneg()
    test_guards()
    print("all pmi-retrieval-value tests passed")
    viz_constants()


if __name__ == "__main__":
    _run_all()
