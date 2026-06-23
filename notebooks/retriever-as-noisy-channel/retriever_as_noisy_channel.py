"""The retriever as a noisy channel — recall, precision, and the information limits of generation, in BITS.

The reference implementation for the formalRAG `retriever-as-noisy-channel` topic, the second node in the
information-theory layer. The PMI topic (`pmi-retrieval-value`) measured how many BITS a retrieved document
adds to the generator's answer — the conditional mutual information I(A;D|Q) = H(A|Q) - H(A|Q,D). This topic
reads that same machinery as a COMMUNICATION CHANNEL: a query enters, the retriever emits a context, and the
generator decodes an answer. The subtitle's two words are the channel's two failure modes, and they are
CO-EQUAL headlines:

  RECALL  ==  ERASURE.  Drop the relevant filing (probability 1 - recall) and the context is erased to a
    non-informative belief — the RAG premise that without the document the generator knows nothing. The bits
    delivered fall EXACTLY as  I_eps = recall * I_0  (the capacity of a Binary Erasure Channel is its
    surviving fraction, here recall). What is lost reappears as residual entropy H(A|Q,D), which rises toward
    log2(K); and FANO'S INEQUALITY turns that residual into a floor on the generator's Bayes (MAP) error:

        P_e  >=  ( H(A|Q,D) - 1 ) / log2(K)        (loose)
        H(A|Q,D)  <=  H_b(P_e) + P_e * log2(K-1)    (tight, inverted numerically)

    No generator, however good, beats this floor. Recall sets the floor.

  PRECISION  ==  SUBSTITUTION.  Return a plausible same-sector distractor (probability eps) and the generator
    reads it and answers CONFIDENTLY WRONG: H(A|Q,D) stays LOW (the model is sure), so Fano's entropy floor
    sees nothing, yet the realized error against the gold label climbs toward 1. The gap between realized
    error and the Bayes floor is precisely what an entropy bound cannot see. Precision governs whether you
    HIT the floor; closing the realized-vs-Bayes gap is the job of the faithfulness / calibration layer.

That last sentence is the load-bearing honesty (rigorFlag territory): Fano bounds the model's BAYES error via
the residual entropy; it does NOT bound confident contamination. The realized-vs-Bayes gap is the forward
edge to `faithfulness-groundedness`, `context-selection-submodular-dpp`, and the calibration topic — named in
prose, not linked.

Movements, every pedagogical claim an assert:

  MOVEMENT 0 — the channel. We REUSE the PMI corpus verbatim (`pmi_corpus`: four sectors, two companies each,
    ONE filing per company, so K = 8 answers = documents; 32 sector-ambiguous queries; TAU = TAU_DOC = 0.2).
    We never rebuild the answer model — `retrieval_dist`, `answer_posterior`, `entropy`, `kl`,
    `cond_mi_breakdown` are imported. The two corruptions (erasure, substitution) are this topic's only new
    machinery.

  MOVEMENT 1 — recall = erasure = BEC. `bec_capacity(recall) = recall`; the delivered bits I_eps = recall*I_0
    are exact (a Q-measurable fallback carries zero conditional information). Residual entropy rises; the Fano
    floor activates once H(A|Q,D) crosses 1 bit and bounds the Bayes error (asserted as a THEOREM).

  MOVEMENT 2 — precision = substitution. `bsc_capacity(p) = 1 - H_b(p)`. The same-sector distractor drives
    realized error 0 -> 1 while H(A|Q,D) stays flat-low; the confident-wrong gap = realized - Bayes widens.

  MOVEMENT 3 — the synthesis. The retriever spends RATE (bits delivered) to buy down DISTORTION (answer
    error); recall and precision trace the operating window. Up-link formalML rate-distortion and
    information-bottleneck; name channel capacity / mutual information / the noisy-channel coding theorem in
    prose (formalML has no slug for them).

Honest caveats inherited from PMI: the answer model p(a|q,d) is a SYNTHETIC vMF/softmax stand-in for a real
LLM generator, so every "bit" is exact for the model and only illustrative of a deployed generator; the gold
is rank-1 on this easy corpus, so recall@k is saturated and the precision panel measures contamination as
ENTROPY ADDED, not an error U-curve. `viz_constants()` prints what `RetrieverAsNoisyChannelLaboratory.tsx`
mirrors to the decimal.

Run:  uv run --with numpy --with scipy \\
        python notebooks/retriever-as-noisy-channel/retriever_as_noisy_channel.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Import the published stack. The direct prereq (pmi-retrieval-value) is ITSELF a dependent topic, so we add
# every ancestor's HYPHENATED dir to the path and import the UNDERSCORED module. We reuse the PMI answer model
# (retrieval distribution, answer posterior, entropy/KL in bits, the three-way MI breakdown, the same-sector
# distractor pick) and the set-metric estimators; we never reimplement any of them.
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "the-retrieval-problem",
    "infonce-contrastive-objective",
    "dense-retrieval-dual-encoders",
    "set-metrics-precision-recall-map-mrr",
    "pmi-retrieval-value",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pmi_retrieval_value import (                                                 # noqa: E402
    query_distributions, answer_posterior,
    entropy, kl, cond_mi_breakdown,
    _distractor_id, _corpus,
    TAU, TAU_DOC,
)
from set_metrics_precision_recall_map_mrr import recall_at_k, precision_at_k      # noqa: E402

# Module constants. The corpus, geometry, seed, and temperatures are inherited from PMI unchanged.
RNC_SEED = 7
EPS_GRID = [round(0.1 * i, 2) for i in range(11)]        # 0.0 .. 1.0
RECALL_GRID = [round(1.0 - 0.1 * i, 2) for i in range(11)]  # 1.0 .. 0.0 (recall falls left->right)
K_GRID = [1, 2, 3, 4, 5, 6, 7, 8]                        # docs read for the precision-entropy frontier


# =========================================================================== #
# Channel-capacity primitives (textbook, in bits).
# =========================================================================== #

def binary_entropy(p: float) -> float:
    """H_b(p) = -p log2 p - (1-p) log2(1-p) in BITS; H_b(0) = H_b(1) = 0."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def bec_capacity(recall: float) -> float:
    """Capacity of a Binary Erasure Channel with erasure probability (1 - recall): C = 1 - eps = recall.

    The interpretation that names this topic: a retrieval miss ERASES the relevant document, and the surviving
    fraction of the relevant bits is exactly the recall. So the bits the channel delivers about the answer are
    recall * I_0 (see `recall_sweep`). GUARD: recall in [0, 1]."""
    if not (0.0 <= recall <= 1.0):
        raise ValueError(f"recall must be in [0, 1], got {recall}")
    return float(recall)


def bsc_capacity(p: float) -> float:
    """Capacity of a Binary Symmetric Channel with crossover (substitution) probability p: C = 1 - H_b(p).

    The precision-side reading: a false positive SUBSTITUTES a wrong document for the right one, a crossover.
    bsc_capacity(0) = 1 (a clean channel passes the full bit); bsc_capacity(0.5) = 0 (a coin flip passes
    nothing). GUARD: p in [0, 1]."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    return float(1.0 - binary_entropy(p))


# =========================================================================== #
# Fano's inequality: the residual entropy is a floor on the Bayes (MAP) error.
# =========================================================================== #

def fano_floor_loose(h_cond: float, K: int) -> float:
    """The loose Fano lower bound on the MAP error: P_e >= (H(A|Q,D) - 1) / log2(K), clipped at 0.

    Derived from H(A|Q,D) <= 1 + P_e log2(K) (bounding H_b(P_e) <= 1 and log2(K-1) <= log2(K)). Vacuous (= 0)
    while the residual entropy is below 1 bit — exactly the regime of a well-retrieving channel. GUARD: K>=2."""
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    return max(0.0, (h_cond - 1.0) / math.log2(K))


def fano_floor_tight(h_cond: float, K: int) -> float:
    """The tight Fano lower bound: the smallest P_e in [0, 1 - 1/K] solving H(A|Q,D) <= H_b(P_e) + P_e log2(K-1).

    The RHS rises monotonically from 0 (at P_e = 0) to log2(K) (at P_e = 1 - 1/K), so a bisection finds the
    floor. Always >= the loose floor. GUARD: K>=2; clamp h_cond to [0, log2(K)]."""
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    log2K = math.log2(K)
    h = min(max(h_cond, 0.0), log2K)
    if h <= 0.0:
        return 0.0
    pe_max = 1.0 - 1.0 / K
    log2Km1 = math.log2(K - 1) if K > 2 else 0.0   # K == 2: the log2(K-1) term vanishes

    def rhs(pe: float) -> float:
        return binary_entropy(pe) + pe * log2Km1

    lo, hi = 0.0, pe_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rhs(mid) < h:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def bayes_error(belief: np.ndarray) -> float:
    """The MAP error of a single belief: 1 - max_a p(a). The error a generator incurs by committing to the
    most probable answer UNDER ITS OWN POSTERIOR — the quantity Fano bounds (averaged over queries)."""
    return float(1.0 - np.asarray(belief, dtype=float).max())


# =========================================================================== #
# Movement 0/1 — the clean channel's per-query quantities, reused by both sweeps.
# =========================================================================== #

def _clean_channel_terms(corpus: dict, tau: float = TAU, tau_doc: float = TAU_DOC) -> dict:
    """Average over queries the clean (recall = 1) single-document channel quantities:
      H_cond0 = E_q E_{d|q}[ H(post_d) ]   — the residual entropy that goes in Fano,
      bayes0  = E_q E_{d|q}[ 1 - max post_d ] — the clean MAP error,
    and read I_0 (the delivered bits) and H(A|Q) straight from the IMPORTED three-way MI breakdown. H_cond0
    must equal the breakdown's H_A_given_QD bit-for-bit (the collapse anchor)."""
    n = corpus["n_docs"]
    hc, bz = [], []
    for qi in range(corpus["n_queries"]):
        qd = query_distributions(corpus, qi, tau, tau_doc)
        pdq, post = qd["pdq"], qd["post"]
        hc.append(sum(pdq[d] * entropy(post[d]) for d in range(n)))
        bz.append(sum(pdq[d] * float(1.0 - post[d].max()) for d in range(n)))
    br = cond_mi_breakdown(corpus, tau, tau_doc)
    return {
        "H_cond0": float(np.mean(hc)),
        "bayes0": float(np.mean(bz)),
        "I0": float(br["entropy_reduction"]),
        "H_A_given_Q": float(br["H_A_given_Q"]),
        "H_A_given_QD": float(br["H_A_given_QD"]),
    }


def recall_sweep(corpus: dict, recalls=tuple(RECALL_GRID),
                 tau: float = TAU, tau_doc: float = TAU_DOC) -> dict:
    """The recall = erasure sweep. With probability `recall` the generator reads the retrieved context; with
    probability 1 - recall the context is erased to the uniform belief (no document, no knowledge). Because the
    erased symbol is independent of the answer given the query, the channel is a Binary Erasure Channel and
    every quantity is a CONVEX BLEND of the clean value and the erased value:

        I_eps    = recall * I_0                                        (BEC capacity * the clean bits)
        H(A|Q,D) = recall * H_cond0 + (1 - recall) * log2(K)          (rises toward log2 K)
        bayes    = recall * bayes0  + (1 - recall) * (1 - 1/K)        (rises toward the uniform-guess error)

    The Fano floor on H(A|Q,D) bounds the Bayes error at every recall (the theorem)."""
    t = _clean_channel_terms(corpus, tau, tau_doc)
    K = corpus["K"]
    unif_H = math.log2(K)
    unif_bayes = 1.0 - 1.0 / K
    rows = []
    for r in recalls:
        r = float(r)
        h_cond = r * t["H_cond0"] + (1.0 - r) * unif_H
        bz = r * t["bayes0"] + (1.0 - r) * unif_bayes
        i_eps = r * t["I0"]
        rows.append({
            "recall": r,
            "I": i_eps,
            "H_cond": h_cond,
            "bayes": bz,
            "fano_loose": fano_floor_loose(h_cond, K),
            "fano_tight": fano_floor_tight(h_cond, K),
        })
    return {"rows": rows, **t, "unif_H": unif_H, "unif_bayes": unif_bayes, "K": int(K)}


# =========================================================================== #
# Movement 2 — precision = substitution: the confident-wrong gap.
# =========================================================================== #

def _substitution_terms(corpus: dict, tau: float = TAU, tau_doc: float = TAU_DOC) -> dict:
    """Per-query gold-vs-distractor read quantities, averaged over queries. The generator reads ONE passage in
    the relevant slot: the gold filing P[a*] (clean) or the same-sector distractor P[_distractor_id] (the
    false positive). Both posteriors are SHARP — the distractor one just peaks on the wrong company."""
    P, protos = corpus["P"], corpus["protos"]
    g_err, d_err, g_H, d_H, g_bz, d_bz = [], [], [], [], [], []
    for qi in range(corpus["n_queries"]):
        q = corpus["Q"][qi]
        a_star = int(corpus["truth"][qi])
        did = _distractor_id(corpus, qi)
        bg = answer_posterior(q, P[a_star], protos, tau)
        bd = answer_posterior(q, P[did], protos, tau)
        g_err.append(0.0 if int(bg.argmax()) == a_star else 1.0)
        d_err.append(0.0 if int(bd.argmax()) == a_star else 1.0)
        g_H.append(entropy(bg)); d_H.append(entropy(bd))
        g_bz.append(bayes_error(bg)); d_bz.append(bayes_error(bd))
    return {
        "g_err": float(np.mean(g_err)), "d_err": float(np.mean(d_err)),
        "g_H": float(np.mean(g_H)), "d_H": float(np.mean(d_H)),
        "g_bayes": float(np.mean(g_bz)), "d_bayes": float(np.mean(d_bz)),
    }


def substitution_sweep(corpus: dict, epss=tuple(EPS_GRID),
                       tau: float = TAU, tau_doc: float = TAU_DOC) -> dict:
    """The precision = substitution sweep. With probability eps the gold filing is replaced by a same-sector
    distractor before the generator reads it. Realized error (vs the gold label) is a convex blend that climbs
    toward 1; the residual entropy H(A|Q,D) and the Bayes error stay flat-LOW (both passages are confident);
    so the Fano floor barely moves while realized error explodes — the confident-wrong gap = realized - Bayes."""
    s = _substitution_terms(corpus, tau, tau_doc)
    K = corpus["K"]
    rows = []
    for e in epss:
        e = float(e)
        realized = (1.0 - e) * s["g_err"] + e * s["d_err"]
        bz = (1.0 - e) * s["g_bayes"] + e * s["d_bayes"]
        h_cond = (1.0 - e) * s["g_H"] + e * s["d_H"]
        rows.append({
            "eps": e,
            "realized": realized,
            "bayes": bz,
            "H_cond": h_cond,
            "fano_loose": fano_floor_loose(h_cond, K),
            "fano_tight": fano_floor_tight(h_cond, K),
            "gap": realized - bz,
        })
    return {"rows": rows, **s, "K": int(K)}


# =========================================================================== #
# Movement 3 — the precision-vs-entropy frontier (contamination as bits of uncertainty added).
# =========================================================================== #

def precision_entropy_frontier(corpus: dict, ks=tuple(K_GRID),
                               tau: float = TAU, tau_doc: float = TAU_DOC) -> list:
    """For each context size k: recall@k (gold filing among the top-k retrieved), precision = 1/k (one
    relevant doc), and the residual entropy of the top-k blended belief. On this easy corpus the gold is
    rank-1, so recall@k is saturated at 1 and the answer-error U-curve is vacuous; the HONEST frontier is
    precision falling while the blended belief's entropy RISES — contamination measured in bits, not errors."""
    rows = []
    for k in ks:
        rec, prec, hs = [], [], []
        for qi in range(corpus["n_queries"]):
            qd = query_distributions(corpus, qi, tau, tau_doc)
            pdq, post = qd["pdq"], qd["post"]
            order = [int(d) for d in np.argsort(-pdq)]      # docs ranked by retrieval probability
            a_star = int(corpus["truth"][qi])
            kk = min(k, corpus["n_docs"])
            rec.append(recall_at_k(order, {a_star}, kk))
            prec.append(precision_at_k(order, {a_star}, kk))
            top = order[:kk]
            w = pdq[top]
            w = w / w.sum()
            belief = w @ post[top]
            hs.append(entropy(belief))
        rows.append({
            "k": int(k),
            "recall": float(np.mean(rec)),
            "precision": float(np.mean(prec)),
            "H": float(np.mean(hs)),
        })
    return rows


# =========================================================================== #
# viz_constants — every number RetrieverAsNoisyChannelLaboratory.tsx mirrors to the decimal.
# =========================================================================== #

WORKED_Q = 5            # a representative query whose clean answer belief Panel A erodes toward uniform


def viz_constants() -> None:
    corpus = _corpus(RNC_SEED)
    rsw = recall_sweep(corpus)
    ssw = substitution_sweep(corpus)
    fr = precision_entropy_frontier(corpus)
    worked = query_distributions(corpus, WORKED_Q)["prior"]   # the RAG marginal answer belief, clean channel

    print("\n=== shared constants (TS recomputes capacities / Fano floors closed-form) ===")
    print(f"K = {corpus['K']}  N_DOCS = {corpus['n_docs']}  N_QUERIES = {corpus['n_queries']}  "
          f"TAU = {TAU}  TAU_DOC = {TAU_DOC}")
    print(f"I0 = {round(rsw['I0'], 4)} bits   H_A_given_Q = {round(rsw['H_A_given_Q'], 4)}   "
          f"H_cond0 = {round(rsw['H_cond0'], 4)}   bayes0 = {round(rsw['bayes0'], 4)}")
    print(f"LOG2K = {round(math.log2(corpus['K']), 4)}   UNIF_BAYES = {round(rsw['unif_bayes'], 4)}")
    print(f"WORKED_Q = {WORKED_Q}   WORKED_BELIEF = {[round(float(v), 4) for v in worked]}")

    print("\n=== Panel A/B — recall = erasure sweep (BEC). Continuous TS: H=r*H_cond0+(1-r)*LOG2K, etc. ===")
    print("  RECALL_GRID = " + str([r["recall"] for r in rsw["rows"]]))
    print("  I_BITS      = " + str([round(r["I"], 4) for r in rsw["rows"]]))
    print("  H_COND      = " + str([round(r["H_cond"], 4) for r in rsw["rows"]]))
    print("  BAYES       = " + str([round(r["bayes"], 4) for r in rsw["rows"]]))
    print("  FANO_LOOSE  = " + str([round(r["fano_loose"], 4) for r in rsw["rows"]]))
    print("  FANO_TIGHT  = " + str([round(r["fano_tight"], 4) for r in rsw["rows"]]))
    cross = next((r["recall"] for r in rsw["rows"] if r["fano_loose"] > 0), None)
    print(f"  FANO_ACTIVATES_AT_RECALL = {cross}  (first recall where H_cond crosses 1 bit)")

    print("\n=== Panel C — precision = substitution sweep. Continuous TS blends gold/distractor means ===")
    print(f"  G_ERR = {round(ssw['g_err'], 4)}  D_ERR = {round(ssw['d_err'], 4)}  "
          f"G_H = {round(ssw['g_H'], 4)}  D_H = {round(ssw['d_H'], 4)}  "
          f"G_BAYES = {round(ssw['g_bayes'], 4)}  D_BAYES = {round(ssw['d_bayes'], 4)}")
    print("  EPS_GRID  = " + str([r["eps"] for r in ssw["rows"]]))
    print("  REALIZED  = " + str([round(r["realized"], 4) for r in ssw["rows"]]))
    print("  SUB_BAYES = " + str([round(r["bayes"], 4) for r in ssw["rows"]]))
    print("  SUB_HCOND = " + str([round(r["H_cond"], 4) for r in ssw["rows"]]))
    print("  SUB_FANO  = " + str([round(r["fano_loose"], 4) for r in ssw["rows"]]))
    print("  SUB_GAP   = " + str([round(r["gap"], 4) for r in ssw["rows"]]))

    print("\n=== Panel D — precision-vs-entropy frontier (recall@k saturated; contamination = bits added) ===")
    print("  K_GRID    = " + str([r["k"] for r in fr]))
    print("  RECALL_AT_K = " + str([round(r["recall"], 4) for r in fr]))
    print("  PRECISION = " + str([round(r["precision"], 4) for r in fr]))
    print("  FRONTIER_H = " + str([round(r["H"], 4) for r in fr]))


# =========================================================================== #
# Harness — every pedagogical claim an assert.
# =========================================================================== #

def test_collapse_to_prereq() -> None:
    """ALWAYS-TRUE anchor: at recall = 1 the noisy channel IS the PMI channel. The clean residual entropy and
    the delivered bits are bit-identical to the imported three-way MI breakdown."""
    corpus = _corpus(RNC_SEED)
    t = _clean_channel_terms(corpus)
    br = cond_mi_breakdown(corpus)
    assert abs(t["H_cond0"] - br["H_A_given_QD"]) < 1e-12, t["H_cond0"]
    assert abs(t["I0"] - br["entropy_reduction"]) < 1e-12, t["I0"]
    rsw = recall_sweep(corpus, recalls=[1.0])
    row = rsw["rows"][0]
    assert abs(row["H_cond"] - br["H_A_given_QD"]) < 1e-12
    assert abs(row["I"] - br["entropy_reduction"]) < 1e-12


def test_fano_is_a_theorem() -> None:
    """ALWAYS-TRUE anchor: at every operating point of both channels, bayes >= fano_tight >= fano_loose."""
    corpus = _corpus(RNC_SEED)
    for sweep in (recall_sweep(corpus), substitution_sweep(corpus)):
        for r in sweep["rows"]:
            assert r["bayes"] >= r["fano_tight"] - 1e-9, r
            assert r["fano_tight"] >= r["fano_loose"] - 1e-9, r


def test_entropy_identity() -> None:
    """ALWAYS-TRUE anchor: per query, H(A|Q) = H(A|Q,D) + I(A;D|Q) — the Jensen gap between the marginal
    belief's entropy and the average per-document confidence IS the bits the document delivers."""
    corpus = _corpus(RNC_SEED)
    n = corpus["n_docs"]
    for qi in range(corpus["n_queries"]):
        qd = query_distributions(corpus, qi)
        pdq, post, prior = qd["pdq"], qd["post"], qd["prior"]
        h_marg = entropy(prior)
        h_cond = sum(pdq[d] * entropy(post[d]) for d in range(n))
        i_q = sum(pdq[d] * kl(post[d], prior) for d in range(n))
        assert abs(h_marg - (h_cond + i_q)) < 1e-12, (qi, h_marg, h_cond, i_q)


def test_bec_capacity_linear() -> None:
    """ALWAYS-TRUE anchor: the delivered bits are EXACTLY recall * I_0 (the BEC reading of recall)."""
    corpus = _corpus(RNC_SEED)
    rsw = recall_sweep(corpus)
    I0 = rsw["I0"]
    for r in rsw["rows"]:
        assert abs(r["I"] - bec_capacity(r["recall"]) * I0) < 1e-12, r
    assert bec_capacity(1.0) == 1.0 and bec_capacity(0.0) == 0.0


def test_capacity_and_fano_definitions() -> None:
    """ALWAYS-TRUE anchor: the channel-capacity twins and the Fano-floor inversion against their closed forms."""
    assert abs(bsc_capacity(0.0) - 1.0) < 1e-12
    assert abs(bsc_capacity(0.5) - 0.0) < 1e-12
    assert abs(binary_entropy(0.5) - 1.0) < 1e-12
    # The tight floor round-trips its defining equation H = H_b(Pe) + Pe*log2(K-1) for H strictly inside (0, log2 K).
    K = 8
    log2Km1 = math.log2(K - 1)
    for h in (0.5, 1.0, 1.5, 2.0, 2.5):
        pe = fano_floor_tight(h, K)
        assert abs((binary_entropy(pe) + pe * log2Km1) - h) < 1e-6, (h, pe)
        assert fano_floor_tight(h, K) >= fano_floor_loose(h, K) - 1e-9
    assert fano_floor_loose(0.8, K) == 0.0   # vacuous below 1 bit


def test_recall_degradation_monotone() -> None:
    """BUILD-AND-RUN gate: as recall falls, delivered bits fall, residual entropy and Bayes error rise, and
    the Fano floor is non-decreasing."""
    corpus = _corpus(RNC_SEED)
    rows = recall_sweep(corpus)["rows"]  # recall 1.0 -> 0.0
    for a, b in zip(rows, rows[1:]):
        assert b["I"] < a["I"] + 1e-12
        assert b["H_cond"] > a["H_cond"] - 1e-12
        assert b["bayes"] > a["bayes"] - 1e-12
        assert b["fano_loose"] >= a["fano_loose"] - 1e-12


def test_fano_activates() -> None:
    """BUILD-AND-RUN gate: the residual entropy crosses 1 bit inside the sweep, so the loose Fano floor goes
    from vacuous (recall = 1) to active (recall = 0)."""
    corpus = _corpus(RNC_SEED)
    rows = recall_sweep(corpus)["rows"]
    assert rows[0]["recall"] == 1.0 and rows[0]["H_cond"] < 1.0      # clean channel: floor vacuous
    assert rows[0]["fano_loose"] == 0.0
    assert rows[-1]["recall"] == 0.0 and rows[-1]["H_cond"] > 1.0    # full erasure: floor active
    assert rows[-1]["fano_loose"] > 0.0


def test_confident_wrong_gap_widens() -> None:
    """BUILD-AND-RUN gate: under substitution, realized error rises while the model stays CONFIDENT (low
    H(A|Q,D)), so Fano is blind and the realized-vs-Bayes gap widens strictly with eps."""
    corpus = _corpus(RNC_SEED)
    rows = substitution_sweep(corpus)["rows"]  # eps 0.0 -> 1.0
    for a, b in zip(rows, rows[1:]):
        assert b["gap"] > a["gap"] - 1e-12               # gap widens
        assert b["realized"] > a["realized"] - 1e-12     # realized error climbs
    assert rows[-1]["realized"] > 0.5                     # confidently wrong at full substitution
    assert rows[-1]["H_cond"] < 1.0                       # ... yet the model stays sure: Fano floor blind
    assert rows[-1]["fano_loose"] == 0.0
    assert rows[-1]["gap"] > rows[0]["gap"]


def test_precision_entropy_frontier() -> None:
    """BUILD-AND-RUN gate: recall@k saturated at 1 (gold rank-1), precision = 1/k falling, blended-belief
    entropy rising — contamination as bits of uncertainty added, not an error U-curve."""
    corpus = _corpus(RNC_SEED)
    fr = precision_entropy_frontier(corpus)
    assert all(abs(r["recall"] - 1.0) < 1e-9 for r in fr)            # saturated recall
    for a, b in zip(fr, fr[1:]):
        assert b["precision"] < a["precision"] + 1e-12               # 1/k falls
        assert b["H"] > a["H"] - 1e-9                                # entropy rises with more docs
    assert all(abs(r["precision"] - 1.0 / r["k"]) < 1e-9 for r in fr)


def test_guards() -> None:
    """Boundary guards (the Gemini-flagged ones): capacities reject out-of-range, Fano needs K>=2, the
    imported set-metric estimators cap k > n and handle the empty relevant set, entropy/KL stay finite."""
    for bad in (-0.1, 1.1):
        for fn in (bec_capacity, bsc_capacity):
            try:
                fn(bad)
                assert False, "out-of-range probability should raise"
            except ValueError:
                pass
    for fn in (fano_floor_loose, fano_floor_tight):
        try:
            fn(1.5, 1)
            assert False, "K < 2 should raise"
        except ValueError:
            pass
    assert fano_floor_tight(5.0, 8) <= 1.0 - 1.0 / 8 + 1e-9          # clamps above log2 K
    assert recall_at_k([1, 2, 3], {1}, 99) == 1.0                   # k > n is harmless (full ranking)
    assert precision_at_k([1, 2, 3], set(), 3) == 0.0
    assert entropy(np.array([1.0, 0.0, 0.0])) == 0.0
    assert kl(np.array([1.0, 0.0]), np.array([0.5, 0.5])) >= 0.0
    assert binary_entropy(0.0) == 0.0 and binary_entropy(1.0) == 0.0


def _run_all() -> None:
    print("retriever_as_noisy_channel — verifying every claim:")
    # Always-true anchors first.
    test_collapse_to_prereq()
    test_fano_is_a_theorem()
    test_entropy_identity()
    test_bec_capacity_linear()
    test_capacity_and_fano_definitions()
    # Build-and-run directional gates.
    test_recall_degradation_monotone()
    test_fano_activates()
    test_confident_wrong_gap_widens()
    test_precision_entropy_frontier()
    test_guards()
    print("all retriever-as-noisy-channel tests passed")
    viz_constants()


if __name__ == "__main__":
    _run_all()
