"""Multi-hop and iterative retrieval as search over an evidence space.

The canonical, tested reference for the `multi-hop-iterative-retrieval` topic -- the bridge in the
"Information Theory of RAG" layer from the SINGLE retrieval step (PMI, noisy-channel,
retrieval-vs-long-context, context-selection) to retrieval as a SEARCH. Context selection was the last
topic to treat the candidate pool as fixed; it closed on the line "when one passage's answer raises a new
question, selection becomes a search over an evolving evidence space." This module makes that precise.

Four movements, the fourth the conceptual climax:

  1. THE COMPOSITIONAL GAP. A 2-hop question -- "the revenue of the company that acquired Company A's primary
     supplier" -- hides its answer in a document UNREACHABLE in one retrieval. On the embedding sphere the
     answer filing is near-orthogonal to the query (cosine ~ 0) yet a short reformulation away through a
     BRIDGE filing that names the supplier. We model the bridge as a "mention": A's filing sits mostly along
     A's direction but carries a sin(alpha) component toward the company B it names,
     f_A = cos(alpha) u_A + sin(alpha) u_B. The reformulation operator q' = normalize(d - <d,q> q) extracts
     EXACTLY that mention (it removes the part of the read filing already explained by the query), so
     reformulating from A's filing points retrieval at B. With near-orthogonal company directions the answer
     is invisible to the query (single-hop answer-recall ~ 0) but one hop away (2-hop ~ 1): the gap.
  2. SEARCH OVER AN EVIDENCE SPACE. State = a belief (posterior) over answer companies; action = a
     reformulated query; transition = the prereq's retrieve-and-select step; reward = information gained. The
     greedy hop maximizes expected marginal information; the optimal policy is a Bellman fixed point we name
     but do not solve (intractable over the belief simplex).
  3. COMPOUNDING RECALL. End-to-end success is the PRODUCT of per-hop recalls (geometric decay), so to hold a
     target rho each hop must over-retrieve to rho^(1/k); positive dependence between hops makes the
     independent product a CONSERVATIVE LOWER bound (FKG/Harris). We REUSE the capstone's cascade machinery
     (cascade_recall, over_fetch_factor, dependence_sweep) so the law is the same one, applied over hops.
  4. WHY GREEDY CANNOT SHORTCUT: THE SUPERMODULAR SYNERGY (the climax). Information gain is NOT submodular,
     and the compositional question is its SUPERMODULAR case -- the bridge alone says nothing about the
     answer, the answer document alone cannot be recognized as relevant, yet together they resolve it: the
     exact XOR witness context-selection used. We REUSE info_gain_xor_witness for the clean general proof,
     and DEMONSTRATE the operational consequence the corpus genuinely exhibits: single-shot selection on the
     query-reachable pool cannot reach the answer (it is below the edge threshold), so only reformulating
     from the bridge harvests the synergy.

The honest hinges (the rigorFlag): greedy hopping has no (1 - 1/e) guarantee (info gain is supermodular on
the compositional pair); the Bellman-optimal policy is intractable; the compounding-recall product holds
under independence and is a lower bound under positive dependence; the stopping rule is a decision-theoretic
motivation (an SPRT / optimal-stopping analogy), not a derived optimum, and the realized hop count is
seed-dependent; the answer model is a synthetic von Mises-Fisher softmax, exact for the model and
illustrative of a real retriever, on one constructed corpus engineered to have a near-orthogonal answer
reachable through a bridge. Every pedagogical claim is an `assert`; `viz_constants()` prints every number
the React lab mirrors.

Run:  uv run --with numpy --with scipy \\
          python notebooks/multi-hop-iterative-retrieval/multi_hop_iterative_retrieval.py

NOTE on imports (notebook import graph != pedagogical DAG): the frontmatter prerequisite is the single graph
edge context-selection-submodular-dpp. These imports SOURCE numbers and reuse verified routines on the one
shared finance cloud (seed 7): the answer model (answer_posterior / answer_posterior_topk / entropy / kl),
the geometry (sample_vmf / normalize / rvlc protos), the scorer (dual_encoder_score), recall_at_k, the
submodular greedy + the XOR witness (context-selection), and the cascade/FKG laws (the capstone -- a SIBLING
numeric source, NOT a prerequisite, so it stays out of frontmatter `prerequisites` and sits in connections).

VIZ <-> PYTHON INVARIANT: the constants printed by `viz_constants()` are mirrored to the decimal in
`src/components/viz/MultiHopLaboratory.tsx`. Change a number here, re-run, then update the .tsx -- never the
reverse. Recall / belief / loop outputs are MODEL OUTPUTS and are BAKED; only closed forms (the geometric
product rho^(1/k), the over-fetch reciprocal, the stopping-threshold crossing, pixel maps) recompute in TS.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.special import logsumexp  # never hand-roll softmax/sigmoid (overflow + a Gemini flag)

# --- import the prereq chain + the capstone (sibling numeric source for the cascade/FKG laws) ----------
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
    "context-selection-submodular-dpp",
    "product-quantization",
    "filtered-incremental-ann",
    "capstone-multimodal-financial-rag",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf  # noqa: E402
from dense_retrieval_dual_encoders import dual_encoder_score, DPR_SEED, DPR_DIM  # noqa: E402
from pmi_retrieval_value import answer_posterior, entropy, kl  # noqa: E402
from set_metrics_precision_recall_map_mrr import recall_at_k  # noqa: E402
from retrieval_vs_long_context import rvlc_corpus, answer_posterior_topk  # noqa: E402
from context_selection_submodular_dpp import (  # noqa: E402
    greedy_select, info_gain_fn, info_gain_xor_witness, submodularity_witness, PAYOFF_TAU,
)
from capstone_multimodal_financial_rag import (  # noqa: E402
    cascade_recall, over_fetch_factor, dependence_sweep,
)


# =========================================================================== #
# Constants. The geometry reuses the rvlc/dense company prototypes (the shared vMF finance cloud, seed 7) as
# the answer companies. Each CHAIN walks across companies in DISTINCT sectors (near-orthogonal) so the answer
# is genuinely unreachable in one hop; a "mention" component sin(alpha) toward the next company is what the
# reformulation operator extracts.
# =========================================================================== #

MHOP_SEED = DPR_SEED                  # 7 -- the shared finance-geometry seed
MHOP_DIM = DPR_DIM                    # 32

TAU_HOP = 0.35                        # retrieval-graph edge threshold on cosine (~2 sigma above the d=32
                                      #   equatorial noise 1/sqrt(d) ~ 0.177; asserted from the corpus)
TAU_ANS = PAYOFF_TAU                  # 0.15 -- the answer-readout temperature (reuse the selection arc's)

ALPHA_MENTION_DEG = 40.0             # the bridge's component toward the company it names: f = cos a u_A + sin a u_B.
                                     #   cos 40 ~ 0.77 keeps the bridge clearly retrievable (above the worst
                                     #   same-sector distractor ~ 0.58); sin 40 ~ 0.64 is the mention magnitude.
KAPPA_COMPANY = 18.0               # company directions spread around their sector mean (same-sector cosine
                                    #   ~ 0.3, max ~ 0.53 < cos 40: confusable distractors that never out-rank
                                    #   the bridge; cross-sector ~ 0: a chain's answer is invisible to its query)
KAPPA_NODE = 8000.0                 # passages VERY tight around each filing direction (cosine ~ 0.996, ~5 deg),
                                    #   so the reformulation operator extracts the mention with little noise
K_RETRIEVE = 3                       # top-k retrieved per hop
M_PER_NODE = 3                       # passages per filing node
MAX_HOPS = 5                         # loop cap (>= longest chain + 1)
REFORM_EPS = 0.47                    # stop when the read filing names no NEW entity: ||d - <d,q>q|| < REFORM_EPS
                                     #   (the expected-new-information stopping rule; an SPRT / optimal-stopping
                                     #   analogue named in prose). Bridge residuals sit in [0.53, 0.79] (the
                                     #   sin 40 mention), terminal answer residuals in [0.05, 0.43] (node noise),
                                     #   so 0.47 sits in the clean gap. The threshold is TUNED and the realized
                                     #   hop count is seed-dependent (the rigorFlag) -- not a derived optimum.

DEMO_R1, DEMO_R2 = 0.6, 0.5         # illustrative middling per-hop retentions for the FKG copula demo
                                     #   (measured per-hop recalls are ~1 here, leaving no room for FKG to bite)

LOG2 = math.log(2.0)

# chain plan: (n_hops). 4 one-hop controls, 6 two-hop, 4 three-hop (the adaptive-stopping demo).
CHAIN_PLAN = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    return np.exp(logits - logsumexp(logits))


# =========================================================================== #
#  ★  THE CORE FUNCTIONS.
# =========================================================================== #

def reformulate(q: np.ndarray, d_read: np.ndarray) -> np.ndarray:
    """★ The reformulation operator: q' = normalize(d_read - <d_read, q> q), the part of the read filing
    ORTHOGONAL to the current query -- the NEW direction it contributes (the entity it names that the query
    did not). On the mention geometry f = cos(a) u_A + sin(a) u_B with q ~ u_A, the residual is sin(a) u_B,
    so reformulation points the next retrieval at the named company B. This is the operator the whole topic
    turns on.

    GUARD (Gemini: empty denominator): if the read filing adds nothing new (d_read ~ q, residual ~ 0),
    return q unchanged -- a hop that learns nothing does not move the query."""
    q = normalize(np.asarray(q, dtype=float).reshape(-1))
    d = normalize(np.asarray(d_read, dtype=float).reshape(-1))
    resid = d - float(d @ q) * q
    nrm = float(np.linalg.norm(resid))
    if nrm < 1e-8:
        return q
    return resid / nrm


def multihop_retrieve(corpus: dict, chain_idx: int, max_hops: int = MAX_HOPS,
                      k: int = K_RETRIEVE, tau: float = TAU_ANS, reform_eps: float = REFORM_EPS) -> dict:
    """★ One trajectory: retrieve -> read -> reformulate -> retrieve, stopping when a hop names no NEW entity.

    State = belief over the answer companies (protos). Each hop ranks all passages by dual_encoder_score
    against the CURRENT query, reads the top unread passage, updates the belief (current query + the doc just
    read -- the reformulated query carries the accumulated trajectory), records belief movement
    KL(b_t || b_{t-1}) and entropy, then computes the reformulation residual ||d - <d,q>q||. That residual is
    the NEW direction the filing opens -- the expected information of the next hop. The stopping rule (an SPRT
    / optimal-stopping analogue, named in prose) continues while the residual clears `reform_eps` and stops
    when it collapses (a terminal answer filing names no one new).

    A myopic belief-movement rule would stop at the low-movement BRIDGE hop -- exactly the supermodular trap
    the climax is about; the reported `kl_moves` show that non-monotonicity, while stopping tracks new-entity
    discovery. Returns {read_ids, beliefs (incl. the query-only prior), kl_moves, entropies, mass_on_truth,
    resid_norms, hops_taken, reached_answer, queries, truth, answer_ids}."""
    passages = corpus["passages"]
    protos = corpus["protos"]
    ch = corpus["chains"][chain_idx]
    truth = int(ch["answer_company"])
    answer_ids = ch["answer_passages"]

    q0 = corpus["q"][chain_idx]
    q = q0.copy()
    read_ids: list[int] = []
    queries = [q0.copy()]
    b_prev = _softmax((q0 @ protos.T) / tau)          # belief before any hop: query-only posterior
    beliefs = [b_prev]
    kl_moves: list[float] = []
    entropies = [entropy(b_prev)]
    mass = [float(b_prev[truth])]
    resid_norms: list[float] = []
    hops_taken = 0
    for _ in range(max_hops):
        scores = dual_encoder_score(q, passages)
        order = np.argsort(-scores, kind="stable")
        top = next((int(i) for i in order if int(i) not in read_ids), None)
        if top is None:
            break
        read_ids.append(top)
        d = passages[top]
        b = answer_posterior(q, d, protos, tau)        # current query + the doc just read
        beliefs.append(b)
        kl_moves.append(kl(b, b_prev))
        entropies.append(entropy(b))
        mass.append(float(b[truth]))
        b_prev = b
        hops_taken += 1
        # the reformulation residual: how much NEW direction this filing opens (its onward mention)
        dd = normalize(d)
        resid = dd - float(dd @ q) * q
        rn = float(np.linalg.norm(resid))
        resid_norms.append(rn)
        if rn < reform_eps:                            # names no new entity -> the answer is terminal -> stop
            break
        q = resid / rn
        queries.append(q.copy())
    reached = bool(set(read_ids) & set(answer_ids))
    return {"read_ids": read_ids, "beliefs": beliefs, "kl_moves": kl_moves,
            "entropies": entropies, "mass_on_truth": mass, "resid_norms": resid_norms,
            "hops_taken": int(hops_taken), "reached_answer": reached, "queries": queries,
            "truth": truth, "answer_ids": list(answer_ids)}


# =========================================================================== #
# The corpus.
# =========================================================================== #

_CORPUS: dict | None = None


def multihop_corpus(seed: int = MHOP_SEED, plan=None, alpha_deg: float = ALPHA_MENTION_DEG,
                    kappa_company: float = KAPPA_COMPANY, kappa_node: float = KAPPA_NODE,
                    m_per_node: int = M_PER_NODE) -> dict:
    """Build the multi-hop corpus on the shared finance cloud (the rvlc sector means, seed 7). Each chain gets
    its OWN companies -- one per hop, each drawn around a DISTINCT sector mean (cross-sector draws are
    near-orthogonal, so a chain's answer company is invisible to its query; same-sector companies of OTHER
    chains, cosine ~ 0.38, are the confusable distractors). Companies are PRIVATE to a chain (no sharing), so
    there is no cross-chain interference. A bridge filing of company X that names the next company Y is drawn
    tight around f = cos(alpha) u_X + sin(alpha) u_Y; the terminal answer filing is drawn around its own
    company. The reformulation operator extracts exactly the sin(alpha) mention.

    Returns:
        protos          (K, dim)   ALL chain companies (the belief support / answer companies)
        sector_of_company (K,)     sector label per company
        passages        (n_pass, dim)
        passage_company (n_pass,)  owning company per passage
        chains          list[dict] each: {n_hops, companies (proto indices), answer_company,
                                          node_dirs, node_passages (id-lists), answer_passages, bridge_passages}
        q               (n_chains, dim) query vectors (EXACTLY the query company direction)
        sector_mu       (n_sectors, dim)
        K, dim, n_chains, n_passages, alpha_deg
    """
    if plan is None:
        plan = CHAIN_PLAN
    base = rvlc_corpus(seed=seed)
    sector_mu = _sector_means_from(base, seed)
    n_sectors = sector_mu.shape[0]
    dim = sector_mu.shape[1]
    alpha = math.radians(alpha_deg)

    protos: list[np.ndarray] = []
    sector_of_company: list[int] = []
    passages: list[np.ndarray] = []
    passage_company: list[int] = []
    chains: list[dict] = []
    queries: list[np.ndarray] = []

    def add_company(sector: int, sd: int) -> int:
        """Draw one private company direction around a sector mean; register it as a new proto."""
        u = normalize(sample_vmf(1, sector_mu[sector], kappa_company, seed=sd)[0])
        idx = len(protos)
        protos.append(u)
        sector_of_company.append(int(sector))
        return idx

    def add_node(center: np.ndarray, company: int, sd: int) -> list[int]:
        draws = sample_vmf(m_per_node, center, kappa_node, seed=sd)
        ids = []
        for p in np.atleast_2d(draws):
            ids.append(len(passages))
            passages.append(normalize(p))
            passage_company.append(int(company))
        return ids

    for ci, n_hops in enumerate(plan):
        sd = seed + 1000 * ci + 13
        # one private company per hop, each in a DISTINCT sector (cross-sector -> near-orthogonal chain)
        comps = [add_company((ci + h) % n_sectors, sd + 31 * h + 3) for h in range(n_hops)]
        answer_company = comps[-1]
        node_dirs, node_passages = [], []
        for h in range(n_hops):
            X = comps[h]
            if h < n_hops - 1:                          # a bridge filing of X that NAMES comps[h+1]
                Y = comps[h + 1]
                direction = normalize(math.cos(alpha) * protos[X] + math.sin(alpha) * protos[Y])
            else:                                       # the terminal (answer) filing: company X itself
                direction = protos[X].copy()
            ids = add_node(direction, X, sd + 7 * h + 1)
            node_dirs.append(direction)
            node_passages.append(ids)
        answer_passages = node_passages[-1]
        bridge_passages = [i for ids in node_passages[:-1] for i in ids]
        q = protos[comps[0]].copy()                     # the query identifies its company exactly
        chains.append({
            "n_hops": int(n_hops), "companies": [int(c) for c in comps],
            "answer_company": int(answer_company), "node_dirs": node_dirs,
            "node_passages": node_passages, "answer_passages": answer_passages,
            "bridge_passages": bridge_passages,
        })
        queries.append(q)

    return {
        "protos": np.array(protos), "sector_of_company": np.array(sector_of_company),
        "passages": np.array(passages), "passage_company": np.array(passage_company),
        "chains": chains, "q": np.array(queries), "sector_mu": sector_mu,
        "K": int(len(protos)), "dim": int(dim), "n_chains": int(len(plan)),
        "n_passages": int(len(passages)), "alpha_deg": float(alpha_deg),
    }


def _sector_means_from(base: dict, seed: int) -> np.ndarray:
    """The rvlc/dense sector means: the first default_rng(seed).standard_normal draw, normalized -- the same
    finance cloud the prereqs live in. (rvlc does not return sector_mu, so re-derive its exact first draw.)"""
    n_sectors = int(base["sector_of_company"].max()) + 1
    dim = base["protos"].shape[1]
    return normalize(np.random.default_rng(seed).standard_normal((n_sectors, dim)))


def _corpus(seed: int = MHOP_SEED) -> dict:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = multihop_corpus(seed)
    return _CORPUS


# =========================================================================== #
# Movement 1 & 3 helpers: recall (single vs multi-hop) and compounding.
# =========================================================================== #

def single_hop_answer_recall(corpus: dict, chain_idx: int, k: int = K_RETRIEVE) -> float:
    """Recall@k of the ANSWER passages from ONE retrieval with the original query."""
    ch = corpus["chains"][chain_idx]
    q = corpus["q"][chain_idx]
    scores = dual_encoder_score(q, corpus["passages"])
    topk = [int(i) for i in np.argsort(-scores, kind="stable")[:k]]
    return recall_at_k(topk, set(ch["answer_passages"]), k)


def multi_hop_answer_recall(corpus: dict, chain_idx: int, max_hops: int = MAX_HOPS,
                            k: int = K_RETRIEVE) -> float:
    """Whether the multi-hop loop ever reads an answer passage (1.0 reached / 0.0 not)."""
    traj = multihop_retrieve(corpus, chain_idx, max_hops, k)
    return 1.0 if traj["reached_answer"] else 0.0


def recall_summary(corpus: dict, k: int = K_RETRIEVE) -> dict:
    """Mean single-hop vs multi-hop answer-recall, split by chain class (1/2/3 hop)."""
    classes = {1: [], 2: [], 3: []}
    single, multi = [], []
    for ci, ch in enumerate(corpus["chains"]):
        s = single_hop_answer_recall(corpus, ci, k)
        m = multi_hop_answer_recall(corpus, ci, k=k)
        single.append(s)
        multi.append(m)
        classes[ch["n_hops"]].append((s, m))
    out = {"single_mean": float(np.mean(single)), "multi_mean": float(np.mean(multi))}
    for c, rows in classes.items():
        if rows:
            out[f"single_{c}hop"] = float(np.mean([r[0] for r in rows]))
            out[f"multi_{c}hop"] = float(np.mean([r[1] for r in rows]))
    # the compositional gap is on the MULTI-hop chains (2 and 3 hop), where single-hop should fail
    comp = [(s, m) for ci, ch in enumerate(corpus["chains"]) if ch["n_hops"] >= 2
            for s, m in [(single[ci], multi[ci])]]
    out["comp_single"] = float(np.mean([r[0] for r in comp]))
    out["comp_multi"] = float(np.mean([r[1] for r in comp]))
    out["comp_gap"] = out["comp_multi"] - out["comp_single"]
    return out


def per_hop_recall(corpus: dict, k: int = K_RETRIEVE) -> list[float]:
    """The per-hop retention r_h: fraction of multi-hop chains whose hop h retrieves its intended target node
    (bridge for h < last, answer for the last). Over chains with at least h hops."""
    max_h = max(ch["n_hops"] for ch in corpus["chains"])
    hits = [0] * max_h
    tot = [0] * max_h
    for ci, ch in enumerate(corpus["chains"]):
        traj = multihop_retrieve(corpus, ci)
        targets = ch["node_passages"]                    # per-hop intended node passage ids
        for h in range(ch["n_hops"]):
            tot[h] += 1
            if h < len(traj["read_ids"]) and traj["read_ids"][h] in set(targets[h]):
                hits[h] += 1
    return [hits[h] / tot[h] if tot[h] else 0.0 for h in range(max_h)]


# =========================================================================== #
# Movement 4 helpers: the supermodular synergy + the single-shot reachable pool.
# =========================================================================== #

def reachable_pool(corpus: dict, q: np.ndarray, tau_hop: float = TAU_HOP) -> list[int]:
    """The passages reachable in ONE hop from query q: those above the edge threshold tau_hop."""
    scores = dual_encoder_score(q, corpus["passages"])
    return [int(i) for i in np.where(scores > tau_hop)[0]]


def singleshot_cannot_reach(corpus: dict) -> dict:
    """The operational supermodularity: single-shot selection cannot pick a passage it cannot even reach.
    For each multi-hop chain, is the answer in the query-reachable pool (single hop)? And for the 2-hop
    chains, is it reachable after ONE reformulation from the bridge? (3-hop answers need two reformulations,
    so the one-reformulation figure is reported over 2-hop chains only.)"""
    in_single, in_reform_2hop = [], []
    for ci, ch in enumerate(corpus["chains"]):
        if ch["n_hops"] < 2:
            continue
        q0 = corpus["q"][ci]
        ans = set(ch["answer_passages"])
        in_single.append(1.0 if (ans & set(reachable_pool(corpus, q0))) else 0.0)
        if ch["n_hops"] == 2:
            scores = dual_encoder_score(q0, corpus["passages"])
            top = int(np.argsort(-scores, kind="stable")[0])
            q1 = reformulate(q0, corpus["passages"][top])
            in_reform_2hop.append(1.0 if (ans & set(reachable_pool(corpus, q1))) else 0.0)
    return {"answer_in_single_pool": float(np.mean(in_single)),
            "answer_in_reformulated_pool_2hop": float(np.mean(in_reform_2hop))}


def greedy_hop_select(corpus: dict, q: np.ndarray, k: int = K_RETRIEVE,
                      tau_hop: float = TAU_HOP, tau: float = TAU_ANS) -> dict:
    """★ The per-hop retrieve-AND-select operator -- this is the prereq's step, made the transition of a
    search. From the tau_hop-REACHABLE pool, select k passages by greedily maximizing the IMPORTED
    info_gain_fn with the IMPORTED greedy_select (the same submodular maximizer context-selection certified).
    Returns {"selected": global passage ids, "pool_ids": the reachable pool, "local": pool-local indices}."""
    pool_ids = reachable_pool(corpus, q, tau_hop)
    if not pool_ids:
        return {"selected": [], "pool_ids": [], "local": []}
    pool = {"vecs": corpus["passages"][pool_ids]}
    view = {"protos": corpus["protos"]}
    f = info_gain_fn(view, pool, q, tau)
    sel = greedy_select(f, len(pool_ids), min(k, len(pool_ids)))["selected"]
    return {"selected": [pool_ids[i] for i in sel], "pool_ids": pool_ids, "local": sel}


def worked_chain(corpus: dict, n_hops: int) -> int:
    """The first chain of the requested depth -- the representative trajectory the viz bakes."""
    return next(i for i, ch in enumerate(corpus["chains"]) if ch["n_hops"] == n_hops)


def geodesic_cosines(corpus: dict) -> dict:
    """The three load-bearing cosines on a worked 2-hop chain's filing directions: query.bridge (= cos alpha,
    retrievable), query.answer (~ 0, the gap), bridge.answer (= sin alpha, the mention), and
    reformulate(query, bridge).answer (~ 1, reached). All on the FILING DIRECTIONS (exact), not noisy draws."""
    ci = worked_chain(corpus, 2)
    ch = corpus["chains"][ci]
    protos = corpus["protos"]
    q = corpus["q"][ci]
    a1 = ch["companies"][1]
    bridge_dir = ch["node_dirs"][0]
    ans_dir = protos[a1]
    q1 = reformulate(q, bridge_dir)
    return {
        "q_dot_bridge": float(q @ bridge_dir),
        "q_dot_answer": float(q @ ans_dir),
        "bridge_dot_answer": float(bridge_dir @ ans_dir),
        "reformulated_dot_answer": float(q1 @ ans_dir),
        "tau_hop": float(TAU_HOP),
    }


# =========================================================================== #
# Tests -- every pedagogical claim is an assertion. Run order: collapse anchors, the compositional-gap
# headline, the search operator, compounding/FKG, the supermodular climax, the adaptive stopping, guards.
# =========================================================================== #

def test_answer_posterior_topk_collapse_anchor() -> None:
    """COLLAPSE ANCHOR: a one-hop belief (current query, single doc, weight [1]) read through the IMPORTED
    answer_posterior_topk equals the IMPORTED answer_posterior bit-for-bit -- the multi-hop belief is the
    same readout the rest of the arc used."""
    c = _corpus()
    ci = worked_chain(c, 2)
    q = c["q"][ci]
    d = c["passages"][c["chains"][ci]["bridge_passages"][0]]
    got = answer_posterior_topk(q, d[None, :], np.array([1.0]), c["protos"], TAU_ANS)
    want = answer_posterior(q, d, c["protos"], TAU_ANS)
    assert np.allclose(got, want, atol=1e-12), "k=1 weight [1] must equal imported answer_posterior"


def test_recall_collapse_anchor() -> None:
    """COLLAPSE ANCHOR: single-hop answer-recall is the IMPORTED recall_at_k of the one-shot ranking."""
    c = _corpus()
    ci = worked_chain(c, 1)
    q = c["q"][ci]
    scores = dual_encoder_score(q, c["passages"])
    topk = [int(i) for i in np.argsort(-scores, kind="stable")[:K_RETRIEVE]]
    got = single_hop_answer_recall(c, ci)
    want = recall_at_k(topk, set(c["chains"][ci]["answer_passages"]), K_RETRIEVE)
    assert abs(got - want) < 1e-12, "single_hop_answer_recall must equal imported recall_at_k"


def test_geodesic_inequalities() -> None:
    """The mention geometry: on a worked 2-hop chain the bridge is retrievable (q.bridge = cos alpha > tau),
    the answer is invisible to the query (q.answer < tau, near 0), the bridge carries the mention
    (bridge.answer = sin alpha > tau), and reformulating from the bridge reaches the answer (> tau)."""
    g = geodesic_cosines(_corpus())
    assert g["q_dot_answer"] < g["tau_hop"], "the answer must be below the single-hop edge threshold"
    assert g["q_dot_bridge"] > g["tau_hop"], "the bridge must be retrievable from the query"
    assert g["bridge_dot_answer"] > g["tau_hop"], "the bridge must carry the mention toward the answer"
    assert g["reformulated_dot_answer"] > g["tau_hop"], "reformulating from the bridge must reach the answer"
    assert g["reformulated_dot_answer"] > g["q_dot_answer"] + 0.5, "reformulation opens the gap to the answer"


def test_compositional_gap() -> None:
    """THE HEADLINE (robust, aggregate): on the compositional (>= 2 hop) chains, single-hop answer-recall is
    ~ 0 (the answer is near-orthogonal to the query) while multi-hop answer-recall is ~ 1 -- a gap > 0.5.
    The 1-hop control is reachable in a single hop (recall 1)."""
    rs = recall_summary(_corpus())
    assert rs["comp_single"] < 0.05, f"single-hop must miss the compositional answer, got {rs['comp_single']}"
    assert rs["comp_multi"] > 0.9, f"multi-hop must reach the compositional answer, got {rs['comp_multi']}"
    assert rs["comp_gap"] > 0.5, "the compositional gap must be large"
    assert rs["single_1hop"] > 0.95, "the 1-hop control must be reachable in a single hop"


def test_greedy_hop_twin() -> None:
    """TWIN: the per-hop retrieve-and-select operator reuses the IMPORTED greedy_select on the IMPORTED
    info_gain_fn -- greedy at k=1 picks the argmax info-gain passage of the reachable pool, byte-for-byte."""
    c = _corpus()
    ci = worked_chain(c, 2)
    q = c["q"][ci]
    res = greedy_hop_select(c, q, k=1)
    pool_ids = res["pool_ids"]
    assert pool_ids, "the query must have a non-empty reachable pool"
    pool = {"vecs": c["passages"][pool_ids]}
    f = info_gain_fn({"protos": c["protos"]}, pool, q, TAU_ANS)
    want_local = int(np.argmax([f([j]) for j in range(len(pool_ids))]))
    assert res["local"][0] == want_local, "greedy_hop@k=1 must equal argmax info-gain over the reachable pool"


def test_chain_recall_product_anchor() -> None:
    """COLLAPSE ANCHOR: the end-to-end chain recall is the IMPORTED cascade_recall of the per-hop retentions,
    and the over-fetch is its reciprocal; equal per-hop r gives geometric decay r^H. The MEASURED per-hop
    recalls are ~ 1 on this clean corpus, which is why the compounding/FKG demos use illustrative middling
    retentions (a high-retention chain leaves no room for the product law or FKG to bite -- the capstone
    precedent)."""
    assert abs(cascade_recall([DEMO_R1, DEMO_R2]) - DEMO_R1 * DEMO_R2) < 1e-12
    assert abs(over_fetch_factor([DEMO_R1, DEMO_R2]) - 1.0 / (DEMO_R1 * DEMO_R2)) < 1e-12
    for H in (1, 2, 3):
        assert abs(cascade_recall([0.7] * H) - 0.7 ** H) < 1e-12, "equal-r chain recall must be geometric"
    assert min(per_hop_recall(_corpus())) > 0.95, "measured per-hop recall is ~1 (the clean-corpus caveat)"


def test_fkg_lower_bound() -> None:
    """Positive dependence between hops makes the realized chain recall sit ABOVE the independent product
    (FKG/Harris), negative dependence below, equal at independence -- verified on the IMPORTED copula."""
    rows = dependence_sweep(DEMO_R1, DEMO_R2, [-0.6, -0.3, 0.0, 0.3, 0.6])
    by = {round(r["rho"], 1): r["gap"] for r in rows}
    assert by[0.6] > 0.02 and by[0.3] > 0.0, "positive dependence: realized recall above the product"
    assert by[-0.6] < -0.02 and by[-0.3] < 0.0, "negative dependence: realized recall below the product"
    assert abs(by[0.0]) < 0.01, "independence: realized recall equals the product"


def test_supermodular_synergy() -> None:
    """THE CLIMAX. (a) The IMPORTED XOR witness proves information gain is supermodular in general: the
    marginal of D2 INCREASES with conditioning (0 alone, 1 given D1). (b) The OPERATIONAL consequence the
    corpus exhibits: the answer is NEVER in the single-hop reachable pool, so single-shot selection cannot
    pick it; only reformulating from the bridge reaches it. (c) The trajectory signature: on a worked
    compositional chain the bridge hop barely moves the belief while the decisive answer hop moves it
    enormously -- the decisive evidence comes LAST (anti-diminishing), so a myopic belief-movement stop
    would halt at the worthless-looking bridge."""
    xor = info_gain_xor_witness()
    assert xor["violation"] > 0.5, "XOR: info gain is supermodular (marginal increases with conditioning)"
    sc = singleshot_cannot_reach(_corpus())
    assert sc["answer_in_single_pool"] == 0.0, "single-shot cannot reach the compositional answer"
    assert sc["answer_in_reformulated_pool_2hop"] > 0.95, "one reformulation reaches the 2-hop answer"
    for n in (2, 3):
        traj = multihop_retrieve(_corpus(), worked_chain(_corpus(), n))
        km = traj["kl_moves"]
        assert km[-1] > 5.0 * km[0], "the decisive belief shift is the LAST hop, not the bridge (supermodular)"
        assert traj["mass_on_truth"][0] < 0.05 and traj["mass_on_truth"][-1] > 0.8, \
            "belief starts off the answer and ends on it"


def test_info_gain_not_submodular_on_pool() -> None:
    """Reusing the IMPORTED submodularity_witness: information gain over a reachable pool is NOT submodular
    (witness clearly negative), the same honesty hinge context-selection established -- which is why greedy
    hopping earns no (1 - 1/e) guarantee."""
    c = _corpus()
    ci = worked_chain(c, 3)
    q = c["q"][ci]
    pool_ids = reachable_pool(c, q)
    pool = {"vecs": c["passages"][pool_ids]}
    f = info_gain_fn({"protos": c["protos"]}, pool, q, TAU_ANS)
    w = submodularity_witness(f, len(pool_ids), trials=300, seed=0)
    assert w < -1e-3, f"info gain must violate submodularity on the pool, witness={w}"


def test_adaptive_stopping_pinned() -> None:
    """PINNED (seed-dependent, the rigorFlag): under the new-entity stopping rule the realized hop count
    equals the chain depth -- 1-hop stops at 1, 2-hop at 2, 3-hop at 3 -- because reformulation keeps opening
    new entities until the terminal answer filing names no one."""
    c = _corpus()
    for ci, ch in enumerate(c["chains"]):
        traj = multihop_retrieve(c, ci)
        assert traj["hops_taken"] == ch["n_hops"], \
            f"chain {ci} ({ch['n_hops']}-hop) stopped at {traj['hops_taken']}"
        assert traj["reached_answer"], f"chain {ci} must reach its answer"


def test_entropy_telescopes() -> None:
    """Sanity anchor: the per-hop belief-entropy reductions telescope to the total H(b_0) - H(b_final)."""
    c = _corpus()
    traj = multihop_retrieve(c, worked_chain(c, 3))
    H = traj["entropies"]
    drops = [H[t] - H[t + 1] for t in range(len(H) - 1)]
    assert abs(sum(drops) - (H[0] - H[-1])) < 1e-9, "entropy reductions must telescope"


def test_guards() -> None:
    """Denominator / degenerate-input guards (the Gemini list)."""
    c = _corpus()
    # reformulate guards the zero residual (a filing already explained by the query)
    q = c["q"][0]
    assert np.allclose(reformulate(q, q), q), "reformulate must return q unchanged when the residual is ~0"
    # empty reachable pool -> empty selection (no crash)
    far = -c["q"][0]                                   # a query pointing away from everything
    assert greedy_hop_select(c, far, tau_hop=0.99)["selected"] == []
    # cascade_recall guards the empty product and the (0,1] range
    assert cascade_recall([]) == 1.0
    # recall_at_k empty relevant set
    assert recall_at_k([0, 1, 2], set(), K_RETRIEVE) == 0.0


def _run_all() -> None:
    tests = [
        test_answer_posterior_topk_collapse_anchor, test_recall_collapse_anchor,
        test_geodesic_inequalities, test_compositional_gap,
        test_greedy_hop_twin, test_chain_recall_product_anchor, test_fkg_lower_bound,
        test_supermodular_synergy, test_info_gain_not_submodular_on_pool,
        test_adaptive_stopping_pinned, test_entropy_telescopes, test_guards,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")


# =========================================================================== #
# Viz constants -- printed for mirroring into MultiHopLaboratory.tsx (cast every scalar).
# =========================================================================== #

def viz_constants() -> None:
    c = _corpus()
    g = geodesic_cosines(c)
    rs = recall_summary(c)
    sc = singleshot_cannot_reach(c)
    ci2, ci3 = worked_chain(c, 2), worked_chain(c, 3)
    t2 = multihop_retrieve(c, ci2)
    t3 = multihop_retrieve(c, ci3)
    sweep = dependence_sweep(DEMO_R1, DEMO_R2, [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    xor = info_gain_xor_witness()

    def r3(v):
        return round(float(v), 3)

    print("// --- baked from viz_constants() in multi_hop_iterative_retrieval.py ---")
    print(f"const MHOP_DIM = {MHOP_DIM};")
    print(f"const TAU_HOP = {r3(TAU_HOP)};")
    print(f"const ALPHA_MENTION_DEG = {r3(ALPHA_MENTION_DEG)};")
    print(f"const K_RETRIEVE = {K_RETRIEVE};")
    print(f"const REFORM_EPS = {r3(REFORM_EPS)};")
    # Panel A: the compositional gap (cosines + recall)
    print(f"const Q_DOT_BRIDGE = {r3(g['q_dot_bridge'])};        // = cos(alpha): the bridge is retrievable")
    print(f"const Q_DOT_ANSWER = {r3(g['q_dot_answer'])};        // ~ 0: the answer is near-orthogonal")
    print(f"const BRIDGE_DOT_ANSWER = {r3(g['bridge_dot_answer'])};   // = sin(alpha): the mention")
    print(f"const REFORMULATED_DOT_ANSWER = {r3(g['reformulated_dot_answer'])};  // ~ 1: reformulation reaches it")
    print(f"const RECALL_SINGLE_HOP = {r3(rs['comp_single'])};   // single-hop answer recall (compositional)")
    print(f"const RECALL_MULTI_HOP = {r3(rs['comp_multi'])};     // multi-hop answer recall")
    print(f"const RECALL_1HOP_CONTROL = {r3(rs['single_1hop'])};  // the 1-hop control IS single-hop reachable")
    print("const GRAPH_DIST_ANSWER = 2;   // the answer sits at graph distance 2 for a 2-hop question")
    # Panel B: compounding recall + FKG + over-fetch
    print(f"const DEMO_R1 = {r3(DEMO_R1)};")
    print(f"const DEMO_R2 = {r3(DEMO_R2)};")
    print(f"const CHAIN_SUCCESS = {r3(cascade_recall([DEMO_R1, DEMO_R2]))};   // = R1*R2")
    print(f"const OVER_FETCH = {r3(over_fetch_factor([DEMO_R1, DEMO_R2]))};   // = 1/(R1*R2)")
    print(f"const GEOMETRIC_DECAY = {[r3(0.7 ** h) for h in range(1, 5)]};   // r=0.7: r, r^2, r^3, r^4")
    print(f"const PER_HOP_RECALL_MEASURED = {[r3(x) for x in per_hop_recall(c)]};   // ~1: the clean-corpus caveat")
    print(f"const FKG_SWEEP = {[{'rho': r3(r['rho']), 'Rtrue': r3(r['R_true']), 'Rindep': r3(r['R_indep']), 'gap': r3(r['gap'])} for r in sweep]};")
    # Panel C: the worked 3-hop trajectory + adaptive stopping
    print(f"const TRAJ3_KL_MOVES = {[r3(x) for x in t3['kl_moves']]};        // bridge hops small, answer hop large")
    print(f"const TRAJ3_RESID = {[r3(x) for x in t3['resid_norms']]};        // > REFORM_EPS until the terminal hop")
    print(f"const TRAJ3_ENTROPY = {[r3(x) for x in t3['entropies']]};")
    print(f"const TRAJ3_MASS_ON_TRUTH = {[r3(x) for x in t3['mass_on_truth']]};  // ~0 until the answer hop")
    print(f"const TRAJ2_KL_MOVES = {[r3(x) for x in t2['kl_moves']]};")
    print(f"const TRAJ2_MASS_ON_TRUTH = {[r3(x) for x in t2['mass_on_truth']]};")
    print("const HOPS_BY_CLASS = {one: 1, two: 2, three: 3};   // adaptive hop count = chain depth")
    # Panel D: the supermodular synergy
    print(f"const XOR_DELTA_EMPTY = {r3(xor['delta_empty'])};       // bits the answer adds alone: 0")
    print(f"const XOR_DELTA_GIVEN_BRIDGE = {r3(xor['delta_given_d1'])};   // bits given the bridge: 1")
    print(f"const XOR_VIOLATION = {r3(xor['violation'])};         // > 0: supermodular, NOT submodular")
    print(f"const ANSWER_IN_SINGLE_POOL = {r3(sc['answer_in_single_pool'])};   // 0: single-shot cannot reach")
    print(f"const ANSWER_IN_REFORM_POOL = {r3(sc['answer_in_reformulated_pool_2hop'])};   // 1: one reformulation reaches it")
    print(f"const BRIDGE_KL = {r3(t2['kl_moves'][0])};   // the bridge hop's belief movement (tiny)")
    print(f"const ANSWER_KL = {r3(t2['kl_moves'][-1])};   // the answer hop's belief movement (large)")


if __name__ == "__main__":
    print("multi_hop_iterative_retrieval: running tests")
    _run_all()
    print("\nviz_constants (mirror into the .tsx):")
    viz_constants()
    print("\nall checks passed.")
