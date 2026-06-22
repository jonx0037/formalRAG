"""Multi-vector ANN: indexing and pruning MaxSim at scale (PLAID) — the reference implementation for
the formalRAG `multi-vector-ann-retrieval` topic.

`late-interaction-learned-sparse` lifted the single-vector rank ceiling by keeping one contextual
vector per TOKEN and scoring by MaxSim, S(q, d) = sum_i max_j <q_i, d_j>, and flagged the bill: an
index roughly 32x a single-vector index, and a candidate-generation step that is now a multi-vector
nearest-neighbor problem. This topic pays that bill. PLAID (the engine behind ColBERTv2) serves
late interaction at scale by reusing the two ANN prerequisites VERBATIM, in three movements, each an
`assert` below:

  MOVEMENT 1 — REPRESENTATION: token IVFADC and the shared-centroid trick. Cluster EVERY token
    embedding in the corpus into ONE shared set of centroids — the IVF coarse quantizer — and store
    each token as its nearest centroid id plus a product-quantized residual. This is IVFADC
    (`ivf-voronoi-partitioning` + `product-quantization`) applied at the token level. The shared
    centroid set is the engineering hinge: because tokens across all documents share centroids, the
    query-token-to-centroid score table is computed once and reused across the whole corpus. The
    payoff is the storage collapse that resolves late interaction's flagged 32x cost. (`plaid_index`,
    `reconstruct_doc`, `storage_collapse`.)

  MOVEMENT 2 — THE CENTROID-MaxSim APPROXIMATION AND ITS CAUCHY-SCHWARZ BOUND (the one clean theorem).
    Approximate <q_i, d_j> by <q_i, c(d_j)>, the centroid the token landed in. The error is EXACTLY
    Cauchy-Schwarz: <q_i, d_j> - <q_i, c(d_j)> = <q_i, r_j> with |<q_i, r_j>| <= ||q_i|| ||r_j||,
    r_j = d_j - c(d_j) the residual. The max over j is 1-Lipschitz in sup-norm, so per query token the
    MaxSim-term error <= ||q_i|| max_j ||r_j||, and summing, |S_full - S_centroid| <= sum_i ||q_i||
    max_j ||r_j||. The residual norm IS the k-means/PQ distortion inherited from the prereqs. HONEST
    GAP (the load-bearing rigorFlag): this is a bound on SCORES, not on the recall ORDERING — a
    uniform additive score bound does not preserve the top-k set (two documents within 2*bound can
    swap), which is precisely why Stage 3 reranks exactly. (`centroid_maxsim`, `centroid_maxsim_matrix`.)

  MOVEMENT 3 — THE CASCADE, AND THE COLLAPSE ANCHOR. Stage 1 generates candidates by probing each
    query token's nearest centroid lists (the IVF probe at token level); Stage 2 prunes them by the
    cheap centroid-MaxSim score; Stage 3 decompresses residuals and computes full MaxSim (the IMPORTED
    `maxsim_score`) only on the survivors. The ONE exact statement is the collapse anchor: probe every
    cell, prune nothing, rerank on the exact (uncompressed) tokens, and the cascade equals brute-force
    MaxSim to floating point — the cascade is a strict superset whose extreme setting is the ground
    truth (the indexing counterpart of IVF's "probing all cells is exact" and late interaction's "m=1
    MaxSim = dot product"). Everything above it is a heuristic speed-for-recall trade measured on a
    frontier. (`plaid_search`, `centroid_only_search`, `head_to_head`.)

Honest caveats (rigorFlag territory): exactly one theorem (the collapse anchor) and one exact bound
(Cauchy-Schwarz per-pair), and the bound does NOT control the per-query-token max, the argmax token
each query token selects, or the post-prune recall ordering. The recall-vs-cost frontier is ONE
synthetic vMF token cloud, not a universal ranking. The tokens are synthetic vMF stand-ins reused
from / built on the hypersphere, IVF, PQ, and late-interaction topics, NOT a trained ColBERT. The
storage win is mitigation (centroid id + compressed residual, ~order of magnitude), not erasure: the
index is still many vectors per document and the compression is lossy by construction.

This module imports its prerequisites (`late-interaction-learned-sparse` for MaxSim, IVF for the
coarse quantizer and residual, PQ for the residual codec) and their grand-ancestors (the vMF token
sampler, the Lloyd primitives) — it never reimplements them. `maxsim_score`/`maxsim_matrix` are the
byte-for-byte anchors. `viz_constants()` prints what MultiVectorANNLaboratory.tsx mirrors to the
decimal.

Run:  uv run --with numpy --with scipy python notebooks/multi-vector-ann-retrieval/multi_vector_ann_retrieval.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

# PLAID = late interaction (MaxSim) made servable by reusing the IVF coarse quantizer and the IVFADC
# residual-PQ scheme. Add EACH ancestor's HYPHENATED dir to the path, import the UNDERSCORED module —
# the established cross-topic pattern. The direct prereqs are late-interaction (MaxSim), IVF (coarse
# quantizer + residual) and PQ (encode/decode); their grand-prereqs supply primitives those do not
# re-export (the vMF token sampler, the Lloyd core), so we add every ancestor explicitly and never
# reimplement any of them. (Importing late_interaction_learned_sparse transitively puts DPR / BM25 /
# embedding-dimension on the path too — side-effect-only at import, harmless.)
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("hypersphere-vmf-geometry", "vector-quantization-lloyd-max",
             "product-quantization", "ivf-voronoi-partitioning",
             "late-interaction-learned-sparse"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf, mean_resultant_length     # noqa: E402
from vector_quantization_lloyd_max import assign                                       # noqa: E402
from product_quantization import train_pq, pq_encode, pq_decode, pq_bits              # noqa: E402
from ivf_voronoi_partitioning import (coarse_quantizer, inverted_lists,               # noqa: E402
                                      nearest_cells, residuals)
from late_interaction_learned_sparse import maxsim_score, maxsim_matrix               # noqa: E402

# --------------------------------------------------------------------------- #
# Corpus geometry. Unit-norm tokens (ColBERT normalizes), so L2 k-means on the tokens IS spherical /
# cosine clustering (||a-b||^2 = 2 - 2<a,b>): the same coarse_quantizer the IVF topic trains. Tokens
# are drawn from a shared bank of TOPIC vMF directions; a document is a bag of tokens from a few
# topics, so documents that share topics share centroids — the structure the centroid-MaxSim
# approximation and the IVF probe exploit.
# --------------------------------------------------------------------------- #
TOKEN_DIM = 16              # per-token vector dimension d
N_TOPICS = 12              # the latent "vocabulary" of topic directions on the sphere
TOKENS_PER_DOC = 8         # m_d: tokens per document (the multi-vector cost)
QUERY_TOKENS = 4           # m_q: tokens per query (queries are shorter, as in ColBERT)
TOPICS_PER_DOC = 3         # each document mixes a few topics (overlap drives candidate generation)
TOPICS_PER_QUERY = 2       # a query is more focused
N_DOCS = 120
N_QUERIES = 40
VMF_KAPPA = 60.0           # concentration: same-topic cosine must clearly beat inter-topic (sized below)

NLIST = 32                 # ~ sqrt(N_DOCS*TOKENS_PER_DOC) = sqrt(960) coarse centroids over the tokens
PQ_M, PQ_KSTAR = 4, 256    # residual PQ: d=16 -> m=4 subspaces of width 4, 256 codewords
TOPK = 10

# Representative ColBERT/PLAID storage parameters (the toy retrieval cloud above is small for speed;
# storage is a property of a real system, so Panel D bakes these — matching the late-interaction lab's
# COLBERT_DIM=128 / COLBERT_AVG_TOKENS=32 so the two labs are consistent and tell the real
# "32x raw multi-vector -> ~1x a single-vector index" story).
COLBERT_DIM = 128
COLBERT_TOKENS = 32
REP_NLIST = 1 << 16        # 65536 shared centroids (ColBERTv2 scale): 16-bit centroid id
REP_PQ_M, REP_PQ_KSTAR = 16, 256   # residual PQ: 16 subspaces x 8 bits = 128 bits/token residual
NPROBE_GRID = (1, 2, 4, 8, 16, 32)
KEEP_GRID = (5, 10, 20, 40, 80, 120)
NLIST_GRID = (4, 8, 16, 32, 64)   # for the centroid-budget / bound sweep (Panels A, B)


# --------------------------------------------------------------------------- #
# Movement 0 — the synthetic multi-vector TOKEN corpus.
# --------------------------------------------------------------------------- #

def _draw_tokens(rng, topic_mu, item_topics, m_tokens):
    """Build an (n_items, m_tokens, d) array of unit-norm tokens. Each item draws its m_tokens from
    its assigned topics (cycled), and tokens are sampled BY TOPIC in batches whose seeds come from the
    SINGLE outer rng — the one-stream rule (per-seed default_rng over consecutive seeds correlates the
    opening draws; drawing the seeds from one stream does not), without reimplementing the vMF scheme."""
    n_items = len(item_topics)
    d = topic_mu.shape[1]
    toks = np.empty((n_items, m_tokens, d))
    # token (item, pos) -> topic index
    tok_topic = np.array([[item_topics[i][p % len(item_topics[i])] for p in range(m_tokens)]
                          for i in range(n_items)])
    for t in range(topic_mu.shape[0]):
        mask = tok_topic == t
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        drawn = sample_vmf(cnt, topic_mu[t], VMF_KAPPA, seed=int(rng.integers(1 << 31)))
        toks[mask] = drawn
    return toks


def token_corpus(seed: int = 0):
    """A synthetic multi-vector TOKEN corpus. Returns (docs, queries, topic_mu) where docs is
    (N_DOCS, TOKENS_PER_DOC, TOKEN_DIM) and queries is (N_QUERIES, QUERY_TOKENS, TOKEN_DIM), both
    UNIT-NORM, and topic_mu is (N_TOPICS, TOKEN_DIM) latent topic directions. Each item samples a few
    topics and draws its tokens vMF(mu_topic, kappa) around them; shared topics across items create
    the centroid overlap the cascade exploits. SYNTHETIC vMF tokens, NOT a trained ColBERT (rigorFlag).
    All randomness flows through one rng stream."""
    rng = np.random.default_rng(seed)
    topic_mu = normalize(rng.standard_normal((N_TOPICS, TOKEN_DIM)))
    doc_topics = [rng.choice(N_TOPICS, size=TOPICS_PER_DOC, replace=False) for _ in range(N_DOCS)]
    qry_topics = [rng.choice(N_TOPICS, size=TOPICS_PER_QUERY, replace=False) for _ in range(N_QUERIES)]
    docs = _draw_tokens(rng, topic_mu, doc_topics, TOKENS_PER_DOC)
    queries = _draw_tokens(rng, topic_mu, qry_topics, QUERY_TOKENS)
    return docs, queries, topic_mu


def stack_tokens(docs: np.ndarray):
    """Flatten (N_DOCS, m_d, d) into one (N_DOCS*m_d, d) token matrix plus an owner array mapping each
    token row back to its document. The IVF coarse quantizer and inverted lists run on this stacked
    view; the owner array recovers per-document membership."""
    n_docs, m_d, d = docs.shape
    X = docs.reshape(n_docs * m_d, d)
    owner = np.repeat(np.arange(n_docs), m_d)
    return X, owner


def _same_inter_cosine(docs, topic_mu):
    """Mean same-topic vs inter-topic token cosine, for the separation guard."""
    rng = np.random.default_rng(99)
    same, inter = [], []
    for _ in range(200):
        t = int(rng.integers(N_TOPICS))
        a, b = sample_vmf(2, topic_mu[t], VMF_KAPPA, seed=int(rng.integers(1 << 31)))
        same.append(float(a @ b))
        t2 = (t + 1 + int(rng.integers(N_TOPICS - 1))) % N_TOPICS
        c = sample_vmf(1, topic_mu[t2], VMF_KAPPA, seed=int(rng.integers(1 << 31)))[0]
        inter.append(float(a @ c))
    return float(np.mean(same)), float(np.mean(inter))


# --------------------------------------------------------------------------- #
# Movement 1 — representation: token IVFADC and the shared centroid vocabulary.
# --------------------------------------------------------------------------- #

def plaid_index(docs: np.ndarray, nlist: int = NLIST, pq_m: int = PQ_M, pq_kstar: int = PQ_KSTAR,
                seed: int = 0, exact: bool = False) -> dict:
    """The PLAID document representation (Movement 1). Stack all tokens, train the IVF coarse quantizer
    (the shared centroid vocabulary), assign each token to its nearest centroid, compute residuals, and
    PQ-compress them (IVFADC at the token level). Returns a dict with the centroids C, per-token labels
    and owners, the inverted lists, the PQ codebooks and codes, per-document centroid-id bags, a
    cell->doc-set map for candidate generation, and (when exact=True) the original tokens so Stage 3
    can rerank losslessly for the collapse anchor. Every piece is imported; only the per-document
    bagging is new."""
    X, owner = stack_tokens(docs)
    n_docs, m_d, _ = docs.shape
    C = coarse_quantizer(X, nlist, seed=seed)
    labels, lists = inverted_lists(X, C)
    res = residuals(X, C, labels)
    codebooks = train_pq(res, pq_m, pq_kstar, seed=seed)
    codes = pq_encode(res, codebooks)
    doc_cids = labels.reshape(n_docs, m_d)                       # centroid id of each (doc, token)
    cell_docs = [set(owner[lst].tolist()) for lst in lists]      # docs touching each cell
    idx = {"C": C, "labels": labels, "owner": owner, "lists": lists, "codebooks": codebooks,
           "codes": codes, "doc_cids": doc_cids, "cell_docs": cell_docs,
           "n_docs": n_docs, "m_d": m_d, "nlist": nlist}
    if exact:
        idx["tokens"] = docs                                     # lossless store for the anchor
    return idx


def reconstruct_doc(doc_id: int, index: dict) -> np.ndarray:
    """Decompress one document's tokens for the Stage-3 rerank: D_hat[j] = c(d_j) + pq_decode(code_j),
    reusing the IMPORTED pq_decode. When the index was built exact=True, returns the stored ORIGINAL
    tokens (the collapse anchor needs a lossless rerank; PQ is lossy by construction)."""
    if "tokens" in index:
        return index["tokens"][doc_id]
    m_d = index["m_d"]
    rows = np.arange(doc_id * m_d, (doc_id + 1) * m_d)
    cids = index["labels"][rows]
    rec_res = pq_decode(index["codes"][rows], index["codebooks"])
    return index["C"][cids] + rec_res


def storage_collapse(n_docs: int = N_DOCS, m_d: int = TOKENS_PER_DOC, dim: int = TOKEN_DIM,
                     nlist: int = NLIST, pq_m: int = PQ_M, pq_kstar: int = PQ_KSTAR,
                     float_bits: int = 32) -> dict:
    """The storage law (Movement 1 / Movement 4). A single-vector index stores one dim-float vector per
    document; raw late interaction stores m_d of them (the flagged multiplier); PLAID stores, per token,
    a centroid id (ceil(log2 nlist) bits) plus a PQ residual code (pq_m*log2(pq_kstar) bits, via the
    imported pq_bits). Returns bits-per-document for each and the multipliers. Reuses the IMPORTED
    pq_bits so the residual-code accounting matches the PQ topic exactly."""
    single = dim * float_bits
    raw_multi = m_d * dim * float_bits
    per_token_plaid = math.ceil(math.log2(nlist)) + pq_bits(pq_m, pq_kstar)
    plaid = m_d * per_token_plaid
    return {"single_bits": int(single), "raw_multi_bits": int(raw_multi), "plaid_bits": int(plaid),
            "raw_mult": round(raw_multi / single, 1), "plaid_mult": round(plaid / single, 1),
            "plaid_vs_raw": round(raw_multi / plaid, 1)}


# --------------------------------------------------------------------------- #
# Movement 2 — the centroid-MaxSim approximation and its Cauchy-Schwarz bound.
# --------------------------------------------------------------------------- #

def centroid_maxsim(Q: np.ndarray, doc_cids: np.ndarray, C: np.ndarray) -> float:
    """Cheap approximate MaxSim (Stage-2 prune score): replace each document token by its centroid and
    score with the IMPORTED maxsim_score. S_centroid(q, d) = sum_i max_j <q_i, c(d_j)>. Reuses the
    prereq; this is 'MaxSim on the centroid-substituted document', not a reimplementation. In practice
    the Q.C table is precomputed once per query and reused across the corpus (the cost model counts
    that), but the score is identical to scoring the gathered centroids directly."""
    return maxsim_score(Q, C[doc_cids])


def centroid_maxsim_matrix(Qs: np.ndarray, index: dict) -> np.ndarray:
    """The full query-by-document centroid-MaxSim matrix, via the IMPORTED maxsim_matrix on the
    centroid-substituted documents Dc[d] = C[doc_cids[d]]. The cheap signal the cascade prunes by."""
    Dc = index["C"][index["doc_cids"]]                          # (n_docs, m_d, d)
    return maxsim_matrix(Qs, Dc)


def cauchy_schwarz_doc_bound(Q: np.ndarray, doc_id: int, index: dict) -> float:
    """The document-level additive bound sum_i ||q_i|| * max_j ||r_j|| on |S_full - S_centroid| for one
    (query, document): the sup-norm/Lipschitz lift of the per-pair Cauchy-Schwarz inequality. r_j is
    the residual of document token j against its own centroid."""
    # The clean theorem uses the COARSE residual r_j = d_j - c(d_j) against the ORIGINAL token, so an
    # exact=True index (original tokens stored) is required — the PQ-reconstructed token is a different,
    # lossier residual.
    if "tokens" not in index:
        raise ValueError("cauchy_schwarz_doc_bound needs an exact=True index (original tokens)")
    cids = index["doc_cids"][doc_id]
    r = index["tokens"][doc_id] - index["C"][cids]              # (m_d, d) coarse residuals
    max_r = float(np.linalg.norm(r, axis=1).max())
    return float(np.linalg.norm(Q, axis=1).sum()) * max_r


# --------------------------------------------------------------------------- #
# Movement 3 — the cascade.
# --------------------------------------------------------------------------- #

def stage1_candidates(Q: np.ndarray, index: dict, nprobe: int) -> set:
    """Stage 1 candidate generation: for each QUERY TOKEN probe its nprobe nearest centroids (the
    imported nearest_cells) and collect every document owning a token in those cells. Returns the
    candidate document-id set (native set union, no list-comp membership filter). At nprobe = nlist
    every document is generated."""
    cand: set = set()
    for i in range(Q.shape[0]):
        for cell in nearest_cells(Q[i], index["C"], nprobe):
            cand |= index["cell_docs"][cell]
    return cand


def stage2_prune(Q: np.ndarray, candidates, index: dict, keep: int):
    """Stage 2: score each candidate by the cheap centroid-MaxSim and keep the top `keep`. Returns the
    survivor doc ids (a list). GUARD: keep capped at len(candidates); empty candidates -> []."""
    cand = list(candidates)
    if not cand:
        return []
    keep = min(max(keep, 1), len(cand))
    cand_arr = np.array(cand)
    Dc = index["C"][index["doc_cids"][cand_arr]]                # (n_cand, m_d, d): centroid-substituted docs
    scores = maxsim_matrix(Q[None, ...], Dc)[0]                 # vectorized centroid-MaxSim over candidates
    order = np.argsort(-scores, kind="stable")[:keep]
    return [cand[i] for i in order]


def stage3_rerank(Q: np.ndarray, survivors, index: dict, topk: int):
    """Stage 3: decompress each survivor's tokens (reconstruct_doc -> imported pq_decode) and score with
    the IMPORTED maxsim_score (full MaxSim on the reconstructed tokens). Returns (topk_ids, scores_dict).
    GUARD: topk capped at len(survivors); empty survivors -> ([], {})."""
    if not survivors:
        return [], {}
    scores = {d: maxsim_score(Q, reconstruct_doc(d, index)) for d in survivors}
    kk = min(topk, len(survivors))
    ranked = sorted(survivors, key=lambda d: -scores[d])[:kk]
    return ranked, scores


def plaid_search(Q: np.ndarray, index: dict, nprobe: int, keep: int, topk: int = TOPK):
    """The full PLAID cascade for one query. Returns (topk_doc_ids, cost) where cost is the mean exact
    distance computations: m_q*nlist for the centroid probe/table (Stage 1+2 share it) plus
    m_q*m_d*(#docs reranked) for the Stage-3 full MaxSim. keep <= candidates, so few docs are reranked."""
    m_q = Q.shape[0]
    cand = stage1_candidates(Q, index, nprobe)
    survivors = stage2_prune(Q, cand, index, keep)
    ranked, _ = stage3_rerank(Q, survivors, index, topk)
    cost = m_q * index["nlist"] + m_q * index["m_d"] * len(survivors)
    return ranked, float(cost)


def centroid_only_search(Q: np.ndarray, index: dict, topk: int = TOPK):
    """The cheap-approximation baseline: rank ALL documents by centroid-MaxSim, no IVF probe, no exact
    rerank. Returns (topk_doc_ids, cost) with cost = m_q*nlist (the Q.C table; the gathers are
    bookkeeping). A ceiling that never reaches recall 1 — it never reranks."""
    m_q = Q.shape[0]
    scores = centroid_maxsim_matrix(Q[None, ...], index)[0]     # vectorized centroid-MaxSim over all docs
    kk = min(topk, index["n_docs"])
    top = np.argsort(-scores, kind="stable")[:kk]
    return top.tolist(), float(m_q * index["nlist"])


# --------------------------------------------------------------------------- #
# Ground truth, recall, and the head-to-head (one cloud, one shared truth).
# --------------------------------------------------------------------------- #

def brute_topk(Qs: np.ndarray, Ds: np.ndarray, topk: int = TOPK):
    """Exact brute-force MaxSim top-k per query via the IMPORTED maxsim_matrix — the shared ground
    truth every recall call re-derives from. Returns a list of topk-id arrays (ranked)."""
    S = maxsim_matrix(Qs, Ds)
    kk = min(topk, Ds.shape[0])
    return [np.argsort(-S[q], kind="stable")[:kk] for q in range(Qs.shape[0])]


def _recall(found_lists, truth, topk: int) -> float:
    """Mean recall@topk against the shared truth (set overlap). GUARDS: non-empty queries, topk>=1."""
    nq = len(truth)
    if nq == 0:
        raise ValueError("queries must be non-empty")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    hits = sum(len(set(found_lists[q].tolist() if hasattr(found_lists[q], "tolist") else found_lists[q])
                   & set(truth[q].tolist())) for q in range(nq))
    return hits / (nq * topk)


def _cost_at_recall(rows, r_target: float):
    """Min cost among frontier rows reaching recall >= r_target (None if none do)."""
    costs = [r["cost"] for r in rows if r["recall"] >= r_target - 1e-9]
    return min(costs) if costs else None


def keep_sweep(docs: np.ndarray, queries: np.ndarray, topk: int = TOPK, keep_grid=KEEP_GRID,
               seed: int = 0):
    """A clean single-knob frontier for the viz: at full nprobe (probe every cell, so the candidate set
    is fixed at every document), walk the prune depth `keep` and report the cost and BOTH recalls — the
    deployed PQ-compressed rerank and the exact-token rerank. The exact curve climbs monotonically to
    recall 1.0 at keep=N (the collapse anchor); the PQ curve approaches it and plateaus below (the lossy
    residual gap). Same cost for both (same #reranked). Returns a list of row dicts."""
    truth = brute_topk(queries, docs, topk)
    idx_pq = plaid_index(docs, seed=seed)
    idx_ex = plaid_index(docs, seed=seed, exact=True)
    m_q, m_d = queries.shape[1], docs.shape[1]
    rows = []
    for keep in keep_grid:
        pq = [np.array(plaid_search(queries[q], idx_pq, idx_pq["nlist"], keep, topk)[0])
              for q in range(len(queries))]
        ex = [np.array(plaid_search(queries[q], idx_ex, idx_ex["nlist"], keep, topk)[0])
              for q in range(len(queries))]
        reranked = min(keep, docs.shape[0])
        rows.append({"keep": int(keep), "cost": float(m_q * idx_pq["nlist"] + m_q * m_d * reranked),
                     "recall_pq": round(_recall(pq, truth, topk), 4),
                     "recall_exact": round(_recall(ex, truth, topk), 4)})
    return rows


def head_to_head(docs: np.ndarray, queries: np.ndarray, topk: int = TOPK,
                 nprobe_grid=NPROBE_GRID, keep_grid=KEEP_GRID, seed: int = 0):
    """Build the PLAID index on the SAME (docs, queries); re-derive ONE shared ground truth from the
    imported maxsim_matrix; trace recall-vs-cost for three scorers compared by distance-comps/query:
      brute    — full MaxSim over every (query token, every doc token): cost m_q*m_d*N (the ceiling)
      centroid — centroid-MaxSim over every document, no rerank: cost m_q*nlist (a cheap plateau)
      plaid    — the cascade at each (nprobe, keep): cost m_q*nlist + m_q*m_d*reranked
    Returns {'brute','centroid','plaid','n_docs','topk','nlist'}."""
    Qs, Ds = queries, docs
    m_q, m_d, n = Qs.shape[1], Ds.shape[1], Ds.shape[0]
    truth = brute_topk(Qs, Ds, topk)
    index = plaid_index(docs, seed=seed)

    brute_cost = float(m_q * m_d * n)
    brute_row = {"recall": _recall(truth, truth, topk), "cost": brute_cost}

    cfound = [np.array(centroid_only_search(Qs[q], index, topk)[0]) for q in range(len(Qs))]
    centroid_row = {"recall": _recall(cfound, truth, topk),
                    "cost": float(m_q * index["nlist"])}

    plaid_rows = []
    for nprobe in nprobe_grid:
        if nprobe > index["nlist"]:
            continue
        for keep in keep_grid:
            found, costs = [], []
            for q in range(len(Qs)):
                ids, cost = plaid_search(Qs[q], index, nprobe, keep, topk)
                found.append(np.array(ids))
                costs.append(cost)
            plaid_rows.append({"nprobe": int(nprobe), "keep": int(keep),
                               "recall": _recall(found, truth, topk),
                               "cost": float(np.mean(costs))})
    return {"brute": brute_row, "centroid": centroid_row, "plaid": plaid_rows,
            "n_docs": n, "topk": topk, "nlist": index["nlist"]}


# --------------------------------------------------------------------------- #
# The small 2-D toy for the geometry / grid panels (legible, baked to the .tsx).
# --------------------------------------------------------------------------- #

def viz_toy(seed: int = 3):
    """A tiny 2-D token toy for Panels A and B: a pool of tokens around 4 topic directions, one demo
    query (3 tokens) and one demo document (4 tokens), all unit-norm. d=2 keeps cosines legible.
    Returns (pool, query, doc, topic_mu_2d)."""
    rng = np.random.default_rng(seed)
    ang = np.array([0.4, 1.9, 3.3, 4.8])                         # 4 topic directions on the circle
    topic_mu = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    def draw(n_per):
        out = []
        for t in range(4):
            out.append(sample_vmf(n_per, topic_mu[t], 12.0, seed=int(rng.integers(1 << 31))))
        return np.vstack(out)
    pool = draw(6)                                               # 24 tokens for training centroids
    doc = np.vstack([sample_vmf(2, topic_mu[0], 12.0, seed=int(rng.integers(1 << 31))),
                     sample_vmf(2, topic_mu[2], 12.0, seed=int(rng.integers(1 << 31)))])   # 4 tokens
    query = np.vstack([sample_vmf(1, topic_mu[0], 12.0, seed=int(rng.integers(1 << 31))),
                       sample_vmf(1, topic_mu[2], 12.0, seed=int(rng.integers(1 << 31))),
                       sample_vmf(1, topic_mu[1], 12.0, seed=int(rng.integers(1 << 31)))])  # 3 tokens
    return pool, query, doc, topic_mu


TOY_K_GRID = (2, 4, 8)


def toy_geometry_sweep(k_grid=TOY_K_GRID):
    """Per-K geometry of the 2-D toy for Panels A & B: for each #centroids K, the trained centroids and
    the pool/doc token assignments. The .tsx recomputes the residual lines, the centroid-MaxSim grid,
    the per-cell error, and the Cauchy-Schwarz bound from these baked centroids + the baked query/doc
    (all closed forms) — only the k-means centroids themselves are baked. Returns per-K row dicts."""
    pool, _, doc, _ = viz_toy()
    rows = []
    for K in k_grid:
        C = coarse_quantizer(pool, K, seed=0)
        pool_lab, _ = assign(pool, C)
        doc_lab, _ = assign(doc, C)
        rows.append({"K": int(K), "centroids": [[round(float(v), 3) for v in c] for c in C],
                     "pool_assign": pool_lab.tolist(), "doc_assign": doc_lab.tolist()})
    return rows


def toy_bound_sweep(nlist_grid=TOY_K_GRID):
    """For the toy doc, sweep the number of centroids K: train K centroids on the pool, substitute the
    doc tokens by their centroids, and report the true vs centroid MaxSim, the max per-cell error, the
    max per-cell Cauchy-Schwarz bound ||q||*||r||, and the mean residual energy. More centroids ->
    smaller residuals -> tighter bound and closer approximation. Returns a list of row dicts."""
    pool, query, doc, _ = viz_toy()
    full = maxsim_score(query, doc)
    rows = []
    for K in nlist_grid:
        C = coarse_quantizer(pool, K, seed=0)
        cids, _ = assign(doc, C)
        Dc = C[cids]
        approx = maxsim_score(query, Dc)
        r = doc - Dc
        rnorm = np.linalg.norm(r, axis=1)
        qnorm = np.linalg.norm(query, axis=1)
        per_cell_err = np.abs(query @ doc.T - query @ Dc.T)      # (m_q, m_d)
        per_cell_bound = np.outer(qnorm, rnorm)                  # (m_q, m_d)
        rows.append({"K": int(K), "full": round(float(full), 4), "centroid": round(float(approx), 4),
                     "max_err": round(float(per_cell_err.max()), 4),
                     "max_bound": round(float(per_cell_bound.max()), 4),
                     "residual_energy": round(float((rnorm ** 2).mean()), 4)})
    return rows


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_cascade_collapses_to_brute() -> None:
    """THE anchor (Theorem 2). At nprobe=nlist (probe every cell), keep=N_DOCS (prune nothing), and an
    exact=True index (rerank on the ORIGINAL tokens), the cascade returns the brute-force MaxSim
    ranking exactly: recall 1.0 AND the same top-k ordering as the imported maxsim_matrix. The
    approximations are the ONLY thing that loses recall; at their extreme settings they vanish."""
    docs, queries, _ = token_corpus()
    truth = brute_topk(queries, docs, TOPK)
    index = plaid_index(docs, exact=True)
    hits = 0
    for q in range(len(queries)):
        ids, _ = plaid_search(queries[q], index, nprobe=index["nlist"], keep=N_DOCS, topk=TOPK)
        assert list(ids) == truth[q].tolist(), f"collapse ordering differs at query {q}"
        hits += len(set(ids) & set(truth[q].tolist()))
    recall = hits / (len(queries) * TOPK)
    assert abs(recall - 1.0) < 1e-12, f"collapse recall must be 1.0, got {recall}"
    print(f"  [ok] Movement 3: cascade collapses to brute MaxSim at full settings — "
          f"recall {recall:.3f}, identical top-{TOPK} ordering on all {len(queries)} queries")


def test_full_probe_no_prune_is_exact() -> None:
    """Sub-anchor: Stage 1 at nprobe=nlist generates EVERY document as a candidate (no candidate-set
    loss) — the IVF 'exhaustive probe is exact' fact lifted to the token level."""
    docs, queries, _ = token_corpus()
    index = plaid_index(docs)
    for q in range(0, len(queries), 7):
        cand = stage1_candidates(queries[q], index, nprobe=index["nlist"])
        assert len(cand) == N_DOCS, f"probe-all must generate all docs, got {len(cand)} at query {q}"
    print(f"  [ok] Movement 3: probing all {index['nlist']} cells generates every one of the "
          f"{N_DOCS} documents (no candidate-set loss)")


def test_centroid_maxsim_collapses_at_zero_residual() -> None:
    """Collapse anchor #1 (Movement 2). With each document token replaced by its centroid, centroid-
    MaxSim IS MaxSim on the centroid-substituted documents, byte-for-byte against the IMPORTED
    maxsim_matrix (<1e-12). The cheap approximation reduces exactly to the prereq."""
    docs, queries, _ = token_corpus()
    index = plaid_index(docs)
    M_fn = np.array([[centroid_maxsim(queries[q], index["doc_cids"][d], index["C"])
                      for d in range(N_DOCS)] for q in range(len(queries))])
    M_mat = centroid_maxsim_matrix(queries, index)
    assert np.allclose(M_fn, M_mat, atol=1e-12), "centroid-MaxSim must equal maxsim_matrix on centroids"
    print("  [ok] Movement 2: centroid-MaxSim == imported maxsim_matrix on the centroid-substituted "
          "documents (<1e-12)")


def test_cauchy_schwarz_bound() -> None:
    """Theorem 1 (Movement 2). Per pair |<q_i,d_j> - <q_i,c(d_j)>| = |<q_i,r_j>| <= ||q_i|| ||r_j||
    (Cauchy-Schwarz, exact), and the document-level additive bound |S_full - S_centroid| <= sum_i
    ||q_i|| max_j ||r_j|| holds for EVERY (query, document) pair. The residual norm is the coarse
    k-means distortion inherited from IVF."""
    docs, queries, _ = token_corpus()
    index = plaid_index(docs, exact=True)
    worst_pair = 0.0
    for q in range(0, len(queries), 5):
        Q = queries[q]
        qnorm = np.linalg.norm(Q, axis=1)
        for d in range(0, N_DOCS, 11):
            D = docs[d]
            cids = index["doc_cids"][d]
            Dc = index["C"][cids]
            r = D - Dc
            rnorm = np.linalg.norm(r, axis=1)
            per_cell_err = np.abs(Q @ D.T - Q @ Dc.T)            # (m_q, m_d)
            per_cell_bound = np.outer(qnorm, rnorm)
            assert np.all(per_cell_err <= per_cell_bound + 1e-9), "per-pair Cauchy-Schwarz violated"
            worst_pair = max(worst_pair, float((per_cell_bound - per_cell_err).min()))
            doc_err = abs(maxsim_score(Q, D) - maxsim_score(Q, Dc))
            doc_bound = cauchy_schwarz_doc_bound(Q, d, index)
            assert doc_err <= doc_bound + 1e-9, "document-level additive bound violated"
    print("  [ok] Movement 2: Cauchy-Schwarz bound holds per-pair AND at the document level for every "
          "tested (q,d); per-pair slack stays >= 0")


def test_token_cloud_separates() -> None:
    """The corpus must EXHIBIT topic separation (the vMF-kappa gotcha: a kappa that 'looks tight' is
    near-orthogonal at high d). Same-topic token cosine must clearly beat inter-topic, matching the vMF
    prediction A_d(kappa)^2."""
    docs, queries, topic_mu = token_corpus()
    same, inter = _same_inter_cosine(docs, topic_mu)
    pred = mean_resultant_length(TOKEN_DIM, VMF_KAPPA) ** 2       # E[cos] for two same-topic vMF draws
    assert same > inter + 0.3, f"tokens do not separate: same {same:.3f} vs inter {inter:.3f}"
    assert same > 0.6, f"same-topic tokens not concentrated enough (k-means won't recover topics): {same:.3f}"
    assert abs(same - pred) < 0.15, f"same-topic cosine {same:.3f} should track A_d(kappa)^2={pred:.3f}"
    print(f"  [ok] Movement 0: token cloud separates — same-topic cos {same:.3f} (vMF predicts "
          f"{pred:.3f}) >> inter-topic cos {inter:.3f}")


def test_spherical_kmeans_equivalence() -> None:
    """Movement 0 / 1. ColBERT normalizes tokens, so for unit a,b the distance identity ||a-b||^2 =
    2 - 2<a,b> holds exactly (<1e-12): L2 k-means on unit tokens optimizes the same objective as cosine
    clustering, so the IVF coarse_quantizer (L2 Lloyd) is the right token quantizer. (We assert the
    exact identity, not an assignment equality — Lloyd cell MEANS are not unit-norm, so strict
    argmin-L2 = argmax-cos fails across centroids; the equivalence is at the objective level.)"""
    docs, _, _ = token_corpus()
    X, _ = stack_tokens(docs)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-9), "tokens must be unit-norm"
    a, b = X[:200], X[200:400]
    lhs = np.sum((a - b) ** 2, axis=1)
    rhs = 2.0 - 2.0 * np.sum(a * b, axis=1)
    assert np.allclose(lhs, rhs, atol=1e-12), "cosine-distance identity off on unit tokens"
    print("  [ok] Movement 0: ||a-b||^2 = 2-2<a,b> on unit tokens (<1e-12) — L2 k-means is cosine "
          "clustering at the objective level")


def test_centroid_maxsim_approximates_full() -> None:
    """Movement 2 (direction). Centroid-MaxSim's per-query ranking has high top-k overlap with full
    MaxSim because the residual norm is small (the inherited k-means distortion). The bound is tight on
    SCORES but loose on RANKINGS — that gap is why the cascade still reranks (Proposition 2)."""
    docs, queries, _ = token_corpus()
    index = plaid_index(docs)
    truth = brute_topk(queries, docs, TOPK)
    cfound = [np.array(centroid_only_search(queries[q], index, TOPK)[0]) for q in range(len(queries))]
    overlap = _recall(cfound, truth, TOPK)
    assert overlap > 0.5, f"centroid-only top-{TOPK} overlap too low: {overlap:.3f}"
    assert overlap < 1.0, "centroid-only should NOT be exact — it never reranks (else the demo is flat)"
    print(f"  [ok] Movement 2: centroid-only top-{TOPK} overlap with brute = {overlap:.3f} "
          f"(good but < 1 — the bound controls scores, not the ranking)")


def test_recall_monotone_in_keep() -> None:
    """Movement 3 (Proposition 3). With an EXACT rerank, recall is provably non-decreasing in the prune
    depth `keep`: a larger survivor set is a SUPERSET, and the exact top-k over a superset can only gain
    true neighbors, never lose one. (Under the realistic LOSSY-PQ rerank this holds only essentially —
    a false positive with a high APPROXIMATE score can displace a true neighbor — which is itself the
    Proposition-2 gap, so monotonicity is the exact-rerank statement.)"""
    docs, queries, _ = token_corpus()
    index = plaid_index(docs, exact=True)
    truth = brute_topk(queries, docs, TOPK)
    recalls = []
    for keep in KEEP_GRID:
        found = [np.array(plaid_search(queries[q], index, index["nlist"], keep, TOPK)[0])
                 for q in range(len(queries))]
        recalls.append(_recall(found, truth, TOPK))
    for a, b in zip(recalls, recalls[1:]):
        assert b >= a - 1e-9, f"recall not monotone in keep (exact rerank): {recalls}"
    assert abs(recalls[-1] - 1.0) < 1e-12, "exact rerank at keep=N must recover brute (recall 1.0)"
    print(f"  [ok] Movement 3: recall non-decreasing in prune depth keep={list(KEEP_GRID)} -> "
          f"{[round(r, 3) for r in recalls]} (exact rerank; reaches 1.0 at keep=N)")


def test_plaid_beats_brute_at_equal_recall() -> None:
    """Movement 3 (the robust, intra-family headline). PLAID reaches a high recall at strictly fewer
    distance computations than brute MaxSim — same scoring family (both end in exact MaxSim on a
    subset), the cascade just evaluates fewer pairs. Target pinned to a recall PLAID actually achieves
    (see head_to_head), well below the brute cost m_q*m_d*N."""
    docs, queries, _ = token_corpus()
    h = head_to_head(docs, queries)
    target = 0.90
    cost = _cost_at_recall(h["plaid"], target)
    assert cost is not None, f"PLAID never reaches recall {target} on this cloud"
    assert cost < h["brute"]["cost"], (f"PLAID cost {cost:.0f} at recall {target} not below brute "
                                       f"{h['brute']['cost']:.0f}")
    print(f"  [ok] Movement 3: PLAID reaches recall {target} at cost {cost:.0f} << brute cost "
          f"{h['brute']['cost']:.0f} (>= {h['brute']['cost'] / cost:.1f}x fewer distance comps)")


def test_plaid_vs_centroid_on_one_cloud() -> None:
    """Movement 3 (the one-cloud verdict, pinned after running — NOT a universal ranking). On this
    synthetic vMF token cloud the centroid-only baseline is cheapest but plateaus below recall 1; the
    PLAID cascade climbs past that plateau by reranking. A different corpus moves the knee."""
    docs, queries, _ = token_corpus()
    h = head_to_head(docs, queries)
    best_plaid = max(r["recall"] for r in h["plaid"])
    assert best_plaid > h["centroid"]["recall"], (f"PLAID max recall {best_plaid:.3f} should exceed the "
                                                  f"centroid-only ceiling {h['centroid']['recall']:.3f}")
    print(f"  [ok] Movement 3 (one cloud): centroid-only ceiling {h['centroid']['recall']:.3f} at cost "
          f"{h['centroid']['cost']:.0f}; PLAID climbs to {best_plaid:.3f} by reranking")


def test_toy_bound_tightens_with_k() -> None:
    """Panels A/B (the geometry the viz shows must exhibit the phenomenon). As the number of centroids K
    grows, the residual norms shrink, so the Cauchy-Schwarz bound and the centroid-MaxSim approximation
    error both tighten and the centroid score approaches the true MaxSim. Asserts the contrast, not the
    decimals (viz_constants bakes those)."""
    sweep = toy_bound_sweep()
    bounds = [r["max_bound"] for r in sweep]
    energies = [r["residual_energy"] for r in sweep]
    full = sweep[0]["full"]
    assert bounds[-1] < bounds[0], f"bound should tighten as K grows: {bounds}"
    assert energies[-1] < energies[0], f"residual energy should fall as K grows: {energies}"
    assert abs(sweep[-1]["centroid"] - full) < abs(sweep[0]["centroid"] - full), \
        "centroid-MaxSim should approach the true MaxSim as K grows"
    print(f"  [ok] Panels A/B: bound tightens with #centroids K={list(TOY_K_GRID)} -> max_bound "
          f"{bounds} and residual energy {energies} both fall")


def test_storage_collapse() -> None:
    """Movement 1 / 4 (Proposition 1). The storage law: raw multi-vector is m_d x a single-vector index;
    PLAID's per-token footprint (centroid id + PQ residual code) is far below the raw float vector, so
    the index shrinks by ~an order of magnitude over raw late interaction. At ColBERT scale the 32x raw
    index collapses to roughly a single-vector index. Reuses the imported pq_bits."""
    # the law holds for any m_d
    for m_d in (TOKENS_PER_DOC, COLBERT_TOKENS):
        s = storage_collapse(m_d=m_d, dim=COLBERT_DIM, nlist=REP_NLIST, pq_m=REP_PQ_M, pq_kstar=REP_PQ_KSTAR)
        assert s["raw_mult"] == float(m_d), f"raw multiplier should be m_d={m_d}"
        assert s["plaid_bits"] < s["raw_multi_bits"], "PLAID must store fewer bits than raw multi-vector"
        assert s["plaid_vs_raw"] > 5.0, f"PLAID should compress raw multi-vector >5x, got {s['plaid_vs_raw']}"
    rep = storage_collapse(m_d=COLBERT_TOKENS, dim=COLBERT_DIM, nlist=REP_NLIST,
                           pq_m=REP_PQ_M, pq_kstar=REP_PQ_KSTAR)
    assert rep["raw_mult"] == 32.0, "representative raw multiplier should be 32x"
    assert rep["plaid_mult"] < 3.0, "PLAID index should collapse to a few x a single-vector index"
    print(f"  [ok] Movement 1: storage/doc (ColBERT scale d={COLBERT_DIM}, m_d={COLBERT_TOKENS}) "
          f"single={rep['single_bits']}b raw-multi={rep['raw_multi_bits']}b ({rep['raw_mult']}x) "
          f"PLAID={rep['plaid_bits']}b ({rep['plaid_mult']}x) — {rep['plaid_vs_raw']}x below raw, "
          f"~a single-vector index")


# --------------------------------------------------------------------------- #
# Finance case study and the viz constants.
# --------------------------------------------------------------------------- #

def finance_demo() -> None:
    """PLAID as a production late-interaction serving stack. A multi-vector retriever over filings /
    transcripts cannot store or scan raw ColBERT token vectors; PLAID's shared centroids + compressed
    residuals make it deployable, the centroid-MaxSim prune throws out most filings cheaply, and full
    MaxSim reranks the survivors. SYNTHETIC vMF tokens, not a trained ColBERT."""
    docs, queries, _ = token_corpus()
    h = head_to_head(docs, queries)
    s = storage_collapse(m_d=COLBERT_TOKENS, dim=COLBERT_DIM, nlist=REP_NLIST,
                         pq_m=REP_PQ_M, pq_kstar=REP_PQ_KSTAR)
    # an operating point that keeps recall high at a fraction of the brute cost
    pick = min((r for r in h["plaid"] if r["recall"] >= 0.95),
               key=lambda r: r["cost"], default=max(h["plaid"], key=lambda r: r["recall"]))
    print("  a multi-vector earnings/filings retriever (SYNTHETIC vMF tokens, not a trained ColBERT):")
    print(f"    {N_DOCS} filings x {TOKENS_PER_DOC} tokens, queries x {QUERY_TOKENS} tokens, d={TOKEN_DIM}")
    print(f"    brute MaxSim:    recall {h['brute']['recall']:.3f} at cost {h['brute']['cost']:.0f} "
          f"distance comps/query")
    print(f"    centroid-only:   recall {h['centroid']['recall']:.3f} at cost {h['centroid']['cost']:.0f} "
          f"(prune, no rerank)")
    print(f"    PLAID cascade:   recall {pick['recall']:.3f} at cost {pick['cost']:.0f} "
          f"(nprobe={pick['nprobe']}, keep={pick['keep']}) -> {h['brute']['cost'] / pick['cost']:.1f}x cheaper")
    print(f"    storage/filing:  raw multi-vector {s['raw_multi_bits']}b -> PLAID {s['plaid_bits']}b "
          f"({s['plaid_vs_raw']}x smaller)")
    print("  -> the conjunction match that needed one vector per token is now affordable to serve; "
          "raise keep when recall matters (the per-pair bound does not protect top-k).")


def viz_constants() -> None:
    """Print every measured number MultiVectorANNLaboratory.tsx mirrors to the decimal. Panel A
    (geometry + residual energy), Panel B (centroid-MaxSim grid + Cauchy-Schwarz bound), Panel C
    (recall/cost frontier), Panel D (storage). numpy scalars are cast to float/int."""
    fmt2 = lambda a: [[round(float(v), 3) for v in row] for row in a]

    # --- Panels A & B: the 2-D toy (per-K geometry baked; .tsx recomputes grids/bounds as closed forms) ---
    pool, query, doc, topic_mu = viz_toy()
    geom = toy_geometry_sweep()
    sweep = toy_bound_sweep()
    print("=== PANEL A & B — 2-D token toy (geometry + centroid-MaxSim grid + bound) ===")
    print(f"  POOL_2D = {fmt2(pool)}")
    print(f"  TOPIC_MU_2D = {fmt2(topic_mu)}")
    print(f"  QUERY_2D = {fmt2(query)}  DOC_2D = {fmt2(doc)}")
    print(f"  TRUE_MAXSIM = {round(float((query @ doc.T).max(axis=1).sum()), 3)}")
    print(f"  GEOM_SWEEP (per K: centroids, pool_assign, doc_assign) = {geom}")
    print(f"  BUDGET_TRADE (per K: full/centroid MaxSim, max_err, max_bound, residual_energy) = {sweep}")

    # --- Panel C: the single-knob keep-sweep frontier on the real 16-D corpus ---
    docs, queries, _ = token_corpus()
    h = head_to_head(docs, queries)
    sweep = keep_sweep(docs, queries)
    knee = next((r["keep"] for r in sweep if r["recall_pq"] >= 0.90), sweep[-1]["keep"])
    print("=== PANEL C — recall/cost frontier (16-D corpus, one shared truth) ===")
    print(f"  N_DOCS = {h['n_docs']}  NLIST = {h['nlist']}  TOPK = {h['topk']}  "
          f"M_Q = {QUERY_TOKENS}  M_D = {TOKENS_PER_DOC}")
    print(f"  BRUTE = {{'recall': {round(h['brute']['recall'], 4)}, 'cost': {round(h['brute']['cost'], 1)}}}")
    print(f"  CENTROID_ONLY = {{'recall': {round(h['centroid']['recall'], 4)}, "
          f"'cost': {round(h['centroid']['cost'], 1)}}}")
    print(f"  KNEE_KEEP = {knee}")
    print(f"  KEEP_SWEEP (full nprobe; exact rerank -> 1.0 = collapse anchor, PQ rerank plateaus) = {sweep}")

    # --- Panel D: storage (representative ColBERT scale, consistent with the late-interaction lab) ---
    s = storage_collapse(m_d=COLBERT_TOKENS, dim=COLBERT_DIM, nlist=REP_NLIST,
                         pq_m=REP_PQ_M, pq_kstar=REP_PQ_KSTAR)
    print("=== PANEL D — storage (bits per document, ColBERT scale d=128, 32 tokens) ===")
    print(f"  STORAGE = {s}")


def _run_all() -> None:
    print("multi_vector_ann_retrieval — verifying every claim:")
    test_cascade_collapses_to_brute()
    test_full_probe_no_prune_is_exact()
    test_centroid_maxsim_collapses_at_zero_residual()
    test_cauchy_schwarz_bound()
    test_token_cloud_separates()
    test_spherical_kmeans_equivalence()
    test_centroid_maxsim_approximates_full()
    test_recall_monotone_in_keep()
    test_plaid_beats_brute_at_equal_recall()
    test_plaid_vs_centroid_on_one_cloud()
    test_toy_bound_tightens_with_k()
    test_storage_collapse()
    print("\nfinance case study:")
    finance_demo()
    print("\nviz constants (mirrored to MultiVectorANNLaboratory.tsx):")
    viz_constants()
    print("\nall claims verified.")


if __name__ == "__main__":
    _run_all()
