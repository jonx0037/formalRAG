"""Late interaction and learned sparse retrieval — the reference implementation for the formalRAG
`late-interaction-learned-sparse` topic.

`embedding-dimension-lower-bounds` proved a single pooled vector has a sign-rank ceiling: relevance
patterns it cannot realize below a critical dimension. This topic covers the two architectures that
ESCAPE that ceiling, by different mechanisms. Every pedagogical claim below is an `assert`:

  MOVEMENT 1 — MaxSim AND THE m=1 COLLAPSE (the one provable fact). Late interaction (ColBERT) keeps
    one vector per TOKEN and scores by MaxSim, S(q, d) = sum_i max_j <E(q_i), E(d_j)>. This is a max
    of inner products — piecewise-linear, NOT a single bilinear form q^T W d — so the rank-<=d
    argument does not apply. The one thing we can prove cleanly: with ONE vector per item the max is
    over a singleton, so MaxSim reduces EXACTLY to the dual-encoder dot product imported from DPR (the
    byte-for-byte anchor). (`maxsim_score`, `maxsim_matrix`.)

  MOVEMENT 2 — MULTI-VECTOR ESCAPES THE SINGLE-VECTOR WALL (demonstrated, not proved). Reusing the
    all-pairs qrel and the single-vector free-embedding wall from the embedding-dimension topic, a
    MaxSim model with m >= 2 vectors per document realizes the pattern at a corpus size PAST the
    single-vector critical n, at the same per-vector dimension. The loss is the IMPORTED qrel row loss
    (`_qrel_loss_grad`); only the MaxSim forward/backward (subgradient through each query token's
    argmax document token) is new. rigorFlag: this is a best-case DEMONSTRATION (free embeddings, the
    LIMIT methodology), not a matching lower bound — the general multi-vector expressivity theorem is
    OPEN (the LIMIT paper explicitly defers it). The cost is storage: m vectors per document.
    (`realize_qrel_maxsim`, `maxsim_critical_n`, `maxsim_escape_curve`.)

  MOVEMENT 3 — SPLADE, THE LEXICAL ESCAPE. Learned sparse retrieval expands into the high-dimensional
    SPARSE vocabulary space (|V| >> d), living in BM25's inverted index. A term's weight is
    w_j = max_i log(1 + ReLU(logit_ij)) over the input tokens, and a FLOPS regularizer L = sum_j
    mean_d(w)^2 controls sparsity. It fixes the vocabulary mismatch BM25 cannot: a query whose terms
    are ABSENT from a relevant document still retrieves it through learned expansion. We reuse BM25's
    index, ranking, and NDCG; the "MLM logits" here are a synthetic association stand-in, not a trained
    BERT. (`splade_weights`, `splade_rank`, `l0_sparsity`, `flops_term`.)

Honest caveats (rigorFlag territory): the escape is EMPIRICAL — MaxSim has no clean expressivity
theorem and no multi-vector sign-rank lower bound is known (the LIMIT paper defers it); the only
proved statement is the m=1 collapse to the dot product. SPLADE's FLOPS regularizer is a heuristic
penalty, not derived, and its relation to BM25 is conceptual, not a formal generalization. Late
interaction's storage cost (m vectors per document) is real; PLAID's centroid pruning (a downstream
topic) mitigates it. All token embeddings and MLM logits here are synthetic stand-ins reused from /
built on the InfoNCE / DPR / BM25 topics, not trained transformers.

This module imports its prerequisites (`dense-retrieval-dual-encoders` for the dot-product anchor,
`embedding-dimension-lower-bounds` for the wall and the qrel loss, `bm25` for the lexical world) and
their grand-prereqs — it never reimplements them. `viz_constants()` prints what
`LateInteractionLaboratory.tsx` mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/late-interaction-learned-sparse/late_interaction_learned_sparse.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

# Established cross-topic pattern: add EACH ancestor's HYPHENATED dir to the path, import the
# UNDERSCORED module. The direct prereqs are DPR (the dot-product anchor), embedding-dimension (the
# wall + the qrel loss), and BM25 (the lexical world); their grand-prereqs supply primitives those do
# not re-export, so we add every ancestor explicitly and never reimplement any of them.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("hypersphere-vmf-geometry", "the-retrieval-problem", "infonce-contrastive-objective",
             "dense-retrieval-dual-encoders", "embedding-dimension-lower-bounds", "bm25"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dense_retrieval_dual_encoders import dual_encoder_score, score_matrix     # noqa: E402
from embedding_dimension_lower_bounds import (                                 # noqa: E402
    allpairs_qrel,
    _qrel_loss_grad,
)
from bm25 import (                                                             # noqa: E402
    build_inverted_index,
    bm25_rank,
    bm25_contributions,
    idf_vector,
    tokenize,
    ndcg_at_k,
    _FINANCE_CORPUS as BM25_CORPUS,
)


# --------------------------------------------------------------------------- #
# Movement 1 — MaxSim and the m=1 collapse.
# --------------------------------------------------------------------------- #

def maxsim_score(Q: np.ndarray, D: np.ndarray) -> float:
    """The late-interaction MaxSim score of one query (m_q x d token vectors) against one document
    (m_d x d token vectors): sum over query tokens of the best-matching document token,
    S = sum_i max_j <Q_i, D_j>. A max of inner products, not a single bilinear form."""
    Q = np.atleast_2d(Q)
    D = np.atleast_2d(D)
    sims = Q @ D.T                                   # (m_q, m_d)
    return float(sims.max(axis=1).sum())


def maxsim_matrix(Qs: np.ndarray, Ds: np.ndarray) -> np.ndarray:
    """The full query-by-document MaxSim score matrix. Qs is (n_q, m_q, d), Ds is (n_d, m_d, d);
    returns (n_q, n_d) with entry [a,b] = sum_i max_j <Qs[a,i], Ds[b,j]>."""
    sims = np.einsum("aid,bjd->abij", Qs, Ds)        # (n_q, n_d, m_q, m_d)
    return sims.max(axis=3).sum(axis=2)


# --------------------------------------------------------------------------- #
# Movement 2 — multi-vector escapes the single-vector wall.
# --------------------------------------------------------------------------- #

def realize_qrel_maxsim(rel: np.ndarray, d: int, n_doc_vecs: int, n_query_vecs: int = 1,
                        restarts: int = 3, steps: int = 700, lr: float = 1.0, seed: int = 0):
    """Best-case (free-embedding) realizability of a qrel mask by a MaxSim model in per-vector
    dimension d with `n_doc_vecs` vectors per document (and `n_query_vecs` per query). A twin of the
    embedding-dimension single-vector `realize_qrel`: the row loss is the IMPORTED `_qrel_loss_grad`
    (the byte-for-byte anchor); only the MaxSim forward/backward is new — the gradient flows through
    each query token's argmax document token. Returns (realized, row_order_accuracy). At
    n_doc_vecs = n_query_vecs = 1 the score is the plain dot product, so this collapses to the
    single-vector case. GUARD: d, n_doc_vecs, n_query_vecs >= 1."""
    rel = np.asarray(rel, dtype=bool)
    nq, n = rel.shape
    d = max(1, int(d))
    mq = max(1, int(n_query_vecs))
    md = max(1, int(n_doc_vecs))
    k_per_row = rel.sum(axis=1)
    b_idx = np.broadcast_to(np.arange(n)[None, :, None], (nq, n, mq))
    best_acc = 0.0
    for r in range(restarts):
        rng = np.random.default_rng(seed + 6271 * r)
        Q = rng.standard_normal((nq, mq, d)) / math.sqrt(d)
        D = rng.standard_normal((n, md, d)) / math.sqrt(d)
        vQ = np.zeros_like(Q)
        vD = np.zeros_like(D)
        for _ in range(steps):
            sims = np.einsum("aid,bjd->abij", Q, D)          # (nq, n, mq, md)
            jstar = sims.argmax(axis=3)                       # (nq, n, mq): best doc token per (a,b,i)
            B = np.take_along_axis(sims, jstar[..., None], axis=3)[..., 0].sum(axis=2)  # (nq, n)
            _, dB = _qrel_loss_grad(B, rel)                  # imported anchor loss on the score matrix
            # Backprop through the argmax: each B[a,b] = sum_i <Q[a,i], D[b, jstar[a,b,i]]>.
            Dstar = D[b_idx, jstar]                           # (nq, n, mq, d): the matched doc tokens
            gQ = np.einsum("ab,abid->aid", dB, Dstar)        # dQ[a,i] = sum_b dB[a,b] D[b, jstar]
            gD = np.zeros_like(D)
            contrib = dB[:, :, None, None] * Q[:, None, :, :]  # (nq, n, mq, d)
            np.add.at(gD, (b_idx.ravel(), jstar.ravel()), contrib.reshape(-1, d))
            vQ = 0.9 * vQ - lr * gQ
            vD = 0.9 * vD - lr * gD
            Q = Q + vQ
            D = D + vD
        B = maxsim_matrix(Q, D)
        correct = 0
        for q in range(nq):
            k = int(k_per_row[q])
            if k == 0:
                continue
            top = np.argpartition(-B[q], kth=k - 1)[:k]
            correct += int(set(top.tolist()) == set(np.flatnonzero(rel[q]).tolist()))
        best_acc = max(best_acc, correct / nq)
        if best_acc >= 1.0:
            break
    return (best_acc >= 1.0 - 1e-12), float(best_acc)


def maxsim_critical_n(d: int, n_grid, n_doc_vecs: int, **kw) -> int:
    """The critical document count of a MaxSim model at per-vector dimension d with `n_doc_vecs`
    vectors per document: the largest n in `n_grid` whose all-pairs qrel it realizes. The multi-vector
    counterpart of the single-vector `free_embedding_critical_n`."""
    crit = 0
    for n in n_grid:
        realized, _ = realize_qrel_maxsim(allpairs_qrel(int(n)), d, n_doc_vecs, **kw)
        if realized:
            crit = int(n)
        else:
            break
    return crit


def maxsim_escape_curve(d: int, n_grid, n_doc_vecs: int = 2, **kw):
    """For a fixed per-vector dimension d, the single-vector vs MaxSim row-order accuracy across
    corpus sizes — the wall and its escape on one figure. Both baselines use the SAME optimizer
    (`realize_qrel_maxsim`, single = one vector per item, MaxSim = `n_doc_vecs`), so the gap is purely
    the multi-vector effect — one cloud, not a comparison across optimizers. Returns a list of
    {n, single, maxsim} accuracies."""
    rows = []
    for n in n_grid:
        rel = allpairs_qrel(int(n))
        _, single = realize_qrel_maxsim(rel, d, 1, **kw)
        _, multi = realize_qrel_maxsim(rel, d, n_doc_vecs, **kw)
        rows.append({"n": int(n), "single": round(single, 4), "maxsim": round(multi, 4)})
    return rows


def storage_cost(n_docs: int, avg_tokens: int, dim: int) -> dict:
    """The storage cost law: a single-vector index stores n_docs vectors; late interaction stores
    one vector per token, n_docs * avg_tokens. Returns the float counts and the multiplier."""
    single = n_docs * dim
    multi = n_docs * avg_tokens * dim
    return {"single_floats": int(single), "multi_floats": int(multi), "multiplier": int(avg_tokens)}


# --------------------------------------------------------------------------- #
# Movement 3 — SPLADE: the learned sparse escape (reusing the BM25 lexical world).
# --------------------------------------------------------------------------- #

# A synthetic association matrix standing in for a trained MLM head: each input term activates a few
# vocabulary terms with a logit. Identity activations preserve exact matches; the off-diagonal
# activations are the LEARNED EXPANSION. (Not a trained BERT — the rigorFlag is explicit about this.)
# Two roles: the QUERY-side mismatch terms (borrowing / costs) share NO term with the on-point filing
# yet expand into its interest/rate/exposure vocabulary; the CONTENT-term expansions give documents a
# spread of weight magnitudes (identity log(1+4) high, expansions lower), so the sparsity threshold
# genuinely prunes terms and the FLOPS/L0 trade-off is real, not flat.
_IDENTITY_LOGIT = 4.0
_ASSOC = {
    # query-side mismatch bridges
    "borrowing": {"interest": 3.0, "rate": 2.5, "exposure": 1.0},
    "borrow": {"interest": 3.0, "rate": 2.5},
    "costs": {"rate": 2.0, "exposure": 1.5, "margin": 1.0},
    "cost": {"rate": 2.0, "exposure": 1.5, "margin": 1.0},
    "sensitivity": {"sensitive": 3.0, "exposure": 2.0, "rate": 1.0},
    # content-term expansions (in-vocab targets) — the document-side learned expansion
    "interest": {"rate": 2.0, "margin": 1.5, "exposure": 1.0},
    "rate": {"interest": 2.0, "policy": 1.5, "exposure": 1.0},
    "exposure": {"risk": 2.0, "currency": 1.5, "sensitive": 1.0},
    "margin": {"interest": 1.5, "rate": 1.0},
    "sensitive": {"exposure": 1.5, "rate": 1.0},
    "currency": {"exchange": 2.0, "foreign": 1.5, "exposure": 1.0},
    "exchange": {"currency": 2.0, "foreign": 1.5},
    "foreign": {"currency": 1.5, "exchange": 1.0},
    "policy": {"rate": 2.0, "central": 1.5, "bank": 1.0},
    "risk": {"exposure": 2.0, "factors": 1.0},
}
SPLADE_QUERY = "borrowing costs"


def _vocab_of(corpus: dict[str, str]) -> dict[str, int]:
    """The shared lexical vocabulary (BM25's inverted-index vocab), term -> column."""
    return build_inverted_index(corpus).vocab


def splade_weights(text: str, vocab: dict[str, int], assoc: dict = None) -> np.ndarray:
    """The SPLADE sparse representation of a piece of text over the vocabulary: for each vocabulary
    term j, w_j = max over input tokens i of log(1 + ReLU(logit_ij)). Identity activations
    (`_IDENTITY_LOGIT`) preserve exact lexical matches; `assoc` adds the learned expansion. Returns a
    dense |V| vector that is sparse in support."""
    assoc = _ASSOC if assoc is None else assoc
    w = np.zeros(len(vocab))
    for tok in tokenize(text):
        # exact (identity) activation
        if tok in vocab:
            w[vocab[tok]] = max(w[vocab[tok]], math.log1p(max(_IDENTITY_LOGIT, 0.0)))
        # learned expansion activations
        for term, logit in assoc.get(tok, {}).items():
            if term in vocab:
                j = vocab[term]
                w[j] = max(w[j], math.log1p(max(logit, 0.0)))     # log(1 + ReLU(logit))
    return w


def l0_sparsity(w: np.ndarray, tau: float = 0.0) -> int:
    """The number of active (non-pruned) terms: |{j : w_j > tau}|. Higher tau (stronger sparsity
    pressure) prunes more terms."""
    return int(np.sum(np.asarray(w) > tau))


def flops_term(W: np.ndarray) -> float:
    """The FLOPS regularizer value sum_j (mean_d w_dj)^2 over a document-by-vocabulary weight matrix
    W — the expected number of nonzero query-document term products, the quantity SPLADE penalizes to
    stay cheap in the inverted index. GUARD: empty -> 0.0."""
    W = np.atleast_2d(W)
    if W.size == 0:
        return 0.0
    return float(np.sum(W.mean(axis=0) ** 2))


def splade_rank(query: str, corpus: dict[str, str], vocab: dict[str, int] = None,
                assoc: dict = None, tau: float = 0.0) -> list[tuple[str, float]]:
    """Rank documents by the SPLADE sparse dot product <splade(query), splade(doc)>, with weights
    below `tau` pruned (the sparsity knob). Returns (doc_id, score) sorted descending."""
    vocab = _vocab_of(corpus) if vocab is None else vocab
    qw = splade_weights(query, vocab, assoc)
    qw = np.where(qw > tau, qw, 0.0)
    scored = []
    for doc_id, text in corpus.items():
        dw = splade_weights(text, vocab, assoc)
        dw = np.where(dw > tau, dw, 0.0)
        scored.append((doc_id, float(qw @ dw)))
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


def splade_doc_matrix(corpus: dict[str, str], vocab: dict[str, int] = None,
                      assoc: dict = None, tau: float = 0.0) -> np.ndarray:
    """The document-by-vocabulary SPLADE weight matrix (for the FLOPS regularizer and sparsity)."""
    vocab = _vocab_of(corpus) if vocab is None else vocab
    rows = []
    for text in corpus.values():
        w = splade_weights(text, vocab, assoc)
        rows.append(np.where(w > tau, w, 0.0))
    return np.array(rows)


# --------------------------------------------------------------------------- #
# Module constants the viz panels step through.
# --------------------------------------------------------------------------- #

ESCAPE_D = 4                         # the per-vector dimension where the escape is shown
ESCAPE_N_GRID = (4, 6, 8, 10, 12)    # corpus sizes for the escape curve
N_DOC_VECS = 2                       # vectors per document for the MaxSim model
TAU_GRID = (0.0, 0.7, 0.95, 1.2, 1.45)  # SPLADE sparsity thresholds (prune log(1+logit) by magnitude)
COLBERT_AVG_TOKENS = 32              # representative tokens/doc for the storage-cost headline
COLBERT_DIM = 128                    # representative ColBERT per-token dimension


def _splade_query_demo():
    """The vocab-mismatch instance: BM25 finds no overlap for the mismatch query; SPLADE expands it
    into the on-point filing's vocabulary. Returns (vocab, bm25_top, splade_top, bm25_onpoint_score)."""
    index = build_inverted_index(BM25_CORPUS)
    vocab = index.vocab
    bm25_order = bm25_rank(SPLADE_QUERY, index, k1=1.5, b=0.75)
    idf = idf_vector(index)
    onpoint_bm25 = sum(bm25_contributions(SPLADE_QUERY, index.doc_ids.index("filing-onpoint"),
                                          index, idf).values())
    splade_order = splade_rank(SPLADE_QUERY, BM25_CORPUS, vocab)
    return vocab, bm25_order, splade_order, float(onpoint_bm25)


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_maxsim_reduces_to_dot_at_m1() -> None:
    """Movement 1, the provable anchor: with one vector per item MaxSim is exactly the dual-encoder
    dot product imported from DPR — the max over a singleton is the identity."""
    rng = np.random.default_rng(0)
    nq, n, d = 6, 5, 8
    Qs = rng.standard_normal((nq, 1, d))
    Ds = rng.standard_normal((n, 1, d))
    M_maxsim = maxsim_matrix(Qs, Ds)
    M_dot = score_matrix(Qs[:, 0, :], Ds[:, 0, :])           # imported DPR dot-product score matrix
    assert np.allclose(M_maxsim, M_dot, atol=1e-12), "MaxSim(m=1) must equal the dot product"
    # And row by row against the imported dual_encoder_score.
    for a in range(nq):
        row = dual_encoder_score(Qs[a, 0, :], Ds[:, 0, :])
        assert np.allclose(M_maxsim[a], row, atol=1e-12), f"row {a} disagrees with dual_encoder_score"
    print("  [ok] Movement 1: MaxSim with one vector per item == imported DPR dot product (|diff|<1e-12)")


def test_maxsim_is_max_of_linear() -> None:
    """Movement 1: MaxSim is a max of inner products, strictly above any single token pairing, so it
    is not a single bilinear form. With two document vectors it can give a query a high score via
    EITHER vector — the nonlinearity a single pooled vector lacks."""
    q = np.array([[1.0, 0.0]])
    D = np.array([[1.0, 0.0], [0.0, 1.0]])                    # two orthogonal document tokens
    s = maxsim_score(q, D)
    assert abs(s - 1.0) < 1e-12, "MaxSim should pick the aligned document token"
    # A second query aligned to the OTHER token also scores high — one document, two 'meanings'.
    q2 = np.array([[0.0, 1.0]])
    assert abs(maxsim_score(q2, D) - 1.0) < 1e-12, "the same document matches an orthogonal query too"
    # A single pooled document vector (the average) cannot match both: its max alignment is lower.
    pooled = D.mean(axis=0, keepdims=True)
    assert maxsim_score(q, pooled) < s - 1e-6, "a single pooled vector cannot match both queries"
    print("  [ok] Movement 1: MaxSim is a max of linear forms — one document matches orthogonal "
          "queries a pooled vector cannot")


def test_multivector_escapes_wall() -> None:
    """Movement 2, the headline (demonstrated): at a corpus size past the single-vector critical n, a
    MaxSim model with two vectors per document realizes the all-pairs qrel that a single vector cannot
    — at the SAME per-vector dimension and the SAME optimizer (single = one vector per item), so the
    gap is purely the multi-vector effect, not a difference between optimizers."""
    d = ESCAPE_D
    crit_single = maxsim_critical_n(d, ESCAPE_N_GRID, 1)
    n_past = next((n for n in ESCAPE_N_GRID if n > crit_single), ESCAPE_N_GRID[-1])
    rel = allpairs_qrel(n_past)
    single_ok, single_acc = realize_qrel_maxsim(rel, d, 1)
    multi_ok, multi_acc = realize_qrel_maxsim(rel, d, N_DOC_VECS)
    assert not single_ok, f"single vector (m=1) should fail past its critical n ({n_past}>{crit_single})"
    assert multi_ok, f"MaxSim with {N_DOC_VECS} vectors should realize n={n_past} at d={d}"
    assert multi_acc > single_acc, "MaxSim should beat the single vector at the wall"
    print(f"  [ok] Movement 2: at d={d}, n={n_past} (> single-vector critical n {crit_single}) — "
          f"single fails (acc {single_acc:.2f}), MaxSim(m={N_DOC_VECS}) realizes it (acc {multi_acc:.2f})")


def test_maxsim_critical_n_exceeds_single() -> None:
    """Movement 2: the MaxSim critical n is at least the single-vector critical n, and strictly larger
    at the demonstration dimension — more vectors per document admit a larger combinatorial corpus."""
    d = ESCAPE_D
    crit_single = maxsim_critical_n(d, ESCAPE_N_GRID, 1)
    crit_multi = maxsim_critical_n(d, ESCAPE_N_GRID, N_DOC_VECS)
    assert crit_multi > crit_single, f"MaxSim should admit more docs: {crit_multi} vs {crit_single}"
    print(f"  [ok] Movement 2: critical n at d={d} — single {crit_single} < MaxSim(m={N_DOC_VECS}) {crit_multi}")


def test_storage_cost() -> None:
    """Movement 2 cost: late interaction stores one vector per token, so its index is avg_tokens times
    larger than a single-vector index — the price of the escape."""
    cost = storage_cost(1000, COLBERT_AVG_TOKENS, COLBERT_DIM)
    assert cost["multi_floats"] == cost["single_floats"] * COLBERT_AVG_TOKENS, "cost law wrong"
    assert cost["multiplier"] == COLBERT_AVG_TOKENS
    print(f"  [ok] Movement 2 cost: late interaction stores {COLBERT_AVG_TOKENS}x the floats of a "
          f"single-vector index")


def test_splade_fixes_vocab_mismatch() -> None:
    """Movement 3, the headline: the mismatch query shares no term with the on-point filing, so BM25
    scores it zero; SPLADE expands the query into the filing's vocabulary and ranks it first."""
    vocab, bm25_order, splade_order, onpoint_bm25 = _splade_query_demo()
    # BM25 has no lexical overlap -> the on-point filing scores exactly zero.
    assert onpoint_bm25 == 0.0, f"BM25 should score the mismatch query 0 on the filing, got {onpoint_bm25}"
    assert bm25_order[0][1] == 0.0, "BM25 cannot rank the mismatch query (all scores zero)"
    # SPLADE expansion retrieves the on-point filing at the top.
    assert splade_order[0][0] == "filing-onpoint", f"SPLADE top-1 should be filing-onpoint, got {splade_order[0]}"
    assert splade_order[0][1] > 0.0, "SPLADE should give the on-point filing a positive score"
    print(f"  [ok] Movement 3: 'borrowing costs' — BM25 scores the on-point filing {onpoint_bm25:.1f} "
          f"(no overlap); SPLADE expansion ranks it #1 (score {splade_order[0][1]:.3f})")


def test_flops_controls_sparsity() -> None:
    """Movement 3: raising the sparsity threshold (the FLOPS pressure) monotonically prunes terms and
    lowers the FLOPS value, but eventually prunes the bridging expansion and breaks the mismatch fix —
    the sparsity-versus-quality trade-off."""
    vocab = _vocab_of(BM25_CORPUS)
    l0s, flops, fixed = [], [], []
    for tau in TAU_GRID:
        W = splade_doc_matrix(BM25_CORPUS, vocab, tau=tau)
        l0s.append(int((W > 0).sum()))
        flops.append(flops_term(W))
        top_id, top_score = splade_rank(SPLADE_QUERY, BM25_CORPUS, vocab, tau=tau)[0]
        fixed.append(top_id == "filing-onpoint" and top_score > 0.0)  # retrieved on merit, not a 0-tie
    assert all(l0s[i] >= l0s[i + 1] for i in range(len(l0s) - 1)), f"L0 should not rise with tau: {l0s}"
    assert all(flops[i] >= flops[i + 1] - 1e-9 for i in range(len(flops) - 1)), f"FLOPS should fall: {flops}"
    assert fixed[0] and not fixed[-1], "the mismatch fix should hold at low tau and break at high tau"
    print(f"  [ok] Movement 3: sparsity trade-off — L0 {l0s} (down with tau), mismatch fixed {fixed} "
          f"(breaks once expansion is pruned)")


# --------------------------------------------------------------------------- #
# Finance demo — the two escapes, printed.
# --------------------------------------------------------------------------- #

def finance_demo() -> dict:
    """The two escapes on the finance thread: ColBERT MaxSim lifts the combinatorial dense wall that
    broke the single vector; SPLADE expansion answers the lexical-mismatch query BM25 cannot."""
    d = ESCAPE_D
    crit_single = maxsim_critical_n(d, ESCAPE_N_GRID, 1)
    crit_multi = maxsim_critical_n(d, ESCAPE_N_GRID, N_DOC_VECS)
    print(f"  COLBERT (dense escape): at per-vector d = {d}, a single vector realizes all-pairs up to "
          f"{crit_single} documents; MaxSim with {N_DOC_VECS} vectors reaches {crit_multi}")
    vocab, bm25_order, splade_order, onpoint_bm25 = _splade_query_demo()
    print(f"  SPLADE (lexical escape): query {SPLADE_QUERY!r} — BM25 on-point score {onpoint_bm25:.1f} "
          f"(no overlap); SPLADE ranks {splade_order[0][0]} #1 ({splade_order[0][1]:.3f})")
    cost = storage_cost(1000, COLBERT_AVG_TOKENS, COLBERT_DIM)
    print(f"  COST: late interaction stores {cost['multiplier']}x the floats of a single-vector index")
    return {"crit_single": crit_single, "crit_multi": crit_multi,
            "splade_top": splade_order[0][0], "onpoint_bm25": round(onpoint_bm25, 3)}


# --------------------------------------------------------------------------- #
# Viz constants — printed for LateInteractionLaboratory.tsx to mirror.
# --------------------------------------------------------------------------- #

def viz_constants() -> None:
    """Print Panel A (a MaxSim token grid), Panel B (the single-vector wall vs the MaxSim escape), and
    Panel C (SPLADE weights + the sparsity trade-off) — all baked to the decimal in the .tsx. TS
    recomputes only CLOSED FORM (L0 counts, MaxSim of the baked token vectors); every MEASURED number
    (the escape accuracies, the SPLADE weights and trade-off) is baked here."""
    print("  PANEL A — a MaxSim token grid (query tokens x document tokens):")
    # A designed, legible grid: 3 query tokens, 4 document tokens as unit vectors on the circle, each
    # query token aligned near a DISTINCT document token — the many-to-many matching MaxSim captures
    # and a single pooled vector cannot. Entries are cosines in [-1, 1].
    ang_d = np.array([0.0, 90.0, 180.0, 270.0]) * np.pi / 180.0
    ang_q = np.array([12.0, 84.0, 200.0]) * np.pi / 180.0
    Dtok = np.round(np.c_[np.cos(ang_d), np.sin(ang_d)], 3)
    Qtok = np.round(np.c_[np.cos(ang_q), np.sin(ang_q)], 3)
    grid = np.round(Qtok @ Dtok.T, 3)
    print(f"    QTOK = {Qtok.tolist()}")
    print(f"    DTOK = {Dtok.tolist()}")
    print(f"    SIM_GRID = {grid.tolist()}   # per query-token max is the MaxSim contribution")
    print(f"    MAXSIM = {round(float(grid.max(axis=1).sum()), 3)}, "
          f"POOLED = {round(maxsim_score(Qtok, Dtok.mean(axis=0, keepdims=True)), 3)}")

    print("  PANEL B — the single-vector wall vs the MaxSim escape:")
    curve = maxsim_escape_curve(ESCAPE_D, ESCAPE_N_GRID, N_DOC_VECS)
    print(f"    ESCAPE_D = {ESCAPE_D}, N_DOC_VECS = {N_DOC_VECS}, N_GRID = {list(ESCAPE_N_GRID)}")
    print(f"    ESCAPE_CURVE = {curve}")
    print(f"    CRIT_SINGLE = {maxsim_critical_n(ESCAPE_D, ESCAPE_N_GRID, 1)}, "
          f"CRIT_MAXSIM = {maxsim_critical_n(ESCAPE_D, ESCAPE_N_GRID, N_DOC_VECS)}")
    cost = storage_cost(1000, COLBERT_AVG_TOKENS, COLBERT_DIM)
    print(f"    STORAGE = {cost}")

    print("  PANEL C — SPLADE: learned sparse weights and the sparsity trade-off:")
    vocab = _vocab_of(BM25_CORPUS)
    qw = splade_weights(SPLADE_QUERY, vocab)
    inv = {j: t for t, j in vocab.items()}
    active = sorted([(inv[j], round(float(qw[j]), 3)) for j in np.flatnonzero(qw)],
                    key=lambda kv: -kv[1])
    print(f"    SPLADE_QUERY = {SPLADE_QUERY!r}")
    print(f"    QUERY_WEIGHTS = {active}   # exact term has none; all are learned expansion")
    trade = []
    for tau in TAU_GRID:
        W = splade_doc_matrix(BM25_CORPUS, vocab, tau=tau)
        top_id, top_score = splade_rank(SPLADE_QUERY, BM25_CORPUS, vocab, tau=tau)[0]
        trade.append({"tau": round(float(tau), 2), "l0": int((W > 0).sum()),
                      "flops": round(flops_term(W), 4),
                      "fixed": top_id == "filing-onpoint" and top_score > 0.0})
    print(f"    TRADE = {trade}")


if __name__ == "__main__":
    print("Late interaction & learned sparse — verification harness")
    test_multivector_escapes_wall()                # the headline runs first
    test_maxsim_reduces_to_dot_at_m1()
    test_maxsim_is_max_of_linear()
    test_maxsim_critical_n_exceeds_single()
    test_storage_cost()
    test_splade_fixes_vocab_mismatch()
    test_flops_controls_sparsity()
    print("Finance demo:")
    finance_demo()
    print("Viz constants (mirrored to the decimal in LateInteractionLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
