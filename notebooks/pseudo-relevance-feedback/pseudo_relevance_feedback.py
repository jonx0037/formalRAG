"""Relevance Feedback and Query Expansion: Rocchio and RM3 — reference.

This module owns every number the topic depends on. Core scoring uses only the
standard library + numpy; the verification harness asserts the topic's claims:

  * pseudo-relevance feedback EXPANDS the query and improves recall by bridging a
    vocabulary mismatch — on the worked corpus the query "rate guidance" misses the
    synonym documents (which say "outlook"/"forecast", never "guidance") until
    RM3 (and Rocchio) add those terms from the top feedback documents;
  * the same mechanism DRIFTS when the feedback set is enlarged to include
    off-topic documents: recall rises with a little feedback and falls with too
    much (the query-drift failure), so PRF has no per-query guarantee (rigorFlag);
  * the RM3 interpolation recovers its endpoints — alpha=0 is the original query
    (the query-likelihood ranking) and alpha=1 is the pure relevance model;
  * the RM1 relevance model P(w|R) = sum_d P(w|d) P(d|q) is a proper distribution;
  * Rocchio moves the query vector toward the centroid of the feedback documents,
    raising its mean similarity to them, and likewise improves recall.

The query-likelihood machinery (Dirichlet-smoothed document models and P(q|d)) is
re-implemented in miniature here so the module is self-contained; the full
treatment is the query-likelihood-language-models topic, on whose KL/cross-entropy
view RM3's re-scoring rests. The re-ranking smoothing MU is small because the toy
documents are short — smoothing strength must scale to document length, the same
corpus-tuned-parameter honesty BM25's k1/b carry (rigorFlag).

Run:  uv run --with numpy python notebooks/pseudo-relevance-feedback/pseudo_relevance_feedback.py
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# --------------------------------------------------------------------------- #
# Tokenization (content terms only — PRF expansion is over content words)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOP = set("the a an and or of for on in to was were this that as is are our we year next ahead "
            "recently after one time flat held drove rose reflects lifts lifting raised improved "
            "revised updated full path more weaker effects continued beat quarter materially hawkish "
            "by with at from".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


# --------------------------------------------------------------------------- #
# Worked corpus — query "rate guidance" with a vocabulary mismatch.
#   r1, r4 : relevant, literal match (rate, guidance) and bridge-rich (outlook, forecast)
#   r2, r3 : relevant, SYNONYM only (rate + outlook/forecast, never "guidance")
#   n2, nb : non-relevant "guidance" blockers (outrank the synonym docs at baseline)
#   n1     : non-relevant rate-polysemy drift injector (tax rate); feeds tax terms
#   n5     : non-relevant drift target (shares tax/effective/charge with n1)
#   n3, n6 : pure off-topic
# --------------------------------------------------------------------------- #

_CORPUS = {
    "r1": "rate guidance outlook forecast outlook forecast",
    "r4": "rate guidance forecast outlook forecast outlook",
    "r2": "rate outlook forecast projection outlook forecast",
    "r3": "rate forecast outlook revision forecast outlook",
    "n2": "guidance segment headcount segment headcount",
    "nb": "guidance costs budget costs budget",
    "n1": "rate tax effective charge tax effective charge rate",
    "n5": "tax effective charge provision tax effective charge",
    "n3": "currency exposure translation currency exposure",
    "n6": "margin earnings dividend margin earnings",
}
_QUERY = "rate guidance"
_RELEVANT = {"r1", "r2", "r3", "r4"}

# Re-ranking / expansion hyperparameters (tuned to the toy corpus; see rigorFlag).
MU = 5.0          # Dirichlet smoothing for the re-ranking document model
ALPHA = 0.5       # RM3 interpolation weight on the relevance model
N_TERMS = 10      # number of expansion terms kept from RM1
K = 4             # evaluation cutoff (there are 4 relevant documents)


# --------------------------------------------------------------------------- #
# Index + language-model machinery (miniature query-likelihood)
# --------------------------------------------------------------------------- #

class Index:
    def __init__(self, corpus: dict[str, str]):
        self.ids = list(corpus)
        self.toks = {d: tokenize(corpus[d]) for d in self.ids}
        self.vocab = sorted({t for ts in self.toks.values() for t in ts})
        self.N = len(self.ids)
        self.avgdl = float(np.mean([len(self.toks[d]) for d in self.ids]))
        self.df = {t: sum(1 for d in self.ids if t in self.toks[d]) for t in self.vocab}
        cf = Counter(t for ts in self.toks.values() for t in ts)
        self.cf = cf
        self.coll_len = sum(cf.values())

    def doc_len(self, d: str) -> int:
        return len(self.toks[d])

    def p_collection(self, t: str) -> float:
        return self.cf[t] / self.coll_len

    def doc_model(self, d: str, mu: float = MU) -> dict[str, float]:
        c = Counter(self.toks[d]); dl = self.doc_len(d)
        return {t: (c.get(t, 0) + mu * self.p_collection(t)) / (dl + mu) for t in self.vocab}

    def p_query_given_doc(self, query: str, d: str, mu: float = MU) -> float:
        c = Counter(self.toks[d]); dl = self.doc_len(d); lp = 0.0
        for t in tokenize(query):
            p = (c.get(t, 0) + mu * self.p_collection(t)) / (dl + mu)
            lp += math.log(p) if p > 0 else -math.inf
        return math.exp(lp)


def bm25_rank(query: str, index: Index, k1: float = 1.5, b: float = 0.75) -> list[str]:
    """Initial retrieval (the ranking feedback documents are drawn from)."""
    scores = {}
    for d in index.ids:
        c = Counter(index.toks[d]); s = 0.0
        for t in set(tokenize(query)):
            if t in c:
                idf = math.log((index.N - index.df.get(t, 0) + 0.5) / (index.df.get(t, 0) + 0.5) + 1.0)
                tf = c[t]
                s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * index.doc_len(d) / index.avgdl))
        scores[d] = s
    return sorted(index.ids, key=lambda d: -scores[d])


# --------------------------------------------------------------------------- #
# RM1 / RM3 (Lavrenko-Croft relevance models)
# --------------------------------------------------------------------------- #

def rm1(feedback: list[str], index: Index, mu: float = MU) -> dict[str, float]:
    """P(w|R) = sum_d P(w|d) P(d|q), with P(d|q) proportional to the query
    likelihood P(q|d) over the feedback documents (uniform document prior)."""
    pq = {d: index.p_query_given_doc(_QUERY, d, mu) for d in feedback}
    z = sum(pq.values()) or 1.0
    return {t: sum(index.doc_model(d, mu)[t] * pq[d] / z for d in feedback) for t in index.vocab}


def rm3_model(feedback: list[str], index: Index, alpha: float = ALPHA,
              n_terms: int = N_TERMS, mu: float = MU) -> dict[str, float]:
    """RM3 = (1-alpha) * original query model + alpha * (top-n_terms of RM1)."""
    qc = Counter(tokenize(_QUERY)); qlen = sum(qc.values())
    qmodel = {t: qc.get(t, 0) / qlen for t in index.vocab}
    if not feedback:
        return qmodel
    pw = rm1(feedback, index, mu)
    top = set(sorted(pw, key=lambda t: -pw[t])[:n_terms])
    return {t: (1 - alpha) * qmodel.get(t, 0.0) + alpha * (pw[t] if t in top else 0.0)
            for t in index.vocab}


def rm3_rank(n_feedback: int, index: Index, alpha: float = ALPHA,
             n_terms: int = N_TERMS, mu: float = MU) -> list[str]:
    """Re-rank by cross-entropy of the (expanded) query model with each document
    model: score(d) = sum_w P_q(w) log P(w|d). With n_feedback=0 this is the
    original query-likelihood ranking."""
    base = bm25_rank(_QUERY, index)
    qmodel = rm3_model(base[:n_feedback], index, alpha, n_terms, mu)
    scores = {}
    for d in index.ids:
        dm = index.doc_model(d, mu)
        scores[d] = sum(qmodel[t] * math.log(dm[t]) for t in index.vocab if qmodel[t] > 0)
    return sorted(index.ids, key=lambda d: -scores[d])


# --------------------------------------------------------------------------- #
# Rocchio (vector-space relevance feedback)
# --------------------------------------------------------------------------- #

def tfidf_vector(d: str, index: Index) -> np.ndarray:
    c = Counter(index.toks[d])
    return np.array([(1 + math.log(c[t])) * math.log(1 + index.N / index.df[t]) if c.get(t, 0) > 0 else 0.0
                     for t in index.vocab])


def query_vector(index: Index) -> np.ndarray:
    c = Counter(tokenize(_QUERY))
    return np.array([(1 + math.log(c[t])) * math.log(1 + index.N / index.df.get(t, index.N)) if c.get(t, 0) > 0 else 0.0
                     for t in index.vocab])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def rocchio_query(n_feedback: int, index: Index, a: float = 1.0, b: float = 0.75) -> np.ndarray:
    """q' = a*q + b * centroid(top-n_feedback documents).  (gamma = 0: PRF.)"""
    q = query_vector(index)
    if n_feedback == 0:
        return q
    fb = bm25_rank(_QUERY, index)[:n_feedback]
    centroid = np.mean([tfidf_vector(d, index) for d in fb], axis=0)
    return a * q + b * centroid


def rocchio_rank(n_feedback: int, index: Index, a: float = 1.0, b: float = 0.75) -> list[str]:
    qp = rocchio_query(n_feedback, index, a, b)
    return sorted(index.ids, key=lambda d: -_cos(qp, tfidf_vector(d, index)))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def recall_at_k(ranking: list[str], k: int = K) -> float:
    return len(set(ranking[:k]) & _RELEVANT) / len(_RELEVANT)


# --------------------------------------------------------------------------- #
# Verification harness — the topic's claims, made executable
# --------------------------------------------------------------------------- #

def test_rm3_improves_recall() -> None:
    """PRF bridges the vocabulary mismatch: the baseline query misses the synonym
    documents, and a little RM3 feedback surfaces them, lifting recall@4 to 1.0."""
    index = Index(_CORPUS)
    base = recall_at_k(rm3_rank(0, index))
    fed = recall_at_k(rm3_rank(2, index))
    assert base == 0.5, base
    assert fed == 1.0, fed
    assert fed > base
    print(f"  [ok] RM3 query expansion improves recall@{K}: {base:.2f} -> {fed:.2f} (vocabulary bridged)")


def test_rm3_query_drift() -> None:
    """Too much feedback drifts: enlarging the feedback set to include off-topic
    documents reinforces their terms and demotes a relevant synonym document, so
    recall falls back below its peak — PRF has no per-query guarantee."""
    index = Index(_CORPUS)
    peak = recall_at_k(rm3_rank(2, index))
    over = recall_at_k(rm3_rank(4, index))
    assert peak == 1.0 and over <= 0.5, (peak, over)
    assert over < peak
    print(f"  [ok] query drift: recall@{K} falls {peak:.2f} -> {over:.2f} when feedback over-expands")


def test_rm3_alpha_limits() -> None:
    """RM3 recovers its endpoints: alpha=0 is the original query (the
    query-likelihood ranking, recall 0.5), and increasing alpha brings in the
    relevance model."""
    index = Index(_CORPUS)
    a0 = rm3_rank(3, index, alpha=0.0)
    base = rm3_rank(0, index)
    assert a0[:K] == base[:K], (a0[:K], base[:K])
    assert recall_at_k(a0) == 0.5
    assert recall_at_k(rm3_rank(2, index, alpha=0.5)) == 1.0   # mixing in RM1 helps
    print("  [ok] RM3 limits: alpha=0 recovers the original query-likelihood ranking")


def test_rm1_is_distribution() -> None:
    """The RM1 relevance model is a proper probability distribution over the
    vocabulary: nonnegative and summing to one."""
    index = Index(_CORPUS)
    for nfb in (1, 2, 3):
        pw = rm1(bm25_rank(_QUERY, index)[:nfb], index)
        total = sum(pw.values())
        assert abs(total - 1.0) < 1e-9, (nfb, total)
        assert all(v >= 0 for v in pw.values())
    print("  [ok] RM1 relevance model P(w|R) is a proper distribution (nonneg, sums to 1)")


def test_rocchio_moves_to_centroid_and_helps() -> None:
    """Rocchio moves the query toward the feedback centroid, raising its mean
    cosine to those documents, and improves recall@4 over the unexpanded query."""
    index = Index(_CORPUS)
    q = query_vector(index)
    fb = bm25_rank(_QUERY, index)[:2]
    qp = rocchio_query(2, index)
    mean_before = np.mean([_cos(q, tfidf_vector(d, index)) for d in fb])
    mean_after = np.mean([_cos(qp, tfidf_vector(d, index)) for d in fb])
    assert mean_after > mean_before, (mean_before, mean_after)
    # q' is exactly a*q + b*centroid
    centroid = np.mean([tfidf_vector(d, index) for d in fb], axis=0)
    assert np.allclose(qp, 1.0 * q + 0.75 * centroid)
    base = recall_at_k(rocchio_rank(0, index))
    fed = recall_at_k(rocchio_rank(2, index))
    assert fed > base, (base, fed)
    print(f"  [ok] Rocchio moves to the feedback centroid and improves recall@{K}: {base:.2f} -> {fed:.2f}")


# --------------------------------------------------------------------------- #
# Demos (printed, non-asserting) — feed the viz panels
# --------------------------------------------------------------------------- #

def recall_curve() -> None:
    index = Index(_CORPUS)
    print(f"  query={_QUERY!r}  relevant={sorted(_RELEVANT)}  (baseline misses the synonym docs)")
    print("  recall@4 vs #feedback documents (RM3 and Rocchio):")
    for nfb in range(0, 6):
        print(f"    n_fb={nfb}: RM3={recall_at_k(rm3_rank(nfb, index)):.2f}  "
              f"Rocchio={recall_at_k(rocchio_rank(nfb, index)):.2f}")


def expansion_terms() -> None:
    index = Index(_CORPUS)
    base = bm25_rank(_QUERY, index)
    for nfb in (2, 4):
        pw = rm1(base[:nfb], index)
        top = sorted(pw, key=lambda t: -pw[t])[:6]
        cells = "  ".join(f"{t}={pw[t]:.3f}" for t in top)
        print(f"  RM1 top terms (n_fb={nfb}, feedback={base[:nfb]}): {cells}")


def viz_constants() -> None:
    """The exact numbers QueryExpansionFeedbackLab.tsx mirrors: per feedback size
    the RM3 ranking, recall@4, and the top RM1 expansion terms with weights. The
    viz indexes this precomputed table by the feedback slider (the relevance-model
    arithmetic is not reproduced in TS)."""
    index = Index(_CORPUS)
    base = bm25_rank(_QUERY, index)
    print(f"  config: MU={MU}  ALPHA={ALPHA}  N_TERMS={N_TERMS}  K={K}")
    print(f"  relevant={sorted(_RELEVANT)}")
    for nfb in range(0, 6):
        ranking = rm3_rank(nfb, index)
        rec = recall_at_k(ranking)
        if nfb == 0:
            terms = "(none)"
        else:
            pw = rm1(base[:nfb], index)
            terms = ", ".join(f"{t}:{pw[t]:.3f}" for t in sorted(pw, key=lambda t: -pw[t])[:5])
        print(f"  n_fb={nfb}: recall@{K}={rec:.2f}  rank={ranking[:6]}")
        print(f"           expansion=[{terms}]")


if __name__ == "__main__":
    print("Relevance Feedback / Rocchio / RM3 verification harness")
    test_rm3_improves_recall()
    test_rm3_query_drift()
    test_rm3_alpha_limits()
    test_rm1_is_distribution()
    test_rocchio_moves_to_centroid_and_helps()
    print("Recall curve (improve then drift):")
    recall_curve()
    print("Expansion terms (clean vs polluted feedback):")
    expansion_terms()
    print("Viz constants (mirrored in QueryExpansionFeedbackLab.tsx):")
    viz_constants()
    print("All checks passed.")
