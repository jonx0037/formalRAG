"""The Vector Space Model and TF-IDF from scratch — reference implementation.

This module owns every number the topic depends on. Core scoring uses only
numpy; the verification harness asserts the topic's mathematical claims:
IDF is exactly the self-information -log(df/N) of a term's presence (Theorem 1),
sublinear tf 1+log(tf) is strictly increasing and concave (Proposition 1),
cosine normalization is invariant to pure magnitude scaling (Proposition 2),
and on the shared finance corpus the raw tf-idf dot product is hijacked by a
padded transcript while cosine normalization surfaces the concise on-point
filing (the length-hijack flip). It cross-checks the cosine ranking against
scikit-learn's TfidfVectorizer when available.

Two IDF forms, on purpose. Theorem 1 is exact for the unsmoothed textbook IDF
log(N/df) = -log(df/N), and we teach it there (a term in every document, like
'rate' in this corpus, carries 0 bits and drops out). Scoring uses the standard
*smoothed* form log(1 + N/df) so universal terms keep a small positive weight
and there is no division by zero — smoothing is a practical convention, not a
theorem (rigorFlag), the same exact-vs-smoothed split BM25 makes with its RSJ
weight. The default idf_variant is therefore 'smooth' for the scoring functions
and 'textbook' is passed explicitly where the self-information identity is shown.

The finance corpus is reused VERBATIM from notebooks/bm25/bm25.py so the
length-hijack flip is provably the same phenomenon BM25 fixes with its b
parameter — the lexical-retrieval arc the-retrieval-problem -> VSM -> BM25
scores one corpus with three escalating fixes.

Run:  uv run --with numpy --with scikit-learn python notebooks/vector-space-model-tfidf/vector_space_model_tfidf.py
The scikit-learn cross-check is skipped gracefully if the package is absent.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Tokenization + inverted index  (same shape as notebooks/bm25/bm25.py)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class InvertedIndex:
    vocab: dict[str, int]            # term -> column index
    tf: np.ndarray                   # (n_docs, n_terms) raw term frequencies
    doc_len: np.ndarray             # (n_docs,) document lengths
    df: np.ndarray                  # (n_terms,) document frequencies
    n_docs: int
    doc_ids: list[str]


def build_inverted_index(corpus: dict[str, str]) -> InvertedIndex:
    doc_ids = list(corpus)
    tokenized = [tokenize(corpus[d]) for d in doc_ids]
    vocab: dict[str, int] = {}
    for toks in tokenized:
        for t in toks:
            vocab.setdefault(t, len(vocab))

    tf = np.zeros((len(doc_ids), len(vocab)))
    doc_len = np.zeros(len(doc_ids))
    for i, toks in enumerate(tokenized):
        doc_len[i] = len(toks)
        for term, count in Counter(toks).items():
            tf[i, vocab[term]] = count
    df = (tf > 0).sum(axis=0)
    return InvertedIndex(vocab, tf, doc_len, df, len(doc_ids), doc_ids)


# --------------------------------------------------------------------------- #
# IDF — the self-information of a term's presence
#   textbook : log(N / df)            == -log(df / N) == self-information (Thm 1)
#   smooth   : log(1 + N / df)        keeps every term strictly positive
# --------------------------------------------------------------------------- #

def idf_scalar(df: float, n_docs: int, variant: str = "textbook", base: float | None = None) -> float:
    """Single-term IDF. With base=2 the textbook variant reads out in bits of surprise."""
    log = math.log if base is None else (lambda x: math.log(x, base))
    if variant == "textbook":
        return log(n_docs / df) if df > 0 else 0.0
    if variant == "smooth":
        return log(1.0 + n_docs / df) if df > 0 else 0.0
    raise ValueError(variant)


def idf_vector(index: InvertedIndex, variant: str = "textbook") -> np.ndarray:
    df, n = index.df.astype(float), index.n_docs
    if variant == "textbook":
        return np.where(df > 0, np.log(n / np.maximum(df, 1.0)), 0.0)
    if variant == "smooth":
        return np.where(df > 0, np.log(1.0 + n / np.maximum(df, 1.0)), 0.0)
    raise ValueError(variant)


# --------------------------------------------------------------------------- #
# Term-frequency scaling and TF-IDF weight vectors
# --------------------------------------------------------------------------- #

def tf_scale(tf: np.ndarray, sublinear: bool) -> np.ndarray:
    """Raw tf, or the sublinear 1 + log(tf) transform (0 where tf == 0)."""
    if not sublinear:
        return tf
    out = np.zeros_like(tf, dtype=float)
    nz = tf > 0
    out[nz] = 1.0 + np.log(tf[nz])
    return out


def weight_matrix(index: InvertedIndex, sublinear: bool = True,
                  idf_variant: str = "smooth") -> np.ndarray:
    """The (n_docs, n_terms) TF-IDF weight matrix w_{t,d} = scale(tf) * idf_t."""
    return tf_scale(index.tf, sublinear) * idf_vector(index, idf_variant)


def query_weights(query: str, index: InvertedIndex, sublinear: bool = True,
                  idf_variant: str = "smooth") -> np.ndarray:
    """TF-IDF weight vector for the query (query tf folded through the same scaling)."""
    idf = idf_vector(index, idf_variant)
    qtf = np.zeros(len(index.vocab))
    for term, count in Counter(tokenize(query)).items():
        j = index.vocab.get(term)
        if j is not None:
            qtf[j] = count
    return tf_scale(qtf, sublinear) * idf


def vsm_contributions(query: str, doc_i: int, index: InvertedIndex, sublinear: bool = True,
                      idf_variant: str = "smooth") -> dict[str, float]:
    """Per-query-term contributions to the raw (unnormalized) dot product."""
    q = query_weights(query, index, sublinear, idf_variant)
    w = weight_matrix(index, sublinear, idf_variant)[doc_i]
    out: dict[str, float] = {}
    for term in dict.fromkeys(tokenize(query)):
        j = index.vocab.get(term)
        if j is not None and q[j] * w[j] != 0.0:
            out[term] = float(q[j] * w[j])
    return out


def vsm_rank(query: str, index: InvertedIndex, normalized: bool = True, sublinear: bool = True,
             idf_variant: str = "smooth") -> list[tuple[str, float]]:
    """Rank documents by TF-IDF similarity. normalized=True -> cosine; False -> raw dot product."""
    q = query_weights(query, index, sublinear, idf_variant)
    w = weight_matrix(index, sublinear, idf_variant)
    raw = w @ q                                   # (n_docs,) raw dot products
    if normalized:
        doc_norms = np.linalg.norm(w, axis=1)
        qnorm = np.linalg.norm(q)
        denom = doc_norms * qnorm
        scores = np.where(denom > 0, raw / np.where(denom > 0, denom, 1.0), 0.0)
    else:
        scores = raw
    order = np.argsort(scores)[::-1]
    return [(index.doc_ids[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------- #
# Shared finance corpus — VERBATIM from notebooks/bm25/bm25.py
# --------------------------------------------------------------------------- #

_FILLER = ("the quarter operations regions headcount capex marketing supply chain inventory "
           "guidance segment logistics demand pricing momentum ") * 15  # ~240 query-free tokens
_FINANCE_CORPUS = {
    "filing-onpoint": "net interest margin is sensitive to interest rate moves and our rate exposure is disclosed",
    "transcript-pad": _FILLER + " interest interest interest rate rate rate rate exposure exposure",
    "filing-fx": "foreign exchange exposure and currency rate movements affect our exposure to translation",
    "news-macro": "the central bank raised the policy rate again as interest rate decisions weigh on markets",
    "filing-boiler": "this filing contains forward looking statements regarding interest rate and other risk exposure factors",
    "transcript-short": "quick update on interest and the rate outlook for the quarter",
}
_QUERY = "interest rate exposure"


# --------------------------------------------------------------------------- #
# Verification harness — the topic's claims, made executable
# --------------------------------------------------------------------------- #

def test_idf_is_self_information() -> None:
    """Theorem 1: idf_t = -log(df_t / N) = log(N / df_t), the self-information of a
    term's presence under a uniform-document draw. Exact, in any base."""
    index = build_inverted_index(_FINANCE_CORPUS)
    n = index.n_docs
    idf = idf_vector(index, "textbook")
    for term, j in index.vocab.items():
        df = index.df[j]
        assert abs(idf[j] - (-math.log(df / n))) < 1e-12, term
    # Corner cases: a term in every document carries 0 bits; a singleton carries log N.
    assert abs(idf_scalar(n, n, "textbook")) < 1e-12          # df == N -> 0 surprise
    assert abs(idf_scalar(1, n, "textbook") - math.log(n)) < 1e-12  # singleton -> log N
    # Base-2 reads out in bits.
    assert abs(idf_scalar(1, n, "textbook", base=2) - math.log2(n)) < 1e-12
    print("  [ok] IDF is the self-information -log(df/N) (0 bits at df=N, log N for a singleton)")


def test_sublinear_tf_monotone_concave() -> None:
    """Proposition 1: g(x) = 1 + log(x) is strictly increasing and strictly concave
    on x >= 1, so each extra occurrence adds strictly less weight than the previous."""
    xs = np.arange(1.0, 50.0, 0.5)
    g = 1.0 + np.log(xs)
    assert np.all(np.diff(g) > 0), "not strictly increasing"
    second_diff = g[2:] - 2.0 * g[1:-1] + g[:-2]
    assert np.all(second_diff < 0), "not strictly concave"
    print("  [ok] sublinear tf 1+log(tf) is strictly increasing and concave")


def test_cosine_normalization_invariant() -> None:
    """Proposition 2: scaling a document's raw counts by c > 0 leaves cosine
    unchanged while the raw dot product scales by c."""
    index = build_inverted_index(_FINANCE_CORPUS)
    w = weight_matrix(index, sublinear=False, idf_variant="smooth")  # linear so scaling is exact
    q = query_weights(_QUERY, index, sublinear=False, idf_variant="smooth")
    d = w[0]
    base_cos = (d @ q) / (np.linalg.norm(d) * np.linalg.norm(q))
    base_dot = d @ q
    for c in (2.0, 5.0, 10.0):
        dc = c * d
        cos_c = (dc @ q) / (np.linalg.norm(dc) * np.linalg.norm(q))
        assert abs(cos_c - base_cos) < 1e-12, f"cosine changed at c={c}"
        assert abs((dc @ q) - c * base_dot) < 1e-9, f"raw dot did not scale at c={c}"
    print("  [ok] cosine is invariant to pure magnitude scaling; raw dot scales by c")


def test_length_hijack_flip() -> None:
    """The headline claim, aligned with BM25's test_length_hijack: with the raw
    tf-idf dot product the padded transcript hijacks #1 by sheer length; cosine
    normalization surfaces the concise on-point filing."""
    index = build_inverted_index(_FINANCE_CORPUS)
    raw = vsm_rank(_QUERY, index, normalized=False)
    cos = vsm_rank(_QUERY, index, normalized=True)
    assert raw[0][0] == "transcript-pad", f"raw dot top = {raw[0][0]}"
    assert cos[0][0] == "filing-onpoint", f"cosine top = {cos[0][0]}"
    print("  [ok] length-hijack flip: raw dot -> padded transcript #1, cosine -> on-point filing #1")


def test_sublinear_unbounded_vs_bm25_bounded() -> None:
    """Proposition 3(a): sublinear tf 1+log(tf) is unbounded, while BM25's
    saturating tf-factor stays below its ceiling k1 + 1 for all tf."""
    k1 = 1.5
    big = np.array([1e1, 1e3, 1e6, 1e9])
    sublinear = 1.0 + np.log(big)
    bm25_factor = (big * (k1 + 1.0)) / (big + k1)        # b = 0; ceiling is k1 + 1
    assert sublinear[-1] > 10.0, "sublinear should grow without bound"
    assert np.all(bm25_factor < k1 + 1.0), "BM25 factor must stay below its ceiling"
    assert sublinear[-1] > bm25_factor[-1], "sublinear overtakes the bounded factor"
    print("  [ok] sublinear tf is unbounded; BM25's saturating factor is capped at k1+1")


def validate_against_sklearn() -> None:
    """Cross-check the cosine ranking order against scikit-learn's TfidfVectorizer.

    sklearn's smooth_idf adds 1 to both df and N; we therefore compare ORDER, not
    raw scores, so a known formula difference is not mistaken for a bug.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("  [skip] scikit-learn not installed (add --with scikit-learn to cross-check)")
        return
    docs = list(_FINANCE_CORPUS.values())
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z0-9]+", sublinear_tf=True, norm="l2", smooth_idf=True)
    doc_mat = vec.fit_transform(docs)
    q_mat = vec.transform([_QUERY])
    sims = cosine_similarity(q_mat, doc_mat).ravel()
    ref_order = [list(_FINANCE_CORPUS)[i] for i in np.argsort(sims)[::-1]]
    index = build_inverted_index(_FINANCE_CORPUS)
    ours_order = [d for d, _ in vsm_rank(_QUERY, index, normalized=True, sublinear=True, idf_variant="smooth")]
    assert ours_order[0] == ref_order[0], f"top doc differs:\n  ours={ours_order}\n  ref ={ref_order}"
    print(f"  [ok] cosine top doc matches scikit-learn TfidfVectorizer ({ref_order[0]})")


# --------------------------------------------------------------------------- #
# Demos (printed, non-asserting) — feed the viz panels
# --------------------------------------------------------------------------- #

def idf_table() -> None:
    """Self-information of each query term (textbook IDF, the theorem) alongside the
    smoothed weight actually used in scoring. Feeds viz Panel A (bits) and the Panel C
    tooltip (smoothed weights)."""
    index = build_inverted_index(_FINANCE_CORPUS)
    print(f"  N={index.n_docs}   query={_QUERY!r}")
    for term in dict.fromkeys(tokenize(_QUERY)):
        j = index.vocab[term]
        df = int(index.df[j])
        bits = idf_scalar(df, index.n_docs, "textbook", base=2)
        smooth = idf_scalar(df, index.n_docs, "smooth")
        print(f"    {term:<9} df={df}/{index.n_docs}   self-info={bits:.3f} bits   "
              f"smoothed-idf={smooth:.3f} nats")


def finance_demo() -> None:
    index = build_inverted_index(_FINANCE_CORPUS)
    for normalized in (False, True):
        order = vsm_rank(_QUERY, index, normalized=normalized)
        tag = "cosine " if normalized else "raw dot"
        rows = "   ".join(f"{d}:{s:.3f}" for d, s in order[:3])
        print(f"  {tag}: #1 = {order[0][0]:<16} | top3  {rows}")
    print("  -> cosine normalization demotes the length-hijacking padded transcript")


def viz_constants() -> None:
    """The exact numbers VectorSpaceLaboratory.tsx mirrors to the decimal: each
    document's full TF-IDF L2 norm (the cosine denominator, which includes the
    padding terms) under both tf-scalings, and the query-vector norm. The
    query-term frequencies and the smoothed IDF are derivable in the viz; these
    full norms are not, so they are the mirrored constants."""
    index = build_inverted_index(_FINANCE_CORPUS)
    q = query_weights(_QUERY, index, sublinear=True)
    print(f"  qnorm={np.linalg.norm(q):.4f}  (identical for both tf-scalings; query tf=1)")
    for sublinear in (True, False):
        norms = np.linalg.norm(weight_matrix(index, sublinear=sublinear), axis=1)
        tag = "sublinear" if sublinear else "raw tf   "
        cells = "  ".join(f"{d}={n:.4f}" for d, n in zip(index.doc_ids, norms))
        print(f"  norms[{tag}]: {cells}")


if __name__ == "__main__":
    print("Vector Space Model / TF-IDF verification harness")
    test_idf_is_self_information()
    test_sublinear_tf_monotone_concave()
    test_cosine_normalization_invariant()
    test_length_hijack_flip()
    test_sublinear_unbounded_vs_bm25_bounded()
    validate_against_sklearn()
    print("IDF table (self-information of each query term):")
    idf_table()
    print("Finance demo:")
    finance_demo()
    print("Viz constants (mirrored to the decimal in VectorSpaceLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
