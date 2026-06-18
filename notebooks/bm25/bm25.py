"""BM25 from scratch — the reference implementation for the formalRAG pilot topic.

Every mathematical claim in the topic has an executable check here. Core scoring
uses only numpy + scipy.sparse; the verification harness asserts the limit
theorems (k1 -> 0 recovers the binary BIM, k1 -> inf recovers length-normalized
raw tf, b in {0, 1} toggles length normalization), checks monotonicity and the
saturation ceiling, cross-validates against rank_bm25 under a matching IDF
variant, and implements NDCG@10 from scratch with a (k1, b) grid sweep.

Run:  uv run --with numpy --with scipy notebooks/bm25/bm25.py
Optional cross-check:  add `--with rank-bm25`
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse

# --------------------------------------------------------------------------- #
# Tokenization + inverted index
# --------------------------------------------------------------------------- #

# Hook point for a finance-aware tokenizer that preserves tickers, CUSIPs, and
# numerics; the default is whitespace + lowercase.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class InvertedIndex:
    vocab: dict[str, int]            # term -> column index
    tf: sparse.csr_matrix           # (n_docs, n_terms) raw term frequencies
    doc_len: np.ndarray             # (n_docs,) document lengths
    avgdl: float
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

    rows, cols, vals = [], [], []
    doc_len = np.zeros(len(doc_ids))
    for i, toks in enumerate(tokenized):
        doc_len[i] = len(toks)
        for term, count in Counter(toks).items():
            rows.append(i)
            cols.append(vocab[term])
            vals.append(count)
    tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(doc_ids), len(vocab)))
    df = np.asarray((tf > 0).sum(axis=0)).ravel()
    return InvertedIndex(vocab, tf, doc_len, float(doc_len.mean()), df, len(doc_ids), doc_ids)


# --------------------------------------------------------------------------- #
# IDF — three variants, side by side
#   bm25     : RSJ weight under a Jeffreys prior, log((N-df+0.5)/(df+0.5) + 1)
#              (non-negative, the Lucene / BM25+ form; matches the viz and topic)
#   textbook : log(N/df)
#   okapi    : classic Robertson IDF log((N-df+0.5)/(df+0.5)) with negative
#              values floored at epsilon * mean_idf (matches rank_bm25 BM25Okapi)
# --------------------------------------------------------------------------- #

def idf_scalar(df: float, n_docs: int, variant: str = "bm25") -> float:
    """Single-term IDF for the topic's RSJ-vs-textbook exposition."""
    if variant == "bm25":
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
    if variant == "textbook":
        return math.log(n_docs / df) if df > 0 else 0.0
    raise ValueError(variant)


def idf_vector(index: InvertedIndex, variant: str = "bm25", epsilon: float = 0.25) -> np.ndarray:
    df, n = index.df.astype(float), index.n_docs
    if variant == "bm25":
        return np.log((n - df + 0.5) / (df + 0.5) + 1.0)
    if variant == "textbook":
        return np.where(df > 0, np.log(n / np.maximum(df, 1.0)), 0.0)
    if variant == "okapi":
        v = np.log((n - df + 0.5) / (df + 0.5))
        floor = epsilon * v.mean()
        return np.where(v < 0, floor, v)
    raise ValueError(variant)


def tf_factor(tf: float, k1: float, b: float, dl: float, avgdl: float) -> float:
    """The BM25 term-frequency factor — the saturating, length-normalized core."""
    return (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def bm25_contributions(query: str, doc_i: int, index: InvertedIndex, idf_vec: np.ndarray,
                       k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    """Per-term score contributions for a single document (powers the tooltip)."""
    out: dict[str, float] = {}
    dl = index.doc_len[doc_i]
    for term in tokenize(query):
        j = index.vocab.get(term)
        if j is None:
            continue
        tf = index.tf[doc_i, j]
        if tf == 0:
            continue
        out[term] = float(idf_vec[j]) * tf_factor(float(tf), k1, b, dl, index.avgdl)
    return out


def bm25_rank(query: str, index: InvertedIndex, k1: float = 1.5, b: float = 0.75,
              idf_variant: str = "bm25") -> list[tuple[str, float]]:
    idf_vec = idf_vector(index, idf_variant)
    scored = [
        (index.doc_ids[i], sum(bm25_contributions(query, i, index, idf_vec, k1, b).values()))
        for i in range(index.n_docs)
    ]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# NDCG@k from scratch
# --------------------------------------------------------------------------- #

def ndcg_at_k(ranking: list[str], qrels: dict[str, float], k: int = 10) -> float:
    def dcg(rels: list[float]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    gains = [qrels.get(doc, 0.0) for doc in ranking[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# Verification harness — the limit theorems, made executable
# --------------------------------------------------------------------------- #

def test_limits() -> None:
    dl, avgdl = 120.0, 80.0
    for tf in (1.0, 3.0, 9.0):
        # (2) k1 -> 0 recovers the binary BIM: factor -> 1 for any tf > 0.
        assert abs(tf_factor(tf, 1e-9, 0.75, dl, avgdl) - 1.0) < 1e-6
        for b in (0.0, 0.5, 1.0):
            B = 1.0 - b + b * dl / avgdl
            # (3) k1 -> inf recovers length-normalized raw tf: factor -> tf / B.
            assert abs(tf_factor(tf, 1e9, b, dl, avgdl) - tf / B) < 1e-3
    # (1) saturation ceiling: as tf -> inf, factor -> k1 + 1.
    assert abs(tf_factor(1e9, 1.5, 0.0, dl, avgdl) - 2.5) < 1e-3
    # (4) b = 0 removes length dependence; b = 1 applies it fully.
    assert tf_factor(5, 1.5, 0.0, 30, avgdl) == tf_factor(5, 1.5, 0.0, 300, avgdl)
    assert tf_factor(5, 1.5, 1.0, 30, avgdl) != tf_factor(5, 1.5, 1.0, 300, avgdl)
    print("  [ok] limit theorems (k1->0 binary, k1->inf raw tf, ceiling k1+1, b toggle)")


def test_monotonicity() -> None:
    prev = -1.0
    for tf in np.arange(0.0, 30.0, 0.5):
        f = tf_factor(float(tf), 1.5, 0.75, 100.0, 80.0)
        assert f >= prev - 1e-12, f"non-monotone at tf={tf}"
        prev = f
    print("  [ok] tf-factor is monotonically increasing in tf")


def validate_against_rank_bm25(corpus: dict[str, str], query: str) -> None:
    """Cross-check the full ranking against rank_bm25's BM25Okapi.

    We match BM25Okapi's IDF variant ('okapi', with epsilon-flooring of negative
    IDFs) so any residual difference would be a genuine scoring bug, not the
    documented IDF-variant difference between Robertson-flooored and Lucene forms.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  [skip] rank_bm25 not installed (add --with rank-bm25 to cross-check)")
        return
    tokenized = [tokenize(corpus[d]) for d in corpus]
    ref = BM25Okapi(tokenized, k1=1.5, b=0.75)
    ref_scores = ref.get_scores(tokenize(query))
    ref_order = [list(corpus)[i] for i in np.argsort(ref_scores)[::-1]]
    index = build_inverted_index(corpus)
    ours_order = [doc for doc, _ in bm25_rank(query, index, k1=1.5, b=0.75, idf_variant="okapi")]
    assert ours_order == ref_order, f"ranking differs:\n  ours={ours_order}\n  ref ={ref_order}"
    print("  [ok] top-k ranking matches rank_bm25 BM25Okapi (matched 'okapi' IDF)")


# --------------------------------------------------------------------------- #
# Finance demo + grid sweep (uses the non-negative 'bm25' IDF, matching the viz)
# --------------------------------------------------------------------------- #

# The on-point filing is short and dense in all three query terms; the padded
# transcript repeats all three terms only slightly more in total but buries them
# in a very long, low-signal document. With no length normalization (b=0) the
# transcript's higher raw counts win; with b=0.75 its length is penalized and the
# concise filing surfaces. Counts are injected exactly so the flip is robust.
# (filler avoids the query terms interest / rate / exposure.)
_FILLER = ("the quarter operations regions headcount capex marketing supply chain inventory "
           "guidance segment logistics demand pricing momentum ") * 15  # ~240 query-free tokens
_FINANCE_CORPUS = {
    # interest:2, rate:2, exposure:1 — short and dense
    "filing-onpoint": "net interest margin is sensitive to interest rate moves and our rate exposure is disclosed",
    # interest:3, rate:4, exposure:2 — only slightly more, but ~250 tokens long
    "transcript-pad": _FILLER + " interest interest interest rate rate rate rate exposure exposure",
    "filing-fx": "foreign exchange exposure and currency rate movements affect our exposure to translation",
    "news-macro": "the central bank raised the policy rate again as interest rate decisions weigh on markets",
    "filing-boiler": "this filing contains forward looking statements regarding interest rate and other risk exposure factors",
    "transcript-short": "quick update on interest and the rate outlook for the quarter",
}
_QUERY = "interest rate exposure"
_QRELS = {"filing-onpoint": 3.0, "filing-fx": 2.0, "filing-boiler": 1.0,
          "news-macro": 1.0, "transcript-short": 1.0, "transcript-pad": 0.0}


def test_length_hijack() -> None:
    """The pedagogical claim, as a test: at b=0 the padded transcript hijacks #1;
    at b=0.75 length normalization surfaces the concise on-point filing."""
    index = build_inverted_index(_FINANCE_CORPUS)
    assert bm25_rank(_QUERY, index, k1=1.5, b=0.0)[0][0] == "transcript-pad"
    assert bm25_rank(_QUERY, index, k1=1.5, b=0.75)[0][0] == "filing-onpoint"
    print("  [ok] length-hijack flip: b=0 -> padded transcript #1, b=0.75 -> on-point filing #1")


def finance_demo() -> None:
    index = build_inverted_index(_FINANCE_CORPUS)
    print(f"  query={_QUERY!r}  N={index.n_docs}  avgdl={index.avgdl:.1f}")
    for b in (0.0, 0.75):
        order = bm25_rank(_QUERY, index, k1=1.5, b=b)
        top = order[0]
        ndcg = ndcg_at_k([d for d, _ in order], _QRELS)
        print(f"  b={b:>4}: #1 = {top[0]:<16} (score {top[1]:.3f})  ndcg@10={ndcg:.3f}")
    print("  -> raising b demotes the length-hijacking padded transcript")


def grid_sweep() -> None:
    index = build_inverted_index(_FINANCE_CORPUS)
    best = (None, -1.0)
    for k1 in (0.5, 1.0, 1.5, 2.0):
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            order = [d for d, _ in bm25_rank(_QUERY, index, k1=k1, b=b)]
            score = ndcg_at_k(order, _QRELS)
            if score > best[1]:
                best = ((k1, b), score)
    print(f"  best (k1, b) = {best[0]} with NDCG@10 = {best[1]:.3f}")


if __name__ == "__main__":
    print("BM25 verification harness")
    test_limits()
    test_monotonicity()
    test_length_hijack()
    validate_against_rank_bm25(_FINANCE_CORPUS, _QUERY)
    print("Finance demo:")
    finance_demo()
    print("Grid sweep:")
    grid_sweep()
    print("All checks passed.")
