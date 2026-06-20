"""The Inverted Index and Safe Dynamic Pruning (WAND, BlockMax-WAND) — reference.

This module owns every number the topic depends on. Core retrieval uses only the
standard library + numpy; the verification harness asserts the topic's claims:

  * WAND and BlockMax-WAND return the EXACT top-k — the same documents and scores
    an exhaustive document-at-a-time (DAAT) scan returns (the safety theorem),
    verified on the worked corpus and on random strict instances;
  * the threshold (the running k-th-best score) is monotonically non-decreasing,
    which is what makes the per-term upper-bound skip valid;
  * dynamic pruning fully scores strictly fewer documents than the exhaustive scan
    (WAND ≤ exhaustive, BlockMax-WAND ≤ WAND), because the block maxima are tighter
    local upper bounds than the global per-term maxima;
  * pruning is EXACT but offers no asymptotic guarantee — a flat-score adversarial
    query prunes nothing and every method degrades to the exhaustive scan
    (rigorFlag), so the win is a data-dependent constant factor, not a better
    complexity class.

BM25 is the scoring function, replicated verbatim from notebooks/bm25/bm25.py
(idf_variant "bm25", k1=1.5, b=0.75) so the documents the inverted index is
computed over score identically to the published BM25 topic. The shared finance
corpus from that topic is used for a cross-topic safety check; a slightly larger
worked corpus drives the visualizer, where the pruning is visible.

Run:  uv run --with numpy python notebooks/inverted-index-dynamic-pruning/inverted_index_dynamic_pruning.py
"""
from __future__ import annotations

import bisect
import heapq
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Tokenization + inverted index with postings lists (sorted by document id)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class InvertedIndex:
    vocab: dict[str, int]
    postings: dict[str, list[tuple[int, int]]]   # term -> sorted [(doc_id, tf)]
    doc_len: np.ndarray
    avgdl: float
    df: dict[str, int]
    n_docs: int
    doc_ids: list[str]


def build_inverted_index(corpus: dict[str, str]) -> InvertedIndex:
    doc_ids = list(corpus)
    tokenized = [tokenize(corpus[d]) for d in doc_ids]
    vocab: dict[str, int] = {}
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_len = np.zeros(len(doc_ids))
    for i, toks in enumerate(tokenized):
        doc_len[i] = len(toks)
        for term, count in Counter(toks).items():
            vocab.setdefault(term, len(vocab))
            postings.setdefault(term, []).append((i, count))
    for term in postings:
        postings[term].sort()                       # ascending document id
    df = {t: len(p) for t, p in postings.items()}
    return InvertedIndex(vocab, postings, doc_len, float(doc_len.mean()), df,
                         len(doc_ids), doc_ids)


# --------------------------------------------------------------------------- #
# BM25 scoring — replicated verbatim from notebooks/bm25/bm25.py
# --------------------------------------------------------------------------- #

K1, B = 1.5, 0.75


def idf(term: str, index: InvertedIndex) -> float:
    df = index.df.get(term, 0)
    return math.log((index.n_docs - df + 0.5) / (df + 0.5) + 1.0)


def tf_factor(tf: float, dl: float, avgdl: float) -> float:
    if avgdl == 0.0:                                  # empty corpus guard
        return 0.0
    return (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avgdl))


def contribution(term: str, doc_id: int, tf: int, index: InvertedIndex) -> float:
    """The BM25 score contribution of one query term in one document."""
    return idf(term, index) * tf_factor(float(tf), index.doc_len[doc_id], index.avgdl)


def term_upper_bound(term: str, index: InvertedIndex) -> float:
    """The global per-term max score (UB_t): the largest contribution this term
    makes in any document. A valid upper bound because every contribution <= it."""
    p = index.postings.get(term, [])
    return max((contribution(term, d, tf, index) for d, tf in p), default=0.0)


# --------------------------------------------------------------------------- #
# Exhaustive DAAT scan (the ground truth)
# --------------------------------------------------------------------------- #

def exhaustive_topk(query: str, index: InvertedIndex, k: int) -> tuple[list[tuple[int, float]], int]:
    """Score every document containing at least one query term; return the top-k
    (doc_id, score) by score and the number of documents fully scored."""
    qterms = [t for t in dict.fromkeys(tokenize(query)) if t in index.postings]
    acc: dict[int, float] = {}
    for t in qterms:
        for d, tf in index.postings[t]:
            acc[d] = acc.get(d, 0.0) + contribution(t, d, tf, index)
    ranked = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k], len(acc)


# --------------------------------------------------------------------------- #
# WAND — document-at-a-time with per-term upper-bound pivoting
# --------------------------------------------------------------------------- #

class _Heap:
    """Tiny fixed-capacity min-by-score store for the current top-k."""
    def __init__(self, k: int):
        self.k = k
        self.items: list[tuple[float, int]] = []   # (score, doc_id)

    def threshold(self) -> float:
        # heapq keeps the smallest at items[0]; that is the k-th best once full
        return self.items[0][0] if len(self.items) >= self.k else -math.inf

    def offer(self, score: float, doc_id: int) -> None:
        if len(self.items) < self.k:
            heapq.heappush(self.items, (score, doc_id))
        elif score > self.threshold():
            heapq.heappushpop(self.items, (score, doc_id))

    def topk(self) -> list[tuple[int, float]]:
        return [(d, s) for s, d in sorted(self.items, key=lambda kv: (-kv[0], kv[1]))]


def wand_topk(query: str, index: InvertedIndex, k: int) -> tuple[list[tuple[int, float]], int, bool]:
    """Safe WAND. Returns (top-k, documents fully scored, threshold-monotone flag)."""
    qterms = [t for t in dict.fromkeys(tokenize(query)) if t in index.postings]
    ub = {t: term_upper_bound(t, index) for t in qterms}
    cur = {t: 0 for t in qterms}                    # cursor index into each postings list
    heap = _Heap(k)
    full_evals = 0
    monotone = True
    prev_theta = -math.inf

    def cur_doc(t: str) -> int:
        return index.postings[t][cur[t]][0] if cur[t] < len(index.postings[t]) else math.inf

    while True:
        active = sorted((t for t in qterms if cur[t] < len(index.postings[t])), key=cur_doc)
        if not active:
            break
        theta = heap.threshold()
        if theta < prev_theta:
            monotone = False
        prev_theta = theta
        # pivot: smallest prefix whose cumulative UB can reach the threshold
        cum, pivot = 0.0, None
        for t in active:
            cum += ub[t]
            if cum >= theta:
                pivot = t
                break
        if pivot is None:
            break                                   # no remaining document can enter the top-k
        pivot_doc = cur_doc(pivot)
        if cur_doc(active[0]) < pivot_doc:
            # the smallest-id cursor lags the pivot -> skip it forward (binary search)
            t = active[0]
            cur[t] = bisect.bisect_left(index.postings[t], pivot_doc, lo=cur[t], key=lambda x: x[0])
        else:
            # active[0] == pivot_doc: every term aligned here contributes -> fully score it
            present = [t for t in active if cur_doc(t) == pivot_doc]
            score = sum(contribution(t, pivot_doc, index.postings[t][cur[t]][1], index) for t in present)
            for t in present:
                cur[t] += 1
            full_evals += 1
            heap.offer(score, pivot_doc)
    return heap.topk(), full_evals, monotone


# --------------------------------------------------------------------------- #
# BlockMax-WAND — block maxima give tighter local upper bounds
# --------------------------------------------------------------------------- #

def _blocks(term: str, index: InvertedIndex, block_size: int) -> list[tuple[int, float]]:
    """Partition a postings list into fixed-size blocks; return per block its
    (last_doc_id, max_contribution)."""
    p = index.postings[term]
    out = []
    for start in range(0, len(p), block_size):
        chunk = p[start:start + block_size]
        last_doc = chunk[-1][0]
        max_c = max(contribution(term, d, tf, index) for d, tf in chunk)
        out.append((last_doc, max_c))
    return out


def bmw_topk(query: str, index: InvertedIndex, k: int,
             block_size: int = 2) -> tuple[list[tuple[int, float]], int]:
    """BlockMax-WAND. Identical pivoting to WAND, but before fully scoring the
    pivot it refines the upper bound with block maxima; if the refined bound
    cannot reach the threshold, the expensive full scoring is skipped. Returns
    (top-k, documents fully scored)."""
    qterms = [t for t in dict.fromkeys(tokenize(query)) if t in index.postings]
    ub = {t: term_upper_bound(t, index) for t in qterms}
    blocks = {t: _blocks(t, index, block_size) for t in qterms}
    cur = {t: 0 for t in qterms}
    heap = _Heap(k)
    full_evals = 0

    def cur_doc(t: str) -> int:
        return index.postings[t][cur[t]][0] if cur[t] < len(index.postings[t]) else math.inf

    def block_ub(t: str, doc: int) -> float:
        """Max contribution of t's first block whose last_doc >= doc (the block
        that could contain doc); 0 if doc is past all of t's postings. Binary
        search on the blocks' last-doc keys."""
        blks = blocks[t]
        i = bisect.bisect_left(blks, doc, key=lambda blk: blk[0])
        return blks[i][1] if i < len(blks) else 0.0

    while True:
        active = sorted((t for t in qterms if cur[t] < len(index.postings[t])), key=cur_doc)
        if not active:
            break
        theta = heap.threshold()
        cum, pivot, pivot_idx = 0.0, None, 0
        for i, t in enumerate(active):
            cum += ub[t]
            if cum >= theta:
                pivot, pivot_idx = t, i
                break
        if pivot is None:
            break
        pivot_doc = cur_doc(pivot)
        if cur_doc(active[0]) < pivot_doc:
            # the smallest-id cursor lags the pivot -> skip it forward
            t = active[0]
            cur[t] = bisect.bisect_left(index.postings[t], pivot_doc, lo=cur[t], key=lambda x: x[0])
            continue
        # active[0] == pivot_doc: refine the upper bound with block maxima over the
        # terms aligned here; if it cannot reach the threshold, skip the full scoring.
        present = [t for t in active if cur_doc(t) == pivot_doc]
        refined = sum(block_ub(t, pivot_doc) for t in present)
        if refined < theta:
            for t in present:
                cur[t] += 1
            continue
        score = sum(contribution(t, pivot_doc, index.postings[t][cur[t]][1], index) for t in present)
        for t in present:
            cur[t] += 1
        full_evals += 1
        heap.offer(score, pivot_doc)
    return heap.topk(), full_evals


# --------------------------------------------------------------------------- #
# Corpora
# --------------------------------------------------------------------------- #

# Shared finance corpus — VERBATIM from notebooks/bm25/bm25.py (cross-topic check)
_FILLER = ("the quarter operations regions headcount capex marketing supply chain inventory "
           "guidance segment logistics demand pricing momentum ") * 15
_FINANCE_CORPUS = {
    "filing-onpoint": "net interest margin is sensitive to interest rate moves and our rate exposure is disclosed",
    "transcript-pad": _FILLER + " interest interest interest rate rate rate rate exposure exposure",
    "filing-fx": "foreign exchange exposure and currency rate movements affect our exposure to translation",
    "news-macro": "the central bank raised the policy rate again as interest rate decisions weigh on markets",
    "filing-boiler": "this filing contains forward looking statements regarding interest rate and other risk exposure factors",
    "transcript-short": "quick update on interest and the rate outlook for the quarter",
}

# Worked corpus for the visualizer — ten short finance snippets engineered so the
# query-term postings overlap unevenly and dynamic pruning visibly skips work.
_WORKED_CORPUS = {
    "d0-margin": "interest rate exposure interest rate exposure interest rate",
    "d1-hedge": "interest rate swap hedges our interest rate exposure materially",
    "d2-fx": "foreign exchange exposure and currency translation exposure only",
    "d3-macro": "the policy rate decision and the rate outlook for markets",
    "d4-credit": "credit risk and default exposure dominate this segment exposure",
    "d5-liquidity": "liquidity and funding with some interest rate sensitivity noted",
    "d6-capital": "capital ratios and buffers under the proposed rate framework",
    "d7-guidance": "full year guidance unchanged with modest rate assumptions only",
    "d8-tax": "effective tax rate guidance and one time items this quarter",
    "d9-boiler": "forward looking statements regarding interest rate and exposure risk",
}
_QUERY = "interest rate exposure"


def _synthetic_corpus(n_docs: int, seed: int) -> dict[str, str]:
    """A larger deterministic corpus with a skewed query-term distribution, so the
    pruning gain is dramatic. Most documents touch the query lightly; a few are
    dense. Drawn from one rng stream for reproducibility."""
    rng = np.random.default_rng(seed)
    fillers = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"]
    qterms = ["interest", "rate", "exposure"]
    corpus = {}
    for i in range(n_docs):
        length = int(rng.integers(8, 30))
        toks = list(rng.choice(fillers, size=length))
        # a minority of documents are query-dense; the rest touch it rarely
        if rng.random() < 0.15:
            for t in qterms:
                toks += [t] * int(rng.integers(2, 6))
        elif rng.random() < 0.5:
            toks += [str(rng.choice(qterms))] * int(rng.integers(1, 3))
        corpus[f"doc{i}"] = " ".join(toks)
    return corpus


# --------------------------------------------------------------------------- #
# Verification harness — the topic's claims, made executable
# --------------------------------------------------------------------------- #

def _scores_equal(a: list[tuple[int, float]], b: list[tuple[int, float]]) -> bool:
    """Two top-k results agree if their score multisets match (ties at the k-th
    boundary may pick different doc ids, but a SAFE method returns the same scores)."""
    sa = sorted(round(s, 9) for _, s in a)
    sb = sorted(round(s, 9) for _, s in b)
    return sa == sb


def test_bm25_matches_published_ranking() -> None:
    """Consistency: the exhaustive DAAT scan over the shared finance corpus puts
    the concise on-point filing first at k1=1.5, b=0.75 — the same winner the
    published BM25 topic reports, confirming the scoring is identical."""
    index = build_inverted_index(_FINANCE_CORPUS)
    top, _ = exhaustive_topk(_QUERY, index, k=6)
    assert index.doc_ids[top[0][0]] == "filing-onpoint", index.doc_ids[top[0][0]]
    print("  [ok] exhaustive BM25 ranking matches the published topic (filing-onpoint #1)")


def test_wand_bmw_safety() -> None:
    """Safety theorem: WAND and BlockMax-WAND return the exact top-k of the
    exhaustive scan, for every k, on the worked and shared corpora."""
    for name, corpus in (("worked", _WORKED_CORPUS), ("finance", _FINANCE_CORPUS)):
        index = build_inverted_index(corpus)
        for k in range(1, index.n_docs + 1):
            exact, _ = exhaustive_topk(_QUERY, index, k)
            w, _, _ = wand_topk(_QUERY, index, k)
            bmw, _ = bmw_topk(_QUERY, index, k, block_size=2)
            assert _scores_equal(exact, w), f"WAND unsafe ({name}, k={k})"
            assert _scores_equal(exact, bmw), f"BMW unsafe ({name}, k={k})"
    print("  [ok] WAND and BlockMax-WAND return the exact top-k (safety) on both corpora, all k")


def test_safety_random_strict() -> None:
    """Safety on random strict instances: across many synthetic corpora and ks,
    pruned results match the exhaustive scan. Strictness (distinct scores) is not
    required because _scores_equal compares the score multiset, which a safe
    method must reproduce even under ties."""
    rng = np.random.default_rng(20260620)
    checked = 0
    while checked < 150:
        corpus = _synthetic_corpus(int(rng.integers(20, 60)), seed=int(rng.integers(1, 1_000_000)))
        index = build_inverted_index(corpus)
        if not any(t in index.postings for t in tokenize(_QUERY)):
            continue
        k = int(rng.integers(1, 12))
        exact, _ = exhaustive_topk(_QUERY, index, k)
        w, _, mono = wand_topk(_QUERY, index, k)
        bmw, _ = bmw_topk(_QUERY, index, k, block_size=int(rng.integers(2, 5)))
        assert _scores_equal(exact, w), "WAND unsafe on random instance"
        assert _scores_equal(exact, bmw), "BMW unsafe on random instance"
        assert mono, "threshold was not monotone non-decreasing"
        checked += 1
    print(f"  [ok] WAND/BMW safe + threshold monotone on {checked} random instances")


def test_pruning_reduces_full_evals() -> None:
    """WAND fully scores no more documents than the exhaustive scan, and
    BlockMax-WAND no more than WAND, with a strict win on a large skewed corpus."""
    index = build_inverted_index(_synthetic_corpus(2000, seed=7))
    k = 10
    _, ex_evals = exhaustive_topk(_QUERY, index, k)
    _, w_evals, _ = wand_topk(_QUERY, index, k)
    _, bmw_evals = bmw_topk(_QUERY, index, k, block_size=4)
    assert w_evals < ex_evals, (w_evals, ex_evals)
    assert bmw_evals <= w_evals, (bmw_evals, w_evals)
    print(f"  [ok] full document scorings — exhaustive {ex_evals}, WAND {w_evals}, "
          f"BlockMax-WAND {bmw_evals}  (k={k}, N={index.n_docs})")


def test_adversarial_flat_scores_prune_nothing() -> None:
    """rigorFlag made executable: when every document scores identically (a flat
    single-term query repeated), the threshold never lets a skip fire and WAND
    degrades to the exhaustive scan — pruning has no asymptotic guarantee."""
    flat = {f"doc{i}": "rate rate rate" for i in range(40)}
    index = build_inverted_index(flat)
    _, ex_evals = exhaustive_topk("rate", index, k=5)
    _, w_evals, _ = wand_topk("rate", index, k=5)
    assert w_evals == ex_evals, (w_evals, ex_evals)
    print(f"  [ok] adversarial flat-score query: WAND evals {w_evals} == exhaustive {ex_evals} "
          "(no pruning gain)")


def test_edge_cases() -> None:
    """k larger than the corpus returns all matched documents; a single-term query
    still returns the safe top-k."""
    index = build_inverted_index(_WORKED_CORPUS)
    exact, _ = exhaustive_topk(_QUERY, index, k=999)
    w, _, _ = wand_topk(_QUERY, index, k=999)
    assert _scores_equal(exact, w)
    single_exact, _ = exhaustive_topk("interest", index, k=3)
    single_w, _, _ = wand_topk("interest", index, k=3)
    assert _scores_equal(single_exact, single_w)
    print("  [ok] edge cases: k > N and single-term query remain safe")


# --------------------------------------------------------------------------- #
# Demos (printed, non-asserting) — feed the viz panels
# --------------------------------------------------------------------------- #

def pruning_demo() -> None:
    for n in (200, 1000, 5000):
        index = build_inverted_index(_synthetic_corpus(n, seed=7))
        _, ex = exhaustive_topk(_QUERY, index, 10)
        _, w, _ = wand_topk(_QUERY, index, 10)
        _, bmw = bmw_topk(_QUERY, index, 10, block_size=4)
        print(f"  N={n:>5}  exhaustive={ex:>5}  WAND={w:>5}  BlockMax-WAND={bmw:>5}  "
              f"(scored {100*bmw/ex:.0f}% of exhaustive)")


def worked_ranking() -> None:
    index = build_inverted_index(_WORKED_CORPUS)
    top, _ = exhaustive_topk(_QUERY, index, k=3)
    print(f"  worked corpus top-3 for {_QUERY!r}:")
    for d, s in top:
        print(f"    {index.doc_ids[d]:<12} score={s:.3f}")


def viz_constants() -> None:
    """The exact numbers InvertedIndexPruningVisualizer.tsx mirrors to the decimal:
    for the worked corpus, each query term's postings (doc id, tf, contribution)
    and its global upper bound UB_t, plus avgdl. The viz reconstructs the BM25
    contributions from these and replays WAND/BMW, so the numbers are shared."""
    index = build_inverted_index(_WORKED_CORPUS)
    print(f"  avgdl={index.avgdl:.4f}  N={index.n_docs}")
    for t in dict.fromkeys(tokenize(_QUERY)):
        print(f"  term {t!r}  df={index.df[t]}  UB={term_upper_bound(t, index):.4f}  idf={idf(t, index):.4f}")
        cells = "  ".join(
            f"{index.doc_ids[d]}(tf{tf},{contribution(t, d, tf, index):.3f})"
            for d, tf in index.postings[t]
        )
        print(f"    postings: {cells}")
    print(f"  doc_len: " + "  ".join(f"{index.doc_ids[i]}={int(L)}" for i, L in enumerate(index.doc_len)))


if __name__ == "__main__":
    print("Inverted Index / WAND / BlockMax-WAND verification harness")
    test_bm25_matches_published_ranking()
    test_wand_bmw_safety()
    test_safety_random_strict()
    test_pruning_reduces_full_evals()
    test_adversarial_flat_scores_prune_nothing()
    test_edge_cases()
    print("Pruning demo (documents fully scored vs corpus size):")
    pruning_demo()
    print("Worked-corpus ranking:")
    worked_ranking()
    print("Viz constants (mirrored to the decimal in InvertedIndexPruningVisualizer.tsx):")
    viz_constants()
    print("All checks passed.")
