"""Rank fusion and the geometry of rank aggregation — the reference implementation
for the formalRAG `rank-fusion-rrf` topic.

Two retrievers disagree. BM25 gives a *lexical* ranking over 10-K filing text; a
dense encoder gives a *semantic* ranking over earnings-call passages. They live on
incompatible score scales (BM25 unbounded; cosine in [-1, 1]) and rank the same
documents differently. This module fuses them and verifies the topic's claims.

The lexical leg is imported from `notebooks/bm25/bm25.py` (its BM25 scorer and
NDCG). The dense leg is a *deterministic toy embedding* — a seeded Gaussian random
projection of bag-of-words counts, L2-normalized, scored by cosine. It is a
self-contained, CPU-only stand-in for a trained bi-encoder: a length-normalized
lexical-cosine ranking that disagrees with BM25 because it carries no IDF weighting
and no term saturation. The rank-aggregation mathematics below (RRF, Borda,
Kendall-tau geometry, the footrule 2-approximation, Kemeny consensus) is identical
regardless of how the second list is produced — that independence is the point.

Every pedagogical claim the topic makes is an `assert` here:
  1. RRF is scale-invariant (depends only on ranks); CombSUM is not.
  2. Diaconis-Graham: K <= F <= 2K for Kendall-tau K and Spearman footrule F.
  3. RRF order converges to the Borda order as k -> infinity (the limit theorem).
  4. The footrule-optimal aggregate is within 2x the Kemeny optimum (Dwork et al.).
  5. NDCG(RRF) > max(NDCG(lexical), NDCG(dense)): hybrid beats either leg.

Run:  uv run --with numpy --with scipy python notebooks/rank-fusion-rrf/rank_fusion_rrf.py
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------- #
# Reuse the lexical leg from the BM25 topic (path-robust import).
# --------------------------------------------------------------------------- #
_BM25_DIR = pathlib.Path(__file__).resolve().parents[1] / "bm25"
if str(_BM25_DIR) not in sys.path:
    sys.path.insert(0, str(_BM25_DIR))

from bm25 import build_inverted_index, bm25_rank, ndcg_at_k, tokenize  # noqa: E402


# --------------------------------------------------------------------------- #
# The finance corpus — engineered so each leg MISSES one of the two prize
# disclosures, and fusion surfaces both.
#
#   "interest" is the RARE query term (df = 2 -> high BM25 IDF); "rate"/"exposure"
#   are COMMON (low IDF). The two genuinely on-point disclosures (both qrel 3) are
#   built to be near-opposites under the two scorers:
#     - `filing-onpoint`  : interest-heavy but written long -> BM25 ranks it #1 on
#       IDF, but its large L2 norm sinks its cosine, so the dense leg buries it.
#     - `transcript-rate` : a short call passage dense in the common terms, no rare
#       term -> top cosine for the dense leg, but BM25 discounts the common terms
#       and finds no rare term, so the lexical leg buries it.
#   Each leg alone strands one qrel-3 disclosure at #3 (NDCG 0.98); RRF, which rewards
#   documents ranked decently by BOTH, recovers the ideal order with both prizes on
#   top (NDCG 1.0) -> hybrid beats either leg. A bonus contrast for the rigor section:
#   `filing-hedging` is every leg's #2; the Kemeny optimum is a 6-way tie at cost 3 (one tie
#   elevates it to #1), and RRF's own order is itself one of those optima -- so RRF is *a* Kemeny
#   consensus here, differing only in which co-optimal tie it selects (test_kemeny_optimum_nonunique).
# --------------------------------------------------------------------------- #
_FILLER = ("management discussion covered operations segment guidance headcount capex "
           "logistics demand pricing momentum outlook regions inventory supply chain")
_FINANCE_CORPUS = {
    # interest x4 (RARE -> high IDF), rate x2, exposure x1, written LONG -> BM25 #1
    # on the rare term, but a large norm sinks its cosine so dense buries it (#3).
    "filing-onpoint": (
        "net interest income and the interest margin move with interest rate "
        "changes while interest sensitivity and the disclosed rate exposure are "
        "reviewed by management across the lending portfolio during this fiscal "
        "period as noted in the related supervisory commentary and appendix tables"
    ),
    # rate x3 + exposure x3 (COMMON), no interest, SHORT and dense -> top cosine for
    # the dense leg; BM25 discounts the common terms and finds no rare term.
    "transcript-rate": "we sized rate exposure and hedged the rate exposure given residual rate exposure",
    # interest x1 + rate x2 + exposure x2 -> ranked #2 by BOTH legs (the rare term
    # earns IDF for BM25; moderate query mass earns mid cosine, above the long
    # on-point filing but below the dense call), so RRF lifts this agreed disclosure.
    "filing-hedging": (
        "interest rate risk and our hedging of rate exposure manages the net "
        "exposure this filing reports"
    ),
    # COMMON term exposure x3 about FX, no interest/rate -> mid on dense, low on BM25.
    "filing-fx": (
        "foreign exchange exposure and currency exposure affect our translation exposure and reserves"
    ),
    # rate x3, no interest, short off-topic macro -> mid on BM25, low on dense (no exposure).
    "news-macro": "the central bank set its policy rate as the headline rate and the rate path shift",
    # low-relevance padded call passage: rate x1, exposure x1 buried in filler.
    "transcript-ops": _FILLER + " we briefly touched on the rate and overall exposure outlook",
}
_QUERY = "interest rate exposure"
# Relevance judgments (ground truth): both on-point disclosures are prizes (3);
# the hedging filing is solid (2); macro/FX are marginal (1); ops is noise (0).
_QRELS = {
    "filing-onpoint": 3.0,
    "transcript-rate": 3.0,
    "filing-hedging": 2.0,
    "news-macro": 1.0,
    "filing-fx": 1.0,
    "transcript-ops": 0.0,
}


# --------------------------------------------------------------------------- #
# The two retrieval legs.
# --------------------------------------------------------------------------- #

def lexical_scores(query: str, corpus: dict[str, str],
                   k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    """BM25 scores per document (the lexical leg, imported scorer)."""
    index = build_inverted_index(corpus)
    return {doc: score for doc, score in bm25_rank(query, index, k1=k1, b=b)}


def toy_dense_scores(query: str, corpus: dict[str, str],
                     dim: int = 256, seed: int = 0) -> dict[str, float]:
    """Cosine scores from a seeded random projection of bag-of-words counts.

    A deterministic, CPU-only stand-in for a dense encoder. By the Johnson-
    Lindenstrauss lemma the random projection approximately preserves cosine, so
    this is a length-normalized lexical-cosine ranking on the bounded [-1, 1]
    scale -- carrying no IDF, it disagrees with BM25 at the top.
    """
    index = build_inverted_index(corpus)
    n_terms = len(index.vocab)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((n_terms, dim)) / np.sqrt(dim)

    doc_emb = np.asarray(index.tf @ proj)               # (n_docs, dim)
    doc_emb /= np.maximum(np.linalg.norm(doc_emb, axis=1, keepdims=True), 1e-12)

    q_bow = np.zeros(n_terms)
    for term in tokenize(query):
        j = index.vocab.get(term)
        if j is not None:
            q_bow[j] += 1.0
    q_emb = q_bow @ proj
    q_emb /= max(np.linalg.norm(q_emb), 1e-12)

    sims = doc_emb @ q_emb
    return {doc: float(s) for doc, s in zip(index.doc_ids, sims)}


def order_of(scores: dict[str, float]) -> list[str]:
    """Ranking (best first) from a score dict, ties broken by document id."""
    return sorted(scores, key=lambda d: (-scores[d], d))


# --------------------------------------------------------------------------- #
# Fusion rules.
# --------------------------------------------------------------------------- #

def rrf_fuse(rankings: list[list[str]], k: float = 60.0) -> list[str]:
    """Reciprocal Rank Fusion: score(d) = sum_lists 1 / (k + rank_1based(d)).

    Depends ONLY on positions, never on the underlying scores -- the source of
    its scale invariance.
    """
    items = {d for r in rankings for d in r}
    score = {d: 0.0 for d in items}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] += 1.0 / (k + rank)
    return sorted(items, key=lambda d: (-score[d], d))


def rrf_contributions(rankings: list[list[str]], k: float = 60.0) -> dict[str, list[float]]:
    """Per-list 1/(k+r) contribution for each document (powers the viz tooltip)."""
    out: dict[str, list[float]] = {d: [] for r in rankings for d in r}
    for ranking in rankings:
        pos = {doc: rank for rank, doc in enumerate(ranking, start=1)}
        for doc in out:
            out[doc].append(1.0 / (k + pos[doc]) if doc in pos else 0.0)
    return out


def borda(rankings: list[list[str]]) -> list[str]:
    """Borda count: each list awards (n - rank_0based) points; sum and sort."""
    items = {d for r in rankings for d in r}
    n = len(items)
    score = {d: 0.0 for d in items}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):          # 0-based
            score[doc] += (n - 1 - rank)
    return sorted(items, key=lambda d: (-score[d], d))


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo
    return {d: (s - lo) / span if span > 0 else 0.0 for d, s in scores.items()}


def combsum(score_dicts: list[dict[str, float]], normalize: bool = False) -> list[str]:
    """CombSUM: sum the (optionally min-max normalized) scores across lists.

    With normalize=False this sums RAW scores, so it is dominated by whichever
    list has the larger numeric range -- the scale sensitivity RRF avoids.
    """
    dicts = [_minmax(d) for d in score_dicts] if normalize else score_dicts
    items = {d for sd in score_dicts for d in sd}
    total = {d: sum(sd.get(d, 0.0) for sd in dicts) for d in items}
    return sorted(items, key=lambda d: (-total[d], d))


# --------------------------------------------------------------------------- #
# The geometry of rankings: Kendall-tau and the Spearman footrule.
# --------------------------------------------------------------------------- #

def _positions(ranking: list[str]) -> dict[str, int]:
    return {doc: i for i, doc in enumerate(ranking)}


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> int:
    """Kendall-tau distance: the number of discordant (mis-ordered) pairs."""
    pa, pb = _positions(rank_a), _positions(rank_b)
    items = list(pa)
    discordant = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            x, y = items[i], items[j]
            if (pa[x] - pa[y]) * (pb[x] - pb[y]) < 0:
                discordant += 1
    return discordant


def spearman_footrule(rank_a: list[str], rank_b: list[str]) -> int:
    """Spearman footrule: sum of absolute position displacements."""
    pa, pb = _positions(rank_a), _positions(rank_b)
    return sum(abs(pa[d] - pb[d]) for d in pa)


# --------------------------------------------------------------------------- #
# Consensus aggregation: footrule (poly-time) and Kemeny (brute force).
# --------------------------------------------------------------------------- #

def footrule_aggregate(rankings: list[list[str]]) -> list[str]:
    """Footrule-optimal consensus via min-cost bipartite matching (Dwork et al.).

    Cost C[item, p] = sum_lists |pos_list(item) - p|; the assignment of items to
    positions minimizing total cost is the footrule-optimal aggregate, solved in
    polynomial time by the Hungarian algorithm.
    """
    items = sorted({d for r in rankings for d in r})
    n = len(items)
    cost = np.zeros((n, n))
    pos = [_positions(r) for r in rankings]
    for i, item in enumerate(items):
        for p in range(n):
            cost[i, p] = sum(abs(pr[item] - p) for pr in pos)
    rows, cols = linear_sum_assignment(cost)
    placed = sorted(zip(cols, [items[r] for r in rows]))   # (position, item)
    return [item for _, item in placed]


def kemeny_cost(consensus: list[str], rankings: list[list[str]]) -> int:
    """Total Kendall-tau distance from a consensus to every input list."""
    return sum(kendall_tau(consensus, r) for r in rankings)


def kemeny_bruteforce(rankings: list[list[str]]) -> tuple[list[str], int]:
    """Exact Kemeny consensus by exhaustive search (small n only)."""
    items = sorted({d for r in rankings for d in r})
    best, best_cost = None, None
    for perm in itertools.permutations(items):
        c = kemeny_cost(list(perm), rankings)
        if best_cost is None or c < best_cost:
            best, best_cost = list(perm), c
    return best, best_cost


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def _legs() -> tuple[dict[str, float], dict[str, float], list[str], list[str]]:
    lex = lexical_scores(_QUERY, _FINANCE_CORPUS)
    den = toy_dense_scores(_QUERY, _FINANCE_CORPUS)
    return lex, den, order_of(lex), order_of(den)


def test_scale_invariance() -> None:
    """RRF depends only on ranks, so scaling a leg's scores cannot change it;
    CombSUM on raw scores collapses onto the scaled-up leg's order."""
    lex, den, lex_order, den_order = _legs()
    rrf = rrf_fuse([lex_order, den_order])

    lex_x1000 = {d: s * 1000.0 for d, s in lex.items()}
    # RRF sees the SAME orders -> identical fused list.
    assert rrf_fuse([order_of(lex_x1000), den_order]) == rrf, "RRF must be scale-invariant"
    # CombSUM(raw) order moves when we rescale, and lands on the BM25 order.
    cs_orig = combsum([lex, den])
    cs_scaled = combsum([lex_x1000, den])
    assert cs_scaled != cs_orig, "CombSUM should be sensitive to score scale"
    assert cs_scaled == lex_order, "scaled-up BM25 should swamp cosine in CombSUM"
    print("  [ok] scale invariance: RRF stable under x1000; CombSUM collapses to BM25 order")


def test_diaconis_graham(trials: int = 2000, n: int = 9, seed: int = 7) -> None:
    """K <= F <= 2K across many random permutation pairs (Diaconis-Graham 1977)."""
    rng = np.random.default_rng(seed)
    base = [f"d{i}" for i in range(n)]
    for _ in range(trials):
        a = list(rng.permutation(base))
        b = list(rng.permutation(base))
        K, F = kendall_tau(a, b), spearman_footrule(a, b)
        assert K <= F <= 2 * K or K == 0, f"violated K<=F<=2K: K={K}, F={F}"
    print(f"  [ok] Diaconis-Graham K <= F <= 2K over {trials} random pairs (n={n})")


def _borda_totals(rankings: list[list[str]]) -> dict[str, int]:
    items = {d for r in rankings for d in r}
    n = len(items)
    return {d: sum(n - 1 - r.index(d) for r in rankings) for d in items}


def test_rrf_to_borda(trials: int = 400, n: int = 7, n_lists: int = 3,
                      seed: int = 5, k_large: float = 1e6) -> None:
    """As k -> infinity the RRF order converges to the Borda order.

    The limit order is Borda exactly when the Borda totals are STRICT: then the
    first-order expansion 1/(k+r) ~ 1/k - r/k^2 makes RRF rank by -sum(rank), which
    is the Borda order. (When totals tie -- as in the symmetric finance instance --
    RRF's second-order convexity term refines the tie instead, which is why we test
    the theorem on strict random instances.)"""
    rng = np.random.default_rng(seed)
    base = [f"d{i}" for i in range(n)]
    tested = 0
    for _ in range(trials):
        lists = [list(rng.permutation(base)) for _ in range(n_lists)]
        totals = _borda_totals(lists)
        if len(set(totals.values())) < n:
            continue                      # skip ties: limit order is a refinement
        assert rrf_fuse(lists, k=k_large) == borda(lists), "RRF(k->inf) must equal Borda"
        tested += 1
    assert tested > 0, "no strict instances generated"
    print(f"  [ok] limit theorem: RRF(k->inf) order equals Borda on {tested} strict instances")


def test_footrule_2approx(trials: int = 200, n: int = 6, n_lists: int = 3, seed: int = 11) -> None:
    """The footrule aggregate's Kemeny cost is within 2x the Kemeny optimum."""
    rng = np.random.default_rng(seed)
    base = [f"d{i}" for i in range(n)]
    # The finance instance...
    _, _, lex_order, den_order = _legs()
    instances = [[lex_order, den_order]]
    # ...plus random multi-list instances.
    for _ in range(trials):
        instances.append([list(rng.permutation(base)) for _ in range(n_lists)])
    for lists in instances:
        f_cost = kemeny_cost(footrule_aggregate(lists), lists)
        _, opt = kemeny_bruteforce(lists)
        assert f_cost <= 2 * opt, f"footrule cost {f_cost} exceeds 2x Kemeny {opt}"
    print(f"  [ok] footrule aggregate within 2x Kemeny optimum ({len(instances)} instances)")


def test_hybrid_beats_either_leg() -> None:
    """NDCG@k of the RRF fusion exceeds both the lexical and the dense leg."""
    lex, den, lex_order, den_order = _legs()
    rrf = rrf_fuse([lex_order, den_order])
    n_lex = ndcg_at_k(lex_order, _QRELS)
    n_den = ndcg_at_k(den_order, _QRELS)
    n_rrf = ndcg_at_k(rrf, _QRELS)
    assert n_rrf > max(n_lex, n_den), f"RRF {n_rrf:.3f} did not beat max({n_lex:.3f}, {n_den:.3f})"
    print(f"  [ok] hybrid wins: NDCG RRF={n_rrf:.3f} > lexical={n_lex:.3f}, dense={n_den:.3f}")


def test_kemeny_optimum_nonunique() -> None:
    """The finance Kemeny optimum is a 6-way tie at cost 3, and RRF's own order is one of them — so
    RRF is *a* Kemeny consensus here, not a worse approximation of a unique one (the rigor-flag claim,
    made precise: the difference from the tie that elevates the unanimous-#2 is only the tie-break)."""
    _, _, lex_order, den_order = _legs()
    lists = [lex_order, den_order]
    items = sorted({d for r in lists for d in r})
    costs = [(list(p), kemeny_cost(list(p), lists)) for p in itertools.permutations(items)]
    best = min(c for _, c in costs)
    optima = [p for p, c in costs if c == best]
    rrf = rrf_fuse(lists)
    assert best == 3, f"Kemeny min cost {best} != 3"
    assert len(optima) == 6, f"expected 6 co-optimal Kemeny orderings, got {len(optima)}"
    assert rrf in optima, "RRF ordering is not itself Kemeny-optimal"
    print(f"  [ok] Kemeny optimum is a {len(optima)}-way tie at cost {best}; RRF's order is one of them")


def finance_demo() -> None:
    lex, den, lex_order, den_order = _legs()
    rrf = rrf_fuse([lex_order, den_order])
    print(f"  query = {_QUERY!r}   N = {len(_FINANCE_CORPUS)} documents")
    print(f"  {'rank':<6}{'lexical (BM25)':<20}{'dense (cosine)':<20}{'fused (RRF)':<20}")
    for i in range(len(rrf)):
        print(f"  {i + 1:<6}{lex_order[i]:<20}{den_order[i]:<20}{rrf[i]:<20}")
    print(f"  -> each leg buries a prize: 'transcript-rate' is lexical "
          f"#{lex_order.index('transcript-rate') + 1}, 'filing-onpoint' is dense "
          f"#{den_order.index('filing-onpoint') + 1}; RRF interleaves both leaders.")
    kem, kem_cost = kemeny_bruteforce([lex_order, den_order])
    print(f"  Kemeny consensus  = {kem}  (total Kendall-tau {kem_cost})")
    print(f"  footrule aggregate = {footrule_aggregate([lex_order, den_order])}")
    items = sorted({d for r in (lex_order, den_order) for d in r})
    n_optima = sum(kemeny_cost(list(p), [lex_order, den_order]) == kem_cost
                   for p in itertools.permutations(items))
    print(f"  -> the Kemeny optimum is a {n_optima}-way tie at cost {kem_cost}; one tie puts "
          f"{kem[0]!r} (every leg's #2) first while RRF breaks it differently, yet RRF's own order "
          f"is itself Kemeny-optimal -- RRF is *a* consensus, not a worse approximation of one.")
    print(f"  Kendall-tau(lexical, dense) = {kendall_tau(lex_order, den_order)} discordant pairs")


if __name__ == "__main__":
    print("Rank-fusion verification harness")
    test_scale_invariance()
    test_diaconis_graham()
    test_rrf_to_borda()
    test_footrule_2approx()
    test_hybrid_beats_either_leg()
    test_kemeny_optimum_nonunique()
    print("Finance demo:")
    finance_demo()
    print("All checks passed.")
