"""The Probability Ranking Principle from scratch — reference implementation.

This module owns every number the topic depends on. The Probability Ranking
Principle says that ranking documents by decreasing probability of relevance
P(R=1 | d, q) is optimal under an additive cost model, simultaneously at every
cutoff. The verification harness makes the claims executable:

  - the theorem itself, brute-forced over ALL permutations of a small corpus
    (sorting by P(R) maximizes expected relevant@k and minimizes expected cost
    at every k);
  - the adjacent-swap (exchange) lemma the proof turns on;
  - the precision@k / recall@k special case;
  - the collapse of any linear cost model to the same sort;
  - rank-invariance of P(R), its odds, and its log-odds (the handoff to BM25);
  - a finance-flavored worked example where the PRP order beats a plausible
    length/recency-biased order;
  - an honest counterexample: with NON-additive (redundancy-discounted) utility,
    the PRP is no longer optimal — the additivity assumption is load-bearing.

Run:  uv run --with numpy python notebooks/probability-ranking-principle/probability_ranking_principle.py
"""
from __future__ import annotations

import math
from itertools import permutations

import numpy as np

# --------------------------------------------------------------------------- #
# Core quantities
# --------------------------------------------------------------------------- #

def expected_relevant_at_k(order: tuple[int, ...], p: np.ndarray, k: int) -> float:
    """Expected number of relevant documents in the top k of an ordering.
    By linearity of expectation this is the prefix sum of relevance
    probabilities — no independence assumption is used."""
    return float(sum(p[d] for d in order[:k]))


def expected_cost(order: tuple[int, ...], p: np.ndarray, k: int, c_fp: float, c_fn: float) -> float:
    """Additive expected cost of retrieving the top k: c_fp per non-relevant
    document retrieved plus c_fn per relevant document missed below the cut."""
    retrieved = order[:k]
    missed = order[k:]
    exp_non_relevant_retrieved = sum(1.0 - p[d] for d in retrieved)
    exp_relevant_missed = sum(p[d] for d in missed)
    return float(c_fp * exp_non_relevant_retrieved + c_fn * exp_relevant_missed)


def prp_order(p: np.ndarray) -> tuple[int, ...]:
    """The PRP ordering: documents by strictly decreasing P(R). Ties broken by
    index for determinism."""
    return tuple(sorted(range(len(p)), key=lambda i: (-p[i], i)))


def inversions(order: tuple[int, ...], p: np.ndarray) -> int:
    """Number of out-of-order pairs (i before j with p[i] < p[j]) — the integer
    the exchange argument drives monotonically to zero."""
    return sum(
        1
        for a in range(len(order))
        for b in range(a + 1, len(order))
        if p[order[a]] < p[order[b]]
    )


def one_bubble_step(order: tuple[int, ...], p: np.ndarray) -> tuple[int, ...]:
    """Swap the first out-of-order adjacent pair (left-to-right). One step of the
    bubble-sort completion of the exchange argument."""
    lst = list(order)
    for j in range(len(lst) - 1):
        if p[lst[j]] < p[lst[j + 1]]:
            lst[j], lst[j + 1] = lst[j + 1], lst[j]
            break
    return tuple(lst)


# --------------------------------------------------------------------------- #
# Finance-flavored worked example (theory-first: P(R) values, not term counts)
# --------------------------------------------------------------------------- #

# Labels reuse the BM25 corpus names for continuity across the probabilistic-IR
# track; here each document carries a calibrated probability of relevance to the
# query "interest rate exposure" rather than term frequencies.
_DOCS = ["filing-onpoint", "news-macro", "transcript-pad", "filing-fx", "transcript-short"]
_P = np.array([0.82, 0.61, 0.55, 0.30, 0.12])  # already in PRP-optimal (descending) order
_QUERY = "interest rate exposure"


# --------------------------------------------------------------------------- #
# Verification harness — the theorem and its corollaries, made executable
# --------------------------------------------------------------------------- #

def test_prp_maximizes_over_all_permutations() -> None:
    """The theorem, brute-forced. Over every permutation of the corpus, the
    PRP order attains the maximum expected relevant@k at every cutoff k, and no
    permutation strictly exceeds it at any k."""
    p = _P
    n = len(p)
    best = prp_order(p)
    for k in range(1, n + 1):
        best_k = expected_relevant_at_k(best, p, k)
        for order in permutations(range(n)):
            assert expected_relevant_at_k(order, p, k) <= best_k + 1e-12, (k, order)
    print("  [ok] PRP order maximizes expected relevant@k over all permutations, every k")


def test_prp_minimizes_cost_over_all_permutations() -> None:
    """The cost-model form: for several linear cost settings, the PRP order
    minimizes expected cost at every cutoff over all permutations."""
    p = _P
    n = len(p)
    best = prp_order(p)
    for c_fp, c_fn in ((1.0, 1.0), (2.0, 1.0), (1.0, 3.0)):
        for k in range(1, n + 1):
            best_cost = expected_cost(best, p, k, c_fp, c_fn)
            for order in permutations(range(n)):
                assert expected_cost(order, p, k, c_fp, c_fn) >= best_cost - 1e-12, (c_fp, c_fn, k)
    print("  [ok] PRP order minimizes expected cost at every k for every linear cost setting")


def test_adjacent_swap_lemma() -> None:
    """Lemma 1: swapping an out-of-order adjacent pair never lowers expected
    relevant@k at any cutoff and strictly raises it at the cutoff between the
    pair. Checked over random orderings from a single RNG stream."""
    rng = np.random.default_rng(20260619)
    for _ in range(2000):
        p = rng.random(5)
        order = tuple(rng.permutation(5).tolist())
        # find an out-of-order adjacent pair
        j = next((i for i in range(4) if p[order[i]] < p[order[i + 1]]), None)
        if j is None:
            continue
        swapped = list(order)
        swapped[j], swapped[j + 1] = swapped[j + 1], swapped[j]
        swapped = tuple(swapped)
        for k in range(1, 6):
            delta = expected_relevant_at_k(swapped, p, k) - expected_relevant_at_k(order, p, k)
            assert delta >= -1e-12, (k, delta)
        # strict improvement exactly at the cutoff between the swapped pair
        strict = (expected_relevant_at_k(swapped, p, j + 1)
                  - expected_relevant_at_k(order, p, j + 1))
        assert strict > 1e-12, strict
    print("  [ok] adjacent-swap lemma: a swap weakly helps every cutoff, strictly one")


def test_bubble_sort_terminates_at_prp() -> None:
    """The exchange argument's completion: repeatedly swapping out-of-order
    adjacent pairs strictly decreases inversions and terminates at the PRP order."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        p = rng.random(6)
        order = tuple(rng.permutation(6).tolist())
        prev_inv = inversions(order, p)
        steps = 0
        while inversions(order, p) > 0:
            order = one_bubble_step(order, p)
            inv = inversions(order, p)
            assert inv < prev_inv, "inversions did not strictly decrease"
            prev_inv = inv
            steps += 1
            assert steps <= 6 * 6, "did not terminate"
        assert order == prp_order(p)
    print("  [ok] bubble-sort completion: inversions fall to zero, ending at the PRP order")


def test_precision_recall_special_case() -> None:
    """Proposition 2: under the 1/0 model, PRP maximizes expected precision@k and
    recall@k for every k (the prefix-sum rearrangement)."""
    p = _P
    n = len(p)
    total = float(p.sum())
    best = prp_order(p)
    for k in range(1, n + 1):
        prec_best = expected_relevant_at_k(best, p, k) / k
        rec_best = expected_relevant_at_k(best, p, k) / total
        for order in permutations(range(n)):
            assert expected_relevant_at_k(order, p, k) / k <= prec_best + 1e-12
            assert expected_relevant_at_k(order, p, k) / total <= rec_best + 1e-12
    print("  [ok] PRP maximizes expected precision@k and recall@k for every k")


def test_monotone_transform_invariance() -> None:
    """Proposition 3 (the handoff to BM25): ranking by P(R) equals ranking by the
    odds P/(1-P) and by the log-odds log P/(1-P), since both transforms are
    strictly increasing."""
    p = _P
    odds = p / (1.0 - p)
    log_odds = np.log(odds)
    by_p = prp_order(p)
    by_odds = tuple(sorted(range(len(p)), key=lambda i: (-odds[i], i)))
    by_log = tuple(sorted(range(len(p)), key=lambda i: (-log_odds[i], i)))
    assert by_p == by_odds == by_log, (by_p, by_odds, by_log)
    print("  [ok] ranking by P(R), by odds, and by log-odds are identical")


def test_prp_beats_plausible_alternative() -> None:
    """The worked-example flip: a length-biased retriever floats verbose documents
    (the long transcript, the wordier FX and macro items) to the top and buries the
    terse on-point filing below the cutoff. It scores strictly worse than the PRP
    order on expected relevant@3."""
    p = _P
    prp = prp_order(p)                                   # filing-onpoint first
    # length-biased order: longest/most verbose first, concise filing-onpoint sunk to #4
    biased = (2, 3, 1, 0, 4)                             # transcript-pad, filing-fx, news-macro, filing-onpoint, transcript-short
    assert _DOCS[biased[0]] == "transcript-pad"
    assert _DOCS[prp[0]] == "filing-onpoint"
    assert biased.index(0) == 3, "the concise on-point filing should be buried at rank 4"
    assert expected_relevant_at_k(prp, p, 3) > expected_relevant_at_k(biased, p, 3) + 1e-9
    print("  [ok] PRP order beats the length-biased order on expected relevant@3 "
          f"({expected_relevant_at_k(prp, p, 3):.2f} vs {expected_relevant_at_k(biased, p, 3):.2f})")


def test_independence_break_diversity() -> None:
    """Honest counterexample: when utility is NON-additive (a redundancy discount
    on a second near-duplicate document), the PRP order is no longer optimal — the
    additivity assumption the theorem rests on genuinely fails here, which is the
    regime diversity / MMR addresses."""
    # d0, d1 are near-duplicates on topic A; d2 is the diverse topic-B document.
    p = np.array([0.80, 0.78, 0.60])
    topic = ["A", "A", "B"]
    gamma = 0.2  # a repeated topic's second hit counts at a discount

    def redundancy_utility(order: tuple[int, ...], k: int) -> float:
        seen: set[str] = set()
        u = 0.0
        for d in order[:k]:
            u += (1.0 if topic[d] not in seen else gamma) * p[d]
            seen.add(topic[d])
        return u

    prp = prp_order(p)                                  # (0, 1, 2): both near-dups first
    best = max(permutations(range(3)), key=lambda o: redundancy_utility(o, 2))
    assert redundancy_utility(best, 2) > redundancy_utility(prp, 2) + 1e-9
    assert topic[best[0]] != topic[best[1]], "the optimal non-additive order is diversified"
    print("  [ok] under redundancy-discounted (non-additive) utility, PRP is NOT optimal")


# --------------------------------------------------------------------------- #
# Demos (printed, non-asserting) — feed the viz and the page
# --------------------------------------------------------------------------- #

def worked_example() -> None:
    p = _P
    prp = prp_order(p)
    biased = (2, 3, 1, 0, 4)
    print(f"  query={_QUERY!r}  P(R) by document:")
    for d, pr in zip(_DOCS, p):
        print(f"    {d:<16} P(R)={pr:.2f}")
    print("  expected relevant in top-k (cumulative Σ P(R)):")
    for k in range(1, len(p) + 1):
        print(f"    k={k}:  PRP={expected_relevant_at_k(prp, p, k):.2f}   "
              f"biased={expected_relevant_at_k(biased, p, k):.2f}")


def viz_constants() -> None:
    """The numbers ExchangeArgumentLaboratory.tsx mirrors: the P(R) values, the
    PRP-optimal cumulative curve, and a scrambled start order's curve."""
    p = _P
    prp = prp_order(p)
    scrambled = (2, 4, 0, 3, 1)  # the laboratory's default starting order
    print(f"  P(R)            = {[round(float(x), 2) for x in p]}")
    print(f"  optimal cum Σ   = {[round(expected_relevant_at_k(prp, p, k), 2) for k in range(1, 6)]}")
    print(f"  scrambled cum Σ = {[round(expected_relevant_at_k(scrambled, p, k), 2) for k in range(1, 6)]}")
    print(f"  scrambled inversions = {inversions(scrambled, p)}")


if __name__ == "__main__":
    print("Probability Ranking Principle verification harness")
    test_prp_maximizes_over_all_permutations()
    test_prp_minimizes_cost_over_all_permutations()
    test_adjacent_swap_lemma()
    test_bubble_sort_terminates_at_prp()
    test_precision_recall_special_case()
    test_monotone_transform_invariance()
    test_prp_beats_plausible_alternative()
    test_independence_break_diversity()
    print("Worked example:")
    worked_example()
    print("Viz constants (mirrored to the decimal in ExchangeArgumentLaboratory.tsx):")
    viz_constants()
    print("All checks passed.")
