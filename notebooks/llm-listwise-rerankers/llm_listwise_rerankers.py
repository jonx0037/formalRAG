"""LLM rerankers — listwise permutation objectives, the sliding window as a sort, and distillation.

The reference implementation for the formalRAG `llm-listwise-rerankers` topic, the TERMINAL node of the
ranking-fusion learning-to-rank sub-track (shipping it completes the track). The predecessor
(`lambdarank-lambdamart-listwise`) scored each document from a FIXED feature vector — blind to the other
documents except through the loss. The LLM reranker drops that assumption: feed the whole candidate list
into one model and it emits a PERMUTATION directly, scoring documents in each other's context (the most
listwise objective there is). The arc:

  - M1 PERMUTATION OBJECTIVE (the ListMLE bridge): the LLM is a Plackett-Luce sampler at temperature tau.
    tau -> 0 emits the ideal order; tau -> inf is uniform. We REUSE the predecessor's PL machinery.
  - M2 SLIDING WINDOW = BUBBLE SORT (RankGPT, Sun et al. 2023): a context window of w < n forces the model
    to slide a window back-to-front and locally re-sort. One pass with a perfect comparator bubbles the
    global best into the top window; the call count is O(n/s) per pass, vs O(n^2) all-pairs and O(n)
    pointwise. The Omega(n log n) comparison floor is the reason a bounded window cannot sort exactly in
    O(n/s) comparisons.
  - M3 POSITIONAL BIAS (lost-in-the-middle, Liu et al. 2024): the in-window order is biased by position;
    averaging over presentation orders / alternating the slide direction flattens it.
  - M4 RANK AGGREGATION (social choice): K noisy permutations are aggregated by Borda, RRF, the Kemeny
    median (NP-hard), or the Dwork et al. 2001 Markov-chain method (the consensus = a comparison random
    walk's stationary distribution). Aggregation concentrates the consensus toward truth.
  - M5 PERMUTATION DISTILLATION (RankVicuna/RankZephyr): the expensive LLM teacher's permutations train a
    cheap linear listwise student via the imported ListMLE fit. A perfect teacher's student IS the
    predecessor's own listwise scorer.
  - M6 THE COST-QUALITY FRONTIER: all methods on ONE shared corpus, cost = LLM calls per query.

THE LOAD-BEARING CONSTRAINT: this site bakes only reproducible numbers and never calls a cloud service.
There is NO real LLM call. The LLM reranker is a SEEDED NOISY PERMUTATION ORACLE: it sees a window of
candidates and emits a permutation = the true oracle ranking corrupted by (a) Plackett-Luce noise at
temperature tau and (b) a position-dependent score bias. Every PROVABLE claim here is therefore
ALGORITHMIC (sorting guarantees, aggregation variance reduction, distillation cost) — NOT "the LLM beats
the cross-encoder", which is empirical, unshowable synthetically, and is OUT of scope (see rigorFlag).

We do NOT rebuild the corpus, legs, grades, PL machinery, social-choice rules, the positional kernel, or
the cost constants — we IMPORT them. Import graph != pedagogical DAG: the frontmatter prereqs are only the
two graph edges (lambdarank-lambdamart-listwise + cross-encoders-reranking).

Headlines, all BUILT-AND-RUN before written:
  - tau -> 0 PL sample == the imported optimal_permutation (the collapse), abilities = -rank_in_pi* so the
    grade-first ideal order is reproduced byte-for-byte; emitted NLL == imported listmle_loss_scores.
  - sliding-window call count == P * (ceil((n-w)/s) + 1), exactly; one perfect back-to-front pass bubbles
    the global best into the top window; seed-averaged top-k recall is MONOTONE in passes (noisy: not
    per-seed); back-to-front beats front-to-back at one pass (the provable asymmetry).
  - the positional dip degrades top-k recall (it BITES) and averaging presentation orders recovers it.
  - aggregating K noisy ballots concentrates the per-doc averaged rank at the 1/sqrt(K) CLT rate and
    monotonically reduces the consensus Kendall-tau to truth; Borda/RRF collapse to the single perm at
    K=1; the Markov-chain consensus approximates the brute Kemeny median.
  - a perfect (tau -> 0) LLM teacher's distilled student == the imported fit_listmle byte-for-byte; the
    student answers at 0 inference LLM calls while the teacher costs O(n/s)*C_LLM per query.

rigorFlag: the LLM is a SIMULATED noisy permutation oracle — the site never calls a real model, so every
"LLM" quality figure is a controlled stand-in and "the LLM is most accurate" is OUT of scope. An
autoregressive permutation model, like LambdaRank, descends NO scalar loss at inference; the Plackett-Luce
objective is what a DISTILLED student is trained on, not what the teacher optimizes per query. One
back-to-front pass guarantees only the top-1, not the full top-k (Omega(n log n) comparison floor) — recall
climbs with passes, it is not exact in one pass. The positional bias is EMPIRICAL (Liu et al.), a baked
per-position kernel, not derived. The aggregation rate is the variance-reduction of K noisy ballots, a
sample-complexity statement asserted on the simulated oracle, not a universal constant. The Kemeny median
is NP-hard (brute force only at small n). C_LLM is a stipulated cost ratio, not measured. On this forgiving
synthetic finance corpus the method deltas sit inside the confidence interval — the seed-free wins are
STRUCTURAL (the call-count law, the bias-correction flattening, the aggregation variance reduction, the
distillation speedup), not the aggregate scores. Grades are exact-MaxSim oracle tertiles, a neutral
stand-in for human editorial judgments.

Run:  uv run --with numpy --with scipy --with scikit-learn \\
        python notebooks/llm-listwise-rerankers/llm_listwise_rerankers.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------------------------------- #
# Import the predecessor + the published stack. Mirror lambdarank's sys.path block (the capstone substrate
# pulls the whole multi-vector subtree; ndcg pulls set-metrics), and append the dirs for rank-fusion-rrf
# (social choice), retrieval-vs-long-context (the lost-in-the-middle kernel), and cross-encoders-reranking
# (the cost constants). Import graph != DAG: the frontmatter prereqs are only lambdarank + cross-encoders.
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "johnson-lindenstrauss",
    "vector-quantization-lloyd-max",
    "product-quantization",
    "ivf-voronoi-partitioning",
    "bm25",
    "rank-fusion-rrf",
    "dense-retrieval-dual-encoders",
    "late-interaction-learned-sparse",
    "multi-vector-ann-retrieval",
    "filtered-incremental-ann",
    "set-metrics-precision-recall-map-mrr",
    "ndcg-discount-geometry",
    "capstone-multimodal-financial-rag",
    "learning-to-rank-pairwise",
    "lambdarank-lambdamart-listwise",
    "the-retrieval-problem",
    "infonce-contrastive-objective",
    "pmi-retrieval-value",
    "retriever-as-noisy-channel",
    "retrieval-vs-long-context",
    "embedding-dimension-lower-bounds",
    "cross-encoders-reranking",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lambdarank_lambdamart_listwise import (                              # noqa: E402
    _corpus, optimal_permutation, candidate_set,
    listmle_loss_scores, listmle_loss, listmle_grad, fit_listmle,
    ranking_from_w, mean_recall_over,
    TOPK, SEED, GAIN, DISCOUNT,
    ndcg_at_k, recall_at_k, metric_summary,
)
from rank_fusion_rrf import rrf_fuse, borda, kendall_tau, kemeny_bruteforce, kemeny_cost   # noqa: E402
from retrieval_vs_long_context import positional_weight, POS_DIP          # noqa: E402
from cross_encoders_reranking import C_RETRIEVE, C_CE                     # noqa: E402

# --- module constants (the simulated-oracle knobs; tuned by build-and-run, the .py owns them) ---------- #
TAU_COLLAPSE = 1e-9          # tau -> 0: the noiseless oracle (the collapse anchor)
TAU_NOISY = 3.0            # a "good but noisy" LLM oracle (abilities = -rank, gap 1; tuned by build-and-run)
N_POOL_NEG = 50             # grade-0 negatives in the rerank pool (the pool is 10 gold + N_POOL_NEG negs)
POOL = TOPK + N_POOL_NEG    # the rerank pool size n (= 60): all 10 gold present, the rest hard negatives
WINDOW = 15                 # the LLM context window w < n; overlap w - s = 5 so docs migrate but one pass
STEP = 10                   # does NOT sort (tuned: a clean rising-then-plateau recall-vs-passes curve)
N_PASSES = 4                # default reranking passes P
RECALL_K = TOPK            # recall@k cutoff for the rerank pool (gold scattered => recall@10 in ~[0.17, 1])
N_AGG = 16                 # K permutations aggregated by default (M4)
AGG_SEEDS = 24             # independent oracle-seed streams averaged to smooth the noisy curves
K_GRID = (1, 2, 4, 8, 16, 32)   # the aggregation-size sweep (sqrt(K) spans an ~5.7x range)
MC4_DAMP = 0.15            # PageRank-style restart so the Markov chain is ergodic (unique stationary dist)
TAU_BIAS = 0.6            # a SHARP oracle for M3, so the lost-in-the-middle bias is isolated from PL noise
POS_BIAS_GAIN = 6.0        # multiplier on log(positional_weight): how hard the lost-in-the-middle dip bites
N_PRESENT = 4             # presentation orders averaged to flatten the positional bias (M3 correction)
C_LLM = 250.0             # one windowed listwise LLM call, priced >> C_CE=25 (a stipulated ratio, rigorFlag)

# A DETERMINISTIC label -> int map for seeding (Python's hash() of strings is randomized per process via
# PYTHONHASHSEED, so it must never seed a baked number — that silently breaks reproducibility run-to-run).
_SEED_TAG = {
    "back_to_front": 0, "front_to_back": 1,
    "none": 2, "biased": 3, "corrected": 4,
    "dense_best_leg": 5, "rrf_3legs": 6, "pointwise_llm": 7,
    "sliding_window_llm": 8, "allpairs_llm": 9, "distilled_student": 10,
}


# =========================================================================== #
# Movement 0 — the substrate (imported verbatim) + the rerank pool.
# =========================================================================== #

def rerank_pool(corpus: dict, q: int, n_neg: int = N_POOL_NEG) -> list[int]:
    """The per-query first-stage pool the LLM reranks: the K=TOPK graded docs plus the n_neg HARDEST
    grade-0 negatives (highest oracle score among the non-relevant). All gold are present, so first-stage
    recall is not the bottleneck — the reranker's job is purely to ORDER the pool. Generalizes the
    imported candidate_set (which uses N_NEG=10) to a larger pool so the window slides legibly."""
    grades_q = corpus["grades"][q]
    oracle = corpus["oracle_scores"][q]
    rel = sorted(grades_q.keys())
    negs = [int(d) for d in np.argsort(-oracle) if d not in grades_q][:n_neg]
    return rel + negs


def shuffled_pool(corpus: dict, q: int, rng: np.random.Generator, n_neg: int = N_POOL_NEG) -> list[int]:
    """A seeded shuffle of the rerank pool — the realistic first-stage order in which gold docs are
    SCATTERED (not gold-first). Without the shuffle the pool is already pi*-ordered and every reranking
    headline (passes help, direction matters, bias bites) is vacuous."""
    pool = rerank_pool(corpus, q, n_neg)
    perm = rng.permutation(len(pool))
    return [pool[i] for i in perm]


# =========================================================================== #
# Movement 1 — the listwise permutation objective: the LLM as a Plackett-Luce sampler.
# =========================================================================== #

def oracle_abilities(corpus: dict, q: int, docs: list[int]) -> np.ndarray:
    """The Plackett-Luce log-abilities of a candidate set, aligned to `docs`: ability_i = -(rank of docs[i]
    in the ideal order pi*). A strictly DECREASING function of pi*-rank, so argsort(-abilities) == pi*
    exactly — the LLM is a noisy copy of the IDEAL ranking. We index abilities by pi*-rank rather than raw
    oracle_scores so the tau -> 0 collapse reproduces optimal_permutation (which sorts grade-FIRST) for any
    grade/oracle relationship; on this corpus grades are oracle tertiles so the two coincide, but -rank is
    the order-faithful construction that does not depend on that coincidence. Uniform gaps (1 per rank)
    give tau a single interpretation across the list."""
    star = optimal_permutation(corpus, q, docs)
    rank_of = {d: r for r, d in enumerate(star)}
    return np.array([-float(rank_of[d]) for d in docs], dtype=float)


def pl_sample(abilities: np.ndarray, tau: float, rng: np.random.Generator) -> list[int]:
    """A seeded Plackett-Luce sample at temperature tau: sorting by Gumbel-perturbed log-abilities IS a PL
    draw (argsort of abilities/tau + Gumbel ~ PL(softmax(abilities/tau))). Returns LOCAL indices into
    `abilities`, best first. tau -> 0: the deterministic abilities dominate -> argsort(-abilities) = pi*.
    tau -> inf: the Gumbel noise dominates -> uniform permutation."""
    g = rng.gumbel(size=abilities.shape[0])
    keys = abilities / max(tau, 1e-12) + g
    return np.argsort(-keys, kind="stable").tolist()


def noisy_oracle_perm(corpus: dict, q: int, docs: list[int], tau: float,
                      rng: np.random.Generator) -> list[int]:
    """The LLM-as-noisy-oracle primitive: emit a permutation of `docs` (returned as DOC IDS, best first) =
    pi* corrupted by Plackett-Luce noise at temperature tau. Every later movement calls this."""
    perm_local = pl_sample(oracle_abilities(corpus, q, docs), tau, rng)
    return [docs[i] for i in perm_local]


def emitted_nll(corpus: dict, q: int, docs: list[int], perm_local: list[int], tau: float) -> float:
    """The listwise negative log-likelihood of an emitted permutation under the temperature-tau abilities —
    the imported ListMLE objective, now scoring an LLM-emitted order. A twin by construction."""
    return listmle_loss_scores(oracle_abilities(corpus, q, docs) / max(tau, 1e-12), perm_local)


def first_position_marginal(abilities: np.ndarray, tau: float, n_draws: int,
                            rng: np.random.Generator) -> np.ndarray:
    """The empirical distribution over which document the oracle ranks FIRST, across n_draws PL samples.
    tau -> 0 concentrates on the best ability; tau -> inf flattens toward uniform (1/m each)."""
    m = abilities.shape[0]
    counts = np.zeros(m)
    for _ in range(n_draws):
        counts[pl_sample(abilities, tau, rng)[0]] += 1.0
    return counts / n_draws


# =========================================================================== #
# Movement 2 — sliding-window reranking = bubble sort (the systems-math core).
# =========================================================================== #

def window_starts(n: int, w: int, s: int) -> list[int]:
    """The ascending window start positions covering [0, n): 0, s, 2s, ..., with the final window pinned to
    end at n. Their count is the per-pass oracle-call count."""
    if n <= w:
        return [0]
    starts = list(range(0, n - w + 1, s))
    if starts[-1] != n - w:
        starts.append(n - w)
    return starts


def oracle_call_count(n: int, w: int, s: int, passes: int) -> int:
    """The exact sliding-window cost law: P * (ceil((n - w)/s) + 1) oracle calls — one per window placement,
    P passes. O(n/s) per pass; O(n) for constant passes — vs O(n^2) all-pairs and O(n) pointwise."""
    if n <= w:
        return passes
    return passes * (math.ceil((n - w) / s) + 1)


def oracle_window_perm(corpus: dict, q: int, window_docs: list[int], tau: float,
                       rng: np.random.Generator, bias: bool = False) -> list[int]:
    """Re-sort one window's documents with the noisy oracle. With bias=True the oracle perceives
    center-of-window documents as weaker (the lost-in-the-middle dip; see M3)."""
    ab = oracle_abilities(corpus, q, window_docs)
    if bias:
        ab = ab + POS_BIAS_GAIN * np.array(
            [math.log(positional_weight(p, len(window_docs))) for p in range(len(window_docs))])
    perm_local = pl_sample(ab, tau, rng)
    return [window_docs[i] for i in perm_local]


def sliding_window_rerank(corpus: dict, q: int, order: list[int], tau: float, rng: np.random.Generator,
                          w: int = WINDOW, s: int = STEP, passes: int = N_PASSES,
                          direction: str = "back_to_front", bias: bool = False) -> list[int]:
    """RankGPT sliding-window reranking: slide a window of size w (step s) across the current order,
    re-sorting each window in place with the noisy oracle, for `passes` passes. Back-to-front (the RankGPT
    default) processes the LAST window first so a relevant doc buried in the tail can rise toward the front
    in a single pass. Returns the final document order."""
    cur = list(order)
    n = len(cur)
    starts = window_starts(n, w, s)
    if direction == "back_to_front":
        starts = list(reversed(starts))
    for _ in range(passes):
        for st in starts:
            window = cur[st:st + w]
            cur[st:st + w] = oracle_window_perm(corpus, q, window, tau, rng, bias=bias)
    return cur


def _pool_recall(corpus: dict, q: int, order: list[int], k: int = RECALL_K) -> float:
    """recall@k of a pool ordering against the query's gold set (the imported set metric; the pool holds
    all gold, so this measures how many gold the reranker has bubbled into the top k)."""
    return recall_at_k(order, corpus["truth_sets"][q], k)


def topk_recall_vs_passes(corpus: dict, qs: list[int], tau: float, max_passes: int,
                          seeds: int = AGG_SEEDS, k: int = RECALL_K,
                          w: int = WINDOW, s: int = STEP) -> list[tuple]:
    """Seed-averaged mean recall@k after each pass count, over a SHUFFLED start order. The noisy curve is
    monotone only in expectation, so we average over `seeds` independent oracle streams."""
    out = []
    for passes in range(0, max_passes + 1):
        vals = []
        for sd in range(seeds):
            rng = np.random.default_rng([SEED, 2, passes, sd])
            r = []
            for q in qs:
                start = shuffled_pool(corpus, q, np.random.default_rng([SEED, 7, q, sd]))
                order = start if passes == 0 else sliding_window_rerank(
                    corpus, q, start, tau, rng, w=w, s=s, passes=passes)
                r.append(_pool_recall(corpus, q, order, k))
            vals.append(float(np.mean(r)))
        out.append((passes, float(np.mean(vals))))
    return out


def direction_asymmetry(corpus: dict, qs: list[int], tau: float, seeds: int = AGG_SEEDS,
                        k: int = RECALL_K, w: int = WINDOW, s: int = STEP) -> dict:
    """One-pass seed-averaged recall@k, back-to-front vs front-to-back, over a SHUFFLED start. Back-to-front
    can carry a tail doc to the front in one pass; front-to-back advances it one window per pass."""
    res = {}
    for direction in ("back_to_front", "front_to_back"):
        vals = []
        for sd in range(seeds):
            rng = np.random.default_rng([SEED, 3, sd, _SEED_TAG[direction]])
            r = []
            for q in qs:
                start = shuffled_pool(corpus, q, np.random.default_rng([SEED, 7, q, sd]))
                order = sliding_window_rerank(corpus, q, start, tau, rng, w=w, s=s, passes=1,
                                              direction=direction)
                r.append(_pool_recall(corpus, q, order, k))
            vals.append(float(np.mean(r)))
        res[direction] = float(np.mean(vals))
    return res


def perfect_pass_bubbles_best(corpus: dict, q: int, w: int = WINDOW, s: int = STEP) -> bool:
    """The perfect-comparator theorem: with tau -> 0, ONE back-to-front pass places the global best
    (pi*-rank 0) document inside the top window [0, w). The witness for the bubble-sort guarantee."""
    rng = np.random.default_rng([SEED, 99, q])
    start = shuffled_pool(corpus, q, np.random.default_rng([SEED, 7, q, 0]))
    pool = rerank_pool(corpus, q)
    best_doc = optimal_permutation(corpus, q, pool)[0]
    order = sliding_window_rerank(corpus, q, start, TAU_COLLAPSE, rng, w=w, s=s, passes=1)
    return order.index(best_doc) < w


# =========================================================================== #
# Movement 3 — positional bias (lost-in-the-middle) and its correction.
# =========================================================================== #

def position_bias_kernel(w: int = WINDOW) -> list[float]:
    """The in-window positional attention multiplier (the imported lost-in-the-middle U-curve): high at the
    two ends, depressed in the middle. A buried passage is a soft erasure."""
    return [float(positional_weight(p, w)) for p in range(w)]


def gold_passes_through_center(corpus: dict, qs: list[int], seeds: int = AGG_SEEDS,
                               w: int = WINDOW, s: int = STEP) -> int:
    """The explicit non-vacuity guard: count (query, seed, pass-1 window) events where a GOLD document
    occupies a window-center slot during reranking — proof the dip can actually act on a relevant doc."""
    count = 0
    lo, hi = w // 2 - 2, w // 2 + 2
    for sd in range(seeds):
        for q in qs:
            gold = corpus["truth_sets"][q]
            start = shuffled_pool(corpus, q, np.random.default_rng([SEED, 7, q, sd]))
            for st in window_starts(len(start), w, s):
                window = start[st:st + w]
                for p in range(lo, hi + 1):
                    if 0 <= p < len(window) and window[p] in gold:
                        count += 1
    return count


def bias_correction_recall(corpus: dict, qs: list[int], mode: str, tau: float = TAU_BIAS,
                           seeds: int = AGG_SEEDS, k: int = RECALL_K,
                           w: int = WINDOW, s: int = STEP) -> float:
    """Seed-averaged recall@k under three regimes, with a SHARP oracle (TAU_BIAS) so the position bias is
    isolated from PL noise. `none`: the unbiased oracle (the ceiling). `biased`: the dip suppresses
    window-center docs. `corrected`: present each window in N_PRESENT random orders and Borda-aggregate, so
    a document at the center in one presentation is at an edge in another and the position bias averages
    out (reversing a window leaves the center fixed, so random presentations — not reversal — are needed)."""
    vals = []
    for sd in range(seeds):
        rng = np.random.default_rng([SEED, 4, sd, _SEED_TAG[mode]])
        r = []
        for q in qs:
            start = shuffled_pool(corpus, q, np.random.default_rng([SEED, 7, q, sd]))
            if mode == "none":
                order = sliding_window_rerank(corpus, q, start, tau, rng, w=w, s=s, passes=N_PASSES)
            elif mode == "biased":
                order = sliding_window_rerank(corpus, q, start, tau, rng, w=w, s=s, passes=N_PASSES,
                                              bias=True)
            elif mode == "corrected":
                order = start
                for _ in range(N_PASSES):
                    order = _corrected_pass(corpus, q, order, tau, rng, w, s)
            else:
                raise ValueError(mode)
            r.append(_pool_recall(corpus, q, order, k))
        vals.append(float(np.mean(r)))
    return float(np.mean(vals))


def _corrected_pass(corpus: dict, q: int, order: list[int], tau: float, rng: np.random.Generator,
                    w: int, s: int) -> list[int]:
    """One bias-corrected back-to-front pass: each window is scored (biased) in N_PRESENT RANDOM
    presentation orders and the emitted ranks are Borda-aggregated. A document visits a different slot in
    each presentation, so the position-dependent penalty averages to a constant across documents and the
    systematic component cancels."""
    cur = list(order)
    n = len(cur)
    for st in reversed(window_starts(n, w, s)):
        window = cur[st:st + w]
        perms = []
        for _ in range(N_PRESENT):
            sh = list(window)
            rng.shuffle(sh)
            perms.append(oracle_window_perm(corpus, q, sh, tau, rng, bias=True))
        cur[st:st + w] = borda(perms)
    return cur


# =========================================================================== #
# Movement 4 — rank aggregation of K noisy permutations (social choice + a fresh Markov chain).
# =========================================================================== #

def sample_k_perms(corpus: dict, q: int, docs: list[int], k: int, tau: float,
                   rng: np.random.Generator) -> list[list[int]]:
    """K independent noisy oracle permutations of `docs` (each a doc-id list, best first) — the K LLM
    windows / prompt-orders / samples that aggregation must reconcile."""
    return [noisy_oracle_perm(corpus, q, docs, tau, rng) for _ in range(k)]


def mc4_aggregate(perms: list[list[int]], docs: list[int], damp: float = MC4_DAMP) -> list[int]:
    """The Dwork et al. (2001) Markov-chain consensus (MC4): a random walk over candidates whose transition
    encodes pairwise majority wins — from a document, step to a uniformly-chosen one that the MAJORITY of
    ballots rank higher, else stay. The consensus ranks documents by the walk's STATIONARY distribution (a
    comparison random walk's stationary distribution — the random-walks up-link). A small restart (damp)
    makes the chain ergodic so the stationary distribution is unique and a clean ranking."""
    m = len(docs)
    above = np.zeros((m, m))                              # above[i, j] = #ballots ranking i before j
    for r in perms:
        pos = {d: p for p, d in enumerate(r)}
        for a in range(m):
            for b in range(m):
                if a != b and pos[docs[a]] < pos[docs[b]]:
                    above[a, b] += 1.0
    P = np.zeros((m, m))
    majority = len(perms) / 2.0                          # hoisted: invariant across the m x m loop
    for j in range(m):                                   # from state j: step up to a majority-better doc
        for i in range(m):
            if i != j and above[i, j] > majority:
                P[j, i] = 1.0 / m
        P[j, j] = 1.0 - P[j].sum()
    P = (1.0 - damp) * P + damp / m                       # ergodic restart (PageRank-style)
    pi = np.full(m, 1.0 / m)
    for _ in range(500):
        pi = pi @ P
    return [docs[i] for i in np.argsort(-pi, kind="stable")]


def aggregate(perms: list[list[int]], docs: list[int], method: str) -> list[int]:
    """Dispatch over the polynomial-time aggregation rules. Borda and RRF return the single permutation at
    K=1; MC4 is the Markov-chain consensus. The Kemeny median is NP-hard and brute-forced separately in
    kemeny_approximation (small candidate sets only) — it is deliberately not a dispatch option here, since
    it would enumerate m! permutations on a full 20-document set."""
    if method == "borda":
        return borda(perms)
    if method == "rrf":
        return rrf_fuse(perms)
    if method == "mc4":
        return mc4_aggregate(perms, docs)
    raise ValueError(method)


def agg_tau_vs_k(corpus: dict, qs: list[int], docs_fn, tau: float, method: str,
                 k_grid=K_GRID, seeds: int = AGG_SEEDS) -> list[tuple]:
    """Seed-averaged mean Kendall-tau between the aggregated consensus and pi*, over the aggregation size K.
    Variance reduction: more ballots -> a consensus closer to truth (tau monotone decreasing in K)."""
    out = []
    for K in k_grid:
        vals = []
        for sd in range(seeds):
            rng = np.random.default_rng([SEED, 5, K, sd])
            taus = []
            for q in qs:
                docs = docs_fn(corpus, q)
                star = optimal_permutation(corpus, q, docs)
                perms = sample_k_perms(corpus, q, docs, K, tau, rng)
                cons = aggregate(perms, docs, method)
                taus.append(kendall_tau(cons, star))
            vals.append(float(np.mean(taus)))
        out.append((K, float(np.mean(vals))))
    return out


def aggregated_rank_concentration(corpus: dict, qs: list[int], docs_fn, tau: float,
                                  k_grid=K_GRID, seeds: int = AGG_SEEDS) -> list[tuple]:
    """The clean 1/sqrt(K) CLT law: the std (over seeds) of a gold document's Borda-aggregated mean rank
    decreases as 1/sqrt(K). Returns (K, std, std*sqrt(K)); the last column is ~constant (the variance
    reduction up-link to concentration inequalities)."""
    out = []
    for K in k_grid:
        ranks = []
        for sd in range(seeds):
            rng = np.random.default_rng([SEED, 6, K, sd])
            per_q = []
            for q in qs:
                docs = docs_fn(corpus, q)
                gold = optimal_permutation(corpus, q, docs)[0]    # the best document
                perms = sample_k_perms(corpus, q, docs, K, tau, rng)
                mean_rank = float(np.mean([r.index(gold) for r in perms]))
                per_q.append(mean_rank)
            ranks.append(float(np.mean(per_q)))
        sd_rank = float(np.std(ranks, ddof=1))
        out.append((K, sd_rank, sd_rank * math.sqrt(K)))
    return out


def kemeny_approximation(corpus: dict, q: int, m: int = 6, tau: float = TAU_NOISY,
                         seed: int = 0) -> dict:
    """On a SMALL candidate set (m <= 7, so the brute Kemeny is tractable), the ratio of each aggregator's
    Kemeny cost to the exact NP-hard optimum — Borda/RRF/MC4 as polynomial-time approximations."""
    rng = np.random.default_rng([SEED, 8, q, seed])
    docs = rerank_pool(corpus, q)[:m]
    perms = sample_k_perms(corpus, q, docs, N_AGG, tau, rng)
    opt, opt_cost = kemeny_bruteforce(perms)
    out = {"opt_cost": int(opt_cost)}
    for method in ("borda", "rrf", "mc4"):
        cost = kemeny_cost(aggregate(perms, docs, method), perms)
        out[method] = float(cost / opt_cost) if opt_cost > 0 else 1.0
    return out


# =========================================================================== #
# Movement 5 — permutation distillation: the LLM teacher trains a cheap listwise student.
# =========================================================================== #

def teacher_targets(corpus: dict, qs: list[int], tau: float, rng: np.random.Generator) -> dict:
    """The LLM teacher's emitted permutation per query, over the imported candidate_set, as candidate-local
    indices — the distillation targets. At tau -> 0 every target == pi* (the imported optimal_permutation)."""
    out = {}
    for q in qs:
        docs = candidate_set(corpus, q)
        pos = {d: i for i, d in enumerate(docs)}
        perm = noisy_oracle_perm(corpus, q, docs, tau, rng)
        out[q] = [pos[d] for d in perm]
    return out


def fit_student_to_teacher(corpus: dict, qs: list[int], tau: float, rng: np.random.Generator,
                           x0: np.ndarray | None = None) -> np.ndarray:
    """Distill the teacher's permutations into a 3-leg linear ListMLE student. This re-expresses the
    imported fit_listmle's convex L-BFGS-B solve with the TEACHER's emitted permutations as targets in
    place of pi* — the ONE place we re-express rather than import (fit_listmle hardcodes pi*). At tau -> 0
    the teacher targets equal pi*, so the blocks, solve, and result are byte-for-byte fit_listmle."""
    targets = teacher_targets(corpus, qs, tau, rng)
    blocks = [(corpus["feats"][q][candidate_set(corpus, q)], targets[q]) for q in qs]

    def loss(w):
        return sum(listmle_loss(w, fc, pl) for fc, pl in blocks)

    def grad(w):
        return sum(listmle_grad(w, fc, pl) for fc, pl in blocks)

    x0 = np.zeros(3) if x0 is None else x0
    return minimize(loss, x0, jac=grad, method="L-BFGS-B").x


def teacher_call_cost(n_queries: int, n: int = POOL, w: int = WINDOW, s: int = STEP,
                      passes: int = N_PASSES) -> float:
    """The LLM teacher's query-time cost: O(n/s) windowed listwise calls per query at C_LLM each."""
    return float(n_queries) * oracle_call_count(n, w, s, passes) * C_LLM


def student_call_cost(n_queries: int) -> float:
    """The distilled student's query-time cost: one precomputed-embedding MIPS pass per query (the LLM
    teacher's cost was paid OFFLINE during distillation; inference is 0 LLM calls)."""
    return float(n_queries) * C_RETRIEVE


def distill_speedup(n_queries: int = 1) -> float:
    """Teacher query-time cost / student query-time cost — the price of the LLM's expressivity the cheap
    student buys back at inference."""
    s = student_call_cost(n_queries)
    return float(teacher_call_cost(n_queries) / s) if s > 0 else 1.0


# =========================================================================== #
# Movement 6 — the cost-quality frontier (all methods on ONE shared corpus; cost = LLM calls/query).
# =========================================================================== #

def _leg_pool_order(corpus: dict, q: int, leg: str, pool: list[int]) -> list[int]:
    """The pool ordered by a single leg's score (descending, stable) — the dense/lexical/late baselines
    restricted to the shared pool."""
    s = corpus["leg_scores"][leg][q]
    return sorted(pool, key=lambda d: (-float(s[d]), d))


def _allpairs_order(corpus: dict, q: int, pool: list[int], tau: float,
                    rng: np.random.Generator) -> list[int]:
    """The O(n^2) all-pairs LLM (pairwise ranking prompting): every pair is compared by the noisy oracle;
    rank by win count. The quality ceiling (most comparisons) at the highest cost."""
    m = len(pool)
    ab = oracle_abilities(corpus, q, pool)
    wins = np.zeros(m)
    for i in range(m):
        for j in range(i + 1, m):
            # a noisy pairwise comparison = a 2-element PL draw on the two abilities
            order = pl_sample(ab[[i, j]], tau, rng)
            if order[0] == 0:
                wins[i] += 1.0
            else:
                wins[j] += 1.0
    return [pool[i] for i in np.argsort(-wins, kind="stable")]


def frontier(corpus: dict, qs: list[int], tau: float = TAU_NOISY, seeds: int = AGG_SEEDS) -> dict:
    """The cost-quality frontier on the shared pool: per method, seed-averaged recall@TOPK and NDCG@TOPK
    over `qs` (the LLM methods averaged over oracle seeds), per-query NDCG samples for the CI, and the
    oracle-call cost. The cross-encoder is NOT a method here — it is the cost UNIT (C_CE) the LLM call is
    priced against. The distilled student answers at 0 inference LLM calls."""
    n = POOL
    student_w = fit_student_to_teacher(corpus, corpus["train_q"], tau,
                                       np.random.default_rng([SEED, 10]))
    methods = {
        "dense_best_leg": {"calls": 0.0},
        "rrf_3legs": {"calls": 0.0},
        "pointwise_llm": {"calls": float(n)},
        "sliding_window_llm": {"calls": float(oracle_call_count(n, WINDOW, STEP, N_PASSES))},
        "allpairs_llm": {"calls": float(n * (n - 1) // 2)},
        "distilled_student": {"calls": 0.0},
    }
    out = {}
    for name, meta in methods.items():
        ndcgs_per_q, recalls_per_q = [], []
        for q in qs:
            pool = rerank_pool(corpus, q)
            if name in ("dense_best_leg", "rrf_3legs", "distilled_student"):
                if name == "dense_best_leg":
                    order = _leg_pool_order(corpus, q, "dense", pool)
                elif name == "rrf_3legs":
                    order = rrf_fuse([_leg_pool_order(corpus, q, leg, pool)
                                      for leg in ("lexical", "dense", "late_interaction")])
                else:
                    full = ranking_from_w(corpus, q, student_w)
                    rank_of = {d: i for i, d in enumerate(full)}
                    order = sorted(pool, key=lambda d: rank_of[d])
                ndcgs_per_q.append(ndcg_at_k(order, corpus["grades"][q], TOPK, GAIN, DISCOUNT))
                recalls_per_q.append(_pool_recall(corpus, q, order, TOPK))
            else:
                nd, rc = [], []
                # The abilities are seed-invariant (deterministic in corpus/q/pool); hoist them out of the
                # seed loop. Pointwise scoring = a single sort by independent Gumbel-perturbed abilities, so
                # each document is judged ONCE in isolation (n single-document judgments).
                ab_pw = oracle_abilities(corpus, q, pool) if name == "pointwise_llm" else None
                for sd in range(seeds):
                    rng = np.random.default_rng([SEED, 11, sd, _SEED_TAG[name]])
                    start = shuffled_pool(corpus, q, np.random.default_rng([SEED, 7, q, sd]))
                    if name == "pointwise_llm":
                        order = [pool[i] for i in pl_sample(ab_pw, tau, rng)]
                    elif name == "sliding_window_llm":
                        order = sliding_window_rerank(corpus, q, start, tau, rng)
                    else:  # allpairs_llm
                        order = _allpairs_order(corpus, q, pool, tau, rng)
                    nd.append(ndcg_at_k(order, corpus["grades"][q], TOPK, GAIN, DISCOUNT))
                    rc.append(_pool_recall(corpus, q, order, TOPK))
                ndcgs_per_q.append(float(np.mean(nd)))
                recalls_per_q.append(float(np.mean(rc)))
        out[name] = {
            "recall": float(np.mean(recalls_per_q)),
            "ndcg": float(np.mean(ndcgs_per_q)),
            "cost": meta["calls"],
            "ci": metric_summary(np.array(ndcgs_per_q)),
        }
    return out


def verdict(fr: dict) -> dict:
    """The honest, seed-free structural reading of the frontier — the call-count law and the distillation
    speedup, NOT 'the LLM is most accurate' (out of scope). The winner is pinned to the run, not pre-baked."""
    best = max(fr, key=lambda m: fr[m]["ndcg"])
    return {
        "winner_ndcg": best,
        "sliding_vs_allpairs_cost_ratio": fr["allpairs_llm"]["cost"] / fr["sliding_window_llm"]["cost"],
        "sliding_calls": fr["sliding_window_llm"]["cost"],
        "allpairs_calls": fr["allpairs_llm"]["cost"],
        "distilled_inference_calls": fr["distilled_student"]["cost"],
        "ndcg_spread": max(fr[m]["ndcg"] for m in fr) - min(fr[m]["ndcg"] for m in fr),
    }


# =========================================================================== #
# Verification — every headline is an assertion (the harness owns the numbers).
# =========================================================================== #

def test_tau_zero_recovers_optimal_permutation() -> None:
    c = _corpus()
    for q in c["train_q"][:4]:
        docs = candidate_set(c, q)
        star = optimal_permutation(c, q, docs)
        for sd in range(3):
            emitted = noisy_oracle_perm(c, q, docs, TAU_COLLAPSE, np.random.default_rng([SEED, sd]))
            assert emitted == star, f"tau->0 must reproduce pi* exactly (q={q}, seed={sd})"


def test_abilities_are_order_faithful() -> None:
    # The -rank construction is order-faithful by design: argsort(-abilities) == pi* for every query, so
    # the tau -> 0 PL sample reproduces optimal_permutation regardless of the grade/oracle relationship.
    c = _corpus()
    for q in c["train_q"][:5]:
        docs = candidate_set(c, q)
        ab = oracle_abilities(c, q, docs)
        by_ability = [docs[i] for i in np.argsort(-ab, kind="stable")]
        assert by_ability == optimal_permutation(c, q, docs), "abilities must order-faithfully encode pi*"


def test_emitted_nll_is_listmle() -> None:
    c = _corpus()
    q = c["train_q"][0]
    docs = candidate_set(c, q)
    rng = np.random.default_rng([SEED, 1])
    perm_local = pl_sample(oracle_abilities(c, q, docs), TAU_NOISY, rng)
    twin = listmle_loss_scores(oracle_abilities(c, q, docs) / TAU_NOISY, perm_local)
    assert abs(emitted_nll(c, q, docs, perm_local, TAU_NOISY) - twin) < 1e-12


def test_pl_sampler_seeded_and_uniform_limit() -> None:
    c = _corpus()
    q = c["train_q"][0]
    docs = candidate_set(c, q)
    ab = oracle_abilities(c, q, docs)
    a = pl_sample(ab, TAU_NOISY, np.random.default_rng([SEED, 42]))
    b = pl_sample(ab, TAU_NOISY, np.random.default_rng([SEED, 42]))
    assert a == b, "same (tau, seed) must reproduce the permutation"
    # tau -> inf flattens the first-position marginal toward uniform (1/m each).
    m = len(docs)
    fuzzy = first_position_marginal(ab, 500.0, 4000, np.random.default_rng([SEED, 43]))
    sharp = first_position_marginal(ab, 0.05, 4000, np.random.default_rng([SEED, 44]))
    assert fuzzy.max() < 2.5 / m, "high tau must flatten the first-position marginal toward 1/m"
    assert sharp.max() > 0.95, "low tau must concentrate the first-position marginal on the best"


def test_call_count_law() -> None:
    # The exact sliding-window cost law, and the reranker issues exactly that many window calls.
    assert oracle_call_count(60, 20, 10, 4) == 4 * (math.ceil((60 - 20) / 10) + 1) == 20
    assert len(window_starts(60, 20, 10)) == 5
    assert oracle_call_count(POOL, WINDOW, STEP, N_PASSES) < POOL * (POOL - 1) // 2  # cheaper than all-pairs


def test_perfect_pass_bubbles_best() -> None:
    c = _corpus()
    assert all(perfect_pass_bubbles_best(c, q) for q in c["train_q"][:6]), \
        "tau->0 one back-to-front pass must bubble the global best into the top window"


def test_passes_monotone_recall() -> None:
    c = _corpus()
    curve = topk_recall_vs_passes(c, c["test_q"], TAU_NOISY, N_PASSES)
    rec = [v for _, v in curve]
    for i in range(1, len(rec)):
        assert rec[i] >= rec[i - 1] - 1e-9, f"seed-averaged recall must be monotone in passes: {rec}"
    assert rec[N_PASSES] - rec[1] > 0.01, f"more passes must help (pass-1 < pass-{N_PASSES}): {rec}"


def test_direction_asymmetry() -> None:
    c = _corpus()
    d = direction_asymmetry(c, c["test_q"], TAU_NOISY)
    assert d["back_to_front"] >= d["front_to_back"] - 1e-9, \
        f"back-to-front must not lose to front-to-back at one pass: {d}"


def test_positional_bias_bites_and_corrects() -> None:
    c = _corpus()
    assert gold_passes_through_center(c, c["test_q"]) >= 1, "the dip must be able to act on a gold doc"
    none = bias_correction_recall(c, c["test_q"], "none")
    biased = bias_correction_recall(c, c["test_q"], "biased")
    corrected = bias_correction_recall(c, c["test_q"], "corrected")
    assert none - biased > 0.02, f"the positional dip must BITE (none {none} > biased {biased})"
    assert corrected > biased + 1e-9, f"correction must help (corrected {corrected} > biased {biased})"
    assert corrected <= none + 1e-9, f"correction must not beat the unbiased oracle (<= none {none})"


def test_aggregation_k1_collapse() -> None:
    c = _corpus()
    q = c["train_q"][0]
    docs = candidate_set(c, q)
    perm = noisy_oracle_perm(c, q, docs, TAU_NOISY, np.random.default_rng([SEED, 70]))
    assert aggregate([perm], docs, "borda") == perm, "Borda at K=1 must be the single permutation"
    assert aggregate([perm], docs, "rrf") == perm, "RRF at K=1 must be the single permutation"


def test_mc4_stationary_is_distribution() -> None:
    c = _corpus()
    q = c["train_q"][0]
    docs = candidate_set(c, q)
    perms = sample_k_perms(c, q, docs, N_AGG, TAU_NOISY, np.random.default_rng([SEED, 71]))
    cons = mc4_aggregate(perms, docs)
    assert sorted(cons) == sorted(docs), "MC4 must return a permutation of the candidate set"


def test_aggregation_reduces_tau() -> None:
    c = _corpus()
    for method in ("borda", "rrf", "mc4"):
        curve = agg_tau_vs_k(c, c["test_q"], candidate_set, TAU_NOISY, method)
        taus = [t for _, t in curve]
        assert taus[0] > 0.5, f"K=1 must leave error to reduce ({method}): {taus[0]}"
        assert taus[-1] < taus[0] - 1e-9, f"aggregation must reduce tau-to-truth ({method}): {taus}"


def test_aggregated_rank_concentration() -> None:
    # The clean 1/sqrt(K) CLT law: std(mean rank) * sqrt(K) is ~constant across K.
    c = _corpus()
    rows = aggregated_rank_concentration(c, c["test_q"], candidate_set, TAU_NOISY)
    consts = [csqrt for _, _, csqrt in rows if csqrt > 0]
    assert max(consts) / min(consts) < 2.5, f"std*sqrt(K) should be ~constant (CLT): {rows}"
    stds = [sd for _, sd, _ in rows]
    assert stds[-1] < stds[0] + 1e-9, f"the aggregated-rank std must not grow with K: {stds}"


def test_mc4_approximates_kemeny() -> None:
    c = _corpus()
    ratios = kemeny_approximation(c, c["train_q"][0])
    for method in ("borda", "rrf", "mc4"):
        assert ratios[method] <= 1.6, f"{method} Kemeny ratio should be near-optimal: {ratios}"


def test_perfect_teacher_student_equals_fit_listmle() -> None:
    c = _corpus()
    # Guard: at tau -> 0 every teacher target equals pi* (so the equality is non-vacuous).
    tgt = teacher_targets(c, c["train_q"], TAU_COLLAPSE, np.random.default_rng([SEED, 12]))
    for q in c["train_q"]:
        docs = candidate_set(c, q)
        pos = {d: i for i, d in enumerate(docs)}
        star_local = [pos[d] for d in optimal_permutation(c, q, docs)]
        assert tgt[q] == star_local, "tau->0 teacher target must equal pi*"
    student = fit_student_to_teacher(c, c["train_q"], TAU_COLLAPSE, np.random.default_rng([SEED, 13]))
    reference = fit_listmle(c, c["train_q"])
    assert np.allclose(student, reference, atol=1e-9), \
        "a perfect teacher's distilled student must equal the imported fit_listmle"


def test_distillation_cost_payoff() -> None:
    c = _corpus()
    nq = len(c["test_q"])
    assert student_call_cost(nq) < teacher_call_cost(nq), "the student must be cheaper at inference"
    # The query-time speedup is the exact ratio O(n/s)*C_LLM : C_RETRIEVE = (24*250)/1 = 6000x.
    assert distill_speedup() == 6000.0, "the windowed teacher costs 6000x the precomputed student per query"
    # The noisy teacher's student still beats the best single leg (it tracks teacher quality).
    student = fit_student_to_teacher(c, c["train_q"], TAU_NOISY, np.random.default_rng([SEED, 14]))
    leg_recall = max(mean_recall_over(c, c["test_q"],
                                      _leg_weight(leg)) for leg in range(3))
    assert mean_recall_over(c, c["test_q"], student) >= leg_recall - 0.05, \
        "the distilled student should be competitive with the best single leg"


def _leg_weight(leg: int) -> np.ndarray:
    w = np.zeros(3)
    w[leg] = 1.0
    return w


def test_frontier_spread_and_structure() -> None:
    c = _corpus()
    fr = frontier(c, c["test_q"])
    v = verdict(fr)
    assert v["ndcg_spread"] > 0.01, f"the frontier must have quality spread: {v}"
    assert fr["sliding_window_llm"]["cost"] < fr["allpairs_llm"]["cost"], "sliding must be cheaper than all-pairs"
    assert fr["distilled_student"]["cost"] == 0.0, "the student answers at 0 inference LLM calls"
    assert fr["pointwise_llm"]["cost"] == float(POOL), "pointwise is one call per document"


def _run_all() -> None:
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")


# =========================================================================== #
# viz_constants — every value the .tsx bakes (reproducible only; numpy scalars cast clean).
# =========================================================================== #

def _r(v, n=4):
    return round(float(v), n)


def viz_constants() -> None:
    """Print every number the viz mirrors. Deterministic (seeded oracle, fixed seed-averaging)."""
    c = _corpus()
    te, tr = c["test_q"], c["train_q"]

    print("=== dims / params ===")
    print("POOL", POOL, "WINDOW", WINDOW, "STEP", STEP, "N_PASSES", N_PASSES, "TOPK", TOPK)
    print("TAU_NOISY", TAU_NOISY, "AGG_SEEDS", AGG_SEEDS, "C_RETRIEVE", C_RETRIEVE, "C_CE", C_CE, "C_LLM", C_LLM)

    print("=== Panel A: sliding-window bubble sort ===")
    print("CALL_COUNT", oracle_call_count(POOL, WINDOW, STEP, N_PASSES))
    print("ALLPAIRS_CALLS", POOL * (POOL - 1) // 2)
    print("N_WINDOWS_PER_PASS", len(window_starts(POOL, WINDOW, STEP)))
    print("RECALL_BY_PASS", [(p, _r(v)) for p, v in topk_recall_vs_passes(c, te, TAU_NOISY, N_PASSES)])
    da = direction_asymmetry(c, te, TAU_NOISY)
    print("DIRECTION back,front", _r(da["back_to_front"]), _r(da["front_to_back"]))
    print("PERFECT_BUBBLE", bool(all(perfect_pass_bubbles_best(c, q) for q in tr[:6])))

    print("=== Panel B: positional bias ===")
    print("BIAS_KERNEL", [_r(v) for v in position_bias_kernel(WINDOW)])
    print("POS_DIP", POS_DIP, "TAU_BIAS", TAU_BIAS, "POS_BIAS_GAIN", POS_BIAS_GAIN, "N_PRESENT", N_PRESENT)
    print("RECALL_NONE", _r(bias_correction_recall(c, te, "none")))
    print("RECALL_BIASED", _r(bias_correction_recall(c, te, "biased")))
    print("RECALL_CORRECTED", _r(bias_correction_recall(c, te, "corrected")))
    print("GOLD_AT_CENTER", gold_passes_through_center(c, te))

    print("=== Panel C: rank aggregation ===")
    for method in ("borda", "rrf", "mc4"):
        print(f"TAU_VS_K_{method}", [(K, _r(t, 3)) for K, t in agg_tau_vs_k(c, te, candidate_set, TAU_NOISY, method)])
    print("RANK_CONCENTRATION", [(K, _r(sd, 3), _r(cs, 3))
                                 for K, sd, cs in aggregated_rank_concentration(c, te, candidate_set, TAU_NOISY)])
    print("KEMENY_RATIOS", {k: _r(v, 3) for k, v in kemeny_approximation(c, tr[0]).items()})
    # one worked query's K noisy perms + pi* for the panel strip
    q0 = te[0]
    docs0 = candidate_set(c, q0)
    rng0 = np.random.default_rng([SEED, 20])
    print("WORKED_Q", q0, "WORKED_DOCS", docs0)
    print("WORKED_STAR", optimal_permutation(c, q0, docs0))
    print("WORKED_PERMS", sample_k_perms(c, q0, docs0, 8, TAU_NOISY, rng0))

    print("=== Panel D: distillation + cost-quality frontier ===")
    nq = len(te)
    print("TEACHER_COST", _r(teacher_call_cost(nq), 1), "STUDENT_COST", _r(student_call_cost(nq), 1),
          "SPEEDUP", _r(distill_speedup(), 1))
    student_perfect = fit_student_to_teacher(c, tr, TAU_COLLAPSE, np.random.default_rng([SEED, 13]))
    student_noisy = fit_student_to_teacher(c, tr, TAU_NOISY, np.random.default_rng([SEED, 14]))
    print("STUDENT_PERFECT_EQ_FITLISTMLE", bool(np.allclose(student_perfect, fit_listmle(c, tr), atol=1e-9)))
    print("STUDENT_RECALL_PERFECT", _r(mean_recall_over(c, te, student_perfect)))
    print("STUDENT_RECALL_NOISY", _r(mean_recall_over(c, te, student_noisy)))
    fr = frontier(c, te)
    for name in ("dense_best_leg", "rrf_3legs", "pointwise_llm", "sliding_window_llm",
                 "allpairs_llm", "distilled_student"):
        m = fr[name]
        print(f"  {name:20s} ndcg={_r(m['ndcg'])} recall={_r(m['recall'])} cost={int(m['cost'])} "
              f"ci=[{_r(m['ci']['ci_lo'])},{_r(m['ci']['ci_hi'])}]")
    print("VERDICT", {k: (_r(v, 2) if isinstance(v, float) else v) for k, v in verdict(fr).items()})


if __name__ == "__main__":
    _run_all()
    print()
    viz_constants()
