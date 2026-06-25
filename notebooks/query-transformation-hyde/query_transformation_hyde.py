"""Query Transformation and HyDE — the reference implementation for the formalRAG
`query-transformation-hyde` topic.

A dual encoder scores a query and a document by the cosine of their embeddings, but a query
is *question-shaped* and a document is *answer-shaped*: the two distributions do not coincide
on the sphere. A bare query sits OFF the document manifold, so its nearest documents are
mediocre. HyDE (Gao, Ma, Lin & Callan, ACL 2023) fixes this without any relevance labels:
it asks an LLM to write a HYPOTHETICAL answer document, embeds THAT, and retrieves real
documents near it — landing back inside the document manifold, closer to the relevant doc
than the query ever was. This module owns every number the topic depends on; the harness
asserts each pedagogical claim.

  MOVEMENT 1 — THE QUERY-DOCUMENT GAP. Reusing the dense-retrieval finance geometry's DOCUMENT
    manifold (one vMF company prototype per passage), we build TOPIC-SPECIFIC off-manifold
    queries: a query for company c is tilted off its answer direction toward a shared
    "generic document-ness" axis g (the corpus centroid) by a distribution-shift angle theta,
    q_c(theta) = normalize(cos(theta) u_c + sin(theta) g). As theta grows the bare query loses
    company specificity and its recall@1 collapses. (The dense topic's own queries are drawn
    kappa=350-tight ON their company, so recall@1 there is already 1.0 — the "too-easy corpus"
    trap; HyDE needs a genuine gap, so we draw our own queries, the tuned-query exception.)

  MOVEMENT 2 — HyDE AS A MONTE-CARLO ESTIMATOR (bias vs variance). A hypothetical document is a
    vMF draw near the answer prototype; averaging k of them, hhat_k = normalize(mean(h_1..h_k)),
    is a Monte-Carlo estimate of the generation distribution's center. At a FAITHFUL generator
    (center = the true answer u_c) recall RISES and the estimate's angular variance FALLS ~ 1/k
    as k grows — and HyDE recovers the answer regardless of theta (it ignores the off-manifold
    query position and synthesizes an on-manifold proxy). But when the generator HALLUCINATES on
    a fraction p of queries (its center tilts to a wrong company), hhat_k is consistent for the
    WRONG center: recall plateaus at ceiling 1 - p no matter how many samples you average. HyDE
    trades query-document mismatch for GENERATION BIAS — the load-bearing rigorFlag.

  MOVEMENT 3 — HyDE IS THE NEURAL GENERALIZATION OF PSEUDO-RELEVANCE FEEDBACK. Rocchio/RM3
    expand a query by moving it toward the centroid of pseudo-relevant documents, q' = a q + b
    centroid; HyDE lifts the SAME update into embedding space with the centroid taken over
    GENERATED hypotheticals instead of RETRIEVED documents. We import the prereq's term-space
    Rocchio/RM3 and reproduce its improve-then-drift curve (the ancestor HyDE generalizes), then
    show that on an off-manifold query HyDE's generated centroid beats real pseudo-relevance
    feedback — whose centroid is polluted because the bad query retrieves bad feedback.

Honest caveats (rigorFlag territory): the hypothetical "generator" is a synthetic vMF model,
not a trained LLM; the off-manifold query geometry and the corpus-centroid offset axis are
demonstrative; hallucination bias is irreducible by averaging but its rate p is a modeling
choice; and "HyDE beats pseudo-relevance feedback" is shown on ONE synthetic geometry, not as
a universal ranking. Collapse anchors pin the construction: a PERFECT (kappa->inf, faithful)
single hypothetical IS the gold document, so its retrieval is byte-for-byte the gold doc's own;
the embedding-space HyDE update at alpha=0 is the bare query; and the imported term-space
Rocchio/RM3 reproduces the prereq exactly.

This module imports its prerequisites (`hypersphere-vmf-geometry`, `dense-retrieval-dual-encoders`,
`pseudo-relevance-feedback`, and the dense topic's own ancestors) for the sphere sampler, the
document manifold + separable score, and the term-space relevance-feedback machinery — it never
reimplements them. `viz_constants()` prints what `QueryTransformationHydeLaboratory.tsx` mirrors
to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/query-transformation-hyde/query_transformation_hyde.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# Established cross-topic pattern: add EACH ancestor's HYPHENATED dir to the path, import the
# UNDERSCORED module. The dense topic pulls in the-retrieval-problem + infonce transitively, so
# we add every ancestor explicitly and never reimplement any of them.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in ("hypersphere-vmf-geometry", "the-retrieval-problem", "infonce-contrastive-objective",
             "dense-retrieval-dual-encoders", "pseudo-relevance-feedback"):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf                       # noqa: E402
from dense_retrieval_dual_encoders import (                                      # noqa: E402
    dpr_finance_matrix, dual_encoder_score, DPR_SEED,
)
from pseudo_relevance_feedback import (                                          # noqa: E402
    Index, _CORPUS, rm3_rank, rocchio_rank, recall_at_k as prf_recall_at_k,
)


# --------------------------------------------------------------------------- #
# Module constants — tuned by a build-and-run diagnostics sweep, never guessed.
# All vMF draws are seeded so every baked number is reproducible.
# --------------------------------------------------------------------------- #

HYDE_SEED = DPR_SEED                       # 7 — the shared finance-geometry seed
THETA_OP_DEG = 75.0                        # operating distribution-shift angle (bare recall ~ 0.4)
SHIFT_GRID_DEG = (0.0, 30.0, 45.0, 60.0, 70.0, 75.0, 80.0, 85.0)   # Panel A theta sweep
KAPPA_QUERY = 200.0                        # bare-query vMF spread around its (shifted) center
QPC = 8                                    # bare queries per company (for smooth recall curves)

KAPPA_HYP = 12.0                           # hypothetical-doc vMF spread (loose: averaging denoises)
K_GRID = (1, 2, 3, 5, 8, 12, 20)           # Panel B: number of hypotheticals averaged
HALLU_GRID = (0.0, 0.25, 0.5)              # Panel B: generator hallucination RATE p
PHI_HALLU_DEG = 60.0                       # severity of a hallucination (past the 45-deg flip)
MC_TRIALS = 60                             # Monte-Carlo trials for the recall curves
VAR_TRIALS = 300                           # trials for the estimator-variance (1/k) law
K_INF = 60                                 # "k -> infinity" stand-in for the bias-floor asserts

ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)   # Panel C: HyDE/Rocchio interpolation weight
PRF_FEEDBACK_M = 3                         # real-doc pseudo-relevance feedback set size
ALPHA_TRIALS = 16                          # trials for the recall-vs-alpha curves


# --------------------------------------------------------------------------- #
# The document manifold (reused) + the off-manifold query construction (new).
# --------------------------------------------------------------------------- #

def document_manifold(seed: int = HYDE_SEED):
    """The reused dense-retrieval finance DOCUMENT manifold: P is one vMF company prototype per
    passage (n_docs = sectors x companies), sector_of_passage colors the block structure. We use
    P and the sectors only — the dense topic's own (too-easy) queries are discarded; this topic
    builds its own off-manifold queries. Returns (P, sector)."""
    _, P, _, sector = dpr_finance_matrix(seed=seed)
    return P, sector


def query_offset_axis(P: np.ndarray) -> np.ndarray:
    """The shared 'generic document-ness' direction g: the unit corpus centroid. A bare query is
    pulled toward g as the distribution shift grows, which is what costs it company specificity
    (every query drifts toward the same generic direction, so the document nearest g starts to
    win regardless of the query's company)."""
    return normalize(P.mean(axis=0))


def bare_query_center(c: int, P: np.ndarray, g: np.ndarray, theta: float) -> np.ndarray:
    """The center of company c's bare-query distribution at distribution-shift angle theta:
    q_c(theta) = normalize(cos(theta) u_c + sin(theta) g), a great-circle interpolation from the
    answer direction u_c (theta = 0, on the manifold) toward the generic offset axis g."""
    return normalize(np.cos(theta) * P[c] + np.sin(theta) * g)


def bare_queries(c: int, P: np.ndarray, g: np.ndarray, theta: float,
                 qpc: int = QPC, kappa_q: float = KAPPA_QUERY,
                 seed: int = HYDE_SEED) -> np.ndarray:
    """qpc seeded vMF draws of company c's bare query around its shifted center."""
    qs = sample_vmf(qpc, bare_query_center(c, P, g, theta), kappa_q, seed=seed + 7 * c + 100)
    return normalize(np.atleast_2d(qs))


def bare_recall_at_theta(P: np.ndarray, g: np.ndarray, theta: float,
                         qpc: int = QPC, kappa_q: float = KAPPA_QUERY,
                         seed: int = HYDE_SEED) -> float:
    """Recall@1 of the BARE query (no transformation) at distribution-shift angle theta, averaged
    over every company's qpc query draws. Falls as theta grows."""
    n_docs = P.shape[0]
    hits = tot = 0
    for c in range(n_docs):
        for q in bare_queries(c, P, g, theta, qpc, kappa_q, seed):
            if int(np.argmax(dual_encoder_score(q, P))) == c:
                hits += 1
            tot += 1
    return hits / tot if tot else 0.0


# --------------------------------------------------------------------------- #
# The HyDE hypothetical + the Monte-Carlo estimator (bias vs variance).
# --------------------------------------------------------------------------- #

def hallucination_targets(P: np.ndarray) -> list[int]:
    """For each company, the nearest OTHER document — the wrong company a hallucinated generator
    drifts toward (same-sector confusion is the realistic, hardest case)."""
    Cd = P @ P.T
    n_docs = P.shape[0]
    return [int(next(j for j in np.argsort(-Cd[c]) if j != c)) for c in range(n_docs)]


def generation_center(c: int, P: np.ndarray, targets: list[int], hallucinated: bool,
                      phi: float = np.radians(PHI_HALLU_DEG)) -> np.ndarray:
    """The center of the generator's hypothetical-document distribution for company c. Faithful:
    the true answer direction u_c. Hallucinated: tilted toward the wrong company by phi, far
    enough past the 45-degree flip boundary that the biased center retrieves the WRONG document."""
    if not hallucinated:
        return P[c].copy()
    return normalize(np.cos(phi) * P[c] + np.sin(phi) * P[targets[c]])


def hyde_centroid(center: np.ndarray, k: int, kappa_h: float = KAPPA_HYP,
                  seed: int = HYDE_SEED) -> np.ndarray:
    """hhat_k = normalize(mean of k vMF(center, kappa_h) hypothetical-document embeddings) — the
    Monte-Carlo estimate of the generation center. GUARD: k >= 1."""
    if k < 1:
        raise ValueError(f"need k >= 1 hypotheticals, got {k}")
    hs = np.atleast_2d(sample_vmf(k, center, kappa_h, seed=seed))
    return normalize(hs.mean(axis=0))


def hyde_recall(P: np.ndarray, targets: list[int], k: int, p: float,
                kappa_h: float = KAPPA_HYP, n_trials: int = MC_TRIALS,
                seed: int = HYDE_SEED) -> float:
    """Recall@1 of HyDE retrieval (retrieve with hhat_k) when the generator hallucinates on a
    fraction p of queries. Averaged over n_trials seeded draws; the per-trial hallucination mask
    is reproducible. HyDE ignores the bare query's position, so this does not depend on theta."""
    n_docs = P.shape[0]
    hits = tot = 0
    for t in range(n_trials):
        mask = np.random.default_rng(seed + 1000 + t).random(n_docs) < p
        for c in range(n_docs):
            center = generation_center(c, P, targets, bool(mask[c]))
            hhat = hyde_centroid(center, k, kappa_h, seed=seed + 5000 + 31 * c + t)
            if int(np.argmax(dual_encoder_score(hhat, P))) == c:
                hits += 1
            tot += 1
    return hits / tot if tot else 0.0


def estimator_deficit(c: int, P: np.ndarray, k: int, kappa_h: float = KAPPA_HYP,
                      n_trials: int = VAR_TRIALS, seed: int = HYDE_SEED) -> float:
    """The mean angular variance proxy 1 - <hhat_k, mu> of the FAITHFUL estimator (mu = u_c),
    averaged over n_trials. The classic Monte-Carlo law: it falls ~ 1/k."""
    mu = P[c]
    vals = [1.0 - float(hyde_centroid(mu, k, kappa_h, seed=seed + 9000 + t) @ mu)
            for t in range(n_trials)]
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# Movement 3 — HyDE as the neural generalization of pseudo-relevance feedback.
# --------------------------------------------------------------------------- #

def hyde_update(q: np.ndarray, centroid: np.ndarray, alpha: float) -> np.ndarray:
    """The Rocchio update lifted into embedding space: q'(alpha) = (1 - alpha) q + alpha centroid,
    then renormalized. alpha = 0 is the bare query; alpha = 1 is the pure pseudo-document (the
    canonical HyDE); intermediate alpha interpolates (HyDE + the original query)."""
    return normalize((1.0 - alpha) * np.asarray(q, dtype=float) + alpha * np.asarray(centroid, dtype=float))


def prf_real_centroid(q: np.ndarray, P: np.ndarray, m: int = PRF_FEEDBACK_M) -> np.ndarray:
    """The pseudo-relevance-feedback centroid in embedding space: the mean of the bare query's
    top-m RETRIEVED documents (Rocchio's feedback set). On an off-manifold query these are the
    wrong documents, so the centroid is polluted — the contrast HyDE's generated centroid beats.
    GUARD: m clamped to [1, n_docs]."""
    n_docs = P.shape[0]
    m = max(1, min(int(m), n_docs))
    idx = np.argsort(-dual_encoder_score(q, P))[:m]
    return normalize(P[idx].mean(axis=0))


def recall_vs_alpha(P: np.ndarray, g: np.ndarray, mode: str, theta: float,
                    k: int = 5, kappa_h: float = KAPPA_HYP, m: int = PRF_FEEDBACK_M,
                    qpc: int = QPC, n_trials: int = ALPHA_TRIALS,
                    seed: int = HYDE_SEED) -> list[float]:
    """For each alpha in ALPHA_GRID, recall@1 of the interpolated query q'(alpha). mode='hyde'
    uses the generated-hypothetical centroid (faithful generator); mode='prf' uses the real-doc
    pseudo-relevance centroid. Both start from the SAME off-manifold bare query at alpha=0."""
    n_docs = P.shape[0]
    out = []
    for alpha in ALPHA_GRID:
        hits = tot = 0
        for t in range(n_trials):
            for c in range(n_docs):
                q = normalize(sample_vmf(1, bare_query_center(c, P, g, theta), KAPPA_QUERY,
                                         seed=seed + 200 + 7 * c + t).ravel())
                if mode == "hyde":
                    centroid = hyde_centroid(P[c], k, kappa_h, seed=seed + 300 + 31 * c + t)
                else:
                    centroid = prf_real_centroid(q, P, m)
                qp = hyde_update(q, centroid, alpha)
                if int(np.argmax(dual_encoder_score(qp, P))) == c:
                    hits += 1
                tot += 1
        out.append(hits / tot if tot else 0.0)
    return out


def perfect_hypothetical_retrieval(c: int, P: np.ndarray) -> np.ndarray:
    """The collapse anchor: a PERFECT (kappa -> inf, faithful) single hypothetical IS the gold
    document direction P[c], so HyDE retrieval with it is byte-for-byte the gold document's own
    retrieval. Returns the score row dual_encoder_score(P[c], P)."""
    return dual_encoder_score(P[c], P)


def prf_ancestor_curve():
    """The IMPORTED term-space pseudo-relevance-feedback curve HyDE generalizes: RM3 and Rocchio
    recall@4 on the prereq's worked corpus across feedback-set sizes (improve, then drift). Pure
    reuse of the prereq's own functions — never reimplemented here. Returns a list of dicts."""
    index = Index(_CORPUS)
    return [{"n_fb": n, "rm3": round(prf_recall_at_k(rm3_rank(n, index)), 3),
             "rocchio": round(prf_recall_at_k(rocchio_rank(n, index)), 3)} for n in range(0, 6)]


# --------------------------------------------------------------------------- #
# Verification harness — each assert is a pedagogical claim the topic makes.
# --------------------------------------------------------------------------- #

def test_collapse_perfect_hypothetical() -> None:
    """Collapse anchor: a perfect (kappa->inf, faithful) single hypothetical is exactly the gold
    document, so its retrieval reproduces the gold document's own retrieval — top-1 is the gold
    and the full ranking is identical (byte-for-byte)."""
    P, _ = document_manifold()
    n_docs = P.shape[0]
    for c in range(n_docs):
        s_hyp = perfect_hypothetical_retrieval(c, P)        # h = P[c]
        s_gold = dual_encoder_score(P[c], P)                # retrieve with the gold doc itself
        assert int(np.argmax(s_hyp)) == c, f"perfect hypothetical top-1 != gold for company {c}"
        assert np.array_equal(np.argsort(-s_hyp), np.argsort(-s_gold)), "ranking not identical"
    print(f"  [ok] collapse: a perfect hypothetical IS the gold doc — top-1 is the answer, "
          f"ranking byte-for-byte the gold's own (all {n_docs} companies)")


def test_gap_exists() -> None:
    """The headline: at the operating distribution shift the bare query's nearest documents are
    mediocre (recall@1 well below 1), while HyDE — which retrieves with a generated hypothetical
    — recovers the answer almost always. Pinned to the observed run."""
    P, _ = document_manifold()
    g = query_offset_axis(P)
    targets = hallucination_targets(P)
    theta = np.radians(THETA_OP_DEG)
    bare = bare_recall_at_theta(P, g, theta)
    hyde = hyde_recall(P, targets, k=K_INF, p=0.0)
    assert bare < 0.6, f"bare recall should be clearly degraded at the operating shift, got {bare}"
    assert hyde > 0.95, f"HyDE recall should recover the answer, got {hyde}"
    assert hyde - bare > 0.4, f"the gap should be large, got bare={bare}, hyde={hyde}"
    print(f"  [ok] the query-document gap: bare recall@1 = {bare:.3f} << HyDE recall@1 = {hyde:.3f} "
          f"at theta = {THETA_OP_DEG:.0f} deg")


def test_bare_degrades_with_shift() -> None:
    """The bare query starts on the manifold (recall@1 = 1.0 at theta = 0) and its recall@1 falls
    monotonically as the distribution shift theta grows."""
    P, _ = document_manifold()
    g = query_offset_axis(P)
    curve = [bare_recall_at_theta(P, g, np.radians(d)) for d in SHIFT_GRID_DEG]
    assert curve[0] >= 0.999, f"bare recall@1 should be ~1.0 on the manifold, got {curve[0]}"
    assert all(curve[i] >= curve[i + 1] - 1e-9 for i in range(len(curve) - 1)), \
        f"bare recall should be non-increasing in theta: {curve}"
    assert curve[-1] < 0.5, f"bare recall should be badly degraded at large shift, got {curve[-1]}"
    print(f"  [ok] bare recall@1 degrades with distribution shift: "
          f"{[round(v, 3) for v in curve]} over theta = {SHIFT_GRID_DEG} deg")


def test_hyde_shift_robust() -> None:
    """HyDE ignores the bare query's off-manifold position and synthesizes an on-manifold proxy,
    so a faithful generator recovers the answer at EVERY distribution shift — HyDE recall@1 is
    ~1.0 across the whole theta range (it does not depend on theta)."""
    P, _ = document_manifold()
    targets = hallucination_targets(P)
    hyde = hyde_recall(P, targets, k=K_INF, p=0.0)
    assert hyde > 0.98, f"faithful HyDE should recover the answer regardless of shift, got {hyde}"
    print(f"  [ok] HyDE is shift-robust: a faithful generator recovers recall@1 = {hyde:.3f} "
          f"independent of theta")


def test_mc_recall_rises_with_k() -> None:
    """Monte-Carlo variance reduction: with a noisy faithful generator, a single hypothetical is
    unreliable, but averaging more hypotheticals denoises the estimate and recall@1 RISES with k
    toward 1.0."""
    P, _ = document_manifold()
    targets = hallucination_targets(P)
    curve = [hyde_recall(P, targets, k, p=0.0) for k in K_GRID]
    assert all(curve[i] <= curve[i + 1] + 0.02 for i in range(len(curve) - 1)), \
        f"recall@1 should rise (non-decreasing within noise) in k: {curve}"
    assert curve[-1] - curve[0] > 0.25, f"averaging should lift recall substantially: {curve}"
    assert curve[-1] > 0.98, f"large-k recall should reach ~1.0, got {curve[-1]}"
    print(f"  [ok] MC variance reduction: faithful recall@1 rises with k "
          f"{[round(v, 3) for v in curve]} (k = {K_GRID})")


def test_mc_variance_falls_one_over_k() -> None:
    """The estimator's angular variance 1 - <hhat_k, mu> falls monotonically in k and approaches
    the Monte-Carlo 1/k rate ASYMPTOTICALLY: each doubling of k drives the deficit-halving ratio
    toward 1/2, and the ratio accelerates toward it as k grows (the small-k regime is pre-
    asymptotic at this loose generator concentration)."""
    P, _ = document_manifold()
    defs = [estimator_deficit(0, P, k) for k in K_GRID]
    assert all(defs[i] > defs[i + 1] for i in range(len(defs) - 1)), \
        f"estimator deficit should strictly fall in k: {defs}"
    assert defs[-1] < defs[0] / 3.0, f"averaging should sharply reduce variance: {defs}"
    # The 1/k law in the tail: doubling k roughly halves the deficit, and the ratio accelerates
    # toward 1/2 as k grows.
    d8, d16, d32 = (estimator_deficit(0, P, k) for k in (8, 16, 32))
    r1, r2 = d16 / d8, d32 / d16
    assert r1 < 0.66 and r2 < r1, f"deficit doubling-ratio should approach 1/2: {r1:.3f} -> {r2:.3f}"
    print(f"  [ok] MC variance falls toward the 1/k rate: deficit 1-<hhat_k,mu> = "
          f"{[round(v, 4) for v in defs]}; doubling ratio {r1:.3f} -> {r2:.3f} (-> 0.5)")


def test_consistency_faithful() -> None:
    """A faithful generator's averaged hypothetical is a CONSISTENT estimator of the true answer:
    cos(hhat_k, u_c) -> 1 as k grows (the deficit at large k is small)."""
    P, _ = document_manifold()
    d1 = estimator_deficit(0, P, 1)
    d_inf = estimator_deficit(0, P, K_INF)
    assert d_inf < 0.1, f"faithful estimator should converge toward the truth, deficit={d_inf}"
    assert d_inf < d1 / 5.0, f"deficit should shrink sharply from k=1 ({d1}) to k={K_INF} ({d_inf})"
    print(f"  [ok] consistency: faithful cos(hhat_k, u_c) -> 1 "
          f"(deficit {d1:.3f} at k=1 -> {d_inf:.4f} at k={K_INF})")


def test_hallucination_bias_floor() -> None:
    """The honest limit: when the generator hallucinates on a fraction p of queries, HyDE recall
    plateaus at a CEILING below 1 — ordered 1.0 (p=0) > p=0.25 > p=0.5, with each ceiling near
    1 - p. Averaging cannot lift recall past the bias."""
    P, _ = document_manifold()
    targets = hallucination_targets(P)
    ceil = [hyde_recall(P, targets, k=K_INF, p=p) for p in HALLU_GRID]
    assert ceil[0] > 0.98, f"p=0 ceiling should be ~1.0, got {ceil[0]}"
    assert all(ceil[i] > ceil[i + 1] + 0.1 for i in range(len(ceil) - 1)), \
        f"recall ceiling should fall with the hallucination rate: {ceil}"
    for p, c in zip(HALLU_GRID, ceil):
        assert c < 1.0 - p + 0.08, f"ceiling at p={p} should sit near 1-p={1 - p}, got {c}"
    print(f"  [ok] hallucination bias floor: recall ceiling {[round(v, 3) for v in ceil]} "
          f"at rates p = {HALLU_GRID} (each near 1 - p)")


def test_bias_is_irreducible() -> None:
    """The mechanism behind the floor: a hallucinated query's averaged hypothetical is consistent
    for the WRONG center — cos(hhat_k, generation_center) -> 1 while cos(hhat_k, u_c) stays below
    1 — and it retrieves the wrong document no matter how large k is. Averaging cannot fix bias."""
    P, _ = document_manifold()
    targets = hallucination_targets(P)
    c = 0
    center = generation_center(c, P, targets, hallucinated=True)
    hhat = hyde_centroid(center, K_INF, seed=HYDE_SEED + 1)
    cos_center = float(hhat @ center)
    cos_truth = float(hhat @ P[c])
    assert cos_center > 0.93, f"estimator should be consistent for the (biased) center, {cos_center}"
    assert cos_truth < cos_center - 0.1, f"the biased estimate should stay off the truth: {cos_truth}"
    assert int(np.argmax(dual_encoder_score(hhat, P))) == targets[c], \
        "a hallucinated query should retrieve the wrong document even at large k"
    print(f"  [ok] bias is irreducible: hallucinated hhat_k -> wrong center "
          f"(cos to center {cos_center:.3f} > cos to truth {cos_truth:.3f}); retrieves doc {targets[c]} != {c}")


def test_bias_beats_variance() -> None:
    """The bias-variance contrast made one statement: increasing k closes the VARIANCE gap at
    p=0 (large gain) but cannot close the BIAS gap at p>0 (the large-k recall stays far below 1)."""
    P, _ = document_manifold()
    targets = hallucination_targets(P)
    gain_p0 = hyde_recall(P, targets, k=K_GRID[-1], p=0.0) - hyde_recall(P, targets, k=1, p=0.0)
    short_p5 = 1.0 - hyde_recall(P, targets, k=K_INF, p=0.5)
    assert gain_p0 > 0.25, f"averaging should help a lot at p=0, gain={gain_p0}"
    assert short_p5 > 0.35, f"averaging cannot fix bias at p=0.5, shortfall={short_p5}"
    print(f"  [ok] bias beats variance: k lifts faithful recall by {gain_p0:.3f}, "
          f"but leaves a {short_p5:.3f} bias shortfall at p=0.5")


def test_hyde_update_alpha0_is_bare() -> None:
    """Degenerate-parameter anchor: the embedding-space HyDE/Rocchio update at alpha=0 is exactly
    the (renormalized) bare query."""
    P, _ = document_manifold()
    g = query_offset_axis(P)
    q = bare_query_center(0, P, g, np.radians(THETA_OP_DEG))
    centroid = hyde_centroid(P[0], 5)
    assert np.allclose(hyde_update(q, centroid, 0.0), normalize(q), atol=1e-12), \
        "alpha=0 should recover the bare query"
    print("  [ok] collapse: HyDE update at alpha=0 is the bare query (< 1e-12)")


def test_hyde_beats_prf_real() -> None:
    """Movement 3 headline (pinned to the observed run): on an off-manifold query, HyDE's
    GENERATED centroid lifts recall toward 1.0 as alpha grows, while real pseudo-relevance
    feedback's RETRIEVED centroid is polluted and actually HURTS — so HyDE dominates at every
    alpha > 0 and especially at alpha=1 (the pure pseudo-document)."""
    P, _ = document_manifold()
    g = query_offset_axis(P)
    theta = np.radians(THETA_OP_DEG)
    hyde = recall_vs_alpha(P, g, "hyde", theta)
    prf = recall_vs_alpha(P, g, "prf", theta)
    assert abs(hyde[0] - prf[0]) < 1e-9, "both must start from the same bare query at alpha=0"
    assert hyde[-1] > 0.95, f"pure HyDE (alpha=1) should recover the answer, got {hyde[-1]}"
    assert prf[-1] < hyde[-1] - 0.4, f"real PRF should trail HyDE badly, got {prf[-1]} vs {hyde[-1]}"
    assert prf[-1] <= prf[0] + 1e-9, f"polluted PRF feedback should not help here: {prf}"
    print(f"  [ok] HyDE beats real PRF: recall vs alpha HyDE {[round(v, 3) for v in hyde]} "
          f"vs PRF {[round(v, 3) for v in prf]}")


def test_prf_ancestor_imported() -> None:
    """The reuse anchor: HyDE generalizes pseudo-relevance feedback, and we IMPORT the prereq's
    own term-space RM3/Rocchio to reproduce its improve-then-drift curve — a little feedback
    bridges the vocabulary mismatch (recall@4 0.5 -> 1.0), too much over-expands and drifts back
    (<= 0.5). These are the prereq's functions, never reimplemented here."""
    curve = prf_ancestor_curve()
    base = next(r for r in curve if r["n_fb"] == 0)["rm3"]
    fed = next(r for r in curve if r["n_fb"] == 2)["rm3"]
    over = next(r for r in curve if r["n_fb"] == 4)["rm3"]
    assert base == 0.5 and fed == 1.0 and over <= 0.5, f"imported PRF curve changed: {curve}"
    print(f"  [ok] PRF ancestor (imported): RM3 recall@4 improves 0.5 -> 1.0 then drifts to {over} "
          f"— the term-space curve HyDE generalizes")


def test_rocchio_form_parallel() -> None:
    """The structural identity documenting the reuse: the embedding-space HyDE update is exactly
    the Rocchio combination a*q + b*centroid (renormalized), the same algebraic form the imported
    term-space rocchio_query uses — only the centroid's source (generated vs retrieved) differs."""
    P, _ = document_manifold()
    g = query_offset_axis(P)
    q = bare_query_center(0, P, g, np.radians(THETA_OP_DEG))
    centroid = hyde_centroid(P[0], 5)
    alpha = 0.4
    generic = normalize((1.0 - alpha) * q + alpha * centroid)   # a=1-alpha, b=alpha Rocchio form
    assert np.allclose(hyde_update(q, centroid, alpha), generic, atol=1e-12), \
        "HyDE update must equal the generic Rocchio form"
    print("  [ok] structural parallel: HyDE update == Rocchio a*q + b*centroid (< 1e-12), "
          "generated centroid in place of retrieved")


def test_guards() -> None:
    """Defensive guards (the gemini-prone cases): k < 1 hypotheticals raise; the PRF feedback
    size is clamped to the corpus; recall denominators are never empty in the harness."""
    P, _ = document_manifold()
    try:
        hyde_centroid(P[0], 0)
        assert False, "k=0 should raise"
    except ValueError:
        pass
    cen = prf_real_centroid(P[0], P, m=999)                 # clamped to n_docs, no crash
    assert cen.shape == (P.shape[1],) and abs(np.linalg.norm(cen) - 1.0) < 1e-9
    print("  [ok] guards: k<1 raises; PRF feedback size clamps to the corpus")


def test_viz_constants_reproducible() -> None:
    """Bake-only-reproducible: every viz number is a seeded vMF computation, so two runs of the
    Panel B recall curve are bit-identical (no random start vector leaks into a baked number)."""
    P, _ = document_manifold()
    targets = hallucination_targets(P)
    a = [hyde_recall(P, targets, k, p=0.25) for k in K_GRID]
    b = [hyde_recall(P, targets, k, p=0.25) for k in K_GRID]
    assert a == b, f"baked Panel B curve is not reproducible: {a} vs {b}"
    print("  [ok] reproducible: seeded HyDE recall curves are bit-identical across runs")


# --------------------------------------------------------------------------- #
# Demo — the headline numbers, printed.
# --------------------------------------------------------------------------- #

def hyde_demo() -> dict:
    """The headline: a bare query sits off the document manifold and retrieves mediocre documents;
    HyDE's generated hypothetical lands back inside the manifold and recovers the answer; averaging
    reduces variance but not the generator's bias; and HyDE is pseudo-relevance feedback with a
    generated rather than a retrieved centroid."""
    P, sector = document_manifold()
    g = query_offset_axis(P)
    targets = hallucination_targets(P)
    theta = np.radians(THETA_OP_DEG)
    bare = bare_recall_at_theta(P, g, theta)
    hyde = hyde_recall(P, targets, k=K_INF, p=0.0)
    print(f"  finance geometry: {P.shape[0]} documents (one vMF company prototype each), dim = {P.shape[1]}, "
          f"sectors = {sector.tolist()}")
    print(f"  MOVEMENT 1: at theta = {THETA_OP_DEG:.0f} deg the bare query is off-manifold — "
          f"recall@1 = {bare:.3f}; HyDE recovers recall@1 = {hyde:.3f}")
    ceil = [hyde_recall(P, targets, k=K_INF, p=p) for p in HALLU_GRID]
    print(f"  MOVEMENT 2: bias floor — HyDE recall ceiling {[round(v, 3) for v in ceil]} "
          f"at hallucination rates {HALLU_GRID} (averaging cannot break it)")
    hyde_a = recall_vs_alpha(P, g, "hyde", theta)
    prf_a = recall_vs_alpha(P, g, "prf", theta)
    print(f"  MOVEMENT 3: HyDE recall vs alpha {[round(v, 3) for v in hyde_a]} dominates real "
          f"pseudo-relevance feedback {[round(v, 3) for v in prf_a]}")
    return {"bare": round(bare, 3), "hyde": round(hyde, 3), "ceiling": [round(v, 3) for v in ceil]}


# --------------------------------------------------------------------------- #
# Viz constants — printed for QueryTransformationHydeLaboratory.tsx to mirror.
# --------------------------------------------------------------------------- #

def _r3(v) -> float:
    return round(float(v), 3)


def _panel_a_geometry(P: np.ndarray, g: np.ndarray, sector: np.ndarray, worked: int = 0):
    """A 2-D PCA projection of the doc manifold + one worked company's bare query and a few of its
    hypotheticals, so the lab can DRAW the gap (query off-manifold, hypothetical inside it). Purely
    illustrative coordinates; the load-bearing numbers are the recall curves."""
    theta = np.radians(THETA_OP_DEG)
    q_center = bare_query_center(worked, P, g, theta)
    hyps = np.atleast_2d(sample_vmf(6, P[worked], KAPPA_HYP, seed=HYDE_SEED + 777))
    hhat = normalize(hyps.mean(axis=0))
    pts = np.vstack([P, q_center, hhat, hyps])
    mean = pts.mean(axis=0)
    U, s, Vt = np.linalg.svd(pts - mean, full_matrices=False)
    coords = (pts - mean) @ Vt[:2].T
    n = P.shape[0]
    return {
        "docs": [[_r3(x), _r3(y)] for x, y in coords[:n]],
        "query": [_r3(coords[n][0]), _r3(coords[n][1])],
        "hyde": [_r3(coords[n + 1][0]), _r3(coords[n + 1][1])],
        "hyps": [[_r3(x), _r3(y)] for x, y in coords[n + 2:]],
        "worked": int(worked),
        "gold_sector": int(sector[worked]),
    }


def viz_constants() -> None:
    """Print every MEASURED number QueryTransformationHydeLaboratory.tsx mirrors to the decimal.
    TS recomputes only CLOSED FORM (the 1-1/k reference law, the 1-p bias reference line, axis
    scales). Every recall curve, variance, and projected coordinate is baked here."""
    P, sector = document_manifold()
    g = query_offset_axis(P)
    targets = hallucination_targets(P)
    n_docs = P.shape[0]

    print("  // ----- shared constants -----")
    print(f"const HYDE_N_DOCS = {n_docs};")
    print(f"const HYDE_SECTOR = {[int(x) for x in sector]};")
    print(f"const HYDE_THETA_OP = {THETA_OP_DEG};")
    print(f"const HYDE_KAPPA_HYP = {KAPPA_HYP};")

    print("  // ----- Panel A: the query-document gap (recall vs distribution shift) -----")
    print(f"const SHIFT_GRID = {[float(d) for d in SHIFT_GRID_DEG]};   // theta in degrees")
    bare = [bare_recall_at_theta(P, g, np.radians(d)) for d in SHIFT_GRID_DEG]
    hyde_flat = hyde_recall(P, targets, k=K_INF, p=0.0)
    print(f"const BARE_RECALL = {[_r3(v) for v in bare]};")
    print(f"const HYDE_RECALL_FLAT = {_r3(hyde_flat)};   // theta-independent (HyDE ignores query position)")
    geo = _panel_a_geometry(P, g, sector)
    print(f"const GEO = {geo};")

    print("  // ----- Panel B: Monte-Carlo bias vs variance -----")
    print(f"const K_GRID = {list(K_GRID)};")
    print(f"const HALLU_GRID = {[float(p) for p in HALLU_GRID]};")
    recall_kp = [[_r3(hyde_recall(P, targets, k, p=p)) for k in K_GRID] for p in HALLU_GRID]
    print(f"const RECALL_K_BY_P = {recall_kp};   // [p index][k index]")
    deficit = [_r3(estimator_deficit(0, P, k)) for k in K_GRID]
    print(f"const VAR_DEFICIT = {deficit};   // 1 - <hhat_k, mu>, faithful")

    print("  // ----- Panel C: HyDE as neural pseudo-relevance feedback -----")
    print(f"const ALPHA_GRID = {[float(a) for a in ALPHA_GRID]};")
    theta = np.radians(THETA_OP_DEG)
    print(f"const RECALL_ALPHA_HYDE = {[_r3(v) for v in recall_vs_alpha(P, g, 'hyde', theta)]};")
    print(f"const RECALL_ALPHA_PRF = {[_r3(v) for v in recall_vs_alpha(P, g, 'prf', theta)]};")
    print(f"const PRF_ANCESTOR = {prf_ancestor_curve()};   // imported term-space RM3/Rocchio recall@4")


def _run_all() -> None:
    test_collapse_perfect_hypothetical()
    test_gap_exists()
    test_bare_degrades_with_shift()
    test_hyde_shift_robust()
    test_mc_recall_rises_with_k()
    test_mc_variance_falls_one_over_k()
    test_consistency_faithful()
    test_hallucination_bias_floor()
    test_bias_is_irreducible()
    test_bias_beats_variance()
    test_hyde_update_alpha0_is_bare()
    test_hyde_beats_prf_real()
    test_prf_ancestor_imported()
    test_rocchio_form_parallel()
    test_guards()
    test_viz_constants_reproducible()


if __name__ == "__main__":
    print("query_transformation_hyde: running tests")
    _run_all()
    print("\nDemo:")
    hyde_demo()
    print("\nviz_constants (mirror into QueryTransformationHydeLaboratory.tsx):")
    viz_constants()
    print("\nall checks passed.")
