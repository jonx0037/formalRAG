"""Faithfulness and Groundedness as Measurable Quantities — the canonical reference.

Run (CPU-only, < 60 s):
    uv run --with numpy --with scipy --with scikit-learn \
        python notebooks/faithfulness-groundedness/faithfulness_groundedness.py

This module turns HyDE's synthetic hallucination rate `p` into a MEASURED, two-sided quantity on a
generated answer:

  * faithfulness = PRECISION of an answer's atomic claims against the retrieved context (what fraction
    of what was SAID is supported), and
  * groundedness / coverage = RECALL of the supportable facts (what fraction of the context was USED).

They diverge (terse vs verbose); a noisy LLM judge measures both with a bias we debias (Rogan-Gladen)
and a confidence we calibrate (ECE / Platt / isotonic); trading coverage for a guaranteed faithfulness
under a distribution-free conformal back-off is the abstention frontier; and a supported claim is one
whose pointwise mutual information with the context is positive (bits-of-grounding).

WHAT IS NEW HERE (vs the imports): the generated answer as a SET of claims, the two-sided precision/
recall PAIR over it, and the answer-generation model that bridges HyDE's scalar p to that pair. The
judge form, the calibration suite, the conformal back-off, the PMI bits, and the precision/recall
primitives are all IMPORTED from the published stack and never reimplemented. Only `query-transformation
-hyde` and `pmi-retrieval-value` are pedagogical prerequisites; everything else is an in-site
`connections[]` sibling whose functions we reuse to source numbers (import graph != the DAG).
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.special import expit

# --------------------------------------------------------------------------- #
# Import the published stack. Add EVERY ancestor's hyphenated dir to the path (importing the judge /
# calibration / conformal modules pulls the whole retrieval + evaluation subtree), then import the
# UNDERSCORED modules. We never reimplement an imported routine and never import a downstream topic.
# --------------------------------------------------------------------------- #
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "the-retrieval-problem",
    "infonce-contrastive-objective",
    "vector-quantization-lloyd-max",
    "product-quantization",
    "ivf-voronoi-partitioning",
    "bm25",
    "vector-space-model-tfidf",
    "query-likelihood-language-models",
    "probability-ranking-principle",
    "dense-retrieval-dual-encoders",
    "late-interaction-learned-sparse",
    "multi-vector-ann-retrieval",
    "set-metrics-precision-recall-map-mrr",
    "ndcg-discount-geometry",
    "significance-testing-calibration",
    "llm-as-judge-ragas",
    "conformal-factuality",
    "pseudo-relevance-feedback",
    "query-transformation-hyde",
    "pmi-retrieval-value",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf                       # noqa: E402
from dense_retrieval_dual_encoders import (                                       # noqa: E402
    dpr_finance_matrix, dual_encoder_score, DPR_SEED, DPR_DIM,
)
from query_transformation_hyde import generation_center                          # noqa: E402
from pmi_retrieval_value import (                                                 # noqa: E402
    answer_prior, answer_posterior, pmi_pointwise,
)
from set_metrics_precision_recall_map_mrr import (                               # noqa: E402
    precision_at_k, recall_at_k, f1_at_k,
)
from llm_as_judge_ragas import (                                                  # noqa: E402
    _logit, rogan_gladen, rogan_gladen_variance, confusion_rates,
    JUDGE_PERFECT, JUDGE_BALANCED,
)
from significance_testing_calibration import (                                    # noqa: E402
    platt_scale, apply_platt, isotonic_calibrate, apply_isotonic,
    expected_calibration_error, auc_pooled, reliability_diagram,
)
from conformal_factuality import (                                               # noqa: E402
    back_off_retained, false_claim_loss, fraction_loss, loss_matrix,
    conformal_risk_control_threshold,
)


# --------------------------------------------------------------------------- #
# Module constants — tuned by a build-and-run `_diagnostics()` sweep, never guessed. Every vMF draw is
# seeded so every baked number is reproducible.
# --------------------------------------------------------------------------- #
FG_SEED = DPR_SEED                 # 7 — the shared finance-geometry seed
FG_DIM = DPR_DIM                   # 32
FG_N_SECTORS = 4
FG_N_COMP = 4                      # companies per sector -> 16 docs; gives same-sector spares OUT of context
FG_QPS = 8                         # queries per sector -> 32 sector-ambiguous queries
FG_KAPPA_QUERY = 30.0              # query concentration around the SECTOR mean (ambiguous at company level)
CTX_K = 3                          # retrieved context size = number of SUPPORTABLE facts per query

KAPPA_CLAIM = 300.0                # a claim is a SHARP assertion: tight vMF around its target fact
COS_SUPPORT = 0.78                 # the geometric entailment oracle: supported iff nearest ctx-fact cos >= this
PHI_HALLU_DEG = 90.0               # hallucinated-claim tilt toward the out-of-context company (90 = squarely on it)

TAU = 0.2                          # the imported answer-model temperature (sign separation needs tau <~ 0.2)
TAU_DOC = 0.2

# The LENIENT judge (in the spirit of conformal-factuality's): informative (AUC ~ 0.85) but a clear
# OVER-ENDORSER — high base sensitivity, low specificity — so (1) faithful and unfaithful CONFIDENCES
# OVERLAP and unsupported claims leak through at high confidence (the gap conformal risk control must
# close), and (2) the naive judge-mean faithfulness is BIASED ABOVE the truth (false endorsements
# outweigh missed true claims), the bias Rogan-Gladen removes. A balanced judge separates the classes
# near-perfectly here, which would make every trade-off vacuous.
JUDGE = dict(sens0=0.95, spec0=0.52, b_len=2.5, b_pos=0.6)

ALPHA = 0.10                       # the central back-off risk level
LAMBDA_GRID = tuple(round(x, 4) for x in np.linspace(0.0, 1.0, 51))   # confidence-cut grid
CALIB_FRAC = 0.5                   # split-conformal calibration fraction (whole answers)

# The panel: one answer per query at a fixed length / faithfulness / coverage, the object the judge,
# calibration and conformal back-off all run over.
PANEL_NCLAIMS = 8
PANEL_R = 0.65                     # base per-slot faithfulness rate of the synthetic generator
PANEL_GCOV = 1.0                   # the generator attempts to cover every context fact

NCLAIM_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12)   # the verbosity sweep (the faithfulness-coverage frontier)
WORKED_Q = 0                       # the worked answer for Panel A (set after pick_worked_query)


# =========================================================================== #
# Movement 0 — the corpus: reuse the dense finance manifold, retrieve a context, name a hallucination
# target. (One vMF company prototype per filing; a context is the top-CTX_K retrieved filings.)
# =========================================================================== #

def faithfulness_corpus(seed: int = FG_SEED, dim: int = FG_DIM, n_sectors: int = FG_N_SECTORS,
                        n_comp: int = FG_N_COMP, qps: int = FG_QPS,
                        kappa_query: float = FG_KAPPA_QUERY, ctx_k: int = CTX_K) -> dict:
    """Build the answer-grounding corpus on the dense-retrieval finance manifold.

    `dpr_finance_matrix` gives P (one unit-vector filing per company, the fact prototypes) and each
    company's sector. We draw sector-ambiguous queries around each SECTOR MEAN (kappa_query), retrieve a
    CONTEXT of the top-ctx_k filings by the dual-encoder score, and name a HALLUCINATION TARGET per query:
    the in-sector company NOT in the context (a real, plausible filing the answer could confuse for the
    retrieved ones — the off-context direction a hallucinated claim points at). Returns a dict with the
    manifold P, the queries Q, the gold company, the per-query context fact ids, and the hallu target."""
    _, P, _, sector_of = dpr_finance_matrix(seed=seed, dim=dim, n_sectors=n_sectors, n_comp=n_comp)
    sector_of = np.asarray(sector_of)
    # Recover the sector means dpr_finance_matrix seeded (its first and only consumption of default_rng).
    sector_mu = normalize(np.random.default_rng(seed).standard_normal((n_sectors, dim)))
    g_axis = normalize(P.mean(axis=0))                      # the generic 'document-ness' direction
    Q, gold, q_sector, context, hallu = [], [], [], [], []
    for si in range(n_sectors):
        comp_ids = np.where(sector_of == si)[0]
        draws = np.atleast_2d(sample_vmf(qps, sector_mu[si], kappa_query, seed=seed + 911 + si))
        for q in draws:
            qn = normalize(q)
            scores = dual_encoder_score(qn, P)
            ctx = list(np.argsort(-scores)[:ctx_k])         # the retrieved context (fact ids)
            g_id = int(comp_ids[int(np.argmax(P[comp_ids] @ qn))])   # gold = nearest in-sector company
            # hallu target: an in-sector company NOT in the context (fall back to the nearest off-context).
            off = [int(c) for c in comp_ids if int(c) not in ctx]
            if off:
                h = int(off[int(np.argmax(P[off] @ P[g_id]))])      # the most confusable off-context peer
            else:
                rest = [d for d in range(P.shape[0]) if d not in ctx]
                h = int(rest[int(np.argmax(P[rest] @ P[g_id]))])
            Q.append(qn); gold.append(g_id); q_sector.append(si)
            context.append([int(c) for c in ctx]); hallu.append(h)
    return {
        "P": P, "protos": P, "sector_of": sector_of, "g_axis": g_axis,
        "Q": np.array(Q), "gold": np.array(gold), "q_sector": np.array(q_sector),
        "context": np.array(context), "hallu_target": np.array(hallu),
        "n_docs": int(P.shape[0]), "n_queries": len(Q), "ctx_k": int(ctx_k),
    }


_CORPUS: dict | None = None


def _corpus(seed: int = FG_SEED) -> dict:
    """Module-scope cache: the finance geometry, queries, contexts and hallucination targets built once."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = faithfulness_corpus(seed)
    return _CORPUS


# =========================================================================== #
# Movement 1 — the answer-generation model (the NEW primitive). An answer is a SET of claim vectors,
# each a supported draw near a covered context fact or a hallucinated draw near the off-context company.
# =========================================================================== #

def claim_support_oracle(cvec: np.ndarray, context_fact_vecs: np.ndarray,
                         cos_support: float = COS_SUPPORT) -> tuple[int, int, float]:
    """The geometric ENTAILMENT oracle for one claim. A claim is SUPPORTED iff its nearest context fact
    has cosine >= cos_support (it is grounded in some retrieved filing). Returns (y, local_fact_idx, cos):
    y in {0,1}, local_fact_idx is the index INTO the context (or -1 if unsupported), cos the best cosine."""
    if context_fact_vecs.shape[0] == 0:
        return 0, -1, -1.0
    sims = context_fact_vecs @ cvec
    j = int(np.argmax(sims))
    best = float(sims[j])
    return (1, j, best) if best >= cos_support else (0, -1, best)


def generate_answer(corpus: dict, q_idx: int, n_claims: int, coverage_target: float = PANEL_GCOV,
                    faithfulness_rate: float = PANEL_R, seed: int = 0,
                    kappa_claim: float = KAPPA_CLAIM, phi_hallu_deg: float = PHI_HALLU_DEG,
                    intents: list[bool] | None = None) -> dict:
    """Generate one answer for query q as a SET of claim vectors and their oracle support labels.

    The generator attempts to cover `coverage_target` of the context's facts: the first
    round(g * ctx_k) context facts are the COVERED facts it tries to ground. For each of n_claims claims,
    with probability `faithfulness_rate` it makes a SUPPORTED claim (a tight vMF draw near the next covered
    fact, cycling through them so a longer answer reaches more facts), otherwise a HALLUCINATED claim (a
    draw near the off-context hallucination target — a plausible figure not in the retrieved filings). The
    oracle then labels each claim by `claim_support_oracle`. This is the bridge from HyDE's scalar
    hallucination rate p = 1 - faithfulness_rate to a measured precision/recall pair.

    `intents` (optional) forces the supported/hallucinated choice per claim (for the deterministic worked
    answer), overriding the random `faithfulness_rate` draw.

    Returns: claims (n, dim), y (n,) oracle support, attribution (n,) the GLOBAL context fact id a supported
    claim grounds (-1 if unsupported), covered_facts, context_facts."""
    P = corpus["P"]
    context_facts = [int(c) for c in corpus["context"][q_idx]]
    ctx_vecs = P[context_facts]
    h = int(corpus["hallu_target"][q_idx])
    targets = [h] * P.shape[0]                                          # generation_center reads targets[c]
    phi = math.radians(phi_hallu_deg)
    n_cov = max(0, min(len(context_facts), int(round(coverage_target * len(context_facts)))))
    covered = context_facts[:n_cov]
    rng = np.random.default_rng(seed * 1000 + q_idx)
    claims, y, attribution = [], [], []
    cover_ptr = 0
    for ci in range(int(n_claims)):
        if intents is not None:
            supported_intent = bool(covered) and bool(intents[ci])
        else:
            supported_intent = bool(covered) and (rng.random() < faithfulness_rate)
        if supported_intent:
            fact = covered[cover_ptr % len(covered)]
            cover_ptr += 1
            center = P[fact]                                            # faithful: the fact's own direction
        else:
            center = generation_center(int(corpus["gold"][q_idx]), P, targets, hallucinated=True, phi=phi)
        cvec = normalize(np.atleast_2d(sample_vmf(1, center, kappa_claim,
                                                  seed=int(rng.integers(1, 2**31)))).ravel())
        yi, local, _ = claim_support_oracle(cvec, ctx_vecs)
        claims.append(cvec)
        y.append(yi)
        attribution.append(context_facts[local] if yi == 1 else -1)
    return {
        "claims": np.array(claims), "y": np.array(y, dtype=int),
        "attribution": np.array(attribution, dtype=int),
        "covered_facts": covered, "context_facts": context_facts,
        "q_idx": int(q_idx), "n_claims": int(n_claims),
    }


# =========================================================================== #
# Movement 2 — the two-sided metric: faithfulness = precision (over claims), coverage = recall (over
# facts). Both ANCHORED to the imported set-metrics; the denominator asymmetry is why they diverge.
# =========================================================================== #

def answer_faithfulness(answer: dict) -> float:
    """Faithfulness = PRECISION of the claims: (# supported claims) / (# claims). Anchored to the imported
    `precision_at_k` over the CLAIM-ID space (each claim a distinct item, relevant = the supported ids)."""
    n = answer["n_claims"]
    if n == 0:
        return 0.0
    supported = {i for i in range(n) if answer["y"][i] == 1}
    return precision_at_k(list(range(n)), supported, n)


def answer_coverage(answer: dict) -> float:
    """Groundedness / coverage = RECALL of the supportable facts: (# distinct context facts grounded by
    some supported claim) / (# context facts). Anchored to the imported `recall_at_k` over the FACT space
    (the claims' attributed fact ids vs the context facts) — a set operation, so duplicates collapse."""
    facts = set(answer["context_facts"])
    if not facts:
        return 0.0
    return recall_at_k(list(answer["attribution"]), facts, answer["n_claims"])


def answer_f1(answer: dict) -> float:
    """F1 = harmonic mean of faithfulness (precision) and coverage (recall), via the imported `f1_at_k`
    over the FACT space (the natural single-number summary of the two-sided quality)."""
    return f1_at_k(list(answer["attribution"]), set(answer["context_facts"]), answer["n_claims"])


# =========================================================================== #
# Build-and-run diagnostics — the geometry tuning aid. Run this FIRST and read the separations before
# trusting any headline; the constants above are set from what it prints.
# =========================================================================== #

def _diagnostics(seed: int = FG_SEED) -> None:
    """Print the separations the constants are tuned against:
      * context composition (is the retrieved context in-sector? are the facts separable?),
      * the oracle separation (supported-claim cosines vs hallucinated-claim cosines to the nearest fact),
      * a quick precision/coverage spread over the verbosity grid."""
    corpus = faithfulness_corpus(seed)
    P = corpus["P"]
    print(f"corpus: {corpus['n_docs']} docs, {corpus['n_queries']} queries, ctx_k={corpus['ctx_k']}, "
          f"dim={P.shape[1]}")
    # context composition
    in_sector = []
    fact_sep = []
    for q in range(corpus["n_queries"]):
        ctx = corpus["context"][q]
        sec = corpus["sector_of"][ctx]
        in_sector.append(float(np.mean(sec == corpus["q_sector"][q])))
        G = P[ctx] @ P[ctx].T
        off = (G.sum() - np.trace(G)) / (len(ctx) * (len(ctx) - 1))
        fact_sep.append(float(off))
    print(f"context in-sector fraction: mean {np.mean(in_sector):.3f} (want ~1.0)")
    print(f"within-context fact cosine: mean {np.mean(fact_sep):.3f} (want < COS_SUPPORT={COS_SUPPORT} "
          f"so a claim attributes to ONE fact)")
    # oracle separation: supported vs hallucinated nearest-context-fact cosines (explicit intent labels)
    sup_cos, hal_cos = _oracle_separation(corpus, seed=1)
    print(f"oracle cos to nearest ctx fact — supported draws: mean {np.mean(sup_cos):.3f} "
          f"(>= {COS_SUPPORT}); hallucinated draws: mean {np.mean(hal_cos):.3f} (< {COS_SUPPORT})")
    print(f"  supported  >= COS_SUPPORT: {np.mean(np.array(sup_cos) >= COS_SUPPORT):.3f} of draws")
    print(f"  hallucinated < COS_SUPPORT: {np.mean(np.array(hal_cos) < COS_SUPPORT):.3f} of draws")
    # precision / coverage spread over verbosity
    print("verbosity sweep (mean precision, coverage, f1 over queries):")
    for n in NCLAIM_GRID:
        ps, cs, fs = [], [], []
        for q in range(corpus["n_queries"]):
            a = generate_answer(corpus, q, n_claims=n, coverage_target=1.0, faithfulness_rate=PANEL_R, seed=2)
            ps.append(answer_faithfulness(a)); cs.append(answer_coverage(a)); fs.append(answer_f1(a))
        print(f"  n={n:2d}: precision {np.mean(ps):.3f}  coverage {np.mean(cs):.3f}  f1 {np.mean(fs):.3f}")


def _oracle_separation(corpus: dict, seed: int = 1):
    """Re-draw supported and hallucinated claims explicitly and report their nearest-context-fact cosines,
    so the oracle threshold COS_SUPPORT can be set to cleanly separate the two populations."""
    P = corpus["P"]
    sup_cos, hal_cos = [], []
    for q in range(corpus["n_queries"]):
        context_facts = [int(c) for c in corpus["context"][q]]
        ctx_vecs = P[context_facts]
        h = int(corpus["hallu_target"][q])
        targets = [h] * P.shape[0]
        rng = np.random.default_rng(seed * 7919 + q)
        for fact in context_facts:                                     # supported draws near each fact
            for _ in range(4):
                c = normalize(np.atleast_2d(sample_vmf(1, P[fact], KAPPA_CLAIM,
                                                       seed=int(rng.integers(1, 2**31)))).ravel())
                _, _, cos = claim_support_oracle(c, ctx_vecs)
                sup_cos.append(cos)
        center = generation_center(int(corpus["gold"][q]), P, targets, hallucinated=True,
                                   phi=math.radians(PHI_HALLU_DEG))
        for _ in range(12):                                            # hallucinated draws near the target
            c = normalize(np.atleast_2d(sample_vmf(1, center, KAPPA_CLAIM,
                                                   seed=int(rng.integers(1, 2**31)))).ravel())
            _, _, cos = claim_support_oracle(c, ctx_vecs)
            hal_cos.append(cos)
    return sup_cos, hal_cos


# =========================================================================== #
# Movement 3 — the faithfulness-coverage frontier (the verbosity sweep) and the terse/verbose divergence.
# =========================================================================== #

def verbosity_curve(corpus: dict, r: float = PANEL_R, g_cov: float = PANEL_GCOV,
                    nclaim_grid=NCLAIM_GRID, seed: int = 2) -> list[dict]:
    """For each answer length, the mean faithfulness (precision), coverage (recall) and F1 over the queries.
    Coverage rises then saturates; precision is roughly flat; F1 peaks at an INTERIOR length — the
    F1-optimal answer is neither terse nor exhaustive."""
    rows = []
    for n in nclaim_grid:
        ps, cs, fs = [], [], []
        for q in range(corpus["n_queries"]):
            a = generate_answer(corpus, q, int(n), g_cov, r, seed)
            ps.append(answer_faithfulness(a)); cs.append(answer_coverage(a)); fs.append(answer_f1(a))
        rows.append({"n_claims": int(n), "faithfulness": float(np.mean(ps)),
                     "coverage": float(np.mean(cs)), "f1": float(np.mean(fs))})
    return rows


def divergence_pair(corpus: dict, seed: int = 4) -> dict:
    """A terse answer (one supported claim) and a verbose answer (many claims, half hallucinated): the
    terse one is faithful but thin, the verbose one covers everything but invents figures — precision and
    coverage move in OPPOSITE directions, the reason a single factuality score hides the trade."""
    terse = [generate_answer(corpus, q, 1, 1.0, 1.0, seed) for q in range(corpus["n_queries"])]
    verbose = [generate_answer(corpus, q, 12, 1.0, 0.5, seed) for q in range(corpus["n_queries"])]
    return {
        "terse": {"faithfulness": float(np.mean([answer_faithfulness(a) for a in terse])),
                  "coverage": float(np.mean([answer_coverage(a) for a in terse]))},
        "verbose": {"faithfulness": float(np.mean([answer_faithfulness(a) for a in verbose])),
                    "coverage": float(np.mean([answer_coverage(a) for a in verbose]))},
    }


# =========================================================================== #
# Movement 4 — the noisy judge over generated claims: the imported logit-bias FORM, applied to claim
# features; Rogan-Gladen debiasing; ECE / Platt / isotonic calibration.
# =========================================================================== #

def _claim_judge_perfect(params: dict) -> bool:
    return (params["sens0"] >= 1.0 and params["spec0"] >= 1.0
            and params.get("b_len", 0.0) == 0.0 and params.get("b_pos", 0.0) == 0.0)


def claim_judge_probs(assert_z: np.ndarray, pos_term: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """The judge's per-claim P(supported), reusing llm-as-judge-ragas's logit-bias form (the IMPORTED
    `_logit` + scipy `expit`): sens_eff where truly supported, fpr_eff where not, both shifted by the
    claim's assertiveness (its cosine to the generic 'document-ness' axis) and its position in the answer.
    A perfect judge (sens=spec=1, no bias) returns the truth deterministically — the collapse anchor."""
    y = np.asarray(y)
    if _claim_judge_perfect(params):
        return y.astype(float)
    shift = params["b_len"] * np.asarray(assert_z, dtype=float) + params["b_pos"] * np.asarray(pos_term, dtype=float)
    sens_eff = expit(_logit(params["sens0"]) + shift)
    fpr_eff = expit(_logit(1.0 - params["spec0"]) + shift)
    return np.where(y == 1, sens_eff, fpr_eff)


def build_panel(corpus: dict, n_claims: int = PANEL_NCLAIMS, r: float = PANEL_R,
                g_cov: float = PANEL_GCOV, seed: int = 3) -> dict:
    """One answer per query at a fixed length — the object the judge, calibration and conformal back-off
    all run over. Returns the answers, the (n_queries, n_claims) oracle truth Y, the per-claim assertiveness
    (cosine to the generic axis) and the position term."""
    g = corpus["g_axis"]
    answers = [generate_answer(corpus, q, n_claims, g_cov, r, seed) for q in range(corpus["n_queries"])]
    Y = np.array([a["y"] for a in answers])
    assert_raw = np.array([a["claims"] @ g for a in answers])          # (n_q, n_claims) cosine to g
    pos = 0.5 - (np.arange(n_claims) / (n_claims - 1) if n_claims > 1 else np.zeros(n_claims))
    return {"answers": answers, "Y": Y, "assert_raw": assert_raw, "pos": pos, "n_claims": int(n_claims)}


def panel_confidence(panel: dict, params: dict = JUDGE) -> np.ndarray:
    """The judge's confidence matrix over the panel (n_queries, n_claims). Assertiveness is z-scored over
    ALL panel claims (the length bias keys on the panel-wide feature); the position term broadcasts."""
    a = panel["assert_raw"]
    sd = float(a.std())
    az = (a - a.mean()) / (sd if sd > 0 else 1.0)
    pos = np.broadcast_to(panel["pos"], a.shape)
    return claim_judge_probs(az, pos, panel["Y"], params)


def panel_corrected_faithfulness(panel: dict, conf: np.ndarray, seed: int = 5) -> dict:
    """Naive vs Rogan-Gladen-corrected faithfulness over the panel. Draw Bernoulli verdicts from the judge
    confidence, audit the empirical sensitivity / specificity, and invert (the IMPORTED `rogan_gladen`).
    The naive observed rate is a BIASED estimate of the oracle faithfulness; the correction removes it."""
    rng = np.random.default_rng(seed)
    verdicts = rng.random(conf.shape) < conf
    cr = confusion_rates(verdicts, panel["Y"].astype(bool))
    p_obs = float(verdicts.mean())
    n = int(panel["Y"].size)
    return {"p_obs": p_obs, "sens": cr["sens"], "spec": cr["spec"],
            "youden": cr["sens"] + cr["spec"] - 1.0,
            "corrected": rogan_gladen(p_obs, cr["sens"], cr["spec"]),
            "var": rogan_gladen_variance(p_obs, cr["sens"], cr["spec"], n),
            "oracle": float(panel["Y"].mean())}


def panel_calibration(panel: dict, conf: np.ndarray, n_bins: int = 10) -> dict:
    """The judge's calibration over the pooled panel claims: raw / Platt / isotonic ECE and AUC, plus the
    reliability bins for each (the IMPORTED suite). Platt and isotonic are fit on the whole pool here; the
    held-out split is the conformal pillar's job. Returns a dict the calibration panel mirrors."""
    c = conf.ravel(); y = panel["Y"].ravel().astype(float)
    a, b = platt_scale(c, y); c_platt = apply_platt(c, a, b)
    iso = isotonic_calibrate(c, y); c_iso = apply_isotonic(c, iso)
    return {
        "ece_raw": expected_calibration_error(c, y, n_bins),
        "ece_platt": expected_calibration_error(c_platt, y, n_bins),
        "ece_iso": expected_calibration_error(c_iso, y, n_bins),
        "auc_raw": auc_pooled(c, y), "auc_platt": auc_pooled(c_platt, y),
        "platt_ab": (float(a), float(b)),
        "rel_raw": reliability_diagram(c, y, n_bins),
        "rel_platt": reliability_diagram(c_platt, y, n_bins),
        "rel_iso": reliability_diagram(c_iso, y, n_bins),
    }


# =========================================================================== #
# Movement 5 — the abstention bridge: split-conformal recall guarantee + conformal risk control (the
# IMPORTED machinery, framed as the trade coverage-for-guaranteed-faithfulness that becomes abstention).
# =========================================================================== #

def split_panel(n_queries: int, calib_frac: float = CALIB_FRAC):
    """The whole-answer calib / test split (queries 0..half-1 calibrate). Splitting on answers keeps an
    answer's claims together — the prereq's held-out idiom."""
    half = max(1, int(round(n_queries * calib_frac)))
    calib = np.arange(n_queries) < half
    return calib, ~calib, half


def calibrated_panel_conf(panel: dict, conf: np.ndarray, calib_mask: np.ndarray,
                          method: str = "platt") -> np.ndarray:
    """Recalibrate the confidence matrix, FITTING on the calibration answers only and applying everywhere
    (mirrors conformal-factuality's `calibrated_confidence`). 'identity' returns the raw confidence —
    conformal validity holds for it too; recalibration buys efficiency, not validity."""
    if method == "identity":
        return conf
    cflat = conf.ravel(); y = panel["Y"].ravel().astype(float)
    cm = np.repeat(calib_mask, panel["n_claims"])
    sc, yc = cflat[cm], y[cm]
    if method == "platt":
        a, b = platt_scale(sc, yc); return apply_platt(conf, a, b)
    if method == "iso":
        iso = isotonic_calibrate(sc, yc); return apply_isotonic(conf, iso)
    raise ValueError(f"unknown calibration method {method}")


def abstention_frontier(panel: dict, conf: np.ndarray, lam_grid=LAMBDA_GRID) -> list[dict]:
    """The faithfulness-coverage frontier swept by the confidence cut tau: at each tau retain claims with
    confidence >= tau, and report the mean RETAINED faithfulness (precision) and RETAINED coverage (recall)
    over the panel. Precision rises, coverage falls — the generation PR curve, made the back-off frontier."""
    rows = []
    for tau in lam_grid:
        precs, covs, rets = [], [], []
        for qi, ans in enumerate(panel["answers"]):
            keep = conf[qi] >= tau
            rets.append(float(keep.mean()))         # a fully-abstained answer retains 0.0 (not skipped)
            if keep.sum() == 0:
                continue
            precs.append(float(ans["y"][keep].mean()))
            attr = [ans["attribution"][i] for i in range(ans["n_claims"]) if keep[i]]
            covs.append(recall_at_k(attr, set(ans["context_facts"]), int(keep.sum())))
        rows.append({"tau": float(tau),
                     "faithfulness": float(np.mean(precs)) if precs else 1.0,
                     "coverage": float(np.mean(covs)) if covs else 0.0,
                     "retention": float(np.mean(rets)) if rets else 0.0})
    return rows


def crc_backoff(panel: dict, conf: np.ndarray, alpha: float = ALPHA,
                calib_frac: float = CALIB_FRAC) -> dict:
    """Conformal risk control over the panel: fit the false-claim-loss threshold lambda_hat on the
    calibration answers (the IMPORTED monotone `loss_matrix` + `conformal_risk_control_threshold`), then
    report the realized TEST false-claim rate and retention. The expected test loss is controlled at alpha."""
    calib, test, _ = split_panel(panel["Y"].shape[0], calib_frac)
    Pc, Yc = conf[calib], panel["Y"][calib]
    Pt, Yt = conf[test], panel["Y"][test]
    L = loss_matrix(Pc, Yc, LAMBDA_GRID, k=panel["n_claims"])
    lam = conformal_risk_control_threshold(L, LAMBDA_GRID, alpha)
    test_loss = float(np.mean([false_claim_loss(Pt[i], Yt[i], lam, k=panel["n_claims"])
                               for i in range(Pt.shape[0])]))
    return {"lambda_hat": float(lam), "alpha": float(alpha),
            "test_false_claim_rate": test_loss, "retention": float((Pt >= lam).mean())}


# =========================================================================== #
# Movement 6 — bits-of-grounding: a supported claim has positive pointwise MI with the context (IMPORTED
# PMI answer model); a hallucinated claim has non-positive bits.
# =========================================================================== #

def context_centroid(corpus: dict, q_idx: int) -> np.ndarray:
    """The retrieved context as a single evidence vector: the unit mean of its fact prototypes."""
    return normalize(corpus["P"][corpus["context"][q_idx]].mean(axis=0))


def claim_bits(corpus: dict, q_idx: int, cvec: np.ndarray) -> float:
    """The bits-of-grounding of one claim: its nearest answer prototype a = argmax cos(claim, P), scored by
    the pointwise mutual information pmi(a; context | q) = log2 p(a|q,ctx)/p(a|q) with the retrieved context
    as the evidence document (the IMPORTED PMI answer model). Supported claims (a in the context) get
    POSITIVE bits; hallucinated claims (a off-context) get NON-POSITIVE bits."""
    P = corpus["P"]; q = corpus["Q"][q_idx]
    a = int(np.argmax(cvec @ P.T))
    d_vec = context_centroid(corpus, q_idx)
    prior = answer_prior(q, P, P, TAU, TAU_DOC)
    post = answer_posterior(q, d_vec, P, TAU)
    return pmi_pointwise(prior, post, a)


def panel_bits(corpus: dict, panel: dict) -> dict:
    """The bits-of-grounding split over every panel claim: the per-claim PMI of supported vs hallucinated
    claims, their means, and the fraction of hallucinated claims with non-positive bits."""
    sup, hal = [], []
    for ans in panel["answers"]:
        for i in range(ans["n_claims"]):
            b = claim_bits(corpus, ans["q_idx"], ans["claims"][i])
            (sup if ans["y"][i] == 1 else hal).append(b)
    return {"supported": sup, "hallucinated": hal,
            "mean_sup": float(np.mean(sup)) if sup else 0.0,
            "mean_hal": float(np.mean(hal)) if hal else 0.0,
            "frac_hal_nonpos": float(np.mean(np.array(hal) <= 0.0)) if hal else 0.0}


# =========================================================================== #
# The worked answer (Panel A): a deterministic 9-claim finance answer, six supported, three hallucinated.
# =========================================================================== #

WORKED_INTENTS = [True, True, True, True, True, False, False, False, True]   # the finance answer's claims


def pick_worked_query(corpus: dict) -> int:
    """A clean worked query: context fully in-sector with the best-separated facts (minimize the mean
    within-context cosine), so each claim attributes to one fact unambiguously."""
    P = corpus["P"]; best, best_sep = 0, 2.0
    for q in range(corpus["n_queries"]):
        ctx = corpus["context"][q]
        if not np.all(corpus["sector_of"][ctx] == corpus["q_sector"][q]):
            continue
        G = P[ctx] @ P[ctx].T
        sep = float((G.sum() - np.trace(G)) / (len(ctx) * (len(ctx) - 1)))
        if sep < best_sep:
            best_sep, best = sep, q
    return best


def worked_answer(corpus: dict, intents=WORKED_INTENTS, seed: int = 11) -> dict:
    """The deterministic worked answer for Panel A, on the cleanest worked query."""
    return generate_answer(corpus, pick_worked_query(corpus), len(intents), 1.0, 1.0, seed, intents=intents)


# =========================================================================== #
# Demo + viz_constants.
# =========================================================================== #

def _r(v, n=4) -> float:
    return round(float(v), n)


def faithfulness_demo() -> dict:
    """The headline numbers printed."""
    corpus = _corpus()
    div = divergence_pair(corpus)
    panel = build_panel(corpus)
    conf = panel_confidence(panel)
    rg = panel_corrected_faithfulness(panel, conf)
    cal = panel_calibration(panel, conf)
    cc = calibrated_panel_conf(panel, conf, split_panel(panel["Y"].shape[0])[0])
    crc = crc_backoff(panel, cc)
    bits = panel_bits(corpus, panel)
    curve = verbosity_curve(corpus)
    f1s = [r["f1"] for r in curve]
    print(f"finance geometry: {corpus['n_docs']} filings (one vMF company prototype each), "
          f"{corpus['n_queries']} queries, context = top-{corpus['ctx_k']}")
    print(f"MOVEMENT 1-3: faithfulness=precision, coverage=recall DIVERGE — "
          f"terse ({div['terse']['faithfulness']:.2f} faith, {div['terse']['coverage']:.2f} cov) vs "
          f"verbose ({div['verbose']['faithfulness']:.2f} faith, {div['verbose']['coverage']:.2f} cov); "
          f"F1-optimal length = {curve[int(np.argmax(f1s))]['n_claims']} claims (interior)")
    print(f"MOVEMENT 4: the lenient judge is biased (naive faithfulness {rg['p_obs']:.3f} vs oracle "
          f"{rg['oracle']:.3f}); Rogan-Gladen corrects to {rg['corrected']:.3f}; raw ECE {cal['ece_raw']:.3f} "
          f"-> Platt {cal['ece_platt']:.3f} -> isotonic {cal['ece_iso']:.3f} (AUC {cal['auc_raw']:.3f})")
    print(f"MOVEMENT 5: conformal back-off controls the false-claim rate at alpha={crc['alpha']} — "
          f"realized {crc['test_false_claim_rate']:.3f}, retaining {crc['retention']:.2f} of claims")
    print(f"MOVEMENT 6: bits-of-grounding sign split — supported {bits['mean_sup']:.3f} bits > 0 > "
          f"hallucinated {bits['mean_hal']:.3f} bits ({bits['frac_hal_nonpos']:.2f} of hallucinations <= 0)")
    return {"divergence": div, "rg": rg, "calibration": cal, "crc": crc, "bits": bits}


def _panel_a_geometry(corpus: dict, ans: dict):
    """A 2-D PCA projection of the worked answer's context facts, claims and hallucination target so the
    lab can DRAW the answer in claim space (supported claims near their fact, hallucinated near the
    off-context target). Purely illustrative coordinates; the load-bearing numbers are y / conf / bits."""
    P = corpus["P"]
    ctx = corpus["context"][ans["q_idx"]]
    h = int(corpus["hallu_target"][ans["q_idx"]])
    pts = np.vstack([P[ctx], P[[h]], ans["claims"]])
    mean = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mean, full_matrices=False)
    coords = (pts - mean) @ Vt[:2].T
    n_ctx = len(ctx)
    return {
        "facts": [[_r(x, 3), _r(y, 3)] for x, y in coords[:n_ctx]],
        "hallu": [_r(coords[n_ctx][0], 3), _r(coords[n_ctx][1], 3)],
        "claims": [[_r(x, 3), _r(y, 3)] for x, y in coords[n_ctx + 1:]],
    }


def viz_constants() -> None:
    """Print every MEASURED number FaithfulnessGroundednessLaboratory.tsx mirrors to the decimal. The TS
    recomputes only CLOSED FORM (F1 from precision/recall, the live-tau frontier point, ECE from the baked
    reliability bins, the histogram bins, the y=1-alpha guarantee line). Every curve is a seeded
    computation, so the baked numbers are reproducible."""
    corpus = _corpus()
    panel = build_panel(corpus)
    conf = panel_confidence(panel)
    calib_mask = split_panel(panel["Y"].shape[0])[0]
    cc = calibrated_panel_conf(panel, conf, calib_mask)
    ans = worked_answer(corpus)

    print("// ----- shared constants -----")
    print(f"const FG_N_DOCS = {corpus['n_docs']};")
    print(f"const FG_CTX_K = {corpus['ctx_k']};")
    print(f"const FG_ALPHA = {ALPHA};")
    print(f"const COS_SUPPORT = {COS_SUPPORT};")

    print("// ----- Panel A: the worked answer (claims, faithfulness=precision, coverage=recall) -----")
    # the worked answer's per-claim confidence (calibrated) and bits, aligned to WORKED_INTENTS
    g = corpus["g_axis"]
    a_assert = ans["claims"] @ g
    az = (a_assert - panel["assert_raw"].mean()) / (panel["assert_raw"].std() or 1.0)
    pos = 0.5 - (np.arange(ans["n_claims"]) / (ans["n_claims"] - 1))
    a_conf = claim_judge_probs(az, pos, ans["y"], JUDGE)
    a_platt = apply_platt(a_conf, *panel_calibration(panel, conf)["platt_ab"])
    a_bits = [claim_bits(corpus, ans["q_idx"], ans["claims"][i]) for i in range(ans["n_claims"])]
    local = [int(np.argmax(corpus["P"][ans["context_facts"]] @ ans["claims"][i])) if ans["y"][i] == 1 else -1
             for i in range(ans["n_claims"])]
    print(f"const WORKED_Y = {[int(v) for v in ans['y']]};")
    print(f"const WORKED_CONF = {[_r(v, 3) for v in a_platt]};   // calibrated judge confidence")
    print(f"const WORKED_BITS = {[_r(v, 3) for v in a_bits]};")
    print(f"const WORKED_ATTR = {local};   // local context-fact index per claim (-1 = unsupported)")
    print(f"const WORKED_FAITHFULNESS = {_r(answer_faithfulness(ans), 3)};")
    print(f"const WORKED_COVERAGE = {_r(answer_coverage(ans), 3)};")
    print(f"const WORKED_F1 = {_r(answer_f1(ans), 3)};")
    print(f"const WORKED_GEO = {_panel_a_geometry(corpus, ans)};")

    print("// ----- Panel B: the faithfulness-coverage frontier (verbosity + the back-off cut) -----")
    print(f"const NCLAIM_GRID = {list(NCLAIM_GRID)};")
    curve = verbosity_curve(corpus)
    print(f"const VERBOSITY = {[{ 'n': r['n_claims'], 'p': _r(r['faithfulness'], 3), 'r': _r(r['coverage'], 3), 'f1': _r(r['f1'], 3)} for r in curve]};")
    front = abstention_frontier(panel, cc)
    # subsample the frontier grid for the viz (every other tau)
    fr = [{"tau": _r(r["tau"], 3), "p": _r(r["faithfulness"], 3), "r": _r(r["coverage"], 3),
           "ret": _r(r["retention"], 3)} for r in front[::2]]
    print(f"const FRONTIER = {fr};")
    crc = crc_backoff(panel, cc)
    print(f"const CRC = {{'lambda': {_r(crc['lambda_hat'], 3)}, 'alpha': {crc['alpha']}, "
          f"'realized': {_r(crc['test_false_claim_rate'], 3)}, 'retention': {_r(crc['retention'], 3)}}};")
    print(f"const DIVERGENCE = {divergence_pair(corpus)};")

    print("// ----- Panel C: judge calibration (reliability bins, ECE, AUC) -----")
    cal = panel_calibration(panel, conf)
    for key in ("rel_raw", "rel_platt", "rel_iso"):
        rows = [[_r(b['conf'], 3), _r(b['acc'], 3), int(b['n'])] for b in cal[key]]
        print(f"const REL_{key.split('_')[1].upper()} = {rows};   // [conf, acc, count] per bin")
    print(f"const ECE = {{'raw': {_r(cal['ece_raw'], 3)}, 'platt': {_r(cal['ece_platt'], 3)}, "
          f"'iso': {_r(cal['ece_iso'], 3)}}};")
    print(f"const AUC = {{'raw': {_r(cal['auc_raw'], 3)}, 'platt': {_r(cal['auc_platt'], 3)}}};")
    rg = panel_corrected_faithfulness(panel, conf)
    print(f"const RG = {{'naive': {_r(rg['p_obs'], 3)}, 'corrected': {_r(rg['corrected'], 3)}, "
          f"'oracle': {_r(rg['oracle'], 3)}, 'sens': {_r(rg['sens'], 3)}, 'spec': {_r(rg['spec'], 3)}}};")

    print("// ----- Panel D: bits-of-grounding (per-claim PMI sign split) -----")
    bits = panel_bits(corpus, panel)
    print(f"const BITS_SUP = {[_r(v, 3) for v in bits['supported']]};")
    print(f"const BITS_HAL = {[_r(v, 3) for v in bits['hallucinated']]};")
    print(f"const BITS_SUMMARY = {{'mean_sup': {_r(bits['mean_sup'], 3)}, "
          f"'mean_hal': {_r(bits['mean_hal'], 3)}, 'frac_hal_nonpos': {_r(bits['frac_hal_nonpos'], 3)}}};")


# =========================================================================== #
# Verification harness — each assert is a pedagogical claim the topic makes.
# =========================================================================== #

def test_perfect_judge_collapse() -> None:
    """Collapse anchor: a PERFECT judge's measured faithfulness equals the answer's true faithfulness,
    which equals the IMPORTED precision_at_k over the claim-id space, to machine precision."""
    corpus = _corpus()
    panel = build_panel(corpus)
    conf = panel_confidence(panel, JUDGE_PERFECT)
    assert np.array_equal(conf, panel["Y"].astype(float)), "perfect judge must return the truth"
    for qi, ans in enumerate(panel["answers"]):
        measured = float(conf[qi].mean())
        assert abs(measured - answer_faithfulness(ans)) < 1e-12, "measured != answer_faithfulness"
        n = ans["n_claims"]
        sup = {i for i in range(n) if ans["y"][i] == 1}
        assert abs(answer_faithfulness(ans) - precision_at_k(list(range(n)), sup, n)) < 1e-12
    print("[ok] collapse: perfect-judge faithfulness == answer faithfulness == precision_at_k (<1e-12)")


def test_faithfulness_is_precision_coverage_is_recall() -> None:
    """The two-sided definition is exactly the imported set metrics: faithfulness == precision_at_k over
    claims (== mean of the support labels), coverage == recall_at_k over the context facts."""
    corpus = _corpus()
    for q in range(corpus["n_queries"]):
        a = generate_answer(corpus, q, 6, 1.0, PANEL_R, seed=7)
        n = a["n_claims"]
        sup = {i for i in range(n) if a["y"][i] == 1}
        assert abs(answer_faithfulness(a) - precision_at_k(list(range(n)), sup, n)) < 1e-12
        assert abs(answer_faithfulness(a) - float(a["y"].mean())) < 1e-12
        assert abs(answer_coverage(a) - recall_at_k(list(a["attribution"]), set(a["context_facts"]), n)) < 1e-12
    print("[ok] faithfulness == precision_at_k == mean(y); coverage == recall_at_k (<1e-12)")


def test_oracle_separation() -> None:
    """The geometry: supported claim draws clear the entailment threshold, hallucinated draws fall below it
    — the support oracle cleanly distinguishes grounded from ungrounded claims."""
    corpus = _corpus()
    sup, hal = _oracle_separation(corpus, seed=1)
    assert np.mean(np.array(sup) >= COS_SUPPORT) > 0.98, f"supported draws should clear COS_SUPPORT: {np.mean(sup)}"
    assert np.mean(np.array(hal) < COS_SUPPORT) > 0.95, f"hallucinated draws should fall below: {np.mean(hal)}"
    assert np.mean(sup) - np.mean(hal) > 0.2, "supported and hallucinated cosines should be well separated"
    print(f"[ok] oracle separation: supported cos {np.mean(sup):.3f} >> hallucinated cos {np.mean(hal):.3f}")


def test_faithfulness_coverage_diverge() -> None:
    """The headline: precision and recall move in OPPOSITE directions — a terse answer is faithful but thin,
    a verbose answer covers more but invents figures. A single factuality score hides the trade."""
    corpus = _corpus()
    d = divergence_pair(corpus)
    assert d["terse"]["faithfulness"] > d["verbose"]["faithfulness"] + 0.2, f"precision should drop: {d}"
    assert d["terse"]["coverage"] < d["verbose"]["coverage"] - 0.3, f"recall should rise: {d}"
    print(f"[ok] divergence: terse ({d['terse']['faithfulness']:.2f}/{d['terse']['coverage']:.2f}) vs "
          f"verbose ({d['verbose']['faithfulness']:.2f}/{d['verbose']['coverage']:.2f}) faith/cov")


def test_interior_f1_optimum() -> None:
    """Coverage saturates while precision is flat, so F1 (their harmonic mean) peaks at an INTERIOR answer
    length — the F1-optimal answer is neither a single claim nor exhaustive."""
    corpus = _corpus()
    curve = verbosity_curve(corpus)
    f1 = [r["f1"] for r in curve]
    cov = [r["coverage"] for r in curve]
    arg = int(np.argmax(f1))
    assert 0 < arg < len(f1) - 1, f"F1 optimum should be interior, got index {arg}: {f1}"
    assert cov[-1] >= cov[0] and cov[-1] - cov[-3] < 0.1, f"coverage should rise then saturate: {cov}"
    print(f"[ok] interior F1 optimum at n={curve[arg]['n_claims']} (F1 {f1[arg]:.3f}); coverage saturates")


def test_judge_overlaps_not_vacuous() -> None:
    """The lenient judge must be INFORMATIVE but OVERLAPPING (faithful and unfaithful confidences mix), or
    every downstream trade-off is vacuous. A balanced judge separates the classes near-perfectly — the
    vacuity guard that justifies the lenient choice."""
    corpus = _corpus()
    panel = build_panel(corpus)
    conf = panel_confidence(panel, JUDGE)
    y = panel["Y"].ravel().astype(bool)
    auc = auc_pooled(conf.ravel(), y.astype(float))
    assert 0.75 < auc < 0.97, f"lenient judge AUC should be informative-but-imperfect: {auc}"
    sup_conf, hal_conf = conf.ravel()[y], conf.ravel()[~y]
    assert hal_conf.max() > sup_conf.min(), "faithful/unfaithful confidences must OVERLAP (else vacuous)"
    auc_bal = auc_pooled(panel_confidence(panel, JUDGE_BALANCED).ravel(), y.astype(float))
    assert auc_bal > auc, f"a balanced judge should separate better than lenient: {auc_bal} vs {auc}"
    print(f"[ok] judge overlaps (not vacuous): lenient AUC {auc:.3f} < balanced AUC {auc_bal:.3f}; "
          f"confidences overlap")


def test_naive_faithfulness_biased_rogan_gladen_corrects() -> None:
    """The naive (judge-mean) faithfulness is a BIASED estimate of the oracle supported fraction; the
    Rogan-Gladen correction with the audited rates moves it closer to the truth."""
    corpus = _corpus()
    panel = build_panel(corpus)
    conf = panel_confidence(panel)
    rg = panel_corrected_faithfulness(panel, conf)
    assert abs(rg["p_obs"] - rg["oracle"]) > 0.03, f"naive should be biased: {rg}"
    assert abs(rg["corrected"] - rg["oracle"]) < abs(rg["p_obs"] - rg["oracle"]), "RG should reduce bias"
    print(f"[ok] Rogan-Gladen: naive {rg['p_obs']:.3f} (oracle {rg['oracle']:.3f}) -> "
          f"corrected {rg['corrected']:.3f}")


def test_raw_judge_overconfident_recalibration_lowers_ece() -> None:
    """The raw judge confidence is over-confident (high ECE); Platt and isotonic both lower it, and Platt
    preserves the ranking exactly (AUC unchanged) — calibration buys efficiency, not validity."""
    corpus = _corpus()
    panel = build_panel(corpus)
    conf = panel_confidence(panel)
    cal = panel_calibration(panel, conf)
    assert cal["ece_raw"] > 0.1, f"raw judge should be over-confident: ECE {cal['ece_raw']}"
    assert cal["ece_platt"] < cal["ece_raw"] and cal["ece_iso"] < cal["ece_raw"], "recalibration should lower ECE"
    assert abs(cal["auc_platt"] - cal["auc_raw"]) < 1e-9, "Platt must preserve the ranking (AUC) exactly"
    print(f"[ok] calibration: raw ECE {cal['ece_raw']:.3f} -> Platt {cal['ece_platt']:.3f} / "
          f"isotonic {cal['ece_iso']:.3f}; AUC preserved")


def test_crc_monotone_and_controls() -> None:
    """The conformal risk control machinery (imported): the false-claim loss is monotone non-increasing in
    the cut, the CRC threshold controls the expected test false-claim rate at alpha, and the naive
    fraction-of-retained loss is NON-monotone (the counterexample that voids CRC)."""
    corpus = _corpus()
    panel = build_panel(corpus)
    conf = calibrated_panel_conf(panel, panel_confidence(panel), split_panel(panel["Y"].shape[0])[0])
    L = loss_matrix(conf, panel["Y"], LAMBDA_GRID, k=panel["n_claims"])
    assert np.all(np.diff(L.mean(axis=0)) <= 1e-12), "mean false-claim loss must be non-increasing in the cut"
    crc = crc_backoff(panel, conf, alpha=0.15)
    assert crc["test_false_claim_rate"] <= crc["alpha"] + 0.10, f"CRC should control the risk: {crc}"
    # the naive fraction loss is not monotone
    frac = np.array([np.mean([fraction_loss(conf[i], panel["Y"][i], lam) for i in range(conf.shape[0])])
                     for lam in LAMBDA_GRID])
    assert np.any(np.diff(frac) > 1e-9), "the fraction-of-retained loss should be non-monotone (counterexample)"
    print(f"[ok] CRC: monotone false-claim loss, realized {crc['test_false_claim_rate']:.3f} <= "
          f"alpha {crc['alpha']}+slack; fraction loss non-monotone")


def test_back_off_matches_conformal() -> None:
    """Reuse anchor: the imported `back_off_retained` over an answer's confidences returns exactly the
    claims at or above the cut — the per-claim back-off is the prereq's machinery, not a fork."""
    corpus = _corpus()
    panel = build_panel(corpus)
    conf = panel_confidence(panel)
    tau = 0.6
    for qi in range(min(5, conf.shape[0])):
        bo = back_off_retained(conf[qi], tau)
        expect = np.array(sorted(np.flatnonzero(conf[qi] >= tau).tolist()))
        assert np.array_equal(bo["retained"], expect), "back_off_retained must match the cut"
    print("[ok] back-off reuse: back_off_retained == {claims with conf >= tau}")


def test_abstention_frontier_monotone() -> None:
    """Raising the confidence cut trades coverage for faithfulness: retained precision rises (weakly) and
    retained coverage falls — the generation precision-recall frontier."""
    corpus = _corpus()
    panel = build_panel(corpus)
    conf = calibrated_panel_conf(panel, panel_confidence(panel), split_panel(panel["Y"].shape[0])[0])
    fr = abstention_frontier(panel, conf)
    lo, hi = fr[5], fr[-6]                                   # away from the degenerate tail
    assert hi["faithfulness"] >= lo["faithfulness"] - 0.02, f"precision should rise with the cut: {lo} {hi}"
    assert hi["coverage"] <= lo["coverage"] + 0.02, f"coverage should fall with the cut: {lo} {hi}"
    print(f"[ok] frontier: cut up -> faithfulness {lo['faithfulness']:.2f}->{hi['faithfulness']:.2f}, "
          f"coverage {lo['coverage']:.2f}->{hi['coverage']:.2f}")


def test_bits_sign_split() -> None:
    """Bits-of-grounding: supported claims carry POSITIVE pointwise MI with the context, hallucinated
    claims carry NON-POSITIVE bits — the information-theoretic face of faithfulness (imported PMI)."""
    corpus = _corpus()
    panel = build_panel(corpus)
    bits = panel_bits(corpus, panel)
    assert bits["mean_sup"] > 0.0 > bits["mean_hal"], f"sign split should hold: {bits}"
    assert bits["mean_sup"] - bits["mean_hal"] > 0.3, f"the bit gap should be clear: {bits}"
    assert bits["frac_hal_nonpos"] > 0.5, f"most hallucinations should have non-positive bits: {bits}"
    print(f"[ok] bits sign split: supported {bits['mean_sup']:.3f} > 0 > hallucinated {bits['mean_hal']:.3f} "
          f"({bits['frac_hal_nonpos']:.2f} of hallucinations <= 0)")


def test_single_claim_bernoulli() -> None:
    """Degenerate anchor: a one-claim answer collapses faithfulness to a Bernoulli — it is 0 or 1 exactly,
    and coverage is the single fact's indicator over the context size."""
    corpus = _corpus()
    for q in range(corpus["n_queries"]):
        a = generate_answer(corpus, q, 1, 1.0, 1.0, seed=9)
        assert answer_faithfulness(a) in (0.0, 1.0), "single-claim faithfulness must be Bernoulli"
    print("[ok] degenerate: a one-claim answer's faithfulness is 0 or 1")


def test_viz_constants_reproducible() -> None:
    """Bake-only-reproducible: two builds of the frontier and the bits split are bit-identical (every draw
    is seeded; no random start vector leaks into a baked number)."""
    corpus = _corpus()
    p1 = build_panel(corpus); p2 = build_panel(corpus)
    c1, c2 = panel_confidence(p1), panel_confidence(p2)
    assert np.array_equal(c1, c2), "the judge confidence panel is not reproducible"
    assert verbosity_curve(corpus) == verbosity_curve(corpus), "the verbosity curve is not reproducible"
    print("[ok] reproducible: seeded panel + verbosity curve are bit-identical across runs")


def test_guards() -> None:
    """Defensive guards (the gemini-prone cases): an empty answer, an empty context, and the imported
    metric denominators all return sane sentinels rather than dividing by zero."""
    corpus = _corpus()
    empty = generate_answer(corpus, 0, 0, 1.0, 1.0, seed=0)
    assert answer_faithfulness(empty) == 0.0 and answer_coverage(empty) == 0.0
    assert recall_at_k([], set(), 3) == 0.0 and precision_at_k([], set(), 0) == 0.0
    print("[ok] guards: empty answer / empty context / empty metric denominators are safe")


def _run_all() -> None:
    test_perfect_judge_collapse()
    test_faithfulness_is_precision_coverage_is_recall()
    test_oracle_separation()
    test_faithfulness_coverage_diverge()
    test_interior_f1_optimum()
    test_judge_overlaps_not_vacuous()
    test_naive_faithfulness_biased_rogan_gladen_corrects()
    test_raw_judge_overconfident_recalibration_lowers_ece()
    test_crc_monotone_and_controls()
    test_back_off_matches_conformal()
    test_abstention_frontier_monotone()
    test_bits_sign_split()
    test_single_claim_bernoulli()
    test_viz_constants_reproducible()
    test_guards()


if __name__ == "__main__":
    print("faithfulness_groundedness: running tests\n")
    _run_all()
    print("\nDemo:")
    faithfulness_demo()
    print("\nviz_constants (mirror into FaithfulnessGroundednessLaboratory.tsx):")
    viz_constants()
    print("\nall checks passed.")
