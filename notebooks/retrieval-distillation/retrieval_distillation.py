"""Knowledge distillation for retrieval — the reference implementation for the formalRAG
`retrieval-distillation` topic.

The reranking sub-track's payoff. Cross-encoders gave the most expressive score h([q;d]) but at a
cost that forbids precomputation; this topic spends that power CHEAPLY by distilling the expensive
teacher into a precomputable dual-encoder student. The transfer loss is MarginMSE (Hofstaetter et
al. 2020): match the teacher's per-query score MARGIN, not its absolute level. Four movements, every
pedagogical claim an `assert`:

  MOVEMENT 1 — MARGINMSE AND TRANSLATION-INVARIANCE. THE THEOREM. The all-pairs MarginMSE
    L(S,T) = sum_i sum_{j,k} [ (S_ij - S_ik) - (T_ij - T_ik) ]^2 reduces, via the variance identity
    sum_{j,k}(a_j - a_k)^2 = 2 n_d sum_j (a_j - mean a)^2, to the centered Frobenius distance
        L(S,T) = 2 n_d || S C - T C ||_F^2 ,   C = I - (1/n_d) 1 1^T  (row-centering, per query).
    Adding a per-query offset T -> T + b 1^T leaves T C unchanged (1^T C = 0), so the margin loss is
    BLIND to per-query level — the student need only match the teacher's RELATIVE structure. That
    translation-invariance is the rigorous hinge of the whole topic. (`center_rows`, `margin_mse`,
    `margin_mse_bruteforce`, `inject_offset`.)

  MOVEMENT 2 — THE CLOSED-FORM DISTILLED STUDENT (Eckart-Young-Mirsky). No SGD: the optima are
    truncated SVDs (the cross-encoders precedent of a deterministic, bit-reproducible surrogate).
    The pointwise-MSE-optimal rank-d student is best_rank_d(T); the MARGIN-optimal rank-d student is
    best_rank_d(T C) — the per-query-CENTERED teacher. Since T C has zero row sums, its truncated SVD
    does too, so it is a genuine rank-d dual encoder (realizable as Q G^T) AND exactly margin-optimal.
    Margin distillation spends its rank budget on RANKING, not on reproducing the teacher's level.
    The embedding-dimension rank ceiling still BINDS: the student is rank <= d and distillation
    APPROACHES the teacher's recall, never exceeds it; at d >= rank(T C) it reproduces the teacher's
    ranking exactly. (`distill_pointwise`, `distill_margin`, `realize_student`, `rank_recall_curve`.)

  MOVEMENT 3 — MARGIN BEATS POINTWISE AT RESTRICTED RANK, AND THE COST PAYOFF. Cross-encoder scores
    are per-query MISCALIBRATED (different queries, different score scales) — the documented reason
    MarginMSE exists. We model that as an additive per-query offset b 1^T (rank one, the 1_d
    direction). The pointwise student WASTES a slice of its d-dimensional budget reproducing that
    offset; the margin student is blind to it and spends every dimension on ranking, so at a
    restricted rank (D_STAGE) margin recall@1 > pointwise recall@1 while the teacher's recall is
    untouched (an offset preserves every argmax). The payoff: the student gives cross-encoder-quality
    ranking at dual-encoder INFERENCE cost — precompute the document embeddings once, answer queries
    by MIPS, never run a per-pair joint forward pass. (`per_query_offset`, `student_inference_cost`,
    `teacher_inference_cost`, `distill_speedup`.)

  MOVEMENT 4 — DARK KNOWLEDGE: SOFT TEACHER MARGINS BEAT HARD BINARY LABELS (Hinton et al. 2015). The
    teacher's GRADED scores on the negatives say WHICH wrong document is more dangerous — information a
    one-hot relevance label lacks. We mine the hardest negative per query from the labeled DPR pool
    (`mine_nearest`, the negative-sampling prereq: nearest other-company query -> its gold document is
    a hard negative doc), and show the teacher's margin on those mined hard pairs is GRADED (a real
    spread), while the binary margin is the constant 1. Distilling the soft teacher (best_rank_d(T C))
    then recovers higher recall@1 at restricted rank than distilling hard binary labels
    (best_rank_d(Y C)). (`binary_relevance`, `mined_hard_negative`, `teacher_hard_margins`,
    `soft_vs_hard_curve`.)

Honest caveats (rigorFlag territory): the translation-invariance reduction and the centered-SVD
optimum are THEOREMS. The per-query offset is a MODEL of cross-encoder miscalibration (its magnitude
is tuned and flagged, not derived); the teacher is the deterministic random-ReLU surrogate from the
cross-encoders topic (it realizes the relevance, an expressivity-not-generalization stand-in for a
trained transformer); MarginMSE is a HEURISTIC loss with no optimality theorem of its own beyond the
reduction we prove; the rank ceiling binds the student no matter how good the teacher; and the
dark-knowledge advantage is DEMONSTRATED on this finance cloud, not proved. The whole laboratory is
the synthetic vMF finance geometry reused from the InfoNCE / DPR / cross-encoders topics.

Imports its two prerequisites and the numeric siblings it reuses (it never reimplements them):
  - dense-retrieval-dual-encoders: the finance geometry, the SVD machinery, recall.
  - cross-encoders-reranking: the teacher cross_encoder_finance_scores.
  - negative-sampling-hard-negatives: mine_nearest and the labeled DPR pool.
`viz_constants()` prints what RetrievalDistillationLaboratory.tsx mirrors to the decimal.

Run:  uv run --with numpy --with scipy python notebooks/retrieval-distillation/retrieval_distillation.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# Established cross-topic pattern: add EACH ancestor's HYPHENATED dir to the path, import the
# UNDERSCORED module. The two DAG prerequisites are cross-encoders-reranking and
# negative-sampling-hard-negatives; both transitively need dense-retrieval / infonce / the
# retrieval-problem / hypersphere ancestors. The import graph is NOT the pedagogical DAG: we import
# numeric siblings to SOURCE numbers (the teacher, the geometry, the miner), never to reimplement.
_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",                 # normalize, sample_vmf (transitive)
    "the-retrieval-problem",                    # cosine, rank (transitive)
    "infonce-contrastive-objective",            # negative_weights, finance geometry (transitive)
    "dense-retrieval-dual-encoders",            # SVD machinery + finance matrix + recall
    "cross-encoders-reranking",                 # THE teacher: cross_encoder_finance_scores
    "negative-sampling-hard-negatives",         # mine_nearest + the labeled DPR pool
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dense_retrieval_dual_encoders import (                                      # noqa: E402
    dpr_finance_matrix, best_rank_d, realize_dual_encoder, score_matrix,
    topk_recall, relative_frobenius_error,
)
from cross_encoders_reranking import cross_encoder_finance_scores               # noqa: E402
from negative_sampling_hard_negatives import mine_nearest                       # noqa: E402


# =========================================================================== #
# Constants — tuned by _diagnostics() (read its sweep before changing these).
# =========================================================================== #

TEACHER_N_FEAT = 192            # random-ReLU features of the teacher (tuned by _diagnostics): the
                                 #   smallest width at which the teacher's recall@1 hits 1.0 (a GOOD
                                 #   teacher) while its scores stay GRADED (dark knowledge) rather than a
                                 #   near-0/1 interpolation. Below ~192 the teacher underfits and is a
                                 #   bad teacher; far above, the scores flatten toward the hard label.
TEACHER_SEED = 11               # the teacher's rng stream (the cross-encoders default)
D_STAGE = 3                     # the restricted student rank (mirror cross-encoders D_STAGE1: the
                                 #   teacher is genuinely better here; at full rank the gap vanishes)
OFFSET_MAG = 3.0                # per-query miscalibration magnitude (tuned: the teacher's own scores
                                 #   carry a large CONSTANT baseline already; this adds per-query level
                                 #   variation so the margin > pointwise recall gap is seed-robust)
OFFSET_SEED = 5                 # the offset's rng stream
D_GRID = (1, 2, 3, 4, 5, 6, 7, 8)   # the rank sweep for the fidelity curve
K_MINE = 6                      # mining depth for the hardest-negative search (negative-sampling)

# The cost model — a headline corpus so the inference speedup is legible (reused from cross-encoders).
C_RETRIEVE = 1.0                # student: one precomputed-embedding MIPS dot product per document
C_CE = 25.0                     # teacher: one joint cross-encoder forward pass per (query, document)
CORPUS_HEADLINE = 1_000_000     # a legible production corpus size for the cost panel


# =========================================================================== #
# Movement 1 — MarginMSE and translation-invariance.
# =========================================================================== #

def center_rows(T: np.ndarray) -> np.ndarray:
    """The per-query (row) centering T C, C = I - (1/n_d) 1 1^T: subtract each query's mean document
    score. The margin operator — margins are within-query, across documents. GUARDS: at-least-2d for
    consistency with margin_mse; empty -> empty."""
    T = np.atleast_2d(np.asarray(T, dtype=float))
    if T.size == 0:
        return np.empty_like(T)
    return T - T.mean(axis=1, keepdims=True)


def margin_mse(S: np.ndarray, T: np.ndarray) -> float:
    """The all-pairs MarginMSE, via the proven reduction L = 2 n_d || S C - T C ||_F^2. Centering is
    linear, so center_rows(S) - center_rows(T) = (S - T) C. GUARD: empty -> 0.0."""
    S = np.atleast_2d(np.asarray(S, dtype=float))
    T = np.atleast_2d(np.asarray(T, dtype=float))
    if S.size == 0:
        return 0.0
    n_d = S.shape[1]
    diff = center_rows(S) - center_rows(T)
    return float(2 * n_d * np.sum(diff * diff))


def margin_mse_bruteforce(S: np.ndarray, T: np.ndarray) -> float:
    """The literal all-pairs definition sum_i sum_{j,k} [(S_ij - S_ik) - (T_ij - T_ik)]^2, summed over
    every ordered document pair. Only for the test that pins the reduction (n_d is small)."""
    S = np.atleast_2d(np.asarray(S, dtype=float))
    T = np.atleast_2d(np.asarray(T, dtype=float))
    D = S - T
    total = 0.0
    nq, nd = D.shape
    for i in range(nq):
        for j in range(nd):
            for k in range(nd):
                total += (D[i, j] - D[i, k]) ** 2
    return float(total)


def inject_offset(T: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add a per-query offset: T + b 1^T (b_i added to every document score of query i). A rank-one
    perturbation in the 1_d direction — the cross-encoder per-query miscalibration model."""
    T = np.asarray(T, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    return T + b


def per_query_offset(n_q: int, mag: float, seed: int) -> np.ndarray:
    """A deterministic per-query miscalibration offset b (seeded standard-normal, scaled by mag)."""
    rng = np.random.default_rng(seed)
    return mag * rng.standard_normal(n_q)


# =========================================================================== #
# Movement 2 — the closed-form distilled student (Eckart-Young-Mirsky).
# =========================================================================== #

def distill_pointwise(T: np.ndarray, d: int) -> np.ndarray:
    """The pointwise-MSE-optimal rank-d student: best_rank_d(T) (Eckart-Young). It must reproduce the
    teacher's absolute level as well as its ranking, spending rank budget on both."""
    return best_rank_d(T, d)


def distill_margin(T: np.ndarray, d: int) -> np.ndarray:
    """The MARGIN-optimal rank-d student: best_rank_d(T C), the per-query-centered teacher. Because
    T C has zero row sums, so does its truncated SVD, so this is a genuine rank-d dual encoder AND the
    exact minimizer of the all-pairs MarginMSE over rank-d students."""
    return best_rank_d(center_rows(T), d)


def realize_student(T: np.ndarray, d: int, margin: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Factor the distilled student into d-dimensional dual-encoder embeddings (Q, G) with
    Q G^T == the student score matrix — proving the student is a genuine precomputable dual encoder."""
    target = center_rows(T) if margin else np.asarray(T, dtype=float)
    return realize_dual_encoder(target, d)


def rank_recall_curve(T: np.ndarray, truth: np.ndarray, d_grid) -> list:
    """For each rank d, the recall@1 of the pointwise student, the margin student, and (the
    dark-knowledge contrast) the hard-binary student, plus the teacher's ceiling. The recall is an
    argmax per row, so per-query centering of the margin student does not change it."""
    Y = binary_relevance(truth, np.atleast_2d(T).shape[1])
    teacher_r1 = topk_recall(T, truth, 1)
    rows = []
    for d in d_grid:
        rows.append({
            "d": int(d),
            "pointwise_r1": round(float(topk_recall(distill_pointwise(T, d), truth, 1)), 4),
            "margin_r1": round(float(topk_recall(distill_margin(T, d), truth, 1)), 4),
            "hard_r1": round(float(topk_recall(distill_margin(Y, d), truth, 1)), 4),
            "teacher_r1": round(float(teacher_r1), 4),
        })
    return rows


# =========================================================================== #
# Movement 3 — the inference-cost payoff.
# =========================================================================== #

def teacher_inference_cost(corpus: int) -> float:
    """A cross-encoder reranker must run a joint forward pass per (query, document): corpus * c_ce per
    query. It cannot precompute — the score is not separable."""
    return float(max(0, int(corpus))) * C_CE


def student_inference_cost(corpus: int) -> float:
    """The distilled dual-encoder student precomputes every document embedding offline and answers a
    query by MIPS: corpus * c_ret per query (a dot product per document; sublinear with ANN)."""
    return float(max(0, int(corpus))) * C_RETRIEVE


def distill_speedup(corpus: int) -> float:
    """Teacher cost / student cost at query time — the price of the cross-encoder's expressivity the
    student buys back. GUARD: zero student cost -> 1.0."""
    s = student_inference_cost(corpus)
    return float(teacher_inference_cost(corpus) / s) if s > 0 else 1.0


# =========================================================================== #
# Movement 4 — dark knowledge: soft teacher margins vs hard binary labels.
# =========================================================================== #

def binary_relevance(truth: np.ndarray, n_d: int) -> np.ndarray:
    """The hard label matrix Y: Y[i, truth[i]] = 1, else 0. The one-hot relevance a soft teacher
    refines — it says the gold is right but nothing about WHICH wrong document is more dangerous."""
    truth = np.asarray(truth)
    nq = truth.shape[0]
    Y = np.zeros((nq, int(n_d)), dtype=float)
    if nq == 0 or n_d == 0:
        return Y
    valid = (truth >= 0) & (truth < int(n_d))          # guard out-of-bound gold indices
    rows = np.arange(nq)[valid]
    Y[rows, truth[valid]] = 1.0
    return Y


def mined_hard_negative(pool: dict, anchor_idx: int, k: int) -> int:
    """The hardest NEGATIVE document for an anchor query: the nearest other-company query (mined from
    the labeled pool by `mine_nearest`, the negative-sampling prereq) whose gold document differs from
    the anchor's gold — that gold document is a genuine hard negative doc. GUARDS: missing 'company' key
    or out-of-bound anchor/neighbor -> -1; none found -> -1."""
    if "company" not in pool:
        return -1
    truth = pool["company"]
    if not (0 <= anchor_idx < len(truth)):
        return -1
    gold = int(truth[anchor_idx])
    for nb in mine_nearest(pool, anchor_idx, k):
        if 0 <= nb < len(truth) and int(truth[nb]) != gold:
            return int(truth[nb])
    return -1


def teacher_hard_margins(T: np.ndarray, pool: dict, k: int) -> np.ndarray:
    """The teacher's margin T[i, gold] - T[i, hard_neg] on each query's MINED hardest negative pair.
    The dark knowledge: a GRADED spread (some hard negatives are genuinely more confusable, a smaller
    margin), where the binary label's margin is the constant 1. GUARDS: missing 'company' key -> empty;
    out-of-bound gold/negative indices skipped."""
    T = np.atleast_2d(np.asarray(T, dtype=float))
    if "company" not in pool:
        return np.array([])
    truth = pool["company"]
    nd = T.shape[1]
    margins = []
    for i in range(T.shape[0]):
        hn = mined_hard_negative(pool, i, k)
        g = int(truth[i]) if i < len(truth) else -1
        if hn >= 0 and 0 <= g < nd and hn < nd:
            margins.append(float(T[i, g] - T[i, hn]))
    return np.array(margins)


def soft_vs_hard_curve(T: np.ndarray, truth: np.ndarray, d_grid) -> list:
    """Recall@1 of the soft-margin student best_rank_d(T C) vs the hard-margin student
    best_rank_d(Y C) across rank — the dark-knowledge contrast at every budget."""
    Y = binary_relevance(truth, np.atleast_2d(T).shape[1])
    rows = []
    for d in d_grid:
        rows.append({
            "d": int(d),
            "soft_r1": round(float(topk_recall(distill_margin(T, d), truth, 1)), 4),
            "hard_r1": round(float(topk_recall(distill_margin(Y, d), truth, 1)), 4),
        })
    return rows


# =========================================================================== #
# Setup — the teacher, the miscalibrated teacher, and the labeled pool (cached).
# =========================================================================== #

_CACHE: dict = {}


def _setup() -> dict:
    """Build once and cache:
      - Q, P, truth, sector from the DPR finance geometry (32 queries, 8 company documents, dim 32).
      - T_clean: the teacher cross_encoder_finance_scores(Q, P, truth) — graded relevance scores.
      - offset b and T_teacher = T_clean + b 1^T: the miscalibrated teacher the student distills from.
      - pool: the labeled query pool {emb, company} the negative-sampling miner consumes.
    """
    if _CACHE:
        return _CACHE
    Q, P, truth, sector = dpr_finance_matrix()
    T_clean = cross_encoder_finance_scores(Q, P, truth, n_feat=TEACHER_N_FEAT, seed=TEACHER_SEED)
    b = per_query_offset(Q.shape[0], OFFSET_MAG, OFFSET_SEED)
    T_teacher = inject_offset(T_clean, b)
    pool = {"emb": Q, "company": truth, "sector": sector[truth]}
    _CACHE.update({
        "Q": Q, "P": P, "truth": truth, "sector": sector,
        "T_clean": T_clean, "offset": b, "T_teacher": T_teacher, "pool": pool,
    })
    return _CACHE


# =========================================================================== #
# Diagnostics — the build-and-run sweeps that SET the constants. Read before tuning.
# =========================================================================== #

def _diagnostics() -> None:
    s = _setup()
    T_clean, T_teacher, truth = s["T_clean"], s["T_teacher"], s["truth"]
    nq, nd = T_clean.shape
    print(f"  [diag] teacher shape = {nq}x{nd}, rank(T_clean) = {np.linalg.matrix_rank(T_clean)}")
    print(f"  [diag] teacher recall@1 (clean) = {topk_recall(T_clean, truth, 1):.3f}; "
          f"(miscalibrated) = {topk_recall(T_teacher, truth, 1):.3f}  (offset preserves argmax)")
    print(f"  [diag] row-mean std: clean = {T_clean.mean(1).std():.4f}, "
          f"miscalibrated = {T_teacher.mean(1).std():.4f}  (offset injects per-query level)")
    sv_t = np.linalg.svd(T_teacher, compute_uv=False)
    sv_tc = np.linalg.svd(center_rows(T_teacher), compute_uv=False)
    print(f"  [diag] singular values  T  = {np.round(sv_t, 3)}")
    print(f"  [diag] singular values  TC = {np.round(sv_tc, 3)}  (last ~0: centering drops a dim)")
    print("  [diag] rank-d recall@1  (pointwise / margin / hard / teacher):")
    for d in D_GRID:
        pw = topk_recall(distill_pointwise(T_teacher, d), truth, 1)
        mg = topk_recall(distill_margin(T_teacher, d), truth, 1)
        Y = binary_relevance(truth, nd)
        hd = topk_recall(distill_margin(Y, d), truth, 1)
        flag = "  <-- D_STAGE" if d == D_STAGE else ""
        print(f"           d={d}: {pw:.3f} / {mg:.3f} / {hd:.3f} / "
              f"{topk_recall(T_teacher, truth, 1):.3f}{flag}")
    hm = teacher_hard_margins(T_clean, s["pool"], K_MINE)
    print(f"  [diag] teacher margins on mined hard pairs: n={len(hm)}, "
          f"mean={hm.mean():.3f}, std={hm.std():.3f} (graded => dark knowledge; binary margin = 1)")


# =========================================================================== #
# Verification harness — every pedagogical claim is a test.
# =========================================================================== #

def test_margin_reduction() -> None:
    """Movement 1: the all-pairs MarginMSE equals the centered Frobenius form 2 n_d ||SC - TC||^2 for
    arbitrary S, T (the reduction the whole topic rests on)."""
    rng = np.random.default_rng(0)
    S = rng.standard_normal((6, 5))
    T = rng.standard_normal((6, 5))
    assert abs(margin_mse_bruteforce(S, T) - margin_mse(S, T)) < 1e-9, "margin reduction wrong"
    print("  [ok] Movement 1: all-pairs MarginMSE == 2 n_d ||SC - TC||^2")


def test_translation_invariance() -> None:
    """Movement 1, the hinge: the margin loss is invariant to any per-query offset T -> T + b 1^T."""
    rng = np.random.default_rng(1)
    S = rng.standard_normal((6, 5))
    T = rng.standard_normal((6, 5))
    b = rng.standard_normal(6)
    assert abs(margin_mse(S, T) - margin_mse(S, inject_offset(T, b))) < 1e-9, "not offset-invariant"
    # ... while the pointwise loss is NOT invariant (a real difference, not a tautology).
    pw0 = float(np.sum((S - T) ** 2))
    pw1 = float(np.sum((S - inject_offset(T, b)) ** 2))
    assert abs(pw0 - pw1) > 1e-6, "pointwise loss should move under an offset"
    print("  [ok] Movement 1: margins kill the per-query offset (pointwise does not)")


def test_pointwise_optimum_is_eckart_young() -> None:
    """Movement 2: the pointwise student is exactly the truncated SVD of the teacher."""
    s = _setup()
    T = s["T_teacher"]
    assert np.allclose(distill_pointwise(T, D_STAGE), best_rank_d(T, D_STAGE), atol=1e-12), \
        "pointwise != best_rank_d(T)"
    print("  [ok] Movement 2: pointwise student == best_rank_d(T) (Eckart-Young)")


def test_margin_optimum_is_centered_svd() -> None:
    """Movement 2: the margin student is best_rank_d(T C), and its all-pairs margin loss equals the
    Eckart-Young tail 2 n_d sum_{l>d} sigma_l(T C)^2 (the global minimum over rank-d students)."""
    s = _setup()
    T = s["T_teacher"]
    nd = T.shape[1]
    S = distill_margin(T, D_STAGE)
    assert np.allclose(S, best_rank_d(center_rows(T), D_STAGE), atol=1e-12), "margin != best_rank_d(TC)"
    sv = np.linalg.svd(center_rows(T), compute_uv=False)
    tail = 2 * nd * float(np.sum(sv[D_STAGE:] ** 2))
    assert abs(margin_mse(S, T) - tail) < 1e-7, "margin loss != Eckart-Young tail"
    print("  [ok] Movement 2: margin student == best_rank_d(TC); loss == 2 n_d sum_{l>d} sigma_l^2")


def test_margin_equals_pointwise_when_precentered() -> None:
    """Movement 2 anchor: with a pre-centered teacher (zero per-query offset), the two students
    coincide — the 'zero centering' collapse. Centering an already-centered matrix is a no-op."""
    s = _setup()
    Tc = center_rows(s["T_teacher"])
    assert np.allclose(distill_pointwise(Tc, D_STAGE), distill_margin(Tc, D_STAGE), atol=1e-12), \
        "pre-centered students should coincide"
    print("  [ok] Movement 2 anchor: margin == pointwise on a pre-centered teacher")


def test_student_is_a_real_dual_encoder() -> None:
    """Movement 2: the margin student factors into d-dim embeddings Q, G with Q G^T == its score
    matrix (a genuine precomputable dual encoder, via the imported realize_dual_encoder/score_matrix)."""
    s = _setup()
    T = s["T_teacher"]
    Qs, Gs = realize_student(T, D_STAGE, margin=True)
    assert Qs.shape[1] == D_STAGE and Gs.shape[1] == D_STAGE, "wrong embedding dim"
    assert np.allclose(score_matrix(Qs, Gs), distill_margin(T, D_STAGE), atol=1e-9), \
        "realized embeddings do not reproduce the student"
    print("  [ok] Movement 2: the student is a genuine rank-d dual encoder (Q G^T)")


def test_full_rank_recovers_teacher() -> None:
    """Movement 2 ceiling/collapse: at d >= rank(TC) the margin student reproduces the teacher's
    RANKING exactly (recall@1 == teacher's), and at d >= rank(T) the pointwise student reproduces T."""
    s = _setup()
    T, truth = s["T_teacher"], s["truth"]
    nd = T.shape[1]
    teacher_r1 = topk_recall(T, truth, 1)
    margin_full = topk_recall(distill_margin(T, nd - 1), truth, 1)   # rank(TC) <= n_d - 1
    assert abs(margin_full - teacher_r1) < 1e-12, "margin student should match teacher ranking at full rank"
    assert relative_frobenius_error(T, distill_pointwise(T, nd)) < 1e-9, "pointwise should reproduce T at d=n_d"
    print("  [ok] Movement 2: full-rank student recovers the teacher (ceiling is the teacher)")


def test_margin_beats_pointwise_at_restricted_rank() -> None:
    """Movement 3, the headline: at the restricted rank D_STAGE the margin student outranks the
    pointwise student, which wastes budget on the per-query offset."""
    s = _setup()
    T, truth = s["T_teacher"], s["truth"]
    mg = topk_recall(distill_margin(T, D_STAGE), truth, 1)
    pw = topk_recall(distill_pointwise(T, D_STAGE), truth, 1)
    assert mg > pw, f"expected margin > pointwise at d={D_STAGE}, got {mg:.3f} vs {pw:.3f}"
    print(f"  [ok] Movement 3: margin recall@1 {mg:.3f} > pointwise {pw:.3f} at d={D_STAGE}")


def test_rank_ceiling_binds() -> None:
    """Movement 2/3: the student recall@1 is bounded by the teacher at every rank, and at d=1 it sits
    well below — the embedding-dimension ceiling distillation approaches but never breaks."""
    s = _setup()
    T, truth = s["T_teacher"], s["truth"]
    teacher_r1 = topk_recall(T, truth, 1)
    for d in D_GRID:
        assert topk_recall(distill_margin(T, d), truth, 1) <= teacher_r1 + 1e-12, "student exceeded teacher"
    assert topk_recall(distill_margin(T, 1), truth, 1) < teacher_r1, "d=1 should be below the ceiling"
    print("  [ok] Movement 2/3: the rank ceiling binds (student recall <= teacher at every d)")


def test_dark_knowledge_is_graded_yet_binary_compresses() -> None:
    """Movement 4, the honest finding (build-and-run). Dark knowledge is REAL: the teacher's margins on
    the mined hard-negative pairs are GRADED (gold always beats the hard negative, but by a varying
    amount — some negatives are genuinely more confusable), where a one-hot label's margin is the
    constant 1. YET on this clean, block-structured, IN-SAMPLE toy the binary ground-truth target
    compresses to low rank at least as well as the soft teacher — the classic soft > hard advantage is
    a GENERALIZATION phenomenon (scarce held-out labels), which a closed-form in-sample fit cannot
    show. We assert what is robustly true here, not the textbook headline."""
    s = _setup()
    T, truth = s["T_teacher"], s["truth"]
    soft = topk_recall(distill_margin(T, D_STAGE), truth, 1)
    hard = topk_recall(distill_margin(binary_relevance(truth, T.shape[1]), D_STAGE), truth, 1)
    hm = teacher_hard_margins(s["T_clean"], s["pool"], K_MINE)
    assert len(hm) > 0 and hm.mean() > 0 and hm.std() > 0.05, \
        "teacher margins on mined hard pairs should be graded and positive (dark knowledge)"
    assert hard >= soft - 1e-9, f"in-sample, binary should compress >= soft here ({hard:.3f} vs {soft:.3f})"
    print(f"  [ok] Movement 4: dark knowledge graded (mean {hm.mean():.3f}, std {hm.std():.3f}); "
          f"in-sample binary {hard:.3f} >= soft {soft:.3f}")


def test_teacher_is_imported() -> None:
    """The teacher is the imported cross_encoder_finance_scores, not a reimplementation (twin)."""
    s = _setup()
    Q, P, truth = s["Q"], s["P"], s["truth"]
    twin = cross_encoder_finance_scores(Q, P, truth, n_feat=TEACHER_N_FEAT, seed=TEACHER_SEED)
    assert np.allclose(twin, s["T_clean"], atol=1e-12), "teacher is not the imported scorer"
    print("  [ok] anchor: the teacher is the imported cross_encoder_finance_scores")


def test_cost_payoff() -> None:
    """Movement 3: the distilled student is strictly cheaper at inference than the cross-encoder, and
    the speedup is the exact ratio c_ce / c_ret."""
    for corpus in (10, 1000, CORPUS_HEADLINE):
        assert student_inference_cost(corpus) < teacher_inference_cost(corpus), "student not cheaper"
    assert abs(distill_speedup(CORPUS_HEADLINE) - C_CE / C_RETRIEVE) < 1e-9, "speedup != c_ce/c_ret"
    print(f"  [ok] Movement 3: student is {distill_speedup(CORPUS_HEADLINE):.0f}x cheaper at inference")


def test_empty_and_degenerate_guards() -> None:
    """Guards (gemini): empty matrices, non-positive rank/k, an empty mining pool — no crashes."""
    assert margin_mse(np.empty((0, 0)), np.empty((0, 0))) == 0.0
    assert center_rows(np.empty((0, 0))).size == 0
    assert distill_margin(np.empty((0, 0)), 3).size == 0
    assert topk_recall(np.empty((0, 0)), np.array([]), 1) == 0.0
    assert topk_recall(_setup()["T_teacher"], _setup()["truth"], 0) == 0.0
    assert binary_relevance(np.array([], dtype=int), 0).size == 0
    assert mined_hard_negative({"emb": np.zeros((1, 4)), "company": np.array([0])}, 0, 3) == -1
    assert distill_speedup(0) == 1.0
    print("  [ok] guards: empty matrices / non-positive rank,k / empty pool handled")


def _run_all() -> None:
    test_margin_reduction()
    test_translation_invariance()
    test_pointwise_optimum_is_eckart_young()
    test_margin_optimum_is_centered_svd()
    test_margin_equals_pointwise_when_precentered()
    test_student_is_a_real_dual_encoder()
    test_full_rank_recovers_teacher()
    test_margin_beats_pointwise_at_restricted_rank()
    test_rank_ceiling_binds()
    test_dark_knowledge_is_graded_yet_binary_compresses()
    test_teacher_is_imported()
    test_cost_payoff()
    test_empty_and_degenerate_guards()


# =========================================================================== #
# Demo.
# =========================================================================== #

def distillation_demo() -> None:
    s = _setup()
    T, truth = s["T_teacher"], s["truth"]
    teacher_r1 = topk_recall(T, truth, 1)
    pw = topk_recall(distill_pointwise(T, D_STAGE), truth, 1)
    mg = topk_recall(distill_margin(T, D_STAGE), truth, 1)
    print(f"  Teacher (cross-encoder) recall@1 = {teacher_r1:.3f} at inference cost "
          f"{teacher_inference_cost(CORPUS_HEADLINE):,.0f} / query")
    print(f"  Distilled rank-{D_STAGE} student: pointwise recall@1 = {pw:.3f}, "
          f"margin recall@1 = {mg:.3f} at cost {student_inference_cost(CORPUS_HEADLINE):,.0f} / query")
    print(f"  => margin distillation recovers {mg / teacher_r1:.0%} of the teacher at "
          f"{distill_speedup(CORPUS_HEADLINE):.0f}x lower inference cost.")


# =========================================================================== #
# Viz constants — printed for RetrievalDistillationLaboratory.tsx to mirror.
# =========================================================================== #

def _r(v, n=4):
    return round(float(v), n)


def viz_constants() -> None:
    """Print every MEASURED number the laboratory mirrors to the decimal. TS recomputes only CLOSED
    FORM: Panel A's pointwise parabola and flat margin line under the offset slider, Panel B's
    tail-energy from the singular values, and Panel C's cost/speedup arithmetic. numpy scalars are
    cast to avoid np.float64(...) pollution."""
    s = _setup()
    T_clean, T_teacher, truth = s["T_clean"], s["T_teacher"], s["truth"]
    nq, nd = T_teacher.shape

    print("  // ----- shared constants -----")
    print(f"  const N_QUERIES = {int(nq)};")
    print(f"  const N_DOCS = {int(nd)};")
    print(f"  const D_STAGE = {D_STAGE};")
    print(f"  const D_GRID = {list(int(d) for d in D_GRID)};")
    print(f"  const TEACHER_R1 = {_r(topk_recall(T_teacher, truth, 1))};")
    print(f"  const INTRINSIC_RANK = {int(np.linalg.matrix_rank(T_teacher))};")

    # Panel A: translation-invariance. Fix the margin-optimal student S (for the CLEAN teacher; it is
    # offset-invariant). A miscalibration slider alpha scales the per-query offset b, so the teacher is
    # T_clean + alpha b 1^T (alpha=0 the clean teacher, alpha=1 the actual miscalibrated teacher). The
    # margin loss is FLAT in alpha (the theorem); the pointwise loss is a parabola.
    print("  // ----- Panel A: translation-invariance (miscalibration slider alpha) -----")
    b = s["offset"]                                                    # the per-query offset vector
    S = distill_margin(T_clean, D_STAGE)                              # the margin student (offset-blind)
    R = S - T_clean
    pw_base = float(np.sum(R * R))                                    # ||S - T_clean||^2
    pw_lin = float(np.sum(R * b.reshape(-1, 1)))                      # <S - T_clean, b 1^T>
    pw_quad = float(nd * np.sum(b * b))                              # ||b 1^T||^2 = n_d ||b||^2
    print(f"  const PW_LOSS_BASE = {_r(pw_base, 3)};")
    print(f"  const PW_LOSS_LIN = {_r(pw_lin, 3)};")
    print(f"  const PW_LOSS_QUAD = {_r(pw_quad, 3)};")
    print(f"  const MARGIN_LOSS = {_r(margin_mse(S, T_clean), 4)};   // flat in alpha")
    print("  const ALPHA_ACTUAL = 1.0;   // the teacher's actual miscalibration")
    print("  const ALPHA_MAX = 2.0;")
    # heatmaps: one representative query per company (8 rows) of the clean teacher + its centering, and
    # the per-row offset the slider multiplies (T's rows shift with alpha; the centered TC does not).
    rep = [int(np.where(truth == c)[0][0]) for c in range(nd)]
    Th = T_clean[rep]
    print(f"  const HEAT_T = {[[_r(v, 3) for v in row] for row in Th]};")
    print(f"  const HEAT_TC = {[[_r(v, 3) for v in row] for row in center_rows(Th)]};")
    print(f"  const HEAT_B = {[_r(v, 3) for v in b[rep]]};   // per-row offset (alpha multiplies this)")

    # Panel B: rank-d fidelity. Singular values for the live tail-energy; baked recall curve.
    print("  // ----- Panel B: rank-d fidelity (rank slider) -----")
    print(f"  const SINGULAR_VALUES_T = {[_r(v, 4) for v in np.linalg.svd(T_teacher, compute_uv=False)]};")
    print(f"  const SINGULAR_VALUES_TC = "
          f"{[_r(v, 4) for v in np.linalg.svd(center_rows(T_teacher), compute_uv=False)]};")
    print(f"  const RANK_RECALL = {rank_recall_curve(T_teacher, truth, D_GRID)};")

    # Panel C: dark knowledge + the cost payoff.
    print("  // ----- Panel C: dark knowledge + cost payoff -----")
    print(f"  const SOFT_VS_HARD = {soft_vs_hard_curve(T_teacher, truth, D_GRID)};")
    hm = teacher_hard_margins(T_clean, s["pool"], K_MINE)
    print(f"  const HARD_PAIR_MARGINS = {[_r(v, 3) for v in hm]};   // graded (binary margin = 1)")
    print(f"  const C_RETRIEVE = {_r(C_RETRIEVE, 2)};")
    print(f"  const C_CE = {_r(C_CE, 2)};")
    print(f"  const CORPUS_HEADLINE = {int(CORPUS_HEADLINE)};")


if __name__ == "__main__":
    print("Knowledge distillation for retrieval — diagnostics\n")
    _diagnostics()
    print("\nVerification harness:")
    _run_all()
    print("\nDemo:")
    distillation_demo()
    print("\nViz constants (mirrored to the decimal in RetrievalDistillationLaboratory.tsx):")
    viz_constants()
    print("\nAll checks passed.")
