"""The Retrieval Problem — the reference implementation for the formalRAG root topic.

Retrieval is ranking by a relevance functional rel(q, d); this module makes the
topic's geometric claims executable. Core similarity uses only numpy. The
verification harness asserts every pedagogical claim: the cosine-distance identity
||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>; rank invariance under any strictly monotone
transform of the score; the keystone equivalence that on the unit sphere the
cosine, dot-product, and (ascending) Euclidean rankings coincide; the off-sphere
divergence (dot and cosine disagree on top-1 when norms vary, and scaling a
document's magnitude flips the dot ranking but not the cosine ranking); the metric
axioms for Euclidean distance; the cosine-distance triangle-inequality counterexample
(with its reported violation gap); the dot product's failure of identity-of-
indiscernibles; and the hyperplane / sphere / cone classification of equal-score
loci. finance_demo() owns the decimals that SimilarityGeometryLaboratory.tsx and the
topic prose mirror.

Run:  uv run --with numpy python notebooks/the-retrieval-problem/the_retrieval_problem.py
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Similarity primitives — the three scores retrieval actually uses
# --------------------------------------------------------------------------- #

def dot(a: np.ndarray, b: np.ndarray) -> float:
    """The inner product <a, b>."""
    return float(np.dot(a, b))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance ||a - b|| — a dissimilarity (rank ascending)."""
    return float(np.linalg.norm(a - b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity <a, b> / (||a|| ||b||); guarded against a (near-)zero norm."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize(v: np.ndarray) -> np.ndarray:
    """Project a nonzero vector onto the unit sphere."""
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n


def rank(query: np.ndarray, corpus: dict[str, np.ndarray], score, *, descending: bool) -> list[str]:
    """argtopk over the whole corpus: document ids ordered by `score(query, doc)`.

    descending=True for similarities (dot, cosine); False for the Euclidean
    distance, where the nearest document ranks first.
    """
    scored = [(name, score(query, vec)) for name, vec in corpus.items()]
    scored.sort(key=lambda kv: kv[1], reverse=descending)
    return [name for name, _ in scored]


# --------------------------------------------------------------------------- #
# The finance corpus — a deterministic 2-D stand-in whose norms vary by design.
#
# q is the query direction ("interest rate exposure"), a unit vector. Each
# document's DIRECTION is what cosine sees; its NORM is the extra magnitude the
# dot product additionally rewards. "filing-onpoint" is concise and best aligned
# with q (so it wins on cosine); "transcript-pad" is a long, padded transcript
# whose larger norm hijacks the dot-product ranking. This is the magnitude
# sensitivity that BM25's length normalization later removes. The numbers below
# are the pedagogical payload SimilarityGeometryLaboratory.tsx mirrors to the
# decimal — change one here, change it there, and re-run this file.
# --------------------------------------------------------------------------- #

_Q = np.array([1.0, 0.0])
_CORPUS: dict[str, np.ndarray] = {
    "filing-onpoint": np.array([0.95, 0.20]),   # concise, best-aligned, small norm
    "transcript-pad": np.array([1.70, 1.30]),   # padded, large norm, less aligned
    "news-macro":     np.array([0.80, 0.55]),   # medium alignment, medium norm
    "filing-fx":      np.array([0.30, 0.90]),   # off-direction (FX), small norm
}
# d-star is the document whose magnitude the laboratory's slider scales.
_DSTAR = "filing-onpoint"


# --------------------------------------------------------------------------- #
# Verification harness — every pedagogical claim, made executable
# --------------------------------------------------------------------------- #

def test_cosine_distance_identity() -> None:
    """Theorem 1: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>, in any dimension."""
    rng = np.random.default_rng(0)
    for d in (2, 3, 8, 64):
        for _ in range(200):
            a, b = rng.standard_normal(d), rng.standard_normal(d)
            lhs = float(np.dot(a - b, a - b))
            rhs = float(np.dot(a, a) + np.dot(b, b) - 2 * np.dot(a, b))
            assert abs(lhs - rhs) < 1e-9
    print("  [ok] cosine-distance identity ||a-b||^2 = ||a||^2+||b||^2-2<a,b> (d up to 64)")


def test_rank_invariant_under_monotone() -> None:
    """Proposition 1: a strictly increasing transform of the score preserves the ranking."""
    base = rank(_Q, _CORPUS, cosine, descending=True)
    transforms = (
        lambda s: s + 5.0,
        lambda s: 3.0 * s,
        lambda s: np.exp(s),
        lambda s: np.log(s + 2.0),   # strictly increasing on the score range here
        lambda s: s ** 3,            # strictly increasing (scores are real)
    )
    for phi in transforms:
        order = rank(_Q, _CORPUS, lambda q, v: float(phi(cosine(q, v))), descending=True)
        assert order == base, f"monotone transform changed the ranking: {order} != {base}"
    print("  [ok] ranking is invariant under strictly monotone transforms of the score")


def test_on_sphere_rankings_coincide() -> None:
    """Theorem 2 (keystone): on the unit sphere, descending-cosine, descending-dot,
    and ascending-Euclidean rankings are identical."""
    sphere = {name: normalize(v) for name, v in _CORPUS.items()}
    q = normalize(_Q)
    by_cos = rank(q, sphere, cosine, descending=True)
    by_dot = rank(q, sphere, dot, descending=True)
    by_euc = rank(q, sphere, euclidean, descending=False)
    assert by_cos == by_dot == by_euc, f"rankings differ on the sphere: {by_cos} {by_dot} {by_euc}"

    # Dimension independence: the same coincidence in higher dimensions.
    rng = np.random.default_rng(1)
    for d in (3, 16, 128):
        q_h = normalize(rng.standard_normal(d))
        docs = {f"d{i}": normalize(rng.standard_normal(d)) for i in range(12)}
        c = rank(q_h, docs, cosine, descending=True)
        dd = rank(q_h, docs, dot, descending=True)
        e = rank(q_h, docs, euclidean, descending=False)
        assert c == dd == e, f"sphere coincidence failed in d={d}"
    print("  [ok] on the unit sphere the cosine, dot, and Euclidean rankings coincide (d up to 128)")


def test_off_sphere_dot_cosine_disagree() -> None:
    """The off-sphere divergence: with norms varying, dot and cosine disagree on top-1."""
    assert rank(_Q, _CORPUS, dot, descending=True)[0] == "transcript-pad"
    assert rank(_Q, _CORPUS, cosine, descending=True)[0] == "filing-onpoint"
    # The disagreement runs deeper than the top: the full orders differ.
    assert rank(_Q, _CORPUS, dot, descending=True) != rank(_Q, _CORPUS, cosine, descending=True)
    print("  [ok] off the sphere: dot top-1 = transcript-pad, cosine top-1 = filing-onpoint")


def test_magnitude_scaling_flips_dot() -> None:
    """Proposition 2: scaling d-star's magnitude lifts its dot rank but leaves cosine fixed."""
    scaled = dict(_CORPUS)
    scaled[_DSTAR] = 2.0 * _CORPUS[_DSTAR]            # scale norm, keep direction
    assert rank(_Q, scaled, dot, descending=True)[0] == _DSTAR        # dot top-1 flips to d-star
    assert rank(_Q, scaled, cosine, descending=True)[0] == _DSTAR     # cosine already had it #1
    # The exact threshold at which the dot top-1 flips from transcript-pad to d-star.
    c_star = dot(_Q, _CORPUS["transcript-pad"]) / dot(_Q, _CORPUS[_DSTAR])
    below = dict(_CORPUS); below[_DSTAR] = (c_star - 0.05) * _CORPUS[_DSTAR]
    above = dict(_CORPUS); above[_DSTAR] = (c_star + 0.05) * _CORPUS[_DSTAR]
    assert rank(_Q, below, dot, descending=True)[0] == "transcript-pad"
    assert rank(_Q, above, dot, descending=True)[0] == _DSTAR
    print(f"  [ok] scaling d-star flips the dot ranking at c* = {c_star:.4f}; cosine is unmoved")


def test_cosine_scale_invariance_and_dot_linearity() -> None:
    """Proposition 2, both halves: cosine ignores magnitude; the dot product is linear in it."""
    d = _CORPUS[_DSTAR]
    base_cos, base_dot = cosine(_Q, d), dot(_Q, d)
    for c in (0.3, 1.0, 2.5, 10.0):
        assert abs(cosine(_Q, c * d) - base_cos) < 1e-12
        assert abs(dot(_Q, c * d) - c * base_dot) < 1e-12
    print("  [ok] cosine is scale-invariant; the dot product is linear in magnitude")


def test_euclidean_is_metric() -> None:
    """Proposition 3: Euclidean distance satisfies all four metric axioms."""
    rng = np.random.default_rng(2)
    pts = [rng.standard_normal(5) for _ in range(40)]
    for a in pts:
        assert euclidean(a, a) < 1e-12                                    # identity of indiscernibles
        for b in pts:
            assert euclidean(a, b) >= 0.0                                 # non-negativity
            assert abs(euclidean(a, b) - euclidean(b, a)) < 1e-12         # symmetry
            if not np.allclose(a, b):
                assert euclidean(a, b) > 0.0
            for c in pts:
                assert euclidean(a, c) <= euclidean(a, b) + euclidean(b, c) + 1e-9  # triangle inequality
    print("  [ok] Euclidean distance is a metric (non-negativity, identity, symmetry, triangle)")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine(a, b)


def test_cosine_distance_violates_triangle() -> float:
    """Proposition 4: 1 - cos is not a metric. Explicit three-point counterexample,
    returning the violation gap the topic quotes."""
    x = np.array([1.0, 0.0])
    y = np.array([1.0, 1.0])
    z = np.array([0.0, 1.0])
    direct = cosine_distance(x, z)
    detour = cosine_distance(x, y) + cosine_distance(y, z)
    gap = direct - detour
    assert direct > detour, "expected a triangle-inequality violation"
    assert abs(gap - (np.sqrt(2.0) - 1.0)) < 1e-9   # gap = 1 - 2(1 - 1/sqrt2) = sqrt(2) - 1
    print(f"  [ok] cosine distance violates the triangle inequality by {gap:.4f} (= sqrt(2)-1)")
    return gap


def test_dot_product_not_a_metric() -> None:
    """Proposition 5: the dot product is not a distance. It fails identity-of-indiscernibles
    (<a,a> = ||a||^2 != 0) and ranks in the opposite sense (larger = more similar)."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        a = rng.standard_normal(4)
        if np.linalg.norm(a) > 1e-6:
            assert abs(dot(a, a)) > 1e-9          # a "distance" from a point to itself is nonzero
    # A document need not be its own maximizer: a longer aligned vector scores higher.
    a = np.array([1.0, 0.0]); longer = np.array([2.0, 0.0])
    assert dot(a, longer) > dot(a, a)
    print("  [ok] the dot product is not a metric (no identity of indiscernibles; self not maximal)")


def test_level_sets() -> None:
    """Proposition 6: equal-score loci are a hyperplane (dot), a sphere (Euclidean),
    and a cone (cosine). Verified by sampling points on each locus through d-star."""
    q = _Q
    dstar = _CORPUS[_DSTAR]
    rng = np.random.default_rng(4)

    # Hyperplane {x : <q, x> = c}: with q = e_1, this is the vertical line x_1 = c.
    c_dot = dot(q, dstar)
    for _ in range(50):
        x = np.array([c_dot, rng.standard_normal()])     # any x_2, fixed x_1 = c
        assert abs(dot(q, x) - c_dot) < 1e-9

    # Sphere {x : ||q - x|| = r}: points at fixed radius r about q.
    r = euclidean(q, dstar)
    for _ in range(50):
        theta = rng.uniform(0, 2 * np.pi)
        x = q + r * np.array([np.cos(theta), np.sin(theta)])
        assert abs(euclidean(q, x) - r) < 1e-9

    # Cone {x : cos(q, x) = c}: rays from the origin at angle +/- arccos(c) to q.
    c_cos = cosine(q, dstar)
    angle = np.arccos(np.clip(c_cos, -1.0, 1.0))
    for sgn in (+1.0, -1.0):
        direction = np.array([np.cos(sgn * angle), np.sin(sgn * angle)])
        for t in (0.2, 1.0, 3.7):
            assert abs(cosine(q, t * direction) - c_cos) < 1e-9
    print("  [ok] level sets: dot -> hyperplane, Euclidean -> sphere, cosine -> cone")


# --------------------------------------------------------------------------- #
# Finance demo — owns the decimals the viz and prose mirror
# --------------------------------------------------------------------------- #

def finance_demo() -> None:
    print(f"  query q = {_Q.tolist()}  (unit direction 'interest rate exposure')")
    print(f"  {'document':<16}{'norm':>8}{'dot':>9}{'cosine':>9}{'euclid':>9}")
    for name, v in _CORPUS.items():
        print(f"  {name:<16}{np.linalg.norm(v):>8.4f}{dot(_Q, v):>9.4f}"
              f"{cosine(_Q, v):>9.4f}{euclidean(_Q, v):>9.4f}")
    print(f"  dot ranking    : {rank(_Q, _CORPUS, dot, descending=True)}")
    print(f"  cosine ranking : {rank(_Q, _CORPUS, cosine, descending=True)}")
    sphere = {n: normalize(v) for n, v in _CORPUS.items()}
    print(f"  on the sphere  : {rank(normalize(_Q), sphere, cosine, descending=True)}"
          f"  (cosine = dot = Euclidean)")
    c_star = dot(_Q, _CORPUS['transcript-pad']) / dot(_Q, _CORPUS[_DSTAR])
    print(f"  dot top-1 flips to {_DSTAR} once its magnitude exceeds c* = {c_star:.4f}")


def test_finance_demo_numbers() -> None:
    """Pin the exact decimals SimilarityGeometryLaboratory.tsx bakes; drift fails here."""
    norms = {n: round(float(np.linalg.norm(v)), 4) for n, v in _CORPUS.items()}
    assert norms == {"filing-onpoint": 0.9708, "transcript-pad": 2.1401,
                     "news-macro": 0.9708, "filing-fx": 0.9487}, norms
    cosines = {n: round(cosine(_Q, v), 4) for n, v in _CORPUS.items()}
    assert cosines == {"filing-onpoint": 0.9785, "transcript-pad": 0.7944,
                       "news-macro": 0.8240, "filing-fx": 0.3162}, cosines
    dots = {n: round(dot(_Q, v), 4) for n, v in _CORPUS.items()}
    assert dots == {"filing-onpoint": 0.95, "transcript-pad": 1.70,
                    "news-macro": 0.80, "filing-fx": 0.30}, dots
    c_star = round(dot(_Q, _CORPUS["transcript-pad"]) / dot(_Q, _CORPUS[_DSTAR]), 4)
    assert c_star == 1.7895, c_star
    print("  [ok] finance-demo decimals pinned (norms, cosines, dots, flip threshold c*)")


if __name__ == "__main__":
    print("The Retrieval Problem — verification harness")
    test_cosine_distance_identity()
    test_rank_invariant_under_monotone()
    test_on_sphere_rankings_coincide()
    test_off_sphere_dot_cosine_disagree()
    test_magnitude_scaling_flips_dot()
    test_cosine_scale_invariance_and_dot_linearity()
    test_euclidean_is_metric()
    test_cosine_distance_violates_triangle()
    test_dot_product_not_a_metric()
    test_level_sets()
    test_finance_demo_numbers()
    print("Finance demo:")
    finance_demo()
    print("All checks passed.")
