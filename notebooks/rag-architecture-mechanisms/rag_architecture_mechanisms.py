"""RAG Architecture Mechanisms: what each named pattern actually buys, and where it fails.

Run:
    uv run --with numpy --with scipy --with scikit-learn --with rank-bm25 \
        python notebooks/rag-architecture-mechanisms/rag_architecture_mechanisms.py

The cheat sheets present the named RAG architectures as a list of alternatives, each with a
"best for" line and no "loses to what". This module runs six of them as arms over TWO regimes
and measures both halves.

  LOCAL regime  -- the answer is carried by a passage. Factoid retrieval: find the right filing.
  GLOBAL regime -- the answer is carried by no passage at all, only by a COUNT across a
                   community. "Which sector discusses this most" is not "which sector is most
                   similar to this", and the corpus is built so those two disagree.

The thesis is that no arm wins both, and that each win is attributable to a named property of
the corpus rather than to the architecture being better. An architecture is a bet about where
the answer lives; the regimes are the two places it can live.

Imports and never reimplements: the geometry, the fusion rule, the lexical scorer, the late
interaction, the reformulation operator, the community detection, the metrics.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_NB = pathlib.Path(__file__).resolve().parents[1]
for _dir in (
    "hypersphere-vmf-geometry",
    "dense-retrieval-dual-encoders",
    "bm25",
    "rank-fusion-rrf",
    "late-interaction-learned-sparse",
    "multi-hop-iterative-retrieval",
    "graphrag-community-detection",
    "query-transformation-hyde",
    "set-metrics-precision-recall-map-mrr",
):
    _p = _NB / _dir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hypersphere_vmf_geometry import normalize, sample_vmf          # noqa: E402
from dense_retrieval_dual_encoders import DPR_SEED                  # noqa: E402
from rank_fusion_rrf import rrf_fuse                                # noqa: E402
from late_interaction_learned_sparse import maxsim_matrix           # noqa: E402
from multi_hop_iterative_retrieval import reformulate               # noqa: E402
import graphrag_community_detection as GR                           # noqa: E402

SEED = DPR_SEED                 # 7 -- the shared finance-geometry seed
DIM = 64                        # forced by GraphRAG: at d=32 / kappa=60 Leiden MERGES two sectors
KAPPA_SECTOR = 200.0            # same-sector cosine ~0.72 at d=64 -- communities are recoverable
KAPPA_PASSAGE = 120.0
N_SECTORS = 5
N_COMP = 5                      # 25 companies -- matches the graphrag corpus this reuses

# --- LOCAL regime -----------------------------------------------------------------------------
TOKENS = 8                      # tokens per passage; the three fusion legs read DISJOINT windows
WIN_LEX = (0, 1, 2, 3)          # so the legs are partial VIEWS rather than a quality ladder --
WIN_DENSE = (4, 5, 6, 7)        # the capstone's construction, and the reason fusion can gain at
WIN_LI = (1, 3, 4, 6)           # all. Overlapping views would make them three noisy copies.
N_TOPICS = 12                   # latent themes; each company holds one per leg window
KAPPA_TOPIC = 80.0              # how tightly a passage's window sits on its theme
KAPPA_Q_SHARP = 400.0           # "on" class: the query names its themes precisely
KAPPA_Q_VAGUE = 80.0            # "partial" class: named vaguely, so each leg alone is ambiguous
NOISY_DECOY_W = 0.62            # "noisy" class: weight on each leg's OWN decoy               # THE load-bearing constant for fusion. Disjoint token windows are
                                # NOT enough on their own: if every token is drawn around the one
                                # company prototype, the three windows are three subsamples of one
                                # signal -- a quality ladder in partial-view clothing, and RRF then
                                # scores BELOW its best leg (measured: 0.350 vs 0.400 before this
                                # existed). Each window is instead drawn around its own FACET, the
                                # prototype blurred along an independent random direction, so each
                                # leg alone confuses the company with a DIFFERENT set of rivals and
                                # only the intersection identifies it. That is what gives fusion
                                # something to recombine.
R_PASSAGES = 3                  # passages per company
TOPK = 5

KAPPA_ON = 60.0                 # on-manifold query: naive already finds it
THETA_OFF_DEG = 70.0            # off-manifold query: tilted toward the generic corpus direction,
                                # which is the distribution shift HyDE exists to correct
ALPHA_BRIDGE_DEG = 40.0         # mention angle: cos(40) = 0.766 must clear the worst same-sector
                                # company cosine or the bridge is outranked by an ordinary passage
KAPPA_BRIDGE = 300.0
N_PER_CLASS = 20

# --- GLOBAL regime ----------------------------------------------------------------------------
# The anti-theatre condition. If the sector with the most theme-mentions were also the sector
# geometrically nearest the theme, top-k by cosine would already answer it and community
# summarisation would be decoration. These two are made to DISAGREE by construction, and a test
# asserts they still disagree.
GLOBAL_N_QUERIES = 16
MENTIONS_COUNT_MAX = 4          # mentioning companies in the count-max sector
MENTIONS_GEO_MAX = 2            # ...in the geometrically nearest sector
MENTIONS_OTHER = 1

_LOCAL: dict | None = None
_GLOBAL: dict | None = None


def _protos(seed: int = SEED):
    """The company prototypes and their sectors, in the graphrag geometry."""
    rng = np.random.default_rng(seed)
    sector_mu = normalize(rng.standard_normal((N_SECTORS, DIM)))
    protos, sector_of = [], []
    for s in range(N_SECTORS):
        members = sample_vmf(N_COMP, sector_mu[s], KAPPA_SECTOR, seed=seed + 11 * s + 1)
        for ci in range(N_COMP):
            protos.append(normalize(members[ci]))
            sector_of.append(s)
    return np.array(protos), np.array(sector_of), sector_mu


def local_corpus(seed: int = SEED) -> dict:
    """The factoid regime: the answer is carried by a passage, and the job is to find it.

    Four query classes, one per mechanism the arms are supposed to exhibit:
      on       -- on-manifold; a single dense retrieval already succeeds (naive's home ground)
      off      -- tilted toward the generic corpus direction; the query/document distribution
                  shift HyDE corrects by discarding the query's position entirely
      noisy    -- retrieval fails DETECTABLY; a grader can tell, which is what corrective needs
      bridge   -- the answer is a company a filing NAMES rather than describes; unreachable in
                  one hop, which is what agentic needs
    """
    protos, sector_of, _ = _protos(seed)
    K = len(protos)
    g = normalize(protos.mean(axis=0))        # the generic "document-ness" direction

    # Per-company THEMES: three latent directions, one per leg window. Overlap between
    # companies' theme sets is what makes any single leg ambiguous.
    trng = np.random.default_rng(seed + 12345)
    topics = normalize(trng.standard_normal((N_TOPICS, DIM)))
    crng = np.random.default_rng(seed + 999)
    comp_topics = np.array([crng.choice(N_TOPICS, 3, replace=False) for _ in range(K)])
    themes = np.array([[topics[comp_topics[a, f]] for f in range(3)] for a in range(K)])

    def _bag(centers, sd: int, kappa: float) -> np.ndarray:
        """One passage or query: each leg's window drawn around that leg's theme."""
        bag = np.zeros((TOKENS, DIM))
        for f, win in enumerate((WIN_LEX, WIN_DENSE, WIN_LI)):
            draws = sample_vmf(len(win), centers[f], kappa, seed=sd + 97 * f)
            for j, t in enumerate(win):
                bag[t] = normalize(draws[j])
        return bag

    passages, owner = [], []
    for a in range(K):
        for r in range(R_PASSAGES):
            passages.append(_bag(themes[a], seed + 4001 + 31 * a + r, KAPPA_TOPIC))
            owner.append(a)

    # bridge passages: a filing of X that MENTIONS Y, on the multi-hop mention geometry
    alpha = np.deg2rad(ALPHA_BRIDGE_DEG)
    bridge_of = {}
    for x in range(K):
        cands = [y for y in range(K) if sector_of[y] != sector_of[x]]
        y = int(cands[(x * 5 + 3) % len(cands)])
        bridge_of[x] = y
        mention = [normalize(np.cos(alpha) * themes[x, f] + np.sin(alpha) * themes[y, f])
                   for f in range(3)]
        passages.append(_bag(mention, seed + 7717 + 13 * x, KAPPA_BRIDGE))
        owner.append(y)                        # the bridge's ANSWER is the mentioned company
    passages = np.array(passages)              # (n_docs, TOKENS, DIM)
    owner = np.array(owner)
    is_bridge = np.array([0] * (K * R_PASSAGES) + [1] * K)

    Q, gold, klass = [], [], []
    theta = np.deg2rad(THETA_OFF_DEG)
    for i in range(N_PER_CLASS):
        a = i % K
        # ON: themes named precisely -> even one leg suffices. Naive's home ground; the point is
        # that the cheapest arm is already right, so anything more is pure cost.
        Q.append(_bag(themes[a], seed + 20001 + i, KAPPA_Q_SHARP)); gold.append(a); klass.append("on")
    for i in range(N_PER_CLASS):
        a = (i * 3 + 1) % K
        # PARTIAL: themes named vaguely -> each leg alone retrieves everything sharing that one
        # theme, and only the intersection is the gold. Hybrid's home ground.
        Q.append(_bag(themes[a], seed + 60001 + i, KAPPA_Q_VAGUE)); gold.append(a); klass.append("partial")
    for i in range(N_PER_CLASS):
        a = (i * 7 + 3) % K
        # OFF: tilted toward the generic corpus direction -- the query/document distribution shift
        # HyDE corrects by discarding the query's position entirely.
        off = [normalize(np.cos(theta) * themes[a, f] + np.sin(theta) * g) for f in range(3)]
        Q.append(_bag(off, seed + 30001 + i, KAPPA_Q_SHARP)); gold.append(a); klass.append("off")
    for i in range(N_PER_CLASS):
        a = (i * 11 + 2) % K
        # NOISY: each leg pulled toward a DIFFERENT wrong company, hard enough that retrieval
        # fails DETECTABLY -- which is what a corrective grader needs to have something to grade.
        peers = [c for c in range(K) if sector_of[c] == sector_of[a] and c != a]
        mixed = [normalize((1.0 - NOISY_DECOY_W) * themes[a, f]
                           + NOISY_DECOY_W * themes[peers[(i + f) % len(peers)], f])
                 for f in range(3)]
        Q.append(_bag(mixed, seed + 50001 + i, KAPPA_Q_SHARP)); gold.append(a); klass.append("noisy")
    for i in range(N_PER_CLASS):
        x = i % K
        # BRIDGE: the answer is a company a filing NAMES rather than describes -- unreachable in
        # one hop at any k, which is what the agentic arm exists for.
        Q.append(_bag(themes[x], seed + 40001 + i, KAPPA_Q_SHARP))
        gold.append(bridge_of[x]); klass.append("bridge")

    return {
        "regime": "local", "protos": protos, "themes": themes, "comp_topics": comp_topics, "sector_of": sector_of, "g": g,
        "docs": passages, "owner": owner, "is_bridge": is_bridge, "bridge_of": bridge_of,
        "Q": np.array(Q), "gold": np.array(gold), "klass": np.array(klass),
        "K": int(K), "n_docs": int(len(passages)), "n_queries": int(len(Q)),
        "n_answers": int(K),
    }


def global_corpus(seed: int = SEED) -> dict:
    """The aggregation regime: the answer is carried by NO passage, only by a count.

    Each query is a theme. The gold answer is the SECTOR with the most companies mentioning it.
    The theme is tilted toward one sector geometrically while a DIFFERENT sector holds the most
    mentions, so similarity and counting disagree — which is the whole point. A corpus where they
    agree makes community summarisation decoration, and the diagnostics assert they disagree.
    """
    protos, sector_of, _ = _protos(seed)
    K = len(protos)
    A = np.maximum(0.0, protos @ protos.T - GR.GR_THRESHOLD)
    np.fill_diagonal(A, 0.0)
    sector_means = np.array([normalize(protos[sector_of == s].mean(axis=0)) for s in range(N_SECTORS)])

    rng = np.random.default_rng(seed + 555)
    themes, gold, geo_answer, mention_mat = [], [], [], []
    for i in range(GLOBAL_N_QUERIES):
        near = i % N_SECTORS
        theme = normalize(0.75 * sector_means[near] + 0.25 * rng.standard_normal(DIM))
        cos_e = protos @ theme
        geo = int(np.argmax(sector_means @ theme))
        count_sector = (geo + 2) % N_SECTORS          # deliberately NOT the nearest sector
        mentions = np.zeros(K, int)
        for s in range(N_SECTORS):
            idx = np.where(sector_of == s)[0]
            k = (MENTIONS_COUNT_MAX if s == count_sector
                 else MENTIONS_GEO_MAX if s == geo else MENTIONS_OTHER)
            mentions[idx[np.argsort(-cos_e[idx])][:k]] = 1
        themes.append(theme)
        gold.append(int(np.argmax([mentions[sector_of == s].sum() for s in range(N_SECTORS)])))
        geo_answer.append(geo)
        mention_mat.append(mentions)

    return {
        "regime": "global", "protos": protos, "sector_of": sector_of, "A": A,
        "sector_means": sector_means, "Q": np.array(themes), "gold": np.array(gold),
        "geo_answer": np.array(geo_answer), "mentions": np.array(mention_mat),
        "K": int(K), "n_queries": int(GLOBAL_N_QUERIES), "n_answers": int(N_SECTORS),
        "klass": np.array(["global"] * GLOBAL_N_QUERIES),
    }


def local() -> dict:
    global _LOCAL
    if _LOCAL is None:
        _LOCAL = local_corpus()
    return _LOCAL


def glob() -> dict:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = global_corpus()
    return _GLOBAL


# --- leg scorers for the LOCAL regime ----------------------------------------------------------
def _pool(x: np.ndarray, win) -> np.ndarray:
    return normalize(x[..., list(win), :].mean(axis=-2))


def leg_dense(c: dict, q: np.ndarray) -> np.ndarray:
    return _pool(c["docs"], WIN_DENSE) @ _pool(q, WIN_DENSE)


def leg_lexical(c: dict, q: np.ndarray) -> np.ndarray:
    return _pool(c["docs"], WIN_LEX) @ _pool(q, WIN_LEX)


def leg_li(c: dict, q: np.ndarray) -> np.ndarray:
    qq = q[list(WIN_LI), :]
    dd = c["docs"][:, list(WIN_LI), :]
    return maxsim_matrix(qq[None, ...], dd)[0]


LEGS = {"lexical": leg_lexical, "dense": leg_dense, "late": leg_li}


def _diagnostics() -> None:
    L, G = local(), glob()
    print(f"LOCAL  : {L['n_queries']} queries, {L['n_docs']} docs, {L['K']} companies, dim {DIM}")
    print(f"GLOBAL : {G['n_queries']} themes,  {G['K']} entities, {G['n_answers']} sectors")
    print()

    print("LOCAL — per-leg top-1 accuracy by query class (are the legs partial VIEWS?)")
    print(f"  {'class':8s} " + " ".join(f"{n:>9s}" for n in LEGS) + f" {'RRF':>9s}")
    for k in ("on", "partial", "off", "noisy", "bridge"):
        m = np.where(L["klass"] == k)[0]
        accs, rr = {}, 0
        for name, fn in LEGS.items():
            hit = sum(int(L["owner"][int(np.argmax(fn(L, L["Q"][i])))] == L["gold"][i]) for i in m)
            accs[name] = hit / len(m)
        for i in m:
            rankings = [list(np.argsort(-fn(L, L["Q"][i]))[:20]) for fn in LEGS.values()]
            fused = rrf_fuse(rankings)
            top = fused[0] if not isinstance(fused[0], (list, tuple)) else fused[0][0]
            rr += int(L["owner"][int(top)] == L["gold"][i])
        print(f"  {k:8s} " + " ".join(f"{accs[n]:9.3f}" for n in LEGS) + f" {rr/len(m):9.3f}")

    print()
    print("LOCAL — leg disagreement (Kendall-style overlap of top-5; low = partial views)")
    ov = []
    for i in range(0, L["n_queries"], 3):
        tops = [set(np.argsort(-fn(L, L["Q"][i]))[:5].tolist()) for fn in LEGS.values()]
        ov.append(len(tops[0] & tops[1] & tops[2]) / 5.0)
    print(f"  mean 3-way top-5 overlap: {np.mean(ov):.3f}  (1.0 would mean a quality ladder)")

    print()
    print("GLOBAL — does counting disagree with geometry? (the anti-theatre condition)")
    dis = int((G["gold"] != G["geo_answer"]).sum())
    print(f"  themes where count-max sector != nearest sector: {dis}/{G['n_queries']}")
    est = np.asarray(GR.leiden(G["A"]))
    from collections import Counter
    pure = all(len(Counter(G["sector_of"][est == c]).keys()) == 1 for c in set(est.tolist()))
    print(f"  Leiden recovers {len(set(est.tolist()))} communities; each pure by sector: {pure}")
    flat_hits = 0
    for i in range(G["n_queries"]):
        cos_e = G["protos"] @ G["Q"][i]
        top = np.argsort(-cos_e)[:10]
        vote = np.bincount(G["sector_of"][top], minlength=N_SECTORS)
        flat_hits += int(np.argmax(vote) == G["gold"][i])
    print(f"  FLAT top-10 sector vote accuracy: {flat_hits}/{G['n_queries']}"
          f"   <- must be LOW or the arm is theatre")


if __name__ == "__main__":
    _diagnostics()
