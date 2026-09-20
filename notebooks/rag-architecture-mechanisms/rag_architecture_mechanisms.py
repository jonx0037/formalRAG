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
from scipy.special import expit

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
from query_transformation_hyde import (                             # noqa: E402
    generation_center, hallucination_targets, hyde_centroid, hyde_update,
)
import graphrag_community_detection as GR                           # noqa: E402

SEED = DPR_SEED                 # 7 -- the shared finance-geometry seed
DIM = 64                        # forced by GraphRAG: at d=32 / kappa=60 Leiden MERGES two sectors
KAPPA_SECTOR = 200.0            # same-sector cosine ~0.72 at d=64 -- communities are recoverable
N_SECTORS = 5
N_COMP = 5                      # 25 companies -- matches the graphrag corpus this reuses

# --- LOCAL regime -----------------------------------------------------------------------------
# A passage is a bag of TOKENS token embeddings. The three fusion legs read DISJOINT windows of
# that bag, so they are partial VIEWS of the same document rather than three noisy copies of one
# score. Overlapping windows would make them a quality ladder, and fusion would have nothing to
# recombine (measured: RRF 0.350 against a best leg of 0.400 before the windows were split).
TOKENS = 9
WIN_LEX = (0, 1, 2)
WIN_DENSE = (3, 4, 5)
WIN_LI = (6, 7, 8)
WINDOWS = (WIN_LEX, WIN_DENSE, WIN_LI)
SLOT_NAMES = ("lexical", "dense", "late")
if sorted(t for w in WINDOWS for t in w) != list(range(TOKENS)):
    # Not a formality. An earlier draft had the late window OVERLAP the other two; because a bag
    # is filled window by window, the late draw silently OVERWROTE half the dense window, so the
    # dense leg was reading a mixture of two themes. A passage of the gold company then matched
    # the query on BOTH components while a rival sharing only the dense theme matched on one,
    # and the dense leg scored a phantom 1.000 on the class built to defeat it. The legs are
    # partial views only if the windows partition the bag, so the partition is asserted.
    raise ValueError("the leg windows must PARTITION the token bag -- see local_corpus")

# Each company is described by a TRIPLE of latent themes, one per leg window. Which companies
# share which themes is the load-bearing structure of the whole regime: a first probe drew the
# triples at random and found leg accuracy is a near-monotone function of how many companies
# share that slot's theme (unique -> 1.00, six-way -> 0.50) and essentially INDEPENDENT of the
# query's concentration. So two query classes that differ only in kappa are one mechanism
# sampled twice. The collision structure is therefore CONSTRUCTED, not drawn, and the query
# classes are defined by it.
SHARED_PER_SLOT = 4             # shared themes available to each slot: topics 4f .. 4f+3
N_SHARED = 3 * SHARED_PER_SLOT  # 12 shared themes
N_DISTINCTIVE = 10              # companies holding a dense-slot theme held by NO other company
N_TOPICS = N_SHARED + N_DISTINCTIVE

KAPPA_TOPIC = 260.0             # how tightly a passage's window sits on its theme
KAPPA_QUERY = 400.0             # how precisely a query names its themes
R_PASSAGES = 3                  # passages per company

THETA_OFF_DEG = 70.0            # 'off' class: tilt toward the generic corpus direction -- the
                                # query/document distribution shift HyDE exists to correct
NOISY_DECOY_W = 0.62            # 'noisy' class: weight on each leg's OWN decoy company
ALPHA_BRIDGE_DEG = 35.0         # 'bridge' class: mention angle, f = cos(a) theme_x + sin(a) theme_y.
                                # Swept over 30/35/40/45 at d=64: the mention must stay near enough
                                # to its own filing to be retrieved (cos a) while leaving a residual
                                # large enough to reformulate off (sin a). 45 deg loses the bridge
                                # entirely (read rate 0.40); 35 is the measured optimum, 0.85.
KAPPA_BRIDGE = 400.0
N_PER_CLASS = 20
CLASSES = ("on", "partial", "off", "noisy", "bridge")

# --- arm parameters ---------------------------------------------------------------------------
K_RETRIEVE = 4                  # passages read per hop. A distinctive company owns exactly
                                # R_PASSAGES of them, so its bridge sits at rank R_PASSAGES.
OVER_FETCH = 20                 # corrective's deeper second look
REFORM_EPS = 0.42               # a read filing opens a NEW direction if its mean per-window
                                # residual clears this. Swept, NOT inherited: the multi-hop topic
                                # set 0.47 at d=32 and the separation here is own 0.359 +- 0.017
                                # against bridge 0.637 +- 0.106, so the gap sits lower.
MAX_HOPS = 3
GRADER_MID = 0.80               # grader midpoint on the normalised full-MaxSim of the top hit.
                                # Swept jointly with the fire rate and the false-positive rate,
                                # never one at a time: 0.85 fires on 86% of queries (including
                                # 79% of the ones already correct) and 0.88 fires on everything,
                                # which would make the arm unconditional rather than adaptive.
GRADER_MID_GLOBAL = 0.62        # the same midpoint for entity cosine, which is the only evidence
                                # the global regime offers a grader
GRADER_KAPPA = 20.0              # calibration of the corrective grader (see grader_auc)
N_HYPOTHETICAL = 4              # hypothetical documents HyDE averages
HALLUCINATION_P = 0.2           # Bernoulli rate at which the generator drafts the wrong company.
                                # A RATE, not a continuous tilt: a tilt gives a step, not a floor.
GLOBAL_TOPK = 10                # entities a flat arm votes over in the global regime

# Cost is reported in two currencies and never collapsed into one, because the exchange rate
# between a vector operation and a generation call is exactly the thing a cost model should not
# invent. ops = token-pair similarity evaluations; calls = generator/grader invocations.
C_POOLED = 1                    # one pooled-window dot product against one document
C_MAXSIM = TOKENS * TOKENS      # one full MaxSim against one document

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


def theme_assignment() -> tuple[np.ndarray, np.ndarray]:
    """Which theme each company holds in each slot, and which companies are DISTINCTIVE.

    Two families, interleaved two-per-sector so family does not correlate with sector:

      distinctive -- the dense-slot theme is EXCLUSIVE to this company, while its lexical and
                     late themes come from the shared pools. One view identifies it outright, so
                     the cheapest arm is already correct and everything above it is pure cost.
      conjunctive -- all three themes come from the shared pools, so every leg alone returns a
                     handful of rivals, and only the company holding ALL THREE is the answer.
                     Fusion here is set intersection, which is the only thing that makes RRF
                     able to beat its own best leg.

    Returns (comp_topics (K,3) theme id per slot, distinctive (K,) 0/1).
    """
    K = N_SECTORS * N_COMP
    distinctive = np.array([1 if (a % N_COMP) < 2 else 0 for a in range(K)])
    if int(distinctive.sum()) != N_DISTINCTIVE:
        raise ValueError(f"expected {N_DISTINCTIVE} distinctive companies, got {int(distinctive.sum())}")
    comp_topics = np.zeros((K, 3), dtype=int)
    d = j = 0
    for a in range(K):
        if distinctive[a]:
            comp_topics[a] = (d % SHARED_PER_SLOT,                       # shared lexical theme
                              N_SHARED + d,                              # EXCLUSIVE dense theme
                              2 * SHARED_PER_SLOT + (3 * d + 1) % SHARED_PER_SLOT)
            d += 1
        else:
            # distinct (t0, t1) pairs give distinct triples; every slot theme is reused 3-4 times
            comp_topics[a] = (j % SHARED_PER_SLOT,
                              SHARED_PER_SLOT + (j // SHARED_PER_SLOT) % SHARED_PER_SLOT,
                              2 * SHARED_PER_SLOT + (j + j // SHARED_PER_SLOT) % SHARED_PER_SLOT)
            j += 1
    if len({tuple(t) for t in comp_topics.tolist()}) != K:
        raise ValueError("theme triples must be unique -- otherwise no arm can separate two companies")
    return comp_topics, distinctive


def local_corpus(seed: int = SEED) -> dict:
    """The factoid regime: the answer is carried by a passage, and the job is to find it.

    Five query classes, each isolating the mechanism one arm is supposed to exhibit:

      on      -- a DISTINCTIVE company named precisely. The dense view alone is decisive, so the
                 cheapest arm is already right; the class exists to price what the others add.
      partial -- a CONJUNCTIVE company. Each leg alone returns rivals and only the intersection
                 of all three identifies it. Hybrid's home ground.
      off     -- a distinctive company, but the query is tilted toward the generic corpus
                 direction: the query/document distribution shift HyDE corrects by discarding the
                 query's position entirely and retrieving with a generated document instead.
      noisy   -- a distinctive company, with each leg dragged toward a DIFFERENT wrong peer. The
                 gold survives at low rank rather than vanishing, which is the precise condition
                 under which a grader plus a deeper, more expensive re-scoring can recover it.
      bridge  -- the answer is a company that a filing NAMES rather than describes. No passage is
                 about it, so no amount of over-fetching reaches it and only a second hop does.
    """
    protos, sector_of, _ = _protos(seed)
    K = len(protos)
    comp_topics, distinctive = theme_assignment()

    trng = np.random.default_rng(seed + 12345)
    topics = normalize(trng.standard_normal((N_TOPICS, DIM)))
    themes = np.array([[topics[comp_topics[a, f]] for f in range(3)] for a in range(K)])

    # The per-slot "document-ness" axis: the mean theme direction that slot's window is drawn
    # around, corpus-wide. Tilting a query toward it is the distribution shift HyDE corrects.
    # It must live in THEME space, not company-prototype space -- the passages do.
    g = np.array([normalize(themes[:, f, :].mean(axis=0)) for f in range(3)])

    def _bag(centers, sd: int, kappa: float) -> np.ndarray:
        """One passage or query: each leg's window drawn around that leg's theme."""
        bag = np.zeros((TOKENS, DIM))
        for f, win in enumerate(WINDOWS):
            draws = sample_vmf(len(win), centers[f], kappa, seed=sd + 97 * f)
            for j, t in enumerate(win):
                bag[t] = normalize(draws[j])
        return bag

    passages, owner = [], []
    for a in range(K):
        for r in range(R_PASSAGES):
            passages.append(_bag(themes[a], seed + 4001 + 31 * a + r, KAPPA_TOPIC))
            owner.append(a)

    # Bridge passages, on the multi-hop mention geometry: a filing OF x that NAMES y. Its owner
    # is x -- it is x's filing, and a retriever that finds it has still not found y. The mention
    # survives only as the component orthogonal to x, which is exactly what reformulation
    # extracts and nothing else reads.
    dist_ids = [a for a in range(K) if distinctive[a]]
    alpha = np.deg2rad(ALPHA_BRIDGE_DEG)
    bridge_of, bridge_doc = {}, {}
    for x in range(K):
        cands = [y for y in dist_ids if sector_of[y] != sector_of[x]]
        y = int(cands[(x * 5 + 3) % len(cands)])
        bridge_of[x] = y
        mention = [normalize(np.cos(alpha) * themes[x, f] + np.sin(alpha) * themes[y, f])
                   for f in range(3)]
        bridge_doc[x] = len(passages)
        passages.append(_bag(mention, seed + 7717 + 13 * x, KAPPA_BRIDGE))
        owner.append(x)
    passages = np.array(passages)
    owner = np.array(owner)
    is_bridge = np.array([0] * (K * R_PASSAGES) + [1] * K)

    conj_ids = [a for a in range(K) if not distinctive[a]]
    Q, gold, klass = [], [], []
    theta = np.deg2rad(THETA_OFF_DEG)
    for i in range(N_PER_CLASS):
        a = dist_ids[i % len(dist_ids)]
        Q.append(_bag(themes[a], seed + 20001 + i, KAPPA_QUERY)); gold.append(a); klass.append("on")
    for i in range(N_PER_CLASS):
        a = conj_ids[(i * 2) % len(conj_ids)]
        Q.append(_bag(themes[a], seed + 60001 + i, KAPPA_QUERY)); gold.append(a); klass.append("partial")
    for i in range(N_PER_CLASS):
        a = dist_ids[(i * 3 + 1) % len(dist_ids)]
        off = [normalize(np.cos(theta) * themes[a, f] + np.sin(theta) * g[f]) for f in range(3)]
        Q.append(_bag(off, seed + 30001 + i, KAPPA_QUERY)); gold.append(a); klass.append("off")
    for i in range(N_PER_CLASS):
        a = dist_ids[(i * 7 + 2) % len(dist_ids)]
        peers = [c for c in range(K) if c != a]
        mixed = [normalize((1.0 - NOISY_DECOY_W) * themes[a, f]
                           + NOISY_DECOY_W * themes[peers[(i * 3 + 5 * f) % len(peers)], f])
                 for f in range(3)]
        Q.append(_bag(mixed, seed + 50001 + i, KAPPA_QUERY)); gold.append(a); klass.append("noisy")
    for i in range(N_PER_CLASS):
        # The SOURCE company must be distinctive. A mention sits at cos(ALPHA_BRIDGE_DEG) = 0.766
        # from the company it is filed under, so it is outranked by anything nearer than that --
        # and a company sharing x's dense theme sits at cosine 1.0. Off a conjunctive source the
        # bridge passage lands at median rank 24 and no second hop ever reads it; off a
        # distinctive one the only passages above it are x's own three.
        x = dist_ids[(i * 9 + 5) % len(dist_ids)]
        Q.append(_bag(themes[x], seed + 40001 + i, KAPPA_QUERY))
        gold.append(bridge_of[x]); klass.append("bridge")

    return {
        "regime": "local", "protos": protos, "themes": themes, "comp_topics": comp_topics,
        "distinctive": distinctive, "sector_of": sector_of, "g": g, "topics": topics,
        "docs": passages, "owner": owner, "is_bridge": is_bridge, "bridge_of": bridge_of,
        "bridge_doc": bridge_doc, "dist_ids": dist_ids, "conj_ids": conj_ids,
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



def full_maxsim(c: dict, q: np.ndarray) -> np.ndarray:
    """The expensive scorer: MaxSim over ALL tokens rather than one 4-token window. Quadratic in
    tokens where a pooled leg is linear, which is why an arm only pays for it on demand."""
    return maxsim_matrix(q[None, ...], c["docs"])[0]


def answer_of(c: dict, doc_ids) -> int:
    """The answer a ranking commits to: the company that owns its top passage."""
    return int(c["owner"][int(doc_ids[0])])


def gold_rank(c: dict, scores: np.ndarray, gold: int) -> int:
    """0-based rank of the gold company's BEST passage. n_docs if the company owns none."""
    order = np.argsort(-scores)
    hits = np.where(c["owner"][order] == gold)[0]
    return int(hits[0]) if len(hits) else c["n_docs"]


def _class_idx(c: dict, k: str) -> np.ndarray:
    return np.where(c["klass"] == k)[0]


def _diagnostics() -> None:
    L, G = local(), glob()
    print(f"LOCAL  : {L['n_queries']} queries, {L['n_docs']} docs, {L['K']} companies, dim {DIM}")
    print(f"GLOBAL : {G['n_queries']} themes,  {G['K']} entities, {G['n_answers']} sectors")
    ct, dist = L["comp_topics"], L["distinctive"]
    share = [[int((ct[:, f] == ct[a, f]).sum()) for f in range(3)] for a in range(L["K"])]
    share = np.array(share)
    print(f"  theme sharing (#companies per slot): distinctive {share[dist == 1].mean(axis=0).round(2)}"
          f"  conjunctive {share[dist == 0].mean(axis=0).round(2)}")
    print()

    print("LOCAL — per-leg top-1 accuracy by query class (do the classes separate?)")
    print(f"  {'class':8s} " + " ".join(f"{n:>9s}" for n in LEGS) + f" {'RRF':>9s} {'full-MS':>9s}")
    for k in CLASSES:
        m = _class_idx(L, k)
        accs = {}
        for name, fn in LEGS.items():
            accs[name] = sum(answer_of(L, np.argsort(-fn(L, L["Q"][i]))) == L["gold"][i]
                             for i in m) / len(m)
        rr = ms = 0
        for i in m:
            rankings = [list(np.argsort(-fn(L, L["Q"][i]))[:20]) for fn in LEGS.values()]
            rr += int(answer_of(L, rrf_fuse(rankings)) == L["gold"][i])
            ms += int(answer_of(L, np.argsort(-full_maxsim(L, L["Q"][i]))) == L["gold"][i])
        print(f"  {k:8s} " + " ".join(f"{accs[n]:9.3f}" for n in LEGS)
              + f" {rr/len(m):9.3f} {ms/len(m):9.3f}")

    print()
    print("LOCAL — where does the gold SIT in the cheap (dense) ranking? corrective fixes rank,")
    print("        not reach: it needs the gold present but demoted, never absent.")
    print(f"  {'class':8s} {'median rank':>12s} {'in top-20':>10s} {'absent':>8s}")
    for k in CLASSES:
        m = _class_idx(L, k)
        ranks = np.array([gold_rank(L, leg_dense(L, L["Q"][i]), L["gold"][i]) for i in m])
        print(f"  {k:8s} {np.median(ranks):12.1f} {float((ranks < 20).mean()):10.2f}"
              f" {float((ranks >= L['n_docs']).mean()):8.2f}")

    print()
    print("LOCAL — bridge residuals: does the mention separate from ordinary passages?")
    for k in ("on", "bridge"):
        m = _class_idx(L, k)
        ord_r, br_r = [], []
        for i in m:
            q = L["Q"][i]
            for d in np.argsort(-leg_dense(L, q))[:5]:
                r = float(np.mean([np.linalg.norm(_pool(L["docs"][d], w) -
                                                  float(_pool(L["docs"][d], w) @ _pool(q, w)) * _pool(q, w))
                                   for w in WINDOWS]))
                (br_r if L["is_bridge"][d] else ord_r).append(r)
        print(f"  {k:8s} ordinary {np.mean(ord_r):.3f}+-{np.std(ord_r):.3f} (n={len(ord_r)})"
              f"   bridge {np.mean(br_r) if br_r else float('nan'):.3f} (n={len(br_r)})")

    print()
    print("GLOBAL — does counting disagree with geometry? (the anti-theatre condition)")
    dis = int((G["gold"] != G["geo_answer"]).sum())
    print(f"  themes where count-max sector != nearest sector: {dis}/{G['n_queries']}")
    est = np.asarray(GR.leiden(G["A"]))
    from collections import Counter
    pure = all(len(Counter(G["sector_of"][est == c]).keys()) == 1 for c in set(est.tolist()))
    print(f"  Leiden recovers {len(set(est.tolist()))} communities; each pure by sector: {pure}")
    for kk in (5, 10, 15):
        hits = 0
        for i in range(G["n_queries"]):
            top = np.argsort(-(G["protos"] @ G["Q"][i]))[:kk]
            hits += int(np.argmax(np.bincount(G["sector_of"][top], minlength=N_SECTORS)) == G["gold"][i])
        print(f"  FLAT top-{kk:<2d} sector vote: {hits}/{G['n_queries']}   <- must be LOW or the arm is theatre")


GR_MENTION_TAU = 0.35           # an entity "discusses" a theme if its cosine clears this. The
                                # community summary records the SET that does; counting over that
                                # set is the operation no ranking of entities performs.


def _rebuild_bag(per_window) -> np.ndarray:
    """A token bag whose every window is filled with that window's direction. Reformulation acts
    on pooled windows, so the reformulated query is written back window by window."""
    bag = np.zeros((TOKENS, DIM))
    for w, v in zip(WINDOWS, per_window):
        for t in w:
            bag[t] = v
    return bag


def residual_norm(c: dict, q: np.ndarray, d: int) -> float:
    """How much of document d points somewhere the query does not: the mean over windows of
    ||d_w - <d_w, q_w> q_w||. An ordinary filing of the company asked about measures 0.359 +-
    0.017 here; one that NAMES another company measures 0.637 +- 0.106."""
    tot = 0.0
    for w in WINDOWS:
        qp, dp = _pool(q, w), _pool(c["docs"][d], w)
        tot += float(np.linalg.norm(dp - float(dp @ qp) * qp))
    return tot / len(WINDOWS)


_COMMUNITIES: dict[str, np.ndarray] = {}


def communities(c: dict) -> np.ndarray:
    """Leiden labels for this regime's entity graph, detected ONCE. Community detection is the
    offline half of GraphRAG -- it is paid at index time, not per query -- so it is cached here
    rather than rerun inside the arm, which would both misreport the cost and dominate it."""
    key = c["regime"]
    if key not in _COMMUNITIES:
        A = c["A"] if key == "global" else local_entity_graph(c)
        _COMMUNITIES[key] = np.asarray(GR.leiden(A))
    return _COMMUNITIES[key]


def local_entity_graph(c: dict) -> np.ndarray:
    """The company co-occurrence graph the local regime's community detection runs on, built the
    same way the global one is: thresholded cosine between company prototypes."""
    A = np.maximum(0.0, c["protos"] @ c["protos"].T - GR.GR_THRESHOLD)
    np.fill_diagonal(A, 0.0)
    return A


# =================================================================================================
# The six arms. Uniform signature: arch_<name>(corpus, i, **params) -> dict with
#   answer  -- the company (local) or sector (global) the arm commits to
#   ops     -- token-pair similarity evaluations spent on this query
#   calls   -- generator / grader invocations spent on this query
#   ranking -- the document ordering the arm finished with (for inspection)
# Every arm dispatches on corpus["regime"], because an architecture is a bet about where the
# answer lives and the two regimes put it in different places.
# =================================================================================================

def _legs_rankings(c: dict, q: np.ndarray, depth: int = 20) -> list[list[int]]:
    return [list(np.argsort(-fn(c, q))[:depth]) for fn in LEGS.values()]


def _global_vote(c: dict, order, topk: int) -> int:
    """The flat answer in the global regime: let the top-k entities vote for their sector."""
    top = np.asarray(order)[:topk]
    return int(np.argmax(np.bincount(c["sector_of"][top], minlength=N_SECTORS)))


def arch_naive(c: dict, i: int, **_) -> dict:
    """One dense retrieval, one answer. The baseline every other arm has to justify itself
    against, and on a query whose answer one view already carries it is unimprovable."""
    q = c["Q"][i]
    if c["regime"] == "global":
        order = np.argsort(-(c["protos"] @ q))
        return {"answer": _global_vote(c, order, GLOBAL_TOPK), "ops": c["K"] * C_POOLED,
                "calls": 0, "ranking": order}
    order = np.argsort(-leg_dense(c, q))
    return {"answer": answer_of(c, order), "ops": c["n_docs"] * C_POOLED, "calls": 0,
            "ranking": order}


def arch_hybrid(c: dict, i: int, **_) -> dict:
    """Three partial views of each document, fused by reciprocal rank. Fusion is set
    intersection: it can only add what the legs disagree about, so it is decisive exactly when
    no single view identifies the answer -- and it is a liability when one already does."""
    q = c["Q"][i]
    if c["regime"] == "global":
        # The two geometric views available without a community structure: similarity to the
        # theme, and similarity to the entity's own sector mean. Fusing them is still geometry.
        r1 = list(np.argsort(-(c["protos"] @ q)))
        r2 = list(np.argsort(-(c["sector_means"][c["sector_of"]] @ q)))
        order = rrf_fuse([r1, r2])
        return {"answer": _global_vote(c, order, GLOBAL_TOPK), "ops": 2 * c["K"] * C_POOLED,
                "calls": 0, "ranking": order}
    order = rrf_fuse(_legs_rankings(c, q))
    return {"answer": answer_of(c, order), "ops": 3 * c["n_docs"] * C_POOLED, "calls": 0,
            "ranking": order}


def arch_hyde(c: dict, i: int, alpha: float = 1.0, k_hyp: int = N_HYPOTHETICAL,
              p: float = HALLUCINATION_P, **_) -> dict:
    """Discard the query's position; generate documents that look like the answer and retrieve
    with those. That is the whole trick, and it is why HyDE is indifferent to how far off the
    manifold the query sat -- and why it inherits whatever the generator got wrong.

    alpha is the Rocchio interpolation weight of the imported hyde_update: alpha = 0 is the bare
    query (and therefore exactly naive), alpha = 1 is the pure pseudo-document.
    """
    q = c["Q"][i]
    if c["regime"] == "global":
        # A hypothetical filing about the theme is still one point in the same space, so the
        # arm still answers "which sector is nearest", which is not the question asked.
        hyp = hyde_centroid(q, k_hyp, kappa_h=80.0, seed=SEED + 811 + i)
        order = np.argsort(-(c["protos"] @ hyp))
        return {"answer": _global_vote(c, order, GLOBAL_TOPK), "ops": c["K"] * C_POOLED,
                "calls": k_hyp, "ranking": order}

    # The generator reads the QUERY and drafts what a passage answering it would look like. It
    # never sees the gold: the company it drafts for is the one the query most resembles in that
    # slot, snapped onto the document manifold. That snap is the entire mechanism -- it is what
    # discards the query's off-manifold position -- and it is also the entire limitation, because
    # a query that resembles the wrong company produces a confident passage about the wrong
    # company. At rate p the generator drafts for the rival it most easily confuses instead.
    hallucinated = bool(np.random.default_rng(SEED + 77 + i).random() < p)
    bag = np.zeros((TOKENS, DIM))
    for f, win in enumerate(WINDOWS):
        P_f = c["themes"][:, f, :]
        # Writing in document style means dropping the component along the axis that separates
        # queries from documents -- the corpus mean direction g, which by construction says
        # nothing about WHICH company. Removing it is available without knowing the answer, and
        # it is the whole of HyDE's correction. It restores a query tilted off the manifold and
        # does nothing at all for a query tilted toward the wrong company.
        qw = _pool(q, win)
        gw = c["g"][f]
        deshift = qw - float(qw @ gw) * gw
        believed = int(np.argmax(P_f @ (normalize(deshift) if np.linalg.norm(deshift) > 1e-9 else qw)))
        center = generation_center(believed, P_f, hallucination_targets(P_f), hallucinated)
        hyp = hyde_centroid(center, k_hyp, kappa_h=KAPPA_TOPIC, seed=SEED + 3300 + 41 * i + 7 * f)
        mixed = hyde_update(_pool(q, win), hyp, alpha)
        for t in win:
            bag[t] = mixed
    order = np.argsort(-leg_dense(c, bag))
    return {"answer": answer_of(c, order), "ops": c["n_docs"] * C_POOLED, "calls": k_hyp,
            "ranking": order}


def grade(c: dict, i: int, order, kappa: float = GRADER_KAPPA,
          mid: float = GRADER_MID) -> float:
    """The corrective grader's confidence that the cheap retrieval can be answered from.

    It re-scores the single top retrieved document PROPERLY -- full MaxSim over every token,
    rather than the one pooled window the cheap leg looked at -- and asks whether the document
    really supports the query. That is what a relevance grader does, it costs one document's
    worth of the expensive scorer, and it is deliberately IMPERFECT: measured AUC 0.83 against
    "was the cheap answer right", not 1.0. A perfect grader would fire exactly when needed and
    make the whole comparison vacuous.

    What it CANNOT see is the bridge class. There the retrieved filing is genuinely, highly
    relevant -- it is the filing of the company the query describes -- and grades 0.846, level
    with the class the cheap arm answers perfectly. Relevance is not correctness when the answer
    is a company that no retrieved passage is about.
    """
    top = int(np.asarray(order)[0])
    if c["regime"] == "global":
        return float(expit(kappa * (float(c["protos"][top] @ c["Q"][i]) - GRADER_MID_GLOBAL)))
    ms = float(maxsim_matrix(c["Q"][i][None, ...], c["docs"][[top]])[0][0]) / TOKENS
    return float(expit(kappa * (ms - mid)))


def arch_corrective(c: dict, i: int, threshold: float = 0.5, over: int = OVER_FETCH,
                    rerank_noise: float = 0.0, grader_mid: float = GRADER_MID, **_) -> dict:
    """Retrieve cheaply, grade the result, and on a bad grade look deeper and score harder.

    Correction moves the answer up a ranking it is already on. It cannot put it there: when the
    answer is a company no retrieved passage is about, over-fetching returns more of the same
    wrong thing and the expensive re-scoring ranks it just as confidently.
    """
    q = c["Q"][i]
    base = arch_naive(c, i)
    g = grade(c, i, base["ranking"], mid=grader_mid)
    out = {"answer": base["answer"], "ops": base["ops"] + (0 if c["regime"] == "global" else C_MAXSIM),
           "calls": 1, "ranking": base["ranking"], "fired": False, "grade": g}
    if g >= threshold:
        return out
    out["fired"] = True
    if c["regime"] == "global":
        # The deeper, more expensive second look: go further down the ranking and keep only the
        # entities that actually discuss the theme, then let those vote.
        #
        # This is the honest qualification on the whole global result. Filtering a deep slice to
        # the mentioning entities IS the aggregation the community summary holds -- performed by
        # hand, at query time, over a slice that has to be several times deeper than the answer
        # to reach it. The claim GraphRAG earns here is not that flat retrieval cannot get
        # there. It is that flat retrieval can only get there by going deeper and paying more.
        cand = np.asarray(base["ranking"])[:min(over, c["K"])]
        keep = cand[c["mentions"][i][cand] == 1]
        out["answer"] = _global_vote(c, keep if len(keep) else cand, len(keep) if len(keep) else 1)
        out["ops"] += min(over, c["K"]) * C_POOLED
        return out
    cand = np.asarray(base["ranking"])[:min(over, c["n_docs"])]
    sc = maxsim_matrix(q[None, ...], c["docs"][cand])[0]
    if rerank_noise > 0.0:
        # A corrective scorer that is not simply a better version of the cheap one. With
        # rerank_noise = 0 the re-scoring here DOMINATES the pooled leg on every query, so
        # correction is a free upgrade that never costs accuracy -- measured, and the reason
        # this parameter exists. Real rerankers are a different model, not a sharper copy, and
        # they are confidently wrong on their own queries. The noise makes that concrete.
        sc = sc + rerank_noise * np.random.default_rng(SEED + 4200 + i).standard_normal(len(sc))
    order = cand[np.argsort(-sc)]
    out["answer"] = answer_of(c, order)
    out["ops"] += len(cand) * C_MAXSIM
    out["ranking"] = order
    return out


def arch_graph(c: dict, i: int, **_) -> dict:
    """Detect communities once, offline, then answer from what a community AGGREGATES.

    This is the only arm that holds a representation of a set rather than of a document, which
    is why it is the only one that can answer a question about a count -- and why, asked which
    single filing says something, it answers with a neighbourhood.
    """
    q = c["Q"][i]
    if c["regime"] == "global":
        labels = communities(c)
        # A community summary records WHICH of its members discuss the theme -- that set is read
        # off the filings at index time, not re-derived from similarity at query time. Answering
        # is then COUNTING over the summary, which is the operation no ranking of entities
        # performs, and the reason this arm can answer a question about a set.
        counts = [int(c["mentions"][i][labels == lab].sum()) for lab in range(labels.max() + 1)]
        lab = int(np.argmax(counts))
        members = np.where(labels == lab)[0]
        answer = int(np.bincount(c["sector_of"][members], minlength=N_SECTORS).argmax())
        return {"answer": answer, "ops": c["K"] * C_POOLED, "calls": 0,
                "ranking": np.argsort(-(c["protos"] @ q))}
    labels = communities(c)
    cent = np.array([normalize(_pool(c["docs"], WIN_DENSE)[c["owner"] == a].mean(axis=0))
                     for a in range(c["K"])])
    comm = np.array([normalize(cent[labels == lab].mean(axis=0)) for lab in range(labels.max() + 1)])
    lab = int(np.argmax(comm @ _pool(q, WIN_DENSE)))
    members = np.where(labels == lab)[0]
    within = members[np.argmax(cent[members] @ _pool(q, WIN_DENSE))]
    order = np.argsort(-leg_dense(c, q))
    return {"answer": int(within), "ops": (labels.max() + 1 + len(members)) * C_POOLED,
            "calls": 0, "ranking": order}


def arch_agentic(c: dict, i: int, max_hops: int = MAX_HOPS, eps: float = REFORM_EPS,
                 k: int = K_RETRIEVE, **_) -> dict:
    """Retrieve, read, reformulate, retrieve again -- stopping when the filing just read opens no
    direction the query did not already have.

    max_hops = 1 is exactly naive: one retrieval, no reading. Every hop beyond the first is the
    bet that the answer is somewhere the query does not point, and the arm pays for that bet on
    every query, including the ones where it was wrong to make it.
    """
    q = c["Q"][i]
    if c["regime"] == "global":
        order = np.argsort(-(c["protos"] @ q))
        ops, hops = c["K"] * C_POOLED, 1
        seen = list(order[:GLOBAL_TOPK])      # the pool naive votes over: one hop must BE naive
        for _h in range(max_hops - 1):
            # Expand along the entity graph: the neighbours of what has been read. This reaches
            # more of the community than one retrieval does, one hop at a time, and pays a full
            # retrieval for each step.
            nxt = [int(j) for d in seen[:k] for j in np.argsort(-c["A"][d])[:k] if c["A"][d][j] > 0]
            if not nxt:
                break
            seen = list(dict.fromkeys(seen + nxt))
            ops += c["K"] * C_POOLED
            hops += 1
        return {"answer": _global_vote(c, seen, min(GLOBAL_TOPK, len(seen))), "ops": ops,
                "calls": hops, "ranking": np.array(seen), "hops": hops}

    cur, ops, hops, read = q, 0, 0, []
    order = np.argsort(-leg_dense(c, cur))
    for _h in range(max_hops):
        order = np.argsort(-leg_dense(c, cur))
        ops += c["n_docs"] * C_POOLED
        hops += 1
        if hops >= max_hops:
            break
        nxt = None
        for d in order[:k]:
            if int(d) in read:
                continue
            if residual_norm(c, cur, int(d)) >= eps:
                nxt = int(d)
                break
        if nxt is None:                      # nothing opens a new direction: the chain is done
            break
        read.append(nxt)
        cur = _rebuild_bag([reformulate(_pool(cur, w), _pool(c["docs"][nxt], w)) for w in WINDOWS])
    return {"answer": answer_of(c, order), "ops": ops, "calls": hops, "ranking": order,
            "hops": hops, "read": read}


ARMS = {"naive": arch_naive, "hybrid": arch_hybrid, "hyde": arch_hyde,
        "corrective": arch_corrective, "graph": arch_graph, "agentic": arch_agentic}


# =================================================================================================
# The 6 x 2 mechanism matrix.
# =================================================================================================

def run_arm(c: dict, name: str, **params) -> dict:
    """One arm over every query of one regime: accuracy overall and per class, and mean cost."""
    fn = ARMS[name]
    ok, ops, calls, fired, hops = [], [], [], [], []
    for i in range(c["n_queries"]):
        r = fn(c, i, **params)
        ok.append(int(r["answer"] == c["gold"][i]))
        ops.append(r["ops"]); calls.append(r["calls"])
        fired.append(int(r.get("fired", False))); hops.append(int(r.get("hops", 1)))
    ok = np.array(ok)
    per_class = {k: float(ok[c["klass"] == k].mean()) for k in dict.fromkeys(c["klass"].tolist())}
    return {"arm": name, "regime": c["regime"], "acc": float(ok.mean()), "per_class": per_class,
            "ops": float(np.mean(ops)), "calls": float(np.mean(calls)),
            "fire_rate": float(np.mean(fired)), "mean_hops": float(np.mean(hops)), "ok": ok}


def mechanism_matrix() -> dict:
    """Every arm on both regimes. The thesis of the topic is that this table has no row that wins
    both columns, and that every win is traceable to a property of the corpus rather than to the
    architecture being better."""
    L, G = local(), glob()
    return {name: {"local": run_arm(L, name), "global": run_arm(G, name)} for name in ARMS}


def _fmt_matrix(M: dict) -> str:
    rows = [f"  {'arm':11s} {'LOCAL':>7s} " + " ".join(f"{k:>8s}" for k in CLASSES)
            + f" | {'GLOBAL':>7s}   {'ops(L)':>8s} {'calls(L)':>8s}"]
    for name, r in M.items():
        pc = r["local"]["per_class"]
        rows.append(f"  {name:11s} {r['local']['acc']:7.3f} "
                    + " ".join(f"{pc.get(k, float('nan')):8.3f}" for k in CLASSES)
                    + f" | {r['global']['acc']:7.3f}   {r['local']['ops']:8.0f}"
                    f" {r['local']['calls']:8.2f}")
    return "\n".join(rows)


# =================================================================================================
# Collapse anchors: every arm must reduce to the baseline at its degenerate setting. These are
# byte-for-byte, because an arm that only APPROXIMATELY reduces to naive is a different arm.
# =================================================================================================

def test_window_partition() -> None:
    """The legs are partial views only if their windows partition the bag. Asserted at import;
    restated here because an overlapping window silently corrupted the dense leg once already."""
    flat = sorted(t for w in WINDOWS for t in w)
    assert flat == list(range(TOKENS)), flat
    assert len({tuple(t) for t in theme_assignment()[0].tolist()}) == N_SECTORS * N_COMP


def test_hyde_alpha_zero_is_naive() -> None:
    """alpha = 0 in the imported Rocchio update returns the bare query, so HyDE with no weight on
    its generated document IS naive -- same ranking, not merely the same accuracy."""
    c = local()
    for i in (0, 25, 50, 75, 99):
        assert np.array_equal(arch_hyde(c, i, alpha=0.0)["ranking"], arch_naive(c, i)["ranking"])
    q = local()["Q"][3]
    assert np.allclose(hyde_update(_pool(q, WIN_LEX), np.ones(DIM) / np.sqrt(DIM), 0.0),
                       _pool(q, WIN_LEX), atol=1e-12)


def test_agentic_one_hop_is_naive() -> None:
    """One hop is one retrieval and no reading, which is exactly the baseline. Everything the
    agentic arm wins and everything it loses is bought by the hops after this one."""
    for c in (local(), glob()):
        for i in range(0, c["n_queries"], 7):
            a, n = arch_agentic(c, i, max_hops=1), arch_naive(c, i)
            assert a["answer"] == n["answer"], (c["regime"], i)
            assert np.array_equal(np.asarray(a["ranking"])[:K_RETRIEVE],
                                  np.asarray(n["ranking"])[:K_RETRIEVE])
            assert a["ops"] == n["ops"]


def test_corrective_never_firing_is_naive() -> None:
    """A grader that never fires leaves the cheap answer alone. The arm then differs from naive
    only by what it spent finding that out -- one document's worth of the expensive scorer."""
    c = local()
    for i in range(0, c["n_queries"], 5):
        r, n = arch_corrective(c, i, threshold=0.0), arch_naive(c, i)
        assert r["answer"] == n["answer"] and not r["fired"]
        assert r["ops"] == n["ops"] + C_MAXSIM


def test_rrf_idempotent_on_identical_legs() -> None:
    """Fusion adds nothing to legs that agree. This is why hybrid is not free: what it buys is
    exactly the disagreement between its views, and where there is none it can only reorder."""
    c = local()
    for i in (0, 40, 80):
        r = list(np.argsort(-leg_dense(c, c["Q"][i]))[:20])
        assert [int(x) for x in rrf_fuse([r, r, r])] == [int(x) for x in r]


def test_maxsim_scorer_matches_import() -> None:
    """The expensive scorer is the imported late-interaction MaxSim, not a local rewrite."""
    c = local()
    q = c["Q"][11]
    assert np.allclose(full_maxsim(c, q), maxsim_matrix(q[None, ...], c["docs"])[0], atol=1e-12)


# =================================================================================================
# The pedagogical claims, as tests. Wins and losses both: a matrix in which every arm is good at
# something is a brochure, and the losses are the part the cheat sheets leave out.
# =================================================================================================

def test_no_arm_wins_both_regimes() -> None:
    """THE thesis. The arm that owns the global regime is mid-table locally, and the arms that
    lead locally are at or below chance globally."""
    M = mechanism_matrix()
    best_local = max(M, key=lambda a: M[a]["local"]["acc"])
    best_global = max(M, key=lambda a: M[a]["global"]["acc"])
    assert best_local != best_global, (best_local, best_global)
    assert M[best_global]["local"]["acc"] < M[best_local]["local"]["acc"]
    assert M[best_local]["global"]["acc"] < M[best_global]["global"]["acc"]
    assert all(r["local"]["acc"] < 1.0 for r in M.values())


def test_each_arm_owns_exactly_one_condition() -> None:
    """Every condition is led by a different arm, and the two arms that tie on one of them tie
    because they are buying the same thing there by different means. That is what makes this a
    set of bets rather than a ranking of architectures."""
    M = mechanism_matrix()
    best = {}
    for k in CLASSES:
        sc = {a: M[a]["local"]["per_class"][k] for a in M}
        best[k] = {a for a, v in sc.items() if v >= max(sc.values()) - 1e-9}
    g = {a: M[a]["global"]["acc"] for a in M}
    best["global"] = {a for a, v in g.items() if v >= max(g.values()) - 1e-9}

    assert best["global"] == {"graph"}
    assert best["bridge"] == {"agentic"}
    assert best["off"] == {"hyde"}
    assert best["on"] == {"naive", "corrective"}           # correction leaves a right answer alone
    assert best["partial"] == {"hybrid", "corrective"}     # both recombine views; neither is free
    assert len({frozenset(v) for v in best.values()}) == len(best), best
    assert set().union(*best.values()) >= {"naive", "hybrid", "hyde", "corrective", "graph", "agentic"}


def test_fusion_damages_a_query_one_view_answers() -> None:
    """Hybrid is not a free upgrade. On the class a single view answers outright, fusing three
    views is strictly worse than not fusing them: the two uninformed legs outvote the informed
    one, and reciprocal rank has no way to know which was which."""
    M = mechanism_matrix()
    assert M["naive"]["local"]["per_class"]["on"] == 1.0
    assert M["hybrid"]["local"]["per_class"]["on"] < M["naive"]["local"]["per_class"]["on"]


def test_fusion_wins_only_where_views_disagree() -> None:
    """...and on the class where no single view identifies the answer, fusion is decisive: it
    beats its own best leg by a margin no leg can reach, because the intersection of three
    ambiguous sets is not ambiguous."""
    c = local()
    m = _class_idx(c, "partial")
    legs = {n: sum(answer_of(c, np.argsort(-fn(c, c["Q"][i]))) == c["gold"][i] for i in m) / len(m)
            for n, fn in LEGS.items()}
    fused = sum(answer_of(c, rrf_fuse(_legs_rankings(c, c["Q"][i]))) == c["gold"][i]
                for i in m) / len(m)
    assert fused > max(legs.values()) + 0.4, (legs, fused)


def test_correction_fixes_rank_not_reach() -> None:
    """Corrective recovers the class where the answer is present but demoted, and recovers
    nothing at all where the answer is a company no retrieved passage is about. Over-fetching
    returns more of the same wrong thing."""
    c = local()
    M = mechanism_matrix()
    ranks = {k: np.median([gold_rank(c, leg_dense(c, c["Q"][i]), c["gold"][i])
                           for i in _class_idx(c, k)]) for k in ("noisy", "bridge")}
    assert ranks["noisy"] < OVER_FETCH <= ranks["bridge"]
    assert M["corrective"]["local"]["per_class"]["noisy"] > M["naive"]["local"]["per_class"]["noisy"]
    assert M["corrective"]["local"]["per_class"]["bridge"] == 0.0


def test_grader_cannot_see_the_reach_failure() -> None:
    """And it cannot even be told to. The grader scores relevance, and on the class it fails the
    retrieved filing is genuinely relevant -- it is the filing of the company the query
    describes. The two classes grade the same; one is answered perfectly and one not at all."""
    c = local()
    g = {k: float(np.mean([grade(c, i, arch_naive(c, i)["ranking"]) for i in _class_idx(c, k)]))
         for k in ("on", "bridge")}
    assert abs(g["on"] - g["bridge"]) < 0.05, g
    M = mechanism_matrix()
    assert M["naive"]["local"]["per_class"]["on"] == 1.0
    assert M["naive"]["local"]["per_class"]["bridge"] == 0.0


def test_only_hops_reach_the_named_company() -> None:
    """No arm that scores passages reaches the bridge answer at any depth, because no passage is
    about it. The mention survives only as the component orthogonal to the filing it sits in."""
    c = local()
    M = mechanism_matrix()
    assert M["agentic"]["local"]["per_class"]["bridge"] >= 0.8
    for a in ("naive", "hybrid", "hyde", "corrective", "graph"):
        assert M[a]["local"]["per_class"]["bridge"] == 0.0, a
    # ...and not because the answer is unreachable in principle: full MaxSim over every token,
    # over the whole corpus, still does not surface it.
    hits = sum(answer_of(c, np.argsort(-full_maxsim(c, c["Q"][i]))) == c["gold"][i]
               for i in _class_idx(c, "bridge"))
    assert hits == 0


def test_agentic_pays_for_its_bet_on_every_query() -> None:
    """The stopping rule reads a filing that opens a new direction, and it cannot tell one that
    names a new company from one that is merely off-target. So the arm hops on almost everything
    and is the WORST arm overall on the regime where it owns a class outright."""
    M = mechanism_matrix()
    assert M["agentic"]["local"]["mean_hops"] > 2.0
    assert M["agentic"]["local"]["acc"] == min(r["local"]["acc"] for r in M.values())
    assert M["agentic"]["local"]["per_class"]["on"] < M["naive"]["local"]["per_class"]["on"]


def test_global_counting_disagrees_with_geometry() -> None:
    """The anti-theatre condition, and the reason the global regime is worth running at all. If
    the sector that discusses a theme most were also the sector nearest it, community
    summarisation would be decoration on a problem top-k already solves."""
    c = glob()
    assert int((c["gold"] != c["geo_answer"]).sum()) == c["n_queries"]
    labels = communities(c)
    from collections import Counter
    assert labels.max() + 1 == N_SECTORS
    assert all(len(Counter(c["sector_of"][labels == lab])) == 1 for lab in range(labels.max() + 1))
    assert run_arm(c, "naive")["acc"] < 1.0 / N_SECTORS      # worse than guessing, not merely bad


def flat_aggregation_curve(depths=(3, 5, 8, 10, 14, 20, 25)) -> list[tuple[int, float]]:
    """Accuracy of doing the aggregation by hand: go this far down a flat ranking, keep the
    entities that discuss the theme, and let them vote."""
    c = glob()
    return [(int(d), run_arm(c, "corrective", over=d, threshold=1.01)["acc"]) for d in depths]


def test_graphrag_buys_depth_not_possibility() -> None:
    """The honest form of the global result. Flat retrieval CAN reach the answer -- by scanning
    far enough down and aggregating by hand. On this corpus it needs the whole entity set to get
    there, which is the cost GraphRAG has already paid, once, offline."""
    curve = dict(flat_aggregation_curve())
    assert curve[5] == 0.0 and curve[10] < 0.5
    assert curve[25] == 1.0
    reach = min(d for d, a in sorted(curve.items()) if a == 1.0)
    assert reach >= glob()["K"], reach
    assert run_arm(glob(), "graph")["acc"] == 1.0


def correction_damage_grid(mids=(0.80, 0.84, 0.88), sigmas=(0.0, 1.0, 2.0, 3.0)) -> dict:
    """Accuracy overall and on the already-correct class, as the grader gets more eager (mid) and
    the corrective scorer less reliable (sigma)."""
    c = local()
    out = {}
    for mid in mids:
        for sg in sigmas:
            r = run_arm(c, "corrective", grader_mid=mid, rerank_noise=sg)
            out[(float(mid), float(sg))] = (r["acc"], r["per_class"]["on"], r["fire_rate"])
    return out


def test_the_average_hides_what_correction_breaks() -> None:
    """The prediction going in was that over-eager correction is a NET loss. It is not, on this
    corpus, and the sweep says so: with the grader firing on every query and a corrective scorer
    at sigma = 3, the arm still beats the baseline it damaged, because the baseline was weak
    enough that a badly reordered shortlist of twenty is an improvement on average.

    What is true is worse, and invisible in the average: correction strictly destroys the one
    class that needed no correcting. The mean rises while a perfect column falls to 0.80.
    """
    grid = correction_damage_grid()
    naive_acc = run_arm(local(), "naive")["acc"]
    assert all(acc > naive_acc for acc, _on, _f in grid.values())          # never a net loss
    assert grid[(0.80, 0.0)][1] == 1.0                                     # at the operating point
    assert grid[(0.88, 3.0)][2] == 1.0                                     # grader fires on all
    assert grid[(0.88, 3.0)][1] <= 0.85                                    # ...and 'on' is broken
    assert grid[(0.88, 3.0)][0] > naive_acc                                # while the mean stays up
    on_col = [grid[(0.88, s)][1] for s in (0.0, 1.0, 2.0, 3.0)]
    assert all(a >= b for a, b in zip(on_col, on_col[1:])), on_col         # monotone destruction


def test_costs_are_ordered_as_the_mechanisms_are() -> None:
    """Every arm that adds a mechanism adds a cost, and the two currencies do not convert: HyDE
    is the cheapest arm in operations and the only local arm that cannot run without a
    generator."""
    M = mechanism_matrix()
    L = {a: M[a]["local"] for a in M}
    assert L["naive"]["ops"] < L["hybrid"]["ops"] < L["corrective"]["ops"]
    assert L["hyde"]["ops"] == L["naive"]["ops"] and L["hyde"]["calls"] > 0
    assert L["naive"]["calls"] == L["hybrid"]["calls"] == 0
    assert L["agentic"]["ops"] > L["naive"]["ops"]
    assert M["graph"]["global"]["ops"] < M["corrective"]["global"]["ops"]


# =================================================================================================
# viz_constants -- this function OWNS every number the laboratory displays. Anything the panels
# recompute live is recomputed from what is printed here, never re-derived in TypeScript.
# =================================================================================================

def _worked_partial_query() -> dict:
    """A query from the conjunctive class, with each leg's shortlist and the companies on it.
    Panel B recomputes the reciprocal-rank fusion of these three lists live, so the baked value
    is the input to fusion, not its output."""
    c = local()
    m = _class_idx(c, "partial")
    i = int(next(j for j in m if answer_of(c, rrf_fuse(_legs_rankings(c, c["Q"][j]))) == c["gold"][j]
                 and all(answer_of(c, np.argsort(-fn(c, c["Q"][j]))) != c["gold"][j]
                         for fn in LEGS.values())))
    legs = {n: [int(d) for d in np.argsort(-fn(c, c["Q"][i]))[:8]] for n, fn in LEGS.items()}
    return {"query": i, "gold": int(c["gold"][i]),
            "legs": legs,
            "owners": {n: [int(c["owner"][d]) for d in v] for n, v in legs.items()},
            "theme_of_gold": [int(t) for t in c["comp_topics"][c["gold"][i]]]}


def viz_constants() -> dict:
    """Every number the laboratory shows, in one place."""
    L, G = local(), glob()
    M = mechanism_matrix()

    matrix = {a: {"local": round(float(M[a]["local"]["acc"]), 3),
                  "global": round(float(M[a]["global"]["acc"]), 3),
                  "classes": {k: round(float(M[a]["local"]["per_class"][k]), 3) for k in CLASSES},
                  "ops": int(round(M[a]["local"]["ops"])),
                  "calls": round(float(M[a]["local"]["calls"]), 2),
                  "ops_global": int(round(M[a]["global"]["ops"]))} for a in ARMS}

    gold_ranks = {k: [int(gold_rank(L, leg_dense(L, L["Q"][i]), L["gold"][i]))
                      for i in _class_idx(L, k)] for k in CLASSES}
    grades = {k: round(float(np.mean([grade(L, i, arch_naive(L, i)["ranking"])
                                      for i in _class_idx(L, k)])), 3) for k in CLASSES}
    dmg = {f"{mid}|{sg}": [round(float(a), 3), round(float(o), 3), round(float(f), 2)]
           for (mid, sg), (a, o, f) in correction_damage_grid().items()}
    hops = {k: round(float(np.mean([arch_agentic(L, i)["hops"] for i in _class_idx(L, k)])), 2)
            for k in CLASSES}
    legs_partial = {n: round(float(sum(answer_of(L, np.argsort(-fn(L, L["Q"][i]))) == L["gold"][i]
                                       for i in _class_idx(L, "partial")) / N_PER_CLASS), 3)
                    for n, fn in LEGS.items()}
    return {
        "dim": DIM, "n_docs": L["n_docs"], "n_queries": L["n_queries"], "n_companies": L["K"],
        "n_sectors": N_SECTORS, "n_global": G["n_queries"], "classes": list(CLASSES),
        "arms": list(ARMS), "matrix": matrix,
        "gold_ranks": gold_ranks, "over_fetch": OVER_FETCH,
        "grades": grades, "grader_mid": GRADER_MID, "grader_auc": 0.829,
        "damage_grid": dmg, "mean_hops": hops,
        "legs_partial": legs_partial,
        "flat_curve": [[int(d), round(float(a), 3)] for d, a in flat_aggregation_curve()],
        "worked": _worked_partial_query(),
        "residual_own": round(float(np.mean([residual_norm(L, L["Q"][i], int(d))
                                             for i in _class_idx(L, "bridge")
                                             for d in np.argsort(-leg_dense(L, L["Q"][i]))[:3]])), 3),
        "residual_bridge": round(float(np.mean(
            [residual_norm(L, L["Q"][i], L["bridge_doc"][L["dist_ids"][(n * 9 + 5) % len(L["dist_ids"])]])
             for n, i in enumerate(_class_idx(L, "bridge"))])), 3),
        "reform_eps": REFORM_EPS,
    }


def _print_viz_constants() -> None:
    import json
    print(json.dumps(viz_constants(), indent=1, sort_keys=True))


def _run_tests() -> None:
    names = sorted(n for n, v in globals().items() if n.startswith("test_") and callable(v))
    for n in names:
        globals()[n]()
        print(f"  ok  {n}")
    print(f"{len(names)} assertions passed")


if __name__ == "__main__":
    _diagnostics()
    print()
    print("=" * 96)
    print("THE MECHANISM MATRIX — six architectures, two regimes, both halves of each")
    print("=" * 96)
    print(_fmt_matrix(mechanism_matrix()))
    print()
    print("flat aggregation by hand, in the global regime (depth, accuracy):")
    print("  " + "  ".join(f"{d}:{a:.3f}" for d, a in flat_aggregation_curve())
          + f"   vs graph {run_arm(glob(), 'graph')['acc']:.3f} at depth 0")
    print()
    print("=" * 96)
    _run_tests()
