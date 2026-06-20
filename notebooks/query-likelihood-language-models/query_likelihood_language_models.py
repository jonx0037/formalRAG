"""Query-Likelihood Language Models and Smoothing — reference implementation.

This module owns every number the topic depends on. Core scoring uses only numpy;
the verification harness asserts the topic's mathematical claims:

  * the query-likelihood score is log P(q | M_d) under a multinomial unigram model
    (Definition), and the maximum-likelihood estimate P_ml(t|d) = tf/|d| already
    divides by document length, so QL is NOT hijacked by the padded transcript that
    captures the raw tf-idf dot product — but it suffers the zero-frequency
    catastrophe instead (any unseen query term sends log P(q|d) to -infinity);
  * Dirichlet smoothing P_mu(t|d) = (tf + mu P(t|C)) / (|d| + mu) is exactly the
    posterior mean of the multinomial parameter under a conjugate Dirichlet prior
    with concentration alpha_t = mu P(t|C) (Theorem 1);
  * ranking by the negative KL divergence -KL(theta_q || theta_d) is rank-equivalent
    to the query-likelihood, because the query-entropy term is constant across
    documents (Theorem 2) — the view that RM3 (pseudo-relevance-feedback) generalizes;
  * the Jelinek-Mercer log score decomposes into a sum over MATCHED query terms whose
    per-term weight rises as the collection probability P(t|C) falls — smoothing
    plays an IDF-like role (Theorem 3, Zhai-Lafferty), plus a query-length constant;
  * Dirichlet smoothing is length-adaptive: the effective interpolation weight
    mu/(|d|+mu) is larger for short documents, and the score carries an explicit
    length penalty |q| log(mu/(|d|+mu)) that grows with |d| (Proposition 1).

The finance corpus is reused VERBATIM from notebooks/bm25/bm25.py and
notebooks/vector-space-model-tfidf/vector_space_model_tfidf.py so the contrast with
the length-hijack flip is provably the same corpus: the lexical-retrieval arc
the-retrieval-problem -> VSM -> BM25 -> query-likelihood scores one corpus with
escalating models.

mu and lambda are tuned empirically, not derived (rigorFlag); the unigram model
assumes term independence, which is false for real text (rigorFlag); "IDF-like" is
an emergent property of the decomposition, not a derived IDF (rigorFlag).

Run:  uv run --with numpy python notebooks/query-likelihood-language-models/query_likelihood_language_models.py
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Tokenization + inverted index  (same shape as the VSM / BM25 notebooks)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class InvertedIndex:
    vocab: dict[str, int]            # term -> column index
    tf: np.ndarray                   # (n_docs, n_terms) raw term frequencies
    doc_len: np.ndarray             # (n_docs,) document lengths
    cf: np.ndarray                  # (n_terms,) collection frequencies (total count)
    coll_len: float                 # total tokens in the collection
    n_docs: int
    doc_ids: list[str]


def build_inverted_index(corpus: dict[str, str]) -> InvertedIndex:
    doc_ids = list(corpus)
    tokenized = [tokenize(corpus[d]) for d in doc_ids]
    vocab: dict[str, int] = {}
    for toks in tokenized:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)

    tf = np.zeros((len(doc_ids), len(vocab)))
    doc_len = np.zeros(len(doc_ids))
    for i, toks in enumerate(tokenized):
        doc_len[i] = len(toks)
        for term, count in Counter(toks).items():
            tf[i, vocab[term]] = count
    cf = tf.sum(axis=0)
    return InvertedIndex(vocab, tf, doc_len, cf, float(doc_len.sum()),
                         len(doc_ids), doc_ids)


# --------------------------------------------------------------------------- #
# Language models: collection, MLE, and the two smoothings
# --------------------------------------------------------------------------- #

def collection_model(index: InvertedIndex) -> np.ndarray:
    """P(t | C) = cf_t / |C|, the maximum-likelihood collection (background) model."""
    return index.cf / index.coll_len


def mle_model(index: InvertedIndex) -> np.ndarray:
    """P_ml(t | d) = tf_{t,d} / |d|, the unsmoothed document model (0 for |d|=0)."""
    dl = np.where(index.doc_len > 0, index.doc_len, 1.0)
    return index.tf / dl[:, None]


def jelinek_mercer(index: InvertedIndex, lam: float) -> np.ndarray:
    """P_lambda(t|d) = (1-lam) P_ml(t|d) + lam P(t|C).  lam in (0,1)."""
    pc = collection_model(index)
    return (1.0 - lam) * mle_model(index) + lam * pc[None, :]


def dirichlet(index: InvertedIndex, mu: float) -> np.ndarray:
    """P_mu(t|d) = (tf_{t,d} + mu P(t|C)) / (|d| + mu).  mu > 0."""
    pc = collection_model(index)
    num = index.tf + mu * pc[None, :]
    den = (index.doc_len + mu)[:, None]
    return num / den


def smoothed_model(index: InvertedIndex, method: str, param: float) -> np.ndarray:
    if method == "none":
        return mle_model(index)
    if method == "jm":
        return jelinek_mercer(index, param)
    if method == "dirichlet":
        return dirichlet(index, param)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# Query likelihood and the KL view
# --------------------------------------------------------------------------- #

def query_counts(query: str, index: InvertedIndex) -> np.ndarray:
    """Raw query term counts c(t,q) over the vocabulary (unseen query terms drop)."""
    qc = np.zeros(len(index.vocab))
    for term, count in Counter(tokenize(query)).items():
        j = index.vocab.get(term)
        if j is not None:
            qc[j] = count
    return qc


def ql_logscore(query: str, index: InvertedIndex, method: str, param: float) -> np.ndarray:
    """log P(q | M_d) = sum_t c(t,q) log P_smooth(t|d) for every document.

    With method='none' a document missing any query term scores -inf (the
    zero-frequency catastrophe). np.errstate silences the deliberate log(0).
    """
    pm = smoothed_model(index, method, param)
    qc = query_counts(query, index)
    cols = np.nonzero(qc)[0]
    with np.errstate(divide="ignore"):
        logp = np.log(pm[:, cols])               # (n_docs, n_query_terms)
    return (logp * qc[cols][None, :]).sum(axis=1)


def ql_rank(query: str, index: InvertedIndex, method: str = "dirichlet",
            param: float = 2000.0) -> list[tuple[str, float]]:
    scores = ql_logscore(query, index, method, param)
    order = np.argsort(scores)[::-1]
    return [(index.doc_ids[i], float(scores[i])) for i in order]


def ql_contributions(query: str, doc_i: int, index: InvertedIndex,
                     method: str = "dirichlet", param: float = 2000.0) -> dict[str, float]:
    """Per-query-term log-probability contributions for one document (feeds the viz)."""
    pm = smoothed_model(index, method, param)
    out: dict[str, float] = {}
    for term in dict.fromkeys(tokenize(query)):
        j = index.vocab.get(term)
        if j is not None:
            out[term] = float(math.log(pm[doc_i, j]))
    return out


def neg_kl_score(query: str, index: InvertedIndex, method: str, param: float) -> np.ndarray:
    """-KL(theta_q || theta_d) where theta_q(t) = c(t,q)/|q| is the empirical query model.

    -KL = -sum_t theta_q(t) log(theta_q(t)/P_smooth(t|d))
        = H(theta_q) + sum_t theta_q(t) log P_smooth(t|d)
    The first term H(theta_q) is constant across documents, so ranking by -KL is
    rank-equivalent to the query likelihood (Theorem 2).
    """
    pm = smoothed_model(index, method, param)
    qc = query_counts(query, index)
    qlen = qc.sum()
    theta_q = qc / qlen
    cols = np.nonzero(qc)[0]
    h_q = -(theta_q[cols] * np.log(theta_q[cols])).sum()      # query entropy (constant)
    with np.errstate(divide="ignore"):
        cross = (theta_q[cols][None, :] * np.log(pm[:, cols])).sum(axis=1)
    return h_q + cross


# --------------------------------------------------------------------------- #
# Shared finance corpus — VERBATIM from notebooks/bm25/bm25.py
# --------------------------------------------------------------------------- #

_FILLER = ("the quarter operations regions headcount capex marketing supply chain inventory "
           "guidance segment logistics demand pricing momentum ") * 15  # ~240 query-free tokens
_FINANCE_CORPUS = {
    "filing-onpoint": "net interest margin is sensitive to interest rate moves and our rate exposure is disclosed",
    "transcript-pad": _FILLER + " interest interest interest rate rate rate rate exposure exposure",
    "filing-fx": "foreign exchange exposure and currency rate movements affect our exposure to translation",
    "news-macro": "the central bank raised the policy rate again as interest rate decisions weigh on markets",
    "filing-boiler": "this filing contains forward looking statements regarding interest rate and other risk exposure factors",
    "transcript-short": "quick update on interest and the rate outlook for the quarter",
}
_QUERY = "interest rate exposure"


# --------------------------------------------------------------------------- #
# Verification harness — the topic's claims, made executable
# --------------------------------------------------------------------------- #

def test_zero_frequency_catastrophe() -> None:
    """Definition + motivation: under the unsmoothed MLE, any document missing a
    query term gets P_ml = 0 -> log P(q|d) = -inf; smoothing makes every score finite.
    On this corpus exactly the documents missing 'interest', 'rate' or 'exposure'
    are killed, while the documents containing all three remain finite."""
    index = build_inverted_index(_FINANCE_CORPUS)
    raw = ql_logscore(_QUERY, index, "none", 0.0)
    finite = {index.doc_ids[i] for i in range(index.n_docs) if np.isfinite(raw[i])}
    killed = {index.doc_ids[i] for i in range(index.n_docs) if not np.isfinite(raw[i])}
    assert finite == {"filing-onpoint", "transcript-pad", "filing-boiler"}, finite
    assert killed == {"filing-fx", "news-macro", "transcript-short"}, killed
    for method, param in (("jm", 0.5), ("dirichlet", 2000.0)):
        s = ql_logscore(_QUERY, index, method, param)
        assert np.all(np.isfinite(s)), f"{method} left a -inf score"
    print(f"  [ok] zero-frequency catastrophe: MLE kills {len(killed)}/6 docs; "
          f"smoothing rescues all 6")


def test_dirichlet_is_posterior_mean() -> None:
    """Theorem 1: Dirichlet smoothing is the Bayesian posterior mean of the
    multinomial parameter theta_t under a conjugate Dirichlet prior with
    alpha_t = mu P(t|C). Posterior is Dirichlet(alpha_t + tf_t); its mean is
    (alpha_t + tf_t) / sum_t(alpha_t + tf_t) = (tf_t + mu P(t|C)) / (|d| + mu)."""
    index = build_inverted_index(_FINANCE_CORPUS)
    pc = collection_model(index)
    for mu in (10.0, 500.0, 2000.0):
        alpha = mu * pc                                   # prior pseudo-counts, sum = mu
        assert abs(alpha.sum() - mu) < 1e-9
        for i in range(index.n_docs):
            post = alpha + index.tf[i]                    # conjugate update
            post_mean = post / post.sum()                 # Dirichlet mean
            smoothed = dirichlet(index, mu)[i]
            assert np.allclose(post_mean, smoothed, atol=1e-12), (mu, i)
        # every smoothed document model is a valid distribution
        assert np.allclose(dirichlet(index, mu).sum(axis=1), 1.0, atol=1e-9)
    print("  [ok] Dirichlet smoothing == posterior mean under a conjugate Dirichlet prior")


def test_kl_rank_equivalence() -> None:
    """Theorem 2: ranking by -KL(theta_q||theta_d) equals ranking by query
    likelihood. Tested on the worked example AND on random strict instances so a
    tie cannot mask a real disagreement."""
    index = build_inverted_index(_FINANCE_CORPUS)
    for method, param in (("jm", 0.3), ("dirichlet", 1500.0)):
        ql = [d for d, _ in ql_rank(_QUERY, index, method, param)]
        kl_scores = neg_kl_score(_QUERY, index, method, param)
        kl = [index.doc_ids[i] for i in np.argsort(kl_scores)[::-1]]
        assert ql == kl, f"{method}: QL {ql} != -KL {kl}"
    # Random strict instances (synthetic corpora + queries), no shared corpus.
    rng = np.random.default_rng(20260620)
    vocab = [f"w{k}" for k in range(12)]
    checked = 0
    while checked < 200:
        docs = {f"d{j}": " ".join(rng.choice(vocab, size=int(rng.integers(8, 40)))) for j in range(6)}
        idx = build_inverted_index(docs)
        q = " ".join(rng.choice(vocab, size=3))
        if query_counts(q, idx).sum() == 0:
            continue
        s_ql = ql_logscore(q, idx, "dirichlet", 1200.0)
        s_kl = neg_kl_score(q, idx, "dirichlet", 1200.0)
        # require strict ordering (no ties) so an order comparison is meaningful
        if len(set(np.round(s_ql, 9))) < idx.n_docs:
            continue
        assert list(np.argsort(s_ql)[::-1]) == list(np.argsort(s_kl)[::-1])
        checked += 1
    print(f"  [ok] -KL ranking == QL ranking (worked example + {checked} strict random instances)")


def test_jm_decomposition_idf_effect() -> None:
    """Theorem 3 (Zhai-Lafferty): for Jelinek-Mercer the log score splits into a
    document-dependent matched-term sum plus a document-independent constant,
        log P(q|d) = sum_{t in q, tf>0} c(t,q) log(1 + ((1-lam)/lam) P_ml(t|d)/P(t|C))
                     + |q| log lam + sum_{t in q} c(t,q) log P(t|C).
    Only the first sum varies across documents, so it alone determines the ranking;
    inside it the per-term weight rises as the collection probability P(t|C) falls,
    an IDF-like effect. The bracketed remainder is constant across documents."""
    index = build_inverted_index(_FINANCE_CORPUS)
    lam = 0.4
    pc = collection_model(index)
    pml = mle_model(index)
    qc = query_counts(_QUERY, index)
    cols = np.nonzero(qc)[0]
    qlen = qc.sum()
    const_term = qlen * math.log(lam) + sum(qc[j] * math.log(pc[j]) for j in cols)
    direct = ql_logscore(_QUERY, index, "jm", lam)
    matched_part = np.empty(index.n_docs)
    for i in range(index.n_docs):
        matched = [j for j in cols if pml[i, j] > 0]
        matched_part[i] = sum(qc[j] * math.log(1.0 + ((1 - lam) / lam) * pml[i, j] / pc[j])
                              for j in matched)
        assert abs(matched_part[i] + const_term - direct[i]) < 1e-9, (index.doc_ids[i], direct[i])
    # The document-dependent matched-term sum alone reproduces the QL ranking.
    assert (list(np.argsort(matched_part)[::-1])
            == list(np.argsort(direct)[::-1])), "matched-term sum does not reproduce QL order"
    # IDF-like monotonicity: at a fixed matched tf, the per-term weight is strictly
    # decreasing in the collection probability P(t|C).
    fixed_pml = 0.05
    pcs = np.array([0.001, 0.01, 0.05, 0.2])
    weights = np.log1p(((1 - lam) / lam) * fixed_pml / pcs)
    assert np.all(np.diff(weights) < 0), weights
    print("  [ok] JM log score = matched-term sum + doc-independent constant; "
          "weight ~ 1/P(t|C) (IDF-like)")


def test_dirichlet_length_adaptivity() -> None:
    """Proposition 1: the Dirichlet effective interpolation weight mu/(|d|+mu) is
    strictly decreasing in |d| (short docs are smoothed more), and the score's
    length-penalty term |q| log(mu/(|d|+mu)) is strictly more negative for longer
    documents — Dirichlet smoothing normalizes length the way BM25's b does."""
    index = build_inverted_index(_FINANCE_CORPUS)
    mu = 2000.0
    # The effective interpolation weight mu/(L+mu) is a strictly decreasing function
    # of length L (checked on distinct lengths).
    lengths = np.array([5.0, 10.0, 20.0, 50.0, 250.0])
    assert np.all(np.diff(mu / (lengths + mu)) < 0), "mu/(L+mu) not strictly decreasing in L"
    # On the corpus it is non-increasing in |d| (two equal-length docs tie) and the
    # shortest document is smoothed strictly more than the longest.
    dl = index.doc_len
    eff = mu / (dl + mu)
    order = np.argsort(dl)
    assert np.all(np.diff(eff[order]) <= 1e-15), "effective smoothing increased with |d|"
    assert eff[np.argmin(dl)] > eff[np.argmax(dl)], "shortest doc not smoothed more than longest"
    penalty = query_counts(_QUERY, index).sum() * np.log(eff)
    assert penalty[np.argmax(dl)] < penalty[np.argmin(dl)], "longest doc not penalized most"
    print("  [ok] Dirichlet is length-adaptive: shorter docs smoothed more, long docs penalized")


def test_ql_resists_length_hijack() -> None:
    """The headline contrast with VSM: the raw tf-idf dot product puts the padded
    transcript at #1 (length hijack); query likelihood with either smoothing puts
    the concise on-point filing at #1, because P_ml = tf/|d| already normalizes
    length. RUN, not assumed."""
    index = build_inverted_index(_FINANCE_CORPUS)
    for method, param in (("jm", 0.5), ("dirichlet", 2000.0)):
        top = ql_rank(_QUERY, index, method, param)[0][0]
        assert top == "filing-onpoint", f"{method} top = {top}"
        # the padded transcript must not win
        ranked = [d for d, _ in ql_rank(_QUERY, index, method, param)]
        assert ranked.index("transcript-pad") > 0, f"{method}: transcript-pad still #1"
    print("  [ok] QL is not length-hijacked: on-point filing #1, padded transcript demoted")


def validate_normalization() -> None:
    """Cross-check: every smoothed document model is a proper probability
    distribution (sums to 1, all entries in [0,1]); the JM model is exactly the
    stated convex combination of the MLE and collection models."""
    index = build_inverted_index(_FINANCE_CORPUS)
    for method, param in (("jm", 0.2), ("jm", 0.7), ("dirichlet", 100.0), ("dirichlet", 3000.0)):
        pm = smoothed_model(index, method, param)
        assert np.allclose(pm.sum(axis=1), 1.0, atol=1e-9), method
        assert np.all(pm >= 0.0) and np.all(pm <= 1.0 + 1e-12), method
    lam = 0.35
    pc, pml = collection_model(index), mle_model(index)
    assert np.allclose(jelinek_mercer(index, lam), (1 - lam) * pml + lam * pc[None, :])
    print("  [ok] smoothed models are valid distributions; JM is the stated convex mixture")


# --------------------------------------------------------------------------- #
# Demos (printed, non-asserting) — feed the viz panels
# --------------------------------------------------------------------------- #

def smoothing_demo() -> None:
    index = build_inverted_index(_FINANCE_CORPUS)
    print(f"  query={_QUERY!r}  N={index.n_docs}  |C|={int(index.coll_len)}")
    print("  log P(q|d) by model (higher is better; -inf = unseen query term):")
    for method, param, tag in (("none", 0.0, "MLE (no smoothing)"),
                               ("jm", 0.5, "Jelinek-Mercer lam=0.5"),
                               ("dirichlet", 2000.0, "Dirichlet mu=2000")):
        order = ql_rank(_QUERY, index, method, param)
        rows = "  ".join(f"{d}:{('-inf' if not math.isfinite(s) else f'{s:.2f}')}" for d, s in order[:3])
        print(f"    {tag:<26} #1={order[0][0]:<16} top3 {rows}")


def collection_table() -> None:
    """Background probability and self-information of each query term — feeds the
    viz IDF-like panel."""
    index = build_inverted_index(_FINANCE_CORPUS)
    pc = collection_model(index)
    for term in dict.fromkeys(tokenize(_QUERY)):
        j = index.vocab[term]
        print(f"    {term:<9} cf={int(index.cf[j]):<3} P(t|C)={pc[j]:.5f}  "
              f"-log P(t|C)={-math.log(pc[j]):.3f} nats")


def viz_constants() -> None:
    """The exact numbers QueryLikelihoodSmoothingLab.tsx mirrors to the decimal.
    The per-document query-term tf and |d| are shown in the viz; the collection
    model P(t|C) (which depends on the whole corpus incl. filler) and the document
    lengths are the mirrored constants the viz cannot recompute from the query terms."""
    index = build_inverted_index(_FINANCE_CORPUS)
    pc = collection_model(index)
    qterms = list(dict.fromkeys(tokenize(_QUERY)))
    print(f"  coll_len={int(index.coll_len)}")
    print(f"  P(t|C): " + "  ".join(f"{t}={pc[index.vocab[t]]:.6f}" for t in qterms))
    print(f"  doc_len: " + "  ".join(f"{d}={int(L)}" for d, L in zip(index.doc_ids, index.doc_len)))
    print("  query-term tf per doc:")
    for i, d in enumerate(index.doc_ids):
        cells = "  ".join(f"{t}={int(index.tf[i, index.vocab[t]])}" for t in qterms)
        print(f"    {d:<16} |d|={int(index.doc_len[i]):<4} {cells}")


if __name__ == "__main__":
    print("Query-Likelihood Language Models / smoothing verification harness")
    test_zero_frequency_catastrophe()
    test_dirichlet_is_posterior_mean()
    test_kl_rank_equivalence()
    test_jm_decomposition_idf_effect()
    test_dirichlet_length_adaptivity()
    test_ql_resists_length_hijack()
    validate_normalization()
    print("Smoothing demo (log P(q|d) by model):")
    smoothing_demo()
    print("Collection model (background probability of each query term):")
    collection_table()
    print("Viz constants (mirrored to the decimal in QueryLikelihoodSmoothingLab.tsx):")
    viz_constants()
    print("All checks passed.")
