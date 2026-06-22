# formalRAG topic brief — Multi-Vector ANN: Indexing and Pruning MaxSim at Scale (PLAID)

**Slug:** `multi-vector-ann-retrieval` · **Domain:** `ann-indexing` · **Pipeline stage:** `index` · **Difficulty:** advanced
**Prerequisites:** `late-interaction-learned-sparse`, `product-quantization`, `ivf-voronoi-partitioning`
**Owns the numbers:** `notebooks/multi-vector-ann-retrieval/multi_vector_ann_retrieval.py` → `viz_constants()` → `MultiVectorANNLaboratory.tsx`

## Thesis

Late interaction lifted the single-vector rank ceiling by keeping one contextual vector per token and
scoring by MaxSim, and flagged the bill: an index ~32× a single-vector index and a multi-vector
candidate-generation problem. This topic pays the bill. **PLAID** (the engine behind ColBERTv2) serves
MaxSim at scale by reusing the two ANN prerequisites *verbatim*: cluster every token into one **shared**
centroid set (the IVF coarse quantizer), store each token as a centroid id + a PQ-compressed residual
(IVFADC at the token level), approximate MaxSim by the centroid a token landed in, and prune with a
**Cauchy–Schwarz** error bound — then rerank survivors with full MaxSim. The spine is one clean theorem
(the collapse anchor) plus one exact bound (Cauchy–Schwarz), wrapped around the prerequisite's
`maxsim_score`/`maxsim_matrix` reused as the byte-for-byte anchor. The distinguishing honesty: the
bound controls **scores, not the recall ordering**, which is exactly why Stage 3 reranks — and why the
recall/cost frontier is *demonstrated on one cloud*, not derived.

## Movement 1 — Representation: token IVFADC and the shared-centroid trick

Cluster all corpus token embeddings into one shared centroid set $\mathcal{C}=\{c_1,\dots,c_K\}$ (the
IVF coarse quantizer; ColBERT normalizes tokens, so L2 k-means is cosine clustering at the objective
level — the distance identity $\lVert a-b\rVert^2=2-2\langle a,b\rangle$ holds exactly on unit tokens).
Store each token as its centroid id $c(d_j)$ plus a product-quantized residual $r_j=d_j-c(d_j)$ —
literally IVFADC at the token level. **The shared centroid set is the engineering hinge:** because
tokens across all documents share centroids, the $|Q|\times K$ query-token-to-centroid score table is
computed once and reused across the whole corpus.

**Proposition 1 (storage collapse).** Per token, PLAID stores $\lceil\log_2 K\rceil + m\log_2 k^\*$
bits (centroid id + PQ code) instead of $d\cdot 32$ raw float bits. At ColBERT scale ($d=128$, 32
tokens, $K=2^{16}$, $m=16$, $k^\*=256$): raw multi-vector is **32×** a single-vector index, PLAID is
**1.1×** — a **28×** compression. The flagged 32× debt collapses to roughly a single-vector index.
(`plaid_index`, `reconstruct_doc`, `storage_collapse`; reuses imported `pq_bits`.)

## Movement 2 — The centroid-MaxSim approximation and its Cauchy–Schwarz bound (the clean theorem)

Approximate $\langle q_i, d_j\rangle$ by $\langle q_i, c(d_j)\rangle$ — cheap because it reads off the
shared $Q\!\cdot\!\mathcal{C}$ table.

**Theorem 1 (approximation error is exactly Cauchy–Schwarz).**
$$\langle q_i,d_j\rangle-\langle q_i,c(d_j)\rangle=\langle q_i,r_j\rangle,\qquad |\langle q_i,r_j\rangle|\le\lVert q_i\rVert\,\lVert r_j\rVert.$$
*Proof.* Substitute $d_j=c(d_j)+r_j$ and use bilinearity, then Cauchy–Schwarz. ∎
**Lift to MaxSim.** The max over $j$ is 1-Lipschitz in sup-norm, so per query token the MaxSim-term
error is $\le\lVert q_i\rVert\max_j\lVert r_j\rVert$, and summing,
$|S_{\text{full}}-S_{\text{centroid}}|\le\sum_i\lVert q_i\rVert\max_j\lVert r_j\rVert$. The residual
norm **is** the k-means/PQ distortion inherited from the prereqs.

**Proposition 2 (the bound does not control the ranking).** A uniform additive bound on scores does
*not* preserve the top-k set: two documents within $2\cdot\text{bound}$ can swap rank, and a max of
approximated terms can select a different argmax than the max of true terms. This is the load-bearing
honesty — it is *why* Stage 3 reranks exactly, and why recall is demonstrated, not derived.
(`centroid_maxsim`, `centroid_maxsim_matrix`, `cauchy_schwarz_doc_bound`. Collapse anchor #1:
centroid-MaxSim == imported `maxsim_matrix` on the centroid-substituted documents, <1e-12.)

## Movement 3 — The cascade, and the collapse anchor

Three stages, each a coarsening of the one below: **Stage 1** generates candidates by probing each
query token's nearest centroid lists (the IVF probe at token level); **Stage 2** prunes by the cheap
centroid-MaxSim score, keeping the top `keep`; **Stage 3** decompresses residuals (`pq_decode` +
centroid) and computes full MaxSim (the imported `maxsim_score`) only on the survivors.

**Theorem 2 (collapse to brute-force MaxSim — the one exact statement).** At `nprobe=nlist` (probe
every cell), `keep=N` (prune nothing), and exact-token rerank, the cascade equals brute-force MaxSim to
floating point — recall 1.0 and identical top-k ordering. *Proof.* Each relaxation is a strict superset:
probe-all generates the full candidate set; prune-nothing keeps all; exact rerank scores the true
MaxSim. ∎ Measured: recall 1.000, identical top-10 ordering on all 40 queries.

**Proposition 3 (recall–cost frontier, demonstrated).** With an *exact* rerank, recall is provably
non-decreasing in `keep` (survivor superset → exact top-k can only gain true neighbors): measured
`[0.48, 0.78, 0.94, 0.99, 1.0, 1.0]`. The realistic *lossy-PQ* cascade reaches recall 0.9 at **2.7×**
fewer distance computations than brute on this cloud (centroid-only plateaus at 0.78 at ~30× cheaper;
PLAID climbs past it by reranking). **Robust headline** (intra-family): PLAID reaches a high recall
below brute cost. The cross-comparison is one synthetic vMF token cloud, not a universal ranking.
(`stage1_candidates`, `stage2_prune`, `stage3_rerank`, `plaid_search`, `centroid_only_search`,
`head_to_head`; cost = $m_q\cdot\text{nlist} + m_q\cdot m_d\cdot\text{reranked}$.)

## Viz intent — `MultiVectorANNLaboratory.tsx` (4 panels, constants mirrored to the decimal)

- **A. Token → centroid + residual.** A 2-D toy: tokens around topic directions, k-means centroids,
  each token connected to its centroid by a residual line; slider #centroids K → residual energy falls.
- **B. Centroid-MaxSim grid + bound.** Query×doc grid: true vs centroid-substituted score per cell + a
  Cauchy–Schwarz bar ($\lVert q_i\rVert\lVert r_j\rVert$); highlight where the centroid argmax differs
  from the true argmax (Proposition 2 made visible). Slider K → error and bound shrink together.
- **C. Recall vs cost (centerpiece).** Three series — brute MaxSim (anchor at recall 1.0), centroid-only
  (ceiling 0.78), PLAID cascade (knee). Slider prune depth → at max the marker sits on the brute line
  (the collapse anchor, interactive).
- **D. Storage.** Bars: single / raw-multi (32×) / PLAID (~1×, 28× compression) — consistent with the
  late-interaction lab's storage readout, closing the flagged 32× debt.

## rigorFlag (IVF/late-interaction register: one theorem, one bound, then honest caveats)

One exact theorem (the collapse anchor) and one exact bound (Cauchy–Schwarz per-pair, lifted to the
document score). The bound controls scores, **not** the per-query-token max, the argmax token, or the
post-prune recall ordering — small per-pair error does not imply correct top-k. The recall/cost frontier
is one synthetic vMF token cloud (the knee, the centroid-only ceiling, and the catch-up depth all move
with the corpus, K, residual budget, and prune depth). Tokens are synthetic vMF stand-ins reused from
the hypersphere/IVF/PQ/late-interaction topics, not a trained ColBERT. The storage win is mitigation
(~order of magnitude), not erasure: still many vectors per document, lossy by construction.

## References (arXiv + DOIs)

- Santhanam, Khattab, Potts & Zaharia (2022), *PLAID: An Efficient Engine for Late Interaction
  Retrieval*, CIKM — `arXiv:2205.09707`. **Load-bearing:** centroid interaction, centroid pruning,
  residual decompression staged into the cascade.
- Santhanam, Khattab, Saad-Falcon, Potts & Zaharia (2022), *ColBERTv2*, NAACL — `arXiv:2112.01488`.
  The centroid-plus-residual token compression PLAID serves.
- Khattab & Zaharia (2020), *ColBERT*, SIGIR — `arXiv:2004.12832`. The MaxSim operator approximated/pruned.
- Jégou, Douze & Schmid (2011), *Product Quantization for NN Search*, IEEE TPAMI —
  `10.1109/TPAMI.2010.57`. IVFADC: the coarse k-means + inverted lists + PQ residuals applied at token level.
- Johnson, Douze & Jégou (2021), *Billion-Scale Similarity Search with GPUs (Faiss)*, IEEE TBDATA —
  `10.1109/TBDATA.2019.2921572`. How production indexes implement the IVFPQ token store.

## Cross-site (UP into formalML; verify slugs exist)

- `formalmlPrereqs`: `clustering` (the shared centroids are k-means; centroid = cell mean makes it a
  meaningful stand-in), `rate-distortion` (token = centroid id + residual code is a rate–distortion
  quantizer trading bits against the residual energy the approximation pays for).
- `connections` (in-site, shipped slugs only): `late-interaction-learned-sparse`,
  `ivf-voronoi-partitioning`, `product-quantization`, `navigable-small-world-graphs` (graph vs partition
  candidate generation). The capstone is named in prose without a link (unbuilt).
