# Brief — `chunking-as-segmentation`

Phase-A spec for "Chunking as a Segmentation and Optimization Problem." Section outline, theorem
statements + prove/cite, viz intent, notebook map, verified cross-site links, references, and the realized
canonical numbers the viz mirrors. The notebook (`notebooks/chunking-as-segmentation/`) is the source of truth.

## Placeholders

| Field | Value |
|---|---|
| title | Chunking as a Segmentation and Optimization Problem |
| subtitle | Where to cut a document, posed as a coherence-maximizing segmentation with an exact dynamic-programming optimum — and the proxy it secretly optimizes |
| slug | `chunking-as-segmentation` |
| domain | `embedding-geometry` · pipelineStage `ingest` |
| difficulty | `intermediate` · financeCaseStudy `true` · modality `[text]` |
| prerequisites | `hypersphere-vmf-geometry` (published) |
| reference notebooks | `notebooks/hypersphere-vmf-geometry/`, `notebooks/bm25/` |

## Positioning

Chunking is the first thing a RAG pipeline does and the least mathematized — usually "split every 512
tokens." Cast properly it is a one-dimensional **segmentation** problem: choose boundaries to maximize
within-chunk coherence. The coherence has a clean closed form that ties straight to the prerequisite: for
L2-normalized sentence embeddings, the within-segment cost is $\text{len} - \lVert\sum_t e_t\rVert =
\text{len}\cdot(1-\bar R)$, where $\bar R$ is the **mean resultant length** — the von Mises–Fisher
concentration statistic. So minimizing total cost carves the document into tight vMF clusters on the sphere.
The objective is additive, so the global optimum is an exact $O(n^2)$ **dynamic program** (Bellman 1961;
Fisher 1958). The honest twist — the differentiator — is that coherence is a **proxy**: the harness shows
boundary recovery (F1) peaking at the true section count while the coherence cost keeps falling under
over-segmentation, so optimizing the proxy is not the same as optimizing downstream retrieval. Links up into
formalML `clustering` (k-means as the unordered cousin) and `graph-laplacians` (spectral cuts as the higher-dim analog).

## Section outline
1. Overview & motivation (chunking as cutting; embed `<ChunkingLaboratory/>`).
2. Coherence = mean resultant length (Def 1, Prop 1): the vMF tie and the closed form.
3. Optimal segmentation by dynamic programming (Thm 1): the $O(n^2)$ recurrence + optimality proof.
4. Heuristic baselines: TextTiling (greedy) and fixed-size, and the optimality gap.
5. Coherence is a proxy: F1 peaks at the true $k$ while cost keeps dropping (over-segmentation).
6. Finance case study (synthetic 10-K). 7. Honest caveats (`<RigorFlag>`). 8. Implementation.

## Theorems (statement + prove/cite)
- **Def 1** segment cost $c(i,j) = (j-i) - \lVert\sum_{t\in[i,j)} e_t\rVert$ for unit embeddings $e_t$; total cost additive over a segmentation.
- **Prop 1 — coherence is mean resultant length.** $c(i,j) = \sum_{t\in[i,j)}(1-\langle e_t,\mu\rangle)$ for the segment mean direction $\mu=\widehat{\sum e_t}$, so $1 - c(i,j)/(j-i) = \bar R$, the mean resultant length. **PROVE** ($\sum_t\langle e_t,\mu\rangle = \langle\sum_t e_t,\mu\rangle = \lVert\sum_t e_t\rVert$); ties to the [vMF topic](/topics/hypersphere-vmf-geometry). Harness asserts to the decimal.
- **Thm 1 — optimal segmentation by DP.** For additive segment cost, $\mathrm{OPT}(j)=\min_{i<j}\mathrm{OPT}(i)+c(i,j)$ (fixed-$k$: $D[s][j]=\min_{i}D[s-1][i]+c(i,j)$) computes the globally optimal segmentation in $O(n^2)$ (resp. $O(kn^2)$). **PROVE in full** by the exchange/optimal-substructure argument: any optimal segmentation's last segment $[i,j)$ leaves an optimal $(k{-}1)$-segmentation of $[0,i)$, else we could improve it. **Verify against brute force** (the harness matches the exhaustive oracle at every $k$).
- **Cited/empirical:** TextTiling (Hearst 1997) is a greedy depth-score heuristic with no optimality; semantic chunking is the modern practice; the segmentation-as-change-point-detection view (Truong et al. 2020). Presented as heuristics/connections, not theorems.

## rigorFlag (load-bearing)
The dynamic program is **provably optimal — for the chosen objective**, and that objective is a **proxy**. Within-chunk coherence (mean resultant length) is a stand-in for the true target, downstream retrieval and generation quality, and optimizing the proxy does not optimize the goal: the harness shows boundary-recovery F1 *peaking at the true section count* and then *declining* under over-segmentation, even as the coherence cost keeps falling monotonically — minimizing cost past the true $k$ buys lower incoherence and worse structure. The DP's optimality is also only relative to the additive-cost modeling assumptions (additivity across segments; cosine-to-mean-direction coherence; a single similarity metric); TextTiling and fixed-size chunking are heuristics with no optimality guarantee at all. Real documents carry structural boundaries — section headers, speaker turns, tables — that a pure-similarity objective ignores, and the right chunk *size* trades retrieval granularity against context in ways coherence alone does not capture. The finance "10-K" is a synthetic sequence of vMF blocks, not a real filing, and the boundary-recovery numbers are measured on it.

## Viz — "Chunking Laboratory" (`ChunkingLaboratory.tsx`)
Segment-count slider over `K_GRID = [2,3,4,5,6,8,10,12]`; all baked from `grid_table()` on a 32-sentence document. Panels:
(i) **Document & boundaries**: a sentence strip colored by planted segment, with the adjacent-gap dissimilarity profile and the DP / greedy / fixed-size / truth boundary markers — at the true $k=5$ the DP lands exactly on the planted boundaries;
(ii) **Granularity tradeoff**: DP coherence cost (monotone down) and DP boundary-F1 (humped, peaking at the true $k$) vs $k$ — the proxy mismatch made visible;
(iii) **Boundary recovery**: F1 bars, DP vs greedy vs fixed, at the selected $k$.

## Notebook map (`chunking_as_segmentation.py` + `01_…ipynb`)
Deps `numpy`, `scipy`. `synthetic_document` (vMF blocks with planted boundaries); `segment_cost` via `_prefix_sums` (the $O(\mathrm{dim})$ closed form); `segment_dp` (fixed-$k$ DP) + `brute_force_segment` (the oracle); `texttiling_greedy` + `fixed_size` (baselines); `adjacent_dissimilarity`; `boundary_f1`. `grid_table()` → {profile, labels, truth, fixed, perK}; `finance_demo()` on a synthetic 10-K. 6 asserts: `test_coherence_is_resultant_length`, `test_dp_matches_brute_force`, `test_dp_beats_heuristics`, `test_cost_monotone_in_k`, `test_dp_recovers_boundaries`, `test_finance_filing`. Both artifacts exit 0 (~0.1 s). Traps: unit-normalize embeddings (the closed form needs it); use UNEVEN planted segments (equal sections let fixed-size match by accident); F1 peaks at the true $k$, cost is monotone — assert each on the right one.

## Cross-site links (slugs verified present on the siblings)
- `prerequisites`: `hypersphere-vmf-geometry`.
- `connections`: `hypersphere-vmf-geometry` (the within-segment coherence is the mean resultant length / vMF concentration developed there, now a per-segment cost).
- `formalmlPrereqs`: `clustering` (✅ — k-means / Lloyd is the *unordered* cousin of one-dimensional segmentation; segmentation is contiguity-constrained clustering on the sphere).
- `formalmlConnections`: `graph-laplacians` (✅ — spectral clustering and normalized cuts are the higher-dimensional analog of the one-dimensional boundary problem).
- No formalStatistics/formalCalculus prereqs (the DP is combinatorial, the coherence directional) — honest minimalism over a forced link.

## References (Chicago N&B; url/DOI each)
Hearst 1997 (TextTiling); Bellman 1961 (DP optimal curve/segment partition); Fisher 1958 (optimal one-dimensional grouping by DP); Mardia & Jupp 2000 (*Directional Statistics* — mean resultant length, vMF); Truong, Oudre & Vayatis 2020 (offline change-point detection review); Beeferman, Berger & Lafferty 1999 (statistical text segmentation, the $P_k$ metric); LangChain semantic-chunking documentation (the modern embedding-similarity practice).

## Realized canonical numbers (viz mirrors to the decimal) — synthetic document, n=32, 5 vMF sections
- planted boundaries (truth): `[4, 12, 17, 27]`; labels per sentence baked; fixed-size: `[6, 13, 19, 26]` (F1 `0.50`)
- `K_GRID = [2, 3, 4, 5, 6, 8, 10, 12]`
- DP cost (monotone down): `20.447, 18.196, 16.233, 14.783, 13.786, 12.216, 10.781, 9.470`
- DP boundary F1 (humped, peak at k=5): `0.40, 0.667, 0.857, 1.000, 0.889, 0.727, 0.615, 0.533`
- greedy cost: `20.752, 18.484, 16.438, 15.946, 15.533, 13.578, 12.065, 10.619`; greedy F1: `0.40, 0.667, 0.857, 0.750, 0.667, 0.727, 0.615, 0.533`
- at the true k=5: DP boundaries `[4,12,17,27]` (F1 **1.00**, cost **14.783**) vs greedy `[12,13,17,28]` (F1 **0.75**, cost **15.946**) vs fixed `[6,13,19,26]` (F1 **0.50**)
- **Finance headline** (synthetic 10-K, 42 sentences, 6 sections): boundary F1 **DP 80%**, greedy 80%, **fixed-size 20%**; within-chunk incoherence DP **19.285** $\le$ greedy 20.200.
