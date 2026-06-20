# Brief — `johnson-lindenstrauss`

Phase-A spec for "Random Projections and the Johnson–Lindenstrauss Lemma." Section outline, theorem
statements + prove/cite, viz intent, notebook map, verified cross-site links, references, and the realized
canonical numbers the viz mirrors. The notebook (`notebooks/johnson-lindenstrauss/`) is the source of truth.

## Placeholders

| Field | Value |
|---|---|
| title | Random Projections and the Johnson–Lindenstrauss Lemma |
| subtitle | Reducing embedding dimension with a matrix that never saw the data — and the dimension-independent price of preserving every distance |
| slug | `johnson-lindenstrauss` |
| domain | `embedding-geometry` · pipelineStage `index` |
| difficulty | `intermediate` · financeCaseStudy `true` · modality `[text]` |
| prerequisites | `pca-dimensionality-reduction` (published) |
| reference notebooks | `notebooks/pca-dimensionality-reduction/`, `notebooks/high-dimensional-geometry/` |

## Positioning

PCA reduces dimension by reading the cloud's own covariance — *data-dependent*, variance-optimal, and it
must see the data first. Johnson–Lindenstrauss does the opposite: multiply by a *random* matrix that has
never seen the data, and still preserve every pairwise distance to within $(1\pm\varepsilon)$. The headline
is dimension independence — the target $k = O(\varepsilon^{-2}\log n)$ depends on $\log n$ and $\varepsilon$,
**not** on the ambient $d$. The honest twists are the differentiators: the bound is worst-case (the union
bound bites — the *typical* pair distorts far less than the worst), it preserves *distances* not *rankings*,
and being data-oblivious it sacrifices the retrieval recall data-dependent PCA keeps. Links up into formalML's
`concentration-inequalities` (the $\chi^2$/sub-Gaussian tail is the whole engine) and contrasts with
`pca-low-rank`. Feeds forward to `locality-sensitive-hashing` (referenced in prose; not yet a `connections`
edge since LSH is unpublished).

## Section outline (overview + theorems + caveats + implementation)
1. Overview & motivation (PCA is data-dependent; can a *random* map work? embed `<RandomProjectionLaboratory/>`).
2. The distributional JL property — one vector (Def 1, Thm 1: $\mathbb E\lVert f(x)\rVert^2=\lVert x\rVert^2$, $\lVert f(x)\rVert^2/\lVert x\rVert^2\sim\chi^2_k/k$).
3. Concentration: the $\chi^2_k$ tail (Thm 2, Laurent–Massart) — why it concentrates at $e^{-ck\varepsilon^2}$.
4. The JL lemma proper (Thm 3): union bound over $\binom n2$ difference vectors $\Rightarrow k\ge 4\ln n/(\varepsilon^2/2-\varepsilon^3/3)$.
5. Dimension independence, and the loose-constant honesty (typical vs worst-case pair; the bound is conservative).
6. Sub-Gaussian and database-friendly variants (Prop 1: Rademacher, sparse Achlioptas — prove unbiasedness, cite the tail).
7. Optimality (Prop 2: $k=\Theta(\varepsilon^{-2}\log n)$ is tight — cite Larsen–Nelson).
8. Data-oblivious vs data-dependent: distances vs rankings, recall, the contrast with PCA.
9. Finance case study. 10. Honest caveats (`<RigorFlag>`). 11. Implementation.

## Theorems (statement + prove/cite)
- **Def 1** the random map $f(x)=\tfrac1{\sqrt k}Ax$, $A\in\mathbb R^{k\times d}$ i.i.d. $\mathcal N(0,1)$. **Def 2** $(1\pm\varepsilon)$-embedding of a point set.
- **Thm 1 — distributional JL.** $\mathbb E\lVert f(x)\rVert^2=\lVert x\rVert^2$ and $\lVert f(x)\rVert^2/\lVert x\rVert^2\sim\chi^2_k/k$. **PROVE in full**: each $a_i\cdot x/\lVert x\rVert\sim\mathcal N(0,1)$ i.i.d. across the $k$ rows, so the normalized squared norm is a sum of $k$ squared standard normals. (Routes the "sum of squared Gaussians" fact to formalStatistics.)
- **Thm 2 — concentration (Laurent–Massart).** $\Pr[\chi^2_k/k-1\ge 2\sqrt{x/k}+2x/k]\le e^{-x}$ and the matching lower tail. **STATE + PROVE the load-bearing upper tail** via the $\chi^2$ moment generating function / Chernoff bound (mirror the concentration machinery of `high-dimensional-geometry`); **CITE Laurent–Massart** for the sharp constants; verify the bound numerically in the harness.
- **Thm 3 — the JL lemma.** For $n$ points and $\varepsilon\in(0,1)$, $k\ge 4\ln n/(\varepsilon^2/2-\varepsilon^3/3)$ gives a single $f$ that preserves **all** $\binom n2$ pairwise squared distances to $(1\pm\varepsilon)$ with positive probability. **PROVE in full**: apply Thm 2 to each difference vector $u-v$, set the per-pair failure $\le 2e^{-k(\varepsilon^2/2-\varepsilon^3/3)}$, union bound over $<n^2/2$ pairs $\Rightarrow$ total failure $<1$.
- **Prop 1 — sub-Gaussian / sparse variants.** Rademacher $\pm1$ and sparse Achlioptas ($s=3$) maps satisfy $\mathbb E\lVert f(x)\rVert^2=\lVert x\rVert^2$. **PROVE unbiasedness** ($\mathbb E[A_{ij}^2]=1$); **CITE Achlioptas (2003)** for the full sub-Gaussian tail; verify same concentration spread numerically.
- **Prop 2 — optimality.** $k=\Theta(\varepsilon^{-2}\log n)$ is tight. **STATE + CITE Larsen–Nelson (2017).**

## rigorFlag (load-bearing — 5 honest joints)
The constant in $k\ge 4\ln n/(\varepsilon^2/2-\varepsilon^3/3)$ is worst-case and **loose**: for $n=500,\varepsilon=0.2$ the guarantee demands $k\ge 1435$, yet at a practical $k=128$ the *typical* pair distorts only $\sim 9\%$ — though the *worst* of $124{,}750$ pairs still distorts $\sim 51\%$, so the union bound is doing real work. JL preserves **Euclidean distances**, hence inner products of *difference* vectors; cosine/angles inherit the same $\varepsilon$ slack only after the metric translation, and exact nearest-neighbor **rankings are not preserved** — recall@10 at $k=128$ is $\sim 23\%$ for random projection vs $\sim 96\%$ for data-dependent PCA, the oblivious price. The guarantee is for a **fixed** point set (queries are covered only via the distributional bound). The $\chi^2_k/k$ law is **exact for the Gaussian map**; Rademacher/sparse match its first two moments and concentrate alike (cited, not re-proved). The finance embeddings are a **synthetic** low-rank-plus-noise stand-in, not a trained encoder.

## Viz — "Random Projection Laboratory" (`RandomProjectionLaboratory.tsx`)
Target-dimension slider over `K_GRID = [8,16,32,64,128,256,512]`; all data **baked to the decimal** from `grid_table()`. Panels:
(i) distortion histogram at the selected $k$ (baked at $k\in\{32,256\}$; the $\pm\varepsilon$ band overlaid) tightening toward 1 as $k$ grows — the visual proof of concentration;
(ii) worst-pair vs typical distortion (max-abs-dev and the p01/p99 band) vs $k$, with the guaranteed-$k$ threshold curve $k=4\ln n/(\varepsilon^2/2-\varepsilon^3/3)$ overlaid for the eps grid;
(iii) projection-family comparison (Gaussian / Rademacher / sparse std at $k=64$) — they coincide;
(iv) recall@10 retained vs $k$, random projection (oblivious) vs PCA (dependent) — the honest gap.

## Notebook map (`johnson_lindenstrauss.py` + `01_…ipynb`)
Deps `numpy`, `scipy`, **`scikit-learn`** (cross-check). `structured_embeddings` (the SAME generator as the PCA topic, for an apples-to-apples oblivious-vs-dependent comparison); `gaussian_/rademacher_/sparse_achlioptas_projection` via one `_draw(gen,…)` so Monte-Carlo sweeps share a single RNG stream (per-seed `default_rng(s)` over consecutive $s$ inflates sampled variance with $d$ — a real NumPy gotcha, flagged in-file); `pairwise_sq_distortion`; `jl_min_dim`; `laurent_massart_upper`; `recall_after_projection` (random vs PCA). `grid_table()` → {distortion, hist, threshold, families, recall}; `finance_demo()` at $d{=}1536$, $n{=}500$, $k{=}128$. 8 asserts: `test_expected_norm_preservation`, `test_chi_square_distribution`, `test_jl_lemma_holds`, `test_dimension_independence`, `test_projection_families_agree`, `test_recall_after_projection`, `test_sklearn_crosscheck`, `test_finance_distortion`. Both artifacts exit 0 (~7 s). Traps: single RNG stream for averaging loops; distortion is on **squared** distances; the worst pair (max) is the union-bound quantity, the mean/spread is the concentration quantity — assert each on the right one.

## Cross-site links
- `prerequisites`: `pca-dimensionality-reduction`.
- `connections`: `pca-dimensionality-reduction` (data-dependent vs oblivious; PCA keeps recall JL sacrifices), `high-dimensional-geometry` (its concentration-of-measure is the engine JL's $\chi^2$ tail instantiates).
- `formalmlPrereqs`: `concentration-inequalities` (✅ used by both published siblings — the $\chi^2$/sub-Gaussian tail), `pca-low-rank` (✅ — the data-dependent contrast).
- `formalstatisticsPrereqs`: **verify in B7** a chi-square / sampling-distributions slug (the $\chi^2_k$ law); fall back to `multivariate-distributions` (✅ used by PCA).
- `formalcalculusPrereqs`: optional — a `change-of-variables`/integration slug for the Gaussian MGF (✅ `eigenvalues-eigenvectors`/`convex-optimization` exist; verify a calculus integration slug before citing, else omit).
- Forward edge `johnson-lindenstrauss → locality-sensitive-hashing` already in the graph; **no `connections` entry** until LSH ships — reference in prose only.

## References (Chicago N&B; url/DOI each)
Johnson & Lindenstrauss 1984 (original lemma); Dasgupta & Gupta 2003 (elementary Gaussian proof); Achlioptas 2003 (database-friendly / sparse); Laurent & Massart 2000 ($\chi^2$ tail bounds); Larsen & Nelson 2017 (optimality lower bound); Indyk & Motwani 1998 (JL → approximate NN / LSH); Vempala 2004 (*The Random Projection Method*); Bingham & Mannila 2001 (random projection in practice); Li, Hastie & Church 2006 (very sparse random projections).

## Realized canonical numbers (viz mirrors to the decimal) — finance cloud, d=1536, n=500, intrinsic 48, 3 clusters
- `K_GRID = [8,16,32,64,128,256,512]`
- distortion **mean**: `0.9798, 0.9611, 0.9475, 0.9572, 0.9620, 1.0063, 1.0116`
- distortion **std** (concentration, → √(2/k)): `0.4070, 0.2996, 0.2085, 0.1426, 0.1062, 0.0777, 0.0556`
- **p01 / p99** band: p01 `0.2436, 0.3901, 0.5246, 0.6604, 0.7518, 0.8402, 0.8857`; p99 `2.1230, 1.7610, 1.5286, 1.3312, 1.2485, 1.2026, 1.1470`
- **max abs dev** (worst pair, union-bound): `2.7370, 1.9258, 1.3313, 0.7104, 0.5112, 0.4985, 0.3092`
- **recall@10** random: `0.046, 0.033, 0.104, 0.129, 0.233, 0.333, 0.549`; PCA: `0.427, 0.547, 0.845, 0.956, 0.960, 0.972, 1.000`
- **guaranteed k** by eps (n=500): `eps=0.1→5327, 0.15→2456, 0.2→1435, 0.3→691, 0.5→299`
- **family std** (k=64): gaussian `0.1540`, rademacher `0.1658`, sparse `0.1622`
- **Finance headline** (k=128, eps=0.2): guaranteed k **1435** (worst-case, all 124 750 pairs); typical pair distortion **9.2%** vs worst-pair **51.1%**; recall@10 **random 23.3%** vs **PCA 96.0%**.
