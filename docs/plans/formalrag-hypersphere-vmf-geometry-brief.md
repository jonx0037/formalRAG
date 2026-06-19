# Brief — `hypersphere-vmf-geometry`

Phase-A spec for "Normalization, the Hypersphere, and von Mises–Fisher Geometry." Section outline,
theorem statements + prove/cite decisions, viz intent, notebook map, verified cross-site links,
references, and the realized canonical numbers the viz mirrors. The notebook
(`notebooks/hypersphere-vmf-geometry/`) is the source of truth; this brief records the design and the
numbers it produced.

## Placeholders

| Field | Value |
|---|---|
| title | Normalization, the Hypersphere, and von Mises–Fisher Geometry |
| subtitle | Why retrieval lives on the unit sphere, and the distribution that models a topical cluster on it |
| slug | `hypersphere-vmf-geometry` |
| domain | `embedding-geometry` |
| difficulty | `intermediate` |
| pipelineStage | `retrieve` |
| financeCaseStudy | `true` · modality `[text]` |
| prerequisites | `high-dimensional-geometry` (published) |
| reference notebooks | `notebooks/high-dimensional-geometry/`, `notebooks/bm25/` |

## Positioning

`high-dimensional-geometry` proved random unit vectors are near-orthogonal (Var⟨u,v⟩ = 1/d) and that
distance concentrates. This is the affirmative sequel: dense retrievers L2-normalize, so embeddings
live on S^{d-1}, cosine similarity *is* the sphere's geometry, and the von Mises–Fisher (vMF) law is
the maximum-entropy model of a topical cluster on it. The equatorial-concentration result is the
*same* 1/d, re-read as a coordinate marginal — the two topics interlock.

## Section outline (8 H2 + caveats + implementation)

1. **Overview & motivation** — retrieval operates after L2 normalization ⟹ working space is S^{d-1}; embed `<HypersphereLaboratory client:visible />`; "what we cover" list.
2. **Why normalize: cosine is the geometry of the sphere** (Prop 1).
3. **The uniform sphere and the equatorial band** (Thm 1; bridge to high-dimensional-geometry's 1/d).
4. **The von Mises–Fisher distribution** (Def 1, Thm 2, Prop 2).
5. **vMF as the maximum-entropy distribution on the sphere** (Thm 3).
6. **Estimating the cluster: MLE for μ and κ** (Thm 4).
7. **Why this matters for retrieval** — κ = measurable tightness; InfoNCE temperature τ ↔ 1/κ; the anisotropy / cone-effect / modality-gap break.
8. **Finance case study** — two synthetic vMF clusters at d=1536.
9. **Honest caveats** (`<RigorFlag>`) · **Implementation**.

## Theorems (statement + prove/cite)

- **Prop 1 — cosine ↔ distance.** `‖x−y‖²=2−2⟨x,y⟩` for unit vectors ⟹ cosine and (negated) distance give the same ranking. **PROVE** (one line) + corollary `cos=⟨x̂,ŷ⟩`.
- **Thm 1 — equatorial coordinate marginal.** `t=⟨u,v⟩` for uniform `v` has density `∝(1−t²)^((d−3)/2)`, mean 0, **Var=1/d**, `E[t⁴]=3/(d(d+2))`. **PROVE** Var=1/d by symmetry (reuses HDG Thm 2); **CITE** the density's Gamma-ratio normalizer (slice/Jacobian, Blum–Hopcroft–Kannan).
- **Def 1 — vMF.** `f=C_d(κ)e^{κμᵀx}`, `C_d(κ)=κ^{d/2−1}/[(2π)^{d/2}I_{d/2−1}(κ)]`.
- **Thm 2 — normalizer.** **PROVE** the surface-integral → 1-D `∫e^{κt}(1−t²)^((d−3)/2)dt` reduction via Thm 1's slice (the insight: C_d is Bessel *because* of the equator); **CITE** the Bessel integral representation (Mardia & Jupp). Verified numerically (density integrates to 1).
- **Prop 2 — mean resultant.** `E[x]=A_d(κ)μ`, `A_d=I_{d/2}/I_{d/2−1}`, monotone 0→1. **PROVE** via the exponential-family identity mean = ∇log-partition.
- **Thm 3 — max-entropy.** Among densities on S^{d-1} with fixed mean direction, vMF maximizes entropy. **PROVE** the Gibbs/KL≥0 direction `H(p)≤H(q)`; **CITE** multiplier existence (convex/exp-family theory). Verified numerically on the circle.
- **Thm 4 — MLE.** `μ̂=R/‖R‖` (**PROVE**, Cauchy–Schwarz), score equation `A_d(κ̂)=r̄` (**PROVE**, via Prop 2); Banerjee `κ̂≈r̄(d−r̄²)/(1−r̄²)` **STATE + CITE as an approximation**, verified vs. the brentq root (within 1.6% on the κ-grid).

## rigorFlag

vMF is a *model*; real embeddings violate its isotropy (anisotropy, cone effect, modality gap → κ̂
biased on raw embeddings — Ethayarajh 2019, Liang et al. 2022); the Banerjee κ̂ is an **approximation**
to the Bessel-ratio root (verified vs. the numerical root, not exact); uniform-sphere concentration is
asymptotic/in-probability; cosine is a calibrated *geometry*, not a calibrated *probability*. Proved
core (Prop 1, Thm 1's Var, Prop 2, Thm 3's inequality, Thm 4's μ̂/score) vs. cited (Gamma-ratio
normalizer, Bessel representation, max-entropy multiplier, κ̂ closed form).

## Viz — "Hypersphere Laboratory" (`src/components/viz/HypersphereLaboratory.tsx`)

Parallel to `ConcentrationLaboratory.tsx`: panel-toggle row, live KaTeX of the active law, sliders,
SVG `Plot` of closed-form log-densities, baked `EQUATOR`/`VMF`/`FINANCE` consts mirrored to the
decimal with a header comment naming the asserting tests. Reuse the closed-form-density / path /
readout idiom. Panels:
- **equator** (d-slider): coordinate-marginal log-density `((d−3)/2)·log(1−t²)`; readouts `Var(t)≈1/d`, band fraction. Numbers from `grid_table()["equator"]`.
- **vmf** (κ-slider, fixed d=100): tilted density `κt+((d−3)/2)log(1−t²)`; marker at ρ=A_d(κ); κ→κ̂ round-trip readout. Numbers from `grid_table()["vmf"]`.
- **distance** (static): the monotone line `‖x−y‖²=2−2cosθ` + "same ranking" badge (exact identity, client-side).
- **clusters** (fixed d=1536, bars): r̄, κ̂, intra/inter cosine for the two finance clusters. Numbers from `finance_demo()`.

## Notebook map (`hypersphere_vmf_geometry.py` + `01_…ipynb`)

Deps `numpy`, `scipy`. Bessel ratio `A_d(κ)` by the **downward continued fraction** (no Bessel eval —
stable at d=1536 where `I_ν` underflows); `C_d(κ)` via `scipy.special.ive`; exact κ̂ via `brentq`;
vMF sampler by **Wood (1994)** rejection. Harness asserts: `test_cosine_distance_identity`,
`test_coordinate_marginal`, `test_vmf_normalization`, `test_mean_resultant`, `test_max_entropy`,
`test_mle_recovery`, `test_finance_clusters`. Both artifacts exit 0 (~1.5 s; notebook ~20 s).

## Cross-site links (all slugs verified present on the adjacent sibling repos)

- `prerequisites`: `high-dimensional-geometry`.
- `connections`: `the-retrieval-problem`, `late-interaction-learned-sparse` (both existing MDX). Forward topics `infonce-contrastive-objective`/`optimized-product-quantization`/`chunking-as-segmentation` are graph edges + prose mentions (no MDX yet).
- `formalmlPrereqs`: `concentration-inequalities`, `shannon-entropy`.
- `formalmlConnections`: `representation-learning`.
- `formalstatisticsPrereqs`: `exponential-families`, `maximum-likelihood`, `multivariate-distributions`.
- `formalcalculusPrereqs`: `change-of-variables`, `surface-integrals`.
- *Not used (absent on siblings):* `maximum-entropy`, `exponential-families`@formalML, `spherical-coordinates`, `gamma-function`.

## References (Chicago N&B; every entry has a url/DOI)

Mardia & Jupp, *Directional Statistics* (2000); Fisher, "Dispersion on a Sphere" (1953); Banerjee,
Dhillon, Ghosh & Sra, "Clustering on the Unit Hypersphere using vMF Distributions," JMLR (2005); Sra,
"A short note on parameter approximation for vMF…" (2012); Wood, "Simulation of the von Mises Fisher
distribution" (1994); Wang & Isola, "Alignment and Uniformity on the Hypersphere," ICML (2020);
Ethayarajh, "How Contextual are Contextualized Word Representations?" EMNLP (2019); Liang et al.,
"Mind the Gap: the Modality Gap…," NeurIPS (2022).

## Realized canonical numbers (the viz mirrors these to the decimal)

**Equator** — `(d, Var(t) empirical, 1/d, band |t|<0.1)`:
```
2    0.49844  0.50000  0.06377      50   0.02004  0.02000  0.51494
3    0.33369  0.33333  0.10000     100   0.00991  0.01000  0.68025
5    0.20038  0.20000  0.14950     200   0.00496  0.00500  0.84218
10   0.09983  0.10000  0.23013     500   0.00200  0.00200  0.97480
20   0.04996  0.05000  0.33374     768   0.00130  0.00130  0.99449
                                  1536   0.00065  0.00065  0.99991
```
**vMF at d=100** — `(κ, ρ=A_d, κ̂ Banerjee, κ̂ exact)`:
```
0    0.00000     0.00     0.00      50   0.41507    50.06    50.00
1    0.01000     1.00     1.00     100   0.61957   100.17   100.00
2    0.01999     2.00     2.00     200   0.78221   200.30   200.00
5    0.04988     5.00     5.00     500   0.90580   500.41   500.00
10   0.09904    10.00    10.00    1000   0.95170  1000.45  1000.00
20   0.19271    20.01    20.00
```
**Finance (d=1536)** — tight `κ=900`: r̄=0.4613, κ̂=899.8, intra-cos=0.2118; loose `κ=300`: r̄=0.1905,
κ̂=303.6, intra-cos=0.0351; inter-cluster cos=−0.0028; mean directions −0.0267-aligned.
