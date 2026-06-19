# Brief — `pca-dimensionality-reduction`

Phase-A spec for "PCA as Optimal Linear Dimensionality Reduction for Embeddings." Section outline,
theorem statements + prove/cite, viz intent, notebook map, verified cross-site links, references, and the
realized canonical numbers the viz mirrors. The notebook (`notebooks/pca-dimensionality-reduction/`) is the
source of truth.

## Placeholders

| Field | Value |
|---|---|
| title | PCA as Optimal Linear Dimensionality Reduction for Embeddings |
| subtitle | The variance-optimal projection of an embedding cloud — what truncating 1536 dimensions to k keeps, and the honest reason it can still hurt retrieval |
| slug | `pca-dimensionality-reduction` |
| domain | `embedding-geometry` · pipelineStage `index` |
| difficulty | `intermediate` · financeCaseStudy `true` · modality `[text]` |
| prerequisites | `high-dimensional-geometry` (published) |
| reference notebooks | `notebooks/high-dimensional-geometry/`, `notebooks/hypersphere-vmf-geometry/` |

## Positioning

PCA answers the question `high-dimensional-geometry` raises: real embeddings have low *effective* rank, so
how much retrieval quality survives projecting 1536→k? PCA is the variance-optimal linear answer; the honest
twist — variance-optimal ≠ retrieval-optimal — is the differentiator. Links up into formalML's published
`spectral-theorem → svd → pca-low-rank` chain; `svd.mdx` is the proof exemplar for Eckart–Young.

## Section outline (10 H2 + caveats + implementation)
1. Overview & motivation (ANN cost ~ d; low effective rank; 1536→k question; embed `<SpectrumLaboratory/>`).
2. PCA three equivalent ways (maximize variance ⟺ minimize reconstruction error ⟺ decorrelate).
3. Principal directions = top covariance eigenvectors (Thm 1).
4. Reconstruction error & the variance ⟺ error equivalence (Thm 2).
5. Eckart–Young–Mirsky and the SVD↔PCA equivalence (Thm 3, Prop 2; EVR + effective rank).
6. What survives projection (Thm 4; PCA data-dependent vs JL data-oblivious).
7. Finance case study.
8. The honest retrieval caveat (variance- vs retrieval-optimal; All-but-the-Top).
9. Honest caveats (`<RigorFlag>`). 10. Implementation.

## Theorems (statement + prove/cite)
- **Def 1** centered covariance `Σ=X̃ᵀX̃/(n−1)`, orthonormal projector `P=WWᵀ`. **Def 2** `EVR(k)`, effective rank `n_eff=(Σλ)²/Σλ²`.
- **Thm 1 — first PC = top eigenvector (Rayleigh quotient).** **PROVE** the first direction via the spectral substitution `w=Qc, ‖c‖=1 ⟹ wᵀΣw=Σλᵢcᵢ²≤λ₁` (no Lagrange machinery; routes to formalCalculus `eigenvalues-eigenvectors` / formalML `spectral-theorem`); prove one-step deflation inline; **CITE** general Courant–Fischer.
- **Thm 2 — variance-max ⟺ reconstruction-min.** `E‖z−WWᵀz‖²=tr(Σ)−tr(WᵀΣW)`; residual `Σ_{i>k}λᵢ`. **PROVE in full** (Pythagoras + iterate Thm 1 for the ceiling).
- **Thm 3 — Eckart–Young–Mirsky (Frobenius).** `‖X̃−X̃_k‖_F²=Σ_{i>k}sᵢ²=(n−1)Σ_{i>k}λᵢ`. **PROVE** the truncated-SVD value + Frobenius optimality (mirror `svd.mdx`); **CITE Mirsky** for all unitarily-invariant norms; verify numerically (beats random rank-k).
- **Prop 2 — SVD↔PCA.** `λᵢ=sᵢ²/(n−1)`, PCs = right singular vectors. **State + one-line derive + CITE** formalML `svd`/`pca-low-rank`; verify via eigh-vs-SVD assert.
- **Thm 4 — projection distortion = EVR.** retained squared-norm & mean pairwise squared-distance fraction `= EVR(k)`. **PROVE in full** (trace identities; page-original tie to retrieval geometry).

## rigorFlag (5 items)
variance-optimal ≠ retrieval-optimal + top-PC nuisance & **All-but-the-Top** (Mu & Viswanath 2018, cited
empirical, never a theorem); linear-on-nonlinear (effective rank upper-bounds intrinsic dimension);
centering/cosine interaction; proved core (Thms 1/2/4 + Thm 3 value) vs cited (SVD existence, Mirsky,
Courant–Fischer, All-but-the-Top); synthetic finance embeddings.

## Viz — "Spectrum Laboratory" (`SpectrumLaboratory.tsx`)
Parallel to the existing labs; **predominantly baked polylines** (the spectrum *is* the data — no
closed-form curve; only the KaTeX band is closed-form). Rank-k slider over `K_GRID`. Panels:
(i) scree + cumulative EVR; (ii) reconstruction error vs k (PCA tail-sum vs faint random reference);
(iii) top-2-PC scatter of the 3 clusters; (iv) recall@10 retained vs kept dim, PCA vs random.

## Notebook map (`pca_dimensionality_reduction.py` + `01_…ipynb`)
Deps `numpy`, `scipy`, **`scikit-learn`** (cross-check). `structured_embeddings` (decaying-spectrum low-rank
+ noise + topical clusters); `pca_via_covariance` (eigh) and `pca_via_svd` (sign-canonicalized agreement);
Eckart–Young + random-projection comparison; EVR + effective rank; `projection_distortion`; recall@k PCA-vs-
random; `grid_table()` → {scree, error, recall, scatter}; `finance_demo()` at d=1536/kept=128. 9 asserts:
`test_pca_eig_svd_agree`, `test_rayleigh_first_pc`, `test_variance_reconstruction_equivalence`,
`test_eckart_young`, `test_explained_variance_and_effective_rank`, `test_projection_distortion`,
`test_recall_after_projection`, `test_sklearn_crosscheck`, `test_finance_spectrum`. Both artifacts exit 0
(~5 s / ~25 s). Traps: center first; `eigh` not `eig`; eigenvector sign canonicalization; assert only
order-invariant quantities past the signal rank.

## Cross-site links (all slugs verified present)
- `prerequisites`: `high-dimensional-geometry`.
- `connections`: `vector-space-model-tfidf` (LSA = truncated SVD of the term–document matrix), `hypersphere-vmf-geometry` (PCA centering vs cosine's uncentered angles).
- `formalmlPrereqs`: `pca-low-rank`, `svd`, `spectral-theorem`.
- `formalstatisticsPrereqs`: `multivariate-distributions`.
- `formalcalculusPrereqs`: `eigenvalues-eigenvectors`, `convex-optimization`.

## References (Chicago N&B; url/DOI each)
Pearson 1901; Hotelling 1933; Eckart & Young 1936; Jolliffe *PCA* 2002; Mu & Viswanath 2018 (All-but-the-Top);
Deerwester et al. 1990 (LSA); Bingham & Mannila 2001 (random projection); Halko–Martinsson–Tropp 2011
(randomized SVD); Horn & Johnson 2013 (Courant–Fischer, Mirsky).

## Realized canonical numbers (viz mirrors to the decimal) — finance cloud, d=1536, intrinsic 48, 3 clusters
- effective rank **6.10**, total variance 34.414; top eigenvalues 10.895, 8.236, 1.001, 0.919, 0.847, 0.780 …
- `K_GRID = [2,4,8,16,32,64,128,256,512,768]`
- EVR@k: `0.5559, 0.6117, 0.7015, 0.8235, 0.9406, 0.9842, 0.9869, 0.9910, 0.9962, 0.9988`
- PCA recon err²: `18324, 16022, 12315, 7282, 2449, 652, 543, 373, 158, 50`; random: `41224 … 20529`
- recall@10 PCA: `0.286, 0.240, 0.299, 0.641, 0.852, 0.941, 0.944, 0.950, 0.965, 0.983`; random: `0.007 … 0.655`
- Finance headline (kept=128): EVR **98.69%**, recall@10 **PCA 94.4%** vs **random 25.7%**.
