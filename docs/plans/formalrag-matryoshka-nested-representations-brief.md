# Brief — `matryoshka-nested-representations`

Phase-A spec for "Matryoshka Representations: Jointly Trained Nested Subspaces." Section outline,
theorem statements + prove/cite, viz intent, notebook map, verified cross-site links, references, and the
realized canonical numbers the viz mirrors. The notebook (`notebooks/matryoshka-nested-representations/`)
is the source of truth.

## Placeholders

| Field | Value |
|---|---|
| title | Matryoshka Representations: Jointly Trained Nested Subspaces |
| subtitle | One embedding whose every prefix is itself a usable representation — and the exact sense in which the linear version is just PCA |
| slug | `matryoshka-nested-representations` |
| domain | `embedding-geometry` · pipelineStage `index` |
| difficulty | `intermediate` · financeCaseStudy `true` · modality `[text]` |
| prerequisites | `pca-dimensionality-reduction` (published) |
| reference notebooks | `notebooks/pca-dimensionality-reduction/`, `notebooks/johnson-lindenstrauss/` |

## Positioning

PCA gives nested subspaces optimal for *reconstruction*; random projection (the sibling JL topic) gives a
fixed-width sketch. Matryoshka Representation Learning (Kusupati et al. 2022) trains a single embedding so
that *every prefix* $z_{1:m}$ is itself a good representation — store one $1536$-d vector, retrieve with its
first $96$ dims when speed matters and its full width when accuracy does. The geometric heart, and the clean
tie to the prerequisite: **in the linear, squared-reconstruction setting the jointly optimal nested basis is
exactly PCA's eigenvalue-ordered basis** — MRL generalizes PCA's nested-subspace optimality from "variance"
to an arbitrary task loss. The honest twist is that the *value* of MRL is in the nonlinear/contrastive case,
where the nesting must be *trained* (it is not free), and the headline guarantees there are empirical, not
theorems. Links up into formalML `pca-low-rank`/`svd` (the Eckart-Young nestedness) and `representation-learning`.

## Section outline
1. Overview & motivation (one vector, many widths; embed `<MatryoshkaLaboratory/>`).
2. The Matryoshka objective: a weighted sum of per-granularity losses.
3. **Linear MRL is PCA** (Thm 1): nestedness + Eckart-Young ⇒ one ordered basis is jointly optimal.
4. Weight invariance (Prop 1): PCA is the joint optimum for *any* positive granularity weights.
5. Nesting must be trained: prefix recall, and the rotation that preserves distances but destroys prefixes.
6. Adaptive (funnel) retrieval: shortlist on a short prefix, rerank at full width.
7. The nonlinear/contrastive case (self-contained; the empirical regularity, honestly flagged).
8. Finance case study. 9. Honest caveats (`<RigorFlag>`). 10. Implementation.

## Theorems (statement + prove/cite)
- **Def 1** nested basis $V=[v_1,\dots,v_d]$, prefix projector $P_m=V_{:m}V_{:m}^\top$; the linear-MRL loss $\sum_m c_m\lVert \tilde X-\tilde X P_m\rVert_F^2$.
- **Thm 1 — linear MRL = PCA.** For squared-reconstruction loss, the PCA eigenvalue-ordered basis attains the Eckart-Young rank-$m$ optimum at *every* $m$ simultaneously (its top-$m$ subspaces are nested), so it minimizes each prefix term and hence the weighted sum for any $c_m\ge 0$. **PROVE in full**: cite Eckart-Young (the per-$m$ optimum, from the [PCA topic](/topics/pca-dimensionality-reduction)); the page-original step is *nestedness* — top-$m$ PCA $\subset$ top-$(m{+}1)$ PCA — so one ordered basis is simultaneously optimal across granularities, which a non-nested basis cannot be. Harness asserts `prefix_recon_error(PCA) == eckart_young_rankm` to the decimal at every $m$.
- **Prop 1 — weight invariance.** Because PCA attains each per-$m$ optimum, the joint loss equals the (unimprovable) weighted sum of the per-$m$ optima for *every* positive weighting, so the argmin basis is PCA independent of $c_m$. **PROVE** (immediate from Thm 1); verify over random weight vectors.
- **Empirical (cited, not theorems):** jointly trained nested prefixes beat independently trained heads; the nonlinear MRL matches full-width accuracy at a fraction of the dimensions (Kusupati et al. 2022). Presented as measured regularities.

## rigorFlag (load-bearing)
The clean theorem — Matryoshka = PCA — holds *only* in the linear, squared-reconstruction special case; there it is exact (nestedness + Eckart-Young), and it is also where MRL reduces to something we already had. The *interesting* MRL is the nonlinear/contrastive one, and its headline claims are **empirical**: there is no general proof that jointly trained prefixes dominate independently trained heads, the granularity weights $c_m$ are an untuned heuristic, and the "nested = ordered importance" reading is exact only in the linear case. Prefix retrieval *quality* is monotone in $m$ for a nested embedding, but "nested" is a property that must be **trained** — a random rotation of the same embedding preserves every full-width distance yet retrieves no better than chance at small prefixes (prefix-96 recall $\sim 94\%$ nested vs $\sim 12\%$ rotated on the finance cloud). The funnel-retrieval recall is *measured*, not tightly bounded. The finance embeddings are a synthetic low-rank-plus-noise stand-in (PCA scores), not a trained MRL encoder; on a real encoder the nesting comes from the contrastive objective, which this linear model only mimics.

## Viz — "Matryoshka Laboratory" (`MatryoshkaLaboratory.tsx`)
Granularity slider over `GRANULARITIES = [24,48,96,192,384,768,1536]`; all baked from `grid_table()`. Panels:
(i) **prefix recall** vs $m$: nested (accent, graceful) vs rotated (faint, collapses at small $m$), meeting at full width — the "nesting must be trained" headline;
(ii) **nestedness**: reconstruction error vs $m$ — nested and the Eckart-Young optimum coincide (one ordered basis = every rank-$m$ optimum) far below a random basis;
(iii) **adaptive funnel**: recall/cost Pareto sweeping the shortlist size at a fixed short prefix — near-full recall at a fraction of exhaustive scoring cost.

## Notebook map (`matryoshka_nested_representations.py` + `01_…ipynb`)
Deps `numpy`, `scipy`, **`scikit-learn`** (cross-check). `structured_embeddings` (same generator as PCA/JL); `pca_basis`/`pca_embedding` (the linear-MRL optimum); `random_rotation` (distance-preserving, nesting-destroying foil); `prefix_recon_error` vs `eckart_young_rankm` (nestedness); `joint_mrl_loss`; `prefix_recall`; `funnel_retrieval`. `grid_table()` → {recall, recon, funnel}; `finance_demo()` at $d=1536$, $n=2000$ (n>d so the nested basis spans all 1536 dims). 7 asserts: `test_linear_mrl_equals_pca`, `test_weight_invariance`, `test_prefix_recall_monotone`, `test_nested_beats_rotated`, `test_funnel_retrieval`, `test_sklearn_crosscheck`, `test_finance_funnel`. Both artifacts exit 0 (~5 s). Traps: $n>d$ for a full-width basis; rerank set is small so `argsort` not `argpartition`; assert reconstruction (sign-invariant), not raw components.

## Cross-site links (slugs verified present on the siblings)
- `prerequisites`: `pca-dimensionality-reduction`.
- `connections`: `pca-dimensionality-reduction` (MRL generalizes its nested-subspace optimality from variance to a task loss).
- `formalmlPrereqs`: `pca-low-rank` (✅ — nested subspaces / Eckart-Young), `svd` (✅ — truncated-SVD optimality).
- `formalmlConnections`: `representation-learning` (✅ — MRL as a learned-representation technique).
- `formalstatisticsPrereqs`: `multivariate-distributions` (✅ — the covariance the linear MRL diagonalizes).
- `formalcalculusPrereqs`: `eigenvalues-eigenvectors` (✅ — the ordered eigenbasis the nesting rests on).
- In-site forward references in prose only: `johnson-lindenstrauss` (sibling; random vs learned nesting — link resolves once its PR merges) and `infonce-contrastive-objective` (the planned contrastive objective the nonlinear case uses).

## References (Chicago N&B; url/DOI each)
Kusupati et al. 2022 (Matryoshka Representation Learning); Eckart & Young 1936 (low-rank optimality); Horn & Johnson 2013 (Eckart-Young-Mirsky / nestedness); van den Oord, Li & Vinyals 2018 (InfoNCE / CPC — the contrastive prefix loss); Jolliffe 2002 (PCA, nested subspaces); OpenAI 2024 (text-embedding-3 dimension shortening — MRL in production, documentation); Nussbaum et al. 2024 (Nomic Embed — open MRL embeddings).

## Realized canonical numbers (viz mirrors to the decimal) — finance cloud, d=1536, n=2000, intrinsic 48, 3 clusters
- `GRANULARITIES = [24, 48, 96, 192, 384, 768, 1536]`
- nested recall@10: `0.7093, 0.9353, 0.9367, 0.9453, 0.9440, 0.9653, 1.0000`
- rotated recall@10: `0.0427, 0.0500, 0.1200, 0.2313, 0.4253, 0.6740, 1.0000`
- nested = optimum reconstruction error: `7274.2, 1159.7, 1041.1, 842.5, 541.7, 187.4, 0.0` (coincide to the decimal)
- random-basis reconstruction error: `58870.3, 57883.8, 56004.0, 52306.0, 44539.4, 29860.7, 0.0`
- funnel Pareto (prefix 48, shortlist → recall @ cost): `10→0.9353@3.6%, 15→0.9987@3.9%, 25→1.000@4.4%, 50→1.000@5.6%, 100→1.000@8.1%, 200→1.000@13.1%`
- **Finance headline** (prefix 96): nested recall **93.7%** vs rotated **12.0%**; funnel (shortlist on 96-dim prefix, rerank at full 1536) recall **100%** at **8.8%** of exhaustive cost.
