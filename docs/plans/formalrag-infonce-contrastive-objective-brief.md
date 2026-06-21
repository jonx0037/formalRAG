# Brief — `infonce-contrastive-objective`

**Title:** Contrastive Learning for Retrieval: InfoNCE, Temperature, and Negative Sampling
**Domain:** neural-retrieval (the track ROOT) · **Difficulty:** advanced · **pipelineStage:** retrieve
**Prereqs:** `hypersphere-vmf-geometry`, `the-retrieval-problem` (both published)
**Why now:** the foundation of the neural-retrieval track; unblocks DPR → embedding-dim-lower-bounds →
late-interaction → PLAID. `matryoshka` already forward-references it.

## Context

The retrieval-problem defined relevance as a similarity functional and hypersphere-vMF told us dense
embeddings live on $S^{d-1}$ — neither said where those embeddings come from. They are *trained*.
InfoNCE is the loss that teaches a dual encoder to place a query near its relevant document and far
from everything else. This topic reads one loss three ways, two of which land squarely on already-built
machinery: the MI-lower-bound (CPC) reaches up into formalML `shannon-entropy`; the alignment/uniformity
decomposition (Wang–Isola) lands on the exact uniform sphere law of `hypersphere-vmf-geometry`
(uniformity = κ→0 vMF), with temperature acting as inverse concentration τ ~ 1/κ.

## Theorem spine (chosen: all three + temperature)

1. **MI lower bound (CPC, van den Oord 2018)** — `I(q;d⁺) ≥ log(N+1) − L_InfoNCE`. Full clean proof
   (optimal critic = density ratio). rigorFlag: the bound is **ceilinged at log(N+1)** → saturates /
   loose at small batch. Verified on a tractable Gaussian joint where `I = −½ln(1−ρ²)` is exact, using
   the **Bayes-optimal critic** (isolates the bound, not an encoder): bound ≤ truth ∀N, tightens
   monotonically, provably loose when `I_true > log(N+1)`.
2. **Alignment + uniformity (Wang–Isola 2020)** — asymptotic split into `L_align = E‖f(x)−f(y)‖²` and
   `L_unif = log E e^{−t‖f(x)−f(y)‖²}`; uniformity optimum = uniform sphere = κ→0 vMF (the prereq's
   exact law). rigorFlag: **asymptotic in M, not finite-sample**. Verified: GD on InfoNCE drives both
   alignment down and mean-resultant→0 (uniformity), landing near the direct align+unif optimum;
   uniform sphere minimizes uniformity vs clumped/high-κ configs.
3. **Temperature / hard-negative gradient** — `∂L/∂s_i = +p_i/τ` over negatives, so gradient mass
   concentrates on the **hardest** negative as τ→0. Full clean derivation. rigorFlag: gradient
   structure is exact; the **right τ is empirical** (uniformity–tolerance dilemma, Wang–Liu 2021).
   Verified: top-1 negative mass ↓ in τ, weight entropy ↑ in τ → log(N−1).

## Three pillars

- **Notebook** `notebooks/infonce-contrastive-objective/infonce_contrastive_objective.py` (+ `01_*.ipynb`).
  Imports prereqs: hypersphere `normalize, sample_uniform_sphere, sample_vmf, mean_resultant_length,
  mle_mu, kappa_hat_exact, _mean_pairwise_cosine`; retrieval-problem `cosine, rank`. One RNG stream for
  the MI Monte-Carlo. `viz_constants()` prints (cast) every baked number. Guards up front: τ, batch N,
  Σ-softmax via logsumexp, n≥2 denominators. <60 s, exits 0.
- **Viz** `src/components/viz/InfoNCEContrastiveLaboratory.tsx` — three pill panels (sphere + softmax
  gradient-weight bars driven by τ; MI bound vs log(N+1) ceiling; alignment/uniformity trade-off vs τ).
  Closed-form recomputed in TS: softmax weights, log(N+1), κ≈1/τ. Everything **measured** (MI bound,
  align/unif, finance scalars) is baked from `viz_constants()`. Sliders only.
- **MDX** `src/content/topics/infonce-contrastive-objective.mdx` — Overview+lab → setup/notation →
  Movement 1 (loss) → Movement 2 (MI bound) → Movement 3 (alignment/uniformity) → Movement 4
  (temperature gradient) → finance case study → rigorFlag → references. Plain `##` headings (LSH
  convention). Forward-reference DPR/negative-sampling/late-interaction **without links** (unbuilt).

## Finance case study (`financeCaseStudy: true`, modality text/pdf/audio)

Fine-tuning the production financial dual encoder with in-batch-negative InfoNCE. Hard negatives =
same-sector-different-company docs. Headline made executable: at small τ the same-sector hard negative
carries the largest gradient weight; larger batch raises the log(N+1) MI ceiling. Honest joints: false
negatives (mined positives mislabel relevant docs as negatives); anisotropy breaks the on-sphere model.

## Cross-site (formalML up-links)

- prereqs: `shannon-entropy` (MI/cross-entropy/KL), `maximum-likelihood` (loss = NLL of (N+1)-way softmax).
- connections: `representation-learning`; `formalstatistics:exponential-families` (softmax = categorical
  exp-family, τ = inverse natural-parameter scale). Verify slugs exist before linking.

## References (verify DOIs/arXiv at build)

CPC arXiv:1807.03748 · Wang–Isola arXiv:2005.10242 · DPR doi:10.18653/v1/2020.emnlp-main.550 · SimCLR
arXiv:2002.05709 · Wang–Liu arXiv:2012.09740 · Sohn N-pair (NeurIPS 2016) · Gutmann–Hyvärinen NCE
(AISTATS 2010) · sentence-transformers MultipleNegativesRankingLoss (documentation).

## Build order & verify

1. Notebook `.py` → RUN (`uv run --with numpy --with scipy python …`), exit 0; **build+run each headline
   before writing it** (MI saturation gap, align≈InfoNCE convergence scale, finance small-τ direction —
   pin assertions to observed). 2. `.ipynb` from nbformat generator → `jupyter execute` exit 0.
   3. Viz `.tsx` baking `viz_constants()`. 4. MDX. 5. `curriculum.ts`/`curriculum-graph.json` (flip
   `infonce-contrastive-objective` status; drop title from neural-retrieval `planned[]`). 6. Gates:
   `pnpm exec astro sync`, `pnpm validate`, `pnpm build`, browser-verify (`.katex-error`=0, lab mounts,
   numbers match notebook). 7. Flip `status: published`, branch → PR off `main`.

## Adversarial review (ultracode)

After the build verifies, run a multi-agent review (math correctness, viz↔python invariant, gemini-style
guard/denominator nits) before opening the PR — the BM25 5-agent review precedent that caught a latent
scaffold math error.
