// Each track declares which curriculum *layer* it belongs to. The /paths page
// reads this to inject layer-section headers above the first track of each
// layer — so adding/removing/reordering tracks here is the single source of
// truth, and the presentation layer derives layer transitions from data.
//
// `planned` lists topic titles on the roadmap that are not yet authored as
// published MDX. As a topic is published, remove it from `planned` here.
// `dependsOnFormalML` surfaces a "need the ML foundations?" pointer on /paths.
export const tracks = [
  {
    domain: 'retrieval-foundations',
    label: 'Retrieval Foundations',
    layer: 'Foundations',
    description:
      'Retrieval as ranking by a relevance functional; the metric and inner-product structure of similarity; and the complexity-theoretic limits of exact search. The root of the dependency graph.',
    planned: [],
    dependsOnFormalML: false,
  },
  {
    domain: 'embedding-geometry',
    label: 'Embedding-Space Geometry',
    layer: 'Foundations',
    description:
      'Where embeddings live and what ANN must contend with: concentration of measure, hypersphere and von Mises–Fisher geometry, PCA and random projections, Johnson–Lindenstrauss, and chunking as segmentation.',
    planned: [],
    dependsOnFormalML: true,
  },
  {
    domain: 'probabilistic-ir',
    label: 'Probabilistic IR',
    layer: 'Retrieval Mechanics',
    description:
      'The classical algebraic and probabilistic retrieval models that form the lexical half of hybrid retrieval: the vector space model, the Probability Ranking Principle, BM25, query-likelihood models, and the inverted index.',
    planned: [],
    dependsOnFormalML: false,
  },
  {
    domain: 'vector-quantization',
    label: 'Vector Quantization',
    layer: 'Retrieval Mechanics',
    description:
      'Lossy compression of embedding vectors for memory-bounded search — rate-distortion and estimation theory: Lloyd–Max optimality, product quantization, and score-aware anisotropic quantization.',
    planned: [],
    dependsOnFormalML: true,
  },
  {
    domain: 'ann-indexing',
    label: 'ANN Index Structures',
    layer: 'Retrieval Mechanics',
    description:
      'The data structures behind sublinear vector retrieval, at the level of their actual mathematics: IVF Voronoi partitioning, LSH sensitivity theory, navigable small-world graphs and HNSW, multi-vector and filtered ANN.',
    planned: [],
    dependsOnFormalML: false,
  },
  {
    domain: 'neural-retrieval',
    label: 'Neural & Learned Retrieval',
    layer: 'Learned Retrieval & Ranking',
    description:
      'Learned representations for retrieval, defined by training objectives and expressivity claims: InfoNCE contrastive training, dense dual encoders, late interaction and learned sparse, cross-encoders, distillation, and cross-modal alignment.',
    planned: [],
    dependsOnFormalML: true,
  },
  {
    domain: 'ranking-fusion',
    label: 'Ranking, Fusion & Reranking',
    layer: 'Learned Retrieval & Ranking',
    description:
      'The mathematics of producing, combining, and reordering ranked lists: learning-to-rank, reciprocal rank fusion and its social-choice grounding, cross-encoder cascades, and LLM listwise rerankers.',
    planned: [
      'LLM Rerankers: Listwise Permutation Objectives and RankGPT',
    ],
    dependsOnFormalML: false,
  },
  {
    domain: 'retrieval-evaluation',
    label: 'Retrieval & RAG Evaluation',
    layer: 'Learned Retrieval & Ranking',
    description:
      'Evaluation treated as statistics: ranking metrics as estimators, significance testing, calibration, drift detection, LLM-as-judge reliability, and distribution-free conformal factuality guarantees.',
    planned: [],
    dependsOnFormalML: true,
  },
  {
    domain: 'generation-grounding',
    label: 'Generation & Grounding',
    layer: 'Generation & Reasoning',
    description:
      'The mathematics of what happens once context is retrieved: the retrieval-vs-long-context tradeoff, query transformation as distribution-shift correction, faithfulness as a measurable quantity, and selective generation.',
    planned: [],
    dependsOnFormalML: true,
  },
  {
    domain: 'rag-information-theory',
    label: 'Information Theory of RAG',
    layer: 'Generation & Reasoning',
    description:
      'The "why retrieval works" layer: mutual information between query, context, and answer; the retriever as a noisy channel; submodular and DPP context selection; multi-hop retrieval; GraphRAG; and the multimodal financial capstone.',
    planned: [],
    dependsOnFormalML: true,
  },
] as const;

export type Layer = (typeof tracks)[number]['layer'];

export type Domain = (typeof tracks)[number]['domain'];

export const domainLabelMap: Record<string, string> = Object.fromEntries(
  tracks.map((t) => [t.domain, t.label]),
);
