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
    planned: [
      'The Retrieval Problem: Relevance, Similarity, and the Geometry of Scores',
      'MIPS Hardness and the Limits of Exact Nearest-Neighbor Search',
    ],
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
    planned: [
      'The Vector Space Model and TF-IDF',
      'The Inverted Index and Safe Dynamic Pruning (WAND, BlockMax-WAND)',
      'The Probability Ranking Principle',
      'Query-Likelihood Language Models and Smoothing',
      'Relevance Feedback and Query Expansion: Rocchio and RM3',
    ],
    dependsOnFormalML: false,
  },
  {
    domain: 'vector-quantization',
    label: 'Vector Quantization',
    layer: 'Retrieval Mechanics',
    description:
      'Lossy compression of embedding vectors for memory-bounded search — rate-distortion and estimation theory: Lloyd–Max optimality, product quantization, and score-aware anisotropic quantization.',
    planned: [
      'Vector Quantization and the Lloyd–Max Optimality Conditions',
      'Product Quantization and Asymmetric Distance Computation',
      'Optimized PQ and Score-Aware Quantization (OPQ, ScaNN)',
    ],
    dependsOnFormalML: true,
  },
  {
    domain: 'ann-indexing',
    label: 'ANN Index Structures',
    layer: 'Retrieval Mechanics',
    description:
      'The data structures behind sublinear vector retrieval, at the level of their actual mathematics: IVF Voronoi partitioning, LSH sensitivity theory, navigable small-world graphs and HNSW, multi-vector and filtered ANN.',
    planned: [
      'Voronoi Partitioning and the IVF Index',
      'Locality-Sensitive Hashing: Collision Probability and the ρ Exponent',
      'Navigable Small-World Graphs and the Mathematics of Greedy Routing',
      'HNSW: Hierarchical Navigable Small-World Construction and Search',
      'Filtered and Incremental ANN: Predicate Search, Deletion, and Graph Connectivity',
      'Multi-Vector ANN: Indexing and Pruning MaxSim at Scale (PLAID)',
    ],
    dependsOnFormalML: false,
  },
  {
    domain: 'neural-retrieval',
    label: 'Neural & Learned Retrieval',
    layer: 'Learned Retrieval & Ranking',
    description:
      'Learned representations for retrieval, defined by training objectives and expressivity claims: InfoNCE contrastive training, dense dual encoders, late interaction and learned sparse, cross-encoders, distillation, and cross-modal alignment.',
    planned: [
      'Contrastive Learning for Retrieval: InfoNCE, Temperature, and Negative Sampling',
      'Hard-Negative Mining and Debiased Contrastive Training (ANCE)',
      'Dense Retrieval and Dual Encoders (DPR)',
      'How Many Dimensions Does Relevance Need? Sign-Rank and Margin Complexity',
      'Late Interaction and Learned Sparse Retrieval: ColBERT and SPLADE',
      'Cross-Encoders and the Reranking Cascade',
      'Knowledge Distillation for Retrieval: Teacher–Student Transfer (MarginMSE)',
      'Cross-Modal Contrastive Alignment and the Modality Gap',
    ],
    dependsOnFormalML: true,
  },
  {
    domain: 'ranking-fusion',
    label: 'Ranking, Fusion & Reranking',
    layer: 'Learned Retrieval & Ranking',
    description:
      'The mathematics of producing, combining, and reordering ranked lists: learning-to-rank, reciprocal rank fusion and its social-choice grounding, cross-encoder cascades, and LLM listwise rerankers.',
    planned: [
      'Learning to Rank: Pointwise, Pairwise, and RankNet',
      'LambdaRank, LambdaMART, and Listwise Objectives',
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
    planned: [
      'Set Metrics: Precision, Recall, MAP, and MRR as Estimators',
      'NDCG: Graded Relevance and Discount Geometry',
      'Score Calibration, Drift Detection, and Significance Testing for Retrieval',
      'LLM-as-Judge and Faithfulness: RAGAS as a Family of Estimators',
      'Conformal Factuality: Distribution-Free Correctness Guarantees for Generation',
    ],
    dependsOnFormalML: true,
  },
  {
    domain: 'generation-grounding',
    label: 'Generation & Grounding',
    layer: 'Generation & Reasoning',
    description:
      'The mathematics of what happens once context is retrieved: the retrieval-vs-long-context tradeoff, query transformation as distribution-shift correction, faithfulness as a measurable quantity, and selective generation.',
    planned: [
      'Retrieval versus Long Context: Attention Complexity and Positional Bias',
      'Query Transformation and HyDE: Correcting Distribution Shift in Embedding Space',
      'Faithfulness and Groundedness as Measurable Quantities',
      'Selective Generation: When a RAG System Should Abstain',
    ],
    dependsOnFormalML: true,
  },
  {
    domain: 'rag-information-theory',
    label: 'Information Theory of RAG',
    layer: 'Generation & Reasoning',
    description:
      'The "why retrieval works" layer: mutual information between query, context, and answer; the retriever as a noisy channel; submodular and DPP context selection; multi-hop retrieval; GraphRAG; and the multimodal financial capstone.',
    planned: [
      'Pointwise Mutual Information: What Retrieval Adds to Generation, in Bits',
      'The Retriever as a Noisy Channel: Recall, Precision, and Information Limits',
      'Context Selection: Submodular Coverage, MMR, and Determinantal Point Processes',
      'Multi-Hop and Iterative Retrieval as Search over an Evidence Space',
      'GraphRAG: Community Detection and the Modularity of Knowledge',
      'Capstone: Architecture and Mathematics of a Production Multimodal Financial RAG System',
    ],
    dependsOnFormalML: true,
  },
] as const;

export type Layer = (typeof tracks)[number]['layer'];

export type Domain = (typeof tracks)[number]['domain'];

export const domainLabelMap: Record<string, string> = Object.fromEntries(
  tracks.map((t) => [t.domain, t.label]),
);
