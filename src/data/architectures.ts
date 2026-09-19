import type { StageId } from './pipeline';

// The named RAG architectures, mapped onto the mathematics that constitutes them.
//
// The popular cheat sheets present these as a flat list of parallel systems.
// They are not. They vary along three largely independent axes:
//
//   control flow   — how many times retrieval fires, and what decides
//   index structure — what the retrieval unit is (chunk, entity graph, summary)
//   representation  — what a document is in the index (one vector, many, an image)
//
// Flattening those axes into one list is what produces the eight-, twelve-, or
// twenty-item taxonomies in circulation, and why the same system appears under
// several names. `reducesTo` records where two names denote one mechanism, and
// `kind` separates genuine architectures from operators that bolt onto any of
// them and from the contrast class that defines the boundary of retrieval.
//
// EDITORIAL RULE, load-bearing: an entry that cannot state a mechanism, a
// condition under which it wins, a condition under which it loses, and at least
// two published topics that constitute it does not belong on this page. The
// failure mode is what separates this from a listicle, and every one below is
// either proved or measured in the topic it links to.

export type ArchitectureKind = 'architecture' | 'operator' | 'contrast';

// 'shipped'  — one published topic IS this architecture
// 'composed' — assembled from several published topics, no single keystone
// 'gap'      — identified by this analysis, not yet built
export type ArchitectureStatus = 'shipped' | 'composed' | 'gap';

// 'linear' — retrieval fires once, stages run in order
// 'cycle'  — a stage can send control backwards; retrieval count is data-dependent
// 'branch' — a decision selects among alternative paths before committing effort
export type ArchitectureFlow = 'linear' | 'cycle' | 'branch';

export interface Architecture {
  id: string;
  name: string;
  industryAliases: readonly string[];
  kind: ArchitectureKind;
  status: ArchitectureStatus;
  flow: ArchitectureFlow;
  /** Ordered walk through the pipeline band. Stages may repeat — that is a cycle. */
  stages: readonly StageId[];
  /** One sentence: the mathematics that distinguishes this from the baseline. */
  mechanism: string;
  /** The corpus or query property under which it wins. */
  winCondition: string;
  /** Where it loses, and why — stated as specifically as the win. */
  failureMode: string;
  /** Published topic slugs that constitute it. Minimum two. */
  composedOf: readonly string[];
  /** The single topic that most nearly IS this architecture, if one exists. */
  keystone: string | null;
  /** Set when this name denotes the same mechanism as another entry. */
  reducesTo?: string;
}

export const architectures: readonly Architecture[] = [
  {
    id: 'naive',
    name: 'Naive RAG',
    industryAliases: ['Vanilla RAG', 'single-pass RAG', 'standard RAG'],
    kind: 'architecture',
    status: 'composed',
    flow: 'linear',
    stages: ['ingest', 'index', 'retrieve', 'generate'],
    mechanism:
      'Retrieval is a fixed preprocessing step that fires exactly once: the context is the top-k of a single similarity functional, chosen before any generation begins.',
    winCondition:
      'The answer lives in one passage the query names more or less directly — a question phrased in the corpus’s own vocabulary, with a single local answer.',
    failureMode:
      'Two failures, and they are different. A query phrased unlike its source documents lands off the document manifold, so the nearest neighbors are the wrong ones. And an answer that requires composing evidence no single passage carries is unreachable at any k, because more context does not manufacture a relation the corpus never stated.',
    composedOf: [
      'the-retrieval-problem',
      'dense-retrieval-dual-encoders',
      'chunking-as-segmentation',
      'retrieval-vs-long-context',
    ],
    keystone: 'dense-retrieval-dual-encoders',
  },
  {
    id: 'hybrid',
    name: 'Hybrid RAG',
    industryAliases: ['hybrid search', 'sparse + dense', 'Advanced RAG'],
    kind: 'architecture',
    status: 'composed',
    flow: 'linear',
    stages: ['ingest', 'index', 'retrieve', 'fuse', 'generate'],
    mechanism:
      'Two or more retrieval legs with different failure modes run independently and are combined on ranks rather than scores, which makes the combination invariant to each leg’s scale.',
    winCondition:
      'The legs are partial views that miss different documents. Fusion gain grows as the legs de-correlate, because what one leg misses another has already ranked.',
    failureMode:
      'When the legs form a quality ladder — several monotone approximations of one underlying score — the best leg dominates and fusion only adds noise, so the gain is negative. "Fused beats best leg" is not a theorem. Even the dominated-leg flip needs a false positive endorsed by both legs: under the usual constant, one top vote is worth less than two mediocre ones.',
    composedOf: [
      'bm25-binary-independence-model',
      'rank-fusion-rrf',
      'dense-retrieval-dual-encoders',
      'capstone-multimodal-financial-rag',
    ],
    keystone: 'rank-fusion-rrf',
  },
  {
    id: 'multimodal',
    name: 'Multimodal RAG',
    industryAliases: ['cross-modal RAG', 'vision RAG'],
    kind: 'architecture',
    status: 'composed',
    flow: 'linear',
    stages: ['ingest', 'index', 'retrieve', 'fuse', 'generate'],
    reducesTo: 'hybrid',
    mechanism:
      'Hybrid’s fusion mechanism applied over heterogeneous encoders: each modality contributes its own leg, and the legs meet either in one shared embedding space or at the rank level.',
    winCondition:
      'Evidence is genuinely distributed across modalities — a figure in a table, a qualification in a transcript, a trend only a chart shows.',
    failureMode:
      'The modality gap. Separately trained encoders place text and images in disjoint cones, so a cross-modal cosine is dominated by the gap direction rather than by relevance. The gap is invisible to inner-product ranking, which is exactly why it goes unnoticed until the scores are used for anything but sorting.',
    composedOf: [
      'cross-modal-alignment',
      'capstone-multimodal-financial-rag',
      'rank-fusion-rrf',
    ],
    keystone: 'cross-modal-alignment',
  },
  {
    id: 'hyde',
    name: 'HyDE',
    industryAliases: ['Hypothetical Document Embeddings', 'query transformation', 'query rewriting'],
    kind: 'architecture',
    status: 'shipped',
    flow: 'linear',
    stages: ['generate', 'retrieve', 'generate'],
    mechanism:
      'Draft a hypothetical answer document, embed that instead of the query, and retrieve with it — moving the probe off query space and onto the document manifold. It is the only common architecture in which generation runs upstream of retrieval.',
    winCondition:
      'Query–document distribution shift: questions phrased unlike the corpus they must match. The correction is independent of how far off-manifold the query started, because the bare query’s position is discarded rather than adjusted.',
    failureMode:
      'It spends a generation call before any retrieval, and buys nothing when the query was already on-manifold — so it falls off the cost frontier first. A generator that drafts the wrong entity at some rate imposes a recall ceiling that averaging more drafts cannot break, because the error is a bias, not variance.',
    composedOf: [
      'query-transformation-hyde',
      'pseudo-relevance-feedback',
      'dense-retrieval-dual-encoders',
    ],
    keystone: 'query-transformation-hyde',
  },
  {
    id: 'corrective',
    name: 'Corrective RAG',
    industryAliases: ['CRAG', 'Self-RAG', 'self-reflective RAG', 'evidence grading'],
    kind: 'architecture',
    status: 'composed',
    flow: 'cycle',
    stages: ['retrieve', 'evaluate', 'retrieve', 'generate'],
    mechanism:
      'A grader scores the retrieved evidence before generation; evidence that fails triggers re-retrieval or a fallback. The architecture is a cycle whose gate is a classifier, so the system is only as good as that classifier is calibrated.',
    winCondition:
      'A query mix containing retrieval failures that are actually detectable, together with a fallback that helps. Both halves are required, and the second is the one usually assumed.',
    failureMode:
      'Three, all quiet. A grader that separates perfectly makes correction vacuous — it is just always doing the right thing, and the comparison measures nothing. An uncalibrated grader fires on good retrievals, and a false-positive correction moves the query off target, so correction strictly hurts. And the confidence being thresholded is typically far more confident than it is accurate.',
    composedOf: [
      'faithfulness-groundedness',
      'conformal-factuality',
      'llm-as-judge-ragas',
      'significance-testing-calibration',
    ],
    keystone: 'faithfulness-groundedness',
  },
  {
    id: 'graph',
    name: 'Graph RAG',
    industryAliases: ['GraphRAG', 'knowledge-graph RAG', 'entity-graph retrieval'],
    kind: 'architecture',
    status: 'shipped',
    flow: 'linear',
    stages: ['ingest', 'index', 'retrieve', 'generate'],
    mechanism:
      'The retrieval unit stops being a flat chunk. Entities and relations are extracted into a graph, partitioned into communities by maximizing modularity, and summarized offline — so most of the work moves from query time to index time.',
    winCondition:
      'Global or thematic queries that require aggregating over a community, where no single passage carries the answer and the summary is the only object that does.',
    failureMode:
      'It loses to plain retrieval on simple fact lookup, at up to tens of thousands of tokens per query against a few hundred. And community detection has an information-theoretic threshold: below it, no algorithm recovers the planted structure, so the failure is not one a better implementation can fix.',
    composedOf: [
      'graphrag-community-detection',
      'multi-hop-iterative-retrieval',
      'chunking-as-segmentation',
    ],
    keystone: 'graphrag-community-detection',
  },
  {
    id: 'agentic',
    name: 'Agentic RAG',
    industryAliases: ['multi-hop RAG', 'iterative RAG', 'retrieval-as-tool', 'agentic search'],
    kind: 'architecture',
    status: 'shipped',
    flow: 'cycle',
    stages: ['retrieve', 'generate', 'retrieve', 'generate'],
    mechanism:
      'Retrieval stops being a pipeline stage and becomes an action: the system reformulates its query from what it just read and retrieves again, until a stopping rule fires. The number of retrievals is determined by the data, not fixed in advance.',
    winCondition:
      'Compositional queries whose answer is near-orthogonal to the query and reachable only through a bridge document — where a single-shot retrieval pool provably does not contain the answer at any k.',
    failureMode:
      'The stopping rule, and it fails in a way that looks correct. Belief movement is tiny at the bridge and enormous at the answer, so a myopic "stop when the belief stops moving" rule halts at the worthless-looking bridge and never reaches the answer. Per-hop retention also compounds multiplicatively, so a chain is only as good as the product of its stages.',
    composedOf: [
      'multi-hop-iterative-retrieval',
      'context-selection-submodular-dpp',
      'pmi-retrieval-value',
    ],
    keystone: 'multi-hop-iterative-retrieval',
  },
  {
    id: 'adaptive',
    name: 'Adaptive RAG',
    industryAliases: ['routing RAG', 'query routing', 'strategy selection'],
    kind: 'architecture',
    status: 'shipped',
    flow: 'branch',
    stages: ['select', 'retrieve', 'generate'],
    mechanism:
      'A router chooses which retrieval strategy to run for each query, using only features available before retrieval, trading answer quality against the cost of the arm it picks.',
    winCondition:
      'Queries that genuinely differ in the effort they require, together with real cost asymmetry across the arms. Without the second, the router’s only available win is cost, and it is worth nothing.',
    failureMode:
      'Routing is worthless — regardless of how good the classifier is — when the arms’ advantage ordering does not vary across queries. The achievable gain is exactly a Jensen gap between the expected maximum and the maximum expectation, and that gap is zero when one arm is uniformly best. A router tuned and scored on the same queries will also look roughly twice as good as it is.',
    composedOf: [
      'adaptive-retrieval-routing',
      'selective-generation-abstention',
      'retrieval-vs-long-context',
      'multi-hop-iterative-retrieval',
    ],
    keystone: 'adaptive-retrieval-routing',
  },
  {
    id: 'reranking',
    name: 'Reranking Cascade',
    industryAliases: ['cross-encoder reranking', 'two-stage retrieval', 'Advanced RAG'],
    kind: 'operator',
    status: 'shipped',
    flow: 'linear',
    stages: ['retrieve', 'rerank', 'generate'],
    mechanism:
      'A model too expensive to run over the corpus rescores a shortlist that a cheap retriever produced. Joint query–document attention escapes the rank ceiling a dual encoder is bound by, because the score is no longer an inner product of independent representations.',
    winCondition:
      'The first stage has high recall at a depth the second stage can afford. Under a known-item judgment the arithmetic is exact: an oracle reranker’s precision at one equals stage one’s recall at the cutoff, which is the entire reason the cascade works.',
    failureMode:
      'This is an operator, not an architecture — it bolts onto any of the patterns above and changes none of their topology, which is why "Advanced RAG" is a maturity level rather than a design. Recall is monotone in the shortlist depth only for an exact reranker; a lossy one can rank a confident false positive above a true neighbor, so a deeper shortlist makes it worse.',
    composedOf: [
      'cross-encoders-reranking',
      'lambdarank-lambdamart-listwise',
      'llm-listwise-rerankers',
      'retrieval-distillation',
    ],
    keystone: 'cross-encoders-reranking',
  },
  {
    id: 'long-context',
    name: 'Long Context',
    industryAliases: ['Cache-Augmented Generation', 'CAG', 'context stuffing', 'no-RAG'],
    kind: 'contrast',
    status: 'shipped',
    flow: 'linear',
    stages: ['ingest', 'generate'],
    mechanism:
      'No retrieval at inference: a bounded corpus is placed in context in full. It is on this list because it marks the boundary — the baseline every architecture above has to beat to justify existing.',
    winCondition:
      'A corpus small enough to fit, where attention over the whole of it is affordable and no selection is needed.',
    failureMode:
      'Attention cost is quadratic in context length, and accuracy is U-shaped in position, so evidence in the middle is attended least. More context is not better: adding passages dilutes the attention budget and raises the entropy of the answer even when retrieval was perfect.',
    composedOf: ['retrieval-vs-long-context', 'context-selection-submodular-dpp'],
    keystone: 'retrieval-vs-long-context',
  },
];

export const architectureById: Record<string, Architecture> = Object.fromEntries(
  architectures.map((a) => [a.id, a]),
);
