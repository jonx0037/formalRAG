// The eight pipeline stages every RAG system is assembled from.
//
// `pipelineStage` has been carried in topic frontmatter since the schema was
// written, but nothing read it. It is the second navigation axis: `domain`
// says what branch of mathematics a topic belongs to, `pipelineStage` says
// where in a running system that mathematics executes.
//
// It is deliberately NOT a browse axis. The distribution across 50 published
// topics is lopsided — retrieve 25, index 8, evaluate 6, rerank 5, generate 2,
// fuse 2, select 1, ingest 1 — so a per-stage index page would be one large
// bucket and six stubs. Its use is structural: /architectures renders these
// eight as a fixed band and draws each architecture as a path through it,
// which is what makes the named architectures comparable at a glance.
export const stages = [
  {
    id: 'ingest',
    label: 'Ingest',
    blurb: 'Segmenting a corpus into retrievable units — the choice that fixes what can ever be retrieved.',
  },
  {
    id: 'index',
    label: 'Index',
    blurb: 'Building the structure searched at query time: a flat vector store, a quantized codebook, a navigable graph, or a community-summarized entity graph.',
  },
  {
    id: 'retrieve',
    label: 'Retrieve',
    blurb: 'Scoring the corpus against a probe and returning candidates — one similarity functional, evaluated sublinearly.',
  },
  {
    id: 'fuse',
    label: 'Fuse',
    blurb: 'Combining several independent rankings into one, on ranks rather than scores so the combination is scale-invariant.',
  },
  {
    id: 'rerank',
    label: 'Rerank',
    blurb: 'Rescoring a shortlist with a model too expensive to run over the corpus — the cascade that buys precision with compute.',
  },
  {
    id: 'select',
    label: 'Select',
    blurb: 'Choosing which candidates actually enter the context window, and which strategy to spend on this query at all.',
  },
  {
    id: 'generate',
    label: 'Generate',
    blurb: 'Conditioning the answer on the selected context — the step whose failures retrieval metrics cannot see.',
  },
  {
    id: 'evaluate',
    label: 'Evaluate',
    blurb: 'Measuring what the system did, as an estimator with a standard error rather than a number on a slide.',
  },
] as const;

export type StageId = (typeof stages)[number]['id'];

export const stageLabelMap: Record<string, string> = Object.fromEntries(
  stages.map((s) => [s.id, s.label]),
);

export const stageOrder: Record<string, number> = Object.fromEntries(
  stages.map((s, i) => [s.id, i]),
);
