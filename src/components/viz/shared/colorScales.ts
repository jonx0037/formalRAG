import * as d3 from 'd3';

// Ten RAG domains get the Tableau10 palette; the three sister sites get neutral
// grays so cross-site nodes read as "external" in the dependency graph.
// Explicit range required because the scale exceeds 10 entries (ordinal scales
// otherwise cycle).
export const domainColorScale = d3
  .scaleOrdinal<string, string>()
  .domain([
    'retrieval-foundations',
    'probabilistic-ir',
    'embedding-geometry',
    'ann-indexing',
    'vector-quantization',
    'neural-retrieval',
    'ranking-fusion',
    'retrieval-evaluation',
    'generation-grounding',
    'rag-information-theory',
    'formalml',
    'formalcalculus',
    'formalstatistics',
  ])
  .range([
    ...d3.schemeTableau10,
    '#9ca3af',
    '#6b7280',
    '#4b5563',
  ]);
