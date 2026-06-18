import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// formalRAG links UP into formalML (its deepest prerequisite layer) as well as
// DOWN into the calculus/statistics foundations — hence 'formalml' joins the
// sibling enum, unlike formalML's own schema which only links down.
const crossSiteRef = z.object({
  topic: z.string(),
  site: z.enum(['formalml', 'formalcalculus', 'formalstatistics']),
  relationship: z.string().min(40),
});

const topics = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/topics' }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    status: z.enum(['draft', 'review', 'published']),
    difficulty: z.enum(['foundational', 'intermediate', 'advanced']),
    prerequisites: z.array(z.string()).default([]),
    tags: z.array(z.string()),
    // Ten retrieval-centric domains — the objects are relevance operators, ANN
    // structures, ranked lists, and channels, not ML learning paradigms.
    domain: z.enum([
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
    ]),
    videoId: z.string().nullable().default(null),
    notebookPath: z.string().nullable().default(null),
    githubUrl: z.string().url().nullable().default(null),
    datePublished: z.coerce.date().optional(),
    dateUpdated: z.coerce.date().optional(),
    estimatedReadTime: z.number().optional(),
    abstract: z.string(),
    // --- formalRAG-specific structure (see plan: schema departures) ---
    // The recurring finance case-study thread + multimodal capstone.
    financeCaseStudy: z.boolean().default(false),
    modality: z
      .array(z.enum(['text', 'pdf', 'audio', 'chart', 'news']))
      .optional(),
    // A second navigation axis orthogonal to the math-domain taxonomy.
    pipelineStage: z
      .enum([
        'ingest',
        'index',
        'retrieve',
        'fuse',
        'rerank',
        'select',
        'generate',
        'evaluate',
      ])
      .optional(),
    // Honesty annotation: celebrated methods whose guarantees are heuristic
    // (HNSW scaling, MMR's missing submodularity guarantee, BM25's tuned k1/b).
    rigorFlag: z.string().optional(),
    connections: z
      .array(
        z.object({
          topic: z.string(),
          relationship: z.string(),
        }),
      )
      .default([]),
    references: z
      .array(
        z.object({
          // 'documentation' added: RAG cites FAISS/Qdrant/ColBERT/RAGAS docs
          // as load-bearing references for the code pillar.
          type: z.enum(['paper', 'book', 'course', 'blog', 'video', 'documentation']),
          title: z.string(),
          authors: z.string().optional(),
          year: z.number().optional(),
          url: z.string().url().optional(),
          note: z.string().optional(),
        }),
      )
      .default([]),
    // Cross-site links. formalRAG links UP into formalML (new) and DOWN into
    // the calculus/statistics foundations (mirrors the sibling-site arrays).
    formalmlPrereqs: z.array(crossSiteRef).optional(),
    formalmlConnections: z.array(crossSiteRef).optional(),
    formalcalculusPrereqs: z.array(crossSiteRef).optional(),
    formalstatisticsPrereqs: z.array(crossSiteRef).optional(),
    formalcalculusConnections: z.array(crossSiteRef).optional(),
    formalstatisticsConnections: z.array(crossSiteRef).optional(),
  }),
});

export const collections = { topics };
