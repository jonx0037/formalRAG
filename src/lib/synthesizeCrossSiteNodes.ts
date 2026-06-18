import type { DAGEdge, DAGNode } from '../components/viz/shared/types';

// formalRAG links UP into formalML (its deepest prerequisites) and DOWN into the
// calculus/statistics foundations — three sister sites, unlike formalML which
// only links to two.
type CrossSiteSite = 'formalml' | 'formalcalculus' | 'formalstatistics';

interface CrossSiteRef {
  topic: string;
  site: CrossSiteSite;
  relationship: string;
}

interface PublishedTopic {
  id: string;
  data: {
    formalmlPrereqs?: CrossSiteRef[];
    formalcalculusPrereqs?: CrossSiteRef[];
    formalstatisticsPrereqs?: CrossSiteRef[];
  };
}

export interface ExternalNode extends DAGNode {
  status: 'external';
  domain: CrossSiteSite;
  url: string;
  external: true;
}

const SITE_DOMAINS: Record<CrossSiteSite, string> = {
  formalml: 'https://formalml.com',
  formalcalculus: 'https://formalcalculus.com',
  formalstatistics: 'https://formalstatistics.com',
};

// Lowercase tokens preserved as uppercase when humanizing a slug — common
// IR / retrieval / ML acronyms that appear in cross-site slugs.
const ACRONYMS = new Set([
  'ab', 'ann', 'anova', 'bim', 'bm25', 'cdf', 'colbert', 'dpp', 'dpr', 'em',
  'fft', 'glm', 'gp', 'hmc', 'hnsw', 'hyde', 'idf', 'iid', 'ivf', 'kl', 'ks',
  'lsh', 'map', 'mcmc', 'mips', 'ml', 'mle', 'mmr', 'mrr', 'ndcg', 'ode', 'opq',
  'pac', 'pca', 'pde', 'pdf', 'pmi', 'pq', 'prp', 'ragas', 'rm3', 'rrf', 'rsj',
  'sgd', 'splade', 'svd', 'tda', 'tf', 'vc', 'vmf', 'wand',
]);

export function humanReadableFromSlug(slug: string): string {
  return slug
    .split('-')
    .map((word) => {
      if (word === 'and') return '&';
      if (ACRONYMS.has(word)) return word.toUpperCase();
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(' ');
}

/**
 * Walks each topic's frontmatter and emits ephemeral cross-site nodes
 * (one per unique sister-site prereq) plus edges pointing into the
 * formalRAG topic. Pure build-time transform — caller merges the result
 * with the canonical curriculum-graph.json data before passing to the
 * renderer.
 *
 * Caller is responsible for filtering to `data.status === 'published'`
 * before calling — this function trusts its input and does not re-filter.
 */
export function synthesizeCrossSiteNodes(
  topics: PublishedTopic[],
): { nodes: ExternalNode[]; edges: DAGEdge[] } {
  const nodes = new Map<string, ExternalNode>();
  const edges: DAGEdge[] = [];

  for (const topic of topics) {
    const refs: Array<{ refs: CrossSiteRef[] | undefined; site: CrossSiteSite }> = [
      { refs: topic.data.formalmlPrereqs, site: 'formalml' },
      { refs: topic.data.formalcalculusPrereqs, site: 'formalcalculus' },
      { refs: topic.data.formalstatisticsPrereqs, site: 'formalstatistics' },
    ];

    for (const { refs: list, site } of refs) {
      if (!list) continue;
      for (const ref of list) {
        const id = `${site}/${ref.topic}`;
        if (!nodes.has(id)) {
          nodes.set(id, {
            id,
            label: humanReadableFromSlug(ref.topic),
            domain: site,
            status: 'external',
            url: `${SITE_DOMAINS[site]}/topics/${ref.topic}`,
            external: true,
          });
        }
        edges.push({ source: id, target: topic.id });
      }
    }
  }

  return { nodes: Array.from(nodes.values()), edges };
}
