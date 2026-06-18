/**
 * Validates that all topic connections and prerequisites reference existing topic IDs.
 * Run with: pnpm validate
 *
 * formalRAG note: curriculum-graph.json holds the FULL 50-topic roadmap, so most
 * graph nodes have no MDX file yet — that is expected and reported as a notice,
 * not an error. Only an MDX file with no graph node, or a prerequisite/connection
 * pointing at a non-existent local topic, is a hard error. The cross-site
 * reference arrays (formalml, formalcalculus, formalstatistics prereqs and
 * connections) are ignored here; they are checked by `pnpm audit:cross-site`.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join, basename } from 'node:path';

const TOPICS_DIR = join(import.meta.dirname, '..', 'content', 'topics');
const CURRICULUM_GRAPH = join(import.meta.dirname, '..', 'data', 'curriculum-graph.json');

// Collect all valid topic IDs from MDX filenames
const mdxFiles = readdirSync(TOPICS_DIR).filter((f) => f.endsWith('.mdx'));
const validIds = new Set(mdxFiles.map((f) => basename(f, '.mdx')));

// Also collect IDs from curriculum-graph.json
const graph = JSON.parse(readFileSync(CURRICULUM_GRAPH, 'utf-8'));
const graphIds = new Set<string>(graph.nodes.map((n: { id: string }) => n.id));

let errors = 0;
let notices = 0;

// Every MDX file must have a graph node (catches slug typos).
for (const id of validIds) {
  if (!graphIds.has(id)) {
    console.error(`[MISMATCH] MDX file "${id}.mdx" has no entry in curriculum-graph.json`);
    errors++;
  }
}
// Graph nodes without an MDX file are roadmap topics not yet authored — notice only.
for (const id of graphIds) {
  if (!validIds.has(id)) {
    notices++;
  }
}

// Check graph edges reference valid nodes
for (const edge of graph.edges) {
  if (!graphIds.has(edge.source)) {
    console.error(`[EDGE] Edge source "${edge.source}" is not a valid node`);
    errors++;
  }
  if (!graphIds.has(edge.target)) {
    console.error(`[EDGE] Edge target "${edge.target}" is not a valid node`);
    errors++;
  }
}

// Extracts the body of a single top-level YAML block (e.g. `connections:`) up to
// the next top-level key, so we can validate connection topics without matching
// the cross-site *Prereqs/*Connections arrays.
function extractBlock(frontmatter: string, key: string): string {
  const re = new RegExp(`(^|\\n)${key}:\\s*\\n([\\s\\S]*?)(?=\\n[a-zA-Z][\\w]*:|$)`);
  const m = frontmatter.match(re);
  return m ? m[2] : '';
}

// Parse frontmatter from each MDX file and validate references
for (const file of mdxFiles) {
  const topicId = basename(file, '.mdx');
  const content = readFileSync(join(TOPICS_DIR, file), 'utf-8');

  // Extract YAML frontmatter between --- delimiters
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) {
    console.error(`[PARSE] Could not extract frontmatter from ${file}`);
    errors++;
    continue;
  }

  const frontmatter = match[1];

  // Validate prerequisites (block form and inline form).
  const prereqBlock = extractBlock(frontmatter, 'prerequisites');
  for (const m of prereqBlock.matchAll(/-\s+"([^"]+)"/g)) {
    if (!validIds.has(m[1])) {
      console.error(`[PREREQ] ${topicId}: prerequisite "${m[1]}" is not a valid topic`);
      errors++;
    }
  }
  const inlinePrereq = frontmatter.match(/prerequisites:\s*\[([^\]]*)\]/);
  if (inlinePrereq) {
    for (const item of inlinePrereq[1].matchAll(/"([^"]+)"/g)) {
      if (!validIds.has(item[1])) {
        console.error(`[PREREQ] ${topicId}: prerequisite "${item[1]}" is not a valid topic`);
        errors++;
      }
    }
  }

  // Validate connections — scoped to the `connections:` block only, so cross-site
  // `- topic:` lines under formalml*/formalcalculus*/formalstatistics* are ignored.
  const connectionsBlock = extractBlock(frontmatter, 'connections');
  for (const m of connectionsBlock.matchAll(/-\s*topic:\s*"([^"]+)"/g)) {
    if (!validIds.has(m[1])) {
      console.error(`[CONNECTION] ${topicId}: connection topic "${m[1]}" is not a valid topic`);
      errors++;
    }
  }
}

if (notices > 0) {
  console.log(`${notices} roadmap node(s) in curriculum-graph.json have no MDX yet (expected).`);
}

if (errors > 0) {
  console.error(`\n${errors} validation error(s) found.`);
  process.exit(1);
} else {
  console.log('All connections, prerequisites, and graph references are valid.');
}
