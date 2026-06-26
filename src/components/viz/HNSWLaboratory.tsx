import { memo, useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * HNSW Laboratory — three panels for the `hnsw` topic:
 *   A. The layer pyramid + descent. A small multi-layer HNSW: step from the top layer (a lone apex
 *      hub) down to layer 0 (all nodes) and watch a query's beam-1 descent refine the entry, then the
 *      layer-0 beam find the true nearest neighbor. Sparse top, dense bottom — the hierarchy made
 *      visible.
 *   B. Naive vs heuristic neighbor selection. One base node and its candidates; toggle between the M
 *      strict-nearest (which cluster on one side) and the diversity-pruned heuristic set (Algorithm 4),
 *      which keeps long-range links.
 *   C. Recall vs cost. The recall@10-versus-distance-computations frontiers of HNSW, flat NSW, and the
 *      inverted file (IVF) on the SAME cloud, plus the entry-descent depth tracking log_M(n).
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): TOY_POINTS, TOY_LEVELS, EDGES_BY_LAYER, ENTRY, TOP_LEVEL,
 * QUERY, DESCENT_PATH, FOUND, DEMO_* , FRONTIER_* and SCALING are mirrored TO THE DECIMAL from
 * notebooks/hnsw/hnsw.py (toy_hnsw_graph / heuristic_demo / head_to_head / scaling_study, printed by
 * viz_constants()). The lab recomputes only the graph layout in TS; every level, edge, recall, and
 * cost is baked. test_layer_geometric_decay / test_heuristic_diversifies /
 * test_hnsw_beats_flat_nsw_at_equal_cost / test_max_level_scales_log assert them. Change a number here
 * -> change it there, and re-run the notebook. SVG text inherits the theme color.
 */

type Pt2 = [number, number];

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: toy_hnsw_graph — 35 points (five blobs), per-node max level, per-layer edges, and one
// query's beam-1 descent. M = 3, so the levels spread (mL = 1/ln 3 ≈ 0.91).
const TOY_POINTS: Pt2[] = [
  [0.436, 6.797], [0.775, 7.684], [0.0, 7.83], [1.138, 8.804], [2.861, 9.253], [2.691, 7.952],
  [2.774, 9.355], [7.412, 8.549], [7.962, 9.296], [7.247, 7.729], [8.326, 8.232], [6.524, 8.324],
  [7.893, 7.784], [7.86, 8.197], [0.365, 3.397], [1.225, 0.0], [1.926, 3.312], [1.533, 3.396],
  [3.401, 1.224], [0.0, 0.888], [3.069, 1.265], [6.64, 0.796], [8.0, 1.977], [8.785, 2.89],
  [7.161, 1.859], [6.979, 2.065], [6.964, 0.92], [9.911, 2.029], [5.579, 7.284], [5.707, 4.897],
  [5.049, 4.33], [5.985, 4.703], [6.586, 4.69], [4.682, 4.623], [5.563, 5.168],
];
const TOY_LEVELS = [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 3, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0];
const EDGES_BY_LAYER: [number, number][][] = [
  // L0 (56 edges) — every node
  [[0, 1], [0, 3], [0, 5], [0, 14], [1, 2], [1, 3], [3, 5], [3, 6], [4, 5], [4, 6], [4, 11], [5, 17],
   [5, 28], [7, 8], [7, 11], [7, 13], [7, 28], [9, 11], [9, 12], [9, 13], [9, 34], [10, 13], [10, 23],
   [10, 27], [11, 28], [12, 13], [14, 17], [14, 19], [15, 18], [15, 19], [15, 20], [16, 17], [16, 18],
   [16, 20], [16, 33], [18, 20], [18, 21], [21, 24], [21, 26], [22, 23], [22, 24], [23, 27], [24, 25],
   [24, 26], [25, 29], [25, 30], [25, 32], [25, 34], [28, 34], [29, 30], [29, 31], [29, 32], [29, 34],
   [30, 33], [31, 32], [33, 34]],
  // L1 (15 edges)
  [[3, 17], [3, 28], [7, 8], [7, 13], [7, 28], [10, 13], [12, 13], [15, 17], [15, 21], [17, 31],
   [21, 23], [21, 26], [23, 26], [23, 31], [28, 31]],
  // L2 (2 edges)
  [[15, 17], [15, 21]],
  // L3 (apex node 15 alone, no edges)
  [],
];
const TOP_LEVEL = 3;
const QUERY: Pt2 = [2.671, 7.191];
const DESCENT_PATH: [number, number][] = [[3, 15], [2, 15], [1, 17], [0, 3]]; // (layer, node) top-first
const FOUND = 5; // the true NN, found by the layer-0 ef=16 beam from the descended entry (node 3)

// Panel B: heuristic_demo — base node, candidate set, and the two kept sets (M = 4).
const DEMO_BASE = 33;
const DEMO_CANDIDATES = [30, 34, 29, 31, 32, 28, 16, 17, 25, 18];
const NAIVE_KEPT = [30, 34, 29, 31];
const HEURISTIC_KEPT = [30, 28, 16];

// Panel C: head_to_head on the SAME cloud (n = 500, M = 8, nlist = 22). Cost = mean distance
// computations per query; recall@10. Three frontiers + the log_M(n) scaling of the descent depth.
const HH_N = 500;
const FRONTIER_HNSW = [
  { ef: 1, recall: 0.1, cost: 34.8 }, { ef: 2, recall: 0.2, cost: 39.9 }, { ef: 4, recall: 0.4, cost: 48.6 },
  { ef: 8, recall: 0.795, cost: 64.4 }, { ef: 16, recall: 1.0, cost: 92.0 }, { ef: 32, recall: 1.0, cost: 135.5 },
  { ef: 64, recall: 1.0, cost: 206.6 }, { ef: 128, recall: 1.0, cost: 307.1 },
];
const FRONTIER_NSW = [
  { ef: 1, recall: 0.1, cost: 71.5 }, { ef: 2, recall: 0.2, cost: 77.1 }, { ef: 4, recall: 0.4, cost: 86.4 },
  { ef: 8, recall: 0.8, cost: 103.3 }, { ef: 16, recall: 0.9975, cost: 134.3 }, { ef: 32, recall: 1.0, cost: 183.9 },
  { ef: 64, recall: 1.0, cost: 260.0 }, { ef: 128, recall: 1.0, cost: 365.0 },
];
const FRONTIER_IVF = [
  { nprobe: 1, recall: 0.5125, cost: 47.0 }, { nprobe: 2, recall: 0.7675, cost: 71.3 }, { nprobe: 4, recall: 0.9325, cost: 120.7 },
  { nprobe: 8, recall: 0.9975, cost: 219.7 }, { nprobe: 16, recall: 1.0, cost: 405.1 }, { nprobe: 22, recall: 1.0, cost: 522.0 },
];
const SCALING = [
  { n: 200, top_level: 2.3, log_M_n: 2.548 }, { n: 350, top_level: 2.583, log_M_n: 2.817 },
  { n: 500, top_level: 2.8, log_M_n: 2.989 }, { n: 750, top_level: 2.9, log_M_n: 3.184 },
  { n: 1000, top_level: 3.167, log_M_n: 3.322 },
];

const PW = 380, PH = 340, PAD = 26;
const wx = (x: number) => PAD + (PW - 2 * PAD) * (x / 10);
const wy = (y: number) => PH - PAD - (PH - 2 * PAD) * (y / 10);
const blobColor = ['var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', '#5fa873', '#6a8caf', '#b07cc6'];
const NSW_COLOR = '#6a8caf', IVF_COLOR = '#b07cc6';

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

// Panel A — the layer pyramid: layer `lev` shows the nodes present at level >= lev and that layer's
// edges; the query's descent node at this layer is ringed. Drawn by D3.
function PyramidPlot({ lev }: { lev: number }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const present = TOY_POINTS.map((_, i) => TOY_LEVELS[i] >= lev);
    const edges = EDGES_BY_LAYER[lev] ?? [];
    const descNode = DESCENT_PATH.find(([l]) => l === lev)?.[1];
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${PW} ${PH}`).attr('width', '100%');
    // faint context: all points
    svg.append('g').selectAll('circle.ctx').data(TOY_POINTS).join('circle').attr('class', 'ctx')
      .attr('cx', (p) => wx(p[0])).attr('cy', (p) => wy(p[1])).attr('r', 2)
      .attr('fill', 'var(--color-border)').attr('opacity', 0.5);
    // this layer's edges
    svg.append('g').selectAll('line').data(edges).join('line')
      .attr('x1', (e) => wx(TOY_POINTS[e[0]][0])).attr('y1', (e) => wy(TOY_POINTS[e[0]][1]))
      .attr('x2', (e) => wx(TOY_POINTS[e[1]][0])).attr('y2', (e) => wy(TOY_POINTS[e[1]][1]))
      .attr('stroke', 'var(--color-accent)').attr('stroke-width', 1).attr('stroke-opacity', 0.55);
    // nodes present at this layer, colored by blob
    svg.append('g').selectAll('circle.pt').data(TOY_POINTS.map((p, i) => ({ p, i })).filter((d) => present[d.i]))
      .join('circle').attr('class', 'pt')
      .attr('cx', (d) => wx(d.p[0])).attr('cy', (d) => wy(d.p[1])).attr('r', 4)
      .attr('fill', (d) => blobColor[Math.floor(d.i / 7) % blobColor.length]).attr('opacity', 0.9);
    // the descent node at this layer
    if (descNode !== undefined) {
      svg.append('circle').attr('cx', wx(TOY_POINTS[descNode][0])).attr('cy', wy(TOY_POINTS[descNode][1])).attr('r', 7.5)
        .attr('fill', 'none').attr('stroke', 'var(--color-text)').attr('stroke-width', 2);
    }
    // at layer 0, mark the true nearest neighbor the beam finds
    if (lev === 0) {
      svg.append('circle').attr('cx', wx(TOY_POINTS[FOUND][0])).attr('cy', wy(TOY_POINTS[FOUND][1])).attr('r', 7)
        .attr('fill', 'none').attr('stroke', '#3c9a5f').attr('stroke-width', 2.5);
    }
    // the query (diamond)
    const q = QUERY;
    svg.append('path').attr('d', `M${wx(q[0])} ${wy(q[1]) - 6} L${wx(q[0]) + 6} ${wy(q[1])} L${wx(q[0])} ${wy(q[1]) + 6} L${wx(q[0]) - 6} ${wy(q[1])} Z`)
      .attr('fill', 'var(--color-text)').attr('stroke', 'var(--color-bg)').attr('stroke-width', 1);
  }, [lev]);
  return <svg ref={ref} role="img" aria-label="HNSW layer pyramid and query descent" />;
}

// Panel B — base node, candidates, and the kept neighbor set for the active selection mode.
function HeuristicPlot({ heuristic }: { heuristic: boolean }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const kept = new Set(heuristic ? HEURISTIC_KEPT : NAIVE_KEPT);
    const cand = new Set(DEMO_CANDIDATES);
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${PW} ${PH}`).attr('width', '100%');
    // edges base -> kept
    svg.append('g').selectAll('line').data([...kept]).join('line')
      .attr('x1', wx(TOY_POINTS[DEMO_BASE][0])).attr('y1', wy(TOY_POINTS[DEMO_BASE][1]))
      .attr('x2', (j) => wx(TOY_POINTS[j][0])).attr('y2', (j) => wy(TOY_POINTS[j][1]))
      .attr('stroke', 'var(--color-accent)').attr('stroke-width', 2).attr('stroke-opacity', 0.9);
    // all points: faint, candidates outlined, kept filled
    svg.append('g').selectAll('circle.pt').data(TOY_POINTS.map((p, i) => ({ p, i }))).join('circle').attr('class', 'pt')
      .attr('cx', (d) => wx(d.p[0])).attr('cy', (d) => wy(d.p[1]))
      .attr('r', (d) => (kept.has(d.i) ? 5.5 : cand.has(d.i) ? 4 : 2.5))
      .attr('fill', (d) => (kept.has(d.i) ? 'var(--color-accent)' : cand.has(d.i) ? 'var(--color-text-secondary)' : 'var(--color-border)'))
      .attr('opacity', (d) => (kept.has(d.i) || cand.has(d.i) ? 0.95 : 0.5));
    // the base node (ringed)
    svg.append('circle').attr('cx', wx(TOY_POINTS[DEMO_BASE][0])).attr('cy', wy(TOY_POINTS[DEMO_BASE][1])).attr('r', 7)
      .attr('fill', 'none').attr('stroke', 'var(--color-text)').attr('stroke-width', 2.4);
  }, [heuristic]);
  return <svg ref={ref} role="img" aria-label="naive vs heuristic neighbor selection" />;
}

// Panel C — the three recall/cost frontiers (baked).
const FrontierPlot = memo(function FrontierPlot() {
  const pw = 580, ph = 280, pad = 46;
  const xmax = 380; // distance computations per query
  const fx = (c: number) => pad + (pw - 2 * pad) * Math.min(c, xmax) / xmax;
  const ry = (r: number) => pad + (ph - 2 * pad) * (1 - r);
  const line = (rows: { recall: number; cost: number }[]) =>
    rows.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.cost).toFixed(1) + ' ' + ry(d.recall).toFixed(1)).join(' ');
  const series: [string, string, { recall: number; cost: number }[]][] = [
    ['HNSW', 'var(--color-accent)', FRONTIER_HNSW], ['flat NSW', NSW_COLOR, FRONTIER_NSW], ['IVF', IVF_COLOR, FRONTIER_IVF],
  ];
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((r) => (<text key={r} x={pad - 6} y={ry(r) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(r * 100).toFixed(0)}%</text>))}
      {[0, 100, 200, 300].map((c) => (<text key={c} x={fx(c)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{c}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">distance computations per query (n = {HH_N})</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>recall@10</text>
      {series.map(([name, color, rows], si) => (
        <g key={name}>
          <path d={line(rows)} fill="none" stroke={color} strokeWidth={2.6} strokeLinejoin="round" />
          {rows.map((d, i) => (<circle key={i} cx={fx(d.cost)} cy={ry(d.recall)} r={3} fill={color} />))}
          <line x1={pw - pad - 96} y1={pad + 4 + si * 16} x2={pw - pad - 78} y2={pad + 4 + si * 16} stroke={color} strokeWidth={2.6} />
          <text x={pw - pad - 74} y={pad + 7 + si * 16} fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{name}</text>
        </g>
      ))}
    </svg>
  );
});

// Panel C alt — the entry-descent depth (mean top level) tracking log_M(n).
const ScalingPlot = memo(function ScalingPlot() {
  const pw = 580, ph = 280, pad = 46;
  const nmax = 1000;
  const sx = (n: number) => pad + (pw - 2 * pad) * (n / nmax);
  const ly = (v: number) => ph - pad - (ph - 2 * pad) * (v / 4);
  const line = (key: 'top_level' | 'log_M_n') => SCALING.map((d, i) => (i === 0 ? 'M' : 'L') + sx(d.n).toFixed(1) + ' ' + ly(d[key]).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 1, 2, 3, 4].map((v) => (<text key={v} x={pad - 6} y={ly(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
      {[200, 500, 1000].map((n) => (<text key={n} x={sx(n)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{n}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">database size n</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>expected top level</text>
      <path d={line('log_M_n')} fill="none" stroke={NSW_COLOR} strokeWidth={2} strokeDasharray="5 3" />
      <path d={line('top_level')} fill="none" stroke="var(--color-accent)" strokeWidth={2.6} />
      {SCALING.map((d, i) => (<circle key={i} cx={sx(d.n)} cy={ly(d.top_level)} r={3.2} fill="var(--color-accent)" />))}
      <text x={sx(1000) - 6} y={ly(3.322) - 8} textAnchor="end" fontSize={9.5} fill={NSW_COLOR} fontFamily="var(--font-sans)">log_M n</text>
      <text x={sx(500)} y={ly(2.8) + 16} textAnchor="middle" fontSize={9.5} fill="var(--color-accent)" fontFamily="var(--font-sans)">mean max level</text>
    </svg>
  );
});

export default function HNSWLaboratory() {
  const [panel, setPanel] = useState<'pyramid' | 'heuristic' | 'frontier'>('pyramid');
  const [lev, setLev] = useState(TOP_LEVEL);
  const [heuristic, setHeuristic] = useState(true);
  const [cView, setCView] = useState<'frontier' | 'scaling'>('frontier');
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'pyramid'
        ? 'P(L \\ge \\ell) = M^{-\\ell}, \\qquad \\mathbb{E}[\\max_i L_i] \\approx \\log_M n'
        : panel === 'heuristic'
          ? '\\text{keep } c \\iff d(c, r) \\ge d(c, \\text{base}) \\ \\ \\forall\\, r \\in \\text{kept}'
          : '\\text{cost} = \\#\\,\\text{distance computations per query} \\ \\ \\text{at fixed recall}@10';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const descNode = DESCENT_PATH.find(([l]) => l === lev)?.[1];
  const layerCount = TOY_LEVELS.filter((l) => l >= lev).length;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        <button onClick={() => setPanel('pyramid')} style={pill(panel === 'pyramid')}>The layer pyramid</button>
        <button onClick={() => setPanel('heuristic')} style={pill(panel === 'heuristic')}>Neighbor selection</button>
        <button onClick={() => setPanel('frontier')} style={pill(panel === 'frontier')}>Recall / cost</button>
      </div>

      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.6rem' }} />

      {panel === 'pyramid' && (
        <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', alignItems: 'start' }}>
          <PyramidPlot lev={lev} />
          <div>
            <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.7rem', flexWrap: 'wrap' }}>
              {[3, 2, 1, 0].map((l) => (
                <button key={l} onClick={() => setLev(l)} style={pill(lev === l)}>Layer {l}</button>
              ))}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', marginBottom: '0.6rem' }}>
              <Readout label={`nodes at layer ${lev}`} value={`${layerCount} of ${TOY_POINTS.length}`} accent={lev === TOP_LEVEL} />
              <Readout label="descent is at node" value={descNode !== undefined ? `${descNode}` : '—'} />
            </div>
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              The query (◆) enters at the lone apex hub (layer {TOP_LEVEL}) and beam-1 descends — node {DESCENT_PATH[0][1]} →{' '}
              {DESCENT_PATH.map((d) => d[1]).slice(1).join(' → ')} — each upper layer cheaply narrowing the region. At{' '}
              <strong>layer 0</strong> a width-ef beam from the descended entry finds the true nearest neighbor
              (<strong style={{ color: '#3c9a5f' }}>green ring</strong>). Step up the layers and watch the graph thin
              geometrically: ~{TOY_POINTS.length} → {TOY_LEVELS.filter((l) => l >= 1).length} → {TOY_LEVELS.filter((l) => l >= 2).length} → {TOY_LEVELS.filter((l) => l >= 3).length} nodes.
            </div>
          </div>
        </div>
      )}

      {panel === 'heuristic' && (
        <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', alignItems: 'start' }}>
          <HeuristicPlot heuristic={heuristic} />
          <div>
            <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.7rem' }}>
              <button onClick={() => setHeuristic(false)} style={pill(!heuristic)}>naive M-nearest</button>
              <button onClick={() => setHeuristic(true)} style={pill(heuristic)}>heuristic (Alg. 4)</button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', marginBottom: '0.6rem' }}>
              <Readout label="kept neighbors" value={(heuristic ? HEURISTIC_KEPT : NAIVE_KEPT).join(', ')} accent={heuristic} />
              <Readout label="count" value={`${(heuristic ? HEURISTIC_KEPT : NAIVE_KEPT).length} of M = 4`} />
            </div>
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              From base node {DEMO_BASE} (○), the <strong>naive</strong> rule keeps the M = 4 strict-nearest
              ({NAIVE_KEPT.join(', ')}) — all bunched in one cluster. The <strong>heuristic</strong> admits a candidate only
              if no already-kept neighbor is closer to it than it is to the base, so it drops the redundant near ones and keeps a
              spread-out set ({HEURISTIC_KEPT.join(', ')}) that preserves a long-range link. No optimality proof — a diversity
              rule that keeps the graph navigable.
            </div>
          </div>
        </div>
      )}

      {panel === 'frontier' && (
        <>
          <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.7rem' }}>
            <button onClick={() => setCView('frontier')} style={pill(cView === 'frontier')}>recall vs cost</button>
            <button onClick={() => setCView('scaling')} style={pill(cView === 'scaling')}>log_M(n) scaling</button>
          </div>
          {cView === 'frontier' ? <FrontierPlot /> : <ScalingPlot />}
          {cView === 'frontier' ? (
            <>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.5rem' }}>
                <Readout label="HNSW: recall 0.9 at" value="92 comps" accent />
                <Readout label="flat NSW: recall 0.9 at" value="134 comps" />
                <Readout label="IVF at 92 comps" value="≈77% recall" />
              </div>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
                All three indexes on the SAME cloud (n = {HH_N}, M = 8) with one shared ground truth. The robust, intra-graph
                headline: HNSW reaches a given recall at no more distance computations than the flat NSW it layers — the upper
                layers buy a cheaper entry. Against the inverted file the comparison is a property of <em>this</em> synthetic
                low-rank cloud, not a universal ranking.
              </div>
            </>
          ) : (
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
              The expected maximum level — the entry-descent depth — tracks log_M(n) (dashed), growing only logarithmically as the
              database grows. This is the provable spine: it follows from the exact level law P(L ≥ ℓ) = M⁻ℓ, measured here from
              the level draws alone, independent of the graph.
            </div>
          )}
        </>
      )}
    </div>
  );
}
