import { memo, useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * Filtered & Incremental ANN Laboratory — three panels for the `filtered-incremental-ann` topic:
 *   A. The over-fetch law (deletion = filtering). A toy HNSW with a tombstone set, beside the single
 *      hyperbola scan = k/r versus the pass-rate r: the tombstone measurements (r = 1 - delta) and the
 *      post-filter measurements (r = s) land on the SAME curve — a tombstone is a persistent global
 *      predicate, a predicate is a per-query tombstone.
 *   B. Connectivity = percolation. Random-deletion giant-component fraction vs retention p, for a
 *      near-regular configuration-model graph (where p_c = 1/(M-1) is exact) and HNSW's real layer-0
 *      graph (where it only approximates); plus the giant fraction rising with the degree M.
 *   C. Predicate search. The pre/post/in-filter recall and cost crossover vs selectivity s, and the
 *      induced-subgraph connectivity of a random vs a spatially-coherent (modality) predicate.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): TOY_POINTS, TOY_MODALITIES, EDGES_L0,
 * QUERY, DELETED, FOUND_LIVE, OVERFETCH_LAW, POSTFILTER_LAW, REG_PC, HNSW_PC, PERCOLATION_REGULAR,
 * PERCOLATION_HNSW, CONNECTIVITY_VS_M, FRONTIER_PRE/POST/IN, SUBGRAPH are mirrored TO THE DECIMAL
 * from notebooks/filtered-incremental-ann/filtered_incremental_ann.py (viz_constants()). The lab
 * recomputes only the toy-graph layout and the closed-form k/r curve in TS; every measured scan,
 * giant fraction, recall, and cost is baked. test_overfetch_law_tombstone / test_postfilter_overfetch_law
 * / test_regular_graph_percolation_threshold / test_giant_component_monotone_in_M /
 * test_filter_strategy_crossover / test_predicate_subgraph_percolation assert them. Change a number
 * here -> change it there, and re-run the notebook. No d3 drag handlers (sliders only). SVG text
 * inherits the theme color.
 */

type Pt2 = [number, number];

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: toy_filtered_graph — 35 points (five modality blobs), per-node max level, modality label,
// layer-0 edges, a ~23% tombstone set, and the live nearest neighbor of one query. M = 3.
const TOY_POINTS: Pt2[] = [
  [3.633, 5.955], [2.334, 7.546], [1.638, 7.828], [0.384, 7.814], [1.308, 10.0], [2.181, 7.718],
  [1.775, 7.466], [7.156, 7.687], [8.386, 7.809], [8.766, 7.84], [8.019, 9.237], [8.436, 7.596],
  [7.854, 8.432], [9.548, 7.784], [1.805, 2.802], [1.291, 1.767], [2.706, 2.464], [2.073, 2.536],
  [0.0, 2.817], [1.232, 0.665], [2.221, 2.56], [7.644, 1.139], [8.021, 1.958], [9.124, 2.598],
  [8.155, 2.889], [7.836, 1.259], [8.467, 2.466], [7.828, 1.374], [5.183, 3.005], [5.552, 5.393],
  [3.689, 5.049], [4.229, 5.606], [3.373, 4.268], [5.568, 5.925], [3.274, 4.602],
];
const TOY_MODALITIES = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4];
const EDGES_L0: [number, number][] = [
  [0, 1], [0, 6], [0, 31], [1, 5], [1, 6], [2, 3], [2, 4], [2, 6], [3, 6], [4, 5], [5, 6], [7, 12],
  [7, 33], [8, 9], [8, 11], [8, 12], [9, 13], [10, 12], [11, 12], [11, 13], [11, 23], [11, 33],
  [14, 17], [14, 18], [14, 32], [15, 17], [15, 18], [15, 19], [16, 20], [16, 32], [17, 20], [20, 28],
  [20, 29], [20, 32], [20, 34], [21, 25], [21, 27], [21, 28], [22, 26], [22, 27], [23, 26], [24, 26],
  [24, 28], [24, 29], [25, 27], [26, 27], [28, 29], [28, 32], [29, 31], [29, 33], [30, 31], [30, 34],
  [32, 34],
];
const QUERY: Pt2 = [1.811, 7.276];
const DELETED = new Set([0, 4, 10, 14, 18, 21, 22, 30]);
const FOUND_LIVE = 6;
const MODALITY_NAMES = ['10-K', 'news', 'PDF', 'chart', 'audio'];

const TOPK = 10;
// Panel A: the two over-fetch measurements, plotted vs the pass-rate r (tombstone r = 1 - delta;
// post-filter r = s). predicted = k / r is the closed-form law (drawn smooth in TS).
const OVERFETCH_LAW = [
  { delta: 0.1, scanned: 11.14, predicted: 11.11 }, { delta: 0.25, scanned: 13.33, predicted: 13.33 },
  { delta: 0.5, scanned: 20.09, predicted: 20.0 }, { delta: 0.7, scanned: 33.85, predicted: 33.33 },
];
const POSTFILTER_LAW = [
  { s: 0.05, scanned: 197.9, predicted: 200.0 }, { s: 0.1, scanned: 94.0, predicted: 100.0 },
  { s: 0.2, scanned: 50.8, predicted: 50.0 }, { s: 0.4, scanned: 24.9, predicted: 25.0 },
  { s: 0.8, scanned: 12.5, predicted: 12.5 },
];

// Panel B: random-deletion percolation. p in {0, .04, ..., .48}; giant_frac = largest component / n.
const PERC_P = [0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48];
const PERCOLATION_REGULAR = [0, 0.0011, 0.0026, 0.0062, 0.0437, 0.1288, 0.1947, 0.2477, 0.2975, 0.346, 0.3907, 0.4394, 0.4729];
const PERCOLATION_HNSW = [0, 0.0057, 0.0097, 0.0183, 0.0432, 0.065, 0.124, 0.2142, 0.2682, 0.3212, 0.3724, 0.4099, 0.4463];
const REG_PC = 0.143, REG_DEG = 8, HNSW_PC = 0.1285, HNSW_KAPPA = 8.784, HNSW_MEAN_DEG = 7.36;
const CONNECTIVITY_VS_M = [
  { M: 4, mean_deg: 4.7, giant_frac: 0.0158 }, { M: 8, mean_deg: 7.36, giant_frac: 0.0344 },
  { M: 16, mean_deg: 9.35, giant_frac: 0.0736 }, { M: 32, mean_deg: 10.02, giant_frac: 0.0822 },
];

// Panel C: the pre/post/in-filter frontier vs selectivity (random predicate, n = 500 index, ef = 64).
const FRONTIER_PRE = [
  { s: 0.04, recall: 1.0, ndist: 20.0 }, { s: 0.084, recall: 1.0, ndist: 42.0 }, { s: 0.224, recall: 1.0, ndist: 112.0 },
  { s: 0.456, recall: 1.0, ndist: 228.0 }, { s: 0.794, recall: 1.0, ndist: 397.0 },
];
const FRONTIER_POST = [
  { s: 0.04, recall: 0.25, ndist: 206.6 }, { s: 0.084, recall: 0.52, ndist: 206.6 }, { s: 0.224, recall: 0.9925, ndist: 206.6 },
  { s: 0.456, recall: 1.0, ndist: 206.6 }, { s: 0.794, recall: 1.0, ndist: 206.6 },
];
const FRONTIER_IN = [
  { s: 0.04, recall: 1.0, ndist: 517.7 }, { s: 0.084, recall: 1.0, ndist: 517.7 }, { s: 0.224, recall: 1.0, ndist: 459.2 },
  { s: 0.456, recall: 1.0, ndist: 323.7 }, { s: 0.794, recall: 1.0, ndist: 237.8 },
];
// Induced-subgraph giant fraction (largest component / n_live) at matched selectivity.
const SUBGRAPH = [
  { s: 0.052, giant_corr: 0.8846, giant_random: 0.1327 }, { s: 0.11, giant_corr: 0.8909, giant_random: 0.1349 },
  { s: 0.288, giant_corr: 0.9583, giant_random: 0.6145 }, { s: 0.52, giant_corr: 0.9846, giant_random: 0.9681 },
  { s: 1.0, giant_corr: 1.0, giant_random: 1.0 },
];

const PW = 360, PH = 320, PAD = 24;
const wx = (x: number) => PAD + (PW - 2 * PAD) * (x / 10);
const wy = (y: number) => PH - PAD - (PH - 2 * PAD) * (y / 10);
const blobColor = ['var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', '#5fa873', '#6a8caf', '#b07cc6'];
const PRE_COLOR = '#5fa873', POST_COLOR = '#b07cc6', IN_COLOR = 'var(--color-accent)';
const REG_COLOR = '#6a8caf', HNSW_COLOR = 'var(--color-accent)';

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

function Slider({ label, value, min, max, step, onChange, fmt }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '8.5rem' }}>{label} = <strong>{fmt(value)}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
    </label>
  );
}

// Panel A left — the toy HNSW with its tombstone set: live nodes colored by modality, tombstoned
// nodes greyed with an ×, the query (◆) and its live nearest neighbor (green ring). Drawn by D3.
function ToyGraphPlot() {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${PW} ${PH}`).attr('width', '100%');
    svg.append('g').selectAll('line').data(EDGES_L0).join('line')
      .attr('x1', (e) => wx(TOY_POINTS[e[0]][0])).attr('y1', (e) => wy(TOY_POINTS[e[0]][1]))
      .attr('x2', (e) => wx(TOY_POINTS[e[1]][0])).attr('y2', (e) => wy(TOY_POINTS[e[1]][1]))
      .attr('stroke', 'var(--color-border)').attr('stroke-width', 1).attr('stroke-opacity', 0.7);
    const nodes = TOY_POINTS.map((p, i) => ({ p, i }));
    // live nodes: filled by modality
    svg.append('g').selectAll('circle.live').data(nodes.filter((d) => !DELETED.has(d.i))).join('circle')
      .attr('cx', (d) => wx(d.p[0])).attr('cy', (d) => wy(d.p[1])).attr('r', 4.5)
      .attr('fill', (d) => blobColor[TOY_MODALITIES[d.i] ?? 0]).attr('opacity', 0.92);
    // tombstoned nodes: grey with an ×
    const dead = svg.append('g');
    nodes.filter((d) => DELETED.has(d.i)).forEach((d) => {
      const x = wx(d.p[0]), y = wy(d.p[1]);
      dead.append('circle').attr('cx', x).attr('cy', y).attr('r', 4).attr('fill', 'var(--color-border)').attr('opacity', 0.6);
      dead.append('path').attr('d', `M${x - 3} ${y - 3} L${x + 3} ${y + 3} M${x - 3} ${y + 3} L${x + 3} ${y - 3}`)
        .attr('stroke', 'var(--color-text-secondary)').attr('stroke-width', 1.2);
    });
    // the live nearest neighbor (green ring) and the query (diamond)
    svg.append('circle').attr('cx', wx(TOY_POINTS[FOUND_LIVE][0])).attr('cy', wy(TOY_POINTS[FOUND_LIVE][1])).attr('r', 7.5)
      .attr('fill', 'none').attr('stroke', '#3c9a5f').attr('stroke-width', 2.5);
    const q = QUERY;
    svg.append('path').attr('d', `M${wx(q[0])} ${wy(q[1]) - 6} L${wx(q[0]) + 6} ${wy(q[1])} L${wx(q[0]) } ${wy(q[1]) + 6} L${wx(q[0]) - 6} ${wy(q[1])} Z`)
      .attr('fill', 'var(--color-text)').attr('stroke', 'var(--color-bg)').attr('stroke-width', 1);
  }, []);
  return <svg ref={ref} role="img" aria-label="toy HNSW graph with a tombstone set" />;
}

// Panel A right — the unified over-fetch hyperbola scan = k/r, with the two measurement sets.
function OverfetchPlot({ rate }: { rate: number }) {
  const pw = 560, ph = 300, pad = 48, ymax = 210;
  const fx = (r: number) => pad + (pw - 2 * pad) * r;
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * Math.min(v, ymax) / ymax;
  const curve = d3.range(0.05, 1.001, 0.01).map((r) => `${fx(r).toFixed(1)} ${fy(TOPK / r).toFixed(1)}`);
  const marker = TOPK / Math.max(rate, 1e-6);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 50, 100, 150, 200].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
      {[0, 0.25, 0.5, 0.75, 1].map((r) => (<text key={r} x={fx(r)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">pass-rate r  (tombstone 1−δ,  predicate s)</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>candidates scanned for k = {TOPK}</text>
      <path d={'M' + curve.join(' L')} fill="none" stroke="var(--color-text-secondary)" strokeWidth={2} strokeDasharray="5 3" />
      {OVERFETCH_LAW.map((d, i) => (<circle key={`t${i}`} cx={fx(1 - d.delta)} cy={fy(d.scanned)} r={4} fill={REG_COLOR} />))}
      {POSTFILTER_LAW.map((d, i) => (<circle key={`p${i}`} cx={fx(d.s)} cy={fy(d.scanned)} r={4} fill={POST_COLOR} />))}
      <line x1={fx(rate)} y1={pad} x2={fx(rate)} y2={ph - pad} stroke="var(--color-accent)" strokeWidth={1} strokeDasharray="3 3" />
      <circle cx={fx(rate)} cy={fy(marker)} r={4.5} fill="var(--color-accent)" />
      <g fontFamily="var(--font-sans)" fontSize={10}>
        <circle cx={pw - pad - 150} cy={pad + 6} r={4} fill={REG_COLOR} /><text x={pw - pad - 142} y={pad + 9} fill="var(--color-text-secondary)">tombstone (1−δ)</text>
        <circle cx={pw - pad - 150} cy={pad + 22} r={4} fill={POST_COLOR} /><text x={pw - pad - 142} y={pad + 25} fill="var(--color-text-secondary)">predicate (s)</text>
      </g>
    </svg>
  );
}

// Panel B — random-deletion giant fraction vs retention p, two series, p_c markers + a p marker.
function PercolationPlot({ p }: { p: number }) {
  const pw = 580, ph = 300, pad = 48, pmax = 0.48;
  const fx = (v: number) => pad + (pw - 2 * pad) * v / pmax;
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * v / 0.5;
  const line = (ys: number[]) => PERC_P.map((px, i) => (i === 0 ? 'M' : 'L') + fx(px).toFixed(1) + ' ' + fy(ys[i]).toFixed(1)).join(' ');
  const series: [string, string, number[], number][] = [
    [`regular (deg ${REG_DEG})`, REG_COLOR, PERCOLATION_REGULAR, REG_PC],
    ['HNSW layer 0', HNSW_COLOR, PERCOLATION_HNSW, HNSW_PC],
  ];
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.25, 0.5].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(2)}</text>))}
      {[0, 0.1, 0.2, 0.3, 0.4].map((v) => (<text key={v} x={fx(v)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">retention probability p (fraction surviving)</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>giant-component fraction</text>
      {series.map(([name, color, ys, pc], si) => (
        <g key={name}>
          <line x1={fx(pc)} y1={pad} x2={fx(pc)} y2={ph - pad} stroke={color} strokeWidth={1} strokeDasharray="2 3" opacity={0.8} />
          <text x={fx(pc)} y={pad - 3} textAnchor="middle" fontSize={8.5} fill={color} fontFamily="var(--font-sans)">p_c={pc.toFixed(3)}</text>
          <path d={line(ys)} fill="none" stroke={color} strokeWidth={2.6} strokeLinejoin="round" />
          {ys.map((v, i) => (<circle key={i} cx={fx(PERC_P[i])} cy={fy(v)} r={2.6} fill={color} />))}
          <line x1={pw - pad - 130} y1={pad + 4 + si * 16} x2={pw - pad - 112} y2={pad + 4 + si * 16} stroke={color} strokeWidth={2.6} />
          <text x={pw - pad - 108} y={pad + 7 + si * 16} fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{name}</text>
        </g>
      ))}
      <line x1={fx(p)} y1={pad} x2={fx(p)} y2={ph - pad} stroke="var(--color-text)" strokeWidth={1} strokeDasharray="4 3" opacity={0.5} />
    </svg>
  );
}

// Panel B alt — the giant fraction at fixed retention rising with the degree M (a bar chart).
const ConnectivityVsMPlot = memo(function ConnectivityVsMPlot({ sel }: { sel: number }) {
  const pw = 580, ph = 300, pad = 48;
  const bw = (pw - 2 * pad) / CONNECTIVITY_VS_M.length;
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * v / 0.1;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.05, 0.1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(2)}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">graph degree M (mean degree below)</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>giant fraction @ p = 0.15</text>
      {CONNECTIVITY_VS_M.map((d, i) => {
        const x = pad + i * bw + bw * 0.2;
        return (
          <g key={d.M}>
            <rect x={x} y={fy(d.giant_frac)} width={bw * 0.6} height={ph - pad - fy(d.giant_frac)}
              fill={i === sel ? 'var(--color-accent)' : 'var(--color-border)'} opacity={i === sel ? 0.95 : 0.7} />
            <text x={x + bw * 0.3} y={fy(d.giant_frac) - 5} textAnchor="middle" fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">{d.giant_frac.toFixed(3)}</text>
            <text x={x + bw * 0.3} y={ph - pad + 14} textAnchor="middle" fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">M={d.M}</text>
            <text x={x + bw * 0.3} y={ph - pad + 26} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">⟨k⟩={d.mean_deg}</text>
          </g>
        );
      })}
    </svg>
  );
});

// Panel C — pre/post/in-filter recall or cost vs selectivity, with an s marker.
function StrategyPlot({ metric, sIdx }: { metric: 'recall' | 'ndist'; sIdx: number }) {
  const pw = 580, ph = 300, pad = 50;
  const ss = FRONTIER_PRE.map((d) => d.s);
  const ymax = metric === 'recall' ? 1.05 : 540;
  const fx = (s: number) => pad + (pw - 2 * pad) * s / 0.85;
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * v / ymax;
  const line = (rows: { s: number; recall: number; ndist: number }[]) =>
    rows.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.s).toFixed(1) + ' ' + fy(d[metric]).toFixed(1)).join(' ');
  const series: [string, string, typeof FRONTIER_PRE][] = [
    ['pre-filter', PRE_COLOR, FRONTIER_PRE], ['post-filter', POST_COLOR, FRONTIER_POST], ['in-filter', IN_COLOR, FRONTIER_IN],
  ];
  const yticks = metric === 'recall' ? [0, 0.5, 1] : [0, 100, 200, 300, 400, 500];
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {yticks.map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{metric === 'recall' ? `${(v * 100).toFixed(0)}%` : v}</text>))}
      {ss.map((s) => (<text key={s} x={fx(s)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{s.toFixed(2)}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">predicate selectivity s (fraction passing)</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>{metric === 'recall' ? 'recall@10' : 'distance comps / query'}</text>
      <line x1={fx(ss[sIdx] ?? ss[0])} y1={pad} x2={fx(ss[sIdx] ?? ss[0])} y2={ph - pad} stroke="var(--color-text)" strokeWidth={1} strokeDasharray="4 3" opacity={0.45} />
      {series.map(([name, color, rows], si) => (
        <g key={name}>
          <path d={line(rows)} fill="none" stroke={color} strokeWidth={2.6} strokeLinejoin="round" />
          {rows.map((d, i) => (<circle key={i} cx={fx(d.s)} cy={fy(d[metric])} r={i === sIdx ? 4.5 : 2.8} fill={color} />))}
          <line x1={pw - pad - 110} y1={pad + 4 + si * 16} x2={pw - pad - 92} y2={pad + 4 + si * 16} stroke={color} strokeWidth={2.6} />
          <text x={pw - pad - 88} y={pad + 7 + si * 16} fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{name}</text>
        </g>
      ))}
    </svg>
  );
}

// Panel C alt — induced-subgraph connectivity: a coherent (modality) vs random predicate.
const SubgraphPlot = memo(function SubgraphPlot() {
  const pw = 580, ph = 300, pad = 50;
  const fx = (s: number) => pad + (pw - 2 * pad) * s;
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * v;
  const line = (key: 'giant_corr' | 'giant_random') => SUBGRAPH.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.s).toFixed(1) + ' ' + fy(d[key]).toFixed(1)).join(' ');
  const series: [string, string, 'giant_corr' | 'giant_random'][] = [
    ['coherent (modality)', PRE_COLOR, 'giant_corr'], ['random predicate', POST_COLOR, 'giant_random'],
  ];
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(v * 100).toFixed(0)}%</text>))}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (<text key={v} x={fx(v)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(2)}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">selectivity s (fraction passing)</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>passing-set giant fraction</text>
      {series.map(([name, color, key], si) => (
        <g key={name}>
          <path d={line(key)} fill="none" stroke={color} strokeWidth={2.6} strokeLinejoin="round" />
          {SUBGRAPH.map((d, i) => (<circle key={i} cx={fx(d.s)} cy={fy(d[key])} r={3} fill={color} />))}
          <line x1={pw - pad - 150} y1={pad + 4 + si * 16} x2={pw - pad - 132} y2={pad + 4 + si * 16} stroke={color} strokeWidth={2.6} />
          <text x={pw - pad - 128} y={pad + 7 + si * 16} fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{name}</text>
        </g>
      ))}
    </svg>
  );
});

export default function FilteredIncrementalANNLaboratory() {
  const [panel, setPanel] = useState<'deletion' | 'percolation' | 'predicate'>('deletion');
  const [rate, setRate] = useState(0.5);
  const [bView, setBView] = useState<'sweep' | 'degree'>('sweep');
  const [pVal, setPVal] = useState(0.2);
  const [mSel, setMSel] = useState(1);
  const [cView, setCView] = useState<'recall' | 'cost' | 'subgraph'>('recall');
  const [sIdx, setSIdx] = useState(0);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'deletion'
        ? '\\mathbb{E}[N_k] = \\frac{k}{r}, \\quad r = 1-\\delta \\ (\\text{tombstone}) = s \\ (\\text{predicate})'
        : panel === 'percolation'
          ? 'p_c = \\frac{1}{M-1}, \\qquad \\kappa = \\langle k^2\\rangle/\\langle k\\rangle'
          : '\\mathbb{E}[N_k] = \\frac{k}{s}, \\qquad \\text{recall} < 1 \\iff s < k/F';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  // nearest baked p row for the readout (sweep) — guarded array lookup
  const pNearIdx = PERC_P.reduce((best, px, i) => (Math.abs(px - pVal) < Math.abs(PERC_P[best] - pVal) ? i : best), 0);
  const sStrat = FRONTIER_PRE.map((d) => d.s);
  const idx = Math.min(Math.max(sIdx, 0), sStrat.length - 1);

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        <button onClick={() => setPanel('deletion')} style={pill(panel === 'deletion')}>Over-fetch law</button>
        <button onClick={() => setPanel('percolation')} style={pill(panel === 'percolation')}>Connectivity</button>
        <button onClick={() => setPanel('predicate')} style={pill(panel === 'predicate')}>Predicate search</button>
      </div>

      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.6rem' }} />

      {panel === 'deletion' && (
        <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', alignItems: 'start' }}>
          <div>
            <ToyGraphPlot />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem', margin: '0.3rem 0', fontFamily: 'var(--font-sans)', fontSize: '0.68rem', color: 'var(--color-text-secondary)' }}>
              {MODALITY_NAMES.map((name, i) => (
                <span key={name} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span style={{ width: 9, height: 9, borderRadius: '50%', background: blobColor[i], display: 'inline-block' }} />{name}
                </span>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginTop: '0.3rem' }}>
              A toy index, colored by modality, with {DELETED.size} tombstoned nodes (×). They still route, but the query (◆)
              keeps only its live nearest neighbor (<strong style={{ color: '#3c9a5f' }}>green</strong>).
            </div>
          </div>
          <div>
            <OverfetchPlot rate={rate} />
            <Slider label="pass-rate r" value={rate} min={0.1} max={0.95} step={0.01} onChange={setRate} fmt={(v) => v.toFixed(2)} />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', marginBottom: '0.5rem' }}>
              <Readout label="predicted scan k/r" value={(TOPK / rate).toFixed(1)} accent />
              <Readout label="if tombstoning" value={`δ = ${(1 - rate).toFixed(2)}`} />
              <Readout label="if filtering" value={`s = ${rate.toFixed(2)}`} />
            </div>
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              One hyperbola, two stories. To return k = {TOPK} results you scan k/r candidates, where r is the fraction that
              survive: tombstone live-rate 1−δ (<span style={{ color: REG_COLOR }}>blue</span>) or predicate selectivity s
              (<span style={{ color: POST_COLOR }}>purple</span>). Both measured sets sit on the same exact law — a tombstone is a
              persistent global predicate; a predicate is a per-query tombstone. The blow-up as r → 0 is the cost of selectivity.
            </div>
          </div>
        </div>
      )}

      {panel === 'percolation' && (
        <>
          <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.7rem' }}>
            <button onClick={() => setBView('sweep')} style={pill(bView === 'sweep')}>vs deletion</button>
            <button onClick={() => setBView('degree')} style={pill(bView === 'degree')}>vs degree M</button>
          </div>
          {bView === 'sweep' ? <PercolationPlot p={pVal} /> : <ConnectivityVsMPlot sel={mSel} />}
          {bView === 'sweep' ? (
            <>
              <Slider label="retention p" value={pVal} min={0} max={0.48} step={0.04} onChange={setPVal} fmt={(v) => v.toFixed(2)} />
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '0.5rem' }}>
                <Readout label={`regular giant @ p=${PERC_P[pNearIdx].toFixed(2)}`} value={`${((PERCOLATION_REGULAR[pNearIdx] ?? 0) * 100).toFixed(1)}%`} />
                <Readout label="HNSW giant" value={`${((PERCOLATION_HNSW[pNearIdx] ?? 0) * 100).toFixed(1)}%`} accent />
                <Readout label="regular p_c = 1/(M−1)" value={REG_PC.toFixed(3)} />
              </div>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
                Retain a fraction p of nodes at random. On a near-regular graph (deg {REG_DEG}) the giant component survives only
                above p_c = 1/(M−1) = {REG_PC.toFixed(3)} — the exact Molloy–Reed / Cohen threshold. HNSW's real layer 0 (κ = {HNSW_KAPPA},
                ⟨k⟩ = {HNSW_MEAN_DEG}) only <em>approximates</em> it. And connectivity ≠ navigability: recall fails far <em>inside</em>
                the connected regime, so this threshold is the floor, never the binding constraint.
              </div>
            </>
          ) : (
            <>
              <div style={{ display: 'flex', gap: '0.4rem', margin: '0.6rem 0' }}>
                {CONNECTIVITY_VS_M.map((d, i) => (<button key={d.M} onClick={() => setMSel(i)} style={pill(mSel === i)}>M = {d.M}</button>))}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '0.5rem' }}>
                <Readout label={`giant @ M=${CONNECTIVITY_VS_M[mSel]?.M ?? '—'}`} value={`${((CONNECTIVITY_VS_M[mSel]?.giant_frac ?? 0) * 100).toFixed(1)}%`} accent />
                <Readout label="mean degree ⟨k⟩" value={`${CONNECTIVITY_VS_M[mSel]?.mean_deg ?? '—'}`} />
              </div>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
                A denser graph survives more deletion: at a fixed, punishing retention p = 0.15 the giant fraction rises
                monotonically with the degree M — the lever an operator pulls to buy churn robustness (the principle behind
                building filtered indexes at higher M).
              </div>
            </>
          )}
        </>
      )}

      {panel === 'predicate' && (
        <>
          <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.7rem', flexWrap: 'wrap' }}>
            <button onClick={() => setCView('recall')} style={pill(cView === 'recall')}>recall vs s</button>
            <button onClick={() => setCView('cost')} style={pill(cView === 'cost')}>cost vs s</button>
            <button onClick={() => setCView('subgraph')} style={pill(cView === 'subgraph')}>predicate subgraph</button>
          </div>
          {cView === 'subgraph' ? <SubgraphPlot /> : <StrategyPlot metric={cView === 'recall' ? 'recall' : 'ndist'} sIdx={idx} />}
          {cView !== 'subgraph' ? (
            <>
              <Slider label="selectivity index" value={idx} min={0} max={sStrat.length - 1} step={1} onChange={setSIdx}
                fmt={(v) => `s = ${(sStrat[Math.round(v)] ?? sStrat[0]).toFixed(3)}`} />
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '0.5rem' }}>
                <Readout label="pre-filter" value={`${(FRONTIER_PRE[idx].recall * 100).toFixed(0)}% @ ${FRONTIER_PRE[idx].ndist.toFixed(0)} comps`} />
                <Readout label="post-filter" value={`${(FRONTIER_POST[idx].recall * 100).toFixed(0)}% @ ${FRONTIER_POST[idx].ndist.toFixed(0)} comps`} />
                <Readout label="in-filter" value={`${(FRONTIER_IN[idx].recall * 100).toFixed(0)}% @ ${FRONTIER_IN[idx].ndist.toFixed(0)} comps`} accent />
              </div>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
                No universal winner — selectivity picks the strategy. <strong style={{ color: PRE_COLOR }}>Pre-filter</strong>
                {' '}is exact and cheapest at <em>low</em> s (a tiny subset to scan). <strong style={{ color: POST_COLOR }}>Post-filter</strong>
                {' '}has constant cost but falls off the recall cliff once s &lt; k/F. <strong style={{ color: 'var(--color-accent)' }}>In-filter</strong>
                {' '}stays exact, but at very low s it must traverse almost the whole graph to collect k passing results — so it
                degenerates toward a full scan. This is exactly why production systems pre-filter below a selectivity threshold or
                build denser, predicate-aware graphs.
              </div>
            </>
          ) : (
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
              The percolation link, and the direction the simulation reveals (not the one intuition guesses). The induced subgraph
              on a <strong style={{ color: POST_COLOR }}>random</strong> passing set is site percolation — it fragments as s falls
              below p_c. A spatially <strong style={{ color: PRE_COLOR }}>coherent</strong> modality predicate (a contiguous cluster
              of the embedding space) stays connected far below that. So predicate hardness depends on <em>spatial coherence</em>,
              not selectivity alone.
            </div>
          )}
        </>
      )}
    </div>
  );
}
