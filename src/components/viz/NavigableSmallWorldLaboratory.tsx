import { memo, useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * Navigable Small-World Laboratory — three panels for the `navigable-small-world-graphs` topic:
 *   A. Kleinberg navigability. The U-curve of mean greedy-routing hops on a ring as the long-range
 *      link exponent α varies; the trough sits at the ring's dimension, α = 1. An α slider walks the
 *      curve and shows how routing collapses when the link law is too uniform or too local.
 *   B. The greedy walk. A small NSW graph: pure greedy hill-climbing (ef = 1) from the entry node
 *      gets STUCK at a local minimum that is not the true nearest neighbor; switching to a beam
 *      (ef = 16) finds it. The honest catch of graph search made visible.
 *   C. The recall / work frontier. recall@10 vs the fraction of the database touched, swept over the
 *      beam width ef: recall climbs to 1 while the work stays a small fraction of n.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): KLEINBERG, TOY_POINTS, EDGES, ENTRY, QUERY, TRUE_NN,
 * GREEDY_WALK, BEAM_FOUND and FRONTIER are mirrored TO THE DECIMAL from
 * notebooks/navigable-small-world-graphs/navigable_small_world_graphs.py (kleinberg_curve /
 * toy_nsw_graph / greedy_walk / greedy_search / recall_vs_ef, printed by viz_constants()). The lab
 * recomputes only the graph layout in TS; every hop count, recall, and the walk are baked.
 * test_kleinberg_navigability / test_greedy_local_minimum / test_recall_monotone_in_ef assert them.
 * Change a number here -> change it there, and re-run the notebook. SVG text inherits the theme color.
 */

type Pt2 = [number, number];

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: kleinberg_curve on the ring (n=20000). Mean greedy hops vs the long-range exponent α.
const KLEINBERG = [
  { alpha: 0, hops: 122.865 }, { alpha: 0.25, hops: 90.853 }, { alpha: 0.5, hops: 66.183 },
  { alpha: 0.75, hops: 52.657 }, { alpha: 1.0, hops: 55.432 }, { alpha: 1.25, hops: 97.88 },
  { alpha: 1.5, hops: 268.967 }, { alpha: 2.0, hops: 1561.598 },
];
const RING_DIM = 1; // the ring's dimension — the optimal exponent α = d

// Panel B: toy_nsw_graph — 45 points, the NSW edges, and the greedy walk that gets stuck.
const TOY_POINTS: Pt2[] = [
  [2.17, 7.53], [1.628, 5.803], [3.62, 9.03], [1.707, 8.696], [2.253, 7.502], [2.88, 7.72],
  [1.704, 7.287], [2.409, 7.911], [2.491, 7.454], [8.114, 7.197], [8.757, 8.169], [8.298, 8.369],
  [7.09, 8.705], [9.851, 6.525], [6.444, 6.646], [8.757, 8.116], [8.971, 8.65], [8.19, 8.256],
  [1.847, 2.782], [0.983, 1.62], [2.219, 3.621], [1.312, 1.029], [1.493, 2.872], [1.788, 3.192],
  [0.315, 3.016], [2.931, 0.723], [2.138, 3.094], [8.079, 2.9], [10.0, 2.247], [7.748, 1.306],
  [8.583, 1.823], [7.839, 1.905], [8.585, 1.04], [6.623, 0.0], [9.079, 2.066], [9.359, 1.992],
  [4.332, 5.43], [4.931, 3.871], [4.203, 6.59], [5.319, 5.375], [4.751, 4.379], [5.802, 4.906],
  [4.317, 4.879], [4.185, 5.171], [6.016, 4.248],
];
const EDGES: [number, number][] = [
  [0, 1], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [1, 4], [1, 6], [2, 3], [2, 5], [2, 7], [3, 5],
  [3, 6], [3, 7], [4, 6], [4, 7], [4, 8], [5, 6], [5, 7], [5, 9], [5, 12], [5, 23], [5, 27], [5, 28],
  [5, 33], [5, 38], [5, 42], [6, 7], [6, 23], [6, 26], [6, 27], [6, 42], [7, 8], [7, 23], [9, 10],
  [9, 12], [9, 13], [9, 14], [9, 15], [9, 16], [9, 27], [9, 41], [9, 42], [10, 11], [10, 12],
  [10, 13], [10, 15], [10, 16], [10, 17], [10, 41], [11, 15], [11, 16], [11, 17], [12, 14], [12, 16],
  [12, 42], [13, 15], [14, 41], [15, 16], [15, 17], [18, 19], [18, 21], [18, 22], [18, 23], [18, 26],
  [19, 21], [19, 22], [19, 24], [20, 22], [20, 23], [20, 24], [20, 26], [21, 25], [22, 23], [22, 25],
  [22, 26], [23, 24], [23, 25], [23, 26], [23, 27], [23, 33], [23, 42], [24, 26], [24, 42], [25, 26],
  [26, 37], [26, 42], [27, 28], [27, 29], [27, 31], [27, 32], [27, 33], [27, 34], [27, 41], [28, 32],
  [28, 33], [28, 34], [28, 35], [29, 31], [29, 32], [29, 34], [30, 31], [30, 32], [30, 34], [30, 35],
  [31, 32], [32, 34], [33, 34], [34, 35], [36, 39], [36, 42], [36, 43], [37, 40], [37, 41], [37, 42],
  [37, 43], [37, 44], [38, 42], [38, 43], [39, 40], [39, 41], [39, 42], [40, 42], [40, 43], [41, 42],
  [41, 43], [41, 44], [42, 43], [42, 44],
];
const ENTRY = 28;
const QUERY: Pt2 = [5.441, 6.895];
const TRUE_NN = 14;
const GREEDY_WALK = [28, 5, 38]; // ef=1 hill-climb, terminates at 38 (a local minimum, not the NN)
const BEAM_FOUND = 14; // ef=16 beam search finds the true NN

// Panel C: recall_vs_ef on the finance-like cloud (n=500, M=8).
const NSW_N = 500;
const FRONTIER = [
  { ef: 1, recall: 0.1, work: 71.5 }, { ef: 2, recall: 0.2, work: 77.1 }, { ef: 4, recall: 0.4, work: 86.4 },
  { ef: 8, recall: 0.8, work: 103.3 }, { ef: 16, recall: 0.9975, work: 134.3 }, { ef: 32, recall: 1.0, work: 183.9 },
  { ef: 64, recall: 1.0, work: 260.0 },
];

const PW = 360, PH = 320, PAD = 24;
const wx = (x: number) => PAD + (PW - 2 * PAD) * (x / 10);
const wy = (y: number) => PH - PAD - (PH - 2 * PAD) * (y / 10);
const blobColor = ['var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', '#5fa873', '#6a8caf', '#b07cc6'];

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

// Panel A — the Kleinberg U-curve (log-y, baked), with the selected α highlighted.
const KleinbergPlot = memo(function KleinbergPlot({ sel }: { sel: number }) {
  const pw = 560, ph = 250, pad = 46;
  const ax = (a: number) => pad + (pw - 2 * pad) * (a / 2);
  const hy = (h: number) => pad + (ph - 2 * pad) * (1 - (Math.log10(h) - Math.log10(40)) / (Math.log10(2000) - Math.log10(40)));
  const path = KLEINBERG.map((d, i) => (i === 0 ? 'M' : 'L') + ax(d.alpha).toFixed(1) + ' ' + hy(d.hops).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[50, 100, 300, 1000].map((h) => (
        <text key={h} x={pad - 6} y={hy(h) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{h}</text>
      ))}
      {[0, 0.5, 1, 1.5, 2].map((a) => (
        <text key={a} x={ax(a)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{a}</text>
      ))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">long-range link exponent α</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>mean greedy hops (log)</text>
      {/* the dimension-matched optimum α = 1 */}
      <line x1={ax(RING_DIM)} y1={pad} x2={ax(RING_DIM)} y2={ph - pad} stroke="#3c9a5f" strokeWidth={1.3} strokeDasharray="5 3" />
      <text x={ax(RING_DIM) + 4} y={pad + 10} fontSize={9.5} fill="#3c9a5f" fontFamily="var(--font-sans)">α = dimension = 1 (navigable)</text>
      <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={2.6} strokeLinejoin="round" />
      {KLEINBERG.map((d, i) => (
        <circle key={i} cx={ax(d.alpha)} cy={hy(d.hops)} r={i === sel ? 6 : 3.2} fill={i === sel ? 'var(--color-accent)' : 'var(--color-accent)'} stroke={i === sel ? 'var(--color-bg)' : 'none'} strokeWidth={1.5} />
      ))}
    </svg>
  );
});

// Panel B — the NSW graph and greedy walk, drawn by D3.
function GraphPlot({ beam }: { beam: boolean }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const found = beam ? BEAM_FOUND : GREEDY_WALK[GREEDY_WALK.length - 1];
    const nnFound = found === TRUE_NN;
    const walkSet = new Set<string>();
    for (let i = 0; i < GREEDY_WALK.length - 1; i++) walkSet.add([GREEDY_WALK[i], GREEDY_WALK[i + 1]].sort((a, b) => a - b).join('-'));
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${PW} ${PH}`).attr('width', '100%');
    // edges (faint); walk edges highlighted only in greedy mode
    svg.append('g').selectAll('line').data(EDGES).join('line')
      .attr('x1', (e) => wx(TOY_POINTS[e[0]][0])).attr('y1', (e) => wy(TOY_POINTS[e[0]][1]))
      .attr('x2', (e) => wx(TOY_POINTS[e[1]][0])).attr('y2', (e) => wy(TOY_POINTS[e[1]][1]))
      .attr('stroke', (e) => (!beam && walkSet.has(e.slice().sort((a, b) => a - b).join('-')) ? 'var(--color-accent)' : 'var(--color-border)'))
      .attr('stroke-width', (e) => (!beam && walkSet.has(e.slice().sort((a, b) => a - b).join('-')) ? 2.6 : 0.7))
      .attr('stroke-opacity', (e) => (!beam && walkSet.has(e.slice().sort((a, b) => a - b).join('-')) ? 1 : 0.4));
    // points colored by blob
    svg.append('g').selectAll('circle.pt').data(TOY_POINTS).join('circle').attr('class', 'pt')
      .attr('cx', (p) => wx(p[0])).attr('cy', (p) => wy(p[1])).attr('r', 3.4)
      .attr('fill', (_, i) => blobColor[Math.floor(i / 9) % blobColor.length]).attr('opacity', 0.85);
    // entry node
    svg.append('circle').attr('cx', wx(TOY_POINTS[ENTRY][0])).attr('cy', wy(TOY_POINTS[ENTRY][1])).attr('r', 6)
      .attr('fill', 'none').attr('stroke', 'var(--color-text)').attr('stroke-width', 2);
    // greedy terminus (only in greedy mode) — the local minimum where hill-climbing stops
    if (!beam) {
      const stuck = GREEDY_WALK[GREEDY_WALK.length - 1];
      svg.append('circle').attr('cx', wx(TOY_POINTS[stuck][0])).attr('cy', wy(TOY_POINTS[stuck][1])).attr('r', 8)
        .attr('fill', 'none').attr('stroke', '#c0392b').attr('stroke-width', 2).attr('stroke-dasharray', '3 2');
    }
    // the true nearest neighbor: green ring if found, red if missed
    svg.append('circle').attr('cx', wx(TOY_POINTS[TRUE_NN][0])).attr('cy', wy(TOY_POINTS[TRUE_NN][1])).attr('r', 7)
      .attr('fill', 'none').attr('stroke', nnFound ? '#3c9a5f' : '#c0392b').attr('stroke-width', 2.5);
    // the query (diamond)
    const q = QUERY;
    svg.append('path').attr('d', `M${wx(q[0])} ${wy(q[1]) - 6} L${wx(q[0]) + 6} ${wy(q[1])} L${wx(q[0])} ${wy(q[1]) + 6} L${wx(q[0]) - 6} ${wy(q[1])} Z`)
      .attr('fill', 'var(--color-text)').attr('stroke', 'var(--color-bg)').attr('stroke-width', 1);
  }, [beam]);
  return <svg ref={ref} role="img" aria-label="NSW graph and greedy walk" />;
}

// Panel C — recall vs fraction of database touched (baked).
const FrontierPlot = memo(function FrontierPlot() {
  const pw = 560, ph = 250, pad = 44;
  const fx = (f: number) => pad + (pw - 2 * pad) * f;
  const ry = (r: number) => pad + (ph - 2 * pad) * (1 - r);
  const path = FRONTIER.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.work / NSW_N).toFixed(1) + ' ' + ry(d.recall).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((r) => (<text key={r} x={pad - 6} y={ry(r) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(r * 100).toFixed(0)}%</text>))}
      {[0, 0.25, 0.5].map((f) => (<text key={f} x={fx(f)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(f * 100).toFixed(0)}%</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">fraction of database touched</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>recall@10</text>
      <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={2.6} strokeLinejoin="round" />
      {FRONTIER.map((d) => (
        <g key={d.ef}>
          <circle cx={fx(d.work / NSW_N)} cy={ry(d.recall)} r={3.4} fill="var(--color-accent)" />
          <text x={fx(d.work / NSW_N) + 6} y={ry(d.recall) + 12} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">ef {d.ef}</text>
        </g>
      ))}
    </svg>
  );
});

export default function NavigableSmallWorldLaboratory() {
  const [panel, setPanel] = useState<'kleinberg' | 'walk' | 'frontier'>('kleinberg');
  const [alphaIdx, setAlphaIdx] = useState(4); // α = 1.0, the optimum
  const [beam, setBeam] = useState(false);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'kleinberg'
        ? 'P(\\text{link } u \\to v) \\propto r(u,v)^{-\\alpha}, \\qquad \\text{navigable} \\iff \\alpha = d'
        : panel === 'walk'
          ? '\\text{greedy: move to } \\arg\\min_{v \\in N(u)} \\lVert v - q \\rVert \\ \\text{ — until a local minimum}'
          : '\\text{recall}@k \\ \\text{non-decreasing in the beam width } \\mathrm{ef}';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const pill = (active: boolean) => ({
    fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
    border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
    background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
  });

  const sel = KLEINBERG[Math.min(alphaIdx, KLEINBERG.length - 1)] ?? KLEINBERG[0];

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        <button onClick={() => setPanel('kleinberg')} style={pill(panel === 'kleinberg')}>Kleinberg navigability</button>
        <button onClick={() => setPanel('walk')} style={pill(panel === 'walk')}>The greedy walk</button>
        <button onClick={() => setPanel('frontier')} style={pill(panel === 'frontier')}>Recall / work frontier</button>
      </div>

      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.6rem' }} />

      {panel === 'kleinberg' && (
        <>
          <KleinbergPlot sel={alphaIdx} />
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.85rem', margin: '0.5rem 0' }}>
            <span style={{ minWidth: '3.5rem' }}>α = <strong>{sel.alpha}</strong></span>
            <input type="range" min={0} max={KLEINBERG.length - 1} step={1} value={alphaIdx}
              onChange={(e) => setAlphaIdx(parseInt(e.target.value, 10))} aria-label="long-range link exponent alpha"
              style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
          </label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`mean greedy hops at α=${sel.alpha}`} value={sel.hops.toFixed(0)} accent={sel.alpha === RING_DIM} />
            <Readout label="best (α = dimension = 1)" value={`${KLEINBERG[4].hops.toFixed(0)} hops`} />
            <Readout label="too local (α = 2)" value={`${KLEINBERG[7].hops.toFixed(0)} hops (≈28×)`} />
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
            Greedy routing is fast only when the long-range links are distributed ∝ r⁻ᵈ — scale-free, matched to the dimension.
            At finite n the empirical trough sits very near α = 1 (the dimension); away from it, routing degrades to a polynomial
            number of hops.
          </div>
        </>
      )}

      {panel === 'walk' && (
        <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', alignItems: 'start' }}>
          <GraphPlot beam={beam} />
          <div>
            <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.7rem' }}>
              <button onClick={() => setBeam(false)} style={pill(!beam)}>greedy (ef = 1)</button>
              <button onClick={() => setBeam(true)} style={pill(beam)}>beam (ef = 16)</button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', marginBottom: '0.6rem' }}>
              <Readout label="search" value={beam ? 'beam (ef = 16)' : 'greedy (ef = 1)'} />
              <Readout label="true neighbor" value={(beam ? BEAM_FOUND : GREEDY_WALK[GREEDY_WALK.length - 1]) === TRUE_NN ? 'found ✓' : 'missed ✗'} accent={(beam ? BEAM_FOUND : GREEDY_WALK[GREEDY_WALK.length - 1]) === TRUE_NN} />
            </div>
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              From the entry node (○), pure greedy hill-climbing follows the highlighted walk and stops at the dashed-red node — a
              <strong> local minimum</strong> where no neighbor is closer to the query (◆), so the true nearest neighbor (ringed) is
              <strong style={{ color: '#c0392b' }}> missed</strong>. A beam of width ef = 16 keeps enough candidates to escape the
              minimum and <strong style={{ color: '#3c9a5f' }}>find</strong> it. The cross-cluster edges are the long-range hubs that
              make the graph a small world.
            </div>
          </div>
        </div>
      )}

      {panel === 'frontier' && (
        <>
          <FrontierPlot />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.5rem' }}>
            <Readout label="recall at ef = 1 (greedy)" value={`${(FRONTIER[0].recall * 100).toFixed(0)}% (touch ${(100 * FRONTIER[0].work / NSW_N).toFixed(0)}%)`} />
            <Readout label="recall at ef = 16" value={`${(FRONTIER[4].recall * 100).toFixed(1)}% (touch ${(100 * FRONTIER[4].work / NSW_N).toFixed(0)}%)`} accent />
            <Readout label="exhaustive scan" value="100% (touch 100%)" />
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
            On the synthetic cloud (n = {NSW_N}, M = 8): a beam of width 16 reaches ~99.8% recall@10 while touching about
            a quarter of the database. Recall is monotone in ef — a wider search never loses a neighbor — but every gain costs more
            distance computations. The graph index is the next index family the topic after this layers into a hierarchy.
          </div>
        </>
      )}
    </div>
  );
}
