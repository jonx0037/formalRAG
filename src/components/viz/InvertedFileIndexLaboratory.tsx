import { memo, useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * Inverted-File Index Laboratory — three panels for the `ivf-voronoi-partitioning` topic:
 *   A. The Voronoi partition + the boundary effect. A 2-D cloud split into nlist=6 k-means cells.
 *      A query probes its nprobe nearest cells (slider); their members light up as candidates. The
 *      query's TRUE nearest neighbor sits in its 3rd-nearest cell, so it is MISSED until nprobe ≥ 3
 *      — the boundary effect made visible.
 *   B. The recall / scan frontier. recall@10 against the fraction of the database scanned, swept
 *      over nprobe on the 256-d finance cloud: a few percent scanned already recovers most recall.
 *   C. IVFADC residual encoding. The coarse quantizer removes 74% of the variance, so product-
 *      quantizing the residual beats flat PQ at the same 64-bit budget.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): TOY_POINTS, TOY_CENTROIDS, TOY_LABELS, BOUNDARY_QUERY,
 * QUERY_CELL_ORDER, FRONTIER, RESIDUAL and IVFADC are mirrored TO THE DECIMAL from
 * notebooks/ivf-voronoi-partitioning/ivf_voronoi_partitioning.py (toy_ivf_cloud / coarse_quantizer /
 * find_boundary_query / recall_vs_nprobe / variance_reduction / ivfadc_recall, printed by
 * viz_constants()). The lab recomputes only the Voronoi diagram (from the baked centroids) and the
 * probed-cell / candidate membership in TS; every recall, fraction, and variance is baked.
 * test_boundary_effect / test_recall_monotone_in_nprobe / test_residual_variance_reduction /
 * test_ivfadc_beats_flat_pq_equal_bits assert them. Change a number here -> change it there, and
 * re-run the notebook. SVG text inherits the theme color via `svg text { fill }`.
 */

type Pt2 = [number, number];

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: toy_ivf_cloud (60 pts), coarse_quantizer (nlist=6), and the boundary-effect query.
const TOY_POINTS: Pt2[] = [
  [2.38, 8.904], [2.363, 6.567], [2.996, 8.491], [1.409, 8.639], [2.401, 8.324], [2.031, 8.601],
  [1.19, 7.821], [1.47, 8.659], [2.044, 7.678], [1.14, 7.717], [5.009, 8.197], [6.423, 9.607],
  [2.018, 6.422], [4.808, 8.036], [5.235, 8.739], [7.33, 7.277], [4.585, 10.0], [5.711, 9.229],
  [4.435, 6.687], [5.184, 8.62], [6.65, 6.748], [7.921, 6.461], [7.892, 7.605], [8.039, 6.943],
  [8.653, 8.48], [8.353, 6.6], [8.805, 6.948], [8.967, 6.321], [9.006, 7.478], [6.626, 7.155],
  [2.56, 2.8], [1.42, 1.282], [2.72, 1.987], [2.759, 3.335], [0.686, 2.78], [3.847, 2.173],
  [1.608, 3.327], [2.779, 3.485], [2.12, 0.87], [2.379, 2.01], [6.353, 2.213], [3.706, 0.685],
  [6.472, 2.748], [4.796, 1.999], [5.99, 2.515], [6.464, 2.282], [5.396, 1.715], [6.661, 0.0],
  [5.347, 2.036], [3.932, 2.366], [7.284, 3.949], [7.862, 3.736], [9.341, 3.421], [7.037, 1.334],
  [9.929, 2.878], [7.243, 3.159], [7.789, 3.937], [8.037, 3.015], [7.214, 3.517], [6.863, 3.732],
];
const TOY_CENTROIDS: Pt2[] = [
  [7.53, 3.071], [1.949, 7.984], [2.114, 2.431], [8.022, 7.092], [5.174, 8.639], [4.959, 1.686],
];
const TOY_LABELS = [
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 4, 4, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 0, 5, 0, 5, 5, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const BOUNDARY_QUERY: Pt2 = [4.239, 5.117];
const TRUE_NN_INDEX = 18;
const NN_CELL = TOY_LABELS[TRUE_NN_INDEX]; // = 4
const QUERY_CELL_ORDER = [2, 5, 4, 1, 0, 3]; // the query's cells, nearest first
const NPROBE_TO_FIND_NN = 3;
const TOY_NLIST = TOY_CENTROIDS.length;

// Panel B: recall_vs_nprobe on the finance cloud (n=800, nlist=32).
const FRONTIER = [
  { nprobe: 1, recall: 0.475, frac: 0.0368 },
  { nprobe: 2, recall: 0.8458, frac: 0.0675 },
  { nprobe: 4, recall: 0.9008, frac: 0.137 },
  { nprobe: 8, recall: 0.9808, frac: 0.267 },
  { nprobe: 16, recall: 1.0, frac: 0.507 },
  { nprobe: 32, recall: 1.0, frac: 1.0 },
];

// Panel C: residual variance reduction and IVFADC vs flat PQ at equal bits.
const RESIDUAL = { rawVar: 21.5588, resVar: 5.6897, removed: 0.7361 };
const IVFADC = { bits: 64, flatRecall: 0.7525, ivfadcRecall: 0.7708 };

const COLORS = ['var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', '#5fa873', '#6a8caf', '#b07cc6', '#c2603f'];
const PW = 360, PH = 320, PAD = 24;
const wx = (x: number) => PAD + (PW - 2 * PAD) * (x / 10);
const wy = (y: number) => PH - PAD - (PH - 2 * PAD) * (y / 10);

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

// Panel A — the Voronoi partition, drawn by D3, with the probed cells / candidates / NN highlighted.
function VoronoiPlot({ nprobe }: { nprobe: number }) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const probed = new Set(QUERY_CELL_ORDER.slice(0, nprobe));
    const found = probed.has(NN_CELL);
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${PW} ${PH}`).attr('width', '100%');
    const del = d3.Delaunay.from(TOY_CENTROIDS, (c) => wx(c[0]), (c) => wy(c[1]));
    const vor = del.voronoi([PAD, PAD, PW - PAD, PH - PAD]);
    // cells: probed ones tinted by their color, others faint
    svg.append('g').selectAll('path').data(TOY_CENTROIDS).join('path')
      .attr('d', (_, i) => vor.renderCell(i))
      .attr('fill', (_, i) => (probed.has(i) ? COLORS[i % COLORS.length] : 'var(--color-muted-bg)'))
      .attr('fill-opacity', (_, i) => (probed.has(i) ? 0.18 : 0.04))
      .attr('stroke', 'var(--color-border)').attr('stroke-width', 1);
    // database points: candidates (in a probed cell) bright, others dim
    svg.append('g').selectAll('circle.pt').data(TOY_POINTS).join('circle').attr('class', 'pt')
      .attr('cx', (p) => wx(p[0])).attr('cy', (p) => wy(p[1])).attr('r', 3.2)
      .attr('fill', (_, i) => COLORS[TOY_LABELS[i] % COLORS.length])
      .attr('opacity', (_, i) => (probed.has(TOY_LABELS[i]) ? 1 : 0.22));
    // centroid markers
    svg.append('g').selectAll('path.cw').data(TOY_CENTROIDS).join('path').attr('class', 'cw')
      .attr('d', (c) => `M${wx(c[0]) - 4} ${wy(c[1])} h8 M${wx(c[0])} ${wy(c[1]) - 4} v8`)
      .attr('stroke', (_, i) => COLORS[i % COLORS.length]).attr('stroke-width', 2.5);
    // the true nearest neighbor: ringed green if found, red if missed
    const nn = TOY_POINTS[TRUE_NN_INDEX];
    svg.append('circle').attr('cx', wx(nn[0])).attr('cy', wy(nn[1])).attr('r', 7)
      .attr('fill', 'none').attr('stroke', found ? '#3c9a5f' : '#c0392b').attr('stroke-width', 2.5);
    // the query marker (a diamond)
    const q = BOUNDARY_QUERY;
    svg.append('path')
      .attr('d', `M${wx(q[0])} ${wy(q[1]) - 6} L${wx(q[0]) + 6} ${wy(q[1])} L${wx(q[0])} ${wy(q[1]) + 6} L${wx(q[0]) - 6} ${wy(q[1])} Z`)
      .attr('fill', 'var(--color-text)').attr('stroke', 'var(--color-bg)').attr('stroke-width', 1);
  }, [nprobe]);
  return <svg ref={ref} role="img" aria-label="Voronoi partition and boundary effect" />;
}

// Panel B — recall vs fraction scanned (baked, pure SVG).
const FrontierPlot = memo(function FrontierPlot() {
  const pw = 560, ph = 250, pad = 44;
  const fx = (f: number) => pad + (pw - 2 * pad) * f;
  const ry = (r: number) => pad + (ph - 2 * pad) * (1 - r);
  const path = FRONTIER.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.frac).toFixed(1) + ' ' + ry(d.recall).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((r) => (
        <text key={r} x={pad - 6} y={ry(r) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(r * 100).toFixed(0)}%</text>
      ))}
      {[0, 0.5, 1].map((f) => (
        <text key={f} x={fx(f)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(f * 100).toFixed(0)}%</text>
      ))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">fraction of database scanned</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>recall@10</text>
      <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={2.6} strokeLinejoin="round" />
      {FRONTIER.map((d) => (
        <g key={d.nprobe}>
          <circle cx={fx(d.frac)} cy={ry(d.recall)} r={3.4} fill="var(--color-accent)" />
          <text x={fx(d.frac) + (d.frac > 0.8 ? -6 : 6)} y={ry(d.recall) + 12} textAnchor={d.frac > 0.8 ? 'end' : 'start'} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">nprobe {d.nprobe}</text>
        </g>
      ))}
    </svg>
  );
});

export default function InvertedFileIndexLaboratory() {
  const [panel, setPanel] = useState<'partition' | 'frontier' | 'ivfadc'>('partition');
  const [nprobe, setNprobe] = useState(1);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'partition'
        ? '\\text{scan} \\approx \\frac{n_{\\text{probe}}}{n_{\\text{list}}}\\, n, \\qquad n_{\\text{list}} \\approx \\sqrt{n}'
        : panel === 'frontier'
          ? '\\text{recall}@k \\ \\text{is non-decreasing in}\\ n_{\\text{probe}}'
          : 'r = x - c_{i(x)}, \\qquad \\operatorname{Var}(r) < \\operatorname{Var}(x)';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const pill = (active: boolean) => ({
    fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
    border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
    background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
  });
  const bar = (v: number, max: number, color: string) => (
    <span style={{ flex: 1, height: '0.8rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
      <span style={{ display: 'block', height: '100%', width: `${Math.min(100, (v / max) * 100)}%`, background: color }} />
    </span>
  );

  const probed = QUERY_CELL_ORDER.slice(0, nprobe);
  const found = probed.includes(NN_CELL);
  const candidates = TOY_LABELS.filter((l) => probed.includes(l)).length;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        <button onClick={() => setPanel('partition')} style={pill(panel === 'partition')}>Voronoi partition</button>
        <button onClick={() => setPanel('frontier')} style={pill(panel === 'frontier')}>Recall / scan frontier</button>
        <button onClick={() => setPanel('ivfadc')} style={pill(panel === 'ivfadc')}>IVFADC residual</button>
      </div>

      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.6rem' }} />

      {panel === 'partition' && (
        <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', alignItems: 'start' }}>
          <VoronoiPlot nprobe={nprobe} />
          <div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.85rem', marginBottom: '0.7rem' }}>
              <span style={{ minWidth: '5rem' }}>nprobe = <strong>{nprobe}</strong></span>
              <input type="range" min={1} max={TOY_NLIST} step={1} value={nprobe}
                onChange={(e) => setNprobe(parseInt(e.target.value, 10))}
                aria-label="number of cells probed"
                style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', marginBottom: '0.6rem' }}>
              <Readout label="cells probed" value={`${nprobe} of ${TOY_NLIST}`} />
              <Readout label="candidates scanned" value={`${candidates} of ${TOY_POINTS.length}`} />
              <Readout label="true neighbor" value={found ? 'found ✓' : 'missed ✗'} accent={found} />
            </div>
            <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              The query (◆) is compared only against the points in its <strong>{nprobe}</strong> nearest cell{nprobe > 1 ? 's' : ''}.
              Its true nearest neighbor (ringed) lives in the query's <strong>{NPROBE_TO_FIND_NN}rd</strong>-nearest cell, so it is
              <strong style={{ color: found ? '#3c9a5f' : '#c0392b' }}>{found ? ' found' : ' missed'}</strong> at this nprobe —
              the boundary effect. Raise nprobe to {NPROBE_TO_FIND_NN} and the ring turns green.
            </div>
          </div>
        </div>
      )}

      {panel === 'frontier' && (
        <>
          <FrontierPlot />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.5rem' }}>
            <Readout label="recall at nprobe=1" value={`${(FRONTIER[0].recall * 100).toFixed(1)}% (scan ${(FRONTIER[0].frac * 100).toFixed(1)}%)`} />
            <Readout label="recall at nprobe=8" value={`${(FRONTIER[3].recall * 100).toFixed(1)}% (scan ${(FRONTIER[3].frac * 100).toFixed(1)}%)`} accent />
            <Readout label="exhaustive (nprobe=nlist)" value="100% (scan 100%)" />
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
            On the 256-d finance cloud (nlist=32): scanning under a third of the database (nprobe=8) already recovers ~98% of
            recall@10. The √n speedup is a balanced-cell heuristic — k-means cells are imbalanced, so list lengths and query time vary.
          </div>
        </>
      )}

      {panel === 'ivfadc' && (
        <>
          <div style={{ display: 'grid', gap: '1.25rem', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
            <div>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', color: 'var(--color-text-secondary)', marginBottom: '0.35rem' }}>total variance — the coarse quantizer removes {(RESIDUAL.removed * 100).toFixed(0)}%</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><span style={{ minWidth: '5.5rem' }}>raw x</span>{bar(RESIDUAL.rawVar, RESIDUAL.rawVar, 'var(--color-text-secondary)')}<span style={{ minWidth: '2.6rem', textAlign: 'right' }}>{RESIDUAL.rawVar.toFixed(1)}</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><span style={{ minWidth: '5.5rem' }}>residual r</span>{bar(RESIDUAL.resVar, RESIDUAL.rawVar, 'var(--color-accent)')}<span style={{ minWidth: '2.6rem', textAlign: 'right' }}>{RESIDUAL.resVar.toFixed(1)}</span></div>
              </div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', color: 'var(--color-text-secondary)', marginBottom: '0.35rem' }}>recall@10 at {IVFADC.bits}-bit codes</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><span style={{ minWidth: '5.5rem' }}>flat PQ</span>{bar(IVFADC.flatRecall, 1, 'var(--color-text-secondary)')}<span style={{ minWidth: '2.6rem', textAlign: 'right' }}>{(IVFADC.flatRecall * 100).toFixed(1)}%</span></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><span style={{ minWidth: '5.5rem' }}>IVFADC</span>{bar(IVFADC.ivfadcRecall, 1, 'var(--color-accent)')}<span style={{ minWidth: '2.6rem', textAlign: 'right' }}>{(IVFADC.ivfadcRecall * 100).toFixed(1)}%</span></div>
              </div>
            </div>
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.75rem 0 0' }}>
            Encoding the residual <em>r = x − c<sub>i(x)</sub></em> instead of the raw vector spends the same {IVFADC.bits} bits on a
            signal with {(RESIDUAL.removed * 100).toFixed(0)}% less variance, so IVFADC reaches higher recall than flat PQ at equal
            bits. This is <code>IndexIVFPQ</code> — the coarse partition and the product quantizer composed.
          </div>
        </>
      )}
    </div>
  );
}
