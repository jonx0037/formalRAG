import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * GraphRAG Community Laboratory — three panels for the `graphrag-community-detection` topic:
 *   A. The entity graph + the resolution parameter γ. Nodes are companies, edges are co-occurrence,
 *      sectors are the planted communities. A γ slider sweeps the resolution: low γ MERGES two
 *      sectors (4 communities), a 0.3–4.0 plateau recovers the 5 planted sectors, high γ FRAGMENTS
 *      toward singletons (10 → 25). Modularity Q is shown against the planted and random baselines.
 *   B. The stochastic block model detectability phase diagram. The Kesten–Stigum parabola
 *      (c_in−c_out)² = 2(c_in+c_out) (drawn closed-form) divides the (c_in, c_out) plane into a
 *      detectable region and an undetectable one; the baked overlap heatmap rides the boundary. An
 *      operating point toggles above/below, with an inset SBM whose blocks are recoverable above the
 *      line and at chance below it — an information-theoretic limit, not an algorithmic one.
 *   C. The resolution limit and Louvain vs Leiden. A large ring of cliques: the modularity-optimal
 *      partition MERGES adjacent cliques (pairs beat singles) once the cliques fall below √(2m),
 *      while a small ring keeps them separate. And a disconnected community that is a Louvain LOCAL
 *      OPTIMUM (local moving cannot repair it) which Leiden's refinement splits into connected pieces.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every modularity / overlap / label / Q number below is
 * mirrored TO THE DECIMAL from notebooks/graphrag-community-detection/graphrag_community_detection.py
 * (viz_constants()). Matching asserts: test_planted_modularity_beats_random / test_louvain_recovers_
 * planted_sectors / test_sbm_detectable_above_threshold / test_sbm_undetectable_below_threshold /
 * test_threshold_sign_matches_grid / test_resolution_limit_merges_cliques /
 * test_leiden_communities_connected_louvain_not. TS recomputes ONLY closed forms: the KS parabola, the
 * SNR readout (c_in−c_out)²/[2(c_in+c_out)], √(2m), and pixel/scale maps.
 */

// --- baked from viz_constants() -------------------------------------------------------
// Panel A — the finance entity graph (25 companies, 5 planted sectors)
const GR_N_NODES = 25;
const GR_EDGES: [number, number, number][] = [
  [0, 1, 0.637], [0, 2, 0.624], [0, 3, 0.541], [0, 4, 0.524], [1, 2, 0.65], [1, 3, 0.567], [1, 4, 0.572],
  [2, 3, 0.531], [2, 4, 0.494], [2, 24, 0.003], [3, 4, 0.489], [4, 9, 0.011], [4, 10, 0.018], [5, 6, 0.618],
  [5, 7, 0.61], [5, 8, 0.465], [5, 9, 0.598], [6, 7, 0.615], [6, 8, 0.518], [6, 9, 0.489], [7, 8, 0.57],
  [7, 9, 0.604], [7, 13, 0.007], [8, 9, 0.505], [8, 13, 0.001], [9, 23, 0.036], [9, 24, 0.047], [10, 11, 0.521],
  [10, 12, 0.498], [10, 13, 0.53], [10, 14, 0.539], [11, 12, 0.518], [11, 13, 0.562], [11, 14, 0.52],
  [12, 13, 0.593], [12, 14, 0.51], [13, 14, 0.603], [15, 16, 0.66], [15, 17, 0.645], [15, 18, 0.612],
  [15, 19, 0.552], [15, 20, 0.093], [15, 22, 0.047], [15, 23, 0.04], [15, 24, 0.006], [16, 17, 0.599],
  [16, 18, 0.552], [16, 19, 0.536], [16, 23, 0.028], [17, 18, 0.635], [17, 19, 0.587], [17, 20, 0.085],
  [17, 22, 0.076], [17, 23, 0.031], [17, 24, 0.026], [18, 19, 0.589], [18, 20, 0.03], [19, 22, 0.024],
  [20, 21, 0.583], [20, 22, 0.606], [20, 23, 0.583], [20, 24, 0.6], [21, 22, 0.574], [21, 23, 0.643],
  [21, 24, 0.588], [22, 23, 0.571], [22, 24, 0.593], [23, 24, 0.616],
];
const GR_LAYOUT: [number, number][] = [
  [0.695, 0.937], [0.673, 0.866], [0.657, 0.859], [0.7, 1.0], [0.652, 0.865], [-0.577, -0.118], [-0.608, -0.195],
  [-0.615, -0.1], [-0.723, -0.168], [-0.771, -0.214], [0.837, -0.971], [0.885, -0.954], [1.0, -0.981],
  [0.78, -1.0], [0.983, -0.963], [-1.0, -0.208], [-0.957, -0.228], [-0.791, -0.228], [-0.881, -0.127],
  [-0.925, -0.186], [-0.844, -0.169], [-0.95, -0.168], [-0.743, -0.265], [-0.78, -0.165], [-0.898, -0.051],
];
const GR_Q_PLANTED = 0.778;
const GR_Q_RANDOM = -0.043;
const GR_Q_BY_GAMMA: { gamma: number; Q: number; nComm: number; labels: number[] }[] = [
  { gamma: 0.05, Q: 0.981, nComm: 4, labels: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] },
  { gamma: 0.1, Q: 0.967, nComm: 4, labels: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] },
  { gamma: 0.3, Q: 0.919, nComm: 5, labels: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4] },
  { gamma: 1.0, Q: 0.778, nComm: 5, labels: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4] },
  { gamma: 4.0, Q: 0.176, nComm: 5, labels: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4] },
  { gamma: 6.0, Q: -0.195, nComm: 10, labels: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6, 8, 7, 8, 9] },
  { gamma: 8.0, Q: -0.322, nComm: 25, labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] },
];

// Panel B — the SBM detectability phase diagram
const SBM_C_GRID = [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0];
const SBM_OVERLAP_GRID = [
  [0.015, 0.077, 0.013, 0.116, 0.76, 0.857, 0.927],
  [0.003, 0.005, 0.011, 0.031, 0.016, 0.381, 0.659],
  [0.008, 0.032, 0.029, 0.001, 0.021, 0.004, 0.163],
  [0.36, 0.004, 0.053, 0.0, 0.017, 0.009, 0.001],
  [0.725, 0.015, 0.031, 0.004, 0.068, 0.037, 0.072],
  [0.864, 0.629, 0.023, 0.021, 0.039, 0.035, 0.047],
  [0.928, 0.765, 0.024, 0.061, 0.007, 0.019, 0.064],
]; // [c_in index][c_out index]
const SBM_ABOVE = { cIn: 10.0, cOut: 1.0, snr: 3.682, overlap: 0.929 };
const SBM_BELOW = { cIn: 6.0, cOut: 4.0, snr: 0.2, overlap: 0.012 };
const SBM_XY: [number, number][] = [
  [1.0, 0.0], [0.988, 0.156], [0.951, 0.309], [0.891, 0.454], [0.809, 0.588], [0.707, 0.707], [0.588, 0.809],
  [0.454, 0.891], [0.309, 0.951], [0.156, 0.988], [0.0, 1.0], [-0.156, 0.988], [-0.309, 0.951], [-0.454, 0.891],
  [-0.588, 0.809], [-0.707, 0.707], [-0.809, 0.588], [-0.891, 0.454], [-0.951, 0.309], [-0.988, 0.156], [-1.0, 0.0],
  [-0.988, -0.156], [-0.951, -0.309], [-0.891, -0.454], [-0.809, -0.588], [-0.707, -0.707], [-0.588, -0.809],
  [-0.454, -0.891], [-0.309, -0.951], [-0.156, -0.988], [-0.0, -1.0], [0.156, -0.988], [0.309, -0.951],
  [0.454, -0.891], [0.588, -0.809], [0.707, -0.707], [0.809, -0.588], [0.891, -0.454], [0.951, -0.309], [0.988, -0.156],
];
const SBM_REC_ABOVE = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const SBM_REC_BELOW = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0];

// Panel C — resolution limit + Louvain-vs-Leiden
const RING_N_CLIQUES = 30;
const RING_CLIQUE_SIZE = 5;
const RING_SQRT_2M = 25.69;
const RING_Q_SINGLES = 0.876;
const RING_Q_PAIRS = 0.888;
const RING_LOUVAIN_NCOMM = 15;
const RING_SMALL_NC = 6;
const RING_SMALL_Q_SINGLES = 0.742;
const RING_SMALL_Q_PAIRS = 0.621;
const WITNESS_EDGES: [number, number][] = [
  [0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9],
];
const WITNESS_LAYOUT: [number, number][] = [
  [1.0, -1.0], [0.901, -0.886], [0.849, -0.865], [-1.0, 0.795], [-0.93, 1.0], [-0.864, 0.788],
  [-0.894, -0.891], [-0.941, -0.875], [-0.848, -0.867], [-0.819, -0.864],
];
const WITNESS_LOUVAIN_LABELS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
const WITNESS_LEIDEN_LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2];

// --- palette --------------------------------------------------------------------------
const ACCENT = 'var(--color-accent)';
const DETECT = '#5fa873';   // detectable / recovered
const FAIL = '#c25b6b';     // undetectable / disconnected
const REACH = '#6c8cd5';    // a second block color
const MUTED = '#9aa3ad';
const EDGEC = 'var(--color-border)';

// distinct, deterministic community colors (golden-angle hue; stable for any community count)
const communityColor = (label: number) => `hsl(${(label * 137.508) % 360}, 58%, 56%)`;

const fmt = (x: number, n = 3) => x.toFixed(n);
const r2 = (x: number) => Math.round(x * 100) / 100;
// closed-form Kesten–Stigum detectability margin and SNR (recomputed in TS, never baked)
const ksMargin = (cIn: number, cOut: number) => (cIn - cOut) ** 2 - 2 * (cIn + cOut);
const ksSnr = (cIn: number, cOut: number) =>
  cIn + cOut > 0 ? (cIn - cOut) ** 2 / (2 * (cIn + cOut)) : 0;

// --- shared UI atoms ------------------------------------------------------------------
function Readout({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: color ?? 'var(--color-text)' }}>{value}</div>
    </div>
  );
}
const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});
const miniPill = (active: boolean, color: string) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.72rem', padding: '0.2rem 0.55rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? color : 'var(--color-border)'}`,
  background: active ? color : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

// map a layout coordinate in [-1, 1] to an SVG box
const mapXY = (lx: number, ly: number, cx: number, cy: number, r: number): [number, number] => [cx + r * lx, cy - r * ly];

// ===== Panel A — the entity graph + γ resolution slider ============================================
function EntityGraphPanel({ gi, setGi }: { gi: number; setGi: (v: number) => void }) {
  const W = 540, H = 360, cx = 270, cy = 178, r = 150;
  const g = GR_Q_BY_GAMMA[gi];
  const labels = g.labels;
  const pos = (i: number) => mapXY(GR_LAYOUT[i][0], GR_LAYOUT[i][1], cx, cy, r);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="finance entity graph colored by detected community at the current resolution"
        style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {GR_EDGES.map(([i, j, w], k) => {
          const [x1, y1] = pos(i); const [x2, y2] = pos(j);
          const same = labels[i] === labels[j];
          return <line key={k} x1={r2(x1)} y1={r2(y1)} x2={r2(x2)} y2={r2(y2)} stroke={same ? communityColor(labels[i]) : EDGEC}
            strokeWidth={same ? 0.4 + 2.2 * w : 0.5} strokeOpacity={same ? 0.5 : 0.4} />;
        })}
        {Array.from({ length: GR_N_NODES }, (_, i) => {
          const [x, y] = pos(i);
          return <circle key={i} cx={r2(x)} cy={r2(y)} r={6.5} fill={communityColor(labels[i])} stroke="var(--color-bg)" strokeWidth={1.2} />;
        })}
      </svg>
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.5rem 0 0.4rem' }}>
        <span style={{ minWidth: '16rem' }}>resolution γ = <strong>{fmt(g.gamma, 2)}</strong> → <strong>{g.nComm}</strong> communities</span>
        <input type="range" min={0} max={GR_Q_BY_GAMMA.length - 1} step={1} value={gi}
          onChange={(e) => setGi(parseInt(e.target.value, 10))} aria-label="resolution parameter gamma"
          style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
      </label>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.3rem', flexWrap: 'wrap' }}>
        <Readout label="modularity Q (at this γ)" value={fmt(g.Q)} color={ACCENT} />
        <Readout label="communities found" value={`${g.nComm}`} color={g.nComm === 5 ? DETECT : FAIL} />
        <Readout label="Q(planted sectors), γ=1" value={fmt(GR_Q_PLANTED)} color={DETECT} />
        <Readout label="Q(random partition)" value={fmt(GR_Q_RANDOM)} color={MUTED} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The {GR_N_NODES} nodes are companies; edges are co-occurrence weights, and the{' '}
        <strong>{GR_Q_BY_GAMMA[3].nComm} sectors are the planted communities</strong>. Modularity scores a partition against
        the degree-preserving null: the planted sectors score {fmt(GR_Q_PLANTED)} against a random partition's{' '}
        {fmt(GR_Q_RANDOM)}. The resolution parameter γ tunes the scale: at <strong>low γ</strong> two sectors{' '}
        <span style={{ color: FAIL }}>merge</span> (4 communities), a broad <strong>plateau</strong> recovers the{' '}
        <span style={{ color: DETECT }}>5 sectors</span>, and at <strong>high γ</strong> the graph{' '}
        <span style={{ color: FAIL }}>fragments</span> toward singletons (10 → 25). No single γ is canonical — the
        resolution limit is intrinsic, not a knob to be tuned away.
      </p>
    </div>
  );
}

// ===== Panel B — the SBM detectability phase diagram ===============================================
function PhasePanel({ above, setAbove }: { above: boolean; setAbove: (v: boolean) => void }) {
  const W = 540, H = 330, padL = 46, padR = 16, padT = 16, padB = 44;
  const cmax = 11;
  const sx = (c: number) => padL + (W - padL - padR) * (c / cmax);
  const sy = (c: number) => H - padB - (H - padT - padB) * (c / cmax);
  // the Kesten–Stigum boundary (c_in − c_out)² = 2(c_in + c_out): for each c_in plot both c_out roots
  // c_out = c_in + 1 ± sqrt(1 + 4 c_in) (from the quadratic in c_out)
  const upper: string[] = [], lower: string[] = [];
  for (let c = 0; c <= cmax; c += 0.1) {
    const disc = Math.sqrt(1 + 4 * c);
    upper.push(`${r2(sx(c))},${r2(sy(c + 1 + disc))}`);
    const lo = c + 1 - disc;
    if (lo >= 0) lower.push(`${r2(sx(c))},${r2(sy(lo))}`);
  }
  const op = above ? SBM_ABOVE : SBM_BELOW;
  const rec = above ? SBM_REC_ABOVE : SBM_REC_BELOW;
  // inset graph box (top-right)
  const iC = [W - 92, 86], iR = 64;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', margin: '0.1rem 0 0.5rem' }}>
        <button type="button" style={miniPill(above, DETECT)} onClick={() => setAbove(true)}>above threshold (c_in=10, c_out=1)</button>
        <button type="button" style={miniPill(!above, FAIL)} onClick={() => setAbove(false)}>below threshold (c_in=6, c_out=4)</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="stochastic block model detectability phase diagram with the Kesten-Stigum parabola"
        style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* the undetectable wedge between the two parabola branches */}
        <polygon points={`${upper.join(' ')} ${lower.slice().reverse().join(' ')}`} fill={FAIL} fillOpacity={0.1} />
        {/* baked overlap heatmap cells riding the boundary */}
        {SBM_OVERLAP_GRID.map((row, i) => row.map((ov, j) => {
          const [x, y] = [sx(SBM_C_GRID[i]), sy(SBM_C_GRID[j])];
          return <circle key={`${i}-${j}`} cx={r2(x)} cy={r2(y)} r={9} fill={DETECT} fillOpacity={Math.max(0.06, ov)} stroke={EDGEC} strokeWidth={0.4} />;
        }))}
        {/* the KS parabola branches */}
        <polyline points={upper.join(' ')} fill="none" stroke={ACCENT} strokeWidth={1.6} />
        <polyline points={lower.join(' ')} fill="none" stroke={ACCENT} strokeWidth={1.6} />
        {/* operating point */}
        <circle cx={r2(sx(op.cIn))} cy={r2(sy(op.cOut))} r={6} fill={above ? DETECT : FAIL} stroke="var(--color-bg)" strokeWidth={1.5} />
        {/* axes */}
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 2.5, 5, 7.5, 10].map((c) => (
          <g key={c}>
            <text x={sx(c)} y={H - padB + 14} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{c}</text>
            <text x={padL - 6} y={sy(c) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{c}</text>
          </g>
        ))}
        <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">c_in (within-block affinity)</text>
        <text x={12} y={(padT + H - padB) / 2} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + H - padB) / 2})`}>c_out</text>
        <text x={sx(8)} y={sy(8.5)} fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">undetectable wedge</text>
        {/* inset: the sampled SBM colored by RECOVERED block */}
        <circle cx={iC[0]} cy={iC[1]} r={iR + 8} fill="var(--color-bg)" stroke={EDGEC} strokeWidth={0.6} />
        {SBM_XY.map(([lx, ly], i) => {
          const [x, y] = mapXY(lx, ly, iC[0], iC[1], iR);
          return <circle key={i} cx={r2(x)} cy={r2(y)} r={2.6} fill={rec[i] === 0 ? REACH : DETECT} />;
        })}
        <line x1={iC[0] - iR - 8} y1={iC[1]} x2={iC[0] + iR + 8} y2={iC[1]} stroke={EDGEC} strokeWidth={0.5} strokeDasharray="3 3" />
        <text x={iC[0]} y={iC[1] - iR - 13} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">recovered blocks (planted = top/bottom)</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.3rem', flexWrap: 'wrap' }}>
        <Readout label="SNR = (c_in−c_out)² / 2(c_in+c_out)" value={fmt(ksSnr(op.cIn, op.cOut), 2)} color={ksSnr(op.cIn, op.cOut) > 1 ? DETECT : FAIL} />
        <Readout label="Kesten–Stigum" value={ksMargin(op.cIn, op.cOut) > 0 ? 'detectable' : 'undetectable'} color={ksMargin(op.cIn, op.cOut) > 0 ? DETECT : FAIL} />
        <Readout label="recovery overlap (measured)" value={fmt(op.overlap)} color={op.overlap > 0.5 ? DETECT : FAIL} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The stochastic block model generates a graph with planted communities; the{' '}
        <span style={{ color: ACCENT }}>Kesten–Stigum parabola</span> (c_in−c_out)² = 2(c_in+c_out) is the{' '}
        <strong>detectability threshold</strong>. <span style={{ color: DETECT }}>Above</span> it (SNR &gt; 1) spectral
        recovery correlates with the truth — the inset's blocks line up with the planted top/bottom split. Inside the{' '}
        <span style={{ color: FAIL }}>wedge</span> (SNR &lt; 1) <em>no algorithm</em> beats a coin flip: the recovered
        colors scatter, overlap ≈ 0. The baked overlap heatmap rides the boundary — an information-theoretic limit, the
        community-detection analogue of a channel's capacity. (At finite n the transition is smeared; the headline is the
        clear above/below contrast, not the exact crossing.)
      </p>
    </div>
  );
}

// ===== Panel C — resolution limit + Louvain vs Leiden ==============================================
function HeuristicsPanel({ mode, setMode }: { mode: 'resolution' | 'connectivity'; setMode: (v: 'resolution' | 'connectivity') => void }) {
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', margin: '0.1rem 0 0.6rem' }}>
        <button type="button" style={miniPill(mode === 'resolution', ACCENT)} onClick={() => setMode('resolution')}>resolution limit (ring of cliques)</button>
        <button type="button" style={miniPill(mode === 'connectivity', ACCENT)} onClick={() => setMode('connectivity')}>Louvain vs Leiden (connectivity)</button>
      </div>
      {mode === 'resolution' ? <ResolutionView /> : <ConnectivityView />}
    </div>
  );
}

function ResolutionView() {
  const W = 540, H = 250, padT = 22, padB = 50, baseY = H - padB;
  const groups = [
    { name: `large ring (${RING_N_CLIQUES} cliques)`, singles: RING_Q_SINGLES, pairs: RING_Q_PAIRS, merges: true },
    { name: `small ring (${RING_SMALL_NC} cliques)`, singles: RING_SMALL_Q_SINGLES, pairs: RING_SMALL_Q_PAIRS, merges: false },
  ];
  const py = (q: number) => baseY - (baseY - padT) * (q / 1.0);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="ring of cliques modularity: singles versus pairs"
        style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={56} y1={baseY} x2={W - 20} y2={baseY} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((q) => (
          <text key={q} x={50} y={py(q) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{q}</text>
        ))}
        {groups.map((gr, gIdx) => {
          const gx = 120 + gIdx * 250;
          const bars = [
            { label: 'one community / clique', v: gr.singles, color: MUTED, dx: -42 },
            { label: 'merge adjacent pairs', v: gr.pairs, color: gr.merges ? FAIL : DETECT, dx: 6 },
          ];
          return (
            <g key={gIdx}>
              {bars.map((b, bi) => (
                <g key={bi}>
                  <rect x={gx + b.dx} y={py(b.v)} width={36} height={baseY - py(b.v)} fill={b.color} fillOpacity={0.82} />
                  <text x={gx + b.dx + 18} y={py(b.v) - 5} textAnchor="middle" fontSize={9.5} fill={b.color} fontFamily="var(--font-sans)" fontWeight={700}>{fmt(b.v, 3)}</text>
                </g>
              ))}
              <text x={gx} y={baseY + 16} textAnchor="middle" fontSize={9} fill="var(--color-text)" fontFamily="var(--font-sans)">{gr.name}</text>
              <text x={gx} y={baseY + 30} textAnchor="middle" fontSize={8.5} fill={gr.merges ? FAIL : DETECT} fontFamily="var(--font-sans)">
                {gr.merges ? 'optimum MERGES cliques' : 'optimum keeps separate'}
              </text>
            </g>
          );
        })}
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.3rem', flexWrap: 'wrap' }}>
        <Readout label="√(2m), the resolution scale" value={fmt(RING_SQRT_2M, 1)} color={ACCENT} />
        <Readout label="clique size (well below √(2m))" value={`${RING_CLIQUE_SIZE}`} color={FAIL} />
        <Readout label={`Louvain communities (of ${RING_N_CLIQUES})`} value={`${RING_LOUVAIN_NCOMM}`} color={FAIL} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Each clique of {RING_CLIQUE_SIZE} nodes is a genuine community, but on the{' '}
        <strong>large ring</strong> the modularity-optimal partition <span style={{ color: FAIL }}>merges adjacent
        cliques in pairs</span> ({fmt(RING_Q_PAIRS, 3)} &gt; {fmt(RING_Q_SINGLES, 3)}) — Louvain finds only{' '}
        {RING_LOUVAIN_NCOMM} communities, not {RING_N_CLIQUES}. The cause is the global √(2m) ≈ {fmt(RING_SQRT_2M, 1)}{' '}
        normalization: a clique below that scale is invisible against the whole graph. On a{' '}
        <strong>small ring</strong> the same cliques are <span style={{ color: DETECT }}>resolved</span>{' '}
        ({fmt(RING_SMALL_Q_SINGLES, 3)} &gt; {fmt(RING_SMALL_Q_PAIRS, 3)}). This is the Fortunato–Barthélemy resolution
        limit — the load-bearing caveat of modularity.
      </p>
    </div>
  );
}

function ConnectivityView() {
  const W = 540, H = 250, cyTop = 120, r = 86;
  const cxL = 150, cxR = 390;
  const pos = (i: number, cx: number) => mapXY(WITNESS_LAYOUT[i][0], WITNESS_LAYOUT[i][1], cx, cyTop, r);
  const renderGraph = (cx: number, labels: number[], title: string, broken: boolean) => (
    <g>
      {WITNESS_EDGES.map(([i, j], k) => {
        const [x1, y1] = pos(i, cx); const [x2, y2] = pos(j, cx);
        return <line key={k} x1={r2(x1)} y1={r2(y1)} x2={r2(x2)} y2={r2(y2)} stroke={EDGEC} strokeWidth={1.1} strokeOpacity={0.7} />;
      })}
      {WITNESS_LAYOUT.map((_, i) => {
        const [x, y] = pos(i, cx);
        return <circle key={i} cx={r2(x)} cy={r2(y)} r={6} fill={communityColor(labels[i])} stroke="var(--color-bg)" strokeWidth={1.1} />;
      })}
      <text x={cx} y={cyTop + r + 18} textAnchor="middle" fontSize={9.5} fill={broken ? FAIL : DETECT} fontFamily="var(--font-sans)" fontWeight={700}>{title}</text>
    </g>
  );
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="a disconnected community is a Louvain local optimum; Leiden refines it"
        style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* highlight the disconnected community on the left */}
        <text x={cxL} y={18} textAnchor="middle" fontSize={8.5} fill={FAIL} fontFamily="var(--font-sans)">community 0 = two triangles, NO edge between them ✗</text>
        {renderGraph(cxL, WITNESS_LOUVAIN_LABELS, 'Louvain: a disconnected community', true)}
        {renderGraph(cxR, WITNESS_LEIDEN_LABELS, 'Leiden: refined into connected pieces', false)}
        <text x={cxR} y={18} textAnchor="middle" fontSize={8.5} fill={DETECT} fontFamily="var(--font-sans)">every community internally connected ✓</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.3rem', flexWrap: 'wrap' }}>
        <Readout label="Louvain: community 0 connected?" value="no ✗" color={FAIL} />
        <Readout label="… is it a local optimum?" value="yes (stuck)" color={FAIL} />
        <Readout label="Leiden: all connected?" value="yes ✓" color={DETECT} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Modularity maximization is NP-hard, so Louvain and Leiden are heuristics. The left partition labels{' '}
        <strong>two triangles with no edge between them</strong> as one community — it is{' '}
        <span style={{ color: FAIL }}>disconnected</span>, yet no single-node move improves modularity, so it is a Louvain{' '}
        <strong>local optimum</strong> that local moving cannot repair. Leiden inserts a{' '}
        <span style={{ color: DETECT }}>refinement</span> phase that splits every community into internally-connected
        pieces, guaranteeing what Louvain cannot. (Run from singletons, Louvain recovers the correct partition here — it
        is not broken generically; the guarantee is what differs.)
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'graph' | 'phase' | 'heuristics';
const TEX: Record<Panel, string> = {
  graph: 'Q_\\gamma = \\frac{1}{2m}\\sum_{ij}\\left[A_{ij} - \\gamma\\,\\frac{k_i k_j}{2m}\\right]\\delta(c_i, c_j)',
  phase: '(c_{\\text{in}} - c_{\\text{out}})^2 \\;>\\; 2\\,(c_{\\text{in}} + c_{\\text{out}}) \\quad\\Longleftrightarrow\\quad \\text{detectable}',
  heuristics: '\\text{resolvable scale} \\sim \\sqrt{2m}, \\qquad B = A - \\frac{\\mathbf{k}\\mathbf{k}^\\top}{2m}, \\quad Q = \\tfrac{1}{4m}\\,\\mathbf{s}^\\top B\\,\\mathbf{s}',
};

export default memo(function GraphRAGCommunityLaboratory() {
  const [panel, setPanel] = useState<Panel>('graph');
  const [gi, setGi] = useState(3);                  // γ = 1.0 (the 5-sector plateau)
  const [above, setAbove] = useState(true);
  const [mode, setMode] = useState<'resolution' | 'connectivity'>('resolution');
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="graphrag-community-detection" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'graph')} onClick={() => setPanel('graph')}>A · entity graph + γ</button>
        <button type="button" style={pill(panel === 'phase')} onClick={() => setPanel('phase')}>B · SBM detectability</button>
        <button type="button" style={pill(panel === 'heuristics')} onClick={() => setPanel('heuristics')}>C · resolution & Louvain/Leiden</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'graph' && <EntityGraphPanel gi={gi} setGi={setGi} />}
      {panel === 'phase' && <PhasePanel above={above} setAbove={setAbove} />}
      {panel === 'heuristics' && <HeuristicsPanel mode={mode} setMode={setMode} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        Finance entity graph on the shared dense-retrieval vMF cloud (seed 7): {GR_N_NODES} companies in 5 sectors, edge
        weight = the sharpened cosine of the company directions, so the sectors are the planted communities. The SBM panel
        is the canonical two-block generative model that says <em>when</em> such structure is detectable at all; the
        ring-of-cliques and the Louvain/Leiden witness are dedicated toy graphs. Every modularity, overlap, label, and Q is
        mirrored from <code>graphrag_community_detection.py</code> (viz_constants()); the lab recomputes only the
        Kesten–Stigum parabola, the SNR, and √(2m) in closed form.
      </p>
    </div>
  );
});
