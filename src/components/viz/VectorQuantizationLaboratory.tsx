import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * Vector Quantization Laboratory — an interactive Lloyd's-algorithm sandbox for the
 * `vector-quantization-lloyd-max` topic, in two panels:
 *   A. Lloyd on a 2-D cloud. Watch the two optimality conditions alternate: ASSIGN each
 *      point to its nearest codeword (the Voronoi cells recolor) then MOVE each codeword
 *      to its cell's mean. The distortion (within-cluster SSE) readout and its sparkline
 *      fall monotonically to a fixed point. "Reseed" cycles three initializations of the
 *      SAME four-blob cloud that converge to three different optima — global (285.2),
 *      a near-miss (313.9), and a badly-stuck diagonal merge (541.1) — making the
 *      local-optimum lesson tangible. Drag a codeword to set your own start, or move k.
 *   B. Rate vs distortion on the finance codebook. Distortion falls with the rate
 *      log2(k) bits/vector, but SLOWLY — Zador's k^(-2/d) with a high effective d.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): the toy cloud (INITIAL_POINTS), the three seed
 * codebooks (SEEDS), and the finance rate-distortion grid (FINANCE_RD) are mirrored TO
 * THE DECIMAL from notebooks/vector-quantization-lloyd-max/vector_quantization_lloyd_max.py
 * (toy_cloud_2d / SEED_* / finance_rate_distortion, printed by viz_constants()). The lab
 * RECOMPUTES assign/update live in TS for interactivity — pure arithmetic with the same
 * lowest-index tie-break and furthest-point empty-cell repair — but the seed cloud, the
 * seed codebooks, and the converged distortions are baked here, never re-sampled in TS.
 * test_toy_local_optima / test_distortion_monotone / test_rate_distortion_monotone assert
 * them. Change a number here -> change it there, and re-run the notebook.
 *
 * SVG text inherits the theme color via the global `svg text { fill: var(--color-text) }`.
 */

type Pt = [number, number];

// --- baked from toy_cloud_2d() : 60 points (four corner blobs) in [0,10]^2 ------------
const INITIAL_POINTS: Pt[] = [
  [2.4006, 7.5943], [2.8123, 7.7839], [1.8715, 7.9893], [3.3432, 8.4577], [1.7370, 6.6877],
  [1.8014, 7.7331], [0.4400, 7.5250], [1.3033, 7.1142], [1.8646, 7.4470], [2.6293, 8.5340],
  [2.1972, 8.7932], [1.7678, 7.9812], [3.0228, 7.7752], [1.7052, 6.9626], [1.9338, 7.8762],
  [6.8923, 7.5327], [7.5726, 8.1327], [7.8717, 7.9843], [7.1769, 7.5963], [8.3272, 8.8947],
  [6.6927, 8.9111], [8.7767, 8.3250], [7.9116, 7.4489], [8.8664, 9.2682], [9.1413, 8.7521],
  [7.9859, 6.7333], [7.6964, 8.2252], [6.6693, 8.0161], [8.0439, 8.2568], [6.7527, 7.1706],
  [7.3509, 1.3642], [9.0915, 1.9033], [7.9632, 2.0931], [8.9668, 3.3563], [8.2067, 0.5372],
  [7.7416, 2.8469], [8.5032, 1.8057], [9.1576, 1.2437], [7.1708, 3.0480], [7.7392, 3.9019],
  [7.8508, 1.7934], [7.3979, 1.4271], [6.6779, 2.8043], [8.1649, 3.3356], [7.0963, 3.6513],
  [2.0701, 3.5595], [1.9538, 1.7116], [2.4998, 3.1252], [2.4288, 1.8316], [1.2270, 1.1788],
  [2.7021, 3.0918], [2.1686, 1.4405], [2.9984, 1.2757], [1.7295, 2.7968], [0.4999, 2.6091],
  [1.8347, 2.3874], [2.2394, 2.4617], [2.8553, 1.6933], [3.4368, 2.8809], [2.9750, 3.2319],
];

// --- baked seed codebooks (k=3) and their verified converged distortions --------------
type Seed = { name: string; note: string; centroids: Pt[]; finalD: number; iters: number };
const SEEDS: Seed[] = [
  { name: 'global optimum', note: 'best adjacent merge', finalD: 285.22, iters: 6,
    centroids: [[4.6, 5.2], [5.4, 4.8], [5.0, 5.6]] },
  { name: 'near-miss local optimum', note: 'a different adjacent merge', finalD: 313.94, iters: 8,
    centroids: [[2.0, 6.0], [3.0, 5.0], [2.5, 4.0]] },
  { name: 'stuck: diagonal merge', note: 'one cell spans two diagonal blobs', finalD: 541.05, iters: 3,
    centroids: [[5.0, 5.0], [2.3, 7.7], [7.7, 2.3]] },
];

// --- baked from finance_rate_distortion() : 256-d synthetic finance codebook ----------
const FINANCE_D0 = 17247.0;
const FINANCE_RAW_BITS = 8192; // 256 float32 dimensions
const FINANCE_RD = [
  { k: 2, bits: 1, norm: 0.7127, recall: 0.000 },
  { k: 4, bits: 2, norm: 0.4907, recall: 0.000 },
  { k: 8, bits: 3, norm: 0.3371, recall: 0.057 },
  { k: 16, bits: 4, norm: 0.2984, recall: 0.157 },
  { k: 32, bits: 5, norm: 0.2628, recall: 0.167 },
  { k: 64, bits: 6, norm: 0.2269, recall: 0.151 },
  { k: 128, bits: 7, norm: 0.1845, recall: 0.445 },
  { k: 256, bits: 8, norm: 0.1288, recall: 0.528 },
];

const CELL_COLORS = [
  'var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', '#5fa873',
  '#6a8caf', '#b06ab3', '#c98a3d', '#7d8aa5', '#5aa6a6',
];

const W = 560, H = 380, PAD = 30, TOL = 1e-9;
const wx = (x: number) => PAD + (W - 2 * PAD) * (x / 10);
const wy = (y: number) => H - PAD - (H - 2 * PAD) * (y / 10);
const pxToWx = (px: number) => ((px - PAD) / (W - 2 * PAD)) * 10;
const pxToWy = (py: number) => ((H - PAD - py) / (H - 2 * PAD)) * 10;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// --- Lloyd's two half-steps, recomputed in TS (mirrors the .py exactly) ---------------
const sqd = (a: Pt, b: Pt) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

function nearest(p: Pt, C: Pt[]): [number, number] {
  let best = 0, bd = sqd(p, C[0]);
  for (let j = 1; j < C.length; j++) {
    const d = sqd(p, C[j]);
    if (d < bd) { bd = d; best = j; } // strict < => lowest-index tie-break (np.argmin)
  }
  return [best, bd];
}

function assignAll(points: Pt[], C: Pt[]): { labels: number[]; distortion: number } {
  let distortion = 0;
  const labels = points.map((p) => {
    const [j, d] = nearest(p, C);
    distortion += d;
    return j;
  });
  return { labels, distortion };
}

function totalDistortion(points: Pt[], C: Pt[], labels: number[]): number {
  // Guard the transition frame after the point-count slider grows `points` but before
  // the reset effect refreshes `assignments`: labels[i] / its codeword can be undefined.
  return points.reduce((s, p, i) => {
    const c = C[labels[i]];
    return c ? s + sqd(p, c) : s;
  }, 0);
}

// M-step with the furthest-point empty-cell repair from the notebook.
function updateCentroids(points: Pt[], labels: number[], k: number, prev: Pt[]): Pt[] {
  const sums: Pt[] = Array.from({ length: k }, () => [0, 0]);
  const counts = new Array(k).fill(0);
  points.forEach((p, i) => {
    const j = labels[i];
    sums[j][0] += p[0]; sums[j][1] += p[1]; counts[j]++;
  });
  const C: Pt[] = sums.map((s, j) => (counts[j] > 0 ? [s[0] / counts[j], s[1] / counts[j]] : [NaN, NaN]));
  const empties = counts.map((c, j) => (c === 0 ? j : -1)).filter((j) => j >= 0);
  if (empties.length) {
    const resid = points.map((p, i) => sqd(p, prev[labels[i]]));
    const order = points.map((_, i) => i).sort((a, b) => resid[b] - resid[a]);
    const taken = new Set<number>();
    let ptr = 0;
    for (const j of empties) {
      while (ptr < order.length && taken.has(order[ptr])) ptr++;
      if (ptr < order.length) { const idx = order[ptr]; C[j] = [points[idx][0], points[idx][1]]; taken.add(idx); ptr++; }
      else C[j] = [prev[j][0], prev[j][1]];
    }
  }
  return C;
}

// Deterministic farthest-first seeding for k != 3 (the baked seeds cover only k = 3).
function farthestFirst(points: Pt[], k: number, start: number): Pt[] {
  const C: Pt[] = [points[start % points.length]];
  const d2 = points.map((p) => sqd(p, C[0]));
  while (C.length < k) {
    let bi = 0;
    for (let i = 1; i < points.length; i++) if (d2[i] > d2[bi]) bi = i;
    C.push([points[bi][0], points[bi][1]]);
    points.forEach((p, i) => { d2[i] = Math.min(d2[i], sqd(p, C[C.length - 1])); });
  }
  return C.map((c) => [c[0], c[1]] as Pt);
}

function seedCodebook(points: Pt[], k: number, seedIdx: number): Pt[] {
  if (k === 3) return SEEDS[seedIdx % SEEDS.length].centroids.map((c) => [c[0], c[1]] as Pt);
  const starts = [0, Math.floor(points.length / 3), Math.floor((2 * points.length) / 3)];
  return farthestFirst(points, k, starts[seedIdx % starts.length]);
}

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  const w = 150, h = 38, p = 3;
  if (values.length < 2) {
    return <svg viewBox={`0 0 ${w} ${h}`} style={{ width: w, height: h }} />;
  }
  const lo = Math.min(...values), hi = Math.max(...values);
  const sx = (i: number) => p + ((w - 2 * p) * i) / (values.length - 1);
  const sy = (v: number) => p + (h - 2 * p) * (hi === lo ? 0.5 : 1 - (v - lo) / (hi - lo));
  const dPath = values.map((v, i) => (i === 0 ? 'M' : 'L') + sx(i).toFixed(1) + ' ' + sy(v).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: w, height: h, display: 'block' }}>
      <path d={dPath} fill="none" stroke="var(--color-accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      {values.map((v, i) => (
        <circle key={i} cx={sx(i)} cy={sy(v)} r={1.8} fill="var(--color-accent)" />
      ))}
    </svg>
  );
}

// --- Panel B: rate–distortion line plot (baked finance grid) --------------------------
function RatePlot() {
  const pw = 540, ph = 230, pad = 34;
  const bx = (b: number) => pad + ((pw - 2 * pad) * (b - 1)) / 7;
  const vy = (v: number) => pad + (ph - 2 * pad) * (1 - v);
  const normPath = FINANCE_RD.map((r, i) => (i === 0 ? 'M' : 'L') + bx(r.bits).toFixed(1) + ' ' + vy(r.norm).toFixed(1)).join(' ');
  const recPath = FINANCE_RD.map((r, i) => (i === 0 ? 'M' : 'L') + bx(r.bits).toFixed(1) + ' ' + vy(r.recall).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (
        <text key={v} x={pad - 6} y={vy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(v * 100).toFixed(0)}%</text>
      ))}
      {FINANCE_RD.map((r) => (
        <text key={r.k} x={bx(r.bits)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.bits}</text>
      ))}
      <text x={pw / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rate = log₂(k) bits / vector</text>
      <path d={normPath} fill="none" stroke="var(--color-accent)" strokeWidth={2.4} strokeLinejoin="round" />
      <path d={recPath} fill="none" stroke="var(--color-accent-secondary, #d98a3d)" strokeWidth={2.4} strokeLinejoin="round" strokeDasharray="5 3" />
      {FINANCE_RD.map((r) => (<circle key={`n${r.k}`} cx={bx(r.bits)} cy={vy(r.norm)} r={2.6} fill="var(--color-accent)" />))}
      {FINANCE_RD.map((r) => (<circle key={`r${r.k}`} cx={bx(r.bits)} cy={vy(r.recall)} r={2.6} fill="var(--color-accent-secondary, #d98a3d)" />))}
    </svg>
  );
}

export default function VectorQuantizationLaboratory() {
  const [panel, setPanel] = useState<'lloyd' | 'rate'>('lloyd');
  const [k, setK] = useState(3);
  const [nPoints, setNPoints] = useState(INITIAL_POINTS.length);
  const [seedIdx, setSeedIdx] = useState(0);
  const points = useMemo<Pt[]>(() => INITIAL_POINTS.slice(0, nPoints), [nPoints]);

  const [centroids, setCentroids] = useState<Pt[]>(() => seedCodebook(INITIAL_POINTS, 3, 0));
  const [assignments, setAssignments] = useState<number[]>(() => assignAll(INITIAL_POINTS, seedCodebook(INITIAL_POINTS, 3, 0)).labels);
  const [distHistory, setDistHistory] = useState<number[]>(() => [assignAll(INITIAL_POINTS, seedCodebook(INITIAL_POINTS, 3, 0)).distortion]);
  const [iter, setIter] = useState(0);
  const [converged, setConverged] = useState(false);
  const [running, setRunning] = useState(false);

  const svgRef = useRef<SVGSVGElement>(null);
  const formulaRef = useRef<HTMLDivElement>(null);
  const centroidsRef = useRef(centroids);
  const assignmentsRef = useRef(assignments);
  const pointsRef = useRef(points);
  const kRef = useRef(k);
  useEffect(() => { centroidsRef.current = centroids; }, [centroids]);
  useEffect(() => { assignmentsRef.current = assignments; }, [assignments]);
  useEffect(() => { pointsRef.current = points; }, [points]);
  useEffect(() => { kRef.current = k; }, [k]);

  // Reset to a fresh seed whenever k, the point count, or the seed selection changes.
  useEffect(() => {
    const C = seedCodebook(points, k, seedIdx);
    const { labels, distortion } = assignAll(points, C);
    setCentroids(C); setAssignments(labels);
    setDistHistory([distortion]); setIter(0); setConverged(false); setRunning(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k, nPoints, seedIdx]);

  const liveDistortion = useMemo(
    () => totalDistortion(points, centroids, assignments),
    [points, centroids, assignments],
  );

  // One Lloyd iteration: move codewords to cell means, then re-assign (matches lloyd()).
  function stepOnce() {
    const C = centroidsRef.current, labels = assignmentsRef.current, pts = pointsRef.current, kk = kRef.current;
    const C_new = updateCentroids(pts, labels, kk, C);
    const shift = Math.max(...C_new.map((c, j) => Math.sqrt(sqd(c, C[j]))));
    const { labels: newLabels, distortion } = assignAll(pts, C_new);
    setCentroids(C_new); setAssignments(newLabels);
    setDistHistory((h) => [...h, distortion]); setIter((i) => i + 1);
    if (shift < TOL) { setConverged(true); setRunning(false); }
  }

  // Run loop — step on an interval slow enough to watch the descent.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => stepOnce(), 480);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  // Live KaTeX: the two optimality conditions and the distortion objective.
  useEffect(() => {
    if (!formulaRef.current || panel !== 'lloyd') return;
    katex.render(
      'Q(x)=\\arg\\min_j\\lVert x-c_j\\rVert^2 \\quad\\Rightarrow\\quad ' +
      'c_j=\\tfrac{1}{|S_j|}\\!\\sum_{x\\in S_j}\\! x \\quad\\Rightarrow\\quad ' +
      'D=\\sum_j\\sum_{x\\in S_j}\\lVert x-c_j\\rVert^2',
      formulaRef.current, { throwOnError: false, displayMode: true },
    );
  }, [panel]);

  // D3 render: Voronoi cells, points colored by assignment, draggable codewords.
  useEffect(() => {
    if (panel !== 'lloyd' || !svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');

    // bounding frame
    svg.append('rect').attr('x', PAD).attr('y', PAD).attr('width', W - 2 * PAD).attr('height', H - 2 * PAD)
      .attr('fill', 'none').attr('stroke', 'var(--color-border)').attr('stroke-width', 1);

    // Voronoi cells of the current codebook (needs >= 2 sites)
    if (centroids.length >= 2) {
      const del = d3.Delaunay.from(centroids, (c) => wx(c[0]), (c) => wy(c[1]));
      const vor = del.voronoi([PAD, PAD, W - PAD, H - PAD]);
      svg.append('g').selectAll('path').data(centroids).join('path')
        .attr('d', (_, i) => vor.renderCell(i))
        .attr('fill', (_, i) => CELL_COLORS[i % CELL_COLORS.length])
        .attr('fill-opacity', 0.1)
        .attr('stroke', 'var(--color-border)').attr('stroke-width', 1);
    }

    // points colored by current assignment
    svg.append('g').selectAll('circle.pt').data(points).join('circle').attr('class', 'pt')
      .attr('cx', (p) => wx(p[0])).attr('cy', (p) => wy(p[1])).attr('r', 3.4)
      .attr('fill', (_, i) => CELL_COLORS[(assignments[i] ?? 0) % CELL_COLORS.length]).attr('opacity', 0.85);

    // codewords (draggable), drawn as a ring + cross
    const cg = svg.append('g').selectAll('g.cw').data(centroids.map((_, i) => i)).join('g').attr('class', 'cw')
      .attr('transform', (i) => `translate(${wx(centroids[i][0])},${wy(centroids[i][1])})`)
      .style('cursor', 'grab');
    cg.append('circle').attr('r', 9).attr('fill', 'var(--color-bg)')
      .attr('stroke', (i) => CELL_COLORS[i % CELL_COLORS.length]).attr('stroke-width', 3);
    cg.append('path').attr('d', 'M-4 0 H4 M0 -4 V4').attr('stroke', (i) => CELL_COLORS[i % CELL_COLORS.length]).attr('stroke-width', 2);

    const drag = d3.drag<SVGGElement, number>()
      .container(svgRef.current)
      .on('drag', (event, i) => {
        const next = centroidsRef.current.map((c) => [c[0], c[1]] as Pt);
        next[i] = [clamp(pxToWx(event.x), 0, 10), clamp(pxToWy(event.y), 0, 10)];
        centroidsRef.current = next;
        setCentroids(next);
        setAssignments(assignAll(pointsRef.current, next).labels);
      })
      .on('end', () => {
        // Recompute from the latest points + centroids rather than the possibly-stale
        // assignmentsRef, which lags a render behind the last drag event.
        const { distortion } = assignAll(pointsRef.current, centroidsRef.current);
        setDistHistory([distortion]); setIter(0); setConverged(false);
      });
    cg.call(drag);
  }, [panel, points, centroids, assignments, k]);

  const pillStyle = (active: boolean) => ({
    fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
    border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
    background: active ? 'var(--color-accent)' : 'transparent',
    color: active ? 'var(--color-bg)' : 'var(--color-text)',
  });

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        <button onClick={() => setPanel('lloyd')} style={pillStyle(panel === 'lloyd')}>Lloyd's algorithm (2-D)</button>
        <button onClick={() => setPanel('rate')} style={pillStyle(panel === 'rate')}>Rate–distortion (finance)</button>
      </div>

      {panel === 'lloyd' && (
        <>
          <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.75rem' }} />

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
            <button onClick={stepOnce} disabled={converged} style={{ ...pillStyle(false), opacity: converged ? 0.5 : 1 }}>Step</button>
            <button onClick={() => setRunning((r) => !r)} disabled={converged} style={pillStyle(running)}>{running ? 'Pause' : 'Run'}</button>
            <button onClick={() => setSeedIdx((s) => (s + 1) % (k === 3 ? SEEDS.length : 3))} style={pillStyle(false)}>Reseed codewords</button>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginBottom: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.85rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{ minWidth: '8.5rem' }}>codewords <strong>k = {k}</strong> ({Math.ceil(Math.log2(Math.max(2, k)))} bits)</span>
              <input type="range" min={1} max={Math.min(8, nPoints)} step={1} value={k}
                onChange={(e) => setK(parseInt(e.target.value, 10))} style={{ accentColor: 'var(--color-accent)' }} />
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{ minWidth: '6rem' }}>points = {nPoints}</span>
              <input type="range" min={12} max={INITIAL_POINTS.length} step={4} value={nPoints}
                onChange={(e) => setNPoints(parseInt(e.target.value, 10))} style={{ accentColor: 'var(--color-accent)' }} />
            </label>
          </div>

          <svg ref={svgRef} role="img" aria-label="Lloyd's algorithm on a 2-D point cloud: Voronoi cells, points, and draggable codewords" />

          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Each <strong>Step</strong> runs one Lloyd iteration: re-assign points to the nearest codeword (cells recolor),
            then move each codeword to its cell's mean. Distortion falls monotonically to a fixed point. <strong>Reseed</strong>
            {' '}to see initialization decide which optimum Lloyd reaches; <strong>drag</strong> a codeword to set your own start.
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'flex-end' }}>
            <Readout label="distortion (within-cluster SSE)" value={liveDistortion.toFixed(2)} accent />
            <Readout label="iteration" value={`${iter}`} />
            <Readout label="converged?" value={converged ? 'yes — fixed point' : 'no'} accent={converged} />
            {k === 3 && <Readout label="this seed reaches" value={SEEDS[seedIdx % SEEDS.length].name} />}
            <div>
              <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>distortion history (non-increasing)</div>
              <Sparkline values={distHistory} />
            </div>
          </div>
        </>
      )}

      {panel === 'rate' && (
        <>
          <RatePlot />
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Normalized distortion D/D₀ (accent) and recall@10 retained (dashed) versus the rate log₂(k) bits per vector,
            for a codebook trained on a synthetic 256-d finance cloud. Distortion falls as the codebook grows — but
            slowly: the cloud's high effective rank is the <em>d</em> in Zador's k<sup>−2/d</sup>, so each extra bit
            buys less and less. Splitting the vector into subspaces (product quantization) is how the next topic escapes it.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="raw vector" value={`${FINANCE_RAW_BITS} bits (256 × float32)`} />
            <Readout label="codeword index at k = 256" value="8 bits → 1024× smaller" accent />
            <Readout label="distortion at 8 bits" value={`${(FINANCE_RD[7].norm * 100).toFixed(1)}% of D₀`} />
            <Readout label="recall@10 at 8 bits" value={`${(FINANCE_RD[7].recall * 100).toFixed(1)}%`} accent />
          </div>
        </>
      )}
    </div>
  );
}
