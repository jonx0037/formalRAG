import { Fragment, memo, useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * Product Quantization Laboratory — three panels for the `product-quantization` topic:
 *   A. Split into subspaces. A 4-D toy vector is two 2-D subspaces, each with its own small
 *      Voronoi codebook. Select a point: its two sub-codes light up and the two per-subspace
 *      distortion bars SUM to the total — the additive decomposition made visible.
 *   B. Scalability frontier. Recall@10 vs the rate (bits) for PQ, against the flat-VQ ceiling
 *      (the ~53% an 8-bit flat codebook reaches and cannot exceed: a flat codebook needs
 *      k=2^B ≤ n centroids). PQ reaches 64 bits at 2048 stored centroids — a budget flat VQ
 *      cannot train.
 *   C. ADC lookup table. The m×k* table of ‖q^j − c^j_i‖² for one query; a database vector's
 *      code highlights m cells whose sum IS the exact ‖q − Q(x)‖² — O(m) lookups per vector.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): TOY_POINTS_4D, SUB_A/SUB_B, ADC_TABLE, and FRONTIER are
 * mirrored TO THE DECIMAL from notebooks/product-quantization/product_quantization.py
 * (toy_pq_cloud / train_pq / adc_table / scalability_frontier, printed by viz_constants()). The
 * lab recomputes only nearest-codeword assignment and the additive sum in TS (same lowest-index
 * tie-break); the codebooks, the ADC table, and the frontier are baked, never retrained in TS.
 * test_additive_decomposition / test_adc_equals_bruteforce / test_pq_recall_beats_flat_at_pq_budget
 * assert them. Big ints (2^64) are baked as STRINGS — never k**m in JS. Change a number here ->
 * change it there, and re-run the notebook. SVG text inherits theme color via `svg text {fill}`.
 */

type Pt2 = [number, number];

// --- baked from viz_constants() -------------------------------------------------------
const TOY_POINTS_4D: number[][] = [
  [4.5719, 4.0124, 7.1916, 4.7847], [7.9110, 1.1104, 3.8590, 7.9347], [7.4949, 3.2549, 4.5178, 2.6481],
  [6.0184, 7.9544, 6.8267, 5.0501], [7.9943, 8.3004, 6.4902, 4.6936], [1.1383, 6.7390, 4.9931, 1.1088],
  [1.9981, 6.1547, 5.1804, 1.9364], [4.5003, 6.9297, 4.2886, 0.5611], [2.8783, 7.2026, 7.3078, 4.8214],
  [4.3210, 4.0184, 4.6820, 7.8583], [8.2284, 0.0000, 6.0899, 1.9701], [2.5598, 3.2862, 5.0520, 7.1078],
  [8.6546, 1.7894, 5.9884, 2.5505], [9.5953, 0.9815, 5.6402, 8.0286], [1.7392, 3.5753, 5.5500, 8.2226],
  [7.5564, 4.8028, 7.3679, 4.9087], [7.7168, 1.7718, 4.1157, 2.6173], [7.0658, 1.2452, 1.8390, 4.8560],
  [6.0307, 3.2250, 4.8773, 7.3743], [3.1683, 3.9887, 5.3679, 7.8798], [6.6322, 9.4425, 4.7379, 2.3119],
  [2.1695, 4.3106, 4.7141, 8.8334], [7.0023, 1.6542, 7.2109, 4.7154], [2.7873, 8.6862, 5.8334, 4.2153],
  [7.6852, 6.8266, 7.6521, 4.9696], [0.9576, 0.8883, 6.8301, 5.9860], [8.0781, 3.6382, 4.2304, 7.6486],
  [2.3111, 6.2645, 4.7164, 2.3518], [3.5040, 1.0275, 2.6019, 4.6319], [6.6800, 3.2142, 4.0369, 2.4376],
  [0.0000, 2.9443, 5.4837, 7.7142], [1.8311, 7.6257, 7.0980, 4.2244], [2.4129, 7.7324, 4.7169, 8.8268],
  [3.2983, 1.6279, 5.0814, 9.3862], [4.1341, 8.3350, 6.5277, 5.3482], [8.4703, 3.8396, 6.8827, 5.3395],
  [3.4057, 8.4707, 4.9957, 7.6633], [7.5869, 5.8592, 4.4794, 9.8396], [7.3447, 6.6151, 4.9536, 6.7900],
  [5.8638, 7.7972, 6.6108, 5.4068], [6.8462, 6.3157, 4.7000, 2.8163], [1.3005, 7.8087, 7.6014, 4.9086],
  [2.9125, 9.0208, 6.7167, 4.3971], [2.4840, 8.6981, 4.5800, 1.1161], [4.1126, 8.8227, 7.7226, 5.9544],
  [4.7799, 3.9130, 2.2463, 4.2910], [7.8906, 2.9873, 3.9389, 7.4217], [7.9269, 2.9402, 3.1362, 7.3146],
];
const SUB_A: Pt2[] = [[7.7535, 2.4303], [6.9965, 7.3889], [2.8245, 3.0539], [2.7291, 7.7493]];
const SUB_B: Pt2[] = [[4.7505, 7.9908], [4.8288, 2.0355], [6.9919, 4.9827], [2.2291, 4.5930]];
const SUBSPACE_DIST_TOTAL: [number, number] = [114.0577, 36.5167]; // dataset totals (A loose, B tight)

// ADC table for the fixed query = toy point 0 (m=2, k*=4): table[j][i] = ‖q^j − c^j_i‖².
const ADC_QUERY_IDX = 0;
const ADC_TABLE: number[][] = [
  [12.6254, 17.2796, 3.9720, 17.3610],
  [16.2389, 13.1412, 0.0791, 24.6639],
];

// scalability_frontier() on the 256-d finance cloud (k*=256). m=1 is the flat VQ ceiling.
const FRONTIER = [
  { m: 1, bits: 8, stored: 256, dist: 2221.25, recall: 0.5283 },
  { m: 2, bits: 16, stored: 512, dist: 2152.54, recall: 0.5558 },
  { m: 4, bits: 32, stored: 1024, dist: 2069.89, recall: 0.6367 },
  { m: 8, bits: 64, stored: 2048, dist: 1877.04, recall: 0.7525 },
];
const FLAT_CEILING = FRONTIER[0].recall; // 0.5283
const EFFECTIVE_64BIT = '2⁶⁴'; // 256^8, baked as a string — never compute in JS

const COLORS = ['var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', '#5fa873', '#6a8caf'];
const SUB_W = 250, SUB_H = 250, PAD = 22;
const wx = (x: number) => PAD + (SUB_W - 2 * PAD) * (x / 10);
const wy = (y: number) => SUB_H - PAD - (SUB_H - 2 * PAD) * (y / 10);

const subA = (p: number[]): Pt2 => [p[0], p[1]];
const subB = (p: number[]): Pt2 => [p[2], p[3]];
const sqd = (a: Pt2, b: Pt2) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
function nearest(p: Pt2, C: Pt2[]): number {
  let best = 0, bd = sqd(p, C[0]);
  for (let j = 1; j < C.length; j++) { const d = sqd(p, C[j]); if (d < bd) { bd = d; best = j; } }
  return best; // strict < => lowest-index tie-break, matches np.argmin
}

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

// One subspace's Voronoi + points + the selected marker, drawn by D3 into a ref.
function SubspacePlot({ codebook, project, selectedIdx, title }: {
  codebook: Pt2[]; project: (p: number[]) => Pt2; selectedIdx: number; title: string;
}) {
  const ref = useRef<SVGSVGElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${SUB_W} ${SUB_H}`).attr('width', '100%');
    svg.append('rect').attr('x', PAD).attr('y', PAD).attr('width', SUB_W - 2 * PAD).attr('height', SUB_H - 2 * PAD)
      .attr('fill', 'none').attr('stroke', 'var(--color-border)').attr('stroke-width', 1);
    const del = d3.Delaunay.from(codebook, (c) => wx(c[0]), (c) => wy(c[1]));
    const vor = del.voronoi([PAD, PAD, SUB_W - PAD, SUB_H - PAD]);
    svg.append('g').selectAll('path').data(codebook).join('path')
      .attr('d', (_, i) => vor.renderCell(i))
      .attr('fill', (_, i) => COLORS[i % COLORS.length]).attr('fill-opacity', 0.1)
      .attr('stroke', 'var(--color-border)').attr('stroke-width', 1);
    svg.append('g').selectAll('circle.pt').data(TOY_POINTS_4D).join('circle').attr('class', 'pt')
      .attr('cx', (p) => wx(project(p)[0])).attr('cy', (p) => wy(project(p)[1])).attr('r', 3.2)
      .attr('fill', (p) => COLORS[nearest(project(p), codebook) % COLORS.length])
      .attr('opacity', (_, i) => (i === selectedIdx ? 1 : 0.55));
    // codeword markers
    svg.append('g').selectAll('path.cw').data(codebook).join('path').attr('class', 'cw')
      .attr('d', (c) => `M${wx(c[0]) - 4} ${wy(c[1])} h8 M${wx(c[0])} ${wy(c[1]) - 4} v8`)
      .attr('stroke', (_, i) => COLORS[i % COLORS.length]).attr('stroke-width', 2.5);
    // selected point ring
    const sp = project(TOY_POINTS_4D[selectedIdx]);
    svg.append('circle').attr('cx', wx(sp[0])).attr('cy', wy(sp[1])).attr('r', 7)
      .attr('fill', 'none').attr('stroke', 'var(--color-text)').attr('stroke-width', 2);
    svg.append('text').attr('x', SUB_W / 2).attr('y', SUB_H - 4).attr('text-anchor', 'middle')
      .attr('font-size', 11).attr('fill', 'var(--color-text-secondary)').attr('font-family', 'var(--font-sans)')
      .text(title);
  }, [codebook, project, selectedIdx, title]);
  return <svg ref={ref} role="img" aria-label={`${title} Voronoi codebook`} />;
}

const FrontierPlot = memo(function FrontierPlot() {
  const pw = 540, ph = 240, pad = 38;
  const bx = (b: number) => pad + ((pw - 2 * pad) * (b - 8)) / 56;
  const vy = (v: number) => pad + (ph - 2 * pad) * (1 - v);
  const pqPath = FRONTIER.map((r, i) => (i === 0 ? 'M' : 'L') + bx(r.bits).toFixed(1) + ' ' + vy(r.recall).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (
        <text key={v} x={pad - 6} y={vy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(v * 100).toFixed(0)}%</text>
      ))}
      {FRONTIER.map((r) => (
        <text key={r.bits} x={bx(r.bits)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.bits}</text>
      ))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rate = m·log₂(k*) bits / vector</text>
      {/* flat VQ ceiling */}
      <line x1={pad} y1={vy(FLAT_CEILING)} x2={pw - pad} y2={vy(FLAT_CEILING)} stroke="var(--color-text-secondary)" strokeWidth={1.4} strokeDasharray="5 3" />
      <text x={pw - pad} y={vy(FLAT_CEILING) - 5} textAnchor="end" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">flat VQ ceiling — unreachable above 8 bits (k = 2ᴮ ≤ n)</text>
      <path d={pqPath} fill="none" stroke="var(--color-accent)" strokeWidth={2.6} strokeLinejoin="round" />
      {FRONTIER.map((r) => (<circle key={r.bits} cx={bx(r.bits)} cy={vy(r.recall)} r={3.4} fill="var(--color-accent)" />))}
    </svg>
  );
});

export default function ProductQuantizationLaboratory() {
  const [panel, setPanel] = useState<'split' | 'frontier' | 'adc'>('split');
  const [selectedIdx, setSelectedIdx] = useState(17);
  const formulaRef = useRef<HTMLDivElement>(null);

  const sel = TOY_POINTS_4D[selectedIdx];
  const codeA = nearest(subA(sel), SUB_A);
  const codeB = nearest(subB(sel), SUB_B);
  const distA = useMemo(() => sqd(subA(sel), SUB_A[codeA]), [sel, codeA]);
  const distB = useMemo(() => sqd(subB(sel), SUB_B[codeB]), [sel, codeB]);
  const total = distA + distB;
  const adc = ADC_TABLE[0][codeA] + ADC_TABLE[1][codeB];

  useEffect(() => {
    if (!formulaRef.current || panel !== 'split') return;
    katex.render(
      '\\lVert x-Q(x)\\rVert^2 = \\lVert x^1-c^1_{q_1}\\rVert^2 + \\lVert x^2-c^2_{q_2}\\rVert^2',
      formulaRef.current, { throwOnError: false, displayMode: true },
    );
  }, [panel]);

  const pill = (active: boolean) => ({
    fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
    border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
    background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
  });
  const bar = (v: number, max: number, color: string) => (
    <span style={{ flex: 1, height: '0.8rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
      <span style={{ display: 'block', height: '100%', width: `${(v / max) * 100}%`, background: color }} />
    </span>
  );
  const barMax = Math.max(distA, distB, total) * 1.02;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        <button onClick={() => setPanel('split')} style={pill(panel === 'split')}>Split into subspaces</button>
        <button onClick={() => setPanel('frontier')} style={pill(panel === 'frontier')}>Scalability frontier</button>
        <button onClick={() => setPanel('adc')} style={pill(panel === 'adc')}>ADC lookup table</button>
      </div>

      {panel === 'split' && (
        <>
          <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.5rem' }} />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem', alignItems: 'center', fontFamily: 'var(--font-sans)', fontSize: '0.85rem' }}>
            <button onClick={() => setSelectedIdx((i) => (i + 1) % TOY_POINTS_4D.length)} style={pill(false)}>Select next point</button>
            <span style={{ color: 'var(--color-text-secondary)' }}>point #{selectedIdx} → code (q₁, q₂) = ({codeA}, {codeB})</span>
          </div>
          <div style={{ display: 'grid', gap: '0.75rem', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
            <figure style={{ margin: 0 }}>
              <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', marginBottom: '0.2rem' }}>subspace 1 — dims (x₀, x₁)</figcaption>
              <SubspacePlot codebook={SUB_A} project={subA} selectedIdx={selectedIdx} title="x¹ → c¹" />
            </figure>
            <figure style={{ margin: 0 }}>
              <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', marginBottom: '0.2rem' }}>subspace 2 — dims (x₂, x₃)</figcaption>
              <SubspacePlot codebook={SUB_B} project={subB} selectedIdx={selectedIdx} title="x² → c²" />
            </figure>
          </div>
          <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.4rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><span style={{ minWidth: '8rem' }}>‖x¹−c¹‖² = {distA.toFixed(2)}</span>{bar(distA, barMax, COLORS[0])}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><span style={{ minWidth: '8rem' }}>‖x²−c²‖² = {distB.toFixed(2)}</span>{bar(distB, barMax, COLORS[1])}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}><span style={{ minWidth: '8rem' }}>total = {total.toFixed(2)}</span>{bar(total, barMax, 'var(--color-text-secondary)')}</div>
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
            Squared distance separates over disjoint blocks, so the selected point's total reconstruction error is exactly the
            sum of its two per-subspace errors. Subspace 1 is looser (dataset distortion {SUBSPACE_DIST_TOTAL[0].toFixed(1)})
            than subspace 2 ({SUBSPACE_DIST_TOTAL[1].toFixed(1)}).
          </div>
        </>
      )}

      {panel === 'frontier' && (
        <>
          <FrontierPlot />
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            ADC recall@10 vs the rate, for PQ at k*=256 (m = 1, 2, 4, 8) on the same 256-d finance cloud as the previous topic.
            m = 1 <em>is</em> the flat codebook (the dashed ceiling). A flat codebook needs k = 2ᴮ centroids — capped at k ≤ n
            and intractable past ~16–20 bits — so it cannot climb above 8 bits here; PQ reaches 64 bits at 2048 stored centroids.
            <strong> The honest caveat:</strong> at equal <em>trainable</em> bits a flat codebook matches or beats PQ; PQ's win is reach, not per-bit quality.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="flat VQ ceiling (8 bits)" value={`${(FLAT_CEILING * 100).toFixed(1)}%`} />
            <Readout label="PQ recall @ 64 bits" value={`${(FRONTIER[3].recall * 100).toFixed(1)}%`} accent />
            <Readout label="stored centroids @ 64 bits" value={`${FRONTIER[3].stored} (vs ${EFFECTIVE_64BIT} effective)`} />
            <Readout label="code size" value="64 bits = 8 bytes (128× smaller)" accent />
          </div>
        </>
      )}

      {panel === 'adc' && (
        <>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', marginBottom: '0.5rem' }}>
            Query <strong>q = point #{ADC_QUERY_IDX}</strong> (kept exact). The table holds ‖q^j − c^j_i‖² for each subspace j
            and sub-centroid i. Database <strong>point #{selectedIdx}</strong> has code ({codeA}, {codeB}) — its two highlighted
            cells sum to the exact distance.
          </div>
          <button onClick={() => setSelectedIdx((i) => (i + 1) % TOY_POINTS_4D.length)} style={{ ...pill(false), marginBottom: '0.6rem' }}>Select next database point</button>
          <div style={{ display: 'grid', gridTemplateColumns: `auto repeat(${ADC_TABLE[0].length}, 1fr)`, gap: '3px', maxWidth: '440px', fontFamily: 'var(--font-sans)', fontSize: '0.78rem' }}>
            <div />
            {ADC_TABLE[0].map((_, i) => (<div key={i} style={{ textAlign: 'center', color: 'var(--color-text-secondary)' }}>c·{i}</div>))}
            {ADC_TABLE.map((row, j) => (
              <Fragment key={`r${j}`}>
                <div style={{ color: 'var(--color-text-secondary)', alignSelf: 'center' }}>subspace {j + 1}</div>
                {row.map((v, i) => {
                  const hit = (j === 0 && i === codeA) || (j === 1 && i === codeB);
                  return (
                    <div key={`${j}-${i}`} style={{
                      textAlign: 'center', padding: '0.4rem 0.2rem', borderRadius: '0.3rem', fontVariantNumeric: 'tabular-nums',
                      border: `1px solid ${hit ? 'var(--color-accent)' : 'var(--color-border)'}`,
                      background: hit ? 'var(--color-accent)' : 'transparent', color: hit ? 'var(--color-bg)' : 'var(--color-text)',
                      fontWeight: hit ? 600 : 400,
                    }}>{v.toFixed(2)}</div>
                  );
                })}
              </Fragment>
            ))}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.8rem' }}>
            <Readout label={`ADC = table[1][${codeA}] + table[2][${codeB}]`} value={adc.toFixed(4)} accent />
            <Readout label="= exact ‖q − Q(x)‖²" value="yes (a memoized sum)" />
            <Readout label="cost per database vector" value="m = 2 lookups (O(m), not O(d))" />
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.6rem 0 0' }}>
            Symmetric distance (SDC) quantizes the query too and reads a k*×k* table — cheaper to precompute but, on average,
            a worse distance estimate, since it adds the query's own quantization error. The ordering can flip on individual pairs.
          </div>
        </>
      )}
    </div>
  );
}
