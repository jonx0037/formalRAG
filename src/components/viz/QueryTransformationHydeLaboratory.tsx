import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Query Transformation & HyDE Laboratory — three panels for the `query-transformation-hyde` topic:
 *   A. The query–document gap. A bare query for company c is tilted off its answer direction toward
 *      the generic "document-ness" axis g by a distribution-shift angle θ; as θ grows the bare query
 *      loses company specificity and recall@1 collapses, while HyDE — which retrieves with a generated
 *      hypothetical document — recovers the answer at every θ. A 2-D projection shows the bare query
 *      off-manifold and the HyDE centroid landing back inside the document cloud near the gold.
 *   B. HyDE as a Monte-Carlo estimator: bias vs variance. Averaging k hypotheticals denoises the
 *      estimate, so a FAITHFUL generator's recall RISES with k toward 1.0 and the angular deficit
 *      1−⟨ĥ_k,μ⟩ falls toward the 1/k rate. But a generator that HALLUCINATES on a fraction p of
 *      queries is consistent for the WRONG center: recall plateaus at a ceiling near 1−p — averaging
 *      cannot fix bias.
 *   C. HyDE as the neural generalization of pseudo-relevance feedback. The Rocchio update
 *      q'=(1−α)q+α·centroid, lifted into embedding space: HyDE's GENERATED centroid lifts recall toward
 *      1.0 while real pseudo-relevance feedback's RETRIEVED centroid is polluted by the bad query and
 *      hurts. An inset shows the imported term-space RM3/Rocchio ancestor (improve, then drift).
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every recall / deficit / coordinate number below is mirrored
 * TO THE DECIMAL from notebooks/query-transformation-hyde/query_transformation_hyde.py (viz_constants()).
 * Matching asserts: test_gap_exists / test_bare_degrades_with_shift / test_hyde_shift_robust /
 * test_mc_recall_rises_with_k / test_mc_variance_falls_one_over_k / test_hallucination_bias_floor /
 * test_hyde_beats_prf_real / test_prf_ancestor_imported. TS recomputes ONLY closed forms: the 1−p
 * ceiling lines, the 1/k reference, and pixel/scale maps.
 */

// --- baked from viz_constants() -------------------------------------------------------
const HYDE_N_DOCS = 8;
const HYDE_SECTOR = [0, 0, 1, 1, 2, 2, 3, 3];

// Panel A — the query–document gap
const SHIFT_GRID = [0, 30, 45, 60, 70, 75, 80, 85];                 // θ in degrees
const BARE_RECALL = [1.0, 1.0, 1.0, 0.906, 0.516, 0.375, 0.203, 0.188];
const HYDE_RECALL_FLAT = 1.0;                                       // θ-independent
const GEO: {
  docs: [number, number][]; query: [number, number]; hyde: [number, number];
  hyps: [number, number][]; worked: number; goldSector: number;
} = {
  docs: [[-0.242, 0.312], [0.109, 0.244], [0.702, -0.16], [0.897, -0.212], [-0.071, -0.709], [-0.345, -0.636], [0.409, 0.38], [0.19, 0.477]],
  query: [0.273, 0.123],
  hyde: [-0.508, 0.149],
  hyps: [[-0.068, 0.234], [-0.048, -0.221], [-0.335, 0.51], [-0.582, -0.247], [-0.693, 0.014], [0.312, -0.258]],
  worked: 0,
  goldSector: 0,
};

// Panel B — Monte-Carlo bias vs variance
const K_GRID = [1, 2, 3, 5, 8, 12, 20];
const HALLU_GRID = [0.0, 0.25, 0.5];
const RECALL_K_BY_P = [
  [0.602, 0.804, 0.86, 0.96, 0.988, 0.992, 1.0],
  [0.535, 0.69, 0.75, 0.802, 0.821, 0.829, 0.802],
  [0.456, 0.565, 0.598, 0.627, 0.621, 0.615, 0.567],
]; // [p index][k index]
const VAR_DEFICIT = [0.658, 0.559, 0.478, 0.375, 0.29, 0.225, 0.153]; // 1 − ⟨ĥ_k, μ⟩, faithful

// Panel C — HyDE as neural pseudo-relevance feedback
const ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0];
const RECALL_ALPHA_HYDE = [0.406, 0.852, 0.977, 0.984, 0.961];
const RECALL_ALPHA_PRF = [0.406, 0.336, 0.281, 0.242, 0.195];
const PRF_ANCESTOR: { nFb: number; rm3: number; rocchio: number }[] = [
  { nFb: 0, rm3: 0.5, rocchio: 0.75 }, { nFb: 1, rm3: 1.0, rocchio: 1.0 }, { nFb: 2, rm3: 1.0, rocchio: 1.0 },
  { nFb: 3, rm3: 0.75, rocchio: 0.75 }, { nFb: 4, rm3: 0.5, rocchio: 0.5 }, { nFb: 5, rm3: 0.75, rocchio: 0.5 },
];

// --- palette --------------------------------------------------------------------------
const ACCENT = 'var(--color-accent)';   // HyDE (the win)
const BAREC = '#c25b6b';                 // the bare query / real-PRF baseline (the loss)
const MUTED = '#9aa3ad';
const GRID = 'var(--color-border)';
const P_COLORS = ['#5fa873', '#d9a441', '#c25b6b']; // p = 0, 0.25, 0.5
const sectorColor = (s: number) => `hsl(${(s * 137.508) % 360}, 55%, 55%)`;

const fmt = (x: number, n = 3) => x.toFixed(n);
const r2 = (x: number) => Math.round(x * 100) / 100;

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
const sliderStyle = { width: '100%', accentColor: 'var(--color-accent)' } as const;

// generic linear scale
const scale = (v: number, d0: number, d1: number, p0: number, p1: number) =>
  p0 + (p1 - p0) * ((v - d0) / (d1 - d0 || 1));
// a polyline points string from (x,y) data arrays mapped through sx, sy
const polyPoints = (xs: number[], ys: number[], sx: (x: number) => number, sy: (y: number) => number) =>
  xs.map((x, i) => `${r2(sx(x))},${r2(sy(ys[i]))}`).join(' ');

// ===== Panel A — the query–document gap =================================================
function GapPanel({ ti, setTi }: { ti: number; setTi: (v: number) => void }) {
  const theta = SHIFT_GRID[ti];
  const bare = BARE_RECALL[ti];

  // geometry inset (left): a 2-D projection of the document cloud + bare query + HyDE centroid
  const GW = 250, GH = 300, gcx = 125, gcy = 150, gr = 120;
  const gx = (lx: number) => gcx + gr * lx;
  const gy = (ly: number) => gcy - gr * ly;

  // recall-vs-θ chart (right)
  const CW = 290, CH = 300, mL = 40, mR = 14, mT = 18, mB = 40;
  const sx = (d: number) => scale(d, 0, 90, mL, CW - mR);
  const sy = (v: number) => scale(v, 0, 1.05, CH - mB, mT);

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
        {/* --- geometry inset --- */}
        <svg viewBox={`0 0 ${GW} ${GH}`} role="img"
          aria-label="2-D projection of the document manifold, the off-manifold bare query, and the HyDE centroid"
          style={{ width: '100%', maxWidth: GW, height: 'auto', display: 'block', flex: '1 1 230px' }}>
          {/* HyDE transform arrow: bare query -> HyDE centroid */}
          <defs>
            <marker id="hyde-arrow" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill={ACCENT} />
            </marker>
          </defs>
          <line x1={gx(GEO.query[0])} y1={gy(GEO.query[1])} x2={gx(GEO.hyde[0])} y2={gy(GEO.hyde[1])}
            stroke={ACCENT} strokeWidth={1.5} strokeDasharray="4 3" markerEnd="url(#hyde-arrow)" opacity={0.8} />
          {/* hypothetical samples */}
          {GEO.hyps.map(([x, y], i) => (
            <circle key={`h${i}`} cx={r2(gx(x))} cy={r2(gy(y))} r={3} fill={ACCENT} opacity={0.3} />
          ))}
          {/* documents, colored by sector; gold doc ringed */}
          {GEO.docs.map(([x, y], i) => (
            <g key={`d${i}`}>
              <circle cx={r2(gx(x))} cy={r2(gy(y))} r={i === GEO.worked ? 8 : 6}
                fill={sectorColor(HYDE_SECTOR[i])} stroke={i === GEO.worked ? 'var(--color-text)' : 'none'} strokeWidth={1.5} />
            </g>
          ))}
          {/* HyDE centroid (star-ish) and the bare query (hollow square) */}
          <circle cx={r2(gx(GEO.hyde[0]))} cy={r2(gy(GEO.hyde[1]))} r={5.5} fill={ACCENT} stroke="var(--color-bg)" strokeWidth={1.5} />
          <rect x={r2(gx(GEO.query[0])) - 5} y={r2(gy(GEO.query[1])) - 5} width={10} height={10}
            fill="none" stroke={BAREC} strokeWidth={2} />
          <text x={r2(gx(GEO.query[0])) + 8} y={r2(gy(GEO.query[1])) - 6} fontSize={9} fill={BAREC}>bare query</text>
          <text x={r2(gx(GEO.hyde[0])) - 4} y={r2(gy(GEO.hyde[1])) + 18} fontSize={9} fill={ACCENT}>HyDE</text>
          <text x={r2(gx(GEO.docs[GEO.worked][0])) + 10} y={r2(gy(GEO.docs[GEO.worked][1])) + 3} fontSize={9} fill="var(--color-text-secondary)">gold</text>
        </svg>

        {/* --- recall vs θ chart --- */}
        <svg viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label="recall at 1 versus distribution-shift angle theta for the bare query and HyDE"
          style={{ width: '100%', maxWidth: CW, height: 'auto', display: 'block', flex: '1 1 270px' }}>
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={mL} y1={sy(v)} x2={CW - mR} y2={sy(v)} stroke={GRID} strokeWidth={0.5} />
              <text x={mL - 5} y={sy(v) + 3} fontSize={9} fill={MUTED} textAnchor="end">{v}</text>
            </g>
          ))}
          {/* HyDE flat line at 1.0 */}
          <line x1={sx(0)} y1={sy(HYDE_RECALL_FLAT)} x2={sx(90)} y2={sy(HYDE_RECALL_FLAT)} stroke={ACCENT} strokeWidth={2} />
          {/* bare recall curve */}
          <polyline points={polyPoints(SHIFT_GRID, BARE_RECALL, sx, sy)} fill="none" stroke={BAREC} strokeWidth={2} />
          {SHIFT_GRID.map((d, i) => (
            <circle key={i} cx={r2(sx(d))} cy={r2(sy(BARE_RECALL[i]))} r={i === ti ? 5 : 2.5}
              fill={i === ti ? BAREC : 'var(--color-bg)'} stroke={BAREC} strokeWidth={1.5} />
          ))}
          {/* θ marker */}
          <line x1={sx(theta)} y1={mT} x2={sx(theta)} y2={CH - mB} stroke={MUTED} strokeWidth={0.75} strokeDasharray="3 3" />
          {[0, 30, 60, 90].map((d) => (
            <text key={d} x={sx(d)} y={CH - mB + 14} fontSize={9} fill={MUTED} textAnchor="middle">{d}°</text>
          ))}
          <text x={(mL + CW - mR) / 2} y={CH - 4} fontSize={9} fill="var(--color-text-secondary)" textAnchor="middle">distribution-shift angle θ</text>
          <text x={sx(78)} y={sy(1.0) - 5} fontSize={9} fill={ACCENT}>HyDE</text>
          <text x={sx(83)} y={sy(0.30)} fontSize={9} fill={BAREC} textAnchor="middle">bare</text>
        </svg>
      </div>

      <div style={{ marginTop: '0.5rem' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
          distribution-shift angle θ = {theta}°
        </label>
        <input type="range" min={0} max={SHIFT_GRID.length - 1} value={ti} step={1}
          onChange={(e) => setTi(Number(e.target.value))} style={sliderStyle} aria-label="distribution-shift angle" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem', marginTop: '0.4rem' }}>
          <Readout label="bare-query recall@1" value={fmt(bare, 3)} color={BAREC} />
          <Readout label="HyDE recall@1" value={fmt(HYDE_RECALL_FLAT, 3)} color={ACCENT} />
        </div>
      </div>
    </div>
  );
}

// ===== Panel B — Monte-Carlo bias vs variance ==========================================
function MonteCarloPanel({ ki, pi, setKi, setPi }: { ki: number; pi: number; setKi: (v: number) => void; setPi: (v: number) => void }) {
  const k = K_GRID[ki];
  const p = HALLU_GRID[pi];
  const recall = RECALL_K_BY_P[pi][ki];
  const ceiling = 1 - p;

  const W = 540, H = 250, mL = 42, mR = 16, mT = 16, mB = 34;
  const xs = K_GRID.map((_, i) => i); // even spacing by index
  const sx = (i: number) => scale(i, 0, K_GRID.length - 1, mL, W - mR);
  const sy = (v: number) => scale(v, 0, 1.05, H - mB, mT);

  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.4rem', flexWrap: 'wrap' }}>
        {HALLU_GRID.map((pv, i) => (
          <button key={i} onClick={() => setPi(i)} style={{
            ...pill(i === pi), borderColor: P_COLORS[i], ...(i === pi ? { background: P_COLORS[i], color: 'var(--color-bg)' } : { color: P_COLORS[i] }),
          }}>hallucination rate p = {pv}</button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="recall at 1 versus number of averaged hypotheticals k, for each hallucination rate"
        style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={mL} y1={sy(v)} x2={W - mR} y2={sy(v)} stroke={GRID} strokeWidth={0.5} />
            <text x={mL - 5} y={sy(v) + 3} fontSize={9} fill={MUTED} textAnchor="end">{v}</text>
          </g>
        ))}
        {/* 1−p ceiling reference lines */}
        {HALLU_GRID.map((pv, i) => (
          <line key={`c${i}`} x1={mL} y1={sy(1 - pv)} x2={W - mR} y2={sy(1 - pv)}
            stroke={P_COLORS[i]} strokeWidth={1} strokeDasharray="2 4" opacity={0.6} />
        ))}
        {/* recall(k) curves */}
        {RECALL_K_BY_P.map((row, i) => (
          <g key={`r${i}`}>
            <polyline points={polyPoints(xs, row, sx, sy)} fill="none" stroke={P_COLORS[i]}
              strokeWidth={i === pi ? 2.5 : 1.25} opacity={i === pi ? 1 : 0.5} />
            {row.map((v, j) => (
              <circle key={j} cx={r2(sx(j))} cy={r2(sy(v))} r={i === pi && j === ki ? 5 : 2}
                fill={i === pi && j === ki ? P_COLORS[i] : 'var(--color-bg)'} stroke={P_COLORS[i]} strokeWidth={1.25} opacity={i === pi ? 1 : 0.5} />
            ))}
          </g>
        ))}
        {/* k marker */}
        <line x1={sx(ki)} y1={mT} x2={sx(ki)} y2={H - mB} stroke={MUTED} strokeWidth={0.75} strokeDasharray="3 3" />
        {K_GRID.map((kv, i) => (
          <text key={i} x={sx(i)} y={H - mB + 14} fontSize={9} fill={MUTED} textAnchor="middle">{kv}</text>
        ))}
        <text x={(mL + W - mR) / 2} y={H - 4} fontSize={9} fill="var(--color-text-secondary)" textAnchor="middle">hypotheticals averaged, k</text>
        <text x={W - mR - 4} y={sy(1) - 4} fontSize={9} fill={P_COLORS[0]} textAnchor="end">faithful → 1.0</text>
      </svg>

      <div style={{ marginTop: '0.4rem' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>hypotheticals averaged, k = {k}</label>
        <input type="range" min={0} max={K_GRID.length - 1} value={ki} step={1}
          onChange={(e) => setKi(Number(e.target.value))} style={sliderStyle} aria-label="number of hypotheticals averaged" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.6rem', marginTop: '0.4rem' }}>
          <Readout label={`recall@1 (p=${p}, k=${k})`} value={fmt(recall, 3)} color={P_COLORS[pi]} />
          <Readout label="bias ceiling 1 − p" value={fmt(ceiling, 2)} color={MUTED} />
          <Readout label={`est. deficit 1−⟨ĥ,μ⟩ (k=${k})`} value={fmt(VAR_DEFICIT[ki], 3)} />
        </div>
      </div>
    </div>
  );
}

// ===== Panel C — HyDE as neural pseudo-relevance feedback ==============================
function NeuralPRFPanel({ ai, setAi }: { ai: number; setAi: (v: number) => void }) {
  const alpha = ALPHA_GRID[ai];
  const hyde = RECALL_ALPHA_HYDE[ai];
  const prf = RECALL_ALPHA_PRF[ai];

  const W = 360, H = 250, mL = 40, mR = 14, mT = 16, mB = 38;
  const sx = (a: number) => scale(a, 0, 1, mL, W - mR);
  const sy = (v: number) => scale(v, 0, 1.05, H - mB, mT);

  // inset: imported term-space RM3 ancestor (improve then drift)
  const IW = 170, IH = 120, iL = 26, iB = 24, iT = 12, iR = 10;
  const ix = (n: number) => scale(n, 0, 5, iL, IW - iR);
  const iy = (v: number) => scale(v, 0, 1.05, IH - iB, iT);

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="recall at 1 versus interpolation weight alpha for HyDE and real pseudo-relevance feedback"
          style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block', flex: '1 1 340px' }}>
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={mL} y1={sy(v)} x2={W - mR} y2={sy(v)} stroke={GRID} strokeWidth={0.5} />
              <text x={mL - 5} y={sy(v) + 3} fontSize={9} fill={MUTED} textAnchor="end">{v}</text>
            </g>
          ))}
          <polyline points={polyPoints(ALPHA_GRID, RECALL_ALPHA_HYDE, sx, sy)} fill="none" stroke={ACCENT} strokeWidth={2.5} />
          <polyline points={polyPoints(ALPHA_GRID, RECALL_ALPHA_PRF, sx, sy)} fill="none" stroke={BAREC} strokeWidth={2.5} />
          {ALPHA_GRID.map((a, i) => (
            <g key={i}>
              <circle cx={r2(sx(a))} cy={r2(sy(RECALL_ALPHA_HYDE[i]))} r={i === ai ? 5 : 2.5}
                fill={i === ai ? ACCENT : 'var(--color-bg)'} stroke={ACCENT} strokeWidth={1.5} />
              <circle cx={r2(sx(a))} cy={r2(sy(RECALL_ALPHA_PRF[i]))} r={i === ai ? 5 : 2.5}
                fill={i === ai ? BAREC : 'var(--color-bg)'} stroke={BAREC} strokeWidth={1.5} />
            </g>
          ))}
          <line x1={sx(alpha)} y1={mT} x2={sx(alpha)} y2={H - mB} stroke={MUTED} strokeWidth={0.75} strokeDasharray="3 3" />
          {[0, 0.5, 1].map((a) => (
            <text key={a} x={sx(a)} y={H - mB + 14} fontSize={9} fill={MUTED} textAnchor="middle">{a}</text>
          ))}
          <text x={(mL + W - mR) / 2} y={H - 4} fontSize={9} fill="var(--color-text-secondary)" textAnchor="middle">interpolation weight α  (0 = bare query, 1 = pure pseudo-doc)</text>
          <text x={sx(0.55)} y={sy(0.99)} fontSize={9} fill={ACCENT}>HyDE (generated)</text>
          <text x={sx(0.5)} y={sy(0.20)} fontSize={9} fill={BAREC}>real PRF (retrieved)</text>
        </svg>

        {/* inset: the imported term-space ancestor */}
        <div style={{ flex: '1 1 180px' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginBottom: '0.2rem' }}>
            term-space ancestor (imported): RM3 recall@4
          </div>
          <svg viewBox={`0 0 ${IW} ${IH}`} role="img" aria-label="imported term-space RM3 recall versus feedback size: improve then drift"
            style={{ width: '100%', maxWidth: IW, height: 'auto', display: 'block' }}>
            {[0, 1].map((v) => (
              <line key={v} x1={iL} y1={iy(v)} x2={IW - iR} y2={iy(v)} stroke={GRID} strokeWidth={0.5} />
            ))}
            <polyline points={polyPoints(PRF_ANCESTOR.map((d) => d.nFb), PRF_ANCESTOR.map((d) => d.rm3), ix, iy)}
              fill="none" stroke={MUTED} strokeWidth={2} />
            {PRF_ANCESTOR.map((d, i) => (
              <circle key={i} cx={r2(ix(d.nFb))} cy={r2(iy(d.rm3))} r={2.5} fill={MUTED} />
            ))}
            <text x={IW / 2} y={IH - 4} fontSize={8} fill={MUTED} textAnchor="middle">feedback docs</text>
            <text x={iL - 4} y={iy(1) + 3} fontSize={8} fill={MUTED} textAnchor="end">1</text>
            <text x={iL - 4} y={iy(0) + 3} fontSize={8} fill={MUTED} textAnchor="end">0</text>
          </svg>
        </div>
      </div>

      <div style={{ marginTop: '0.4rem' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>interpolation weight α = {alpha}</label>
        <input type="range" min={0} max={ALPHA_GRID.length - 1} value={ai} step={1}
          onChange={(e) => setAi(Number(e.target.value))} style={sliderStyle} aria-label="interpolation weight alpha" />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem', marginTop: '0.4rem' }}>
          <Readout label={`HyDE recall@1 (α=${alpha})`} value={fmt(hyde, 3)} color={ACCENT} />
          <Readout label={`real-PRF recall@1 (α=${alpha})`} value={fmt(prf, 3)} color={BAREC} />
        </div>
      </div>
    </div>
  );
}

// ===== shell ===========================================================================
type Panel = 'gap' | 'mc' | 'prf';
const TEX: Record<Panel, string> = {
  gap: 'q_c(\\theta) = \\operatorname{normalize}\\!\\big(\\cos\\theta\\,u_c + \\sin\\theta\\,g\\big)',
  mc: '\\hat h_k = \\operatorname{normalize}\\!\\Big(\\tfrac{1}{k}\\textstyle\\sum_{i=1}^{k} h_i\\Big), \\qquad \\text{recall} \\le 1 - p',
  prf: "q'(\\alpha) = (1-\\alpha)\\,q + \\alpha\\,\\operatorname{centroid}(\\text{hypotheticals})",
};

export default memo(function QueryTransformationHydeLaboratory() {
  const [panel, setPanel] = useState<Panel>('gap');
  const [ti, setTi] = useState(5);   // θ = 75° (the operating shift)
  const [ki, setKi] = useState(3);   // k = 5
  const [pi, setPi] = useState(1);   // p = 0.25
  const [ai, setAi] = useState(2);   // α = 0.5
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="query-transformation-hyde" style={{
      border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem',
      background: 'var(--color-surface)', fontFamily: 'var(--font-sans)', margin: '1.5rem 0',
    }}>
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.3rem', flexWrap: 'wrap' }}>
        <button onClick={() => setPanel('gap')} style={pill(panel === 'gap')}>A · the query–document gap</button>
        <button onClick={() => setPanel('mc')} style={pill(panel === 'mc')}>B · bias vs variance</button>
        <button onClick={() => setPanel('prf')} style={pill(panel === 'prf')}>C · HyDE as neural PRF</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem', overflowX: 'auto' }} />
      {panel === 'gap' && <GapPanel ti={ti} setTi={setTi} />}
      {panel === 'mc' && <MonteCarloPanel ki={ki} pi={pi} setKi={setKi} setPi={setPi} />}
      {panel === 'prf' && <NeuralPRFPanel ai={ai} setAi={setAi} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.8rem', marginBottom: 0 }}>
        Numbers are baked from <code>query_transformation_hyde.py</code>'s <code>viz_constants()</code>; the
        document manifold is the dense-retrieval finance geometry ({HYDE_N_DOCS} company prototypes, dim 32),
        and the queries are drawn off it. A synthetic von Mises–Fisher generator stands in for the LLM.
      </p>
    </div>
  );
});
