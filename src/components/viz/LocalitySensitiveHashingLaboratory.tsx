import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Locality-Sensitive Hashing Laboratory — three panels for the `locality-sensitive-hashing` topic:
 *   A. Collision probability. A unit circle with two vectors at an adjustable angle theta and evenly
 *      spaced random-hyperplane normals colored by whether they separate the pair; beside it the law
 *      P[collision] = 1 - theta/pi, the closed form drawn smooth with the measured points overlaid.
 *   B. Amplification: the S-curve g(p) = 1 - (1-p^k)^L. The live closed-form curve for the (k, L) the
 *      sliders pick, against the base-hash diagonal g = p and three baked reference curves — k sharpens
 *      the threshold, L lifts the floor.
 *   C. The rho exponent and the head-to-head. rho = ln(1/p1)/ln(1/p2) vs the approximation factor c,
 *      and the recall-vs-cost frontier of LSH vs IVF vs HNSW on one shared normalized cloud.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): COLLISION, SCURVE_REF, RHO_CURVE, H2H_LSH, H2H_IVF, H2H_HNSW
 * (and H2H_N/TOPK/LSH_K) are mirrored TO THE DECIMAL from
 * notebooks/locality-sensitive-hashing/locality_sensitive_hashing.py (viz_constants()). The lab
 * recomputes only the CLOSED-FORM curves in TS — 1 - theta/pi, g(p) = 1-(1-p^k)^L, the unit-circle
 * geometry — exactly as the source's tests assert them; every MEASURED collision rate, rho, recall,
 * and cost is baked. test_collision_probability_matches_theory / test_s_curve_sharpens /
 * test_rho_below_one / test_lsh_recall_cost_tradeoff / test_head_to_head assert these. Change a number
 * here -> change it there, and re-run the notebook. Sliders only (no d3 drag). SVG text inherits theme.
 */

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: empirical SimHash collision rate vs theory 1 - theta/pi, at theta = frac * pi (8000 planes).
const COLLISION = [
  { tf: 0.1, emp: 0.9009, thy: 0.9 }, { tf: 0.2, emp: 0.801, thy: 0.8 }, { tf: 0.3, emp: 0.7025, thy: 0.7 },
  { tf: 0.4, emp: 0.6044, thy: 0.6 }, { tf: 0.5, emp: 0.5019, thy: 0.5 }, { tf: 0.6, emp: 0.4006, thy: 0.4 },
  { tf: 0.7, emp: 0.3029, thy: 0.3 }, { tf: 0.8, emp: 0.198, thy: 0.2 }, { tf: 0.9, emp: 0.1066, thy: 0.1 },
];
// Panel B: g(p) = 1-(1-p^k)^L sampled on P_GRID for three configs — the decimal anchor on the closed form.
const P_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0];
const SCURVE_REF: { k: number; L: number; g: number[] }[] = [
  { k: 4, L: 4, g: [0.0, 0.0, 0.0004, 0.002, 0.0064, 0.0155, 0.032, 0.0587, 0.0985, 0.1542, 0.2275, 0.3188, 0.426, 0.5446, 0.6666, 0.7816, 0.8785, 0.9478, 0.986, 0.9988, 1.0] },
  { k: 8, L: 8, g: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0005, 0.0018, 0.0052, 0.0134, 0.0308, 0.0651, 0.1267, 0.2282, 0.3781, 0.57, 0.7699, 0.9215, 0.9889, 0.9998, 1.0] },
  { k: 16, L: 4, g: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0003, 0.0011, 0.0041, 0.0132, 0.0395, 0.1079, 0.2655, 0.5595, 0.9017, 1.0] },
];
// Panel C: rho = ln(1/p1)/ln(1/p2) at theta1 = 0.15*pi, far angle theta2 = c*theta1.
const RHO_CURVE = [
  { c: 1.25, p1: 0.85, p2: 0.8125, rho: 0.7827 }, { c: 1.5, p1: 0.85, p2: 0.775, rho: 0.6376 },
  { c: 2.0, p1: 0.85, p2: 0.7, rho: 0.4556 }, { c: 2.5, p1: 0.85, p2: 0.625, rho: 0.3458 },
  { c: 3.0, p1: 0.85, p2: 0.55, rho: 0.2718 }, { c: 4.0, p1: 0.85, p2: 0.4, rho: 0.1774 },
];
// Panel C: recall-vs-cost frontier on one normalized n=500 cloud, one shared ground truth.
// cost = distance computations per query (LSH: L*k hashing + candidates; IVF: frac*n + nlist; graph: ndist).
const H2H_N = 500, H2H_TOPK = 10, H2H_LSH_K = 14;
const H2H_LSH = [
  { L: 1, recall: 0.125, cost: 15.8 }, { L: 2, recall: 0.2175, cost: 32.0 }, { L: 4, recall: 0.4225, cost: 64.0 },
  { L: 8, recall: 0.615, cost: 126.7 }, { L: 16, recall: 0.795, cost: 249.0 }, { L: 32, recall: 0.94, cost: 486.4 },
  { L: 64, recall: 0.9975, cost: 955.8 },
];
const H2H_IVF = [
  { nprobe: 1, recall: 0.6, cost: 44.9 }, { nprobe: 2, recall: 0.8225, cost: 68.8 }, { nprobe: 4, recall: 0.965, cost: 113.0 },
  { nprobe: 8, recall: 1.0, cost: 207.9 }, { nprobe: 16, recall: 1.0, cost: 392.6 }, { nprobe: 22, recall: 1.0, cost: 522.0 },
];
const H2H_HNSW = [
  { ef: 1, recall: 0.1, cost: 33.4 }, { ef: 2, recall: 0.2, cost: 38.4 }, { ef: 4, recall: 0.4, cost: 46.2 },
  { ef: 8, recall: 0.8, cost: 58.3 }, { ef: 16, recall: 0.9975, cost: 82.2 }, { ef: 32, recall: 1.0, cost: 123.1 },
  { ef: 64, recall: 1.0, cost: 183.9 }, { ef: 128, recall: 1.0, cost: 278.9 },
];

const LSH_COLOR = 'var(--color-accent)';
const IVF_COLOR = '#6a8caf';
const HNSW_COLOR = '#5fa873';
const THEORY_COLOR = 'var(--color-text-secondary)';

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '9.5rem' }}>{label} = <strong>{fmt(value)}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
    </label>
  );
}

const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

// ===== Panel A — the unit circle: two vectors at angle theta, normals colored by separation ========
const N_NORMALS = 72;   // evenly spaced random-hyperplane normals (a deterministic Monte-Carlo of theta/pi)

// round trig-derived coordinates to a fixed precision so SSR (Node) and client (browser) serialize
// identical strings — full-precision Math.cos/sin differ in the last ULP across engines and warn on hydrate.
const r2 = (v: number) => Math.round(v * 100) / 100;

function CollisionCircle({ thetaFrac }: { thetaFrac: number }) {
  const S = 300, c = S / 2, R = 110;
  const theta = thetaFrac * Math.PI;
  // u points up; v is rotated by theta (clockwise in screen coords)
  const uAng = -Math.PI / 2;
  const vAng = -Math.PI / 2 + theta;
  const ux = r2(c + R * Math.cos(uAng)), uy = r2(c + R * Math.sin(uAng));
  const vx = r2(c + R * Math.cos(vAng)), vy = r2(c + R * Math.sin(vAng));
  // a normal h separates u,v iff sign(<h,u>) != sign(<h,v>)
  const normals = Array.from({ length: N_NORMALS }, (_, i) => {
    const a = (2 * Math.PI * i) / N_NORMALS;
    const hx = Math.cos(a), hy = Math.sin(a);
    const su = Math.sign(hx * Math.cos(uAng) + hy * Math.sin(uAng));
    const sv = Math.sign(hx * Math.cos(vAng) + hy * Math.sin(vAng));
    return { a, sep: su !== sv };
  });
  const sepCount = normals.filter((n) => n.sep).length;
  return (
    <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="unit circle with two vectors and separating hyperplane normals" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <circle cx={c} cy={c} r={R} fill="none" stroke="var(--color-border)" strokeWidth={1} />
      {/* tick for each normal direction, colored by whether it separates the pair */}
      {normals.map((n, i) => {
        const x1 = r2(c + (R - 7) * Math.cos(n.a)), y1 = r2(c + (R - 7) * Math.sin(n.a));
        const x2 = r2(c + (R + 7) * Math.cos(n.a)), y2 = r2(c + (R + 7) * Math.sin(n.a));
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={n.sep ? LSH_COLOR : HNSW_COLOR} strokeWidth={n.sep ? 2 : 1.4} opacity={n.sep ? 0.95 : 0.55} />;
      })}
      {/* the two vectors */}
      <line x1={c} y1={c} x2={ux} y2={uy} stroke="var(--color-text)" strokeWidth={2.4} />
      <line x1={c} y1={c} x2={vx} y2={vy} stroke="var(--color-text)" strokeWidth={2.4} />
      <circle cx={ux} cy={uy} r={4} fill="var(--color-text)" /><text x={r2(ux + 6)} y={r2(uy - 4)} fontSize={12} fill="var(--color-text)" fontFamily="var(--font-sans)">x</text>
      <circle cx={vx} cy={vy} r={4} fill="var(--color-text)" /><text x={r2(vx + 6)} y={r2(vy + 12)} fontSize={12} fill="var(--color-text)" fontFamily="var(--font-sans)">y</text>
      <text x={c} y={S - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
        {sepCount}/{N_NORMALS} normals separate ≈ θ/π = {(thetaFrac).toFixed(2)}
      </text>
    </svg>
  );
}

function CollisionCurve({ thetaFrac }: { thetaFrac: number }) {
  const pw = 360, ph = 300, pad = 44;
  const fx = (tf: number) => pad + (pw - 2 * pad) * tf;            // theta/pi in [0,1]
  const fy = (p: number) => ph - pad - (ph - 2 * pad) * p;         // collision prob in [0,1]
  const curve = Array.from({ length: 51 }, (_, i) => i / 50).map((tf) => `${fx(tf).toFixed(1)} ${fy(1 - tf).toFixed(1)}`);
  const markP = 1 - thetaFrac;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {[0, 0.5, 1].map((v) => (<text key={v} x={fx(v)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      <text x={pw / 2} y={ph - 5} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">angle θ / π</text>
      <text x={12} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${ph / 2})`}>collision probability</text>
      <path d={'M' + curve.join(' L')} fill="none" stroke={THEORY_COLOR} strokeWidth={2} strokeDasharray="5 3" />
      {COLLISION.map((d, i) => (<circle key={i} cx={fx(d.tf)} cy={fy(d.emp)} r={3.5} fill={LSH_COLOR} />))}
      <line x1={fx(thetaFrac)} y1={pad} x2={fx(thetaFrac)} y2={ph - pad} stroke="var(--color-accent)" strokeWidth={1} strokeDasharray="3 3" />
      <circle cx={fx(thetaFrac)} cy={fy(markP)} r={4.5} fill="var(--color-accent)" />
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={pw - pad - 120} y1={pad + 4} x2={pw - pad - 104} y2={pad + 4} stroke={THEORY_COLOR} strokeWidth={2} strokeDasharray="5 3" />
        <text x={pw - pad - 100} y={pad + 7} fill="var(--color-text-secondary)">1 − θ/π</text>
        <circle cx={pw - pad - 112} cy={pad + 20} r={3.5} fill={LSH_COLOR} /><text x={pw - pad - 100} y={pad + 23} fill="var(--color-text-secondary)">measured</text>
      </g>
    </svg>
  );
}

// ===== Panel B — the S-curve g(p) = 1-(1-p^k)^L =====================================================
function SCurvePlot({ k, L }: { k: number; L: number }) {
  const pw = 620, ph = 320, pad = 48;
  const fx = (p: number) => pad + (pw - 2 * pad) * p;
  const fy = (g: number) => ph - pad - (ph - 2 * pad) * g;
  const g = (p: number) => 1 - Math.pow(1 - Math.pow(p, k), L);
  const live = Array.from({ length: 101 }, (_, i) => i / 100).map((p) => `${fx(p).toFixed(1)} ${fy(g(p)).toFixed(1)}`);
  const refLine = (ys: number[]) => P_GRID.map((p, i) => (i === 0 ? 'M' : 'L') + fx(p).toFixed(1) + ' ' + fy(ys[i]).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (<text key={v} x={fx(v)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(2)}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">base collision probability p = 1 − θ/π</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>composite collision g(p)</text>
      {/* base-hash diagonal g = p (k=L=1) */}
      <line x1={fx(0)} y1={fy(0)} x2={fx(1)} y2={fy(1)} stroke="var(--color-border)" strokeWidth={1.2} strokeDasharray="2 3" />
      {/* baked reference curves */}
      {SCURVE_REF.map((r, i) => (<path key={i} d={refLine(r.g)} fill="none" stroke={THEORY_COLOR} strokeWidth={1.3} opacity={0.5} />))}
      {/* the live (k, L) curve */}
      <path d={'M' + live.join(' L')} fill="none" stroke={LSH_COLOR} strokeWidth={2.8} />
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={pw - pad - 150} y1={pad + 4} x2={pw - pad - 134} y2={pad + 4} stroke={LSH_COLOR} strokeWidth={2.8} />
        <text x={pw - pad - 130} y={pad + 7} fill="var(--color-text-secondary)">g(p; k={k}, L={L})</text>
        <line x1={pw - pad - 150} y1={pad + 20} x2={pw - pad - 134} y2={pad + 20} stroke={THEORY_COLOR} strokeWidth={1.3} opacity={0.6} />
        <text x={pw - pad - 130} y={pad + 23} fill="var(--color-text-secondary)">(4,4) (8,8) (16,4)</text>
        <line x1={pw - pad - 150} y1={pad + 36} x2={pw - pad - 134} y2={pad + 36} stroke="var(--color-border)" strokeWidth={1.2} strokeDasharray="2 3" />
        <text x={pw - pad - 130} y={pad + 39} fill="var(--color-text-secondary)">base hash g = p</text>
      </g>
    </svg>
  );
}

// ===== Panel C — rho curve and the head-to-head frontier ===========================================
function RhoPlot({ idx }: { idx: number }) {
  const pw = 320, ph = 300, pad = 46;
  const cmin = 1, cmax = 4.2;
  const fx = (c: number) => pad + (pw - 2 * pad) * (c - cmin) / (cmax - cmin);
  const fy = (r: number) => ph - pad - (ph - 2 * pad) * r;        // rho in [0,1]
  const path = RHO_CURVE.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.c).toFixed(1) + ' ' + fy(d.rho).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={fy(1)} x2={pw - pad} y2={fy(1)} stroke="var(--color-border)" strokeWidth={1} strokeDasharray="3 3" />
      <text x={pw - pad} y={fy(1) - 4} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">ρ = 1 (linear)</text>
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {[1, 2, 3, 4].map((v) => (<text key={v} x={fx(v)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
      <text x={pw / 2} y={ph - 5} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">approximation factor c = r₂/r₁</text>
      <text x={12} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${ph / 2})`}>exponent ρ  (query O(nᵖ))</text>
      <path d={path} fill="none" stroke={LSH_COLOR} strokeWidth={2.6} />
      {RHO_CURVE.map((d, i) => (<circle key={i} cx={fx(d.c)} cy={fy(d.rho)} r={i === idx ? 5.5 : 3} fill={LSH_COLOR} opacity={i === idx ? 1 : 0.7} />))}
    </svg>
  );
}

function HeadToHeadPlot({ recallTarget }: { recallTarget: number }) {
  const pw = 360, ph = 300, pad = 46;
  const cmax = 600;
  const fx = (cost: number) => pad + (pw - 2 * pad) * Math.min(cost, cmax) / cmax;
  const fy = (r: number) => ph - pad - (ph - 2 * pad) * r;
  const series: [string, string, { recall: number; cost: number }[]][] = [
    ['HNSW', HNSW_COLOR, H2H_HNSW], ['IVF', IVF_COLOR, H2H_IVF], ['LSH', LSH_COLOR, H2H_LSH],
  ];
  const line = (rows: { recall: number; cost: number }[]) => rows.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.cost).toFixed(1) + ' ' + fy(d.recall).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {[0, 200, 400, 600].map((v) => (<text key={v} x={fx(v)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
      <text x={pw / 2} y={ph - 5} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">cost (distance comps / query)</text>
      <text x={12} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${ph / 2})`}>recall@{H2H_TOPK}</text>
      <line x1={pad} y1={fy(recallTarget)} x2={pw - pad} y2={fy(recallTarget)} stroke="var(--color-text)" strokeWidth={1} strokeDasharray="4 3" opacity={0.5} />
      {series.map(([name, color, rows]) => (
        <g key={name}>
          <path d={line(rows)} fill="none" stroke={color} strokeWidth={2.4} strokeLinejoin="round" />
          {rows.map((d, i) => (<circle key={i} cx={fx(d.cost)} cy={fy(d.recall)} r={2.6} fill={color} />))}
        </g>
      ))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        {series.map(([name, color], i) => (
          <g key={name}>
            <line x1={pw - pad - 86} y1={pad + 4 + i * 15} x2={pw - pad - 70} y2={pad + 4 + i * 15} stroke={color} strokeWidth={2.4} />
            <text x={pw - pad - 66} y={pad + 7 + i * 15} fill="var(--color-text-secondary)">{name}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'collision' | 'amplify' | 'frontier';

function minCostAtRecall(rows: { recall: number; cost: number }[], target: number): number | null {
  const ok = rows.filter((r) => r.recall >= target - 1e-9).map((r) => r.cost);
  return ok.length ? Math.min(...ok) : null;
}

export default memo(function LocalitySensitiveHashingLaboratory() {
  const [panel, setPanel] = useState<Panel>('collision');
  const [thetaFrac, setThetaFrac] = useState(0.3);
  const [k, setK] = useState(8);
  const [L, setL] = useState(8);
  const [rhoIdx, setRhoIdx] = useState(2);
  const [recallTarget, setRecallTarget] = useState(0.9);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'collision'
        ? '\\Pr[h(\\mathbf{x}) = h(\\mathbf{y})] = 1 - \\frac{\\theta}{\\pi}'
        : panel === 'amplify'
          ? 'g(p) = 1 - (1 - p^{k})^{L}, \\qquad p = 1 - \\tfrac{\\theta}{\\pi}'
          : '\\rho = \\frac{\\ln(1/p_1)}{\\ln(1/p_2)} < 1, \\qquad \\text{query } O(n^{\\rho})';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const lshC = minCostAtRecall(H2H_LSH, recallTarget);
  const ivfC = minCostAtRecall(H2H_IVF, recallTarget);
  const hnswC = minCostAtRecall(H2H_HNSW, recallTarget);
  const rho = RHO_CURVE[rhoIdx];

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'collision')} onClick={() => setPanel('collision')}>A · collision probability</button>
        <button style={pill(panel === 'amplify')} onClick={() => setPanel('amplify')}>B · amplification (S-curve)</button>
        <button style={pill(panel === 'frontier')} onClick={() => setPanel('frontier')}>C · ρ & head-to-head</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'collision' && (
        <div>
          <Slider label="angle θ / π" value={thetaFrac} min={0.05} max={0.95} step={0.05} onChange={setThetaFrac} fmt={(v) => v.toFixed(2)} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.1fr', gap: '0.8rem', alignItems: 'center' }}>
            <CollisionCircle thetaFrac={thetaFrac} />
            <CollisionCurve thetaFrac={thetaFrac} />
          </div>
          <div style={{ display: 'flex', gap: '2rem', marginTop: '0.5rem' }}>
            <Readout label="exact collision 1 − θ/π" value={(1 - thetaFrac).toFixed(2)} accent />
            <Readout label="angle θ" value={`${(thetaFrac * 180).toFixed(0)}°`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            One bit per random hyperplane. A plane separates <strong>x</strong> and <strong>y</strong> exactly when its normal
            lands in the wedge between them — angular measure θ/π — so the two collide with probability 1 − θ/π. The measured
            points sit on the line because the law is exact, not fitted.
          </p>
        </div>
      )}

      {panel === 'amplify' && (
        <div>
          <Slider label="bits per table k (AND)" value={k} min={1} max={20} step={1} onChange={setK} fmt={(v) => v.toFixed(0)} />
          <Slider label="number of tables L (OR)" value={L} min={1} max={32} step={1} onChange={setL} fmt={(v) => v.toFixed(0)} />
          <SCurvePlot k={k} L={L} />
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Concatenating k bits is an AND (a table-collision needs all k, probability p<sup>k</sup>); unioning L tables is an OR
            (collide if any matches, 1 − (1 − p<sup>k</sup>)<sup>L</sup>). Larger k pushes the threshold right and steepens it;
            larger L lifts the whole curve. At k = L = 1 the curve is the diagonal g = p — the bare hash.
          </p>
        </div>
      )}

      {panel === 'frontier' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.1fr', gap: '0.8rem', alignItems: 'start' }}>
            <div>
              <Slider label="far/near gap c" value={rhoIdx} min={0} max={RHO_CURVE.length - 1} step={1} onChange={setRhoIdx} fmt={(v) => `${RHO_CURVE[v].c}`} />
              <RhoPlot idx={rhoIdx} />
              <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.3rem' }}>
                <Readout label={`ρ at c = ${rho.c}`} value={rho.rho.toFixed(3)} accent />
                <Readout label="p₁ / p₂" value={`${rho.p1} / ${rho.p2}`} />
              </div>
            </div>
            <div>
              <Slider label="recall target" value={recallTarget} min={0.5} max={1} step={0.05} onChange={setRecallTarget} fmt={(v) => v.toFixed(2)} />
              <HeadToHeadPlot recallTarget={recallTarget} />
              <div style={{ display: 'flex', gap: '1.1rem', marginTop: '0.3rem' }}>
                <Readout label="HNSW cost" value={hnswC === null ? '—' : hnswC.toFixed(0)} />
                <Readout label="IVF cost" value={ivfC === null ? '—' : ivfC.toFixed(0)} />
                <Readout label="LSH cost" value={lshC === null ? '—' : lshC.toFixed(0)} accent />
              </div>
            </div>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', lineHeight: 1.5 }}>
            Left: the exponent ρ = ln(1/p₁)/ln(1/p₂) falls below 1 — query time O(n<sup>ρ</sup>) is sublinear — and shrinks as the
            near/far gap c widens. Right: all four indexes on one normalized {H2H_N}-vector cloud and one shared ground truth, by
            distance computations per query (LSH's cost includes its L·k = up to {64 * H2H_LSH_K} hashing projections). On this
            low-rank cloud the data-aware indexes dominate the data-oblivious hash — LSH's promise is its distribution-free
            worst-case guarantee, not its constant factors here.
          </p>
        </div>
      )}
    </div>
  );
});
