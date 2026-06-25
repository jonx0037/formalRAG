import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Negative-Sampling Laboratory — four panels for the `negative-sampling-hard-negatives` topic:
 *   A. The gradient weights by similarity. The anchor's mixed negative batch (same-sector hard negatives
 *      first, then near-orthogonal cross-sector ones), each bar sized by its softmax gradient weight
 *      p_i = softmax(cos/tau). The temperature slider routes the gradient onto the few hard negatives:
 *      the hard share (recomputed live) climbs from the count fraction toward 1 as tau -> 0.
 *   B. The false-negative problem. The false-negative rate tau+ of the MINED set vs RANDOM sampling as the
 *      mining depth k changes: mining the nearest neighbors is heavily contaminated by same-company
 *      accidental positives at small k, while random sampling hugs the class prior.
 *   C. The debiased estimator. The biased / true / debiased estimator of the negative expectation: the
 *      convergence of the mean absolute error as N grows (biased plateaus at the bias, debiased -> 0), and
 *      the debiased bar (E_p - tau+ E_p+)/(1 - tau+) recomputed LIVE as the tau+ slider moves — it meets
 *      the oracle exactly at the class prior and collapses to the biased mean at tau+ = 0.
 *   D. ANCE staleness. The frozen index's overlap with the fresh encoder decays with steps-since-refresh
 *      (and its gold recall falls below the fresh ceiling); the refresh interval R trades staleness against
 *      re-encode cost, and the total-cost knee is recomputed live as the staleness weight slider moves.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): NEG_COS, NEG_SAME_SECTOR, HARD_SHARE_CURVE, BATCH_LOSS,
 * TAU_PLUS_CURVE, E_POS_MEAN, E_BIASED_MEAN, E_ORACLE_MEAN, MI_FLOOR, DEBIASED_CONVERGENCE,
 * STALENESS_CURVE, REFRESH_TRADEOFF are mirrored TO THE DECIMAL from
 * notebooks/negative-sampling-hard-negatives/negative_sampling_hard_negatives.py (viz_constants()). The
 * lab recomputes only CLOSED FORM in TS — the softmax gradient weights and hard share softmax(NEG_COS/tau),
 * the debiased estimator (E_p - tau+ E_p+)/(1 - tau+) floored at MI_FLOOR, and the refresh total-cost knee
 * — exactly as the source's tests assert (test_hard_share_rises_as_tau_falls,
 * test_debiased_identity_at_full_pool, test_refresh_tradeoff_monotone). Every MEASURED number (the
 * false-negative curve, the convergence MAE, the staleness decay) is baked. Change a number here -> change
 * it there, and re-run the notebook. Sliders only (no d3 drag). SVG text inherits theme.
 */

// --- baked from viz_constants() -------------------------------------------------------
const CLASS_PRIOR_TAU_PLUS = 0.0968;
const GRAD_TAU = 0.2;

// Panel A: the anchor's mixed negative set (hardest first) and the same-sector (hard) mask. The gradient
// weights and the hard share are the closed form softmax(cos/tau), recomputed in TS. (The .py also bakes
// HARD_SHARE_CURVE as the decimal anchor its test pins; the live recompute here is the source of truth.)
const NEG_COS = [
  0.5793, 0.5596, 0.5402, 0.513, 0.4536, 0.3995, 0.3984, 0.3739, 0.2551, 0.2217, 0.1979, 0.1909,
  0.1689, 0.1627, 0.1579, 0.1569, 0.1153, 0.0714, 0.0297, 0.0248, -0.0135, -0.0294, -0.0324, -0.1492,
  -0.1582, -0.1658, -0.166, -0.1931,
];
const NEG_SAME_SECTOR = [
  true, true, true, true, false, false, false, false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false, false, false, false, false, false,
];
const HARD_COUNT_FRACTION = 0.1429;
const BATCH_LOSS = { hard_loss: 0.5206, random_loss: 0.1942 };

// Panel B: the false-negative rate of mined vs random negatives as the mining depth k tightens. MEASURED.
const TAU_PLUS_CURVE = [
  { k: 1, mined: 1.0, random: 0.125, prior: 0.0968 }, { k: 2, mined: 1.0, random: 0.125, prior: 0.0968 },
  { k: 3, mined: 1.0, random: 0.1146, prior: 0.0968 }, { k: 5, mined: 0.6, random: 0.0625, prior: 0.0968 },
  { k: 8, mined: 0.375, random: 0.0781, prior: 0.0968 }, { k: 12, mined: 0.25, random: 0.0703, prior: 0.0968 },
  { k: 16, mined: 0.1875, random: 0.0879, prior: 0.0968 },
];

// Panel C: the three negative-expectation means (g = e^{cos/tau}) and the convergence MAE. The biased and
// oracle means are baked; the debiased bar is recomputed live from them as the tau+ slider moves.
const E_POS_MEAN = 102.763;
const E_BIASED_MEAN = 13.96;        // E_p[g] over all items (the biased in-batch mean)
const E_ORACLE_MEAN = 4.4454;       // E_p-[g] over the true negatives (the target)
const MI_FLOOR = 0.0067;            // the e^{-1/tau} clamp
const DEBIASED_CONVERGENCE = [
  { N: 4, biased_mae: 10.0283, debiased_mae: 9.264 }, { N: 8, biased_mae: 9.2352, debiased_mae: 5.9759 },
  { N: 16, biased_mae: 9.9674, debiased_mae: 5.4462 }, { N: 32, biased_mae: 9.9757, debiased_mae: 4.2591 },
  { N: 64, biased_mae: 9.3045, debiased_mae: 2.9736 }, { N: 128, biased_mae: 9.3612, debiased_mae: 2.225 },
  { N: 256, biased_mae: 9.4858, debiased_mae: 1.6328 }, { N: 512, biased_mae: 9.4194, debiased_mae: 1.1514 },
];

// Panel D: the ANCE staleness decay and the refresh-interval tradeoff. MEASURED; TS recomputes the knee.
const STALENESS_CURVE = [
  { t: 0, overlap: 1.0, stale_gold_r1: 1.0, fresh_gold_r1: 1.0 },
  { t: 1, overlap: 0.9844, stale_gold_r1: 1.0, fresh_gold_r1: 1.0 },
  { t: 2, overlap: 0.974, stale_gold_r1: 1.0, fresh_gold_r1: 1.0 },
  { t: 4, overlap: 0.9479, stale_gold_r1: 1.0, fresh_gold_r1: 1.0 },
  { t: 8, overlap: 0.8021, stale_gold_r1: 0.9167, fresh_gold_r1: 1.0 },
  { t: 16, overlap: 0.3542, stale_gold_r1: 0.0833, fresh_gold_r1: 1.0 },
  { t: 32, overlap: 0.1771, stale_gold_r1: 0.0, fresh_gold_r1: 1.0 },
  { t: 64, overlap: 0.1458, stale_gold_r1: 0.0, fresh_gold_r1: 1.0 },
];
const REFRESH_TRADEOFF = [
  { R: 1, avg_staleness: 1.0, reencode_cost: 1.0 }, { R: 2, avg_staleness: 0.994, reencode_cost: 0.5 },
  { R: 4, avg_staleness: 0.9849, reencode_cost: 0.25 }, { R: 8, avg_staleness: 0.9653, reencode_cost: 0.125 },
  { R: 16, avg_staleness: 0.8696, reencode_cost: 0.0625 }, { R: 32, avg_staleness: 0.5023, reencode_cost: 0.0312 },
];

const HARD_COLOR = '#7C3AED';        // same-sector hard negatives / mined
const EASY_COLOR = 'var(--color-accent)';
const ORACLE_COLOR = '#5fa873';      // the true-negative target
const PRIOR_COLOR = '#6a8caf';
const MUTED = 'var(--color-text-secondary)';

// round trig-derived coordinates so SSR (Node) and client serialize identical strings (hydration parity).
const r2 = (v: number) => Math.round(v * 100) / 100;

// closed-form softmax over an array at temperature tau (the gradient weights over candidates).
function softmax(xs: number[], tau: number): number[] {
  const z = xs.map((x) => x / tau);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
// the debiased estimator (E_p - tau+ E_p+)/(1 - tau+), floored at e^{-1/tau} — Movement 3's closed form.
function debiasedMean(tauPlus: number): number {
  if (tauPlus >= 1) return MI_FLOOR;
  return Math.max((E_BIASED_MEAN - tauPlus * E_POS_MEAN) / (1 - tauPlus), MI_FLOOR);
}

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
      <span style={{ minWidth: '12rem' }}>{label} = <strong>{fmt(value)}</strong></span>
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

// ===== Panel A — the mixed negative batch, bars sized by gradient weight =============================
function GradientBars({ weights }: { weights: number[] }) {
  const pw = 640, ph = 300, pad = 44;
  const n = weights.length;
  const gap = (pw - 2 * pad) / n;
  const bw = gap * 0.74;
  const wMax = Math.max(...weights);
  const fy = (w: number) => ph - pad - (ph - 2 * pad) * (w / wMax);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="the anchor's negative batch, each bar sized by its gradient weight" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">negatives, hardest (nearest) → easiest (near-orthogonal)</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>gradient weight pᵢ</text>
      {weights.map((w, i) => {
        const x = pad + gap * i + (gap - bw) / 2;
        const hard = NEG_SAME_SECTOR[i];
        return <rect key={i} x={r2(x)} y={r2(fy(w))} width={r2(bw)} height={r2(ph - pad - fy(w))} fill={hard ? HARD_COLOR : EASY_COLOR} opacity={hard ? 0.95 : 0.5} rx={1.2} />;
      })}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <rect x={pw - pad - 150} y={pad - 2} width={11} height={11} fill={HARD_COLOR} opacity={0.95} rx={1.5} />
        <text x={pw - pad - 135} y={pad + 7} fill={MUTED}>same-sector hard negatives</text>
        <rect x={pw - pad - 150} y={pad + 14} width={11} height={11} fill={EASY_COLOR} opacity={0.5} rx={1.5} />
        <text x={pw - pad - 135} y={pad + 23} fill={MUTED}>cross-sector easy negatives</text>
      </g>
    </svg>
  );
}

// ===== Panel B — the false-negative rate as the mining radius tightens ==============================
function FalseNegativePlot({ idx }: { idx: number }) {
  const pw = 620, ph = 320, pad = 50;
  const n = TAU_PLUS_CURVE.length;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * v;
  const mined = TAU_PLUS_CURVE.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.mined).toFixed(1)).join(' ');
  const rand = TAU_PLUS_CURVE.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.random).toFixed(1)).join(' ');
  const prior = CLASS_PRIOR_TAU_PLUS;
  // the contamination band between the mined curve and the random curve.
  const band = TAU_PLUS_CURVE.map((d, i) => `${fx(i).toFixed(1)} ${fy(d.mined).toFixed(1)}`).join(' L ')
    + ' L ' + [...TAU_PLUS_CURVE].reverse().map((d, j) => `${fx(n - 1 - j).toFixed(1)} ${fy(d.random).toFixed(1)}`).join(' L ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">{v.toFixed(2)}</text>))}
      {TAU_PLUS_CURVE.map((d, i) => (<text key={i} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">{d.k}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">mining depth k (tighter net ← → wider net)</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>false-negative rate τ⁺</text>
      <path d={`M ${band} Z`} fill={HARD_COLOR} opacity={0.1} stroke="none" />
      {/* class prior reference */}
      <line x1={pad} y1={fy(prior)} x2={pw - pad} y2={fy(prior)} stroke={PRIOR_COLOR} strokeWidth={1.1} strokeDasharray="5 3" opacity={0.8} />
      <text x={pw - pad} y={fy(prior) - 5} textAnchor="end" fontSize={9} fill={PRIOR_COLOR} fontFamily="var(--font-sans)">class prior τ⁺ = {prior.toFixed(3)}</text>
      <path d={mined} fill="none" stroke={HARD_COLOR} strokeWidth={2.8} />
      <path d={rand} fill="none" stroke={EASY_COLOR} strokeWidth={2} />
      {TAU_PLUS_CURVE.map((d, i) => (<circle key={`m${i}`} cx={fx(i)} cy={fy(d.mined)} r={i === idx ? 5.5 : 3} fill={HARD_COLOR} opacity={i === idx ? 1 : 0.7} />))}
      {TAU_PLUS_CURVE.map((d, i) => (<circle key={`r${i}`} cx={fx(i)} cy={fy(d.random)} r={i === idx ? 5 : 2.5} fill={EASY_COLOR} opacity={i === idx ? 1 : 0.6} />))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={pad + 12} y1={pad + 4} x2={pad + 28} y2={pad + 4} stroke={HARD_COLOR} strokeWidth={2.8} />
        <text x={pad + 32} y={pad + 7} fill={MUTED}>mined (nearest neighbors)</text>
        <line x1={pad + 12} y1={pad + 20} x2={pad + 28} y2={pad + 20} stroke={EASY_COLOR} strokeWidth={2} />
        <text x={pad + 32} y={pad + 23} fill={MUTED}>random sampling</text>
      </g>
    </svg>
  );
}

// ===== Panel C — biased / true / debiased convergence + the live debiased bar =======================
function ConvergencePlot({ idx }: { idx: number }) {
  const pw = 380, ph = 300, pad = 46;
  const n = DEBIASED_CONVERGENCE.length;
  const yMax = 11;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * v / yMax;
  const biased = DEBIASED_CONVERGENCE.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.biased_mae).toFixed(1)).join(' ');
  const debiased = DEBIASED_CONVERGENCE.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.debiased_mae).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 5, 10].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">{v}</text>))}
      {DEBIASED_CONVERGENCE.map((d, i) => (<text key={i} x={fx(i)} y={ph - pad + 13} textAnchor="middle" fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">{d.N}</text>))}
      <text x={pw / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">sample size N</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>mean abs. error vs oracle</text>
      <path d={biased} fill="none" stroke={EASY_COLOR} strokeWidth={2.6} />
      <path d={debiased} fill="none" stroke={ORACLE_COLOR} strokeWidth={2.6} />
      {DEBIASED_CONVERGENCE.map((d, i) => (<circle key={`b${i}`} cx={fx(i)} cy={fy(d.biased_mae)} r={i === idx ? 5 : 2.5} fill={EASY_COLOR} opacity={i === idx ? 1 : 0.6} />))}
      {DEBIASED_CONVERGENCE.map((d, i) => (<circle key={`d${i}`} cx={fx(i)} cy={fy(d.debiased_mae)} r={i === idx ? 5 : 2.5} fill={ORACLE_COLOR} opacity={i === idx ? 1 : 0.6} />))}
      <g fontFamily="var(--font-sans)" fontSize={9}>
        <line x1={pad + 10} y1={pad + 2} x2={pad + 26} y2={pad + 2} stroke={EASY_COLOR} strokeWidth={2.6} />
        <text x={pad + 30} y={pad + 5} fill={MUTED}>biased (in-batch)</text>
        <line x1={pad + 10} y1={pad + 17} x2={pad + 26} y2={pad + 17} stroke={ORACLE_COLOR} strokeWidth={2.6} />
        <text x={pad + 30} y={pad + 20} fill={MUTED}>debiased</text>
      </g>
    </svg>
  );
}

function EstimatorBars({ tauPlus }: { tauPlus: number }) {
  const pw = 280, ph = 300, pad = 44;
  const yMax = 16;
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * Math.min(v, yMax) / yMax;
  const debiased = debiasedMean(tauPlus);
  const bars = [
    { label: 'biased', val: E_BIASED_MEAN, color: EASY_COLOR },
    { label: 'debiased', val: debiased, color: HARD_COLOR },
    { label: 'oracle', val: E_ORACLE_MEAN, color: ORACLE_COLOR },
  ];
  const bw = 46, gap = (pw - 2 * pad - 3 * bw) / 2;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 5, 10, 15].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">{v}</text>))}
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>negative expectation E[g]</text>
      {/* oracle target line */}
      <line x1={pad} y1={fy(E_ORACLE_MEAN)} x2={pw - pad} y2={fy(E_ORACLE_MEAN)} stroke={ORACLE_COLOR} strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
      {bars.map((b, i) => {
        const x = pad + i * (bw + gap);
        return (
          <g key={i}>
            <rect x={r2(x)} y={r2(fy(b.val))} width={bw} height={r2(ph - pad - fy(b.val))} fill={b.color} opacity={0.82} rx={2} />
            <text x={x + bw / 2} y={ph - pad + 14} textAnchor="middle" fontSize={9.5} fill={MUTED} fontFamily="var(--font-sans)">{b.label}</text>
            <text x={x + bw / 2} y={r2(fy(b.val)) - 5} textAnchor="middle" fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">{b.val.toFixed(2)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ===== Panel D — ANCE staleness decay + the refresh-cost knee =======================================
function StalenessPlot({ idx }: { idx: number }) {
  const pw = 380, ph = 300, pad = 44;
  const n = STALENESS_CURVE.length;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * v;
  const overlap = STALENESS_CURVE.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.overlap).toFixed(1)).join(' ');
  const staleR = STALENESS_CURVE.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.stale_gold_r1).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {STALENESS_CURVE.map((d, i) => (<text key={i} x={fx(i)} y={ph - pad + 13} textAnchor="middle" fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">{d.t}</text>))}
      <text x={pw / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">steps since index refresh</text>
      {/* fresh ceiling at 1.0 */}
      <line x1={pad} y1={fy(1)} x2={pw - pad} y2={fy(1)} stroke={ORACLE_COLOR} strokeWidth={1} strokeDasharray="4 3" opacity={0.6} />
      <text x={pw - pad} y={fy(1) + 11} textAnchor="end" fontSize={8.5} fill={ORACLE_COLOR} fontFamily="var(--font-sans)">fresh gold r@1 = 1.0</text>
      <path d={overlap} fill="none" stroke={HARD_COLOR} strokeWidth={2.8} />
      <path d={staleR} fill="none" stroke={EASY_COLOR} strokeWidth={2} strokeDasharray="5 3" />
      {STALENESS_CURVE.map((d, i) => (<circle key={i} cx={fx(i)} cy={fy(d.overlap)} r={i === idx ? 5.5 : 3} fill={HARD_COLOR} opacity={i === idx ? 1 : 0.7} />))}
      <g fontFamily="var(--font-sans)" fontSize={9}>
        <line x1={pad + 10} y1={pad + 2} x2={pad + 26} y2={pad + 2} stroke={HARD_COLOR} strokeWidth={2.8} />
        <text x={pad + 30} y={pad + 5} fill={MUTED}>index overlap</text>
        <line x1={pad + 10} y1={pad + 17} x2={pad + 26} y2={pad + 17} stroke={EASY_COLOR} strokeWidth={2} strokeDasharray="5 3" />
        <text x={pad + 30} y={pad + 20} fill={MUTED}>stale gold r@1</text>
      </g>
    </svg>
  );
}

function RefreshCostPlot({ lambda }: { lambda: number }) {
  const pw = 380, ph = 300, pad = 44;
  const rows = REFRESH_TRADEOFF.map((d) => ({ ...d, total: lambda * (1 - d.avg_staleness) + d.reencode_cost }));
  const n = rows.length;
  const yMax = Math.max(1.05, ...rows.map((r) => r.total));
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (v: number) => ph - pad - (ph - 2 * pad) * v / yMax;
  const total = rows.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.total).toFixed(1)).join(' ');
  const cost = rows.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.reencode_cost).toFixed(1)).join(' ');
  const stalePen = rows.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(lambda * (1 - d.avg_staleness)).toFixed(1)).join(' ');
  const kneeIdx = rows.reduce((best, r, i) => (r.total < rows[best].total ? i : best), 0);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {REFRESH_TRADEOFF.map((d, i) => (<text key={i} x={fx(i)} y={ph - pad + 13} textAnchor="middle" fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">{d.R}</text>))}
      <text x={pw / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">refresh interval R (steps between rebuilds)</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>cost (lower is better)</text>
      <path d={cost} fill="none" stroke={PRIOR_COLOR} strokeWidth={1.6} strokeDasharray="4 3" />
      <path d={stalePen} fill="none" stroke={HARD_COLOR} strokeWidth={1.6} strokeDasharray="4 3" />
      <path d={total} fill="none" stroke="var(--color-text)" strokeWidth={2.8} />
      {rows.map((d, i) => (<circle key={i} cx={fx(i)} cy={fy(d.total)} r={i === kneeIdx ? 6 : 3} fill={i === kneeIdx ? ORACLE_COLOR : 'var(--color-text)'} opacity={i === kneeIdx ? 1 : 0.7} />))}
      <text x={fx(kneeIdx)} y={fy(rows[kneeIdx].total) - 9} textAnchor="middle" fontSize={9.5} fill={ORACLE_COLOR} fontFamily="var(--font-sans)">R* = {rows[kneeIdx].R}</text>
      <g fontFamily="var(--font-sans)" fontSize={9}>
        <line x1={pad + 10} y1={pad + 2} x2={pad + 26} y2={pad + 2} stroke="var(--color-text)" strokeWidth={2.8} />
        <text x={pad + 30} y={pad + 5} fill={MUTED}>total cost</text>
        <line x1={pad + 10} y1={pad + 16} x2={pad + 26} y2={pad + 16} stroke={PRIOR_COLOR} strokeWidth={1.6} strokeDasharray="4 3" />
        <text x={pad + 30} y={pad + 19} fill={MUTED}>re-encode 1/R</text>
        <line x1={pad + 10} y1={pad + 30} x2={pad + 26} y2={pad + 30} stroke={HARD_COLOR} strokeWidth={1.6} strokeDasharray="4 3" />
        <text x={pad + 30} y={pad + 33} fill={MUTED}>λ·staleness penalty</text>
      </g>
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'gradient' | 'falseneg' | 'debiased' | 'staleness';

export default memo(function NegativeSamplingLaboratory() {
  const [panel, setPanel] = useState<Panel>('gradient');
  const [tau, setTau] = useState(GRAD_TAU);
  const [kIdx, setKIdx] = useState(0);
  const [nIdx, setNIdx] = useState(3);
  const [tauPlus, setTauPlus] = useState(CLASS_PRIOR_TAU_PLUS);
  const [tIdx, setTIdx] = useState(4);
  const [lambda, setLambda] = useState(1.0);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'gradient'
        ? '\\frac{\\partial \\mathcal{L}}{\\partial s_i^-} = \\frac{p_i}{\\tau}, \\qquad \\text{hard share} = \\sum_{i \\,\\in\\, \\text{hard}} p_i, \\quad p_i = \\frac{e^{s_i/\\tau}}{\\sum_j e^{s_j/\\tau}}'
        : panel === 'falseneg'
          ? '\\tau^+ = \\Pr\\!\\left[\\, \\text{a mined ``negative\'\' is actually a positive} \\,\\right]'
          : panel === 'debiased'
            ? '\\mathbb{E}_{p^-}[g] = \\frac{\\mathbb{E}_{p}[g] - \\tau^+\\, \\mathbb{E}_{p^+}[g]}{1 - \\tau^+}, \\qquad g(x) = e^{\\,s(q,x)/\\tau}'
            : '\\text{overlap}(t) \\downarrow \\text{ as } (t - t_{\\text{refresh}}) \\uparrow, \\qquad \\text{cost}(R) = \\lambda\\,(1 - \\overline{\\text{overlap}}) + \\tfrac{1}{R}';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  // Panel A: the gradient weights and the hard share are the closed form softmax(NEG_COS / tau).
  const weights = softmax(NEG_COS, tau);
  const hardShare = weights.reduce((a, w, i) => a + (NEG_SAME_SECTOR[i] ? w : 0), 0);
  const fn = TAU_PLUS_CURVE[kIdx];
  const conv = DEBIASED_CONVERGENCE[nIdx];
  const debiased = debiasedMean(tauPlus);
  const st = STALENESS_CURVE[tIdx];

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'gradient')} onClick={() => setPanel('gradient')}>A · why hard negatives</button>
        <button style={pill(panel === 'falseneg')} onClick={() => setPanel('falseneg')}>B · false negatives</button>
        <button style={pill(panel === 'debiased')} onClick={() => setPanel('debiased')}>C · the debiased estimator</button>
        <button style={pill(panel === 'staleness')} onClick={() => setPanel('staleness')}>D · ANCE staleness</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'gradient' && (
        <div>
          <Slider label="temperature τ" value={tau} min={0.02} max={1} step={0.01} onChange={setTau} fmt={(v) => v.toFixed(2)} />
          <GradientBars weights={weights} />
          <div style={{ display: 'flex', gap: '2rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            <Readout label="hard gradient share Σ pᵢ" value={hardShare.toFixed(3)} accent />
            <Readout label="hard count fraction" value={HARD_COUNT_FRACTION.toFixed(3)} />
            <Readout label="hard-batch loss / random-batch loss" value={`${BATCH_LOSS.hard_loss.toFixed(2)} / ${BATCH_LOSS.random_loss.toFixed(2)}`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Each negative repels the query with a force equal to its softmax weight pᵢ = e<sup>sᵢ/τ</sup> / Σ. The four
            same-sector hard negatives are only {(HARD_COUNT_FRACTION * 100).toFixed(0)}% of the batch, but because the weight is
            exponential in similarity they carry the share above — {(hardShare * 100).toFixed(0)}% of the gradient at this τ, rising
            toward 100% as τ → 0. And a hard-mined batch is harder to classify, so its loss ({BATCH_LOSS.hard_loss.toFixed(2)}) — the
            per-step gradient magnitude — far exceeds a random batch's ({BATCH_LOSS.random_loss.toFixed(2)}). That is why we mine.
          </p>
        </div>
      )}

      {panel === 'falseneg' && (
        <div>
          <Slider label="mining depth k" value={kIdx} min={0} max={TAU_PLUS_CURVE.length - 1} step={1} onChange={setKIdx} fmt={(v) => `${TAU_PLUS_CURVE[v].k}`} />
          <FalseNegativePlot idx={kIdx} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="mined τ⁺ at k" value={fn.mined.toFixed(3)} accent />
            <Readout label="random τ⁺ at k" value={fn.random.toFixed(3)} />
            <Readout label="class prior τ⁺" value={CLASS_PRIOR_TAU_PLUS.toFixed(3)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Mining samples <em>near</em> the anchor — but near the anchor is where the unlabeled true positives live. At k = 1 the
            nearest neighbor is the anchor's own company {(fn.k === 1 ? '' : '(see k = 1)')}: the mined "negative" is a false
            negative with probability {TAU_PLUS_CURVE[0].mined.toFixed(2)}. Random sampling, by contrast, hits accidental positives
            only at the global class prior τ⁺ = {CLASS_PRIOR_TAU_PLUS.toFixed(3)}. The shaded band is the contamination hard mining
            adds — the false-negative problem the debiased estimator exists to correct.
          </p>
        </div>
      )}

      {panel === 'debiased' && (
        <div>
          <Slider label="assumed class prior τ⁺" value={tauPlus} min={0} max={0.3} step={0.005} onChange={setTauPlus} fmt={(v) => v.toFixed(3)} />
          <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: '0.8rem', alignItems: 'center' }}>
            <ConvergencePlot idx={nIdx} />
            <EstimatorBars tauPlus={tauPlus} />
          </div>
          <Slider label="sample size N (left plot)" value={nIdx} min={0} max={DEBIASED_CONVERGENCE.length - 1} step={1} onChange={setNIdx} fmt={(v) => `${DEBIASED_CONVERGENCE[v].N}`} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.2rem', flexWrap: 'wrap' }}>
            <Readout label="debiased estimate" value={debiased.toFixed(3)} accent />
            <Readout label="oracle E_p⁻[g]" value={E_ORACLE_MEAN.toFixed(3)} />
            <Readout label="biased / debiased MAE at N" value={`${conv.biased_mae.toFixed(2)} / ${conv.debiased_mae.toFixed(2)}`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The biased in-batch mean averages g over <em>all</em> sampled items, accidental positives included, so its error vs the
            true-negative oracle plateaus at the contamination bias no matter how many samples (the blue curve). The debiased
            estimator (E<sub>p</sub> − τ⁺E<sub>p⁺</sub>)/(1 − τ⁺) — recomputed live in the bar as you slide τ⁺ — converges to the
            oracle (green) and, at the true class prior τ⁺ = {CLASS_PRIOR_TAU_PLUS.toFixed(3)}, meets it exactly. At τ⁺ = 0 it
            collapses back to the biased mean. That exact recovery is the topic's one theorem.
          </p>
        </div>
      )}

      {panel === 'staleness' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem', alignItems: 'center' }}>
            <StalenessPlot idx={tIdx} />
            <RefreshCostPlot lambda={lambda} />
          </div>
          <Slider label="steps since refresh (left)" value={tIdx} min={0} max={STALENESS_CURVE.length - 1} step={1} onChange={setTIdx} fmt={(v) => `${STALENESS_CURVE[v].t}`} />
          <Slider label="staleness weight λ (right)" value={lambda} min={0} max={4} step={0.1} onChange={setLambda} fmt={(v) => v.toFixed(1)} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.2rem', flexWrap: 'wrap' }}>
            <Readout label="index overlap at t" value={st.overlap.toFixed(3)} accent />
            <Readout label="stale gold r@1 at t" value={st.stale_gold_r1.toFixed(3)} />
            <Readout label="fresh gold r@1" value={st.fresh_gold_r1.toFixed(3)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            ANCE mines hard negatives from an index frozen at the last refresh. As the encoder drifts, the frozen index's mined set
            overlaps the fresh encoder's less and less (left), and its gold recall@1 falls far below the fresh ceiling — a stale
            index surfaces stale negatives. The refresh interval R (right) trades that staleness penalty against the re-encode cost
            1/R; the cost-minimizing R* shifts as the staleness weight λ grows. There is no convergence bound here — the staleness
            curve is measured, not derived. (An isometric re-encoding of a <em>refreshed</em> index, by contrast, changes nothing.)
          </p>
        </div>
      )}
    </div>
  );
});
