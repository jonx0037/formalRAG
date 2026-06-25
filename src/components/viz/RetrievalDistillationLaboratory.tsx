import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Retrieval-Distillation Laboratory — three panels for the `retrieval-distillation` topic:
 *   A. Translation-invariance. The all-pairs MarginMSE reduces to the centered Frobenius distance
 *      2 n_d ||SC - TC||^2, so a per-query offset T -> T + b 1^T is INVISIBLE to it. A miscalibration
 *      slider alpha scales the teacher's per-query offset (alpha=1 = the actual teacher): the pointwise
 *      loss of the fixed margin student is a parabola that explodes, while its margin loss stays flat —
 *      both recomputed LIVE in TS. The heatmaps show the teacher's rows shifting with alpha while the
 *      centered teacher does not move.
 *   B. Rank-d fidelity. The pointwise student best_rank_d(T) wastes budget on the teacher's per-query
 *      level (its huge top singular value); the margin student best_rank_d(TC) spends every dimension on
 *      ranking, so margin recall@1 >= pointwise recall@1 at every rank, with the biggest gap at the
 *      restricted D_STAGE. The recall curve is baked; the TC tail-energy (the ranking compressibility)
 *      is recomputed LIVE from the singular values as the rank slider moves.
 *   C. Dark knowledge and the cost payoff. The teacher's margins on the mined hard-negative pairs are
 *      GRADED (a real spread) where a binary label's margin is the constant 1 — yet on this clean
 *      in-sample toy the binary target compresses to low rank at least as well (the soft-beats-hard
 *      advantage is a generalization phenomenon). The corpus slider recomputes the student-vs-teacher
 *      inference cost and the speedup LIVE.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): PW_LOSS_BASE, PW_LOSS_LIN, PW_LOSS_QUAD, MARGIN_LOSS, HEAT_T,
 * HEAT_TC, HEAT_B, SINGULAR_VALUES_T, SINGULAR_VALUES_TC, RANK_RECALL, SOFT_VS_HARD, HARD_PAIR_MARGINS,
 * and the cost constants are mirrored TO THE DECIMAL from
 * notebooks/retrieval-distillation/retrieval_distillation.py (viz_constants()). The lab recomputes only
 * CLOSED FORM in TS — Panel A's pointwise parabola PW_BASE - 2a*PW_LIN + a^2*PW_QUAD and the flat margin
 * line, Panel B's TC tail-energy sum_{l>d} sigma_l^2 / sum sigma_l^2, and Panel C's cost/speedup — exactly
 * as the source's tests assert (test_margin_reduction, test_translation_invariance,
 * test_margin_beats_pointwise_at_restricted_rank, test_cost_payoff). Every MEASURED number is baked.
 * Change a number here -> change it there, and re-run the notebook. Sliders only (no d3 drag).
 */

// --- baked from viz_constants() -------------------------------------------------------
const D_STAGE = 3;
const TEACHER_R1 = 1.0;

// Panel A: translation-invariance. The fixed margin student's two losses under the offset slider alpha:
// margin loss flat; pointwise loss PW_BASE - 2a*PW_LIN + a^2*PW_QUAD. alpha=1 is the actual teacher.
const PW_LOSS_BASE = 11.14;
const PW_LOSS_LIN = 23.31;
const PW_LOSS_QUAD = 1846.463;
const MARGIN_LOSS = 114.2476;
const ALPHA_ACTUAL = 1.0;
const ALPHA_MAX = 2.0;
const HEAT_T = [
  [0.919, 0.143, 0.008, -0.029, 0.247, -0.224, 0.027, -0.092],
  [0.119, 0.656, 0.044, 0.027, 0.128, 0.066, -0.108, 0.068],
  [0.039, 0.104, 0.831, 0.115, 0.026, -0.19, 0.075, 0.0],
  [-0.063, 0.084, 0.064, 0.715, 0.017, -0.187, 0.352, 0.018],
  [0.063, 0.085, 0.017, -0.098, 0.958, -0.017, 0.072, -0.081],
  [-0.304, -0.077, 0.145, 0.084, 0.063, 1.055, 0.007, 0.027],
  [0.048, 0.14, -0.036, 0.034, -0.175, 0.148, 0.568, 0.273],
  [0.166, 0.063, 0.088, 0.239, 0.071, 0.012, -0.167, 0.528],
];
const HEAT_TC = [
  [0.794, 0.018, -0.117, -0.154, 0.122, -0.349, -0.098, -0.217],
  [-0.006, 0.531, -0.081, -0.098, 0.003, -0.059, -0.233, -0.057],
  [-0.086, -0.021, 0.706, -0.01, -0.099, -0.315, -0.05, -0.125],
  [-0.188, -0.041, -0.061, 0.59, -0.108, -0.312, 0.227, -0.107],
  [-0.062, -0.04, -0.108, -0.223, 0.833, -0.142, -0.053, -0.206],
  [-0.429, -0.202, 0.02, -0.041, -0.062, 0.93, -0.118, -0.098],
  [-0.077, 0.015, -0.161, -0.091, -0.3, 0.023, 0.443, 0.148],
  [0.041, -0.062, -0.037, 0.114, -0.054, -0.113, -0.292, 0.403],
];
const HEAT_B = [-2.406, 3.408, 2.246, -2.875, -0.251, -2.14, 1.229, -2.942];

// Panel B: the two spectra (T carries a huge top singular value = the per-query level; TC is flat = all
// ranking) and the baked recall-vs-rank curve. The TC tail-energy is recomputed live.
const SINGULAR_VALUES_T = [42.4747, 2.2468, 2.0725, 1.9223, 1.7463, 1.3549, 1.0596, 0.984];
const SINGULAR_VALUES_TC = [2.2468, 2.0734, 1.9448, 1.7618, 1.3842, 1.0624, 0.9959, 0.0];
const RANK_RECALL = [
  { d: 1, pointwise_r1: 0.2188, margin_r1: 0.25, teacher_r1: 1.0 },
  { d: 2, pointwise_r1: 0.2812, margin_r1: 0.4375, teacher_r1: 1.0 },
  { d: 3, pointwise_r1: 0.5, margin_r1: 0.6875, teacher_r1: 1.0 },
  { d: 4, pointwise_r1: 0.6875, margin_r1: 0.7188, teacher_r1: 1.0 },
  { d: 5, pointwise_r1: 0.7188, margin_r1: 0.8438, teacher_r1: 1.0 },
  { d: 6, pointwise_r1: 0.8438, margin_r1: 0.9062, teacher_r1: 1.0 },
  { d: 7, pointwise_r1: 0.875, margin_r1: 1.0, teacher_r1: 1.0 },
  { d: 8, pointwise_r1: 1.0, margin_r1: 1.0, teacher_r1: 1.0 },
];

// Panel C: dark knowledge (soft vs hard recall by rank) + the teacher's graded margins on the mined hard
// pairs (binary margin = 1) + the inference cost constants.
const SOFT_VS_HARD = [
  { d: 1, soft_r1: 0.25, hard_r1: 0.25 }, { d: 2, soft_r1: 0.4375, hard_r1: 0.625 },
  { d: 3, soft_r1: 0.6875, hard_r1: 0.875 }, { d: 4, soft_r1: 0.7188, hard_r1: 0.875 },
  { d: 5, soft_r1: 0.8438, hard_r1: 0.875 }, { d: 6, soft_r1: 0.9062, hard_r1: 1.0 },
  { d: 7, soft_r1: 1.0, hard_r1: 1.0 }, { d: 8, soft_r1: 1.0, hard_r1: 1.0 },
];
const HARD_PAIR_MARGINS = [
  0.775, 0.748, 0.886, 1.292, 0.537, 0.569, 0.843, 0.542, 0.715, 0.667, 0.776, 0.728, 0.651, 0.262,
  0.741, 0.781, 0.975, 0.897, 0.466, 0.85, 0.992, 0.642, 0.989, 0.78, 0.295, 0.192, 0.301, 0.116,
  0.694, 0.259, 0.672, 0.599,
];
const C_RETRIEVE = 1.0;
const C_CE = 25.0;

const MARGIN_COLOR = '#7C3AED';      // the margin student / soft target
const POINT_COLOR = 'var(--color-accent)';   // the pointwise student / hard target
const TEACHER_COLOR = '#5fa873';     // the teacher ceiling
const MUTED = 'var(--color-text-secondary)';

// round trig-derived coordinates so SSR (Node) and client serialize identical strings (hydration parity).
const r2 = (v: number) => Math.round(v * 100) / 100;

// closed-form: the pointwise loss of the fixed margin student as the offset multiplier alpha grows.
function pointwiseLoss(alpha: number): number {
  return PW_LOSS_BASE - 2 * alpha * PW_LOSS_LIN + alpha * alpha * PW_LOSS_QUAD;
}
// closed-form: the relative reconstruction error of a target's ranking structure at rank d (tail energy).
function tailEnergy(sv: number[], d: number): number {
  const sq = sv.map((s) => s * s);
  const total = sq.reduce((a, b) => a + b, 0);
  const tail = sq.slice(d).reduce((a, b) => a + b, 0);
  return total > 0 ? tail / total : 0;
}
// a diverging heat color around zero: negative -> muted blue, positive -> violet, magnitude -> saturation.
function heatColor(v: number, vmax: number): string {
  const t = Math.max(-1, Math.min(1, v / Math.max(vmax, 1e-9)));
  if (t >= 0) return `rgba(124, 58, 237, ${r2(0.12 + 0.83 * t)})`;     // violet
  return `rgba(96, 140, 175, ${r2(0.12 + 0.83 * -t)})`;                // muted blue
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
      <span style={{ minWidth: '13rem' }}>{label} = <strong>{fmt(value)}</strong></span>
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

// ===== Panel A — translation-invariance: the loss curves and the two heatmaps =======================
function LossCurves({ alpha }: { alpha: number }) {
  const pw = 360, ph = 240, padL = 52, padR = 14, padT = 16, padB = 34;
  const yMax = pointwiseLoss(ALPHA_MAX) * 1.05;
  const fx = (a: number) => padL + (pw - padL - padR) * (a / ALPHA_MAX);
  const fy = (v: number) => ph - padB - (ph - padT - padB) * (v / yMax);
  const STEPS = 40;
  const pwPts = Array.from({ length: STEPS + 1 }, (_, i) => {
    const a = (ALPHA_MAX * i) / STEPS;
    return `${r2(fx(a))},${r2(fy(pointwiseLoss(a)))}`;
  }).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="pointwise loss explodes with miscalibration while the margin loss stays flat" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={padL} y1={ph - padB} x2={pw - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={padL} y1={padT} x2={padL} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <text x={(padL + pw - padR) / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">miscalibration α (offset multiplier)</text>
      <text x={13} y={(padT + ph - padB) / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${(padT + ph - padB) / 2})`}>distillation loss</text>
      {/* flat margin line */}
      <line x1={padL} y1={r2(fy(MARGIN_LOSS))} x2={pw - padR} y2={r2(fy(MARGIN_LOSS))} stroke={MARGIN_COLOR} strokeWidth={2.2} />
      {/* pointwise parabola */}
      <polyline points={pwPts} fill="none" stroke={POINT_COLOR} strokeWidth={2.2} />
      {/* the current alpha marker */}
      <line x1={r2(fx(alpha))} y1={padT} x2={r2(fx(alpha))} y2={ph - padB} stroke={MUTED} strokeWidth={1} strokeDasharray="3 3" />
      <circle cx={r2(fx(alpha))} cy={r2(fy(pointwiseLoss(alpha)))} r={3.4} fill={POINT_COLOR} />
      <circle cx={r2(fx(alpha))} cy={r2(fy(MARGIN_LOSS))} r={3.4} fill={MARGIN_COLOR} />
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <rect x={pw - padR - 132} y={padT} width={11} height={11} fill={POINT_COLOR} rx={1.5} />
        <text x={pw - padR - 117} y={padT + 9} fill={MUTED}>pointwise loss (grows)</text>
        <rect x={pw - padR - 132} y={padT + 15} width={11} height={11} fill={MARGIN_COLOR} rx={1.5} />
        <text x={pw - padR - 117} y={padT + 24} fill={MUTED}>margin loss (flat)</text>
      </g>
    </svg>
  );
}

function Heatmap({ rows, title }: { rows: number[][]; title: string }) {
  const n = rows.length, m = rows[0].length;
  const cell = 19, padL = 6, padT = 18;
  const w = padL + m * cell + 6, h = padT + n * cell + 6;
  let vmax = 0;
  for (const row of rows) for (const v of row) vmax = Math.max(vmax, Math.abs(v));
  return (
    <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={title} style={{ width: '100%', height: 'auto', display: 'block', maxWidth: '220px' }}>
      <text x={padL} y={11} fontSize={9.5} fill={MUTED} fontFamily="var(--font-sans)">{title}</text>
      {rows.map((row, i) => (
        <g key={i}>
          {row.map((v, j) => (
            <rect key={j} x={padL + j * cell} y={padT + i * cell} width={cell - 1.5} height={cell - 1.5}
              rx={1.5} fill={heatColor(v, vmax)} />
          ))}
        </g>
      ))}
    </svg>
  );
}

// ===== Panel B — recall vs rank, with the live TC tail-energy =======================================
function RecallCurve({ d }: { d: number }) {
  const pw = 460, ph = 260, padL = 44, padR = 14, padT = 16, padB = 34;
  const dMin = RANK_RECALL[0].d, dMax = RANK_RECALL[RANK_RECALL.length - 1].d;
  const fx = (dd: number) => padL + (pw - padL - padR) * ((dd - dMin) / (dMax - dMin));
  const fy = (r: number) => ph - padB - (ph - padT - padB) * r;
  const line = (key: 'pointwise_r1' | 'margin_r1' | 'teacher_r1') =>
    RANK_RECALL.map((row) => `${r2(fx(row.d))},${r2(fy(row[key]))}`).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="recall at 1 vs student rank for pointwise, margin, and teacher" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={padL} y1={ph - padB} x2={pw - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={padL} y1={padT} x2={padL} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((r) => (
        <g key={r}>
          <line x1={padL} y1={r2(fy(r))} x2={pw - padR} y2={r2(fy(r))} stroke="var(--color-border)" strokeWidth={0.4} strokeDasharray="2 3" />
          <text x={padL - 5} y={r2(fy(r)) + 3} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">{r.toFixed(1)}</text>
        </g>
      ))}
      <text x={(padL + pw - padR) / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">student rank d</text>
      <text x={13} y={(padT + ph - padB) / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${(padT + ph - padB) / 2})`}>recall@1</text>
      <line x1={r2(fx(d))} y1={padT} x2={r2(fx(d))} y2={ph - padB} stroke={MUTED} strokeWidth={1} strokeDasharray="3 3" />
      <polyline points={line('teacher_r1')} fill="none" stroke={TEACHER_COLOR} strokeWidth={1.8} strokeDasharray="5 3" />
      <polyline points={line('pointwise_r1')} fill="none" stroke={POINT_COLOR} strokeWidth={2.2} />
      <polyline points={line('margin_r1')} fill="none" stroke={MARGIN_COLOR} strokeWidth={2.2} />
      {RANK_RECALL.map((row) => (
        <g key={row.d}>
          <circle cx={r2(fx(row.d))} cy={r2(fy(row.pointwise_r1))} r={2.6} fill={POINT_COLOR} />
          <circle cx={r2(fx(row.d))} cy={r2(fy(row.margin_r1))} r={2.6} fill={MARGIN_COLOR} />
        </g>
      ))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <rect x={pw - padR - 118} y={padT + 40} width={11} height={11} fill={MARGIN_COLOR} rx={1.5} />
        <text x={pw - padR - 103} y={padT + 49} fill={MUTED}>margin student</text>
        <rect x={pw - padR - 118} y={padT + 55} width={11} height={11} fill={POINT_COLOR} rx={1.5} />
        <text x={pw - padR - 103} y={padT + 64} fill={MUTED}>pointwise student</text>
        <rect x={pw - padR - 118} y={padT + 70} width={11} height={11} fill={TEACHER_COLOR} rx={1.5} />
        <text x={pw - padR - 103} y={padT + 79} fill={MUTED}>teacher ceiling</text>
      </g>
    </svg>
  );
}

function SpectrumBars() {
  const pw = 220, ph = 150, padL = 28, padB = 26, padT = 12;
  const n = SINGULAR_VALUES_T.length;
  const gap = (pw - padL - 6) / n;
  const bw = gap * 0.36;
  const vMax = Math.max(...SINGULAR_VALUES_T);
  const fy = (v: number) => ph - padB - (ph - padT - padB) * (v / vMax);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="singular values of the teacher and the centered teacher" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={padL} y1={ph - padB} x2={pw - 4} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <text x={pw / 2} y={ph - 4} textAnchor="middle" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">singular value index</text>
      {SINGULAR_VALUES_T.map((v, i) => {
        const x = padL + gap * i + gap * 0.12;
        const vc = SINGULAR_VALUES_TC[i];
        return (
          <g key={i}>
            <rect x={r2(x)} y={r2(fy(v))} width={r2(bw)} height={r2(ph - padB - fy(v))} fill={POINT_COLOR} opacity={0.85} rx={1} />
            <rect x={r2(x + bw + 1)} y={r2(fy(vc))} width={r2(bw)} height={r2(ph - padB - fy(vc))} fill={MARGIN_COLOR} opacity={0.9} rx={1} />
          </g>
        );
      })}
      <text x={padL} y={padT + 2} fontSize={9} fill={POINT_COLOR} fontFamily="var(--font-sans)">σ(T): σ₁ = {SINGULAR_VALUES_T[0].toFixed(0)} (the level)</text>
      <text x={padL} y={padT + 13} fontSize={9} fill={MARGIN_COLOR} fontFamily="var(--font-sans)">σ(TC): flat — all ranking</text>
    </svg>
  );
}

// ===== Panel C — dark knowledge (soft vs hard) and the margin strip =================================
function SoftHardBars() {
  const pw = 420, ph = 230, padL = 40, padR = 12, padT = 14, padB = 34;
  const n = SOFT_VS_HARD.length;
  const gap = (pw - padL - padR) / n;
  const bw = gap * 0.34;
  const fy = (r: number) => ph - padB - (ph - padT - padB) * r;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="soft teacher vs hard binary recall by rank" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={padL} y1={ph - padB} x2={pw - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((r) => (
        <text key={r} x={padL - 5} y={r2(fy(r)) + 3} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">{r.toFixed(1)}</text>
      ))}
      <text x={(padL + pw - padR) / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">student rank d</text>
      <text x={13} y={(padT + ph - padB) / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${(padT + ph - padB) / 2})`}>recall@1</text>
      {SOFT_VS_HARD.map((row, i) => {
        const x = padL + gap * i + gap * 0.14;
        return (
          <g key={row.d}>
            <rect x={r2(x)} y={r2(fy(row.soft_r1))} width={r2(bw)} height={r2(ph - padB - fy(row.soft_r1))} fill={MARGIN_COLOR} opacity={0.9} rx={1} />
            <rect x={r2(x + bw + 1.5)} y={r2(fy(row.hard_r1))} width={r2(bw)} height={r2(ph - padB - fy(row.hard_r1))} fill={POINT_COLOR} opacity={0.75} rx={1} />
            <text x={r2(x + bw)} y={ph - padB + 11} textAnchor="middle" fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">{row.d}</text>
          </g>
        );
      })}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <rect x={pw - padR - 150} y={padT} width={11} height={11} fill={MARGIN_COLOR} opacity={0.9} rx={1.5} />
        <text x={pw - padR - 135} y={padT + 9} fill={MUTED}>soft teacher (graded)</text>
        <rect x={pw - padR - 150} y={padT + 15} width={11} height={11} fill={POINT_COLOR} opacity={0.75} rx={1.5} />
        <text x={pw - padR - 135} y={padT + 24} fill={MUTED}>hard binary labels</text>
      </g>
    </svg>
  );
}

function MarginStrip() {
  const pw = 420, ph = 70, padL = 8, padR = 8, padT = 22, padB = 18;
  const vMax = Math.max(...HARD_PAIR_MARGINS) * 1.05;
  const fx = (v: number) => padL + (pw - padL - padR) * (v / vMax);
  const mean = HARD_PAIR_MARGINS.reduce((a, b) => a + b, 0) / HARD_PAIR_MARGINS.length;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="teacher margins on the mined hard-negative pairs" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <text x={padL} y={11} fontSize={9.5} fill={MUTED} fontFamily="var(--font-sans)">teacher margin on each mined hard pair (binary label's margin = 1, the dotted line)</text>
      <line x1={padL} y1={ph - padB} x2={pw - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={r2(fx(1))} y1={padT - 4} x2={r2(fx(1))} y2={ph - padB} stroke={POINT_COLOR} strokeWidth={1} strokeDasharray="3 3" />
      <text x={r2(fx(1)) + 3} y={padT + 4} fontSize={8.5} fill={POINT_COLOR} fontFamily="var(--font-sans)">1</text>
      {HARD_PAIR_MARGINS.map((v, i) => (
        <circle key={i} cx={r2(fx(v))} cy={padT + 18} r={3} fill={MARGIN_COLOR} opacity={0.55} />
      ))}
      <line x1={r2(fx(mean))} y1={padT + 8} x2={r2(fx(mean))} y2={padT + 28} stroke={MARGIN_COLOR} strokeWidth={2} />
      <text x={r2(fx(mean))} y={ph - 4} textAnchor="middle" fontSize={8.5} fill={MARGIN_COLOR} fontFamily="var(--font-sans)">mean {mean.toFixed(2)}</text>
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'invariance' | 'fidelity' | 'darkknowledge';

export default memo(function RetrievalDistillationLaboratory() {
  const [panel, setPanel] = useState<Panel>('invariance');
  const [alpha, setAlpha] = useState(ALPHA_ACTUAL);
  const [dIdx, setDIdx] = useState(D_STAGE - 1);
  const [corpusExp, setCorpusExp] = useState(6);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'invariance'
        ? '\\mathcal{L}(S,T) = 2\\,n_d\\,\\lVert SC - TC\\rVert_F^2, \\qquad T \\mapsto T + b\\,\\mathbf{1}^\\top \\;\\Rightarrow\\; TC \\text{ unchanged}'
        : panel === 'fidelity'
          ? 'S^\\star_{\\text{margin}} = \\operatorname{best\\,rank}_d(TC), \\qquad S^\\star_{\\text{point}} = \\operatorname{best\\,rank}_d(T)'
          : '\\text{soft target } T \\text{ (graded)} \\quad\\text{vs}\\quad \\text{hard label } Y \\in \\{0,1\\}, \\qquad \\text{speedup} = c_{\\text{ce}} / c_{\\text{ret}}';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  // Panel A live: the fixed margin student's two losses at the current alpha.
  const pwNow = pointwiseLoss(alpha);
  const pwInflation = pwNow / PW_LOSS_BASE;
  // Panel A heatmap: the teacher's rows shift by alpha * offset; the centered teacher does not.
  const heatTalpha = HEAT_T.map((row, i) => row.map((v) => v + alpha * HEAT_B[i]));

  // Panel B live: the ranking compressibility (TC tail-energy) at the current rank.
  const rr = RANK_RECALL[dIdx];
  const d = rr.d;
  const tcTail = tailEnergy(SINGULAR_VALUES_TC, d);

  // Panel C live: the inference cost and speedup at the current corpus size.
  const corpus = Math.round(Math.pow(10, corpusExp));
  const studentCost = corpus * C_RETRIEVE;
  const teacherCost = corpus * C_CE;
  const speedup = C_CE / C_RETRIEVE;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'invariance')} onClick={() => setPanel('invariance')}>A · translation-invariance</button>
        <button style={pill(panel === 'fidelity')} onClick={() => setPanel('fidelity')}>B · rank-d fidelity</button>
        <button style={pill(panel === 'darkknowledge')} onClick={() => setPanel('darkknowledge')}>C · dark knowledge & cost</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'invariance' && (
        <div>
          <Slider label="miscalibration α (offset multiplier)" value={alpha} min={0} max={ALPHA_MAX} step={0.02} onChange={setAlpha} fmt={(v) => v.toFixed(2)} />
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '0.8rem', alignItems: 'center' }}>
            <LossCurves alpha={alpha} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <Heatmap rows={heatTalpha} title={`teacher T + α·b·1ᵀ (rows shift)`} />
              <Heatmap rows={HEAT_TC} title="centered teacher TC (invariant)" />
            </div>
          </div>
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="margin loss (flat)" value={MARGIN_LOSS.toFixed(1)} accent />
            <Readout label="pointwise loss" value={pwNow.toFixed(1)} />
            <Readout label="pointwise inflation ×" value={`${pwInflation.toFixed(1)}×`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The all-pairs MarginMSE is the centered Frobenius distance 2nₐ‖SC − TC‖², so a per-query offset α·b
            (the cross-encoder's per-query miscalibration) is <em>invisible</em> to it: the margin loss of the fixed margin
            student stays flat at {MARGIN_LOSS.toFixed(1)} for every α, while the same student's <em>pointwise</em> loss
            inflates {pwInflation.toFixed(1)}× by α = {alpha.toFixed(2)}. The teacher's rows brighten and darken with α; the
            centered teacher does not move. That is why distillation matches margins, not absolute scores.
          </p>
        </div>
      )}

      {panel === 'fidelity' && (
        <div>
          <Slider label="student rank d" value={dIdx} min={0} max={RANK_RECALL.length - 1} step={1} onChange={setDIdx} fmt={() => `${d}`} />
          <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '0.8rem', alignItems: 'center' }}>
            <RecallCurve d={d} />
            <SpectrumBars />
          </div>
          <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="margin recall@1" value={rr.margin_r1.toFixed(3)} accent />
            <Readout label="pointwise recall@1" value={rr.pointwise_r1.toFixed(3)} />
            <Readout label="margin − pointwise" value={`+${(rr.margin_r1 - rr.pointwise_r1).toFixed(3)}`} />
            <Readout label="TC reconstruction error" value={`${(tcTail * 100).toFixed(0)}%`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The teacher's score matrix carries a huge top singular value σ₁ = {SINGULAR_VALUES_T[0].toFixed(0)} — the per-query
            level — that the pointwise student best_rank_d(T) must spend a dimension reproducing. The centered teacher TC has a
            flat spectrum, all of it ranking, so the margin student best_rank_d(TC) spends every dimension on the ordering and
            its recall@1 ({rr.margin_r1.toFixed(3)}) leads the pointwise student ({rr.pointwise_r1.toFixed(3)}) at rank d = {d},
            with the biggest gap at the restricted rank {D_STAGE}. At {tcTail < 1e-9 ? 'full rank' : `d = ${d}`} the margin
            student leaves {(tcTail * 100).toFixed(0)}% of the ranking structure unreconstructed; the rank ceiling binds both
            students below the teacher ({TEACHER_R1.toFixed(1)}).
          </p>
        </div>
      )}

      {panel === 'darkknowledge' && (
        <div>
          <Slider label="log₁₀ corpus size" value={corpusExp} min={2} max={8} step={1} onChange={setCorpusExp} fmt={() => corpus.toLocaleString()} />
          <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '0.8rem', alignItems: 'start' }}>
            <SoftHardBars />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <div style={{ display: 'flex', gap: '1.4rem', flexWrap: 'wrap' }}>
                <Readout label="teacher cost / query" value={teacherCost.toLocaleString()} />
                <Readout label="student cost / query" value={studentCost.toLocaleString()} accent />
                <Readout label="speedup" value={`${speedup.toFixed(0)}×`} accent />
              </div>
              <MarginStrip />
            </div>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The teacher's margins on the mined hard-negative pairs are <em>graded</em> (mean ≈ 0.66, a real spread) where a
            one-hot label's margin is the constant 1 — that is the dark knowledge. Yet on this clean, in-sample, block-structured
            toy the binary ground-truth target compresses to low rank at least as well as the soft teacher (the soft-beats-hard
            advantage is a generalization phenomenon a closed-form in-sample fit cannot show). The real payoff is cost: at a
            corpus of {corpus.toLocaleString()} the cross-encoder teacher costs {teacherCost.toLocaleString()} per query while
            the precomputable student costs {studentCost.toLocaleString()} — a {speedup.toFixed(0)}× speedup, cross-encoder
            ranking at dual-encoder inference cost.
          </p>
        </div>
      )}
    </div>
  );
});
