import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * LLM-as-Judge Laboratory — five panels for the `llm-as-judge-ragas` topic. An LLM judge is a noisy
 * measurement instrument; every RAGAS metric is an estimator built from its verdicts, read correctly
 * only by CORRECTING for the instrument:
 *   A. The noisy instrument. Sensitivity/specificity sliders drive the observed faithfulness p_obs away
 *      from the truth pi; Rogan–Gladen inverts the error model to recover pi, with a variance inflated
 *      by 1/(se+sp−1)^2 that explodes at the Youden line. A constructed two-system flip shows the
 *      correction REVERSING a ranking the raw score got wrong.
 *   B. Agreement is not accuracy. A 2×2 agreement table with a prevalence slider: at fixed observed
 *      agreement, Cohen's kappa collapses under skewed marginals (the Feinstein–Cicchetti paradox)
 *      while Gwet's AC1 stays stable.
 *   C. Judge calibration. The reliability diagram of the judge's stated confidence (raw / Platt /
 *      isotonic), ECE recomputed in TS, with the AUC unchanged (calibration ⟂ ranking) and a paired
 *      swap test detecting position bias.
 *   D. The variance floor. SE = sqrt(var_q/Q + var_j/J + var_e/(QJ)); more queries shrink the query
 *      term but never the judge-variance floor var_j/J — only more judges do (the budget lever).
 *   E. Dawid–Skene. With no gold labels, latent-class EM recovers each judge's sensitivity/specificity
 *      from the agreement structure alone — the rates Panel A's correction needs but never observes.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every baked constant below is mirrored TO THE DECIMAL from
 * notebooks/llm-as-judge-ragas/llm_as_judge_ragas.py (viz_constants()). Matching asserts:
 * test_rogan_gladen_recovers_pi_exact / test_variance_explodes_toward_coinflip / test_kappa_paradox /
 * test_judge_overconfident_then_recalibrated / test_platt_preserves_judge_ranking /
 * test_swap_detects_injected_bias / test_judge_variance_is_a_floor / test_budget_lever_multi_judge_wins /
 * test_dawid_skene_recovers_error_rates / test_correction_flips_ranking. The lab recomputes ONLY closed
 * forms in TS (the Rogan–Gladen estimate + its variance, kappa/AC1 from the reconstructed table,
 * ECE/MCE from the baked reliability bins, the SE decomposition); all corpus-derived rates, reliability
 * bins, variance components, paradox tables, and EM recoveries are baked. Change a number here ->
 * change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_QUERIES = 40;
const N_DOCS = 120;
const K = 10;
const N_JUDGES = 5;

// Panel A — the lenient judge's measured (audited) rates on each leg, and the variance inflation. The
// lab recomputes p_obs, pi_hat, and the variance in closed form from the se/sp/n/pi sliders; this table
// is the corpus anchor showing the same judge inflates each real system.
type Leg = 'lexical' | 'dense' | 'late_interaction';
const LEG_RG: Record<Leg, { pObs: number; sens: number; spec: number; piCorr: number; piOracle: number; varInfl: number }> = {
  lexical: { pObs: 0.6075, sens: 0.9458, spec: 0.7411, piCorr: 0.5075, piOracle: 0.5075, varInfl: 2.119 },
  dense: { pObs: 0.7025, sens: 0.9228, spec: 0.7021, piCorr: 0.6475, piOracle: 0.6475, varInfl: 2.561 },
  late_interaction: { pObs: 0.7875, sens: 0.9167, spec: 0.6705, piCorr: 0.78, piOracle: 0.78, varInfl: 2.901 },
};
// the constructed two-system ranking flip (test_correction_flips_ranking).
const RANK_FLIP = {
  A: { pObs: 0.8, sens: 0.95, spec: 0.55, piOracle: 0.5 },
  B: { pObs: 0.74, sens: 0.9, spec: 0.85, piOracle: 0.66 },
};

// Panel B — the kappa paradox: two tables with identical observed agreement, divergent kappa.
const KAPPA_PARADOX = {
  balanced: { po: 0.85, kappa: 0.7, ac1: 0.7007 },
  skewed: { po: 0.85, kappa: 0.3182, ac1: 0.808 },
};

// Panel C — judge-confidence calibration on the dense leg, three judge profiles × three conditions.
type Judge = 'lenient' | 'balanced' | 'strict';
const JUDGE_LABEL: Record<Judge, string> = { lenient: 'lenient', balanced: 'balanced', strict: 'strict' };
type Cond = 'raw' | 'platt' | 'isotonic';
const RELIABILITY: Record<Judge, Record<Cond, number[][]>> = {
  lenient: {
    raw: [[0.1563, 0.0, 39], [0.2904, 0.0, 41], [0.489, 0.0, 40], [0.7271, 0.5, 40], [0.862, 0.975, 40], [0.9055, 1.0, 40], [0.9323, 1.0, 40], [0.9513, 1.0, 40], [0.9689, 1.0, 40], [0.9849, 1.0, 40]],
    platt: [[0.0, 0.0, 39], [0.0, 0.0, 41], [0.0013, 0.0, 40], [0.4887, 0.5, 40], [0.9888, 0.975, 40], [0.9977, 1.0, 40], [0.9992, 1.0, 40], [0.9996, 1.0, 40], [0.9998, 1.0, 40], [0.9999, 1.0, 40]],
    isotonic: [[0.0278, 0.0278, 144], [0.9961, 0.9961, 256]],
  },
  balanced: {
    raw: [[0.1076, 0.0, 39], [0.1474, 0.0, 41], [0.2026, 0.0, 40], [0.4919, 0.475, 40], [0.7837, 1.0, 40], [0.8146, 1.0, 40], [0.8383, 1.0, 40], [0.8582, 1.0, 40], [0.882, 1.0, 40], [0.9129, 1.0, 40]],
    platt: [[0.0, 0.0, 39], [0.0, 0.0, 41], [0.0, 0.0, 40], [0.475, 0.475, 40], [1.0, 1.0, 40], [1.0, 1.0, 40], [1.0, 1.0, 40], [1.0, 1.0, 40], [1.0, 1.0, 40], [1.0, 1.0, 40]],
    isotonic: [[0.6475, 0.6475, 400]],
  },
  strict: {
    raw: [[0.0597, 0.0, 40], [0.0649, 0.0, 39], [0.071, 0.0, 41], [0.388, 0.4615, 39], [0.7608, 1.0, 41], [0.7699, 1.0, 40], [0.7776, 1.0, 40], [0.7843, 1.0, 40], [0.7931, 1.0, 40], [0.8071, 1.0, 40]],
    platt: [[0.0, 0.0, 40], [0.0, 0.0, 39], [0.0, 0.0, 41], [0.4615, 0.4615, 39], [1.0, 1.0, 41], [1.0, 1.0, 40], [1.0, 1.0, 40], [1.0, 1.0, 40], [1.0, 1.0, 40], [1.0, 1.0, 40]],
    isotonic: [[0.6475, 0.6475, 400]],
  },
};
const JUDGE_META: Record<Judge, { eceRaw: number; aucRaw: number; aucPlatt: number; plattA: number; plattB: number; swapBias: number; swapP: number }> = {
  lenient: { eceRaw: 0.1536, aucRaw: 0.998494, aucPlatt: 0.998494, plattA: 35.9814, plattB: -26.3816, swapBias: 0.0702, swapP: 2.302e-105 },
  balanced: { eceRaw: 0.1386, aucRaw: 1.0, aucPlatt: 1.0, plattA: 41.8602, plattB: -21.5512, swapBias: 0.0413, swapP: 1.78e-246 },
  strict: { eceRaw: 0.1581, aucRaw: 1.0, aucPlatt: 1.0, plattA: 33.5152, plattB: -14.2128, swapBias: 0.0134, swapP: 1.217e-176 },
};
const SWAP_CONTROL = { bias: 0.0, p: 1.0 };
const N_PAIRS = 400;
const BASE_RATE = 0.6475;

// Panel D — the variance decomposition of the per-query faithfulness across the 5-judge panel.
const VAR = { q: 0.01021, j: 0.00484, e: 0.01341, icc21: 0.3587 };
const BUDGET = 200;

// Panel E — Dawid–Skene recovery of the planted per-judge error rates (no gold labels).
const DS = {
  plantedSens: [0.92, 0.8, 0.7, 0.85, 0.78], emSens: [0.927, 0.823, 0.703, 0.845, 0.841],
  plantedSpec: [0.88, 0.9, 0.75, 0.82, 0.8], emSpec: [0.908, 0.94, 0.751, 0.787, 0.775],
  emAcc: 0.975, majAcc: 0.9725, nIter: 16,
};

const ACCENT = 'var(--color-accent)';
const MUTED = '#9aa3ad';
const POS_COLOR = '#6fa389';
const NEG_COLOR = '#c0726a';
const TRUTH_COLOR = '#6a8caf';
const fmt = (x: number, n = 3) => x.toFixed(n);
const fmtP = (p: number) => (p <= 0 ? '<0.001' : p < 1e-3 ? p.toExponential(1) : p.toFixed(3));

// --- closed-form TS recomputation -----------------------------------------------------
// Rogan–Gladen: pi_hat = (p_obs + sp − 1)/(se + sp − 1), defined only above the Youden line J > 0.
const youden = (se: number, sp: number) => se + sp - 1;
const pObsOf = (pi: number, se: number, sp: number) => pi * se + (1 - pi) * (1 - sp);
const roganGladen = (pObs: number, se: number, sp: number) => {
  const J = youden(se, sp);
  return J <= 1e-9 ? NaN : Math.min(1, Math.max(0, (pObs + sp - 1) / J));
};
const rgVariance = (pObs: number, se: number, sp: number, n: number) => {
  const J = youden(se, sp);
  return J <= 1e-9 || n <= 0 ? NaN : (pObs * (1 - pObs)) / (n * J * J);
};
// kappa and AC1 from a symmetric 2×2 reconstructed from (observed agreement p_o, "+" prevalence q).
const kappaOf = (po: number, q: number) => {
  const pe = q * q + (1 - q) * (1 - q);
  return 1 - pe > 1e-9 ? (po - pe) / (1 - pe) : 0;
};
const ac1Of = (po: number, q: number) => {
  const pe = 2 * q * (1 - q);
  return 1 - pe > 1e-9 ? (po - pe) / (1 - pe) : 0;
};
// ECE = Σ_b (n_b/N)|acc_b − conf_b|; MCE = max_b|acc_b − conf_b| — from the baked [conf, acc, n] bins.
const eceOf = (bins: number[][]) => {
  const N = bins.reduce((s, b) => s + b[2], 0);
  return N ? bins.reduce((s, b) => s + (b[2] / N) * Math.abs(b[1] - b[0]), 0) : 0;
};
const mceOf = (bins: number[][]) => bins.reduce((m, b) => Math.max(m, Math.abs(b[1] - b[0])), 0);
// the two-way standard error of the mean faithfulness over Q queries and J judges.
const seOf = (Q: number, J: number) => Math.sqrt(VAR.q / Q + VAR.j / J + VAR.e / (Q * J));

// --- shared UI atoms ------------------------------------------------------------------
function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}
function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display: string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '12rem' }}>{label} = <strong>{display}</strong></span>
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

// ===== Panel A — the noisy instrument (Rogan–Gladen) ==============================================
const PI_PRESETS = [0.5, 0.65, 0.9];
function InstrumentPanel({ se, sp, n, pi, setSe, setSp, setN, setPi }: {
  se: number; sp: number; n: number; pi: number;
  setSe: (v: number) => void; setSp: (v: number) => void; setN: (v: number) => void; setPi: (v: number) => void;
}) {
  const J = youden(se, sp);
  const pObs = pObsOf(pi, se, sp);
  const piHat = roganGladen(pObs, se, sp);
  const variance = rgVariance(pObs, se, sp, n);
  const ciHalf = Number.isNaN(variance) ? NaN : 1.96 * Math.sqrt(variance);
  const infl = J > 1e-9 ? 1 / (J * J) : NaN;
  // thermometer [0,1]
  const W = 460, H = 96, padL = 14, padR = 14, axisY = 54;
  const tx = (v: number) => padL + (W - padL - padR) * v;
  const tick = (v: number, color: string, label: string, dy: number) => (
    <g>
      <line x1={tx(v)} y1={axisY - 9} x2={tx(v)} y2={axisY + 9} stroke={color} strokeWidth={2.4} />
      <text x={tx(v)} y={axisY + dy} textAnchor="middle" fontSize={8.5} fill={color} fontFamily="var(--font-sans)">{label} {fmt(v, 3)}</text>
    </g>
  );
  // variance-explosion curve 1/J^2 vs J in (0,1]
  const VW = 300, VH = 130, vPadL = 30, vPadR = 8, vPadT = 10, vPadB = 22;
  const vx = (j: number) => vPadL + (VW - vPadL - vPadR) * j;
  const vyMax = 12;
  const vy = (val: number) => VH - vPadB - (VH - vPadT - vPadB) * (Math.min(val, vyMax) / vyMax);
  const curve = Array.from({ length: 60 }, (_v, i) => {
    const j = 0.05 + (0.95 * (i + 1)) / 60;
    return `${i ? 'L' : 'M'} ${fmt(vx(j), 1)} ${fmt(vy(1 / (j * j)), 1)}`;
  }).join(' ');
  // ranking flip recomputation
  const fA = roganGladen(RANK_FLIP.A.pObs, RANK_FLIP.A.sens, RANK_FLIP.A.spec);
  const fB = roganGladen(RANK_FLIP.B.pObs, RANK_FLIP.B.sens, RANK_FLIP.B.spec);
  const FW = 300, FH = 120, fb = 26, fgap = 64;
  const fy = (v: number) => FH - 20 - (FH - 34) * v;
  const bars = (x0: number, vA: number, vB: number, label: string) => (
    <g>
      <rect x={x0} y={fy(vA)} width={18} height={FH - 20 - fy(vA)} fill={NEG_COLOR} fillOpacity={0.85} />
      <rect x={x0 + 22} y={fy(vB)} width={18} height={FH - 20 - fy(vB)} fill={POS_COLOR} fillOpacity={0.85} />
      <text x={x0 + 20} y={FH - 8} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{label}</text>
      <text x={x0 + 9} y={fy(vA) - 2} textAnchor="middle" fontSize={7.5} fill={NEG_COLOR} fontFamily="var(--font-sans)">{fmt(vA, 2)}</text>
      <text x={x0 + 31} y={fy(vB) - 2} textAnchor="middle" fontSize={7.5} fill={POS_COLOR} fontFamily="var(--font-sans)">{fmt(vB, 2)}</text>
    </g>
  );
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-sans)' }}>latent faithfulness π</span>
        {PI_PRESETS.map((p) => (
          <button key={p} type="button" style={pill(Math.abs(pi - p) < 1e-9)} onClick={() => setPi(p)}>{fmt(p, 2)}</button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="observed versus corrected faithfulness on the unit interval" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={tx(0)} y1={axisY} x2={tx(1)} y2={axisY} stroke="var(--color-border)" strokeWidth={1.4} />
        {tick(pi, TRUTH_COLOR, 'π truth', -14)}
        {tick(pObs, NEG_COLOR, 'p_obs', 24)}
        {!Number.isNaN(piHat) && tick(piHat, POS_COLOR, 'π̂ RG', 38)}
        {!Number.isNaN(ciHalf) && (
          <line x1={tx(Math.max(0, piHat - ciHalf))} y1={axisY + 14} x2={tx(Math.min(1, piHat + ciHalf))} y2={axisY + 14} stroke={POS_COLOR} strokeWidth={2} />
        )}
        <text x={tx(0)} y={axisY - 16} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">0</text>
        <text x={tx(1)} y={axisY - 16} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">1</text>
      </svg>
      <Slider label="sensitivity se" value={se} min={0.5} max={1} step={0.01} onChange={setSe} display={fmt(se, 2)} />
      <Slider label="specificity sp" value={sp} min={0.5} max={1} step={0.01} onChange={setSp} display={fmt(sp, 2)} />
      <Slider label="claims n" value={n} min={5} max={200} step={1} onChange={(v) => setN(Math.round(v))} display={`${n}`} />
      <div style={{ display: 'flex', gap: '1.2rem', flexWrap: 'wrap', alignItems: 'flex-start', marginTop: '0.3rem' }}>
        <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label="variance inflation one over Youden squared" style={{ width: '100%', maxWidth: VW, height: 'auto', flex: '1 1 260px' }}>
          <line x1={vPadL} y1={VH - vPadB} x2={VW - vPadR} y2={VH - vPadB} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={vPadL} y1={vPadT} x2={vPadL} y2={VH - vPadB} stroke="var(--color-border)" strokeWidth={1} />
          <path d={curve} fill="none" stroke={MUTED} strokeWidth={1.8} />
          {J > 1e-9 && <circle cx={vx(J)} cy={vy(1 / (J * J))} r={4} fill={ACCENT} />}
          <text x={(vPadL + VW - vPadR) / 2} y={vPadT + 2} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">variance inflation 1/J²</text>
          <text x={(vPadL + VW - vPadR) / 2} y={VH - 4} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">Youden index J = se + sp − 1</text>
        </svg>
        <div style={{ flex: '1 1 150px', display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
          <Readout label="observed p_obs" value={fmt(pObs, 3)} />
          <Readout label="corrected π̂ (Rogan–Gladen)" value={Number.isNaN(piHat) ? 'undefined (J ≤ 0)' : fmt(piHat, 3)} accent />
          <Readout label="bias p_obs − π" value={fmt(pObs - pi, 3)} />
          <Readout label="Youden J · inflation 1/J²" value={J <= 0 ? `${fmt(J, 2)} · ∞` : `${fmt(J, 2)} · ${fmt(infl, 2)}×`} accent={J <= 0} />
        </div>
      </div>
      <div style={{ marginTop: '0.7rem', borderTop: '1px solid var(--color-border)', paddingTop: '0.6rem' }}>
        <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-sans)', marginBottom: '0.3rem' }}>
          Correcting for the instrument can <strong>reverse</strong> a ranking (same lenient judge, two systems):
        </div>
        <svg viewBox={`0 0 ${FW} ${FH}`} role="img" aria-label="raw versus corrected versus oracle faithfulness for two systems" style={{ width: '100%', maxWidth: FW, height: 'auto' }}>
          {bars(fb, RANK_FLIP.A.pObs, RANK_FLIP.B.pObs, 'raw p_obs')}
          {bars(fb + fgap + 30, fA, fB, 'corrected π̂')}
          {bars(fb + 2 * (fgap + 30), RANK_FLIP.A.piOracle, RANK_FLIP.B.piOracle, 'oracle π')}
          <text x={FW - 4} y={12} textAnchor="end" fontSize={8} fill={NEG_COLOR} fontFamily="var(--font-sans)">■ system A</text>
          <text x={FW - 4} y={23} textAnchor="end" fontSize={8} fill={POS_COLOR} fontFamily="var(--font-sans)">■ system B</text>
        </svg>
        <p style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', marginTop: '0.3rem', lineHeight: 1.45 }}>
          Raw faithfulness ranks <strong>A &gt; B</strong> ({fmt(RANK_FLIP.A.pObs, 2)} vs {fmt(RANK_FLIP.B.pObs, 2)}); the
          judge's low specificity on A's documents inflates it. Rogan–Gladen restores the truth:{' '}
          <strong>B &gt; A</strong> ({fmt(fB, 2)} vs {fmt(fA, 2)}), matching the oracle.
        </p>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Faithfulness is the mean of the judge's per-claim verdicts, so an imperfect judge makes it a{' '}
        <strong>biased</strong> estimate of the true grounded fraction π. Rogan–Gladen inverts the error model to
        recover π, but its variance is inflated by <strong>1/J²</strong> — as the judge approaches a coin flip
        (J → 0) the correction is amplified to uselessness, and below the Youden line it is undefined. On the corpus the
        same lenient judge inflates dense faithfulness to <strong>{fmt(LEG_RG.dense.pObs, 3)}</strong> over a true{' '}
        <strong>{fmt(LEG_RG.dense.piOracle, 3)}</strong>; the audited correction recovers it.
      </p>
    </div>
  );
}

// ===== Panel B — agreement is not accuracy ========================================================
function AgreementPanel({ po, q, setPo, setQ }: {
  po: number; q: number; setPo: (v: number) => void; setQ: (v: number) => void;
}) {
  const pe = q * q + (1 - q) * (1 - q);
  const kappa = kappaOf(po, q);
  const ac1 = ac1Of(po, q);
  const paradox = po >= 0.85 && kappa <= 0.4;
  // reconstructed symmetric 2×2 cell proportions
  const dis = (1 - po) / 2;
  const a = Math.max(0, Math.min(1, q - dis));
  const d = Math.max(0, po - a);
  const cells = [[a, dis], [dis, d]];
  const GW = 150, GH = 150, gp = 22;
  const cs = (GW - 2 * gp) / 2;
  // kappa-vs-q and AC1-vs-q sweeps at fixed p_o
  const SW = 300, SH = 150, sPadL = 28, sPadR = 8, sPadT = 12, sPadB = 24;
  const sx = (qq: number) => sPadL + (SW - sPadL - sPadR) * ((qq - 0.05) / 0.9);
  const sy = (v: number) => SH - sPadB - (SH - sPadT - sPadB) * Math.max(0, Math.min(1, v));
  const sweep = (fn: (po: number, q: number) => number) => Array.from({ length: 61 }, (_v, i) => {
    const qq = 0.05 + (0.9 * i) / 60;
    return `${i ? 'L' : 'M'} ${fmt(sx(qq), 1)} ${fmt(sy(fn(po, qq)), 1)}`;
  }).join(' ');
  return (
    <div>
      <Slider label="observed agreement p_o" value={po} min={0.5} max={1} step={0.01} onChange={setPo} display={fmt(po, 2)} />
      <Slider label="prevalence of “supported” q" value={q} min={0.05} max={0.95} step={0.01} onChange={setQ} display={fmt(q, 2)} />
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${GW} ${GH}`} role="img" aria-label="two by two agreement table" style={{ width: '100%', maxWidth: GW, height: 'auto', flex: '0 0 150px' }}>
          {cells.map((row, i) => row.map((v, j) => (
            <g key={`${i}-${j}`}>
              <rect x={gp + j * cs} y={gp + i * cs} width={cs - 2} height={cs - 2}
                fill={i === j ? POS_COLOR : NEG_COLOR} fillOpacity={0.15 + 0.8 * v} />
              <text x={gp + j * cs + cs / 2} y={gp + i * cs + cs / 2 + 3} textAnchor="middle" fontSize={9} fill="var(--color-text)" fontFamily="var(--font-sans)">{fmt(v, 2)}</text>
            </g>
          )))}
          <text x={gp + cs} y={14} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rater B</text>
          <text x={10} y={gp + cs} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 10 ${gp + cs})`}>rater A</text>
        </svg>
        <svg viewBox={`0 0 ${SW} ${SH}`} role="img" aria-label="kappa and AC1 versus prevalence at fixed agreement" style={{ width: '100%', maxWidth: SW, height: 'auto', flex: '1 1 260px' }}>
          <line x1={sPadL} y1={SH - sPadB} x2={SW - sPadR} y2={SH - sPadB} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={sPadL} y1={sPadT} x2={sPadL} y2={SH - sPadB} stroke="var(--color-border)" strokeWidth={1} />
          <path d={sweep(kappaOf)} fill="none" stroke={ACCENT} strokeWidth={1.8} />
          <path d={sweep(ac1Of)} fill="none" stroke={MUTED} strokeWidth={1.8} strokeDasharray="4 3" />
          <line x1={sx(q)} y1={sPadT} x2={sx(q)} y2={SH - sPadB} stroke="var(--color-border)" strokeWidth={1} strokeDasharray="2 2" />
          <circle cx={sx(q)} cy={sy(kappa)} r={3.5} fill={ACCENT} />
          <circle cx={sx(q)} cy={sy(ac1)} r={3.5} fill={MUTED} />
          <text x={SW - sPadR} y={sy(0.93)} textAnchor="end" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">κ (Cohen)</text>
          <text x={SW - sPadR} y={sy(0.06)} textAnchor="end" fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">AC1 (Gwet)</text>
          <text x={(sPadL + SW - sPadR) / 2} y={SH - 6} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">prevalence q at fixed p_o = {fmt(po, 2)}</text>
        </svg>
      </div>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.4rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <Readout label="p_o (observed)" value={fmt(po, 3)} />
        <Readout label="p_e (chance)" value={fmt(pe, 3)} />
        <Readout label="Cohen's κ" value={fmt(kappa, 3)} accent={paradox} />
        <Readout label="Gwet's AC1" value={fmt(ac1, 3)} />
        {paradox && <span style={{ ...pill(true), background: NEG_COLOR, border: `1px solid ${NEG_COLOR}` }}>paradox: high agreement, low κ</span>}
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Cohen's κ rescales observed agreement by the chance agreement p_e, which swells as the marginals skew
        (q → 0 or 1), so κ collapses even at high p_o — the Feinstein–Cicchetti paradox. The baked worked pair has{' '}
        <strong>identical</strong> agreement p_o = {fmt(KAPPA_PARADOX.balanced.po, 2)} yet κ ={' '}
        <strong>{fmt(KAPPA_PARADOX.balanced.kappa, 2)}</strong> (balanced) vs{' '}
        <strong>{fmt(KAPPA_PARADOX.skewed.kappa, 2)}</strong> (skewed), while Gwet's AC1 stays{' '}
        {fmt(KAPPA_PARADOX.balanced.ac1, 2)} vs {fmt(KAPPA_PARADOX.skewed.ac1, 2)}. Read κ alongside the marginals and a
        prevalence-robust coefficient — never alone.
      </p>
    </div>
  );
}

// ===== Panel C — judge-confidence calibration =====================================================
const LEG_COLOR = '#6a8caf';
function CalibrationPanel({ judge, cond, setJudge, setCond }: {
  judge: Judge; cond: Cond; setJudge: (j: Judge) => void; setCond: (c: Cond) => void;
}) {
  const bins = RELIABILITY[judge][cond];
  const meta = JUDGE_META[judge];
  const ece = eceOf(bins);
  const mce = mceOf(bins);
  const W = 300, H = 300, pad = 36;
  const ax = (v: number) => pad + (W - 2 * pad) * v;
  const ay = (v: number) => H - pad - (H - 2 * pad) * v;
  const maxN = Math.max(...bins.map((b) => b[2]));
  const r = (cnt: number) => 3 + 7 * Math.sqrt(cnt / maxN);
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
        {(['lenient', 'balanced', 'strict'] as Judge[]).map((jj) => (
          <button key={jj} type="button" style={pill(judge === jj)} onClick={() => setJudge(jj)}>{JUDGE_LABEL[jj]} judge</button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
        {(['raw', 'platt', 'isotonic'] as Cond[]).map((c) => (
          <button key={c} type="button" style={pill(cond === c)} onClick={() => setCond(c)}>{c === 'raw' ? 'raw confidence' : c === 'platt' ? 'Platt' : 'isotonic'}</button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="reliability diagram, confidence versus empirical relevance" style={{ width: '100%', maxWidth: W, height: 'auto', flex: '1 1 260px' }}>
          <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={ax(0)} y1={ay(0)} x2={ax(1)} y2={ay(1)} stroke={MUTED} strokeWidth={1.2} strokeDasharray="4 3" />
          <text x={ax(0.62)} y={ay(0.74)} fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-45 ${ax(0.62)} ${ay(0.74)})`}>perfect calibration</text>
          {bins.map((b, i) => (
            <g key={i}>
              <line x1={ax(b[0])} y1={ay(b[0])} x2={ax(b[0])} y2={ay(b[1])} stroke={LEG_COLOR} strokeWidth={1} strokeOpacity={0.5} />
              <circle cx={ax(b[0])} cy={ay(b[1])} r={r(b[2])} fill={LEG_COLOR} fillOpacity={0.8} />
            </g>
          ))}
          <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">judge confidence</text>
          <text x={12} y={H / 2} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${H / 2})`}>empirical relevance</text>
        </svg>
        <div style={{ flex: '1 1 180px' }}>
          <Readout label={`ECE (${cond})`} value={fmt(ece, 4)} accent={cond !== 'raw'} />
          <div style={{ height: '0.5rem' }} />
          <Readout label="MCE (worst bin)" value={fmt(mce, 4)} />
          <div style={{ height: '0.5rem' }} />
          <Readout label="AUC raw → recalibrated" value={`${fmt(meta.aucRaw, 4)} = ${fmt(meta.aucPlatt, 4)}`} />
          <div style={{ height: '0.5rem' }} />
          <Readout label="position-bias swap test" value={`bias ${fmt(meta.swapBias, 3)}, p ${fmtP(meta.swapP)}`} accent={meta.swapP < 0.05} />
          <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem', lineHeight: 1.4 }}>
            Platt is σ({fmt(meta.plattA, 1)}·s {meta.plattB < 0 ? '−' : '+'} {fmt(Math.abs(meta.plattB), 1)}), strictly
            monotone, so the AUC is <strong>identical</strong> before and after. A bias-free judge's swap test does not
            reject (bias {fmt(SWAP_CONTROL.bias, 2)}, p {fmt(SWAP_CONTROL.p, 2)}).
          </p>
        </div>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The judge's stated confidence ranks claims almost perfectly (AUC ≈ {fmt(meta.aucRaw, 2)}) yet is badly{' '}
        <strong>over-confident</strong> raw (ECE {fmt(meta.eceRaw, 2)}): high-confidence verdicts are right more often
        than the confidence claims. <strong>Platt</strong> and <strong>isotonic</strong> pull the dots onto the diagonal,
        lowering ECE without moving a single ranking ({N_PAIRS} verdicts, base rate {fmt(BASE_RATE, 2)}) — calibration is
        orthogonal to ranking. The paired swap test separately detects the judge's position bias.
      </p>
    </div>
  );
}

// ===== Panel D — the variance floor ================================================================
function VariancePanel({ nq, nj, setNq, setNj }: {
  nq: number; nj: number; setNq: (v: number) => void; setNj: (v: number) => void;
}) {
  const seTotal = seOf(nq, nj);
  const seFloor = Math.sqrt(VAR.j / nj);
  const seQuery = Math.sqrt(VAR.q / nq + VAR.e / (nq * nj));
  const seMulti = seOf(40, 5);
  const seSingle = seOf(BUDGET, 1);
  // SE vs Q curve at fixed J
  const W = 320, H = 170, padL = 38, padR = 10, padT = 12, padB = 24;
  const qMax = 640;
  const cx = (Q: number) => padL + (W - padL - padR) * (Math.log(Q) - Math.log(10)) / (Math.log(qMax) - Math.log(10));
  const yMax = 0.1;
  const cy = (v: number) => H - padB - (H - padT - padB) * Math.min(v, yMax) / yMax;
  const curve = Array.from({ length: 64 }, (_v, i) => {
    const Q = 10 * Math.pow(qMax / 10, i / 63);
    return `${i ? 'L' : 'M'} ${fmt(cx(Q), 1)} ${fmt(cy(seOf(Q, nj)), 1)}`;
  }).join(' ');
  return (
    <div>
      <Slider label="queries Q" value={nq} min={10} max={640} step={10} onChange={(v) => setNq(Math.round(v))} display={`${nq}`} />
      <Slider label="judges per query J" value={nj} min={1} max={9} step={1} onChange={(v) => setNj(Math.round(v))} display={`${nj}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="standard error versus query count with the judge-variance floor" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={cy(seFloor)} x2={W - padR} y2={cy(seFloor)} stroke={NEG_COLOR} strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={W - padR} y={cy(seFloor) - 3} textAnchor="end" fontSize={8} fill={NEG_COLOR} fontFamily="var(--font-sans)">judge floor √(σ²_j/J) = {fmt(seFloor, 3)}</text>
        <path d={curve} fill="none" stroke={ACCENT} strokeWidth={1.8} />
        <circle cx={cx(nq)} cy={cy(seTotal)} r={4} fill={ACCENT} />
        <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">queries Q (log scale) · SE of mean faithfulness</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="total SE" value={fmt(seTotal, 4)} accent />
        <Readout label="query component" value={fmt(seQuery, 4)} />
        <Readout label="judge floor" value={fmt(seFloor, 4)} />
        <Readout label="ICC(2,1)" value={fmt(VAR.icc21, 3)} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The faithfulness standard error decomposes as SE² = σ²_q/Q + σ²_j/J + σ²_e/(QJ). Adding queries drives the
        query term to zero, but the total SE <strong>plateaus</strong> at the judge floor √(σ²_j/J) — only more judges
        lower it. At a fixed budget of {BUDGET} judge-calls, five judges over forty queries beat one judge over{' '}
        {BUDGET} queries: SE <strong>{fmt(seMulti, 3)}</strong> vs <strong>{fmt(seSingle, 3)}</strong>. The judge is an
        irreducible source of noise; you cannot buy confidence in a faithfulness score with queries alone.
      </p>
    </div>
  );
}

// ===== Panel E — Dawid–Skene recovery =============================================================
function DawidSkenePanel() {
  const W = 460, H = 200, padL = 36, padR = 10, padT = 18, padB = 40;
  const J = DS.plantedSens.length;
  const groupW = (W - padL - padR) / J;
  const by = (v: number) => H - padB - (H - padT - padB) * v;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="planted versus recovered sensitivity and specificity per judge" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((g) => (
          <g key={g}>
            <line x1={padL} y1={by(g)} x2={W - padR} y2={by(g)} stroke="var(--color-border)" strokeWidth={0.5} strokeOpacity={0.5} />
            <text x={padL - 4} y={by(g) + 3} textAnchor="end" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{fmt(g, 1)}</text>
          </g>
        ))}
        {DS.plantedSens.map((ps, j) => {
          const x0 = padL + j * groupW + 6;
          const bw = (groupW - 12) / 4;
          const vals: [number, string, number][] = [
            [ps, MUTED, 0], [DS.emSens[j], ACCENT, 1], [DS.plantedSpec[j], MUTED, 2], [DS.emSpec[j], POS_COLOR, 3],
          ];
          return (
            <g key={j}>
              {vals.map(([v, color, k]) => (
                <rect key={k} x={x0 + k * bw} y={by(v)} width={bw - 1.5} height={(H - padB) - by(v)} fill={color} fillOpacity={0.85} />
              ))}
              <text x={x0 + 2 * bw} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">judge {j + 1}</text>
            </g>
          );
        })}
        <text x={padL} y={12} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          ▮ planted   ▮ EM sens   ▮ EM spec   (each judge: planted-se, EM-se, planted-sp, EM-sp)
        </text>
        <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">sensitivity & specificity per judge</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="EM label accuracy" value={fmt(DS.emAcc, 3)} accent />
        <Readout label="majority-vote accuracy" value={fmt(DS.majAcc, 3)} />
        <Readout label="EM iterations" value={`${DS.nIter}`} />
        <Readout label="gold labels used" value="none" accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Rogan–Gladen needed the judge's sensitivity and specificity. When there is no gold label, the Dawid–Skene
        latent-class EM recovers each judge's confusion matrix and the hidden true labels from the{' '}
        <strong>agreement structure alone</strong> — here matching every planted rate within ≈0.06 and beating
        majority vote on label accuracy ({fmt(DS.emAcc, 3)} vs {fmt(DS.majAcc, 3)}). It is identifiable only up to a
        label permutation and its likelihood is non-convex, so it is initialized at the majority vote. This closes the
        loop: EM supplies the error rates the instrument correction in Panel A assumes.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'instrument' | 'agreement' | 'calibration' | 'variance' | 'dawidskene';
const TEX: Record<Panel, string> = {
  instrument: '\\hat\\pi = \\dfrac{p_{\\mathrm{obs}} + \\mathrm{sp} - 1}{\\mathrm{se} + \\mathrm{sp} - 1}, \\quad \\operatorname{Var}(\\hat\\pi) = \\dfrac{p_{\\mathrm{obs}}(1-p_{\\mathrm{obs}})}{n\\,(\\mathrm{se}+\\mathrm{sp}-1)^2}',
  agreement: '\\kappa = \\dfrac{p_o - p_e}{1 - p_e}, \\quad \\mathrm{AC1} = \\dfrac{p_o - 2q(1-q)}{1 - 2q(1-q)}',
  calibration: '\\mathrm{ECE} = \\sum_b \\dfrac{n_b}{N}\\,\\big|\\mathrm{acc}_b - \\mathrm{conf}_b\\big|, \\quad \\mathrm{AUC}(\\sigma(as+b)) = \\mathrm{AUC}(s),\\ a>0',
  variance: '\\operatorname{SE}^2 = \\dfrac{\\sigma_q^2}{Q} + \\dfrac{\\sigma_j^2}{J} + \\dfrac{\\sigma_e^2}{Q\\,J}, \\quad \\mathrm{ICC} = \\dfrac{\\sigma_q^2}{\\sigma_q^2+\\sigma_j^2+\\sigma_e^2}',
  dawidskene: '\\hat z_i = \\arg\\max_k\\ \\pi_k \\prod_j \\theta^{(j)}_{k,\\,r_{ij}}, \\quad \\theta^{(j)}_{kl} = \\Pr(r_{ij}=l \\mid z_i=k)',
};

export default memo(function LLMJudgeLaboratory() {
  const [panel, setPanel] = useState<Panel>('instrument');
  // Panel A
  const [se, setSe] = useState(0.92);
  const [sp, setSp] = useState(0.7);
  const [n, setN] = useState(40);
  const [pi, setPi] = useState(0.65);
  // Panel B
  const [po, setPo] = useState(0.85);
  const [q, setQ] = useState(0.85);
  // Panel C
  const [judge, setJudge] = useState<Judge>('lenient');
  const [cond, setCond] = useState<Cond>('raw');
  // Panel D
  const [nq, setNq] = useState(40);
  const [nj, setNj] = useState(1);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="llm-judge" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'instrument')} onClick={() => setPanel('instrument')}>A · noisy instrument</button>
        <button type="button" style={pill(panel === 'agreement')} onClick={() => setPanel('agreement')}>B · agreement</button>
        <button type="button" style={pill(panel === 'calibration')} onClick={() => setPanel('calibration')}>C · calibration</button>
        <button type="button" style={pill(panel === 'variance')} onClick={() => setPanel('variance')}>D · variance floor</button>
        <button type="button" style={pill(panel === 'dawidskene')} onClick={() => setPanel('dawidskene')}>E · Dawid–Skene</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'instrument' && <InstrumentPanel se={se} sp={sp} n={n} pi={pi} setSe={setSe} setSp={setSp} setN={setN} setPi={setPi} />}
      {panel === 'agreement' && <AgreementPanel po={po} q={q} setPo={setPo} setQ={setQ} />}
      {panel === 'calibration' && <CalibrationPanel judge={judge} cond={cond} setJudge={setJudge} setCond={setCond} />}
      {panel === 'variance' && <VariancePanel nq={nq} nj={nj} setNq={setNq} setNj={setNj} />}
      {panel === 'dawidskene' && <DawidSkenePanel />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_DOCS} synthetic finance documents, {N_QUERIES} queries, top-{K} retrieved claims; the judge is a synthetic
        Bernoulli rater of the MaxSim oracle ({N_JUDGES}-judge panel in Panels D–E). Numbers mirror{' '}
        <code>llm_as_judge_ragas.py</code>; the lab recomputes the Rogan–Gladen estimate and its variance, κ/AC1 from
        the reconstructed table, ECE/MCE from the baked reliability bins, and the SE decomposition in closed form, and
        bakes the corpus-derived rates, reliability bins, variance components, paradox tables, and EM recoveries.
      </p>
    </div>
  );
});
