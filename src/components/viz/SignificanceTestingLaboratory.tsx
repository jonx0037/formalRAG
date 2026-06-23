import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Significance / Calibration / Drift Laboratory — four panels for the
 * `significance-testing-calibration` topic. Each panel compares TWO distributions of per-query
 * quantities (the unifying thesis):
 *   A. The paired test. The worked pair's per-query difference strip d_q = metric_A − metric_B, and the
 *      confidence interval on the mean difference under a query-count slider — paired (narrow, excludes 0
 *      sooner) vs unpaired (wide). Resolves the prereqs' "how many queries?" cliffhanger: NDCG
 *      lexical/dense paired 57 vs unpaired 185 vs 80%-power 116.
 *   B. Distribution-free tests & corrections. The sign-flip permutation null with the observed statistic
 *      marked (tail = p, agreeing with the t and bootstrap p), and the 3-pair correction grid
 *      (raw / Bonferroni / Holm / BH).
 *   C. Score calibration. The reliability diagram (confidence vs empirical relevance per bin, diagonal =
 *      perfect), ECE recomputed in TS, toggling raw / Platt / isotonic — and the AUC badge showing the
 *      ranking is UNCHANGED (calibration ⟂ ranking).
 *   D. Drift detection. Overlaid reference/current per-query NDCG distributions under a degradation knob,
 *      the KS staircase with its sup-gap, and the PSI traffic light — plus the silent-decay and
 *      input-vs-outcome contrasts.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every baked constant below is mirrored TO THE DECIMAL from
 * notebooks/significance-testing-calibration/significance_testing_calibration.py (viz_constants()).
 * Matching asserts: test_paired_separates_sooner_than_overlap / test_permutation_approximates_t /
 * test_multiple_comparison_changes_verdict / test_recalibration_lowers_ece / test_platt_preserves_ranking_exactly /
 * test_psi_ks_monotone_in_knob / test_silent_decay_paired_beats_unpaired / test_input_vs_outcome_drift.
 * The lab recomputes ONLY closed forms in TS (the CI half-widths 1.96·σ/√n, the ECE/MCE from the baked
 * reliability bins, the histograms & empirical CDFs & KS sup-gap from the baked per-query arrays); all
 * corpus-derived means, per-query arrays, reliability bins, p-values, and PSI/KS values are baked.
 * Change a number here -> change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_QUERIES = 40;
const N_DOCS = 120;
const R_SIZE = 10;

type Leg = 'lexical' | 'dense' | 'late_interaction';
const LEG_LABEL: Record<Leg, string> = {
  lexical: 'lexical (BM25)', dense: 'dense (MIPS)', late_interaction: 'late interaction',
};
const LEG_NAMES: Leg[] = ['lexical', 'dense', 'late_interaction'];
const LEG_COLOR: Record<Leg, string> = {
  lexical: '#c08457', dense: '#6a8caf', late_interaction: '#9b6abf',
};

type Metric = 'map' | 'ndcg';
const METRIC_LABEL: Record<Metric, string> = { map: 'MAP', ndcg: 'NDCG' };

// Panel A — the worked pair (lexical − dense), per metric. std_unpaired makes the unpaired SE at n=40
// equal se_unpaired; the lab recomputes both half-widths as 1.96·σ/√n.
type PairedStat = {
  meanD: number; stdD: number; stdUnpaired: number; sePaired: number; seUnpaired: number;
  t: number; p: number; dz: number; pairedSepN: number; unpairedOverlapN: number; powerN: number;
  corr: number; varRatio: number; perQDiff: number[];
};
const PAIRED: Record<Metric, PairedStat> = {
  map: {
    meanD: -0.1161, stdD: 0.1898, stdUnpaired: 0.2296, sePaired: 0.03, seUnpaired: 0.0363,
    t: -3.87, p: 4.03e-4, dz: -0.6119, pairedSepN: 11, unpairedOverlapN: 30, powerN: 21,
    corr: 0.323, varRatio: 0.685,
    perQDiff: [-0.43, -0.375, -0.513, 0.058, -0.336, 0.026, -0.184, -0.004, -0.114, 0.029, 0.075, -0.371, -0.096, -0.076, -0.051, -0.106, -0.099, -0.209, -0.125, -0.114, 0.085, 0.184, -0.24, -0.417, -0.033, -0.353, -0.291, -0.435, -0.147, -0.173, 0.127, 0.084, 0.056, 0.036, -0.133, -0.088, -0.125, -0.197, 0.293, 0.136],
  },
  ndcg: {
    meanD: -0.0491, stdD: 0.1885, stdUnpaired: 0.2454, sePaired: 0.0298, seUnpaired: 0.0388,
    t: -1.649, p: 0.1071, dz: -0.2608, pairedSepN: 57, unpairedOverlapN: 185, powerN: 116,
    corr: 0.445, varRatio: 0.589,
    perQDiff: [-0.329, -0.083, -0.222, 0.11, -0.133, 0.005, -0.271, -0.078, 0.054, 0.198, 0.085, -0.096, -0.058, -0.048, -0.01, -0.02, -0.118, -0.173, -0.006, -0.012, 0.175, 0.257, -0.149, -0.715, 0.121, -0.27, -0.178, -0.255, -0.151, -0.069, 0.03, 0.215, 0.172, 0.088, -0.018, 0.026, -0.152, -0.174, 0.341, -0.055],
  },
};

// Panel B — the permutation null (worked MAP pair) and the 3-pair correction grids.
const PERM = { nullStd: 0.0349, observed: -0.1161, tP: 4.03e-4, permP: 7.0e-4, bootP: 1.0e-4 };
type Grid = { pairs: [Leg, Leg][]; raw: number[]; bonf: number[]; holm: number[]; bh: number[] };
const PAIRWISE: Record<Metric, Grid> = {
  map: {
    pairs: [['lexical', 'dense'], ['lexical', 'late_interaction'], ['dense', 'late_interaction']],
    raw: [4.03e-4, 1.861e-11, 1.807e-7], bonf: [0.0012, 0, 0], holm: [0.0004, 0, 0], bh: [0.0004, 0, 0],
  },
  ndcg: {
    pairs: [['lexical', 'dense'], ['lexical', 'late_interaction'], ['dense', 'late_interaction']],
    raw: [0.1071, 5.095e-7, 3.496e-7], bonf: [0.3214, 0, 0], holm: [0.1071, 0, 0], bh: [0.1071, 0, 0],
  },
};

// Panel C — reliability bins [conf, acc, n] per leg × condition; TS recomputes ECE/MCE from them.
type Cond = 'raw' | 'platt' | 'isotonic';
const RELIABILITY: Record<Leg, Record<Cond, number[][]>> = {
  lexical: {
    raw: [[0.0029, 0.0056, 2851], [0.1899, 0.123, 496], [0.2517, 0.0937, 491], [0.3391, 0.1459, 473], [0.5018, 0.4254, 489]],
    platt: [[0.0095, 0.0056, 2851], [0.05, 0.123, 496], [0.0843, 0.0937, 491], [0.1746, 0.1459, 473], [0.4581, 0.4254, 489]],
    isotonic: [[0.0058, 0.0056, 2851], [0.1, 0.0263, 38], [0.1049, 0.1062, 725], [0.1225, 0.1183, 575], [0.391, 0.3895, 611]],
  },
  dense: {
    raw: [[0.1971, 0.0, 480], [0.3097, 0.0, 480], [0.3778, 0.0021, 480], [0.4376, 0.0063, 480], [0.4906, 0.0104, 480], [0.5433, 0.0229, 480], [0.5963, 0.0396, 480], [0.652, 0.0667, 480], [0.7191, 0.1354, 480], [0.8317, 0.55, 480]],
    platt: [[0.0001, 0.0, 480], [0.0003, 0.0, 480], [0.0009, 0.0021, 480], [0.0024, 0.0063, 480], [0.0055, 0.0104, 480], [0.0127, 0.0229, 480], [0.0291, 0.0396, 480], [0.0683, 0.0667, 480], [0.1757, 0.1354, 480], [0.5382, 0.55, 480]],
    isotonic: [[0.0, 0.0, 1039], [0.0015, 0.0015, 646], [0.0091, 0.0091, 329], [0.0113, 0.0113, 532], [0.032, 0.032, 656], [0.0536, 0.0536, 522], [0.1165, 0.1165, 515], [0.5009, 0.5009, 561]],
  },
  late_interaction: {
    raw: [[0.1796, 0.0, 480], [0.2581, 0.0, 479], [0.3075, 0.0, 481], [0.3564, 0.0, 480], [0.4154, 0.0, 480], [0.5079, 0.0, 480], [0.5829, 0.0063, 478], [0.6277, 0.0312, 481], [0.6768, 0.1455, 481], [0.8521, 0.65, 480]],
    platt: [[0.0, 0.0, 480], [0.0, 0.0, 479], [0.0001, 0.0, 481], [0.0001, 0.0, 480], [0.0005, 0.0, 480], [0.0038, 0.0, 480], [0.0155, 0.0063, 478], [0.0376, 0.0312, 481], [0.0996, 0.1455, 481], [0.6759, 0.65, 480]],
    isotonic: [[0.0006, 0.0006, 3339], [0.0274, 0.0275, 473], [0.1335, 0.1336, 479], [0.6331, 0.6306, 509]],
  },
};
const CAL_META: Record<Leg, { aucRaw: number; aucPlatt: number; plattA: number; plattB: number }> = {
  lexical: { aucRaw: 0.882313, aucPlatt: 0.882313, plattA: 0.994, plattB: -4.6953 },
  dense: { aucRaw: 0.926866, aucPlatt: 0.926866, plattA: 9.551, plattB: -6.1216 },
  late_interaction: { aucRaw: 0.974335, aucPlatt: 0.974335, plattA: 5.4067, plattB: -13.7099 },
};
const N_PAIRS = 4800;
const BASE_RATE = 0.0833;

// Panel D — drift over the degradation knob (per-query NDCG of the dense leg, oracle grades fixed).
const KNOB_LEVELS = [0, 0.05, 0.1, 0.2, 0.35, 0.5];
type DriftRow = { sigma: number; psi: number; ksD: number; ksP: number; meanNdcg: number; shiftSe: number };
const DRIFT_TABLE: DriftRow[] = [
  { sigma: 0.0, psi: 0.0, ksD: 0.0, ksP: 1.0, meanNdcg: 0.7699, shiftSe: 0.0 },
  { sigma: 0.05, psi: 0.2571, ksD: 0.25, ksP: 0.165, meanNdcg: 0.7272, shiftSe: -1.99 },
  { sigma: 0.1, psi: 1.1763, ksD: 0.475, ksP: 1.879e-4, meanNdcg: 0.6038, shiftSe: -7.72 },
  { sigma: 0.2, psi: 2.7043, ksD: 0.8, ksP: 5.393e-13, meanNdcg: 0.3948, shiftSe: -17.44 },
  { sigma: 0.35, psi: 3.0062, ksD: 0.925, ksP: 1.528e-18, meanNdcg: 0.236, shiftSe: -24.82 },
  { sigma: 0.5, psi: 3.0062, ksD: 0.925, ksP: 1.528e-18, meanNdcg: 0.1678, shiftSe: -27.99 },
];
const REF_NDCG = [0.921, 0.918, 0.709, 0.665, 0.928, 0.87, 0.827, 0.727, 0.758, 0.716, 0.787, 0.821, 0.866, 0.694, 0.963, 0.843, 0.906, 0.707, 0.955, 0.782, 0.446, 0.567, 0.262, 0.826, 0.777, 0.599, 0.837, 0.81, 0.706, 0.729, 0.791, 0.769, 0.75, 0.815, 0.829, 0.818, 0.892, 0.831, 0.608, 0.768];
const CUR_NDCG: Record<string, number[]> = {
  '0.05': [0.873, 0.886, 0.692, 0.501, 0.868, 0.746, 0.747, 0.715, 0.758, 0.732, 0.634, 0.743, 0.824, 0.67, 0.953, 0.668, 0.841, 0.7, 0.852, 0.616, 0.46, 0.539, 0.369, 0.621, 0.774, 0.54, 0.837, 0.651, 0.806, 0.712, 0.784, 0.85, 0.618, 0.651, 0.754, 0.858, 0.924, 0.962, 0.613, 0.744],
  '0.1': [0.835, 0.39, 0.581, 0.482, 0.757, 0.615, 0.558, 0.63, 0.578, 0.715, 0.479, 0.557, 0.755, 0.502, 0.846, 0.438, 0.787, 0.444, 0.767, 0.494, 0.467, 0.124, 0.361, 0.437, 0.757, 0.273, 0.817, 0.598, 0.737, 0.703, 0.538, 0.837, 0.506, 0.406, 0.727, 0.85, 0.747, 0.849, 0.559, 0.646],
  '0.2': [0.784, 0.064, 0.305, 0.351, 0.59, 0.307, 0.517, 0.271, 0.152, 0.624, 0.347, 0.354, 0.356, 0.36, 0.533, 0.199, 0.507, 0.305, 0.641, 0.166, 0.221, 0.158, 0.249, 0.348, 0.669, 0.121, 0.592, 0.474, 0.522, 0.288, 0.264, 0.61, 0.317, 0.124, 0.522, 0.721, 0.619, 0.606, 0.308, 0.324],
  '0.35': [0.727, 0.026, 0.088, 0.352, 0.347, 0.0, 0.439, 0.078, 0.059, 0.379, 0.196, 0.137, 0.088, 0.17, 0.317, 0.117, 0.198, 0.248, 0.319, 0.106, 0.156, 0.145, 0.268, 0.256, 0.49, 0.12, 0.492, 0.226, 0.288, 0.044, 0.222, 0.533, 0.199, 0.0, 0.305, 0.323, 0.538, 0.245, 0.199, 0.0],
  '0.5': [0.711, 0.0, 0.068, 0.258, 0.386, 0.0, 0.371, 0.0, 0.048, 0.309, 0.196, 0.123, 0.096, 0.17, 0.306, 0.074, 0.164, 0.246, 0.016, 0.015, 0.0, 0.159, 0.18, 0.203, 0.433, 0.099, 0.468, 0.0, 0.048, 0.0, 0.134, 0.406, 0.199, 0.0, 0.061, 0.246, 0.386, 0.0, 0.137, 0.0],
};
const NULL_PSI = 0.0594;
const NULL_KS_P = 0.92;
const SILENT = { sigma: 0.05, meanShift: -0.0427, meanShiftSe: -1.99, unpairedOverlap: true, pairedP: 1.99e-3, ksP: 0.165 };
const IN_OUT = { inputPsiCov: 0.4322, outcomePairedCov: 0.0, outcomePairedDecay: -0.602, decayP: 9.36e-22 };
const PSI_AMBER = 0.1;
const PSI_RED = 0.25;

const ACCENT = 'var(--color-accent)';
const MUTED = '#9aa3ad';
const POS_COLOR = '#6fa389';   // d_q > 0 (A wins)
const NEG_COLOR = '#c0726a';   // d_q < 0 (B wins)
const fmt = (x: number, n = 3) => x.toFixed(n);
const fmtP = (p: number) => (p < 1e-3 ? p.toExponential(1) : p.toFixed(3));

// --- closed-form TS recomputation -----------------------------------------------------
const ciHalf = (sd: number, n: number) => (n > 0 ? (1.96 * sd) / Math.sqrt(n) : 0);
// ECE = Σ_b (n_b/N)|acc_b − conf_b|; MCE = max_b|acc_b − conf_b| — from the baked [conf, acc, n] bins.
const eceOf = (bins: number[][]) => {
  const N = bins.reduce((s, b) => s + b[2], 0);
  return N ? bins.reduce((s, b) => s + (b[2] / N) * Math.abs(b[1] - b[0]), 0) : 0;
};
const mceOf = (bins: number[][]) => bins.reduce((m, b) => Math.max(m, Math.abs(b[1] - b[0])), 0);
// fixed [0,1] histogram (10 bins) of a per-query NDCG array.
const histProps = (arr: number[], nb = 10) => {
  const c = new Array(nb).fill(0);
  arr.forEach((v) => { c[Math.min(nb - 1, Math.max(0, Math.floor(v * nb)))]++; });
  return c.map((x) => x / arr.length);
};
// empirical CDF value F(x) and the KS sup-gap between two samples.
const cdfAt = (s: number[], x: number) => s.reduce((c, v) => c + (v <= x + 1e-12 ? 1 : 0), 0) / s.length;
const ksGap = (a: number[], b: number[]) => {
  const grid = [...a, ...b].sort((u, v) => u - v);
  let D = 0, at = grid[0] ?? 0;
  for (const x of grid) { const g = Math.abs(cdfAt(a, x) - cdfAt(b, x)); if (g > D) { D = g; at = x; } }
  return { D, at };
};

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
      <span style={{ minWidth: '11rem' }}>{label} = <strong>{display}</strong></span>
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
function LegPills({ leg, setLeg }: { leg: Leg; setLeg: (l: Leg) => void }) {
  return (
    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
      {LEG_NAMES.map((l) => (
        <button key={l} type="button" style={pill(leg === l)} onClick={() => setLeg(l)}>{LEG_LABEL[l]}</button>
      ))}
    </div>
  );
}
function MetricPills({ metric, setMetric }: { metric: Metric; setMetric: (m: Metric) => void }) {
  return (
    <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
      {(['map', 'ndcg'] as Metric[]).map((m) => (
        <button key={m} type="button" style={pill(metric === m)} onClick={() => setMetric(m)}>{METRIC_LABEL[m]}</button>
      ))}
    </div>
  );
}

// ===== Panel A — the paired test ===================================================================
function PairedPanel({ metric, setMetric, n, setN }: {
  metric: Metric; setMetric: (m: Metric) => void; n: number; setN: (v: number) => void;
}) {
  const s = PAIRED[metric];
  const pairedHalf = ciHalf(s.stdD, n);
  const unpairedHalf = ciHalf(s.stdUnpaired, n);
  const pairedExcl = s.meanD + pairedHalf < 0 || s.meanD - pairedHalf > 0;
  const unpairedExcl = s.meanD + unpairedHalf < 0 || s.meanD - unpairedHalf > 0;
  // d_q strip
  const W = 540, H = 120, padL = 32, padR = 10, padT = 12, padB = 22;
  const bw = (W - padL - padR) / N_QUERIES;
  const dMax = Math.max(...s.perQDiff.map((d) => Math.abs(d)), 0.1);
  const zeroY = padT + (H - padT - padB) / 2;
  const dy = (v: number) => zeroY - (v / dMax) * ((H - padT - padB) / 2);
  // CI axis
  const CW = 420, CH = 96, cPadL = 14, cPadR = 14, cPadT = 18;
  const xMin = Math.min(s.meanD - unpairedHalf, -0.02) - 0.02;
  const xMax = Math.max(s.meanD + unpairedHalf, 0.02) + 0.02;
  const cx = (v: number) => cPadL + (CW - cPadL - cPadR) * ((v - xMin) / (xMax - xMin));
  const ciBar = (yy: number, half: number, color: string, excl: boolean, label: string) => (
    <g>
      <line x1={cx(s.meanD - half)} y1={yy} x2={cx(s.meanD + half)} y2={yy} stroke={color} strokeWidth={2.6} />
      <line x1={cx(s.meanD - half)} y1={yy - 6} x2={cx(s.meanD - half)} y2={yy + 6} stroke={color} strokeWidth={2} />
      <line x1={cx(s.meanD + half)} y1={yy - 6} x2={cx(s.meanD + half)} y2={yy + 6} stroke={color} strokeWidth={2} />
      <circle cx={cx(s.meanD)} cy={yy} r={3.5} fill={color} />
      <text x={cPadL} y={yy - 9} fontSize={8.5} fill={color} fontFamily="var(--font-sans)">{label} {excl ? '· excludes 0' : '· contains 0'}</text>
    </g>
  );
  return (
    <div>
      <MetricPills metric={metric} setMetric={setMetric} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="per-query difference strip for the worked leg pair" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={zeroY} x2={W - padR} y2={zeroY} stroke="var(--color-border)" strokeWidth={1} />
        <text x={6} y={zeroY + 3} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">0</text>
        {s.perQDiff.map((d, i) => {
          const x = padL + i * bw;
          const y = dy(d);
          return <rect key={i} x={x + 0.6} y={Math.min(y, zeroY)} width={bw - 1.2} height={Math.abs(zeroY - y)} fill={d >= 0 ? POS_COLOR : NEG_COLOR} fillOpacity={0.85} />;
        })}
        <line x1={padL} y1={dy(s.meanD)} x2={W - padR} y2={dy(s.meanD)} stroke={ACCENT} strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={W - padR} y={dy(s.meanD) - 3} textAnchor="end" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">mean d = {fmt(s.meanD, 3)}</text>
        <text x={padL + 2} y={padT + 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">dₖ = {METRIC_LABEL[metric]}(lexical) − {METRIC_LABEL[metric]}(dense), per query</text>
      </svg>
      <Slider label="query count n" value={n} min={5} max={N_QUERIES} step={1} onChange={(v) => setN(Math.round(v))} display={`${n}`} />
      <svg viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label="paired versus unpaired confidence interval on the mean difference" style={{ width: '100%', maxWidth: CW, height: 'auto', display: 'block' }}>
        <line x1={cx(0)} y1={cPadT - 8} x2={cx(0)} y2={CH - 6} stroke="var(--color-border)" strokeWidth={1.4} />
        <text x={cx(0)} y={CH - 1} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">0 (no difference)</text>
        {ciBar(cPadT + 14, pairedHalf, ACCENT, pairedExcl, 'paired')}
        {ciBar(cPadT + 48, unpairedHalf, MUTED, unpairedExcl, 'unpaired (two-CI)')}
      </svg>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="paired t · p" value={`${fmt(s.t, 2)} · ${fmtP(s.p)}`} accent={s.p < 0.05} />
        <Readout label="paired separates at n" value={`${s.pairedSepN}`} accent />
        <Readout label="unpaired needs n" value={`${s.unpairedOverlapN}`} />
        <Readout label="80%-power n" value={`${s.powerN}`} />
        <Readout label="var(d)/(varA+varB)" value={fmt(s.varRatio, 2)} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Pairing cancels the shared per-query difficulty (the legs correlate {fmt(s.corr, 2)} across queries), so
        var(d) is only <strong>{fmt(s.varRatio, 2)}×</strong> the unpaired variance and the{' '}
        <span style={{ color: ACCENT }}>paired</span> interval excludes 0 at <strong>n = {s.pairedSepN}</strong> where the{' '}
        <span style={{ color: MUTED }}>unpaired</span> two-CI read needs <strong>{s.unpairedOverlapN}</strong>.
        {metric === 'ndcg'
          ? <> This closes the NDCG cliffhanger: <strong>185 → 116</strong> (rigorous 80%-power) versus <strong>57</strong> (a single interval clearing 0); at all {N_QUERIES} queries the gap is <strong>not yet</strong> significant (p = {fmt(s.p, 3)}).</>
          : <> The MAP gap <em>is</em> significant at {N_QUERIES} queries (p = {fmtP(s.p)}) — pairing tightens the test without manufacturing significance.</>}
      </p>
    </div>
  );
}

// ===== Panel B — distribution-free tests & corrections =============================================
function PermPanel({ metric, setMetric }: { metric: Metric; setMetric: (m: Metric) => void }) {
  const grid = PAIRWISE[metric];
  // permutation null ~ N(0, nullStd) (sign-flip CLT); shade beyond |observed|.
  const W = 380, H = 150, padL = 10, padR = 10, padT = 12, padB = 22;
  const xR = 4 * PERM.nullStd;
  const px = (v: number) => padL + (W - padL - padR) * ((v + xR) / (2 * xR));
  const dens = (v: number) => Math.exp(-(v * v) / (2 * PERM.nullStd * PERM.nullStd));
  const yMax = 1;
  const py = (v: number) => H - padB - (H - padT - padB) * (v / yMax);
  const pts = Array.from({ length: 81 }, (_v, i) => { const v = -xR + (2 * xR * i) / 80; return [px(v), py(dens(v))] as [number, number]; });
  const curve = pts.map(([x, y], i) => (i ? 'L' : 'M') + fmt(x, 1) + ' ' + fmt(y, 1)).join(' ');
  const obs = Math.max(-xR, Math.min(xR, PERM.observed));
  const tail = (sign: number) => {
    const seg = pts.filter(([x]) => (sign < 0 ? x <= px(-Math.abs(obs)) : x >= px(Math.abs(obs))));
    if (!seg.length) return '';
    const base = `M ${fmt(seg[0][0], 1)} ${py(0)} ` + seg.map(([x, y]) => `L ${fmt(x, 1)} ${fmt(y, 1)}`).join(' ');
    return base + ` L ${fmt(seg[seg.length - 1][0], 1)} ${py(0)} Z`;
  };
  const cell = (v: number) => {
    const sig = v < 0.05;
    return { background: sig ? 'rgba(111,163,137,0.22)' : 'transparent', color: sig ? 'var(--color-text)' : 'var(--color-text-secondary)', fontWeight: sig ? 600 : 400 };
  };
  const rows: [string, number[]][] = [['raw p', grid.raw], ['Bonferroni', grid.bonf], ['Holm', grid.holm], ['BH (FDR)', grid.bh]];
  return (
    <div>
      <MetricPills metric={metric} setMetric={setMetric} />
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="sign-flip permutation null distribution of the mean difference" style={{ width: '100%', maxWidth: W, height: 'auto', flex: '1 1 300px' }}>
          <line x1={padL} y1={py(0)} x2={W - padR} y2={py(0)} stroke="var(--color-border)" strokeWidth={1} />
          <path d={tail(-1)} fill={ACCENT} fillOpacity={0.25} />
          <path d={tail(1)} fill={ACCENT} fillOpacity={0.25} />
          <path d={curve} fill="none" stroke={MUTED} strokeWidth={1.8} />
          <line x1={px(PERM.observed)} y1={py(0)} x2={px(PERM.observed)} y2={padT} stroke={ACCENT} strokeWidth={2} />
          <text x={px(PERM.observed)} y={padT - 2} textAnchor="middle" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">observed</text>
          <text x={px(0)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">0</text>
          <text x={(padL + W - padR) / 2} y={padT + 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">sign-flip null of mean d (worked MAP pair)</text>
        </svg>
        <div style={{ flex: '1 1 150px', fontSize: '0.78rem', fontFamily: 'var(--font-sans)' }}>
          <Readout label="t-test p" value={fmtP(PERM.tP)} />
          <div style={{ height: '0.4rem' }} />
          <Readout label="permutation p" value={fmtP(PERM.permP)} accent />
          <div style={{ height: '0.4rem' }} />
          <Readout label="bootstrap p" value={fmtP(PERM.bootP)} />
          <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem', lineHeight: 1.4 }}>
            All three agree — the t-test's normality is only a convenience.
          </p>
        </div>
      </div>
      <div style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
        <table style={{ borderCollapse: 'collapse', fontFamily: 'var(--font-sans)', fontSize: '0.74rem', width: '100%' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '0.25rem 0.5rem', color: 'var(--color-text-secondary)' }}>correction</th>
              {grid.pairs.map((pr, j) => (
                <th key={j} style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: 'var(--color-text-secondary)' }}>{pr[0].slice(0, 3)}–{pr[1].slice(0, 3)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, vals]) => (
              <tr key={name}>
                <td style={{ padding: '0.25rem 0.5rem', color: 'var(--color-text-secondary)' }}>{name}</td>
                {vals.map((v, j) => (
                  <td key={j} style={{ padding: '0.25rem 0.5rem', textAlign: 'right', ...cell(v) }}>{fmtP(v)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Three legs means three simultaneous tests, so the family-wise error inflates. {metric === 'ndcg'
          ? <>On NDCG the genuine separations survive every correction, while the marginal lexical–dense pair is pushed from {fmt(grid.raw[0], 3)} to <strong>{fmt(grid.bonf[0], 2)}</strong> (Bonferroni) — firmly not distinguishable.</>
          : <>On MAP every pair is real and survives Bonferroni, Holm, and BH (shaded = reject at 0.05).</>}
      </p>
    </div>
  );
}

// ===== Panel C — score calibration =================================================================
function CalibrationPanel({ leg, setLeg, cond, setCond }: {
  leg: Leg; setLeg: (l: Leg) => void; cond: Cond; setCond: (c: Cond) => void;
}) {
  const bins = RELIABILITY[leg][cond];
  const meta = CAL_META[leg];
  const ece = eceOf(bins);
  const mce = mceOf(bins);
  const W = 300, H = 300, pad = 36;
  const ax = (v: number) => pad + (W - 2 * pad) * v;
  const ay = (v: number) => H - pad - (H - 2 * pad) * v;
  const maxN = Math.max(...bins.map((b) => b[2]));
  const r = (cnt: number) => 3 + 7 * Math.sqrt(cnt / maxN);
  return (
    <div>
      <LegPills leg={leg} setLeg={setLeg} />
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
        {(['raw', 'platt', 'isotonic'] as Cond[]).map((c) => (
          <button key={c} type="button" style={pill(cond === c)} onClick={() => setCond(c)}>{c === 'raw' ? 'raw score' : c === 'platt' ? 'Platt' : 'isotonic'}</button>
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
              <line x1={ax(b[0])} y1={ay(b[0])} x2={ax(b[0])} y2={ay(b[1])} stroke={LEG_COLOR[leg]} strokeWidth={1} strokeOpacity={0.5} />
              <circle cx={ax(b[0])} cy={ay(b[1])} r={r(b[2])} fill={LEG_COLOR[leg]} fillOpacity={0.8} />
            </g>
          ))}
          <text x={(W) / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">confidence (mean score)</text>
          <text x={12} y={H / 2} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${H / 2})`}>empirical relevance</text>
        </svg>
        <div style={{ flex: '1 1 180px' }}>
          <Readout label={`ECE (${cond})`} value={fmt(ece, 4)} accent={cond !== 'raw'} />
          <div style={{ height: '0.5rem' }} />
          <Readout label="MCE (worst bin)" value={fmt(mce, 4)} />
          <div style={{ height: '0.5rem' }} />
          <Readout label="AUC raw → recalibrated" value={`${fmt(meta.aucRaw, 4)} = ${fmt(meta.aucPlatt, 4)}`} />
          <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem', lineHeight: 1.4 }}>
            Platt is σ({fmt(meta.plattA, 2)}·s {meta.plattB < 0 ? '−' : '+'} {fmt(Math.abs(meta.plattB), 2)}), strictly
            monotone — so the AUC (and every ranking metric) is <strong>identical</strong> before and after.
          </p>
        </div>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Dots above the diagonal are under-confident, below are over-confident; dot size is the bin count, and the
        shaded drop to the diagonal summed is the ECE. The smooth cosine/MaxSim scores are wildly{' '}
        <strong>over-confident</strong> raw (ECE up to 0.43), which is exactly why rank fusion discards scores and fuses
        ranks. <strong>Platt</strong> and <strong>isotonic</strong> pull the dots onto the diagonal — lowering ECE without
        moving a single ranking ({N_PAIRS} pairs, base rate {fmt(BASE_RATE, 3)}). Calibration is orthogonal to ranking.
      </p>
    </div>
  );
}

// ===== Panel D — drift detection ===================================================================
function DriftPanel({ knob, setKnob }: { knob: number; setKnob: (v: number) => void }) {
  const sigma = KNOB_LEVELS[knob];
  const row = DRIFT_TABLE[knob];
  const cur = sigma === 0 ? REF_NDCG : (CUR_NDCG[String(sigma)] ?? REF_NDCG);
  const pRef = histProps(REF_NDCG);
  const pCur = histProps(cur);
  const ks = ksGap(REF_NDCG, cur);
  const light = row.psi > PSI_RED ? '#c0726a' : row.psi > PSI_AMBER ? '#c79a3b' : '#6fa389';
  const lightLabel = row.psi > PSI_RED ? 'significant shift' : row.psi > PSI_AMBER ? 'moderate shift' : 'stable';
  // overlaid histograms
  const W = 300, H = 150, padL = 28, padR = 8, padT = 10, padB = 22;
  const nb = pRef.length;
  const bw = (W - padL - padR) / nb;
  const hMax = Math.max(...pRef, ...pCur, 0.05);
  const hy = (v: number) => H - padB - (H - padT - padB) * (v / hMax);
  // KS staircase
  const SW = 300, SH = 150, sPadL = 28, sPadR = 8, sPadT = 10, sPadB = 22;
  const sx = (v: number) => sPadL + (SW - sPadL - sPadR) * v;
  const sy = (v: number) => SH - sPadB - (SH - sPadT - sPadB) * v;
  const step = (s: number[]) => {
    const srt = [...s].sort((a, b) => a - b);
    let d = `M ${sx(0)} ${sy(0)}`;
    srt.forEach((v, i) => { d += ` L ${fmt(sx(v), 1)} ${fmt(sy(i / srt.length), 1)} L ${fmt(sx(v), 1)} ${fmt(sy((i + 1) / srt.length), 1)}`; });
    d += ` L ${sx(1)} ${sy(1)}`;
    return d;
  };
  return (
    <div>
      <Slider label="embedding degradation σ" value={knob} min={0} max={KNOB_LEVELS.length - 1} step={1}
        onChange={(v) => setKnob(Math.round(v))} display={`${fmt(sigma, 2)}`} />
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="reference versus current per-query NDCG histograms" style={{ width: '100%', maxWidth: W, height: 'auto', flex: '1 1 260px' }}>
          <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
          {pRef.map((v, i) => (<rect key={`r${i}`} x={padL + i * bw + 1} y={hy(v)} width={bw / 2 - 1} height={(H - padB) - hy(v)} fill={MUTED} fillOpacity={0.8} />))}
          {pCur.map((v, i) => (<rect key={`c${i}`} x={padL + i * bw + bw / 2} y={hy(v)} width={bw / 2 - 1} height={(H - padB) - hy(v)} fill={ACCENT} fillOpacity={0.75} />))}
          <text x={padL + 2} y={padT + 4} fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">▮ then (reference)</text>
          <text x={padL + 2} y={padT + 15} fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">▮ now (current)</text>
          <text x={(padL + W - padR) / 2} y={H - 5} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">per-query NDCG</text>
        </svg>
        <svg viewBox={`0 0 ${SW} ${SH}`} role="img" aria-label="empirical CDF staircases with the KS sup-gap" style={{ width: '100%', maxWidth: SW, height: 'auto', flex: '1 1 260px' }}>
          <line x1={sPadL} y1={SH - sPadB} x2={SW - sPadR} y2={SH - sPadB} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={sPadL} y1={sPadT} x2={sPadL} y2={SH - sPadB} stroke="var(--color-border)" strokeWidth={1} />
          <path d={step(REF_NDCG)} fill="none" stroke={MUTED} strokeWidth={1.6} />
          <path d={step(cur)} fill="none" stroke={ACCENT} strokeWidth={1.6} />
          <line x1={sx(ks.at)} y1={sy(cdfAt(REF_NDCG, ks.at))} x2={sx(ks.at)} y2={sy(cdfAt(cur, ks.at))} stroke="#c0726a" strokeWidth={2.4} />
          <text x={(sPadL + SW - sPadR) / 2} y={sPadT + 4} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">empirical CDFs · KS gap D = {fmt(ks.D, 3)}</text>
          <text x={(sPadL + SW - sPadR) / 2} y={SH - 5} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">NDCG</text>
        </svg>
      </div>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.4rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ width: '0.9rem', height: '0.9rem', borderRadius: '50%', background: light, display: 'inline-block' }} />
          <Readout label={`PSI (${lightLabel})`} value={fmt(row.psi, 3)} />
        </div>
        <Readout label="KS p-value" value={fmtP(row.ksP)} accent={row.ksP < 0.05} />
        <Readout label="mean NDCG shift" value={`${fmt(row.shiftSe, 1)} SE`} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Drift is a two-sample test: reference (then) vs current (now). PSI = Σ(pᵦ−qᵦ)ln(pᵦ/qᵦ) is the credit-risk
        traffic light (green &lt; {PSI_AMBER}, amber, red &gt; {PSI_RED}); the KS statistic is the sup-gap between the
        two CDF staircases. Under the matched-n <strong>null</strong> the detector is silent (mean PSI {fmt(NULL_PSI, 3)},
        KS p {fmt(NULL_KS_P, 2)}). The subtle cases: a <strong>silent decay</strong> at σ = {SILENT.sigma} the aggregate
        two-CI read calls a tie is caught by the <strong>paired</strong> test (p = {fmtP(SILENT.pairedP)}); and a pure{' '}
        <strong>covariate shift</strong> fires the input PSI ({fmt(IN_OUT.inputPsiCov, 2)}) with no quality loss
        (paired outcome 0), so input drift alone cannot diagnose decay — you need a labelled paired outcome test.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'paired' | 'permutation' | 'calibration' | 'drift';
const TEX: Record<Panel, string> = {
  paired: 't = \\dfrac{\\bar d}{s_d/\\sqrt{n}}, \\quad d_q = \\mathrm{metric}_A(q) - \\mathrm{metric}_B(q), \\quad \\mathrm{var}(d) = \\sigma_A^2 + \\sigma_B^2 - 2\\,\\mathrm{cov}(A,B)',
  permutation: 'p = \\Pr_{\\,s\\in\\{\\pm1\\}^n}\\!\\big(|\\,\\overline{s\\odot d}\\,| \\ge |\\bar d|\\big), \\quad p^{\\mathrm{Holm}}_{(i)} = (m-i+1)\\,p_{(i)}',
  calibration: '\\mathrm{ECE} = \\sum_{b}\\frac{n_b}{N}\\,\\big|\\mathrm{acc}_b - \\mathrm{conf}_b\\big|, \\quad \\mathrm{BS} = \\mathrm{rel} - \\mathrm{res} + \\mathrm{unc}',
  drift: '\\mathrm{PSI} = \\sum_b (p_b - q_b)\\ln\\frac{p_b}{q_b} = \\mathrm{KL}(p\\|q) + \\mathrm{KL}(q\\|p), \\quad D = \\sup_x |F_{\\mathrm{ref}}(x) - F_{\\mathrm{cur}}(x)|',
};

export default memo(function SignificanceTestingLaboratory() {
  const [panel, setPanel] = useState<Panel>('paired');
  const [metric, setMetric] = useState<Metric>('ndcg');
  const [n, setN] = useState(40);
  const [permMetric, setPermMetric] = useState<Metric>('ndcg');
  const [leg, setLeg] = useState<Leg>('dense');
  const [cond, setCond] = useState<Cond>('raw');
  const [knob, setKnob] = useState(2);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="significance" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'paired')} onClick={() => setPanel('paired')}>A · paired test</button>
        <button type="button" style={pill(panel === 'permutation')} onClick={() => setPanel('permutation')}>B · distribution-free</button>
        <button type="button" style={pill(panel === 'calibration')} onClick={() => setPanel('calibration')}>C · calibration</button>
        <button type="button" style={pill(panel === 'drift')} onClick={() => setPanel('drift')}>D · drift detection</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'paired' && <PairedPanel metric={metric} setMetric={setMetric} n={n} setN={setN} />}
      {panel === 'permutation' && <PermPanel metric={permMetric} setMetric={setPermMetric} />}
      {panel === 'calibration' && <CalibrationPanel leg={leg} setLeg={setLeg} cond={cond} setCond={setCond} />}
      {panel === 'drift' && <DriftPanel knob={knob} setKnob={setKnob} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_DOCS} synthetic finance documents, {N_QUERIES} queries, top-{R_SIZE} relevant; the three legs and the
        MaxSim oracle are the prerequisites' shared corpus. Numbers mirror{' '}
        <code>significance_testing_calibration.py</code>; the lab recomputes the CI half-widths, ECE/MCE, histograms,
        CDFs, and the KS gap in closed form, and bakes the corpus-derived means, p-values, reliability bins, and PSI.
      </p>
    </div>
  );
});
