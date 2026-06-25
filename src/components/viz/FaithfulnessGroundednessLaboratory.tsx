import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Faithfulness and Groundedness Laboratory — four panels for the `faithfulness-groundedness` topic. A
 * generated answer is a SET of atomic claims; faithfulness is their PRECISION against the retrieved
 * context, groundedness their RECALL of the supportable facts.
 *   A. The answer, decomposed. A worked finance answer's claims, colored by truth (supported / hallucinated)
 *      with the noisy judge's confidence; a confidence cut τ slider drops low-confidence claims and the
 *      live faithfulness = precision and coverage = recall readouts recompute.
 *   B. The faithfulness–coverage frontier. The PR plane: raising τ trades coverage for faithfulness along the
 *      back-off frontier; the terse and verbose answers sit at the extremes; the conformal-risk-control
 *      guarantee certifies faithfulness ≥ 1 − α. An inset shows F1 peaking at an interior answer length.
 *   C. Judge calibration. The reliability diagram (raw / Platt / isotonic) and ECE — the raw judge is
 *      over-confident; recalibration lowers ECE; Rogan–Gladen debiases the naive faithfulness number.
 *   D. Bits of grounding. Per-claim pointwise mutual information with the context: supported claims carry
 *      positive bits, hallucinated claims non-positive — the information-theoretic face of faithfulness.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every baked constant below is mirrored TO THE DECIMAL from
 * notebooks/faithfulness-groundedness/faithfulness_groundedness.py (viz_constants()). Matching asserts:
 * test_perfect_judge_collapse / test_faithfulness_is_precision_coverage_is_recall /
 * test_faithfulness_coverage_diverge / test_interior_f1_optimum / test_judge_overlaps_not_vacuous /
 * test_naive_faithfulness_biased_rogan_gladen_corrects / test_raw_judge_overconfident_recalibration_lowers_ece /
 * test_crc_monotone_and_controls / test_bits_sign_split. The lab recomputes ONLY closed forms in TS (Panel
 * A's precision/coverage at the live τ from the baked claim labels; Panel C's ECE from the baked reliability
 * bins; Panel D's histogram bins). Everything corpus-derived — the judge confidences, the frontier, the
 * calibration bins, the per-claim bits — is baked (TS cannot reproduce vMF draws, Platt/isotonic fits, or
 * PMI). Change a number here -> change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const FG_N_DOCS = 16;
const FG_CTX_K = 3;
const COS_SUPPORT = 0.78;

// Panel A — the worked finance answer (nine atomic claims). Labels are narrative; y / conf / bits / attr
// are baked from the .py. y: 1 = supported by context, 0 = hallucinated. attr: local context-fact index.
const WORKED_LABELS = [
  'Q3 revenue was $4.2B', 'Revenue up 8% YoY', 'Beat consensus', 'Gross margin expanded to 41%',
  'Raised full-year guidance', 'Cloud segment grew 25%', '$2B buyback announced',
  'New data center in Austin', 'FX headwinds ~2 points',
];
const WORKED_Y = [1, 1, 1, 1, 1, 0, 0, 0, 1];
const WORKED_CONF = [0.86, 0.859, 0.857, 0.86, 0.858, 0.187, 0.377, 0.467, 0.857];
const WORKED_BITS = [0.728, 0.63, 0.307, 0.728, 0.63, -0.262, -0.262, -0.262, 0.307];
const WORKED_ATTR = [0, 1, 2, 0, 1, -1, -1, -1, 2];

// Panel B — the verbosity sweep (answer length -> precision, recall, F1) and the τ back-off frontier.
const VERBOSITY = [
  { n: 1, p: 0.594, r: 0.198, f1: 0.297 }, { n: 2, p: 0.562, r: 0.375, f1: 0.45 },
  { n: 3, p: 0.562, r: 0.562, f1: 0.562 }, { n: 4, p: 0.586, r: 0.75, f1: 0.643 },
  { n: 5, p: 0.588, r: 0.875, f1: 0.656 }, { n: 6, p: 0.594, r: 0.958, f1: 0.639 },
  { n: 8, p: 0.613, r: 0.99, f1: 0.54 }, { n: 10, p: 0.638, r: 1.0, f1: 0.462 },
  { n: 12, p: 0.63, r: 1.0, f1: 0.4 },
];
type FrontierRow = { tau: number; p: number; r: number; ret: number };
const FRONTIER: FrontierRow[] = [
  { tau: 0.0, p: 0.621, r: 0.979, ret: 1.0 }, { tau: 0.04, p: 0.723, r: 0.969, ret: 0.879 },
  { tau: 0.08, p: 0.764, r: 0.958, ret: 0.824 }, { tau: 0.12, p: 0.781, r: 0.948, ret: 0.797 },
  { tau: 0.16, p: 0.8, r: 0.938, ret: 0.746 }, { tau: 0.2, p: 0.81, r: 0.896, ret: 0.715 },
  { tau: 0.24, p: 0.808, r: 0.865, ret: 0.703 }, { tau: 0.28, p: 0.815, r: 0.854, ret: 0.688 },
  { tau: 0.32, p: 0.814, r: 0.844, ret: 0.68 }, { tau: 0.36, p: 0.821, r: 0.844, ret: 0.668 },
  { tau: 0.4, p: 0.842, r: 0.844, ret: 0.648 }, { tau: 0.44, p: 0.858, r: 0.844, ret: 0.633 },
  { tau: 0.48, p: 0.863, r: 0.833, ret: 0.621 }, { tau: 0.52, p: 0.883, r: 0.833, ret: 0.605 },
  { tau: 0.56, p: 0.888, r: 0.802, ret: 0.586 }, { tau: 0.6, p: 0.888, r: 0.792, ret: 0.578 },
  { tau: 0.64, p: 0.895, r: 0.771, ret: 0.559 }, { tau: 0.68, p: 0.897, r: 0.763, ret: 0.556 },
  { tau: 0.72, p: 0.9, r: 0.742, ret: 0.54 }, { tau: 0.76, p: 0.911, r: 0.71, ret: 0.504 },
  { tau: 0.8, p: 0.913, r: 0.701, ret: 0.487 }, { tau: 0.84, p: 0.928, r: 0.704, ret: 0.417 },
];
const CRC = { lambda: 0.84, alpha: 0.1, realized: 0.031, retention: 0.305 };
const DIVERGENCE = {
  terse: { p: 1.0, r: 0.333 },
  verbose: { p: 0.534, r: 1.0 },
};

// Panel C — judge calibration: reliability bins [conf, acc, count], ECE, AUC, Rogan–Gladen.
type Bin = [number, number, number];
const REL_RAW: Bin[] = [
  [0.048, 0.038, 26], [0.221, 0.32, 25], [0.412, 0.538, 26], [0.61, 0.32, 25], [0.785, 0.846, 26],
  [0.898, 0.692, 26], [0.951, 0.72, 25], [0.973, 0.8, 25], [0.991, 0.923, 26], [0.999, 1.0, 26],
];
const REL_PLATT: Bin[] = [
  [0.124, 0.038, 26], [0.222, 0.32, 25], [0.374, 0.538, 26], [0.567, 0.32, 25], [0.722, 0.846, 26],
  [0.803, 0.68, 25], [0.834, 0.731, 26], [0.847, 0.8, 25], [0.856, 0.923, 26], [0.86, 1.0, 26],
];
const REL_ISO: Bin[] = [
  [0.0, 0.0, 20], [0.222, 0.222, 27], [0.385, 0.385, 13], [0.476, 0.476, 42], [0.705, 0.705, 61],
  [0.85, 0.85, 40], [0.962, 0.962, 53],
];
const ECE = { raw: 0.125, platt: 0.12, iso: 0.0 };
const AUC = { raw: 0.832, platt: 0.832 };
const RG = { naive: 0.699, corrected: 0.621, oracle: 0.621, sens: 0.818, spec: 0.495 };

// Panel D — bits-of-grounding (per-claim PMI), supported vs hallucinated.
const BITS_SUP = [0.728, 0.63, 0.307, 0.728, 0.63, 0.307, 0.163, 0.71, 0.378, 0.163, 0.103, 0.764, 0.358, 0.103, 0.764, 0.53, 0.579, 0.996, 0.53, 0.579, 0.229, 1.299, 1.046, 0.229, 1.299, 0.809, 0.209, 0.344, 0.802, 0.453, 0.344, 0.802, 0.453, 0.295, 0.132, 0.392, 0.295, 0.616, 0.674, 1.173, 0.616, 0.393, 0.399, 0.877, 0.393, 0.393, 0.723, 0.796, 0.393, 0.723, 0.491, 0.491, 0.431, 0.491, 0.491, 0.431, 0.278, 0.254, 0.239, 0.278, 0.543, 0.578, 0.467, 0.543, 0.578, 0.467, 0.543, 0.578, 1.106, 0.94, 1.244, 1.106, 0.226, 0.588, 0.569, 0.226, 0.588, 0.256, 0.212, 0.31, 0.256, 0.212, 0.495, 0.675, 0.72, 0.495, 0.675, 0.72, 0.495, 0.121, 0.406, 0.421, 0.121, 0.406, 0.421, 0.121, 0.654, 0.624, 1.01, 0.654, 0.624, 1.01, 0.654, 0.256, 0.263, 0.544, 0.256, 0.263, 0.544, 0.26, 0.196, 0.378, 0.26, 0.196, 0.376, 0.623, 0.45, 0.353, 0.248, 0.428, 0.254, 0.269, 0.428, 0.254, 0.269, 0.436, 0.864, 0.302, 0.154, 0.728, -0.582, 0.154, 0.728, 0.8, 0.637, 0.74, 0.8, 0.637, 0.74, 0.98, 0.29, -1.334, 0.98, 0.29, -1.334, 1.105, 0.899, 1.285, -1.535, 1.105, 1.285, 0.804, 0.622, 0.718, 0.804, 0.622, 0.58, 0.35, 0.134];
const BITS_HAL = [-0.262, -0.262, -1.449, -1.449, -1.449, -1.449, -0.931, -0.931, -0.931, -1.084, -1.084, -1.084, -1.816, -1.816, -1.816, -0.728, -0.728, -0.728, -0.728, -0.728, -0.728, -1.409, -1.409, -1.365, -1.365, -1.365, -1.365, -0.326, -0.326, -0.326, -0.326, 0.278, 0.278, 0.278, 0.278, -0.876, -0.876, -0.876, -1.046, -1.046, -1.164, -1.164, -1.164, -1.164, -0.502, -0.502, -0.502, -0.502, -1.05, -1.05, -1.05, -0.563, -0.563, -0.563, -0.626, -0.672, -0.276, -1.057, -1.057, -1.074, -1.074, -1.074, -0.551, -0.551, -0.551, -0.551, -0.551, -0.589, -0.589, -0.589, -0.589, -0.589, -0.589, -1.391, -1.391, -0.085, -0.085, -0.085, -0.085, -0.085, -0.815, -0.815, -0.815, -1.0, -1.0, -0.466, -0.466, 0.899, 0.899, -1.019, -1.019, -1.019, -0.185, -0.185, -0.185, -0.185, -0.185];
const BITS_SUMMARY = { mean_sup: 0.493, mean_hal: -0.73, frac_hal_nonpos: 0.938 };

// --- theme + helpers ------------------------------------------------------------------
const ACCENT = 'var(--color-accent)';
const MUTED = '#9aa3ad';
const POS_COLOR = '#6fa389';   // supported / faithful / certified
const NEG_COLOR = '#c0726a';   // hallucinated / unfaithful
const GUAR_COLOR = '#6a8caf';  // the conformal guarantee
const fmt = (x: number, n = 3) => (Number.isFinite(x) ? x.toFixed(n) : '∞');
const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

function histo(vals: number[], nb: number, lo: number, hi: number): number[] {
  const c = new Array(nb).fill(0);
  for (const v of vals) {
    const t = (v - lo) / (hi - lo);
    const b = Math.min(nb - 1, Math.max(0, Math.floor(t * nb)));
    c[b] += 1;
  }
  return c;
}
const eceOf = (bins: Bin[]) => {
  const N = bins.reduce((s, b) => s + b[2], 0) || 1;
  return bins.reduce((s, [conf, acc, n]) => s + (n / N) * Math.abs(acc - conf), 0);
};

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
      <span style={{ minWidth: '13rem' }}>{label} = <strong>{display}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
    </label>
  );
}
const Row = ({ children }: { children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: '1.4rem', flexWrap: 'wrap', margin: '0.6rem 0 0.2rem' }}>{children}</div>
);
const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

// ===== Panel A — the answer, decomposed ============================================================
function claimMetricsAt(tau: number) {
  const keep = WORKED_CONF.map((c) => c >= tau);
  const nKeep = keep.filter(Boolean).length;
  const supKeep = keep.filter((k, i) => k && WORKED_Y[i] === 1).length;
  const facts = new Set<number>();
  keep.forEach((k, i) => { if (k && WORKED_Y[i] === 1) facts.add(WORKED_ATTR[i]); });
  const precision = nKeep ? supKeep / nKeep : 1;
  const coverage = facts.size / FG_CTX_K;
  const hallKeep = keep.filter((k, i) => k && WORKED_Y[i] === 0).length;
  return { keep, nKeep, precision, coverage, hallKeep };
}
function ClaimsPanel({ tau, setTau }: { tau: number; setTau: (v: number) => void }) {
  const W = 560, rowH = 22, padL = 200, padT = 8, padR = 14;
  const H = padT + WORKED_LABELS.length * rowH + 8;
  const sx = (v: number) => padL + v * (W - padL - padR);
  const m = claimMetricsAt(tau);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="the worked answer's claims by judge confidence, colored by truth, with the confidence cut" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {WORKED_LABELS.map((label, i) => {
          const y = padT + i * rowH;
          const supported = WORKED_Y[i] === 1;
          const kept = m.keep[i];
          const color = supported ? POS_COLOR : NEG_COLOR;
          return (
            <g key={i} opacity={kept ? 1 : 0.28}>
              <text x={padL - 8} y={y + rowH / 2 + 3} textAnchor="end" fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">{label}</text>
              <rect x={padL} y={y + 3} width={Math.max(1, sx(WORKED_CONF[i]) - padL)} height={rowH - 8} rx={2} fill={color} fillOpacity={kept ? 0.85 : 0.4} />
              <text x={W - 6} y={y + rowH / 2 + 3} textAnchor="end" fontSize={8.5} fill={WORKED_BITS[i] > 0 ? POS_COLOR : NEG_COLOR} fontFamily="var(--font-sans)">{WORKED_BITS[i] > 0 ? '+' : ''}{fmt(WORKED_BITS[i], 2)}b</text>
            </g>
          );
        })}
        <line x1={sx(tau)} y1={padT - 2} x2={sx(tau)} y2={H - 4} stroke={ACCENT} strokeWidth={1.6} strokeDasharray="4 3" />
        <text x={sx(tau)} y={H - 1} textAnchor="middle" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">τ = {fmt(tau, 2)}</text>
      </svg>
      <Slider label="confidence cut τ (retain claims with conf ≥ τ)" value={tau} min={0} max={1} step={0.01}
        onChange={setTau} display={fmt(tau, 2)} />
      <Row>
        <Readout label="faithfulness = precision" value={fmt(m.precision, 2)} accent />
        <Readout label="coverage = recall" value={fmt(m.coverage, 2)} />
        <Readout label="claims retained" value={`${m.nKeep} / ${WORKED_LABELS.length}`} />
        <Readout label="hallucinations kept" value={String(m.hallKeep)} accent={m.hallKeep > 0} />
      </Row>
    </div>
  );
}

// ===== Panel B — the faithfulness–coverage frontier ================================================
function FrontierPanel({ ti, setTi }: { ti: number; setTi: (v: number) => void }) {
  const W = 560, H = 250, padL = 42, padR = 14, padT = 12, padB = 34;
  const x = (r: number) => padL + r * (W - padL - padR);     // coverage on x
  const y = (p: number) => H - padB - p * (H - padT - padB); // faithfulness on y
  const row = FRONTIER[ti];
  const guar = 1 - CRC.alpha;
  const path = FRONTIER.map((d, i) => `${i ? 'L' : 'M'}${x(d.r)},${y(d.p)}`).join(' ');
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="the faithfulness-coverage frontier with the conformal guarantee" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* the certified region: faithfulness >= 1 - alpha */}
        <rect x={padL} y={padT} width={W - padL - padR} height={y(guar) - padT} fill={POS_COLOR} fillOpacity={0.08} />
        <line x1={padL} y1={y(guar)} x2={W - padR} y2={y(guar)} stroke={GUAR_COLOR} strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={W - padR} y={y(guar) - 4} textAnchor="end" fontSize={8.5} fill={GUAR_COLOR} fontFamily="var(--font-sans)">certified faithful ≥ 1 − α = {fmt(guar, 2)}</text>
        {/* axes */}
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <text x={x(t)} y={H - padB + 13} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
            <text x={padL - 6} y={y(t) + 3} textAnchor="end" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
          </g>
        ))}
        {/* the back-off frontier */}
        <path d={path} fill="none" stroke={ACCENT} strokeWidth={2} />
        {/* the terse / verbose divergence extremes */}
        <circle cx={x(DIVERGENCE.terse.r)} cy={y(DIVERGENCE.terse.p)} r={4} fill={MUTED} />
        <text x={x(DIVERGENCE.terse.r) + 6} y={y(DIVERGENCE.terse.p) + 3} fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">terse</text>
        <circle cx={x(DIVERGENCE.verbose.r)} cy={y(DIVERGENCE.verbose.p)} r={4} fill={MUTED} />
        <text x={x(DIVERGENCE.verbose.r) - 6} y={y(DIVERGENCE.verbose.p) + 12} textAnchor="end" fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">verbose</text>
        {/* the CRC operating point */}
        <circle cx={x(0.704)} cy={y(0.928)} r={5} fill={GUAR_COLOR} />
        <text x={x(0.704)} y={y(0.928) - 8} textAnchor="middle" fontSize={8} fill={GUAR_COLOR} fontFamily="var(--font-sans)">CRC λ̂</text>
        {/* the current operating point */}
        <circle cx={x(row.r)} cy={y(row.p)} r={5.5} fill={ACCENT} stroke="var(--color-bg)" strokeWidth={1} />
        <text x={(padL + W) / 2} y={H - 4} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">coverage = recall →</text>
        <text x={12} y={(padT + H - padB) / 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + H - padB) / 2})`}>faithfulness = precision →</text>
      </svg>
      {/* verbosity inset: F1 versus answer length peaks at an interior length */}
      {(() => {
        const w = 560, h = 64, pl = 30, pr = 12, pt = 8, pb = 18;
        const nMax = VERBOSITY[VERBOSITY.length - 1].n;
        const vx = (n: number) => pl + (n / nMax) * (w - pl - pr);
        const vy = (f: number) => h - pb - f * (h - pt - pb);
        const best = VERBOSITY.reduce((a, b) => (b.f1 > a.f1 ? b : a));
        const line = VERBOSITY.map((d, i) => `${i ? 'L' : 'M'}${vx(d.n)},${vy(d.f1)}`).join(' ');
        return (
          <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="F1 versus answer length, peaking at an interior length" style={{ width: '100%', maxWidth: w, height: 'auto', display: 'block' }}>
            <path d={line} fill="none" stroke={GUAR_COLOR} strokeWidth={1.6} />
            {VERBOSITY.map((d) => (
              <circle key={d.n} cx={vx(d.n)} cy={vy(d.f1)} r={d === best ? 4 : 2} fill={d === best ? ACCENT : GUAR_COLOR} />
            ))}
            <text x={vx(best.n)} y={vy(best.f1) - 6} textAnchor="middle" fontSize={8} fill={ACCENT} fontFamily="var(--font-sans)">F1-optimal at {best.n} claims</text>
            <text x={pl} y={h - 5} fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">F1 vs answer length (claims) — interior optimum, neither terse nor exhaustive</text>
          </svg>
        );
      })()}
      <Slider label="confidence cut τ" value={ti} min={0} max={FRONTIER.length - 1} step={1}
        onChange={(v) => setTi(Math.round(v))} display={fmt(row.tau, 2)} />
      <Row>
        <Readout label="faithfulness" value={fmt(row.p, 2)} accent />
        <Readout label="coverage" value={fmt(row.r, 2)} />
        <Readout label="retention" value={pct(row.ret)} />
        <Readout label="CRC false-claim rate" value={`${fmt(CRC.realized, 3)} ≤ ${CRC.alpha}`} accent />
      </Row>
    </div>
  );
}

// ===== Panel C — judge calibration =================================================================
function CalibrationPanel({ cond, setCond }: { cond: 'raw' | 'platt' | 'iso'; setCond: (c: 'raw' | 'platt' | 'iso') => void }) {
  const bins = cond === 'raw' ? REL_RAW : cond === 'platt' ? REL_PLATT : REL_ISO;
  const ece = eceOf(bins);
  const W = 320, H = 250, pad = 34;
  const sx = (v: number) => pad + v * (W - 2 * pad);
  const sy = (v: number) => H - pad - v * (H - 2 * pad);
  const nMax = Math.max(...bins.map((b) => b[2])) || 1;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.4rem' }}>
        {(['raw', 'platt', 'iso'] as const).map((c) => (
          <button key={c} type="button" style={pill(cond === c)} onClick={() => setCond(c)}>
            {c === 'raw' ? 'raw judge' : c === 'platt' ? 'Platt' : 'isotonic'}
          </button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="reliability diagram of the judge confidence" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
          <line x1={pad} y1={H - pad} x2={W - pad} y2={pad} stroke={MUTED} strokeWidth={1} strokeDasharray="3 3" />
          <text x={W - pad} y={pad - 4} textAnchor="end" fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">perfect calibration</text>
          <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--color-border)" />
          <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--color-border)" />
          {bins.map((b, i) => {
            const [conf, acc, n] = b;
            const over = conf > acc;   // over-confident bins
            return (
              <g key={i}>
                <line x1={sx(conf)} y1={sy(conf)} x2={sx(conf)} y2={sy(acc)} stroke={over ? NEG_COLOR : POS_COLOR} strokeWidth={1.2} />
                <circle cx={sx(conf)} cy={sy(acc)} r={2 + 4 * (n / nMax)} fill={ACCENT} fillOpacity={0.8} />
              </g>
            );
          })}
          <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">judge confidence →</text>
          <text x={12} y={H / 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${H / 2})`}>empirical accuracy →</text>
        </svg>
        <div style={{ minWidth: '12rem' }}>
          <Row>
            <Readout label={`ECE (${cond})`} value={fmt(ece, 3)} accent />
            <Readout label="AUC (rank)" value={fmt(cond === 'raw' ? AUC.raw : AUC.platt, 3)} />
          </Row>
          <p style={{ fontSize: '0.74rem', color: 'var(--color-text-secondary)', lineHeight: 1.5, marginTop: '0.4rem' }}>
            The raw judge is over-confident (ECE {fmt(ECE.raw, 3)}); Platt lowers it to {fmt(ECE.platt, 3)} and
            isotonic to {fmt(ECE.iso, 3)} — while Platt preserves the ranking exactly (AUC {fmt(AUC.raw, 3)}).
          </p>
          {/* Rogan–Gladen debiasing bar */}
          <div style={{ marginTop: '0.6rem' }}>
            <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.25rem' }}>Rogan–Gladen debiasing</div>
            <svg viewBox="0 0 240 64" role="img" aria-label="naive versus corrected faithfulness against the oracle" style={{ width: '100%', maxWidth: 240, height: 'auto' }}>
              {([['naive', RG.naive, NEG_COLOR, 8], ['corrected', RG.corrected, POS_COLOR, 30]] as const).map(([lab, v, col, yy]) => (
                <g key={lab}>
                  <rect x={40} y={yy} width={v * 180} height={12} fill={col} fillOpacity={0.8} />
                  <text x={36} y={yy + 10} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{lab}</text>
                  <text x={40 + v * 180 + 3} y={yy + 10} fontSize={8} fill="var(--color-text)" fontFamily="var(--font-sans)">{fmt(v, 3)}</text>
                </g>
              ))}
              <line x1={40 + RG.oracle * 180} y1={2} x2={40 + RG.oracle * 180} y2={50} stroke={ACCENT} strokeWidth={1.5} strokeDasharray="3 2" />
              <text x={40 + RG.oracle * 180} y={60} textAnchor="middle" fontSize={7.5} fill={ACCENT} fontFamily="var(--font-sans)">oracle {fmt(RG.oracle, 2)}</text>
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

// ===== Panel D — bits of grounding =================================================================
function BitsPanel() {
  const W = 560, H = 210, padL = 38, padR = 14, padT = 12, padB = 30;
  const lo = -2, hi = 1.6, nb = 36;
  const sup = histo(BITS_SUP, nb, lo, hi);
  const hal = histo(BITS_HAL, nb, lo, hi);
  const cMax = Math.max(...sup, ...hal) || 1;
  const bw = (W - padL - padR) / nb;
  const sx = (v: number) => padL + ((v - lo) / (hi - lo)) * (W - padL - padR);
  const mid = (H - padT - padB) / 2 + padT;
  const upH = (n: number) => (n / cMax) * (mid - padT);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="histogram of per-claim pointwise mutual information with the context, supported above and hallucinated below" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={mid} x2={W - padR} y2={mid} stroke="var(--color-border)" />
        {sup.map((n, i) => (
          <rect key={`s${i}`} x={padL + i * bw + 0.5} y={mid - upH(n)} width={bw - 1} height={upH(n)} fill={POS_COLOR} fillOpacity={0.8} />
        ))}
        {hal.map((n, i) => (
          <rect key={`h${i}`} x={padL + i * bw + 0.5} y={mid} width={bw - 1} height={upH(n)} fill={NEG_COLOR} fillOpacity={0.8} />
        ))}
        {/* zero-bits line */}
        <line x1={sx(0)} y1={padT} x2={sx(0)} y2={H - padB} stroke={ACCENT} strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={sx(0) + 3} y={padT + 8} fontSize={8} fill={ACCENT} fontFamily="var(--font-sans)">0 bits</text>
        <text x={padL + 4} y={padT + 8} fontSize={8.5} fill={POS_COLOR} fontFamily="var(--font-sans)">supported claims (positive bits)</text>
        <text x={padL + 4} y={H - padB - 4} fontSize={8.5} fill={NEG_COLOR} fontFamily="var(--font-sans)">hallucinated claims (non-positive bits)</text>
        {[-2, -1, 0, 1].map((t) => (
          <text key={t} x={sx(t)} y={H - padB + 12} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
        ))}
        <text x={(padL + W) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">pmi(claim ; context | query)  [bits]</text>
      </svg>
      <Row>
        <Readout label="mean bits (supported)" value={`+${fmt(BITS_SUMMARY.mean_sup, 3)}`} accent />
        <Readout label="mean bits (hallucinated)" value={fmt(BITS_SUMMARY.mean_hal, 3)} />
        <Readout label="hallucinations ≤ 0 bits" value={pct(BITS_SUMMARY.frac_hal_nonpos)} accent />
      </Row>
    </div>
  );
}

// ===== root ========================================================================================
type Panel = 'claims' | 'frontier' | 'calibration' | 'bits';
const TEX: Record<Panel, string> = {
  claims: '\\text{faithfulness} = \\dfrac{|\\hat C \\cap S|}{|\\hat C|}, \\qquad \\text{coverage} = \\dfrac{|\\{\\,f : \\exists\\, c\\in\\hat C\\cap S\\text{ grounding }f\\,\\}|}{|F|}',
  frontier: '\\hat\\lambda = \\inf\\Big\\{\\lambda : \\tfrac{n}{n+1}\\hat R_n(\\lambda) + \\tfrac{B}{n+1} \\le \\alpha\\Big\\}, \\quad \\mathbb{E}\\big[\\,1-\\text{faithfulness}(\\hat\\lambda)\\,\\big] \\le \\alpha',
  calibration: '\\mathrm{ECE} = \\sum_b \\tfrac{n_b}{N}\\,\\big|\\mathrm{acc}_b - \\mathrm{conf}_b\\big|, \\qquad \\hat\\pi = \\dfrac{p_{\\mathrm{obs}} + \\mathrm{sp} - 1}{\\mathrm{se} + \\mathrm{sp} - 1}',
  bits: '\\mathrm{pmi}(c\\,;\\,\\text{ctx}\\mid q) = \\log_2 \\dfrac{p(c \\mid \\text{ctx}, q)}{p(c \\mid q)} > 0 \\iff \\text{supported}',
};

export default memo(function FaithfulnessGroundednessLaboratory() {
  const [panel, setPanel] = useState<Panel>('claims');
  const [tau, setTau] = useState(0.5);                 // Panel A confidence cut
  const [ti, setTi] = useState(5);                     // Panel B frontier index (τ ≈ 0.20)
  const [cond, setCond] = useState<'raw' | 'platt' | 'iso'>('raw');
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="faithfulness-groundedness" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'claims')} onClick={() => setPanel('claims')}>A · atomic claims</button>
        <button type="button" style={pill(panel === 'frontier')} onClick={() => setPanel('frontier')}>B · faithfulness–coverage</button>
        <button type="button" style={pill(panel === 'calibration')} onClick={() => setPanel('calibration')}>C · judge calibration</button>
        <button type="button" style={pill(panel === 'bits')} onClick={() => setPanel('bits')}>D · bits-of-grounding</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'claims' && <ClaimsPanel tau={tau} setTau={setTau} />}
      {panel === 'frontier' && <FrontierPanel ti={ti} setTi={setTi} />}
      {panel === 'calibration' && <CalibrationPanel cond={cond} setCond={setCond} />}
      {panel === 'bits' && <BitsPanel />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {FG_N_DOCS} synthetic finance filings (one vMF company prototype each), context = top-{FG_CTX_K}
        retrieved; a claim is supported iff its cosine to a context fact clears {COS_SUPPORT}. The lenient judge
        (AUC {fmt(AUC.raw, 2)}) over-endorses, so the naive faithfulness {fmt(RG.naive, 2)} is biased above the
        oracle {fmt(RG.oracle, 2)}. Numbers mirror <code>faithfulness_groundedness.py</code>; the lab recomputes
        Panel A's precision/coverage at the live τ and Panel C's ECE from the baked reliability bins, and bakes
        the corpus-derived confidences, frontier, calibration, and per-claim bits.
      </p>
    </div>
  );
});
