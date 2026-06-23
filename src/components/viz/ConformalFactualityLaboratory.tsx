import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Conformal Factuality Laboratory — four panels for the `conformal-factuality` topic. The calibrated
 * per-claim judge confidence (from llm-as-judge-ragas) is the nonconformity score; conformal prediction
 * turns it into a distribution-free guarantee.
 *   A. The calibration quantile. The sorted faithful-claim scores s = 1 − c̃ with the split-conformal
 *      threshold q̂ = the ⌈(1−α)(n+1)⌉-th smallest, recomputed live as the α slider moves, and the
 *      realized vs target (1−α) coverage from the Monte-Carlo resplit (validity demonstrated).
 *   B. Per-claim back-off. One worked answer's ten claims sorted by confidence, the cut τ = 1 − q̂, and
 *      the retained/removed split — colored by truth, exposing the high-confidence hallucination the
 *      lenient judge endorsed and that a recall guarantee cannot see.
 *   C. The risk–coverage frontier. split-conformal recall lets the false-claim rate run UNCONTROLLED (red,
 *      above the y = α line); Conformal Risk Control holds it ≤ α (green, on/under the line), at the cost
 *      of retention.
 *   D. Covariate shift. Coverage vs shift strength β: the unweighted (split) threshold collapses below
 *      1 − α as exchangeability breaks; weighted conformal (known likelihood ratio) restores it.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every baked constant below is mirrored TO THE DECIMAL from
 * notebooks/conformal-factuality/conformal_factuality.py (viz_constants()). Matching asserts:
 * test_perfect_judge_collapse / test_split_conformal_quantile_rank / test_realized_coverage_near_target /
 * test_crc_monotone_and_bound / test_false_claim_loss_monotone / test_fraction_loss_not_monotone /
 * test_weighted_collapses_to_split_under_no_shift / test_covariate_shift_breaks_then_weighted_restores.
 * The lab recomputes ONLY closed forms in TS (Panel A's quantile index/q̂ from the baked sorted scores as
 * α moves; Panel B's sort/filter/retained-faithful mean at the shared cut). Everything corpus-derived —
 * the calibration scores, the MC coverage, the risk–coverage frontier, the covariate-shift coverage — is
 * baked (TS cannot reproduce Platt fits, MC resplits, or the weighted quantile). Change a number here ->
 * change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_QUERIES = 40;
const N_DOCS = 120;
const K = 10;
const WORKED_LEG = 'dense';
const ALPHA_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4];
const CALIB_QUALITY = { eceRaw: 0.0928, ecePlatt: 0.0622, aucRaw: 0.8981, aucPlatt: 0.8981 };

// Panel A — the sorted faithful-claim calibration scores s = 1 − c̃ (n = 137), ascending.
const CALIB_SCORES = [
  0.0282, 0.0282, 0.0283, 0.0286, 0.029, 0.0299, 0.0304, 0.0306, 0.0306, 0.0315, 0.0317, 0.0325, 0.0326,
  0.0336, 0.0342, 0.0362, 0.0365, 0.0366, 0.0367, 0.0381, 0.0384, 0.0386, 0.0386, 0.0392, 0.0396, 0.0412,
  0.0414, 0.0416, 0.0422, 0.0432, 0.0437, 0.0458, 0.0461, 0.0476, 0.0476, 0.0491, 0.0499, 0.0506, 0.0507,
  0.051, 0.052, 0.0522, 0.0522, 0.0525, 0.053, 0.0544, 0.0546, 0.055, 0.0557, 0.0577, 0.0586, 0.0598,
  0.0598, 0.0602, 0.061, 0.062, 0.0622, 0.0642, 0.0643, 0.0644, 0.0664, 0.0682, 0.0709, 0.0711, 0.0749,
  0.0789, 0.0802, 0.0804, 0.0804, 0.0807, 0.0857, 0.0868, 0.0877, 0.0961, 0.0976, 0.098, 0.0991, 0.1015,
  0.1024, 0.1025, 0.1042, 0.1053, 0.1058, 0.1063, 0.1073, 0.1078, 0.1116, 0.1127, 0.1188, 0.1194, 0.1311,
  0.1554, 0.1618, 0.1912, 0.1949, 0.1949, 0.2064, 0.2077, 0.2104, 0.2117, 0.2145, 0.2161, 0.2322, 0.258,
  0.266, 0.266, 0.2715, 0.2722, 0.2745, 0.2745, 0.2904, 0.3025, 0.3048, 0.3048, 0.3048, 0.3197, 0.3328,
  0.3447, 0.3778, 0.3861, 0.3993, 0.4125, 0.4212, 0.4349, 0.4377, 0.4818, 0.5192, 0.5231, 0.5298, 0.567,
  0.618, 0.618, 0.6265, 0.6931, 0.7163, 0.8766, 0.8889,
];
// realized Monte-Carlo coverage (mean, std) at each grid α — keyed by α string.
const MC_COVERAGE: Record<string, { mean: number; std: number }> = {
  '0.02': { mean: 0.9829, std: 0.0185 }, '0.05': { mean: 0.9494, std: 0.0387 },
  '0.1': { mean: 0.9021, std: 0.0514 }, '0.15': { mean: 0.8551, std: 0.0588 },
  '0.2': { mean: 0.8047, std: 0.0616 }, '0.3': { mean: 0.7014, std: 0.0693 },
  '0.4': { mean: 0.6007, std: 0.0734 },
};

// Panel B — one worked test answer (query 20): per-claim calibrated confidence + truth, and precision@k.
const WORKED = {
  query: 20,
  precisionAtK: 0.5,
  conf: [0.9705, 0.0362, 0.9367, 0.7965, 0.0456, 0.8051, 0.9336, 0.3537, 0.8912, 0.104],
  y: [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
};

// Panel C — the risk–coverage frontier across the α grid.
type FrontierRow = {
  alpha: number; splitRecall: number; splitFalse: number; splitRetention: number;
  crcFalse: number; crcRetention: number;
};
const FRONTIER: FrontierRow[] = [
  { alpha: 0.02, splitRecall: 1.0, splitFalse: 0.23, splitRetention: 0.84, crcFalse: 0.0, crcRetention: 0.0 },
  { alpha: 0.05, splitRecall: 0.959, splitFalse: 0.16, splitRetention: 0.745, crcFalse: 0.01, crcRetention: 0.3 },
  { alpha: 0.1, splitRecall: 0.943, splitFalse: 0.115, splitRetention: 0.69, crcFalse: 0.085, crcRetention: 0.595 },
  { alpha: 0.15, splitRecall: 0.91, splitFalse: 0.105, splitRetention: 0.66, crcFalse: 0.14, crcRetention: 0.715 },
  { alpha: 0.2, splitRecall: 0.893, splitFalse: 0.1, splitRetention: 0.645, crcFalse: 0.205, crcRetention: 0.81 },
  { alpha: 0.3, splitRecall: 0.836, splitFalse: 0.09, splitRetention: 0.6, crcFalse: 0.3, crcRetention: 0.91 },
  { alpha: 0.4, splitRecall: 0.647, splitFalse: 0.04, splitRetention: 0.435, crcFalse: 0.39, crcRetention: 1.0 },
];

// Panel D — coverage vs covariate-shift strength β, split vs weighted-conformal repair (target 1 − 0.10).
type ShiftRow = { beta: number; target: number; split: number; weighted: number };
const SHIFT: ShiftRow[] = [
  { beta: 0.0, target: 0.9, split: 0.948, weighted: 0.948 },
  { beta: 0.5, target: 0.9, split: 0.851, weighted: 0.88 },
  { beta: 1.0, target: 0.9, split: 0.764, weighted: 0.934 },
  { beta: 1.5, target: 0.9, split: 0.646, weighted: 1.0 },
  { beta: 2.0, target: 0.9, split: 0.48, weighted: 1.0 },
  { beta: 3.0, target: 0.9, split: 0.266, weighted: 1.0 },
];

// --- theme tokens ---------------------------------------------------------------------
const ACCENT = 'var(--color-accent)';
const MUTED = '#9aa3ad';
const POS_COLOR = '#6fa389';   // faithful / retained-good / under control
const NEG_COLOR = '#c0726a';   // unfaithful / leaked hallucination / uncontrolled
const fmt = (x: number, n = 3) => (Number.isFinite(x) ? x.toFixed(n) : '∞');
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

// --- closed-form TS recomputation -----------------------------------------------------
// Split-conformal threshold q̂ = the ⌈(1−α)(n+1)⌉-th smallest calibration score (Theorem 1), from the
// baked ASCENDING scores. rank > n -> +∞ (cannot certify; retain everything).
const qHatOf = (alpha: number) => {
  const n = CALIB_SCORES.length;
  const rank = Math.max(1, Math.ceil((1 - alpha) * (n + 1)));   // alpha=1 -> rank 0 -> CALIB_SCORES[-1]; cap to min
  return rank > n ? Infinity : CALIB_SCORES[rank - 1];
};
// fixed-range histogram counts of the calibration scores.
const histo = (scores: number[], nb: number, lo: number, hi: number) => {
  const c = new Array(nb).fill(0);
  scores.forEach((v) => { c[Math.min(nb - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * nb)))]++; });
  return c;
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
      <span style={{ minWidth: '12rem' }}>{label} = <strong>{display}</strong></span>
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

// ===== Panel A — the calibration quantile =========================================================
function QuantilePanel({ ai, setAi }: { ai: number; setAi: (v: number) => void }) {
  const alpha = ALPHA_GRID[ai];
  const qHat = qHatOf(alpha);
  const tau = 1 - qHat;                       // the confidence cut (retain c̃ ≥ τ)
  const cov = MC_COVERAGE[String(alpha)];
  const target = 1 - alpha;
  const W = 560, H = 170, padL = 34, padR = 12, padT = 12, padB = 26;
  const lo = 0, hi = 0.9, nb = 30;
  const counts = histo(CALIB_SCORES, nb, lo, hi);
  const cMax = Math.max(...counts) || 1;   // guard: never divide by an empty-histogram max
  const bw = (W - padL - padR) / nb;
  const sx = (v: number) => padL + ((v - lo) / (hi - lo)) * (W - padL - padR);
  const by = (n: number) => H - padB - (n / cMax) * (H - padT - padB);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="calibration score histogram with the split-conformal quantile threshold" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {counts.map((n, i) => {
          const x = padL + i * bw;
          const binHi = lo + ((i + 1) / nb) * (hi - lo);
          const above = binHi > qHat;            // the upper-tail α mass (dropped faithful claims)
          return <rect key={i} x={x + 0.5} y={by(n)} width={bw - 1} height={H - padB - by(n)}
            fill={above ? NEG_COLOR : MUTED} fillOpacity={above ? 0.7 : 0.5} />;
        })}
        {Number.isFinite(qHat) && (
          <g>
            <line x1={sx(qHat)} y1={padT} x2={sx(qHat)} y2={H - padB} stroke={ACCENT} strokeWidth={1.6} strokeDasharray="4 3" />
            <text x={sx(qHat)} y={padT + 8} textAnchor="middle" fontSize={9} fill={ACCENT} fontFamily="var(--font-sans)">q̂ = {fmt(qHat)}</text>
          </g>
        )}
        <text x={padL} y={H - 6} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">nonconformity score s = 1 − c̃ over the {CALIB_SCORES.length} faithful calibration claims</text>
      </svg>
      <Slider label="miscoverage level α" value={ai} min={0} max={ALPHA_GRID.length - 1} step={1}
        onChange={(v) => setAi(Math.round(v))} display={fmt(alpha, 2)} />
      {/* coverage bar: realized (MC) vs target 1 − α */}
      <svg viewBox="0 0 560 54" role="img" aria-label="realized versus target coverage" style={{ width: '100%', maxWidth: 560, height: 'auto', display: 'block' }}>
        {(() => {
          const x0 = 34, x1 = 548, w = x1 - x0;
          const bx = (v: number) => x0 + v * w;
          return (
            <g>
              <line x1={x0} y1={40} x2={x1} y2={40} stroke="var(--color-border)" />
              {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                <g key={t}><line x1={bx(t)} y1={37} x2={bx(t)} y2={43} stroke="var(--color-border)" />
                  <text x={bx(t)} y={52} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text></g>
              ))}
              <rect x={x0} y={14} width={cov.mean * w} height={12} fill={POS_COLOR} fillOpacity={0.8} />
              <line x1={bx(cov.mean - cov.std)} y1={20} x2={bx(cov.mean + cov.std)} y2={20} stroke="var(--color-text)" strokeWidth={1} />
              <line x1={bx(target)} y1={8} x2={bx(target)} y2={32} stroke={ACCENT} strokeWidth={1.8} strokeDasharray="3 2" />
              <text x={bx(target)} y={6} textAnchor="middle" fontSize={8} fill={ACCENT} fontFamily="var(--font-sans)">target 1−α = {fmt(target, 2)}</text>
            </g>
          );
        })()}
      </svg>
      <Row>
        <Readout label="threshold q̂" value={fmt(qHat)} accent />
        <Readout label="confidence cut τ = 1 − q̂" value={fmt(tau)} />
        <Readout label="realized coverage" value={pct(cov.mean)} accent />
        <Readout label="target 1 − α" value={pct(target)} />
      </Row>
    </div>
  );
}

// ===== Panel B — per-claim back-off ===============================================================
function BackoffPanel({ ai }: { ai: number }) {
  const alpha = ALPHA_GRID[ai];
  const tau = 1 - qHatOf(alpha);
  // sort claims by descending confidence (closed form over the baked array)
  const claims = WORKED.conf.map((conf, i) => ({ conf, y: WORKED.y[i] }))
    .sort((a, b) => b.conf - a.conf);
  const retained = claims.filter((cl) => cl.conf >= tau);
  const retainedFaithful = retained.length ? retained.filter((cl) => cl.y === 1).length / retained.length : 0;
  const leaked = retained.filter((cl) => cl.y === 0).length;
  const W = 560, H = 220, padL = 120, padR = 16, padT = 10, rowH = (H - padT - 16) / K;
  const sx = (v: number) => padL + v * (W - padL - padR);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="per-claim back-off for the worked answer" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {claims.map((cl, i) => {
          const y = padT + i * rowH;
          const kept = cl.conf >= tau;
          const color = cl.y === 1 ? POS_COLOR : NEG_COLOR;
          return (
            <g key={i}>
              <text x={6} y={y + rowH / 2 + 3} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
                claim {i + 1} · {cl.y === 1 ? 'supported' : 'UNSUPPORTED'}
              </text>
              <rect x={padL} y={y + 2} width={sx(cl.conf) - padL} height={rowH - 4} fill={color} fillOpacity={kept ? 0.85 : 0.22} />
              <rect x={sx(cl.conf) - 1} y={y + 2} width={2} height={rowH - 4} fill={color} />
            </g>
          );
        })}
        <line x1={sx(tau)} y1={padT} x2={sx(tau)} y2={H - 16} stroke={ACCENT} strokeWidth={1.8} strokeDasharray="4 3" />
        <text x={sx(tau)} y={H - 4} textAnchor="middle" fontSize={9} fill={ACCENT} fontFamily="var(--font-sans)">cut τ = {fmt(tau)} (retain confidence ≥ τ →)</text>
      </svg>
      <Row>
        <Readout label="claims retained" value={`${retained.length} / ${K}`} accent />
        <Readout label="retained-faithful" value={pct(retainedFaithful)} />
        <Readout label="hallucinations retained" value={`${leaked}`} accent={leaked > 0} />
        <Readout label="precision@k (perfect judge)" value={fmt(WORKED.precisionAtK, 2)} />
      </Row>
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.45 }}>
        The lenient judge endorsed an unsupported claim at high confidence — it survives the cut at low α and
        is backed off only once τ rises past it (dropping genuine claims too). Controlling that leakage
        directly is what Panel C's risk control does. The collapse anchor: a <em>perfect</em> judge's
        retained fraction equals precision@k exactly.
      </p>
    </div>
  );
}

// ===== Panel C — the risk–coverage frontier =======================================================
function FrontierPanel({ ai }: { ai: number }) {
  const alpha = ALPHA_GRID[ai];
  const row = FRONTIER[ai];
  const W = 540, H = 300, padL = 44, padR = 14, padT = 14, padB = 40;
  const ax = (a: number) => padL + (a / 0.4) * (W - padL - padR);
  const ay = (v: number) => H - padB - (v / 0.4) * (H - padT - padB);   // false-claim rate axis, 0..0.4
  const splitPath = FRONTIER.map((r, i) => `${i ? 'L' : 'M'}${ax(r.alpha)},${ay(r.splitFalse)}`).join(' ');
  const crcPath = FRONTIER.map((r, i) => `${i ? 'L' : 'M'}${ax(r.alpha)},${ay(r.crcFalse)}`).join(' ');
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="risk-coverage frontier: split versus conformal risk control" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        {/* the guarantee line y = α (the diagonal); CRC stays on/under it, split runs above */}
        <line x1={ax(0)} y1={ay(0)} x2={ax(0.4)} y2={ay(0.4)} stroke={ACCENT} strokeWidth={1.2} strokeDasharray="4 3" />
        <text x={ax(0.4)} y={ay(0.4) - 4} textAnchor="end" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">guarantee y = α</text>
        <path d={splitPath} fill="none" stroke={NEG_COLOR} strokeWidth={2} />
        <path d={crcPath} fill="none" stroke={POS_COLOR} strokeWidth={2} />
        {FRONTIER.map((r, i) => (
          <g key={i}>
            <circle cx={ax(r.alpha)} cy={ay(r.splitFalse)} r={i === ai ? 4.5 : 2.6} fill={NEG_COLOR} />
            <circle cx={ax(r.alpha)} cy={ay(r.crcFalse)} r={i === ai ? 4.5 : 2.6} fill={POS_COLOR} />
          </g>
        ))}
        {[0, 0.1, 0.2, 0.3, 0.4].map((t) => (
          <g key={t}>
            <text x={padL - 6} y={ay(t) + 3} textAnchor="end" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
            <text x={ax(t)} y={H - padB + 14} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
          </g>
        ))}
        <text x={(padL + W) / 2} y={H - 6} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">target level α</text>
        <text x={ax(0.30)} y={ay(0.31)} fontSize={8.5} fill={NEG_COLOR} fontFamily="var(--font-sans)">split recall — false rate UNCONTROLLED</text>
        <text x={ax(0.06)} y={ay(0.12)} fontSize={8.5} fill={POS_COLOR} fontFamily="var(--font-sans)">CRC — false rate ≤ α</text>
      </svg>
      <Row>
        <Readout label="α" value={fmt(alpha, 2)} accent />
        <Readout label="split false-claim rate" value={pct(row.splitFalse)} />
        <Readout label="CRC false-claim rate" value={pct(row.crcFalse)} accent />
        <Readout label="split / CRC retention" value={`${pct(row.splitRetention)} / ${pct(row.crcRetention)}`} />
      </Row>
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.45 }}>
        The split-conformal recall guarantee says nothing about hallucination leakage — its false-claim rate
        (red) sits above the y = α line. Conformal risk control (green) holds the false-claim rate at or below
        α, paying in retention. The small overshoot near α = 0.2 is honest: CRC controls the <em>expected</em>
        risk over calibration draws, not a single realization.
      </p>
    </div>
  );
}

// ===== Panel D — covariate shift breaks coverage; weighted conformal repairs ======================
function ShiftPanel({ bi, setBi }: { bi: number; setBi: (v: number) => void }) {
  const row = SHIFT[bi];
  const W = 540, H = 280, padL = 44, padR = 16, padT = 14, padB = 38;
  const betaMax = SHIFT[SHIFT.length - 1].beta;
  const bx = (b: number) => padL + (b / betaMax) * (W - padL - padR);
  const yv = (v: number) => H - padB - v * (H - padT - padB);
  const splitPath = SHIFT.map((r, i) => `${i ? 'L' : 'M'}${bx(r.beta)},${yv(r.split)}`).join(' ');
  const weightedPath = SHIFT.map((r, i) => `${i ? 'L' : 'M'}${bx(r.beta)},${yv(r.weighted)}`).join(' ');
  const target = SHIFT[0].target;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="coverage versus covariate-shift strength" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={yv(target)} x2={W - padR} y2={yv(target)} stroke={ACCENT} strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={W - padR} y={yv(target) - 4} textAnchor="end" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">target 1 − α = {fmt(target, 2)}</text>
        <path d={splitPath} fill="none" stroke={MUTED} strokeWidth={2} />
        <path d={weightedPath} fill="none" stroke={POS_COLOR} strokeWidth={2} />
        {SHIFT.map((r, i) => (
          <g key={i}>
            <circle cx={bx(r.beta)} cy={yv(r.split)} r={i === bi ? 4.5 : 2.6} fill={MUTED} />
            <circle cx={bx(r.beta)} cy={yv(r.weighted)} r={i === bi ? 4.5 : 2.6} fill={POS_COLOR} />
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={padL - 6} y={yv(t) + 3} textAnchor="end" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
        ))}
        {SHIFT.map((r) => (
          <text key={r.beta} x={bx(r.beta)} y={H - padB + 14} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{fmt(r.beta, 1)}</text>
        ))}
        <text x={(padL + W) / 2} y={H - 6} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">covariate-shift strength β</text>
        <text x={bx(2.0)} y={yv(0.42)} fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">split (unweighted) — collapses</text>
        <text x={bx(0.6)} y={yv(0.97)} fontSize={8.5} fill={POS_COLOR} fontFamily="var(--font-sans)">weighted conformal — restored</text>
      </svg>
      <Slider label="covariate-shift strength β" value={bi} min={0} max={SHIFT.length - 1} step={1}
        onChange={(v) => setBi(Math.round(v))} display={fmt(row.beta, 1)} />
      <Row>
        <Readout label="β" value={fmt(row.beta, 1)} accent />
        <Readout label="split coverage" value={pct(row.split)} accent={row.split < target} />
        <Readout label="weighted coverage" value={pct(row.weighted)} />
        <Readout label="target 1 − α" value={pct(target)} />
      </Row>
    </div>
  );
}

type Panel = 'quantile' | 'backoff' | 'frontier' | 'shift';
const TEX: Record<Panel, string> = {
  quantile: '\\hat q = \\lceil (1-\\alpha)(n+1)\\rceil\\text{-th smallest of } \\{1-\\tilde c_i\\}, \\quad \\Pr\\!\\big(\\tilde c_{\\mathrm{new}} \\ge 1-\\hat q \\mid \\text{faithful}\\big) \\ge 1-\\alpha',
  backoff: '\\hat C(x) = \\{\\, i : \\tilde c_i \\ge \\tau \\,\\}, \\quad \\tau = 1 - \\hat q \\quad\\text{(remove the least-confident claims)}',
  frontier: '\\hat\\lambda = \\inf\\Big\\{\\lambda : \\tfrac{n}{n+1}\\hat R_n(\\lambda) + \\tfrac{B}{n+1} \\le \\alpha\\Big\\}, \\quad \\mathbb{E}\\big[L_{n+1}(\\hat\\lambda)\\big] \\le \\alpha',
  shift: 'w(x) = \\dfrac{dP_{\\mathrm{test}}}{dP_{\\mathrm{calib}}}(x), \\quad \\hat q_w = \\inf\\Big\\{ s : \\tfrac{\\sum_i w_i\\,\\mathbf{1}\\{S_i \\le s\\}}{\\sum_i w_i + w_{\\mathrm{new}}} \\ge 1-\\alpha \\Big\\}',
};

export default memo(function ConformalFactualityLaboratory() {
  const [panel, setPanel] = useState<Panel>('quantile');
  const [ai, setAi] = useState(2);       // α index — shared across panels A, B, C (default α = 0.10)
  const [bi, setBi] = useState(2);       // β index — Panel D
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="conformal-factuality" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'quantile')} onClick={() => setPanel('quantile')}>A · calibration quantile</button>
        <button type="button" style={pill(panel === 'backoff')} onClick={() => setPanel('backoff')}>B · per-claim back-off</button>
        <button type="button" style={pill(panel === 'frontier')} onClick={() => setPanel('frontier')}>C · risk–coverage</button>
        <button type="button" style={pill(panel === 'shift')} onClick={() => setPanel('shift')}>D · covariate shift</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'quantile' && <QuantilePanel ai={ai} setAi={setAi} />}
      {panel === 'backoff' && <BackoffPanel ai={ai} />}
      {panel === 'frontier' && <FrontierPanel ai={ai} />}
      {panel === 'shift' && <ShiftPanel bi={bi} setBi={setBi} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_DOCS} synthetic finance documents, {N_QUERIES} queries, {K} claims per answer; the lenient judge
        (AUC {fmt(CALIB_QUALITY.aucRaw, 2)}) and the MaxSim oracle are the prerequisites' shared corpus, leg{' '}
        <code>{WORKED_LEG}</code>. Numbers mirror <code>conformal_factuality.py</code>; the lab recomputes the
        quantile index/q̂ and the back-off filter in closed form and bakes the corpus-derived scores, MC
        coverage, frontier, and covariate-shift coverage.
      </p>
    </div>
  );
});
