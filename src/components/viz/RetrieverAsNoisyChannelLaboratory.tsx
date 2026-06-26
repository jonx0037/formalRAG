import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Retriever-as-a-Noisy-Channel Laboratory — four panels for the `retriever-as-noisy-channel` topic:
 *   A. The channel & its capacity. A recall slider erases the retrieved context with probability 1 - recall;
 *      the answer belief flattens toward uniform and the delivered bits fall EXACTLY as I = recall · I_0
 *      (the Binary Erasure Channel capacity).
 *   B. The recall floor (Fano). As recall falls the residual entropy H(A|Q,D) rises and Fano's inequality
 *      turns it into a floor on the generator's Bayes error — vacuous while H < 1 bit, then climbing. The
 *      information ceiling no generator beats. Recall sets the floor.
 *   C. The precision gap (confidently wrong). A substitution slider swaps the relevant filing for a
 *      same-sector distractor with probability eps; the realized error climbs to 1 while H(A|Q,D) stays
 *      LOW, so the Fano floor sees nothing — the realized-vs-Bayes gap is what calibration must close.
 *   D. The precision–entropy frontier. Reading k documents leaves recall@k pinned at 1 (gold is rank-1)
 *      while precision = 1/k falls and the blended belief's entropy rises — contamination measured in bits.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every probability / entropy / mean below is mirrored TO THE DECIMAL
 * from notebooks/retriever-as-noisy-channel/retriever_as_noisy_channel.py (viz_constants()). Matching
 * asserts: test_bec_capacity_linear / test_fano_is_a_theorem / test_recall_degradation_monotone /
 * test_fano_activates / test_confident_wrong_gap_widens / test_precision_entropy_frontier. The lab recomputes
 * ONLY closed forms in TS (Shannon/binary entropy, the loose & tight Fano floors, the BEC/BSC capacities, the
 * convex blends along each sweep, and SVG geometry). Change a number here -> change it in the .py and re-run.
 */

// --- baked from viz_constants() -------------------------------------------------------
const K = 8;                       // companies = answers = documents
const N_QUERIES = 32;
const SECTOR = [0, 0, 1, 1, 2, 2, 3, 3];
const WORKED_Q = 5;                // a representative query whose clean answer belief Panel A erodes
const WORKED_BELIEF = [0.2272, 0.459, 0.01, 0.0134, 0.0293, 0.0352, 0.173, 0.0529];

const I0 = 1.1832;                 // clean delivered bits I(A;D|Q)
const H_A_GIVEN_Q = 2.0592;        // marginal answer entropy
const H_COND0 = 0.876;             // clean residual entropy H(A|Q,D)
const BAYES0 = 0.1983;             // clean Bayes (MAP) error
const LOG2K = 3.0;                 // log2(8)
const UNIF_BAYES = 0.875;          // 1 - 1/K, the uniform-guess error after full erasure

// Panel C — substitution (precision). Gold read vs same-sector distractor read, averaged over queries.
const G_ERR = 0.0, D_ERR = 1.0;    // realized error of reading the gold filing vs the distractor
const G_BAYES = 0.1234, D_BAYES = 0.2378;
const G_H = 0.6025, D_H = 0.8728;

// Panel D — precision-vs-entropy frontier (recall@k saturated at 1 on this easy corpus).
const K_GRID = [1, 2, 3, 4, 5, 6, 7, 8];
const PRECISION = [1.0, 0.5, 0.3333, 0.25, 0.2, 0.1667, 0.1429, 0.125];
const FRONTIER_H = [0.6025, 1.0736, 1.4616, 1.6651, 1.8273, 1.9366, 2.0081, 2.0592];

const ACCENT = 'var(--color-accent)';
const POS = '#5fa873';             // bits delivered / capacity / correct
const NEG = '#c25b6b';             // error / bits lost / the confident-wrong gap
const FLOOR = '#6c8cd5';           // the Fano floor
const MUTED = '#9aa3ad';

const fmt = (x: number, n = 3) => x.toFixed(n);
const r2 = (x: number) => Math.round(x * 100) / 100;

// --- closed-form TS recomputation (mirrors the .py exactly) ---------------------------
const entropyBits = (p: number[]) => -p.reduce((s, x) => s + (x > 0 ? x * Math.log2(x) : 0), 0);
const binaryEntropy = (p: number) => (p <= 0 || p >= 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p));
const fanoLoose = (h: number) => Math.max(0, (h - 1) / LOG2K);
const fanoTight = (h: number) => {
  const log2Km1 = Math.log2(K - 1);
  const hc = Math.min(Math.max(h, 0), LOG2K);
  if (hc <= 0) return 0;
  const peMax = 1 - 1 / K;
  let lo = 0, hi = peMax;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (binaryEntropy(mid) + mid * log2Km1 < hc) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
};
// erasure (recall) channel: convex blends of the clean value and the erased (uniform) value.
const hCondR = (r: number) => r * H_COND0 + (1 - r) * LOG2K;
const bayesR = (r: number) => r * BAYES0 + (1 - r) * UNIF_BAYES;
const bitsR = (r: number) => r * I0;
const beliefR = (r: number) => WORKED_BELIEF.map((p) => r * p + (1 - r) / K);
// substitution (precision) channel.
const realizedE = (e: number) => (1 - e) * G_ERR + e * D_ERR;
const subBayesE = (e: number) => (1 - e) * G_BAYES + e * D_BAYES;
const subHE = (e: number) => (1 - e) * G_H + e * D_H;

// --- shared UI atoms ------------------------------------------------------------------
function Readout({ label, value, accent, color }: { label: string; value: string; accent?: boolean; color?: string }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: color ?? (accent ? 'var(--color-accent)' : 'var(--color-text)') }}>{value}</div>
    </div>
  );
}
function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display: string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '14rem' }}>{label} = <strong>{display}</strong></span>
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
// polyline from sampled (x,y) with pixel mappers.
const path = (xs: number[], f: (x: number) => number, px: (x: number) => number, py: (y: number) => number) =>
  xs.map((x, i) => `${i ? 'L' : 'M'}${r2(px(x))},${r2(py(f(x)))}`).join(' ');

const GRID = Array.from({ length: 81 }, (_v, i) => i / 80);   // fine sweep grid for smooth curves

// ===== Panel A — the channel & capacity =============================================================
function ChannelPanel({ recall, setRecall }: { recall: number; setRecall: (v: number) => void }) {
  const W = 520, H = 196, padL = 30, padR = 12, padT = 18, padB = 30;
  const belief = beliefR(recall);
  const gw = (W - padL - padR) / K;
  const py = (v: number) => H - padB - (H - padT - padB) * v;
  const hBelief = entropyBits(belief);
  const delivered = bitsR(recall);
  return (
    <div>
      <Slider label="retriever recall (1 − erasure rate)" value={recall} min={0} max={1} step={0.05}
        onChange={setRecall} display={`${fmt(recall, 2)}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="answer belief eroding toward uniform as recall falls" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (
          <text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>
        ))}
        {/* uniform reference line — what the belief erodes to */}
        <line x1={padL} y1={py(1 / K)} x2={W - padR} y2={py(1 / K)} stroke={MUTED} strokeWidth={1} strokeDasharray="3 3" />
        <text x={W - padR} y={py(1 / K) - 3} textAnchor="end" fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">uniform 1/K</text>
        {belief.map((v, a) => (
          <g key={a}>
            <rect x={r2(padL + a * gw + gw * 0.2)} y={r2(py(v))} width={r2(gw * 0.6)} height={r2((H - padB) - py(v))}
              fill={a === 1 ? POS : ACCENT} fillOpacity={a === 1 ? 0.85 : 0.55} />
            <text x={r2(padL + a * gw + gw / 2)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{a === 1 ? 'a*' : a}</text>
          </g>
        ))}
        <text x={padL} y={11} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          answer belief p(a | q) through the channel (worked query {WORKED_Q})
        </text>
      </svg>
      {/* delivered-bits meter: recall · I_0 out of I_0 */}
      <div style={{ margin: '0.3rem 0 0.1rem', fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
        bits delivered I = recall · I₀
      </div>
      <div style={{ position: 'relative', height: '0.7rem', borderRadius: '999px', background: 'var(--color-border)', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, width: `${(delivered / I0) * 100}%`, background: POS, opacity: 0.8 }} />
      </div>
      <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
        <Readout label="BEC capacity = recall" value={fmt(recall, 2)} color={POS} />
        <Readout label="bits delivered I = recall·I₀" value={`${fmt(delivered)} bits`} accent />
        <Readout label="answer-belief entropy" value={`${fmt(hBelief)} bits`} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        A retrieval miss <strong>erases</strong> the relevant filing (probability 1 − recall), and the generator falls back
        to a non-informative belief. Because the erased symbol is independent of the answer, the retriever is a{' '}
        <strong>Binary Erasure Channel</strong>: the surviving fraction of the relevant bits is exactly the recall, so the
        delivered information is <span style={{ color: POS }}>I = recall · I₀</span> and the belief flattens toward uniform.
      </p>
    </div>
  );
}

// ===== Panel B — the recall floor (Fano) ============================================================
const ACTIVATE_RECALL = 0.9416;   // recall where H(A|Q,D) crosses 1 bit (1 = 0.876r + 3(1−r))
function RecallFloorPanel({ recall, setRecall }: { recall: number; setRecall: (v: number) => void }) {
  const W = 520, H = 230, padL = 36, padR = 40, padT = 16, padB = 32;
  // x = recall, drawn 1.0 (left) -> 0.0 (right): degradation goes rightward.
  const px = (r: number) => padL + (W - padL - padR) * (1 - r);
  const pe = (y: number) => H - padB - (H - padT - padB) * y;            // error prob, 0..1 (left axis)
  const ph = (h: number) => H - padB - (H - padT - padB) * (h / LOG2K);  // residual entropy, 0..3 (right axis)
  return (
    <div>
      <Slider label="retriever recall (1 − erasure rate)" value={recall} min={0} max={1} step={0.05}
        onChange={setRecall} display={`${fmt(recall, 2)}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Fano floor and Bayes error rising as recall falls" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (<text key={v} x={padL - 5} y={pe(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
        {[1, 0.5, 0].map((r) => (<text key={r} x={px(r)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r}</text>))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">recall (degradation →)</text>
        {/* unreachable region below the loose Fano floor */}
        <path d={`${GRID.map((g, i) => `${i ? 'L' : 'M'}${r2(px(g))},${r2(pe(fanoLoose(hCondR(g))))}`).join(' ')} L${r2(px(1))},${r2(pe(0))} L${r2(px(0))},${r2(pe(0))} Z`} fill={FLOOR} fillOpacity={0.1} />
        {/* residual entropy (right axis, muted dashed) */}
        <path d={path(GRID, (r) => hCondR(r), px, (h) => ph(h))} fill="none" stroke={MUTED} strokeWidth={1.4} strokeDasharray="4 3" />
        {/* 1-bit crossover marker */}
        <line x1={px(ACTIVATE_RECALL)} y1={padT} x2={px(ACTIVATE_RECALL)} y2={H - padB} stroke="var(--color-text)" strokeWidth={1} strokeDasharray="2 3" />
        <text x={px(ACTIVATE_RECALL) - 3} y={padT + 8} textAnchor="end" fontSize={8} fill="var(--color-text)" fontFamily="var(--font-sans)">H = 1 bit</text>
        {/* Fano floors and Bayes error */}
        <path d={path(GRID, (r) => fanoLoose(hCondR(r)), px, pe)} fill="none" stroke={FLOOR} strokeWidth={1.6} />
        <path d={path(GRID, (r) => fanoTight(hCondR(r)), px, pe)} fill="none" stroke={FLOOR} strokeWidth={2.2} />
        <path d={path(GRID, (r) => bayesR(r), px, pe)} fill="none" stroke={NEG} strokeWidth={2.2} />
        {/* slider marker */}
        <circle cx={px(recall)} cy={pe(bayesR(recall))} r={3.5} fill={NEG} />
        <circle cx={px(recall)} cy={pe(fanoTight(hCondR(recall)))} r={3.5} fill={FLOOR} />
        <text x={padL + 4} y={padT + 8} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={NEG}>▬ Bayes error</tspan> · <tspan fill={FLOOR}>▬ Fano floor (tight/loose)</tspan> · <tspan fill={MUTED}>┄ H(A|Q,D)</tspan>
        </text>
        <text x={W - padR + 4} y={pe(1) + 3} fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">3</text>
        <text x={W - padR + 4} y={pe(0) + 3} fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">0</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="H(A|Q,D) residual" value={`${fmt(hCondR(recall))} bits`} />
        <Readout label="Fano floor (tight)" value={fmt(fanoTight(hCondR(recall)))} color={FLOOR} />
        <Readout label="Bayes error" value={fmt(bayesR(recall))} color={NEG} />
        <Readout label="bits delivered" value={`${fmt(bitsR(recall))} bits`} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        What recall fails to deliver reappears as residual entropy H(A|Q,D), and <strong>Fano's inequality</strong> turns it
        into a <span style={{ color: FLOOR }}>floor</span> on the generator's Bayes error: P<sub>e</sub> ≥ (H − 1)/log₂K. The
        floor is <em>vacuous</em> while H &lt; 1 bit (a well-retrieving channel), then climbs — no generator, however good,
        reaches into the shaded region. <strong>Recall sets the floor.</strong>
      </p>
    </div>
  );
}

// ===== Panel C — the precision gap (confidently wrong) ==============================================
function PrecisionGapPanel({ eps, setEps }: { eps: number; setEps: (v: number) => void }) {
  const W = 520, H = 230, padL = 36, padR = 16, padT = 16, padB = 32;
  const px = (e: number) => padL + (W - padL - padR) * e;
  const pe = (y: number) => H - padB - (H - padT - padB) * y;
  return (
    <div>
      <Slider label="substitution rate eps (1 − precision-ish)" value={eps} min={0} max={1} step={0.05}
        onChange={setEps} display={`${fmt(eps, 2)}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="realized error rising while the Fano floor stays blind under substitution" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (<text key={v} x={padL - 5} y={pe(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
        {[0, 0.5, 1].map((e) => (<text key={e} x={px(e)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{e}</text>))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">substitution rate eps (contamination →)</text>
        {/* the confident-wrong gap: between realized error and the Bayes floor */}
        <path d={`${path(GRID, realizedE, px, pe)} L${r2(px(1))},${r2(pe(subBayesE(1)))} ${GRID.slice().reverse().map((g) => `L${r2(px(g))},${r2(pe(subBayesE(g)))}`).join(' ')} Z`} fill={NEG} fillOpacity={0.12} />
        {/* residual entropy stays low (right-scaled to 0..1 by /LOG2K for context) */}
        <path d={path(GRID, (e) => subHE(e) / LOG2K, px, pe)} fill="none" stroke={MUTED} strokeWidth={1.4} strokeDasharray="4 3" />
        <path d={path(GRID, (e) => fanoLoose(subHE(e)), px, pe)} fill="none" stroke={FLOOR} strokeWidth={2.2} />
        <path d={path(GRID, subBayesE, px, pe)} fill="none" stroke={ACCENT} strokeWidth={1.8} />
        <path d={path(GRID, realizedE, px, pe)} fill="none" stroke={NEG} strokeWidth={2.4} />
        <circle cx={px(eps)} cy={pe(realizedE(eps))} r={3.5} fill={NEG} />
        <text x={padL + 4} y={padT + 8} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={NEG}>▬ realized error</tspan> · <tspan fill={ACCENT}>▬ Bayes error</tspan> · <tspan fill={FLOOR}>▬ Fano floor</tspan> · <tspan fill={MUTED}>┄ H(A|Q,D)/log₂K</tspan>
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="realized error (vs gold)" value={fmt(realizedE(eps))} color={NEG} />
        <Readout label="Bayes error" value={fmt(subBayesE(eps))} color={ACCENT} />
        <Readout label="Fano floor" value={fmt(fanoLoose(subHE(eps)))} color={FLOOR} />
        <Readout label="confident-wrong gap" value={fmt(realizedE(eps) - subBayesE(eps))} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        A false positive <strong>substitutes</strong> a plausible same-sector distractor for the relevant filing. The
        generator reads it and answers <span style={{ color: NEG }}>confidently wrong</span>: the realized error climbs to 1,
        yet H(A|Q,D) stays below 1 bit, so the <span style={{ color: FLOOR }}>Fano floor never moves</span> — an entropy bound
        cannot see confident contamination. The growing <span style={{ color: NEG }}>gap</span> is exactly what calibration and
        faithfulness must close. <strong>Precision governs whether you hit the floor.</strong>
      </p>
    </div>
  );
}

// ===== Panel D — the precision–entropy frontier =====================================================
function FrontierPanel() {
  const W = 520, H = 210, padL = 36, padR = 40, padT = 18, padB = 34;
  const n = K_GRID.length;
  const gw = (W - padL - padR) / n;
  const py1 = (v: number) => H - padB - (H - padT - padB) * v;             // precision / recall (0..1)
  const pyH = (h: number) => H - padB - (H - padT - padB) * (h / H_A_GIVEN_Q); // entropy (0..H(A|Q))
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="precision falling and blended-belief entropy rising as more documents are read" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (<text key={v} x={padL - 5} y={py1(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
        {/* recall@k pinned at 1 */}
        <line x1={padL} y1={py1(1)} x2={W - padR} y2={py1(1)} stroke={POS} strokeWidth={1.6} strokeDasharray="5 3" />
        <text x={W - padR} y={py1(1) - 3} textAnchor="end" fontSize={8} fill={POS} fontFamily="var(--font-sans)">recall@k = 1</text>
        {/* precision bars (1/k) */}
        {PRECISION.map((p, i) => (
          <g key={i}>
            <rect x={r2(padL + i * gw + gw * 0.25)} y={r2(py1(p))} width={r2(gw * 0.5)} height={r2((H - padB) - py1(p))} fill={ACCENT} fillOpacity={0.5} />
            <text x={r2(padL + i * gw + gw / 2)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{K_GRID[i]}</text>
          </g>
        ))}
        {/* blended-belief entropy line (right axis) */}
        <path d={FRONTIER_H.map((h, i) => `${i ? 'L' : 'M'}${r2(padL + i * gw + gw / 2)},${r2(pyH(h))}`).join(' ')} fill="none" stroke={NEG} strokeWidth={2.2} />
        {FRONTIER_H.map((h, i) => (<circle key={i} cx={r2(padL + i * gw + gw / 2)} cy={r2(pyH(h))} r={2.6} fill={NEG} />))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">documents read k</text>
        <text x={padL + 4} y={padT + 8} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={ACCENT}>▮ precision = 1/k</tspan> · <tspan fill={NEG}>▬ blended-belief entropy</tspan> · <tspan fill={POS}>┄ recall@k</tspan>
        </text>
        <text x={W - padR + 3} y={pyH(H_A_GIVEN_Q) + 3} fontSize={8} fill={NEG} fontFamily="var(--font-sans)">{fmt(H_A_GIVEN_Q, 1)}</text>
        <text x={W - padR + 3} y={pyH(0) + 3} fontSize={8} fill={NEG} fontFamily="var(--font-sans)">0</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="recall@k (every k)" value="1.000" color={POS} />
        <Readout label="precision at k=8" value={fmt(PRECISION[7])} accent />
        <Readout label="entropy added (k:1→8)" value={`${fmt(FRONTIER_H[0])} → ${fmt(FRONTIER_H[7])} bits`} color={NEG} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Reading more documents trades <strong>rate</strong> for <strong>distortion</strong>. On this corpus the gold filing is
        rank-1, so <span style={{ color: POS }}>recall@k stays 1</span> and precision = 1/k falls; the contamination shows up
        not as wrong answers but as <span style={{ color: NEG }}>bits of uncertainty added</span> — the blended belief drifts
        toward the full marginal entropy H(A|Q) = {fmt(H_A_GIVEN_Q)}. The retriever's operating point is a rate–distortion choice.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'channel' | 'recall' | 'precision' | 'frontier';
const TEX: Record<Panel, string> = {
  channel: 'I_{\\varepsilon}(A;D\\mid Q) = \\mathrm{recall}\\cdot I(A;D\\mid Q), \\qquad C_{\\mathrm{BEC}} = 1 - \\varepsilon = \\mathrm{recall}',
  recall: 'P_e \\;\\ge\\; \\frac{H(A\\mid Q,D) - 1}{\\log_2 K} \\qquad\\text{(Fano)}',
  precision: '\\text{realized error} \\;\\gg\\; \\underbrace{\\tfrac{H(A\\mid Q,D)-1}{\\log_2 K}}_{\\text{Fano floor} \\,=\\, 0} \\quad\\text{(confidently wrong)}',
  frontier: 'C_{\\mathrm{BSC}} = 1 - H_b(p), \\qquad \\text{rate (bits read)} \\;\\leftrightarrow\\; \\text{distortion (answer error)}',
};

export default memo(function RetrieverAsNoisyChannelLaboratory() {
  const [panel, setPanel] = useState<Panel>('channel');
  const [recall, setRecall] = useState(0.6);
  const [eps, setEps] = useState(0.5);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="retriever-as-noisy-channel" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'channel')} onClick={() => setPanel('channel')}>A · the channel</button>
        <button type="button" style={pill(panel === 'recall')} onClick={() => setPanel('recall')}>B · recall floor</button>
        <button type="button" style={pill(panel === 'precision')} onClick={() => setPanel('precision')}>C · precision gap</button>
        <button type="button" style={pill(panel === 'frontier')} onClick={() => setPanel('frontier')}>D · rate–distortion</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'channel' && <ChannelPanel recall={recall} setRecall={setRecall} />}
      {panel === 'recall' && <RecallFloorPanel recall={recall} setRecall={setRecall} />}
      {panel === 'precision' && <PrecisionGapPanel eps={eps} setEps={setEps} />}
      {panel === 'frontier' && <FrontierPanel />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        Finance vMF corpus reused from the PMI topic ({K} companies across {SECTOR[SECTOR.length - 1] + 1} sectors, one filing
        per company), {N_QUERIES} sector-ambiguous queries; the answer model is a synthetic softmax stand-in, not an LLM. Clean
        channel: I₀ = {fmt(I0)} bits, H(A|Q) = {fmt(H_A_GIVEN_Q)} → H(A|Q,D) = {fmt(H_COND0)}. Numbers mirror{' '}
        <code>retriever_as_noisy_channel.py</code>; the lab recomputes entropy, the Fano floors, and the channel capacities in
        closed form.
      </p>
    </div>
  );
});
