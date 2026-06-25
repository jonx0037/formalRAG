import { memo, useEffect, useRef, useState, type ReactNode } from 'react';
import katex from 'katex';

/**
 * Selective Generation & Abstention Laboratory — four panels for the `selective-generation-abstention`
 * topic. The two predecessors decided WHICH CLAIMS to keep inside an answer; this gate decides WHETHER TO
 * ANSWER AT ALL, per query.
 *   A. The risk–coverage curve. Emit the k most-confident answers: coverage rises, selective risk rises.
 *      The achievable curve (ordering by the judge score) sits ABOVE the oracle (ordering by truth); the
 *      gap is the excess AURC, which closes only as the signal's AUC → 1. AURC is area-under-RC, the
 *      answer-level mirror of AP = area-under-PR.
 *   B. Chow's rule. Two cost sliders set the optimal emit cutoff t⋆ = 1 − c_abs/c_err; the expected-cost
 *      curve (recomputed live from the baked score cloud) is U-shaped, and selective generation beats both
 *      always-emit and always-abstain.
 *   C. The two-stage composition. Claim-level conformal back-off certifies faithful claims; the answer is
 *      emitted only if it is not too RISKY (score ≥ t⋆) and not too THIN (≥ MIN_CLAIMS certified claims).
 *   D. The conformal selective-risk guarantee. An α slider sets a distribution-free threshold (recomputed
 *      live from the baked calibration cloud) that controls the test wrong-emission rate at α.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every baked constant below is mirrored TO THE DECIMAL from
 * notebooks/selective-generation-abstention/selective_generation_abstention.py (viz_constants()). Matching
 * asserts: test_coverage_one_is_base_error / test_perfect_judge_collapses_to_oracle /
 * test_chow_threshold_degenerate_costs / test_chow_interior_beats_baselines / test_aurc_is_riemann_area /
 * test_achievable_above_oracle_not_vacuous / test_selective_conformal_controls_and_reuses. The lab
 * recomputes ONLY closed forms in TS from the baked (score, correct) cloud: Panel A re-integrates AURC,
 * Panel B scans the expected-cost curve, Panel C applies the two gates, Panel D scans the conformal-risk-
 * control threshold. Everything corpus-derived (the scores, the per-answer retained counts) is baked (TS
 * cannot reproduce the vMF draws, the Platt fit, or the claim-level back-off). Change a number here ->
 * change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_QUERIES = 32;
const N_CALIB = 16; // split_panel: queries 0..15 calibrate, 16..31 test
const BASE_ERROR_RATE = 0.406;
const ALPHA = 0.1;
const C_ERR = 5.0;
const C_ABS = 1.0;
const EMIT_CUTOFF = 0.8; // Chow cutoff = 1 - c_abs/c_err
const MIN_CLAIMS = 3;

// the per-answer (score, correct) cloud — the source of truth the live recomputes scan.
const SCORES = [0.783, 0.917, 0.737, 0.93, 0.762, 0.504, 0.91, 0.916, 0.53, 0.404, 0.486, 0.571, 0.677, 0.869, 0.533, 0.47, 0.387, 0.486, 0.259, 0.462, 0.442, 0.361, 0.239, 0.187, 0.918, 0.747, 0.763, 0.914, 0.819, 0.824, 0.901, 0.731];
const CORRECT = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0];

type RCPoint = { c: number; risk: number };
const RC_ACHIEVABLE: RCPoint[] = [{ c: 0.0, risk: 0.0 }, { c: 0.031, risk: 0.0 }, { c: 0.062, risk: 0.0 }, { c: 0.094, risk: 0.0 }, { c: 0.125, risk: 0.0 }, { c: 0.156, risk: 0.0 }, { c: 0.188, risk: 0.0 }, { c: 0.219, risk: 0.0 }, { c: 0.25, risk: 0.0 }, { c: 0.281, risk: 0.0 }, { c: 0.312, risk: 0.0 }, { c: 0.344, risk: 0.0 }, { c: 0.375, risk: 0.0 }, { c: 0.406, risk: 0.077 }, { c: 0.438, risk: 0.143 }, { c: 0.469, risk: 0.2 }, { c: 0.5, risk: 0.25 }, { c: 0.531, risk: 0.235 }, { c: 0.562, risk: 0.222 }, { c: 0.594, risk: 0.263 }, { c: 0.625, risk: 0.3 }, { c: 0.656, risk: 0.333 }, { c: 0.688, risk: 0.364 }, { c: 0.719, risk: 0.348 }, { c: 0.75, risk: 0.375 }, { c: 0.781, risk: 0.36 }, { c: 0.812, risk: 0.346 }, { c: 0.844, risk: 0.37 }, { c: 0.875, risk: 0.393 }, { c: 0.906, risk: 0.379 }, { c: 0.938, risk: 0.367 }, { c: 0.969, risk: 0.387 }, { c: 1.0, risk: 0.406 }];
const RC_ORACLE: RCPoint[] = [{ c: 0.0, risk: 0.0 }, { c: 0.031, risk: 0.0 }, { c: 0.062, risk: 0.0 }, { c: 0.094, risk: 0.0 }, { c: 0.125, risk: 0.0 }, { c: 0.156, risk: 0.0 }, { c: 0.188, risk: 0.0 }, { c: 0.219, risk: 0.0 }, { c: 0.25, risk: 0.0 }, { c: 0.281, risk: 0.0 }, { c: 0.312, risk: 0.0 }, { c: 0.344, risk: 0.0 }, { c: 0.375, risk: 0.0 }, { c: 0.406, risk: 0.0 }, { c: 0.438, risk: 0.0 }, { c: 0.469, risk: 0.0 }, { c: 0.5, risk: 0.0 }, { c: 0.531, risk: 0.0 }, { c: 0.562, risk: 0.0 }, { c: 0.594, risk: 0.0 }, { c: 0.625, risk: 0.05 }, { c: 0.656, risk: 0.095 }, { c: 0.688, risk: 0.136 }, { c: 0.719, risk: 0.174 }, { c: 0.75, risk: 0.208 }, { c: 0.781, risk: 0.24 }, { c: 0.812, risk: 0.269 }, { c: 0.844, risk: 0.296 }, { c: 0.875, risk: 0.321 }, { c: 0.906, risk: 0.345 }, { c: 0.938, risk: 0.367 }, { c: 0.969, risk: 0.387 }, { c: 1.0, risk: 0.406 }];
const AURC = { achievable: 0.185, oracle: 0.097, gap: 0.088, auc: 0.773 };

const COST = { at_chow: 0.656, always_emit: 2.031, always_abstain: 1.0 };

// Panel C — the two-stage worked decision (score, correct, retained-claim count). emit is recomputed live.
type StageRow = { score: number; correct: number; retained: number };
const TWO_STAGE: StageRow[] = [{ score: 0.783, correct: 1, retained: 6 }, { score: 0.917, correct: 1, retained: 6 }, { score: 0.737, correct: 0, retained: 5 }, { score: 0.93, correct: 1, retained: 8 }, { score: 0.762, correct: 0, retained: 5 }, { score: 0.504, correct: 0, retained: 2 }, { score: 0.91, correct: 1, retained: 6 }, { score: 0.916, correct: 1, retained: 7 }, { score: 0.53, correct: 0, retained: 2 }, { score: 0.404, correct: 0, retained: 3 }, { score: 0.486, correct: 0, retained: 4 }, { score: 0.571, correct: 1, retained: 3 }, { score: 0.677, correct: 1, retained: 3 }, { score: 0.869, correct: 1, retained: 5 }, { score: 0.533, correct: 0, retained: 4 }, { score: 0.47, correct: 0, retained: 1 }, { score: 0.387, correct: 0, retained: 1 }, { score: 0.486, correct: 1, retained: 0 }, { score: 0.259, correct: 1, retained: 0 }, { score: 0.462, correct: 1, retained: 2 }, { score: 0.442, correct: 1, retained: 2 }, { score: 0.361, correct: 1, retained: 0 }, { score: 0.239, correct: 0, retained: 1 }, { score: 0.187, correct: 0, retained: 1 }, { score: 0.918, correct: 1, retained: 7 }, { score: 0.747, correct: 0, retained: 4 }, { score: 0.763, correct: 1, retained: 6 }, { score: 0.914, correct: 1, retained: 6 }, { score: 0.819, correct: 1, retained: 6 }, { score: 0.824, correct: 1, retained: 5 }, { score: 0.901, correct: 1, retained: 7 }, { score: 0.731, correct: 0, retained: 5 }];

// Panel D recomputes the conformal threshold live from the calibration half (below); at α = 0.1 it
// reproduces the .py's CONFORMAL = {threshold: 0.78, realized_risk: 0.0, coverage: 0.312}.
const CAL_GAP = { ece: 0.22, brier: 0.195 };

// --- theme + helpers ------------------------------------------------------------------
const ACCENT = 'var(--color-accent)';
const MUTED = '#9aa3ad';
const POS_COLOR = '#6fa389'; // correct / emit-correctly
const NEG_COLOR = '#c0726a'; // wrong / risky
const GUAR_COLOR = '#6a8caf'; // oracle / the conformal guarantee
const fmt = (x: number, n = 3) => (Number.isFinite(x) ? x.toFixed(n) : '∞');
const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

// the conformal-risk-control grid (LAMBDA_GRID): 51 points on [0,1], matching the .py.
const GRID = Array.from({ length: 51 }, (_, i) => Math.round((i / 50) * 1e4) / 1e4);

function trapz(pts: RCPoint[]): number {
  let a = 0;
  for (let i = 1; i < pts.length; i++) a += 0.5 * (pts[i].risk + pts[i - 1].risk) * (pts[i].c - pts[i - 1].c);
  return a;
}

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
      <span style={{ minWidth: '15rem' }}>{label} = <strong>{display}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
    </label>
  );
}
const Row = ({ children }: { children: ReactNode }) => (
  <div style={{ display: 'flex', gap: '1.4rem', flexWrap: 'wrap', margin: '0.6rem 0 0.2rem' }}>{children}</div>
);
const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

// ===== Panel A — the risk–coverage curve ===========================================================
function RiskCoveragePanel({ idx, setIdx }: { idx: number; setIdx: (v: number) => void }) {
  const W = 560, H = 270, padL = 44, padR = 16, padT = 14, padB = 36;
  const yMax = 0.45;
  const x = (c: number) => padL + c * (W - padL - padR);
  const y = (r: number) => H - padB - (r / yMax) * (H - padT - padB);
  const pathOf = (pts: RCPoint[]) => pts.map((d, i) => `${i ? 'L' : 'M'}${x(d.c)},${y(d.risk)}`).join(' ');
  const op = RC_ACHIEVABLE[idx];
  const opOracle = RC_ORACLE[idx];
  const aurcLive = trapz(RC_ACHIEVABLE); // recomputed live; equals the baked AURC.achievable
  // the gap band between the two curves
  const band = RC_ACHIEVABLE.map((d, i) => `${i ? 'L' : 'M'}${x(d.c)},${y(d.risk)}`).join(' ')
    + ' ' + RC_ORACLE.slice().reverse().map((d) => `L${x(d.c)},${y(d.risk)}`).join(' ') + ' Z';
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="the risk-coverage curve: achievable versus oracle, with the excess AURC gap" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <path d={band} fill={NEG_COLOR} fillOpacity={0.1} stroke="none" />
        {/* axes */}
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={`x${t}`} x={x(t)} y={H - padB + 13} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
        ))}
        {[0, 0.1, 0.2, 0.3, 0.4].map((t) => (
          <text key={`y${t}`} x={padL - 6} y={y(t) + 3} textAnchor="end" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t.toFixed(1)}</text>
        ))}
        {/* the oracle (lower envelope) and achievable curves */}
        <path d={pathOf(RC_ORACLE)} fill="none" stroke={GUAR_COLOR} strokeWidth={1.8} strokeDasharray="5 3" />
        <path d={pathOf(RC_ACHIEVABLE)} fill="none" stroke={ACCENT} strokeWidth={2} />
        {/* operating points */}
        <circle cx={x(opOracle.c)} cy={y(opOracle.risk)} r={4} fill={GUAR_COLOR} />
        <circle cx={x(op.c)} cy={y(op.risk)} r={5.5} fill={ACCENT} stroke="var(--color-bg)" strokeWidth={1} />
        <line x1={x(op.c)} y1={padT} x2={x(op.c)} y2={H - padB} stroke={ACCENT} strokeWidth={1} strokeDasharray="3 3" opacity={0.5} />
        <text x={W - padR} y={padT + 10} textAnchor="end" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">achievable (judge score)</text>
        <text x={W - padR} y={padT + 22} textAnchor="end" fontSize={8.5} fill={GUAR_COLOR} fontFamily="var(--font-sans)">oracle (truth)</text>
        <text x={(padL + W) / 2} y={H - 4} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">coverage = fraction of queries answered →</text>
        <text x={12} y={(padT + H - padB) / 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + H - padB) / 2})`}>selective risk →</text>
      </svg>
      <Slider label="coverage (emit the k most-confident answers)" value={idx} min={0} max={RC_ACHIEVABLE.length - 1} step={1}
        onChange={(v) => setIdx(Math.round(v))} display={pct(op.c)} />
      <Row>
        <Readout label="selective risk (achievable)" value={fmt(op.risk, 3)} accent />
        <Readout label="selective risk (oracle)" value={fmt(opOracle.risk, 3)} />
        <Readout label="AURC (achievable)" value={fmt(aurcLive, 3)} accent />
        <Readout label="excess AURC (gap)" value={fmt(AURC.gap, 3)} />
        <Readout label="signal AUC" value={fmt(AURC.auc, 3)} />
      </Row>
    </div>
  );
}

// ===== Panel B — Chow's rule cost model ============================================================
function expectedCost(t: number, cErr: number, cAbs: number) {
  let s = 0;
  for (let i = 0; i < SCORES.length; i++) {
    if (SCORES[i] >= t) s += CORRECT[i] === 1 ? 0 : cErr;
    else s += cAbs;
  }
  return s / SCORES.length;
}
function ChowPanel({ cErr, setCErr, cAbs, setCAbs }: {
  cErr: number; setCErr: (v: number) => void; cAbs: number; setCAbs: (v: number) => void;
}) {
  const W = 560, H = 250, padL = 44, padR = 16, padT = 14, padB = 34;
  const tStar = cErr > 0 ? Math.min(1, Math.max(0, 1 - cAbs / cErr)) : 0;
  const grid = Array.from({ length: 101 }, (_, i) => i / 100);
  const curve = grid.map((t) => ({ t, cost: expectedCost(t, cErr, cAbs) }));
  const costMax = Math.max(...curve.map((d) => d.cost), cAbs) * 1.05 || 1;
  const best = curve.reduce((a, b) => (b.cost < a.cost ? b : a));
  const alwaysEmit = expectedCost(0, cErr, cAbs);
  const x = (t: number) => padL + t * (W - padL - padR);
  const y = (c: number) => H - padB - (c / costMax) * (H - padT - padB);
  const path = curve.map((d, i) => `${i ? 'L' : 'M'}${x(d.t)},${y(d.cost)}`).join(' ');
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="the expected-cost curve with the Chow optimal threshold and the abstain region" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* abstain region: scores below t-star are abstained */}
        <rect x={padL} y={padT} width={x(tStar) - padL} height={H - padB - padT} fill={MUTED} fillOpacity={0.08} />
        <text x={padL + 4} y={padT + 9} fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">abstain region (score &lt; t⋆)</text>
        {/* baseline cost lines */}
        <line x1={padL} y1={y(cAbs)} x2={W - padR} y2={y(cAbs)} stroke={GUAR_COLOR} strokeWidth={1} strokeDasharray="4 3" />
        <text x={W - padR} y={y(cAbs) - 3} textAnchor="end" fontSize={8} fill={GUAR_COLOR} fontFamily="var(--font-sans)">always-abstain = c_abs</text>
        {/* axes */}
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={x(t)} y={H - padB + 13} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
        ))}
        {/* the cost curve */}
        <path d={path} fill="none" stroke={ACCENT} strokeWidth={2} />
        {/* Chow cutoff line + empirical minimum */}
        <line x1={x(tStar)} y1={padT} x2={x(tStar)} y2={H - padB} stroke={NEG_COLOR} strokeWidth={1.6} strokeDasharray="4 3" />
        <text x={x(tStar)} y={padT + 9} textAnchor="middle" fontSize={8.5} fill={NEG_COLOR} fontFamily="var(--font-sans)">t⋆ = {fmt(tStar, 2)}</text>
        <circle cx={x(best.t)} cy={y(best.cost)} r={5} fill={POS_COLOR} stroke="var(--color-bg)" strokeWidth={1} />
        <text x={x(best.t)} y={y(best.cost) - 8} textAnchor="middle" fontSize={8} fill={POS_COLOR} fontFamily="var(--font-sans)">min cost {fmt(best.cost, 2)}</text>
        <text x={(padL + W) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">emit threshold t (emit iff score ≥ t) →</text>
        <text x={12} y={(padT + H - padB) / 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + H - padB) / 2})`}>expected cost →</text>
      </svg>
      <Slider label="cost of a wrong answer c_err" value={cErr} min={1} max={10} step={0.5} onChange={setCErr} display={fmt(cErr, 1)} />
      <Slider label="cost of abstaining c_abs" value={cAbs} min={0} max={5} step={0.5} onChange={setCAbs} display={fmt(cAbs, 1)} />
      <Row>
        <Readout label="Chow cutoff t⋆ = 1 − c_abs/c_err" value={fmt(tStar, 3)} accent />
        <Readout label="cost at the optimum" value={fmt(best.cost, 3)} accent />
        <Readout label="always-emit cost" value={fmt(alwaysEmit, 3)} />
        <Readout label="always-abstain cost" value={fmt(cAbs, 3)} />
      </Row>
    </div>
  );
}

// ===== Panel C — the two-stage composition =========================================================
function TwoStagePanel({ minClaims, setMinClaims }: { minClaims: number; setMinClaims: (v: number) => void }) {
  const W = 560, H = 270, padL = 44, padR = 16, padT = 14, padB = 36;
  const retMax = 8;
  const x = (s: number) => padL + s * (W - padL - padR);
  const y = (r: number) => H - padB - (r / retMax) * (H - padT - padB);
  const decide = (d: StageRow) => d.score >= EMIT_CUTOFF && d.retained >= minClaims;
  const emit = TWO_STAGE.filter(decide);
  const risky = TWO_STAGE.filter((d) => d.score < EMIT_CUTOFF).length;
  const thin = TWO_STAGE.filter((d) => d.score >= EMIT_CUTOFF && d.retained < minClaims).length;
  const nEmit = emit.length;
  const wrongEmit = emit.filter((d) => d.correct === 0).length;
  const residual = nEmit ? wrongEmit / nEmit : 0;
  const cost = TWO_STAGE.reduce((s, d) => s + (decide(d) ? (d.correct === 1 ? 0 : C_ERR) : C_ABS), 0) / TWO_STAGE.length;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="the two-stage decision: each answer by score and certified-claim count, with the risk and thinness gates" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* emit region: score >= cutoff AND retained >= minClaims */}
        <rect x={x(EMIT_CUTOFF)} y={padT} width={W - padR - x(EMIT_CUTOFF)} height={y(minClaims) - padT} fill={POS_COLOR} fillOpacity={0.09} />
        <text x={W - padR - 4} y={padT + 10} textAnchor="end" fontSize={8} fill={POS_COLOR} fontFamily="var(--font-sans)">emit region</text>
        {/* axes */}
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={x(t)} y={H - padB + 13} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
        ))}
        {[0, 2, 4, 6, 8].map((t) => (
          <text key={t} x={padL - 6} y={y(t) + 3} textAnchor="end" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
        ))}
        {/* gate lines */}
        <line x1={x(EMIT_CUTOFF)} y1={padT} x2={x(EMIT_CUTOFF)} y2={H - padB} stroke={NEG_COLOR} strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={x(EMIT_CUTOFF)} y={H - padB + 24} textAnchor="middle" fontSize={8} fill={NEG_COLOR} fontFamily="var(--font-sans)">risk gate t⋆ = {fmt(EMIT_CUTOFF, 2)}</text>
        <line x1={padL} y1={y(minClaims)} x2={W - padR} y2={y(minClaims)} stroke={ACCENT} strokeWidth={1.4} strokeDasharray="4 3" />
        {/* the answers */}
        {TWO_STAGE.map((d, i) => {
          const e = decide(d);
          return (
            <circle key={i} cx={x(d.score)} cy={y(d.retained) + (i % 3 - 1) * 2.4} r={e ? 5 : 3.4}
              fill={d.correct === 1 ? POS_COLOR : NEG_COLOR} fillOpacity={e ? 0.95 : 0.45}
              stroke={e ? 'var(--color-bg)' : 'none'} strokeWidth={e ? 1 : 0} />
          );
        })}
        <text x={(padL + W) / 2} y={H - 4} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">answer score (risk gate) →</text>
        <text x={12} y={(padT + H - padB) / 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + H - padB) / 2})`}>certified claims (thinness gate) →</text>
      </svg>
      <Slider label="thinness gate: minimum certified claims" value={minClaims} min={0} max={8} step={1}
        onChange={(v) => setMinClaims(Math.round(v))} display={String(minClaims)} />
      <Row>
        <Readout label="emit" value={`${nEmit} / ${N_QUERIES}`} accent />
        <Readout label="abstain — too risky" value={String(risky)} />
        <Readout label="abstain — too thin" value={String(thin)} accent={thin > 0} />
        <Readout label="residual error (emitted)" value={fmt(residual, 3)} accent={residual > 0} />
        <Readout label="cost" value={fmt(cost, 3)} />
      </Row>
    </div>
  );
}

// ===== Panel D — the conformal selective-risk guarantee ============================================
function conformalThreshold(alpha: number) {
  // CRC: the smallest grid cut t with (n/(n+1))*Rhat(t) + 1/(n+1) <= alpha, on the calibration half.
  const n = N_CALIB;
  for (const t of GRID) {
    let wrongEmit = 0;
    for (let i = 0; i < n; i++) if (SCORES[i] >= t && CORRECT[i] === 0) wrongEmit += 1;
    const adjusted = (n / (n + 1)) * (wrongEmit / n) + 1 / (n + 1);
    if (adjusted <= alpha) return t;
  }
  return GRID[GRID.length - 1];
}
function ConformalPanel({ alpha, setAlpha }: { alpha: number; setAlpha: (v: number) => void }) {
  const t = conformalThreshold(alpha);
  // realized test wrong-emission rate + coverage at this threshold
  let wrong = 0, emit = 0;
  for (let i = N_CALIB; i < N_QUERIES; i++) {
    if (SCORES[i] >= t) { emit += 1; if (CORRECT[i] === 0) wrong += 1; }
  }
  const realized = (N_QUERIES - N_CALIB) > 0 ? wrong / (N_QUERIES - N_CALIB) : 0;
  const coverage = (N_QUERIES - N_CALIB) > 0 ? emit / (N_QUERIES - N_CALIB) : 0;
  const W = 560, H = 230, padL = 30, padR = 16, padT = 16, padB = 40;
  const x = (s: number) => padL + s * (W - padL - padR);
  const calib = SCORES.slice(0, N_CALIB);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="the calibration scores with the conformal threshold cut" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* emit region (score >= threshold) */}
        <rect x={x(t)} y={padT} width={W - padR - x(t)} height={H - padB - padT} fill={POS_COLOR} fillOpacity={0.08} />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text key={v} x={x(v)} y={H - padB + 13} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>
        ))}
        {/* calibration scores as ticks, colored by truth */}
        {calib.map((sv, i) => (
          <line key={i} x1={x(sv)} y1={H - padB} x2={x(sv)} y2={H - padB - 30 - (i % 5) * 9}
            stroke={CORRECT[i] === 1 ? POS_COLOR : NEG_COLOR} strokeWidth={2} opacity={0.8} />
        ))}
        {/* the conformal threshold */}
        <line x1={x(t)} y1={padT} x2={x(t)} y2={H - padB} stroke={GUAR_COLOR} strokeWidth={1.8} strokeDasharray="4 3" />
        <text x={x(t)} y={padT + 10} textAnchor="middle" fontSize={8.5} fill={GUAR_COLOR} fontFamily="var(--font-sans)">conformal cut t̂ = {fmt(t, 2)}</text>
        <text x={x(t) + 4} y={H - padB - 4} fontSize={8} fill={POS_COLOR} fontFamily="var(--font-sans)">emit ⟶</text>
        <text x={padL} y={padT + 10} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">calibration answers (green = correct, red = wrong)</text>
        <text x={(padL + W) / 2} y={H - 6} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">answer score →</text>
      </svg>
      <Slider label="target wrong-emission rate α" value={alpha} min={0.02} max={0.5} step={0.02} onChange={setAlpha} display={fmt(alpha, 2)} />
      <Row>
        <Readout label="conformal threshold t̂" value={fmt(t, 3)} accent />
        <Readout label="realized wrong-emission (test)" value={`${fmt(realized, 3)} ≤ ${fmt(alpha, 2)}`} accent={realized <= alpha} />
        <Readout label="coverage (test)" value={pct(coverage)} />
        <Readout label="score ECE (calibration gap)" value={fmt(CAL_GAP.ece, 3)} />
      </Row>
    </div>
  );
}

// ===== root ========================================================================================
type Panel = 'riskcoverage' | 'chow' | 'twostage' | 'conformal';
const TEX: Record<Panel, string> = {
  riskcoverage: '\\mathrm{AURC} = \\int_0^1 \\mathrm{risk}(c)\\,dc, \\qquad \\mathrm{risk}(c) = \\Pr\\!\\big(\\text{wrong}\\mid\\text{emit at coverage }c\\big)',
  chow: '\\text{emit} \\iff \\Pr(\\text{correct}) \\ge 1 - \\dfrac{c_{\\mathrm{abs}}}{c_{\\mathrm{err}}} \\;=\\; t^\\star \\qquad (\\text{Chow 1970})',
  twostage: '\\text{emit} \\iff \\underbrace{s \\ge t^\\star}_{\\text{not too risky}} \\;\\wedge\\; \\underbrace{|\\hat C_{\\hat\\lambda}| \\ge m}_{\\text{not too thin}}',
  conformal: '\\hat t = \\inf\\Big\\{t : \\tfrac{n}{n+1}\\hat R_n(t) + \\tfrac{1}{n+1} \\le \\alpha\\Big\\}, \\quad \\mathbb{E}\\big[\\text{wrong-emission rate}\\big] \\le \\alpha',
};

export default memo(function SelectiveGenerationAbstentionLaboratory() {
  const [panel, setPanel] = useState<Panel>('riskcoverage');
  const [idx, setIdx] = useState(10); // Panel A coverage index (~0.31, the operating region)
  const [cErr, setCErr] = useState(C_ERR);
  const [cAbs, setCAbs] = useState(C_ABS);
  const [minClaims, setMinClaims] = useState(MIN_CLAIMS);
  const [alpha, setAlpha] = useState(ALPHA);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="selective-generation-abstention" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'riskcoverage')} onClick={() => setPanel('riskcoverage')}>A · risk–coverage</button>
        <button type="button" style={pill(panel === 'chow')} onClick={() => setPanel('chow')}>B · Chow's rule</button>
        <button type="button" style={pill(panel === 'twostage')} onClick={() => setPanel('twostage')}>C · two-stage gate</button>
        <button type="button" style={pill(panel === 'conformal')} onClick={() => setPanel('conformal')}>D · conformal guarantee</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'riskcoverage' && <RiskCoveragePanel idx={idx} setIdx={setIdx} />}
      {panel === 'chow' && <ChowPanel cErr={cErr} setCErr={setCErr} cAbs={cAbs} setCAbs={setCAbs} />}
      {panel === 'twostage' && <TwoStagePanel minClaims={minClaims} setMinClaims={setMinClaims} />}
      {panel === 'conformal' && <ConformalPanel alpha={alpha} setAlpha={setAlpha} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_QUERIES} synthetic finance answers (one per query, base error {fmt(BASE_ERROR_RATE, 3)}); the
        lenient judge's answer score has AUC {fmt(AURC.auc, 2)} — informative but imperfect, so the
        achievable risk–coverage curve sits above the oracle (excess AURC {fmt(AURC.gap, 3)}). At the
        default cost ratio selective generation costs {fmt(COST.at_chow, 2)}, beating always-emit
        ({fmt(COST.always_emit, 2)}) and always-abstain ({fmt(COST.always_abstain, 2)}). Numbers mirror
        <code> selective_generation_abstention.py</code>; the lab recomputes the cost curve, the two gates,
        and the conformal threshold live from the baked (score, correct) cloud. The Chow rule assumes a
        calibrated score (ECE {fmt(CAL_GAP.ece, 3)} here) — the calibration gap is why the realized optimum
        can drift from t⋆.
      </p>
    </div>
  );
});
