import { memo, useEffect, useRef, useState, type ReactNode } from 'react';
import katex from 'katex';

/**
 * Adaptive Retrieval Routing Laboratory — four panels for `adaptive-retrieval-routing`.
 * The published gate decided emit-or-abstain for one answer; this router decides WHICH RETRIEVAL
 * STRATEGY to run, per query, from features available before any retrieval fires.
 *   A. The three arms. Quality and answer-correctness by query class. The compositional class is the
 *      reversal that makes routing possible at all: a single retrieval NEVER reaches the mentioned
 *      company, iterating usually does, and on the easy class every arm is already correct.
 *   B. The Jensen gap. A lambda slider recomputes E[max_a U_a] and max_a E[U_a] LIVE from the baked
 *      per-query quality matrix. The gap is what any router could win; it collapses when one arm is
 *      uniformly best, which is the theorem stated as a picture.
 *   C. The frontier. Three nested curves on HELD-OUT queries: the no-feature hull of the fixed arms,
 *      the achievable router, the oracle.
 *   D. What it actually wins. In-sample against held-out at four operating points, with the paired
 *      p-value. At lambda = 0.10 the realizable router LOSES although the oracle would win.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every baked constant below is mirrored TO THE DECIMAL from
 * notebooks/adaptive-retrieval-routing/adaptive_retrieval_routing.py (viz_constants()). Matching
 * asserts: test_corpus_is_not_vacuous / test_bridge_needs_iteration /
 * test_jensen_gap_nonnegative_and_positive_here / test_gap_vanishes_when_one_arm_dominates /
 * test_displayed_frontier_is_held_out_and_ordered / test_in_sample_overstates_the_gain /
 * test_router_can_lose_to_the_best_fixed_arm / test_gain_is_reported_as_an_estimate. The lab
 * recomputes ONLY closed forms in TS: Panel B re-derives the whole Jensen decomposition from
 * Q_MATRIX for any lambda. Everything corpus-derived (the vMF draws, the ridge fit, the CV) is
 * baked — TS cannot reproduce it. Change a number here -> change it there, and re-run the notebook.
 */

// --- baked from adaptive_retrieval_routing.py ---
const ARMS = ['none', 'single', 'iterative'];
const ARM_COST: Record<string, number> = { none: 0.0, single: 1.0, iterative: 3.0 };
const CLASSES = ['known', 'local', 'bridge'];
const QUALITY_BY_CLASS: Record<string, number[]> = {'known': [0.424, 0.769, 0.744], 'local': [0.292, 0.654, 0.639], 'bridge': [0.051, 0.048, 0.278]};
const CORRECT_BY_CLASS: Record<string, number[]> = {'known': [24, 24, 24], 'local': [24, 23, 22], 'bridge': [0, 0, 14]};  // out of 24 each
const FEATURES_BY_CLASS: Record<string, number[]> = {'known': [2.857, 0.963, 0.33, 0.266], 'local': [3.356, 0.713, 0.218, 0.266], 'bridge': [3.185, 0.882, 0.282, 0.391]};  // [H(prior), max cos, top2 margin, cross-sector]
const JENSEN: { lam: number; oracle: number; best_fixed: number; gap: number }[] = [{'lam': 0.0, 'oracle': 0.5708, 'best_fixed': 0.5535, 'gap': 0.0172}, {'lam': 0.02, 'oracle': 0.5449, 'best_fixed': 0.4935, 'gap': 0.0514}, {'lam': 0.05, 'oracle': 0.5065, 'best_fixed': 0.4402, 'gap': 0.0663}, {'lam': 0.1, 'oracle': 0.4451, 'best_fixed': 0.3902, 'gap': 0.0549}, {'lam': 0.2, 'oracle': 0.3624, 'best_fixed': 0.2902, 'gap': 0.0722}, {'lam': 0.4, 'oracle': 0.2614, 'best_fixed': 0.2558, 'gap': 0.0056}];
const FIXED_ARM_POINTS = [[0.0, 0.2558], [1.0, 0.4902], [3.0, 0.5535]];
const N_QUERIES = 72;  const N_PER_CLASS = 24;
const Q_MATRIX = [[0.4216,0.7919,0.7919], [0.3804,0.7315,0.7315], [0.4062,0.7534,0.7534], [0.4174,0.8124,0.8124], [0.3934,0.717,0.6293], [0.4665,0.7309,0.5495], [0.4392,0.7348,0.5603], [0.4928,0.8554,0.8554], [0.4819,0.7542,0.5881], [0.4353,0.7849,0.7849], [0.4725,0.808,0.808], [0.4713,0.808,0.808], [0.3353,0.7058,0.7058], [0.4053,0.7032,0.7032], [0.4145,0.8153,0.8153], [0.3909,0.7419,0.7419], [0.4146,0.7992,0.7992], [0.3912,0.7274,0.7274], [0.4342,0.7668,0.7668], [0.4251,0.8458,0.8458], [0.4036,0.7335,0.7335], [0.4285,0.7683,0.7683], [0.4175,0.7931,0.7931], [0.447,0.7781,0.7781], [0.2919,0.7025,0.7025], [0.244,0.6157,0.6157], [0.3247,0.6794,0.6794], [0.3021,0.7028,0.7028], [0.31,0.6616,0.6616], [0.2287,0.2534,0.2534], [0.2005,0.3592,0.3592], [0.3188,0.7441,0.7441], [0.2902,0.6375,0.6375], [0.1957,0.4081,0.4081], [0.3492,0.737,0.737], [0.2601,0.6652,0.6652], [0.3566,0.7345,0.7345], [0.2957,0.7418,0.7418], [0.2724,0.6954,0.6954], [0.2889,0.7377,0.7377], [0.2213,0.5647,0.5647], [0.3485,0.6014,0.2476], [0.3943,0.751,0.751], [0.2498,0.7428,0.7428], [0.2653,0.6349,0.6349], [0.3794,0.827,0.827], [0.3269,0.7594,0.7594], [0.285,0.7356,0.7356], [0.0296,0.0026,0.0026], [0.1289,0.1721,0.4108], [0.041,0.0493,0.3103], [0.0478,0.0708,0.4198], [0.0431,0.0478,0.3762], [0.0198,0.0008,0.0008], [0.0294,0.0031,0.0031], [0.0318,0.0282,0.3518], [0.0344,0.023,0.036], [0.0711,0.0868,0.6078], [0.0448,0.05,0.4282], [0.0615,0.0832,0.6968], [0.1107,0.0471,0.0471], [0.094,0.1209,0.7492], [0.0674,0.0127,0.0127], [0.0257,0.002,0.002], [0.0299,0.0015,0.0015], [0.1046,0.137,0.401], [0.0346,0.0427,0.2723], [0.0411,0.058,0.5867], [0.0411,0.0435,0.2973], [0.0369,0.0041,0.0017], [0.0344,0.0309,0.3408], [0.0265,0.0237,0.3066]];
const KLASS_OF_QUERY = ['known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'known', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge', 'bridge'];
const TEST_IDX = [3, 5, 7, 8, 9, 11, 13, 15, 17, 18, 21, 23, 25, 27, 29, 30, 31, 33, 34, 35, 38, 41, 43, 44, 48, 49, 50, 52, 54, 56, 57, 58, 60, 62, 66, 67];
const ROUTER_REPORT: { lam: number; inSample: number; heldOut: number; oracle: number; frac: number; p: number; ece: number }[] = [{'lam': 0.02, 'inSample': 0.0262, 'heldOut': 0.0184, 'oracle': 0.0592, 'frac': 0.31, 'p': 0.2364, 'ece': 0.179}, {'lam': 0.05, 'inSample': 0.0311, 'heldOut': 0.0022, 'oracle': 0.0623, 'frac': 0.035, 'p': 0.9019, 'ece': 0.141}, {'lam': 0.1, 'inSample': 0.0137, 'heldOut': -0.0203, 'oracle': 0.0543, 'frac': -0.374, 'p': 0.0687, 'ece': 0.141}, {'lam': 0.2, 'inSample': 0.0459, 'heldOut': 0.0432, 'oracle': 0.0749, 'frac': 0.577, 'p': 0.0218, 'ece': 0.173}];
const ROUTER_FRONTIER = [[2.111, 0.5314], [2.056, 0.5314], [2.0, 0.5314], [1.944, 0.5311], [1.889, 0.5311], [1.833, 0.5311], [1.722, 0.5262], [1.611, 0.5157], [1.5, 0.5157], [1.444, 0.5012], [1.333, 0.5012], [1.25, 0.492], [0.944, 0.4797], [0.806, 0.4741], [0.722, 0.4595], [0.583, 0.4244], [0.444, 0.3836], [0.306, 0.3445], [0.167, 0.3093], [0.0, 0.2548]];   // held-out queries only
const ORACLE_FRONTIER = [[1.25, 0.546], [1.222, 0.5453], [1.056, 0.5312], [0.889, 0.5141], [0.806, 0.5035], [0.778, 0.4991], [0.611, 0.469], [0.556, 0.4561], [0.472, 0.4329], [0.194, 0.3362], [0.028, 0.2685], [0.0, 0.2548]];   // held-out queries only
const FIXED_HULL = [[0.0, 0.2548], [1.0, 0.4719], [3.0, 0.5218]];
const N_TEST = 36;
const CHOW_THRESHOLD = 0.8;  // c_err=5.0, c_abs=1.0
const LAM_HEADLINE = 0.2;
const LAM_ROUTER_LOSES = 0.1;

const COSTS = [ARM_COST.none, ARM_COST.single, ARM_COST.iterative];
const ARM_LABEL: Record<string, string> = { none: 'no retrieval', single: 'retrieve once', iterative: 'iterate' };
const CLASS_LABEL: Record<string, string> = { known: 'answerable from the query', local: 'one company, ambiguous', bridge: 'compositional (two entities)' };

const fmt = (x: number, n = 3) => (Number.isFinite(x) ? x.toFixed(n) : '\u221e');
const signed = (x: number, n = 3) => (x >= 0 ? '+' : '\u2212') + Math.abs(x).toFixed(n);

const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

function Readout({ label, value, accent, warn }: { label: string; value: string; accent?: boolean; warn?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: warn ? 'var(--color-badge-red-text)' : accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
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

const Note = ({ children }: { children: ReactNode }) => (
  <p style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: '0.6rem 0 0' }}>{children}</p>
);

/** THE live recompute: the whole Jensen decomposition, closed form, for any lambda. */
function jensenAt(lam: number) {
  const U = Q_MATRIX.map((row) => row.map((q, a) => q - lam * COSTS[a]));
  const n = U.length;
  const oracle = U.reduce((s, r) => s + Math.max(...r), 0) / n;
  const perArm = COSTS.map((_, a) => U.reduce((s, r) => s + r[a], 0) / n);
  const bestFixed = Math.max(...perArm);
  const bestArm = perArm.indexOf(bestFixed);
  const wins = [0, 0, 0];
  for (const r of U) {
    let bi = 0;
    for (let i = 1; i < r.length; i++) if (r[i] > r[bi]) bi = i;
    wins[bi] += 1;
  }
  return { oracle, perArm, bestFixed, bestArm, gap: oracle - bestFixed, wins };
}

// ===== Panel A — the three arms ====================================================================
function ArmsPanel() {
  const W = 560, H = 250, padL = 150, padR = 90, padT = 22, padB = 30;
  const rowH = (H - padT - padB) / CLASSES.length;
  const barW = (v: number) => v * (W - padL - padR);
  const colors = ['var(--color-text-secondary)', 'var(--color-accent)', 'var(--color-accent-secondary)'];
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="mean answer quality by query class and retrieval arm" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {CLASSES.map((k, ci) => {
          const q = QUALITY_BY_CLASS[k];
          const corr = CORRECT_BY_CLASS[k];
          const y0 = padT + ci * rowH;
          return (
            <g key={k}>
              <text x={padL - 8} y={y0 + rowH / 2 - 6} textAnchor="end" fontSize="10" fill="var(--color-text)" fontWeight="600">{k}</text>
              <text x={padL - 8} y={y0 + rowH / 2 + 7} textAnchor="end" fontSize="8" fill="var(--color-text-secondary)">{CLASS_LABEL[k]}</text>
              {q.map((v, ai) => {
                const y = y0 + 6 + ai * ((rowH - 14) / 3);
                return (
                  <g key={ai}>
                    <rect x={padL} y={y} width={Math.max(barW(v), 1)} height={(rowH - 18) / 3} fill={colors[ai]} opacity={0.85} rx={1} />
                    <text x={padL + barW(v) + 5} y={y + (rowH - 18) / 6 + 3} fontSize="8" fill="var(--color-text-secondary)">
                      {fmt(v, 2)} · {corr[ai]}/{N_PER_CLASS} right
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        <text x={padL} y={H - padB + 14} fontSize="9" fill="var(--color-text-secondary)">0</text>
        <text x={W - padR} y={H - padB + 14} textAnchor="end" fontSize="9" fill="var(--color-text-secondary)">1 · posterior mass on the gold company</text>
      </svg>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', fontSize: '0.72rem', marginTop: '0.4rem' }}>
        {ARMS.map((a, i) => (
          <span key={a} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ width: 10, height: 10, background: colors[i], display: 'inline-block', borderRadius: 2 }} />
            {ARM_LABEL[a]} <span style={{ color: 'var(--color-text-secondary)' }}>(cost {ARM_COST[a]})</span>
          </span>
        ))}
      </div>
      <Note>
        The compositional class is the whole reason a router can exist here. A single retrieval reaches
        the mentioned company <strong>{CORRECT_BY_CLASS.bridge[1]} times out of {N_PER_CLASS}</strong> —
        never — because the bridge filing points mostly at the company it is <em>about</em>, not the one
        it <em>names</em>; only reformulating off it recovers the second direction. On the easy class all
        three arms are already right {CORRECT_BY_CLASS.known[0]}/{N_PER_CLASS} of the time, so routing
        there is purely a question of what you are willing to pay.
      </Note>
    </div>
  );
}

// ===== Panel B — the Jensen gap, recomputed live ===================================================
function GapPanel({ lam, setLam, degenerate, setDegenerate }: {
  lam: number; setLam: (v: number) => void; degenerate: boolean; setDegenerate: (v: boolean) => void;
}) {
  const W = 560, H = 210, padL = 46, padR = 16, padT = 16, padB = 34;
  // the degenerate control: one shared signal plus fixed per-arm offsets -> ordering cannot vary
  const live = (() => {
    if (!degenerate) return jensenAt(lam);
    const base = Q_MATRIX.map((r) => r.reduce((s, v) => s + v, 0) / r.length);
    const lo = Math.min(...base), hi = Math.max(...base);
    const scaled = base.map((b) => (hi - lo > 1e-12 ? (0.6 * (b - lo)) / (hi - lo) : 0));
    const offsets = [0.0, 0.3, 0.1];
    const U = scaled.map((b) => offsets.map((o, a) => b + o - lam * COSTS[a]));
    const n = U.length;
    const oracle = U.reduce((s, r) => s + Math.max(...r), 0) / n;
    const perArm = COSTS.map((_, a) => U.reduce((s, r) => s + r[a], 0) / n);
    const bestFixed = Math.max(...perArm);
    const wins = [0, 0, 0];
    for (const r of U) { let bi = 0; for (let i = 1; i < r.length; i++) if (r[i] > r[bi]) bi = i; wins[bi] += 1; }
    return { oracle, perArm, bestFixed, bestArm: perArm.indexOf(bestFixed), gap: oracle - bestFixed, wins };
  })();
  const yMax = 0.75;
  const y = (v: number) => H - padB - (Math.max(v, 0) / yMax) * (H - padT - padB);
  const barW = 70;
  const bars = [
    { label: 'best fixed arm', v: live.bestFixed, c: 'var(--color-text-secondary)' },
    { label: 'oracle router', v: live.oracle, c: 'var(--color-accent)' },
  ];
  return (
    <div>
      <Slider label="cost weight λ" value={lam} min={0} max={0.5} step={0.005} onChange={setLam} display={fmt(lam, 3)} />
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.78rem', margin: '0 0 0.6rem' }}>
        <input type="checkbox" checked={degenerate} onChange={(e) => setDegenerate(e.target.checked)} style={{ accentColor: 'var(--color-accent)' }} />
        <span>make one arm uniformly best (the degenerate control)</span>
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="the Jensen gap between the oracle router and the best fixed arm" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        {bars.map((b, i) => {
          const x = padL + 30 + i * (barW + 46);
          return (
            <g key={b.label}>
              <rect x={x} y={y(b.v)} width={barW} height={H - padB - y(b.v)} fill={b.c} opacity={0.85} rx={2} />
              <text x={x + barW / 2} y={y(b.v) - 5} textAnchor="middle" fontSize="10" fill="var(--color-text)" fontWeight="600">{fmt(b.v, 4)}</text>
              <text x={x + barW / 2} y={H - padB + 14} textAnchor="middle" fontSize="9" fill="var(--color-text-secondary)">{b.label}</text>
            </g>
          );
        })}
        {/* the gap itself */}
        <line x1={padL + 30 + barW + 23} y1={y(live.bestFixed)} x2={padL + 30 + barW + 23} y2={y(live.oracle)}
          stroke="var(--color-badge-amber-text)" strokeWidth={2} />
        <text x={padL + 30 + barW + 30} y={(y(live.bestFixed) + y(live.oracle)) / 2 + 3} fontSize="10" fill="var(--color-badge-amber-text)" fontWeight="600">
          gap {signed(live.gap, 4)}
        </text>
        <text x={W - padR} y={padT + 6} textAnchor="end" fontSize="9" fill="var(--color-text-secondary)">
          per-query winner: {ARMS.map((a, i) => `${ARM_LABEL[a]} ${live.wins[i]}`).join(' · ')}
        </text>
      </svg>
      <Row>
        <Readout label="E[max U]  (oracle)" value={fmt(live.oracle, 4)} accent />
        <Readout label="max E[U]  (best fixed)" value={fmt(live.bestFixed, 4)} />
        <Readout label="Jensen gap" value={signed(live.gap, 4)} warn={live.gap < 1e-9} />
        <Readout label="ordering varies?" value={live.wins.filter((w) => w > 0).length > 1 ? 'yes' : 'no'} />
      </Row>
      <Note>
        Both bars are recomputed in the browser from the baked per-query quality matrix, so the whole
        decomposition is live. Tick the control and every arm gets the same per-query signal plus a fixed
        offset: the argmax can no longer move, the gap goes to exactly zero, and routing becomes
        worthless <em>however good the classifier is</em>. That is the theorem — the gap is not about
        prediction, it is about whether the arms ever change places.
      </Note>
    </div>
  );
}

// ===== Panel C — the three nested frontiers ========================================================
function FrontierPanel() {
  const W = 560, H = 280, padL = 52, padR = 18, padT = 16, padB = 40;
  const asc = (pts: number[][]) => [...pts].sort((a, b) => a[0] - b[0]);
  const all = [...ROUTER_FRONTIER, ...ORACLE_FRONTIER, ...FIXED_HULL];
  const cMax = Math.max(...all.map((p) => p[0])) * 1.05;
  const qLo = Math.min(...all.map((p) => p[1])) * 0.97;
  const qHi = Math.max(...all.map((p) => p[1])) * 1.03;
  const x = (c: number) => padL + (c / cMax) * (W - padL - padR);
  const y = (q: number) => H - padB - ((q - qLo) / (qHi - qLo)) * (H - padT - padB);
  const path = (pts: number[][]) => asc(pts).map((p, i) => `${i ? 'L' : 'M'}${x(p[0])},${y(p[1])}`).join(' ');
  const curves = [
    { pts: FIXED_HULL, c: 'var(--color-text-secondary)', label: 'fixed arms (no features)', dash: '4 3' },
    { pts: ROUTER_FRONTIER, c: 'var(--color-accent)', label: 'router (held out)', dash: undefined },
    { pts: ORACLE_FRONTIER, c: 'var(--color-badge-amber-text)', label: 'oracle', dash: '2 2' },
  ];
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="three nested cost-quality frontiers: fixed arms, the achievable router, and the oracle" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        {curves.map((cv) => (
          <g key={cv.label}>
            <path d={path(cv.pts)} fill="none" stroke={cv.c} strokeWidth={1.8} strokeDasharray={cv.dash} />
            {asc(cv.pts).map((p, i) => <circle key={i} cx={x(p[0])} cy={y(p[1])} r={2} fill={cv.c} />)}
          </g>
        ))}
        {FIXED_ARM_POINTS.map((p, i) => (
          <g key={i}>
            <circle cx={x(p[0])} cy={y(p[1])} r={4} fill="none" stroke="var(--color-text)" strokeWidth={1.2} />
            <text x={x(p[0]) + 6} y={y(p[1]) - 6} fontSize="8.5" fill="var(--color-text-secondary)">{ARM_LABEL[ARMS[i]]}</text>
          </g>
        ))}
        <text x={(W - padL) / 2 + padL / 2} y={H - 8} textAnchor="middle" fontSize="9.5" fill="var(--color-text-secondary)">mean retrieval cost per query</text>
        <text x={12} y={padT + 60} fontSize="9.5" fill="var(--color-text-secondary)" transform={`rotate(-90 12 ${padT + 60})`}>mean answer quality</text>
      </svg>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', fontSize: '0.72rem', marginTop: '0.3rem' }}>
        {curves.map((cv) => (
          <span key={cv.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ width: 16, height: 0, borderTop: `2px ${cv.dash ? 'dashed' : 'solid'} ${cv.c}`, display: 'inline-block' }} />
            {cv.label}
          </span>
        ))}
      </div>
      <Note>
        All three curves are scored on the <strong>{N_TEST} held-out queries</strong>; the quality model
        was fitted on the other half, and its ridge chosen by cross-validation inside that half.
        Evaluating the router on the queries that fitted it would inflate precisely the curve this panel
        is about. Read the vertical gaps: at matched cost the oracle is what perfect foresight buys, and
        the distance down to the solid line is what estimation error costs.
      </Note>
    </div>
  );
}

// ===== Panel D — what it actually wins =============================================================
function GainPanel() {
  const W = 560, H = 250, padL = 52, padR = 100, padT = 18, padB = 34;
  const rows = ROUTER_REPORT;
  const rowH = (H - padT - padB) / rows.length;
  const vMax = Math.max(...rows.flatMap((r) => [r.inSample, r.heldOut, r.oracle])) * 1.15;
  const vMin = Math.min(0, ...rows.map((r) => r.heldOut)) * 1.3;
  const x = (v: number) => padL + ((v - vMin) / (vMax - vMin)) * (W - padL - padR);
  const zero = x(0);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="in-sample against held-out routing gain at four operating points, with the oracle gap" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {rows.map((r, i) => {
          const y0 = padT + i * rowH;
          const bars = [
            { v: r.oracle, c: 'var(--color-badge-amber-text)', o: 0.35 },
            { v: r.inSample, c: 'var(--color-text-secondary)', o: 0.75 },
            { v: r.heldOut, c: r.heldOut < 0 ? 'var(--color-badge-red-text)' : 'var(--color-accent)', o: 0.95 },
          ];
          return (
            <g key={r.lam}>
              <text x={padL - 8} y={y0 + rowH / 2 + 3} textAnchor="end" fontSize="10" fill="var(--color-text)">λ = {r.lam}</text>
              {bars.map((b, bi) => {
                const h = (rowH - 12) / 3;
                const yy = y0 + 4 + bi * h;
                const xa = Math.min(zero, x(b.v)), xb = Math.max(zero, x(b.v));
                return <rect key={bi} x={xa} y={yy} width={Math.max(xb - xa, 1)} height={h - 2} fill={b.c} opacity={b.o} rx={1} />;
              })}
              <text x={W - padR + 6} y={y0 + rowH / 2 - 2} fontSize="8.5" fill="var(--color-text-secondary)">
                held out {signed(r.heldOut, 3)}
              </text>
              <text x={W - padR + 6} y={y0 + rowH / 2 + 9} fontSize="8.5"
                fill={r.p < 0.05 ? 'var(--color-badge-green-text)' : 'var(--color-text-secondary)'}>
                p = {fmt(r.p, 3)}{r.p < 0.05 ? ' ✓' : ''}
              </text>
            </g>
          );
        })}
        <line x1={zero} y1={padT} x2={zero} y2={H - padB} stroke="var(--color-border)" strokeWidth={1.2} />
        <text x={zero} y={H - padB + 14} textAnchor="middle" fontSize="9" fill="var(--color-text-secondary)">0</text>
      </svg>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', fontSize: '0.72rem', marginTop: '0.3rem' }}>
        {[['oracle gap', 'var(--color-badge-amber-text)'], ['in-sample', 'var(--color-text-secondary)'], ['held out', 'var(--color-accent)']].map(([l, c]) => (
          <span key={l} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ width: 10, height: 10, background: c, display: 'inline-block', borderRadius: 2 }} />{l}
          </span>
        ))}
      </div>
      <Note>
        In-sample exceeds held-out at <em>every</em> operating point, and at λ = {LAM_ROUTER_LOSES} the
        realizable router is <strong>worse than always retrieving once</strong> even though the oracle
        gap there is {signed(ROUTER_REPORT[2].oracle, 3)} — a gap that exists is not a gap you can
        collect. Only λ = {LAM_HEADLINE} shows a gain a paired test can distinguish from zero at
        n = {N_TEST}. The predictor's calibration error is {fmt(ROUTER_REPORT[3].ece, 2)}, which is the
        reason: a router is no better than the calibration of the numbers it compares.
      </Note>
    </div>
  );
}

type Panel = 'arms' | 'gap' | 'frontier' | 'gain';
const TEX: Record<Panel, string> = {
  arms: '\\pi(x) \\in \\{\\text{none},\\ \\text{single},\\ \\text{iterative}\\}, \\qquad Q_a(x) = p\\big(a^\\star \\mid q,\\ \\mathrm{ctx}_a(x)\\big)',
  gap: '\\underbrace{\\mathbb{E}\\big[\\max_a U_a\\big]}_{\\text{oracle}} - \\underbrace{\\max_a \\mathbb{E}\\big[U_a\\big]}_{\\text{best fixed arm}} \\;\\ge\\; 0, \\qquad U_a = Q_a - \\lambda c_a',
  frontier: '\\mathcal{A} = \\mathrm{conv}\\big\\{(\\mathbb{E}[c_{\\pi}],\\ \\mathbb{E}[Q_{\\pi}]) : \\pi = \\pi_\\lambda,\\ \\lambda \\ge 0\\big\\}',
  gain: '\\hat\\pi(x) = \\arg\\max_a \\big\\{ \\widehat{\\mathbb{E}}[Q_a \\mid \\phi(x)] - \\lambda c_a \\big\\}, \\qquad \\text{excess} \\le \\textstyle\\sum_a \\big\\| \\hat Q_a - Q_a \\big\\|_1',
};

export default memo(function AdaptiveRoutingLaboratory() {
  const [panel, setPanel] = useState<Panel>('arms');
  const [lam, setLam] = useState(LAM_HEADLINE);
  const [degenerate, setDegenerate] = useState(false);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="adaptive-retrieval-routing" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'arms')} onClick={() => setPanel('arms')}>A · the three arms</button>
        <button type="button" style={pill(panel === 'gap')} onClick={() => setPanel('gap')}>B · the Jensen gap</button>
        <button type="button" style={pill(panel === 'frontier')} onClick={() => setPanel('frontier')}>C · the frontier</button>
        <button type="button" style={pill(panel === 'gain')} onClick={() => setPanel('gain')}>D · what it wins</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'arms' && <ArmsPanel />}
      {panel === 'gap' && <GapPanel lam={lam} setLam={setLam} degenerate={degenerate} setDegenerate={setDegenerate} />}
      {panel === 'frontier' && <FrontierPanel />}
      {panel === 'gain' && <GainPanel />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_QUERIES} synthetic finance queries in three classes over one passage set, {N_TEST} held out.
        Arm costs are {ARM_COST.none} / {ARM_COST.single} / {ARM_COST.iterative} retrievals; the spread is
        load-bearing, since with costs close together the frontier collapses to a point. Every arm is the
        same answer model reading different evidence, so a quality difference is attributable to the
        strategy alone. Numbers mirror <code>adaptive_retrieval_routing.py</code>; Panel B recomputes the
        entire Jensen decomposition live from the baked per-query matrix. The corpus, the features, and the
        arm costs were all chosen against this corpus — the transferable claim is the shape of the
        argument, not these decimals.
      </p>
    </div>
  );
});
