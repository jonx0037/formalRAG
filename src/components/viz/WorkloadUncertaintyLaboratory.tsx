import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import katex from 'katex';

/**
 * Workload Uncertainty Laboratory — four panels for the successor to `rag-architecture-pareto`.
 * That topic read the architecture off a partition of the workload simplex, treating the workload
 * as known. It is not known; it is estimated, and a partition punishes estimation error at its
 * boundaries.
 *   A. The three rules across the boundary. Plug-in and Bayes recompute LIVE and EXACTLY from the
 *      utility kernel; the hedge is read from a baked grid. That split is not an implementation
 *      detail — it is the topic's own result, since utility is linear in w and the two point
 *      rules therefore have closed forms while the hedge does not.
 *   B. Shrinkage. The Dirichlet posterior mean in closed form, live: the entire gap between
 *      plug-in and Bayes, vanishing like 1/n.
 *   C. How much traffic is enough. Regret decay at a boundary against decay in a cell interior —
 *      the interior costs nothing past n = 20, the boundary is still paying at n = 5120.
 *   D. What the hedge costs. Realized regret either side of the boundary: hedging is a bet on
 *      centrality, and the bet loses on the far side.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): this block is EMITTED from viz_constants() in
 * notebooks/rag-architecture-workload-uncertainty/rag_architecture_workload_uncertainty.py, and
 * test_laboratory_constants_match_the_module parses this file back and compares it to a fresh
 * bake. Only Monte-Carlo results are baked — the hedge grid and the realized-regret table.
 * Everything closed-form (plug-in, Bayes, the posterior mean, every utility) is recomputed live
 * in TypeScript from KERNEL_M and KERNEL_CONST, so it cannot drift.
 */

// --- emitted from rag_architecture_workload_uncertainty.py viz_constants() ---
const ARMS = ["naive","hybrid","hyde","corrective","graph","agentic"] as const;
const CONDITIONS = ["on","partial","off","noisy","bridge","global"] as const;
const KERNEL_M: number[][] = [[0.997,0.347,0.097,-0.003,-0.003,0.0617],[0.791,0.991,0.391,0.291,-0.009,0.061],[0.777,0.077,0.677,-0.123,-0.123,-0.1207],[0.9646,0.933,0.266,0.316,-0.0354,0.6562],[0.7997,0.4497,0.1997,-0.0003,-0.0003,0.9992],[0.0609,0.051,0.001,0.051,0.7609,-0.0297]];
const KERNEL_CONST: number[] = [-0.0,-0.0,-0.0,-0.0,-9e-06,-0.0];
const N_OBSERVED: number[] = [10,20,40,80,160,320,640,1280,2560,5120];
const GRID_FRACS: number[] = [0.0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1.0];
const MINIMAX_GRID: number[][] = [[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,5,5,5,5,5,5],[3,3,5,5,5,5,5,5,5,5],[3,3,5,5,5,5,5,5,5,5],[3,5,5,5,5,5,5,5,5,5],[3,5,5,5,5,5,5,5,5,5],[3,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5,5]];
const DECAY_BOUNDARY: number[][] = [[10,0.1432],[20,0.1397],[40,0.1176],[80,0.1031],[160,0.0767],[320,0.0593],[640,0.0446],[1280,0.032],[2560,0.0231],[5120,0.016]];
const DECAY_INTERIOR: number[][] = [[10,0.0779],[20,0.0],[40,0.0],[80,0.0],[160,0.0],[320,0.0],[640,0.0],[1280,0.0],[2560,0.0],[5120,0.0]];
const HALVING: number[] = [0.9755,0.8419,0.8766,0.7437,0.7734,0.7521,0.7185,0.7194,0.6937];
const AGREEMENT_N: Record<string, number|null> = {"boundary":5120,"interior":10,"local_only":320,"uniform":10};
const COST_OF_HEDGING = [{"frac":0.34,"plug_in":0.01677,"bayes":0.00412,"minimax":0.00412,"hedge_worse":false},{"frac":0.4,"plug_in":0.01693,"bayes":0.00794,"minimax":0.00763,"hedge_worse":false},{"frac":0.43,"plug_in":0.00165,"bayes":0.00046,"minimax":0.00017,"hedge_worse":false},{"frac":0.46,"plug_in":0.01474,"bayes":0.02227,"minimax":0.02203,"hedge_worse":true},{"frac":0.52,"plug_in":0.01604,"bayes":0.03029,"minimax":0.03005,"hedge_worse":true}];
const REALIZED = {"boundary":{"plug_in":0.00165,"bayes":0.00046,"minimax":0.00017},"interior":{"plug_in":0.0,"bayes":0.0,"minimax":0.0}};
const PRIOR_ALPHA = 1.0, REGRET_Q = 0.95, DRAWS = 2000;
const N_HEADLINE_OBS = 40, DISAGREEMENT_RATE = 0.0767;
const LAM = 3e-05, RHO = 1000.0;

const ARM_COLOR: Record<string, string> = {
  naive: 'var(--color-text-secondary)', hybrid: 'var(--color-accent)',
  hyde: 'var(--color-badge-red-text)', corrective: 'var(--color-definition-border)',
  graph: 'var(--color-theorem-border)', agentic: 'var(--color-text)',
};
const fmt = (x: number, n = 3) => (Number.isFinite(x) ? x.toFixed(n) : '∞');
const pill = (a: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${a ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: a ? 'var(--color-accent)' : 'transparent', color: a ? 'var(--color-bg)' : 'var(--color-text)',
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
      <span style={{ minWidth: '16rem' }}>{label} = <strong>{display}</strong></span>
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

// ===== the live recomputation — exact, because utility is linear in w =========================
/** The bridge-share family the predecessor's crossover sits on. */
function bridgeFamily(frac: number): number[] {
  const rest = (1 - frac) / (CONDITIONS.length - 1);
  return CONDITIONS.map((c) => (c === 'bridge' ? frac : rest));
}
/** U(a, w) = (M w)_a + const_a — the imported utility, written out as the affine map it is. */
function utilities(w: number[]): number[] {
  return KERNEL_M.map((row, a) => row.reduce((s, v, k) => s + v * w[k], 0) + KERNEL_CONST[a]);
}
const argmax = (v: number[]) => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);
/** The Dirichlet posterior MEAN, closed form: (n w_hat + alpha) / (n + C alpha). */
function posteriorMean(wHat: number[], n: number): number[] {
  const conc = wHat.map((x) => n * x + PRIOR_ALPHA);
  const tot = conc.reduce((a, b) => a + b, 0);
  return conc.map((x) => x / tot);
}
const rulePlugIn = (wHat: number[]) => argmax(utilities(wHat));
const ruleBayes = (wHat: number[], n: number) => argmax(utilities(posteriorMean(wHat, n)));
/** The hedge is the one rule with no closed form, so it is the one rule read from the bake. */
function ruleMinimax(frac: number, nIdx: number): number {
  let gi = 0, best = Infinity;
  GRID_FRACS.forEach((f, i) => { const d = Math.abs(f - frac); if (d < best) { best = d; gi = i; } });
  return MINIMAX_GRID[gi][nIdx];
}

// ===== Panel A — the three rules across the boundary ==========================================
function RulesPanel({ frac, setFrac, nIdx, setNIdx }: {
  frac: number; setFrac: (v: number) => void; nIdx: number; setNIdx: (v: number) => void;
}) {
  const n = N_OBSERVED[nIdx];
  const snapped = GRID_FRACS.reduce((b, f) => (Math.abs(f - frac) < Math.abs(b - frac) ? f : b), GRID_FRACS[0]);
  const w = useMemo(() => bridgeFamily(snapped), [snapped]);
  const pi = rulePlugIn(w), by = ruleBayes(w, n), mm = ruleMinimax(snapped, nIdx);
  const agree = pi === mm;
  const W = 600, H = 190, padL = 74, padT = 26, rowH = 34;
  const X = (f: number) => padL + f * (W - padL - 120);

  const rows: [string, (f: number) => number, number][] = [
    ['plug-in', (f) => rulePlugIn(bridgeFamily(f)), pi],
    ['Bayes', (f) => ruleBayes(bridgeFamily(f), n), by],
    ['minimax', (f) => ruleMinimax(f, nIdx), mm],
  ];
  return (
    <div>
      <Slider label="share of traffic that is a bridge question" value={frac} min={0} max={1} step={0.025}
        onChange={setFrac} display={`${(snapped * 100).toFixed(1)}%`} />
      <Slider label="queries observed (n)" value={nIdx} min={0} max={N_OBSERVED.length - 1} step={1}
        onChange={setNIdx} display={String(n)} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Which architecture each decision rule selects, across the workload boundary">
        {rows.map(([label, fn, cur], ri) => (
          <g key={label}>
            <text x={padL - 10} y={padT + ri * rowH + 15} textAnchor="end" fontSize="11"
              fontFamily="var(--font-sans)" fill="var(--color-text)">{label}</text>
            {GRID_FRACS.map((f, i) => {
              const a = fn(f);
              const wdt = (W - padL - 120) / GRID_FRACS.length;
              return (
                <rect key={i} x={X(f)} y={padT + ri * rowH} width={wdt + 0.6} height={rowH - 10} rx="1.5"
                  fill={ARM_COLOR[ARMS[a]]} fillOpacity={0.8} />
              );
            })}
            <text x={W - 112} y={padT + ri * rowH + 15} fontSize="10.5" fontFamily="var(--font-sans)"
              fill={ARM_COLOR[ARMS[cur]]} fontWeight={600}>{ARMS[cur]}</text>
          </g>
        ))}
        <line x1={X(snapped)} y1={padT - 6} x2={X(snapped)} y2={padT + 3 * rowH - 4}
          stroke="var(--color-accent)" strokeWidth="1.6" />
        <text x={padL} y={H - 10} fontSize="9.5" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          0% bridge
        </text>
        <text x={W - 124} y={H - 10} textAnchor="end" fontSize="9.5" fill="var(--color-text-secondary)"
          fontFamily="var(--font-sans)">100% bridge</text>
      </svg>
      <Row>
        <Readout label="plug-in" value={ARMS[pi]} />
        <Readout label="Bayes" value={ARMS[by]} />
        <Readout label="minimax (the hedge)" value={ARMS[mm]} accent={!agree} />
        <Readout label="verdict" value={agree ? 'all agree' : 'the hedge differs'} warn={!agree} />
      </Row>
      <Note>
        Plug-in and Bayes are recomputed here <em>exactly</em>, live, from the utility kernel — because
        utility is linear in the workload, both have closed forms. The hedge does not, so it is the one
        rule read from a precomputed grid. That asymmetry is the topic's own result showing up in its
        laboratory. Slide across the boundary and the three bands part company only in a narrow region;
        slide <strong>n</strong> and the disagreement shrinks everywhere except on the boundary itself.
      </Note>
    </div>
  );
}

// ===== Panel B — shrinkage, the whole of Bayes ================================================
function ShrinkagePanel({ nIdx, setNIdx, frac, setFrac }: {
  nIdx: number; setNIdx: (v: number) => void; frac: number; setFrac: (v: number) => void;
}) {
  const n = N_OBSERVED[nIdx];
  const bi = CONDITIONS.indexOf('bridge');
  const uniform = 1 / CONDITIONS.length;
  const curve = useMemo(
    () => N_OBSERVED.map((m) => [m, posteriorMean(bridgeFamily(frac), m)[bi]] as const), [frac, bi]);
  const here = posteriorMean(bridgeFamily(frac), n)[bi];
  const W = 600, H = 210, padL = 52, padB = 38, padT = 14, padR = 118;
  const X = (m: number) => padL + (Math.log10(m) - 1) / (Math.log10(N_OBSERVED[N_OBSERVED.length - 1]) - 1) * (W - padL - padR);
  const lo = Math.min(uniform, frac) - 0.06, hi = Math.max(uniform, frac) + 0.06;
  const Y = (y: number) => H - padB - ((y - lo) / (hi - lo)) * (H - padT - padB);

  return (
    <div>
      <Slider label="the estimate ŵ (bridge share)" value={frac} min={0} max={1} step={0.025}
        onChange={setFrac} display={`${(frac * 100).toFixed(1)}%`} />
      <Slider label="queries observed (n)" value={nIdx} min={0} max={N_OBSERVED.length - 1} step={1}
        onChange={setNIdx} display={String(n)} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="The Dirichlet posterior mean shrinking toward the prior as observations accumulate">
        <line x1={padL} y1={Y(frac)} x2={W - padR} y2={Y(frac)} stroke="var(--color-accent)"
          strokeWidth="1.4" strokeDasharray="5 3" />
        <text x={W - padR + 6} y={Y(frac) + 3} fontSize="9.5" fill="var(--color-accent)"
          fontFamily="var(--font-sans)">the estimate ŵ</text>
        <line x1={padL} y1={Y(uniform)} x2={W - padR} y2={Y(uniform)} stroke="var(--color-border)" strokeWidth="1.2" />
        <text x={W - padR + 6} y={Y(uniform) + 3} fontSize="9.5" fill="var(--color-text-secondary)"
          fontFamily="var(--font-sans)">the prior ({fmt(uniform, 2)})</text>
        <path d={curve.map(([m, y], i) => `${i ? 'L' : 'M'}${X(m)},${Y(y)}`).join(' ')}
          fill="none" stroke="var(--color-text)" strokeWidth="1.8" />
        <line x1={X(n)} y1={padT} x2={X(n)} y2={H - padB} stroke="var(--color-accent)" strokeWidth="1.1" />
        <circle cx={X(n)} cy={Y(here)} r="4.5" fill="var(--color-accent)" />
        <text x={padL} y={H - 10} fontSize="9.5" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          queries observed (log) &rarr;
        </text>
      </svg>
      <Row>
        <Readout label="estimate ŵ" value={fmt(frac, 4)} />
        <Readout label="posterior mean (= Bayes)" value={fmt(here, 4)} accent />
        <Readout label="shrinkage" value={fmt(Math.abs(frac - here), 4)} warn={Math.abs(frac - here) > 0.02} />
        <Readout label="prior weight" value={fmt(CONDITIONS.length * PRIOR_ALPHA / (n + CONDITIONS.length * PRIOR_ALPHA), 3)} />
      </Row>
      <Note>
        Bayes is not a third rule. Utility is linear in the workload, so the expected utility of an arm
        over the posterior equals its utility <em>at the posterior mean</em> — exactly, not approximately.
        The Bayes rule is therefore the plug-in rule applied to a shrunk estimate, and this curve is the
        entire difference between them. It vanishes like <code>1/n</code>: the prior's weight is{' '}
        {CONDITIONS.length}&alpha;/(n + {CONDITIONS.length}&alpha;), which is most of the answer at n = 10
        and almost none of it by n = 1280.
      </Note>
    </div>
  );
}

// ===== Panel C — how much traffic is enough ===================================================
function DataPanel({ nIdx, setNIdx }: { nIdx: number; setNIdx: (v: number) => void }) {
  const n = N_OBSERVED[nIdx];
  const W = 600, H = 220, padL = 54, padB = 40, padT = 14, padR = 128;
  const maxR = Math.max(...DECAY_BOUNDARY.map((d) => d[1]), 0.02);
  const X = (m: number) => padL + (Math.log10(m) - 1) / (Math.log10(N_OBSERVED[N_OBSERVED.length - 1]) - 1) * (W - padL - padR);
  const Y = (r: number) => H - padB - (r / maxR) * (H - padT - padB);
  const at = (curve: number[][]) => curve.find(([m]) => m === n)?.[1] ?? 0;

  return (
    <div>
      <Slider label="queries observed (n)" value={nIdx} min={0} max={N_OBSERVED.length - 1} step={1}
        onChange={setNIdx} display={String(n)} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Worst-case regret against observation count, at a boundary and in a cell interior">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={padL} y1={Y(t * maxR)} x2={W - padR} y2={Y(t * maxR)} stroke="var(--color-border)" strokeWidth="0.5" />
            <text x={padL - 6} y={Y(t * maxR) + 3} textAnchor="end" fontSize="9"
              fill="var(--color-text-secondary)">{fmt(t * maxR, 2)}</text>
          </g>
        ))}
        {([[DECAY_BOUNDARY, 'var(--color-badge-red-text)', 'on the boundary'],
           [DECAY_INTERIOR, 'var(--color-accent)', 'inside a cell']] as const).map(([curve, col, lab]) => (
          <g key={lab}>
            <path d={(curve as number[][]).map(([m, r], i) => `${i ? 'L' : 'M'}${X(m)},${Y(r)}`).join(' ')}
              fill="none" stroke={col} strokeWidth="1.9" />
            <text x={W - padR + 6} y={Y((curve as number[][])[2][1]) + 3} fontSize="9.5" fill={col}
              fontFamily="var(--font-sans)">{lab}</text>
          </g>
        ))}
        <line x1={X(n)} y1={padT} x2={X(n)} y2={H - padB} stroke="var(--color-accent)" strokeWidth="1.1" />
        <text x={padL} y={H - 10} fontSize="9.5" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          queries observed (log) &rarr;
        </text>
        <text x={16} y={padT + 52} fontSize="9.5" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)"
          transform={`rotate(-90 16 ${padT + 52})`}>worst-case regret &rarr;</text>
      </svg>
      <Row>
        <Readout label="on the boundary" value={fmt(at(DECAY_BOUNDARY), 4)} warn />
        <Readout label="inside a cell" value={fmt(at(DECAY_INTERIOR), 4)} accent />
        <Readout label="agree from n =" value={String(AGREEMENT_N.interior ?? '—')} />
        <Readout label="...on the boundary" value={String(AGREEMENT_N.boundary ?? 'never')} warn />
      </Row>
      <Note>
        Inside a cell the uncertainty costs nothing past about twenty queries — the recommendation is
        simply not sensitive to where in the cell you are. On a boundary it is still being paid for at{' '}
        {AGREEMENT_N.boundary}, which is {Math.round((AGREEMENT_N.boundary ?? 1) / (AGREEMENT_N.interior || 1))}&times;
        more traffic. The halving ratios are {HALVING.slice(-3).map((r) => fmt(r, 2)).join(', ')} against the{' '}
        {fmt(1 / Math.SQRT2, 3)} a clean <code>1/&radic;n</code> law would give: they approach it from above
        rather than sitting on it, so the honest claim is the monotone decay and the tail, not the law.
      </Note>
    </div>
  );
}

// ===== Panel D — what the hedge costs =========================================================
function CostPanel() {
  const W = 600, H = 210, padL = 60, padB = 42, padT = 16, padR = 108;
  const rows = COST_OF_HEDGING;
  const maxV = Math.max(...rows.flatMap((r) => [r.plug_in, r.bayes, r.minimax])) * 1.15;
  const bw = (W - padL - padR) / rows.length;
  const Y = (v: number) => H - padB - (v / maxV) * (H - padT - padB);
  const series: [string, 'plug_in' | 'minimax', string][] = [
    ['plug-in', 'plug_in', 'var(--color-text-secondary)'],
    ['minimax', 'minimax', 'var(--color-accent)'],
  ];
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Realized regret of each decision rule either side of the workload boundary">
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        {rows.map((r, i) => (
          <g key={r.frac}>
            {series.map(([, key, col], si) => (
              <rect key={key} x={padL + i * bw + 6 + si * (bw - 16) / 2} y={Y(r[key])}
                width={(bw - 16) / 2 - 2} height={H - padB - Y(r[key])} rx="2"
                fill={col} fillOpacity={0.85} />
            ))}
            <text x={padL + i * bw + bw / 2} y={H - padB + 14} textAnchor="middle" fontSize="9.5"
              fill={r.hedge_worse ? 'var(--color-badge-red-text)' : 'var(--color-text-secondary)'}
              fontFamily="var(--font-sans)">{fmt(r.frac, 2)}</text>
            {r.hedge_worse && (
              <text x={padL + i * bw + bw / 2} y={H - padB + 26} textAnchor="middle" fontSize="8.5"
                fill="var(--color-badge-red-text)" fontFamily="var(--font-sans)">hedge loses</text>
            )}
          </g>
        ))}
        <line x1={padL + 3 * bw} y1={padT} x2={padL + 3 * bw} y2={H - padB} stroke="var(--color-accent)"
          strokeWidth="1.2" strokeDasharray="4 3" />
        {series.map(([lab, , col], si) => (
          <g key={lab}>
            <rect x={W - padR + 8} y={padT + si * 18} width={10} height={10} rx="2" fill={col} fillOpacity={0.85} />
            <text x={W - padR + 23} y={padT + si * 18 + 9} fontSize="9.5" fontFamily="var(--font-sans)"
              fill="var(--color-text)">{lab}</text>
          </g>
        ))}
        <text x={padL} y={H - 6} fontSize="9.5" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          bridge share of the TRUE workload &rarr;
        </text>
      </svg>
      <Row>
        <Readout label="boundary, plug-in" value={fmt(REALIZED.boundary.plug_in, 5)} />
        <Readout label="boundary, hedge" value={fmt(REALIZED.boundary.minimax, 5)} accent />
        <Readout label="inside a cell" value={`${fmt(REALIZED.interior.plug_in, 5)} either way`} />
        <Readout label="hedge loses on" value={`${COST_OF_HEDGING.filter((r) => r.hedge_worse).length} of ${COST_OF_HEDGING.length}`} warn />
      </Row>
      <Note>
        Realized regret against the truth, after observing {N_HEADLINE_OBS} queries. The hedge is not free
        insurance — it is a bet that the traffic is closer to the middle of the simplex than the estimate
        says, because that is what shrinking toward the prior means. Below the boundary the bet pays and
        the hedge costs a fraction of what the plug-in rule does. Above it, where the truth genuinely sits
        in a narrow cell, the plug-in rule was correctly confident and the hedge drags it back out; the
        worst loss there is <em>larger</em> than the best gain below. Over workloads drawn without regard
        to where the boundaries are, the two rules differ on {fmt(DISAGREEMENT_RATE * 100, 1)}% of them.
      </Note>
    </div>
  );
}

type Panel = 'rules' | 'shrinkage' | 'data' | 'cost';
const TEX: Record<Panel, string> = {
  rules: '\\hat a_{\\text{plug}} = \\arg\\max_a U(a, \\hat w), \\qquad \\hat a_{\\text{mm}} = \\arg\\min_a \\ \\max_{w} \\big[ U(a^\\ast(w), w) - U(a, w) \\big]',
  shrinkage: '\\mathbb{E}_{w \\sim \\pi}\\big[ U(a, w) \\big] = U\\big(a, \\mathbb{E}_\\pi[w]\\big), \\qquad \\mathbb{E}_\\pi[w] = \\frac{n \\hat w + \\alpha}{n + C\\alpha}',
  data: 'R_q(n) = \\mathrm{Quantile}_q \\Big( \\max_b U(b, w) - U(a, w) \\Big), \\quad w \\sim \\mathrm{Dir}(n\\hat w + \\alpha)',
  cost: '\\mathbb{E}\\big[ \\text{regret} \\big] = \\mathbb{E}_{\\hat w \\mid w^\\ast} \\Big[ \\max_b U(b, w^\\ast) - U\\big(\\text{rule}(\\hat w), w^\\ast\\big) \\Big]',
};

export default memo(function WorkloadUncertaintyLaboratory() {
  const [panel, setPanel] = useState<Panel>('rules');
  const [frac, setFrac] = useState(0.45);
  const [nIdx, setNIdx] = useState(N_OBSERVED.indexOf(N_HEADLINE_OBS) >= 0 ? N_OBSERVED.indexOf(N_HEADLINE_OBS) : 2);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="workload-uncertainty" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'rules')} onClick={() => setPanel('rules')}>A · three rules</button>
        <button type="button" style={pill(panel === 'shrinkage')} onClick={() => setPanel('shrinkage')}>B · shrinkage</button>
        <button type="button" style={pill(panel === 'data')} onClick={() => setPanel('data')}>C · how much traffic</button>
        <button type="button" style={pill(panel === 'cost')} onClick={() => setPanel('cost')}>D · what the hedge costs</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'rules' && <RulesPanel frac={frac} setFrac={setFrac} nIdx={nIdx} setNIdx={setNIdx} />}
      {panel === 'shrinkage' && <ShrinkagePanel nIdx={nIdx} setNIdx={setNIdx} frac={frac} setFrac={setFrac} />}
      {panel === 'data' && <DataPanel nIdx={nIdx} setNIdx={setNIdx} />}
      {panel === 'cost' && <CostPanel />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        The six architectures and their quality and costs arrive unchanged from the two predecessor topics;
        nothing is re-measured here. Prices are fixed at &lambda; = {LAM}, &rho; = {RHO}. The posterior is
        Dirichlet with a uniform prior (&alpha; = {PRIOR_ALPHA}) over the {CONDITIONS.length} kinds of
        question, and worst-case regret is its {REGRET_Q} quantile over {DRAWS} draws rather than a maximum,
        which would grow with the draw count and could not be baked. Plug-in, Bayes and every utility on
        this page are recomputed live and exactly from the utility kernel; only the hedge and the realized
        regret are Monte-Carlo and therefore baked, and both are seeded.
      </p>
    </div>
  );
});
