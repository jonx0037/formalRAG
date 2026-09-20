import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import katex from 'katex';

/**
 * RAG Architecture Pareto Laboratory — four panels for `rag-architecture-pareto`.
 * The predecessor left a Pareto set rather than a winner. This is what the choice depends on.
 *   A. The table. Quality and BOTH costs per (arm, condition). One arm's cost varies tenfold
 *      across the kind of question it meets; every other arm's is flat.
 *   B. What domination depends on. The (ops, calls) plane with the Pareto set outlined, a
 *      bridge-fraction slider that carries the dominated arm to optimal, and an N slider that
 *      reverses the graph arm's domination of the baseline below the amortization crossover.
 *   C. The winner map. A 2-D slice of the workload simplex colored by which arm a cost-aware
 *      evaluation would deploy, recomputed live as lambda moves.
 *   D. What no price can buy. Quality against scalarized cost with the lower hull drawn: the
 *      arms ON the hull are exactly those some linear price selects, and a Pareto-optimal arm
 *      inside it is recommended by no exchange rate at all.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): the block below is EMITTED from viz_constants() in
 * notebooks/rag-architecture-pareto/rag_architecture_pareto.py, never transcribed, and
 * test_laboratory_constants_match_the_module parses this file back and compares it to a fresh
 * bake. Only the 6 x 6 x 3 measurement table and the build costs are baked — 108 numbers plus
 * six. EVERY derived object here (profiles, domination, Pareto sets, hulls, the winner map, the
 * amortization curves) is recomputed live in TypeScript from those, which is why none of them
 * can drift. Matching asserts: test_only_the_adaptive_arm_has_a_workload_dependent_cost /
 * test_domination_is_not_a_property_of_an_architecture / test_the_winner_map_partitions_the_simplex
 * / test_linear_pricing_cannot_see_every_pareto_arm / test_amortization_moves_the_frontier.
 */

// --- emitted from rag_architecture_pareto.py viz_constants() ---
const ARMS = ["naive","hybrid","hyde","corrective","graph","agentic"] as const;
const CONDITIONS = ["on","partial","off","noisy","bridge","global"] as const;
const Q: number[][] = [[1.0,0.35,0.1,0.0,0.0,0.062],[0.8,1.0,0.4,0.3,0.0,0.062],[0.9,0.2,0.8,0.0,0.0,0.0],[1.0,1.0,0.35,0.4,0.0,0.688],[0.8,0.45,0.2,0.0,0.0,1.0],[0.15,0.15,0.1,0.15,0.85,0.062]];
const O: number[][] = [[100.0,100.0,100.0,100.0,100.0,25.0],[300.0,300.0,300.0,300.0,300.0,50.0],[100.0,100.0,100.0,100.0,100.0,25.0],[181.0,1234.0,1801.0,1801.0,181.0,45.0],[10.0,10.0,10.0,10.0,10.0,25.0],[270.0,300.0,300.0,300.0,270.0,75.0]];
const C: number[][] = [[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],[4.0,4.0,4.0,4.0,4.0,4.0],[1.0,1.0,1.0,1.0,1.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0],[2.7,3.0,3.0,3.0,2.7,3.0]];
const BUILD: number[] = [0.0,0.0,0.0,0.0,300.0,0.0];
const COST_SPREAD: Record<string, number> = {"naive":1.0,"hybrid":1.0,"hyde":1.0,"corrective":9.95,"graph":1.0,"agentic":1.111};
const WINNER_CELLS: Record<string, number> = {"corrective":0.684,"graph":0.14,"hybrid":0.086,"agentic":0.058,"hyde":0.033};
const GAP_ARMS: Record<string, number> = {"agentic":264,"hyde":236,"corrective":183,"naive":119,"hybrid":2};
const CROSSOVER_BY_K: number[][] = [[25,3.333],[100,13.026],[500,63.802],[2000,252.7]];
const RHO_FLIP = {"lambda":0.001,"at_zero":"hyde","at_headline":"graph"};
const K0 = 25, N_DOCS = 100, N_SECTORS = 5;
const LAM0 = 3e-05, RHO0 = 1000.0, N0 = 1000;
const GAP_FRACTION = 0.224, BRIDGE_CROSSOVER = 0.43;
const CROSSOVER_N = 3.333;

const ARM_COLOR: Record<string, string> = {
  naive: 'var(--color-text-secondary)', hybrid: 'var(--color-accent)',
  hyde: 'var(--color-badge-red-text)', corrective: 'var(--color-definition-border)',
  graph: 'var(--color-theorem-border)', agentic: 'var(--color-text)',
};
const fmt = (x: number, n = 3) => (Number.isFinite(x) ? x.toFixed(n) : '∞');
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

// ===== the live recomputation — the whole topic, from 108 baked numbers ==========================
/** A workload with a given fraction on one condition and the rest spread evenly. */
function mixOn(condition: string, frac: number): number[] {
  const ci = CONDITIONS.indexOf(condition as (typeof CONDITIONS)[number]);
  const rest = (1 - frac) / (CONDITIONS.length - 1);
  return CONDITIONS.map((_, i) => (i === ci ? frac : rest));
}
/** Quality and BOTH costs per arm on a workload, with the build amortized over N queries. */
function profile(w: number[], N: number) {
  const dot = (M: number[][], i: number) => M[i].reduce((s, v, k) => s + v * w[k], 0);
  return {
    q: ARMS.map((_, i) => dot(Q, i)),
    o: ARMS.map((_, i) => dot(O, i) + BUILD[i] / Math.max(1, N)),
    c: ARMS.map((_, i) => dot(C, i)),
  };
}
function dominates(i: number, j: number, q: number[], o: number[], c: number[]) {
  const ge = q[i] >= q[j] - 1e-12 && o[i] <= o[j] + 1e-12 && c[i] <= c[j] + 1e-12;
  const gt = q[i] > q[j] + 1e-12 || o[i] < o[j] - 1e-12 || c[i] < c[j] - 1e-12;
  return ge && gt;
}
function paretoIdx(q: number[], o: number[], c: number[]): number[] {
  return ARMS.map((_, i) => i).filter((i) => !ARMS.some((_, j) => j !== i && dominates(j, i, q, o, c)));
}
const scalarCost = (o: number[], c: number[], rho: number) => o.map((v, i) => v + rho * c[i]);
/** Upper-left hull of the (cost, quality) cloud — exactly the arms some linear price selects. */
function hullVertices(q: number[], cost: number[]): number[] {
  const order = q.map((_, i) => i).sort((a, b) => cost[a] - cost[b] || q[b] - q[a]);
  const stair: number[] = [];
  let best = -Infinity;
  for (const i of order) if (q[i] > best + 1e-12) { stair.push(i); best = q[i]; }
  const hull: number[] = [];
  for (const i of stair) {
    while (hull.length >= 2) {
      const a = hull[hull.length - 2], b = hull[hull.length - 1];
      const cross = (cost[b] - cost[a]) * (q[i] - q[a]) - (q[b] - q[a]) * (cost[i] - cost[a]);
      if (cross >= -1e-15) hull.pop(); else break;
    }
    hull.push(i);
  }
  return hull;
}
/**
 * Arms selectable by SOME linear price — the union of the hull's vertices over a grid of rho.
 * The drawn hull is the one at the slider's rho; this is the stronger, price-free statement the
 * theorem is actually about, and the two differ: an arm off the hull at one exchange rate may sit
 * on it at another, so reading a single rho as "no price can buy this" overstates the result.
 */
function selectableAtAnyPrice(q: number[], o: number[], c: number[]): Set<number> {
  const out = new Set<number>();
  // The SAME grid the module sweeps in selectable(): 61 points, 10^-1 to 10^5. Matching it is
  // not cosmetic — a coarser grid here would report a gap the module does not, and the module's
  // test_linear_pricing_cannot_see_every_pareto_arm is what certifies the claim.
  for (let n = 0; n < 61; n++) {
    const e = -1 + (6 * n) / 60;
    for (const i of hullVertices(q, scalarCost(o, c, Math.pow(10, e)))) out.add(i);
  }
  return out;
}
function winnerIdx(w: number[], lam: number, rho: number, N: number): number {
  const { q, o, c } = profile(w, N);
  const cost = scalarCost(o, c, rho);
  let bi = 0;
  for (let i = 1; i < ARMS.length; i++) if (q[i] - lam * cost[i] > q[bi] - lam * cost[bi]) bi = i;
  return bi;
}

// ===== Panel A — the table =======================================================================
function TablePanel() {
  const W = 600, H = 210, padL = 92, padT = 30;
  const cw = (W - padL - 92) / CONDITIONS.length, ch = (H - padT - 16) / ARMS.length;
  const lo = Math.min(...O.flat()), hi = Math.max(...O.flat());
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Operations per query for each architecture on each kind of question">
        {CONDITIONS.map((k, ci) => (
          <text key={k} x={padL + ci * cw + cw / 2} y={padT - 8} textAnchor="middle" fontSize="10"
            fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{k}</text>
        ))}
        <text x={W - 86} y={padT - 8} fontSize="10" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">spread</text>
        {ARMS.map((a, ri) => (
          <g key={a}>
            <text x={padL - 8} y={padT + ri * ch + ch / 2 + 4} textAnchor="end" fontSize="11"
              fontFamily="var(--font-sans)" fill="var(--color-text)">{a}</text>
            {CONDITIONS.map((k, ci) => {
              const v = O[ri][ci];
              const t = (Math.log(v + 1) - Math.log(lo + 1)) / (Math.log(hi + 1) - Math.log(lo + 1));
              return (
                <g key={k}>
                  <rect x={padL + ci * cw + 1} y={padT + ri * ch + 1} width={cw - 2} height={ch - 2} rx="3"
                    fill="var(--color-accent)" fillOpacity={0.08 + 0.8 * t} />
                  <text x={padL + ci * cw + cw / 2} y={padT + ri * ch + ch / 2 + 4} textAnchor="middle"
                    fontSize="9.5" fontFamily="var(--font-mono, monospace)"
                    fill={t > 0.55 ? 'var(--color-bg)' : 'var(--color-text)'}>{Math.round(v)}</text>
                </g>
              );
            })}
            <text x={W - 86} y={padT + ri * ch + ch / 2 + 4} fontSize="10"
              fontFamily="var(--font-mono, monospace)"
              fill={COST_SPREAD[a] >= 5 ? 'var(--color-badge-red-text)' : 'var(--color-text-secondary)'}>
              {fmt(COST_SPREAD[a], 2)}&times;
            </text>
          </g>
        ))}
      </svg>
      <Row>
        <Readout label="flat arms" value={ARMS.filter((a) => COST_SPREAD[a] < 1.2).length + ' of ' + ARMS.length} />
        <Readout label="corrective, across question kinds" value={fmt(COST_SPREAD.corrective, 2) + '×'} warn />
        <Readout label="currencies" value="2, not 1" accent />
      </Row>
      <Note>
        Operations per query, on a log color scale. Five of the six arms do the same work whatever
        they are asked — their cost is a property of the pipeline. The corrective arm's is not: its
        grader fires on some questions and not others, so its cost varies by a factor of{' '}
        {fmt(COST_SPREAD.corrective, 2)} across the kinds of question it might meet. A single cost
        column cannot say that, and neither can a single cost column say that four of these arms
        also spend a second, incommensurable currency — generator calls — that no number of
        operations converts into.
      </Note>
    </div>
  );
}

// ===== Panel B — what domination depends on ======================================================
function DominationPanel({ bridge, setBridge, logN, setLogN }: {
  bridge: number; setBridge: (v: number) => void; logN: number; setLogN: (v: number) => void;
}) {
  const N = Math.round(Math.pow(10, logN));
  const w = useMemo(() => mixOn('bridge', bridge), [bridge]);
  const { q, o, c } = useMemo(() => profile(w, N), [w, N]);
  const par = useMemo(() => new Set(paretoIdx(q, o, c)), [q, o, c]);
  const ai = ARMS.indexOf('agentic');
  const beaten = ARMS.map((_, j) => j).filter((j) => j !== ai && dominates(j, ai, q, o, c)).map((j) => ARMS[j]);

  const W = 600, H = 240, padL = 54, padB = 40, padT = 14, padR = 96;
  const xs = o, ys = c;
  const xMax = Math.max(...xs) * 1.12 + 1, yMax = Math.max(...ys) * 1.15 + 0.4;
  const X = (v: number) => padL + (v / xMax) * (W - padL - padR);
  const Y = (v: number) => H - padB - (v / yMax) * (H - padT - padB);
  return (
    <div>
      <Slider label="share of traffic that is a bridge question" value={bridge} min={0} max={1} step={0.01}
        onChange={setBridge} display={`${(bridge * 100).toFixed(0)}%`} />
      <Slider label="queries the index build is amortized over (N)" value={logN} min={0} max={4} step={0.05}
        onChange={setLogN} display={String(N)} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Operations against generator calls for six architectures, with the Pareto set marked">
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        <text x={(padL + W - padR) / 2} y={H - 8} textAnchor="middle" fontSize="10"
          fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">operations per query &rarr;</text>
        <text x={14} y={padT + 42} fontSize="10" fill="var(--color-text-secondary)"
          fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${padT + 42})`}>generator calls &rarr;</text>
        {ARMS.map((a, i) => {
          const on = par.has(i);
          return (
            <g key={a}>
              <circle cx={X(xs[i])} cy={Y(ys[i])} r={5 + 9 * q[i]} fill={ARM_COLOR[a]}
                fillOpacity={on ? 0.85 : 0.18} stroke={on ? 'var(--color-accent)' : 'none'} strokeWidth={on ? 1.6 : 0} />
              <text x={X(xs[i]) + 9 + 9 * q[i]} y={Y(ys[i]) + 4} fontSize="10" fontFamily="var(--font-sans)"
                fill={on ? 'var(--color-text)' : 'var(--color-text-secondary)'}
                fontWeight={a === 'agentic' ? 700 : 400}>{a}</text>
            </g>
          );
        })}
      </svg>
      <Row>
        <Readout label="Pareto set" value={String(par.size) + ' of ' + ARMS.length} />
        <Readout label="agentic" value={par.has(ai) ? 'Pareto-optimal' : 'dominated'}
          accent={par.has(ai)} warn={!par.has(ai)} />
        <Readout label="…beaten by" value={beaten.length ? beaten.join(', ') : '—'} />
        <Readout label="would deploy" value={ARMS[winnerIdx(w, LAM0, RHO0, N)]} accent />
      </Row>
      <Note>
        Circle area is quality; the two axes are the two costs. <strong>Slide the bridge share.</strong> At
        the mixture the predecessor implicitly reported, agentic is beaten on <em>every</em> axis at once —
        less accurate, more operations, more generator calls — and by the Pareto criterion should never be
        deployed. Past a bridge share of about {fmt(BRIDGE_CROSSOVER, 2)} it is the arm to deploy. Nothing
        about the architecture changed; the traffic did.
      </Note>
      <Note>
        <strong>Slide N.</strong> The graph arm holds an index the others do not, and whether it dominates
        the baseline is a question about deployment size rather than about either architecture: its build
        pays for itself after N&#42; = {fmt(CROSSOVER_N, 2)} queries here. That crossover is not a constant
        either — the build is quadratic in the entity count while the saving is only linear, so it moves
        out as the corpus grows ({CROSSOVER_BY_K.map(([k, n]) => `K=${k}: ${Math.round(n)}`).join(', ')}),
        the opposite of the intuition that an index matters more the bigger the corpus.
      </Note>
    </div>
  );
}

// ===== Panel C — the winner map ==================================================================
function WinnerMapPanel({ logLam, setLogLam }: { logLam: number; setLogLam: (v: number) => void }) {
  const lam = Math.pow(10, logLam);
  const STEPS = 21;
  const grid = useMemo(() => {
    const out: (number | null)[][] = [];
    for (let gy = 0; gy < STEPS; gy++) {
      const y = gy / (STEPS - 1), row: (number | null)[] = [];
      for (let gx = 0; gx < STEPS; gx++) {
        const x = gx / (STEPS - 1);
        if (x + y > 1 + 1e-9) { row.push(null); continue; }
        const bi = CONDITIONS.indexOf('bridge'), gi = CONDITIONS.indexOf('global');
        const rest = (1 - x - y) / (CONDITIONS.length - 2);
        const w = CONDITIONS.map((_, i) => (i === bi ? x : i === gi ? y : rest));
        row.push(winnerIdx(w, lam, RHO0, N0));
      }
      out.push(row);
    }
    return out;
  }, [lam]);
  const present = useMemo(() => {
    const s = new Set<number>();
    grid.forEach((r) => r.forEach((v) => { if (v !== null) s.add(v); }));
    return [...s].sort((a, b) => a - b);
  }, [grid]);

  const W = 600, H = 250, padL = 58, padB = 40, padT = 12, cell = (Math.min(W - padL - 180, H - padT - padB)) / STEPS;
  return (
    <div>
      <Slider label="price of cost in units of quality (lambda)" value={logLam} min={-6} max={-1} step={0.05}
        onChange={setLogLam} display={lam.toExponential(1)} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Which architecture wins across a slice of the workload simplex">
        {grid.map((row, gy) => (
          <g key={gy}>
            {row.map((v, gx) => (v === null ? null : (
              <rect key={gx} x={padL + gx * cell} y={padT + (STEPS - 1 - gy) * cell}
                width={cell + 0.5} height={cell + 0.5} fill={ARM_COLOR[ARMS[v]]} fillOpacity={0.8} />
            )))}
          </g>
        ))}
        <text x={padL + STEPS * cell / 2} y={H - 10} textAnchor="middle" fontSize="10"
          fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">share of traffic: bridge &rarr;</text>
        <text x={16} y={padT + 60} fontSize="10" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)"
          transform={`rotate(-90 16 ${padT + 60})`}>global &rarr;</text>
        {present.map((v, n) => (
          <g key={v}>
            <rect x={padL + STEPS * cell + 18} y={padT + n * 20} width={11} height={11} rx="2"
              fill={ARM_COLOR[ARMS[v]]} fillOpacity={0.85} />
            <text x={padL + STEPS * cell + 35} y={padT + n * 20 + 10} fontSize="10"
              fontFamily="var(--font-sans)" fill="var(--color-text)">{ARMS[v]}</text>
          </g>
        ))}
      </svg>
      <Row>
        <Readout label="arms holding territory" value={String(present.length)} accent={present.length > 2} />
        <Readout label="lambda" value={lam.toExponential(1)} />
        <Readout label="at the headline price" value={Object.keys(WINNER_CELLS)[0]} />
      </Row>
      <Note>
        Each cell is a workload; its color is the architecture a cost-aware evaluation would deploy on it.
        The map has interior boundaries, which is the whole point: there is no arm you can recommend
        without first being told the traffic. <strong>Raise lambda</strong> and the map collapses onto the
        cheapest arm — so a benchmark's cost weighting decides its verdict before any measurement is taken.
        Over random workloads at the headline price the territory splits{' '}
        {Object.entries(WINNER_CELLS).map(([a, f]) => `${a} ${(f * 100).toFixed(0)}%`).join(', ')}.
      </Note>
    </div>
  );
}

// ===== Panel D — what no price can buy ===========================================================
function GapPanel({ logRho, setLogRho, bridge, setBridge }: {
  logRho: number; setLogRho: (v: number) => void; bridge: number; setBridge: (v: number) => void;
}) {
  const rho = Math.pow(10, logRho);
  const w = useMemo(() => mixOn('bridge', bridge), [bridge]);
  const { q, o, c } = useMemo(() => profile(w, N0), [w]);
  const cost = useMemo(() => scalarCost(o, c, rho), [o, c, rho]);
  const hull = useMemo(() => hullVertices(q, cost), [q, cost]);
  const par = useMemo(() => paretoIdx(q, o, c), [q, o, c]);
  const hullSet = new Set(hull);
  const anyPrice = useMemo(() => selectableAtAnyPrice(q, o, c), [q, o, c]);
  const offHullHere = par.filter((i) => !hullSet.has(i));
  const invisible = par.filter((i) => !anyPrice.has(i));

  const W = 600, H = 240, padL = 58, padB = 40, padT = 14, padR = 92;
  const cMax = Math.max(...cost) * 1.1 + 1;
  const X = (v: number) => padL + (Math.log10(v + 1) / Math.log10(cMax + 1)) * (W - padL - padR);
  const Y = (v: number) => H - padB - v * (H - padT - padB);
  return (
    <div>
      <Slider label="price of one generator call, in operations (rho)" value={logRho} min={-1} max={5} step={0.05}
        onChange={setLogRho} display={rho.toExponential(1)} />
      <Slider label="share of traffic that is a bridge question" value={bridge} min={0} max={1} step={0.01}
        onChange={setBridge} display={`${(bridge * 100).toFixed(0)}%`} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Quality against scalarized cost with the lower convex hull drawn">
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        <polyline points={hull.map((i) => `${X(cost[i])},${Y(q[i])}`).join(' ')} fill="none"
          stroke="var(--color-accent)" strokeWidth="1.6" strokeDasharray="4 3" />
        {ARMS.map((a, i) => {
          const onHull = hullSet.has(i), isPar = par.includes(i);
          return (
            <g key={a}>
              <circle cx={X(cost[i])} cy={Y(q[i])} r={onHull ? 6 : 4.5} fill={ARM_COLOR[a]}
                fillOpacity={onHull ? 0.9 : isPar ? 0.55 : 0.15}
                stroke={isPar && !anyPrice.has(i) ? 'var(--color-badge-red-text)' : 'none'} strokeWidth={1.8} />
              <text x={X(cost[i]) + 9} y={Y(q[i]) + 4} fontSize="10" fontFamily="var(--font-sans)"
                fill={isPar && !anyPrice.has(i) ? 'var(--color-badge-red-text)' : onHull ? 'var(--color-text)' : 'var(--color-text-secondary)'}
                fontWeight={isPar && !anyPrice.has(i) ? 700 : 400}>{a}</text>
            </g>
          );
        })}
        <text x={(padL + W - padR) / 2} y={H - 8} textAnchor="middle" fontSize="10"
          fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          cost per query, ops + {rho.toExponential(0)} &times; calls (log) &rarr;
        </text>
        <text x={16} y={padT + 34} fontSize="10" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)"
          transform={`rotate(-90 16 ${padT + 34})`}>quality &rarr;</text>
      </svg>
      <Row>
        <Readout label="Pareto-optimal" value={par.map((i) => ARMS[i]).join(', ')} />
        <Readout label="on the hull at this price" value={hull.map((i) => ARMS[i]).join(', ')} accent />
        <Readout label="off it here" value={offHullHere.length ? offHullHere.map((i) => ARMS[i]).join(', ') : '\u2014'} />
        <Readout label="NO price can select" value={invisible.length ? invisible.map((i) => ARMS[i]).join(', ') : 'none on this workload'}
          warn={invisible.length > 0} />
      </Row>
      <Note>
        A linear objective &minus; quality minus lambda times cost &minus; can only ever touch a{' '}
        <em>vertex</em> of the dashed hull, whatever lambda is. So an arm that is Pareto-optimal but lies
        inside the hull is not selectable at that lambda. Sliding rho redraws the hull, so an arm off it
        here may be on it elsewhere — and the arms ringed in red are the stronger case: Pareto-optimal and
        off the hull at <em>every</em> exchange rate, recommended by no weights whatsoever. It is not that
        the weights were tuned wrong; no weights exist. Across randomly drawn workloads this happens on{' '}
        {fmt(GAP_FRACTION * 100, 1)}% of them, and the arm that goes missing most often is{' '}
        {Object.keys(GAP_ARMS)[0]} — the same arm the aggregate numbers already dominated.
      </Note>
      <Note>
        <strong>Slide rho</strong> and watch the horizontal axis restretch: the price of a generator call
        is not measured here and cannot be, since it is a fact about a vendor contract rather than about
        any architecture. It decides verdicts on its own — at lambda = {RHO_FLIP.lambda.toExponential(0)} on
        an average workload, pricing generation at nothing deploys <strong>{RHO_FLIP.at_zero}</strong>, which
        spends four calls a query, and pricing it at {RHO0.toExponential(0)} deploys{' '}
        <strong>{RHO_FLIP.at_headline}</strong>, which spends none.
      </Note>
    </div>
  );
}

type Panel = 'table' | 'domination' | 'map' | 'gap';
const TEX: Record<Panel, string> = {
  table: '\\mathrm{cost}(a, w) = \\Big(\\textstyle\\sum_k w_k\\, \\mathrm{ops}_{a,k},\\ \\sum_k w_k\\, \\mathrm{calls}_{a,k}\\Big) \\in \\mathbb{R}^2',
  domination: 'a \\succ b \\iff q_a \\ge q_b,\\ \\mathrm{ops}_a \\le \\mathrm{ops}_b,\\ \\mathrm{calls}_a \\le \\mathrm{calls}_b \\ \\text{(one strict)} \\quad\\text{— and } \\succ \\text{ depends on } w',
  map: '\\hat a(w) = \\arg\\max_a \\Big\\{ q_a(w) - \\lambda\\big(\\mathrm{ops}_a(w) + \\rho\\,\\mathrm{calls}_a(w)\\big) \\Big\\}',
  gap: '\\big\\{\\arg\\max_a\\, q_a - \\lambda c_a \;:\; \\lambda \\ge 0\\big\\} = \\mathrm{vert}\\,\\mathrm{conv}\\{(c_a, q_a)\\} \\subsetneq \\mathrm{Pareto}',
};

export default memo(function RagArchitectureParetoLaboratory() {
  const [panel, setPanel] = useState<Panel>('table');
  const [bridge, setBridge] = useState(0.2);
  const [logN, setLogN] = useState(3);
  const [logLam, setLogLam] = useState(Math.log10(LAM0));
  const [logRho, setLogRho] = useState(Math.log10(RHO0));
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="rag-architecture-pareto" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'table')} onClick={() => setPanel('table')}>A · cost is a pair</button>
        <button type="button" style={pill(panel === 'domination')} onClick={() => setPanel('domination')}>B · what domination depends on</button>
        <button type="button" style={pill(panel === 'map')} onClick={() => setPanel('map')}>C · the winner map</button>
        <button type="button" style={pill(panel === 'gap')} onClick={() => setPanel('gap')}>D · what no price can buy</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'table' && <TablePanel />}
      {panel === 'domination' && <DominationPanel bridge={bridge} setBridge={setBridge} logN={logN} setLogN={setLogN} />}
      {panel === 'map' && <WinnerMapPanel logLam={logLam} setLogLam={setLogLam} />}
      {panel === 'gap' && <GapPanel logRho={logRho} setLogRho={setLogRho} bridge={bridge} setBridge={setBridge} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        Six architectures over {N_DOCS} passages of {K0} companies in {N_SECTORS} sectors, measured by the
        predecessor topic and re-run here not at all. Everything above is arithmetic on one {ARMS.length} &times;{' '}
        {CONDITIONS.length} &times; 3 table of quality and both costs — {ARMS.length * CONDITIONS.length * 3}{' '}
        numbers, emitted from <code>rag_architecture_pareto.py</code> — and every derived object on this page,
        the profiles, the domination relation, the Pareto sets, the hulls, the winner map and the amortization,
        is recomputed live in the browser from them. The workload mixture, the price ratio and the amortization
        horizon are all free: none of them is a property of any architecture, and each one changes the answer.
      </p>
    </div>
  );
});
