import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import katex from 'katex';

/**
 * Bandit Exploration Laboratory — four panels for the successor to
 * `rag-architecture-switching-hysteresis`. Four topics chose an architecture from a table of
 * measured quality and cost. A deployment has no such table: it sees only the arm it ran.
 *   A. What a deployment cannot see. The full outcome table against the sliver of it one
 *      deployment actually observes, as the horizon slider runs.
 *   B. Regret against the best FIXED arm, cumulative. An exploring policy crosses ZERO and keeps
 *      going: once the workload drifts, no single architecture is right for the whole deployment.
 *   C. The predecessor's instrument, where its premise no longer holds. The margin sweep with its
 *      standard errors, against UCB's flat line — most of the grid is indistinguishable.
 *   D. A switching cost is a tax on exploration. Recomputed LIVE, and exactly: a policy's
 *      trajectory does not depend on the price, so total = regret + price x switches.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): the block below is EMITTED from viz_constants() in
 * notebooks/rag-architecture-bandit-exploration/rag_architecture_bandit_exploration.py, and
 * test_laboratory_constants_match_the_module parses this file back and compares it to a fresh
 * bake. Panel D's entire cost table is recomputed here from the baked (regret, switches) pairs
 * rather than read from a baked table, and test_total_cost_is_affine_in_the_price_of_a_switch
 * pins that arithmetic against the module.
 */

// --- emitted from rag_architecture_bandit_exploration.py viz_constants() ---
const ARMS = ["naive","hybrid","hyde","corrective","graph","agentic"] as const;
const CLASSES = ["on","partial","off","noisy","bridge"] as const;
const N_DEPLOY = 24, UCB_C = 0.5;
const HORIZON = 2400;
// the table a deployment never sees: every arm's outcome on every query
const OK: number[][] = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]];
const KLASS: number[] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4];
const UTILITY: number[][] = [[0.997,0.347,0.097,-0.003,-0.003],[0.791,0.991,0.391,0.291,-0.009],[0.777,0.077,0.677,-0.123,-0.123],[0.965,0.933,0.266,0.316,-0.035],[0.8,0.45,0.2,-0.0,-0.0],[0.061,0.051,0.001,0.051,0.761]];
const POLICIES = [{"policy":"greedy","regret":123.139,"std":143.371,"se":29.266,"switches":16.167,"beats_fixed":1,"n":24},{"policy":"margin margin=0.1","regret":123.462,"std":117.306,"se":23.945,"switches":8.958,"beats_fixed":0,"n":24},{"policy":"ucb","regret":-93.16,"std":27.844,"se":5.684,"switches":396.583,"beats_fixed":24,"n":24},{"policy":"thompson","regret":42.386,"std":47.67,"se":9.73,"switches":1517.542,"beats_fixed":4,"n":24},{"policy":"ucb window=400","regret":-101.057,"std":31.559,"se":6.442,"switches":342.042,"beats_fixed":24,"n":24}];
const PAIRED = [{"policy":"greedy","diff":216.3,"t":7.10241,"p":3.10223e-07,"ci_lo":153.3,"ci_hi":279.299},{"policy":"margin margin=0.1","diff":216.622,"t":8.57391,"p":1.28046e-08,"ci_lo":164.357,"ci_hi":268.887},{"policy":"thompson","diff":135.547,"t":13.5419,"p":1.90981e-12,"ci_lo":114.841,"ci_hi":156.253},{"policy":"ucb window=400","diff":-7.89637,"t":-1.35671,"p":0.188043,"ci_lo":-19.9365,"ci_hi":4.14375}];
const MARGIN_SWEEP = [{"margin":-0.2,"regret":136.364,"std":130.327,"se":26.603,"switches":1097.917,"beats_fixed":0,"n":24},{"margin":-0.12,"regret":119.389,"std":124.315,"se":25.376,"switches":505.917,"beats_fixed":2,"n":24},{"margin":-0.08,"regret":115.68,"std":125.29,"se":25.575,"switches":159.0,"beats_fixed":1,"n":24},{"margin":-0.05,"regret":112.412,"std":126.092,"se":25.738,"switches":51.875,"beats_fixed":1,"n":24},{"margin":-0.03,"regret":125.718,"std":118.218,"se":24.131,"switches":20.875,"beats_fixed":0,"n":24},{"margin":-0.015,"regret":126.132,"std":117.793,"se":24.044,"switches":14.958,"beats_fixed":0,"n":24},{"margin":0.0,"regret":123.139,"std":143.371,"se":29.266,"switches":16.167,"beats_fixed":1,"n":24},{"margin":0.015,"regret":131.056,"std":130.92,"se":26.724,"switches":12.417,"beats_fixed":0,"n":24},{"margin":0.03,"regret":139.648,"std":157.13,"se":32.074,"switches":11.458,"beats_fixed":1,"n":24},{"margin":0.05,"regret":126.169,"std":116.924,"se":23.867,"switches":10.667,"beats_fixed":0,"n":24},{"margin":0.1,"regret":123.462,"std":117.306,"se":23.945,"switches":8.958,"beats_fixed":0,"n":24},{"margin":0.2,"regret":171.288,"std":183.995,"se":37.558,"switches":7.5,"beats_fixed":0,"n":24}];
const MARGINS_TIED = {"best_margin":-0.05,"best_regret":112.412,"se":25.738,"n_tied":10,"n_total":12,"tied":[-0.2,-0.12,-0.08,-0.05,-0.03,-0.015,0.0,0.015,0.05,0.1],"spread_ratio":4.529,"margin_std":126.092,"ucb_std":27.844,"ucb_regret":-93.16};
const CROSSOVER = {"cost":0.56,"before":"ucb","after":"margin","switches":{"greedy":16.167,"ucb":396.583,"thompson":1517.542}};
const RATE = {"horizons":[300.0,600.0,1200.0,2400.0,4800.0],"regret":[23.697,36.083,54.253,71.76,83.821],"loglog_slope":0.464,"log_r2":0.995,"sqrt_slope":0.5,"two_thirds_slope":0.667};
const TRAJECTORIES: Record<string, number[]> = {"greedy":[-0.244,-4.794,-10.201,-15.769,-20.971,-25.572,-30.798,-35.105,-39.337,-44.085,-47.655,-51.293,-56.844,-60.346,-64.354,-67.562,-71.122,-73.873,-77.299,-82.137,-85.501,-88.449,-91.904,-94.137,-96.501,-100.051,-102.204,-104.244,-105.75,-107.472,-109.17,-110.865,-111.037,-112.208,-111.937,-112.309,-113.329,-113.973,-114.09,-113.957,-113.983,-114.173,-114.395,-113.392,-112.704,-112.062,-110.856,-109.424,-108.342,-106.684,-106.229,-104.53,-103.24,-101.373,-100.625,-98.654,-96.556,-94.051,-90.626,-88.765,-86.796,-84.438,-81.491,-77.765,-74.161,-70.51,-67.506,-63.338,-59.38,-54.455,-51.089,-47.36,-42.95,-38.638,-34.492,-28.622,-24.067,-19.558,-13.844,-9.237,-3.406,1.871,8.089,13.68,19.049,24.045,30.883,36.978,43.096,49.627,56.726,63.234,67.876,74.046,79.828,86.495,93.273,100.864,108.178,116.103,123.139],"margin":[-0.244,-5.184,-11.439,-17.537,-23.641,-29.018,-35.135,-40.249,-45.326,-50.575,-54.851,-59.142,-65.451,-69.811,-74.77,-78.167,-82.348,-85.629,-89.478,-94.729,-98.932,-102.371,-106.324,-108.867,-111.282,-115.252,-117.932,-120.738,-122.974,-124.95,-127.29,-129.119,-129.721,-131.327,-131.915,-132.701,-134.009,-134.974,-135.435,-135.55,-136.095,-136.631,-137.365,-136.85,-136.405,-135.779,-134.86,-133.649,-132.81,-131.613,-131.381,-130.17,-129.194,-127.78,-127.24,-125.576,-123.7,-121.364,-118.199,-116.441,-114.649,-112.379,-109.503,-105.993,-102.266,-98.661,-95.854,-91.935,-88.273,-82.871,-78.373,-74.102,-69.36,-64.688,-60.122,-53.643,-48.213,-42.99,-36.499,-31.512,-24.686,-18.607,-11.605,-4.924,1.74,7.344,14.984,22.196,28.733,35.827,43.812,50.966,56.353,64.18,71.41,79.523,87.998,96.659,105.167,115.022,123.462],"ucb":[-0.244,-4.591,-9.918,-14.827,-20.318,-25.268,-30.504,-34.861,-39.721,-44.256,-48.68,-52.963,-59.292,-64.122,-69.354,-72.884,-77.172,-80.016,-84.164,-89.606,-93.836,-97.014,-100.792,-103.387,-105.758,-109.668,-112.186,-114.806,-117.571,-119.326,-121.724,-123.373,-124.19,-125.561,-125.817,-126.987,-128.328,-129.412,-129.858,-129.936,-130.48,-131.132,-132.108,-131.795,-131.298,-130.747,-129.679,-128.291,-127.459,-126.112,-125.59,-124.824,-124.383,-123.81,-123.15,-122.14,-121.387,-120.277,-119.326,-117.902,-117.132,-116.141,-114.737,-112.599,-110.704,-109.313,-108.353,-107.021,-106.072,-105.27,-104.132,-102.808,-101.477,-99.979,-98.43,-97.432,-96.54,-95.888,-94.981,-94.152,-93.328,-93.152,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16,-93.16],"thompson":[-0.244,-4.163,-8.842,-13.173,-17.908,-22.002,-26.74,-30.795,-35.208,-39.509,-42.986,-46.555,-51.951,-55.619,-60.123,-63.204,-66.525,-69.303,-72.747,-77.072,-81.102,-83.914,-87.095,-89.231,-91.062,-94.55,-96.986,-99.312,-101.316,-102.939,-104.999,-106.813,-107.296,-108.258,-108.104,-108.975,-109.906,-110.636,-111.282,-111.276,-111.388,-111.25,-111.485,-111.07,-110.459,-109.322,-108.78,-107.39,-106.384,-104.481,-103.849,-102.477,-101.4,-99.766,-99.097,-97.41,-95.413,-92.834,-89.879,-88.185,-86.314,-84.014,-81.337,-78.053,-74.831,-71.634,-68.736,-65.545,-61.924,-57.389,-53.16,-49.318,-45.385,-41.679,-37.632,-32.547,-27.955,-23.754,-19.53,-15.682,-11.339,-7.136,-2.427,1.313,5.372,7.995,11.827,15.13,18.37,20.838,23.498,26.023,28.204,30.465,32.224,34.2,36.383,38.466,40.025,41.638,42.386]};

const ARM_COLOR: Record<string, string> = {
  naive: 'var(--color-text-secondary)', hybrid: 'var(--color-accent)',
  hyde: 'var(--color-badge-red-text)', corrective: 'var(--color-definition-border)',
  graph: 'var(--color-theorem-border)', agentic: 'var(--color-text)',
};
const POLICY_COLOR: Record<string, string> = {
  greedy: 'var(--color-text-secondary)', margin: 'var(--color-badge-red-text)',
  ucb: 'var(--color-accent)', thompson: 'var(--color-theorem-border)',
};
const fmt = (x: number, n = 1) => (Number.isFinite(x) ? x.toFixed(n) : '∞');
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
      <span style={{ minWidth: '17rem' }}>{label} = <strong>{display}</strong></span>
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

// ===== the live engine — total cost is AFFINE in the price of a switch ========================
/**
 * A policy's trajectory does not depend on what a switch costs; only the bill does. So the whole
 * price sweep is arithmetic on the baked (regret, switches) pairs:  total = regret + price*switches.
 * The module asserts this identity, so Panel D is exact rather than an approximation.
 */
const totalAt = (regret: number, switches: number, price: number) => regret + price * switches;
const POLICY_BY = Object.fromEntries(POLICIES.map((p) => [p.policy, p])) as Record<string, typeof POLICIES[number]>;
/** The best margin setting at a given price, chosen the way the module chooses it. */
function bestMarginAt(price: number) {
  let best = MARGIN_SWEEP[0], bestTot = Infinity;
  for (const r of MARGIN_SWEEP) {
    const t = totalAt(r.regret, r.switches, price);
    if (t < bestTot) { bestTot = t; best = r; }
  }
  return { margin: best.margin, total: bestTot };
}
const CONTENDERS = ['greedy', 'ucb', 'thompson'] as const;

// ===== Panel A — what a deployment cannot see =================================================
function TablePanel({ share, setShare }: { share: number; setShare: (v: number) => void }) {
  const nq = OK[0].length;
  const util = useMemo(() => {
    const n = CLASSES.length;
    const w = CLASSES.map((c) => (c === 'bridge' ? share : (1 - share) / (n - 1)));
    return UTILITY.map((row) => row.reduce((s, v, k) => s + v * w[k], 0));
  }, [share]);
  const best = util.reduce((b, x, i) => (x > util[b] ? i : b), 0);
  const W = 620, H = 150, padL = 74, padT = 10, cw = (W - padL - 8) / nq, rh = 18;
  const seen = HORIZON, total = HORIZON * ARMS.length;

  return (
    <div>
      <Slider label="bridge share of the arriving traffic" value={share} min={0} max={1} step={0.01}
        onChange={setShare} display={share.toFixed(2)} />
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img"
        aria-label="every architecture's outcome on every query of the corpus">
        {OK.map((row, a) => (
          <g key={ARMS[a]}>
            <text x={padL - 6} y={padT + a * rh + 13} fontSize="10" textAnchor="end"
              fill={a === best ? 'var(--color-accent)' : 'var(--color-text-secondary)'}
              fontWeight={a === best ? 700 : 400} fontFamily="var(--font-sans)">{ARMS[a]}</text>
            {row.map((v, i) => (
              <rect key={i} x={padL + i * cw} y={padT + a * rh} width={Math.max(cw - 0.4, 0.6)} height={rh - 2}
                fill={v ? ARM_COLOR[ARMS[a]] : 'var(--color-border)'} opacity={v ? 0.85 : 0.35} />
            ))}
          </g>
        ))}
        {KLASS.map((k, i) => (
          <rect key={`k${i}`} x={padL + i * cw} y={padT + ARMS.length * rh + 3}
            width={Math.max(cw - 0.4, 0.6)} height={7}
            fill={k === CLASSES.indexOf('bridge') ? 'var(--color-text)' : 'var(--color-text-secondary)'}
            opacity={k === CLASSES.indexOf('bridge') ? 0.85 : 0.22 + 0.1 * k} />
        ))}
        <text x={padL} y={padT + ARMS.length * rh + 24} fontSize="10" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          {nq} queries — solid means that architecture answered it correctly; the band below is the
          question class, with `bridge` darkest
        </text>
      </svg>
      <Row>
        <Readout label="best arm at this workload" value={ARMS[best]} accent />
        <Readout label="its utility" value={util[best].toFixed(3)} />
        <Readout label="outcomes a deployment observes" value={`${seen.toLocaleString()} of ${total.toLocaleString()}`} warn />
        <Readout label="that is" value={`1 in ${ARMS.length}`} warn />
      </Row>
      <Note>
        Every previous topic in this arc read a table like this one as given. A deployment cannot.
        It runs one architecture per query and observes that row only; the other{' '}
        {ARMS.length - 1} entries in each column are counterfactual and stay that way. Over a
        {' '}{HORIZON.toLocaleString()}-query deployment that is {seen.toLocaleString()} observations out
        of {total.toLocaleString()} outcomes — and the missing five-sixths are exactly the ones that
        would tell you whether the arm you ran was the right one. Drag the slider: the best arm
        changes as the traffic does, and a deployment has to notice that from its own sliver.
      </Note>
    </div>
  );
}

// ===== Panel B — regret against the best fixed arm ============================================
function RegretPanel() {
  const keys = Object.keys(TRAJECTORIES);
  const all = keys.flatMap((k) => TRAJECTORIES[k]);
  const lo = Math.min(...all), hi = Math.max(...all);
  const n = TRAJECTORIES[keys[0]].length;
  const W = 620, H = 230, padL = 56, padR = 96, padT = 16, padB = 30;
  const X = (i: number) => padL + (i / (n - 1)) * (W - padL - padR);
  const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo || 1)) * (H - padT - padB);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img"
        aria-label="cumulative regret against the best fixed arm, by policy">
        <line x1={padL} y1={Y(0)} x2={W - padR} y2={Y(0)} stroke="var(--color-text)" strokeDasharray="4 3" />
        <text x={padL + 4} y={Y(0) - 5} fontSize="10" fill="var(--color-text)" fontFamily="var(--font-sans)">
          the best fixed arm
        </text>
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        {keys.map((k) => (
          <g key={k}>
            <path d={TRAJECTORIES[k].map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}
              fill="none" stroke={POLICY_COLOR[k]} strokeWidth={2} />
            <text x={W - padR + 5} y={Y(TRAJECTORIES[k][n - 1]) + 3} fontSize="10.5"
              fill={POLICY_COLOR[k]} fontFamily="var(--font-sans)">{k}</text>
          </g>
        ))}
        <text x={padL} y={H - 8} fontSize="10" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">query 1</text>
        <text x={W - padR} y={H - 8} fontSize="10" textAnchor="end" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          query {HORIZON.toLocaleString()}
        </text>
      </svg>
      <Row>
        {POLICIES.filter((p) => !p.policy.includes('window')).map((p) => (
          <Readout key={p.policy} label={p.policy.split(' ')[0]} value={fmt(p.regret)}
            accent={p.regret < 0} warn={p.regret > 100} />
        ))}
      </Row>
      <Note>
        Cumulative regret against the best single architecture chosen in hindsight, averaged over{' '}
        {N_DEPLOY} deployments. An uncertainty-weighted policy does not merely approach that
        benchmark — it crosses zero and keeps going, on{' '}
        {POLICY_BY['ucb'].beats_fixed} of {POLICY_BY['ucb'].n} deployments. That is not a policy
        beating an oracle. It is a benchmark that drift has made weak: the best arm early is not the
        best arm late, so no fixed choice is right throughout, and the fixed-arm framing that the
        previous four topics optimized was leaving that on the table.
      </Note>
    </div>
  );
}

// ===== Panel C — the predecessor's instrument, where its premise fails ========================
function MarginPanel() {
  const ucb = POLICY_BY['ucb'];
  const vals = MARGIN_SWEEP.flatMap((r) => [r.regret - r.se, r.regret + r.se]).concat([ucb.regret]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const W = 620, H = 235, padL = 58, padR = 20, padT = 16, padB = 40;
  const X = (i: number) => padL + (i / (MARGIN_SWEEP.length - 1)) * (W - padL - padR);
  const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo || 1)) * (H - padT - padB);
  const best = MARGIN_SWEEP.reduce((b, r) => (r.regret < b.regret ? r : b), MARGIN_SWEEP[0]);
  const band = best.regret + best.se;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img"
        aria-label="the margin sweep with standard errors, against UCB">
        <rect x={padL} y={Y(band)} width={W - padL - padR} height={Math.max(Y(lo) - Y(band), 0)}
          fill="var(--color-border)" opacity={0.28} />
        <text x={W - padR - 4} y={Y(band) - 4} fontSize="10" textAnchor="end"
          fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          within one standard error of the best margin
        </text>
        <line x1={padL} y1={Y(ucb.regret)} x2={W - padR} y2={Y(ucb.regret)}
          stroke="var(--color-accent)" strokeWidth={2} />
        <text x={padL + 4} y={Y(ucb.regret) - 5} fontSize="10.5" fill="var(--color-accent)" fontFamily="var(--font-sans)">
          UCB — the whole margin family is above this line
        </text>
        {MARGIN_SWEEP.map((r, i) => (
          <g key={r.margin}>
            <line x1={X(i)} y1={Y(r.regret - r.se)} x2={X(i)} y2={Y(r.regret + r.se)}
              stroke="var(--color-badge-red-text)" strokeWidth={1.4} />
            <circle cx={X(i)} cy={Y(r.regret)} r={3.2} fill="var(--color-badge-red-text)" />
            <text x={X(i)} y={H - padB + 13} fontSize="8.5" textAnchor="middle"
              fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
              {r.margin > 0 ? `+${r.margin}` : r.margin}
            </text>
          </g>
        ))}
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        <text x={W / 2} y={H - 6} fontSize="10" textAnchor="middle" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          the margin a challenger must clear before the incumbent is displaced
        </text>
      </svg>
      <Row>
        <Readout label="margins within one se of the best" value={`${MARGINS_TIED.n_tied} of ${MARGINS_TIED.n_total}`} warn />
        <Readout label="spread, margin vs UCB" value={`${fmt(MARGINS_TIED.spread_ratio, 2)}×`} warn />
        <Readout label="best margin regret" value={fmt(MARGINS_TIED.best_regret)} />
        <Readout label="UCB regret" value={fmt(MARGINS_TIED.ucb_regret)} accent />
      </Row>
      <Note>
        The previous topic found a clean interior optimum in exactly this parameter, and both ends of
        its range lost on every seed. Here {MARGINS_TIED.n_tied} of {MARGINS_TIED.n_total} settings sit
        within one standard error of the best of them. The instrument has not shifted or inverted — it
        has stopped resolving, because a <em>constant</em> margin cannot express how well any
        particular arm is currently known, and that is the only thing worth knowing when the table is
        unknown. It also pays for that in reliability: {fmt(MARGINS_TIED.spread_ratio, 2)}× the
        deployment-to-deployment spread of an uncertainty-weighted policy.
      </Note>
      <Note>
        Paired against UCB on the same deployments, which cancels the shared difficulty of a traffic
        draw: {PAIRED.filter((p) => p.p < 0.05).map((p) => `${p.policy.split(' ')[0]} (p = ${p.p.toPrecision(2)})`).join(', ')} all
        lose decisively — while the sliding-window variant does <em>not</em> separate from plain UCB
        (p = {PAIRED.find((p) => p.policy.includes('window'))!.p.toPrecision(2)}). Forgetting is the
        standard remedy for a moving target, and on a single slow changeover it is not measurably
        worth it here. That one is reported rather than claimed.
      </Note>
    </div>
  );
}

// ===== Panel D — a switching cost is a tax on exploration =====================================
function CostPanel({ price, setPrice }: { price: number; setPrice: (v: number) => void }) {
  const PMAX = 1.2;
  const axis = useMemo(() => Array.from({ length: 61 }, (_, i) => (i / 60) * PMAX), []);
  const curves = useMemo(() => {
    const out: Record<string, number[]> = { margin: axis.map((p) => bestMarginAt(p).total) };
    for (const k of CONTENDERS) out[k] = axis.map((p) => totalAt(POLICY_BY[k].regret, POLICY_BY[k].switches, p));
    return out;
  }, [axis]);
  const here = useMemo(() => {
    const t: Record<string, number> = { margin: bestMarginAt(price).total };
    for (const k of CONTENDERS) t[k] = totalAt(POLICY_BY[k].regret, POLICY_BY[k].switches, price);
    return t;
  }, [price]);
  const winner = Object.keys(here).reduce((b, k) => (here[k] < here[b] ? k : b), 'ucb');
  const keys = Object.keys(curves);
  const all = keys.flatMap((k) => curves[k]);
  const lo = Math.min(...all), hi = Math.min(Math.max(...all), 600);
  const W = 620, H = 225, padL = 56, padR = 86, padT = 16, padB = 32;
  const X = (p: number) => padL + (p / PMAX) * (W - padL - padR);
  const Y = (v: number) => padT + (1 - (Math.min(v, hi) - lo) / (hi - lo || 1)) * (H - padT - padB);
  return (
    <div>
      <Slider label="price of one switch" value={price} min={0} max={PMAX} step={0.02}
        onChange={setPrice} display={price === 0 ? '0 — switching is free' : price.toFixed(2)} />
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img"
        aria-label="total cost against the price of a switch, by policy">
        <line x1={padL} y1={Y(0)} x2={W - padR} y2={Y(0)} stroke="var(--color-text)" strokeDasharray="4 3" />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" />
        {keys.map((k) => (
          <g key={k}>
            <path d={curves[k].map((v, i) => `${i ? 'L' : 'M'}${X(axis[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}
              fill="none" stroke={POLICY_COLOR[k]} strokeWidth={k === winner ? 2.6 : 1.5}
              opacity={k === winner ? 1 : 0.7} />
            <text x={W - padR + 5} y={Y(curves[k][curves[k].length - 1]) + 3} fontSize="10"
              fill={POLICY_COLOR[k]} fontFamily="var(--font-sans)">{k}</text>
          </g>
        ))}
        <line x1={X(CROSSOVER.cost)} y1={padT} x2={X(CROSSOVER.cost)} y2={H - padB}
          stroke="var(--color-theorem-border)" strokeDasharray="3 3" />
        <text x={X(CROSSOVER.cost) + 4} y={padT + 11} fontSize="10" fill="var(--color-theorem-border)" fontFamily="var(--font-sans)">
          exploring stops paying at {CROSSOVER.cost}
        </text>
        <line x1={X(price)} y1={padT} x2={X(price)} y2={H - padB} stroke="var(--color-badge-red-text)" strokeWidth={1.4} />
        {[0, 0.4, 0.8, 1.2].map((p) => (
          <text key={p} x={X(p)} y={H - padB + 13} fontSize="10" textAnchor="middle" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{p}</text>
        ))}
      </svg>
      <Row>
        <Readout label="winner at this price" value={winner} accent />
        {keys.map((k) => <Readout key={k} label={k} value={fmt(here[k])} warn={here[k] > 300} />)}
      </Row>
      <Note>
        Total cost is regret plus the price of every switch, and a policy's trajectory does not depend
        on that price — so this whole panel is recomputed from the same numbers the module measured,
        not read off a stored curve. The tax falls in proportion to how much a policy explores:
        thompson switches {fmt(CROSSOVER.switches.thompson, 0)} times, UCB{' '}
        {fmt(CROSSOVER.switches.ucb, 0)}, and a rule that mostly holds still only{' '}
        {fmt(CROSSOVER.switches.greedy, 0)}. Past a price of {CROSSOVER.cost} the cheap rule wins — not
        because it knows more, but because it has stopped paying to find out. That is the honest shape
        of the trade this arc has been circling: information costs something, and when it costs enough,
        ignorance is the cheaper policy.
      </Note>
    </div>
  );
}

type Panel = 'table' | 'regret' | 'margin' | 'cost';
const TEX: Record<Panel, string> = {
  table: 'r_t = Q\\big(a_t, q_t\\big) - \\lambda\\big[\\,\\mathrm{ops}(a_t, q_t) + \\rho\\,\\mathrm{calls}(a_t, q_t)\\,\\big] \\quad \\text{observed for } a_t \\text{ only}',
  regret: 'R_T = \\max_{a} \\sum_{t=1}^{T} u\\big(a, w_t\\big) \; - \; \\sum_{t=1}^{T} u\\big(a_t, w_t\\big)',
  margin: 'a_t = \\begin{cases} \\arg\\max_a \\hat\\mu_a & \\text{if } \\max_a \\hat\\mu_a - \\hat\\mu_{a_{t-1}} > \\delta \\\\ a_{t-1} & \\text{otherwise} \\end{cases} \\qquad \\text{vs} \\qquad \\arg\\max_a \\Big[ \\hat\\mu_a + c\\sqrt{\\tfrac{2\\log t}{n_a}} \\Big]',
  cost: '\\mathrm{total}(p) = R_T \; + \; p \\cdot \\#\\{\\text{switches}\\}',
};

export default memo(function BanditExplorationLaboratory() {
  const [panel, setPanel] = useState<Panel>('table');
  const [share, setShare] = useState(0.5);
  const [price, setPrice] = useState(0);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="bandit-exploration" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'table')} onClick={() => setPanel('table')}>A · what you cannot see</button>
        <button type="button" style={pill(panel === 'regret')} onClick={() => setPanel('regret')}>B · regret</button>
        <button type="button" style={pill(panel === 'margin')} onClick={() => setPanel('margin')}>C · the old instrument</button>
        <button type="button" style={pill(panel === 'cost')} onClick={() => setPanel('cost')}>D · the tax on exploring</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'table' && <TablePanel share={share} setShare={setShare} />}
      {panel === 'regret' && <RegretPanel />}
      {panel === 'margin' && <MarginPanel />}
      {panel === 'cost' && <CostPanel price={price} setPrice={setPrice} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        The six architectures, their per-query outcomes and their costs arrive unchanged from the four
        predecessor topics; nothing about retrieval is re-measured, and the only new object is the
        feedback structure. Every figure is averaged over {N_DEPLOY} deployments of{' '}
        {HORIZON.toLocaleString()} queries, because a single run is one draw of exactly the noise this
        topic is about. UCB uses c&nbsp;=&nbsp;{UCB_C}. On a fixed corpus UCB's regret grows like{' '}
        log&nbsp;T — the data fit that with R²&nbsp;=&nbsp;{RATE.log_r2} — so the celebrated √T and
        T<sup>2/3</sup> rates, which are worst cases over instances rather than properties of one, are
        cited in the text and deliberately not fitted here.
      </p>
    </div>
  );
});
