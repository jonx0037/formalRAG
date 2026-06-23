import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * NDCG Laboratory — four panels for the `ndcg-discount-geometry` topic:
 *   A. Graded relevance & the ideal ranking. The worked query's top-K, each rank's discounted gain as a
 *      bar (their sum is DCG) against the ideal ordering's outline (sum IDCG); NDCG = DCG/IDCG. A gain
 *      toggle (linear vs exponential) reshapes the bars — exponential rewards the top grades.
 *   B. Discount geometry. The log2, geometric (RBP, slider p), and 1/i discount curves on one axis, with
 *      head-mass-in-top-K bars and the RBP user model E[docs] = 1/(1-p). The namesake panel.
 *   C. Convention sensitivity. Per-leg mean NDCG under each gain/discount convention (a stable quality
 *      ladder here), plus two constructed examples that each REVERSE the verdict (gain flip, discount flip).
 *   D. NDCG as estimator. A query-count slider shrinks the 95% CI like 1/sqrt(n); the clearest pair of
 *      legs separates by n=15, the closest (lexical/dense) would need ~185 — the significance cliffhanger.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): IDEAL_GRADES, GRADES_IN_RANK_ORDER, MNDCG, GAIN_FLIP, DISC_FLIP,
 * PER_Q_NDCG, SE_SCALING, SEP_CLEAR, SEP_CLOSEST are mirrored TO THE DECIMAL from
 * notebooks/ndcg-discount-geometry/ndcg_discount_geometry.py (viz_constants()). Matching asserts:
 * test_ndcg_matches_bm25_twin / test_rearrangement_inequality / test_discount_geometry /
 * test_ndcg_se_scales_as_inv_sqrt_n / test_two_leg_separation_contrast / test_convention_flip_constructed.
 * The lab recomputes ONLY closed forms in TS (DCG/IDCG/NDCG from the grade sequences, the discount curves
 * & head-mass, the sample mean/std, and the projected CI 1.96*std/sqrt(n)); the corpus-derived means and
 * per-query arrays are baked. Change a number here -> change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_DOCS = 120;
const R_SIZE = 10;
const N_QUERIES = 40;
const WORKED_Q = 30;
const RBP_P_DEFAULT = 0.85;

type Leg = 'lexical' | 'dense' | 'late_interaction';
const LEG_NAMES: Leg[] = ['lexical', 'dense', 'late_interaction'];
const LEG_LABEL: Record<Leg, string> = {
  lexical: 'lexical (BM25)', dense: 'dense (MIPS)', late_interaction: 'late interaction',
};
const LEG_COLOR: Record<Leg, string> = {
  lexical: '#c08457', dense: '#6a8caf', late_interaction: '#9b6abf',
};

// Panel A — the worked query (q=30): the ideal grade profile and each leg's grades in its rank order (top-K).
const IDEAL_GRADES: number[] = [3, 2, 2, 2, 2, 2, 1, 1, 1, 1];
const GRADES_IN_RANK_ORDER: Record<Leg, number[]> = {
  lexical: [2, 3, 2, 2, 2, 2, 0, 0, 0, 0],
  dense: [3, 2, 2, 2, 0, 0, 0, 0, 1, 0],
  late_interaction: [2, 3, 2, 2, 2, 2, 0, 0, 0, 1],
};

// Panel C — per-leg mean NDCG under the four conventions (corpus means, baked).
type Conv = 'lin_log' | 'exp_log' | 'exp_geo' | 'lin_geo';
const MNDCG: Record<Leg, Record<Conv, number>> = {
  lexical: { lin_log: 0.6917, exp_log: 0.7207, exp_geo: 0.7326, lin_geo: 0.7046 },
  dense: { lin_log: 0.761, exp_log: 0.7699, exp_geo: 0.7697, lin_geo: 0.7645 },
  late_interaction: { lin_log: 0.8749, exp_log: 0.8818, exp_geo: 0.8968, lin_geo: 0.8893 },
};
// the constructed flips (one query each): gain flip reverses headline<->broad; discount flip top_heavy<->deep.
const GAIN_FLIP = { exp: { headline: 0.818, broad: 0.601 }, lin: { headline: 0.658, broad: 0.75 } };
const DISC_FLIP = { geo: { top_heavy: 0.571, deep: 0.5 }, log: { top_heavy: 0.469, deep: 0.733 } };
const PQ_GAIN_REVERSALS = 5;

// Panel D — per-query NDCG (exponential gain, log2 discount); TS recomputes mean/std as a closed-form check.
const PER_Q_NDCG: Record<Leg, number[]> = {
  lexical: [0.592, 0.835, 0.486, 0.775, 0.796, 0.875, 0.556, 0.649, 0.812, 0.914, 0.872, 0.724, 0.807, 0.646, 0.953, 0.823, 0.788, 0.533, 0.949, 0.771, 0.621, 0.824, 0.114, 0.111, 0.898, 0.329, 0.66, 0.556, 0.556, 0.66, 0.821, 0.984, 0.922, 0.904, 0.811, 0.843, 0.741, 0.656, 0.949, 0.713],
  dense: [0.921, 0.918, 0.709, 0.665, 0.928, 0.87, 0.827, 0.727, 0.758, 0.716, 0.787, 0.821, 0.866, 0.694, 0.963, 0.843, 0.906, 0.707, 0.955, 0.782, 0.446, 0.567, 0.262, 0.826, 0.777, 0.599, 0.837, 0.81, 0.706, 0.729, 0.791, 0.769, 0.75, 0.815, 0.829, 0.818, 0.892, 0.831, 0.608, 0.768],
  late_interaction: [0.962, 0.929, 0.873, 0.741, 0.984, 0.846, 0.868, 0.797, 0.807, 0.959, 0.862, 0.986, 0.934, 0.65, 0.944, 0.882, 0.799, 0.897, 0.987, 0.955, 0.829, 0.95, 0.497, 0.846, 0.983, 0.796, 0.82, 1.0, 0.817, 0.824, 0.84, 0.99, 0.956, 0.965, 0.926, 0.902, 0.902, 0.982, 1.0, 0.787],
};
// SE ~ 1/sqrt(n) (late_interaction): [n, empirical SE]; theory SE = std/sqrt(n) recomputed in TS.
const SE_SCALING: [number, number][] = [[5, 0.0487], [10, 0.0331], [20, 0.0229], [40, 0.0166], [80, 0.0112]];
const SEP_CLEAR = 15;       // lexical vs late_interaction separate at n=15
const SEP_CLOSEST = 185;    // lexical vs dense would need ~185 queries (overlap across all 40)
const PAIRS: Record<'clearest' | 'closest', [Leg, Leg]> = {
  clearest: ['lexical', 'late_interaction'],
  closest: ['lexical', 'dense'],
};

const ACCENT = 'var(--color-accent)';
const MUTED = '#9aa3ad';
const ENV_COLOR = '#c79a3b';
const GRADE_COLOR = ['#dfe3e8', '#a9c5b4', '#6fa389', '#3f7d5e']; // grade 0,1,2,3

const fmt = (x: number, n = 3) => x.toFixed(n);

// --- closed-form TS recomputation -----------------------------------------------------
const gainLin = (g: number) => g;
const gainExp = (g: number) => 2 ** g - 1;
const discLog = (i: number) => 1 / Math.log2(i + 1);
const discGeo = (i: number, p: number) => p ** (i - 1);
const discRec = (i: number) => 1 / i;
// DCG of a grade sequence already in rank order: Σ_i gain(g_i)·discount(i), i 1-indexed.
const dcg = (grades: number[], gain: (g: number) => number, disc: (i: number) => number) =>
  grades.reduce((s, g, i) => s + gain(g) * disc(i + 1), 0);
const idcg = (grades: number[], gain: (g: number) => number, disc: (i: number) => number) =>
  dcg([...grades].sort((a, b) => b - a), gain, disc);
// NDCG = DCG/IDCG is computed inline in Panel A (where DCG@10 and IDCG@10 are also shown), so no
// separate helper — keeping the ratio at its one call site avoids recomputing dcg/idcg twice.
const headMass = (disc: (i: number) => number, k: number, n: number) => {
  let head = 0, total = 0;
  for (let i = 1; i <= n; i++) { const d = disc(i); total += d; if (i <= k) head += d; }
  return total > 0 ? head / total : 0;
};
const mean = (a: number[]) => (a.length === 0 ? 0 : a.reduce((s, x) => s + x, 0) / a.length);
const stdev = (a: number[]) => {
  if (a.length <= 1) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1));
};
const ciHalf = (sd: number, n: number) => (n > 0 ? (1.96 * sd) / Math.sqrt(n) : 0);

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

// ===== Panel A — graded relevance & the ideal ranking ==============================================
function GradedPanel({ leg, setLeg, expGain, setExpGain }: {
  leg: Leg; setLeg: (l: Leg) => void; expGain: boolean; setExpGain: (v: boolean) => void;
}) {
  const grades = GRADES_IN_RANK_ORDER[leg];
  const gain = expGain ? gainExp : gainLin;
  const W = 540, H = 200, padL = 38, padR = 12, padT = 16, padB = 34;
  const bw = (W - padL - padR) / R_SIZE;
  const contrib = grades.map((g, i) => gain(g) * discLog(i + 1));
  const idealContrib = IDEAL_GRADES.map((g, i) => gain(g) * discLog(i + 1));
  const maxC = Math.max(...idealContrib, ...contrib, 1e-9);
  const py = (v: number) => H - padB - (H - padT - padB) * (v / maxC);
  const d = dcg(grades, gain, discLog), id = idcg(IDEAL_GRADES, gain, discLog);
  const nd = id > 0 ? d / id : 0;
  return (
    <div>
      <LegPills leg={leg} setLeg={setLeg} />
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <button type="button" style={pill(!expGain)} onClick={() => setExpGain(false)}>linear gain g</button>
        <button type="button" style={pill(expGain)} onClick={() => setExpGain(true)}>exponential gain 2ᵍ−1</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="per-rank discounted gain bars versus the ideal ordering" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <text x={12} y={(padT + H - padB) / 2} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + H - padB) / 2})`}>gain · discount</text>
        {grades.map((g, i) => {
          const x = padL + i * bw;
          const ideal = idealContrib[i] ?? 0;
          return (
            <g key={i}>
              {/* ideal contribution outline */}
              <rect x={x + 1} y={py(ideal)} width={bw - 2} height={(H - padB) - py(ideal)}
                fill="none" stroke={ENV_COLOR} strokeWidth={1.2} strokeDasharray="3 2" />
              {/* actual contribution, colored by grade */}
              <rect x={x + 2.5} y={py(contrib[i])} width={bw - 5} height={(H - padB) - py(contrib[i])}
                fill={GRADE_COLOR[g] ?? GRADE_COLOR[0]} fillOpacity={0.92} />
              <text x={x + bw / 2} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{i + 1}</text>
              <text x={x + bw / 2} y={py(contrib[i]) - 3} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">g{g}</text>
            </g>
          );
        })}
        <text x={padL + 2} y={padT + 2} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">worked query {WORKED_Q}; bars = actual ranking · dashed = ideal ordering</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="DCG@10" value={fmt(d, 3)} />
        <Readout label="IDCG@10 (ideal)" value={fmt(id, 3)} />
        <Readout label="NDCG@10 = DCG/IDCG" value={fmt(nd, 4)} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Each bar is a rank's contribution gain(grade)·discount(rank); their sum is <strong>DCG</strong>, the dashed
        outline is the <span style={{ color: ENV_COLOR }}>ideal ordering</span> (sum IDCG), and the shaded fraction is
        NDCG = {fmt(nd, 3)}. Switch to <strong>exponential gain</strong>: grade-3 jumps from 3 to 7, so a leg that plants
        the perfect document near the top is rewarded far more than one that spreads marginal hits.
      </p>
    </div>
  );
}

// ===== Panel B — discount geometry =================================================================
function DiscountPanel({ p, setP }: { p: number; setP: (v: number) => void }) {
  const W = 380, H = 220, padL = 40, padR = 12, padT = 14, padB = 34;
  const K = 20;
  const px = (i: number) => padL + (W - padL - padR) * ((i - 1) / (K - 1));
  const py = (v: number) => H - padB - (H - padT - padB) * v;
  const curve = (disc: (i: number) => number) =>
    Array.from({ length: K }, (_v, j) => [px(j + 1), py(disc(j + 1))] as [number, number])
      .map(([x, y], j) => (j ? 'L' : 'M') + x + ' ' + y).join(' ');
  const headLog = headMass(discLog, R_SIZE, N_DOCS);
  const headRec = headMass(discRec, R_SIZE, N_DOCS);
  const headGeo = headMass((i) => discGeo(i, p), R_SIZE, N_DOCS);
  const eDocs = p < 1 ? 1 / (1 - p) : Infinity;
  const bars: [string, number, string][] = [
    ['log₂', headLog, MUTED], ['1/i', headRec, '#b58a4c'], [`geo(${fmt(p, 2)})`, headGeo, ACCENT],
  ];
  return (
    <div>
      <Slider label="RBP persistence p" value={p} min={0.5} max={0.95} step={0.01} onChange={setP} display={fmt(p, 2)} />
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="discount curves by rank" style={{ width: '100%', maxWidth: W, height: 'auto', flex: '1 1 300px' }}>
          <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
          {[0, 0.5, 1].map((v) => (<text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
          {[1, 5, 10, 15, 20].map((i) => (<text key={i} x={px(i)} y={H - padB + 13} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{i}</text>))}
          <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rank</text>
          <path d={curve(discLog)} fill="none" stroke={MUTED} strokeWidth={2} />
          <path d={curve(discRec)} fill="none" stroke="#b58a4c" strokeWidth={1.6} strokeDasharray="4 2" />
          <path d={curve((i) => discGeo(i, p))} fill="none" stroke={ACCENT} strokeWidth={2.2} />
          <text x={px(13)} y={py(discLog(13)) - 4} fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">1/log₂(i+1)</text>
          <text x={px(4)} y={py(discGeo(4, p)) - 4} fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">pⁱ⁻¹</text>
        </svg>
        <div style={{ flex: '1 1 180px' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginBottom: '0.3rem' }}>head mass in top {R_SIZE}</div>
          {bars.map(([name, v, color]) => (
            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', margin: '0.2rem 0' }}>
              <span style={{ minWidth: '4.5rem', fontSize: '0.74rem', fontFamily: 'var(--font-sans)' }}>{name}</span>
              <div style={{ flex: 1, height: '0.7rem', background: 'var(--color-border)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${v * 100}%`, height: '100%', background: color }} />
              </div>
              <span style={{ minWidth: '2.6rem', fontSize: '0.74rem', textAlign: 'right' }}>{fmt(v, 2)}</span>
            </div>
          ))}
          <div style={{ marginTop: '0.6rem' }}>
            <Readout label="RBP E[docs examined] = 1/(1−p)" value={p < 1 ? fmt(eDocs, 2) : '∞'} accent />
          </div>
        </div>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The <span style={{ color: ACCENT }}>geometric</span> discount is <strong>light-tailed</strong> — it concentrates
        weight in the head ({fmt(headGeo, 2)} of it in the top {R_SIZE}) and carries a user model (expected documents
        examined {p < 1 ? fmt(eDocs, 2) : '∞'}). The <span style={{ color: MUTED }}>logarithmic</span> discount is
        <strong> heavy-tailed</strong> (only {fmt(headLog, 2)} in the top {R_SIZE}), so it keeps crediting deep results —
        and it carries no such user model; it is a heuristic chosen for that slow decay.
      </p>
    </div>
  );
}

// ===== Panel C — convention sensitivity ============================================================
const CONV_LABEL: Record<Conv, string> = {
  lin_log: 'linear · log₂', exp_log: 'exp · log₂', exp_geo: 'exp · geometric', lin_geo: 'linear · geometric',
};
function FlipPair({ title, a, b, labelA, labelB, winnerA, winnerB, condA, condB }: {
  title: string; a: { x: number; y: number }; b: { x: number; y: number };
  labelA: string; labelB: string; winnerA: string; winnerB: string; condA: string; condB: string;
}) {
  const bar = (v: number, color: string) => (
    <div style={{ flex: 1, height: '0.7rem', background: 'var(--color-border)', borderRadius: 3, overflow: 'hidden' }}>
      <div style={{ width: `${v * 100}%`, height: '100%', background: color }} />
    </div>
  );
  const row = (label: string, v: number, win: boolean) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', margin: '0.15rem 0' }}>
      <span style={{ minWidth: '4.4rem', fontSize: '0.72rem', fontFamily: 'var(--font-sans)' }}>{label}</span>
      {bar(v, win ? ACCENT : MUTED)}
      <span style={{ minWidth: '2.4rem', fontSize: '0.72rem', textAlign: 'right' }}>{fmt(v, 2)}</span>
    </div>
  );
  return (
    <div style={{ flex: '1 1 220px', border: '1px solid var(--color-border)', borderRadius: '0.4rem', padding: '0.5rem 0.6rem' }}>
      <div style={{ fontSize: '0.76rem', fontWeight: 600, marginBottom: '0.3rem' }}>{title}</div>
      <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)' }}>{condA} → <strong>{winnerA}</strong> wins</div>
      {row(labelA, a.x, winnerA === labelA)}
      {row(labelB, a.y, winnerA === labelB)}
      <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginTop: '0.35rem' }}>{condB} → <strong>{winnerB}</strong> wins</div>
      {row(labelA, b.x, winnerB === labelA)}
      {row(labelB, b.y, winnerB === labelB)}
    </div>
  );
}
function ConventionPanel({ conv, setConv }: { conv: Conv; setConv: (c: Conv) => void }) {
  const W = 360, H = 150, padL = 90, padR = 40, padT = 10, padB = 14;
  const bh = (H - padT - padB) / LEG_NAMES.length;
  const maxV = 1;
  const bx = (v: number) => padL + (W - padL - padR) * (v / maxV);
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
        {(Object.keys(CONV_LABEL) as Conv[]).map((c) => (
          <button key={c} type="button" style={pill(conv === c)} onClick={() => setConv(c)}>{CONV_LABEL[c]}</button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="per-leg mean NDCG under the chosen convention" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {LEG_NAMES.map((leg, i) => {
          const v = MNDCG[leg][conv];
          const y = padT + i * bh;
          return (
            <g key={leg}>
              <text x={padL - 6} y={y + bh / 2 + 3} textAnchor="end" fontSize={9} fill="var(--color-text)" fontFamily="var(--font-sans)">{LEG_LABEL[leg]}</text>
              <rect x={padL} y={y + 3} width={bx(v) - padL} height={bh - 6} rx={2} fill={LEG_COLOR[leg]} fillOpacity={0.85} />
              <text x={bx(v) + 4} y={y + bh / 2 + 3} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{fmt(v, 3)}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.6rem', lineHeight: 1.5 }}>
        On this corpus the legs are a <strong>quality ladder</strong> under every convention, so the aggregate order does
        not flip — but the convention reverses the per-query verdict on <strong>{PQ_GAIN_REVERSALS}</strong> leg-pair
        instances, and two minimal constructed examples flip it starkly:
      </p>
      <div style={{ display: 'flex', gap: '0.7rem', flexWrap: 'wrap' }}>
        <FlipPair title="Gain flip (1 perfect + 3 marginal docs)"
          a={{ x: GAIN_FLIP.exp.headline, y: GAIN_FLIP.exp.broad }} b={{ x: GAIN_FLIP.lin.headline, y: GAIN_FLIP.lin.broad }}
          labelA="headline" labelB="broad" condA="exponential gain" condB="linear gain"
          winnerA="headline" winnerB="broad" />
        <FlipPair title="Discount flip (3 equal docs)"
          a={{ x: DISC_FLIP.geo.top_heavy, y: DISC_FLIP.geo.deep }} b={{ x: DISC_FLIP.log.top_heavy, y: DISC_FLIP.log.deep }}
          labelA="top_heavy" labelB="deep" condA="steep geometric" condB="log₂ (heavy tail)"
          winnerA="top_heavy" winnerB="deep" />
      </div>
    </div>
  );
}

// ===== Panel D — NDCG as estimator =================================================================
function EstimatorPanel({ pairKey, setPairKey, n, setN }: {
  pairKey: 'clearest' | 'closest'; setPairKey: (k: 'clearest' | 'closest') => void; n: number; setN: (v: number) => void;
}) {
  const pair = PAIRS[pairKey];
  const W = 400, H = 150, padL = 40, padR = 16, padT = 24, padB = 28;
  const xMin = 0.55, xMax = 1.0;
  const px = (v: number) => padL + (W - padL - padR) * ((v - xMin) / (xMax - xMin));
  const rows = pair.map((leg) => {
    const arr = PER_Q_NDCG[leg];
    const sd = stdev(arr), m = mean(arr), half = ciHalf(sd, n);
    return { leg, mean: m, half, lo: m - half, hi: m + half };
  });
  const overlap = rows[0].lo <= rows[1].hi && rows[1].lo <= rows[0].hi;
  const sepN = pairKey === 'clearest' ? SEP_CLEAR : SEP_CLOSEST;
  // inset: SE vs n (baked empirical dots for late_interaction + theory curve std/sqrt(n))
  const sdLate = stdev(PER_Q_NDCG.late_interaction);
  const iW = 150, iH = 96, iPadL = 26, iPadB = 20, iPadT = 8, iPadR = 8;
  const nMax = 80, seMax = 0.055;
  const ipx = (nn: number) => iPadL + (iW - iPadL - iPadR) * (nn / nMax);
  const ipy = (se: number) => iH - iPadB - (iH - iPadT - iPadB) * (se / seMax);
  const theory = Array.from({ length: 16 }, (_v, i) => { const nn = 5 + i * 5; return [nn, sdLate / Math.sqrt(nn)] as [number, number]; });
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.4rem', flexWrap: 'wrap' }}>
        <button type="button" style={pill(pairKey === 'clearest')} onClick={() => setPairKey('clearest')}>clearest pair</button>
        <button type="button" style={pill(pairKey === 'closest')} onClick={() => setPairKey('closest')}>closest pair</button>
      </div>
      <Slider label="query count n" value={n} min={5} max={N_QUERIES} step={1} onChange={(v) => setN(Math.round(v))} display={`${n}`} />
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="two legs' mean NDCG with 95 percent confidence intervals" style={{ width: '100%', maxWidth: W, height: 'auto', flex: '1 1 320px' }}>
          <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
          {[0.6, 0.7, 0.8, 0.9, 1.0].map((v) => (<text key={v} x={px(v)} y={H - padB + 14} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
          <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">mean NDCG</text>
          {rows.map((r, i) => {
            const y = padT + i * 46;
            return (
              <g key={r.leg}>
                <line x1={px(r.lo)} y1={y} x2={px(r.hi)} y2={y} stroke={LEG_COLOR[r.leg]} strokeWidth={2.4} />
                <line x1={px(r.lo)} y1={y - 6} x2={px(r.lo)} y2={y + 6} stroke={LEG_COLOR[r.leg]} strokeWidth={2} />
                <line x1={px(r.hi)} y1={y - 6} x2={px(r.hi)} y2={y + 6} stroke={LEG_COLOR[r.leg]} strokeWidth={2} />
                <circle cx={px(r.mean)} cy={y} r={4} fill={LEG_COLOR[r.leg]} />
                <text x={px(r.mean)} y={y - 10} textAnchor="middle" fontSize={9} fill={LEG_COLOR[r.leg]} fontFamily="var(--font-sans)">{LEG_LABEL[r.leg]}</text>
              </g>
            );
          })}
        </svg>
        <svg viewBox={`0 0 ${iW} ${iH}`} role="img" aria-label="standard error versus query count" style={{ width: iW, height: 'auto' }}>
          <line x1={iPadL} y1={iH - iPadB} x2={iW - iPadR} y2={iH - iPadB} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={iPadL} y1={iPadT} x2={iPadL} y2={iH - iPadB} stroke="var(--color-border)" strokeWidth={1} />
          <path d={theory.map(([nn, se], i) => (i ? 'L' : 'M') + ipx(nn) + ' ' + ipy(se)).join(' ')} fill="none" stroke={MUTED} strokeWidth={1.4} />
          {SE_SCALING.filter(([nn]) => nn <= nMax).map(([nn, se], i) => (<circle key={i} cx={ipx(nn)} cy={ipy(se)} r={2.6} fill={LEG_COLOR.late_interaction} />))}
          <circle cx={ipx(Math.min(n, nMax))} cy={ipy(sdLate / Math.sqrt(n))} r={3.6} fill={ACCENT} />
          <text x={iPadL} y={6} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">SE</text>
          <text x={iW - iPadR} y={iH - iPadB + 13} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">n →</text>
          <text x={(iPadL + iW) / 2} y={iPadT + 2} textAnchor="middle" fontSize={7.5} fill={MUTED} fontFamily="var(--font-sans)">σ̂/√n</text>
        </svg>
      </div>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label={`${LEG_LABEL[pair[0]]} ± 95% CI`} value={`${fmt(rows[0].mean, 3)} ± ${fmt(rows[0].half, 3)}`} />
        <Readout label={`${LEG_LABEL[pair[1]]} ± 95% CI`} value={`${fmt(rows[1].mean, 3)} ± ${fmt(rows[1].half, 3)}`} />
        <Readout label={`at n = ${n}`} value={overlap ? 'CIs overlap — tied' : 'CIs separate'} accent={!overlap} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Mean NDCG is a <strong>sample mean</strong> with standard error σ̂/√n (inset, 1/√n decay). The{' '}
        <strong>{pairKey}</strong> pair — gap {fmt(Math.abs(rows[0].mean - rows[1].mean), 3)} —{' '}
        {pairKey === 'clearest'
          ? <>separates by <strong>n = {sepN}</strong> queries: a winner you can report.</>
          : <>has intervals that <strong>overlap across all {N_QUERIES} queries</strong> and would need roughly <strong>{sepN}</strong> to resolve.</>}
        {' '}An observed gap is not real until it clears sampling error — the question a significance test settles.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'graded' | 'discount' | 'convention' | 'estimator';
const TEX: Record<Panel, string> = {
  graded: '\\mathrm{NDCG@}k = \\frac{\\mathrm{DCG@}k}{\\mathrm{IDCG@}k}, \\quad \\mathrm{DCG@}k = \\sum_{i=1}^{k}\\frac{\\mathrm{gain}(g_i)}{\\log_2(i+1)}',
  discount: '\\mathrm{disc}_{\\log}(i) = \\frac{1}{\\log_2(i+1)}, \\quad \\mathrm{disc}_{\\mathrm{geo}}(i) = p^{\\,i-1}, \\quad \\mathbb{E}[\\text{docs}] = \\tfrac{1}{1-p}',
  convention: '\\mathrm{gain}_{\\exp}(g) = 2^{g}-1 \\;\\;\\text{vs}\\;\\; \\mathrm{gain}_{\\mathrm{lin}}(g) = g',
  estimator: '\\widehat{\\mathrm{SE}} = \\frac{\\hat\\sigma}{\\sqrt{n}}, \\quad \\mathrm{CI}_{95} = \\overline{\\mathrm{NDCG}} \\pm 1.96\\,\\widehat{\\mathrm{SE}}',
};

export default memo(function NDCGLaboratory() {
  const [panel, setPanel] = useState<Panel>('graded');
  const [leg, setLeg] = useState<Leg>('dense');
  const [expGain, setExpGain] = useState(true);
  const [p, setP] = useState(RBP_P_DEFAULT);
  const [conv, setConv] = useState<Conv>('exp_log');
  const [pairKey, setPairKey] = useState<'clearest' | 'closest'>('clearest');
  const [n, setN] = useState(8);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="ndcg" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'graded')} onClick={() => setPanel('graded')}>A · graded relevance</button>
        <button type="button" style={pill(panel === 'discount')} onClick={() => setPanel('discount')}>B · discount geometry</button>
        <button type="button" style={pill(panel === 'convention')} onClick={() => setPanel('convention')}>C · convention flip</button>
        <button type="button" style={pill(panel === 'estimator')} onClick={() => setPanel('estimator')}>D · NDCG as estimator</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'graded' && <GradedPanel leg={leg} setLeg={setLeg} expGain={expGain} setExpGain={setExpGain} />}
      {panel === 'discount' && <DiscountPanel p={p} setP={setP} />}
      {panel === 'convention' && <ConventionPanel conv={conv} setConv={setConv} />}
      {panel === 'estimator' && <EstimatorPanel pairKey={pairKey} setPairKey={setPairKey} n={n} setN={setN} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_DOCS} synthetic finance documents, {N_QUERIES} queries; relevance is graded by global tertiles of the
        exact-MaxSim oracle score over the top-{R_SIZE} (so grade ≥ 1 is the binary relevant set). Numbers mirror{' '}
        <code>ndcg_discount_geometry.py</code>; the lab recomputes DCG/IDCG/NDCG, the discount curves, the mean, and
        the CI in closed form.
      </p>
    </div>
  );
});
