import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Retrieval-vs-Long-Context Laboratory — four panels for the `retrieval-vs-long-context` topic:
 *   A. Attention is quadratic. cost(k) = (kL)^2 grows as the square of the context; a k-slider walks from
 *      "retrieve the answer" (small k) to "stuff the window" (k = n), and the cost ratio explodes.
 *   B. More context is not better. With the answer reliably retrieved (recall@1 ~ 1), answer quality Q(k)
 *      peaks at the SMALLEST covering context and declines as k grows: relevant passages past the first are
 *      redundant (precision ~ 1, Q flat), then same-sector distractors dilute the budget (precision falls,
 *      answer entropy H rises). A dashed hard-retrieval curve shows the contrast.
 *   C. The rate–distortion frontier. Plot Q against cost: the focused-retrieval point sits up and to the
 *      left and Pareto-dominates stuffing the window — quadratically more compute for strictly worse answers.
 *   D. Lost in the middle. A relevant passage buried mid-context is read at attenuated attention weight
 *      u(pos) — a SOFT ERASURE — and answer quality dips when the gold sits in the center.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every quality / entropy / recall / precision number below is mirrored
 * TO THE DECIMAL from notebooks/retrieval-vs-long-context/retrieval_vs_long_context.py (viz_constants()).
 * Matching asserts: test_attention_cost_quadratic / test_more_context_hurts / test_less_is_more_mechanism /
 * test_retrieval_quality_regimes / test_buried_gold_soft_erasure / test_pareto_dominates_stuff_all. The lab
 * recomputes ONLY closed forms in TS (the quadratic cost (kL)^2 and the positional weight u(pos)); Q, H,
 * recall, precision, util, and the positional quality curve are MODEL OUTPUTS and are baked. Change a number
 * here -> change it in the .py and re-run.
 */

// --- baked from viz_constants() -------------------------------------------------------
const K = 16;                       // companies = answers
const R_RELEVANT = 4;               // relevant passages per company (the relevant set)
const N_QUERIES = 32;
const TAU_GEN = 0.3;
const TAU_ATTN = 0.45;
const PASSAGE_LEN = 256;            // tokens per passage (the cost axis)
const KAPPA_QUERY = 60.0;           // good retrieval (recall@1 ~ 1)
const KAPPA_QUERY_HARD = 15.0;      // hard-retrieval contrast (recall@1 ~ 0.47)

const K_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const Q = [0.71, 0.7041, 0.6985, 0.6903, 0.6501, 0.6211, 0.5972, 0.5789, 0.5644, 0.5516, 0.5408, 0.5317, 0.5227, 0.5167, 0.5109, 0.5054];
const ENTROPY = [1.5378, 1.559, 1.5781, 1.6101, 1.7292, 1.8154, 1.8781, 1.9306, 1.9693, 2.0114, 2.0467, 2.0802, 2.1088, 2.1324, 2.1555, 2.1775];
const RECALL_R = [0.25, 0.4844, 0.7188, 0.9375, 0.9453, 0.9766, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 1.0, 1.0, 1.0, 1.0];
const PRECISION = [1.0, 0.9688, 0.9583, 0.9375, 0.7563, 0.651, 0.567, 0.4961, 0.441, 0.3969, 0.3608, 0.3307, 0.3077, 0.2857, 0.2667, 0.25];
const HARD_Q = [0.3584, 0.3856, 0.3862, 0.3763, 0.3476, 0.3298, 0.3187, 0.312, 0.3024, 0.2928, 0.2875, 0.283, 0.2804, 0.275, 0.271, 0.2679];
const K_STAR = 1;                   // the optimum: the smallest covering context
const Q_1 = 0.71, Q_N = 0.5054;
const LOG2K = 4.0;                  // log2(16) — entropy ceiling
const N_CTX = 16;
const POS_DIP = 0.55;
const POS_Q = [0.5316, 0.5311, 0.5302, 0.5289, 0.5254, 0.5227, 0.5208, 0.5198, 0.5198, 0.5208, 0.5226, 0.5253, 0.5287, 0.5323, 0.5364, 0.5406];
const SAT_STANDALONE = 0.4111, SAT_REDUNDANT = 0.0812, SAT_NOVEL = 0.2472;
const SAME_SECTOR_COSINE = 0.5974;

const ACCENT = 'var(--color-accent)';
const POS = '#5fa873';              // recall / coverage
const NEG = '#c25b6b';              // entropy / cost / the dilution
const FLOOR = '#6c8cd5';            // the optimum / positional weight
const MUTED = '#9aa3ad';

const fmt = (x: number, n = 3) => x.toFixed(n);
const r2 = (x: number) => Math.round(x * 100) / 100;

// --- closed-form TS recomputation (the ONLY things not baked) -------------------------
const cost = (k: number) => (k * PASSAGE_LEN) ** 2;                       // attention FLOPs, exact
const positionalWeight = (pos: number, n = N_CTX, dip = POS_DIP) =>       // the U, mirrors the .py
  n <= 1 ? 1 : 1 - dip * Math.sin((Math.PI * (pos + 0.5)) / n);

// polyline from (i -> value) with pixel mappers.
const poly = (vals: number[], px: (i: number) => number, py: (v: number) => number) =>
  vals.map((v, i) => `${i ? 'L' : 'M'}${r2(px(i))},${r2(py(v))}`).join(' ');

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
      <span style={{ minWidth: '15rem' }}>{label} = <strong>{display}</strong></span>
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

// ===== Panel A — attention is quadratic =============================================================
function CostPanel({ k, setK }: { k: number; setK: (v: number) => void }) {
  const W = 520, H = 210, padL = 48, padR = 14, padT = 18, padB = 34;
  const n = K_GRID.length;
  const costMax = cost(n);
  const px = (i: number) => padL + (W - padL - padR) * (i / (n - 1));
  const py = (c: number) => H - padB - (H - padT - padB) * (c / costMax);
  const i = k - 1;
  const ratio = cost(k) / cost(1);
  return (
    <div>
      <Slider label="context depth k (passages read)" value={k} min={1} max={n} step={1}
        onChange={(v) => setK(Math.round(v))} display={`${k}  (${k * PASSAGE_LEN} tokens)`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="attention cost growing quadratically with context depth" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={padL - 5} y={py(f * costMax) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{Math.round((f * costMax) / 1e6)}M</text>
        ))}
        {/* the quadratic curve */}
        <path d={poly(K_GRID.map((kk) => cost(kk)), px, py)} fill="none" stroke={NEG} strokeWidth={2.4} />
        <path d={`${poly(K_GRID.map((kk) => cost(kk)), px, py)} L${r2(px(n - 1))},${r2(py(0))} L${r2(px(0))},${r2(py(0))} Z`} fill={NEG} fillOpacity={0.08} />
        {/* retrieve-k* vs stuff-n markers */}
        <circle cx={px(K_STAR - 1)} cy={py(cost(K_STAR))} r={4} fill={FLOOR} />
        <text x={px(K_STAR - 1) + 6} y={py(cost(K_STAR)) - 4} fontSize={8.5} fill={FLOOR} fontFamily="var(--font-sans)">retrieve k* = {K_STAR}</text>
        <circle cx={px(n - 1)} cy={py(cost(n))} r={4} fill={NEG} />
        <text x={px(n - 1) - 4} y={py(cost(n)) + 12} textAnchor="end" fontSize={8.5} fill={NEG} fontFamily="var(--font-sans)">stuff the window</text>
        {/* slider marker */}
        <circle cx={px(i)} cy={py(cost(k))} r={3.5} fill={ACCENT} />
        <line x1={px(i)} y1={py(cost(k))} x2={px(i)} y2={H - padB} stroke={ACCENT} strokeWidth={1} strokeDasharray="2 3" />
        {K_GRID.filter((_v, j) => j % 3 === 0).map((kk) => (
          <text key={kk} x={px(kk - 1)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{kk}</text>
        ))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">context depth k</text>
        <text x={padL} y={11} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">attention cost (kL)² — FLOP-units (millions)</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
        <Readout label="tokens attended (kL)" value={`${k * PASSAGE_LEN}`} />
        <Readout label="cost (kL)²" value={`${(cost(k) / 1e6).toFixed(2)}M`} color={NEG} />
        <Readout label="× cost vs k = 1" value={`${ratio.toFixed(0)}×`} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Full self-attention forms a <strong>(kL) × (kL)</strong> score matrix, so the arithmetic cost is{' '}
        <span style={{ color: NEG }}>Θ((kL)²)</span> — doubling the context quadruples the compute. Stuffing the whole
        window (k = {K_GRID[K_GRID.length - 1]}) costs <strong>{(cost(K_GRID.length) / cost(1)).toFixed(0)}×</strong> what
        reading the answer (k = {K_STAR}) does. This is the <em>rate</em> the long-context option pays; FlashAttention
        lowers the <em>memory</em> to O(n) but leaves these FLOPs unchanged.
      </p>
    </div>
  );
}

// ===== Panel B — more context is not better =========================================================
function QualityPanel({ k, setK }: { k: number; setK: (v: number) => void }) {
  const W = 520, H = 232, padL = 36, padR = 40, padT = 18, padB = 34;
  const n = K_GRID.length;
  const px = (i: number) => padL + (W - padL - padR) * (i / (n - 1));
  const py = (v: number) => H - padB - (H - padT - padB) * v;             // [0,1]: Q / recall / precision
  const ph = (h: number) => H - padB - (H - padT - padB) * (h / LOG2K);   // [0, log2K]: entropy (right axis)
  const i = k - 1;
  return (
    <div>
      <Slider label="context depth k (passages read)" value={k} min={1} max={n} step={1}
        onChange={(v) => setK(Math.round(v))} display={`${k}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="answer quality declining and answer entropy rising as more passages are read" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* two-regime shading: all-relevant (redundant) up to R, then distractors dilute */}
        <rect x={px(0)} y={padT} width={px(R_RELEVANT - 1) - px(0)} height={H - padT - padB} fill={POS} fillOpacity={0.06} />
        <rect x={px(R_RELEVANT - 1)} y={padT} width={px(n - 1) - px(R_RELEVANT - 1)} height={H - padT - padB} fill={NEG} fillOpacity={0.06} />
        <text x={(px(0) + px(R_RELEVANT - 1)) / 2} y={padT + 9} textAnchor="middle" fontSize={7.5} fill={POS} fontFamily="var(--font-sans)">all relevant (redundant)</text>
        <text x={(px(R_RELEVANT - 1) + px(n - 1)) / 2} y={padT + 9} textAnchor="middle" fontSize={7.5} fill={NEG} fontFamily="var(--font-sans)">distractors dilute</text>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (<text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
        {/* recall (rising) and precision (falling) — context, muted */}
        <path d={poly(RECALL_R, px, py)} fill="none" stroke={POS} strokeWidth={1.5} strokeDasharray="5 3" />
        <path d={poly(PRECISION, px, py)} fill="none" stroke={MUTED} strokeWidth={1.5} strokeDasharray="2 3" />
        {/* entropy (right axis, rising) */}
        <path d={poly(ENTROPY, px, ph)} fill="none" stroke={NEG} strokeWidth={1.8} />
        {/* hard-retrieval Q (dashed contrast) */}
        <path d={poly(HARD_Q, px, py)} fill="none" stroke={ACCENT} strokeWidth={1.4} strokeDasharray="3 3" opacity={0.55} />
        {/* the headline: Q(k), bold, declining */}
        <path d={poly(Q, px, py)} fill="none" stroke={ACCENT} strokeWidth={2.6} />
        {/* k* peak marker */}
        <circle cx={px(K_STAR - 1)} cy={py(Q[K_STAR - 1])} r={4} fill={FLOOR} />
        <text x={px(K_STAR - 1) + 5} y={py(Q[K_STAR - 1]) - 5} fontSize={8.5} fill={FLOOR} fontFamily="var(--font-sans)">k* peak</text>
        {/* slider marker */}
        <circle cx={px(i)} cy={py(Q[i])} r={3.5} fill={ACCENT} />
        <line x1={px(i)} y1={padT} x2={px(i)} y2={H - padB} stroke="var(--color-text)" strokeWidth={0.8} strokeDasharray="2 3" />
        {K_GRID.filter((_v, j) => j % 3 === 0).map((kk) => (
          <text key={kk} x={px(kk - 1)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{kk}</text>
        ))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">context depth k</text>
        <text x={padL + 4} y={padT + 20} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={ACCENT}>▬ Q(k) quality</tspan> · <tspan fill={NEG}>▬ entropy H</tspan> · <tspan fill={POS}>┄ recall</tspan> · <tspan fill={MUTED}>┄ precision</tspan>
        </text>
        <text x={W - padR + 3} y={ph(LOG2K) + 3} fontSize={8} fill={NEG} fontFamily="var(--font-sans)">{fmt(LOG2K, 0)}</text>
        <text x={W - padR + 3} y={ph(0) + 3} fontSize={8} fill={NEG} fontFamily="var(--font-sans)">0</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="answer quality Q(k)" value={fmt(Q[i])} accent />
        <Readout label="answer entropy H" value={`${fmt(ENTROPY[i])} bits`} color={NEG} />
        <Readout label="recall_R / precision" value={`${fmt(RECALL_R[i], 2)} / ${fmt(PRECISION[i], 2)}`} />
        <Readout label="vs peak Q(k*)" value={`${(Q[i] - Q_1 >= 0 ? '+' : '') + fmt(Q[i] - Q_1)}`} color={Q[i] < Q_1 ? NEG : POS} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Even though the answer is reliably retrieved, <span style={{ color: ACCENT }}>quality Q(k)</span> is highest at the{' '}
        <strong>smallest covering context</strong> and falls as k grows. While the top-k is all-relevant the extra passages
        are <strong>redundant</strong> (precision ≈ 1, Q nearly flat); past k = {R_RELEVANT} same-sector distractors enter, the
        attention budget bleeds off the answer, and <span style={{ color: NEG }}>entropy rises</span> toward the prior. Chasing
        <span style={{ color: POS }}> recall</span> by enlarging k buys nothing — you need the answer, not the whole set. The
        faint <span style={{ color: ACCENT }}>dashed</span> curve is the same model under <em>hard</em> retrieval (κ = {KAPPA_QUERY_HARD}):
        lower throughout, and only there does a small interior optimum appear.
      </p>
    </div>
  );
}

// ===== Panel C — the rate–distortion frontier =======================================================
function FrontierPanel({ k, setK }: { k: number; setK: (v: number) => void }) {
  const W = 520, H = 224, padL = 40, padR = 16, padT = 18, padB = 40;
  const n = K_GRID.length;
  const lx = (c: number) => Math.log(c);
  const xmin = lx(cost(1)), xmax = lx(cost(n));
  const px = (c: number) => padL + (W - padL - padR) * ((lx(c) - xmin) / (xmax - xmin));
  const qmin = 0.45, qmax = 0.75;
  const py = (q: number) => H - padB - (H - padT - padB) * ((q - qmin) / (qmax - qmin));
  const i = k - 1;
  return (
    <div>
      <Slider label="context depth k (rate = cost)" value={k} min={1} max={n} step={1}
        onChange={(v) => setK(Math.round(v))} display={`${k}  ((kL)² = ${(cost(k) / 1e6).toFixed(2)}M)`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="quality versus cost frontier — focused retrieval dominates stuffing the window" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0.5, 0.6, 0.7].map((v) => (<text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
        {/* dominated region: anything below-and-right of k* */}
        <rect x={px(cost(1))} y={py(Q[0])} width={W - padR - px(cost(1))} height={(H - padB) - py(Q[0])} fill={NEG} fillOpacity={0.06} />
        <text x={(px(cost(1)) + W - padR) / 2} y={H - padB - 6} textAnchor="middle" fontSize={7.5} fill={NEG} fontFamily="var(--font-sans)">dominated: more cost, less quality</text>
        {/* good-retrieval frontier Q vs cost */}
        <path d={Q.map((q, j) => `${j ? 'L' : 'M'}${r2(px(cost(K_GRID[j])))},${r2(py(q))}`).join(' ')} fill="none" stroke={ACCENT} strokeWidth={2.4} />
        {Q.map((q, j) => (<circle key={j} cx={r2(px(cost(K_GRID[j])))} cy={r2(py(q))} r={2.2} fill={ACCENT} />))}
        {/* k* (retrieve) and stuff markers */}
        <circle cx={px(cost(K_STAR))} cy={py(Q[K_STAR - 1])} r={4.5} fill={FLOOR} />
        <text x={px(cost(K_STAR)) + 6} y={py(Q[K_STAR - 1]) - 5} fontSize={8.5} fill={FLOOR} fontFamily="var(--font-sans)">k* — retrieve</text>
        <circle cx={px(cost(n))} cy={py(Q_N)} r={4.5} fill={NEG} />
        <text x={px(cost(n)) - 4} y={py(Q_N) - 6} textAnchor="end" fontSize={8.5} fill={NEG} fontFamily="var(--font-sans)">stuff the window</text>
        {/* slider marker */}
        <circle cx={px(cost(k))} cy={py(Q[i])} r={3.5} fill="var(--color-text)" />
        <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">attention cost (kL)² — log scale (rate →)</text>
        <text x={padL + 4} y={padT + 8} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">answer quality Q (↑ better)</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="at k* — quality / cost" value={`${fmt(Q_1)} / ${(cost(K_STAR) / 1e6).toFixed(2)}M`} color={FLOOR} />
        <Readout label="stuffing — quality / cost" value={`${fmt(Q_N)} / ${(cost(N_CTX) / 1e6).toFixed(2)}M`} color={NEG} />
        <Readout label="you trade" value={`${(cost(N_CTX) / cost(K_STAR)).toFixed(0)}× cost for −${fmt(Q_1 - Q_N)} Q`} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Reading more passages trades <strong>rate</strong> (attention compute) for <strong>distortion</strong> (answer error).
        The focused-retrieval point sits <span style={{ color: FLOOR }}>up and to the left</span> and{' '}
        <strong>Pareto-dominates</strong> stuffing the window: every passage past k* costs quadratically more and answers no
        better. The right move is not a bigger window but a <em>better-chosen</em> one — the gate to context selection.
      </p>
    </div>
  );
}

// ===== Panel D — lost in the middle =================================================================
function PositionPanel({ pos, setPos }: { pos: number; setPos: (v: number) => void }) {
  const W = 520, H = 224, padL = 40, padR = 44, padT = 18, padB = 36;
  const n = N_CTX;
  const px = (i: number) => padL + (W - padL - padR) * (i / (n - 1));
  const pw = (w: number) => H - padB - (H - padT - padB) * w;            // weight in [0,1] (left axis)
  const qmin = 0.50, qmax = 0.55;
  const pq = (q: number) => H - padB - (H - padT - padB) * ((q - qmin) / (qmax - qmin));   // zoomed Q (right)
  const weights = Array.from({ length: n }, (_v, i) => positionalWeight(i));
  return (
    <div>
      <Slider label="position of the gold passage in the context" value={pos} min={0} max={n - 1} step={1}
        onChange={(v) => setPos(Math.round(v))} display={`${pos} of ${n - 1}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="a buried gold passage is read at low attention weight and answer quality dips" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (<text key={v} x={padL - 5} y={pw(v) + 3} textAnchor="end" fontSize={8} fill={FLOOR} fontFamily="var(--font-sans)">{v}</text>))}
        {/* the dramatic positional-weight U (the empirical input) */}
        <path d={poly(weights, px, pw)} fill="none" stroke={FLOOR} strokeWidth={2.2} />
        <path d={`${poly(weights, px, pw)} L${r2(px(n - 1))},${r2(pw(0))} L${r2(px(0))},${r2(pw(0))} Z`} fill={FLOOR} fillOpacity={0.07} />
        {/* the resulting quality dip (zoomed right axis) */}
        <path d={poly(POS_Q, px, pq)} fill="none" stroke={NEG} strokeWidth={2.2} />
        {POS_Q.map((q, i) => (<circle key={i} cx={r2(px(i))} cy={r2(pq(q))} r={2} fill={NEG} />))}
        {/* slider marker on both curves */}
        <line x1={px(pos)} y1={padT} x2={px(pos)} y2={H - padB} stroke="var(--color-text)" strokeWidth={0.8} strokeDasharray="2 3" />
        <circle cx={px(pos)} cy={pw(positionalWeight(pos))} r={3.5} fill={FLOOR} />
        <circle cx={px(pos)} cy={pq(POS_Q[pos])} r={3.5} fill={NEG} />
        {[0, 4, 8, 12, 15].map((p) => (<text key={p} x={px(p)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{p}</text>))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">gold passage position (start → middle → end)</text>
        <text x={padL + 4} y={padT + 8} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={FLOOR}>▬ attention weight u(pos)</tspan> · <tspan fill={NEG}>▬ answer quality (zoomed)</tspan>
        </text>
        <text x={W - padR + 3} y={pq(qmax) + 3} fontSize={8} fill={NEG} fontFamily="var(--font-sans)">{fmt(qmax, 2)}</text>
        <text x={W - padR + 3} y={pq(qmin) + 3} fontSize={8} fill={NEG} fontFamily="var(--font-sans)">{fmt(qmin, 2)}</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="attention weight u(pos)" value={fmt(positionalWeight(pos))} color={FLOOR} />
        <Readout label="answer quality here" value={fmt(POS_Q[pos])} color={NEG} />
        <Readout label="end vs middle Q" value={`${fmt(POS_Q[0])} → ${fmt(POS_Q[Math.floor(n / 2)])}`} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Transformers read the <strong>middle</strong> of a long context at attenuated attention (lost-in-the-middle, Liu et
        al. 2024). We model that as a U-shaped weight <span style={{ color: FLOOR }}>u(pos)</span> — about{' '}
        {fmt(1 - POS_DIP, 2)} at the center versus 1 at the ends. A relevant passage buried mid-context is read at low weight:
        a <strong>soft erasure</strong> that drops <span style={{ color: NEG }}>answer quality</span> even though the passage
        was retrieved. The U-shape is an <em>empirical</em> property baked in, not derived.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'cost' | 'quality' | 'frontier' | 'position';
const TEX: Record<Panel, string> = {
  cost: '\\text{cost}(k) = (kL)^2 = \\Theta(k^2) \\qquad\\text{(arithmetic; FlashAttention: } O(n)\\text{ memory)}',
  quality: 'Q(k) = \\mathbb{E}\\big[\\,p(a^\\star \\mid q,\\ \\text{top-}k)\\,\\big], \\qquad \\textstyle\\sum_{j} w_j = 1',
  frontier: '\\min_{k}\\ \\text{cost}(k)=(kL)^2 \\ \\ \\text{s.t.}\\ \\ Q(k)\\ \\text{high} \\;\\Longrightarrow\\; k^\\star \\ll n',
  position: 'w_j \\;\\propto\\; \\text{rel}_j \\cdot u(\\text{pos}_j), \\qquad u(\\text{middle}) \\;<\\; u(\\text{ends})',
};

export default memo(function RetrievalVsLongContextLaboratory() {
  const [panel, setPanel] = useState<Panel>('cost');
  const [k, setK] = useState(4);
  const [pos, setPos] = useState(N_CTX - 1);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="retrieval-vs-long-context" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'cost')} onClick={() => setPanel('cost')}>A · attention cost</button>
        <button type="button" style={pill(panel === 'quality')} onClick={() => setPanel('quality')}>B · more ≠ better</button>
        <button type="button" style={pill(panel === 'frontier')} onClick={() => setPanel('frontier')}>C · rate–distortion</button>
        <button type="button" style={pill(panel === 'position')} onClick={() => setPanel('position')}>D · lost in the middle</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'cost' && <CostPanel k={k} setK={setK} />}
      {panel === 'quality' && <QualityPanel k={k} setK={setK} />}
      {panel === 'frontier' && <FrontierPanel k={k} setK={setK} />}
      {panel === 'position' && <PositionPanel pos={pos} setPos={setPos} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        Finance vMF corpus extended from the dense-retrieval geometry: {K} companies across 4 sectors, {R_RELEVANT} relevant
        passages each (same-sector cosine ≈ {fmt(SAME_SECTOR_COSINE, 2)}), {N_QUERIES} queries; the answer model is a synthetic
        softmax stand-in with a finite attention budget (τ = {TAU_GEN}, τ_attn = {TAU_ATTN}), not an LLM. Queries identify the
        company (good retrieval, κ = {fmt(KAPPA_QUERY, 0)}). Diminishing returns (imported): a first filing moves belief{' '}
        {fmt(SAT_STANDALONE)} bits, a genuinely novel one {fmt(SAT_NOVEL)}, but a redundant copy only {fmt(SAT_REDUNDANT)}.
        Numbers mirror <code>retrieval_vs_long_context.py</code>; the lab recomputes only the quadratic cost and the positional weight.
      </p>
    </div>
  );
});
