import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Multi-Hop Laboratory — four panels for the `multi-hop-iterative-retrieval` topic:
 *   A. The compositional gap. On the "cosine to the current query" axis the bridge filing sits above the edge
 *      threshold (retrievable) while the answer sits below it (invisible in one hop) — until reformulation
 *      moves the answer to ~1. Single-hop answer recall ≈ 0, multi-hop ≈ 1.
 *   B. Compounding recall. End-to-end success is the PRODUCT of per-hop recalls (geometric decay); to hold a
 *      target ρ each hop must over-retrieve to ρ^(1/k). Positive dependence (FKG) makes the product a
 *      conservative LOWER bound — the realized recall sits above it.
 *   C. The stopping rule. A worked 3-hop trajectory: the residual (the new direction each filing opens) clears
 *      the threshold through every bridge hop and collapses at the terminal answer filing, so the adaptive hop
 *      count equals the chain depth. The belief-movement (KL) is tiny at the bridge and huge at the answer.
 *   D. The supermodular synergy. The answer document alone is worth 0 bits; given the bridge it is worth 1 —
 *      the marginal INCREASES with conditioning (the XOR witness). So single-shot selection cannot reach the
 *      answer (it is never in the one-hop pool); only reformulating from the bridge harvests the synergy.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every cosine / recall / residual / KL / FKG number below is mirrored TO
 * THE DECIMAL from notebooks/multi-hop-iterative-retrieval/multi_hop_iterative_retrieval.py (viz_constants()).
 * Matching asserts: test_geodesic_inequalities / test_compositional_gap / test_chain_recall_product_anchor /
 * test_fkg_lower_bound / test_adaptive_stopping_pinned / test_supermodular_synergy. The lab recomputes ONLY
 * closed forms in TS (the running product ∏rᵢ, ρ^(1/k), the over-fetch reciprocal, the threshold crossing,
 * pixel/angle maps); cosines, recalls, the trajectory, and the FKG sweep are MODEL OUTPUTS and are baked.
 */

// --- baked from viz_constants() -------------------------------------------------------
const TAU_HOP = 0.35;                 // the retrieval-graph edge threshold on cosine
const ALPHA_MENTION_DEG = 40.0;
const K_RETRIEVE = 3;
const REFORM_EPS = 0.47;              // stop when the read filing opens no new direction

// Panel A — the compositional gap (cosines to the query + recall)
const Q_DOT_BRIDGE = 0.816;          // = cos α adjusted for non-orthogonal companies: the bridge is retrievable
const Q_DOT_ANSWER = 0.193;          // below τ: the answer is invisible in one hop
const BRIDGE_DOT_ANSWER = 0.725;     // the mention the bridge carries
const REFORMULATED_DOT_ANSWER = 0.981; // after reformulation the answer is reachable
const RECALL_SINGLE_HOP = 0.0;
const RECALL_MULTI_HOP = 1.0;
const GRAPH_DIST_ANSWER = 2;

// Panel B — compounding recall + FKG + over-fetch
const CHAIN_SUCCESS = 0.3;           // = 0.6·0.5 (illustrative per-hop retentions)
const OVER_FETCH = 3.333;            // = 1/(R1·R2)
const FKG_SWEEP = [
  { rho: -0.6, Rtrue: 0.198, Rindep: 0.3, gap: -0.102 },
  { rho: -0.4, Rtrue: 0.234, Rindep: 0.3, gap: -0.066 },
  { rho: -0.2, Rtrue: 0.267, Rindep: 0.3, gap: -0.033 },
  { rho: 0.0, Rtrue: 0.298, Rindep: 0.3, gap: -0.002 },
  { rho: 0.2, Rtrue: 0.332, Rindep: 0.3, gap: 0.032 },
  { rho: 0.4, Rtrue: 0.366, Rindep: 0.3, gap: 0.066 },
  { rho: 0.6, Rtrue: 0.4, Rindep: 0.3, gap: 0.1 },
];

// Panel C — the worked 3-hop trajectory (bridge, bridge, answer) + adaptive stopping
const TRAJ3_RESID = [0.543, 0.589, 0.397];          // > REFORM_EPS until the terminal hop
const TRAJ3_KL_MOVES = [0.13, 8.015, 7.709];        // bridge hops small, answer hop large
const HOPS_BY_CLASS: Record<string, number> = { '1-hop': 1, '2-hop': 2, '3-hop': 3 };

// Panel D — the supermodular synergy
const XOR_DELTA_EMPTY = 0.0;         // bits the answer document adds alone
const XOR_DELTA_GIVEN_BRIDGE = 1.0;  // ... given the bridge — INCREASES (supermodular)
const ANSWER_IN_SINGLE_POOL = 0.0;   // single-shot cannot reach the answer
const ANSWER_IN_REFORM_POOL = 1.0;   // one reformulation reaches it
const BRIDGE_KL = 0.081;             // the bridge hop's belief movement (myopically worthless)
const ANSWER_KL = 9.168;             // the answer hop's belief movement (decisive — and LAST)

// --- palette --------------------------------------------------------------------------
const ACCENT = 'var(--color-accent)';
const BRIDGEC = '#d9a23b';           // the bridge
const ANSWERC = '#5fa873';           // the answer
const MISS = '#c25b6b';              // unreachable / failure
const REACH = '#6c8cd5';            // reachable / over-fetch
const MUTED = '#9aa3ad';

const fmt = (x: number, n = 3) => x.toFixed(n);
const r2 = (x: number) => Math.round(x * 100) / 100;
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
const miniPill = (active: boolean, color: string) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.72rem', padding: '0.2rem 0.55rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? color : 'var(--color-border)'}`,
  background: active ? color : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

// ===== Panel A — the compositional gap ==============================================================
function GapPanel({ reformulated, setReformulated }: { reformulated: boolean; setReformulated: (v: boolean) => void }) {
  const W = 520, H = 250, padL = 40, padR = 20, padT = 56, padB = 64;
  // a "cosine to the current query" axis: bridge fixed; the answer moves once the query is reformulated
  const cmin = -0.2, cmax = 1.0;
  const cx = (c: number) => padL + (W - padL - padR) * ((c - cmin) / (cmax - cmin));
  const axisY = padT + 44;
  const answerCos = reformulated ? REFORMULATED_DOT_ANSWER : Q_DOT_ANSWER;
  const answerReach = answerCos > TAU_HOP;
  const dot = (c: number, color: string, label: string, sub: string, up: boolean) => (
    <g>
      <line x1={cx(c)} y1={axisY} x2={cx(c)} y2={up ? axisY - 18 : axisY + 18} stroke={color} strokeWidth={1} />
      <circle cx={cx(c)} cy={axisY} r={6} fill={color} />
      <text x={cx(c)} y={up ? axisY - 23 : axisY + 30} textAnchor="middle" fontSize={9} fill={color} fontFamily="var(--font-sans)" fontWeight={700}>{label}</text>
      <text x={cx(c)} y={up ? axisY - 23 - 11 : axisY + 30 + 11} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{sub}</text>
    </g>
  );
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', margin: '0.1rem 0 0.6rem' }}>
        <button type="button" style={miniPill(!reformulated, ACCENT)} onClick={() => setReformulated(false)}>single hop (query = A)</button>
        <button type="button" style={miniPill(reformulated, ANSWERC)} onClick={() => setReformulated(true)}>after reformulating from the bridge</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="cosine of the bridge and answer to the current query, against the edge threshold" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* the retrieval-reachable band: cosine > tau */}
        <rect x={cx(TAU_HOP)} y={padT} width={cx(cmax) - cx(TAU_HOP)} height={56} fill={REACH} fillOpacity={0.08} />
        <text x={cx(cmax) - 4} y={padT + 11} textAnchor="end" fontSize={8} fill={REACH} fontFamily="var(--font-sans)">retrievable (cos &gt; τ)</text>
        {/* axis */}
        <line x1={padL} y1={axisY} x2={W - padR} y2={axisY} stroke="var(--color-border)" strokeWidth={1.2} />
        {[-0.2, 0, 0.35, 0.6, 0.8, 1.0].map((t) => (
          <g key={t}>
            <line x1={cx(t)} y1={axisY - 3} x2={cx(t)} y2={axisY + 3} stroke="var(--color-border)" strokeWidth={1} />
            <text x={cx(t)} y={H - padB + 30} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t}</text>
          </g>
        ))}
        {/* threshold */}
        <line x1={cx(TAU_HOP)} y1={padT} x2={cx(TAU_HOP)} y2={axisY + 18} stroke={MISS} strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={cx(TAU_HOP)} y={padT - 6} textAnchor="middle" fontSize={8.5} fill={MISS} fontFamily="var(--font-sans)">edge threshold τ = {TAU_HOP}</text>
        {dot(Q_DOT_BRIDGE, BRIDGEC, 'bridge', `cos = ${fmt(Q_DOT_BRIDGE, 2)}`, true)}
        {dot(answerCos, answerReach ? ANSWERC : MISS, 'answer', `cos = ${fmt(answerCos, 2)}`, false)}
        <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">cosine to the current query</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="answer reachable now?" value={answerReach ? 'yes ✓' : 'no ✗'} color={answerReach ? ANSWERC : MISS} />
        <Readout label="single-hop answer recall" value={fmt(RECALL_SINGLE_HOP, 2)} color={MISS} />
        <Readout label="multi-hop answer recall" value={fmt(RECALL_MULTI_HOP, 2)} color={ANSWERC} />
        <Readout label="answer graph distance" value={`${GRAPH_DIST_ANSWER} hops`} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        A 2-hop question hides its answer in a filing that is <strong>near-orthogonal to the query</strong> (cosine{' '}
        {fmt(Q_DOT_ANSWER, 2)} &lt; τ): single-hop retrieval never surfaces it, so single-hop answer recall is{' '}
        <span style={{ color: MISS }}>{fmt(RECALL_SINGLE_HOP, 2)}</span>. The <span style={{ color: BRIDGEC }}>bridge</span>{' '}
        filing — company A's, which names the supplier — is retrievable (cosine {fmt(Q_DOT_BRIDGE, 2)}) and carries a{' '}
        component toward the answer ({fmt(BRIDGE_DOT_ANSWER, 2)}). Reformulating from it,{' '}
        <code>q' = normalize(d − ⟨d,q⟩q)</code>, points the next retrieval straight at the answer (cosine jumps to{' '}
        <span style={{ color: ANSWERC }}>{fmt(REFORMULATED_DOT_ANSWER, 2)}</span>), so multi-hop recall is{' '}
        <span style={{ color: ANSWERC }}>{fmt(RECALL_MULTI_HOP, 2)}</span>. The answer sits at graph distance{' '}
        {GRAPH_DIST_ANSWER}: a path, not a point.
      </p>
    </div>
  );
}

// ===== Panel B — compounding recall =================================================================
function CompoundingPanel({ rho, setRho }: { rho: number; setRho: (v: number) => void }) {
  const W = 520, H = 244, padL = 42, padR = 92, padT = 22, padB = 36;
  const KS = [1, 2, 3, 4, 5];
  const px = (i: number) => padL + (W - padL - padR) * (i / (KS.length - 1));
  const py = (v: number) => H - padB - (H - padT - padB) * v;        // recall in [0,1]
  // closed-form TS recomputation (NOT baked): chain recall at a single per-hop rate, and the required ρ^(1/k)
  const baseR = 0.7;
  const chain = KS.map((k) => Math.pow(baseR, k));                   // ∏ r over k hops (geometric)
  const required = KS.map((k) => Math.pow(rho, 1 / k));              // per-hop recall to hold ρ end-to-end
  const overfetch = KS.map((k) => 1 / Math.pow(baseR, k));          // 1/∏r
  const ofMax = overfetch[overfetch.length - 1];
  const pof = (v: number) => H - padB - (H - padT - padB) * Math.min(v / ofMax, 1);
  return (
    <div>
      <Slider label="end-to-end recall target ρ" value={rho} min={0.5} max={0.99} step={0.01}
        onChange={setRho} display={fmt(rho, 2)} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="chain recall decays geometrically while required per-hop recall and over-fetch climb" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={padL - 5} y={py(f) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{f}</text>
        ))}
        {/* chain recall ∏r (declining) */}
        <path d={poly(chain, px, py)} fill="none" stroke={MISS} strokeWidth={2.4} />
        {chain.map((v, i) => (<circle key={i} cx={px(i)} cy={py(v)} r={2.4} fill={MISS} />))}
        {/* required per-hop recall ρ^(1/k) (rising toward 1) */}
        <path d={poly(required, px, py)} fill="none" stroke={ANSWERC} strokeWidth={2.2} strokeDasharray="5 3" />
        {required.map((v, i) => (<circle key={i} cx={px(i)} cy={py(v)} r={2.4} fill={ANSWERC} />))}
        {/* over-fetch on a right axis (climbing) */}
        <path d={poly(overfetch, px, pof)} fill="none" stroke={REACH} strokeWidth={2} />
        {overfetch.map((v, i) => (<circle key={i} cx={px(i)} cy={pof(v)} r={2.2} fill={REACH} />))}
        {KS.map((k, i) => (
          <text key={k} x={px(i)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{k}</text>
        ))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">number of hops k</text>
        <text x={padL + 2} y={padT + 2} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={MISS}>▬ chain recall ∏rᵢ (r=0.7)</tspan> · <tspan fill={ANSWERC}>┄ required ρ^(1/k)</tspan> · <tspan fill={REACH}>▬ over-fetch 1/∏r</tspan>
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.2rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="2-hop chain recall (r=0.7)" value={fmt(chain[1], 2)} color={MISS} />
        <Readout label="required per-hop for ρ at k=3" value={fmt(Math.pow(rho, 1 / 3), 3)} color={ANSWERC} />
        <Readout label="demo R₁·R₂ = 0.6·0.5" value={fmt(CHAIN_SUCCESS, 2)} accent />
        <Readout label="over-fetch 1/(R₁·R₂)" value={`${fmt(OVER_FETCH, 2)}×`} color={REACH} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Each hop succeeds only if it retrieves its target, so end-to-end success is the <strong>product</strong> of the
        per-hop recalls — <span style={{ color: MISS }}>∏rᵢ</span> decays geometrically. To hold an end-to-end target{' '}
        <strong>ρ = {fmt(rho, 2)}</strong>, each of k hops must reach <span style={{ color: ANSWERC }}>ρ^(1/k)</span> (→ 1 as
        k grows), so the front end must <span style={{ color: REACH }}>over-fetch</span> ever harder — exactly the
        capstone's cascade law, now over hops. Hops are not independent, though: a query hard at one hop is hard at the
        next, and by FKG that positive dependence makes the realized chain recall sit <em>above</em> the product (next
        panel's sweep). The independent product is the conservative, safe-to-provision lower bound.
      </p>
      <FkgStrip />
    </div>
  );
}

function FkgStrip() {
  const W = 520, H = 96, padL = 42, padR = 16, padT = 12, padB = 24;
  const n = FKG_SWEEP.length;
  const px = (i: number) => padL + (W - padL - padR) * (i / (n - 1));
  const gMax = 0.12;
  const py = (g: number) => padT + (H - padT - padB) * (0.5 - g / (2 * gMax));   // 0 gap at the middle
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="FKG: realized minus independent chain recall versus hop dependence" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block', marginTop: '0.3rem' }}>
      <line x1={padL} y1={py(0)} x2={W - padR} y2={py(0)} stroke="var(--color-border)" strokeWidth={1} />
      <path d={poly(FKG_SWEEP.map((d) => d.gap), px, py)} fill="none" stroke={ACCENT} strokeWidth={2} />
      {FKG_SWEEP.map((d, i) => (
        <g key={i}>
          <circle cx={px(i)} cy={py(d.gap)} r={2.6} fill={d.gap >= 0 ? ANSWERC : MISS} />
          <text x={px(i)} y={H - 6} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{d.rho}</text>
        </g>
      ))}
      <text x={padL + 2} y={padT + 4} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">FKG gap = realized − independent chain recall</text>
      <text x={W - padR} y={py(0.1) + 3} textAnchor="end" fontSize={7.5} fill={ANSWERC} fontFamily="var(--font-sans)">+ (ρ&gt;0: realized higher)</text>
      <text x={W - padR} y={py(-0.085) + 3} textAnchor="end" fontSize={7.5} fill={MISS} fontFamily="var(--font-sans)">− (ρ&lt;0)</text>
      <text x={(padL + W - padR) / 2} y={H - 1} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">hop dependence ρ</text>
    </svg>
  );
}

// ===== Panel C — the stopping rule ==================================================================
function StoppingPanel() {
  const W = 520, H = 250, padL = 42, padR = 16, padT = 24, padB = 40;
  const n = TRAJ3_RESID.length;
  const slot = (W - padL - padR) / n;
  const bx = (j: number) => padL + slot * (j + 0.5);
  const py = (v: number) => H - padB - (H - padT - padB) * Math.min(v / 1.0, 1);
  const stopHop = TRAJ3_RESID.findIndex((r) => r < REFORM_EPS);     // closed-form crossing in TS
  const hopsTaken = stopHop === -1 ? n : stopHop + 1;
  const labels = ['hop 1: bridge', 'hop 2: bridge', 'hop 3: answer'];
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="per-hop reformulation residual against the stopping threshold" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={padL - 5} y={py(f) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{f}</text>
        ))}
        {/* stopping threshold */}
        <line x1={padL} y1={py(REFORM_EPS)} x2={W - padR} y2={py(REFORM_EPS)} stroke={MISS} strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={W - padR - 2} y={py(REFORM_EPS) - 4} textAnchor="end" fontSize={8} fill={MISS} fontFamily="var(--font-sans)">stop threshold ε = {REFORM_EPS}</text>
        {TRAJ3_RESID.map((rv, j) => {
          const cont = rv >= REFORM_EPS;
          return (
            <g key={j}>
              <rect x={bx(j) - slot * 0.22} y={py(rv)} width={slot * 0.44} height={(H - padB) - py(rv)}
                fill={cont ? ANSWERC : MISS} fillOpacity={0.8} />
              <text x={bx(j)} y={py(rv) - 5} textAnchor="middle" fontSize={9} fill={cont ? ANSWERC : MISS} fontFamily="var(--font-sans)" fontWeight={700}>{fmt(rv, 2)}</text>
              <text x={bx(j)} y={H - padB + 13} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{labels[j]}</text>
              <text x={bx(j)} y={H - padB + 24} textAnchor="middle" fontSize={7.5} fill={MUTED} fontFamily="var(--font-sans)">KL move {fmt(TRAJ3_KL_MOVES[j], 2)}</text>
            </g>
          );
        })}
        <text x={padL + 2} y={padT + 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          residual ‖d − ⟨d,q⟩q‖ = the new direction each filing opens
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="adaptive hops taken (3-hop chain)" value={`${hopsTaken}`} accent />
        <Readout label="bridge belief-movement (KL)" value={fmt(BRIDGE_KL, 2)} color={MUTED} />
        <Readout label="answer belief-movement (KL)" value={fmt(ANSWER_KL, 2)} color={ANSWERC} />
        <Readout label="adaptive count = depth" value={`${HOPS_BY_CLASS['1-hop']} / ${HOPS_BY_CLASS['2-hop']} / ${HOPS_BY_CLASS['3-hop']}`} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Each filing's <strong>residual</strong> — the part orthogonal to the current query — is the new direction it
        opens, the expected information of the next hop. It clears the threshold through every{' '}
        <span style={{ color: ANSWERC }}>bridge</span> hop (a new entity to chase) and{' '}
        <span style={{ color: MISS }}>collapses</span> at the terminal answer filing, which names no one new — so the
        loop stops, and the adaptive hop count equals the chain depth (1 / 2 / 3). Note the trap: belief-<em>movement</em>{' '}
        is <strong>tiny at the bridge</strong> ({fmt(BRIDGE_KL, 2)}) and <strong>huge at the answer</strong>{' '}
        ({fmt(ANSWER_KL, 2)}) — the decisive evidence comes <em>last</em>. A myopic "stop when the belief stops moving"
        rule would halt at the worthless-looking bridge and never reach the answer. That is the supermodular trap, made
        operational.
      </p>
    </div>
  );
}

// ===== Panel D — the supermodular synergy ===========================================================
function SynergyPanel({ multi, setMulti }: { multi: boolean; setMulti: (v: boolean) => void }) {
  const W = 520, H = 232, padL = 44, padR = 16, padT = 26, padB = 46;
  const bars = [
    { label: 'answer alone', v: XOR_DELTA_EMPTY, color: MISS },
    { label: 'answer | bridge', v: XOR_DELTA_GIVEN_BRIDGE, color: ANSWERC },
  ];
  const slot = (W - padL - padR) / bars.length;
  const bx = (j: number) => padL + slot * (j + 0.5);
  const py = (v: number) => H - padB - (H - padT - padB) * Math.min(v / 1.1, 1);
  const reach = multi ? ANSWER_IN_REFORM_POOL : ANSWER_IN_SINGLE_POOL;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', margin: '0.1rem 0 0.6rem' }}>
        <button type="button" style={miniPill(!multi, MISS)} onClick={() => setMulti(false)}>single-shot selection</button>
        <button type="button" style={miniPill(multi, ANSWERC)} onClick={() => setMulti(true)}>multi-hop (reformulate)</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="information gain of the answer document alone versus given the bridge" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={padL - 5} y={py(f) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{f}</text>
        ))}
        {bars.map((b, j) => (
          <g key={j}>
            <rect x={bx(j) - slot * 0.2} y={py(b.v)} width={slot * 0.4} height={(H - padB) - py(b.v)} fill={b.color} fillOpacity={0.82} />
            <text x={bx(j)} y={py(b.v) - 6} textAnchor="middle" fontSize={11} fill={b.color} fontFamily="var(--font-sans)" fontWeight={700}>{fmt(b.v, 0)} bit{b.v === 1 ? '' : 's'}</text>
            <text x={bx(j)} y={H - padB + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{b.label}</text>
          </g>
        ))}
        {/* the increasing-marginal arrow */}
        <path d={`M${bx(0)},${py(XOR_DELTA_EMPTY) - 22} C${(bx(0) + bx(1)) / 2},${py(1) - 40} ${(bx(0) + bx(1)) / 2},${py(1) - 40} ${bx(1)},${py(XOR_DELTA_GIVEN_BRIDGE) - 22}`}
          fill="none" stroke={ACCENT} strokeWidth={1.4} markerEnd="url(#mhead)" />
        <defs>
          <marker id="mhead" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={ACCENT} />
          </marker>
        </defs>
        <text x={(bx(0) + bx(1)) / 2} y={py(1) - 46} textAnchor="middle" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">marginal INCREASES with conditioning → supermodular</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label={multi ? 'answer reachable (multi-hop)' : 'answer reachable (single-shot)'} value={reach >= 1 ? 'yes ✓' : 'no ✗'} color={reach >= 1 ? ANSWERC : MISS} />
        <Readout label="answer in the single-hop pool" value={fmt(ANSWER_IN_SINGLE_POOL, 2)} color={MISS} />
        <Readout label="reached after one reformulation" value={fmt(ANSWER_IN_REFORM_POOL, 2)} color={ANSWERC} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Information gain is <strong>not submodular</strong>, and the compositional question is its{' '}
        <strong>supermodular</strong> case: the answer document <span style={{ color: MISS }}>alone</span> is worth{' '}
        {fmt(XOR_DELTA_EMPTY, 0)} bits about the question, but <span style={{ color: ANSWERC }}>given the bridge</span> it is
        worth {fmt(XOR_DELTA_GIVEN_BRIDGE, 0)} — the marginal <em>increases</em> with conditioning, the exact opposite of
        diminishing returns (the XOR witness from context selection). So a single-shot selector, optimizing a one-pass
        objective, cannot reach the answer: it is <span style={{ color: MISS }}>never in the one-hop pool</span>{' '}
        ({fmt(ANSWER_IN_SINGLE_POOL, 2)}). Only <span style={{ color: ANSWERC }}>reformulating from the bridge</span>{' '}
        converts its named entity into a query that reaches the answer ({fmt(ANSWER_IN_REFORM_POOL, 2)}). Multi-hop is the
        synergy-exploiting move; greedy single-shot earns no (1−1/e) guarantee here because the objective is not submodular.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'gap' | 'compounding' | 'stopping' | 'synergy';
const TEX: Record<Panel, string> = {
  gap: '\\cos\\angle(q, d_{\\text{ans}}) < \\tau < \\cos\\angle(q, d_{\\text{br}}), \\qquad q\' = \\frac{d - \\langle d, q\\rangle q}{\\lVert d - \\langle d, q\\rangle q\\rVert}',
  compounding: 'R_{\\text{chain}} = \\prod_{i=1}^{k} r_i, \\qquad r_i = \\rho^{1/k}, \\qquad \\text{over-fetch} = 1/\\textstyle\\prod_i r_i',
  stopping: '\\text{stop when } \\lVert d - \\langle d, q\\rangle q\\rVert < \\varepsilon \\quad(\\text{the filing names no new entity})',
  synergy: '\\Delta(d_{\\text{ans}}\\mid\\varnothing) = 0 \\;<\\; \\Delta(d_{\\text{ans}}\\mid\\{d_{\\text{br}}\\}) = 1 \\quad(\\text{supermodular})',
};

export default memo(function MultiHopLaboratory() {
  const [panel, setPanel] = useState<Panel>('gap');
  const [reformulated, setReformulated] = useState(false);
  const [rho, setRho] = useState(0.9);
  const [multi, setMulti] = useState(false);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="multi-hop-iterative-retrieval" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'gap')} onClick={() => setPanel('gap')}>A · compositional gap</button>
        <button type="button" style={pill(panel === 'compounding')} onClick={() => setPanel('compounding')}>B · compounding recall</button>
        <button type="button" style={pill(panel === 'stopping')} onClick={() => setPanel('stopping')}>C · stopping rule</button>
        <button type="button" style={pill(panel === 'synergy')} onClick={() => setPanel('synergy')}>D · supermodular synergy</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'gap' && <GapPanel reformulated={reformulated} setReformulated={setReformulated} />}
      {panel === 'compounding' && <CompoundingPanel rho={rho} setRho={setRho} />}
      {panel === 'stopping' && <StoppingPanel />}
      {panel === 'synergy' && <SynergyPanel multi={multi} setMulti={setMulti} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        Finance vMF corpus on the shared dense-retrieval cloud (seed 7, dim {32}): companies in distinct sectors are
        near-orthogonal, so a chain's answer company is invisible to its query; a bridge filing for company A that names
        company B is drawn around cos α·u_A + sin α·u_B with α = {ALPHA_MENTION_DEG}°, and the reformulation operator
        extracts the sin α mention. The answer is read through the imported <code>answer_posterior</code> (a synthetic
        softmax stand-in, not an LLM); the compounding/FKG demo reuses the capstone's cascade machinery. Numbers mirror{' '}
        <code>multi_hop_iterative_retrieval.py</code> (top-k = {K_RETRIEVE}); the lab recomputes only ∏rᵢ, ρ^(1/k), the
        over-fetch reciprocal, and the threshold crossing.
      </p>
    </div>
  );
});
