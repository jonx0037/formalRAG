import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * InfoNCE Contrastive Laboratory — three panels for the `infonce-contrastive-objective` topic:
 *   A. The hard-negative gradient. A unit circle with the query, its positive, and a fixed spread of
 *      negatives; each negative's glyph (and the bar beside it) is sized by its softmax gradient weight
 *      p_i = softmax(s_i/tau). The temperature slider concentrates the push onto the hardest negative as
 *      tau -> 0 (entropy of the weights -> 0); spreads it uniformly as tau grows (entropy -> log N).
 *   B. The mutual-information bound. The achieved bound log(N+1) - L against its ceiling log(N+1) and the
 *      true MI, as the negative count grows: the bound tightens toward I_true and SATURATES, and at small
 *      N the ceiling itself sits below the truth — the honest reason large batches help.
 *   C. Alignment and uniformity. The (uniformity, alignment) plane: a random init, then the SAME endpoint
 *      reached by minimizing alignment+uniformity directly and by minimizing InfoNCE — both collapse
 *      alignment and spread to uniformity. The temperature slider walks the trained config along the
 *      uniformity axis (tau ~ inverse vMF concentration kappa).
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): POS_COS, NEG_COS, TAU_GRID, TEMP_CURVE, MI_BOUND, MI_TARGET_NATS,
 * UNIF_T, CONVERGE, ALIGN_UNIF, FIN_SHARE are mirrored TO THE DECIMAL from
 * notebooks/infonce-contrastive-objective/infonce_contrastive_objective.py (viz_constants()). The lab
 * recomputes only CLOSED FORM in TS — the softmax weights p_i = softmax(NEG_COS/tau), the ceiling
 * log(N+1), and the equivalent concentration kappa ~ 1/tau — exactly as the source's tests assert; every
 * MEASURED number (the MI bound, the alignment/uniformity values, the finance hard-negative share) is
 * baked. test_mi_lower_bound_holds_and_saturates / test_alignment_uniformity_decomposition /
 * test_gradient_structure / test_temperature_concentrates_gradient / test_finance_hard_negatives_dominate
 * assert these. Change a number here -> change it there, and re-run the notebook. Sliders only (no d3
 * drag). SVG text inherits theme.
 */

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: the synthetic geometry — the positive's cosine to the query and a spread of negative cosines,
// hardest first. The gradient weights are the closed form softmax(cos/tau), recomputed in TS.
const POS_COS = 0.92;
const NEG_COS = [0.86, 0.71, 0.58, 0.44, 0.31, 0.17, 0.02, -0.15];
const TAU_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0];
// Baked top-1 mass and weight entropy of softmax(NEG_COS/tau) — the decimal anchor on the closed form.
const TEMP_CURVE = [
  { tau: 0.01, top1: 1.0, entropy: 0.0 }, { tau: 0.02, top1: 0.9994, entropy: 0.0047 },
  { tau: 0.05, top1: 0.949, entropy: 0.2157 }, { tau: 0.1, top1: 0.7667, entropy: 0.7254 },
  { tau: 0.2, top1: 0.5106, entropy: 1.355 }, { tau: 0.35, top1: 0.3454, entropy: 1.7469 },
  { tau: 0.5, top1: 0.2742, entropy: 1.8969 }, { tau: 0.7, top1: 0.2277, entropy: 1.9805 },
  { tau: 1.0, top1: 0.1943, entropy: 2.0292 },
];
// Panel B: the MI lower bound log(N+1) - L on a Gaussian joint with I_true = 2.0 nats, Bayes-optimal
// critic, against its ceiling log(N+1). The bound is MEASURED (Monte-Carlo) and baked.
const MI_TARGET_NATS = 2.0;
const MI_BOUND = [
  { n_neg: 1, ceiling: 0.6931, bound: 0.579 }, { n_neg: 3, ceiling: 1.3863, bound: 1.0644 },
  { n_neg: 7, ceiling: 2.0794, bound: 1.437 }, { n_neg: 15, ceiling: 2.7726, bound: 1.6794 },
  { n_neg: 31, ceiling: 3.4657, bound: 1.8279 }, { n_neg: 63, ceiling: 4.1589, bound: 1.9016 },
  { n_neg: 127, ceiling: 4.852, bound: 1.9537 }, { n_neg: 255, ceiling: 5.5452, bound: 1.9702 },
];
// Panel C: the alignment/uniformity convergence — a random init and the two coincident optima — plus the
// trained config at each temperature (alignment, uniformity). All MEASURED on a d=3 sphere toy and baked.
const UNIF_T = 2.0;
const CONVERGE = [
  { label: 'init', alignment: 1.8921, uniformity: -2.0646, mean_resultant: 0.1637 },
  { label: 'align+unif', alignment: 0.0, uniformity: -2.2377, mean_resultant: 0.0013 },
  { label: 'infonce', alignment: 0.0, uniformity: -2.2388, mean_resultant: 0.0018 },
];
const ALIGN_UNIF = [
  { tau: 0.01, kappa_equiv: 100.0, alignment: 1.0687, uniformity: -2.1641 },
  { tau: 0.02, kappa_equiv: 50.0, alignment: 0.0033, uniformity: -2.1734 },
  { tau: 0.05, kappa_equiv: 20.0, alignment: 0.0, uniformity: -2.2389 },
  { tau: 0.1, kappa_equiv: 10.0, alignment: 0.0, uniformity: -2.2395 },
  { tau: 0.2, kappa_equiv: 5.0, alignment: 0.0, uniformity: -2.2386 },
  { tau: 0.35, kappa_equiv: 2.86, alignment: 0.0, uniformity: -2.2321 },
  { tau: 0.5, kappa_equiv: 2.0, alignment: 0.0, uniformity: -2.221 },
  { tau: 0.7, kappa_equiv: 1.43, alignment: 0.0, uniformity: -2.207 },
  { tau: 1.0, kappa_equiv: 1.0, alignment: 0.0, uniformity: -2.1821 },
];
// Finance: the same-sector hard-negative share of the gradient at three temperatures (baked, measured).
const FIN_SHARE = [
  { tau: 0.05, hard_share: 0.9379, max_hard_weight: 0.7143 },
  { tau: 0.2, hard_share: 0.5747, max_hard_weight: 0.2646 },
  { tau: 1.0, hard_share: 0.2767, max_hard_weight: 0.0992 },
];

const POS_COLOR = '#5fa873';        // the positive — the target to pull toward
const NEG_COLOR = 'var(--color-accent)';
const HARD_COLOR = '#7C3AED';        // the hardest negative, highlighted
const CEIL_COLOR = '#6a8caf';
const THEORY_COLOR = 'var(--color-text-secondary)';

// round trig-derived coordinates to a fixed precision so SSR (Node) and client (browser) serialize
// identical strings — full-precision Math.cos/sin differ in the last ULP across engines and warn on hydrate.
const r2 = (v: number) => Math.round(v * 100) / 100;

// closed-form softmax over an array at temperature tau (the gradient weights over candidates)
function softmax(xs: number[], tau: number): number[] {
  const z = xs.map((x) => x / tau);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
function entropy(p: number[]): number {
  return -p.reduce((a, w) => a + (w > 0 ? w * Math.log(w) : 0), 0);
}

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '11rem' }}>{label} = <strong>{fmt(value)}</strong></span>
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

// ===== Panel A — the unit circle: query, positive, negatives sized by gradient weight ===============
function GradientCircle({ weights }: { weights: number[] }) {
  const S = 300, c = S / 2, R = 112;
  const uAng = -Math.PI / 2;                                   // query points up
  const posAng = uAng - Math.acos(POS_COS);                    // positive on the left
  const px = r2(c + R * Math.cos(posAng)), py = r2(c + R * Math.sin(posAng));
  const wMax = Math.max(...weights);
  const negs = NEG_COS.map((cos, i) => {
    const ang = uAng + Math.acos(cos);                         // harder (higher cos) sits nearer the query
    return { x: r2(c + R * Math.cos(ang)), y: r2(c + R * Math.sin(ang)), w: weights[i], hardest: weights[i] === wMax };
  });
  return (
    <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="query, positive, and negatives on the unit circle sized by gradient weight" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <circle cx={c} cy={c} r={R} fill="none" stroke="var(--color-border)" strokeWidth={1} />
      {/* query */}
      <line x1={c} y1={c} x2={c} y2={r2(c - R)} stroke="var(--color-text)" strokeWidth={2.4} />
      <circle cx={c} cy={r2(c - R)} r={4.5} fill="var(--color-text)" />
      <text x={c + 7} y={r2(c - R + 4)} fontSize={12} fill="var(--color-text)" fontFamily="var(--font-sans)">q</text>
      {/* positive */}
      <line x1={c} y1={c} x2={px} y2={py} stroke={POS_COLOR} strokeWidth={2.2} />
      <circle cx={px} cy={py} r={6} fill={POS_COLOR} />
      <text x={r2(px - 16)} y={r2(py - 6)} fontSize={12} fill={POS_COLOR} fontFamily="var(--font-sans)">d⁺</text>
      {/* negatives, glyph radius proportional to the gradient weight */}
      {negs.map((n, i) => (
        <g key={i}>
          <line x1={c} y1={c} x2={n.x} y2={n.y} stroke={n.hardest ? HARD_COLOR : NEG_COLOR} strokeWidth={1} opacity={0.4} />
          <circle cx={n.x} cy={n.y} r={r2(3 + 16 * n.w)} fill={n.hardest ? HARD_COLOR : NEG_COLOR} opacity={n.hardest ? 0.95 : 0.6} />
        </g>
      ))}
      <text x={c} y={S - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
        glyph area ∝ gradient weight pᵢ
      </text>
    </svg>
  );
}

function WeightBars({ weights }: { weights: number[] }) {
  const pw = 360, ph = 300, pad = 40;
  const n = weights.length;
  const bw = (pw - 2 * pad) / n * 0.7;
  const gap = (pw - 2 * pad) / n;
  const wMax = Math.max(...weights);
  const fy = (w: number) => ph - pad - (ph - 2 * pad) * w;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">negatives, hardest → easiest</text>
      <text x={12} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${ph / 2})`}>gradient weight pᵢ</text>
      {weights.map((w, i) => {
        const x = pad + gap * i + (gap - bw) / 2;
        const hardest = w === wMax;
        return <rect key={i} x={r2(x)} y={r2(fy(w))} width={bw} height={r2(ph - pad - fy(w))} fill={hardest ? HARD_COLOR : NEG_COLOR} opacity={hardest ? 0.95 : 0.6} rx={1.5} />;
      })}
    </svg>
  );
}

// ===== Panel B — the MI lower bound vs the log(N+1) ceiling =========================================
function MIBoundPlot({ idx }: { idx: number }) {
  const pw = 620, ph = 320, pad = 50;
  const n = MI_BOUND.length;
  const yMax = 6;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (nats: number) => ph - pad - (ph - 2 * pad) * nats / yMax;
  const ceil = MI_BOUND.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.ceiling).toFixed(1)).join(' ');
  const bound = MI_BOUND.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.bound).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 2, 4, 6].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
      {MI_BOUND.map((d, i) => (<text key={i} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{d.n_neg + 1}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">candidates N + 1 (one positive + N negatives)</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>nats</text>
      {/* true MI */}
      <line x1={pad} y1={fy(MI_TARGET_NATS)} x2={pw - pad} y2={fy(MI_TARGET_NATS)} stroke="var(--color-text)" strokeWidth={1.2} strokeDasharray="5 3" opacity={0.7} />
      <text x={pw - pad} y={fy(MI_TARGET_NATS) - 5} textAnchor="end" fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">true MI I(q; d⁺) = {MI_TARGET_NATS.toFixed(1)}</text>
      {/* ceiling log(N+1) */}
      <path d={ceil} fill="none" stroke={CEIL_COLOR} strokeWidth={2} strokeDasharray="4 3" />
      {/* achieved bound */}
      <path d={bound} fill="none" stroke={NEG_COLOR} strokeWidth={2.8} />
      {MI_BOUND.map((d, i) => (<circle key={i} cx={fx(i)} cy={fy(d.bound)} r={i === idx ? 5.5 : 3} fill={NEG_COLOR} opacity={i === idx ? 1 : 0.7} />))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={pad + 12} y1={pad + 4} x2={pad + 28} y2={pad + 4} stroke={CEIL_COLOR} strokeWidth={2} strokeDasharray="4 3" />
        <text x={pad + 32} y={pad + 7} fill="var(--color-text-secondary)">ceiling log(N+1)</text>
        <line x1={pad + 12} y1={pad + 20} x2={pad + 28} y2={pad + 20} stroke={NEG_COLOR} strokeWidth={2.8} />
        <text x={pad + 32} y={pad + 23} fill="var(--color-text-secondary)">bound log(N+1) − ℒ</text>
      </g>
    </svg>
  );
}

// ===== Panel C — the alignment/uniformity plane ====================================================
function AlignUnifPlot({ tauIdx }: { tauIdx: number }) {
  const pw = 460, ph = 320, pad = 50;
  const ux0 = -2.30, ux1 = -2.0;       // uniformity axis (x): lower = more spread
  const ay0 = 0, ay1 = 2.0;            // alignment axis (y): lower = positives closer
  const fx = (u: number) => pad + (pw - 2 * pad) * (u - ux0) / (ux1 - ux0);
  const fy = (a: number) => ph - pad - (ph - 2 * pad) * (a - ay0) / (ay1 - ay0);
  const sel = ALIGN_UNIF[tauIdx];
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[-2.3, -2.2, -2.1, -2.0].map((v) => (<text key={v} x={fx(v)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {[0, 0.5, 1.0, 1.5, 2.0].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">uniformity loss (← more spread)</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>alignment loss (↓ positives closer)</text>
      {/* the trained config at each temperature */}
      {ALIGN_UNIF.map((d, i) => (
        <circle key={i} cx={fx(d.uniformity)} cy={fy(d.alignment)} r={i === tauIdx ? 6 : 3} fill={NEG_COLOR} opacity={i === tauIdx ? 1 : 0.45} />
      ))}
      {/* the convergence: init and the two coincident optima */}
      <circle cx={fx(CONVERGE[0].uniformity)} cy={fy(CONVERGE[0].alignment)} r={6} fill="var(--color-text)" />
      <text x={fx(CONVERGE[0].uniformity) + 9} y={fy(CONVERGE[0].alignment) + 4} fontSize={10} fill="var(--color-text)" fontFamily="var(--font-sans)">random init</text>
      <circle cx={fx(CONVERGE[1].uniformity)} cy={fy(CONVERGE[1].alignment)} r={7} fill="none" stroke={POS_COLOR} strokeWidth={2.5} />
      <circle cx={fx(CONVERGE[2].uniformity)} cy={fy(CONVERGE[2].alignment)} r={4} fill={POS_COLOR} />
      <text x={fx(CONVERGE[2].uniformity) - 6} y={fy(CONVERGE[2].alignment) - 10} textAnchor="end" fontSize={10} fill={POS_COLOR} fontFamily="var(--font-sans)">align+unif ≡ InfoNCE</text>
      {/* arrow from init to the optimum */}
      <line x1={fx(CONVERGE[0].uniformity)} y1={fy(CONVERGE[0].alignment)} x2={fx(CONVERGE[2].uniformity) + 6} y2={fy(CONVERGE[2].alignment) + 8} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="3 3" opacity={0.6} />
      <text x={pw - pad} y={pad + 4} textAnchor="end" fontSize={9.5} fill={NEG_COLOR} fontFamily="var(--font-sans)">trained at τ = {sel.tau} (κ ≈ {sel.kappa_equiv})</text>
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'gradient' | 'mibound' | 'geometry';

export default memo(function InfoNCEContrastiveLaboratory() {
  const [panel, setPanel] = useState<Panel>('gradient');
  const [tau, setTau] = useState(0.2);
  const [nIdx, setNIdx] = useState(3);
  const [tauIdx, setTauIdx] = useState(4);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'gradient'
        ? '\\frac{\\partial \\mathcal{L}}{\\partial s_i^-} = \\frac{p_i}{\\tau}, \\qquad p_i = \\frac{e^{s_i/\\tau}}{\\sum_j e^{s_j/\\tau}}'
        : panel === 'mibound'
          ? 'I(q; d^+) \\;\\ge\\; \\log(N{+}1) - \\mathcal{L}_{\\text{InfoNCE}}'
          : '\\mathcal{L} \\longrightarrow \\underbrace{\\mathbb{E}\\,\\lVert f(x){-}f(y)\\rVert^2}_{\\text{alignment}} + \\underbrace{\\log\\,\\mathbb{E}\\,e^{-t\\lVert f(x){-}f(y)\\rVert^2}}_{\\text{uniformity}}';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  // Panel A: the gradient weights are the closed form softmax(NEG_COS / tau).
  const weights = softmax(NEG_COS, tau);
  const top1 = Math.max(...weights);
  const H = entropy(weights);
  const mi = MI_BOUND[nIdx];

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'gradient')} onClick={() => setPanel('gradient')}>A · temperature & hard negatives</button>
        <button style={pill(panel === 'mibound')} onClick={() => setPanel('mibound')}>B · the MI bound</button>
        <button style={pill(panel === 'geometry')} onClick={() => setPanel('geometry')}>C · alignment & uniformity</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'gradient' && (
        <div>
          <Slider label="temperature τ" value={tau} min={0.02} max={1} step={0.01} onChange={setTau} fmt={(v) => v.toFixed(2)} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.1fr', gap: '0.8rem', alignItems: 'center' }}>
            <GradientCircle weights={weights} />
            <WeightBars weights={weights} />
          </div>
          <div style={{ display: 'flex', gap: '2rem', marginTop: '0.5rem' }}>
            <Readout label="hardest-negative weight max pᵢ" value={top1.toFixed(3)} accent />
            <Readout label="weight entropy H(p)" value={`${H.toFixed(3)} / ${Math.log(NEG_COS.length).toFixed(3)}`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Each negative repels the query with a force equal to its softmax weight pᵢ = e<sup>sᵢ/τ</sup> / Σ. As τ → 0 the mass
            collapses onto the single hardest negative — the one nearest the query — and the weight entropy falls toward 0; as τ
            grows the push spreads to all negatives and the entropy rises toward log N = {Math.log(NEG_COS.length).toFixed(2)}. The
            gradient structure is exact; which τ is <em>right</em> is empirical.
          </p>
        </div>
      )}

      {panel === 'mibound' && (
        <div>
          <Slider label="candidates N + 1" value={nIdx} min={0} max={MI_BOUND.length - 1} step={1} onChange={setNIdx} fmt={(v) => `${MI_BOUND[v].n_neg + 1}`} />
          <MIBoundPlot idx={nIdx} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="ceiling log(N+1)" value={mi.ceiling.toFixed(3)} />
            <Readout label="achieved bound" value={mi.bound.toFixed(3)} accent />
            <Readout label="gap to true MI" value={(MI_TARGET_NATS - mi.bound).toFixed(3)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Minimizing InfoNCE maximizes the lower bound log(N+1) − ℒ on the mutual information. The bound tightens toward the true
            MI as the candidate count grows — but it is <strong>ceilinged at log(N+1)</strong>, so with only a few candidates it
            cannot even reach the truth (the ceiling sits below the dashed line), and it <strong>saturates</strong> as N grows.
            More negatives raise the ceiling: the honest reason large batches help.
          </p>
        </div>
      )}

      {panel === 'geometry' && (
        <div>
          <Slider label="temperature τ" value={tauIdx} min={0} max={ALIGN_UNIF.length - 1} step={1} onChange={setTauIdx} fmt={(v) => `${ALIGN_UNIF[v].tau}`} />
          <AlignUnifPlot tauIdx={tauIdx} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="alignment at τ" value={ALIGN_UNIF[tauIdx].alignment.toFixed(3)} />
            <Readout label="uniformity at τ" value={ALIGN_UNIF[tauIdx].uniformity.toFixed(3)} accent />
            <Readout label="equivalent κ ≈ 1/τ" value={ALIGN_UNIF[tauIdx].kappa_equiv.toFixed(1)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            As the negatives grow, InfoNCE splits into <strong>alignment</strong> (pulling positives together) and
            <strong> uniformity</strong> (spreading every embedding toward the uniform sphere — the κ → 0 vMF law of the hypersphere
            topic). Minimizing alignment + uniformity directly and minimizing InfoNCE reach the <em>same</em> configuration from a
            random init (the two markers coincide). Temperature tunes where the trained cloud lands along the uniformity axis: low
            τ behaves like a high vMF concentration κ ≈ 1/τ. (Below τ ≈ 0.02 the optimization itself destabilizes — the lone point
            off the floor — a finite-sample edge, not the asymptotic theory.)
          </p>
        </div>
      )}
    </div>
  );
});
