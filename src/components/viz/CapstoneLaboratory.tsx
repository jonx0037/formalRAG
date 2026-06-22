import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Capstone Laboratory — four panels for the `capstone-multimodal-financial-rag` topic:
 *   A. Cascade composition. A funnel of per-stage retentions whose product is end-to-end recall, plus
 *      the FKG demonstration: a dependence slider showing that under positive dependence the
 *      independent product is a conservative LOWER bound on true recall (and the over-fetch 1/∏rᵢ).
 *   B. Hybrid fusion gain. Per-leg recall vs the RRF-fused recall, a de-correlation slider showing the
 *      gain grow as the legs disagree, and the dominated-leg FLIP (co-endorsed false positive), the
 *      RRF scores recomputed in TS to show fused recall dropping below the strong leg.
 *   C. Budget water-filling. The three legs' concave retention-vs-cost curves; a budget slider pours a
 *      fixed compute budget across them, equalizing the marginal log-recall (the water level) and
 *      beating uniform and all-in-one allocation.
 *   D. Storage + collapse. Single-vector vs raw multi-vector (32×) vs PLAID (~1×), and the one exact
 *      statement: a full-budget pipeline recovers brute-force retrieval (recall 1.0).
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): STAGE_RET, CASCADE_PRODUCT, DEP, OVERFETCH, PER_LEG, FUSED,
 * DECORR, FLIP_STRONG/FLIP_NOISY, CURVES, COST_PER_DOC, DEPTH_GRID, BUDGET_FRAC, STORAGE, and COLLAPSE
 * are mirrored TO THE DECIMAL from notebooks/capstone-multimodal-financial-rag/
 * capstone_multimodal_financial_rag.py (viz_constants()). The matching asserts are
 * test_dependence_direction / test_fusion_gain_under_complementarity / test_dominated_leg_flip /
 * test_decorrelation_endpoints / test_waterfilling_optimality_and_beats_baselines / test_storage_collapse
 * / test_cascade_collapse_anchor. The lab recomputes only CLOSED FORM in TS (RRF 1/(60+rank), set
 * recall, the cascade product, the greedy water-filling over the baked grids); only the MEASURED curves
 * (k-means / MaxSim driven) are baked. Change a number here -> change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
// Panel A — the measured 3-stage cascade (nprobe=8, keep=40) and the FKG dependence copula demo.
const STAGE_RET = [1.0, 0.9225, 0.9225];     // candidate-gen, fuse/prune, rerank retentions
const CASCADE_PRODUCT = 0.851;               // ∏ rᵢ (lower bound); measured end-to-end recall 0.9225
const MEASURED_END = 0.9225;
const R_INDEP = 0.3;     // r₁·r₂ for the illustrative copula demo (DEMO_R1 = 0.6, DEMO_R2 = 0.5)
// rows: [rho, R_true, R_indep, gap] from the bivariate-normal survival copula
const DEP = [
  [-0.6, 0.1982, 0.3, -0.1018], [-0.3, 0.2511, 0.3, -0.0489], [0.0, 0.2981, 0.3, -0.0019],
  [0.3, 0.3494, 0.3, 0.0494], [0.6, 0.3997, 0.3, 0.0997], [0.9, 0.4669, 0.3, 0.1669],
];

// Panel B — three heterogeneous legs over the shared corpus, scored against the MaxSim truth.
const PER_LEG: Record<string, number> = { lexical: 0.425, dense: 0.5475, late_interaction: 0.48 };
const FUSED = 0.58, FUSION_GAIN = 0.0325;
// de-correlation sweep: [frac, mean Kendall-tau, global fusion gain]
const DECORR = [[0.0, 0.0, 0.0], [0.25, 905.5, 0.01], [0.5, 1777.7, 0.03], [0.75, 2374.9, 0.03], [1.0, 2586.6, 0.03]];
// the dominated-leg flip: a false positive co-endorsed by both legs (recompute RRF in TS).
const FLIP_STRONG = ['r1', 'r2', 'r3', 'x1', 'x2'];   // strong leg: relevant first, then co-endorses x1
const FLIP_NOISY = ['x1', 'x2', 'r1', 'r2', 'r3'];    // noisy leg: junk on top, r3 buried
const FLIP_RELEVANT = ['r1', 'r2', 'r3'];
const RRF_C = 60;

// Panel C — each leg's concave retention curve [cost (comps/query), retention], and the per-doc costs.
const CURVES: Record<string, number[][]> = {
  lexical: [[8.0, 0.1525], [20.0, 0.285], [40.0, 0.425], [80.0, 0.5875], [160.0, 0.8275], [320.0, 0.935]],
  dense: [[32.0, 0.1775], [80.0, 0.3675], [160.0, 0.5475], [320.0, 0.735], [640.0, 0.89], [1280.0, 0.9825]],
  late_interaction: [[512.0, 0.185], [1280.0, 0.3225], [2560.0, 0.48], [5120.0, 0.6275], [10240.0, 0.86], [20480.0, 0.985]],
};
const COST_PER_DOC: Record<string, number> = { lexical: 4, dense: 16, late_interaction: 256 };
// the Panel-C demo budget (≈ 0.12 of the all-max cost) is the slider's default in the component.

// Panel D — ColBERT-scale storage (d=128, 32 tokens) and the exact collapse anchor.
const STORAGE = { single_bits: 4096, raw_multi_bits: 131072, plaid_bits: 4608, raw_mult: 32.0, plaid_mult: 1.1 };
const COLLAPSE_RECALL = 1.0;

const LEG_NAMES = ['lexical', 'dense', 'late_interaction'];
const LEG_LABEL: Record<string, string> = { lexical: 'lexical (BM25)', dense: 'dense (MIPS)', late_interaction: 'late interaction' };
const LEG_COLOR: Record<string, string> = { lexical: '#c08457', dense: '#6a8caf', late_interaction: '#9b6abf' };

const POS_COLOR = '#5fa873';     // fusion gain / water-filling win
const ACCENT = 'var(--color-accent)';
const NEG_COLOR = '#c25b6b';     // the flip / negative gap
const MUTED = '#9aa3ad';

const clampIdx = (i: number, n: number) => Math.max(0, Math.min(Math.round(i), n - 1));
const fmt = (x: number, n = 3) => x.toFixed(n);

// --- closed-form TS recomputation (RRF, recall, water-filling) -----------------------
function rrfFuse(rankings: string[][]): string[] {
  const items = new Set<string>();
  rankings.forEach((r) => r.forEach((d) => items.add(d)));
  const score: Record<string, number> = {};
  items.forEach((d) => (score[d] = 0));
  rankings.forEach((r) => r.forEach((d, i) => (score[d] += 1 / (RRF_C + i + 1))));
  return [...items].sort((a, b) => score[b] - score[a] || (a < b ? -1 : 1));
}
function recallAtK(ranking: string[], relevant: string[], k: number): number {
  const rel = new Set(relevant);
  const hit = ranking.slice(0, k).filter((d) => rel.has(d)).length;
  return hit / Math.min(k, relevant.length);
}
type Alloc = Record<string, number>;
const allocCost = (a: Alloc) => LEG_NAMES.reduce((s, k) => s + CURVES[k][a[k]][0], 0);
const endToEnd = (a: Alloc) => LEG_NAMES.reduce((p, k) => p * CURVES[k][a[k]][1], 1);
function waterFill(B: number): Alloc {
  const a: Alloc = { lexical: 0, dense: 0, late_interaction: 0 };
  if (allocCost(a) > B + 1e-9) return a;
  for (;;) {
    let bestS: string | null = null, bestM = -Infinity;
    for (const s of LEG_NAMES) {
      const j = a[s];
      if (j + 1 >= CURVES[s].length) continue;
      const [c0, g0] = CURVES[s][j], [c1, g1] = CURVES[s][j + 1];
      const marg = (Math.log(g1) - Math.log(g0)) / Math.max(c1 - c0, 1e-9);
      if (allocCost(a) + (c1 - c0) <= B + 1e-9 && marg > bestM) { bestM = marg; bestS = s; }
    }
    if (bestS === null) break;
    a[bestS] += 1;
  }
  return a;
}
function uniformAlloc(B: number): Alloc {
  const share = B / LEG_NAMES.length;
  const a: Alloc = { lexical: 0, dense: 0, late_interaction: 0 };
  for (const s of LEG_NAMES) for (let i = 0; i < CURVES[s].length; i++) if (CURVES[s][i][0] <= share + 1e-9) a[s] = i;
  return a;
}
function allInOne(B: number, stage: string): Alloc {
  const a: Alloc = { lexical: 0, dense: 0, late_interaction: 0 };
  const floor = LEG_NAMES.filter((s) => s !== stage).reduce((s, k) => s + CURVES[k][0][0], 0);
  for (let i = 0; i < CURVES[stage].length; i++) if (floor + CURVES[stage][i][0] <= B + 1e-9) a[stage] = i;
  return a;
}
function marginalLogRecall(a: Alloc): Record<string, number> {
  const out: Record<string, number> = {};
  for (const s of LEG_NAMES) {
    const j = a[s];
    const [c0, g0] = CURVES[s][j === 0 ? 0 : j - 1], [c1, g1] = CURVES[s][j === 0 ? 1 : j];
    out[s] = (Math.log(g1) - Math.log(g0)) / Math.max(c1 - c0, 1e-9);
  }
  return out;
}
const MAX_COST = LEG_NAMES.reduce((s, k) => s + CURVES[k][CURVES[k].length - 1][0], 0);

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
      <span style={{ minWidth: '13rem' }}>{label} = <strong>{display}</strong></span>
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

// ===== Panel A — cascade funnel + FKG dependence ====================================================
function CascadePanel({ rhoIdx, setRhoIdx }: { rhoIdx: number; setRhoIdx: (v: number) => void }) {
  const W = 460, H = 250, padL = 50, padR = 18, padT = 18, padB = 42;
  const row = DEP[clampIdx(rhoIdx, DEP.length)];
  const [rho, rTrue, , gap] = row;
  // funnel: cumulative retention after each stage
  const cum = STAGE_RET.reduce<number[]>((acc, r) => [...acc, (acc.length ? acc[acc.length - 1] : 1) * r], []);
  const fH = 70, fW = 360, fx0 = 60;
  // dependence plot scales
  const yMin = 0.1, yMax = 0.5;
  const px = (i: number) => padL + (W - padL - padR) * (i / (DEP.length - 1));
  const py = (v: number) => H - padB - (H - padT - padB) * ((v - yMin) / (yMax - yMin));
  return (
    <div>
      {/* funnel schematic */}
      <svg viewBox={`0 0 480 110`} role="img" aria-label="cascade funnel of per-stage retentions" style={{ width: '100%', maxWidth: 480, height: 'auto', display: 'block', margin: '0 auto 0.3rem' }}>
        {['candidate gen', 'fuse / prune', 'rerank'].map((lbl, i) => {
          const w0 = (i === 0 ? 1 : cum[i - 1]) * fW, w1 = cum[i] * fW;
          const x = fx0 + i * (fW / 3), bw = fW / 3 - 8;
          const yTop0 = 18 + (fH - w0 * 0.18) / 2, yTop1 = 18 + (fH - w1 * 0.18) / 2;
          return (
            <g key={lbl}>
              <polygon points={`${x},${yTop0} ${x + bw},${yTop1} ${x + bw},${yTop1 + w1 * 0.18} ${x},${yTop0 + w0 * 0.18}`}
                fill={POS_COLOR} opacity={0.18 + 0.12 * i} stroke={POS_COLOR} strokeWidth={1} />
              <text x={x + bw / 2} y={12} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{lbl}</text>
              <text x={x + bw / 2} y={104} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--color-text)" fontFamily="var(--font-sans)">rᵢ = {STAGE_RET[i].toFixed(3)}</text>
            </g>
          );
        })}
      </svg>
      <Slider label="copula dependence ρ" value={rhoIdx} min={0} max={DEP.length - 1} step={1} onChange={setRhoIdx} display={`${rho.toFixed(1)}`} />
      {/* dependence plot: R_true vs the independent product line */}
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="true recall versus the independent product across dependence" style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0.1, 0.2, 0.3, 0.4, 0.5].map((v) => (<text key={v} x={padL - 6} y={py(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
        {DEP.map((d, i) => (<text key={i} x={px(i)} y={H - padB + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{(d[0] as number).toFixed(1)}</text>))}
        <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">dependence ρ across stages</text>
        {/* independent product line */}
        <line x1={padL} y1={py(R_INDEP)} x2={W - padR} y2={py(R_INDEP)} stroke={MUTED} strokeWidth={1.4} strokeDasharray="5 4" />
        <text x={W - padR} y={py(R_INDEP) - 5} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">independent product r₁·r₂ = {R_INDEP}</text>
        {/* R_true curve */}
        <path d={DEP.map((d, i) => (i ? 'L' : 'M') + px(i) + ' ' + py(d[1] as number)).join(' ')} fill="none" stroke={ACCENT} strokeWidth={2.4} />
        {DEP.map((d, i) => {
          const above = (d[1] as number) >= R_INDEP;
          return <circle key={i} cx={px(i)} cy={py(d[1] as number)} r={i === clampIdx(rhoIdx, DEP.length) ? 5.5 : 3} fill={above ? POS_COLOR : NEG_COLOR} />;
        })}
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="cascade product ∏rᵢ (lower bound)" value={fmt(CASCADE_PRODUCT)} />
        <Readout label="measured end-to-end recall" value={fmt(MEASURED_END)} accent />
        <Readout label="over-fetch 1/∏rᵢ" value={`${(1 / CASCADE_PRODUCT).toFixed(2)}×`} />
        <Readout label={`R_true at ρ=${rho.toFixed(1)}  (gap ${gap >= 0 ? '+' : ''}${gap.toFixed(3)})`} value={fmt(rTrue)} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        End-to-end retention is the <strong>product</strong> of the per-stage retentions ({STAGE_RET.map((r) => r.toFixed(2)).join(' · ')} = {CASCADE_PRODUCT}),
        and to deliver k survivors the front end over-fetches by 1/∏rᵢ ≈ {(1 / CASCADE_PRODUCT).toFixed(2)}×. That product is <em>exact</em> only
        under independent per-stage survival; under the positive dependence real queries exhibit, FKG/Harris makes it a conservative
        <strong> lower bound</strong> — drag ρ above 0 and the true recall lifts <span style={{ color: POS_COLOR }}>above</span> the dashed product line, below 0 it
        falls <span style={{ color: NEG_COLOR }}>under</span> it, with equality at ρ = 0. The measured pipeline (recall {MEASURED_END}) sits above its product {CASCADE_PRODUCT}, as the bound predicts.
      </p>
    </div>
  );
}

// ===== Panel B — fusion gain + the dominated-leg flip ===============================================
function FusionPanel({ fracIdx, setFracIdx, showFlip, setShowFlip }: {
  fracIdx: number; setFracIdx: (v: number) => void; showFlip: boolean; setShowFlip: (v: boolean) => void;
}) {
  const row = DECORR[clampIdx(fracIdx, DECORR.length)];
  const [, tau, gain] = row;
  // flip recomputed in TS
  const fusedFlip = rrfFuse([FLIP_STRONG, FLIP_NOISY]);
  const rhoStrong = recallAtK(FLIP_STRONG, FLIP_RELEVANT, 3);
  const rhoFused = recallAtK(fusedFlip, FLIP_RELEVANT, 3);
  const fusedClean = rrfFuse([FLIP_RELEVANT, FLIP_NOISY]);
  const rhoClean = recallAtK(fusedClean, FLIP_RELEVANT, 3);
  const bestLeg = Math.max(...Object.values(PER_LEG));
  const W = 360, barMax = 200;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(!showFlip)} onClick={() => setShowFlip(false)}>fusion gain</button>
        <button type="button" style={pill(showFlip)} onClick={() => setShowFlip(true)}>the dominated-leg flip</button>
      </div>
      {!showFlip ? (
        <div>
          {/* per-leg + fused recall bars */}
          <svg viewBox={`0 0 ${W} 150`} role="img" aria-label="per-leg recall versus fused recall" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
            {LEG_NAMES.map((name, i) => (
              <g key={name}>
                <text x={0} y={20 + i * 30} fontSize={10} fill="var(--color-text)" fontFamily="var(--font-sans)">{LEG_LABEL[name]}</text>
                <rect x={130} y={10 + i * 30} width={barMax * PER_LEG[name]} height={15} rx={2.5} fill={LEG_COLOR[name]} opacity={0.85} />
                <text x={134 + barMax * PER_LEG[name]} y={22 + i * 30} fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{PER_LEG[name].toFixed(3)}</text>
              </g>
            ))}
            <g>
              <text x={0} y={20 + 3 * 30} fontSize={10} fontWeight={700} fill={POS_COLOR} fontFamily="var(--font-sans)">RRF fused</text>
              <rect x={130} y={10 + 3 * 30} width={barMax * FUSED} height={15} rx={2.5} fill={POS_COLOR} />
              <text x={134 + barMax * FUSED} y={22 + 3 * 30} fontSize={9.5} fontWeight={700} fill={POS_COLOR} fontFamily="var(--font-sans)">{FUSED.toFixed(3)}</text>
              <line x1={130 + barMax * bestLeg} y1={4} x2={130 + barMax * bestLeg} y2={130} stroke={MUTED} strokeWidth={1} strokeDasharray="3 3" />
            </g>
          </svg>
          <Slider label="leg de-correlation" value={fracIdx} min={0} max={DECORR.length - 1} step={1} onChange={setFracIdx} display={`Kendall-τ ≈ ${tau.toFixed(0)}`} />
          <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.2rem', flexWrap: 'wrap' }}>
            <Readout label="best single leg" value={fmt(bestLeg)} />
            <Readout label="RRF fused recall" value={fmt(FUSED)} accent />
            <Readout label="fusion gain (3 legs)" value={`+${FUSION_GAIN.toFixed(3)}`} accent />
            <Readout label={`gain at this de-correlation`} value={`+${gain.toFixed(3)}`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Each leg sees a different partial view of the document, so each recalls true neighbors the others miss. Reciprocal-rank
            fusion recombines them and beats the best single leg ({bestLeg.toFixed(3)} → {FUSED.toFixed(3)}, gain +{FUSION_GAIN.toFixed(3)}). Slide the
            de-correlation up and the gain grows from 0 (a cloned, redundant leg) to +{DECORR[DECORR.length - 1][2].toFixed(2)} (disjoint coverage): fusion pays
            exactly when the legs disagree. It is a <em>condition</em>, not a guarantee — see the flip.
          </p>
        </div>
      ) : (
        <div>
          <FlipBoard fused={fusedFlip} />
          <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="strong leg alone, recall@3" value={fmt(rhoStrong)} />
            <Readout label="RRF fused, recall@3" value={fmt(rhoFused)} accent />
            <Readout label="control: junk NOT co-endorsed" value={fmt(rhoClean)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            "Fused ≥ best leg" is <strong>false</strong> in general. Under RRF's c = {RRF_C} a single top vote (1/{RRF_C + 1} ≈ {(1 / (RRF_C + 1)).toFixed(4)}) is weaker than
            two mid votes (≈ {(2 / (RRF_C + 3)).toFixed(4)}), so a false positive flips the ranking only when <strong>co-endorsed by both legs</strong>. Here x1 is endorsed by both,
            so it overtakes the buried true neighbor r3: fused recall drops to {rhoFused.toFixed(3)} below the strong leg's {rhoStrong.toFixed(1)}. Drop x1 from the strong leg
            (no co-endorsement) and r3 survives — recall returns to {rhoClean.toFixed(1)}. Co-endorsement is the mechanism, not junk-on-top alone.
          </p>
        </div>
      )}
    </div>
  );
}

function FlipBoard({ fused }: { fused: string[] }) {
  const cols = [['strong leg', FLIP_STRONG], ['noisy leg', FLIP_NOISY], ['RRF fused', fused.slice(0, 5)]] as const;
  const colW = 120, rowH = 26;
  const rel = new Set(FLIP_RELEVANT);
  return (
    <svg viewBox={`0 0 ${cols.length * colW + 10} 180`} role="img" aria-label="strong, noisy, and fused rankings showing the co-endorsed flip" style={{ width: '100%', maxWidth: 380, height: 'auto', display: 'block' }}>
      {cols.map(([title, order], ci) => (
        <g key={title}>
          <text x={ci * colW + colW / 2} y={14} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--color-text)" fontFamily="var(--font-sans)">{title}</text>
          {(order as readonly string[]).map((d, ri) => {
            const isRel = rel.has(d);
            const inTop3 = ci === 2 && ri < 3;
            return (
              <g key={d}>
                <rect x={ci * colW + 14} y={24 + ri * rowH} width={colW - 28} height={rowH - 5} rx={3}
                  fill={isRel ? POS_COLOR : NEG_COLOR} opacity={inTop3 || ci < 2 ? 0.85 : 0.3}
                  stroke={ci === 2 && ri === 2 ? 'var(--color-text)' : 'none'} strokeWidth={1.5} strokeDasharray="3 2" />
                <text x={ci * colW + colW / 2} y={24 + ri * rowH + 13} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--color-bg)" fontFamily="var(--font-sans)">{d}</text>
              </g>
            );
          })}
        </g>
      ))}
      <text x={2 * colW + colW / 2} y={24 + 3 * rowH + 18} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">↑ top-3 cut (r3 dropped)</text>
    </svg>
  );
}

// ===== Panel C — budget water-filling ===============================================================
function BudgetPanel({ budget, setBudget }: { budget: number; setBudget: (v: number) => void }) {
  const B = budget;
  const wf = waterFill(B), un = uniformAlloc(B);
  const Rwf = endToEnd(wf), Run = endToEnd(un);
  const aio = Math.max(...LEG_NAMES.map((s) => endToEnd(allInOne(B, s))));
  const marg = marginalLogRecall(wf);
  const W = 460, ph = 200, padL = 46, padR = 14, padT = 14, padB = 38;
  const xMax = Math.log10(20480 + 1), yMin = 0, yMax = 1.02;
  const fx = (c: number) => padL + (W - padL - padR) * (Math.log10(c + 1) / xMax);
  const fy = (r: number) => ph - padB - (ph - padT - padB) * ((r - yMin) / (yMax - yMin));
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${ph}`} role="img" aria-label="per-leg retention versus cost with the water-filling operating points" style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={padL} y1={ph - padB} x2={W - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1.0].map((v) => (<text key={v} x={padL - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
        {[10, 100, 1000, 10000].map((v) => (<text key={v} x={fx(v)} y={ph - padB + 13} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
        <text x={(padL + W - padR) / 2} y={ph - 3} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">cost (distance-comps / query, log)</text>
        <text x={12} y={(padT + ph - padB) / 2} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + ph - padB) / 2})`}>retention</text>
        {LEG_NAMES.map((s) => (
          <g key={s}>
            <path d={CURVES[s].map((p, i) => (i ? 'L' : 'M') + fx(p[0]) + ' ' + fy(p[1])).join(' ')} fill="none" stroke={LEG_COLOR[s]} strokeWidth={2} opacity={0.8} />
            <circle cx={fx(CURVES[s][wf[s]][0])} cy={fy(CURVES[s][wf[s]][1])} r={6} fill={LEG_COLOR[s]} stroke="var(--color-bg)" strokeWidth={1.5} />
          </g>
        ))}
        <g fontFamily="var(--font-sans)" fontSize={9}>
          {LEG_NAMES.map((s, i) => (<g key={s}><rect x={padL + 8 + i * 120} y={padT} width={10} height={3} fill={LEG_COLOR[s]} /><text x={padL + 21 + i * 120} y={padT + 4} fill="var(--color-text-secondary)">{LEG_LABEL[s]}</text></g>))}
        </g>
      </svg>
      <Slider label="compute budget B (comps/query)" value={B} min={1000} max={Math.round(MAX_COST * 0.6)} step={100} onChange={setBudget} display={`${B.toFixed(0)}`} />
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.2rem', flexWrap: 'wrap' }}>
        <Readout label="water-filling R = ∏gᵢ" value={fmt(Rwf)} accent />
        <Readout label="uniform allocation R" value={fmt(Run)} />
        <Readout label="best all-in-one R" value={fmt(aio)} />
        <Readout label="marginal log-recall (lex / dns / li)" value={LEG_NAMES.map((s) => marg[s].toExponential(1)).join(' · ')} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Each leg's retention is concave in the documents it scans, but late interaction costs far more per document
        ({COST_PER_DOC.lexical} : {COST_PER_DOC.dense} : {COST_PER_DOC.late_interaction} comps). Maximizing the separable product ∏gᵢ(cᵢ) under the budget
        equalizes the <strong>marginal log-recall per cost</strong> across the active legs — the water level — so water-filling ({Rwf.toFixed(3)}) beats naive
        uniform splitting ({Run.toFixed(3)}) and dwarfs pouring the budget into one leg ({aio.toFixed(3)}, a starved factor tanks the product). The product is a
        lower bound the real fused pipeline sits above; full budget everywhere drives it to 1.0 (Panel D's collapse anchor).
      </p>
    </div>
  );
}

// ===== Panel D — storage + collapse anchor ==========================================================
function StoragePanel() {
  const pw = 460, ph = 150, padL = 120, padR = 70, padT = 12, rowH = 36, gap = 12;
  const xMax = STORAGE.raw_multi_bits;
  const bw = (v: number) => (pw - padL - padR) * (v / xMax);
  const rows = [
    { label: 'single vector', bits: STORAGE.single_bits, mult: '1×', color: '#6a8caf' },
    { label: 'raw multi-vector', bits: STORAGE.raw_multi_bits, mult: `${STORAGE.raw_mult}×`, color: MUTED },
    { label: 'PLAID (id + residual)', bits: STORAGE.plaid_bits, mult: `${STORAGE.plaid_mult}×`, color: POS_COLOR },
  ];
  return (
    <div>
      <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="storage bits per document: single vs raw multi-vector vs PLAID" style={{ width: '100%', height: 'auto', display: 'block' }}>
        {rows.map((r, i) => {
          const y = padT + i * (rowH + gap);
          return (
            <g key={r.label}>
              <text x={padL - 8} y={y + rowH / 2 + 3} textAnchor="end" fontSize={10} fill="var(--color-text)" fontFamily="var(--font-sans)">{r.label}</text>
              <rect x={padL} y={y} width={Math.max(2, bw(r.bits))} height={rowH} rx={3} fill={r.color} opacity={0.85} />
              <text x={padL + Math.max(2, bw(r.bits)) + 6} y={y + rowH / 2 + 3} fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.bits.toLocaleString()} b · {r.mult}</text>
            </g>
          );
        })}
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
        <Readout label="raw multi-vector vs single" value={`${STORAGE.raw_mult}×`} />
        <Readout label="PLAID vs single" value={`${STORAGE.plaid_mult}×`} accent />
        <Readout label="collapse anchor recall" value={fmt(COLLAPSE_RECALL, 1)} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        At ColBERT scale (d = 128, 32 tokens/document), the served late-interaction leg's raw index is <strong>{STORAGE.raw_mult}×</strong> a single-vector
        index; PLAID's centroid-id-plus-residual collapses it back to <strong>{STORAGE.plaid_mult}×</strong>. And the one exact statement of the whole
        capstone: probe every cell, prune nothing, rerank by exact MaxSim, and the pipeline recovers brute-force retrieval —
        recall <strong>{COLLAPSE_RECALL.toFixed(1)}</strong>, the anchor that pins every relaxation above it as a strict superset.
      </p>
    </div>
  );
}

// ===== main component ==============================================================================
type Panel = 'cascade' | 'fusion' | 'budget' | 'storage';

export default memo(function CapstoneLaboratory() {
  const [panel, setPanel] = useState<Panel>('cascade');
  const [rhoIdx, setRhoIdx] = useState(3);          // ρ = 0.3 (positive dependence) by default
  const [fracIdx, setFracIdx] = useState(4);        // disjoint legs by default
  const [showFlip, setShowFlip] = useState(false);
  const [budget, setBudget] = useState(2650);       // the demo budget (BUDGET_FRAC of all-max cost)
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'cascade'
        ? 'R = \\prod_i r_i, \\qquad \\text{over-fetch} = \\tfrac{1}{\\prod_i r_i}, \\qquad \\text{(positive dependence)}\\;\\; R_{\\text{true}} \\ge \\prod_i r_i'
        : panel === 'fusion'
          ? '\\rho_{\\text{fused}} > \\max_L \\rho_L \\;\\text{ iff the legs cover disjoint truths;}\\;\\; \\mathrm{RRF}(d) = \\sum_L \\tfrac{1}{c + r_L(d)}'
          : panel === 'budget'
            ? '\\max_{\\sum_i c_i \\le B} \\prod_i g_i(c_i) \\;\\Longrightarrow\\; \\frac{g_i\'(c_i)}{g_i(c_i)} = \\lambda \\;\\;\\text{(water-filling)}'
            : '\\text{probe all} \\,\\wedge\\, \\text{prune none} \\,\\wedge\\, \\text{exact rerank} \\implies \\text{recall} = 1';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="capstone" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'cascade')} onClick={() => setPanel('cascade')}>A · cascade recall</button>
        <button type="button" style={pill(panel === 'fusion')} onClick={() => setPanel('fusion')}>B · hybrid fusion</button>
        <button type="button" style={pill(panel === 'budget')} onClick={() => setPanel('budget')}>C · budget water-filling</button>
        <button type="button" style={pill(panel === 'storage')} onClick={() => setPanel('storage')}>D · storage + anchor</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'cascade' && <CascadePanel rhoIdx={rhoIdx} setRhoIdx={setRhoIdx} />}
      {panel === 'fusion' && <FusionPanel fracIdx={fracIdx} setFracIdx={setFracIdx} showFlip={showFlip} setShowFlip={setShowFlip} />}
      {panel === 'budget' && <BudgetPanel budget={budget} setBudget={setBudget} />}
      {panel === 'storage' && <StoragePanel />}
    </div>
  );
});
