import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Cross-Encoder & Reranking-Cascade Laboratory — three panels for the `cross-encoders-reranking` topic:
 *   A. The rank ceiling. On a full-rank (signed-identity) target, the best rank-d dual encoder — which is
 *      ALSO the best learned bilinear q^T W d, since S = QWG^T = (QW)G^T is still rank <= d — plateaus at a
 *      positive reconstruction error for d < n, while the nonlinear cross-encoder h([q;d]) reconstructs it
 *      at every d. The slider sweeps the embedding width d; the ceiling descends to 0 only at d = rank.
 *   B. The reranking cascade. A cheap rank-3 dual first stage retrieves a top-K pool; the cross-encoder
 *      reranks it. An ORACLE rerank lifts recall@1 to the pool's recall@K (the recall pinch) and caps it
 *      there; a real reranker captures part of the lift. The K slider drives the recall frontier and the
 *      cost c_ret + K c_ce against the brute |C| c_ce.
 *   C. When reranking hurts. A reranker-quality slider (the corruption sigma; 0 = oracle) buckets each
 *      query into fixed / kept / broke / missed: a good reranker fixes same-sector hard negatives (net
 *      lift > 0), a confidently-wrong one breaks true top-1s (net lift < 0) — the dip the oracle's
 *      monotonicity forbids.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): RECON_CURVE, CEILING_RANK, SIGN_SINGULAR_VALUES, CROSS_MATRIX_RANK,
 * CASCADE, STAGE1_R1, SIGMA_GRID, BUCKETS_BY_SIGMA, BUCKETS_CONFIDENT_WRONG, DIP_WITNESS, and the shared
 * scalars are mirrored TO THE DECIMAL from
 * notebooks/cross-encoders-reranking/cross_encoders_reranking.py (viz_constants()). The lab recomputes
 * only CLOSED FORM in TS — the cost arithmetic c_ret + K c_ce and its speedup, and the net lift. The
 * asserts test_cross_encoder_breaks_rank_ceiling / test_recall_pinch_identity /
 * test_oracle_rerank_monotone_in_k / test_lossy_rerank_can_dip pin these. Change a number here -> change
 * it there, and re-run the notebook. Sliders only (no d3 drag); SVG text inherits theme.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_QUERIES = 32;
const D_STAGE1 = 3;
const SIGN_N = 6;
const C_RETRIEVE = 1.0;
const C_CE = 25.0;
const CORPUS_HEADLINE = 1_000_000;

// Panel A: reconstruction error of the rank-d ceiling (best dual = best bilinear = truncated SVD) and the
// d-independent error of the nonlinear cross-encoder, on the full-rank signed-identity(6) target.
type ReconRow = { d: number; ceiling: number; cross: number };
const RECON_CURVE: ReconRow[] = [
  { d: 1, ceiling: 0.745356, cross: 0.0 },
  { d: 2, ceiling: 0.666667, cross: 0.0 },
  { d: 3, ceiling: 0.57735, cross: 0.0 },
  { d: 4, ceiling: 0.471405, cross: 0.0 },
  { d: 5, ceiling: 0.333333, cross: 0.0 },
  { d: 6, ceiling: 0.0, cross: 0.0 },
];
const CEILING_RANK = 6; // dual/bilinear hit 0 only at d = rank
const SIGN_SINGULAR_VALUES = [4.0, 2.0, 2.0, 2.0, 2.0, 2.0];
const CROSS_MATRIX_RANK = 6;

// Panel B: the cascade recall-vs-K frontier. stage1_r1 is flat; oracle == stage1_rk (the recall pinch);
// lossy is a decent-but-imperfect reranker (sigma = 0.25).
type CascadeRow = { K: number; stage1_r1: number; stage1_rk: number; oracle: number; lossy: number };
const CASCADE: CascadeRow[] = [
  { K: 1, stage1_r1: 0.7812, stage1_rk: 0.7812, oracle: 0.7812, lossy: 0.7812 },
  { K: 2, stage1_r1: 0.7812, stage1_rk: 0.9375, oracle: 0.9375, lossy: 0.8125 },
  { K: 3, stage1_r1: 0.7812, stage1_rk: 1.0, oracle: 1.0, lossy: 0.875 },
  { K: 4, stage1_r1: 0.7812, stage1_rk: 1.0, oracle: 1.0, lossy: 0.8125 },
  { K: 6, stage1_r1: 0.7812, stage1_rk: 1.0, oracle: 1.0, lossy: 0.7812 },
  { K: 8, stage1_r1: 0.7812, stage1_rk: 1.0, oracle: 1.0, lossy: 0.7812 },
];
const STAGE1_R1 = 0.7812;

// Panel C: per-query rerank buckets by reranker quality (the corruption sigma; 0 = oracle), at K = DIP_K.
const DIP_K = 8;
const SIGMA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0];
type Buckets = { sigma: number; fixed: number; kept: number; broke: number; missed: number; net_lift: number };
const BUCKETS_BY_SIGMA: Buckets[] = [
  { sigma: 0.0, fixed: 7, kept: 25, broke: 0, missed: 0, net_lift: 0.2188 },
  { sigma: 0.25, fixed: 4, kept: 21, broke: 4, missed: 3, net_lift: 0.0 },
  { sigma: 0.5, fixed: 3, kept: 16, broke: 9, missed: 4, net_lift: -0.1875 },
  { sigma: 1.0, fixed: 3, kept: 8, broke: 17, missed: 4, net_lift: -0.4375 },
  { sigma: 2.0, fixed: 2, kept: 5, broke: 20, missed: 5, net_lift: -0.5625 },
];
const BUCKETS_CONFIDENT_WRONG = { fixed: 0, kept: 0, broke: 25, missed: 7, net_lift: -0.7812 };
const DIP_WITNESS = { query: 0, gold: 0, stage1_top: 0, reranked_top: 1, sector_gold: 0, sector_wrong: 0 };

// --- colors & helpers -----------------------------------------------------------------
const ACCENT = 'var(--color-accent)';
const POS_COLOR = '#5fa873'; // fixed / cross-encoder / oracle
const NEG_COLOR = '#c0726a'; // broke (the confident-wrong dip)
const CEIL_COLOR = '#6a8caf'; // the rank-d ceiling / first stage
const KEPT_COLOR = '#7c93ab';
const MUTED = '#9aa3ad';

// round derived coordinates so SSR (Node) and client (browser) serialize identical strings.
const r2 = (v: number) => Math.round(v * 100) / 100;
const clampIdx = (i: number, n: number) => Math.max(0, Math.min(i, n - 1));
const fmt = (x: number, n = 3) => x.toFixed(n);

function Readout({ label, value, accent, danger }: { label: string; value: string; accent?: boolean; danger?: boolean }) {
  const color = danger ? NEG_COLOR : accent ? 'var(--color-accent)' : 'var(--color-text)';
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color }}>{value}</div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt: fm }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '13rem' }}>{label} = <strong>{fm(value)}</strong></span>
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

// ===== Panel A — the rank ceiling ==================================================================
function CeilingCurve({ dIdx }: { dIdx: number }) {
  const pw = 460, ph = 230, pad = 46;
  const n = RECON_CURVE.length;
  const yMax = 0.8;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (e: number) => ph - pad - (ph - 2 * pad) * Math.min(e, yMax) / yMax;
  const ceilPath = RECON_CURVE.map((r, i) => (i === 0 ? 'M' : 'L') + r2(fx(i)) + ' ' + r2(fy(r.ceiling))).join(' ');
  const crossPath = RECON_CURVE.map((r, i) => (i === 0 ? 'M' : 'L') + r2(fx(i)) + ' ' + r2(fy(r.cross))).join(' ');
  const rankX = fx(CEILING_RANK - 1);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="reconstruction error versus embedding dimension" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.4, 0.8].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {RECON_CURVE.map((r, i) => (<text key={i} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.d}</text>))}
      <text x={pw / 2} y={ph - 3} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">embedding dimension d</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>relative reconstruction error</text>
      {/* the rank marker: ceiling reaches 0 only here */}
      <line x1={rankX} y1={pad} x2={rankX} y2={ph - pad} stroke={CEIL_COLOR} strokeWidth={1.2} strokeDasharray="4 3" />
      <text x={rankX - 4} y={pad + 10} textAnchor="end" fontSize={9} fill={CEIL_COLOR} fontFamily="var(--font-sans)">rank = {CEILING_RANK}</text>
      <path d={ceilPath} fill="none" stroke={CEIL_COLOR} strokeWidth={2.8} />
      <path d={crossPath} fill="none" stroke={POS_COLOR} strokeWidth={2.8} />
      {RECON_CURVE.map((r, i) => (<circle key={`c${i}`} cx={fx(i)} cy={fy(r.ceiling)} r={i === dIdx ? 5.5 : 3} fill={CEIL_COLOR} opacity={i === dIdx ? 1 : 0.7} />))}
      {RECON_CURVE.map((r, i) => (<circle key={`x${i}`} cx={fx(i)} cy={fy(r.cross)} r={i === dIdx ? 5 : 2.6} fill={POS_COLOR} opacity={i === dIdx ? 1 : 0.7} />))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={pad + 14} y1={pad + 4} x2={pad + 30} y2={pad + 4} stroke={CEIL_COLOR} strokeWidth={2.6} />
        <text x={pad + 34} y={pad + 7} fill="var(--color-text-secondary)">dual = learned bilinear (rank ≤ d)</text>
        <line x1={pad + 14} y1={pad + 20} x2={pad + 30} y2={pad + 20} stroke={POS_COLOR} strokeWidth={2.6} />
        <text x={pad + 34} y={pad + 23} fill="var(--color-text-secondary)">cross-encoder h([q;d])</text>
      </g>
    </svg>
  );
}

// ===== Panel B — the cascade frontier =============================================================
function CascadeRecall({ kIdx }: { kIdx: number }) {
  const pw = 460, ph = 230, pad = 46;
  const n = CASCADE.length;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (r: number) => ph - pad - (ph - 2 * pad) * r;
  const oraclePath = CASCADE.map((r, i) => (i === 0 ? 'M' : 'L') + r2(fx(i)) + ' ' + r2(fy(r.oracle))).join(' ');
  const lossyPath = CASCADE.map((r, i) => (i === 0 ? 'M' : 'L') + r2(fx(i)) + ' ' + r2(fy(r.lossy))).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="recall at 1 versus over-fetch depth K" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {CASCADE.map((r, i) => (<text key={i} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.K}</text>))}
      <text x={pw / 2} y={ph - 3} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">over-fetch depth K (pool size)</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>recall@1 (32 queries)</text>
      {/* stage-1 recall@1 baseline (flat) */}
      <line x1={pad} y1={fy(STAGE1_R1)} x2={pw - pad} y2={fy(STAGE1_R1)} stroke={CEIL_COLOR} strokeWidth={1.4} strokeDasharray="5 3" />
      <text x={pw - pad} y={fy(STAGE1_R1) - 4} textAnchor="end" fontSize={9} fill={CEIL_COLOR} fontFamily="var(--font-sans)">first stage recall@1</text>
      <path d={oraclePath} fill="none" stroke={POS_COLOR} strokeWidth={2.8} />
      <path d={lossyPath} fill="none" stroke={ACCENT} strokeWidth={2.4} strokeDasharray="2 2" />
      {CASCADE.map((r, i) => (<circle key={`o${i}`} cx={fx(i)} cy={fy(r.oracle)} r={i === kIdx ? 5.5 : 3} fill={POS_COLOR} opacity={i === kIdx ? 1 : 0.7} />))}
      {CASCADE.map((r, i) => (<circle key={`l${i}`} cx={fx(i)} cy={fy(r.lossy)} r={i === kIdx ? 5 : 2.6} fill={ACCENT} opacity={i === kIdx ? 1 : 0.7} />))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={pad + 14} y1={pad + 4} x2={pad + 30} y2={pad + 4} stroke={POS_COLOR} strokeWidth={2.6} />
        <text x={pad + 34} y={pad + 7} fill="var(--color-text-secondary)">oracle rerank = pool recall@K</text>
        <line x1={pad + 14} y1={pad + 20} x2={pad + 30} y2={pad + 20} stroke={ACCENT} strokeWidth={2.6} strokeDasharray="2 2" />
        <text x={pad + 34} y={pad + 23} fill="var(--color-text-secondary)">a real reranker</text>
      </g>
    </svg>
  );
}

function CascadeCost({ kIdx }: { kIdx: number }) {
  const pw = 460, ph = 230, pad = 46;
  const n = CASCADE.length;
  const brute = CORPUS_HEADLINE * C_CE;
  const cost = (K: number) => C_RETRIEVE + K * C_CE;
  const yTop = Math.log10(brute) + 0.3;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (c: number) => ph - pad - (ph - 2 * pad) * (Math.log10(Math.max(c, 1))) / yTop;
  const cascadePath = CASCADE.map((r, i) => (i === 0 ? 'M' : 'L') + r2(fx(i)) + ' ' + r2(fy(cost(r.K)))).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="cost versus over-fetch depth K" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 2, 4, 6, 8].map((e) => (<text key={e} x={pad - 6} y={fy(Math.pow(10, e)) + 3} textAnchor="end" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">10{e === 0 ? '⁰' : ''}<tspan dy={-3} fontSize={6.5}>{e === 0 ? '' : e}</tspan></text>))}
      {CASCADE.map((r, i) => (<text key={i} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.K}</text>))}
      <text x={pw / 2} y={ph - 3} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">over-fetch depth K</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>cross-encoder calls (cost units)</text>
      {/* brute: score the whole corpus */}
      <line x1={pad} y1={fy(brute)} x2={pw - pad} y2={fy(brute)} stroke={NEG_COLOR} strokeWidth={1.6} strokeDasharray="5 3" />
      <text x={pw - pad} y={fy(brute) - 4} textAnchor="end" fontSize={9} fill={NEG_COLOR} fontFamily="var(--font-sans)">brute |C| · c_ce</text>
      <path d={cascadePath} fill="none" stroke={ACCENT} strokeWidth={2.8} />
      {CASCADE.map((r, i) => (<circle key={i} cx={fx(i)} cy={fy(cost(r.K))} r={i === kIdx ? 5.5 : 3} fill={ACCENT} opacity={i === kIdx ? 1 : 0.7} />))}
      <text x={fx(n - 1)} y={fy(cost(CASCADE[n - 1].K)) - 8} textAnchor="end" fontSize={9} fill={ACCENT} fontFamily="var(--font-sans)">cascade c_ret + K·c_ce</text>
    </svg>
  );
}

// ===== Panel C — when reranking hurts =============================================================
function QualityBuckets({ sIdx }: { sIdx: number }) {
  const b = BUCKETS_BY_SIGMA[clampIdx(sIdx, BUCKETS_BY_SIGMA.length)];
  const pw = 460, ph = 120, pad = 12;
  const total = N_QUERIES;
  const segs: { key: string; n: number; color: string; label: string }[] = [
    { key: 'fixed', n: b.fixed, color: POS_COLOR, label: 'fixed' },
    { key: 'kept', n: b.kept, color: KEPT_COLOR, label: 'kept' },
    { key: 'broke', n: b.broke, color: NEG_COLOR, label: 'broke' },
    { key: 'missed', n: b.missed, color: MUTED, label: 'missed' },
  ];
  const barW = pw - 2 * pad;
  let x = pad;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="per-query rerank outcome buckets" style={{ width: '100%', height: 'auto', display: 'block' }}>
      {segs.map((s) => {
        const w = barW * (s.n / total);
        const seg = (
          <g key={s.key}>
            <rect x={r2(x)} y={28} width={r2(Math.max(0, w - 1))} height={34} rx={3} fill={s.color} opacity={0.85} />
            {s.n > 0 && <text x={r2(x + w / 2)} y={49} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--color-bg)" fontFamily="var(--font-sans)">{s.n}</text>}
            <text x={r2(x + w / 2)} y={76} textAnchor="middle" fontSize={9} fill={s.color} fontFamily="var(--font-sans)">{s.n > 0 ? s.label : ''}</text>
          </g>
        );
        x += w;
        return seg;
      })}
      <text x={pad} y={20} fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">32 queries · reranked top-{DIP_K}</text>
      <text x={pw - pad} y={20} textAnchor="end" fontSize={10} fontWeight={600} fill={b.net_lift >= 0 ? POS_COLOR : NEG_COLOR} fontFamily="var(--font-sans)">net lift {b.net_lift >= 0 ? '+' : ''}{b.net_lift.toFixed(3)}</text>
      <text x={pw / 2} y={ph - 12} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">fixed = hard-negative win · broke = confidently-wrong dip below the first stage</text>
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'expressivity' | 'cascade' | 'quality';

export default memo(function CrossEncoderRerankingLaboratory() {
  const [panel, setPanel] = useState<Panel>('expressivity');
  const [dIdx, setDIdx] = useState(0);   // index into RECON_CURVE (d = 1)
  const [kIdx, setKIdx] = useState(2);   // index into CASCADE (K = 3)
  const [sIdx, setSIdx] = useState(0);   // index into SIGMA_GRID (sigma = 0, the oracle)
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'expressivity'
        ? 'S_{\\text{bilinear}} = Q W G^{\\top} = (QW) G^{\\top}\\ (\\operatorname{rank}\\le d)\\quad\\text{vs}\\quad s = h([q;d])\\ \\text{(no rank ceiling)}'
        : panel === 'cascade'
          ? '\\underbrace{r@1_{\\text{oracle-rerank}} = r@K_{\\text{stage-1}}}_{\\text{recall pinch}},\\qquad \\text{cost} = c_{\\text{ret}} + K\\,c_{\\text{ce}} \\;\\ll\\; |\\mathcal{C}|\\,c_{\\text{ce}}'
          : '\\Delta r@1 = \\tfrac{1}{N}\\big(\\#\\{\\text{stage-1 wrong},\\,\\text{CE right}\\} - \\#\\{\\text{stage-1 right},\\,\\text{CE wrong}\\}\\big)';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const dRow = RECON_CURVE[clampIdx(dIdx, RECON_CURVE.length)];
  const kRow = CASCADE[clampIdx(kIdx, CASCADE.length)];
  const sRow = BUCKETS_BY_SIGMA[clampIdx(sIdx, BUCKETS_BY_SIGMA.length)];

  // closed-form cost recompute (TS owns this, not the notebook)
  const cascadeCost = C_RETRIEVE + kRow.K * C_CE;
  const bruteCost = CORPUS_HEADLINE * C_CE;
  const speedup = bruteCost / cascadeCost;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'expressivity')} onClick={() => setPanel('expressivity')}>A · the rank ceiling</button>
        <button style={pill(panel === 'cascade')} onClick={() => setPanel('cascade')}>B · the reranking cascade</button>
        <button style={pill(panel === 'quality')} onClick={() => setPanel('quality')}>C · when reranking hurts</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'expressivity' && (
        <div>
          <Slider label="embedding dimension d" value={dIdx} min={0} max={RECON_CURVE.length - 1} step={1} onChange={setDIdx} fmt={(v) => `${RECON_CURVE[clampIdx(v, RECON_CURVE.length)].d}`} />
          <CeilingCurve dIdx={clampIdx(dIdx, RECON_CURVE.length)} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label={`rank-${dRow.d} ceiling error (dual = bilinear)`} value={fmt(dRow.ceiling, 3)} />
            <Readout label="cross-encoder error" value={fmt(dRow.cross, 3)} accent />
            <Readout label="cross-encoder matrix rank" value={`${CROSS_MATRIX_RANK} (full)`} />
            <Readout label="ceiling reaches 0 only at" value={`d = ${CEILING_RANK}`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The target is the signed-identity relevance pattern — full rank {SIGN_N}, with singular values{' '}
            {SIGN_SINGULAR_VALUES[0].toFixed(0)}, {SIGN_SINGULAR_VALUES.slice(1).map((v) => v.toFixed(0)).join(', ')}.
            A dual encoder can only realize a rank-≤d matrix, and a <strong>learned bilinear</strong> q<sup>⊤</sup>Wd
            does not escape that: S = QWG<sup>⊤</sup> = (QW)G<sup>⊤</sup> is still a product through a d-dimensional
            bottleneck, so its best is the same truncated-SVD ceiling (the blue line). Only the nonlinear cross-encoder
            h([q;d]), which scores the fused pair rather than a factorization, reconstructs the target at every d — it has
            no rank to bound. This is the escape from the sign-rank wall the embedding-dimension topic proved; the
            cross-encoder is, as that topic put it, <em>not a factorization at all</em>.
          </p>
        </div>
      )}

      {panel === 'cascade' && (
        <div>
          <Slider label="over-fetch depth K (pool size)" value={kIdx} min={0} max={CASCADE.length - 1} step={1} onChange={setKIdx} fmt={(v) => `${CASCADE[clampIdx(v, CASCADE.length)].K}`} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem', alignItems: 'center' }}>
            <CascadeRecall kIdx={clampIdx(kIdx, CASCADE.length)} />
            <CascadeCost kIdx={clampIdx(kIdx, CASCADE.length)} />
          </div>
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="first stage recall@1" value={fmt(STAGE1_R1, 3)} />
            <Readout label={`oracle rerank recall@1 (= pool recall@${kRow.K})`} value={fmt(kRow.oracle, 3)} accent />
            <Readout label="a real reranker recall@1" value={fmt(kRow.lossy, 3)} />
            <Readout label={`cascade cost (K = ${kRow.K})`} value={`${cascadeCost.toFixed(0)} vs ${bruteCost.toLocaleString()}`} />
            <Readout label="speedup over brute" value={`${Math.round(speedup).toLocaleString()}×`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            A rank-{D_STAGE1} dual encoder retrieves a top-K pool ({fmt(STAGE1_R1, 3)} recall@1 — it confuses
            same-sector companies); the cross-encoder reranks only those K. An <strong>oracle</strong> reranker lifts
            recall@1 to exactly the pool's recall@K — the <em>recall pinch</em> — and can never exceed it, since it cannot
            recover a gold the first stage dropped. So you over-fetch until recall@K saturates (here K = {CASCADE.find((r) => r.oracle >= 1)?.K}),
            then rerank: the cost is c_ret + K·c_ce, not the brute |C|·c_ce of scoring the whole corpus. A real reranker
            (dashed) captures part of the lift — and, foreshadowing the next panel, can even decline as K grows.
          </p>
        </div>
      )}

      {panel === 'quality' && (
        <div>
          <Slider label="reranker corruption σ (0 = oracle)" value={sIdx} min={0} max={SIGMA_GRID.length - 1} step={1} onChange={setSIdx} fmt={(v) => `${SIGMA_GRID[clampIdx(v, SIGMA_GRID.length)].toFixed(2)}`} />
          <QualityBuckets sIdx={clampIdx(sIdx, BUCKETS_BY_SIGMA.length)} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="hard negatives fixed" value={`${sRow.fixed}`} accent />
            <Readout label="true top-1s broken" value={`${sRow.broke}`} danger={sRow.broke > sRow.fixed} />
            <Readout label="net recall@1 lift" value={`${sRow.net_lift >= 0 ? '+' : ''}${sRow.net_lift.toFixed(3)}`} accent={sRow.net_lift >= 0} danger={sRow.net_lift < 0} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            At σ = 0 the reranker is the oracle: it fixes all {BUCKETS_BY_SIGMA[0].fixed} same-sector hard negatives the
            first stage missed (net lift {BUCKETS_BY_SIGMA[0].net_lift >= 0 ? '+' : ''}{BUCKETS_BY_SIGMA[0].net_lift.toFixed(3)}),
            and its recall is monotone in K. Corrupt it and it starts being <strong>confidently wrong</strong> — promoting a
            same-sector distractor above the true document — so it <em>breaks</em> queries the first stage got right and the
            net lift turns negative. A worst-case confident-wrong reranker breaks every recoverable query
            (net {BUCKETS_CONFIDENT_WRONG.net_lift.toFixed(3)}). The dip witness: query {DIP_WITNESS.query}'s gold
            (document {DIP_WITNESS.gold}, sector {DIP_WITNESS.sector_gold}) demoted below document {DIP_WITNESS.reranked_top}
            of the same sector. Monotonicity in K is a guarantee for the <em>oracle</em> only; a real reranker can make more
            over-fetch hurt.
          </p>
        </div>
      )}
    </div>
  );
});
