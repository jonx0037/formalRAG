import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Multi-Vector ANN Laboratory — four panels for the `multi-vector-ann-retrieval` (PLAID) topic:
 *   A. Representation. A 2-D token cloud, its shared k-means centroids, and each token's residual to
 *      its centroid; raising the number of centroids K shrinks the residuals (the quantity the bound
 *      depends on).
 *   B. Centroid-MaxSim + the Cauchy-Schwarz bound. The query x document-token grid scored by the true
 *      inner product and by the centroid substitution, with the per-cell error and its bound; cells
 *      where the centroid argmax differs from the true argmax are flagged (the bound controls scores,
 *      not the ranking).
 *   C. The cascade frontier. Recall vs distance-computations-per-query for brute MaxSim (the anchor),
 *      centroid-only (a cheap ceiling), and the PLAID cascade with an exact rerank (climbs to the
 *      brute line at full keep — the collapse anchor) and the deployed PQ rerank (plateaus below).
 *   D. Storage. Single-vector vs raw multi-vector (32x) vs PLAID (centroid id + PQ residual, ~1x).
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): POOL_2D, TOPIC_MU_2D, QUERY_2D, DOC_2D, GEOM_SWEEP,
 * BUDGET_TRADE, KEEP_SWEEP, BRUTE, CENTROID_ONLY, KNEE_KEEP, and STORAGE are mirrored TO THE DECIMAL
 * from notebooks/multi-vector-ann-retrieval/multi_vector_ann_retrieval.py (viz_constants()).
 * test_toy_bound_tightens_with_k / test_cascade_collapses_to_brute / test_recall_monotone_in_keep /
 * test_plaid_beats_brute_at_equal_recall / test_storage_collapse assert these. The lab recomputes only
 * CLOSED FORM in TS (inner products of the baked tokens/centroids, per-cell error and the bound
 * ||q||*||r||, the per-row maxima); only the k-means centroids themselves are baked. Change a number
 * here -> change it there, and re-run the notebook. Sliders only (no d3 drag).
 */

// --- baked from viz_constants() -------------------------------------------------------
// Panels A & B: a 2-D unit-norm token toy. The pool trains the centroids; one query (3 tokens) is
// scored against one document (4 tokens). GEOM_SWEEP holds, per number-of-centroids K, the trained
// centroids and the pool/doc assignments; everything else is recomputed in TS as a closed form.
const POOL_2D = [
  [0.903, 0.43], [0.613, 0.79], [0.967, 0.255], [0.994, 0.11], [0.707, 0.707], [0.916, 0.4],
  [-0.429, 0.903], [-0.536, 0.844], [-0.432, 0.902], [-0.584, 0.812], [0.067, 0.998], [-0.187, 0.982],
  [-0.856, -0.516], [-0.818, -0.576], [-0.995, -0.101], [-0.995, -0.098], [-0.98, -0.197], [-0.943, 0.333],
  [0.171, -0.985], [0.145, -0.989], [-0.205, -0.979], [-0.536, -0.844], [-0.058, -0.998], [-0.235, -0.972],
];
const QUERY_2D = [[0.865, 0.502], [-0.994, -0.11], [-0.62, 0.784]];
const DOC_2D = [[0.998, -0.057], [0.923, 0.385], [-0.991, -0.135], [-0.945, -0.328]];

type GeomK = { K: number; centroids: number[][]; pool_assign: number[]; doc_assign: number[] };
const GEOM_SWEEP: GeomK[] = [
  { K: 2, centroids: [[-0.525, -0.577], [0.25, 0.678]],
    pool_assign: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], doc_assign: [1, 1, 0, 0] },
  { K: 4, centroids: [[-0.35, 0.907], [-0.931, -0.193], [0.85, 0.449], [-0.12, -0.961]],
    pool_assign: [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3], doc_assign: [2, 2, 1, 1] },
  { K: 8, centroids: [[-0.06, 0.99], [-0.258, -0.948], [0.945, 0.299], [0.158, -0.987], [-0.837, -0.546], [0.66, 0.749], [-0.978, -0.016], [-0.495, 0.865]],
    pool_assign: [2, 5, 2, 2, 5, 2, 7, 7, 7, 7, 0, 0, 4, 4, 6, 6, 6, 6, 3, 3, 1, 1, 1, 1], doc_assign: [2, 2, 6, 4] },
];
// per-K MaxSim values + the worst-cell error/bound + residual energy (cross-check against the TS closed forms)
const BUDGET_TRADE = [
  { K: 2, full: 2.4997, centroid: 1.5186, max_err: 1.0409, max_bound: 1.0492, residual_energy: 0.5722 },
  { K: 4, full: 2.4997, centroid: 2.3336, max_err: 0.4891, max_bound: 0.5275, residual_energy: 0.0782 },
  { K: 8, full: 2.4997, centroid: 2.5357, max_err: 0.3126, max_bound: 0.3603, residual_energy: 0.0528 },
];
const TRUE_MAXSIM = 2.5;

// Panel C: the cascade frontier on the 16-D corpus, one shared truth. cost = distance comps / query.
const N_DOCS = 120, NLIST = 32, TOPK = 10, M_Q = 4, M_D = 8;
const BRUTE = { recall: 1.0, cost: 3840.0 };
const CENTROID_ONLY = { recall: 0.78, cost: 128.0 };
const KNEE_KEEP = 40;
const KEEP_SWEEP = [
  { keep: 5, cost: 288.0, recall_pq: 0.48, recall_exact: 0.48 },
  { keep: 10, cost: 448.0, recall_pq: 0.78, recall_exact: 0.78 },
  { keep: 20, cost: 768.0, recall_pq: 0.89, recall_exact: 0.9425 },
  { keep: 40, cost: 1408.0, recall_pq: 0.915, recall_exact: 0.9925 },
  { keep: 80, cost: 2688.0, recall_pq: 0.9125, recall_exact: 1.0 },
  { keep: 120, cost: 3968.0, recall_pq: 0.9125, recall_exact: 1.0 },
];

// Panel D: bits per document at ColBERT scale (d=128, 32 tokens), consistent with the late-interaction lab.
const STORAGE = { single_bits: 4096, raw_multi_bits: 131072, plaid_bits: 4608, raw_mult: 32.0, plaid_mult: 1.1, plaid_vs_raw: 28.4 };

const POS_COLOR = '#5fa873';     // the cascade / win
const ACCENT = 'var(--color-accent)';
const HARD_COLOR = '#7C3AED';    // the bound / argmax-disagreement / prune line
const CEIL_COLOR = '#6a8caf';    // centroid-only ceiling
const MUTED = '#9aa3ad';
const CLUSTER_COLORS = ['#5fa873', '#6a8caf', '#c08457', '#9b6abf', '#c25b6b', '#4f9da6', '#b08a3e', '#8a7d6b'];

const clampIdx = (i: number, n: number) => Math.max(0, Math.min(Math.round(i), n - 1));
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1];
const norm2 = (a: number[]) => Math.hypot(a[0], a[1]);
const argmax = (r: number[]) => r.indexOf(Math.max(...r));

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
      <span style={{ minWidth: '12rem' }}>{label} = <strong>{fmt(value)}</strong></span>
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

// closed-form grid + bound for a given K (only the centroids are baked) -------------------------------
function gridFor(g: GeomK) {
  const Dc = DOC_2D.map((_, j) => g.centroids[g.doc_assign[j]]);          // centroid of each doc token
  const simTrue = QUERY_2D.map((q) => DOC_2D.map((d) => dot(q, d)));
  const simCent = QUERY_2D.map((q) => Dc.map((c) => dot(q, c)));
  const rNorms = DOC_2D.map((d, j) => norm2([d[0] - Dc[j][0], d[1] - Dc[j][1]]));
  const qNorms = QUERY_2D.map((q) => norm2(q));
  const bounds = qNorms.map((qn) => rNorms.map((rn) => qn * rn));
  const errs = simTrue.map((row, i) => row.map((v, j) => Math.abs(v - simCent[i][j])));
  const trueArg = simTrue.map(argmax);
  const centArg = simCent.map(argmax);
  const trueMaxsim = simTrue.reduce((s, r) => s + Math.max(...r), 0);
  const centMaxsim = simCent.reduce((s, r) => s + Math.max(...r), 0);
  const resEnergy = rNorms.reduce((s, r) => s + r * r, 0) / rNorms.length;
  return { Dc, simCent, rNorms, bounds, errs, trueArg, centArg, trueMaxsim, centMaxsim, resEnergy };
}

// ===== Panel A — the token cloud, centroids, and residuals ==========================================
function TokenCloud({ g }: { g: GeomK }) {
  const W = 320, H = 320, cx = W / 2, cy = H / 2, R = 130;
  const px = (p: number[]) => cx + p[0] * R;
  const py = (p: number[]) => cy - p[1] * R;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="token cloud with centroids and residuals" style={{ width: '100%', maxWidth: 360, height: 'auto', display: 'block', margin: '0 auto' }}>
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--color-border)" strokeWidth={1} strokeDasharray="3 4" />
      {/* residual lines from each pool token to its centroid */}
      {POOL_2D.map((p, i) => {
        const c = g.centroids[g.pool_assign[i]];
        return <line key={`r-${i}`} x1={px(p)} y1={py(p)} x2={px(c)} y2={py(c)} stroke={CLUSTER_COLORS[g.pool_assign[i] % CLUSTER_COLORS.length]} strokeWidth={0.8} opacity={0.45} />;
      })}
      {/* pool tokens */}
      {POOL_2D.map((p, i) => (
        <circle key={`p-${i}`} cx={px(p)} cy={py(p)} r={3.4} fill={CLUSTER_COLORS[g.pool_assign[i] % CLUSTER_COLORS.length]} opacity={0.9} />
      ))}
      {/* centroids */}
      {g.centroids.map((c, i) => (
        <g key={`c-${i}`}>
          <circle cx={px(c)} cy={py(c)} r={6.5} fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} stroke="var(--color-bg)" strokeWidth={1.6} />
          <circle cx={px(c)} cy={py(c)} r={6.5} fill="none" stroke="var(--color-text)" strokeWidth={0.8} opacity={0.5} />
        </g>
      ))}
    </svg>
  );
}

// ===== Panel B — the centroid-MaxSim grid + Cauchy-Schwarz bound ====================================
function ApproxGrid({ g }: { g: GeomK }) {
  const G = gridFor(g);
  const nq = QUERY_2D.length, nd = DOC_2D.length;
  const cell = 64, padL = 36, padT = 24, W = padL + nd * cell + 8, H = padT + nq * cell + 10;
  const simTrue = QUERY_2D.map((q) => DOC_2D.map((d) => dot(q, d)));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="centroid-MaxSim grid with per-cell error and bound" style={{ width: '100%', height: 'auto', display: 'block' }}>
      {DOC_2D.map((_, j) => (
        <text key={`dh-${j}`} x={padL + j * cell + cell / 2} y={padT - 8} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">d{j + 1}</text>
      ))}
      {QUERY_2D.map((_, i) => (
        <g key={`row-${i}`}>
          <text x={padL - 8} y={padT + i * cell + cell / 2 + 3} textAnchor="end" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">q{i + 1}</text>
          {DOC_2D.map((_, j) => {
            const t = simTrue[i][j], c = G.simCent[i][j], err = G.errs[i][j], bnd = G.bounds[i][j];
            const isTrueMax = j === G.trueArg[i];
            const isCentMax = j === G.centArg[i];
            const disagree = isCentMax && !isTrueMax;
            const x = padL + j * cell, y = padT + i * cell;
            return (
              <g key={`c-${i}-${j}`}>
                <rect x={x} y={y} width={cell - 4} height={cell - 4} rx={3}
                  fill={t >= 0 ? ACCENT : MUTED} opacity={Math.min(1, Math.abs(t) * 0.7 + 0.08)}
                  stroke={isTrueMax ? POS_COLOR : disagree ? HARD_COLOR : 'none'}
                  strokeWidth={isTrueMax ? 2.4 : disagree ? 2.4 : 0}
                  strokeDasharray={disagree ? '3 2' : undefined} />
                <text x={x + (cell - 4) / 2} y={y + 16} textAnchor="middle" fontSize={9.5} fontWeight={600}
                  fill={Math.abs(t) > 0.5 ? 'var(--color-bg)' : 'var(--color-text)'} fontFamily="var(--font-sans)">{t.toFixed(2)}</text>
                <text x={x + (cell - 4) / 2} y={y + 29} textAnchor="middle" fontSize={8}
                  fill={Math.abs(t) > 0.5 ? 'var(--color-bg)' : 'var(--color-text-secondary)'} fontFamily="var(--font-sans)">≈ {c.toFixed(2)}</text>
                {/* error bar (filled) within the bound track */}
                <rect x={x + 8} y={y + cell - 18} width={cell - 20} height={5} rx={2.5} fill="var(--color-bg)" opacity={0.55} />
                <rect x={x + 8} y={y + cell - 18} width={(cell - 20) * Math.min(1, bnd / 1.05)} height={5} rx={2.5} fill={HARD_COLOR} opacity={0.28} />
                <rect x={x + 8} y={y + cell - 18} width={(cell - 20) * Math.min(1, err / 1.05)} height={5} rx={2.5} fill={HARD_COLOR} opacity={0.85} />
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

// ===== Panel C — the recall/cost frontier ===========================================================
function Frontier({ kIdx }: { kIdx: number }) {
  const pw = 460, ph = 250, padL = 50, padR = 18, padT = 18, padB = 42;
  const xMax = 4000, yMin = 0.4, yMax = 1.02;
  const fx = (c: number) => padL + (pw - padL - padR) * (c / xMax);
  const fy = (r: number) => ph - padB - (ph - padT - padB) * ((r - yMin) / (yMax - yMin));
  const pathFor = (key: 'recall_pq' | 'recall_exact') =>
    KEEP_SWEEP.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.cost).toFixed(1) + ' ' + fy(d[key]).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="recall versus distance computations per query" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={padL} y1={ph - padB} x2={pw - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={padL} y1={padT} x2={padL} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      {[0.4, 0.6, 0.8, 1.0].map((v) => (<text key={v} x={padL - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {[0, 1000, 2000, 3000, 4000].map((v) => (<text key={v} x={fx(v)} y={ph - padB + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
      <text x={(padL + pw - padR) / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">distance computations / query</text>
      <text x={13} y={(padT + ph - padB) / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${(padT + ph - padB) / 2})`}>recall@{TOPK}</text>

      {/* brute anchor line at recall 1.0 */}
      <line x1={padL} y1={fy(1.0)} x2={pw - padR} y2={fy(1.0)} stroke={MUTED} strokeWidth={1.2} strokeDasharray="5 4" />
      <circle cx={fx(BRUTE.cost)} cy={fy(BRUTE.recall)} r={4} fill="var(--color-text)" />
      <text x={fx(BRUTE.cost) - 6} y={fy(1.0) - 6} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">brute MaxSim</text>

      {/* centroid-only point (cheap ceiling) */}
      <circle cx={fx(CENTROID_ONLY.cost)} cy={fy(CENTROID_ONLY.recall)} r={4.5} fill={CEIL_COLOR} />
      <text x={fx(CENTROID_ONLY.cost) + 6} y={fy(CENTROID_ONLY.recall) + 12} fontSize={9} fill={CEIL_COLOR} fontFamily="var(--font-sans)">centroid-only</text>

      {/* PLAID cascade: exact rerank (dashed -> reaches brute) and PQ rerank (solid, plateaus) */}
      <path d={pathFor('recall_exact')} fill="none" stroke={POS_COLOR} strokeWidth={2.2} strokeDasharray="5 3" opacity={0.85} />
      <path d={pathFor('recall_pq')} fill="none" stroke={POS_COLOR} strokeWidth={3} />
      {KEEP_SWEEP.map((d, i) => (
        <g key={`pt-${i}`}>
          <circle cx={fx(d.cost)} cy={fy(d.recall_exact)} r={i === kIdx ? 5 : 3} fill={POS_COLOR} opacity={0.6} />
          <circle cx={fx(d.cost)} cy={fy(d.recall_pq)} r={i === kIdx ? 5.5 : 3.2} fill={POS_COLOR} />
        </g>
      ))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={padL + 16} y1={padT + 8} x2={padL + 34} y2={padT + 8} stroke={POS_COLOR} strokeWidth={3} />
        <text x={padL + 38} y={padT + 11} fill="var(--color-text-secondary)">PLAID (PQ rerank)</text>
        <line x1={padL + 16} y1={padT + 24} x2={padL + 34} y2={padT + 24} stroke={POS_COLOR} strokeWidth={2.2} strokeDasharray="5 3" />
        <text x={padL + 38} y={padT + 27} fill="var(--color-text-secondary)">PLAID (exact rerank)</text>
      </g>
    </svg>
  );
}

// ===== Panel D — storage bars =======================================================================
function StorageBars() {
  const pw = 460, ph = 168, padL = 116, padR = 64, padT = 14, rowH = 40, gap = 14;
  const xMax = STORAGE.raw_multi_bits;
  const bw = (v: number) => (pw - padL - padR) * (v / xMax);
  const rows = [
    { label: 'single vector', bits: STORAGE.single_bits, mult: '1×', color: CEIL_COLOR },
    { label: 'raw multi-vector', bits: STORAGE.raw_multi_bits, mult: `${STORAGE.raw_mult}×`, color: MUTED },
    { label: 'PLAID (id + residual)', bits: STORAGE.plaid_bits, mult: `${STORAGE.plaid_mult}×`, color: POS_COLOR },
  ];
  return (
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
  );
}

// ===== main component ==============================================================================
type Panel = 'represent' | 'approx' | 'cascade' | 'storage';

export default memo(function MultiVectorANNLaboratory() {
  const [panel, setPanel] = useState<Panel>('represent');
  const [kIdxA, setKIdxA] = useState(0);   // index into GEOM_SWEEP for Panels A & B
  const [keepIdx, setKeepIdx] = useState(0); // index into KEEP_SWEEP for Panel C
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'represent'
        ? 'c(d_j) = \\arg\\min_k \\lVert d_j - c_k\\rVert, \\qquad r_j = d_j - c(d_j), \\qquad \\hat d_j = c(d_j) + \\hat r_j'
        : panel === 'approx'
          ? '\\langle q_i, d_j\\rangle - \\langle q_i, c(d_j)\\rangle = \\langle q_i, r_j\\rangle, \\qquad |\\langle q_i, r_j\\rangle| \\le \\lVert q_i\\rVert\\,\\lVert r_j\\rVert'
          : panel === 'cascade'
            ? '\\text{probe all} \\,\\wedge\\, \\text{prune none} \\,\\wedge\\, \\text{exact rerank} \\implies \\text{cascade} = S(q,d)'
            : '\\text{bits/token} = \\lceil \\log_2 K\\rceil + m\\log_2 k^{*} \\;\\ll\\; d \\cdot 32';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const gA = GEOM_SWEEP[clampIdx(kIdxA, GEOM_SWEEP.length)];
  const bA = BUDGET_TRADE[clampIdx(kIdxA, BUDGET_TRADE.length)];
  const grid = gridFor(gA);
  const keepRow = KEEP_SWEEP[clampIdx(keepIdx, KEEP_SWEEP.length)];

  return (
    <div data-lab="multi-vector-ann" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'represent')} onClick={() => setPanel('represent')}>A · token → centroid + residual</button>
        <button type="button" style={pill(panel === 'approx')} onClick={() => setPanel('approx')}>B · centroid-MaxSim + bound</button>
        <button type="button" style={pill(panel === 'cascade')} onClick={() => setPanel('cascade')}>C · recall / cost frontier</button>
        <button type="button" style={pill(panel === 'storage')} onClick={() => setPanel('storage')}>D · storage</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'represent' && (
        <div>
          <Slider label="number of centroids K" value={kIdxA} min={0} max={GEOM_SWEEP.length - 1} step={1} onChange={setKIdxA} fmt={(v) => `${GEOM_SWEEP[clampIdx(v, GEOM_SWEEP.length)].K}`} />
          <TokenCloud g={gA} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            <Readout label="centroids K (the shared vocabulary)" value={`${gA.K}`} accent />
            <Readout label="mean residual energy ⟨‖r‖²⟩" value={bA.residual_energy.toFixed(3)} />
            <Readout label="corpus tokens / centroids" value={`${POOL_2D.length} / ${gA.K}`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Every token in the corpus is filed under its nearest of <strong>K shared centroids</strong> (the IVF coarse quantizer,
            trained over all tokens) and stored as that centroid's id plus the <em>residual</em> — the line from each token to its
            centroid. Tokens are unit vectors, so this k-means is cosine clustering. Raising K shrinks the residuals (mean energy
            {' '}{BUDGET_TRADE[0].residual_energy.toFixed(2)} → {BUDGET_TRADE[BUDGET_TRADE.length - 1].residual_energy.toFixed(2)}),
            which is exactly the quantity the next panel's bound depends on.
          </p>
        </div>
      )}

      {panel === 'approx' && (
        <div>
          <Slider label="number of centroids K" value={kIdxA} min={0} max={GEOM_SWEEP.length - 1} step={1} onChange={setKIdxA} fmt={(v) => `${GEOM_SWEEP[clampIdx(v, GEOM_SWEEP.length)].K}`} />
          <ApproxGrid g={gA} />
          <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            <Readout label="true MaxSim" value={grid.trueMaxsim.toFixed(3)} />
            <Readout label="centroid-MaxSim (approx)" value={grid.centMaxsim.toFixed(3)} accent />
            <Readout label="max per-cell error" value={Math.max(...grid.errs.flat()).toFixed(3)} />
            <Readout label="max bound ‖q‖·‖r‖" value={Math.max(...grid.bounds.flat()).toFixed(3)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Each cell shows the true inner product ⟨qᵢ, dⱼ⟩ and, below it, the centroid substitution
            ⟨qᵢ, c(dⱼ)⟩; the purple bar is the per-cell error, always inside its Cauchy–Schwarz bound
            ‖qᵢ‖·‖rⱼ‖ (the lighter track). A green border marks each query token's <em>true</em> best
            match; a dashed purple border marks a cell where the centroid approximation's argmax <em>disagrees</em> — small per-cell
            error, wrong ranking. That gap is why the cascade still reranks. Raising K shrinks both the error and the bound, and the
            centroid-MaxSim ({grid.centMaxsim.toFixed(2)}) approaches the true MaxSim ({TRUE_MAXSIM.toFixed(2)}).
          </p>
        </div>
      )}

      {panel === 'cascade' && (
        <div>
          <Slider label="prune depth keep" value={keepIdx} min={0} max={KEEP_SWEEP.length - 1} step={1} onChange={setKeepIdx} fmt={(v) => `${KEEP_SWEEP[clampIdx(v, KEEP_SWEEP.length)].keep}`} />
          <Frontier kIdx={clampIdx(keepIdx, KEEP_SWEEP.length)} />
          <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label={`PLAID recall (PQ) @ keep=${keepRow.keep}`} value={keepRow.recall_pq.toFixed(3)} accent />
            <Readout label="PLAID recall (exact rerank)" value={keepRow.recall_exact.toFixed(3)} />
            <Readout label="cost (comps/query)" value={`${keepRow.cost.toFixed(0)}`} />
            <Readout label="vs brute" value={`${(BRUTE.cost / keepRow.cost).toFixed(1)}× cheaper`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Brute MaxSim scores every query token against every document token — recall 1.0 at {BRUTE.cost} comps/query (the dashed
            anchor). Centroid-only pruning is ~{(BRUTE.cost / CENTROID_ONLY.cost).toFixed(0)}× cheaper but plateaus at recall
            {' '}{CENTROID_ONLY.recall.toFixed(2)} (it never reranks). The PLAID cascade probes, prunes by centroid-MaxSim, and reranks
            the survivors: with an <em>exact</em> rerank (dashed) it climbs to the brute line at keep = N — the collapse anchor — while
            the deployed PQ-compressed rerank (solid) reaches ~{Math.max(...KEEP_SWEEP.map((d) => d.recall_pq)).toFixed(2)} at the knee
            (keep ≈ {KNEE_KEEP}), the gap being the lossy residual compression. One synthetic cloud — the knee moves with the corpus.
          </p>
        </div>
      )}

      {panel === 'storage' && (
        <div>
          <StorageBars />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            <Readout label="raw multi-vector vs single" value={`${STORAGE.raw_mult}×`} />
            <Readout label="PLAID vs single" value={`${STORAGE.plaid_mult}×`} accent />
            <Readout label="PLAID compression of raw" value={`${STORAGE.plaid_vs_raw}×`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            At ColBERT scale (d = 128, 32 tokens/document), late interaction's raw index is <strong>{STORAGE.raw_mult}×</strong> a
            single-vector index — the bill the previous topic flagged. PLAID stores each token as a centroid id
            (⌈log₂ K⌉ bits) plus a product-quantized residual (m·log₂ k* bits), collapsing the index to
            {' '}<strong>{STORAGE.plaid_mult}×</strong> a single-vector index — a {STORAGE.plaid_vs_raw}× compression of the raw
            multi-vector store. Mitigation, not erasure: still many vectors per document, and the residual codes are lossy by
            construction (the recall gap in Panel C).
          </p>
        </div>
      )}
    </div>
  );
});
