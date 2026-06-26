import { memo, useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Dense Dual-Encoder Laboratory — three panels for the `dense-retrieval-dual-encoders` topic:
 *   A. Factorization & MIPS. A bi-encoder's separable score lets the document matrix be encoded ONCE
 *      offline, so a query is answered by one matrix-vector product + top-k (query-time cost constant in
 *      the corpus); a cross-encoder must run |C| joint forward passes per query. The toggle contrasts the
 *      two; the cost curve shows the bi-encoder flat and the cross-encoder linear in the corpus size.
 *   B. The rank-d ceiling. A query-by-document relevance heatmap with finance-sector block structure, and
 *      its best rank-d reconstruction (truncated SVD). Sliding the embedding dimension d below the
 *      pattern's intrinsic rank smears the blocks and recall@1 collapses; at d >= rank it saturates.
 *   C. The in-batch Gram trick. The B x B Gram matrix (positives on the diagonal); the batch-size slider
 *      shows B(B-1) in-batch negatives appearing from only 2B encoder forward passes — quadratic from
 *      linear. One row is the single InfoNCE problem the prerequisite formalized.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): COST_VS_CORPUS, SINGULAR_VALUES, RANK_RECALL, D_RECOVER,
 * DISPLAY_MATRIX, DISPLAY_SECTOR, DISPLAY_U/S/VT, COUNTING, GRAM_TAU, INBATCH_LOSS are mirrored TO THE
 * DECIMAL from notebooks/dense-retrieval-dual-encoders/dense_retrieval_dual_encoders.py (viz_constants()).
 * The lab recomputes only CLOSED FORM in TS — the per-query cost arithmetic, B(B-1) and 2B, the heatmap
 * color scale, and the truncated-SVD reconstruction M_d = sum_{k<d} s_k u_k v_k^T from the baked (U,s,Vt).
 * test_rank_ceiling_recall / test_eckart_young_crosscheck / test_counting_law /
 * test_inbatch_equals_imported_infonce assert these. Change a number here -> change it there, and re-run
 * the notebook. The Gram matrix in Panel C IS the relevance matrix of Panel B (the batch is the company
 * set), so DISPLAY_MATRIX serves both. Sliders only (no d3 drag); SVG text inherits theme.
 */

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: query-time forward passes vs corpus size. Bi-encoder is constant (encode the query once, then
// MIPS over the precomputed matrix); cross-encoder is linear (one joint pass per document). Arithmetic,
// grid baked. Headline scale: a Wikipedia-size passage index at a real embedding dimension.
const COST_VS_CORPUS = [
  { corpus: 10, bi_passes: 1, cross_passes: 10 },
  { corpus: 100, bi_passes: 1, cross_passes: 100 },
  { corpus: 1000, bi_passes: 1, cross_passes: 1000 },
  { corpus: 10000, bi_passes: 1, cross_passes: 10000 },
  { corpus: 100000, bi_passes: 1, cross_passes: 100000 },
];
const CORPUS_HEADLINE = 21_000_000;

// Panel B: the finance score matrix S = Q P^T (32 queries x 8 company documents, dim 32). Its singular
// values and the recall@1/recall@3 of the best rank-d dual encoder, measured over all 32 queries.
const N_PASSAGES = 8;
const INTRINSIC_RANK = 8;
const RANK_RECALL = [
  { d: 1, r1: 0.25, r3: 0.5312, recon_err: 0.723 },
  { d: 2, r1: 0.4375, r3: 1.0, recon_err: 0.5065 },
  { d: 3, r1: 0.7812, r3: 1.0, recon_err: 0.3158 },
  { d: 4, r1: 0.9688, r3: 1.0, recon_err: 0.1765 },
  { d: 5, r1: 0.9688, r3: 1.0, recon_err: 0.1267 },
  { d: 6, r1: 1.0, r3: 1.0, recon_err: 0.0911 },
  { d: 7, r1: 1.0, r3: 1.0, recon_err: 0.0477 },
  { d: 8, r1: 1.0, r3: 1.0, recon_err: 0.0 },
];
const D_RECOVER = 6;

// The 8x8 display heatmap: row j is the first query of company j scored against every company document,
// so the diagonal is the positive (own company) and same-sector companies sit near (the block structure).
const DISPLAY_MATRIX = [
  [0.9513, 0.5622, -0.0748, -0.168, 0.074, 0.1646, 0.212, 0.3519],
  [0.5171, 0.9758, 0.2018, 0.1587, 0.133, -0.003, 0.369, 0.3397],
  [-0.0984, 0.0578, 0.9535, 0.6087, -0.1106, -0.2267, 0.0803, -0.0456],
  [-0.1476, 0.1047, 0.6866, 0.9578, -0.008, -0.152, 0.2684, 0.09],
  [0.0499, 0.2084, 0.1853, 0.1163, 0.9304, 0.5225, 0.0714, 0.0595],
  [0.0703, 0.0136, -0.1713, -0.2623, 0.566, 0.9569, -0.0653, -0.1682],
  [0.218, 0.3756, 0.2148, 0.4384, 0.0335, -0.0674, 0.9604, 0.763],
  [0.3949, 0.3857, 0.0322, 0.142, -0.039, -0.0286, 0.7344, 0.9632],
];
// The thin SVD of DISPLAY_MATRIX, so TS can reconstruct the best rank-d approximation in closed form.
const DISPLAY_U = [
  [-0.3266, 0.4034, -0.1516, 0.4898, -0.494, 0.2345, -0.3697, 0.1792],
  [-0.4502, 0.1716, 0.0276, 0.5127, 0.5397, -0.3801, 0.2328, -0.1194],
  [-0.1846, -0.5067, 0.3078, 0.3046, -0.3744, 0.208, 0.5717, 0.107],
  [-0.2729, -0.4927, 0.3556, 0.0118, -0.085, -0.2699, -0.6255, -0.2912],
  [-0.1162, 0.2642, 0.6653, -0.107, 0.3443, 0.5769, -0.0969, 0.0428],
  [0.0963, 0.47, 0.5056, -0.1578, -0.415, -0.5217, 0.1936, -0.0857],
  [-0.548, -0.0519, -0.0486, -0.4666, 0.0233, -0.1791, 0.0074, 0.6665],
  [-0.5089, 0.1218, -0.2334, -0.3876, -0.1612, 0.2176, 0.2101, -0.6356],
];
const DISPLAY_S = [2.4037, 1.8246, 1.3954, 0.893, 0.4312, 0.3326, 0.2696, 0.1593];
const DISPLAY_VT = [
  [-0.3347, -0.4523, -0.2504, -0.3085, -0.0473, 0.0474, -0.5151, -0.5057],
  [0.3716, 0.2205, -0.469, -0.5036, 0.3387, 0.4623, 0.002, 0.106],
  [-0.1768, 0.0244, 0.4108, 0.3211, 0.6222, 0.4963, -0.0755, -0.2388],
  [0.4794, 0.4987, 0.291, -0.0391, -0.1328, -0.1748, -0.4585, -0.4205],
  [-0.4918, 0.5358, -0.3128, -0.0104, 0.3938, -0.4623, -0.0066, -0.0658],
  [0.255, -0.3774, 0.251, -0.2264, 0.5199, -0.476, -0.2504, 0.3446],
  [-0.3776, 0.1974, 0.547, -0.6716, -0.16, 0.1186, 0.1015, 0.1349],
  [0.1982, -0.1706, 0.0624, -0.2099, 0.1649, -0.2299, 0.6676, -0.5983],
];

// Panel C: the counting law and the in-batch Gram matrix (= DISPLAY_MATRIX; the batch is the 8 companies).
const GRAM_TAU = 0.05;
const INBATCH_LOSS = 0.0045;

const POS_COLOR = '#5fa873';        // the positive — a query's own document
const ACCENT = 'var(--color-accent)';
const HARD_COLOR = '#7C3AED';        // same-sector hard negatives / the highlighted row
const CEIL_COLOR = '#6a8caf';

const clampIdx = (i: number, n: number) => Math.max(0, Math.min(i, n - 1));

// closed-form truncated SVD reconstruction M_d = sum_{k<d} s_k u_k v_k^T from the baked factors.
function reconstruct(d: number): number[][] {
  const n = DISPLAY_S.length;
  const dd = Math.max(1, Math.min(d, n));
  const M: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let k = 0; k < dd; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) M[i][j] += DISPLAY_S[k] * DISPLAY_U[i][k] * DISPLAY_VT[k][j];
    }
  }
  return M;
}
// closed-form relative Frobenius error of the rank-d reconstruction of the display matrix: tail energy.
function displayReconErr(d: number): number {
  const total = DISPLAY_S.reduce((a, s) => a + s * s, 0);
  const tail = DISPLAY_S.slice(Math.max(1, Math.min(d, DISPLAY_S.length))).reduce((a, s) => a + s * s, 0);
  return Math.sqrt(tail / Math.max(total, 1e-12));
}

const MAT_LO = Math.min(...DISPLAY_MATRIX.flat());
const MAT_HI = Math.max(...DISPLAY_MATRIX.flat());
// map a score to a fill opacity over the matrix's range (guarded denominator).
function tone(v: number): number {
  const t = (v - MAT_LO) / Math.max(MAT_HI - MAT_LO, 1e-9);
  return Math.max(0, Math.min(1, t));
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

// ===== Panel A — factorization & MIPS ==============================================================
function ArchSchematic({ cross }: { cross: boolean }) {
  const S = 440, H = 200;
  return (
    <svg viewBox={`0 0 ${S} ${H}`} role="img" aria-label="bi-encoder versus cross-encoder data flow" style={{ width: '100%', height: 'auto', display: 'block' }}>
      {/* documents */}
      {[0, 1, 2].map((i) => (
        <rect key={i} x={14} y={30 + i * 42} width={62} height={28} rx={3} fill="var(--color-muted-bg, transparent)" stroke="var(--color-border)" />
      ))}
      <text x={45} y={H - 8} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">documents</text>
      <text x={45} y={22} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">corpus C</text>
      {/* query */}
      <rect x={14} y={150} width={62} height={28} rx={3} fill="none" stroke="var(--color-text)" />
      <text x={45} y={168} textAnchor="middle" fontSize={11} fill="var(--color-text)" fontFamily="var(--font-sans)">query q</text>

      {!cross ? (
        <g>
          {/* bi-encoder: E_P precomputes the document matrix once, offline */}
          <rect x={150} y={34} width={120} height={120} rx={6} fill={POS_COLOR} opacity={0.12} stroke={POS_COLOR} strokeDasharray="4 3" />
          <text x={210} y={28} textAnchor="middle" fontSize={10} fill={POS_COLOR} fontFamily="var(--font-sans)">passage matrix G</text>
          <text x={210} y={96} textAnchor="middle" fontSize={10} fill={POS_COLOR} fontFamily="var(--font-sans)">(encoded offline,</text>
          <text x={210} y={110} textAnchor="middle" fontSize={10} fill={POS_COLOR} fontFamily="var(--font-sans)">once)</text>
          {[0, 1, 2].map((i) => (<line key={i} x1={76} y1={44 + i * 42} x2={150} y2={70 + i * 20} stroke="var(--color-border)" strokeWidth={1} />))}
          {/* query encode -> single MIPS */}
          <line x1={76} y1={164} x2={150} y2={150} stroke="var(--color-text)" strokeWidth={1.4} />
          <rect x={310} y={70} width={110} height={48} rx={6} fill="none" stroke={ACCENT} />
          <text x={365} y={90} textAnchor="middle" fontSize={11} fill={ACCENT} fontFamily="var(--font-sans)">MIPS</text>
          <text x={365} y={106} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">G · E_Q(q)</text>
          <line x1={270} y1={94} x2={310} y2={94} stroke={ACCENT} strokeWidth={2} markerEnd="url(#ah)" />
          <line x1={76} y1={150} x2={310} y2={108} stroke="var(--color-text)" strokeWidth={1} strokeDasharray="3 2" opacity={0.5} />
        </g>
      ) : (
        <g>
          {/* cross-encoder: every (q, d) must be fused and re-encoded at query time */}
          {[0, 1, 2].map((i) => (
            <g key={i}>
              <line x1={76} y1={44 + i * 42} x2={170} y2={56 + i * 38} stroke="var(--color-border)" strokeWidth={1} />
              <line x1={76} y1={164} x2={170} y2={64 + i * 38} stroke="var(--color-text)" strokeWidth={1} opacity={0.5} />
              <rect x={170} y={44 + i * 38} width={150} height={28} rx={4} fill={HARD_COLOR} opacity={0.12} stroke={HARD_COLOR} />
              <text x={245} y={62 + i * 38} textAnchor="middle" fontSize={9.5} fill={HARD_COLOR} fontFamily="var(--font-sans)">Enc([q ; dᵢ]) → sᵢ</text>
            </g>
          ))}
          <text x={245} y={H - 8} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">|C| joint passes, per query</text>
        </g>
      )}
      <defs>
        <marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={ACCENT} />
        </marker>
      </defs>
    </svg>
  );
}

function CostCurve({ cross }: { cross: boolean }) {
  const pw = 440, ph = 230, pad = 44;
  const fx = (corpus: number) => pad + (pw - 2 * pad) * (Math.log10(corpus) - 1) / (5 - 1);
  const fy = (passes: number) => ph - pad - (ph - 2 * pad) * (Math.log10(Math.max(passes, 1))) / 5;
  const biPath = COST_VS_CORPUS.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.corpus).toFixed(1) + ' ' + fy(d.bi_passes).toFixed(1)).join(' ');
  const crossPath = COST_VS_CORPUS.map((d, i) => (i === 0 ? 'M' : 'L') + fx(d.corpus).toFixed(1) + ' ' + fy(d.cross_passes).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="query-time forward passes versus corpus size" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[1, 2, 3, 4, 5].map((e) => (<text key={e} x={fx(Math.pow(10, e))} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">10{e === 1 ? '' : <tspan dy={-4} fontSize={7}>{e}</tspan>}</text>))}
      <text x={pw / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">corpus size |C|</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>query-time encoder passes</text>
      <path d={crossPath} fill="none" stroke={HARD_COLOR} strokeWidth={cross ? 3 : 1.6} opacity={cross ? 1 : 0.5} />
      <path d={biPath} fill="none" stroke={POS_COLOR} strokeWidth={cross ? 1.6 : 3} opacity={cross ? 0.5 : 1} />
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={pad + 12} y1={pad + 4} x2={pad + 28} y2={pad + 4} stroke={POS_COLOR} strokeWidth={2.6} />
        <text x={pad + 32} y={pad + 7} fill="var(--color-text-secondary)">bi-encoder (constant: 1)</text>
        <line x1={pad + 12} y1={pad + 20} x2={pad + 28} y2={pad + 20} stroke={HARD_COLOR} strokeWidth={2.6} />
        <text x={pad + 32} y={pad + 23} fill="var(--color-text-secondary)">cross-encoder (linear: |C|)</text>
      </g>
    </svg>
  );
}

// ===== Panel B / C — heatmap ========================================================================
function Heatmap({ M, label, diag, highlightRow }: { M: number[][]; label: string; diag?: boolean; highlightRow?: number }) {
  const n = M.length, cell = 30, pad = 4, S = n * cell + 2 * pad;
  return (
    <svg viewBox={`0 0 ${S} ${S + 16}`} role="img" aria-label={label} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {M.map((row, i) => (
        <g key={i}>
          {row.map((v, j) => {
            const isDiag = diag && i === j;
            return (
              <rect key={`${i}-${j}`} x={pad + j * cell} y={pad + i * cell} width={cell - 1.5} height={cell - 1.5} rx={2}
                fill={isDiag ? POS_COLOR : ACCENT} opacity={isDiag ? 0.55 + 0.4 * tone(v) : tone(v)}
                stroke={isDiag ? POS_COLOR : (highlightRow === i ? HARD_COLOR : 'none')} strokeWidth={isDiag ? 1.4 : (highlightRow === i ? 1.6 : 0)} />
            );
          })}
        </g>
      ))}
      <text x={S / 2} y={S + 12} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{label}</text>
    </svg>
  );
}

function RecallCurve({ dIdx }: { dIdx: number }) {
  const pw = 420, ph = 210, pad = 42;
  const n = RANK_RECALL.length;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (r: number) => ph - pad - (ph - 2 * pad) * r;
  const path = RANK_RECALL.map((d, i) => (i === 0 ? 'M' : 'L') + fx(i).toFixed(1) + ' ' + fy(d.r1).toFixed(1)).join(' ');
  const rankX = fx(INTRINSIC_RANK - 1);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="recall at 1 versus embedding dimension" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {RANK_RECALL.map((d, i) => (<text key={i} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{d.d}</text>))}
      <text x={pw / 2} y={ph - 3} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">embedding dimension d</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>recall@1 (32 queries)</text>
      {/* intrinsic rank marker */}
      <line x1={rankX} y1={pad} x2={rankX} y2={ph - pad} stroke={CEIL_COLOR} strokeWidth={1.2} strokeDasharray="4 3" />
      <text x={rankX - 4} y={pad + 10} textAnchor="end" fontSize={9} fill={CEIL_COLOR} fontFamily="var(--font-sans)">rank = {INTRINSIC_RANK}</text>
      <path d={path} fill="none" stroke={ACCENT} strokeWidth={2.8} />
      {RANK_RECALL.map((d, i) => (<circle key={i} cx={fx(i)} cy={fy(d.r1)} r={i === dIdx ? 5.5 : 3} fill={ACCENT} opacity={i === dIdx ? 1 : 0.7} />))}
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'mips' | 'rank' | 'gram';

export default memo(function DenseDualEncoderLaboratory() {
  const [panel, setPanel] = useState<Panel>('mips');
  const [cross, setCross] = useState(false);
  const [dIdx, setDIdx] = useState(2);          // index into RANK_RECALL (d = 3)
  const [bIdx, setBIdx] = useState(N_PASSAGES); // batch size B (2..8); default = full company set
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'mips'
        ? 's(q, d) = \\langle E_Q(q),\\, E_P(d)\\rangle \\;\\Rightarrow\\; \\text{top-}k_d\\, s(q,d) = \\text{top-}k_j\\,(P\\,E_Q(q))_j'
        : panel === 'rank'
          ? 'S = Q G^{\\top},\\quad \\operatorname{rank}(S) \\le d \\qquad (\\text{best rank-}d:\\ \\text{truncated SVD})'
          : 'S = Q G^{\\top} \\in \\mathbb{R}^{B\\times B}:\\quad 2B \\text{ encodings} \\;\\Rightarrow\\; B(B-1)\\ \\text{negatives}';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const d = RANK_RECALL[clampIdx(dIdx, RANK_RECALL.length)].d;
  const recon = useMemo(() => reconstruct(d), [d]);
  const reconErr = displayReconErr(d);
  const recall = RANK_RECALL[clampIdx(dIdx, RANK_RECALL.length)];

  const B = Math.max(2, Math.min(bIdx, N_PASSAGES));
  const gramSub = useMemo(() => DISPLAY_MATRIX.slice(0, B).map((row) => row.slice(0, B)), [B]);
  const negatives = B * (B - 1);
  const passes = 2 * B;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'mips')} onClick={() => setPanel('mips')}>A · factorization & MIPS</button>
        <button style={pill(panel === 'rank')} onClick={() => setPanel('rank')}>B · the rank-d ceiling</button>
        <button style={pill(panel === 'gram')} onClick={() => setPanel('gram')}>C · in-batch Gram trick</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'mips' && (
        <div>
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <button style={pill(!cross)} onClick={() => setCross(false)}>bi-encoder (dual)</button>
            <button style={pill(cross)} onClick={() => setCross(true)}>cross-encoder</button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem', alignItems: 'center' }}>
            <ArchSchematic cross={cross} />
            <CostCurve cross={cross} />
          </div>
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            <Readout label="query-time encoder passes" value={cross ? '|C| (one per document)' : '1 (the query) + MIPS'} accent={!cross} />
            <Readout label={`at |C| = ${CORPUS_HEADLINE.toLocaleString()}`} value={cross ? `${CORPUS_HEADLINE.toLocaleString()} passes` : '1 pass + index lookup'} />
            <Readout label="passages precomputable?" value={cross ? 'no — must fuse with q' : 'yes — encode once offline'} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            A dual encoder's score <em>separates</em>: query and document meet only in a single inner product. So every
            document vector is encoded <strong>once, offline</strong> into the matrix G, and a query is answered by one
            matrix-vector product G·E<sub>Q</sub>(q) followed by top-k — maximum-inner-product search whose query-time cost is
            constant in the corpus size. A cross-encoder fuses [q ; d] and must run a joint forward pass for <strong>every</strong>
            document at query time, so it cannot precompute anything. That separability is exactly what lets the rest of the
            curriculum index dense vectors (IVF, HNSW, PQ) — and why exact MIPS at scale is its own hard problem.
          </p>
        </div>
      )}

      {panel === 'rank' && (
        <div>
          <Slider label="embedding dimension d" value={dIdx} min={0} max={RANK_RECALL.length - 1} step={1} onChange={setDIdx} fmt={(v) => `${RANK_RECALL[clampIdx(v, RANK_RECALL.length)].d}`} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'start' }}>
            <Heatmap M={DISPLAY_MATRIX} label="relevance matrix S (full rank 8)" diag />
            <Heatmap M={recon} label={`best rank-${d} dual encoder`} diag />
          </div>
          <div style={{ marginTop: '0.6rem' }}><RecallCurve dIdx={clampIdx(dIdx, RANK_RECALL.length)} /></div>
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="recall@1 at d (32 queries)" value={recall.r1.toFixed(4)} accent />
            <Readout label="recall@3 at d" value={recall.r3.toFixed(4)} />
            <Readout label="reconstruction error" value={reconErr.toFixed(4)} />
            <Readout label="recovers full recall@1 at" value={`d = ${D_RECOVER}`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The relevance matrix has four bright sector blocks, each split into two companies. A d-dimensional dual encoder can
            only realize a matrix of rank ≤ d, and the <strong>best</strong> such approximation is the truncated SVD
            (Eckart–Young). Drag d below the pattern's intrinsic rank ({INTRINSIC_RANK}) and the blocks blur together — recall@1
            collapses toward {RANK_RECALL[0].r1.toFixed(2)}; at d ≥ {D_RECOVER} the companies separate and recall saturates at 1.
            The four largest singular values are the sectors; the next four, smaller, are the within-sector company splits — which
            is why you need most of the rank to tell companies apart. (Rank ≤ d is what the architecture <em>can</em> express; the
            tight dimension relevance <em>needs</em> — its sign-rank — is a separate question.)
          </p>
        </div>
      )}

      {panel === 'gram' && (
        <div>
          <Slider label="batch size B" value={bIdx} min={2} max={N_PASSAGES} step={1} onChange={setBIdx} fmt={(v) => `${Math.max(2, Math.min(v, N_PASSAGES))}`} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'center' }}>
            <Heatmap M={gramSub} label={`B × B Gram matrix (B = ${B})`} diag highlightRow={0} />
            <div>
              <div style={{ display: 'flex', gap: '1.4rem', flexWrap: 'wrap' }}>
                <Readout label="encoder forward passes" value={`2B = ${passes}`} />
                <Readout label="in-batch negative pairs" value={`B(B−1) = ${negatives}`} accent />
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', lineHeight: 1.5 }}>
                The diagonal cells (green) are the positives — each query against its own document. Every off-diagonal cell is an
                in-batch negative, free of any extra encoding. The boxed top row is a single (N+1)-way InfoNCE problem: the loss
                the prerequisite topic defined, here read off the Gram matrix the architecture already produces.
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            <Readout label="negatives per encoder pass" value={(negatives / passes).toFixed(2)} />
            <Readout label={`finance batch loss (τ = ${GRAM_TAU})`} value={INBATCH_LOSS.toFixed(4)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            A batch of B (query, document) pairs costs only 2B encoder forward passes, yet the B × B Gram matrix hands you
            B(B−1) = Θ(B²) negative comparisons — quadratic utility from linear cost. That is what makes in-batch-negative
            training the default, and the cost is exact arithmetic. (These negatives are <em>shared</em> across the batch, so they
            are correlated, not the independent draws the InfoNCE bound assumes — and some are false negatives.)
          </p>
        </div>
      )}
    </div>
  );
});
