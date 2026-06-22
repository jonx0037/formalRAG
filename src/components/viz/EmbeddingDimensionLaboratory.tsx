import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Embedding-Dimension Laboratory — three panels for the `embedding-dimension-lower-bounds` topic:
 *   A. Rank vs sign-rank. The signed identity (+1 on the diagonal, -1 off) has full rank n but
 *      sign-rank 3: a relevance pattern that looks expensive is cheap. The achievable margin of a
 *      correct-sign realization is zero below the sign-rank and grows once the dimension clears it.
 *   B. The Forster wall. A Hadamard relevance pattern has ||H||_2 = sqrt(N), so Forster's spectral
 *      bound forces sign-rank >= sqrt(N) — a closed-form lower bound rising without bound in N.
 *   C. The free-embedding wall. Even perfect, freely optimized embeddings realize the all-pairs qrel
 *      only up to a critical corpus size that grows slowly with d; the finance flip shows the d that
 *      solves single-company retrieval failing combinatorial multi-company queries.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): SIGN_MATRIX, RANK, SIGN_RANK, MARGIN_BY_DIM, FORSTER,
 * HADAMARD_4, CRITICAL_N, and FINANCE_FLIP are mirrored TO THE DECIMAL from
 * notebooks/embedding-dimension-lower-bounds/embedding_dimension_lower_bounds.py (viz_constants()).
 * test_rank_vs_signrank_gap / test_forster_on_hadamard / test_margin_grows_with_dimension /
 * test_free_embedding_wall / test_finance_headline_flip assert these. The lab recomputes only CLOSED
 * FORM in TS — the Forster bound sqrt(N) and the all-pairs query count C(n,2). Change a number here ->
 * change it there, and re-run the notebook. Sliders only (no d3 drag); SVG text inherits the theme.
 */

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: the signed identity (+1 diagonal, -1 off), its rank, sign-rank, and the achievable margin
// of a correct-sign rank-d realization (0 below the sign-rank, positive and non-shrinking above).
const SIGN_MATRIX = [
  [1, -1, -1, -1],
  [-1, 1, -1, -1],
  [-1, -1, 1, -1],
  [-1, -1, -1, 1],
];
const RANK = 4;
const SIGN_RANK = 3;
const MARGIN_BY_DIM = [
  { d: 1, realized: false, margin: 0.0 },
  { d: 2, realized: false, margin: 0.0 },
  { d: 3, realized: true, margin: 0.3026 },
  { d: 4, realized: true, margin: 0.3498 },
  { d: 5, realized: true, margin: 0.3665 },
  { d: 6, realized: true, margin: 0.3713 },
];

// Panel B: the Sylvester-Hadamard family. ||H||_2 = sqrt(N) exactly (orthogonal rows), so the Forster
// bound sqrt(mn)/||H|| = sqrt(N) is the closed-form lower bound on the sign-rank.
const FORSTER = [
  { k: 1, N: 2, spectral: 1.4142, forster_bound: 1.4142 },
  { k: 2, N: 4, spectral: 2.0, forster_bound: 2.0 },
  { k: 3, N: 8, spectral: 2.8284, forster_bound: 2.8284 },
  { k: 4, N: 16, spectral: 4.0, forster_bound: 4.0 },
];
const HADAMARD_4 = [
  [1, 1, 1, 1],
  [1, -1, 1, -1],
  [1, 1, -1, -1],
  [1, -1, -1, 1],
];

// Panel C: the free-embedding critical n — the largest all-pairs corpus realizable in d dimensions,
// even with perfect embeddings — and the finance headline flip.
const CRITICAL_N = [
  { d: 2, critical_n: 4 },
  { d: 3, critical_n: 4 },
  { d: 4, critical_n: 6 },
  { d: 5, critical_n: 12 },
  { d: 6, critical_n: 16 },
];
const FINANCE_FLIP = { d_recover: 6, n_docs_combo: 24, acc_combo_at_recover: 0.9384, d_combo: 8 };

const POS_COLOR = '#5fa873';        // +1 (relevant)
const NEG_COLOR = '#c2604f';        // -1 (not relevant)
const ACCENT = 'var(--color-accent)';
const CEIL_COLOR = '#6a8caf';
const MUTED = '#9aa3ad';

// round derived coordinates so SSR (Node) and client (browser) serialize identical strings.
const f1 = (v: number) => v.toFixed(1);
const clampIdx = (i: number, n: number) => Math.max(0, Math.min(i, n - 1));

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

// ===== a +/-1 sign heatmap (green = +1 relevant, red = -1 not) ======================================
function SignHeatmap({ M, label, diag }: { M: number[][]; label: string; diag?: boolean }) {
  const n = M.length, cell = 34, pad = 4, S = n * cell + 2 * pad;
  return (
    <svg viewBox={`0 0 ${S} ${S + 16}`} role="img" aria-label={label} style={{ width: '100%', maxWidth: '230px', height: 'auto', display: 'block' }}>
      {M.map((row, i) => (
        <g key={i}>
          {row.map((v, j) => {
            const isDiag = diag && i === j;
            return (
              <rect key={`${i}-${j}`} x={pad + j * cell} y={pad + i * cell} width={cell - 2} height={cell - 2} rx={3}
                fill={v > 0 ? POS_COLOR : NEG_COLOR} opacity={v > 0 ? 0.88 : 0.5}
                stroke={isDiag ? 'var(--color-text)' : 'none'} strokeWidth={isDiag ? 1.4 : 0} />
            );
          })}
        </g>
      ))}
      <text x={S / 2} y={S + 12} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{label}</text>
    </svg>
  );
}

// ===== Panel A — the margin-by-dimension bar chart ==================================================
function MarginBars({ dIdx }: { dIdx: number }) {
  const pw = 420, ph = 220, pad = 44;
  const n = MARGIN_BY_DIM.length;
  const maxM = Math.max(...MARGIN_BY_DIM.map((r) => r.margin), 0.4);
  const bw = (pw - 2 * pad) / n * 0.6;
  const fx = (i: number) => pad + (pw - 2 * pad) * (i + 0.5) / n;
  const fy = (m: number) => ph - pad - (ph - 2 * pad) * (m / maxM);
  const srX = fx(SIGN_RANK - 1);
  const rkX = fx(RANK - 1);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="achievable margin versus embedding dimension" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, maxM / 2, maxM].map((v) => (
        <text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(2)}</text>
      ))}
      {/* sign-rank and rank markers */}
      <line x1={srX} y1={pad} x2={srX} y2={ph - pad} stroke={POS_COLOR} strokeWidth={1.2} strokeDasharray="4 3" />
      <text x={srX} y={pad - 3} textAnchor="middle" fontSize={9} fill={POS_COLOR} fontFamily="var(--font-sans)">sign-rank = {SIGN_RANK}</text>
      <line x1={rkX} y1={pad} x2={rkX} y2={ph - pad} stroke={CEIL_COLOR} strokeWidth={1.2} strokeDasharray="4 3" />
      <text x={rkX} y={ph - pad + 22} textAnchor="middle" fontSize={9} fill={CEIL_COLOR} fontFamily="var(--font-sans)">rank = {RANK}</text>
      {MARGIN_BY_DIM.map((r, i) => {
        const h = r.realized ? (ph - pad) - fy(r.margin) : 3;
        return (
          <g key={r.d}>
            <rect x={fx(i) - bw / 2} y={r.realized ? fy(r.margin) : ph - pad - 3} width={bw} height={h} rx={2}
              fill={r.realized ? ACCENT : MUTED} opacity={i === dIdx ? 1 : 0.55} />
            <text x={fx(i)} y={ph - pad + 13} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.d}</text>
            {!r.realized && (
              <text x={fx(i)} y={ph - pad - 8} textAnchor="middle" fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">none</text>
            )}
          </g>
        );
      })}
      <text x={pw / 2} y={ph - 3} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">embedding dimension d</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>achievable margin</text>
    </svg>
  );
}

// ===== Panel B — the Forster bound rising in N =====================================================
function ForsterCurve({ kIdx }: { kIdx: number }) {
  const pw = 420, ph = 220, pad = 46;
  const n = FORSTER.length;
  const maxB = Math.max(...FORSTER.map((r) => r.forster_bound));
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (b: number) => ph - pad - (ph - 2 * pad) * (b / maxB);
  const path = FORSTER.map((r, i) => (i === 0 ? 'M' : 'L') + f1(fx(i)) + ' ' + f1(fy(r.forster_bound))).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="Forster lower bound versus matrix size N" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, maxB / 2, maxB].map((v) => (
        <text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>
      ))}
      {FORSTER.map((r, i) => (
        <text key={r.N} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.N}</text>
      ))}
      <text x={pw / 2} y={ph - 3} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">matrix size N (documents)</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>sign-rank lower bound √N</text>
      <path d={path} fill="none" stroke={ACCENT} strokeWidth={2.8} />
      {FORSTER.map((r, i) => (
        <circle key={r.N} cx={fx(i)} cy={fy(r.forster_bound)} r={i === kIdx ? 5.5 : 3} fill={ACCENT} opacity={i === kIdx ? 1 : 0.7} />
      ))}
    </svg>
  );
}

// ===== Panel C — the free-embedding critical-n wall ================================================
function CriticalCurve({ dIdx }: { dIdx: number }) {
  const pw = 420, ph = 220, pad = 46;
  const n = CRITICAL_N.length;
  const maxC = Math.max(...CRITICAL_N.map((r) => r.critical_n));
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (c: number) => ph - pad - (ph - 2 * pad) * (c / maxC);
  const path = CRITICAL_N.map((r, i) => (i === 0 ? 'M' : 'L') + f1(fx(i)) + ' ' + f1(fy(r.critical_n))).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="free-embedding critical corpus size versus dimension" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, maxC / 2, maxC].map((v) => (
        <text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{Math.round(v)}</text>
      ))}
      {CRITICAL_N.map((r, i) => (
        <text key={r.d} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.d}</text>
      ))}
      <text x={pw / 2} y={ph - 3} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">embedding dimension d</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>largest realizable corpus (critical n)</text>
      <path d={path} fill="none" stroke={ACCENT} strokeWidth={2.8} />
      {CRITICAL_N.map((r, i) => (
        <circle key={r.d} cx={fx(i)} cy={fy(r.critical_n)} r={i === dIdx ? 5.5 : 3} fill={ACCENT} opacity={i === dIdx ? 1 : 0.7} />
      ))}
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'gap' | 'forster' | 'wall';

export default memo(function EmbeddingDimensionLaboratory() {
  const [panel, setPanel] = useState<Panel>('gap');
  const [dIdx, setDIdx] = useState(2);     // index into MARGIN_BY_DIM (d = 3)
  const [kIdx, setKIdx] = useState(3);     // index into FORSTER (N = 16)
  const [cIdx, setCIdx] = useState(4);     // index into CRITICAL_N (d = 6)
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'gap'
        ? '\\operatorname{rank}(M) = n \\quad\\text{but}\\quad \\operatorname{rank}_\\pm(M) \\le 3'
        : panel === 'forster'
          ? '\\lVert H\\rVert_2 = \\sqrt{N} \\;\\Rightarrow\\; \\operatorname{rank}_\\pm(H) \\ge \\frac{\\sqrt{N\\cdot N}}{\\sqrt{N}} = \\sqrt{N}'
          : '\\text{critical } n(d):\\ \\text{largest all-pairs corpus realizable in } \\mathbb{R}^d';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const marg = MARGIN_BY_DIM[clampIdx(dIdx, MARGIN_BY_DIM.length)];
  const forster = FORSTER[clampIdx(kIdx, FORSTER.length)];
  const crit = CRITICAL_N[clampIdx(cIdx, CRITICAL_N.length)];
  const critQueries = (crit.critical_n * (crit.critical_n - 1)) / 2;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'gap')} onClick={() => setPanel('gap')}>A · rank vs sign-rank</button>
        <button style={pill(panel === 'forster')} onClick={() => setPanel('forster')}>B · the Forster wall</button>
        <button style={pill(panel === 'wall')} onClick={() => setPanel('wall')}>C · the retrieval wall</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'gap' && (
        <div>
          <Slider label="embedding dimension d" value={dIdx} min={0} max={MARGIN_BY_DIM.length - 1} step={1} onChange={setDIdx} fmt={(v) => `${MARGIN_BY_DIM[clampIdx(v, MARGIN_BY_DIM.length)].d}`} />
          <div style={{ display: 'grid', gridTemplateColumns: '0.7fr 1.3fr', gap: '1rem', alignItems: 'center' }}>
            <SignHeatmap M={SIGN_MATRIX} label="signed identity M (rank 4)" diag />
            <MarginBars dIdx={clampIdx(dIdx, MARGIN_BY_DIM.length)} />
          </div>
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="rank of M" value={`${RANK}`} />
            <Readout label="sign-rank of M" value={`${SIGN_RANK}`} accent />
            <Readout label={`realizable at d = ${marg.d}?`} value={marg.realized ? 'yes' : 'no'} />
            <Readout label="achievable margin at d" value={marg.realized ? marg.margin.toFixed(4) : '0 (unrealizable)'} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The signed identity — each query relevant to its own document and no other — has <strong>full rank {RANK}</strong>,
            yet a rank-{SIGN_RANK} model reproduces every one of its signs: its <strong>sign-rank is {SIGN_RANK}</strong>. Below
            the sign-rank no correct realization exists (the margin is zero); at and above it the achievable margin is positive and
            grows with d — more dimensions buy a more robust separation. Rank counts directions in the scores; sign-rank counts the
            dimensions the relevance <em>pattern</em> needs, and here most of the rank is wasted on magnitudes the ranking never uses.
          </p>
        </div>
      )}

      {panel === 'forster' && (
        <div>
          <Slider label="matrix size N = 2ᵏ" value={kIdx} min={0} max={FORSTER.length - 1} step={1} onChange={setKIdx} fmt={(v) => `${FORSTER[clampIdx(v, FORSTER.length)].N}`} />
          <div style={{ display: 'grid', gridTemplateColumns: '0.7fr 1.3fr', gap: '1rem', alignItems: 'center' }}>
            <SignHeatmap M={HADAMARD_4} label="Hadamard pattern H (N = 4)" />
            <ForsterCurve kIdx={clampIdx(kIdx, FORSTER.length)} />
          </div>
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="documents N" value={`${forster.N}`} />
            <Readout label="spectral norm ‖H‖₂" value={`${forster.spectral.toFixed(4)} = √${forster.N}`} />
            <Readout label="Forster bound: dimensions needed" value={`≥ ${forster.forster_bound.toFixed(4)}`} accent />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            A Hadamard matrix has orthogonal rows, so HHᵀ = N·I and every singular value equals √N — hence ‖H‖₂ = √N exactly.
            Forster's spectral bound, sign-rank ≥ √(N·N)/‖H‖₂, then forces <strong>sign-rank ≥ √N</strong>: a relevance pattern
            shaped like H provably <em>needs</em> at least √N dimensions, a closed-form lower bound with no optimization. As N grows
            the wall rises without bound — the first genuinely negative expressivity result, certified rather than searched for.
          </p>
        </div>
      )}

      {panel === 'wall' && (
        <div>
          <Slider label="embedding dimension d" value={cIdx} min={0} max={CRITICAL_N.length - 1} step={1} onChange={setCIdx} fmt={(v) => `${CRITICAL_N[clampIdx(v, CRITICAL_N.length)].d}`} />
          <CriticalCurve dIdx={clampIdx(cIdx, CRITICAL_N.length)} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label={`largest realizable corpus at d = ${crit.d}`} value={`${crit.critical_n} documents`} accent />
            <Readout label="all-pairs queries it covers" value={`C(${crit.critical_n},2) = ${critQueries}`} />
          </div>
          <div style={{ marginTop: '0.7rem', padding: '0.6rem 0.8rem', border: `1px solid var(--color-border)`, borderRadius: '0.4rem', background: 'var(--color-muted-bg, transparent)' }}>
            <div style={{ fontSize: '0.74rem', color: 'var(--color-text-secondary)', marginBottom: '0.3rem', fontFamily: 'var(--font-sans)' }}>the finance flip (same corpus, two relevance patterns)</div>
            <div style={{ display: 'flex', gap: '1.6rem', flexWrap: 'wrap' }}>
              <Readout label="single-company retrieval solved at" value={`d = ${FINANCE_FLIP.d_recover}`} />
              <Readout label={`combinatorial all-pairs over ${FINANCE_FLIP.n_docs_combo} cos. at d = ${FINANCE_FLIP.d_recover}`} value={`fails (acc ${FINANCE_FLIP.acc_combo_at_recover.toFixed(3)})`} accent />
              <Readout label="combinatorial realized only at" value={`d = ${FINANCE_FLIP.d_combo}`} />
            </div>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', lineHeight: 1.5 }}>
            Even with <strong>perfect, freely optimized</strong> embeddings, the largest corpus whose all-pairs relevance pattern a
            d-dimensional model can realize — the critical n — grows only slowly with d. Beyond it, some combination of relevant
            documents is <em>unrepresentable</em>, not merely poorly trained. The finance flip makes it concrete: the embedding
            dimension that perfectly solves single-company retrieval cannot satisfy combinatorial multi-company queries over a
            comparable corpus. The escape is architectural — multi-vector late interaction or a sparse lexical leg — not a bigger d.
          </p>
        </div>
      )}
    </div>
  );
});
