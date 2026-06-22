import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Late-Interaction Laboratory — three panels for the `late-interaction-learned-sparse` topic:
 *   A. MaxSim. A query-token x document-token similarity grid with each query token's best match
 *      highlighted; the toggle contrasts the multi-vector MaxSim score with a single pooled vector,
 *      which cannot point at several document tokens at once.
 *   B. The escape. On the all-pairs qrel from the embedding-dimension topic, the single-vector model's
 *      row-order accuracy collapses past its critical n while a two-vector MaxSim model (same
 *      optimizer) stays at 1.0 — the wall lifted (flagged empirical).
 *   C. SPLADE. The learned-sparse expansion of a vocabulary-mismatch query into terms it never
 *      mentioned, and the sparsity-versus-quality trade-off the FLOPS regularizer controls.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): SIM_GRID, MAXSIM, POOLED, ESCAPE_CURVE, CRIT_SINGLE,
 * CRIT_MAXSIM, STORAGE, QUERY_WEIGHTS, and TRADE are mirrored TO THE DECIMAL from
 * notebooks/late-interaction-learned-sparse/late_interaction_learned_sparse.py (viz_constants()).
 * test_maxsim_reduces_to_dot_at_m1 / test_multivector_escapes_wall / test_maxsim_critical_n_exceeds_single
 * / test_storage_cost / test_splade_fixes_vocab_mismatch / test_flops_controls_sparsity assert these.
 * The lab recomputes only CLOSED FORM in TS (the per-row max of the baked grid, active-term counts).
 * Change a number here -> change it there, and re-run the notebook. Sliders only (no d3 drag).
 */

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: a designed MaxSim grid — 3 query tokens, 4 document tokens (unit vectors), each query
// token's best match landing on a distinct document token. Entries are cosines.
const SIM_GRID = [
  [0.978, 0.208, -0.978, -0.208],
  [0.105, 0.995, -0.105, -0.995],
  [-0.94, -0.342, 0.94, 0.342],
];
const MAXSIM = 2.913;   // sum of per-query-token maxima
const POOLED = 0.0;     // a single pooled document vector (the token average) cannot match all three

// Panel B: the single-vector wall vs the MaxSim escape on the all-pairs qrel at per-vector dim 4.
const ESCAPE_D = 4;
const N_DOC_VECS = 2;
const ESCAPE_CURVE = [
  { n: 4, single: 1.0, maxsim: 1.0 },
  { n: 6, single: 1.0, maxsim: 1.0 },
  { n: 8, single: 1.0, maxsim: 1.0 },
  { n: 10, single: 0.9333, maxsim: 1.0 },
  { n: 12, single: 0.8636, maxsim: 1.0 },
];
const CRIT_SINGLE = 8;
const CRIT_MAXSIM = 12;
const STORAGE = { single_floats: 128000, multi_floats: 4096000, multiplier: 32 };

// Panel C: SPLADE — the mismatch query's learned expansion weights, and the sparsity trade-off.
const SPLADE_QUERY = 'borrowing costs';
const QUERY_WEIGHTS: [string, number][] = [
  ['interest', 1.386],
  ['rate', 1.253],
  ['exposure', 0.916],
  ['margin', 0.693],
];
const ONPOINT_BM25 = 0.0;   // BM25 score of the on-point filing for the mismatch query (no overlap)
const TRADE = [
  { tau: 0.0, l0: 100, flops: 16.1417, fixed: true },
  { tau: 0.7, l0: 95, flops: 15.2864, fixed: true },
  { tau: 0.95, l0: 83, flops: 13.1372, fixed: true },
  { tau: 1.2, l0: 79, flops: 12.0161, fixed: true },
  { tau: 1.45, l0: 79, flops: 12.0161, fixed: false },
];

const POS_COLOR = '#5fa873';
const ACCENT = 'var(--color-accent)';
const HARD_COLOR = '#7C3AED';
const CEIL_COLOR = '#6a8caf';
const MUTED = '#9aa3ad';

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

// ===== Panel A — the MaxSim grid ====================================================================
function MaxSimGrid({ pooled }: { pooled: boolean }) {
  const nq = SIM_GRID.length, nd = SIM_GRID[0].length;
  const cell = 46, padL = 70, padT = 26, W = padL + nd * cell + 10, H = padT + nq * cell + 16;
  const argmax = SIM_GRID.map((row) => row.indexOf(Math.max(...row)));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="MaxSim query-token by document-token grid" style={{ width: '100%', height: 'auto', display: 'block' }}>
      {SIM_GRID[0].map((_, j) => (
        <text key={`dh-${j}`} x={padL + j * cell + cell / 2} y={padT - 8} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">d{j + 1}</text>
      ))}
      {SIM_GRID.map((row, i) => (
        <g key={`row-${i}`}>
          <text x={padL - 8} y={padT + i * cell + cell / 2 + 3} textAnchor="end" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">q{i + 1}</text>
          {row.map((v, j) => {
            const isMax = !pooled && j === argmax[i];
            return (
              <g key={`c-${i}-${j}`}>
                <rect x={padL + j * cell} y={padT + i * cell} width={cell - 3} height={cell - 3} rx={3}
                  fill={v >= 0 ? ACCENT : MUTED} opacity={Math.min(1, Math.abs(v) * 0.85 + 0.08)}
                  stroke={isMax ? POS_COLOR : 'none'} strokeWidth={isMax ? 2.4 : 0} />
                <text x={padL + j * cell + (cell - 3) / 2} y={padT + i * cell + (cell - 3) / 2 + 3} textAnchor="middle" fontSize={9}
                  fill={Math.abs(v) > 0.55 ? 'var(--color-bg)' : 'var(--color-text)'} fontFamily="var(--font-sans)">{v.toFixed(2)}</text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

// ===== Panel B — the escape curve ===================================================================
function EscapeCurve({ nIdx }: { nIdx: number }) {
  const pw = 440, ph = 230, pad = 46;
  const n = ESCAPE_CURVE.length;
  const fx = (i: number) => pad + (pw - 2 * pad) * i / (n - 1);
  const fy = (a: number) => ph - pad - (ph - 2 * pad) * a;
  const single = ESCAPE_CURVE.map((d, i) => (i === 0 ? 'M' : 'L') + f1(fx(i)) + ' ' + f1(fy(d.single))).join(' ');
  const multi = ESCAPE_CURVE.map((d, i) => (i === 0 ? 'M' : 'L') + f1(fx(i)) + ' ' + f1(fy(d.maxsim))).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="single-vector versus MaxSim row-order accuracy by corpus size" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((v) => (<text key={v} x={pad - 6} y={fy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>))}
      {ESCAPE_CURVE.map((d, i) => (<text key={i} x={fx(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{d.n}</text>))}
      <text x={pw / 2} y={ph - 3} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">corpus size n (documents)</text>
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>row-order accuracy</text>
      <path d={single} fill="none" stroke={CEIL_COLOR} strokeWidth={2.6} />
      <path d={multi} fill="none" stroke={POS_COLOR} strokeWidth={3} />
      {ESCAPE_CURVE.map((d, i) => (
        <g key={`pt-${i}`}>
          <circle cx={fx(i)} cy={fy(d.single)} r={i === nIdx ? 5 : 3} fill={CEIL_COLOR} />
          <circle cx={fx(i)} cy={fy(d.maxsim)} r={i === nIdx ? 5 : 3} fill={POS_COLOR} />
        </g>
      ))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <line x1={pad + 14} y1={pad + 6} x2={pad + 30} y2={pad + 6} stroke={POS_COLOR} strokeWidth={2.6} />
        <text x={pad + 34} y={pad + 9} fill="var(--color-text-secondary)">MaxSim (m = {N_DOC_VECS})</text>
        <line x1={pad + 14} y1={pad + 22} x2={pad + 30} y2={pad + 22} stroke={CEIL_COLOR} strokeWidth={2.6} />
        <text x={pad + 34} y={pad + 25} fill="var(--color-text-secondary)">single vector (m = 1)</text>
      </g>
    </svg>
  );
}

// ===== Panel C — SPLADE expansion weights ===========================================================
function SpladeBars({ tau }: { tau: number }) {
  const pw = 440, ph = 200, pad = 44;
  const n = QUERY_WEIGHTS.length;
  const maxW = Math.max(...QUERY_WEIGHTS.map(([, w]) => w));
  const bw = (pw - 2 * pad) / n * 0.55;
  const fx = (i: number) => pad + (pw - 2 * pad) * (i + 0.5) / n;
  const fy = (w: number) => ph - pad - (ph - 2 * pad) * (w / maxW);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="SPLADE learned expansion weights for the mismatch query" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {/* the pruning threshold line */}
      {tau <= maxW && (
        <g>
          <line x1={pad} y1={fy(tau)} x2={pw - pad} y2={fy(tau)} stroke={HARD_COLOR} strokeWidth={1.2} strokeDasharray="4 3" />
          <text x={pw - pad} y={fy(tau) - 4} textAnchor="end" fontSize={9} fill={HARD_COLOR} fontFamily="var(--font-sans)">prune ≤ τ = {tau.toFixed(2)}</text>
        </g>
      )}
      {QUERY_WEIGHTS.map(([term, w], i) => {
        const pruned = w <= tau;
        return (
          <g key={term}>
            <rect x={fx(i) - bw / 2} y={fy(w)} width={bw} height={(ph - pad) - fy(w)} rx={2}
              fill={pruned ? MUTED : ACCENT} opacity={pruned ? 0.4 : 1} />
            <text x={fx(i)} y={ph - pad + 13} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{term}</text>
            <text x={fx(i)} y={fy(w) - 4} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{w.toFixed(2)}</text>
          </g>
        );
      })}
      <text x={13} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${ph / 2})`}>learned weight</text>
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'maxsim' | 'escape' | 'splade';

export default memo(function LateInteractionLaboratory() {
  const [panel, setPanel] = useState<Panel>('maxsim');
  const [pooled, setPooled] = useState(false);
  const [nIdx, setNIdx] = useState(4);     // index into ESCAPE_CURVE (n = 12)
  const [tIdx, setTIdx] = useState(0);     // index into TRADE
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'maxsim'
        ? 'S(q, d) = \\sum_{i} \\max_{j} \\langle q_i, d_j\\rangle \\qquad (m=1:\\ \\langle q_1, d_1\\rangle,\\ \\text{the dot product})'
        : panel === 'escape'
          ? '\\text{all-pairs qrel at } d = ' + ESCAPE_D + ':\\quad \\text{single vector collapses, MaxSim holds}'
          : 'w_j = \\max_{i} \\log\\!\\big(1 + \\mathrm{ReLU}(\\ell_{ij})\\big), \\qquad \\mathcal{L}_{\\mathrm{FLOPS}} = \\sum_j \\bar{a}_j^{2}';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const nRow = ESCAPE_CURVE[clampIdx(nIdx, ESCAPE_CURVE.length)];
  const trade = TRADE[clampIdx(tIdx, TRADE.length)];

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'maxsim')} onClick={() => setPanel('maxsim')}>A · MaxSim</button>
        <button style={pill(panel === 'escape')} onClick={() => setPanel('escape')}>B · the escape</button>
        <button style={pill(panel === 'splade')} onClick={() => setPanel('splade')}>C · SPLADE</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'maxsim' && (
        <div>
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.6rem' }}>
            <button style={pill(!pooled)} onClick={() => setPooled(false)}>MaxSim (one vector / token)</button>
            <button style={pill(pooled)} onClick={() => setPooled(true)}>single pooled vector</button>
          </div>
          <MaxSimGrid pooled={pooled} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
            <Readout label="MaxSim score (sum of per-token maxima)" value={MAXSIM.toFixed(3)} accent={!pooled} />
            <Readout label="single pooled vector score" value={POOLED.toFixed(3)} accent={pooled} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Each query token (row) matches its <strong>best</strong> document token (the green-bordered cell), and MaxSim sums those
            maxima. Three query tokens land on three <em>distinct</em> document tokens — a many-to-many match. A single pooled
            document vector (the token average) cannot point at three things at once, so it scores {POOLED.toFixed(2)}. With one
            vector per item the max is over a singleton and MaxSim is exactly the dual-encoder dot product — the provable anchor.
          </p>
        </div>
      )}

      {panel === 'escape' && (
        <div>
          <Slider label="corpus size n" value={nIdx} min={0} max={ESCAPE_CURVE.length - 1} step={1} onChange={setNIdx} fmt={(v) => `${ESCAPE_CURVE[clampIdx(v, ESCAPE_CURVE.length)].n}`} />
          <EscapeCurve nIdx={clampIdx(nIdx, ESCAPE_CURVE.length)} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label={`single vector at n = ${nRow.n}`} value={nRow.single.toFixed(3)} />
            <Readout label={`MaxSim (m=${N_DOC_VECS}) at n = ${nRow.n}`} value={nRow.maxsim.toFixed(3)} accent />
            <Readout label="critical n: single vs MaxSim" value={`${CRIT_SINGLE} → ${CRIT_MAXSIM}`} />
            <Readout label="storage cost of late interaction" value={`${STORAGE.multiplier}× the floats`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            On the all-pairs relevance pattern at per-vector dimension {ESCAPE_D}, a single vector realizes the order perfectly up to
            {' '}{CRIT_SINGLE} documents, then collapses; a MaxSim model with {N_DOC_VECS} vectors per document — the <em>same</em>{' '}
            free-embedding optimizer — stays at 1.0 through {CRIT_MAXSIM}. The gap is the multi-vector effect, not a difference between
            optimizers (m = 1 <em>is</em> the single vector). This is a best-case demonstration, not a theorem: no multi-vector
            sign-rank lower bound is known. The escape costs storage — one vector per token, {STORAGE.multiplier}× a single-vector index.
          </p>
        </div>
      )}

      {panel === 'splade' && (
        <div>
          <Slider label="sparsity threshold τ" value={tIdx} min={0} max={TRADE.length - 1} step={1} onChange={setTIdx} fmt={(v) => `${TRADE[clampIdx(v, TRADE.length)].tau.toFixed(2)}`} />
          <SpladeBars tau={trade.tau} />
          <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="active terms ‖w‖₀ (corpus)" value={`${trade.l0}`} />
            <Readout label="FLOPS value" value={trade.flops.toFixed(2)} />
            <Readout label="mismatch fixed?" value={trade.fixed ? 'yes — filing-onpoint #1' : 'no — expansion pruned'} accent={trade.fixed} />
            <Readout label="BM25 score (no lexical overlap)" value={ONPOINT_BM25.toFixed(1)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The query <strong>"{SPLADE_QUERY}"</strong> shares no term with the on-point filing, so BM25 scores it {ONPOINT_BM25.toFixed(1)}.
            SPLADE expands it into the filing's vocabulary — <em>interest, rate, exposure, margin</em>, every weight learned, none a
            literal query term — and retrieves it. Raising τ prunes low-weight terms, lowering the active-term count
            ({TRADE[0].l0} → {TRADE[TRADE.length - 1].l0}) and the FLOPS cost — but once the bridging expansion is pruned, the
            mismatch fix breaks. Expand enough to bridge, stay sparse enough to keep the inverted index fast.
          </p>
        </div>
      )}
    </div>
  );
});
