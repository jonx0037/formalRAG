import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Fusion Laboratory — three ranked columns (lexical BM25, dense cosine, fused)
 * driven by a method toggle and the RRF constant k, validating the rank-fusion
 * topic's claims:
 *   - RRF reads only positions, so dragging the "BM25 score scale" leaves the
 *     fused order untouched under RRF/Borda but collapses CombSUM onto the BM25
 *     order (Proposition 1, scale invariance).
 *   - A live NDCG readout shows the hybrid (RRF) strictly beating either leg.
 *   - A consensus panel reports Kendall-tau from the fused list to each leg and
 *     to the Kemeny-optimal consensus.
 *
 * Fully deterministic. The corpus scores below are mirrored TO THE DECIMAL from
 * notebooks/rank-fusion-rrf/rank_fusion_rrf.py — its finance_demo() prints these
 * same two rankings and the k=60 fused order, and test_hybrid_beats_either_leg()
 * asserts the NDCG values shown here (lexical 0.980, dense 0.980, RRF 1.000).
 * Change a number here -> change it there, and re-run the notebook.
 */

type Doc = { id: string; label: string; kind: string; bm25: number; cos: number; qrel: number };
const QUERY = 'interest rate exposure';

// BM25 and cosine scores: rank_fusion_rrf.py, query "interest rate exposure".
const CORPUS: Doc[] = [
  { id: 'filing-onpoint',  label: '10-K · net interest margin sensitivity', kind: 'filing',     bm25: 1.9812, cos: 0.4755, qrel: 3 },
  { id: 'transcript-rate', label: 'Call · rate-exposure hedging',           kind: 'transcript', bm25: 0.8885, cos: 0.6816, qrel: 3 },
  { id: 'filing-hedging',  label: '10-K · interest-rate hedging policy',     kind: 'filing',     bm25: 1.8993, cos: 0.6601, qrel: 2 },
  { id: 'filing-fx',       label: '10-K · FX translation exposure',          kind: 'filing',     bm25: 0.4502, cos: 0.4413, qrel: 1 },
  { id: 'news-macro',      label: 'News · central-bank rate decision',       kind: 'news',       bm25: 0.4274, cos: 0.3036, qrel: 1 },
  { id: 'transcript-ops',  label: 'Call · operations update',                kind: 'transcript', bm25: 0.4274, cos: 0.2857, qrel: 0 },
];
const N = CORPUS.length;
// Kemeny-optimal consensus (brute force in the notebook): lifts the agreed #2.
const KEMENY = ['filing-hedging', 'filing-onpoint', 'transcript-rate', 'filing-fx', 'news-macro', 'transcript-ops'];
const byId = Object.fromEntries(CORPUS.map((d) => [d.id, d]));

// order(best first), ties broken by id ascending — matches the notebook's order_of.
const orderBy = (score: (d: Doc) => number) =>
  [...CORPUS].sort((a, b) => score(b) - score(a) || (a.id < b.id ? -1 : 1)).map((d) => d.id);
const rankOf = (order: string[]) => Object.fromEntries(order.map((id, i) => [id, i + 1])); // 1-based

const dcg = (gains: number[]) => gains.reduce((s, g, i) => s + g / Math.log2(i + 2), 0);
const idealDcg = dcg([...CORPUS.map((d) => d.qrel)].sort((a, b) => b - a));
const ndcg = (order: string[]) => dcg(order.map((id) => byId[id].qrel)) / idealDcg;

// Kendall-tau distance (discordant pairs) between two full orderings.
function kendallTau(a: string[], b: string[]): number {
  const pa = rankOf(a), pb = rankOf(b);
  let d = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = i + 1; j < a.length; j++) {
      const x = a[i], y = a[j];
      if ((pa[x] - pa[y]) * (pb[x] - pb[y]) < 0) d++;
    }
  return d;
}

const KIND_COLOR: Record<string, string> = {
  filing: 'var(--color-accent)',
  transcript: 'var(--color-accent-secondary)',
  news: 'var(--color-text-secondary)',
};

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem' }}>
      <span style={{ minWidth: '9rem' }}>{label} = <strong>{value.toFixed(0)}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
    </label>
  );
}

// A single ranked column. Defined at module scope (not inside FusionLaboratory)
// so its component identity is stable across renders — otherwise every slider
// drag or hover would unmount and remount all three columns instead of
// reconciling them. hover/setHover are threaded in as props.
function Column({ title, order, sub, hover, setHover }: {
  title: string; order: string[]; sub: string;
  hover: string | null; setHover: (v: string | null) => void;
}) {
  return (
    <div style={{ flex: 1, minWidth: '190px' }}>
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.15rem' }}>{title}</div>
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginBottom: '0.4rem', minHeight: '1.5em' }}>{sub}</div>
      <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
        {order.map((id, i) => {
          const d = byId[id];
          const active = hover === id;
          return (
            <li key={id}
              onMouseEnter={() => setHover(id)} onMouseLeave={() => setHover(null)}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.25rem 0.4rem',
                fontFamily: 'var(--font-sans)', fontSize: '0.72rem', borderRadius: '0.4rem',
                border: `1px solid ${active ? 'var(--color-accent)' : 'transparent'}`,
                background: active ? 'var(--color-muted-bg)' : 'transparent',
                transition: 'background 0.25s, border-color 0.25s',
              }}>
              <span style={{ minWidth: '1rem', color: 'var(--color-text-secondary)' }}>{i + 1}</span>
              <span style={{ width: '0.5rem', height: '0.5rem', borderRadius: '999px', background: KIND_COLOR[d.kind], flexShrink: 0 }} />
              <span style={{ flex: 1, fontWeight: d.qrel >= 2 ? 600 : 400 }}>{d.label}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

type Method = 'rrf' | 'borda' | 'combsum';

export default function FusionLaboratory() {
  const [method, setMethod] = useState<Method>('rrf');
  const [k, setK] = useState(60);
  const [bm25Scale, setBm25Scale] = useState(1);
  const [hover, setHover] = useState<string | null>(null);
  const formulaRef = useRef<HTMLDivElement>(null);

  // The two input legs are fixed orderings; their ORDER never depends on scale.
  const lexOrder = useMemo(() => orderBy((d) => d.bm25), []);
  const denOrder = useMemo(() => orderBy((d) => d.cos), []);
  const lexRank = useMemo(() => rankOf(lexOrder), [lexOrder]);
  const denRank = useMemo(() => rankOf(denOrder), [denOrder]);

  // live KaTeX of the active fusion rule
  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      method === 'rrf'
        ? `\\mathrm{RRF}(d)=\\frac{1}{${k}+r_{\\text{lex}}(d)}+\\frac{1}{${k}+r_{\\text{dense}}(d)}`
        : method === 'borda'
        ? `\\mathrm{Borda}(d)=\\bigl(${N}-r_{\\text{lex}}(d)\\bigr)+\\bigl(${N}-r_{\\text{dense}}(d)\\bigr)`
        : `\\mathrm{CombSUM}(d)=${bm25Scale}\\cdot s_{\\text{bm25}}(d)+s_{\\cos}(d)`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [method, k, bm25Scale]);

  // fused score by the active method (RRF/Borda read positions; CombSUM reads scores)
  const fusedScore = useMemo(() => {
    return (d: Doc): number => {
      if (method === 'rrf') return 1 / (k + lexRank[d.id]) + 1 / (k + denRank[d.id]);
      if (method === 'borda') return (N - lexRank[d.id]) + (N - denRank[d.id]);
      return bm25Scale * d.bm25 + d.cos; // combsum on RAW scores -> scale-sensitive
    };
  }, [method, k, bm25Scale, lexRank, denRank]);

  const fusedOrder = useMemo(() => orderBy(fusedScore), [fusedScore]);
  const fusedRank = useMemo(() => rankOf(fusedOrder), [fusedOrder]);

  const metrics = useMemo(() => ({
    ndcgLex: ndcg(lexOrder), ndcgDen: ndcg(denOrder), ndcgFused: ndcg(fusedOrder),
    tauLex: kendallTau(fusedOrder, lexOrder), tauDen: kendallTau(fusedOrder, denOrder),
    tauKem: kendallTau(fusedOrder, KEMENY),
  }), [lexOrder, denOrder, fusedOrder]);

  const hovered = hover ? byId[hover] : null;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      {/* method toggle */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        {(['rrf', 'borda', 'combsum'] as Method[]).map((m) => (
          <button key={m} onClick={() => setMethod(m)}
            style={{
              fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
              border: `1px solid ${method === m ? 'var(--color-accent)' : 'var(--color-border)'}`,
              background: method === m ? 'var(--color-accent)' : 'transparent',
              color: method === m ? 'var(--color-bg)' : 'var(--color-text)',
            }}>
            {m === 'rrf' ? 'RRF' : m === 'borda' ? 'Borda' : 'CombSUM'}
          </button>
        ))}
      </div>

      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.75rem' }} />

      {/* controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginBottom: '1rem' }}>
        {method === 'rrf' && <Slider label="RRF constant k" value={k} min={1} max={200} step={1} onChange={setK} />}
        <Slider label="BM25 score scale ×" value={bm25Scale} min={1} max={1000} step={1} onChange={setBm25Scale} />
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
          {method === 'combsum'
            ? 'Drag the BM25 scale up — CombSUM collapses onto the lexical order (scale sensitivity).'
            : 'Drag the BM25 scale up — the fused order does not move: RRF and Borda read only positions.'}
        </div>
      </div>

      {/* three ranked columns */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
        <Column title="Lexical · BM25" order={lexOrder} sub="exact-term matches over 10-K text" hover={hover} setHover={setHover} />
        <Column title="Dense · cosine" order={denOrder} sub="semantic matches over call passages" hover={hover} setHover={setHover} />
        <Column
          title={`Fused · ${method === 'rrf' ? 'RRF' : method === 'borda' ? 'Borda' : 'CombSUM'}`}
          order={fusedOrder}
          sub={`query “${QUERY}”`} hover={hover} setHover={setHover} />
      </div>

      {/* hover detail: per-leg rank and the 1/(k+r) contributions */}
      <div style={{ marginTop: '0.75rem', minHeight: '2.2em', fontFamily: 'var(--font-sans)', fontSize: '0.74rem', color: 'var(--color-text-secondary)' }}>
        {hovered ? (
          <span>
            <strong style={{ color: 'var(--color-text)' }}>{hovered.label}</strong> — lexical #{lexRank[hovered.id]}, dense #{denRank[hovered.id]}, fused #{fusedRank[hovered.id]}
            {method === 'rrf' && (
              <> · RRF = 1/({k}+{lexRank[hovered.id]}) + 1/({k}+{denRank[hovered.id]}) = <strong style={{ color: 'var(--color-text)' }}>{(1 / (k + lexRank[hovered.id]) + 1 / (k + denRank[hovered.id])).toFixed(5)}</strong></>
            )}
          </span>
        ) : (
          <span>Hover a document to see its rank in each leg and its fused contribution.</span>
        )}
      </div>

      {/* metrics: NDCG per column + consensus distances */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--color-border)', fontFamily: 'var(--font-sans)', fontSize: '0.74rem' }}>
        <div>
          <div style={{ color: 'var(--color-text-secondary)', marginBottom: '0.2rem' }}>NDCG@{N} (higher is better)</div>
          <div>lexical <strong>{metrics.ndcgLex.toFixed(3)}</strong> · dense <strong>{metrics.ndcgDen.toFixed(3)}</strong> · fused{' '}
            <strong style={{ color: metrics.ndcgFused >= Math.max(metrics.ndcgLex, metrics.ndcgDen) - 1e-9 ? 'var(--color-accent)' : 'var(--color-text)' }}>
              {metrics.ndcgFused.toFixed(3)}
            </strong>
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--color-text-secondary)', marginBottom: '0.2rem' }}>Kendall-τ of fused to…</div>
          <div>lexical <strong>{metrics.tauLex}</strong> · dense <strong>{metrics.tauDen}</strong> · Kemeny consensus <strong>{metrics.tauKem}</strong></div>
        </div>
      </div>
    </div>
  );
}
