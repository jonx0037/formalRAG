import { useMemo, useState } from 'react';

/**
 * Inverted Index Pruning Visualizer — for the WAND / BlockMax-WAND topic.
 * Driven by a method selector (exhaustive / WAND / BlockMax-WAND) and a top-k slider:
 *   A. postings grid: rows are query terms, columns the worked corpus's documents,
 *      each cell the BM25 contribution; per-term upper bound UB_t at the row end.
 *      Under a pruning method, documents never fully scored are dimmed.
 *   B. documents fully scored: exhaustive vs WAND vs BlockMax-WAND, the selected
 *      method highlighted — the pruning payoff, exact-top-k preserved throughout.
 *   C. the resulting top-k ranking (identical across all three methods: safety).
 *
 * The per-(term, document) BM25 contributions and the per-term upper bounds are
 * mirrored to the decimal from inverted_index_dynamic_pruning.py's viz_constants();
 * the three algorithms are replayed here over those numbers, so the counts match
 * the notebook (worked corpus, k=3: exhaustive 10, WAND 6, BlockMax-WAND 6).
 */

const QTERMS = ['interest', 'rate', 'exposure'] as const;
const DOCS = [
  'd0-margin', 'd1-hedge', 'd2-fx', 'd3-macro', 'd4-credit',
  'd5-liquidity', 'd6-capital', 'd7-guidance', 'd8-tax', 'd9-boiler',
];
const DOC_IDX: Record<string, number> = Object.fromEntries(DOCS.map((d, i) => [d, i]));

// Postings mirrored from viz_constants(): [docId, contribution], sorted by docId.
type Posting = { doc: number; c: number };
const POSTINGS: Record<string, Posting[]> = {
  interest: [
    { doc: 0, c: 1.532259 }, { doc: 1, c: 1.276883 }, { doc: 5, c: 0.893818 }, { doc: 9, c: 0.893818 },
  ],
  rate: [
    { doc: 0, c: 0.441993 }, { doc: 1, c: 0.368327 }, { doc: 3, c: 0.355626 }, { doc: 5, c: 0.257829 },
    { doc: 6, c: 0.257829 }, { doc: 7, c: 0.257829 }, { doc: 8, c: 0.245552 }, { doc: 9, c: 0.257829 },
  ],
  exposure: [
    { doc: 0, c: 1.026885 }, { doc: 1, c: 0.693147 }, { doc: 2, c: 1.026885 }, { doc: 4, c: 0.99021 }, { doc: 9, c: 0.693147 },
  ],
};
const UB: Record<string, number> = { interest: 1.532259, rate: 0.441993, exposure: 1.026885 };

type Method = 'exhaustive' | 'wand' | 'bmw';
type Result = { topk: { doc: number; s: number }[]; fullEvals: number; scored: Set<number> };

function exhaustive(k: number): Result {
  const acc: Record<number, number> = {};
  for (const t of QTERMS) for (const p of POSTINGS[t]) acc[p.doc] = (acc[p.doc] ?? 0) + p.c;
  const ranked = Object.entries(acc).map(([d, s]) => ({ doc: +d, s })).sort((a, b) => b.s - a.s || a.doc - b.doc);
  return { topk: ranked.slice(0, k), fullEvals: ranked.length, scored: new Set(ranked.map((r) => r.doc)) };
}

// shared DAAT pivot loop; `bmw` adds the block-max refinement gate
function pruned(k: number, useBlocks: boolean, blockSize = 2): Result {
  const cur: Record<string, number> = Object.fromEntries(QTERMS.map((t) => [t, 0]));
  const curDoc = (t: string) => (cur[t] < POSTINGS[t].length ? POSTINGS[t][cur[t]].doc : Infinity);
  const blocks: Record<string, { last: number; max: number }[]> = {};
  for (const t of QTERMS) {
    const b: { last: number; max: number }[] = [];
    for (let s = 0; s < POSTINGS[t].length; s += blockSize) {
      const chunk = POSTINGS[t].slice(s, s + blockSize);
      b.push({ last: chunk[chunk.length - 1].doc, max: Math.max(...chunk.map((p) => p.c)) });
    }
    blocks[t] = b;
  }
  const blockUB = (t: string, doc: number) => {
    for (const blk of blocks[t]) if (blk.last >= doc) return blk.max;
    return 0;
  };
  const heap: { s: number; doc: number }[] = [];
  const threshold = () => (heap.length >= k ? Math.min(...heap.map((h) => h.s)) : -Infinity);
  let fullEvals = 0;
  const scored = new Set<number>();
  // guard against any pathological non-termination in the browser
  for (let guard = 0; guard < 10000; guard++) {
    const active = QTERMS.filter((t) => cur[t] < POSTINGS[t].length).sort((a, b) => curDoc(a) - curDoc(b));
    if (!active.length) break;
    const theta = threshold();
    let cum = 0;
    let pivot: string | null = null;
    for (const t of active) { cum += UB[t]; if (cum >= theta) { pivot = t; break; } }
    if (pivot === null) break;
    const pd = curDoc(pivot);
    if (curDoc(active[0]) < pd) {
      // the smallest-id cursor lags the pivot -> skip it forward, then re-pivot
      const t = active[0];
      while (cur[t] < POSTINGS[t].length && POSTINGS[t][cur[t]].doc < pd) cur[t]++;
      continue;
    }
    // active[0] === pd: the terms aligned here are the only contributors
    const present = active.filter((t) => curDoc(t) === pd);
    if (useBlocks) {
      const refined = present.reduce((s, t) => s + blockUB(t, pd), 0);
      if (refined < theta) { for (const t of present) cur[t]++; continue; }
    }
    let score = 0;
    for (const t of present) { score += POSTINGS[t][cur[t]].c; cur[t]++; }
    fullEvals++; scored.add(pd);
    if (heap.length < k) heap.push({ s: score, doc: pd });
    else { const m = Math.min(...heap.map((h) => h.s)); if (score > m) { const wi = heap.reduce((mi, h, i, a) => (h.s < a[mi].s ? i : mi), 0); heap[wi] = { s: score, doc: pd }; } }
  }
  const topk = heap.slice().sort((a, b) => b.s - a.s || a.doc - b.doc);
  return { topk, fullEvals, scored };
}

const heat = (c: number) => {
  const max = 1.6;
  return `color-mix(in srgb, var(--color-accent) ${Math.round((c / max) * 80 + 15)}%, transparent)`;
};

function MethodTab({ id, label, active, onClick }: { id: Method; label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      role="radio" aria-checked={active} onClick={onClick}
      style={{
        padding: '0.3rem 0.8rem', cursor: 'pointer', border: 'none',
        background: active ? 'var(--color-accent)' : 'transparent',
        color: active ? 'var(--color-bg)' : 'var(--color-accent)',
        fontFamily: 'var(--font-sans)', fontSize: '0.78rem', transition: 'all 0.2s',
      }}
    >{label}</button>
  );
}

export default function InvertedIndexPruningVisualizer() {
  const [method, setMethod] = useState<Method>('wand');
  const [k, setK] = useState(3);

  const ex = useMemo(() => exhaustive(k), [k]);
  const wand = useMemo(() => pruned(k, false), [k]);
  const bmw = useMemo(() => pruned(k, true), [k]);
  const sel = method === 'exhaustive' ? ex : method === 'wand' ? wand : bmw;
  const counts = [
    { id: 'exhaustive' as Method, label: 'exhaustive', n: ex.fullEvals },
    { id: 'wand' as Method, label: 'WAND', n: wand.fullEvals },
    { id: 'bmw' as Method, label: 'BlockMax-WAND', n: bmw.fullEvals },
  ];
  const maxN = Math.max(...counts.map((c) => c.n));

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem 1.5rem', marginBottom: '1rem' }}>
        <div role="radiogroup" aria-label="retrieval method" style={{ display: 'inline-flex', border: '1px solid var(--color-accent)', borderRadius: '999px', overflow: 'hidden' }}>
          <MethodTab id="exhaustive" label="exhaustive" active={method === 'exhaustive'} onClick={() => setMethod('exhaustive')} />
          <MethodTab id="wand" label="WAND" active={method === 'wand'} onClick={() => setMethod('wand')} />
          <MethodTab id="bmw" label="BlockMax-WAND" active={method === 'bmw'} onClick={() => setMethod('bmw')} />
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
          top-k = {k}
          <input type="range" min={1} max={6} step={1} value={k} onChange={(e) => setK(+e.target.value)} aria-label="top-k" />
        </label>
      </div>

      {/* Panel A — postings grid */}
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.35rem' }}>
        A. postings lists — cell = BM25 contribution; columns dimmed under <strong>{counts.find((c) => c.id === method)!.label}</strong> were never fully scored
      </div>
      <div style={{ overflowX: 'auto', marginBottom: '1rem' }}>
        <table style={{ borderCollapse: 'collapse', fontFamily: 'var(--font-sans)', fontSize: '0.7rem', minWidth: '640px' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '2px 6px' }}></th>
              {DOCS.map((d) => (
                <th key={d} style={{ padding: '2px 4px', textAlign: 'center', opacity: sel.scored.has(DOC_IDX[d]) ? 1 : 0.32, writingMode: 'vertical-rl', transform: 'rotate(200deg)', whiteSpace: 'nowrap' }}>{d}</th>
              ))}
              <th style={{ padding: '2px 6px', textAlign: 'right' }}>UB</th>
            </tr>
          </thead>
          <tbody>
            {QTERMS.map((t) => {
              const byDoc: Record<number, number> = Object.fromEntries(POSTINGS[t].map((p) => [p.doc, p.c]));
              return (
                <tr key={t}>
                  <td style={{ padding: '2px 6px', fontWeight: 600 }}>{t}</td>
                  {DOCS.map((d) => {
                    const i = DOC_IDX[d];
                    const c = byDoc[i];
                    return (
                      <td key={d} style={{ padding: '2px 4px', textAlign: 'center', background: c ? heat(c) : 'transparent', opacity: sel.scored.has(i) ? 1 : 0.32, fontVariantNumeric: 'tabular-nums' }}>
                        {c ? c.toFixed(2) : '·'}
                      </td>
                    );
                  })}
                  <td style={{ padding: '2px 6px', textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>{UB[t].toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Panel B — documents fully scored */}
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.35rem' }}>
        B. documents fully scored (lower is cheaper) — all three return the <em>same</em> top-{k}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', marginBottom: '1rem' }}>
        {counts.map((c) => (
          <div key={c.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem', opacity: c.id === method ? 1 : 0.6 }}>
            <span style={{ minWidth: '9rem' }}>{c.label}</span>
            <span style={{ flex: 1, height: '0.8rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
              <span style={{ display: 'block', height: '100%', width: `${(c.n / maxN) * 100}%`, background: c.id === method ? 'var(--color-accent)' : 'var(--color-accent-secondary)', transition: 'width 0.4s' }} />
            </span>
            <span style={{ minWidth: '4.5rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{c.n} / {DOCS.length}</span>
          </div>
        ))}
      </div>

      {/* Panel C — ranking */}
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.35rem' }}>
        C. top-{k} for <em>“interest rate exposure”</em>
      </div>
      <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
        {sel.topk.map((r, i) => (
          <li key={r.doc} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
            <span style={{ minWidth: '1.25rem', color: 'var(--color-text-secondary)' }}>{i + 1}.</span>
            <span style={{ minWidth: '9rem', fontWeight: i === 0 ? 600 : 400 }}>{DOCS[r.doc]}</span>
            <span style={{ flex: 1, height: '0.7rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
              <span style={{ display: 'block', height: '100%', width: `${(r.s / sel.topk[0].s) * 100}%`, background: i === 0 ? 'var(--color-accent)' : 'var(--color-accent-secondary)' }} />
            </span>
            <span style={{ minWidth: '3.5rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{r.s.toFixed(3)}</span>
          </li>
        ))}
      </ol>
      <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', marginBottom: 0 }}>
        On ten documents the gap is small; at scale it is not — the notebook scores 209 of 5000 documents with BlockMax-WAND where the exhaustive scan scores 2870, for the identical top-k.
      </p>
    </div>
  );
}
