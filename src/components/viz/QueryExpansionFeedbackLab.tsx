import { useState } from 'react';

/**
 * Query Expansion Feedback Laboratory — for the Rocchio / RM3 topic.
 * A single feedback-size slider drives three panels:
 *   A. recall@4 vs number of feedback documents — the improve-then-drift curve.
 *   B. the RM1 relevance-model expansion terms and weights at the selected
 *      feedback size, colored by whether they bridge the query (outlook/forecast)
 *      or pollute it (budget/costs/tax) — drift made visible.
 *   C. the re-ranked top-6 with the four relevant documents highlighted.
 *
 * Every number is mirrored to the decimal from pseudo_relevance_feedback.py's
 * viz_constants(); the slider indexes that precomputed table (the relevance-model
 * arithmetic is owned by the notebook, not reproduced here).
 */

const REL = new Set(['r1', 'r2', 'r3', 'r4']);
const LABEL: Record<string, string> = {
  r1: '10-K · rate guidance + outlook',
  r4: '10-K · rate guidance + forecast',
  r2: '10-K · rate outlook/forecast (synonym)',
  r3: 'News · rate forecast/outlook (synonym)',
  n2: '10-K · segment guidance',
  nb: '10-K · cost guidance / budget',
  n1: '10-K · tax rate charge',
};
// bridging (on-topic) expansion terms vs drift (off-topic) ones
const BRIDGE = new Set(['outlook', 'forecast', 'rate', 'guidance', 'projection', 'revision']);

type State = { recall: number; rank: string[]; terms: { t: string; w: number }[] };
// mirrored from viz_constants(): config MU=5, ALPHA=0.5, N_TERMS=10, K=4
const TABLE: State[] = [
  { recall: 0.5, rank: ['r1', 'r4', 'n2', 'nb', 'n1', 'r2'], terms: [] },
  { recall: 1.0, rank: ['r1', 'r4', 'r2', 'r3', 'nb', 'n2'], terms: [
    { t: 'forecast', w: 0.243 }, { t: 'outlook', w: 0.243 }, { t: 'rate', w: 0.137 }, { t: 'guidance', w: 0.122 }, { t: 'charge', w: 0.031 } ] },
  { recall: 1.0, rank: ['r1', 'r4', 'r2', 'r3', 'nb', 'n2'], terms: [
    { t: 'forecast', w: 0.243 }, { t: 'outlook', w: 0.243 }, { t: 'rate', w: 0.137 }, { t: 'guidance', w: 0.122 }, { t: 'charge', w: 0.031 } ] },
  { recall: 0.75, rank: ['r1', 'r4', 'n2', 'r2', 'r3', 'nb'], terms: [
    { t: 'forecast', w: 0.214 }, { t: 'outlook', w: 0.214 }, { t: 'guidance', w: 0.124 }, { t: 'rate', w: 0.123 }, { t: 'headcount', w: 0.050 } ] },
  { recall: 0.5, rank: ['r1', 'r4', 'n2', 'nb', 'r2', 'r3'], terms: [
    { t: 'forecast', w: 0.193 }, { t: 'outlook', w: 0.193 }, { t: 'guidance', w: 0.125 }, { t: 'rate', w: 0.112 }, { t: 'budget', w: 0.045 } ] },
  { recall: 0.75, rank: ['r1', 'r4', 'nb', 'r2', 'r3', 'n2'], terms: [
    { t: 'forecast', w: 0.179 }, { t: 'outlook', w: 0.179 }, { t: 'rate', w: 0.120 }, { t: 'guidance', w: 0.116 }, { t: 'charge', w: 0.046 } ] },
];

export default function QueryExpansionFeedbackLab() {
  const [nfb, setNfb] = useState(2);
  const s = TABLE[nfb];
  const maxW = Math.max(0.001, ...s.terms.map((x) => x.w));

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontFamily: 'var(--font-sans)', fontSize: '0.85rem', marginBottom: '1rem' }}>
        feedback documents = {nfb}
        <input type="range" min={0} max={5} step={1} value={nfb} onChange={(e) => setNfb(+e.target.value)} aria-label="number of feedback documents" />
        <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.78rem' }}>
          query <em>“rate guidance”</em> · pseudo-relevance feedback (RM3)
        </span>
      </label>

      <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
        {/* Panel A — recall curve */}
        <figure style={{ margin: 0 }}>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.4rem' }}>
            A. recall@4 vs feedback size — rises, then drifts
          </figcaption>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.4rem', height: '120px' }}>
            {TABLE.map((st, i) => (
              <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', height: '100%' }}
                onClick={() => setNfb(i)} title={`n_fb=${i}`}>
                <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', color: 'var(--color-text-secondary)' }}>{st.recall.toFixed(2)}</span>
                <span style={{ width: '100%', height: `${st.recall * 80}%`, background: i === nfb ? 'var(--color-accent)' : 'var(--color-accent-secondary)', borderRadius: '3px 3px 0 0', transition: 'all 0.3s', cursor: 'pointer' }} />
                <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', marginTop: '2px', fontWeight: i === nfb ? 600 : 400 }}>{i}</span>
              </div>
            ))}
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'var(--color-text-secondary)', textAlign: 'center', marginTop: '0.2rem' }}>number of feedback documents</div>
        </figure>

        {/* Panel B — expansion terms */}
        <figure style={{ margin: 0 }}>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.4rem' }}>
            B. RM1 relevance model — <span style={{ color: 'var(--color-accent)' }}>bridge</span> vs <span style={{ color: 'var(--color-accent-secondary)' }}>off-topic</span> terms
          </figcaption>
          {s.terms.length === 0 ? (
            <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>No feedback — the original query <em>rate guidance</em> only.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
              {s.terms.map((x) => (
                <div key={x.t} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.78rem' }}>
                  <span style={{ minWidth: '5.5rem' }}>{x.t}</span>
                  <span style={{ flex: 1, height: '0.7rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
                    <span style={{ display: 'block', height: '100%', width: `${(x.w / maxW) * 100}%`, background: BRIDGE.has(x.t) ? 'var(--color-accent)' : 'var(--color-accent-secondary)', transition: 'width 0.3s' }} />
                  </span>
                  <span style={{ minWidth: '3rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{x.w.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}
        </figure>
      </div>

      {/* Panel C — ranking */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.4rem' }}>
          C. re-ranked top-6 — relevant documents in <span style={{ color: 'var(--color-accent)' }}>accent</span>; recall@4 = <strong>{s.recall.toFixed(2)}</strong>
        </div>
        <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
          {s.rank.map((d, i) => {
            const rel = REL.has(d);
            const within = i < 4; // recall@4 cutoff
            return (
              <li key={d} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
                <span style={{ minWidth: '1.25rem', color: 'var(--color-text-secondary)' }}>{i + 1}.</span>
                <span style={{ minWidth: '16rem', color: rel ? 'var(--color-accent)' : 'var(--color-text)', fontWeight: rel ? 600 : 400 }}>{LABEL[d] ?? d}</span>
                <span style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)' }}>{within ? (rel ? '✓ relevant, in top-4' : 'in top-4') : ''}</span>
              </li>
            );
          })}
        </ol>
        <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', marginBottom: 0 }}>
          A little feedback (1–2 docs) adds <em>outlook</em>/<em>forecast</em> and surfaces the synonym filings — recall climbs to 1.0. Push the slider higher and off-topic feedback adds <em>budget</em>/<em>costs</em>, the query drifts, and a relevant document drops out.
        </p>
      </div>
    </div>
  );
}
