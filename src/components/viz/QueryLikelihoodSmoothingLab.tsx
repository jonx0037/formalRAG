import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * Query-Likelihood Smoothing Laboratory — three panels for the query-likelihood
 * topic, driven by a smoothing-method selector (none / Jelinek–Mercer / Dirichlet)
 * and one parameter slider (λ for JM, μ for Dirichlet):
 *   A. document model bars: for a chosen document, the smoothed probability
 *      P(t|d) of each query term, with the raw MLE (marker) and the collection
 *      model P(t|C) (reference line) — watch smoothing interpolate between them,
 *      and watch a missing term's probability climb off the floor.
 *   B. effective smoothing weight per document: Dirichlet's λ_d = μ/(|d|+μ) is
 *      larger for shorter documents (length adaptivity, Proposition 2); JM is a
 *      flat line at λ; "none" is zero.
 *   C. live re-ranking of the shared finance corpus by log P(q|d). With no
 *      smoothing the three documents missing a query term sit at −∞; turn on
 *      either smoothing and all six become finite, the concise on-point filing
 *      first — query likelihood is never length-hijacked because P_ml = tf/|d|.
 *
 * Fully deterministic, so it doubles as a reproducible figure. SVG text inherits
 * the theme color via the global `svg text { fill: var(--color-text) }` rule.
 */

// --- Deterministic finance corpus ---------------------------------------
// query: "interest rate exposure". The corpus text is reused VERBATIM from
// notebooks/bm25/bm25.py; tf records the query-term counts and dl the length.
// collProb (P(t|C)) and COLL_LEN are the only numbers not derivable here — they
// depend on the whole corpus including the filler tokens — and are mirrored to
// the decimal from query_likelihood_language_models.py's viz_constants(). Every
// score below is recomputed from tf, dl, and P(t|C), so the viz and the notebook
// share one set of numbers by construction.
type Doc = { id: string; label: string; dl: number; tf: Record<string, number> };
const QUERY = ['interest', 'rate', 'exposure'];
const COLL_LEN = 316;
const collProb: Record<string, number> = { interest: 0.025316, rate: 0.034810, exposure: 0.018987 };
const CORPUS: Doc[] = [
  { id: 'filing-onpoint', label: '10-K · net interest margin sensitivity', dl: 15, tf: { interest: 2, rate: 2, exposure: 1 } },
  { id: 'transcript-pad', label: 'Earnings call · long Q&A (padded)', dl: 249, tf: { interest: 3, rate: 4, exposure: 2 } },
  { id: 'filing-fx', label: '10-K · foreign-exchange risk', dl: 12, tf: { interest: 0, rate: 1, exposure: 2 } },
  { id: 'news-macro', label: 'News · Fed rate decision', dl: 15, tf: { interest: 1, rate: 2, exposure: 0 } },
  { id: 'filing-boiler', label: '10-K · boilerplate legal', dl: 14, tf: { interest: 1, rate: 1, exposure: 1 } },
  { id: 'transcript-short', label: 'Earnings call · brief update', dl: 11, tf: { interest: 1, rate: 1, exposure: 0 } },
];

type Method = 'none' | 'jm' | 'dirichlet';

// P(t|d) under the active smoothing. method='none' is the bare MLE tf/|d|.
const probTerm = (d: Doc, t: string, method: Method, lambda: number, mu: number): number => {
  const tf = d.tf[t] ?? 0;
  const pml = tf / d.dl;
  const pc = collProb[t];
  if (method === 'jm') return (1 - lambda) * pml + lambda * pc;
  if (method === 'dirichlet') return (tf + mu * pc) / (d.dl + mu);
  return pml;
};

// log P(q|d) = Σ_t c(t,q) log P(t|d); query tf is 1 for each distinct term.
// Returns -Infinity when an unsmoothed model meets a missing query term.
const logScore = (d: Doc, method: Method, lambda: number, mu: number): number =>
  QUERY.reduce((s, t) => {
    const p = probTerm(d, t, method, lambda, mu);
    return s + (p > 0 ? Math.log(p) : -Infinity);
  }, 0);

function Segmented({ value, onChange }: { value: Method; onChange: (m: Method) => void }) {
  const opts: { id: Method; label: string }[] = [
    { id: 'none', label: 'no smoothing (MLE)' },
    { id: 'jm', label: 'Jelinek–Mercer' },
    { id: 'dirichlet', label: 'Dirichlet' },
  ];
  return (
    <div role="radiogroup" aria-label="smoothing method" style={{ display: 'inline-flex', border: '1px solid var(--color-accent)', borderRadius: '999px', overflow: 'hidden' }}>
      {opts.map((o) => (
        <button
          key={o.id}
          role="radio"
          aria-checked={value === o.id}
          onClick={() => onChange(o.id)}
          style={{
            padding: '0.3rem 0.8rem', cursor: 'pointer', border: 'none',
            background: value === o.id ? 'var(--color-accent)' : 'transparent',
            color: value === o.id ? 'var(--color-bg)' : 'var(--color-accent)',
            fontFamily: 'var(--font-sans)', fontSize: '0.78rem', transition: 'all 0.2s',
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export default function QueryLikelihoodSmoothingLab() {
  const [method, setMethod] = useState<Method>('dirichlet');
  const [lambda, setLambda] = useState(0.5);
  const [mu, setMu] = useState(2000);
  const [docId, setDocId] = useState('filing-fx');
  const probDefRef = useRef<HTMLDivElement>(null);
  const scoreDefRef = useRef<HTMLDivElement>(null);
  const modelRef = useRef<SVGSVGElement>(null);
  const weightRef = useRef<SVGSVGElement>(null);

  const selectedDoc = CORPUS.find((d) => d.id === docId) ?? CORPUS[0];

  // live KaTeX: the smoothed-probability definition (reflects the method)
  useEffect(() => {
    if (!probDefRef.current) return;
    const tex =
      method === 'jm'
        ? `P_\\lambda(t\\mid d)=(1-\\lambda)\\tfrac{\\text{tf}_{t,d}}{|d|}+\\lambda\\,P(t\\mid C)`
        : method === 'dirichlet'
        ? `P_\\mu(t\\mid d)=\\dfrac{\\text{tf}_{t,d}+\\mu\\,P(t\\mid C)}{|d|+\\mu}`
        : `P_{\\mathrm{ml}}(t\\mid d)=\\dfrac{\\text{tf}_{t,d}}{|d|}`;
    katex.render(tex, probDefRef.current, { throwOnError: false, displayMode: true });
  }, [method]);

  // live KaTeX: the query-likelihood score (static form)
  useEffect(() => {
    if (!scoreDefRef.current) return;
    katex.render(
      `\\log P(q\\mid d)=\\sum_{t\\in q} c(t,q)\\,\\log P(t\\mid d)`,
      scoreDefRef.current, { throwOnError: false, displayMode: true },
    );
  }, []);

  // Panel A — document model bars for the selected document
  useEffect(() => {
    const svg = d3.select(modelRef.current);
    svg.selectAll('*').remove();
    const W = 360, H = 240, m = { t: 16, r: 16, b: 50, l: 48 };
    const smoothed = QUERY.map((t) => probTerm(selectedDoc, t, method, lambda, mu));
    const mle = QUERY.map((t) => (selectedDoc.tf[t] ?? 0) / selectedDoc.dl);
    const pc = QUERY.map((t) => collProb[t]);
    const yMax = Math.max(...smoothed, ...mle, ...pc, 1e-6) * 1.15;
    const x = d3.scaleBand<string>().domain(QUERY).range([m.l, W - m.r]).padding(0.3);
    const y = d3.scaleLinear().domain([0, yMax]).range([H - m.b, m.t]);
    const g = svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
    g.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('.3f')));
    // smoothed probability bars
    g.selectAll('rect').data(QUERY).join('rect')
      .attr('x', (t) => x(t) ?? 0).attr('width', x.bandwidth())
      .attr('y', (_, i) => y(smoothed[i])).attr('height', (_, i) => y(0) - y(smoothed[i]))
      .attr('fill', 'var(--color-accent)').attr('opacity', 0.85);
    // MLE markers (where the raw estimate sits — 0 for a missing term)
    g.selectAll('line.mle').data(QUERY).join('line').attr('class', 'mle')
      .attr('x1', (t) => x(t) ?? 0).attr('x2', (t) => (x(t) ?? 0) + x.bandwidth())
      .attr('y1', (_, i) => y(mle[i])).attr('y2', (_, i) => y(mle[i]))
      .attr('stroke', 'var(--color-text)').attr('stroke-width', 2).attr('stroke-dasharray', '4 2');
    // collection-model reference ticks
    g.selectAll('line.pc').data(QUERY).join('line').attr('class', 'pc')
      .attr('x1', (t) => x(t) ?? 0).attr('x2', (t) => (x(t) ?? 0) + x.bandwidth())
      .attr('y1', (_, i) => y(pc[i])).attr('y2', (_, i) => y(pc[i]))
      .attr('stroke', 'var(--color-accent-secondary)').attr('stroke-width', 1.5);
    g.selectAll('text.k').data(QUERY).join('text').attr('class', 'k')
      .attr('x', (t) => (x(t) ?? 0) + x.bandwidth() / 2).attr('y', H - m.b + 14)
      .attr('text-anchor', 'middle').attr('font-size', 10)
      .text((t) => `${t} (tf ${selectedDoc.tf[t] ?? 0})`);
    g.append('text').attr('x', W / 2).attr('y', H - 6).attr('text-anchor', 'middle').attr('font-size', 11)
      .text('P(t|d): bar = smoothed, dashed = MLE, line = P(t|C)');
  }, [method, lambda, mu, docId]);

  // Panel B — effective smoothing weight per document (length adaptivity)
  useEffect(() => {
    const svg = d3.select(weightRef.current);
    svg.selectAll('*').remove();
    const W = 360, H = 240, m = { t: 16, r: 16, b: 64, l: 40 };
    const byLen = [...CORPUS].sort((a, b) => a.dl - b.dl);
    const eff = (d: Doc) => (method === 'jm' ? lambda : method === 'dirichlet' ? mu / (d.dl + mu) : 0);
    const x = d3.scaleBand<string>().domain(byLen.map((d) => d.id)).range([m.l, W - m.r]).padding(0.25);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
    const g = svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
    g.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4));
    g.selectAll('rect').data(byLen).join('rect')
      .attr('x', (d) => x(d.id) ?? 0).attr('width', x.bandwidth())
      .attr('y', (d) => y(eff(d))).attr('height', (d) => y(0) - y(eff(d)))
      .attr('fill', (d) => (d.id === docId ? 'var(--color-accent)' : 'var(--color-accent-secondary)')).attr('opacity', 0.85);
    g.selectAll('text.l').data(byLen).join('text').attr('class', 'l')
      .attr('transform', (d) => `translate(${(x(d.id) ?? 0) + x.bandwidth() / 2},${H - m.b + 8}) rotate(40)`)
      .attr('text-anchor', 'start').attr('font-size', 9)
      .text((d) => `|d|=${d.dl}`);
    g.append('text').attr('x', W / 2).attr('y', H - 4).attr('text-anchor', 'middle').attr('font-size', 11)
      .text(method === 'dirichlet' ? 'effective weight μ/(|d|+μ) ↓ as |d| ↑' : method === 'jm' ? 'effective weight λ (flat)' : 'no smoothing');
  }, [method, lambda, mu, docId]);

  // Panel C — live ranking
  const ranked = useMemo(
    () => CORPUS.map((d) => ({ ...d, s: logScore(d, method, lambda, mu) })).sort((a, c) => c.s - a.s),
    [method, lambda, mu],
  );
  const finite = ranked.map((d) => d.s).filter((s) => Number.isFinite(s));
  const best = finite.length ? Math.max(...finite) : -1;
  const worst = finite.length ? Math.min(...finite) : -1;
  const span = best - worst || 1;
  const topId = ranked[0].id;
  // bar width: map finite scores into [12%, 100%]; -inf → 0
  const barPct = (s: number) => (Number.isFinite(s) ? 12 + 88 * ((s - worst) / span) : 0);

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem 2rem', marginBottom: '0.75rem' }}>
        <div ref={probDefRef} style={{ overflowX: 'auto' }} />
        <div ref={scoreDefRef} style={{ overflowX: 'auto' }} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem 1.5rem', marginBottom: '1rem' }}>
        <Segmented value={method} onChange={setMethod} />
        {method === 'jm' && (
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
            λ = {lambda.toFixed(2)}
            <input type="range" min={0.01} max={0.99} step={0.01} value={lambda} onChange={(e) => setLambda(+e.target.value)} aria-label="Jelinek-Mercer lambda" />
          </label>
        )}
        {method === 'dirichlet' && (
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
            μ = {mu}
            <input type="range" min={1} max={5000} step={1} value={mu} onChange={(e) => setMu(+e.target.value)} aria-label="Dirichlet mu" />
          </label>
        )}
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem' }}>
          panel A document
          <select value={docId} onChange={(e) => setDocId(e.target.value)} aria-label="document for panel A"
            style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.2rem 0.4rem', borderRadius: '0.4rem', border: '1px solid var(--color-border)', background: 'var(--color-bg)', color: 'var(--color-text)' }}>
            {CORPUS.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
          </select>
        </label>
      </div>
      <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
        <figure style={{ margin: 0 }}>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
            A. document model — pick the FX filing (no “interest”) and watch its zero MLE lift to P(t|C).
          </figcaption>
          <svg ref={modelRef} role="img" aria-label="Smoothed document model probabilities for the selected document" />
        </figure>
        <figure style={{ margin: 0 }}>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
            B. effective smoothing weight — Dirichlet smooths short documents more.
          </figcaption>
          <svg ref={weightRef} role="img" aria-label="Effective smoothing weight per document" />
        </figure>
      </div>
      <div style={{ marginTop: '1rem' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
          C. live ranking by <em>log P(q|d)</em> for <em>“interest rate exposure”</em> — switch to <strong>no smoothing</strong> to send the three documents missing a query term to <em>−∞</em>
        </div>
        <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          {ranked.map((d, i) => (
            <li key={d.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem', transition: 'all 0.4s' }}>
              <span style={{ minWidth: '1.25rem', color: 'var(--color-text-secondary)' }}>{i + 1}.</span>
              <span style={{ minWidth: '15rem', fontWeight: d.id === topId ? 600 : 400, opacity: Number.isFinite(d.s) ? 1 : 0.55 }}>{d.label}</span>
              <span style={{ flex: 1, height: '0.75rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
                <span style={{ display: 'block', height: '100%', width: `${barPct(d.s)}%`, background: d.id === topId ? 'var(--color-accent)' : 'var(--color-accent-secondary)', transition: 'width 0.4s' }} />
              </span>
              <span style={{ minWidth: '3.5rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{Number.isFinite(d.s) ? d.s.toFixed(2) : '−∞'}</span>
            </li>
          ))}
        </ol>
        <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', marginBottom: 0 }}>
          The maximum-likelihood estimate <em>tf/|d|</em> already divides by length, so the padded transcript never hijacks the top — query likelihood’s only real problem is the zero, which smoothing removes.
        </p>
      </div>
    </div>
  );
}
