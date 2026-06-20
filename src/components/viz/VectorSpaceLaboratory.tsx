import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * Vector Space Laboratory — three panels for the TF-IDF topic, driven by two
 * toggles (sublinear tf, cosine normalization):
 *   A. self-information bars: idf_t = log2(N/df_t) in BITS per query term, making
 *      Theorem 1 visible — a term in every document ("rate") carries 0 bits.
 *   B. tf-scaling curve: raw tf (dashed) vs 1 + log(tf) (solid), with NO ceiling
 *      line — sublinear tf is concave but UNBOUNDED, the gap BM25 closes.
 *   C. live re-ranking of the shared finance corpus: with the raw tf-idf dot
 *      product the padded transcript hijacks #1; switch on cosine normalization
 *      and the concise on-point filing surfaces — the same winner BM25 reaches
 *      with b = 0.75.
 *
 * Fully deterministic, so it doubles as a reproducible figure. SVG text inherits
 * the theme color via the global `svg text { fill: var(--color-text) }` rule.
 */

// --- Deterministic finance corpus ---------------------------------------
// query: "interest rate exposure". The corpus text is reused VERBATIM from
// notebooks/bm25/bm25.py; tf records the query-term counts and dl the length.
// normSub / normRaw are each document's FULL TF-IDF L2 norm (the cosine
// denominator, which includes the padding terms) under the 1+log(tf) and raw-tf
// scalings — these are the only numbers not derivable here, mirrored to the
// decimal from vector_space_model_tfidf.py's viz_constants() (the same smoothed
// IDF whose ranking test_length_hijack_flip() asserts).
type Doc = { id: string; label: string; dl: number; tf: Record<string, number>; normSub: number; normRaw: number };
const QUERY = ['interest', 'rate', 'exposure'];
const CORPUS: Doc[] = [
  { id: 'filing-onpoint', label: '10-K · net interest margin sensitivity', dl: 15, tf: { interest: 2, rate: 2, exposure: 1 }, normSub: 6.2024, normRaw: 6.6340 },
  { id: 'transcript-pad', label: 'Earnings call · long Q&A (padded)', dl: 249, tf: { interest: 3, rate: 4, exposure: 2 }, normSub: 27.9248, normRaw: 112.4646 },
  { id: 'filing-fx', label: '10-K · foreign-exchange risk', dl: 12, tf: { interest: 0, rate: 1, exposure: 2 }, normSub: 5.5036, normRaw: 5.5894 },
  { id: 'news-macro', label: 'News · Fed rate decision', dl: 15, tf: { interest: 1, rate: 2, exposure: 0 }, normSub: 6.4389, normRaw: 6.5858 },
  { id: 'filing-boiler', label: '10-K · boilerplate legal', dl: 14, tf: { interest: 1, rate: 1, exposure: 1 }, normSub: 6.3755, normRaw: 6.3755 },
  { id: 'transcript-short', label: 'Earnings call · brief update', dl: 11, tf: { interest: 1, rate: 1, exposure: 0 }, normSub: 4.9388, normRaw: 5.0754 },
];

const N = CORPUS.length;
// document frequency per query term (presence across the corpus)
const DF: Record<string, number> = Object.fromEntries(
  QUERY.map((t) => [t, CORPUS.filter((d) => (d.tf[t] ?? 0) > 0).length]),
);
// Theorem 1: the self-information of a term's presence, in bits (textbook idf).
const idfBits = (t: string) => Math.log2(N / DF[t]);
// The smoothed IDF actually used in scoring: log(1 + N/df), strictly positive so
// the universal term "rate" keeps a small weight (smoothing is a convention).
const idfSmooth = (t: string) => Math.log(1 + N / DF[t]);
const scale = (tf: number, sublinear: boolean) => (sublinear ? (tf > 0 ? 1 + Math.log(tf) : 0) : tf);

// query weight = scale(1)*idf = idf (query tf is 1), so qnorm is the same in both scalings
const Q_NORM = Math.sqrt(QUERY.reduce((s, t) => s + idfSmooth(t) ** 2, 0));
const rawDot = (d: Doc, sublinear: boolean) =>
  QUERY.reduce((s, t) => s + idfSmooth(t) * (scale(d.tf[t] ?? 0, sublinear) * idfSmooth(t)), 0);
const score = (d: Doc, cosine: boolean, sublinear: boolean) => {
  const dot = rawDot(d, sublinear);
  if (!cosine) return dot;
  const docNorm = sublinear ? d.normSub : d.normRaw;
  return dot / (docNorm * Q_NORM);
};

function Toggle({ label, on, onLabel, offLabel, onToggle }: {
  label: string; on: boolean; onLabel: string; offLabel: string; onToggle: (v: boolean) => void;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontFamily: 'var(--font-sans)', fontSize: '0.85rem' }}>
      <span style={{ minWidth: '9rem' }}>{label}</span>
      <button
        onClick={() => onToggle(!on)}
        style={{
          padding: '0.3rem 0.8rem', borderRadius: '999px', cursor: 'pointer',
          border: '1px solid var(--color-accent)',
          background: on ? 'var(--color-accent)' : 'transparent',
          color: on ? 'var(--color-bg)' : 'var(--color-accent)',
          fontFamily: 'var(--font-sans)', fontSize: '0.8rem', transition: 'all 0.2s',
        }}
        aria-pressed={on}
        aria-label={`${label}: ${on ? onLabel : offLabel}`}
      >
        {on ? onLabel : offLabel}
      </button>
    </div>
  );
}

export default function VectorSpaceLaboratory() {
  const [sublinear, setSublinear] = useState(true);
  const [cosine, setCosine] = useState(true);
  const weightRef = useRef<HTMLDivElement>(null);
  const scoreRef = useRef<HTMLDivElement>(null);
  const idfRef = useRef<SVGSVGElement>(null);
  const curveRef = useRef<SVGSVGElement>(null);

  // live KaTeX: the weight definition (reflects the tf-scaling toggle)
  useEffect(() => {
    if (!weightRef.current) return;
    const tfPart = sublinear ? '(1+\\log \\text{tf})' : '\\text{tf}';
    katex.render(
      `w_{t,d}=${tfPart}\\cdot\\log\\!\\left(1+\\tfrac{N}{\\text{df}_t}\\right)`,
      weightRef.current, { throwOnError: false, displayMode: true },
    );
  }, [sublinear]);

  // live KaTeX: the score (reflects the cosine toggle)
  useEffect(() => {
    if (!scoreRef.current) return;
    const tex = cosine
      ? `\\text{score}(q,d)=\\frac{\\sum_{t\\in q} w_{t,q}\\,w_{t,d}}{\\lVert w_q\\rVert\\,\\lVert w_d\\rVert}`
      : `\\text{score}(q,d)=\\sum_{t\\in q} w_{t,q}\\,w_{t,d}`;
    katex.render(tex, scoreRef.current, { throwOnError: false, displayMode: true });
  }, [cosine]);

  // Panel A — self-information bars (bits), static; the theorem made visible
  useEffect(() => {
    const svg = d3.select(idfRef.current);
    svg.selectAll('*').remove();
    const W = 360, H = 240, m = { t: 16, r: 16, b: 48, l: 36 };
    const x = d3.scaleBand<string>().domain(QUERY).range([m.l, W - m.r]).padding(0.3);
    const y = d3.scaleLinear().domain([0, Math.max(1, d3.max(QUERY, idfBits) ?? 1)]).range([H - m.b, m.t]);
    const g = svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
    g.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4));
    g.selectAll('rect').data(QUERY).join('rect')
      .attr('x', (t) => x(t) ?? 0).attr('width', x.bandwidth())
      .attr('y', (t) => y(idfBits(t))).attr('height', (t) => y(0) - y(idfBits(t)))
      .attr('fill', 'var(--color-accent)').attr('opacity', 0.85);
    g.selectAll('text.v').data(QUERY).join('text').attr('class', 'v')
      .attr('x', (t) => (x(t) ?? 0) + x.bandwidth() / 2).attr('y', (t) => y(idfBits(t)) - 4)
      .attr('text-anchor', 'middle').attr('font-size', 10)
      .text((t) => `${idfBits(t).toFixed(2)}`);
    g.selectAll('text.k').data(QUERY).join('text').attr('class', 'k')
      .attr('x', (t) => (x(t) ?? 0) + x.bandwidth() / 2).attr('y', H - m.b + 14)
      .attr('text-anchor', 'middle').attr('font-size', 10)
      .text((t) => `${t} (df ${DF[t]}/${N})`);
    g.append('text').attr('x', W / 2).attr('y', H - 6).attr('text-anchor', 'middle').attr('font-size', 11)
      .text('self-information  idf = log₂(N/df)  [bits]');
  }, []);

  // Panel B — tf-scaling curve: raw vs 1+log, NO ceiling line
  useEffect(() => {
    const svg = d3.select(curveRef.current);
    svg.selectAll('*').remove();
    const W = 360, H = 240, m = { t: 16, r: 16, b: 36, l: 32 };
    const x = d3.scaleLinear().domain([0, 20]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 20]).range([H - m.b, m.t]);
    const g = svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
    g.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5));
    g.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
    const xs = d3.range(0, 20.05, 0.25);
    const line = (f: (t: number) => number, color: string, width: number, dash?: string) =>
      g.append('path').datum(xs).attr('fill', 'none').attr('stroke', color).attr('stroke-width', width)
        .attr('stroke-dasharray', dash ?? null)
        .attr('d', d3.line<number>().x((t) => x(t)).y((t) => y(Math.min(20, f(t))))(xs) as string);
    // raw tf (y = x): the unbounded baseline; emphasized when sublinear is off
    line((t) => t, sublinear ? 'var(--color-muted-border)' : 'var(--color-accent)', sublinear ? 1.5 : 2.5, '5 3');
    // 1 + log(tf): concave but still unbounded; emphasized when sublinear is on
    line((t) => (t > 0 ? 1 + Math.log(t) : 0), sublinear ? 'var(--color-accent)' : 'var(--color-text-secondary)', sublinear ? 2.5 : 1.5);
    g.append('text').attr('x', W - m.r).attr('y', y(19) ).attr('text-anchor', 'end').attr('font-size', 10)
      .attr('fill', 'var(--color-text-secondary)').text('no ceiling: both grow without bound');
    g.append('text').attr('x', W / 2).attr('y', H - 4).attr('text-anchor', 'middle').attr('font-size', 11)
      .text('term frequency  tf');
  }, [sublinear]);

  // Panel C — live re-ranking
  const ranked = useMemo(
    () => CORPUS.map((d) => ({ ...d, s: score(d, cosine, sublinear) })).sort((a, c) => c.s - a.s),
    [cosine, sublinear],
  );
  const maxScore = Math.max(...ranked.map((d) => d.s), 1e-9);
  const topId = ranked[0].id;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem 2rem', marginBottom: '0.75rem' }}>
        <div ref={weightRef} style={{ overflowX: 'auto' }} />
        <div ref={scoreRef} style={{ overflowX: 'auto' }} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem 2rem', marginBottom: '1rem' }}>
        <Toggle label="term frequency" on={sublinear} onLabel="sublinear  1+log(tf)" offLabel="raw  tf" onToggle={setSublinear} />
        <Toggle label="normalization" on={cosine} onLabel="cosine (÷ ‖w‖)" offLabel="raw dot product" onToggle={setCosine} />
      </div>
      <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
        <figure style={{ margin: 0 }}>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
            A. inverse document frequency = self-information (bits). “rate” is in every doc → 0 bits.
          </figcaption>
          <svg ref={idfRef} role="img" aria-label="Self-information of each query term in bits" />
        </figure>
        <figure style={{ margin: 0 }}>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
            B. tf scaling — solid: active transform, dashed: raw tf. Neither saturates.
          </figcaption>
          <svg ref={curveRef} role="img" aria-label="Term-frequency scaling curve" />
        </figure>
      </div>
      <div style={{ marginTop: '1rem' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
          C. live ranking for query <em>“interest rate exposure”</em> — switch normalization to <strong>raw dot product</strong> to watch the padded transcript hijack the top spot
        </div>
        <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          {ranked.map((d, i) => (
            <li key={d.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem', transition: 'all 0.4s' }}>
              <span style={{ minWidth: '1.25rem', color: 'var(--color-text-secondary)' }}>{i + 1}.</span>
              <span style={{ minWidth: '15rem', fontWeight: d.id === topId ? 600 : 400 }}>{d.label}</span>
              <span style={{ flex: 1, height: '0.75rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
                <span style={{ display: 'block', height: '100%', width: `${(d.s / maxScore) * 100}%`, background: d.id === topId ? 'var(--color-accent)' : 'var(--color-accent-secondary)', transition: 'width 0.4s' }} />
              </span>
              <span style={{ minWidth: '3.5rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{d.s.toFixed(3)}</span>
            </li>
          ))}
        </ol>
        <p style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', marginBottom: 0 }}>
          Cosine normalization is all-or-nothing — it divides by the whole document norm. BM25 turns this into a dial (the <em>b</em> parameter) and adds a saturation ceiling.
        </p>
      </div>
    </div>
  );
}
