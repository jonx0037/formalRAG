import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * BM25 Scoring Laboratory — three linked panels driven by two shared sliders
 * (k1, b), validating the three pillars of the BM25 pilot topic:
 *   A. tf-saturation curve, with the (k1+1) asymptote and raw-tf / log(1+tf)
 *      reference curves — the limit theorems made visible.
 *   B. length normalization B(dl) = 1 - b + b·dl/avgdl across documents.
 *   C. live re-ranking of a deterministic toy finance corpus — at b=0 a verbose
 *      length-hijacking transcript wins; raising b surfaces the on-point filing.
 *
 * Fully deterministic, so it doubles as a reproducible figure. SVG text inherits
 * the theme color via the global `svg text { fill: var(--color-text) }` rule.
 */

// --- Deterministic toy finance corpus -----------------------------------
// query: "interest rate exposure". Each doc records its length (dl, in tokens)
// and the term frequency of each query term. Doc "transcript-pad" keyword-stuffs
// "rate" in a long, low-signal earnings-call transcript; "filing-onpoint" is a
// concise 10-K risk sentence that actually answers the query.
// EDIT ME: this corpus is the pedagogical payload — tune freely.
type Doc = { id: string; label: string; dl: number; tf: Record<string, number> };
const QUERY = ['interest', 'rate', 'exposure'];
// Counts and lengths match the verified corpus in notebooks/bm25/bm25.py exactly,
// so this panel reproduces the same b=0 -> padded-transcript-#1, b=0.75 ->
// on-point-filing-#1 flip that the notebook's test_length_hijack() asserts.
const CORPUS: Doc[] = [
  { id: 'filing-onpoint', label: '10-K · net interest margin sensitivity', dl: 15, tf: { interest: 2, rate: 2, exposure: 1 } },
  { id: 'transcript-pad', label: 'Earnings call · long Q&A (padded)', dl: 249, tf: { interest: 3, rate: 4, exposure: 2 } },
  { id: 'filing-fx', label: '10-K · foreign-exchange risk', dl: 12, tf: { interest: 0, rate: 1, exposure: 2 } },
  { id: 'news-macro', label: 'News · Fed rate decision', dl: 15, tf: { interest: 1, rate: 2, exposure: 0 } },
  { id: 'filing-boiler', label: '10-K · boilerplate legal', dl: 14, tf: { interest: 1, rate: 1, exposure: 1 } },
  { id: 'transcript-short', label: 'Earnings call · brief update', dl: 11, tf: { interest: 1, rate: 1, exposure: 0 } },
];

const AVGDL = CORPUS.reduce((s, d) => s + d.dl, 0) / CORPUS.length;
const N = CORPUS.length;
// document frequency per query term (presence across the corpus)
const DF: Record<string, number> = Object.fromEntries(
  QUERY.map((t) => [t, CORPUS.filter((d) => (d.tf[t] ?? 0) > 0).length]),
);
// BM25 IDF with the +0.5 continuity (Jeffreys) correction, floored at 0.
const idf = (t: string) => Math.max(0, Math.log((N - DF[t] + 0.5) / (DF[t] + 0.5) + 1));

// the BM25 tf-factor for a single term
const tfFactor = (tf: number, k1: number, b: number, dl: number) =>
  (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / AVGDL));

const bm25 = (doc: Doc, k1: number, b: number) =>
  QUERY.reduce((s, t) => s + idf(t) * tfFactor(doc.tf[t] ?? 0, k1, b, doc.dl), 0);

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem' }}>
      <span style={{ minWidth: '5.5rem' }}>{label} = <strong>{value.toFixed(2)}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
    </label>
  );
}

export default function BM25ScoringLaboratory() {
  const [k1, setK1] = useState(1.5);
  const [b, setB] = useState(0.75);
  const formulaRef = useRef<HTMLDivElement>(null);
  const curveRef = useRef<SVGSVGElement>(null);
  const lenRef = useRef<SVGSVGElement>(null);

  // live KaTeX formula with the current k1, b substituted
  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(
      `\\text{score}(q,d)=\\sum_{t\\in q}\\text{idf}(t)\\cdot\\frac{\\text{tf}\\,(${(k1 + 1).toFixed(2)})}{\\text{tf}+${k1.toFixed(2)}\\,(1-${b.toFixed(2)}+${b.toFixed(2)}\\,\\frac{\\text{dl}}{\\text{avgdl}})}`,
      formulaRef.current,
      { throwOnError: false, displayMode: true },
    );
  }, [k1, b]);

  // Panel A — tf saturation curve (fix dl = avgdl so normalization = 1)
  useEffect(() => {
    const svg = d3.select(curveRef.current);
    svg.selectAll('*').remove();
    const W = 360, H = 240, m = { t: 16, r: 16, b: 36, l: 40 };
    const x = d3.scaleLinear().domain([0, 30]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 6]).range([H - m.b, m.t]);
    const g = svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
    g.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6));
    g.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
    g.append('text').attr('x', (W) / 2).attr('y', H - 4).attr('text-anchor', 'middle').attr('font-size', 11).text('term frequency  tf');

    const xs = d3.range(0, 30.1, 0.25);
    const line = (f: (t: number) => number, color: string, dash?: string) =>
      g.append('path').datum(xs).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2)
        .attr('stroke-dasharray', dash ?? null)
        .attr('d', d3.line<number>().x((t) => x(t)).y((t) => y(Math.min(6, f(t))))(xs) as string);

    // raw tf (the k1->inf limit) and log(1+tf) references
    line((t) => t, 'var(--color-muted-border)', '4 3');
    line((t) => Math.log(1 + t), 'var(--color-text-secondary)', '2 3');
    // the BM25 tf-factor at the current k1 (dl = avgdl)
    line((t) => tfFactor(t, k1, b, AVGDL), 'var(--color-accent)');
    // (k1+1) asymptote
    g.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(k1 + 1)).attr('y2', y(k1 + 1))
      .attr('stroke', 'var(--color-accent-secondary)').attr('stroke-width', 1).attr('stroke-dasharray', '2 2');
    g.append('text').attr('x', W - m.r).attr('y', y(k1 + 1) - 4).attr('text-anchor', 'end').attr('font-size', 10)
      .attr('fill', 'var(--color-accent-secondary)').text(`asymptote k₁+1 = ${(k1 + 1).toFixed(2)}`);
  }, [k1, b]);

  // Panel B — length normalization B(dl) per document
  useEffect(() => {
    const svg = d3.select(lenRef.current);
    svg.selectAll('*').remove();
    const W = 360, H = 240, m = { t: 16, r: 16, b: 64, l: 40 };
    const docs = [...CORPUS].sort((a, c) => a.dl - c.dl);
    const x = d3.scaleBand<string>().domain(docs.map((d) => d.id)).range([m.l, W - m.r]).padding(0.25);
    const norm = (dl: number) => 1 - b + (b * dl) / AVGDL;
    const y = d3.scaleLinear().domain([0, Math.max(2, d3.max(docs, (d) => norm(d.dl)) ?? 2)]).range([H - m.b, m.t]);
    const g = svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
    g.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4));
    g.append('g').attr('transform', `translate(0,${y(1)})`).append('line')
      .attr('x1', m.l).attr('x2', W - m.r).attr('stroke', 'var(--color-muted-border)').attr('stroke-dasharray', '2 2');
    g.selectAll('rect').data(docs).join('rect')
      .attr('x', (d) => x(d.id) ?? 0).attr('width', x.bandwidth())
      .attr('y', (d) => Math.min(y(norm(d.dl)), y(1)))
      .attr('height', (d) => Math.abs(y(norm(d.dl)) - y(1)))
      .attr('fill', (d) => (norm(d.dl) >= 1 ? 'var(--color-accent-secondary)' : 'var(--color-accent)'))
      .attr('opacity', 0.8);
    g.selectAll('text.dl').data(docs).join('text').attr('class', 'dl')
      .attr('x', (d) => (x(d.id) ?? 0) + x.bandwidth() / 2).attr('y', H - m.b + 14)
      .attr('text-anchor', 'middle').attr('font-size', 9).text((d) => `${d.dl}`);
    g.append('text').attr('x', W / 2).attr('y', H - 4).attr('text-anchor', 'middle').attr('font-size', 11)
      .text(`document length  dl   (avgdl = ${AVGDL.toFixed(0)})`);
  }, [b]);

  // Panel C — live re-ranking
  const ranked = useMemo(
    () => CORPUS.map((d) => ({ ...d, score: bm25(d, k1, b) })).sort((a, c) => c.score - a.score),
    [k1, b],
  );
  const maxScore = Math.max(...ranked.map((d) => d.score), 1e-9);

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.75rem' }} />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1rem' }}>
        <Slider label="k₁" value={k1} min={0} max={3} step={0.05} onChange={setK1} />
        <Slider label="b" value={b} min={0} max={1} step={0.05} onChange={setB} />
      </div>
      <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
        <figure style={{ margin: 0 }}>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
            A. tf saturation (dl = avgdl) — solid: BM25, dashed: raw tf, dotted: log(1+tf)
          </figcaption>
          <svg ref={curveRef} role="img" aria-label="BM25 term-frequency saturation curve" />
        </figure>
        <figure style={{ margin: 0 }}>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
            B. length normalization B(dl) = 1 − b + b·dl/avgdl
          </figcaption>
          <svg ref={lenRef} role="img" aria-label="BM25 length-normalization factor per document" />
        </figure>
      </div>
      <div style={{ marginTop: '1rem' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
          C. live ranking for query <em>“interest rate exposure”</em> — drag b to 0 to watch the padded transcript hijack the top spot
        </div>
        <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          {ranked.map((d, i) => (
            <li key={d.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem', transition: 'all 0.4s' }}>
              <span style={{ minWidth: '1.25rem', color: 'var(--color-text-secondary)' }}>{i + 1}.</span>
              <span style={{ minWidth: '15rem' }}>{d.label}</span>
              <span style={{ flex: 1, height: '0.75rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
                <span style={{ display: 'block', height: '100%', width: `${(d.score / maxScore) * 100}%`, background: 'var(--color-accent)', transition: 'width 0.4s' }} />
              </span>
              <span style={{ minWidth: '3rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{d.score.toFixed(2)}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
