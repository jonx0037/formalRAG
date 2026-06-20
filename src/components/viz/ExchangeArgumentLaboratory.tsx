import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import katex from 'katex';

/**
 * Exchange-Argument Laboratory — the Probability Ranking Principle, made tactile.
 * The reader reorders five documents (each with a probability of relevance) with
 * up/down buttons, and watches the cumulative expected-relevant curve race toward
 * the PRP-optimal envelope. A single "one adjacent swap" button performs one
 * bubble-sort step of the exchange argument: it swaps the first out-of-order
 * adjacent pair, which can only raise the curve, never lower it.
 *
 * The payload: the order sorted by decreasing P(R) dominates every other order's
 * expected-relevant curve at EVERY cutoff at once, and is reachable by adjacent
 * swaps that never hurt.
 *
 * Deterministic; numbers mirror probability_ranking_principle.py viz_constants()
 * (P(R) = [0.82, 0.61, 0.55, 0.30, 0.12]; the scrambled start (2,4,0,3,1) has
 * cumulative [0.55, 0.67, 1.49, 1.79, 2.40] and 6 inversions). SVG text inherits
 * the theme color via the global `svg text { fill: var(--color-text) }` rule.
 */

// --- Documents: label + probability of relevance to "interest rate exposure" --
// Labels reuse the BM25 corpus names; P(R) values mirror _P in the notebook.
type Doc = { id: string; label: string; p: number };
const DOCS: Doc[] = [
  { id: 'filing-onpoint', label: '10-K · net interest margin sensitivity', p: 0.82 },
  { id: 'news-macro', label: 'News · Fed rate decision', p: 0.61 },
  { id: 'transcript-pad', label: 'Earnings call · long Q&A (padded)', p: 0.55 },
  { id: 'filing-fx', label: '10-K · foreign-exchange risk', p: 0.30 },
  { id: 'transcript-short', label: 'Earnings call · brief update', p: 0.12 },
];
const SCRAMBLED = [2, 4, 0, 3, 1]; // the default starting order (matches viz_constants)
const N = DOCS.length;
const PRP_ORDER = DOCS.map((_, i) => i).sort((a, b) => DOCS[b].p - DOCS[a].p);

const cumulative = (order: number[]) => {
  const out: number[] = [];
  let s = 0;
  for (const d of order) { s += DOCS[d].p; out.push(s); }
  return out;
};
const inversions = (order: number[]) => {
  let n = 0;
  for (let a = 0; a < order.length; a++)
    for (let b = a + 1; b < order.length; b++)
      if (DOCS[order[a]].p < DOCS[order[b]].p) n++;
  return n;
};
const firstOutOfOrderPair = (order: number[]) => {
  for (let j = 0; j < order.length - 1; j++)
    if (DOCS[order[j]].p < DOCS[order[j + 1]].p) return j;
  return -1;
};

export default function ExchangeArgumentLaboratory() {
  const [order, setOrder] = useState<number[]>(SCRAMBLED);
  const chartRef = useRef<SVGSVGElement>(null);
  const formulaRef = useRef<HTMLDivElement>(null);

  const cur = useMemo(() => cumulative(order), [order]);
  const opt = useMemo(() => cumulative(PRP_ORDER), []);
  const inv = useMemo(() => inversions(order), [order]);
  const swapJ = useMemo(() => firstOutOfOrderPair(order), [order]);
  const isOptimal = inv === 0;

  const move = (pos: number, dir: -1 | 1) => {
    const t = pos + dir;
    if (t < 0 || t >= N) return;
    const next = [...order];
    [next[pos], next[t]] = [next[t], next[pos]];
    setOrder(next);
  };
  const oneSwap = () => {
    if (swapJ < 0) return;
    const next = [...order];
    [next[swapJ], next[swapJ + 1]] = [next[swapJ + 1], next[swapJ]];
    setOrder(next);
  };

  // live KaTeX: the prefix-sum identity the chart plots
  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(
      `\\mathbb{E}[\\#\\text{rel} \\,\\text{in top-}k] = \\sum_{i=1}^{k} p_{\\pi(i)}`,
      formulaRef.current, { throwOnError: false, displayMode: true },
    );
  }, []);

  // Panel: cumulative expected-relevant curve — current vs PRP-optimal envelope
  useEffect(() => {
    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();
    const W = 420, H = 260, m = { t: 16, r: 16, b: 40, l: 40 };
    const x = d3.scaleLinear().domain([1, N]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, Math.ceil(opt[N - 1])]).range([H - m.b, m.t]);
    const g = svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
    g.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(N).tickFormat(d3.format('d')));
    g.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
    g.append('text').attr('x', W / 2).attr('y', H - 4).attr('text-anchor', 'middle').attr('font-size', 11).text('cutoff  k');

    const ks = d3.range(1, N + 1);
    const lineGen = (data: number[]) =>
      d3.line<number>().x((_, i) => x(i + 1)).y((v) => y(v))(data) as string;

    // PRP-optimal envelope (dashed) — the upper bound the current order can only touch
    g.append('path').datum(opt).attr('fill', 'none').attr('stroke', 'var(--color-accent-secondary)')
      .attr('stroke-width', 2).attr('stroke-dasharray', '5 3').attr('d', lineGen(opt));
    g.selectAll('circle.opt').data(opt).join('circle').attr('class', 'opt')
      .attr('cx', (_, i) => x(i + 1)).attr('cy', (v) => y(v)).attr('r', 2.5)
      .attr('fill', 'var(--color-accent-secondary)');

    // current order (solid)
    g.append('path').datum(cur).attr('fill', 'none').attr('stroke', 'var(--color-accent)')
      .attr('stroke-width', 2.5).attr('d', lineGen(cur));
    g.selectAll('circle.cur').data(cur).join('circle').attr('class', 'cur')
      .attr('cx', (_, i) => x(i + 1)).attr('cy', (v) => y(v)).attr('r', 3.5)
      .attr('fill', 'var(--color-accent)');

    g.append('text').attr('x', W - m.r).attr('y', m.t + 4).attr('text-anchor', 'end').attr('font-size', 10)
      .attr('fill', 'var(--color-accent-secondary)').text('PRP-optimal (envelope)');
  }, [cur, opt]);

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.75rem' }} />
      <div style={{ display: 'grid', gap: '1.25rem', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
        {/* Reorder control */}
        <div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
            Reorder the ranking for <em>“interest rate exposure”</em> — each document's P(R) is fixed.
          </div>
          <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            {order.map((d, pos) => {
              const boundaryOutOfOrder = pos === swapJ;
              return (
                <li key={DOCS[d].id} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem',
                  borderBottom: boundaryOutOfOrder ? '2px solid var(--color-accent-secondary)' : '2px solid transparent', paddingBottom: '0.2rem', transition: 'all 0.3s' }}>
                  <span style={{ minWidth: '1.1rem', color: 'var(--color-text-secondary)' }}>{pos + 1}.</span>
                  <span style={{ display: 'inline-flex', flexDirection: 'column' }}>
                    <button onClick={() => move(pos, -1)} disabled={pos === 0} aria-label="move up"
                      style={{ lineHeight: 1, border: 'none', background: 'none', cursor: pos === 0 ? 'default' : 'pointer', color: pos === 0 ? 'var(--color-muted-border)' : 'var(--color-accent)', fontSize: '0.7rem' }}>▲</button>
                    <button onClick={() => move(pos, 1)} disabled={pos === N - 1} aria-label="move down"
                      style={{ lineHeight: 1, border: 'none', background: 'none', cursor: pos === N - 1 ? 'default' : 'pointer', color: pos === N - 1 ? 'var(--color-muted-border)' : 'var(--color-accent)', fontSize: '0.7rem' }}>▼</button>
                  </span>
                  <span style={{ flex: 1 }}>{DOCS[d].label}</span>
                  <span style={{ fontVariantNumeric: 'tabular-nums', color: 'var(--color-text-secondary)' }}>P(R)={DOCS[d].p.toFixed(2)}</span>
                </li>
              );
            })}
          </ol>
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
            <button onClick={oneSwap} disabled={isOptimal}
              style={{ padding: '0.35rem 0.8rem', borderRadius: '0.4rem', cursor: isOptimal ? 'default' : 'pointer',
                border: '1px solid var(--color-accent)', background: isOptimal ? 'transparent' : 'var(--color-accent)',
                color: isOptimal ? 'var(--color-muted-border)' : 'var(--color-bg)', fontFamily: 'var(--font-sans)', fontSize: '0.78rem' }}>
              one adjacent swap →
            </button>
            <button onClick={() => setOrder(SCRAMBLED)}
              style={{ padding: '0.35rem 0.8rem', borderRadius: '0.4rem', cursor: 'pointer',
                border: '1px solid var(--color-border)', background: 'transparent', color: 'var(--color-text)', fontFamily: 'var(--font-sans)', fontSize: '0.78rem' }}>
              reset
            </button>
          </div>
        </div>

        {/* Chart + readouts */}
        <div>
          <figure style={{ margin: 0 }}>
            <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
              cumulative expected relevant — your order (solid) vs PRP-optimal (dashed envelope)
            </figcaption>
            <svg ref={chartRef} role="img" aria-label="Cumulative expected relevant documents versus cutoff" />
          </figure>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', marginTop: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>E[#rel in top-3], your order</span>
              <strong style={{ fontVariantNumeric: 'tabular-nums' }}>{cur[2].toFixed(2)}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--color-text-secondary)' }}>
              <span>E[#rel in top-3], PRP-optimal</span>
              <span style={{ fontVariantNumeric: 'tabular-nums' }}>{opt[2].toFixed(2)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>inversions remaining</span>
              <strong style={{ fontVariantNumeric: 'tabular-nums' }}>{inv}</strong>
            </div>
            <div style={{ marginTop: '0.3rem', padding: '0.4rem 0.6rem', borderRadius: '0.4rem',
              background: isOptimal ? 'var(--color-definition-bg)' : 'var(--color-muted-bg)',
              borderLeft: `3px solid ${isOptimal ? 'var(--color-definition-border)' : 'var(--color-muted-border)'}`, fontSize: '0.78rem' }}>
              {isOptimal
                ? 'PRP order: optimal at every cutoff at once — the curve sits on the envelope.'
                : swapJ >= 0
                  ? `Rows ${swapJ + 1} and ${swapJ + 2} are out of order (underlined). One swap there raises the curve toward the envelope and never lowers it elsewhere.`
                  : ''}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
