import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Similarity Geometry Laboratory — a 2-D playground for the three similarity
 * scores retrieval uses. A query q and four finance document vectors sit in the
 * plane; a score toggle (Euclidean / dot / cosine) draws the equal-score locus
 * through the slider-controlled document d-star (a hyperplane, a circle, or a
 * cone) and ranks the corpus by that score. The magnitude slider scales d-star
 * along its direction: under the dot product its rank climbs and overtakes the
 * large-norm padded transcript at c* ~ 1.79, while under cosine its rank never
 * moves. The "normalize to the unit circle" toggle snaps every vector to the
 * sphere, where the three rankings coincide.
 *
 * Fully deterministic. The base vectors and the c = 1 readouts below are mirrored
 * TO THE DECIMAL from notebooks/the-retrieval-problem/the_retrieval_problem.py
 * (finance_demo / test_finance_demo_numbers): q = [1, 0]; norms 0.9708, 2.1401,
 * 0.9708, 0.9487; cosines 0.9785, 0.7944, 0.8240, 0.3162; dots 0.95, 1.70, 0.80,
 * 0.30; flip threshold c* = 1.7895. The live dot/cosine/Euclidean values are
 * recomputed from the base vectors with the same formulas the .py asserts. Change
 * a number here -> change it there, and re-run the notebook.
 */

type Vec = [number, number];
type Doc = { id: string; label: string; v: Vec; isStar?: boolean };

const Q: Vec = [1, 0];
const DOCS: Doc[] = [
  { id: 'filing-onpoint', label: '10-K · on-point (concise)', v: [0.95, 0.20], isStar: true },
  { id: 'transcript-pad', label: 'Earnings call · padded (long)', v: [1.70, 1.30] },
  { id: 'news-macro', label: 'News · Fed rate decision', v: [0.80, 0.55] },
  { id: 'filing-fx', label: '10-K · FX risk (off-topic)', v: [0.30, 0.90] },
];
const C_STAR = 1.7895; // dot top-1 flips from transcript-pad to d-star past this magnitude

type Score = 'euclidean' | 'dot' | 'cosine';

const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1];
const norm = (a: Vec) => Math.hypot(a[0], a[1]);
const euclid = (a: Vec, b: Vec) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const cosine = (a: Vec, b: Vec) => (norm(a) < 1e-12 || norm(b) < 1e-12 ? 0 : dot(a, b) / (norm(a) * norm(b)));
const unit = (a: Vec): Vec => { const n = norm(a); return n === 0 ? a : [a[0] / n, a[1] / n]; };
const scale = (a: Vec, c: number): Vec => [a[0] * c, a[1] * c];

// --- plane geometry: equal world scale in x and y so circles stay circular ---
const S = 80;                 // pixels per unit
const XMIN = -1.5, XMAX = 2.5, YMIN = -1.5, YMAX = 1.9;
const PW = (XMAX - XMIN) * S; // 320
const PH = (YMAX - YMIN) * S; // 272
const px = (wx: number) => (wx - XMIN) * S;
const py = (wy: number) => (YMAX - wy) * S;

const SCORES: { id: Score; label: string; locus: string }[] = [
  { id: 'euclidean', label: 'Euclidean', locus: 'sphere' },
  { id: 'dot', label: 'dot product', locus: 'hyperplane' },
  { id: 'cosine', label: 'cosine', locus: 'cone' },
];

export default function SimilarityGeometryLaboratory() {
  const [score, setScore] = useState<Score>('cosine');
  const [c, setC] = useState(1.0);            // magnitude multiplier on d-star
  const [normalized, setNormalized] = useState(false);
  const formulaRef = useRef<HTMLDivElement>(null);

  // Effective document positions: on the sphere when normalized, else d-star scaled by c.
  const docs = useMemo(
    () => DOCS.map((d) => ({
      ...d,
      pos: normalized ? unit(d.v) : (d.isStar ? scale(d.v, c) : d.v),
    })),
    [c, normalized],
  );
  const q: Vec = useMemo(() => (normalized ? unit(Q) : Q), [normalized]);
  const star = docs.find((d) => d.isStar)!;

  // Ranking under the active score (descending for similarities, ascending for distance).
  const ranked = useMemo(() => {
    const f = (d: Vec) => (score === 'dot' ? dot(q, d) : score === 'cosine' ? cosine(q, d) : euclid(q, d));
    const rows = docs.map((d) => ({ ...d, s: f(d.pos) }));
    rows.sort((a, b) => (score === 'euclidean' ? a.s - b.s : b.s - a.s));
    const lo = Math.min(...rows.map((r) => r.s));
    const hi = Math.max(...rows.map((r) => r.s));
    const good = (s: number) => (hi === lo ? 1 : score === 'euclidean' ? (hi - s) / (hi - lo) : (s - lo) / (hi - lo));
    return rows.map((r) => ({ ...r, good: good(r.s) }));
  }, [docs, q, score]);

  // The three rankings coincide on the sphere — flag it when the toggle is on.
  const coincide = useMemo(() => {
    const order = (sc: Score) => {
      const f = (d: Vec) => (sc === 'dot' ? dot(q, d) : sc === 'cosine' ? cosine(q, d) : euclid(q, d));
      return [...docs].sort((a, b) => (sc === 'euclidean' ? f(a.pos) - f(b.pos) : f(b.pos) - f(a.pos)))
        .map((d) => d.id).join(',');
    };
    return order('dot') === order('cosine') && order('cosine') === order('euclidean');
  }, [docs, q]);

  // Live KaTeX of the active score's definition and its level-set shape.
  useEffect(() => {
    if (!formulaRef.current) return;
    const tex = score === 'dot'
      ? `\\langle q, d\\rangle \\;\\;\\text{— equal-score locus: a \\textbf{hyperplane}}`
      : score === 'cosine'
      ? `\\frac{\\langle q, d\\rangle}{\\lVert q\\rVert\\,\\lVert d\\rVert} \\;\\;\\text{— equal-score locus: a \\textbf{cone}}`
      : `\\lVert q - d\\rVert \\;\\;\\text{— equal-score locus: a \\textbf{sphere}}`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [score]);

  // Equal-score locus through d-star, in pixel coordinates.
  const locus = useMemo(() => {
    if (score === 'dot') {
      const k = dot(q, star.pos);                       // with q = [1,0], the line x = k
      const x = q[0] === 0 ? 0 : k / q[0];
      return <line x1={px(x)} y1={py(YMIN)} x2={px(x)} y2={py(YMAX)}
        stroke="var(--color-accent-secondary)" strokeWidth={1.5} strokeDasharray="4 3" />;
    }
    if (score === 'euclidean') {
      const r = euclid(q, star.pos);
      return <circle cx={px(q[0])} cy={py(q[1])} r={r * S} fill="none"
        stroke="var(--color-accent-secondary)" strokeWidth={1.5} strokeDasharray="4 3" />;
    }
    // cosine: two rays from the origin at +/- the angle between q and d-star.
    const ct = Math.max(-1, Math.min(1, cosine(q, star.pos)));
    const base = Math.atan2(q[1], q[0]);
    const th = Math.acos(ct);
    const L = 3.2;
    return (
      <>
        {[base + th, base - th].map((a, i) => (
          <line key={i} x1={px(0)} y1={py(0)}
            x2={px(L * Math.cos(a))} y2={py(L * Math.sin(a))}
            stroke="var(--color-accent-secondary)" strokeWidth={1.5} strokeDasharray="4 3" />
        ))}
      </>
    );
  }, [score, q, star.pos]);

  const starDot = dot(q, star.pos);
  const starCos = cosine(q, star.pos);

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      {/* score toggle */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        {SCORES.map((s) => (
          <button key={s.id} onClick={() => setScore(s.id)}
            style={{
              fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
              border: `1px solid ${score === s.id ? 'var(--color-accent)' : 'var(--color-border)'}`,
              background: score === s.id ? 'var(--color-accent)' : 'transparent',
              color: score === s.id ? 'var(--color-bg)' : 'var(--color-text)',
            }}>
            {s.label}
          </button>
        ))}
      </div>

      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.5rem' }} />

      <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
        {/* Panel A — the plane */}
        <figure style={{ margin: 0 }}>
          <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto', display: 'block' }}
            role="img" aria-label="Query and document vectors in the plane with the active score's level set">
            {/* axes */}
            <line x1={0} y1={py(0)} x2={PW} y2={py(0)} stroke="var(--color-border)" strokeWidth={1} />
            <line x1={px(0)} y1={0} x2={px(0)} y2={PH} stroke="var(--color-border)" strokeWidth={1} />
            {/* unit circle */}
            <circle cx={px(0)} cy={py(0)} r={S} fill="none" stroke="var(--color-muted-border)" strokeWidth={1} strokeDasharray="2 3" />
            {/* equal-score locus through d-star */}
            {locus}
            {/* query vector */}
            <line x1={px(0)} y1={py(0)} x2={px(q[0])} y2={py(q[1])} stroke="var(--color-text)" strokeWidth={2} />
            <circle cx={px(q[0])} cy={py(q[1])} r={4} fill="var(--color-text)" />
            <text x={px(q[0]) + 6} y={py(q[1]) - 6} fontSize={11} fill="var(--color-text)" fontFamily="var(--font-sans)">q</text>
            {/* documents */}
            {docs.map((d) => (
              <g key={d.id}>
                {d.isStar && (
                  <line x1={px(0)} y1={py(0)} x2={px(d.pos[0])} y2={py(d.pos[1])}
                    stroke="var(--color-accent)" strokeWidth={1} opacity={0.5} strokeDasharray="1 3" />
                )}
                <circle cx={px(d.pos[0])} cy={py(d.pos[1])} r={d.isStar ? 5 : 4}
                  fill={d.isStar ? 'var(--color-accent)' : 'var(--color-text-secondary)'} />
                <text x={px(d.pos[0]) + 6} y={py(d.pos[1]) + 3} fontSize={9.5}
                  fill={d.isStar ? 'var(--color-accent)' : 'var(--color-text-secondary)'} fontFamily="var(--font-sans)">
                  {d.id === 'filing-onpoint' ? 'd★' : d.id.split('-')[0]}
                </text>
              </g>
            ))}
          </svg>
          <figcaption style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.25rem' }}>
            Dashed locus: all points scoring the same as d★ under the active score. Faint ring: the unit circle.
          </figcaption>
        </figure>

        {/* Panel B — live ranking */}
        <div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
            ranking by <strong>{SCORES.find((s) => s.id === score)!.label}</strong>
            {coincide && (
              <span style={{ color: 'var(--color-accent)', marginLeft: '0.5rem' }}>· all three coincide ✓</span>
            )}
          </div>
          <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {ranked.map((d, i) => (
              <li key={d.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'var(--font-sans)', fontSize: '0.76rem', transition: 'all 0.4s' }}>
                <span style={{ minWidth: '1.1rem', color: 'var(--color-text-secondary)' }}>{i + 1}.</span>
                <span style={{ minWidth: '11rem', color: d.isStar ? 'var(--color-accent)' : 'var(--color-text)' }}>{d.label}</span>
                <span style={{ flex: 1, height: '0.7rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
                  <span style={{ display: 'block', height: '100%', width: `${Math.max(3, d.good * 100)}%`, background: d.isStar ? 'var(--color-accent)' : 'var(--color-text-secondary)', transition: 'width 0.4s' }} />
                </span>
                <span style={{ minWidth: '3rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{d.s.toFixed(3)}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>

      {/* magnitude slider on d-star */}
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.85rem', marginTop: '1rem', opacity: normalized ? 0.4 : 1 }}>
        <span style={{ minWidth: '8.5rem' }}>‖d★‖ multiplier c = <strong>{c.toFixed(2)}</strong></span>
        <input type="range" min={0.3} max={2.5} step={0.01} value={c} disabled={normalized}
          onChange={(e) => setC(parseFloat(e.target.value))}
          style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
      </label>

      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '1.25rem', marginTop: '0.75rem' }}>
        <Readout label="‖d★‖" value={norm(star.pos).toFixed(3)} />
        <Readout label="⟨q, d★⟩  (grows with c)" value={starDot.toFixed(3)} accent={score === 'dot'} />
        <Readout label="cos(q, d★)  (fixed in c)" value={starCos.toFixed(4)} accent={score === 'cosine'} />
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontFamily: 'var(--font-sans)', fontSize: '0.8rem', cursor: 'pointer' }}>
          <input type="checkbox" checked={normalized} onChange={(e) => setNormalized(e.target.checked)}
            style={{ accentColor: 'var(--color-accent)' }} />
          normalize all to the unit circle
        </label>
      </div>

      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem' }}>
        {normalized
          ? 'On the unit circle the cosine, dot-product, and Euclidean rankings induce the same order.'
          : `Slide c past c* = ${C_STAR} and, under the dot product, d★ overtakes the padded transcript for the top spot — while under cosine its rank never changes.`}
      </div>
    </div>
  );
}

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.0rem', fontWeight: 600, fontVariantNumeric: 'tabular-nums', color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}
