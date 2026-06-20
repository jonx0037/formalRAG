import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * MIPS Hardness Laboratory — two panels for the topic's thesis that exact
 * high-dimensional search is hopeless, so we approximate.
 *   - Pruning (primary): a dimension slider drives the fraction of candidates a
 *     triangle-inequality pivot filter cannot prune. As d grows, distances
 *     concentrate and the inspected fraction climbs toward 1 — exact search
 *     degenerates into a linear scan.
 *   - Lifting: the norm-equalizing transform that reduces MIPS to Euclidean NN
 *     preserves the argmax (A wins under both the inner product and the lifted
 *     distance) but distorts approximation ratios (B is 1.06x on MIPS, 1.73x on
 *     lifted distance).
 *
 * Fully deterministic. The PRUNING grid and the LIFT readouts are mirrored TO THE
 * DECIMAL from notebooks/mips-hardness-and-sublinearity-limits/mips_hardness_and_sublinearity_limits.py:
 * grid_table() prints the inspected fractions (test_pruning_collapses asserts the
 * climb), and lifting_distortion() prints q=[1,0], A=[0.90,0.20], B=[0.85,0],
 * M=0.92195, <q,A>=0.90, <q,B>=0.85, lifted distances 0.22361 / 0.38730, and the
 * factors 1.0588 / 1.7321 (test_lifting_distorts_approximation_ratio asserts they
 * straddle c=1.3). Change a number here -> change it there, and re-run the notebook.
 */

type Panel = 'pruning' | 'lifting';

// grid_table() in mips_hardness_and_sublinearity_limits.py.
const PRUNING: { d: number; frac: number }[] = [
  { d: 2, frac: 0.0014 },
  { d: 4, frac: 0.0277 },
  { d: 6, frac: 0.2335 },
  { d: 8, frac: 0.4772 },
  { d: 12, frac: 0.7977 },
  { d: 16, frac: 0.9024 },
  { d: 24, frac: 0.9961 },
  { d: 32, frac: 0.9977 },
  { d: 64, frac: 1.0 },
  { d: 128, frac: 1.0 },
];

// lifting_distortion(): the explicit approximation-ratio counterexample.
const LIFT = {
  q: [1, 0] as const,
  A: [0.9, 0.2] as const,   // MIPS winner / lifted Euclidean NN
  B: [0.85, 0] as const,    // runner-up that distorts
  sA: 0.9, sB: 0.85,        // inner products <q, .>
  distA: 0.22361, distB: 0.38730, // lifted Euclidean distances
  mipsFactor: 1.0588, nnFactor: 1.7321,
};

const PLOT_W = 360, PLOT_H = 210, PAD = 30;

const PANELS: { id: Panel; label: string }[] = [
  { id: 'pruning', label: 'Pruning collapse' },
  { id: 'lifting', label: 'Lifting transform' },
];

export default function MIPSHardnessLaboratory() {
  const [panel, setPanel] = useState<Panel>('pruning');
  const [idx, setIdx] = useState(3); // start at d = 8, mid-collapse
  const formulaRef = useRef<HTMLDivElement>(null);
  const row = PRUNING[idx];

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex = panel === 'pruning'
      ? `\\text{prune } x \\iff |D(q,p) - D(p,x)| > r \\quad\\xrightarrow{\\;d\\to\\infty\\;}\\quad \\text{inspected fraction}\\to 1`
      : `\\lVert \\tilde q - \\tilde p\\rVert^2 = \\lVert q\\rVert^2 + M^2 - 2\\langle q, p\\rangle \\;\\;\\Rightarrow\\;\\; \\arg\\max\\langle q, p\\rangle = \\arg\\min \\lVert \\tilde q - \\tilde p\\rVert`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  // Pruning curve: inspected fraction vs dimension (log-spaced x).
  const curve = useMemo(() => {
    const xs = PRUNING.map((r) => Math.log2(r.d));
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const px = (lx: number) => PAD + ((PLOT_W - 2 * PAD) * (lx - xmin)) / (xmax - xmin);
    const py = (f: number) => PAD + (PLOT_H - 2 * PAD) * (1 - f);
    const path = PRUNING.map((r, i) => (i === 0 ? 'M' : 'L') + px(Math.log2(r.d)).toFixed(1) + ' ' + py(r.frac).toFixed(1)).join(' ');
    return { px, py, path, xmin, xmax };
  }, []);

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        {PANELS.map((p) => (
          <button key={p.id} onClick={() => setPanel(p.id)}
            style={{
              fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
              border: `1px solid ${panel === p.id ? 'var(--color-accent)' : 'var(--color-border)'}`,
              background: panel === p.id ? 'var(--color-accent)' : 'transparent',
              color: panel === p.id ? 'var(--color-bg)' : 'var(--color-text)',
            }}>
            {p.label}
          </button>
        ))}
      </div>

      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.75rem' }} />

      {panel === 'pruning' && (
        <>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', marginBottom: '1rem' }}>
            <span style={{ minWidth: '7rem' }}>dimension d = <strong>{row.d}</strong></span>
            <input type="range" min={0} max={PRUNING.length - 1} step={1} value={idx}
              onChange={(e) => setIdx(parseInt(e.target.value, 10))}
              style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
          </label>

          <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}
            role="img" aria-label="Inspected fraction versus dimension">
            <line x1={PAD} y1={curve.py(0)} x2={PLOT_W - PAD} y2={curve.py(0)} stroke="var(--color-border)" strokeWidth={1} />
            <line x1={PAD} y1={curve.py(1)} x2={PLOT_W - PAD} y2={curve.py(1)} stroke="var(--color-muted-border)" strokeWidth={1} strokeDasharray="2 3" />
            <text x={PAD - 4} y={curve.py(1) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">1.0</text>
            <text x={PAD - 4} y={curve.py(0) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">0</text>
            <path d={curve.path} fill="none" stroke="var(--color-accent)" strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" />
            {/* marker at current d */}
            <line x1={curve.px(Math.log2(row.d))} y1={PAD - 4} x2={curve.px(Math.log2(row.d))} y2={curve.py(0)} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="3 3" />
            <circle cx={curve.px(Math.log2(row.d))} cy={curve.py(row.frac)} r={4} fill="var(--color-accent)" />
            <text x={PLOT_W / 2} y={PLOT_H - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">dimension d (log scale)</text>
          </svg>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.75rem' }}>
            <Readout label={`inspected fraction at d = ${row.d}`} value={`${(row.frac * 100).toFixed(1)}%`} accent={row.frac > 0.9} />
            <Readout label="pruned by the pivot filter" value={`${((1 - row.frac) * 100).toFixed(1)}%`} />
            <Readout label="exact search is..." value={row.frac > 0.9 ? 'a linear scan' : 'still pruning'} accent={row.frac > 0.9} />
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem' }}>
            As the dimension grows, distance concentration erases the triangle-inequality certificates, so a metric index must inspect essentially every candidate — the geometric face of the hardness.
          </div>
        </>
      )}

      {panel === 'lifting' && (
        <>
          <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
            <ComparisonBar
              title="inner product ⟨q, ·⟩  (larger = better)"
              rows={[
                { name: 'A (winner)', val: LIFT.sA, good: 1, star: true },
                { name: 'B (runner-up)', val: LIFT.sB, good: LIFT.sB / LIFT.sA },
              ]}
              fmt={(v) => v.toFixed(3)}
            />
            <ComparisonBar
              title="lifted distance ‖q̃ − ·̃‖  (smaller = better)"
              rows={[
                { name: 'A (winner)', val: LIFT.distA, good: 1, star: true },
                { name: 'B (runner-up)', val: LIFT.distB, good: LIFT.distA / LIFT.distB },
              ]}
              fmt={(v) => v.toFixed(3)}
            />
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.9rem' }}>
            <Readout label="winner (both spaces)" value="A — argmax preserved" accent />
            <Readout label="B's MIPS factor" value={`${LIFT.mipsFactor.toFixed(4)}×`} />
            <Readout label="B's lifted-distance factor" value={`${LIFT.nnFactor.toFixed(4)}×`} accent />
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem' }}>
            The lift preserves the exact winner A, but B is only 1.06× behind on the inner product while 1.73× behind on lifted distance: a 1.3-approximate MIPS solution need not be a 1.3-approximate Euclidean neighbor. The additive constant ‖q‖²+M² does not survive a multiplicative ratio.
          </div>
        </>
      )}
    </div>
  );
}

function ComparisonBar({ title, rows, fmt }: {
  title: string;
  rows: { name: string; val: number; good: number; star?: boolean }[];
  fmt: (v: number) => string;
}) {
  return (
    <div>
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.76rem', marginBottom: '0.5rem', color: 'var(--color-text)' }}>{title}</div>
      {rows.map((r) => (
        <div key={r.name} style={{ marginBottom: '0.6rem', fontFamily: 'var(--font-sans)', fontSize: '0.74rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <span style={{ color: r.star ? 'var(--color-accent)' : 'var(--color-text)' }}>{r.name}</span>
            <span style={{ fontVariantNumeric: 'tabular-nums' }}>{fmt(r.val)}</span>
          </div>
          <div style={{ height: '0.7rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
            <div style={{ width: `${Math.max(3, r.good * 100)}%`, height: '100%', background: r.star ? 'var(--color-accent)' : 'var(--color-text-secondary)', borderRadius: '999px', transition: 'width 0.3s' }} />
          </div>
        </div>
      ))}
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
