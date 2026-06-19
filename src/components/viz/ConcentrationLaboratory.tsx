import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Concentration Laboratory — a dimension slider that drives four panels, each
 * dramatizing one high-dimensional concentration phenomenon from the topic:
 *   - Distances: the query-to-point distance density narrows and the relative
 *     contrast (D_max - D_min)/D_min collapses toward 0 (Beyer et al. 1999).
 *   - Angles: the inner-product density of two random unit vectors collapses to a
 *     spike at 0 (near-orthogonality), with Var<u,v> tracking 1/d.
 *   - Thin shell: ||x||/sqrt(d) concentrates at 1, and the unit ball's mass flees
 *     into a vanishingly thin outer shell.
 *   - Intrinsic dimension: at the production embedding dimension d=1536, structured
 *     (low-intrinsic-dim) data keeps its contrast where i.i.d. data destroys it.
 *
 * Fully deterministic. The GRID and FINANCE readouts below are mirrored TO THE
 * DECIMAL from notebooks/high-dimensional-geometry/high_dimensional_geometry.py —
 * its grid_table() prints these contrast / Var<u,v> / shell_std / shell-mass
 * numbers and finance_demo() prints the d=1536 structured-vs-i.i.d. figures.
 * test_distance_concentration / test_near_orthogonality / test_norm_concentration
 * / test_volume_concentration / test_intrinsic_dimension assert them. Change a
 * number here -> change it there, and re-run the notebook. The density curves are
 * the closed-form laws the notebook samples (chi for norms/distances, the
 * (1 - t^2)^((d-3)/2) projection law for inner products), drawn in log-space so
 * they auto-scale without a Gamma function.
 */

type Row = {
  d: number;
  contrast: number;   // mean (D_max - D_min)/D_min, query-to-point, Gaussian data
  ipVar: number;      // empirical Var<u, v>, ~ 1/d
  shellStd: number;   // std of ||x|| / sqrt(d)
  shellFrac10: number;// fraction of the unit ball within 10% of the surface
};

// grid_table() in high_dimensional_geometry.py (i.i.d. Gaussian, fixed seeds).
const GRID: Row[] = [
  { d: 1,    contrast: 1225.0917, ipVar: 0.99960, shellStd: 0.6026, shellFrac10: 0.1000 },
  { d: 2,    contrast: 47.0984,   ipVar: 0.50414, shellStd: 0.4619, shellFrac10: 0.1900 },
  { d: 3,    contrast: 13.8922,   ipVar: 0.33244, shellStd: 0.3946, shellFrac10: 0.2710 },
  { d: 5,    contrast: 4.9086,    ipVar: 0.20211, shellStd: 0.3064, shellFrac10: 0.4095 },
  { d: 10,   contrast: 2.3012,    ipVar: 0.10193, shellStd: 0.2235, shellFrac10: 0.6513 },
  { d: 20,   contrast: 1.2498,    ipVar: 0.05122, shellStd: 0.1577, shellFrac10: 0.8784 },
  { d: 50,   contrast: 0.6236,    ipVar: 0.02042, shellStd: 0.1007, shellFrac10: 0.9948 },
  { d: 100,  contrast: 0.4015,    ipVar: 0.00998, shellStd: 0.0719, shellFrac10: 1.0000 },
  { d: 200,  contrast: 0.2741,    ipVar: 0.00505, shellStd: 0.0502, shellFrac10: 1.0000 },
  { d: 500,  contrast: 0.1651,    ipVar: 0.00192, shellStd: 0.0318, shellFrac10: 1.0000 },
  { d: 1000, contrast: 0.1129,    ipVar: 0.00099, shellStd: 0.0223, shellFrac10: 1.0000 },
];

// finance_demo(): structured (intrinsic k=10) vs i.i.d. in R^1536.
const FINANCE = {
  dim: 1536, k: 10,
  structContrast: 2.183, structTwoNN: 9.58,
  iidContrast: 0.088, iidTwoNN: 208.26,
};

type Panel = 'distance' | 'angle' | 'shell' | 'intrinsic';

const PLOT_W = 540;
const PLOT_H = 190;
const PAD = 26;

// Sample a log-density over [xmin, xmax] and return an SVG path, normalized to the
// plot height by subtracting the max log-value (no normalizing constant needed).
function densityPath(logf: (x: number) => number, xmin: number, xmax: number): string {
  const npts = 180;
  const xs: number[] = [];
  const lg: number[] = [];
  for (let i = 0; i < npts; i++) {
    const x = xmin + ((xmax - xmin) * i) / (npts - 1);
    xs.push(x);
    lg.push(logf(x));
  }
  let mx = -Infinity;
  for (const v of lg) if (Number.isFinite(v) && v > mx) mx = v;
  const plotW = PLOT_W - 2 * PAD;
  const plotH = PLOT_H - 2 * PAD;
  let path = '';
  for (let i = 0; i < npts; i++) {
    // +Infinity is a true pole (e.g. the d<3 angle law) -> pin to the top; -Infinity
    // is a true zero (the density vanishes at the boundary) -> pin to the bottom.
    let y = Number.isFinite(lg[i]) ? Math.exp(lg[i] - mx) : (lg[i] === Infinity ? 1 : 0);
    y = Math.min(1, Math.max(0, y));
    const px = PAD + (plotW * (xs[i] - xmin)) / (xmax - xmin);
    const py = PAD + plotH * (1 - y);
    path += (i === 0 ? 'M' : 'L') + px.toFixed(1) + ' ' + py.toFixed(1) + ' ';
  }
  return path;
}

// Closed-form log-densities (up to a constant) for the current panel.
const logChiNormalized = (d: number) => (u: number) =>
  u <= 0 ? -Infinity : (d - 1) * Math.log(u) - (d * u * u) / 2; // ||x||/sqrt(d) and D/mean share this shape
const logProjection = (d: number) => (t: number) =>
  // <u, v> for v uniform on the sphere; let the arithmetic carry the boundary sign —
  // (d-3)/2 < 0 gives +Infinity (a pole, d<3), > 0 gives -Infinity (a zero, d>3).
  t <= -1 || t >= 1 ? ((d - 3) / 2) * -Infinity : ((d - 3) / 2) * Math.log(1 - t * t);

function Plot({ paths, xmin, xmax, ticks, markerX }: {
  paths: { d: string; color: string; width: number }[];
  xmin: number; xmax: number; ticks: { x: number; label: string }[]; markerX?: number;
}) {
  const px = (x: number) => PAD + ((PLOT_W - 2 * PAD) * (x - xmin)) / (xmax - xmin);
  return (
    <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
      {markerX !== undefined && (
        <line x1={px(markerX)} y1={PAD - 6} x2={px(markerX)} y2={PLOT_H - PAD}
          stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="3 3" />
      )}
      {ticks.map((t) => (
        <text key={t.label} x={px(t.x)} y={PLOT_H - PAD + 14} textAnchor="middle"
          fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t.label}</text>
      ))}
      {paths.map((p, i) => (
        <path key={i} d={p.d} fill="none" stroke={p.color} strokeWidth={p.width}
          strokeLinejoin="round" strokeLinecap="round" opacity={p.width < 2 ? 0.45 : 1} />
      ))}
    </svg>
  );
}

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

const PANELS: { id: Panel; label: string }[] = [
  { id: 'distance', label: 'Distances' },
  { id: 'angle', label: 'Angles' },
  { id: 'shell', label: 'Thin shell' },
  { id: 'intrinsic', label: 'Intrinsic dimension' },
];

export default function ConcentrationLaboratory() {
  const [panel, setPanel] = useState<Panel>('distance');
  const [idx, setIdx] = useState(4); // GRID index; start at d = 10
  const row = GRID[idx];
  const d = row.d;
  const formulaRef = useRef<HTMLDivElement>(null);

  // Live KaTeX of the law the active panel illustrates.
  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'distance'
        ? `\\frac{\\operatorname{Var}(D^2)}{\\mathbb{E}[D^2]^2}=\\frac{2}{d}\\xrightarrow{d\\to\\infty}0,\\qquad \\frac{D_{\\max}-D_{\\min}}{D_{\\min}}\\to 0`
        : panel === 'angle'
        ? `\\mathbb{E}\\langle u,v\\rangle=0,\\qquad \\operatorname{Var}\\langle u,v\\rangle=\\tfrac{1}{d}`
        : panel === 'shell'
        ? `\\frac{\\lVert x\\rVert}{\\sqrt d}\\to 1,\\qquad \\text{mass within }\\varepsilon\\text{ of the surface}=1-(1-\\varepsilon)^d`
        : `\\text{contrast is governed by the intrinsic dimension }k,\\text{ not the ambient }d`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel, d]);

  const plot = useMemo(() => {
    if (panel === 'angle') {
      return (
        <Plot xmin={-1} xmax={1} markerX={0}
          ticks={[{ x: -1, label: '−1' }, { x: 0, label: '0' }, { x: 1, label: '+1' }]}
          paths={[
            { d: densityPath(logProjection(2), -1, 1), color: 'var(--color-text-secondary)', width: 1.4 },
            { d: densityPath(logProjection(d), -1, 1), color: 'var(--color-accent)', width: 2.4 },
          ]} />
      );
    }
    // distance and shell share the chi shape on a normalized axis centered at 1.
    return (
      <Plot xmin={0} xmax={2} markerX={1}
        ticks={[{ x: 0, label: '0' }, { x: 1, label: '1' }, { x: 2, label: '2' }]}
        paths={[
          { d: densityPath(logChiNormalized(2), 0, 2), color: 'var(--color-text-secondary)', width: 1.4 },
          { d: densityPath(logChiNormalized(d), 0, 2), color: 'var(--color-accent)', width: 2.4 },
        ]} />
    );
  }, [panel, d]);

  const maxContrast = FINANCE.structContrast;

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      {/* panel toggle */}
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

      {/* dimension slider (drives every panel except the fixed-d intrinsic view) */}
      {panel !== 'intrinsic' && (
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', marginBottom: '1rem' }}>
          <span style={{ minWidth: '7rem' }}>dimension d = <strong>{d}</strong></span>
          <input type="range" min={0} max={GRID.length - 1} step={1} value={idx}
            onChange={(e) => setIdx(parseInt(e.target.value, 10))}
            style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
        </label>
      )}

      {panel === 'distance' && (
        <>
          {plot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Query-to-point distance, as a fraction of its mean (faint curve: d = 2). As d grows the
            distribution collapses onto its mean — every point is the same distance away.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`relative contrast at d = ${d}`} value={row.contrast.toFixed(row.contrast >= 10 ? 1 : 4)} accent={row.contrast < 0.3} />
            <Readout label="Var(D²)/E[D²]² = 2/d" value={(2 / d).toFixed(5)} />
            <Readout label="nearest ≈ farthest?" value={row.contrast < 0.3 ? 'yes — NN loses meaning' : 'no'} accent={row.contrast < 0.3} />
          </div>
        </>
      )}

      {panel === 'angle' && (
        <>
          {plot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Inner product of two random unit vectors (faint curve: d = 2, the U-shaped arcsine law).
            As d grows it spikes at 0 — random directions are almost always orthogonal.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`Var⟨u, v⟩ at d = ${d}`} value={row.ipVar.toFixed(5)} accent={d >= 100} />
            <Readout label="theory: 1/d" value={(1 / d).toFixed(5)} />
            <Readout label="typical angle" value={`${(90 - (180 / Math.PI) * Math.asin(Math.min(1, Math.sqrt(row.ipVar)))).toFixed(1)}–${(90 + (180 / Math.PI) * Math.asin(Math.min(1, Math.sqrt(row.ipVar)))).toFixed(1)}°`} />
          </div>
        </>
      )}

      {panel === 'shell' && (
        <>
          {plot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            The norm ||x|| / √d of a Gaussian vector (faint curve: d = 2). It concentrates on a thin
            shell at radius √d; meanwhile the ball's volume flees to its surface.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`shell width std(||x||/√d) at d = ${d}`} value={row.shellStd.toFixed(4)} accent={row.shellStd < 0.1} />
            <Readout label="mass in the outer 10% shell" value={`${(row.shellFrac10 * 100).toFixed(1)}%`} accent={row.shellFrac10 > 0.95} />
          </div>
        </>
      )}

      {panel === 'intrinsic' && (
        <div style={{ fontFamily: 'var(--font-sans)' }}>
          <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', marginBottom: '0.9rem' }}>
            At the production embedding dimension d = {FINANCE.dim}: structured data with intrinsic
            dimension k = {FINANCE.k} keeps its contrast (nearest neighbor is meaningful); i.i.d. data
            filling R<sup>{FINANCE.dim}</sup> loses it. The curse is governed by intrinsic, not ambient, dimension.
          </div>
          {[
            { name: `structured (intrinsic k = ${FINANCE.k})`, c: FINANCE.structContrast, twonn: FINANCE.structTwoNN, color: 'var(--color-accent)' },
            { name: 'i.i.d. (fills R¹⁵³⁶)', c: FINANCE.iidContrast, twonn: FINANCE.iidTwoNN, color: 'var(--color-text-secondary)' },
          ].map((b) => (
            <div key={b.name} style={{ marginBottom: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', marginBottom: '0.2rem' }}>
                <span>{b.name}</span>
                <span style={{ color: 'var(--color-text-secondary)' }}>contrast <strong style={{ color: 'var(--color-text)' }}>{b.c.toFixed(3)}</strong> · TwoNN dim <strong style={{ color: 'var(--color-text)' }}>{b.twonn.toFixed(1)}</strong></span>
              </div>
              <div style={{ height: '0.8rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
                <div style={{ width: `${Math.max(2, (b.c / maxContrast) * 100)}%`, height: '100%', background: b.color, borderRadius: '999px', transition: 'width 0.3s' }} />
              </div>
            </div>
          ))}
          <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem' }}>
            TwoNN recovers k ≈ {FINANCE.structTwoNN.toFixed(0)} for the structured set; for i.i.d. data it reads far higher
            (the estimator saturates below the ambient {FINANCE.dim} at finite sample size, but stays {(FINANCE.iidTwoNN / FINANCE.structTwoNN).toFixed(0)}× larger).
          </div>
        </div>
      )}
    </div>
  );
}
