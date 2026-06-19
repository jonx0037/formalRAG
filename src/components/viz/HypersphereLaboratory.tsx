import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Hypersphere Laboratory — four panels on the geometry of the unit sphere S^{d-1},
 * the space dense retrieval actually searches:
 *   - Equator: the uniform coordinate marginal (1 - t^2)^((d-3)/2) collapses to a
 *     spike at t = 0 as d grows, with Var(t) tracking 1/d — the equatorial band.
 *   - von Mises-Fisher: the cluster density e^{kappa t} (1 - t^2)^((d-3)/2) tilts the
 *     equatorial law toward the mean direction; its peak sweeps to rho = A_d(kappa).
 *   - Cosine vs distance: the monotone line ||x-y||^2 = 2 - 2 cos(theta), so ranking
 *     by cosine similarity and by Euclidean distance is the same ranking.
 *   - Clusters: two synthetic vMF clusters at the production dimension d = 1536, one
 *     tight, one loose — cluster tightness is a measurable mean resultant length.
 *
 * Fully deterministic. The EQUATOR, VMF, and FINANCE readouts below are mirrored TO
 * THE DECIMAL from notebooks/hypersphere-vmf-geometry/hypersphere_vmf_geometry.py —
 * its grid_table() prints the equator Var(t)/band and the per-kappa rho / kappa-hat
 * numbers, and finance_demo() prints the d=1536 cluster figures.
 * test_coordinate_marginal / test_mean_resultant / test_mle_recovery /
 * test_finance_clusters assert them. Change a number here -> change it there, and
 * re-run the notebook. The density curves are the closed-form laws the notebook
 * samples, drawn in log-space so they auto-scale without a Bessel function.
 */

// --- baked from grid_table()["equator"] (uniform-sphere coordinate marginal) ----
type EqRow = { d: number; projVar: number; bandFrac: number };
const EQUATOR: EqRow[] = [
  { d: 2,    projVar: 0.49844, bandFrac: 0.06377 },
  { d: 3,    projVar: 0.33369, bandFrac: 0.10000 },
  { d: 5,    projVar: 0.20038, bandFrac: 0.14950 },
  { d: 10,   projVar: 0.09983, bandFrac: 0.23013 },
  { d: 20,   projVar: 0.04996, bandFrac: 0.33374 },
  { d: 50,   projVar: 0.02004, bandFrac: 0.51494 },
  { d: 100,  projVar: 0.00991, bandFrac: 0.68025 },
  { d: 200,  projVar: 0.00496, bandFrac: 0.84218 },
  { d: 500,  projVar: 0.00200, bandFrac: 0.97480 },
  { d: 768,  projVar: 0.00130, bandFrac: 0.99449 },
  { d: 1536, projVar: 0.00065, bandFrac: 0.99991 },
];

// --- baked from grid_table()["vmf"] (mean resultant + kappa round-trip at d=100) -
const VMF_DIM = 100;
type VmfRow = { kappa: number; rho: number; kHatBan: number; kHatExact: number };
const VMF: VmfRow[] = [
  { kappa: 0,    rho: 0.00000, kHatBan: 0.00,    kHatExact: 0.00 },
  { kappa: 1,    rho: 0.01000, kHatBan: 1.00,    kHatExact: 1.00 },
  { kappa: 2,    rho: 0.01999, kHatBan: 2.00,    kHatExact: 2.00 },
  { kappa: 5,    rho: 0.04988, kHatBan: 5.00,    kHatExact: 5.00 },
  { kappa: 10,   rho: 0.09904, kHatBan: 10.00,   kHatExact: 10.00 },
  { kappa: 20,   rho: 0.19271, kHatBan: 20.01,   kHatExact: 20.00 },
  { kappa: 50,   rho: 0.41507, kHatBan: 50.06,   kHatExact: 50.00 },
  { kappa: 100,  rho: 0.61957, kHatBan: 100.17,  kHatExact: 100.00 },
  { kappa: 200,  rho: 0.78221, kHatBan: 200.30,  kHatExact: 200.00 },
  { kappa: 500,  rho: 0.90580, kHatBan: 500.41,  kHatExact: 500.00 },
  { kappa: 1000, rho: 0.95170, kHatBan: 1000.45, kHatExact: 1000.00 },
];

// --- baked from finance_demo() (two synthetic vMF clusters in R^1536) -----------
const FINANCE = {
  dim: 1536,
  tight: { kappa: 900, rbar: 0.4613, kHat: 899.8, intraCos: 0.2118 },
  loose: { kappa: 300, rbar: 0.1905, kHat: 303.6, intraCos: 0.0351 },
  interCos: -0.0028,
};

type Panel = 'equator' | 'vmf' | 'distance' | 'clusters';

const PLOT_W = 540;
const PLOT_H = 190;
const PAD = 26;

// Sample a log-density over [xmin, xmax] and return an SVG path, normalized to the
// plot height by subtracting the max log-value (no normalizing constant needed).
function densityPath(logf: (x: number) => number, xmin: number, xmax: number): string {
  const npts = 200;
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
    // +Infinity is a true pole (the d<3 equatorial law) -> pin to the top; -Infinity
    // is a true zero (the density vanishes at the boundary) -> pin to the bottom.
    let y = Number.isFinite(lg[i]) ? Math.exp(lg[i] - mx) : (lg[i] === Infinity ? 1 : 0);
    y = Math.min(1, Math.max(0, y));
    const px = PAD + (plotW * (xs[i] - xmin)) / (xmax - xmin);
    const py = PAD + plotH * (1 - y);
    path += (i === 0 ? 'M' : 'L') + px.toFixed(1) + ' ' + py.toFixed(1) + ' ';
  }
  return path;
}

// Map an arbitrary value function into the plot box (used for the distance line).
function valuePath(f: (x: number) => number, xmin: number, xmax: number, ymin: number, ymax: number): string {
  const npts = 80;
  const plotW = PLOT_W - 2 * PAD;
  const plotH = PLOT_H - 2 * PAD;
  let path = '';
  for (let i = 0; i < npts; i++) {
    const x = xmin + ((xmax - xmin) * i) / (npts - 1);
    const y = Math.min(1, Math.max(0, (f(x) - ymin) / (ymax - ymin)));
    const px = PAD + (plotW * (x - xmin)) / (xmax - xmin);
    const py = PAD + plotH * (1 - y);
    path += (i === 0 ? 'M' : 'L') + px.toFixed(1) + ' ' + py.toFixed(1) + ' ';
  }
  return path;
}

// Closed-form log-densities (up to a constant). The vMF law is the uniform
// coordinate marginal tilted by the exponential kappa*t; at kappa=0 they coincide.
// Let the arithmetic carry the boundary sign: (d-3)/2 < 0 gives +Infinity (a pole,
// d<3), > 0 gives -Infinity (a zero, d>3).
const logUniformCoord = (d: number) => (t: number) =>
  t <= -1 || t >= 1 ? ((d - 3) / 2) * -Infinity : ((d - 3) / 2) * Math.log(1 - t * t);
const logVMF = (d: number, kappa: number) => (t: number) =>
  t <= -1 || t >= 1 ? ((d - 3) / 2) * -Infinity : kappa * t + ((d - 3) / 2) * Math.log(1 - t * t);

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
  { id: 'equator', label: 'Equatorial band' },
  { id: 'vmf', label: 'von Mises–Fisher' },
  { id: 'distance', label: 'Cosine vs distance' },
  { id: 'clusters', label: 'Cluster tightness' },
];

export default function HypersphereLaboratory() {
  const [panel, setPanel] = useState<Panel>('equator');
  const [dIdx, setDIdx] = useState(6); // EQUATOR index; start at d = 100
  const [kIdx, setKIdx] = useState(7); // VMF index; start at kappa = 100
  const eq = EQUATOR[dIdx];
  const vm = VMF[kIdx];
  const formulaRef = useRef<HTMLDivElement>(null);

  // Live KaTeX of the law the active panel illustrates.
  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'equator'
        ? `p(t)\\propto(1-t^2)^{(d-3)/2},\\qquad \\operatorname{Var}(t)=\\tfrac{1}{d}`
        : panel === 'vmf'
        ? `f(x)=C_d(\\kappa)\\,e^{\\kappa\\,\\mu^{\\top}x},\\qquad \\rho=A_d(\\kappa)=\\frac{I_{d/2}(\\kappa)}{I_{d/2-1}(\\kappa)}`
        : panel === 'distance'
        ? `\\lVert x-y\\rVert^2=2-2\\langle x,y\\rangle=2(1-\\cos\\theta)`
        : `\\hat\\mu=\\frac{R}{\\lVert R\\rVert},\\qquad A_d(\\hat\\kappa)=\\bar r=\\frac{\\lVert R\\rVert}{n}`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const equatorPlot = useMemo(() => (
    <Plot xmin={-1} xmax={1} markerX={0}
      ticks={[{ x: -1, label: '−1' }, { x: 0, label: '0' }, { x: 1, label: '+1' }]}
      paths={[
        { d: densityPath(logUniformCoord(2), -1, 1), color: 'var(--color-text-secondary)', width: 1.4 },
        { d: densityPath(logUniformCoord(eq.d), -1, 1), color: 'var(--color-accent)', width: 2.4 },
      ]} />
  ), [eq.d]);

  const vmfPlot = useMemo(() => (
    <Plot xmin={-1} xmax={1} markerX={vm.rho}
      ticks={[{ x: -1, label: '−1' }, { x: 0, label: '0' }, { x: 1, label: '+1' }]}
      paths={[
        { d: densityPath(logVMF(VMF_DIM, 0), -1, 1), color: 'var(--color-text-secondary)', width: 1.4 },
        { d: densityPath(logVMF(VMF_DIM, vm.kappa), -1, 1), color: 'var(--color-accent)', width: 2.4 },
      ]} />
  ), [vm.kappa, vm.rho]);

  const distancePlot = useMemo(() => (
    <Plot xmin={-1} xmax={1}
      ticks={[{ x: -1, label: 'cos −1' }, { x: 0, label: '0' }, { x: 1, label: '+1' }]}
      paths={[{ d: valuePath((t) => 2 - 2 * t, -1, 1, 0, 4), color: 'var(--color-accent)', width: 2.4 }]} />
  ), []);

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

      {/* dimension slider (equatorial panel) */}
      {panel === 'equator' && (
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', marginBottom: '1rem' }}>
          <span style={{ minWidth: '7rem' }}>dimension d = <strong>{eq.d}</strong></span>
          <input type="range" min={0} max={EQUATOR.length - 1} step={1} value={dIdx}
            onChange={(e) => setDIdx(parseInt(e.target.value, 10))}
            style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
        </label>
      )}

      {/* concentration slider (vMF panel) */}
      {panel === 'vmf' && (
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', marginBottom: '1rem' }}>
          <span style={{ minWidth: '7rem' }}>concentration κ = <strong>{vm.kappa}</strong></span>
          <input type="range" min={0} max={VMF.length - 1} step={1} value={kIdx}
            onChange={(e) => setKIdx(parseInt(e.target.value, 10))}
            style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
        </label>
      )}

      {panel === 'equator' && (
        <>
          {equatorPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Projection t = ⟨u, v⟩ of a uniform random unit vector onto a fixed axis (faint curve: d = 2,
            the spread-out arcsine law). As d grows it spikes at the equator t = 0, with variance 1/d.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`Var(t) at d = ${eq.d}`} value={eq.projVar.toFixed(5)} accent={eq.d >= 100} />
            <Readout label="theory: 1/d" value={(1 / eq.d).toFixed(5)} />
            <Readout label="mass within 0.1 of the equator" value={`${(eq.bandFrac * 100).toFixed(1)}%`} accent={eq.bandFrac > 0.9} />
          </div>
        </>
      )}

      {panel === 'vmf' && (
        <>
          {vmfPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            The von Mises–Fisher density of t = μ·x at d = {VMF_DIM} (faint curve: κ = 0, the uniform
            equatorial law). Raising κ tilts the mass toward the mean direction; the dashed marker is the
            mean resultant ρ = A<sub>d</sub>(κ), where the peak settles.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`mean resultant ρ = A_d(κ)`} value={vm.rho.toFixed(4)} accent={vm.rho > 0.5} />
            <Readout label="κ̂ from ρ (exact root)" value={vm.kHatExact.toFixed(0)} />
            <Readout label="κ̂ (Banerjee approximation)" value={vm.kHatBan.toFixed(2)} accent={Math.abs(vm.kHatBan - vm.kappa) > 0.4} />
          </div>
        </>
      )}

      {panel === 'distance' && (
        <>
          {distancePlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Squared Euclidean distance ‖x − y‖² between unit vectors as a function of their cosine
            similarity. The map 2 − 2cos θ is strictly decreasing, so the most-similar candidate is the
            nearest one — ranking by cosine and by distance is the same ranking.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="cos θ = 0.90" value={`dist ${Math.sqrt(2 - 2 * 0.9).toFixed(3)}`} />
            <Readout label="cos θ = 0.50" value={`dist ${Math.sqrt(2 - 2 * 0.5).toFixed(3)}`} />
            <Readout label="cos θ = 0.00" value={`dist ${Math.sqrt(2 - 2 * 0.0).toFixed(3)}`} />
            <Readout label="argmax cos = argmin dist" value="always" accent />
          </div>
        </>
      )}

      {panel === 'clusters' && (
        <div style={{ fontFamily: 'var(--font-sans)' }}>
          <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', marginBottom: '0.9rem' }}>
            Two synthetic vMF clusters of document embeddings at the production dimension d = {FINANCE.dim}:
            a tight theme (κ = {FINANCE.tight.kappa}) and a loose one (κ = {FINANCE.loose.kappa}). The bar is
            the mean resultant length r̄ — how tightly the cluster concentrates. Cluster tightness is
            measurable, and distinct subtopics separate (inter-cluster cosine {FINANCE.interCos.toFixed(4)}).
          </div>
          {[
            { name: `tight  (κ = ${FINANCE.tight.kappa})`, r: FINANCE.tight.rbar, kHat: FINANCE.tight.kHat, intra: FINANCE.tight.intraCos, color: 'var(--color-accent)' },
            { name: `loose  (κ = ${FINANCE.loose.kappa})`, r: FINANCE.loose.rbar, kHat: FINANCE.loose.kHat, intra: FINANCE.loose.intraCos, color: 'var(--color-text-secondary)' },
          ].map((b) => (
            <div key={b.name} style={{ marginBottom: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', marginBottom: '0.2rem' }}>
                <span>{b.name}</span>
                <span style={{ color: 'var(--color-text-secondary)' }}>r̄ <strong style={{ color: 'var(--color-text)' }}>{b.r.toFixed(4)}</strong> · κ̂ <strong style={{ color: 'var(--color-text)' }}>{b.kHat.toFixed(0)}</strong> · intra-cos <strong style={{ color: 'var(--color-text)' }}>{b.intra.toFixed(4)}</strong></span>
              </div>
              <div style={{ height: '0.8rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
                <div style={{ width: `${Math.max(2, b.r * 100)}%`, height: '100%', background: b.color, borderRadius: '999px', transition: 'width 0.3s' }} />
              </div>
            </div>
          ))}
          <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem' }}>
            The maximum-likelihood estimator recovers each planted concentration (κ̂ ≈ κ), so "how tight is
            this theme?" has a number — and the tight cluster's members are far more alike
            ({FINANCE.tight.intraCos.toFixed(4)}) than they are to the other cluster ({FINANCE.interCos.toFixed(4)}).
          </div>
        </div>
      )}
    </div>
  );
}
