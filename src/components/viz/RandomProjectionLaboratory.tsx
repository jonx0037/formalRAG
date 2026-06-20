import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Random Projection Laboratory — a target-dimension slider over four panels on the
 * Johnson-Lindenstrauss embedding of a 1536-d financial cloud (n = 500):
 *   - Distortion histogram: the pairwise squared-distortion ||f(u)-f(v)||^2/||u-v||^2
 *     spreads across all bins at small k and concentrates into the +/-eps band as k
 *     grows — the visual proof that ||f(x)||^2/||x||^2 ~ chi^2_k/k -> 1.
 *   - Typical vs worst: the p01-p99 band tightens fast, but the WORST of 124,750 pairs
 *     stays outside +/-eps until k is large (the union bound), which is why the
 *     guaranteed k is ~1435 for eps = 0.2.
 *   - Families: Gaussian, Rademacher, and sparse Achlioptas concentrate at the same
 *     sqrt(2/k) rate — any of them is a valid JL map.
 *   - Recall after projection: recall@10 retained, random projection (oblivious) vs
 *     PCA (data-dependent) — distances are preserved, exact rankings are not.
 *
 * Every number is a BAKED POLYLINE/BAR mirrored TO THE DECIMAL from
 * notebooks/johnson-lindenstrauss/johnson_lindenstrauss.py (grid_table / finance_demo);
 * test_chi_square_distribution / test_dimension_independence / test_projection_families_agree
 * / test_recall_after_projection / test_finance_distortion assert them. Change a number
 * here -> change it there, and re-run the notebook.
 */

// --- baked from grid_table() on the finance cloud (d=1536, n=500, intrinsic 48, 3 clusters)
const K_GRID = [8, 16, 32, 64, 128, 256, 512];
const DIST_MEAN = [0.9798, 0.9611, 0.9475, 0.9572, 0.962, 1.0063, 1.0116];
const DIST_STD = [0.407, 0.2996, 0.2085, 0.1426, 0.1062, 0.0777, 0.0556];
const P01 = [0.2436, 0.3901, 0.5246, 0.6604, 0.7518, 0.8402, 0.8857];
const P99 = [2.123, 1.761, 1.5286, 1.3312, 1.2485, 1.2026, 1.147];
const MAXDEV = [2.737, 1.9258, 1.3313, 0.7104, 0.5112, 0.4985, 0.3092];
const RECALL_RAND = [0.046, 0.033, 0.104, 0.129, 0.233, 0.333, 0.549];
const RECALL_PCA = [0.427, 0.547, 0.845, 0.956, 0.96, 0.972, 1.0];

// distortion histograms (30 bins, range 0.3..1.7), one row per K_GRID value
const HIST_CENTERS = [
  0.323, 0.37, 0.417, 0.463, 0.51, 0.557, 0.603, 0.65, 0.697, 0.743, 0.79, 0.837, 0.883, 0.93, 0.977,
  1.023, 1.07, 1.117, 1.163, 1.21, 1.257, 1.303, 1.35, 1.397, 1.443, 1.49, 1.537, 1.583, 1.63, 1.677,
];
const HIST: number[][] = [
  [1575, 2102, 2574, 3113, 3625, 4260, 4650, 5199, 5595, 5800, 5810, 5925, 5923, 5940, 5723, 5487, 5202, 4845, 4458, 4081, 3668, 3518, 3121, 2743, 2438, 2105, 1877, 1638, 1404, 1209],
  [390, 738, 1282, 1998, 2855, 3669, 4693, 5636, 6262, 6707, 7326, 7796, 7980, 7695, 7281, 7180, 6675, 6061, 5404, 4763, 4112, 3491, 2988, 2421, 2016, 1596, 1304, 968, 832, 596],
  [16, 65, 169, 400, 787, 1485, 2743, 4161, 5645, 7888, 9703, 11253, 11812, 11837, 11007, 9835, 8491, 6860, 5388, 4055, 3155, 2293, 1630, 1216, 863, 614, 408, 300, 204, 157],
  [0, 0, 2, 10, 34, 108, 358, 1176, 2759, 5443, 8689, 12803, 15993, 17094, 16058, 13356, 10284, 7415, 5134, 3387, 2090, 1229, 671, 361, 161, 78, 38, 12, 3, 3],
  [0, 0, 0, 0, 0, 0, 5, 59, 341, 1563, 5405, 12709, 20668, 22904, 20001, 15911, 10766, 6741, 3899, 2123, 996, 425, 158, 57, 16, 3, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 2, 34, 365, 2308, 8785, 19877, 29458, 28102, 19297, 10151, 4421, 1474, 357, 88, 25, 4, 1, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 304, 3033, 14821, 34372, 40031, 23173, 7352, 1447, 182, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0],
];

// jl_min_dim(n=500, eps): the guaranteed target dimension (all pairs, worst case)
const THRESHOLD: [number, number][] = [[0.1, 5327], [0.15, 2456], [0.2, 1435], [0.3, 691], [0.5, 299]];
// distortion std at k=64 for each family (test_projection_families_agree)
const FAMILY_STD: [string, number][] = [['Gaussian', 0.154], ['Rademacher', 0.1658], ['Sparse ±1/0', 0.1622]];

const FINANCE = { dim: 1536, n: 500, eps: 0.2, kGuaranteed: 1435, kept: 128, meanDev: 0.092, maxDev: 0.511, recallRand: 0.233, recallPca: 0.96 };
const EPS = 0.2;

type Panel = 'hist' | 'band' | 'families' | 'recall';

const PLOT_W = 540;
const PLOT_H = 210;
const PAD = 30;

const LOG_KMIN = Math.log(K_GRID[0]);
const LOG_KMAX = Math.log(K_GRID[K_GRID.length - 1]);
const kx = (k: number) => PAD + ((PLOT_W - 2 * PAD) * (Math.log(k) - LOG_KMIN)) / (LOG_KMAX - LOG_KMIN);
const vy = (v: number, lo: number, hi: number) =>
  PAD + (PLOT_H - 2 * PAD) * (1 - Math.min(1, Math.max(0, (v - lo) / (hi - lo))));

function kCurve(vals: number[], lo: number, hi: number): string {
  return vals.map((v, i) => (i === 0 ? 'M' : 'L') + kx(K_GRID[i]).toFixed(1) + ' ' + vy(v, lo, hi).toFixed(1)).join(' ');
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
  { id: 'hist', label: 'Distortion histogram' },
  { id: 'band', label: 'Typical vs worst' },
  { id: 'families', label: 'Projection families' },
  { id: 'recall', label: 'Recall after projection' },
];

export default function RandomProjectionLaboratory() {
  const [panel, setPanel] = useState<Panel>('hist');
  const [kIdx, setKIdx] = useState(4); // K_GRID index; start at k = 128
  const k = K_GRID[kIdx];
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'hist'
        ? `\\frac{\\lVert f(u)-f(v)\\rVert^2}{\\lVert u-v\\rVert^2}\\sim\\frac{\\chi^2_k}{k}\\;\\longrightarrow\\;1`
        : panel === 'band'
        ? `\\Pr\\!\\left[\\,\\bigl|\\tfrac{\\lVert f(x)\\rVert^2}{\\lVert x\\rVert^2}-1\\bigr|>\\varepsilon\\right]\\le 2e^{-\\frac{k}{2}(\\varepsilon^2/2-\\varepsilon^3/3)}`
        : panel === 'families'
        ? `\\mathbb{E}\\lVert f(x)\\rVert^2=\\lVert x\\rVert^2,\\qquad \\mathrm{std}\\approx\\sqrt{2/k}`
        : `\\text{recall@10 after }\\mathbb{R}^{1536}\\to\\mathbb{R}^{k}:\\ \\text{oblivious vs data-dependent}`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  // --- Panel: distortion histogram (bars), with the +/-eps band shaded -------------
  const histPlot = useMemo(() => {
    const counts = HIST[kIdx];
    const maxC = Math.max(...counts);
    const x0 = PAD, x1 = PLOT_W - PAD, lo = HIST_CENTERS[0] - 0.03, hi = HIST_CENTERS[HIST_CENTERS.length - 1] + 0.03;
    const hx = (r: number) => x0 + ((x1 - x0) * (r - lo)) / (hi - lo);
    const bw = (x1 - x0) / HIST_CENTERS.length;
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <rect x={hx(1 - EPS)} y={PAD - 6} width={hx(1 + EPS) - hx(1 - EPS)} height={PLOT_H - PAD - (PAD - 6)}
          fill="var(--color-accent)" opacity={0.1} />
        {counts.map((c, i) => {
          const h = ((PLOT_H - 2 * PAD) * c) / maxC;
          return <rect key={i} x={hx(HIST_CENTERS[i]) - bw / 2 + 0.5} y={PLOT_H - PAD - h} width={bw - 1} height={h}
            fill="var(--color-accent)" opacity={0.78} />;
        })}
        <line x1={hx(1)} y1={PAD - 6} x2={hx(1)} y2={PLOT_H - PAD} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="3 3" />
        <line x1={x0} y1={PLOT_H - PAD} x2={x1} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
        {[0.5, 1.0, 1.5].map((r) => (
          <text key={r} x={hx(r)} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10}
            fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{r.toFixed(1)}</text>
        ))}
        <text x={hx(1)} y={PAD - 10} textAnchor="middle" fontSize={9} fill="var(--color-accent)" fontFamily="var(--font-sans)">1 ± ε band</text>
      </svg>
    );
  }, [kIdx]);

  // --- Panel: typical (p01-p99) vs worst (1 +/- maxdev) band, vs k -----------------
  const bandPlot = useMemo(() => {
    const lo = 0.0, hi = 2.6;
    const kticks = [8, 32, 128, 512];
    const upper = (i: number) => 1 + MAXDEV[i], lower = (i: number) => Math.max(0, 1 - MAXDEV[i]);
    const worstTop = K_GRID.map((kk, i) => `${i === 0 ? 'M' : 'L'}${kx(kk).toFixed(1)} ${vy(upper(i), lo, hi).toFixed(1)}`).join(' ');
    const worstBot = K_GRID.map((kk, i) => `${i === 0 ? 'M' : 'L'}${kx(kk).toFixed(1)} ${vy(lower(i), lo, hi).toFixed(1)}`).join(' ');
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <rect x={PAD} y={vy(1 + EPS, lo, hi)} width={PLOT_W - 2 * PAD} height={vy(1 - EPS, lo, hi) - vy(1 + EPS, lo, hi)}
          fill="var(--color-accent)" opacity={0.1} />
        <line x1={PAD} y1={vy(1, lo, hi)} x2={PLOT_W - PAD} y2={vy(1, lo, hi)} stroke="var(--color-text-secondary)" strokeWidth={0.75} strokeDasharray="3 3" />
        <path d={kCurve(P99, lo, hi)} fill="none" stroke="var(--color-accent)" strokeWidth={2} strokeLinejoin="round" />
        <path d={kCurve(P01, lo, hi)} fill="none" stroke="var(--color-accent)" strokeWidth={2} strokeLinejoin="round" />
        <path d={worstTop} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.3} strokeDasharray="4 2" />
        <path d={worstBot} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.3} strokeDasharray="4 2" />
        <line x1={kx(k)} y1={PAD - 6} x2={kx(k)} y2={PLOT_H - PAD} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
        <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
        {kticks.map((kk) => (
          <text key={kk} x={kx(kk)} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10}
            fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{kk}</text>
        ))}
        <text x={PLOT_W - PAD} y={vy(1 + EPS, lo, hi) - 4} textAnchor="end" fontSize={9} fill="var(--color-accent)" fontFamily="var(--font-sans)">±ε</text>
        <text x={PAD + 4} y={vy(upper(0), lo, hi) + 3} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">worst pair</text>
        <text x={kx(64)} y={vy(P99[3], lo, hi) - 5} fontSize={9} fill="var(--color-accent)" fontFamily="var(--font-sans)">p01–p99</text>
      </svg>
    );
  }, [k]);

  // --- Panel: family std bars at k=64 ----------------------------------------------
  const familiesPlot = useMemo(() => {
    const target = Math.sqrt(2 / 64);
    const x0 = PAD + 40, x1 = PLOT_W - PAD - 20, hiV = 0.22;
    const by = (v: number) => PLOT_H - PAD - ((PLOT_H - 2 * PAD) * v) / hiV;
    const slot = (x1 - x0) / FAMILY_STD.length;
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={PAD} y1={by(target)} x2={PLOT_W - PAD} y2={by(target)} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="4 3" />
        <text x={PLOT_W - PAD} y={by(target) - 4} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">√(2/k) = {target.toFixed(3)}</text>
        {FAMILY_STD.map(([name, sd], i) => {
          const cx = x0 + slot * (i + 0.5);
          return (
            <g key={name}>
              <rect x={cx - slot * 0.32} y={by(sd)} width={slot * 0.64} height={PLOT_H - PAD - by(sd)} fill="var(--color-accent)" opacity={0.78} />
              <text x={cx} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{name}</text>
              <text x={cx} y={by(sd) - 5} textAnchor="middle" fontSize={10} fill="var(--color-accent)" fontFamily="var(--font-sans)">{sd.toFixed(3)}</text>
            </g>
          );
        })}
        <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
      </svg>
    );
  }, []);

  // --- Panel: recall@10 vs k, random vs PCA ----------------------------------------
  const recallPlot = useMemo(() => {
    const kticks = [8, 32, 128, 512];
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {[0, 0.5, 1].map((v) => (
          <text key={v} x={PAD - 6} y={vy(v, 0, 1) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v === 0 ? '0' : `${v * 100}%`}</text>
        ))}
        <line x1={kx(k)} y1={PAD - 6} x2={kx(k)} y2={PLOT_H - PAD} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
        <path d={kCurve(RECALL_RAND, 0, 1)} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.5} strokeLinejoin="round" />
        <path d={kCurve(RECALL_PCA, 0, 1)} fill="none" stroke="var(--color-accent)" strokeWidth={2.4} strokeLinejoin="round" />
        <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
        {kticks.map((kk) => (
          <text key={kk} x={kx(kk)} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{kk}</text>
        ))}
        <text x={kx(512)} y={vy(RECALL_PCA[6], 0, 1) - 6} textAnchor="end" fontSize={9} fill="var(--color-accent)" fontFamily="var(--font-sans)">PCA (dependent)</text>
        <text x={kx(512)} y={vy(RECALL_RAND[6], 0, 1) - 6} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">random (oblivious)</text>
      </svg>
    );
  }, [k]);

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

      {panel !== 'families' && (
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', marginBottom: '1rem' }}>
          <span style={{ minWidth: '10rem' }}>project to <strong>{k}</strong> of 1536 dims</span>
          <input type="range" min={0} max={K_GRID.length - 1} step={1} value={kIdx}
            onChange={(e) => setKIdx(parseInt(e.target.value, 10))}
            style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
        </label>
      )}

      {panel === 'hist' && (
        <>
          {histPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Histogram of the pairwise squared-distortion ‖f(u)−f(v)‖²/‖u−v‖² across all {(FINANCE.n * (FINANCE.n - 1) / 2).toLocaleString()} pairs.
            At small k the distortion spreads everywhere; as k grows it concentrates into the ±ε band around 1 — the chi-squared law tightening at rate √(2/k).
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`distortion std at k = ${k}`} value={DIST_STD[kIdx].toFixed(3)} accent />
            <Readout label="√(2/k)" value={Math.sqrt(2 / k).toFixed(3)} />
            <Readout label="mean (≈ 1)" value={DIST_MEAN[kIdx].toFixed(3)} />
          </div>
        </>
      )}

      {panel === 'band' && (
        <>
          {bandPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            The typical band (p01–p99, accent) tightens into ±ε quickly, but the <em>worst</em> of all pairs (1 ± max deviation, dashed) stays outside far longer.
            Bounding <em>every</em> pair is what makes the guaranteed dimension large — the union bound at work.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`worst-pair distortion at k = ${k}`} value={`${(MAXDEV[kIdx] * 100).toFixed(0)}%`} accent />
            <Readout label="typical (p99 − 1)" value={`${((P99[kIdx] - 1) * 100).toFixed(0)}%`} />
            <Readout label={`guaranteed k (ε = ${FINANCE.eps}, all pairs)`} value={`${FINANCE.kGuaranteed}`} accent />
          </div>
        </>
      )}

      {panel === 'families' && (
        <>
          {familiesPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Distortion spread (std at k = 64) for the dense Gaussian map and two database-friendly alternatives: Rademacher ±1 and a sparse matrix that is two-thirds zeros.
            All three concentrate at the same √(2/k) rate, so any is a valid JL map — the sparse one for a fraction of the multiplications.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="Gaussian std" value={FAMILY_STD[0][1].toFixed(3)} accent />
            <Readout label="Rademacher std" value={FAMILY_STD[1][1].toFixed(3)} />
            <Readout label="Sparse std" value={FAMILY_STD[2][1].toFixed(3)} />
          </div>
        </>
      )}

      {panel === 'recall' && (
        <>
          {recallPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Nearest-neighbor recall@10 retained after projecting to k dimensions: data-dependent PCA (accent) against the data-oblivious random projection (faint).
            JL preserves <em>distances</em> — the currency an approximate-search index trades in — but not exact <em>rankings</em>, so a random projection is an ANN front end, not a standalone retriever; the exact order is restored by a downstream rerank.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`random recall@10 at k = ${k}`} value={`${(RECALL_RAND[kIdx] * 100).toFixed(1)}%`} />
            <Readout label="PCA recall@10" value={`${(RECALL_PCA[kIdx] * 100).toFixed(1)}%`} accent />
            <Readout label="distances preserved?" value="yes (±ε)" accent />
          </div>
        </>
      )}
    </div>
  );
}
