import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Spectrum Laboratory — a kept-dimension slider over four panels on PCA of an
 * embedding cloud:
 *   - Explained variance: the cumulative EVR(k) rises steeply then flattens (a
 *     concentrated spectrum / scree), so a few directions hold most of the variance.
 *   - Reconstruction error: ||X~ - X~_k||_F^2 = sum_{i>k} s_i^2 collapses as k grows,
 *     and always beats a random rank-k projection (Eckart-Young-Mirsky).
 *   - Top-2 PC scatter: three topical clusters separate in two of 1536 dimensions.
 *   - Recall after projection: nearest-neighbor recall@10 retained, PCA vs random.
 *
 * UNLIKE the other formalRAG labs, there is no closed-form spectrum to draw — the
 * spectrum IS the data — so every curve here is a BAKED POLYLINE mirrored TO THE
 * DECIMAL from notebooks/pca-dimensionality-reduction/pca_dimensionality_reduction.py;
 * only the per-panel KaTeX formula band is closed-form. grid_table() prints the
 * scree / error / recall / scatter blocks and finance_demo() the d=1536 headline;
 * test_explained_variance_and_effective_rank / test_eckart_young /
 * test_recall_after_projection / test_finance_spectrum assert them. Change a number
 * here -> change it there, and re-run the notebook.
 */

// --- baked from grid_table() on the finance cloud (d=1536, intrinsic 48, 3 clusters)
const K_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 768];
const LAMBDA1 = 10.8953;
const SCREE_AT_K = [8.2357, 0.91913, 0.70172, 0.40186, 0.1479, 0.00159, 0.00129, 0.00094, 0.0005, 0.00023];
const EVR_AT_K = [0.5559, 0.6117, 0.7015, 0.8235, 0.9406, 0.9842, 0.9869, 0.9910, 0.9962, 0.9988];
const PCA_ERR = [18324.3, 16021.8, 12314.9, 7282.0, 2449.0, 652.2, 542.5, 372.9, 157.9, 49.9];
const RAND_ERR = [41224.1, 41145.8, 41018.2, 40845.4, 40391.4, 39577.5, 37916.7, 34395.0, 27579.3, 20529.3];
const RECALL_PCA = [0.286, 0.240, 0.299, 0.641, 0.852, 0.941, 0.944, 0.950, 0.965, 0.983];
const RECALL_RAND = [0.007, 0.011, 0.034, 0.063, 0.068, 0.103, 0.218, 0.457, 0.564, 0.655];
const MAX_ERR = RAND_ERR[0];

const FINANCE = { dim: 1536, effRank: 6.10, kept: 128, evrAtKept: 0.9869, recallPca: 0.944, recallRand: 0.257 };

// finance_demo() top-2 principal-component scores of a 75-point cluster subsample.
const SCATTER: [number, number, number][] = [
  [4.467, 1.835, 0], [4.114, 2.761, 0], [4.706, 2.334, 0], [4.477, 3.689, 0], [4.764, 1.906, 0],
  [4.132, 0.434, 0], [4.964, 1.67, 0], [4.934, 0.418, 0], [4.086, 2.46, 0], [3.623, 1.861, 0],
  [5.136, 2.897, 0], [2.841, 2.771, 0], [5.014, 1.988, 0], [5.444, 0.232, 0], [3.939, 1.646, 0],
  [4.219, 1.868, 0], [4.486, 1.917, 0], [5.023, 1.657, 0], [3.391, 1.154, 0], [4.223, 1.68, 0],
  [4.455, 3.066, 0], [2.923, 1.694, 0], [2.619, 2.675, 0], [5.074, 2.055, 0], [4.156, 1.894, 0],
  [-3.828, 1.703, 1], [-4.389, 1.646, 1], [-4.204, 3.149, 1], [-3.338, 3.026, 1], [-4.055, 3.832, 1],
  [-3.545, 3.404, 1], [-5.143, 2.442, 1], [-5.028, 3.407, 1], [-5.133, 2.801, 1], [-4.352, 1.79, 1],
  [-3.69, 3.513, 1], [-3.85, 2.964, 1], [-4.133, 2.687, 1], [-3.911, 2.797, 1], [-4.47, 1.318, 1],
  [-3.441, 2.33, 1], [-3.379, 2.534, 1], [-2.528, 3.043, 1], [-3.893, 1.256, 1], [-3.657, 1.831, 1],
  [-4.582, 3.013, 1], [-3.651, 2.85, 1], [-3.967, 3.102, 1], [-3.81, 2.118, 1], [-2.597, 2.279, 1],
  [0.405, -3.08, 2], [0.167, -3.116, 2], [-0.887, -4.562, 2], [-0.957, -3.426, 2], [1.435, -3.376, 2],
  [-0.065, -3.919, 2], [-0.169, -2.734, 2], [-1.397, -2.819, 2], [-0.228, -3.446, 2], [-0.746, -3.818, 2],
  [0.592, -3.65, 2], [-1.596, -4.201, 2], [-0.139, -2.861, 2], [0.043, -3.539, 2], [-1.151, -2.238, 2],
  [-0.98, -4.486, 2], [1.211, -3.627, 2], [0.48, -2.288, 2], [-0.85, -4.272, 2], [-1.04, -4.631, 2],
  [-0.572, -4.052, 2], [0.466, -4.542, 2], [-0.237, -3.62, 2], [0.28, -3.224, 2], [-1.027, -4.131, 2],
];
const CLUSTER_COLORS = ['var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', 'var(--color-text-secondary)'];

type Panel = 'variance' | 'error' | 'scatter' | 'recall';

const PLOT_W = 540;
const PLOT_H = 200;
const PAD = 28;

// Log-spaced kept-dimension axis shared by the variance / error / recall panels.
const LOG_KMIN = Math.log(K_GRID[0]);
const LOG_KMAX = Math.log(K_GRID[K_GRID.length - 1]);
const kx = (k: number) => PAD + ((PLOT_W - 2 * PAD) * (Math.log(k) - LOG_KMIN)) / (LOG_KMAX - LOG_KMIN);
const vy = (v: number, lo: number, hi: number) =>
  PAD + (PLOT_H - 2 * PAD) * (1 - Math.min(1, Math.max(0, (v - lo) / (hi - lo))));

function kCurve(vals: number[], lo: number, hi: number): string {
  return vals.map((v, i) => (i === 0 ? 'M' : 'L') + kx(K_GRID[i]).toFixed(1) + ' ' + vy(v, lo, hi).toFixed(1)).join(' ');
}

function KPlot({ paths, markerK, yticks }: {
  paths: { d: string; color: string; width: number }[];
  markerK: number; yticks?: { y: number; label: string }[];
}) {
  const kticks = [2, 8, 32, 128, 768];
  return (
    <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
      {yticks?.map((t) => (
        <text key={t.label} x={PAD - 6} y={t.y + 3} textAnchor="end" fontSize={9}
          fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{t.label}</text>
      ))}
      <line x1={kx(markerK)} y1={PAD - 6} x2={kx(markerK)} y2={PLOT_H - PAD}
        stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="3 3" />
      {kticks.map((k) => (
        <text key={k} x={kx(k)} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10}
          fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{k}</text>
      ))}
      {paths.map((p, i) => (
        <path key={i} d={p.d} fill="none" stroke={p.color} strokeWidth={p.width}
          strokeLinejoin="round" strokeLinecap="round" opacity={p.width < 2 ? 0.5 : 1} />
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
  { id: 'variance', label: 'Explained variance' },
  { id: 'error', label: 'Reconstruction error' },
  { id: 'scatter', label: 'Top-2 PC scatter' },
  { id: 'recall', label: 'Recall after projection' },
];

export default function SpectrumLaboratory() {
  const [panel, setPanel] = useState<Panel>('variance');
  const [kIdx, setKIdx] = useState(6); // K_GRID index; start at k = 128
  const k = K_GRID[kIdx];
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'variance'
        ? `\\mathrm{EVR}(k)=\\frac{\\sum_{i\\le k}\\lambda_i}{\\sum_i \\lambda_i},\\qquad n_{\\mathrm{eff}}=\\frac{(\\sum_i\\lambda_i)^2}{\\sum_i\\lambda_i^2}`
        : panel === 'error'
        ? `\\lVert \\tilde X-\\tilde X_k\\rVert_F^2=\\sum_{i>k}\\sigma_i^2=(n-1)\\sum_{i>k}\\lambda_i`
        : panel === 'scatter'
        ? `\\text{scores } V_2^{\\top}(x-\\bar x)\\in\\mathbb{R}^2 \\text{ of } x\\in\\mathbb{R}^{1536}`
        : `\\text{recall@10 retained after projecting } \\mathbb{R}^{1536}\\to\\mathbb{R}^{k}`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const variancePlot = useMemo(() => (
    <KPlot markerK={k}
      yticks={[{ y: vy(1, 0, 1), label: '1.0' }, { y: vy(0.5, 0, 1), label: '0.5' }, { y: vy(0, 0, 1), label: '0' }]}
      paths={[
        { d: `M${PAD} ${vy(1, 0, 1).toFixed(1)} L${PLOT_W - PAD} ${vy(1, 0, 1).toFixed(1)}`, color: 'var(--color-border)', width: 1 },
        { d: kCurve(SCREE_AT_K.map((v) => v / LAMBDA1), 0, 1), color: 'var(--color-text-secondary)', width: 1.4 },
        { d: kCurve(EVR_AT_K, 0, 1), color: 'var(--color-accent)', width: 2.4 },
      ]} />
  ), [k]);

  const errorPlot = useMemo(() => (
    <KPlot markerK={k}
      paths={[
        { d: kCurve(RAND_ERR.map((v) => v / MAX_ERR), 0, 1), color: 'var(--color-text-secondary)', width: 1.4 },
        { d: kCurve(PCA_ERR.map((v) => v / MAX_ERR), 0, 1), color: 'var(--color-accent)', width: 2.4 },
      ]} />
  ), [k]);

  const recallPlot = useMemo(() => (
    <KPlot markerK={k}
      yticks={[{ y: vy(1, 0, 1), label: '100%' }, { y: vy(0.5, 0, 1), label: '50%' }, { y: vy(0, 0, 1), label: '0' }]}
      paths={[
        { d: kCurve(RECALL_RAND, 0, 1), color: 'var(--color-text-secondary)', width: 1.4 },
        { d: kCurve(RECALL_PCA, 0, 1), color: 'var(--color-accent)', width: 2.4 },
      ]} />
  ), [k]);

  const scatterPlot = useMemo(() => {
    const sx = (x: number) => PAD + ((PLOT_W - 2 * PAD) * (x + 6)) / 12;
    const sy = (y: number) => PAD + ((PLOT_H - 2 * PAD) * (1 - (y + 5.2) / 9.4));
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={sx(0)} y1={PAD - 4} x2={sx(0)} y2={PLOT_H - PAD + 4} stroke="var(--color-border)" strokeWidth={0.75} />
        <line x1={PAD} y1={sy(0)} x2={PLOT_W - PAD} y2={sy(0)} stroke="var(--color-border)" strokeWidth={0.75} />
        {SCATTER.map((p, i) => (
          <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={3.4} fill={CLUSTER_COLORS[p[2]]} opacity={0.82} />
        ))}
        <text x={PLOT_W - PAD} y={PLOT_H - PAD + 14} textAnchor="end" fontSize={10}
          fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">PC1</text>
        <text x={sx(0) + 6} y={PAD + 2} fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">PC2</text>
      </svg>
    );
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

      {panel !== 'scatter' && (
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', marginBottom: '1rem' }}>
          <span style={{ minWidth: '9rem' }}>keep top <strong>{k}</strong> of 1536 dims</span>
          <input type="range" min={0} max={K_GRID.length - 1} step={1} value={kIdx}
            onChange={(e) => setKIdx(parseInt(e.target.value, 10))}
            style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
        </label>
      )}

      {panel === 'variance' && (
        <>
          {variancePlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Cumulative explained variance (accent) and the per-dimension eigenvalue / scree (faint), against
            the kept dimension k (log axis). The steep rise is a concentrated spectrum: a few directions hold
            most of the variance.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`EVR at k = ${k}`} value={`${(EVR_AT_K[kIdx] * 100).toFixed(2)}%`} accent />
            <Readout label="effective rank n_eff" value={FINANCE.effRank.toFixed(1)} />
            <Readout label="kept / ambient" value={`${k} / ${FINANCE.dim}`} />
          </div>
        </>
      )}

      {panel === 'error' && (
        <>
          {errorPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Squared Frobenius reconstruction error vs kept dimension (normalized): PCA / truncated SVD
            (accent) collapses, while a random rank-k projection of the same width (faint) barely improves.
            Eckart-Young: the truncated SVD is optimal.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`PCA error² at k = ${k}`} value={PCA_ERR[kIdx].toFixed(0)} accent />
            <Readout label="random projection error²" value={RAND_ERR[kIdx].toFixed(0)} />
            <Readout label="truncated SVD optimal?" value="yes — by Eckart–Young" accent />
          </div>
        </>
      )}

      {panel === 'scatter' && (
        <>
          {scatterPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Three topical clusters of 1536-dimensional embeddings, projected onto their top two principal
            components. Two of 1536 directions already separate the topics — the variance lives in a few
            directions, which is why projection keeps retrieval intact.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="ambient dimension" value={`${FINANCE.dim}`} />
            <Readout label="components shown" value="2 (PC1, PC2)" />
            <Readout label="clusters separated?" value="yes" accent />
          </div>
        </>
      )}

      {panel === 'recall' && (
        <>
          {recallPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Nearest-neighbor recall@10 retained after projecting to k dimensions: PCA (accent), the
            data-dependent projection, against a data-oblivious random projection (faint) of the same width.
            PCA spends its dimensions where the cloud lives, so it keeps far more recall per dimension.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`PCA recall@10 at k = ${k}`} value={`${(RECALL_PCA[kIdx] * 100).toFixed(1)}%`} accent />
            <Readout label="random projection recall@10" value={`${(RECALL_RAND[kIdx] * 100).toFixed(1)}%`} />
            <Readout label="PCA beats random?" value={RECALL_PCA[kIdx] >= RECALL_RAND[kIdx] ? 'yes' : 'no'} accent />
          </div>
        </>
      )}
    </div>
  );
}
