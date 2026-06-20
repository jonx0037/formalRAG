import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Optimized Product Quantization Laboratory — three panels for the
 * `optimized-product-quantization` topic:
 *   A. The rotation. Switch raw / PCA-only / balanced-heuristic / learned-OPQ on the SAME
 *      variance-imbalanced cloud: the PQ distortion readout drops 85.6 → 83.3 → 15.4 → 14.4 and
 *      the four per-subspace variance bars go from wildly imbalanced (raw) to product-balanced.
 *   B. Alternating-optimization convergence. The non-parametric OPQ distortion trajectory,
 *      monotone non-increasing, starting at the balanced heuristic and descending below it.
 *   C. Score-aware (anisotropic) quantization. A datapoint x and candidate codewords; the residual
 *      to the η-chosen codeword splits into a PARALLEL leg (along x, the inner-product-relevant
 *      axis) and an ORTHOGONAL leg. Raise η and the choice SWAPS to the codeword with no parallel
 *      residual. A baked finance readout shows the score-aware codebook's lower high-score IP error.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): ROT, TRAJECTORY, BALANCED_REF, SA_X/SA_CANDS, ETA_GRID and
 * FIN are mirrored TO THE DECIMAL from
 * notebooks/optimized-product-quantization/optimized_product_quantization.py (rotation_distortion_study
 * / nonparametric_opq / anisotropic_codeword / score_aware_demo, printed by viz_constants()).
 * test_opq_beats_balanced_heuristic / test_opq_trajectory_monotone / test_anisotropic_penalizes_parallel
 * / test_anisotropic_encoding_lowers_highscore_error assert them. The lab recomputes only the
 * Panel-C residual geometry (exact 2-D projection onto x) and bar scaling in TS; every distortion,
 * variance, loss and finance number is baked, never recomputed. Change a number here -> change it
 * there, and re-run the notebook. SVG text inherits the theme color via `svg text { fill }`.
 */

type Pt2 = [number, number];

// --- baked from viz_constants() -------------------------------------------------------
// Panel A: rotation_distortion_study + nonparametric_opq on variance_imbalanced_cloud(n=600,d=8),
// m=4, k*=16. `dist` = total PQ reconstruction distortion; `vars` = the m per-subspace variance sums.
const ROT = [
  { key: 'raw', label: 'raw (native axes)', dist: 85.5826, vars: [1.3543, 0.1658, 0.0227, 0.0031] },
  { key: 'pca_only', label: 'PCA-only (naive)', dist: 83.2735, vars: [1.3557, 0.1645, 0.0226, 0.0031] },
  { key: 'balanced', label: 'balanced heuristic', dist: 15.3538, vars: [0.9845, 0.3743, 0.1223, 0.0648] },
  { key: 'opq', label: 'learned OPQ', dist: 14.4016, vars: [0.9844, 0.3743, 0.1223, 0.0649] },
] as const;
const VAR_MAX = 1.4; // common scale for the variance bars (just above pca_only's lead subspace)
const DIST_MAX = 88; // common scale for the distortion comparison bars

// Panel B: the non-parametric OPQ distortion trajectory (monotone) and the heuristic baseline.
const TRAJECTORY = [15.3538, 14.4591, 14.4169, 14.4055, 14.4038, 14.403, 14.4025, 14.4016, 14.4016, 14.4016, 14.4016, 14.4016];
const BALANCED_REF = 15.3538;

// Panel C: the datapoint, candidate codewords, the η sweep (chosen index + per-candidate losses),
// and the finance score-aware comparison. SA_X = [1,0] so the parallel axis is the horizontal.
const SA_X: Pt2 = [1.0, 0.0];
const SA_CANDS: Pt2[] = [[0.8, 0.1], [1.0, 0.28], [0.6, 0.0], [0.85, 0.35]];
const ETA_GRID = [
  { eta: 1, chosen: 0, losses: [0.05, 0.0784, 0.16, 0.145] },
  { eta: 2, chosen: 1, losses: [0.09, 0.0784, 0.32, 0.1675] },
  { eta: 4, chosen: 1, losses: [0.17, 0.0784, 0.64, 0.2125] },
  { eta: 8, chosen: 1, losses: [0.33, 0.0784, 1.28, 0.3025] },
];
const FIN = { eta: 8, isoMse: 0.00439, anisoMse: 0.00284, isoRecall: 0.12, anisoRecall: 0.1308 };

const COLORS = ['var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', '#5fa873', '#6a8caf'];

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

// Panel B — the convergence polyline (baked, pure SVG; no D3 ref to go stale).
const ConvergencePlot = memo(function ConvergencePlot() {
  const pw = 560, ph = 250, pad = 44;
  const den = Math.max(1, TRAJECTORY.length - 1);      // guard the x-divisor for a 1-point trajectory
  const ymin = 14.3, ymax = 15.45;
  const xi = (i: number) => pad + ((pw - 2 * pad) * i) / den;
  const yy = (v: number) => pad + (ph - 2 * pad) * (1 - (v - ymin) / (ymax - ymin));
  const path = TRAJECTORY.map((v, i) => (i === 0 ? 'M' : 'L') + xi(i).toFixed(1) + ' ' + yy(v).toFixed(1)).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={pad} y1={ph - pad} x2={pw - pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={ph - pad} stroke="var(--color-border)" strokeWidth={1} />
      {[14.4, 14.8, 15.2].map((v) => (
        <text key={v} x={pad - 6} y={yy(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v.toFixed(1)}</text>
      ))}
      {TRAJECTORY.map((_, i) => (
        i % 2 === 0 ? <text key={i} x={xi(i)} y={ph - pad + 14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{i}</text> : null
      ))}
      <text x={pw / 2} y={ph - 6} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">alternating-optimization iteration</text>
      <text x={14} y={ph / 2} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${ph / 2})`}>PQ distortion</text>
      {/* balanced-heuristic baseline */}
      <line x1={pad} y1={yy(BALANCED_REF)} x2={pw - pad} y2={yy(BALANCED_REF)} stroke="var(--color-text-secondary)" strokeWidth={1.4} strokeDasharray="5 3" />
      <text x={pw - pad} y={yy(BALANCED_REF) - 5} textAnchor="end" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">balanced heuristic ({BALANCED_REF.toFixed(2)}) — OPQ starts here and descends</text>
      <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={2.6} strokeLinejoin="round" />
      {TRAJECTORY.map((v, i) => (<circle key={i} cx={xi(i)} cy={yy(v)} r={3} fill="var(--color-accent)" />))}
      <text x={xi(den)} y={yy(TRAJECTORY[TRAJECTORY.length - 1]) + 16} textAnchor="end" fontSize={10} fontWeight={600} fill="var(--color-accent)" fontFamily="var(--font-sans)">→ {TRAJECTORY[TRAJECTORY.length - 1].toFixed(2)}</text>
    </svg>
  );
});

// Panel C — the 2-D score-aware scene (residual split for the η-chosen codeword).
function ScoreAwareScene({ chosen }: { chosen: number }) {
  const w = 430, h = 270, pad = 34;
  const xmin = -0.05, xmax = 1.18, ymin = -0.12, ymax = 0.46;
  const sx = (x: number) => pad + ((w - 2 * pad) * (x - xmin)) / (xmax - xmin);
  const sy = (y: number) => h - pad - ((h - 2 * pad) * (y - ymin)) / (ymax - ymin);
  const c = SA_CANDS[Math.min(chosen, SA_CANDS.length - 1)] ?? SA_CANDS[0]; // guard the index lookup
  // exact 2-D residual decomposition for x = [1,0]: r = x - c; r_par along x is horizontal, r_orth vertical.
  const elbow: Pt2 = [SA_X[0], c[1]];                 // c -> [1, c_y] is the parallel leg, then -> x is orthogonal
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxWidth: '460px', height: 'auto', display: 'block' }} role="img" aria-label="score-aware residual decomposition">
      {/* the datapoint direction (parallel axis) */}
      <line x1={sx(0)} y1={sy(0)} x2={sx(1.12)} y2={sy(0)} stroke="var(--color-border)" strokeWidth={1} strokeDasharray="4 3" />
      <text x={sx(1.12)} y={sy(0) + 14} textAnchor="end" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">direction of x (parallel axis)</text>
      {/* residual elbow: parallel leg then orthogonal leg */}
      <line x1={sx(c[0])} y1={sy(c[1])} x2={sx(elbow[0])} y2={sy(elbow[1])} stroke={COLORS[0]} strokeWidth={2.6} />
      <line x1={sx(elbow[0])} y1={sy(elbow[1])} x2={sx(SA_X[0])} y2={sy(SA_X[1])} stroke={COLORS[1]} strokeWidth={2.6} />
      {/* candidates */}
      {SA_CANDS.map((p, i) => (
        <g key={i}>
          <circle cx={sx(p[0])} cy={sy(p[1])} r={i === chosen ? 6.5 : 4} fill={i === chosen ? 'var(--color-accent)' : 'var(--color-muted-bg)'} stroke="var(--color-border)" strokeWidth={1} />
          <text x={sx(p[0])} y={sy(p[1]) - 9} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">c{i}</text>
        </g>
      ))}
      {/* the datapoint x */}
      <circle cx={sx(SA_X[0])} cy={sy(SA_X[1])} r={5.5} fill="var(--color-text)" />
      <text x={sx(SA_X[0]) + 8} y={sy(SA_X[1]) + 4} fontSize={11} fontWeight={600} fill="var(--color-text)" fontFamily="var(--font-sans)">x</text>
      {/* legend */}
      <text x={sx(0)} y={sy(0.43)} fontSize={10} fill={COLORS[0]} fontFamily="var(--font-sans)">▬ parallel residual r∥ (moves ⟨q,x⟩)</text>
      <text x={sx(0)} y={sy(0.38)} fontSize={10} fill={COLORS[1]} fontFamily="var(--font-sans)">▬ orthogonal residual r⊥</text>
    </svg>
  );
}

export default function OptimizedProductQuantizationLaboratory() {
  const [panel, setPanel] = useState<'rotation' | 'converge' | 'scann'>('rotation');
  const [rotIdx, setRotIdx] = useState(0);
  const [etaIdx, setEtaIdx] = useState(0);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'rotation'
        ? '\\min_{R^{\\mathsf T}R=I,\\;\\{\\mathcal{C}^j\\}}\\ \\sum_i \\lVert Rx_i - c(Rx_i)\\rVert^2'
        : panel === 'converge'
          ? '\\text{R-step:}\\quad X^{\\mathsf T}Q = U\\Sigma V^{\\mathsf T}\\ \\Rightarrow\\ R = V U^{\\mathsf T}'
          : '\\ell(x,\\tilde x,\\eta) = \\eta\\,\\lVert r_\\parallel\\rVert^2 + \\lVert r_\\perp\\rVert^2,\\qquad \\eta \\ge 1';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const pill = (active: boolean) => ({
    fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
    border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
    background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
  });
  const bar = (v: number, max: number, color: string) => (
    <span style={{ flex: 1, height: '0.8rem', background: 'var(--color-muted-bg)', borderRadius: '999px', overflow: 'hidden' }}>
      <span style={{ display: 'block', height: '100%', width: `${Math.min(100, (v / max) * 100)}%`, background: color }} />
    </span>
  );

  const rot = ROT[Math.min(rotIdx, ROT.length - 1)] ?? ROT[0];
  const eta = ETA_GRID[Math.min(etaIdx, ETA_GRID.length - 1)] ?? ETA_GRID[0];

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.75rem' }}>
        <button onClick={() => setPanel('rotation')} style={pill(panel === 'rotation')}>The rotation</button>
        <button onClick={() => setPanel('converge')} style={pill(panel === 'converge')}>Alternating optimization</button>
        <button onClick={() => setPanel('scann')} style={pill(panel === 'scann')}>Score-aware quantization</button>
      </div>

      <div ref={formulaRef} style={{ overflowX: 'auto', marginBottom: '0.6rem' }} />

      {panel === 'rotation' && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.6rem' }}>
            {ROT.map((r, i) => (
              <button key={r.key} onClick={() => setRotIdx(i)} style={pill(i === rotIdx)}>{r.label}</button>
            ))}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '0.8rem' }}>
            <Readout label="PQ distortion ∑‖x−Q(x)‖²" value={rot.dist.toFixed(2)} accent={rot.key === 'opq'} />
            <Readout label="vs raw" value={`${(100 * (1 - rot.dist / ROT[0].dist)).toFixed(0)}% lower`} />
          </div>
          <div style={{ display: 'grid', gap: '1.25rem', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
            <div>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', color: 'var(--color-text-secondary)', marginBottom: '0.35rem' }}>per-subspace variance (m = 4) — OPQ balances the <em>product</em></div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontFamily: 'var(--font-sans)', fontSize: '0.78rem' }}>
                {rot.vars.map((v, j) => (
                  <div key={j} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ minWidth: '5.5rem' }}>subspace {j + 1}</span>{bar(v, VAR_MAX, COLORS[j])}<span style={{ minWidth: '2.6rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{v.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', color: 'var(--color-text-secondary)', marginBottom: '0.35rem' }}>distortion across rotations</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontFamily: 'var(--font-sans)', fontSize: '0.78rem' }}>
                {ROT.map((r, i) => (
                  <div key={r.key} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: i === rotIdx ? 1 : 0.6 }}>
                    <span style={{ minWidth: '7rem' }}>{r.label}</span>{bar(r.dist, DIST_MAX, i === rotIdx ? 'var(--color-accent)' : 'var(--color-text-secondary)')}<span style={{ minWidth: '3rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{r.dist.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.75rem 0 0' }}>
            A naive PCA-alignment barely helps (85.6 → 83.3): it concentrates variance into subspace 1 and starves the rest.
            Balancing variance across subspaces collapses the distortion (→ 15.4), and the learned OPQ rotation refines it
            further (→ 14.4) via the alternating optimization in the next panel. The balanced and OPQ variance bars look alike
            because OPQ's extra gain is a within-block rotation, not a different split.
          </div>
        </>
      )}

      {panel === 'converge' && (
        <>
          <ConvergencePlot />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.6rem' }}>
            <Readout label="start (parametric warm start)" value={TRAJECTORY[0].toFixed(2)} />
            <Readout label="converged OPQ distortion" value={TRAJECTORY[TRAJECTORY.length - 1].toFixed(2)} accent />
            <Readout label="vs balanced heuristic" value={`${(100 * (1 - TRAJECTORY[TRAJECTORY.length - 1] / BALANCED_REF)).toFixed(1)}% lower`} />
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.6rem 0 0' }}>
            Each iteration fixes the codebooks and moves R to the closed-form Orthogonal Procrustes optimum, then retrains the
            codebooks at the new rotation. Both subproblems are solved globally, so the distortion is <strong>monotone
            non-increasing</strong> — converging to a <em>local</em> optimum, not a certified global one (the honest caveat).
          </div>
        </>
      )}

      {panel === 'scann' && (
        <>
          <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', alignItems: 'start' }}>
            <ScoreAwareScene chosen={eta.chosen} />
            <div>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.85rem', marginBottom: '0.6rem' }}>
                <span style={{ minWidth: '3.5rem' }}>η = <strong>{eta.eta}</strong></span>
                <input type="range" min={0} max={ETA_GRID.length - 1} step={1} value={etaIdx}
                  onChange={(e) => setEtaIdx(parseInt(e.target.value, 10))}
                  aria-label="anisotropic weight eta"
                  style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
              </label>
              <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', marginBottom: '0.5rem', color: 'var(--color-text-secondary)' }}>
                chosen codeword: <strong style={{ color: 'var(--color-accent)' }}>c{eta.chosen}</strong>
                {eta.chosen === 1 ? ' — preserves the parallel component (zero r∥)' : ' — the Euclidean-closest'}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontFamily: 'var(--font-sans)', fontSize: '0.76rem' }}>
                {SA_CANDS.map((_, i) => {
                  const lo = eta.losses[Math.min(i, eta.losses.length - 1)] ?? 0;
                  const isMin = i === eta.chosen;
                  return (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: isMin ? 600 : 400, color: isMin ? 'var(--color-accent)' : 'var(--color-text)' }}>
                      <span style={{ minWidth: '5.5rem' }}>loss(x, c{i})</span>{bar(lo, 2, isMin ? 'var(--color-accent)' : 'var(--color-text-secondary)')}<span style={{ minWidth: '3rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{lo.toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.78rem', fontWeight: 600, margin: '0.9rem 0 0.4rem' }}>
            On the finance cloud: a score-aware codebook (η = {FIN.eta}) vs isotropic k-means, same size
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="high-score IP error — isotropic" value={FIN.isoMse.toFixed(5)} />
            <Readout label="high-score IP error — score-aware" value={`${FIN.anisoMse.toFixed(5)} (${(100 * (1 - FIN.anisoMse / FIN.isoMse)).toFixed(0)}% less)`} accent />
            <Readout label="MIPS recall@10" value={`${(FIN.isoRecall * 100).toFixed(1)}% → ${(FIN.anisoRecall * 100).toFixed(1)}%`} />
          </div>
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.6rem 0 0' }}>
            For an aligned (high-scoring) query, the inner-product error is dominated by the residual's component <em>parallel</em>
            to x. Weighting it by η &gt; 1 makes the quantizer prefer codewords that preserve that component — here the choice
            swaps from the Euclidean-closest c0 to c1, which lies exactly on x's direction (zero parallel residual). DIRECTION
            only: η and the score threshold are tuned, and the gain depends on the query distribution.
          </div>
        </>
      )}
    </div>
  );
}
