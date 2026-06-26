import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Matryoshka Laboratory — a granularity slider over three panels on nested
 * representations of a 1536-d financial cloud (n = 2000):
 *   - Prefix recall: recall@10 using only the first m coordinates, nested (PCA-ordered,
 *     the linear-MRL optimum) vs a random rotation of the SAME embedding. The rotation
 *     preserves every full-width distance but destroys the prefixes — nesting must be
 *     trained.
 *   - Nestedness: prefix reconstruction error vs m. The nested basis sits exactly on the
 *     Eckart-Young rank-m optimum (one ordered basis is optimal at every granularity),
 *     far below a random orthonormal basis.
 *   - Adaptive funnel: recall/cost Pareto — shortlist on a cheap 48-dim prefix, rerank
 *     the shortlist at full width; near-full recall at a fraction of the scoring cost.
 *
 * Every number is BAKED TO THE DECIMAL from
 * notebooks/matryoshka-nested-representations/matryoshka_nested_representations.py
 * (grid_table / finance_demo); test_linear_mrl_equals_pca / test_nested_beats_rotated /
 * test_funnel_retrieval / test_finance_funnel assert them. Change a number here -> change
 * it there, and re-run the notebook.
 */

// --- baked from grid_table() on the finance cloud (d=1536, n=2000, intrinsic 48, 3 clusters)
const GRAN = [24, 48, 96, 192, 384, 768, 1536];
const NESTED_RECALL = [0.7093, 0.9353, 0.9367, 0.9453, 0.944, 0.9653, 1.0];
const ROTATED_RECALL = [0.0427, 0.05, 0.12, 0.2313, 0.4253, 0.674, 1.0];
const RECON_NESTED = [7274.2, 1159.7, 1041.1, 842.5, 541.7, 187.4, 0.0]; // = Eckart-Young optimum
const RECON_RANDOM = [58870.3, 57883.8, 56004.0, 52306.0, 44539.4, 29860.7, 0.0];
const RECON_MAX = RECON_RANDOM[0];

// funnel Pareto at a fixed 48-dim prefix: sweep the shortlist size
const FUNNEL_SHORTLIST = [10, 15, 25, 50, 100, 200];
const FUNNEL_RECALL = [0.9353, 0.9987, 1.0, 1.0, 1.0, 1.0];
const FUNNEL_COST = [0.0362, 0.0387, 0.0437, 0.0563, 0.0813, 0.1313];

type Panel = 'recall' | 'nested' | 'funnel';

const PLOT_W = 540;
const PLOT_H = 210;
const PAD = 32;

const LOG_MMIN = Math.log(GRAN[0]);
const LOG_MMAX = Math.log(GRAN[GRAN.length - 1]);
const mx = (m: number) => PAD + ((PLOT_W - 2 * PAD) * (Math.log(m) - LOG_MMIN)) / (LOG_MMAX - LOG_MMIN);
const vy = (v: number, lo: number, hi: number) =>
  PAD + (PLOT_H - 2 * PAD) * (1 - Math.min(1, Math.max(0, (v - lo) / (hi - lo))));

function mCurve(vals: number[], lo: number, hi: number): string {
  return vals.map((v, i) => (i === 0 ? 'M' : 'L') + mx(GRAN[i]).toFixed(1) + ' ' + vy(v, lo, hi).toFixed(1)).join(' ');
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
  { id: 'recall', label: 'Prefix recall' },
  { id: 'nested', label: 'Nestedness' },
  { id: 'funnel', label: 'Adaptive funnel' },
];

const mticks = [24, 96, 384, 1536];

function MAxis() {
  return (
    <>
      <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
      {mticks.map((m) => (
        <text key={m} x={mx(m)} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10}
          fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{m}</text>
      ))}
    </>
  );
}

export default function MatryoshkaLaboratory() {
  const [panel, setPanel] = useState<Panel>('recall');
  const [mIdx, setMIdx] = useState(2); // GRAN index; start at m = 96
  const m = GRAN[mIdx];
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'recall'
        ? `\\text{recall@10 using } z_{1:m}\\ \\text{of}\\ z\\in\\mathbb{R}^{1536}`
        : panel === 'nested'
        ? `\\lVert \\tilde X-\\tilde X V_{:m}V_{:m}^{\\top}\\rVert_F^2 = \\sum_{i>m}\\sigma_i^2\\ \\text{(Eckart–Young)}`
        : `\\text{shortlist on } z_{1:48},\\ \\text{rerank at full width}`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  const recallPlot = useMemo(() => (
    <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {[0, 0.5, 1].map((v) => (
        <text key={v} x={PAD - 6} y={vy(v, 0, 1) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v === 0 ? '0' : `${v * 100}%`}</text>
      ))}
      <line x1={mx(m)} y1={PAD - 6} x2={mx(m)} y2={PLOT_H - PAD} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
      <path d={mCurve(ROTATED_RECALL, 0, 1)} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.5} strokeLinejoin="round" />
      <path d={mCurve(NESTED_RECALL, 0, 1)} fill="none" stroke="var(--color-accent)" strokeWidth={2.4} strokeLinejoin="round" />
      <MAxis />
      <text x={mx(1536)} y={vy(NESTED_RECALL[1], 0, 1) - 6} textAnchor="end" fontSize={9} fill="var(--color-accent)" fontFamily="var(--font-sans)">nested (Matryoshka)</text>
      <text x={mx(768)} y={vy(ROTATED_RECALL[5], 0, 1) + 14} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rotated (no nesting)</text>
    </svg>
  ), [m]);

  const nestedPlot = useMemo(() => (
    <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={mx(m)} y1={PAD - 6} x2={mx(m)} y2={PLOT_H - PAD} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
      <path d={mCurve(RECON_RANDOM.map((v) => v / RECON_MAX), 0, 1)} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.5} strokeLinejoin="round" />
      <path d={mCurve(RECON_NESTED.map((v) => v / RECON_MAX), 0, 1)} fill="none" stroke="var(--color-accent)" strokeWidth={2.4} strokeLinejoin="round" />
      <MAxis />
      <text x={mx(384)} y={vy(RECON_RANDOM[4] / RECON_MAX, 0, 1) - 6} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">random basis</text>
      <text x={mx(192)} y={vy(RECON_NESTED[3] / RECON_MAX, 0, 1) - 8} textAnchor="middle" fontSize={9} fill="var(--color-accent)" fontFamily="var(--font-sans)">nested = optimum</text>
    </svg>
  ), [m]);

  const funnelPlot = useMemo(() => {
    const lo = 0.9, hi = 1.005, cmax = 0.15;
    const fx = (c: number) => PAD + ((PLOT_W - 2 * PAD) * c) / cmax;
    const pts = FUNNEL_COST.map((c, i) => `${fx(c).toFixed(1)},${vy(FUNNEL_RECALL[i], lo, hi).toFixed(1)}`);
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {[0.9, 0.95, 1.0].map((v) => (
          <text key={v} x={PAD - 6} y={vy(v, lo, hi) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{`${(v * 100).toFixed(0)}%`}</text>
        ))}
        <polyline points={pts.join(' ')} fill="none" stroke="var(--color-accent)" strokeWidth={2} strokeLinejoin="round" />
        {FUNNEL_COST.map((c, i) => (
          <g key={i}>
            <circle cx={fx(c)} cy={vy(FUNNEL_RECALL[i], lo, hi)} r={3.6} fill="var(--color-accent)" />
            <text x={fx(c)} y={vy(FUNNEL_RECALL[i], lo, hi) - 8} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{FUNNEL_SHORTLIST[i]}</text>
          </g>
        ))}
        <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.05, 0.1, 0.15].map((c) => (
          <text key={c} x={fx(c)} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{`${(c * 100).toFixed(0)}%`}</text>
        ))}
        <text x={PLOT_W - PAD} y={PLOT_H - PAD + 14} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">scoring cost</text>
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

      {panel !== 'funnel' && (
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', marginBottom: '1rem' }}>
          <span style={{ minWidth: '11rem' }}>prefix width <strong>{m}</strong> of 1536 dims</span>
          <input type="range" min={0} max={GRAN.length - 1} step={1} value={mIdx}
            onChange={(e) => setMIdx(parseInt(e.target.value, 10))}
            style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
        </label>
      )}

      {panel === 'recall' && (
        <>
          {recallPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Recall@10 using only the first m coordinates: the nested (Matryoshka / PCA-ordered) embedding (accent) degrades gracefully, while a random
            rotation of the <em>same</em> embedding (faint) — identical full-width distances — retrieves near chance until almost every dimension is kept. Nesting is trained, not free.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`nested recall@10 at m = ${m}`} value={`${(NESTED_RECALL[mIdx] * 100).toFixed(1)}%`} accent />
            <Readout label="rotated (no nesting)" value={`${(ROTATED_RECALL[mIdx] * 100).toFixed(1)}%`} />
            <Readout label="kept / full" value={`${m} / 1536`} />
          </div>
        </>
      )}

      {panel === 'nested' && (
        <>
          {nestedPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Prefix reconstruction error vs m (normalized). The nested basis (accent) sits exactly on the Eckart–Young rank-m optimum at every granularity —
            one eigenvalue-ordered basis is simultaneously optimal at all widths — far below a random orthonormal basis (faint). That coincidence is the theorem: linear Matryoshka is PCA.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`nested error² at m = ${m}`} value={RECON_NESTED[mIdx].toFixed(0)} accent />
            <Readout label="random basis error²" value={RECON_RANDOM[mIdx].toFixed(0)} />
            <Readout label="= Eckart–Young optimum?" value="yes" accent />
          </div>
        </>
      )}

      {panel === 'funnel' && (
        <>
          {funnelPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Adaptive retrieval: shortlist candidates with a cheap 48-dim prefix, then rerank the shortlist at full 1536-d width. Each point is a shortlist size (labeled);
            recall climbs to full while the scoring cost stays a small fraction of an exhaustive full-width scan. The knee — near-full recall, tiny cost — is the Matryoshka payoff.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="recall @ shortlist 15" value={`${(FUNNEL_RECALL[1] * 100).toFixed(1)}%`} accent />
            <Readout label="cost @ shortlist 15" value={`${(FUNNEL_COST[1] * 100).toFixed(1)}%`} />
            <Readout label="vs exhaustive full-width" value="100% cost" />
          </div>
        </>
      )}
    </div>
  );
}
