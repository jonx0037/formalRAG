import { useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Chunking Laboratory — a segment-count slider over three panels on optimal
 * segmentation of a 32-sentence document (5 planted von Mises-Fisher topic blocks):
 *   - Document & boundaries: a sentence strip colored by planted segment, the adjacent-gap
 *     dissimilarity profile, and DP-optimal / greedy / fixed-size / truth boundary markers.
 *     At the true k=5 the DP lands exactly on the planted boundaries.
 *   - Granularity tradeoff: DP coherence cost (monotone down) and DP boundary-F1 (humped,
 *     peaking at the true k) vs k — coherence is a proxy, so the optimum of the objective
 *     is not the optimum of the structure.
 *   - Boundary recovery: F1 bars, DP vs greedy vs fixed-size, at the selected k.
 *
 * Every number is BAKED TO THE DECIMAL from
 * notebooks/chunking-as-segmentation/chunking_as_segmentation.py (grid_table / finance_demo);
 * test_dp_matches_brute_force / test_cost_monotone_in_k / test_dp_recovers_boundaries assert
 * them. Change a number here -> change it there, and re-run the notebook.
 */

// --- baked from grid_table() on the 32-sentence synthetic document (5 vMF sections)
const N = 32;
const LABELS = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4];
const TRUTH = [4, 12, 17, 27];
const FIXED = [6, 13, 19, 26];
const PROFILE = [0.771, 0.854, 0.767, 0.877, 0.749, 0.748, 0.795, 0.818, 0.734, 0.833, 0.876, 1.068, 0.892, 0.718, 0.549, 0.844, 1.056, 0.887, 0.672, 0.54, 0.594, 0.624, 0.668, 0.614, 0.756, 0.763, 0.865, 0.948, 0.762, 0.69, 0.676];

const K_GRID = [2, 3, 4, 5, 6, 8, 10, 12];
const DP_COST = [20.447, 18.196, 16.233, 14.783, 13.786, 12.216, 10.781, 9.47];
const DP_F1 = [0.4, 0.667, 0.857, 1.0, 0.889, 0.727, 0.615, 0.533];
const GREEDY_COST = [20.752, 18.484, 16.438, 15.946, 15.533, 13.578, 12.065, 10.619];
const GREEDY_F1 = [0.4, 0.667, 0.857, 0.75, 0.667, 0.727, 0.615, 0.533];
const COST_MAX = DP_COST[0] * 1.05;

// DP and greedy internal boundaries at each K_GRID value (for the strip overlay)
const DP_BOUNDS: number[][] = [
  [17], [17, 27], [12, 17, 27], [4, 12, 17, 27], [4, 7, 12, 17, 27],
  [4, 7, 12, 17, 23, 25, 28], [2, 4, 7, 10, 12, 17, 23, 25, 28], [2, 4, 7, 10, 12, 17, 19, 23, 25, 27, 30],
];
const GREEDY_BOUNDS: number[][] = [
  [12], [12, 17], [12, 17, 28], [12, 13, 17, 28], [12, 13, 17, 18, 28],
  [4, 11, 12, 13, 17, 18, 28], [2, 4, 11, 12, 13, 17, 18, 27, 28], [2, 4, 10, 11, 12, 13, 16, 17, 18, 27, 28],
];

const TRUE_KIDX = 3; // k = 5
const FIXED_F1 = 0.5;
const FINANCE = { n: 42, sections: 6, dpF1: 0.8, greedyF1: 0.8, fixedF1: 0.2 };

const SEG_COLORS = ['var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)', 'var(--color-text-secondary)', 'var(--color-accent)', 'var(--color-accent-secondary, #d98a3d)'];

type Panel = 'strip' | 'tradeoff' | 'recovery';

const PLOT_W = 540;
const PLOT_H = 210;
const PAD = 32;

const sx = (s: number) => PAD + ((PLOT_W - 2 * PAD) * s) / N;
const vy = (v: number, lo: number, hi: number) => PAD + (PLOT_H - 2 * PAD) * (1 - Math.min(1, Math.max(0, (v - lo) / (hi - lo))));

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

const PANELS: { id: Panel; label: string }[] = [
  { id: 'strip', label: 'Document & boundaries' },
  { id: 'tradeoff', label: 'Granularity tradeoff' },
  { id: 'recovery', label: 'Boundary recovery' },
];

function BoundaryMarks({ bounds, y0, y1, color, dash, width }: { bounds: number[]; y0: number; y1: number; color: string; dash?: string; width?: number }) {
  return <>{bounds.map((b) => <line key={b} x1={sx(b)} y1={y0} x2={sx(b)} y2={y1} stroke={color} strokeWidth={width ?? 1.5} strokeDasharray={dash} />)}</>;
}

export default function ChunkingLaboratory() {
  const [panel, setPanel] = useState<Panel>('strip');
  const [kIdx, setKIdx] = useState(TRUE_KIDX); // K_GRID index; start at the true k = 5
  const k = K_GRID[kIdx];
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'strip'
        ? `\\mathrm{OPT}(j)=\\min_{i<j}\\,\\mathrm{OPT}(i)+c(i,j),\\quad c(i,j)=(j-i)-\\big\\lVert{\\textstyle\\sum} e_t\\big\\rVert`
        : panel === 'tradeoff'
        ? `c(i,j)=(j-i)(1-\\bar R)\\ \\downarrow\\ \\text{in }k,\\quad \\text{but recovery peaks at the true }k`
        : `\\text{boundary }F_1\\ \\text{vs planted topic shifts}`;
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  // --- Panel: document strip + dissimilarity profile + boundary marks --------------
  const stripPlot = useMemo(() => {
    const stripY0 = PAD - 4, stripH = 26;
    const profTop = stripY0 + stripH + 14, profBot = PLOT_H - PAD;
    const pmax = Math.max(...PROFILE);
    const profPath = PROFILE.map((v, i) => `${i === 0 ? 'M' : 'L'}${((sx(i + 0.5) + sx(i + 1.5)) / 2).toFixed(1)} ${(profBot - (profBot - profTop) * (v / pmax)).toFixed(1)}`).join(' ');
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {/* sentence strip colored by planted segment */}
        {LABELS.map((lab, s) => (
          <rect key={s} x={sx(s)} y={stripY0} width={sx(s + 1) - sx(s) - 0.5} height={stripH} fill={SEG_COLORS[lab % SEG_COLORS.length]} opacity={0.32} />
        ))}
        {/* truth boundaries across the strip */}
        <BoundaryMarks bounds={TRUTH} y0={stripY0 - 6} y1={stripY0 + stripH + 6} color="var(--color-text)" dash="2 2" width={1.3} />
        {/* dissimilarity profile */}
        <path d={profPath} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.3} strokeLinejoin="round" opacity={0.8} />
        {/* method boundaries on the profile band */}
        <BoundaryMarks bounds={DP_BOUNDS[kIdx]} y0={profTop} y1={profBot} color="var(--color-accent)" width={2} />
        <BoundaryMarks bounds={GREEDY_BOUNDS[kIdx]} y0={profTop} y1={profBot} color="var(--color-accent-secondary, #d98a3d)" dash="3 3" width={1.4} />
        {kIdx === TRUE_KIDX && <BoundaryMarks bounds={FIXED} y0={profTop} y1={profBot} color="var(--color-border)" dash="1 3" width={1.2} />}
        <line x1={PAD} y1={profBot} x2={PLOT_W - PAD} y2={profBot} stroke="var(--color-border)" strokeWidth={1} />
        <text x={PAD} y={stripY0 - 9} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">sentences (colored by topic) · dashed = true boundaries</text>
        <text x={PAD} y={profBot + 13} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">gap dissimilarity · accent = DP optimum · dashed orange = greedy</text>
      </svg>
    );
  }, [kIdx]);

  // --- Panel: cost (down) and F1 (humped) vs k -------------------------------------
  const tradeoffPlot = useMemo(() => {
    const tx = (i: number) => PAD + ((PLOT_W - 2 * PAD) * i) / (K_GRID.length - 1);
    const costPath = DP_COST.map((v, i) => `${i === 0 ? 'M' : 'L'}${tx(i).toFixed(1)} ${vy(v / COST_MAX, 0, 1).toFixed(1)}`).join(' ');
    const f1Path = DP_F1.map((v, i) => `${i === 0 ? 'M' : 'L'}${tx(i).toFixed(1)} ${vy(v, 0, 1).toFixed(1)}`).join(' ');
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={tx(kIdx)} y1={PAD - 6} x2={tx(kIdx)} y2={PLOT_H - PAD} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
        <line x1={tx(TRUE_KIDX)} y1={PAD - 6} x2={tx(TRUE_KIDX)} y2={PLOT_H - PAD} stroke="var(--color-accent)" strokeWidth={1} strokeDasharray="1 3" opacity={0.5} />
        <path d={costPath} fill="none" stroke="var(--color-text-secondary)" strokeWidth={1.6} strokeLinejoin="round" />
        <path d={f1Path} fill="none" stroke="var(--color-accent)" strokeWidth={2.4} strokeLinejoin="round" />
        {DP_F1.map((v, i) => <circle key={i} cx={tx(i)} cy={vy(v, 0, 1)} r={2.6} fill="var(--color-accent)" />)}
        <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
        {K_GRID.map((kk, i) => <text key={kk} x={tx(i)} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{kk}</text>)}
        <text x={tx(1)} y={vy(DP_COST[1] / COST_MAX, 0, 1) - 6} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">coherence cost ↓</text>
        <text x={tx(TRUE_KIDX)} y={vy(1, 0, 1) - 4} textAnchor="middle" fontSize={9} fill="var(--color-accent)" fontFamily="var(--font-sans)">F1 peaks at true k</text>
        <text x={PLOT_W - PAD} y={PLOT_H - PAD + 14} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">#segments k</text>
      </svg>
    );
  }, [kIdx]);

  // --- Panel: F1 bars DP/greedy/fixed at current k ---------------------------------
  const recoveryPlot = useMemo(() => {
    const bars = [['DP optimum', DP_F1[kIdx], 'var(--color-accent)'], ['greedy', GREEDY_F1[kIdx], 'var(--color-accent-secondary, #d98a3d)'], ['fixed-size', kIdx === TRUE_KIDX ? FIXED_F1 : null, 'var(--color-text-secondary)']] as const;
    const x0 = PAD + 30, x1 = PLOT_W - PAD - 20, hiV = 1.0;
    const by = (v: number) => PLOT_H - PAD - ((PLOT_H - 2 * PAD) * v) / hiV;
    const slot = (x1 - x0) / bars.length;
    return (
      <svg viewBox={`0 0 ${PLOT_W} ${PLOT_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {[0, 0.5, 1].map((v) => <text key={v} x={PAD - 6} y={by(v) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{`${v * 100}%`}</text>)}
        {bars.map(([name, v, color], i) => {
          const cx = x0 + slot * (i + 0.5);
          if (v === null) return <text key={name} x={cx} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{name} (k=5)</text>;
          return (
            <g key={name}>
              <rect x={cx - slot * 0.3} y={by(v)} width={slot * 0.6} height={PLOT_H - PAD - by(v)} fill={color} opacity={0.78} />
              <text x={cx} y={PLOT_H - PAD + 14} textAnchor="middle" fontSize={10} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{name}</text>
              <text x={cx} y={by(v) - 5} textAnchor="middle" fontSize={10} fill={color} fontFamily="var(--font-sans)">{`${(v * 100).toFixed(0)}%`}</text>
            </g>
          );
        })}
        <line x1={PAD} y1={PLOT_H - PAD} x2={PLOT_W - PAD} y2={PLOT_H - PAD} stroke="var(--color-border)" strokeWidth={1} />
      </svg>
    );
  }, [kIdx]);

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

      <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', marginBottom: '1rem' }}>
        <span style={{ minWidth: '9rem' }}>split into <strong>{k}</strong> chunks</span>
        <input type="range" min={0} max={K_GRID.length - 1} step={1} value={kIdx}
          onChange={(e) => setKIdx(parseInt(e.target.value, 10))}
          style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
      </label>

      {panel === 'strip' && (
        <>
          {stripPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            The document as a sentence strip colored by its planted topic, with the adjacent-gap dissimilarity below. At the true k = 5 the DP-optimal boundaries (accent)
            land exactly on the topic shifts; the greedy boundaries (dashed orange) chase local dissimilarity peaks and the fixed-size cuts ignore meaning.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="DP boundary F1" value={`${(DP_F1[kIdx] * 100).toFixed(0)}%`} accent />
            <Readout label="greedy boundary F1" value={`${(GREEDY_F1[kIdx] * 100).toFixed(0)}%`} />
            <Readout label="true # sections" value="5" />
          </div>
        </>
      )}

      {panel === 'tradeoff' && (
        <>
          {tradeoffPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            DP coherence cost (faint, normalized) falls monotonically as you split more — but boundary-recovery F1 (accent) <em>peaks at the true k = 5</em> and then declines under over-segmentation.
            Minimizing the objective past the true count buys lower incoherence and worse structure: coherence is a proxy for the real target.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label={`DP cost at k = ${k}`} value={DP_COST[kIdx].toFixed(2)} />
            <Readout label={`DP F1 at k = ${k}`} value={`${(DP_F1[kIdx] * 100).toFixed(0)}%`} accent />
            <Readout label="F1 maximized at" value="k = 5 (true)" accent />
          </div>
        </>
      )}

      {panel === 'recovery' && (
        <>
          {recoveryPlot}
          <div style={{ fontFamily: 'var(--font-sans)', fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.75rem' }}>
            Boundary-recovery F1 against the planted topic shifts, at the chosen k. The DP optimum is best, the greedy heuristic close behind, and fixed-size chunking — shown at the true k = 5 —
            misses badly because the sections are uneven. Equal-size chunking only hits boundaries by accident.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
            <Readout label="DP F1 (finance 10-K)" value={`${(FINANCE.dpF1 * 100).toFixed(0)}%`} accent />
            <Readout label="greedy F1 (10-K)" value={`${(FINANCE.greedyF1 * 100).toFixed(0)}%`} />
            <Readout label="fixed-size F1 (10-K)" value={`${(FINANCE.fixedF1 * 100).toFixed(0)}%`} />
          </div>
        </>
      )}
    </div>
  );
}
