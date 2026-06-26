import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Set-Metrics Laboratory — four panels for the `set-metrics-precision-recall-map-mrr` topic:
 *   A. Precision & recall at a cutoff. A draggable rank cutoff k over the worked query's ranking, slots
 *      colored TP / FP / FN; P@k climbs nowhere-decreasing while precision wobbles.
 *   B. The PR curve & Average Precision. AP is literally the area under the raw PR curve (sum of the
 *      recall-step bars); the interpolated monotone envelope's area is >= raw AP — a convention that
 *      inflates the number (the rigorFlag panel).
 *   C. MAP / MRR across queries. Per-query AP (or RR) as a strip with the mean line; in the known-item
 *      regime AP = RR per query so MAP = MRR exactly (the collapse anchor).
 *   D. Metrics as estimators. A query-count slider shrinks the 95% CI like 1/sqrt(n); dense and
 *      late-interaction overlap at small n and separate at n = 12 — the significance-testing cliffhanger.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): REL_RANKS, PER_Q_AP, PER_Q_RR, MAP, MRR, AP_STD, SE_SCALING,
 * SEP_N, and FLIP are mirrored TO THE DECIMAL from notebooks/set-metrics-precision-recall-map-mrr/
 * set_metrics_precision_recall_map_mrr.py (viz_constants()). Matching asserts: test_ap_equals_area_under_pr
 * / test_interpolated_ge_raw_and_monotone / test_map_is_mean_of_ap / test_map_equals_mrr_known_item /
 * test_se_scales_as_inv_sqrt_n / test_two_leg_overlap_motivates_significance / test_metric_choice_flips_verdict.
 * The lab recomputes ONLY closed forms in TS (precision/recall, the PR sawtooth + AP area, the
 * interpolated envelope, the sample mean/std, and the projected CI 1.96*std/sqrt(n)); the rng/k-means-
 * driven scalars (the per-query arrays, MAP/MRR/std, the empirical SE) are baked. Change a number here ->
 * change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_DOCS = 120;
const R_SIZE = 10;
const N_QUERIES = 40;
const WORKED_Q = 24;

type Leg = 'lexical' | 'dense' | 'late_interaction';
const LEG_NAMES: Leg[] = ['lexical', 'dense', 'late_interaction'];
const LEG_LABEL: Record<Leg, string> = {
  lexical: 'lexical (BM25)', dense: 'dense (MIPS)', late_interaction: 'late interaction',
};
const LEG_COLOR: Record<Leg, string> = {
  lexical: '#c08457', dense: '#6a8caf', late_interaction: '#9b6abf',
};

// Panel A & B — the worked query's relevant-doc rank positions (1-indexed) in each leg's full ranking.
const REL_RANKS: Record<Leg, number[]> = {
  lexical: [1, 2, 3, 4, 7, 8, 15, 18, 19, 23],
  dense: [1, 2, 3, 4, 5, 7, 11, 12, 29, 72],
  late_interaction: [1, 2, 3, 4, 5, 7, 8, 9, 10, 26],
};

// Panel C — per-query AP (set regime, |R| = 10) and per-query RR (known-item regime, |R| = 1).
const PER_Q_AP: Record<Leg, number[]> = {
  lexical: [0.511, 0.479, 0.313, 0.622, 0.471, 0.678, 0.562, 0.681, 0.708, 0.697, 0.76, 0.485, 0.736, 0.733, 0.92, 0.732, 0.758, 0.355, 0.689, 0.538, 0.567, 0.643, 0.143, 0.277, 0.728, 0.321, 0.591, 0.313, 0.66, 0.483, 0.742, 0.904, 0.714, 0.848, 0.807, 0.751, 0.682, 0.7, 0.745, 0.71],
  dense: [0.942, 0.854, 0.826, 0.563, 0.807, 0.652, 0.745, 0.685, 0.822, 0.668, 0.685, 0.857, 0.832, 0.809, 0.971, 0.838, 0.857, 0.565, 0.814, 0.651, 0.482, 0.459, 0.383, 0.694, 0.761, 0.674, 0.881, 0.748, 0.806, 0.657, 0.615, 0.819, 0.658, 0.811, 0.94, 0.84, 0.808, 0.898, 0.452, 0.573],
  late_interaction: [0.963, 0.828, 0.905, 0.672, 0.949, 0.871, 0.883, 0.791, 0.843, 0.872, 0.67, 0.933, 0.926, 0.738, 0.971, 0.932, 0.914, 0.961, 0.946, 0.875, 0.897, 0.946, 0.586, 0.687, 0.891, 0.783, 0.798, 1.0, 0.926, 0.821, 0.81, 0.967, 0.83, 0.961, 0.944, 0.917, 0.897, 1.0, 1.0, 0.859],
};
const PER_Q_RR: Record<Leg, number[]> = {
  lexical: [1.0, 1.0, 1.0, 0.333, 0.5, 1.0, 0.111, 0.333, 0.5, 1.0, 0.25, 0.5, 0.5, 0.2, 0.5, 0.167, 0.5, 1.0, 1.0, 1.0, 0.25, 0.25, 0.143, 0.05, 0.333, 0.02, 0.2, 1.0, 0.2, 1.0, 0.5, 1.0, 0.5, 0.143, 0.5, 1.0, 0.5, 0.056, 0.5, 0.333],
  dense: [0.5, 1.0, 0.333, 0.25, 1.0, 1.0, 0.083, 0.333, 0.25, 1.0, 0.1, 0.167, 0.333, 0.167, 0.111, 0.25, 0.333, 1.0, 0.5, 1.0, 0.03, 0.5, 0.25, 1.0, 0.091, 0.143, 0.5, 0.333, 0.2, 0.333, 1.0, 1.0, 0.077, 1.0, 0.2, 0.333, 0.5, 0.167, 1.0, 0.5],
  late_interaction: [0.333, 1.0, 1.0, 0.25, 0.5, 0.5, 0.167, 0.333, 0.167, 1.0, 1.0, 0.333, 1.0, 0.2, 0.167, 0.2, 0.25, 1.0, 0.5, 0.5, 0.333, 0.25, 1.0, 1.0, 0.5, 0.5, 0.333, 1.0, 0.25, 0.5, 0.5, 0.333, 0.5, 0.125, 1.0, 0.2, 1.0, 0.111, 1.0, 0.333],
};

// Panel C/D readouts (mirrored; the lab also recomputes MAP/MRR/std from the arrays as a closed-form check).
const MAP: Record<Leg, number> = { lexical: 0.619, dense: 0.7351, late_interaction: 0.8741 };
const MRR: Record<Leg, number> = { lexical: 0.5218, dense: 0.4717, late_interaction: 0.5292 };

// Panel D — the SE ~ 1/sqrt(n) demonstration (dense): [n, empirical SE]; the theory SE = std/sqrt(n) is
// recomputed in TS. The dense/late-interaction projected 95% CIs separate at SEP_N queries.
const SE_SCALING: [number, number][] = [[5, 0.0651], [10, 0.0435], [20, 0.0324], [40, 0.0228], [80, 0.0159]];
const SEP_N = 12;
// The metric-choice flip: a pairwise reversal — dense beats lexical on MAP but loses on MRR.
const FLIP = { pair: ['lexical', 'dense'] as Leg[], mapWinner: 'dense' as Leg, mrrWinner: 'lexical' as Leg };

const ACCENT = 'var(--color-accent)';
const TP_COLOR = '#5fa873';      // true positive (relevant, retrieved)
const FP_COLOR = '#9aa3ad';      // false positive (retrieved, not relevant)
const FN_COLOR = '#c25b6b';      // false negative (relevant, missed)
const ENV_COLOR = '#c79a3b';     // interpolated envelope
const MUTED = '#9aa3ad';

const fmt = (x: number, n = 3) => x.toFixed(n);

// --- closed-form TS recomputation -----------------------------------------------------
const countLE = (ranks: number[], k: number) => ranks.reduce((s, r) => s + (r <= k ? 1 : 0), 0);
const precisionAt = (ranks: number[], k: number) => (k <= 0 ? 0 : countLE(ranks, k) / k);
const recallAt = (ranks: number[], k: number) => countLE(ranks, k) / R_SIZE;
const f1At = (ranks: number[], k: number) => {
  const p = precisionAt(ranks, k), r = recallAt(ranks, k);
  return p + r <= 0 ? 0 : (2 * p * r) / (p + r);
};
// AP = (1/|R|) Σ_i i/pos_i  ==  Σ_i (R_i - R_{i-1}) P_i  (each hit advances recall by 1/|R|).
const apFromRanks = (ranks: number[]) => ranks.reduce((s, pos, i) => s + (i + 1) / pos, 0) / R_SIZE;
// raw PR "bars": [recall_lo, recall_hi, precision] per relevant hit; their areas sum to AP.
const prBars = (ranks: number[]) =>
  ranks.map((pos, i) => [i / R_SIZE, (i + 1) / R_SIZE, (i + 1) / pos] as [number, number, number]);
// interpolated envelope height per bar = max precision at recall >= that bar's level (monotone).
const envBars = (ranks: number[]) => {
  const P = ranks.map((pos, i) => (i + 1) / pos);
  return ranks.map((_pos, i) => [i / R_SIZE, (i + 1) / R_SIZE, Math.max(...P.slice(i))] as [number, number, number]);
};
const interpAp = (ranks: number[]) => {
  const P = ranks.map((pos, i) => (i + 1) / pos);
  return P.reduce((s, _p, i) => s + Math.max(...P.slice(i)), 0) / R_SIZE;
};
const mean = (a: number[]) => (a.length === 0 ? 0 : a.reduce((s, x) => s + x, 0) / a.length);
const stdev = (a: number[]) => {
  if (a.length <= 1) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1));
};
const ciHalf = (sd: number, n: number) => (n > 0 ? (1.96 * sd) / Math.sqrt(n) : 0);

// --- shared UI atoms ------------------------------------------------------------------
function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}
function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display: string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '11rem' }}>{label} = <strong>{display}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
    </label>
  );
}
const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});
function LegPills({ leg, setLeg }: { leg: Leg; setLeg: (l: Leg) => void }) {
  return (
    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
      {LEG_NAMES.map((l) => (
        <button key={l} type="button" style={pill(leg === l)} onClick={() => setLeg(l)}>{LEG_LABEL[l]}</button>
      ))}
    </div>
  );
}

// ===== Panel A — precision & recall at a cutoff =====================================================
const N_SLOTS = 25;
function CutoffPanel({ leg, setLeg, k, setK }: { leg: Leg; setLeg: (l: Leg) => void; k: number; setK: (v: number) => void }) {
  const ranks = REL_RANKS[leg];
  const relSet = new Set(ranks);
  const W = 520, cell = 18, gap = 2, x0 = 8, y0 = 28, H = 92;
  const cutX = x0 + k * (cell + gap);
  const p = precisionAt(ranks, k), r = recallAt(ranks, k), f1 = f1At(ranks, k);
  const tp = countLE(ranks, k);
  return (
    <div>
      <LegPills leg={leg} setLeg={setLeg} />
      <Slider label="rank cutoff k" value={k} min={1} max={N_SLOTS} step={1} onChange={(v) => setK(Math.round(v))} display={`${k}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="ranking slots colored true/false positive and false negative at the cutoff" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {Array.from({ length: N_SLOTS }, (_v, idx) => {
          const rank = idx + 1;
          const within = rank <= k;
          const rel = relSet.has(rank);
          const fill = within ? (rel ? TP_COLOR : FP_COLOR) : 'transparent';
          const stroke = rel ? (within ? TP_COLOR : FN_COLOR) : 'var(--color-border)';
          return (
            <g key={rank}>
              <rect x={x0 + idx * (cell + gap)} y={y0} width={cell} height={cell} rx={3}
                fill={fill} fillOpacity={within && !rel ? 0.5 : 0.85} stroke={stroke} strokeWidth={rel ? 2 : 1} />
              {rank % 5 === 0 && (
                <text x={x0 + idx * (cell + gap) + cell / 2} y={y0 + cell + 12} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{rank}</text>
              )}
            </g>
          );
        })}
        <line x1={cutX} y1={y0 - 8} x2={cutX} y2={y0 + cell + 6} stroke={ACCENT} strokeWidth={2} />
        <text x={cutX} y={y0 - 12} textAnchor="middle" fontSize={9} fill={ACCENT} fontFamily="var(--font-sans)">k = {k}</text>
        <text x={x0} y={14} fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rank 1 → (worked query {WORKED_Q}); ▮ TP relevant&retrieved · ▯ FP retrieved · ▯ FN missed-relevant</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label={`relevant in top-${k}`} value={`${tp} / ${R_SIZE}`} />
        <Readout label="precision@k = TP/k" value={fmt(p)} accent />
        <Readout label="recall@k = TP/|R|" value={fmt(r)} accent />
        <Readout label="F1@k" value={fmt(f1)} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Slide k right: <strong>recall never falls</strong> (each new relevant doc can only add coverage), while
        <strong> precision wobbles</strong> — it jumps up when slot k is relevant and decays when it is not. Recall reaches
        1.0 once all {R_SIZE} relevant docs are passed; precision at that point is only {R_SIZE}/(last relevant rank).
      </p>
    </div>
  );
}

// ===== Panel B — the PR curve & Average Precision ===================================================
function PRPanel({ leg, setLeg, showInterp, setShowInterp }: {
  leg: Leg; setLeg: (l: Leg) => void; showInterp: boolean; setShowInterp: (v: boolean) => void;
}) {
  const ranks = REL_RANKS[leg];
  const W = 360, H = 280, padL = 44, padR = 14, padT = 14, padB = 38;
  const px = (recall: number) => padL + (W - padL - padR) * recall;
  const py = (prec: number) => H - padB - (H - padT - padB) * prec;
  const raw = prBars(ranks), env = envBars(ranks);
  const ap = apFromRanks(ranks), iap = interpAp(ranks);
  // 11-point interpolated dots
  const P = ranks.map((pos, i) => (i + 1) / pos);
  const Rc = ranks.map((_pos, i) => (i + 1) / R_SIZE);
  const elevens = Array.from({ length: 11 }, (_v, j) => {
    const t = j / 10;
    const cand = P.filter((_p, idx) => Rc[idx] >= t - 1e-9);
    return [t, cand.length ? Math.max(...cand) : 0] as [number, number];
  });
  return (
    <div>
      <LegPills leg={leg} setLeg={setLeg} />
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <button type="button" style={pill(!showInterp)} onClick={() => setShowInterp(false)}>raw PR curve</button>
        <button type="button" style={pill(showInterp)} onClick={() => setShowInterp(true)}>+ interpolated envelope</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="precision-recall curve with the area equal to average precision" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* axes */}
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <text x={padL - 6} y={py(v) + 3} textAnchor="end" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>
            <text x={px(v)} y={H - padB + 13} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>
          </g>
        ))}
        <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">recall</text>
        <text x={12} y={(padT + H - padB) / 2} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + H - padB) / 2})`}>precision</text>
        {/* raw AP bars (their areas sum to AP) */}
        {raw.map(([r0, r1, pr], i) => (
          <rect key={`r${i}`} x={px(r0)} y={py(pr)} width={px(r1) - px(r0)} height={(H - padB) - py(pr)}
            fill={LEG_COLOR[leg]} fillOpacity={0.22} stroke={LEG_COLOR[leg]} strokeWidth={0.6} />
        ))}
        {/* interpolated monotone envelope */}
        {showInterp && env.map(([r0, r1, pr], i) => (
          <line key={`e${i}`} x1={px(r0)} y1={py(pr)} x2={px(r1)} y2={py(pr)} stroke={ENV_COLOR} strokeWidth={2.2} />
        ))}
        {showInterp && elevens.map(([t, pr], i) => (
          <circle key={`d${i}`} cx={px(t)} cy={py(pr)} r={2.6} fill={ENV_COLOR} />
        ))}
        {/* raw corner points */}
        {raw.map(([, r1, pr], i) => (<circle key={`c${i}`} cx={px(r1)} cy={py(pr)} r={2.4} fill={LEG_COLOR[leg]} />))}
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="AP = area under the PR curve" value={fmt(ap, 4)} accent />
        <Readout label="interpolated AP (envelope area)" value={fmt(iap, 4)} />
        <Readout label="inflation" value={`+${fmt(iap - ap, 4)}`} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The shaded bars are the recall steps; each relevant hit advances recall by 1/|R|, so their total area is exactly
        <strong> AP = {fmt(ap, 3)}</strong>. The <span style={{ color: ENV_COLOR }}>interpolated envelope</span> replaces each
        step with the best precision at any higher recall — its area, {fmt(iap, 3)}, is always <strong>≥ the raw AP</strong>.
        "AP" without a stated convention is ambiguous; the raw area is what MAP averages.
      </p>
    </div>
  );
}

// ===== Panel C — MAP / MRR across queries, and the MAP = MRR identity ===============================
function AggregatePanel({ leg, setLeg, knownItem, setKnownItem }: {
  leg: Leg; setLeg: (l: Leg) => void; knownItem: boolean; setKnownItem: (v: boolean) => void;
}) {
  const samples = knownItem ? PER_Q_RR[leg] : PER_Q_AP[leg];
  const m = mean(samples);
  const W = 540, H = 170, padL = 36, padR = 12, padT = 12, padB = 26;
  const bw = (W - padL - padR) / N_QUERIES;
  const py = (v: number) => H - padB - (H - padT - padB) * v;
  return (
    <div>
      <LegPills leg={leg} setLeg={setLeg} />
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <button type="button" style={pill(!knownItem)} onClick={() => setKnownItem(false)}>set regime · AP → MAP</button>
        <button type="button" style={pill(knownItem)} onClick={() => setKnownItem(true)}>known-item · RR → MRR</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="per-query metric strip with the mean line" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (<text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
        {samples.map((v, q) => (
          <rect key={q} x={padL + q * bw + 0.6} y={py(v)} width={Math.max(bw - 1.2, 1)} height={(H - padB) - py(v)}
            fill={LEG_COLOR[leg]} fillOpacity={0.6} />
        ))}
        {/* mean line */}
        <line x1={padL} y1={py(m)} x2={W - padR} y2={py(m)} stroke={ACCENT} strokeWidth={2} strokeDasharray="5 3" />
        <text x={W - padR} y={py(m) - 4} textAnchor="end" fontSize={9} fill={ACCENT} fontFamily="var(--font-sans)">
          {knownItem ? 'MRR' : 'MAP'} = {fmt(m, 3)}
        </text>
        <text x={padL + 2} y={H - padB + 18} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">one bar per query (1…{N_QUERIES})</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="MAP (set regime)" value={fmt(MAP[leg], 4)} accent={!knownItem} />
        <Readout label="MRR (known-item)" value={fmt(MRR[leg], 4)} accent={knownItem} />
        <Readout label="per-query spread (std)" value={fmt(stdev(samples), 3)} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        {knownItem ? (
          <>With exactly one relevant doc per query, AP = 1/(its rank) = RR for <em>every</em> query — so MAP and MRR
          coincide exactly ({fmt(MRR[leg], 3)}). The strip is the reciprocal-rank sample whose mean is MRR.</>
        ) : (
          <>MAP is the <strong>mean</strong> of a noisy per-query AP sample (spread {fmt(stdev(samples), 3)}). A single
          query says little; the average is the estimate. Switch to known-item to see MAP collapse onto MRR.</>
        )}
        {' '}<strong>The verdict can reverse by metric:</strong> {LEG_LABEL[FLIP.mapWinner]} outranks{' '}
        {LEG_LABEL[FLIP.mrrWinner]} by MAP ({fmt(MAP[FLIP.mapWinner], 3)} &gt; {fmt(MAP[FLIP.mrrWinner], 3)}), yet{' '}
        {LEG_LABEL[FLIP.mrrWinner]} wins on MRR ({fmt(MRR[FLIP.mrrWinner], 3)} &gt; {fmt(MRR[FLIP.mapWinner], 3)}).
      </p>
    </div>
  );
}

// ===== Panel D — metrics as estimators: the CI that shrinks =========================================
const PAIR: [Leg, Leg] = ['dense', 'late_interaction'];
function EstimatorPanel({ n, setN }: { n: number; setN: (v: number) => void }) {
  const W = 400, H = 150, padL = 40, padR = 16, padT = 22, padB = 28;
  const xMin = 0.55, xMax = 0.98;
  const px = (v: number) => padL + (W - padL - padR) * ((v - xMin) / (xMax - xMin));
  const rows = PAIR.map((leg) => {
    const sd = stdev(PER_Q_AP[leg]);
    const half = ciHalf(sd, n);
    return { leg, mean: mean(PER_Q_AP[leg]), half, lo: mean(PER_Q_AP[leg]) - half, hi: mean(PER_Q_AP[leg]) + half };
  });
  const overlap = rows[0].lo <= rows[1].hi && rows[1].lo <= rows[0].hi;
  // inset: SE vs n (baked empirical dots + theory curve std/sqrt(n))
  const sdDense = stdev(PER_Q_AP.dense);
  const iW = 150, iH = 96, iPadL = 26, iPadB = 20, iPadT = 8, iPadR = 8;
  const nMax = 80, seMax = 0.07;
  const ipx = (nn: number) => iPadL + (iW - iPadL - iPadR) * (nn / nMax);
  const ipy = (se: number) => iH - iPadB - (iH - iPadT - iPadB) * (se / seMax);
  const theory = Array.from({ length: 16 }, (_v, i) => { const nn = 5 + i * 5; return [nn, sdDense / Math.sqrt(nn)] as [number, number]; });
  return (
    <div>
      <Slider label="query count n" value={n} min={5} max={N_QUERIES} step={1} onChange={(v) => setN(Math.round(v))} display={`${n}`} />
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="two legs' MAP with 95 percent confidence intervals" style={{ width: '100%', maxWidth: W, height: 'auto', flex: '1 1 320px' }}>
          <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
          {[0.6, 0.7, 0.8, 0.9].map((v) => (<text key={v} x={px(v)} y={H - padB + 14} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
          <text x={(padL + W - padR) / 2} y={H - 4} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">MAP</text>
          {rows.map((r, i) => {
            const y = padT + i * 46;
            return (
              <g key={r.leg}>
                <line x1={px(r.lo)} y1={y} x2={px(r.hi)} y2={y} stroke={LEG_COLOR[r.leg]} strokeWidth={2.4} />
                <line x1={px(r.lo)} y1={y - 6} x2={px(r.lo)} y2={y + 6} stroke={LEG_COLOR[r.leg]} strokeWidth={2} />
                <line x1={px(r.hi)} y1={y - 6} x2={px(r.hi)} y2={y + 6} stroke={LEG_COLOR[r.leg]} strokeWidth={2} />
                <circle cx={px(r.mean)} cy={y} r={4} fill={LEG_COLOR[r.leg]} />
                <text x={px(r.mean)} y={y - 10} textAnchor="middle" fontSize={9} fill={LEG_COLOR[r.leg]} fontFamily="var(--font-sans)">{LEG_LABEL[r.leg]}</text>
              </g>
            );
          })}
        </svg>
        <svg viewBox={`0 0 ${iW} ${iH}`} role="img" aria-label="standard error versus query count" style={{ width: iW, height: 'auto' }}>
          <line x1={iPadL} y1={iH - iPadB} x2={iW - iPadR} y2={iH - iPadB} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={iPadL} y1={iPadT} x2={iPadL} y2={iH - iPadB} stroke="var(--color-border)" strokeWidth={1} />
          <path d={theory.map(([nn, se], i) => (i ? 'L' : 'M') + ipx(nn) + ' ' + ipy(se)).join(' ')} fill="none" stroke={MUTED} strokeWidth={1.4} />
          {SE_SCALING.filter(([nn]) => nn <= nMax).map(([nn, se], i) => (<circle key={i} cx={ipx(nn)} cy={ipy(se)} r={2.6} fill={LEG_COLOR.dense} />))}
          <circle cx={ipx(Math.min(n, nMax))} cy={ipy(sdDense / Math.sqrt(n))} r={3.6} fill={ACCENT} />
          <text x={iPadL} y={6} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">SE</text>
          <text x={iW - iPadR} y={iH - iPadB + 13} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">n →</text>
          <text x={(iPadL + iW) / 2} y={iPadT + 2} textAnchor="middle" fontSize={7.5} fill={MUTED} fontFamily="var(--font-sans)">σ̂/√n</text>
        </svg>
      </div>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="dense MAP ± 95% CI" value={`${fmt(rows[0].mean, 3)} ± ${fmt(rows[0].half, 3)}`} />
        <Readout label="late-interaction MAP ± 95% CI" value={`${fmt(rows[1].mean, 3)} ± ${fmt(rows[1].half, 3)}`} />
        <Readout label={`at n = ${n}`} value={overlap ? 'CIs overlap — tied' : 'CIs separate — distinguishable'} accent={!overlap} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        A reported MAP is a <strong>sample mean</strong>, with standard error σ̂/√n that shrinks as 1/√n (inset). At few
        queries the two systems' 95% CIs <strong>overlap</strong> — you cannot tell them apart; they separate only at
        <strong> n = {SEP_N}</strong>. An observed MAP gap is not a real one until it clears sampling error — which is exactly
        what a significance test decides.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'cutoff' | 'pr' | 'aggregate' | 'estimator';
const TEX: Record<Panel, string> = {
  cutoff: 'P@k = \\frac{|\\,\\mathrm{top}_k \\cap R\\,|}{k}, \\qquad R@k = \\frac{|\\,\\mathrm{top}_k \\cap R\\,|}{|R|}',
  pr: '\\mathrm{AP} = \\sum_k (R_k - R_{k-1})\\,P_k = \\frac{1}{|R|}\\sum_{i}\\frac{i}{\\mathrm{pos}_i}',
  aggregate: '\\mathrm{MAP} = \\frac{1}{Q}\\sum_q \\mathrm{AP}_q, \\qquad \\mathrm{MRR} = \\frac{1}{Q}\\sum_q \\frac{1}{\\mathrm{rank}_q}',
  estimator: '\\widehat{\\mathrm{SE}} = \\frac{\\hat\\sigma}{\\sqrt{n}}, \\qquad \\mathrm{CI}_{95} = \\mathrm{MAP} \\pm 1.96\\,\\widehat{\\mathrm{SE}}',
};

export default memo(function SetMetricsLaboratory() {
  const [panel, setPanel] = useState<Panel>('cutoff');
  const [leg, setLeg] = useState<Leg>('dense');
  const [k, setK] = useState(5);
  const [showInterp, setShowInterp] = useState(false);
  const [knownItem, setKnownItem] = useState(false);
  const [n, setN] = useState(8);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="set-metrics" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'cutoff')} onClick={() => setPanel('cutoff')}>A · precision & recall</button>
        <button type="button" style={pill(panel === 'pr')} onClick={() => setPanel('pr')}>B · PR curve & AP</button>
        <button type="button" style={pill(panel === 'aggregate')} onClick={() => setPanel('aggregate')}>C · MAP & MRR</button>
        <button type="button" style={pill(panel === 'estimator')} onClick={() => setPanel('estimator')}>D · metrics as estimators</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'cutoff' && <CutoffPanel leg={leg} setLeg={setLeg} k={k} setK={setK} />}
      {panel === 'pr' && <PRPanel leg={leg} setLeg={setLeg} showInterp={showInterp} setShowInterp={setShowInterp} />}
      {panel === 'aggregate' && <AggregatePanel leg={leg} setLeg={setLeg} knownItem={knownItem} setKnownItem={setKnownItem} />}
      {panel === 'estimator' && <EstimatorPanel n={n} setN={setN} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_DOCS} synthetic finance documents, {N_QUERIES} queries; relevance is the top-{R_SIZE} exact-MaxSim
        neighbor set (a neutral oracle). Numbers mirror <code>set_metrics_precision_recall_map_mrr.py</code>; the lab
        recomputes precision, recall, the PR/AP area, the mean, and the CI in closed form.
      </p>
    </div>
  );
});
