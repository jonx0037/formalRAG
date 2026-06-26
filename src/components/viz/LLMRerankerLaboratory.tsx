import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * LLM Reranker Laboratory — four panels for the `llm-listwise-rerankers` topic:
 *   A. Sliding-window bubble sort (RankGPT). The call count P·(⌈(n−w)/s⌉+1) recomputed LIVE on the
 *      passes/window/step sliders, against the O(n²) all-pairs cost; the seed-averaged recall@10-by-pass
 *      curve (a rising-then-plateau climb); and the back-to-front vs front-to-back asymmetry.
 *   B. Positional bias (lost in the middle). The in-window U-curve kernel recomputed LIVE from POS_DIP,
 *      its flattened (presentation-averaged) form, and the measured none/biased/corrected recall.
 *   C. Rank aggregation. The consensus Kendall-τ-to-π* falling with K for Borda/RRF/MC4, with the worked
 *      query's Borda consensus + its τ recomputed LIVE on the K slider, the 1/√K rank-concentration CLT,
 *      and the Kemeny-cost ratios.
 *   D. Distillation + the cost–quality Pareto frontier. The six methods' (LLM calls, NDCG@10) with CI
 *      bars and the non-dominated hull recomputed LIVE under a linear/log cost axis.
 *
 * VIZ ↔ PYTHON INVARIANT (CLAUDE.md): every baked number is mirrored TO THE DECIMAL from
 * notebooks/llm-listwise-rerankers/llm_listwise_rerankers.py (viz_constants()). Matching asserts:
 * test_call_count_law / test_passes_monotone_recall / test_direction_asymmetry /
 * test_positional_bias_bites_and_corrects / test_aggregation_reduces_tau /
 * test_aggregated_rank_concentration / test_mc4_approximates_kemeny /
 * test_perfect_teacher_student_equals_fit_listmle / test_frontier_spread_and_structure. The lab recomputes
 * ONLY closed forms in TS (the call-count law, the U-curve, the Borda/RRF consensus + Kendall-τ, the Pareto
 * hull); the seed-averaged scalars (the recall curves, the τ-vs-K curves, the frontier, the CIs) are baked.
 * The LLM is a SIMULATED noisy permutation oracle — no real model is called. Change a number here → change
 * it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const POOL = 60;          // rerank pool n (10 gold + 50 hard negatives)
const WINDOW = 15;        // context window w
const STEP = 10;          // slide step s
const N_PASSES = 4;       // default passes P
const TOPK = 10;
const ALLPAIRS_CALLS = 1770;   // C(60,2)

// Panel A — the recall-vs-passes climb (seed-averaged @ w=15, s=10), and the slide-direction asymmetry.
const RECALL_BY_PASS: [number, number][] = [[0, 0.1672], [1, 0.5253], [2, 0.7578], [3, 0.8461], [4, 0.8682]];
const DIRECTION = { back: 0.5076, front: 0.2471 };   // one-pass recall@10, back-to-front vs front-to-back

// Panel B — the lost-in-the-middle U-curve and the measured recall under three regimes. The kernel itself
// is recomputed live from POS_DIP (1 − dip·sin(π(p+½)/w)) — the .py bakes BIAS_KERNEL and a test pins it.
const POS_DIP = 0.55;
const RECALL_NONE = 0.9812;
const RECALL_BIASED = 0.9104;
const RECALL_CORRECTED = 0.9677;

// Panel C — aggregation. τ-to-π* by K (averaged), the worked query's K noisy perms, the CLT, Kemeny ratios.
const K_GRID = [1, 2, 4, 8, 16, 32];
const TAU_VS_K: Record<'borda' | 'rrf' | 'mc4', number[]> = {
  borda: [29.539, 22.695, 15.297, 9.896, 6.354, 3.247],
  rrf: [29.539, 23.352, 15.716, 10.31, 6.661, 3.456],
  mc4: [29.539, 23.398, 15.089, 9.664, 6.336, 3.453],
};
const RANK_CONCENTRATION: [number, number, number][] = [   // (K, std(mean rank), std·√K)
  [1, 0.396, 0.396], [2, 0.295, 0.417], [4, 0.197, 0.395], [8, 0.115, 0.326], [16, 0.098, 0.392], [32, 0.091, 0.514],
];
const KEMENY_RATIOS = { borda: 1.027, rrf: 1.027, mc4: 1.0 };
const WORKED_STAR = [51, 109, 80, 29, 36, 100, 62, 119, 32, 21, 17, 54, 9, 106, 4, 10, 58, 3, 27, 86];
const WORKED_PERMS: number[][] = [
  [80, 29, 51, 32, 109, 119, 36, 100, 17, 62, 21, 54, 9, 106, 58, 3, 4, 86, 10, 27],
  [100, 80, 109, 51, 36, 62, 29, 21, 17, 9, 32, 119, 86, 106, 54, 4, 10, 58, 3, 27],
  [51, 109, 100, 80, 32, 36, 17, 29, 62, 119, 21, 86, 9, 54, 4, 27, 10, 106, 58, 3],
  [109, 51, 62, 80, 100, 36, 29, 32, 119, 21, 54, 4, 17, 10, 27, 106, 9, 58, 3, 86],
  [51, 109, 80, 100, 21, 17, 62, 10, 36, 29, 119, 32, 27, 9, 58, 54, 4, 106, 86, 3],
  [51, 109, 29, 80, 58, 36, 62, 100, 9, 32, 119, 17, 54, 10, 106, 21, 4, 27, 3, 86],
  [109, 100, 29, 80, 51, 62, 36, 32, 54, 119, 4, 17, 9, 21, 3, 106, 10, 58, 86, 27],
  [29, 80, 51, 109, 62, 119, 3, 36, 100, 17, 32, 86, 54, 21, 106, 27, 58, 9, 4, 10],
];

// Panel D — distillation + the cost-quality frontier (NDCG@10, LLM calls/query, NDCG CI).
const SPEEDUP = 6000;
const STUDENT_PERFECT_EQ_FITLISTMLE = true;
type Method = 'dense_best_leg' | 'rrf_3legs' | 'distilled_student' | 'pointwise_llm' | 'sliding_window_llm' | 'allpairs_llm';
interface FrontierPt { method: Method; label: string; ndcg: number; cost: number; lo: number; hi: number; llm: boolean }
const FRONTIER: FrontierPt[] = [
  { method: 'dense_best_leg', label: 'dense (best leg)', ndcg: 0.6891, cost: 0, lo: 0.5852, hi: 0.7931, llm: false },
  { method: 'rrf_3legs', label: 'RRF (3 legs)', ndcg: 0.7318, cost: 0, lo: 0.6716, hi: 0.792, llm: false },
  { method: 'distilled_student', label: 'distilled student', ndcg: 0.769, cost: 0, lo: 0.7029, hi: 0.8351, llm: false },
  { method: 'sliding_window_llm', label: 'sliding-window LLM', ndcg: 0.8647, cost: 24, lo: 0.8436, hi: 0.8858, llm: true },
  { method: 'pointwise_llm', label: 'pointwise LLM', ndcg: 0.868, cost: 60, lo: 0.8523, hi: 0.8837, llm: true },
  { method: 'allpairs_llm', label: 'all-pairs LLM', ndcg: 0.9438, cost: 1770, lo: 0.9283, hi: 0.9594, llm: true },
];

const ACCENT = 'var(--color-accent)';
const GOLD = '#5fa873';
const NEG = '#c25b6b';
const MUTED = '#7a8a99';
const AGG_COLOR: Record<'borda' | 'rrf' | 'mc4', string> = { borda: '#5fa873', rrf: '#6a8caf', mc4: '#c08457' };
const fmt = (x: number, n = 3) => x.toFixed(n);

// --- closed-form helpers (the same rules the notebook tests) -------------------------------------------
const callCount = (n: number, w: number, s: number, p: number) =>
  n <= w ? p : p * (Math.ceil((n - w) / s) + 1);
// the lost-in-the-middle multiplier, 1 − dip·sin(π(pos+0.5)/w) — verbatim TS mirror of positional_weight
const posWeight = (pos: number, w: number, dip: number) => 1 - dip * Math.sin((Math.PI * (pos + 0.5)) / w);
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;

function kendallTau(a: number[], b: number[]): number {
  const pa = new Map<number, number>(), pb = new Map<number, number>();
  a.forEach((d, i) => pa.set(d, i));
  b.forEach((d, i) => pb.set(d, i));
  let disc = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = i + 1; j < a.length; j++) {
      const x = a[i], y = a[j];
      if ((pa.get(x)! - pa.get(y)!) * (pb.get(x)! - pb.get(y)!) < 0) disc++;
    }
  return disc;
}
function bordaConsensus(perms: number[][]): number[] {
  const items = perms[0], n = items.length;
  const score = new Map<number, number>(); items.forEach((d) => score.set(d, 0));
  for (const r of perms) r.forEach((d, rank) => score.set(d, score.get(d)! + (n - 1 - rank)));
  return [...items].sort((x, y) => score.get(y)! - score.get(x)! || x - y);
}

// --- shared UI atoms ------------------------------------------------------------------
function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? ACCENT : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}
function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display: string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '12rem' }}>{label} = <strong>{display}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label} style={{ flex: 1, accentColor: ACCENT }} />
    </label>
  );
}
const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? ACCENT : 'var(--color-border)'}`,
  background: active ? ACCENT : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

// ===== Panel A — sliding-window bubble sort =========================================================
function WindowPanel({ passes, setPasses, w, setW, s, setS }: {
  passes: number; setPasses: (v: number) => void; w: number; setW: (v: number) => void; s: number; setS: (v: number) => void;
}) {
  const nWin = callCount(POOL, w, s, 1);
  const calls = callCount(POOL, w, s, passes);
  const recall = RECALL_BY_PASS[Math.min(passes, RECALL_BY_PASS.length - 1)][1];
  const W = 540, H = 200, padL = 38, padR = 14, padT = 14, padB = 26;
  const px = (p: number) => padL + (p / N_PASSES) * (W - padL - padR);
  const py = (r: number) => padT + (1 - r) * (H - padT - padB);
  return (
    <div>
      <Slider label="passes P" value={passes} min={0} max={N_PASSES} step={1} onChange={setPasses} display={`${passes}`} />
      <Slider label="window w" value={w} min={10} max={30} step={5} onChange={setW} display={`${w}`} />
      <Slider label="step s" value={s} min={5} max={20} step={5} onChange={setS} display={`${s}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="recall@10 versus number of passes" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((r) => (
          <g key={r}>
            <line x1={padL} y1={py(r)} x2={W - padR} y2={py(r)} stroke="var(--color-border)" strokeWidth={0.5} />
            <text x={padL - 5} y={py(r) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)">{r.toFixed(2)}</text>
          </g>
        ))}
        <polyline points={RECALL_BY_PASS.map(([p, r]) => `${px(p)},${py(r)}`).join(' ')}
          fill="none" stroke={GOLD} strokeWidth={2} />
        {RECALL_BY_PASS.map(([p, r]) => <circle key={p} cx={px(p)} cy={py(r)} r={2.5} fill={GOLD} />)}
        <line x1={px(passes)} y1={padT} x2={px(passes)} y2={H - padB} stroke={ACCENT} strokeWidth={1} strokeDasharray="3 2" />
        <circle cx={px(passes)} cy={py(recall)} r={4} fill={ACCENT} />
        {RECALL_BY_PASS.map(([p]) => <text key={p} x={px(p)} y={H - 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)">{p}</text>)}
        <text x={(W) / 2} y={H - 1} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">passes · recall@10 (w=15, s=10)</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="LLM calls = P·(⌈(n−w)/s⌉+1)" value={`${calls}`} accent />
        <Readout label={`windows/pass · vs all-pairs C(${POOL},2)`} value={`${nWin} · ${ALLPAIRS_CALLS}`} />
        <Readout label="recall@10 at this pass" value={fmt(recall, 3)} />
      </div>
      <div style={{ display: 'flex', gap: '0.8rem', marginTop: '0.7rem', alignItems: 'flex-end' }}>
        {(['back', 'front'] as const).map((dir) => (
          <div key={dir} style={{ textAlign: 'center' }}>
            <div style={{ width: 80, height: 70 * DIRECTION[dir], background: dir === 'back' ? GOLD : MUTED, borderRadius: '3px 3px 0 0' }} />
            <div style={{ fontSize: '0.72rem', marginTop: '0.2rem' }}>{dir === 'back' ? 'back→front' : 'front→back'}<br />{fmt(DIRECTION[dir], 3)}</div>
          </div>
        ))}
        <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', lineHeight: 1.5, flex: 1 }}>
          One <strong style={{ color: GOLD }}>back-to-front</strong> pass carries a buried document to the top in a single
          sweep; front-to-back advances it one window per pass. The call count is <strong>O(n/s)</strong> per pass — cheaper
          than all-pairs by {Math.round(ALLPAIRS_CALLS / callCount(POOL, WINDOW, STEP, N_PASSES))}× at {N_PASSES} passes — and a
          noisy comparator's recall climbs with passes toward a plateau, never exact in one pass (the Ω(n log n) floor).
        </p>
      </div>
    </div>
  );
}

// ===== Panel B — positional bias (lost in the middle) ==============================================
function BiasPanel({ dip, setDip }: { dip: number; setDip: (v: number) => void }) {
  const kernel = Array.from({ length: WINDOW }, (_, p) => posWeight(p, WINDOW, dip));
  const flat = mean(kernel);          // presentation-averaged effective weight (a constant) — the correction
  const W = 540, H = 150, padL = 34, padR = 14, padT = 12, padB = 24;
  const bx = (p: number) => padL + (p / (WINDOW - 1)) * (W - padL - padR);
  const by = (v: number) => padT + (1 - v) * (H - padT - padB);
  const bars: [string, number, string][] = [
    ['unbiased', RECALL_NONE, GOLD], ['biased', RECALL_BIASED, NEG], ['corrected', RECALL_CORRECTED, '#6a8caf'],
  ];
  return (
    <div>
      <Slider label="positional dip" value={dip} min={0} max={0.9} step={0.05} onChange={setDip} display={dip.toFixed(2)} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="in-window positional weight kernel" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={padL} y1={by(v)} x2={W - padR} y2={by(v)} stroke="var(--color-border)" strokeWidth={0.5} />
            <text x={padL - 4} y={by(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)">{v.toFixed(1)}</text>
          </g>
        ))}
        <polyline points={kernel.map((v, p) => `${bx(p)},${by(v)}`).join(' ')} fill="none" stroke={NEG} strokeWidth={2} />
        {kernel.map((v, p) => <circle key={p} cx={bx(p)} cy={by(v)} r={2} fill={NEG} />)}
        <line x1={padL} y1={by(flat)} x2={W - padR} y2={by(flat)} stroke="#6a8caf" strokeWidth={1.5} strokeDasharray="4 3" />
        <text x={W - padR} y={by(flat) - 4} textAnchor="end" fontSize={8} fill="#6a8caf">corrected (presentation-averaged)</text>
        <text x={(W) / 2} y={H - 1} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">window position · attention weight b(p)</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <Readout label="center weight 1−dip" value={fmt(1 - dip, 3)} accent />
        {bars.map(([name, v, c]) => (
          <div key={name} style={{ textAlign: 'center' }}>
            <div style={{ width: 64, height: 60 * v, background: c, borderRadius: '3px 3px 0 0' }} />
            <div style={{ fontSize: '0.7rem', marginTop: '0.15rem' }}>{name}<br />{fmt(v, 3)}</div>
          </div>
        ))}
      </div>
      <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The reranker attends most to the <strong>ends</strong> of its window and least to the <strong style={{ color: NEG }}>middle</strong>
        {' '}(Liu et al.). The dip suppresses center documents, dropping recall@10 from {fmt(RECALL_NONE, 3)} to {fmt(RECALL_BIASED, 3)}.
        Presenting each window in several <em>random</em> orders and Borda-aggregating flattens b(p) to its mean
        ({fmt(flat, 3)} at this dip), recovering {fmt(RECALL_CORRECTED, 3)} — toward, but not beating, the unbiased oracle.
        (The measured bars are at dip = {POS_DIP.toFixed(2)}; the slider explores the kernel shape.)
      </p>
    </div>
  );
}

// ===== Panel C — rank aggregation ===================================================================
function AggregatePanel({ k, setK }: { k: number; setK: (v: number) => void }) {
  const consensus = bordaConsensus(WORKED_PERMS.slice(0, k));
  const tau = kendallTau(consensus, WORKED_STAR);
  const tau1 = kendallTau(bordaConsensus(WORKED_PERMS.slice(0, 1)), WORKED_STAR);
  const W = 540, H = 180, padL = 34, padR = 14, padT = 12, padB = 28;
  const maxTau = 31, kMax = 32;
  const lx = (kk: number) => padL + (Math.log2(kk) / Math.log2(kMax)) * (W - padL - padR);
  const ly = (t: number) => padT + (1 - t / maxTau) * (H - padT - padB);
  return (
    <div>
      <Slider label="ballots K (worked query)" value={k} min={1} max={8} step={1} onChange={setK} display={`${k}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="consensus Kendall-tau to the ideal order versus K" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {[0, 10, 20, 30].map((t) => (
          <g key={t}>
            <line x1={padL} y1={ly(t)} x2={W - padR} y2={ly(t)} stroke="var(--color-border)" strokeWidth={0.5} />
            <text x={padL - 4} y={ly(t) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)">{t}</text>
          </g>
        ))}
        {(['borda', 'rrf', 'mc4'] as const).map((m) => (
          <g key={m}>
            <polyline points={K_GRID.map((kk, i) => `${lx(kk)},${ly(TAU_VS_K[m][i])}`).join(' ')} fill="none" stroke={AGG_COLOR[m]} strokeWidth={2} />
            {K_GRID.map((kk, i) => <circle key={kk} cx={lx(kk)} cy={ly(TAU_VS_K[m][i])} r={2} fill={AGG_COLOR[m]} />)}
          </g>
        ))}
        {K_GRID.map((kk) => <text key={kk} x={lx(kk)} y={H - 14} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)">{kk}</text>)}
        <line x1={lx(Math.max(k, 1))} y1={padT} x2={lx(Math.max(k, 1))} y2={H - padB} stroke={ACCENT} strokeWidth={1} strokeDasharray="3 2" />
        <text x={(W) / 2} y={H - 1} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">K (log) · mean Kendall-τ to π*</text>
      </svg>
      <div style={{ display: 'flex', gap: '0.9rem', marginTop: '0.3rem', flexWrap: 'wrap' }}>
        {(['borda', 'rrf', 'mc4'] as const).map((m) => (
          <span key={m} style={{ fontSize: '0.72rem', color: AGG_COLOR[m] }}>● {m === 'mc4' ? 'Markov chain' : m.toUpperCase()}</span>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
        <Readout label={`worked-query Borda τ at K=${k}`} value={`${tau}`} accent />
        <Readout label="vs K=1 (single ballot)" value={`${tau1}`} />
        <Readout label="Kemeny ratio mc4 / borda" value={`${fmt(KEMENY_RATIOS.mc4, 3)} / ${fmt(KEMENY_RATIOS.borda, 3)}`} />
      </div>
      <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Aggregating <strong>K</strong> noisy permutations concentrates the consensus toward truth: the per-document averaged
        rank concentrates at the <strong>1/√K</strong> CLT rate (std·√K ≈ {fmt(RANK_CONCENTRATION[0][2], 2)}–{fmt(RANK_CONCENTRATION[4][2], 2)}),
        so the consensus Kendall-τ falls monotonically. Borda, RRF, and the Markov-chain (MC4) consensus all approximate the
        NP-hard Kemeny median within {fmt(Math.max(KEMENY_RATIOS.borda, KEMENY_RATIOS.rrf), 2)}× of optimal.
      </p>
    </div>
  );
}

// ===== Panel D — distillation + the cost-quality Pareto frontier ===================================
function paretoHull(pts: FrontierPt[]): Set<Method> {
  const sorted = [...pts].sort((a, b) => a.cost - b.cost || b.ndcg - a.ndcg);
  const keep = new Set<Method>();
  let best = -Infinity;
  for (const p of sorted) {
    if (p.ndcg > best) { keep.add(p.method); best = p.ndcg; }
  }
  return keep;
}
function FrontierPanel({ logCost, setLogCost }: { logCost: boolean; setLogCost: (v: boolean) => void }) {
  const hull = paretoHull(FRONTIER);
  const W = 540, H = 230, padL = 42, padR = 16, padT = 14, padB = 34;
  const maxCost = ALLPAIRS_CALLS;
  const cx = (c: number) => logCost
    ? padL + (Math.log10(c + 1) / Math.log10(maxCost + 1)) * (W - padL - padR)
    : padL + (c / maxCost) * (W - padL - padR);
  const yLo = 0.65, yHi = 0.97;
  const cy = (v: number) => padT + (1 - (v - yLo) / (yHi - yLo)) * (H - padT - padB);
  const hullPts = FRONTIER.filter((p) => hull.has(p.method)).sort((a, b) => a.cost - b.cost);
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <button type="button" style={pill(logCost)} onClick={() => setLogCost(true)}>log cost</button>
        <button type="button" style={pill(!logCost)} onClick={() => setLogCost(false)}>linear cost</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="cost versus NDCG@10 frontier" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {[0.7, 0.8, 0.9].map((v) => (
          <g key={v}>
            <line x1={padL} y1={cy(v)} x2={W - padR} y2={cy(v)} stroke="var(--color-border)" strokeWidth={0.5} />
            <text x={padL - 4} y={cy(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)">{v.toFixed(1)}</text>
          </g>
        ))}
        <polyline points={hullPts.map((p) => `${cx(p.cost)},${cy(p.ndcg)}`).join(' ')} fill="none" stroke={ACCENT} strokeWidth={1.5} strokeDasharray="5 3" />
        {FRONTIER.map((p) => (
          <g key={p.method}>
            <line x1={cx(p.cost)} y1={cy(p.lo)} x2={cx(p.cost)} y2={cy(p.hi)} stroke={p.llm ? ACCENT : MUTED} strokeWidth={1} opacity={0.5} />
            <circle cx={cx(p.cost)} cy={cy(p.ndcg)} r={hull.has(p.method) ? 5 : 3.5}
              fill={p.llm ? ACCENT : GOLD} stroke={hull.has(p.method) ? 'var(--color-text)' : 'none'} strokeWidth={hull.has(p.method) ? 1 : 0} />
            <text x={cx(p.cost)} y={cy(p.ndcg) - 8} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{p.label}</text>
            <text x={cx(p.cost)} y={H - 18} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)">{p.cost}</text>
          </g>
        ))}
        <text x={(W) / 2} y={H - 3} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">LLM calls / query ({logCost ? 'log' : 'linear'}) · NDCG@10</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="distillation speedup (teacher / student)" value={`${SPEEDUP.toLocaleString()}×`} accent />
        <Readout label="perfect teacher ⇒ student == fit_listmle" value={STUDENT_PERFECT_EQ_FITLISTMLE ? '✓ exact' : '—'} />
        <Readout label="sliding vs all-pairs cost" value={`${callCount(POOL, WINDOW, STEP, N_PASSES)} vs ${ALLPAIRS_CALLS}`} />
      </div>
      <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        All six methods on one shared pool; cost is <strong>LLM calls per query</strong> (the cross-encoder is the unit the
        call is priced against, not a method). The <strong style={{ color: ACCENT }}>sliding window</strong> reaches near the
        all-pairs ceiling at {callCount(POOL, WINDOW, STEP, N_PASSES)} calls vs {ALLPAIRS_CALLS}; the
        {' '}<strong style={{ color: GOLD }}>distilled student</strong> answers at <strong>zero</strong> inference calls. The seed-free
        wins are structural (the call-count law, the {SPEEDUP.toLocaleString()}× distillation speedup) — the NDCG deltas sit inside the CI.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'window' | 'bias' | 'aggregate' | 'frontier';
const TEX: Record<Panel, string> = {
  window: '\\text{calls} = P\\cdot\\Big(\\Big\\lceil \\tfrac{n-w}{s} \\Big\\rceil + 1\\Big) = O\\!\\big(\\tfrac{n}{s}\\big)\\,P \\;\\ll\\; \\binom{n}{2}',
  bias: 'b(p) = 1 - \\beta\\,\\sin\\!\\Big(\\tfrac{\\pi (p+\\tfrac12)}{w}\\Big),\\qquad \\bar b = \\tfrac1w\\textstyle\\sum_p b(p)\\ \\text{(presentation-averaged)}',
  aggregate: '\\tau\\big(\\hat\\pi_K,\\pi^*\\big)\\downarrow,\\qquad \\operatorname{std}\\!\\big(\\overline{\\operatorname{rank}}_K(d)\\big)\\propto \\tfrac{1}{\\sqrt{K}}\\ \\text{(CLT)}',
  frontier: '\\text{cost} = \\#\\{\\text{LLM calls}\\},\\qquad \\text{distill: } w^\\star_{\\tau\\to 0} = \\arg\\min_w \\sum_q L_{\\text{ListMLE}}\\big(w;\\pi^*_q\\big)',
};

export default memo(function LLMRerankerLaboratory() {
  const [panel, setPanel] = useState<Panel>('window');
  const [passes, setPasses] = useState(N_PASSES);
  const [w, setW] = useState(WINDOW);
  const [s, setS] = useState(STEP);
  const [dip, setDip] = useState(POS_DIP);
  const [k, setK] = useState(8);
  const [logCost, setLogCost] = useState(true);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="llm-listwise-rerankers" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'window')} onClick={() => setPanel('window')}>A · sliding-window bubble sort</button>
        <button type="button" style={pill(panel === 'bias')} onClick={() => setPanel('bias')}>B · positional bias</button>
        <button type="button" style={pill(panel === 'aggregate')} onClick={() => setPanel('aggregate')}>C · rank aggregation</button>
        <button type="button" style={pill(panel === 'frontier')} onClick={() => setPanel('frontier')}>D · distillation & Pareto</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem', overflowX: 'auto' }} />
      {panel === 'window' && <WindowPanel passes={passes} setPasses={setPasses} w={w} setW={setW} s={s} setS={setS} />}
      {panel === 'bias' && <BiasPanel dip={dip} setDip={setDip} />}
      {panel === 'aggregate' && <AggregatePanel k={k} setK={setK} />}
      {panel === 'frontier' && <FrontierPanel logCost={logCost} setLogCost={setLogCost} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {POOL} synthetic finance candidates ({TOPK} relevant + {POOL - TOPK} hard negatives); the LLM is a <strong>simulated
        noisy permutation oracle</strong> (seeded — no real model is called). Numbers mirror <code>llm_listwise_rerankers.py</code>;
        the lab recomputes the call-count law, the U-curve, the Borda consensus + Kendall-τ, and the Pareto hull in closed form.
      </p>
    </div>
  );
});
