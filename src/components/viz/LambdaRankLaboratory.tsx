import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * LambdaRank Laboratory — four panels for the `lambdarank-lambdamart-listwise` topic:
 *   A. RankNet vs LambdaRank λ forces. The worked query's top-k with each document's force under the
 *      uniform (RankNet) and ΔNDCG-weighted (LambdaRank) rule, and the per-rank gradient mass: LambdaRank
 *      concentrates its force at the head (top-3 share 0.40 vs 0.16) because the |ΔNDCG| weight decays
 *      with rank.
 *   B. Is λ a gradient? The score-space λ field on a 3-document toy, recomputed live in closed form.
 *      RankNet's field is conservative (continuous, symmetric Jacobian, zero circulation); LambdaRank's
 *      is discontinuous across a swap (a spectator pair's weight jumps), so it is the gradient of no
 *      scalar loss — globally.
 *   C. Listwise objectives. The convex ListMLE / ListNet loss bowl (a single basin) with a slider tracing
 *      the monotone descent to the optimum — a proper loss, the principled contrast to LambdaRank.
 *   D. LambdaMART & the comparison. The boosting NDCG climb over rounds, the constructed XOR instance
 *      where trees escape the linear ceiling (NDCG 1.0 vs 0.89), and held-out NDCG/recall bars for all
 *      five learned methods against RRF and the best single leg.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every baked number is mirrored TO THE DECIMAL from
 * notebooks/lambdarank-lambdamart-listwise/lambdarank_lambdamart_listwise.py (viz_constants()). Matching
 * asserts: test_delta_ndcg_closed_form_matches_brute / test_lambdarank_collapses_to_ranknet /
 * test_weight_strictly_decreasing_in_rank / test_gradient_top_concentrated /
 * test_within_cell_jacobian_symmetric_both / test_lambdarank_field_discontinuous_across_swap /
 * test_field_circulation / test_listwise_convex_unique / test_lambdamart_beats_linear_on_xor /
 * test_method_comparison_runs. The lab recomputes ONLY closed forms in TS (the score-space λ field and
 * its discontinuity, the bar geometry, the boost-curve interpolation, the bowl marker); the
 * Newton/tree-driven scalars (fitted weights, per-method metrics, the boosting curve, the loss bowl, the
 * λ forces, the gradient mass) are baked. Change a number here -> change it there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_DOCS = 120;
const N_QUERIES = 40;
const N_TRAIN = 24;
const N_TEST = 16;
const TOPK = 10;

// Panel A — worked query 19: top-10 order, grades, the two λ rules, the per-rank gradient mass.
const WORKED_Q = 19;
const WORKED_RANKING = [60, 62, 37, 21, 14, 77, 48, 108, 119, 81];
const WORKED_GRADE = [3, 3, 1, 0, 3, 0, 0, 1, 0, 0];                  // grade of each doc in ranking order
const RANKNET_LAMBDA = [-0.0749, -2.3513, -1.3202, 6.2506, -8.4319, 5.4226, 5.2083, -8.9696, 4.8144, 4.649];
const LAMBDARANK_LAMBDA = [-0.0173, -0.2623, 0.0592, 0.1127, -0.3788, 0.065, 0.0566, -0.0172, 0.0444, 0.04];
const RANKNET_MASS = [0.0749, 2.3513, 4.8084, 5.6589, 7.131, 4.5858, 4.4454, 8.7494, 3.6853, 3.5914];
const LAMBDARANK_MASS = [0.0173, 0.2623, 0.1186, 0.0941, 0.3454, 0.0438, 0.033, 0.0426, 0.019, 0.0144];
const TOP3_SHARE = { ranknet: 0.1605, lambdarank: 0.4021 };
const WEIGHT_HEAD_1_2 = 0.3691;                                       // |D(1)-D(2)|, the head swap weight
const WEIGHT_TAIL_9_10 = 0.012;                                      // |D(9)-D(10)|, the tail swap weight

// Panel B — the 3-doc integrability toy (grades 2,1,0). The field is recomputed live; these are anchors.
const TOY_GRADES = [2, 1, 0];
const TOY_S_BASE = [1.6, 1.0, 0.0];
const SWAP_JUMP_RANKNET = 0.000009;     // O(eps): continuous field
const SWAP_JUMP_LAMBDARANK = 0.082;     // Θ(1): a genuine discontinuity
const CIRC_RANKNET = 0.0;               // conservative -> zero circulation
const CIRC_LAMBDARANK = -0.0443;        // non-conservative -> nonzero circulation
const JAC_ASYM_RANKNET = 0.0;           // within-cell Jacobian symmetric
const JAC_ASYM_LAMBDARANK = 0.0;        // within-cell Jacobian symmetric too (the subtlety)

// Panel C — the convex loss bowls over (w_dense, w_li) at w_lex = optimum (13×13 grid on [-3,3]).
const BOWL_GRID = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const LISTMLE_BOWL = [[113.529, 104.208, 95.144, 86.552, 78.785, 72.321, 67.643, 65.069, 64.603, 65.843, 68.203, 71.241, 74.738], [106.855, 97.581, 88.563, 80.016, 72.306, 65.945, 61.487, 59.289, 59.276, 60.93, 63.643, 67.018, 70.881], [100.495, 91.285, 82.335, 73.862, 66.247, 60.048, 55.881, 54.112, 54.568, 56.643, 59.737, 63.5, 67.784], [94.509, 85.384, 76.526, 68.161, 60.691, 54.722, 50.922, 49.632, 50.576, 53.091, 56.605, 60.798, 65.502], [88.959, 79.94, 71.202, 62.982, 55.714, 50.053, 46.698, 45.937, 47.395, 50.38, 54.341, 58.949, 64.001], [83.899, 75.007, 66.416, 58.383, 51.38, 46.111, 43.285, 43.107, 45.109, 48.579, 52.961, 57.908, 63.218], [79.369, 70.623, 62.206, 54.404, 47.736, 42.953, 40.741, 41.196, 43.755, 47.676, 52.403, 57.604, 63.105], [75.389, 66.803, 58.588, 51.068, 44.818, 40.624, 39.107, 40.22, 43.302, 47.598, 52.584, 57.971, 63.618], [71.952, 63.543, 55.565, 48.393, 42.66, 39.155, 38.381, 40.129, 43.66, 48.247, 53.425, 58.952, 64.712], [69.037, 60.829, 53.138, 46.4, 41.29, 38.545, 38.512, 40.827, 44.722, 49.53, 54.851, 60.485, 66.334], [66.618, 58.65, 51.32, 45.116, 40.712, 38.748, 39.403, 42.2, 46.384, 51.361, 56.792, 62.507, 68.425], [64.678, 57.013, 50.135, 44.55, 40.887, 39.673, 40.941, 44.14, 48.554, 53.663, 59.177, 64.955, 70.924], [63.219, 55.94, 49.597, 44.672, 41.734, 41.211, 43.014, 46.55, 51.153, 56.367, 61.945, 67.768, 73.775]];
const LISTNET_BOWL = [[12.86, 11.788, 10.741, 9.744, 8.829, 8.037, 7.39, 6.876, 6.463, 6.118, 5.826, 5.592, 5.447], [11.962, 10.893, 9.849, 8.852, 7.934, 7.137, 6.487, 5.978, 5.58, 5.267, 5.035, 4.909, 4.945], [11.088, 10.024, 8.984, 7.988, 7.069, 6.269, 5.621, 5.124, 4.759, 4.511, 4.397, 4.464, 4.745], [10.243, 9.185, 8.15, 7.158, 6.24, 5.443, 4.805, 4.339, 4.042, 3.921, 4.008, 4.317, 4.807], [9.429, 8.379, 7.351, 6.365, 5.455, 4.67, 4.061, 3.665, 3.499, 3.583, 3.906, 4.409, 5.019], [8.65, 7.608, 6.59, 5.615, 4.72, 3.966, 3.426, 3.162, 3.203, 3.516, 4.021, 4.632, 5.296], [7.905, 6.874, 5.869, 4.912, 4.049, 3.36, 2.947, 2.888, 3.156, 3.644, 4.25, 4.912, 5.6], [7.197, 6.179, 5.193, 4.266, 3.461, 2.887, 2.671, 2.841, 3.286, 3.876, 4.532, 5.217, 5.915], [6.525, 5.526, 4.567, 3.691, 2.984, 2.586, 2.599, 2.959, 3.514, 4.156, 4.836, 5.532, 6.234], [5.892, 4.918, 4.003, 3.208, 2.648, 2.467, 2.687, 3.174, 3.789, 4.458, 5.15, 5.851, 6.556], [5.302, 4.366, 3.519, 2.844, 2.472, 2.502, 2.874, 3.438, 4.087, 4.771, 5.468, 6.172, 6.878], [4.762, 3.885, 3.137, 2.616, 2.441, 2.645, 3.118, 3.728, 4.396, 5.088, 5.789, 6.494, 7.202], [4.286, 3.494, 2.873, 2.522, 2.523, 2.853, 3.392, 4.03, 4.71, 5.407, 6.111, 6.817, 7.525]];
const W_LISTMLE: [number, number, number] = [0.3212, 0.616, 0.2493];   // [lex, dense, li]
const W_LISTNET: [number, number, number] = [0.6886, 1.021, 0.0306];
const PL_LOGPROB_STAR = -39.1967;

// Panel D — boosting curve, the XOR instance, held-out metrics, the headline CI.
const BOOST_CURVE: [number, number][] = [[0, 0.0658], [1, 0.6935], [2, 0.7038], [5, 0.7171], [10, 0.7431], [15, 0.7627], [20, 0.754], [25, 0.7642], [30, 0.7716]];
const CONSTRUCTED = { lambdamart: 1.0, bestLinear: 0.89 };
type Method = 'ranknet' | 'lambdarank' | 'listnet' | 'listmle' | 'lambdamart' | 'rrf' | 'best_leg';
const METHODS: Method[] = ['ranknet', 'lambdarank', 'listnet', 'listmle', 'lambdamart', 'rrf', 'best_leg'];
const METHOD_LABEL: Record<Method, string> = {
  ranknet: 'RankNet', lambdarank: 'LambdaRank', listnet: 'ListNet', listmle: 'ListMLE',
  lambdamart: 'LambdaMART', rrf: 'RRF', best_leg: 'best leg (dense)',
};
const METHOD_NDCG: Record<Method, number> = {
  ranknet: 0.7668, lambdarank: 0.7697, listnet: 0.7729, listmle: 0.7689, lambdamart: 0.7716, rrf: 0.7326, best_leg: 0.6889,
};
const METHOD_RECALL: Record<Method, number> = {
  ranknet: 0.6375, lambdarank: 0.6437, listnet: 0.6438, listmle: 0.6437, lambdamart: 0.6562, rrf: 0.6187, best_leg: 0.5875,
};
const HEADLINE_CI = { n: 16, mean: 0.7729, se: 0.0381, lo: 0.6981, hi: 0.8476 };  // the top method's NDCG CI

const ACCENT = 'var(--color-accent)';
const RN_COLOR = '#7a8a99';     // RankNet (uniform weight)
const LR_COLOR = '#5fa873';     // LambdaRank (ΔNDCG weight)
const GRADE_COLOR = ['#cdd3da', '#b9c79a', '#86a35e', '#4f7a3a'];      // grade 0,1,2,3
const UP_COLOR = '#5fa873';     // pulled up (negative λ)
const DOWN_COLOR = '#c25b6b';   // pulled down (positive λ)
const METHOD_COLOR: Record<Method, string> = {
  ranknet: '#7a8a99', lambdarank: '#5fa873', listnet: '#6a8caf', listmle: '#9b6abf',
  lambdamart: '#c08457', rrf: '#b07d9b', best_leg: '#9aa3ad',
};
const fmt = (x: number, n = 3) => x.toFixed(n);
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const disc = (r: number) => 1 / Math.log2(r + 1);                     // log2 discount, 1-indexed
const gainExp = (g: number) => 2 ** g - 1;

// The score-space λ field of the 3-doc toy (grades 2,1,0), recomputed in CLOSED FORM — the same field
// the notebook tests. Pairs (i,j) with y_i > y_j: (0,1),(0,2),(1,2). LambdaRank weights each by |ΔNDCG|.
function toyField(s: number[], lambdarank: boolean): number[] {
  const G = TOY_GRADES.map(gainExp);                                  // [3,1,0]
  const pairs: [number, number][] = [[0, 1], [0, 2], [1, 2]];
  const order = [0, 1, 2].sort((i, j) => (s[j] - s[i]) || (i - j));   // stable descending
  const rank = [0, 0, 0];
  order.forEach((d, r) => { rank[d] = r + 1; });
  const D = [0, 1, 2].map((d) => disc(rank[d]));
  const idcg = G[0] * disc(1) + G[1] * disc(2) + G[2] * disc(3);
  const lam = [0, 0, 0];
  for (const [i, j] of pairs) {
    const g = sigmoid(-(s[i] - s[j]));
    const w = lambdarank ? Math.abs((G[i] - G[j]) * (D[i] - D[j])) / idcg : 1;
    lam[i] += -g * w;
    lam[j] += g * w;
  }
  return lam;
}

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
      <span style={{ minWidth: '13rem' }}>{label} = <strong>{display}</strong></span>
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

// ===== Panel A — RankNet vs LambdaRank λ forces (top-concentration) =================================
function ForcesPanel({ rule, setRule }: { rule: 'ranknet' | 'lambdarank'; setRule: (r: 'ranknet' | 'lambdarank') => void }) {
  const lam = rule === 'ranknet' ? RANKNET_LAMBDA : LAMBDARANK_LAMBDA;
  const mass = rule === 'ranknet' ? RANKNET_MASS : LAMBDARANK_MASS;
  const share = TOP3_SHARE[rule];
  const maxAbsLam = Math.max(...lam.map(Math.abs));
  const maxMass = Math.max(...mass);
  const W = 540, rowH = 24, x0 = 8, barX = 250, barW = 150, H = WORKED_RANKING.length * rowH + 30;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        <button type="button" style={pill(rule === 'ranknet')} onClick={() => setRule('ranknet')}>RankNet (uniform weight)</button>
        <button type="button" style={pill(rule === 'lambdarank')} onClick={() => setRule('lambdarank')}>LambdaRank (ΔNDCG weight)</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="worked query top-k with per-document forces" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <text x={x0} y={14} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rank · doc · grade</text>
        <text x={barX + barW / 2} y={14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">← pull down · λ force · pull up →</text>
        {WORKED_RANKING.map((doc, i) => {
          const y = 24 + i * rowH;
          const l = lam[i];
          const g = WORKED_GRADE[i];
          const bl = (Math.abs(l) / maxAbsLam) * (barW / 2);
          const cx = barX + barW / 2;
          return (
            <g key={doc}>
              <rect x={x0} y={y - 10} width={15} height={15} rx={3} fill={GRADE_COLOR[g]} stroke={g === 3 ? ACCENT : 'var(--color-border)'} strokeWidth={g === 3 ? 2 : 1} />
              <text x={x0 + 24} y={y + 1} fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">{i + 1}.  doc {doc}  ·  g{g}</text>
              <line x1={cx} y1={y - 9} x2={cx} y2={y + 7} stroke="var(--color-border)" strokeWidth={1} />
              {l < 0
                ? <rect x={cx} y={y - 5} width={bl} height={10} fill={UP_COLOR} fillOpacity={0.8} />
                : <rect x={cx - bl} y={y - 5} width={bl} height={10} fill={DOWN_COLOR} fillOpacity={0.8} />}
            </g>
          );
        })}
      </svg>
      <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0.2rem', fontFamily: 'var(--font-sans)' }}>
        gradient mass by rank position (where the force lives)
      </div>
      <svg viewBox="0 0 540 80" role="img" aria-label="gradient mass by rank" style={{ width: '100%', maxWidth: 540, height: 'auto', display: 'block' }}>
        {mass.map((m, i) => {
          const x = 8 + i * 52;
          const h = (m / maxMass) * 58;
          return (
            <g key={i}>
              <rect x={x} y={66 - h} width={40} height={h} rx={2} fill={rule === 'ranknet' ? RN_COLOR : LR_COLOR} fillOpacity={i < 3 ? 0.95 : 0.45} />
              <text x={x + 20} y={78} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{i + 1}</text>
            </g>
          );
        })}
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
        <Readout label="top-3 ranks' share of gradient mass" value={fmt(share, 3)} accent={rule === 'lambdarank'} />
        <Readout label="|ΔNDCG| weight: head (1,2) vs tail (9,10)" value={`${fmt(WEIGHT_HEAD_1_2)} vs ${fmt(WEIGHT_TAIL_9_10)}`} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Toggle the rule. <span style={{ color: RN_COLOR }}><strong>RankNet</strong></span> weights every preference
        pair equally, so its force is spread down the list (top-3 share {fmt(TOP3_SHARE.ranknet, 2)}).
        <span style={{ color: LR_COLOR }}><strong> LambdaRank</strong></span> multiplies each pair by the
        |ΔNDCG| a swap would cause; because the discount marginal is steep at the head ({fmt(WEIGHT_HEAD_1_2)}) and
        flat in the tail ({fmt(WEIGHT_TAIL_9_10)}), the gradient mass <strong>concentrates at the top</strong> (share
        {' '}{fmt(TOP3_SHARE.lambdarank, 2)}). Same documents, same scores — only the weighting moves.
      </p>
    </div>
  );
}

// ===== Panel B — is λ a gradient? (the score-space field, recomputed live) ==========================
function FieldPanel({ field, setField, probe, setProbe }: {
  field: 'ranknet' | 'lambdarank'; setField: (f: 'ranknet' | 'lambdarank') => void; probe: number; setProbe: (v: number) => void;
}) {
  const isLR = field === 'lambdarank';
  const sb = TOY_S_BASE[1], sc = TOY_S_BASE[2];                        // s_b = 1, s_c = 0 fixed
  const saMin = 0.2, saMax = 1.8;
  // sample λ_a(s_a) and λ_b(s_a) across the sweep (s_a crosses the swap at s_a = s_b)
  const N = 161;
  const sas = Array.from({ length: N }, (_, i) => saMin + (saMax - saMin) * i / (N - 1));
  const la = sas.map((sa) => toyField([sa, sb, sc], isLR)[0]);
  const lb = sas.map((sa) => toyField([sa, sb, sc], isLR)[1]);
  const all = [...la, ...lb];
  const lo = Math.min(...all), hi = Math.max(...all);
  const W = 540, H = 230, padL = 44, padR = 12, padT = 14, padB = 30;
  const px = (sa: number) => padL + (W - padL - padR) * (sa - saMin) / (saMax - saMin);
  const py = (v: number) => H - padB - (H - padT - padB) * (v - lo) / (hi - lo + 1e-9);
  // split each curve at the swap (s_a = s_b) so a discontinuity shows as a true break
  const swapX = px(sb);
  // split each curve at the swap boundary so a discontinuity renders as a true break (two paths)
  const seg = (vals: number[]) => {
    let dL = '', dR = '';
    sas.forEach((sa, i) => {
      const p = `${px(sa)} ${py(vals[i])}`;
      if (sa <= sb) dL += (dL ? ' L ' : 'M ') + p;
      else dR += (dR ? ' L ' : 'M ') + p;
    });
    return [dL, dR];
  };
  const [laL, laR] = seg(la);
  const [lbL, lbR] = seg(lb);
  const saProbe = saMin + (saMax - saMin) * probe;
  const lamProbe = toyField([saProbe, sb, sc], isLR);
  const jump = isLR ? SWAP_JUMP_LAMBDARANK : SWAP_JUMP_RANKNET;
  const circ = isLR ? CIRC_LAMBDARANK : CIRC_RANKNET;
  const jacAsym = isLR ? JAC_ASYM_LAMBDARANK : JAC_ASYM_RANKNET;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        <button type="button" style={pill(field === 'ranknet')} onClick={() => setField('ranknet')}>RankNet field</button>
        <button type="button" style={pill(field === 'lambdarank')} onClick={() => setField('lambdarank')}>LambdaRank field</button>
      </div>
      <Slider label="probe score s₀ (doc 0, crossing s₁=1)" value={probe} min={0} max={1} step={0.01}
        onChange={setProbe} display={`s₀ = ${fmt(saProbe, 2)}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="score-space lambda field, doc 0 crossing the swap boundary" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {py(0) > padT && py(0) < H - padB && (
          <line x1={padL} y1={py(0)} x2={W - padR} y2={py(0)} stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="2 3" />
        )}
        {/* swap boundary s_a = s_b */}
        <line x1={swapX} y1={padT} x2={swapX} y2={H - padB} stroke={ACCENT} strokeWidth={1} strokeDasharray="4 3" />
        <text x={swapX + 3} y={padT + 9} fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">swap s₀=s₁</text>
        {/* λ_a (doc 0) and λ_b (doc 1), each split at the boundary */}
        <path d={laL} fill="none" stroke={LR_COLOR} strokeWidth={2} />
        <path d={laR} fill="none" stroke={LR_COLOR} strokeWidth={2} />
        <path d={lbL} fill="none" stroke={DOWN_COLOR} strokeWidth={1.6} strokeDasharray="4 2" />
        <path d={lbR} fill="none" stroke={DOWN_COLOR} strokeWidth={1.6} strokeDasharray="4 2" />
        <text x={W - padR - 4} y={py(la[N - 1]) - 4} textAnchor="end" fontSize={8.5} fill={LR_COLOR} fontFamily="var(--font-sans)">λ₀ (doc 0)</text>
        <text x={W - padR - 4} y={py(lb[N - 1]) + 10} textAnchor="end" fontSize={8.5} fill={DOWN_COLOR} fontFamily="var(--font-sans)">λ₁ (doc 1)</text>
        <line x1={px(saProbe)} y1={padT} x2={px(saProbe)} y2={H - padB} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="2 2" />
        <circle cx={px(saProbe)} cy={py(lamProbe[0])} r={4} fill={LR_COLOR} />
        <text x={px(0.5 * (saMin + saMax))} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">probe score s₀ (doc 0)</text>
        <text x={14} y={(padT + H - padB) / 2} textAnchor="middle" fontSize={9} fill={ACCENT} fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${(padT + H - padB) / 2})`}>λ force</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="λ₀ at the probe" value={fmt(lamProbe[0], 4)} accent />
        <Readout label="jump in λ₀ across the swap" value={fmt(jump, 4)} accent={isLR} />
        <Readout label="loop circulation ∮λ·ds" value={fmt(circ, 4)} accent={isLR} />
        <Readout label="within-cell Jacobian asymmetry" value={fmt(jacAsym, 4)} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Drag s₀ across the <span style={{ color: ACCENT }}>swap boundary</span> s₀ = s₁. For the
        <span style={{ color: RN_COLOR }}><strong> RankNet field</strong></span> the force λ₀ is
        <strong> continuous</strong> — the field is the gradient of the RankNet loss, its Jacobian is symmetric
        ({fmt(JAC_ASYM_RANKNET, 2)}), and a closed loop integrates to {fmt(CIRC_RANKNET, 2)}. The
        <span style={{ color: LR_COLOR }}><strong> LambdaRank field</strong></span> is symmetric <em>within</em> each
        ranking cell (asymmetry {fmt(JAC_ASYM_LAMBDARANK, 2)}) but <strong>jumps</strong> by {fmt(SWAP_JUMP_LAMBDARANK, 3)} as
        docs 0 and 1 swap — a spectator pair's |ΔNDCG| weight is discontinuous — so the loop integrates to a
        nonzero {fmt(CIRC_LAMBDARANK, 3)}. A discontinuous field is the gradient of no scalar loss: LambdaRank
        optimizes nothing, globally.
      </p>
    </div>
  );
}

// ===== Panel C — listwise objectives: the convex loss bowl ==========================================
function ListwisePanel({ obj, setObj, t, setT }: {
  obj: 'listmle' | 'listnet'; setObj: (o: 'listmle' | 'listnet') => void; t: number; setT: (v: number) => void;
}) {
  const bowl = obj === 'listmle' ? LISTMLE_BOWL : LISTNET_BOWL;
  const wOpt = obj === 'listmle' ? W_LISTMLE : W_LISTNET;
  const n = BOWL_GRID.length;
  let lo = Infinity, hi = -Infinity;
  for (const row of bowl) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  // bilinear interpolation of the bowl at (wd, wl). The .py builds bowl[i_dense][j_li] (outer loop over
  // w_dense, inner over w_li), so i0 indexes w_dense and j0 indexes w_li — read bowl[i0][j0].
  const lerpBowl = (wd: number, wl: number) => {
    const gi = (wd - BOWL_GRID[0]) / (BOWL_GRID[n - 1] - BOWL_GRID[0]) * (n - 1);   // w_dense index
    const gj = (wl - BOWL_GRID[0]) / (BOWL_GRID[n - 1] - BOWL_GRID[0]) * (n - 1);   // w_li index
    const i0 = Math.max(0, Math.min(n - 2, Math.floor(gi))), j0 = Math.max(0, Math.min(n - 2, Math.floor(gj)));
    const fi = gi - i0, fj = gj - j0;
    const a = bowl[i0][j0], b = bowl[i0 + 1][j0], c = bowl[i0][j0 + 1], d = bowl[i0 + 1][j0 + 1];
    return a * (1 - fi) * (1 - fj) + b * fi * (1 - fj) + c * (1 - fi) * fj + d * fi * fj;
  };
  // a straight descent path from the corner (-3,-3) to the optimum
  const wdPath = -3 + (wOpt[1] - (-3)) * t;
  const wlPath = -3 + (wOpt[2] - (-3)) * t;
  const lossHere = lerpBowl(wdPath, wlPath);
  const W = 300, H = 300, pad = 28;
  const cell = (W - 2 * pad) / n;
  const col = (v: number) => {
    const u = (v - lo) / (hi - lo + 1e-9);                            // 0 = basin, 1 = rim
    const r = Math.round(70 + 150 * u), g = Math.round(130 - 60 * u), b = Math.round(110 - 40 * u);
    return `rgb(${r},${g},${b})`;
  };
  // map a weight to the CENTER of its grid cell (w_dense on x, w_li on y) so the marker sits on the heatmap
  const step = BOWL_GRID[1] - BOWL_GRID[0];
  const gx = (wd: number) => pad + ((wd - BOWL_GRID[0]) / step + 0.5) * cell;
  const gy = (wl: number) => H - pad - ((wl - BOWL_GRID[0]) / step + 0.5) * cell;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        <button type="button" style={pill(obj === 'listmle')} onClick={() => setObj('listmle')}>ListMLE (Plackett–Luce)</button>
        <button type="button" style={pill(obj === 'listnet')} onClick={() => setObj('listnet')}>ListNet (top-one)</button>
      </div>
      <Slider label="descend the bowl: corner → optimum" value={t} min={0} max={1} step={0.01}
        onChange={setT} display={`t = ${fmt(t, 2)}`} />
      <div style={{ display: 'flex', gap: '1.2rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="convex listwise loss bowl over two weights" style={{ width: '100%', maxWidth: 300, height: 'auto', display: 'block' }}>
          {bowl.map((row, iD) => (
            <g key={iD}>
              {row.map((v, jL) => (
                <rect key={jL} x={pad + iD * cell} y={H - pad - (jL + 1) * cell} width={cell + 0.5} height={cell + 0.5} fill={col(v)} fillOpacity={0.85} />
              ))}
            </g>
          ))}
          <line x1={gx(0)} y1={pad} x2={gx(0)} y2={H - pad} stroke="var(--color-border)" strokeWidth={0.5} />
          <line x1={pad} y1={gy(0)} x2={W - pad} y2={gy(0)} stroke="var(--color-border)" strokeWidth={0.5} />
          {/* descent path + marker */}
          <line x1={gx(-3)} y1={gy(-3)} x2={gx(wOpt[1])} y2={gy(wOpt[2])} stroke="#fff" strokeWidth={1} strokeDasharray="3 2" strokeOpacity={0.7} />
          <circle cx={gx(wOpt[1])} cy={gy(wOpt[2])} r={4} fill="#fff" stroke={ACCENT} strokeWidth={1.5} />
          <circle cx={gx(wdPath)} cy={gy(wlPath)} r={5} fill={ACCENT} />
          <text x={gx(wOpt[1]) + 6} y={gy(wOpt[2]) - 4} fontSize={8.5} fill="#fff" fontFamily="var(--font-sans)">w*</text>
          <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">w · dense</text>
          <text x={10} y={H / 2} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 10 ${H / 2})`}>w · late-interaction</text>
        </svg>
        <div style={{ flex: '1 1 200px' }}>
          <div style={{ display: 'flex', gap: '1.4rem', flexWrap: 'wrap' }}>
            <Readout label="loss at the marker" value={fmt(lossHere, 2)} />
            <Readout label="loss at the optimum w*" value={fmt(lerpBowl(wOpt[1], wOpt[2]), 2)} accent />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', lineHeight: 1.5 }}>
            A single basin: the loss <strong>decreases monotonically</strong> as the slider walks toward w*, the
            signature of a <strong>convex</strong> objective (log-sum-exp minus linear, with a positive-semidefinite
            Hessian). Unlike LambdaRank's field, ListMLE <em>is</em> the negative log-likelihood of the Plackett–Luce
            model given the ideal permutation — a proper loss with a single optimum (here ListMLE's log-likelihood at
            w* is {fmt(PL_LOGPROB_STAR, 2)}). ListNet minimizes the top-one cross-entropy; both are convex for the
            linear scorer.
          </p>
        </div>
      </div>
    </div>
  );
}

// ===== Panel D — LambdaMART boosting + the method comparison ========================================
function BoostPanel({ metric, setMetric, rounds, setRounds }: {
  metric: 'ndcg' | 'recall'; setMetric: (m: 'ndcg' | 'recall') => void; rounds: number; setRounds: (v: number) => void;
}) {
  const table = metric === 'ndcg' ? METHOD_NDCG : METHOD_RECALL;
  // interpolate the boost curve at `rounds`
  const ndcgAt = (() => {
    let prev = BOOST_CURVE[0];
    for (const pt of BOOST_CURVE) { if (pt[0] <= rounds) prev = pt; else break; }
    const next = BOOST_CURVE.find((p) => p[0] > rounds);
    if (!next || next[0] === prev[0]) return prev[1];
    const f = (rounds - prev[0]) / (next[0] - prev[0]);
    return prev[1] + f * (next[1] - prev[1]);
  })();
  const W = 540, H = 150, padL = 40, padR = 12, padT = 12, padB = 26;
  const cMin = 0.0, cMax = 0.8;
  const cx = (r: number) => padL + (W - padL - padR) * r / 30;
  const cy = (v: number) => H - padB - (H - padT - padB) * (v - cMin) / (cMax - cMin);
  const curveD = BOOST_CURVE.map((p, i) => (i ? 'L' : 'M') + cx(p[0]) + ' ' + cy(p[1])).join(' ');
  // bars
  const BW = 540, BH = 196, bpadL = 100, bpadR = 64, bpadT = 8, browH = 26;
  const xMax = 0.85;
  const bx = (v: number) => bpadL + (BW - bpadL - bpadR) * (v / xMax);
  const gapRrf = table.lambdamart - table.rrf;
  const gapLeg = table.lambdamart - table.best_leg;
  return (
    <div>
      <Slider label="LambdaMART boosting rounds" value={rounds} min={0} max={30} step={1}
        onChange={(v) => setRounds(Math.round(v))} display={`${rounds}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="LambdaMART boosting NDCG over rounds" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0.0, 0.2, 0.4, 0.6, 0.8].map((v) => (
          <text key={v} x={padL - 5} y={cy(v) + 3} textAnchor="end" fontSize={8} fill={ACCENT} fontFamily="var(--font-sans)">{v.toFixed(1)}</text>
        ))}
        <path d={curveD} fill="none" stroke={METHOD_COLOR.lambdamart} strokeWidth={2} />
        {BOOST_CURVE.map((p) => <circle key={p[0]} cx={cx(p[0])} cy={cy(p[1])} r={2.5} fill={METHOD_COLOR.lambdamart} />)}
        <line x1={cx(rounds)} y1={padT} x2={cx(rounds)} y2={H - padB} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="2 2" />
        <circle cx={cx(rounds)} cy={cy(ndcgAt)} r={4} fill={ACCENT} />
        <text x={(padL + W) / 2} y={H - 5} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">boosting rounds → (test NDCG@10 climbs, then plateaus)</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', margin: '0.4rem 0', flexWrap: 'wrap' }}>
        <Readout label={`LambdaMART test NDCG@10 at ${rounds} rounds`} value={fmt(ndcgAt, 4)} accent />
        <Readout label="constructed XOR: LambdaMART vs best linear" value={`${fmt(CONSTRUCTED.lambdamart, 2)} vs ${fmt(CONSTRUCTED.bestLinear, 2)}`} accent />
      </div>
      <div style={{ display: 'flex', gap: '0.5rem', margin: '0.6rem 0', flexWrap: 'wrap' }}>
        <button type="button" style={pill(metric === 'ndcg')} onClick={() => setMetric('ndcg')}>NDCG@10</button>
        <button type="button" style={pill(metric === 'recall')} onClick={() => setMetric('recall')}>recall@10</button>
      </div>
      <svg viewBox={`0 0 ${BW} ${BH}`} role="img" aria-label="held-out metric bars per method" style={{ width: '100%', maxWidth: BW, height: 'auto', display: 'block' }}>
        {METHODS.map((m, i) => {
          const y = bpadT + i * browH;
          const v = table[m];
          const learned = m !== 'rrf' && m !== 'best_leg';
          return (
            <g key={m}>
              <text x={bpadL - 8} y={y + browH / 2 + 1} textAnchor="end" fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">{METHOD_LABEL[m]}</text>
              <rect x={bpadL} y={y + 4} width={Math.max(0, bx(v) - bpadL)} height={browH - 12} rx={2}
                fill={METHOD_COLOR[m]} fillOpacity={learned ? 0.9 : 0.5}
                stroke={m === 'lambdamart' ? ACCENT : 'none'} strokeWidth={m === 'lambdamart' ? 1.5 : 0} />
              <text x={bx(v) + 5} y={y + browH / 2 + 1} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{fmt(v, 3)}</text>
            </g>
          );
        })}
        <line x1={bx(table.rrf)} y1={bpadT} x2={bx(table.rrf)} y2={BH - 6} stroke={METHOD_COLOR.rrf} strokeWidth={1} strokeDasharray="3 2" />
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.2rem', flexWrap: 'wrap' }}>
        <Readout label="LambdaMART − RRF / − best leg" value={`+${fmt(gapRrf, 3)} / +${fmt(gapLeg, 3)}`} />
        <Readout label={`top method ${metric.toUpperCase()} 95% CI`} value={`[${fmt(HEADLINE_CI.lo, 3)}, ${fmt(HEADLINE_CI.hi, 3)}]`} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        <span style={{ color: METHOD_COLOR.lambdamart }}><strong>LambdaMART</strong></span> boosts the LambdaRank λ
        into regression trees; its test NDCG climbs and plateaus. On the constructed XOR instance the trees reach
        NDCG {fmt(CONSTRUCTED.lambdamart, 1)} where every linear scorer is capped at {fmt(CONSTRUCTED.bestLinear, 2)} — the
        nonlinearity the linear models cannot express. On the real leg features all five learned methods clearly beat
        <span style={{ color: METHOD_COLOR.rrf }}> RRF</span> (+{fmt(gapRrf, 3)}) and the best single leg
        (+{fmt(gapLeg, 3)}), but cluster <em>within</em> the confidence interval [{fmt(HEADLINE_CI.lo, 2)}, {fmt(HEADLINE_CI.hi, 2)}] —
        on this forgiving corpus the seed-free wins are structural (the top-concentration, the XOR ceiling), not the
        aggregate deltas.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'forces' | 'field' | 'listwise' | 'boost';
const TEX: Record<Panel, string> = {
  forces: '\\lambda_{ij}=-\\sigma\\!\\big(-(s_i-s_j)\\big)\\,\\big|\\Delta\\mathrm{NDCG}_{ij}\\big|,\\quad \\big|\\Delta\\mathrm{NDCG}_{ij}\\big|=\\frac{|G(g_i)-G(g_j)|\\,|D(r_i)-D(r_j)|}{\\mathrm{IDCG}}',
  field: '\\partial\\lambda_i/\\partial s_j = \\partial\\lambda_j/\\partial s_i\\ \\text{(RankNet, everywhere)};\\qquad \\oint \\boldsymbol{\\lambda}\\cdot d\\mathbf{s}\\neq 0\\ \\text{(LambdaRank)}',
  listwise: 'L_{\\text{ListMLE}}(s)=-\\log P(\\pi^\\*\\mid s)=\\sum_{r}\\Big[\\operatorname{logsumexp}\\big(s_{\\geq r}\\big)-s_{\\pi^\\*(r)}\\Big]',
  boost: 'F_T(\\mathbf{x})=\\sum_{t=1}^{T}\\nu\\,h_t(\\mathbf{x}),\\qquad h_t \\leftarrow \\text{tree fit to } -\\lambda^{(t)}',
};

export default memo(function LambdaRankLaboratory() {
  const [panel, setPanel] = useState<Panel>('forces');
  const [rule, setRule] = useState<'ranknet' | 'lambdarank'>('lambdarank');
  const [field, setField] = useState<'ranknet' | 'lambdarank'>('lambdarank');
  const [probe, setProbe] = useState(0.3);
  const [obj, setObj] = useState<'listmle' | 'listnet'>('listmle');
  const [t, setT] = useState(0);
  const [metric, setMetric] = useState<'ndcg' | 'recall'>('ndcg');
  const [rounds, setRounds] = useState(30);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="lambdarank-listwise" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'forces')} onClick={() => setPanel('forces')}>A · λ forces & top-concentration</button>
        <button type="button" style={pill(panel === 'field')} onClick={() => setPanel('field')}>B · is λ a gradient?</button>
        <button type="button" style={pill(panel === 'listwise')} onClick={() => setPanel('listwise')}>C · listwise convex bowl</button>
        <button type="button" style={pill(panel === 'boost')} onClick={() => setPanel('boost')}>D · LambdaMART & comparison</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem', overflowX: 'auto' }} />
      {panel === 'forces' && <ForcesPanel rule={rule} setRule={setRule} />}
      {panel === 'field' && <FieldPanel field={field} setField={setField} probe={probe} setProbe={setProbe} />}
      {panel === 'listwise' && <ListwisePanel obj={obj} setObj={setObj} t={t} setT={setT} />}
      {panel === 'boost' && <BoostPanel metric={metric} setMetric={setMetric} rounds={rounds} setRounds={setRounds} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_DOCS} synthetic finance documents, {N_QUERIES} queries ({N_TRAIN} train / {N_TEST} test); features are the
        three legs' per-document scores, grades are the top-{TOPK} exact-MaxSim oracle tertiles; worked query {WORKED_Q}.
        Numbers mirror <code>lambdarank_lambdamart_listwise.py</code>; the lab recomputes the score-space λ field, the
        bowl marker, the boosting interpolation, and the bar geometry in closed form.
      </p>
    </div>
  );
});
