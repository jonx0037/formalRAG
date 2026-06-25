import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Cross-Modal Alignment Laboratory — three panels for the `cross-modal-alignment` topic:
 *   A. The decomposition. The cross-modal alignment splits orthogonally, L_align = gap^2 + dispersion
 *      (a Frobenius-Pythagoras split of the per-pair difference matrix). A beta slider scans the
 *      modality tilt: the gap^2 (coherent) part grows and the dispersion (incoherent) part shrinks,
 *      always summing to L_align. A 2D PCA projection shows the two cones (text left, chart right) and
 *      the gap vector between their centroids.
 *   B. The headline. Under a gap-removing offset c_j -> c_j + alpha g, every MIPS score changes by the
 *      PER-QUERY constant alpha <t_i, g>, so recall@1 is EXACTLY invariant — recomputed LIVE in TS from
 *      the baked score matrix, it does not move as the slider runs. Cosine, which renormalizes the
 *      offset keys, is NOT gap-invariant: its recall (baked) rises and peaks as the gap is removed.
 *   C. The cone effect and temperature. A deterministic full-batch CLIP-loss descent closes the gap at
 *      moderate temperature; lower temperature preserves a LARGER residual gap. A tau slider reads the
 *      baked temperature ladder (gap before -> after, with the alignment and union uniformity).
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): DECOMP, PROJ_TEXT, PROJ_CHART, SECTOR, ALPHA_GRID, COSINE_RECALL,
 * SCORES, GAP_VEC_SCORE, TRUTH, TAU_GRID, RESIDUAL_GAP are mirrored TO THE DECIMAL from
 * notebooks/cross-modal-alignment/cross_modal_alignment.py (viz_constants()). The lab recomputes only
 * CLOSED FORM in TS — Panel A's gap^2/dispersion bars from DECOMP, Panel B's MIPS recall@1 from
 * SCORES + GAP_VEC_SCORE (argsort-invariant, so it stays flat), exactly as the source asserts
 * (test_mips_ranking_is_gap_invariant, test_alignment_decomposition_is_orthogonal,
 * test_temperature_preserves_larger_residual_gap). Change a number here -> change it there, and re-run
 * the notebook. Sliders only (no d3 drag).
 */

// --- baked from viz_constants() -------------------------------------------------------
const DECOMP = [
  { beta: 0.0, L_align: 0.4226, gap2: 0.0116, dispersion: 0.411, gap: 0.1079 },
  { beta: 0.1, L_align: 0.4416, gap2: 0.0353, dispersion: 0.4063, gap: 0.188 },
  { beta: 0.2, L_align: 0.5072, gap2: 0.1185, dispersion: 0.3887, gap: 0.3443 },
  { beta: 0.3, L_align: 0.6414, gap2: 0.2894, dispersion: 0.352, gap: 0.538 },
  { beta: 0.4, L_align: 0.8547, gap2: 0.5623, dispersion: 0.2924, gap: 0.7498 },
  { beta: 0.5, L_align: 1.122, gap2: 0.9076, dispersion: 0.2144, gap: 0.9527 },
  { beta: 0.6, L_align: 1.3816, gap2: 1.248, dispersion: 0.1335, gap: 1.1172 },
  { beta: 0.7, L_align: 1.5768, gap2: 1.509, dispersion: 0.0677, gap: 1.2284 },
];
const PROJ_TEXT = [
  [-0.447, -0.224], [-0.502, -0.138], [-0.487, -0.332], [-0.471, -0.145], [-0.429, -0.235],
  [-0.385, -0.328], [-0.343, -0.466], [-0.479, -0.372], [-0.591, -0.266], [-0.44, -0.389],
  [-0.337, -0.29], [-0.53, -0.371], [-0.52, 0.623], [-0.507, 0.325], [-0.492, 0.614],
  [-0.565, 0.522], [-0.566, 0.448], [-0.643, 0.596], [-0.435, -0.024], [-0.38, 0.074],
  [-0.471, -0.008], [-0.446, 0.194], [-0.539, 0.044], [-0.423, -0.073],
];
const PROJ_CHART = [
  [0.477, -0.134], [0.337, -0.295], [0.601, -0.223], [0.458, -0.129], [0.494, -0.303],
  [0.5, -0.168], [0.468, -0.413], [0.511, -0.405], [0.352, -0.325], [0.417, -0.281],
  [0.553, -0.217], [0.457, -0.362], [0.587, 0.516], [0.502, 0.402], [0.525, 0.61],
  [0.462, 0.503], [0.366, 0.505], [0.547, 0.621], [0.456, -0.043], [0.52, -0.016],
  [0.355, 0.018], [0.422, 0.123], [0.544, 0.14], [0.518, 0.096],
];
const SECTOR = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3];

const ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5];
const COSINE_RECALL = [0.4583, 0.7917, 0.875, 0.9583, 1.0, 1.0, 1.0];
const SCORES = [
  [0.4534, 0.471, 0.2018, 0.3965, 0.2206, 0.259, -0.0027, 0.1076, 0.0437, 0.1089, 0.0725, 0.1357, -0.0432, 0.0281, -0.0926, -0.0515, -0.0105, -0.2835, 0.1563, -0.0063, 0.1526, 0.0183, 0.0171, 0.1223],
  [0.2733, 0.5231, 0.2559, 0.3099, 0.241, 0.2703, -0.0235, 0.0122, 0.0642, 0.0482, 0.0436, 0.0344, -0.0485, 0.0054, -0.1073, 0.0344, -0.0141, -0.1171, 0.1614, -0.0143, 0.1949, 0.0356, -0.0065, 0.0596],
  [0.3193, 0.3902, 0.2832, 0.2091, 0.2232, 0.1919, 0.0285, 0.032, 0.1219, 0.1436, 0.106, 0.1631, -0.2163, -0.008, -0.2693, -0.0612, -0.0988, -0.3796, 0.0889, -0.1067, 0.0696, -0.0149, -0.0393, -0.0447],
  [0.3908, 0.5006, 0.1826, 0.4975, 0.2156, 0.2689, 0.0121, 0.0715, 0.1033, 0.0585, 0.0154, 0.1229, -0.123, 0.0503, -0.0298, -0.0224, 0.0652, -0.1338, 0.1846, 0.0925, 0.2393, 0.0583, -0.0265, 0.0911],
  [0.3149, 0.502, 0.2484, 0.3684, 0.4076, 0.2047, 0.032, 0.1398, 0.1688, 0.0752, 0.0802, 0.1898, -0.0851, 0.0179, -0.0428, -0.0582, 0.0678, -0.2621, 0.2704, 0.0388, 0.2773, 0.0335, 0.0732, 0.1822],
  [0.3572, 0.5228, 0.2751, 0.2972, 0.1557, 0.5167, 0.2064, 0.2911, 0.1734, 0.2116, 0.0917, 0.2103, -0.1395, 0.0406, -0.1678, 0.1021, 0.0847, -0.2347, 0.081, -0.0615, 0.0444, -0.0114, -0.1654, -0.0376],
  [0.0756, 0.2203, 0.0711, 0.2077, 0.1539, 0.1366, 0.5995, 0.4899, 0.4694, 0.5219, 0.3346, 0.4582, -0.195, -0.106, -0.1688, -0.0976, -0.0219, -0.2173, 0.184, 0.1844, 0.2475, 0.2589, 0.0482, 0.1657],
  [0.0389, 0.2802, 0.0276, 0.1489, 0.0763, 0.1689, 0.4071, 0.4264, 0.386, 0.4097, 0.2347, 0.3533, -0.1613, -0.0546, -0.2323, 0.0158, 0.0509, -0.2746, 0.2016, 0.0999, 0.1917, 0.2305, 0.0374, 0.0951],
  [-0.008, 0.16, -0.1071, 0.0547, 0.0083, -0.0316, 0.1827, 0.1993, 0.4786, 0.1968, 0.1582, 0.3156, -0.2305, -0.1247, -0.2136, -0.1478, 0.0168, -0.2989, -0.0092, -0.1093, -0.0373, -0.0117, -0.1585, -0.0956],
  [0.0136, 0.2817, 0.0077, 0.0992, 0.1137, 0.0519, 0.458, 0.3648, 0.3674, 0.5373, 0.2384, 0.3664, -0.1836, -0.0721, -0.2608, -0.0821, 0.0321, -0.2309, 0.1341, 0.1068, 0.0987, 0.1222, -0.0731, 0.0125],
  [0.1562, 0.3741, 0.1071, 0.3033, 0.0814, 0.1549, 0.3058, 0.4518, 0.4852, 0.3556, 0.4807, 0.45, 0.0105, -0.0369, -0.0547, 0.0814, 0.2205, -0.1508, 0.2449, 0.074, 0.177, 0.1921, 0.0755, 0.231],
  [0.0385, 0.237, -0.0609, 0.1776, 0.0513, 0.0149, 0.289, 0.3234, 0.3019, 0.3583, 0.2273, 0.3939, -0.3078, -0.0971, -0.1787, -0.1294, -0.0335, -0.2779, 0.0414, -0.0286, 0.0244, 0.0825, -0.1514, -0.0937],
  [-0.0518, -0.1282, -0.2677, -0.093, -0.2667, -0.1447, -0.1983, -0.2744, -0.1161, -0.0725, -0.1159, -0.2185, 0.283, 0.131, 0.2793, 0.2342, 0.3116, 0.2037, -0.0571, -0.1116, 0.0192, 0.0593, -0.0422, 0.0063],
  [0.0608, 0.139, -0.1032, 0.162, -0.0293, 0.0197, -0.0773, -0.0686, 0.0558, 0.0867, -0.032, 0.081, 0.0151, 0.3862, 0.1464, 0.2413, 0.3364, 0.1018, 0.0121, -0.0471, 0.0518, 0.118, -0.0209, -0.0509],
  [-0.1111, -0.1288, -0.2911, 0.0067, -0.2811, -0.2089, -0.1814, -0.1727, -0.0004, -0.0603, -0.1191, -0.0551, 0.1674, 0.1788, 0.3657, 0.2095, 0.3714, 0.2363, -0.0033, -0.0614, 0.0653, 0.1503, -0.0096, 0.1166],
  [-0.0547, 0.0124, -0.215, 0.0274, -0.2832, -0.0867, -0.2777, -0.2024, -0.1682, -0.0889, -0.1416, -0.1057, 0.0755, 0.2409, 0.1896, 0.3713, 0.3792, 0.0451, -0.1133, -0.1915, -0.0064, -0.0204, -0.0844, -0.0401],
  [-0.0324, 0.0765, -0.1484, 0.0647, -0.0925, -0.073, -0.178, -0.1348, 0.0249, -0.0381, -0.134, -0.081, 0.1171, 0.1446, 0.2299, 0.2435, 0.4565, 0.0543, 0.0257, -0.0648, 0.0659, 0.0902, -0.0524, -0.0236],
  [-0.2733, -0.1679, -0.3446, -0.1685, -0.3708, -0.2667, -0.3346, -0.4176, -0.1776, -0.1678, -0.2588, -0.2898, 0.0527, -0.0423, 0.1058, 0.0654, 0.1846, 0.1804, -0.1778, -0.23, -0.0742, -0.0785, -0.2259, -0.1638],
  [0.086, 0.2573, -0.0076, 0.2251, 0.1497, -0.0857, 0.1728, 0.1368, 0.1438, 0.1753, 0.0606, 0.1408, -0.0414, 0.0527, -0.0372, -0.052, 0.1029, -0.0897, 0.5188, 0.3754, 0.4975, 0.3899, 0.3033, 0.3264],
  [0.0519, 0.2476, -0.0044, 0.2325, 0.1488, -0.0768, 0.2445, 0.145, 0.1883, 0.3129, 0.0735, 0.1803, -0.0255, 0.1259, 0.0336, 0.046, 0.2355, 0.0411, 0.4686, 0.4984, 0.5326, 0.4628, 0.2654, 0.3965],
  [-0.0081, 0.2218, -0.0253, 0.211, 0.1294, -0.123, 0.1025, 0.0137, 0.1408, 0.1706, -0.0444, 0.0191, -0.0633, -0.1417, -0.0901, -0.0788, -0.0199, -0.1129, 0.2884, 0.3105, 0.6124, 0.2814, 0.2141, 0.3523],
  [-0.0089, 0.1282, -0.1185, 0.0298, -0.1173, -0.0678, 0.1136, 0.1039, 0.0719, 0.2169, -0.0432, -0.003, 0.0133, 0.0264, 0.0049, 0.1037, 0.2409, 0.0178, 0.3623, 0.3042, 0.3814, 0.5365, 0.2556, 0.3054],
  [0.009, 0.1613, -0.1894, 0.0818, -0.0871, -0.166, 0.0335, 0.033, 0.0103, 0.0936, -0.0398, -0.0007, -0.1177, -0.0584, -0.1718, -0.0359, 0.0012, -0.1399, 0.3231, 0.2075, 0.4163, 0.3639, 0.3069, 0.2825],
  [0.0389, 0.1807, -0.0102, 0.219, 0.1374, -0.1214, 0.2296, 0.1434, 0.1572, 0.2234, 0.1095, 0.167, -0.0577, -0.0841, 0.0205, -0.0982, 0.0329, -0.106, 0.3625, 0.2879, 0.443, 0.3009, 0.2513, 0.4214],
];
const GAP_VEC_SCORE = [0.4432, 0.4927, 0.4804, 0.4611, 0.4242, 0.3821, 0.3469, 0.4731, 0.5715, 0.4325, 0.3376, 0.5185, 0.4961, 0.482, 0.4677, 0.5365, 0.5366, 0.609, 0.4271, 0.3713, 0.4598, 0.4334, 0.5247, 0.4206];
const TRUTH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23];

const TAU_GRID = [0.04, 0.07, 0.1, 0.2, 0.3, 0.4];
const RESIDUAL_GAP = [
  { tau: 0.04, gap_before: 0.962, gap_after: 0.6398, alignment: 0.4298, uniformity: -2.5251 },
  { tau: 0.07, gap_before: 0.962, gap_after: 0.5428, alignment: 0.2992, uniformity: -2.9882 },
  { tau: 0.1, gap_before: 0.962, gap_after: 0.3076, alignment: 0.0953, uniformity: -3.0786 },
  { tau: 0.2, gap_before: 0.962, gap_after: 0.0, alignment: 0.0, uniformity: -3.0824 },
  { tau: 0.3, gap_before: 0.962, gap_after: 0.0, alignment: 0.0, uniformity: -3.0827 },
  { tau: 0.4, gap_before: 0.962, gap_after: 0.0, alignment: 0.0, uniformity: -3.0827 },
];

const TEXT_COLOR = '#7C3AED';                  // the text modality / the coherent gap
const CHART_COLOR = 'var(--color-accent)';     // the chart modality / cosine
const MIPS_COLOR = '#5fa873';                  // MIPS recall — the invariant
const GAP_COLOR = '#c2693e';                   // the gap vector
const MUTED = 'var(--color-text-secondary)';
const SECTOR_HUES = ['#7C3AED', '#2f8f5b', '#c2693e', '#3b7bb5'];

const r2 = (v: number) => Math.round(v * 100) / 100;

// closed-form: MIPS recall@1 under the gap-removing offset alpha, recomputed from the baked scores.
// The added term alpha*GAP_VEC_SCORE[i] is a PER-QUERY constant, so the argmax — and recall — never move.
function mipsRecallAt(alpha: number): number {
  let hits = 0;
  for (let i = 0; i < SCORES.length; i++) {
    const off = alpha * GAP_VEC_SCORE[i];
    let best = -Infinity, bestJ = -1;
    for (let j = 0; j < SCORES[i].length; j++) {
      const s = SCORES[i][j] + off;
      if (s > best) { best = s; bestJ = j; }
    }
    if (bestJ === TRUTH[i]) hits++;
  }
  return hits / SCORES.length;
}
// The MIPS recall curve is a pure function of the baked constants, so compute it once at module
// load — it stays flat under the offset, the whole point, but never needs to recompute per render.
const MIPS_RECALL_LIVE = ALPHA_GRID.map((a) => mipsRecallAt(a));

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt: (v: number) => string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '13rem' }}>{label} = <strong>{fmt(value)}</strong></span>
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

// ===== Panel A — the decomposition bars + the 2D cone projection ====================================
function DecompBars({ idx }: { idx: number }) {
  const pw = 360, ph = 240, padL = 40, padR = 12, padT = 16, padB = 34;
  const n = DECOMP.length;
  const gap = (pw - padL - padR) / n;
  const bw = gap * 0.5;
  const vMax = Math.max(...DECOMP.map((d) => d.L_align)) * 1.05;
  const fy = (v: number) => ph - padB - (ph - padT - padB) * (v / vMax);
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="gap squared plus dispersion summing to the alignment loss across the modality tilt beta" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={padL} y1={ph - padB} x2={pw - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={padL} y1={padT} x2={padL} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <text x={(padL + pw - padR) / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">modality tilt β</text>
      <text x={12} y={(padT + ph - padB) / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + ph - padB) / 2})`}>L_align</text>
      {DECOMP.map((d, i) => {
        const x = padL + gap * i + (gap - bw) / 2;
        const yGap = fy(d.gap2);
        const yTop = fy(d.L_align);
        const active = i === idx;
        return (
          <g key={i} opacity={active ? 1 : 0.5}>
            {/* gap^2 (coherent) at the base */}
            <rect x={r2(x)} y={r2(yGap)} width={r2(bw)} height={r2(ph - padB - yGap)} fill={TEXT_COLOR} rx={1} />
            {/* dispersion (incoherent) stacked on top */}
            <rect x={r2(x)} y={r2(yTop)} width={r2(bw)} height={r2(yGap - yTop)} fill={MIPS_COLOR} opacity={0.7} rx={1} />
            <text x={r2(x + bw / 2)} y={ph - padB + 11} textAnchor="middle" fontSize={8} fill={active ? 'var(--color-text)' : MUTED} fontFamily="var(--font-sans)">{d.beta.toFixed(1)}</text>
          </g>
        );
      })}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <rect x={padL + 4} y={padT} width={11} height={11} fill={TEXT_COLOR} rx={1.5} />
        <text x={padL + 19} y={padT + 9} fill={MUTED}>gap² (coherent)</text>
        <rect x={padL + 4} y={padT + 15} width={11} height={11} fill={MIPS_COLOR} opacity={0.7} rx={1.5} />
        <text x={padL + 19} y={padT + 24} fill={MUTED}>dispersion (incoherent)</text>
      </g>
    </svg>
  );
}

function ConeScatter() {
  const pw = 300, ph = 240, pad = 22;
  const all = [...PROJ_TEXT, ...PROJ_CHART];
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const fx = (x: number) => pad + (pw - 2 * pad) * ((x - xMin) / (xMax - xMin));
  const fy = (y: number) => ph - pad - (ph - 2 * pad) * ((y - yMin) / (yMax - yMin));
  const cTx = PROJ_TEXT.reduce((a, p) => a + p[0], 0) / PROJ_TEXT.length;
  const cTy = PROJ_TEXT.reduce((a, p) => a + p[1], 0) / PROJ_TEXT.length;
  const cCx = PROJ_CHART.reduce((a, p) => a + p[0], 0) / PROJ_CHART.length;
  const cCy = PROJ_CHART.reduce((a, p) => a + p[1], 0) / PROJ_CHART.length;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="2D projection of the text and chart cones with the gap vector between their centroids" style={{ width: '100%', height: 'auto', display: 'block' }}>
      {PROJ_TEXT.map((p, i) => (
        <circle key={`t${i}`} cx={r2(fx(p[0]))} cy={r2(fy(p[1]))} r={3.2} fill={SECTOR_HUES[SECTOR[i]]} opacity={0.85} />
      ))}
      {PROJ_CHART.map((p, i) => (
        <rect key={`c${i}`} x={r2(fx(p[0]) - 3)} y={r2(fy(p[1]) - 3)} width={6} height={6} fill={SECTOR_HUES[SECTOR[i]]} opacity={0.85} rx={1} />
      ))}
      {/* the gap vector between the two cone centroids */}
      <line x1={r2(fx(cCx))} y1={r2(fy(cCy))} x2={r2(fx(cTx))} y2={r2(fy(cTy))} stroke={GAP_COLOR} strokeWidth={2.2} markerEnd="url(#cm-arrow)" />
      <defs>
        <marker id="cm-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={GAP_COLOR} />
        </marker>
      </defs>
      <circle cx={r2(fx(cTx))} cy={r2(fy(cTy))} r={4.5} fill="none" stroke={GAP_COLOR} strokeWidth={1.6} />
      <circle cx={r2(fx(cCx))} cy={r2(fy(cCy))} r={4.5} fill="none" stroke={GAP_COLOR} strokeWidth={1.6} />
      <g fontFamily="var(--font-sans)" fontSize={9}>
        <text x={r2(fx(cTx))} y={r2(fy(cTy)) - 7} textAnchor="middle" fill={MUTED}>text cone ●</text>
        <text x={r2(fx(cCx))} y={r2(fy(cCy)) - 7} textAnchor="middle" fill={MUTED}>chart cone ■</text>
        <text x={pw / 2} y={ph - 5} textAnchor="middle" fill={GAP_COLOR}>gap = ‖mean(text) − mean(chart)‖ (β = 0.5)</text>
      </g>
    </svg>
  );
}

// ===== Panel B — MIPS-flat vs cosine-moves under the gap-removing offset =============================
function RecallCurves({ idx, mipsLive }: { idx: number; mipsLive: number[] }) {
  const pw = 460, ph = 250, padL = 44, padR = 14, padT = 16, padB = 34;
  const n = ALPHA_GRID.length;
  const fx = (i: number) => padL + (pw - padL - padR) * (i / (n - 1));
  const fy = (r: number) => ph - padB - (ph - padT - padB) * r;
  const cos = ALPHA_GRID.map((_, i) => `${r2(fx(i))},${r2(fy(COSINE_RECALL[i]))}`).join(' ');
  const mips = ALPHA_GRID.map((_, i) => `${r2(fx(i))},${r2(fy(mipsLive[i]))}`).join(' ');
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="MIPS recall stays flat while cosine recall rises as the gap-removing offset grows" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={padL} y1={ph - padB} x2={pw - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={padL} y1={padT} x2={padL} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      {[0, 0.5, 1].map((r) => (
        <g key={r}>
          <line x1={padL} y1={r2(fy(r))} x2={pw - padR} y2={r2(fy(r))} stroke="var(--color-border)" strokeWidth={0.4} strokeDasharray="2 3" />
          <text x={padL - 5} y={r2(fy(r)) + 3} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">{r.toFixed(1)}</text>
        </g>
      ))}
      <text x={(padL + pw - padR) / 2} y={ph - 4} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">gap-removing offset α (multiples of g)</text>
      <text x={13} y={(padT + ph - padB) / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${(padT + ph - padB) / 2})`}>recall@1</text>
      <line x1={r2(fx(idx))} y1={padT} x2={r2(fx(idx))} y2={ph - padB} stroke={MUTED} strokeWidth={1} strokeDasharray="3 3" />
      <polyline points={cos} fill="none" stroke={CHART_COLOR} strokeWidth={2.2} />
      <polyline points={mips} fill="none" stroke={MIPS_COLOR} strokeWidth={2.4} />
      {ALPHA_GRID.map((_, i) => (
        <g key={i}>
          <circle cx={r2(fx(i))} cy={r2(fy(COSINE_RECALL[i]))} r={2.8} fill={CHART_COLOR} />
          <circle cx={r2(fx(i))} cy={r2(fy(mipsLive[i]))} r={2.8} fill={MIPS_COLOR} />
        </g>
      ))}
      <g fontFamily="var(--font-sans)" fontSize={9.5}>
        <rect x={pw - padR - 150} y={padT} width={11} height={11} fill={MIPS_COLOR} rx={1.5} />
        <text x={pw - padR - 135} y={padT + 9} fill={MUTED}>MIPS recall (flat — invariant)</text>
        <rect x={pw - padR - 150} y={padT + 15} width={11} height={11} fill={CHART_COLOR} rx={1.5} />
        <text x={pw - padR - 135} y={padT + 24} fill={MUTED}>cosine recall (moves)</text>
      </g>
    </svg>
  );
}

// ===== Panel C — the cone effect & temperature ladder ===============================================
function TemperatureCurve({ idx }: { idx: number }) {
  const pw = 430, ph = 250, padL = 44, padR = 14, padT = 16, padB = 36;
  const n = TAU_GRID.length;
  const fx = (i: number) => padL + (pw - padL - padR) * (i / (n - 1));
  const gMax = RESIDUAL_GAP[0].gap_before * 1.1;
  const fy = (v: number) => ph - padB - (ph - padT - padB) * (v / gMax);
  const after = TAU_GRID.map((_, i) => `${r2(fx(i))},${r2(fy(RESIDUAL_GAP[i].gap_after))}`).join(' ');
  const before = RESIDUAL_GAP[0].gap_before;
  return (
    <svg viewBox={`0 0 ${pw} ${ph}`} role="img" aria-label="residual modality gap after training rises as temperature falls" style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={padL} y1={ph - padB} x2={pw - padR} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      <line x1={padL} y1={padT} x2={padL} y2={ph - padB} stroke="var(--color-border)" strokeWidth={1} />
      {/* the initialization gap (before training) */}
      <line x1={padL} y1={r2(fy(before))} x2={pw - padR} y2={r2(fy(before))} stroke={MUTED} strokeWidth={1.2} strokeDasharray="4 3" />
      <text x={pw - padR} y={r2(fy(before)) - 4} textAnchor="end" fontSize={9} fill={MUTED} fontFamily="var(--font-sans)">gap at init {before.toFixed(2)}</text>
      <text x={(padL + pw - padR) / 2} y={ph - 5} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)">temperature τ (low → high)</text>
      <text x={13} y={(padT + ph - padB) / 2} textAnchor="middle" fontSize={10} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 13 ${(padT + ph - padB) / 2})`}>residual gap after training</text>
      <line x1={r2(fx(idx))} y1={padT} x2={r2(fx(idx))} y2={ph - padB} stroke={MUTED} strokeWidth={1} strokeDasharray="3 3" />
      <polyline points={after} fill="none" stroke={TEXT_COLOR} strokeWidth={2.4} />
      {TAU_GRID.map((t, i) => (
        <g key={i}>
          <circle cx={r2(fx(i))} cy={r2(fy(RESIDUAL_GAP[i].gap_after))} r={3} fill={TEXT_COLOR} />
          <text x={r2(fx(i))} y={ph - padB + 12} textAnchor="middle" fontSize={8.5} fill={i === idx ? 'var(--color-text)' : MUTED} fontFamily="var(--font-sans)">{t.toFixed(2)}</text>
        </g>
      ))}
    </svg>
  );
}

// ===== main component ==============================================================================
type Panel = 'decomposition' | 'ranking' | 'temperature';

export default memo(function CrossModalAlignmentLaboratory() {
  const [panel, setPanel] = useState<Panel>('ranking');
  const [betaIdx, setBetaIdx] = useState(5);     // beta = 0.5
  const [alphaIdx, setAlphaIdx] = useState(0);   // alpha = 0 (gap present)
  const [tauIdx, setTauIdx] = useState(0);       // tau = 0.04 (largest residual gap)
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    const tex =
      panel === 'decomposition'
        ? 'L_{\\text{align}} = \\lVert g\\rVert^2 + \\tfrac1n\\textstyle\\sum_i\\lVert d_i - g\\rVert^2 = \\text{gap}^2 + \\text{dispersion}'
        : panel === 'ranking'
          ? "S'_{ij} = \\langle t_i, c_j + \\alpha g\\rangle = \\langle t_i, c_j\\rangle + \\alpha\\langle t_i, g\\rangle \\;\\Rightarrow\\; \\text{argsort}_j \\text{ unchanged}"
          : '\\mathcal{L} = \\tfrac12\\big(\\mathcal{L}_{t\\to c} + \\mathcal{L}_{c\\to t}\\big), \\qquad \\tau\\downarrow \\;\\Rightarrow\\; \\text{larger residual gap}';
    katex.render(tex, formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  // Panel A live: the decomposition at the current beta.
  const dec = DECOMP[betaIdx];
  const coherentPct = dec.L_align > 0 ? (100 * dec.gap2) / dec.L_align : 0;

  // Panel B: MIPS recall@1 derived from the baked scores at every alpha (it does not move) — computed
  // once at module load (MIPS_RECALL_LIVE), since it depends only on the baked constants, not on state.
  const mipsLive = MIPS_RECALL_LIVE;
  const mipsNow = mipsLive[alphaIdx];
  const cosNow = COSINE_RECALL[alphaIdx];

  // Panel C live: the residual gap at the current temperature.
  const rg = RESIDUAL_GAP[tauIdx];

  return (
    <div style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button style={pill(panel === 'decomposition')} onClick={() => setPanel('decomposition')}>A · the decomposition</button>
        <button style={pill(panel === 'ranking')} onClick={() => setPanel('ranking')}>B · gap-invariant ranking</button>
        <button style={pill(panel === 'temperature')} onClick={() => setPanel('temperature')}>C · cone effect & temperature</button>
      </div>

      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />

      {panel === 'decomposition' && (
        <div>
          <Slider label="modality tilt β" value={betaIdx} min={0} max={DECOMP.length - 1} step={1} onChange={setBetaIdx} fmt={() => dec.beta.toFixed(1)} />
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '0.8rem', alignItems: 'center' }}>
            <DecompBars idx={betaIdx} />
            <ConeScatter />
          </div>
          <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="modality gap ‖g‖" value={dec.gap.toFixed(3)} accent />
            <Readout label="gap² (coherent)" value={dec.gap2.toFixed(3)} />
            <Readout label="dispersion (incoherent)" value={dec.dispersion.toFixed(3)} />
            <Readout label="L_align = gap² + dispersion" value={dec.L_align.toFixed(3)} />
            <Readout label="coherent share" value={`${coherentPct.toFixed(0)}%`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            The cross-modal alignment loss splits <em>orthogonally</em>: L_align = gap² + dispersion, a
            Frobenius-Pythagoras decomposition of the per-pair difference matrix into its coherent rank-1
            part (the gap, the shared offset every pair carries) and its incoherent complement. As the
            modality tilt β grows the gap² climbs to {coherentPct.toFixed(0)}% of the misalignment while the
            dispersion shrinks — contrastive training can drive L_align down by collapsing dispersion while a
            coherent gap survives. At β = {dec.beta.toFixed(1)} the cones sit ‖g‖ = {dec.gap.toFixed(3)} apart,
            and gap² ≤ L_align always (a projection is a contraction).
          </p>
        </div>
      )}

      {panel === 'ranking' && (
        <div>
          <Slider label="gap-removing offset α" value={alphaIdx} min={0} max={ALPHA_GRID.length - 1} step={1} onChange={setAlphaIdx} fmt={() => ALPHA_GRID[alphaIdx].toFixed(2)} />
          <RecallCurves idx={alphaIdx} mipsLive={mipsLive} />
          <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="MIPS recall@1 (live)" value={mipsNow.toFixed(4)} accent />
            <Readout label="cosine recall@1" value={cosNow.toFixed(4)} />
            <Readout label="offset α" value={`${ALPHA_GRID[alphaIdx].toFixed(2)} · g`} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            Offsetting every chart embedding by α·g changes each MIPS score by the <em>per-query</em> constant
            α⟨t_i, g⟩, so the argsort — and recall@1 — cannot move. The green line is recomputed live from the
            baked score matrix as you drag: it stays pinned at {mipsNow.toFixed(4)} for every α. The gap is
            invisible to ranking. Cosine, which renormalizes the offset keys, is <em>not</em> gap-invariant:
            its recall climbs to {Math.max(...COSINE_RECALL).toFixed(3)} as α → 1 removes the gap. The gap is a
            calibration artifact — it shifts absolute similarities (a threshold, a softmax), never the order.
          </p>
        </div>
      )}

      {panel === 'temperature' && (
        <div>
          <Slider label="temperature τ" value={tauIdx} min={0} max={TAU_GRID.length - 1} step={1} onChange={setTauIdx} fmt={() => TAU_GRID[tauIdx].toFixed(2)} />
          <TemperatureCurve idx={tauIdx} />
          <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
            <Readout label="residual gap after training" value={rg.gap_after.toFixed(3)} accent />
            <Readout label="gap at initialization" value={rg.gap_before.toFixed(3)} />
            <Readout label="alignment loss" value={rg.alignment.toFixed(3)} />
            <Readout label="union uniformity" value={rg.uniformity.toFixed(3)} />
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
            A deterministic full-batch descent on the symmetric CLIP loss starts from a gap of
            {' '}{rg.gap_before.toFixed(2)} at initialization. At moderate-to-high temperature (τ ≥ 0.2) the
            descent <em>closes</em> the gap to zero; at τ = {rg.tau.toFixed(2)} it leaves a residual gap of
            {' '}{rg.gap_after.toFixed(3)}. Lower temperature preserves a <em>larger</em> residual gap — the
            cone effect. (This is a measured surrogate, not SGD: we assert only the monotone direction, never
            that training preserves the gap.)
          </p>
        </div>
      )}
    </div>
  );
});
