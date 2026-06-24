import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Context-Selection Laboratory — four panels for the `context-selection-submodular-dpp` topic:
 *   A. Diminishing returns. Greedy facility-location coverage f(S_k) rises and FLATTENS; each pick's marginal
 *      gain shrinks — the discrete signature of submodularity, the saturation PMI measured in bits. An inset
 *      contrasts the submodular backbone (facility witness ≈ 0) with info gain (witness < 0: NOT submodular).
 *   B. The greedy guarantee. On a constructed instance greedy is STRICTLY below OPT (a visible gap) yet never
 *      below the (1−1/e)·OPT floor — Nemhauser–Wolsey–Fisher. (On the smooth finance pool greedy reaches OPT.)
 *   C. Coverage vs diversity vs relevance (the geometry). The candidate pool in 2D, colored by facet; toggle
 *      the selection method and watch top-k cluster on the redundant sector-generics while facility/DPP reach
 *      the lone DISAMBIGUATOR and spread. det(L_S) = (∏ qᵢ²)·det(S_S): quality × squared volume.
 *   D. The payoff. Answer quality Q(k) (mass on the true company, via the IMPORTED answer_posterior_topk) per
 *      method at a fixed budget: diversity-aware selection beats top-k because top-k burns the budget on
 *      near-duplicates while coverage/DPP buy the disambiguating evidence.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every coverage / NWF / Q / selection number below is mirrored TO THE
 * DECIMAL from notebooks/context-selection-submodular-dpp/context_selection_submodular_dpp.py (viz_constants()).
 * Matching asserts: test_marginal_gains_diminish / test_nwf_gap_visible / test_dpp_factorization /
 * test_info_gain_not_submodular / test_diversity_beats_topk_on_coverage / test_payoff_winner_pinned. The lab
 * recomputes ONLY closed forms in TS (the marginal-gain differences of baked coverage, the (1−1/e) floor, pixel
 * maps); coverage values, greedy/OPT, selections, 2D points, and Q(k) are MODEL OUTPUTS and are baked.
 */

// --- baked from viz_constants() -------------------------------------------------------
const POOL_SIZE = 9;
const N_GENERIC = 5;                 // sector-generic near-duplicates (the redundant cluster)
const N_DISTRACT = 3;
const N_QUERIES = 16;
const K_SELECT = 3;                  // the selection budget for the payoff bars
const MMR_LAM = 0.5;
const ONE_MINUS_INV_E = 0.6321;      // the NWF floor

// Panel A — greedy facility-location coverage on the worked pool (rising, concave). Δ recomputed in TS.
const COVERAGE = [4.0343, 4.2642, 4.3644, 4.4541, 4.543, 4.6289, 4.6991, 4.7557, 4.7557];
const FACILITY_WITNESS = -0.0;       // ≥ 0: facility location IS submodular
const INFOGAIN_WITNESS = -0.7105;    // < 0: info gain is NOT submodular on this corpus
const XOR_DELTA_EMPTY = 0.0;         // synergistic pair: gain of D2 alone
const XOR_DELTA_GIVEN_D1 = 1.0;      // ... gain of D2 after D1 — INCREASES (supermodular)

// Panel B — the constructed worst-case instance (greedy < OPT at k = 2, both above the floor)
const NWF_KS = [1, 2, 3];
const NWF_GREEDY = [6, 9, 12];
const NWF_OPT = [6, 12, 12];
const NWF_FLOOR = [3.7927, 7.5854, 7.5854];
const NWF_RATIO = [1.0, 0.75, 1.0];

// Panel C — the worked pool in 2D, facets, relevance, and each method's selected subset
const POINTS_2D = [[0.3804, 0.1108], [0.4449, -0.0771], [0.4045, 0.012], [0.4057, 0.0404], [0.4768, 0.0579], [0.071, 0.2121], [-0.7994, 0.0567], [-0.4838, -0.9556], [-0.9001, 0.5429]];
const FACET = [0, 0, 0, 0, 0, 1, 3, 3, 3];     // 0 generic, 1 A-specific (disambiguator), 3 distractor
const REL = [0.8161, 0.8299, 0.8071, 0.7406, 0.8048, 0.5907, 0.0644, 0.1021, -0.1547];
const SEL_TOPK = [1, 0, 2];
const SEL_MMR = [1, 6, 7];
const SEL_FACILITY = [1, 5, 0];
const SEL_DPP = [1, 5, 7];
const A_SPECIFIC = N_GENERIC;                   // index 5: the disambiguator
const DPP_DET_LS = 2.5955, DPP_QUALITY_SQ = 4.3544, DPP_VOLUME_SQ = 0.5961;

// Panel D — answer quality Q(k) per method (mass on the true company) + the headline bars
const PAYOFF_KS = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const QK_TOPK = [0.3757, 0.3674, 0.3778, 0.3828, 0.3749, 0.4453, 0.4428, 0.4377, 0.4229];
const QK_MMR = [0.3757, 0.4571, 0.4445, 0.4653, 0.4429, 0.4376, 0.4358, 0.4338, 0.4229];
const QK_FACILITY = [0.3525, 0.5758, 0.5378, 0.494, 0.4726, 0.4495, 0.4406, 0.4203, 0.4229];
const QK_DPP = [0.3757, 0.5839, 0.5448, 0.4954, 0.4706, 0.4448, 0.4369, 0.4315, 0.4229];
const PAYOFF = {
  topk: { Q: 0.3778, cover: 0.8994 },
  mmr: { Q: 0.4445, cover: 0.9035 },
  facility: { Q: 0.5378, cover: 0.9287 },
  dpp: { Q: 0.5448, cover: 0.9209 },
};

// --- palette --------------------------------------------------------------------------
const ACCENT = 'var(--color-accent)';
const TOPK = '#c25b6b';              // the naive relevance baseline
const MMRC = '#d9a23b';             // MMR (heuristic)
const FACL = '#5fa873';             // facility-location coverage (provable)
const DPPC = '#6c8cd5';            // DPP diversity (provable)
const GOLD = '#d9a23b';            // the disambiguator facet
const MUTED = '#9aa3ad';
const METHOD_COLOR: Record<Method, string> = { topk: TOPK, mmr: MMRC, facility: FACL, dpp: DPPC };
const METHOD_LABEL: Record<Method, string> = { topk: 'top-k', mmr: 'MMR', facility: 'facility-loc.', dpp: 'DPP' };
const SELS: Record<Method, number[]> = { topk: SEL_TOPK, mmr: SEL_MMR, facility: SEL_FACILITY, dpp: SEL_DPP };

const fmt = (x: number, n = 3) => x.toFixed(n);
const r2 = (x: number) => Math.round(x * 100) / 100;

// closed-form TS recomputation (the ONLY things not baked): marginal gains = differences of baked coverage.
const covGain = (i: number) => (i === 0 ? COVERAGE[0] : COVERAGE[i] - COVERAGE[i - 1]);

const poly = (vals: number[], px: (i: number) => number, py: (v: number) => number) =>
  vals.map((v, i) => `${i ? 'L' : 'M'}${r2(px(i))},${r2(py(v))}`).join(' ');

const facetFill = (f: number) => (f === 1 ? GOLD : f === 0 ? MUTED : 'var(--color-border)');
const facetName = (f: number) => (f === 1 ? 'disambiguator' : f === 0 ? 'generic' : 'distractor');

// --- shared UI atoms ------------------------------------------------------------------
function Readout({ label, value, accent, color }: { label: string; value: string; accent?: boolean; color?: string }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: color ?? (accent ? 'var(--color-accent)' : 'var(--color-text)') }}>{value}</div>
    </div>
  );
}
function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display: string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '15rem' }}>{label} = <strong>{display}</strong></span>
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
const miniPill = (active: boolean, color: string) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.72rem', padding: '0.2rem 0.55rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? color : 'var(--color-border)'}`,
  background: active ? color : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

// ===== Panel A — diminishing returns / submodular coverage ==========================================
function CoveragePanel({ k, setK }: { k: number; setK: (v: number) => void }) {
  const W = 520, H = 232, padL = 40, padR = 16, padT = 20, padB = 34;
  const n = COVERAGE.length;
  const fMax = COVERAGE[n - 1];
  const px = (i: number) => padL + (W - padL - padR) * (i / (n - 1));
  const py = (v: number) => H - padB - (H - padT - padB) * (v / fMax);
  // marginal-gain bars use their own scale (the first pick dwarfs the rest), so scale to the 2nd gain
  const gMax = covGain(1);
  const pg = (g: number) => H - padB - (H - padT - padB) * Math.min(g / gMax, 1);
  const i = k - 1;
  const barW = (W - padL - padR) / n * 0.42;
  return (
    <div>
      <Slider label="passages selected k (greedy coverage)" value={k} min={1} max={n} step={1}
        onChange={(v) => setK(Math.round(v))} display={`${k}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="greedy coverage rising and flattening, marginal gains diminishing" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={padL - 5} y={py(f * fMax) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{fmt(f * fMax, 1)}</text>
        ))}
        {/* marginal-gain bars (k ≥ 2): the diminishing tail. The first pick is annotated, not drawn to scale. */}
        {COVERAGE.map((_v, j) => (
          <g key={j}>
            {j >= 1 && (
              <rect x={px(j) - barW / 2} y={pg(covGain(j))} width={barW} height={(H - padB) - pg(covGain(j))}
                fill={FACL} fillOpacity={0.28} />
            )}
          </g>
        ))}
        {/* the concave coverage curve f(S_k) */}
        <path d={poly(COVERAGE, px, py)} fill="none" stroke={ACCENT} strokeWidth={2.6} />
        <path d={`${poly(COVERAGE, px, py)} L${r2(px(n - 1))},${r2(py(0))} L${r2(px(0))},${r2(py(0))} Z`} fill={ACCENT} fillOpacity={0.06} />
        {COVERAGE.map((v, j) => (<circle key={j} cx={r2(px(j))} cy={r2(py(v))} r={2.1} fill={ACCENT} />))}
        {/* first-pick annotation */}
        <text x={px(0) + 5} y={py(COVERAGE[0]) - 5} fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">first pick covers {fmt(COVERAGE[0], 1)}</text>
        {/* slider marker */}
        <circle cx={px(i)} cy={py(COVERAGE[i])} r={3.6} fill={ACCENT} />
        <line x1={px(i)} y1={py(COVERAGE[i])} x2={px(i)} y2={H - padB} stroke="var(--color-text)" strokeWidth={0.7} strokeDasharray="2 3" />
        {COVERAGE.map((_v, j) => (
          <text key={j} x={px(j)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{j + 1}</text>
        ))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">passages selected k</text>
        <text x={padL + 2} y={padT + 2} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={ACCENT}>▬ coverage f(Sₖ)</tspan> · <tspan fill={FACL}>▮ marginal gain Δₖ (diminishing)</tspan>
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="coverage f(Sₖ)" value={fmt(COVERAGE[i], 3)} accent />
        <Readout label="marginal gain of this pick" value={`+${fmt(covGain(i), 3)}`} color={FACL} />
        <Readout label="facility witness (submodular?)" value={`${fmt(FACILITY_WITNESS, 2)} ✓`} color={FACL} />
        <Readout label="info-gain witness" value={`${fmt(INFOGAIN_WITNESS, 2)} ✗`} color={TOPK} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Greedy coverage <span style={{ color: ACCENT }}>f(Sₖ)</span> rises and <strong>flattens</strong>: each pick's{' '}
        <span style={{ color: FACL }}>marginal gain Δₖ</span> is no larger than the one before — the diminishing returns that
        define a <strong>submodular</strong> set function, the same saturation PMI measured in bits. Facility-location coverage
        is submodular (witness ≈ {fmt(FACILITY_WITNESS, 2)}); the answer-information gain we'd <em>like</em> to optimize is{' '}
        <strong>not</strong> (witness {fmt(INFOGAIN_WITNESS, 2)} &lt; 0) — synergistic evidence breaks it — so coverage is the
        clean backbone that earns a theorem. (The cleanest counterexample: a synergistic pair where each
        observation alone is worth {fmt(XOR_DELTA_EMPTY, 0)} bits yet together they are worth{' '}
        {fmt(XOR_DELTA_GIVEN_D1, 0)} bit — the marginal gain <em>increases</em> with conditioning, the opposite of
        diminishing returns.)
      </p>
    </div>
  );
}

// ===== Panel B — the 1−1/e guarantee ================================================================
function GuaranteePanel({ k, setK }: { k: number; setK: (v: number) => void }) {
  const W = 520, H = 224, padL = 40, padR = 16, padT = 22, padB = 36;
  const n = NWF_KS.length;
  const vMax = Math.max(...NWF_OPT) * 1.08;
  const slot = (W - padL - padR) / n;
  const cx = (j: number) => padL + slot * (j + 0.5);
  const py = (v: number) => H - padB - (H - padT - padB) * (v / vMax);
  const bw = slot * 0.26;
  const j = NWF_KS.indexOf(k) >= 0 ? NWF_KS.indexOf(k) : 0;
  return (
    <div>
      <Slider label="budget k (constructed worst-case instance)" value={k} min={NWF_KS[0]} max={NWF_KS[n - 1]} step={1}
        onChange={(v) => setK(Math.round(v))} display={`${k}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="greedy versus optimum versus the one-minus-one-over-e floor" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={padL - 5} y={py(f * vMax) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{Math.round(f * vMax)}</text>
        ))}
        {NWF_KS.map((kk, idx) => (
          <g key={kk}>
            {/* OPT bar */}
            <rect x={cx(idx) - bw - 2} y={py(NWF_OPT[idx])} width={bw} height={(H - padB) - py(NWF_OPT[idx])} fill={FACL} fillOpacity={0.85} />
            {/* greedy bar */}
            <rect x={cx(idx) + 2} y={py(NWF_GREEDY[idx])} width={bw} height={(H - padB) - py(NWF_GREEDY[idx])} fill={ACCENT} fillOpacity={0.85} />
            {/* (1−1/e)·OPT floor tick */}
            <line x1={cx(idx) - bw - 6} y1={py(NWF_FLOOR[idx])} x2={cx(idx) + bw + 6} y2={py(NWF_FLOOR[idx])} stroke={DPPC} strokeWidth={1.6} strokeDasharray="4 3" />
            <text x={cx(idx)} y={H - padB + 12} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">k = {kk}</text>
            {NWF_RATIO[idx] < 0.999 && (
              <text x={cx(idx)} y={py(NWF_OPT[idx]) - 4} textAnchor="middle" fontSize={8} fill={TOPK} fontFamily="var(--font-sans)">gap! {fmt(NWF_RATIO[idx], 2)}</text>
            )}
          </g>
        ))}
        {/* highlight the selected k slot */}
        <rect x={cx(j) - slot / 2 + 2} y={padT} width={slot - 4} height={H - padT - padB} fill="var(--color-text)" fillOpacity={0.04} />
        <text x={padL + 2} y={padT - 6} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={FACL}>▮ OPT</tspan> · <tspan fill={ACCENT}>▮ greedy</tspan> · <tspan fill={DPPC}>┄ (1−1/e)·OPT floor</tspan>  (facets covered)
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label={`greedy / OPT at k = ${k}`} value={`${NWF_GREEDY[j]} / ${NWF_OPT[j]}`} accent />
        <Readout label="realized ratio" value={fmt(NWF_RATIO[j], 3)} color={NWF_RATIO[j] < 0.999 ? TOPK : FACL} />
        <Readout label="guaranteed floor" value={`${fmt(ONE_MINUS_INV_E, 3)} × OPT`} color={DPPC} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Greedy maximization of a monotone submodular objective is guaranteed to return at least{' '}
        <span style={{ color: DPPC }}>(1−1/e) ≈ 0.632</span> of the optimum (Nemhauser–Wolsey–Fisher). At{' '}
        <strong>k = 2</strong> here greedy grabs the biggest set first and falls to <span style={{ color: ACCENT }}>9</span> of
        the optimal <span style={{ color: FACL }}>12</span> — a real gap, but comfortably above the floor. The bound is a{' '}
        worst-case promise; on the smooth finance pool greedy actually <em>reaches</em> OPT.
      </p>
    </div>
  );
}

// ===== Panel C — coverage vs diversity vs relevance (the geometry) ==================================
function GeometryPanel() {
  const [method, setMethod] = useState<Method>('topk');
  const W = 520, H = 250, padL = 16, padR = 16, padT = 28, padB = 22;
  const xs = POINTS_2D.map((p) => p[0]), ys = POINTS_2D.map((p) => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const px = (x: number) => padL + (W - padL - padR) * ((x - xmin) / (xmax - xmin || 1)) * 0.92 + 0.04 * (W - padL - padR);
  const py = (y: number) => H - padB - (H - padT - padB) * ((y - ymin) / (ymax - ymin || 1)) * 0.9 - 0.05 * (H - padT - padB);
  const sel = SELS[method];
  const selSet = new Set(sel);
  const hasDisambiguator = selSet.has(A_SPECIFIC);
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', margin: '0.1rem 0 0.5rem' }}>
        {(['topk', 'mmr', 'facility', 'dpp'] as Method[]).map((m) => (
          <button key={m} type="button" style={miniPill(method === m, METHOD_COLOR[m])} onClick={() => setMethod(m)}>
            {METHOD_LABEL[m]}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="candidate pool in 2D with the selected subset highlighted per method" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {/* the redundant generic cluster, annotated */}
        <text x={px(0.42)} y={padT - 8} textAnchor="middle" fontSize={8} fill={MUTED} fontFamily="var(--font-sans)">redundant generic cluster →</text>
        {/* candidate points — radius scales with query relevance (top-k chases the biggest points) */}
        {POINTS_2D.map((p, i) => {
          const chosen = selSet.has(i);
          const f = FACET[i] ?? 3;
          const rel = REL[i] ?? 0;
          const baseR = 4 + 5 * Math.max(0, Math.min(1, rel));
          return (
            <g key={i}>
              <circle cx={px(p[0])} cy={py(p[1])} r={chosen ? baseR + 2.5 : baseR}
                fill={chosen ? METHOD_COLOR[method] : facetFill(f)}
                fillOpacity={chosen ? 0.9 : 0.4}
                stroke={chosen ? METHOD_COLOR[method] : facetFill(f)} strokeWidth={chosen ? 2 : 1} />
              {f === 1 && (
                <text x={px(p[0])} y={py(p[1]) - 11} textAnchor="middle" fontSize={8} fill={GOLD} fontFamily="var(--font-sans)" fontWeight={600}>disambiguator</text>
              )}
              {chosen && (
                <text x={px(p[0])} y={py(p[1]) + 3} textAnchor="middle" fontSize={7.5} fill="var(--color-bg)" fontFamily="var(--font-sans)" fontWeight={700}>{i}</text>
              )}
            </g>
          );
        })}
        <text x={padL + 2} y={padT - 14} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          selecting with <tspan fill={METHOD_COLOR[method]} fontWeight={700}>{METHOD_LABEL[method]}</tspan> — filled = chosen
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.3rem', marginTop: '0.3rem', flexWrap: 'wrap' }}>
        <Readout label={`${METHOD_LABEL[method]} picks (facets)`} value={sel.map((i) => facetName(FACET[i] ?? 3)[0].toUpperCase()).join(' ')} color={METHOD_COLOR[method]} />
        <Readout label="reaches the disambiguator?" value={hasDisambiguator ? 'yes ✓' : 'no ✗'} color={hasDisambiguator ? FACL : TOPK} />
        <Readout label="det(L_S) = quality² × volume²" value={`${fmt(DPP_DET_LS, 2)} = ${fmt(DPP_QUALITY_SQ, 2)}×${fmt(DPP_VOLUME_SQ, 2)}`} color={DPPC} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The candidate pool, projected to 2D: a tight cluster of <span style={{ color: MUTED }}>sector-generic near-duplicates</span>{' '}
        (top-right), the lone <span style={{ color: GOLD }}>disambiguator</span> near the gold company, and scattered{' '}
        distractors. <span style={{ color: TOPK }}>Top-k</span> spends all three picks inside the redundant cluster;{' '}
        <span style={{ color: FACL }}>facility-location</span> and <span style={{ color: DPPC }}>DPP</span> reach the
        disambiguator. A DPP's probability det(L_S) factors into <strong>quality² × volume²</strong> — near-collinear
        duplicates span a near-flat parallelepiped (volume → 0) and are repelled.
      </p>
    </div>
  );
}

// ===== Panel D — the answer-quality payoff ==========================================================
function PayoffPanel({ k, setK }: { k: number; setK: (v: number) => void }) {
  const W = 520, H = 232, padL = 40, padR = 92, padT = 20, padB = 34;
  const n = PAYOFF_KS.length;
  const px = (i: number) => padL + (W - padL - padR) * (i / (n - 1));
  const qmin = 0.32, qmax = 0.62;
  const py = (q: number) => H - padB - (H - padT - padB) * ((q - qmin) / (qmax - qmin));
  const i = k - 1;
  const curves: { m: Method; vals: number[] }[] = [
    { m: 'topk', vals: QK_TOPK }, { m: 'mmr', vals: QK_MMR }, { m: 'facility', vals: QK_FACILITY }, { m: 'dpp', vals: QK_DPP },
  ];
  const gap = QK_DPP[i] - QK_TOPK[i];
  // headline bars at the current k, to the right
  const bx0 = W - padR + 12, bw = 14;
  const order: Method[] = ['topk', 'mmr', 'facility', 'dpp'];
  return (
    <div>
      <Slider label="selection budget k" value={k} min={1} max={n} step={1}
        onChange={(v) => setK(Math.round(v))} display={`${k} of ${POOL_SIZE}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="answer quality per selection method versus budget" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0.35, 0.45, 0.55].map((v) => (<text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
        {curves.map(({ m, vals }) => (
          <g key={m}>
            <path d={poly(vals, px, py)} fill="none" stroke={METHOD_COLOR[m]} strokeWidth={m === 'topk' ? 2.4 : 2} strokeDasharray={m === 'mmr' ? '4 3' : undefined} />
          </g>
        ))}
        {/* slider marker line */}
        <line x1={px(i)} y1={padT} x2={px(i)} y2={H - padB} stroke="var(--color-text)" strokeWidth={0.8} strokeDasharray="2 3" />
        {curves.map(({ m, vals }) => (<circle key={m} cx={px(i)} cy={py(vals[i])} r={3} fill={METHOD_COLOR[m]} />))}
        {/* headline bars at current k */}
        {order.map((m, bi) => {
          const v = (m === 'topk' ? QK_TOPK : m === 'mmr' ? QK_MMR : m === 'facility' ? QK_FACILITY : QK_DPP)[i];
          const x = bx0 + bi * (bw + 4);
          return (
            <g key={m}>
              <rect x={x} y={py(v)} width={bw} height={(H - padB) - py(v)} fill={METHOD_COLOR[m]} fillOpacity={0.85} />
            </g>
          );
        })}
        <text x={bx0 + (order.length * (bw + 4)) / 2} y={padT - 6} textAnchor="middle" fontSize={7.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">Q at k = {k}</text>
        {PAYOFF_KS.filter((_v, j) => j % 2 === 0).map((kk) => (
          <text key={kk} x={px(kk - 1)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{kk}</text>
        ))}
        <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">selection budget k</text>
        <text x={padL + 2} y={padT + 2} fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          <tspan fill={DPPC}>▬ DPP</tspan> · <tspan fill={FACL}>▬ facility</tspan> · <tspan fill={MMRC}>┄ MMR</tspan> · <tspan fill={TOPK}>▬ top-k</tspan>  — Q = mass on truth
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.2rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="DPP / facility Q(k)" value={`${fmt(QK_DPP[i])} / ${fmt(QK_FACILITY[i])}`} color={DPPC} />
        <Readout label="top-k / MMR Q(k)" value={`${fmt(QK_TOPK[i])} / ${fmt(QK_MMR[i])}`} color={TOPK} />
        <Readout label="diversity advantage (DPP − top-k)" value={`${gap >= 0 ? '+' : ''}${fmt(gap)}`} color={gap > 0 ? FACL : TOPK} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        At a fixed budget, <span style={{ color: DPPC }}>DPP</span> and <span style={{ color: FACL }}>facility-location</span>{' '}
        put markedly more mass on the true company than <span style={{ color: TOPK }}>top-k</span> — biggest at small k
        (k = {K_SELECT}: {fmt(PAYOFF.dpp.Q, 2)} vs {fmt(PAYOFF.topk.Q, 2)}), the realistic budget regime. Top-k burns the
        budget on redundant generics; coverage and diversity buy the disambiguating evidence. <span style={{ color: MMRC }}>MMR</span>{' '}
        helps a little, but it is a heuristic with no guarantee. The gap closes only once k is large enough to read nearly
        everything — by which point you have paid the long-context cost the previous topic warned about.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'coverage' | 'guarantee' | 'geometry' | 'payoff';
type Method = 'topk' | 'mmr' | 'facility' | 'dpp';
const TEX: Record<Panel, string> = {
  coverage: 'f(S) = \\sum_{i} w_i \\max_{j\\in S}\\operatorname{sim}(i,j), \\qquad \\Delta(e\\mid A)\\ \\ge\\ \\Delta(e\\mid B)\\ \\ (A\\subseteq B)',
  guarantee: 'f(S_{\\text{greedy}})\\ \\ge\\ \\left(1-\\tfrac{1}{e}\\right) f(S^\\star)\\ \\approx\\ 0.632\\, f(S^\\star)',
  geometry: '\\mathcal{P}(S)\\ \\propto\\ \\det(L_S) = \\Big(\\textstyle\\prod_{i\\in S} q_i^2\\Big)\\det(S_S)\\ =\\ \\text{quality}^2 \\times \\text{volume}^2',
  payoff: 'Q_{\\text{method}}(k) = \\mathbb{E}_q\\big[\\,p(a^\\star \\mid q,\\ S^{\\text{method}}_k)\\,\\big]',
};

export default memo(function ContextSelectionLaboratory() {
  const [panel, setPanel] = useState<Panel>('coverage');
  const [k, setK] = useState(3);
  const [nwfK, setNwfK] = useState(2);
  const [payK, setPayK] = useState(3);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="context-selection-submodular-dpp" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'coverage')} onClick={() => setPanel('coverage')}>A · diminishing returns</button>
        <button type="button" style={pill(panel === 'guarantee')} onClick={() => setPanel('guarantee')}>B · 1−1/e guarantee</button>
        <button type="button" style={pill(panel === 'geometry')} onClick={() => setPanel('geometry')}>C · coverage vs diversity</button>
        <button type="button" style={pill(panel === 'payoff')} onClick={() => setPanel('payoff')}>D · the payoff</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'coverage' && <CoveragePanel k={k} setK={setK} />}
      {panel === 'guarantee' && <GuaranteePanel k={nwfK} setK={setNwfK} />}
      {panel === 'geometry' && <GeometryPanel />}
      {panel === 'payoff' && <PayoffPanel k={payK} setK={setPayK} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        Finance vMF corpus reused from the dense-retrieval / long-context geometry: {N_QUERIES} query pools of {POOL_SIZE}{' '}
        candidates each — {N_GENERIC} sector-generic near-duplicates (ambiguous between the gold company and its confusable
        peer, the <em>highest</em> relevance), one lower-relevance disambiguator near the gold prototype, and {N_DISTRACT}{' '}
        distractors. The answer is read through the imported <code>answer_posterior_topk</code> (a synthetic softmax stand-in,
        not an LLM). MMR uses λ = {MMR_LAM}. Numbers mirror <code>context_selection_submodular_dpp.py</code>; the lab recomputes
        only the marginal-gain differences and the (1−1/e) floor.
      </p>
    </div>
  );
});
