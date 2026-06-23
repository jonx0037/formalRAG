import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * PMI Retrieval-Value Laboratory — four panels for the `pmi-retrieval-value` topic:
 *   A. Prior -> posterior as a document is retrieved. A slider over the worked query's candidate filings
 *      reshapes the answer distribution; the entropy meter shows H(A|Q) -> H(A|Q,d) and pmi(a*;d) flips
 *      sign between the gold filing (positive bits) and a same-sector distractor (negative bits).
 *   B. pmi at the true answer, relevant vs distractor. Two histograms of pmi(a*;d|q): relevant filings
 *      pile up on positive bits, distractors on negative — a distractor COSTS bits at the true answer.
 *   C. Diminishing returns. Belief movement (KL, bits) of a first filing, then a redundant second copy
 *      (adds almost nothing) vs a different second filing (adds a lot) — the chain rule made visible.
 *   D. The InfoNCE ceiling + bits != recall. Left: the lower bound log2(m) - L/ln2 rising toward its
 *      log2(m) ceiling as in-batch negatives grow (it saturates). Right: recall is pinned at 1.0 for
 *      every query while the bits added vary widely — presence is not contribution.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): PRIOR, POST_ALL, H_A_GIVEN_Q/QD, I_ADQ, PMI_REL, PMI_DISTR,
 * SAT, INFONCE, SCATTER_BITS are mirrored TO THE DECIMAL from notebooks/pmi-retrieval-value/
 * pmi_retrieval_value.py (viz_constants()). Matching asserts: test_pmi_sign_separation /
 * test_three_way_mi_agreement / test_saturation_diminishing_returns / test_infonce_bound_saturates /
 * test_bits_vs_recall_differ. The lab recomputes ONLY closed forms in TS (Shannon entropy and pmi from
 * the baked probabilities, histogram bin counts, the log2(m) ceiling, and SVG geometry); every softmax /
 * KL / InfoNCE value is baked. Change a number here -> change it in the .py, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const K = 8;                       // companies = answers = documents
const N_QUERIES = 32;
const WORKED_Q = 5;
const A_STAR = 1;                  // the worked query's gold company
const DISTRACTOR = 0;              // its same-sector distractor
const Q_SECTOR = 0;
const SECTOR = [0, 0, 1, 1, 2, 2, 3, 3];   // sector of each company/document

const PRIOR = [0.2272, 0.459, 0.01, 0.0134, 0.0293, 0.0352, 0.173, 0.0529];
// p(a|q,d), one row per candidate document d (companies 0..7 across).
const POST_ALL: number[][] = [
  [0.8082, 0.1667, 0.0003, 0.0002, 0.0011, 0.0015, 0.0133, 0.0087],
  [0.0592, 0.9149, 0.0006, 0.0006, 0.0011, 0.0006, 0.0165, 0.0064],
  [0.0448, 0.2319, 0.5022, 0.1132, 0.0076, 0.0037, 0.0807, 0.0159],
  [0.0222, 0.1668, 0.0756, 0.5343, 0.0058, 0.0025, 0.1699, 0.023],
  [0.0407, 0.1231, 0.0021, 0.0024, 0.6918, 0.1108, 0.0219, 0.0073],
  [0.049, 0.0601, 0.0009, 0.0009, 0.0968, 0.768, 0.0181, 0.0062],
  [0.0249, 0.0865, 0.0011, 0.0034, 0.0011, 0.001, 0.7826, 0.0996],
  [0.0592, 0.123, 0.0008, 0.0017, 0.0013, 0.0012, 0.3636, 0.4493],
];
const H_A_GIVEN_Q = 2.0592;        // averaged over queries (Panel A readout)
const H_A_GIVEN_QD = 0.876;
const I_ADQ = 1.18317;             // three-way-verified conditional mutual information (bits)

// Panel B — pmi at the true answer for the gold filing vs the same-sector distractor, per query.
const PMI_REL = [1.152, 0.811, 1.287, 1.311, 1.474, 0.995, 1.04, 1.124, 1.198, 0.423, 1.172, 0.671, 0.485, 0.722, 1.293, 0.763, 0.59, 1.064, 0.77, 1.199, 1.282, 1.039, 1.208, 0.688, 0.585, 0.876, 0.828, 0.931, 1.044, 1.033, 1.342, 1.084];
const PMI_DISTR = [-1.792, -1.709, -1.395, -1.578, -0.896, -1.461, -1.959, -1.803, -1.111, -0.744, -0.988, -0.96, -1.264, -1.161, -0.899, -1.2, -1.57, -1.683, -1.734, -0.757, -1.372, -1.769, -1.013, -1.396, -0.541, -0.501, -0.519, -0.519, -0.435, 0.006, -0.292, -0.5];
const MEAN_PMI_REL = 0.9839;
const MEAN_PMI_DISTR = -1.1099;
const FRAC_DISTR_NEG = 0.969;

// Panel C — belief movement (KL, bits): first filing, then a redundant vs a novel second filing.
const SAT = { standalone: 0.6734, redundant: 0.0927, novel: 0.4438 };

// Panel D — the InfoNCE bound vs its ceiling as in-batch candidates m grow; and per-query (recall, bits).
const INFONCE: { m: number; ceiling: number; bound: number }[] = [
  { m: 2, ceiling: 1.0, bound: 0.2168 },
  { m: 3, ceiling: 1.585, bound: 0.9159 },
  { m: 4, ceiling: 2.0, bound: 1.1244 },
  { m: 5, ceiling: 2.3219, bound: 1.4435 },
  { m: 6, ceiling: 2.585, bound: 1.5508 },
  { m: 7, ceiling: 2.8074, bound: 1.7835 },
  { m: 8, ceiling: 3.0, bound: 1.8381 },
];
const SCATTER_BITS = [1.3109, 1.0969, 1.4059, 1.1749, 1.5222, 1.2358, 0.9827, 1.2393, 1.4497, 0.9181, 1.4445, 1.1712, 0.6779, 1.0612, 1.604, 0.9894, 0.8223, 1.2219, 0.8859, 1.5189, 1.5247, 1.0794, 1.5605, 1.0926, 0.8513, 1.0642, 0.9442, 1.1188, 1.1067, 1.3848, 1.3335, 1.067];
const BITS_MIN = 0.6779;
const BITS_MAX = 1.604;

const ACCENT = 'var(--color-accent)';
const POS_COLOR = '#5fa873';       // a relevant doc — adds bits at the true answer
const NEG_COLOR = '#c25b6b';       // a distractor — costs bits at the true answer
const PRIOR_COLOR = '#9aa3ad';     // the prior (muted)
const MUTED = '#9aa3ad';

const fmt = (x: number, n = 3) => x.toFixed(n);
const r2 = (x: number) => Math.round(x * 100) / 100;

// --- closed-form TS recomputation -----------------------------------------------------
const entropyBits = (p: number[]) =>
  -p.reduce((s, x) => s + (x > 0 ? x * Math.log2(x) : 0), 0);
const pmiBits = (post: number[], prior: number[], a: number) => Math.log2(post[a] / prior[a]);
// histogram bin counts over [lo, hi] in `bins` bins (closed form).
const histogram = (xs: number[], lo: number, hi: number, bins: number) => {
  const counts = new Array(bins).fill(0);
  const w = (hi - lo) / bins;
  for (const x of xs) {
    let b = Math.floor((x - lo) / w);
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    counts[b] += 1;
  }
  return counts;
};

const docLabel = (d: number) =>
  `doc ${d} · company ${d} (sector ${SECTOR[d]})${d === A_STAR ? ' — gold' : d === DISTRACTOR ? ' — distractor' : ''}`;

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

// ===== Panel A — prior -> posterior as a document is retrieved ======================================
function AnswerModelPanel({ d, setD }: { d: number; setD: (v: number) => void }) {
  const post = POST_ALL[d];
  const W = 520, H = 196, padL = 30, padR = 12, padT = 18, padB = 30;
  const groups = K;
  const gw = (W - padL - padR) / groups;
  const py = (v: number) => H - padB - (H - padT - padB) * v;     // probabilities in [0,1]
  const hPost = entropyBits(post);
  const hPrior = entropyBits(PRIOR);
  const pmiStar = pmiBits(post, PRIOR, A_STAR);
  const bitsRemoved = hPrior - hPost;
  return (
    <div>
      <Slider label="retrieved document d" value={d} min={0} max={K - 1} step={1}
        onChange={(v) => setD(Math.round(v))} display={docLabel(d)} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="prior and posterior answer distributions over companies" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (
          <text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>
        ))}
        {Array.from({ length: groups }, (_v, a) => {
          const x = padL + a * gw;
          const bw = gw * 0.36;
          const isStar = a === A_STAR;
          return (
            <g key={a}>
              {/* prior bar (muted) */}
              <rect x={r2(x + gw * 0.14)} y={r2(py(PRIOR[a]))} width={r2(bw)} height={r2((H - padB) - py(PRIOR[a]))}
                fill={PRIOR_COLOR} fillOpacity={0.55} />
              {/* posterior bar (accent; the true answer highlighted) */}
              <rect x={r2(x + gw * 0.14 + bw)} y={r2(py(post[a]))} width={r2(bw)} height={r2((H - padB) - py(post[a]))}
                fill={isStar ? POS_COLOR : ACCENT} fillOpacity={isStar ? 0.9 : 0.6} />
              <text x={r2(x + gw / 2)} y={H - padB + 12} textAnchor="middle" fontSize={8} fill={isStar ? POS_COLOR : 'var(--color-text-secondary)'} fontFamily="var(--font-sans)">
                {isStar ? 'a*' : a}
              </text>
            </g>
          );
        })}
        <text x={padL} y={11} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          ▮ prior p(a|q) · ▮ posterior p(a|q,d); answer = company (worked query {WORKED_Q}, sector {Q_SECTOR})
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="H(A|Q) prior" value={`${fmt(hPrior)} bits`} />
        <Readout label="H(A|Q,d) posterior" value={`${fmt(hPost)} bits`} />
        <Readout label="bits removed = H(A|Q) − H(A|Q,d)" value={fmt(bitsRemoved)} accent />
        <Readout label="pmi(a*; d | q)" value={`${pmiStar >= 0 ? '+' : ''}${fmt(pmiStar)} bits`} color={pmiStar >= 0 ? POS_COLOR : NEG_COLOR} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        The prior spreads over the sector's companies (uncertain). The <strong>gold filing</strong> (doc {A_STAR}) sharpens
        the posterior onto a* and gives a <span style={{ color: POS_COLOR }}>positive pmi</span>; the same-sector{' '}
        <strong>distractor</strong> (doc {DISTRACTOR}) sharpens onto the <em>wrong</em> company, so pmi at a* turns{' '}
        <span style={{ color: NEG_COLOR }}>negative</span> — it costs bits. A far-sector filing barely touches a*.
      </p>
    </div>
  );
}

// ===== Panel B — pmi-at-true-answer histogram, relevant vs distractor ===============================
const HIST_LO = -2.2, HIST_HI = 1.6, HIST_BINS = 19;
function SignPanel() {
  const W = 520, H = 210, padL = 32, padR = 12, padT = 14, padB = 30;
  const rel = histogram(PMI_REL, HIST_LO, HIST_HI, HIST_BINS);
  const dis = histogram(PMI_DISTR, HIST_LO, HIST_HI, HIST_BINS);
  const maxC = Math.max(...rel, ...dis, 1);
  const bw = (W - padL - padR) / HIST_BINS;
  const px = (v: number) => padL + (W - padL - padR) * ((v - HIST_LO) / (HIST_HI - HIST_LO));
  const py = (c: number) => H - padB - (H - padT - padB) * (c / maxC);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="histograms of pmi at the true answer for relevant and distractor documents" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[-2, -1, 0, 1].map((v) => (
          <g key={v}>
            <text x={px(v)} y={H - padB + 13} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>
          </g>
        ))}
        {/* zero line — the sign boundary */}
        <line x1={px(0)} y1={padT} x2={px(0)} y2={H - padB} stroke="var(--color-text)" strokeWidth={1} strokeDasharray="3 3" />
        <text x={px(0)} y={padT - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text)" fontFamily="var(--font-sans)">pmi = 0</text>
        {dis.map((c, i) => (
          c > 0 ? <rect key={`d${i}`} x={r2(padL + i * bw + 0.5)} y={r2(py(c))} width={r2(bw - 1)} height={r2((H - padB) - py(c))} fill={NEG_COLOR} fillOpacity={0.6} /> : null
        ))}
        {rel.map((c, i) => (
          c > 0 ? <rect key={`r${i}`} x={r2(padL + i * bw + 0.5)} y={r2(py(c))} width={r2(bw - 1)} height={r2((H - padB) - py(c))} fill={POS_COLOR} fillOpacity={0.6} /> : null
        ))}
        <line x1={px(MEAN_PMI_DISTR)} y1={padT} x2={px(MEAN_PMI_DISTR)} y2={H - padB} stroke={NEG_COLOR} strokeWidth={2} />
        <line x1={px(MEAN_PMI_REL)} y1={padT} x2={px(MEAN_PMI_REL)} y2={H - padB} stroke={POS_COLOR} strokeWidth={2} />
        <text x={padL + 4} y={padT + 8} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          ▮ relevant filings · ▮ distractor filings (pmi at the true answer a*)
        </text>
      </svg>
      <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="mean pmi(a*) · relevant" value={`+${fmt(MEAN_PMI_REL)} bits`} color={POS_COLOR} />
        <Readout label="mean pmi(a*) · distractor" value={`${fmt(MEAN_PMI_DISTR)} bits`} color={NEG_COLOR} />
        <Readout label="distractors with pmi(a*) < 0" value={`${fmt(FRAC_DISTR_NEG * 100, 1)}%`} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Expected information gain KL(post‖prior) is always ≥ 0 — but the <em>pointwise</em> pmi at the true answer is not.
        A relevant filing lands almost entirely on the <span style={{ color: POS_COLOR }}>positive</span> side; a
        plausible same-sector distractor lands on the <span style={{ color: NEG_COLOR }}>negative</span> side — it moves
        belief (its KL is positive), just toward the wrong company, so it <strong>costs</strong> bits at the truth.
      </p>
    </div>
  );
}

// ===== Panel C — diminishing returns =================================================================
function SaturationPanel({ novel, setNovel }: { novel: boolean; setNovel: (v: boolean) => void }) {
  const W = 420, H = 200, padL = 40, padR = 16, padT = 18, padB = 30;
  const second = novel ? SAT.novel : SAT.redundant;
  const total = SAT.standalone + second;
  const secondColor = novel ? POS_COLOR : NEG_COLOR;
  const yMax = SAT.standalone + SAT.novel + 0.1;
  const py = (v: number) => H - padB - (H - padT - padB) * (v / yMax);
  const bw = 90;
  const xs = [padL + 40, padL + 40 + bw + 60];
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <button type="button" style={pill(!novel)} onClick={() => setNovel(false)}>2nd doc: identical (redundant)</button>
        <button type="button" style={pill(novel)} onClick={() => setNovel(true)}>2nd doc: different (novel)</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="belief movement of a first filing and a redundant versus novel second filing" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0, 0.5, 1].map((v) => (
          <text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>
        ))}
        <text x={12} y={(padT + H - padB) / 2} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 12 ${(padT + H - padB) / 2})`}>bits moved</text>
        {/* stacked: first filing, then the increment from the second */}
        <rect x={xs[0]} y={r2(py(SAT.standalone))} width={bw} height={r2((H - padB) - py(SAT.standalone))} fill={ACCENT} fillOpacity={0.65} />
        <text x={xs[0] + bw / 2} y={r2(py(SAT.standalone)) - 5} textAnchor="middle" fontSize={9} fill={ACCENT} fontFamily="var(--font-sans)">{fmt(SAT.standalone)}</text>
        <text x={xs[0] + bw / 2} y={H - padB + 13} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">1st filing (gold)</text>
        {/* second filing stacked on top of the first */}
        <rect x={xs[1]} y={r2(py(total))} width={bw} height={r2(py(SAT.standalone) - py(total))} fill={secondColor} fillOpacity={0.7} />
        <rect x={xs[1]} y={r2(py(SAT.standalone))} width={bw} height={r2((H - padB) - py(SAT.standalone))} fill={ACCENT} fillOpacity={0.2} />
        <text x={xs[1] + bw / 2} y={r2(py(total)) - 5} textAnchor="middle" fontSize={9} fill={secondColor} fontFamily="var(--font-sans)">+{fmt(second)}</text>
        <text x={xs[1] + bw / 2} y={H - padB + 13} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{novel ? '+ different filing' : '+ identical filing'}</text>
      </svg>
      <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="1st filing moves belief" value={`${fmt(SAT.standalone)} bits`} accent />
        <Readout label="2nd identical filing adds" value={`${fmt(SAT.redundant)} bits`} color={NEG_COLOR} />
        <Readout label="2nd different filing adds" value={`${fmt(SAT.novel)} bits`} color={POS_COLOR} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Belief movement is KL(new ‖ old). A <strong>redundant</strong> second copy moves the posterior almost nowhere
        ({fmt(SAT.redundant)} bits); a <strong>different</strong> filing moves it far more ({fmt(SAT.novel)} bits). That is
        the chain rule of mutual information — the marginal value of evidence diminishes with redundancy, which is exactly
        why selecting a <em>diverse</em> context matters.
      </p>
    </div>
  );
}

// ===== Panel D — the InfoNCE ceiling (left) + bits vs recall (right) ================================
function BoundPanel() {
  // left: bound vs ceiling as m grows.
  const W = 300, H = 190, padL = 34, padR = 14, padT = 16, padB = 30;
  const mMax = 8, yMax = 3.2;
  const px = (m: number) => padL + (W - padL - padR) * ((m - 2) / (mMax - 2));
  const py = (v: number) => H - padB - (H - padT - padB) * (v / yMax);
  const ceilPath = INFONCE.map((r, i) => (i ? 'L' : 'M') + r2(px(r.m)) + ' ' + r2(py(r.ceiling))).join(' ');
  const boundPath = INFONCE.map((r, i) => (i ? 'L' : 'M') + r2(px(r.m)) + ' ' + r2(py(r.bound))).join(' ');
  // right: scatter (recall = 1.0 for all; bits vary).
  const sW = 220, sH = 190, sPadL = 40, sPadR = 12, sPadT = 16, sPadB = 30;
  const bMax = 1.8;
  const sx = (recall: number) => sPadL + (sW - sPadL - sPadR) * recall;
  const sy = (bits: number) => sH - sPadB - (sH - sPadT - sPadB) * (bits / bMax);
  const last = INFONCE[INFONCE.length - 1];
  return (
    <div>
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="InfoNCE lower bound rising toward its log2(m) ceiling" style={{ width: '100%', maxWidth: W, height: 'auto', flex: '1 1 280px' }}>
          <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
          {[0, 1, 2, 3].map((v) => (<text key={v} x={padL - 5} y={py(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
          {[2, 4, 6, 8].map((m) => (<text key={m} x={px(m)} y={H - padB + 13} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{m}</text>))}
          <text x={(padL + W - padR) / 2} y={H - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">in-batch candidates m</text>
          <path d={ceilPath} fill="none" stroke={MUTED} strokeWidth={1.6} strokeDasharray="4 3" />
          <path d={boundPath} fill="none" stroke={ACCENT} strokeWidth={2} />
          {INFONCE.map((r) => (<circle key={r.m} cx={px(r.m)} cy={py(r.bound)} r={2.4} fill={ACCENT} />))}
          <text x={px(6)} y={py(2.85)} fontSize={8.5} fill={MUTED} fontFamily="var(--font-sans)">ceiling log₂(m)</text>
          <text x={px(5.2)} y={py(1.2)} fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">bound</text>
        </svg>
        <svg viewBox={`0 0 ${sW} ${sH}`} role="img" aria-label="per-query recall versus bits added scatter" style={{ width: '100%', maxWidth: sW, height: 'auto', flex: '1 1 200px' }}>
          <line x1={sPadL} y1={sH - sPadB} x2={sW - sPadR} y2={sH - sPadB} stroke="var(--color-border)" strokeWidth={1} />
          <line x1={sPadL} y1={sPadT} x2={sPadL} y2={sH - sPadB} stroke="var(--color-border)" strokeWidth={1} />
          {[0, 0.5, 1].map((v) => (<text key={v} x={sx(v)} y={sH - sPadB + 13} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
          {[0, 1].map((v) => (<text key={v} x={sPadL - 5} y={sy(v) + 3} textAnchor="end" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{v}</text>))}
          <text x={(sPadL + sW - sPadR) / 2} y={sH - 3} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">recall@3</text>
          <text x={11} y={(sPadT + sH - sPadB) / 2} textAnchor="middle" fontSize={8.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)" transform={`rotate(-90 11 ${(sPadT + sH - sPadB) / 2})`}>bits added</text>
          {SCATTER_BITS.map((b, i) => (<circle key={i} cx={sx(1) + ((i % 5) - 2) * 2.2} cy={sy(b)} r={2.6} fill={ACCENT} fillOpacity={0.6} />))}
          <text x={sx(1)} y={sPadT + 6} textAnchor="middle" fontSize={8} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">recall ≡ 1.0</text>
        </svg>
      </div>
      <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label={`bound at m = ${last.m}`} value={`${fmt(last.bound)} bits`} accent />
        <Readout label={`ceiling log₂(${last.m})`} value={`${fmt(last.ceiling)} bits`} />
        <Readout label="recall@3 (every query)" value="1.000" />
        <Readout label="bits added (range)" value={`${fmt(BITS_MIN)} – ${fmt(BITS_MAX)}`} accent />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        <strong>Left:</strong> the dense encoder was trained by minimizing InfoNCE, i.e. maximizing the lower bound
        log₂(m) − L on the query–document information — but the bound can never exceed its <span style={{ color: MUTED }}>ceiling
        log₂(m)</span>, so it <strong>saturates</strong>; more in-batch negatives are needed to certify more bits.{' '}
        <strong>Right:</strong> on this easy corpus recall is pinned at 1.0 for every query, yet the bits retrieval adds
        range from {fmt(BITS_MIN)} to {fmt(BITS_MAX)} — recall sees every query as "perfect" while information gain does not.
        Presence is not contribution.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'model' | 'sign' | 'saturation' | 'bound';
const TEX: Record<Panel, string> = {
  model: '\\mathrm{pmi}(a; d \\mid q) = \\log_2 \\frac{p(a \\mid q, d)}{p(a \\mid q)}, \\qquad p(a\\mid q) = \\sum_d p(d\\mid q)\\,p(a\\mid q,d)',
  sign: 'I(A;D\\mid Q) = H(A\\mid Q) - H(A\\mid Q,D) = \\mathbb{E}\\big[\\mathrm{KL}(p(\\cdot\\mid q,d)\\,\\|\\,p(\\cdot\\mid q))\\big] \\ge 0',
  saturation: 'I(A;D_1,D_2\\mid Q) = I(A;D_1\\mid Q) + I(A;D_2\\mid Q,D_1)',
  bound: 'I(Q;D) \\;\\ge\\; \\log_2(N+1) - \\mathcal{L}_{\\mathrm{InfoNCE}}',
};

export default memo(function PMIRetrievalValueLaboratory() {
  const [panel, setPanel] = useState<Panel>('model');
  const [d, setD] = useState(A_STAR);
  const [novel, setNovel] = useState(false);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="pmi-retrieval-value" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'model')} onClick={() => setPanel('model')}>A · prior → posterior</button>
        <button type="button" style={pill(panel === 'sign')} onClick={() => setPanel('sign')}>B · pmi sign</button>
        <button type="button" style={pill(panel === 'saturation')} onClick={() => setPanel('saturation')}>C · diminishing returns</button>
        <button type="button" style={pill(panel === 'bound')} onClick={() => setPanel('bound')}>D · InfoNCE & recall</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'model' && <AnswerModelPanel d={d} setD={setD} />}
      {panel === 'sign' && <SignPanel />}
      {panel === 'saturation' && <SaturationPanel novel={novel} setNovel={setNovel} />}
      {panel === 'bound' && <BoundPanel />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        Finance vMF corpus reused from the dense-retrieval topic ({K} companies across {SECTOR[SECTOR.length - 1] + 1} sectors,
        one filing per company), {N_QUERIES} sector-ambiguous queries; the answer model is a synthetic softmax stand-in, not
        an LLM. I(A;D|Q) = {fmt(I_ADQ)} bits (H(A|Q) = {fmt(H_A_GIVEN_Q)} → H(A|Q,D) = {fmt(H_A_GIVEN_QD)}). Numbers mirror{' '}
        <code>pmi_retrieval_value.py</code>; the lab recomputes entropy, pmi, and histogram bins in closed form.
      </p>
    </div>
  );
});
