import { memo, useEffect, useRef, useState } from 'react';
import katex from 'katex';

/**
 * Learning-to-Rank Laboratory — four panels for the `learning-to-rank-pairwise` topic:
 *   A. The three reductions. The constructed two-query witness for "ranking is not regression": toggle
 *      pointwise / pairwise / listwise. Pointwise (global least squares) has lower MSE yet mis-orders a
 *      query; pairwise (RankNet) orders both better at higher MSE — order beats calibration.
 *   B. Rank metrics are flat. Morphing the weights from the weakest leg to the learned combiner, NDCG
 *      climbs as a STAIRCASE (piecewise-constant, jumps at swaps) while the smooth surrogate loss falls
 *      continuously — the theorem that forces a surrogate, made visible.
 *   C. RankNet mechanics. The worked query's top-k with each document's per-document lambda force (the
 *      single grade-3 doc is under-ranked and feels the strongest upward pull); a margin slider sharpens
 *      the pairwise probability P(i > j) = sigma(s_i - s_j).
 *   D. Learned fusion vs RRF. Recall@10 / NDCG@10 bars for the three legs, RRF, pointwise, and the
 *      learned RankNet; the learned weights; and a training-set-size slider showing the learned ranker
 *      overtake RRF within a couple of labeled queries, then plateau.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every baked number is mirrored TO THE DECIMAL from
 * notebooks/learning-to-rank-pairwise/learning_to_rank_pairwise.py (viz_constants()). Matching asserts:
 * test_pointwise_equals_lstsq / test_ranknet_loss_convex_unique / test_ranknet_grad_fd /
 * test_constructed_ranking_beats_regression / test_surrogate_is_stepwise_vs_smooth /
 * test_gradient_factorizes / test_learned_vs_baselines_runs. The lab recomputes ONLY closed forms in TS
 * (the witness scores + induced orders, the linear scores s = w.x and sigma(s_i - s_j), the bar geometry,
 * the gaps); the rng/Newton-driven scalars (the fitted weights, per-method metrics, the NDCG staircase,
 * the loss curve, the learning curve, the lambda forces) are baked. Change a number here -> change it
 * there, and re-run the notebook.
 */

// --- baked from viz_constants() -------------------------------------------------------
const N_DOCS = 120;
const N_QUERIES = 40;
const N_TRAIN = 24;
const N_TEST = 16;
const TOPK = 10;

type Method = 'lexical' | 'dense' | 'late_interaction' | 'rrf' | 'pointwise' | 'learned';
const METHODS: Method[] = ['lexical', 'dense', 'late_interaction', 'rrf', 'pointwise', 'learned'];
const METHOD_LABEL: Record<Method, string> = {
  lexical: 'lexical', dense: 'dense', late_interaction: 'late interaction',
  rrf: 'RRF', pointwise: 'pointwise', learned: 'learned RankNet',
};
const METHOD_RECALL: Record<Method, number> = {
  lexical: 0.4688, dense: 0.5875, late_interaction: 0.475, rrf: 0.6187, pointwise: 0.6312, learned: 0.6375,
};
const METHOD_NDCG: Record<Method, number> = {
  lexical: 0.6038, dense: 0.6889, late_interaction: 0.5143, rrf: 0.7326, pointwise: 0.7678, learned: 0.7668,
};
const LEARNED_W: [number, number, number] = [0.6976, 1.443, 0.6842];   // [lex, dense, li]
const LEARNED_CI = { mean: 0.6375, se: 0.0397, lo: 0.5598, hi: 0.7152 };

// Panel A — the constructed two-query witness (fixed pedagogical toy; weights, MSE, NDCG baked).
const H1 = { msePt: 0.8, msePw: 1.2083, ndcgPt: 0.7153, ndcgPw: 0.8155 };
const H1_W_PT: [number, number, number] = [0.6, 1.12, -0.48];          // [x1(order), x2(query-level), intercept]
const H1_W_PW = -1.0986;                                               // pairwise uses x1 only
type WDoc = { id: string; x1: number; x2: number; g: number };
const WITNESS: { q: number; docs: WDoc[] }[] = [
  { q: 1, docs: [{ id: 'A', x1: 1, x2: 1.5, g: 3 }, { id: 'B', x1: 0, x2: 1.5, g: 0 }] },
  { q: 2, docs: [{ id: 'C', x1: 0, x2: 0.25, g: 1 }, { id: 'D', x1: 1, x2: 0.25, g: 0 },
                 { id: 'E', x1: 1, x2: 0.25, g: 0 }, { id: 'F', x1: 1, x2: 0.25, g: 0 }] },
];

// Panel C — the worked query (id 13): trained top-k order, grades, per-doc lambda forces, features.
const WORKED_Q = 13;
const WORKED_RANKING = [35, 18, 111, 97, 85, 57, 91, 47, 3, 17];
const WORKED_GRADE = [2, 2, 2, 2, 2, 3, 2, 0, 0, 0];                   // grade of each doc in ranking order
const WORKED_LAMBDA = [-0.052, -0.2071, -0.372, -1.0485, -2.0235, -9.6084, -5.0309, 3.9107, 3.5194, 3.3017];
const WORKED_FEATS: [number, number, number][] = [
  [3.2466, 2.0757, 1.8174], [2.6441, 2.2667, 1.8161], [2.6441, 2.1484, 1.8659], [3.5917, 1.5813, 1.4581],
  [1.5941, 2.2912, 1.3546], [2.0128, 1.5849, 1.422], [0.8091, 2.0344, 1.4581], [0.272, 2.1048, 1.3546],
  [0.617, 1.7323, 1.3844], [0.272, 1.8006, 1.3546],
];

// Panel B — the weights morph from the weakest leg (late interaction) to the learned combiner.
const SWEEP_TS = [0, 0.0209, 0.0418, 0.0628, 0.0837, 0.1046, 0.1297, 0.1506, 0.1715, 0.1925, 0.2134, 0.2343, 0.2552, 0.2762, 0.2971, 0.318, 0.3389, 0.3598, 0.3849, 0.4059, 0.4268, 0.4477, 0.4686, 0.4895, 0.5105, 0.5314, 0.5523, 0.5732, 0.5941, 0.6151, 0.6402, 0.6611, 0.682, 0.7029, 0.7238, 0.7448, 0.7657, 0.7866, 0.8075, 0.8285, 0.8494, 0.8703, 0.8954, 0.9163, 0.9372, 0.9582, 0.9791, 1.0];
const NDCG_SWEEP = [0.5143, 0.5363, 0.5774, 0.5883, 0.591, 0.6013, 0.6294, 0.6323, 0.6473, 0.6505, 0.6573, 0.6683, 0.683, 0.6956, 0.6986, 0.7086, 0.7197, 0.7255, 0.7272, 0.7332, 0.7352, 0.7447, 0.7468, 0.7529, 0.7531, 0.7538, 0.7539, 0.7545, 0.7558, 0.7572, 0.7518, 0.7526, 0.7552, 0.7555, 0.7557, 0.7654, 0.7655, 0.7655, 0.7655, 0.7657, 0.7679, 0.7679, 0.7686, 0.7686, 0.7667, 0.7669, 0.7669, 0.7668];
const LOSS_SWEEP = [6033.6, 5852.6, 5681.7, 5520.6, 5368.6, 5225.3, 5064.1, 4938.2, 4819.5, 4707.6, 4602.2, 4502.9, 4409.2, 4321, 4237.9, 4159.6, 4085.9, 4016.4, 3938.4, 3877.6, 3820.4, 3766.6, 3715.9, 3668.4, 3623.7, 3581.7, 3542.3, 3505.3, 3470.7, 3438.3, 3402.2, 3374.2, 3348.1, 3323.8, 3301.2, 3280.3, 3260.8, 3242.8, 3226.2, 3211, 3197, 3184.2, 3170.4, 3160.1, 3150.9, 3142.6, 3135.4, 3129];

// Panel D — learned test recall vs #train queries (it overtakes RRF by 2 queries, then plateaus).
const LEARN_CURVE: [number, number][] = [[1, 0.5813], [2, 0.6312], [3, 0.6312], [4, 0.6437], [6, 0.6437], [8, 0.6375], [12, 0.6437], [16, 0.6375], [20, 0.6437], [24, 0.6375]];

const ACCENT = 'var(--color-accent)';
const LEG_COLOR: Record<string, string> = {
  lexical: '#c08457', dense: '#6a8caf', late_interaction: '#9b6abf',
  rrf: '#7a8a99', pointwise: '#b07d9b', learned: '#5fa873',
};
const GRADE_COLOR = ['#cdd3da', '#b9c79a', '#86a35e', '#4f7a3a'];      // grade 0,1,2,3
const UP_COLOR = '#5fa873';     // upward lambda (pulled up)
const DOWN_COLOR = '#c25b6b';   // downward lambda (pulled down)
const MUTED = '#9aa3ad';
const fmt = (x: number, n = 3) => x.toFixed(n);
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const argsortDesc = (s: number[]) => s.map((v, i) => [v, i] as [number, number])
  .sort((a, b) => (b[0] - a[0]) || (a[1] - b[1])).map(([, i]) => i);
// NDCG of a grade sequence in rank order (exponential gain, log2 discount) — closed form, for the witness.
const ndcgOfGrades = (gradesInOrder: number[]) => {
  const dcg = gradesInOrder.reduce((s, g, i) => s + (2 ** g - 1) / Math.log2(i + 2), 0);
  const ideal = [...gradesInOrder].sort((a, b) => b - a);
  const idcg = ideal.reduce((s, g, i) => s + (2 ** g - 1) / Math.log2(i + 2), 0);
  return idcg > 0 ? dcg / idcg : 0;
};

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
      <span style={{ minWidth: '12rem' }}>{label} = <strong>{display}</strong></span>
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

// ===== Panel A — the three reductions (the constructed witness) =====================================
type Reduction = 'pointwise' | 'pairwise' | 'listwise';
function ReductionsPanel({ red, setRed }: { red: Reduction; setRed: (r: Reduction) => void }) {
  const score = (d: WDoc) =>
    red === 'pairwise' ? H1_W_PW * d.x1 : H1_W_PT[0] * d.x1 + H1_W_PT[1] * d.x2 + H1_W_PT[2];
  const queryRows = WITNESS.map(({ q, docs }) => {
    const order = argsortDesc(docs.map(score));
    const ordered = order.map((i) => docs[i]);
    return { q, ordered, ndcg: ndcgOfGrades(ordered.map((d) => d.g)) };
  });
  const meanNdcg = red === 'pairwise' ? H1.ndcgPw : H1.ndcgPt;     // baked (matches the recomputed order)
  const mse = red === 'pairwise' ? H1.msePw : H1.msePt;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        {(['pointwise', 'pairwise', 'listwise'] as Reduction[]).map((r) => (
          <button key={r} type="button" style={pill(red === r)} onClick={() => setRed(r)}>{r}</button>
        ))}
      </div>
      {red === 'listwise' ? (
        <div style={{ border: '1px dashed var(--color-border)', borderRadius: '0.5rem', padding: '1.2rem', color: 'var(--color-text-secondary)', fontSize: '0.85rem', lineHeight: 1.6 }}>
          <strong>Listwise (preview).</strong> Rather than scoring documents one at a time (pointwise) or in
          pairs (pairwise), a listwise method scores the <em>whole permutation</em> at once — e.g. ListNet
          maps the scores to a Plackett–Luce distribution over orderings and minimizes cross-entropy
          against the ideal list. It is the subject of a later topic; here we develop pointwise and
          pairwise in full.
        </div>
      ) : (
        <div style={{ display: 'flex', gap: '1.4rem', flexWrap: 'wrap' }}>
          {queryRows.map(({ q, ordered, ndcg }) => (
            <div key={q} style={{ flex: '1 1 230px' }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', marginBottom: '0.35rem', fontFamily: 'var(--font-sans)' }}>
                query {q} — ranked by {red} score
              </div>
              <div style={{ display: 'flex', gap: '0.35rem' }}>
                {ordered.map((d, rank) => (
                  <div key={d.id} title={`grade ${d.g}`} style={{
                    width: '2.1rem', height: '2.1rem', borderRadius: '0.35rem', display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center', background: GRADE_COLOR[d.g],
                    border: rank === 0 ? `2px solid ${ACCENT}` : '1px solid var(--color-border)',
                    color: d.g >= 2 ? '#fff' : 'var(--color-text)', fontSize: '0.74rem', fontWeight: 600,
                  }}>
                    <span>{d.id}</span><span style={{ fontSize: '0.6rem', fontWeight: 400 }}>g{d.g}</span>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.3rem' }}>
                NDCG = <strong>{fmt(ndcg)}</strong>{rankNote(q, red)}
              </div>
            </div>
          ))}
        </div>
      )}
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.7rem', flexWrap: 'wrap' }}>
        <Readout label="calibrated MSE (lower = better fit)" value={fmt(mse, 3)} accent={red === 'pointwise'} />
        <Readout label="mean NDCG (higher = better order)" value={fmt(meanNdcg, 3)} accent={red === 'pairwise'} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        <strong>Ranking is not regression.</strong> Pointwise least squares is the global MSE minimizer
        ({fmt(H1.msePt, 2)} vs {fmt(H1.msePw, 2)}) and even spends a query-level feature on calibration — yet
        it mis-orders query 2 and scores NDCG {fmt(H1.ndcgPt, 3)}. Pairwise RankNet ignores that feature,
        orders both queries better, and scores NDCG {fmt(H1.ndcgPw, 3)} at <em>worse</em> MSE. Order beats
        calibration.
      </p>
    </div>
  );
}
const rankNote = (q: number, red: Reduction) =>
  red === 'pointwise' && q === 2 ? ' — the relevant doc C is mis-ranked' : '';

// ===== Panel B — rank metrics are flat (the surrogate sweep) ========================================
function SweepPanel({ idx, setIdx }: { idx: number; setIdx: (v: number) => void }) {
  const W = 540, H = 240, padL = 44, padR = 48, padT = 16, padB = 34;
  const n = SWEEP_TS.length;
  const lossMin = Math.min(...LOSS_SWEEP), lossMax = Math.max(...LOSS_SWEEP);
  const nd0 = 0.5, nd1 = 0.8;                                  // NDCG axis bounds
  const px = (t: number) => padL + (W - padL - padR) * t;
  const pyN = (v: number) => H - padB - (H - padT - padB) * ((v - nd0) / (nd1 - nd0));
  const pyL = (v: number) => H - padB - (H - padT - padB) * ((v - lossMin) / (lossMax - lossMin));
  // NDCG as a step path (hold then jump); loss as a smooth polyline.
  let stepD = `M ${px(SWEEP_TS[0])} ${pyN(NDCG_SWEEP[0])}`;
  for (let i = 1; i < n; i++) stepD += ` L ${px(SWEEP_TS[i])} ${pyN(NDCG_SWEEP[i - 1])} L ${px(SWEEP_TS[i])} ${pyN(NDCG_SWEEP[i])}`;
  const lossD = LOSS_SWEEP.map((v, i) => (i ? 'L' : 'M') + px(SWEEP_TS[i]) + ' ' + pyL(v)).join(' ');
  const t = SWEEP_TS[idx];
  return (
    <div>
      <Slider label="weights w(t): weakest leg → learned" value={idx} min={0} max={n - 1} step={1}
        onChange={(v) => setIdx(Math.round(v))} display={`t = ${fmt(t, 2)}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="NDCG staircase and smooth surrogate loss along the weight sweep" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        <line x1={W - padR} y1={padT} x2={W - padR} y2={H - padB} stroke="var(--color-border)" strokeWidth={1} />
        {[0.5, 0.6, 0.7, 0.8].map((v) => (
          <text key={v} x={padL - 6} y={pyN(v) + 3} textAnchor="end" fontSize={8.5} fill={ACCENT} fontFamily="var(--font-sans)">{v}</text>
        ))}
        <text x={px(0.5)} y={H - 6} textAnchor="middle" fontSize={9.5} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">interpolation t (weight morph)</text>
        <text x={14} y={(padT + H - padB) / 2} textAnchor="middle" fontSize={9.5} fill={ACCENT} fontFamily="var(--font-sans)" transform={`rotate(-90 14 ${(padT + H - padB) / 2})`}>NDCG@10 (metric)</text>
        <text x={W - 10} y={(padT + H - padB) / 2} textAnchor="middle" fontSize={9.5} fill={MUTED} fontFamily="var(--font-sans)" transform={`rotate(-90 ${W - 10} ${(padT + H - padB) / 2})`}>pairwise loss (surrogate)</text>
        <path d={lossD} fill="none" stroke={MUTED} strokeWidth={2} strokeDasharray="5 3" />
        <path d={stepD} fill="none" stroke={ACCENT} strokeWidth={2} />
        <line x1={px(t)} y1={padT} x2={px(t)} y2={H - padB} stroke="var(--color-text-secondary)" strokeWidth={1} strokeDasharray="2 2" />
        <circle cx={px(t)} cy={pyN(NDCG_SWEEP[idx])} r={4} fill={ACCENT} />
        <circle cx={px(t)} cy={pyL(LOSS_SWEEP[idx])} r={4} fill={MUTED} />
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="NDCG@10 here (a flat plateau)" value={fmt(NDCG_SWEEP[idx], 4)} accent />
        <Readout label="surrogate loss here (descending)" value={fmt(LOSS_SWEEP[idx], 1)} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Nudge the slider: across most steps the <span style={{ color: ACCENT }}>NDCG</span> does not move at all
        — it is <strong>piecewise-constant</strong>, flat between swaps and jumping at them, so its gradient is
        zero almost everywhere and you cannot descend it. The <span style={{ color: MUTED }}>surrogate loss</span>
        we actually optimize falls <strong>smoothly</strong> the whole way. The metric is what we want; the
        surrogate is what we can train.
      </p>
    </div>
  );
}

// ===== Panel C — RankNet mechanics: the lambda forces ===============================================
function MechanicsPanel({ margin, setMargin }: { margin: number; setMargin: (v: number) => void }) {
  const scores = WORKED_FEATS.map(([a, b, c]) => LEARNED_W[0] * a + LEARNED_W[1] * b + LEARNED_W[2] * c);
  const gradeIdx = Math.max(0, WORKED_GRADE.indexOf(3));        // the lone grade-3 doc (fallback: top rank)
  const top = 0;
  const gap = scores[top] - scores[gradeIdx];                   // the model ranks doc above the true best
  const pTrue = sigmoid(margin * -gap);                         // P(grade-3 doc outranks the current top)
  const maxAbsLam = Math.max(...WORKED_LAMBDA.map(Math.abs));
  const W = 540, rowH = 26, x0 = 8, barX = 250, barW = 150, H = WORKED_RANKING.length * rowH + 30;
  return (
    <div>
      <Slider label="score margin (sharpness of σ)" value={margin} min={0.2} max={3} step={0.05}
        onChange={setMargin} display={`×${fmt(margin, 2)}`} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="worked query top-k with per-document lambda forces" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        <text x={x0} y={14} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rank · doc · grade</text>
        <text x={barX + barW / 2} y={14} textAnchor="middle" fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">← pull down · λ force · pull up →</text>
        {WORKED_RANKING.map((doc, i) => {
          const y = 24 + i * rowH;
          const lam = WORKED_LAMBDA[i];
          const g = WORKED_GRADE[i];
          const bl = (Math.abs(lam) / maxAbsLam) * (barW / 2);
          const cx = barX + barW / 2;
          const isG3 = i === gradeIdx;
          return (
            <g key={doc}>
              <rect x={x0} y={y - 11} width={16} height={16} rx={3} fill={GRADE_COLOR[g]} stroke={isG3 ? ACCENT : 'var(--color-border)'} strokeWidth={isG3 ? 2 : 1} />
              <text x={x0 + 26} y={y + 1} fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">
                {i + 1}.  doc {doc}  ·  g{g}{isG3 ? '  ← the answer, under-ranked' : ''}
              </text>
              <line x1={cx} y1={y - 9} x2={cx} y2={y + 7} stroke="var(--color-border)" strokeWidth={1} />
              {lam < 0
                ? <rect x={cx} y={y - 5} width={bl} height={10} fill={UP_COLOR} fillOpacity={0.8} />
                : <rect x={cx - bl} y={y - 5} width={bl} height={10} fill={DOWN_COLOR} fillOpacity={0.8} />}
              <text x={lam < 0 ? cx + bl + 4 : cx - bl - 4} y={y + 3} textAnchor={lam < 0 ? 'start' : 'end'} fontSize={8} fill={lam < 0 ? UP_COLOR : DOWN_COLOR} fontFamily="var(--font-sans)">{fmt(lam, 1)}</text>
            </g>
          );
        })}
      </svg>
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.4rem', flexWrap: 'wrap' }}>
        <Readout label="P(answer ≻ current top) = σ(margin · Δ)" value={fmt(pTrue, 3)} accent />
        <Readout label="this lost pair's loss = −log P" value={fmt(-Math.log(Math.max(pTrue, 1e-9)), 3)} />
        <Readout label="λ on the grade-3 doc" value={fmt(WORKED_LAMBDA[gradeIdx], 2)} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        Each document feels a <strong>lambda force</strong> λ<sub>i</sub> = Σ<sub>j</sub> λ<sub>ij</sub> — the net pull from
        its preference pairs. On query {WORKED_Q} the single <span style={{ color: GRADE_COLOR[3] }}><strong>grade-3</strong></span>
        document is ranked sixth and feels by far the strongest <span style={{ color: UP_COLOR }}>upward</span> pull
        ({fmt(WORKED_LAMBDA[gradeIdx], 1)}), because the model is confidently wrong about the pairs it should win.
        The margin slider sharpens P(i ≻ j) = σ(s<sub>i</sub> − s<sub>j</sub>): a wider score gap means a more confident — and
        more heavily penalized — mistake. LambdaRank scales each force by the ΔNDCG the swap would cause.
      </p>
    </div>
  );
}

// ===== Panel D — learned fusion vs RRF ==============================================================
function FusionPanel({ metric, setMetric, ntr, setNtr }: {
  metric: 'recall' | 'ndcg'; setMetric: (m: 'recall' | 'ndcg') => void; ntr: number; setNtr: (v: number) => void;
}) {
  const table = metric === 'recall' ? METHOD_RECALL : METHOD_NDCG;
  const W = 540, H = 196, padL = 100, padR = 70, padT = 10, padB = 18, rowH = 28;
  const xMax = 0.85;
  const bx = (v: number) => padL + (W - padL - padR) * (v / xMax);
  const learnedAtN = LEARN_CURVE.reduce((best, [k, v]) => (k <= ntr ? [k, v] : best), LEARN_CURVE[0])[1];
  const gapLeg = METHOD_RECALL.learned - Math.max(METHOD_RECALL.lexical, METHOD_RECALL.dense, METHOD_RECALL.late_interaction);
  const gapRrf = METHOD_RECALL.learned - METHOD_RECALL.rrf;
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
        <button type="button" style={pill(metric === 'recall')} onClick={() => setMetric('recall')}>recall@10</button>
        <button type="button" style={pill(metric === 'ndcg')} onClick={() => setMetric('ndcg')}>NDCG@10</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="held-out metric bars per method" style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }}>
        {METHODS.map((m, i) => {
          const y = padT + i * rowH;
          const v = table[m];
          return (
            <g key={m}>
              <text x={padL - 8} y={y + rowH / 2 + 1} textAnchor="end" fontSize={9.5} fill="var(--color-text)" fontFamily="var(--font-sans)">{METHOD_LABEL[m]}</text>
              <rect x={padL} y={y + 4} width={bx(v) - padL} height={rowH - 12} rx={2}
                fill={LEG_COLOR[m]} fillOpacity={m === 'learned' ? 0.95 : 0.55}
                stroke={m === 'learned' ? ACCENT : 'none'} strokeWidth={m === 'learned' ? 1.5 : 0} />
              <text x={bx(v) + 5} y={y + rowH / 2 + 1} fontSize={9} fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{fmt(v, 3)}</text>
            </g>
          );
        })}
        {/* RRF reference line */}
        <line x1={bx(table.rrf)} y1={padT} x2={bx(table.rrf)} y2={H - padB} stroke={LEG_COLOR.rrf} strokeWidth={1} strokeDasharray="3 2" />
      </svg>
      <div style={{ display: 'flex', gap: '1.2rem', flexWrap: 'wrap', alignItems: 'center', margin: '0.3rem 0' }}>
        {(['lexical', 'dense', 'late_interaction'] as const).map((leg, j) => (
          <Readout key={leg} label={`learned w · ${METHOD_LABEL[leg]}`} value={fmt(LEARNED_W[j], 2)} />
        ))}
      </div>
      <Slider label="labeled training queries" value={ntr} min={1} max={N_TRAIN} step={1}
        onChange={(v) => setNtr(Math.round(v))} display={`${ntr}`} />
      <div style={{ display: 'flex', gap: '1.6rem', marginTop: '0.2rem', flexWrap: 'wrap' }}>
        <Readout label={`learned recall at n = ${ntr}`} value={fmt(learnedAtN, 4)} accent />
        <Readout label="vs RRF at this n" value={learnedAtN >= METHOD_RECALL.rrf ? 'ahead of RRF' : 'below RRF'} accent={learnedAtN >= METHOD_RECALL.rrf} />
        <Readout label="learned − best leg / − RRF" value={`+${fmt(gapLeg, 3)} / +${fmt(gapRrf, 3)}`} />
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', lineHeight: 1.5 }}>
        On the {N_TEST} held-out queries the <span style={{ color: LEG_COLOR.learned }}><strong>learned RankNet</strong></span>
        beats every single leg (+{fmt(gapLeg, 3)} recall over the best) and edges out
        <span style={{ color: LEG_COLOR.rrf }}> RRF</span> (+{fmt(gapRrf, 3)}). The margin over RRF is within the
        confidence interval ({fmt(LEARNED_CI.mean, 3)} ± {fmt(1.96 * LEARNED_CI.se, 3)}), so the robust claim is
        "matches RRF, clearly beats any single leg." With one labeled query the ranker is shaky (below RRF); by two
        it has overtaken RRF and then plateaus — a three-feature linear ranker is data-efficient.
      </p>
    </div>
  );
}

// ===== root =========================================================================================
type Panel = 'reductions' | 'flat' | 'mechanics' | 'fusion';
const TEX: Record<Panel, string> = {
  reductions: '\\text{pointwise: } \\min_{\\mathbf{w}}\\textstyle\\sum_i (\\mathbf{w}^\\top\\mathbf{x}_i - g_i)^2 \\quad\\bigm|\\quad \\text{pairwise: } \\min_{\\mathbf{w}}\\textstyle\\sum_{i\\succ j} \\log\\!\\big(1+e^{-\\mathbf{w}^\\top(\\mathbf{x}_i-\\mathbf{x}_j)}\\big)',
  flat: '\\nabla_{\\mathbf{w}}\\,\\mathrm{NDCG} = \\mathbf{0}\\ \\text{a.e.}; \\qquad L_{\\text{surr}}(\\mathbf{w}) = \\sum_{i\\succ j}\\log\\!\\big(1+e^{-(s_i-s_j)}\\big)\\ \\text{is } C^1',
  mechanics: 'P(i\\succ j)=\\sigma(s_i-s_j), \\qquad \\nabla_{\\mathbf{w}}L = \\sum_i \\lambda_i\\,\\mathbf{x}_i, \\quad \\lambda_i=\\sum_j \\lambda_{ij}',
  fusion: 's_d = w_{\\text{lex}}\\,\\text{lex}_d + w_{\\text{dense}}\\,\\text{dense}_d + w_{\\text{late}}\\,\\text{late}_d',
};

export default memo(function LearningToRankLaboratory() {
  const [panel, setPanel] = useState<Panel>('reductions');
  const [red, setRed] = useState<Reduction>('pointwise');
  const [idx, setIdx] = useState(0);
  const [margin, setMargin] = useState(1);
  const [metric, setMetric] = useState<'recall' | 'ndcg'>('recall');
  const [ntr, setNtr] = useState(1);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="learning-to-rank" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'reductions')} onClick={() => setPanel('reductions')}>A · three reductions</button>
        <button type="button" style={pill(panel === 'flat')} onClick={() => setPanel('flat')}>B · rank metrics are flat</button>
        <button type="button" style={pill(panel === 'mechanics')} onClick={() => setPanel('mechanics')}>C · RankNet & λ forces</button>
        <button type="button" style={pill(panel === 'fusion')} onClick={() => setPanel('fusion')}>D · learned fusion vs RRF</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem', overflowX: 'auto' }} />
      {panel === 'reductions' && <ReductionsPanel red={red} setRed={setRed} />}
      {panel === 'flat' && <SweepPanel idx={idx} setIdx={setIdx} />}
      {panel === 'mechanics' && <MechanicsPanel margin={margin} setMargin={setMargin} />}
      {panel === 'fusion' && <FusionPanel metric={metric} setMetric={setMetric} ntr={ntr} setNtr={setNtr} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_DOCS} synthetic finance documents, {N_QUERIES} queries ({N_TRAIN} train / {N_TEST} test); features are the
        three legs' per-document scores, grades are the top-{TOPK} exact-MaxSim oracle tertiles. Numbers mirror{' '}
        <code>learning_to_rank_pairwise.py</code>; the lab recomputes the witness orders, the linear scores and σ,
        the bars, and the gaps in closed form.
      </p>
    </div>
  );
});
