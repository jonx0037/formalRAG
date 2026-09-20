import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import katex from 'katex';

/**
 * RAG Architecture Mechanisms Laboratory — four panels for `rag-architecture-mechanisms`.
 * Six named architectures run as arms over TWO regimes, with both halves of each measured.
 *   A. The matrix. Six arms by five local query classes and the global regime. No row wins both
 *      columns, and the delta view shows what each arm costs the baseline where it does not help.
 *   B. Fusion as set intersection. A worked conjunctive query: each leg's shortlist, and the RRF
 *      constant c as a slider that re-fuses the three lists LIVE. Fusion is decisive here and a
 *      liability on the class one view already answers.
 *   C. Rank against reach. The gold's position in the cheap ranking, per class, with the
 *      over-fetch depth as a slider: the recoverable fraction is recomputed live. Correction
 *      reaches everything in the noisy class by depth 20 and almost nothing in the bridge class.
 *   D. Aggregation, precomputed or by hand. The flat-aggregation curve is recomputed EXACTLY in
 *      the browser from the baked entity ranking and mention flags, against GraphRAG's flat line.
 *
 * VIZ <-> PYTHON INVARIANT (CLAUDE.md): every constant below is EMITTED FROM viz_constants() in
 * notebooks/rag-architecture-mechanisms/rag_architecture_mechanisms.py rather than transcribed,
 * so the two cannot drift. Matching asserts: test_no_arm_wins_both_regimes /
 * test_each_arm_owns_exactly_one_condition / test_fusion_damages_a_query_one_view_answers /
 * test_fusion_wins_only_where_views_disagree / test_correction_fixes_rank_not_reach /
 * test_grader_cannot_see_the_reach_failure / test_only_hops_reach_the_named_company /
 * test_graphrag_buys_depth_not_possibility / test_the_average_hides_what_correction_breaks /
 * test_viz_constants_reproduce_the_curve (which asserts THIS panel D recomputation lands on the
 * curve the module measured). Panels B, C and D recompute closed forms in TS; the corpus itself
 * (vMF draws, Leiden, MaxSim) is baked, since TS cannot reproduce it.
 */

// --- emitted from rag_architecture_mechanisms.py viz_constants() ---
const ARMS = ["naive","hybrid","hyde","corrective","graph","agentic"] as const;
type Arm = (typeof ARMS)[number];
const CLASSES = ["on","partial","off","noisy","bridge"] as const;
const MATRIX: Record<string, { local: number; global: number; classes: Record<string, number>; ops: number; calls: number; ops_global: number }> = {"naive":{"local":0.29,"global":0.062,"classes":{"on":1.0,"partial":0.35,"off":0.1,"noisy":0.0,"bridge":0.0},"ops":100,"calls":0.0,"ops_global":25},"hybrid":{"local":0.5,"global":0.062,"classes":{"on":0.8,"partial":1.0,"off":0.4,"noisy":0.3,"bridge":0.0},"ops":300,"calls":0.0,"ops_global":50},"hyde":{"local":0.38,"global":0.0,"classes":{"on":0.9,"partial":0.2,"off":0.8,"noisy":0.0,"bridge":0.0},"ops":100,"calls":4.0,"ops_global":25},"corrective":{"local":0.55,"global":0.688,"classes":{"on":1.0,"partial":1.0,"off":0.35,"noisy":0.4,"bridge":0.0},"ops":1040,"calls":1.0,"ops_global":45},"graph":{"local":0.29,"global":1.0,"classes":{"on":0.8,"partial":0.45,"off":0.2,"noisy":0.0,"bridge":0.0},"ops":10,"calls":0.0,"ops_global":25},"agentic":{"local":0.28,"global":0.062,"classes":{"on":0.15,"partial":0.15,"off":0.1,"noisy":0.15,"bridge":0.85},"ops":288,"calls":2.88,"ops_global":75}};
const GOLD_RANKS: Record<string, number[]> = {"on":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"partial":[0,2,2,0,0,6,1,1,8,3,7,2,3,4,0,0,0,4,0,3],"off":[15,10,1,2,15,2,4,12,5,0,3,1,0,3,21,2,22,1,4,8],"noisy":[5,16,8,6,16,4,12,16,4,16,16,16,16,4,12,16,5,16,5,7],"bridge":[7,79,57,31,18,35,6,62,10,8,7,64,68,13,28,56,10,64,11,8]};
const GRADES: Record<string, number> = {"on":0.71,"partial":0.25,"off":0.02,"noisy":0.013,"bridge":0.714};
const LEGS_PARTIAL: Record<string, number> = {"lexical":0.05,"dense":0.35,"late":0.05};
const WORKED = {"query":21,"gold":4,"legs":{"lexical":[80,15,37,47,13,38,45,55],"dense":[6,7,12,8,11,14,23,21],"late":[91,48,44,13,20,27,19,12]},"owners":{"lexical":[5,5,12,15,4,12,15,18],"dense":[2,2,4,2,3,4,7,7],"late":[16,16,14,4,6,9,6,4]},"theme_of_gold":[2,4,10]};
const DAMAGE: Record<string, number[]> = {"0.8|0.0":[0.55,1.0,0.53],"0.8|1.0":[0.52,1.0,0.53],"0.8|2.0":[0.49,1.0,0.53],"0.8|3.0":[0.46,1.0,0.53],"0.84|0.0":[0.55,1.0,0.65],"0.84|1.0":[0.52,1.0,0.65],"0.84|2.0":[0.48,1.0,0.65],"0.84|3.0":[0.45,1.0,0.65],"0.88|0.0":[0.55,1.0,1.0],"0.88|1.0":[0.51,0.95,1.0],"0.88|2.0":[0.45,0.9,1.0],"0.88|3.0":[0.4,0.8,1.0]};
const MEAN_HOPS: Record<string, number> = {"on":2.7,"partial":3.0,"off":3.0,"noisy":3.0,"bridge":2.7};
const G_ORDER: number[][] = [[1,0,2,6,3,4,10,5,8,9,7,24,23,20,11,13,21,22,12,14,18,16,19,17,15],[8,6,5,7,9,1,2,4,0,12,10,19,13,14,3,11,24,17,23,22,21,18,16,20,15],[10,12,14,3,4,11,13,20,2,22,0,1,24,9,5,23,7,6,19,21,15,17,18,8,16],[15,16,18,19,1,17,0,2,3,4,9,24,13,12,14,21,23,20,8,22,10,6,5,11,7],[20,21,24,22,17,15,23,19,16,18,12,2,7,4,5,9,13,14,11,1,10,6,0,3,8],[2,0,3,14,1,4,11,13,7,10,12,5,8,9,6,23,24,17,19,21,20,22,15,16,18],[7,5,9,6,8,22,23,24,20,12,17,10,21,18,13,19,4,14,16,11,1,3,15,2,0],[10,14,7,13,11,5,6,12,8,9,18,17,16,15,3,19,23,0,20,24,1,2,4,22,21],[20,22,15,18,16,17,24,19,21,14,23,12,8,11,4,3,13,9,10,1,0,7,2,6,5],[22,23,20,24,21,3,4,1,0,12,2,10,19,16,15,18,11,14,17,13,9,5,6,7,8],[3,1,2,4,0,11,9,24,10,23,19,13,20,21,5,16,15,12,14,6,22,18,7,17,8],[5,7,6,9,8,0,12,4,14,11,2,22,10,1,21,13,3,24,23,19,20,17,16,18,15],[12,13,11,10,14,20,22,24,23,2,21,15,17,3,4,18,5,1,19,0,16,6,7,9,8],[18,19,17,16,15,9,6,8,5,7,13,24,21,20,22,12,23,14,0,10,2,4,11,1,3],[23,20,22,21,24,3,8,18,1,2,16,17,4,15,6,0,7,5,9,14,19,13,10,12,11],[3,1,4,0,2,20,23,24,21,15,16,19,17,18,10,5,11,9,22,6,14,7,13,8,12]];
const G_MENTION: number[][] = [[1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0,1],[0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,1],[0,0,0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0,1,1,1],[1,1,1,1,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1],[0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0],[1,0,1,0,0,0,0,1,0,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0],[0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1,0,0],[0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,1,1],[1,1,0,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0,0],[0,0,0,1,0,1,1,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0],[0,1,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1],[1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1,0,0],[0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,1,1,1],[1,1,1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,1,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0],[0,1,0,1,0,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,1,0,0,0,0]];
const G_SECTOR: number[] = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4];
const G_GOLD: number[] = [2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2];
const N_DOCS = 100, N_QUERIES = 100, N_COMPANIES = 25;
const N_SECTORS = 5, N_GLOBAL = 16, DIM = 64;
const OVER_FETCH = 20, GRADER_AUC = 0.829;
const RESID_OWN = 0.359, RESID_BRIDGE = 0.588, REFORM_EPS = 0.42;

const COND = [...CLASSES, 'global'] as const;
const CLASS_BLURB: Record<string, string> = {
  on: 'one view identifies the company outright',
  partial: 'every view returns rivals; only the conjunction is unique',
  off: 'the query is tilted off the document manifold',
  noisy: 'each view is dragged toward a different wrong peer',
  bridge: 'the answer is a company a filing NAMES rather than describes',
  global: 'the answer is a COUNT across a community, carried by no passage',
};
const MECHANISM: Record<string, string> = {
  naive: 'One retrieval, one answer. Unimprovable where a single view already carries the answer, and the thing every other arm has to justify itself against.',
  hybrid: 'Three partial views fused by reciprocal rank. Fusion is set intersection: it can only add what the views disagree about.',
  hyde: 'Discard the query position, generate a document that looks like the answer, retrieve with that. Corrects distribution shift; inherits the generator.',
  corrective: 'Grade the cheap retrieval; on a bad grade look deeper and score harder. Moves the answer up a ranking it is already on.',
  graph: 'Detect communities once, offline, and answer from what a community aggregates. The only arm holding a representation of a SET.',
  agentic: 'Retrieve, read, reformulate, retrieve again. The only arm that reaches an answer no passage is about, and it pays on every query.',
};

const fmt = (x: number, n = 3) => x.toFixed(n);
const signed = (x: number, n = 3) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(n);
const pill = (active: boolean) => ({
  fontFamily: 'var(--font-sans)', fontSize: '0.78rem', padding: '0.3rem 0.7rem', borderRadius: '999px', cursor: 'pointer',
  border: `1px solid ${active ? 'var(--color-accent)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-accent)' : 'transparent', color: active ? 'var(--color-bg)' : 'var(--color-text)',
});

function Readout({ label, value, accent, warn }: { label: string; value: string; accent?: boolean; warn?: boolean }) {
  return (
    <div>
      <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.7rem', marginBottom: '0.15rem' }}>{label}</div>
      <div style={{ fontSize: '1.05rem', fontWeight: 600, color: warn ? 'var(--color-badge-red-text)' : accent ? 'var(--color-accent)' : 'var(--color-text)' }}>{value}</div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display: string;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontFamily: 'var(--font-sans)', fontSize: '0.82rem', margin: '0.2rem 0 0.6rem' }}>
      <span style={{ minWidth: '17rem' }}>{label} = <strong>{display}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
    </label>
  );
}

const Row = ({ children }: { children: ReactNode }) => (
  <div style={{ display: 'flex', gap: '1.4rem', flexWrap: 'wrap', margin: '0.6rem 0 0.2rem' }}>{children}</div>
);
const Note = ({ children }: { children: ReactNode }) => (
  <p style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: '0.6rem 0 0' }}>{children}</p>
);

/** Reciprocal rank fusion, the imported rule: score(d) = sum over legs of 1/(c + rank). */
function rrfFuse(lists: number[][], c: number): number[] {
  const score = new Map<number, number>();
  for (const list of lists) {
    list.forEach((d, r) => score.set(d, (score.get(d) ?? 0) + 1 / (c + r + 1)));
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(([d]) => d);
}

// ===== Panel A — the matrix ======================================================================
function MatrixPanel({ arm, setArm, delta, setDelta }: {
  arm: Arm; setArm: (a: Arm) => void; delta: boolean; setDelta: (b: boolean) => void;
}) {
  const W = 600, H = 250, padL = 96, padT = 46, cellH = (H - padT - 14) / ARMS.length;
  const cellW = (W - padL - 10) / COND.length;
  const valueOf = (a: string, k: string) => (k === 'global' ? MATRIX[a].global : MATRIX[a].classes[k]);
  const base = (k: string) => valueOf('naive', k);

  return (
    <div>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.4rem' }}>
        <button type="button" style={pill(!delta)} onClick={() => setDelta(false)}>accuracy</button>
        <button type="button" style={pill(delta)} onClick={() => setDelta(true)}>change against naive</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Accuracy of six RAG architectures across five local query classes and the global regime">
        {COND.map((k, ci) => (
          <text key={k} x={padL + ci * cellW + cellW / 2} y={padT - 22} textAnchor="middle"
            fontSize="10.5" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">{k}</text>
        ))}
        <text x={padL + COND.length * cellW / 2} y={14} textAnchor="middle" fontSize="10" fill="var(--color-text-secondary)"
          fontFamily="var(--font-sans)">— the answer is carried by a passage —{'   '}| aggregate</text>
        {ARMS.map((a, ri) => (
          <g key={a}>
            <text x={padL - 8} y={padT + ri * cellH + cellH / 2 + 4} textAnchor="end" fontSize="11"
              fontFamily="var(--font-sans)" style={{ cursor: 'pointer' }} onClick={() => setArm(a)}
              fill={a === arm ? 'var(--color-accent)' : 'var(--color-text)'}
              fontWeight={a === arm ? 700 : 400}>{a}</text>
            {COND.map((k, ci) => {
              const v = valueOf(a, k);
              const d = v - base(k);
              const shown = delta ? d : v;
              const mag = delta ? Math.min(1, Math.abs(d) / 0.9) : v;
              const good = !delta || d >= 0;
              return (
                <g key={k}>
                  <rect x={padL + ci * cellW + 1} y={padT + ri * cellH + 1} width={cellW - 2} height={cellH - 2}
                    rx="3" fill={good ? 'var(--color-accent)' : 'var(--color-badge-red-text)'}
                    fillOpacity={delta && Math.abs(d) < 1e-9 ? 0.05 : 0.1 + 0.75 * mag}
                    stroke={a === arm ? 'var(--color-accent)' : 'none'} strokeWidth={a === arm ? 1.4 : 0} />
                  <text x={padL + ci * cellW + cellW / 2} y={padT + ri * cellH + cellH / 2 + 4}
                    textAnchor="middle" fontSize="10.5" fontFamily="var(--font-mono, monospace)"
                    fill={mag > 0.55 ? 'var(--color-bg)' : 'var(--color-text)'}>
                    {delta ? (Math.abs(d) < 1e-9 ? '·' : signed(shown, 2)) : fmt(shown, 2)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <Row>
        <Readout label="arm" value={arm} accent />
        <Readout label="best local class" value={(() => {
          const cs = CLASSES.map((k) => [k, MATRIX[arm].classes[k]] as const);
          return cs.reduce((m, x) => (x[1] > m[1] ? x : m))[0];
        })()} />
        <Readout label="local overall" value={fmt(MATRIX[arm].local, 2)} />
        <Readout label="global" value={fmt(MATRIX[arm].global, 2)}
          accent={MATRIX[arm].global === 1} />
        <Readout label="ops / query" value={String(MATRIX[arm].ops)} />
        <Readout label="generator calls" value={fmt(MATRIX[arm].calls, 2)}
          warn={MATRIX[arm].calls > 0} />
      </Row>
      <Note><strong>{arm}.</strong> {MECHANISM[arm]}{' '}
        {arm === 'agentic' && `It hops ${MEAN_HOPS.on} times on average even on the class naive answers perfectly, because the test for "this filing points somewhere new" (mean residual ${RESID_BRIDGE} against ${RESID_OWN} for an ordinary filing, threshold ${REFORM_EPS}) cannot tell a genuine mention from an off-target retrieval.`}
      </Note>
      <dl style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '0.15rem 0.6rem', margin: '0.7rem 0 0', fontSize: '0.74rem', lineHeight: 1.45 }}>
        {COND.map((k) => (
          <div key={k} style={{ display: 'contents' }}>
            <dt style={{ fontFamily: 'var(--font-mono, monospace)', color: 'var(--color-accent)' }}>{k}</dt>
            <dd style={{ margin: 0, color: 'var(--color-text-secondary)' }}>{CLASS_BLURB[k]}</dd>
          </div>
        ))}
      </dl>
      <Note>
        No row wins both columns. The arm that takes the global regime outright is mid-table locally;
        the arm that owns the bridge class is the <em>worst</em> arm locally overall. Every flat arm
        scores at or below {fmt(1 / N_SECTORS, 2)} globally — worse than guessing among {N_SECTORS} sectors,
        because the corpus is built so that the sector nearest a theme is never the sector that discusses it most.
      </Note>
    </div>
  );
}

// ===== Panel B — fusion as set intersection ======================================================
function FusionPanel({ c, setC }: { c: number; setC: (v: number) => void }) {
  const legNames = Object.keys(WORKED.legs) as (keyof typeof WORKED.legs)[];
  const fused = useMemo(() => rrfFuse(legNames.map((n) => WORKED.legs[n]), c), [c, legNames]);
  const ownerOf = useMemo(() => {
    const m = new Map<number, number>();
    legNames.forEach((n) => WORKED.legs[n].forEach((d, i) => m.set(d, WORKED.owners[n][i])));
    return m;
  }, [legNames]);
  const topOwner = ownerOf.get(fused[0]) ?? -1;
  const hit = topOwner === WORKED.gold;
  const W = 600, H = 210, colW = 128, padT = 34, rowH = 20;

  return (
    <div>
      <Slider label="RRF constant c" value={c} min={0} max={120} step={1} onChange={setC} display={String(c)} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Three retrieval legs' shortlists for one conjunctive query, and their reciprocal-rank fusion">
        {legNames.map((n, li) => (
          <g key={n}>
            <text x={li * colW + 8} y={padT - 12} fontSize="10.5" fill="var(--color-text-secondary)"
              fontFamily="var(--font-sans)">{n} · {fmt(LEGS_PARTIAL[n], 2)}</text>
            {WORKED.legs[n].slice(0, 8).map((d, r) => {
              const own = WORKED.owners[n][r];
              const isGold = own === WORKED.gold;
              return (
                <g key={d}>
                  <rect x={li * colW + 8} y={padT + r * rowH} width={colW - 20} height={rowH - 3} rx="2"
                    fill={isGold ? 'var(--color-accent)' : 'var(--color-border)'} fillOpacity={isGold ? 0.85 : 0.3} />
                  <text x={li * colW + 14} y={padT + r * rowH + 12} fontSize="9.5"
                    fontFamily="var(--font-mono, monospace)"
                    fill={isGold ? 'var(--color-bg)' : 'var(--color-text-secondary)'}>
                    {r + 1}. company {own}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
        <line x1={3 * colW + 2} y1={padT - 18} x2={3 * colW + 2} y2={H - 6} stroke="var(--color-border)" />
        <text x={3 * colW + 14} y={padT - 12} fontSize="10.5" fill="var(--color-accent)"
          fontFamily="var(--font-sans)">fused (c = {c})</text>
        {fused.slice(0, 8).map((d, r) => {
          const own = ownerOf.get(d) ?? -1;
          const isGold = own === WORKED.gold;
          return (
            <g key={d}>
              <rect x={3 * colW + 14} y={padT + r * rowH} width={colW - 20} height={rowH - 3} rx="2"
                fill={isGold ? 'var(--color-accent)' : 'var(--color-border)'} fillOpacity={isGold ? 0.85 : 0.3} />
              <text x={3 * colW + 20} y={padT + r * rowH + 12} fontSize="9.5"
                fontFamily="var(--font-mono, monospace)"
                fill={isGold ? 'var(--color-bg)' : 'var(--color-text-secondary)'}>
                {r + 1}. company {own}
              </text>
            </g>
          );
        })}
      </svg>
      <Row>
        <Readout label="gold company" value={`#${WORKED.gold}`} accent />
        <Readout label="its three themes" value={WORKED.theme_of_gold.join(' · ')} />
        <Readout label="best single leg" value={fmt(Math.max(...Object.values(LEGS_PARTIAL)), 2)} />
        <Readout label="fused, on this class" value="1.00" accent />
        <Readout label="fusion picks" value={hit ? `company ${topOwner} ✓` : `company ${topOwner} ✗`}
          warn={!hit} />
      </Row>
      <Note>
        No leg puts the gold first; every leg puts it somewhere. A leg alone retrieves everything sharing
        the one theme it can see, so it returns rivals — and a different set of rivals each time. The
        company on <em>all three</em> shortlists is the one holding all three themes, which is the answer.
        That is the whole of what fusion buys, and it is why the same arm is a liability on the class one
        view already answers: there, two uninformed legs outvote the informed one and reciprocal rank has
        no way to know which was which.
      </Note>
    </div>
  );
}

// ===== Panel C — rank against reach ==============================================================
function ReachPanel({ depth, setDepth }: { depth: number; setDepth: (v: number) => void }) {
  const W = 600, H = 190, padL = 76, padT = 20, rowH = (H - padT - 26) / CLASSES.length;
  const xOf = (r: number) => padL + (Math.min(r, N_DOCS) / N_DOCS) * (W - padL - 16);
  const recoverable = (k: string) => GOLD_RANKS[k].filter((r) => r < depth).length / GOLD_RANKS[k].length;

  return (
    <div>
      <Slider label="over-fetch depth (documents re-scored)" value={depth} min={1} max={N_DOCS} step={1}
        onChange={setDepth} display={String(depth)} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Position of the correct company in the cheap ranking, by query class, against over-fetch depth">
        <rect x={padL} y={padT - 6} width={xOf(depth) - padL} height={H - padT - 18} fill="var(--color-accent)" fillOpacity={0.08} />
        <line x1={xOf(depth)} y1={padT - 6} x2={xOf(depth)} y2={H - 24} stroke="var(--color-accent)" strokeWidth="1.4" />
        {CLASSES.map((k, ri) => (
          <g key={k}>
            <text x={padL - 8} y={padT + ri * rowH + rowH / 2 + 4} textAnchor="end" fontSize="10.5"
              fontFamily="var(--font-sans)" fill="var(--color-text)">{k}</text>
            <line x1={padL} y1={padT + ri * rowH + rowH / 2} x2={W - 16} y2={padT + ri * rowH + rowH / 2}
              stroke="var(--color-border)" strokeWidth="0.6" />
            {GOLD_RANKS[k].map((r, i) => (
              <circle key={i} cx={xOf(r)} cy={padT + ri * rowH + rowH / 2} r="3.4"
                fill={r < depth ? 'var(--color-accent)' : 'var(--color-badge-red-text)'} fillOpacity={0.75} />
            ))}
            <text x={W - 12} y={padT + ri * rowH + rowH / 2 + 4} textAnchor="end" fontSize="9.5"
              fontFamily="var(--font-mono, monospace)" fill="var(--color-text-secondary)">
              {fmt(recoverable(k), 2)}
            </text>
          </g>
        ))}
        <text x={padL} y={H - 8} fontSize="9.5" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">rank 0</text>
        <text x={W - 16} y={H - 8} textAnchor="end" fontSize="9.5" fill="var(--color-text-secondary)"
          fontFamily="var(--font-sans)">rank {N_DOCS}</text>
      </svg>
      <Row>
        <Readout label="noisy, within reach" value={fmt(recoverable('noisy'), 2)} accent />
        <Readout label="bridge, within reach" value={fmt(recoverable('bridge'), 2)}
          warn={recoverable('bridge') < 0.5} />
        <Readout label="grade, on class" value={fmt(GRADES.on, 2)} />
        <Readout label="grade, bridge class" value={fmt(GRADES.bridge, 2)} warn />
      </Row>
      <Note>
        Correction moves an answer up a ranking it is already on. At the operating depth of {OVER_FETCH} the
        whole noisy class is within reach; the bridge class is not, and never becomes so for a reason no depth
        fixes — no passage is <em>about</em> the answer. The grader cannot warn you either: it scores relevance,
        and on the bridge class the retrieved filing is genuinely relevant (it is the filing of the company the
        query describes), grading {fmt(GRADES.bridge, 2)} against {fmt(GRADES.on, 2)} for the class answered
        perfectly. Measured AUC against "was the cheap answer right" is {GRADER_AUC} — informative, and blind here.
      </Note>
      <Note>
        <strong>What the average hides.</strong> Correcting is never a net loss on this corpus: with the grader
        firing on every query and a corrective scorer at &sigma; = 3 the arm still scores{' '}
        {fmt(DAMAGE['0.88|3.0'][0], 2)} against the baseline's {fmt(MATRIX.naive.local, 2)}. But the class that
        needed no correcting falls from {fmt(DAMAGE['0.88|0.0'][1], 2)} to {fmt(DAMAGE['0.88|3.0'][1], 2)} as
        &sigma; runs 0 &rarr; 3. The mean rises while a perfect column is destroyed.
      </Note>
    </div>
  );
}

// ===== Panel D — aggregation, precomputed or by hand =============================================
/** THE live recompute: the flat-aggregation curve, exactly as the module measures it. */
function flatAccuracyAt(depth: number): number {
  let hits = 0;
  for (let q = 0; q < G_GOLD.length; q++) {
    const order = G_ORDER[q], ment = G_MENTION[q];
    const keep = order.slice(0, depth).filter((e) => ment[e] === 1);
    const pool = keep.length ? keep : order.slice(0, 1);
    const votes = new Array(N_SECTORS).fill(0);
    for (const e of pool) votes[G_SECTOR[e]] += 1;
    let best = 0;
    for (let s = 1; s < N_SECTORS; s++) if (votes[s] > votes[best]) best = s;
    if (best === G_GOLD[q]) hits += 1;
  }
  return hits / G_GOLD.length;
}

function AggregationPanel({ depth, setDepth }: { depth: number; setDepth: (v: number) => void }) {
  const W = 600, H = 220, padL = 46, padR = 122, padT = 16, padB = 34;
  const curve = useMemo(
    () => Array.from({ length: N_COMPANIES }, (_, i) => [i + 1, flatAccuracyAt(i + 1)] as const), []);
  const xOf = (d: number) => padL + ((d - 1) / (N_COMPANIES - 1)) * (W - padL - padR);
  const yOf = (a: number) => H - padB - a * (H - padT - padB);
  const here = flatAccuracyAt(depth);
  const path = curve.map(([d, a], i) => `${i ? 'L' : 'M'}${xOf(d)},${yOf(a)}`).join(' ');

  return (
    <div>
      <Slider label="how far down the flat ranking you look" value={depth} min={1} max={N_COMPANIES} step={1}
        onChange={setDepth} display={`${depth} of ${N_COMPANIES} entities`} />
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label="Accuracy of hand aggregation over a flat ranking against depth, compared with precomputed community summaries">
        {[0, 0.25, 0.5, 0.75, 1].map((a) => (
          <g key={a}>
            <line x1={padL} y1={yOf(a)} x2={W - padR} y2={yOf(a)} stroke="var(--color-border)" strokeWidth="0.5" />
            <text x={padL - 6} y={yOf(a) + 3} textAnchor="end" fontSize="9" fill="var(--color-text-secondary)">{a}</text>
          </g>
        ))}
        <line x1={padL} y1={yOf(1)} x2={W - padR} y2={yOf(1)} stroke="var(--color-accent)" strokeWidth="1.6" strokeDasharray="5 3" />
        <text x={W - padR + 6} y={yOf(1) + 3} fontSize="9.5" fill="var(--color-accent)" fontFamily="var(--font-sans)">
          graph · 1.00 at depth 0
        </text>
        <path d={path} fill="none" stroke="var(--color-text)" strokeWidth="1.8" />
        <line x1={xOf(depth)} y1={padT} x2={xOf(depth)} y2={H - padB} stroke="var(--color-accent)" strokeWidth="1.2" />
        <circle cx={xOf(depth)} cy={yOf(here)} r="4.5" fill="var(--color-accent)" />
        {/* When hand aggregation reaches 1.00 it sits on the graph line; drop its label clear. */}
        <text x={W - padR + 6} y={yOf(here) + (Math.abs(here - 1) < 0.06 ? 16 : 3)} fontSize="9.5"
          fill="var(--color-text)" fontFamily="var(--font-sans)">
          by hand · {fmt(here, 2)}
        </text>
        <text x={padL} y={H - 10} fontSize="9.5" fill="var(--color-text-secondary)" fontFamily="var(--font-sans)">
          entities scanned, then filtered to those that discuss the theme
        </text>
      </svg>
      <Row>
        <Readout label="depth" value={String(depth)} />
        <Readout label="flat aggregation" value={fmt(here, 3)} warn={here < 1} />
        <Readout label="community summaries" value="1.000" accent />
        <Readout label="themes" value={String(N_GLOBAL)} />
      </Row>
      <Note>
        The honest form of the global result: flat retrieval <em>can</em> reach this answer. It gets there by
        scanning far enough down the ranking, keeping the entities that actually discuss the theme, and letting
        those vote — which is the aggregation a community summary holds, performed by hand at query time. On
        this corpus it needs the entire entity set. GraphRAG answers at depth zero because it paid for that
        aggregation once, offline. The claim is a cost, not an impossibility.
      </Note>
    </div>
  );
}

type Panel = 'matrix' | 'fusion' | 'reach' | 'aggregate';
const TEX: Record<Panel, string> = {
  matrix: '\\text{no arm } a \\text{ maximizes both } \\ \\mathbb{E}_{\\text{local}}[\\mathbb{1}\\{\\hat y_a = y\\}] \\ \\text{ and } \\ \\mathbb{E}_{\\text{global}}[\\mathbb{1}\\{\\hat y_a = y\\}]',
  fusion: '\\mathrm{RRF}(d) = \\sum_{\\ell=1}^{3} \\frac{1}{c + r_\\ell(d)}, \\qquad \\{\\text{answer}\\} = \\bigcap_{\\ell} \\{a : \\theta_\\ell(a) = \\theta_\\ell(q)\\}',
  reach: '\\Pr[\\text{recoverable at depth } k] = \\Pr[\\mathrm{rank}(y) < k], \\qquad \\text{grade} = \\sigma\\!\\big(\\kappa(\\mathrm{MaxSim}(q, d_1)/T - m)\\big)',
  aggregate: '\\hat s = \\arg\\max_{s} \\sum_{e \\in \\mathcal{C}_s} \\mathbb{1}\\{e \\text{ mentions } \\theta\\} \\ \\neq \\ \\arg\\max_{s} \\ \\langle \\mu_s, \\theta \\rangle',
};

export default memo(function RagArchitectureMechanismsLaboratory() {
  const [panel, setPanel] = useState<Panel>('matrix');
  const [arm, setArm] = useState<Arm>('naive');
  const [delta, setDelta] = useState(false);
  const [c, setC] = useState(60);
  const [depth, setDepth] = useState(OVER_FETCH);
  const [gdepth, setGdepth] = useState(10);
  const formulaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!formulaRef.current) return;
    katex.render(TEX[panel], formulaRef.current, { throwOnError: false, displayMode: true });
  }, [panel]);

  return (
    <div data-lab="rag-architecture-mechanisms" style={{ border: '1px solid var(--color-border)', borderRadius: '0.6rem', padding: '1rem 1.1rem', margin: '1.4rem 0', background: 'var(--color-bg)' }}>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        <button type="button" style={pill(panel === 'matrix')} onClick={() => setPanel('matrix')}>A · the matrix</button>
        <button type="button" style={pill(panel === 'fusion')} onClick={() => setPanel('fusion')}>B · fusion as intersection</button>
        <button type="button" style={pill(panel === 'reach')} onClick={() => setPanel('reach')}>C · rank against reach</button>
        <button type="button" style={pill(panel === 'aggregate')} onClick={() => setPanel('aggregate')}>D · aggregation</button>
      </div>
      <div ref={formulaRef} style={{ margin: '0.4rem 0 0.8rem', minHeight: '2.2rem' }} />
      {panel === 'matrix' && <MatrixPanel arm={arm} setArm={setArm} delta={delta} setDelta={setDelta} />}
      {panel === 'fusion' && <FusionPanel c={c} setC={setC} />}
      {panel === 'reach' && <ReachPanel depth={depth} setDepth={setDepth} />}
      {panel === 'aggregate' && <AggregationPanel depth={gdepth} setDepth={setGdepth} />}
      <p style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.7rem', lineHeight: 1.45 }}>
        {N_QUERIES} synthetic finance queries in five classes over {N_DOCS} passages of {N_COMPANIES} companies,
        and {N_GLOBAL} themes over the same {N_COMPANIES} entities in {N_SECTORS} sectors, at dimension {DIM}.
        Every arm is the same retrieval geometry read differently, so an accuracy difference is attributable to
        the architecture alone. Numbers are emitted from <code>rag_architecture_mechanisms.py</code> rather than
        transcribed; panels B, C and D recompute their closed forms live, and panel D reproduces the module's
        flat-aggregation curve exactly. The corpus, the query classes and the mention angle were all chosen
        against this corpus — what transfers is the shape of the argument, not these decimals.
      </p>
    </div>
  );
});
