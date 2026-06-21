# formalRAG topic brief — HNSW: Hierarchical Navigable Small-World Construction and Search

**Slug:** `hnsw` · **Domain:** `ann-indexing` · **Pipeline stage:** `retrieve` · **Difficulty:** advanced
**Prerequisites:** `navigable-small-world-graphs`, `ivf-voronoi-partitioning`
**Owns the numbers:** `notebooks/hnsw/hnsw.py` → `viz_constants()` → `HNSWLaboratory.tsx`

## Thesis

NSW gave us a single navigable graph and a beam search that walks it; its honest catch is that the
walk starts somewhere arbitrary and the small-world property is *empirical*. HNSW adds **one
structural idea — a randomized hierarchy of nested graphs** — that turns the arbitrary entry into a
**provably logarithmic descent**. The topic's spine is therefore a clean probability theorem (the
level-assignment law), wrapped around the prerequisite's beam search reused verbatim per layer. The
distinguishing *engineering* idea — the heuristic neighbor selection that keeps the graph navigable —
is honestly flagged as a heuristic with no optimality proof. The arc closes with a head-to-head
against the inverted file (IVF) on one shared cloud: graph index vs partition index, recall vs cost.

## Movement 1 — The hierarchy and the level-assignment theorem (the provable spine)

Each inserted node draws a maximum level

$$L=\big\lfloor -\ln(U)\cdot m_L\big\rfloor,\qquad U\sim\mathrm{Uniform}(0,1),\quad m_L=\frac{1}{\ln M},$$

and is inserted into the graphs at every layer $0,1,\dots,L$. Layer 0 holds every node; each higher
layer holds a geometrically thinning random subset. This is the continuous analogue of a skip list.

**Theorem (level distribution).** With $m_L=1/\ln M$:
- **(i) Survival law.** $P(L\ge \ell)=M^{-\ell}$ for integer $\ell\ge 0$.
  *Proof.* $L\ge\ell \iff -\ln(U)m_L\ge\ell \iff U\le e^{-\ell/m_L}=M^{-\ell}$; uniformity gives the
  probability $M^{-\ell}$. ∎ The per-level mass is $P(L=\ell)=M^{-\ell}(1-1/M)$ — geometric, ratio $1/M$.
- **(ii) Geometric occupancy, O(1) top, O(log_M n) height.** Expected nodes reaching layer $\ell$ is
  $nM^{-\ell}$; the top non-empty layer has expected occupancy $\to1$, and
  $\mathbb{E}[\max_i L_i]\approx \log_M n + O(1)$. The greedy entry-descent visits $\approx\log_M n$ layers.
- **(iii) Bounded per-layer degree.** Construction caps neighbors at `Mmax = M` per upper layer and
  `Mmax0 = 2M` at layer 0, so degree is $O(M)$ independent of $n$.

Together: descent height $O(\log_M n)$ × bounded per-layer work = the $O(\log n)$ *intuition*. We
**prove and measure** the level law and the height scaling; we are explicit that the end-to-end search
cost is empirical (Movement 3 / rigorFlag).

## Movement 2 — Heuristic neighbor selection (the pedagogical centerpiece)

The difference between HNSW and "link to the M nearest" is **Algorithm 4 (Malkov–Yashunin)**: scanning
candidates by increasing distance to the base point, admit a candidate $c$ **only if no
already-kept neighbor $r$ is closer to $c$ than $c$ is to the base** ($d(c,r)\ge d(c,\text{base})$ for
all kept $r$). Strict-M-nearest clusters all links on one side of a point; the heuristic spreads them
across directions, **preserving the long-range links** that make the graph a small world. No
optimality proof — a well-motivated diversity rule. Panel B visualizes naive vs heuristic side by side.

## Movement 3 — Hierarchical search and the IVF head-to-head

**Search (Algorithm 5):** from the single top entry point, greedy `ef=1` descent through the upper
layers refines the entry; at layer 0 a width-`ef` beam (the prerequisite's `greedy_search`, restricted
to a per-layer adjacency) returns the top-k. Total cost = distance computations summed over layers.

**Provably-one-cloud comparison.** Build HNSW, flat-NSW, **and** IVF on the *same* `nsw_dataset`
`(X, queries)`; re-derive one shared `truth`; sweep `efSearch` (HNSW/NSW) and `nprobe` (IVF) to trace
recall-vs-cost frontiers, cost = mean exact distance computations/query (IVF cost =
`candidate_fraction·n + nlist`). The **robust headline** is intra-graph: *HNSW reaches a given recall
at ≤ the cost of flat NSW* (the hierarchy's whole point). The HNSW-vs-IVF verdict is stated honestly
as a property of this one synthetic cloud, never a universal ranking.

## Viz intent — `HNSWLaboratory.tsx` (3 panels, constants mirrored to the decimal)

- **A. The layer pyramid.** A small 2-D multi-layer graph (`toy_hnsw_graph`): sparse top → dense
  bottom; one query's `ef=1` descent path animated top-down.
- **B. Naive vs heuristic neighbors.** One base node; toggle between its M strict-nearest and the
  diversity-pruned set; show the preserved long-range link.
- **C. Recall vs cost.** Three frontiers (HNSW / flat-NSW / IVF) + entry-descent-depth-vs-$\log_M n$
  scaling points.

## rigorFlag (NSW register: one clean theorem, then honest caveats)

The level law is exact probability (survival $M^{-\ell}$, geometric occupancy, $\log_M n$ height —
verified). Caveats, each asserted only as a *direction*: (1) the end-to-end $O(\log n)$ search rests on
each layer being navigable, which on real embeddings is empirical, inherited from the NSW heuristic,
not a theorem; (2) `select_neighbors_heuristic` has no optimality proof; (3) recall is empirical and
depends on $M$, `efSearch`, $m_L$, and the entry point; (4) the IVF head-to-head is one synthetic
cloud with a shared ground truth — a statement about this cloud, not a universal ranking.

## References (DOIs verified via redirect)

- Malkov & Yashunin (2020), *Efficient and Robust ANN Search Using HNSW Graphs*, IEEE TPAMI —
  `10.1109/TPAMI.2018.2889473` → IEEE doc 8594636. **Load-bearing:** the algorithm and its heuristics.
- Malkov, Ponomarenko, Logvinov & Krylov (2014), *ANN Algorithm Based on NSW Graphs*, Information
  Systems — `10.1016/j.is.2013.10.006`. The flat NSW precursor HNSW layers.
- Kleinberg (2000), *The Small-World Phenomenon: An Algorithmic Perspective*, STOC —
  `10.1145/335305.335325`. The navigability theorem behind why a single layer is searchable.
- Pugh (1990), *Skip Lists: A Probabilistic Alternative to Balanced Trees*, CACM —
  `10.1145/78973.78977`. The randomized-level data structure HNSW generalizes to a metric space.

## Cross-site (UP into formalML; mirror NSW, verify slugs exist)

- `formalmlPrereqs`: `expander-graphs` (logarithmic-diameter sparse graphs underlie the hierarchy).
- `formalmlConnections`: `random-walks` (descent as a walk; hitting/mixing time), `concentration-inequalities`
  (the max-of-n-geometrics height concentrates at $\log_M n$).
