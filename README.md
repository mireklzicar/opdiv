# OPDiv

[![PyPI](https://img.shields.io/pypi/v/opdiv.svg)](https://pypi.org/project/opdiv/)

Molecular portfolio selection and diversity evaluation.

Choose exactly **k** candidates with the highest mean score while enforcing a
pairwise similarity ceiling. Use the resulting mean to evaluate a screening or
generative method's candidate archive.

Implements **Optimal Portfolio Diversity (OPDiv)**, its score-ordered greedy
approximation **GPDiv**, from
*Measuring Diversity of Top-K Molecules as an Optimal Portfolio Selection Problem*
by Miroslav Lžičař (preprint draft, September 2026).

## Install

Python 3.10 or newer:

```bash
pip install opdiv
```

For development, from this repository:

```bash
pip install -e '.[chem,dev]'
```

The core requires NumPy and OR-Tools. The optional `chem` extra adds RDKit for
building Morgan/Tanimoto similarities from SMILES.

## Select a portfolio

```python
from opdiv import select, opdiv, gpdiv

# The paper's counterexample: A conflicts with B and C; B and C are compatible.
scores = [16.0, 12.0, 11.0, 6.0]  # A, B, C, D; larger is better
conflicts = [
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
]

result = select(scores, k=2, conflicts=conflicts)
print(result.indices)  # (1, 2): B and C
print(result.value)    # 11.5
print(result.status)   # optimal

# Metric-only calls also perform selection internally.
print(opdiv(scores, k=2, conflicts=conflicts))  # 11.5
print(gpdiv(scores, k=2, conflicts=conflicts))  # 11.0: A and D

greedy = select(scores, k=2, conflicts=conflicts, method="greedy")
```

`indices` are zero-based input positions, sorted by descending score, then input
position. Index a Python list with `[items[i] for i in result.indices]`, or a
DataFrame with `df.iloc[list(result.indices)]`. Scores are **utilities**: for
lower-is-better docking energies, pass their negatives.

## From molecules or custom similarities

Install RDKit support with `pip install 'opdiv[chem]'`.

```python
from opdiv import select, tanimoto_similarity

smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O"]
scores = [0.9, 0.8, 0.7, 0.6]
similarities = tanimoto_similarity(smiles)  # Morgan radius 2, 2048 bits, no chirality

result = select(scores, k=2, similarities=similarities, max_similarity=0.4)
selected_smiles = [smiles[i] for i in result.indices]
```

Any finite symmetric similarity matrix works, including shape or electrostatic
comparisons. A pair conflicts exactly when **similarity > max_similarity**;
equality is allowed, and no numerical epsilon is added. The similarity diagonal
is ignored. Asymmetric matrices must be explicitly symmetrized before use.

Alternatively, supply a symmetric boolean/0–1 `conflicts` matrix with a false
diagonal. For distances, construct conflicts with `distances < min_distance`
and set the diagonal to false. Use `<=` if equality should conflict. With neither
matrix, `select(scores, k)` returns ordinary top-k.

Filter ineligible candidates and resolve molecular identity **before selection**.
The SMILES helper rejects invalid/empty molecules and duplicate canonical
isomeric SMILES; it does not normalize salts or tautomers. Matrix-based calls
assume each row is a distinct eligible candidate.

## Feasibility and bounds

```python
result = select(scores, k=2, similarities=similarities,
                max_similarity=0.4, time_limit=10.0)
print(result.status, result.value, result.upper_bound, result.gap)
```

| Status | Meaning |
|---|---|
| `optimal` | Optimum certified within the numerical tolerance below; inspect `gap`. |
| `feasible` | Full portfolio found; its mean is a lower bound on OPDiv. |
| `incomplete` | Greedy stopped short; another full portfolio may exist. |
| `infeasible` | The requested capacity is proven infeasible. |
| `unknown` | Search stopped without finding a full portfolio or proving infeasibility. |

`value` is the mean of a full selection, or `None` when underfilled. A partial
greedy result retains its indices, but has no GPDiv value at the requested k.
`upper_bound` bounds the optimum; `gap` is the absolute difference from `value`.
Neither repeating candidates nor relaxing the diversity threshold fills a result.

`opdiv(...)` raises `MetricUndefinedError` unless optimality is established;
`gpdiv(...)` raises it when its metric is undefined. The exception
has a `.result` containing the selection and available bounds. Use `select(...)`
when you want to inspect a time-limited result rather than require a scalar metric.

Optimal selection uses [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver),
with binary variables, exactly k selections, and one exclusion constraint per
conflict edge. A complete greedy portfolio supplies a solver hint and an objective
lower bound. One search worker and a fixed seed make runs reproducible within a
solver version.

Scores are centered and rescaled before conversion to integer coefficients at
precision 1e-9 in normalized units. The largest k coefficient-rounding errors
are included in the returned upper bound. `optimal` means CP-SAT proved the
integer optimum and the original-score mean gap is at most 1e-8 times the
centered score scale (the largest absolute centered score, or 1 for constant
scores), subject to floating-point arithmetic. A small nonzero `gap` can remain
due to rounding; extremely close portfolios can be indistinguishable at this
precision. Returned `value` always uses the original scores.

There is no default time limit; `time_limit` limits the solver only. On a stopped
search, a complete greedy portfolio is retained if it beats the solver's candidate.
Greedy ties follow input order. Equally optimal selections can differ between
solver versions. Pairwise matrices require O(m²) memory; difficult conflict graphs
can be expensive to optimize.

This package uses the paper's CP-SAT backend and OPDiv/GPDiv definitions. It keeps
a direct full-graph model; the experiment runner's clique compression, score-prefix
relaxations, and benchmark pipeline are not included. Its normalized integer
scaling also differs from the experiment runner's fixed score multiplier.
For reproduction, use identical eligible candidates, row ordering, scores, and
conflict graphs: experiment-specific threshold tolerances must be encoded in
`conflicts` explicitly. The molecular helper matches the paper's non-chiral
fingerprint settings; compute fingerprints from the original archived graphs
when reproducing the archive experiments.

## Development

```bash
python -m pytest
ruff check .
python -m build
python -m twine check dist/*
```

Tests compare optimal results and rounding-aware bounds with exhaustive enumeration,
exercise the paper's greedy failure, and cover threshold boundaries, negative
utilities, time-limit outcomes, input validation, and optional molecular support.

The source repository is [mireklzicar/opdiv](https://github.com/mireklzicar/opdiv);
[Deep-MedChem/opdiv](https://github.com/Deep-MedChem/opdiv) is its organization fork.

## Attribution

The portfolio metrics and their molecular evaluation application are introduced
in the paper above. The underlying graph optimization and score-ordered greedy
algorithms are established methods; this package does not claim their invention.
See the paper for the scientific formulation and prior work. MIT licensed.
