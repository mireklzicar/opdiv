"""Fixed-cardinality portfolio objectives, independent of molecular representation."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ._similarity import conflicts_from_similarity

Status = Literal["optimal", "feasible", "infeasible", "incomplete", "unknown"]


@dataclass(frozen=True)
class Portfolio:
    """Selection and its mean-utility bounds at the requested capacity.

    ``indices`` refer to input rows, ordered by descending score, then input
    index. ``value`` is None unless exactly k compatible candidates were found.
    For pairwise selection, ``upper_bound`` bounds OPDiv; for cluster selection
    it bounds CPDiv. Optimality and solver bounds use floating-point tolerances.
    """

    indices: tuple[int, ...]
    k: int
    value: float | None
    upper_bound: float | None
    status: Status

    @property
    def is_complete(self) -> bool:
        return len(self.indices) == self.k

    @property
    def gap(self) -> float | None:
        """Absolute gap in mean-utility units, or None without a full portfolio."""
        if self.value is None or self.upper_bound is None:
            return None
        return max(0.0, self.upper_bound - self.value)


class MetricUndefinedError(ValueError):
    """No full portfolio or no proven optimum; inspect ``result`` for details."""

    def __init__(self, metric: str, result: Portfolio):
        self.result = result
        super().__init__(
            f"{metric} is unavailable (status={result.status}, "
            f"selected={len(result.indices)}/{result.k}); use select() to inspect bounds"
        )


def _inputs(scores: ArrayLike, k: int) -> NDArray[np.float64]:
    q = np.asarray(scores, dtype=float)
    if q.ndim != 1 or not np.isfinite(q).all():
        raise ValueError("scores must be a one-dimensional array of finite utilities")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    return q


def _graph(
    m: int,
    conflicts: ArrayLike | None,
    similarities: ArrayLike | None,
    max_similarity: float | None,
) -> NDArray[np.bool_]:
    if similarities is not None:
        if conflicts is not None or max_similarity is None:
            raise ValueError("Supply similarities with max_similarity, or conflicts, not both")
        a = conflicts_from_similarity(similarities, max_similarity)
    else:
        if max_similarity is not None:
            raise ValueError("max_similarity requires similarities")
        if conflicts is None:
            return np.zeros((m, m), dtype=bool)
        raw = np.asarray(conflicts)
        if not np.isin(raw, [0, 1]).all():
            raise ValueError("conflicts must contain only booleans or 0/1")
        a = raw.astype(bool)
    if a.shape != (m, m) or not np.array_equal(a, a.T):
        raise ValueError("The pairwise matrix must be symmetric and match the number of scores")
    if np.diag(a).any():
        raise ValueError("conflicts must have a false diagonal")
    return a


def _result(q, picks, k, upper, status) -> Portfolio:
    indices = tuple(sorted((int(i) for i in picks), key=lambda i: (-q[i], i)))
    # Divide first to avoid overflowing the sum of finite scores.
    value = float(np.sum(q[list(indices)] / k)) if len(indices) == k else None
    return Portfolio(indices, int(k), value, upper, status)


def _greedy(q, a, k) -> list[int]:
    blocked = np.zeros(len(q), dtype=bool)
    picks = []
    for i in np.argsort(-q, kind="stable"):
        if not blocked[i]:
            picks.append(int(i))
            blocked |= a[i]
            blocked[i] = True
            if len(picks) == k:
                break
    return picks


def select(
    scores: ArrayLike,
    k: int,
    *,
    similarities: ArrayLike | None = None,
    max_similarity: float | None = None,
    conflicts: ArrayLike | None = None,
    method: Literal["optimal", "greedy"] = "optimal",
    time_limit: float | None = None,
) -> Portfolio:
    """Select exactly k candidates maximizing mean score under pairwise exclusion.

    Larger scores are better: negate docking energies before calling. Supply
    either a similarity matrix and ceiling (equality allowed), a boolean conflict
    matrix, or neither for ordinary top-k. Candidates must already be eligible
    and distinct; matrix inputs cannot establish molecular identity.

    ``optimal`` uses SciPy/HiGHS mixed-integer optimization without a time limit
    by default. ``time_limit`` limits solver seconds, excluding graph preparation.
    A stopped search returns a feasible lower bound if available, never a falsely
    certified metric. ``greedy`` scans by score with input-order tie breaking;
    underfilling does not establish infeasibility. Equally optimal portfolios may
    differ across solver versions; returned indices are always score-sorted.
    """
    q = _inputs(scores, k)
    a = _graph(len(q), conflicts, similarities, max_similarity)
    if method not in ("optimal", "greedy"):
        raise ValueError("method must be 'optimal' or 'greedy'")
    if time_limit is not None:
        if not np.isscalar(time_limit) or not np.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("time_limit must be positive and finite")
        if method != "optimal":
            raise ValueError("time_limit applies only to optimal selection")
    if k > len(q):
        return _result(q, [], k, None, "infeasible")
    top = np.argsort(-q, kind="stable")[:k]
    upper = float(np.sum(q[top] / k))
    if not a[np.ix_(top, top)].any():
        return _result(q, top, k, upper, "optimal")
    picks = _greedy(q, a, k)
    if method == "greedy":
        return _result(q, picks, k, upper, "feasible" if len(picks) == k else "incomplete")

    # One exact-cardinality row, followed by x_i + x_j <= 1 per conflict edge.
    ii, jj = np.where(np.triu(a, 1))
    edges, m = len(ii), len(q)
    rows = np.concatenate([np.zeros(m, dtype=int), np.arange(1, edges + 1).repeat(2)])
    cols = np.concatenate([np.arange(m), np.column_stack([ii, jj]).ravel()])
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(edges + 1, m)).tocsc()
    lower = np.concatenate([[k], np.full(edges, -np.inf)])
    upper_constraints = np.concatenate([[k], np.ones(edges)])
    options = {"mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    # At fixed k, centering and positive rescaling preserve the optimum. Center
    # first so a large score offset does not hide meaningful differences.
    center = float(np.max(q) / 2 + np.min(q) / 2)
    centered = q - center
    scale = float(np.max(np.abs(centered))) or 1.0
    weights = centered / scale
    solved = milp(
        -weights,
        integrality=np.ones(m),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper_constraints),
        options=options,
    )
    if solved.status == 2:
        if len(picks) == k:
            raise RuntimeError("Solver reported infeasibility despite a feasible greedy portfolio")
        return _result(q, [], k, None, "infeasible")
    if solved.status not in (0, 1):
        raise RuntimeError(f"Portfolio solver failed: {solved.message}")
    if len(picks) != k:
        picks = []
    if solved.x is not None:
        candidate = np.flatnonzero(solved.x > 0.5).tolist()
        if len(candidate) != k or a[np.ix_(candidate, candidate)].any():
            raise RuntimeError("Solver returned an invalid portfolio")
        if not picks or np.sum(weights[candidate]) > np.sum(weights[picks]):
            picks = candidate
    if solved.status == 0 and not picks:
        raise RuntimeError("Solver reported optimality without a portfolio")
    bound = getattr(solved, "mip_dual_bound", None)
    if bound is not None and np.isfinite(bound):
        upper = min(upper, float((-bound / k) * scale + center))
    if picks:
        value = float(np.sum(q[picks] / k))
        upper = max(upper, value)  # Protect against round-off in the solver bound.
    status = "optimal" if solved.status == 0 else "feasible" if picks else "unknown"
    return _result(q, picks, k, upper, status)


def opdiv(
    scores: ArrayLike,
    k: int,
    *,
    similarities: ArrayLike | None = None,
    max_similarity: float | None = None,
    conflicts: ArrayLike | None = None,
    time_limit: float | None = None,
) -> float:
    """Optimal Portfolio Diversity: the best feasible size-k mean utility.

    Raise MetricUndefinedError for infeasibility or an unproven optimum.
    The exception's result retains any feasible portfolio and its bounds.
    """
    result = select(
        scores,
        k,
        similarities=similarities,
        max_similarity=max_similarity,
        conflicts=conflicts,
        time_limit=time_limit,
    )
    if result.status != "optimal" or result.value is None:
        raise MetricUndefinedError("OPDiv", result)
    return result.value


def gpdiv(
    scores: ArrayLike,
    k: int,
    *,
    similarities: ArrayLike | None = None,
    max_similarity: float | None = None,
    conflicts: ArrayLike | None = None,
) -> float:
    """Greedy Portfolio Diversity; raise MetricUndefinedError if underfilled."""
    result = select(
        scores,
        k,
        similarities=similarities,
        max_similarity=max_similarity,
        conflicts=conflicts,
        method="greedy",
    )
    if result.value is None:
        raise MetricUndefinedError("GPDiv", result)
    return result.value


def select_clusters(
    scores: ArrayLike, k: int, labels: Sequence[Hashable], *, cap: int = 1
) -> Portfolio:
    """Optimal size-k selection under a uniform upper cap per disjoint cluster.

    Cluster labels must be hashable, non-missing values. This policy does not
    enforce pairwise separation. Ties follow input order. Exhaustion certifies
    infeasibility, in contrast to pairwise greedy selection.
    """
    q = _inputs(scores, k)
    if isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap < 1:
        raise ValueError("cap must be a positive integer")
    if len(labels) != len(q):
        raise ValueError("labels must match the number of scores")
    for label in labels:
        try:
            hash(label)
            if label is None or label != label:
                raise ValueError("labels must not contain missing values")
        except TypeError as exc:
            raise ValueError("labels must be hashable, non-missing values") from exc
    picks, counts = [], {}
    for i in np.argsort(-q, kind="stable"):
        label = labels[i]
        if counts.get(label, 0) < cap:
            counts[label] = counts.get(label, 0) + 1
            picks.append(int(i))
            if len(picks) == k:
                value = float(np.sum(q[picks] / k))
                return _result(q, picks, k, value, "optimal")
    return _result(q, picks, k, None, "infeasible")


def cpdiv(scores: ArrayLike, k: int, labels: Sequence[Hashable], *, cap: int = 1) -> float:
    """Clustered Portfolio Diversity; raise MetricUndefinedError if infeasible."""
    result = select_clusters(scores, k, labels, cap=cap)
    if result.value is None:
        raise MetricUndefinedError("CPDiv", result)
    return result.value
