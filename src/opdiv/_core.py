"""Fixed-cardinality portfolio objectives, independent of molecular representation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ortools.sat.python import cp_model

from ._similarity import conflicts_from_similarity

_OBJECTIVE_SCALE = 1_000_000_000
_OPTIMALITY_TOLERANCE = 1e-8

Status = Literal["optimal", "feasible", "infeasible", "incomplete", "unknown"]


@dataclass(frozen=True)
class Portfolio:
    """Selection and its mean-utility bounds at the requested capacity.

    ``indices`` refer to input rows, ordered by descending score, then input
    index. ``value`` is None unless exactly k compatible candidates were found.
    ``upper_bound`` bounds OPDiv, including integer-coefficient rounding error.
    Optimality is numerical: the certified mean gap is at most 1e-8 times the
    centered score scale, subject to floating-point arithmetic.
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

    ``optimal`` uses OR-Tools CP-SAT optimization without a time limit
    by default. ``time_limit`` limits solver seconds, excluding graph preparation.
    Integer rounding is included in the returned upper bound. Optimality is
    certified within 1e-8 times the centered score scale. A stopped search
    returns a feasible lower bound if available. ``greedy`` scans by score with
    input-order tie breaking;
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

    # Center/rescale before integer conversion so offsets and units do not
    # erase score differences. At fixed k this preserves the exact objective.
    center = float(np.max(q) / 2 + np.min(q) / 2)
    centered = q - center
    scale = float(np.max(np.abs(centered))) or 1.0
    normalized = centered / scale
    weights = np.rint(normalized * _OBJECTIVE_SCALE).astype(np.int64)
    # For ANY size-k set, the largest k coefficient errors bound mean error.
    errors = np.abs(normalized - weights / _OBJECTIVE_SCALE)
    rounding_error = float(np.sum(np.sort(errors)[-k:] / k))

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(len(q))]
    model.add(sum(x) == k)
    ii, jj = np.where(np.triu(a, 1))
    for i, j in zip(ii, jj, strict=True):
        model.add_at_most_one(x[i], x[j])
    objective = sum(int(w) * var for w, var in zip(weights, x, strict=True))
    model.maximize(objective)
    if len(picks) == k:
        chosen = set(picks)
        model.add(objective >= sum(int(weights[i]) for i in picks))
        for i, var in enumerate(x):
            model.add_hint(var, int(i in chosen))
    else:
        picks = []

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 20260914
    solver.parameters.relative_gap_limit = 0
    solver.parameters.absolute_gap_limit = 0
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    code = solver.solve(model)
    if code == cp_model.INFEASIBLE:
        if picks:
            raise RuntimeError("Solver reported infeasibility despite a feasible greedy portfolio")
        return _result(q, [], k, None, "infeasible")
    if code not in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN):
        raise RuntimeError(f"Portfolio solver failed: {solver.solution_info()}")

    status = "feasible" if picks else "unknown"
    if code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        candidate = [i for i, var in enumerate(x) if solver.value(var)]
        if len(candidate) != k or a[np.ix_(candidate, candidate)].any():
            raise RuntimeError("Solver returned an invalid portfolio")
        if not picks or np.sum(normalized[candidate]) > np.sum(normalized[picks]):
            picks = candidate
        # UNKNOWN can expose an uninitialized solver bound; retain top-k then.
        normalized_upper = (
            solver.best_objective_bound / (_OBJECTIVE_SCALE * k) + rounding_error
        )
        # Small outward allowance for arithmetic in scaling and bound conversion.
        normalized_upper += 16 * np.finfo(float).eps
        with np.errstate(over="ignore"):
            upper = min(upper, float(normalized_upper * scale + center))
        value = float(np.sum(q[picks] / k))
        upper = max(upper, value)
        normalized_gap = normalized_upper - float(np.sum(normalized[picks] / k))
        status = (
            "optimal"
            if code == cp_model.OPTIMAL and normalized_gap <= _OPTIMALITY_TOLERANCE
            else "feasible"
        )
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
    Optimality uses the numerical tolerance documented by select().
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
