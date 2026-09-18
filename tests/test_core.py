from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from ortools.sat.python import cp_model

from opdiv import MetricUndefinedError, gpdiv, opdiv, select


def graph():
    return np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], bool)


def test_paper_counterexample():
    scores = [16, 12, 11, 6]
    result = select(scores, 2, conflicts=graph())
    assert result.indices == (1, 2)
    assert result.status == "optimal"
    assert result.is_complete
    assert result.gap == pytest.approx(0)
    assert opdiv(scores, 2, conflicts=graph()) == 11.5
    assert gpdiv(scores, 2, conflicts=graph()) == 11


def test_greedy_underfill_is_not_infeasibility():
    a = graph()[:3, :3]
    greedy = select([16, 12, 11], 2, conflicts=a, method="greedy")
    assert greedy.status == "incomplete"
    assert greedy.indices == (0,)
    assert greedy.value is None and greedy.gap is None
    assert not greedy.is_complete
    with pytest.raises(MetricUndefinedError) as exc:
        gpdiv([16, 12, 11], 2, conflicts=a)
    assert exc.value.result == greedy
    assert opdiv([16, 12, 11], 2, conflicts=a) == 11.5


@pytest.mark.parametrize("seed", range(12))
def test_optimum_and_bounds_against_exhaustive_enumeration(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=9)
    a = np.triu(rng.random((9, 9)) < 0.45, 1)
    a |= a.T
    for k in (1, 3, 6, 9):
        values = [
            float(q[list(s)].mean()) for s in combinations(range(9), k) if not a[np.ix_(s, s)].any()
        ]
        result = select(q, k, conflicts=a)
        if not values:
            assert result.status == "infeasible"
            assert result.value is None and result.upper_bound is None
        else:
            optimum = max(values)
            assert result.status == "optimal"
            assert result.value == pytest.approx(optimum)
            assert result.upper_bound >= optimum - 1e-8
            assert result.value <= optimum + 1e-8
            assert len(set(result.indices)) == k
            assert not a[np.ix_(result.indices, result.indices)].any()


@pytest.mark.parametrize("scores,k", [([], 1), ([1], 2)])
def test_not_enough_candidates(scores, k):
    assert select(scores, k).status == "infeasible"
    with pytest.raises(MetricUndefinedError):
        opdiv(scores, k)


def test_negative_scores_and_stable_ties():
    assert select([-4, -2, -2, -8], 3).indices == (1, 2, 0)
    assert opdiv([-4, -2, -2, -8], 3) == pytest.approx(-8 / 3)


@pytest.mark.parametrize("scale,offset", [(1e-12, 0), (1e-3, 1e9), (1e6, -1e12)])
def test_positive_affine_score_transform_preserves_selection(scale, offset):
    q = np.array([16, 12, 11, 6]) * scale + offset
    result = select(q, 2, conflicts=graph())
    assert result.indices == (1, 2)
    assert result.value == pytest.approx(float(q[[1, 2]].mean()))


def mock_solver(monkeypatch, code, selected=(), bound=0):
    class Solver:
        parameters = SimpleNamespace()
        best_objective_bound = bound

        def solve(self, model):
            return code

        def value(self, var):
            return var.index in selected

        def solution_info(self):
            return "mock model error"

    monkeypatch.setattr("opdiv._core.cp_model.CpSolver", Solver)


def test_timeout_retains_greedy_and_refuses_scalar_metric(monkeypatch):
    mock_solver(monkeypatch, cp_model.UNKNOWN)
    result = select([16, 12, 11, 6], 2, conflicts=graph(), time_limit=0.001)
    assert result.status == "feasible"
    assert result.indices == (0, 3)
    assert result.value == 11 and result.upper_bound == 14
    with pytest.raises(MetricUndefinedError) as exc:
        opdiv([16, 12, 11, 6], 2, conflicts=graph(), time_limit=0.001)
    assert exc.value.result == result
    unknown = select([16, 12, 11], 2, conflicts=graph()[:3, :3], time_limit=0.001)
    assert unknown.status == "unknown"
    assert unknown.value is None and unknown.upper_bound == 14


def test_timeout_with_solver_incumbent_and_bound(monkeypatch):
    mock_solver(monkeypatch, cp_model.FEASIBLE, selected=(1, 2), bound=400_000_000)
    result = select([16, 12, 11, 6], 2, conflicts=graph(), time_limit=0.001)
    assert result.status == "feasible"
    assert result.value == 11.5
    assert result.upper_bound == pytest.approx(12)
    assert result.gap == pytest.approx(0.5)


def test_rejects_invalid_solver_portfolio(monkeypatch):
    mock_solver(monkeypatch, cp_model.OPTIMAL, selected=(0, 1))
    with pytest.raises(RuntimeError, match="invalid portfolio"):
        select([16, 12, 11, 6], 2, conflicts=graph())


@pytest.mark.parametrize(
    "scores,k,kwargs",
    [
        ([1, np.nan], 1, {}),
        ([[1, 2]], 1, {}),
        ([1, 2], 0, {}),
        ([1, 2], True, {}),
        ([1, 2], 1.5, {}),
        ([1, 2], 1, {"conflicts": [[0, 1], [0, 0]]}),
        ([1, 2], 1, {"conflicts": [[1, 0], [0, 0]]}),
        ([1, 2], 1, {"conflicts": [[0, 0.5], [0.5, 0]]}),
        ([1, 2], 1, {"similarities": np.eye(3), "max_similarity": 0.5}),
        ([1, 2], 1, {"similarities": np.eye(2)}),
        ([1, 2], 1, {"max_similarity": 0.5}),
        ([1, 2], 1, {"method": "typo"}),
        ([1, 2], 1, {"time_limit": -1}),
        ([1, 2], 1, {"time_limit": np.nan}),
        ([1, 2], 1, {"time_limit": 1, "method": "greedy"}),
        (
            [1, 2],
            1,
            {"similarities": np.eye(2), "max_similarity": 0.5, "conflicts": np.zeros((2, 2))},
        ),
    ],
)
def test_invalid_inputs(scores, k, kwargs):
    with pytest.raises(ValueError):
        select(scores, k, **kwargs)


@pytest.mark.parametrize("seed", range(12))
def test_rounding_bounds_cover_original_float_optimum(seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1, 1, 8)
    a = np.triu(rng.random((8, 8)) < 0.3, 1)
    a |= a.T
    k = 3
    feasible = [q[list(s)].mean() for s in combinations(range(8), k) if not a[np.ix_(s, s)].any()]
    result = select(q, k, conflicts=a)
    if feasible:
        optimum = max(feasible)
        assert result.value <= optimum + 1e-14
        assert result.upper_bound >= optimum - 1e-14
        assert result.gap <= 1e-8 * np.ptp(q)


def test_integer_optimum_does_not_hide_rounding_gap(monkeypatch):
    # Both portfolios tie after integer rounding; B,C have better real utility.
    q = [1, 0.75, 0.75000000001, 0.5]
    mock_solver(monkeypatch, cp_model.OPTIMAL, selected=(0, 3), bound=0)
    result = select(q, 2, conflicts=graph())
    assert result.value == 0.75
    assert result.upper_bound >= np.mean(q[1:3])
    assert result.gap > 0


def test_model_errors_are_not_reported_as_infeasibility(monkeypatch):
    mock_solver(monkeypatch, cp_model.MODEL_INVALID)
    with pytest.raises(RuntimeError, match="mock model error"):
        select([16, 12, 11, 6], 2, conflicts=graph())


def test_public_api_contains_only_pairwise_metrics():
    import opdiv as package

    assert not hasattr(package, "cpdiv")
    assert not hasattr(package, "select_clusters")
