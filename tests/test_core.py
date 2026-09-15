from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from opdiv import MetricUndefinedError, cpdiv, gpdiv, opdiv, select, select_clusters


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


@pytest.mark.parametrize("cap", [1, 2, 4])
def test_clusters_against_enumeration(cap):
    q = np.array([5, 4, -1, -2, -3.0])
    labels = ["a", "a", "b", "b", "c"]
    for k in range(1, 7):
        values = [
            q[list(s)].mean()
            for s in combinations(range(len(q)), k)
            if all(sum(labels[i] == label for i in s) <= cap for label in set(labels))
        ]
        result = select_clusters(q, k, labels, cap=cap)
        if values:
            assert result.status == "optimal"
            assert cpdiv(q, k, labels, cap=cap) == pytest.approx(max(values))
        else:
            assert result.status == "infeasible" and result.value is None
            with pytest.raises(MetricUndefinedError):
                cpdiv(q, k, labels, cap=cap)


def test_timeout_retains_greedy_and_refuses_scalar_metric(monkeypatch):
    monkeypatch.setattr(
        "opdiv._core.milp", lambda *a, **kw: OptimizeResult(status=1, x=None, mip_dual_bound=None)
    )
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
    monkeypatch.setattr(
        "opdiv._core.milp",
        lambda *a, **kw: OptimizeResult(status=1, x=np.array([0, 1, 1, 0]), mip_dual_bound=-0.4),
    )
    result = select([16, 12, 11, 6], 2, conflicts=graph(), time_limit=0.001)
    assert result.status == "feasible"
    assert result.value == 11.5 and result.upper_bound == 12
    assert result.gap == 0.5


def test_rejects_invalid_solver_portfolio(monkeypatch):
    monkeypatch.setattr(
        "opdiv._core.milp", lambda *a, **kw: OptimizeResult(status=0, x=np.array([1, 1, 0, 0]))
    )
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


@pytest.mark.parametrize(
    "labels,cap",
    [(["a"], 1), ([None, "a"], 1), ([[], "a"], 1), ([np.nan, "a"], 1), (["a", "b"], 0)],
)
def test_invalid_clusters(labels, cap):
    with pytest.raises(ValueError):
        select_clusters([1, 2], 1, labels, cap=cap)
