"""Molecular portfolio selection and diversity evaluation."""

from ._core import (
    MetricUndefinedError,
    Portfolio,
    gpdiv,
    opdiv,
    select,
)
from ._similarity import conflicts_from_similarity, tanimoto_similarity

__version__ = "0.1.0"
__all__ = [
    "MetricUndefinedError",
    "Portfolio",
    "conflicts_from_similarity",
    "gpdiv",
    "opdiv",
    "select",
    "tanimoto_similarity",
]
