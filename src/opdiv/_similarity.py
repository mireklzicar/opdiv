"""Pairwise conflict graphs and optional molecular fingerprints."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def conflicts_from_similarity(similarities: ArrayLike, max_similarity: float) -> NDArray[np.bool_]:
    """Build a graph with an edge exactly when s(i, j) > max_similarity.

    Equality is compatible. Require a finite, exactly symmetric square matrix;
    the diagonal is ignored. Similarities need not be in [0, 1]. No implicit
    symmetrization or threshold tolerance is applied.
    """
    s = np.asarray(similarities, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("similarities must be a square matrix")
    if not np.isfinite(s).all() or not np.array_equal(s, s.T):
        raise ValueError("similarities must be finite and symmetric")
    if not np.isscalar(max_similarity) or not np.isfinite(max_similarity):
        raise ValueError("max_similarity must be finite")
    graph = s > max_similarity
    np.fill_diagonal(graph, False)
    return graph


def tanimoto_similarity(
    smiles: Sequence[str], *, radius: int = 2, n_bits: int = 2048
) -> NDArray[np.float64]:
    """Morgan bit-fingerprint Tanimoto matrix; requires ``opdiv[chem]``.

    Uses chirality and preserves input order. Invalid, empty, or duplicate
    canonical isomeric SMILES raise ValueError. No salt/tautomer normalization
    is performed; establish your molecular identity policy before calling.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise ImportError("Install molecular support with: pip install 'opdiv[chem]'") from exc

    if isinstance(smiles, str):
        raise ValueError("smiles must be a sequence of SMILES strings")
    for name, value, minimum in [("radius", radius, 0), ("n_bits", n_bits, 1)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius), fpSize=int(n_bits), includeChirality=True
    )
    fingerprints, seen = [], set()
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) and text.strip() else None
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty SMILES at index {i}: {text!r}")
        identity = Chem.MolToSmiles(mol, isomericSmiles=True)
        if identity in seen:
            raise ValueError(f"Duplicate molecule at index {i}: {text!r}")
        seen.add(identity)
        fingerprints.append(generator.GetFingerprint(mol))
    matrix = np.eye(len(fingerprints), dtype=float)
    for i in range(1, len(fingerprints)):
        matrix[i, :i] = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        matrix[:i, i] = matrix[i, :i]
    return matrix
