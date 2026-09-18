import numpy as np
import pytest

from opdiv import conflicts_from_similarity, opdiv, select, tanimoto_similarity


def test_threshold_equality_and_adjacent_float():
    s = np.array([[1, 0.7], [0.7, 1]])
    original = s.copy()
    assert not conflicts_from_similarity(s, 0.7).any()
    assert opdiv([2, 1], 2, similarities=s, max_similarity=0.7) == 1.5
    threshold = np.nextafter(0.7, -np.inf)
    assert select([2, 1], 2, similarities=s, max_similarity=threshold).status == "infeasible"
    np.testing.assert_array_equal(s, original)


@pytest.mark.parametrize(
    "matrix,threshold",
    [
        ([[1, 0.2], [0.3, 1]], 0.5),
        ([[1, np.nan], [np.nan, 1]], 0.5),
        ([[1, 0]], 0.5),
        (np.eye(2), np.inf),
    ],
)
def test_invalid_similarity(matrix, threshold):
    with pytest.raises(ValueError):
        conflicts_from_similarity(matrix, threshold)


def test_morgan_tanimoto_matches_rdkit():
    pytest.importorskip("rdkit")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O"]
    matrix = tanimoto_similarity(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    for i in range(len(smiles)):
        for j in range(len(smiles)):
            assert matrix[i, j] == DataStructs.TanimotoSimilarity(fps[i], fps[j])
    assert select([0.9, 0.8, 0.7, 0.6], 2, similarities=matrix, max_similarity=0.4).is_complete
    assert tanimoto_similarity([]).shape == (0, 0)


@pytest.mark.parametrize("smiles", [["CCO", "OCC"], [""], ["invalid"], "CCO"])
def test_invalid_molecules(smiles):
    pytest.importorskip("rdkit")
    with pytest.raises(ValueError):
        tanimoto_similarity(smiles)


def test_morgan_ignores_chirality_as_in_paper():
    pytest.importorskip("rdkit")
    s = tanimoto_similarity(["C[C@H](O)F", "C[C@@H](O)F"])
    assert s[0, 1] == 1.0
