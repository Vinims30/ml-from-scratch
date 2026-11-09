"""Testes de mlscratch.utils.distances contra o scikit-learn."""

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from mlscratch.utils.distances import (
    euclidean_one_loop,
    euclidean_two_loops,
    euclidean_vectorized,
)

IMPLEMENTACOES = [euclidean_two_loops, euclidean_one_loop, euclidean_vectorized]


@pytest.fixture(params=IMPLEMENTACOES, ids=lambda f: f.__name__)
def distancia(request):
    return request.param


@pytest.mark.parametrize("n, m, d", [(1, 1, 1), (5, 3, 2), (30, 20, 50), (100, 1, 10)])
def test_paridade_com_sklearn(distancia, n, m, d):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, d))
    B = rng.normal(size=(m, d))
    D = distancia(A, B)
    assert D.shape == (n, m)
    assert np.allclose(D, pairwise_distances(A, B))


def test_valor_conhecido(distancia):
    # Triângulo 3-4-5: distância entre (0, 0) e (3, 4) é 5.
    A = np.array([[0.0, 0.0]])
    B = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(distancia(A, B), [[5.0, 0.0]])


def test_mesmo_conjunto_simetrica_e_diagonal_zero(distancia):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 8))
    D = distancia(X, X)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)


def test_nao_negativa_e_sem_nan_com_pontos_quase_iguais(distancia):
    # Pontos grandes e quase idênticos: na vetorizada, ||a||² + ||b||² - 2aᵀb
    # pode ficar levemente negativo por arredondamento. O clip deve evitar nan.
    rng = np.random.default_rng(2)
    base = rng.normal(size=(1, 50)) * 1e4
    X = base + rng.normal(size=(20, 50)) * 1e-8
    D = distancia(X, X)
    assert not np.isnan(D).any()
    assert (D >= 0).all()


def test_tres_versoes_concordam_entre_si():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(25, 6))
    B = rng.normal(size=(15, 6))
    D2, D1, Dv = (f(A, B) for f in IMPLEMENTACOES)
    assert np.allclose(D2, D1)
    assert np.allclose(D2, Dv)
