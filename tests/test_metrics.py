"""Testes de mlscratch.utils.metrics contra o scikit-learn."""

import numpy as np
import pytest
from sklearn import metrics as skm

from mlscratch.utils import metrics as m

REGRESSAO = [
    (m.mean_squared_error, skm.mean_squared_error),
    (m.root_mean_squared_error, skm.root_mean_squared_error),
    (m.mean_absolute_error, skm.mean_absolute_error),
    (m.r2_score, skm.r2_score),
]
IDS_REGRESSAO = ["mse", "rmse", "mae", "r2"]


@pytest.fixture
def regressao():
    rng = np.random.default_rng(0)
    y_true = rng.normal(loc=50, scale=10, size=200)
    y_pred = y_true + rng.normal(scale=5, size=200)  # um modelo razoável, com erro
    return y_true, y_pred


# ---------------------------------------------------------------- Regressão


@pytest.mark.parametrize("minha, sk", REGRESSAO, ids=IDS_REGRESSAO)
def test_regressao_paridade_com_sklearn(regressao, minha, sk):
    y_true, y_pred = regressao
    assert np.isclose(minha(y_true, y_pred), sk(y_true, y_pred))


@pytest.mark.parametrize("minha, sk", REGRESSAO, ids=IDS_REGRESSAO)
def test_regressao_aceita_listas(minha, sk):
    y_true, y_pred = [3, -0.5, 2, 7], [2.5, 0.0, 2, 8]
    assert np.isclose(minha(y_true, y_pred), sk(y_true, y_pred))


def test_valores_conhecidos():
    # Erros: 0.5, -0.5, 0, -1  ->  quadrados: 0.25, 0.25, 0, 1
    y_true, y_pred = [3, -0.5, 2, 7], [2.5, 0.0, 2, 8]
    assert np.isclose(m.mean_squared_error(y_true, y_pred), 0.375)
    assert np.isclose(m.root_mean_squared_error(y_true, y_pred), np.sqrt(0.375))
    assert np.isclose(m.mean_absolute_error(y_true, y_pred), 0.5)


def test_previsao_perfeita():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert m.mean_squared_error(y, y) == 0
    assert m.root_mean_squared_error(y, y) == 0
    assert m.mean_absolute_error(y, y) == 0
    assert m.r2_score(y, y) == 1.0


def test_rmse_e_raiz_do_mse(regressao):
    y_true, y_pred = regressao
    assert np.isclose(m.root_mean_squared_error(y_true, y_pred), np.sqrt(m.mean_squared_error(y_true, y_pred)))


def test_r2_prever_a_media_da_zero():
    # Um "modelo" que sempre prevê a média de y tem R² = 0, por definição.
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert np.isclose(m.r2_score(y_true, y_pred), 0.0)


def test_r2_pode_ser_negativo():
    # Pior que prever a média: R² < 0 (não é limitado em 0).
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([3.0, 2.0, 1.0])
    assert m.r2_score(y_true, y_pred) < 0
    assert np.isclose(m.r2_score(y_true, y_pred), skm.r2_score(y_true, y_pred))


@pytest.mark.parametrize(
    "y_pred, esperado", [([5.0, 5.0, 5.0], 1.0), ([4.0, 5.0, 6.0], 0.0)], ids=["perfeito", "com_erro"]
)
def test_r2_com_y_true_constante(y_pred, esperado):
    # y_true constante: a variância é 0 e o R² seria 0/0. O sklearn devolve 1.0 se a
    # previsão for perfeita e 0.0 caso contrário.
    y_true = [5.0, 5.0, 5.0]
    assert m.r2_score(y_true, y_pred) == esperado
    assert m.r2_score(y_true, y_pred) == skm.r2_score(y_true, y_pred)


@pytest.mark.parametrize("minha, sk", REGRESSAO, ids=IDS_REGRESSAO)
def test_regressao_y_pred_em_coluna(regressao, minha, sk):
    # Muitos modelos devolvem previsões com shape (n, 1). Com y_true de shape (n,),
    # subtrair direto faz broadcasting para (n, n) e a métrica sai errada sem nenhum erro.
    y_true, y_pred = regressao
    y_pred_coluna = y_pred.reshape(-1, 1)
    assert np.isclose(minha(y_true, y_pred_coluna), sk(y_true, y_pred_coluna))


@pytest.mark.parametrize("y_pred", [[1.0, 2.0], [2.0]], ids=["2_de_3", "1_de_3"])
@pytest.mark.parametrize("minha, sk", REGRESSAO, ids=IDS_REGRESSAO)
def test_regressao_tamanhos_diferentes_lanca_erro(minha, sk, y_pred):
    # Com 1 elemento, o broadcasting compara todos os y_true com o mesmo valor, sem erro.
    with pytest.raises(ValueError):
        minha([1.0, 2.0, 3.0], y_pred)


# ------------------------------------------------------- Matriz de confusão


def test_confusion_exemplo_calculado_a_mao():
    y_true = [0, 0, 0, 1, 1, 1, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 1, 1, 0]
    assert np.array_equal(m.confusion_matrix(y_true, y_pred), [[3, 1], [1, 3]])


@pytest.mark.parametrize("k", [2, 3, 10])
def test_confusion_paridade_com_sklearn(k):
    rng = np.random.default_rng(k)
    y_true = rng.integers(0, k, 300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, rng.integers(0, k, 300))  # ~70% de acerto
    assert np.array_equal(m.confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred))


def test_confusion_rotulos_de_texto():
    y_true = ["sim", "nao", "sim", "talvez", "nao", "sim"]
    y_pred = ["sim", "sim", "nao", "talvez", "nao", "sim"]
    assert np.array_equal(m.confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred))


def test_confusion_rotulos_nao_consecutivos():
    # Rótulos 1, 5, 9 não servem direto como índice: a matriz tem que ser 3x3, não 10x10.
    y_true = [1, 5, 9, 5, 1]
    y_pred = [1, 9, 9, 5, 5]
    M = m.confusion_matrix(y_true, y_pred)
    assert M.shape == (3, 3)
    assert np.array_equal(M, skm.confusion_matrix(y_true, y_pred))


def test_confusion_classe_que_so_aparece_na_previsao():
    # O modelo previu a classe 2, que não existe em y_true: ela ainda ganha linha e coluna.
    y_true = [0, 1, 0, 1]
    y_pred = [0, 2, 0, 1]
    M = m.confusion_matrix(y_true, y_pred)
    assert M.shape == (3, 3)
    assert np.array_equal(M, skm.confusion_matrix(y_true, y_pred))


def test_confusion_classe_nunca_prevista():
    # O modelo nunca previu a classe 1: a coluna 1 fica zerada (é aqui que o minlength importa).
    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 0, 0]
    assert np.array_equal(m.confusion_matrix(y_true, y_pred), [[2, 0], [2, 0]])


def test_confusion_propriedades():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 200)
    y_pred = rng.integers(0, 4, 200)
    M = m.confusion_matrix(y_true, y_pred)
    assert M.sum() == 200                                              # toda amostra contada uma vez
    assert np.array_equal(M.sum(axis=1), np.bincount(y_true))         # linhas = quantas de cada classe real
    assert np.array_equal(M.sum(axis=0), np.bincount(y_pred))         # colunas = quantas de cada prevista
    assert np.trace(M) == np.sum(y_true == y_pred)                    # diagonal = acertos


def test_confusion_y_pred_em_coluna():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([[0], [1], [0], [0]])
    assert np.array_equal(m.confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred.ravel()))


@pytest.mark.parametrize("y_pred", [[0, 1], [1]], ids=["2_de_3", "1_de_3"])
def test_confusion_tamanhos_diferentes_lanca_erro(y_pred):
    with pytest.raises(ValueError):
        m.confusion_matrix([0, 1, 0], y_pred)
