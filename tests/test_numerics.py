"""Testes de mlscratch.utils.numerics contra o scipy.special."""

import numpy as np
import pytest
from scipy import special

from mlscratch.utils.numerics import log_softmax, logsumexp, sigmoid, softmax

# Faz overflow, divisão por zero e operações inválidas (nan) virarem erro dentro do teste.
# Underflow (exp(-800) -> 0) é esperado e permitido.
ESTRITO = dict(over="raise", divide="raise", invalid="raise")


@pytest.fixture
def Z():
    rng = np.random.default_rng(0)
    return rng.normal(size=(50, 7)) * 5


# ---------------------------------------------------------------- softmax


def test_softmax_paridade_com_scipy(Z):
    assert np.allclose(softmax(Z), special.softmax(Z, axis=1))


def test_softmax_linhas_somam_1_e_entre_0_e_1(Z):
    P = softmax(Z)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert ((P >= 0) & (P <= 1)).all()


@pytest.mark.parametrize("logits", [[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
def test_softmax_estavel_com_logits_extremos(logits):
    with np.errstate(**ESTRITO):
        P = softmax(np.array([logits]))
    assert np.allclose(P, special.softmax(np.array([logits]), axis=1))


def test_softmax_invariante_a_deslocamento(Z):
    # softmax(z + c) = softmax(z): é por isso que subtrair o máximo não muda o resultado.
    assert np.allclose(softmax(Z), softmax(Z + 123.4))


# -------------------------------------------------------------- logsumexp


@pytest.mark.parametrize("axis", [-1, 1, 0])
def test_logsumexp_paridade_com_scipy(Z, axis):
    assert np.allclose(logsumexp(Z, axis=axis), special.logsumexp(Z, axis=axis, keepdims=True))


def test_logsumexp_igual_a_versao_ingenua_com_valores_pequenos(Z):
    ingenua = np.log(np.sum(np.exp(Z), axis=1, keepdims=True))
    assert np.allclose(logsumexp(Z, axis=1), ingenua)


def test_logsumexp_sem_overflow():
    Z = np.array([[1000.0, 1000.0], [-1000.0, -1000.0]])
    with np.errstate(**ESTRITO):
        resultado = logsumexp(Z, axis=1)
    assert np.allclose(resultado, [[1000 + np.log(2)], [-1000 + np.log(2)]])


# ------------------------------------------------------------ log_softmax


def test_log_softmax_paridade_com_scipy(Z):
    assert np.allclose(log_softmax(Z), special.log_softmax(Z, axis=1))


def test_exp_do_log_softmax_igual_softmax(Z):
    assert np.allclose(np.exp(log_softmax(Z)), softmax(Z))


def test_log_softmax_finito_quando_probabilidade_vira_zero():
    # exp(-800) dá 0 em float64, então log(softmax) daria -inf.
    # O log_softmax direto deve dar o valor finito correto (~ -800.31).
    Z = np.array([[0.0, -800.0, -1.0]])
    with np.errstate(**ESTRITO):
        L = log_softmax(Z)
    assert np.isfinite(L).all()
    assert np.allclose(L, special.log_softmax(Z, axis=1))


# ---------------------------------------------------------------- sigmoid


@pytest.mark.parametrize("z", [0.0, 2.5, -2.5, 30.0, -30.0, 1000.0, -1000.0])
def test_sigmoid_escalar_paridade_e_estabilidade(z):
    with np.errstate(**ESTRITO):
        s = sigmoid(z)
    assert np.isclose(s, special.expit(z))


def test_sigmoid_em_array():
    # Uso real em ML: aplicar em um vetor inteiro de uma vez (ex.: z = X @ w na logística).
    z = np.linspace(-50, 50, 101)
    assert np.allclose(sigmoid(z), special.expit(z))


def test_sigmoid_em_array_com_valores_extremos_sem_overflow():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    with np.errstate(**ESTRITO):
        s = sigmoid(z)
    assert np.allclose(s, [0.0, special.expit(-1.0), 0.5, special.expit(1.0), 1.0])


def test_sigmoid_simetria():
    z = np.linspace(-10, 10, 21)
    assert np.allclose(sigmoid(-z), 1 - sigmoid(z))
