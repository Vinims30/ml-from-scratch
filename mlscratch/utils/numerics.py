#Funções numericamente estáveis (Fase 1, Semana 2, Sessão 2).
import numpy as np

def softmax(Z):
    maior_linha = np.max(Z,keepdims=True, axis=1)
    exp = np.exp(Z - maior_linha)
    softmax = exp / np.sum(exp, axis=1, keepdims=True)
    
    return softmax


def logsumexp(Z, axis = -1):
    maior_linha = np.max(Z, keepdims=True,axis=axis)
    exp = np.exp(Z - maior_linha)
    logsumexp = maior_linha + np.log(np.sum(exp, axis=axis, keepdims=True))
    return logsumexp


def log_softmax(Z, axis = -1):
    return Z - logsumexp(Z, axis)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    resultado = np.empty_like(z)

    pos = z >= 0
    resultado[pos] = 1 / (1 + np.exp(-z[pos]))

    exp_neg = np.exp(z[~pos])
    resultado[~pos] = exp_neg / ( exp_neg + 1)

    return resultado
    