"""Métricas (Fase 1, Semana 3, Sessões 2 e 3).

Regressão:
- mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

Classificação:
- confusion_matrix  (via np.bincount)
- accuracy_score, precision_score, recall_score, f1_score  (binário e macro; zero_division)
- log_loss          (clip em [eps, 1 - eps])
- roc_auc_score     (pela curva/trapézio e por ranks/Mann-Whitney, com empates)

Paridade: sklearn.metrics
"""

import numpy as np


def _validar(y_true, y_pred):
    """Converte para arrays 1D e confere se os dois têm o mesmo número de amostras.

    Sem essa checagem, um array de 1 elemento passaria por broadcasting
    (ex.: [2] viraria [2, 2, 2]) e a métrica sairia errada sem nenhum erro.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true e y_pred precisam ter o mesmo número de amostras: "
            f"y_true tem {len(y_true)}, y_pred tem {len(y_pred)}."
        )
    return y_true, y_pred


def mean_squared_error(y_true,y_pred):
    y_true, y_pred = _validar(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = _validar(y_true, y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _validar(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    y_true, y_pred = _validar(y_true, y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)

def confusion_matrix(y_true, y_pred):
    y_true, y_pred = _validar(y_true, y_pred)
    n = len(y_true)

    todos = np.concatenate((y_true, y_pred))
    classes, posicoes = np.unique(todos, return_inverse=True)
    k = len(classes)

    y_true = posicoes[:n]
    y_pred = posicoes[n:]

    matriz = np.bincount(k * y_true + y_pred, minlength= k * k).reshape(k,k)
    return matriz
