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


def _divisao_segura(numerador, denominador, zero_division=0.0):
    """Divide elemento a elemento; onde o denominador é 0, devolve zero_division.

    A divisão só é feita nas posições com denominador diferente de 0
    (parâmetro where do np.divide), então não há nan nem aviso de divisão por zero.
    """
    numerador = np.asarray(numerador, dtype=float)
    denominador = np.asarray(denominador, dtype=float)
    return np.divide(
        numerador,
        denominador,
        out=np.full(numerador.shape, float(zero_division)),
        where=denominador != 0,
    )


def _aplicar_average(valores, classes, average, pos_label, zero_division):
    """Reduz o array com o valor de cada classe conforme o average.

    valores: precision, recall ou F1 de cada classe, na mesma ordem de `classes`
             (a ordem das linhas e colunas da matriz de confusão).

    - None:     devolve o array inteiro, um valor por classe;
    - "macro":  devolve a média simples entre as classes;
    - "binary": devolve só o valor da classe pos_label.
    """
    if average is None:
        return valores

    if average == "macro":
        return float(np.mean(valores))

    if average == "binary":
        if len(classes) > 2:
            raise ValueError(
                f"average='binary' exige no máximo 2 classes, mas foram encontradas "
                f"{len(classes)}: {classes.tolist()}. Use average='macro' ou average=None."
            )
        posicoes = np.where(classes == pos_label)[0]
        if len(posicoes) == 0:
            if len(classes) == 2:
                # Duas classes e nenhuma é o pos_label: provavelmente rótulos de texto
                # sem pos_label informado (ex.: "sim"/"nao" com pos_label=1).
                raise ValueError(
                    f"pos_label={pos_label!r} não é um rótulo válido. "
                    f"Deve ser um destes: {classes.tolist()}."
                )
            # Só uma classe aparece e não é a positiva: VP = FP = 0, ou seja, 0/0.
            return float(zero_division)
        return float(valores[posicoes[0]])

    raise ValueError(f"average deve ser 'binary', 'macro' ou None, recebeu {average!r}.")


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

def accuracy_score(y_true, y_pred):
    y_true, y_pred = _validar(y_true, y_pred)

    matriz = confusion_matrix(y_true, y_pred)

    return np.trace(matriz) / np.sum(matriz)

def precision_score(y_true, y_pred, average = "binary", pos_label = 1, zero_division = 0):
    y_true, y_pred = _validar(y_true, y_pred)
    matriz = confusion_matrix(y_true, y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))

    precision = _divisao_segura(np.diag(matriz), matriz.sum(axis=0), zero_division)

    return _aplicar_average(precision, classes, average, pos_label, zero_division)

    