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
                raise ValueError(
                    f"pos_label={pos_label!r} não é um rótulo válido. "
                    f"Deve ser um destes: {classes.tolist()}."
                )
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

def recall_score(y_true, y_pred, average = "binary", pos_label = 1, zero_division = 0):
    y_true, y_pred = _validar(y_true, y_pred)
    matriz = confusion_matrix(y_true, y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))

    recall = _divisao_segura(np.diag(matriz), matriz.sum(axis=1), zero_division)

    return _aplicar_average(recall, classes, average, pos_label, zero_division)

def f1_score(y_true, y_pred, average = "binary", pos_label = 1, zero_division = 0):
    y_true, y_pred = _validar(y_true, y_pred)
    matriz = confusion_matrix(y_true, y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))
    vp =np.diag(matriz)
    fp = matriz.sum(axis=0) - vp
    fn = matriz.sum(axis=1) - vp

    f1 = _divisao_segura(2 * vp, 2 * vp + fp + fn, zero_division)

    return _aplicar_average(f1, classes, average, pos_label, zero_division)

def log_loss(y_true, y_proba):
    """Log loss (entropia cruzada) a partir das probabilidades previstas.

    y_proba pode vir em dois formatos:
    - 1D, shape (n,): probabilidade da classe 1 (caso binário, rótulos 0 e 1);
    - 2D, shape (n, k): uma coluna por classe, como devolve o predict_proba.
      A coluna j corresponde à j-ésima classe de y_true em ordem crescente.

    Em todos os casos a conta é a mesma: pegar, em cada linha, a probabilidade
    que o modelo deu para a classe CERTA e fazer -média(log(...)).
    """
    eps = 1e-15
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba, dtype=float)

    # Uma coluna só (n, 1) é o mesmo que o formato 1D.
    if y_proba.ndim == 2 and y_proba.shape[1] == 1:
        y_proba = y_proba.ravel()

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"y_true e y_proba precisam ter o mesmo número de amostras: "
            f"y_true tem {len(y_true)}, y_proba tem {len(y_proba)}."
        )

    # Formato 1D (binário): a fórmula original, com y_true em 0 e 1.
    if y_proba.ndim == 1:
        p = np.clip(y_proba, eps, 1 - eps)
        return -np.mean((1 - y_true) * np.log(1 - p) + y_true * np.log(p))

    # Formato 2D: cada rótulo vira a posição da sua coluna (0, 1, ..., k-1).
    classes, coluna_certa = np.unique(y_true, return_inverse=True)
    if y_proba.shape[1] != len(classes):
        raise ValueError(
            f"y_proba tem {y_proba.shape[1]} colunas, mas y_true tem {len(classes)} "
            f"classes: {classes.tolist()}."
        )

    p = np.clip(y_proba, eps, 1 - eps)
    # Para cada linha i, a probabilidade da classe certa: p[i, coluna_certa[i]].
    p_classe_certa = p[np.arange(len(y_true)), coluna_certa]
    return -np.mean(np.log(p_classe_certa))
