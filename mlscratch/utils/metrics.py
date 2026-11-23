"""Métricas (Fase 1, Semana 3, Sessões 2 e 3).

Regressão:
- mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

Classificação:
- confusion_matrix  (via np.bincount)
- accuracy_score, precision_score, recall_score, f1_score  (binário e macro; zero_division)
- log_loss          (clip em [eps, 1 - eps])
- roc_curve, auc    (curva ROC e área pela regra do trapézio)
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
    eps = 1e-15
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim == 2 and y_proba.shape[1] == 1:
        y_proba = y_proba.ravel()

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"y_true e y_proba precisam ter o mesmo número de amostras: "
            f"y_true tem {len(y_true)}, y_proba tem {len(y_proba)}."
        )

    # Formato 1D (binário)
    if y_proba.ndim == 1:
        p = np.clip(y_proba, eps, 1 - eps)
        return -np.mean((1 - y_true) * np.log(1 - p) + y_true * np.log(p))

    # Formato 2D:
    classes, coluna_certa = np.unique(y_true, return_inverse=True)
    if y_proba.shape[1] != len(classes):
        raise ValueError(
            f"y_proba tem {y_proba.shape[1]} colunas, mas y_true tem {len(classes)} "
            f"classes: {classes.tolist()}."
        )

    p = np.clip(y_proba, eps, 1 - eps)
    # Para cada linha i, a probabilidade da classe certa: p[i, coluna_certa[i]]
    p_classe_certa = p[np.arange(len(y_true)), coluna_certa]
    return -np.mean(np.log(p_classe_certa))


def _positivos_binarios(y_true):
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            f"A ROC-AUC precisa de exatamente 2 classes em y_true, "
            f"mas foram encontradas {len(classes)}: {classes.tolist()}."
        )
    return (y_true == classes[1]).astype(int)


def roc_curve(y_true, y_score):
    y_true, y_score = _validar(y_true, y_score)
    y_score = y_score.astype(float)
    positivo = _positivos_binarios(y_true)

    ordem = np.argsort(-y_score, kind="mergesort")
    scores = y_score[ordem]
    positivo = positivo[ordem]

    fim_do_grupo = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]

    vp = np.cumsum(positivo)[fim_do_grupo] 
    fp = (fim_do_grupo + 1) - vp         

    vp = np.r_[0, vp]
    fp = np.r_[0, fp]
    thresholds = np.r_[np.inf, scores[fim_do_grupo]]

    tpr = vp / vp[-1]  
    fpr = fp / fp[-1]  
    return fpr, tpr, thresholds

def auc(x, y):
    """Área sob uma curva pela regra do trapézio (x crescente)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    larguras = x[1:] - x[:-1]
    alturas_medias = (y[1:] + y[:-1]) / 2
    return float(np.sum(larguras * alturas_medias))


def _ranks_medios(valores):
    unicos, grupo, contagem = np.unique(valores, return_inverse=True, return_counts=True)
    inicio = np.cumsum(contagem) - contagem
    rank_medio_do_grupo = inicio + (contagem + 1) / 2
    return rank_medio_do_grupo[grupo]


def roc_auc_score(y_true, y_score, metodo="ranks"):
    if metodo == "curva":
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return auc(fpr, tpr)

    if metodo != "ranks":
        raise ValueError(f"metodo deve ser 'ranks' ou 'curva', recebeu {metodo!r}.")

    y_true, y_score = _validar(y_true, y_score)
    positivo = _positivos_binarios(y_true)
    n_pos = positivo.sum()
    n_neg = len(positivo) - n_pos

    ranks = _ranks_medios(y_score.astype(float))
    soma_ranks_positivos = ranks[positivo == 1].sum()
    return float((soma_ranks_positivos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

