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


# ---------------------------------------------------------------- Accuracy

# Exemplos com valores calculados à mão (e conferidos com o sklearn).
BIN_TRUE = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
BIN_PRED = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]   # VP=3, FP=2, FN=1, VN=4
MC_TRUE = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
MC_PRED = [0, 0, 0, 1, 1, 2, 2, 2, 0, 0]


def test_accuracy_exemplos():
    assert np.isclose(m.accuracy_score(BIN_TRUE, BIN_PRED), 0.7)
    assert np.isclose(m.accuracy_score(MC_TRUE, MC_PRED), 0.5)


@pytest.mark.parametrize("k", [2, 3, 10])
def test_accuracy_paridade_com_sklearn(k):
    rng = np.random.default_rng(k)
    y_true = rng.integers(0, k, 300)
    y_pred = np.where(rng.random(300) < 0.6, y_true, rng.integers(0, k, 300))
    assert np.isclose(m.accuracy_score(y_true, y_pred), skm.accuracy_score(y_true, y_pred))


def test_accuracy_perfeita_zero_e_texto():
    assert m.accuracy_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert m.accuracy_score([0, 0, 1], [1, 1, 0]) == 0.0
    assert np.isclose(m.accuracy_score(["a", "b", "a"], ["a", "a", "a"]), 2 / 3)


# --------------------------------------------- Precision, recall e F1

PRF = [
    (m.precision_score, skm.precision_score),
    (m.recall_score, skm.recall_score),
    (m.f1_score, skm.f1_score),
]
IDS_PRF = ["precision", "recall", "f1"]


def sk_sem_aviso(funcao, *args, **kwargs):
    # O sklearn emite UndefinedMetricWarning nos casos 0/0; aqui só queremos o valor.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return funcao(*args, **kwargs)


@pytest.mark.parametrize(
    "minha, esperado",
    [(m.precision_score, 0.6), (m.recall_score, 0.75), (m.f1_score, 2 * 0.6 * 0.75 / 1.35)],
    ids=IDS_PRF,
)
def test_binario_exemplo_calculado_a_mao(minha, esperado):
    assert np.isclose(minha(BIN_TRUE, BIN_PRED), esperado)


@pytest.mark.parametrize(
    "minha, por_classe",
    [
        (m.precision_score, [0.6, 0.5, 1 / 3]),
        (m.recall_score, [0.75, 1 / 3, 1 / 3]),
        (m.f1_score, [2 / 3, 0.4, 1 / 3]),
    ],
    ids=IDS_PRF,
)
def test_multiclasse_exemplo_calculado_a_mao(minha, por_classe):
    assert np.allclose(minha(MC_TRUE, MC_PRED, average=None), por_classe)
    assert np.isclose(minha(MC_TRUE, MC_PRED, average="macro"), np.mean(por_classe))


def test_f1_macro_e_media_dos_f1_nao_f1_das_medias():
    # Armadilha comum: calcular o F1 a partir da precision macro e do recall macro.
    p = m.precision_score(MC_TRUE, MC_PRED, average="macro")
    r = m.recall_score(MC_TRUE, MC_PRED, average="macro")
    f1_das_medias = 2 * p * r / (p + r)                       # ~0.4750 (errado)
    f1_macro = m.f1_score(MC_TRUE, MC_PRED, average="macro")  # ~0.4667 (certo)
    assert np.isclose(f1_macro, skm.f1_score(MC_TRUE, MC_PRED, average="macro"))
    assert not np.isclose(f1_macro, f1_das_medias)


@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_binario_paridade_com_sklearn(minha, sk, seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, 200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, 1 - y_true)
    assert np.isclose(minha(y_true, y_pred), sk(y_true, y_pred))


@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
@pytest.mark.parametrize("k", [3, 5])
def test_macro_e_none_paridade_com_sklearn(minha, sk, k):
    rng = np.random.default_rng(k)
    y_true = rng.integers(0, k, 300)
    y_pred = np.where(rng.random(300) < 0.6, y_true, rng.integers(0, k, 300))
    assert np.isclose(minha(y_true, y_pred, average="macro"), sk(y_true, y_pred, average="macro"))
    assert np.allclose(minha(y_true, y_pred, average=None), sk(y_true, y_pred, average=None))


@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
def test_binario_e_macro_devolvem_numero(minha, sk):
    assert np.ndim(minha(BIN_TRUE, BIN_PRED)) == 0
    assert np.ndim(minha(MC_TRUE, MC_PRED, average="macro")) == 0


@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
def test_pos_label_zero(minha, sk):
    # Tratar a classe 0 como positiva.
    assert np.isclose(minha(BIN_TRUE, BIN_PRED, pos_label=0), sk(BIN_TRUE, BIN_PRED, pos_label=0))


@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
def test_rotulos_de_texto_com_pos_label(minha, sk):
    y_true = ["fraude", "normal", "normal", "fraude", "normal"]
    y_pred = ["fraude", "fraude", "normal", "normal", "normal"]
    assert np.isclose(minha(y_true, y_pred, pos_label="fraude"), sk(y_true, y_pred, pos_label="fraude"))


@pytest.mark.filterwarnings("error::RuntimeWarning")  # divisão por zero vira falha
@pytest.mark.parametrize("zero_division", [0, 1])
@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([0, 1, 0, 1], [0, 0, 0, 0]),   # classe 1 nunca prevista: precision 0/0
        ([0, 0, 0, 0], [0, 1, 0, 1]),   # classe 1 nunca real: recall 0/0
        ([0, 0, 0], [0, 0, 0]),         # só a classe 0 aparece: pos_label=1 ausente
    ],
    ids=["nunca_prevista", "nunca_real", "so_classe_0"],
)
def test_zero_division(minha, sk, y_true, y_pred, zero_division):
    meu = minha(y_true, y_pred, zero_division=zero_division)
    assert np.isclose(meu, sk_sem_aviso(sk, y_true, y_pred, zero_division=zero_division))


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
def test_zero_division_no_macro(minha, sk):
    y_true, y_pred = [0, 1, 2, 0], [0, 0, 0, 0]   # classes 1 e 2 nunca previstas
    for zd in [0, 1]:
        meu = minha(y_true, y_pred, average="macro", zero_division=zd)
        assert np.isclose(meu, sk_sem_aviso(sk, y_true, y_pred, average="macro", zero_division=zd))


@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
def test_binary_com_3_classes_lanca_erro(minha, sk):
    with pytest.raises(ValueError):
        minha(MC_TRUE, MC_PRED)


@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
def test_texto_sem_pos_label_lanca_erro(minha, sk):
    with pytest.raises(ValueError):
        minha(["sim", "nao", "sim"], ["sim", "sim", "nao"])


@pytest.mark.parametrize("minha, sk", PRF, ids=IDS_PRF)
def test_average_invalido_lanca_erro(minha, sk):
    with pytest.raises(ValueError):
        minha(BIN_TRUE, BIN_PRED, average="media")


# ---------------------------------------------------------------- Log loss


def test_log_loss_valor_calculado_a_mao():
    # Classe 1 com p=0.8 -> -log(0.8); classe 0 com p=0.3 -> -log(1 - 0.3)
    esperado = -(np.log(0.8) + np.log(0.7)) / 2
    assert np.isclose(m.log_loss([1, 0], [0.8, 0.3]), esperado)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_log_loss_paridade_com_sklearn(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, 200)
    y_proba = np.clip(y_true * 0.6 + rng.random(200) * 0.4, 0.01, 0.99)
    assert np.isclose(m.log_loss(y_true, y_proba), skm.log_loss(y_true, y_proba))


def test_log_loss_chute_de_50_porcento_da_log_2():
    # Sem informação nenhuma (p = 0.5 sempre), a log loss é log(2) ~ 0.693.
    assert np.isclose(m.log_loss([0, 1, 1, 0], [0.5] * 4), np.log(2))


def test_log_loss_previsao_perfeita_quase_zero():
    assert np.isclose(m.log_loss([1, 0, 1], [1.0, 0.0, 1.0]), 0.0, atol=1e-12)


def test_log_loss_confiante_e_errado_e_pior_que_inseguro():
    # A log loss pune muito mais quem erra com certeza.
    assert m.log_loss([1], [0.01]) > m.log_loss([1], [0.4]) > m.log_loss([1], [0.9])


@pytest.mark.filterwarnings("error::RuntimeWarning")  # log(0) vira falha
def test_log_loss_probabilidades_0_e_1_sem_log_de_zero():
    # Sem o clip, log(0) = -inf. Com o clip, o resultado é grande mas finito.
    valor = m.log_loss([1, 0], [0.0, 1.0])
    assert np.isfinite(valor)
    assert valor > 10


def test_log_loss_aceita_listas_e_coluna():
    y_true, y_proba = [1, 0, 1, 1, 0], [0.9, 0.2, 0.6, 0.8, 0.1]
    esperado = skm.log_loss(y_true, y_proba)
    assert np.isclose(m.log_loss(y_true, y_proba), esperado)
    assert np.isclose(m.log_loss(y_true, np.array(y_proba).reshape(-1, 1)), esperado)


def test_log_loss_tamanhos_diferentes_lanca_erro():
    with pytest.raises(ValueError):
        m.log_loss([1, 0, 1], [0.5])


def test_log_loss_binario_com_duas_colunas():
    # predict_proba devolve shape (n, 2): [P(classe 0), P(classe 1)] em cada linha.
    y_true = [1, 0, 1, 1, 0]
    p1 = np.array([0.9, 0.2, 0.6, 0.8, 0.1])
    y_proba = np.column_stack([1 - p1, p1])
    assert np.isclose(m.log_loss(y_true, y_proba), skm.log_loss(y_true, y_proba))


def test_log_loss_multiclasse():
    y_true = [0, 1, 2, 2]
    y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]])
    assert np.isclose(m.log_loss(y_true, y_proba), skm.log_loss(y_true, y_proba))


def test_log_loss_multiclasse_valor_calculado_a_mao():
    # Probabilidades da classe certa em cada linha: 0.7, 0.8, 0.6 e 0.4.
    y_true = [0, 1, 2, 2]
    y_proba = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]]
    esperado = -np.mean(np.log([0.7, 0.8, 0.6, 0.4]))
    assert np.isclose(m.log_loss(y_true, y_proba), esperado)


def test_log_loss_multiclasse_rotulos_de_texto():
    # As colunas seguem a ordem das classes: "baixo", "medio", "alto" -> ordem alfabética.
    y_true = ["medio", "alto", "baixo"]
    y_proba = [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]  # colunas: alto, baixo, medio
    assert np.isclose(m.log_loss(y_true, y_proba), skm.log_loss(y_true, y_proba))


@pytest.mark.parametrize("seed", [0, 1])
def test_log_loss_multiclasse_paridade_com_sklearn(seed):
    from mlscratch.utils.numerics import softmax

    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 4, 100)
    y_proba = softmax(rng.normal(size=(100, 4)) * 2)  # probabilidades válidas, somando 1
    assert np.isclose(m.log_loss(y_true, y_proba), skm.log_loss(y_true, y_proba))


def test_log_loss_um_e_dois_formatos_dao_o_mesmo_resultado():
    y_true = [1, 0, 1, 1, 0]
    p1 = np.array([0.9, 0.2, 0.6, 0.8, 0.1])
    assert np.isclose(m.log_loss(y_true, p1), m.log_loss(y_true, np.column_stack([1 - p1, p1])))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_log_loss_multiclasse_probabilidade_zero_sem_log_de_zero():
    valor = m.log_loss([0, 1], [[0.0, 1.0], [1.0, 0.0]])  # 100% de certeza na classe errada
    assert np.isfinite(valor) and valor > 10


def test_log_loss_numero_de_colunas_errado_lanca_erro():
    with pytest.raises(ValueError):
        m.log_loss([0, 1, 2], [[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])  # 3 classes, 2 colunas
