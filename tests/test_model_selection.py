"""Testes de mlscratch.utils.model_selection.

O embaralhamento é diferente do sklearn, então não comparamos índices.
Testamos propriedades: tamanhos, disjunção, cobertura, alinhamento X-y e proporção de classes.
"""

import inspect

import numpy as np
import pytest
from sklearn.model_selection import KFold as SkKFold
from sklearn.model_selection import StratifiedKFold as SkStratifiedKFold
from sklearn.model_selection import train_test_split as sk_train_test_split

from mlscratch.utils.model_selection import KFold, StratifiedKFold, train_test_split


def dados_com_id(n):
    # A coluna 0 de X é o id da linha, e y = id * 10: dá para rastrear cada amostra.
    X = np.column_stack([np.arange(n), np.arange(n) * 2.0])
    y = np.arange(n) * 10
    return X, y


# ------------------------------------------------------- train_test_split


@pytest.mark.parametrize(
    "n, test_size", [(100, 0.25), (10, 0.25), (15, 0.25), (7, 0.3), (10, 0.3), (50, 0.2)]
)
def test_tamanhos_iguais_ao_sklearn(n, test_size):
    X, y = dados_com_id(n)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, rng=np.random.default_rng(0))
    sk = sk_train_test_split(X, y, test_size=test_size, random_state=0)
    assert len(X_te) == len(sk[1])
    assert len(X_tr) == len(sk[0])


def test_treino_e_teste_disjuntos_e_cobrem_tudo():
    X, y = dados_com_id(100)
    X_tr, X_te, _, _ = train_test_split(X, y, rng=np.random.default_rng(0))
    ids_tr, ids_te = set(X_tr[:, 0]), set(X_te[:, 0])
    assert ids_tr.isdisjoint(ids_te)
    assert ids_tr | ids_te == set(range(100))
    assert len(X_tr) + len(X_te) == 100  # sem linhas repetidas


def test_x_e_y_continuam_alinhados():
    X, y = dados_com_id(100)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, rng=np.random.default_rng(0))
    assert np.array_equal(y_tr, X_tr[:, 0] * 10)
    assert np.array_equal(y_te, X_te[:, 0] * 10)


def test_embaralha():
    X, y = dados_com_id(100)
    _, X_te, _, _ = train_test_split(X, y, rng=np.random.default_rng(0))
    # Se não embaralhasse, o teste seria um bloco contíguo de ids.
    assert not np.array_equal(np.sort(X_te[:, 0]), np.arange(X_te[0, 0], X_te[0, 0] + len(X_te)))


def test_mesmo_rng_mesma_divisao_e_rng_diferente_divisao_diferente():
    X, y = dados_com_id(100)
    a = train_test_split(X, y, rng=np.random.default_rng(42))
    b = train_test_split(X, y, rng=np.random.default_rng(42))
    c = train_test_split(X, y, rng=np.random.default_rng(7))
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_funciona_sem_passar_rng():
    X, y = dados_com_id(20)
    X_tr, X_te, _, _ = train_test_split(X, y)
    assert len(X_tr) + len(X_te) == 20


def test_shuffle_false_igual_ao_sklearn():
    # Sem embaralhar, o sklearn mantém a ordem: treino = primeiras linhas, teste = últimas.
    # (É o que se quer em séries temporais: treinar no passado, testar no futuro.)
    X, y = dados_com_id(20)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, shuffle=False)
    sk = sk_train_test_split(X, y, test_size=0.25, shuffle=False)
    assert np.array_equal(X_tr, sk[0])
    assert np.array_equal(X_te, sk[1])


def test_preserva_tipos_de_y():
    X, _ = dados_com_id(20)
    y_txt = np.array(["sim", "nao"] * 10)
    y_int = np.arange(20)
    assert train_test_split(X, y_txt, rng=np.random.default_rng(0))[2].dtype == y_txt.dtype
    assert train_test_split(X, y_int, rng=np.random.default_rng(0))[2].dtype == y_int.dtype


def test_aceita_listas():
    X = [[i, i * 2] for i in range(12)]
    y = [i % 2 for i in range(12)]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, rng=np.random.default_rng(0))
    assert (len(X_tr), len(X_te), len(y_tr), len(y_te)) == (9, 3, 9, 3)


def test_nao_modifica_entrada():
    X, y = dados_com_id(30)
    X_copia, y_copia = X.copy(), y.copy()
    train_test_split(X, y, rng=np.random.default_rng(0))
    assert np.array_equal(X, X_copia)
    assert np.array_equal(y, y_copia)


# ------------------------------------------------ train_test_split estratificado


@pytest.mark.parametrize(
    "proporcoes",
    [
        [80, 20],
        [50, 30, 20],
        pytest.param(
            [90, 5, 5],
            marks=pytest.mark.xfail(
                reason="Pendente: ceil por classe manda amostras demais das classes pequenas para o teste",
                strict=True,
            ),
        ),
    ],
)
def test_estratificado_preserva_proporcao(proporcoes):
    y = np.repeat(np.arange(len(proporcoes)), proporcoes)  # ex.: 80 zeros e 20 uns
    X = np.arange(len(y)).reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, rng=np.random.default_rng(0)
    )
    original = np.bincount(y) / len(y)
    # Proporção de cada classe no treino e no teste igual à original (tolerância de 1 amostra).
    assert np.allclose(np.bincount(y_tr, minlength=len(proporcoes)) / len(y_tr), original, atol=1 / len(y_tr))
    assert np.allclose(np.bincount(y_te, minlength=len(proporcoes)) / len(y_te), original, atol=1 / len(y_te))


def test_estratificado_disjunto_cobre_tudo_e_alinhado():
    X, y_ids = dados_com_id(100)
    classes = np.array([0] * 70 + [1] * 30)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_ids, stratify=classes, rng=np.random.default_rng(0)
    )
    assert set(X_tr[:, 0]).isdisjoint(X_te[:, 0])
    assert set(X_tr[:, 0]) | set(X_te[:, 0]) == set(range(100))
    assert np.array_equal(y_te, X_te[:, 0] * 10)


def test_estratificado_por_array_diferente_de_y():
    # Estratificar por um grupo (ex.: região) enquanto y é outra coisa.
    n = 100
    X = np.arange(n).reshape(-1, 1)
    y = np.arange(n) % 2
    regiao = np.array(["N"] * 20 + ["S"] * 80)
    X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.25, stratify=regiao, rng=np.random.default_rng(0))
    regiao_teste = regiao[X_te[:, 0]]
    assert np.mean(regiao_teste == "N") == pytest.approx(0.2, abs=1 / len(X_te))


def test_estratificado_tamanho_proximo_do_sklearn():
    # Com dezenas de amostras por classe, o tamanho do teste fica a no máximo 1 amostra por classe do sklearn.
    y = np.repeat([0, 1, 2], [45, 35, 20])
    X = np.arange(len(y)).reshape(-1, 1)
    meu = train_test_split(X, y, test_size=0.25, stratify=y, rng=np.random.default_rng(0))
    sk = sk_train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    assert abs(len(meu[1]) - len(sk[1])) <= 3


# ------------------------------------------------------------------- KFold


def test_kfold_split_e_um_gerador():
    assert inspect.isgenerator(KFold(n_splits=3).split(np.arange(10)))


def test_kfold_split_aceita_so_x():
    # No sklearn, y é opcional em split(X, y=None): o KFold não usa y.
    folds = list(KFold(n_splits=3).split(np.arange(10)))
    assert len(folds) == 3


def test_kfold_split_aceita_x_e_y():
    X, y = np.arange(10).reshape(-1, 1), np.arange(10) % 2
    assert len(list(KFold(n_splits=3).split(X, y))) == 3


@pytest.mark.parametrize("n, k", [(10, 5), (10, 3), (11, 3), (13, 5), (100, 10), (7, 7)])
def test_kfold_sem_shuffle_identico_ao_sklearn(n, k):
    # Sem embaralhar não há aleatoriedade: os índices têm que ser exatamente os do sklearn.
    X = np.arange(n).reshape(-1, 1)
    meus = list(KFold(n_splits=k).split(X))
    deles = list(SkKFold(n_splits=k).split(X))
    assert len(meus) == len(deles) == k
    for (tr, te), (sk_tr, sk_te) in zip(meus, deles):
        assert np.array_equal(te, sk_te)
        assert np.array_equal(tr, sk_tr)


@pytest.mark.parametrize("n, k", [(10, 3), (11, 3), (13, 5), (100, 10)])
def test_kfold_tamanhos_iguais_ao_sklearn_com_shuffle(n, k):
    X = np.arange(n).reshape(-1, 1)
    meus = [len(te) for _, te in KFold(n_splits=k, shuffle=True, rng=np.random.default_rng(0)).split(X)]
    deles = [len(te) for _, te in SkKFold(n_splits=k, shuffle=True, random_state=0).split(X)]
    assert meus == deles


def test_kfold_tamanhos_diferem_em_no_maximo_1():
    tamanhos = [len(te) for _, te in KFold(n_splits=4, shuffle=True, rng=np.random.default_rng(0)).split(np.arange(23))]
    assert max(tamanhos) - min(tamanhos) <= 1


def test_kfold_cada_indice_no_teste_exatamente_uma_vez():
    n = 23
    testes = [te for _, te in KFold(n_splits=4, shuffle=True, rng=np.random.default_rng(0)).split(np.arange(n))]
    todos = np.concatenate(testes)
    assert np.array_equal(np.sort(todos), np.arange(n))  # união = tudo, sem repetição


def test_kfold_treino_e_teste_de_cada_fold_disjuntos_e_completos():
    n = 23
    for tr, te in KFold(n_splits=4, shuffle=True, rng=np.random.default_rng(0)).split(np.arange(n)):
        assert set(tr).isdisjoint(te)
        assert set(tr) | set(te) == set(range(n))
        assert len(tr) + len(te) == n


def test_kfold_shuffle_embaralha():
    X = np.arange(30)
    sem = [te for _, te in KFold(n_splits=3).split(X)]
    com = [te for _, te in KFold(n_splits=3, shuffle=True, rng=np.random.default_rng(0)).split(X)]
    assert not all(np.array_equal(a, b) for a, b in zip(sem, com))


def test_kfold_mesmo_rng_mesmos_folds():
    X = np.arange(30)
    a = [te for _, te in KFold(n_splits=3, shuffle=True, rng=np.random.default_rng(42)).split(X)]
    b = [te for _, te in KFold(n_splits=3, shuffle=True, rng=np.random.default_rng(42)).split(X)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_kfold_get_n_splits():
    assert KFold(n_splits=4).get_n_splits() == 4


def test_kfold_aceita_lista():
    X = [[i, i * 2] for i in range(9)]
    assert [len(te) for _, te in KFold(n_splits=3).split(X)] == [3, 3, 3]


def test_kfold_n_splits_menor_que_2_lanca_erro():
    # O sklearn recusa com ValueError em vez de trocar o valor em silêncio.
    with pytest.raises(ValueError):
        KFold(n_splits=1)


def test_kfold_mais_folds_que_amostras_lanca_erro():
    # Com 5 amostras não dá para fazer 10 folds: alguns ficariam com teste vazio.
    with pytest.raises(ValueError):
        list(KFold(n_splits=10).split(np.arange(5)))


# --------------------------------------------------------- StratifiedKFold


def contagens_por_fold(folds, y):
    """Matriz (n_folds, n_classes): quantas amostras de cada classe cada fold tem no teste."""
    classes = np.unique(y)
    return np.array([[np.sum(y[te] == c) for c in classes] for _, te in folds])


def test_stratified_split_e_um_gerador():
    y = np.array([0, 1] * 10)
    assert inspect.isgenerator(StratifiedKFold(n_splits=2).split(np.zeros(20), y))


@pytest.mark.parametrize(
    "proporcoes, k",
    [([80, 20], 5), ([90, 10], 5), ([50, 30, 20], 4), ([45, 35, 20], 3), ([6, 3], 3), ([13, 7, 4], 4)],
)
@pytest.mark.parametrize("shuffle", [False, True])
def test_stratified_cada_classe_dividida_igualmente_entre_folds(proporcoes, k, shuffle):
    y = np.repeat(np.arange(len(proporcoes)), proporcoes)
    folds = list(StratifiedKFold(n_splits=k, shuffle=shuffle, rng=np.random.default_rng(0)).split(np.zeros(len(y)), y))
    contagens = contagens_por_fold(folds, y)
    # Cada classe aparece em todos os folds, com diferença de no máximo 1 amostra entre folds.
    assert (contagens.max(axis=0) - contagens.min(axis=0) <= 1).all()


@pytest.mark.parametrize("proporcoes, k", [([80, 20], 5), ([50, 30, 20], 4), ([45, 35, 20], 3), ([13, 7, 4], 4)])
def test_stratified_contagens_por_classe_iguais_ao_sklearn(proporcoes, k):
    # Os índices diferem, mas quantas amostras de cada classe vão para cada fold tem que bater
    # (comparando as contagens ordenadas, porque a ordem dos folds pode ser outra).
    y = np.repeat(np.arange(len(proporcoes)), proporcoes)
    X = np.zeros(len(y))
    meus = contagens_por_fold(list(StratifiedKFold(n_splits=k).split(X, y)), y)
    deles = contagens_por_fold(list(SkStratifiedKFold(n_splits=k).split(X, y)), y)
    assert np.array_equal(np.sort(meus, axis=0), np.sort(deles, axis=0))


def test_stratified_tamanhos_diferem_em_no_maximo_1():
    y = np.repeat([0, 1, 2], [13, 7, 4])
    tamanhos = [len(te) for _, te in StratifiedKFold(n_splits=4).split(np.zeros(len(y)), y)]
    assert max(tamanhos) - min(tamanhos) <= 1


@pytest.mark.parametrize("shuffle", [False, True])
def test_stratified_cada_indice_no_teste_exatamente_uma_vez(shuffle):
    y = np.repeat([0, 1, 2], [13, 7, 4])
    n = len(y)
    folds = list(StratifiedKFold(n_splits=4, shuffle=shuffle, rng=np.random.default_rng(0)).split(np.zeros(n), y))
    assert np.array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(n))
    for tr, te in folds:
        assert set(tr).isdisjoint(te)
        assert set(tr) | set(te) == set(range(n))


def test_stratified_resolve_o_problema_do_kfold_com_dados_ordenados():
    # Dados ordenados por classe (comum em datasets reais): o KFold sem shuffle
    # gera folds sem nenhum exemplo da classe minoritária; o estratificado não.
    y = np.array([0] * 90 + [1] * 10)
    X = np.zeros(100)
    minoritaria_kfold = [np.sum(y[te] == 1) for _, te in KFold(n_splits=5).split(X)]
    minoritaria_strat = [np.sum(y[te] == 1) for _, te in StratifiedKFold(n_splits=5).split(X, y)]
    assert 0 in minoritaria_kfold
    assert minoritaria_strat == [2, 2, 2, 2, 2]


def test_stratified_shuffle_embaralha_e_mesmo_rng_reproduz():
    y = np.repeat([0, 1], [30, 15])
    X = np.zeros(len(y))
    sem = [te for _, te in StratifiedKFold(n_splits=3).split(X, y)]
    a = [te for _, te in StratifiedKFold(n_splits=3, shuffle=True, rng=np.random.default_rng(42)).split(X, y)]
    b = [te for _, te in StratifiedKFold(n_splits=3, shuffle=True, rng=np.random.default_rng(42)).split(X, y)]
    assert not all(np.array_equal(x, z) for x, z in zip(sem, a))
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_stratified_aceita_rotulos_de_texto():
    y = np.array(["sim"] * 12 + ["nao"] * 6)
    folds = list(StratifiedKFold(n_splits=3).split(np.zeros(len(y)), y))
    for _, te in folds:
        assert np.sum(y[te] == "sim") == 4
        assert np.sum(y[te] == "nao") == 2


def test_stratified_get_n_splits():
    assert StratifiedKFold(n_splits=4).get_n_splits() == 4


def test_stratified_n_splits_menor_que_2_lanca_erro():
    with pytest.raises(ValueError):
        StratifiedKFold(n_splits=1)


def test_stratified_mais_folds_que_amostras_lanca_erro():
    with pytest.raises(ValueError):
        list(StratifiedKFold(n_splits=10).split(np.zeros(5), np.array([0, 1, 0, 1, 0])))
