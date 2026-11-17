"""Testes de mlscratch.utils.preprocessing contra o scikit-learn."""

import numpy as np
import pytest
from sklearn import preprocessing as skp

from mlscratch.utils.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


@pytest.fixture
def dados():
    rng = np.random.default_rng(0)
    # Colunas com escalas bem diferentes, como em dados reais.
    X_treino = rng.normal(loc=[0, 50, -3], scale=[1, 10, 0.1], size=(100, 3))
    X_teste = rng.normal(loc=[0, 50, -3], scale=[1, 10, 0.1], size=(30, 3))
    return X_treino, X_teste


# ---------------------------------------------------------- StandardScaler


def test_standard_fit_retorna_self(dados):
    X_treino, _ = dados
    scaler = StandardScaler()
    assert scaler.fit(X_treino) is scaler


def test_standard_atributos_aprendidos_paridade(dados):
    X_treino, _ = dados
    meu = StandardScaler().fit(X_treino)
    sk = skp.StandardScaler().fit(X_treino)
    assert np.allclose(meu.mean_, sk.mean_)
    assert np.allclose(meu.scale_, sk.scale_)


def test_standard_transform_paridade(dados):
    X_treino, _ = dados
    meu = StandardScaler().fit(X_treino)
    sk = skp.StandardScaler().fit(X_treino)
    assert np.allclose(meu.transform(X_treino), sk.transform(X_treino))


def test_standard_saida_com_media_0_e_desvio_1(dados):
    X_treino, _ = dados
    Xs = StandardScaler().fit_transform(X_treino)
    assert np.allclose(Xs.mean(axis=0), 0.0)
    assert np.allclose(Xs.std(axis=0), 1.0)


def test_standard_teste_usa_estatisticas_do_treino(dados):
    # Sem data leakage: o transform do teste usa a média e o desvio aprendidos no treino.
    X_treino, X_teste = dados
    meu = StandardScaler().fit(X_treino)
    sk = skp.StandardScaler().fit(X_treino)
    assert np.allclose(meu.transform(X_teste), sk.transform(X_teste))
    # E o transform não pode reajustar nada:
    media_antes = meu.mean_.copy()
    meu.transform(X_teste)
    assert np.array_equal(meu.mean_, media_antes)


def test_standard_fit_transform_igual_fit_depois_transform(dados):
    X_treino, _ = dados
    a = StandardScaler().fit_transform(X_treino)
    b = StandardScaler().fit(X_treino).transform(X_treino)
    assert np.allclose(a, b)


def test_standard_feature_constante():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    meu = StandardScaler().fit(X)
    sk = skp.StandardScaler().fit(X)
    Xs = meu.transform(X)
    assert meu.scale_[1] == 1.0
    assert not np.isnan(Xs).any()
    assert np.allclose(Xs[:, 1], 0.0)
    assert np.allclose(Xs, sk.transform(X))


def test_standard_inverse_transform_volta_ao_original(dados):
    X_treino, X_teste = dados
    meu = StandardScaler().fit(X_treino)
    sk = skp.StandardScaler().fit(X_treino)
    assert np.allclose(meu.inverse_transform(meu.transform(X_teste)), X_teste)
    Z = meu.transform(X_teste)
    assert np.allclose(meu.inverse_transform(Z), sk.inverse_transform(Z))


def test_standard_aceita_lista_e_inteiros():
    X = [[1, 2], [3, 4], [5, 6]]
    meu = StandardScaler().fit(X)
    sk = skp.StandardScaler().fit(np.array(X, dtype=float))
    assert np.allclose(meu.transform(X), sk.transform(np.array(X, dtype=float)))


# ------------------------------------------------------------ MinMaxScaler


def test_minmax_fit_retorna_self(dados):
    X_treino, _ = dados
    scaler = MinMaxScaler()
    assert scaler.fit(X_treino) is scaler


@pytest.mark.parametrize("feature_range", [(0, 1), (-1, 1), (2, 5)])
def test_minmax_atributos_aprendidos_paridade(dados, feature_range):
    X_treino, _ = dados
    meu = MinMaxScaler(feature_range=feature_range).fit(X_treino)
    sk = skp.MinMaxScaler(feature_range=feature_range).fit(X_treino)
    for attr in ["data_min_", "data_max_", "data_range_", "scale_", "min_"]:
        assert np.allclose(getattr(meu, attr), getattr(sk, attr)), attr


@pytest.mark.parametrize("feature_range", [(0, 1), (-1, 1), (2, 5)])
def test_minmax_transform_paridade_e_intervalo(dados, feature_range):
    X_treino, _ = dados
    a, b = feature_range
    meu = MinMaxScaler(feature_range=feature_range).fit(X_treino)
    sk = skp.MinMaxScaler(feature_range=feature_range).fit(X_treino)
    Xs = meu.transform(X_treino)
    assert np.allclose(Xs, sk.transform(X_treino))
    assert np.allclose(Xs.min(axis=0), a)
    assert np.allclose(Xs.max(axis=0), b)


def test_minmax_teste_usa_estatisticas_do_treino(dados):
    # Valores do teste fora do intervalo do treino podem sair de [0, 1]; isso é esperado.
    X_treino, X_teste = dados
    meu = MinMaxScaler().fit(X_treino)
    sk = skp.MinMaxScaler().fit(X_treino)
    assert np.allclose(meu.transform(X_teste), sk.transform(X_teste))


def test_minmax_feature_constante():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    meu = MinMaxScaler().fit(X)
    sk = skp.MinMaxScaler().fit(X)
    Xs = meu.transform(X)
    assert np.isfinite(Xs).all()
    assert np.allclose(Xs, sk.transform(X))


def test_minmax_fit_transform_igual_fit_depois_transform(dados):
    X_treino, _ = dados
    a = MinMaxScaler().fit_transform(X_treino)
    b = MinMaxScaler().fit(X_treino).transform(X_treino)
    assert np.allclose(a, b)


@pytest.mark.parametrize("feature_range", [(0, 1), (-1, 1)])
def test_minmax_inverse_transform_volta_ao_original(dados, feature_range):
    X_treino, X_teste = dados
    meu = MinMaxScaler(feature_range=feature_range).fit(X_treino)
    assert np.allclose(meu.inverse_transform(meu.transform(X_teste)), X_teste)


def test_minmax_aceita_lista_e_inteiros():
    X = [[1, 10], [3, 20], [5, 40]]
    meu = MinMaxScaler().fit(X)
    sk = skp.MinMaxScaler().fit(np.array(X, dtype=float))
    assert np.allclose(meu.transform(X), sk.transform(np.array(X, dtype=float)))


# ----------------------------------------------------------- OneHotEncoder


@pytest.fixture
def categorias():
    # Duas colunas com números diferentes de categorias, de tipos diferentes.
    X_treino = np.array(
        [["SP", 1], ["RJ", 2], ["SP", 3], ["MG", 1], ["RJ", 2]], dtype=object
    )
    X_teste = np.array([["MG", 3], ["SP", 1]], dtype=object)
    return X_treino, X_teste


def sk_encoder(**kwargs):
    # O sklearn retorna matriz esparsa por padrão; aqui pedimos array normal para comparar.
    return skp.OneHotEncoder(sparse_output=False, **kwargs)


def test_onehot_fit_retorna_self(categorias):
    X_treino, _ = categorias
    enc = OneHotEncoder()
    assert enc.fit(X_treino) is enc


def test_onehot_categories_paridade(categorias):
    X_treino, _ = categorias
    meu = OneHotEncoder().fit(X_treino)
    sk = sk_encoder().fit(X_treino)
    assert len(meu.categories_) == len(sk.categories_)
    for minhas, delas in zip(meu.categories_, sk.categories_):
        assert list(minhas) == list(delas)


def test_onehot_transform_paridade(categorias):
    X_treino, X_teste = categorias
    meu = OneHotEncoder().fit(X_treino)
    sk = sk_encoder().fit(X_treino)
    assert np.array_equal(meu.transform(X_treino), sk.transform(X_treino))
    assert np.array_equal(meu.transform(X_teste), sk.transform(X_teste))


def test_onehot_formato_e_um_1_por_coluna_original(categorias):
    X_treino, _ = categorias
    Xe = OneHotEncoder().fit_transform(X_treino)
    # 3 estados + 3 números = 6 colunas
    assert Xe.shape == (5, 6)
    assert set(np.unique(Xe)) <= {0.0, 1.0}
    # Cada linha tem exatamente um 1 em cada bloco: 2 colunas originais -> soma 2
    assert np.array_equal(Xe.sum(axis=1), np.full(5, 2.0))


def test_onehot_fit_transform_igual_fit_depois_transform(categorias):
    X_treino, _ = categorias
    a = OneHotEncoder().fit_transform(X_treino)
    b = OneHotEncoder().fit(X_treino).transform(X_treino)
    assert np.array_equal(a, b)


def test_onehot_categoria_desconhecida_com_error_lanca_valueerror(categorias):
    X_treino, _ = categorias
    enc = OneHotEncoder(handle_unknown="error").fit(X_treino)
    with pytest.raises(ValueError):
        enc.transform(np.array([["BA", 1]], dtype=object))


def test_onehot_error_e_o_padrao(categorias):
    X_treino, _ = categorias
    enc = OneHotEncoder().fit(X_treino)
    with pytest.raises(ValueError):
        enc.transform(np.array([["SP", 99]], dtype=object))


def test_onehot_categoria_desconhecida_com_ignore_vira_zeros(categorias):
    X_treino, _ = categorias
    X_novo = np.array([["BA", 1], ["SP", 99]], dtype=object)
    meu = OneHotEncoder(handle_unknown="ignore").fit(X_treino)
    sk = sk_encoder(handle_unknown="ignore").fit(X_treino)
    Xe = meu.transform(X_novo)
    assert np.array_equal(Xe, sk.transform(X_novo))
    assert np.array_equal(Xe[0, :3], [0, 0, 0])  # "BA" desconhecido: bloco do estado zerado
    assert np.array_equal(Xe[1, 3:], [0, 0, 0])  # 99 desconhecido: bloco do número zerado


def test_onehot_transform_nao_altera_categorias(categorias):
    X_treino, _ = categorias
    enc = OneHotEncoder(handle_unknown="ignore").fit(X_treino)
    antes = [list(c) for c in enc.categories_]
    enc.transform(np.array([["BA", 99]], dtype=object))
    assert [list(c) for c in enc.categories_] == antes


def test_onehot_aceita_lista_de_strings():
    X = [["a", "x"], ["b", "y"], ["a", "y"]]
    meu = OneHotEncoder().fit(X)
    sk = sk_encoder().fit(X)
    assert np.array_equal(meu.transform(X), sk.transform(X))
