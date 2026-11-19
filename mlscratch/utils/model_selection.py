"""Divisão de dados (Fase 1, Semana 3, Sessão 1).

- train_test_split(X, y, test_size, stratify, shuffle, rng)
- KFold            (gerador: yield train_idx, test_idx)
- StratifiedKFold  (gerador: yield train_idx, test_idx)

Testar propriedades, não índices: folds disjuntos, união = todos os índices,
tamanhos diferindo em no máximo 1, proporção de classes preservada.
"""
import numpy as np
import math

def train_test_split(X, y, test_size=0.25, stratify=None, shuffle=True, rng=None):
    if rng is None:                      
        rng = np.random.default_rng()

    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    if stratify is not None:               
        stratify = np.asarray(stratify)
        idx_treino, idx_teste = [], []
        for cls in np.unique(stratify):                
            idx_cls = np.where(stratify == cls)[0]     
            idx_cls = rng.permutation(idx_cls)
            corte = math.ceil(round(len(idx_cls) * test_size, 10))
            idx_treino.extend(idx_cls[:len(idx_cls) - corte])
            idx_teste.extend(idx_cls[len(idx_cls) - corte:])

        idx_treino = rng.permutation(idx_treino)
        idx_teste = rng.permutation(idx_teste)
    else:
        indices = rng.permutation(n) if shuffle else np.arange(n)
        corte = math.ceil(round(n * test_size, 10))
        idx_treino = indices[:n - corte]
        idx_teste = indices[n - corte:]

    return X[idx_treino], X[idx_teste], y[idx_treino], y[idx_teste]

class KFold:
    def __init__(self, n_splits = 5, shuffle = False, rng = None):
        self.n_splits = n_splits
        self.rng = rng
        self.shuffle = shuffle
        if rng is None:                      
                self.rng = np.random.default_rng()
        if n_splits < 2:
            raise ValueError(
                f"n_splits precisa ser pelo menos 2, recebeu n_splits={n_splits}."
            )
        

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y = None):
        X = np.asarray(X)
        n = len(X)
        if self.n_splits > n:
            raise ValueError(
                f"n_splits={self.n_splits} não pode ser maior que o número de amostras ({n})."
            )
        tamanhos = np.full(self.n_splits, n // self.n_splits)
        tamanhos[: n % self.n_splits] += 1

        indices = self.rng.permutation(n) if self.shuffle else np.arange(n)

        inicio = 0

        for tamanho in tamanhos:
             fim = inicio + tamanho
             teste = indices[inicio:fim]
             treino = np.concatenate((indices[:inicio], indices[fim:]))
             yield treino, teste
             inicio = fim

class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, rng=None):
        self.n_splits = n_splits
        self.rng = rng
        self.shuffle = shuffle
        if rng is None:
            self.rng = np.random.default_rng()
        if n_splits < 2:
            raise ValueError(
                f"n_splits precisa ser pelo menos 2, recebeu n_splits={n_splits}."
            )

    def split(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)

        if self.n_splits > n:
            raise ValueError(
                f"n_splits={self.n_splits} não pode ser maior que o número de amostras ({n})."
            )

        classes = np.unique(y)
        lista = []
        for cls in classes:
            idx = np.where(y == cls)[0]
            if self.shuffle:
                idx = self.rng.permutation(idx)
            lista.append(idx)

        ordem = np.concatenate(lista)

        fold = np.arange(n) % self.n_splits
        for f in range(self.n_splits):
            test_idx = ordem[fold == f]
            train_idx = ordem[fold != f]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None):
        return self.n_splits

    

