"""Transformers de pré-processamento (Fase 1, Semana 2, Sessão 3).

A implementar (API fit / transform / fit_transform, atributos com sufixo _):
- StandardScaler   (mean_, scale_; ddof=0; feature constante -> scale_ = 1)
- MinMaxScaler
- OneHotEncoder    (categories_; handle_unknown='error' | 'ignore')

Paridade: sklearn.preprocessing
"""
import numpy as np

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_==0] = 1.0
        return self

    def transform(self,X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self,X):
        return self.mean_ + (np.asarray(X,dtype=float) * self.scale_)

class MinMaxScaler:
    def __init__(self,feature_range = (0,1)):
        self.feature_range = feature_range

    def fit(self, X):
        self.a_, self.b_ = self.feature_range[0], self.feature_range[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_==0] = 1.0
        self.scale_ = (self.b_ - self.a_ ) / (self.data_range_)
        self.min_ = self.a_ - self.data_min_ * self.scale_
        return self

    def transform(self,X):
        return np.asarray(X,dtype=float) * self.scale_ + self.min_

    def inverse_transform(self, X):
        scale_reverse = (self.data_range_) / (self.b_ - self.a_)
        return self.data_min_ + (np.asarray(X,dtype=float) - self.a_) *scale_reverse

    def fit_transform(self, X):
        return self.fit(X).transform(X)